#![allow(clippy::unwrap_used)]
//! Distributed Query Federation Service
//!
//! Cross-node vector search across multiple Needle instances with query routing,
//! result merging, and automatic shard discovery. Builds on the existing
//! [`Federation`](crate::search::federated::Federation) and
//! [`ShardManager`](crate::persistence::shard::ShardManager) scaffolds.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::distributed_federation::{
//!     FederationService, FederationServiceConfig, NodeEndpoint,
//!     ScatterGatherResult, ShardAssignment,
//! };
//!
//! let config = FederationServiceConfig::default();
//! let mut service = FederationService::new(config);
//!
//! // Register nodes
//! service.add_node(NodeEndpoint::new("node-1", "10.0.0.1:8080")).unwrap();
//! service.add_node(NodeEndpoint::new("node-2", "10.0.0.2:8080")).unwrap();
//!
//! // Assign shards
//! service.assign_shard("shard-0", "node-1").unwrap();
//! service.assign_shard("shard-1", "node-2").unwrap();
//!
//! // Plan a federated query
//! let plan = service.plan_query("my_collection", 10).unwrap();
//! assert!(plan.target_nodes.len() <= 2);
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};
use crate::search::federated::{
    FederationConfig, HealthStatus, MergeStrategy as CoreMergeStrategy,
    RoutingStrategy as CoreRoutingStrategy,
};
use crate::persistence::shard::{ShardConfig, ShardId, ShardState as CoreShardState};

// ── Node Discovery & Membership ──────────────────────────────────────────────

/// Configuration for the federation service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationServiceConfig {
    /// Maximum nodes in the federation.
    pub max_nodes: usize,
    /// Gossip interval in milliseconds.
    pub gossip_interval_ms: u64,
    /// Number of virtual nodes per physical node for consistent hashing.
    pub virtual_nodes_per_node: usize,
    /// Number of replicas per shard.
    pub replication_factor: usize,
    /// Timeout for scatter-gather queries (ms).
    pub query_timeout_ms: u64,
    /// Minimum successful responses for a quorum read.
    pub quorum_size: usize,
    /// Enable automatic shard rebalancing.
    pub auto_rebalance: bool,
    /// Maximum concurrent scatter requests.
    pub max_scatter_concurrency: usize,
    /// Core federation config for underlying scaffold.
    pub core_config: FederationConfig,
    /// Core shard config for underlying scaffold.
    pub shard_config: ShardConfig,
}

impl Default for FederationServiceConfig {
    fn default() -> Self {
        Self {
            max_nodes: 128,
            gossip_interval_ms: 1000,
            virtual_nodes_per_node: 64,
            replication_factor: 2,
            query_timeout_ms: 5000,
            quorum_size: 1,
            auto_rebalance: true,
            max_scatter_concurrency: 16,
            core_config: FederationConfig::default()
                .with_routing(CoreRoutingStrategy::LatencyAware)
                .with_merge(CoreMergeStrategy::DistanceBased)
                .with_partial_results(true),
            shard_config: ShardConfig::new(8)
                .with_virtual_nodes(64)
                .with_replication(2)
                .with_auto_rebalance(true),
        }
    }
}

/// A node endpoint in the federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEndpoint {
    /// Unique node identifier.
    pub node_id: String,
    /// Network address (host:port).
    pub address: String,
    /// Current node status.
    pub status: NodeStatus,
    /// Metadata tags (e.g., region, rack).
    pub tags: HashMap<String, String>,
    /// Last heartbeat timestamp (unix seconds).
    pub last_heartbeat: u64,
    /// Node load (0.0 to 1.0).
    pub load: f64,
    /// Health status bridged from core scaffold.
    pub health: HealthStatus,
}

impl NodeEndpoint {
    pub fn new(node_id: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            address: address.into(),
            status: NodeStatus::Joining,
            tags: HashMap::new(),
            last_heartbeat: now_secs(),
            load: 0.0,
            health: HealthStatus::Healthy,
        }
    }

    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

/// Node status in the federation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Joining,
    Active,
    Suspect,
    Down,
    Leaving,
    Left,
}

/// Gossip message exchanged between nodes, bridging core GossipMessage types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GossipMessage {
    Heartbeat { node_id: String, generation: u64, load: f64 },
    Join { node: NodeEndpoint },
    Leave { node_id: String },
    ShardUpdate { shard_id: String, assigned_to: String, version: u64 },
    StateSync { nodes: Vec<NodeEndpoint>, shards: Vec<ShardAssignment> },
}

/// Shard assignment mapping a shard to a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    pub shard_id: String,
    /// Underlying ShardId from the shard scaffold.
    pub core_shard_id: u32,
    pub primary_node: String,
    pub replica_nodes: Vec<String>,
    pub version: u64,
    pub state: ShardState,
}

/// Shard state, wrapping [`CoreShardState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardState {
    Initializing,
    Active,
    Migrating,
    ReadOnly,
    Offline,
}

impl From<CoreShardState> for ShardState {
    fn from(s: CoreShardState) -> Self {
        match s {
            CoreShardState::Active => ShardState::Active,
            CoreShardState::ReadOnly => ShardState::ReadOnly,
            CoreShardState::Migrating => ShardState::Migrating,
            CoreShardState::Offline => ShardState::Offline,
        }
    }
}

impl From<ShardState> for CoreShardState {
    fn from(s: ShardState) -> Self {
        match s {
            ShardState::Active | ShardState::Initializing => CoreShardState::Active,
            ShardState::Migrating => CoreShardState::Migrating,
            ShardState::ReadOnly => CoreShardState::ReadOnly,
            ShardState::Offline => CoreShardState::Offline,
        }
    }
}

// ── Consistent Hashing ──────────────────────────────────────────────────────

/// Internal consistent hash ring (mirrors `shard::ConsistentHashRing` for node-level routing).
#[derive(Debug, Clone)]
struct NodeHashRing {
    ring: Vec<(u64, String)>,
    virtual_nodes: usize,
}

impl NodeHashRing {
    fn new(virtual_nodes: usize) -> Self {
        Self { ring: Vec::new(), virtual_nodes }
    }

    fn add_node(&mut self, node_id: &str) {
        for i in 0..self.virtual_nodes {
            let hash = fnv1a(&format!("{node_id}:vn{i}"));
            self.ring.push((hash, node_id.to_string()));
        }
        self.ring.sort_by_key(|(h, _)| *h);
    }

    fn remove_node(&mut self, node_id: &str) {
        self.ring.retain(|(_, n)| n != node_id);
    }

    fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() { return None; }
        let hash = fnv1a(key);
        let idx = match self.ring.binary_search_by_key(&hash, |(h, _)| *h) {
            Ok(i) | Err(i) => i % self.ring.len(),
        };
        Some(&self.ring[idx].1)
    }

    fn get_nodes(&self, key: &str, count: usize) -> Vec<String> {
        if self.ring.is_empty() { return Vec::new(); }
        let hash = fnv1a(key);
        let start = match self.ring.binary_search_by_key(&hash, |(h, _)| *h) {
            Ok(i) | Err(i) => i % self.ring.len(),
        };
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for offset in 0..self.ring.len() {
            let idx = (start + offset) % self.ring.len();
            if seen.insert(self.ring[idx].1.clone()) {
                result.push(self.ring[idx].1.clone());
                if result.len() >= count { break; }
            }
        }
        result
    }
}

fn fnv1a(key: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in key.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

// ── Query Routing & Scatter-Gather ──────────────────────────────────────────

/// A federated query plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub target_nodes: Vec<String>,
    pub collection: String,
    pub k_per_node: usize,
    pub k_total: usize,
    pub quorum: bool,
    pub strategy: QueryStrategy,
    /// Timeout from the core config.
    pub timeout: Duration,
}

/// How the query is routed across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryStrategy {
    Broadcast,
    ShardTargeted,
    LoadBalanced,
    LocalityAware,
}

/// Result from a single node in a scatter-gather.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    pub node_id: String,
    pub results: Vec<FederatedHit>,
    pub latency_us: u64,
    pub partial: bool,
}

/// A single hit from federated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedHit {
    pub id: String,
    pub distance: f32,
    pub shard_id: Option<String>,
    pub node_id: String,
    pub metadata: Option<serde_json::Value>,
}

/// Aggregated scatter-gather result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScatterGatherResult {
    pub results: Vec<FederatedHit>,
    pub node_results: Vec<NodeResult>,
    pub total_latency_us: u64,
    pub nodes_responded: usize,
    pub nodes_timed_out: usize,
    pub strategy: QueryStrategy,
}

/// Merge strategy for combining results from multiple nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    BestDistance,
    RankFusion,
    ReliabilityWeighted,
}

// ── Rebalancing & Fault Tolerance ───────────────────────────────────────────

/// Rebalance operation status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalanceOperation {
    pub id: String,
    pub shard_id: String,
    pub from_node: String,
    pub to_node: String,
    pub progress: f64,
    pub status: RebalanceStatus,
    pub started_at: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalanceStatus { Pending, InProgress, Validating, Complete, Failed }

/// Federation service statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FederationServiceStats {
    pub total_queries: u64,
    pub partial_queries: u64,
    pub failed_queries: u64,
    pub rebalance_ops: u64,
    pub avg_latency_us: u64,
    pub gossip_sent: u64,
    pub gossip_received: u64,
}

// ── Federation Service ──────────────────────────────────────────────────────

/// Core federation service managing distributed query execution.
/// Bridges [`FederationConfig`](crate::search::federated::FederationConfig) and
/// [`ShardConfig`](crate::persistence::shard::ShardConfig) for high-level orchestration.
pub struct FederationService {
    config: FederationServiceConfig,
    nodes: HashMap<String, NodeEndpoint>,
    shards: HashMap<String, ShardAssignment>,
    hash_ring: NodeHashRing,
    rebalance_ops: Vec<RebalanceOperation>,
    stats: FederationServiceStats,
    gossip_generation: u64,
    next_shard_id: u32,
}

impl FederationService {
    pub fn new(config: FederationServiceConfig) -> Self {
        let vn = config.virtual_nodes_per_node;
        Self {
            config,
            nodes: HashMap::new(),
            shards: HashMap::new(),
            hash_ring: NodeHashRing::new(vn),
            rebalance_ops: Vec::new(),
            stats: FederationServiceStats::default(),
            gossip_generation: 0,
            next_shard_id: 0,
        }
    }

    /// Create a core `FederationConfig` reflecting current service state.
    pub fn core_federation_config(&self) -> FederationConfig {
        self.config.core_config.clone()
            .with_timeout(Duration::from_millis(self.config.query_timeout_ms))
            .with_min_instances(self.config.quorum_size)
    }

    /// Create a core `ShardConfig` reflecting current service state.
    pub fn core_shard_config(&self) -> ShardConfig {
        ShardConfig::new(self.shards.len() as u32)
            .with_virtual_nodes(self.config.virtual_nodes_per_node as u32)
            .with_replication(self.config.replication_factor as u32)
            .with_auto_rebalance(self.config.auto_rebalance)
    }

    pub fn add_node(&mut self, mut node: NodeEndpoint) -> Result<()> {
        if self.nodes.len() >= self.config.max_nodes {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum nodes ({}) reached", self.config.max_nodes
            )));
        }
        if self.nodes.contains_key(&node.node_id) {
            return Err(NeedleError::Conflict(format!("Node '{}' already exists", node.node_id)));
        }
        node.status = NodeStatus::Active;
        node.last_heartbeat = now_secs();
        node.health = HealthStatus::Healthy;
        self.hash_ring.add_node(&node.node_id);
        self.nodes.insert(node.node_id.clone(), node);
        Ok(())
    }

    pub fn remove_node(&mut self, node_id: &str) -> Result<()> {
        if !self.nodes.contains_key(node_id) {
            return Err(NeedleError::NotFound(format!("Node '{node_id}'")));
        }
        self.hash_ring.remove_node(node_id);
        if self.config.auto_rebalance {
            self.trigger_rebalance_for_node(node_id);
        }
        self.nodes.remove(node_id);
        Ok(())
    }

    pub fn node(&self, node_id: &str) -> Option<&NodeEndpoint> { self.nodes.get(node_id) }

    pub fn active_nodes(&self) -> Vec<&NodeEndpoint> {
        self.nodes.values().filter(|n| n.status == NodeStatus::Active).collect()
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }

    pub fn process_heartbeat(&mut self, node_id: &str, load: f64) -> Result<()> {
        let node = self.nodes.get_mut(node_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Node '{node_id}'")))?;
        node.last_heartbeat = now_secs();
        node.load = load;
        if node.status == NodeStatus::Suspect { node.status = NodeStatus::Active; }
        node.health = if load > 0.9 { HealthStatus::Degraded } else { HealthStatus::Healthy };
        self.stats.gossip_received += 1;
        Ok(())
    }

    pub fn detect_failures(&mut self, timeout_secs: u64) -> Vec<String> {
        let now = now_secs();
        let mut suspected = Vec::new();
        for node in self.nodes.values_mut() {
            if node.status == NodeStatus::Active && now - node.last_heartbeat > timeout_secs {
                node.status = NodeStatus::Suspect;
                node.health = HealthStatus::Unhealthy;
                suspected.push(node.node_id.clone());
            }
        }
        suspected
    }

    pub fn process_gossip(&mut self, message: GossipMessage) -> Result<Option<GossipMessage>> {
        self.stats.gossip_received += 1;
        match message {
            GossipMessage::Heartbeat { node_id, load, .. } => {
                self.process_heartbeat(&node_id, load)?;
                Ok(None)
            }
            GossipMessage::Join { node } => { if let Err(e) = self.add_node(node) { tracing::warn!("gossip: failed to add node: {e}"); } Ok(None) }
            GossipMessage::Leave { node_id } => { if let Err(e) = self.remove_node(&node_id) { tracing::warn!("gossip: failed to remove node: {e}"); } Ok(None) }
            GossipMessage::ShardUpdate { shard_id, assigned_to, version } => {
                if let Some(shard) = self.shards.get_mut(&shard_id) {
                    if version > shard.version {
                        shard.primary_node = assigned_to;
                        shard.version = version;
                    }
                }
                Ok(None)
            }
            GossipMessage::StateSync { nodes, shards } => {
                for node in nodes {
                    if !self.nodes.contains_key(&node.node_id) { if let Err(e) = self.add_node(node) { tracing::warn!("gossip: failed to add node during state sync: {e}"); } }
                }
                for shard in shards {
                    let entry = self.shards.entry(shard.shard_id.clone()).or_insert_with(|| shard.clone());
                    if shard.version > entry.version { *entry = shard; }
                }
                Ok(None)
            }
        }
    }

    pub fn generate_heartbeat(&mut self, self_node_id: &str) -> GossipMessage {
        self.gossip_generation += 1;
        self.stats.gossip_sent += 1;
        let load = self.nodes.get(self_node_id).map_or(0.0, |n| n.load);
        GossipMessage::Heartbeat { node_id: self_node_id.to_string(), generation: self.gossip_generation, load }
    }

    pub fn generate_state_sync(&self) -> GossipMessage {
        GossipMessage::StateSync {
            nodes: self.nodes.values().cloned().collect(),
            shards: self.shards.values().cloned().collect(),
        }
    }

    // ── Shard Management ──────────────────────────────────────────────────

    pub fn assign_shard(&mut self, shard_id: impl Into<String>, node_id: impl Into<String>) -> Result<()> {
        let shard_id = shard_id.into();
        let node_id = node_id.into();
        if !self.nodes.contains_key(&node_id) {
            return Err(NeedleError::NotFound(format!("Node '{node_id}'")));
        }
        let replicas = self.hash_ring.get_nodes(&shard_id, self.config.replication_factor)
            .into_iter().filter(|n| *n != node_id).collect();
        let version = self.shards.get(&shard_id).map_or(1, |s| s.version + 1);
        let core_id = self.next_shard_id;
        self.next_shard_id += 1;
        self.shards.insert(shard_id.clone(), ShardAssignment {
            shard_id, core_shard_id: core_id, primary_node: node_id,
            replica_nodes: replicas, version, state: ShardState::Active,
        });
        Ok(())
    }

    pub fn shard(&self, shard_id: &str) -> Option<&ShardAssignment> { self.shards.get(shard_id) }
    pub fn shards(&self) -> Vec<&ShardAssignment> { self.shards.values().collect() }

    pub fn auto_assign_shard(&mut self, shard_id: impl Into<String>) -> Result<String> {
        let shard_id = shard_id.into();
        let node_id = self.hash_ring.get_node(&shard_id)
            .ok_or_else(|| NeedleError::InvalidOperation("No nodes available".into()))?
            .to_string();
        self.assign_shard(shard_id, node_id.clone())?;
        Ok(node_id)
    }

    /// Map a vector ID to a ShardId using the core shard config's virtual nodes.
    pub fn route_vector_id(&self, vector_id: &str) -> Option<&ShardAssignment> {
        let node = self.hash_ring.get_node(vector_id)?;
        self.shards.values().find(|s| s.primary_node == node)
    }

    // ── Query Planning & Scatter-Gather ─────────────────────────────────

    pub fn plan_query(&self, collection: &str, k: usize) -> Result<QueryPlan> {
        let active: Vec<String> = self.nodes.values()
            .filter(|n| n.status == NodeStatus::Active)
            .map(|n| n.node_id.clone()).collect();
        if active.is_empty() {
            return Err(NeedleError::InvalidOperation("No active nodes available".into()));
        }
        let (target_nodes, strategy) = if self.shards.is_empty() {
            (active, QueryStrategy::Broadcast)
        } else {
            let shard_nodes: Vec<String> = self.shards.values()
                .filter(|s| s.state == ShardState::Active)
                .map(|s| s.primary_node.clone())
                .collect::<std::collections::HashSet<_>>().into_iter().collect();
            if shard_nodes.is_empty() { (active, QueryStrategy::Broadcast) }
            else { (shard_nodes, QueryStrategy::ShardTargeted) }
        };
        let k_per_node = if target_nodes.len() == 1 { k } else { (k * 2).min(k + 50) };
        Ok(QueryPlan {
            target_nodes, collection: collection.to_string(),
            k_per_node, k_total: k,
            quorum: self.config.quorum_size > 1,
            strategy,
            timeout: Duration::from_millis(self.config.query_timeout_ms),
        })
    }

    pub fn merge_results(
        &mut self, node_results: Vec<NodeResult>, k: usize, merge_strategy: MergeStrategy,
    ) -> ScatterGatherResult {
        self.stats.total_queries += 1;
        let nodes_responded = node_results.iter().filter(|r| !r.partial).count();
        let nodes_timed_out = node_results.iter().filter(|r| r.partial).count();
        if nodes_timed_out > 0 { self.stats.partial_queries += 1; }
        let total_latency_us = node_results.iter().map(|r| r.latency_us).max().unwrap_or(0);
        self.stats.avg_latency_us = if self.stats.total_queries == 1 {
            total_latency_us
        } else {
            (self.stats.avg_latency_us + total_latency_us) / 2
        };

        let mut all_hits: Vec<FederatedHit> = node_results.iter().flat_map(|r| r.results.clone()).collect();
        match merge_strategy {
            MergeStrategy::BestDistance => {
                all_hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
            MergeStrategy::RankFusion => {
                let mut rrf_scores: HashMap<String, f64> = HashMap::new();
                for nr in &node_results {
                    for (rank, hit) in nr.results.iter().enumerate() {
                        *rrf_scores.entry(hit.id.clone()).or_default() += 1.0 / (60.0 + rank as f64);
                    }
                }
                all_hits.sort_by(|a, b| {
                    let sa = rrf_scores.get(&a.id).unwrap_or(&0.0);
                    let sb = rrf_scores.get(&b.id).unwrap_or(&0.0);
                    sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            MergeStrategy::ReliabilityWeighted => {
                all_hits.sort_by(|a, b| {
                    let aw: f32 = if node_results.iter().any(|r| r.node_id == a.node_id && !r.partial) { 1.0 } else { 0.5 };
                    let bw: f32 = if node_results.iter().any(|r| r.node_id == b.node_id && !r.partial) { 1.0 } else { 0.5 };
                    (a.distance * aw).partial_cmp(&(b.distance * bw)).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        // Dedup by ID
        let mut seen = std::collections::HashSet::new();
        all_hits.retain(|h| seen.insert(h.id.clone()));
        all_hits.truncate(k);
        ScatterGatherResult {
            results: all_hits, node_results, total_latency_us,
            nodes_responded, nodes_timed_out, strategy: QueryStrategy::Broadcast,
        }
    }

    // ── Rebalancing ─────────────────────────────────────────────────────

    pub fn initiate_rebalance(&mut self, shard_id: &str, to_node: &str) -> Result<String> {
        let shard = self.shards.get(shard_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Shard '{shard_id}'")))?;
        if shard.state == ShardState::Migrating {
            return Err(NeedleError::Conflict(format!("Shard '{shard_id}' is already migrating")));
        }
        if !self.nodes.contains_key(to_node) {
            return Err(NeedleError::NotFound(format!("Node '{to_node}'")));
        }
        let from_node = shard.primary_node.clone();
        let op_id = format!("rebal-{shard_id}-{}", now_secs());
        self.rebalance_ops.push(RebalanceOperation {
            id: op_id.clone(), shard_id: shard_id.to_string(),
            from_node, to_node: to_node.to_string(),
            progress: 0.0, status: RebalanceStatus::Pending, started_at: now_secs(),
        });
        self.stats.rebalance_ops += 1;
        if let Some(shard) = self.shards.get_mut(shard_id) { shard.state = ShardState::Migrating; }
        Ok(op_id)
    }

    pub fn complete_rebalance(&mut self, op_id: &str) -> Result<()> {
        let op = self.rebalance_ops.iter_mut().find(|o| o.id == op_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Rebalance op '{op_id}'")))?;
        op.status = RebalanceStatus::Complete;
        op.progress = 1.0;
        let shard_id = op.shard_id.clone();
        let to_node = op.to_node.clone();
        if let Some(shard) = self.shards.get_mut(&shard_id) {
            shard.primary_node = to_node;
            shard.state = ShardState::Active;
            shard.version += 1;
        }
        Ok(())
    }

    pub fn fail_rebalance(&mut self, op_id: &str) -> Result<()> {
        let op = self.rebalance_ops.iter_mut().find(|o| o.id == op_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Rebalance op '{op_id}'")))?;
        op.status = RebalanceStatus::Failed;
        let shard_id = op.shard_id.clone();
        if let Some(shard) = self.shards.get_mut(&shard_id) { shard.state = ShardState::Active; }
        Ok(())
    }

    pub fn pending_rebalances(&self) -> Vec<&RebalanceOperation> {
        self.rebalance_ops.iter()
            .filter(|o| matches!(o.status, RebalanceStatus::Pending | RebalanceStatus::InProgress | RebalanceStatus::Validating))
            .collect()
    }

    pub fn stats(&self) -> &FederationServiceStats { &self.stats }
    pub fn config(&self) -> &FederationServiceConfig { &self.config }

    fn trigger_rebalance_for_node(&mut self, leaving_node: &str) {
        let shards_to_move: Vec<String> = self.shards.values()
            .filter(|s| s.primary_node == leaving_node)
            .map(|s| s.shard_id.clone()).collect();
        for shard_id in shards_to_move {
            if let Some(new_node) = self.hash_ring.get_node(&shard_id) {
                let new_node = new_node.to_string();
                if new_node != leaving_node { if let Err(e) = self.initiate_rebalance(&shard_id, &new_node) { tracing::warn!("gossip: failed to initiate rebalance for shard {shard_id}: {e}"); } }
            }
        }
    }
}

impl Default for FederationService {
    fn default() -> Self { Self::new(FederationServiceConfig::default()) }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    fn make_service() -> FederationService {
        FederationService::new(FederationServiceConfig { max_nodes: 10, ..Default::default() })
    }

    #[test]
    fn test_add_and_remove_nodes() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        svc.add_node(NodeEndpoint::new("n2", "10.0.0.2:8080")).unwrap();
        assert_eq!(svc.node_count(), 2);
        svc.remove_node("n1").unwrap();
        assert_eq!(svc.node_count(), 1);
    }

    #[test]
    fn test_duplicate_node() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        assert!(svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).is_err());
    }

    #[test]
    fn test_shard_assignment() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        svc.assign_shard("s0", "n1").unwrap();
        let s = svc.shard("s0").unwrap();
        assert_eq!(s.primary_node, "n1");
        assert_eq!(s.state, ShardState::Active);
    }

    #[test]
    fn test_plan_query_broadcast() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        svc.add_node(NodeEndpoint::new("n2", "10.0.0.2:8080")).unwrap();
        let plan = svc.plan_query("coll", 10).unwrap();
        assert_eq!(plan.strategy, QueryStrategy::Broadcast);
        assert_eq!(plan.target_nodes.len(), 2);
    }

    #[test]
    fn test_plan_query_shard_targeted() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        svc.add_node(NodeEndpoint::new("n2", "10.0.0.2:8080")).unwrap();
        svc.assign_shard("s0", "n1").unwrap();
        svc.assign_shard("s1", "n2").unwrap();
        let plan = svc.plan_query("coll", 10).unwrap();
        assert_eq!(plan.strategy, QueryStrategy::ShardTargeted);
    }

    #[test]
    fn test_merge_results_best_distance() {
        let mut svc = make_service();
        let nr = vec![
            NodeResult { node_id: "n1".into(), results: vec![
                FederatedHit { id: "a".into(), distance: 0.5, shard_id: None, node_id: "n1".into(), metadata: None },
            ], latency_us: 100, partial: false },
            NodeResult { node_id: "n2".into(), results: vec![
                FederatedHit { id: "c".into(), distance: 0.3, shard_id: None, node_id: "n2".into(), metadata: None },
            ], latency_us: 150, partial: false },
        ];
        let result = svc.merge_results(nr, 2, MergeStrategy::BestDistance);
        assert_eq!(result.results[0].id, "c");
        assert_eq!(result.nodes_responded, 2);
    }

    #[test]
    fn test_rebalance_lifecycle() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        svc.add_node(NodeEndpoint::new("n2", "10.0.0.2:8080")).unwrap();
        svc.assign_shard("s0", "n1").unwrap();
        let op_id = svc.initiate_rebalance("s0", "n2").unwrap();
        assert_eq!(svc.shard("s0").unwrap().state, ShardState::Migrating);
        svc.complete_rebalance(&op_id).unwrap();
        assert_eq!(svc.shard("s0").unwrap().primary_node, "n2");
        assert_eq!(svc.shard("s0").unwrap().state, ShardState::Active);
    }

    #[test]
    fn test_failure_detection() {
        let mut svc = make_service();
        let mut node = NodeEndpoint::new("n1", "10.0.0.1:8080");
        node.last_heartbeat = now_secs().saturating_sub(120);
        node.status = NodeStatus::Active;
        svc.nodes.insert("n1".to_string(), node);
        let suspected = svc.detect_failures(60);
        assert_eq!(suspected, vec!["n1"]);
        assert_eq!(svc.node("n1").unwrap().health, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_gossip_state_sync() {
        let mut svc1 = make_service();
        svc1.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        svc1.assign_shard("s0", "n1").unwrap();
        let sync_msg = svc1.generate_state_sync();
        let mut svc2 = make_service();
        svc2.process_gossip(sync_msg).unwrap();
        assert!(svc2.node("n1").is_some());
    }

    #[test]
    fn test_core_config_bridge() {
        let svc = make_service();
        let core = svc.core_federation_config();
        assert_eq!(core.query_timeout, Duration::from_millis(5000));
        let shard = svc.core_shard_config();
        assert!(shard.auto_rebalance);
    }

    #[test]
    fn test_shard_state_conversion() {
        assert_eq!(ShardState::from(CoreShardState::Active), ShardState::Active);
        assert_eq!(ShardState::from(CoreShardState::Migrating), ShardState::Migrating);
        assert_eq!(CoreShardState::from(ShardState::Active), CoreShardState::Active);
    }

    #[test]
    fn test_no_active_nodes_error() {
        let svc = make_service();
        assert!(svc.plan_query("coll", 10).is_err());
    }

    #[test]
    fn test_node_tags() {
        let mut svc = make_service();
        let node = NodeEndpoint::new("n1", "10.0.0.1:8080").with_tag("region", "us-east-1");
        svc.add_node(node).unwrap();
        assert_eq!(svc.node("n1").unwrap().tags.get("region").unwrap(), "us-east-1");
    }

    #[test]
    fn test_auto_assign_shard() {
        let mut svc = make_service();
        svc.add_node(NodeEndpoint::new("n1", "10.0.0.1:8080")).unwrap();
        let node_id = svc.auto_assign_shard("my-shard").unwrap();
        assert_eq!(node_id, "n1");
    }

    #[test]
    fn test_dedup_in_merge() {
        let mut svc = make_service();
        let nr = vec![
            NodeResult { node_id: "n1".into(), results: vec![
                FederatedHit { id: "a".into(), distance: 0.5, shard_id: None, node_id: "n1".into(), metadata: None },
            ], latency_us: 100, partial: false },
            NodeResult { node_id: "n2".into(), results: vec![
                FederatedHit { id: "a".into(), distance: 0.6, shard_id: None, node_id: "n2".into(), metadata: None },
            ], latency_us: 50, partial: false },
        ];
        let result = svc.merge_results(nr, 10, MergeStrategy::BestDistance);
        assert_eq!(result.results.len(), 1);
    }
}
