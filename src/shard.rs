//! # Sharding Module
//!
//! Provides sharding primitives for horizontal scaling of vector data.
//! This module implements consistent hashing for shard assignment and
//! shard management for distributed deployments.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      ShardRouter                             │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
//! │  │   Shard 0   │  │   Shard 1   │  │   Shard 2   │  ...    │
//! │  │ [vectors]   │  │ [vectors]   │  │ [vectors]   │         │
//! │  └─────────────┘  └─────────────┘  └─────────────┘         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::shard::{ShardConfig, ShardManager, ShardId};
//!
//! // Create a shard manager with 4 shards
//! let config = ShardConfig::new(4);
//! let manager = ShardManager::new(config);
//!
//! // Route a vector ID to its shard
//! let shard_id = manager.route_id("my_vector_id");
//!
//! // Get shard info
//! let shard = manager.get_shard(shard_id);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use thiserror::Error;

/// Shard errors
#[derive(Error, Debug)]
pub enum ShardError {
    #[error("Shard not found: {0}")]
    ShardNotFound(ShardId),

    #[error("Invalid shard configuration: {0}")]
    InvalidConfig(String),

    #[error("Shard is read-only")]
    ReadOnly,

    #[error("Shard migration in progress")]
    MigrationInProgress,

    #[error("Rebalancing failed: {0}")]
    RebalanceFailed(String),
}

pub type ShardResult<T> = std::result::Result<T, ShardError>;

/// Unique identifier for a shard
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl ShardId {
    /// Create a new shard ID
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the numeric ID
    pub fn id(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard-{}", self.0)
    }
}

/// Shard state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardState {
    /// Shard is active and accepting reads/writes
    Active,
    /// Shard is read-only (during migration)
    ReadOnly,
    /// Shard is being migrated
    Migrating,
    /// Shard is offline
    Offline,
}

/// Shard configuration
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Number of shards
    pub num_shards: u32,
    /// Number of virtual nodes per shard for consistent hashing
    pub virtual_nodes: u32,
    /// Replication factor
    pub replication_factor: u32,
    /// Enable automatic rebalancing
    pub auto_rebalance: bool,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            num_shards: 4,
            virtual_nodes: 150,
            replication_factor: 1,
            auto_rebalance: true,
        }
    }
}

impl ShardConfig {
    /// Create a new config with specified number of shards
    pub fn new(num_shards: u32) -> Self {
        Self {
            num_shards,
            ..Default::default()
        }
    }

    /// Set number of virtual nodes
    pub fn with_virtual_nodes(mut self, nodes: u32) -> Self {
        self.virtual_nodes = nodes;
        self
    }

    /// Set replication factor
    pub fn with_replication(mut self, factor: u32) -> Self {
        self.replication_factor = factor;
        self
    }

    /// Enable/disable auto rebalancing
    pub fn with_auto_rebalance(mut self, enabled: bool) -> Self {
        self.auto_rebalance = enabled;
        self
    }
}

/// Information about a shard
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Shard ID
    pub id: ShardId,
    /// Current state
    pub state: ShardState,
    /// Number of vectors
    pub vector_count: u64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// Node address (for distributed mode)
    pub node_address: Option<String>,
}

impl ShardInfo {
    fn new(id: ShardId) -> Self {
        Self {
            id,
            state: ShardState::Active,
            vector_count: 0,
            memory_bytes: 0,
            node_address: None,
        }
    }
}

/// Point on the consistent hash ring
#[derive(Debug, Clone)]
struct RingPoint {
    hash: u64,
    shard_id: ShardId,
}

/// Consistent hash ring for shard routing
#[derive(Debug)]
pub struct ConsistentHashRing {
    ring: Vec<RingPoint>,
    virtual_nodes: u32,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring
    pub fn new(shard_ids: &[ShardId], virtual_nodes: u32) -> Self {
        let mut ring = Vec::with_capacity(shard_ids.len() * virtual_nodes as usize);

        for &shard_id in shard_ids {
            for vn in 0..virtual_nodes {
                let key = format!("{}:{}", shard_id, vn);
                let hash = Self::hash_key(&key);
                ring.push(RingPoint { hash, shard_id });
            }
        }

        // Sort by hash for binary search
        ring.sort_by_key(|p| p.hash);

        Self { ring, virtual_nodes }
    }

    /// Hash a key using FNV-1a
    fn hash_key(key: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Route a key to a shard
    pub fn route(&self, key: &str) -> Option<ShardId> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = Self::hash_key(key);

        // Binary search for the first point >= hash
        let idx = match self.ring.binary_search_by_key(&hash, |p| p.hash) {
            Ok(i) => i,
            Err(i) => {
                if i >= self.ring.len() {
                    0 // Wrap around to first shard
                } else {
                    i
                }
            }
        };

        Some(self.ring[idx].shard_id)
    }

    /// Get N shards for replication
    pub fn route_with_replicas(&self, key: &str, n: usize) -> Vec<ShardId> {
        if self.ring.is_empty() {
            return vec![];
        }

        let hash = Self::hash_key(key);
        let idx = match self.ring.binary_search_by_key(&hash, |p| p.hash) {
            Ok(i) => i,
            Err(i) => i % self.ring.len(),
        };

        let mut shards = Vec::with_capacity(n);
        let mut seen = std::collections::HashSet::new();
        let mut i = idx;

        while shards.len() < n && seen.len() < self.ring.len() {
            let shard = self.ring[i].shard_id;
            if !seen.contains(&shard) {
                seen.insert(shard);
                shards.push(shard);
            }
            i = (i + 1) % self.ring.len();
        }

        shards
    }

    /// Add a shard to the ring
    pub fn add_shard(&mut self, shard_id: ShardId) {
        for vn in 0..self.virtual_nodes {
            let key = format!("{}:{}", shard_id, vn);
            let hash = Self::hash_key(&key);
            let point = RingPoint { hash, shard_id };

            let idx = match self.ring.binary_search_by_key(&hash, |p| p.hash) {
                Ok(i) => i,
                Err(i) => i,
            };
            self.ring.insert(idx, point);
        }
    }

    /// Remove a shard from the ring
    pub fn remove_shard(&mut self, shard_id: ShardId) {
        self.ring.retain(|p| p.shard_id != shard_id);
    }
}

/// Manages shard allocation and routing
pub struct ShardManager {
    config: ShardConfig,
    shards: HashMap<ShardId, ShardInfo>,
    ring: ConsistentHashRing,
    stats: ShardStats,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new(config: ShardConfig) -> Self {
        let shard_ids: Vec<ShardId> = (0..config.num_shards).map(ShardId::new).collect();
        let mut shards = HashMap::new();

        for &id in &shard_ids {
            shards.insert(id, ShardInfo::new(id));
        }

        let ring = ConsistentHashRing::new(&shard_ids, config.virtual_nodes);

        Self {
            config,
            shards,
            ring,
            stats: ShardStats::default(),
        }
    }

    /// Route a vector ID to its shard
    pub fn route_id(&self, id: &str) -> ShardId {
        self.ring.route(id).unwrap_or(ShardId::new(0))
    }

    /// Route with replicas
    pub fn route_with_replicas(&self, id: &str) -> Vec<ShardId> {
        self.ring
            .route_with_replicas(id, self.config.replication_factor as usize)
    }

    /// Get shard info
    pub fn get_shard(&self, id: ShardId) -> Option<&ShardInfo> {
        self.shards.get(&id)
    }

    /// Get mutable shard info
    pub fn get_shard_mut(&mut self, id: ShardId) -> Option<&mut ShardInfo> {
        self.shards.get_mut(&id)
    }

    /// List all shards
    pub fn list_shards(&self) -> Vec<&ShardInfo> {
        self.shards.values().collect()
    }

    /// Get number of shards
    pub fn num_shards(&self) -> u32 {
        self.config.num_shards
    }

    /// Add a new shard
    pub fn add_shard(&mut self) -> ShardResult<ShardId> {
        let id = ShardId::new(self.config.num_shards);
        self.config.num_shards += 1;
        self.shards.insert(id, ShardInfo::new(id));
        self.ring.add_shard(id);
        Ok(id)
    }

    /// Remove a shard (requires migration first)
    pub fn remove_shard(&mut self, id: ShardId) -> ShardResult<()> {
        if let Some(shard) = self.shards.get(&id) {
            if shard.vector_count > 0 {
                return Err(ShardError::RebalanceFailed(
                    "Shard has vectors, migrate first".to_string(),
                ));
            }
        }

        self.shards.remove(&id);
        self.ring.remove_shard(id);
        Ok(())
    }

    /// Update shard vector count
    pub fn update_vector_count(&mut self, id: ShardId, count: u64) {
        if let Some(shard) = self.shards.get_mut(&id) {
            shard.vector_count = count;
        }
    }

    /// Update shard memory usage
    pub fn update_memory_usage(&mut self, id: ShardId, bytes: u64) {
        if let Some(shard) = self.shards.get_mut(&id) {
            shard.memory_bytes = bytes;
        }
    }

    /// Set shard state
    pub fn set_shard_state(&mut self, id: ShardId, state: ShardState) -> ShardResult<()> {
        if let Some(shard) = self.shards.get_mut(&id) {
            shard.state = state;
            Ok(())
        } else {
            Err(ShardError::ShardNotFound(id))
        }
    }

    /// Get rebalance plan
    pub fn get_rebalance_plan(&self) -> Vec<RebalanceMove> {
        let total_vectors: u64 = self.shards.values().map(|s| s.vector_count).sum();
        let target_per_shard = total_vectors / self.config.num_shards as u64;
        let threshold = target_per_shard / 10; // 10% tolerance

        let mut moves = Vec::new();

        // Find overloaded and underloaded shards
        let mut overloaded: Vec<_> = self
            .shards
            .values()
            .filter(|s| s.vector_count > target_per_shard + threshold)
            .collect();

        let mut underloaded: Vec<_> = self
            .shards
            .values()
            .filter(|s| s.vector_count < target_per_shard.saturating_sub(threshold))
            .collect();

        overloaded.sort_by(|a, b| b.vector_count.cmp(&a.vector_count));
        underloaded.sort_by(|a, b| a.vector_count.cmp(&b.vector_count));

        // Generate moves
        for from in &overloaded {
            for to in &underloaded {
                let excess = from.vector_count.saturating_sub(target_per_shard);
                let deficit = target_per_shard.saturating_sub(to.vector_count);
                let move_count = excess.min(deficit);

                if move_count > 0 {
                    moves.push(RebalanceMove {
                        from_shard: from.id,
                        to_shard: to.id,
                        vector_count: move_count,
                    });
                }
            }
        }

        moves
    }

    /// Get statistics
    pub fn stats(&self) -> &ShardStats {
        &self.stats
    }
}

/// A planned rebalance move
#[derive(Debug, Clone)]
pub struct RebalanceMove {
    /// Source shard
    pub from_shard: ShardId,
    /// Destination shard
    pub to_shard: ShardId,
    /// Number of vectors to move
    pub vector_count: u64,
}

/// Shard statistics
#[derive(Debug, Default)]
pub struct ShardStats {
    /// Total routing operations
    pub routes: AtomicU64,
    /// Total vectors across all shards
    pub total_vectors: AtomicU64,
    /// Rebalance operations
    pub rebalances: AtomicU64,
}

impl ShardStats {
    /// Increment route count
    pub fn record_route(&self) {
        self.routes.fetch_add(1, Ordering::Relaxed);
    }

    /// Get snapshot of stats
    pub fn snapshot(&self) -> ShardStatsSnapshot {
        ShardStatsSnapshot {
            routes: self.routes.load(Ordering::Relaxed),
            total_vectors: self.total_vectors.load(Ordering::Relaxed),
            rebalances: self.rebalances.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of shard stats
#[derive(Debug, Clone)]
pub struct ShardStatsSnapshot {
    pub routes: u64,
    pub total_vectors: u64,
    pub rebalances: u64,
}

/// Shard-aware wrapper for collections
pub struct ShardedCollection<C> {
    manager: Arc<ShardManager>,
    shards: HashMap<ShardId, C>,
}

impl<C> ShardedCollection<C> {
    /// Create a new sharded collection
    pub fn new(manager: Arc<ShardManager>) -> Self {
        Self {
            manager,
            shards: HashMap::new(),
        }
    }

    /// Add a shard collection
    pub fn add_shard(&mut self, id: ShardId, collection: C) {
        self.shards.insert(id, collection);
    }

    /// Get shard for an ID
    pub fn get_shard(&self, vector_id: &str) -> Option<&C> {
        let shard_id = self.manager.route_id(vector_id);
        self.shards.get(&shard_id)
    }

    /// Get mutable shard for an ID
    pub fn get_shard_mut(&mut self, vector_id: &str) -> Option<&mut C> {
        let shard_id = self.manager.route_id(vector_id);
        self.shards.get_mut(&shard_id)
    }

    /// Get all shards
    pub fn all_shards(&self) -> impl Iterator<Item = (&ShardId, &C)> {
        self.shards.iter()
    }

    /// Get all shards mutably
    pub fn all_shards_mut(&mut self) -> impl Iterator<Item = (&ShardId, &mut C)> {
        self.shards.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_id() {
        let id = ShardId::new(5);
        assert_eq!(id.id(), 5);
        assert_eq!(format!("{}", id), "shard-5");
    }

    #[test]
    fn test_consistent_hash_ring() {
        let shards = vec![ShardId::new(0), ShardId::new(1), ShardId::new(2)];
        let ring = ConsistentHashRing::new(&shards, 100);

        // Same key should always route to same shard
        let shard1 = ring.route("test_key");
        let shard2 = ring.route("test_key");
        assert_eq!(shard1, shard2);

        // Different keys should distribute across shards
        let mut distribution = HashMap::new();
        for i in 0..1000 {
            let shard = ring.route(&format!("key_{}", i)).unwrap();
            *distribution.entry(shard).or_insert(0) += 1;
        }

        // All shards should have some keys
        for shard in &shards {
            assert!(distribution.get(shard).unwrap_or(&0) > &100);
        }
    }

    #[test]
    fn test_shard_manager() {
        let config = ShardConfig::new(4);
        let mut manager = ShardManager::new(config);

        assert_eq!(manager.num_shards(), 4);
        assert_eq!(manager.list_shards().len(), 4);

        // Route should be deterministic
        let shard1 = manager.route_id("my_vector");
        let shard2 = manager.route_id("my_vector");
        assert_eq!(shard1, shard2);
    }

    #[test]
    fn test_add_remove_shard() {
        let config = ShardConfig::new(2);
        let mut manager = ShardManager::new(config);

        // Add a shard
        let new_id = manager.add_shard().unwrap();
        assert_eq!(manager.num_shards(), 3);
        assert!(manager.get_shard(new_id).is_some());

        // Remove empty shard
        manager.remove_shard(new_id).unwrap();
        assert!(manager.get_shard(new_id).is_none());
    }

    #[test]
    fn test_route_with_replicas() {
        let config = ShardConfig::new(4).with_replication(2);
        let manager = ShardManager::new(config);

        let replicas = manager.route_with_replicas("test_key");
        assert_eq!(replicas.len(), 2);
        // Replicas should be different shards
        assert_ne!(replicas[0], replicas[1]);
    }

    #[test]
    fn test_rebalance_plan() {
        let config = ShardConfig::new(3);
        let mut manager = ShardManager::new(config);

        // Create imbalance
        manager.update_vector_count(ShardId::new(0), 1000);
        manager.update_vector_count(ShardId::new(1), 100);
        manager.update_vector_count(ShardId::new(2), 200);

        let plan = manager.get_rebalance_plan();
        // Should suggest moving from shard 0 to others
        assert!(!plan.is_empty());
        assert_eq!(plan[0].from_shard, ShardId::new(0));
    }

    #[test]
    fn test_sharded_collection() {
        let config = ShardConfig::new(2);
        let manager = Arc::new(ShardManager::new(config));

        let mut sharded: ShardedCollection<Vec<String>> = ShardedCollection::new(manager.clone());
        sharded.add_shard(ShardId::new(0), vec!["data0".to_string()]);
        sharded.add_shard(ShardId::new(1), vec!["data1".to_string()]);

        // Should be able to get shard for a key
        let shard = sharded.get_shard("test_key");
        assert!(shard.is_some());
    }
}
