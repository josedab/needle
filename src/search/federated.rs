//! # Federated Multi-Instance Search
//!
//! Enables cross-datacenter queries with latency-aware routing and distributed
//! result merging for horizontal scaling across multiple Needle instances.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Federation Coordinator                        │
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │Instance Registry│ │Health Monitor│ │Result Merger │          │
//! │  └──────────────┘  └──────────────┘  └──────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!           ┌──────────────────┼──────────────────┐
//!           │                  │                  │
//!           ▼                  ▼                  ▼
//!     ┌──────────┐      ┌──────────┐      ┌──────────┐
//!     │Instance A│      │Instance B│      │Instance C│
//!     │ (US-West)│      │ (US-East)│      │ (EU)     │
//!     └──────────┘      └──────────┘      └──────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::federated::{Federation, FederationConfig, InstanceConfig};
//!
//! // Create federation
//! let config = FederationConfig::default();
//! let federation = Federation::new(config);
//!
//! // Register instances
//! federation.register_instance(InstanceConfig {
//!     id: "us-west-1".to_string(),
//!     endpoint: "http://needle-west.example.com:8080".to_string(),
//!     region: "us-west".to_string(),
//!     priority: 1,
//! });
//!
//! // Federated search
//! let results = federation.search("my_collection", &query_vector, 10).await?;
//! ```

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Federation errors
#[derive(Error, Debug)]
pub enum FederationError {
    #[error("Instance not found: {0}")]
    InstanceNotFound(String),

    #[error("No healthy instances available")]
    NoHealthyInstances,

    #[error("Query timeout after {0:?}")]
    Timeout(Duration),

    #[error("Partial results: {success} of {total} instances responded")]
    PartialResults { success: usize, total: usize },

    #[error("All instances failed: {0}")]
    AllInstancesFailed(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Instance unhealthy: {0}")]
    UnhealthyInstance(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Quorum not reached: need {required}, got {available}")]
    QuorumNotReached { required: usize, available: usize },
}

pub type FederationResult<T> = std::result::Result<T, FederationError>;

// ============================================================================
// Configuration
// ============================================================================

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Query timeout
    pub query_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Number of retries for failed requests
    pub max_retries: usize,
    /// Minimum instances that must respond
    pub min_instances: usize,
    /// Allow partial results
    pub allow_partial: bool,
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Result merge strategy
    pub merge_strategy: MergeStrategy,
    /// Enable adaptive routing based on latency
    pub adaptive_routing: bool,
    /// Latency weight for routing decisions (0-1)
    pub latency_weight: f32,
    /// Unhealthy threshold (consecutive failures)
    pub unhealthy_threshold: usize,
    /// Recovery threshold (consecutive successes)
    pub recovery_threshold: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            query_timeout: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(30),
            max_retries: 2,
            min_instances: 1,
            allow_partial: true,
            routing_strategy: RoutingStrategy::LatencyAware,
            merge_strategy: MergeStrategy::DistanceBased,
            adaptive_routing: true,
            latency_weight: 0.7,
            unhealthy_threshold: 3,
            recovery_threshold: 2,
        }
    }
}

impl FederationConfig {
    /// Set query timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Set minimum instances
    pub fn with_min_instances(mut self, min: usize) -> Self {
        self.min_instances = min;
        self
    }

    /// Set routing strategy
    pub fn with_routing(mut self, strategy: RoutingStrategy) -> Self {
        self.routing_strategy = strategy;
        self
    }

    /// Set merge strategy
    pub fn with_merge(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Enable/disable partial results
    pub fn with_partial_results(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }
}

/// Routing strategy for selecting instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Query all instances (scatter-gather)
    Broadcast,
    /// Select based on latency
    LatencyAware,
    /// Select based on geographic proximity
    GeographicProximity,
    /// Round-robin selection
    RoundRobin,
    /// Random selection
    Random,
    /// Priority-based (prefer higher priority instances)
    PriorityBased,
    /// Quorum-based (wait for N responses)
    Quorum(usize),
}

/// Strategy for merging results from multiple instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge by distance (default for vector search)
    DistanceBased,
    /// Use reciprocal rank fusion
    ReciprocalRankFusion,
    /// First response wins (for latency-critical queries)
    FirstResponse,
    /// Weighted by instance priority
    PriorityWeighted,
    /// Consensus-based (deduplicate by ID)
    Consensus,
}

// ============================================================================
// Instance Management
// ============================================================================

/// Configuration for a remote instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Unique instance identifier
    pub id: String,
    /// HTTP endpoint
    pub endpoint: String,
    /// Geographic region
    pub region: String,
    /// Priority (higher = preferred)
    pub priority: u32,
    /// Collections available on this instance
    pub collections: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl InstanceConfig {
    /// Create a new instance config
    pub fn new(id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            region: "default".to_string(),
            priority: 1,
            collections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add a collection
    pub fn with_collection(mut self, collection: impl Into<String>) -> Self {
        self.collections.push(collection.into());
        self
    }
}

/// Health status of an instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Instance is healthy and responding
    Healthy,
    /// Instance has degraded performance
    Degraded,
    /// Instance is unhealthy (not responding)
    Unhealthy,
    /// Instance status is unknown
    Unknown,
}

/// Detailed instance information
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// Configuration
    pub config: InstanceConfig,
    /// Current health status
    pub status: HealthStatus,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Total queries processed
    pub total_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Last successful health check
    pub last_healthy: Option<u64>,
    /// Consecutive failures
    pub consecutive_failures: usize,
    /// Consecutive successes
    pub consecutive_successes: usize,
}

impl InstanceInfo {
    fn new(config: InstanceConfig) -> Self {
        Self {
            config,
            status: HealthStatus::Unknown,
            avg_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            total_queries: 0,
            failed_queries: 0,
            last_healthy: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            1.0
        } else {
            1.0 - (self.failed_queries as f64 / self.total_queries as f64)
        }
    }

    /// Calculate routing score (lower is better)
    /// Compute routing score (lower is better for routing priority).
    /// Returns a value >= 0.01 to ensure positive scores.
    pub fn routing_score(&self, latency_weight: f32) -> f64 {
        let latency_score = self.avg_latency_ms / 1000.0; // Normalize to seconds
        let health_score = match self.status {
            HealthStatus::Healthy => 0.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Unhealthy => 10.0,
            HealthStatus::Unknown => 1.0,
        };
        let failure_score = self.consecutive_failures as f64 * 0.1;
        // Priority multiplier: higher priority = lower score (better)
        let priority_multiplier = 1.0 / (self.config.priority as f64 + 1.0);

        let base_score = latency_weight as f64 * latency_score
            + (1.0 - latency_weight as f64) * health_score
            + failure_score
            + 0.1; // Base offset to ensure positive scores

        // Apply priority as a multiplier (higher priority reduces score)
        (base_score * priority_multiplier).max(0.01)
    }
}

/// Instance registry for tracking remote instances
pub struct InstanceRegistry {
    instances: RwLock<HashMap<String, InstanceInfo>>,
    regions: RwLock<HashMap<String, Vec<String>>>,
    round_robin_counter: AtomicUsize,
}

impl Default for InstanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl InstanceRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            instances: RwLock::new(HashMap::new()),
            regions: RwLock::new(HashMap::new()),
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Register an instance
    pub fn register(&self, config: InstanceConfig) {
        let region = config.region.clone();
        let id = config.id.clone();

        let mut instances = self.instances.write();
        instances.insert(id.clone(), InstanceInfo::new(config));

        let mut regions = self.regions.write();
        regions.entry(region).or_default().push(id);
    }

    /// Unregister an instance
    pub fn unregister(&self, id: &str) -> Option<InstanceInfo> {
        let mut instances = self.instances.write();
        let info = instances.remove(id)?;

        let mut regions = self.regions.write();
        if let Some(region_instances) = regions.get_mut(&info.config.region) {
            region_instances.retain(|i| i != id);
        }

        Some(info)
    }

    /// Get instance info
    pub fn get(&self, id: &str) -> Option<InstanceInfo> {
        self.instances.read().get(id).cloned()
    }

    /// Get all instances
    pub fn list(&self) -> Vec<InstanceInfo> {
        self.instances.read().values().cloned().collect()
    }

    /// Get healthy instances
    pub fn healthy_instances(&self) -> Vec<InstanceInfo> {
        self.instances
            .read()
            .values()
            .filter(|i| i.status == HealthStatus::Healthy || i.status == HealthStatus::Degraded)
            .cloned()
            .collect()
    }

    /// Get instances by region
    pub fn instances_by_region(&self, region: &str) -> Vec<InstanceInfo> {
        let regions = self.regions.read();
        let instances = self.instances.read();

        regions
            .get(region)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| instances.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get instances with a specific collection
    pub fn instances_with_collection(&self, collection: &str) -> Vec<InstanceInfo> {
        self.instances
            .read()
            .values()
            .filter(|i| {
                i.config.collections.is_empty()
                    || i.config.collections.contains(&collection.to_string())
            })
            .cloned()
            .collect()
    }

    /// Update instance health
    pub fn update_health(&self, id: &str, status: HealthStatus) {
        let mut instances = self.instances.write();
        if let Some(info) = instances.get_mut(id) {
            info.status = status;
            match status {
                HealthStatus::Healthy => {
                    info.consecutive_successes += 1;
                    info.consecutive_failures = 0;
                    info.last_healthy = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
                }
                HealthStatus::Unhealthy => {
                    info.consecutive_failures += 1;
                    info.consecutive_successes = 0;
                }
                _ => {}
            }
        }
    }

    /// Record query result
    pub fn record_query(&self, id: &str, latency_ms: f64, success: bool) {
        let mut instances = self.instances.write();
        if let Some(info) = instances.get_mut(id) {
            info.total_queries += 1;

            if success {
                // Update average latency with exponential moving average
                let alpha = 0.1;
                info.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * info.avg_latency_ms;

                // Update P99 (simplified - just track max recent)
                if latency_ms > info.p99_latency_ms {
                    info.p99_latency_ms = latency_ms;
                } else {
                    info.p99_latency_ms = 0.99 * info.p99_latency_ms + 0.01 * latency_ms;
                }

                info.consecutive_successes += 1;
                info.consecutive_failures = 0;
            } else {
                info.failed_queries += 1;
                info.consecutive_failures += 1;
                info.consecutive_successes = 0;
            }
        }
    }

    /// Select instance using round-robin
    pub fn select_round_robin(&self) -> Option<String> {
        let instances = self.instances.read();
        let healthy: Vec<_> = instances
            .iter()
            .filter(|(_, i)| i.status == HealthStatus::Healthy)
            .map(|(id, _)| id.clone())
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % healthy.len();
        Some(healthy[idx].clone())
    }

    /// Select instance by lowest latency
    pub fn select_by_latency(&self, latency_weight: f32) -> Option<String> {
        let instances = self.instances.read();
        instances
            .iter()
            .filter(|(_, i)| {
                i.status == HealthStatus::Healthy || i.status == HealthStatus::Degraded
            })
            .min_by(|(_, a), (_, b)| {
                a.routing_score(latency_weight)
                    .partial_cmp(&b.routing_score(latency_weight))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.clone())
    }

    /// Get instance count
    pub fn len(&self) -> usize {
        self.instances.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.instances.read().is_empty()
    }
}

// ============================================================================
// Search Results
// ============================================================================

/// Search result from a federated query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchResult {
    /// Vector ID
    pub id: String,
    /// Distance to query
    pub distance: f32,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
    /// Source instance
    pub source_instance: String,
    /// Collection name
    pub collection: String,
}

/// Aggregated results from federated search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchResponse {
    /// Merged results
    pub results: Vec<FederatedSearchResult>,
    /// Total results before limiting
    pub total_found: usize,
    /// Instances that responded
    pub instances_responded: Vec<String>,
    /// Instances that failed
    pub instances_failed: Vec<String>,
    /// Query execution time
    pub execution_time_ms: f64,
    /// Per-instance latencies
    pub instance_latencies: HashMap<String, f64>,
    /// Whether results are partial
    pub is_partial: bool,
}

// ============================================================================
// Result Merger
// ============================================================================

/// Merges results from multiple instances
pub struct ResultMerger {
    strategy: MergeStrategy,
}

impl ResultMerger {
    /// Create a new merger
    pub fn new(strategy: MergeStrategy) -> Self {
        Self { strategy }
    }

    /// Merge results from multiple instances
    pub fn merge(
        &self,
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        match self.strategy {
            MergeStrategy::DistanceBased => self.merge_by_distance(instance_results, k),
            MergeStrategy::ReciprocalRankFusion => self.merge_rrf(instance_results, k),
            MergeStrategy::FirstResponse => self.merge_first(instance_results, k),
            MergeStrategy::PriorityWeighted => self.merge_by_distance(instance_results, k), // Simplified
            MergeStrategy::Consensus => self.merge_consensus(instance_results, k),
        }
    }

    fn merge_by_distance(
        &self,
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        let mut all_results: Vec<FederatedSearchResult> = instance_results
            .into_iter()
            .flat_map(|(_, results)| results)
            .collect();

        // Sort by distance
        all_results.sort_by_key(|r| OrderedFloat(r.distance));

        // Deduplicate by ID (keep lowest distance)
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|r| seen.insert(r.id.clone()));

        // Take top k
        all_results.truncate(k);
        all_results
    }

    fn merge_rrf(
        &self,
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        let rrf_k = 60.0; // RRF constant

        // Calculate RRF scores
        let mut scores: HashMap<String, (f64, FederatedSearchResult)> = HashMap::new();

        for (_, results) in instance_results {
            for (rank, result) in results.into_iter().enumerate() {
                let rrf_score = 1.0 / (rrf_k + rank as f64 + 1.0);

                scores
                    .entry(result.id.clone())
                    .and_modify(|(score, _)| *score += rrf_score)
                    .or_insert((rrf_score, result));
            }
        }

        // Sort by RRF score (descending)
        let mut sorted: Vec<_> = scores.into_values().collect();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        sorted.into_iter().take(k).map(|(_, r)| r).collect()
    }

    fn merge_first(
        &self,
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        // Take results from first non-empty response
        instance_results
            .into_iter()
            .find(|(_, results)| !results.is_empty())
            .map(|(_, mut results)| {
                results.truncate(k);
                results
            })
            .unwrap_or_default()
    }

    fn merge_consensus(
        &self,
        instance_results: Vec<(String, Vec<FederatedSearchResult>)>,
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        // Count occurrences and average distances
        let mut consensus: HashMap<String, (usize, f32, FederatedSearchResult)> = HashMap::new();

        for (_, results) in instance_results {
            for result in results {
                consensus
                    .entry(result.id.clone())
                    .and_modify(|(count, total_dist, _)| {
                        *count += 1;
                        *total_dist += result.distance;
                    })
                    .or_insert((1, result.distance, result));
            }
        }

        // Sort by count (desc) then distance (asc)
        let mut sorted: Vec<_> = consensus.into_values().collect();
        sorted.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| OrderedFloat(a.1 / a.0 as f32).cmp(&OrderedFloat(b.1 / b.0 as f32)))
        });

        sorted.into_iter().take(k).map(|(_, _, r)| r).collect()
    }
}

// ============================================================================
// Health Monitor
// ============================================================================

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub instance_id: String,
    pub status: HealthStatus,
    pub latency_ms: f64,
    pub error: Option<String>,
    pub timestamp: u64,
}

/// Health monitor for tracking instance health
pub struct HealthMonitor {
    registry: Arc<InstanceRegistry>,
    config: FederationConfig,
    check_results: RwLock<HashMap<String, Vec<HealthCheckResult>>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(registry: Arc<InstanceRegistry>, config: FederationConfig) -> Self {
        Self {
            registry,
            config,
            check_results: RwLock::new(HashMap::new()),
        }
    }

    /// Record a health check result
    pub fn record_check(&self, result: HealthCheckResult) {
        let instance_id = result.instance_id.clone();

        // Update registry
        self.registry.update_health(&instance_id, result.status);

        // Store result history
        let mut results = self.check_results.write();
        let history = results.entry(instance_id).or_default();
        history.push(result);

        // Keep only recent history (last 100 checks)
        if history.len() > 100 {
            history.remove(0);
        }
    }

    /// Perform a simulated health check (in real impl, this would make HTTP calls)
    pub fn check_instance(&self, instance_id: &str) -> HealthCheckResult {
        let start = Instant::now();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Simulate health check - in real implementation this would ping the instance
        let info = self.registry.get(instance_id);

        match info {
            Some(info) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;

                // Determine health based on consecutive failures
                let status = if info.consecutive_failures >= self.config.unhealthy_threshold {
                    HealthStatus::Unhealthy
                } else if info.consecutive_failures > 0 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                HealthCheckResult {
                    instance_id: instance_id.to_string(),
                    status,
                    latency_ms: latency,
                    error: None,
                    timestamp,
                }
            }
            None => HealthCheckResult {
                instance_id: instance_id.to_string(),
                status: HealthStatus::Unknown,
                latency_ms: 0.0,
                error: Some("Instance not found".to_string()),
                timestamp,
            },
        }
    }

    /// Get health check history for an instance
    pub fn get_history(&self, instance_id: &str) -> Vec<HealthCheckResult> {
        self.check_results
            .read()
            .get(instance_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get overall federation health status
    pub fn federation_health(&self) -> FederationHealth {
        let instances = self.registry.list();
        let total = instances.len();
        let healthy = instances
            .iter()
            .filter(|i| i.status == HealthStatus::Healthy)
            .count();
        let degraded = instances
            .iter()
            .filter(|i| i.status == HealthStatus::Degraded)
            .count();
        let unhealthy = instances
            .iter()
            .filter(|i| i.status == HealthStatus::Unhealthy)
            .count();

        let status = if healthy == total {
            HealthStatus::Healthy
        } else if healthy + degraded >= self.config.min_instances {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        FederationHealth {
            status,
            total_instances: total,
            healthy_instances: healthy,
            degraded_instances: degraded,
            unhealthy_instances: unhealthy,
            avg_latency_ms: instances.iter().map(|i| i.avg_latency_ms).sum::<f64>()
                / total.max(1) as f64,
        }
    }
}

/// Overall federation health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationHealth {
    pub status: HealthStatus,
    pub total_instances: usize,
    pub healthy_instances: usize,
    pub degraded_instances: usize,
    pub unhealthy_instances: usize,
    pub avg_latency_ms: f64,
}

// ============================================================================
// Federation Coordinator
// ============================================================================

/// Main federation coordinator
pub struct Federation {
    config: FederationConfig,
    registry: Arc<InstanceRegistry>,
    merger: ResultMerger,
    monitor: HealthMonitor,
    stats: FederationStats,
}

impl Federation {
    /// Create a new federation
    pub fn new(config: FederationConfig) -> Self {
        let registry = Arc::new(InstanceRegistry::new());
        let merger = ResultMerger::new(config.merge_strategy);
        let monitor = HealthMonitor::new(registry.clone(), config.clone());

        Self {
            config,
            registry,
            merger,
            monitor,
            stats: FederationStats::default(),
        }
    }

    /// Register an instance
    pub fn register_instance(&self, config: InstanceConfig) {
        self.registry.register(config);
    }

    /// Unregister an instance
    pub fn unregister_instance(&self, id: &str) -> Option<InstanceInfo> {
        self.registry.unregister(id)
    }

    /// Get instance info
    pub fn get_instance(&self, id: &str) -> Option<InstanceInfo> {
        self.registry.get(id)
    }

    /// List all instances
    pub fn list_instances(&self) -> Vec<InstanceInfo> {
        self.registry.list()
    }

    /// Select instances for a query based on routing strategy
    pub fn select_instances(&self, collection: &str) -> FederationResult<Vec<String>> {
        let candidates = self.registry.instances_with_collection(collection);

        if candidates.is_empty() {
            return Err(FederationError::NoHealthyInstances);
        }

        let healthy: Vec<_> = candidates
            .into_iter()
            .filter(|i| i.status == HealthStatus::Healthy || i.status == HealthStatus::Degraded)
            .collect();

        if healthy.is_empty() {
            return Err(FederationError::NoHealthyInstances);
        }

        match self.config.routing_strategy {
            RoutingStrategy::Broadcast => Ok(healthy.iter().map(|i| i.config.id.clone()).collect()),

            RoutingStrategy::LatencyAware => {
                let mut sorted = healthy;
                sorted.sort_by(|a, b| {
                    a.routing_score(self.config.latency_weight)
                        .partial_cmp(&b.routing_score(self.config.latency_weight))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(sorted.iter().map(|i| i.config.id.clone()).collect())
            }

            RoutingStrategy::GeographicProximity => {
                // In real implementation, would sort by geographic distance
                Ok(healthy.iter().map(|i| i.config.id.clone()).collect())
            }

            RoutingStrategy::RoundRobin => self
                .registry
                .select_round_robin()
                .map(|id| vec![id])
                .ok_or(FederationError::NoHealthyInstances),

            RoutingStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let mut ids: Vec<_> = healthy.iter().map(|i| i.config.id.clone()).collect();
                ids.shuffle(&mut rng);
                Ok(ids)
            }

            RoutingStrategy::PriorityBased => {
                let mut sorted = healthy;
                sorted.sort_by(|a, b| b.config.priority.cmp(&a.config.priority));
                Ok(sorted.iter().map(|i| i.config.id.clone()).collect())
            }

            RoutingStrategy::Quorum(n) => {
                if healthy.len() < n {
                    return Err(FederationError::QuorumNotReached {
                        required: n,
                        available: healthy.len(),
                    });
                }
                Ok(healthy
                    .iter()
                    .take(n)
                    .map(|i| i.config.id.clone())
                    .collect())
            }
        }
    }

    /// Execute a federated search (simulated - returns mock results)
    pub fn search(
        &self,
        collection: &str,
        _query: &[f32],
        k: usize,
    ) -> FederationResult<FederatedSearchResponse> {
        let start = Instant::now();

        // Select instances
        let instances = self.select_instances(collection)?;

        if instances.is_empty() {
            return Err(FederationError::NoHealthyInstances);
        }

        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        // Simulate parallel queries to instances
        let mut instance_results: Vec<(String, Vec<FederatedSearchResult>)> = Vec::new();
        let mut instance_latencies: HashMap<String, f64> = HashMap::new();
        let mut instances_failed: Vec<String> = Vec::new();

        for instance_id in &instances {
            let query_start = Instant::now();

            // Simulate query - in real implementation, this would make HTTP calls
            let info = self.registry.get(instance_id);

            if let Some(info) = info {
                if info.status == HealthStatus::Unhealthy {
                    instances_failed.push(instance_id.clone());
                    self.registry.record_query(instance_id, 0.0, false);
                    continue;
                }

                // Generate mock results
                let results: Vec<FederatedSearchResult> = (0..k.min(5))
                    .map(|i| FederatedSearchResult {
                        id: format!("{}_{}", instance_id, i),
                        distance: 0.1 + (i as f32 * 0.1),
                        metadata: None,
                        source_instance: instance_id.clone(),
                        collection: collection.to_string(),
                    })
                    .collect();

                let latency = query_start.elapsed().as_secs_f64() * 1000.0;
                instance_latencies.insert(instance_id.clone(), latency);
                self.registry.record_query(instance_id, latency, true);

                instance_results.push((instance_id.clone(), results));
            } else {
                instances_failed.push(instance_id.clone());
            }
        }

        // Check minimum instances
        if instance_results.len() < self.config.min_instances && !self.config.allow_partial {
            self.stats.failed_queries.fetch_add(1, Ordering::Relaxed);
            return Err(FederationError::PartialResults {
                success: instance_results.len(),
                total: instances.len(),
            });
        }

        // Merge results
        let total_found: usize = instance_results.iter().map(|(_, r)| r.len()).sum();
        let instances_responded: Vec<_> =
            instance_results.iter().map(|(id, _)| id.clone()).collect();
        let merged = self.merger.merge(instance_results, k);

        let execution_time = start.elapsed();

        let is_partial = !instances_failed.is_empty();
        if is_partial {
            self.stats.partial_results.fetch_add(1, Ordering::Relaxed);
        }

        Ok(FederatedSearchResponse {
            results: merged,
            total_found,
            instances_responded,
            instances_failed,
            execution_time_ms: execution_time.as_secs_f64() * 1000.0,
            instance_latencies,
            is_partial,
        })
    }

    /// Get federation health
    pub fn health(&self) -> FederationHealth {
        self.monitor.federation_health()
    }

    /// Get statistics
    pub fn stats(&self) -> FederationStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get configuration
    pub fn config(&self) -> &FederationConfig {
        &self.config
    }

    /// Perform health check on an instance
    pub fn check_instance_health(&self, id: &str) -> HealthCheckResult {
        let result = self.monitor.check_instance(id);
        self.monitor.record_check(result.clone());
        result
    }
}

/// Federation statistics
#[derive(Debug, Default)]
pub struct FederationStats {
    total_queries: AtomicU64,
    failed_queries: AtomicU64,
    partial_results: AtomicU64,
    timeouts: AtomicU64,
}

impl FederationStats {
    fn snapshot(&self) -> FederationStatsSnapshot {
        FederationStatsSnapshot {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            failed_queries: self.failed_queries.load(Ordering::Relaxed),
            partial_results: self.partial_results.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of federation stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStatsSnapshot {
    pub total_queries: u64,
    pub failed_queries: u64,
    pub partial_results: u64,
    pub timeouts: u64,
}

// ---------------------------------------------------------------------------
// Auto-Discovery: instances register/deregister via heartbeat
// ---------------------------------------------------------------------------

/// Configuration for instance auto-discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// How often instances should send heartbeats.
    pub heartbeat_interval: Duration,
    /// After how many missed heartbeats an instance is marked unhealthy.
    pub missed_heartbeat_threshold: u32,
    /// Whether to automatically remove stale instances.
    pub auto_remove_stale: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval: Duration::from_secs(10),
            missed_heartbeat_threshold: 3,
            auto_remove_stale: true,
        }
    }
}

/// Tracks heartbeat state per instance.
#[derive(Debug, Clone)]
struct HeartbeatState {
    last_seen: Instant,
    missed_count: u32,
    metadata: HashMap<String, String>,
}

/// Service that manages instance discovery through heartbeats.
pub struct DiscoveryService {
    registry: Arc<InstanceRegistry>,
    config: DiscoveryConfig,
    heartbeats: RwLock<HashMap<String, HeartbeatState>>,
}

impl DiscoveryService {
    /// Create a new discovery service.
    pub fn new(registry: Arc<InstanceRegistry>, config: DiscoveryConfig) -> Self {
        Self {
            registry,
            config,
            heartbeats: RwLock::new(HashMap::new()),
        }
    }

    /// Record a heartbeat from an instance. Auto-registers unknown instances.
    pub fn heartbeat(&self, instance_id: &str, endpoint: &str, metadata: HashMap<String, String>) {
        // Register if unknown
        if self.registry.get(instance_id).is_none() {
            let mut inst = InstanceConfig::new(instance_id, endpoint);
            if let Some(region) = metadata.get("region") {
                inst = inst.with_region(region);
            }
            self.registry.register(inst);
        }
        self.registry
            .update_health(instance_id, HealthStatus::Healthy);

        let mut hb = self.heartbeats.write();
        hb.insert(
            instance_id.to_string(),
            HeartbeatState {
                last_seen: Instant::now(),
                missed_count: 0,
                metadata,
            },
        );
    }

    /// Check all instances for missed heartbeats.
    pub fn check_heartbeats(&self) -> Vec<String> {
        let threshold = self.config.heartbeat_interval * self.config.missed_heartbeat_threshold;
        let mut stale = Vec::new();
        let mut hb = self.heartbeats.write();

        for (id, state) in hb.iter_mut() {
            if state.last_seen.elapsed() > threshold {
                state.missed_count += 1;
                self.registry.update_health(id, HealthStatus::Unhealthy);
                stale.push(id.clone());
            }
        }

        if self.config.auto_remove_stale {
            for id in &stale {
                if let Some(state) = hb.get(id) {
                    if state.missed_count > self.config.missed_heartbeat_threshold * 2 {
                        self.registry.unregister(id);
                        // Will be cleaned up from hb map below
                    }
                }
            }
            hb.retain(|id, state| {
                state.missed_count <= self.config.missed_heartbeat_threshold * 2
                    || self.registry.get(id).is_some()
            });
        }

        stale
    }

    /// Number of tracked instances.
    pub fn tracked_count(&self) -> usize {
        self.heartbeats.read().len()
    }
}

// ---------------------------------------------------------------------------
// Cross-Instance Deduplication
// ---------------------------------------------------------------------------

/// Deduplicates search results that appear from multiple instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DedupStrategy {
    /// Keep the result with the smallest distance.
    BestDistance,
    /// Keep the result from the first instance that returned it.
    FirstSeen,
    /// Average the distances across instances.
    AverageDistance,
}

impl Default for DedupStrategy {
    fn default() -> Self {
        Self::BestDistance
    }
}

/// Deduplicates and merges results from multiple instances.
pub struct CrossInstanceDedup {
    strategy: DedupStrategy,
}

impl CrossInstanceDedup {
    pub fn new(strategy: DedupStrategy) -> Self {
        Self { strategy }
    }

    /// Deduplicate results. Each inner Vec is from one instance.
    pub fn dedup(
        &self,
        results: &[Vec<FederatedSearchResult>],
        k: usize,
    ) -> Vec<FederatedSearchResult> {
        let mut seen: HashMap<String, (FederatedSearchResult, usize)> = HashMap::new();

        for instance_results in results {
            for r in instance_results {
                match seen.get_mut(&r.id) {
                    Some((existing, count)) => match self.strategy {
                        DedupStrategy::BestDistance => {
                            if r.distance < existing.distance {
                                *existing = r.clone();
                            }
                        }
                        DedupStrategy::FirstSeen => {
                            // keep existing
                        }
                        DedupStrategy::AverageDistance => {
                            *count += 1;
                            existing.distance = (existing.distance * (*count - 1) as f32
                                + r.distance)
                                / *count as f32;
                        }
                    },
                    None => {
                        seen.insert(r.id.clone(), (r.clone(), 1));
                    }
                }
            }
        }

        let mut merged: Vec<FederatedSearchResult> = seen.into_values().map(|(r, _)| r).collect();
        merged.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(k);
        merged
    }
}

// ---------------------------------------------------------------------------
// Query Consistency Controls
// ---------------------------------------------------------------------------

/// Consistency level for federated queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Return results from any single healthy instance.
    One,
    /// Require results from a majority of instances.
    Quorum,
    /// Require results from all instances.
    All,
    /// Best-effort: return whatever is available before timeout.
    BestEffort,
}

impl Default for ConsistencyLevel {
    fn default() -> Self {
        Self::BestEffort
    }
}

/// A federated query plan describing which instances to query and how.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub target_instances: Vec<String>,
    pub consistency: ConsistencyLevel,
    pub dedup: DedupStrategy,
    pub k: usize,
    pub timeout: Duration,
}

/// Plans and validates federated queries before execution.
pub struct QueryPlanner {
    registry: Arc<InstanceRegistry>,
}

impl QueryPlanner {
    pub fn new(registry: Arc<InstanceRegistry>) -> Self {
        Self { registry }
    }

    /// Create a query plan for the given collection and consistency level.
    pub fn plan(
        &self,
        collection: &str,
        k: usize,
        consistency: ConsistencyLevel,
        timeout: Duration,
    ) -> Result<QueryPlan, FederationError> {
        let candidates = self.registry.instances_with_collection(collection);
        let healthy: Vec<InstanceInfo> = candidates
            .into_iter()
            .filter(|i| i.status == HealthStatus::Healthy)
            .collect();

        if healthy.is_empty() {
            // Fall back to all healthy instances
            let all_healthy = self.registry.healthy_instances();
            if all_healthy.is_empty() {
                return Err(FederationError::NoHealthyInstances);
            }
            return Ok(QueryPlan {
                target_instances: all_healthy.iter().map(|i| i.config.id.clone()).collect(),
                consistency,
                dedup: DedupStrategy::BestDistance,
                k,
                timeout,
            });
        }

        let required = match consistency {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => (healthy.len() / 2) + 1,
            ConsistencyLevel::All => healthy.len(),
            ConsistencyLevel::BestEffort => healthy.len(),
        };

        if healthy.len() < required {
            return Err(FederationError::QuorumNotReached {
                required,
                available: healthy.len(),
            });
        }

        let targets: Vec<String> = healthy
            .iter()
            .take(required)
            .map(|i| i.config.id.clone())
            .collect();

        Ok(QueryPlan {
            target_instances: targets,
            consistency,
            dedup: DedupStrategy::BestDistance,
            k,
            timeout,
        })
    }

    /// Validate whether a query plan is still executable.
    pub fn validate(&self, plan: &QueryPlan) -> bool {
        for id in &plan.target_instances {
            match self.registry.get(id) {
                Some(info) if info.status == HealthStatus::Healthy => {}
                _ => return false,
            }
        }
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_federation() -> Federation {
        let config = FederationConfig::default();
        let federation = Federation::new(config);

        // Register test instances
        federation.register_instance(
            InstanceConfig::new("us-west-1", "http://localhost:8081")
                .with_region("us-west")
                .with_priority(2),
        );

        federation.register_instance(
            InstanceConfig::new("us-east-1", "http://localhost:8082")
                .with_region("us-east")
                .with_priority(1),
        );

        federation.register_instance(
            InstanceConfig::new("eu-west-1", "http://localhost:8083")
                .with_region("eu-west")
                .with_priority(1),
        );

        // Mark all as healthy
        federation
            .registry
            .update_health("us-west-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("us-east-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("eu-west-1", HealthStatus::Healthy);

        federation
    }

    #[test]
    fn test_instance_registry() {
        let registry = InstanceRegistry::new();

        registry.register(InstanceConfig::new("test-1", "http://localhost:8080"));

        assert_eq!(registry.len(), 1);
        assert!(registry.get("test-1").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_instance_registration() {
        let federation = create_test_federation();

        assert_eq!(federation.list_instances().len(), 3);
        assert!(federation.get_instance("us-west-1").is_some());
    }

    #[test]
    fn test_instance_health_tracking() {
        let federation = create_test_federation();

        // Record some queries
        federation.registry.record_query("us-west-1", 10.0, true);
        federation.registry.record_query("us-west-1", 15.0, true);
        federation.registry.record_query("us-west-1", 20.0, false);

        let info = federation.get_instance("us-west-1").unwrap();
        assert_eq!(info.total_queries, 3);
        assert_eq!(info.failed_queries, 1);
        assert!(info.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_select_instances_broadcast() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Broadcast);
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("inst-1", "http://localhost:8081"));
        federation.register_instance(InstanceConfig::new("inst-2", "http://localhost:8082"));
        federation
            .registry
            .update_health("inst-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-2", HealthStatus::Healthy);

        let instances = federation.select_instances("test_collection").unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_select_instances_latency_aware() {
        let federation = create_test_federation();

        // Simulate different latencies
        federation.registry.record_query("us-west-1", 100.0, true);
        federation.registry.record_query("us-east-1", 50.0, true);
        federation.registry.record_query("eu-west-1", 200.0, true);

        let instances = federation.select_instances("test_collection").unwrap();

        // Should prefer us-east-1 (lowest latency) but also consider priority
        assert!(!instances.is_empty());
    }

    #[test]
    fn test_select_instances_no_healthy() {
        let config = FederationConfig::default();
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("unhealthy", "http://localhost:8080"));
        federation
            .registry
            .update_health("unhealthy", HealthStatus::Unhealthy);

        let result = federation.select_instances("test_collection");
        assert!(matches!(result, Err(FederationError::NoHealthyInstances)));
    }

    #[test]
    fn test_federated_search() {
        let federation = create_test_federation();

        let query = vec![0.1; 128];
        let result = federation.search("test_collection", &query, 10).unwrap();

        assert!(!result.results.is_empty());
        assert!(!result.instances_responded.is_empty());
        assert!(result.execution_time_ms > 0.0);
    }

    #[test]
    fn test_result_merger_distance_based() {
        let merger = ResultMerger::new(MergeStrategy::DistanceBased);

        let results = vec![
            (
                "inst-1".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "a".to_string(),
                        distance: 0.5,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "b".to_string(),
                        distance: 0.8,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
            (
                "inst-2".to_string(),
                vec![FederatedSearchResult {
                    id: "c".to_string(),
                    distance: 0.3,
                    metadata: None,
                    source_instance: "inst-2".to_string(),
                    collection: "test".to_string(),
                }],
            ),
        ];

        let merged = merger.merge(results, 2);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "c"); // Lowest distance
        assert_eq!(merged[1].id, "a");
    }

    #[test]
    fn test_result_merger_rrf() {
        let merger = ResultMerger::new(MergeStrategy::ReciprocalRankFusion);

        let results = vec![
            (
                "inst-1".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "a".to_string(),
                        distance: 0.1,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "b".to_string(),
                        distance: 0.2,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
            (
                "inst-2".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "a".to_string(),
                        distance: 0.15,
                        metadata: None,
                        source_instance: "inst-2".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "c".to_string(),
                        distance: 0.25,
                        metadata: None,
                        source_instance: "inst-2".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
        ];

        let merged = merger.merge(results, 3);

        // 'a' should be first (appears in both lists)
        assert_eq!(merged[0].id, "a");
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_result_merger_consensus() {
        let merger = ResultMerger::new(MergeStrategy::Consensus);

        let results = vec![
            (
                "inst-1".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "common".to_string(),
                        distance: 0.1,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "unique-1".to_string(),
                        distance: 0.05,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
            (
                "inst-2".to_string(),
                vec![FederatedSearchResult {
                    id: "common".to_string(),
                    distance: 0.15,
                    metadata: None,
                    source_instance: "inst-2".to_string(),
                    collection: "test".to_string(),
                }],
            ),
            (
                "inst-3".to_string(),
                vec![FederatedSearchResult {
                    id: "common".to_string(),
                    distance: 0.12,
                    metadata: None,
                    source_instance: "inst-3".to_string(),
                    collection: "test".to_string(),
                }],
            ),
        ];

        let merged = merger.merge(results, 2);

        // 'common' should be first (appears in all 3 lists)
        assert_eq!(merged[0].id, "common");
    }

    #[test]
    fn test_federation_health() {
        let federation = create_test_federation();

        let health = federation.health();

        assert_eq!(health.total_instances, 3);
        assert_eq!(health.healthy_instances, 3);
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_federation_health_degraded() {
        let federation = create_test_federation();

        // Mark one instance as unhealthy
        federation
            .registry
            .update_health("eu-west-1", HealthStatus::Unhealthy);

        let health = federation.health();

        assert_eq!(health.healthy_instances, 2);
        assert_eq!(health.unhealthy_instances, 1);
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_routing_score() {
        let mut info = InstanceInfo::new(InstanceConfig::new("test", "http://localhost:8080"));

        info.status = HealthStatus::Healthy;
        info.avg_latency_ms = 100.0;
        info.config.priority = 2;

        let score = info.routing_score(0.7);

        // Score should be reasonable
        assert!(score > 0.0);
        assert!(score < 10.0);
    }

    #[test]
    fn test_instance_success_rate() {
        let mut info = InstanceInfo::new(InstanceConfig::new("test", "http://localhost:8080"));

        info.total_queries = 100;
        info.failed_queries = 10;

        assert!((info.success_rate() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_quorum_routing() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Quorum(2));
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("inst-1", "http://localhost:8081"));
        federation.register_instance(InstanceConfig::new("inst-2", "http://localhost:8082"));
        federation.register_instance(InstanceConfig::new("inst-3", "http://localhost:8083"));

        federation
            .registry
            .update_health("inst-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-2", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-3", HealthStatus::Healthy);

        let instances = federation.select_instances("test_collection").unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_quorum_not_reached() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Quorum(3));
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("inst-1", "http://localhost:8081"));
        federation.register_instance(InstanceConfig::new("inst-2", "http://localhost:8082"));

        federation
            .registry
            .update_health("inst-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-2", HealthStatus::Healthy);

        let result = federation.select_instances("test_collection");
        assert!(matches!(
            result,
            Err(FederationError::QuorumNotReached { .. })
        ));
    }

    #[test]
    fn test_federation_stats() {
        let federation = create_test_federation();

        let query = vec![0.1; 128];
        let _ = federation.search("test_collection", &query, 10);
        let _ = federation.search("test_collection", &query, 10);

        let stats = federation.stats();
        assert_eq!(stats.total_queries, 2);
    }

    #[test]
    fn test_health_check() {
        let federation = create_test_federation();

        let result = federation.check_instance_health("us-west-1");

        assert_eq!(result.instance_id, "us-west-1");
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_instances_by_region() {
        let federation = create_test_federation();

        let us_west = federation.registry.instances_by_region("us-west");
        let us_east = federation.registry.instances_by_region("us-east");

        assert_eq!(us_west.len(), 1);
        assert_eq!(us_east.len(), 1);
    }

    #[test]
    fn test_round_robin_selection() {
        let registry = InstanceRegistry::new();

        registry.register(InstanceConfig::new("inst-1", "http://localhost:8081"));
        registry.register(InstanceConfig::new("inst-2", "http://localhost:8082"));

        registry.update_health("inst-1", HealthStatus::Healthy);
        registry.update_health("inst-2", HealthStatus::Healthy);

        let first = registry.select_round_robin();
        let second = registry.select_round_robin();

        assert!(first.is_some());
        assert!(second.is_some());
        // Should alternate (though order depends on internal storage)
    }

    #[test]
    fn test_federation_config_builder() {
        let config = FederationConfig::default()
            .with_timeout(Duration::from_secs(5))
            .with_min_instances(2)
            .with_routing(RoutingStrategy::PriorityBased)
            .with_merge(MergeStrategy::Consensus)
            .with_partial_results(false);

        assert_eq!(config.query_timeout, Duration::from_secs(5));
        assert_eq!(config.min_instances, 2);
        assert_eq!(config.routing_strategy, RoutingStrategy::PriorityBased);
        assert_eq!(config.merge_strategy, MergeStrategy::Consensus);
        assert!(!config.allow_partial);
    }

    // ---- Discovery Service tests ----

    #[test]
    fn test_discovery_heartbeat() {
        let registry = Arc::new(InstanceRegistry::new());
        let svc = DiscoveryService::new(registry.clone(), DiscoveryConfig::default());

        svc.heartbeat("inst-1", "http://localhost:8081", HashMap::new());
        assert_eq!(svc.tracked_count(), 1);
        assert!(registry.get("inst-1").is_some());
        assert_eq!(
            registry.get("inst-1").unwrap().status,
            HealthStatus::Healthy
        );
    }

    #[test]
    fn test_discovery_with_region() {
        let registry = Arc::new(InstanceRegistry::new());
        let svc = DiscoveryService::new(registry.clone(), DiscoveryConfig::default());

        let mut meta = HashMap::new();
        meta.insert("region".to_string(), "eu-west".to_string());
        svc.heartbeat("eu-1", "http://eu:8081", meta);

        let eu = registry.instances_by_region("eu-west");
        assert_eq!(eu.len(), 1);
    }

    #[test]
    fn test_discovery_check_heartbeats() {
        let registry = Arc::new(InstanceRegistry::new());
        let config = DiscoveryConfig {
            heartbeat_interval: Duration::from_millis(1),
            missed_heartbeat_threshold: 1,
            auto_remove_stale: false,
        };
        let svc = DiscoveryService::new(registry.clone(), config);

        svc.heartbeat("inst-1", "http://localhost:8081", HashMap::new());
        std::thread::sleep(Duration::from_millis(5));

        let stale = svc.check_heartbeats();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], "inst-1");
    }

    // ---- Cross-Instance Dedup tests ----

    #[test]
    fn test_dedup_best_distance() {
        let dedup = CrossInstanceDedup::new(DedupStrategy::BestDistance);
        let r1 = vec![
            FederatedSearchResult {
                id: "v1".into(),
                distance: 0.5,
                metadata: None,
                source_instance: "inst-1".into(),
                collection: "test".into(),
            },
            FederatedSearchResult {
                id: "v2".into(),
                distance: 0.3,
                metadata: None,
                source_instance: "inst-1".into(),
                collection: "test".into(),
            },
        ];
        let r2 = vec![FederatedSearchResult {
            id: "v1".into(),
            distance: 0.2,
            metadata: None,
            source_instance: "inst-2".into(),
            collection: "test".into(),
        }];

        let merged = dedup.dedup(&[r1, r2], 10);
        assert_eq!(merged.len(), 2);
        // v1 should have distance 0.2 (best)
        let v1 = merged.iter().find(|r| r.id == "v1").unwrap();
        assert!((v1.distance - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_average_distance() {
        let dedup = CrossInstanceDedup::new(DedupStrategy::AverageDistance);
        let r1 = vec![FederatedSearchResult {
            id: "v1".into(),
            distance: 0.2,
            metadata: None,
            source_instance: "inst-1".into(),
            collection: "test".into(),
        }];
        let r2 = vec![FederatedSearchResult {
            id: "v1".into(),
            distance: 0.4,
            metadata: None,
            source_instance: "inst-2".into(),
            collection: "test".into(),
        }];

        let merged = dedup.dedup(&[r1, r2], 10);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].distance - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_truncates() {
        let dedup = CrossInstanceDedup::new(DedupStrategy::BestDistance);
        let results: Vec<Vec<FederatedSearchResult>> = (0..20)
            .map(|i| {
                vec![FederatedSearchResult {
                    id: format!("v{}", i),
                    distance: i as f32 * 0.1,
                    metadata: None,
                    source_instance: "inst-1".into(),
                    collection: "test".into(),
                }]
            })
            .collect();

        let merged = dedup.dedup(&results, 5);
        assert_eq!(merged.len(), 5);
    }

    // ---- Query Planner tests ----

    #[test]
    fn test_query_planner_basic() {
        let registry = Arc::new(InstanceRegistry::new());
        registry
            .register(InstanceConfig::new("i1", "http://localhost:8081").with_collection("docs"));
        registry
            .register(InstanceConfig::new("i2", "http://localhost:8082").with_collection("docs"));
        registry.update_health("i1", HealthStatus::Healthy);
        registry.update_health("i2", HealthStatus::Healthy);

        let planner = QueryPlanner::new(registry);
        let plan = planner
            .plan("docs", 10, ConsistencyLevel::One, Duration::from_secs(5))
            .unwrap();
        assert_eq!(plan.target_instances.len(), 1);
        assert_eq!(plan.k, 10);
    }

    #[test]
    fn test_query_planner_quorum() {
        let registry = Arc::new(InstanceRegistry::new());
        for i in 0..5 {
            registry.register(
                InstanceConfig::new(format!("i{}", i), format!("http://localhost:{}", 8080 + i))
                    .with_collection("data"),
            );
            registry.update_health(&format!("i{}", i), HealthStatus::Healthy);
        }

        let planner = QueryPlanner::new(registry);
        let plan = planner
            .plan("data", 10, ConsistencyLevel::Quorum, Duration::from_secs(5))
            .unwrap();
        assert_eq!(plan.target_instances.len(), 3); // 5/2 + 1 = 3
    }

    #[test]
    fn test_query_planner_validate() {
        let registry = Arc::new(InstanceRegistry::new());
        registry
            .register(InstanceConfig::new("i1", "http://localhost:8081").with_collection("docs"));
        registry.update_health("i1", HealthStatus::Healthy);

        let planner = QueryPlanner::new(registry.clone());
        let plan = planner
            .plan("docs", 10, ConsistencyLevel::One, Duration::from_secs(5))
            .unwrap();
        assert!(planner.validate(&plan));

        registry.update_health("i1", HealthStatus::Unhealthy);
        assert!(!planner.validate(&plan));
    }
}
