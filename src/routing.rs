//! # Query Routing Module
//!
//! Provides distributed query routing and result aggregation for sharded deployments.
//! This module handles routing queries to the appropriate shards, parallel execution,
//! and merging results from multiple shards.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────┐
//! │   Query Router   │
//! └────────┬─────────┘
//!          │ Route query to shards
//!          ▼
//! ┌──────────────────────────────────────┐
//! │  Parallel Shard Execution            │
//! │  ┌────────┐ ┌────────┐ ┌────────┐   │
//! │  │Shard 0 │ │Shard 1 │ │Shard 2 │   │
//! │  └───┬────┘ └───┬────┘ └───┬────┘   │
//! └──────┼──────────┼──────────┼────────┘
//!        │          │          │
//!        ▼          ▼          ▼
//! ┌──────────────────────────────────────┐
//! │         Result Aggregator            │
//! │    Merge and sort by distance        │
//! └──────────────────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::routing::{QueryRouter, RouteConfig};
//! use needle::shard::ShardManager;
//!
//! let router = QueryRouter::new(shard_manager);
//!
//! // Search across all shards
//! let results = router.search(&query_vector, 10).await?;
//! ```

use crate::shard::{ShardId, ShardManager, ShardState};
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Routing errors
#[derive(Error, Debug)]
pub enum RoutingError {
    #[error("No shards available")]
    NoShardsAvailable,

    #[error("Query timeout after {0:?}")]
    Timeout(Duration),

    #[error("Partial results: {success} of {total} shards responded")]
    PartialResults { success: usize, total: usize },

    #[error("All shards failed: {0}")]
    AllShardsFailed(String),

    #[error("Shard error: {0}")]
    ShardError(String),
}

pub type RoutingResult<T> = std::result::Result<T, RoutingError>;

/// Configuration for query routing
#[derive(Debug, Clone)]
pub struct RouteConfig {
    /// Timeout for shard queries
    pub timeout: Duration,
    /// Minimum shards that must respond
    pub min_shards: usize,
    /// Allow partial results on timeout
    pub allow_partial: bool,
    /// Maximum concurrent shard queries
    pub max_concurrent: usize,
    /// Retry failed shards
    pub retry_failed: bool,
    /// Load balancing strategy
    pub load_balancing: LoadBalancing,
}

impl Default for RouteConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(10),
            min_shards: 1,
            allow_partial: true,
            max_concurrent: 16,
            retry_failed: true,
            load_balancing: LoadBalancing::RoundRobin,
        }
    }
}

impl RouteConfig {
    /// Set query timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set minimum responding shards
    pub fn with_min_shards(mut self, min: usize) -> Self {
        self.min_shards = min;
        self
    }

    /// Set whether partial results are allowed
    pub fn with_partial_results(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancing {
    /// Simple round-robin
    RoundRobin,
    /// Random selection
    Random,
    /// Least loaded shard
    LeastLoaded,
    /// Locality-aware (prefer local shards)
    LocalityAware,
}

/// A search result from a shard
#[derive(Debug, Clone)]
pub struct ShardSearchResult {
    /// Shard that produced this result
    pub shard_id: ShardId,
    /// Vector ID
    pub id: String,
    /// Distance to query
    pub distance: f32,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Aggregated search results from multiple shards
#[derive(Debug, Clone)]
pub struct AggregatedResults {
    /// Merged and sorted results
    pub results: Vec<ShardSearchResult>,
    /// Total results before limiting
    pub total_found: usize,
    /// Shards that responded
    pub shards_responded: Vec<ShardId>,
    /// Shards that failed
    pub shards_failed: Vec<ShardId>,
    /// Query execution time
    pub execution_time: Duration,
}

/// Query router for distributed search
pub struct QueryRouter {
    shard_manager: Arc<ShardManager>,
    config: RouteConfig,
    stats: RouterStats,
    round_robin_counter: AtomicU64,
}

impl QueryRouter {
    /// Create a new query router
    pub fn new(shard_manager: Arc<ShardManager>) -> Self {
        Self {
            shard_manager,
            config: RouteConfig::default(),
            stats: RouterStats::default(),
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Create with custom config
    pub fn with_config(shard_manager: Arc<ShardManager>, config: RouteConfig) -> Self {
        Self {
            shard_manager,
            config,
            stats: RouterStats::default(),
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Get all active shards
    pub fn get_active_shards(&self) -> Vec<ShardId> {
        self.shard_manager
            .list_shards()
            .into_iter()
            .filter(|s| s.state == ShardState::Active || s.state == ShardState::ReadOnly)
            .map(|s| s.id)
            .collect()
    }

    /// Select a shard based on load balancing strategy
    pub fn select_shard(&self) -> Option<ShardId> {
        let active = self.get_active_shards();
        if active.is_empty() {
            return None;
        }

        match self.config.load_balancing {
            LoadBalancing::RoundRobin => {
                let idx =
                    self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize % active.len();
                Some(active[idx])
            }
            LoadBalancing::Random => {
                use rand::Rng;
                let idx = rand::thread_rng().gen_range(0..active.len());
                Some(active[idx])
            }
            LoadBalancing::LeastLoaded => {
                active
                    .iter()
                    .min_by_key(|id| {
                        self.shard_manager
                            .get_shard(**id)
                            .map(|s| s.vector_count)
                            .unwrap_or(u64::MAX)
                    })
                    .copied()
            }
            LoadBalancing::LocalityAware => {
                // For now, just use round-robin
                // In a real implementation, this would prefer shards on the same node
                let idx =
                    self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize % active.len();
                Some(active[idx])
            }
        }
    }

    /// Route a vector ID to its primary shard
    pub fn route(&self, vector_id: &str) -> ShardId {
        self.stats.total_routes.fetch_add(1, Ordering::Relaxed);
        self.shard_manager.route_id(vector_id)
    }

    /// Route with replicas
    pub fn route_with_replicas(&self, vector_id: &str) -> Vec<ShardId> {
        self.shard_manager.route_with_replicas(vector_id)
    }

    /// Get shard for a single-shard query (like get by ID)
    pub fn route_single(&self, vector_id: &str) -> RoutingResult<ShardId> {
        let shard_id = self.route(vector_id);

        // Check if shard is available
        if let Some(shard) = self.shard_manager.get_shard(shard_id) {
            if shard.state == ShardState::Active || shard.state == ShardState::ReadOnly {
                return Ok(shard_id);
            }
        }

        Err(RoutingError::NoShardsAvailable)
    }

    /// Get all shards for a scatter query (like search)
    pub fn route_scatter(&self) -> RoutingResult<Vec<ShardId>> {
        let shards = self.get_active_shards();
        if shards.is_empty() {
            return Err(RoutingError::NoShardsAvailable);
        }
        Ok(shards)
    }

    /// Merge results from multiple shards
    pub fn merge_results(
        &self,
        shard_results: Vec<(ShardId, Vec<ShardSearchResult>)>,
        k: usize,
        start_time: Instant,
    ) -> AggregatedResults {
        let mut all_results: Vec<ShardSearchResult> = shard_results
            .iter()
            .flat_map(|(_, results)| results.clone())
            .collect();

        let total_found = all_results.len();
        let shards_responded: Vec<ShardId> = shard_results.iter().map(|(id, _)| *id).collect();

        // Sort by distance (ascending)
        all_results.sort_by_key(|r| OrderedFloat(r.distance));

        // Take top k
        all_results.truncate(k);

        self.stats.merges.fetch_add(1, Ordering::Relaxed);

        AggregatedResults {
            results: all_results,
            total_found,
            shards_responded,
            shards_failed: vec![],
            execution_time: start_time.elapsed(),
        }
    }

    /// Merge with partial failures
    pub fn merge_with_failures(
        &self,
        successful: Vec<(ShardId, Vec<ShardSearchResult>)>,
        failed: Vec<ShardId>,
        k: usize,
        start_time: Instant,
    ) -> RoutingResult<AggregatedResults> {
        // Check minimum shards requirement
        if successful.len() < self.config.min_shards
            && !self.config.allow_partial {
                return Err(RoutingError::PartialResults {
                    success: successful.len(),
                    total: successful.len() + failed.len(),
                });
            }

        let mut result = self.merge_results(successful, k, start_time);
        result.shards_failed = failed;

        Ok(result)
    }

    /// Get router statistics
    pub fn stats(&self) -> RouterStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get configuration
    pub fn config(&self) -> &RouteConfig {
        &self.config
    }
}

/// Router statistics
#[derive(Debug, Default)]
pub struct RouterStats {
    total_routes: AtomicU64,
    scatter_queries: AtomicU64,
    single_queries: AtomicU64,
    merges: AtomicU64,
    timeouts: AtomicU64,
    partial_results: AtomicU64,
}

impl RouterStats {
    /// Record a scatter query
    pub fn record_scatter(&self) {
        self.scatter_queries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a single-shard query
    pub fn record_single(&self) {
        self.single_queries.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a timeout
    pub fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record partial results
    pub fn record_partial(&self) {
        self.partial_results.fetch_add(1, Ordering::Relaxed);
    }

    /// Get snapshot
    pub fn snapshot(&self) -> RouterStatsSnapshot {
        RouterStatsSnapshot {
            total_routes: self.total_routes.load(Ordering::Relaxed),
            scatter_queries: self.scatter_queries.load(Ordering::Relaxed),
            single_queries: self.single_queries.load(Ordering::Relaxed),
            merges: self.merges.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
            partial_results: self.partial_results.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of router stats
#[derive(Debug, Clone)]
pub struct RouterStatsSnapshot {
    pub total_routes: u64,
    pub scatter_queries: u64,
    pub single_queries: u64,
    pub merges: u64,
    pub timeouts: u64,
    pub partial_results: u64,
}

/// Result collector for parallel shard queries
pub struct ResultCollector {
    results: HashMap<ShardId, Vec<ShardSearchResult>>,
    errors: HashMap<ShardId, String>,
    expected_shards: usize,
}

impl ResultCollector {
    /// Create a new collector
    pub fn new(expected_shards: usize) -> Self {
        Self {
            results: HashMap::new(),
            errors: HashMap::new(),
            expected_shards,
        }
    }

    /// Add results from a shard
    pub fn add_results(&mut self, shard_id: ShardId, results: Vec<ShardSearchResult>) {
        self.results.insert(shard_id, results);
    }

    /// Add an error from a shard
    pub fn add_error(&mut self, shard_id: ShardId, error: String) {
        self.errors.insert(shard_id, error);
    }

    /// Check if all expected shards have responded
    pub fn is_complete(&self) -> bool {
        self.results.len() + self.errors.len() >= self.expected_shards
    }

    /// Get successful results
    pub fn successful(&self) -> Vec<(ShardId, Vec<ShardSearchResult>)> {
        self.results
            .iter()
            .map(|(id, results)| (*id, results.clone()))
            .collect()
    }

    /// Get failed shards
    pub fn failed(&self) -> Vec<ShardId> {
        self.errors.keys().copied().collect()
    }

    /// Get success count
    pub fn success_count(&self) -> usize {
        self.results.len()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shard::ShardConfig;

    fn create_test_router() -> QueryRouter {
        let config = ShardConfig::new(4);
        let manager = Arc::new(ShardManager::new(config));
        QueryRouter::new(manager)
    }

    #[test]
    fn test_route_single() {
        let router = create_test_router();

        // Route should work for active shards
        let result = router.route_single("test_vector");
        assert!(result.is_ok());
    }

    #[test]
    fn test_route_scatter() {
        let router = create_test_router();

        let shards = router.route_scatter().unwrap();
        assert_eq!(shards.len(), 4);
    }

    #[test]
    fn test_select_shard_round_robin() {
        let router = create_test_router();

        let shard1 = router.select_shard();
        let shard2 = router.select_shard();
        let shard3 = router.select_shard();
        let shard4 = router.select_shard();
        let shard5 = router.select_shard();

        // Should cycle through shards
        assert!(shard1.is_some());
        assert_eq!(shard5, shard1); // Back to first after 4
    }

    #[test]
    fn test_merge_results() {
        let router = create_test_router();
        let start = Instant::now();

        let shard_results = vec![
            (
                ShardId::new(0),
                vec![
                    ShardSearchResult {
                        shard_id: ShardId::new(0),
                        id: "a".to_string(),
                        distance: 0.5,
                        metadata: None,
                    },
                    ShardSearchResult {
                        shard_id: ShardId::new(0),
                        id: "b".to_string(),
                        distance: 0.8,
                        metadata: None,
                    },
                ],
            ),
            (
                ShardId::new(1),
                vec![ShardSearchResult {
                    shard_id: ShardId::new(1),
                    id: "c".to_string(),
                    distance: 0.3,
                    metadata: None,
                }],
            ),
        ];

        let merged = router.merge_results(shard_results, 2, start);

        assert_eq!(merged.results.len(), 2);
        assert_eq!(merged.results[0].id, "c"); // Lowest distance
        assert_eq!(merged.results[1].id, "a");
        assert_eq!(merged.total_found, 3);
        assert_eq!(merged.shards_responded.len(), 2);
    }

    #[test]
    fn test_result_collector() {
        let mut collector = ResultCollector::new(3);

        collector.add_results(
            ShardId::new(0),
            vec![ShardSearchResult {
                shard_id: ShardId::new(0),
                id: "a".to_string(),
                distance: 0.5,
                metadata: None,
            }],
        );

        collector.add_error(ShardId::new(1), "Connection failed".to_string());

        assert!(!collector.is_complete());
        assert_eq!(collector.success_count(), 1);
        assert_eq!(collector.error_count(), 1);

        collector.add_results(ShardId::new(2), vec![]);
        assert!(collector.is_complete());
    }

    #[test]
    fn test_router_stats() {
        let router = create_test_router();

        // Make some routes
        router.route("key1");
        router.route("key2");
        router.route("key3");

        let stats = router.stats();
        assert_eq!(stats.total_routes, 3);
    }

    #[test]
    fn test_route_config() {
        let config = RouteConfig::default()
            .with_timeout(Duration::from_secs(5))
            .with_min_shards(2)
            .with_partial_results(false);

        assert_eq!(config.timeout, Duration::from_secs(5));
        assert_eq!(config.min_shards, 2);
        assert!(!config.allow_partial);
    }
}
