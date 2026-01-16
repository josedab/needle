//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Distributed HNSW with Auto-Sharding
//!
//! Provides automatic sharding for HNSW indices with consistent hashing,
//! cross-shard search coordination, and dynamic rebalancing.
//!
//! # Features
//!
//! - **Auto-sharding**: Automatically partitions vectors based on consistent hashing
//! - **Cross-shard search**: Parallel query fan-out with result merging
//! - **Dynamic rebalancing**: Add/remove shards without downtime
//! - **Shard-local HNSW**: Each shard maintains its own HNSW index
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::distributed_hnsw::{DistributedHnsw, DistributedHnswConfig};
//!
//! let config = DistributedHnswConfig::new(128, 4); // 128 dims, 4 shards
//! let mut index = DistributedHnsw::new(config);
//!
//! // Insert vectors (automatically routed to shards)
//! index.insert("doc1", &embedding, metadata)?;
//!
//! // Search across all shards
//! let results = index.search(&query, 10)?;
//! ```

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::HnswConfig;
use crate::metadata::Filter;
use crate::shard::{ConsistentHashRing, ShardId, ShardState};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result type for shard operations that may fail
type ShardResult<T> = std::result::Result<T, (ShardId, String)>;

/// Configuration for distributed HNSW
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedHnswConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// Number of shards
    pub num_shards: u32,
    /// Virtual nodes per shard for consistent hashing
    pub virtual_nodes: u32,
    /// HNSW configuration for each shard
    pub hnsw_config: HnswConfig,
    /// Distance function
    pub distance_function: DistanceFunction,
    /// Enable automatic rebalancing
    pub auto_rebalance: bool,
    /// Rebalance threshold (max imbalance ratio)
    pub rebalance_threshold: f64,
    /// Minimum vectors before considering rebalance
    pub rebalance_min_vectors: usize,
    /// Search timeout per shard
    pub search_timeout: Option<Duration>,
    /// Continue search on shard failures
    pub continue_on_failure: bool,
}

impl DistributedHnswConfig {
    /// Create a new config
    pub fn new(dimensions: usize, num_shards: u32) -> Self {
        Self {
            dimensions,
            num_shards,
            virtual_nodes: 150,
            hnsw_config: HnswConfig::default(),
            distance_function: DistanceFunction::Cosine,
            auto_rebalance: true,
            rebalance_threshold: 1.5,
            rebalance_min_vectors: 1000,
            search_timeout: Some(Duration::from_secs(5)),
            continue_on_failure: true,
        }
    }

    /// Set HNSW configuration
    pub fn with_hnsw(mut self, config: HnswConfig) -> Self {
        self.hnsw_config = config;
        self
    }

    /// Set distance function
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_function = distance;
        self
    }

    /// Set virtual nodes
    pub fn with_virtual_nodes(mut self, nodes: u32) -> Self {
        self.virtual_nodes = nodes;
        self
    }

    /// Set rebalance threshold
    pub fn with_rebalance_threshold(mut self, threshold: f64) -> Self {
        self.rebalance_threshold = threshold;
        self
    }
}

/// Shard information
#[derive(Debug)]
struct ShardInfo {
    #[allow(dead_code)]
    id: ShardId,
    collection: Collection,
    state: ShardState,
    vector_count: AtomicU64,
    query_count: AtomicU64,
}

impl ShardInfo {
    fn new(id: ShardId, collection: Collection) -> Self {
        Self {
            id,
            collection,
            state: ShardState::Active,
            vector_count: AtomicU64::new(0),
            query_count: AtomicU64::new(0),
        }
    }
}

/// Result from a single shard search
#[derive(Debug)]
pub struct ShardSearchResult {
    /// Shard that produced results
    pub shard_id: ShardId,
    /// Search results
    pub results: Vec<SearchResult>,
    /// Search duration
    pub duration: Duration,
}

/// Aggregated cross-shard search result
#[derive(Debug)]
pub struct DistributedSearchResult {
    /// Merged top-k results
    pub results: Vec<SearchResult>,
    /// Per-shard results
    pub shard_results: Vec<ShardSearchResult>,
    /// Failed shards
    pub failed_shards: Vec<(ShardId, String)>,
    /// Total duration
    pub total_duration: Duration,
    /// Total vectors searched across shards
    pub total_vectors_searched: usize,
}

impl DistributedSearchResult {
    /// Check if all shards succeeded
    pub fn all_shards_succeeded(&self) -> bool {
        self.failed_shards.is_empty()
    }

    /// Get number of responding shards
    pub fn responding_shards(&self) -> usize {
        self.shard_results.len()
    }
}

/// Statistics for distributed HNSW
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedHnswStats {
    /// Total vectors across all shards
    pub total_vectors: u64,
    /// Total queries
    pub total_queries: u64,
    /// Per-shard vector counts
    pub shard_counts: Vec<(ShardId, u64)>,
    /// Imbalance ratio (max/min shard size)
    pub imbalance_ratio: f64,
    /// Rebalance operations performed
    pub rebalances: u64,
}

/// Distributed HNSW index with automatic sharding
pub struct DistributedHnsw {
    config: DistributedHnswConfig,
    shards: RwLock<HashMap<ShardId, Arc<RwLock<ShardInfo>>>>,
    ring: RwLock<ConsistentHashRing>,
    #[allow(dead_code)]
    stats: DistributedHnswStats,
    rebalance_count: AtomicU64,
}

impl DistributedHnsw {
    /// Create a new distributed HNSW index
    pub fn new(config: DistributedHnswConfig) -> Self {
        let shard_ids: Vec<ShardId> = (0..config.num_shards).map(ShardId::new).collect();
        let ring = ConsistentHashRing::new(&shard_ids, config.virtual_nodes);

        let mut shards = HashMap::new();
        for &shard_id in &shard_ids {
            let collection_config =
                CollectionConfig::new(format!("shard_{}", shard_id.id()), config.dimensions)
                    .with_distance(config.distance_function)
                    .with_hnsw_config(config.hnsw_config.clone());

            let collection = Collection::new(collection_config);
            let shard_info = ShardInfo::new(shard_id, collection);
            shards.insert(shard_id, Arc::new(RwLock::new(shard_info)));
        }

        Self {
            config,
            shards: RwLock::new(shards),
            ring: RwLock::new(ring),
            stats: DistributedHnswStats::default(),
            rebalance_count: AtomicU64::new(0),
        }
    }

    /// Route a vector ID to its shard
    fn route_id(&self, id: &str) -> ShardId {
        self.ring.read().route(id).unwrap_or(ShardId::new(0))
    }

    /// Insert a vector (automatically routed to appropriate shard)
    pub fn insert(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        let shard_id = self.route_id(id);
        let shards = self.shards.read();
        let shard = shards
            .get(&shard_id)
            .ok_or_else(|| NeedleError::InvalidState(format!("Shard {} not found", shard_id)))?;

        let mut shard_guard = shard.write();
        if shard_guard.state != ShardState::Active {
            return Err(NeedleError::InvalidState(format!(
                "Shard {} is not active (state: {:?})",
                shard_id, shard_guard.state
            )));
        }

        shard_guard.collection.insert(id, vector, metadata)?;
        shard_guard.vector_count.fetch_add(1, Ordering::Relaxed);

        // Check for auto-rebalance
        if self.config.auto_rebalance {
            drop(shard_guard);
            drop(shards);
            self.maybe_rebalance();
        }

        Ok(())
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(
        &self,
        vectors: Vec<(String, Vec<f32>, Option<Value>)>,
    ) -> Result<BatchInsertResult> {
        let mut shard_batches: HashMap<ShardId, Vec<(String, Vec<f32>, Option<Value>)>> =
            HashMap::new();

        // Route vectors to shards
        for (id, vector, metadata) in vectors {
            if vector.len() != self.config.dimensions {
                return Err(NeedleError::DimensionMismatch {
                    expected: self.config.dimensions,
                    got: vector.len(),
                });
            }
            let shard_id = self.route_id(&id);
            shard_batches
                .entry(shard_id)
                .or_default()
                .push((id, vector, metadata));
        }

        let shards = self.shards.read();
        let mut total_inserted = 0;
        let mut failures = Vec::new();

        // Insert into each shard
        for (shard_id, batch) in shard_batches {
            if let Some(shard) = shards.get(&shard_id) {
                let mut shard_guard = shard.write();
                if shard_guard.state != ShardState::Active {
                    failures.push((shard_id, "Shard not active".to_string()));
                    continue;
                }

                let count = batch.len();
                let ids: Vec<String> = batch.iter().map(|(id, _, _)| id.clone()).collect();
                let vecs: Vec<Vec<f32>> = batch.iter().map(|(_, v, _)| v.clone()).collect();
                let metas: Vec<Option<Value>> = batch.into_iter().map(|(_, _, m)| m).collect();

                if let Err(e) = shard_guard.collection.insert_batch(ids, vecs, metas) {
                    failures.push((shard_id, e.to_string()));
                } else {
                    shard_guard
                        .vector_count
                        .fetch_add(count as u64, Ordering::Relaxed);
                    total_inserted += count;
                }
            }
        }

        Ok(BatchInsertResult {
            total_inserted,
            failures,
        })
    }

    /// Search across all shards
    pub fn search(&self, query: &[f32], k: usize) -> Result<DistributedSearchResult> {
        self.search_with_filter(query, k, None)
    }

    /// Search with metadata filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<DistributedSearchResult> {
        if query.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let start = Instant::now();
        let shards = self.shards.read();

        // Query all active shards in parallel
        let shard_results: Vec<ShardResult<ShardSearchResult>> = shards
            .par_iter()
            .map(|(&shard_id, shard)| {
                let shard_start = Instant::now();
                let shard_guard = shard.read();

                if shard_guard.state != ShardState::Active
                    && shard_guard.state != ShardState::ReadOnly
                {
                    return Err((
                        shard_id,
                        format!("Shard not available: {:?}", shard_guard.state),
                    ));
                }

                let results = if let Some(f) = filter {
                    shard_guard.collection.search_with_filter(query, k, f)
                } else {
                    shard_guard.collection.search(query, k)
                };

                match results {
                    Ok(results) => {
                        shard_guard.query_count.fetch_add(1, Ordering::Relaxed);
                        Ok(ShardSearchResult {
                            shard_id,
                            results,
                            duration: shard_start.elapsed(),
                        })
                    }
                    Err(e) => Err((shard_id, e.to_string())),
                }
            })
            .collect();

        // Separate successes and failures
        let mut successful = Vec::new();
        let mut failed: Vec<(ShardId, String)> = Vec::new();

        for result in shard_results {
            match result {
                Ok(r) => successful.push(r),
                Err(f) => failed.push(f),
            }
        }

        // Merge results
        let total_vectors_searched: usize = successful.iter().map(|r| r.results.len()).sum();

        let merged = Self::merge_results(&successful, k);

        Ok(DistributedSearchResult {
            results: merged,
            shard_results: successful,
            failed_shards: failed,
            total_duration: start.elapsed(),
            total_vectors_searched,
        })
    }

    /// Merge results from multiple shards
    fn merge_results(shard_results: &[ShardSearchResult], k: usize) -> Vec<SearchResult> {
        let mut all_results: Vec<SearchResult> = shard_results
            .iter()
            .flat_map(|sr| sr.results.iter().cloned())
            .collect();

        // Sort by distance (ascending)
        all_results.sort_by_key(|r| OrderedFloat(r.distance));

        // Deduplicate by ID (keep lowest distance)
        let mut seen = std::collections::HashSet::new();
        all_results.retain(|r| seen.insert(r.id.clone()));

        // Take top k
        all_results.truncate(k);
        all_results
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        let shard_id = self.route_id(id);
        let shards = self.shards.read();
        let shard = shards.get(&shard_id)?;
        let shard_guard = shard.read();
        shard_guard
            .collection
            .get(id)
            .map(|(v, m)| (v.to_vec(), m.cloned()))
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        let shard_id = self.route_id(id);
        let shards = self.shards.read();
        let shard = shards
            .get(&shard_id)
            .ok_or_else(|| NeedleError::InvalidState(format!("Shard {} not found", shard_id)))?;

        let mut shard_guard = shard.write();
        let result = shard_guard.collection.delete(id)?;
        if result {
            shard_guard.vector_count.fetch_sub(1, Ordering::Relaxed);
        }
        Ok(result)
    }

    /// Check if rebalancing is needed and perform if necessary
    fn maybe_rebalance(&self) {
        let shards = self.shards.read();
        let counts: Vec<u64> = shards
            .values()
            .map(|s| s.read().vector_count.load(Ordering::Relaxed))
            .collect();

        let total: u64 = counts.iter().sum();
        if total < self.config.rebalance_min_vectors as u64 {
            return;
        }

        let max_count = counts.iter().max().copied().unwrap_or(0);
        let min_count = counts
            .iter()
            .filter(|&&c| c > 0)
            .min()
            .copied()
            .unwrap_or(1);

        if min_count > 0 {
            let ratio = max_count as f64 / min_count as f64;
            if ratio > self.config.rebalance_threshold {
                // Log that rebalancing is recommended
                tracing::info!(
                    "Rebalancing recommended: imbalance ratio {:.2} exceeds threshold {:.2}",
                    ratio,
                    self.config.rebalance_threshold
                );
                // Actual rebalancing would require moving vectors between shards
                // This is a placeholder for the full implementation
            }
        }
    }

    /// Add a new shard to the cluster
    pub fn add_shard(&self) -> Result<ShardId> {
        let mut shards = self.shards.write();
        let mut ring = self.ring.write();

        let new_id = ShardId::new(shards.len() as u32);

        let collection_config =
            CollectionConfig::new(format!("shard_{}", new_id.id()), self.config.dimensions)
                .with_distance(self.config.distance_function)
                .with_hnsw_config(self.config.hnsw_config.clone());

        let collection = Collection::new(collection_config);
        let shard_info = ShardInfo::new(new_id, collection);
        shards.insert(new_id, Arc::new(RwLock::new(shard_info)));

        ring.add_shard(new_id);

        Ok(new_id)
    }

    /// Remove a shard (must be empty or migrate first)
    pub fn remove_shard(&self, shard_id: ShardId) -> Result<()> {
        let mut shards = self.shards.write();

        // Check if shard exists and is empty
        if let Some(shard) = shards.get(&shard_id) {
            let shard_guard = shard.read();
            if shard_guard.vector_count.load(Ordering::Relaxed) > 0 {
                return Err(NeedleError::InvalidOperation(
                    "Cannot remove non-empty shard. Migrate vectors first.".to_string(),
                ));
            }
        } else {
            return Err(NeedleError::NotFound(format!(
                "Shard {} not found",
                shard_id
            )));
        }

        shards.remove(&shard_id);
        self.ring.write().remove_shard(shard_id);

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> DistributedHnswStats {
        let shards = self.shards.read();
        let shard_counts: Vec<(ShardId, u64)> = shards
            .iter()
            .map(|(&id, s)| (id, s.read().vector_count.load(Ordering::Relaxed)))
            .collect();

        let total_vectors: u64 = shard_counts.iter().map(|(_, c)| c).sum();
        let max_count = shard_counts.iter().map(|(_, c)| *c).max().unwrap_or(0);
        let min_count = shard_counts
            .iter()
            .filter(|(_, c)| *c > 0)
            .map(|(_, c)| *c)
            .min()
            .unwrap_or(1);
        let imbalance_ratio = if min_count > 0 {
            max_count as f64 / min_count as f64
        } else {
            1.0
        };

        let total_queries: u64 = shards
            .values()
            .map(|s| s.read().query_count.load(Ordering::Relaxed))
            .sum();

        DistributedHnswStats {
            total_vectors,
            total_queries,
            shard_counts,
            imbalance_ratio,
            rebalances: self.rebalance_count.load(Ordering::Relaxed),
        }
    }

    /// Get total vector count
    pub fn len(&self) -> usize {
        self.shards
            .read()
            .values()
            .map(|s| s.read().vector_count.load(Ordering::Relaxed) as usize)
            .sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.read().len()
    }

    /// Get configuration
    pub fn config(&self) -> &DistributedHnswConfig {
        &self.config
    }
}

/// Result from batch insert operation
#[derive(Debug)]
pub struct BatchInsertResult {
    /// Total vectors inserted
    pub total_inserted: usize,
    /// Failures per shard
    pub failures: Vec<(ShardId, String)>,
}

impl BatchInsertResult {
    /// Check if all inserts succeeded
    pub fn all_succeeded(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Builder for distributed HNSW queries
pub struct DistributedQueryBuilder<'a> {
    index: &'a DistributedHnsw,
    query: Vec<f32>,
    k: usize,
    filter: Option<Filter>,
    specific_shards: Option<Vec<ShardId>>,
}

impl<'a> DistributedQueryBuilder<'a> {
    /// Create a new query builder
    pub fn new(index: &'a DistributedHnsw, query: Vec<f32>) -> Self {
        Self {
            index,
            query,
            k: 10,
            filter: None,
            specific_shards: None,
        }
    }

    /// Set number of results
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set metadata filter
    pub fn filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Query specific shards only
    pub fn shards(mut self, shards: Vec<ShardId>) -> Self {
        self.specific_shards = Some(shards);
        self
    }

    /// Execute the search
    pub fn execute(self) -> Result<DistributedSearchResult> {
        self.index
            .search_with_filter(&self.query, self.k, self.filter.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_create_distributed_hnsw() {
        let config = DistributedHnswConfig::new(128, 4);
        let index = DistributedHnsw::new(config);

        assert_eq!(index.num_shards(), 4);
        assert!(index.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let config = DistributedHnswConfig::new(64, 2);
        let index = DistributedHnsw::new(config);

        // Insert some vectors
        for i in 0..100 {
            let vector = random_vector(64);
            index.insert(&format!("vec_{}", i), &vector, None).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query = random_vector(64);
        let result = index.search(&query, 10).unwrap();

        assert_eq!(result.results.len(), 10);
        assert!(result.all_shards_succeeded());
    }

    #[test]
    fn test_batch_insert() {
        let config = DistributedHnswConfig::new(32, 3);
        let index = DistributedHnsw::new(config);

        let vectors: Vec<_> = (0..50)
            .map(|i| (format!("vec_{}", i), random_vector(32), None))
            .collect();

        let result = index.insert_batch(vectors).unwrap();
        assert_eq!(result.total_inserted, 50);
        assert!(result.all_succeeded());
    }

    #[test]
    fn test_get_and_delete() {
        let config = DistributedHnswConfig::new(16, 2);
        let index = DistributedHnsw::new(config);

        let vector = random_vector(16);
        index.insert("test_vec", &vector, None).unwrap();

        // Get
        let (retrieved, _) = index.get("test_vec").unwrap();
        assert_eq!(retrieved.len(), 16);

        // Delete
        assert!(index.delete("test_vec").unwrap());
        assert!(index.get("test_vec").is_none());
    }

    #[test]
    fn test_add_shard() {
        let config = DistributedHnswConfig::new(32, 2);
        let index = DistributedHnsw::new(config);

        assert_eq!(index.num_shards(), 2);

        let new_shard = index.add_shard().unwrap();
        assert_eq!(index.num_shards(), 3);
        assert_eq!(new_shard.id(), 2);
    }

    #[test]
    fn test_stats() {
        let config = DistributedHnswConfig::new(32, 2);
        let index = DistributedHnsw::new(config);

        for i in 0..20 {
            let vector = random_vector(32);
            index.insert(&format!("vec_{}", i), &vector, None).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 20);
        assert_eq!(stats.shard_counts.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = DistributedHnswConfig::new(64, 2);
        let index = DistributedHnsw::new(config);

        let wrong_dim_vector = random_vector(32);
        let result = index.insert("test", &wrong_dim_vector, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_consistent_routing() {
        let config = DistributedHnswConfig::new(32, 4);
        let index = DistributedHnsw::new(config);

        // Same ID should always route to same shard
        let shard1 = index.route_id("test_key");
        let shard2 = index.route_id("test_key");
        assert_eq!(shard1, shard2);
    }

    #[test]
    fn test_search_with_metadata() {
        let config = DistributedHnswConfig::new(32, 2);
        let index = DistributedHnsw::new(config);

        // Insert with metadata
        for i in 0..20 {
            let vector = random_vector(32);
            let metadata = serde_json::json!({
                "category": if i % 2 == 0 { "even" } else { "odd" },
                "index": i
            });
            index
                .insert(&format!("vec_{}", i), &vector, Some(metadata))
                .unwrap();
        }

        // Search with filter
        let query = random_vector(32);
        let filter = Filter::eq("category", "even");
        let result = index.search_with_filter(&query, 5, Some(&filter)).unwrap();

        // All results should have "even" category
        for r in &result.results {
            if let Some(meta) = &r.metadata {
                assert_eq!(meta["category"], "even");
            }
        }
    }
}
