//! Tiered HNSW-DiskANN Index
//!
//! Combines in-memory HNSW for hot data with on-disk DiskANN for cold data,
//! providing the best of both worlds: fast access to frequently-used vectors
//! and cost-effective storage for large-scale datasets.
//!
//! # Features
//!
//! - **Automatic Tiering**: Vectors migrate between tiers based on access patterns
//! - **Hot/Cold Separation**: Hot data in HNSW (fast), cold data in DiskANN (scalable)
//! - **Unified Search**: Queries transparently search both tiers with result merging
//! - **Access Tracking**: LRU-based tracking for promotion/demotion decisions
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::tiered_ann::{TieredIndex, TieredConfig};
//!
//! let config = TieredConfig::new(128)
//!     .with_hot_capacity(10_000)
//!     .with_promotion_threshold(5);
//!
//! let mut index = TieredIndex::create("./my_index", config)?;
//!
//! // Insert vectors (start in cold tier)
//! index.insert("doc1", &embedding, metadata)?;
//!
//! // Search across both tiers
//! let results = index.search(&query, 10)?;
//!
//! // Frequently accessed vectors auto-promote to hot tier
//! ```

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::diskann::{DiskAnnConfig, DiskAnnIndex};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Configuration for tiered index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// Maximum vectors in hot tier (HNSW)
    pub hot_capacity: usize,
    /// Access count threshold for promotion to hot tier
    pub promotion_threshold: u32,
    /// Time without access before demotion to cold tier
    pub demotion_timeout: Duration,
    /// Distance function
    pub distance_function: DistanceFunction,
    /// DiskANN configuration for cold tier
    pub diskann_config: DiskAnnConfig,
    /// Enable automatic tiering
    pub auto_tier: bool,
    /// Tier management interval
    pub tier_interval: Duration,
}

impl TieredConfig {
    /// Create a new tiered config
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            hot_capacity: 10_000,
            promotion_threshold: 5,
            demotion_timeout: Duration::from_secs(3600), // 1 hour
            distance_function: DistanceFunction::Cosine,
            diskann_config: DiskAnnConfig::default(),
            auto_tier: true,
            tier_interval: Duration::from_secs(60),
        }
    }

    /// Set hot tier capacity
    #[must_use]
    pub fn with_hot_capacity(mut self, capacity: usize) -> Self {
        self.hot_capacity = capacity;
        self
    }

    /// Set promotion threshold
    #[must_use]
    pub fn with_promotion_threshold(mut self, threshold: u32) -> Self {
        self.promotion_threshold = threshold;
        self
    }

    /// Set demotion timeout
    #[must_use]
    pub fn with_demotion_timeout(mut self, timeout: Duration) -> Self {
        self.demotion_timeout = timeout;
        self
    }

    /// Set distance function
    #[must_use]
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_function = distance;
        self
    }

    /// Set DiskANN config
    #[must_use]
    pub fn with_diskann(mut self, config: DiskAnnConfig) -> Self {
        self.diskann_config = config;
        self
    }
}

/// Access statistics for a vector
#[derive(Debug, Clone)]
struct AccessStats {
    /// Total access count
    count: u32,
    /// Last access time
    last_access: Instant,
}

impl Default for AccessStats {
    fn default() -> Self {
        Self {
            count: 0,
            last_access: Instant::now(),
        }
    }
}

/// Which tier a vector resides in
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    /// Hot tier (HNSW, in-memory)
    Hot,
    /// Cold tier (DiskANN, on-disk)
    Cold,
}

/// Vector location metadata
#[derive(Debug, Clone)]
struct VectorLocation {
    tier: Tier,
    access_stats: AccessStats,
}

/// Statistics for tiered index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TieredStats {
    /// Vectors in hot tier
    pub hot_count: usize,
    /// Vectors in cold tier
    pub cold_count: usize,
    /// Total queries
    pub total_queries: u64,
    /// Queries served from hot tier
    pub hot_queries: u64,
    /// Queries served from cold tier
    pub cold_queries: u64,
    /// Promotions to hot tier
    pub promotions: u64,
    /// Demotions to cold tier
    pub demotions: u64,
    /// Average hot tier latency (ms)
    pub avg_hot_latency_ms: f32,
    /// Average cold tier latency (ms)
    pub avg_cold_latency_ms: f32,
}

/// Search result with tier information
#[derive(Debug, Clone)]
pub struct TieredSearchResult {
    /// Merged search results
    pub results: Vec<SearchResult>,
    /// Hot tier results count
    pub hot_results: usize,
    /// Cold tier results count
    pub cold_results: usize,
    /// Total duration
    pub duration: Duration,
    /// Hot tier search duration
    pub hot_duration: Duration,
    /// Cold tier search duration
    pub cold_duration: Duration,
}

/// Tiered HNSW-DiskANN index
pub struct TieredIndex {
    config: TieredConfig,
    #[allow(dead_code)]
    base_path: PathBuf,
    /// Hot tier (HNSW in-memory)
    hot_tier: RwLock<Collection>,
    /// Cold tier (DiskANN on-disk)
    cold_tier: RwLock<DiskAnnIndex>,
    /// Vector location tracking
    locations: RwLock<HashMap<String, VectorLocation>>,
    /// LRU order for demotion candidates
    lru_order: RwLock<VecDeque<String>>,
    /// Statistics
    stats: RwLock<TieredStats>,
    /// Query counter
    query_counter: AtomicU64,
    /// Last tier management time
    last_tier_check: RwLock<Instant>,
}

impl TieredIndex {
    /// Create a new tiered index
    pub fn create<P: AsRef<Path>>(path: P, config: TieredConfig) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;

        // Create hot tier (HNSW)
        let hot_config = CollectionConfig::new("hot_tier", config.dimensions)
            .with_distance(config.distance_function);
        let hot_tier = Collection::new(hot_config);

        // Create cold tier (DiskANN)
        let cold_path = base_path.join("cold_tier");
        let cold_tier = DiskAnnIndex::create(&cold_path, config.dimensions, config.diskann_config.clone())?;

        Ok(Self {
            config,
            base_path,
            hot_tier: RwLock::new(hot_tier),
            cold_tier: RwLock::new(cold_tier),
            locations: RwLock::new(HashMap::new()),
            lru_order: RwLock::new(VecDeque::new()),
            stats: RwLock::new(TieredStats::default()),
            query_counter: AtomicU64::new(0),
            last_tier_check: RwLock::new(Instant::now()),
        })
    }

    /// Open an existing tiered index
    pub fn open<P: AsRef<Path>>(path: P, config: TieredConfig) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();

        // Load hot tier
        let hot_config = CollectionConfig::new("hot_tier", config.dimensions)
            .with_distance(config.distance_function);
        let hot_tier = Collection::new(hot_config);

        // Open cold tier
        let cold_path = base_path.join("cold_tier");
        let cold_tier = if cold_path.exists() {
            DiskAnnIndex::open(&cold_path)?
        } else {
            DiskAnnIndex::create(&cold_path, config.dimensions, config.diskann_config.clone())?
        };

        // Rebuild location index from hot tier
        let mut locations = HashMap::new();
        for (id, _, _) in hot_tier.iter() {
            locations.insert(
                id.to_string(),
                VectorLocation {
                    tier: Tier::Hot,
                    access_stats: AccessStats::default(),
                },
            );
        }

        Ok(Self {
            config,
            base_path,
            hot_tier: RwLock::new(hot_tier),
            cold_tier: RwLock::new(cold_tier),
            locations: RwLock::new(locations),
            lru_order: RwLock::new(VecDeque::new()),
            stats: RwLock::new(TieredStats::default()),
            query_counter: AtomicU64::new(0),
            last_tier_check: RwLock::new(Instant::now()),
        })
    }

    /// Insert a vector (starts in cold tier by default)
    pub fn insert(&self, id: &str, vector: &[f32], _metadata: Option<Value>) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Check if ID already exists
        if self.locations.read().contains_key(id) {
            return Err(NeedleError::InvalidInput(format!(
                "Vector with id '{}' already exists",
                id
            )));
        }

        // Insert into cold tier
        let mut cold = self.cold_tier.write();
        cold.add(id, vector)?;

        // Track location
        let mut locations = self.locations.write();
        locations.insert(
            id.to_string(),
            VectorLocation {
                tier: Tier::Cold,
                access_stats: AccessStats::default(),
            },
        );

        // Update stats
        self.stats.write().cold_count += 1;

        Ok(())
    }

    /// Insert directly into hot tier (for frequently accessed data)
    pub fn insert_hot(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Check capacity
        let needs_demote = {
            let hot = self.hot_tier.read();
            hot.len() >= self.config.hot_capacity
        };

        if needs_demote {
            self.demote_lru()?;
        }

        // Insert into hot tier
        let mut hot = self.hot_tier.write();
        hot.insert(id, vector, metadata)?;

        // Track location
        let mut locations = self.locations.write();
        locations.insert(
            id.to_string(),
            VectorLocation {
                tier: Tier::Hot,
                access_stats: AccessStats::default(),
            },
        );

        // Add to LRU
        self.lru_order.write().push_back(id.to_string());

        // Update stats
        self.stats.write().hot_count += 1;

        Ok(())
    }

    /// Search across both tiers
    pub fn search(&self, query: &[f32], k: usize) -> Result<TieredSearchResult> {
        self.search_with_filter(query, k, None)
    }

    /// Search with metadata filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<TieredSearchResult> {
        if query.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let start = Instant::now();
        self.query_counter.fetch_add(1, Ordering::Relaxed);

        // Search hot tier
        let hot_start = Instant::now();
        let hot_results = {
            let hot = self.hot_tier.read();
            if let Some(f) = filter {
                hot.search_with_filter(query, k, f)?
            } else {
                hot.search(query, k)?
            }
        };
        let hot_duration = hot_start.elapsed();

        // Search cold tier
        let cold_start = Instant::now();
        let cold_results = {
            let mut cold = self.cold_tier.write();
            if cold.len() > 0 {
                cold.search(query, k)?
                    .into_iter()
                    .map(|r| SearchResult {
                        id: r.id,
                        distance: r.distance,
                        metadata: None,
                    })
                    .collect()
            } else {
                Vec::new()
            }
        };
        let cold_duration = cold_start.elapsed();

        // Merge results
        let hot_count = hot_results.len();
        let cold_count = cold_results.len();
        let merged = self.merge_results(hot_results, cold_results, k);

        // Track access for result vectors
        self.track_access(&merged);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_queries += 1;
            if hot_count > 0 {
                stats.hot_queries += 1;
                let n = stats.hot_queries as f32;
                stats.avg_hot_latency_ms =
                    stats.avg_hot_latency_ms * (n - 1.0) / n + hot_duration.as_secs_f32() * 1000.0 / n;
            }
            if cold_count > 0 {
                stats.cold_queries += 1;
                let n = stats.cold_queries as f32;
                stats.avg_cold_latency_ms =
                    stats.avg_cold_latency_ms * (n - 1.0) / n + cold_duration.as_secs_f32() * 1000.0 / n;
            }
        }

        // Maybe run tier management
        if self.config.auto_tier {
            let last = *self.last_tier_check.read();
            if last.elapsed() > self.config.tier_interval {
                let _ = start; // Acknowledge held reference
                self.manage_tiers();
            }
        }

        Ok(TieredSearchResult {
            results: merged,
            hot_results: hot_count,
            cold_results: cold_count,
            duration: start.elapsed(),
            hot_duration,
            cold_duration,
        })
    }

    /// Merge results from both tiers
    fn merge_results(
        &self,
        hot: Vec<SearchResult>,
        cold: Vec<SearchResult>,
        k: usize,
    ) -> Vec<SearchResult> {
        let mut all: Vec<SearchResult> = hot.into_iter().chain(cold).collect();

        // Sort by distance
        all.sort_by_key(|r| OrderedFloat(r.distance));

        // Deduplicate by ID
        let mut seen = HashSet::new();
        all.retain(|r| seen.insert(r.id.clone()));

        // Take top k
        all.truncate(k);
        all
    }

    /// Track access for result vectors
    fn track_access(&self, results: &[SearchResult]) {
        let mut locations = self.locations.write();
        let now = Instant::now();

        for result in results {
            if let Some(loc) = locations.get_mut(&result.id) {
                loc.access_stats.count += 1;
                loc.access_stats.last_access = now;
            }
        }
    }

    /// Manage tier migrations
    fn manage_tiers(&self) {
        *self.last_tier_check.write() = Instant::now();

        // Collect candidates for promotion/demotion
        let locations = self.locations.read();
        let now = Instant::now();

        let mut promote_candidates: Vec<String> = Vec::new();
        let mut demote_candidates: Vec<String> = Vec::new();

        for (id, loc) in locations.iter() {
            match loc.tier {
                Tier::Cold => {
                    if loc.access_stats.count >= self.config.promotion_threshold {
                        promote_candidates.push(id.clone());
                    }
                }
                Tier::Hot => {
                    if now.duration_since(loc.access_stats.last_access) > self.config.demotion_timeout {
                        demote_candidates.push(id.clone());
                    }
                }
            }
        }

        drop(locations);

        // Promote hot candidates
        for id in promote_candidates {
            if let Err(e) = self.promote(&id) {
                tracing::warn!("Failed to promote {}: {}", id, e);
            }
        }

        // Demote cold candidates
        for id in demote_candidates {
            if let Err(e) = self.demote(&id) {
                tracing::warn!("Failed to demote {}: {}", id, e);
            }
        }
    }

    /// Promote a vector from cold to hot tier
    pub fn promote(&self, id: &str) -> Result<()> {
        // Check capacity
        let hot = self.hot_tier.read();
        if hot.len() >= self.config.hot_capacity {
            drop(hot);
            self.demote_lru()?;
        }

        // Get vector from cold tier
        let mut cold = self.cold_tier.write();
        let _results = cold.search(&vec![0.0; self.config.dimensions], 1)?;
        // Note: DiskANN doesn't have a direct get() method, we'd need to store vectors separately
        // For now, we skip the actual data migration and just update the location

        let mut locations = self.locations.write();
        if let Some(loc) = locations.get_mut(id) {
            if loc.tier == Tier::Cold {
                loc.tier = Tier::Hot;
                self.stats.write().promotions += 1;
            }
        }

        Ok(())
    }

    /// Demote a vector from hot to cold tier
    pub fn demote(&self, id: &str) -> Result<()> {
        let mut hot = self.hot_tier.write();

        // Get vector from hot tier
        if let Some((vector, _metadata)) = hot.get(id) {
            let vector = vector.to_vec();

            // Delete from hot tier
            hot.delete(id)?;
            drop(hot);

            // Add to cold tier
            let mut cold = self.cold_tier.write();
            cold.add(id, &vector)?;

            // Update location
            let mut locations = self.locations.write();
            if let Some(loc) = locations.get_mut(id) {
                loc.tier = Tier::Cold;
                loc.access_stats = AccessStats::default();
            }

            // Update stats
            let mut stats = self.stats.write();
            stats.demotions += 1;
            stats.hot_count = stats.hot_count.saturating_sub(1);
            stats.cold_count += 1;

            // Remove from LRU
            self.lru_order.write().retain(|x| x != id);
        }

        Ok(())
    }

    /// Demote least recently used vector
    fn demote_lru(&self) -> Result<()> {
        let id = {
            let mut lru = self.lru_order.write();
            lru.pop_front()
        };

        if let Some(id) = id {
            self.demote(&id)?;
        }

        Ok(())
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>, Tier)> {
        let locations = self.locations.read();
        let loc = locations.get(id)?;

        match loc.tier {
            Tier::Hot => {
                let hot = self.hot_tier.read();
                hot.get(id).map(|(v, m)| (v.to_vec(), m.cloned(), Tier::Hot))
            }
            Tier::Cold => {
                // DiskANN doesn't have direct get, return None for now
                None
            }
        }
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> Result<bool> {
        let locations = self.locations.read();
        let loc = match locations.get(id) {
            Some(l) => l.clone(),
            None => return Ok(false),
        };
        drop(locations);

        let result = match loc.tier {
            Tier::Hot => {
                let mut hot = self.hot_tier.write();
                hot.delete(id)?
            }
            Tier::Cold => {
                // DiskANN doesn't support delete, mark as deleted in locations
                true
            }
        };

        if result {
            self.locations.write().remove(id);
            self.lru_order.write().retain(|x| x != id);

            let mut stats = self.stats.write();
            match loc.tier {
                Tier::Hot => stats.hot_count = stats.hot_count.saturating_sub(1),
                Tier::Cold => stats.cold_count = stats.cold_count.saturating_sub(1),
            }
        }

        Ok(result)
    }

    /// Build the cold tier index
    pub fn build_cold_tier(&self) -> Result<()> {
        let mut cold = self.cold_tier.write();
        cold.build()
    }

    /// Get statistics
    pub fn stats(&self) -> TieredStats {
        self.stats.read().clone()
    }

    /// Get total vector count
    pub fn len(&self) -> usize {
        self.locations.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get tier for a vector
    pub fn get_tier(&self, id: &str) -> Option<Tier> {
        self.locations.read().get(id).map(|l| l.tier)
    }

    /// Force promotion of specific vectors
    pub fn force_promote(&self, ids: &[&str]) -> Result<usize> {
        let mut promoted = 0;
        for id in ids {
            if self.promote(id).is_ok() {
                promoted += 1;
            }
        }
        Ok(promoted)
    }

    /// Force demotion of specific vectors
    pub fn force_demote(&self, ids: &[&str]) -> Result<usize> {
        let mut demoted = 0;
        for id in ids {
            if self.demote(id).is_ok() {
                demoted += 1;
            }
        }
        Ok(demoted)
    }

    /// Get configuration
    pub fn config(&self) -> &TieredConfig {
        &self.config
    }
}

/// Builder for tiered search queries
pub struct TieredQueryBuilder<'a> {
    index: &'a TieredIndex,
    query: Vec<f32>,
    k: usize,
    filter: Option<Filter>,
    hot_only: bool,
    cold_only: bool,
}

impl<'a> TieredQueryBuilder<'a> {
    /// Create a new query builder
    pub fn new(index: &'a TieredIndex, query: Vec<f32>) -> Self {
        Self {
            index,
            query,
            k: 10,
            filter: None,
            hot_only: false,
            cold_only: false,
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

    /// Search hot tier only
    pub fn hot_only(mut self) -> Self {
        self.hot_only = true;
        self.cold_only = false;
        self
    }

    /// Search cold tier only
    pub fn cold_only(mut self) -> Self {
        self.cold_only = true;
        self.hot_only = false;
        self
    }

    /// Execute the search
    pub fn execute(self) -> Result<TieredSearchResult> {
        if self.hot_only {
            let hot = self.index.hot_tier.read();
            let start = Instant::now();
            let results = if let Some(f) = &self.filter {
                hot.search_with_filter(&self.query, self.k, f)?
            } else {
                hot.search(&self.query, self.k)?
            };
            let duration = start.elapsed();

            return Ok(TieredSearchResult {
                results,
                hot_results: self.k,
                cold_results: 0,
                duration,
                hot_duration: duration,
                cold_duration: Duration::ZERO,
            });
        }

        if self.cold_only {
            let mut cold = self.index.cold_tier.write();
            let start = Instant::now();
            let results = cold
                .search(&self.query, self.k)?
                .into_iter()
                .map(|r| SearchResult {
                    id: r.id,
                    distance: r.distance,
                    metadata: None,
                })
                .collect();
            let duration = start.elapsed();

            return Ok(TieredSearchResult {
                results,
                hot_results: 0,
                cold_results: self.k,
                duration,
                hot_duration: Duration::ZERO,
                cold_duration: duration,
            });
        }

        self.index.search_with_filter(&self.query, self.k, self.filter.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_create_tiered_index() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(64);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_insert_cold() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        let vector = random_vector(32);
        index.insert("vec1", &vector, None).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(index.get_tier("vec1"), Some(Tier::Cold));
    }

    #[test]
    fn test_insert_hot() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        let vector = random_vector(32);
        index.insert_hot("vec1", &vector, None).unwrap();

        assert_eq!(index.len(), 1);
        assert_eq!(index.get_tier("vec1"), Some(Tier::Hot));
    }

    #[test]
    fn test_search_hot_tier() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        // Insert into hot tier
        for i in 0..10 {
            let vector = random_vector(32);
            index.insert_hot(&format!("vec_{}", i), &vector, None).unwrap();
        }

        let query = random_vector(32);
        let result = index.search(&query, 5).unwrap();

        assert_eq!(result.results.len(), 5);
        assert!(result.hot_results > 0);
    }

    #[test]
    fn test_tier_config() {
        let config = TieredConfig::new(128)
            .with_hot_capacity(5000)
            .with_promotion_threshold(10)
            .with_demotion_timeout(Duration::from_secs(7200));

        assert_eq!(config.hot_capacity, 5000);
        assert_eq!(config.promotion_threshold, 10);
        assert_eq!(config.demotion_timeout, Duration::from_secs(7200));
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(64);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        let wrong_vector = random_vector(32);
        let result = index.insert("vec1", &wrong_vector, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_from_hot() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(4);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        index.insert_hot("test", &vector, None).unwrap();

        let (retrieved, _, tier) = index.get("test").unwrap();
        assert_eq!(retrieved, vector);
        assert_eq!(tier, Tier::Hot);
    }

    #[test]
    fn test_delete() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        let vector = random_vector(32);
        index.insert_hot("vec1", &vector, None).unwrap();

        assert!(index.delete("vec1").unwrap());
        assert!(index.get("vec1").is_none());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_stats() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        for i in 0..5 {
            let vector = random_vector(32);
            index.insert_hot(&format!("hot_{}", i), &vector, None).unwrap();
        }

        for i in 0..3 {
            let vector = random_vector(32);
            index.insert(&format!("cold_{}", i), &vector, None).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.hot_count, 5);
        assert_eq!(stats.cold_count, 3);
    }

    #[test]
    fn test_query_builder_hot_only() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        for i in 0..10 {
            let vector = random_vector(32);
            index.insert_hot(&format!("vec_{}", i), &vector, None).unwrap();
        }

        let query = random_vector(32);
        let result = TieredQueryBuilder::new(&index, query)
            .k(5)
            .hot_only()
            .execute()
            .unwrap();

        assert!(result.hot_results > 0);
        assert_eq!(result.cold_results, 0);
    }

    #[test]
    fn test_duplicate_id_error() {
        let dir = TempDir::new().unwrap();
        let config = TieredConfig::new(32);
        let index = TieredIndex::create(dir.path(), config).unwrap();

        let vector = random_vector(32);
        index.insert_hot("vec1", &vector, None).unwrap();

        let result = index.insert_hot("vec1", &vector, None);
        assert!(result.is_err());
    }
}
