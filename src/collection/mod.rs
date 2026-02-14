//! Vector Collection Management
//!
//! A collection is a container for vectors with the same dimensionality.
//! It provides vector storage, HNSW indexing, and metadata management.
//!
//! # Overview
//!
//! Collections are the primary way to organize vectors in Needle. Each collection:
//! - Has a fixed dimensionality (all vectors must match)
//! - Uses a single distance function for similarity
//! - Maintains an HNSW index for fast approximate search
//! - Supports JSON metadata attached to each vector
//!
//! # Example
//!
//! ```
//! use needle::{Collection, CollectionConfig, DistanceFunction};
//! use serde_json::json;
//!
//! // Create a collection for 128-dimensional embeddings
//! let config = CollectionConfig::new("embeddings", 128)
//!     .with_distance(DistanceFunction::Cosine);
//! let mut collection = Collection::new(config);
//!
//! // Insert a vector with metadata
//! let embedding = vec![0.1; 128];
//! collection.insert("doc1", &embedding, Some(json!({"title": "Hello World"})))?;
//!
//! // Search for similar vectors
//! let query = vec![0.1; 128];
//! let results = collection.search(&query, 10)?;
//! # Ok::<(), needle::error::NeedleError>(())
//! ```
//!
//! # Thread Safety
//!
//! Collections are not thread-safe by themselves. Use `Database` for concurrent
//! access, which wraps collections in `RwLock` and provides `CollectionRef` handles.
//!
//! # Search Methods
//!
//! - [`search`](Collection::search) - Basic k-NN search
//! - [`search_with_filter`](Collection::search_with_filter) - Search with metadata filter
//! - [`search_builder`](Collection::search_builder) - Fluent search configuration
//! - [`batch_search`](Collection::batch_search) - Parallel multi-query search

pub mod config;
pub use config::*;
pub mod search;
pub use search::*;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswIndex, VectorId};
use crate::metadata::{Filter, MetadataStore};
use crate::storage::VectorStore;
use lru::LruCache;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;

/// Over-fetch multiplier for filtered searches: retrieve `k * FILTER_CANDIDATE_MULTIPLIER`
/// candidates from the HNSW index to compensate for results removed by filtering.
const FILTER_CANDIDATE_MULTIPLIER: usize = 10;

/// Fluent builder for configuring and executing vector similarity searches.
///
/// `SearchBuilder` provides fine-grained control over search behavior, including
/// filtering strategies, performance tuning, and distance function overrides.
///
/// # Filtering Strategies
///
/// Two filtering modes are available:
///
/// | Mode | Method | When to Use |
/// |------|--------|-------------|
/// | **Pre-filter** | [`filter()`](Self::filter) | Filter is fast and selective; filters during HNSW traversal |
/// | **Post-filter** | [`post_filter()`](Self::post_filter) | Need to guarantee k candidates; filter after ANN search |
///
/// Pre-filtering integrates with HNSW search, skipping non-matching candidates during
/// traversal. Post-filtering fetches extra candidates (`k * post_filter_factor`), then
/// filters and truncates.
///
/// # Performance Tuning
///
/// - [`ef_search()`](Self::ef_search): Higher values improve recall but increase latency
/// - [`include_metadata(false)`](Self::include_metadata): Skip metadata loading for faster searches
/// - [`post_filter_factor()`](Self::post_filter_factor): Adjust over-fetch ratio for post-filtering
///
/// # Example: Basic Search
///
/// ```
/// use needle::Collection;
///
/// let mut collection = Collection::with_dimensions("docs", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
///
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
///
/// # Example: Pre-Filter vs Post-Filter
///
/// ```
/// use needle::{Collection, Filter};
/// use serde_json::json;
///
/// let mut collection = Collection::with_dimensions("docs", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a", "score": 10})))?;
/// collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "b", "score": 20})))?;
///
/// // Pre-filter: filter candidates BEFORE ANN search
/// // Use when: filter is fast, you don't need exactly k results
/// let pre_filter = Filter::eq("type", "a");
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .filter(&pre_filter)
///     .execute()?;
///
/// // Post-filter: filter results AFTER ANN search
/// // Use when: need to guarantee k candidates before filtering
/// let post_filter = Filter::gt("score", 15);
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .post_filter(&post_filter)
///     .post_filter_factor(5)  // Fetch 50 candidates, filter, keep 10
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
///
/// # Example: Performance Tuning
///
/// ```
/// use needle::Collection;
///
/// let mut collection = Collection::with_dimensions("docs", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
///
/// // Higher ef_search for better recall
/// let high_recall = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .ef_search(200)  // Default is 50
///     .execute()?;
///
/// // Skip metadata for faster response
/// let fast_search = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .include_metadata(false)
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
#[derive(Clone)]
pub struct SearchBuilder<'a> {
    collection: &'a Collection,
    query: &'a [f32],
    k: usize,
    filter: Option<&'a Filter>,
    post_filter: Option<&'a Filter>,
    post_filter_factor: usize,
    ef_search: Option<usize>,
    include_metadata: bool,
    /// Override the distance function for this query.
    /// When set to a different function than the collection's default,
    /// search will fall back to brute-force for accuracy.
    distance_override: Option<DistanceFunction>,
}

impl<'a> SearchBuilder<'a> {
    /// Create a new search builder
    #[must_use]
    pub fn new(collection: &'a Collection, query: &'a [f32]) -> Self {
        Self {
            collection,
            query,
            k: 10,
            filter: None,
            post_filter: None,
            post_filter_factor: 3,
            ef_search: None,
            include_metadata: true,
            distance_override: None,
        }
    }

    /// Set the number of results to return
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a pre-filter (applied during ANN search).
    ///
    /// Pre-filtering is efficient when the filter is selective and fast to evaluate.
    /// Candidates that don't match the filter are skipped during search.
    #[must_use]
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a post-filter (applied after ANN search).
    ///
    /// Post-filtering is useful when:
    /// - You need to guarantee k results before filtering
    /// - The filter involves expensive computation
    /// - The filter is highly selective and pre-filtering would miss results
    ///
    /// The search fetches `k * post_filter_factor` candidates, then filters.
    /// Default over-fetch factor is 3x.
    #[must_use]
    pub fn post_filter(mut self, filter: &'a Filter) -> Self {
        self.post_filter = Some(filter);
        self
    }

    /// Set the over-fetch factor for post-filtering (default: 3).
    ///
    /// When post-filtering, the search fetches `k * factor` candidates
    /// to ensure enough results remain after filtering.
    #[must_use]
    pub fn post_filter_factor(mut self, factor: usize) -> Self {
        self.post_filter_factor = factor.max(1);
        self
    }

    /// Set ef_search parameter for this query
    #[must_use]
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Whether to include metadata in results (default: true)
    #[must_use]
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Override the distance function for this query.
    ///
    /// When the distance function differs from the collection's configured function,
    /// search falls back to brute-force linear scan for accurate results.
    /// This allows querying with different similarity metrics without rebuilding the index.
    ///
    /// **Warning:** Brute-force search is O(n) and may be slow on large collections.
    /// A warning is logged when this fallback occurs.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, DistanceFunction};
    ///
    /// let mut collection = Collection::with_dimensions("docs", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// // Query with Euclidean distance even though collection uses Cosine
    /// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
    ///     .k(10)
    ///     .distance(DistanceFunction::Euclidean)
    ///     .execute()?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    #[must_use]
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_override = Some(distance);
        self
    }

    /// Execute the search and return results
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        self.validate_query()?;

        // Check if we need brute-force search due to distance override
        if let Some(distance_fn) = self.brute_force_distance() {
            warn!(
                "Distance override ({:?}) differs from index ({:?}), using brute-force search",
                distance_fn, self.collection.config.distance
            );
            return self.collection.brute_force_search(&BruteForceSearchParams {
                query: self.query,
                k: self.k,
                distance_fn,
                filter: self.filter,
                post_filter: self.post_filter,
                include_metadata: self.include_metadata,
            });
        }

        let fetch_count = self.calculate_fetch_count();
        let raw_results = self.fetch_raw_results(fetch_count);
        let non_expired = self.filter_expired(raw_results);
        let post_filter_factor = if self.post_filter.is_some() {
            self.post_filter_factor
        } else {
            1
        };
        let pre_filtered = self.apply_pre_filter(non_expired, self.k * post_filter_factor.max(1));
        let mut enriched = self.enrich(pre_filtered)?;
        self.apply_post_filter(&mut enriched);
        Ok(enriched)
    }

    /// Validate query dimensions and vector values.
    fn validate_query(&self) -> Result<()> {
        if self.query.len() != self.collection.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.collection.config.dimensions,
                got: self.query.len(),
            });
        }
        Collection::validate_vector(self.query)
    }

    /// Returns the override distance function if brute-force fallback is needed.
    fn brute_force_distance(&self) -> Option<DistanceFunction> {
        self.distance_override
            .filter(|&d| d != self.collection.config.distance)
    }

    /// Calculate how many candidates to fetch from the index.
    fn calculate_fetch_count(&self) -> usize {
        let pre_filter_factor = if self.filter.is_some() {
            FILTER_CANDIDATE_MULTIPLIER
        } else {
            1
        };
        let post_filter_factor = if self.post_filter.is_some() {
            self.post_filter_factor
        } else {
            1
        };
        self.k * pre_filter_factor * post_filter_factor
    }

    /// Fetch raw results from the HNSW index.
    fn fetch_raw_results(&self, fetch_count: usize) -> Vec<(VectorId, f32)> {
        if let Some(ef) = self.ef_search {
            self.collection.index.search_with_ef(
                self.query,
                fetch_count,
                ef,
                self.collection.vectors.as_slice(),
            )
        } else {
            self.collection.index.search(
                self.query,
                fetch_count,
                self.collection.vectors.as_slice(),
            )
        }
    }

    /// Remove expired vectors if lazy expiration is enabled.
    fn filter_expired(&self, results: Vec<(VectorId, f32)>) -> Vec<(VectorId, f32)> {
        if self.collection.config.lazy_expiration {
            results
                .into_iter()
                .filter(|(id, _)| !self.collection.is_expired(*id))
                .collect()
        } else {
            results
        }
    }

    /// Apply the pre-filter (metadata filter during ANN search phase).
    fn apply_pre_filter(
        &self,
        results: Vec<(VectorId, f32)>,
        limit: usize,
    ) -> Vec<(VectorId, f32)> {
        if let Some(filter) = self.filter {
            results
                .into_iter()
                .filter(|(id, _)| {
                    self.collection
                        .metadata
                        .get(*id)
                        .map_or(false, |entry| filter.matches(entry.data.as_ref()))
                })
                .take(limit)
                .collect()
        } else {
            results.into_iter().take(limit).collect()
        }
    }

    /// Enrich raw results with metadata and external IDs.
    fn enrich(&self, pre_filtered: Vec<(VectorId, f32)>) -> Result<Vec<SearchResult>> {
        if self.include_metadata || self.post_filter.is_some() {
            self.collection.enrich_results(pre_filtered)
        } else {
            pre_filtered
                .into_iter()
                .map(|(id, distance)| {
                    let entry =
                        self.collection.metadata.get(id).ok_or_else(|| {
                            NeedleError::Index("Missing metadata for vector".into())
                        })?;
                    Ok(SearchResult {
                        id: entry.external_id.clone(),
                        distance,
                        metadata: None,
                    })
                })
                .collect::<Result<Vec<_>>>()
        }
    }

    /// Apply post-filter, truncate to k, and strip metadata if not requested.
    fn apply_post_filter(&self, enriched: &mut Vec<SearchResult>) {
        if let Some(post_filter) = self.post_filter {
            *enriched = enriched
                .drain(..)
                .filter(|result| post_filter.matches(result.metadata.as_ref()))
                .take(self.k)
                .collect();
        } else {
            enriched.truncate(self.k);
        }
        // Strip metadata if not requested (but was needed for post-filter)
        if !self.include_metadata && self.post_filter.is_some() {
            for result in enriched.iter_mut() {
                result.metadata = None;
            }
        }
    }

    /// Execute the search and return only IDs with distances
    pub fn execute_ids_only(self) -> Result<Vec<(String, f32)>> {
        self.include_metadata(false)
            .execute()
            .map(|results| results.into_iter().map(|r| (r.id, r.distance)).collect())
    }
}

/// Cache key for query result caching.
///
/// Uses OrderedFloat to make f32 values hashable while handling NaN/Inf correctly.
#[derive(Clone, PartialEq, Eq)]
struct QueryCacheKey {
    /// Query vector converted to ordered floats for hashing
    query: Vec<OrderedFloat<f32>>,
    /// Number of results requested
    k: usize,
}

impl Hash for QueryCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        for &v in &self.query {
            v.hash(state);
        }
    }
}

impl QueryCacheKey {
    fn new(query: &[f32], k: usize) -> Self {
        Self {
            query: query.iter().map(|&f| OrderedFloat(f)).collect(),
            k,
        }
    }
}

/// Cached search result entry
#[derive(Clone)]
struct CachedSearchResult {
    results: Vec<SearchResult>,
}

/// A collection of vectors with the same dimensions.
///
/// # Thread Safety
///
/// **Warning**: `Collection` is **NOT** thread-safe for concurrent access.
/// For multi-threaded use, always access collections through
/// [`Database::collection()`] which returns a thread-safe [`CollectionRef`].
///
/// Direct `Collection` usage is intended for:
/// - Single-threaded applications
/// - Unit testing
/// - Embedded scenarios where the caller manages synchronization
///
/// # Example (Thread-Safe Access)
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use needle::Database;
///
/// let db = Arc::new(Database::in_memory());
/// db.create_collection("embeddings", 128).unwrap();
///
/// // Safe concurrent access via CollectionRef
/// let handles: Vec<_> = (0..4).map(|i| {
///     let db = Arc::clone(&db);
///     thread::spawn(move || {
///         let coll = db.collection("embeddings").unwrap();
///         // CollectionRef wraps Collection in RwLock for safe access
///         let query = vec![0.1f32; 128];
///         coll.search(&query, 10).unwrap()
///     })
/// }).collect();
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct Collection {
    /// Collection configuration
    config: CollectionConfig,
    /// Vector storage
    vectors: VectorStore,
    /// HNSW index
    index: HnswIndex,
    /// Metadata storage
    metadata: MetadataStore,
    /// Query result cache (wrapped in Mutex for interior mutability)
    /// Skipped during serialization - cache is rebuilt on load
    #[serde(skip)]
    query_cache: Option<Mutex<QueryCache>>,
    /// TTL expiration tracking: internal_id -> expiration timestamp (Unix epoch seconds)
    /// Vectors with expired timestamps are filtered out during search (lazy)
    /// or removed during explicit sweep operations.
    #[serde(default)]
    expirations: std::collections::HashMap<usize, u64>,
}

/// Internal query cache with statistics tracking
struct QueryCache {
    cache: LruCache<QueryCacheKey, CachedSearchResult>,
    hits: u64,
    misses: u64,
}

impl std::fmt::Debug for QueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCache")
            .field("size", &self.cache.len())
            .field("capacity", &self.cache.cap())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl QueryCache {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: LruCache::new(capacity),
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &QueryCacheKey) -> Option<&CachedSearchResult> {
        if let Some(result) = self.cache.get(key) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }

    fn put(&mut self, key: QueryCacheKey, value: CachedSearchResult) {
        self.cache.put(key, value);
    }

    fn clear(&mut self) {
        self.cache.clear();
    }

    fn stats(&self, capacity: usize) -> QueryCacheStats {
        QueryCacheStats {
            hits: self.hits,
            misses: self.misses,
            size: self.cache.len(),
            capacity,
        }
    }
}

struct BruteForceSearchParams<'a> {
    query: &'a [f32],
    k: usize,
    distance_fn: DistanceFunction,
    filter: Option<&'a Filter>,
    post_filter: Option<&'a Filter>,
    include_metadata: bool,
}

impl Collection {
    /// Create a new collection
    pub fn new(config: CollectionConfig) -> Self {
        let query_cache = if config.query_cache.is_enabled() {
            NonZeroUsize::new(config.query_cache.capacity)
                .map(|cap| Mutex::new(QueryCache::new(cap)))
        } else {
            None
        };

        Self {
            vectors: VectorStore::new(config.dimensions),
            index: HnswIndex::new(config.hnsw.clone(), config.distance),
            metadata: MetadataStore::new(),
            query_cache,
            expirations: HashMap::new(),
            config,
        }
    }

    /// Create a collection with just name and dimensions
    pub fn with_dimensions(name: impl Into<String>, dimensions: usize) -> Self {
        Self::new(CollectionConfig::new(name, dimensions))
    }

    /// Create a collection with validated dimensions.
    pub fn try_with_dimensions(name: impl Into<String>, dimensions: usize) -> Result<Self> {
        CollectionConfig::try_new(name, dimensions).map(Self::new)
    }

    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the vector dimensions
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Get the number of active vectors (not including deleted)
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get the total number of vectors including deleted (for storage stats)
    pub fn total_vectors(&self) -> usize {
        self.vectors.len()
    }

    /// Get the collection configuration
    pub fn config(&self) -> &CollectionConfig {
        &self.config
    }

    /// Set ef_search for queries
    pub fn set_ef_search(&mut self, ef: usize) {
        self.index.set_ef_search(ef);
    }

    /// Get the slow query threshold in microseconds.
    ///
    /// Returns `None` if slow query logging is disabled.
    pub fn slow_query_threshold_us(&self) -> Option<u64> {
        self.config.slow_query_threshold_us
    }

    /// Set the slow query threshold in microseconds.
    ///
    /// When set, search queries that exceed this duration will be logged
    /// at the warn level. Set to `None` to disable slow query logging.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 128);
    ///
    /// // Enable slow query logging for queries > 50ms
    /// collection.set_slow_query_threshold_us(Some(50_000));
    ///
    /// // Disable slow query logging
    /// collection.set_slow_query_threshold_us(None);
    /// ```
    pub fn set_slow_query_threshold_us(&mut self, threshold_us: Option<u64>) {
        self.config.slow_query_threshold_us = threshold_us;
    }

    /// Check if query caching is enabled.
    pub fn is_query_cache_enabled(&self) -> bool {
        self.query_cache.is_some()
    }

    /// Enable query result caching with the specified capacity.
    ///
    /// If caching was already enabled, this replaces the existing cache.
    /// The old cache is cleared.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 128);
    ///
    /// // Enable caching with 500 entries
    /// collection.enable_query_cache(500);
    /// assert!(collection.is_query_cache_enabled());
    /// ```
    pub fn enable_query_cache(&mut self, capacity: usize) {
        if let Some(cap) = NonZeroUsize::new(capacity) {
            self.query_cache = Some(Mutex::new(QueryCache::new(cap)));
            self.config.query_cache = QueryCacheConfig::new(capacity);
        }
    }

    /// Disable query result caching.
    ///
    /// Clears the existing cache if present.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, CollectionConfig, QueryCacheConfig};
    ///
    /// let config = CollectionConfig::new("test", 128)
    ///     .with_query_cache_capacity(100);
    /// let mut collection = Collection::new(config);
    ///
    /// assert!(collection.is_query_cache_enabled());
    /// collection.disable_query_cache();
    /// assert!(!collection.is_query_cache_enabled());
    /// ```
    pub fn disable_query_cache(&mut self) {
        self.query_cache = None;
        self.config.query_cache = QueryCacheConfig::disabled();
    }

    /// Clear the query cache.
    ///
    /// This removes all cached results but keeps caching enabled.
    /// No-op if caching is disabled.
    pub fn clear_query_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.lock().clear();
        }
    }

    /// Get query cache statistics.
    ///
    /// Returns `None` if caching is disabled.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, CollectionConfig, QueryCacheConfig};
    ///
    /// let config = CollectionConfig::new("test", 128)
    ///     .with_query_cache_capacity(100);
    /// let mut collection = Collection::new(config);
    /// collection.insert("v1", &[0.0; 128], None).unwrap();
    ///
    /// // First search - cache miss
    /// let _ = collection.search(&[0.0; 128], 10);
    ///
    /// // Second search - cache hit
    /// let _ = collection.search(&[0.0; 128], 10);
    ///
    /// let stats = collection.query_cache_stats().unwrap();
    /// assert_eq!(stats.hits, 1);
    /// assert_eq!(stats.misses, 1);
    /// ```
    pub fn query_cache_stats(&self) -> Option<QueryCacheStats> {
        self.query_cache
            .as_ref()
            .map(|cache| cache.lock().stats(self.config.query_cache.capacity))
    }

    /// Helper to invalidate the query cache.
    /// Called automatically on mutations (insert, update, delete).
    fn invalidate_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.lock().clear();
        }
    }

    /// Helper to get a cached result or compute and cache it.
    fn search_with_cache<F>(&self, query: &[f32], k: usize, compute: F) -> Result<Vec<SearchResult>>
    where
        F: FnOnce() -> Result<Vec<SearchResult>>,
    {
        if let Some(ref cache) = self.query_cache {
            let cache_key = QueryCacheKey::new(query, k);

            // Try to get from cache
            {
                let mut cache_guard = cache.lock();
                if let Some(cached) = cache_guard.get(&cache_key) {
                    return Ok(cached.results.clone());
                }
            }

            // Cache miss - compute result
            let results = compute()?;

            // Store in cache
            {
                let mut cache_guard = cache.lock();
                cache_guard.put(
                    cache_key,
                    CachedSearchResult {
                        results: results.clone(),
                    },
                );
            }

            Ok(results)
        } else {
            // No cache - just compute
            compute()
        }
    }

    /// Validate that a vector contains only finite values (no NaN or Inf)
    fn validate_vector(vector: &[f32]) -> Result<()> {
        for (i, &val) in vector.iter().enumerate() {
            if val.is_nan() {
                return Err(NeedleError::InvalidVector(format!(
                    "Vector contains NaN at index {}",
                    i
                )));
            }
            if val.is_infinite() {
                return Err(NeedleError::InvalidVector(format!(
                    "Vector contains Inf at index {}",
                    i
                )));
            }
        }
        Ok(())
    }

    /// Validate query vector dimensions and values
    fn validate_query(&self, query: &[f32]) -> Result<()> {
        if query.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }
        Self::validate_vector(query)
    }

    /// Clamp k to the collection size to avoid wasting resources
    #[inline]
    fn clamp_k(&self, k: usize) -> usize {
        k.min(self.len())
    }

    /// Validate insert input (dimensions and vector values)
    fn validate_insert_input(&self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }
        Self::validate_vector(vector)
    }

    /// Insert a vector with ID and optional metadata.
    ///
    /// Adds a new vector to the collection with an associated ID and optional
    /// JSON metadata. The vector is indexed immediately for search.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The embedding vector (must match collection dimensions)
    /// * `metadata` - Optional JSON metadata (use `serde_json::json!` macro)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Vector dimensions don't match collection
    /// - [`NeedleError::InvalidVector`] - Vector contains NaN or Infinity values
    /// - [`NeedleError::VectorAlreadyExists`] - A vector with the same ID exists
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("docs", 4);
    /// collection.insert(
    ///     "doc1",
    ///     &[0.1, 0.2, 0.3, 0.4],
    ///     Some(json!({"title": "Hello", "category": "greeting"}))
    /// )?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.insert_with_ttl(id, vector, metadata, self.config.default_ttl_seconds)
    }

    /// Insert a vector with ID, optional metadata, and explicit TTL.
    ///
    /// Similar to `insert()`, but allows specifying a TTL (time-to-live) in seconds.
    /// The vector will automatically expire after the specified duration.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The embedding vector (must match collection dimensions)
    /// * `metadata` - Optional JSON metadata
    /// * `ttl_seconds` - Optional TTL in seconds; if `None`, uses collection default
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Vector dimensions don't match collection
    /// - [`NeedleError::InvalidVector`] - Vector contains NaN or Infinity values
    /// - [`NeedleError::VectorAlreadyExists`] - A vector with the same ID exists
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("ephemeral", 4);
    ///
    /// // Insert with 1-hour TTL
    /// collection.insert_with_ttl(
    ///     "temp1",
    ///     &[0.1, 0.2, 0.3, 0.4],
    ///     Some(json!({"type": "temporary"})),
    ///     Some(3600)
    /// )?;
    ///
    /// // Insert without TTL (permanent)
    /// collection.insert_with_ttl(
    ///     "perm1",
    ///     &[0.5, 0.6, 0.7, 0.8],
    ///     None,
    ///     None
    /// )?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn insert_with_ttl(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        self.insert_vec_with_ttl(id, vector.to_vec(), metadata, ttl_seconds)
    }

    /// Insert a vector with ID and optional metadata, taking ownership of the vector
    ///
    /// This is more efficient than `insert()` when you already have a `Vec<f32>`
    /// as it avoids an unnecessary allocation.
    pub fn insert_vec(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.insert_vec_with_ttl(id, vector, metadata, self.config.default_ttl_seconds)
    }

    /// Insert a vector with ID, optional metadata, and explicit TTL, taking ownership.
    ///
    /// Combines the efficiency of `insert_vec()` with TTL support. This is the core
    /// insert implementation — all other insert variants delegate here.
    pub fn insert_vec_with_ttl(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        let id = id.into();

        // Validate dimensions and vector values
        self.validate_insert_input(&vector)?;

        // Check if ID already exists
        if self.metadata.contains(&id) {
            return Err(NeedleError::VectorAlreadyExists(id));
        }

        // Add to vector store (no clone needed - we own the vector)
        let internal_id = self.vectors.add(vector)?;

        // Add to metadata
        self.metadata.insert(internal_id, id, metadata)?;

        // Add to index - get vector reference from store
        let vector_ref = self
            .vectors
            .get(internal_id)
            .ok_or_else(|| NeedleError::Index("Vector not found after insert".into()))?;
        self.index
            .insert(internal_id, vector_ref, self.vectors.as_slice())?;

        // Track expiration if TTL is specified
        if let Some(ttl) = ttl_seconds {
            let expiration = Self::now_unix() + ttl;
            self.expirations.insert(internal_id, expiration);
        }

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(())
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(
        &mut self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: Vec<Option<Value>>,
    ) -> Result<()> {
        if ids.len() != vectors.len() || ids.len() != metadata.len() {
            return Err(NeedleError::InvalidConfig(
                "Batch sizes must match".to_string(),
            ));
        }

        let mut seen_ids = HashSet::new();
        for id in &ids {
            if !seen_ids.insert(id.as_str()) {
                return Err(NeedleError::InvalidConfig(
                    "Batch contains duplicate IDs".to_string(),
                ));
            }
            if self.metadata.contains(id) {
                return Err(NeedleError::VectorAlreadyExists(id.clone()));
            }
        }

        for vector in &vectors {
            self.validate_insert_input(vector)?;
        }

        // Use insert_vec to avoid unnecessary clones
        let mut inserted_ids = Vec::new();
        for ((id, vector), meta) in ids.into_iter().zip(vectors).zip(metadata) {
            let id_string = id;
            match self.insert_vec(id_string.clone(), vector, meta) {
                Ok(_) => inserted_ids.push(id_string),
                Err(err) => {
                    for inserted in inserted_ids {
                        let _ = self.delete(&inserted);
                    }
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    /// Create a search builder for fluent search configuration
    ///
    /// # Example
    /// ```ignore
    /// let results = collection
    ///     .search_builder(&query)
    ///     .k(10)
    ///     .filter(&filter)
    ///     .ef_search(100)
    ///     .execute()?;
    /// ```
    pub fn search_builder<'a>(&'a self, query: &'a [f32]) -> SearchBuilder<'a> {
        SearchBuilder::new(self, query)
    }

    /// Search for the k nearest neighbors to a query vector.
    ///
    /// Performs approximate nearest neighbor search using the HNSW index.
    /// Results are sorted by distance (ascending) and include metadata.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Returns up to `k` [`SearchResult`]s, sorted by distance (closest first).
    /// May return fewer than `k` results if the collection has fewer vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match collection
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity values
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// let results = collection.search(&[1.0, 0.0, 0.0, 0.0], 5)?;
    /// assert_eq!(results[0].id, "v1"); // Closest match
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Use cache if enabled
        let results = self.search_with_cache(query, k, || {
            let raw_results = self.index.search(query, k, self.vectors.as_slice());

            // Apply lazy expiration filtering if enabled
            let filtered_results = if self.config.lazy_expiration {
                raw_results
                    .into_iter()
                    .filter(|(id, _)| !self.is_expired(*id))
                    .collect()
            } else {
                raw_results
            };

            self.enrich_results(filtered_results)
        })?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    results_count = results.len(),
                    collection_size = self.len(),
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Brute-force linear search with a specified distance function.
    ///
    /// This method scans all vectors in the collection and computes distances
    /// using the specified distance function. It's used when the query's distance
    /// function differs from the index's configured function.
    ///
    /// **Warning:** This is O(n) complexity and may be slow on large collections.
    fn brute_force_search(&self, params: &BruteForceSearchParams<'_>) -> Result<Vec<SearchResult>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let k = self.clamp_k(params.k);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Use a max-heap (via Reverse) to track top-k by smallest distance
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        // Linear scan over all non-deleted vectors
        for (internal_id, vector) in self.vectors.as_slice().iter().enumerate() {
            // Skip deleted vectors
            if self.index.is_deleted(internal_id) {
                continue;
            }

            // Skip expired vectors if lazy expiration is enabled
            if self.config.lazy_expiration && self.is_expired(internal_id) {
                continue;
            }

            // Apply pre-filter if present
            if let Some(f) = params.filter {
                if let Some(entry) = self.metadata.get(internal_id) {
                    if !f.matches(entry.data.as_ref()) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Compute distance with the specified function
            let dist = params.distance_fn.compute(params.query, vector);

            // Add to heap
            heap.push((Reverse(OrderedFloat(dist)), internal_id));

            // Keep only top-k
            if heap.len() > k {
                heap.pop();
            }
        }

        // Extract results in order (smallest distance first)
        let mut results: Vec<_> = heap.into_sorted_vec();
        results.reverse(); // BinaryHeap::into_sorted_vec returns largest first

        // Convert to SearchResults
        let mut search_results = Vec::with_capacity(results.len());
        for (Reverse(OrderedFloat(distance)), internal_id) in results {
            if let Some(entry) = self.metadata.get(internal_id) {
                let metadata = if params.include_metadata || params.post_filter.is_some() {
                    entry.data.clone()
                } else {
                    None
                };
                search_results.push(SearchResult {
                    id: entry.external_id.clone(),
                    distance,
                    metadata,
                });
            }
        }

        // Apply post-filter if present
        if let Some(pf) = params.post_filter {
            search_results.retain(|r| pf.matches(r.metadata.as_ref()));
        }

        // Strip metadata if not requested
        if !params.include_metadata {
            for result in &mut search_results {
                result.metadata = None;
            }
        }

        Ok(search_results)
    }

    /// Search with detailed query execution profiling.
    ///
    /// Returns both the search results and a [`SearchExplain`] struct containing
    /// detailed timing and statistics for query optimization and debugging.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A tuple of `(results, explain)` where:
    /// - `results` - The search results (same as [`search()`](Self::search))
    /// - `explain` - Detailed profiling information
    ///
    /// # Errors
    ///
    /// Returns an error if the query is invalid (wrong dimensions, NaN/Inf values).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// for i in 0..100 {
    ///     collection.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
    /// }
    ///
    /// let (results, explain) = collection.search_explain(&[50.0, 0.0, 0.0, 0.0], 10)?;
    ///
    /// println!("Search completed in {}μs", explain.total_time_us);
    /// println!("HNSW traversal: {}μs ({} nodes visited)",
    ///          explain.index_time_us, explain.hnsw_stats.visited_nodes);
    /// println!("Effective k: {} (requested: {})", explain.effective_k, explain.requested_k);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_explain(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, SearchExplain)> {
        use std::time::Instant;

        let total_start = Instant::now();
        self.validate_query(query)?;

        let effective_k = self.clamp_k(k);
        if effective_k == 0 {
            let mut explain = SearchExplain {
                dimensions: self.config.dimensions,
                collection_size: self.len(),
                requested_k: k,
                effective_k: 0,
                ef_search: self.index.config().ef_search,
                distance_function: format!("{:?}", self.config.distance),
                ..Default::default()
            };
            explain.total_time_us = total_start.elapsed().as_micros() as u64;
            return Ok((Vec::new(), explain));
        }

        // HNSW index search with stats
        let index_start = Instant::now();
        let (raw_results, hnsw_stats) =
            self.index
                .search_with_stats(query, effective_k, self.vectors.as_slice());
        let index_time = index_start.elapsed();

        let candidates_before_filter = raw_results.len();

        // Enrich results with metadata
        let enrich_start = Instant::now();
        let results = self.enrich_results(raw_results)?;
        let enrich_time = enrich_start.elapsed();

        let total_time = total_start.elapsed();

        let explain = SearchExplain {
            total_time_us: total_time.as_micros() as u64,
            index_time_us: index_time.as_micros() as u64,
            filter_time_us: 0,
            enrich_time_us: enrich_time.as_micros() as u64,
            candidates_before_filter,
            candidates_after_filter: results.len(),
            hnsw_stats,
            dimensions: self.config.dimensions,
            collection_size: self.len(),
            requested_k: k,
            effective_k,
            ef_search: self.index.config().ef_search,
            filter_applied: false,
            distance_function: format!("{:?}", self.config.distance),
        };

        Ok((results, explain))
    }

    /// Search with metadata filter and detailed profiling.
    ///
    /// Combines filtered search with query execution profiling.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, Filter};
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"})))?;
    /// collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "b"})))?;
    ///
    /// let filter = Filter::eq("type", "a");
    /// let (results, explain) = collection.search_with_filter_explain(
    ///     &[1.0, 0.0, 0.0, 0.0],
    ///     10,
    ///     &filter
    /// )?;
    ///
    /// println!("Filter reduced {} -> {} candidates",
    ///          explain.candidates_before_filter,
    ///          explain.candidates_after_filter);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_with_filter_explain(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<(Vec<SearchResult>, SearchExplain)> {
        use std::time::Instant;

        let total_start = Instant::now();
        self.validate_query(query)?;

        let effective_k = self.clamp_k(k);
        if effective_k == 0 {
            let mut explain = SearchExplain {
                dimensions: self.config.dimensions,
                collection_size: self.len(),
                requested_k: k,
                effective_k: 0,
                ef_search: self.index.config().ef_search,
                filter_applied: true,
                distance_function: format!("{:?}", self.config.distance),
                ..Default::default()
            };
            explain.total_time_us = total_start.elapsed().as_micros() as u64;
            return Ok((Vec::new(), explain));
        }

        // HNSW index search with stats (fetch extra candidates for filtering)
        let index_start = Instant::now();
        let (candidates, hnsw_stats) = self.index.search_with_stats(
            query,
            effective_k * FILTER_CANDIDATE_MULTIPLIER,
            self.vectors.as_slice(),
        );
        let index_time = index_start.elapsed();

        let candidates_before_filter = candidates.len();

        // Apply filter
        let filter_start = Instant::now();
        let filtered: Vec<(crate::hnsw::VectorId, f32)> = candidates
            .into_iter()
            .filter(|(id, _)| {
                if let Some(entry) = self.metadata.get(*id) {
                    filter.matches(entry.data.as_ref())
                } else {
                    false
                }
            })
            .take(effective_k)
            .collect();
        let filter_time = filter_start.elapsed();

        let candidates_after_filter = filtered.len();

        // Enrich results with metadata
        let enrich_start = Instant::now();
        let results = self.enrich_results(filtered)?;
        let enrich_time = enrich_start.elapsed();

        let total_time = total_start.elapsed();

        let explain = SearchExplain {
            total_time_us: total_time.as_micros() as u64,
            index_time_us: index_time.as_micros() as u64,
            filter_time_us: filter_time.as_micros() as u64,
            enrich_time_us: enrich_time.as_micros() as u64,
            candidates_before_filter,
            candidates_after_filter,
            hnsw_stats,
            dimensions: self.config.dimensions,
            collection_size: self.len(),
            requested_k: k,
            effective_k,
            ef_search: self.index.config().ef_search,
            filter_applied: true,
            distance_function: format!("{:?}", self.config.distance),
        };

        Ok((results, explain))
    }

    /// Search and return only IDs (faster, no metadata lookup)
    pub fn search_ids(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(query, k, self.vectors.as_slice());

        results
            .into_iter()
            .map(|(id, distance)| {
                let entry = self
                    .metadata
                    .get(id)
                    .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;
                Ok((entry.external_id.clone(), distance))
            })
            .collect()
    }

    /// Search and return borrowed ID references (zero-copy, fastest)
    ///
    /// This method returns references to the stored IDs rather than cloning them,
    /// making it the most efficient option when you only need IDs and distances.
    ///
    /// # Example
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// # collection.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    /// let query = vec![1.0, 0.0, 0.0, 0.0];
    /// let results = collection.search_ids_ref(&query, 5).unwrap();
    /// for (id, distance) in results {
    ///     println!("{}: {}", id, distance);
    /// }
    /// ```
    pub fn search_ids_ref(&self, query: &[f32], k: usize) -> Result<Vec<(&str, f32)>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(query, k, self.vectors.as_slice());

        results
            .into_iter()
            .map(|(id, distance)| {
                let entry = self
                    .metadata
                    .get(id)
                    .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;
                Ok((entry.external_id.as_str(), distance))
            })
            .collect()
    }

    /// Batch search for multiple queries in parallel
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        use std::time::Instant;
        let start = Instant::now();

        // Validate all queries have correct dimensions and values
        for query in queries.iter() {
            self.validate_query(query)?;
        }
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        // Perform parallel search
        let results: Vec<Result<Vec<SearchResult>>> = queries
            .par_iter()
            .map(|query| {
                let raw_results = self.index.search(query, k, self.vectors.as_slice());
                self.enrich_results(raw_results)
            })
            .collect();

        // Collect results, propagating any errors
        let results: Vec<Vec<SearchResult>> = results.into_iter().collect::<Result<_>>()?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    batch_size = queries.len(),
                    collection_size = self.len(),
                    query_type = "batch",
                    "slow batch query detected"
                );
            }
        }

        Ok(results)
    }

    /// Batch search with metadata filter in parallel
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use std::time::Instant;
        let start = Instant::now();

        // Validate all queries have correct dimensions and values
        for query in queries.iter() {
            self.validate_query(query)?;
        }
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        // Perform parallel filtered search
        let results: Vec<Result<Vec<SearchResult>>> = queries
            .par_iter()
            .map(|query| {
                let candidates = self.index.search(
                    query,
                    k * FILTER_CANDIDATE_MULTIPLIER,
                    self.vectors.as_slice(),
                );
                let filtered: Vec<(VectorId, f32)> = candidates
                    .into_iter()
                    .filter(|(id, _)| {
                        if let Some(entry) = self.metadata.get(*id) {
                            filter.matches(entry.data.as_ref())
                        } else {
                            false
                        }
                    })
                    .take(k)
                    .collect();
                self.enrich_results(filtered)
            })
            .collect();

        let results: Vec<Vec<SearchResult>> = results.into_iter().collect::<Result<_>>()?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    batch_size = queries.len(),
                    collection_size = self.len(),
                    query_type = "batch",
                    filter_applied = true,
                    "slow batch query detected"
                );
            }
        }

        Ok(results)
    }

    /// Search for nearest neighbors with metadata filtering.
    ///
    /// Combines approximate nearest neighbor search with metadata-based
    /// pre-filtering. Only vectors matching the filter are returned.
    ///
    /// Internally, this fetches more candidates than requested (`k * FILTER_CANDIDATE_MULTIPLIER`)
    /// to compensate for filtered-out results, then filters and returns
    /// the top `k` matches.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    /// * `filter` - MongoDB-style filter for metadata (see [`Filter`])
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, Filter};
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("docs", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "article"})))?;
    /// collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "image"})))?;
    ///
    /// // Search only for articles
    /// let filter = Filter::eq("type", "article");
    /// let results = collection.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter)?;
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].id, "v1");
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        // For filtered search, we need to retrieve more candidates and filter
        let candidates = self.index.search(
            query,
            k * FILTER_CANDIDATE_MULTIPLIER,
            self.vectors.as_slice(),
        );

        let filtered: Vec<(VectorId, f32)> = candidates
            .into_iter()
            .filter(|(id, _)| {
                if let Some(entry) = self.metadata.get(*id) {
                    filter.matches(entry.data.as_ref())
                } else {
                    false
                }
            })
            .take(k)
            .collect();

        let results = self.enrich_results(filtered)?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    results_count = results.len(),
                    collection_size = self.len(),
                    filter_applied = true,
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Find all vectors within a given distance from the query.
    ///
    /// Unlike top-k search which returns exactly k results regardless of distance,
    /// range queries return all vectors within `max_distance` from the query. This is
    /// useful for clustering, deduplication, or finding all semantically similar items.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return (caps output size)
    ///
    /// # Returns
    ///
    /// All vectors with `distance <= max_distance`, up to `limit` results,
    /// sorted by distance ascending.
    ///
    /// # Performance
    ///
    /// This method uses HNSW search internally with `limit` as the search bound.
    /// For very large ranges or small collections, consider using exact search instead.
    /// The method benefits from early termination when all nearby candidates exceed
    /// the distance threshold.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("embeddings", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// collection.insert("v2", &[0.99, 0.1, 0.0, 0.0], None)?;  // Close to v1
    /// collection.insert("v3", &[0.0, 1.0, 0.0, 0.0], None)?;   // Far from v1
    ///
    /// // Find all vectors within distance 0.2 from the query
    /// let results = collection.search_radius(&[1.0, 0.0, 0.0, 0.0], 0.2, 100)?;
    ///
    /// // Only nearby vectors are returned
    /// for r in &results {
    ///     assert!(r.distance <= 0.2);
    /// }
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_radius(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;

        if limit == 0 || max_distance < 0.0 {
            return Ok(Vec::new());
        }

        // Search for up to limit candidates
        let candidates = self.index.search(query, limit, self.vectors.as_slice());

        // Filter to only include results within max_distance
        let within_range: Vec<(VectorId, f32)> = candidates
            .into_iter()
            .take_while(|(_, distance)| *distance <= max_distance)
            .collect();

        let results = self.enrich_results(within_range)?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    limit = limit,
                    max_distance = max_distance,
                    results_count = results.len(),
                    collection_size = self.len(),
                    query_type = "radius",
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Find all vectors within a given distance with metadata filtering.
    ///
    /// Combines range queries with metadata pre-filtering. Only vectors that
    /// both match the filter AND are within `max_distance` are returned.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return
    /// * `filter` - MongoDB-style filter for metadata
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, Filter};
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("products", 4);
    /// collection.insert("p1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"category": "electronics"})))?;
    /// collection.insert("p2", &[0.99, 0.1, 0.0, 0.0], Some(json!({"category": "books"})))?;
    ///
    /// // Find electronics within distance 0.5
    /// let filter = Filter::eq("category", "electronics");
    /// let results = collection.search_radius_with_filter(&[1.0, 0.0, 0.0, 0.0], 0.5, 100, &filter)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_radius_with_filter(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;

        if limit == 0 || max_distance < 0.0 {
            return Ok(Vec::new());
        }

        // Over-fetch to compensate for filtering
        let fetch_count = limit * FILTER_CANDIDATE_MULTIPLIER;
        let candidates = self
            .index
            .search(query, fetch_count, self.vectors.as_slice());

        // Filter by distance first (can stop early), then by metadata
        let within_range: Vec<(VectorId, f32)> = candidates
            .into_iter()
            .take_while(|(_, distance)| *distance <= max_distance)
            .filter(|(id, _)| {
                if let Some(entry) = self.metadata.get(*id) {
                    filter.matches(entry.data.as_ref())
                } else {
                    false
                }
            })
            .take(limit)
            .collect();

        let results = self.enrich_results(within_range)?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    limit = limit,
                    max_distance = max_distance,
                    results_count = results.len(),
                    collection_size = self.len(),
                    query_type = "radius",
                    filter_applied = true,
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Convert internal results to SearchResult with metadata
    fn enrich_results(&self, results: Vec<(VectorId, f32)>) -> Result<Vec<SearchResult>> {
        results
            .into_iter()
            .map(|(id, distance)| {
                let entry = self
                    .metadata
                    .get(id)
                    .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;

                Ok(SearchResult {
                    id: entry.external_id.clone(),
                    distance,
                    metadata: entry.data.clone(),
                })
            })
            .collect()
    }

    /// Retrieve a vector and its metadata by ID.
    ///
    /// Returns a reference to the vector data and optional metadata if found.
    ///
    /// # Arguments
    ///
    /// * `id` - The external ID of the vector to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some((vector, metadata))` if the vector exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"name": "test"})))?;
    ///
    /// if let Some((vector, metadata)) = collection.get("v1") {
    ///     assert_eq!(vector, &[1.0, 2.0, 3.0, 4.0]);
    ///     assert!(metadata.is_some());
    /// }
    ///
    /// assert!(collection.get("nonexistent").is_none());
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn get(&self, id: &str) -> Option<(&[f32], Option<&Value>)> {
        let internal_id = self.metadata.get_internal_id(id)?;
        let vector = self.vectors.get(internal_id)?;
        let metadata = self.metadata.get(internal_id).and_then(|e| e.data.as_ref());
        Some((vector, metadata))
    }

    /// Check if a vector with the given ID exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(collection.contains("v1"));
    /// assert!(!collection.contains("v2"));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn contains(&self, id: &str) -> bool {
        self.metadata.contains(id)
    }

    /// Delete a vector by its ID.
    ///
    /// Removes the vector from the index, but does not immediately reclaim
    /// storage space. Call [`compact()`](Self::compact) after many deletions
    /// to reclaim space.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the vector was deleted, `Ok(false)` if no vector
    /// with that ID existed.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(collection.delete("v1")?);
    /// assert!(!collection.delete("v1")?); // Already deleted
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let internal_id = match self.metadata.get_internal_id(id) {
            Some(id) => id,
            None => return Ok(false),
        };

        self.metadata.delete(internal_id);
        self.index.delete(internal_id)?;

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(true)
    }

    /// Delete multiple vectors by their external IDs
    /// Returns the number of vectors actually deleted
    pub fn delete_batch(&mut self, ids: &[impl AsRef<str>]) -> Result<usize> {
        let mut deleted = 0;
        for id in ids {
            if self.delete(id.as_ref())? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Update an existing vector and its metadata.
    ///
    /// Replaces both the vector data and metadata for an existing vector.
    /// The vector is re-indexed after the update.
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the vector to update
    /// * `vector` - New vector data (must match collection dimensions)
    /// * `metadata` - New metadata (replaces existing metadata)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::VectorNotFound`] - No vector with the given ID exists
    /// - [`NeedleError::DimensionMismatch`] - Vector dimensions don't match
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"version": 1})))?;
    ///
    /// // Update the vector and metadata
    /// collection.update("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"version": 2})))?;
    ///
    /// let (vec, meta) = collection.get("v1").unwrap();
    /// assert_eq!(vec, &[0.0, 1.0, 0.0, 0.0]);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn update(&mut self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        // Check dimensions
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Get internal ID
        let internal_id = self
            .metadata
            .get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        // Update vector in storage
        self.vectors.update(internal_id, vector.to_vec())?;

        // Update metadata
        self.metadata.update_data(internal_id, metadata)?;

        // Re-index the vector (delete and re-insert in index)
        self.index.delete(internal_id)?;
        self.index
            .insert(internal_id, vector, self.vectors.as_slice())?;

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(())
    }

    /// Update only the metadata for an existing vector
    /// Returns error if the vector doesn't exist
    pub fn update_metadata(&mut self, id: &str, metadata: Option<Value>) -> Result<()> {
        let internal_id = self
            .metadata
            .get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        self.metadata.update_data(internal_id, metadata)?;

        // Invalidate cache since metadata is part of search results
        self.invalidate_cache();

        Ok(())
    }

    /// Insert a vector, or update it if it already exists
    pub fn upsert(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<bool> {
        let id = id.into();

        if self.contains(&id) {
            self.update(&id, vector, metadata)?;
            Ok(false) // Updated existing
        } else {
            self.insert(id, vector, metadata)?;
            Ok(true) // Inserted new
        }
    }

    /// Get collection statistics
    pub fn stats(&self) -> CollectionStats {
        let vector_count = self.vectors.len();
        let dimensions = self.config.dimensions;

        // Estimate memory usage
        let vector_memory = vector_count * dimensions * std::mem::size_of::<f32>();
        let metadata_memory = self.metadata.estimated_memory();
        let index_memory = self.index.estimated_memory();

        CollectionStats {
            name: self.config.name.clone(),
            vector_count,
            dimensions,
            distance_function: self.config.distance,
            vector_memory_bytes: vector_memory,
            metadata_memory_bytes: metadata_memory,
            index_memory_bytes: index_memory,
            total_memory_bytes: vector_memory + metadata_memory + index_memory,
            index_stats: self.index.stats(),
        }
    }

    /// Count vectors matching an optional filter
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        match filter {
            None => self.len(),
            Some(f) => self
                .metadata
                .iter()
                .filter(|(internal_id, entry)| {
                    !self.index.is_deleted(*internal_id) && f.matches(entry.data.as_ref())
                })
                .count(),
        }
    }

    /// Get the number of deleted vectors pending compaction
    pub fn deleted_count(&self) -> usize {
        self.index.deleted_count()
    }

    /// Iterate over all vectors in the collection
    /// Returns an iterator of (external_id, vector, metadata)
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f32], Option<&Value>)> {
        self.metadata
            .iter()
            .filter_map(move |(internal_id, entry)| {
                // Skip deleted vectors
                if self.index.is_deleted(internal_id) {
                    return None;
                }
                let vector = self.vectors.get(internal_id)?;
                Some((
                    entry.external_id.as_str(),
                    vector.as_slice(),
                    entry.data.as_ref(),
                ))
            })
    }

    /// Get all vector IDs in the collection
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.metadata
            .iter()
            .filter(move |(internal_id, _)| !self.index.is_deleted(*internal_id))
            .map(|(_, entry)| entry.external_id.as_str())
    }

    /// Compact the collection by removing deleted and expired vectors
    /// This rebuilds the index and reclaims storage
    /// Returns the number of vectors removed
    pub fn compact(&mut self) -> Result<usize> {
        // First, expire any TTL'd vectors
        let expired_count = self.expire_vectors()?;

        let deleted_count = self.index.deleted_count();

        if deleted_count == 0 {
            return Ok(expired_count);
        }

        // Get the ID mapping from compacting the index
        let id_map = self.index.compact(self.vectors.as_slice());

        // Rebuild vectors and metadata with new IDs
        let mut new_vectors = VectorStore::new(self.config.dimensions);
        let mut new_metadata = MetadataStore::new();
        let mut new_expirations = HashMap::new();

        // Sort by new ID to maintain order
        let mut mappings: Vec<_> = id_map.into_iter().collect();
        mappings.sort_by_key(|(_, new_id)| *new_id);

        for (old_id, new_id) in mappings {
            if let Some(vector) = self.vectors.get(old_id) {
                let added_id = new_vectors.add(vector.clone())?;
                debug_assert_eq!(added_id, new_id);

                if let Some(entry) = self.metadata.get(old_id) {
                    new_metadata.insert(new_id, entry.external_id.clone(), entry.data.clone())?;
                }

                // Remap expiration entry if it exists
                if let Some(expiration) = self.expirations.get(&old_id) {
                    new_expirations.insert(new_id, *expiration);
                }
            }
        }

        self.vectors = new_vectors;
        self.metadata = new_metadata;
        self.expirations = new_expirations;

        // Invalidate cache since internal IDs changed
        self.invalidate_cache();

        Ok(deleted_count + expired_count)
    }

    /// Check if the collection needs compaction
    /// Returns true if deleted vectors exceed the given threshold (0.0-1.0)
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.index.needs_compaction(threshold)
    }

    // ============ TTL/Expiration Methods ============

    /// Get the current Unix timestamp in seconds
    #[inline]
    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Check if a vector has expired based on its internal ID
    #[inline]
    fn is_expired(&self, internal_id: usize) -> bool {
        if let Some(&expiration) = self.expirations.get(&internal_id) {
            Self::now_unix() >= expiration
        } else {
            false
        }
    }

    /// Sweep and delete all expired vectors.
    ///
    /// This is the "eager" expiration strategy. Call this periodically to
    /// remove expired vectors and reclaim storage space.
    ///
    /// # Returns
    ///
    /// The number of vectors that were expired and deleted.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, CollectionConfig};
    ///
    /// let config = CollectionConfig::new("ephemeral", 4)
    ///     .with_default_ttl_seconds(1); // 1 second TTL
    /// let mut collection = Collection::new(config);
    ///
    /// collection.insert("temp", &[0.1, 0.2, 0.3, 0.4], None)?;
    ///
    /// // Wait for expiration...
    /// std::thread::sleep(std::time::Duration::from_secs(2));
    ///
    /// let expired = collection.expire_vectors()?;
    /// assert_eq!(expired, 1);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn expire_vectors(&mut self) -> Result<usize> {
        let now = Self::now_unix();
        let mut expired_ids = Vec::new();

        // Find all expired vectors
        for (&internal_id, &expiration) in &self.expirations {
            if now >= expiration {
                expired_ids.push(internal_id);
            }
        }

        // Delete each expired vector
        for internal_id in &expired_ids {
            // Mark as deleted in index
            self.index.delete(*internal_id)?;
            // Remove metadata
            self.metadata.delete(*internal_id);
            // Remove from expirations tracking
            self.expirations.remove(internal_id);
        }

        if !expired_ids.is_empty() {
            self.invalidate_cache();
        }

        Ok(expired_ids.len())
    }

    /// Check if an expiration sweep is needed based on a threshold.
    ///
    /// Returns true if the ratio of expired vectors to total vectors
    /// exceeds the given threshold (0.0-1.0).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Ratio threshold (e.g., 0.1 = 10% expired)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let collection = Collection::with_dimensions("test", 4);
    ///
    /// // Check if more than 10% of vectors are expired
    /// if collection.needs_expiration_sweep(0.1) {
    ///     // Run sweep...
    /// }
    /// ```
    pub fn needs_expiration_sweep(&self, threshold: f64) -> bool {
        if self.expirations.is_empty() {
            return false;
        }

        let now = Self::now_unix();
        let expired_count = self.expirations.values().filter(|&&exp| now >= exp).count();
        let total = self.len();

        if total == 0 {
            return expired_count > 0;
        }

        (expired_count as f64 / total as f64) > threshold
    }

    /// Get TTL statistics for the collection.
    ///
    /// Returns a tuple of (total_with_ttl, expired_count, earliest_expiration, latest_expiration).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let collection = Collection::with_dimensions("test", 4);
    /// let (total, expired, earliest, latest) = collection.ttl_stats();
    /// println!("TTL vectors: {}, expired: {}", total, expired);
    /// ```
    pub fn ttl_stats(&self) -> (usize, usize, Option<u64>, Option<u64>) {
        let now = Self::now_unix();
        let total = self.expirations.len();
        let expired = self.expirations.values().filter(|&&exp| now >= exp).count();
        let earliest = self.expirations.values().copied().min();
        let latest = self.expirations.values().copied().max();

        (total, expired, earliest, latest)
    }

    /// Get the expiration timestamp for a vector by external ID.
    ///
    /// Returns `None` if the vector doesn't exist or has no TTL set.
    pub fn get_ttl(&self, id: &str) -> Option<u64> {
        let internal_id = self.metadata.get_internal_id(id)?;
        self.expirations.get(&internal_id).copied()
    }

    /// Set or update the TTL for an existing vector.
    ///
    /// # Arguments
    ///
    /// * `id` - External vector ID
    /// * `ttl_seconds` - TTL in seconds from now, or `None` to remove TTL
    ///
    /// # Errors
    ///
    /// Returns an error if the vector doesn't exist.
    pub fn set_ttl(&mut self, id: &str, ttl_seconds: Option<u64>) -> Result<()> {
        let internal_id = self
            .metadata
            .get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        match ttl_seconds {
            Some(ttl) => {
                let expiration = Self::now_unix() + ttl;
                self.expirations.insert(internal_id, expiration);
            }
            None => {
                self.expirations.remove(&internal_id);
            }
        }

        Ok(())
    }

    /// Serialize the collection to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize a collection from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Iterate over vectors matching a filter
    ///
    /// Returns an iterator yielding (id, vector, metadata) tuples
    /// that match the given filter
    pub fn iter_filtered<'a>(
        &'a self,
        filter: &'a Filter,
    ) -> impl Iterator<Item = (&'a str, &'a [f32], Option<&'a Value>)> + 'a {
        self.iter()
            .filter(move |(_, _, meta)| filter.matches(*meta))
    }

    /// Get all vector IDs as a collected Vec
    pub fn all_ids(&self) -> Vec<String> {
        self.metadata.all_external_ids()
    }

    /// Estimate if a dataset of given size would fit in memory
    pub fn estimate_memory(
        vector_count: usize,
        dimensions: usize,
        avg_metadata_bytes: usize,
    ) -> usize {
        let vector_bytes = vector_count * dimensions * std::mem::size_of::<f32>();
        let metadata_bytes = vector_count * avg_metadata_bytes;
        let index_overhead = vector_count * 200; // ~200 bytes per vector for HNSW
        vector_bytes + metadata_bytes + index_overhead
    }
}

/// Iterator over collection entries.
///
/// Yields `(id, vector, metadata)` tuples for each vector in the collection.
/// Deleted vectors are automatically skipped.
pub struct CollectionIter<'a> {
    collection: &'a Collection,
    ids: Vec<String>,
    index: usize,
}

impl<'a> CollectionIter<'a> {
    fn new(collection: &'a Collection) -> Self {
        let ids = collection.all_ids();
        Self {
            collection,
            ids,
            index: 0,
        }
    }
}

impl<'a> Iterator for CollectionIter<'a> {
    type Item = (String, Vec<f32>, Option<Value>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.ids.len() {
            let id = &self.ids[self.index];
            self.index += 1;

            if let Some((vector, metadata)) = self.collection.get(id) {
                return Some((id.clone(), vector.to_vec(), metadata.cloned()));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ids.len().saturating_sub(self.index);
        (0, Some(remaining))
    }
}

impl<'a> IntoIterator for &'a Collection {
    type Item = (String, Vec<f32>, Option<Value>);
    type IntoIter = CollectionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CollectionIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_collection_basic() {
        let mut collection = Collection::with_dimensions("test", 128);

        assert_eq!(collection.name(), "test");
        assert_eq!(collection.dimensions(), 128);
        assert!(collection.is_empty());

        // Insert a vector
        let vec = random_vector(128);
        collection
            .insert("doc1", &vec, Some(json!({"title": "Hello"})))
            .unwrap();

        assert_eq!(collection.len(), 1);
        assert!(!collection.is_empty());
        assert!(collection.contains("doc1"));
    }

    #[test]
    fn test_collection_search() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Insert some vectors
        for i in 0..100 {
            let vec = random_vector(32);
            collection
                .insert(format!("doc{}", i), &vec, Some(json!({"index": i})))
                .unwrap();
        }

        // Search
        let query = random_vector(32);
        let results = collection.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }

    #[test]
    fn test_collection_search_with_filter() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Insert vectors with category metadata
        for i in 0..100 {
            let vec = random_vector(32);
            let category = if i % 2 == 0 { "even" } else { "odd" };
            collection
                .insert(
                    format!("doc{}", i),
                    &vec,
                    Some(json!({"index": i, "category": category})),
                )
                .unwrap();
        }

        // Search with filter
        let query = random_vector(32);
        let filter = Filter::eq("category", "even");
        let results = collection.search_with_filter(&query, 10, &filter).unwrap();

        // All results should have category "even"
        for result in &results {
            let meta = result.metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "even");
        }
    }

    #[test]
    fn test_collection_get_and_delete() {
        let mut collection = Collection::with_dimensions("test", 32);

        let vec = random_vector(32);
        collection
            .insert("doc1", &vec, Some(json!({"title": "Test"})))
            .unwrap();

        // Get
        let (retrieved_vec, metadata) = collection.get("doc1").unwrap();
        assert_eq!(retrieved_vec, vec.as_slice());
        assert_eq!(metadata.unwrap()["title"], "Test");

        // Delete
        assert!(collection.delete("doc1").unwrap());
        assert!(!collection.contains("doc1"));
        assert!(collection.get("doc1").is_none());
    }

    #[test]
    fn test_collection_dimension_mismatch() {
        let mut collection = Collection::with_dimensions("test", 32);

        let wrong_dim_vec = random_vector(64);
        let result = collection.insert("doc1", &wrong_dim_vec, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_collection_duplicate_id() {
        let mut collection = Collection::with_dimensions("test", 32);

        let vec1 = random_vector(32);
        let vec2 = random_vector(32);

        collection.insert("doc1", &vec1, None).unwrap();
        let result = collection.insert("doc1", &vec2, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_collection_serialization() {
        let mut collection = Collection::with_dimensions("test", 32);

        for i in 0..10 {
            let vec = random_vector(32);
            collection
                .insert(format!("doc{}", i), &vec, Some(json!({"i": i})))
                .unwrap();
        }

        // Serialize
        let bytes = collection.to_bytes().unwrap();

        // Deserialize
        let restored = Collection::from_bytes(&bytes).unwrap();

        assert_eq!(collection.name(), restored.name());
        assert_eq!(collection.dimensions(), restored.dimensions());
        assert_eq!(collection.len(), restored.len());
    }

    #[test]
    fn test_batch_search() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Insert some vectors
        for i in 0..100 {
            let vec = random_vector(32);
            collection
                .insert(format!("doc{}", i), &vec, Some(json!({"index": i})))
                .unwrap();
        }

        // Batch search with multiple queries
        let queries: Vec<Vec<f32>> = (0..5).map(|_| random_vector(32)).collect();
        let results = collection.batch_search(&queries, 10).unwrap();

        assert_eq!(results.len(), 5);
        for result_set in &results {
            assert_eq!(result_set.len(), 10);
            // Results should be sorted by distance
            for i in 1..result_set.len() {
                assert!(result_set[i].distance >= result_set[i - 1].distance);
            }
        }
    }

    #[test]
    fn test_batch_search_with_filter() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Insert vectors with category metadata
        for i in 0..100 {
            let vec = random_vector(32);
            let category = if i % 2 == 0 { "even" } else { "odd" };
            collection
                .insert(
                    format!("doc{}", i),
                    &vec,
                    Some(json!({"index": i, "category": category})),
                )
                .unwrap();
        }

        // Batch search with filter
        let queries: Vec<Vec<f32>> = (0..3).map(|_| random_vector(32)).collect();
        let filter = Filter::eq("category", "even");
        let results = collection
            .batch_search_with_filter(&queries, 5, &filter)
            .unwrap();

        assert_eq!(results.len(), 3);
        for result_set in &results {
            // All results should have category "even"
            for result in result_set {
                let meta = result.metadata.as_ref().unwrap();
                assert_eq!(meta["category"], "even");
            }
        }
    }

    #[test]
    fn test_search_builder_post_filter() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Insert vectors with numeric scores
        for i in 0..100 {
            let vec = random_vector(32);
            collection
                .insert(
                    format!("doc{}", i),
                    &vec,
                    Some(
                        json!({"score": i, "type": if i % 3 == 0 { "special" } else { "normal" }}),
                    ),
                )
                .unwrap();
        }

        let query = random_vector(32);

        // Post-filter: only keep results with score > 50
        let score_filter = Filter::gt("score", 50);
        let results = collection
            .search_builder(&query)
            .k(10)
            .post_filter(&score_filter)
            .execute()
            .unwrap();

        // All results should have score > 50
        for result in &results {
            let meta = result.metadata.as_ref().unwrap();
            let score = meta["score"].as_i64().unwrap();
            assert!(score > 50, "Score {} should be > 50", score);
        }

        // Combined: pre-filter + post-filter
        let type_filter = Filter::eq("type", "special"); // Pre-filter for type
        let high_score_filter = Filter::gt("score", 30); // Post-filter for score

        let combined_results = collection
            .search_builder(&query)
            .k(5)
            .filter(&type_filter) // Pre-filter
            .post_filter(&high_score_filter) // Post-filter
            .execute()
            .unwrap();

        // Results should match both filters
        for result in &combined_results {
            let meta = result.metadata.as_ref().unwrap();
            assert_eq!(meta["type"], "special");
            let score = meta["score"].as_i64().unwrap();
            assert!(score > 30, "Score {} should be > 30", score);
        }
    }

    #[test]
    fn test_search_builder_post_filter_factor() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Insert vectors with sparse matching criteria (only 10% pass)
        for i in 0..100 {
            let vec = random_vector(32);
            collection
                .insert(
                    format!("doc{}", i),
                    &vec,
                    Some(json!({"rare": i % 10 == 0})),
                )
                .unwrap();
        }

        let query = random_vector(32);
        let rare_filter = Filter::eq("rare", true);

        // With higher post_filter_factor, we should get more results
        let results = collection
            .search_builder(&query)
            .k(5)
            .post_filter(&rare_filter)
            .post_filter_factor(10) // Fetch 50 candidates to find 5 rare ones
            .execute()
            .unwrap();

        // All results should have rare=true
        for result in &results {
            let meta = result.metadata.as_ref().unwrap();
            assert_eq!(meta["rare"], true);
        }
    }

    #[test]
    fn test_search_builder_post_filter_no_metadata() {
        let mut collection = Collection::with_dimensions("test", 32);

        for i in 0..50 {
            let vec = random_vector(32);
            collection
                .insert(format!("doc{}", i), &vec, Some(json!({"keep": i < 25})))
                .unwrap();
        }

        let query = random_vector(32);
        let keep_filter = Filter::eq("keep", true);

        // Post-filter with include_metadata=false should strip metadata after filtering
        let results = collection
            .search_builder(&query)
            .k(10)
            .post_filter(&keep_filter)
            .include_metadata(false)
            .execute()
            .unwrap();

        // Results should have no metadata
        for result in &results {
            assert!(result.metadata.is_none());
        }
    }

    #[test]
    fn test_search_radius() {
        use crate::DistanceFunction;

        // Use Euclidean distance for predictable distance values
        let config = CollectionConfig::new("test", 4).with_distance(DistanceFunction::Euclidean);
        let mut collection = Collection::new(config);

        // Insert vectors at known positions
        collection
            .insert("origin", &[0.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        collection
            .insert("close", &[0.1, 0.0, 0.0, 0.0], None)
            .unwrap(); // dist = 0.1
        collection
            .insert("medium", &[0.5, 0.0, 0.0, 0.0], None)
            .unwrap(); // dist = 0.5
        collection
            .insert("far", &[2.0, 0.0, 0.0, 0.0], None)
            .unwrap(); // dist = 2.0

        // Query at origin
        let query = [0.0, 0.0, 0.0, 0.0];

        // Search within radius 0.15 - should only find origin and close
        let results = collection.search_radius(&query, 0.15, 100).unwrap();
        assert!(!results.is_empty() && results.len() <= 2);
        for r in &results {
            assert!(
                r.distance <= 0.15,
                "Distance {} exceeds max 0.15",
                r.distance
            );
        }

        // Search within radius 0.6 - should find origin, close, and medium
        let results = collection.search_radius(&query, 0.6, 100).unwrap();
        assert!(results.len() >= 2 && results.len() <= 3);
        for r in &results {
            assert!(r.distance <= 0.6, "Distance {} exceeds max 0.6", r.distance);
        }

        // Search within radius 3.0 - should find all
        let results = collection.search_radius(&query, 3.0, 100).unwrap();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.distance <= 3.0);
        }

        // Edge case: zero limit
        let results = collection.search_radius(&query, 1.0, 0).unwrap();
        assert!(results.is_empty());

        // Edge case: negative max_distance
        let results = collection.search_radius(&query, -1.0, 100).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_radius_with_filter() {
        use crate::DistanceFunction;

        let config = CollectionConfig::new("test", 4).with_distance(DistanceFunction::Euclidean);
        let mut collection = Collection::new(config);

        // Insert vectors with category
        collection
            .insert("a1", &[0.1, 0.0, 0.0, 0.0], Some(json!({"type": "A"})))
            .unwrap();
        collection
            .insert("a2", &[0.2, 0.0, 0.0, 0.0], Some(json!({"type": "A"})))
            .unwrap();
        collection
            .insert("b1", &[0.15, 0.0, 0.0, 0.0], Some(json!({"type": "B"})))
            .unwrap();
        collection
            .insert("far", &[5.0, 0.0, 0.0, 0.0], Some(json!({"type": "A"})))
            .unwrap();

        let query = [0.0, 0.0, 0.0, 0.0];
        let filter = Filter::eq("type", "A");

        // Search within radius 0.3 for type A only
        let results = collection
            .search_radius_with_filter(&query, 0.3, 100, &filter)
            .unwrap();

        // Should find a1 and a2 (type A, within 0.3), but not b1 (type B) or far (beyond 0.3)
        assert!(!results.is_empty() && results.len() <= 2);
        for r in &results {
            assert!(r.distance <= 0.3);
            let meta = r.metadata.as_ref().unwrap();
            assert_eq!(meta["type"], "A");
        }
    }

    #[test]
    fn test_slow_query_threshold_config() {
        // Test configuration via CollectionConfig builder
        let config = CollectionConfig::new("test", 64).with_slow_query_threshold_us(100_000); // 100ms threshold

        assert_eq!(config.slow_query_threshold_us, Some(100_000));

        let collection = Collection::new(config);
        assert_eq!(collection.slow_query_threshold_us(), Some(100_000));
    }

    #[test]
    fn test_slow_query_threshold_runtime() {
        let mut collection = Collection::with_dimensions("test", 64);

        // Initially no threshold
        assert!(collection.slow_query_threshold_us().is_none());

        // Set threshold at runtime
        collection.set_slow_query_threshold_us(Some(50_000));
        assert_eq!(collection.slow_query_threshold_us(), Some(50_000));

        // Update threshold
        collection.set_slow_query_threshold_us(Some(25_000));
        assert_eq!(collection.slow_query_threshold_us(), Some(25_000));

        // Disable threshold
        collection.set_slow_query_threshold_us(None);
        assert!(collection.slow_query_threshold_us().is_none());
    }

    #[test]
    fn test_slow_query_threshold_serialization() {
        let config = CollectionConfig::new("test", 64).with_slow_query_threshold_us(75_000);
        let mut collection = Collection::new(config);

        // Insert some data
        for i in 0..10 {
            collection
                .insert(format!("v{}", i), &random_vector(64), None)
                .unwrap();
        }

        // Serialize and deserialize
        let serialized = serde_json::to_string(&collection).unwrap();
        let deserialized: Collection = serde_json::from_str(&serialized).unwrap();

        // Threshold should be preserved
        assert_eq!(deserialized.slow_query_threshold_us(), Some(75_000));
    }

    #[test]
    fn test_query_cache_config() {
        // Test configuration via CollectionConfig builder
        let config = CollectionConfig::new("test", 64).with_query_cache_capacity(100);

        assert!(config.query_cache.is_enabled());
        assert_eq!(config.query_cache.capacity, 100);

        let collection = Collection::new(config);
        assert!(collection.is_query_cache_enabled());
    }

    #[test]
    fn test_query_cache_disabled_by_default() {
        let collection = Collection::with_dimensions("test", 64);
        assert!(!collection.is_query_cache_enabled());
        assert!(collection.query_cache_stats().is_none());
    }

    #[test]
    fn test_query_cache_hit_miss() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert some vectors
        for i in 0..10 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // First search - cache miss
        let results1 = collection.search(&query, 5).unwrap();

        let stats = collection.query_cache_stats().unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.size, 1); // One entry cached

        // Second search with same query - cache hit
        let results2 = collection.search(&query, 5).unwrap();

        let stats = collection.query_cache_stats().unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);

        // Results should be identical
        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.id, r2.id);
            assert_eq!(r1.distance, r2.distance);
        }
    }

    #[test]
    fn test_query_cache_different_k() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert some vectors
        for i in 0..10 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // Search with k=5
        let _ = collection.search(&query, 5).unwrap();

        // Search with k=3 (different k, should miss)
        let _ = collection.search(&query, 3).unwrap();

        let stats = collection.query_cache_stats().unwrap();
        assert_eq!(stats.misses, 2); // Both were misses due to different k
        assert_eq!(stats.size, 2);
    }

    #[test]
    fn test_query_cache_invalidation_on_insert() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert initial vectors
        for i in 0..5 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // Cache the query
        let _ = collection.search(&query, 5).unwrap();
        assert_eq!(collection.query_cache_stats().unwrap().size, 1);

        // Insert a new vector - should invalidate cache
        collection.insert("new", &random_vector(32), None).unwrap();
        assert_eq!(collection.query_cache_stats().unwrap().size, 0);
    }

    #[test]
    fn test_query_cache_invalidation_on_delete() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // Cache the query
        let _ = collection.search(&query, 5).unwrap();
        assert_eq!(collection.query_cache_stats().unwrap().size, 1);

        // Delete a vector - should invalidate cache
        collection.delete("v0").unwrap();
        assert_eq!(collection.query_cache_stats().unwrap().size, 0);
    }

    #[test]
    fn test_query_cache_invalidation_on_update() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // Cache the query
        let _ = collection.search(&query, 5).unwrap();
        assert_eq!(collection.query_cache_stats().unwrap().size, 1);

        // Update a vector - should invalidate cache
        collection.update("v0", &random_vector(32), None).unwrap();
        assert_eq!(collection.query_cache_stats().unwrap().size, 0);
    }

    #[test]
    fn test_query_cache_enable_disable_runtime() {
        let mut collection = Collection::with_dimensions("test", 32);

        // Initially disabled
        assert!(!collection.is_query_cache_enabled());

        // Enable at runtime
        collection.enable_query_cache(100);
        assert!(collection.is_query_cache_enabled());

        // Insert and search
        for i in 0..5 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }
        let query = random_vector(32);
        let _ = collection.search(&query, 5).unwrap();

        let stats = collection.query_cache_stats().unwrap();
        assert_eq!(stats.capacity, 100);

        // Disable
        collection.disable_query_cache();
        assert!(!collection.is_query_cache_enabled());
        assert!(collection.query_cache_stats().is_none());
    }

    #[test]
    fn test_query_cache_clear() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        // Cache some queries
        for _ in 0..10 {
            let query = random_vector(32);
            let _ = collection.search(&query, 5).unwrap();
        }

        assert!(collection.query_cache_stats().unwrap().size > 0);

        // Clear cache
        collection.clear_query_cache();
        assert_eq!(collection.query_cache_stats().unwrap().size, 0);

        // Cache should still be enabled
        assert!(collection.is_query_cache_enabled());
    }

    #[test]
    fn test_query_cache_hit_ratio() {
        let config = CollectionConfig::new("test", 32).with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // 1 miss
        let _ = collection.search(&query, 5).unwrap();

        // 4 hits
        for _ in 0..4 {
            let _ = collection.search(&query, 5).unwrap();
        }

        let stats = collection.query_cache_stats().unwrap();
        assert_eq!(stats.hits, 4);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio() - 0.8).abs() < 0.001);
    }

    // ========== TTL Tests ==========

    #[test]
    fn test_ttl_insert_with_ttl() {
        let mut collection = Collection::new(CollectionConfig::new("test", 32));

        let vec = random_vector(32);
        collection
            .insert_with_ttl("doc1", &vec, None, Some(3600))
            .unwrap();

        // Vector should exist
        assert!(collection.get("doc1").is_some());

        // Check TTL stats - returns (total_with_ttl, expired, earliest, latest)
        let (total_with_ttl, _expired, _earliest, _latest) = collection.ttl_stats();
        assert_eq!(total_with_ttl, 1);
    }

    #[test]
    fn test_ttl_get_and_set_ttl() {
        let mut collection = Collection::new(CollectionConfig::new("test", 32));

        let vec = random_vector(32);
        collection.insert("doc1", &vec, None).unwrap();

        // Initially no TTL
        assert!(collection.get_ttl("doc1").is_none());

        // Set TTL
        collection.set_ttl("doc1", Some(3600)).unwrap();
        let ttl = collection.get_ttl("doc1");
        assert!(ttl.is_some());

        // Remove TTL
        collection.set_ttl("doc1", None).unwrap();
        assert!(collection.get_ttl("doc1").is_none());
    }

    #[test]
    fn test_ttl_expire_vectors() {
        let mut collection = Collection::new(CollectionConfig::new("test", 32));

        // Insert with TTL of 0 (immediately expires)
        let vec = random_vector(32);
        collection
            .insert_with_ttl("doc1", &vec, None, Some(0))
            .unwrap();

        // Wait a tiny bit for expiration to kick in
        std::thread::sleep(std::time::Duration::from_millis(10));

        let expired = collection.expire_vectors().unwrap();
        assert_eq!(expired, 1);

        // Vector should be gone
        assert!(collection.get("doc1").is_none());
    }

    #[test]
    fn test_ttl_needs_expiration_sweep() {
        let mut collection = Collection::new(CollectionConfig::new("test", 32));

        // Insert many vectors with TTL
        for i in 0..100 {
            let vec = random_vector(32);
            collection
                .insert_with_ttl(format!("doc{}", i), &vec, None, Some(0))
                .unwrap();
        }

        // Should need sweep since many have TTL
        assert!(collection.needs_expiration_sweep(0.1));
    }

    #[test]
    fn test_ttl_lazy_expiration_filters_search() {
        let config = CollectionConfig::new("test", 32).with_lazy_expiration(true);
        let mut collection = Collection::new(config);

        // Insert expired vector
        let vec1 = random_vector(32);
        collection
            .insert_with_ttl("expired", &vec1, None, Some(0))
            .unwrap();

        // Insert valid vector
        let vec2 = random_vector(32);
        collection.insert("valid", &vec2, None).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        // Search should filter out expired vector
        let results = collection.search_builder(&vec1).k(10).execute().unwrap();
        assert!(results.iter().all(|r| r.id != "expired"));
    }

    #[test]
    fn test_ttl_compact_handles_expirations() {
        let mut collection = Collection::new(CollectionConfig::new("test", 32));

        // Insert some vectors with TTL
        for i in 0..10 {
            let vec = random_vector(32);
            collection
                .insert_with_ttl(format!("doc{}", i), &vec, None, Some(3600))
                .unwrap();
        }

        // Delete some vectors
        collection.delete("doc0").unwrap();
        collection.delete("doc5").unwrap();

        // Compact should handle TTL entries correctly
        let compacted = collection.compact().unwrap();
        assert!(compacted > 0);

        // Remaining vectors should still have TTL
        let (total_with_ttl, _expired, _earliest, _latest) = collection.ttl_stats();
        assert_eq!(total_with_ttl, 8); // 10 - 2 deleted
    }

    // ========== Distance Override Tests ==========

    #[test]
    fn test_search_builder_distance_override() {
        use crate::DistanceFunction;

        let mut collection = Collection::new(CollectionConfig::new("test", 4));

        collection.insert("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Search with euclidean distance override
        let results = collection
            .search_builder(&query)
            .k(2)
            .distance(DistanceFunction::Euclidean)
            .execute()
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert!(results[0].distance < 0.001); // Euclidean distance to itself is 0
    }

    #[test]
    fn test_brute_force_search_correctness() {
        use crate::DistanceFunction;

        // Create collection with cosine distance
        let config = CollectionConfig::new("test", 3).with_distance(DistanceFunction::Cosine);
        let mut collection = Collection::new(config);

        // Vectors that have different results for cosine vs euclidean
        collection.insert("unit_x", &[1.0, 0.0, 0.0], None).unwrap();
        collection
            .insert("scaled_x", &[5.0, 0.0, 0.0], None)
            .unwrap();
        collection.insert("unit_y", &[0.0, 1.0, 0.0], None).unwrap();

        let query = vec![2.0, 0.0, 0.0];

        // Cosine search (uses HNSW) - direction matters, not magnitude
        let cosine_results = collection.search_builder(&query).k(3).execute().unwrap();
        // Both x-axis vectors should have same cosine distance (0)
        assert!(cosine_results[0].id == "unit_x" || cosine_results[0].id == "scaled_x");

        // Euclidean override (uses brute-force) - magnitude matters
        let euclidean_results = collection
            .search_builder(&query)
            .k(3)
            .distance(DistanceFunction::Euclidean)
            .execute()
            .unwrap();

        // scaled_x (5,0,0) is closer to query (2,0,0) in euclidean (dist=3)
        // than unit_x (1,0,0) (dist=1)
        // Actually: ||(2,0,0) - (1,0,0)|| = 1, ||(2,0,0) - (5,0,0)|| = 3
        // So unit_x is closer
        assert_eq!(euclidean_results[0].id, "unit_x");
    }

    #[test]
    fn test_search_builder_distance_same_as_index() {
        use crate::DistanceFunction;

        // Create with euclidean
        let config = CollectionConfig::new("test", 32).with_distance(DistanceFunction::Euclidean);
        let mut collection = Collection::new(config);

        for i in 0..50 {
            collection
                .insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }

        let query = random_vector(32);

        // Override with same distance - should use HNSW (efficient)
        let results = collection
            .search_builder(&query)
            .k(10)
            .distance(DistanceFunction::Euclidean)
            .execute()
            .unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_brute_force_with_filter() {
        use crate::metadata::Filter;
        use crate::DistanceFunction;
        use serde_json::json;

        let config = CollectionConfig::new("test", 4).with_distance(DistanceFunction::Cosine);
        let mut collection = Collection::new(config);

        collection
            .insert("a", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "x"})))
            .unwrap();
        collection
            .insert("b", &[0.0, 1.0, 0.0, 0.0], Some(json!({"type": "y"})))
            .unwrap();
        collection
            .insert("c", &[0.5, 0.5, 0.0, 0.0], Some(json!({"type": "x"})))
            .unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let filter = Filter::eq("type", "x");

        // Brute force with filter
        let results = collection
            .search_builder(&query)
            .k(10)
            .distance(DistanceFunction::Euclidean)
            .filter(&filter)
            .execute()
            .unwrap();

        // Should only return type=x vectors
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.id == "a" || r.id == "c");
        }
    }
}
