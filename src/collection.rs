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

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex, VectorId};
use crate::metadata::{Filter, MetadataStore};
use crate::storage::VectorStore;
use lru::LruCache;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use tracing::warn;

/// Search result with metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// External vector ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
    /// Associated metadata
    pub metadata: Option<Value>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(id: impl Into<String>, distance: f32, metadata: Option<Value>) -> Self {
        Self {
            id: id.into(),
            distance,
            metadata,
        }
    }
}

impl From<(String, f32)> for SearchResult {
    fn from((id, distance): (String, f32)) -> Self {
        Self {
            id,
            distance,
            metadata: None,
        }
    }
}

impl From<(String, f32, Option<Value>)> for SearchResult {
    fn from((id, distance, metadata): (String, f32, Option<Value>)) -> Self {
        Self {
            id,
            distance,
            metadata,
        }
    }
}

impl From<SearchResult> for (String, f32, Option<Value>) {
    fn from(result: SearchResult) -> Self {
        (result.id, result.distance, result.metadata)
    }
}

/// Detailed query execution plan and profiling information.
///
/// Returned by search operations when explain mode is enabled, providing
/// insights into query performance for optimization and debugging.
///
/// # Example
///
/// ```
/// use needle::Collection;
///
/// let mut collection = Collection::with_dimensions("test", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
///
/// let (results, explain) = collection.search_explain(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
/// println!("Total time: {}μs", explain.total_time_us);
/// println!("Nodes visited: {}", explain.hnsw_stats.visited_nodes);
/// ```
#[derive(Debug, Clone, Default)]
pub struct SearchExplain {
    /// Total search time in microseconds
    pub total_time_us: u64,
    /// Time spent in HNSW index traversal (microseconds)
    pub index_time_us: u64,
    /// Time spent evaluating metadata filters (microseconds)
    pub filter_time_us: u64,
    /// Time spent enriching results with metadata (microseconds)
    pub enrich_time_us: u64,
    /// Number of results before filtering
    pub candidates_before_filter: usize,
    /// Number of results after filtering
    pub candidates_after_filter: usize,
    /// HNSW index statistics
    pub hnsw_stats: crate::hnsw::SearchStats,
    /// Collection dimensions
    pub dimensions: usize,
    /// Collection vector count
    pub collection_size: usize,
    /// Requested k value
    pub requested_k: usize,
    /// Effective k (clamped to collection size)
    pub effective_k: usize,
    /// ef_search parameter used
    pub ef_search: usize,
    /// Whether a filter was applied
    pub filter_applied: bool,
    /// Distance function used
    pub distance_function: String,
}

/// Builder for configuring and executing searches.
///
/// Supports both pre-filtering (filter before ANN search) and post-filtering
/// (filter after ANN search). Use pre-filtering when the filter is selective
/// and fast; use post-filtering when you need to guarantee k results or when
/// the filter involves expensive computation.
///
/// # Example
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
/// let pre_filter = Filter::eq("type", "a");
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .filter(&pre_filter)
///     .execute()?;
///
/// // Post-filter: filter results AFTER ANN search
/// let post_filter = Filter::gt("score", 15);
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .post_filter(&post_filter)
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
}

impl<'a> SearchBuilder<'a> {
    /// Create a new search builder
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
        }
    }

    /// Set the number of results to return
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a pre-filter (applied during ANN search).
    ///
    /// Pre-filtering is efficient when the filter is selective and fast to evaluate.
    /// Candidates that don't match the filter are skipped during search.
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
    pub fn post_filter(mut self, filter: &'a Filter) -> Self {
        self.post_filter = Some(filter);
        self
    }

    /// Set the over-fetch factor for post-filtering (default: 3).
    ///
    /// When post-filtering, the search fetches `k * factor` candidates
    /// to ensure enough results remain after filtering.
    pub fn post_filter_factor(mut self, factor: usize) -> Self {
        self.post_filter_factor = factor.max(1);
        self
    }

    /// Set ef_search parameter for this query
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Whether to include metadata in results (default: true)
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Execute the search and return results
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        if self.query.len() != self.collection.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.collection.config.dimensions,
                got: self.query.len(),
            });
        }

        Collection::validate_vector(self.query)?;

        // Calculate how many candidates to fetch:
        // - Pre-filter: 10x to compensate for filtered-out results
        // - Post-filter: use post_filter_factor
        let pre_filter_factor = if self.filter.is_some() { 10 } else { 1 };
        let post_filter_factor = if self.post_filter.is_some() { self.post_filter_factor } else { 1 };
        let fetch_count = self.k * pre_filter_factor * post_filter_factor;

        // Get raw results with optional ef_search override
        let raw_results = if let Some(ef) = self.ef_search {
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
        };

        // Apply pre-filter if present (filter during ANN search phase)
        let pre_filtered: Vec<(VectorId, f32)> = if let Some(filter) = self.filter {
            raw_results
                .into_iter()
                .filter(|(id, _)| {
                    if let Some(entry) = self.collection.metadata.get(*id) {
                        filter.matches(entry.data.as_ref())
                    } else {
                        false
                    }
                })
                .take(self.k * post_filter_factor.max(1))
                .collect()
        } else {
            raw_results
                .into_iter()
                .take(self.k * post_filter_factor.max(1))
                .collect()
        };

        // Enrich results with metadata (needed for post-filter)
        let mut enriched = if self.include_metadata || self.post_filter.is_some() {
            self.collection.enrich_results(pre_filtered)?
        } else {
            // Return results without metadata
            pre_filtered
                .into_iter()
                .map(|(id, distance)| {
                    let entry = self
                        .collection
                        .metadata
                        .get(id)
                        .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;
                    Ok(SearchResult {
                        id: entry.external_id.clone(),
                        distance,
                        metadata: None,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        };

        // Apply post-filter if present (filter after ANN search)
        if let Some(post_filter) = self.post_filter {
            enriched = enriched
                .into_iter()
                .filter(|result| post_filter.matches(result.metadata.as_ref()))
                .take(self.k)
                .collect();
        } else {
            enriched.truncate(self.k);
        }

        // Strip metadata if not requested (but was needed for post-filter)
        if !self.include_metadata && self.post_filter.is_some() {
            for result in &mut enriched {
                result.metadata = None;
            }
        }

        Ok(enriched)
    }

    /// Execute the search and return only IDs with distances
    pub fn execute_ids_only(self) -> Result<Vec<(String, f32)>> {
        self.include_metadata(false)
            .execute()
            .map(|results| results.into_iter().map(|r| (r.id, r.distance)).collect())
    }
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Collection name
    pub name: String,
    /// Number of vectors
    pub vector_count: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function used
    pub distance_function: DistanceFunction,
    /// Estimated memory for vectors (bytes)
    pub vector_memory_bytes: usize,
    /// Estimated memory for metadata (bytes)
    pub metadata_memory_bytes: usize,
    /// Estimated memory for index (bytes)
    pub index_memory_bytes: usize,
    /// Total estimated memory (bytes)
    pub total_memory_bytes: usize,
    /// HNSW index statistics
    pub index_stats: crate::hnsw::HnswStats,
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

/// Query cache statistics
#[derive(Debug, Clone, Default)]
pub struct QueryCacheStats {
    /// Number of cache hits
    pub hits: u64,
    /// Number of cache misses
    pub misses: u64,
    /// Current number of cached entries
    pub size: usize,
    /// Maximum cache capacity
    pub capacity: usize,
}

impl QueryCacheStats {
    /// Returns the cache hit ratio (0.0 to 1.0)
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Configuration for query result caching.
///
/// Query caching stores search results to avoid redundant HNSW traversals
/// for identical queries. This is beneficial when the same queries are
/// executed repeatedly, such as in benchmarking or when serving repeated
/// user requests.
///
/// # Example
///
/// ```
/// use needle::{CollectionConfig, QueryCacheConfig};
///
/// // Enable caching with 1000 entries
/// let cache_config = QueryCacheConfig::new(1000);
///
/// let config = CollectionConfig::new("embeddings", 128)
///     .with_query_cache(cache_config);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCacheConfig {
    /// Maximum number of query results to cache.
    /// Set to 0 to disable caching.
    pub capacity: usize,
}

impl QueryCacheConfig {
    /// Create a new query cache configuration.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of query results to cache
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }

    /// Create a disabled cache configuration.
    pub fn disabled() -> Self {
        Self { capacity: 0 }
    }

    /// Check if caching is enabled.
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Name of the collection
    pub name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function
    pub distance: DistanceFunction,
    /// HNSW configuration
    pub hnsw: HnswConfig,
    /// Slow query threshold in microseconds.
    /// If set, queries exceeding this time will be logged at warn level.
    #[serde(default)]
    pub slow_query_threshold_us: Option<u64>,
    /// Query cache configuration
    #[serde(default)]
    pub query_cache: QueryCacheConfig,
}

impl CollectionConfig {
    /// Create a new collection config with default settings
    ///
    /// # Panics
    /// Panics if dimensions is 0.
    pub fn new(name: impl Into<String>, dimensions: usize) -> Self {
        assert!(dimensions > 0, "Vector dimensions must be greater than 0");
        Self {
            name: name.into(),
            dimensions,
            distance: DistanceFunction::Cosine,
            hnsw: HnswConfig::default(),
            slow_query_threshold_us: None,
            query_cache: QueryCacheConfig::default(),
        }
    }

    /// Set the distance function
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance = distance;
        self
    }

    /// Set the HNSW M parameter
    pub fn with_m(mut self, m: usize) -> Self {
        self.hnsw = HnswConfig::with_m(m);
        self
    }

    /// Set ef_construction
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw.ef_construction = ef;
        self
    }

    /// Set the slow query threshold in microseconds.
    ///
    /// When set, search queries that exceed this duration will be logged
    /// at the warn level with query details.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::CollectionConfig;
    ///
    /// // Log queries slower than 100ms
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_slow_query_threshold_us(100_000);
    /// ```
    pub fn with_slow_query_threshold_us(mut self, threshold_us: u64) -> Self {
        self.slow_query_threshold_us = Some(threshold_us);
        self
    }

    /// Enable query result caching with the specified configuration.
    ///
    /// Query caching stores search results to avoid redundant HNSW traversals
    /// for identical queries. The cache is automatically invalidated when
    /// vectors are inserted, updated, or deleted.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{CollectionConfig, QueryCacheConfig};
    ///
    /// // Enable caching with 1000 entries
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_query_cache(QueryCacheConfig::new(1000));
    /// ```
    pub fn with_query_cache(mut self, cache_config: QueryCacheConfig) -> Self {
        self.query_cache = cache_config;
        self
    }

    /// Enable query result caching with a specified capacity.
    ///
    /// Shorthand for `with_query_cache(QueryCacheConfig::new(capacity))`.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::CollectionConfig;
    ///
    /// // Enable caching with 500 entries
    /// let config = CollectionConfig::new("embeddings", 128)
    ///     .with_query_cache_capacity(500);
    /// ```
    pub fn with_query_cache_capacity(mut self, capacity: usize) -> Self {
        self.query_cache = QueryCacheConfig::new(capacity);
        self
    }
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
            config,
        }
    }

    /// Create a collection with just name and dimensions
    pub fn with_dimensions(name: impl Into<String>, dimensions: usize) -> Self {
        Self::new(CollectionConfig::new(name, dimensions))
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
        self.query_cache.as_ref().map(|cache| {
            cache.lock().stats(self.config.query_cache.capacity)
        })
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
                cache_guard.put(cache_key, CachedSearchResult {
                    results: results.clone(),
                });
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
        let id = id.into();

        // Validate dimensions and vector values
        self.validate_insert_input(vector)?;

        // Check if ID already exists
        if self.metadata.contains(&id) {
            return Err(NeedleError::VectorAlreadyExists(id));
        }

        // Add to vector store
        let internal_id = self.vectors.add(vector.to_vec())?;

        // Add to metadata
        self.metadata.insert(internal_id, id, metadata)?;

        // Add to index
        self.index
            .insert(internal_id, vector, self.vectors.as_slice())?;

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(())
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
        let vector_ref = self.vectors.get(internal_id)
            .ok_or_else(|| NeedleError::Index("Vector not found after insert".into()))?;
        self.index
            .insert(internal_id, vector_ref, self.vectors.as_slice())?;

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

        // Use insert_vec to avoid unnecessary clones
        for ((id, vector), meta) in ids.into_iter().zip(vectors).zip(metadata) {
            self.insert_vec(id, vector, meta)?;
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
            self.enrich_results(raw_results)
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
        let (raw_results, hnsw_stats) = self.index.search_with_stats(
            query,
            effective_k,
            self.vectors.as_slice(),
        );
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
            effective_k * 10,
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
                let candidates = self.index.search(query, k * 10, self.vectors.as_slice());
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
    /// Internally, this fetches more candidates than requested (`k * 10`)
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
        let candidates = self.index.search(query, k * 10, self.vectors.as_slice());

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
        let fetch_count = limit * 10;
        let candidates = self.index.search(query, fetch_count, self.vectors.as_slice());

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
    pub fn update(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        // Check dimensions
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Get internal ID
        let internal_id = self.metadata.get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        // Update vector in storage
        self.vectors.update(internal_id, vector.to_vec())?;

        // Update metadata
        self.metadata.update_data(internal_id, metadata)?;

        // Re-index the vector (delete and re-insert in index)
        self.index.delete(internal_id)?;
        self.index.insert(internal_id, vector, self.vectors.as_slice())?;

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(())
    }

    /// Update only the metadata for an existing vector
    /// Returns error if the vector doesn't exist
    pub fn update_metadata(&mut self, id: &str, metadata: Option<Value>) -> Result<()> {
        let internal_id = self.metadata.get_internal_id(id)
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
        self.metadata.iter().filter_map(move |(internal_id, entry)| {
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

    /// Compact the collection by removing deleted vectors
    /// This rebuilds the index and reclaims storage
    /// Returns the number of vectors removed
    pub fn compact(&mut self) -> Result<usize> {
        let deleted_count = self.index.deleted_count();

        if deleted_count == 0 {
            return Ok(0);
        }

        // Get the ID mapping from compacting the index
        let id_map = self.index.compact(self.vectors.as_slice());

        // Rebuild vectors and metadata with new IDs
        let mut new_vectors = VectorStore::new(self.config.dimensions);
        let mut new_metadata = MetadataStore::new();

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
            }
        }

        self.vectors = new_vectors;
        self.metadata = new_metadata;

        // Invalidate cache since internal IDs changed
        self.invalidate_cache();

        Ok(deleted_count)
    }

    /// Check if the collection needs compaction
    /// Returns true if deleted vectors exceed the given threshold (0.0-1.0)
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.index.needs_compaction(threshold)
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
    pub fn iter_filtered<'a>(&'a self, filter: &'a Filter) -> impl Iterator<Item = (&'a str, &'a [f32], Option<&'a Value>)> + 'a {
        self.iter().filter(move |(_, _, meta)| filter.matches(*meta))
    }

    /// Get all vector IDs as a collected Vec
    pub fn all_ids(&self) -> Vec<String> {
        self.metadata.all_external_ids()
    }

    /// Estimate if a dataset of given size would fit in memory
    pub fn estimate_memory(vector_count: usize, dimensions: usize, avg_metadata_bytes: usize) -> usize {
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
                return Some((
                    id.clone(),
                    vector.to_vec(),
                    metadata.cloned(),
                ));
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
                    Some(json!({"score": i, "type": if i % 3 == 0 { "special" } else { "normal" }})),
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
                .insert(
                    format!("doc{}", i),
                    &vec,
                    Some(json!({"keep": i < 25})),
                )
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
        let config = CollectionConfig::new("test", 4)
            .with_distance(DistanceFunction::Euclidean);
        let mut collection = Collection::new(config);

        // Insert vectors at known positions
        collection.insert("origin", &[0.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("close", &[0.1, 0.0, 0.0, 0.0], None).unwrap();  // dist = 0.1
        collection.insert("medium", &[0.5, 0.0, 0.0, 0.0], None).unwrap(); // dist = 0.5
        collection.insert("far", &[2.0, 0.0, 0.0, 0.0], None).unwrap();    // dist = 2.0

        // Query at origin
        let query = [0.0, 0.0, 0.0, 0.0];

        // Search within radius 0.15 - should only find origin and close
        let results = collection.search_radius(&query, 0.15, 100).unwrap();
        assert!(!results.is_empty() && results.len() <= 2);
        for r in &results {
            assert!(r.distance <= 0.15, "Distance {} exceeds max 0.15", r.distance);
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

        let config = CollectionConfig::new("test", 4)
            .with_distance(DistanceFunction::Euclidean);
        let mut collection = Collection::new(config);

        // Insert vectors with category
        collection.insert("a1", &[0.1, 0.0, 0.0, 0.0], Some(json!({"type": "A"}))).unwrap();
        collection.insert("a2", &[0.2, 0.0, 0.0, 0.0], Some(json!({"type": "A"}))).unwrap();
        collection.insert("b1", &[0.15, 0.0, 0.0, 0.0], Some(json!({"type": "B"}))).unwrap();
        collection.insert("far", &[5.0, 0.0, 0.0, 0.0], Some(json!({"type": "A"}))).unwrap();

        let query = [0.0, 0.0, 0.0, 0.0];
        let filter = Filter::eq("type", "A");

        // Search within radius 0.3 for type A only
        let results = collection.search_radius_with_filter(&query, 0.3, 100, &filter).unwrap();

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
        let config = CollectionConfig::new("test", 64)
            .with_slow_query_threshold_us(100_000); // 100ms threshold

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
        let config = CollectionConfig::new("test", 64)
            .with_slow_query_threshold_us(75_000);
        let mut collection = Collection::new(config);

        // Insert some data
        for i in 0..10 {
            collection.insert(format!("v{}", i), &random_vector(64), None).unwrap();
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
        let config = CollectionConfig::new("test", 64)
            .with_query_cache_capacity(100);

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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert some vectors
        for i in 0..10 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert some vectors
        for i in 0..10 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert initial vectors
        for i in 0..5 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
        let config = CollectionConfig::new("test", 32)
            .with_query_cache_capacity(100);
        let mut collection = Collection::new(config);

        // Insert vectors
        for i in 0..5 {
            collection.insert(format!("v{}", i), &random_vector(32), None).unwrap();
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
}
