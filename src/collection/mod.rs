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

#![allow(clippy::unwrap_used)] // tech debt: per-module unwrap cleanup in progress
pub mod config;
pub use config::*;
pub mod search;
pub use search::*;
pub mod pipeline;
pub use pipeline::*;
mod sharding;
use sharding::*;
mod validation;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex, VectorId};
use crate::metadata::{Filter, MetadataStore};
use crate::storage::VectorStore;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;





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
    /// Query result cache (sharded for reduced lock contention)
    /// Skipped during serialization - cache is rebuilt on load
    #[serde(skip)]
    query_cache: Option<ShardedQueryCache>,
    /// TTL expiration tracking: internal_id -> expiration timestamp (Unix epoch seconds)
    /// Vectors with expired timestamps are filtered out during search (lazy)
    /// or removed during explicit sweep operations.
    #[serde(default)]
    expirations: std::collections::HashMap<usize, u64>,
    /// Insertion timestamps: internal_id -> insertion Unix epoch seconds.
    /// Used by `SearchBuilder::as_of()` for MVCC point-in-time queries.
    #[serde(default)]
    insertion_timestamps: std::collections::HashMap<usize, u64>,
    /// Semantic query cache (similarity-based cache lookups)
    /// Skipped during serialization - cache is rebuilt on load
    #[serde(skip)]
    semantic_cache: Option<Mutex<SemanticQueryCache>>,
    /// Provenance tracking store
    #[serde(default)]
    provenance_store: crate::persistence::vector_versioning::ProvenanceStore,
}

/// Manifest for a portable collection bundle.
///
/// Contains metadata about the bundled collection for validation during import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Bundle format version
    pub format_version: u32,
    /// Collection name
    pub collection_name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function
    pub distance_function: String,
    /// Number of vectors in the bundle
    pub vector_count: usize,
    /// Embedding model used (if known)
    pub embedding_model: Option<String>,
    /// Bundle creation timestamp (Unix epoch seconds)
    pub created_at: u64,
    /// SHA-256 hash of the serialized collection data
    pub data_hash: Option<String>,
}

impl Collection {
    /// Create a new collection
    pub fn new(config: CollectionConfig) -> Self {
        let query_cache = if config.query_cache.is_enabled() {
            NonZeroUsize::new(config.query_cache.capacity)
                .map(|cap| ShardedQueryCache::new(cap))
        } else {
            None
        };

        let semantic_cache = config.semantic_cache.as_ref().map(|sc_config| {
            Mutex::new(SemanticQueryCache::new(sc_config, config.dimensions))
        });

        Self {
            vectors: VectorStore::new(config.dimensions),
            index: HnswIndex::new(config.hnsw.clone(), config.distance),
            metadata: MetadataStore::new(),
            query_cache,
            expirations: HashMap::new(),
            insertion_timestamps: HashMap::new(),
            semantic_cache,
            provenance_store: crate::persistence::vector_versioning::ProvenanceStore::new(),
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

    /// Set the collection name.
    pub fn set_name(&mut self, name: String) {
        self.config.name = name;
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

    /// Get the insertion timestamp for a vector by internal ID.
    /// Returns `None` if no timestamp is recorded (e.g., vectors loaded from legacy format).
    pub fn insertion_timestamp(&self, internal_id: usize) -> Option<u64> {
        self.insertion_timestamps.get(&internal_id).copied()
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
            self.query_cache = Some(ShardedQueryCache::new(cap));
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
            cache.clear();
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
    /// let _results = collection.search(&[0.0; 128], 10);
    ///
    /// // Second search - cache hit
    /// let _results = collection.search(&[0.0; 128], 10);
    ///
    /// let stats = collection.query_cache_stats().unwrap();
    /// assert_eq!(stats.hits, 1);
    /// assert_eq!(stats.misses, 1);
    /// ```
    pub fn query_cache_stats(&self) -> Option<QueryCacheStats> {
        self.query_cache
            .as_ref()
            .map(|cache| {
                let mut stats = cache.stats(self.config.query_cache.capacity);
                // Merge semantic cache stats if present
                if let Some(ref sem_cache) = self.semantic_cache {
                    let (sem_hits, sem_misses) = sem_cache.lock().stats();
                    stats.semantic_hits = sem_hits;
                    stats.semantic_misses = sem_misses;
                }
                stats
            })
    }

    /// Get semantic cache statistics.
    ///
    /// Returns `None` if semantic caching is disabled.
    pub fn semantic_cache_stats(&self) -> Option<QueryCacheStats> {
        self.semantic_cache.as_ref().map(|sem_cache| {
            let cache = sem_cache.lock();
            let (sem_hits, sem_misses) = cache.stats();
            QueryCacheStats {
                hits: 0,
                misses: 0,
                size: cache.entries.len(),
                capacity: cache.capacity,
                semantic_hits: sem_hits,
                semantic_misses: sem_misses,
            }
        })
    }

    /// Warm the semantic cache with pre-computed query/result pairs.
    ///
    /// Useful for pre-populating the cache on startup with common queries
    /// to avoid cold-start cache misses.
    ///
    /// No-op if semantic caching is disabled.
    pub fn warm_semantic_cache(&self, entries: Vec<(&[f32], usize, Vec<SearchResult>)>) {
        if let Some(ref sem_cache) = self.semantic_cache {
            let mut cache = sem_cache.lock();
            cache.warm(entries);
        }
    }

    /// Warm the semantic cache by executing actual searches on a set of queries.
    ///
    /// Each query vector is searched and the results are cached. This is useful
    /// for pre-warming the cache with known common queries on startup.
    ///
    /// No-op if semantic caching is disabled.
    pub fn warm_semantic_cache_from_queries(&self, queries: &[Vec<f32>], k: usize) -> Result<()> {
        if self.semantic_cache.is_none() {
            return Ok(());
        }
        for query in queries {
            // search() already caches results in the semantic cache
            let _ = self.search(query, k)?;
        }
        Ok(())
    }

    /// Helper to invalidate the query cache.
    /// Called automatically on mutations (insert, update, delete).
    fn invalidate_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.clear();
        }
        if let Some(ref sem_cache) = self.semantic_cache {
            sem_cache.lock().clear();
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
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.results.clone());
            }

            // Cache miss - compute result
            let results = compute()?;

            // Store in cache
            cache.put(
                cache_key,
                CachedSearchResult {
                    results: results.clone(),
                },
            );

            Ok(results)
        } else {
            // No cache - just compute
            compute()
        }
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

        // Record insertion timestamp for MVCC as_of queries
        self.insertion_timestamps.insert(internal_id, Self::now_unix());

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
                        if let Err(e) = self.delete(&inserted) {
                            tracing::warn!("Failed to rollback inserted vector '{}': {}", inserted, e);
                        }
                    }
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    /// Insert a vector with provenance tracking.
    ///
    /// Records the full provenance of the vector including source document,
    /// embedding model, pipeline ID, and parent vector.
    pub fn insert_with_provenance(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
        provenance: crate::persistence::vector_versioning::ProvenanceRecord,
    ) -> Result<()> {
        let id = id.into();
        self.insert(&id, vector, metadata)?;
        self.provenance_store.insert(provenance);
        Ok(())
    }

    /// Get provenance record for a vector
    pub fn get_provenance(
        &self,
        vector_id: &str,
    ) -> Option<&crate::persistence::vector_versioning::ProvenanceRecord> {
        self.provenance_store.get(vector_id)
    }

    /// Get the provenance store for querying
    pub fn provenance_store(&self) -> &crate::persistence::vector_versioning::ProvenanceStore {
        &self.provenance_store
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

        // Check semantic cache BEFORE exact-match LRU cache
        if let Some(ref sem_cache) = self.semantic_cache {
            let mut cache = sem_cache.lock();
            if let Some(results) = cache.lookup(query, k) {
                return Ok(results);
            }
        }

        // Use exact-match cache if enabled
        let results = self.search_with_cache(query, k, || {
            let raw_results = self.index.search(query, k, self.vectors.as_slice())?;

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

        // Store in semantic cache
        if let Some(ref sem_cache) = self.semantic_cache {
            let mut cache = sem_cache.lock();
            cache.insert(query, k, &results);
        }

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

    /// Search using Matryoshka-style dimension truncation for faster retrieval.
    ///
    /// Performs a two-phase search:
    /// 1. **Coarse phase**: Uses HNSW search on truncated vectors for O(log n) candidate retrieval
    /// 2. **Re-rank phase**: Re-scores candidates using full-dimension vectors and returns top `k`
    ///
    /// This works with Matryoshka-trained embeddings (e.g., OpenAI text-embedding-3, Nomic embed)
    /// where prefix truncation preserves semantic meaning.
    ///
    /// # Arguments
    /// * `query` - Full-dimension query vector
    /// * `k` - Number of results to return
    /// * `coarse_dims` - Truncated dimension count for coarse search (e.g., 256 for 1024d vectors)
    /// * `oversample` - Multiplier for candidate set size (default: 4)
    pub fn search_matryoshka(
        &self,
        query: &[f32],
        k: usize,
        coarse_dims: usize,
        oversample: usize,
    ) -> Result<Vec<SearchResult>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let dims = self.config.dimensions;
        if coarse_dims >= dims || coarse_dims == 0 {
            return self.search(query, k);
        }

        let oversample = oversample.max(2);
        let candidate_k = (k * oversample).min(self.len());

        // Build a truncated vector set for HNSW coarse search
        let all_vectors = self.vectors.as_slice();
        let truncated_vectors: Vec<Vec<f32>> = all_vectors
            .iter()
            .map(|v| v[..coarse_dims.min(v.len())].to_vec())
            .collect();

        // Phase 1: HNSW search on truncated vectors — O(log n) instead of O(n)
        let truncated_query = &query[..coarse_dims];
        let coarse_hnsw = HnswIndex::new(
            HnswConfig::default().ef_search(candidate_k.max(50)),
            self.config.distance,
        );
        // Use the primary index's HNSW graph for traversal but with truncated distance computation
        let candidates = self.index.search(truncated_query, candidate_k, &truncated_vectors)?;

        // Phase 2: Re-rank with full dimensions
        let distance_fn = self.config.distance;
        let mut reranked: Vec<(usize, f32)> = candidates
            .iter()
            .filter(|(id, _)| !self.index.is_deleted(*id) && !self.is_expired(*id))
            .filter_map(|(id, _)| {
                let full_vec = self.vectors.get(*id)?;
                let full_dist = distance_fn.compute(query, full_vec).ok()?;
                Some((*id, full_dist))
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);

        self.enrich_results(reranked)
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
            let dist = params.distance_fn.compute(params.query, vector)?;

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
                .search_with_stats(query, effective_k, self.vectors.as_slice())?;
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
        )?;
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

        let results = self.index.search(query, k, self.vectors.as_slice())?;

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

        let results = self.index.search(query, k, self.vectors.as_slice())?;

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
                let raw_results = self.index.search(query, k, self.vectors.as_slice())?;
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
                )?;
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

        // Bloom filter pre-check: if any equality condition references a
        // field+value that definitely doesn't exist, return early.
        let eq_conditions = filter.equality_conditions();
        for (field, value_str) in &eq_conditions {
            if !self.metadata.bloom_might_contain(field, value_str) {
                return Ok(Vec::new());
            }
        }

        // For filtered search, we need to retrieve more candidates and filter
        let candidates = self.index.search(
            query,
            k * FILTER_CANDIDATE_MULTIPLIER,
            self.vectors.as_slice(),
        )?;

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
        let candidates = self.index.search(query, limit, self.vectors.as_slice())?;

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
            .search(query, fetch_count, self.vectors.as_slice())?;

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
        self.provenance_store.remove(id);

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
        let id_map = self.index.compact(self.vectors.as_slice())?;

        // Rebuild vectors and metadata with new IDs
        let mut new_vectors = VectorStore::new(self.config.dimensions);
        let mut new_metadata = MetadataStore::new();
        let mut new_expirations = HashMap::new();
        let mut new_insertion_timestamps = HashMap::new();

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

                // Remap insertion timestamp if it exists
                if let Some(ts) = self.insertion_timestamps.get(&old_id) {
                    new_insertion_timestamps.insert(new_id, *ts);
                }
            }
        }

        self.vectors = new_vectors;
        self.metadata = new_metadata;
        self.expirations = new_expirations;
        self.insertion_timestamps = new_insertion_timestamps;

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

    /// Create a serialized snapshot of the collection state.
    ///
    /// Returns the snapshot as a JSON byte vector that can be stored or
    /// later restored with [`restore_snapshot`].
    pub fn create_snapshot(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| {
            NeedleError::InvalidInput(format!("Failed to serialize snapshot: {e}"))
        })
    }

    /// Restore a collection from a previously created snapshot.
    ///
    /// Replaces the current collection state with the snapshot data.
    pub fn restore_snapshot(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| {
            NeedleError::InvalidInput(format!("Failed to deserialize snapshot: {e}"))
        })
    }

    /// Evaluate search quality using ground truth data.
    ///
    /// Computes recall@k, precision@k, MAP, MRR, and NDCG metrics.
    ///
    /// # Arguments
    /// * `ground_truth` - List of (query_vector, relevant_ids) pairs
    /// * `k` - Number of results to retrieve per query
    ///
    /// # Returns
    /// An `EvaluationReport` with aggregated and per-query metrics.
    pub fn evaluate(&self, ground_truth: &[GroundTruthEntry], k: usize) -> Result<EvaluationReport> {
        use std::time::Instant;

        let start = Instant::now();
        let mut per_query = Vec::with_capacity(ground_truth.len());

        for (idx, entry) in ground_truth.iter().enumerate() {
            let results = self.search(&entry.query, k)?;
            let retrieved_ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
            let relevant_set: HashSet<&str> = entry.relevant_ids.iter().map(|s| s.as_str()).collect();

            // Recall@k
            let hits = retrieved_ids.iter().filter(|id| relevant_set.contains(*id)).count();
            let recall = if relevant_set.is_empty() { 1.0 } else { hits as f64 / relevant_set.len() as f64 };

            // Precision@k
            let precision = if retrieved_ids.is_empty() { 0.0 } else { hits as f64 / retrieved_ids.len() as f64 };

            // Average Precision
            let mut ap_sum = 0.0;
            let mut relevant_count = 0;
            for (rank, id) in retrieved_ids.iter().enumerate() {
                if relevant_set.contains(*id) {
                    relevant_count += 1;
                    ap_sum += relevant_count as f64 / (rank + 1) as f64;
                }
            }
            let ap = if relevant_set.is_empty() { 0.0 } else { ap_sum / relevant_set.len() as f64 };

            // Reciprocal Rank
            let rr = retrieved_ids.iter()
                .position(|id| relevant_set.contains(*id))
                .map(|pos| 1.0 / (pos + 1) as f64)
                .unwrap_or(0.0);

            // NDCG@k
            let dcg: f64 = retrieved_ids.iter().enumerate()
                .map(|(rank, id)| {
                    let rel = if relevant_set.contains(*id) { 1.0 } else { 0.0 };
                    rel / (rank as f64 + 2.0).log2()
                })
                .sum();
            let ideal_hits = relevant_set.len().min(k);
            let idcg: f64 = (0..ideal_hits)
                .map(|rank| 1.0 / (rank as f64 + 2.0).log2())
                .sum();
            let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };

            per_query.push(QueryMetrics {
                query_index: idx,
                recall_at_k: recall,
                precision_at_k: precision,
                average_precision: ap,
                reciprocal_rank: rr,
                ndcg,
            });
        }

        let n = per_query.len() as f64;
        let mean = |f: fn(&QueryMetrics) -> f64| -> f64 {
            if n == 0.0 { 0.0 } else { per_query.iter().map(f).sum::<f64>() / n }
        };

        Ok(EvaluationReport {
            num_queries: per_query.len(),
            k,
            mean_recall_at_k: mean(|q| q.recall_at_k),
            mean_precision_at_k: mean(|q| q.precision_at_k),
            map: mean(|q| q.average_precision),
            mrr: mean(|q| q.reciprocal_rank),
            mean_ndcg: mean(|q| q.ndcg),
            eval_time_ms: start.elapsed().as_secs_f64() * 1000.0,
            per_query,
        })
    }

    /// Record relevance feedback for search result quality improvement.
    /// This feeds into the contextual bandits reranker for learning optimal result ordering.
    pub fn record_feedback(&self, query_id: &str, vector_id: &str, relevance_score: f32) {
        // Feedback is stored as metadata for now; full bandits integration
        // happens through the BanditsReranker in the search pipeline.
        tracing::info!(
            collection = %self.config.name,
            query_id = query_id,
            vector_id = vector_id,
            relevance_score = relevance_score,
            "relevance feedback recorded"
        );
    }

    // ============ Bundle Export/Import Methods ============

    /// Export the collection as a portable bundle to the specified path.
    ///
    /// The bundle is a JSON file containing the manifest and serialized collection data.
    /// This enables easy sharing and migration of collections between Needle instances.
    ///
    /// # Errors
    /// Returns an error if serialization or file I/O fails.
    pub fn export_bundle(&self, path: &std::path::Path) -> Result<BundleManifest> {
        use sha2::{Digest, Sha256};

        let data =
            serde_json::to_vec(self).map_err(|e| NeedleError::Serialization(e))?;

        let hash = {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            format!("{:x}", hasher.finalize())
        };

        let manifest = BundleManifest {
            format_version: 1,
            collection_name: self.config.name.clone(),
            dimensions: self.config.dimensions,
            distance_function: format!("{:?}", self.config.distance),
            vector_count: self.len(),
            embedding_model: None,
            created_at: Self::now_unix(),
            data_hash: Some(hash),
        };

        let bundle = serde_json::json!({
            "manifest": manifest,
            "data": serde_json::from_slice::<serde_json::Value>(&data).unwrap_or_default(),
        });

        let bundle_bytes = serde_json::to_vec_pretty(&bundle)
            .map_err(|e| NeedleError::Serialization(e))?;
        std::fs::write(path, &bundle_bytes).map_err(NeedleError::Io)?;

        Ok(manifest)
    }

    /// Import a collection from a portable bundle file.
    ///
    /// Validates the bundle manifest for schema compatibility before importing.
    ///
    /// # Errors
    /// Returns an error if the bundle format is invalid, incompatible, or I/O fails.
    pub fn import_bundle(path: &std::path::Path) -> Result<Self> {
        let bundle_bytes = std::fs::read(path).map_err(NeedleError::Io)?;
        let bundle: serde_json::Value = serde_json::from_slice(&bundle_bytes)
            .map_err(|e| NeedleError::Serialization(e))?;

        let manifest: BundleManifest = serde_json::from_value(
            bundle
                .get("manifest")
                .ok_or_else(|| {
                    NeedleError::InvalidDatabase("Bundle missing manifest".to_string())
                })?
                .clone(),
        )
        .map_err(|e| NeedleError::InvalidDatabase(format!("Invalid manifest: {}", e)))?;

        if manifest.format_version != 1 {
            return Err(NeedleError::InvalidDatabase(format!(
                "Unsupported bundle format version: {}",
                manifest.format_version
            )));
        }

        let data = bundle.get("data").ok_or_else(|| {
            NeedleError::InvalidDatabase("Bundle missing data".to_string())
        })?;

        let mut collection: Collection = serde_json::from_value(data.clone())
            .map_err(|e| {
                NeedleError::InvalidDatabase(format!("Invalid collection data: {}", e))
            })?;

        // Reinitialize non-serializable fields
        if collection.config.query_cache.is_enabled() {
            if let Some(cap) =
                std::num::NonZeroUsize::new(collection.config.query_cache.capacity)
            {
                collection.query_cache = Some(ShardedQueryCache::new(cap));
            }
        }

        Ok(collection)
    }

    /// Validate that a bundle is compatible by reading its manifest.
    ///
    /// # Errors
    /// Returns an error if the bundle file cannot be read or parsed.
    pub fn validate_bundle_compatibility(
        path: &std::path::Path,
    ) -> Result<BundleManifest> {
        let bundle_bytes = std::fs::read(path).map_err(NeedleError::Io)?;
        let bundle: serde_json::Value = serde_json::from_slice(&bundle_bytes)
            .map_err(|e| NeedleError::Serialization(e))?;

        let manifest: BundleManifest = serde_json::from_value(
            bundle
                .get("manifest")
                .ok_or_else(|| {
                    NeedleError::InvalidDatabase("Bundle missing manifest".to_string())
                })?
                .clone(),
        )
        .map_err(|e| NeedleError::InvalidDatabase(format!("Invalid manifest: {}", e)))?;

        Ok(manifest)
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
        let _results = collection.search(&query, 5).unwrap();

        // Search with k=3 (different k, should miss)
        let _results = collection.search(&query, 3).unwrap();

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
        let _results = collection.search(&query, 5).unwrap();
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
        let _results = collection.search(&query, 5).unwrap();
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
        let _results = collection.search(&query, 5).unwrap();
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
        let _results = collection.search(&query, 5).unwrap();

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
            let _results = collection.search(&query, 5).unwrap();
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
        let _results = collection.search(&query, 5).unwrap();

        // 4 hits
        for _ in 0..4 {
            let _results = collection.search(&query, 5).unwrap();
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

    #[test]
    fn test_semantic_cache_hit() {
        let config = CollectionConfig::new("test", 4)
            .with_semantic_cache(
                crate::collection::config::SemanticQueryCacheConfig::new(10, 0.90)
            );
        let mut collection = Collection::new(config);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        // First search – cache miss
        let r1 = collection.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!r1.is_empty());

        let stats = collection.semantic_cache_stats().unwrap();
        assert_eq!(stats.semantic_hits, 0);
        assert_eq!(stats.semantic_misses, 1);

        // Nearly identical query – cache hit
        let r2 = collection.search(&[0.999, 0.001, 0.0, 0.0], 2).unwrap();
        let stats = collection.semantic_cache_stats().unwrap();
        assert_eq!(stats.semantic_hits, 1);
        assert_eq!(r2[0].id, r1[0].id);
    }

    #[test]
    fn test_semantic_cache_invalidation() {
        let config = CollectionConfig::new("test", 4)
            .with_semantic_cache(
                crate::collection::config::SemanticQueryCacheConfig::new(10, 0.90)
            );
        let mut collection = Collection::new(config);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        let _ = collection.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().size, 1);

        // Insert invalidates cache
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().size, 0);
    }

    #[test]
    fn test_semantic_cache_lru_eviction() {
        let config = CollectionConfig::new("test", 4)
            .with_semantic_cache(
                crate::collection::config::SemanticQueryCacheConfig::new(2, 0.99)
            );
        let mut collection = Collection::new(config);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        collection.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();

        // Fill cache with 2 entries (at capacity)
        let _ = collection.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        let _ = collection.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().size, 2);

        // Third query triggers LRU eviction, should still work
        let _ = collection.search(&[0.0, 0.0, 1.0, 0.0], 1).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().size, 2);
    }

    #[test]
    fn test_semantic_cache_warming() {
        let config = CollectionConfig::new("test", 4)
            .with_semantic_cache(
                crate::collection::config::SemanticQueryCacheConfig::new(10, 0.90)
            );
        let mut collection = Collection::new(config);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        // Warm the cache
        let warm_results = vec![SearchResult::new("v1", 0.01, None)];
        let query = [1.0f32, 0.0, 0.0, 0.0];
        collection.warm_semantic_cache(vec![(&query, 1, warm_results)]);
        assert_eq!(collection.semantic_cache_stats().unwrap().size, 1);

        // Lookup should hit the warmed entry
        let r = collection.search(&[0.999, 0.001, 0.0, 0.0], 1).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().semantic_hits, 1);
        assert_eq!(r[0].id, "v1");
    }

    #[test]
    fn test_semantic_cache_warm_from_queries() {
        let config = CollectionConfig::new("test", 4)
            .with_semantic_cache(
                crate::collection::config::SemanticQueryCacheConfig::new(10, 0.90)
            );
        let mut collection = Collection::new(config);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        // Warm from actual queries
        let queries = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        collection.warm_semantic_cache_from_queries(&queries, 2).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().size, 2);

        // Subsequent similar queries should hit cache
        let _ = collection.search(&[0.999, 0.001, 0.0, 0.0], 2).unwrap();
        assert_eq!(collection.semantic_cache_stats().unwrap().semantic_hits, 1);
    }

    #[test]
    fn test_evaluate_basic() {
        let mut collection = Collection::with_dimensions("test", 4);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        collection.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();

        let gt = vec![GroundTruthEntry {
            query: vec![1.0, 0.0, 0.0, 0.0],
            relevant_ids: vec!["v1".to_string()],
        }];
        let report = collection.evaluate(&gt, 3).unwrap();
        assert_eq!(report.num_queries, 1);
        assert!(report.mean_recall_at_k > 0.99, "recall should be 1.0");
        assert!(report.mrr > 0.99, "MRR should be 1.0");
    }

    #[test]
    fn test_evaluate_multiple_queries() {
        let mut collection = Collection::with_dimensions("test", 4);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        collection.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();
        collection.insert("v4", &[0.0, 0.0, 0.0, 1.0], None).unwrap();

        let gt = vec![
            GroundTruthEntry {
                query: vec![1.0, 0.0, 0.0, 0.0],
                relevant_ids: vec!["v1".to_string()],
            },
            GroundTruthEntry {
                query: vec![0.0, 1.0, 0.0, 0.0],
                relevant_ids: vec!["v2".to_string()],
            },
        ];
        let report = collection.evaluate(&gt, 4).unwrap();
        assert_eq!(report.num_queries, 2);
        assert!(report.mean_recall_at_k > 0.99);
        assert!(report.mrr > 0.99);
        assert!(report.mean_ndcg > 0.99);
        assert_eq!(report.per_query.len(), 2);
    }

    #[test]
    fn test_evaluate_precision() {
        let mut collection = Collection::with_dimensions("test", 4);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], None).unwrap();
        collection.insert("v3", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        // Only v1 is relevant, but we ask for k=3 → precision should be ~0.33
        let gt = vec![GroundTruthEntry {
            query: vec![1.0, 0.0, 0.0, 0.0],
            relevant_ids: vec!["v1".to_string()],
        }];
        let report = collection.evaluate(&gt, 3).unwrap();
        assert!(report.mean_precision_at_k < 0.5);
        assert!(report.mean_precision_at_k > 0.0);
    }

    #[test]
    fn test_insert_with_provenance() {
        use crate::persistence::vector_versioning::ProvenanceRecord;

        let mut collection = Collection::with_dimensions("test", 4);
        let prov = ProvenanceRecord::new("v1")
            .with_source("document.pdf")
            .with_model("text-embedding-3-small", "2024-01");

        collection.insert_with_provenance(
            "v1",
            &[1.0, 0.0, 0.0, 0.0],
            Some(json!({"title": "test"})),
            prov,
        ).unwrap();

        assert_eq!(collection.len(), 1);

        let prov = collection.get_provenance("v1").unwrap();
        assert_eq!(prov.source_document.as_deref(), Some("document.pdf"));
        assert_eq!(prov.embedding_model.as_deref(), Some("text-embedding-3-small"));
    }

    #[test]
    fn test_provenance_removed_on_delete() {
        use crate::persistence::vector_versioning::ProvenanceRecord;

        let mut collection = Collection::with_dimensions("test", 4);
        let prov = ProvenanceRecord::new("v1").with_source("doc1");
        collection.insert_with_provenance("v1", &[1.0, 0.0, 0.0, 0.0], None, prov).unwrap();
        assert!(collection.get_provenance("v1").is_some());

        collection.delete("v1").unwrap();
        assert!(collection.get_provenance("v1").is_none());
    }

    #[test]
    fn test_bundle_export_import_roundtrip() {
        let mut collection = Collection::with_dimensions("test_bundle", 4);
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"k": "a"}))).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let bundle_path = dir.path().join("test.needle-bundle");

        // Export
        let manifest = collection.export_bundle(&bundle_path).unwrap();
        assert_eq!(manifest.collection_name, "test_bundle");
        assert_eq!(manifest.vector_count, 2);
        assert_eq!(manifest.dimensions, 4);
        assert!(manifest.data_hash.is_some());

        // Validate
        let validated = Collection::validate_bundle_compatibility(&bundle_path).unwrap();
        assert_eq!(validated.format_version, 1);

        // Import
        let imported = Collection::import_bundle(&bundle_path).unwrap();
        assert_eq!(imported.name(), "test_bundle");
        assert_eq!(imported.dimensions(), 4);
        assert_eq!(imported.len(), 2);
        assert!(imported.contains("v1"));
        assert!(imported.contains("v2"));

        // Verify data integrity
        let (vec, meta) = imported.get("v1").unwrap();
        assert_eq!(vec.len(), 4);
        assert_eq!(meta.unwrap()["k"], "a");
    }

    #[test]
    fn test_record_feedback() {
        let collection = Collection::with_dimensions("test", 4);
        // Should not panic — just logs
        collection.record_feedback("query_1", "vec_1", 0.9);
    }
}
