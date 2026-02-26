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
pub mod dedup;
pub use dedup::*;
pub mod pipeline;
pub use pipeline::*;
mod sharding;
use sharding::*;
mod validation;
mod insert;
mod mutations;
mod batch;
mod ttl;
mod bundle;
mod cache;
mod search_methods;
mod accessors;
pub use accessors::CollectionIter;
/// Collection-level change data capture with sequence numbers and cursor resumption.
pub mod cdc;
pub use cdc::{CdcConfig, CdcEvent, CdcEventType, CdcLog};

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
use std::sync::Arc;
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
    /// CDC event log for change data capture
    #[serde(default)]
    cdc_log: Option<CdcLog>,
    /// Cached embedded model runtime for text-to-vector operations.
    /// Feature-gated behind `embedded-models`. Shared across insert_text/search_text calls.
    /// Uses Box<dyn Any> to avoid requiring Debug on EmbeddingRuntime.
    #[cfg(feature = "embedded-models")]
    #[serde(skip)]
    embedded_runtime: Option<Arc<crate::ml::embedded_runtime::EmbeddingRuntime>>,
}

/// Manifest for a portable collection bundle.
///
/// Contains metadata about the bundled collection for validation during import.
/// The `.needle-bundle` format includes data + index + schema + manifest with
/// semver-based format versioning for forward/backward compatibility.
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
    /// Semantic version string (e.g., "1.0.0") for bundle format compatibility.
    #[serde(default = "default_bundle_semver")]
    pub semver: String,
    /// Optional description of the bundle contents.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional registry URI for pull-based distribution (e.g., "needle://registry/datasets/my-embeddings:v1").
    #[serde(default)]
    pub registry_uri: Option<String>,
    /// Optional tags for discoverability.
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_bundle_semver() -> String {
    "1.0.0".to_string()
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
            cdc_log: None,
            #[cfg(feature = "embedded-models")]
            embedded_runtime: None,
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

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vector;
    use serde_json::json;

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
