//! Thread-safe collection reference for concurrent access.
//!
//! `CollectionRef` wraps a database reference and collection name to provide
//! a convenient and safe API for concurrent read/write operations.

use serde_json::Value;

use crate::collection::SearchResult;
use crate::distance::DistanceFunction;
use crate::error::Result;
use crate::metadata::Filter;

use super::{Database, ExportEntry};

/// Fluent builder for configuring searches on a [`CollectionRef`].
///
/// # Example
///
/// ```
/// use needle::{Database, Filter};
///
/// let db = Database::in_memory();
/// db.create_collection("docs", 4)?;
/// let coll = db.collection("docs")?;
/// coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None)?;
///
/// let results = coll.query(&[1.0, 0.0, 0.0, 0.0])
///     .limit(5)
///     .filter(&Filter::eq("category", "science"))
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
pub struct SearchParams<'a> {
    db: &'a Database,
    collection: &'a str,
    query: &'a [f32],
    k: usize,
    filter: Option<&'a Filter>,
    post_filter: Option<&'a Filter>,
    post_filter_factor: usize,
    distance_override: Option<DistanceFunction>,
}

impl<'a> SearchParams<'a> {
    /// Set the maximum number of results to return (default: 10).
    #[must_use]
    pub fn limit(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a pre-filter applied during ANN search.
    #[must_use]
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a post-filter applied after ANN search.
    #[must_use]
    pub fn post_filter(mut self, filter: &'a Filter) -> Self {
        self.post_filter = Some(filter);
        self
    }

    /// Set the over-fetch factor for post-filtering (default: 3).
    #[must_use]
    pub fn post_filter_factor(mut self, factor: usize) -> Self {
        self.post_filter_factor = factor;
        self
    }

    /// Override the distance function for this query.
    /// Falls back to brute-force search when different from the collection's default.
    #[must_use]
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_override = Some(distance);
        self
    }

    /// Execute the search and return results.
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        self.db.search_with_options_internal(
            self.collection,
            self.query,
            self.k,
            self.distance_override,
            self.filter,
            self.post_filter,
            self.post_filter_factor,
        )
    }
}

/// A thread-safe reference to a collection for concurrent access.
///
/// `CollectionRef` is the primary way to interact with collections when using
/// `Database`. It wraps collection operations with proper locking, enabling
/// safe concurrent read and write access from multiple threads.
///
/// # Thread Safety
///
/// - Multiple readers can access the collection simultaneously
/// - Writers get exclusive access (blocking other readers and writers)
/// - All operations are atomic at the method level
///
/// # Obtaining a CollectionRef
///
/// Use [`Database::collection()`] to get a reference:
///
/// ```
/// use needle::Database;
///
/// let db = Database::in_memory();
/// db.create_collection("embeddings", 128)?;
///
/// let collection = db.collection("embeddings")?;
/// // Use collection for insert, search, delete operations
/// # Ok::<(), needle::NeedleError>(())
/// ```
///
/// # Example: Concurrent Access
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use needle::Database;
///
/// let db = Arc::new(Database::in_memory());
/// db.create_collection("docs", 4).unwrap();
///
/// // Insert from main thread
/// let coll = db.collection("docs").unwrap();
/// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
///
/// // Search from multiple threads
/// let handles: Vec<_> = (0..4).map(|_| {
///     let db = Arc::clone(&db);
///     thread::spawn(move || {
///         let coll = db.collection("docs").unwrap();
///         coll.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap()
///     })
/// }).collect();
///
/// for h in handles {
///     let results = h.join().unwrap();
///     assert!(!results.is_empty());
/// }
/// ```
pub struct CollectionRef<'a> {
    db: &'a Database,
    name: String,
}

impl<'a> CollectionRef<'a> {
    /// Create a new collection reference (crate-internal).
    pub(super) fn new(db: &'a Database, name: String) -> Self {
        Self { db, name }
    }

    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of vectors
    pub fn len(&self) -> usize {
        self.db.collection_len(&self.name)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the vector dimensions
    pub fn dimensions(&self) -> Option<usize> {
        self.db.collection_dimensions(&self.name)
    }

    /// Get collection statistics (vector count, dimensions, memory usage, etc.)
    pub fn stats(&self) -> Result<crate::collection::CollectionStats> {
        self.db.collection_stats_internal(&self.name)
    }

    /// Insert a vector into the collection.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The vector data (must match collection dimensions)
    /// * `metadata` - Optional JSON metadata to associate with the vector
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::DimensionMismatch`] if the vector length doesn't match
    /// the collection's configured dimensions.
    ///
    /// Returns [`NeedleError::InvalidVector`] if the vector contains NaN or Infinity values.
    ///
    /// Returns [`NeedleError::DuplicateId`] if a vector with this ID already exists.
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_internal(&self.name, id, vector, metadata)
    }

    /// Insert a vector, taking ownership (more efficient when you have a Vec).
    ///
    /// This variant avoids an allocation when you already have a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Same as [`insert`](Self::insert).
    pub fn insert_vec(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_vec_internal(&self.name, id, vector, metadata)
    }

    /// Insert a vector with explicit TTL (time-to-live).
    ///
    /// The vector will automatically expire after the specified TTL.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The vector data
    /// * `metadata` - Optional JSON metadata
    /// * `ttl_seconds` - TTL in seconds; if `None`, uses collection default
    pub fn insert_with_ttl(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        self.db.insert_with_ttl_internal(&self.name, id, vector, metadata, ttl_seconds)
    }

    /// Insert a vector with TTL, taking ownership (more efficient).
    pub fn insert_vec_with_ttl(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        self.db.insert_vec_with_ttl_internal(&self.name, id, vector, metadata, ttl_seconds)
    }

    /// Search for the k most similar vectors to the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::DimensionMismatch`] if the query length doesn't match
    /// the collection's configured dimensions.
    ///
    /// Returns [`NeedleError::InvalidVector`] if the query contains NaN or Infinity values.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.db.search_internal(&self.name, query, k)
    }

    /// Create a fluent search builder for this collection.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Database, Filter};
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// let results = coll.query(&[1.0, 0.0, 0.0, 0.0])
    ///     .limit(5)
    ///     .execute()?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn query<'b>(&'b self, query: &'b [f32]) -> SearchParams<'b>
    where
        'a: 'b,
    {
        SearchParams {
            db: self.db,
            collection: &self.name,
            query,
            k: 10,
            filter: None,
            post_filter: None,
            post_filter_factor: 3,
            distance_override: None,
        }
    }

    /// Search for similar vectors with metadata filtering.
    ///
    /// Applies the filter before searching, potentially reducing the search space.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::DimensionMismatch`] if the query length doesn't match
    /// the collection's configured dimensions.
    ///
    /// Returns [`NeedleError::InvalidVector`] if the query contains NaN or Infinity values.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_with_filter_internal(&self.name, query, k, filter)
    }

    /// Search with full options including distance override, filters, and post-filter.
    ///
    /// This method provides access to all search options without using the builder pattern.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of results to return
    /// * `distance_override` - Optional distance function override (falls back to brute-force)
    /// * `filter` - Optional pre-filter (applied during ANN search)
    /// * `post_filter` - Optional post-filter (applied after ANN search)
    /// * `post_filter_factor` - Over-fetch factor for post-filtering (default: 3)
    pub fn search_with_options(
        &self,
        query: &[f32],
        k: usize,
        distance_override: Option<crate::DistanceFunction>,
        filter: Option<&Filter>,
        post_filter: Option<&Filter>,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db.search_with_options_internal(
            &self.name,
            query,
            k,
            distance_override,
            filter,
            post_filter,
            post_filter_factor,
        )
    }

    /// Search with detailed query execution profiling.
    ///
    /// Returns both the search results and a [`SearchExplain`](crate::SearchExplain)
    /// struct containing detailed timing and statistics.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let collection = db.collection("docs")?;
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// let (results, explain) = collection.search_explain(&[1.0, 0.0, 0.0, 0.0], 10)?;
    /// println!("Search took {}μs, visited {} nodes",
    ///          explain.total_time_us, explain.hnsw_stats.visited_nodes);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_explain(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        self.db.search_explain_internal(&self.name, query, k)
    }

    /// Search with metadata filter and detailed profiling.
    ///
    /// Combines filtered search with query execution profiling.
    pub fn search_with_filter_explain(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        self.db
            .search_with_filter_explain_internal(&self.name, query, k, filter)
    }

    /// Find all vectors within a given distance from the query.
    ///
    /// Unlike top-k search which returns exactly k results regardless of distance,
    /// range queries return all vectors within `max_distance` from the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let collection = db.collection("docs")?;
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// // Find all vectors within distance 0.5
    /// let results = collection.search_radius(&[1.0, 0.0, 0.0, 0.0], 0.5, 100)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_radius(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_radius_internal(&self.name, query, max_distance, limit)
    }

    /// Find all vectors within a given distance with metadata filtering.
    ///
    /// Combines range queries with metadata pre-filtering.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return
    /// * `filter` - MongoDB-style filter for metadata
    pub fn search_radius_with_filter(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_radius_with_filter_internal(&self.name, query, max_distance, limit, filter)
    }

    /// Retrieve a vector and its metadata by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The external ID of the vector to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some((vector, metadata))` if found, `None` otherwise.
    /// The vector data is cloned for thread-safety.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"title": "Hello"})))?;
    ///
    /// if let Some((vector, metadata)) = coll.get("v1") {
    ///     assert_eq!(vector, vec![1.0, 2.0, 3.0, 4.0]);
    ///     assert!(metadata.is_some());
    /// }
    ///
    /// assert!(coll.get("nonexistent").is_none());
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        self.db.get_internal(&self.name, id)
    }

    /// Delete a vector by its ID.
    ///
    /// Removes the vector from the index. Storage space is not immediately
    /// reclaimed; call [`compact()`](Self::compact) after many deletions.
    ///
    /// # Arguments
    ///
    /// * `id` - The external ID of the vector to delete
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the vector was deleted, `Ok(false)` if no vector
    /// with that ID existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(coll.delete("v1")?);       // Returns true
    /// assert!(!coll.delete("v1")?);      // Already deleted, returns false
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete(&self, id: &str) -> Result<bool> {
        self.db.delete_internal(&self.name, id)
    }

    /// Export all vectors from the collection.
    ///
    /// Returns all vectors with their IDs and metadata, useful for backup
    /// or migration purposes.
    ///
    /// # Returns
    ///
    /// A vector of `(id, vector, metadata)` tuples for all vectors in the collection.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"x": 1})))?;
    /// coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// let exported = coll.export_all()?;
    /// assert_eq!(exported.len(), 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn export_all(&self) -> Result<Vec<ExportEntry>> {
        self.db.export_internal(&self.name)
    }

    /// Get all vector IDs in the collection.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("doc1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// coll.insert("doc2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// let ids = coll.ids()?;
    /// assert!(ids.contains(&"doc1".to_string()));
    /// assert!(ids.contains(&"doc2".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn ids(&self) -> Result<Vec<String>> {
        self.db.ids_internal(&self.name)
    }

    /// Compact the collection, removing deleted vectors from storage.
    ///
    /// After many deletions, storage space is not immediately reclaimed.
    /// Calling `compact()` rebuilds internal structures to reclaim space.
    ///
    /// # Returns
    ///
    /// The number of deleted vectors that were removed from storage.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    ///
    /// // Insert and delete vectors
    /// for i in 0..100 {
    ///     coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
    /// }
    /// for i in 0..50 {
    ///     coll.delete(&format!("v{}", i))?;
    /// }
    ///
    /// // Reclaim storage space
    /// let removed = coll.compact()?;
    /// assert_eq!(removed, 50);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn compact(&self) -> Result<usize> {
        self.db.compact_internal(&self.name)
    }

    /// Check if the collection needs compaction.
    ///
    /// Returns `true` if the ratio of deleted vectors to total vectors exceeds
    /// the given threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The deleted/total ratio above which compaction is needed (0.0-1.0)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    ///
    /// for i in 0..10 {
    ///     coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
    /// }
    /// for i in 0..8 {
    ///     coll.delete(&format!("v{}", i))?;
    /// }
    ///
    /// // 8 deleted out of 10 = 80% deleted, so threshold 0.5 should trigger
    /// assert!(coll.needs_compaction(0.5));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.db.needs_compaction_internal(&self.name, threshold)
    }

    // ============ TTL Methods ============

    /// Sweep and delete all expired vectors.
    ///
    /// Returns the number of vectors that were expired and deleted.
    pub fn expire_vectors(&self) -> Result<usize> {
        self.db.expire_vectors_internal(&self.name)
    }

    /// Check if an expiration sweep is needed based on a threshold.
    ///
    /// Returns true if the ratio of expired vectors to total vectors
    /// exceeds the given threshold (0.0-1.0).
    pub fn needs_expiration_sweep(&self, threshold: f64) -> bool {
        self.db.needs_expiration_sweep_internal(&self.name, threshold)
    }

    /// Get TTL statistics for the collection.
    ///
    /// Returns (total_with_ttl, expired_count, earliest_expiration, latest_expiration).
    pub fn ttl_stats(&self) -> (usize, usize, Option<u64>, Option<u64>) {
        self.db.ttl_stats_internal(&self.name)
    }

    /// Get the expiration timestamp for a vector by external ID.
    pub fn get_ttl(&self, id: &str) -> Option<u64> {
        self.db.get_ttl_internal(&self.name, id)
    }

    /// Set or update the TTL for an existing vector.
    pub fn set_ttl(&self, id: &str, ttl_seconds: Option<u64>) -> Result<()> {
        self.db.set_ttl_internal(&self.name, id, ttl_seconds)
    }

    /// Count vectors in the collection, optionally matching a filter.
    ///
    /// # Arguments
    ///
    /// * `filter` - Optional metadata filter; if `None`, counts all vectors
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Database, Filter};
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"})))?;
    /// coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"type": "b"})))?;
    /// coll.insert("v3", &[0.0, 0.0, 1.0, 0.0], Some(json!({"type": "a"})))?;
    ///
    /// assert_eq!(coll.count(None)?, 3);
    ///
    /// let filter = Filter::eq("type", "a");
    /// assert_eq!(coll.count(Some(&filter))?, 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn count(&self, filter: Option<&Filter>) -> Result<usize> {
        self.db.count_internal(&self.name, filter)
    }

    /// Get the number of deleted vectors pending compaction.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// assert_eq!(coll.deleted_count(), 0);
    ///
    /// coll.delete("v1")?;
    /// assert_eq!(coll.deleted_count(), 1);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn deleted_count(&self) -> usize {
        self.db.deleted_count_internal(&self.name)
    }

    /// Search and return only IDs with distances (faster than full search).
    ///
    /// Skips metadata lookup, making this faster when you only need vector IDs.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity
    /// - [`NeedleError::CollectionNotFound`] - Collection no longer exists
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// coll.insert("v2", &[0.9, 0.1, 0.0, 0.0], None)?;
    ///
    /// let results = coll.search_ids(&[1.0, 0.0, 0.0, 0.0], 10)?;
    /// for (id, distance) in results {
    ///     println!("{}: {:.4}", id, distance);
    /// }
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_ids(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.db.search_ids_internal(&self.name, query, k)
    }

    /// Check if a vector with the given ID exists in the collection.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(coll.contains("v1"));
    /// assert!(!coll.contains("v2"));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Update an existing vector and its metadata.
    ///
    /// Replaces both the vector data and metadata. The vector is re-indexed
    /// after the update.
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
    /// - [`NeedleError::CollectionNotFound`] - Collection no longer exists
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"version": 1})))?;
    ///
    /// // Update the vector and metadata
    /// coll.update("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"version": 2})))?;
    ///
    /// let (vec, _meta) = coll.get("v1").unwrap();
    /// assert_eq!(vec, vec![0.0, 1.0, 0.0, 0.0]);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn update(
        &self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.update_internal(&self.name, id, vector, metadata)
    }

    /// Search with post-filter support.
    ///
    /// Post-filtering applies the filter after ANN search, which is useful when:
    /// - You need to guarantee k candidates before filtering
    /// - The filter involves expensive computation
    /// - The filter is highly selective and pre-filtering would miss results
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of results to return
    /// * `pre_filter` - Optional filter applied during ANN search (efficient)
    /// * `post_filter` - Filter applied after ANN search
    /// * `post_filter_factor` - Over-fetch factor (search fetches k * factor candidates)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Database, Filter};
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    ///
    /// // Insert vectors
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(serde_json::json!({"score": 10})))?;
    /// coll.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(serde_json::json!({"score": 20})))?;
    ///
    /// // Search with post-filter: find similar vectors, then filter by score
    /// let post_filter = Filter::gt("score", 15);
    /// let results = coll.search_with_post_filter(
    ///     &[1.0, 0.0, 0.0, 0.0],
    ///     10,
    ///     None,
    ///     &post_filter,
    ///     3,
    /// )?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_with_post_filter(
        &self,
        query: &[f32],
        k: usize,
        pre_filter: Option<&Filter>,
        post_filter: &Filter,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db.search_with_post_filter_internal(
            &self.name,
            query,
            k,
            pre_filter,
            post_filter,
            post_filter_factor,
        )
    }
}
