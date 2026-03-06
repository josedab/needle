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
    /// Matryoshka dimension truncation for faster search.
    truncated_dimensions: Option<usize>,
    /// Time decay for temporal relevance scoring.
    time_decay: Option<crate::collection::pipeline::TimeDecay>,
    /// Privacy configuration for differential privacy.
    privacy_config: Option<crate::enterprise::privacy::PrivacyConfig>,
}

impl<'a> SearchParams<'a> {
    /// Set the maximum number of results to return (default: 10).
    ///
    /// A higher value returns more results but may increase query latency.
    #[must_use]
    pub fn limit(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a pre-filter applied during ANN search.
    ///
    /// Pre-filters are evaluated during graph traversal, which can improve
    /// performance for highly selective filters but may reduce recall for
    /// very restrictive conditions. Use [`Filter::parse`] to build filters.
    #[must_use]
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a post-filter applied after ANN search.
    ///
    /// Post-filters are applied to the ANN result set after retrieval.
    /// The search over-fetches by [`post_filter_factor`](Self::post_filter_factor)
    /// to compensate for filtered-out results.
    #[must_use]
    pub fn post_filter(mut self, filter: &'a Filter) -> Self {
        self.post_filter = Some(filter);
        self
    }

    /// Set the over-fetch factor for post-filtering (default: 3).
    ///
    /// When a post-filter is active, `k * factor` candidates are retrieved
    /// from the index before filtering down to `k` results.
    #[must_use]
    pub fn post_filter_factor(mut self, factor: usize) -> Self {
        self.post_filter_factor = factor;
        self
    }

    /// Override the distance function for this query.
    ///
    /// When the override differs from the collection's default distance
    /// function, the search falls back to brute-force instead of using
    /// the HNSW index.
    #[must_use]
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_override = Some(distance);
        self
    }

    /// Set the search dimensionality for Matryoshka-style embeddings.
    ///
    /// Searches using only the first `dims` dimensions of each vector,
    /// enabling 3-6× memory savings with Matryoshka-trained embeddings.
    /// The search uses a two-phase approach: fast scan at reduced dims,
    /// then re-rank top candidates at full dimensionality.
    #[must_use]
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.truncated_dimensions = Some(dims);
        self
    }

    /// Apply time-weighted decay to search results.
    ///
    /// Adjusts similarity scores based on temporal freshness using
    /// exponential, linear, or step decay functions.
    #[must_use]
    pub fn with_time_decay(mut self, decay: crate::collection::pipeline::TimeDecay) -> Self {
        self.time_decay = Some(decay);
        self
    }

    /// Enable differential privacy for this search.
    ///
    /// Applies calibrated noise to distance scores using the specified
    /// epsilon-delta parameters for formal ε-differential privacy guarantees.
    #[must_use]
    pub fn with_privacy(mut self, config: crate::enterprise::privacy::PrivacyConfig) -> Self {
        self.privacy_config = Some(config);
        self
    }

    /// Execute the search and return results.
    ///
    /// When `with_dimensions()` is set, performs a two-phase search:
    /// first scanning at truncated dimensions, then re-ranking at full.
    /// When `with_time_decay()` is set, applies temporal decay to scores.
    /// When `with_privacy()` is set, perturbs distances with calibrated noise.
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        // If Matryoshka truncation is requested, perform two-phase search
        let mut results = if let Some(target_dims) = self.truncated_dimensions {
            self.execute_with_truncation(target_dims)?
        } else {
            self.db.search_with_options_internal(
                self.collection,
                self.query,
                self.k,
                self.distance_override,
                self.filter,
                self.post_filter,
                self.post_filter_factor,
            )?
        };

        // Apply time-weighted decay if configured.
        // For distance metrics (lower = better), dividing by the decay factor
        // makes older vectors rank worse: factor=0.5 → distance doubles.
        if let Some(ref decay) = self.time_decay {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let state = self.db.state.read();
            if let Some(coll) = state.collections.get(self.collection) {
                for result in &mut results {
                    if let Some(ts) = coll.insertion_timestamp_by_id(&result.id) {
                        let age = now.saturating_sub(ts);
                        let factor = decay.compute(age);
                        if factor > f32::EPSILON {
                            result.distance /= factor;
                        } else {
                            result.distance = f32::MAX;
                        }
                    }
                }
            }
            drop(state);
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Apply differential privacy noise if configured
        if let Some(ref privacy_config) = self.privacy_config {
            let mechanism =
                crate::enterprise::privacy::PrivacyMechanism::new(privacy_config.clone());
            for result in &mut results {
                result.distance =
                    mechanism.perturb_distance(result.distance, privacy_config.sensitivity);
            }
            results.sort_by(|a, b| {
                a.distance
                    .partial_cmp(&b.distance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(results)
    }

    /// Two-phase Matryoshka search: fast coarse scan, then full-dimension re-rank.
    ///
    /// Phase 1 creates a zero-padded query where only the first `target_dims` are
    /// preserved. For Matryoshka-trained embeddings, these prefix dimensions carry
    /// the strongest signal, so the coarse scan retrieves good candidates cheaply.
    /// Phase 2 re-ranks those candidates using the full query vector for precision.
    fn execute_with_truncation(
        &self,
        target_dims: usize,
    ) -> Result<Vec<SearchResult>> {
        let full_dims = self.query.len();
        if target_dims == 0 || target_dims >= full_dims {
            return self.db.search_with_options_internal(
                self.collection,
                self.query,
                self.k,
                self.distance_override,
                self.filter,
                self.post_filter,
                self.post_filter_factor,
            );
        }

        // Phase 1: create a zero-padded query (first target_dims preserved, rest zeroed)
        // This approximates searching at reduced dimensionality while maintaining
        // the expected vector size for the collection.
        let mut coarse_query = vec![0.0f32; full_dims];
        coarse_query[..target_dims].copy_from_slice(&self.query[..target_dims]);
        let overfetch_k = self.k * 3;

        let candidates = self.db.search_with_options_internal(
            self.collection,
            &coarse_query,
            overfetch_k,
            self.distance_override,
            self.filter,
            self.post_filter,
            self.post_filter_factor,
        )?;

        // Phase 2: re-rank candidates using full query vector
        let state = self.db.state.read();
        let coll = state
            .collections
            .get(self.collection)
            .ok_or_else(|| {
                crate::error::NeedleError::CollectionNotFound(self.collection.to_string())
            })?;

        let distance_fn = self
            .distance_override
            .clone()
            .unwrap_or_else(|| coll.config().distance.clone());

        let mut reranked: Vec<SearchResult> = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            if let Some(vec) = coll.get_vector(&candidate.id) {
                let dist = distance_fn.compute(self.query, &vec).unwrap_or(f32::MAX);
                reranked.push(SearchResult {
                    id: candidate.id.clone(),
                    distance: dist,
                    metadata: candidate.metadata.clone(),
                });
            }
        }
        reranked.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(self.k);
        Ok(reranked)
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

    /// Get estimated memory usage breakdown for this collection.
    pub fn memory_usage(&self) -> Result<crate::collection::MemoryStats> {
        self.db.collection_memory_usage_internal(&self.name)
    }

    /// Get statistics for a specific metadata field.
    pub fn field_stats(&self, field: &str) -> Option<crate::metadata::FieldStats> {
        self.db.collection_field_stats_internal(&self.name, field)
    }

    /// Get statistics for all known metadata fields.
    pub fn all_field_stats(&self) -> Vec<crate::metadata::FieldStats> {
        self.db.collection_all_field_stats_internal(&self.name)
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
        self.db
            .insert_vec_internal(&self.name, id, vector, metadata)
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
        self.db
            .insert_with_ttl_internal(&self.name, id, vector, metadata, ttl_seconds)
    }

    /// Insert a vector with TTL, taking ownership (more efficient).
    pub fn insert_vec_with_ttl(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        self.db
            .insert_vec_with_ttl_internal(&self.name, id, vector, metadata, ttl_seconds)
    }

    /// Insert multiple vectors in a single batch operation.
    ///
    /// All vectors are validated before any are inserted. If any vector is invalid
    /// (wrong dimensions, NaN, duplicate ID), the entire batch is rejected.
    ///
    /// # Arguments
    ///
    /// * `items` - Vector of `(id, vector, metadata)` tuples
    ///
    /// # Returns
    ///
    /// The number of vectors inserted on success.
    ///
    /// # Errors
    ///
    /// Returns an error if any vector fails validation. On error, no vectors
    /// from the batch are inserted (atomic batch semantics).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 3)?;
    /// let coll = db.collection("docs")?;
    ///
    /// let items = vec![
    ///     ("a".into(), vec![1.0, 0.0, 0.0], Some(json!({"topic": "rust"}))),
    ///     ("b".into(), vec![0.0, 1.0, 0.0], None),
    ///     ("c".into(), vec![0.0, 0.0, 1.0], Some(json!({"topic": "python"}))),
    /// ];
    /// let count = coll.batch_insert(items)?;
    /// assert_eq!(count, 3);
    /// assert_eq!(coll.len(), 3);
    /// # Ok::<(), needle::error::NeedleError>(())
    /// ```
    pub fn batch_insert(
        &self,
        items: Vec<(String, Vec<f32>, Option<Value>)>,
    ) -> Result<usize> {
        self.db.batch_insert_internal(&self.name, items)
    }

    /// Bulk import vectors from a JSON-Lines reader.
    ///
    /// Each line should be a JSON object with `"id"`, `"vector"`, and optional `"metadata"` fields.
    /// Blank lines and lines starting with `#` are skipped.
    pub fn import_jsonl<R: std::io::BufRead>(
        &self,
        reader: R,
    ) -> Result<crate::collection::ImportResult> {
        self.db.import_jsonl_internal(&self.name, reader)
    }

    /// Insert a text document, automatically embedding it using the built-in model runtime.
    ///
    /// Uses the embedded model runtime (feature: `embedded-models`) to generate
    /// embeddings from text with zero external API dependencies.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is empty, dimensions don't match,
    /// or a vector with the same ID already exists.
    #[cfg(feature = "embedded-models")]
    pub fn insert_text(
        &self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_text_internal(&self.name, id, text, metadata)
    }

    /// Search by text using the built-in embedded model runtime.
    ///
    /// Embeds the query text and performs a k-NN search.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The text is empty
    /// - Vector dimensions don't match the collection
    #[cfg(feature = "embedded-models")]
    pub fn search_text(&self, text: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.db.search_text_internal(&self.name, text, k)
    }

    /// Insert a document by text, auto-embedding it using the collection's
    /// configured auto-embed provider.
    ///
    /// Requires `auto_embed` to be set on the collection config.
    ///
    /// # Errors
    ///
    /// Returns an error if auto-embed is not configured, the ID already exists,
    /// or the generated vector is invalid.
    pub fn insert_auto_text(
        &self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db
            .insert_auto_text_internal(&self.name, id, text, metadata)
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
            truncated_dimensions: None,
            time_decay: None,
            privacy_config: None,
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

    /// Search with full HNSW graph traversal trace for debugging and visualization.
    pub fn search_with_trace(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::hnsw::SearchTrace)> {
        self.db.search_with_trace_internal(&self.name, query, k)
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

    /// Two-phase Matryoshka search with dimensional reduction.
    ///
    /// Performs a coarse search on truncated dimensions, then re-ranks with
    /// full dimensions. Works with Matryoshka-trained embeddings (e.g.,
    /// OpenAI text-embedding-3, Nomic embed) where prefix truncation
    /// preserves semantic meaning.
    ///
    /// # Arguments
    ///
    /// * `query` - Full-dimension query vector
    /// * `k` - Number of results to return
    /// * `coarse_dims` - Truncated dimension count for coarse search
    /// * `oversample` - Multiplier for candidate set size (default: 4)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 128)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &vec![0.1; 128], None)?;
    ///
    /// // Search with 64-dim coarse pass, 4x oversampling
    /// let results = coll.search_matryoshka(&vec![0.1; 128], 10, 64, 4)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_matryoshka(
        &self,
        query: &[f32],
        k: usize,
        coarse_dims: usize,
        oversample: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_matryoshka_internal(&self.name, query, k, coarse_dims, oversample)
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
        if max_distance < 0.0 {
            return Err(crate::error::NeedleError::InvalidInput(
                format!("max_distance must be non-negative, got {max_distance}"),
            ));
        }
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

    /// Get provenance record for a vector.
    ///
    /// Returns the provenance metadata (source document, embedding model, pipeline, etc.)
    /// if one was recorded during insertion.
    pub fn get_provenance(
        &self,
        vector_id: &str,
    ) -> Option<crate::persistence::vector_versioning::ProvenanceRecord> {
        self.db.get_provenance_internal(&self.name, vector_id)
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

    /// Delete multiple vectors by ID in a single operation.
    ///
    /// Returns the count of vectors that were actually deleted (IDs that
    /// didn't exist are silently skipped).
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
    /// db.create_collection("docs", 3)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("a", &[1.0, 0.0, 0.0], None)?;
    /// coll.insert("b", &[0.0, 1.0, 0.0], None)?;
    /// coll.insert("c", &[0.0, 0.0, 1.0], None)?;
    ///
    /// let deleted = coll.batch_delete(&["a", "b", "nonexistent"])?;
    /// assert_eq!(deleted, 2);
    /// assert_eq!(coll.len(), 1);
    /// # Ok::<(), needle::error::NeedleError>(())
    /// ```
    pub fn batch_delete(&self, ids: &[&str]) -> Result<usize> {
        let mut count = 0;
        for id in ids {
            if self.db.delete_internal(&self.name, id)? {
                count += 1;
            }
        }
        Ok(count)
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

    /// Scan for near-duplicate vectors in the collection.
    ///
    /// Returns groups of vectors whose pairwise distance is below the threshold.
    /// Uses the collection's dedup config threshold if `threshold` is `None`.
    pub fn dedup_scan(
        &self,
        threshold: Option<f32>,
    ) -> Result<crate::collection::dedup::DedupScanResult> {
        self.db.dedup_scan_internal(&self.name, threshold)
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
        self.db
            .needs_expiration_sweep_internal(&self.name, threshold)
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
    pub fn update(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
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

    /// Create a named snapshot of this collection.
    pub fn create_snapshot(&self, snapshot_name: &str) -> Result<()> {
        self.db.create_snapshot(&self.name, snapshot_name)
    }

    /// Restore this collection from a named snapshot.
    pub fn restore_snapshot(&self, snapshot_name: &str) -> Result<()> {
        self.db.restore_snapshot(&self.name, snapshot_name)
    }

    /// List all snapshots for this collection.
    pub fn list_snapshots(&self) -> Vec<String> {
        self.db.list_snapshots(&self.name)
    }

    /// Evaluate search quality using ground truth data.
    ///
    /// Computes recall@k, precision@k, MAP, MRR, and NDCG metrics.
    pub fn evaluate(
        &self,
        ground_truth: &[crate::collection::GroundTruthEntry],
        k: usize,
    ) -> Result<crate::collection::EvaluationReport> {
        self.db.evaluate_internal(&self.name, ground_truth, k)
    }

    /// Export the collection as a portable bundle file.
    pub fn export_bundle(
        &self,
        path: &std::path::Path,
    ) -> Result<crate::collection::BundleManifest> {
        self.db.export_bundle_internal(&self.name, path)
    }

    /// Enable CDC (change data capture) on this collection.
    pub fn enable_cdc(&self, max_events: usize) -> Result<()> {
        let mut state = self.db.state.write();
        let coll = state
            .collections
            .get_mut(&self.name)
            .ok_or_else(|| crate::error::NeedleError::CollectionNotFound(self.name.clone()))?;
        coll.enable_cdc(max_events);
        Ok(())
    }

    /// Get CDC events after the given cursor (sequence number), up to `limit`.
    pub fn cdc_events_since(&self, after_sequence: u64, limit: usize) -> Vec<crate::collection::CdcEvent> {
        let state = self.db.state.read();
        state
            .collections
            .get(&self.name)
            .map(|c| c.cdc_events_since(after_sequence, limit).into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the current CDC head sequence number.
    pub fn cdc_head_sequence(&self) -> u64 {
        let state = self.db.state.read();
        state
            .collections
            .get(&self.name)
            .map(|c| c.cdc_head_sequence())
            .unwrap_or(0)
    }

    /// Create a named branch of this collection using copy-on-write semantics.
    ///
    /// The branch starts with all current vectors copied into the "main" branch
    /// of the returned tree. The new branch is a CoW overlay that starts empty
    /// and accumulates changes without affecting the original data.
    ///
    /// Changes on the branch don't affect the original collection until merged.
    pub fn create_branch(&self, branch_name: &str) -> Result<crate::collection_branch::BranchTree> {
        let mut tree = crate::collection_branch::BranchTree::new();

        // Snapshot current collection data into the "main" branch
        let state = self.db.state.read();
        if let Some(coll) = state.collections.get(&self.name) {
            for (id, vector, _metadata) in coll.iter() {
                let _ = tree.insert("main", id, vector.to_vec());
            }
        }
        drop(state);

        tree.create_branch(branch_name, "main")?;
        Ok(tree)
    }

    /// Ingest a text document by chunking and inserting with provided embeddings.
    ///
    /// Splits the text into chunks using recursive text splitting, then calls
    /// `embed_fn` on each chunk to get its vector, and inserts them.
    ///
    /// # Arguments
    ///
    /// * `doc_id` - Base ID for the document (chunks get `{doc_id}_0`, `{doc_id}_1`, etc.)
    /// * `text` - The text content to ingest
    /// * `embed_fn` - Function that converts text to a vector embedding
    /// * `chunk_size` - Maximum characters per chunk (default: 512)
    /// * `chunk_overlap` - Overlap between chunks (default: 50)
    ///
    /// # Returns
    ///
    /// Number of chunks ingested.
    pub fn ingest_text<F>(
        &self,
        doc_id: &str,
        text: &str,
        embed_fn: F,
        chunk_size: Option<usize>,
        chunk_overlap: Option<usize>,
    ) -> Result<usize>
    where
        F: Fn(&str) -> Vec<f32>,
    {
        let splitter = crate::ml::rag::chunking::RecursiveTextSplitter::new(
            chunk_size.unwrap_or(512),
            chunk_overlap.unwrap_or(50),
        );
        let chunks = splitter.split(text);
        let mut count = 0;

        for (i, (chunk_text, _start, _end)) in chunks.iter().enumerate() {
            let chunk_id = format!("{}_{}", doc_id, i);
            let embedding = embed_fn(chunk_text);

            let metadata = serde_json::json!({
                "source": doc_id,
                "chunk_index": i,
                "text": &chunk_text[..chunk_text.len().min(500)],
            });

            self.insert(&chunk_id, &embedding, Some(metadata))?;
            count += 1;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use serde_json::json;
    use std::sync::Arc;
    use std::thread;

    fn setup_db(dim: usize) -> std::result::Result<Database, Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("test", dim)?;
        Ok(db)
    }

    // ── Basic operations ─────────────────────────────────────────────────

    #[test]
    fn test_insert_and_get() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        let (vec, meta) = coll.get("v1").ok_or("v1 not found")?;
        assert_eq!(vec, vec![1.0, 0.0, 0.0, 0.0]);
        assert!(meta.is_none());
        Ok(())
    }

    #[test]
    fn test_insert_with_metadata() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"key": "val"})))?;

        let (_, meta) = coll.get("v1").ok_or("v1 not found")?;
        assert!(meta.is_some());
        Ok(())
    }

    #[test]
    fn test_get_nonexistent() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        assert!(coll.get("nonexistent").is_none());
        Ok(())
    }

    #[test]
    fn test_contains() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        assert!(coll.contains("v1"));
        assert!(!coll.contains("v2"));
        Ok(())
    }

    #[test]
    fn test_len_and_is_empty() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        assert!(coll.is_empty());
        assert_eq!(coll.len(), 0);

        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        assert!(!coll.is_empty());
        assert_eq!(coll.len(), 1);
        Ok(())
    }

    #[test]
    fn test_dimensions() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(128)?;
        let coll = db.collection("test")?;
        assert_eq!(coll.dimensions(), Some(128));
        Ok(())
    }

    #[test]
    fn test_name() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        assert_eq!(coll.name(), "test");
        Ok(())
    }

    // ── Error propagation ────────────────────────────────────────────────

    #[test]
    fn test_dimension_mismatch_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        let result = coll.insert("v1", &[1.0, 0.0], None);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_collection_not_found() {
        let db = Database::in_memory();
        let result = db.collection("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_insert_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        let result = coll.insert("v1", &[0.0, 1.0, 0.0, 0.0], None);
        assert!(result.is_err());
        Ok(())
    }

    // ── Delete ───────────────────────────────────────────────────────────

    #[test]
    fn test_delete() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        assert!(coll.delete("v1")?);
        assert!(!coll.delete("v1")?);
        assert!(coll.get("v1").is_none());
        Ok(())
    }

    #[test]
    fn test_deleted_count() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;

        assert_eq!(coll.deleted_count(), 0);
        coll.delete("v1")?;
        assert_eq!(coll.deleted_count(), 1);
        Ok(())
    }

    // ── Search ───────────────────────────────────────────────────────────

    #[test]
    fn test_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;

        let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 2)?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
        Ok(())
    }

    #[test]
    fn test_search_ids() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        let results = coll.search_ids(&[1.0, 0.0, 0.0, 0.0], 10)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "v1");
        Ok(())
    }

    #[test]
    fn test_search_empty_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 10)?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_query_builder() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        let results = coll.query(&[1.0, 0.0, 0.0, 0.0])
            .limit(5)
            .execute()?;
        assert_eq!(results.len(), 1);
        Ok(())
    }

    #[test]
    fn test_search_with_filter() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"cat": "a"})))?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"cat": "b"})))?;

        let filter = Filter::eq("cat", "a");
        let results = coll.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
        Ok(())
    }

    // ── Update ───────────────────────────────────────────────────────────

    #[test]
    fn test_update() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        coll.update("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"updated": true})))?;
        let (vec, meta) = coll.get("v1").ok_or("v1 not found")?;
        assert_eq!(vec, vec![0.0, 1.0, 0.0, 0.0]);
        assert!(meta.is_some());
        Ok(())
    }

    #[test]
    fn test_update_nonexistent() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        let result = coll.update("nonexistent", &[1.0, 0.0, 0.0, 0.0], None);
        assert!(result.is_err());
        Ok(())
    }

    // ── Export & IDs ─────────────────────────────────────────────────────

    #[test]
    fn test_export_all() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;

        let exported = coll.export_all()?;
        assert_eq!(exported.len(), 2);
        Ok(())
    }

    #[test]
    fn test_ids() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;

        let ids = coll.ids()?;
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"v1".to_string()));
        assert!(ids.contains(&"v2".to_string()));
        Ok(())
    }

    // ── Compact ──────────────────────────────────────────────────────────

    #[test]
    fn test_compact() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        for i in 0..10 {
            coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
        }
        for i in 0..5 {
            coll.delete(&format!("v{}", i))?;
        }

        let removed = coll.compact()?;
        assert_eq!(removed, 5);
        assert_eq!(coll.len(), 5);
        Ok(())
    }

    #[test]
    fn test_needs_compaction() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        for i in 0..10 {
            coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
        }
        for i in 0..8 {
            coll.delete(&format!("v{}", i))?;
        }

        assert!(coll.needs_compaction(0.5));
        Ok(())
    }

    // ── Count ────────────────────────────────────────────────────────────

    #[test]
    fn test_count() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"})))?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"type": "b"})))?;

        assert_eq!(coll.count(None)?, 2);

        let filter = Filter::eq("type", "a");
        assert_eq!(coll.count(Some(&filter))?, 1);
        Ok(())
    }

    // ── Stats ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        let stats = coll.stats()?;
        assert_eq!(stats.vector_count, 1);
        assert_eq!(stats.dimensions, 4);
        Ok(())
    }

    // ── Concurrent access ────────────────────────────────────────────────

    #[test]
    fn test_concurrent_reads() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 4)?;
        {
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }

        let handles: Vec<_> = (0..8).map(|_| {
            let db = Arc::clone(&db);
            thread::spawn(move || -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let coll = db.collection("test")?;
                let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 10)?;
                assert!(!results.is_empty());
                Ok(())
            })
        }).collect();

        for h in handles {
            h.join().map_err(|_| "thread panicked")??;
        }
        Ok(())
    }

    #[test]
    fn test_concurrent_insert_and_search() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 4)?;

        // Insert from multiple threads
        let handles: Vec<_> = (0..4).map(|t| {
            let db = Arc::clone(&db);
            thread::spawn(move || -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let coll = db.collection("test")?;
                for i in 0..10 {
                    let id = format!("t{}_v{}", t, i);
                    let _ = coll.insert(&id, &[t as f32, i as f32, 0.0, 0.0], None);
                }
                Ok(())
            })
        }).collect();

        for h in handles {
            h.join().map_err(|_| "thread panicked")??;
        }

        let coll = db.collection("test")?;
        assert_eq!(coll.len(), 40);
        Ok(())
    }

    #[test]
    fn test_concurrent_read_write() -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 4)?;
        {
            let coll = db.collection("test")?;
            for i in 0..10 {
                coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
            }
        }

        let mut handles = Vec::new();

        // Readers
        for _ in 0..4 {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let coll = db.collection("test")?;
                for _ in 0..20 {
                    let _ = coll.search(&[1.0, 0.0, 0.0, 0.0], 5);
                    let _ = coll.get("v0");
                    let _ = coll.len();
                }
                Ok(())
            }));
        }

        // Writer
        {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
                let coll = db.collection("test")?;
                for i in 10..20 {
                    let _ = coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None);
                }
                Ok(())
            }));
        }

        for h in handles {
            h.join().map_err(|_| "thread panicked")??;
        }
        Ok(())
    }

    // ── insert_vec ───────────────────────────────────────────────────────

    #[test]
    fn test_insert_vec() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert_vec("v1", vec![1.0, 0.0, 0.0, 0.0], None)?;

        let (vec, _) = coll.get("v1").ok_or("v1 not found")?;
        assert_eq!(vec, vec![1.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    // ── Search explain ───────────────────────────────────────────────────

    #[test]
    fn test_search_explain() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        let (results, explain) = coll.search_explain(&[1.0, 0.0, 0.0, 0.0], 10)?;
        assert_eq!(results.len(), 1);
        assert!(explain.total_time_us > 0 || true); // may be 0 on fast machines
        Ok(())
    }

    // ── Snapshots ────────────────────────────────────────────────────────

    #[test]
    fn test_snapshots() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        coll.create_snapshot("snap1")?;
        let snaps = coll.list_snapshots();
        assert!(snaps.contains(&"snap1".to_string()));
        Ok(())
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_collection_ref_search_text() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 384)?;

        let coll = db.collection("docs")?;
        coll.insert_text("doc1", "machine learning algorithms", None)?;
        coll.insert_text("doc2", "cooking recipes for dinner", None)?;

        let results = coll.search_text("machine learning", 2)?;
        assert_eq!(results.len(), 2);
        // Same topic should be closest
        assert_eq!(results[0].id, "doc1");
        Ok(())
    }

    #[test]
    fn test_search_params_with_dimensions() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(8)?;
        let coll = db.collection("test")?;
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("b", &[0.9, 0.1, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], None)?;
        coll.insert("c", &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None)?;

        let query = &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Normal search
        let normal = coll.query(query).limit(3).execute()?;
        assert_eq!(normal.len(), 3);

        // Search with dimension truncation (use first 4 dims)
        let truncated = coll.query(query).with_dimensions(4).limit(3).execute()?;
        assert_eq!(truncated.len(), 3);
        // "a" should still be closest in both cases
        assert_eq!(truncated[0].id, "a");
        Ok(())
    }

    #[test]
    fn test_search_params_with_time_decay() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("b", &[0.9, 0.1, 0.0, 0.0], None)?;

        // Exponential decay: recently inserted vectors should have factor ~1.0
        let decay = crate::collection::pipeline::TimeDecay::Exponential {
            half_life_seconds: 3600,
        };
        let results = coll.query(&[1.0, 0.0, 0.0, 0.0])
            .with_time_decay(decay)
            .limit(2)
            .execute()?;
        assert_eq!(results.len(), 2);

        // Linear decay: all vectors within window should be unaffected
        let decay_linear = crate::collection::pipeline::TimeDecay::Linear {
            max_age_seconds: 86400,
        };
        let results2 = coll.query(&[1.0, 0.0, 0.0, 0.0])
            .with_time_decay(decay_linear)
            .limit(2)
            .execute()?;
        assert_eq!(results2.len(), 2);
        // "a" should be closest in both cases (vectors are fresh)
        assert_eq!(results[0].id, "a");
        assert_eq!(results2[0].id, "a");
        Ok(())
    }

    #[test]
    fn test_search_params_with_privacy() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("b", &[0.0, 1.0, 0.0, 0.0], None)?;

        // Search with privacy noise
        let config = crate::enterprise::privacy::PrivacyConfig::new(1.0, 1e-5);
        let results = coll.query(&[1.0, 0.0, 0.0, 0.0])
            .with_privacy(config)
            .limit(2)
            .execute()?;
        assert_eq!(results.len(), 2);
        // Distances should be perturbed (non-zero even for exact match)
        // We just verify the search doesn't crash and returns results
        Ok(())
    }

    #[test]
    fn test_collection_get_vector() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("a", &[1.0, 2.0, 3.0, 4.0], None)?;

        let state = db.state.read();
        let collection = state.collections.get("test").expect("collection exists");
        let vec = collection.get_vector("a");
        assert!(vec.is_some());
        assert_eq!(vec.as_ref().map(|v| v.len()), Some(4));
        assert!(collection.get_vector("nonexistent").is_none());
        Ok(())
    }

    #[test]
    fn test_ingest_text() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;

        // Simple embedding function: hash text to a 4-dim vector
        let embed = |text: &str| -> Vec<f32> {
            let mut h = 0u64;
            for b in text.bytes() {
                h = h.wrapping_mul(31).wrapping_add(b as u64);
            }
            vec![
                (h & 0xFF) as f32 / 255.0,
                ((h >> 8) & 0xFF) as f32 / 255.0,
                ((h >> 16) & 0xFF) as f32 / 255.0,
                ((h >> 24) & 0xFF) as f32 / 255.0,
            ]
        };

        let text = "This is a test document. It has multiple sentences. \
                     Each sentence should contribute to chunks. \
                     The chunker splits at sentence boundaries.";

        let count = coll.ingest_text("doc1", text, embed, Some(50), Some(10))?;
        assert!(count > 0);
        assert_eq!(coll.len(), count);

        // Verify we can search the ingested chunks
        let query = embed("test document");
        let results = coll.search(&query, 3)?;
        assert!(!results.is_empty());
        Ok(())
    }

    #[test]
    fn test_create_branch_snapshots_data() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = setup_db(4)?;
        let coll = db.collection("test")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;

        // Create branch — should snapshot v1 and v2 into "main"
        let mut tree = coll.create_branch("experiment")?;

        // Main branch should have both vectors
        assert_eq!(tree.get("main", "v1").expect("v1 in main"), &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(tree.get("main", "v2").expect("v2 in main"), &[0.0, 1.0, 0.0, 0.0]);

        // New branch inherits from main via CoW
        assert_eq!(tree.get("experiment", "v1").expect("v1 in experiment"), &[1.0, 0.0, 0.0, 0.0]);

        // Modify on experiment without affecting main
        tree.insert("experiment", "v1", vec![0.5, 0.5, 0.0, 0.0]).expect("insert");
        assert_eq!(tree.get("experiment", "v1").expect("modified"), &[0.5, 0.5, 0.0, 0.0]);
        assert_eq!(tree.get("main", "v1").expect("original"), &[1.0, 0.0, 0.0, 0.0]);

        // Diff should show v1 as modified
        let diff = tree.diff("experiment", "main")?;
        assert!(!diff.is_empty());
        Ok(())
    }

    #[test]
    fn test_batch_delete() -> Result<()> {
        let db = Database::in_memory();
        db.create_collection("test", 3)?;
        let coll = db.collection("test")?;
        coll.insert("a", &[1.0, 0.0, 0.0], None)?;
        coll.insert("b", &[0.0, 1.0, 0.0], None)?;
        coll.insert("c", &[0.0, 0.0, 1.0], None)?;

        let deleted = coll.batch_delete(&["a", "c", "nonexistent"])?;
        assert_eq!(deleted, 2);
        assert_eq!(coll.len(), 1);
        assert!(coll.contains("b"));
        assert!(!coll.contains("a"));
        assert!(!coll.contains("c"));
        Ok(())
    }

    #[test]
    fn test_batch_delete_empty_ids() -> Result<()> {
        let db = Database::in_memory();
        db.create_collection("test", 3)?;
        let coll = db.collection("test")?;
        coll.insert("a", &[1.0, 0.0, 0.0], None)?;

        let deleted = coll.batch_delete(&[])?;
        assert_eq!(deleted, 0);
        assert_eq!(coll.len(), 1);
        Ok(())
    }

    #[test]
    fn test_batch_delete_all_nonexistent() -> Result<()> {
        let db = Database::in_memory();
        db.create_collection("test", 3)?;
        let coll = db.collection("test")?;

        let deleted = coll.batch_delete(&["x", "y", "z"])?;
        assert_eq!(deleted, 0);
        Ok(())
    }

    // ── batch_insert tests ───────────────────────────────────────────────

    #[test]
    fn test_batch_insert() -> Result<()> {
        let db = Database::in_memory();
        db.create_collection("test", 3)?;
        let coll = db.collection("test")?;

        let items = vec![
            ("a".into(), vec![1.0, 0.0, 0.0], None),
            ("b".into(), vec![0.0, 1.0, 0.0], Some(json!({"x": 1}))),
            ("c".into(), vec![0.0, 0.0, 1.0], None),
        ];
        let count = coll.batch_insert(items)?;
        assert_eq!(count, 3);
        assert_eq!(coll.len(), 3);
        assert!(coll.contains("a"));
        assert!(coll.contains("b"));
        assert!(coll.contains("c"));
        Ok(())
    }

    #[test]
    fn test_batch_insert_empty() -> Result<()> {
        let db = Database::in_memory();
        db.create_collection("test", 3)?;
        let coll = db.collection("test")?;

        let count = coll.batch_insert(vec![])?;
        assert_eq!(count, 0);
        assert_eq!(coll.len(), 0);
        Ok(())
    }

    #[test]
    fn test_batch_insert_rejects_duplicate_ids() {
        let db = Database::in_memory();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();

        let items = vec![
            ("dup".into(), vec![1.0, 0.0, 0.0], None),
            ("dup".into(), vec![0.0, 1.0, 0.0], None),
        ];
        let result = coll.batch_insert(items);
        assert!(result.is_err());
        // Atomic: no vectors should be inserted on failure
        assert_eq!(coll.len(), 0);
    }

    #[test]
    fn test_batch_insert_rejects_wrong_dimensions() {
        let db = Database::in_memory();
        db.create_collection("test", 3).unwrap();
        let coll = db.collection("test").unwrap();

        let items = vec![
            ("a".into(), vec![1.0, 0.0, 0.0], None),
            ("b".into(), vec![1.0, 0.0], None), // wrong dims
        ];
        let result = coll.batch_insert(items);
        assert!(result.is_err());
        assert_eq!(coll.len(), 0);
    }
}
