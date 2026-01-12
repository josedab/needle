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
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

/// Builder for configuring and executing searches
#[derive(Clone)]
pub struct SearchBuilder<'a> {
    collection: &'a Collection,
    query: &'a [f32],
    k: usize,
    filter: Option<&'a Filter>,
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
            ef_search: None,
            include_metadata: true,
        }
    }

    /// Set the number of results to return
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a metadata filter
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
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

        // Get raw results with optional ef_search override
        let raw_results = if let Some(ef) = self.ef_search {
            self.collection.index.search_with_ef(
                self.query,
                self.k * if self.filter.is_some() { 10 } else { 1 },
                ef,
                self.collection.vectors.as_slice(),
            )
        } else {
            self.collection.index.search(
                self.query,
                self.k * if self.filter.is_some() { 10 } else { 1 },
                self.collection.vectors.as_slice(),
            )
        };

        // Apply filter if present
        let filtered: Vec<(VectorId, f32)> = if let Some(filter) = self.filter {
            raw_results
                .into_iter()
                .filter(|(id, _)| {
                    if let Some(entry) = self.collection.metadata.get(*id) {
                        filter.matches(entry.data.as_ref())
                    } else {
                        false
                    }
                })
                .take(self.k)
                .collect()
        } else {
            raw_results.into_iter().take(self.k).collect()
        };

        if self.include_metadata {
            self.collection.enrich_results(filtered)
        } else {
            // Return results without metadata
            filtered
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
                .collect()
        }
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
}

/// A collection of vectors with the same dimensions
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
}

impl Collection {
    /// Create a new collection
    pub fn new(config: CollectionConfig) -> Self {
        Self {
            vectors: VectorStore::new(config.dimensions),
            index: HnswIndex::new(config.hnsw.clone(), config.distance),
            metadata: MetadataStore::new(),
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

    /// Insert a vector with ID and optional metadata
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

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(query, k, self.vectors.as_slice());

        self.enrich_results(results)
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
        results.into_iter().collect()
    }

    /// Batch search with metadata filter in parallel
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
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

        results.into_iter().collect()
    }

    /// Search with a metadata filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
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

        self.enrich_results(filtered)
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

    /// Get a vector by its external ID
    pub fn get(&self, id: &str) -> Option<(&[f32], Option<&Value>)> {
        let internal_id = self.metadata.get_internal_id(id)?;
        let vector = self.vectors.get(internal_id)?;
        let metadata = self.metadata.get(internal_id).and_then(|e| e.data.as_ref());
        Some((vector, metadata))
    }

    /// Check if a vector ID exists
    pub fn contains(&self, id: &str) -> bool {
        self.metadata.contains(id)
    }

    /// Delete a vector by its external ID
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let internal_id = match self.metadata.get_internal_id(id) {
            Some(id) => id,
            None => return Ok(false),
        };

        self.metadata.delete(internal_id);
        self.index.delete(internal_id)?;

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

    /// Update an existing vector and its metadata
    /// Returns error if the vector doesn't exist
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

        Ok(())
    }

    /// Update only the metadata for an existing vector
    /// Returns error if the vector doesn't exist
    pub fn update_metadata(&mut self, id: &str, metadata: Option<Value>) -> Result<()> {
        let internal_id = self.metadata.get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        self.metadata.update_data(internal_id, metadata)?;
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
}
