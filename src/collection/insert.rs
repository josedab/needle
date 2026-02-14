use super::*;

impl Collection {
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

    /// Insert a text document, automatically embedding it using the provided model.
    ///
    /// This convenience method embeds `text` using the given `EmbeddingModel` and
    /// stores the resulting vector with optional metadata. The model's output
    /// dimensions must match the collection's configured dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The embedding model fails
    /// - Vector dimensions don't match the collection
    /// - A vector with the same ID already exists
    pub fn insert_with_text(
        &mut self,
        id: impl Into<String>,
        text: &str,
        model: &dyn crate::ml::embedded_runtime::EmbeddingModel,
        metadata: Option<Value>,
    ) -> Result<()> {
        let embedding = model.embed(text)?;
        self.insert_vec(id, embedding, metadata)
    }

    /// Insert multiple text documents in batch, embedding them with the given model.
    pub fn insert_batch_with_text(
        &mut self,
        ids: Vec<String>,
        texts: &[&str],
        model: &dyn crate::ml::embedded_runtime::EmbeddingModel,
        metadata: Vec<Option<Value>>,
    ) -> Result<()> {
        let embeddings = model.embed_batch(texts)?;
        self.insert_batch(ids, embeddings, metadata)
    }

    /// Insert multiple vectors in batch.
    ///
    /// Atomically inserts all vectors or rolls back on failure. Each ID must be
    /// unique both within the batch and against existing vectors in the collection.
    ///
    /// # Arguments
    ///
    /// * `ids` - External IDs for the vectors (must be unique)
    /// * `vectors` - Vector data (each must match collection dimensions)
    /// * `metadata` - Optional JSON metadata for each vector
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::InvalidConfig`] - Batch sizes don't match or batch contains duplicate IDs
    /// - [`NeedleError::VectorAlreadyExists`] - An ID already exists in the collection
    /// - [`NeedleError::DimensionMismatch`] - Any vector has wrong dimensions
    /// - [`NeedleError::InvalidVector`] - Any vector contains NaN or Infinity
    ///
    /// On error, all previously inserted vectors in the batch are rolled back.
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # use serde_json::json;
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// let ids = vec!["a".to_string(), "b".to_string()];
    /// let vectors = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    /// let metadata = vec![Some(json!({"type": "x"})), None];
    /// collection.insert_batch(ids, vectors, metadata)?;
    /// assert_eq!(collection.len(), 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
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
}
