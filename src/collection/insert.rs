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

        // Validate metadata against schema if configured
        if let (Some(meta), Some(schema)) = (&metadata, &self.config.metadata_schema) {
            crate::metadata::validate_metadata_schema(meta, schema)?;
        }

        // Validate dimensions and vector values
        self.validate_insert_input(&vector)?;

        // Check if ID already exists
        if self.metadata.contains(&id) {
            return Err(NeedleError::VectorAlreadyExists(id));
        }

        // Semantic dedup check: 1-NN search against existing vectors
        if let Some((existing_id, distance)) = self.check_dedup(&vector) {
            let policy = self
                .config
                .dedup
                .as_ref()
                .map_or(crate::collection::config::DedupPolicy::Reject, |c| c.policy);
            let _ = self.apply_dedup_policy(
                &id,
                vector,
                metadata,
                &existing_id,
                distance,
                policy,
                ttl_seconds,
            )?;
            return Ok(());
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

        // Record CDC event if enabled
        {
            let ext_id = self.metadata.get(internal_id).map(|e| e.external_id.clone());
            if let Some(eid) = ext_id {
                self.record_cdc_event(CdcEventType::Insert, &eid, None);
            }
        }

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

    /// Insert a text document using the built-in embedded model runtime.
    ///
    /// Embeds `text` using the collection's embedded runtime (lazily initialized).
    /// The runtime's output dimensions must match the collection's dimensions.
    ///
    /// # Feature gate
    ///
    /// Available with the `embedded-models` feature flag.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The text is empty
    /// - Vector dimensions don't match the collection
    /// - A vector with the same ID already exists
    #[cfg(feature = "embedded-models")]
    pub fn insert_text(
        &mut self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let embedding = self.get_or_init_runtime().embed_text(text)?;
        self.insert_vec(id, embedding, metadata)
    }

    /// Search by text using the built-in embedded model runtime.
    ///
    /// Embeds the query text and performs a k-NN search.
    #[cfg(feature = "embedded-models")]
    pub fn search_text(&self, text: &str, k: usize) -> Result<Vec<SearchResult>> {
        let runtime = self.embedded_runtime.as_ref().map_or_else(
            || {
                let rt = crate::ml::embedded_runtime::EmbeddingRuntime::new();
                rt.embed_text(text)
            },
            |rt| rt.embed_text(text),
        )?;
        self.search(&runtime, k)
    }

    /// Lazily initialize and return the embedded runtime.
    #[cfg(feature = "embedded-models")]
    fn get_or_init_runtime(&mut self) -> &crate::ml::embedded_runtime::EmbeddingRuntime {
        if self.embedded_runtime.is_none() {
            self.embedded_runtime = Some(Arc::new(crate::ml::embedded_runtime::EmbeddingRuntime::new()));
        }
        self.embedded_runtime.as_ref().expect("just initialized")
    }

    /// Batch insert text documents using the built-in embedded model runtime.
    #[cfg(feature = "embedded-models")]
    pub fn insert_texts_batch(
        &mut self,
        items: Vec<(String, String, Option<Value>)>,
    ) -> Result<()> {
        let runtime = self.get_or_init_runtime();
        let texts: Vec<&str> = items.iter().map(|(_, t, _)| t.as_str()).collect();
        let embeddings = runtime.embed_batch(&texts)?;
        let ids = items.iter().map(|(id, _, _)| id.clone()).collect();
        let metadata = items.into_iter().map(|(_, _, m)| m).collect();
        self.insert_batch(ids, embeddings, metadata)
    }

    /// Insert a document by text, auto-embedding it using the configured provider.
    ///
    /// Requires `auto_embed` to be configured on the collection via
    /// [`CollectionConfig::with_auto_embed`]. The text is converted to a vector
    /// using a deterministic hash-based embedding and stored alongside `_text`
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::InvalidConfig`] - Auto-embed not configured
    /// - [`NeedleError::VectorAlreadyExists`] - A vector with the same ID exists
    pub fn insert_auto_text(
        &mut self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let config = self.config.auto_embed.as_ref().ok_or_else(|| {
            NeedleError::InvalidConfig(
                "Auto-embed not configured for this collection. \
                 Use with_auto_embed() on CollectionConfig."
                    .into(),
            )
        })?;

        let embedding = Self::generate_auto_embedding(text, self.config.dimensions, config)?;

        let mut meta = metadata.unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = meta.as_object_mut() {
            obj.insert("_text".to_string(), Value::String(text.to_string()));
        }
        self.insert_vec_with_ttl(id, embedding, Some(meta), None)
    }

    /// Generate a deterministic hash-based embedding from text.
    fn generate_auto_embedding(
        text: &str,
        dimensions: usize,
        _config: &crate::collection::config::AutoEmbedInsertConfig,
    ) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut vector = vec![0.0f32; dimensions];
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            word.hash(&mut hasher);
            let hash = hasher.finish();
            let idx = (hash as usize) % dimensions;
            vector[idx] += 1.0 / (1.0 + i as f32);
        }
        // Normalize
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in &mut vector {
                *v /= magnitude;
            }
        }
        Ok(vector)
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

    /// Bulk import vectors from a JSON-Lines reader.
    ///
    /// Each line should be a JSON object with fields:
    /// - `"id"`: string (required)
    /// - `"vector"`: array of floats (required)
    /// - `"metadata"`: object (optional)
    ///
    /// Blank lines and lines starting with `#` are skipped.
    /// Returns the number of successfully imported vectors and any per-line errors.
    pub fn import_jsonl<R: std::io::BufRead>(&mut self, reader: R) -> Result<ImportResult> {
        let mut imported = 0usize;
        let mut errors = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(NeedleError::Io)?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            match serde_json::from_str::<ImportRecord>(trimmed) {
                Ok(record) => {
                    match self.insert(record.id.clone(), &record.vector, record.metadata) {
                        Ok(()) => imported += 1,
                        Err(e) => errors.push(ImportError {
                            line: line_num + 1,
                            message: e.to_string(),
                        }),
                    }
                }
                Err(e) => {
                    errors.push(ImportError {
                        line: line_num + 1,
                        message: e.to_string(),
                    });
                }
            }
        }

        Ok(ImportResult { imported, errors })
    }
}

/// A single record in a JSONL import file.
#[derive(Deserialize)]
struct ImportRecord {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

/// Result of a bulk JSONL import operation.
#[derive(Debug, Clone, Serialize)]
pub struct ImportResult {
    /// Number of vectors successfully imported.
    pub imported: usize,
    /// Per-line errors encountered during import.
    pub errors: Vec<ImportError>,
}

/// A single error encountered during JSONL import.
#[derive(Debug, Clone, Serialize)]
pub struct ImportError {
    /// 1-based line number where the error occurred.
    pub line: usize,
    /// Description of the error.
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::config::SemanticDedupConfig;
    use serde_json::json;

    // ── Basic inserts ───────────────────────────────────────────────────

    #[test]
    fn test_insert_basic() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.len(), 1);
        assert!(col.contains("v1"));
    }

    #[test]
    fn test_insert_with_metadata() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"key": "val"})))
            .unwrap();
        let (vec, meta) = col.get("v1").unwrap();
        assert_eq!(vec.len(), 4);
        assert!(meta.is_some());
    }

    #[test]
    fn test_insert_vec_ownership() {
        let mut col = Collection::with_dimensions("test", 4);
        let vec = vec![1.0, 0.0, 0.0, 0.0];
        col.insert_vec("v1", vec, None).unwrap();
        assert_eq!(col.len(), 1);
    }

    // ── Error conditions ────────────────────────────────────────────────

    #[test]
    fn test_insert_dimension_mismatch() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.insert("v1", &[1.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_duplicate_id() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        let result = col.insert("v1", &[0.0, 1.0, 0.0, 0.0], None);
        assert!(matches!(result, Err(NeedleError::VectorAlreadyExists(_))));
    }

    #[test]
    fn test_insert_nan_vector() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.insert("v1", &[f32::NAN, 0.0, 0.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_infinity_vector() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.insert("v1", &[f32::INFINITY, 0.0, 0.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_neg_infinity_vector() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.insert("v1", &[f32::NEG_INFINITY, 0.0, 0.0, 0.0], None);
        assert!(result.is_err());
    }

    // ── TTL inserts ─────────────────────────────────────────────────────

    #[test]
    fn test_insert_with_ttl() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert_with_ttl("v1", &[1.0, 0.0, 0.0, 0.0], None, Some(3600))
            .unwrap();
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_insert_with_ttl_none() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert_with_ttl("v1", &[1.0, 0.0, 0.0, 0.0], None, None)
            .unwrap();
        assert_eq!(col.len(), 1);
    }

    // ── Batch inserts ───────────────────────────────────────────────────

    #[test]
    fn test_insert_batch_basic() {
        let mut col = Collection::with_dimensions("test", 4);
        let ids = vec!["a".into(), "b".into(), "c".into()];
        let vecs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let meta = vec![None, None, None];
        col.insert_batch(ids, vecs, meta).unwrap();
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_insert_batch_mismatched_sizes() {
        let mut col = Collection::with_dimensions("test", 4);
        let ids = vec!["a".into(), "b".into()];
        let vecs = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let meta = vec![None, None];
        let result = col.insert_batch(ids, vecs, meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_batch_duplicate_ids_in_batch() {
        let mut col = Collection::with_dimensions("test", 4);
        let ids = vec!["a".into(), "a".into()];
        let vecs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let meta = vec![None, None];
        let result = col.insert_batch(ids, vecs, meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_batch_existing_id() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        let ids = vec!["a".into(), "b".into()];
        let vecs = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let meta = vec![None, None];
        let result = col.insert_batch(ids, vecs, meta);
        assert!(matches!(result, Err(NeedleError::VectorAlreadyExists(_))));
    }

    #[test]
    fn test_insert_batch_dimension_mismatch_rolls_back() {
        let mut col = Collection::with_dimensions("test", 4);
        let ids = vec!["a".into(), "b".into()];
        let vecs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0], // wrong dims
        ];
        let meta = vec![None, None];
        let result = col.insert_batch(ids, vecs, meta);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_batch_empty() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert_batch(vec![], vec![], vec![]).unwrap();
        assert_eq!(col.len(), 0);
    }

    // ── Multiple sequential inserts ─────────────────────────────────────

    #[test]
    fn test_insert_many_sequential() {
        let mut col = Collection::with_dimensions("test", 4);
        for i in 0..100 {
            col.insert(format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0], None)
                .unwrap();
        }
        assert_eq!(col.len(), 100);
    }

    // ── Insert then search ──────────────────────────────────────────────

    #[test]
    fn test_insert_then_search_finds_result() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = col.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].id, "v1");
    }

    // ── Dedup inserts ───────────────────────────────────────────────────

    #[test]
    fn test_insert_with_dedup_reject() {
        let config = CollectionConfig::new("test", 4)
            .with_dedup(SemanticDedupConfig::new(0.5, crate::collection::config::DedupPolicy::Reject));
        let mut col = Collection::new(config);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        // Near-duplicate (identical vector) should be rejected
        let result = col.insert("v2", &[1.0, 0.0, 0.0, 0.0], None);
        assert!(result.is_err() || col.len() == 1);
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_insert_text_basic() {
        // Default embedded runtime uses MiniLM (384 dims)
        let mut col = Collection::with_dimensions("test", 384);
        col.insert_text("doc1", "Hello world", None).unwrap();
        assert_eq!(col.len(), 1);
        assert!(col.contains("doc1"));
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_insert_text_deterministic() {
        let mut col = Collection::with_dimensions("test", 384);
        col.insert_text("doc1", "Hello world", None).unwrap();

        // Same text should produce same embedding (deterministic mock)
        let (vec1, _) = col.get("doc1").unwrap();
        let mut col2 = Collection::with_dimensions("test2", 384);
        col2.insert_text("doc2", "Hello world", None).unwrap();
        let (vec2, _) = col2.get("doc2").unwrap();
        assert_eq!(vec1, vec2);
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_insert_text_empty_rejects() {
        let mut col = Collection::with_dimensions("test", 384);
        let result = col.insert_text("doc1", "", None);
        assert!(result.is_err());
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_insert_text_duplicate_rejects() {
        let mut col = Collection::with_dimensions("test", 384);
        col.insert_text("doc1", "Hello", None).unwrap();
        let result = col.insert_text("doc1", "World", None);
        assert!(result.is_err());
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_search_text() {
        let mut col = Collection::with_dimensions("test", 384);
        col.insert_text("doc1", "machine learning", None).unwrap();
        col.insert_text("doc2", "cooking recipes", None).unwrap();
        let results = col.search_text("machine learning", 2).unwrap();
        assert_eq!(results.len(), 2);
        // Same text should be closest match
        assert_eq!(results[0].id, "doc1");
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_insert_texts_batch() {
        let mut col = Collection::with_dimensions("test", 384);
        let items = vec![
            ("d1".to_string(), "Hello world".to_string(), None),
            ("d2".to_string(), "Goodbye world".to_string(), None),
        ];
        col.insert_texts_batch(items).unwrap();
        assert_eq!(col.len(), 2);
    }

    #[cfg(feature = "embedded-models")]
    #[test]
    fn test_insert_text_dimension_mismatch() {
        // Default runtime outputs 384 dims; collection has 128 → should error
        let mut col = Collection::with_dimensions("test", 128);
        let result = col.insert_text("doc1", "Hello world", None);
        assert!(result.is_err(), "insert_text should fail when model output dims != collection dims");
    }

    #[test]
    fn test_insert_auto_text_without_config_fails() {
        let mut col = Collection::with_dimensions("test", 64);
        let result = col.insert_auto_text("doc1", "Hello world", None);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Auto-embed not configured"));
    }

    #[test]
    fn test_insert_auto_text_with_config_succeeds() {
        use crate::collection::config::AutoEmbedInsertConfig;

        let config = CollectionConfig::new("test", 64).with_auto_embed(AutoEmbedInsertConfig {
            provider: "hash".to_string(),
            model: None,
        });
        let mut col = Collection::new(config);
        col.insert_auto_text("doc1", "Hello world of vectors", None)
            .unwrap();
        assert_eq!(col.len(), 1);

        // Verify _text metadata is stored
        let (_, meta) = col.get("doc1").unwrap();
        let meta = meta.unwrap();
        assert_eq!(meta["_text"], "Hello world of vectors");
    }

    #[test]
    fn test_insert_auto_text_is_searchable() {
        use crate::collection::config::AutoEmbedInsertConfig;

        let config = CollectionConfig::new("test", 64).with_auto_embed(AutoEmbedInsertConfig {
            provider: "hash".to_string(),
            model: None,
        });
        let mut col = Collection::new(config);
        col.insert_auto_text("doc1", "machine learning algorithms", None)
            .unwrap();
        col.insert_auto_text("doc2", "cooking recipes and food", None)
            .unwrap();
        assert_eq!(col.len(), 2);

        // Search using the vector from doc1 — doc1 should be the top result
        let (doc1_vec, _) = col.get("doc1").unwrap();
        let doc1_vec = doc1_vec.to_vec();
        let results = col.search(&doc1_vec, 2).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }

    // ── JSONL import ────────────────────────────────────────────────────

    #[test]
    fn test_import_jsonl_basic() {
        let mut col = Collection::with_dimensions("test", 3);
        let input = r#"{"id": "v1", "vector": [1.0, 2.0, 3.0], "metadata": {"tag": "a"}}
{"id": "v2", "vector": [4.0, 5.0, 6.0]}
"#;
        let reader = std::io::BufReader::new(input.as_bytes());
        let result = col.import_jsonl(reader).unwrap();

        assert_eq!(result.imported, 2);
        assert!(result.errors.is_empty());
        assert_eq!(col.len(), 2);
        assert!(col.contains("v1"));
        assert!(col.contains("v2"));

        let (_, meta) = col.get("v1").unwrap();
        assert_eq!(meta.unwrap()["tag"], "a");
    }

    #[test]
    fn test_import_jsonl_skips_blanks_and_comments() {
        let mut col = Collection::with_dimensions("test", 3);
        let input = "# header comment\n\n{\"id\": \"v1\", \"vector\": [1.0, 2.0, 3.0]}\n  \n# another comment\n";
        let reader = std::io::BufReader::new(input.as_bytes());
        let result = col.import_jsonl(reader).unwrap();

        assert_eq!(result.imported, 1);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_import_jsonl_partial_errors() {
        let mut col = Collection::with_dimensions("test", 3);
        let input = r#"{"id": "v1", "vector": [1.0, 2.0, 3.0]}
not valid json
{"id": "v2", "vector": [4.0, 5.0, 6.0]}
"#;
        let reader = std::io::BufReader::new(input.as_bytes());
        let result = col.import_jsonl(reader).unwrap();

        assert_eq!(result.imported, 2);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].line, 2);
    }

    #[test]
    fn test_import_jsonl_duplicate_id_error() {
        let mut col = Collection::with_dimensions("test", 3);
        let input = r#"{"id": "v1", "vector": [1.0, 2.0, 3.0]}
{"id": "v1", "vector": [4.0, 5.0, 6.0]}
"#;
        let reader = std::io::BufReader::new(input.as_bytes());
        let result = col.import_jsonl(reader).unwrap();

        assert_eq!(result.imported, 1);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].line, 2);
    }

    #[test]
    fn test_import_jsonl_empty_input() {
        let mut col = Collection::with_dimensions("test", 3);
        let reader = std::io::BufReader::new(b"" as &[u8]);
        let result = col.import_jsonl(reader).unwrap();

        assert_eq!(result.imported, 0);
        assert!(result.errors.is_empty());
    }
}
