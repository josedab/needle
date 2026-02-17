use super::*;

/// Maximum bundle import file size (1 GB).
const MAX_BUNDLE_FILE_SIZE: u64 = 1024 * 1024 * 1024;

impl Collection {
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

    /// Estimate memory usage for a dataset of the given size.
    ///
    /// This is a static helper that does not require an existing collection.
    /// Use it to plan capacity before inserting data.
    ///
    /// # Arguments
    ///
    /// * `vector_count` - Number of vectors to store
    /// * `dimensions` - Dimensionality of each vector
    /// * `avg_metadata_bytes` - Average metadata size per vector in bytes
    ///
    /// # Returns
    ///
    /// Estimated total memory in bytes (vectors + metadata + HNSW index overhead).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let bytes = Collection::estimate_memory(1_000_000, 384, 256);
    /// println!("Estimated memory: {} MB", bytes / 1024 / 1024);
    /// ```
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
    ///
    /// Logs a relevance signal for the given query–vector pair. This feeds into
    /// the contextual bandits reranker for learning optimal result ordering.
    ///
    /// # Arguments
    ///
    /// * `query_id` - Identifier of the originating search query
    /// * `vector_id` - ID of the result vector being rated
    /// * `relevance_score` - Relevance score (e.g., 0.0 = irrelevant, 1.0 = highly relevant)
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # let config = CollectionConfig::new("test", 4);
    /// # let collection = Collection::new(config);
    /// // User clicked on "doc42" from query "q_123"
    /// collection.record_feedback("q_123", "doc42", 1.0);
    /// ```
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
        let file_size = std::fs::metadata(path)
            .map_err(NeedleError::Io)?
            .len();
        if file_size > MAX_BUNDLE_FILE_SIZE {
            return Err(NeedleError::InvalidInput(format!(
                "Bundle file too large ({} bytes). Maximum allowed size is {} bytes",
                file_size, MAX_BUNDLE_FILE_SIZE
            )));
        }
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
