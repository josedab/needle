use super::{Database, ExportEntry};
use crate::collection::{Collection, SearchResult};
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::tuning::{RecommendedIndex, WorkloadObservation};
use serde_json::Value;

impl Database {
    // Internal methods for CollectionRef

    pub(crate) fn insert_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert(&id, vector, metadata)?;
        drop(state);

        self.mark_modified();

        // Record version if versioning is enabled
        if let Ok(mut store) = self.versioned_store_mut(collection) {
            let _ = store.put(&id, vector, None);
        }

        Ok(())
    }

    pub(crate) fn insert_vec_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();

        // Record version before insert (we need the vector data)
        let version_vec = vector.clone();
        let version_meta = metadata.clone();

        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_vec(&id, vector, metadata)?;
        drop(state);

        self.mark_modified();
        if let Some(aim) = &self.adaptive_index_manager {
            aim.record_insert();
        }
        #[cfg(feature = "observability")]
        if let Some(metrics) = &self.dashboard_metrics {
            metrics.record_insert(collection);
        }

        // Record version if versioning is enabled for this collection
        if let Ok(mut store) = self.versioned_store_mut(collection) {
            let _ = store.put(&id, &version_vec, version_meta);
        }

        Ok(())
    }

    pub(crate) fn insert_with_ttl_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        let id = id.into();
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_with_ttl(&id, vector, metadata, ttl_seconds)?;
        drop(state);

        self.mark_modified();

        if let Ok(mut store) = self.versioned_store_mut(collection) {
            let _ = store.put(&id, vector, None);
        }

        Ok(())
    }

    pub(crate) fn insert_vec_with_ttl_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        let id = id.into();
        let version_vec = vector.clone();

        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_vec_with_ttl(&id, vector, metadata, ttl_seconds)?;
        drop(state);

        self.mark_modified();

        if let Ok(mut store) = self.versioned_store_mut(collection) {
            let _ = store.put(&id, &version_vec, None);
        }

        Ok(())
    }

    #[cfg(feature = "embedded-models")]
    pub(crate) fn insert_text_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_text(id, text, metadata)?;
        self.mark_modified();
        Ok(())
    }

    #[cfg(feature = "embedded-models")]
    pub(crate) fn search_text_internal(
        &self,
        collection: &str,
        text: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let runtime = crate::ml::embedded_runtime::EmbeddingRuntime::new();
        let embedding = runtime.embed_text(text)?;
        self.search_internal(collection, &embedding, k)
    }

    pub(crate) fn update_internal(
        &self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let version_meta = metadata.clone();

        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.update(id, vector, metadata)?;
        drop(state);

        self.mark_modified();

        // Record new version if versioning is enabled
        if let Ok(mut store) = self.versioned_store_mut(collection) {
            let _ = store.put(id, vector, version_meta);
        }

        Ok(())
    }

    pub(crate) fn search_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();

        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let results = coll.search(query, k)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Feed latency data to adaptive tuner if attached
        if let Some(tuner) = &self.adaptive_tuner {
            let stats = coll.stats();
            let obs = WorkloadObservation {
                vector_count: stats.vector_count,
                dimensions: stats.dimensions,
                qps: 0.0,
                insert_rate: 0.0,
                avg_latency_ms: latency_ms,
                measured_recall: 1.0, // unknown in production; assume best-case
                memory_bytes: stats.total_memory_bytes as u64,
                current_index: RecommendedIndex::Hnsw,
                current_config: None,
            };
            tuner.feedback(&obs, 1.0, latency_ms);
        }

        // Feed workload data to adaptive index manager if attached
        if let Some(aim) = &self.adaptive_index_manager {
            aim.record_query(latency_ms, results.len());
        }

        // Record metrics for observability dashboard
        #[cfg(feature = "observability")]
        if let Some(metrics) = &self.dashboard_metrics {
            metrics.record_query(collection, (latency_ms * 1000.0) as u64, results.len());
        }

        Ok(results)
    }

    pub(crate) fn search_with_filter_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_with_filter(query, k, filter)
    }

    pub(crate) fn search_with_options_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        distance_override: Option<crate::DistanceFunction>,
        filter: Option<&Filter>,
        post_filter: Option<&Filter>,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let mut builder = coll.search_builder(query).k(k);

        if let Some(dist) = distance_override {
            builder = builder.distance(dist);
        }

        if let Some(f) = filter {
            builder = builder.filter(f);
        }

        if let Some(pf) = post_filter {
            builder = builder
                .post_filter(pf)
                .post_filter_factor(post_filter_factor);
        }

        builder.execute()
    }

    /// Search across multiple collections and merge results.
    ///
    /// Uses the federated search module with min-max normalization and RRF merge by default.
    pub(crate) fn federated_search_internal(
        &self,
        query: &[f32],
        k: usize,
        collection_names: &[&str],
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();

        // Search each collection and collect results with collection name
        let mut all_results: Vec<(String, Vec<SearchResult>)> = Vec::new();
        for &name in collection_names {
            let coll = state
                .collections
                .get(name)
                .ok_or_else(|| NeedleError::CollectionNotFound(name.to_string()))?;

            // Truncate or pad query to match collection dimensions
            let dims = coll.dimensions();
            let adapted_query: Vec<f32> = if query.len() == dims {
                query.to_vec()
            } else if query.len() > dims {
                query[..dims].to_vec()
            } else {
                let mut padded = query.to_vec();
                padded.resize(dims, 0.0);
                padded
            };

            let results = coll.search(&adapted_query, k * 2)?;
            all_results.push((name.to_string(), results));
        }

        // Merge using RRF (reciprocal rank fusion)
        let rrf_k = 60;
        let mut scores: std::collections::HashMap<String, (f64, SearchResult)> =
            std::collections::HashMap::new();

        for (_collection, results) in &all_results {
            for (rank, result) in results.iter().enumerate() {
                let rrf_score = 1.0 / (rrf_k as f64 + rank as f64 + 1.0);
                let entry = scores
                    .entry(result.id.clone())
                    .or_insert((0.0, result.clone()));
                entry.0 += rrf_score;
            }
        }

        let mut merged: Vec<(f64, SearchResult)> = scores.into_values().collect();
        merged.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(merged.into_iter().take(k).map(|(_, r)| r).collect())
    }

    pub(crate) fn search_explain_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_explain(query, k)
    }

    pub(crate) fn search_with_filter_explain_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_with_filter_explain(query, k, filter)
    }

    pub(crate) fn search_with_trace_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::hnsw::SearchTrace)> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_with_trace(query, k)
    }

    pub(crate) fn search_with_post_filter_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        pre_filter: Option<&Filter>,
        post_filter: &Filter,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let mut builder = coll
            .search_builder(query)
            .k(k)
            .post_filter(post_filter)
            .post_filter_factor(post_filter_factor);

        if let Some(pf) = pre_filter {
            builder = builder.filter(pf);
        }

        builder.execute()
    }

    pub(crate) fn search_matryoshka_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        coarse_dims: usize,
        oversample: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_matryoshka(query, k, coarse_dims, oversample)
    }

    pub(crate) fn search_radius_internal(
        &self,
        collection: &str,
        query: &[f32],
        max_distance: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_radius(query, max_distance, limit)
    }

    pub(crate) fn search_radius_with_filter_internal(
        &self,
        collection: &str,
        query: &[f32],
        max_distance: f32,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_radius_with_filter(query, max_distance, limit, filter)
    }

    pub(crate) fn get_internal(&self, collection: &str, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        let state = self.state.read();
        let coll = state.collections.get(collection)?;
        let (vec, meta) = coll.get(id)?;
        Some((vec.to_vec(), meta.cloned()))
    }

    pub(crate) fn delete_internal(&self, collection: &str, id: &str) -> Result<bool> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let deleted = coll.delete(id)?;
        if deleted {
            self.mark_modified();
            drop(state);

            // Record version tombstone if versioning is enabled
            if let Ok(mut store) = self.versioned_store_mut(collection) {
                let _ = store.delete(id);
            }
        }
        Ok(deleted)
    }

    pub(crate) fn collection_len(&self, collection: &str) -> usize {
        self.state
            .read()
            .collections
            .get(collection)
            .map_or(0, |c| c.len())
    }

    pub(crate) fn collection_dimensions(&self, collection: &str) -> Option<usize> {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.dimensions())
    }

    pub(crate) fn collection_stats_internal(
        &self,
        collection: &str,
    ) -> Result<crate::collection::CollectionStats> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        Ok(coll.stats())
    }

    pub(crate) fn get_provenance_internal(
        &self,
        collection: &str,
        vector_id: &str,
    ) -> Option<crate::persistence::vector_versioning::ProvenanceRecord> {
        let state = self.state.read();
        let coll = state.collections.get(collection)?;
        coll.get_provenance(vector_id).cloned()
    }

    pub(crate) fn evaluate_internal(
        &self,
        collection: &str,
        ground_truth: &[crate::collection::GroundTruthEntry],
        k: usize,
    ) -> Result<crate::collection::EvaluationReport> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        coll.evaluate(ground_truth, k)
    }

    pub(crate) fn export_internal(&self, collection: &str) -> Result<Vec<ExportEntry>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        Ok(coll
            .iter()
            .map(|(id, vec, meta)| (id.to_string(), vec.to_vec(), meta.cloned()))
            .collect())
    }

    pub(crate) fn ids_internal(&self, collection: &str) -> Result<Vec<String>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        Ok(coll.ids().map(|s| s.to_string()).collect())
    }

    pub(crate) fn compact_internal(&self, collection: &str) -> Result<usize> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let deleted = coll.compact()?;
        if deleted > 0 {
            self.mark_modified();
        }
        Ok(deleted)
    }

    pub(crate) fn needs_compaction_internal(&self, collection: &str, threshold: f64) -> bool {
        self.state
            .read()
            .collections
            .get(collection)
            .is_some_and(|c| c.needs_compaction(threshold))
    }

    pub(crate) fn dedup_scan_internal(
        &self,
        collection: &str,
        threshold: Option<f32>,
    ) -> Result<crate::collection::dedup::DedupScanResult> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        Ok(coll.dedup_scan(threshold))
    }

    // ============ TTL Internal Methods ============

    pub(crate) fn expire_vectors_internal(&self, collection: &str) -> Result<usize> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let count = coll.expire_vectors()?;
        if count > 0 {
            self.mark_modified();
        }
        Ok(count)
    }

    pub(crate) fn needs_expiration_sweep_internal(&self, collection: &str, threshold: f64) -> bool {
        self.state
            .read()
            .collections
            .get(collection)
            .is_some_and(|c| c.needs_expiration_sweep(threshold))
    }

    pub(crate) fn ttl_stats_internal(&self, collection: &str) -> (usize, usize, Option<u64>, Option<u64>) {
        self.state
            .read()
            .collections
            .get(collection)
            .map_or((0, 0, None, None), |c| c.ttl_stats())
    }

    pub(crate) fn get_ttl_internal(&self, collection: &str, id: &str) -> Option<u64> {
        self.state
            .read()
            .collections
            .get(collection)
            .and_then(|c| c.get_ttl(id))
    }

    pub(crate) fn set_ttl_internal(&self, collection: &str, id: &str, ttl_seconds: Option<u64>) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.set_ttl(id, ttl_seconds)?;
        self.mark_modified();
        Ok(())
    }

    pub(crate) fn count_internal(&self, collection: &str, filter: Option<&Filter>) -> Result<usize> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        Ok(coll.count(filter))
    }

    pub(crate) fn deleted_count_internal(&self, collection: &str) -> usize {
        self.state
            .read()
            .collections
            .get(collection)
            .map_or(0, |c| c.deleted_count())
    }

    pub(crate) fn search_ids_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_ids(query, k)
    }

    pub(crate) fn export_bundle_internal(
        &self,
        collection: &str,
        path: &std::path::Path,
    ) -> Result<crate::collection::BundleManifest> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        coll.export_bundle(path)
    }

    /// Import a collection from a portable bundle file into this database.
    ///
    /// # Errors
    /// Returns an error if the bundle is invalid or I/O fails.
    pub fn import_bundle(
        &mut self,
        path: &std::path::Path,
        name_override: Option<&str>,
    ) -> Result<crate::collection::BundleManifest> {
        let mut collection = Collection::import_bundle(path)?;

        let name = if let Some(n) = name_override {
            collection.set_name(n.to_string());
            n.to_string()
        } else {
            collection.name().to_string()
        };

        let manifest = crate::collection::BundleManifest {
            format_version: 1,
            collection_name: name.clone(),
            dimensions: collection.dimensions(),
            distance_function: format!("{:?}", collection.config().distance),
            vector_count: collection.len(),
            embedding_model: None,
            created_at: 0,
            data_hash: None,
            semver: "1.0.0".to_string(),
            description: None,
            registry_uri: None,
            tags: Vec::new(),
        };

        let mut state = self.state.write();
        state.collections.insert(name, collection);
        drop(state);
        self.mark_modified();

        Ok(manifest)
    }
}

#[cfg(test)]
mod tests {
    use crate::Database;
    use crate::error::NeedleError;
    use crate::metadata::Filter;
    use serde_json::json;

    fn setup_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("coll", 4).unwrap();
        db
    }

    fn setup_db_with_vectors() -> Database {
        let db = setup_db();
        db.insert_internal("coll", "v1", &[1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        db.insert_internal(
            "coll",
            "v2",
            &[0.0, 1.0, 0.0, 0.0],
            Some(json!({"tag": "a"})),
        )
        .unwrap();
        db
    }

    // ── insert_internal ─────────────────────────────────────────────────

    #[test]
    fn test_insert_internal_basic() {
        let db = setup_db();
        db.insert_internal("coll", "v1", &[1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(db.collection_len("coll"), 1);
    }

    #[test]
    fn test_insert_internal_missing_collection() {
        let db = setup_db();
        let result = db.insert_internal("nonexistent", "v1", &[1.0, 0.0, 0.0, 0.0], None);
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── insert_vec_internal ─────────────────────────────────────────────

    #[test]
    fn test_insert_vec_internal() {
        let db = setup_db();
        db.insert_vec_internal("coll", "v1", vec![1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        assert_eq!(db.collection_len("coll"), 1);
    }

    // ── insert_with_ttl_internal ────────────────────────────────────────

    #[test]
    fn test_insert_with_ttl_internal() {
        let db = setup_db();
        db.insert_with_ttl_internal("coll", "v1", &[1.0, 0.0, 0.0, 0.0], None, Some(3600))
            .unwrap();
        assert!(db.get_ttl_internal("coll", "v1").is_some());
    }

    // ── update_internal ─────────────────────────────────────────────────

    #[test]
    fn test_update_internal() {
        let db = setup_db_with_vectors();
        db.update_internal("coll", "v1", &[0.5, 0.5, 0.0, 0.0], Some(json!({"new": true})))
            .unwrap();
        let (vec, meta) = db.get_internal("coll", "v1").unwrap();
        assert_eq!(vec, vec![0.5, 0.5, 0.0, 0.0]);
        assert!(meta.is_some());
    }

    #[test]
    fn test_update_internal_missing_collection() {
        let db = setup_db();
        let result = db.update_internal("bad", "v1", &[1.0, 0.0, 0.0, 0.0], None);
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── search_internal ─────────────────────────────────────────────────

    #[test]
    fn test_search_internal_roundtrip() {
        let db = setup_db_with_vectors();
        let results = db.search_internal("coll", &[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_search_internal_missing_collection() {
        let db = setup_db();
        let result = db.search_internal("bad", &[1.0, 0.0, 0.0, 0.0], 1);
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── search_with_filter_internal ─────────────────────────────────────

    #[test]
    fn test_search_with_filter_internal() {
        let db = setup_db_with_vectors();
        let filter = Filter::eq("tag", "a");
        let results = db
            .search_with_filter_internal("coll", &[0.0, 1.0, 0.0, 0.0], 5, &filter)
            .unwrap();
        for r in &results {
            assert_eq!(r.id, "v2");
        }
    }

    // ── get_internal ────────────────────────────────────────────────────

    #[test]
    fn test_get_internal() {
        let db = setup_db_with_vectors();
        let result = db.get_internal("coll", "v1");
        assert!(result.is_some());
        let (vec, _) = result.unwrap();
        assert_eq!(vec, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_get_internal_not_found() {
        let db = setup_db_with_vectors();
        assert!(db.get_internal("coll", "missing").is_none());
    }

    #[test]
    fn test_get_internal_missing_collection() {
        let db = setup_db();
        assert!(db.get_internal("bad", "v1").is_none());
    }

    // ── delete_internal ─────────────────────────────────────────────────

    #[test]
    fn test_delete_internal() {
        let db = setup_db_with_vectors();
        assert!(db.delete_internal("coll", "v1").unwrap());
        assert!(db.get_internal("coll", "v1").is_none());
    }

    #[test]
    fn test_delete_internal_not_found() {
        let db = setup_db_with_vectors();
        assert!(!db.delete_internal("coll", "missing").unwrap());
    }

    #[test]
    fn test_delete_internal_missing_collection() {
        let db = setup_db();
        let result = db.delete_internal("bad", "v1");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── collection_len / collection_dimensions ──────────────────────────

    #[test]
    fn test_collection_len() {
        let db = setup_db_with_vectors();
        assert_eq!(db.collection_len("coll"), 2);
    }

    #[test]
    fn test_collection_len_missing() {
        let db = setup_db();
        assert_eq!(db.collection_len("bad"), 0);
    }

    #[test]
    fn test_collection_dimensions() {
        let db = setup_db();
        assert_eq!(db.collection_dimensions("coll"), Some(4));
    }

    #[test]
    fn test_collection_dimensions_missing() {
        let db = setup_db();
        assert_eq!(db.collection_dimensions("bad"), None);
    }

    // ── collection_stats_internal ───────────────────────────────────────

    #[test]
    fn test_collection_stats_internal() {
        let db = setup_db_with_vectors();
        let stats = db.collection_stats_internal("coll").unwrap();
        assert_eq!(stats.vector_count, 2);
        assert_eq!(stats.dimensions, 4);
    }

    #[test]
    fn test_collection_stats_missing() {
        let db = setup_db();
        let result = db.collection_stats_internal("bad");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── export_internal ─────────────────────────────────────────────────

    #[test]
    fn test_export_internal() {
        let db = setup_db_with_vectors();
        let entries = db.export_internal("coll").unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_export_internal_missing() {
        let db = setup_db();
        let result = db.export_internal("bad");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── ids_internal ────────────────────────────────────────────────────

    #[test]
    fn test_ids_internal() {
        let db = setup_db_with_vectors();
        let ids = db.ids_internal("coll").unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"v1".to_string()));
        assert!(ids.contains(&"v2".to_string()));
    }

    // ── compact_internal ────────────────────────────────────────────────

    #[test]
    fn test_compact_internal() {
        let db = setup_db_with_vectors();
        db.delete_internal("coll", "v1").unwrap();
        let removed = db.compact_internal("coll").unwrap();
        assert!(removed > 0);
    }

    #[test]
    fn test_compact_internal_missing() {
        let db = setup_db();
        let result = db.compact_internal("bad");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── needs_compaction_internal ────────────────────────────────────────

    #[test]
    fn test_needs_compaction_internal() {
        let db = setup_db_with_vectors();
        assert!(!db.needs_compaction_internal("coll", 0.5));
        db.delete_internal("coll", "v1").unwrap();
        // After deleting 1 of 2 vectors, 50% are deleted
        assert!(db.needs_compaction_internal("coll", 0.1));
    }

    // ── count_internal ──────────────────────────────────────────────────

    #[test]
    fn test_count_internal() {
        let db = setup_db_with_vectors();
        assert_eq!(db.count_internal("coll", None).unwrap(), 2);
    }

    #[test]
    fn test_count_internal_with_filter() {
        let db = setup_db_with_vectors();
        let filter = Filter::eq("tag", "a");
        let count = db.count_internal("coll", Some(&filter)).unwrap();
        assert_eq!(count, 1);
    }

    // ── deleted_count_internal ──────────────────────────────────────────

    #[test]
    fn test_deleted_count_internal() {
        let db = setup_db_with_vectors();
        assert_eq!(db.deleted_count_internal("coll"), 0);
        db.delete_internal("coll", "v1").unwrap();
        assert_eq!(db.deleted_count_internal("coll"), 1);
    }

    // ── search_ids_internal ─────────────────────────────────────────────

    #[test]
    fn test_search_ids_internal() {
        let db = setup_db_with_vectors();
        let results = db
            .search_ids_internal("coll", &[1.0, 0.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "v1");
    }

    // ── federated_search_internal ───────────────────────────────────────

    #[test]
    fn test_federated_search_internal() {
        let db = Database::in_memory();
        db.create_collection("c1", 4).unwrap();
        db.create_collection("c2", 4).unwrap();
        db.insert_internal("c1", "a", &[1.0, 0.0, 0.0, 0.0], None)
            .unwrap();
        db.insert_internal("c2", "b", &[0.0, 1.0, 0.0, 0.0], None)
            .unwrap();
        let results = db
            .federated_search_internal(&[1.0, 0.0, 0.0, 0.0], 5, &["c1", "c2"])
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_federated_search_missing_collection() {
        let db = setup_db();
        let result = db.federated_search_internal(&[1.0, 0.0, 0.0, 0.0], 5, &["coll", "bad"]);
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── TTL internal methods ────────────────────────────────────────────

    #[test]
    fn test_ttl_stats_internal() {
        let db = setup_db();
        db.insert_with_ttl_internal("coll", "v1", &[1.0, 0.0, 0.0, 0.0], None, Some(3600))
            .unwrap();
        let (total, _, _, _) = db.ttl_stats_internal("coll");
        assert_eq!(total, 1);
    }

    #[test]
    fn test_set_ttl_internal() {
        let db = setup_db_with_vectors();
        db.set_ttl_internal("coll", "v1", Some(7200)).unwrap();
        assert!(db.get_ttl_internal("coll", "v1").is_some());
    }

    #[test]
    fn test_set_ttl_internal_remove() {
        let db = setup_db();
        db.insert_with_ttl_internal("coll", "v1", &[1.0, 0.0, 0.0, 0.0], None, Some(3600))
            .unwrap();
        db.set_ttl_internal("coll", "v1", None).unwrap();
        assert!(db.get_ttl_internal("coll", "v1").is_none());
    }

    // ── search_with_options_internal ────────────────────────────────────

    #[test]
    fn test_search_with_options_internal() {
        let db = setup_db_with_vectors();
        let results = db
            .search_with_options_internal(
                "coll",
                &[1.0, 0.0, 0.0, 0.0],
                2,
                None,
                None,
                None,
                10,
            )
            .unwrap();
        assert!(!results.is_empty());
    }

    // ── insert_vec_with_ttl_internal ────────────────────────────────────

    #[test]
    fn test_insert_vec_with_ttl_internal() {
        let db = setup_db();
        db.insert_vec_with_ttl_internal("coll", "v1", vec![1.0, 0.0, 0.0, 0.0], None, Some(100))
            .unwrap();
        assert_eq!(db.collection_len("coll"), 1);
    }

    // ── expire_vectors_internal ─────────────────────────────────────────

    #[test]
    fn test_expire_vectors_internal_none_expired() {
        let db = setup_db();
        db.insert_with_ttl_internal("coll", "v1", &[1.0, 0.0, 0.0, 0.0], None, Some(99999))
            .unwrap();
        let expired = db.expire_vectors_internal("coll").unwrap();
        assert_eq!(expired, 0);
    }

    // ── needs_expiration_sweep_internal ──────────────────────────────────

    #[test]
    fn test_needs_expiration_sweep_internal() {
        let db = setup_db_with_vectors();
        assert!(!db.needs_expiration_sweep_internal("coll", 0.1));
    }
}
