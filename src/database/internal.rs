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
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert(id, vector, metadata)?;
        self.mark_modified();
        Ok(())
    }

    pub(crate) fn insert_vec_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_vec(id, vector, metadata)?;
        self.mark_modified();
        if let Some(aim) = &self.adaptive_index_manager {
            aim.record_insert();
        }
        #[cfg(feature = "observability")]
        if let Some(metrics) = &self.dashboard_metrics {
            metrics.record_insert(collection);
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
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_with_ttl(id, vector, metadata, ttl_seconds)?;
        self.mark_modified();
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
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_vec_with_ttl(id, vector, metadata, ttl_seconds)?;
        self.mark_modified();
        Ok(())
    }

    pub(crate) fn update_internal(
        &self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.update(id, vector, metadata)?;
        self.mark_modified();
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
        };

        let mut state = self.state.write();
        state.collections.insert(name, collection);
        drop(state);
        self.mark_modified();

        Ok(manifest)
    }
}
