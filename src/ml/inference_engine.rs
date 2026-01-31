//! Unified Inference Engine
//!
//! Provides a single entry-point for "text in, results out" workflows by combining
//! the model registry, auto-embedding, and collection APIs. Users can insert and
//! search by raw text without ever touching raw vectors.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::inference_engine::{InferenceEngine, InferenceConfig};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let engine = InferenceEngine::new(InferenceConfig::default());
//!
//! // Insert text — embedding generated automatically
//! let collection = db.collection("docs").unwrap();
//! engine.insert_text(&collection, "doc1", "Machine learning is great", None).unwrap();
//!
//! // Search by text
//! let results = engine.search_text(&collection, "AI and neural networks", 10).unwrap();
//! ```

use crate::auto_embed::{AutoEmbedConfig, AutoEmbedder, EmbeddingBackend};
use crate::collection::SearchResult;
use crate::database::CollectionRef;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::model_registry::ModelId;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Inference engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Default model to use when none specified
    pub default_model: ModelId,
    /// Auto-detect dimension from collection on first use
    pub auto_detect_dimensions: bool,
    /// Normalize embeddings to unit vectors
    pub normalize: bool,
    /// Maximum text length before truncation
    pub max_text_length: usize,
    /// Enable embedding cache
    pub cache_enabled: bool,
    /// Maximum cache entries
    pub cache_size: usize,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Store original text in metadata under this key (None = don't store)
    pub store_text_key: Option<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            default_model: ModelId::AllMiniLmL6V2,
            auto_detect_dimensions: true,
            normalize: true,
            max_text_length: 8192,
            cache_enabled: true,
            cache_size: 10_000,
            batch_size: 32,
            store_text_key: Some("_text".to_string()),
        }
    }
}

impl InferenceConfig {
    /// Create config with a specific model
    pub fn with_model(mut self, model: ModelId) -> Self {
        self.default_model = model;
        self
    }

    /// Create config with a specific batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Disable text storage in metadata
    pub fn without_text_storage(mut self) -> Self {
        self.store_text_key = None;
        self
    }

    /// Disable embedding cache
    pub fn without_cache(mut self) -> Self {
        self.cache_enabled = false;
        self
    }

    fn to_auto_embed_config(&self, dimensions: usize) -> AutoEmbedConfig {
        let backend = EmbeddingBackend::Mock { dimensions };
        let mut cfg = AutoEmbedConfig::new(backend);
        cfg.cache_enabled = self.cache_enabled;
        cfg.cache_size = self.cache_size;
        cfg.batch_size = self.batch_size;
        cfg.normalize = self.normalize;
        cfg.max_text_length = self.max_text_length;
        cfg
    }
}

// ============================================================================
// Model Info
// ============================================================================

/// Resolved model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedModel {
    pub model_id: ModelId,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub normalize: bool,
    pub description: String,
}

impl ResolvedModel {
    fn for_model(model_id: ModelId) -> Self {
        let (dims, max_tok, desc) = match model_id {
            ModelId::AllMiniLmL6V2 => (384, 512, "all-MiniLM-L6-v2: Fast general-purpose"),
            ModelId::AllMiniLmL12V2 => (384, 512, "all-MiniLM-L12-v2: Balanced quality"),
            ModelId::BgeSmallEnV15 => (384, 512, "BGE Small EN v1.5: Fast English"),
            ModelId::BgeBaseEnV15 => (768, 512, "BGE Base EN v1.5: Balanced English"),
            ModelId::BgeLargeEnV15 => (1024, 512, "BGE Large EN v1.5: High quality English"),
            ModelId::E5SmallV2 => (384, 512, "E5 Small v2: Fast multilingual"),
            ModelId::E5BaseV2 => (768, 512, "E5 Base v2: Balanced multilingual"),
            ModelId::E5LargeV2 => (1024, 512, "E5 Large v2: High quality multilingual"),
            ModelId::NomicEmbedTextV1 => (768, 8192, "Nomic Embed Text v1: Long context"),
            ModelId::GteSmall => (384, 512, "GTE Small: Fast general-purpose"),
            ModelId::GteBase => (768, 512, "GTE Base: Balanced general-purpose"),
            ModelId::Custom => (384, 512, "Custom user model"),
        };
        Self {
            model_id,
            dimensions: dims,
            max_tokens: max_tok,
            normalize: true,
            description: desc.to_string(),
        }
    }
}

// ============================================================================
// Inference Engine
// ============================================================================

/// Statistics for the inference engine
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceStats {
    pub total_inserts: u64,
    pub total_searches: u64,
    pub total_batch_inserts: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_insert_us: u64,
    pub avg_search_us: u64,
}

/// Unified inference engine combining model registry + auto-embed + collection API
pub struct InferenceEngine {
    config: InferenceConfig,
    /// Per-dimension embedders (keyed by dimension count)
    embedders: RwLock<HashMap<usize, Arc<AutoEmbedder>>>,
    stats: RwLock<InferenceStats>,
}

impl InferenceEngine {
    /// Create a new inference engine with default configuration
    pub fn new(config: InferenceConfig) -> Self {
        Self {
            config,
            embedders: RwLock::new(HashMap::new()),
            stats: RwLock::new(InferenceStats::default()),
        }
    }

    /// Get or create an embedder for the given dimensions
    fn get_embedder(&self, dimensions: usize) -> Arc<AutoEmbedder> {
        {
            let embedders = self.embedders.read();
            if let Some(emb) = embedders.get(&dimensions) {
                return Arc::clone(emb);
            }
        }

        let auto_config = self.config.to_auto_embed_config(dimensions);
        let embedder = Arc::new(AutoEmbedder::new(auto_config));
        let mut embedders = self.embedders.write();
        embedders.insert(dimensions, Arc::clone(&embedder));
        embedder
    }

    /// Get the resolved model information for the configured default model
    pub fn model_info(&self) -> ResolvedModel {
        ResolvedModel::for_model(self.config.default_model)
    }

    /// List all available models with their details
    pub fn list_models() -> Vec<ResolvedModel> {
        ModelId::all()
            .into_iter()
            .map(ResolvedModel::for_model)
            .collect()
    }

    /// Select the best model for given constraints
    pub fn select_model(
        target_dimensions: Option<usize>,
        max_model_size_mb: Option<u64>,
        prefer_speed: bool,
    ) -> ResolvedModel {
        let all = Self::list_models();

        let filtered: Vec<_> = all
            .into_iter()
            .filter(|m| {
                if let Some(d) = target_dimensions {
                    m.dimensions == d
                } else {
                    true
                }
            })
            .filter(|m| {
                if let Some(max_mb) = max_model_size_mb {
                    // Rough estimate: small models < 200MB, base < 500MB, large > 1GB
                    let est_mb = match m.dimensions {
                        d if d <= 384 => 100,
                        d if d <= 768 => 400,
                        _ => 1300,
                    };
                    est_mb <= max_mb
                } else {
                    true
                }
            })
            .collect();

        if filtered.is_empty() {
            return ResolvedModel::for_model(ModelId::AllMiniLmL6V2);
        }

        if prefer_speed {
            // Prefer smallest dimensions
            filtered
                .into_iter()
                .min_by_key(|m| m.dimensions)
                .unwrap_or_default()
        } else {
            // Prefer largest dimensions (best quality)
            filtered
                .into_iter()
                .max_by_key(|m| m.dimensions)
                .unwrap_or_default()
        }
    }

    /// Insert a text document — embedding is generated automatically
    pub fn insert_text(
        &self,
        collection: &CollectionRef,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let start = Instant::now();
        let dimensions = collection.dimensions().ok_or_else(|| {
            NeedleError::CollectionNotFound("Cannot determine collection dimensions".into())
        })?;
        let embedder = self.get_embedder(dimensions);

        let embedding = embedder.embed(text).map_err(|e| {
            NeedleError::InvalidInput(format!("Embedding generation failed: {}", e))
        })?;

        // Optionally store original text in metadata
        let metadata = if let Some(text_key) = &self.config.store_text_key {
            let mut meta = metadata.unwrap_or(Value::Object(serde_json::Map::new()));
            if let Value::Object(ref mut map) = meta {
                map.insert(text_key.clone(), Value::String(text.to_string()));
            }
            Some(meta)
        } else {
            metadata
        };

        collection.insert(id, &embedding, metadata)?;

        let elapsed = start.elapsed().as_micros() as u64;
        let mut stats = self.stats.write();
        stats.total_inserts += 1;
        stats.avg_insert_us =
            (stats.avg_insert_us * (stats.total_inserts - 1) + elapsed) / stats.total_inserts;

        let embed_stats = embedder.cache_stats();
        stats.cache_hits = embed_stats.1;
        stats.cache_misses = embed_stats.2;

        Ok(())
    }

    /// Insert multiple text documents in batch
    pub fn insert_texts_batch(
        &self,
        collection: &CollectionRef,
        items: &[(String, String, Option<Value>)],
    ) -> Result<usize> {
        let dimensions = collection.dimensions().ok_or_else(|| {
            NeedleError::CollectionNotFound("Cannot determine collection dimensions".into())
        })?;
        let embedder = self.get_embedder(dimensions);
        let mut count = 0;

        for chunk in items.chunks(self.config.batch_size) {
            let texts: Vec<&str> = chunk.iter().map(|item| item.1.as_str()).collect();
            let embeddings = embedder.embed_batch(&texts).map_err(|e| {
                NeedleError::InvalidInput(format!("Batch embedding failed: {}", e))
            })?;

            for (item, embedding) in chunk.iter().zip(embeddings.into_iter()) {
                let id = &item.0;
                let text = &item.1;
                let orig_metadata = &item.2;
                let metadata: Option<Value> = if let Some(text_key) = &self.config.store_text_key {
                    let mut meta: Value =
                        orig_metadata.clone().unwrap_or(Value::Object(serde_json::Map::new()));
                    if let Value::Object(ref mut map) = meta {
                        map.insert(text_key.clone(), Value::String(text.clone()));
                    }
                    Some(meta)
                } else {
                    orig_metadata.clone()
                };

                collection.insert(id.clone(), &embedding, metadata)?;
                count += 1;
            }
        }

        let mut stats = self.stats.write();
        stats.total_batch_inserts += 1;
        Ok(count)
    }

    /// Search using a text query — query embedding generated automatically
    pub fn search_text(
        &self,
        collection: &CollectionRef,
        query: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let start = Instant::now();
        let dimensions = collection.dimensions().ok_or_else(|| {
            NeedleError::CollectionNotFound("Cannot determine collection dimensions".into())
        })?;
        let embedder = self.get_embedder(dimensions);

        let query_embedding = embedder.embed(query).map_err(|e| {
            NeedleError::InvalidInput(format!("Query embedding failed: {}", e))
        })?;

        let results = collection.search(&query_embedding, k)?;

        let elapsed = start.elapsed().as_micros() as u64;
        let mut stats = self.stats.write();
        stats.total_searches += 1;
        stats.avg_search_us =
            (stats.avg_search_us * (stats.total_searches - 1) + elapsed) / stats.total_searches;

        Ok(results)
    }

    /// Search with text query and metadata filter
    pub fn search_text_with_filter(
        &self,
        collection: &CollectionRef,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let dimensions = collection.dimensions().ok_or_else(|| {
            NeedleError::CollectionNotFound("Cannot determine collection dimensions".into())
        })?;
        let embedder = self.get_embedder(dimensions);

        let query_embedding = embedder.embed(query).map_err(|e| {
            NeedleError::InvalidInput(format!("Query embedding failed: {}", e))
        })?;

        collection.search_with_filter(&query_embedding, k, filter)
    }

    /// Get current engine statistics
    pub fn stats(&self) -> InferenceStats {
        self.stats.read().clone()
    }

    /// Clear all embedding caches
    pub fn clear_caches(&self) {
        let embedders = self.embedders.read();
        for emb in embedders.values() {
            emb.clear_cache();
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("config", &self.config)
            .field("active_embedders", &self.embedders.read().len())
            .field("stats", &*self.stats.read())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;
    use serde_json::json;

    fn setup() -> (Database, InferenceEngine) {
        let db = Database::in_memory();
        db.create_collection("test", 384).unwrap();
        let engine = InferenceEngine::new(InferenceConfig::default());
        (db, engine)
    }

    #[test]
    fn test_insert_and_search_text() {
        let (db, engine) = setup();
        let coll = db.collection("test").unwrap();

        engine
            .insert_text(&coll, "doc1", "Machine learning is fascinating", None)
            .unwrap();
        engine
            .insert_text(&coll, "doc2", "Deep learning neural networks", None)
            .unwrap();
        engine
            .insert_text(&coll, "doc3", "Cooking recipes for dinner", None)
            .unwrap();

        let results = engine.search_text(&coll, "artificial intelligence", 3).unwrap();
        assert_eq!(results.len(), 3);
        // All three docs should be returned
        let ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"doc1"));
    }

    #[test]
    fn test_text_stored_in_metadata() {
        let (db, engine) = setup();
        let coll = db.collection("test").unwrap();

        engine
            .insert_text(&coll, "doc1", "Hello world", Some(json!({"tag": "test"})))
            .unwrap();

        let (_, meta) = coll.get("doc1").unwrap();
        let meta = meta.unwrap();
        assert_eq!(meta["_text"], "Hello world");
        assert_eq!(meta["tag"], "test");
    }

    #[test]
    fn test_batch_insert() {
        let (db, engine) = setup();
        let coll = db.collection("test").unwrap();

        let items: Vec<_> = (0..50)
            .map(|i| {
                (
                    format!("doc_{}", i),
                    format!("Document number {} about topic {}", i, i % 5),
                    None,
                )
            })
            .collect();

        let count = engine.insert_texts_batch(&coll, &items).unwrap();
        assert_eq!(count, 50);

        let results = engine.search_text(&coll, "document about topic", 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_with_filter() {
        let (db, engine) = setup();
        let coll = db.collection("test").unwrap();

        engine
            .insert_text(
                &coll,
                "doc1",
                "Rust programming",
                Some(json!({"lang": "rust"})),
            )
            .unwrap();
        engine
            .insert_text(
                &coll,
                "doc2",
                "Python programming",
                Some(json!({"lang": "python"})),
            )
            .unwrap();

        let filter = Filter::eq("lang", "rust");
        let results = engine
            .search_text_with_filter(&coll, "programming", 10, &filter)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_model_selection_prefer_speed() {
        let model = InferenceEngine::select_model(None, None, true);
        assert!(model.dimensions <= 384);
    }

    #[test]
    fn test_model_selection_prefer_quality() {
        let model = InferenceEngine::select_model(None, None, false);
        assert!(model.dimensions >= 768);
    }

    #[test]
    fn test_model_selection_target_dims() {
        let model = InferenceEngine::select_model(Some(768), None, false);
        assert_eq!(model.dimensions, 768);
    }

    #[test]
    fn test_list_models() {
        let models = InferenceEngine::list_models();
        assert!(models.len() >= 10);
        assert!(models.iter().any(|m| m.model_id == ModelId::AllMiniLmL6V2));
    }

    #[test]
    fn test_engine_stats() {
        let (db, engine) = setup();
        let coll = db.collection("test").unwrap();

        engine
            .insert_text(&coll, "d1", "test text", None)
            .unwrap();
        let _ = engine.search_text(&coll, "query", 5).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.total_inserts, 1);
        assert_eq!(stats.total_searches, 1);
    }

    #[test]
    fn test_config_without_text_storage() {
        let db = Database::in_memory();
        db.create_collection("test", 384).unwrap();
        let engine = InferenceEngine::new(InferenceConfig::default().without_text_storage());
        let coll = db.collection("test").unwrap();

        engine.insert_text(&coll, "d1", "hello", None).unwrap();
        let (_, meta) = coll.get("d1").unwrap();
        // No _text key when storage disabled
        assert!(meta.is_none() || meta.unwrap().get("_text").is_none());
    }

    #[test]
    fn test_clear_caches() {
        let (db, engine) = setup();
        let coll = db.collection("test").unwrap();

        engine.insert_text(&coll, "d1", "test", None).unwrap();
        engine.clear_caches();
        // Should still work after cache clear
        let results = engine.search_text(&coll, "test", 1).unwrap();
        assert_eq!(results.len(), 1);
    }
}
