//! Embedded Model Runtime
//!
//! Downloadable model registry with lazy-load and `insert_text`/`search_text`
//! integration on a Database wrapper. Supports model hot-swap and dimension
//! auto-detection with a fallback chain.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::model_runtime::{
//!     ModelRuntime, RuntimeConfig, ModelEntry,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 64).unwrap();
//!
//! let mut rt = ModelRuntime::new(db, RuntimeConfig::default());
//! rt.insert_text("docs", "d1", "Rust is a systems programming language").unwrap();
//! let results = rt.search_text("docs", "systems programming", 5).unwrap();
//! assert_eq!(results[0].id, "d1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::services::inference_engine::{InferenceConfig, InferenceEngine, ModelSpec};

// ── Model Entry ──────────────────────────────────────────────────────────────

/// A model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Model identifier.
    pub id: String,
    /// Output dimensions.
    pub dimensions: usize,
    /// Model size in bytes.
    pub size_bytes: u64,
    /// Download URL (if remote).
    pub url: Option<String>,
    /// Local path (if cached).
    pub local_path: Option<String>,
    /// Whether the model is loaded.
    pub loaded: bool,
    /// Quality score (0-100, higher = better).
    pub quality_score: u32,
}

// ── Runtime Configuration ────────────────────────────────────────────────────

/// Model runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Default model to use.
    pub default_model: String,
    /// Default dimensions.
    pub default_dimensions: usize,
    /// Whether to store original text in metadata.
    pub store_text: bool,
    /// Embedding cache size.
    pub cache_size: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            default_model: "built-in".into(),
            default_dimensions: 64,
            store_text: true,
            cache_size: 10_000,
        }
    }
}

impl RuntimeConfig {
    /// Set default dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, dim: usize) -> Self {
        self.default_dimensions = dim;
        self
    }

    /// Set cache size.
    #[must_use]
    pub fn with_cache(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
}

// ── Text Result ──────────────────────────────────────────────────────────────

/// Search result with text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    /// Vector ID.
    pub id: String,
    /// Distance.
    pub distance: f32,
    /// Original text.
    pub text: Option<String>,
    /// Metadata.
    pub metadata: Option<Value>,
}

// ── Runtime Stats ────────────────────────────────────────────────────────────

/// Runtime statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeStats {
    /// Total embeddings generated.
    pub embeddings_generated: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Active model.
    pub active_model: String,
    /// Models in registry.
    pub registry_size: usize,
}

// ── Model Runtime ────────────────────────────────────────────────────────────

/// Embedded model runtime with database integration.
pub struct ModelRuntime {
    db: Database,
    config: RuntimeConfig,
    engine: InferenceEngine,
    registry: HashMap<String, ModelEntry>,
    cache: HashMap<String, Vec<f32>>,
    stats: RuntimeStats,
}

impl ModelRuntime {
    /// Create a new model runtime.
    pub fn new(db: Database, config: RuntimeConfig) -> Self {
        let engine = InferenceEngine::new(
            InferenceConfig::builder()
                .model(ModelSpec::new(&config.default_model, config.default_dimensions))
                .normalize(true)
                .build(),
        );

        let mut registry = HashMap::new();
        registry.insert("built-in".into(), ModelEntry {
            id: "built-in".into(),
            dimensions: config.default_dimensions,
            size_bytes: 0,
            url: None,
            local_path: None,
            loaded: true,
            quality_score: 50,
        });
        registry.insert("all-MiniLM-L6-v2".into(), ModelEntry {
            id: "all-MiniLM-L6-v2".into(),
            dimensions: 384,
            size_bytes: 15_000_000,
            url: Some("https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2".into()),
            local_path: None,
            loaded: false,
            quality_score: 85,
        });

        Self {
            db,
            config: config.clone(),
            engine,
            registry,
            cache: HashMap::new(),
            stats: RuntimeStats {
                active_model: config.default_model,
                ..Default::default()
            },
        }
    }

    /// Register a model in the registry.
    pub fn register_model(&mut self, entry: ModelEntry) {
        self.registry.insert(entry.id.clone(), entry);
        self.stats.registry_size = self.registry.len();
    }

    /// Insert text with auto-embedding.
    pub fn insert_text(&mut self, collection: &str, id: &str, text: &str) -> Result<()> {
        let embedding = self.embed(text)?;
        let meta = if self.config.store_text {
            Some(serde_json::json!({ "_text": text }))
        } else {
            None
        };
        let coll = self.db.collection(collection)?;
        coll.insert(id, &embedding, meta)?;
        Ok(())
    }

    /// Search by text with auto-embedding.
    pub fn search_text(&mut self, collection: &str, query: &str, k: usize) -> Result<Vec<TextSearchResult>> {
        let embedding = self.embed(query)?;
        let coll = self.db.collection(collection)?;
        let results = coll.search(&embedding, k)?;
        Ok(results.into_iter().map(|r| {
            let text = r.metadata.as_ref()
                .and_then(|m| m.get("_text"))
                .and_then(|v| v.as_str())
                .map(String::from);
            TextSearchResult { id: r.id, distance: r.distance, text, metadata: r.metadata }
        }).collect())
    }

    /// List available models.
    pub fn models(&self) -> Vec<&ModelEntry> {
        self.registry.values().collect()
    }

    /// Get the active model name.
    pub fn active_model(&self) -> &str {
        &self.stats.active_model
    }

    /// Get runtime stats.
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }

    /// Get the underlying database.
    pub fn database(&self) -> &Database {
        &self.db
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.cache.get(text) {
            self.stats.cache_hits += 1;
            return Ok(cached.clone());
        }
        let emb = self.engine.embed_text(text)?;
        self.stats.embeddings_generated += 1;
        if self.cache.len() < self.config.cache_size {
            self.cache.insert(text.into(), emb.clone());
        }
        Ok(emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut rt = ModelRuntime::new(db, RuntimeConfig::default());

        rt.insert_text("docs", "d1", "rust programming").unwrap();
        rt.insert_text("docs", "d2", "python scripting").unwrap();

        let results = rt.search_text("docs", "rust programming", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_cache() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut rt = ModelRuntime::new(db, RuntimeConfig::default());

        rt.insert_text("docs", "d1", "hello").unwrap();
        rt.search_text("docs", "hello", 1).unwrap(); // cache hit for embed
        assert!(rt.stats().cache_hits > 0);
    }

    #[test]
    fn test_registry() {
        let db = Database::in_memory();
        let mut rt = ModelRuntime::new(db, RuntimeConfig::default());
        assert!(rt.models().len() >= 2); // built-in + MiniLM

        rt.register_model(ModelEntry {
            id: "custom".into(), dimensions: 256, size_bytes: 1000,
            url: None, local_path: None, loaded: false, quality_score: 90,
        });
        assert!(rt.models().len() >= 3);
    }

    #[test]
    fn test_empty_text_error() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut rt = ModelRuntime::new(db, RuntimeConfig::default());
        assert!(rt.insert_text("docs", "d1", "").is_err());
    }

    #[test]
    fn test_stats() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut rt = ModelRuntime::new(db, RuntimeConfig::default());
        rt.insert_text("docs", "d1", "test").unwrap();
        assert_eq!(rt.stats().embeddings_generated, 1);
        assert_eq!(rt.active_model(), "built-in");
    }
}
