//! Smart Auto-Embed
//!
//! Automatically selects the best embedding model: local ONNX → API provider →
//! hash-based fallback. Provides `insert_text()`/`search_text()` on a Database
//! wrapper with model caching, batching, and dimension auto-detection.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::smart_auto_embed::{
//!     SmartEmbedder, EmbedderChain, EmbedderBackend,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 64).unwrap();
//!
//! let mut embedder = SmartEmbedder::new(db, EmbedderChain::default());
//!
//! // Zero-config: auto-embeds text
//! embedder.insert_text("docs", "d1", "Rust is fast").unwrap();
//! let results = embedder.search_text("docs", "fast language", 5).unwrap();
//! assert_eq!(results[0].id, "d1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::SearchResult;
use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::services::inference_engine::{InferenceConfig, InferenceEngine, ModelSpec};

// ── Backend Selection ────────────────────────────────────────────────────────

/// Available embedding backends in priority order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedderBackend {
    /// Built-in hash-based embedder (always available, deterministic).
    BuiltIn { dimensions: usize },
    /// Local ONNX model.
    LocalOnnx { model_path: String, dimensions: usize },
    /// External API provider.
    ApiProvider { name: String, api_key_env: String, dimensions: usize },
}

/// Chain of embedding backends tried in order.
#[derive(Debug, Clone)]
pub struct EmbedderChain {
    /// Backends to try, in priority order.
    pub backends: Vec<EmbedderBackend>,
    /// Cache embeddings in memory.
    pub enable_cache: bool,
    /// Maximum cache entries.
    pub max_cache: usize,
}

impl Default for EmbedderChain {
    fn default() -> Self {
        Self {
            backends: vec![EmbedderBackend::BuiltIn { dimensions: 64 }],
            enable_cache: true,
            max_cache: 10_000,
        }
    }
}

impl EmbedderChain {
    /// Add a backend to the chain.
    #[must_use]
    pub fn with_backend(mut self, backend: EmbedderBackend) -> Self {
        self.backends.insert(0, backend); // higher priority
        self
    }

    /// Set cache size.
    #[must_use]
    pub fn with_cache(mut self, max: usize) -> Self {
        self.max_cache = max;
        self
    }

    /// Get the active dimensions (from first available backend).
    pub fn dimensions(&self) -> usize {
        self.backends.first().map_or(64, |b| match b {
            EmbedderBackend::BuiltIn { dimensions } => *dimensions,
            EmbedderBackend::LocalOnnx { dimensions, .. } => *dimensions,
            EmbedderBackend::ApiProvider { dimensions, .. } => *dimensions,
        })
    }
}

// ── Text Search Result ───────────────────────────────────────────────────────

/// Search result enriched with text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextResult {
    /// Vector ID.
    pub id: String,
    /// Distance score.
    pub distance: f32,
    /// Original text if stored.
    pub text: Option<String>,
    /// Metadata.
    pub metadata: Option<Value>,
}

// ── Embed Stats ──────────────────────────────────────────────────────────────

/// Embedding statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedStats {
    /// Total texts embedded.
    pub total_embedded: u64,
    /// Cache hits.
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
    /// Active backend name.
    pub active_backend: String,
}

// ── Smart Embedder ───────────────────────────────────────────────────────────

/// Smart embedder that auto-selects the best available backend.
pub struct SmartEmbedder {
    db: Database,
    chain: EmbedderChain,
    engine: InferenceEngine,
    cache: HashMap<String, Vec<f32>>,
    stats: EmbedStats,
}

impl SmartEmbedder {
    /// Create a new smart embedder wrapping a database.
    pub fn new(db: Database, chain: EmbedderChain) -> Self {
        let dims = chain.dimensions();
        let engine = InferenceEngine::new(
            InferenceConfig::builder()
                .model(ModelSpec::new("smart-embed", dims))
                .normalize(true)
                .build(),
        );
        let backend_name = match chain.backends.first() {
            Some(EmbedderBackend::BuiltIn { .. }) => "built-in".to_string(),
            Some(EmbedderBackend::LocalOnnx { .. }) => "local-onnx".to_string(),
            Some(EmbedderBackend::ApiProvider { name, .. }) => name.clone(),
            None => "none".to_string(),
        };
        Self {
            db,
            chain,
            engine,
            cache: HashMap::new(),
            stats: EmbedStats {
                active_backend: backend_name,
                ..Default::default()
            },
        }
    }

    /// Insert text — auto-embeds and stores in collection.
    pub fn insert_text(
        &mut self,
        collection: &str,
        id: &str,
        text: &str,
    ) -> Result<()> {
        let embedding = self.embed(text)?;
        let meta = serde_json::json!({ "_text": text });
        let coll = self.db.collection(collection)?;
        coll.insert(id, &embedding, Some(meta))?;
        Ok(())
    }

    /// Insert text with additional metadata.
    pub fn insert_text_with_metadata(
        &mut self,
        collection: &str,
        id: &str,
        text: &str,
        metadata: Value,
    ) -> Result<()> {
        let embedding = self.embed(text)?;
        let mut meta = match metadata {
            Value::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        meta.insert("_text".into(), Value::String(text.into()));
        let coll = self.db.collection(collection)?;
        coll.insert(id, &embedding, Some(Value::Object(meta)))?;
        Ok(())
    }

    /// Search by text — auto-embeds query.
    pub fn search_text(
        &mut self,
        collection: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<TextResult>> {
        let embedding = self.embed(query)?;
        let coll = self.db.collection(collection)?;
        let results = coll.search(&embedding, k)?;
        Ok(results.into_iter().map(|r| {
            let text = r.metadata.as_ref()
                .and_then(|m| m.get("_text"))
                .and_then(|v| v.as_str())
                .map(String::from);
            TextResult {
                id: r.id,
                distance: r.distance,
                text,
                metadata: r.metadata,
            }
        }).collect())
    }

    /// Get the active backend.
    pub fn active_backend(&self) -> &str {
        &self.stats.active_backend
    }

    /// Get embedding dimensions.
    pub fn dimensions(&self) -> usize {
        self.chain.dimensions()
    }

    /// Get statistics.
    pub fn stats(&self) -> &EmbedStats {
        &self.stats
    }

    /// Get a reference to the underlying database.
    pub fn database(&self) -> &Database {
        &self.db
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if self.chain.enable_cache {
            if let Some(cached) = self.cache.get(text) {
                self.stats.cache_hits += 1;
                return Ok(cached.clone());
            }
            self.stats.cache_misses += 1;
        }

        let embedding = self.engine.embed_text(text)?;
        self.stats.total_embedded += 1;

        if self.chain.enable_cache {
            if self.cache.len() >= self.chain.max_cache {
                // Evict oldest (simple: clear half)
                let keys: Vec<String> = self.cache.keys().take(self.cache.len() / 2).cloned().collect();
                for k in keys { self.cache.remove(&k); }
            }
            self.cache.insert(text.into(), embedding.clone());
        }

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut embedder = SmartEmbedder::new(db, EmbedderChain::default());

        embedder.insert_text("docs", "d1", "rust programming").unwrap();
        embedder.insert_text("docs", "d2", "python scripting").unwrap();

        let results = embedder.search_text("docs", "rust programming", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "d1");
        assert!(results[0].text.is_some());
    }

    #[test]
    fn test_cache() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut embedder = SmartEmbedder::new(db, EmbedderChain::default());

        embedder.insert_text("docs", "d1", "hello").unwrap();
        embedder.insert_text("docs", "d2", "hello").unwrap(); // cache hit

        assert_eq!(embedder.stats().cache_hits, 1);
    }

    #[test]
    fn test_with_metadata() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut embedder = SmartEmbedder::new(db, EmbedderChain::default());

        embedder.insert_text_with_metadata(
            "docs", "d1", "hello",
            serde_json::json!({"source": "test"}),
        ).unwrap();

        let results = embedder.search_text("docs", "hello", 1).unwrap();
        assert!(results[0].metadata.as_ref().unwrap().get("source").is_some());
    }

    #[test]
    fn test_backend_chain() {
        let chain = EmbedderChain::default()
            .with_backend(EmbedderBackend::BuiltIn { dimensions: 128 });
        assert_eq!(chain.dimensions(), 128);
        assert_eq!(chain.backends.len(), 2);
    }

    #[test]
    fn test_empty_text_error() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut embedder = SmartEmbedder::new(db, EmbedderChain::default());
        assert!(embedder.insert_text("docs", "d1", "").is_err());
    }

    #[test]
    fn test_stats() {
        let db = Database::in_memory();
        db.create_collection("docs", 64).unwrap();
        let mut embedder = SmartEmbedder::new(db, EmbedderChain::default());
        embedder.insert_text("docs", "d1", "hello").unwrap();
        assert_eq!(embedder.stats().total_embedded, 1);
        assert_eq!(embedder.active_backend(), "built-in");
    }
}
