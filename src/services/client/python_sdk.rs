#![allow(clippy::unwrap_used)]
//! Python SDK Enhancement
//!
//! Enhanced Python SDK wrapper with TextCollection for zero-config RAG,
//! type-safe builders, and auto-embedding support.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::python_sdk::{
//!     PythonSdkConfig, TextCollectionApi, PySdkResult,
//! };
//!
//! let config = PythonSdkConfig::default();
//! let mut api = TextCollectionApi::new("docs", config);
//! api.add("d1", "Rust is fast").unwrap();
//! let results = api.search("fast language", 5).unwrap();
//! assert_eq!(results[0].id, "d1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::services::inference_engine::{InferenceConfig, InferenceEngine, ModelSpec};

// ── Configuration ────────────────────────────────────────────────────────────

/// Python SDK configuration.
#[derive(Debug, Clone)]
pub struct PythonSdkConfig {
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Store original text in metadata.
    pub store_text: bool,
    /// Distance function.
    pub distance: DistanceFunction,
}

impl Default for PythonSdkConfig {
    fn default() -> Self {
        Self {
            dimensions: 64,
            store_text: true,
            distance: DistanceFunction::Cosine,
        }
    }
}

// ── SDK Result ───────────────────────────────────────────────────────────────

/// Python-friendly search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PySdkResult {
    /// Document ID.
    pub id: String,
    /// Distance score.
    pub distance: f32,
    /// Original text.
    pub text: Option<String>,
    /// Metadata as JSON dict.
    pub metadata: Option<Value>,
}

// ── Text Collection API ──────────────────────────────────────────────────────

/// Simplified API matching the Python SDK pattern: `add()`, `search()`, `get()`.
pub struct TextCollectionApi {
    name: String,
    collection: Collection,
    engine: InferenceEngine,
    texts: HashMap<String, String>,
    config: PythonSdkConfig,
}

impl TextCollectionApi {
    /// Create a new text collection.
    pub fn new(name: &str, config: PythonSdkConfig) -> Self {
        let coll = Collection::new(
            CollectionConfig::new(name, config.dimensions).with_distance(config.distance),
        );
        let engine = InferenceEngine::new(
            InferenceConfig::builder()
                .model(ModelSpec::new("py-sdk", config.dimensions))
                .normalize(true)
                .build(),
        );
        Self {
            name: name.into(),
            collection: coll,
            engine,
            texts: HashMap::new(),
            config,
        }
    }

    /// Add a text document (Python: `col.add("id", "text")`).
    pub fn add(&mut self, id: &str, text: &str) -> Result<()> {
        let emb = self.engine.embed_text(text)?;
        let meta = if self.config.store_text {
            Some(serde_json::json!({ "_text": text }))
        } else {
            None
        };
        self.collection.insert(id, &emb, meta)?;
        self.texts.insert(id.into(), text.into());
        Ok(())
    }

    /// Add with metadata (Python: `col.add("id", "text", metadata={"k": "v"})`).
    pub fn add_with_metadata(&mut self, id: &str, text: &str, metadata: Value) -> Result<()> {
        let emb = self.engine.embed_text(text)?;
        let mut meta = match metadata {
            Value::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        if self.config.store_text {
            meta.insert("_text".into(), Value::String(text.into()));
        }
        self.collection.insert(id, &emb, Some(Value::Object(meta)))?;
        self.texts.insert(id.into(), text.into());
        Ok(())
    }

    /// Search by text (Python: `col.search("query", k=10)`).
    pub fn search(&mut self, query: &str, k: usize) -> Result<Vec<PySdkResult>> {
        let emb = self.engine.embed_text(query)?;
        let results = self.collection.search(&emb, k)?;
        Ok(results.into_iter().map(|r| PySdkResult {
            text: self.texts.get(&r.id).cloned(),
            id: r.id, distance: r.distance, metadata: r.metadata,
        }).collect())
    }

    /// Get a document by ID (Python: `col.get("id")`).
    pub fn get(&self, id: &str) -> Option<PySdkResult> {
        self.collection.get(id).map(|(vec, meta)| PySdkResult {
            id: id.into(),
            distance: 0.0,
            text: self.texts.get(id).cloned(),
            metadata: meta.cloned(),
        })
    }

    /// Delete a document (Python: `col.delete("id")`).
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        self.texts.remove(id);
        self.collection.delete(id)
    }

    /// Count documents (Python: `len(col)`).
    pub fn count(&self) -> usize {
        self.collection.len()
    }

    /// Collection name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_search() {
        let mut api = TextCollectionApi::new("test", PythonSdkConfig::default());
        api.add("d1", "rust programming").unwrap();
        api.add("d2", "python scripting").unwrap();
        let results = api.search("rust", 5).unwrap();
        assert_eq!(results[0].id, "d1");
        assert_eq!(results[0].text.as_deref(), Some("rust programming"));
    }

    #[test]
    fn test_get() {
        let mut api = TextCollectionApi::new("test", PythonSdkConfig::default());
        api.add("d1", "hello").unwrap();
        let result = api.get("d1").unwrap();
        assert_eq!(result.text.as_deref(), Some("hello"));
    }

    #[test]
    fn test_delete() {
        let mut api = TextCollectionApi::new("test", PythonSdkConfig::default());
        api.add("d1", "hello").unwrap();
        api.delete("d1").unwrap();
        assert_eq!(api.count(), 0);
    }

    #[test]
    fn test_with_metadata() {
        let mut api = TextCollectionApi::new("test", PythonSdkConfig::default());
        api.add_with_metadata("d1", "hello", serde_json::json!({"source": "test"})).unwrap();
        let result = api.get("d1").unwrap();
        assert!(result.metadata.unwrap().get("source").is_some());
    }

    #[test]
    fn test_count() {
        let mut api = TextCollectionApi::new("test", PythonSdkConfig::default());
        api.add("d1", "a").unwrap();
        api.add("d2", "b").unwrap();
        assert_eq!(api.count(), 2);
    }
}
