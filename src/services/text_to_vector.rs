//! Zero-Config Text-to-Vector
//!
//! Auto-embed text using the built-in inference engine. Provides `TextCollection`
//! wrapper with `insert_text()` and `search_text()` — no external embedding
//! service, no configuration, just text in / results out.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::text_to_vector::{
//!     TextVectorCollection, TextVectorConfig,
//! };
//!
//! let mut col = TextVectorCollection::new(TextVectorConfig::default());
//! col.insert_text("doc1", "Rust is a systems programming language").unwrap();
//! col.insert_text("doc2", "Python is great for data science").unwrap();
//!
//! let results = col.search_text("systems programming", 5).unwrap();
//! assert_eq!(results[0].id, "doc1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::services::inference_engine::{InferenceConfig, InferenceEngine, ModelSpec};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for text-to-vector collection.
#[derive(Debug, Clone)]
pub struct TextVectorConfig {
    /// Model specification for embedding generation.
    pub model: ModelSpec,
    /// Collection name.
    pub collection_name: String,
    /// Distance function.
    pub distance: DistanceFunction,
    /// Whether to store original text in metadata.
    pub store_text: bool,
}

impl Default for TextVectorConfig {
    fn default() -> Self {
        Self {
            model: ModelSpec::new("built-in", 64),
            collection_name: "text_collection".into(),
            distance: DistanceFunction::Cosine,
            store_text: true,
        }
    }
}

impl TextVectorConfig {
    /// Use a specific model.
    #[must_use]
    pub fn with_model(mut self, model: ModelSpec) -> Self {
        self.model = model;
        self
    }

    /// Set collection name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.collection_name = name.into();
        self
    }
}

// ── Text Search Result ───────────────────────────────────────────────────────

/// Search result with original text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    /// Document ID.
    pub id: String,
    /// Similarity distance.
    pub distance: f32,
    /// Original text (if stored).
    pub text: Option<String>,
    /// Additional metadata.
    pub metadata: Option<Value>,
}

// ── Text Vector Collection ───────────────────────────────────────────────────

/// A collection that automatically embeds text using the built-in inference engine.
pub struct TextVectorCollection {
    config: TextVectorConfig,
    engine: InferenceEngine,
    collection: Collection,
    texts: HashMap<String, String>,
}

impl TextVectorCollection {
    /// Create a new text-to-vector collection.
    pub fn new(config: TextVectorConfig) -> Self {
        let dims = config.model.dimensions;
        let inference_config = InferenceConfig::builder()
            .model(config.model.clone())
            .normalize(true)
            .build();
        let engine = InferenceEngine::new(inference_config);
        let coll_config = CollectionConfig::new(&config.collection_name, dims)
            .with_distance(config.distance);
        let collection = Collection::new(coll_config);

        Self {
            config,
            engine,
            collection,
            texts: HashMap::new(),
        }
    }

    /// Insert text — auto-embeds and stores.
    pub fn insert_text(&mut self, id: &str, text: &str) -> Result<()> {
        let embedding = self.engine.embed_text(text)?;
        let mut metadata = serde_json::Map::new();
        if self.config.store_text {
            metadata.insert("_text".into(), Value::String(text.into()));
        }
        let meta = if metadata.is_empty() {
            None
        } else {
            Some(Value::Object(metadata))
        };
        self.collection.insert(id, &embedding, meta)?;
        if self.config.store_text {
            self.texts.insert(id.into(), text.into());
        }
        Ok(())
    }

    /// Insert text with additional metadata.
    pub fn insert_text_with_metadata(
        &mut self,
        id: &str,
        text: &str,
        metadata: Value,
    ) -> Result<()> {
        let embedding = self.engine.embed_text(text)?;
        let mut meta_map = match metadata {
            Value::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        if self.config.store_text {
            meta_map.insert("_text".into(), Value::String(text.into()));
        }
        self.collection
            .insert(id, &embedding, Some(Value::Object(meta_map)))?;
        if self.config.store_text {
            self.texts.insert(id.into(), text.into());
        }
        Ok(())
    }

    /// Search by text — auto-embeds the query.
    pub fn search_text(&mut self, query: &str, k: usize) -> Result<Vec<TextSearchResult>> {
        let query_embedding = self.engine.embed_text(query)?;
        let results = self.collection.search(&query_embedding, k)?;
        Ok(results
            .into_iter()
            .map(|r| TextSearchResult {
                id: r.id.clone(),
                distance: r.distance,
                text: self.texts.get(&r.id).cloned(),
                metadata: r.metadata,
            })
            .collect())
    }

    /// Search by raw embedding vector.
    pub fn search_vector(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.collection.search(query, k)
    }

    /// Get a document's text by ID.
    pub fn get_text(&self, id: &str) -> Option<&str> {
        self.texts.get(id).map(|s| s.as_str())
    }

    /// Delete a document.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        self.texts.remove(id);
        self.collection.delete(id)
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.collection.len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.collection.is_empty()
    }

    /// Get embedding dimensions.
    pub fn dimensions(&self) -> usize {
        self.collection.dimensions()
    }

    /// Get the underlying inference engine stats.
    pub fn inference_count(&self) -> u64 {
        self.engine.total_inferences()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search_text() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        col.insert_text("doc1", "rust programming language").unwrap();
        col.insert_text("doc2", "python data science").unwrap();

        let results = col.search_text("rust programming", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
        assert_eq!(results[0].text.as_deref(), Some("rust programming language"));
    }

    #[test]
    fn test_deterministic_embedding() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        col.insert_text("a", "hello world").unwrap();
        col.insert_text("b", "hello world").unwrap();

        // Same text should produce same embedding, so searching should find both
        let results = col.search_text("hello world", 5).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_delete() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        col.insert_text("d1", "test text").unwrap();
        assert_eq!(col.len(), 1);
        col.delete("d1").unwrap();
        assert_eq!(col.len(), 0);
        assert!(col.get_text("d1").is_none());
    }

    #[test]
    fn test_with_metadata() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        col.insert_text_with_metadata(
            "d1",
            "test",
            serde_json::json!({"category": "test"}),
        )
        .unwrap();

        let results = col.search_text("test", 1).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_empty_text_error() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        assert!(col.insert_text("bad", "").is_err());
    }

    #[test]
    fn test_inference_count() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        col.insert_text("a", "hello").unwrap();
        col.search_text("hello", 1).unwrap();
        assert_eq!(col.inference_count(), 2); // 1 insert + 1 search
    }
}
