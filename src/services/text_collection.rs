//! Text Collection Service
//!
//! High-level "text-in, text-out" API that combines a Database collection with
//! the local inference engine. Users insert raw text and search with natural
//! language queries — embedding generation happens transparently.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::{Database};
//! use needle::text_collection::{TextCollection, TextCollectionConfig};
//!
//! let db = Database::in_memory();
//!
//! let config = TextCollectionConfig::builder()
//!     .name("docs")
//!     .model("mock-384")
//!     .store_text(true)
//!     .build();
//!
//! let tc = TextCollection::create(&db, config).unwrap();
//!
//! tc.insert_text("doc1", "Rust is a systems programming language", None).unwrap();
//! tc.insert_text("doc2", "Python is great for data science", None).unwrap();
//!
//! let results = tc.search_text("systems language", 5).unwrap();
//! assert!(!results.is_empty());
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collection::SearchResult;
use crate::database::Database;
use crate::error::Result;
use crate::local_inference::{InferenceConfig, InferenceEngine};

/// Configuration for creating a TextCollection.
#[derive(Debug, Clone)]
pub struct TextCollectionConfig {
    /// Collection name.
    pub name: String,
    /// Model ID for embeddings (from builtin registry).
    pub model_id: String,
    /// Whether to store original text in vector metadata.
    pub store_text: bool,
    /// Path for model caching.
    pub cache_dir: String,
}

impl Default for TextCollectionConfig {
    fn default() -> Self {
        Self {
            name: "documents".into(),
            model_id: "mock-384".into(),
            cache_dir: ".needle/models".into(),
            store_text: true,
        }
    }
}

pub struct TextCollectionConfigBuilder {
    config: TextCollectionConfig,
}

impl TextCollectionConfig {
    pub fn builder() -> TextCollectionConfigBuilder {
        TextCollectionConfigBuilder {
            config: Self::default(),
        }
    }
}

impl TextCollectionConfigBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

    pub fn store_text(mut self, store: bool) -> Self {
        self.config.store_text = store;
        self
    }

    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.cache_dir = dir.into();
        self
    }

    pub fn build(self) -> TextCollectionConfig {
        self.config
    }
}

/// Result of a text-based search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    pub id: String,
    pub text: Option<String>,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<Value>,
}

impl TextSearchResult {
    fn from_search_result(sr: &SearchResult, store_text: bool) -> Self {
        let text = if store_text {
            sr.metadata
                .as_ref()
                .and_then(|m| m.get("_text"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        Self {
            id: sr.id.clone(),
            text,
            distance: sr.distance,
            score: 1.0 / (1.0 + sr.distance),
            metadata: sr.metadata.clone(),
        }
    }
}

/// A text-first collection that handles embedding generation transparently.
pub struct TextCollection<'a> {
    db: &'a Database,
    engine: InferenceEngine,
    collection_name: String,
    store_text: bool,
}

impl<'a> TextCollection<'a> {
    /// Create a new TextCollection, creating the underlying vector collection if needed.
    pub fn create(db: &'a Database, config: TextCollectionConfig) -> Result<Self> {
        let inference_config = InferenceConfig {
            model_id: config.model_id.clone(),
            cache_dir: config.cache_dir.into(),
            ..InferenceConfig::default()
        };

        let engine = InferenceEngine::new(inference_config)?;
        let dim = engine.dimension();

        // Create collection if it doesn't exist
        if db.collection(&config.name).is_err() {
            db.create_collection(&config.name, dim)?;
        }

        Ok(Self {
            db,
            engine,
            collection_name: config.name,
            store_text: config.store_text,
        })
    }

    /// Open an existing TextCollection.
    pub fn open(db: &'a Database, collection_name: &str, model_id: &str) -> Result<Self> {
        let _ = db.collection(collection_name)?;

        let inference_config = InferenceConfig {
            model_id: model_id.into(),
            ..InferenceConfig::default()
        };
        let engine = InferenceEngine::new(inference_config)?;

        Ok(Self {
            db,
            engine,
            collection_name: collection_name.to_string(),
            store_text: true,
        })
    }

    /// Insert a text document. The text is embedded automatically.
    pub fn insert_text(
        &self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let embedding = self.engine.embed_text(text)?;
        let mut meta = metadata.unwrap_or(json!({}));

        if self.store_text {
            if let Value::Object(ref mut map) = meta {
                map.insert("_text".into(), Value::String(text.to_string()));
            }
        }

        let coll = self.db.collection(&self.collection_name)?;
        coll.insert(id, &embedding, Some(meta))?;
        Ok(())
    }

    /// Insert multiple text documents in a batch.
    pub fn insert_texts(
        &self,
        documents: &[(String, String, Option<Value>)],
    ) -> Result<usize> {
        let texts: Vec<&str> = documents.iter().map(|(_, t, _)| t.as_str()).collect();
        let embeddings = self.engine.embed_batch(&texts)?;
        let coll = self.db.collection(&self.collection_name)?;

        let mut count = 0;
        for (i, (id, text, meta)) in documents.iter().enumerate() {
            let mut m = meta.clone().unwrap_or(json!({}));
            if self.store_text {
                if let Value::Object(ref mut map) = m {
                    map.insert("_text".into(), Value::String(text.clone()));
                }
            }
            coll.insert(id, &embeddings[i], Some(m))?;
            count += 1;
        }
        Ok(count)
    }

    /// Search with a natural language query. The query is embedded automatically.
    pub fn search_text(&self, query: &str, k: usize) -> Result<Vec<TextSearchResult>> {
        let query_embedding = self.engine.embed_text(query)?;
        let coll = self.db.collection(&self.collection_name)?;
        let results = coll.search(&query_embedding, k)?;

        Ok(results
            .iter()
            .map(|sr| TextSearchResult::from_search_result(sr, self.store_text))
            .collect())
    }

    /// Get the embedding dimension used by this collection.
    pub fn dimension(&self) -> usize {
        self.engine.dimension()
    }

    /// Get the model ID being used.
    pub fn model_id(&self) -> &str {
        &self.engine.model_spec().id
    }

    /// Get inference statistics.
    pub fn inference_stats(&self) -> crate::local_inference::InferenceStats {
        self.engine.stats()
    }

    /// Get available models.
    pub fn available_models() -> Vec<crate::local_inference::ModelSpec> {
        InferenceEngine::available_models()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_collection_create_and_insert() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("docs")
            .model("mock-384")
            .store_text(true)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "Hello world", None).unwrap();
        tc.insert_text("doc2", "Goodbye world", None).unwrap();

        let results = tc.search_text("Hello", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1".to_string().as_str());
    }

    #[test]
    fn test_text_collection_batch_insert() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("batch")
            .model("mock-384")
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        let docs = vec![
            ("d1".into(), "First document".into(), None),
            ("d2".into(), "Second document".into(), None),
            ("d3".into(), "Third document".into(), None),
        ];

        let count = tc.insert_texts(&docs).unwrap();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_text_collection_stores_text_in_metadata() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("meta")
            .model("mock-384")
            .store_text(true)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "Hello embedded world", None)
            .unwrap();

        let results = tc.search_text("Hello", 1).unwrap();
        assert_eq!(results.len(), 1);
        // Text should be preserved in search results
        assert!(results[0].text.is_some());
    }

    #[test]
    fn test_text_collection_dimension() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("dim")
            .model("mock-384")
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        assert_eq!(tc.dimension(), 384);
    }

    #[test]
    fn test_available_models() {
        let models = TextCollection::available_models();
        assert!(!models.is_empty());
    }
}
