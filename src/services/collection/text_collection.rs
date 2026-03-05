#![allow(clippy::unwrap_used)]
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
use crate::metadata::Filter;

/// Strategy for splitting text into chunks before embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// No chunking — embed the full document as one vector.
    None,
    /// Split on sentence boundaries (period + space).
    Sentence,
    /// Split into fixed-size windows of approximately `n` characters with overlap.
    FixedSize { chars: usize, overlap: usize },
    /// Split on paragraph boundaries (double newline).
    Paragraph,
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::None
    }
}

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
    /// Chunking strategy for long documents.
    pub chunking: ChunkingStrategy,
}

impl Default for TextCollectionConfig {
    fn default() -> Self {
        Self {
            name: "documents".into(),
            model_id: "mock-384".into(),
            cache_dir: ".needle/models".into(),
            store_text: true,
            chunking: ChunkingStrategy::None,
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
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    #[must_use]
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.config.model_id = model_id.into();
        self
    }

    #[must_use]
    pub fn store_text(mut self, store: bool) -> Self {
        self.config.store_text = store;
        self
    }

    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.cache_dir = dir.into();
        self
    }

    #[must_use]
    pub fn chunking(mut self, strategy: ChunkingStrategy) -> Self {
        self.config.chunking = strategy;
        self
    }

    pub fn build(self) -> TextCollectionConfig {
        self.config
    }
}

/// Result of a text-based search, including passage citation info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    pub id: String,
    pub text: Option<String>,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<Value>,
    /// Source document ID (for chunks, this is the parent doc).
    pub source_doc: Option<String>,
    /// Chunk index within the source document (0 if not chunked).
    pub chunk_index: Option<usize>,
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
        let source_doc = sr
            .metadata
            .as_ref()
            .and_then(|m| m.get("_source_doc"))
            .and_then(|v| v.as_str())
            .map(String::from);
        let chunk_index = sr
            .metadata
            .as_ref()
            .and_then(|m| m.get("_chunk_index"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        Self {
            id: sr.id.clone(),
            text,
            distance: sr.distance,
            score: 1.0 / (1.0 + sr.distance),
            metadata: sr.metadata.clone(),
            source_doc,
            chunk_index,
        }
    }
}

/// A text-first collection that handles embedding generation transparently.
pub struct TextCollection<'a> {
    db: &'a Database,
    engine: InferenceEngine,
    collection_name: String,
    store_text: bool,
    chunking: ChunkingStrategy,
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
            chunking: config.chunking,
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
            chunking: ChunkingStrategy::None,
        })
    }

    /// Insert a text document. The text is embedded automatically.
    /// If a chunking strategy is configured, the text is split and each chunk
    /// is inserted as a separate vector with `_source_doc` / `_chunk_index` metadata.
    pub fn insert_text(
        &self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let chunks = self.chunk_text(text);

        if chunks.len() <= 1 {
            let embedding = self.engine.embed_text(text)?;
            let mut meta = metadata.unwrap_or(json!({}));
            if self.store_text {
                if let Value::Object(ref mut map) = meta {
                    map.insert("_text".into(), Value::String(text.to_string()));
                }
            }
            let coll = self.db.collection(&self.collection_name)?;
            coll.insert(&id, &embedding, Some(meta))?;
        } else {
            let coll = self.db.collection(&self.collection_name)?;
            for (i, chunk) in chunks.iter().enumerate() {
                let chunk_id = format!("{id}__chunk_{i}");
                let embedding = self.engine.embed_text(chunk)?;
                let mut meta = metadata.clone().unwrap_or(json!({}));
                if let Value::Object(ref mut map) = meta {
                    map.insert("_source_doc".into(), Value::String(id.clone()));
                    map.insert("_chunk_index".into(), json!(i));
                    if self.store_text {
                        map.insert("_text".into(), Value::String(chunk.clone()));
                    }
                }
                coll.insert(&chunk_id, &embedding, Some(meta))?;
            }
        }
        Ok(())
    }

    /// Insert multiple text documents in a batch.
    pub fn insert_texts(&self, documents: &[(String, String, Option<Value>)]) -> Result<usize> {
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

    /// Search with a natural language query and a metadata filter.
    pub fn search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<TextSearchResult>> {
        let query_embedding = self.engine.embed_text(query)?;
        let coll = self.db.collection(&self.collection_name)?;
        let results = coll.search_with_filter(&query_embedding, k, filter)?;

        Ok(results
            .iter()
            .map(|sr| TextSearchResult::from_search_result(sr, self.store_text))
            .collect())
    }

    /// Ask a question and get ranked passages with citations.
    /// This is the highest-level API: embed query → search → return passages
    /// with source document references.
    pub fn ask(&self, question: &str, k: usize) -> Result<Vec<TextSearchResult>> {
        self.search_text(question, k)
    }

    /// Delete a document (and all its chunks if chunked).
    pub fn delete(&self, id: &str) -> Result<bool> {
        let coll = self.db.collection(&self.collection_name)?;
        // Try deleting the exact ID first
        let direct = coll.delete(id)?;
        // Also delete any chunks (id__chunk_0, id__chunk_1, ...)
        let mut i = 0;
        loop {
            let chunk_id = format!("{id}__chunk_{i}");
            match coll.delete(&chunk_id) {
                Ok(true) => i += 1,
                _ => break,
            }
        }
        Ok(direct || i > 0)
    }

    /// Count documents in the collection.
    pub fn count(&self) -> Result<usize> {
        let coll = self.db.collection(&self.collection_name)?;
        Ok(coll.len())
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

    fn chunk_text(&self, text: &str) -> Vec<String> {
        match self.chunking {
            ChunkingStrategy::None => vec![text.to_string()],
            ChunkingStrategy::Sentence => {
                let sentences: Vec<String> = text
                    .split(". ")
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if sentences.is_empty() {
                    vec![text.to_string()]
                } else {
                    sentences
                }
            }
            ChunkingStrategy::Paragraph => {
                let paragraphs: Vec<String> = text
                    .split("\n\n")
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if paragraphs.is_empty() {
                    vec![text.to_string()]
                } else {
                    paragraphs
                }
            }
            ChunkingStrategy::FixedSize { chars, overlap } => {
                let chars = chars.max(1);
                let overlap = overlap.min(chars.saturating_sub(1));
                let bytes = text.as_bytes();
                let mut chunks = Vec::new();
                let mut start = 0;
                while start < bytes.len() {
                    let end = (start + chars).min(bytes.len());
                    // Avoid splitting in the middle of a multi-byte char
                    let end = (start..=end)
                        .rev()
                        .find(|&i| text.is_char_boundary(i))
                        .unwrap_or(end);
                    let chunk = &text[start..end];
                    if !chunk.trim().is_empty() {
                        chunks.push(chunk.to_string());
                    }
                    start = if end > overlap {
                        end - overlap
                    } else {
                        end
                    };
                    if start >= bytes.len() || end == bytes.len() {
                        break;
                    }
                }
                if chunks.is_empty() {
                    vec![text.to_string()]
                } else {
                    chunks
                }
            }
        }
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

    #[test]
    fn test_ask_returns_passages() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("ask_test")
            .model("mock-384")
            .store_text(true)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "Rust is a systems language", None)
            .unwrap();
        tc.insert_text("doc2", "Python is for data science", None)
            .unwrap();

        let results = tc.ask("systems language", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].text.is_some());
    }

    #[test]
    fn test_search_with_filter() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("filter_test")
            .model("mock-384")
            .store_text(true)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text(
            "doc1",
            "Rust is fast",
            Some(serde_json::json!({"lang": "rust"})),
        )
        .unwrap();
        tc.insert_text(
            "doc2",
            "Python is slow",
            Some(serde_json::json!({"lang": "python"})),
        )
        .unwrap();

        let filter = Filter::eq("lang", "rust");
        let results = tc.search_with_filter("fast language", 5, &filter).unwrap();
        for r in &results {
            let lang = r
                .metadata
                .as_ref()
                .and_then(|m| m.get("lang"))
                .and_then(|v| v.as_str());
            assert_eq!(lang, Some("rust"));
        }
    }

    #[test]
    fn test_delete() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("del_test")
            .model("mock-384")
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "Hello world", None).unwrap();
        assert_eq!(tc.count().unwrap(), 1);

        assert!(tc.delete("doc1").unwrap());
        assert_eq!(tc.count().unwrap(), 0);
    }

    #[test]
    fn test_sentence_chunking() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("chunk_test")
            .model("mock-384")
            .store_text(true)
            .chunking(ChunkingStrategy::Sentence)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "First sentence. Second sentence. Third sentence", None)
            .unwrap();

        // Should create 3 chunks
        assert_eq!(tc.count().unwrap(), 3);

        let results = tc.search_text("First", 5).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].source_doc.as_deref() == Some("doc1"));
    }

    #[test]
    fn test_paragraph_chunking() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("para_test")
            .model("mock-384")
            .chunking(ChunkingStrategy::Paragraph)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "Para one.\n\nPara two.\n\nPara three.", None)
            .unwrap();

        assert_eq!(tc.count().unwrap(), 3);
    }

    #[test]
    fn test_delete_chunked_document() {
        let db = Database::in_memory();
        let config = TextCollectionConfig::builder()
            .name("del_chunk")
            .model("mock-384")
            .chunking(ChunkingStrategy::Sentence)
            .build();

        let tc = TextCollection::create(&db, config).unwrap();
        tc.insert_text("doc1", "First. Second. Third", None)
            .unwrap();
        assert_eq!(tc.count().unwrap(), 3);

        assert!(tc.delete("doc1").unwrap());
        assert_eq!(tc.count().unwrap(), 0);
    }
}
