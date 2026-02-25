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

// ── Chunking Strategies ─────────────────────────────────────────────────────

/// Text chunking strategy for splitting documents before embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// No chunking — embed the full text.
    None,
    /// Fixed-size sliding window with configurable overlap.
    SlidingWindow {
        /// Maximum characters per chunk.
        chunk_size: usize,
        /// Number of overlap characters between consecutive chunks.
        overlap: usize,
    },
    /// Split on sentence boundaries (period, question mark, exclamation).
    Sentence {
        /// Maximum sentences per chunk.
        max_sentences: usize,
    },
    /// Split on paragraph boundaries (double newline).
    Paragraph {
        /// Maximum paragraphs per chunk.
        max_paragraphs: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::None
    }
}

/// A text chunk with metadata about its position in the source document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Chunk text.
    pub text: String,
    /// Chunk index within the source document.
    pub chunk_index: usize,
    /// Total chunks from the source document.
    pub total_chunks: usize,
    /// Character offset in the source document.
    pub char_offset: usize,
}

/// Apply a chunking strategy to split text into chunks.
pub fn chunk_text(text: &str, strategy: &ChunkingStrategy) -> Vec<TextChunk> {
    match strategy {
        ChunkingStrategy::None => {
            vec![TextChunk {
                text: text.to_string(),
                chunk_index: 0,
                total_chunks: 1,
                char_offset: 0,
            }]
        }
        ChunkingStrategy::SlidingWindow { chunk_size, overlap } => {
            let mut chunks = Vec::new();
            let chars: Vec<char> = text.chars().collect();
            let step = chunk_size.saturating_sub(*overlap).max(1);
            let mut start = 0;
            while start < chars.len() {
                let end = (start + chunk_size).min(chars.len());
                let chunk_text: String = chars[start..end].iter().collect();
                chunks.push(TextChunk {
                    text: chunk_text,
                    chunk_index: chunks.len(),
                    total_chunks: 0, // filled in below
                    char_offset: start,
                });
                start += step;
                if end == chars.len() {
                    break;
                }
            }
            let total = chunks.len();
            for chunk in &mut chunks {
                chunk.total_chunks = total;
            }
            chunks
        }
        ChunkingStrategy::Sentence { max_sentences } => {
            let sentences: Vec<&str> = text.split_inclusive(&['.', '?', '!'][..])
                .filter(|s| !s.trim().is_empty())
                .collect();
            let mut chunks = Vec::new();
            let mut offset = 0;
            for group in sentences.chunks(*max_sentences) {
                let chunk_text = group.join("");
                let len = chunk_text.len();
                chunks.push(TextChunk {
                    text: chunk_text,
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                    char_offset: offset,
                });
                offset += len;
            }
            let total = chunks.len();
            for chunk in &mut chunks {
                chunk.total_chunks = total;
            }
            chunks
        }
        ChunkingStrategy::Paragraph { max_paragraphs } => {
            let paragraphs: Vec<&str> = text.split("\n\n")
                .filter(|s| !s.trim().is_empty())
                .collect();
            let mut chunks = Vec::new();
            let mut offset = 0;
            for group in paragraphs.chunks(*max_paragraphs) {
                let chunk_text = group.join("\n\n");
                let len = chunk_text.len();
                chunks.push(TextChunk {
                    text: chunk_text,
                    chunk_index: chunks.len(),
                    total_chunks: 0,
                    char_offset: offset,
                });
                offset += len + 2; // account for \n\n
            }
            let total = chunks.len();
            for chunk in &mut chunks {
                chunk.total_chunks = total;
            }
            chunks
        }
    }
}

// ── Multimodal Input Types ──────────────────────────────────────────────────

/// Input modality for the auto-embedding pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputModality {
    /// Plain text input.
    Text(String),
    /// Image input (raw bytes, typically PNG/JPEG).
    Image { data: Vec<u8>, mime_type: String },
    /// Audio input (raw bytes, typically WAV/MP3).
    Audio { data: Vec<u8>, mime_type: String },
    /// Pre-computed embedding (passthrough).
    Embedding(Vec<f32>),
}

// ── Matryoshka Reduction ────────────────────────────────────────────────────

/// Reduce embedding dimensions using Matryoshka truncation.
///
/// Matryoshka embeddings are trained so that the first N dimensions
/// form a valid (lower-quality) embedding. This function simply
/// truncates to the target dimensions.
pub fn matryoshka_reduce(embedding: &[f32], target_dims: usize) -> Vec<f32> {
    if target_dims >= embedding.len() {
        return embedding.to_vec();
    }
    let mut reduced = embedding[..target_dims].to_vec();
    // Re-normalize after truncation
    let norm: f32 = reduced.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in &mut reduced {
            *x /= norm;
        }
    }
    reduced
}

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
    /// Chunking strategy for splitting long documents.
    pub chunking: ChunkingStrategy,
    /// Optional Matryoshka target dimensions (None = use full dimensions).
    pub matryoshka_dims: Option<usize>,
}

impl Default for TextVectorConfig {
    fn default() -> Self {
        Self {
            model: ModelSpec::new("built-in", 64),
            collection_name: "text_collection".into(),
            distance: DistanceFunction::Cosine,
            store_text: true,
            chunking: ChunkingStrategy::default(),
            matryoshka_dims: None,
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

    /// Set chunking strategy.
    #[must_use]
    pub fn with_chunking(mut self, strategy: ChunkingStrategy) -> Self {
        self.chunking = strategy;
        self
    }

    /// Set Matryoshka target dimensions.
    #[must_use]
    pub fn with_matryoshka_dims(mut self, dims: usize) -> Self {
        self.matryoshka_dims = Some(dims);
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
        let dims = config.matryoshka_dims.unwrap_or(config.model.dimensions);
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
    ///
    /// If chunking is enabled, the text is split into chunks and each chunk
    /// is stored as a separate vector with chunk metadata.
    pub fn insert_text(&mut self, id: &str, text: &str) -> Result<()> {
        let chunks = chunk_text(text, &self.config.chunking);

        if chunks.len() == 1 {
            // Single chunk — use the original ID
            let mut embedding = self.engine.embed_text(text)?;
            if let Some(target) = self.config.matryoshka_dims {
                embedding = matryoshka_reduce(&embedding, target);
            }
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
        } else {
            // Multiple chunks — store each with a chunk suffix
            for chunk in &chunks {
                let chunk_id = format!("{id}#chunk{}", chunk.chunk_index);
                let mut embedding = self.engine.embed_text(&chunk.text)?;
                if let Some(target) = self.config.matryoshka_dims {
                    embedding = matryoshka_reduce(&embedding, target);
                }
                let mut metadata = serde_json::Map::new();
                if self.config.store_text {
                    metadata.insert("_text".into(), Value::String(chunk.text.clone()));
                }
                metadata.insert("_parent_id".into(), Value::String(id.into()));
                metadata.insert("_chunk_index".into(), serde_json::json!(chunk.chunk_index));
                metadata.insert("_total_chunks".into(), serde_json::json!(chunk.total_chunks));
                self.collection
                    .insert(&chunk_id, &embedding, Some(Value::Object(metadata)))?;
                if self.config.store_text {
                    self.texts.insert(chunk_id, chunk.text.clone());
                }
            }
            if self.config.store_text {
                self.texts.insert(id.into(), text.into());
            }
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
        let mut query_embedding = self.engine.embed_text(query)?;
        if let Some(target) = self.config.matryoshka_dims {
            query_embedding = matryoshka_reduce(&query_embedding, target);
        }
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

    /// Re-embed all documents using a new model.
    ///
    /// This is useful when the embedding model is updated and all vectors
    /// need to be regenerated to maintain consistency.
    pub fn re_embed_with_model(&mut self, new_model: ModelSpec) -> Result<ReEmbedResult> {
        let new_config = InferenceConfig::builder()
            .model(new_model.clone())
            .normalize(true)
            .build();
        let mut new_engine = InferenceEngine::new(new_config);

        let texts_snapshot: Vec<(String, String)> = self.texts.iter()
            .filter(|(id, _)| !id.contains("#chunk"))
            .map(|(id, text)| (id.clone(), text.clone()))
            .collect();

        let total = texts_snapshot.len();
        let mut re_embedded = 0usize;
        let mut failed = 0usize;

        for (id, text) in &texts_snapshot {
            match new_engine.embed_text(text) {
                Ok(mut embedding) => {
                    if let Some(target) = self.config.matryoshka_dims {
                        embedding = matryoshka_reduce(&embedding, target);
                    }
                    let meta = serde_json::json!({"_text": text, "_re_embedded": true});
                    // Delete old and insert new
                    let _ = self.collection.delete(id);
                    if self.collection.insert(id, &embedding, Some(meta)).is_ok() {
                        re_embedded += 1;
                    } else {
                        failed += 1;
                    }
                }
                Err(_) => {
                    failed += 1;
                }
            }
        }

        // Update engine to new model
        self.engine = new_engine;
        self.config.model = new_model;

        Ok(ReEmbedResult { total, re_embedded, failed })
    }
}

/// Result of a re-embedding operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReEmbedResult {
    /// Total documents considered.
    pub total: usize,
    /// Successfully re-embedded.
    pub re_embedded: usize,
    /// Failed to re-embed.
    pub failed: usize,
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

    #[test]
    fn test_sliding_window_chunking() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunk_text(text, &ChunkingStrategy::SlidingWindow {
            chunk_size: 10,
            overlap: 3,
        });
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].char_offset, 0);
        for chunk in &chunks {
            assert!(chunk.text.len() <= 10);
            assert_eq!(chunk.total_chunks, chunks.len());
        }
    }

    #[test]
    fn test_sentence_chunking() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = chunk_text(text, &ChunkingStrategy::Sentence { max_sentences: 2 });
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_paragraph_chunking() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = chunk_text(text, &ChunkingStrategy::Paragraph { max_paragraphs: 1 });
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Para one.");
    }

    #[test]
    fn test_insert_with_chunking() {
        let config = TextVectorConfig::default()
            .with_chunking(ChunkingStrategy::SlidingWindow {
                chunk_size: 15,
                overlap: 5,
            });
        let mut col = TextVectorCollection::new(config);
        col.insert_text("doc1", "This is a longer text that should be chunked into multiple pieces for embedding").unwrap();

        // Should have multiple vectors (chunks)
        assert!(col.len() > 1);
    }

    #[test]
    fn test_matryoshka_reduce() {
        let embedding = vec![0.5, 0.3, 0.1, 0.8, 0.2, 0.4];
        let reduced = matryoshka_reduce(&embedding, 3);
        assert_eq!(reduced.len(), 3);
        // Should be normalized
        let norm: f32 = reduced.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_matryoshka_no_reduction() {
        let embedding = vec![1.0, 0.0, 0.0];
        let reduced = matryoshka_reduce(&embedding, 5);
        assert_eq!(reduced.len(), 3); // no reduction needed
    }

    #[test]
    fn test_re_embed_with_model() {
        let mut col = TextVectorCollection::new(TextVectorConfig::default());
        col.insert_text("doc1", "hello world").unwrap();
        col.insert_text("doc2", "goodbye world").unwrap();

        let new_model = ModelSpec::new("new-model", 64);
        let result = col.re_embed_with_model(new_model).unwrap();
        assert_eq!(result.total, 2);
        assert_eq!(result.re_embedded, 2);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_input_modality_types() {
        let text = InputModality::Text("hello".into());
        let emb = InputModality::Embedding(vec![1.0, 2.0]);
        assert!(matches!(text, InputModality::Text(_)));
        assert!(matches!(emb, InputModality::Embedding(_)));
    }
}
