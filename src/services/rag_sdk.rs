//! One-Line RAG SDK
//!
//! End-to-end Retrieval-Augmented Generation pipeline: chunk text → embed →
//! index → search → format context for LLM. Zero configuration required.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::rag_sdk::{RagPipeline, RagConfig, RagAnswer};
//!
//! let mut rag = RagPipeline::new(RagConfig::default());
//! rag.add("doc1", "Rust is a systems programming language focused on safety.").unwrap();
//! rag.add("doc2", "Python is great for data science and ML.").unwrap();
//!
//! let answer = rag.ask("What language is good for systems programming?", 3).unwrap();
//! println!("Context:\n{}", answer.context);
//! println!("Sources: {:?}", answer.sources);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::services::inference_engine::{InferenceConfig, InferenceEngine, ModelSpec};

/// RAG pipeline configuration.
#[derive(Debug, Clone)]
pub struct RagConfig {
    pub dimensions: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub top_k: usize,
    pub context_template: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            dimensions: 64, chunk_size: 200, chunk_overlap: 50, top_k: 3,
            context_template: "Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}".into(),
        }
    }
}

/// RAG answer with context and sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagAnswer {
    pub context: String,
    pub sources: Vec<RagSource>,
    pub prompt: String,
    pub chunks_searched: usize,
}

/// A source chunk used in the answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagSource {
    pub doc_id: String,
    pub chunk_id: String,
    pub text: String,
    pub distance: f32,
}

/// Document metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocMeta { doc_id: String, chunk_idx: usize, text: String }

/// Zero-config RAG pipeline.
pub struct RagPipeline {
    config: RagConfig,
    collection: Collection,
    engine: InferenceEngine,
    docs: HashMap<String, Vec<String>>, // doc_id → chunk texts
}

impl RagPipeline {
    /// Create a new RAG pipeline.
    pub fn new(config: RagConfig) -> Self {
        let collection = Collection::new(CollectionConfig::new("__rag__", config.dimensions).with_distance(DistanceFunction::Cosine));
        let engine = InferenceEngine::new(InferenceConfig::builder().model(ModelSpec::new("rag", config.dimensions)).normalize(true).build());
        Self { config, collection, engine, docs: HashMap::new() }
    }

    /// Add a document — auto-chunks and embeds.
    pub fn add(&mut self, doc_id: &str, text: &str) -> Result<usize> {
        let chunks = self.chunk(text);
        let count = chunks.len();
        for (i, chunk) in chunks.iter().enumerate() {
            let emb = self.engine.embed_text(chunk)?;
            let chunk_id = format!("{doc_id}__chunk_{i}");
            let meta = serde_json::json!({"doc_id": doc_id, "chunk_idx": i, "_text": chunk});
            self.collection.insert(&chunk_id, &emb, Some(meta))?;
        }
        self.docs.insert(doc_id.into(), chunks);
        Ok(count)
    }

    /// Ask a question — retrieves context and formats prompt.
    pub fn ask(&mut self, question: &str, top_k: usize) -> Result<RagAnswer> {
        let query_emb = self.engine.embed_text(question)?;
        let results = self.collection.search(&query_emb, top_k)?;

        let mut sources = Vec::new();
        let mut context_parts = Vec::new();
        for (i, r) in results.iter().enumerate() {
            let text = r.metadata.as_ref()
                .and_then(|m| m.get("_text")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let doc_id = r.metadata.as_ref()
                .and_then(|m| m.get("doc_id")).and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            context_parts.push(format!("[{}] {}", i + 1, text));
            sources.push(RagSource { doc_id, chunk_id: r.id.clone(), text, distance: r.distance });
        }

        let context = context_parts.join("\n\n");
        let prompt = self.config.context_template
            .replace("{context}", &context).replace("{question}", question);

        Ok(RagAnswer { context, sources, prompt, chunks_searched: results.len() })
    }

    /// Search without formatting (raw results).
    pub fn search(&mut self, query: &str, k: usize) -> Result<Vec<RagSource>> {
        let emb = self.engine.embed_text(query)?;
        let results = self.collection.search(&emb, k)?;
        Ok(results.into_iter().map(|r| {
            let text = r.metadata.as_ref().and_then(|m| m.get("_text")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let doc_id = r.metadata.as_ref().and_then(|m| m.get("doc_id")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            RagSource { doc_id, chunk_id: r.id, text, distance: r.distance }
        }).collect())
    }

    /// Number of documents.
    pub fn doc_count(&self) -> usize { self.docs.len() }
    /// Total chunks indexed.
    pub fn chunk_count(&self) -> usize { self.collection.len() }

    fn chunk(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() <= self.config.chunk_size { return vec![text.to_string()]; }
        let step = self.config.chunk_size.saturating_sub(self.config.chunk_overlap).max(1);
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < words.len() {
            let end = (i + self.config.chunk_size).min(words.len());
            chunks.push(words[i..end].join(" "));
            i += step;
            if end == words.len() { break; }
        }
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_ask() {
        let mut rag = RagPipeline::new(RagConfig::default());
        rag.add("d1", "Rust is a systems programming language").unwrap();
        rag.add("d2", "Python is for data science").unwrap();
        let answer = rag.ask("systems programming", 3).unwrap();
        assert!(!answer.context.is_empty());
        assert!(!answer.sources.is_empty());
        assert!(answer.prompt.contains("systems programming"));
    }

    #[test]
    fn test_chunking() {
        let mut rag = RagPipeline::new(RagConfig { chunk_size: 5, chunk_overlap: 2, ..Default::default() });
        let long = (0..20).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let chunks = rag.add("d1", &long).unwrap();
        assert!(chunks > 1);
    }

    #[test]
    fn test_search() {
        let mut rag = RagPipeline::new(RagConfig::default());
        rag.add("d1", "hello world").unwrap();
        let results = rag.search("hello", 5).unwrap();
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn test_counts() {
        let mut rag = RagPipeline::new(RagConfig::default());
        rag.add("d1", "short text").unwrap();
        assert_eq!(rag.doc_count(), 1);
        assert!(rag.chunk_count() >= 1);
    }

    #[test]
    fn test_empty_question() {
        let mut rag = RagPipeline::new(RagConfig::default());
        rag.add("d1", "some text").unwrap();
        // Empty question should still work (empty embedding)
        assert!(rag.ask("", 1).is_err()); // empty text can't be embedded
    }
}
