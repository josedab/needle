//! RAG (Retrieval-Augmented Generation) Pipeline
//!
//! Complete pipeline for building RAG applications:
//! - Document chunking (semantic, sliding window, hierarchical)
//! - Multi-stage retrieval (dense + sparse + rerank)
//! - Context assembly and prompt building
//! - Citation tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::rag::{RagPipeline, ChunkingStrategy, RagConfig};
//!
//! let pipeline = RagPipeline::new(db, RagConfig::default());
//!
//! // Ingest documents with automatic chunking
//! pipeline.ingest_document("doc1", &text, ChunkingStrategy::Semantic)?;
//!
//! // Query with full RAG pipeline
//! let response = pipeline.query("What is machine learning?", 5)?;
//! println!("Context: {:?}", response.chunks);
//! println!("Citations: {:?}", response.citations);
//! ```

use crate::database::Database;
use crate::error::Result;
use crate::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Chunking strategy for documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks with overlap
    FixedSize {
        chunk_size: usize,
        overlap: usize,
    },
    /// Sliding window with token count
    SlidingWindow {
        window_size: usize,
        step_size: usize,
    },
    /// Semantic chunking based on sentence boundaries
    Semantic {
        max_chunk_size: usize,
        min_chunk_size: usize,
    },
    /// Hierarchical chunking (parent-child relationships)
    Hierarchical {
        levels: Vec<usize>, // chunk sizes for each level
    },
    /// Paragraph-based chunking
    Paragraph {
        max_paragraphs: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Semantic {
            max_chunk_size: 512,
            min_chunk_size: 100,
        }
    }
}

/// Configuration for RAG pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Collection name for storing chunks
    pub collection_name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Default chunking strategy
    pub chunking: ChunkingStrategy,
    /// Number of chunks to retrieve
    pub top_k: usize,
    /// Enable reranking
    pub rerank: bool,
    /// Reranking top-k (rerank this many, return top_k)
    pub rerank_top_k: usize,
    /// Enable hybrid search (dense + sparse)
    pub hybrid_search: bool,
    /// Hybrid search alpha (0 = sparse only, 1 = dense only)
    pub hybrid_alpha: f32,
    /// Include parent chunks in context
    pub include_parents: bool,
    /// Deduplicate similar chunks
    pub deduplicate: bool,
    /// Deduplication threshold
    pub dedup_threshold: f32,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            collection_name: "rag_chunks".to_string(),
            dimensions: 384,
            chunking: ChunkingStrategy::default(),
            top_k: 5,
            rerank: true,
            rerank_top_k: 20,
            hybrid_search: true,
            hybrid_alpha: 0.7,
            include_parents: false,
            deduplicate: true,
            dedup_threshold: 0.95,
        }
    }
}

/// A chunk of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk ID
    pub id: String,
    /// Source document ID
    pub document_id: String,
    /// Chunk text content
    pub text: String,
    /// Start position in original document
    pub start_pos: usize,
    /// End position in original document
    pub end_pos: usize,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Total chunks in document
    pub total_chunks: usize,
    /// Parent chunk ID (for hierarchical)
    pub parent_id: Option<String>,
    /// Child chunk IDs (for hierarchical)
    pub children: Vec<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// Retrieved chunk with score
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    /// The chunk
    pub chunk: Chunk,
    /// Retrieval score (distance)
    pub score: f32,
    /// Rerank score (if reranking enabled)
    pub rerank_score: Option<f32>,
    /// Combined score
    pub final_score: f32,
}

/// Citation for a retrieved chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Document ID
    pub document_id: String,
    /// Chunk ID
    pub chunk_id: String,
    /// Text snippet
    pub snippet: String,
    /// Position in document
    pub position: (usize, usize),
    /// Relevance score
    pub score: f32,
}

/// RAG query response
#[derive(Debug, Clone)]
pub struct RagResponse {
    /// Retrieved chunks
    pub chunks: Vec<RetrievedChunk>,
    /// Assembled context
    pub context: String,
    /// Citations
    pub citations: Vec<Citation>,
    /// Query metadata
    pub metadata: RagQueryMetadata,
}

/// Query metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQueryMetadata {
    /// Total chunks retrieved
    pub chunks_retrieved: usize,
    /// Chunks after deduplication
    pub chunks_after_dedup: usize,
    /// Retrieval latency in ms
    pub retrieval_latency_ms: u64,
    /// Rerank latency in ms
    pub rerank_latency_ms: Option<u64>,
    /// Total latency in ms
    pub total_latency_ms: u64,
}

/// Document with chunking info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID
    pub id: String,
    /// Full text
    pub text: String,
    /// Chunk IDs
    pub chunk_ids: Vec<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Ingestion timestamp
    pub ingested_at: u64,
}

/// RAG Pipeline
pub struct RagPipeline {
    db: Arc<Database>,
    config: RagConfig,
    documents: HashMap<String, Document>,
    chunks: HashMap<String, Chunk>,
}

impl RagPipeline {
    /// Create a new RAG pipeline
    pub fn new(db: Arc<Database>, config: RagConfig) -> Result<Self> {
        // Create collection if it doesn't exist
        if !db.has_collection(&config.collection_name) {
            db.create_collection(&config.collection_name, config.dimensions)?;
        }

        Ok(Self {
            db,
            config,
            documents: HashMap::new(),
            chunks: HashMap::new(),
        })
    }

    /// Ingest a document with automatic chunking
    pub fn ingest_document(
        &mut self,
        doc_id: &str,
        text: &str,
        metadata: Option<serde_json::Value>,
        embedder: &dyn Embedder,
    ) -> Result<Document> {
        self.ingest_with_strategy(doc_id, text, metadata, &self.config.chunking.clone(), embedder)
    }

    /// Ingest with specific chunking strategy
    pub fn ingest_with_strategy(
        &mut self,
        doc_id: &str,
        text: &str,
        metadata: Option<serde_json::Value>,
        strategy: &ChunkingStrategy,
        embedder: &dyn Embedder,
    ) -> Result<Document> {
        // Chunk the document
        let chunk_texts = self.chunk_text(text, strategy);
        let total_chunks = chunk_texts.len();

        let collection = self.db.collection(&self.config.collection_name)?;
        let mut chunk_ids = Vec::new();

        for (i, (chunk_text, start, end)) in chunk_texts.into_iter().enumerate() {
            let chunk_id = format!("{}_{}", doc_id, i);

            let chunk = Chunk {
                id: chunk_id.clone(),
                document_id: doc_id.to_string(),
                text: chunk_text.clone(),
                start_pos: start,
                end_pos: end,
                chunk_index: i,
                total_chunks,
                parent_id: None,
                children: Vec::new(),
                metadata: metadata.clone(),
            };

            // Generate embedding
            let embedding = embedder.embed(&chunk_text)?;

            // Store chunk metadata
            let chunk_meta = serde_json::json!({
                "document_id": doc_id,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "start_pos": start,
                "end_pos": end,
                "text": chunk_text,
            });

            // Insert into collection
            collection.insert(&chunk_id, &embedding, Some(chunk_meta))?;

            self.chunks.insert(chunk_id.clone(), chunk);
            chunk_ids.push(chunk_id);
        }

        let document = Document {
            id: doc_id.to_string(),
            text: text.to_string(),
            chunk_ids: chunk_ids.clone(),
            metadata,
            ingested_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock should be after Unix epoch")
                .as_secs(),
        };

        self.documents.insert(doc_id.to_string(), document.clone());

        Ok(document)
    }

    /// Query the RAG pipeline
    pub fn query(
        &self,
        query: &str,
        embedder: &dyn Embedder,
    ) -> Result<RagResponse> {
        self.query_with_filter(query, None, embedder)
    }

    /// Query with metadata filter
    pub fn query_with_filter(
        &self,
        query: &str,
        filter: Option<&crate::metadata::Filter>,
        embedder: &dyn Embedder,
    ) -> Result<RagResponse> {
        let start_time = std::time::Instant::now();

        // Generate query embedding
        let query_embedding = embedder.embed(query)?;

        let collection = self.db.collection(&self.config.collection_name)?;

        // Retrieve candidates
        let retrieval_start = std::time::Instant::now();
        let candidates = if let Some(f) = filter {
            collection.search_with_filter(&query_embedding, self.config.rerank_top_k, f)?
        } else {
            collection.search(&query_embedding, self.config.rerank_top_k)?
        };
        let retrieval_latency = retrieval_start.elapsed().as_millis() as u64;

        // Convert to retrieved chunks
        let mut retrieved: Vec<RetrievedChunk> = candidates
            .into_iter()
            .filter_map(|result| {
                let chunk = self.result_to_chunk(&result)?;
                Some(RetrievedChunk {
                    chunk,
                    score: result.distance,
                    rerank_score: None,
                    final_score: 1.0 - result.distance, // Convert distance to similarity
                })
            })
            .collect();

        // Rerank if enabled
        let rerank_latency = if self.config.rerank && !retrieved.is_empty() {
            let rerank_start = std::time::Instant::now();
            self.rerank_chunks(query, &mut retrieved);
            Some(rerank_start.elapsed().as_millis() as u64)
        } else {
            None
        };

        let chunks_retrieved = retrieved.len();

        // Deduplicate if enabled
        if self.config.deduplicate {
            retrieved = self.deduplicate_chunks(retrieved);
        }

        let chunks_after_dedup = retrieved.len();

        // Take top_k
        retrieved.truncate(self.config.top_k);

        // Assemble context
        let context = self.assemble_context(&retrieved);

        // Build citations
        let citations = self.build_citations(&retrieved);

        let total_latency = start_time.elapsed().as_millis() as u64;

        Ok(RagResponse {
            chunks: retrieved,
            context,
            citations,
            metadata: RagQueryMetadata {
                chunks_retrieved,
                chunks_after_dedup,
                retrieval_latency_ms: retrieval_latency,
                rerank_latency_ms: rerank_latency,
                total_latency_ms: total_latency,
            },
        })
    }

    /// Chunk text according to strategy
    fn chunk_text(&self, text: &str, strategy: &ChunkingStrategy) -> Vec<(String, usize, usize)> {
        match strategy {
            ChunkingStrategy::FixedSize { chunk_size, overlap } => {
                self.chunk_fixed_size(text, *chunk_size, *overlap)
            }
            ChunkingStrategy::SlidingWindow { window_size, step_size } => {
                self.chunk_sliding_window(text, *window_size, *step_size)
            }
            ChunkingStrategy::Semantic { max_chunk_size, min_chunk_size } => {
                self.chunk_semantic(text, *max_chunk_size, *min_chunk_size)
            }
            ChunkingStrategy::Hierarchical { levels } => {
                self.chunk_hierarchical(text, levels)
            }
            ChunkingStrategy::Paragraph { max_paragraphs } => {
                self.chunk_paragraphs(text, *max_paragraphs)
            }
        }
    }

    fn chunk_fixed_size(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<(String, usize, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            chunks.push((chunk, start, end));

            if end >= chars.len() {
                break;
            }

            start = if overlap < chunk_size {
                start + chunk_size - overlap
            } else {
                start + 1
            };
        }

        chunks
    }

    fn chunk_sliding_window(&self, text: &str, window_size: usize, step_size: usize) -> Vec<(String, usize, usize)> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut word_start = 0;

        while word_start < words.len() {
            let word_end = (word_start + window_size).min(words.len());
            let chunk = words[word_start..word_end].join(" ");

            // Calculate character positions (approximate)
            let char_start = words[..word_start].iter().map(|w| w.len() + 1).sum::<usize>();
            let char_end = char_start + chunk.len();

            chunks.push((chunk, char_start, char_end));

            if word_end >= words.len() {
                break;
            }

            word_start += step_size;
        }

        chunks
    }

    fn chunk_semantic(&self, text: &str, max_size: usize, min_size: usize) -> Vec<(String, usize, usize)> {
        let mut chunks = Vec::new();
        let sentences = self.split_sentences(text);

        let mut current_chunk = String::new();
        let mut chunk_start = 0;
        let mut current_start = 0;

        for sentence in sentences {
            let sentence_len = sentence.len();

            if current_chunk.len() + sentence_len > max_size && current_chunk.len() >= min_size {
                // Save current chunk
                let end = current_start;
                chunks.push((current_chunk.trim().to_string(), chunk_start, end));

                // Start new chunk
                current_chunk = sentence.to_string();
                chunk_start = current_start;
            } else {
                if !current_chunk.is_empty() {
                    current_chunk.push(' ');
                }
                current_chunk.push_str(&sentence);
            }

            current_start += sentence_len + 1; // +1 for space/newline
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push((current_chunk.trim().to_string(), chunk_start, text.len()));
        }

        chunks
    }

    fn chunk_hierarchical(&self, text: &str, levels: &[usize]) -> Vec<(String, usize, usize)> {
        // For simplicity, use the smallest level for now
        // A full implementation would create parent-child relationships
        let chunk_size = levels.last().copied().unwrap_or(512);
        self.chunk_fixed_size(text, chunk_size, chunk_size / 4)
    }

    fn chunk_paragraphs(&self, text: &str, max_paragraphs: usize) -> Vec<(String, usize, usize)> {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut pos = 0;

        for para_group in paragraphs.chunks(max_paragraphs) {
            let chunk = para_group.join("\n\n");
            let end = pos + chunk.len();
            chunks.push((chunk, pos, end));
            pos = end + 2; // +2 for \n\n
        }

        chunks
    }

    fn split_sentences(&self, text: &str) -> Vec<String> {
        // Simple sentence splitting (could be improved with NLP)
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);
            if (c == '.' || c == '!' || c == '?')
                && current.len() > 1 {
                    sentences.push(current.trim().to_string());
                    current = String::new();
                }
        }

        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        sentences
    }

    fn result_to_chunk(&self, result: &SearchResult) -> Option<Chunk> {
        let meta = result.metadata.as_ref()?;

        Some(Chunk {
            id: result.id.clone(),
            document_id: meta.get("document_id")?.as_str()?.to_string(),
            text: meta.get("text")?.as_str()?.to_string(),
            start_pos: meta.get("start_pos")?.as_u64()? as usize,
            end_pos: meta.get("end_pos")?.as_u64()? as usize,
            chunk_index: meta.get("chunk_index")?.as_u64()? as usize,
            total_chunks: meta.get("total_chunks")?.as_u64()? as usize,
            parent_id: None,
            children: Vec::new(),
            metadata: result.metadata.clone(),
        })
    }

    fn rerank_chunks(&self, query: &str, chunks: &mut [RetrievedChunk]) {
        // Simple reranking based on query term overlap
        // In production, use a cross-encoder model
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        for chunk in chunks.iter_mut() {
            let chunk_text = chunk.chunk.text.to_lowercase();
            let mut overlap_score = 0.0;

            for term in &query_terms {
                if chunk_text.contains(term) {
                    overlap_score += 1.0;
                }
            }

            // Normalize by query length
            let rerank_score = overlap_score / query_terms.len().max(1) as f32;
            chunk.rerank_score = Some(rerank_score);

            // Combine scores (weighted average)
            chunk.final_score = 0.7 * chunk.final_score + 0.3 * rerank_score;
        }

        // Sort by final score (descending)
        chunks.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn deduplicate_chunks(&self, chunks: Vec<RetrievedChunk>) -> Vec<RetrievedChunk> {
        let mut result = Vec::new();

        for chunk in chunks {
            let is_duplicate = result.iter().any(|existing: &RetrievedChunk| {
                self.text_similarity(&existing.chunk.text, &chunk.chunk.text) > self.config.dedup_threshold
            });

            if !is_duplicate {
                result.push(chunk);
            }
        }

        result
    }

    fn text_similarity(&self, a: &str, b: &str) -> f32 {
        // Jaccard similarity on words
        let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();

        let intersection = a_words.intersection(&b_words).count();
        let union = a_words.union(&b_words).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn assemble_context(&self, chunks: &[RetrievedChunk]) -> String {
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i + 1, c.chunk.text))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn build_citations(&self, chunks: &[RetrievedChunk]) -> Vec<Citation> {
        chunks
            .iter()
            .map(|c| Citation {
                document_id: c.chunk.document_id.clone(),
                chunk_id: c.chunk.id.clone(),
                snippet: if c.chunk.text.len() > 100 {
                    format!("{}...", &c.chunk.text[..100])
                } else {
                    c.chunk.text.clone()
                },
                position: (c.chunk.start_pos, c.chunk.end_pos),
                score: c.final_score,
            })
            .collect()
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: &str) -> Option<&Document> {
        self.documents.get(doc_id)
    }

    /// Delete document and its chunks
    pub fn delete_document(&mut self, doc_id: &str) -> Result<bool> {
        if let Some(doc) = self.documents.remove(doc_id) {
            let collection = self.db.collection(&self.config.collection_name)?;

            for chunk_id in &doc.chunk_ids {
                collection.delete(chunk_id)?;
                self.chunks.remove(chunk_id);
            }

            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all documents
    pub fn list_documents(&self) -> Vec<&Document> {
        self.documents.values().collect()
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> RagStats {
        RagStats {
            total_documents: self.documents.len(),
            total_chunks: self.chunks.len(),
            avg_chunks_per_doc: if self.documents.is_empty() {
                0.0
            } else {
                self.chunks.len() as f64 / self.documents.len() as f64
            },
        }
    }
}

/// RAG pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub avg_chunks_per_doc: f64,
}

/// Trait for embedding text
pub trait Embedder: Send + Sync {
    /// Generate embedding for text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;
}

/// Simple mock embedder for testing
pub struct MockEmbedder {
    dimensions: usize,
}

impl MockEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate deterministic pseudo-random embedding
        let mut rng = SimpleRng::new(seed);
        let embedding: Vec<f32> = (0..self.dimensions)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(embedding.into_iter().map(|x| x / norm).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// Simple RNG for deterministic embeddings
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_chunking() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let text = "This is a test document. It has multiple sentences. We want to chunk it.";
        let chunks = pipeline.chunk_fixed_size(text, 20, 5);

        assert!(!chunks.is_empty());
        for (chunk, start, end) in &chunks {
            assert!(!chunk.is_empty());
            assert!(start < end);
        }
    }

    #[test]
    fn test_semantic_chunking() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let text = "First sentence here. Second sentence follows. Third one is longer and contains more content. Final sentence.";
        let chunks = pipeline.chunk_semantic(text, 50, 10);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_paragraph_chunking() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        let chunks = pipeline.chunk_paragraphs(text, 2);

        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_ingest_and_query() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest document
        let doc = pipeline
            .ingest_document("doc1", "Machine learning is a subset of artificial intelligence.", None, &embedder)
            .unwrap();

        assert_eq!(doc.id, "doc1");
        assert!(!doc.chunk_ids.is_empty());

        // Query
        let response = pipeline.query("What is machine learning?", &embedder).unwrap();

        assert!(!response.chunks.is_empty());
        assert!(!response.context.is_empty());
        assert!(!response.citations.is_empty());
    }

    #[test]
    fn test_text_similarity() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let sim = pipeline.text_similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.001);

        let sim = pipeline.text_similarity("hello world", "goodbye world");
        assert!(sim > 0.0 && sim < 1.0);

        let sim = pipeline.text_similarity("hello", "goodbye");
        assert!(sim < 0.5);
    }

    #[test]
    fn test_mock_embedder() {
        let embedder = MockEmbedder::new(128);

        let emb1 = embedder.embed("hello world").unwrap();
        let emb2 = embedder.embed("hello world").unwrap();
        let emb3 = embedder.embed("different text").unwrap();

        assert_eq!(emb1.len(), 128);
        assert_eq!(emb1, emb2); // Same text = same embedding
        assert_ne!(emb1, emb3); // Different text = different embedding
    }
}
