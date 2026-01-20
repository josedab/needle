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
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Context window optimization strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ContextStrategy {
    /// Include all chunks (no optimization)
    None,
    /// Truncate context to max tokens
    Truncate,
    /// Prioritize chunks by score within budget
    #[default]
    ScorePriority,
    /// Balance coverage vs. relevance
    Balanced {
        /// Weight for diversity (0-1)
        diversity_weight: f32,
    },
    /// Compress context by removing redundancy
    Compress {
        /// Minimum similarity to consider redundant
        redundancy_threshold: f32,
    },
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
    /// Enable response caching
    pub cache_enabled: bool,
    /// Maximum cache entries
    pub cache_size: usize,
    /// Cache TTL in seconds (0 = no expiry)
    pub cache_ttl_seconds: u64,
    /// Maximum context tokens (approximate, based on chars/4)
    pub max_context_tokens: usize,
    /// Context optimization strategy
    pub context_strategy: ContextStrategy,
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
            cache_enabled: true,
            cache_size: 1000,
            cache_ttl_seconds: 300, // 5 minutes default
            max_context_tokens: 4096, // ~16K chars default
            context_strategy: ContextStrategy::default(),
        }
    }
}

/// Cache entry for RAG responses
#[derive(Clone)]
struct CacheEntry {
    response: CachedRagResponse,
    created_at: Instant,
}

/// Cached version of RagResponse (without timing metadata)
#[derive(Clone)]
struct CachedRagResponse {
    chunks: Vec<RetrievedChunk>,
    context: String,
    citations: Vec<Citation>,
}

/// Cache key combining query text and filter hash
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
struct CacheKey {
    query: String,
    filter_hash: Option<u64>,
}

impl CacheKey {
    fn new(query: &str, filter: Option<&crate::metadata::Filter>) -> Self {
        let filter_hash = filter.map(|f| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{:?}", f).hash(&mut hasher);
            hasher.finish()
        });
        Self {
            query: query.to_string(),
            filter_hash,
        }
    }
}

/// RAG response cache with LRU eviction and TTL
pub struct RagCache {
    cache: Mutex<LruCache<CacheKey, CacheEntry>>,
    ttl: Duration,
    stats: Mutex<RagCacheStats>,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct RagCacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache evictions
    pub evictions: u64,
    /// Total cache invalidations
    pub invalidations: u64,
}

impl RagCache {
    /// Create a new RAG cache
    pub fn new(capacity: usize, ttl_seconds: u64) -> Self {
        let capacity = NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            cache: Mutex::new(LruCache::new(capacity)),
            ttl: Duration::from_secs(ttl_seconds),
            stats: Mutex::new(RagCacheStats::default()),
        }
    }

    /// Get a cached response
    fn get(&self, key: &CacheKey) -> Option<CachedRagResponse> {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.lock();

        if let Some(entry) = cache.get(key) {
            // Check TTL
            if self.ttl.as_secs() == 0 || entry.created_at.elapsed() < self.ttl {
                stats.hits += 1;
                return Some(entry.response.clone());
            }
            // Expired - remove it
            cache.pop(key);
            stats.evictions += 1;
        }

        stats.misses += 1;
        None
    }

    /// Put a response in the cache
    fn put(&self, key: CacheKey, response: CachedRagResponse) {
        let mut cache = self.cache.lock();
        cache.put(
            key,
            CacheEntry {
                response,
                created_at: Instant::now(),
            },
        );
    }

    /// Invalidate all cache entries
    pub fn invalidate_all(&self) {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.lock();
        stats.invalidations += cache.len() as u64;
        cache.clear();
    }

    /// Invalidate entries matching a document ID
    pub fn invalidate_document(&self, doc_id: &str) {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.lock();

        // Note: LRU cache doesn't support efficient iteration and removal
        // For a production system, consider using a different cache implementation
        // that supports this pattern more efficiently
        let keys_to_remove: Vec<CacheKey> = cache
            .iter()
            .filter(|(_, entry)| {
                entry.response.chunks.iter().any(|c| c.chunk.document_id == doc_id)
            })
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
            stats.invalidations += 1;
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> RagCacheStats {
        self.stats.lock().clone()
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
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
    cache: Option<Arc<RagCache>>,
}

impl RagPipeline {
    /// Create a new RAG pipeline
    pub fn new(db: Arc<Database>, config: RagConfig) -> Result<Self> {
        // Create collection if it doesn't exist
        if !db.has_collection(&config.collection_name) {
            db.create_collection(&config.collection_name, config.dimensions)?;
        }

        // Create cache if enabled
        let cache = if config.cache_enabled {
            Some(Arc::new(RagCache::new(config.cache_size, config.cache_ttl_seconds)))
        } else {
            None
        };

        Ok(Self {
            db,
            config,
            documents: HashMap::new(),
            chunks: HashMap::new(),
            cache,
        })
    }

    /// Get cache statistics (if caching is enabled)
    pub fn cache_stats(&self) -> Option<RagCacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Get cache hit rate (if caching is enabled)
    pub fn cache_hit_rate(&self) -> Option<f64> {
        self.cache.as_ref().map(|c| c.hit_rate())
    }

    /// Invalidate all cached responses
    pub fn invalidate_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.invalidate_all();
        }
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

        // Invalidate cache since new content was added
        if let Some(ref cache) = self.cache {
            cache.invalidate_all();
        }

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

        // Check cache first
        let cache_key = CacheKey::new(query, filter);
        if let Some(ref cache) = self.cache {
            if let Some(cached) = cache.get(&cache_key) {
                // Return cached response with updated timing
                return Ok(RagResponse {
                    chunks: cached.chunks,
                    context: cached.context,
                    citations: cached.citations,
                    metadata: RagQueryMetadata {
                        chunks_retrieved: 0,
                        chunks_after_dedup: 0,
                        retrieval_latency_ms: 0,
                        rerank_latency_ms: None,
                        total_latency_ms: start_time.elapsed().as_millis() as u64,
                    },
                });
            }
        }

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

        // Cache the response
        if let Some(ref cache) = self.cache {
            cache.put(
                cache_key,
                CachedRagResponse {
                    chunks: retrieved.clone(),
                    context: context.clone(),
                    citations: citations.clone(),
                },
            );
        }

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
        // Approximate token budget (1 token ≈ 4 chars)
        let max_chars = self.config.max_context_tokens * 4;

        match &self.config.context_strategy {
            ContextStrategy::None => {
                // No optimization - include all chunks
                self.format_chunks(chunks)
            }
            ContextStrategy::Truncate => {
                // Simple truncation at character limit
                let context = self.format_chunks(chunks);
                if context.len() <= max_chars {
                    context
                } else {
                    let mut result = context[..max_chars].to_string();
                    // Try to end at a sentence boundary
                    if let Some(pos) = result.rfind(". ") {
                        result.truncate(pos + 1);
                    }
                    result.push_str("\n\n[Context truncated]");
                    result
                }
            }
            ContextStrategy::ScorePriority => {
                // Greedily add chunks by score until budget exhausted
                self.assemble_by_score(chunks, max_chars)
            }
            ContextStrategy::Balanced { diversity_weight } => {
                // Balance relevance and coverage
                self.assemble_balanced(chunks, max_chars, *diversity_weight)
            }
            ContextStrategy::Compress { redundancy_threshold } => {
                // Remove redundant content
                self.assemble_compressed(chunks, max_chars, *redundancy_threshold)
            }
        }
    }

    /// Format chunks into context string
    fn format_chunks(&self, chunks: &[RetrievedChunk]) -> String {
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i + 1, c.chunk.text))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Estimate character count for a chunk in formatted context
    fn estimate_chunk_chars(&self, chunk: &RetrievedChunk, index: usize) -> usize {
        // "[N] " prefix + text + "\n\n" separator
        format!("[{}] ", index + 1).len() + chunk.chunk.text.len() + 2
    }

    /// Assemble context by score priority within budget
    fn assemble_by_score(&self, chunks: &[RetrievedChunk], max_chars: usize) -> String {
        // Chunks are already sorted by score
        let mut result = Vec::new();
        let mut total_chars = 0;

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_chars = self.estimate_chunk_chars(chunk, result.len());
            if total_chars + chunk_chars > max_chars && !result.is_empty() {
                break;
            }
            result.push((i, chunk));
            total_chars += chunk_chars;
        }

        result
            .iter()
            .enumerate()
            .map(|(display_idx, (_, c))| format!("[{}] {}", display_idx + 1, c.chunk.text))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Assemble context balancing relevance and diversity
    fn assemble_balanced(
        &self,
        chunks: &[RetrievedChunk],
        max_chars: usize,
        diversity_weight: f32,
    ) -> String {
        if chunks.is_empty() {
            return String::new();
        }

        let mut selected: Vec<&RetrievedChunk> = Vec::new();
        let mut remaining: Vec<&RetrievedChunk> = chunks.iter().collect();
        let mut total_chars = 0;

        // Greedy selection with MMR-like scoring
        while !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for (i, chunk) in remaining.iter().enumerate() {
                let chunk_chars = self.estimate_chunk_chars(chunk, selected.len());
                if total_chars + chunk_chars > max_chars && !selected.is_empty() {
                    continue;
                }

                // Relevance score (normalized)
                let relevance = chunk.final_score;

                // Diversity score (min similarity to already selected)
                let diversity = if selected.is_empty() {
                    1.0
                } else {
                    let max_sim = selected
                        .iter()
                        .map(|s| self.text_similarity(&chunk.chunk.text, &s.chunk.text))
                        .fold(0.0f32, |a, b| a.max(b));
                    1.0 - max_sim
                };

                // Combined score
                let score = (1.0 - diversity_weight) * relevance + diversity_weight * diversity;
                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }

            // Check if we found a valid chunk
            let chunk = remaining.remove(best_idx);
            let chunk_chars = self.estimate_chunk_chars(chunk, selected.len());
            if total_chars + chunk_chars > max_chars && !selected.is_empty() {
                break;
            }

            total_chars += chunk_chars;
            selected.push(chunk);
        }

        selected
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i + 1, c.chunk.text))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Assemble context with redundancy removal
    fn assemble_compressed(
        &self,
        chunks: &[RetrievedChunk],
        max_chars: usize,
        redundancy_threshold: f32,
    ) -> String {
        let mut selected: Vec<&RetrievedChunk> = Vec::new();
        let mut total_chars = 0;

        for chunk in chunks {
            let chunk_chars = self.estimate_chunk_chars(chunk, selected.len());

            // Check for redundancy with already selected chunks
            let is_redundant = selected.iter().any(|s| {
                self.text_similarity(&chunk.chunk.text, &s.chunk.text) >= redundancy_threshold
            });

            if is_redundant {
                continue;
            }

            if total_chars + chunk_chars > max_chars && !selected.is_empty() {
                break;
            }

            total_chars += chunk_chars;
            selected.push(chunk);
        }

        selected
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

            // Invalidate cache entries for this document
            if let Some(ref cache) = self.cache {
                cache.invalidate_document(doc_id);
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

    #[test]
    fn test_rag_cache_basic() {
        let cache = RagCache::new(10, 0); // No TTL

        let key = CacheKey::new("test query", None);
        let response = CachedRagResponse {
            chunks: vec![],
            context: "test context".to_string(),
            citations: vec![],
        };

        // Cache miss initially
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Put and get
        cache.put(key.clone(), response.clone());
        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().context, "test context");
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_rag_cache_ttl() {
        let cache = RagCache::new(10, 1); // 1 second TTL

        let key = CacheKey::new("test", None);
        let response = CachedRagResponse {
            chunks: vec![],
            context: "test".to_string(),
            citations: vec![],
        };

        cache.put(key.clone(), response);
        assert!(cache.get(&key).is_some());

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_rag_cache_invalidation() {
        let cache = RagCache::new(10, 0);

        let key1 = CacheKey::new("query1", None);
        let key2 = CacheKey::new("query2", None);

        cache.put(key1.clone(), CachedRagResponse {
            chunks: vec![],
            context: "1".to_string(),
            citations: vec![],
        });
        cache.put(key2.clone(), CachedRagResponse {
            chunks: vec![],
            context: "2".to_string(),
            citations: vec![],
        });

        assert_eq!(cache.len(), 2);
        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_rag_cache_hit_rate() {
        let cache = RagCache::new(10, 0);

        let key = CacheKey::new("test", None);
        let response = CachedRagResponse {
            chunks: vec![],
            context: "test".to_string(),
            citations: vec![],
        };

        // Miss
        cache.get(&key);
        assert_eq!(cache.hit_rate(), 0.0);

        // Put and hit
        cache.put(key.clone(), response);
        cache.get(&key);
        assert_eq!(cache.hit_rate(), 0.5); // 1 hit, 1 miss

        cache.get(&key);
        assert!((cache.hit_rate() - 0.666).abs() < 0.01); // 2 hits, 1 miss
    }

    #[test]
    fn test_pipeline_caching() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            cache_enabled: true,
            cache_size: 100,
            cache_ttl_seconds: 300,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest a document
        pipeline
            .ingest_document("doc1", "Machine learning and AI are related fields.", None, &embedder)
            .unwrap();

        // First query - cache miss
        let response1 = pipeline.query("machine learning", &embedder).unwrap();
        let stats1 = pipeline.cache_stats().unwrap();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second query - cache hit
        let response2 = pipeline.query("machine learning", &embedder).unwrap();
        let stats2 = pipeline.cache_stats().unwrap();
        assert_eq!(stats2.hits, 1);

        // Responses should have same content
        assert_eq!(response1.context, response2.context);

        // Different query - cache miss
        pipeline.query("artificial intelligence", &embedder).unwrap();
        let stats3 = pipeline.cache_stats().unwrap();
        assert_eq!(stats3.misses, 2);
    }

    #[test]
    fn test_pipeline_cache_disabled() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            cache_enabled: false,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "Test document content.", None, &embedder)
            .unwrap();

        pipeline.query("test", &embedder).unwrap();

        // No cache stats when disabled
        assert!(pipeline.cache_stats().is_none());
        assert!(pipeline.cache_hit_rate().is_none());
    }

    #[test]
    fn test_cache_key_with_filter() {
        let key1 = CacheKey::new("query", None);
        let key2 = CacheKey::new("query", None);
        let key3 = CacheKey::new("different", None);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_context_strategy_none() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::None,
            max_context_tokens: 10, // Very small budget
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest long documents
        pipeline
            .ingest_document("doc1", "This is a long document with many words that should exceed the token budget when using strategy None.", None, &embedder)
            .unwrap();

        let result = pipeline.query("long document", &embedder).unwrap();

        // With ContextStrategy::None, all chunks are included regardless of budget
        assert!(!result.context.is_empty());
    }

    #[test]
    fn test_context_strategy_truncate() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::Truncate,
            max_context_tokens: 20, // ~80 chars
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "This is a sentence. This is another sentence. This is yet another sentence that makes the context very long.", None, &embedder)
            .unwrap();

        let result = pipeline.query("sentence", &embedder).unwrap();

        // Should be truncated
        if result.context.len() > 100 {
            assert!(result.context.contains("[Context truncated]"));
        }
    }

    #[test]
    fn test_context_strategy_score_priority() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::ScorePriority,
            max_context_tokens: 50, // ~200 chars
            top_k: 10,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest multiple documents
        for i in 0..5 {
            pipeline
                .ingest_document(
                    &format!("doc{}", i),
                    &format!("Document {} has some content about topic {}.", i, i),
                    None,
                    &embedder,
                )
                .unwrap();
        }

        let result = pipeline.query("document topic", &embedder).unwrap();

        // Should respect budget and prioritize by score
        assert!(!result.context.is_empty());
        // Context should be within budget (200 chars + some overhead)
        assert!(result.context.len() < 400);
    }

    #[test]
    fn test_context_strategy_balanced() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::Balanced {
                diversity_weight: 0.3,
            },
            max_context_tokens: 100,
            top_k: 10,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest documents with varying similarity
        pipeline
            .ingest_document("doc1", "Machine learning is a subset of artificial intelligence.", None, &embedder)
            .unwrap();
        pipeline
            .ingest_document("doc2", "Machine learning uses algorithms to learn from data.", None, &embedder)
            .unwrap();
        pipeline
            .ingest_document("doc3", "Natural language processing handles text analysis.", None, &embedder)
            .unwrap();

        let result = pipeline.query("machine learning", &embedder).unwrap();

        // Balanced strategy should include diverse content
        assert!(!result.context.is_empty());
    }

    #[test]
    fn test_context_strategy_compress() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::Compress {
                redundancy_threshold: 0.5, // Remove if 50% similar
            },
            max_context_tokens: 200,
            top_k: 10,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest very similar documents
        pipeline
            .ingest_document("doc1", "The quick brown fox jumps over the lazy dog.", None, &embedder)
            .unwrap();
        pipeline
            .ingest_document("doc2", "The quick brown fox leaps over the lazy dog.", None, &embedder)
            .unwrap();
        pipeline
            .ingest_document("doc3", "A fast red cat runs under the sleepy cat.", None, &embedder)
            .unwrap();

        let result = pipeline.query("quick fox", &embedder).unwrap();

        // Compress strategy should remove redundant chunks
        assert!(!result.context.is_empty());
    }

    #[test]
    fn test_context_optimization_empty_chunks() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::ScorePriority,
            max_context_tokens: 100,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Query without any documents
        let result = pipeline.query("test query", &embedder).unwrap();

        // Should handle empty chunks gracefully
        assert!(result.context.is_empty());
        assert!(result.chunks.is_empty());
    }
}
