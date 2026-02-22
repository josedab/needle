use super::cache::{CacheKey, CachedRagResponse, RagCache, RagCacheStats};
use super::chunking::{LoadedDocument, RecursiveTextSplitter};
use super::{
    BatchIngestOptions, BatchIngestResult, Chunk, ChunkingStrategy, Citation, ContextStrategy,
    Document, MultiQueryMerge, MultiQueryOptions, RagConfig, RagQueryMetadata, RagResponse,
    RagStats, RetrievedChunk,
};
use super::embedder::Embedder;
use crate::database::Database;
use crate::error::Result;
use crate::SearchResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// RAG Pipeline
pub struct RagPipeline {
    db: Arc<Database>,
    config: RagConfig,
    documents: HashMap<String, Document>,
    chunks: HashMap<String, Chunk>,
    cache: Option<Arc<RagCache>>,
    query_count: AtomicU64,
    total_query_latency_ms: AtomicU64,
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
            Some(Arc::new(RagCache::new(
                config.cache_size,
                config.cache_ttl_seconds,
            )))
        } else {
            None
        };

        Ok(Self {
            db,
            config,
            documents: HashMap::new(),
            chunks: HashMap::new(),
            cache,
            query_count: AtomicU64::new(0),
            total_query_latency_ms: AtomicU64::new(0),
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

    /// Ingest a pre-loaded document (from [`DocumentLoader`]).
    pub fn ingest_loaded(
        &mut self,
        doc: &LoadedDocument,
        embedder: &dyn Embedder,
    ) -> Result<Document> {
        self.ingest_document(&doc.id, &doc.text, doc.metadata.clone(), embedder)
    }

    /// Ingest a document with automatic chunking
    pub fn ingest_document(
        &mut self,
        doc_id: &str,
        text: &str,
        metadata: Option<serde_json::Value>,
        embedder: &dyn Embedder,
    ) -> Result<Document> {
        self.ingest_with_strategy(
            doc_id,
            text,
            metadata,
            &self.config.chunking.clone(),
            embedder,
        )
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
                .unwrap_or_default()
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
    pub fn query(&self, query: &str, embedder: &dyn Embedder) -> Result<RagResponse> {
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
                let chunk = Self::result_to_chunk(&result)?;
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
            Self::rerank_chunks(query, &mut retrieved);
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
        let citations = Self::build_citations(&retrieved);

        let total_latency = start_time.elapsed().as_millis() as u64;

        // Track query metrics
        self.query_count.fetch_add(1, Ordering::Relaxed);
        self.total_query_latency_ms
            .fetch_add(total_latency, Ordering::Relaxed);

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
    pub(crate) fn chunk_text(&self, text: &str, strategy: &ChunkingStrategy) -> Vec<(String, usize, usize)> {
        match strategy {
            ChunkingStrategy::FixedSize {
                chunk_size,
                overlap,
            } => Self::chunk_fixed_size(text, *chunk_size, *overlap),
            ChunkingStrategy::SlidingWindow {
                window_size,
                step_size,
            } => Self::chunk_sliding_window(text, *window_size, *step_size),
            ChunkingStrategy::Semantic {
                max_chunk_size,
                min_chunk_size,
            } => self.chunk_semantic(text, *max_chunk_size, *min_chunk_size),
            ChunkingStrategy::Hierarchical { levels } => self.chunk_hierarchical(text, levels),
            ChunkingStrategy::Paragraph { max_paragraphs } => {
                Self::chunk_paragraphs(text, *max_paragraphs)
            }
            ChunkingStrategy::Recursive { chunk_size, chunk_overlap } => {
                RecursiveTextSplitter::new(*chunk_size, *chunk_overlap).split(text)
            }
        }
    }

    pub(crate) fn chunk_fixed_size(
        text: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Vec<(String, usize, usize)> {
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

    fn chunk_sliding_window(
        text: &str,
        window_size: usize,
        step_size: usize,
    ) -> Vec<(String, usize, usize)> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut chunks = Vec::new();
        let mut word_start = 0;

        while word_start < words.len() {
            let word_end = (word_start + window_size).min(words.len());
            let chunk = words[word_start..word_end].join(" ");

            // Calculate character positions (approximate)
            let char_start = words[..word_start]
                .iter()
                .map(|w| w.len() + 1)
                .sum::<usize>();
            let char_end = char_start + chunk.len();

            chunks.push((chunk, char_start, char_end));

            if word_end >= words.len() {
                break;
            }

            word_start += step_size;
        }

        chunks
    }

    pub(crate) fn chunk_semantic(
        &self,
        text: &str,
        max_size: usize,
        min_size: usize,
    ) -> Vec<(String, usize, usize)> {
        let mut chunks = Vec::new();
        let sentences = Self::split_sentences(text);

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
        Self::chunk_fixed_size(text, chunk_size, chunk_size / 4)
    }

    pub(crate) fn chunk_paragraphs(text: &str, max_paragraphs: usize) -> Vec<(String, usize, usize)> {
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

    fn split_sentences(text: &str) -> Vec<String> {
        // Simple sentence splitting (could be improved with NLP)
        let mut sentences = Vec::new();
        let mut current = String::new();

        for c in text.chars() {
            current.push(c);
            if (c == '.' || c == '!' || c == '?') && current.len() > 1 {
                sentences.push(current.trim().to_string());
                current = String::new();
            }
        }

        if !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
        }

        sentences
    }

    fn result_to_chunk(result: &SearchResult) -> Option<Chunk> {
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

    fn rerank_chunks(query: &str, chunks: &mut [RetrievedChunk]) {
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
                Self::text_similarity(&existing.chunk.text, &chunk.chunk.text)
                    > self.config.dedup_threshold
            });

            if !is_duplicate {
                result.push(chunk);
            }
        }

        result
    }

    pub(crate) fn text_similarity(a: &str, b: &str) -> f32 {
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
                Self::format_chunks(chunks)
            }
            ContextStrategy::Truncate => {
                // Simple truncation at character limit
                let context = Self::format_chunks(chunks);
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
            ContextStrategy::Compress {
                redundancy_threshold,
            } => {
                // Remove redundant content
                self.assemble_compressed(chunks, max_chars, *redundancy_threshold)
            }
        }
    }

    /// Format chunks into context string
    fn format_chunks(chunks: &[RetrievedChunk]) -> String {
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{}] {}", i + 1, c.chunk.text))
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Estimate character count for a chunk in formatted context
    fn estimate_chunk_chars(chunk: &RetrievedChunk, index: usize) -> usize {
        // "[N] " prefix + text + "\n\n" separator
        format!("[{}] ", index + 1).len() + chunk.chunk.text.len() + 2
    }

    /// Assemble context by score priority within budget
    fn assemble_by_score(&self, chunks: &[RetrievedChunk], max_chars: usize) -> String {
        // Chunks are already sorted by score
        let mut result = Vec::new();
        let mut total_chars = 0;

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_chars = Self::estimate_chunk_chars(chunk, result.len());
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
                let chunk_chars = Self::estimate_chunk_chars(chunk, selected.len());
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
                        .map(|s| Self::text_similarity(&chunk.chunk.text, &s.chunk.text))
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
            let chunk_chars = Self::estimate_chunk_chars(chunk, selected.len());
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
            let chunk_chars = Self::estimate_chunk_chars(chunk, selected.len());

            // Check for redundancy with already selected chunks
            let is_redundant = selected.iter().any(|s| {
                Self::text_similarity(&chunk.chunk.text, &s.chunk.text) >= redundancy_threshold
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

    fn build_citations(chunks: &[RetrievedChunk]) -> Vec<Citation> {
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
            total_queries: self.query_count.load(Ordering::Relaxed),
            avg_query_latency_ms: {
                let count = self.query_count.load(Ordering::Relaxed);
                if count == 0 {
                    0.0
                } else {
                    self.total_query_latency_ms.load(Ordering::Relaxed) as f64 / count as f64
                }
            },
            cache_hit_rate: self.cache_hit_rate(),
        }
    }

    /// Ingest multiple documents in batch
    pub fn batch_ingest(
        &mut self,
        documents: &[(&str, &str, Option<serde_json::Value>)],
        embedder: &dyn Embedder,
        options: &BatchIngestOptions,
    ) -> BatchIngestResult {
        let start = std::time::Instant::now();
        let mut result = BatchIngestResult {
            ingested: 0,
            skipped: 0,
            failed: 0,
            errors: Vec::new(),
            elapsed_ms: 0,
        };

        let strategy = options.chunking.clone().unwrap_or_else(|| self.config.chunking.clone());

        for (doc_id, text, metadata) in documents {
            if options.skip_existing && self.documents.contains_key(*doc_id) {
                result.skipped += 1;
                continue;
            }

            match self.ingest_with_strategy(doc_id, text, metadata.clone(), &strategy, embedder) {
                Ok(_) => result.ingested += 1,
                Err(e) => {
                    result.failed += 1;
                    result.errors.push(format!("{}: {}", doc_id, e));
                }
            }
        }

        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Query with multiple query expansions for improved recall.
    ///
    /// Generates paraphrased queries and merges results from all expansions.
    pub fn multi_query(
        &mut self,
        queries: &[&str],
        embedder: &dyn Embedder,
        options: &MultiQueryOptions,
    ) -> Result<RagResponse> {
        let start_time = std::time::Instant::now();

        let mut all_chunks: Vec<RetrievedChunk> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for query in queries {
            if let Ok(response) = self.query(query, embedder) {
                for chunk in response.chunks {
                    if seen_ids.insert(chunk.chunk.id.clone()) {
                        all_chunks.push(chunk);
                    }
                }
            }
        }

        // Merge based on strategy
        match &options.merge_strategy {
            MultiQueryMerge::RoundRobin => {
                // Already in interleaved order from sequential queries
            }
            MultiQueryMerge::ReciprocalRankFusion { k } => {
                // RRF: score = sum(1 / (k + rank_i)) across all queries
                let mut rrf_scores: HashMap<String, f32> = HashMap::new();
                for (rank, chunk) in all_chunks.iter().enumerate() {
                    *rrf_scores.entry(chunk.chunk.id.clone()).or_default() +=
                        1.0 / (k + rank as f32 + 1.0);
                }
                all_chunks.sort_by(|a, b| {
                    let sa = rrf_scores.get(&a.chunk.id).copied().unwrap_or(0.0);
                    let sb = rrf_scores.get(&b.chunk.id).copied().unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            MultiQueryMerge::UnionBestScore => {
                all_chunks.sort_by(|a, b| {
                    b.final_score
                        .partial_cmp(&a.final_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        all_chunks.truncate(self.config.top_k);

        let context = self.assemble_context(&all_chunks);
        let citations = Self::build_citations(&all_chunks);
        let total_latency = start_time.elapsed().as_millis() as u64;

        Ok(RagResponse {
            chunks: all_chunks,
            context,
            citations,
            metadata: RagQueryMetadata {
                chunks_retrieved: queries.len() * self.config.top_k,
                chunks_after_dedup: 0,
                retrieval_latency_ms: total_latency,
                rerank_latency_ms: None,
                total_latency_ms: total_latency,
            },
        })
    }
}

/// Builder for constructing a RAG pipeline with a fluent API.
///
/// # Example
/// ```rust,ignore
/// use needle::rag::{RagPipelineBuilder, ChunkingStrategy, ContextStrategy};
///
/// let pipeline = RagPipelineBuilder::new()
///     .collection("docs")
///     .dimensions(384)
///     .chunker(ChunkingStrategy::Semantic { max_chunk_size: 512, min_chunk_size: 100 })
///     .context_strategy(ContextStrategy::ScorePriority)
///     .top_k(10)
///     .rerank_top_n(5)
///     .with_cache(1000, 3600)
///     .build(db)?;
/// ```
pub struct RagPipelineBuilder {
    config: RagConfig,
    custom_rerank_top_n: Option<usize>,
}

impl RagPipelineBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: RagConfig::default(),
            custom_rerank_top_n: None,
        }
    }

    /// Set the collection name.
    #[must_use]
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.config.collection_name = name.into();
        self
    }

    /// Set embedding dimensions.
    #[must_use]
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.config.dimensions = dims;
        self
    }

    /// Set the chunking strategy.
    #[must_use]
    pub fn chunker(mut self, strategy: ChunkingStrategy) -> Self {
        self.config.chunking = strategy;
        self
    }

    /// Set context assembly strategy.
    #[must_use]
    pub fn context_strategy(mut self, strategy: ContextStrategy) -> Self {
        self.config.context_strategy = strategy;
        self
    }

    /// Set the number of top results to retrieve.
    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.config.top_k = k;
        self
    }

    /// Set maximum context budget in tokens.
    #[must_use]
    pub fn max_context_tokens(mut self, tokens: usize) -> Self {
        self.config.max_context_tokens = tokens;
        self
    }

    /// Enable reranking with top-N limit.
    #[must_use]
    pub fn rerank_top_n(mut self, n: usize) -> Self {
        self.custom_rerank_top_n = Some(n);
        self
    }

    /// Enable response caching with capacity and TTL.
    #[must_use]
    pub fn with_cache(mut self, capacity: usize, ttl_seconds: u64) -> Self {
        self.config.cache_enabled = true;
        self.config.cache_size = capacity;
        self.config.cache_ttl_seconds = ttl_seconds;
        self
    }

    /// Disable response caching.
    #[must_use]
    pub fn without_cache(mut self) -> Self {
        self.config.cache_enabled = false;
        self
    }

    /// Enable hybrid (dense + sparse) search.
    #[must_use]
    pub fn hybrid_search(mut self, alpha: f32) -> Self {
        self.config.hybrid_search = true;
        self.config.hybrid_alpha = alpha;
        self
    }

    /// Enable deduplication with the given similarity threshold.
    #[must_use]
    pub fn deduplicate(mut self, threshold: f32) -> Self {
        self.config.deduplicate = true;
        self.config.dedup_threshold = threshold;
        self
    }

    /// Build the pipeline from a shared `Database`.
    pub fn build(self, db: Arc<Database>) -> Result<RagPipeline> {
        RagPipeline::new(db, self.config)
    }

    /// Get the built config without creating the pipeline.
    pub fn into_config(self) -> RagConfig {
        self.config
    }
}

impl Default for RagPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to create a RAG pipeline with minimal configuration.
pub fn quick_rag_pipeline(db: Arc<Database>, dimensions: usize) -> Result<RagPipeline> {
    RagPipelineBuilder::new()
        .dimensions(dimensions)
        .chunker(ChunkingStrategy::Semantic {
            max_chunk_size: 512,
            min_chunk_size: 100,
        })
        .context_strategy(ContextStrategy::ScorePriority)
        .top_k(10)
        .with_cache(500, 3600)
        .build(db)
}
