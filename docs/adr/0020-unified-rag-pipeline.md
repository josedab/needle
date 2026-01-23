# ADR-0020: Unified RAG Pipeline Architecture

## Status

Accepted

## Context

Retrieval-Augmented Generation (RAG) is the dominant use case for vector databases. However, building a production RAG system requires orchestrating many components:

1. **Document ingestion** — Chunking, overlap handling, metadata extraction
2. **Retrieval** — Vector search, filtering, reranking
3. **Context optimization** — Token limits, relevance ordering, deduplication
4. **Response caching** — Avoid redundant LLM calls for similar queries
5. **Evaluation** — Measure retrieval quality and answer relevance

Most users rely on LangChain or LlamaIndex, adding external dependencies and integration complexity. A unified pipeline inside the vector database simplifies the stack and enables optimizations impossible with external orchestration.

## Decision

Needle implements a **built-in RAG pipeline** with the following design:

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RagPipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Chunker         │  Split documents into searchable units   │
├──────────────────┼──────────────────────────────────────────┤
│  Retriever       │  Multi-strategy search (dense + sparse)  │
├──────────────────┼──────────────────────────────────────────┤
│  Reranker        │  Cross-encoder reranking for precision   │
├──────────────────┼──────────────────────────────────────────┤
│  Context Builder │  Assemble LLM-ready context with limits  │
├──────────────────┼──────────────────────────────────────────┤
│  Cache           │  Response caching for efficiency         │
└──────────────────┴──────────────────────────────────────────┘
```

### RAG Configuration

```rust
pub struct RagConfig {
    /// Chunking strategy
    pub chunking: ChunkingStrategy,
    /// Maximum chunk size in tokens
    pub max_chunk_tokens: usize,
    /// Chunk overlap in tokens
    pub chunk_overlap: usize,
    /// Number of chunks to retrieve
    pub top_k: usize,
    /// Enable reranking
    pub enable_reranking: bool,
    /// Maximum context tokens for LLM
    pub max_context_tokens: usize,
    /// Cache configuration
    pub cache_config: Option<CacheConfig>,
}
```

### Chunking Strategies

```rust
pub enum ChunkingStrategy {
    /// Fixed-size chunks by token count
    FixedSize { tokens: usize, overlap: usize },
    /// Sentence-based chunking
    Sentence { max_sentences: usize },
    /// Paragraph-based chunking
    Paragraph,
    /// Semantic chunking (split at topic boundaries)
    Semantic { similarity_threshold: f32 },
    /// Recursive character splitting (like LangChain)
    Recursive { separators: Vec<String> },
}
```

### RAG Response Structure

```rust
pub struct RagResponse {
    /// Retrieved chunks in relevance order
    pub chunks: Vec<RetrievedChunk>,
    /// Formatted context string for LLM
    pub context: String,
    /// Total tokens in context
    pub context_tokens: usize,
    /// Whether response was served from cache
    pub from_cache: bool,
    /// Query metadata
    pub metadata: RagQueryMetadata,
}

pub struct RagQueryMetadata {
    /// Time spent in retrieval
    pub retrieval_time: Duration,
    /// Time spent in reranking
    pub reranking_time: Duration,
    /// Number of candidates before reranking
    pub candidates_before_rerank: usize,
    /// Search strategy used
    pub strategy: SearchStrategy,
}
```

### Response Caching

```rust
pub struct RagCache {
    /// LRU cache of query -> response
    cache: LruCache<String, CachedResponse>,
    /// Semantic similarity threshold for cache hits
    similarity_threshold: f32,
    /// Cache statistics
    stats: RagCacheStats,
}

impl RagCache {
    pub fn get(&self, query_embedding: &[f32]) -> Option<&CachedResponse> {
        // Find semantically similar cached queries
        for (cached_query, response) in self.cache.iter() {
            let similarity = cosine_similarity(query_embedding, &cached_query);
            if similarity > self.similarity_threshold {
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Some(response);
            }
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }
}
```

### Pipeline Usage

```rust
// Create pipeline
let config = RagConfig {
    chunking: ChunkingStrategy::Semantic { similarity_threshold: 0.7 },
    max_chunk_tokens: 512,
    chunk_overlap: 50,
    top_k: 10,
    enable_reranking: true,
    max_context_tokens: 4096,
    cache_config: Some(CacheConfig::default()),
};

let mut pipeline = RagPipeline::new(db.clone(), config)?;

// Ingest documents
pipeline.ingest("collection", documents).await?;

// Query
let response = pipeline.query("What is HNSW?", Some(filter)).await?;

// Use context with LLM
let answer = llm.complete(&format!(
    "Context:\n{}\n\nQuestion: {}\n\nAnswer:",
    response.context,
    query
));
```

### Code References

- `src/rag.rs:97-150` — `RagConfig` configuration
- `src/rag.rs:193-220` — `RagCache` semantic caching
- `src/rag.rs:367-410` — `RagResponse` and `RagQueryMetadata`
- `src/rag.rs:409-600` — `RagPipeline` implementation
- `src/rag.rs:1140-1180` — `RagStats` statistics

## Consequences

### Benefits

1. **Simplified stack** — No LangChain/LlamaIndex dependency for basic RAG
2. **Performance** — Integrated pipeline avoids serialization overhead
3. **Semantic caching** — Similar queries reuse cached responses (30-50% cost reduction)
4. **Unified configuration** — Single place to configure the entire RAG flow
5. **Observability** — Built-in metrics for retrieval quality and latency
6. **Type safety** — Rust types ensure configuration correctness at compile time

### Tradeoffs

1. **Less flexibility** — LangChain offers more chain types and integrations
2. **No LLM integration** — Pipeline returns context; user calls LLM separately
3. **Learning curve** — Users familiar with LangChain need to learn new API
4. **Feature scope** — Not all LangChain features are replicated
5. **Maintenance burden** — Must maintain parity with evolving RAG best practices

### What This Enabled

- Single-binary RAG applications without Python dependencies
- Sub-millisecond retrieval-to-context pipeline
- Production RAG with built-in caching and monitoring
- Mobile/embedded RAG applications (WASM compatible)
- Type-safe RAG configuration

### What This Prevented

- Arbitrary chain composition (agent loops, tool use)
- Direct LLM integration (by design)
- Dynamic prompt engineering within pipeline
- Community-contributed chain types

### Context Optimization Strategy

```rust
impl RagPipeline {
    fn build_context(&self, chunks: &[RetrievedChunk]) -> String {
        let mut context = String::new();
        let mut total_tokens = 0;

        for chunk in chunks {
            let chunk_tokens = self.count_tokens(&chunk.text);

            // Check if adding this chunk exceeds limit
            if total_tokens + chunk_tokens > self.config.max_context_tokens {
                break;
            }

            // Add chunk with source attribution
            context.push_str(&format!(
                "[Source: {}]\n{}\n\n",
                chunk.metadata.get("source").unwrap_or(&"unknown".into()),
                chunk.text
            ));

            total_tokens += chunk_tokens;
        }

        context
    }
}
```

### Evaluation Metrics

```rust
pub struct RagStats {
    /// Total queries processed
    pub queries: u64,
    /// Average retrieval latency
    pub avg_retrieval_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average chunks per query
    pub avg_chunks_retrieved: f64,
    /// Average context tokens
    pub avg_context_tokens: f64,
}
```

### Integration with Hybrid Search

The RAG pipeline leverages existing hybrid search (ADR-0012) for multi-strategy retrieval:

```rust
let retrieval_results = match self.config.retrieval_strategy {
    RetrievalStrategy::DenseOnly => {
        self.dense_search(query_embedding, k).await?
    }
    RetrievalStrategy::Hybrid { dense_weight } => {
        let dense = self.dense_search(query_embedding, k * 2).await?;
        let sparse = self.bm25_search(query_text, k * 2).await?;
        reciprocal_rank_fusion(&dense, &sparse, dense_weight)
    }
    RetrievalStrategy::HyDE => {
        // Hypothetical Document Embeddings
        let hypothetical = self.generate_hypothetical(query_text)?;
        self.dense_search(&hypothetical, k).await?
    }
};
```
