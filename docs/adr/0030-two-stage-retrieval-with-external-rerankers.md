# ADR-0030: Two-Stage Retrieval with External Rerankers

## Status

Accepted

## Context

Vector similarity search using bi-encoders (separate query and document embeddings) is fast but has limitations:

1. **Semantic gap** — Bi-encoders compress documents to fixed-size vectors, losing nuance
2. **Query-document interaction** — Bi-encoders can't model fine-grained term interactions
3. **Recall vs. precision tradeoff** — Fast retrieval optimizes for recall, not precision
4. **Production RAG quality** — Users expect Google-quality relevance, not just similarity

Cross-encoders (jointly encoding query and document) produce more accurate relevance scores but are too slow to run on the entire corpus. The industry solution is **two-stage retrieval**:

```
Stage 1: Fast bi-encoder retrieval (HNSW) → top-100 candidates
Stage 2: Slow cross-encoder reranking → top-10 results
```

Options for reranking:

| Approach | Latency | Quality | Cost |
|----------|---------|---------|------|
| No reranking | Lowest | Baseline | Free |
| Local cross-encoder | Medium | High | Compute |
| Cohere Rerank API | Medium | Very High | API cost |
| ColBERT (late interaction) | Medium | High | Memory |

## Decision

Support **external rerankers** via an async trait interface, with built-in implementations for Cohere and HuggingFace cross-encoders.

### Reranker Trait

```rust
pub trait Reranker: Send + Sync {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = RerankerResult<Vec<RerankResult>>> + Send + 'a>>;
}

pub struct RerankResult {
    pub index: usize,      // Original document index
    pub score: f32,        // Relevance score (higher = more relevant)
    pub text: Option<String>,
}
```

### Built-in Implementations

```rust
// Cohere API (cloud)
let reranker = CohereReranker::new(api_key, "rerank-english-v2.0");

// HuggingFace cross-encoder (local)
let reranker = HuggingFaceReranker::new("cross-encoder/ms-marco-MiniLM-L-6-v2");

// Generic cross-encoder (custom model)
let reranker = CrossEncoderReranker::new(model_path, tokenizer_path);
```

### Integration Pattern

```rust
// Two-stage search
let candidates = collection.search(&query_vec, 100, None)?;  // Stage 1: HNSW

let documents: Vec<&str> = candidates.iter()
    .map(|r| r.metadata.get("text").unwrap().as_str().unwrap())
    .collect();

let reranked = reranker.rerank(query_text, &documents, 10).await?;  // Stage 2

// Map back to original results
let final_results: Vec<SearchResult> = reranked.iter()
    .map(|r| candidates[r.index].clone())
    .collect();
```

### Code References

- `src/reranker.rs:36-71` — RerankerError types for API/model failures
- `src/reranker.rs:73-100` — RerankResult structure
- `src/reranker.rs` — Reranker trait and implementations

## Consequences

### Benefits

1. **Significant quality improvement** — Cross-encoders can improve MRR@10 by 10-30%
2. **Pluggable architecture** — Swap rerankers without changing search code
3. **Cost optimization** — Only rerank top candidates, not entire corpus
4. **Model flexibility** — Use cloud APIs or local models based on requirements
5. **Async-native** — Non-blocking API calls don't stall the server

### Tradeoffs

1. **Added latency** — Reranking adds 50-200ms depending on model and batch size
2. **External dependency** — Cloud rerankers require API keys and network access
3. **Text required** — Reranking needs original text, not just vectors
4. **Cost accumulation** — API-based reranking has per-query costs

### What This Enabled

- Production-quality RAG pipelines with state-of-the-art relevance
- A/B testing different reranking models
- Hybrid strategies (local for cost, cloud for quality)
- Integration with LLM providers' reranking endpoints

### What This Prevented

- Forced choice between speed (bi-encoder only) and quality (full cross-encoder)
- Lock-in to a single reranking provider
- Blocking I/O during reranking operations

### Reranking Quality Guidelines

| Scenario | Recommendation |
|----------|----------------|
| Prototype/testing | No reranking (fastest iteration) |
| Internal search | Local cross-encoder (no API costs) |
| Customer-facing | Cohere/cloud (highest quality) |
| High-volume | Local with batching (cost control) |

### Error Handling

```rust
pub enum RerankerError {
    ApiError(String),      // API request failed
    RateLimited(String),   // Rate limit exceeded (retry with backoff)
    InvalidInput(String),  // Bad query or documents
    ModelError(String),    // Local model inference failed
    NetworkError(String),  // Connection issues
}
```

Reranking failures are non-fatal — the system falls back to Stage 1 results if reranking fails, ensuring degraded quality rather than total failure.
