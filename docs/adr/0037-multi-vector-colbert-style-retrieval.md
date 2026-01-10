# ADR-0037: Multi-Vector ColBERT-Style Retrieval

## Status

Accepted

## Context

Single-vector document representations have fundamental limitations:

1. **Information bottleneck** — Compressing a document to one vector loses nuance
2. **Length sensitivity** — Long documents dilute important passages
3. **Query-document mismatch** — Short queries vs. long documents have different semantics
4. **Polysemy** — Same word in different contexts needs different representations

ColBERT (Contextualized Late Interaction over BERT) introduced **late interaction**:
- Documents are represented as multiple vectors (one per token)
- Queries are also multiple vectors
- Similarity computed via **MaxSim**: max similarity between each query token and all document tokens

```
Query:  [q1] [q2] [q3]
           ↓    ↓    ↓
         ┌─────────────────────────┐
         │ max  max  max           │  ← MaxSim
         │  ↓    ↓    ↓            │
Doc:    [d1] [d2] [d3] [d4] [d5]   │
         └─────────────────────────┘
Score = sum of max similarities
```

## Decision

Implement **multi-vector document representation** with **MaxSim scoring** for ColBERT-style late interaction retrieval.

### MultiVector Structure

```rust
/// A document represented as multiple vectors (token-level embeddings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVector {
    /// Document identifier
    pub id: String,

    /// Token-level vectors (e.g., 128-dim ColBERT embeddings)
    pub vectors: Vec<Vec<f32>>,

    /// Optional: token strings for debugging/visualization
    pub tokens: Option<Vec<String>>,

    /// Document metadata
    pub metadata: Value,
}

impl MultiVector {
    pub fn new(id: &str, vectors: Vec<Vec<f32>>) -> Self {
        Self {
            id: id.to_string(),
            vectors,
            tokens: None,
            metadata: Value::Null,
        }
    }

    /// Number of vectors (tokens) in this document
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Vector dimensionality
    pub fn dim(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }
}
```

### MaxSim Scoring

```rust
impl MultiVectorIndex {
    /// MaxSim: for each query vector, find max similarity to any doc vector
    pub fn maxsim_score(
        &self,
        query_vectors: &[Vec<f32>],
        doc: &MultiVector,
    ) -> f32 {
        query_vectors.iter()
            .map(|q| {
                // Max similarity between this query token and all doc tokens
                doc.vectors.iter()
                    .map(|d| self.similarity(q, d))
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .sum()  // Sum of max similarities
    }

    /// Search with MaxSim scoring
    pub fn search(
        &self,
        query_vectors: &[Vec<f32>],
        k: usize,
    ) -> Vec<MultiVectorSearchResult> {
        let mut scores: Vec<(String, f32)> = self.documents.iter()
            .map(|doc| (doc.id.clone(), self.maxsim_score(query_vectors, doc)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(k);

        scores.into_iter()
            .map(|(id, score)| MultiVectorSearchResult { id, score })
            .collect()
    }
}
```

### Efficient Retrieval with Candidate Generation

Full MaxSim is O(Q × D × N) where Q=query tokens, D=doc tokens, N=documents. For efficiency:

```rust
impl MultiVectorIndex {
    /// Two-stage retrieval for efficiency
    pub fn search_efficient(
        &self,
        query_vectors: &[Vec<f32>],
        k: usize,
        candidate_factor: usize,  // e.g., 10x
    ) -> Vec<MultiVectorSearchResult> {
        // Stage 1: Get candidates via single-vector approximation
        // Use centroid of query vectors for initial HNSW search
        let query_centroid = self.centroid(query_vectors);
        let candidates = self.hnsw_index.search(&query_centroid, k * candidate_factor);

        // Stage 2: Re-rank candidates with full MaxSim
        let mut scored: Vec<(String, f32)> = candidates.iter()
            .filter_map(|c| self.documents.get(&c.id))
            .map(|doc| (doc.id.clone(), self.maxsim_score(query_vectors, doc)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.truncate(k);

        scored.into_iter()
            .map(|(id, score)| MultiVectorSearchResult { id, score })
            .collect()
    }
}
```

### Storage Optimization

ColBERT vectors are typically 128-dimensional (vs. 768+ for single vectors), but there are many per document:

```rust
pub struct MultiVectorStorageConfig {
    /// Quantization for storage efficiency
    pub quantization: QuantizationType,  // None, Scalar8, Binary

    /// Max vectors per document (truncate long docs)
    pub max_vectors_per_doc: usize,

    /// Whether to store token strings
    pub store_tokens: bool,
}

// Typical ColBERT document:
// - 512 tokens × 128 dims × 4 bytes = 256KB per document (uncompressed)
// - With scalar quantization: 512 × 128 × 1 byte = 64KB per document
// - With binary quantization: 512 × 128 / 8 = 8KB per document
```

### Code References

- `src/multivec.rs` — MultiVector and MultiVectorIndex implementation
- `src/lib.rs` — Public exports for multi-vector types
- `CLAUDE.md` — Multi-vector ColBERT-style retrieval documentation

## Consequences

### Benefits

1. **Superior retrieval quality** — ColBERT outperforms single-vector by 10-20% on benchmarks
2. **Token-level matching** — Captures fine-grained semantic relationships
3. **Query term importance** — Each query token contributes independently
4. **Passage-level precision** — Important passages aren't diluted by document length

### Tradeoffs

1. **Storage overhead** — ~100x more vectors per document
2. **Indexing time** — More vectors to add to index
3. **Query latency** — MaxSim is more expensive than single cosine
4. **Model dependency** — Requires ColBERT-style encoder (not standard embeddings)

### What This Enabled

- State-of-the-art retrieval quality for production RAG
- Fine-grained relevance scoring at token level
- Better handling of long documents
- Interpretable matching (which tokens matched)

### What This Prevented

- Information loss from single-vector compression
- Over-reliance on document length normalization heuristics
- Missing partial matches (some query terms match, others don't)

### Performance Characteristics

| Metric | Single Vector | Multi-Vector (ColBERT) |
|--------|---------------|------------------------|
| Storage per doc | 3KB (768-dim) | 64KB (512×128, quantized) |
| Index build time | 1x | 10-50x |
| Query latency | 1ms | 5-20ms |
| MRR@10 | Baseline | +10-20% |
| Recall@100 | Baseline | +5-15% |

### Usage Example

```rust
use needle::multivec::{MultiVector, MultiVectorIndex, MultiVectorConfig};

// Create index
let config = MultiVectorConfig {
    dimension: 128,  // ColBERT dimension
    max_vectors_per_doc: 512,
    quantization: QuantizationType::Scalar8,
};
let mut index = MultiVectorIndex::new(config);

// Add document (requires ColBERT encoder output)
let doc_vectors: Vec<Vec<f32>> = colbert_encode(&document_text);
let doc = MultiVector::new("doc1", doc_vectors)
    .with_metadata(json!({"title": "Introduction to ML"}));
index.add(doc)?;

// Search (query also encoded with ColBERT)
let query_vectors: Vec<Vec<f32>> = colbert_encode(&query_text);
let results = index.search(&query_vectors, 10)?;

for result in results {
    println!("ID: {}, Score: {:.4}", result.id, result.score);
}
```

### Hybrid with Single-Vector

For cost/quality tradeoff:

```rust
// Stage 1: Fast single-vector retrieval (cheap)
let candidates = single_vector_index.search(&query_embedding, 100)?;

// Stage 2: Re-rank with ColBERT MaxSim (expensive but accurate)
let query_vectors = colbert_encode(&query_text);
let reranked = multi_vector_index.rerank(&query_vectors, &candidates, 10)?;
```
