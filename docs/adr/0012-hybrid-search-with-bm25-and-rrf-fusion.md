# ADR-0012: Hybrid Search with BM25 and RRF Fusion

## Status

Accepted

## Context

Vector similarity search excels at finding semantically similar content but has limitations:

| Query Type | Vector Search | Keyword Search |
|------------|---------------|----------------|
| "documents about machine learning" | ✓ Excellent | ✓ Good |
| "error code E1234" | ✗ Poor (no semantic meaning) | ✓ Excellent |
| "papers by Smith et al. 2023" | ✗ Poor (proper nouns) | ✓ Excellent |
| "happy vs joyful sentiment" | ✓ Excellent | ✗ Poor (different words) |

Production search systems often need **hybrid search** combining:
1. **Semantic search** (vector similarity) — Understands meaning
2. **Lexical search** (keyword matching) — Exact term matching

The challenge is combining results from two different ranking systems:
- Vector search returns distances (lower is better)
- BM25 returns scores (higher is better)
- Scores are not directly comparable

## Decision

Implement **hybrid search** via optional BM25 indexing with **Reciprocal Rank Fusion (RRF)** for result combination.

### BM25 Index

BM25 (Best Match 25) is the industry-standard text ranking algorithm:

```rust
#[cfg(feature = "hybrid")]
pub struct Bm25Index {
    /// Inverted index: term -> [(doc_id, term_frequency)]
    inverted_index: HashMap<String, Vec<(String, f32)>>,

    /// Document lengths for normalization
    doc_lengths: HashMap<String, usize>,

    /// Average document length
    avg_doc_length: f32,

    /// Total document count
    doc_count: usize,

    /// BM25 parameters
    k1: f32,  // Term frequency saturation (default: 1.2)
    b: f32,   // Length normalization (default: 0.75)
}

impl Bm25Index {
    pub fn index_document(&mut self, id: &str, text: &str) {
        let tokens = self.tokenize(text);
        // Update inverted index, doc lengths, etc.
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<(String, f32)> {
        let query_tokens = self.tokenize(query);
        // BM25 scoring with IDF weighting
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        // Lowercase, stem (via rust-stemmers), remove stopwords
    }
}
```

### Reciprocal Rank Fusion (RRF)

RRF combines ranked lists without requiring score normalization:

```rust
pub fn reciprocal_rank_fusion(
    vector_results: &[(String, f32)],
    bm25_results: &[(String, f32)],
    config: &RrfConfig,
    k: usize,
) -> Vec<HybridResult> {
    let mut scores: HashMap<String, f32> = HashMap::new();

    // RRF formula: score = Σ 1/(k + rank)
    // where k is a constant (default: 60)

    for (rank, (id, _distance)) in vector_results.iter().enumerate() {
        *scores.entry(id.clone()).or_default() +=
            config.vector_weight / (config.k + rank as f32 + 1.0);
    }

    for (rank, (id, _score)) in bm25_results.iter().enumerate() {
        *scores.entry(id.clone()).or_default() +=
            config.bm25_weight / (config.k + rank as f32 + 1.0);
    }

    // Sort by combined score and return top k
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results.truncate(k);

    results.into_iter().map(|(id, score)| HybridResult { id, score }).collect()
}
```

### RRF Configuration

```rust
pub struct RrfConfig {
    /// Ranking constant (default: 60)
    /// Higher values reduce the impact of high ranks
    pub k: f32,

    /// Weight for vector search results (default: 1.0)
    pub vector_weight: f32,

    /// Weight for BM25 results (default: 1.0)
    pub bm25_weight: f32,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            vector_weight: 1.0,
            bm25_weight: 1.0,
        }
    }
}
```

### Code References

- `src/hybrid.rs` — BM25Index and RRF implementation
- `Cargo.toml:137` — `hybrid = ["rust-stemmers"]` feature
- `src/lib.rs` — Conditional export of hybrid module
- BM25 uses rust-stemmers for Porter stemming

## Consequences

### Benefits

1. **Best of both worlds** — Semantic understanding + exact matching
2. **No score normalization needed** — RRF uses ranks, not scores
3. **Tunable weighting** — Adjust vector vs BM25 importance
4. **Industry-proven** — BM25 and RRF are well-established algorithms
5. **Optional feature** — No overhead for users who don't need it

### Tradeoffs

1. **Dual indexing** — Must maintain both HNSW and BM25 indices
2. **Increased memory** — BM25 inverted index adds storage overhead
3. **Two queries** — Hybrid search runs vector + BM25 queries
4. **Text preprocessing** — Requires tokenization, stemming decisions
5. **Feature dependency** — Adds rust-stemmers crate

### What This Enabled

- E-commerce search: "red nike shoes size 10" (semantic + exact)
- Document retrieval: Technical terms + conceptual queries
- RAG applications: Better context retrieval for LLMs
- Fallback behavior: BM25 catches what vectors miss

### What This Prevented

- Fully integrated scoring (would need learned fusion)
- Query-time BM25 parameter tuning (fixed at index time)
- Multi-language stemming (rust-stemmers is English-focused)
- Real-time BM25 index updates (requires reindexing for optimal performance)

### Usage Example

```rust
use needle::{Database, Bm25Index, reciprocal_rank_fusion, RrfConfig};

let db = Database::open("products.needle")?;

// Create BM25 index alongside vector collection
let mut bm25 = Bm25Index::default();

// Index text content
for product in products {
    // Add vector to collection
    db.insert("products", &product.id, product.embedding.clone())?;

    // Add text to BM25 index
    let text = format!("{} {}", product.title, product.description);
    bm25.index_document(&product.id, &text);
}

// Hybrid search
fn hybrid_search(
    db: &Database,
    bm25: &Bm25Index,
    query_text: &str,
    query_vector: &[f32],
    k: usize,
) -> Vec<HybridResult> {
    // Vector search
    let vector_results = db.search("products", query_vector, k * 2)?;

    // BM25 search
    let bm25_results = bm25.search(query_text, k * 2);

    // Fuse results
    let config = RrfConfig::default();
    reciprocal_rank_fusion(&vector_results, &bm25_results, &config, k)
}

// Search for "red nike running shoes"
let results = hybrid_search(
    &db,
    &bm25,
    "red nike running shoes",  // Text query for BM25
    &query_embedding,           // Vector for semantic search
    10,
)?;
```

### Why RRF Over Other Fusion Methods

| Method | Pros | Cons |
|--------|------|------|
| **RRF** | No calibration needed, robust | Ignores score magnitudes |
| Score normalization | Uses score information | Requires careful calibration |
| Learned fusion | Optimal for specific dataset | Needs training data |
| Simple interleaving | Easy to implement | No principled ranking |

RRF was chosen because:
1. Works out-of-the-box without training
2. Robust to score distribution differences
3. Well-studied in information retrieval literature
4. Tunable via weights and k parameter

### Performance Considerations

```
Hybrid Query Latency = max(Vector Search, BM25 Search) + RRF Fusion

Typical breakdown:
- Vector search (HNSW): 1-5ms
- BM25 search: 1-3ms
- RRF fusion: <0.1ms

Total: ~5-8ms (queries run in parallel)
```

For optimal performance:
- Run vector and BM25 searches in parallel
- Fetch more candidates (k*2) for better fusion
- Pre-filter with metadata before hybrid search if possible
