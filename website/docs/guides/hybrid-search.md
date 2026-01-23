---
sidebar_position: 3
---

# Hybrid Search

Hybrid search combines vector similarity search with traditional keyword search to get the best of both worlds. This guide shows how to implement hybrid search with Needle.

## Why Hybrid Search?

| Aspect | Vector Search | Keyword Search | Hybrid |
|--------|---------------|----------------|--------|
| Semantic understanding | Excellent | Poor | Excellent |
| Exact match | Poor | Excellent | Excellent |
| Rare terms | Poor | Excellent | Excellent |
| Synonyms | Excellent | Poor | Excellent |
| New vocabulary | Good | Good | Excellent |

Example queries where hybrid excels:
- **"Python decorator @lru_cache"** - Need exact match for `@lru_cache`
- **"machine learning tutorial"** - Need semantic understanding
- **"error code E0382"** - Need exact match for error code

## Architecture

```
          ┌─────────────────┐
          │      Query      │
          └────────┬────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐     ┌───────────────┐
│ Vector Search │     │  BM25 Search  │
│   (Needle)    │     │   (Needle)    │
└───────┬───────┘     └───────┬───────┘
        │                     │
        └──────────┬──────────┘
                   ▼
          ┌───────────────┐
          │  RRF Fusion   │
          └───────┬───────┘
                  │
                  ▼
          ┌───────────────┐
          │    Results    │
          └───────────────┘
```

## Setting Up Hybrid Search

### Enable the Feature

```toml
# Cargo.toml
[dependencies]
needle = { version = "0.1", features = ["hybrid"] }
```

### Create a Collection with BM25

```rust
use needle::{Database, DistanceFunction, Bm25Index};

fn setup_hybrid_search() -> needle::Result<(Database, Bm25Index)> {
    let db = Database::open("hybrid.needle")?;

    // Create vector collection
    db.create_collection("documents", 384, DistanceFunction::Cosine)?;

    // Create BM25 index
    let bm25 = Bm25Index::default();

    Ok((db, bm25))
}
```

### Index Documents

```rust
use needle::EmbeddingModel;
use serde_json::json;

fn index_document(
    db: &Database,
    bm25: &mut Bm25Index,
    model: &EmbeddingModel,
    id: &str,
    title: &str,
    content: &str,
) -> needle::Result<()> {
    let collection = db.collection("documents")?;

    // Create text for embedding and indexing
    let full_text = format!("{} {}", title, content);

    // Vector index
    let embedding = model.encode(&full_text)?;
    collection.insert(id, &embedding, json!({
        "title": title,
        "content": content,
    }))?;

    // BM25 index
    bm25.index_document(id, &full_text);

    Ok(())
}
```

## Performing Hybrid Search

### Basic Hybrid Search

```rust
use needle::{reciprocal_rank_fusion, RrfConfig};

fn hybrid_search(
    db: &Database,
    bm25: &Bm25Index,
    model: &EmbeddingModel,
    query: &str,
    k: usize,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;

    // Vector search
    let query_embedding = model.encode(query)?;
    let vector_results = collection.search(&query_embedding, k * 2, None)?;

    // BM25 search
    let bm25_results = bm25.search(query, k * 2);

    // Fuse with RRF
    let config = RrfConfig::default();
    let fused = reciprocal_rank_fusion(&vector_results, &bm25_results, &config, k);

    Ok(fused)
}
```

### Configuring RRF

Reciprocal Rank Fusion (RRF) combines rankings from multiple sources:

```
RRF_score(d) = Σ 1 / (k + rank(d))
```

Where `k` is a constant (default 60) that determines how much weight to give to top-ranked results.

```rust
// Customize RRF parameters
let config = RrfConfig {
    k: 60,                    // Ranking constant
    vector_weight: 0.7,       // Weight for vector search
    keyword_weight: 0.3,      // Weight for BM25 search
};

let fused = reciprocal_rank_fusion(&vector_results, &bm25_results, &config, k);
```

### Weighted Combination

For more control, use weighted score combination:

```rust
fn weighted_hybrid_search(
    db: &Database,
    bm25: &Bm25Index,
    model: &EmbeddingModel,
    query: &str,
    k: usize,
    vector_weight: f32,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;

    // Get results
    let query_embedding = model.encode(query)?;
    let vector_results = collection.search(&query_embedding, k * 3, None)?;
    let bm25_results = bm25.search(query, k * 3);

    // Normalize scores
    let vector_scores = normalize_scores(&vector_results);
    let bm25_scores = normalize_scores(&bm25_results);

    // Combine scores
    let mut combined: HashMap<String, f32> = HashMap::new();

    for (id, score) in vector_scores {
        *combined.entry(id).or_insert(0.0) += vector_weight * score;
    }

    for (id, score) in bm25_scores {
        *combined.entry(id).or_insert(0.0) += (1.0 - vector_weight) * score;
    }

    // Sort by combined score
    let mut results: Vec<_> = combined.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Return top k
    Ok(results.into_iter().take(k).map(|(id, score)| {
        SearchResult { id, score, ..Default::default() }
    }).collect())
}

fn normalize_scores(results: &[SearchResult]) -> Vec<(String, f32)> {
    if results.is_empty() {
        return vec![];
    }

    let max_score = results.iter().map(|r| r.score).fold(f32::MIN, f32::max);
    let min_score = results.iter().map(|r| r.score).fold(f32::MAX, f32::min);
    let range = max_score - min_score;

    results.iter().map(|r| {
        let normalized = if range > 0.0 {
            (r.score - min_score) / range
        } else {
            1.0
        };
        (r.id.clone(), normalized)
    }).collect()
}
```

## BM25 Configuration

### Tokenization

BM25 uses configurable tokenization:

```rust
use needle::{Bm25Index, Bm25Config, Tokenizer};

// Default tokenizer (whitespace + lowercase)
let bm25 = Bm25Index::default();

// Custom tokenizer with stemming
let config = Bm25Config {
    tokenizer: Tokenizer::Stemming("english"),
    k1: 1.2,  // Term frequency saturation
    b: 0.75,  // Length normalization
};
let bm25 = Bm25Index::with_config(config);
```

### BM25 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k1` | 1.2 | Term frequency saturation (higher = more weight to repeated terms) |
| `b` | 0.75 | Length normalization (0 = no normalization, 1 = full normalization) |

```rust
// For short documents (tweets, titles)
let config = Bm25Config {
    k1: 1.2,
    b: 0.3,  // Less length normalization
    ..Default::default()
};

// For long documents (articles, papers)
let config = Bm25Config {
    k1: 1.5,
    b: 0.75,  // Standard length normalization
    ..Default::default()
};
```

## Advanced Techniques

### Query-Dependent Weighting

Adjust weights based on query characteristics:

```rust
fn adaptive_hybrid_search(
    db: &Database,
    bm25: &Bm25Index,
    model: &EmbeddingModel,
    query: &str,
    k: usize,
) -> needle::Result<Vec<SearchResult>> {
    // Analyze query
    let has_exact_terms = query.contains('"') || query.contains(":");
    let is_short = query.split_whitespace().count() <= 3;
    let has_technical_terms = query.chars().any(|c| c.is_ascii_digit());

    // Adjust weights based on query
    let vector_weight = if has_exact_terms || has_technical_terms {
        0.3  // Favor keyword search
    } else if is_short {
        0.6  // Balanced
    } else {
        0.7  // Favor semantic search
    };

    weighted_hybrid_search(db, bm25, model, query, k, vector_weight)
}
```

### Filtering in Hybrid Search

Apply metadata filters to hybrid results:

```rust
use needle::Filter;

fn filtered_hybrid_search(
    db: &Database,
    bm25: &Bm25Index,
    model: &EmbeddingModel,
    query: &str,
    filter: &Filter,
    k: usize,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;

    // Vector search with filter
    let query_embedding = model.encode(query)?;
    let vector_results = collection.search(&query_embedding, k * 2, Some(filter))?;

    // BM25 search (no filter - apply post-hoc)
    let bm25_results = bm25.search(query, k * 4);

    // Filter BM25 results
    let filtered_bm25: Vec<_> = bm25_results
        .into_iter()
        .filter(|r| {
            let metadata = collection.get(&r.id).ok()
                .map(|v| v.metadata.clone())
                .unwrap_or_default();
            filter.matches(&metadata)
        })
        .take(k * 2)
        .collect();

    // Fuse
    let config = RrfConfig::default();
    reciprocal_rank_fusion(&vector_results, &filtered_bm25, &config, k)
}
```

### Boosting Specific Fields

Boost matches in important fields:

```rust
fn field_boosted_hybrid(
    db: &Database,
    title_bm25: &Bm25Index,  // Index of titles only
    content_bm25: &Bm25Index, // Index of content only
    model: &EmbeddingModel,
    query: &str,
    k: usize,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;

    // Vector search
    let query_embedding = model.encode(query)?;
    let vector_results = collection.search(&query_embedding, k * 2, None)?;

    // BM25 on titles (boosted)
    let title_results = title_bm25.search(query, k * 2);

    // BM25 on content
    let content_results = content_bm25.search(query, k * 2);

    // Combine with field-specific weights
    let config = RrfConfig {
        vector_weight: 0.5,
        keyword_weight: 0.5,
        ..Default::default()
    };

    // First fuse title and content BM25 with title boost
    let keyword_combined = weighted_combine(
        &title_results,
        &content_results,
        1.5,  // Title boost
        1.0,
    );

    // Then fuse with vector
    reciprocal_rank_fusion(&vector_results, &keyword_combined, &config, k)
}
```

## Performance Optimization

### Parallel Search

```rust
use rayon::prelude::*;

fn parallel_hybrid_search(
    db: &Database,
    bm25: &Bm25Index,
    model: &EmbeddingModel,
    query: &str,
    k: usize,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;
    let query_embedding = model.encode(query)?;

    // Run searches in parallel
    let (vector_results, bm25_results) = rayon::join(
        || collection.search(&query_embedding, k * 2, None).unwrap(),
        || bm25.search(query, k * 2),
    );

    let config = RrfConfig::default();
    Ok(reciprocal_rank_fusion(&vector_results, &bm25_results, &config, k))
}
```

### Caching BM25 Index

```rust
use std::sync::Arc;

struct HybridSearcher {
    db: Database,
    bm25: Arc<Bm25Index>,
    model: EmbeddingModel,
}

impl HybridSearcher {
    fn new(db_path: &str) -> needle::Result<Self> {
        let db = Database::open(db_path)?;
        let bm25 = Arc::new(Self::build_bm25_from_db(&db)?);
        let model = EmbeddingModel::load("all-MiniLM-L6-v2")?;

        Ok(Self { db, bm25, model })
    }

    fn build_bm25_from_db(db: &Database) -> needle::Result<Bm25Index> {
        let mut bm25 = Bm25Index::default();
        let collection = db.collection("documents")?;

        for (id, metadata) in collection.iter_metadata()? {
            let text = format!(
                "{} {}",
                metadata["title"].as_str().unwrap_or(""),
                metadata["content"].as_str().unwrap_or("")
            );
            bm25.index_document(&id, &text);
        }

        Ok(bm25)
    }
}
```

## Evaluation

Compare hybrid vs. pure vector search:

```rust
fn evaluate_search_methods(
    db: &Database,
    bm25: &Bm25Index,
    model: &EmbeddingModel,
    test_queries: &[(String, Vec<String>)],  // (query, relevant_ids)
) -> (f32, f32, f32) {
    let mut vector_recall = 0.0;
    let mut bm25_recall = 0.0;
    let mut hybrid_recall = 0.0;

    for (query, relevant_ids) in test_queries {
        let collection = db.collection("documents").unwrap();
        let k = 10;

        // Vector search
        let query_embedding = model.encode(query).unwrap();
        let vector_results = collection.search(&query_embedding, k, None).unwrap();
        vector_recall += compute_recall(&vector_results, relevant_ids);

        // BM25 search
        let bm25_results = bm25.search(query, k);
        bm25_recall += compute_recall(&bm25_results, relevant_ids);

        // Hybrid search
        let hybrid_results = reciprocal_rank_fusion(
            &vector_results,
            &bm25_results,
            &RrfConfig::default(),
            k,
        );
        hybrid_recall += compute_recall(&hybrid_results, relevant_ids);
    }

    let n = test_queries.len() as f32;
    (vector_recall / n, bm25_recall / n, hybrid_recall / n)
}
```

## Next Steps

- [Quantization](/docs/guides/quantization) - Reduce memory usage
- [Production Deployment](/docs/guides/production) - Scale your search
- [API Reference](/docs/api-reference) - Complete API documentation
