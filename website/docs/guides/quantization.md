---
sidebar_position: 4
---

# Quantization Guide

Quantization reduces memory usage by representing vectors with fewer bits. This guide explains Needle's quantization options and how to choose the right one.

## Why Quantization?

Without quantization, each vector dimension uses 4 bytes (32-bit float):

| Vectors | Dimensions | Memory (Full Precision) |
|---------|------------|-------------------------|
| 100K | 384 | 150 MB |
| 1M | 384 | 1.5 GB |
| 10M | 384 | 15 GB |
| 100M | 384 | 150 GB |

Quantization can reduce this by 4-32x while maintaining good search quality.

## Quantization Types

Needle supports three quantization methods:

| Type | Compression | Recall Impact | Best For |
|------|-------------|---------------|----------|
| Scalar | 4x | &lt;1% loss | General use |
| Product | 8-32x | 2-5% loss | Large scale |
| Binary | 32x | 5-10% loss | Massive scale |

## Scalar Quantization

Converts 32-bit floats to 8-bit integers. Simple and effective.

```rust
use needle::{CollectionConfig, DistanceFunction, QuantizationType};

let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Scalar);

db.create_collection_with_config("documents", config)?;
```

### How It Works

1. Find min/max values across all vectors
2. Map each float to an integer in [0, 255]
3. Store integers instead of floats

```
Original: [0.1, 0.5, -0.3, 0.8]
Quantized: [89, 153, 25, 204]  // 4 bytes instead of 16
```

### When to Use

- **General purpose**: Good default for most applications
- **Memory reduction**: Need ~4x compression
- **Quality sensitive**: Can't afford significant recall loss

## Product Quantization (PQ)

Divides vectors into subvectors and quantizes each separately using codebooks.

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Product {
        num_subvectors: 48,      // 384 / 48 = 8 dims per subvector
        num_centroids: 256,       // 8-bit codes
    });

db.create_collection_with_config("documents", config)?;
```

### How It Works

1. Split vector into `num_subvectors` parts
2. Learn `num_centroids` cluster centers for each part
3. Represent each subvector by its nearest centroid ID

```
Original: [0.1, 0.2, ..., 0.3, 0.4]  // 384 floats = 1536 bytes
          └─── subvec 1 ───┘ ... └─── subvec 48 ───┘

Quantized: [23, 156, ..., 89, 201]    // 48 bytes (one per subvector)
```

### Configuration

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `num_subvectors` | Parts to divide into | Dimension / 8 (e.g., 48 for 384-dim) |
| `num_centroids` | Codes per subvector | 256 (8-bit) or 65536 (16-bit) |

```rust
// 8x compression (48 bytes for 384-dim vector)
QuantizationType::Product {
    num_subvectors: 48,
    num_centroids: 256,
}

// 16x compression (24 bytes for 384-dim vector)
QuantizationType::Product {
    num_subvectors: 24,
    num_centroids: 256,
}

// Higher quality with 16-bit codes
QuantizationType::Product {
    num_subvectors: 48,
    num_centroids: 65536,  // 16-bit
}
```

### When to Use

- **Large scale**: Millions to billions of vectors
- **Memory constrained**: Need 8-32x compression
- **Can train**: Have representative data for codebook learning

## Binary Quantization

Converts each dimension to a single bit based on sign.

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Binary);

db.create_collection_with_config("documents", config)?;
```

### How It Works

1. Set bit to 1 if value > 0, else 0
2. Use Hamming distance for comparison
3. Optionally rerank with original vectors

```
Original: [0.1, -0.5, 0.3, -0.8, ...]
Binary:   [1,    0,   1,    0,   ...]  // 384 bits = 48 bytes
```

### When to Use

- **Massive scale**: Billions of vectors
- **Speed critical**: Need fastest possible search
- **Reranking available**: Can afford two-stage retrieval

### Two-Stage Search with Binary

```rust
// Stage 1: Fast binary search for candidates
let candidates = collection.search_binary(&query, 1000)?;

// Stage 2: Rerank with full precision
let reranked = collection.rerank_full(&query, &candidates, 10)?;
```

## Choosing Quantization

```
How many vectors?
├── < 1M → Scalar (simple, good quality)
├── 1M-100M → Product (balanced)
└── > 100M → Binary (maximum compression)

Memory budget?
├── Generous → Scalar or none
├── Moderate → Product
└── Tight → Binary

Quality requirements?
├── High (>98% recall) → Scalar
├── Moderate (95%+) → Product
└── Acceptable (90%+) → Binary with reranking
```

## Quantization Workflow

### Training Product Quantization

PQ requires training on representative data:

```rust
use needle::quantization::ProductQuantizer;

// Sample training vectors (10-100K recommended)
let training_vectors: Vec<Vec<f32>> = sample_vectors(&all_vectors, 50000);

// Train codebooks
let pq = ProductQuantizer::train(
    &training_vectors,
    48,    // num_subvectors
    256,   // num_centroids
)?;

// Create collection with trained PQ
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_product_quantizer(pq);

db.create_collection_with_config("documents", config)?;
```

### Migration: Full Precision to Quantized

```rust
fn migrate_to_quantized(
    source_db: &Database,
    target_db: &Database,
    source_collection: &str,
    target_collection: &str,
) -> needle::Result<()> {
    let source = source_db.collection(source_collection)?;
    let target = target_db.collection(target_collection)?;

    // Export from source
    for (id, vector, metadata) in source.iter()? {
        target.insert(&id, &vector, metadata)?;
    }

    target_db.save()?;
    Ok(())
}

// Usage
let source_db = Database::open("original.needle")?;
let target_db = Database::open("quantized.needle")?;

// Create quantized collection
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Scalar);
target_db.create_collection_with_config("documents", config)?;

migrate_to_quantized(&source_db, &target_db, "documents", "documents")?;
```

## Performance Comparison

Benchmarks on 1M vectors, 384 dimensions:

| Quantization | Memory | Search Latency | Recall@10 |
|--------------|--------|----------------|-----------|
| None | 1.5 GB | 5 ms | 99.5% |
| Scalar | 384 MB | 4 ms | 98.8% |
| PQ (48 sub) | 96 MB | 6 ms | 96.2% |
| PQ (24 sub) | 48 MB | 5 ms | 93.5% |
| Binary | 48 MB | 2 ms | 88.1% |
| Binary + rerank | 48 MB | 8 ms | 97.3% |

## Best Practices

### 1. Benchmark on Your Data

Quantization impact varies by data distribution:

```rust
fn benchmark_quantization(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<String>],
) {
    let quantization_types = [
        QuantizationType::None,
        QuantizationType::Scalar,
        QuantizationType::Product { num_subvectors: 48, num_centroids: 256 },
        QuantizationType::Binary,
    ];

    for qt in &quantization_types {
        let db = create_test_db(vectors, qt);
        let (recall, latency) = evaluate(&db, queries, ground_truth);
        println!("{:?}: recall={:.2}%, latency={:?}", qt, recall * 100.0, latency);
    }
}
```

### 2. Consider Hybrid Approaches

Use binary for initial filtering, full precision for final ranking:

```rust
// Fast initial search with binary
let candidates = binary_collection.search(&query_binary, 500)?;

// Accurate reranking with full precision
let candidate_ids: Vec<_> = candidates.iter().map(|r| &r.id).collect();
let final_results = full_collection.search_subset(&query, &candidate_ids, 10)?;
```

### 3. Monitor Quality

Track recall in production:

```rust
// Periodically sample and check
fn monitor_recall(
    quantized: &Collection,
    full_precision: &Collection,
    sample_queries: &[Vec<f32>],
) -> f32 {
    let mut total_recall = 0.0;

    for query in sample_queries {
        let quantized_results = quantized.search(query, 10, None).unwrap();
        let full_results = full_precision.search(query, 10, None).unwrap();

        total_recall += compute_recall(&quantized_results, &full_results);
    }

    total_recall / sample_queries.len() as f32
}
```

### 4. Tuning HNSW with Quantization

Quantized vectors may need different HNSW parameters:

```rust
// Compensate for quantization loss with higher ef_search
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Scalar)
    .with_hnsw_m(24)              // Slightly higher M
    .with_hnsw_ef_construction(300);  // Higher ef_construction

// Or increase ef_search at query time
let results = collection.search_with_params(&query, 10, None, 100)?;  // ef_search=100
```

## Next Steps

- [Production Deployment](/docs/guides/production) - Scale with quantization
- [HNSW Index](/docs/concepts/hnsw-index) - Tune search parameters
- [API Reference](/docs/api-reference) - Complete quantization API
