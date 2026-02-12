---
sidebar_position: 4
---

# Distance Functions

Distance functions measure how similar two vectors are. Choosing the right distance function is crucial for search quality. This page explains the options available in Needle.

## Available Distance Functions

Needle supports four distance functions:

```rust
use needle::{DistanceFunction, CollectionConfig};

// Choose when creating a collection
db.create_collection("docs", 384)?;  // Cosine distance (default)

let config = CollectionConfig::new("images", 512)
    .with_distance(DistanceFunction::Euclidean);
db.create_collection_with_config(config)?;

let config = CollectionConfig::new("recommendations", 256)
    .with_distance(DistanceFunction::DotProduct);
db.create_collection_with_config(config)?;

let config = CollectionConfig::new("locations", 2)
    .with_distance(DistanceFunction::Manhattan);
db.create_collection_with_config(config)?;
```

## Cosine Distance

**Best for:** Text embeddings, semantic similarity

Cosine distance measures the angle between two vectors, ignoring their magnitude. Two vectors pointing in the same direction have cosine distance 0, while opposite vectors have distance 2.

```
Cosine Distance = 1 - (A · B) / (|A| * |B|)
Range: [0, 2]
```

### When to Use Cosine

- **Text embeddings**: Most text embedding models (OpenAI, Sentence Transformers, Cohere) are designed for cosine similarity
- **When magnitude doesn't matter**: Comparing documents of different lengths
- **Normalized vectors**: When your embeddings are already normalized

```rust
// Cosine is the most common choice for text
db.create_collection("documents", 384)?;

// Needle normalizes vectors automatically for cosine distance
collection.insert("doc1", &embedding, Some(json!({})))?;
```

## Euclidean Distance

**Best for:** Image embeddings, spatial data, dense embeddings

Euclidean distance measures the straight-line distance between two points in space. It's sensitive to both direction and magnitude.

```
Euclidean Distance = sqrt(sum((A[i] - B[i])^2))
Range: [0, ∞)
```

### When to Use Euclidean

- **Image embeddings**: CLIP, ImageBind, and similar models often work well with Euclidean
- **Spatial data**: Geographic coordinates, physical measurements
- **When magnitude matters**: Comparing vectors where scale is meaningful

```rust
// Good for image search
let config = CollectionConfig::new("images", 512)
    .with_distance(DistanceFunction::Euclidean);
db.create_collection_with_config(config)?;

// Also good for geographic data
let config = CollectionConfig::new("locations", 2)
    .with_distance(DistanceFunction::Euclidean);
db.create_collection_with_config(config)?;
```

## Dot Product Distance

**Best for:** Recommendation systems, retrieval with learned scores

Dot product measures similarity by multiplying corresponding elements and summing. Higher values indicate more similarity. Needle converts this to a distance (negative dot product).

```
Dot Product Distance = -(A · B)
Range: (-∞, ∞)
```

### When to Use Dot Product

- **Recommendation systems**: When vectors encode both direction and confidence
- **Learned retrieval**: Models trained with dot product similarity
- **Maximum inner product search (MIPS)**: When you need actual similarity scores

```rust
// Recommendation systems often use dot product
let config = CollectionConfig::new("user_item", 128)
    .with_distance(DistanceFunction::DotProduct);
db.create_collection_with_config(config)?;

// The vectors encode both preference direction and strength
let user_vector = model.encode_user(&user)?;
let item_vector = model.encode_item(&item)?;
// Higher dot product = stronger recommendation
```

### Important: Normalization

For normalized vectors, cosine and dot product are equivalent:
```rust
// If vectors are normalized:
// Cosine Distance = 1 - Dot Product

// So these produce the same ranking:
let cosine_results = cosine_collection.search(&query, 10)?;
let dot_results = dot_collection.search(&query, 10)?;
```

## Manhattan Distance

**Best for:** Sparse data, high-dimensional spaces

Manhattan distance (L1 norm) measures the sum of absolute differences. It's less sensitive to outliers than Euclidean distance.

```
Manhattan Distance = sum(|A[i] - B[i]|)
Range: [0, ∞)
```

### When to Use Manhattan

- **Sparse vectors**: When most values are zero
- **High dimensions**: More robust in high-dimensional spaces
- **Outlier resistance**: Less affected by extreme values

```rust
// Good for sparse feature vectors
let config = CollectionConfig::new("features", 1000)
    .with_distance(DistanceFunction::Manhattan);
db.create_collection_with_config(config)?;
```

## Choosing the Right Distance Function

### Decision Guide

```
Is your data text embeddings?
├── Yes → Use Cosine
└── No
    ├── Are vectors normalized?
    │   ├── Yes → Cosine or Dot (equivalent)
    │   └── No
    │       ├── Is magnitude meaningful?
    │       │   ├── Yes → Euclidean or Dot
    │       │   └── No → Cosine
    └── Is data sparse or high-dimensional?
        ├── Yes → Consider Manhattan
        └── No → Euclidean
```

### Common Use Cases

| Use Case | Recommended | Alternative |
|----------|-------------|-------------|
| Text search (OpenAI, Cohere) | Cosine | - |
| Text search (Sentence Transformers) | Cosine | - |
| Image search (CLIP) | Cosine or Euclidean | Dot |
| Recommendations | Dot | Cosine |
| Geographic data | Euclidean | Manhattan |
| Sparse features | Manhattan | Euclidean |
| Face recognition | Cosine | Euclidean |

## SIMD Optimization

Needle can use SIMD (Single Instruction, Multiple Data) for faster distance computations:

```rust
// Enable SIMD in Cargo.toml
[dependencies]
needle = { version = "0.1", features = ["simd"] }
```

SIMD provides 2-8x speedup for distance calculations, especially beneficial for:
- Large-scale search
- High-dimensional vectors
- Batch operations

## Performance Comparison

Typical performance on 1M vectors, 384 dimensions:

| Distance | Without SIMD | With SIMD | Notes |
|----------|--------------|-----------|-------|
| Cosine | 5ms | 1.5ms | Requires normalization |
| Euclidean | 4ms | 1.2ms | No normalization needed |
| Dot | 3ms | 0.8ms | Fastest computation |
| Manhattan | 6ms | 2ms | Slowest (no fused ops) |

## Mixing Distance Functions

Different collections can use different distance functions:

```rust
// Text collection with cosine
db.create_collection("text_docs", 384)?;

// Image collection with euclidean
let config = CollectionConfig::new("images", 512)
    .with_distance(DistanceFunction::Euclidean);
db.create_collection_with_config(config)?;

// Recommendations with dot product
let config = CollectionConfig::new("recommendations", 128)
    .with_distance(DistanceFunction::DotProduct);
db.create_collection_with_config(config)?;
```

## Custom Distance Functions

For advanced use cases, you can implement custom distance functions by working with the raw vectors:

```rust
fn weighted_euclidean(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .zip(weights.iter())
        .map(|((x, y), w)| w * (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// Use with brute-force search
let query = collection.get("query_id")?.vector;
let mut results: Vec<_> = collection
    .iter_vectors()?
    .map(|(id, vec)| (id, weighted_euclidean(&query, &vec, &weights)))
    .collect();
results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
```

## Next Steps

- [Metadata Filtering](/docs/concepts/metadata-filtering) - Filter results by attributes
- [Hybrid Search](/docs/guides/hybrid-search) - Combine vector and text search
- [HNSW Index](/docs/concepts/hnsw-index) - Tune search parameters
