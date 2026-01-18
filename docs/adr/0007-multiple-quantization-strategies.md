# ADR-0007: Multiple Quantization Strategies

## Status

Accepted

## Context

High-dimensional vectors consume significant memory:

| Dimensions | Vectors | Float32 Size | With HNSW Overhead |
|------------|---------|--------------|-------------------|
| 384 | 1M | 1.5 GB | ~2 GB |
| 768 | 1M | 3 GB | ~4 GB |
| 1536 | 1M | 6 GB | ~8 GB |

For large-scale deployments, this memory footprint limits:
1. **Dataset size** — RAM constrains how many vectors fit in memory
2. **Query throughput** — Memory bandwidth becomes the bottleneck
3. **Cost** — High-memory instances are expensive

**Quantization** compresses vectors by reducing precision, trading accuracy for efficiency:

| Method | Compression | Recall Impact | Use Case |
|--------|-------------|---------------|----------|
| Scalar (SQ8) | 4x | <1% drop | General purpose |
| Product (PQ) | 8-16x | 2-5% drop | Large scale |
| Binary | 32x | 5-15% drop | Extreme scale |

## Decision

Provide **three quantization strategies** as optional, user-selected compression methods with training-based calibration.

### Quantization Types

**1. Scalar Quantization (ScalarQuantizer)**

Converts float32 to uint8 by mapping the value range to [0, 255]:

```rust
pub struct ScalarQuantizer {
    min_vals: Vec<f32>,  // Per-dimension minimum
    max_vals: Vec<f32>,  // Per-dimension maximum
}

impl ScalarQuantizer {
    pub fn train(vectors: &[Vec<f32>]) -> Self {
        // Compute min/max per dimension from training data
    }

    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter().enumerate().map(|(i, &v)| {
            let normalized = (v - self.min_vals[i]) / (self.max_vals[i] - self.min_vals[i]);
            (normalized * 255.0).clamp(0.0, 255.0) as u8
        }).collect()
    }

    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        // Reverse the mapping
    }
}
```

**2. Product Quantization (ProductQuantizer)**

Splits vectors into subvectors and quantizes each to a codebook index:

```rust
pub struct ProductQuantizer {
    num_subvectors: usize,     // e.g., 8
    bits_per_subvector: usize, // e.g., 8 (256 centroids)
    codebooks: Vec<Vec<Vec<f32>>>,  // [subvector][centroid][values]
}

impl ProductQuantizer {
    pub fn train(vectors: &[Vec<f32>], num_subvectors: usize) -> Self {
        // K-means clustering per subvector
    }

    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        // Find nearest centroid per subvector
    }
}
```

**3. Binary Quantization (BinaryQuantizer)**

Reduces each dimension to a single bit (positive/negative):

```rust
pub struct BinaryQuantizer {
    thresholds: Vec<f32>,  // Per-dimension threshold (usually 0 or mean)
}

impl BinaryQuantizer {
    pub fn quantize(&self, vector: &[f32]) -> BitVec {
        vector.iter().enumerate().map(|(i, &v)| {
            v > self.thresholds[i]
        }).collect()
    }
}
```

### Training Requirement

All quantizers require **training on representative data** before use:

```rust
// Collect representative vectors
let training_data: Vec<Vec<f32>> = collection
    .sample_vectors(10_000)?;

// Train quantizer
let quantizer = ScalarQuantizer::train(&training_data);

// Apply to collection
collection.enable_quantization(quantizer)?;
```

This ensures quantization parameters (min/max, codebooks, thresholds) match the actual data distribution.

### Code References

- `src/quantization.rs:91-186` — Quantizer implementations
- `src/lib.rs:168` — Public exports
- Training methods at `quantization.rs:129-186`

## Consequences

### Benefits

1. **User choice** — Different domains need different tradeoffs
2. **Significant compression** — 4x to 32x memory reduction
3. **Training-based accuracy** — Calibration on real data maximizes quality
4. **Composable with search** — Quantized distance computation available
5. **Optional optimization** — Collections work without quantization

### Tradeoffs

1. **Lossy compression** — Recall decreases with higher compression
2. **Training overhead** — Requires representative sample before use
3. **Not automatic** — User must explicitly enable and configure
4. **Storage complexity** — Quantized and full vectors may coexist

### Compression vs Accuracy Tradeoffs

| Quantizer | Compression | Typical Recall@10 | Best For |
|-----------|-------------|-------------------|----------|
| None | 1x | 100% (baseline) | Small datasets, exact needs |
| Scalar | 4x | 98-99% | General production use |
| Product | 8-16x | 95-98% | Large scale, cost-sensitive |
| Binary | 32x | 85-95% | Extreme scale, coarse filtering |

### What This Enabled

- Million-vector datasets on commodity hardware
- Two-stage retrieval: binary filter → full precision rerank
- Memory-bandwidth optimization for high-QPS workloads
- Cost reduction for cloud deployments

### What This Prevented

- Automatic quantization (requires user decision and training)
- Learned quantization via neural networks (would need ONNX dependency)
- On-the-fly quantization (training must precede use)

### Usage Example

```rust
use needle::{Database, ScalarQuantizer, ProductQuantizer};

let db = Database::open("large_dataset.needle")?;
let collection = db.collection("embeddings")?;

// Sample training data
let training_vectors = collection.sample_vectors(10_000)?;

// Train and apply scalar quantization
let sq = ScalarQuantizer::train(&training_vectors);
collection.set_quantizer(Box::new(sq))?;

// Search uses quantized distances automatically
let results = collection.search(&query, 10)?;

// For higher accuracy, use two-stage retrieval
let candidates = collection.search_quantized(&query, 100)?;
let refined = collection.rerank_exact(&query, &candidates, 10)?;
```

### Design Philosophy

Quantization in Needle follows these principles:

1. **Explicit over implicit** — Users opt-in, understanding the tradeoff
2. **Training required** — No generic quantization that ignores data distribution
3. **Composable** — Works with existing search, filtering, and indexing
4. **Non-destructive** — Original vectors preserved, quantization is an overlay
