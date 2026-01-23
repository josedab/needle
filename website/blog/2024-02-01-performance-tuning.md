---
slug: performance-tuning-guide
title: "Performance Tuning: Getting the Most Out of Needle"
authors: [needle-team]
tags: [performance, tutorial, vector-search]
---

Learn how to optimize Needle for your specific workload, whether you need maximum speed, highest recall, or minimum memory usage.

<!-- truncate -->

## Understanding the Trade-offs

Vector search always involves trade-offs between three factors:

1. **Search speed** - How quickly can we return results?
2. **Recall quality** - How many of the true nearest neighbors do we find?
3. **Memory usage** - How much RAM does the index consume?

Needle gives you control over all three through HNSW parameters and quantization options.

## HNSW Parameters

### M (Max Connections)

The `M` parameter controls how many connections each node has in the HNSW graph:

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(24);  // Default is 16
```

| M Value | Recall | Memory | Build Time |
|---------|--------|--------|------------|
| 8 | Lower | Less | Faster |
| 16 | Good | Moderate | Moderate |
| 32 | Higher | More | Slower |

**Recommendation**: Start with 16, increase to 24-32 if you need higher recall.

### ef_construction

This parameter controls search depth during index building:

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_ef_construction(400);  // Default is 200
```

Higher values create a better-quality index but take longer to build. Rule of thumb: `ef_construction >= 2 * M`.

### ef_search

This is your query-time knob for the recall/speed trade-off:

```rust
// Faster, lower recall
let results = collection.search_with_params(&query, 10, None, 30)?;

// Slower, higher recall
let results = collection.search_with_params(&query, 10, None, 200)?;
```

You can adjust this per-query based on requirements.

## Quantization for Memory Efficiency

For large datasets, quantization dramatically reduces memory usage:

### Scalar Quantization (4x reduction)

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Scalar);
```

Converts 32-bit floats to 8-bit integers. Minimal recall impact.

### Product Quantization (8-32x reduction)

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Product {
        num_subvectors: 48,
        num_centroids: 256,
    });
```

Best for very large datasets where memory is critical.

### Binary Quantization (32x reduction)

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Binary);
```

Most aggressive compression. Use for initial candidate retrieval with re-ranking.

## Auto-Tuning

Let Needle find optimal parameters for your constraints:

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::Balanced)
    .with_memory_budget(4 * 1024 * 1024 * 1024)  // 4GB
    .with_latency_target_ms(5)
    .with_recall_target(0.95);

let result = auto_tune(&constraints);
println!("Recommended M: {}", result.config.hnsw_m);
println!("Recommended ef_construction: {}", result.config.ef_construction);
```

## Benchmarking Your Setup

Use `search_explain` to understand search performance:

```rust
let (results, explain) = collection.search_explain(&query, 10, None)?;

println!("Nodes visited: {}", explain.nodes_visited);
println!("Distance computations: {}", explain.distance_computations);
println!("Index time: {:?}", explain.index_time);
println!("Filter time: {:?}", explain.filter_time);
println!("Total time: {:?}", explain.total_time);
```

## Quick Reference

| Use Case | M | ef_construction | ef_search | Quantization |
|----------|---|-----------------|-----------|--------------|
| Low latency | 12 | 150 | 30 | Scalar |
| Balanced | 16 | 200 | 50 | None |
| High recall | 24 | 400 | 150 | None |
| Large scale | 16 | 200 | 100 | Product |

Check out our [HNSW Tuning Guide](/docs/configuration/hnsw-tuning) for more details.
