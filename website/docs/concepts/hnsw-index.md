---
sidebar_position: 3
---

# HNSW Index

Needle uses the **Hierarchical Navigable Small World (HNSW)** algorithm for approximate nearest neighbor search. This page explains how HNSW works and how to tune it for your use case.

## How HNSW Works

HNSW builds a multi-layer graph where:
- **Higher layers** have fewer nodes and longer-range connections (for fast navigation)
- **Lower layers** have more nodes and shorter-range connections (for precise search)

### Search Process

1. Start at the top layer's entry point
2. Greedily navigate to the nearest node at each layer
3. Move down to the next layer
4. Repeat until reaching the bottom layer
5. Return the k nearest neighbors found

This achieves O(log n) search complexity on average, compared to O(n) for brute force.

```
Layer 2:  [A] -------- [B]          (few nodes, long connections)
           |            |
Layer 1:  [A] -- [C] -- [B] -- [D]  (more nodes, medium connections)
           |     |      |      |
Layer 0:  [A]-[E]-[C]-[F]-[B]-[G]-[D]  (all nodes, short connections)
```

## HNSW Parameters

### M (Max Connections)

The maximum number of connections per node in the graph.

```rust
use needle::CollectionConfig;

// Default: M = 16
let config = CollectionConfig::new("collection", 384)
    .with_distance(DistanceFunction::Cosine)
    .with_hnsw_m(32);  // Higher M for better recall
```

| M Value | Recall | Memory | Build Time |
|---------|--------|--------|------------|
| 8 | Lower | Lower | Faster |
| 16 | Balanced | Balanced | Balanced |
| 32 | Higher | Higher | Slower |
| 64 | Highest | Highest | Slowest |

**Guidelines:**
- Use M=8-12 for memory-constrained environments
- Use M=16 (default) for most applications
- Use M=32-48 for high-recall requirements
- Rarely need M>64

### ef_construction

The search depth during index construction. Higher values create better indexes but take longer to build.

```rust
let config = CollectionConfig::new("collection", 384)
    .with_distance(DistanceFunction::Cosine)
    .with_hnsw_ef_construction(400);  // Higher for better index quality
```

| ef_construction | Index Quality | Build Time |
|-----------------|---------------|------------|
| 100 | Lower | Faster |
| 200 | Balanced | Balanced |
| 400 | Higher | Slower |
| 800 | Highest | Slowest |

**Guidelines:**
- Use ef_construction >= 2 * M for good results
- For critical applications, use 10-20x M
- This only affects build time, not search performance

### ef_search

The search depth during queries. Higher values improve recall but slow down searches.

```rust
// Set ef_search per query
let results = collection.search_with_params(
    &query,
    10,     // k
    None,   // filter
    100     // ef_search
)?;
```

| ef_search | Recall | Latency |
|-----------|--------|---------|
| 10 | Lower | ~1ms |
| 50 | Balanced | ~3ms |
| 100 | Higher | ~5ms |
| 500 | Highest | ~20ms |

**Guidelines:**
- ef_search should be >= k (number of results requested)
- Start with ef_search = 50-100
- Increase if recall is insufficient
- Decrease if latency is too high

## Auto-Tuning

Needle can automatically tune HNSW parameters based on your data and constraints:

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(1_000_000, 384)  // 1M vectors, 384 dims
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(4 * 1024 * 1024 * 1024);  // 4GB

let result = auto_tune(&constraints);

println!("Recommended M: {}", result.config.hnsw_m);
println!("Recommended ef_construction: {}", result.config.ef_construction);
println!("Estimated recall: {:.2}%", result.estimated_recall * 100.0);
```

### Performance Profiles

| Profile | Focus | Typical Use Case |
|---------|-------|------------------|
| `HighRecall` | Maximize accuracy | Search engines, RAG |
| `Balanced` | Balance speed/quality | General purpose |
| `LowLatency` | Minimize response time | Real-time applications |
| `LowMemory` | Minimize RAM usage | Resource-constrained |

## Measuring Performance

### Recall

Recall measures what fraction of true nearest neighbors are returned:

```rust
// Compute ground truth with brute force
let ground_truth = collection.brute_force_search(&query, k)?;

// Compare with HNSW results
let hnsw_results = collection.search(&query, k)?;

let recall = compute_recall(&ground_truth, &hnsw_results);
println!("Recall@{}: {:.2}%", k, recall * 100.0);
```

### Query Latency

```rust
use std::time::Instant;

let start = Instant::now();
let results = collection.search(&query, 10)?;
let latency = start.elapsed();

println!("Query latency: {:?}", latency);
```

### Search Explain

Use `search_explain` to understand query performance:

```rust
let (results, explain) = collection.search_explain(&query, 10, None)?;

println!("Nodes visited: {}", explain.nodes_visited);
println!("Distance computations: {}", explain.distance_computations);
println!("Layers traversed: {}", explain.layers_traversed);
println!("Time breakdown:");
println!("  Index lookup: {:?}", explain.index_time);
println!("  Distance calc: {:?}", explain.distance_time);
println!("  Filtering: {:?}", explain.filter_time);
```

## Common Tuning Scenarios

### Scenario 1: Low Recall

**Symptoms:** Missing relevant results, poor search quality

**Solutions:**
1. Increase `ef_search`:
   ```rust
   let results = collection.search_with_params(&query, 10, None, 200)?;
   ```

2. If building new index, increase `M` and `ef_construction`:
   ```rust
   let config = CollectionConfig::new("collection", 384)
       .with_distance(DistanceFunction::Cosine)
       .with_hnsw_m(32)
       .with_hnsw_ef_construction(400);
   ```

### Scenario 2: High Latency

**Symptoms:** Slow search responses

**Solutions:**
1. Decrease `ef_search`:
   ```rust
   let results = collection.search_with_params(&query, 10, None, 30)?;
   ```

2. Use quantization:
   ```rust
   let config = CollectionConfig::new("collection", 384)
       .with_distance(DistanceFunction::Cosine)
       .with_quantization(QuantizationType::Scalar);
   ```

3. Reduce dimensions if possible (use a smaller embedding model)

### Scenario 3: High Memory Usage

**Symptoms:** Out of memory errors, high RAM consumption

**Solutions:**
1. Use quantization:
   ```rust
   // Binary quantization: 32x compression
   let config = CollectionConfig::new("collection", 384)
       .with_distance(DistanceFunction::Cosine)
       .with_quantization(QuantizationType::Binary);
   ```

2. Reduce `M`:
   ```rust
   let config = CollectionConfig::new("collection", 384)
       .with_distance(DistanceFunction::Cosine)
       .with_hnsw_m(8);
   ```

3. Shard across multiple collections

### Scenario 4: Slow Index Building

**Symptoms:** Long time to insert vectors

**Solutions:**
1. Reduce `ef_construction`:
   ```rust
   let config = CollectionConfig::new("collection", 384)
       .with_distance(DistanceFunction::Cosine)
       .with_hnsw_ef_construction(100);
   ```

2. Use parallel insertion:
   ```rust
   use rayon::prelude::*;

   vectors.par_iter().for_each(|(id, vec, meta)| {
       collection.insert(id, vec, Some(meta.clone())).unwrap();
   });
   ```

## Comparison with Other Algorithms

| Algorithm | Search Time | Memory | Recall | Build Time |
|-----------|-------------|--------|--------|------------|
| Brute Force | O(n) | O(n*d) | 100% | O(n) |
| HNSW | O(log n) | O(n*M) | 95-99% | O(n*log n) |
| IVF | O(√n) | O(n*d) | 90-95% | O(n) |
| DiskANN | O(log n) | O(n) | 95-99% | O(n*log n) |

HNSW is the default in Needle because it offers the best balance of speed, recall, and flexibility for most use cases.

## Next Steps

- [Distance Functions](/docs/concepts/distance-functions) - Choose the right similarity metric
- [Quantization Guide](/docs/guides/quantization) - Reduce memory usage
- [Production Deployment](/docs/guides/production) - Optimize for production
