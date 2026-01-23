---
sidebar_position: 1
---

# HNSW Tuning

This guide provides detailed recommendations for tuning HNSW parameters based on your use case.

## Parameters Overview

| Parameter | Range | Default | Impact |
|-----------|-------|---------|--------|
| `M` | 4-64 | 16 | Graph connectivity |
| `ef_construction` | 50-800 | 200 | Index build quality |
| `ef_search` | 10-500 | 50 | Search quality |

## Parameter Details

### M (Max Connections)

Controls how many connections each node has in the HNSW graph.

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(24);
```

**Impact:**
- Higher M → Better recall, more memory, slower builds
- Lower M → Lower recall, less memory, faster builds

**Recommendations:**

| Dataset Size | Use Case | Recommended M |
|--------------|----------|---------------|
| &lt; 100K | General | 8-12 |
| 100K - 1M | General | 16 (default) |
| 100K - 1M | High recall | 24-32 |
| 1M - 10M | General | 16-24 |
| 1M - 10M | High recall | 32-48 |
| &gt; 10M | Memory constrained | 8-12 |
| &gt; 10M | High recall | 24-32 |

### ef_construction

Controls search depth during index building.

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_ef_construction(400);
```

**Impact:**
- Higher → Better index quality, slower builds
- Lower → Lower index quality, faster builds

**Rule of thumb:** `ef_construction >= 2 * M`

| Priority | Recommended ef_construction |
|----------|---------------------------|
| Fast builds | 100-150 |
| Balanced | 200 (default) |
| High quality | 400-500 |
| Maximum quality | 600-800 |

### ef_search

Controls search depth at query time.

```rust
let results = collection.search_with_params(&query, 10, None, 100)?;
```

**Impact:**
- Higher → Better recall, slower queries
- Lower → Lower recall, faster queries

**Constraint:** `ef_search >= k` (number of results)

| Query Latency Target | Recommended ef_search |
|----------------------|----------------------|
| &lt; 1ms | 10-30 |
| 1-5ms | 30-100 |
| 5-10ms | 100-200 |
| &gt; 10ms | 200-500 |

## Tuning Workflow

### 1. Establish Baseline

```rust
// Start with defaults
let config = CollectionConfig::new(384, DistanceFunction::Cosine);
// M=16, ef_construction=200, ef_search=50
```

### 2. Measure Performance

```rust
fn benchmark(collection: &Collection, queries: &[Vec<f32>]) -> (f32, Duration) {
    let mut total_recall = 0.0;
    let mut total_time = Duration::ZERO;

    for query in queries {
        let start = Instant::now();
        let results = collection.search(query, 10, None)?;
        total_time += start.elapsed();

        // Compare with ground truth
        let ground_truth = collection.brute_force_search(query, 10)?;
        total_recall += compute_recall(&results, &ground_truth);
    }

    (
        total_recall / queries.len() as f32,
        total_time / queries.len() as u32
    )
}
```

### 3. Tune Based on Results

```
If recall is too low:
  1. Increase ef_search first (cheapest)
  2. If still low, increase M (requires rebuild)
  3. If still low, increase ef_construction (requires rebuild)

If latency is too high:
  1. Decrease ef_search first
  2. Consider quantization
  3. Decrease M (requires rebuild)
```

### 4. Use Auto-Tuning

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::Balanced)
    .with_memory_budget(4 * 1024 * 1024 * 1024)
    .with_latency_target_ms(5)
    .with_recall_target(0.95);

let result = auto_tune(&constraints);

let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(result.config.hnsw_m)
    .with_hnsw_ef_construction(result.config.ef_construction);
```

## Memory Estimation

Memory usage per vector:

```
Memory = (dimensions * sizeof(f32)) + (M * 2 * sizeof(u32)) + overhead

For 384 dims, M=16:
Memory ≈ (384 * 4) + (16 * 2 * 4) + 64 ≈ 1728 bytes/vector
```

| Vectors | M=8 | M=16 | M=32 |
|---------|-----|------|------|
| 100K | 150 MB | 170 MB | 210 MB |
| 1M | 1.5 GB | 1.7 GB | 2.1 GB |
| 10M | 15 GB | 17 GB | 21 GB |

## Recall-Latency Trade-offs

Typical results for 1M vectors, 384 dimensions:

| Configuration | Recall@10 | Latency |
|---------------|-----------|---------|
| M=8, ef=30 | 92% | 1.5ms |
| M=16, ef=50 | 97% | 3ms |
| M=16, ef=100 | 99% | 5ms |
| M=32, ef=100 | 99.5% | 6ms |
| M=32, ef=200 | 99.8% | 10ms |

## Scenario-Based Recommendations

### Semantic Search (RAG)

High recall is critical; latency can be higher.

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(24)
    .with_hnsw_ef_construction(400);

// At search time
let results = collection.search_with_params(&query, 10, None, 150)?;
```

### Real-Time Recommendations

Low latency is critical; moderate recall acceptable.

```rust
let config = CollectionConfig::new(256, DistanceFunction::Dot)
    .with_hnsw_m(12)
    .with_hnsw_ef_construction(150);

// At search time
let results = collection.search_with_params(&query, 20, None, 30)?;
```

### Large-Scale Image Search

Memory is critical; use quantization.

```rust
let config = CollectionConfig::new(512, DistanceFunction::Euclidean)
    .with_hnsw_m(16)
    .with_hnsw_ef_construction(200)
    .with_quantization(QuantizationType::Scalar);

// At search time - increase ef_search to compensate for quantization
let results = collection.search_with_params(&query, 10, None, 100)?;
```

### Hybrid Search

Need to over-fetch due to post-filtering.

```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(20)
    .with_hnsw_ef_construction(250);

// Increase ef_search when heavy filtering expected
let ef_search = if has_restrictive_filter { 200 } else { 50 };
let results = collection.search_with_params(&query, 10, filter, ef_search)?;
```

## Dynamic ef_search

Adjust ef_search based on query requirements:

```rust
fn adaptive_ef_search(k: usize, has_filter: bool, quality_tier: &str) -> usize {
    let base = k.max(10);

    let multiplier = match quality_tier {
        "low_latency" => 1,
        "balanced" => 3,
        "high_recall" => 5,
        _ => 3,
    };

    let filter_boost = if has_filter { 2 } else { 1 };

    base * multiplier * filter_boost
}
```

## Monitoring and Alerting

Track these metrics in production:

```rust
// After each search
let (results, explain) = collection.search_explain(&query, k, filter)?;

metrics.observe("search_latency", explain.total_time);
metrics.observe("nodes_visited", explain.nodes_visited);
metrics.observe("distance_computations", explain.distance_computations);

// Alert if too many nodes visited (index may need rebuild)
if explain.nodes_visited > expected_nodes * 2 {
    alert("High node visitation - consider index rebuild");
}
```

## Next Steps

- [Feature Flags](/docs/configuration/feature-flags)
- [HNSW Index Concepts](/docs/concepts/hnsw-index)
- [Production Deployment](/docs/guides/production)
