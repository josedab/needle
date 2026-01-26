# Index Selection Guide

This guide helps you choose the right index type for your Needle deployment based on dataset size, performance requirements, and resource constraints.

## Quick Decision Tree

```
                    Dataset Size?
                         │
           ┌─────────────┼─────────────┐
           │             │             │
       < 1M vectors   1M-100M      > 100M
           │             │             │
           ▼             ▼             ▼
         HNSW          IVF         DiskANN
     (in-memory)   (clustered)   (disk-based)
```

**TL;DR:**
- **< 1M vectors**: Use HNSW (default) - fast, accurate, simple
- **1M-100M vectors**: Use IVF with Product Quantization - balanced memory/speed
- **> 100M vectors**: Use DiskANN - disk-efficient, scales to billions

---

## Index Comparison

| Feature | HNSW | IVF | DiskANN |
|---------|------|-----|---------|
| **Best for** | Small-medium datasets | Large datasets | Huge datasets |
| **Max practical size** | ~50M vectors | ~500M vectors | Billions |
| **Memory per vector** | ~120 bytes | ~40 bytes | ~20 bytes |
| **Query latency (p50)** | 1-5ms | 5-20ms | 10-50ms |
| **Recall@10** | 95-99% | 90-95% | 85-95% |
| **Build time** | Medium | Fast | Slow |
| **Incremental updates** | Yes | Partial | No (rebuild) |
| **Disk-based** | No | Optional | Yes |

---

## HNSW (Hierarchical Navigable Small World)

### When to Use
- Dataset fits in memory (< 50M vectors)
- Need highest recall (> 95%)
- Need fast incremental updates
- Low-latency requirements (< 10ms)

### Configuration

```rust
use needle::{CollectionConfig, HnswConfig};

let hnsw = HnswConfig::default()
    .m(16)                // Connections per node (4-64)
    .m_max_0(32)          // Layer 0 connections (usually 2*M)
    .ef_construction(200) // Build-time beam width (50-500)
    .ef_search(50);       // Query-time beam width (10-500)

let config = CollectionConfig::new("my_collection", 384)
    .with_hnsw_config(hnsw);
```

### Parameter Tuning Guide

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| **M** | Less memory, lower recall | More memory, higher recall |
| **ef_construction** | Faster build, lower recall | Slower build, higher recall |
| **ef_search** | Faster queries, lower recall | Slower queries, higher recall |

#### Recommended Presets

| Use Case | M | ef_construction | ef_search | Memory | Recall |
|----------|---|-----------------|-----------|--------|--------|
| **High Recall** | 32 | 400 | 200 | 2x | 99%+ |
| **Balanced** (default) | 16 | 200 | 50 | 1x | 95% |
| **Low Latency** | 8 | 100 | 20 | 0.5x | 90% |
| **Memory Constrained** | 8 | 100 | 30 | 0.5x | 92% |

### Memory Calculation

```
Memory = N × (D × 4 + M × 8 × L) bytes

Where:
  N = number of vectors
  D = dimensions
  M = HNSW M parameter
  L = average layers (~1.5)

Example (1M vectors, 384 dims, M=16):
  1M × (384 × 4 + 16 × 8 × 1.5) = 1.7 GB
```

---

## IVF (Inverted File Index)

### When to Use
- Dataset too large for HNSW (1M-500M vectors)
- Can tolerate slightly lower recall (90-95%)
- Memory is constrained
- Need faster index builds

### Configuration

```rust
use needle::{IvfIndex, IvfConfig};

let config = IvfConfig::new(256)    // Number of clusters
    .with_nprobe(16)                // Clusters to search
    .with_product_quantization(8);  // PQ subvectors (optional)

let mut ivf = IvfIndex::new(384, config);

// Train on sample data (1-10% of dataset)
ivf.train(&sample_vectors)?;

// Add vectors
for (id, vector) in vectors {
    ivf.add(&id, &vector)?;
}
```

### Parameter Tuning Guide

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| **n_clusters** | Number of Voronoi cells | √N to 4√N (N = dataset size) |
| **n_probe** | Clusters searched per query | Start at 1-5% of n_clusters |
| **PQ subvectors** | Product quantization segments | 8-32 (must divide dimensions) |

#### Cluster Count Recommendations

| Dataset Size | n_clusters | n_probe (recall ~95%) |
|--------------|------------|----------------------|
| 1M | 256-1024 | 8-32 |
| 10M | 1024-4096 | 32-64 |
| 100M | 4096-16384 | 64-128 |

### Memory Calculation (with PQ)

```
Memory = n_clusters × D × 4 + N × PQ_subvectors bytes

Example (10M vectors, 384 dims, 1024 clusters, 8 PQ subvectors):
  1024 × 384 × 4 + 10M × 8 = 1.6 MB + 80 MB = ~82 MB
```

---

## DiskANN

### When to Use
- Billion-scale datasets
- Memory is severely constrained
- Can tolerate higher latency (10-50ms)
- Index rebuilt periodically (not real-time updates)

### Configuration

```rust
use needle::{DiskAnnIndex, DiskAnnConfig};

let config = DiskAnnConfig::new()
    .with_max_degree(64)       // Graph degree
    .with_build_complexity(100) // Build beam width
    .with_search_complexity(50) // Search beam width
    .with_pq_chunks(32);       // PQ compression

let index = DiskAnnIndex::build(&vectors, &config)?;
index.save("vectors.diskann")?;

// Later: load and search
let index = DiskAnnIndex::load("vectors.diskann")?;
let results = index.search(&query, 10)?;
```

### Parameter Tuning Guide

| Parameter | Description | Trade-off |
|-----------|-------------|-----------|
| **max_degree** | Max edges per node | Higher = better recall, more disk I/O |
| **build_complexity** | Build-time beam width | Higher = better index, slower build |
| **search_complexity** | Query-time beam width | Higher = better recall, slower queries |
| **pq_chunks** | PQ compression level | More = better compression, lower recall |

### Disk Usage Calculation

```
Disk = N × (D × 4 / compression + max_degree × 4) bytes

Example (1B vectors, 384 dims, 32x compression, degree=64):
  1B × (384 × 4 / 32 + 64 × 4) = 48 GB + 256 GB = ~304 GB
```

---

## Hybrid Approaches

### HNSW + IVF (Two-Stage)

For datasets in the 10M-100M range, consider a two-stage approach:

```rust
// Stage 1: Coarse search with IVF
let ivf_results = ivf.search(&query, 1000)?;

// Stage 2: Rerank with HNSW on candidates
let candidates: Vec<_> = ivf_results.iter().map(|r| r.vector).collect();
let final_results = hnsw.search_within(&query, &candidates, 10)?;
```

### Quantization + Any Index

Combine quantization with any index type for memory savings:

```rust
use needle::ScalarQuantizer;

// Train quantizer
let sq = ScalarQuantizer::train(&sample_vectors);

// Quantize before indexing
let quantized_vectors: Vec<_> = vectors
    .iter()
    .map(|v| sq.quantize(v))
    .collect();

// Use quantized vectors with any index
```

---

## Auto-Tuning

Let Needle automatically select parameters:

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(4 * 1024 * 1024 * 1024)  // 4GB
    .with_target_latency_ms(10);

let result = auto_tune(&constraints);

println!("Recommended index: {:?}", result.recommended_index);
println!("Configuration: {:?}", result.config);
println!("Expected recall: {:.1}%", result.expected_recall * 100.0);
println!("Expected memory: {} MB", result.expected_memory_mb);
println!("Expected latency: {} ms", result.expected_latency_ms);
```

### Performance Profiles

| Profile | Optimizes For | Typical Use Case |
|---------|--------------|------------------|
| `HighRecall` | Accuracy (>99%) | Legal, medical, compliance |
| `Balanced` | Recall/latency trade-off | General search |
| `LowLatency` | Speed (<5ms) | Real-time recommendations |
| `MemoryConstrained` | Memory efficiency | Edge devices, containers |

---

## Migration Between Index Types

### HNSW to IVF

```rust
use needle::{Database, IvfIndex, IvfConfig};

// Export from existing HNSW collection
let db = Database::open("source.needle")?;
let hnsw_coll = db.collection("documents")?;

// Create and train IVF
let ivf_config = IvfConfig::new(1024).with_nprobe(32);
let mut ivf = IvfIndex::new(384, ivf_config);

// Collect training samples
let samples: Vec<Vec<f32>> = hnsw_coll.iter()
    .take(hnsw_coll.len() / 10)
    .map(|(_, v, _)| v)
    .collect();

ivf.train(&samples)?;

// Migrate all vectors
for (id, vector, metadata) in hnsw_coll.iter() {
    ivf.add_with_metadata(&id, &vector, metadata)?;
}
```

### Verify Migration

```rust
// Sample queries for comparison
let test_queries: Vec<_> = /* ... */;

for query in &test_queries {
    let hnsw_results = hnsw_coll.search(query, 10)?;
    let ivf_results = ivf.search(query, 10)?;

    let overlap = calculate_overlap(&hnsw_results, &ivf_results);
    println!("Recall@10: {:.1}%", overlap * 100.0);
}
```

---

## Troubleshooting

### Low Recall

1. **HNSW**: Increase `ef_search` (try 100-200)
2. **IVF**: Increase `n_probe` (try doubling)
3. **DiskANN**: Increase `search_complexity`
4. Check if vectors are normalized (for cosine distance)

### High Latency

1. **HNSW**: Decrease `ef_search`, consider quantization
2. **IVF**: Decrease `n_probe`, add PQ compression
3. **DiskANN**: Use faster SSD, increase RAM for caching
4. Profile with `search_explain()` to find bottleneck

### Out of Memory

1. Enable quantization (Scalar: 4x, PQ: 8-32x, Binary: 32x)
2. Switch to IVF or DiskANN
3. Reduce HNSW `M` parameter
4. Enable memory-mapped storage

---

## See Also

- [HNSW Parameter Tuning](how-to-guides.md#hnsw-parameter-tuning) - Detailed tuning guide
- [Quantization Strategies](how-to-guides.md#quantization-strategies) - Compression options
- [Architecture](architecture.md) - Internal index implementations
- [Benchmarks](../benches/README.md) - Performance measurement methodology
