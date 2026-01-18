# ADR-0002: HNSW as Primary Index Algorithm

## Status

Accepted

## Context

Vector similarity search requires efficient algorithms to avoid O(n) brute-force scans. For a database targeting sub-10ms query latency with high recall, the choice of approximate nearest neighbor (ANN) index is critical.

Several well-established ANN algorithms were considered:

| Algorithm | Search Complexity | Insert Complexity | Memory | Strengths |
|-----------|------------------|-------------------|--------|-----------|
| **HNSW** | O(log n) | O(log n) | High | Fast search, online insertion |
| **IVF** | O(n/k + k) | O(1) amortized | Medium | Good for very large datasets |
| **DiskANN** | O(log n) | Batch only | Low (disk) | Massive datasets exceeding RAM |
| **Annoy** | O(log n) | Batch only | Medium | Simple, read-only after build |
| **LSH** | O(1) expected | O(1) | High | Sub-linear but lower recall |

Key requirements for Needle:

1. **Online insertion** — Users insert vectors one at a time; batch-only algorithms are unsuitable
2. **High recall** — Must achieve >95% recall for production use cases
3. **Sub-millisecond latency** — Target <10ms for typical queries
4. **Tunable parameters** — Different use cases need different accuracy/speed tradeoffs
5. **Memory efficiency** — Should work on laptops, not just servers

## Decision

Adopt **Hierarchical Navigable Small World (HNSW)** as the default and primary index algorithm.

### Algorithm Overview

HNSW constructs a multi-layer graph where:
- Layer 0 contains all vectors
- Higher layers contain exponentially fewer vectors (selected probabilistically)
- Search starts at the top layer and greedily descends, using upper layers for fast traversal and lower layers for precision

```
Layer 2:    A -------- D
            |          |
Layer 1:    A --- B -- D --- F
            |    |     |     |
Layer 0:    A-B--C--D--E--F--G--H
```

### Configurable Parameters

```rust
pub struct HnswConfig {
    /// Maximum connections per node (default: 16)
    pub m: usize,

    /// Connections during construction (default: 200)
    pub ef_construction: usize,

    /// Candidates during search (default: 50)
    pub ef_search: usize,

    /// Level multiplier (default: 1/ln(M))
    pub ml: f64,
}
```

**Parameter effects:**
- Higher `M` → Better recall, more memory, slower insertion
- Higher `ef_construction` → Better graph quality, slower build
- Higher `ef_search` → Better recall, slower queries

### Code References

- `src/hnsw.rs:1-99` — Algorithm documentation and comparison table
- `src/hnsw.rs:125-201` — HnswConfig with defaults and validation
- `src/collection.rs:48` — Collection uses HNSW by default
- `src/tuning.rs` — Auto-tuning of HNSW parameters based on workload

### Auto-Tuning Support

```rust
let constraints = TuningConstraints::new(vector_count, dimensions)
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(500 * 1024 * 1024);

let result = auto_tune(&constraints);
// result.config contains optimized M, ef_construction, ef_search
```

## Consequences

### Benefits

1. **O(log n) search and insertion** — Scales to millions of vectors without linear degradation
2. **Online insertion** — No need to rebuild index; vectors added incrementally
3. **High recall achievable** — With proper tuning, >99% recall is possible
4. **Well-understood algorithm** — Published in 2016, widely implemented and studied
5. **Tunable tradeoffs** — Users can prioritize speed vs accuracy via parameters
6. **Parallel-friendly** — Search is read-only; multiple queries execute concurrently

### Tradeoffs

1. **Memory overhead** — Each vector requires ~M×8 bytes for graph edges (vs. zero for brute force)
2. **Deletion complexity** — Removing vectors requires graph repair or tombstoning
3. **Not optimal for all workloads** — Very high dimensions (>1000) or very small datasets may prefer other approaches
4. **Parameter sensitivity** — Suboptimal parameters significantly degrade performance

### What This Enabled

- Sub-millisecond queries for typical workloads (384-1536 dimensions, <1M vectors)
- Incremental index updates without full rebuilds
- `batch_search` parallelization via Rayon for throughput
- Auto-tuning based on dataset characteristics and performance profiles

### What This Prevented

- Disk-based indices for datasets exceeding RAM (would require DiskANN or similar)
- Zero-memory-overhead search (brute force is always available as fallback)
- Guaranteed exact nearest neighbors (HNSW is approximate by design)

### Future Considerations

The architecture supports adding alternative index types:
- IVF for very large datasets with memory constraints
- DiskANN for datasets exceeding available RAM
- Flat/brute-force for small datasets or exact search requirements

These would be selectable per-collection while maintaining the same search API.
