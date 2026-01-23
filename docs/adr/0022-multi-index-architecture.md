# ADR-0022: Multi-Index Architecture with Constraint-Based Selection

## Status

Accepted

## Context

Different vector search workloads have fundamentally different requirements:

| Workload | Vector Count | Memory Budget | Latency SLA | Update Frequency |
|----------|--------------|---------------|-------------|------------------|
| Real-time recommendations | 100K-1M | High | <10ms | Frequent |
| Document search | 1M-10M | Medium | <50ms | Moderate |
| Image similarity | 10M-100M | Low | <100ms | Rare |
| Archival search | 100M+ | Minimal | <500ms | Never |

HNSW (Hierarchical Navigable Small World) is Needle's primary index and excels at low-latency search with moderate memory overhead. However, it's not optimal for all scenarios:

1. **Memory pressure** — HNSW requires ~1KB per vector for graph structure; at 100M vectors, that's 100GB just for the index
2. **Disk-based workloads** — Some applications can't fit indices in memory
3. **Write-heavy workloads** — HNSW rebuilds are expensive

### Alternatives Considered

1. **HNSW-only with tuning** — Adjust M and ef parameters; doesn't solve fundamental memory limits
2. **External index delegation** — Use Faiss/Annoy externally; adds operational complexity
3. **Single "best" index** — No index is universally optimal

## Decision

Needle implements a **multi-index architecture** with three index types and a **constraint-based selection system**:

### Supported Index Types

```
┌─────────────────────────────────────────────────────────────────┐
│                        Index Selection                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │    HNSW     │   │     IVF     │   │   DiskANN   │           │
│  │             │   │             │   │             │           │
│  │ <1M vectors │   │ 1M-100M     │   │ >10M        │           │
│  │ In-memory   │   │ Clustered   │   │ Disk-based  │           │
│  │ <10ms p99   │   │ <50ms p99   │   │ <100ms p99  │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### HNSW (Default)

- **Best for**: <1M vectors, latency-critical, frequent updates
- **Memory**: ~1KB per vector (graph structure)
- **Latency**: <10ms typical
- **Implementation**: `src/hnsw.rs`

#### IVF (Inverted File Index)

- **Best for**: 1M-100M vectors, memory-constrained, batch updates
- **Memory**: ~100 bytes per vector (cluster assignments)
- **Latency**: 10-50ms typical (depends on nprobe)
- **Implementation**: `src/ivf.rs`

#### DiskANN

- **Best for**: >10M vectors, disk-based, archival
- **Memory**: Minimal (index lives on disk)
- **Latency**: 50-100ms typical (includes disk I/O)
- **Implementation**: `src/diskann.rs`

### Constraint-Based Selection

The `tuning.rs` module provides automatic index recommendation:

```rust
// src/tuning.rs
pub struct IndexSelectionConstraints {
    /// Number of vectors to index
    pub vector_count: usize,

    /// Vector dimensionality
    pub dimensions: usize,

    /// Maximum memory budget in bytes
    pub memory_budget: Option<usize>,

    /// Target p99 latency in milliseconds
    pub latency_sla_ms: Option<u64>,

    /// Expected queries per second
    pub expected_qps: Option<u64>,

    /// How often vectors are updated
    pub update_frequency: UpdateFrequency,
}

pub enum UpdateFrequency {
    Realtime,   // Multiple updates per second
    Frequent,   // Updates every few minutes
    Moderate,   // Updates every few hours
    Rare,       // Updates daily or less
    Never,      // Static dataset
}

pub fn recommend_index(constraints: &IndexSelectionConstraints) -> IndexRecommendation {
    // Decision tree based on constraints
    let recommendation = if constraints.vector_count < 100_000 {
        RecommendedIndex::Hnsw(hnsw_params_for_small())
    } else if constraints.vector_count < 1_000_000 {
        match constraints.memory_budget {
            Some(budget) if budget < constraints.vector_count * 1024 => {
                RecommendedIndex::Ivf(ivf_params_memory_constrained())
            }
            _ => RecommendedIndex::Hnsw(hnsw_params_for_medium())
        }
    } else if constraints.vector_count < 10_000_000 {
        RecommendedIndex::Ivf(ivf_params_for_scale())
    } else {
        RecommendedIndex::DiskAnn(diskann_params_for_massive())
    };

    IndexRecommendation {
        index: recommendation,
        confidence: calculate_confidence(&constraints),
        rationale: generate_rationale(&constraints, &recommendation),
    }
}
```

### Index Interface Abstraction

All indices implement a common trait:

```rust
pub trait VectorIndex: Send + Sync {
    /// Insert a vector with the given ID
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<()>;

    /// Search for k nearest neighbors
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>>;

    /// Delete a vector by ID
    fn delete(&mut self, id: VectorId) -> Result<bool>;

    /// Number of indexed vectors
    fn len(&self) -> usize;

    /// Memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Index type identifier
    fn index_type(&self) -> IndexType;
}
```

### Runtime Index Switching

Collections can switch indices without data loss:

```rust
impl Collection {
    /// Rebuild the collection with a different index type
    pub fn rebuild_index(&mut self, new_index: IndexType, config: IndexConfig) -> Result<()> {
        // 1. Extract all vectors
        let vectors: Vec<_> = self.iter().collect();

        // 2. Create new index
        let mut new_index = match new_index {
            IndexType::Hnsw => Box::new(HnswIndex::new(config.into())),
            IndexType::Ivf => Box::new(IvfIndex::new(config.into())),
            IndexType::DiskAnn => Box::new(DiskAnnIndex::new(config.into())),
        };

        // 3. Bulk insert into new index
        for (id, vector, _metadata) in vectors {
            new_index.insert(id, vector)?;
        }

        // 4. Atomic swap
        self.index = new_index;
        Ok(())
    }
}
```

## Consequences

### Benefits

1. **Right-sized indices** — Each workload gets the optimal index for its constraints
2. **Graceful scaling** — Start with HNSW, migrate to IVF/DiskANN as data grows
3. **Memory flexibility** — Can trade latency for memory savings
4. **Future extensibility** — New index types can be added via the trait

### Tradeoffs

1. **Complexity** — Three index implementations to maintain
2. **Migration overhead** — Switching indices requires full rebuild
3. **Configuration burden** — Users must understand tradeoffs (mitigated by auto-selection)
4. **Testing matrix** — Each index type needs thorough testing

### What This Enabled

- **100M+ vector deployments** — DiskANN makes massive scale feasible
- **Memory-constrained environments** — IVF runs on smaller machines
- **Hybrid deployments** — Different collections can use different indices

### What This Prevented

- **One-size-fits-all limitations** — No single index dominates all scenarios
- **Forced external tools** — Users don't need Faiss for scale

## References

- HNSW implementation: `src/hnsw.rs`
- IVF implementation: `src/ivf.rs`
- DiskANN implementation: `src/diskann.rs`
- Index selection logic: `src/tuning.rs:50-200`
- Index trait: `src/collection.rs` (VectorIndex trait)
