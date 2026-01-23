# ADR-0040: DiskANN and IVF Alternative Index Types

## Status

Accepted

## Context

HNSW (ADR-0002) is the default index, but it has limitations:

1. **Memory requirements** — HNSW keeps the full graph in memory
2. **Large-scale deployments** — Billion-vector collections exceed RAM
3. **Cost optimization** — SSD storage is 10-100x cheaper than RAM
4. **Different workloads** — Some use cases prioritize throughput over latency

Alternative index types address different tradeoffs:

| Index | Memory Usage | Latency | Throughput | Build Time |
|-------|--------------|---------|------------|------------|
| HNSW | High (full graph) | Low (<10ms) | Medium | Medium |
| IVF | Medium (centroids) | Medium (10-50ms) | High | Fast |
| DiskANN | Low (graph on disk) | Medium (10-100ms) | Medium | Slow |
| Flat | Low (no index) | High (linear) | Low | None |

## Decision

Implement **IVF (Inverted File Index)** and **DiskANN** as alternative index types, selectable via auto-tuning or explicit configuration.

### Index Type Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexType {
    /// Hierarchical Navigable Small World (default)
    Hnsw,

    /// Inverted File Index with clustering
    Ivf,

    /// Disk-based Approximate Nearest Neighbor
    DiskAnn,

    /// No index (brute-force search)
    Flat,
}
```

### IVF Index

IVF partitions vectors into clusters, searching only relevant clusters:

```rust
pub struct IvfIndex {
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,

    /// Inverted lists: cluster_id → vectors in cluster
    inverted_lists: Vec<Vec<VectorEntry>>,

    /// Number of clusters
    n_clusters: usize,

    /// Number of clusters to search (nprobe)
    n_probe: usize,
}

impl IvfIndex {
    /// Build index by clustering vectors
    pub fn build(vectors: &[Vec<f32>], config: IvfConfig) -> Result<Self> {
        // K-means clustering to find centroids
        let centroids = kmeans(vectors, config.n_clusters, config.max_iterations)?;

        // Assign vectors to nearest centroid
        let mut inverted_lists = vec![Vec::new(); config.n_clusters];
        for (i, vector) in vectors.iter().enumerate() {
            let cluster_id = find_nearest_centroid(vector, &centroids);
            inverted_lists[cluster_id].push(VectorEntry {
                id: i,
                vector: vector.clone(),
            });
        }

        Ok(Self {
            centroids,
            inverted_lists,
            n_clusters: config.n_clusters,
            n_probe: config.n_probe,
        })
    }

    /// Search by probing nearest clusters
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        // Find n_probe nearest centroids
        let nearest_clusters = self.find_nearest_centroids(query, self.n_probe);

        // Search within those clusters
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for cluster_id in nearest_clusters {
            for entry in &self.inverted_lists[cluster_id] {
                let distance = cosine_distance(query, &entry.vector);
                candidates.push((entry.id, distance));
            }
        }

        // Sort and return top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates.into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}

pub struct IvfConfig {
    /// Number of clusters (typically sqrt(n) to 4*sqrt(n))
    pub n_clusters: usize,

    /// Clusters to search per query (tradeoff: recall vs. speed)
    pub n_probe: usize,

    /// K-means iterations
    pub max_iterations: usize,
}
```

### DiskANN Index

DiskANN stores the graph on SSD with a small in-memory navigation structure:

```rust
pub struct DiskAnnIndex {
    /// In-memory entry points for graph navigation
    entry_points: Vec<usize>,

    /// Memory-mapped graph file
    graph_mmap: Mmap,

    /// Memory-mapped vector file
    vectors_mmap: Mmap,

    /// In-memory compressed vectors for fast distance computation
    compressed_vectors: Vec<CompressedVector>,

    /// Configuration
    config: DiskAnnConfig,
}

impl DiskAnnIndex {
    /// Build index (expensive, run offline)
    pub fn build(
        vectors: &[Vec<f32>],
        output_path: &Path,
        config: DiskAnnConfig,
    ) -> Result<Self> {
        // Phase 1: Build graph in memory
        let graph = build_vamana_graph(vectors, &config)?;

        // Phase 2: Write vectors to disk
        let vectors_path = output_path.join("vectors.bin");
        write_vectors(&vectors_path, vectors)?;

        // Phase 3: Write graph to disk
        let graph_path = output_path.join("graph.bin");
        write_graph(&graph_path, &graph)?;

        // Phase 4: Build compressed vectors for distance computation
        let compressed = compress_vectors(vectors, config.pq_dims)?;

        // Phase 5: Memory-map files
        Self::open(output_path)
    }

    /// Search using beam search with SSD reads
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // Start from entry points
        for &entry in &self.entry_points {
            let distance = self.compute_distance_compressed(query, entry);
            candidates.push(Reverse((OrderedFloat(distance), entry)));
        }

        // Beam search
        while let Some(Reverse((dist, node))) = candidates.pop() {
            if visited.contains(&node) {
                continue;
            }
            visited.insert(node);

            // Add to results
            results.push((dist, node));
            if results.len() > k {
                results.pop();
            }

            // Read neighbors from disk (SSD read)
            let neighbors = self.read_neighbors(node)?;

            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    let distance = self.compute_distance_compressed(query, neighbor);
                    candidates.push(Reverse((OrderedFloat(distance), neighbor)));
                }
            }
        }

        // Rerank with full vectors from disk
        let mut final_results: Vec<SearchResult> = results.into_iter()
            .map(|(_, id)| {
                let full_vector = self.read_vector(id)?;
                let exact_distance = cosine_distance(query, &full_vector);
                Ok(SearchResult { id, distance: exact_distance })
            })
            .collect::<Result<_>>()?;

        final_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(final_results)
    }
}

pub struct DiskAnnConfig {
    /// Graph degree (R in paper)
    pub max_degree: usize,

    /// Build-time beam width (L in paper)
    pub build_beam_width: usize,

    /// Search beam width
    pub search_beam_width: usize,

    /// Product quantization dimensions
    pub pq_dims: usize,
}
```

### Auto-Tuning Index Selection

```rust
impl auto_tune {
    /// Select best index type based on constraints
    pub fn select_index_type(constraints: &TuningConstraints) -> IndexType {
        let vector_count = constraints.expected_vectors;
        let memory_budget = constraints.memory_budget;
        let latency_budget = constraints.latency_budget_ms;

        // Estimate memory requirements
        let hnsw_memory = estimate_hnsw_memory(vector_count, constraints.dimension);
        let ivf_memory = estimate_ivf_memory(vector_count, constraints.dimension);
        let diskann_memory = estimate_diskann_memory(vector_count, constraints.dimension);

        // Decision tree
        if hnsw_memory <= memory_budget && latency_budget >= 10 {
            IndexType::Hnsw  // Best latency if it fits
        } else if ivf_memory <= memory_budget && latency_budget >= 30 {
            IndexType::Ivf  // Good throughput, moderate memory
        } else if diskann_memory <= memory_budget {
            IndexType::DiskAnn  // Minimal memory, SSD-based
        } else {
            IndexType::Flat  // Fallback: no memory for index
        }
    }
}
```

### Code References

- `src/ivf.rs` — IVF index implementation
- `src/diskann.rs` — DiskANN index implementation
- `src/tuning.rs` — Auto-tuning index selection
- `src/hnsw.rs` — HNSW (default) implementation

## Consequences

### Benefits

1. **Scale beyond RAM** — DiskANN handles billion-vector collections
2. **Cost optimization** — Use cheaper SSD storage instead of RAM
3. **Workload matching** — IVF for high-throughput batch processing
4. **Automatic selection** — Auto-tuning picks best index for constraints
5. **Future extensibility** — Framework supports additional index types

### Tradeoffs

1. **DiskANN latency** — SSD reads add 10-50ms per query
2. **IVF recall** — Clustering may miss vectors in wrong clusters
3. **Build time** — DiskANN index building is expensive (hours for billions)
4. **Complexity** — Multiple code paths for different indices

### Index Selection Guidelines

| Scenario | Recommended Index | Rationale |
|----------|-------------------|-----------|
| < 1M vectors, low latency | HNSW | Fits in memory, fastest |
| 1M-100M vectors, high throughput | IVF | Good balance |
| > 100M vectors | DiskANN | Exceeds RAM |
| Prototype/testing | Flat | No index build time |
| Memory-constrained edge | IVF (small) | Centroids only in memory |

### What This Enabled

- Billion-scale vector search on commodity hardware
- Cost-effective deployment (SSD vs. RAM pricing)
- Workload-specific optimization
- Gradual scaling path as data grows

### What This Prevented

- Forced choice of expensive RAM for large collections
- One-size-fits-all indexing that doesn't fit all workloads
- Complex external index management

### Performance Characteristics

```
Query Latency (ms) vs. Collection Size

Latency
  ^
100|                                    ┌── Flat
   |                                ┌───┘
 50|                        ┌───────┤     DiskANN
   |                ┌───────┤       └───
 20|        ┌───────┤       │              IVF
   |    ┌───┤       │       │
 10|────┤   │       │       │                HNSW
   |    │   │       │       │
  0+────┼───┼───────┼───────┼───────────> Vectors
       10K 100K    1M     10M    100M
```

### Usage Example

```rust
use needle::{Database, CollectionConfig, IndexType};

// Explicit index selection
let config = CollectionConfig::default()
    .with_dimension(384)
    .with_index_type(IndexType::DiskAnn)
    .with_diskann_config(DiskAnnConfig {
        max_degree: 64,
        build_beam_width: 128,
        search_beam_width: 64,
        pq_dims: 96,
    });

db.create_collection("large_collection", config)?;

// Or let auto-tuning decide
let constraints = TuningConstraints::new(1_000_000_000, 384)  // 1B vectors
    .with_memory_budget(16 * 1024 * 1024 * 1024)  // 16GB RAM
    .with_latency_budget_ms(50);

let tuned = auto_tune(&constraints);
println!("Selected index: {:?}", tuned.index_type);  // Likely DiskANN
```
