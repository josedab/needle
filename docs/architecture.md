# Needle Architecture

This document describes the internal architecture of Needle, an embedded vector database written in Rust.

## Overview

Needle is designed as "SQLite for vectors" - a single-file, embedded vector database that provides high-performance approximate nearest neighbor (ANN) search with zero configuration.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Application                              │
├─────────────────────────────────────────────────────────────────┤
│                      Needle Public API                           │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Database │  │ Collection │  │  Filter  │  │ SearchResult │  │
│  └──────────┘  └────────────┘  └──────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                        Core Components                           │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │   HNSW   │  │  Metadata  │  │ Storage  │  │ Quantization │  │
│  │  Index   │  │   Store    │  │  Layer   │  │    Layer     │  │
│  └──────────┘  └────────────┘  └──────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Distance Functions                           │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌──────────────┐  │
│  │  Cosine  │  │ Euclidean  │  │   Dot    │  │  Manhattan   │  │
│  │  (SIMD)  │  │   (SIMD)   │  │ (SIMD)   │  │   (SIMD)     │  │
│  └──────────┘  └────────────┘  └──────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      File System / Memory                        │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### Database

The `Database` struct is the main entry point. It manages multiple collections and handles persistence.

```rust
pub struct Database {
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
    path: Option<PathBuf>,
    config: DatabaseConfig,
    dirty: AtomicBool,
}
```

**Key responsibilities:**
- Collection lifecycle (create, drop, list)
- Persistence (save/load from file)
- Thread-safe access via `CollectionRef`

### Collection

A `Collection` holds vectors, their metadata, and the HNSW index for fast search.

```rust
pub struct Collection {
    name: String,
    dimensions: usize,
    distance: DistanceFunction,
    vectors: VectorStore,
    metadata: MetadataStore,
    index: HnswIndex,
    config: CollectionConfig,
}
```

**Key responsibilities:**
- Vector CRUD operations
- Search with optional filtering
- Index maintenance

### HNSW Index

The Hierarchical Navigable Small World (HNSW) index provides approximate nearest neighbor search with logarithmic time complexity.

```
Layer 2:  [A]─────────────────[D]
           │                   │
Layer 1:  [A]────[B]────[C]───[D]────[E]
           │      │      │     │      │
Layer 0:  [A]─[B]─[C]─[D]─[E]─[F]─[G]─[H]─[I]─[J]
```

**Structure:**
- Multiple layers with decreasing density
- Top layers for coarse navigation
- Bottom layer contains all vectors
- Each node connected to M neighbors

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `M` | 16 | Max connections per node per layer |
| `M_max_0` | 32 | Max connections at layer 0 |
| `ef_construction` | 200 | Search width during insertion |
| `ef_search` | 50 | Search width during query |
| `ml` | 1/ln(M) | Level generation multiplier |

**Search algorithm:**
1. Start at entry point on top layer
2. Greedily descend through layers
3. At layer 0, expand search with ef_search candidates
4. Return k nearest neighbors

### Metadata Store

Stores and indexes metadata for filtering during search.

```rust
pub struct MetadataStore {
    data: HashMap<VectorId, Value>,
    id_map: HashMap<String, VectorId>,
    reverse_map: HashMap<VectorId, String>,
}
```

**Filter operations:**
- Equality: `{"field": "value"}`
- Comparison: `{"field": {"$gt": 10}}`
- Logical: `{"$and": [...]}`, `{"$or": [...]}`
- Membership: `{"field": {"$in": [...]}}`

### Storage Layer

Handles persistence with optional memory-mapping for large files.

```rust
pub struct VectorStore {
    vectors: Vec<Vec<f32>>,
    deleted: HashSet<VectorId>,
}
```

**File format:**
```
┌────────────────────────────────────────┐
│ Magic Number (8 bytes): "NEEDLE01"     │
├────────────────────────────────────────┤
│ Header                                  │
│  - Version (4 bytes)                   │
│  - Flags (4 bytes)                     │
│  - Collection count (4 bytes)          │
├────────────────────────────────────────┤
│ Collection 1                           │
│  - Name length + name                  │
│  - Dimensions (4 bytes)                │
│  - Vector count (8 bytes)              │
│  - Vectors [f32 × dimensions × count]  │
│  - Metadata (MessagePack)              │
│  - HNSW Index                          │
├────────────────────────────────────────┤
│ Collection 2...                        │
└────────────────────────────────────────┘
```

### Distance Functions

SIMD-optimized distance calculations:

| Function | Formula | Use Case |
|----------|---------|----------|
| Cosine | 1 - (a·b)/(‖a‖‖b‖) | Semantic similarity |
| Euclidean | √Σ(aᵢ-bᵢ)² | Spatial distance |
| Dot Product | -a·b | When vectors are normalized |
| Manhattan | Σ\|aᵢ-bᵢ\| | Sparse vectors |

**SIMD implementations:**
- x86_64: AVX2 (256-bit) and AVX-512
- ARM: NEON (128-bit)
- Fallback: Scalar implementation

### Quantization

Reduces memory usage with minimal accuracy loss:

**Scalar Quantization (SQ8):**
- Maps f32 → u8
- 4x memory reduction
- <1% recall loss

**Product Quantization (PQ):**
- Splits vector into subvectors
- Each subvector → centroid ID
- 16-64x compression

**Binary Quantization:**
- Sign of each dimension → bit
- 32x compression
- Good for candidate generation

## Thread Safety

Needle uses a hierarchical locking strategy:

```
Database (RwLock)
└── Collections (HashMap)
    └── Collection (RwLock via CollectionRef)
        ├── VectorStore (internal sync)
        ├── MetadataStore (internal sync)
        └── HnswIndex (internal sync)
```

**Guarantees:**
- Multiple readers can access simultaneously
- Writers get exclusive access
- No deadlocks (lock ordering enforced)

## Distributed Architecture

For horizontal scaling, Needle supports sharding and replication:

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Router                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Consistent Hash Ring                      │  │
│  │   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐     │  │
│  │   │S0 │───│S1 │───│S2 │───│S3 │───│S0 │───│S1 │     │  │
│  │   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘     │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Shard 0      Shard 1      Shard 2      Shard 3            │
│  ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐             │
│  │Needle│    │Needle│    │Needle│    │Needle│             │
│  │ Node │    │ Node │    │ Node │    │ Node │             │
│  └──────┘    └──────┘    └──────┘    └──────┘             │
└─────────────────────────────────────────────────────────────┘
```

**Sharding:**
- Consistent hashing for even distribution
- Virtual nodes for balance
- Automatic rebalancing

**Replication (Raft):**
- Leader election
- Log replication
- Consistency guarantees

**CRDT (Conflict-free):**
- Eventual consistency
- No coordination required
- Hybrid logical clocks

## Performance Characteristics

### Time Complexity

| Operation | Average | Worst |
|-----------|---------|-------|
| Insert | O(log N × M × ef) | O(N × M) |
| Search | O(log N × ef) | O(N) |
| Delete | O(1) | O(1) |
| Get by ID | O(1) | O(1) |

### Memory Usage

```
Per vector: dimensions × 4 bytes (f32)
HNSW overhead: ~M × 8 bytes per vector per layer
Metadata: variable (JSON storage)

Example (1M vectors, 384 dimensions, M=16):
- Vectors: 1M × 384 × 4 = 1.5 GB
- HNSW: ~1M × 16 × 8 × 1.5 layers = ~192 MB
- Total: ~1.7 GB
```

### Disk Usage

Similar to memory, plus:
- Collection metadata
- Index serialization overhead (~10%)

## Feature Flags

| Flag | Description |
|------|-------------|
| `simd` | SIMD-optimized distance functions |
| `server` | HTTP REST API server |
| `metrics` | Prometheus metrics |
| `hybrid` | BM25 hybrid search |
| `embeddings` | ONNX embedding inference |
| `full` | All non-binding features |

## Security Model

- **Encryption at rest:** ChaCha20-Poly1305 (optional)
- **RBAC:** Role-based access control
- **Audit logging:** All operations logged
- **Rate limiting:** Per-IP request limits

## Future Considerations

1. **GPU acceleration:** CUDA/OpenCL distance computation
2. **Tiered storage:** Hot/warm/cold data separation
3. **Streaming ingestion:** Real-time vector updates
4. **Federated search:** Cross-cluster queries
