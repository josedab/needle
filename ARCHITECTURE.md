# Architecture

> For the full architecture document with data flow diagrams and implementation details,
> see [docs/architecture.md](docs/architecture.md).

## Overview

Needle is an embedded vector database ("SQLite for vectors") providing ANN search
with single-file storage, HNSW indexing, and zero configuration.

```
Application → Database → Collection → HNSW Index → Distance Functions (SIMD)
                                    → Metadata Store
                                    → Storage Layer → .needle File / mmap
```

## Module Map

```
src/
├── lib.rs                  # Library entry, re-exports public API
├── main.rs                 # CLI application
│
├── Core
│   ├── collection/mod.rs   # Collection: vectors + metadata + index
│   ├── database/mod.rs     # Database: multi-collection management
│   ├── error.rs            # Error types with structured codes
│   ├── storage.rs          # File I/O, mmap, vector storage
│   └── metadata.rs         # Metadata storage and filtering
│
├── Indexing (src/indexing/)
│   ├── hnsw.rs             # HNSW index (primary index)
│   ├── ivf.rs              # IVF (Inverted File) index
│   ├── diskann.rs          # DiskANN on-disk index
│   ├── sparse.rs           # Sparse vector inverted index
│   ├── quantization.rs     # Scalar, Product, Binary quantization
│   └── multivec.rs         # Multi-vector (ColBERT) support
│
├── Search (src/search/)
│   ├── query_lang.rs       # NeedleQL query language
│   ├── query_planner.rs    # Query planning and optimization
│   ├── reranker.rs         # Cross-encoder reranking
│   └── federated.rs        # Multi-instance federated search
│
├── distance.rs             # Distance functions (Cosine, Euclidean, Dot, Manhattan)
│
├── Interfaces
│   ├── server.rs           # HTTP REST API (feature: server)
│   ├── python.rs           # Python bindings (feature: python)
│   ├── wasm.rs             # WASM bindings (feature: wasm)
│   └── tui.rs              # Terminal UI (feature: tui)
│
├── Enterprise (src/enterprise/) — Beta
│   ├── encryption.rs       # ChaCha20-Poly1305 encryption at rest
│   ├── security.rs         # RBAC and audit logging
│   ├── wal.rs              # Write-ahead logging
│   ├── raft.rs             # Raft consensus
│   └── namespace.rs        # Multi-tenancy
│
├── Persistence (src/persistence/)
│   ├── backup.rs           # Backup and restore
│   ├── versioning.rs       # Vector version control
│   ├── cloud_storage/      # S3/GCS/Azure backends
│   └── wal.rs              # Write-ahead logging
│
└── Experimental (src/experimental/) — ⚠️ APIs may change without notice
    ├── gpu.rs              # GPU acceleration (scaffolding, CPU fallback)
    ├── agentic_memory.rs
    ├── playground.rs
    └── ... (~25 more modules)
```

## Key Types

| Type | Module | Role |
|------|--------|------|
| `Database` | `database/mod.rs` | Entry point, manages collections and persistence |
| `Collection` | `collection/mod.rs` | Holds vectors, HNSW index, and metadata |
| `CollectionRef` | `database/collection_ref.rs` | Thread-safe reference (Arc + RwLock) |
| `HnswIndex` | `indexing/hnsw.rs` | Hierarchical Navigable Small World graph |
| `Filter` | `metadata.rs` | MongoDB-style metadata query filters |
| `SearchResult` | `collection/search.rs` | Result with id, distance, and metadata |
| `NeedleError` | `error.rs` | Structured error with error codes |

## API Stability Tiers

See [docs/api-stability.md](docs/api-stability.md) for the full policy.

- **Stable**: Core types (`Database`, `Collection`, `Filter`, `SearchResult`, etc.)
- **Beta**: Enterprise features (backup, encryption, Raft, sharding)
- **Experimental**: GPU, cloud control, agentic memory, etc.

## Thread Safety

- `Database` uses `parking_lot::RwLock` for interior mutability
- `CollectionRef` provides safe concurrent access
- Read operations take read locks; writes take write locks
