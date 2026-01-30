# Architecture

Needle is an embedded vector database written in Rust, designed as "SQLite for vectors." It provides HNSW-based approximate nearest neighbor search with single-file storage and zero configuration.

## High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│              CLI · REST API · Python · WASM              │
├─────────────────────────────────────────────────────────┤
│                    Database Layer                         │
│          Database ──→ CollectionRef (thread-safe)         │
├─────────────────────────────────────────────────────────┤
│                   Collection Layer                        │
│     Vectors + HNSW Index + Metadata + Search Pipeline     │
├─────────────────────────────────────────────────────────┤
│                    Index Layer                            │
│       HNSW · IVF · DiskANN · Sparse · Multi-vector       │
├──────────────────────┬──────────────────────────────────┤
│   Distance Functions │        Storage Layer              │
│  Cosine · Euclidean  │  Single-file · mmap · WAL         │
│  Dot · Manhattan     │  Backup · Cloud Storage            │
└──────────────────────┴──────────────────────────────────┘
```

## Key Modules

| Module | Path | Purpose |
|--------|------|---------|
| **Database** | `src/database/` | Multi-collection container, persistence, thread-safe access |
| **Collection** | `src/collection/` | Vector storage, HNSW index, metadata, and search pipeline |
| **HNSW Index** | `src/indexing/hnsw.rs` | Primary ANN index (Hierarchical Navigable Small Worlds) |
| **Distance** | `src/distance.rs` | Similarity functions with optional SIMD acceleration |
| **Metadata** | `src/metadata.rs` | MongoDB-style metadata storage and query filtering |
| **Storage** | `src/storage.rs` | File I/O, memory mapping, vector storage |
| **Server** | `src/server.rs` | HTTP REST API with auth and rate limiting |
| **Error** | `src/error.rs` | Structured error types with recovery hints |
| **Services** | `src/services/` | High-level service wrappers ([organized by domain](src/services/README.md)) |

## Thread Safety

`Database` uses `parking_lot::RwLock` for interior mutability. `CollectionRef` provides the safe concurrent API — read operations take read locks, writes take write locks.

## Data Flow

1. **Insert**: `CollectionRef::insert()` → validates dimensions → stores vector → updates HNSW graph → indexes metadata
2. **Search**: `CollectionRef::search()` → HNSW nearest neighbor scan → apply metadata filters → return ranked results
3. **Persist**: `Database::save()` → serializes all collections → writes single `.needle` file

## Full Documentation

For the complete architecture document with detailed module descriptions, data flow diagrams, and design decisions, see [docs/architecture.md](docs/architecture.md).
