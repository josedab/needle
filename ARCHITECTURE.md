# Architecture

> **Full architecture documentation:** [docs/architecture.md](docs/architecture.md)
>
> **Services architecture:** [docs/ARCHITECTURE_SERVICES.md](docs/ARCHITECTURE_SERVICES.md)
>
> This file is a high-level summary. See the linked documents for detailed module descriptions, data flow diagrams, service-layer design, and design decisions.

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

## Module Summary

| Layer | Path | Responsibility |
|-------|------|----------------|
| **Core** | `src/collection/`, `src/database/`, `src/indexing/`, `src/distance.rs`, `src/storage.rs`, `src/metadata.rs`, `src/error.rs` | Vector storage, HNSW index, metadata filtering, distance functions, single-file persistence |
| **API — CLI** | `src/cli/` | Command-line interface (clap-based) for all database operations |
| **API — REST** | `src/server/` | HTTP server (Axum) with handlers, middleware, auth (feature: `server`) |
| **API — Bindings** | `src/python.rs`, `src/wasm.rs`, `src/uniffi_bindings.rs` | Python (PyO3), WebAssembly, Swift/Kotlin bindings |
| **Search** | `src/search/`, `src/hybrid.rs` | Query planning, reranking, BM25 + RRF hybrid fusion |
| **Persistence** | `src/persistence/` | Backup, WAL, versioning, cloud storage adapters |
| **Enterprise** | `src/enterprise/` | Raft replication, ChaCha20 encryption, RBAC, namespaces |
| **Experimental** | `src/experimental/` | GPU acceleration, cloud control plane, agentic memory — APIs may change without notice |

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`parking_lot::RwLock`** over `std::sync::RwLock` | Faster uncontended locks, no poisoning, smaller footprint. `Database` uses interior mutability; `CollectionRef` provides the safe concurrent API. |
| **Single-file storage** | Simplifies deployment, backup, and distribution — inspired by SQLite. All collections, indexes, and metadata serialize into one `.needle` file. |
| **HNSW-first indexing** | HNSW provides the best recall/latency tradeoff for the common case (< 10M vectors). IVF and DiskANN are available for specialized workloads. |
| **`#![deny(unsafe_code)]`** crate-wide | Safety by default. Only `distance.rs`, `storage.rs`, and select experimental modules are exempted for SIMD and mmap. |
| **Feature flags for optional layers** | Keeps the default binary small. `server`, `hybrid`, `metrics`, `encryption`, and `experimental` are opt-in via Cargo features. |

## Thread Safety

`Database` uses `parking_lot::RwLock` for interior mutability. `CollectionRef` provides the safe concurrent API — read operations take read locks, writes take write locks.

## Data Flow

1. **Insert**: `CollectionRef::insert()` → validates dimensions → stores vector → updates HNSW graph → indexes metadata
2. **Search**: `CollectionRef::search()` → HNSW nearest neighbor scan → apply metadata filters → return ranked results
3. **Persist**: `Database::save()` → serializes all collections → writes single `.needle` file
