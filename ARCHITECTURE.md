# Architecture

> **Full architecture documentation:** [docs/architecture.md](docs/architecture.md)
>
> This file is a high-level summary. See the full document for detailed module descriptions, data flow diagrams, and design decisions.

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

## Thread Safety

`Database` uses `parking_lot::RwLock` for interior mutability. `CollectionRef` provides the safe concurrent API — read operations take read locks, writes take write locks.

## Data Flow

1. **Insert**: `CollectionRef::insert()` → validates dimensions → stores vector → updates HNSW graph → indexes metadata
2. **Search**: `CollectionRef::search()` → HNSW nearest neighbor scan → apply metadata filters → return ranked results
3. **Persist**: `Database::save()` → serializes all collections → writes single `.needle` file
