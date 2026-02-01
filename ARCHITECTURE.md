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

### Stable

| Module | Path | Purpose |
|--------|------|---------|
| **Database** | `src/database/` | Multi-collection container, persistence, thread-safe access |
| **Collection** | `src/collection/` | Vector storage, HNSW index, metadata, and search pipeline |
| **HNSW Index** | `src/indexing/hnsw.rs` | Primary ANN index (Hierarchical Navigable Small Worlds) |
| **Distance** | `src/distance.rs` | Similarity functions with optional SIMD acceleration |
| **Metadata** | `src/metadata.rs` | MongoDB-style metadata storage and query filtering |
| **Storage** | `src/storage.rs` | File I/O, memory mapping, vector storage |
| **Error** | `src/error.rs` | Structured error types with recovery hints |
| **Indexing** | `src/indexing/` | HNSW, IVF, DiskANN, sparse, multi-vector, and quantization indexes |
| **Search** | `src/search/` | Query planning, reranking, federated search, and natural language filtering |
| **Persistence** | `src/persistence/` | WAL, backups, migrations, sharding, and cloud storage |
| **Tuning** | `src/tuning.rs` | Auto-tuning HNSW parameters based on workload profiling |
| **ML** | `src/ml/` | Auto-embedding, model registry, RAG, and dimensionality reduction |
| **Enterprise** | `src/enterprise/` | Encryption, RBAC, multi-tenancy, Raft consensus, autoscaling |
| **Observe** | `src/observe/` | Telemetry, drift detection, anomaly detection, and profiling |
| **Integrations** | `src/integrations/` | LangChain, LlamaIndex, Haystack, Semantic Kernel adapters |
| **Services** | `src/services/` | High-level service wrappers ([organized by domain](src/services/README.md)) |
| **MCP** | `src/mcp/` | Model Context Protocol tools for LLM integration |

### Feature-Gated

| Module | Path | Feature Flag | Purpose |
|--------|------|-------------|---------|
| **Server** | `src/server/` | `server` | HTTP REST API with auth, rate limiting, and OpenAPI spec |
| **Streaming** | `src/streaming/` | `server` | Real-time vector streaming support |
| **Async API** | `src/async_api/` | `async` | Async database API with streaming |
| **Hybrid** | `src/hybrid/` | `hybrid` | BM25 text search and hybrid vector+text search with RRF |
| **Web UI** | `src/web_ui/` | `web-ui` | Web-based administration UI |
| **Metrics** | `src/metrics/` | `metrics` | Prometheus metrics export and monitoring |
| **Embeddings** | `src/embeddings/` | `embeddings` | ONNX Runtime embedding inference |
| **TUI** | `src/tui/` | `tui` | Terminal UI for interactive database management |
| **Python** | `src/python.rs` | `python` | Python bindings via PyO3 |
| **WASM** | `src/wasm.rs` | `wasm` | WebAssembly bindings for browser and Node.js |
| **UniFFI Bindings** | `src/uniffi_bindings/` | `uniffi-bindings` | Swift and Kotlin bindings via UniFFI |

### Beta & Experimental

| Module | Path | Purpose |
|--------|------|---------|
| **Beta API** | `src/beta_api/` | Beta types approaching stability (minor breaking changes possible) |
| **Experimental API** | `src/experimental_api/` | Experimental types under active development |
| **Experimental** | `src/experimental/` | GPU acceleration, clustering, CRDT, graph ops, and more (APIs may change without notice) |

## Thread Safety

`Database` uses `parking_lot::RwLock` for interior mutability. `CollectionRef` provides the safe concurrent API — read operations take read locks, writes take write locks.

## Data Flow

1. **Insert**: `CollectionRef::insert()` → validates dimensions → stores vector → updates HNSW graph → indexes metadata
2. **Search**: `CollectionRef::search()` → HNSW nearest neighbor scan → apply metadata filters → return ranked results
3. **Persist**: `Database::save()` → serializes all collections → writes single `.needle` file

## Full Documentation

For the complete architecture document with detailed module descriptions, data flow diagrams, and design decisions, see [docs/architecture.md](docs/architecture.md).
