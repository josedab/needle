# `src/` Directory Map

Quick orientation for contributors. See [ARCHITECTURE.md](../ARCHITECTURE.md) for full details.

## Core

The foundational data-path modules — every query and mutation flows through these.

| Module | Purpose |
|--------|---------|
| `database/` | Multi-collection management, persistence, open/save |
| `collection/` | Vector storage, HNSW search pipeline, config, stats |
| `indexing/` | Index implementations: HNSW, IVF, quantization, sparse, multi-vector |
| `distance.rs` | Distance functions (Cosine, Euclidean, Dot, Manhattan); SIMD variants |
| `error.rs` | `NeedleError` enum, `ErrorCode`, `Result<T>` alias |
| `storage.rs` | File I/O, mmap, vector page layout |
| `metadata.rs` | Metadata storage, MongoDB-style `Filter` parsing |

## API Layers

Entry points that expose Core to the outside world.

| Module | Purpose |
|--------|---------|
| `server/` | HTTP REST API (Axum) — feature: `server` |
| `cli/` | CLI binary (`main.rs` → `cli/`) |
| `python.rs` | Python bindings (PyO3) — feature: `python` |
| `wasm.rs` | WebAssembly bindings — feature: `wasm` |
| `uniffi_bindings.rs` | Swift/Kotlin bindings (UniFFI) — feature: `uniffi-bindings` |
| `mcp.rs` | Model Context Protocol tool interface |

## Extended

Higher-level features, services, and optional/unstable modules.

| Module | Purpose |
|--------|---------|
| `search/` | Query planning, reranking, federation |
| `hybrid.rs` | BM25 + RRF hybrid search — feature: `hybrid` |
| `persistence/` | Backup, WAL, versioning, cloud storage |
| `services/` | Adaptive indexing, ingestion pipelines, plugin runtime |
| `ml/` | Auto-embed, RAG helpers, model registry |
| `observe/` | Observability, diagnostics |
| `metrics.rs` | Prometheus metrics — feature: `metrics` |
| `embeddings.rs` | ONNX embedding inference — feature: `embeddings` |
| `integrations/` | LangChain / LlamaIndex adapters |
| `streaming/` | CDC connectors (Kafka, Pulsar, etc.) |
| `tuning.rs` | Auto-tune HNSW parameters |
| `enterprise/` | Encryption, RBAC, Raft, namespaces |
| `experimental/` | ⚠️ Unstable APIs — GPU, zero-copy, cloud control plane |
| `tui.rs` | Terminal user interface — feature: `tui` |
| `web_ui.rs` | Web-based admin UI — feature: `web-ui` |
