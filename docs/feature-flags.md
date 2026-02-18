# Feature Flags Decision Guide

Needle uses Cargo feature flags to keep the default build minimal while allowing opt-in functionality. This guide helps you pick the right flags for your use case.

## Quick Decision Tree

```
What are you doing?
│
├─ Learning / exploring the library
│  └─ cargo build                          (no flags needed)
│
├─ Developing core logic (collections, indexing, search)
│  └─ cargo build                          (no flags needed)
│
├─ Working on the HTTP server or REST API
│  └─ cargo build --features server
│
├─ Working on hybrid search (BM25 + vector)
│  └─ cargo build --features hybrid
│
├─ Working on metrics / observability
│  └─ cargo build --features server,metrics
│
├─ Working on enterprise features (encryption, RBAC)
│  └─ cargo build --features encryption
│
├─ Working on services/ or experimental/ modules
│  └─ cargo build --features full
│
├─ Running full CI locally
│  └─ cargo build --features full          (mirrors CI)
│
└─ Building for production deployment
   └─ cargo build --release --features server,metrics,hybrid,encryption
```

## Common Contributor Scenarios

| Scenario | Command | What It Enables |
|----------|---------|-----------------|
| **Minimal dev** | `cargo build` | Core library: collections, HNSW, metadata, distance |
| **Server dev** | `cargo build --features server` | HTTP REST API (Axum), async runtime |
| **Full CI mirror** | `cargo build --features full` | Everything CI tests — server, metrics, hybrid, web-ui, encryption, experimental |
| **Production build** | `cargo build --release --features server,metrics,hybrid` | Optimized binary with common production features |
| **Python bindings** | `maturin develop` (in `crates/needle-python/`) | PyO3 bindings |
| **WASM bindings** | `cargo build --features wasm --target wasm32-unknown-unknown` | WebAssembly target |

## Feature Flag Reference

### Stable Flags

| Flag | Dependencies | Purpose |
|------|-------------|---------|
| `simd` | — | SIMD-optimized distance functions |
| `async` | tokio, futures | Async API without HTTP server |
| `server` | async + axum, tower, etc. | HTTP REST API server |
| `web-ui` | async + axum | Web-based admin UI |
| `metrics` | prometheus | Prometheus metrics export |
| `hybrid` | rust-stemmers | BM25 + RRF hybrid search |
| `encryption` | chacha20poly1305, hkdf, etc. | ChaCha20-Poly1305 at-rest encryption |
| `diskann` | bincode | DiskANN index support |
| `integrations` | uuid | LangChain / LlamaIndex integration |
| `observability` | — (no extra crates) | Telemetry, drift detection, anomaly detection, and profiling. Enables the observability dashboard, query tracing with configurable sample rates, data drift monitoring, anomaly alerts, and runtime profiling hooks. |

### Cloud Storage Flags

| Flag | Dependencies | Purpose |
|------|-------------|---------|
| `cloud-storage-s3` | aws-sdk-s3 | Amazon S3 backend |
| `cloud-storage-gcs` | google-cloud-storage | Google Cloud Storage backend |
| `cloud-storage-azure` | azure_storage | Azure Blob Storage backend |
| `cloud-storage` | all three above | All cloud backends |

### GPU Flags

| Flag | Dependencies | Purpose |
|------|-------------|---------|
| `gpu` | wide | GPU acceleration (base) |
| `gpu-cuda` | gpu + cudarc | NVIDIA CUDA backend |
| `gpu-metal` | gpu + metal, objc | Apple Metal backend |

### CDC / Streaming Flags

| Flag | Dependencies | Purpose |
|------|-------------|---------|
| `cdc-kafka` | rdkafka | Kafka CDC connector |
| `cdc-pulsar` | pulsar | Pulsar CDC connector |
| `cdc-postgres` | tokio-postgres | PostgreSQL CDC connector |
| `cdc-mongodb` | mongodb | MongoDB CDC connector |
| `cdc` | all four above | All CDC connectors |

### Binding Flags

| Flag | Dependencies | Purpose |
|------|-------------|---------|
| `python` | pyo3, pythonize | Python bindings |
| `wasm` | wasm-bindgen, js-sys | WebAssembly bindings |
| `uniffi-bindings` | uniffi | Swift/Kotlin bindings (UniFFI) |

### Meta Flags

| Flag | Includes | Purpose |
|------|----------|---------|
| `full` | server, metrics, hybrid, web-ui, embedding-providers, encryption, diskann, integrations, observability, experimental | All stable features — used for CI |
| `experimental` | — | Experimental/preview modules (APIs may change) |

### Unstable / Preview Flags

| Flag | Dependencies | Purpose |
|------|-------------|---------|
| `embeddings` | ort, ndarray, tokenizers | ONNX embedding inference (pre-release ort) |
| `embedding-providers` | async + reqwest | OpenAI/Cohere/Ollama embedding providers |
| `embedded-models` | — | Candle-based model runtime (mock backend) |
| `tui` | ratatui, crossterm | Terminal UI |
| `streaming-kafka` | cdc-kafka | Streaming ingestion via Kafka |

## Makefile and Cargo Alias Reconciliation

The project provides both Makefile targets and `.cargo/config.toml` aliases for common operations:

| Task | Makefile | Cargo Alias | Equivalent Command |
|------|----------|-------------|-------------------|
| Build (default) | `make build` | `cargo build` | `cargo build` |
| Build (all features) | `make build-all` | `cargo build-all` | `cargo build --features full` |
| Build (server + metrics) | — | `cargo build-quick` | `cargo build --features server,metrics` |
| Test (all features) | `make test` | `cargo t` | `cargo test --features full` |
| Test (unit only) | `make test-unit` | `cargo test-quick` | `cargo test --lib` |
| Lint (all features) | `make lint` | `cargo lint` | `cargo clippy --features full` |
| Fast feedback | `make quick` | — | fmt-check + lint + unit tests |
| Full pre-commit | `make check` | — | fmt-check + lint + all tests |

> **Tip**: `cargo t` is the shortest way to run the full test suite. Use `make quick` for the fastest pre-push check.

## Build Time Estimates

Approximate times on a modern laptop (M-series Mac or 8-core x86):

| Configuration | Clean Build | Incremental |
|---------------|-------------|-------------|
| `cargo build` (no flags) | ~30s | ~2-5s |
| `--features server` | ~45s | ~3-8s |
| `--features full` | ~90s | ~5-15s |
| `--release --features full` | ~3-5min | ~15-30s |

> These are rough estimates. Actual times depend on hardware, linker, and cache state. See `.cargo/config.toml` for linker optimization tips.
