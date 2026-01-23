---
sidebar_position: 2
---

# Feature Flags

Needle uses Cargo feature flags for optional functionality. This allows you to include only what you need, minimizing binary size and compile time.

## Available Features

| Feature | Description | Dependencies Added |
|---------|-------------|-------------------|
| `simd` | SIMD-optimized distance functions | None |
| `server` | HTTP REST API server | axum, tokio, tower |
| `metrics` | Prometheus metrics | prometheus |
| `hybrid` | BM25 hybrid search | rust-stemmers |
| `embeddings` | ONNX embedding inference | ort, ndarray, tokenizers |
| `full` | server + metrics + hybrid | (combined) |
| `python` | Python bindings | pyo3 |
| `wasm` | WebAssembly bindings | wasm-bindgen |
| `uniffi-bindings` | Swift/Kotlin bindings | uniffi |

## Configuration

### Default (No Features)

```toml
[dependencies]
needle = "0.1"
```

Includes only the core vector database functionality:
- Database and collection management
- HNSW indexing
- Vector search
- Metadata filtering
- Quantization

### Common Configurations

#### Server Application

```toml
[dependencies]
needle = { version = "0.1", features = ["server", "metrics"] }
```

#### High-Performance Search

```toml
[dependencies]
needle = { version = "0.1", features = ["simd"] }
```

#### Full-Featured

```toml
[dependencies]
needle = { version = "0.1", features = ["full"] }
```

This is equivalent to:
```toml
[dependencies]
needle = { version = "0.1", features = ["server", "metrics", "hybrid"] }
```

#### RAG Application

```toml
[dependencies]
needle = { version = "0.1", features = ["hybrid", "simd"] }
```

## Feature Details

### simd

Enables SIMD (Single Instruction, Multiple Data) optimizations for distance calculations.

```rust
// No code changes needed - automatically uses SIMD when available
let results = collection.search(&query, 10, None)?;
```

**Performance impact:**
- 2-4x faster on x86_64 with AVX2
- 1.5-2x faster on ARM with NEON

**Requirements:**
- x86_64: AVX2 support (most CPUs since 2013)
- ARM: NEON support (all ARMv7+ and ARM64)

### server

Enables the built-in HTTP REST API server.

```rust
use needle::{Database, server};

let db = Database::open("vectors.needle")?;
server::run("0.0.0.0:8080", db).await?;
```

**Adds commands:**
```bash
needle serve -a 0.0.0.0:8080 -d vectors.needle
```

**Dependencies:** axum, tokio, tower, hyper

### metrics

Enables Prometheus metrics export.

```rust
use needle::metrics;

// Start metrics server on separate port
metrics::start_server("0.0.0.0:9090")?;
```

**Available metrics:**
- `needle_search_latency_seconds`
- `needle_insert_latency_seconds`
- `needle_vectors_total`
- `needle_collections_total`
- `needle_memory_bytes`

### hybrid

Enables BM25 text search and hybrid search capabilities.

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

let mut bm25 = Bm25Index::default();
bm25.index_document("doc1", "Hello world");

let bm25_results = bm25.search("hello", 10);
let fused = reciprocal_rank_fusion(&vector_results, &bm25_results, &RrfConfig::default(), 10);
```

**Features:**
- BM25 text indexing
- Stemming (English and other languages)
- Reciprocal Rank Fusion

### embeddings

Enables ONNX Runtime embedding inference.

```rust
use needle::EmbeddingModel;

let model = EmbeddingModel::load("all-MiniLM-L6-v2")?;
let embedding = model.encode("Hello world")?;
```

**Supported models:**
- all-MiniLM-L6-v2
- BGE-small-en-v1.5
- Custom ONNX models

**Dependencies:** ort (ONNX Runtime), ndarray, tokenizers

**Note:** Adds significant binary size (~50MB with runtime).

### python

Enables Python bindings via PyO3.

```bash
# Build Python wheel
maturin build --features python
pip install target/wheels/needle-*.whl
```

```python
from needle import Database, DistanceFunction
db = Database.open("vectors.needle")
```

### wasm

Enables WebAssembly bindings.

```bash
# Build WASM package
wasm-pack build --target web --features wasm
```

```javascript
import init, { Database } from './needle.js';
await init();
const db = Database.inMemory();
```

### uniffi-bindings

Enables Swift and Kotlin bindings via UniFFI.

```bash
# Generate bindings
cargo build --features uniffi-bindings
```

## Build Size Impact

Approximate binary sizes (release build, stripped):

| Configuration | Size |
|---------------|------|
| Default | 4 MB |
| + simd | 4.5 MB |
| + server | 8 MB |
| + metrics | 5 MB |
| + hybrid | 5 MB |
| full | 10 MB |
| + embeddings | 60 MB |

## Compile Time Impact

Approximate compile times (release, cold cache):

| Configuration | Time |
|---------------|------|
| Default | 30s |
| + simd | 35s |
| + server | 90s |
| full | 120s |
| + embeddings | 180s |

## Conditional Compilation

### In Your Code

```rust
#[cfg(feature = "hybrid")]
use needle::{Bm25Index, reciprocal_rank_fusion};

fn search(query: &str, query_embedding: &[f32]) -> Vec<SearchResult> {
    let vector_results = collection.search(query_embedding, 10, None)?;

    #[cfg(feature = "hybrid")]
    {
        let bm25_results = bm25.search(query, 10);
        reciprocal_rank_fusion(&vector_results, &bm25_results, &RrfConfig::default(), 10)
    }

    #[cfg(not(feature = "hybrid"))]
    {
        vector_results
    }
}
```

### Feature Detection at Runtime

```rust
fn print_features() {
    println!("Needle features:");
    println!("  SIMD: {}", cfg!(feature = "simd"));
    println!("  Server: {}", cfg!(feature = "server"));
    println!("  Metrics: {}", cfg!(feature = "metrics"));
    println!("  Hybrid: {}", cfg!(feature = "hybrid"));
    println!("  Embeddings: {}", cfg!(feature = "embeddings"));
}
```

## Recommendations

### Embedded Library

```toml
# Minimal footprint
needle = { version = "0.1", features = ["simd"] }
```

### Microservice

```toml
# HTTP API with observability
needle = { version = "0.1", features = ["server", "metrics", "simd"] }
```

### RAG Pipeline

```toml
# Hybrid search for better retrieval
needle = { version = "0.1", features = ["hybrid", "simd"] }
```

### End-to-End ML

```toml
# Include embedding inference
needle = { version = "0.1", features = ["full", "embeddings", "simd"] }
```

### Multi-Platform Library

```toml
# Different features per target
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
needle = { version = "0.1", features = ["simd"] }

[target.'cfg(target_arch = "wasm32")'.dependencies]
needle = { version = "0.1", features = ["wasm"] }
```

## Next Steps

- [HNSW Tuning](/docs/configuration/hnsw-tuning)
- [API Reference](/docs/api-reference)
- [Production Deployment](/docs/guides/production)
