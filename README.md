# Needle

[![CI](https://github.com/anthropics/needle/actions/workflows/ci.yml/badge.svg)](https://github.com/anthropics/needle/actions/workflows/ci.yml)
[![Security](https://github.com/anthropics/needle/actions/workflows/security.yml/badge.svg)](https://github.com/anthropics/needle/actions/workflows/security.yml)
[![codecov](https://codecov.io/gh/anthropics/needle/branch/main/graph/badge.svg)](https://codecov.io/gh/anthropics/needle)
[![Crates.io](https://img.shields.io/crates/v/needle.svg)](https://crates.io/crates/needle)
[![docs.rs](https://docs.rs/needle/badge.svg)](https://docs.rs/needle)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SQLite for Vectors** - An embedded vector database written in Rust.

Needle is a high-performance, zero-configuration vector database designed for AI applications. It provides fast approximate nearest neighbor (ANN) search with a single-file storage format.

> **New here?** Jump to the [Quickstart](QUICKSTART.md) — three copy-paste paths to a working setup.

## Get Started

### Try it (no Rust required)

```bash
docker compose --profile demo up -d --build
# Search the pre-loaded demo collection:
curl -X POST http://127.0.0.1:8080/collections/demo/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.3],"k":3}'
```

### Develop with it (Rust)

```bash
git clone https://github.com/anthropics/needle.git
cd needle
cargo run --example basic_usage
```

<details>
<summary><strong>More ways to run</strong></summary>

```bash
# One-command demo (builds server, seeds data, runs a search)
./scripts/quickstart.sh

# With custom port / debug logging
NEEDLE_PORT=9090 RUST_LOG=debug ./scripts/quickstart.sh

# Check your environment for prerequisites
./scripts/doctor.sh

# Run the server manually
cargo run --features server -- serve -a 127.0.0.1:8080
curl http://127.0.0.1:8080/health

# Task runner shortcuts (make works out of the box; just needs: cargo install just)
make demo
just demo

# Docker: run the image directly
docker run --rm -p 8080:8080 ghcr.io/anthropics/needle:latest

# Docker: build from source
docker compose -f docker-compose.yml -f docker-compose.source.yml up -d --build
```

</details>

## Performance

Benchmarks on 1M vectors (384 dimensions, M=16, ef_search=50):

| Operation | Latency (p50) | Latency (p99) | Throughput |
|-----------|---------------|---------------|------------|
| Single search | 3.2ms | 8.5ms | ~300 QPS |
| Batch search (100 queries) | 1.8ms/query | 4.2ms/query | ~3,000 QPS |
| Insert | 0.8ms | 2.1ms | ~1,200 ops/s |
| Filtered search (10% selectivity) | 4.5ms | 12ms | ~220 QPS |

Memory usage: ~1.7GB for 1M vectors (384 dims) with HNSW index.

## When to Use Needle

**Needle is a great fit for:**
- Embedded applications requiring vector search (desktop apps, mobile, edge devices)
- Single-node deployments up to ~50M vectors
- Projects that value SQLite-like simplicity (single file, zero config)
- RAG applications, semantic search, recommendation systems
- Prototyping and development with easy migration to production

**Consider alternatives when you need:**
- Billion-scale vector search (consider Milvus, Pinecone, Weaviate)
- Multi-region active-active replication
- Managed cloud service with SLAs
- Real-time streaming ingestion at >100K vectors/second

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/how-to-guides.md) | First database, search, and common tasks |
| [HTTP Quickstart](docs/http-quickstart.md) | Run the REST API and make your first requests |
| [RAG Quickstart](docs/rag-quickstart.md) | End-to-end RAG pipeline with OpenAI embeddings |
| [API Reference](docs/api-reference.md) | Complete method documentation |
| [Architecture](docs/architecture.md) | Internal design and data flow diagrams |
| [Index Selection Guide](docs/index-selection-guide.md) | HNSW vs IVF vs DiskANN decision guide |
| [Production Checklist](docs/production-checklist.md) | Pre-launch verification checklist |
| [Operations Guide](docs/OPERATIONS.md) | Monitoring, backup, and tuning |
| [Deployment Guide](docs/deployment.md) | Docker, Kubernetes, and cloud deployment |
| [Distributed Operations](docs/distributed-operations.md) | Sharding, replication, and clustering |
| [Examples](examples/README.md) | Runnable example commands |

## Features

### Core (Stable)
- **HNSW Index**: Sub-10ms queries using Hierarchical Navigable Small World graphs
- **SIMD Optimized**: Distance functions optimized for AVX2 (x86_64) and NEON (ARM)
- **Single-File Storage**: All data in one file, easy to backup and distribute
- **Metadata Filtering**: Filter search results by JSON metadata fields
- **Multiple Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan
- **Quantization**: Scalar (4x), Product (8-32x), and Binary (32x) compression
- **Batch Search**: Parallel search for multiple queries using Rayon
- **Multi-Language**: Rust, Python, JavaScript/WASM, Swift, and Kotlin bindings

### Advanced (Stable)
- **Hybrid Search**: BM25 + vector search with Reciprocal Rank Fusion (RRF)
- **IVF Index**: Inverted File Index for large-scale approximate search
- **Sparse Vectors**: TF-IDF and SPLADE support with inverted index
- **Multi-Vector (ColBERT)**: MaxSim search for token-level embeddings
- **Reranking**: Cross-encoder reranking with Cohere, HuggingFace, and custom providers

### Enterprise (Beta)
- **Encryption at Rest**: ChaCha20-Poly1305 authenticated encryption
- **RBAC**: Role-based access control with audit logging
- **Write-Ahead Log (WAL)**: Crash recovery and durability guarantees
- **Sharding**: Consistent hash ring for horizontal scaling
- **Multi-Tenancy**: Namespace isolation with access control
- **Raft Consensus**: Leader election and log replication for high availability
- **CRDT Support**: Conflict-free replicated data types for eventual consistency

### Experimental
- **GPU Acceleration**: CUDA/Metal/OpenCL support for distance computation *(scaffolding only - CPU fallback)*
- **Cloud Storage**: S3, Azure Blob, and GCS backends *(interface only - not production-ready)*

## Installation

### CLI / Server

```bash
# From crates.io (builds from source)
cargo install needle

# Or with pre-built binary (no compilation)
cargo binstall needle
```

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
needle = "0.1"
```

### Python

```bash
pip install needle-db
```

See [Python Installation](docs/python.md) for wheels vs source builds.

### Python (from source)

```bash
pip install maturin
maturin develop --features python
python -c "import needle; print('needle import ok')"
```

### JavaScript/WASM

```bash
npm install @anthropic/needle
```

For local SDK development, see [sdk/js/README.md](sdk/js/README.md).

## Quick Start

### Rust

```rust
use needle::{Database, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    // Create an in-memory database
    let db = Database::in_memory();

    // Create a collection for 384-dimensional vectors
    db.create_collection("documents", 384)?;

    // Get a reference to the collection
    let collection = db.collection("documents")?;

    // Insert vectors with metadata
    let embedding = vec![0.1; 384];
    collection.insert(
        "doc1",
        &embedding,
        Some(json!({"title": "Hello World", "category": "greeting"}))
    )?;

    // Search for similar vectors
    let query = vec![0.1; 384];
    let results = collection.search(&query, 10)?;

    for result in results {
        println!("ID: {}, Distance: {}", result.id, result.distance);
    }

    Ok(())
}
```

### Python

```python
import needle

# Create a collection
collection = needle.PyCollection("documents", 384, "cosine")

# Insert vectors
collection.insert("doc1", [0.1] * 384, {"title": "Hello"})

# Search
results = collection.search([0.1] * 384, k=10)
for result in results:
    print(f"ID: {result.id}, Distance: {result.distance}")

# Search with filter
results = collection.search_with_filter(
    [0.1] * 384,
    k=10,
    filter={"category": {"$eq": "greeting"}}
)
```

### JavaScript/WASM

```javascript
import { WasmCollection } from '@anthropic/needle';

// Create a collection
const collection = new WasmCollection("documents", 384, "cosine");

// Insert vectors
collection.insertWithObject("doc1", new Float32Array(384).fill(0.1), {
    title: "Hello World"
});

// Search
const results = collection.search(new Float32Array(384).fill(0.1), 10);
for (const result of results) {
    console.log(`ID: ${result.id}, Distance: ${result.distance}`);
}
```

## Persistence

```rust
use needle::Database;

// Open or create a database file
let mut db = Database::open("vectors.needle")?;

db.create_collection("my_collection", 128)?;

let collection = db.collection("my_collection")?;
// ... insert vectors ...

// Save changes to disk
db.save()?;
```

## Metadata Filtering

Needle supports MongoDB-style query syntax for filtering:

```rust
use needle::metadata::Filter;

// Simple equality
let filter = Filter::eq("category", "greeting");

// Numeric comparison
let filter = Filter::gt("score", 0.5);

// In array
let filter = Filter::is_in("status", vec![json!("active"), json!("pending")]);

// Combine with AND/OR
let filter = Filter::and(vec![
    Filter::eq("category", "greeting"),
    Filter::gt("priority", 0),
]);
```

JSON filter syntax:

```json
{
    "category": {"$eq": "greeting"},
    "score": {"$gte": 0.5},
    "$or": [
        {"status": "active"},
        {"status": "pending"}
    ]
}
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$contains`, `$and`, `$or`, `$not`

## Distance Functions

| Function | Use Case |
|----------|----------|
| `Cosine` | Normalized embeddings (default) |
| `Euclidean` | Spatial data, image features |
| `DotProduct` | When vectors are pre-normalized |
| `Manhattan` | Sparse data, L1 distance |

```rust
use needle::{CollectionConfig, DistanceFunction};

let config = CollectionConfig::new("my_collection", 128)
    .with_distance(DistanceFunction::Euclidean);
```

## HNSW Configuration

Fine-tune the index for your use case:

```rust
use needle::CollectionConfig;

let config = CollectionConfig::new("my_collection", 128)
    .with_m(16)              // Connections per node (default: 16)
    .with_ef_construction(200); // Build-time search width (default: 200)

// At query time
collection.set_ef_search(100);  // Query-time search width (default: 50)
```

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `M` | Graph connectivity | Memory vs recall |
| `ef_construction` | Build quality | Build time vs recall |
| `ef_search` | Query accuracy | Query time vs recall |

## Quantization

Reduce memory usage with quantization:

```rust
use needle::quantization::{ScalarQuantizer, ProductQuantizer, BinaryQuantizer};

// Scalar Quantization (4x compression, minimal quality loss)
let sq = ScalarQuantizer::train(&vectors);
let quantized = sq.quantize(&vector);

// Product Quantization (8-32x compression)
let pq = ProductQuantizer::train(&vectors, 8, 256);  // 8 subspaces, 256 centroids
let codes = pq.quantize(&vector);

// Binary Quantization (32x compression)
let bq = BinaryQuantizer::new();
let binary = bq.quantize(&vector);
```

## Batch Search

Search multiple queries in parallel:

```rust
let queries = vec![
    vec![0.1; 128],
    vec![0.2; 128],
    vec![0.3; 128],
];

// Parallel search
let results = collection.batch_search(&queries, 10)?;
```

## Building Language Bindings

### Python

```bash
cargo build --release --features python
maturin build --features python
```

### WebAssembly

```bash
cargo build --release --target wasm32-unknown-unknown --features wasm
wasm-pack build --features wasm
```

### Swift/Kotlin (UniFFI)

```bash
cargo build --release --features uniffi-bindings
```

## Command Line Interface

Needle includes a CLI for database management:

```bash
# Run without installing
cargo run --bin needle -- --help

# Or build once and run the binary
cargo build --release
./target/release/needle --help

# Optional: install to PATH
cargo install --path .

# Create a new database
./target/release/needle create mydata.needle

# Create a collection
./target/release/needle create-collection mydata.needle -n documents -d 384 --distance cosine

# Show database info
./target/release/needle info mydata.needle

# List collections
./target/release/needle collections mydata.needle

# Insert vectors from stdin (JSON format)
echo '{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"Hello"}}' | ./target/release/needle insert mydata.needle -c documents

# Get a vector by ID
./target/release/needle get mydata.needle -c documents -i doc1

# Search for similar vectors
./target/release/needle search mydata.needle -c documents -q "0.1,0.2,0.3" -k 10

# Delete a vector
./target/release/needle delete mydata.needle -c documents -i doc1

# Export collection to JSON
./target/release/needle export mydata.needle -c documents

# Compact database (remove deleted vectors)
./target/release/needle compact mydata.needle

# Show collection statistics
./target/release/needle stats mydata.needle -c documents
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new database file |
| `create-collection` | Create a new collection with specified dimensions |
| `info` | Show database information |
| `collections` | List all collections |
| `stats` | Show collection statistics (including deleted vectors) |
| `insert` | Insert vectors from stdin (JSON format, one per line) |
| `get` | Retrieve a vector by ID |
| `search` | Search for similar vectors |
| `delete` | Delete a vector by ID |
| `export` | Export collection to JSON |
| `import` | Import vectors from JSON file |
| `count` | Count vectors in a collection |
| `clear` | Delete all vectors from a collection |
| `compact` | Remove deleted vectors and reclaim space |
| `serve` | Start HTTP REST API server (requires `server` feature) |
| `tune` | Auto-tune HNSW parameters for a workload |

## Feature Flags

Needle uses Cargo feature flags to enable optional functionality. By default, no features are enabled for minimal compile time.

| Flag | Description | Stability |
|------|-------------|-----------|
| `simd` | SIMD-optimized distance functions (AVX2, NEON) | Stable |
| `server` | HTTP REST API server (Axum-based) | Stable |
| `web-ui` | Web-based admin UI | Stable |
| `metrics` | Prometheus metrics endpoint | Stable |
| `hybrid` | BM25 + vector hybrid search with RRF fusion | Stable |
| `encryption` | ChaCha20-Poly1305 authenticated encryption at rest | Stable |
| `diskann` | DiskANN index for large-scale on-disk search | Stable |
| `integrations` | LangChain / LlamaIndex adapters | Stable |
| `embeddings` | ONNX embedding inference | **Unstable** (pre-release dependency) |
| `embedding-providers` | OpenAI, Cohere, Ollama embedding providers | Stable |
| `tui` | Terminal user interface | Stable |
| `full` | All stable features (server + web-ui + metrics + hybrid + encryption + diskann + integrations + embedding-providers) | Stable |
| `python` | Python bindings via PyO3 | Stable |
| `wasm` | WebAssembly bindings | Stable |
| `uniffi-bindings` | Swift/Kotlin bindings via UniFFI | Stable |

> **Note**: The `embeddings` feature uses a pre-release version of the `ort` crate (ONNX Runtime). API stability is not guaranteed for this feature until `ort` reaches a stable release.

> **Note**: Some enterprise features are in development:
> - **Cloud Storage** (S3, GCS, Azure): Interface defined but implementations are mock/stub only. Not recommended for production use.
> - **GPU Acceleration**: Scaffolding only, currently falls back to CPU for all operations.

Build with features:

```bash
# Build with all stable features
cargo build --features full

# Build with specific features
cargo build --features server,metrics

# Build with all features including unstable
cargo build --features full,embeddings
```

## Testing

```bash
# Fast feedback (format check + lint + unit tests)
make quick          # or: just quick

# Full checks (format check + lint + full tests)
make check          # or: just check

# Cargo fallback
cargo test --features full
```

## Benchmarks

Run benchmarks:

```bash
cargo bench
```

## Performance Tips

1. **Use batch operations** for inserting many vectors
2. **Tune `ef_search`** based on accuracy needs (higher = more accurate but slower)
3. **Enable SIMD** with `--features simd` for optimized distance calculations
4. **Use appropriate quantization** for memory-constrained environments
5. **Pre-normalize vectors** when using cosine distance

## File Format

Needle uses a custom single-file format (`.needle`):

```
+------------------+
| Header (4KB)     | Magic, version, offsets, checksums
+------------------+
| Index Pages      | HNSW graph structure
+------------------+
| Vector Pages     | Raw vector data
+------------------+
| Metadata Pages   | JSON metadata and ID mappings
+------------------+
```

## License

MIT License

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## Acknowledgments

- HNSW algorithm: Malkov & Yashunin (2016)
- Inspired by SQLite's simplicity and ChromaDB's API design
