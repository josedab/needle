# Needle

**SQLite for Vectors** - An embedded vector database written in Rust.

Needle is a high-performance, zero-configuration vector database designed for AI applications. It provides fast approximate nearest neighbor (ANN) search with a single-file storage format.

## Features

### Core
- **HNSW Index**: Sub-10ms queries using Hierarchical Navigable Small World graphs
- **SIMD Optimized**: Distance functions optimized for AVX2 (x86_64) and NEON (ARM)
- **Single-File Storage**: All data in one file, easy to backup and distribute
- **Metadata Filtering**: Filter search results by JSON metadata fields
- **Multiple Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan
- **Quantization**: Scalar (4x), Product (8-32x), and Binary (32x) compression
- **Batch Search**: Parallel search for multiple queries using Rayon
- **Multi-Language**: Rust, Python, JavaScript/WASM, Swift, and Kotlin bindings

### Advanced
- **Hybrid Search**: BM25 + vector search with Reciprocal Rank Fusion (RRF)
- **IVF Index**: Inverted File Index for large-scale approximate search
- **Sparse Vectors**: TF-IDF and SPLADE support with inverted index
- **Multi-Vector (ColBERT)**: MaxSim search for token-level embeddings
- **Reranking**: Cross-encoder reranking with Cohere, HuggingFace, and custom providers
- **GPU Acceleration**: CUDA/OpenCL support for distance computation

### Enterprise
- **Encryption at Rest**: ChaCha20-Poly1305 authenticated encryption
- **RBAC**: Role-based access control with audit logging
- **Write-Ahead Log (WAL)**: Crash recovery and durability guarantees
- **Cloud Storage**: S3, Azure Blob, and GCS backends with caching
- **Sharding**: Consistent hash ring for horizontal scaling
- **Multi-Tenancy**: Namespace isolation with access control
- **Raft Consensus**: Leader election and log replication for high availability
- **CRDT Support**: Conflict-free replicated data types for eventual consistency

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
needle = "0.1"
```

### Python

```bash
pip install needle-db
```

### JavaScript/WASM

```bash
npm install @anthropic/needle
```

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
# Build the CLI
cargo build --release

# Create a new database
needle create mydata.needle

# Create a collection
needle create-collection mydata.needle -n documents -d 384 --distance cosine

# Show database info
needle info mydata.needle

# List collections
needle collections mydata.needle

# Insert vectors from stdin (JSON format)
echo '{"id":"doc1","vector":[0.1,0.2,...],"metadata":{"title":"Hello"}}' | needle insert mydata.needle -c documents

# Get a vector by ID
needle get mydata.needle -c documents -i doc1

# Search for similar vectors
needle search mydata.needle -c documents -q "0.1,0.2,0.3,..." -k 10

# Delete a vector
needle delete mydata.needle -c documents -i doc1

# Export collection to JSON
needle export mydata.needle -c documents

# Compact database (remove deleted vectors)
needle compact mydata.needle

# Show collection statistics
needle stats mydata.needle -c documents
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
| `embeddings` | ONNX embedding inference | **Unstable** (pre-release dependency) |
| `embedding-providers` | OpenAI, Cohere, Ollama embedding providers | Stable |
| `tui` | Terminal user interface | Stable |
| `full` | All stable features (server + web-ui + metrics + hybrid + embedding-providers) | Stable |
| `python` | Python bindings via PyO3 | Stable |
| `wasm` | WebAssembly bindings | Stable |
| `uniffi-bindings` | Swift/Kotlin bindings via UniFFI | Stable |

> **Note**: The `embeddings` feature uses a pre-release version of the `ort` crate (ONNX Runtime). API stability is not guaranteed for this feature until `ort` reaches a stable release.

Build with features:

```bash
# Build with all stable features
cargo build --features full

# Build with specific features
cargo build --features server,metrics

# Build with all features including unstable
cargo build --features full,embeddings
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
