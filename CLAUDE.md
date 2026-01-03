# CLAUDE.md - Needle Vector Database

## Project Overview

Needle is an embedded vector database written in Rust, designed as "SQLite for vectors". It provides high-performance approximate nearest neighbor (ANN) search with HNSW indexing, single-file storage, and zero configuration.

### Key Features

- HNSW index for sub-10ms approximate nearest neighbor search
- Single-file storage format (easy backup/distribution)
- Hybrid search (BM25 + vector + RRF fusion)
- HTTP REST API server mode
- Prometheus metrics for observability
- Auto-tuning of HNSW parameters
- Sparse vector support (TF-IDF, SPLADE)
- Multi-vector ColBERT-style retrieval
- ONNX embedding inference (optional)

## Quick Commands

```bash
# Build
cargo build
cargo build --release
cargo build --features full     # Server + metrics + hybrid search

# Test
cargo test                    # Run all tests
cargo test --lib              # Unit tests only
cargo test --features full    # Test with all features
cargo test --test property_tests  # Property-based tests

# Lint & Format
cargo clippy                  # Linting
cargo clippy --features full  # Lint with all features
cargo fmt                     # Format code

# Benchmarks
cargo bench

# Run CLI
cargo run -- <command>
cargo run -- info mydb.needle
cargo run -- create-collection mydb.needle -n docs -d 384

# Run HTTP Server
cargo run --features server -- serve -a 127.0.0.1:8080 -d mydb.needle
```

## Architecture

```
src/
├── lib.rs          # Library entry, re-exports public API
├── main.rs         # CLI application
├── collection.rs   # Collection: vectors + metadata + index
├── database.rs     # Database: multi-collection management, persistence
├── hnsw.rs         # HNSW index implementation
├── distance.rs     # Distance functions (Cosine, Euclidean, Dot, Manhattan)
├── metadata.rs     # Metadata storage and filtering (with Filter::parse)
├── storage.rs      # File I/O, mmap, vector storage
├── quantization.rs # Scalar, Product, Binary quantization
├── sparse.rs       # Sparse vector support
├── multivec.rs     # Multi-vector (ColBERT) support
├── tuning.rs       # Auto-tuning HNSW parameters
├── error.rs        # Error types
├── hybrid.rs       # BM25 + RRF hybrid search (feature: hybrid)
├── server.rs       # HTTP REST API (feature: server)
├── metrics.rs      # Prometheus metrics (feature: metrics)
├── embeddings.rs   # ONNX embedding inference (feature: embeddings)
├── python.rs       # Python bindings (feature: python)
├── wasm.rs         # WASM bindings (feature: wasm)
└── uniffi_bindings.rs  # Swift/Kotlin bindings (feature: uniffi-bindings)

tests/
└── property_tests.rs  # Proptest-based property tests

benches/
└── benchmarks.rs      # Criterion benchmarks
```

## Key Types

### Core Types
- **`Database`** - Main entry point, manages collections and persistence
- **`Collection`** - Holds vectors, HNSW index, and metadata for one collection
- **`CollectionRef`** - Thread-safe reference to a collection via Database
- **`HnswIndex`** - Hierarchical Navigable Small World graph for ANN search
- **`Filter`** - MongoDB-style metadata query filters
- **`SearchResult`** - Search result with id, distance, and metadata

### Advanced Types
- **`SparseVector`** - Sparse vector representation for TF-IDF/SPLADE
- **`SparseIndex`** - Inverted index for sparse vector search
- **`MultiVector`** - ColBERT-style multi-vector documents
- **`MultiVectorIndex`** - MaxSim search for multi-vector retrieval
- **`Bm25Index`** - BM25 text index for hybrid search
- **`TuningResult`** - Auto-tuned HNSW configuration

## Code Patterns

### Error Handling
```rust
use crate::error::{NeedleError, Result};

// Use Result<T> (alias for std::result::Result<T, NeedleError>)
pub fn some_operation() -> Result<()> {
    // Use ? for propagation
    self.validate()?;
    Ok(())
}
```

### Adding New Collection Methods
1. Add method to `Collection` in `src/collection.rs`
2. Add internal method to `Database` (e.g., `fn foo_internal(...)`)
3. Add public method to `CollectionRef` that calls the internal method
4. Export in `src/lib.rs` if needed

### Thread Safety
- `Database` uses `parking_lot::RwLock` for interior mutability
- `CollectionRef` provides safe concurrent access
- Read operations take read locks, writes take write locks

### Filter Parsing (MongoDB-style)
```rust
use crate::metadata::Filter;

// Parse JSON filter
let filter = Filter::parse(&json!({
    "category": "books",
    "price": { "$lt": 50 }
}))?;

// Supported operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or, $not
```

## Testing Guidelines

- Unit tests go in `#[cfg(test)] mod tests` within each module
- Integration tests go in `tests/`
- Property-based tests use `proptest` crate
- Use `Database::in_memory()` for tests (no file I/O)
- Random vectors: `(0..dim).map(|_| rng.gen::<f32>()).collect()`

## CLI Commands

| Command | Description |
|---------|-------------|
| `create` | Create new database |
| `create-collection` | Create collection with dimensions |
| `info` | Show database info |
| `collections` | List collections |
| `stats` | Collection statistics |
| `insert` | Insert from stdin (JSON) |
| `get` | Get vector by ID |
| `search` | Search similar vectors |
| `delete` | Delete by ID |
| `export` | Export to JSON |
| `import` | Import from JSON |
| `count` | Count vectors |
| `clear` | Delete all vectors |
| `compact` | Remove deleted vectors |
| `serve` | Start HTTP server (feature: server) |
| `tune` | Auto-tune HNSW parameters |

## Feature Flags

```toml
[features]
default = []
simd = []                       # SIMD-optimized distance functions
server = ["axum", "tokio", ...] # HTTP REST API server
metrics = ["prometheus"]        # Prometheus metrics
hybrid = ["rust-stemmers"]      # BM25 hybrid search
embeddings = ["ort", "ndarray", "tokenizers"]  # ONNX embeddings
full = ["server", "metrics", "hybrid"]  # All non-binding features
python = ["pyo3", "pythonize"]  # Python bindings
wasm = ["wasm-bindgen", ...]    # WebAssembly bindings
uniffi-bindings = ["uniffi"]    # Swift/Kotlin bindings
```

## HTTP REST API (feature: server)

```bash
# Start server
cargo run --features server -- serve -a 127.0.0.1:8080

# Endpoints
GET  /health                          # Health check
GET  /collections                     # List collections
POST /collections                     # Create collection
GET  /collections/:name               # Get collection info
DELETE /collections/:name             # Delete collection
POST /collections/:name/vectors       # Insert vectors
GET  /collections/:name/vectors/:id   # Get vector
DELETE /collections/:name/vectors/:id # Delete vector
POST /collections/:name/search        # Search (with optional explain)
POST /collections/:name/compact       # Compact collection
GET  /collections/:name/export        # Export collection
POST /save                            # Save database
```

## Hybrid Search (feature: hybrid)

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

// Create BM25 index
let mut bm25 = Bm25Index::default();
bm25.index_document("doc1", "machine learning and AI");

// Search
let bm25_results = bm25.search("machine learning", 10);

// Fuse with vector results
let hybrid = reciprocal_rank_fusion(&vector_results, &bm25_results, &RrfConfig::default(), 10);
```

## Auto-Tuning

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(100_000, 384)
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(500 * 1024 * 1024); // 500MB

let result = auto_tune(&constraints);
// Use result.config for collection creation
```

## Performance Notes

- HNSW parameters: `M=16`, `ef_construction=200`, `ef_search=50` (defaults)
- Higher `M` = better recall, more memory
- Higher `ef_search` = better recall, slower queries
- Use `batch_search` for multiple queries (parallel via Rayon)
- Call `compact()` after many deletions to reclaim space
- Mmap enabled for files > 10MB (`MMAP_THRESHOLD` in storage.rs)
- Use `auto_tune()` to optimize parameters for your workload

## Common Tasks

### Add a new error type
Edit `src/error.rs`:
```rust
#[derive(Error, Debug)]
pub enum NeedleError {
    // ... existing variants
    #[error("New error: {0}")]
    NewError(String),
}
```

### Add a new distance function
1. Add variant to `DistanceFunction` enum in `src/distance.rs`
2. Implement `compute()` method
3. Add SIMD version if `simd` feature is enabled

### Add a new CLI command
1. Add variant to `Commands` enum in `src/main.rs`
2. Add match arm in `main()`
3. Implement `fn command_name(...) -> Result<()>`

### Add a new REST endpoint
1. Add handler function in `src/server.rs`
2. Register route in `create_router()`
3. Define request/response types as needed

## Dependencies

Key crates:
- `thiserror` - Error derive macro
- `serde` / `serde_json` - Serialization
- `parking_lot` - Fast RwLock
- `rayon` - Parallel iteration
- `ordered-float` - OrderedFloat for BinaryHeap
- `memmap2` - Memory-mapped files
- `clap` - CLI argument parsing
- `axum` - HTTP server (optional)
- `prometheus` - Metrics (optional)
- `rust-stemmers` - Text stemming (optional)
- `ort` - ONNX Runtime (optional)
- `proptest` - Property-based testing (dev)
- `criterion` - Benchmarking (dev)
