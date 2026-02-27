# Needle Documentation

Welcome to the Needle documentation. Needle is an embedded vector database written in Rust, designed as "SQLite for vectors".

## Quick Links

| I want to... | Go to... |
|--------------|----------|
| Get started quickly | [Getting Started](#getting-started) |
| Run the HTTP API | [HTTP Quickstart](http-quickstart.md) |
| Install Python bindings | [Python Installation](python.md) |
| Understand the API | [API Reference](api-reference.md) |
| Deploy to production | [Production Checklist](production-checklist.md) |
| Choose an index type | [Index Selection Guide](index-selection-guide.md) |
| Set up a distributed cluster | [Distributed Operations](distributed-operations.md) |
| Migrate from another database | [Migration Guide](migration-upgrade-guide.md) |

---

## Getting Started

### Installation

```toml
# Cargo.toml
[dependencies]
needle = "0.1"
```

### Python (from source)

```bash
pip install maturin
maturin develop --features python
python -c "import needle; print('needle import ok')"
```

### Quick Example

```rust
use needle::{Database, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    // Create in-memory database
    let db = Database::in_memory();

    // Create collection (384 dimensions = common embedding size)
    db.create_collection("documents", 384)?;
    let coll = db.collection("documents")?;

    // Insert vector with metadata
    let embedding = vec![0.1; 384];
    coll.insert("doc1", &embedding, Some(json!({"title": "Hello World"})))?;

    // Search
    let results = coll.search(&embedding, 10)?;

    Ok(())
}
```

For more examples, see [How-To Guides](how-to-guides.md).

---

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [API Reference](api-reference.md) | Complete API documentation with method signatures and examples |
| [API Stability Policy](api-stability.md) | Stability tiers (Stable, Beta, Experimental) and semver guarantees |
| [NeedleQL Guide](needleql.md) | SQL-like query language for vector operations |
| [Architecture](architecture.md) | Internal design, data flow diagrams, and module relationships |
| [Services Architecture](ARCHITECTURE_SERVICES.md) | Service-layer modules in `src/services/` |
| [Benchmarks](benchmarks.md) | Performance comparisons for vector search operations |
| [How-To Guides](how-to-guides.md) | Practical tutorials for common tasks |
| [HTTP Quickstart](http-quickstart.md) | Zero-to-search walkthrough for the HTTP REST API |
| [RAG Quickstart](rag-quickstart.md) | Build a RAG pipeline with Needle and OpenAI in 5 minutes |

### Operations

| Document | Description |
|----------|-------------|
| [Production Checklist](production-checklist.md) | Pre-launch verification checklist |
| [Operations Guide](OPERATIONS.md) | Day-to-day operations, monitoring, and maintenance |
| [Deployment Guide](deployment.md) | Docker, Kubernetes, and cloud deployment options |

### Advanced Topics

| Document | Description |
|----------|-------------|
| [Index Selection Guide](index-selection-guide.md) | HNSW vs IVF vs DiskANN decision guide |
| [Feature Flags Guide](feature-flags.md) | Cargo feature flags decision guide |
| [Distributed Operations](distributed-operations.md) | Sharding, Raft replication, and clustering |
| [Migration Guide](migration-upgrade-guide.md) | Comprehensive version upgrades, index changes, and external DB migration |
| [Migration Quick Reference](MIGRATION.md) | Breaking changes and deprecated APIs between specific versions |
| [MCP Integration](mcp-integration.md) | Model Context Protocol server for AI agent access |
| [WASM Guide](WASM_GUIDE.md) | WebAssembly integration guide |
| [Experimental Triage](experimental-triage.md) | Experimental module status and maturity report |
| [Launch Announcement](launch-announcement.md) | Needle v0.1.0 launch announcement |

### Developer Resources

| Document | Description |
|----------|-------------|
| [Language Bindings](language-bindings.md) | Overview of all language bindings (Python, JS, WASM, Swift, Kotlin) |
| [Examples](../examples/README.md) | Runnable example commands |
| [Python SDK](../python/README.md) | Python client installation and usage |
| [JavaScript SDK](../sdk/js/README.md) | SDK build and usage notes |

### Framework Integrations

| Document | Description |
|----------|-------------|
| [LangChain Integration](../python/needle_langchain/README.md) | LangChain vector store wrapper (reference implementation) |
| [LlamaIndex Integration](../python/needle_llamaindex/README.md) | LlamaIndex vector store wrapper (reference implementation) |

### Reference

| Document | Description |
|----------|-------------|
| [Architecture Decision Records](adr/) | Design decisions and rationale |
| [Governance](GOVERNANCE.md) | Project governance model and decision-making process |
| [Support](SUPPORT.md) | How to get help, report bugs, and request features |
| [Test Coverage TODOs](TODO-test-coverage.md) | Consolidated test coverage gaps and priorities |
| [CLAUDE.md](../CLAUDE.md) | AI assistant context and codebase overview |

---

## Feature Overview

### Core Features (Stable)

- **HNSW Index**: Sub-10ms approximate nearest neighbor search
- **Single-File Storage**: All data in one `.needle` file
- **Metadata Filtering**: MongoDB-style query filters
- **Multiple Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan
- **Quantization**: Scalar (4x), Product (8-32x), Binary (32x) compression
- **Multi-Language**: Rust, Python, JavaScript/WASM, Swift, Kotlin

### Advanced Features (Stable)

- **Hybrid Search**: BM25 + vector search with RRF fusion
- **IVF Index**: Inverted File Index for large datasets
- **Sparse Vectors**: TF-IDF and SPLADE support
- **Multi-Vector**: ColBERT-style MaxSim search

### Enterprise Features (Beta)

- **Encryption at Rest**: ChaCha20-Poly1305
- **RBAC**: Role-based access control
- **WAL**: Write-ahead logging for durability
- **Sharding**: Consistent hashing for horizontal scaling
- **Raft Consensus**: High availability replication

---

## Common Tasks

### HNSW Tuning

```rust
use needle::{CollectionConfig, HnswConfig};

let hnsw = HnswConfig::default()
    .m(16)                // Graph connectivity
    .ef_construction(200) // Build quality
    .ef_search(50);       // Query accuracy

let config = CollectionConfig::new("my_collection", 384)
    .with_hnsw_config(hnsw);
```

See [HNSW Parameter Tuning](how-to-guides.md#hnsw-parameter-tuning) for details.

### Filtered Search

```rust
use needle::Filter;

// Simple equality
let filter = Filter::eq("category", "tech");

// Numeric range
let filter = Filter::and(vec![
    Filter::gte("price", 10.0),
    Filter::lt("price", 100.0),
]);

// Search with filter
let results = coll.search_with_filter(&query, 10, &filter)?;
```

See [Metadata Filtering Patterns](how-to-guides.md#metadata-filtering-patterns) for more.

### Hybrid Search

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

// Vector search
let vector_results = coll.search(&query_embedding, 100)?;

// BM25 text search
let bm25_results = bm25.search("machine learning", 100);

// Fuse results
let hybrid = reciprocal_rank_fusion(&vector_results, &bm25_results, &RrfConfig::default(), 10);
```

See [Hybrid Search Setup](how-to-guides.md#hybrid-search-setup) for details.

---

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/anthropics/needle/issues)
- **API Docs**: [docs.rs/needle](https://docs.rs/needle)
- **Examples**: See the `examples/` directory in the repository

---

## Version Information

- **Current Version**: 0.1.x
- **Minimum Rust Version**: 1.85+
- **File Format Version**: 1

See [Migration Guide](migration-upgrade-guide.md) for upgrade information.
