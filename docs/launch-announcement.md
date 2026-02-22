# Introducing Needle: SQLite for Vectors

**TL;DR**: Needle is an embedded vector database written in Rust. Single-file storage, zero configuration, sub-10ms search, 5 language bindings. Think SQLite, but for vector similarity search.

## Why Another Vector Database?

Every vector database today requires a server. Qdrant, Milvus, Weaviate, Pinecone — they're all client-server architectures. If you want to add semantic search to your app, you need to deploy and manage infrastructure.

We asked: **what if vector search was as easy as SQLite?**

```rust
use needle::{Database, Filter};
use serde_json::json;

let db = Database::open("my_app.needle")?;
db.create_collection("docs", 384)?;

let col = db.collection("docs")?;
col.insert("doc1", &embedding, Some(json!({"type": "article"})))?;

let results = col.search(&query_embedding, 10)?;
```

No server. No Docker. No configuration. Just a library and a file.

## Performance

On 1M vectors (384 dimensions):

| Operation | Latency (p50) | Throughput |
|-----------|:---:|:---:|
| Search | 3.2ms | ~300 QPS |
| Batch search (100) | 1.8ms/query | ~3,000 QPS |
| Insert | 0.8ms | ~1,200 ops/s |
| Filtered search | 4.5ms | ~220 QPS |

SIMD-optimized (AVX2/NEON) distance functions. HNSW index with auto-tuning.

## What's Included

- **15 index types**: HNSW, IVF, DiskANN, sparse vectors (SPLADE), multi-vector (ColBERT), graph-vector fusion
- **Hybrid search**: BM25 + vector with Reciprocal Rank Fusion
- **Metadata filtering**: MongoDB-style queries ($eq, $gt, $in, $and, $or)
- **Quantization**: Scalar (4×), Product (8-32×), Binary (32×) compression
- **Enterprise**: Encryption at rest, RBAC, WAL, Raft consensus, multi-tenancy
- **Bindings**: Python, JavaScript/WASM, Swift, Kotlin
- **Server mode**: REST API when you need it
- **MCP support**: AI agent memory via Model Context Protocol

## Get Started

```bash
# Rust
cargo add needle

# Python
pip install needle-db

# Try it
cargo run --example quickstart
```

## Links

- **GitHub**: https://github.com/anthropics/needle
- **Docs**: https://docs.rs/needle
- **Quickstart**: https://github.com/anthropics/needle/blob/main/QUICKSTART.md

We'd love your feedback. Star the repo, try the quickstart, file issues. We're looking for contributors — check out the [good first issues](https://github.com/anthropics/needle/issues?q=label%3A%22good+first+issue%22).

---

*Needle is MIT-licensed. Built with ❤️ in Rust.*
