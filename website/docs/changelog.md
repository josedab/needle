---
sidebar_position: 12
---

# Changelog

All notable changes to Needle are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- IVF (Inverted File) index support for large-scale datasets
- DiskANN index for memory-efficient billion-scale search
- RBAC (Role-Based Access Control) with audit logging

### Changed
- Improved HNSW build performance by 20%

### Fixed
- Memory leak in batch search operations

### Documentation
- Added comprehensive [Troubleshooting Guide](/docs/troubleshooting) with solutions for common issues
- Added [Performance & Benchmarks](/docs/benchmarks) page with methodology and comparison data
- Added [Migration Guide](/docs/migration) with step-by-step instructions for Chroma, Qdrant, Pinecone, pgvector, FAISS, and Milvus
- Enhanced landing page with architecture diagram and multi-language install tabs
- Added Mermaid diagrams to RAG and Hybrid Search guides
- Improved cross-linking between documentation pages
- Added [Architecture Deep-Dive](/docs/architecture) with system overview, data flow, and storage format diagrams
- Added [Index Selection Guide](/docs/guides/index-selection) with HNSW/IVF/DiskANN decision flowchart
- Added [Docker & HTTP Quickstart](/docs/guides/docker-quickstart) with full REST API endpoint table
- Added [Production Checklist](/docs/guides/production-checklist) for capacity, security, and monitoring
- Added [Operations Guide](/docs/advanced/operations) with Prometheus metrics and troubleshooting runbooks
- Added [Deployment Guide](/docs/advanced/deployment) for Docker, Kubernetes, and Helm
- Added [Distributed Operations](/docs/advanced/distributed) covering Raft replication and sharding
- Added [API Stability](/docs/api-stability) tiers (Stable, Beta, Experimental)
- Audited all Rust code examples for API accuracy (`create_collection`, `search`, `insert` signatures)
- Added Playground link to navbar

---

## [0.1.0] - 2024-01-15

### Added

#### Core Features
- **HNSW Index**: Hierarchical Navigable Small World graph for approximate nearest neighbor search
  - Sub-10ms search latency on million-scale datasets
  - Configurable M, ef_construction, and ef_search parameters
  - Auto-tuning based on workload characteristics

- **Single-File Storage**: All data stored in a single `.needle` file
  - Easy backup and distribution
  - Memory-mapped I/O for efficient access
  - Automatic file format versioning

- **Multiple Distance Functions**
  - Cosine similarity (default)
  - Euclidean distance
  - Dot product
  - Manhattan distance

- **Metadata Filtering**: MongoDB-style query syntax
  - Operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`
  - Logical: `$and`, `$or`, `$not`
  - Array: `$contains`

- **Quantization Support**
  - Scalar quantization (4x compression)
  - Product quantization (8-32x compression)
  - Binary quantization (32x compression)

#### Advanced Features
- **Hybrid Search** (feature: `hybrid`)
  - BM25 text indexing
  - Reciprocal Rank Fusion (RRF) for combining results
  - Configurable fusion parameters

- **HTTP REST API** (feature: `server`)
  - Full CRUD operations
  - Search with filtering
  - Prometheus metrics endpoint
  - Rate limiting and CORS support

- **Sparse Vector Support**
  - TF-IDF vectors
  - SPLADE embeddings
  - Inverted index for efficient sparse search

- **Multi-Vector (ColBERT)**
  - Token-level embeddings
  - MaxSim search algorithm
  - Late interaction support

#### Language Bindings
- **Python** (feature: `python`)
  - PyO3-based native bindings
  - NumPy array support
  - Pythonic API design

- **JavaScript/WASM** (feature: `wasm`)
  - Browser and Node.js support
  - TypeScript definitions
  - Async API

- **Swift/Kotlin** (feature: `uniffi-bindings`)
  - UniFFI-generated bindings
  - iOS and Android support

#### Enterprise Features
- **Encryption at Rest**
  - ChaCha20-Poly1305 authenticated encryption
  - Key derivation with HKDF

- **Sharding**
  - Consistent hash ring distribution
  - Cross-shard search aggregation

- **Multi-Tenancy**
  - Namespace isolation
  - Per-tenant access control

#### CLI
- `create` - Create new database
- `create-collection` - Create collection with dimensions
- `info` - Show database information
- `collections` - List all collections
- `stats` - Collection statistics
- `insert` - Insert vectors from JSON
- `get` - Retrieve vector by ID
- `search` - Search similar vectors
- `delete` - Delete vector by ID
- `export` - Export to JSON
- `import` - Import from JSON
- `compact` - Reclaim deleted space
- `tune` - Auto-tune HNSW parameters

#### Performance
- SIMD-optimized distance functions (AVX2, NEON)
- Parallel batch search with Rayon
- Memory-mapped file I/O
- LRU query result caching

### Notes

This is the initial public release of Needle. Some features are marked as experimental:

- **Cloud Storage** (S3, GCS, Azure): Interface defined but implementations are mock/stub only
- **GPU Acceleration**: Scaffolding only, falls back to CPU
- **Embeddings** (feature: `embeddings`): Uses pre-release `ort` crate, API may change

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-01-15 | Initial release |

## Upgrade Guide

### Upgrading to 0.1.x

This is the initial release. No upgrade steps required.

### Future Compatibility

Needle aims for backward compatibility of the file format. Databases created with 0.1.x will be readable by future versions. If breaking changes are necessary, migration tools will be provided.

## Links

- [GitHub Releases](https://github.com/anthropics/needle/releases)
- [Crates.io](https://crates.io/crates/needle)
- [Migration Guides](/docs/comparison#migration-guides)
