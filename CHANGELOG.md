# Changelog

All notable changes to Needle are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### CI/CD & Deployment
- GitHub Actions workflows for CI, Docker builds, releases, and security scanning
- Helm chart for production Kubernetes deployment
- Kubernetes manifests and monitoring configuration (Prometheus, Grafana)
- Dockerfile and docker-compose for containerized deployment

#### Documentation
- README with comprehensive feature documentation
- API reference documentation
- Architecture documentation
- Usage examples for basic operations, persistence, and quantization
- Fuzzing documentation

#### Testing
- Fuzzing targets for query parser, filters, and distance functions
- Criterion benchmarks for index and search performance
- Integration, async, and concurrency tests
- Property-based tests using proptest

#### User Interfaces
- Terminal UI (TUI) for interactive database management
- Web UI for browser-based administration
- CLI application with comprehensive commands:
  - `create`, `create-collection`, `info`, `collections`
  - `stats`, `insert`, `search`, `delete`, `get`
  - `export`, `import`, `count`, `clear`, `compact`
  - `serve`, `tune`

#### Language Bindings
- Python bindings via PyO3
- WebAssembly bindings for browser/Node.js
- Swift and Kotlin bindings via UniFFI

#### Server & API
- HTTP REST API server with Axum
- Async database API with streaming support
- Change streams for real-time updates
- Prometheus metrics and observability

#### ML & Embeddings
- ONNX embedding inference
- LangChain integration
- RAG (Retrieval-Augmented Generation) pipeline support
- Embedding providers: OpenAI, Cohere, Ollama

#### Cloud & Storage
- S3-compatible storage backend
- Azure Blob Storage backend
- Google Cloud Storage backend
- Tiered storage (hot/warm/cold)
- GPU acceleration for distance computation

#### Distributed Features
- Sharding with consistent hashing
- Query routing and aggregation
- CRDT support for conflict-free replication
- Raft consensus for leader election
- Automatic rebalancing

#### Enterprise Features
- RBAC (Role-Based Access Control)
- Audit logging
- Encryption at rest (ChaCha20-Poly1305)
- Vector versioning and branching
- Write-Ahead Log (WAL) for durability
- Backup and restore functionality
- Multi-tenancy with namespaces

#### Analytics & ML
- Data lineage tracking
- Distribution drift detection
- Query profiling and optimization
- Temporal indexing with decay functions
- K-means and hierarchical clustering
- Deduplication with configurable thresholds
- Anomaly detection (Isolation Forest, LOF)
- Semantic graph construction and traversal

#### Search & Indexing
- NeedleQL query language
- Natural language filter parsing
- BM25 + vector hybrid search with RRF fusion
- Cross-encoder reranking
- Scalar, Product, and Binary quantization
- Auto-tuning for HNSW parameters
- IVF (Inverted File) index
- DiskANN on-disk index
- Sparse vector support (TF-IDF, SPLADE)
- Multi-vector (ColBERT) support

#### Core
- HNSW index implementation
- Collection and database management
- Metadata storage and MongoDB-style filtering
- Multiple distance functions (Cosine, Euclidean, DotProduct, Manhattan)
- SIMD-optimized distance calculations (AVX2, NEON)
- Memory-mapped file I/O
- Single-file storage format

## [0.1.0] - Initial Release

### Added
- Core library foundation
- Error handling with thiserror
- Distance functions with optional SIMD
- Storage layer with mmap support

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | - | Initial release with core functionality |

## Migration Guides

### Upgrading to 0.2.0 (Future)

*No breaking changes planned.*

---

## Links

- [GitHub Repository](https://github.com/anthropics/needle)
- [Documentation](./docs/)
- [API Reference](./docs/api-reference.md)
