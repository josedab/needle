# Changelog

All notable changes to Needle are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

*No changes yet.*

## [0.1.0] - 2026-02-20

### Added

#### Core
- HNSW index implementation with sub-10ms approximate nearest neighbor search
- Collection and database management with single-file `.needle` storage format
- Metadata storage and MongoDB-style filtering (`$eq`, `$ne`, `$gt`, `$lt`, `$in`, `$or`, etc.)
- Multiple distance functions (Cosine, Euclidean, DotProduct, Manhattan)
- SIMD-optimized distance calculations (AVX2 on x86_64, NEON on ARM)
- Memory-mapped file I/O for efficient large-file access
- Error handling with structured error codes and recovery hints

#### Search & Indexing
- NeedleQL query language
- Natural language filter parsing
- BM25 + vector hybrid search with Reciprocal Rank Fusion (feature: `hybrid`)
- Cross-encoder reranking (Cohere, HuggingFace, custom providers)
- Scalar, Product, and Binary quantization for memory efficiency
- Auto-tuning for HNSW parameters
- IVF (Inverted File) index for large-scale approximate search
- DiskANN on-disk index (feature: `diskann`)
- Sparse vector support (TF-IDF, SPLADE)
- Multi-vector (ColBERT) MaxSim search

#### Server & API
- HTTP REST API server with Axum (feature: `server`)
- Async database API with streaming support (feature: `async`)
- Change streams for real-time updates
- Prometheus metrics and observability (feature: `metrics`)
- Rate limiting and JWT authentication

#### User Interfaces
- CLI application with commands: `create`, `create-collection`, `info`, `collections`, `stats`, `insert`, `search`, `delete`, `get`, `export`, `import`, `count`, `clear`, `compact`, `serve`, `tune`
- Terminal UI for interactive database management (feature: `tui`)
- Web UI for browser-based administration (feature: `web-ui`)

#### Language Bindings
- Python bindings via PyO3 (feature: `python`)
- WebAssembly bindings for browser/Node.js (feature: `wasm`)
- Swift and Kotlin bindings via UniFFI (feature: `uniffi-bindings`)

#### ML & Embeddings
- ONNX embedding inference (feature: `embeddings`, unstable)
- Embedding providers: OpenAI, Cohere, Ollama (feature: `embedding-providers`)
- LangChain and LlamaIndex integration (feature: `integrations`)
- RAG pipeline support

#### Enterprise Features (Beta)
- Encryption at rest with ChaCha20-Poly1305 (feature: `encryption`)
- RBAC with audit logging
- Write-Ahead Log (WAL) for durability
- Backup and restore functionality
- Multi-tenancy with namespaces
- Raft consensus for leader election
- Sharding with consistent hashing

#### Experimental (feature: `experimental`)
- GPU acceleration scaffolding (CPU fallback at runtime)
- Cloud storage backends (S3, GCS, Azure — interface only, not production-ready)
- Temporal indexing with decay functions
- K-means and hierarchical clustering
- Deduplication, anomaly detection, semantic graphs
- CRDT support for eventual consistency

#### CI/CD & Deployment
- GitHub Actions workflows for CI, Docker builds, releases, and security scanning
- Helm chart for Kubernetes deployment
- Dockerfile and docker-compose for containerized deployment
- Criterion benchmarks and fuzzing targets
- Property-based tests using proptest

#### Documentation
- Comprehensive README with quick start, benchmarks, and usage examples
- API reference, architecture docs, and how-to guides
- Production checklist and deployment guide

### Known Issues
- **GPU & Cloud Storage**: Scaffolding/interface only — CPU/in-memory fallback at runtime
- **Embeddings feature**: Depends on pre-release `ort` crate; not included in `--features full`
- **CDC connectors**: Require external services and their respective feature flags

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2026-02-20 | Initial release with HNSW, hybrid search, REST API, multi-language bindings |

## Migration Guides

### Upgrading to 0.2.0 (Future)

*No breaking changes planned.*

---

## Links

- [GitHub Repository](https://github.com/anthropics/needle)
- [Documentation](./docs/)
- [API Reference](./docs/api-reference.md)
