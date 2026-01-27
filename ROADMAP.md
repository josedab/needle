# Roadmap

This document outlines the planned development direction for Needle. Items are organized by priority and expected milestone, though timelines may shift based on community feedback and priorities.

## Current Status

Needle is currently in **v0.1.x** (pre-1.0), focusing on API stabilization and core feature completion.

## Released: v0.1.0

### Core
- [x] HNSW index with sub-10ms approximate nearest neighbor search
- [x] Collection and database management with single-file `.needle` storage
- [x] Metadata storage and MongoDB-style filtering
- [x] Multiple distance functions (Cosine, Euclidean, DotProduct, Manhattan)
- [x] SIMD-optimized distance calculations (AVX2, NEON)
- [x] Memory-mapped file I/O
- [x] Structured error codes and recovery hints

### Search & Indexing
- [x] BM25 + vector hybrid search with Reciprocal Rank Fusion
- [x] Scalar, Product, and Binary quantization
- [x] Auto-tuning for HNSW parameters
- [x] IVF (Inverted File) index
- [x] DiskANN on-disk index
- [x] Sparse vector support (TF-IDF, SPLADE)
- [x] Multi-vector (ColBERT) MaxSim search
- [x] NeedleQL query language
- [x] Natural language filter parsing
- [x] Cross-encoder reranking

### Server & API
- [x] HTTP REST API server with Axum
- [x] Async database API with streaming
- [x] Prometheus metrics and observability
- [x] Rate limiting and JWT authentication
- [x] CLI with full command set
- [x] Terminal UI (TUI) for interactive management
- [x] Web UI for browser-based administration

### Language Bindings
- [x] Python bindings via PyO3
- [x] WebAssembly bindings
- [x] Swift/Kotlin bindings via UniFFI

### ML & Embeddings
- [x] ONNX embedding inference (unstable)
- [x] Embedding providers (OpenAI, Cohere, Ollama)
- [x] LangChain and LlamaIndex integration
- [x] RAG pipeline support

### Enterprise (Beta)
- [x] Encryption at rest (ChaCha20-Poly1305)
- [x] RBAC with audit logging
- [x] Write-Ahead Log (WAL)
- [x] Backup and restore
- [x] Multi-tenancy with namespaces
- [x] Raft consensus for leader election
- [x] Sharding with consistent hashing

### Infrastructure
- [x] Helm chart for Kubernetes
- [x] Dockerfile and docker-compose
- [x] CI/CD with GitHub Actions
- [x] Criterion benchmarks and fuzzing targets

## Upcoming: v0.2.0

### Stability & Polish
- [ ] API stabilization and documentation review
- [ ] Performance benchmarking and optimization pass
- [ ] Upgrade ONNX Runtime (`ort`) to stable release
- [ ] Comprehensive error message improvements

### Testing & Quality
- [ ] Increase test coverage to 80%+
- [ ] Add more property-based tests for edge cases
- [ ] Stress testing under high load
- [ ] Memory leak detection and profiling

## Future: v0.3.0

### Performance
- [ ] Improved SIMD support (AVX-512)
- [ ] GPU acceleration improvements
- [ ] Query planning and optimization
- [ ] Connection pooling for server mode

### Features
- [ ] GraphQL API option
- [ ] Improved streaming and pagination
- [ ] Custom distance function plugins
- [ ] Vector compression improvements

## Future: v1.0.0

### Stability Guarantees
- [ ] Stable public API with semver guarantees
- [ ] Stable file format with migration tooling
- [ ] Long-term support commitment
- [ ] Production deployment guides

### Enterprise Readiness
- [ ] SOC 2 compliance documentation
- [ ] High availability deployment patterns
- [ ] Disaster recovery documentation
- [ ] Performance SLA guidelines

## Exploration (No Timeline)

These items are under consideration but not yet planned:

- **Vector Streaming**: Real-time vector ingestion from message queues
- **Federated Search**: Cross-instance search coordination
- **Vector Compression Research**: New quantization techniques
- **Hardware Acceleration**: TPU/specialized hardware support
- **Managed Service**: Cloud-hosted Needle offering

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas for contributions:
- Documentation improvements
- Test coverage expansion
- Performance optimizations
- Language binding improvements

## Feedback

Have suggestions for the roadmap? Please:
1. Open a [GitHub Issue](https://github.com/anthropics/needle/issues) for feature requests
2. Join discussions in existing roadmap-related issues
3. Submit PRs for items you'd like to work on
