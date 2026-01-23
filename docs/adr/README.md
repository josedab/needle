# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the Needle vector database. ADRs document significant architectural decisions, providing context for why the system is built the way it is.

## What is an ADR?

An Architecture Decision Record captures a single architectural decision along with its context and consequences. They help new team members understand "why is it built this way?" and provide a historical record of technical choices.

## ADR Index

| ADR | Title | Status | Summary |
|-----|-------|--------|---------|
| [0001](0001-single-file-storage-with-memory-mapping.md) | Single-File Storage with Memory Mapping | Accepted | SQLite-inspired single-file format with mmap for large files |
| [0002](0002-hnsw-as-primary-index-algorithm.md) | HNSW as Primary Index Algorithm | Accepted | Hierarchical Navigable Small World graph for O(log n) ANN search |
| [0003](0003-interior-mutability-with-parking-lot-rwlock.md) | Interior Mutability with Parking Lot RwLock | Accepted | Thread-safe concurrent access via parking_lot's fast RwLock |
| [0004](0004-generation-based-dirty-tracking.md) | Generation-Based Dirty Tracking | Accepted | Race-free modification tracking via atomic generation counters |
| [0005](0005-mongodb-style-filter-query-language.md) | MongoDB-Style Filter Query Language | Accepted | Familiar JSON query syntax for metadata filtering |
| [0006](0006-async-api-as-optional-feature-layer.md) | Async API as Optional Feature Layer | Accepted | Sync-first design with opt-in async for server deployments |
| [0007](0007-multiple-quantization-strategies.md) | Multiple Quantization Strategies | Accepted | Scalar, Product, and Binary quantization for memory optimization |
| [0008](0008-structured-error-handling-with-thiserror.md) | Structured Error Handling with Thiserror | Accepted | Type-safe errors with context via thiserror derive macros |
| [0009](0009-modular-feature-flag-architecture.md) | Modular Feature-Flag Architecture | Accepted | Cargo features for selective compilation and minimal builds |
| [0010](0010-raft-based-replication-layer.md) | Raft-Based Replication Layer | Accepted | Composable Raft consensus for distributed deployments |
| [0011](0011-multiple-distance-functions-with-simd.md) | Multiple Distance Functions with SIMD | Accepted | Cosine/Euclidean/Dot/Manhattan with optional SIMD optimization |
| [0012](0012-hybrid-search-with-bm25-and-rrf-fusion.md) | Hybrid Search with BM25 and RRF Fusion | Accepted | Combined semantic + lexical search via Reciprocal Rank Fusion |
| [0013](0013-cloud-storage-abstraction-layer.md) | Cloud Storage Abstraction Layer | Accepted | S3/GCS/Azure backends with three-tier caching |
| [0014](0014-gpu-acceleration-architecture.md) | GPU Acceleration Architecture | Accepted | CUDA/Metal/OpenCL backends with automatic selection |
| [0015](0015-streaming-cdc-integration.md) | Streaming CDC Integration | Accepted | Real-time Change Data Capture from PostgreSQL/MongoDB |
| [0016](0016-time-travel-with-mvcc.md) | Time-Travel with MVCC | Accepted | Point-in-time vector queries via multi-version control |
| [0017](0017-predictive-autoscaling.md) | Predictive Auto-Scaling | Accepted | ML-based load forecasting and resource management |
| [0018](0018-federated-multi-instance-search.md) | Federated Multi-Instance Search | Accepted | Cross-datacenter queries with latency-aware routing |
| [0019](0019-embedded-model-finetuning.md) | Embedded Model Fine-Tuning | Accepted | In-database contrastive learning from user feedback |
| [0020](0020-unified-rag-pipeline.md) | Unified RAG Pipeline | Accepted | Built-in retrieval-augmented generation with caching |
| [0021](0021-query-result-caching-with-lru.md) | Query Result Caching with LRU | Accepted | In-memory LRU cache for repeated vector searches |
| [0022](0022-multi-index-architecture.md) | Multi-Index Architecture | Accepted | HNSW/IVF/DiskANN with constraint-based selection |
| [0023](0023-dual-vector-id-system.md) | Dual Vector ID System | Accepted | User-facing strings with internal integer mapping |
| [0024](0024-observable-query-execution.md) | Observable Query Execution | Accepted | Built-in SearchExplain profiling for query analysis |
| [0025](0025-write-ahead-logging.md) | Write-Ahead Logging | Accepted | WAL with checkpoints for crash recovery |
| [0026](0026-needleql-query-language.md) | NeedleQL Query Language | Accepted | SQL-like DSL for vector operations |
| [0027](0027-namespace-based-multi-tenancy.md) | Namespace-Based Multi-Tenancy | Accepted | Logical tenant isolation with resource quotas |
| [0028](0028-semantic-knowledge-graph.md) | Semantic Knowledge Graph | Accepted | Entity-relationship overlay for structured knowledge |

## Reading Order

For new team members, we recommend reading the ADRs in this order to understand the system's evolution:

### Core Architecture (Start Here)
1. **ADR-0001** — Establishes the "SQLite for vectors" philosophy
2. **ADR-0002** — Explains the primary search algorithm
3. **ADR-0003** — Shows how thread safety is achieved
4. **ADR-0004** — Covers persistence and consistency

### Query Interface
5. **ADR-0005** — Metadata filtering design
6. **ADR-0011** — Distance function selection
7. **ADR-0012** — Hybrid search capabilities

### System Design
8. **ADR-0008** — Error handling philosophy
9. **ADR-0009** — Feature flag architecture
10. **ADR-0006** — Async API design

### Advanced Features
11. **ADR-0007** — Memory optimization via quantization
12. **ADR-0010** — Distributed deployments

### Next-Gen Features
13. **ADR-0013** — Cloud storage backends (S3, GCS, Azure)
14. **ADR-0014** — GPU acceleration (CUDA, Metal)
15. **ADR-0015** — Real-time CDC integration
16. **ADR-0016** — Time-travel queries with MVCC
17. **ADR-0017** — Predictive auto-scaling
18. **ADR-0018** — Federated multi-instance search
19. **ADR-0019** — Embedded model fine-tuning
20. **ADR-0020** — Unified RAG pipeline

### Performance & Observability
21. **ADR-0021** — Query result caching with LRU
22. **ADR-0022** — Multi-index architecture (HNSW, IVF, DiskANN)
23. **ADR-0023** — Dual vector ID system
24. **ADR-0024** — Observable query execution with SearchExplain
25. **ADR-0025** — Write-ahead logging for durability

### Developer Experience
26. **ADR-0026** — NeedleQL: SQL-like query language
27. **ADR-0027** — Namespace-based multi-tenancy
28. **ADR-0028** — Semantic knowledge graph overlay

## ADR Format

Each ADR follows this structure:

- **Title** — ADR-NNNN: Short Decision Title
- **Status** — Accepted, Superseded, or Deprecated
- **Context** — What prompted this decision?
- **Decision** — What was decided?
- **Consequences** — Tradeoffs, implications, what this enabled/prevented

## Contributing

When making significant architectural changes:

1. Create a new ADR with the next available number
2. Follow the established format
3. Link to relevant code with file:line references
4. Update this index
5. If superseding an existing ADR, update its status

## Design Principles

These ADRs collectively reflect Needle's core design principles:

1. **SQLite-inspired simplicity** — Single file, zero configuration
2. **Composability** — Features opt-in via flags, core remains minimal
3. **Performance-conscious defaults** — HNSW, RwLock, SIMD where beneficial
4. **Safety via Rust** — Type safety prevents entire classes of bugs
5. **Flexible deployment** — Embedded library to distributed cluster
6. **User choice** — Quantization, distance functions, replication all configurable
