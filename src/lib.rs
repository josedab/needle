//! # Needle - Embedded Vector Database
//!
//! Needle is an embedded vector database written in Rust, designed to be the "SQLite for vectors."
//! It provides high-performance approximate nearest neighbor (ANN) search with a single-file
//! storage format, zero configuration, and seamless integration for AI applications.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use needle::{Database, Filter};
//! use serde_json::json;
//!
//! fn main() -> needle::Result<()> {
//!     // Create an in-memory database
//!     let db = Database::in_memory();
//!
//!     // Create a collection for 384-dimensional vectors
//!     db.create_collection("documents", 384)?;
//!
//!     // Get a reference to the collection
//!     let collection = db.collection("documents")?;
//!
//!     // Insert vectors with metadata
//!     let embedding = vec![0.1; 384]; // Your embedding here
//!     collection.insert(
//!         "doc1",
//!         &embedding,
//!         Some(json!({"title": "Hello World", "category": "greeting"}))
//!     )?;
//!
//!     // Search for similar vectors
//!     let query = vec![0.1; 384];
//!     let results = collection.search(&query, 10)?;
//!
//!     for result in results {
//!         println!("ID: {}, Distance: {}", result.id, result.distance);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **HNSW Index**: Fast approximate nearest neighbor search with sub-10ms queries
//! - **SIMD Optimized**: Distance functions optimized for AVX2 and NEON
//! - **Single-File Storage**: All data stored in one file, easy to backup and distribute
//! - **Metadata Filtering**: Filter search results by metadata fields
//! - **Multiple Distance Metrics**: Cosine, Euclidean, Dot Product, Manhattan
//! - **Quantization**: Scalar and Product Quantization for memory efficiency
//!
//! ## Persistence
//!
//! ```rust,no_run
//! use needle::Database;
//!
//! fn main() -> needle::Result<()> {
//!     // Open or create a database file
//!     let mut db = Database::open("vectors.needle")?;
//!
//!     db.create_collection("my_collection", 128)?;
//!     // ... insert vectors ...
//!
//!     // Save changes to disk
//!     db.save()?;
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

// ── Core ──────────────────────────────────────────────────────────────────────
// Fundamental types: database, collections, vectors, errors, storage, metadata.

/// Collection management: vectors, HNSW index, metadata, and search pipeline.
pub mod collection;
/// Database management: multi-collection container, persistence, and thread-safe access.
pub mod database;
/// Distance functions for vector similarity (Cosine, Euclidean, Dot Product, Manhattan).
pub mod distance;
/// Error types, error codes, and recovery hints for all Needle operations.
pub mod error;
#[cfg(feature = "observability")]
pub(crate) use observe::lineage;
/// Metadata storage and MongoDB-style query filtering (`$eq`, `$gt`, `$in`, `$or`, etc.).
pub mod metadata;
pub(crate) mod storage;
/// Auto-tuning for HNSW parameters based on workload profiling.
pub mod tuning;
/// Automated recall benchmarking: synthetic datasets, brute-force ground truth, recall@k, QPS, latency.
pub mod recall_benchmark;
/// Programmable pre/post-processing hooks for insert and search operations.
pub mod hooks;
/// Git-like collection branching with copy-on-write pages and merge with conflict detection.
pub mod collection_branch;
/// Columnar storage for efficient batch vector operations.
pub use indexing::columnar;
/// Data processing pipelines for vector ingestion and transformation.
pub use search::pipeline;
/// Graph-based retrieval augmented generation (GraphRAG) support.
pub use search::graphrag;

// ── Indexing ──────────────────────────────────────────────────────────────────
// Vector index implementations: HNSW, DiskANN, IVF, sparse, multi-vector.

/// Vector index implementations: HNSW, IVF, DiskANN, sparse, multi-vector, and quantization.
pub mod indexing;
#[cfg(feature = "diskann")]
pub use indexing::{diskann, tiered_ann};
pub use indexing::{
    float16, graph_vector_index, hnsw, cow_hnsw, compression_advisor, cli_playground, hybrid_ann, incremental, ivf, multimodal_index, multivec,
    quantization, sparse,
};

// ── Search & Query ───────────────────────────────────────────────────────────

/// Search query planning, reranking, federated search, and natural language filtering.
pub mod search;
#[cfg(feature = "experimental")]
pub use search::collaborative_search;
pub use search::{
    cross_collection, federated, nl_filter, query_builder, query_explain, query_lang,
    query_planner, reranker, routing, sql_analytics,
};

// ── Storage & Persistence ────────────────────────────────────────────────────

/// Storage and persistence: WAL, backups, migrations, sharding, and cloud storage.
pub mod persistence;
pub use persistence::{
    backup, cloud_storage, cow_storage, incremental_backup, managed_backup, migrations,
    replica_manager, schema_evolution, shard, snapshot_replication, sync_protocol, tiered,
    time_travel, transaction, vector_versioning, versioning, wal,
};

// ── Embeddings & ML ──────────────────────────────────────────────────────────

/// Machine learning utilities: auto-embedding, model registry, RAG, and dimensionality reduction.
pub mod ml;
pub use ml::{
    auto_embed, dimreduce, embedded_runtime, embeddings_gateway, finetuning, llm_provider,
    local_inference, matryoshka, model_registry, multimodal, rag,
};
#[cfg(feature = "experimental")]
pub use ml::inference_engine;

// ── Enterprise Features ──────────────────────────────────────────────────────
// Security, encryption, multi-tenancy, RBAC, Raft consensus.

/// Enterprise features: encryption, RBAC, multi-tenancy, Raft consensus, and autoscaling.
pub mod enterprise;
#[cfg(feature = "encryption")]
pub use enterprise::encryption;
pub use enterprise::{autoscaling, namespace, raft, security, tenant_isolation};

// ── Observability ────────────────────────────────────────────────────────────
// Telemetry, drift detection, anomaly detection, profiling.

/// Observability: telemetry, drift detection, anomaly detection, and profiling.
/// Requires the `observability` feature flag (included in `full`).
#[cfg(feature = "observability")]
pub mod observe;
#[cfg(feature = "observability")]
pub use observe::{anomaly, audit, dashboard, drift, observability, otel_service, profiler, telemetry};

// ── Framework Integrations ───────────────────────────────────────────────────
// Adapters for LangChain, LlamaIndex, Haystack, Semantic Kernel.

/// Framework integrations: LangChain, LlamaIndex, Haystack, and Semantic Kernel adapters.
pub mod integrations;
pub(crate) use integrations::framework_common;
pub use integrations::{haystack, semantic_kernel};
#[cfg(feature = "integrations")]
pub use integrations::{langchain, llamaindex};

// ── High-Level Services ──────────────────────────────────────────────────────
// Database-level service wrappers with builders and lifecycle management.
// See src/services/README.md for the full directory layout.

/// High-level service wrappers: ingestion, multi-modal, PITR, text collections, and more.
///
/// Services are organized into domain subdirectories:
/// `ai/`, `client/`, `collection/`, `compute/`, `embedding/`, `governance/`,
/// `infrastructure/`, `observability/`, `pipeline/`, `plugin/`, `search/`,
/// `storage/`, `sync/`. All modules are re-exported at the `services::` level
/// for backward compatibility.
pub mod services;
#[cfg(feature = "experimental")]
pub use services::{adaptive_service, ingestion_pipeline, plugin_runtime};
pub use services::{
    ingestion_service, multimodal_service, pitr_service, text_collection, tiered_service,
    live_migration_service, vector_namespace,
};

// ── Experimental Services ────────────────────────────────────────────────────
// Experimental service modules grouped by domain. Requires: --features experimental

// AI & LLM integration
#[cfg(feature = "experimental")]
pub use services::{
    agentic_memory_protocol, agentic_workflow, graph_knowledge_service, graph_query,
    graphrag_service, llm_cache_middleware, llm_tools, rag_sdk, semantic_cache,
};

// Embedding & model management
#[cfg(feature = "experimental")]
pub use services::{
    auto_embed_endpoint, embedding_router, managed_embeddings,
    matryoshka_service, model_downloader, model_runtime, smart_auto_embed, text_to_vector,
};

// Search & query
#[cfg(feature = "experimental")]
pub use search::{needleql_executor, needleql_lsp};
#[cfg(feature = "experimental")]
pub use services::{
    adaptive_index_selector, encrypted_search, nl_filter_parser,
    query_cache_middleware, query_optimizer, query_replay,
};
#[cfg(feature = "experimental")]
pub use search::cost_estimator;
#[cfg(feature = "experimental")]
pub use indexing::{graph_vector_fusion, multimodal_fusion};

// Pipeline & ingestion
#[cfg(feature = "experimental")]
pub use services::{
    cdc_framework, pipeline_manager, realtime_streaming, streaming_ingest, streaming_protocol,
    vector_pipeline,
};

// Sync & replication
#[cfg(feature = "experimental")]
pub use services::{
    change_stream, crdt_sync, distributed_federation, incremental_sync, live_replication,
    multi_writer, sync_engine,
};

// Collection management
#[cfg(feature = "experimental")]
pub use services::{
    collection_bundle, collection_federation, collection_rbac, materialized_views,
    multimodal_collection, snapshot_time_travel, typed_schema,
};

// Compute & transactions
#[cfg(feature = "experimental")]
pub use services::{
    adaptive_optimizer, gpu_kernels, time_travel_query, transactional_api, vector_transactions,
};

// Infrastructure & deployment
#[cfg(feature = "experimental")]
pub use services::{
    cloud_deploy, cloud_service, cluster_bootstrap, edge_runtime, edge_serverless, managed_cloud,
    pricing, readiness_probe, tenant_router,
};

// Plugin & WASM
#[cfg(feature = "experimental")]
pub use services::{
    plugin_api, plugin_ecosystem, wasm_browser, wasm_persistence, wasm_plugin_runtime, wasm_sdk,
};

// Client SDKs & protocols
#[cfg(feature = "experimental")]
pub use services::{
    client_sdk, grpc_schema, notebook, python_sdk, vscode_extension, webhook_delivery, ws_protocol,
};

// Observability & benchmarking
#[cfg(feature = "experimental")]
pub use services::{
    ann_benchmark, benchmark_runner, benchmark_suite, drift_monitor, evidence_collector,
    otel_tracing, triage_report, vector_lineage, visual_explorer,
};

// Governance & compliance
#[cfg(feature = "experimental")]
pub use services::{
    api_stability, community, compliance, format_spec, format_validator, module_audit,
    unwrap_audit, version_control,
};

// Storage
#[cfg(feature = "experimental")]
pub use services::{
    backup_command, hnsw_compactor, snapshot_manager, storage_backends,
};

// ── Experimental / Advanced ──────────────────────────────────────────────────
// Features under active development. APIs may change without notice.
// Enable experimental modules with: --features experimental

/// Experimental modules under active development. APIs may change without notice.
///
/// Enable with `--features experimental`. Includes adaptive indexing, GPU acceleration,
/// clustering, CRDT replication, graph operations, and more.
pub mod experimental;

// Backward-compatible re-exports so `crate::module_name` still works
/// Adaptive index selection based on workload patterns. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::adaptive_index;
/// Adaptive runtime for dynamic resource allocation. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::adaptive_runtime;
/// Agentic memory protocol for LLM agent context management. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::agentic_memory;
/// Analytics and aggregation over vector collections. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::analytics;
/// Cloud control plane for managed deployments. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::cloud_control;
/// K-means and hierarchical clustering of vectors. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::clustering;
/// Conflict-free Replicated Data Types for eventual consistency. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::crdt;
/// Vector deduplication using similarity thresholds. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::dedup;
/// Distributed HNSW index across multiple nodes. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::distributed_hnsw;
/// Optimized builds for edge and resource-constrained devices. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::edge_optimized;
/// Graph-based edge partitioning for distributed workloads. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::edge_partitioning;
/// GPU-accelerated distance computations. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::gpu;
/// Semantic graph operations on vector relationships. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::graph;
/// Plugin system for extending Needle with custom logic. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::plugin;
/// Temporal indexing with time-decay functions. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::temporal;
// NOTE: experimental::edge_runtime is accessed via crate::experimental::edge_runtime
// to avoid conflict with the services::edge_runtime re-export above.
/// GraphRAG index for knowledge-graph-augmented retrieval. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::graphrag_index;
/// Knowledge graph construction and traversal. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::knowledge_graph;
/// Learned parameter tuning using workload feedback. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::learned_tuning;
/// LLM response caching with semantic similarity. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::llm_cache;
/// Query optimizer for complex search pipelines. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::optimizer;
/// Platform-specific adapters (iOS, Android, embedded). **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::platform_adapters;
/// Interactive playground for exploring vector operations. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::playground;
/// Registry for managing and discovering plugins. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::plugin_registry;
/// Apache Arrow interop for Python-based analytics. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::python_arrow;
/// Online index rebalancing for skewed data distributions. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::rebalance;
/// Serverless runtime for event-driven vector operations. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::serverless_runtime;
/// Streaming upsert pipeline for high-throughput ingestion. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::streaming_upsert;
/// Real-time vector streaming with backpressure. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::vector_streaming;
/// Zero-copy deserialization for memory-mapped vectors. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::zero_copy;
/// Zero-copy IPC protocol for multi-process shared memory access. **Experimental**: API may change without notice.
#[cfg(feature = "experimental")]
pub use experimental::ipc_protocol;

// ── Streaming (feature-gated) ────────────────────────────────────────────────

/// Real-time vector streaming support. Requires the `server` feature.
#[cfg(feature = "server")]
pub mod streaming;

// ── Feature-Gated Modules ────────────────────────────────────────────────────
// Optional modules enabled by Cargo feature flags.

/// Terminal UI for interactive database management. Requires the `tui` feature.
#[cfg(feature = "tui")]
pub mod tui;

/// Async database API with streaming support. Requires the `async` feature.
#[cfg(feature = "async")]
pub mod async_api;

/// BM25 text search and hybrid vector+text search with Reciprocal Rank Fusion. Requires the `hybrid` feature.
#[cfg(feature = "hybrid")]
pub mod hybrid;

/// HTTP REST API server with authentication and rate limiting. Requires the `server` feature.
#[cfg(feature = "server")]
pub mod server;

/// Model Context Protocol (MCP) tools for LLM integration.
pub mod mcp;

/// Web-based administration UI. Requires the `web-ui` feature.
#[cfg(feature = "web-ui")]
pub mod web_ui;

/// Prometheus metrics export and monitoring. Requires the `metrics` feature.
#[cfg(feature = "metrics")]
pub mod metrics;

/// ONNX Runtime embedding inference. Requires the `embeddings` feature.
#[cfg(feature = "embeddings")]
pub mod embeddings;

/// Embedding providers for OpenAI, Cohere, and Ollama. Requires the `embedding-providers` feature.
#[cfg(feature = "embedding-providers")]
pub use ml::embeddings_provider;

/// Python bindings via PyO3. Requires the `python` feature.
#[cfg(feature = "python")]
pub mod python;

/// WebAssembly bindings for browser and Node.js. Requires the `wasm` feature.
#[cfg(feature = "wasm")]
pub mod wasm;

/// Swift and Kotlin bindings via UniFFI. Requires the `uniffi-bindings` feature.
#[cfg(feature = "uniffi-bindings")]
pub mod uniffi_bindings;

// UniFFI scaffolding must be at crate root
#[cfg(feature = "uniffi-bindings")]
uniffi::setup_scaffolding!();

// ── Stable API ───────────────────────────────────────────────────────────────
// These types form the core stable API surface. Breaking changes follow semver.
pub use collection::{
    BundleManifest, Collection, CollectionConfig, CollectionIter, CollectionStats,
    EvaluationReport, GroundTruthEntry, QueryCacheConfig, QueryCacheStats, QueryMetrics,
    SearchExplain, SearchResult, SemanticQueryCacheConfig,
};
pub use database::collection_ref::SearchParams;
pub use database::{CollectionRef, Database, DatabaseConfig, ExportEntry};
pub use distance::DistanceFunction;
pub use error::{ErrorCode, NeedleError, Recoverable, RecoveryHint, Result};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats, SearchStats};
pub use metadata::{Filter, MetadataStore};
pub use multivec::{MultiVector, MultiVectorConfig, MultiVectorIndex, MultiVectorSearchResult};
pub use quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
pub use sparse::{SparseDistance, SparseIndex, SparseVector};
pub use tuning::{
    auto_tune, quick_recommend_index, recommend_index, AdaptiveRecommendation, AdaptiveTuner,
    DataProfile, DataProfiler, IndexRecommendation, IndexSelectionConstraints, MigrationState,
    MigrationStatus, OnlineMigrationManager, PerformanceProfile, RecommendedIndex,
    SmartIndexSelection, SmartIndexSelector, TuningConstraints, TuningResult, WorkloadObservation,
};

// Automatic embedding generation
pub use auto_embed::{
    AutoEmbedCollectionBuilder, AutoEmbedConfig, AutoEmbedStats, AutoEmbedder, EmbeddingBackend,
    EmbeddingModelManager, ModelArtifact, ModelEntry, ModelHub, ModelStatus, ModelType,
    TextFirstCollection, TextInsertable, TextSearchResult,
};

// ── Beta & Experimental API ──────────────────────────────────────────────────
// Beta and experimental types have been moved behind dedicated module paths.
// Use `needle::beta_api::*` for beta types and `needle::experimental_api::*`
// for experimental types.
/// Beta API types that are approaching stability but may still have minor breaking changes.
pub mod beta_api;
/// Experimental API types under active development. APIs may change without notice.
pub mod experimental_api;

#[cfg(feature = "hybrid")]
pub use hybrid::{
    reciprocal_rank_fusion, AdaptiveFusion, AdaptiveFusionStats, Bm25Index, HybridConfig,
    HybridSearchResult, LearnedWeightStats, QueryFeatures, QueryType, RrfConfig, SearchFeedback,
};

#[cfg(feature = "server")]
pub use server::{serve, ServerConfig};

#[cfg(feature = "async")]
pub use async_api::{
    AsyncDatabase, AsyncDatabaseConfig, BatchOperationBuilder, BatchResult, ExportStream,
    SearchStream,
};

#[cfg(feature = "metrics")]
pub use metrics::{
    generate_alerting_rules, generate_grafana_dashboard, metrics, AlertingConfig, AnomalyDetector,
    AnomalyResult, GrafanaDashboardConfig, NeedleMetrics,
};

#[cfg(feature = "embeddings")]
pub use embeddings::{
    EmbedderBuilder, EmbedderConfig, EmbeddingError, PoolingStrategy, TextEmbedder,
};

#[cfg(feature = "web-ui")]
pub use web_ui::{
    create_web_ui_router, serve_web_ui, serve_web_ui_default, WebUiConfig, WebUiState,
};

#[cfg(feature = "embedding-providers")]
pub use ml::embeddings_provider::{
    BatchConfig, CachedProvider, CohereConfig, CohereInputType, EmbeddingProvider,
    EmbeddingProviderError, MockConfig, MockProvider, OllamaConfig, OllamaProvider, OpenAIConfig,
    OpenAIProvider, RateLimiter,
};

/// Prelude module for convenient imports.
///
/// ```rust
/// use needle::prelude::*;
/// ```
pub mod prelude {
    pub use crate::collection::{Collection, CollectionConfig, SearchResult};
    pub use crate::database::collection_ref::SearchParams;
    pub use crate::database::{CollectionRef, Database, DatabaseConfig};
    pub use crate::distance::DistanceFunction;
    pub use crate::error::{NeedleError, Result};
    pub use crate::hnsw::HnswConfig;
    pub use crate::metadata::Filter;
}

#[cfg(test)]
pub(crate) mod test_utils;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vector;
    use serde_json::json;

    #[test]
    fn test_end_to_end() {
        // Create database
        let db = Database::in_memory();

        // Create collection
        db.create_collection("test", 128).unwrap();

        let collection = db.collection("test").unwrap();

        // Insert vectors
        for i in 0..100 {
            let vector = random_vector(128);
            let metadata = json!({
                "title": format!("Document {}", i),
                "category": if i % 2 == 0 { "even" } else { "odd" },
                "score": i as f64 / 100.0
            });
            collection
                .insert(format!("doc_{}", i), &vector, Some(metadata))
                .unwrap();
        }

        // Basic search
        let query = random_vector(128);
        let results = collection.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);

        // Search with filter
        let filter = Filter::eq("category", "even");
        let filtered_results = collection.search_with_filter(&query, 10, &filter).unwrap();

        for result in &filtered_results {
            let meta = result.metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "even");
        }

        // Get specific vector
        let (vec, meta) = collection.get("doc_50").unwrap();
        assert_eq!(vec.len(), 128);
        assert_eq!(meta.unwrap()["title"], "Document 50");

        // Delete vector
        assert!(collection.delete("doc_50").unwrap());
        assert!(collection.get("doc_50").is_none());
    }

    #[test]
    fn test_distance_functions() {
        let db = Database::in_memory();

        // Test with Euclidean distance
        db.create_collection_with_config(
            CollectionConfig::new("euclidean", 64).with_distance(DistanceFunction::Euclidean),
        )
        .unwrap();

        let collection = db.collection("euclidean").unwrap();

        let v1 = random_vector(64);
        let v2 = random_vector(64);

        collection.insert("v1", &v1, None).unwrap();
        collection.insert("v2", &v2, None).unwrap();

        // Search should return v1 as closest to itself
        let results = collection.search(&v1, 2).unwrap();
        assert_eq!(results[0].id, "v1");
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_quantization_integration() {
        // Test scalar quantization
        let vectors: Vec<Vec<f32>> = (0..100).map(|_| random_vector(64)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let sq = ScalarQuantizer::train(&refs);

        for vec in &vectors {
            let quantized = sq.quantize(vec);
            let dequantized = sq.dequantize(&quantized);

            // Check that dequantization is reasonable
            let error: f32 = vec
                .iter()
                .zip(dequantized.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            // Error should be small relative to vector magnitude
            let magnitude: f32 = vec.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
            assert!(error < magnitude * 0.5, "Quantization error too large");
        }
    }
}
