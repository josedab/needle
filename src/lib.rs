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
pub mod collection;
pub mod database;
pub mod distance;
pub mod error;
pub(crate) mod lineage;
pub mod metadata;
pub(crate) mod storage;
pub mod tuning;

// ── Indexing ──────────────────────────────────────────────────────────────────
// Vector index implementations: HNSW, DiskANN, IVF, sparse, multi-vector.
pub mod indexing;
#[cfg(feature = "diskann")]
pub use indexing::{diskann, tiered_ann};
pub use indexing::{
    float16, graph_vector_index, hnsw, hybrid_ann, incremental, ivf, multimodal_index, multivec,
    quantization, sparse,
};

// ── Search & Query ───────────────────────────────────────────────────────────
pub mod search;
#[cfg(feature = "experimental")]
pub use search::collaborative_search;
pub use search::{
    cross_collection, federated, nl_filter, query_builder, query_explain, query_lang,
    query_planner, reranker, routing,
};

// ── Storage & Persistence ────────────────────────────────────────────────────
pub mod persistence;
pub use persistence::{
    backup, cloud_storage, managed_backup, migrations, shard, snapshot_replication, sync_protocol,
    tiered, time_travel, transaction, versioning, wal,
};

// ── Embeddings & ML ──────────────────────────────────────────────────────────
pub mod ml;
pub use ml::{
    auto_embed, dimreduce, embeddings_gateway, finetuning, local_inference, matryoshka,
    model_registry, multimodal, rag,
};

// ── Enterprise Features ──────────────────────────────────────────────────────
// Security, encryption, multi-tenancy, RBAC, Raft consensus.
pub mod enterprise;
#[cfg(feature = "encryption")]
pub use enterprise::encryption;
pub use enterprise::{autoscaling, namespace, raft, security, tenant_isolation};

// ── Observability ────────────────────────────────────────────────────────────
// Telemetry, drift detection, anomaly detection, profiling.
pub mod observe;
pub use observe::{anomaly, drift, observability, otel_service, profiler, telemetry};

// ── Framework Integrations ───────────────────────────────────────────────────
// Adapters for LangChain, LlamaIndex, Haystack, Semantic Kernel.
pub mod integrations;
pub(crate) use integrations::framework_common;
pub use integrations::{haystack, semantic_kernel};
#[cfg(feature = "integrations")]
pub use integrations::{langchain, llamaindex};

// ── High-Level Services ──────────────────────────────────────────────────────
// Database-level service wrappers with builders and lifecycle management.
pub mod services;
#[cfg(feature = "experimental")]
pub use services::{adaptive_service, ingestion_pipeline, plugin_runtime};
pub use services::{
    ingestion_service, multimodal_service, pitr_service, text_collection, tiered_service,
};

// ── Next-Gen Services ────────────────────────────────────────────────────────
pub use services::{
    adaptive_optimizer, edge_runtime, graphrag_service, incremental_sync, managed_embeddings,
    nl_filter_parser, streaming_ingest, time_travel_query, visual_explorer, wasm_sdk,
};

// ── Experimental / Advanced ──────────────────────────────────────────────────
// Features under active development. APIs may change without notice.
// Enable experimental modules with: --features experimental
pub mod experimental;

// Backward-compatible re-exports so `crate::module_name` still works
#[cfg(feature = "experimental")]
pub use experimental::adaptive_index;
#[cfg(feature = "experimental")]
pub use experimental::adaptive_runtime;
#[cfg(feature = "experimental")]
pub use experimental::agentic_memory;
#[cfg(feature = "experimental")]
pub use experimental::analytics;
#[cfg(feature = "experimental")]
pub use experimental::cloud_control;
pub use experimental::clustering;
#[cfg(feature = "experimental")]
pub use experimental::crdt;
pub use experimental::dedup;
#[cfg(feature = "experimental")]
pub use experimental::distributed_hnsw;
#[cfg(feature = "experimental")]
pub use experimental::edge_optimized;
#[cfg(feature = "experimental")]
pub use experimental::edge_partitioning;
pub use experimental::gpu;
pub use experimental::graph;
pub use experimental::plugin;
pub use experimental::temporal;
// NOTE: experimental::edge_runtime is accessed via crate::experimental::edge_runtime
// to avoid conflict with the services::edge_runtime re-export above.
#[cfg(feature = "experimental")]
pub use experimental::graphrag_index;
#[cfg(feature = "experimental")]
pub use experimental::knowledge_graph;
#[cfg(feature = "experimental")]
pub use experimental::learned_tuning;
#[cfg(feature = "experimental")]
pub use experimental::llm_cache;
#[cfg(feature = "experimental")]
pub use experimental::optimizer;
#[cfg(feature = "experimental")]
pub use experimental::platform_adapters;
#[cfg(feature = "experimental")]
pub use experimental::playground;
#[cfg(feature = "experimental")]
pub use experimental::plugin_registry;
#[cfg(feature = "experimental")]
pub use experimental::python_arrow;
#[cfg(feature = "experimental")]
pub use experimental::rebalance;
#[cfg(feature = "experimental")]
pub use experimental::serverless_runtime;
#[cfg(feature = "experimental")]
pub use experimental::streaming_upsert;
#[cfg(feature = "experimental")]
pub use experimental::vector_streaming;
#[cfg(feature = "experimental")]
pub use experimental::zero_copy;

// ── Streaming (feature-gated) ────────────────────────────────────────────────
#[cfg(feature = "server")]
pub mod streaming;

// ── Feature-Gated Modules ────────────────────────────────────────────────────
// Optional modules enabled by Cargo feature flags.

#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "async")]
pub mod async_api;

#[cfg(feature = "hybrid")]
pub mod hybrid;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "web-ui")]
pub mod web_ui;

#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "embeddings")]
pub mod embeddings;

#[cfg(feature = "embedding-providers")]
pub mod embeddings_provider;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "uniffi-bindings")]
pub mod uniffi_bindings;

// UniFFI scaffolding must be at crate root
#[cfg(feature = "uniffi-bindings")]
uniffi::setup_scaffolding!();

// ── Stable API ───────────────────────────────────────────────────────────────
// These types form the core stable API surface. Breaking changes follow semver.
pub use collection::{
    Collection, CollectionConfig, CollectionIter, CollectionStats, QueryCacheConfig,
    QueryCacheStats, SearchExplain, SearchResult,
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
    EmbeddingModelManager, ModelEntry, ModelHub, ModelType, TextFirstCollection, TextInsertable,
    TextSearchResult,
};

// ── Beta & Experimental API ──────────────────────────────────────────────────
// Beta and experimental types have been moved behind dedicated module paths.
// Use `needle::beta_api::*` for beta types and `needle::experimental_api::*`
// for experimental types.
pub mod beta_api;
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
pub use embeddings_provider::{
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
mod tests {
    use super::*;
    use serde_json::json;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

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
