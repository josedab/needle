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

// ── Core ──────────────────────────────────────────────────────────────────────
// Fundamental types: database, collections, vectors, errors, storage, metadata.
pub mod collection;
pub mod database;
pub mod distance;
pub mod error;
pub mod metadata;
pub mod tuning;
pub(crate) mod storage;
pub(crate) mod lineage;

// ── Indexing ──────────────────────────────────────────────────────────────────
// Vector index implementations: HNSW, DiskANN, IVF, sparse, multi-vector.
pub mod indexing;
pub use indexing::{hnsw, ivf, sparse, multivec, incremental, graph_vector_index, hybrid_ann, quantization, float16, multimodal_index};
#[cfg(feature = "diskann")]
pub use indexing::{diskann, tiered_ann};

// ── Search & Query ───────────────────────────────────────────────────────────
pub mod search;
pub use search::{query_builder, query_lang, query_explain, query_planner, nl_filter, cross_collection, federated, routing, reranker};
#[cfg(feature = "experimental")]
pub use search::collaborative_search;

// ── Storage & Persistence ────────────────────────────────────────────────────
pub mod persistence;
pub use persistence::{wal, transaction, time_travel, backup, managed_backup, cloud_storage, versioning, migrations, snapshot_replication, tiered, shard, sync_protocol};

// ── Embeddings & ML ──────────────────────────────────────────────────────────
pub mod ml;
pub use ml::{auto_embed, embeddings_gateway, local_inference, model_registry, finetuning, matryoshka, rag, dimreduce, multimodal};

// ── Enterprise Features ──────────────────────────────────────────────────────
// Security, encryption, multi-tenancy, RBAC, Raft consensus.
pub mod enterprise;
pub use enterprise::{security, raft, tenant_isolation, namespace, autoscaling};
#[cfg(feature = "encryption")]
pub use enterprise::encryption;

// ── Observability ────────────────────────────────────────────────────────────
// Telemetry, drift detection, anomaly detection, profiling.
pub mod observe;
pub use observe::{telemetry, observability, anomaly, drift, profiler, otel_service};

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
pub use services::{ingestion_service, text_collection, pitr_service, multimodal_service, tiered_service};
#[cfg(feature = "experimental")]
pub use services::{adaptive_service, plugin_runtime, ingestion_pipeline};

// ── Next-Gen Services ────────────────────────────────────────────────────────
pub use services::{
    streaming_ingest, adaptive_optimizer, wasm_sdk, managed_embeddings,
    time_travel_query, graphrag_service, edge_runtime, nl_filter_parser,
    incremental_sync, visual_explorer,
};

// ── Experimental / Advanced ──────────────────────────────────────────────────
// Features under active development. APIs may change without notice.
// Enable experimental modules with: --features experimental
pub mod experimental;

// Backward-compatible re-exports so `crate::module_name` still works
pub use experimental::clustering;
pub use experimental::dedup;
pub use experimental::gpu;
pub use experimental::graph;
pub use experimental::plugin;
pub use experimental::temporal;
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
#[cfg(feature = "experimental")]
pub use experimental::crdt;
#[cfg(feature = "experimental")]
pub use experimental::distributed_hnsw;
#[cfg(feature = "experimental")]
pub use experimental::edge_optimized;
#[cfg(feature = "experimental")]
pub use experimental::edge_partitioning;
#[cfg(feature = "experimental")]
pub use experimental::edge_runtime;
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
pub use collection::{Collection, CollectionConfig, CollectionIter, CollectionStats, QueryCacheConfig, QueryCacheStats, SearchExplain, SearchResult};
pub use database::{CollectionRef, Database, DatabaseConfig, ExportEntry};
pub use database::collection_ref::SearchParams;
pub use distance::DistanceFunction;
pub use error::{ErrorCode, NeedleError, Recoverable, RecoveryHint, Result};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats, SearchStats};
pub use metadata::{Filter, MetadataStore};
pub use multivec::{MultiVector, MultiVectorConfig, MultiVectorIndex, MultiVectorSearchResult};
pub use quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
pub use sparse::{SparseDistance, SparseIndex, SparseVector};
pub use tuning::{
    auto_tune, IndexRecommendation, IndexSelectionConstraints, PerformanceProfile,
    RecommendedIndex, TuningConstraints, TuningResult, quick_recommend_index, recommend_index,
    DataProfile, DataProfiler, SmartIndexSelection, SmartIndexSelector,
    AdaptiveRecommendation, AdaptiveTuner, MigrationState, MigrationStatus,
    OnlineMigrationManager, WorkloadObservation,
};

// Automatic embedding generation
pub use auto_embed::{
    AutoEmbedConfig, AutoEmbedCollectionBuilder, AutoEmbedStats, AutoEmbedder,
    EmbeddingBackend, EmbeddingModelManager, ModelEntry, ModelHub, ModelType,
    TextFirstCollection, TextInsertable, TextSearchResult,
};

// ── Beta & Experimental API ──────────────────────────────────────────────────
// Beta and experimental types have been moved behind dedicated module paths.
// Use `needle::beta_api::*` for beta types and `needle::experimental_api::*`
// for experimental types.
pub mod beta_api;
pub mod experimental_api;

#[cfg(feature = "hybrid")]
pub use hybrid::{
    AdaptiveFusion, AdaptiveFusionStats, Bm25Index, HybridConfig, HybridSearchResult,
    LearnedWeightStats, QueryFeatures, QueryType, RrfConfig, SearchFeedback,
    reciprocal_rank_fusion,
};

#[cfg(feature = "server")]
pub use server::{ServerConfig, serve};

#[cfg(feature = "async")]
pub use async_api::{AsyncDatabase, AsyncDatabaseConfig, BatchOperationBuilder, BatchResult, ExportStream, SearchStream};

#[cfg(feature = "metrics")]
pub use metrics::{
    metrics, NeedleMetrics, 
    generate_grafana_dashboard, generate_alerting_rules,
    GrafanaDashboardConfig, AlertingConfig,
    AnomalyDetector, AnomalyResult,
};

#[cfg(feature = "embeddings")]
pub use embeddings::{EmbedderBuilder, EmbedderConfig, EmbeddingError, PoolingStrategy, TextEmbedder};

#[cfg(feature = "web-ui")]
pub use web_ui::{WebUiConfig, WebUiState, serve_web_ui, serve_web_ui_default, create_web_ui_router};

#[cfg(feature = "embedding-providers")]
pub use embeddings_provider::{
    BatchConfig, CachedProvider, CohereConfig, CohereInputType, EmbeddingProvider,
    EmbeddingProviderError, MockConfig, MockProvider, OllamaConfig, OllamaProvider,
    OpenAIConfig, OpenAIProvider, RateLimiter,
};

/// Prelude module for convenient imports.
///
/// ```rust
/// use needle::prelude::*;
/// ```
pub mod prelude {
    pub use crate::collection::{Collection, CollectionConfig, SearchResult};
    pub use crate::database::{CollectionRef, Database, DatabaseConfig};
    pub use crate::database::collection_ref::SearchParams;
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
