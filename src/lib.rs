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

pub mod anomaly;
pub mod backup;
pub mod cloud_storage;
pub mod clustering;
pub mod collection;
pub mod crdt;
pub mod database;
pub mod dedup;
pub mod dimreduce;
pub mod diskann;
pub mod distance;
pub mod drift;
pub mod encryption;
pub mod error;
pub mod float16;
pub mod gpu;
pub mod graph;
pub mod hnsw;
pub mod ivf;
pub mod knowledge_graph;
pub mod langchain;
pub mod lineage;
pub mod metadata;
pub mod multivec;
pub mod namespace;
pub mod nl_filter;
pub mod optimizer;
pub mod profiler;
pub mod quantization;
pub mod query_lang;
pub mod raft;
pub mod rebalance;
pub mod rag;
pub mod reranker;
pub mod routing;
pub mod security;
pub mod shard;
pub mod sparse;
pub mod storage;
#[cfg(feature = "server")]
pub mod streaming;
pub mod telemetry;
pub mod temporal;
pub mod tiered;
pub mod tuning;
pub mod versioning;
pub mod wal;

#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "server")]
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

// Re-export main types at crate root
pub use collection::{Collection, CollectionConfig, CollectionIter, CollectionStats, SearchResult};
pub use database::{CollectionRef, Database, DatabaseConfig, ExportEntry};
pub use distance::DistanceFunction;
pub use error::{NeedleError, Result};
pub use hnsw::{HnswConfig, HnswIndex, HnswStats};
pub use metadata::{Filter, MetadataStore};
pub use multivec::{MultiVector, MultiVectorConfig, MultiVectorIndex, MultiVectorSearchResult};
pub use quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
pub use sparse::{SparseDistance, SparseIndex, SparseVector};
pub use tuning::{auto_tune, PerformanceProfile, TuningConstraints, TuningResult};

// Analytics and advanced features
pub use anomaly::{
    DistanceOutlierDetector, EnsembleAnomalyDetector, IsolationForest, LocalOutlierFactor,
    StatisticalOutlierDetector,
};
pub use backup::{BackupConfig, BackupManager, BackupMetadata, BackupType};
pub use clustering::{
    ClusteringConfig, HierarchicalClustering, KMeans, Linkage, MiniBatchKMeans, elbow_method,
    silhouette_score,
};
pub use dedup::{DeduplicationConfig, DuplicateDetector, DuplicateGroup, DuplicateResult};
pub use dimreduce::{NeighborEmbedding, PCA, RandomProjection};
pub use graph::{Community, GraphConfig, GraphPath, NeighborhoodResult, SemanticGraph};
pub use namespace::{
    AccessControl, AccessLevel, Namespace, NamespaceCollection, NamespaceManager, TenantConfig,
};
#[cfg(feature = "server")]
pub use streaming::{
    ChangeEvent, ChangeEventFilter, ChangeStream, EventLog, OperationType, PubSub, ReplayOptions,
    ResumeToken, StreamError, StreamManager, StreamManagerConfig, StreamResult, StreamStats,
    Subscriber,
};

// Next-gen features
pub use crdt::{CRDTVector, Delta, HLC, Operation, VectorCRDT};
pub use diskann::{DiskAnnConfig, DiskAnnIndex, DiskAnnResult};
pub use drift::{DriftConfig, DriftDetector, DriftReport};
pub use encryption::{EncryptedVector, EncryptionConfig, KeyManager, VectorEncryptor};
pub use gpu::{DataType, DistanceType, GpuAccelerator, GpuBackend, GpuConfig, GpuDevice, GpuMetrics};
pub use knowledge_graph::{Entity, KnowledgeGraph, KnowledgeGraphConfig, Relation};
pub use lineage::{LineageTracker, SourceInfo, Transformation, VectorLineage};
pub use nl_filter::NLFilterParser;
pub use optimizer::{QueryOptimizer, QueryStats};
pub use profiler::{OptimizationHint, PlanNode, QueryProfile, QueryProfiler};
pub use query_lang::{
    Query, QueryContext, QueryError, QueryExecutor, QueryParser, QueryPlan, QueryResponse,
    QueryResult, QueryValidator,
};
pub use raft::{
    AppendEntries, Command, FileStorage as RaftFileStorage, LogEntry, MemoryStorage as RaftMemoryStorage,
    PersistentState, RaftNode, RaftState, RaftStorage, RequestVote, Snapshot, SnapshotBuilder, SnapshotMetadata,
};
pub use rag::{Chunk, ChunkingStrategy, RagConfig, RagPipeline};
pub use rebalance::{
    DryRunMigration, MigrationCheckpoint, MigrationSource, MigrationState, MigrationTarget,
    MigrationTask, PlanState, RebalanceConfig, RebalanceCoordinator, RebalancePlan, RebalanceStats,
    TransferBatch, VectorTransfer,
};
pub use telemetry::{Metric, MetricValue, Span, SpanStatus, Telemetry, TelemetryConfig, TraceContext};
pub use temporal::{DecayFunction, TemporalConfig, TemporalIndex, VectorVersion};
pub use tiered::{StorageTier, TierPolicy, TieredStorage, VectorMetadata as TieredVectorMetadata};
pub use versioning::{Branch, ChangeType, Commit, VectorDiff, VectorRepo};

// New features: IVF indexing, reranking, and half-precision floats
pub use float16::{Bf16, Bf16Vector, F16, F16Vector, HalfPrecision};
pub use ivf::{ClusterStats, IvfConfig, IvfError, IvfIndex, IvfResult};
pub use reranker::{
    CohereConfig as CohereRerankerConfig, CohereReranker, EnsembleReranker,
    HuggingFaceConfig as HuggingFaceRerankerConfig, HuggingFaceReranker,
    NoOpReranker, Reranker, RerankerError, RerankerResult, RerankResult,
};
pub use routing::{
    AggregatedResults, LoadBalancing, QueryRouter, ResultCollector, RouteConfig, RoutingError,
    RoutingResult, RouterStatsSnapshot, ShardSearchResult,
};
pub use shard::{
    ConsistentHashRing, RebalanceMove, ShardConfig, ShardError, ShardId, ShardInfo, ShardManager,
    ShardResult, ShardState, ShardStats, ShardStatsSnapshot, ShardedCollection,
};
pub use wal::{
    BatchEntry as WalBatchEntry, Lsn, WalApplicator, WalConfig, WalEntry, WalManager, WalRecord,
    WalStats,
};

// Cloud storage
pub use cloud_storage::{
    AzureBlobBackend, AzureBlobConfig, CacheConfig, CachedBackend, CacheStats,
    ConnectionPool, GCSBackend, GCSConfig, LocalBackend, MultipartUploader,
    PoolStats, RetryPolicy, S3Backend, S3Config, StorageBackend, StorageConfig,
    StreamChunk, StreamingReader, StreamingWriter,
};

#[cfg(feature = "hybrid")]
pub use hybrid::{Bm25Index, HybridSearchResult, RrfConfig, reciprocal_rank_fusion};

#[cfg(feature = "server")]
pub use server::{ServerConfig, serve};

#[cfg(feature = "server")]
pub use async_api::{AsyncDatabase, AsyncDatabaseConfig, BatchOperationBuilder, BatchResult, ExportStream, SearchStream};

#[cfg(feature = "metrics")]
pub use metrics::{metrics, NeedleMetrics};

#[cfg(feature = "embeddings")]
pub use embeddings::{EmbedderBuilder, EmbedderConfig, EmbeddingError, PoolingStrategy, TextEmbedder};

#[cfg(feature = "web-ui")]
pub use web_ui::{WebUiConfig, WebUiState, serve_web_ui, serve_web_ui_default, create_web_ui_router};

// Security - RBAC and Audit Logging
pub use security::{
    AccessControl as SecurityAccessControl, AccessController, AuditAction, AuditEvent,
    AuditLogger, AuditQuery, AuditResult, FileAuditLog, FileAuditLogConfig, InMemoryAuditLog,
    Permission, PermissionGrant, PolicyDecision, Resource, Role, SecurityContext, User,
};

#[cfg(feature = "embedding-providers")]
pub use embeddings_provider::{
    BatchConfig, CachedProvider, CohereConfig, CohereInputType, EmbeddingProvider,
    EmbeddingProviderError, MockConfig, MockProvider, OllamaConfig, OllamaProvider,
    OpenAIConfig, OpenAIProvider, RateLimiter,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::collection::{Collection, SearchResult};
    pub use crate::database::Database;
    pub use crate::distance::DistanceFunction;
    pub use crate::error::{NeedleError, Result};
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
