//! Experimental API — types under active development. APIs may change without notice.
//!
//! Access these via `needle::experimental_api::*`.

#[cfg(feature = "experimental")]
pub use crate::adaptive_index::{
    AdaptiveIndex, AdaptiveIndexConfig, IndexRecommendation as AdaptiveIndexRecommendation,
    IndexStrategy, MigrationEvent, SearchExplanation,
    WorkloadProfile as AdaptiveWorkloadProfile,
};
#[cfg(feature = "diskann")]
pub use crate::diskann::{DiskAnnConfig, DiskAnnIndex, DiskAnnResult};
#[cfg(feature = "experimental")]
pub use crate::distributed_hnsw::{
    BatchInsertResult as DistributedBatchInsertResult, DistributedHnsw, DistributedHnswConfig,
    DistributedHnswStats, DistributedQueryBuilder, DistributedSearchResult, ShardSearchResult as DistributedShardSearchResult,
};
pub use crate::incremental::{
    BulkInsertResult, IncrementalConfig, IncrementalIndex, IncrementalSearchResult,
    IncrementalStats, MergeResult, OptimizationResult,
};
pub use crate::graph_vector_index::{
    ConnectedEntity, EdgeType, GraphEdge, GraphPath as GraphVectorPath, GraphVectorConfig,
    GraphVectorIndex, GraphVectorSearchResult, GraphVectorStats,
};
pub use crate::drift::{
    AdaptiveThreshold, AdaptiveThresholdConfig, DriftAlert, DriftConfig, DriftConfigBuilder,
    DriftDetector, DriftMetrics, DriftReport, DriftSeverity, DriftTrend, MultiBaselineDetector,
    NamedBaseline, RealTimeCheckResult, RealTimeDriftMonitor, RealTimeMonitorConfig,
    RealTimeMonitorStats, RecoveryDetector, RecoveryState, VectorStats,
};
#[cfg(feature = "experimental")]
pub use crate::vector_streaming::{
    BackpressureConfig, BackpressureController, BackpressureState, BackpressureStats,
    CdcEvent, CdcEventType, CdcStream, ConsumerConfig, ConsumerStats, MessageSource, ProducerConfig,
    ProducerStats, ReplayManager, StreamMetrics, StreamMetricsSnapshot, StreamOp, StreamProcessor,
    StreamSnapshot, VectorConsumer, VectorFormat, VectorMessage, VectorProducer, VectorStreamPipeline,
};
#[cfg(feature = "encryption")]
pub use crate::encryption::{
    EncryptedVector, EncryptionConfig, KekProvider, KeyManager, KeyRotationManager,
    LocalKekProvider, VectorEncryptor, WrappedKey,
};
pub use crate::federated::{
    ConsistencyLevel as FederatedConsistencyLevel, CrossInstanceDedup, DedupStrategy,
    DiscoveryConfig, DiscoveryService, Federation, FederationConfig, FederationError,
    FederationHealth, FederationResult, FederatedSearchResult, FederatedSearchResponse,
    HealthCheckResult, HealthMonitor, HealthStatus, InstanceConfig, InstanceInfo,
    InstanceRegistry, MergeStrategy, QueryPlan as FederatedQueryPlan,
    QueryPlanner as FederatedQueryPlanner, ResultMerger, RoutingStrategy,
};
pub use crate::finetuning::{
    ContrastivePair, EmbeddingStore, FineTuneConfig, FineTuner, FineTunerState, FineTunerStats,
    Interaction, InteractionType, LinearTransform, LossFunction, SharedFineTuner, TrainingBatch,
    TrainingResult, Triplet,
};
pub use crate::gpu::{DataType, DistanceType, ExecutionBackend, FallbackSearchResult, GpuAccelerator,
    GpuBackend, GpuConfig, GpuDevice, GpuMetrics, GpuResidentIndex, HardwareCapabilities,
    KernelDispatch, KernelProfile, MultiGpuConfig, MultiGpuShardManager, ShardStrategy,
    TransparentFallbackManager, select_kernel,
};
pub use crate::hybrid_ann::{
    HybridConfig as HybridAnnConfig, HybridSearch, HybridSearchResult as HybridAnnSearchResult,
    HybridSearchStats as HybridAnnStats, QualityAwareSearch, RecallStats, SearchStrategy,
};
#[cfg(feature = "experimental")]
pub use crate::knowledge_graph::{
    ChunkContext, Entity, EntityLinker, GraphRAGConfig, GraphRAGQueryBuilder, GraphRAGRetriever,
    GraphSearchResult, GraphStats, KnowledgeGraph, KnowledgeGraphConfig, MultiHopReasoner,
    ReasoningPath, Relation,
};
#[cfg(feature = "experimental")]
pub use crate::learned_tuning::{
    AdaptiveExecutor, LearnedTuner, LearnedTunerState, LearnedTunerStats, QueryFeedback,
    RecommendedParams, TunerConfig, WorkloadProfile,
};
#[cfg(feature = "experimental")]
pub use crate::llm_cache::{
    AdaptiveThresholdConfig as LlmAdaptiveThresholdConfig, CacheHit, CacheWarmingConfig, CachedLlm,
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, CostAnalytics, CostSavingsReport,
    EnhancedLlmCache, LlmCache, LlmCacheConfig, LlmCacheQueryBuilder, LlmCacheStats,
    MultiTierCache, MultiTierStats, OpenAIProxy, WarmingQuery,
};
pub use crate::multimodal::{
    CrossModalConfig, CrossModalSearch, EmbedInput, EmbedderBackend,
    EmbedderStats as MultiModalEmbedderStats, EmbeddingModelRegistry, FusionStrategy,
    ImagePreprocessor, LateFusion, Modality, ModalityStats, ModelManifest, MultiModalConfig,
    MultiModalDocument, MultiModalEmbedder, MultiModalEmbedding, MultiModalQuery,
    MultiModalSearchResult, TextPreprocessor, UnifiedMultiModalIndex,
};
pub use crate::query_builder::{
    AlternativeQuery, CollectionProfile, CollectionStats as QueryCollectionStats, CostEstimate,
    FieldProfile, FieldSuggestion, FieldType, HintCategory, HintImpact, HintSeverity, IndexProfile,
    OptimizationHint, QueryAnalysis, QueryAnalyzer, QueryBuildResult, QueryClass, QueryComplexity,
    QueryExplanation, QuerySuggestion, SuggestionType, VisualQueryBuilder,
};
pub use crate::query_lang::{
    CollectionStatistics, CostBasedOptimizer, CostEstimate as QueryCostEstimate, OptimizedPlan,
    OptimizedStep, Query, QueryContext, QueryError, QueryExecutor, QueryParser, QueryPlan,
    QueryResponse, QueryResult, QueryValidator, SearchStrategy as QuerySearchStrategy,
};
pub use crate::nl_filter::{
    ConversationContext, ConversationalQueryParser, ContextEntry, Entity as NLEntity,
    IntentClassification, NLFilterParser, ParsedQuery, QueryBuilder as NLQueryBuilder,
    QueryIntent, QuerySuggester, TemporalConstraint,
};
pub use crate::raft::{Command, NodeId, RaftConfig, RaftNode, RaftState, RaftStorage};
pub use crate::rag::{Chunk, ChunkingStrategy, RagConfig, RagPipeline, RagPipelineBuilder, quick_rag_pipeline};
pub use crate::telemetry::{Metric, MetricValue, Span, SpanStatus, Telemetry, TelemetryConfig, TraceContext};
pub use crate::temporal::{DecayFunction, TemporalConfig, TemporalIndex, VectorVersion};
pub use crate::time_travel::{
    Branch as MvccBranch, BranchManager, ConflictStrategy, ConflictType, DateTimeComponents,
    GcResult, MergeConflict, MergeResult as MvccMergeResult, MetadataDiff, MvccConfig, NamedTime,
    Snapshot, TimeExpression, TimeMarker, TimeTravelIndex, TimeTravelQueryBuilder,
    TimeTravelSearchResult, TimeTravelStats, VectorDiff as TimeTravelVectorDiff,
    VectorVersion as MvccVectorVersion,
};
pub use crate::autoscaling::{
    AutoScaler, MetricPoint, ScalingAction, ScalingConfig, ScalingDecision, ScalingReason,
    ScheduledScaling, SeasonalityPattern, ShardMetrics, SharedAutoScaler,
};
pub use crate::tiered::{StorageTier, TierPolicy, TieredStorage, VectorMetadata as TieredVectorMetadata};
#[cfg(feature = "diskann")]
pub use crate::tiered_ann::{
    Tier, TieredConfig, TieredIndex, TieredQueryBuilder, TieredSearchResult,
    TieredStats,
};
pub use crate::versioning::{Branch, ChangeType, Commit, VectorDiff, VectorRepo};
pub use crate::sync_protocol::{
    SegmentInfo, SyncConfig, SyncManager as VectorSyncManager, SyncManifest,
    SyncStats as VectorSyncStats, SyncTarget,
};
pub use crate::tenant_isolation::{
    AccessPolicy as TenantAccessPolicy, AuditLogEntry as TenantAuditLogEntry,
    BillingSummary, ExportFormat as TenantExportFormat, GdprExport,
    MeteringConfig, Permission as TenantPermission, ResourceUsage,
    Tenant as IsolatedTenant, TenantConfig as TenantIsolationConfig, TenantId, TenantManager,
    TenantRateLimiter, TenantRole, TenantStatus as IsolatedTenantStatus, UsageEvent as TenantUsageEvent,
    UsageEventKind, UsageMeter,
};
pub use crate::managed_backup::{
    BackupChain, BackupChainManager, BackupManifest, BackupSchedule, BackupStats,
    BackupStatus as ManagedBackupStatus, BackupTarget, ManagedBackupConfig, ManagedBackupManager,
    PitrManager, PitrResult, RestoreOptions, RestoreTarget, RetentionPolicy,
};
#[cfg(feature = "experimental")]
pub use crate::agentic_memory::{
    AgentMemory, ConversationConfig, ConversationRole, ConversationTracker, ConversationTurn,
    ContextSection, ContextWindowManager, DecayFunction as AgentDecayFunction,
    Memory, MemoryConfig, MemoryType, RecallResult, ToolCallCache, ToolCallCacheStats,
};
#[cfg(feature = "experimental")]
pub use crate::analytics::{
    Alert, AlertCondition, AlertManager, AlertRule, AlertSeverity, AnalyticsConfig,
    AnalyticsDashboard, CollectionAnalytics, DashboardInsights, MetricsStore, QueryEvent,
    QueryPattern, QueryTracker, SlowQuery, TimeSeries, TimeSeriesPoint,
    generate_dashboard_html,
};
#[cfg(feature = "experimental")]
pub use crate::cloud_control::{
    ApiKey, AutoRecoveryConfig, BillingInfo, ConnectionState, ControlPlane, ControlPlaneConfig,
    InstanceHealth, InstanceRecoveryManager, InstanceStatus, LimitCheckResult,
    LimitViolation, ManagedInstance, NeedleCloudClient, OrchestratorConfig, OverageCharge,
    PaymentStatus, Permission as CloudPermission, RecoveryAction, RecoveryStatus,
    Region, RegionHealth, RegionRouter, RegionalEndpoint,
    ResourceTier, RoutingStrategy as CloudRoutingStrategy, SdkConfig, SdkStats,
    ServiceOrchestrator, SlaBreach, SlaBreachType, SlaMonitor, SlaPolicy, SlaReport,
    SupportLevel, Tenant as CloudTenant,
    TenantConfig as CloudTenantConfig, TenantStatus as CloudTenantStatus, TenantUsage,
    TierLimits, UsageEvent as CloudUsageEvent,
    UsageEventType, UsageSummary, UsageUtilization,
};
pub use crate::matryoshka::{
    MatryoshkaConfig, MatryoshkaEmbedding, MatryoshkaIndex, MatryoshkaStats, SearchStrategy as MatryoshkaSearchStrategy,
};
pub use crate::model_registry::{
    AutoEmbedConfig as HubAutoEmbedConfig, AutoEmbedHub, BenchmarkResult, LocalModelInference,
    ModelId, ModelInfo, ModelPerformanceTracker, ModelRegistry, ModelSelector, ModelStatus,
    RegistryConfig, UseCase,
};
#[cfg(feature = "experimental")]
pub use crate::playground::{
    Bookmark, ClusterInfo, CustomDataset, Dataset, DatasetInfo, Difficulty, ExecutionResult,
    HistoryEntry, Playground, PlaygroundConfig, PlaygroundState, ProjectionEngine,
    ProjectionInput, ProjectionMethod, ProjectionScene, SceneBounds, Theme, Tutorial, TutorialInfo,
    TutorialStep, VisualizationConfig, VisualizationPoint, generate_playground_html,
};
#[cfg(feature = "experimental")]
pub use crate::zero_copy::{
    ArrowBatch, ArrowField, SharedMemoryHandle, VectorBatch, ZeroCopyBuffer,
};
#[cfg(feature = "experimental")]
pub use crate::edge_runtime::{
    EdgeCacheConfig, EdgeConfig, EdgeHnswConfig, EdgeManifest, EdgeRuntime, EdgeRuntimeStats,
    EdgeStorage, IndexSegment, InMemoryEdgeStorage, Platform, SearchCache, SegmentMetadata,
};
#[cfg(feature = "experimental")]
pub use crate::edge_partitioning::{
    ClusterPartitioner, HashPartitioner, HierarchicalPartitioner, PartitionConfig, PartitionId,
    PartitionManager, PartitionManifest, PartitionMetadata, PartitionRouter,
};
#[cfg(feature = "experimental")]
pub use crate::platform_adapters::{
    ChunkedEdgeStorage, CloudflareKvAdapter, CloudflareR2Adapter, DenoKvAdapter,
    TieredEdgeStorage, VercelBlobAdapter, VercelEdgeConfigAdapter,
};
pub use crate::embeddings_gateway::{
    EmbeddingResult, EmbeddingsGateway, GatewayConfig, GatewayMetrics, ProviderConfig,
    ProviderMetrics, ProviderRouter, ProviderStatus, ProviderType, RoutingStrategy as GatewayRoutingStrategy,
    SemanticCache, BatchEmbeddingResult,
};
#[cfg(feature = "experimental")]
pub use crate::collaborative_search::{
    Annotation, AnnotationDelta, AnnotationMergeResult, CollaborativeCollection,
    CollaborativeCollectionStats, CollaborativeDelta, CollaborativeEvent, CollaborativeMergeResult,
    CollaborativeSearchResult, CollaborativeVector, EventReceiver, LiveQuery, LiveQueryConfig,
    Presence, PresenceEvent, PresenceStatus, PresenceTracker, Session, SessionAccess, SessionInfo,
    SessionManager, SharedSearch, SyncManager, SyncStats,
};
pub use crate::cross_collection::{
    CollectionFilter, CrossCollectionAnalytics, CrossCollectionConfig, CrossCollectionQueryBuilder,
    CrossCollectionResult, CrossCollectionSearch, CrossCollectionStats,
    ScoreAggregation as CrossCollectionScoreAggregation,
};
pub use crate::migrations::{
    CompatibilityResult, IndexType as MigrationIndexType, IssueSeverity, Migration,
    MigrationContext, MigrationError, MigrationManager, MigrationOperation, MigrationPreview,
    MigrationRecord, MigrationResult, MigrationStatus as SchemaMigrationStatus, SchemaVersion, ValidationIssue,
    built_in_migrations,
};
pub use crate::float16::{Bf16, Bf16Vector, F16, F16Vector, HalfPrecision};
pub use crate::ivf::{ClusterStats, IvfConfig, IvfError, IvfIndex, IvfResult};
pub use crate::reranker::{
    CohereConfig as CohereRerankerConfig, CohereReranker, EnsembleReranker,
    HuggingFaceConfig as HuggingFaceRerankerConfig, HuggingFaceReranker,
    NoOpReranker, Reranker, RerankerError, RerankerResult, RerankResult,
};
pub use crate::routing::{LoadBalancing, QueryRouter, RouteConfig, RoutingError, RoutingResult};
pub use crate::shard::{
    CrossShardSearchConfig, CrossShardSearchResult, ShardConfig, ShardError, ShardId, ShardInfo,
    ShardManager, ShardSearchResult, ShardSearchable, ShardState, ShardedCollection,
    merge_shard_results,
};
pub use crate::wal::{Lsn, WalConfig, WalManager, WalStats};
pub use crate::plugin::{
    DistancePlugin, HookResult, Plugin, PluginError, PluginInfo, PluginManager,
    PluginManifest, PluginType, PostSearchHook, PreSearchHook, SearchHookResult,
};
pub use crate::transaction::{
    IsolationLevel, SnapshotView, Transaction, TransactionId, TransactionLog,
    TransactionManager, TransactionStats, TransactionStatus, WriteOperation,
};
#[cfg(feature = "experimental")]
pub use crate::optimizer::{
    AdaptiveOptimizer, CostEstimate as OptimizerCostEstimate, CostModel,
    CollectionStatistics as OptimizerCollectionStatistics, FeedbackCollector,
    IndexType as OptimizerIndexType, QueryOptimizer, QueryPlan as OptimizerQueryPlan, QueryStrategy,
};
#[cfg(feature = "experimental")]
pub use crate::ingestion_pipeline::{
    BackpressureState as PipelineBackpressureState, Checkpoint, CheckpointStore,
    DeadLetterQueue, FailedRecord, IdentityTransform, InMemoryCheckpointStore,
    InMemorySource, InMemorySink, IngestionPipeline, PipelineBuilder, PipelineConfig,
    PipelineStats, Record, RecordData, RetryPolicy as PipelineRetryPolicy,
    Sink, Source, TextToVectorTransform, Transform,
};
pub use crate::cloud_storage::{
    AzureBlobBackend, AzureBlobConfig, CacheConfig, CachedBackend, GCSBackend,
    GCSConfig, LocalBackend, RetryPolicy, S3Backend, S3Config, StorageBackend, StorageConfig,
};
pub use crate::security::{
    AccessControl as SecurityAccessControl, AccessController, AuditAction, AuditEvent,
    AuditLogger, AuditQuery, AuditResult, FileAuditLog, FileAuditLogConfig, InMemoryAuditLog,
    Permission, PermissionGrant, PolicyDecision, Resource, Role, SecurityContext, User,
};
pub use crate::haystack::{
    ContentType, DocumentStoreConfig, DuplicatePolicy, HaystackDocument, NeedleDocumentStore,
};
#[cfg(feature = "integrations")]
pub use crate::langchain::{
    Document, DocumentWithScore, FilterBuilder, NeedleVectorStore, NeedleVectorStoreConfig,
    RelevanceScoreFunction, CollectionStats as LangChainCollectionStats,
};
#[cfg(feature = "integrations")]
pub use crate::llamaindex::{
    ChatMessage as LlamaIndexChatMessage, ChunkConfig, ConversationMemory, DocumentChunker,
    MemoryConfig as ConversationMemoryConfig, MessageRole, NeedleIndexConfig, NeedleVectorStoreIndex,
    NodeRelationship, NodeWithScore, RelatedNode, RetrieverQueryEngine, TextNode,
};
pub use crate::semantic_kernel::{
    MemoryQueryResult, MemoryRecord, NeedleMemoryStore,
};
#[cfg(feature = "experimental")]
pub use crate::streaming_upsert::{
    ArrowVectorBatch, BackpressureGate, BatchInsertResult as StreamingBatchInsertResult,
    Frame, FrameType, IngestionEngine, IngestionEngineConfig, IngestionMetrics,
    PressureLevel, StreamingUpsertConfig, StreamingUpsertConfigBuilder,
    ThroughputTracker, UpsertPipeline, UpsertStats, VectorBatchBuilder, VectorRecord,
    VectorBatch as StreamingVectorBatch,
};
#[cfg(feature = "experimental")]
pub use crate::adaptive_runtime::{
    ActiveIndexType, AdaptiveRuntime, AdaptiveRuntimeConfig, MigrationPhase, MigrationPolicy,
    OperationType as AdaptiveOperationType, RuntimeHealth, SelectionReason,
    WorkloadClassifier, WorkloadSample,
};
#[cfg(feature = "experimental")]
pub use crate::python_arrow::{
    BatchInsertRequest, BatchInsertResponse, BatchSearchRequest, BatchSearchResponse,
    CollectionInfo as PythonCollectionInfo, DatabaseInfo as PythonDatabaseInfo,
    MemoryOrder, OperationStats, PythonSearchResult, TypedValue, VectorArrayView,
};
pub use crate::snapshot_replication::{
    CompressionType, DeltaOperation, FollowerInfo, FollowerSyncStatus, InMemoryTransport,
    ReplicationConfig as SnapshotReplicationConfig, ReplicationFollower, ReplicationHealth,
    ReplicationLeaderNode, Snapshot as ReplicationSnapshot, SnapshotDelta, SnapshotTransport,
    SnapshotType, WalReplicationEntry, compute_replication_health,
};
pub use crate::local_inference::{
    CachedModel, InferenceConfig, InferenceEngine, InferenceStats, ModelCache,
    ModelQuantization, ModelSource, ModelSpec, SimpleTokenizer,
    builtin_model_registry, find_model,
};
#[cfg(feature = "experimental")]
pub use crate::graphrag_index::{
    EntityCategory, EntityExtractor, EntityPattern, ExtractionConfig, ExtractedEntity,
    ExtractedRelation, GraphEdge as GraphRagEdge, GraphNode, GraphRagConfig, GraphRagIndex,
    GraphRagResult, GraphStats as GraphRagStats, KnowledgeGraphStore, TraversalPath,
};
#[cfg(feature = "experimental")]
pub use crate::edge_optimized::{
    CompactIndex, EdgePartitionManager, EdgePersistence, EdgePlatform,
    EdgeSearchResult as EdgeOptSearchResult,
    EdgeRuntimeStats as EdgeOptRuntimeStats, InMemoryPersistence, OfflineCacheConfig,
    OfflineSearchCache, PartitionConfig as EdgePartitionConfig,
    PartitionManifest as EdgeOptPartitionManifest, PlatformLimits,
};
pub use crate::query_explain::{
    AdaptiveEfTuner, CostModel as ExplainCostModel, CostModelParams, ExplainPlan, ExplainStrategy, FilterMode,
    OptimizerConfig as ExplainOptimizerConfig, OptimizerStats as ExplainOptimizerStats,
    PlanStep, QueryExplainOptimizer, QueryFeedback as ExplainQueryFeedback, QuerySpec,
};
pub use crate::multimodal_index::{
    FusionStrategy as MultiModalFusionStrategy, Modality as MultiModalModality,
    ModalityConfig, MultiModalDoc, MultiModalDocBuilder, MultiModalIndexConfig,
    MultiModalSearchResult as UnifiedMultiModalSearchResult, MultiModalStats as UnifiedMultiModalStats,
    MultiModalUnifiedIndex as UnifiedMultiModal,
};
#[cfg(feature = "experimental")]
pub use crate::plugin_registry::{
    DistanceFn, L3Distance, LoadedPlugin, NormalizeHook, PluginManifest as RegistryPluginManifest,
    PluginRegistry as PluginMarketplace, PluginRuntimeManager, PluginStatus,
    PostSearchHookFn, PreSearchHookFn, PublishedPlugin, RegistryConfig as MarketplaceConfig,
    RegistryPluginType, ThresholdFilter, TransformFn, TruncateTransform,
};
pub use crate::ingestion_service::{
    BackpressureLevel, DeadLetterRecord, IngestionService, IngestionServiceConfig,
    IngestionServiceConfigBuilder, IngestionStats,
};
#[cfg(feature = "experimental")]
pub use crate::adaptive_service::{
    AdaptiveIndexService, AdaptiveServiceConfig, AdaptiveServiceConfigBuilder,
    AdaptiveStatusReport,
};
pub use crate::text_collection::{
    TextCollection, TextCollectionConfig, TextCollectionConfigBuilder,
    TextSearchResult as TextSearchHit,
};
pub use crate::pitr_service::{
    PitrService, PitrServiceConfig, PitrServiceConfigBuilder, PitrStats,
    RecoveryResult, RecoveryTarget, RestorePoint as PitrRestorePoint,
};
pub use crate::query_planner::{
    CollectionStats as PlannerCollectionStats, PlanStep as PlannerStep,
    PlanStrategy, QueryPlan as PlannerQueryPlan, QueryPlanner, QueryPlannerConfig,
    QueryRequest as PlannerQueryRequest,
};
pub use crate::multimodal_service::{
    ModalInput, MultiModalResult, MultiModalService, MultiModalServiceConfig,
    MultiModalServiceConfigBuilder,
};
pub use crate::tiered_service::{
    MaintenanceReport, Tier as ServiceTier, TierPolicy as ServiceTierPolicy,
    TieredService, TieredServiceConfig, TieredServiceConfigBuilder,
    TieredServiceStats,
};
pub use crate::otel_service::{
    CollectionMetrics as OtelCollectionMetrics,
    MetricsSnapshot, ObservabilityConfig, ObservabilityConfigBuilder,
    ObservabilityService, OperationType as OtelOperationType, SpanRecord,
};
#[cfg(feature = "experimental")]
pub use crate::plugin_runtime::{
    PluginHook, PluginRuntime, PluginRuntimeConfig, PluginRuntimeConfigBuilder,
    HookInfo as PluginHookInfo,
};
