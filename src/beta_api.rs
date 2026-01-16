//! Beta API — feature-complete types that may see breaking changes before 1.0.
//!
//! Access these via `needle::beta_api::*`.

pub use crate::anomaly::{
    DistanceOutlierDetector, EnsembleAnomalyDetector, IsolationForest, LocalOutlierFactor,
    StatisticalOutlierDetector,
};
pub use crate::backup::{
    BackupConfig, BackupManager, BackupMetadata, BackupType, CloudProvider, CloudSyncConfig,
    CloudSyncResult, ConsistencyLevel, FollowerState, FollowerStatus, IncrementalBackupInfo,
    IncrementalBackupManager, IncrementalState, PitrConfig, ReplicationConfig, ReplicationLeader,
    RestorePoint, RestorePointType, SnapshotSegment, WalEntry, WalOperation,
};
pub use crate::clustering::{
    elbow_method, silhouette_score, ClusteringConfig, HierarchicalClustering, KMeans, Linkage,
    MiniBatchKMeans,
};
pub use crate::dedup::{DeduplicationConfig, DuplicateDetector, DuplicateGroup, DuplicateResult};
pub use crate::dimreduce::{NeighborEmbedding, RandomProjection, PCA};
pub use crate::graph::{Community, GraphConfig, GraphPath, NeighborhoodResult, SemanticGraph};
pub use crate::namespace::{
    AccessControl, AccessLevel, Namespace, NamespaceCollection, NamespaceManager, TenantConfig,
};
#[cfg(feature = "server")]
pub use crate::streaming::{
    ChangeEvent, ChangeEventFilter, ChangeStream, EventLog, OperationType, PubSub, ReplayOptions,
    ResumeToken, StreamError, StreamManager, StreamManagerConfig, StreamResult, StreamStats,
    Subscriber,
};
