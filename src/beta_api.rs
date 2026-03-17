//! Beta API — feature-complete types that may see breaking changes before 1.0.
//!
//! Access these via `needle::beta_api::*`.

#[cfg(feature = "observability")]
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
#[cfg(feature = "experimental")]
pub use crate::clustering::{
    elbow_method, silhouette_score, ClusteringConfig, HierarchicalClustering, KMeans, Linkage,
    MiniBatchKMeans,
};
#[cfg(feature = "experimental")]
pub use crate::dedup::{DeduplicationConfig, DuplicateDetector, DuplicateGroup, DuplicateResult};
pub use crate::dimreduce::{NeighborEmbedding, RandomProjection, PCA};
#[cfg(feature = "experimental")]
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_dimreduce_random_projection_accessible() {
        let _rp = RandomProjection::new(10, 3);
    }

    #[test]
    fn test_namespace_manager_accessible() {
        let manager = NamespaceManager::new();
        assert!(manager.list_namespaces().is_empty());
    }

    #[test]
    fn test_backup_config_default() {
        let config = BackupConfig::default();
        assert!(config.compression);
        assert!(config.verify);
    }
}
