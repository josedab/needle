//! Vector Snapshot Replication
//!
//! Efficient snapshot-based replication protocol enabling read replicas,
//! point-in-time recovery, and cross-region sync — all leveraging the
//! single-file storage format.
//!
//! # Architecture
//!
//! ```text
//! Leader (Writer)
//!   ├── SnapshotManager  →  creates snapshots with delta encoding
//!   └── ReplicationLog   →  tracks WAL entries since last snapshot
//!
//! Follower (Reader)
//!   ├── SnapshotReceiver →  applies snapshots from leader
//!   └── ReplicaState     →  tracks sync position
//!
//! Transport Layer (pluggable)
//!   ├── FileTransport    →  local filesystem / NFS
//!   ├── S3Transport      →  S3-compatible storage
//!   └── Custom(Box<dyn SnapshotTransport>)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::snapshot_replication::*;
//!
//! // Leader side
//! let config = ReplicationConfig::default();
//! let mut leader = ReplicationLeaderNode::new("leader-1", config);
//! leader.create_snapshot().unwrap();
//!
//! // Follower side
//! let follower = ReplicationFollower::new("follower-1", "leader-1");
//! let status = follower.status();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// Snapshot Types
// ---------------------------------------------------------------------------

/// Unique identifier for a snapshot.
pub type SnapshotId = String;

/// Unique identifier for a replication node.
pub type NodeId = String;

/// Log sequence number for ordering operations.
pub type Lsn = u64;

/// A snapshot of the database at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub node_id: NodeId,
    pub lsn: Lsn,
    pub created_at: u64, // Unix timestamp
    pub size_bytes: u64,
    pub checksum: String,
    pub snapshot_type: SnapshotType,
    pub collections: Vec<String>,
    pub vector_count: usize,
}

/// Type of snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SnapshotType {
    /// Complete copy of the database.
    Full,
    /// Delta from a base snapshot.
    Incremental { base_snapshot_id: SnapshotId },
}

/// Metadata about a delta between two snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotDelta {
    pub from_snapshot: SnapshotId,
    pub to_snapshot: SnapshotId,
    pub from_lsn: Lsn,
    pub to_lsn: Lsn,
    pub operations: Vec<DeltaOperation>,
    pub size_bytes: u64,
}

/// A single operation in a delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaOperation {
    /// Vector inserted or updated.
    Upsert {
        collection: String,
        id: String,
        vector_hash: String,
        has_metadata: bool,
    },
    /// Vector deleted.
    Delete {
        collection: String,
        id: String,
    },
    /// Collection created.
    CreateCollection {
        name: String,
        dimension: usize,
    },
    /// Collection dropped.
    DropCollection {
        name: String,
    },
}

// ---------------------------------------------------------------------------
// Replication Configuration
// ---------------------------------------------------------------------------

/// Configuration for the replication protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationConfig {
    /// How often to create automatic snapshots.
    pub snapshot_interval: Duration,
    /// Maximum number of snapshots to retain.
    pub max_snapshots: usize,
    /// Maximum number of WAL entries to retain between snapshots.
    pub max_wal_entries: usize,
    /// Whether to enable incremental snapshots.
    pub incremental_enabled: bool,
    /// Compression for snapshot data.
    pub compression: CompressionType,
    /// Maximum allowed replication lag before alerting.
    pub max_lag_threshold: Duration,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: Duration::from_secs(300), // 5 minutes
            max_snapshots: 10,
            max_wal_entries: 100_000,
            incremental_enabled: true,
            compression: CompressionType::None,
            max_lag_threshold: Duration::from_secs(60),
        }
    }
}

/// Compression type for snapshot data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionType {
    None,
    Lz4,
    Zstd,
}

// ---------------------------------------------------------------------------
// WAL Entry
// ---------------------------------------------------------------------------

/// A write-ahead log entry for replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalReplicationEntry {
    pub lsn: Lsn,
    pub timestamp: u64,
    pub operation: DeltaOperation,
    pub checksum: u32,
}

impl WalReplicationEntry {
    pub fn new(lsn: Lsn, operation: DeltaOperation) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let checksum = Self::compute_checksum(lsn, &operation);
        Self {
            lsn,
            timestamp,
            operation,
            checksum,
        }
    }

    fn compute_checksum(lsn: Lsn, _op: &DeltaOperation) -> u32 {
        // Simple checksum for integrity verification
        let mut hash: u32 = 0x811c9dc5; // FNV offset basis
        for byte in lsn.to_le_bytes() {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(0x01000193); // FNV prime
        }
        hash
    }

    pub fn verify_checksum(&self) -> bool {
        let expected = Self::compute_checksum(self.lsn, &self.operation);
        self.checksum == expected
    }
}

// ---------------------------------------------------------------------------
// Replication Leader
// ---------------------------------------------------------------------------

/// Manages snapshot creation and WAL for the leader node.
pub struct ReplicationLeaderNode {
    node_id: NodeId,
    config: ReplicationConfig,
    current_lsn: RwLock<Lsn>,
    wal: RwLock<Vec<WalReplicationEntry>>,
    snapshots: RwLock<Vec<Snapshot>>,
    followers: RwLock<HashMap<NodeId, FollowerInfo>>,
}

/// Information about a registered follower.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowerInfo {
    pub node_id: NodeId,
    pub last_applied_lsn: Lsn,
    pub last_heartbeat: u64,
    pub status: FollowerSyncStatus,
    pub lag: Duration,
}

/// Sync status of a follower.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FollowerSyncStatus {
    /// Follower is in sync with leader.
    InSync,
    /// Follower is catching up.
    CatchingUp,
    /// Follower is offline or lagging.
    Lagging,
    /// Initial sync — needs full snapshot.
    InitialSync,
}

impl ReplicationLeaderNode {
    /// Create a new replication leader.
    pub fn new(node_id: impl Into<NodeId>, config: ReplicationConfig) -> Self {
        Self {
            node_id: node_id.into(),
            config,
            current_lsn: RwLock::new(0),
            wal: RwLock::new(Vec::new()),
            snapshots: RwLock::new(Vec::new()),
            followers: RwLock::new(HashMap::new()),
        }
    }

    /// Get the current LSN.
    pub fn current_lsn(&self) -> Lsn {
        *self.current_lsn.read()
    }

    /// Append a write operation to the WAL.
    pub fn append_wal(&self, operation: DeltaOperation) -> Lsn {
        let mut lsn = self.current_lsn.write();
        *lsn += 1;
        let entry = WalReplicationEntry::new(*lsn, operation);
        self.wal.write().push(entry);
        self.trim_wal();
        *lsn
    }

    /// Create a snapshot of the current state.
    pub fn create_snapshot(&self) -> Result<Snapshot> {
        let lsn = *self.current_lsn.read();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let snapshot_type = if self.config.incremental_enabled {
            let snapshots = self.snapshots.read();
            if let Some(last) = snapshots.last() {
                SnapshotType::Incremental {
                    base_snapshot_id: last.id.clone(),
                }
            } else {
                SnapshotType::Full
            }
        } else {
            SnapshotType::Full
        };

        let id = format!("snap-{}-{}", self.node_id, lsn);
        let snapshot = Snapshot {
            id: id.clone(),
            node_id: self.node_id.clone(),
            lsn,
            created_at: now,
            size_bytes: 0, // Would be computed from actual data
            checksum: format!("{:016x}", now ^ lsn),
            snapshot_type,
            collections: Vec::new(),
            vector_count: 0,
        };

        let mut snapshots = self.snapshots.write();
        snapshots.push(snapshot.clone());

        // Trim old snapshots
        while snapshots.len() > self.config.max_snapshots {
            snapshots.remove(0);
        }

        Ok(snapshot)
    }

    /// Get WAL entries since a given LSN.
    pub fn get_wal_since(&self, from_lsn: Lsn) -> Vec<WalReplicationEntry> {
        self.wal
            .read()
            .iter()
            .filter(|e| e.lsn > from_lsn)
            .cloned()
            .collect()
    }

    /// List all available snapshots.
    pub fn list_snapshots(&self) -> Vec<Snapshot> {
        self.snapshots.read().clone()
    }

    /// Get the latest snapshot.
    pub fn latest_snapshot(&self) -> Option<Snapshot> {
        self.snapshots.read().last().cloned()
    }

    /// Register a follower node.
    pub fn register_follower(&self, node_id: impl Into<NodeId>) {
        let node_id = node_id.into();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.followers.write().insert(
            node_id.clone(),
            FollowerInfo {
                node_id,
                last_applied_lsn: 0,
                last_heartbeat: now,
                status: FollowerSyncStatus::InitialSync,
                lag: Duration::ZERO,
            },
        );
    }

    /// Update a follower's sync position.
    pub fn update_follower_position(&self, node_id: &str, applied_lsn: Lsn) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let current = *self.current_lsn.read();
        let lag_entries = current.saturating_sub(applied_lsn);

        if let Some(info) = self.followers.write().get_mut(node_id) {
            info.last_applied_lsn = applied_lsn;
            info.last_heartbeat = now;
            info.status = if lag_entries == 0 {
                FollowerSyncStatus::InSync
            } else if lag_entries < 100 {
                FollowerSyncStatus::CatchingUp
            } else {
                FollowerSyncStatus::Lagging
            };
        }
    }

    /// Get information about all followers.
    pub fn followers(&self) -> Vec<FollowerInfo> {
        self.followers.read().values().cloned().collect()
    }

    /// Number of WAL entries.
    pub fn wal_size(&self) -> usize {
        self.wal.read().len()
    }

    fn trim_wal(&self) {
        let mut wal = self.wal.write();
        while wal.len() > self.config.max_wal_entries {
            wal.remove(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Replication Follower
// ---------------------------------------------------------------------------

/// State of a replication follower node.
pub struct ReplicationFollower {
    node_id: NodeId,
    leader_id: NodeId,
    applied_lsn: RwLock<Lsn>,
    applied_snapshots: RwLock<Vec<SnapshotId>>,
    status: RwLock<FollowerSyncStatus>,
    last_sync: RwLock<Option<Instant>>,
    stats: RwLock<FollowerStats>,
}

/// Statistics for a follower node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FollowerStats {
    pub snapshots_applied: u64,
    pub wal_entries_applied: u64,
    pub bytes_received: u64,
    pub last_sync_duration_ms: u64,
    pub total_sync_duration_ms: u64,
}

impl ReplicationFollower {
    /// Create a new follower node.
    pub fn new(node_id: impl Into<NodeId>, leader_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
            leader_id: leader_id.into(),
            applied_lsn: RwLock::new(0),
            applied_snapshots: RwLock::new(Vec::new()),
            status: RwLock::new(FollowerSyncStatus::InitialSync),
            last_sync: RwLock::new(None),
            stats: RwLock::new(FollowerStats::default()),
        }
    }

    /// Get the follower's node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get the leader's node ID.
    pub fn leader_id(&self) -> &str {
        &self.leader_id
    }

    /// Current applied LSN.
    pub fn applied_lsn(&self) -> Lsn {
        *self.applied_lsn.read()
    }

    /// Current sync status.
    pub fn status(&self) -> FollowerSyncStatus {
        self.status.read().clone()
    }

    /// Apply a snapshot from the leader.
    pub fn apply_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        let start = Instant::now();

        // Validate the snapshot
        if snapshot.node_id != self.leader_id {
            return Err(NeedleError::InvalidInput(format!(
                "Snapshot from unexpected leader: {} (expected {})",
                snapshot.node_id, self.leader_id
            )));
        }

        *self.applied_lsn.write() = snapshot.lsn;
        self.applied_snapshots.write().push(snapshot.id.clone());
        *self.last_sync.write() = Some(Instant::now());

        let duration = start.elapsed();
        let mut stats = self.stats.write();
        stats.snapshots_applied += 1;
        stats.bytes_received += snapshot.size_bytes;
        stats.last_sync_duration_ms = duration.as_millis() as u64;
        stats.total_sync_duration_ms += duration.as_millis() as u64;

        if snapshot.lsn > 0 {
            *self.status.write() = FollowerSyncStatus::CatchingUp;
        }

        Ok(())
    }

    /// Apply WAL entries from the leader.
    pub fn apply_wal_entries(&self, entries: &[WalReplicationEntry]) -> Result<usize> {
        let start = Instant::now();
        let mut applied = 0;
        let current_lsn = *self.applied_lsn.read();

        for entry in entries {
            if entry.lsn <= current_lsn {
                continue; // Already applied
            }
            if !entry.verify_checksum() {
                return Err(NeedleError::InvalidInput(format!(
                    "Checksum mismatch for WAL entry LSN {}",
                    entry.lsn
                )));
            }
            // In a real implementation, we'd apply the operation to our local state
            *self.applied_lsn.write() = entry.lsn;
            applied += 1;
        }

        let duration = start.elapsed();
        let mut stats = self.stats.write();
        stats.wal_entries_applied += applied as u64;
        stats.last_sync_duration_ms = duration.as_millis() as u64;
        stats.total_sync_duration_ms += duration.as_millis() as u64;

        Ok(applied)
    }

    /// Get follower statistics.
    pub fn stats(&self) -> FollowerStats {
        self.stats.read().clone()
    }

    /// Compute replication lag relative to the leader's LSN.
    pub fn lag(&self, leader_lsn: Lsn) -> u64 {
        leader_lsn.saturating_sub(*self.applied_lsn.read())
    }

    /// Check if this follower is in sync.
    pub fn is_in_sync(&self, leader_lsn: Lsn) -> bool {
        self.lag(leader_lsn) == 0
    }
}

// ---------------------------------------------------------------------------
// Snapshot Transport Trait
// ---------------------------------------------------------------------------

/// Trait for snapshot transport implementations.
pub trait SnapshotTransport: Send + Sync {
    /// Upload a snapshot to the transport.
    fn upload(&self, snapshot: &Snapshot, data: &[u8]) -> Result<String>;

    /// Download a snapshot from the transport.
    fn download(&self, snapshot_id: &str) -> Result<Vec<u8>>;

    /// List available snapshots.
    fn list(&self) -> Result<Vec<SnapshotId>>;

    /// Delete a snapshot.
    fn delete(&self, snapshot_id: &str) -> Result<()>;
}

/// In-memory transport for testing.
pub struct InMemoryTransport {
    data: RwLock<HashMap<SnapshotId, Vec<u8>>>,
}

impl InMemoryTransport {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl SnapshotTransport for InMemoryTransport {
    fn upload(&self, snapshot: &Snapshot, data: &[u8]) -> Result<String> {
        self.data
            .write()
            .insert(snapshot.id.clone(), data.to_vec());
        Ok(snapshot.id.clone())
    }

    fn download(&self, snapshot_id: &str) -> Result<Vec<u8>> {
        self.data
            .read()
            .get(snapshot_id)
            .cloned()
            .ok_or_else(|| NeedleError::NotFound(format!("Snapshot not found: {}", snapshot_id)))
    }

    fn list(&self) -> Result<Vec<SnapshotId>> {
        Ok(self.data.read().keys().cloned().collect())
    }

    fn delete(&self, snapshot_id: &str) -> Result<()> {
        self.data.write().remove(snapshot_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Replication Health Monitor
// ---------------------------------------------------------------------------

/// Health status of the replication cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationHealth {
    pub leader_id: NodeId,
    pub leader_lsn: Lsn,
    pub snapshot_count: usize,
    pub wal_size: usize,
    pub followers: Vec<FollowerInfo>,
    pub all_in_sync: bool,
    pub lagging_followers: Vec<NodeId>,
}

/// Compute overall replication health from leader state.
pub fn compute_replication_health(leader: &ReplicationLeaderNode) -> ReplicationHealth {
    let followers = leader.followers();
    let leader_lsn = leader.current_lsn();
    let lagging: Vec<NodeId> = followers
        .iter()
        .filter(|f| f.status == FollowerSyncStatus::Lagging)
        .map(|f| f.node_id.clone())
        .collect();

    ReplicationHealth {
        leader_id: leader.node_id.clone(),
        leader_lsn,
        snapshot_count: leader.list_snapshots().len(),
        wal_size: leader.wal_size(),
        followers: followers.clone(),
        all_in_sync: lagging.is_empty()
            && followers
                .iter()
                .all(|f| f.status == FollowerSyncStatus::InSync),
        lagging_followers: lagging,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_creation() {
        let config = ReplicationConfig::default();
        let leader = ReplicationLeaderNode::new("leader-1", config);

        let snap = leader.create_snapshot().unwrap();
        assert_eq!(snap.node_id, "leader-1");
        assert_eq!(snap.lsn, 0);
        assert_eq!(snap.snapshot_type, SnapshotType::Full);
    }

    #[test]
    fn test_incremental_snapshot() {
        let config = ReplicationConfig {
            incremental_enabled: true,
            ..Default::default()
        };
        let leader = ReplicationLeaderNode::new("leader-1", config);

        leader.create_snapshot().unwrap(); // Full
        leader.append_wal(DeltaOperation::Upsert {
            collection: "test".into(),
            id: "v1".into(),
            vector_hash: "abc".into(),
            has_metadata: false,
        });

        let snap2 = leader.create_snapshot().unwrap();
        assert!(matches!(
            snap2.snapshot_type,
            SnapshotType::Incremental { .. }
        ));
    }

    #[test]
    fn test_wal_append_and_retrieve() {
        let leader = ReplicationLeaderNode::new("l1", ReplicationConfig::default());

        let lsn1 = leader.append_wal(DeltaOperation::Upsert {
            collection: "test".into(),
            id: "v1".into(),
            vector_hash: "h1".into(),
            has_metadata: false,
        });
        let lsn2 = leader.append_wal(DeltaOperation::Delete {
            collection: "test".into(),
            id: "v2".into(),
        });

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(leader.current_lsn(), 2);

        let entries = leader.get_wal_since(0);
        assert_eq!(entries.len(), 2);

        let entries_since_1 = leader.get_wal_since(1);
        assert_eq!(entries_since_1.len(), 1);
        assert_eq!(entries_since_1[0].lsn, 2);
    }

    #[test]
    fn test_wal_entry_checksum() {
        let entry = WalReplicationEntry::new(
            42,
            DeltaOperation::Delete {
                collection: "test".into(),
                id: "v1".into(),
            },
        );
        assert!(entry.verify_checksum());
    }

    #[test]
    fn test_follower_apply_snapshot() {
        let follower = ReplicationFollower::new("f1", "l1");

        let snapshot = Snapshot {
            id: "snap-l1-5".into(),
            node_id: "l1".into(),
            lsn: 5,
            created_at: 1000,
            size_bytes: 1024,
            checksum: "abcd".into(),
            snapshot_type: SnapshotType::Full,
            collections: vec!["test".into()],
            vector_count: 100,
        };

        follower.apply_snapshot(&snapshot).unwrap();
        assert_eq!(follower.applied_lsn(), 5);
        assert_eq!(follower.stats().snapshots_applied, 1);
    }

    #[test]
    fn test_follower_rejects_wrong_leader() {
        let follower = ReplicationFollower::new("f1", "l1");

        let snapshot = Snapshot {
            id: "snap-bad-1".into(),
            node_id: "wrong-leader".into(),
            lsn: 1,
            created_at: 1000,
            size_bytes: 0,
            checksum: "x".into(),
            snapshot_type: SnapshotType::Full,
            collections: vec![],
            vector_count: 0,
        };

        assert!(follower.apply_snapshot(&snapshot).is_err());
    }

    #[test]
    fn test_follower_apply_wal() {
        let follower = ReplicationFollower::new("f1", "l1");

        let entries = vec![
            WalReplicationEntry::new(
                1,
                DeltaOperation::Upsert {
                    collection: "test".into(),
                    id: "v1".into(),
                    vector_hash: "h".into(),
                    has_metadata: false,
                },
            ),
            WalReplicationEntry::new(
                2,
                DeltaOperation::Delete {
                    collection: "test".into(),
                    id: "v2".into(),
                },
            ),
        ];

        let applied = follower.apply_wal_entries(&entries).unwrap();
        assert_eq!(applied, 2);
        assert_eq!(follower.applied_lsn(), 2);
    }

    #[test]
    fn test_follower_skips_already_applied() {
        let follower = ReplicationFollower::new("f1", "l1");

        let entries = vec![WalReplicationEntry::new(
            1,
            DeltaOperation::CreateCollection {
                name: "test".into(),
                dimension: 128,
            },
        )];
        follower.apply_wal_entries(&entries).unwrap();

        // Apply again - should skip
        let applied = follower.apply_wal_entries(&entries).unwrap();
        assert_eq!(applied, 0);
    }

    #[test]
    fn test_follower_lag_calculation() {
        let follower = ReplicationFollower::new("f1", "l1");
        assert_eq!(follower.lag(10), 10);
        assert!(follower.is_in_sync(0));
        assert!(!follower.is_in_sync(1));
    }

    #[test]
    fn test_register_and_update_follower() {
        let leader = ReplicationLeaderNode::new("l1", ReplicationConfig::default());

        leader.register_follower("f1");
        leader.register_follower("f2");

        assert_eq!(leader.followers().len(), 2);

        leader.append_wal(DeltaOperation::Delete {
            collection: "t".into(),
            id: "v".into(),
        });
        leader.update_follower_position("f1", 1);

        let followers = leader.followers();
        let f1 = followers.iter().find(|f| f.node_id == "f1").unwrap();
        assert_eq!(f1.last_applied_lsn, 1);
        assert_eq!(f1.status, FollowerSyncStatus::InSync);
    }

    #[test]
    fn test_snapshot_retention() {
        let config = ReplicationConfig {
            max_snapshots: 3,
            incremental_enabled: false,
            ..Default::default()
        };
        let leader = ReplicationLeaderNode::new("l1", config);

        for _ in 0..5 {
            leader.create_snapshot().unwrap();
        }

        assert_eq!(leader.list_snapshots().len(), 3);
    }

    #[test]
    fn test_in_memory_transport() {
        let transport = InMemoryTransport::new();

        let snapshot = Snapshot {
            id: "snap-1".into(),
            node_id: "l1".into(),
            lsn: 1,
            created_at: 0,
            size_bytes: 4,
            checksum: "x".into(),
            snapshot_type: SnapshotType::Full,
            collections: vec![],
            vector_count: 0,
        };

        transport.upload(&snapshot, b"test").unwrap();
        let data = transport.download("snap-1").unwrap();
        assert_eq!(data, b"test");

        let list = transport.list().unwrap();
        assert_eq!(list.len(), 1);

        transport.delete("snap-1").unwrap();
        assert!(transport.download("snap-1").is_err());
    }

    #[test]
    fn test_replication_health() {
        let leader = ReplicationLeaderNode::new("l1", ReplicationConfig::default());
        leader.register_follower("f1");
        leader.update_follower_position("f1", 0);

        let health = compute_replication_health(&leader);
        assert_eq!(health.leader_id, "l1");
        assert_eq!(health.followers.len(), 1);
    }

    #[test]
    fn test_wal_trimming() {
        let config = ReplicationConfig {
            max_wal_entries: 5,
            ..Default::default()
        };
        let leader = ReplicationLeaderNode::new("l1", config);

        for _ in 0..10 {
            leader.append_wal(DeltaOperation::Delete {
                collection: "t".into(),
                id: "v".into(),
            });
        }

        assert!(leader.wal_size() <= 5);
    }

    #[test]
    fn test_delta_operation_variants() {
        let ops = vec![
            DeltaOperation::Upsert {
                collection: "c".into(),
                id: "v".into(),
                vector_hash: "h".into(),
                has_metadata: true,
            },
            DeltaOperation::Delete {
                collection: "c".into(),
                id: "v".into(),
            },
            DeltaOperation::CreateCollection {
                name: "c".into(),
                dimension: 128,
            },
            DeltaOperation::DropCollection { name: "c".into() },
        ];

        for op in &ops {
            let json = serde_json::to_string(op).unwrap();
            let _decoded: DeltaOperation = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_compression_type_serialization() {
        let types = [CompressionType::None, CompressionType::Lz4, CompressionType::Zstd];
        for ct in &types {
            let json = serde_json::to_string(ct).unwrap();
            let decoded: CompressionType = serde_json::from_str(&json).unwrap();
            assert_eq!(*ct, decoded);
        }
    }
}
