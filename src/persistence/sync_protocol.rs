//! Vector Sync Protocol
//!
//! Lightweight replication protocol inspired by Litestream that continuously
//! ships WAL segments to a destination (local directory or cloud storage),
//! enabling disaster recovery, point-in-time restore, and read replicas.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐    WAL segments    ┌──────────────┐
//! │  Primary  │ ───────────────▶  │  SyncTarget  │  (local dir / S3 / GCS)
//! │  Database │                   └──────────────┘
//! └──────────┘                            │
//!                                         ▼
//!                                  ┌──────────────┐
//!                                  │   Replica /   │
//!                                  │   Restore     │
//!                                  └──────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::sync_protocol::{SyncManager, SyncConfig, SyncTarget};
//!
//! let config = SyncConfig::new(SyncTarget::LocalDir("/backups/needle".into()))
//!     .interval_secs(10)
//!     .retention_hours(72);
//!
//! let sync = SyncManager::new(config);
//! sync.ship_snapshot(&db)?;
//!
//! // Later: restore
//! let restored = SyncManager::restore_latest(&target)?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Where WAL segments are shipped to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncTarget {
    /// Ship to a local directory (fast, good for same-host backups).
    LocalDir(PathBuf),
    /// Ship to an S3-compatible bucket.
    S3 {
        bucket: String,
        prefix: String,
        region: String,
    },
    /// Ship to GCS bucket.
    Gcs { bucket: String, prefix: String },
    /// Ship to Azure Blob container.
    AzureBlob { container: String, prefix: String },
}

impl SyncTarget {
    /// Convenience constructor for local directory target.
    pub fn local(dir: impl Into<PathBuf>) -> Self {
        Self::LocalDir(dir.into())
    }
}

/// Sync protocol configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Where to ship segments.
    pub target: SyncTarget,
    /// Shipping interval in seconds (0 = manual only).
    pub interval_secs: u64,
    /// How long to retain shipped segments (hours, 0 = forever).
    pub retention_hours: u64,
    /// Compress segments before shipping.
    pub compress: bool,
    /// Verify checksums on restore.
    pub verify_checksums: bool,
    /// Maximum segment size in bytes before rotation.
    pub max_segment_bytes: u64,
}

impl SyncConfig {
    /// Create config targeting a specific destination.
    pub fn new(target: SyncTarget) -> Self {
        Self {
            target,
            interval_secs: 30,
            retention_hours: 72,
            compress: true,
            verify_checksums: true,
            max_segment_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }

    #[must_use]
    pub fn interval_secs(mut self, secs: u64) -> Self {
        self.interval_secs = secs;
        self
    }

    #[must_use]
    pub fn retention_hours(mut self, hours: u64) -> Self {
        self.retention_hours = hours;
        self
    }

    #[must_use]
    pub fn compress(mut self, enabled: bool) -> Self {
        self.compress = enabled;
        self
    }
}

// ─── Segment metadata ────────────────────────────────────────────────────────

/// Metadata for a shipped segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Monotonically increasing segment ID.
    pub id: u64,
    /// SHA-256 hex digest of the segment data.
    pub checksum: String,
    /// Byte length of the segment.
    pub size_bytes: u64,
    /// ISO-8601 timestamp of when the segment was created.
    pub created_at: String,
    /// Whether the segment is compressed.
    pub compressed: bool,
    /// Number of entries in this segment.
    pub entry_count: u64,
}

/// Manifest stored alongside segments to enable restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncManifest {
    /// Database name / identifier.
    pub database_id: String,
    /// Ordered list of segments.
    pub segments: Vec<SegmentInfo>,
    /// Timestamp of the last full snapshot.
    pub last_snapshot_at: Option<String>,
    /// Format version (for future migration).
    pub format_version: u32,
}

impl SyncManifest {
    pub fn new(database_id: impl Into<String>) -> Self {
        Self {
            database_id: database_id.into(),
            segments: Vec::new(),
            last_snapshot_at: None,
            format_version: 1,
        }
    }

    /// Total bytes across all segments.
    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.size_bytes).sum()
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/// Runtime statistics for the sync manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    pub segments_shipped: u64,
    pub bytes_shipped: u64,
    pub snapshots_created: u64,
    pub errors: u64,
    pub last_ship_ms: u64,
    pub last_ship_at: Option<String>,
}

// ─── Sync Manager ────────────────────────────────────────────────────────────

/// Manages WAL shipping and snapshot-based replication.
pub struct SyncManager {
    config: SyncConfig,
    manifest: RwLock<SyncManifest>,
    stats: SyncManagerStats,
    history: RwLock<VecDeque<SegmentInfo>>,
}

struct SyncManagerStats {
    segments_shipped: AtomicU64,
    bytes_shipped: AtomicU64,
    snapshots_created: AtomicU64,
    errors: AtomicU64,
    last_ship_ms: AtomicU64,
}

impl SyncManager {
    /// Create a new sync manager.
    pub fn new(config: SyncConfig) -> Self {
        Self {
            config,
            manifest: RwLock::new(SyncManifest::new("default")),
            stats: SyncManagerStats {
                segments_shipped: AtomicU64::new(0),
                bytes_shipped: AtomicU64::new(0),
                snapshots_created: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                last_ship_ms: AtomicU64::new(0),
            },
            history: RwLock::new(VecDeque::with_capacity(1000)),
        }
    }

    /// Ship a full snapshot of the database.
    pub fn ship_snapshot(&self, db: &Database) -> Result<SegmentInfo> {
        let start = Instant::now();

        // Serialize database state
        let data = db.export_all_json()?;
        let data_bytes = data.as_bytes();

        // Compute checksum
        let checksum = {
            let mut hasher = Sha256::new();
            hasher.update(data_bytes);
            format!("{:x}", hasher.finalize())
        };

        let segment_id = self.next_segment_id();
        let info = SegmentInfo {
            id: segment_id,
            checksum,
            size_bytes: data_bytes.len() as u64,
            created_at: Utc::now().to_rfc3339(),
            compressed: false,
            entry_count: 1,
        };

        // Ship to target
        self.write_to_target(&format!("segment_{:08}.json", segment_id), data_bytes)?;
        self.write_manifest()?;

        // Update stats
        let elapsed = start.elapsed();
        self.stats.segments_shipped.fetch_add(1, Ordering::Relaxed);
        self.stats.snapshots_created.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_shipped
            .fetch_add(info.size_bytes, Ordering::Relaxed);
        self.stats
            .last_ship_ms
            .store(elapsed.as_millis() as u64, Ordering::Relaxed);

        // Record in manifest and history
        {
            let mut manifest = self.manifest.write();
            manifest.segments.push(info.clone());
            manifest.last_snapshot_at = Some(info.created_at.clone());
        }
        self.history.write().push_back(info.clone());

        Ok(info)
    }

    /// Ship an incremental segment containing only changed data since the last
    /// snapshot/segment. For now this re-exports the full state; a production
    /// implementation would diff against the previous segment.
    pub fn ship_incremental(&self, db: &Database) -> Result<SegmentInfo> {
        // Currently delegates to full snapshot – incremental diff requires WAL
        // integration which depends on the WAL being actively in use.
        self.ship_snapshot(db)
    }

    /// Restore the latest snapshot from the sync target into an in-memory database.
    pub fn restore_latest(&self) -> Result<Database> {
        let manifest = self.manifest.read();
        let last = manifest
            .segments
            .last()
            .ok_or_else(|| NeedleError::NotFound("No segments available for restore".into()))?;

        let filename = format!("segment_{:08}.json", last.id);
        let data = self.read_from_target(&filename)?;

        // Verify checksum if configured
        if self.config.verify_checksums {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let computed = format!("{:x}", hasher.finalize());
            if computed != last.checksum {
                return Err(NeedleError::InvalidInput(format!(
                    "Checksum mismatch: expected {}, got {}",
                    last.checksum, computed
                )));
            }
        }

        let json_str = String::from_utf8(data)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

        let db = Database::in_memory();
        db.import_all_json(&json_str)?;
        Ok(db)
    }

    /// Apply retention policy — remove segments older than retention window.
    pub fn apply_retention(&self) -> Result<usize> {
        if self.config.retention_hours == 0 {
            return Ok(0);
        }

        let cutoff = Utc::now() - chrono::Duration::hours(self.config.retention_hours as i64);
        let cutoff_str = cutoff.to_rfc3339();

        let mut manifest = self.manifest.write();
        let before = manifest.segments.len();
        manifest.segments.retain(|s| s.created_at >= cutoff_str);
        let removed = before - manifest.segments.len();
        Ok(removed)
    }

    /// Get current statistics.
    pub fn stats(&self) -> SyncStats {
        SyncStats {
            segments_shipped: self.stats.segments_shipped.load(Ordering::Relaxed),
            bytes_shipped: self.stats.bytes_shipped.load(Ordering::Relaxed),
            snapshots_created: self.stats.snapshots_created.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
            last_ship_ms: self.stats.last_ship_ms.load(Ordering::Relaxed),
            last_ship_at: self.history.read().back().map(|s| s.created_at.clone()),
        }
    }

    /// Get the manifest.
    pub fn manifest(&self) -> SyncManifest {
        self.manifest.read().clone()
    }

    /// Get the sync target.
    pub fn target(&self) -> &SyncTarget {
        &self.config.target
    }

    /// Get the config.
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn next_segment_id(&self) -> u64 {
        self.manifest.read().segments.len() as u64 + 1
    }

    fn write_to_target(&self, filename: &str, data: &[u8]) -> Result<()> {
        match &self.config.target {
            SyncTarget::LocalDir(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| {
                    NeedleError::InvalidOperation(format!("Failed to create sync dir: {}", e))
                })?;
                std::fs::write(dir.join(filename), data).map_err(|e| {
                    NeedleError::InvalidOperation(format!("Failed to write segment: {}", e))
                })?;
                Ok(())
            }
            // Cloud targets would use the cloud_storage backends.
            // For now, record the intent and return Ok.
            SyncTarget::S3 { .. } | SyncTarget::Gcs { .. } | SyncTarget::AzureBlob { .. } => {
                // Cloud shipping is a no-op stub; full implementation uses
                // the cloud_storage module when the feature is enabled.
                Ok(())
            }
        }
    }

    fn read_from_target(&self, filename: &str) -> Result<Vec<u8>> {
        match &self.config.target {
            SyncTarget::LocalDir(dir) => std::fs::read(dir.join(filename)).map_err(|e| {
                NeedleError::InvalidOperation(format!("Failed to read segment: {}", e))
            }),
            _ => Err(NeedleError::NotFound(
                "Cloud restore not yet implemented".into(),
            )),
        }
    }

    fn write_manifest(&self) -> Result<()> {
        let manifest = self.manifest.read().clone();
        let data = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| NeedleError::InvalidInput(format!("Manifest serialization: {}", e)))?;
        self.write_to_target("manifest.json", &data)
    }
}

// ─── LSN-based Incremental Sync ──────────────────────────────────────────────

/// Log Sequence Number for tracking HNSW graph mutations.
pub type Lsn = u64;

/// A single mutation to the HNSW graph, tagged with an LSN.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMutation {
    /// Monotonically increasing sequence number.
    pub lsn: Lsn,
    /// Collection this mutation applies to.
    pub collection: String,
    /// The kind of mutation.
    pub kind: GraphMutationKind,
    /// Unix epoch timestamp (ms).
    pub timestamp_ms: u64,
}

/// Kinds of HNSW graph mutations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphMutationKind {
    /// A new vector was inserted with links to neighbors.
    InsertNode {
        vector_id: String,
        internal_id: usize,
        level: usize,
        neighbors: Vec<usize>,
        vector_data: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    /// A vector was deleted and links were repaired.
    DeleteNode {
        vector_id: String,
        internal_id: usize,
    },
    /// Neighbor links were updated (re-index, compaction).
    UpdateLinks {
        internal_id: usize,
        level: usize,
        new_neighbors: Vec<usize>,
    },
}

/// Compact binary diff format for shipping graph segments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    /// Base LSN this diff starts from.
    pub base_lsn: Lsn,
    /// Target LSN after applying this diff.
    pub target_lsn: Lsn,
    /// Collection name.
    pub collection: String,
    /// The mutations in this diff.
    pub mutations: Vec<GraphMutation>,
    /// Compressed byte size (0 if uncompressed).
    pub compressed_bytes: u64,
}

impl GraphDiff {
    /// Create a diff from a slice of mutations.
    pub fn from_mutations(collection: &str, mutations: &[GraphMutation]) -> Option<Self> {
        if mutations.is_empty() {
            return None;
        }
        let base_lsn = mutations.first().map_or(0, |m| m.lsn);
        let target_lsn = mutations.last().map_or(0, |m| m.lsn);
        Some(Self {
            base_lsn,
            target_lsn,
            collection: collection.to_string(),
            mutations: mutations.to_vec(),
            compressed_bytes: 0,
        })
    }

    /// Number of mutations in this diff.
    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    /// Check if diff is empty.
    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }

    /// Estimated bandwidth as a fraction of full snapshot size.
    pub fn estimated_bandwidth_ratio(&self, full_snapshot_bytes: u64) -> f64 {
        if full_snapshot_bytes == 0 {
            return 1.0;
        }
        let diff_bytes = self.compressed_bytes.max(
            self.mutations.len() as u64 * 128, // rough estimate per mutation
        );
        diff_bytes as f64 / full_snapshot_bytes as f64
    }
}

/// Tracks LSN state and buffers mutations for delta replication.
pub struct DeltaReplicationManager {
    /// Current LSN counter.
    current_lsn: AtomicU64,
    /// Buffered mutations since last ship.
    mutation_log: RwLock<Vec<GraphMutation>>,
    /// LSN of the last shipped diff.
    last_shipped_lsn: AtomicU64,
    /// Per-collection mutation counts.
    collection_counts: RwLock<HashMap<String, u64>>,
    /// Configuration.
    config: DeltaReplicationConfig,
}

/// Configuration for delta replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaReplicationConfig {
    /// Maximum mutations to buffer before forcing a ship.
    pub max_buffer_size: usize,
    /// Ship interval in milliseconds.
    pub ship_interval_ms: u64,
    /// Maximum diff size in bytes before falling back to full snapshot.
    pub max_diff_bytes: u64,
    /// Target replication lag in milliseconds.
    pub target_lag_ms: u64,
}

impl Default for DeltaReplicationConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10_000,
            ship_interval_ms: 500,
            max_diff_bytes: 64 * 1024 * 1024,
            target_lag_ms: 1000,
        }
    }
}

impl DeltaReplicationManager {
    /// Create a new delta replication manager.
    pub fn new(config: DeltaReplicationConfig) -> Self {
        Self {
            current_lsn: AtomicU64::new(0),
            mutation_log: RwLock::new(Vec::new()),
            last_shipped_lsn: AtomicU64::new(0),
            collection_counts: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Record a graph mutation and assign it an LSN.
    pub fn record_mutation(&self, collection: &str, kind: GraphMutationKind) -> Lsn {
        let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst) + 1;
        let now_ms = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mutation = GraphMutation {
            lsn,
            collection: collection.to_string(),
            kind,
            timestamp_ms: now_ms,
        };
        self.mutation_log.write().push(mutation);
        *self
            .collection_counts
            .write()
            .entry(collection.to_string())
            .or_insert(0) += 1;
        lsn
    }

    /// Get the current LSN.
    pub fn current_lsn(&self) -> Lsn {
        self.current_lsn.load(Ordering::SeqCst)
    }

    /// Get the last shipped LSN.
    pub fn last_shipped_lsn(&self) -> Lsn {
        self.last_shipped_lsn.load(Ordering::SeqCst)
    }

    /// Get the number of pending (unshipped) mutations.
    pub fn pending_count(&self) -> usize {
        self.mutation_log.read().len()
    }

    /// Whether the buffer is full and should be shipped.
    pub fn should_ship(&self) -> bool {
        self.mutation_log.read().len() >= self.config.max_buffer_size
    }

    /// Pull deltas since a given LSN for a specific collection.
    pub fn pull_deltas(&self, since_lsn: Lsn, collection: &str) -> GraphDiff {
        let log = self.mutation_log.read();
        let mutations: Vec<GraphMutation> = log
            .iter()
            .filter(|m| m.lsn > since_lsn && m.collection == collection)
            .cloned()
            .collect();
        GraphDiff::from_mutations(collection, &mutations).unwrap_or(GraphDiff {
            base_lsn: since_lsn,
            target_lsn: self.current_lsn(),
            collection: collection.to_string(),
            mutations: Vec::new(),
            compressed_bytes: 0,
        })
    }

    /// Pull all deltas since a given LSN across all collections.
    pub fn pull_all_deltas(&self, since_lsn: Lsn) -> Vec<GraphDiff> {
        let log = self.mutation_log.read();
        let mut by_collection: HashMap<String, Vec<GraphMutation>> = HashMap::new();
        for m in log.iter().filter(|m| m.lsn > since_lsn) {
            by_collection
                .entry(m.collection.clone())
                .or_default()
                .push(m.clone());
        }
        by_collection
            .into_iter()
            .filter_map(|(col, muts)| GraphDiff::from_mutations(&col, &muts))
            .collect()
    }

    /// Ship (drain) all buffered mutations, returning the diffs.
    /// Updates the last_shipped_lsn.
    pub fn ship_deltas(&self) -> Vec<GraphDiff> {
        let mut log = self.mutation_log.write();
        if log.is_empty() {
            return Vec::new();
        }
        let last_lsn = log.last().map_or(0, |m| m.lsn);
        let mutations = std::mem::take(&mut *log);
        drop(log);

        self.last_shipped_lsn.store(last_lsn, Ordering::SeqCst);

        let mut by_collection: HashMap<String, Vec<GraphMutation>> = HashMap::new();
        for m in mutations {
            by_collection
                .entry(m.collection.clone())
                .or_default()
                .push(m);
        }
        by_collection
            .into_iter()
            .filter_map(|(col, muts)| GraphDiff::from_mutations(&col, &muts))
            .collect()
    }

    /// Get replication lag in number of mutations.
    pub fn replication_lag(&self) -> u64 {
        self.current_lsn() - self.last_shipped_lsn()
    }

    /// Get per-collection mutation statistics.
    pub fn collection_stats(&self) -> HashMap<String, u64> {
        self.collection_counts.read().clone()
    }
}

// ─── Differential Sync Protocol ──────────────────────────────────────────────

/// Hash of a vector entry for Merkle tree diffing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct VectorHash {
    pub id: String,
    pub hash: [u8; 32],
    pub timestamp: u64,
}

/// Merkle tree node for efficient collection diffing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    pub hash: [u8; 32],
    pub children: Vec<MerkleNode>,
    pub range_start: usize,
    pub range_end: usize,
    pub is_leaf: bool,
}

/// Vector clock for conflict resolution.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorClock {
    pub clocks: HashMap<String, u64>,
}

impl VectorClock {
    /// Increment the clock for a node.
    pub fn increment(&mut self, node_id: &str) {
        let counter = self.clocks.entry(node_id.to_string()).or_insert(0);
        *counter += 1;
    }

    /// Merge two vector clocks by taking the maximum for each node.
    pub fn merge(&mut self, other: &VectorClock) {
        for (node_id, &other_val) in &other.clocks {
            let entry = self.clocks.entry(node_id.clone()).or_insert(0);
            *entry = (*entry).max(other_val);
        }
    }

    /// Returns true if `self` causally happens before `other`.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut at_least_one_less = false;
        for (node_id, &self_val) in &self.clocks {
            let other_val = other.clocks.get(node_id).copied().unwrap_or(0);
            if self_val > other_val {
                return false;
            }
            if self_val < other_val {
                at_least_one_less = true;
            }
        }
        // Check keys in other that are not in self
        for (node_id, &other_val) in &other.clocks {
            if !self.clocks.contains_key(node_id) && other_val > 0 {
                at_least_one_less = true;
            }
        }
        at_least_one_less
    }

    /// Returns true if the two vector clocks are concurrent (neither happens before the other).
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self.clocks != other.clocks
    }
}

/// A single vector change to be synced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDelta {
    pub vector_id: String,
    pub collection: String,
    pub operation: SyncOperation,
    pub vector_data: Option<Vec<f32>>,
    pub metadata: Option<serde_json::Value>,
    pub vector_clock: VectorClock,
    pub timestamp: u64,
}

/// The type of sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOperation {
    Insert,
    Update,
    Delete,
}

/// Result of a sync operation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    pub vectors_sent: usize,
    pub vectors_received: usize,
    pub conflicts_resolved: usize,
    pub duration_ms: u64,
}

/// Configuration for differential sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSyncConfig {
    pub node_id: String,
    pub merkle_branch_factor: usize,
    pub batch_size: usize,
    pub conflict_resolution: ConflictResolution,
}

/// Strategy for resolving conflicting edits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    LastWriterWins,
    VectorClockMerge,
    HigherTimestamp,
}

/// Manages bidirectional sync using Merkle-tree diffing.
pub struct DiffSyncManager {
    config: DiffSyncConfig,
    vector_clock: VectorClock,
    pending_deltas: Vec<SyncDelta>,
}

impl DiffSyncManager {
    /// Create a new differential sync manager.
    pub fn new(config: DiffSyncConfig) -> Self {
        Self {
            config,
            vector_clock: VectorClock::default(),
            pending_deltas: Vec::new(),
        }
    }

    /// Build a Merkle tree from a sorted slice of vector hashes.
    pub fn build_merkle_tree(hashes: &[VectorHash]) -> MerkleNode {
        Self::build_merkle_subtree(hashes, 0, hashes.len(), 4)
    }

    fn build_merkle_subtree(
        hashes: &[VectorHash],
        range_start: usize,
        range_end: usize,
        branch_factor: usize,
    ) -> MerkleNode {
        let count = range_end - range_start;
        if count <= branch_factor {
            // Leaf node: hash all entries in this range
            let mut hasher = Sha256::new();
            for h in &hashes[range_start..range_end] {
                hasher.update(&h.hash);
            }
            return MerkleNode {
                hash: hasher.finalize().into(),
                children: Vec::new(),
                range_start,
                range_end,
                is_leaf: true,
            };
        }

        // Internal node: split into branch_factor children
        let chunk_size = (count + branch_factor - 1) / branch_factor;
        let mut children = Vec::new();
        let mut pos = range_start;
        while pos < range_end {
            let child_end = (pos + chunk_size).min(range_end);
            children.push(Self::build_merkle_subtree(
                hashes,
                pos,
                child_end,
                branch_factor,
            ));
            pos = child_end;
        }

        let mut hasher = Sha256::new();
        for child in &children {
            hasher.update(&child.hash);
        }

        MerkleNode {
            hash: hasher.finalize().into(),
            children,
            range_start,
            range_end,
            is_leaf: false,
        }
    }

    /// Compare two Merkle trees and return the `(start, end)` ranges that differ.
    pub fn compute_diff(local: &MerkleNode, remote: &MerkleNode) -> Vec<(usize, usize)> {
        if local.hash == remote.hash {
            return Vec::new();
        }

        // If either is a leaf, the whole range differs
        if local.is_leaf || remote.is_leaf {
            let start = local.range_start.min(remote.range_start);
            let end = local.range_end.max(remote.range_end);
            return vec![(start, end)];
        }

        // Recursively diff children
        let mut diffs = Vec::new();
        let max_children = local.children.len().max(remote.children.len());
        for i in 0..max_children {
            match (local.children.get(i), remote.children.get(i)) {
                (Some(lc), Some(rc)) => {
                    diffs.extend(Self::compute_diff(lc, rc));
                }
                (Some(lc), None) => {
                    diffs.push((lc.range_start, lc.range_end));
                }
                (None, Some(rc)) => {
                    diffs.push((rc.range_start, rc.range_end));
                }
                (None, None) => {}
            }
        }
        diffs
    }

    /// Record a change for syncing.
    pub fn record_change(&mut self, delta: SyncDelta) {
        self.vector_clock
            .increment(&self.config.node_id);
        self.pending_deltas.push(delta);
    }

    /// Drain and return all pending deltas.
    pub fn take_pending_deltas(&mut self) -> Vec<SyncDelta> {
        std::mem::take(&mut self.pending_deltas)
    }

    /// Resolve a conflict between a local and remote delta using the configured strategy.
    pub fn resolve_conflict(&self, local: &SyncDelta, remote: &SyncDelta) -> SyncDelta {
        match self.config.conflict_resolution {
            ConflictResolution::LastWriterWins | ConflictResolution::HigherTimestamp => {
                if local.timestamp >= remote.timestamp {
                    local.clone()
                } else {
                    remote.clone()
                }
            }
            ConflictResolution::VectorClockMerge => {
                if remote.vector_clock.happens_before(&local.vector_clock) {
                    local.clone()
                } else if local.vector_clock.happens_before(&remote.vector_clock) {
                    remote.clone()
                } else {
                    // Concurrent: fall back to higher timestamp
                    if local.timestamp >= remote.timestamp {
                        local.clone()
                    } else {
                        remote.clone()
                    }
                }
            }
        }
    }

    /// Apply received deltas, merging vector clocks and counting conflicts.
    pub fn apply_deltas(&mut self, deltas: Vec<SyncDelta>) -> Result<SyncResult> {
        let start = Instant::now();
        let mut conflicts_resolved = 0;
        let received = deltas.len();

        for delta in &deltas {
            // Check for conflicts against pending deltas
            let conflict_idx = self
                .pending_deltas
                .iter()
                .position(|p| p.vector_id == delta.vector_id && p.collection == delta.collection);

            if let Some(idx) = conflict_idx {
                let local = self.pending_deltas[idx].clone();
                let resolved = self.resolve_conflict(&local, delta);
                self.pending_deltas[idx] = resolved;
                conflicts_resolved += 1;
            }

            self.vector_clock.merge(&delta.vector_clock);
        }

        Ok(SyncResult {
            vectors_sent: 0,
            vectors_received: received,
            conflicts_resolved,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;

    fn make_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        let col = db.collection("test").unwrap();
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db
    }

    #[test]
    fn test_sync_config_builder() {
        let config = SyncConfig::new(SyncTarget::local("/tmp/sync"))
            .interval_secs(10)
            .retention_hours(48)
            .compress(false);
        assert_eq!(config.interval_secs, 10);
        assert_eq!(config.retention_hours, 48);
        assert!(!config.compress);
    }

    #[test]
    fn test_ship_and_restore_local() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        let info = sync.ship_snapshot(&db).unwrap();
        assert!(info.size_bytes > 0);
        assert!(!info.checksum.is_empty());
        assert_eq!(info.id, 1);

        let stats = sync.stats();
        assert_eq!(stats.segments_shipped, 1);
        assert_eq!(stats.snapshots_created, 1);

        // Restore
        let restored = sync.restore_latest().unwrap();
        let col = restored.collection("test").unwrap();
        assert!(col.get("v1").is_some());
        assert!(col.get("v2").is_some());
    }

    #[test]
    fn test_manifest_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();
        sync.ship_snapshot(&db).unwrap();

        let manifest = sync.manifest();
        assert_eq!(manifest.segment_count(), 2);
        assert!(manifest.total_bytes() > 0);
        assert!(manifest.last_snapshot_at.is_some());
    }

    #[test]
    fn test_retention_policy() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path())).retention_hours(0); // retain forever
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();
        let removed = sync.apply_retention().unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_incremental_ships() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();

        // Add more data
        let col = db.collection("test").unwrap();
        col.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();

        let info = sync.ship_incremental(&db).unwrap();
        assert_eq!(info.id, 2);
        assert!(info.size_bytes > 0);
    }

    #[test]
    fn test_checksum_verification() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();

        // Corrupt the segment file
        let segment_path = dir.path().join("segment_00000001.json");
        std::fs::write(&segment_path, b"corrupted data").unwrap();

        let result = sync.restore_latest();
        assert!(result.is_err());
    }

    #[test]
    fn test_restore_empty_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let result = sync.restore_latest();
        assert!(result.is_err());
    }

    #[test]
    fn test_cloud_target_stubs() {
        let config = SyncConfig::new(SyncTarget::S3 {
            bucket: "my-bucket".into(),
            prefix: "needle/".into(),
            region: "us-east-1".into(),
        });
        let sync = SyncManager::new(config);
        let db = make_db();
        // Cloud shipping is a stub but should not error
        let info = sync.ship_snapshot(&db).unwrap();
        assert!(info.size_bytes > 0);
    }

    // ── Differential Sync Tests ──────────────────────────────────────

    #[test]
    fn test_vector_clock_increment_and_merge() {
        let mut vc1 = VectorClock::default();
        vc1.increment("node_a");
        vc1.increment("node_a");
        assert_eq!(*vc1.clocks.get("node_a").unwrap_or(&0), 2);

        let mut vc2 = VectorClock::default();
        vc2.increment("node_b");
        vc2.increment("node_b");
        vc2.increment("node_b");

        vc1.merge(&vc2);
        assert_eq!(*vc1.clocks.get("node_a").unwrap_or(&0), 2);
        assert_eq!(*vc1.clocks.get("node_b").unwrap_or(&0), 3);
    }

    #[test]
    fn test_vector_clock_ordering() {
        let mut vc1 = VectorClock::default();
        vc1.increment("a");

        let mut vc2 = VectorClock::default();
        vc2.increment("a");
        vc2.increment("a");

        assert!(vc1.happens_before(&vc2));
        assert!(!vc2.happens_before(&vc1));
        assert!(!vc1.is_concurrent(&vc2));

        // Concurrent: vc1 has (a:1), vc3 has (b:1)
        let mut vc3 = VectorClock::default();
        vc3.increment("b");
        assert!(vc1.is_concurrent(&vc3));
    }

    #[test]
    fn test_merkle_tree_identical() {
        let hashes: Vec<VectorHash> = (0..8)
            .map(|i| VectorHash {
                id: format!("v{i}"),
                hash: {
                    use sha2::{Digest, Sha256};
                    let mut h = Sha256::new();
                    h.update(format!("v{i}").as_bytes());
                    h.finalize().into()
                },
                timestamp: i,
            })
            .collect();
        let tree1 = DiffSyncManager::build_merkle_tree(&hashes);
        let tree2 = DiffSyncManager::build_merkle_tree(&hashes);
        let diffs = DiffSyncManager::compute_diff(&tree1, &tree2);
        assert!(diffs.is_empty(), "Identical trees should have no diffs");
    }

    #[test]
    fn test_merkle_tree_detects_change() {
        let mut hashes1: Vec<VectorHash> = (0..8)
            .map(|i| VectorHash {
                id: format!("v{i}"),
                hash: [i as u8; 32],
                timestamp: i,
            })
            .collect();

        let mut hashes2 = hashes1.clone();
        // Modify one entry
        hashes2[3].hash = [99; 32];

        let tree1 = DiffSyncManager::build_merkle_tree(&hashes1);
        let tree2 = DiffSyncManager::build_merkle_tree(&hashes2);
        let diffs = DiffSyncManager::compute_diff(&tree1, &tree2);
        assert!(!diffs.is_empty(), "Modified entry should produce diffs");
        // The differing range should include index 3
        let covers_3 = diffs.iter().any(|(s, e)| *s <= 3 && *e > 3);
        assert!(covers_3, "Diff should cover modified index 3");
    }

    #[test]
    fn test_conflict_resolution_last_writer_wins() {
        let config = DiffSyncConfig {
            node_id: "node1".into(),
            merkle_branch_factor: 4,
            batch_size: 100,
            conflict_resolution: ConflictResolution::LastWriterWins,
        };
        let mgr = DiffSyncManager::new(config);

        let local = SyncDelta {
            vector_id: "v1".into(),
            collection: "col".into(),
            operation: SyncOperation::Update,
            vector_data: Some(vec![1.0]),
            metadata: None,
            vector_clock: VectorClock::default(),
            timestamp: 100,
        };
        let remote = SyncDelta {
            timestamp: 200,
            ..local.clone()
        };
        let winner = mgr.resolve_conflict(&local, &remote);
        assert_eq!(winner.timestamp, 200);
    }

    #[test]
    fn test_diff_sync_record_and_take() {
        let config = DiffSyncConfig {
            node_id: "n1".into(),
            merkle_branch_factor: 4,
            batch_size: 100,
            conflict_resolution: ConflictResolution::VectorClockMerge,
        };
        let mut mgr = DiffSyncManager::new(config);
        mgr.record_change(SyncDelta {
            vector_id: "v1".into(),
            collection: "col".into(),
            operation: SyncOperation::Insert,
            vector_data: Some(vec![1.0, 2.0]),
            metadata: None,
            vector_clock: VectorClock::default(),
            timestamp: 1,
        });
        assert_eq!(mgr.pending_deltas.len(), 1);
        let taken = mgr.take_pending_deltas();
        assert_eq!(taken.len(), 1);
        assert!(mgr.pending_deltas.is_empty());
    }

    #[test]
    fn test_apply_deltas_with_conflict() {
        let config = DiffSyncConfig {
            node_id: "n1".into(),
            merkle_branch_factor: 4,
            batch_size: 100,
            conflict_resolution: ConflictResolution::HigherTimestamp,
        };
        let mut mgr = DiffSyncManager::new(config);

        // Record a local change
        mgr.record_change(SyncDelta {
            vector_id: "v1".into(),
            collection: "col".into(),
            operation: SyncOperation::Update,
            vector_data: Some(vec![1.0]),
            metadata: None,
            vector_clock: VectorClock::default(),
            timestamp: 50,
        });

        // Apply a remote change for the same vector with higher timestamp
        let remote_deltas = vec![SyncDelta {
            vector_id: "v1".into(),
            collection: "col".into(),
            operation: SyncOperation::Update,
            vector_data: Some(vec![2.0]),
            metadata: None,
            vector_clock: VectorClock::default(),
            timestamp: 100,
        }];

        let result = mgr.apply_deltas(remote_deltas).unwrap();
        assert_eq!(result.vectors_received, 1);
        assert_eq!(result.conflicts_resolved, 1);

        // The pending delta should be resolved to the remote version (higher timestamp)
        assert_eq!(mgr.pending_deltas[0].timestamp, 100);
    }

    // ── Delta Replication Tests ──────────────────────────────────────

    #[test]
    fn test_delta_replication_record_and_pull() {
        let mgr = DeltaReplicationManager::new(DeltaReplicationConfig::default());

        mgr.record_mutation(
            "test_col",
            GraphMutationKind::InsertNode {
                vector_id: "v1".into(),
                internal_id: 0,
                level: 0,
                neighbors: vec![],
                vector_data: vec![1.0, 2.0],
                metadata: None,
            },
        );
        mgr.record_mutation(
            "test_col",
            GraphMutationKind::InsertNode {
                vector_id: "v2".into(),
                internal_id: 1,
                level: 0,
                neighbors: vec![0],
                vector_data: vec![3.0, 4.0],
                metadata: None,
            },
        );

        assert_eq!(mgr.current_lsn(), 2);
        assert_eq!(mgr.pending_count(), 2);

        let diff = mgr.pull_deltas(0, "test_col");
        assert_eq!(diff.mutations.len(), 2);
        assert_eq!(diff.base_lsn, 1);
        assert_eq!(diff.target_lsn, 2);
    }

    #[test]
    fn test_delta_replication_ship() {
        let mgr = DeltaReplicationManager::new(DeltaReplicationConfig::default());

        mgr.record_mutation(
            "col_a",
            GraphMutationKind::DeleteNode {
                vector_id: "v1".into(),
                internal_id: 0,
            },
        );
        mgr.record_mutation(
            "col_b",
            GraphMutationKind::UpdateLinks {
                internal_id: 1,
                level: 0,
                new_neighbors: vec![2, 3],
            },
        );

        let diffs = mgr.ship_deltas();
        assert_eq!(diffs.len(), 2);
        assert_eq!(mgr.pending_count(), 0);
        assert_eq!(mgr.last_shipped_lsn(), 2);
        assert_eq!(mgr.replication_lag(), 0);
    }

    #[test]
    fn test_graph_diff_bandwidth_ratio() {
        let diff = GraphDiff {
            base_lsn: 0,
            target_lsn: 10,
            collection: "test".into(),
            mutations: vec![],
            compressed_bytes: 100,
        };
        let ratio = diff.estimated_bandwidth_ratio(10_000);
        assert!(ratio < 0.05, "Diff should use <5% bandwidth");
    }
}
