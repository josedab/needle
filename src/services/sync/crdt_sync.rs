//! Real-Time CRDT Vector Sync Service
//!
//! Bi-directional, conflict-free vector synchronization across devices/instances
//! using CRDTs. Enables offline-first mobile/edge apps that sync when connected.
//!
//! When the `experimental` feature is enabled, the service delegates core
//! operations to [`crate::experimental::crdt::VectorCRDT`] for a unified CRDT
//! backend while preserving the service-level API (peers, stats, conflicts).
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::crdt_sync::{
//!     SyncService, SyncConfig, SyncPeer, SyncDelta, VectorOp,
//! };
//!
//! let mut svc = SyncService::new("node-1".into(), SyncConfig::default());
//!
//! // Record a local insert
//! svc.record_insert("doc-1", &[0.1, 0.2, 0.3], None);
//!
//! // Generate delta for syncing
//! let delta = svc.generate_delta(0);
//! assert!(!delta.operations.is_empty());
//! ```

use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

#[allow(unused_imports)]
use tracing::warn;

// ── Core CRDT scaffold imports (feature-gated) ──────────────────────────────

#[cfg(feature = "experimental")]
#[allow(unused_imports)]
use crate::experimental::crdt::{
    Delta as CoreDelta, HLC as CoreHLC, MergeResult as CoreMergeResult,
    Operation as CoreOp, PeerRegistry, ReplicaId, SyncConfig as CoreSyncConfig,
    SyncState as CoreSyncState, VectorCRDT,
};

// ── Hybrid Logical Clock ────────────────────────────────────────────────────

/// Hybrid Logical Clock for causal ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HLC {
    /// Physical timestamp (wall clock ms).
    pub physical: u64,
    /// Logical counter for disambiguation.
    pub logical: u32,
    /// Node ID for tiebreaking.
    pub node_id_hash: u32,
}

impl HLC {
    pub fn new(node_id: &str) -> Self {
        Self {
            physical: now_ms(),
            logical: 0,
            node_id_hash: hash_node_id(node_id),
        }
    }

    /// Tick the clock for a local event.
    pub fn tick(&mut self) {
        let now = now_ms();
        if now > self.physical {
            self.physical = now;
            self.logical = 0;
        } else {
            self.logical += 1;
        }
    }

    /// Merge with a remote clock (on receive).
    pub fn merge(&mut self, remote: &HLC) {
        let now = now_ms();
        if now > self.physical && now > remote.physical {
            self.physical = now;
            self.logical = 0;
        } else if self.physical > remote.physical {
            self.logical += 1;
        } else if remote.physical > self.physical {
            self.physical = remote.physical;
            self.logical = remote.logical + 1;
        } else {
            self.logical = self.logical.max(remote.logical) + 1;
        }
    }

    /// Convert to the core scaffold [`CoreHLC`].
    #[cfg(feature = "experimental")]
    pub fn to_core_hlc(&self) -> CoreHLC {
        CoreHLC {
            physical: self.physical,
            logical: self.logical,
            replica: u64::from(self.node_id_hash),
        }
    }

    /// Create from a core scaffold [`CoreHLC`].
    #[cfg(feature = "experimental")]
    pub fn from_core_hlc(core: &CoreHLC) -> Self {
        Self {
            physical: core.physical,
            logical: core.logical,
            node_id_hash: core.replica as u32,
        }
    }
}

// ── Vector Operations (CRDT) ────────────────────────────────────────────────

/// A CRDT operation on a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorOp {
    /// Operation type.
    pub op_type: OpType,
    /// Vector ID.
    pub vector_id: String,
    /// The vector data (for insert/update).
    pub vector: Option<Vec<f32>>,
    /// Metadata (for insert/update).
    pub metadata: Option<serde_json::Value>,
    /// Timestamp of the operation.
    pub timestamp: HLC,
    /// Originating node.
    pub origin_node: String,
}

impl VectorOp {
    /// Convert to a core scaffold [`CoreOp`].
    #[cfg(feature = "experimental")]
    pub fn to_core_op(&self) -> CoreOp {
        match self.op_type {
            OpType::Insert => CoreOp::Add {
                id: self.vector_id.clone(),
                vector: self.vector.clone().unwrap_or_default(),
                metadata: json_to_string_map(self.metadata.as_ref()),
            },
            OpType::Update => CoreOp::Update {
                id: self.vector_id.clone(),
                vector: self.vector.clone().unwrap_or_default(),
            },
            OpType::Delete => CoreOp::Delete {
                id: self.vector_id.clone(),
            },
            OpType::MetadataUpdate => CoreOp::UpdateMetadata {
                id: self.vector_id.clone(),
                key: "metadata".to_string(),
                value: self.metadata.as_ref().map(|m| m.to_string()),
            },
        }
    }

    /// Create from a core scaffold [`CoreOp`] plus context.
    #[cfg(feature = "experimental")]
    pub fn from_core_op(
        core: &CoreOp,
        timestamp: HLC,
        origin_node: String,
    ) -> Self {
        match core {
            CoreOp::Add {
                id,
                vector,
                metadata,
            } => Self {
                op_type: OpType::Insert,
                vector_id: id.clone(),
                vector: Some(vector.clone()),
                metadata: Some(string_map_to_json(metadata)),
                timestamp,
                origin_node,
            },
            CoreOp::Update { id, vector } => Self {
                op_type: OpType::Update,
                vector_id: id.clone(),
                vector: Some(vector.clone()),
                metadata: None,
                timestamp,
                origin_node,
            },
            CoreOp::Delete { id } => Self {
                op_type: OpType::Delete,
                vector_id: id.clone(),
                vector: None,
                metadata: None,
                timestamp,
                origin_node,
            },
            CoreOp::UpdateMetadata { id, key, value } => {
                let meta = value.as_ref().map(|v| {
                    serde_json::json!({ key.clone(): v.clone() })
                });
                Self {
                    op_type: OpType::MetadataUpdate,
                    vector_id: id.clone(),
                    vector: None,
                    metadata: meta,
                    timestamp,
                    origin_node,
                }
            }
        }
    }
}

/// Operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpType {
    Insert,
    Update,
    Delete,
    MetadataUpdate,
}

/// State of a vector in the CRDT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorState {
    /// Latest vector data.
    pub vector: Vec<f32>,
    /// Latest metadata.
    pub metadata: Option<serde_json::Value>,
    /// Last-write timestamp (LWW register).
    pub last_write: HLC,
    /// Whether this vector has been tombstoned.
    pub deleted: bool,
    /// Tombstone timestamp (if deleted).
    pub deleted_at: Option<HLC>,
    /// All nodes that have seen this state.
    pub seen_by: HashSet<String>,
}

// ── Sync Delta ──────────────────────────────────────────────────────────────

/// A delta (batch of operations) to send to peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDelta {
    /// Operations in this delta.
    pub operations: Vec<VectorOp>,
    /// Source node.
    pub source_node: String,
    /// Delta sequence number.
    pub sequence: u64,
    /// Clock at time of delta generation.
    pub clock: HLC,
    /// Whether this is a compressed delta.
    pub compressed: bool,
}

impl SyncDelta {
    /// Convert to a core scaffold [`CoreDelta`].
    #[cfg(feature = "experimental")]
    pub fn to_core_delta(&self) -> CoreDelta {
        use crate::experimental::crdt::TimestampedOp;

        let replica = ReplicaId(u64::from(hash_node_id(&self.source_node)));
        let operations: Vec<TimestampedOp> = self
            .operations
            .iter()
            .map(|op| {
                let origin = ReplicaId(u64::from(hash_node_id(&op.origin_node)));
                TimestampedOp::new(op.to_core_op(), op.timestamp.to_core_hlc(), origin)
            })
            .collect();
        let from_timestamp = operations.first().map(|op| op.timestamp);
        let to_timestamp = operations.last().map(|op| op.timestamp);

        CoreDelta {
            operations,
            from_timestamp,
            to_timestamp,
            origin: replica,
        }
    }

    /// Create from a core scaffold [`CoreDelta`] plus service context.
    #[cfg(feature = "experimental")]
    pub fn from_core_delta(
        core: &CoreDelta,
        source_node: &str,
        sequence: u64,
        clock: HLC,
    ) -> Self {
        let operations = core
            .operations
            .iter()
            .map(|top| {
                let ts = HLC::from_core_hlc(&top.timestamp);
                let origin = format!("replica-{}", top.origin.0);
                VectorOp::from_core_op(&top.op, ts, origin)
            })
            .collect();

        Self {
            operations,
            source_node: source_node.to_string(),
            sequence,
            clock,
            compressed: false,
        }
    }
}

/// Result of applying a sync delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Number of operations applied.
    pub applied: usize,
    /// Number of operations skipped (already seen).
    pub skipped: usize,
    /// Number of conflicts resolved.
    pub conflicts_resolved: usize,
    /// Conflict details (for UX display).
    pub conflict_details: Vec<ConflictInfo>,
}

#[cfg(feature = "experimental")]
impl SyncResult {
    /// Create from a core [`CoreMergeResult`], with empty conflict details.
    pub fn from_core_merge_result(core: &CoreMergeResult) -> Self {
        Self {
            applied: core.applied,
            skipped: core.skipped,
            conflicts_resolved: core.conflicts,
            conflict_details: Vec::new(),
        }
    }
}

/// Details about a resolved conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictInfo {
    /// Vector ID with the conflict.
    pub vector_id: String,
    /// Resolution strategy used.
    pub resolution: ConflictResolution,
    /// Winning node.
    pub winner: String,
    /// Losing node.
    pub loser: String,
}

/// How a conflict was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last-Writer-Wins (by HLC timestamp).
    LastWriterWins,
    /// Merge metadata fields.
    MetadataMerge,
    /// Keep both (create duplicate with suffix).
    KeepBoth,
}

// ── Sync Peer ───────────────────────────────────────────────────────────────

/// A sync peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncPeer {
    /// Peer node ID.
    pub node_id: String,
    /// Peer endpoint address.
    pub address: String,
    /// Last synced sequence number from this peer.
    pub last_synced_seq: u64,
    /// Connection status.
    pub status: PeerStatus,
    /// Last sync timestamp.
    pub last_sync_at: u64,
    /// Number of successful syncs.
    pub sync_count: u64,
}

/// Peer connection status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeerStatus {
    Connected,
    Disconnected,
    Syncing,
    Error,
}

// ── Sync Config ─────────────────────────────────────────────────────────────

/// Sync service configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Maximum operations per delta batch.
    pub max_delta_size: usize,
    /// Sync interval in milliseconds.
    pub sync_interval_ms: u64,
    /// Tombstone retention period (seconds).
    pub tombstone_retention_secs: u64,
    /// Enable delta compression.
    pub enable_compression: bool,
    /// Default conflict resolution strategy.
    pub conflict_strategy: ConflictResolution,
    /// Maximum number of peers.
    pub max_peers: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_delta_size: 1000,
            sync_interval_ms: 5000,
            tombstone_retention_secs: 86400 * 7, // 7 days
            enable_compression: true,
            conflict_strategy: ConflictResolution::LastWriterWins,
            max_peers: 32,
        }
    }
}

#[cfg(feature = "experimental")]
impl SyncConfig {
    /// Convert to a core scaffold [`CoreSyncConfig`].
    pub fn to_core_config(&self) -> CoreSyncConfig {
        CoreSyncConfig {
            max_ops_per_batch: self.max_delta_size,
            ..CoreSyncConfig::default()
        }
    }
}

// ── Sync Service ────────────────────────────────────────────────────────────

/// Sync service statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    pub total_ops: u64,
    pub total_syncs: u64,
    pub total_conflicts: u64,
    pub total_deltas_sent: u64,
    pub total_deltas_received: u64,
    pub pending_ops: u64,
}

/// CRDT-based vector sync service.
///
/// When the `experimental` feature is enabled, the service maintains an
/// internal [`VectorCRDT`] that mirrors every mutation for a unified backend.
pub struct SyncService {
    node_id: String,
    config: SyncConfig,
    clock: HLC,
    states: HashMap<String, VectorState>,
    op_log: Vec<VectorOp>,
    peers: HashMap<String, SyncPeer>,
    sequence: u64,
    stats: SyncStats,
    /// Core CRDT backend (experimental feature).
    #[cfg(feature = "experimental")]
    crdt: VectorCRDT,
    /// Replica identifier for the core CRDT backend.
    #[cfg(feature = "experimental")]
    replica_id: ReplicaId,
}

impl SyncService {
    /// Create a new sync service for the given node.
    pub fn new(node_id: String, config: SyncConfig) -> Self {
        #[cfg(feature = "experimental")]
        let replica_id = ReplicaId(u64::from(hash_node_id(&node_id)));

        Self {
            clock: HLC::new(&node_id),
            #[cfg(feature = "experimental")]
            crdt: VectorCRDT::new(replica_id),
            #[cfg(feature = "experimental")]
            replica_id,
            node_id,
            config,
            states: HashMap::new(),
            op_log: Vec::new(),
            peers: HashMap::new(),
            sequence: 0,
            stats: SyncStats::default(),
        }
    }

    /// Record a local vector insert.
    pub fn record_insert(
        &mut self,
        vector_id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) {
        self.clock.tick();

        // Delegate to core CRDT backend
        #[cfg(feature = "experimental")]
        {
            let meta_map = json_to_string_map(metadata.as_ref());
            if let Err(e) = self.crdt.add(vector_id, vector, meta_map) {
                warn!(vector_id, error = %e, "CRDT add failed during record_insert");
            }
        }

        let op = VectorOp {
            op_type: OpType::Insert,
            vector_id: vector_id.to_string(),
            vector: Some(vector.to_vec()),
            metadata: metadata.clone(),
            timestamp: self.clock,
            origin_node: self.node_id.clone(),
        };
        let mut seen_by = HashSet::new();
        seen_by.insert(self.node_id.clone());
        self.states.insert(
            vector_id.to_string(),
            VectorState {
                vector: vector.to_vec(),
                metadata,
                last_write: self.clock,
                deleted: false,
                deleted_at: None,
                seen_by,
            },
        );
        self.op_log.push(op);
        self.sequence += 1;
        self.stats.total_ops += 1;
        self.stats.pending_ops += 1;
    }

    /// Record a local vector update.
    pub fn record_update(
        &mut self,
        vector_id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) {
        self.clock.tick();

        // Delegate to core CRDT backend
        #[cfg(feature = "experimental")]
        {
            if let Err(e) = self.crdt.update(vector_id, vector) {
                warn!(vector_id, error = %e, "CRDT update failed during record_update");
            }
        }

        let op = VectorOp {
            op_type: OpType::Update,
            vector_id: vector_id.to_string(),
            vector: Some(vector.to_vec()),
            metadata: metadata.clone(),
            timestamp: self.clock,
            origin_node: self.node_id.clone(),
        };
        if let Some(state) = self.states.get_mut(vector_id) {
            state.vector = vector.to_vec();
            state.metadata = metadata;
            state.last_write = self.clock;
            state.seen_by.insert(self.node_id.clone());
        }
        self.op_log.push(op);
        self.sequence += 1;
        self.stats.total_ops += 1;
        self.stats.pending_ops += 1;
    }

    /// Record a local vector delete.
    pub fn record_delete(&mut self, vector_id: &str) {
        self.clock.tick();

        // Delegate to core CRDT backend
        #[cfg(feature = "experimental")]
        {
            if let Err(e) = self.crdt.delete(vector_id) {
                warn!(vector_id, error = %e, "CRDT delete failed during record_delete");
            }
        }

        let op = VectorOp {
            op_type: OpType::Delete,
            vector_id: vector_id.to_string(),
            vector: None,
            metadata: None,
            timestamp: self.clock,
            origin_node: self.node_id.clone(),
        };
        if let Some(state) = self.states.get_mut(vector_id) {
            state.deleted = true;
            state.deleted_at = Some(self.clock);
            state.seen_by.insert(self.node_id.clone());
        }
        self.op_log.push(op);
        self.sequence += 1;
        self.stats.total_ops += 1;
        self.stats.pending_ops += 1;
    }

    /// Generate a delta of operations since a given sequence.
    ///
    /// When `experimental` is enabled, also generates a [`CoreDelta`] via the
    /// core backend; the returned [`SyncDelta`] is built from the local op log
    /// so existing callers see identical behaviour.
    pub fn generate_delta(&mut self, since_seq: u64) -> SyncDelta {
        // Delegate to core CRDT backend
        #[cfg(feature = "experimental")]
        {
            let since_hlc = if since_seq > 0 {
                self.op_log
                    .get((since_seq as usize).saturating_sub(1))
                    .map(|op| op.timestamp.to_core_hlc())
            } else {
                None
            };
            let _core_delta = self.crdt.delta_since(since_hlc);
        }

        let ops: Vec<VectorOp> = self
            .op_log
            .iter()
            .skip(since_seq as usize)
            .take(self.config.max_delta_size)
            .cloned()
            .collect();
        self.stats.total_deltas_sent += 1;
        self.stats.pending_ops = 0;
        SyncDelta {
            operations: ops,
            source_node: self.node_id.clone(),
            sequence: self.sequence,
            clock: self.clock,
            compressed: false,
        }
    }

    /// Apply a remote delta.
    ///
    /// When `experimental` is enabled, the delta is also merged into the core
    /// [`VectorCRDT`] backend.
    pub fn apply_delta(&mut self, delta: SyncDelta) -> SyncResult {
        self.clock.merge(&delta.clock);

        // Delegate to core CRDT backend
        #[cfg(feature = "experimental")]
        {
            let core_delta = delta.to_core_delta();
            if let Err(e) = self.crdt.merge(core_delta) {
                warn!(error = %e, "CRDT merge failed during apply_delta");
            }
        }

        self.stats.total_deltas_received += 1;
        self.stats.total_syncs += 1;

        let mut applied = 0;
        let mut skipped = 0;
        let mut conflicts_resolved = 0;
        let mut conflict_details = Vec::new();

        for op in &delta.operations {
            match self.apply_op(op) {
                ApplyResult::Applied => applied += 1,
                ApplyResult::Skipped => skipped += 1,
                ApplyResult::Conflict(info) => {
                    conflicts_resolved += 1;
                    conflict_details.push(info);
                }
            }
        }

        self.stats.total_conflicts += conflicts_resolved as u64;

        // Update peer tracking
        if let Some(peer) = self.peers.get_mut(&delta.source_node) {
            peer.last_synced_seq = delta.sequence;
            peer.last_sync_at = now_ms() / 1000;
            peer.sync_count += 1;
        }

        SyncResult {
            applied,
            skipped,
            conflicts_resolved,
            conflict_details,
        }
    }

    fn apply_op(&mut self, op: &VectorOp) -> ApplyResult {
        if let Some(existing) = self.states.get(&op.vector_id) {
            // Check if we've already seen this (idempotency)
            if existing.seen_by.contains(&op.origin_node) && existing.last_write >= op.timestamp {
                return ApplyResult::Skipped;
            }
            // Conflict: concurrent writes to same vector
            if existing.last_write != op.timestamp
                && !existing.seen_by.contains(&op.origin_node)
            {
                let winner = if op.timestamp > existing.last_write {
                    op.origin_node.clone()
                } else {
                    self.node_id.clone()
                };
                let loser = if winner == op.origin_node {
                    self.node_id.clone()
                } else {
                    op.origin_node.clone()
                };

                match self.config.conflict_strategy {
                    ConflictResolution::LastWriterWins => {
                        if op.timestamp > existing.last_write {
                            self.apply_op_to_state(op);
                        }
                    }
                    ConflictResolution::MetadataMerge => {
                        self.merge_metadata(op);
                    }
                    ConflictResolution::KeepBoth => {
                        // Apply the remote op under a suffixed ID
                        let mut dup_op = op.clone();
                        dup_op.vector_id = format!("{}_{}", op.vector_id, op.origin_node);
                        self.apply_op_to_state(&dup_op);
                    }
                }

                return ApplyResult::Conflict(ConflictInfo {
                    vector_id: op.vector_id.clone(),
                    resolution: self.config.conflict_strategy,
                    winner,
                    loser,
                });
            }
        }

        self.apply_op_to_state(op);
        ApplyResult::Applied
    }

    fn apply_op_to_state(&mut self, op: &VectorOp) {
        match op.op_type {
            OpType::Insert | OpType::Update => {
                let mut seen_by = HashSet::new();
                seen_by.insert(self.node_id.clone());
                seen_by.insert(op.origin_node.clone());
                self.states.insert(
                    op.vector_id.clone(),
                    VectorState {
                        vector: op.vector.clone().unwrap_or_default(),
                        metadata: op.metadata.clone(),
                        last_write: op.timestamp,
                        deleted: false,
                        deleted_at: None,
                        seen_by,
                    },
                );
            }
            OpType::Delete => {
                if let Some(state) = self.states.get_mut(&op.vector_id) {
                    state.deleted = true;
                    state.deleted_at = Some(op.timestamp);
                    state.seen_by.insert(op.origin_node.clone());
                }
            }
            OpType::MetadataUpdate => {
                if let Some(state) = self.states.get_mut(&op.vector_id) {
                    if let Some(meta) = &op.metadata {
                        state.metadata = Some(meta.clone());
                    }
                    state.last_write = op.timestamp;
                    state.seen_by.insert(op.origin_node.clone());
                }
            }
        }
    }

    fn merge_metadata(&mut self, op: &VectorOp) {
        if let Some(state) = self.states.get_mut(&op.vector_id) {
            // Merge metadata objects (remote overwrites individual keys)
            if let (Some(existing_meta), Some(new_meta)) = (&state.metadata, &op.metadata) {
                if let (Some(existing_obj), Some(new_obj)) =
                    (existing_meta.as_object(), new_meta.as_object())
                {
                    let mut merged = existing_obj.clone();
                    for (k, v) in new_obj {
                        merged.insert(k.clone(), v.clone());
                    }
                    state.metadata = Some(serde_json::Value::Object(merged));
                }
            }
            // Update vector if newer
            if op.timestamp > state.last_write {
                if let Some(vec) = &op.vector {
                    state.vector = vec.clone();
                }
                state.last_write = op.timestamp;
            }
            state.seen_by.insert(op.origin_node.clone());
        }
    }

    // ── Peer Management ─────────────────────────────────────────────────

    /// Add a sync peer.
    pub fn add_peer(&mut self, node_id: String, address: String) -> Result<()> {
        if self.peers.len() >= self.config.max_peers {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum peers ({}) reached",
                self.config.max_peers
            )));
        }
        self.peers.insert(
            node_id.clone(),
            SyncPeer {
                node_id,
                address,
                last_synced_seq: 0,
                status: PeerStatus::Disconnected,
                last_sync_at: 0,
                sync_count: 0,
            },
        );
        Ok(())
    }

    /// Remove a sync peer.
    pub fn remove_peer(&mut self, node_id: &str) {
        self.peers.remove(node_id);
    }

    /// Get a peer.
    pub fn peer(&self, node_id: &str) -> Option<&SyncPeer> {
        self.peers.get(node_id)
    }

    /// List all peers.
    pub fn peers(&self) -> Vec<&SyncPeer> {
        self.peers.values().collect()
    }

    /// Garbage-collect old tombstones.
    pub fn gc_tombstones(&mut self) -> usize {
        let cutoff_ms = now_ms().saturating_sub(self.config.tombstone_retention_secs * 1000);
        let before = self.states.len();
        self.states.retain(|_, state| {
            if state.deleted {
                state
                    .deleted_at
                    .map_or(true, |ts| ts.physical > cutoff_ms)
            } else {
                true
            }
        });
        before - self.states.len()
    }

    /// Get current state of a vector.
    pub fn vector_state(&self, vector_id: &str) -> Option<&VectorState> {
        self.states.get(vector_id)
    }

    /// Count active (non-deleted) vectors.
    pub fn active_vector_count(&self) -> usize {
        self.states.values().filter(|s| !s.deleted).count()
    }

    /// Get the current HLC clock.
    pub fn clock(&self) -> &HLC {
        &self.clock
    }

    /// Get node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get stats.
    pub fn stats(&self) -> &SyncStats {
        &self.stats
    }

    /// Get config.
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    /// Get current sequence.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    // ── Core CRDT backend accessors (experimental) ──────────────────────

    /// Get a reference to the underlying [`VectorCRDT`] backend.
    #[cfg(feature = "experimental")]
    pub fn crdt(&self) -> &VectorCRDT {
        &self.crdt
    }

    /// Get the [`ReplicaId`] used by the core backend.
    #[cfg(feature = "experimental")]
    pub fn replica_id(&self) -> ReplicaId {
        self.replica_id
    }

    /// Get the core backend's [`CoreSyncState`].
    #[cfg(feature = "experimental")]
    pub fn core_sync_state(&self) -> CoreSyncState {
        self.crdt.sync_state()
    }
}

enum ApplyResult {
    Applied,
    Skipped,
    Conflict(ConflictInfo),
}

// ── Helper functions ────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn hash_node_id(node_id: &str) -> u32 {
    let mut hash: u32 = 0;
    for byte in node_id.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(u32::from(byte));
    }
    hash
}

// ── Bridge helpers (experimental) ───────────────────────────────────────────

/// Convert a JSON metadata value to a `HashMap<String, String>`.
#[cfg(feature = "experimental")]
fn json_to_string_map(value: Option<&serde_json::Value>) -> HashMap<String, String> {
    match value.and_then(|v| v.as_object()) {
        Some(obj) => obj
            .iter()
            .map(|(k, v)| {
                let s = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (k.clone(), s)
            })
            .collect(),
        None => HashMap::new(),
    }
}

/// Convert a `HashMap<String, String>` to a JSON object.
#[cfg(feature = "experimental")]
fn string_map_to_json(map: &HashMap<String, String>) -> serde_json::Value {
    let obj: serde_json::Map<String, serde_json::Value> = map
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect();
    serde_json::Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_service(node_id: &str) -> SyncService {
        SyncService::new(node_id.into(), SyncConfig::default())
    }

    #[test]
    fn test_record_insert() {
        let mut svc = make_service("n1");
        svc.record_insert("doc-1", &[0.1, 0.2, 0.3], None);
        assert_eq!(svc.active_vector_count(), 1);
        let state = svc.vector_state("doc-1").unwrap();
        assert_eq!(state.vector.len(), 3);
        assert!(!state.deleted);
    }

    #[test]
    fn test_record_update() {
        let mut svc = make_service("n1");
        svc.record_insert("doc-1", &[0.1, 0.2], None);
        svc.record_update("doc-1", &[0.3, 0.4], None);
        let state = svc.vector_state("doc-1").unwrap();
        assert!((state.vector[0] - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_record_delete() {
        let mut svc = make_service("n1");
        svc.record_insert("doc-1", &[0.1, 0.2], None);
        svc.record_delete("doc-1");
        let state = svc.vector_state("doc-1").unwrap();
        assert!(state.deleted);
        assert_eq!(svc.active_vector_count(), 0);
    }

    #[test]
    fn test_generate_delta() {
        let mut svc = make_service("n1");
        svc.record_insert("doc-1", &[0.1, 0.2], None);
        svc.record_insert("doc-2", &[0.3, 0.4], None);
        let delta = svc.generate_delta(0);
        assert_eq!(delta.operations.len(), 2);
        assert_eq!(delta.source_node, "n1");
    }

    #[test]
    fn test_apply_delta() {
        let mut svc1 = make_service("n1");
        svc1.record_insert(
            "doc-1",
            &[0.1, 0.2],
            Some(serde_json::json!({"title": "Hello"})),
        );
        let delta = svc1.generate_delta(0);

        let mut svc2 = make_service("n2");
        let result = svc2.apply_delta(delta);
        assert_eq!(result.applied, 1);
        assert_eq!(svc2.active_vector_count(), 1);
    }

    #[test]
    fn test_conflict_lww() {
        let mut svc1 = make_service("n1");
        let mut svc2 = make_service("n2");

        // Both write to same ID
        svc1.record_insert("doc-1", &[0.1, 0.2], None);
        svc2.record_insert("doc-1", &[0.3, 0.4], None);

        // Sync from svc1 to svc2
        let delta = svc1.generate_delta(0);
        let result = svc2.apply_delta(delta);
        // One of them wins by HLC
        assert!(result.applied + result.conflicts_resolved == 1);
    }

    #[test]
    fn test_hlc_ordering() {
        let mut c1 = HLC::new("n1");
        let mut c2 = HLC::new("n2");
        c1.tick();
        c2.tick();
        c2.merge(&c1);
        assert!(c2 >= c1);
    }

    #[test]
    fn test_peer_management() {
        let mut svc = make_service("n1");
        svc.add_peer("n2".into(), "10.0.0.2:8080".into()).unwrap();
        assert_eq!(svc.peers().len(), 1);
        svc.remove_peer("n2");
        assert_eq!(svc.peers().len(), 0);
    }

    #[test]
    fn test_max_peers() {
        let mut svc = SyncService::new(
            "n1".into(),
            SyncConfig {
                max_peers: 2,
                ..Default::default()
            },
        );
        svc.add_peer("n2".into(), "a".into()).unwrap();
        svc.add_peer("n3".into(), "b".into()).unwrap();
        assert!(svc.add_peer("n4".into(), "c".into()).is_err());
    }

    #[test]
    fn test_gc_tombstones() {
        let mut svc = SyncService::new(
            "n1".into(),
            SyncConfig {
                tombstone_retention_secs: 0, // expire immediately
                ..Default::default()
            },
        );
        svc.record_insert("doc-1", &[0.1], None);
        svc.record_delete("doc-1");
        let cleaned = svc.gc_tombstones();
        assert_eq!(cleaned, 1);
    }

    #[test]
    fn test_stats_tracking() {
        let mut svc = make_service("n1");
        svc.record_insert("a", &[0.1], None);
        svc.record_insert("b", &[0.2], None);
        svc.record_delete("a");
        assert_eq!(svc.stats().total_ops, 3);
    }

    #[test]
    fn test_idempotent_apply() {
        let mut svc1 = make_service("n1");
        svc1.record_insert("doc-1", &[0.1, 0.2], None);
        let delta = svc1.generate_delta(0);

        let mut svc2 = make_service("n2");
        let r1 = svc2.apply_delta(delta.clone());
        let r2 = svc2.apply_delta(delta);
        assert_eq!(r1.applied, 1);
        assert_eq!(r2.skipped, 1); // second apply is idempotent
    }

    #[test]
    fn test_metadata_in_sync() {
        let mut svc1 = make_service("n1");
        svc1.record_insert(
            "doc-1",
            &[0.1],
            Some(serde_json::json!({"category": "test"})),
        );
        let delta = svc1.generate_delta(0);

        let mut svc2 = make_service("n2");
        svc2.apply_delta(delta);
        let state = svc2.vector_state("doc-1").unwrap();
        assert_eq!(state.metadata.as_ref().unwrap()["category"], "test");
    }

    #[test]
    fn test_sequence_numbers() {
        let mut svc = make_service("n1");
        assert_eq!(svc.sequence(), 0);
        svc.record_insert("a", &[0.1], None);
        assert_eq!(svc.sequence(), 1);
        svc.record_insert("b", &[0.2], None);
        assert_eq!(svc.sequence(), 2);
    }
}
