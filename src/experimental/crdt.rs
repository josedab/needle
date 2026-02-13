//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Edge Sync - Offline-first CRDT-based synchronization for vector databases.
//!
//! Enables conflict-free replication between edge devices and central servers
//! using Conflict-free Replicated Data Types (CRDTs).
//!
//! # Features
//!
//! - **Offline-first**: Work without connectivity, sync when available
//! - **Conflict-free**: Automatic conflict resolution without coordination
//! - **Vector CRDTs**: Specialized CRDTs for vector embeddings
//! - **Delta sync**: Efficient incremental synchronization
//! - **Causal ordering**: Preserve operation ordering
//!
//! # Example
//!
//! ```ignore
//! use needle::crdt::{VectorCRDT, ReplicaId, SyncManager};
//!
//! let replica_id = ReplicaId::new();
//! let mut crdt = VectorCRDT::new(replica_id);
//!
//! // Add vectors (works offline)
//! crdt.add("vec1", &embedding, metadata)?;
//!
//! // Get delta for sync
//! let delta = crdt.delta_since(last_sync);
//!
//! // Merge remote changes
//! crdt.merge(remote_delta)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// Unique replica identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaId(pub u64);

impl ReplicaId {
    /// Generate a new replica ID.
    pub fn new() -> Self {
        use rand::Rng;
        ReplicaId(rand::thread_rng().gen())
    }

    /// Create from specific value.
    pub fn from(id: u64) -> Self {
        ReplicaId(id)
    }
}

impl Default for ReplicaId {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid Logical Clock for causal ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HLC {
    /// Physical timestamp.
    pub physical: u64,
    /// Logical counter.
    pub logical: u32,
    /// Replica ID.
    pub replica: u64,
}

impl HLC {
    /// Create new HLC.
    pub fn new(replica: ReplicaId) -> Self {
        Self {
            physical: Self::now(),
            logical: 0,
            replica: replica.0,
        }
    }

    /// Get current time.
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_millis() as u64
    }

    /// Tick the clock for local event.
    pub fn tick(&mut self) -> HLC {
        let now = Self::now();
        if now > self.physical {
            self.physical = now;
            self.logical = 0;
        } else {
            self.logical += 1;
        }
        *self
    }

    /// Update clock on receiving remote timestamp.
    pub fn receive(&mut self, remote: HLC) -> HLC {
        let now = Self::now();
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
        *self
    }
}

/// Operation types for the CRDT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    /// Add a vector.
    Add {
        id: String,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    },
    /// Update a vector.
    Update { id: String, vector: Vec<f32> },
    /// Update metadata.
    UpdateMetadata {
        id: String,
        key: String,
        value: Option<String>,
    },
    /// Delete a vector.
    Delete { id: String },
}

/// An operation with its timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedOp {
    /// The operation.
    pub op: Operation,
    /// HLC timestamp.
    pub timestamp: HLC,
    /// Origin replica.
    pub origin: ReplicaId,
}

impl TimestampedOp {
    /// Create new timestamped operation.
    pub fn new(op: Operation, timestamp: HLC, origin: ReplicaId) -> Self {
        Self {
            op,
            timestamp,
            origin,
        }
    }
}

/// Vector entry in the CRDT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRDTVector {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata (LWW-Register per key).
    pub metadata: HashMap<String, (String, HLC)>,
    /// Creation timestamp.
    pub created_at: HLC,
    /// Last update timestamp.
    pub updated_at: HLC,
    /// Tombstone (for deletion).
    pub deleted: Option<HLC>,
}

impl CRDTVector {
    /// Create new vector entry.
    pub fn new(
        id: &str,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
        timestamp: HLC,
    ) -> Self {
        let meta_with_ts: HashMap<String, (String, HLC)> = metadata
            .into_iter()
            .map(|(k, v)| (k, (v, timestamp)))
            .collect();

        Self {
            id: id.to_string(),
            vector,
            metadata: meta_with_ts,
            created_at: timestamp,
            updated_at: timestamp,
            deleted: None,
        }
    }

    /// Get metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|(v, _)| v.as_str())
    }

    /// Get all metadata.
    pub fn get_all_metadata(&self) -> HashMap<String, String> {
        self.metadata
            .iter()
            .map(|(k, (v, _))| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if deleted.
    pub fn is_deleted(&self) -> bool {
        self.deleted.is_some()
    }
}

/// Delta for synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Operations in this delta.
    pub operations: Vec<TimestampedOp>,
    /// Minimum timestamp in delta.
    pub from_timestamp: Option<HLC>,
    /// Maximum timestamp in delta.
    pub to_timestamp: Option<HLC>,
    /// Origin replica.
    pub origin: ReplicaId,
}

impl Delta {
    /// Create empty delta.
    pub fn empty(origin: ReplicaId) -> Self {
        Self {
            operations: Vec::new(),
            from_timestamp: None,
            to_timestamp: None,
            origin,
        }
    }

    /// Check if delta is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

/// CRDT-based vector store.
pub struct VectorCRDT {
    /// This replica's ID.
    replica_id: ReplicaId,
    /// Hybrid logical clock.
    clock: HLC,
    /// Vector entries.
    vectors: HashMap<String, CRDTVector>,
    /// Operation log for delta sync.
    operation_log: BTreeMap<HLC, TimestampedOp>,
    /// Peers we know about.
    known_peers: HashSet<ReplicaId>,
    /// Last sync timestamp per peer.
    peer_sync_state: HashMap<ReplicaId, HLC>,
}

impl VectorCRDT {
    /// Create new CRDT.
    pub fn new(replica_id: ReplicaId) -> Self {
        Self {
            replica_id,
            clock: HLC::new(replica_id),
            vectors: HashMap::new(),
            operation_log: BTreeMap::new(),
            known_peers: HashSet::new(),
            peer_sync_state: HashMap::new(),
        }
    }

    /// Get replica ID.
    pub fn replica_id(&self) -> ReplicaId {
        self.replica_id
    }

    /// Add a vector.
    pub fn add(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: HashMap<String, String>,
    ) -> Result<HLC> {
        let timestamp = self.clock.tick();

        let op = Operation::Add {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata: metadata.clone(),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Update a vector.
    pub fn update(&mut self, id: &str, vector: &[f32]) -> Result<HLC> {
        if !self.vectors.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        let timestamp = self.clock.tick();

        let op = Operation::Update {
            id: id.to_string(),
            vector: vector.to_vec(),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Update metadata.
    pub fn update_metadata(&mut self, id: &str, key: &str, value: Option<&str>) -> Result<HLC> {
        if !self.vectors.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        let timestamp = self.clock.tick();

        let op = Operation::UpdateMetadata {
            id: id.to_string(),
            key: key.to_string(),
            value: value.map(|s| s.to_string()),
        };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Delete a vector.
    pub fn delete(&mut self, id: &str) -> Result<HLC> {
        if !self.vectors.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        let timestamp = self.clock.tick();

        let op = Operation::Delete { id: id.to_string() };

        self.apply_operation(op.clone(), timestamp)?;

        let timestamped = TimestampedOp::new(op, timestamp, self.replica_id);
        self.operation_log.insert(timestamp, timestamped);

        Ok(timestamp)
    }

    /// Apply an operation.
    fn apply_operation(&mut self, op: Operation, timestamp: HLC) -> Result<()> {
        match op {
            Operation::Add {
                id,
                vector,
                metadata,
            } => {
                if let Some(existing) = self.vectors.get(&id) {
                    // LWW: only apply if newer
                    if timestamp > existing.updated_at {
                        let entry = CRDTVector::new(&id, vector, metadata, timestamp);
                        self.vectors.insert(id, entry);
                    }
                } else {
                    let entry = CRDTVector::new(&id, vector, metadata, timestamp);
                    self.vectors.insert(id, entry);
                }
            }
            Operation::Update { id, vector } => {
                if let Some(entry) = self.vectors.get_mut(&id) {
                    if timestamp > entry.updated_at && entry.deleted.is_none() {
                        entry.vector = vector;
                        entry.updated_at = timestamp;
                    }
                }
            }
            Operation::UpdateMetadata { id, key, value } => {
                if let Some(entry) = self.vectors.get_mut(&id) {
                    if entry.deleted.is_none() {
                        let should_update = entry
                            .metadata
                            .get(&key)
                            .map(|(_, ts)| timestamp > *ts)
                            .unwrap_or(true);

                        if should_update {
                            if let Some(v) = value {
                                entry.metadata.insert(key, (v, timestamp));
                            } else {
                                entry.metadata.remove(&key);
                            }
                            entry.updated_at = timestamp;
                        }
                    }
                }
            }
            Operation::Delete { id } => {
                if let Some(entry) = self.vectors.get_mut(&id) {
                    // Only delete if timestamp is newer
                    if entry.deleted.map(|d| timestamp > d).unwrap_or(true)
                        && timestamp > entry.updated_at
                    {
                        entry.deleted = Some(timestamp);
                    }
                }
            }
        }
        Ok(())
    }

    /// Get a vector.
    pub fn get(&self, id: &str) -> Option<&CRDTVector> {
        self.vectors.get(id).filter(|v| !v.is_deleted())
    }

    /// List all vectors (excluding deleted).
    pub fn list(&self) -> Vec<&CRDTVector> {
        self.vectors.values().filter(|v| !v.is_deleted()).collect()
    }

    /// Get delta since a timestamp.
    pub fn delta_since(&self, since: Option<HLC>) -> Delta {
        let operations: Vec<TimestampedOp> = match since {
            Some(ts) => self
                .operation_log
                .range(ts..)
                .filter(|(t, _)| **t > ts)
                .map(|(_, op)| op.clone())
                .collect(),
            None => self.operation_log.values().cloned().collect(),
        };

        let from_timestamp = operations.first().map(|op| op.timestamp);
        let to_timestamp = operations.last().map(|op| op.timestamp);

        Delta {
            operations,
            from_timestamp,
            to_timestamp,
            origin: self.replica_id,
        }
    }

    /// Get delta for a specific peer.
    pub fn delta_for_peer(&self, peer: ReplicaId) -> Delta {
        let last_sync = self.peer_sync_state.get(&peer).copied();
        self.delta_since(last_sync)
    }

    /// Merge a delta from another replica.
    pub fn merge(&mut self, delta: Delta) -> Result<MergeResult> {
        if delta.is_empty() {
            return Ok(MergeResult::default());
        }

        let mut applied = 0;
        let mut skipped = 0;
        let mut conflicts = 0;

        // Update clock
        if let Some(ts) = delta.to_timestamp {
            self.clock.receive(ts);
        }

        // Track peer
        self.known_peers.insert(delta.origin);

        for timestamped in delta.operations {
            // Skip our own operations
            if timestamped.origin == self.replica_id {
                skipped += 1;
                continue;
            }

            // Check if we already have this operation
            if self.operation_log.contains_key(&timestamped.timestamp) {
                skipped += 1;
                continue;
            }

            // Check for conflicts (same key, concurrent updates)
            let has_conflict = match &timestamped.op {
                Operation::Update { id, .. } | Operation::Delete { id } => {
                    if let Some(existing) = self.vectors.get(id) {
                        existing.updated_at.replica != timestamped.timestamp.replica
                            && existing.updated_at.physical == timestamped.timestamp.physical
                    } else {
                        false
                    }
                }
                _ => false,
            };

            if has_conflict {
                conflicts += 1;
            }

            // Apply operation
            self.apply_operation(timestamped.op.clone(), timestamped.timestamp)?;
            self.operation_log
                .insert(timestamped.timestamp, timestamped);
            applied += 1;
        }

        // Update sync state
        if let Some(ts) = delta.to_timestamp {
            self.peer_sync_state.insert(delta.origin, ts);
        }

        Ok(MergeResult {
            applied,
            skipped,
            conflicts,
        })
    }

    /// Compact the operation log.
    pub fn compact(&mut self, keep_after: HLC) {
        self.operation_log.retain(|ts, _| *ts >= keep_after);
    }

    /// Get current clock value.
    pub fn current_clock(&self) -> HLC {
        self.clock
    }

    /// Get number of vectors (excluding deleted).
    pub fn len(&self) -> usize {
        self.vectors.values().filter(|v| !v.is_deleted()).count()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get operation log size.
    pub fn log_size(&self) -> usize {
        self.operation_log.len()
    }

    /// Get sync state.
    pub fn sync_state(&self) -> SyncState {
        SyncState {
            replica_id: self.replica_id,
            clock: self.clock,
            vector_count: self.len(),
            log_size: self.log_size(),
            known_peers: self.known_peers.iter().copied().collect(),
        }
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone, Default)]
pub struct MergeResult {
    /// Operations applied.
    pub applied: usize,
    /// Operations skipped (duplicates).
    pub skipped: usize,
    /// Conflicts resolved.
    pub conflicts: usize,
}

/// Sync state information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// This replica's ID.
    pub replica_id: ReplicaId,
    /// Current clock.
    pub clock: HLC,
    /// Number of vectors.
    pub vector_count: usize,
    /// Operation log size.
    pub log_size: usize,
    /// Known peers.
    pub known_peers: Vec<ReplicaId>,
}

// ── Bandwidth-Aware Sync ─────────────────────────────────────────────────────

/// Configuration for bandwidth-aware synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Maximum bytes per sync batch (0 = unlimited).
    pub max_batch_bytes: usize,
    /// Maximum operations per batch (0 = unlimited).
    pub max_ops_per_batch: usize,
    /// Priority for which vectors to sync first.
    pub priority: SyncPriority,
    /// Filter to selectively sync only matching vector IDs.
    pub id_prefix_filter: Option<String>,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_batch_bytes: 0,
            max_ops_per_batch: 1000,
            priority: SyncPriority::Chronological,
            id_prefix_filter: None,
        }
    }
}

/// Priority ordering for sync operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncPriority {
    /// Send oldest operations first (FIFO).
    Chronological,
    /// Send newest operations first (most recent data arrives first).
    ReverseChronological,
}

/// A bandwidth-constrained batch of operations for sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncBatch {
    /// The operations in this batch.
    pub operations: Vec<TimestampedOp>,
    /// Whether more batches remain after this one.
    pub has_more: bool,
    /// Estimated size of this batch in bytes.
    pub estimated_bytes: usize,
    /// Resume token to continue from in the next batch.
    pub resume_token: Option<HLC>,
}

impl VectorCRDT {
    /// Generate a bandwidth-aware delta for a peer, respecting size limits.
    ///
    /// Use `resume_token` from the previous batch's response to continue
    /// where the last batch left off.
    pub fn delta_batched(
        &self,
        peer: ReplicaId,
        config: &SyncConfig,
        resume_token: Option<HLC>,
    ) -> SyncBatch {
        let since = resume_token.or_else(|| self.peer_sync_state.get(&peer).copied());

        let all_ops: Vec<&TimestampedOp> = if let Some(ref since_ts) = since {
            self.operation_log
                .range(since_ts..)
                .map(|(_, op)| op)
                .filter(|op| op.origin != self.replica_id || resume_token.is_some())
                .collect()
        } else {
            self.operation_log.values().collect()
        };

        // Apply selective filter
        let filtered_ops: Vec<&TimestampedOp> = if let Some(ref prefix) = config.id_prefix_filter {
            all_ops
                .into_iter()
                .filter(|op| op.op.affected_id().map_or(true, |id| id.starts_with(prefix)))
                .collect()
        } else {
            all_ops
        };

        // Apply priority ordering
        let mut sorted_ops = filtered_ops;
        if config.priority == SyncPriority::ReverseChronological {
            sorted_ops.reverse();
        }

        let mut batch_ops = Vec::new();
        let mut total_bytes = 0;
        let max_ops = if config.max_ops_per_batch == 0 {
            usize::MAX
        } else {
            config.max_ops_per_batch
        };

        for op in &sorted_ops {
            let op_size = Self::estimate_op_size(op);
            if config.max_batch_bytes > 0 && total_bytes + op_size > config.max_batch_bytes {
                break;
            }
            if batch_ops.len() >= max_ops {
                break;
            }
            batch_ops.push((*op).clone());
            total_bytes += op_size;
        }

        let has_more = batch_ops.len() < sorted_ops.len();
        let resume = batch_ops.last().map(|op| op.timestamp);

        SyncBatch {
            operations: batch_ops,
            has_more,
            estimated_bytes: total_bytes,
            resume_token: if has_more { resume } else { None },
        }
    }

    /// Estimate the serialized size of an operation in bytes.
    fn estimate_op_size(op: &TimestampedOp) -> usize {
        match &op.op {
            Operation::Add { id, vector, metadata } => {
                id.len() + vector.len() * 4 + metadata.len() * 32 + 16
            }
            Operation::Update { id, vector } => id.len() + vector.len() * 4 + 16,
            Operation::Delete { id } => id.len() + 16,
            Operation::UpdateMetadata { id, key, value } => {
                id.len()
                    + key.len()
                    + value.as_ref().map_or(0, |v| v.len())
                    + 16
            }
        }
    }
}

impl Operation {
    /// Get the vector ID affected by this operation.
    fn affected_id(&self) -> Option<&str> {
        match self {
            Operation::Add { id, .. }
            | Operation::Update { id, .. }
            | Operation::Delete { id }
            | Operation::UpdateMetadata { id, .. } => Some(id),
        }
    }
}

// ── Peer Discovery ───────────────────────────────────────────────────────────

/// Information about a discovered peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// The peer's replica ID.
    pub replica_id: ReplicaId,
    /// Human-readable peer name.
    pub name: String,
    /// Network address (e.g., "192.168.1.5:8080").
    pub address: String,
    /// Timestamp when this peer was last seen (seconds since epoch).
    pub last_seen: u64,
    /// Number of vectors this peer holds.
    pub vector_count: usize,
}

/// Simple peer registry for discovery.
pub struct PeerRegistry {
    peers: HashMap<ReplicaId, PeerInfo>,
    /// Peers older than this (seconds) are considered stale.
    stale_threshold_secs: u64,
}

impl PeerRegistry {
    pub fn new(stale_threshold_secs: u64) -> Self {
        Self {
            peers: HashMap::new(),
            stale_threshold_secs,
        }
    }

    /// Register or update a peer.
    pub fn upsert(&mut self, info: PeerInfo) {
        self.peers.insert(info.replica_id, info);
    }

    /// Get all active (non-stale) peers.
    pub fn active_peers(&self) -> Vec<&PeerInfo> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.peers
            .values()
            .filter(|p| now.saturating_sub(p.last_seen) < self.stale_threshold_secs)
            .collect()
    }

    /// Remove stale peers. Returns number removed.
    pub fn prune_stale(&mut self) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let before = self.peers.len();
        self.peers
            .retain(|_, p| now.saturating_sub(p.last_seen) < self.stale_threshold_secs);
        before - self.peers.len()
    }

    /// Total number of registered peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
}

// ============================================================================
// Merkle Tree Anti-Entropy Protocol
// ============================================================================

/// A node in the Merkle tree used for efficient delta detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    /// Hash of this node (derived from children or leaf data)
    pub hash: u64,
    /// Range of vector IDs covered by this node
    pub range_start: String,
    pub range_end: String,
    /// Number of vectors in this subtree
    pub count: usize,
}

/// Merkle tree for detecting differences between replicas.
///
/// Divides the key space into buckets and builds a binary tree of hashes.
/// Two replicas can compare roots, then drill down into differing subtrees
/// to efficiently identify the minimal set of changed keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    /// Tree nodes organized as a flat array (index 0 = root)
    nodes: Vec<MerkleNode>,
    /// Number of leaf buckets (power of 2)
    bucket_count: usize,
    /// Depth of the tree
    depth: usize,
}

impl MerkleTree {
    /// Build a Merkle tree from a set of (id, hash) pairs.
    pub fn build(entries: &mut [(String, u64)], bucket_count: usize) -> Self {
        let bucket_count = bucket_count.next_power_of_two().max(2);
        let depth = (bucket_count as f64).log2() as usize;
        let total_nodes = 2 * bucket_count - 1;
        let mut nodes = Vec::with_capacity(total_nodes);

        // Sort entries by ID for consistent bucketing
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        // Create leaf nodes
        let entries_per_bucket = entries.len().max(1) / bucket_count;
        for i in 0..bucket_count {
            let start = i * entries_per_bucket;
            let end = if i == bucket_count - 1 {
                entries.len()
            } else {
                ((i + 1) * entries_per_bucket).min(entries.len())
            };

            let bucket_entries = if start < entries.len() {
                &entries[start..end.min(entries.len())]
            } else {
                &[]
            };

            let hash = Self::hash_bucket(bucket_entries);
            let range_start = bucket_entries
                .first()
                .map(|(id, _)| id.clone())
                .unwrap_or_default();
            let range_end = bucket_entries
                .last()
                .map(|(id, _)| id.clone())
                .unwrap_or_default();

            nodes.push(MerkleNode {
                hash,
                range_start,
                range_end,
                count: bucket_entries.len(),
            });
        }

        // Build internal nodes bottom-up
        let mut level_start = 0;
        let mut level_size = bucket_count;
        while level_size > 1 {
            let next_level_size = level_size / 2;
            for i in 0..next_level_size {
                let left = &nodes[level_start + i * 2];
                let right = &nodes[level_start + i * 2 + 1];
                let hash = left.hash.wrapping_mul(31).wrapping_add(right.hash);
                let range_start = left.range_start.clone();
                let range_end = right.range_end.clone();
                let count = left.count + right.count;
                nodes.push(MerkleNode {
                    hash,
                    range_start,
                    range_end,
                    count,
                });
            }
            level_start += level_size;
            level_size = next_level_size;
        }

        Self {
            nodes,
            bucket_count,
            depth,
        }
    }

    /// Get the root hash of the tree
    pub fn root_hash(&self) -> u64 {
        self.nodes.last().map(|n| n.hash).unwrap_or(0)
    }

    /// Compare two Merkle trees and return bucket indices that differ
    pub fn diff(&self, other: &MerkleTree) -> Vec<usize> {
        if self.root_hash() == other.root_hash() {
            return Vec::new(); // Trees are identical
        }

        let mut differing_buckets = Vec::new();
        let min_buckets = self.bucket_count.min(other.bucket_count);

        for i in 0..min_buckets {
            if i < self.nodes.len() && i < other.nodes.len() {
                if self.nodes[i].hash != other.nodes[i].hash {
                    differing_buckets.push(i);
                }
            }
        }

        differing_buckets
    }

    /// Get the IDs that need to be synced based on differing buckets
    pub fn keys_in_bucket(&self, bucket_idx: usize) -> (&str, &str) {
        if bucket_idx < self.nodes.len() {
            (
                &self.nodes[bucket_idx].range_start,
                &self.nodes[bucket_idx].range_end,
            )
        } else {
            ("", "")
        }
    }

    /// Get tree depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get total entry count
    pub fn total_count(&self) -> usize {
        self.nodes.last().map(|n| n.count).unwrap_or(0)
    }

    fn hash_bucket(entries: &[(String, u64)]) -> u64 {
        let mut hash = 0u64;
        for (id, val) in entries {
            let mut id_hash = 0u64;
            for byte in id.bytes() {
                id_hash = id_hash.wrapping_mul(31).wrapping_add(byte as u64);
            }
            hash = hash.wrapping_mul(17).wrapping_add(id_hash).wrapping_add(*val);
        }
        hash
    }
}

impl VectorCRDT {
    /// Build a Merkle tree of the current state for anti-entropy comparison.
    pub fn build_merkle_tree(&self, bucket_count: usize) -> MerkleTree {
        let mut entries: Vec<(String, u64)> = self
            .vectors
            .iter()
            .filter(|(_, v)| !v.is_deleted())
            .map(|(id, v)| {
                // Hash the vector content + timestamp for change detection
                let mut hash = v.timestamp.physical;
                hash = hash
                    .wrapping_mul(31)
                    .wrapping_add(v.timestamp.logical as u64);
                for &val in &v.vector {
                    hash = hash.wrapping_mul(17).wrapping_add(val.to_bits() as u64);
                }
                (id.clone(), hash)
            })
            .collect();
        MerkleTree::build(&mut entries, bucket_count)
    }

    /// Compute the delta needed to sync with a remote replica based on
    /// Merkle tree comparison. Returns only operations affecting keys in
    /// differing buckets (much more efficient than full delta).
    pub fn merkle_delta(&self, remote_tree: &MerkleTree, bucket_count: usize) -> Delta {
        let local_tree = self.build_merkle_tree(bucket_count);
        let differing = local_tree.diff(remote_tree);

        if differing.is_empty() {
            return Delta::empty(self.replica_id);
        }

        // Collect key ranges from differing buckets
        let mut key_ranges: Vec<(String, String)> = Vec::new();
        for bucket_idx in &differing {
            let (start, end) = local_tree.keys_in_bucket(*bucket_idx);
            if !start.is_empty() || !end.is_empty() {
                key_ranges.push((start.to_string(), end.to_string()));
            }
        }

        // Filter operations to only those in differing ranges
        let ops: Vec<TimestampedOp> = self
            .operation_log
            .values()
            .filter(|op| {
                if let Some(id) = op.op.affected_id() {
                    key_ranges.iter().any(|(start, end)| {
                        (start.is_empty() || id >= start.as_str())
                            && (end.is_empty() || id <= end.as_str())
                    })
                } else {
                    false
                }
            })
            .cloned()
            .collect();

        Delta {
            operations: ops,
            origin: self.replica_id,
            from_timestamp: None,
            to_timestamp: Some(self.clock),
        }
    }
}

// ============================================================================
// Delta Compression
// ============================================================================

/// Compressed delta using simple run-length encoding for vector data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedDelta {
    /// Compressed operation data
    pub data: Vec<u8>,
    /// Original operation count
    pub operation_count: usize,
    /// Original size in bytes (estimated)
    pub original_bytes: usize,
    /// Compressed size in bytes
    pub compressed_bytes: usize,
    /// Origin replica
    pub origin: ReplicaId,
}

impl Delta {
    /// Compress this delta for network transfer.
    /// Uses a simple variable-length encoding to reduce bandwidth.
    pub fn compress(&self) -> CompressedDelta {
        let original_bytes = self.operations.iter().map(VectorCRDT::estimate_op_size).sum();
        let serialized = self.serialize_compact();
        let compressed = Self::simple_compress(&serialized);

        CompressedDelta {
            data: compressed.clone(),
            operation_count: self.operations.len(),
            original_bytes,
            compressed_bytes: compressed.len(),
            origin: self.origin,
        }
    }

    /// Serialize operations to a compact binary format
    fn serialize_compact(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.operations.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.origin.0.to_le_bytes());

        for op in &self.operations {
            buf.extend_from_slice(&op.timestamp.physical.to_le_bytes());
            buf.extend_from_slice(&(op.timestamp.logical as u32).to_le_bytes());
            buf.extend_from_slice(&op.origin.0.to_le_bytes());

            match &op.op {
                Operation::Add { id, vector, metadata } => {
                    buf.push(0x01);
                    buf.extend_from_slice(&(id.len() as u16).to_le_bytes());
                    buf.extend_from_slice(id.as_bytes());
                    buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                    for v in vector {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                    buf.extend_from_slice(&(metadata.len() as u16).to_le_bytes());
                    for (k, v) in metadata {
                        buf.extend_from_slice(&(k.len() as u16).to_le_bytes());
                        buf.extend_from_slice(k.as_bytes());
                        buf.extend_from_slice(&(v.len() as u16).to_le_bytes());
                        buf.extend_from_slice(v.as_bytes());
                    }
                }
                Operation::Update { id, vector } => {
                    buf.push(0x02);
                    buf.extend_from_slice(&(id.len() as u16).to_le_bytes());
                    buf.extend_from_slice(id.as_bytes());
                    buf.extend_from_slice(&(vector.len() as u32).to_le_bytes());
                    for v in vector {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Operation::Delete { id } => {
                    buf.push(0x03);
                    buf.extend_from_slice(&(id.len() as u16).to_le_bytes());
                    buf.extend_from_slice(id.as_bytes());
                }
                Operation::UpdateMetadata { id, key, value } => {
                    buf.push(0x04);
                    buf.extend_from_slice(&(id.len() as u16).to_le_bytes());
                    buf.extend_from_slice(id.as_bytes());
                    buf.extend_from_slice(&(key.len() as u16).to_le_bytes());
                    buf.extend_from_slice(key.as_bytes());
                    let v_bytes = value.as_deref().unwrap_or("").as_bytes();
                    buf.extend_from_slice(&(v_bytes.len() as u16).to_le_bytes());
                    buf.extend_from_slice(v_bytes);
                }
            }
        }
        buf
    }

    /// Simple LZ-style compression: look for repeated byte patterns
    fn simple_compress(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;
        while i < data.len() {
            // Find run of identical bytes
            let start = i;
            while i + 1 < data.len() && data[i] == data[i + 1] && i - start < 254 {
                i += 1;
            }
            let run_len = i - start + 1;
            if run_len >= 4 {
                // Encode as: 0xFF, count, byte
                result.push(0xFF);
                result.push(run_len as u8);
                result.push(data[start]);
            } else {
                // Copy literal bytes
                for j in start..=i {
                    if data[j] == 0xFF {
                        // Escape the marker byte
                        result.push(0xFF);
                        result.push(1);
                        result.push(0xFF);
                    } else {
                        result.push(data[j]);
                    }
                }
            }
            i += 1;
        }
        result
    }
}

// ── Bidirectional Delta Sync Engine ──────────────────────────────────────────

/// Outcome of a bidirectional sync session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncOutcome {
    /// Operations sent to the remote peer.
    pub sent: usize,
    /// Operations received from the remote peer.
    pub received: usize,
    /// Conflicts resolved during merge.
    pub conflicts: usize,
    /// Whether the sync completed fully (no more batches).
    pub complete: bool,
}

/// Bidirectional delta sync engine for two `VectorCRDT` instances.
///
/// Orchestrates a sync session: computes deltas from both sides,
/// applies them, and returns a summary of changes.
pub struct DeltaSyncEngine;

impl DeltaSyncEngine {
    /// Perform a full bidirectional sync between two CRDT replicas.
    ///
    /// Both replicas exchange deltas and merge them. After sync,
    /// both replicas converge to the same state (eventually consistent).
    pub fn sync(
        local: &mut VectorCRDT,
        remote: &mut VectorCRDT,
    ) -> Result<SyncOutcome> {
        let remote_id = remote.replica_id();
        let local_id = local.replica_id();

        // Compute deltas from each side
        let local_last_sync = local.peer_sync_state.get(&remote_id).copied();
        let remote_last_sync = remote.peer_sync_state.get(&local_id).copied();

        let local_delta = local.delta_since(local_last_sync);
        let remote_delta = remote.delta_since(remote_last_sync);

        let sent = local_delta.len();
        let received = remote_delta.len();

        // Apply remote delta to local
        let local_merge = local.merge(remote_delta)?;
        // Apply local delta to remote
        let remote_merge = remote.merge(local_delta)?;

        Ok(SyncOutcome {
            sent,
            received,
            conflicts: local_merge.conflicts + remote_merge.conflicts,
            complete: true,
        })
    }

    /// Perform a one-way sync: push local changes to remote.
    pub fn push(
        local: &VectorCRDT,
        remote: &mut VectorCRDT,
    ) -> Result<SyncOutcome> {
        let remote_id = remote.replica_id();
        let last_sync = local.peer_sync_state.get(&remote_id).copied();
        let delta = local.delta_since(last_sync);
        let sent = delta.len();
        let merge_result = remote.merge(delta)?;

        Ok(SyncOutcome {
            sent,
            received: 0,
            conflicts: merge_result.conflicts,
            complete: true,
        })
    }

    /// Perform a one-way sync: pull remote changes to local.
    pub fn pull(
        local: &mut VectorCRDT,
        remote: &VectorCRDT,
    ) -> Result<SyncOutcome> {
        let local_id = local.replica_id();
        let last_sync = remote.peer_sync_state.get(&local_id).copied();
        let delta = remote.delta_since(last_sync);
        let received = delta.len();
        let merge_result = local.merge(delta)?;

        Ok(SyncOutcome {
            sent: 0,
            received,
            conflicts: merge_result.conflicts,
            complete: true,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_crdt() {
        let replica = ReplicaId::new();
        let crdt = VectorCRDT::new(replica);

        assert!(crdt.is_empty());
        assert_eq!(crdt.replica_id(), replica);
    }

    #[test]
    fn test_add_vector() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();

        assert_eq!(crdt.len(), 1);
        assert!(crdt.get("vec1").is_some());
    }

    #[test]
    fn test_update_vector() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();
        crdt.update("vec1", &[4.0, 5.0, 6.0]).unwrap();

        let vec = crdt.get("vec1").unwrap();
        assert_eq!(vec.vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_delete_vector() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0, 2.0, 3.0], HashMap::new()).unwrap();
        crdt.delete("vec1").unwrap();

        assert!(crdt.get("vec1").is_none());
        assert_eq!(crdt.len(), 0);
    }

    #[test]
    fn test_metadata() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());

        crdt.add("vec1", &[1.0], meta).unwrap();

        let vec = crdt.get("vec1").unwrap();
        assert_eq!(vec.get_metadata("key"), Some("value"));
    }

    #[test]
    fn test_update_metadata() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt.update_metadata("vec1", "key", Some("value")).unwrap();

        let vec = crdt.get("vec1").unwrap();
        assert_eq!(vec.get_metadata("key"), Some("value"));
    }

    #[test]
    fn test_delta_sync() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut crdt1 = VectorCRDT::new(replica1);
        let mut crdt2 = VectorCRDT::new(replica2);

        // Add on replica 1
        crdt1.add("vec1", &[1.0, 2.0], HashMap::new()).unwrap();
        crdt1.add("vec2", &[3.0, 4.0], HashMap::new()).unwrap();

        // Get delta and merge to replica 2
        let delta = crdt1.delta_since(None);
        let result = crdt2.merge(delta).unwrap();

        assert_eq!(result.applied, 2);
        assert_eq!(crdt2.len(), 2);
    }

    #[test]
    fn test_concurrent_updates() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut crdt1 = VectorCRDT::new(replica1);
        let mut crdt2 = VectorCRDT::new(replica2);

        // Both add same ID concurrently
        crdt1.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt2.add("vec1", &[2.0], HashMap::new()).unwrap();

        // Merge in both directions
        let delta1 = crdt1.delta_since(None);
        let delta2 = crdt2.delta_since(None);

        crdt1.merge(delta2).unwrap();
        crdt2.merge(delta1).unwrap();

        // Both should converge to same value (LWW)
        let vec1 = crdt1.get("vec1").unwrap().vector.clone();
        let vec2 = crdt2.get("vec1").unwrap().vector.clone();
        assert_eq!(vec1, vec2);
    }

    #[test]
    fn test_incremental_delta() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        let checkpoint = crdt.current_clock();

        crdt.add("vec2", &[2.0], HashMap::new()).unwrap();
        crdt.add("vec3", &[3.0], HashMap::new()).unwrap();

        let delta = crdt.delta_since(Some(checkpoint));
        assert_eq!(delta.len(), 2); // Only vec2 and vec3
    }

    #[test]
    fn test_hlc_ordering() {
        let replica = ReplicaId::from(1);
        let mut clock = HLC::new(replica);

        let t1 = clock.tick();
        let t2 = clock.tick();
        let t3 = clock.tick();

        assert!(t1 < t2);
        assert!(t2 < t3);
    }

    #[test]
    fn test_hlc_receive() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut clock1 = HLC::new(replica1);
        let mut clock2 = HLC::new(replica2);

        let t1 = clock1.tick();
        clock2.receive(t1);

        let t2 = clock2.tick();
        assert!(t2 > t1);
    }

    #[test]
    fn test_compact() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt.add("vec2", &[2.0], HashMap::new()).unwrap();
        let checkpoint = crdt.current_clock(); // After vec2

        assert_eq!(crdt.log_size(), 2);

        // Compact keeps only entries >= checkpoint (which is vec2's timestamp)
        crdt.compact(checkpoint);
        assert_eq!(crdt.log_size(), 1);
    }

    #[test]
    fn test_sync_state() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();

        let state = crdt.sync_state();
        assert_eq!(state.vector_count, 1);
        assert_eq!(state.log_size, 1);
    }

    #[test]
    fn test_delta_for_peer() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);

        let mut crdt1 = VectorCRDT::new(replica1);

        crdt1.add("vec1", &[1.0], HashMap::new()).unwrap();

        // First sync
        let delta = crdt1.delta_for_peer(replica2);
        assert_eq!(delta.len(), 1);

        // Simulate sync complete
        crdt1
            .peer_sync_state
            .insert(replica2, crdt1.current_clock());

        crdt1.add("vec2", &[2.0], HashMap::new()).unwrap();

        // Second sync should only have new ops
        let delta2 = crdt1.delta_for_peer(replica2);
        assert_eq!(delta2.len(), 1);
    }

    #[test]
    fn test_list_vectors() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();
        crdt.add("vec2", &[2.0], HashMap::new()).unwrap();
        crdt.add("vec3", &[3.0], HashMap::new()).unwrap();
        crdt.delete("vec2").unwrap();

        let list = crdt.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_skip_own_operations() {
        let replica = ReplicaId::from(1);
        let mut crdt = VectorCRDT::new(replica);

        crdt.add("vec1", &[1.0], HashMap::new()).unwrap();

        let delta = crdt.delta_since(None);
        let result = crdt.merge(delta).unwrap();

        assert_eq!(result.applied, 0);
        assert_eq!(result.skipped, 1);
    }

    #[test]
    fn test_bandwidth_limited_sync() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);
        let mut crdt = VectorCRDT::new(replica1);

        // Add several vectors
        for i in 0..10 {
            crdt.add(&format!("vec{i}"), &[i as f32; 4], HashMap::new())
                .unwrap();
        }

        let config = SyncConfig {
            max_ops_per_batch: 3,
            ..Default::default()
        };

        let batch1 = crdt.delta_batched(replica2, &config, None);
        assert_eq!(batch1.operations.len(), 3);
        assert!(batch1.has_more);
        assert!(batch1.resume_token.is_some());

        // Continue from where we left off
        let batch2 = crdt.delta_batched(replica2, &config, batch1.resume_token);
        assert!(!batch2.operations.is_empty());
    }

    #[test]
    fn test_selective_sync() {
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);
        let mut crdt = VectorCRDT::new(replica1);

        crdt.add("docs_1", &[1.0], HashMap::new()).unwrap();
        crdt.add("docs_2", &[2.0], HashMap::new()).unwrap();
        crdt.add("images_1", &[3.0], HashMap::new()).unwrap();

        let config = SyncConfig {
            id_prefix_filter: Some("docs_".into()),
            ..Default::default()
        };

        let batch = crdt.delta_batched(replica2, &config, None);
        assert_eq!(batch.operations.len(), 2);
    }

    #[test]
    fn test_peer_registry() {
        let mut registry = PeerRegistry::new(60);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        registry.upsert(PeerInfo {
            replica_id: ReplicaId::from(1),
            name: "peer-1".into(),
            address: "192.168.1.1:8080".into(),
            last_seen: now,
            vector_count: 100,
        });

        registry.upsert(PeerInfo {
            replica_id: ReplicaId::from(2),
            name: "peer-2".into(),
            address: "192.168.1.2:8080".into(),
            last_seen: now - 120, // stale
            vector_count: 200,
        });

        assert_eq!(registry.len(), 2);
        assert_eq!(registry.active_peers().len(), 1);

        let pruned = registry.prune_stale();
        assert_eq!(pruned, 1);
        assert_eq!(registry.len(), 1);
    }
}
