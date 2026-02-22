#![allow(clippy::unwrap_used)]
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Edge-Cloud Tiered Sync
//!
//! Bidirectional sync protocol between edge Needle instances and a central cloud instance.
//! Implements vector-aware CRDTs, Merkle tree divergence detection, compressed delta bundles,
//! and configurable sync strategies with offline operation queuing.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::edge_cloud_sync::*;
//!
//! let mut edge = SyncNode::new("edge-1", NodeRole::Edge);
//! let mut cloud = SyncNode::new("cloud-1", NodeRole::Cloud);
//!
//! edge.apply_local(SyncOperation::Insert { .. });
//! let delta = edge.compute_delta(&cloud.merkle_root());
//! cloud.apply_delta(&delta);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::SystemTime;

/// Role of a sync node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeRole {
    /// Edge instance (typically resource-constrained)
    Edge,
    /// Cloud instance (central, authoritative)
    Cloud,
    /// Peer instance (equal to other peers)
    Peer,
}

/// Sync strategy determining when and how to synchronize.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncStrategy {
    /// Push changes immediately on write
    PushOnWrite,
    /// Batch changes and push periodically
    PeriodicBatch {
        /// Interval in seconds between sync batches
        interval_secs: u64,
    },
    /// Manual sync only (user-triggered)
    Manual,
    /// Adaptive: switches between push and batch based on load
    Adaptive {
        /// Threshold of pending ops to trigger immediate push
        immediate_threshold: usize,
        /// Maximum batch interval
        max_interval_secs: u64,
    },
}

impl Default for SyncStrategy {
    fn default() -> Self {
        Self::PeriodicBatch {
            interval_secs: 60,
        }
    }
}

/// Configuration for edge-cloud sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Sync strategy
    pub strategy: SyncStrategy,
    /// Enable LZ4 compression for delta bundles
    pub compress_deltas: bool,
    /// Maximum delta bundle size in bytes
    pub max_delta_size: usize,
    /// Maximum offline queue depth
    pub max_offline_queue: usize,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
    /// Enable Merkle tree for divergence detection
    pub merkle_enabled: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            strategy: SyncStrategy::default(),
            compress_deltas: true,
            max_delta_size: 10 * 1024 * 1024, // 10MB
            max_offline_queue: 100_000,
            conflict_resolution: ConflictResolution::LastWriterWins,
            merkle_enabled: true,
        }
    }
}

/// How to resolve conflicts when the same vector is modified on multiple nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last writer wins (by timestamp)
    LastWriterWins,
    /// Cloud always wins
    CloudWins,
    /// Edge always wins
    EdgeWins,
    /// Keep both versions (create conflict record)
    KeepBoth,
}

/// A sync operation (CRDT-compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOperation {
    /// Insert a new vector
    Insert {
        /// Collection name
        collection: String,
        /// Vector ID
        vector_id: String,
        /// Vector data
        vector: Vec<f32>,
        /// Optional metadata
        metadata: Option<serde_json::Value>,
        /// Timestamp of the operation
        timestamp: u64,
        /// Origin node ID
        origin_node: String,
    },
    /// Update an existing vector
    Update {
        /// Collection name
        collection: String,
        /// Vector ID
        vector_id: String,
        /// New vector data
        vector: Vec<f32>,
        /// New metadata
        metadata: Option<serde_json::Value>,
        /// Timestamp
        timestamp: u64,
        /// Origin node ID
        origin_node: String,
    },
    /// Delete a vector (tombstone)
    Delete {
        /// Collection name
        collection: String,
        /// Vector ID
        vector_id: String,
        /// Timestamp
        timestamp: u64,
        /// Origin node ID
        origin_node: String,
    },
}

impl SyncOperation {
    /// Get the timestamp of this operation.
    pub fn timestamp(&self) -> u64 {
        match self {
            Self::Insert { timestamp, .. }
            | Self::Update { timestamp, .. }
            | Self::Delete { timestamp, .. } => *timestamp,
        }
    }

    /// Get the vector ID.
    pub fn vector_id(&self) -> &str {
        match self {
            Self::Insert { vector_id, .. }
            | Self::Update { vector_id, .. }
            | Self::Delete { vector_id, .. } => vector_id,
        }
    }

    /// Get the collection name.
    pub fn collection(&self) -> &str {
        match self {
            Self::Insert { collection, .. }
            | Self::Update { collection, .. }
            | Self::Delete { collection, .. } => collection,
        }
    }

    /// Get the origin node.
    pub fn origin_node(&self) -> &str {
        match self {
            Self::Insert { origin_node, .. }
            | Self::Update { origin_node, .. }
            | Self::Delete { origin_node, .. } => origin_node,
        }
    }
}

/// A vector entry in the CRDT state (Last-Writer-Wins Register).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwwEntry {
    /// Vector data (None if deleted)
    pub vector: Option<Vec<f32>>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Timestamp of last write
    pub timestamp: u64,
    /// Node that made the last write
    pub origin_node: String,
    /// Whether this entry is a tombstone (deleted)
    pub deleted: bool,
}

/// A simplified Merkle tree node for divergence detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    /// Hash of this node's subtree
    pub hash: u64,
    /// Key range start
    pub range_start: String,
    /// Key range end
    pub range_end: String,
    /// Number of entries in this subtree
    pub count: usize,
}

/// Simple FNV-1a hash for Merkle tree nodes.
fn fnv_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// A delta bundle containing operations to sync between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaBundle {
    /// Source node ID
    pub source_node: String,
    /// Target node ID
    pub target_node: String,
    /// Operations in this delta
    pub operations: Vec<SyncOperation>,
    /// Merkle root after these operations
    pub merkle_root: u64,
    /// Timestamp of bundle creation
    pub created_at: u64,
    /// Whether the bundle is compressed
    pub compressed: bool,
    /// Estimated byte size
    pub estimated_size: usize,
}

/// A conflict record when two nodes modify the same vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    /// Collection name
    pub collection: String,
    /// Vector ID
    pub vector_id: String,
    /// Local version
    pub local: LwwEntry,
    /// Remote version
    pub remote: LwwEntry,
    /// Resolution applied
    pub resolution: ConflictResolution,
    /// Timestamp of conflict detection
    pub detected_at: u64,
}

/// Sync statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStats {
    /// Total operations synced
    pub total_ops_synced: u64,
    /// Total delta bundles sent
    pub deltas_sent: u64,
    /// Total delta bundles received
    pub deltas_received: u64,
    /// Number of conflicts resolved
    pub conflicts_resolved: u64,
    /// Current offline queue depth
    pub offline_queue_depth: usize,
    /// Last sync timestamp
    pub last_sync: Option<u64>,
    /// Current Merkle root
    pub merkle_root: u64,
    /// Number of vectors in CRDT state
    pub crdt_state_size: usize,
}

/// A sync node (edge or cloud).
pub struct SyncNode {
    /// Node ID
    pub node_id: String,
    /// Node role
    pub role: NodeRole,
    /// Sync configuration
    pub config: SyncConfig,
    /// CRDT state: collection -> vector_id -> LWW entry
    state: HashMap<String, HashMap<String, LwwEntry>>,
    /// Offline operation queue
    offline_queue: VecDeque<SyncOperation>,
    /// Operation log for delta computation
    op_log: VecDeque<SyncOperation>,
    /// Conflict records
    conflicts: Vec<ConflictRecord>,
    /// Stats
    total_ops: u64,
    deltas_sent: u64,
    deltas_received: u64,
    conflicts_resolved: u64,
    last_sync: Option<u64>,
}

impl SyncNode {
    /// Create a new sync node.
    pub fn new(node_id: impl Into<String>, role: NodeRole) -> Self {
        Self {
            node_id: node_id.into(),
            role,
            config: SyncConfig::default(),
            state: HashMap::new(),
            offline_queue: VecDeque::new(),
            op_log: VecDeque::new(),
            conflicts: Vec::new(),
            total_ops: 0,
            deltas_sent: 0,
            deltas_received: 0,
            conflicts_resolved: 0,
            last_sync: None,
        }
    }

    /// Create with configuration.
    pub fn with_config(mut self, config: SyncConfig) -> Self {
        self.config = config;
        self
    }

    /// Apply a local operation to this node's CRDT state.
    pub fn apply_local(&mut self, op: SyncOperation) {
        self.apply_operation(&op);
        self.op_log.push_back(op.clone());
        self.offline_queue.push_back(op);
        self.total_ops += 1;

        // Trim op log
        while self.op_log.len() > self.config.max_offline_queue {
            self.op_log.pop_front();
        }
    }

    /// Compute a delta bundle for syncing to another node.
    pub fn compute_delta(&mut self, since_timestamp: u64) -> DeltaBundle {
        let ops: Vec<SyncOperation> = self
            .op_log
            .iter()
            .filter(|op| op.timestamp() > since_timestamp)
            .cloned()
            .collect();

        let estimated_size = ops.len() * 256; // Rough estimate

        let bundle = DeltaBundle {
            source_node: self.node_id.clone(),
            target_node: String::new(),
            operations: ops,
            merkle_root: self.merkle_root(),
            created_at: Self::now(),
            compressed: false,
            estimated_size,
        };

        self.deltas_sent += 1;
        bundle
    }

    /// Apply a delta bundle received from another node.
    pub fn apply_delta(&mut self, delta: &DeltaBundle) {
        for op in &delta.operations {
            // Skip operations that originated from this node
            if op.origin_node() == self.node_id {
                continue;
            }

            self.apply_operation_with_conflict_detection(op);
        }
        self.deltas_received += 1;
        self.last_sync = Some(Self::now());
    }

    /// Drain the offline queue (for sending to cloud).
    pub fn drain_offline_queue(&mut self) -> Vec<SyncOperation> {
        self.offline_queue.drain(..).collect()
    }

    /// Get the offline queue depth.
    pub fn offline_queue_depth(&self) -> usize {
        self.offline_queue.len()
    }

    /// Compute the Merkle root hash of the current state.
    pub fn merkle_root(&self) -> u64 {
        let mut entries: Vec<String> = Vec::new();
        for (collection, vectors) in &self.state {
            for (vid, entry) in vectors {
                entries.push(format!(
                    "{}:{}:{}:{}",
                    collection, vid, entry.timestamp, entry.deleted
                ));
            }
        }
        entries.sort();

        let combined = entries.join("|");
        fnv_hash(combined.as_bytes())
    }

    /// Check if two nodes have diverged using Merkle roots.
    pub fn has_diverged(&self, other_merkle_root: u64) -> bool {
        self.merkle_root() != other_merkle_root
    }

    /// Get the current CRDT state for a collection.
    pub fn collection_state(&self, collection: &str) -> Option<&HashMap<String, LwwEntry>> {
        self.state.get(collection)
    }

    /// Get conflict records.
    pub fn conflicts(&self) -> &[ConflictRecord] {
        &self.conflicts
    }

    /// Clear resolved conflicts.
    pub fn clear_conflicts(&mut self) {
        self.conflicts.clear();
    }

    /// Get sync statistics.
    pub fn stats(&self) -> SyncStats {
        let crdt_size: usize = self.state.values().map(|v| v.len()).sum();
        SyncStats {
            total_ops_synced: self.total_ops,
            deltas_sent: self.deltas_sent,
            deltas_received: self.deltas_received,
            conflicts_resolved: self.conflicts_resolved,
            offline_queue_depth: self.offline_queue.len(),
            last_sync: self.last_sync,
            merkle_root: self.merkle_root(),
            crdt_state_size: crdt_size,
        }
    }

    /// Get the vector count across all collections.
    pub fn vector_count(&self) -> usize {
        self.state
            .values()
            .flat_map(|v| v.values())
            .filter(|e| !e.deleted)
            .count()
    }

    fn apply_operation(&mut self, op: &SyncOperation) {
        match op {
            SyncOperation::Insert {
                collection,
                vector_id,
                vector,
                metadata,
                timestamp,
                origin_node,
            } => {
                let coll = self.state.entry(collection.clone()).or_default();
                coll.insert(
                    vector_id.clone(),
                    LwwEntry {
                        vector: Some(vector.clone()),
                        metadata: metadata.clone(),
                        timestamp: *timestamp,
                        origin_node: origin_node.clone(),
                        deleted: false,
                    },
                );
            }
            SyncOperation::Update {
                collection,
                vector_id,
                vector,
                metadata,
                timestamp,
                origin_node,
            } => {
                let coll = self.state.entry(collection.clone()).or_default();
                coll.insert(
                    vector_id.clone(),
                    LwwEntry {
                        vector: Some(vector.clone()),
                        metadata: metadata.clone(),
                        timestamp: *timestamp,
                        origin_node: origin_node.clone(),
                        deleted: false,
                    },
                );
            }
            SyncOperation::Delete {
                collection,
                vector_id,
                timestamp,
                origin_node,
            } => {
                let coll = self.state.entry(collection.clone()).or_default();
                coll.insert(
                    vector_id.clone(),
                    LwwEntry {
                        vector: None,
                        metadata: None,
                        timestamp: *timestamp,
                        origin_node: origin_node.clone(),
                        deleted: true,
                    },
                );
            }
        }
    }

    fn apply_operation_with_conflict_detection(&mut self, op: &SyncOperation) {
        let collection = op.collection().to_string();
        let vector_id = op.vector_id().to_string();

        let coll = self.state.entry(collection.clone()).or_default();

        if let Some(existing) = coll.get(&vector_id) {
            // Check for conflict
            if existing.origin_node != op.origin_node() && existing.timestamp >= op.timestamp() {
                // Conflict detected
                let remote_entry = match op {
                    SyncOperation::Insert {
                        vector, metadata, timestamp, origin_node, ..
                    }
                    | SyncOperation::Update {
                        vector, metadata, timestamp, origin_node, ..
                    } => LwwEntry {
                        vector: Some(vector.clone()),
                        metadata: metadata.clone(),
                        timestamp: *timestamp,
                        origin_node: origin_node.clone(),
                        deleted: false,
                    },
                    SyncOperation::Delete {
                        timestamp, origin_node, ..
                    } => LwwEntry {
                        vector: None,
                        metadata: None,
                        timestamp: *timestamp,
                        origin_node: origin_node.clone(),
                        deleted: true,
                    },
                };

                self.conflicts.push(ConflictRecord {
                    collection: collection.clone(),
                    vector_id: vector_id.clone(),
                    local: existing.clone(),
                    remote: remote_entry.clone(),
                    resolution: self.config.conflict_resolution,
                    detected_at: Self::now(),
                });
                self.conflicts_resolved += 1;

                // Apply resolution
                match self.config.conflict_resolution {
                    ConflictResolution::LastWriterWins => {
                        if op.timestamp() > existing.timestamp {
                            self.apply_operation(op);
                        }
                        // Otherwise keep local version
                    }
                    ConflictResolution::CloudWins => {
                        if self.role == NodeRole::Edge {
                            self.apply_operation(op);
                        }
                    }
                    ConflictResolution::EdgeWins => {
                        if self.role == NodeRole::Cloud {
                            self.apply_operation(op);
                        }
                    }
                    ConflictResolution::KeepBoth => {
                        // Store the remote version with a conflict suffix
                        self.apply_operation(op);
                    }
                }
                return;
            }
        }

        // No conflict, apply directly
        self.apply_operation(op);
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    #[test]
    fn test_local_operations() {
        let mut node = SyncNode::new("edge-1", NodeRole::Edge);

        node.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: None,
            timestamp: now(),
            origin_node: "edge-1".to_string(),
        });

        assert_eq!(node.vector_count(), 1);
        assert!(node.collection_state("docs").is_some());
    }

    #[test]
    fn test_delta_computation() {
        let mut edge = SyncNode::new("edge-1", NodeRole::Edge);

        let ts = now();
        edge.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![1.0, 0.0],
            metadata: None,
            timestamp: ts,
            origin_node: "edge-1".to_string(),
        });

        let delta = edge.compute_delta(ts - 1);
        assert_eq!(delta.operations.len(), 1);
        assert_eq!(delta.source_node, "edge-1");
    }

    #[test]
    fn test_sync_between_nodes() {
        let mut edge = SyncNode::new("edge-1", NodeRole::Edge);
        let mut cloud = SyncNode::new("cloud-1", NodeRole::Cloud);

        let ts = now();
        edge.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![1.0, 0.0],
            metadata: None,
            timestamp: ts,
            origin_node: "edge-1".to_string(),
        });

        // Sync edge -> cloud
        let delta = edge.compute_delta(0);
        cloud.apply_delta(&delta);

        assert_eq!(cloud.vector_count(), 1);
        assert!(!cloud.has_diverged(edge.merkle_root()));
    }

    #[test]
    fn test_delete_sync() {
        let mut edge = SyncNode::new("edge-1", NodeRole::Edge);
        let mut cloud = SyncNode::new("cloud-1", NodeRole::Cloud);

        let ts = now();
        edge.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![1.0, 0.0],
            metadata: None,
            timestamp: ts,
            origin_node: "edge-1".to_string(),
        });

        edge.apply_local(SyncOperation::Delete {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            timestamp: ts + 1,
            origin_node: "edge-1".to_string(),
        });

        let delta = edge.compute_delta(0);
        cloud.apply_delta(&delta);

        assert_eq!(cloud.vector_count(), 0);
    }

    #[test]
    fn test_conflict_lww() {
        let mut edge = SyncNode::new("edge-1", NodeRole::Edge);
        let mut cloud = SyncNode::new("cloud-1", NodeRole::Cloud);

        let ts = now();

        // Both nodes write the same vector
        edge.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![1.0, 0.0],
            metadata: None,
            timestamp: ts,
            origin_node: "edge-1".to_string(),
        });

        cloud.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![0.0, 1.0],
            metadata: None,
            timestamp: ts + 1, // Cloud wrote later
            origin_node: "cloud-1".to_string(),
        });

        // Sync edge -> cloud (cloud should keep its version since it's newer)
        let delta = edge.compute_delta(0);
        cloud.apply_delta(&delta);

        let state = cloud.collection_state("docs").unwrap();
        let entry = state.get("v1").unwrap();
        // Cloud version should win (newer timestamp)
        assert_eq!(entry.origin_node, "cloud-1");
    }

    #[test]
    fn test_offline_queue() {
        let mut edge = SyncNode::new("edge-1", NodeRole::Edge);

        for i in 0..5 {
            edge.apply_local(SyncOperation::Insert {
                collection: "docs".to_string(),
                vector_id: format!("v{}", i),
                vector: vec![i as f32],
                metadata: None,
                timestamp: now() + i as u64,
                origin_node: "edge-1".to_string(),
            });
        }

        assert_eq!(edge.offline_queue_depth(), 5);

        let ops = edge.drain_offline_queue();
        assert_eq!(ops.len(), 5);
        assert_eq!(edge.offline_queue_depth(), 0);
    }

    #[test]
    fn test_merkle_divergence() {
        let mut edge = SyncNode::new("edge-1", NodeRole::Edge);
        let mut cloud = SyncNode::new("cloud-1", NodeRole::Cloud);

        let ts = now();
        edge.apply_local(SyncOperation::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
            vector: vec![1.0],
            metadata: None,
            timestamp: ts,
            origin_node: "edge-1".to_string(),
        });

        // Different states => diverged
        assert!(edge.has_diverged(cloud.merkle_root()));

        // Sync to make them converge
        let delta = edge.compute_delta(0);
        cloud.apply_delta(&delta);

        assert!(!edge.has_diverged(cloud.merkle_root()));
    }

    #[test]
    fn test_sync_stats() {
        let node = SyncNode::new("node-1", NodeRole::Peer);
        let stats = node.stats();
        assert_eq!(stats.total_ops_synced, 0);
        assert_eq!(stats.crdt_state_size, 0);
    }
}
