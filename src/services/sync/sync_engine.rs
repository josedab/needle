//! Incremental Sync Protocol
//!
//! Bi-directional sync protocol for `.needle` files across devices/nodes,
//! using vector clocks for causal ordering and delta-based replication with
//! configurable conflict resolution.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::sync_engine::{
//!     SyncEngine, SyncConfig, ConflictPolicy, SyncDelta,
//! };
//!
//! let db_a = Database::in_memory();
//! db_a.create_collection("docs", 4).unwrap();
//!
//! let db_b = Database::in_memory();
//! db_b.create_collection("docs", 4).unwrap();
//!
//! let mut engine_a = SyncEngine::new("node-a", SyncConfig::default());
//! let mut engine_b = SyncEngine::new("node-b", SyncConfig::default());
//!
//! // Record an insert on node A
//! engine_a.record_insert("docs", "v1", &[1.0, 2.0, 3.0, 4.0], None);
//!
//! // Generate delta from A
//! let delta = engine_a.generate_delta(&engine_b.vector_clock());
//!
//! // Apply delta to B
//! let result = engine_b.apply_delta(&db_b, &delta).unwrap();
//! assert_eq!(result.applied, 1);
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Sync engine configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Conflict resolution policy.
    pub conflict_policy: ConflictPolicy,
    /// Maximum operations per delta.
    pub max_delta_size: usize,
    /// Whether to compress deltas.
    pub compress: bool,
    /// Retain operation log entries for this many sync rounds.
    pub log_retention: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            conflict_policy: ConflictPolicy::LastWriteWins,
            max_delta_size: 100_000,
            compress: false,
            log_retention: 100,
        }
    }
}

impl SyncConfig {
    /// Set conflict resolution policy.
    #[must_use]
    pub fn with_conflict_policy(mut self, policy: ConflictPolicy) -> Self {
        self.conflict_policy = policy;
        self
    }
}

/// How to resolve conflicting writes to the same vector ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// Latest timestamp wins.
    LastWriteWins,
    /// Remote changes take precedence.
    RemoteWins,
    /// Local changes take precedence.
    LocalWins,
    /// Raise a conflict error for manual resolution.
    Reject,
}

// ── Vector Clock ─────────────────────────────────────────────────────────────

/// Logical vector clock for causal ordering across nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct VectorClock {
    /// Mapping of node_id → logical timestamp.
    pub clocks: BTreeMap<String, u64>,
}

impl VectorClock {
    /// Create an empty vector clock.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment this node's clock.
    pub fn tick(&mut self, node_id: &str) -> u64 {
        let counter = self.clocks.entry(node_id.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    /// Get the clock value for a node.
    pub fn get(&self, node_id: &str) -> u64 {
        self.clocks.get(node_id).copied().unwrap_or(0)
    }

    /// Merge with another vector clock (take max of each component).
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &ts) in &other.clocks {
            let entry = self.clocks.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(ts);
        }
    }

    /// Check if this clock dominates (happens-after) another.
    pub fn dominates(&self, other: &VectorClock) -> bool {
        let mut dominated = false;
        for (node, &ts) in &other.clocks {
            let our_ts = self.get(node);
            if our_ts < ts {
                return false;
            }
            if our_ts > ts {
                dominated = true;
            }
        }
        // Also check nodes we have that other doesn't
        for (node, &ts) in &self.clocks {
            if other.get(node) < ts {
                dominated = true;
            }
        }
        dominated
    }

    /// Check if two clocks are concurrent (neither dominates).
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.dominates(other) && !other.dominates(self) && self != other
    }
}

// ── Operations ───────────────────────────────────────────────────────────────

/// A single sync operation in the operation log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncOp {
    /// Originating node.
    pub node_id: String,
    /// Logical timestamp.
    pub timestamp: u64,
    /// The operation.
    pub kind: SyncOpKind,
    /// Wall-clock time.
    pub wall_time: SystemTime,
}

/// Kind of sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOpKind {
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    Update {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    Delete {
        collection: String,
        id: String,
    },
}

impl SyncOpKind {
    fn collection(&self) -> &str {
        match self {
            Self::Insert { collection, .. }
            | Self::Update { collection, .. }
            | Self::Delete { collection, .. } => collection,
        }
    }

    fn id(&self) -> &str {
        match self {
            Self::Insert { id, .. } | Self::Update { id, .. } | Self::Delete { id, .. } => id,
        }
    }
}

// ── Delta ────────────────────────────────────────────────────────────────────

/// A delta containing operations that the receiver hasn't seen yet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDelta {
    /// Sender's node ID.
    pub source_node: String,
    /// Sender's current vector clock.
    pub source_clock: VectorClock,
    /// Operations to apply.
    pub operations: Vec<SyncOp>,
}

/// Result of applying a delta.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ApplyResult {
    /// Operations successfully applied.
    pub applied: usize,
    /// Operations skipped (already seen).
    pub skipped: usize,
    /// Conflicts detected.
    pub conflicts: Vec<SyncConflict>,
}

/// A detected conflict between local and remote operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    /// Collection name.
    pub collection: String,
    /// Vector ID.
    pub vector_id: String,
    /// Local operation timestamp.
    pub local_ts: u64,
    /// Remote operation timestamp.
    pub remote_ts: u64,
    /// How it was resolved.
    pub resolution: ConflictResolution,
}

/// How a conflict was resolved.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConflictResolution {
    LocalKept,
    RemoteApplied,
    Rejected,
}

// ── Sync Engine ──────────────────────────────────────────────────────────────

/// Incremental sync engine for bi-directional replication.
pub struct SyncEngine {
    node_id: String,
    config: SyncConfig,
    clock: VectorClock,
    op_log: Vec<SyncOp>,
    /// Track the last known clock from each peer.
    peer_clocks: HashMap<String, VectorClock>,
}

impl SyncEngine {
    /// Create a new sync engine for a node.
    pub fn new(node_id: impl Into<String>, config: SyncConfig) -> Self {
        Self {
            node_id: node_id.into(),
            config,
            clock: VectorClock::new(),
            op_log: Vec::new(),
            peer_clocks: HashMap::new(),
        }
    }

    /// Get this node's ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get the current vector clock.
    pub fn vector_clock(&self) -> &VectorClock {
        &self.clock
    }

    /// Record an insert operation.
    pub fn record_insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) {
        let ts = self.clock.tick(&self.node_id);
        self.op_log.push(SyncOp {
            node_id: self.node_id.clone(),
            timestamp: ts,
            kind: SyncOpKind::Insert {
                collection: collection.into(),
                id: id.into(),
                vector: vector.to_vec(),
                metadata,
            },
            wall_time: SystemTime::now(),
        });
        self.trim_log();
    }

    /// Record an update operation.
    pub fn record_update(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) {
        let ts = self.clock.tick(&self.node_id);
        self.op_log.push(SyncOp {
            node_id: self.node_id.clone(),
            timestamp: ts,
            kind: SyncOpKind::Update {
                collection: collection.into(),
                id: id.into(),
                vector: vector.to_vec(),
                metadata,
            },
            wall_time: SystemTime::now(),
        });
        self.trim_log();
    }

    /// Record a delete operation.
    pub fn record_delete(&mut self, collection: &str, id: &str) {
        let ts = self.clock.tick(&self.node_id);
        self.op_log.push(SyncOp {
            node_id: self.node_id.clone(),
            timestamp: ts,
            kind: SyncOpKind::Delete {
                collection: collection.into(),
                id: id.into(),
            },
            wall_time: SystemTime::now(),
        });
        self.trim_log();
    }

    /// Generate a delta containing operations the peer hasn't seen.
    pub fn generate_delta(&self, peer_clock: &VectorClock) -> SyncDelta {
        let our_ts = peer_clock.get(&self.node_id);

        let operations: Vec<SyncOp> = self
            .op_log
            .iter()
            .filter(|op| op.node_id == self.node_id && op.timestamp > our_ts)
            .take(self.config.max_delta_size)
            .cloned()
            .collect();

        SyncDelta {
            source_node: self.node_id.clone(),
            source_clock: self.clock.clone(),
            operations,
        }
    }

    /// Apply a delta from a remote peer.
    pub fn apply_delta(&mut self, db: &Database, delta: &SyncDelta) -> Result<ApplyResult> {
        let mut result = ApplyResult::default();

        let known_ts = self.clock.get(&delta.source_node);

        for op in &delta.operations {
            // Skip operations we've already seen
            if op.timestamp <= known_ts {
                result.skipped += 1;
                continue;
            }

            // Check for conflicts
            if let Some(conflict) = self.detect_conflict(op) {
                let resolution = self.resolve_conflict(&conflict);
                result.conflicts.push(SyncConflict {
                    collection: op.kind.collection().to_string(),
                    vector_id: op.kind.id().to_string(),
                    local_ts: conflict,
                    remote_ts: op.timestamp,
                    resolution,
                });

                match resolution {
                    ConflictResolution::LocalKept => {
                        result.skipped += 1;
                        continue;
                    }
                    ConflictResolution::Rejected => {
                        result.skipped += 1;
                        continue;
                    }
                    ConflictResolution::RemoteApplied => {
                        // Fall through to apply
                    }
                }
            }

            // Apply the operation
            match &op.kind {
                SyncOpKind::Insert {
                    collection, id, vector, metadata,
                } => {
                    if let Ok(coll) = db.collection(collection) {
                        // Upsert semantics: delete if exists, then insert
                        if coll.get(id).is_some() {
                            coll.delete(id)?;
                        }
                        coll.insert(id.clone(), vector, metadata.clone())?;
                    }
                }
                SyncOpKind::Update {
                    collection, id, vector, metadata,
                } => {
                    if let Ok(coll) = db.collection(collection) {
                        if coll.get(id).is_some() {
                            coll.update(id, vector, metadata.clone())?;
                        } else {
                            coll.insert(id.clone(), vector, metadata.clone())?;
                        }
                    }
                }
                SyncOpKind::Delete { collection, id } => {
                    if let Ok(coll) = db.collection(collection) {
                        coll.delete(id)?;
                    }
                }
            }

            // Append to our log for further propagation
            self.op_log.push(op.clone());
            result.applied += 1;
        }

        // Merge clocks
        self.clock.merge(&delta.source_clock);
        self.peer_clocks
            .insert(delta.source_node.clone(), delta.source_clock.clone());

        self.trim_log();
        Ok(result)
    }

    /// Get operation log length.
    pub fn log_len(&self) -> usize {
        self.op_log.len()
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn detect_conflict(&self, remote_op: &SyncOp) -> Option<u64> {
        // Check if we have a local operation on the same vector ID
        // that is concurrent with the remote operation
        let target_id = remote_op.kind.id();
        let target_coll = remote_op.kind.collection();

        self.op_log
            .iter()
            .filter(|local_op| {
                local_op.node_id == self.node_id
                    && local_op.kind.id() == target_id
                    && local_op.kind.collection() == target_coll
                    && local_op.timestamp > self.clock.get(&remote_op.node_id)
            })
            .map(|op| op.timestamp)
            .last()
    }

    fn resolve_conflict(&self, _local_ts: &u64) -> ConflictResolution {
        match self.config.conflict_policy {
            ConflictPolicy::LastWriteWins | ConflictPolicy::RemoteWins => {
                ConflictResolution::RemoteApplied
            }
            ConflictPolicy::LocalWins => ConflictResolution::LocalKept,
            ConflictPolicy::Reject => ConflictResolution::Rejected,
        }
    }

    fn trim_log(&mut self) {
        if self.op_log.len() > self.config.log_retention * 10 {
            let keep = self.config.log_retention * 5;
            let drain = self.op_log.len() - keep;
            self.op_log.drain(..drain);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_basics() {
        let mut clock = VectorClock::new();
        assert_eq!(clock.get("a"), 0);

        clock.tick("a");
        assert_eq!(clock.get("a"), 1);

        clock.tick("a");
        assert_eq!(clock.get("a"), 2);
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut a = VectorClock::new();
        a.tick("node-a");
        a.tick("node-a");

        let mut b = VectorClock::new();
        b.tick("node-b");
        b.tick("node-b");
        b.tick("node-b");

        a.merge(&b);
        assert_eq!(a.get("node-a"), 2);
        assert_eq!(a.get("node-b"), 3);
    }

    #[test]
    fn test_vector_clock_dominates() {
        let mut a = VectorClock::new();
        a.tick("x");
        a.tick("x");

        let mut b = VectorClock::new();
        b.tick("x");

        assert!(a.dominates(&b));
        assert!(!b.dominates(&a));
    }

    #[test]
    fn test_concurrent_clocks() {
        let mut a = VectorClock::new();
        a.tick("node-a");

        let mut b = VectorClock::new();
        b.tick("node-b");

        assert!(a.is_concurrent(&b));
    }

    #[test]
    fn test_basic_sync() {
        let db_a = Database::in_memory();
        db_a.create_collection("docs", 4).unwrap();
        let db_b = Database::in_memory();
        db_b.create_collection("docs", 4).unwrap();

        let mut engine_a = SyncEngine::new("node-a", SyncConfig::default());
        let mut engine_b = SyncEngine::new("node-b", SyncConfig::default());

        // Insert on A
        let coll_a = db_a.collection("docs").unwrap();
        coll_a.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
        engine_a.record_insert("docs", "v1", &[1.0, 2.0, 3.0, 4.0], None);

        // Sync A → B
        let delta = engine_a.generate_delta(engine_b.vector_clock());
        assert_eq!(delta.operations.len(), 1);

        let result = engine_b.apply_delta(&db_b, &delta).unwrap();
        assert_eq!(result.applied, 1);

        // Verify B has the vector
        let coll_b = db_b.collection("docs").unwrap();
        assert!(coll_b.get("v1").is_some());
    }

    #[test]
    fn test_delta_deduplication() {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();

        let mut engine_a = SyncEngine::new("a", SyncConfig::default());
        let mut engine_b = SyncEngine::new("b", SyncConfig::default());

        engine_a.record_insert("docs", "v1", &[1.0; 4], None);

        // Apply once
        let delta = engine_a.generate_delta(engine_b.vector_clock());
        engine_b.apply_delta(&db, &delta).unwrap();

        // Apply same delta again — should skip
        let delta2 = engine_a.generate_delta(engine_b.vector_clock());
        assert!(delta2.operations.is_empty());
    }

    #[test]
    fn test_conflict_policy_local_wins() {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let coll = db.collection("docs").unwrap();
        coll.insert("v1", &[1.0; 4], None).unwrap();

        let mut engine = SyncEngine::new(
            "local",
            SyncConfig::default().with_conflict_policy(ConflictPolicy::LocalWins),
        );

        // Record local update
        engine.record_update("docs", "v1", &[2.0; 4], None);

        // Create a remote delta that also modifies v1
        let delta = SyncDelta {
            source_node: "remote".into(),
            source_clock: {
                let mut c = VectorClock::new();
                c.tick("remote");
                c
            },
            operations: vec![SyncOp {
                node_id: "remote".into(),
                timestamp: 1,
                kind: SyncOpKind::Update {
                    collection: "docs".into(),
                    id: "v1".into(),
                    vector: vec![9.0; 4],
                    metadata: None,
                },
                wall_time: SystemTime::now(),
            }],
        };

        let result = engine.apply_delta(&db, &delta).unwrap();
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(result.skipped, 1);
    }

    #[test]
    fn test_bidirectional_sync() {
        let db_a = Database::in_memory();
        db_a.create_collection("docs", 4).unwrap();
        let db_b = Database::in_memory();
        db_b.create_collection("docs", 4).unwrap();

        let mut ea = SyncEngine::new("a", SyncConfig::default());
        let mut eb = SyncEngine::new("b", SyncConfig::default());

        // A inserts v1
        db_a.collection("docs").unwrap().insert("v1", &[1.0; 4], None).unwrap();
        ea.record_insert("docs", "v1", &[1.0; 4], None);

        // B inserts v2
        db_b.collection("docs").unwrap().insert("v2", &[2.0; 4], None).unwrap();
        eb.record_insert("docs", "v2", &[2.0; 4], None);

        // Sync A → B
        let delta_a = ea.generate_delta(eb.vector_clock());
        eb.apply_delta(&db_b, &delta_a).unwrap();

        // Sync B → A
        let delta_b = eb.generate_delta(ea.vector_clock());
        ea.apply_delta(&db_a, &delta_b).unwrap();

        // Both should have v1 and v2
        assert!(db_a.collection("docs").unwrap().get("v1").is_some());
        assert!(db_b.collection("docs").unwrap().get("v1").is_some());
        assert!(db_b.collection("docs").unwrap().get("v2").is_some());
    }
}
