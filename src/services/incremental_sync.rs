//! Incremental Backup & Sync
//!
//! Delta-based replication protocol enabling master→replica sync, cloud
//! incremental backup, and cross-device synchronization.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::incremental_sync::{
//!     SyncManager, SyncConfig, DeltaLog, SyncStats,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 4).unwrap();
//!
//! let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();
//!
//! // Track changes
//! sync.record_insert("docs", "v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
//! sync.record_insert("docs", "v2", &[5.0, 6.0, 7.0, 8.0], None).unwrap();
//!
//! // Generate a delta since the last checkpoint
//! let delta = sync.generate_delta().unwrap();
//! assert_eq!(delta.operations.len(), 2);
//!
//! // Apply delta to a replica
//! let replica_db = Database::in_memory();
//! replica_db.create_collection("docs", 4).unwrap();
//! let mut replica_sync = SyncManager::new(&replica_db, SyncConfig::default()).unwrap();
//! let applied = replica_sync.apply_delta(&delta).unwrap();
//! assert_eq!(applied, 2);
//! ```

use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Sync configuration.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Maximum operations per delta.
    pub max_delta_ops: usize,
    /// Conflict resolution strategy.
    pub conflict_strategy: ConflictStrategy,
    /// Whether to include vector data in deltas (large!).
    pub include_vectors: bool,
    /// Compression for delta payloads.
    pub compress_deltas: bool,
    /// Maximum delta age before forced full sync (seconds).
    pub max_delta_age_secs: u64,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_delta_ops: 10_000,
            conflict_strategy: ConflictStrategy::LastWriterWins,
            include_vectors: true,
            compress_deltas: false,
            max_delta_age_secs: 86400 * 7, // 7 days
        }
    }
}

/// Conflict resolution strategy for concurrent writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictStrategy {
    /// Last writer wins (by timestamp).
    LastWriterWins,
    /// Source wins (incoming delta takes precedence).
    SourceWins,
    /// Target wins (local data takes precedence).
    TargetWins,
    /// Reject conflicting operations.
    Reject,
}

// ── Delta Operations ─────────────────────────────────────────────────────────

/// A single change operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaOp {
    /// Sequential operation number.
    pub sequence: u64,
    /// Operation type.
    pub op_type: OpType,
    /// Target collection name.
    pub collection: String,
    /// Vector ID.
    pub vector_id: String,
    /// Vector data (if include_vectors is true).
    pub vector: Option<Vec<f32>>,
    /// Metadata.
    pub metadata: Option<Value>,
    /// Timestamp (epoch milliseconds).
    pub timestamp_ms: u64,
    /// SHA-256 hash for integrity verification.
    pub checksum: Option<String>,
}

/// Operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpType {
    /// Insert or update.
    Upsert,
    /// Delete.
    Delete,
    /// Collection created.
    CreateCollection,
    /// Collection dropped.
    DropCollection,
}

/// A delta: a batch of operations representing changes since a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaLog {
    /// Source node identifier.
    pub source_id: String,
    /// Sequence range: [from_sequence, to_sequence].
    pub from_sequence: u64,
    /// Inclusive upper bound.
    pub to_sequence: u64,
    /// The operations in this delta.
    pub operations: Vec<DeltaOp>,
    /// When this delta was generated (epoch ms).
    pub generated_at_ms: u64,
    /// Format version.
    pub version: u32,
}

impl DeltaLog {
    /// Serialized size estimate in bytes.
    pub fn estimated_size_bytes(&self) -> usize {
        self.operations
            .iter()
            .map(|op| {
                op.vector_id.len()
                    + op.collection.len()
                    + op.vector.as_ref().map_or(0, |v| v.len() * 4)
                    + 64 // fixed overhead
            })
            .sum()
    }

    /// Whether this delta is empty.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

// ── Sync Statistics ──────────────────────────────────────────────────────────

/// Statistics for sync operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    /// Total operations recorded.
    pub total_ops_recorded: u64,
    /// Total deltas generated.
    pub deltas_generated: u64,
    /// Total deltas applied.
    pub deltas_applied: u64,
    /// Total operations applied.
    pub ops_applied: u64,
    /// Total conflicts encountered.
    pub conflicts: u64,
    /// Total operations skipped due to conflict resolution.
    pub ops_skipped: u64,
    /// Last sequence number.
    pub last_sequence: u64,
    /// Last checkpoint sequence.
    pub last_checkpoint: u64,
}

/// Conflict record for audit purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRecord {
    /// Vector ID with conflict.
    pub vector_id: String,
    /// Collection name.
    pub collection: String,
    /// Incoming operation.
    pub incoming_op: OpType,
    /// Resolution strategy applied.
    pub resolution: ConflictStrategy,
    /// Timestamp.
    pub timestamp_ms: u64,
}

// ── Sync Manager ─────────────────────────────────────────────────────────────

/// Manages incremental sync between database instances.
pub struct SyncManager<'a> {
    db: &'a Database,
    config: SyncConfig,
    node_id: String,
    /// Append-only operation log.
    op_log: Vec<DeltaOp>,
    /// Current sequence counter.
    sequence: u64,
    /// Last checkpoint sequence.
    checkpoint: u64,
    /// Stats.
    stats: SyncStats,
    /// Conflict log.
    conflicts: Vec<ConflictRecord>,
}

impl<'a> SyncManager<'a> {
    /// Create a new sync manager.
    pub fn new(db: &'a Database, config: SyncConfig) -> Result<Self> {
        let node_id = format!(
            "node-{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                % 0xFFFFFFFF
        );

        Ok(Self {
            db,
            config,
            node_id,
            op_log: Vec::new(),
            sequence: 0,
            checkpoint: 0,
            stats: SyncStats::default(),
            conflicts: Vec::new(),
        })
    }

    /// Create with a specific node ID.
    pub fn with_node_id(
        db: &'a Database,
        config: SyncConfig,
        node_id: impl Into<String>,
    ) -> Result<Self> {
        Ok(Self {
            db,
            config,
            node_id: node_id.into(),
            op_log: Vec::new(),
            sequence: 0,
            checkpoint: 0,
            stats: SyncStats::default(),
            conflicts: Vec::new(),
        })
    }

    /// Record a vector insert/update operation.
    pub fn record_insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<u64> {
        self.sequence += 1;
        let seq = self.sequence;

        let op = DeltaOp {
            sequence: seq,
            op_type: OpType::Upsert,
            collection: collection.to_string(),
            vector_id: id.to_string(),
            vector: if self.config.include_vectors {
                Some(vector.to_vec())
            } else {
                None
            },
            metadata,
            timestamp_ms: now_ms(),
            checksum: None,
        };

        // Also write to the actual database
        let coll = self.db.collection(collection)?;
        coll.insert(id, vector, op.metadata.clone())?;

        self.op_log.push(op);
        self.stats.total_ops_recorded += 1;
        self.stats.last_sequence = seq;

        // Trim log if too large
        if self.op_log.len() > self.config.max_delta_ops * 2 {
            let keep_from = self.op_log.len() - self.config.max_delta_ops;
            self.op_log.drain(0..keep_from);
        }

        Ok(seq)
    }

    /// Record a vector delete operation.
    pub fn record_delete(&mut self, collection: &str, id: &str) -> Result<u64> {
        self.sequence += 1;
        let seq = self.sequence;

        let op = DeltaOp {
            sequence: seq,
            op_type: OpType::Delete,
            collection: collection.to_string(),
            vector_id: id.to_string(),
            vector: None,
            metadata: None,
            timestamp_ms: now_ms(),
            checksum: None,
        };

        let coll = self.db.collection(collection)?;
        let _ = coll.delete(id);

        self.op_log.push(op);
        self.stats.total_ops_recorded += 1;
        self.stats.last_sequence = seq;

        Ok(seq)
    }

    /// Generate a delta with all operations since the last checkpoint.
    pub fn generate_delta(&mut self) -> Result<DeltaLog> {
        let ops: Vec<DeltaOp> = self
            .op_log
            .iter()
            .filter(|op| op.sequence > self.checkpoint)
            .take(self.config.max_delta_ops)
            .cloned()
            .collect();

        let to_seq = ops.last().map(|op| op.sequence).unwrap_or(self.checkpoint);

        let delta = DeltaLog {
            source_id: self.node_id.clone(),
            from_sequence: self.checkpoint,
            to_sequence: to_seq,
            operations: ops,
            generated_at_ms: now_ms(),
            version: 1,
        };

        self.checkpoint = to_seq;
        self.stats.deltas_generated += 1;
        self.stats.last_checkpoint = to_seq;

        Ok(delta)
    }

    /// Apply a delta from another node.
    pub fn apply_delta(&mut self, delta: &DeltaLog) -> Result<usize> {
        let mut applied = 0usize;

        for op in &delta.operations {
            match self.apply_op(op) {
                Ok(true) => applied += 1,
                Ok(false) => {
                    self.stats.ops_skipped += 1;
                }
                Err(e) => {
                    // Log conflict but continue
                    self.conflicts.push(ConflictRecord {
                        vector_id: op.vector_id.clone(),
                        collection: op.collection.clone(),
                        incoming_op: op.op_type,
                        resolution: self.config.conflict_strategy,
                        timestamp_ms: now_ms(),
                    });
                    self.stats.conflicts += 1;

                    if self.config.conflict_strategy == ConflictStrategy::Reject {
                        return Err(e);
                    }
                }
            }
        }

        self.stats.deltas_applied += 1;
        self.stats.ops_applied += applied as u64;

        Ok(applied)
    }

    /// Apply a single operation, returning true if applied.
    fn apply_op(&self, op: &DeltaOp) -> Result<bool> {
        match op.op_type {
            OpType::Upsert => {
                let vector = op.vector.as_ref().ok_or_else(|| {
                    NeedleError::InvalidArgument("delta op missing vector data".into())
                })?;
                let coll = self.db.collection(&op.collection)?;
                coll.insert(&op.vector_id, vector, op.metadata.clone())?;
                Ok(true)
            }
            OpType::Delete => {
                let coll = self.db.collection(&op.collection)?;
                coll.delete(&op.vector_id)?;
                Ok(true)
            }
            OpType::CreateCollection | OpType::DropCollection => {
                // Schema operations are handled at a higher level
                Ok(false)
            }
        }
    }

    /// Get sync statistics.
    pub fn stats(&self) -> &SyncStats {
        &self.stats
    }

    /// Get conflict log.
    pub fn conflicts(&self) -> &[ConflictRecord] {
        &self.conflicts
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get the current sequence number.
    pub fn current_sequence(&self) -> u64 {
        self.sequence
    }

    /// Get the last checkpoint sequence.
    pub fn last_checkpoint(&self) -> u64 {
        self.checkpoint
    }

    /// Number of operations pending (since last checkpoint).
    pub fn pending_ops(&self) -> usize {
        self.op_log
            .iter()
            .filter(|op| op.sequence > self.checkpoint)
            .count()
    }

    /// Reset checkpoint to force full re-sync.
    pub fn reset_checkpoint(&mut self) {
        self.checkpoint = 0;
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_record_and_generate_delta() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();

        sync.record_insert("test", "v1", &[1.0; 4], None).unwrap();
        sync.record_insert("test", "v2", &[2.0; 4], None).unwrap();

        assert_eq!(sync.pending_ops(), 2);

        let delta = sync.generate_delta().unwrap();
        assert_eq!(delta.operations.len(), 2);
        assert_eq!(delta.from_sequence, 0);
        assert_eq!(delta.to_sequence, 2);

        assert_eq!(sync.pending_ops(), 0);
    }

    #[test]
    fn test_apply_delta_to_replica() {
        let source_db = test_db();
        let mut source = SyncManager::new(&source_db, SyncConfig::default()).unwrap();

        source
            .record_insert("test", "v1", &[1.0; 4], Some(serde_json::json!({"a": 1})))
            .unwrap();
        source.record_insert("test", "v2", &[2.0; 4], None).unwrap();
        let delta = source.generate_delta().unwrap();

        // Apply to replica
        let replica_db = test_db();
        let mut replica = SyncManager::new(&replica_db, SyncConfig::default()).unwrap();
        let applied = replica.apply_delta(&delta).unwrap();
        assert_eq!(applied, 2);

        // Verify replica has the data
        let coll = replica_db.collection("test").unwrap();
        let (vec, _) = coll.get("v1").unwrap();
        assert_eq!(vec, vec![1.0; 4]);
    }

    #[test]
    fn test_delete_sync() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();

        sync.record_insert("test", "v1", &[1.0; 4], None).unwrap();
        sync.record_delete("test", "v1").unwrap();

        let delta = sync.generate_delta().unwrap();
        assert_eq!(delta.operations.len(), 2);
        assert_eq!(delta.operations[1].op_type, OpType::Delete);
    }

    #[test]
    fn test_incremental_deltas() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();

        sync.record_insert("test", "v1", &[1.0; 4], None).unwrap();
        let delta1 = sync.generate_delta().unwrap();
        assert_eq!(delta1.operations.len(), 1);

        sync.record_insert("test", "v2", &[2.0; 4], None).unwrap();
        let delta2 = sync.generate_delta().unwrap();
        assert_eq!(delta2.operations.len(), 1);
        assert_eq!(delta2.from_sequence, 1);
    }

    #[test]
    fn test_delta_size_estimate() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();

        sync.record_insert("test", "v1", &[1.0; 4], None).unwrap();
        let delta = sync.generate_delta().unwrap();

        assert!(delta.estimated_size_bytes() > 0);
    }

    #[test]
    fn test_node_id() {
        let db = test_db();
        let sync = SyncManager::with_node_id(&db, SyncConfig::default(), "node-1").unwrap();
        assert_eq!(sync.node_id(), "node-1");
    }

    #[test]
    fn test_empty_delta() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();
        let delta = sync.generate_delta().unwrap();
        assert!(delta.is_empty());
    }

    #[test]
    fn test_stats() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();

        sync.record_insert("test", "v1", &[1.0; 4], None).unwrap();
        sync.record_insert("test", "v2", &[2.0; 4], None).unwrap();
        sync.generate_delta().unwrap();

        let stats = sync.stats();
        assert_eq!(stats.total_ops_recorded, 2);
        assert_eq!(stats.deltas_generated, 1);
        assert_eq!(stats.last_sequence, 2);
    }

    #[test]
    fn test_reset_checkpoint() {
        let db = test_db();
        let mut sync = SyncManager::new(&db, SyncConfig::default()).unwrap();

        sync.record_insert("test", "v1", &[1.0; 4], None).unwrap();
        sync.generate_delta().unwrap();
        assert_eq!(sync.pending_ops(), 0);

        sync.reset_checkpoint();
        assert_eq!(sync.pending_ops(), 1);
    }
}
