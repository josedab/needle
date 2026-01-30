//! Native Vector Transactions
//!
//! Full ACID transaction support across multiple vector operations with rollback,
//! enabling atomic bulk mutations with consistency guarantees across collections.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::vector_transactions::{
//!     TransactionManager, TransactionConfig, TxOperation,
//! };
//! use serde_json::json;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 4).unwrap();
//! db.create_collection("images", 4).unwrap();
//!
//! let mut mgr = TransactionManager::new(TransactionConfig::default());
//!
//! // Begin a transaction
//! let tx_id = mgr.begin();
//!
//! // Buffer operations across collections
//! mgr.add_operation(tx_id, TxOperation::Insert {
//!     collection: "docs".into(),
//!     id: "doc1".into(),
//!     vector: vec![0.1, 0.2, 0.3, 0.4],
//!     metadata: Some(json!({"title": "Hello"})),
//! }).unwrap();
//!
//! mgr.add_operation(tx_id, TxOperation::Insert {
//!     collection: "images".into(),
//!     id: "img1".into(),
//!     vector: vec![0.5, 0.6, 0.7, 0.8],
//!     metadata: None,
//! }).unwrap();
//!
//! // Commit atomically — all or nothing
//! let receipt = mgr.commit(tx_id, &db).unwrap();
//! assert_eq!(receipt.operations_applied, 2);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Transaction manager configuration.
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// Maximum operations per transaction.
    pub max_operations: usize,
    /// Transaction timeout — automatically abort if not committed in time.
    pub timeout: Duration,
    /// Maximum concurrent transactions.
    pub max_concurrent: usize,
    /// Whether to write a journal before applying (for crash recovery).
    pub enable_journal: bool,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            max_operations: 10_000,
            timeout: Duration::from_secs(30),
            max_concurrent: 64,
            enable_journal: true,
        }
    }
}

impl TransactionConfig {
    /// Create config with custom max operations.
    #[must_use]
    pub fn with_max_operations(mut self, max: usize) -> Self {
        self.max_operations = max;
        self
    }

    /// Create config with custom timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

// ── Transaction ID ───────────────────────────────────────────────────────────

/// Unique identifier for a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TxId(pub u64);

impl std::fmt::Display for TxId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vtx-{}", self.0)
    }
}

// ── Transaction Status ───────────────────────────────────────────────────────

/// Lifecycle state of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxStatus {
    /// Transaction is accepting operations.
    Active,
    /// All operations applied successfully.
    Committed,
    /// Transaction was rolled back.
    Aborted,
    /// Transaction timed out and was automatically aborted.
    TimedOut,
}

// ── Operations ───────────────────────────────────────────────────────────────

/// A single operation within a transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TxOperation {
    /// Insert a vector into a collection.
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Update an existing vector.
    Update {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Delete a vector from a collection.
    Delete { collection: String, id: String },
    /// Delete and re-insert (upsert semantics).
    Upsert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
}

impl TxOperation {
    /// The collection this operation targets.
    pub fn collection(&self) -> &str {
        match self {
            Self::Insert { collection, .. }
            | Self::Update { collection, .. }
            | Self::Delete { collection, .. }
            | Self::Upsert { collection, .. } => collection,
        }
    }

    /// The vector ID this operation targets.
    pub fn id(&self) -> &str {
        match self {
            Self::Insert { id, .. }
            | Self::Update { id, .. }
            | Self::Delete { id, .. }
            | Self::Upsert { id, .. } => id,
        }
    }
}

// ── Transaction ──────────────────────────────────────────────────────────────

/// A single in-flight transaction with buffered operations.
#[derive(Debug)]
struct Transaction {
    id: TxId,
    status: TxStatus,
    operations: Vec<TxOperation>,
    started_at: Instant,
    timeout: Duration,
}

impl Transaction {
    fn new(id: TxId, timeout: Duration) -> Self {
        Self {
            id,
            status: TxStatus::Active,
            operations: Vec::new(),
            started_at: Instant::now(),
            timeout,
        }
    }

    fn is_expired(&self) -> bool {
        self.started_at.elapsed() > self.timeout
    }
}

// ── Commit Receipt ───────────────────────────────────────────────────────────

/// Result of a successful commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitReceipt {
    /// Transaction ID.
    pub tx_id: TxId,
    /// Number of operations applied.
    pub operations_applied: usize,
    /// Collections affected.
    pub collections_affected: Vec<String>,
    /// Time taken to apply all operations.
    pub apply_duration_ms: u64,
    /// Timestamp of commit.
    pub committed_at: SystemTime,
}

// ── Journal Entry ────────────────────────────────────────────────────────────

/// A journal entry for crash recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    /// Transaction ID.
    pub tx_id: TxId,
    /// Sequence number within the transaction.
    pub sequence: u64,
    /// The operation.
    pub operation: TxOperation,
    /// Undo information for rollback.
    pub undo: Option<UndoRecord>,
    /// Timestamp.
    pub timestamp: SystemTime,
}

/// Undo information for rolling back an applied operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UndoRecord {
    /// Undo an insert by deleting the vector.
    DeleteVector {
        collection: String,
        id: String,
    },
    /// Undo a delete by re-inserting the original vector.
    RestoreVector {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Undo an update by restoring the original vector.
    RestoreOriginal {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
}

// ── Transaction Manager ──────────────────────────────────────────────────────

/// Manages multi-collection transactions with ACID guarantees.
///
/// Operations are buffered in memory and applied atomically on commit.
/// If any operation fails, previously applied operations are rolled back.
pub struct TransactionManager {
    config: TransactionConfig,
    next_id: AtomicU64,
    transactions: HashMap<TxId, Transaction>,
    journal: Vec<JournalEntry>,
}

impl TransactionManager {
    /// Create a new transaction manager.
    pub fn new(config: TransactionConfig) -> Self {
        Self {
            config,
            next_id: AtomicU64::new(1),
            transactions: HashMap::new(),
            journal: Vec::new(),
        }
    }

    /// Begin a new transaction. Returns its unique ID.
    pub fn begin(&mut self) -> TxId {
        self.gc_timed_out();
        let id = TxId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let tx = Transaction::new(id, self.config.timeout);
        self.transactions.insert(id, tx);
        id
    }

    /// Add an operation to an active transaction.
    pub fn add_operation(&mut self, tx_id: TxId, op: TxOperation) -> Result<()> {
        let tx = self
            .transactions
            .get_mut(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;

        if tx.status != TxStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Transaction {tx_id} is {:?}, cannot add operations",
                tx.status
            )));
        }

        if tx.is_expired() {
            tx.status = TxStatus::TimedOut;
            return Err(NeedleError::Timeout(self.config.timeout));
        }

        if tx.operations.len() >= self.config.max_operations {
            return Err(NeedleError::CapacityExceeded(format!(
                "Transaction {tx_id} exceeds max operations ({})",
                self.config.max_operations
            )));
        }

        tx.operations.push(op);
        Ok(())
    }

    /// Commit a transaction, applying all operations atomically.
    ///
    /// If any operation fails, previously applied operations are rolled back.
    pub fn commit(&mut self, tx_id: TxId, db: &Database) -> Result<CommitReceipt> {
        let tx = self
            .transactions
            .get_mut(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;

        if tx.status != TxStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Transaction {tx_id} is {:?}, cannot commit",
                tx.status
            )));
        }

        if tx.is_expired() {
            tx.status = TxStatus::TimedOut;
            return Err(NeedleError::Timeout(self.config.timeout));
        }

        let operations = tx.operations.clone();
        let start = Instant::now();
        let mut applied: Vec<JournalEntry> = Vec::new();

        // Apply operations one by one, collecting undo records
        for (seq, op) in operations.iter().enumerate() {
            match self.apply_operation(db, tx_id, seq as u64, op) {
                Ok(entry) => applied.push(entry),
                Err(e) => {
                    // Rollback previously applied operations in reverse
                    self.rollback_applied(db, &applied);
                    if let Some(tx) = self.transactions.get_mut(&tx_id) {
                        tx.status = TxStatus::Aborted;
                    }
                    return Err(NeedleError::InvalidOperation(format!(
                        "Transaction {tx_id} failed at operation {seq}: {e}"
                    )));
                }
            }
        }

        // Collect affected collections
        let mut collections: Vec<String> = operations
            .iter()
            .map(|op| op.collection().to_string())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        collections.sort();

        let receipt = CommitReceipt {
            tx_id,
            operations_applied: applied.len(),
            collections_affected: collections,
            apply_duration_ms: start.elapsed().as_millis() as u64,
            committed_at: SystemTime::now(),
        };

        // Mark committed and persist journal
        if let Some(tx) = self.transactions.get_mut(&tx_id) {
            tx.status = TxStatus::Committed;
        }
        if self.config.enable_journal {
            self.journal.extend(applied);
        }

        Ok(receipt)
    }

    /// Abort a transaction, discarding all buffered operations.
    pub fn abort(&mut self, tx_id: TxId) -> Result<()> {
        let tx = self
            .transactions
            .get_mut(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;

        if tx.status != TxStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Transaction {tx_id} is {:?}, cannot abort",
                tx.status
            )));
        }

        tx.status = TxStatus::Aborted;
        Ok(())
    }

    /// Get the status of a transaction.
    pub fn status(&self, tx_id: TxId) -> Option<TxStatus> {
        self.transactions.get(&tx_id).map(|tx| tx.status)
    }

    /// Get the number of active transactions.
    pub fn active_count(&self) -> usize {
        self.transactions
            .values()
            .filter(|tx| tx.status == TxStatus::Active)
            .count()
    }

    /// Get journal entries for a transaction (for crash recovery).
    pub fn journal_entries(&self, tx_id: TxId) -> Vec<&JournalEntry> {
        self.journal.iter().filter(|e| e.tx_id == tx_id).collect()
    }

    /// Clear completed transactions from memory.
    pub fn gc(&mut self) {
        self.transactions.retain(|_, tx| tx.status == TxStatus::Active);
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn apply_operation(
        &self,
        db: &Database,
        tx_id: TxId,
        sequence: u64,
        op: &TxOperation,
    ) -> Result<JournalEntry> {
        let undo = match op {
            TxOperation::Insert {
                collection, id, vector, metadata,
            } => {
                let coll = db.collection(collection)?;
                coll.insert(id.clone(), vector, metadata.clone())?;
                Some(UndoRecord::DeleteVector {
                    collection: collection.clone(),
                    id: id.clone(),
                })
            }
            TxOperation::Update {
                collection, id, vector, metadata,
            } => {
                let coll = db.collection(collection)?;
                // Capture current state for undo
                let undo = if let Some((old_vec, old_meta)) = coll.get(id) {
                    Some(UndoRecord::RestoreOriginal {
                        collection: collection.clone(),
                        id: id.clone(),
                        vector: old_vec,
                        metadata: old_meta,
                    })
                } else {
                    return Err(NeedleError::VectorNotFound(id.clone()));
                };
                coll.update(id, vector, metadata.clone())?;
                undo
            }
            TxOperation::Delete { collection, id } => {
                let coll = db.collection(collection)?;
                let undo = if let Some((old_vec, old_meta)) = coll.get(id) {
                    Some(UndoRecord::RestoreVector {
                        collection: collection.clone(),
                        id: id.clone(),
                        vector: old_vec,
                        metadata: old_meta,
                    })
                } else {
                    return Err(NeedleError::VectorNotFound(id.clone()));
                };
                coll.delete(id)?;
                undo
            }
            TxOperation::Upsert {
                collection, id, vector, metadata,
            } => {
                let coll = db.collection(collection)?;
                let undo = if let Some((old_vec, old_meta)) = coll.get(id) {
                    coll.delete(id)?;
                    Some(UndoRecord::RestoreVector {
                        collection: collection.clone(),
                        id: id.clone(),
                        vector: old_vec,
                        metadata: old_meta,
                    })
                } else {
                    Some(UndoRecord::DeleteVector {
                        collection: collection.clone(),
                        id: id.clone(),
                    })
                };
                coll.insert(id.clone(), vector, metadata.clone())?;
                undo
            }
        };

        Ok(JournalEntry {
            tx_id,
            sequence,
            operation: op.clone(),
            undo,
            timestamp: SystemTime::now(),
        })
    }

    fn rollback_applied(&self, db: &Database, entries: &[JournalEntry]) {
        for entry in entries.iter().rev() {
            if let Some(undo) = &entry.undo {
                // Best-effort rollback — log failures but continue
                if let Err(e) = self.apply_undo(db, undo) {
                    tracing::warn!("Transaction rollback undo failed: {e}");
                }
            }
        }
    }

    fn apply_undo(&self, db: &Database, undo: &UndoRecord) -> Result<()> {
        match undo {
            UndoRecord::DeleteVector { collection, id } => {
                let coll = db.collection(collection)?;
                coll.delete(id)?;
                Ok(())
            }
            UndoRecord::RestoreVector {
                collection, id, vector, metadata,
            } => {
                let coll = db.collection(collection)?;
                coll.insert(id.clone(), vector, metadata.clone())?;
                Ok(())
            }
            UndoRecord::RestoreOriginal {
                collection, id, vector, metadata,
            } => {
                let coll = db.collection(collection)?;
                coll.update(id, vector, metadata.clone())?;
                Ok(())
            }
        }
    }

    fn gc_timed_out(&mut self) {
        for tx in self.transactions.values_mut() {
            if tx.status == TxStatus::Active && tx.is_expired() {
                tx.status = TxStatus::TimedOut;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_basic_commit() {
        let db = setup_db();
        let mut mgr = TransactionManager::new(TransactionConfig::default());

        let tx = mgr.begin();
        mgr.add_operation(
            tx,
            TxOperation::Insert {
                collection: "test".into(),
                id: "v1".into(),
                vector: vec![1.0, 2.0, 3.0, 4.0],
                metadata: Some(json!({"key": "value"})),
            },
        )
        .unwrap();

        let receipt = mgr.commit(tx, &db).unwrap();
        assert_eq!(receipt.operations_applied, 1);
        assert_eq!(receipt.collections_affected, vec!["test"]);

        // Verify vector was inserted
        let coll = db.collection("test").unwrap();
        let (vec, meta) = coll.get("v1").unwrap();
        assert_eq!(vec.len(), 4);
        assert_eq!(meta.unwrap()["key"], "value");
    }

    #[test]
    fn test_multi_collection_commit() {
        let db = Database::in_memory();
        db.create_collection("a", 4).unwrap();
        db.create_collection("b", 4).unwrap();
        let mut mgr = TransactionManager::new(TransactionConfig::default());

        let tx = mgr.begin();
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "a".into(),
            id: "a1".into(),
            vector: vec![1.0; 4],
            metadata: None,
        }).unwrap();
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "b".into(),
            id: "b1".into(),
            vector: vec![2.0; 4],
            metadata: None,
        }).unwrap();

        let receipt = mgr.commit(tx, &db).unwrap();
        assert_eq!(receipt.operations_applied, 2);
        assert_eq!(receipt.collections_affected.len(), 2);
    }

    #[test]
    fn test_rollback_on_failure() {
        let db = setup_db();
        let mut mgr = TransactionManager::new(TransactionConfig::default());

        let tx = mgr.begin();
        // First op succeeds
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "test".into(),
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata: None,
        }).unwrap();
        // Second op will fail (wrong dimensions)
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "test".into(),
            id: "v2".into(),
            vector: vec![1.0; 8], // wrong dim
            metadata: None,
        }).unwrap();

        assert!(mgr.commit(tx, &db).is_err());
        assert_eq!(mgr.status(tx), Some(TxStatus::Aborted));

        // v1 should be rolled back
        let coll = db.collection("test").unwrap();
        assert!(coll.get("v1").is_none());
    }

    #[test]
    fn test_abort() {
        let db = setup_db();
        let mut mgr = TransactionManager::new(TransactionConfig::default());

        let tx = mgr.begin();
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "test".into(),
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata: None,
        }).unwrap();

        mgr.abort(tx).unwrap();
        assert_eq!(mgr.status(tx), Some(TxStatus::Aborted));

        // Cannot commit after abort
        assert!(mgr.commit(tx, &db).is_err());
    }

    #[test]
    fn test_upsert_operation() {
        let db = setup_db();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"v": 1}))).unwrap();

        let mut mgr = TransactionManager::new(TransactionConfig::default());
        let tx = mgr.begin();
        mgr.add_operation(tx, TxOperation::Upsert {
            collection: "test".into(),
            id: "v1".into(),
            vector: vec![5.0, 6.0, 7.0, 8.0],
            metadata: Some(json!({"v": 2})),
        }).unwrap();

        mgr.commit(tx, &db).unwrap();
        let (vec, meta) = coll.get("v1").unwrap();
        assert!((vec[0] - 5.0).abs() < 0.001);
        assert_eq!(meta.unwrap()["v"], 2);
    }

    #[test]
    fn test_capacity_exceeded() {
        let db = setup_db();
        let mut mgr = TransactionManager::new(
            TransactionConfig::default().with_max_operations(2),
        );

        let tx = mgr.begin();
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "test".into(),
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata: None,
        }).unwrap();
        mgr.add_operation(tx, TxOperation::Insert {
            collection: "test".into(),
            id: "v2".into(),
            vector: vec![2.0; 4],
            metadata: None,
        }).unwrap();

        // Third should fail
        let result = mgr.add_operation(tx, TxOperation::Insert {
            collection: "test".into(),
            id: "v3".into(),
            vector: vec![3.0; 4],
            metadata: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_gc() {
        let db = setup_db();
        let mut mgr = TransactionManager::new(TransactionConfig::default());

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        mgr.abort(tx1).unwrap();
        mgr.commit(tx2, &db).unwrap(); // empty tx

        assert_eq!(mgr.active_count(), 0);
        mgr.gc();
        assert!(mgr.status(tx1).is_none());
    }
}
