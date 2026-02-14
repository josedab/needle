//! MVCC-based Snapshot Isolation & Transactions for Needle.
//!
//! Provides multi-version concurrency control so that concurrent readers and
//! writers can operate without blocking each other. Only write-write conflicts
//! on the same vector ID cause a transaction to abort.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// TransactionId
// ---------------------------------------------------------------------------

/// Monotonically increasing identifier for a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransactionId(pub u64);

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tx-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// TransactionStatus
// ---------------------------------------------------------------------------

/// Current lifecycle state of a transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    Active,
    Committed,
    Aborted,
}

// ---------------------------------------------------------------------------
// IsolationLevel
// ---------------------------------------------------------------------------

/// Supported isolation levels. Only [`SnapshotIsolation`] is fully implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadCommitted,
    SnapshotIsolation,
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        Self::SnapshotIsolation
    }
}

// ---------------------------------------------------------------------------
// WriteOperation
// ---------------------------------------------------------------------------

/// A single buffered write inside a transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WriteOperation {
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    Update {
        id: String,
        vector: Option<Vec<f32>>,
        metadata: Option<serde_json::Value>,
    },
    Delete {
        id: String,
    },
}

impl WriteOperation {
    /// Returns the vector ID this operation targets.
    pub fn target_id(&self) -> &str {
        match self {
            WriteOperation::Insert { id, .. }
            | WriteOperation::Update { id, .. }
            | WriteOperation::Delete { id } => id,
        }
    }
}

// ---------------------------------------------------------------------------
// TransactionLog
// ---------------------------------------------------------------------------

/// Append-only log of write operations keyed by transaction.
#[derive(Debug, Default)]
pub struct TransactionLog {
    entries: HashMap<TransactionId, Vec<WriteOperation>>,
}

impl TransactionLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an operation for the given transaction.
    pub fn append(&mut self, tx_id: TransactionId, op: WriteOperation) {
        self.entries.entry(tx_id).or_default().push(op);
    }

    /// Return the operations recorded for `tx_id`, or an empty slice.
    pub fn get_operations(&self, tx_id: TransactionId) -> &[WriteOperation] {
        self.entries
            .get(&tx_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Remove all operations for a transaction.
    pub fn clear(&mut self, tx_id: TransactionId) {
        self.entries.remove(&tx_id);
    }
}

// ---------------------------------------------------------------------------
// SnapshotView
// ---------------------------------------------------------------------------

/// A point-in-time snapshot that determines which committed transactions are
/// visible to a reader.
#[derive(Debug, Clone)]
pub struct SnapshotView {
    pub tx_id: TransactionId,
    pub visible_txns: HashSet<TransactionId>,
}

impl SnapshotView {
    pub fn new(tx_id: TransactionId, visible_txns: HashSet<TransactionId>) -> Self {
        Self {
            tx_id,
            visible_txns,
        }
    }

    /// Returns `true` when `other_tx` was committed before this snapshot was
    /// taken, making its writes visible to us.
    pub fn is_visible(&self, other_tx: TransactionId) -> bool {
        self.visible_txns.contains(&other_tx)
    }
}

// ---------------------------------------------------------------------------
// Transaction
// ---------------------------------------------------------------------------

/// An in-progress transaction that buffers writes and tracks reads.
#[derive(Debug)]
pub struct Transaction {
    pub id: TransactionId,
    pub status: TransactionStatus,
    pub snapshot: SnapshotView,
    pub write_set: Vec<WriteOperation>,
    pub read_set: HashSet<String>,
    pub isolation_level: IsolationLevel,
}

impl Transaction {
    /// Start a new transaction with the supplied snapshot.
    pub fn begin(id: TransactionId, snapshot: SnapshotView) -> Self {
        Self {
            id,
            status: TransactionStatus::Active,
            snapshot,
            write_set: Vec::new(),
            read_set: HashSet::new(),
            isolation_level: IsolationLevel::SnapshotIsolation,
        }
    }

    /// Mark the transaction as committed (local state only — the
    /// [`TransactionManager`] performs conflict detection).
    pub fn commit(&mut self) -> Result<()> {
        match self.status {
            TransactionStatus::Active => {
                self.status = TransactionStatus::Committed;
                Ok(())
            }
            TransactionStatus::Committed => Err(NeedleError::InvalidOperation(format!(
                "Transaction {} is already committed",
                self.id
            ))),
            TransactionStatus::Aborted => Err(NeedleError::InvalidOperation(format!(
                "Cannot commit aborted transaction {}",
                self.id
            ))),
        }
    }

    /// Abort the transaction, discarding all buffered writes.
    pub fn rollback(&mut self) -> Result<()> {
        match self.status {
            TransactionStatus::Active => {
                self.status = TransactionStatus::Aborted;
                self.write_set.clear();
                self.read_set.clear();
                Ok(())
            }
            TransactionStatus::Committed => Err(NeedleError::InvalidOperation(format!(
                "Cannot rollback committed transaction {}",
                self.id
            ))),
            TransactionStatus::Aborted => Err(NeedleError::InvalidOperation(format!(
                "Transaction {} is already aborted",
                self.id
            ))),
        }
    }

    /// Buffer an insert operation.
    pub fn insert(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.ensure_active()?;
        self.write_set.push(WriteOperation::Insert {
            id,
            vector,
            metadata,
        });
        Ok(())
    }

    /// Buffer a delete operation.
    pub fn delete(&mut self, id: String) -> Result<()> {
        self.ensure_active()?;
        self.write_set.push(WriteOperation::Delete { id });
        Ok(())
    }

    /// Buffer an update operation.
    pub fn update(
        &mut self,
        id: String,
        vector: Option<Vec<f32>>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        self.ensure_active()?;
        self.write_set.push(WriteOperation::Update {
            id,
            vector,
            metadata,
        });
        Ok(())
    }

    /// Record that we read a particular vector ID (for conflict tracking).
    pub fn record_read(&mut self, id: String) {
        self.read_set.insert(id);
    }

    /// Collect the set of IDs targeted by all buffered writes.
    pub fn written_ids(&self) -> HashSet<String> {
        self.write_set
            .iter()
            .map(|op| op.target_id().to_owned())
            .collect()
    }

    fn ensure_active(&self) -> Result<()> {
        if self.status != TransactionStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Transaction {} is not active (status: {:?})",
                self.id, self.status
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TransactionStats
// ---------------------------------------------------------------------------

/// Lightweight counters exposed by the manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransactionStats {
    pub active_count: usize,
    pub total_committed: u64,
    pub total_aborted: u64,
    pub total_conflicts: u64,
}

// ---------------------------------------------------------------------------
// TransactionManager
// ---------------------------------------------------------------------------

/// Thread-safe transaction coordinator.
///
/// Manages the lifecycle of transactions, enforces snapshot isolation, and
/// detects write-write conflicts at commit time.
pub struct TransactionManager {
    next_tx_id: AtomicU64,
    inner: RwLock<ManagerInner>,
}

/// State protected by the `RwLock`.
struct ManagerInner {
    active_transactions: HashMap<TransactionId, Transaction>,
    /// Committed transaction log — each entry stores the write operations that
    /// were applied, in commit order.
    committed_log: Vec<(TransactionId, Vec<WriteOperation>)>,
    stats: TransactionStats,
}

impl std::fmt::Debug for TransactionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionManager")
            .field("next_tx_id", &self.next_tx_id.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionManager {
    pub fn new() -> Self {
        Self {
            next_tx_id: AtomicU64::new(1),
            inner: RwLock::new(ManagerInner {
                active_transactions: HashMap::new(),
                committed_log: Vec::new(),
                stats: TransactionStats::default(),
            }),
        }
    }

    /// Begin a new transaction and return its ID.
    ///
    /// The snapshot captures every transaction committed *before* this call.
    pub fn begin(&self) -> Result<TransactionId> {
        let tx_id = TransactionId(self.next_tx_id.fetch_add(1, Ordering::SeqCst));

        let mut inner = self.inner.write();
        let visible: HashSet<TransactionId> =
            inner.committed_log.iter().map(|(id, _)| *id).collect();

        let snapshot = SnapshotView::new(tx_id, visible);
        let txn = Transaction::begin(tx_id, snapshot);
        inner.active_transactions.insert(tx_id, txn);
        inner.stats.active_count = inner.active_transactions.len();

        Ok(tx_id)
    }

    /// Commit a transaction after validating that no write-write conflicts
    /// exist with transactions committed since our snapshot was taken.
    pub fn commit(&self, tx_id: TransactionId) -> Result<()> {
        let mut inner = self.inner.write();

        let txn = inner
            .active_transactions
            .get(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;

        if txn.status != TransactionStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Transaction {tx_id} is not active (status: {:?})",
                txn.status
            )));
        }

        let our_written_ids = txn.written_ids();
        let snapshot = txn.snapshot.clone();

        // Conflict detection: check committed txns that were NOT in our
        // snapshot (i.e. committed after we started).
        let conflict = if our_written_ids.is_empty() {
            None
        } else {
            inner
                .committed_log
                .iter()
                .filter(|(ctx, _)| !snapshot.is_visible(*ctx))
                .find_map(|(ctx, ops)| {
                    ops.iter().find_map(|op| {
                        if our_written_ids.contains(op.target_id()) {
                            Some((*ctx, op.target_id().to_owned()))
                        } else {
                            None
                        }
                    })
                })
        };

        if let Some((conflicting_tx, conflicting_id)) = conflict {
            let txn_mut = inner
                .active_transactions
                .get_mut(&tx_id)
                .expect("just looked up");
            txn_mut.status = TransactionStatus::Aborted;
            txn_mut.write_set.clear();
            inner.active_transactions.remove(&tx_id);
            inner.stats.total_aborted += 1;
            inner.stats.total_conflicts += 1;
            inner.stats.active_count = inner.active_transactions.len();

            return Err(NeedleError::Conflict(format!(
                "Write-write conflict on id '{conflicting_id}' between {tx_id} and {conflicting_tx}",
            )));
        }

        // No conflicts — commit.
        let txn_mut = inner
            .active_transactions
            .get_mut(&tx_id)
            .expect("just looked up");
        txn_mut.status = TransactionStatus::Committed;

        let ops = std::mem::take(&mut txn_mut.write_set);
        inner.committed_log.push((tx_id, ops));
        inner.active_transactions.remove(&tx_id);
        inner.stats.total_committed += 1;
        inner.stats.active_count = inner.active_transactions.len();

        Ok(())
    }

    /// Roll back a transaction, discarding all buffered writes.
    pub fn rollback(&self, tx_id: TransactionId) -> Result<()> {
        let mut inner = self.inner.write();

        let txn = inner
            .active_transactions
            .get_mut(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;

        if txn.status != TransactionStatus::Active {
            return Err(NeedleError::InvalidOperation(format!(
                "Transaction {tx_id} is not active (status: {:?})",
                txn.status
            )));
        }

        txn.status = TransactionStatus::Aborted;
        txn.write_set.clear();
        txn.read_set.clear();
        inner.active_transactions.remove(&tx_id);
        inner.stats.total_aborted += 1;
        inner.stats.active_count = inner.active_transactions.len();

        Ok(())
    }

    /// Access the snapshot for a given active transaction.
    pub fn get_snapshot(&self, tx_id: TransactionId) -> Result<SnapshotView> {
        let inner = self.inner.read();
        let txn = inner
            .active_transactions
            .get(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;
        Ok(txn.snapshot.clone())
    }

    /// Execute a closure with mutable access to the [`Transaction`].
    pub fn with_transaction_mut<F, R>(&self, tx_id: TransactionId, f: F) -> Result<R>
    where
        F: FnOnce(&mut Transaction) -> Result<R>,
    {
        let mut inner = self.inner.write();
        let txn = inner
            .active_transactions
            .get_mut(&tx_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Transaction {tx_id} not found")))?;
        f(txn)
    }

    /// Return current statistics.
    pub fn stats(&self) -> TransactionStats {
        self.inner.read().stats.clone()
    }

    /// Return the committed operations log (for inspection / replay).
    pub fn committed_log(&self) -> Vec<(TransactionId, Vec<WriteOperation>)> {
        self.inner.read().committed_log.clone()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----------------------------------------------------------

    fn sample_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| i as f32 * 0.1).collect()
    }

    // ---- TransactionId ----------------------------------------------------

    #[test]
    fn transaction_id_ordering_and_display() {
        let a = TransactionId(1);
        let b = TransactionId(2);
        assert!(a < b);
        assert_eq!(format!("{a}"), "tx-1");
    }

    // ---- TransactionLog ---------------------------------------------------

    #[test]
    fn transaction_log_append_get_clear() {
        let mut log = TransactionLog::new();
        let tx = TransactionId(1);

        assert!(log.get_operations(tx).is_empty());

        log.append(
            tx,
            WriteOperation::Insert {
                id: "v1".into(),
                vector: vec![1.0],
                metadata: None,
            },
        );
        assert_eq!(log.get_operations(tx).len(), 1);

        log.append(tx, WriteOperation::Delete { id: "v2".into() });
        assert_eq!(log.get_operations(tx).len(), 2);

        log.clear(tx);
        assert!(log.get_operations(tx).is_empty());
    }

    // ---- SnapshotView -----------------------------------------------------

    #[test]
    fn snapshot_visibility() {
        let visible: HashSet<TransactionId> =
            [TransactionId(1), TransactionId(2)].into_iter().collect();
        let snap = SnapshotView::new(TransactionId(3), visible);

        assert!(snap.is_visible(TransactionId(1)));
        assert!(snap.is_visible(TransactionId(2)));
        assert!(!snap.is_visible(TransactionId(3)));
        assert!(!snap.is_visible(TransactionId(99)));
    }

    // ---- Transaction (standalone) -----------------------------------------

    #[test]
    fn transaction_begin_commit_rollback() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);

        assert_eq!(txn.status, TransactionStatus::Active);
        txn.commit().unwrap();
        assert_eq!(txn.status, TransactionStatus::Committed);
    }

    #[test]
    fn transaction_rollback_clears_sets() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);

        txn.insert("a".into(), vec![1.0], None).unwrap();
        txn.record_read("b".into());
        assert!(!txn.write_set.is_empty());
        assert!(!txn.read_set.is_empty());

        txn.rollback().unwrap();
        assert!(txn.write_set.is_empty());
        assert!(txn.read_set.is_empty());
        assert_eq!(txn.status, TransactionStatus::Aborted);
    }

    #[test]
    fn transaction_double_commit_fails() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);
        txn.commit().unwrap();
        assert!(txn.commit().is_err());
    }

    #[test]
    fn transaction_commit_after_rollback_fails() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);
        txn.rollback().unwrap();
        assert!(txn.commit().is_err());
    }

    #[test]
    fn transaction_rollback_after_commit_fails() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);
        txn.commit().unwrap();
        assert!(txn.rollback().is_err());
    }

    #[test]
    fn transaction_double_rollback_fails() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);
        txn.rollback().unwrap();
        assert!(txn.rollback().is_err());
    }

    #[test]
    fn transaction_write_after_commit_fails() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);
        txn.commit().unwrap();
        assert!(txn.insert("x".into(), vec![1.0], None).is_err());
        assert!(txn.delete("x".into()).is_err());
        assert!(txn.update("x".into(), None, None).is_err());
    }

    #[test]
    fn transaction_written_ids() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        let mut txn = Transaction::begin(TransactionId(1), snap);
        txn.insert("a".into(), vec![1.0], None).unwrap();
        txn.delete("b".into()).unwrap();
        txn.update("c".into(), Some(vec![2.0]), None).unwrap();

        let ids = txn.written_ids();
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
        assert!(ids.contains("c"));
        assert_eq!(ids.len(), 3);
    }

    // ---- TransactionManager -----------------------------------------------

    #[test]
    fn manager_begin_and_commit() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin().unwrap();

        mgr.with_transaction_mut(tx, |txn| txn.insert("v1".into(), sample_vector(4), None))
            .unwrap();

        mgr.commit(tx).unwrap();

        let log = mgr.committed_log();
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].0, tx);
        assert_eq!(log[0].1.len(), 1);
    }

    #[test]
    fn manager_begin_and_rollback() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin().unwrap();

        mgr.with_transaction_mut(tx, |txn| txn.insert("v1".into(), sample_vector(4), None))
            .unwrap();

        mgr.rollback(tx).unwrap();

        assert!(mgr.committed_log().is_empty());
        let stats = mgr.stats();
        assert_eq!(stats.total_aborted, 1);
        assert_eq!(stats.active_count, 0);
    }

    #[test]
    fn manager_write_write_conflict_detection() {
        let mgr = TransactionManager::new();

        // tx1 and tx2 start concurrently
        let tx1 = mgr.begin().unwrap();
        let tx2 = mgr.begin().unwrap();

        // Both write to the same ID
        mgr.with_transaction_mut(tx1, |txn| txn.insert("shared".into(), vec![1.0], None))
            .unwrap();
        mgr.with_transaction_mut(tx2, |txn| txn.insert("shared".into(), vec![2.0], None))
            .unwrap();

        // tx1 commits first — succeeds
        mgr.commit(tx1).unwrap();

        // tx2 tries to commit — conflict
        let result = mgr.commit(tx2);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, NeedleError::Conflict(_)),
            "expected Conflict, got: {err:?}"
        );

        let stats = mgr.stats();
        assert_eq!(stats.total_committed, 1);
        assert_eq!(stats.total_aborted, 1);
        assert_eq!(stats.total_conflicts, 1);
    }

    #[test]
    fn manager_no_conflict_on_different_ids() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin().unwrap();
        let tx2 = mgr.begin().unwrap();

        mgr.with_transaction_mut(tx1, |txn| txn.insert("a".into(), vec![1.0], None))
            .unwrap();
        mgr.with_transaction_mut(tx2, |txn| txn.insert("b".into(), vec![2.0], None))
            .unwrap();

        mgr.commit(tx1).unwrap();
        mgr.commit(tx2).unwrap();

        let stats = mgr.stats();
        assert_eq!(stats.total_committed, 2);
        assert_eq!(stats.total_conflicts, 0);
    }

    #[test]
    fn manager_snapshot_isolation_visibility() {
        let mgr = TransactionManager::new();

        // tx1 inserts and commits
        let tx1 = mgr.begin().unwrap();
        mgr.with_transaction_mut(tx1, |txn| txn.insert("v1".into(), vec![1.0], None))
            .unwrap();
        mgr.commit(tx1).unwrap();

        // tx2 starts after tx1 committed — should see tx1
        let tx2 = mgr.begin().unwrap();
        let snap2 = mgr.get_snapshot(tx2).unwrap();
        assert!(snap2.is_visible(tx1));

        // tx3 started before tx1 committed — should NOT see tx1
        // (simulate by checking that tx2's snapshot doesn't include tx2 itself)
        assert!(!snap2.is_visible(tx2));

        mgr.rollback(tx2).unwrap();
    }

    #[test]
    fn manager_uncommitted_txn_not_visible() {
        let mgr = TransactionManager::new();

        // tx1 begins but does NOT commit
        let tx1 = mgr.begin().unwrap();
        mgr.with_transaction_mut(tx1, |txn| txn.insert("v1".into(), vec![1.0], None))
            .unwrap();

        // tx2 starts while tx1 is still active
        let tx2 = mgr.begin().unwrap();
        let snap2 = mgr.get_snapshot(tx2).unwrap();

        // tx1 is not committed → not visible to tx2
        assert!(!snap2.is_visible(tx1));

        mgr.rollback(tx1).unwrap();
        mgr.rollback(tx2).unwrap();
    }

    #[test]
    fn manager_commit_unknown_tx_fails() {
        let mgr = TransactionManager::new();
        let result = mgr.commit(TransactionId(999));
        assert!(result.is_err());
    }

    #[test]
    fn manager_rollback_unknown_tx_fails() {
        let mgr = TransactionManager::new();
        let result = mgr.rollback(TransactionId(999));
        assert!(result.is_err());
    }

    #[test]
    fn manager_double_commit_via_manager_fails() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin().unwrap();
        mgr.commit(tx).unwrap();
        // tx was removed from active set after commit
        assert!(mgr.commit(tx).is_err());
    }

    #[test]
    fn manager_commit_after_rollback_via_manager_fails() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin().unwrap();
        mgr.rollback(tx).unwrap();
        assert!(mgr.commit(tx).is_err());
    }

    #[test]
    fn manager_stats_tracking() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin().unwrap();
        let tx2 = mgr.begin().unwrap();
        let tx3 = mgr.begin().unwrap();

        assert_eq!(mgr.stats().active_count, 3);

        mgr.commit(tx1).unwrap();
        assert_eq!(mgr.stats().active_count, 2);
        assert_eq!(mgr.stats().total_committed, 1);

        mgr.rollback(tx2).unwrap();
        assert_eq!(mgr.stats().active_count, 1);
        assert_eq!(mgr.stats().total_aborted, 1);

        mgr.commit(tx3).unwrap();
        assert_eq!(mgr.stats().active_count, 0);
        assert_eq!(mgr.stats().total_committed, 2);
    }

    #[test]
    fn manager_empty_transaction_commits() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin().unwrap();
        // No writes — should still commit fine
        mgr.commit(tx).unwrap();
        assert_eq!(mgr.stats().total_committed, 1);
    }

    #[test]
    fn manager_multiple_writes_same_txn() {
        let mgr = TransactionManager::new();
        let tx = mgr.begin().unwrap();

        mgr.with_transaction_mut(tx, |txn| {
            txn.insert("a".into(), vec![1.0], None)?;
            txn.update(
                "a".into(),
                Some(vec![2.0]),
                Some(serde_json::json!({"updated": true})),
            )?;
            txn.delete("b".into())?;
            Ok(())
        })
        .unwrap();

        mgr.commit(tx).unwrap();

        let log = mgr.committed_log();
        assert_eq!(log[0].1.len(), 3);
    }

    #[test]
    fn manager_concurrent_readers_dont_conflict() {
        let mgr = TransactionManager::new();

        // Commit some initial data
        let setup = mgr.begin().unwrap();
        mgr.with_transaction_mut(setup, |txn| txn.insert("v1".into(), vec![1.0], None))
            .unwrap();
        mgr.commit(setup).unwrap();

        // Multiple read-only transactions
        let r1 = mgr.begin().unwrap();
        let r2 = mgr.begin().unwrap();

        mgr.with_transaction_mut(r1, |txn| {
            txn.record_read("v1".into());
            Ok(())
        })
        .unwrap();
        mgr.with_transaction_mut(r2, |txn| {
            txn.record_read("v1".into());
            Ok(())
        })
        .unwrap();

        // Both commit without conflict
        mgr.commit(r1).unwrap();
        mgr.commit(r2).unwrap();
        assert_eq!(mgr.stats().total_committed, 3);
        assert_eq!(mgr.stats().total_conflicts, 0);
    }

    #[test]
    fn write_operation_target_id() {
        let insert = WriteOperation::Insert {
            id: "a".into(),
            vector: vec![],
            metadata: None,
        };
        let update = WriteOperation::Update {
            id: "b".into(),
            vector: None,
            metadata: None,
        };
        let delete = WriteOperation::Delete { id: "c".into() };

        assert_eq!(insert.target_id(), "a");
        assert_eq!(update.target_id(), "b");
        assert_eq!(delete.target_id(), "c");
    }

    #[test]
    fn isolation_level_default() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn manager_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(TransactionManager::new());
        let mut handles = Vec::new();

        for i in 0..10 {
            let mgr = Arc::clone(&mgr);
            handles.push(thread::spawn(move || {
                let tx = mgr.begin().unwrap();
                mgr.with_transaction_mut(tx, |txn| {
                    txn.insert(format!("thread-{i}"), vec![i as f32], None)
                })
                .unwrap();
                mgr.commit(tx).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let stats = mgr.stats();
        assert_eq!(stats.total_committed, 10);
        assert_eq!(stats.active_count, 0);
    }

    #[test]
    fn manager_conflict_with_delete_and_update() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin().unwrap();
        let tx2 = mgr.begin().unwrap();

        // tx1 deletes "x", tx2 updates "x"
        mgr.with_transaction_mut(tx1, |txn| txn.delete("x".into()))
            .unwrap();
        mgr.with_transaction_mut(tx2, |txn| txn.update("x".into(), Some(vec![1.0]), None))
            .unwrap();

        mgr.commit(tx1).unwrap();
        assert!(mgr.commit(tx2).is_err());
        assert_eq!(mgr.stats().total_conflicts, 1);
    }

    #[test]
    fn snapshot_view_empty_visible_set() {
        let snap = SnapshotView::new(TransactionId(1), HashSet::new());
        assert!(!snap.is_visible(TransactionId(0)));
        assert!(!snap.is_visible(TransactionId(1)));
    }

    #[test]
    fn transaction_id_hash_and_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TransactionId(1));
        set.insert(TransactionId(1));
        assert_eq!(set.len(), 1);
        set.insert(TransactionId(2));
        assert_eq!(set.len(), 2);
    }
}
