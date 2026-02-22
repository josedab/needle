#![allow(clippy::unwrap_used)]
//! Copy-on-Write (COW) B-tree backed persistence for incremental HNSW updates.
//!
//! Instead of serialising the entire database state and performing a full-file
//! rewrite on every `save()` (the current [`StorageEngine::atomic_save`] path),
//! this module tracks page-level deltas and flushes only the modified pages.
//!
//! # Design
//!
//! ```text
//! ┌───────────────────────────────────────────────────────┐
//! │                    CowStorage                          │
//! │  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │
//! │  │ PagePool  │  │ MvccMgr    │  │  DeltaFlusher    │  │
//! │  └──────────┘  └────────────┘  └──────────────────┘  │
//! ├───────────────────────────────────────────────────────┤
//! │                  Page File (4 KB pages)                │
//! └───────────────────────────────────────────────────────┘
//! ```
//!
//! ## MVCC
//!
//! Readers acquire a *snapshot* — a frozen view of the page table at a given
//! transaction id.  Writers create new page copies (COW) so readers never see
//! partially-written state.  Old page versions are garbage-collected once the
//! last snapshot referencing them is released.
//!
//! ## Crash Recovery
//!
//! Every page carries a CRC-32 checksum.  On recovery the engine scans the
//! most recent flush batch and verifies each page's checksum before applying
//! it to the in-memory page table.

use crate::error::{NeedleError, Result};
use crate::storage::crc32;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Default page size (4 KB, matching OS page and storage header).
pub const PAGE_SIZE: usize = 4096;

/// Maximum dirty pages before an automatic flush is triggered.
const DEFAULT_AUTO_FLUSH_THRESHOLD: usize = 1024;

// ─── Identifiers ─────────────────────────────────────────────────────────────

/// Unique identifier for a page in the storage file.
pub type PageId = u64;

/// Transaction / snapshot identifier for MVCC.
pub type TxnId = u64;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the COW persistence engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CowConfig {
    /// Flush when dirty page count reaches this value (0 = manual only).
    pub auto_flush_threshold: usize,
    /// Verify page checksums on read.
    pub enable_checksums: bool,
    /// Maximum concurrent MVCC snapshots.
    pub max_snapshots: usize,
    /// Page size in bytes (default 4096).
    pub page_size: usize,
}

impl Default for CowConfig {
    fn default() -> Self {
        Self {
            auto_flush_threshold: DEFAULT_AUTO_FLUSH_THRESHOLD,
            enable_checksums: true,
            max_snapshots: 8,
            page_size: PAGE_SIZE,
        }
    }
}

impl CowConfig {
    /// New config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set auto-flush threshold.
    #[must_use]
    pub fn with_auto_flush(mut self, threshold: usize) -> Self {
        self.auto_flush_threshold = threshold;
        self
    }

    /// Toggle checksum verification.
    #[must_use]
    pub fn with_checksums(mut self, enabled: bool) -> Self {
        self.enable_checksums = enabled;
        self
    }

    /// Set maximum retained snapshots.
    #[must_use]
    pub fn with_max_snapshots(mut self, max: usize) -> Self {
        self.max_snapshots = max;
        self
    }
}

// ─── Page ────────────────────────────────────────────────────────────────────

/// Classification of page contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageType {
    /// Root page holding B-tree metadata.
    Root,
    /// Internal B-tree node with child pointers.
    Internal,
    /// Leaf page containing vector/index data.
    Leaf,
    /// Overflow page for values larger than one page.
    Overflow,
    /// Free page available for reuse.
    Free,
}

/// A single page in the COW B-tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    /// Unique page identifier.
    pub id: PageId,
    /// Transaction that created this version.
    pub created_by: TxnId,
    /// Page classification.
    pub page_type: PageType,
    /// Payload.
    pub data: Vec<u8>,
    /// CRC-32 over `data`.
    pub checksum: u32,
}

impl Page {
    /// Create a new page, computing the checksum automatically.
    pub fn new(id: PageId, txn_id: TxnId, page_type: PageType, data: Vec<u8>) -> Self {
        let checksum = crc32(&data);
        Self { id, created_by: txn_id, page_type, data, checksum }
    }

    /// Verify the CRC-32 checksum.
    pub fn verify(&self) -> bool {
        crc32(&self.data) == self.checksum
    }

    /// Byte length of the payload.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ─── Delta ───────────────────────────────────────────────────────────────────

/// Describes the kind of change captured in a [`Delta`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeltaKind {
    /// Vectors were inserted.
    Insert { collection: String, count: usize },
    /// Vectors were deleted.
    Delete { collection: String, count: usize },
    /// The HNSW index was restructured.
    IndexRebuild { collection: String },
    /// Only metadata changed.
    MetadataUpdate { collection: String },
    /// Multiple deltas merged by compaction.
    Compaction,
}

/// An incremental change produced by a committed transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    /// Transaction that produced this delta.
    pub txn_id: TxnId,
    /// Wall-clock millis since epoch.
    pub timestamp_ms: u64,
    /// Pages written (new COW copies).
    pub written_pages: Vec<PageId>,
    /// Pages freed (old versions).
    pub freed_pages: Vec<PageId>,
    /// Classification.
    pub kind: DeltaKind,
}

// ─── MVCC Snapshot ───────────────────────────────────────────────────────────

/// A frozen read view of the page table.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Snapshot (transaction) id.
    pub id: TxnId,
    /// Creation time (millis).
    pub created_at_ms: u64,
    /// Root page visible to this snapshot.
    pub root_page: PageId,
    /// Active reader count.
    readers: usize,
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/// Aggregate statistics exposed to callers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CowStats {
    /// Pages currently in the table.
    pub total_pages: u64,
    /// Pages modified since last flush.
    pub dirty_pages: u64,
    /// Pages available for reuse.
    pub free_pages: u64,
    /// Total flush batches written.
    pub flushes: u64,
    /// Active MVCC snapshots.
    pub active_snapshots: usize,
    /// Latest committed transaction id.
    pub head_txn: TxnId,
    /// Cumulative bytes written by flushes.
    pub bytes_flushed: u64,
}

// ─── Flush Batch (serialised to disk) ────────────────────────────────────────

/// A serialisable bundle of deltas + page data written during a flush.
#[derive(Debug, Serialize, Deserialize)]
pub struct FlushBatch {
    pub deltas: Vec<Delta>,
    pub pages: Vec<Page>,
}

// ─── Recovery Report ─────────────────────────────────────────────────────────

/// Result of crash recovery.
#[derive(Debug, Clone)]
pub struct RecoveryReport {
    /// Pages successfully loaded.
    pub pages_recovered: usize,
    /// Pages that failed checksum validation (skipped).
    pub pages_corrupted: usize,
    /// Deltas replayed.
    pub deltas_replayed: usize,
}

// ─── COW Storage Engine ──────────────────────────────────────────────────────

/// The copy-on-write storage engine.
///
/// Writers call [`begin_write`] to obtain a [`WriteTxn`], accumulate page
/// changes, then [`WriteTxn::commit`] to atomically apply them.
///
/// Readers call [`snapshot`] for an MVCC-consistent read view, then
/// [`read_page`] with that snapshot id.
pub struct CowStorage {
    config: CowConfig,

    // ── id generators ──
    next_txn: AtomicU64,
    next_page: AtomicU64,

    // ── page table ──
    pages: RwLock<BTreeMap<PageId, Page>>,
    dirty: RwLock<HashMap<PageId, Page>>,
    free_list: RwLock<VecDeque<PageId>>,

    // ── deltas awaiting flush ──
    pending: RwLock<Vec<Delta>>,

    // ── MVCC ──
    snapshots: RwLock<Vec<Snapshot>>,
    /// Old page versions retained for active snapshots.
    /// Key = page id, value = list of (txn_id, page) in ascending order.
    old_versions: RwLock<HashMap<PageId, Vec<(TxnId, Page)>>>,

    // ── stats ──
    stats: RwLock<CowStats>,
}

impl CowStorage {
    /// Create a new in-memory COW storage engine.
    pub fn new(config: CowConfig) -> Self {
        Self {
            config,
            next_txn: AtomicU64::new(1),
            next_page: AtomicU64::new(1),
            pages: RwLock::new(BTreeMap::new()),
            dirty: RwLock::new(HashMap::new()),
            free_list: RwLock::new(VecDeque::new()),
            pending: RwLock::new(Vec::new()),
            snapshots: RwLock::new(Vec::new()),
            old_versions: RwLock::new(HashMap::new()),
            stats: RwLock::new(CowStats::default()),
        }
    }

    // ── Write path ──────────────────────────────────────────────────────────

    /// Begin a new write transaction.
    pub fn begin_write(&self) -> WriteTxn<'_> {
        let txn_id = self.next_txn.fetch_add(1, Ordering::SeqCst);
        WriteTxn {
            txn_id,
            engine: self,
            new_pages: HashMap::new(),
            freed: Vec::new(),
            committed: false,
        }
    }

    /// Allocate a page id, reusing from the free list when possible.
    fn alloc_page_id(&self) -> PageId {
        self.free_list
            .write()
            .pop_front()
            .unwrap_or_else(|| self.next_page.fetch_add(1, Ordering::SeqCst))
    }

    /// Apply a committed transaction's pages into the live table.
    fn apply(
        &self,
        txn_id: TxnId,
        new_pages: HashMap<PageId, Page>,
        freed: &[PageId],
    ) {
        let mut table = self.pages.write();
        let mut dirty = self.dirty.write();
        let mut old = self.old_versions.write();
        let mut fl = self.free_list.write();
        let need_old = !self.snapshots.read().is_empty();

        for (pid, page) in &new_pages {
            // Preserve old version for active snapshots
            if need_old {
                if let Some(prev) = table.get(pid) {
                    old.entry(*pid)
                        .or_default()
                        .push((prev.created_by, prev.clone()));
                }
            }
            table.insert(*pid, page.clone());
            dirty.insert(*pid, page.clone());
        }

        for pid in freed {
            if let Some(prev) = table.remove(pid) {
                if need_old {
                    old.entry(*pid).or_default().push((prev.created_by, prev));
                }
            }
            fl.push_back(*pid);
        }

        // Update stats
        let mut s = self.stats.write();
        s.dirty_pages = dirty.len() as u64;
        s.free_pages = fl.len() as u64;
        s.total_pages = table.len() as u64;
        s.head_txn = txn_id;
    }

    // ── Read path ───────────────────────────────────────────────────────────

    /// Create an MVCC snapshot for consistent reads.
    pub fn snapshot(&self) -> Result<Snapshot> {
        let txn_id = self.next_txn.load(Ordering::SeqCst);
        let ts = now_ms();
        let root = self.pages.read().keys().next().copied().unwrap_or(0);

        let snap = Snapshot { id: txn_id, created_at_ms: ts, root_page: root, readers: 1 };

        let mut snaps = self.snapshots.write();
        // Enforce limit — evict oldest with zero readers
        if snaps.len() >= self.config.max_snapshots {
            if let Some(pos) = snaps.iter().position(|s| s.readers == 0) {
                let old = snaps.remove(pos);
                self.gc_old_versions(old.id);
            }
        }
        if snaps.len() >= self.config.max_snapshots {
            return Err(NeedleError::InvalidOperation(
                "Too many active MVCC snapshots".into(),
            ));
        }
        snaps.push(snap.clone());
        self.stats.write().active_snapshots = snaps.len();
        Ok(snap)
    }

    /// Release a snapshot (decrement readers, GC when zero).
    pub fn release_snapshot(&self, snap_id: TxnId) {
        let mut snaps = self.snapshots.write();
        if let Some(s) = snaps.iter_mut().find(|s| s.id == snap_id) {
            s.readers = s.readers.saturating_sub(1);
            if s.readers == 0 {
                let id = s.id;
                snaps.retain(|s| s.id != id);
                drop(snaps); // release lock before GC
                self.gc_old_versions(id);
                self.stats.write().active_snapshots = self.snapshots.read().len();
                return;
            }
        }
        self.stats.write().active_snapshots = snaps.len();
    }

    /// Read a page, respecting snapshot visibility.
    pub fn read_page(&self, page_id: PageId, snap_id: Option<TxnId>) -> Option<Page> {
        if let Some(sid) = snap_id {
            let old = self.old_versions.read();
            if let Some(versions) = old.get(&page_id) {
                // Return newest version visible to this snapshot
                for (txn, page) in versions.iter().rev() {
                    if *txn <= sid {
                        return Some(page.clone());
                    }
                }
            }
        }
        self.pages.read().get(&page_id).cloned()
    }

    // ── Flush ───────────────────────────────────────────────────────────────

    /// Flush all pending deltas into a serialised [`FlushBatch`].
    ///
    /// Returns the serialised bytes that would be appended to the storage file.
    /// An empty `Vec` means nothing to flush.
    pub fn flush(&self) -> Result<Vec<u8>> {
        let deltas: Vec<Delta> = self.pending.write().drain(..).collect();
        let dirty_pages: Vec<Page> = self.dirty.write().drain().map(|(_, p)| p).collect();

        if deltas.is_empty() && dirty_pages.is_empty() {
            return Ok(Vec::new());
        }

        let batch = FlushBatch { deltas, pages: dirty_pages };
        let data = serde_json::to_vec(&batch).map_err(NeedleError::Serialization)?;

        let mut s = self.stats.write();
        s.flushes += 1;
        s.bytes_flushed += data.len() as u64;
        s.dirty_pages = 0;

        Ok(data)
    }

    /// Whether the dirty-page count has reached the auto-flush threshold.
    pub fn should_auto_flush(&self) -> bool {
        let threshold = self.config.auto_flush_threshold;
        threshold > 0 && self.dirty.read().len() >= threshold
    }

    // ── Recovery ────────────────────────────────────────────────────────────

    /// Recover state from a serialised flush batch, validating checksums.
    pub fn recover(&self, data: &[u8]) -> Result<RecoveryReport> {
        let batch: FlushBatch =
            serde_json::from_slice(data).map_err(NeedleError::Serialization)?;

        let mut report = RecoveryReport {
            pages_recovered: 0,
            pages_corrupted: 0,
            deltas_replayed: batch.deltas.len(),
        };

        let mut table = self.pages.write();
        for page in batch.pages {
            if self.config.enable_checksums && !page.verify() {
                report.pages_corrupted += 1;
                tracing::warn!(page_id = page.id, "Skipping corrupted page during recovery");
                continue;
            }
            table.insert(page.id, page);
            report.pages_recovered += 1;
        }
        self.stats.write().total_pages = table.len() as u64;

        Ok(report)
    }

    // ── Stats ───────────────────────────────────────────────────────────────

    /// Current statistics.
    pub fn stats(&self) -> CowStats {
        self.stats.read().clone()
    }

    // ── Internal ────────────────────────────────────────────────────────────

    fn gc_old_versions(&self, released_id: TxnId) {
        let min_snap = self.snapshots.read().iter().map(|s| s.id).min();
        let cutoff = min_snap.unwrap_or(released_id);
        let mut old = self.old_versions.write();
        old.retain(|_, versions| {
            versions.retain(|(txn, _)| *txn >= cutoff);
            !versions.is_empty()
        });
    }
}

// ─── WriteTxn ────────────────────────────────────────────────────────────────

/// A write transaction that accumulates page changes and commits atomically.
///
/// Dropping without calling [`commit`] is an implicit rollback — no changes
/// are applied to the storage engine.
pub struct WriteTxn<'a> {
    txn_id: TxnId,
    engine: &'a CowStorage,
    new_pages: HashMap<PageId, Page>,
    freed: Vec<PageId>,
    committed: bool,
}

impl<'a> WriteTxn<'a> {
    /// Transaction id.
    pub fn id(&self) -> TxnId { self.txn_id }

    /// Allocate and write a new page, returning its id.
    pub fn write_page(&mut self, page_type: PageType, data: Vec<u8>) -> PageId {
        let pid = self.engine.alloc_page_id();
        let page = Page::new(pid, self.txn_id, page_type, data);
        self.new_pages.insert(pid, page);
        pid
    }

    /// Create a COW copy of an existing page with new data.
    pub fn update_page(&mut self, page_id: PageId, data: Vec<u8>) -> Result<()> {
        let existing = self.engine.read_page(page_id, None)
            .ok_or_else(|| NeedleError::VectorNotFound(format!("Page {page_id}")))?;
        let page = Page::new(page_id, self.txn_id, existing.page_type, data);
        self.new_pages.insert(page_id, page);
        Ok(())
    }

    /// Mark a page as freed.
    pub fn free_page(&mut self, page_id: PageId) {
        self.freed.push(page_id);
    }

    /// Commit all changes atomically.
    pub fn commit(mut self, kind: DeltaKind) -> Result<TxnId> {
        let delta = Delta {
            txn_id: self.txn_id,
            timestamp_ms: now_ms(),
            written_pages: self.new_pages.keys().copied().collect(),
            freed_pages: self.freed.clone(),
            kind,
        };

        self.engine.apply(
            self.txn_id,
            std::mem::take(&mut self.new_pages),
            &self.freed,
        );
        self.engine.pending.write().push(delta);

        if self.engine.should_auto_flush() {
            if let Err(e) = self.engine.flush() {
                tracing::warn!("auto-flush failed: {e}");
            }
        }

        self.committed = true;
        Ok(self.txn_id)
    }
}

impl Drop for WriteTxn<'_> {
    fn drop(&mut self) {
        if !self.committed {
            tracing::debug!(txn = self.txn_id, "WriteTxn dropped without commit (rollback)");
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|e| {
            tracing::warn!("system clock before UNIX epoch: {e}");
            std::time::Duration::default()
        })
        .as_millis() as u64
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> CowStorage {
        CowStorage::new(CowConfig::default())
    }

    #[test]
    fn write_and_read_page() {
        let e = engine();
        let mut tx = e.begin_write();
        let pid = tx.write_page(PageType::Leaf, vec![1, 2, 3, 4]);
        tx.commit(DeltaKind::Insert { collection: "c".into(), count: 1 })
            .expect("commit");

        let page = e.read_page(pid, None).expect("page exists");
        assert_eq!(page.data, vec![1, 2, 3, 4]);
        assert!(page.verify());
    }

    #[test]
    fn snapshot_isolation() {
        let e = engine();

        // Write initial data
        let mut tx1 = e.begin_write();
        let pid = tx1.write_page(PageType::Leaf, vec![10, 20]);
        tx1.commit(DeltaKind::Insert { collection: "c".into(), count: 1 }).unwrap();

        // Take snapshot
        let snap = e.snapshot().unwrap();

        // Overwrite the page
        let mut tx2 = e.begin_write();
        tx2.update_page(pid, vec![30, 40]).unwrap();
        tx2.commit(DeltaKind::MetadataUpdate { collection: "c".into() }).unwrap();

        // Current read sees new data
        assert_eq!(e.read_page(pid, None).unwrap().data, vec![30, 40]);

        // Snapshot read sees old data
        assert_eq!(e.read_page(pid, Some(snap.id)).unwrap().data, vec![10, 20]);

        e.release_snapshot(snap.id);
    }

    #[test]
    fn flush_produces_data() {
        let e = engine();
        let mut tx = e.begin_write();
        tx.write_page(PageType::Leaf, vec![1]);
        tx.commit(DeltaKind::Insert { collection: "c".into(), count: 1 }).unwrap();

        let data = e.flush().unwrap();
        assert!(!data.is_empty());

        // Second flush is a no-op
        assert!(e.flush().unwrap().is_empty());
    }

    #[test]
    fn page_checksum_detects_corruption() {
        let page = Page::new(1, 1, PageType::Leaf, vec![1, 2, 3]);
        assert!(page.verify());

        let mut bad = page.clone();
        bad.data[0] = 99;
        assert!(!bad.verify());
    }

    #[test]
    fn recovery_from_flush_batch() {
        let e1 = engine();
        let mut tx = e1.begin_write();
        tx.write_page(PageType::Leaf, vec![5, 6]);
        tx.write_page(PageType::Leaf, vec![7, 8]);
        tx.commit(DeltaKind::Insert { collection: "c".into(), count: 2 }).unwrap();

        let data = e1.flush().unwrap();

        let e2 = engine();
        let report = e2.recover(&data).unwrap();
        assert_eq!(report.pages_recovered, 2);
        assert_eq!(report.pages_corrupted, 0);
        assert_eq!(report.deltas_replayed, 1);
    }

    #[test]
    fn free_pages_are_reused() {
        let e = engine();

        let mut tx1 = e.begin_write();
        let pid = tx1.write_page(PageType::Leaf, vec![1]);
        tx1.commit(DeltaKind::Insert { collection: "c".into(), count: 1 }).unwrap();

        let mut tx2 = e.begin_write();
        tx2.free_page(pid);
        tx2.commit(DeltaKind::Delete { collection: "c".into(), count: 1 }).unwrap();

        assert_eq!(e.stats().free_pages, 1);

        // Next allocation should reuse the freed page id
        let mut tx3 = e.begin_write();
        let pid2 = tx3.write_page(PageType::Leaf, vec![2]);
        tx3.commit(DeltaKind::Insert { collection: "c".into(), count: 1 }).unwrap();
        assert_eq!(pid, pid2);
    }

    #[test]
    fn rollback_on_drop() {
        let e = engine();
        {
            let mut tx = e.begin_write();
            tx.write_page(PageType::Leaf, vec![1, 2, 3]);
            // dropped without commit
        }
        assert!(e.pages.read().is_empty(), "No pages applied on rollback");
    }

    #[test]
    fn auto_flush_triggers() {
        let e = CowStorage::new(CowConfig::default().with_auto_flush(2));

        let mut tx = e.begin_write();
        tx.write_page(PageType::Leaf, vec![1]);
        tx.write_page(PageType::Leaf, vec![2]);
        // Commit should trigger auto-flush since dirty >= 2
        tx.commit(DeltaKind::Insert { collection: "c".into(), count: 2 }).unwrap();

        // After auto-flush, dirty should be zero
        assert_eq!(e.stats().dirty_pages, 0);
        assert!(e.stats().flushes >= 1);
    }

    #[test]
    fn snapshot_limit_enforced() {
        let e = CowStorage::new(CowConfig::default().with_max_snapshots(2));

        let s1 = e.snapshot().unwrap();
        let s2 = e.snapshot().unwrap();
        // Third should fail because the first two still have readers
        let r = e.snapshot();
        assert!(r.is_err());

        e.release_snapshot(s1.id);
        // Now it should succeed
        let _s3 = e.snapshot().unwrap();
        e.release_snapshot(s2.id);
    }
}
