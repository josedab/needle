//! Incremental Delta Sync
//!
//! Compact binary delta format for pull-based index replication.
//! Each delta contains WAL entries between two LSN positions, enabling
//! replicas to stay synchronized with minimal data transfer.

use crate::error::Result;
use crate::persistence::wal::{Lsn, WalEntry, WalRecord};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

/// A compact delta containing WAL entries between two LSN positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDelta {
    /// LSN of the first entry in this delta.
    pub from_lsn: Lsn,
    /// LSN of the last entry in this delta (inclusive).
    pub to_lsn: Lsn,
    /// Number of entries in this delta.
    pub entry_count: usize,
    /// Compressed size in bytes (0 if uncompressed).
    pub compressed_size: usize,
    /// Uncompressed size in bytes.
    pub raw_size: usize,
    /// Timestamp when this delta was created.
    pub created_at: u64,
    /// The entries in this delta.
    pub entries: Vec<DeltaEntry>,
}

/// A single entry in a sync delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaEntry {
    /// Log sequence number.
    pub lsn: Lsn,
    /// The WAL entry.
    pub entry: WalEntry,
    /// Timestamp of the original write.
    pub timestamp: u64,
}

impl SyncDelta {
    /// Create a new empty delta starting from a given LSN.
    pub fn new(from_lsn: Lsn) -> Self {
        Self {
            from_lsn,
            to_lsn: from_lsn,
            entry_count: 0,
            compressed_size: 0,
            raw_size: 0,
            created_at: now_secs(),
            entries: Vec::new(),
        }
    }

    /// Add an entry to the delta.
    pub fn push(&mut self, lsn: Lsn, entry: WalEntry, timestamp: u64) {
        self.to_lsn = lsn;
        self.entry_count += 1;
        self.entries.push(DeltaEntry {
            lsn,
            entry,
            timestamp,
        });
    }

    /// Build a delta from a slice of WAL records.
    pub fn from_records(records: &[WalRecord]) -> Self {
        let from_lsn = records.first().map_or(0, |r| r.lsn);
        let mut delta = Self::new(from_lsn);
        for record in records {
            delta.push(record.lsn, record.entry.clone(), record.timestamp);
        }
        delta.raw_size = delta
            .entries
            .iter()
            .map(|_e| std::mem::size_of::<DeltaEntry>() + 64)
            .sum();
        delta
    }

    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(data)?)
    }

    /// Check if this delta is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the LSN span of this delta.
    pub fn lsn_span(&self) -> u64 {
        self.to_lsn.saturating_sub(self.from_lsn)
    }
}

/// Manages delta generation and application for replication.
pub struct DeltaSyncManager {
    /// Buffer of recent WAL records for delta generation.
    buffer: VecDeque<WalRecord>,
    /// Maximum records to retain in the buffer.
    max_buffer_size: usize,
    /// Last LSN that was synced.
    last_synced_lsn: Lsn,
    /// Total deltas generated.
    deltas_generated: u64,
    /// Total entries synced.
    entries_synced: u64,
}

impl DeltaSyncManager {
    /// Create a new delta sync manager.
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_buffer_size),
            max_buffer_size,
            last_synced_lsn: 0,
            deltas_generated: 0,
            entries_synced: 0,
        }
    }

    /// Record a WAL record for delta generation.
    pub fn record(&mut self, record: WalRecord) {
        if self.buffer.len() >= self.max_buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(record);
    }

    /// Generate a delta from a given LSN to the latest buffered record.
    /// Returns `None` if no records are available after `from_lsn`.
    pub fn generate_delta(&mut self, from_lsn: Lsn) -> Option<SyncDelta> {
        let records: Vec<_> = self
            .buffer
            .iter()
            .filter(|r| r.lsn > from_lsn)
            .cloned()
            .collect();

        if records.is_empty() {
            return None;
        }

        let delta = SyncDelta::from_records(&records);
        self.deltas_generated += 1;
        self.entries_synced += delta.entry_count as u64;
        self.last_synced_lsn = delta.to_lsn;
        Some(delta)
    }

    /// Check if a full snapshot is needed (gap too large).
    pub fn needs_snapshot(&self, from_lsn: Lsn) -> bool {
        let oldest = self.buffer.front().map_or(0, |r| r.lsn);
        from_lsn < oldest
    }

    /// Get the latest LSN in the buffer.
    pub fn latest_lsn(&self) -> Lsn {
        self.buffer.back().map_or(0, |r| r.lsn)
    }

    /// Get the oldest LSN available for delta generation.
    pub fn oldest_available_lsn(&self) -> Lsn {
        self.buffer.front().map_or(0, |r| r.lsn)
    }

    /// Get sync statistics.
    pub fn stats(&self) -> DeltaSyncStats {
        DeltaSyncStats {
            buffer_size: self.buffer.len(),
            max_buffer_size: self.max_buffer_size,
            last_synced_lsn: self.last_synced_lsn,
            deltas_generated: self.deltas_generated,
            entries_synced: self.entries_synced,
            oldest_lsn: self.oldest_available_lsn(),
            latest_lsn: self.latest_lsn(),
        }
    }
}

/// Statistics for the delta sync manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaSyncStats {
    /// Current buffer size.
    pub buffer_size: usize,
    /// Maximum buffer capacity.
    pub max_buffer_size: usize,
    /// Last LSN that was synced.
    pub last_synced_lsn: Lsn,
    /// Total deltas generated.
    pub deltas_generated: u64,
    /// Total entries synced across all deltas.
    pub entries_synced: u64,
    /// Oldest LSN available in buffer.
    pub oldest_lsn: Lsn,
    /// Latest LSN in buffer.
    pub latest_lsn: Lsn,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Status of a replication replica.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaStatus {
    /// Replica identifier.
    pub replica_id: String,
    /// Last LSN acknowledged by this replica.
    pub last_ack_lsn: Lsn,
    /// Estimated replication lag in entries.
    pub lag_entries: u64,
    /// Last sync timestamp.
    pub last_sync_secs: u64,
}

impl DeltaSyncManager {
    /// Register a replica and get its status.
    pub fn replica_status(&self, replica_id: &str, replica_lsn: Lsn) -> ReplicaStatus {
        let latest = self.latest_lsn();
        let lag = latest.saturating_sub(replica_lsn);
        ReplicaStatus {
            replica_id: replica_id.to_string(),
            last_ack_lsn: replica_lsn,
            lag_entries: lag,
            last_sync_secs: now_secs(),
        }
    }

    /// Generate a delta for a replica, returning the delta and whether a snapshot is needed.
    pub fn sync_for_replica(
        &mut self,
        replica_id: &str,
        replica_lsn: Lsn,
    ) -> SyncResponse {
        if self.needs_snapshot(replica_lsn) {
            return SyncResponse::SnapshotRequired {
                replica_id: replica_id.to_string(),
                replica_lsn,
                oldest_available: self.oldest_available_lsn(),
            };
        }
        match self.generate_delta(replica_lsn) {
            Some(delta) => SyncResponse::Delta(delta),
            None => SyncResponse::UpToDate {
                current_lsn: self.latest_lsn(),
            },
        }
    }
}

/// Response from a sync request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncResponse {
    /// Delta with new entries since the replica's last LSN.
    Delta(SyncDelta),
    /// Replica is up to date.
    UpToDate { current_lsn: Lsn },
    /// Gap is too large; replica needs a full snapshot.
    SnapshotRequired {
        replica_id: String,
        replica_lsn: Lsn,
        oldest_available: Lsn,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(lsn: Lsn) -> WalRecord {
        WalRecord {
            lsn,
            timestamp: now_secs(),
            checksum: 0,
            length: 0,
            entry: WalEntry::Insert {
                collection: "test".to_string(),
                id: format!("v{lsn}"),
                vector: vec![0.1; 4],
                metadata: None,
            },
        }
    }

    #[test]
    fn test_delta_from_records() {
        let records: Vec<_> = (1..=5).map(make_record).collect();
        let delta = SyncDelta::from_records(&records);
        assert_eq!(delta.from_lsn, 1);
        assert_eq!(delta.to_lsn, 5);
        assert_eq!(delta.entry_count, 5);
    }

    #[test]
    fn test_delta_serialization() {
        let records: Vec<_> = (1..=3).map(make_record).collect();
        let delta = SyncDelta::from_records(&records);
        let bytes = delta.to_bytes().expect("serialize");
        let restored = SyncDelta::from_bytes(&bytes).expect("deserialize");
        assert_eq!(restored.from_lsn, delta.from_lsn);
        assert_eq!(restored.to_lsn, delta.to_lsn);
        assert_eq!(restored.entry_count, delta.entry_count);
    }

    #[test]
    fn test_delta_sync_manager_generate() {
        let mut mgr = DeltaSyncManager::new(1000);
        for i in 1..=10 {
            mgr.record(make_record(i));
        }

        let delta = mgr.generate_delta(5).expect("delta");
        assert_eq!(delta.from_lsn, 6);
        assert_eq!(delta.to_lsn, 10);
        assert_eq!(delta.entry_count, 5);
    }

    #[test]
    fn test_delta_sync_manager_no_delta() {
        let mut mgr = DeltaSyncManager::new(1000);
        for i in 1..=5 {
            mgr.record(make_record(i));
        }
        assert!(mgr.generate_delta(5).is_none());
    }

    #[test]
    fn test_delta_sync_manager_needs_snapshot() {
        let mut mgr = DeltaSyncManager::new(5);
        for i in 10..=20 {
            mgr.record(make_record(i));
        }
        assert!(mgr.needs_snapshot(1));
        assert!(!mgr.needs_snapshot(17));
    }

    #[test]
    fn test_delta_sync_stats() {
        let mut mgr = DeltaSyncManager::new(100);
        for i in 1..=10 {
            mgr.record(make_record(i));
        }
        mgr.generate_delta(5);

        let stats = mgr.stats();
        assert_eq!(stats.buffer_size, 10);
        assert_eq!(stats.deltas_generated, 1);
        assert_eq!(stats.entries_synced, 5);
    }

    #[test]
    fn test_sync_for_replica_delta() {
        let mut mgr = DeltaSyncManager::new(1000);
        for i in 1..=10 {
            mgr.record(make_record(i));
        }

        match mgr.sync_for_replica("replica-1", 5) {
            SyncResponse::Delta(delta) => {
                assert_eq!(delta.entry_count, 5);
                assert_eq!(delta.from_lsn, 6);
            }
            other => panic!("Expected Delta, got {:?}", other),
        }
    }

    #[test]
    fn test_sync_for_replica_up_to_date() {
        let mut mgr = DeltaSyncManager::new(1000);
        for i in 1..=5 {
            mgr.record(make_record(i));
        }

        match mgr.sync_for_replica("replica-1", 5) {
            SyncResponse::UpToDate { current_lsn } => {
                assert_eq!(current_lsn, 5);
            }
            other => panic!("Expected UpToDate, got {:?}", other),
        }
    }

    #[test]
    fn test_sync_for_replica_snapshot_required() {
        let mut mgr = DeltaSyncManager::new(5);
        for i in 10..=20 {
            mgr.record(make_record(i));
        }

        match mgr.sync_for_replica("replica-1", 1) {
            SyncResponse::SnapshotRequired { replica_lsn, .. } => {
                assert_eq!(replica_lsn, 1);
            }
            other => panic!("Expected SnapshotRequired, got {:?}", other),
        }
    }

    #[test]
    fn test_replica_status() {
        let mut mgr = DeltaSyncManager::new(100);
        for i in 1..=10 {
            mgr.record(make_record(i));
        }

        let status = mgr.replica_status("r1", 7);
        assert_eq!(status.replica_id, "r1");
        assert_eq!(status.last_ack_lsn, 7);
        assert_eq!(status.lag_entries, 3);
    }
}
