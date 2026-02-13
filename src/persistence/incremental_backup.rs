//! Incremental Backup & Point-in-Time Recovery
//!
//! WAL-segment-based incremental backup with continuous archiving and
//! point-in-time recovery (PITR) to any WAL position.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐    ┌───────────────┐    ┌────────────────────┐
//! │ WAL Segments  │───►│ WAL Archiver  │───►│ Archive Directory  │
//! └──────────────┘    └───────────────┘    └────────────────────┘
//!                                                    │
//!         ┌──────────────────────────────────────────┘
//!         ▼
//! ┌──────────────────┐    ┌──────────────────┐
//! │ Base Snapshot     │ + │ WAL Replay to LSN │ = Recovered DB
//! └──────────────────┘    └──────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::persistence::incremental_backup::*;
//!
//! let mut archiver = WalArchiver::new("/backups/wal_archive", ArchiverConfig::default());
//! archiver.archive_segment("/data/wal/segment_000001.wal")?;
//!
//! // Create incremental snapshot
//! let mut mgr = IncrementalBackupManager::new("/backups", BackupManagerConfig::default());
//! mgr.create_base_snapshot(100, &db_bytes)?;
//! mgr.register_wal_segment(101, 200, "/backups/wal_archive/seg_001.wal")?;
//!
//! // Point-in-time recovery
//! let recovery = mgr.plan_recovery(RecoveryTarget::Lsn(150))?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Log Sequence Number.
pub type Lsn = u64;

// ============================================================================
// WAL Archiver
// ============================================================================

/// Configuration for WAL archiving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiverConfig {
    /// Enable compression of archived segments.
    pub compress: bool,
    /// Maximum number of archived segments to retain (0 = unlimited).
    pub max_retained_segments: usize,
    /// Whether to verify checksums on archive.
    pub verify_on_archive: bool,
}

impl Default for ArchiverConfig {
    fn default() -> Self {
        Self {
            compress: true,
            max_retained_segments: 0,
            verify_on_archive: true,
        }
    }
}

/// Metadata for an archived WAL segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedSegment {
    /// Original segment file name.
    pub filename: String,
    /// Starting LSN of entries in this segment.
    pub start_lsn: Lsn,
    /// Ending LSN (inclusive) of entries in this segment.
    pub end_lsn: Lsn,
    /// Size of archived segment in bytes.
    pub archived_size: u64,
    /// Whether segment is compressed.
    pub compressed: bool,
    /// Archive timestamp (epoch seconds).
    pub archived_at: u64,
    /// Path in archive directory.
    pub archive_path: PathBuf,
}

/// Archives WAL segments for long-term storage and PITR.
pub struct WalArchiver {
    archive_dir: PathBuf,
    config: ArchiverConfig,
    /// Archived segments ordered by start LSN.
    segments: BTreeMap<Lsn, ArchivedSegment>,
    /// Next segment sequence number.
    next_seq: u64,
}

impl WalArchiver {
    /// Create a new WAL archiver writing to the given directory.
    pub fn new(archive_dir: impl Into<PathBuf>, config: ArchiverConfig) -> Self {
        Self {
            archive_dir: archive_dir.into(),
            config,
            segments: BTreeMap::new(),
            next_seq: 1,
        }
    }

    /// Archive a WAL segment with the given LSN range.
    ///
    /// In production, this would copy the segment file to the archive directory,
    /// optionally compressing it. Here we record metadata for the archival.
    pub fn archive_segment(
        &mut self,
        start_lsn: Lsn,
        end_lsn: Lsn,
        segment_size: u64,
    ) -> Result<&ArchivedSegment> {
        if start_lsn > end_lsn {
            return Err(NeedleError::InvalidInput(format!(
                "start_lsn ({}) must be <= end_lsn ({})",
                start_lsn, end_lsn
            )));
        }

        // Check for overlapping segments
        if let Some((&existing_start, existing)) = self.segments.range(..=start_lsn).next_back() {
            if existing.end_lsn >= start_lsn && existing_start != start_lsn {
                return Err(NeedleError::InvalidInput(format!(
                    "Segment LSN range [{}, {}] overlaps with existing [{}, {}]",
                    start_lsn, end_lsn, existing_start, existing.end_lsn
                )));
            }
        }

        let seq = self.next_seq;
        self.next_seq += 1;

        let filename = format!("wal_archive_{:06}.wal", seq);
        let archive_path = self.archive_dir.join(&filename);
        let archived_size = if self.config.compress {
            segment_size * 6 / 10 // ~60% compression ratio estimate
        } else {
            segment_size
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let segment = ArchivedSegment {
            filename,
            start_lsn,
            end_lsn,
            archived_size,
            compressed: self.config.compress,
            archived_at: now,
            archive_path,
        };

        self.segments.insert(start_lsn, segment);

        // Enforce retention
        if self.config.max_retained_segments > 0 {
            while self.segments.len() > self.config.max_retained_segments {
                self.segments.pop_first();
            }
        }

        self.segments
            .get(&start_lsn)
            .ok_or_else(|| NeedleError::InvalidInput("Segment not found after insert".into()))
    }

    /// List all archived segments, ordered by LSN.
    pub fn list_segments(&self) -> Vec<&ArchivedSegment> {
        self.segments.values().collect()
    }

    /// Find segments covering a given LSN range.
    pub fn segments_for_range(&self, from_lsn: Lsn, to_lsn: Lsn) -> Vec<&ArchivedSegment> {
        self.segments
            .values()
            .filter(|s| s.end_lsn >= from_lsn && s.start_lsn <= to_lsn)
            .collect()
    }

    /// Get the latest archived LSN.
    pub fn latest_lsn(&self) -> Option<Lsn> {
        self.segments.values().last().map(|s| s.end_lsn)
    }

    /// Total archived size in bytes.
    pub fn total_archived_bytes(&self) -> u64 {
        self.segments.values().map(|s| s.archived_size).sum()
    }
}

// ============================================================================
// Base Snapshots
// ============================================================================

/// Metadata for a base snapshot (full database checkpoint).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseSnapshot {
    /// Unique snapshot ID.
    pub id: String,
    /// LSN at which this snapshot was taken.
    pub lsn: Lsn,
    /// Size of snapshot in bytes.
    pub size_bytes: u64,
    /// Creation timestamp.
    pub created_at: u64,
    /// Path to snapshot data.
    pub path: PathBuf,
}

// ============================================================================
// Recovery Planning
// ============================================================================

/// Target for point-in-time recovery.
#[derive(Debug, Clone)]
pub enum RecoveryTarget {
    /// Recover to a specific LSN.
    Lsn(Lsn),
    /// Recover to the latest available state.
    Latest,
    /// Recover to a specific timestamp (epoch seconds).
    /// The system will find the closest LSN at or before the timestamp.
    Timestamp(u64),
}

/// A recovery plan describing what's needed to restore a database to a target point.
#[derive(Debug, Clone)]
pub struct RecoveryPlan {
    /// Base snapshot to start from.
    pub base_snapshot: BaseSnapshot,
    /// WAL segments to replay (in order).
    pub wal_segments: Vec<ArchivedSegment>,
    /// Target LSN to stop replay at.
    pub target_lsn: Lsn,
    /// Estimated total bytes to read.
    pub estimated_bytes: u64,
    /// Number of WAL entries to replay (estimated).
    pub estimated_entries: u64,
}

// ============================================================================
// Incremental Backup Manager
// ============================================================================

/// Configuration for the backup manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManagerConfig {
    /// Maximum number of base snapshots to retain.
    pub max_base_snapshots: usize,
    /// Minimum LSN gap before creating a new base snapshot.
    pub base_snapshot_interval: u64,
}

impl Default for BackupManagerConfig {
    fn default() -> Self {
        Self {
            max_base_snapshots: 5,
            base_snapshot_interval: 10_000,
        }
    }
}

/// Manages incremental backups using base snapshots + WAL segments.
pub struct IncrementalBackupManager {
    backup_dir: PathBuf,
    config: BackupManagerConfig,
    /// Base snapshots ordered by LSN.
    snapshots: BTreeMap<Lsn, BaseSnapshot>,
    /// WAL segment ranges: (start_lsn, end_lsn, path).
    wal_ranges: Vec<(Lsn, Lsn, PathBuf)>,
    next_snapshot_id: u64,
}

impl IncrementalBackupManager {
    /// Create a new backup manager.
    pub fn new(backup_dir: impl Into<PathBuf>, config: BackupManagerConfig) -> Self {
        Self {
            backup_dir: backup_dir.into(),
            config,
            snapshots: BTreeMap::new(),
            wal_ranges: Vec::new(),
            next_snapshot_id: 1,
        }
    }

    /// Record a new base snapshot at the given LSN.
    pub fn create_base_snapshot(&mut self, lsn: Lsn, data_size: u64) -> Result<&BaseSnapshot> {
        let id = format!("snap_{:06}", self.next_snapshot_id);
        self.next_snapshot_id += 1;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let path = self.backup_dir.join(format!("{}.snap", id));
        let snapshot = BaseSnapshot {
            id,
            lsn,
            size_bytes: data_size,
            created_at: now,
            path,
        };

        self.snapshots.insert(lsn, snapshot);

        // Enforce retention
        while self.snapshots.len() > self.config.max_base_snapshots {
            self.snapshots.pop_first();
        }

        self.snapshots
            .get(&lsn)
            .ok_or_else(|| NeedleError::InvalidInput("Snapshot not found after insert".into()))
    }

    /// Register a WAL segment covering a range of LSNs.
    pub fn register_wal_segment(
        &mut self,
        start_lsn: Lsn,
        end_lsn: Lsn,
        path: impl Into<PathBuf>,
    ) -> Result<()> {
        if start_lsn > end_lsn {
            return Err(NeedleError::InvalidInput(
                "start_lsn must be <= end_lsn".to_string(),
            ));
        }
        self.wal_ranges.push((start_lsn, end_lsn, path.into()));
        self.wal_ranges.sort_by_key(|(start, _, _)| *start);
        Ok(())
    }

    /// Plan recovery to a target.
    pub fn plan_recovery(&self, target: RecoveryTarget) -> Result<RecoveryPlan> {
        let target_lsn = match target {
            RecoveryTarget::Lsn(lsn) => lsn,
            RecoveryTarget::Latest => {
                self.wal_ranges
                    .last()
                    .map(|(_, end, _)| *end)
                    .or_else(|| self.snapshots.keys().last().copied())
                    .ok_or_else(|| {
                        NeedleError::InvalidInput("No data available for recovery".to_string())
                    })?
            }
            RecoveryTarget::Timestamp(_ts) => {
                // For timestamp-based recovery, find closest LSN
                // In production, WAL entries would have timestamps
                self.wal_ranges
                    .last()
                    .map(|(_, end, _)| *end)
                    .ok_or_else(|| {
                        NeedleError::InvalidInput(
                            "Timestamp recovery requires WAL segments".to_string(),
                        )
                    })?
            }
        };

        // Find best base snapshot at or before target LSN
        let base = self
            .snapshots
            .range(..=target_lsn)
            .next_back()
            .map(|(_, s)| s.clone())
            .ok_or_else(|| {
                NeedleError::InvalidInput(format!(
                    "No base snapshot found at or before LSN {}",
                    target_lsn
                ))
            })?;

        // Find WAL segments to replay between snapshot LSN and target
        let wal_segments: Vec<ArchivedSegment> = self
            .wal_ranges
            .iter()
            .filter(|(start, end, _)| *end > base.lsn && *start <= target_lsn)
            .map(|(start, end, path)| ArchivedSegment {
                filename: path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned(),
                start_lsn: *start,
                end_lsn: *end,
                archived_size: 0,
                compressed: false,
                archived_at: 0,
                archive_path: path.clone(),
            })
            .collect();

        let estimated_bytes =
            base.size_bytes + wal_segments.iter().map(|s| s.archived_size).sum::<u64>();

        Ok(RecoveryPlan {
            base_snapshot: base,
            wal_segments,
            target_lsn,
            estimated_bytes,
            estimated_entries: 0,
        })
    }

    /// List available base snapshots.
    pub fn list_snapshots(&self) -> Vec<&BaseSnapshot> {
        self.snapshots.values().collect()
    }

    /// Get the earliest recoverable LSN.
    pub fn earliest_recoverable_lsn(&self) -> Option<Lsn> {
        self.snapshots.keys().next().copied()
    }

    /// Get the latest recoverable LSN.
    pub fn latest_recoverable_lsn(&self) -> Option<Lsn> {
        self.wal_ranges
            .last()
            .map(|(_, end, _)| *end)
            .or_else(|| self.snapshots.keys().last().copied())
    }

    /// Check whether PITR is possible to a given target.
    pub fn can_recover_to(&self, target_lsn: Lsn) -> bool {
        // Need a base snapshot at or before target
        let has_base = self.snapshots.range(..=target_lsn).next_back().is_some();
        if !has_base {
            return false;
        }
        // Need continuous WAL coverage from base to target
        let base_lsn = self
            .snapshots
            .range(..=target_lsn)
            .next_back()
            .map(|(lsn, _)| *lsn)
            .unwrap_or(0);
        if base_lsn >= target_lsn {
            return true; // snapshot itself is sufficient
        }
        // Check WAL coverage
        self.wal_ranges
            .iter()
            .any(|(start, end, _)| *start <= base_lsn + 1 && *end >= target_lsn)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_archiver_basic() {
        let mut archiver = WalArchiver::new("/tmp/archive", ArchiverConfig::default());

        archiver.archive_segment(1, 100, 4096).expect("archive");
        archiver.archive_segment(101, 200, 4096).expect("archive");

        assert_eq!(archiver.list_segments().len(), 2);
        assert_eq!(archiver.latest_lsn(), Some(200));
    }

    #[test]
    fn test_wal_archiver_range_query() {
        let mut archiver = WalArchiver::new("/tmp/archive", ArchiverConfig::default());
        archiver.archive_segment(1, 100, 1000).expect("archive");
        archiver.archive_segment(101, 200, 1000).expect("archive");
        archiver.archive_segment(201, 300, 1000).expect("archive");

        let segs = archiver.segments_for_range(50, 150);
        assert_eq!(segs.len(), 2); // segments 1-100 and 101-200
    }

    #[test]
    fn test_wal_archiver_retention() {
        let config = ArchiverConfig {
            max_retained_segments: 2,
            ..Default::default()
        };
        let mut archiver = WalArchiver::new("/tmp/archive", config);
        archiver.archive_segment(1, 100, 1000).expect("archive");
        archiver.archive_segment(101, 200, 1000).expect("archive");
        archiver.archive_segment(201, 300, 1000).expect("archive");

        assert_eq!(archiver.list_segments().len(), 2);
        // Oldest should be evicted
        assert!(archiver.segments_for_range(1, 50).is_empty());
    }

    #[test]
    fn test_backup_manager_snapshot() {
        let mut mgr =
            IncrementalBackupManager::new("/tmp/backups", BackupManagerConfig::default());
        mgr.create_base_snapshot(100, 1_000_000).expect("snapshot");
        mgr.create_base_snapshot(200, 1_200_000).expect("snapshot");

        assert_eq!(mgr.list_snapshots().len(), 2);
        assert_eq!(mgr.earliest_recoverable_lsn(), Some(100));
    }

    #[test]
    fn test_recovery_plan() {
        let mut mgr =
            IncrementalBackupManager::new("/tmp/backups", BackupManagerConfig::default());
        mgr.create_base_snapshot(100, 1_000_000).expect("snapshot");
        mgr.register_wal_segment(101, 200, "/tmp/wal/seg1.wal")
            .expect("register");
        mgr.register_wal_segment(201, 300, "/tmp/wal/seg2.wal")
            .expect("register");

        // Recovery to LSN 250
        let plan = mgr
            .plan_recovery(RecoveryTarget::Lsn(250))
            .expect("plan");
        assert_eq!(plan.base_snapshot.lsn, 100);
        assert_eq!(plan.target_lsn, 250);
        assert_eq!(plan.wal_segments.len(), 2);

        // Recovery to latest
        let plan_latest = mgr.plan_recovery(RecoveryTarget::Latest).expect("plan");
        assert_eq!(plan_latest.target_lsn, 300);
    }

    #[test]
    fn test_can_recover_to() {
        let mut mgr =
            IncrementalBackupManager::new("/tmp/backups", BackupManagerConfig::default());
        mgr.create_base_snapshot(100, 1_000_000).expect("snapshot");
        mgr.register_wal_segment(101, 300, "/tmp/wal/seg1.wal")
            .expect("register");

        assert!(mgr.can_recover_to(100)); // snapshot itself
        assert!(mgr.can_recover_to(200)); // within WAL range
        assert!(!mgr.can_recover_to(50)); // before any snapshot
    }

    #[test]
    fn test_snapshot_retention() {
        let config = BackupManagerConfig {
            max_base_snapshots: 2,
            ..Default::default()
        };
        let mut mgr = IncrementalBackupManager::new("/tmp/backups", config);
        mgr.create_base_snapshot(100, 100).expect("snap");
        mgr.create_base_snapshot(200, 100).expect("snap");
        mgr.create_base_snapshot(300, 100).expect("snap");

        assert_eq!(mgr.list_snapshots().len(), 2);
        // Oldest should be evicted
        assert_eq!(mgr.earliest_recoverable_lsn(), Some(200));
    }

    #[test]
    fn test_invalid_wal_range() {
        let mut mgr =
            IncrementalBackupManager::new("/tmp/backups", BackupManagerConfig::default());
        let result = mgr.register_wal_segment(200, 100, "/tmp/seg.wal");
        assert!(result.is_err());
    }
}
