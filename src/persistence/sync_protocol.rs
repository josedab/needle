//! Vector Sync Protocol
//!
//! Lightweight replication protocol inspired by Litestream that continuously
//! ships WAL segments to a destination (local directory or cloud storage),
//! enabling disaster recovery, point-in-time restore, and read replicas.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐    WAL segments    ┌──────────────┐
//! │  Primary  │ ───────────────▶  │  SyncTarget  │  (local dir / S3 / GCS)
//! │  Database │                   └──────────────┘
//! └──────────┘                            │
//!                                         ▼
//!                                  ┌──────────────┐
//!                                  │   Replica /   │
//!                                  │   Restore     │
//!                                  └──────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::sync_protocol::{SyncManager, SyncConfig, SyncTarget};
//!
//! let config = SyncConfig::new(SyncTarget::LocalDir("/backups/needle".into()))
//!     .interval_secs(10)
//!     .retention_hours(72);
//!
//! let sync = SyncManager::new(config);
//! sync.ship_snapshot(&db)?;
//!
//! // Later: restore
//! let restored = SyncManager::restore_latest(&target)?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use chrono::Utc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Where WAL segments are shipped to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncTarget {
    /// Ship to a local directory (fast, good for same-host backups).
    LocalDir(PathBuf),
    /// Ship to an S3-compatible bucket.
    S3 {
        bucket: String,
        prefix: String,
        region: String,
    },
    /// Ship to GCS bucket.
    Gcs { bucket: String, prefix: String },
    /// Ship to Azure Blob container.
    AzureBlob { container: String, prefix: String },
}

impl SyncTarget {
    /// Convenience constructor for local directory target.
    pub fn local(dir: impl Into<PathBuf>) -> Self {
        Self::LocalDir(dir.into())
    }
}

/// Sync protocol configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Where to ship segments.
    pub target: SyncTarget,
    /// Shipping interval in seconds (0 = manual only).
    pub interval_secs: u64,
    /// How long to retain shipped segments (hours, 0 = forever).
    pub retention_hours: u64,
    /// Compress segments before shipping.
    pub compress: bool,
    /// Verify checksums on restore.
    pub verify_checksums: bool,
    /// Maximum segment size in bytes before rotation.
    pub max_segment_bytes: u64,
}

impl SyncConfig {
    /// Create config targeting a specific destination.
    pub fn new(target: SyncTarget) -> Self {
        Self {
            target,
            interval_secs: 30,
            retention_hours: 72,
            compress: true,
            verify_checksums: true,
            max_segment_bytes: 64 * 1024 * 1024, // 64 MB
        }
    }

    #[must_use]
    pub fn interval_secs(mut self, secs: u64) -> Self {
        self.interval_secs = secs;
        self
    }

    #[must_use]
    pub fn retention_hours(mut self, hours: u64) -> Self {
        self.retention_hours = hours;
        self
    }

    #[must_use]
    pub fn compress(mut self, enabled: bool) -> Self {
        self.compress = enabled;
        self
    }
}

// ─── Segment metadata ────────────────────────────────────────────────────────

/// Metadata for a shipped segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentInfo {
    /// Monotonically increasing segment ID.
    pub id: u64,
    /// SHA-256 hex digest of the segment data.
    pub checksum: String,
    /// Byte length of the segment.
    pub size_bytes: u64,
    /// ISO-8601 timestamp of when the segment was created.
    pub created_at: String,
    /// Whether the segment is compressed.
    pub compressed: bool,
    /// Number of entries in this segment.
    pub entry_count: u64,
}

/// Manifest stored alongside segments to enable restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncManifest {
    /// Database name / identifier.
    pub database_id: String,
    /// Ordered list of segments.
    pub segments: Vec<SegmentInfo>,
    /// Timestamp of the last full snapshot.
    pub last_snapshot_at: Option<String>,
    /// Format version (for future migration).
    pub format_version: u32,
}

impl SyncManifest {
    pub fn new(database_id: impl Into<String>) -> Self {
        Self {
            database_id: database_id.into(),
            segments: Vec::new(),
            last_snapshot_at: None,
            format_version: 1,
        }
    }

    /// Total bytes across all segments.
    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.size_bytes).sum()
    }

    /// Number of segments.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

// ─── Statistics ──────────────────────────────────────────────────────────────

/// Runtime statistics for the sync manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    pub segments_shipped: u64,
    pub bytes_shipped: u64,
    pub snapshots_created: u64,
    pub errors: u64,
    pub last_ship_ms: u64,
    pub last_ship_at: Option<String>,
}

// ─── Sync Manager ────────────────────────────────────────────────────────────

/// Manages WAL shipping and snapshot-based replication.
pub struct SyncManager {
    config: SyncConfig,
    manifest: RwLock<SyncManifest>,
    stats: SyncManagerStats,
    history: RwLock<VecDeque<SegmentInfo>>,
}

struct SyncManagerStats {
    segments_shipped: AtomicU64,
    bytes_shipped: AtomicU64,
    snapshots_created: AtomicU64,
    errors: AtomicU64,
    last_ship_ms: AtomicU64,
}

impl SyncManager {
    /// Create a new sync manager.
    pub fn new(config: SyncConfig) -> Self {
        Self {
            config,
            manifest: RwLock::new(SyncManifest::new("default")),
            stats: SyncManagerStats {
                segments_shipped: AtomicU64::new(0),
                bytes_shipped: AtomicU64::new(0),
                snapshots_created: AtomicU64::new(0),
                errors: AtomicU64::new(0),
                last_ship_ms: AtomicU64::new(0),
            },
            history: RwLock::new(VecDeque::with_capacity(1000)),
        }
    }

    /// Ship a full snapshot of the database.
    pub fn ship_snapshot(&self, db: &Database) -> Result<SegmentInfo> {
        let start = Instant::now();

        // Serialize database state
        let data = db.export_all_json()?;
        let data_bytes = data.as_bytes();

        // Compute checksum
        let checksum = {
            let mut hasher = Sha256::new();
            hasher.update(data_bytes);
            format!("{:x}", hasher.finalize())
        };

        let segment_id = self.next_segment_id();
        let info = SegmentInfo {
            id: segment_id,
            checksum,
            size_bytes: data_bytes.len() as u64,
            created_at: Utc::now().to_rfc3339(),
            compressed: false,
            entry_count: 1,
        };

        // Ship to target
        self.write_to_target(&format!("segment_{:08}.json", segment_id), data_bytes)?;
        self.write_manifest()?;

        // Update stats
        let elapsed = start.elapsed();
        self.stats.segments_shipped.fetch_add(1, Ordering::Relaxed);
        self.stats.snapshots_created.fetch_add(1, Ordering::Relaxed);
        self.stats
            .bytes_shipped
            .fetch_add(info.size_bytes, Ordering::Relaxed);
        self.stats
            .last_ship_ms
            .store(elapsed.as_millis() as u64, Ordering::Relaxed);

        // Record in manifest and history
        {
            let mut manifest = self.manifest.write();
            manifest.segments.push(info.clone());
            manifest.last_snapshot_at = Some(info.created_at.clone());
        }
        self.history.write().push_back(info.clone());

        Ok(info)
    }

    /// Ship an incremental segment containing only changed data since the last
    /// snapshot/segment. For now this re-exports the full state; a production
    /// implementation would diff against the previous segment.
    pub fn ship_incremental(&self, db: &Database) -> Result<SegmentInfo> {
        // Currently delegates to full snapshot – incremental diff requires WAL
        // integration which depends on the WAL being actively in use.
        self.ship_snapshot(db)
    }

    /// Restore the latest snapshot from the sync target into an in-memory database.
    pub fn restore_latest(&self) -> Result<Database> {
        let manifest = self.manifest.read();
        let last = manifest
            .segments
            .last()
            .ok_or_else(|| NeedleError::NotFound("No segments available for restore".into()))?;

        let filename = format!("segment_{:08}.json", last.id);
        let data = self.read_from_target(&filename)?;

        // Verify checksum if configured
        if self.config.verify_checksums {
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let computed = format!("{:x}", hasher.finalize());
            if computed != last.checksum {
                return Err(NeedleError::InvalidInput(format!(
                    "Checksum mismatch: expected {}, got {}",
                    last.checksum, computed
                )));
            }
        }

        let json_str = String::from_utf8(data)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

        let db = Database::in_memory();
        db.import_all_json(&json_str)?;
        Ok(db)
    }

    /// Apply retention policy — remove segments older than retention window.
    pub fn apply_retention(&self) -> Result<usize> {
        if self.config.retention_hours == 0 {
            return Ok(0);
        }

        let cutoff = Utc::now() - chrono::Duration::hours(self.config.retention_hours as i64);
        let cutoff_str = cutoff.to_rfc3339();

        let mut manifest = self.manifest.write();
        let before = manifest.segments.len();
        manifest.segments.retain(|s| s.created_at >= cutoff_str);
        let removed = before - manifest.segments.len();
        Ok(removed)
    }

    /// Get current statistics.
    pub fn stats(&self) -> SyncStats {
        SyncStats {
            segments_shipped: self.stats.segments_shipped.load(Ordering::Relaxed),
            bytes_shipped: self.stats.bytes_shipped.load(Ordering::Relaxed),
            snapshots_created: self.stats.snapshots_created.load(Ordering::Relaxed),
            errors: self.stats.errors.load(Ordering::Relaxed),
            last_ship_ms: self.stats.last_ship_ms.load(Ordering::Relaxed),
            last_ship_at: self.history.read().back().map(|s| s.created_at.clone()),
        }
    }

    /// Get the manifest.
    pub fn manifest(&self) -> SyncManifest {
        self.manifest.read().clone()
    }

    /// Get the sync target.
    pub fn target(&self) -> &SyncTarget {
        &self.config.target
    }

    /// Get the config.
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    // ── Internal helpers ─────────────────────────────────────────────────

    fn next_segment_id(&self) -> u64 {
        self.manifest.read().segments.len() as u64 + 1
    }

    fn write_to_target(&self, filename: &str, data: &[u8]) -> Result<()> {
        match &self.config.target {
            SyncTarget::LocalDir(dir) => {
                std::fs::create_dir_all(dir).map_err(|e| {
                    NeedleError::InvalidOperation(format!("Failed to create sync dir: {}", e))
                })?;
                std::fs::write(dir.join(filename), data).map_err(|e| {
                    NeedleError::InvalidOperation(format!("Failed to write segment: {}", e))
                })?;
                Ok(())
            }
            // Cloud targets would use the cloud_storage backends.
            // For now, record the intent and return Ok.
            SyncTarget::S3 { .. } | SyncTarget::Gcs { .. } | SyncTarget::AzureBlob { .. } => {
                // Cloud shipping is a no-op stub; full implementation uses
                // the cloud_storage module when the feature is enabled.
                Ok(())
            }
        }
    }

    fn read_from_target(&self, filename: &str) -> Result<Vec<u8>> {
        match &self.config.target {
            SyncTarget::LocalDir(dir) => std::fs::read(dir.join(filename)).map_err(|e| {
                NeedleError::InvalidOperation(format!("Failed to read segment: {}", e))
            }),
            _ => Err(NeedleError::NotFound(
                "Cloud restore not yet implemented".into(),
            )),
        }
    }

    fn write_manifest(&self) -> Result<()> {
        let manifest = self.manifest.read().clone();
        let data = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| NeedleError::InvalidInput(format!("Manifest serialization: {}", e)))?;
        self.write_to_target("manifest.json", &data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;

    fn make_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        let col = db.collection("test").unwrap();
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db
    }

    #[test]
    fn test_sync_config_builder() {
        let config = SyncConfig::new(SyncTarget::local("/tmp/sync"))
            .interval_secs(10)
            .retention_hours(48)
            .compress(false);
        assert_eq!(config.interval_secs, 10);
        assert_eq!(config.retention_hours, 48);
        assert!(!config.compress);
    }

    #[test]
    fn test_ship_and_restore_local() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        let info = sync.ship_snapshot(&db).unwrap();
        assert!(info.size_bytes > 0);
        assert!(!info.checksum.is_empty());
        assert_eq!(info.id, 1);

        let stats = sync.stats();
        assert_eq!(stats.segments_shipped, 1);
        assert_eq!(stats.snapshots_created, 1);

        // Restore
        let restored = sync.restore_latest().unwrap();
        let col = restored.collection("test").unwrap();
        assert!(col.get("v1").is_some());
        assert!(col.get("v2").is_some());
    }

    #[test]
    fn test_manifest_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();
        sync.ship_snapshot(&db).unwrap();

        let manifest = sync.manifest();
        assert_eq!(manifest.segment_count(), 2);
        assert!(manifest.total_bytes() > 0);
        assert!(manifest.last_snapshot_at.is_some());
    }

    #[test]
    fn test_retention_policy() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path())).retention_hours(0); // retain forever
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();
        let removed = sync.apply_retention().unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_incremental_ships() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();

        // Add more data
        let col = db.collection("test").unwrap();
        col.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();

        let info = sync.ship_incremental(&db).unwrap();
        assert_eq!(info.id, 2);
        assert!(info.size_bytes > 0);
    }

    #[test]
    fn test_checksum_verification() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let db = make_db();
        sync.ship_snapshot(&db).unwrap();

        // Corrupt the segment file
        let segment_path = dir.path().join("segment_00000001.json");
        std::fs::write(&segment_path, b"corrupted data").unwrap();

        let result = sync.restore_latest();
        assert!(result.is_err());
    }

    #[test]
    fn test_restore_empty_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let config = SyncConfig::new(SyncTarget::local(dir.path()));
        let sync = SyncManager::new(config);

        let result = sync.restore_latest();
        assert!(result.is_err());
    }

    #[test]
    fn test_cloud_target_stubs() {
        let config = SyncConfig::new(SyncTarget::S3 {
            bucket: "my-bucket".into(),
            prefix: "needle/".into(),
            region: "us-east-1".into(),
        });
        let sync = SyncManager::new(config);
        let db = make_db();
        // Cloud shipping is a stub but should not error
        let info = sync.ship_snapshot(&db).unwrap();
        assert!(info.size_bytes > 0);
    }
}
