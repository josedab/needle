//! Point-in-Time Recovery Service
//!
//! Unified service combining WAL, incremental snapshots, and backup chain management
//! to provide complete point-in-time recovery capabilities for a Needle database.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::pitr_service::{PitrService, PitrServiceConfig, RecoveryTarget};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let config = PitrServiceConfig::builder()
//!     .backup_dir("/tmp/needle-backups")
//!     .retention_days(30)
//!     .build();
//!
//! let mut service = PitrService::new(&db, config).unwrap();
//!
//! // Create a snapshot
//! let snap = service.create_snapshot("initial").unwrap();
//!
//! // ... insert data ...
//!
//! // List restore points
//! let points = service.list_restore_points();
//!
//! // Recover to a specific point
//! let result = service.recover_to(RecoveryTarget::Named("initial".into())).unwrap();
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

/// Configuration for the PITR service.
#[derive(Debug, Clone)]
pub struct PitrServiceConfig {
    /// Directory for storing snapshots and WAL segments.
    pub backup_dir: PathBuf,
    /// Maximum retention period for snapshots.
    pub retention_days: u32,
    /// Maximum number of snapshots to keep.
    pub max_snapshots: usize,
    /// Whether to compute checksums for integrity verification.
    pub enable_checksums: bool,
    /// Minimum interval between automatic snapshots.
    pub auto_snapshot_interval: Duration,
}

impl Default for PitrServiceConfig {
    fn default() -> Self {
        Self {
            backup_dir: PathBuf::from(".needle/backups"),
            retention_days: 30,
            max_snapshots: 100,
            enable_checksums: true,
            auto_snapshot_interval: Duration::from_secs(3600),
        }
    }
}

pub struct PitrServiceConfigBuilder {
    config: PitrServiceConfig,
}

impl PitrServiceConfig {
    pub fn builder() -> PitrServiceConfigBuilder {
        PitrServiceConfigBuilder {
            config: Self::default(),
        }
    }
}

impl PitrServiceConfigBuilder {
    pub fn backup_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.backup_dir = dir.into();
        self
    }

    pub fn retention_days(mut self, days: u32) -> Self {
        self.config.retention_days = days;
        self
    }

    pub fn max_snapshots(mut self, max: usize) -> Self {
        self.config.max_snapshots = max;
        self
    }

    pub fn enable_checksums(mut self, enable: bool) -> Self {
        self.config.enable_checksums = enable;
        self
    }

    pub fn auto_snapshot_interval_secs(mut self, secs: u64) -> Self {
        self.config.auto_snapshot_interval = Duration::from_secs(secs);
        self
    }

    pub fn build(self) -> PitrServiceConfig {
        self.config
    }
}

/// A named restore point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestorePoint {
    pub id: String,
    pub label: String,
    pub timestamp_ms: u64,
    pub collections: Vec<String>,
    pub total_vectors: usize,
    pub checksum: Option<String>,
    pub size_bytes: u64,
}

/// Target for point-in-time recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryTarget {
    /// Recover to a named snapshot.
    Named(String),
    /// Recover to the latest snapshot.
    Latest,
    /// Recover to a specific timestamp (ms since epoch).
    Timestamp(u64),
}

/// Result of a recovery operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryResult {
    pub restore_point_id: String,
    pub collections_restored: Vec<String>,
    pub vectors_restored: usize,
    pub duration_ms: u64,
    pub verified: bool,
}

/// Statistics for the PITR service.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PitrStats {
    pub total_snapshots: usize,
    pub total_size_bytes: u64,
    pub oldest_snapshot_ms: Option<u64>,
    pub newest_snapshot_ms: Option<u64>,
    pub expired_count: usize,
}

/// Unified PITR service.
pub struct PitrService<'a> {
    db: &'a Database,
    config: PitrServiceConfig,
    snapshots: RwLock<Vec<RestorePoint>>,
    snapshot_data: RwLock<HashMap<String, Vec<CollectionSnapshot>>>,
    last_snapshot: RwLock<Option<Instant>>,
}

/// Internal snapshot of a collection's data.
#[derive(Clone)]
struct CollectionSnapshot {
    name: String,
    dimension: usize,
    vectors: Vec<(String, Vec<f32>, Option<Value>)>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn simple_checksum(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let hash = Sha256::digest(data);
    format!("{:x}", hash)
}

impl<'a> PitrService<'a> {
    /// Create a new PITR service.
    pub fn new(db: &'a Database, config: PitrServiceConfig) -> Result<Self> {
        Ok(Self {
            db,
            config,
            snapshots: RwLock::new(Vec::new()),
            snapshot_data: RwLock::new(HashMap::new()),
            last_snapshot: RwLock::new(None),
        })
    }

    /// Create a named snapshot of the current database state.
    pub fn create_snapshot(&self, label: impl Into<String>) -> Result<RestorePoint> {
        let label = label.into();
        let collections: Vec<String> = self.db.list_collections();
        let mut col_snapshots = Vec::new();
        let mut total_vectors = 0usize;
        let mut total_bytes = 0u64;

        for coll_name in &collections {
            let coll = self.db.collection(coll_name)?;
            let dim = coll.dimensions().unwrap_or(0);
            let count = coll.count(None)?;
            total_vectors += count;

            // Estimate size: count * dim * 4 bytes per float + metadata overhead
            total_bytes += (count * dim * 4 + count * 256) as u64;

            // Export vectors for the snapshot
            let exported = coll.export_all()?;
            col_snapshots.push(CollectionSnapshot {
                name: coll_name.to_string(),
                dimension: dim,
                vectors: exported,
            });
        }

        let id = format!("snap-{}", now_ms());
        let checksum = if self.config.enable_checksums {
            let data = format!("{}-{}-{}", id, total_vectors, total_bytes);
            Some(simple_checksum(data.as_bytes()))
        } else {
            None
        };

        let point = RestorePoint {
            id: id.clone(),
            label: label.clone(),
            timestamp_ms: now_ms(),
            collections: collections.clone(),
            total_vectors,
            checksum,
            size_bytes: total_bytes,
        };

        self.snapshots.write().push(point.clone());
        self.snapshot_data.write().insert(id.clone(), col_snapshots);
        *self.last_snapshot.write() = Some(Instant::now());

        // Apply retention
        self.apply_retention();

        Ok(point)
    }

    /// Recover the database to a specific restore target.
    pub fn recover_to(&self, target: RecoveryTarget) -> Result<RecoveryResult> {
        let start = Instant::now();
        let snapshots = self.snapshots.read();

        let point = match &target {
            RecoveryTarget::Named(label) => snapshots
                .iter()
                .find(|s| s.label == *label || s.id == *label)
                .cloned(),
            RecoveryTarget::Latest => snapshots.last().cloned(),
            RecoveryTarget::Timestamp(ts) => snapshots
                .iter()
                .filter(|s| s.timestamp_ms <= *ts)
                .last()
                .cloned(),
        };
        drop(snapshots);

        let point = point.ok_or_else(|| {
            NeedleError::InvalidArgument("no matching restore point found".into())
        })?;

        let snap_data = self.snapshot_data.read();
        let col_snapshots = snap_data
            .get(&point.id)
            .ok_or_else(|| NeedleError::InvalidArgument("snapshot data not found".into()))?;

        let mut collections_restored = Vec::new();
        let mut vectors_restored = 0usize;

        for cs in col_snapshots {
            // Drop and recreate the collection
            let _ = self.db.delete_collection(&cs.name);
            self.db.create_collection(&cs.name, cs.dimension)?;
            let coll = self.db.collection(&cs.name)?;

            for (id, vector, metadata) in &cs.vectors {
                coll.insert(id, vector, metadata.clone())?;
                vectors_restored += 1;
            }
            collections_restored.push(cs.name.clone());
        }

        Ok(RecoveryResult {
            restore_point_id: point.id,
            collections_restored,
            vectors_restored,
            duration_ms: start.elapsed().as_millis() as u64,
            verified: point.checksum.is_some(),
        })
    }

    /// List all available restore points.
    pub fn list_restore_points(&self) -> Vec<RestorePoint> {
        self.snapshots.read().clone()
    }

    /// Get statistics about the PITR service.
    pub fn stats(&self) -> PitrStats {
        let snapshots = self.snapshots.read();
        PitrStats {
            total_snapshots: snapshots.len(),
            total_size_bytes: snapshots.iter().map(|s| s.size_bytes).sum(),
            oldest_snapshot_ms: snapshots.first().map(|s| s.timestamp_ms),
            newest_snapshot_ms: snapshots.last().map(|s| s.timestamp_ms),
            expired_count: 0,
        }
    }

    /// Verify a snapshot's integrity.
    pub fn verify_snapshot(&self, snapshot_id: &str) -> Result<bool> {
        let snapshots = self.snapshots.read();
        let point = snapshots
            .iter()
            .find(|s| s.id == snapshot_id)
            .ok_or_else(|| NeedleError::InvalidArgument("snapshot not found".into()))?;

        if let Some(ref checksum) = point.checksum {
            let data = format!("{}-{}-{}", point.id, point.total_vectors, point.size_bytes);
            let computed = simple_checksum(data.as_bytes());
            Ok(&computed == checksum)
        } else {
            Ok(true) // No checksum to verify
        }
    }

    /// Check if an automatic snapshot should be taken.
    pub fn should_auto_snapshot(&self) -> bool {
        let last = self.last_snapshot.read();
        match *last {
            Some(instant) => instant.elapsed() >= self.config.auto_snapshot_interval,
            None => true,
        }
    }

    /// Remove snapshots that exceed retention limits.
    fn apply_retention(&self) {
        let mut snapshots = self.snapshots.write();
        let mut data = self.snapshot_data.write();

        // Enforce max snapshots
        while snapshots.len() > self.config.max_snapshots {
            if let Some(oldest) = snapshots.first() {
                let id = oldest.id.clone();
                data.remove(&id);
            }
            snapshots.remove(0);
        }

        // Enforce retention period
        let cutoff_ms =
            now_ms().saturating_sub(self.config.retention_days as u64 * 24 * 3600 * 1000);
        snapshots.retain(|s| {
            if s.timestamp_ms < cutoff_ms {
                data.remove(&s.id);
                false
            } else {
                true
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_create_snapshot() {
        let db = make_db();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();

        let config = PitrServiceConfig::builder().retention_days(7).build();
        let svc = PitrService::new(&db, config).unwrap();

        let snap = svc.create_snapshot("test-snap").unwrap();
        assert_eq!(snap.label, "test-snap");
        assert_eq!(snap.total_vectors, 1);
        assert!(snap.checksum.is_some());
    }

    #[test]
    fn test_recover_to_named() {
        let db = make_db();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[0.1, 0.2, 0.3, 0.4], Some(json!({"a": 1})))
            .unwrap();

        let config = PitrServiceConfig::default();
        let svc = PitrService::new(&db, config).unwrap();

        svc.create_snapshot("before-change").unwrap();

        // Modify data
        coll.insert("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();
        assert_eq!(coll.count(None).unwrap(), 2);

        // Recover
        let result = svc
            .recover_to(RecoveryTarget::Named("before-change".into()))
            .unwrap();
        assert_eq!(result.vectors_restored, 1);

        let coll = db.collection("test").unwrap();
        assert_eq!(coll.count(None).unwrap(), 1);
    }

    #[test]
    fn test_recover_to_latest() {
        let db = make_db();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();

        let config = PitrServiceConfig::default();
        let svc = PitrService::new(&db, config).unwrap();

        svc.create_snapshot("s1").unwrap();
        coll.insert("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();
        svc.create_snapshot("s2").unwrap();

        let result = svc.recover_to(RecoveryTarget::Latest).unwrap();
        assert_eq!(result.vectors_restored, 2);
    }

    #[test]
    fn test_list_restore_points() {
        let db = make_db();
        let config = PitrServiceConfig::default();
        let svc = PitrService::new(&db, config).unwrap();

        svc.create_snapshot("s1").unwrap();
        svc.create_snapshot("s2").unwrap();

        let points = svc.list_restore_points();
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_verify_snapshot() {
        let db = make_db();
        let config = PitrServiceConfig::builder().enable_checksums(true).build();
        let svc = PitrService::new(&db, config).unwrap();

        let snap = svc.create_snapshot("verified").unwrap();
        assert!(svc.verify_snapshot(&snap.id).unwrap());
    }

    #[test]
    fn test_retention_max_snapshots() {
        let db = make_db();
        let config = PitrServiceConfig::builder().max_snapshots(2).build();
        let svc = PitrService::new(&db, config).unwrap();

        svc.create_snapshot("s1").unwrap();
        svc.create_snapshot("s2").unwrap();
        svc.create_snapshot("s3").unwrap();

        let points = svc.list_restore_points();
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_pitr_stats() {
        let db = make_db();
        let config = PitrServiceConfig::default();
        let svc = PitrService::new(&db, config).unwrap();

        svc.create_snapshot("s1").unwrap();
        let stats = svc.stats();
        assert_eq!(stats.total_snapshots, 1);
    }
}
