//! Time-Travel Query Service
//!
//! Query the database as of any past timestamp with MVCC snapshots, supporting
//! diff between versions, rollback, and audit trails.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::time_travel_query::{
//!     TimeTravelService, TimeTravelConfig, VersionedVector,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 4).unwrap();
//!
//! let mut service = TimeTravelService::new(&db, "docs", TimeTravelConfig::default()).unwrap();
//!
//! // Versioned insert
//! service.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
//! let ts1 = service.current_version();
//!
//! // Update the vector
//! service.insert("v1", &[5.0, 6.0, 7.0, 8.0], None).unwrap();
//! let ts2 = service.current_version();
//!
//! // Query at historical point
//! let result = service.get_at("v1", ts1).unwrap();
//! assert_eq!(result.unwrap().vector, vec![1.0, 2.0, 3.0, 4.0]);
//!
//! // Diff between versions
//! let diff = service.diff(ts1, ts2);
//! assert_eq!(diff.modified.len(), 1);
//! ```

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Time-travel service configuration.
#[derive(Debug, Clone)]
pub struct TimeTravelConfig {
    /// Maximum number of versions to retain per vector.
    pub max_versions_per_key: usize,
    /// Maximum total snapshots to retain.
    pub max_snapshots: usize,
    /// Whether to auto-create snapshots on flush.
    pub auto_snapshot: bool,
    /// Snapshot interval (every N operations).
    pub snapshot_interval: u64,
}

impl Default for TimeTravelConfig {
    fn default() -> Self {
        Self {
            max_versions_per_key: 100,
            max_snapshots: 1000,
            auto_snapshot: true,
            snapshot_interval: 100,
        }
    }
}

// ── Version Types ────────────────────────────────────────────────────────────

/// A monotonically increasing version number.
pub type Version = u64;

/// A versioned vector record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedVector {
    /// Vector ID.
    pub id: String,
    /// Version at which this record was written.
    pub version: Version,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata at this version.
    pub metadata: Option<Value>,
    /// Whether this is a tombstone (deletion marker).
    pub deleted: bool,
}

/// A snapshot: a named point-in-time marker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    /// Snapshot name/label.
    pub name: String,
    /// Version at snapshot time.
    pub version: Version,
    /// Optional description.
    pub description: Option<String>,
    /// Timestamp (epoch seconds).
    pub created_at: u64,
}

// ── Diff Types ───────────────────────────────────────────────────────────────

/// Diff between two versions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionDiff {
    /// IDs added between from_version and to_version.
    pub added: Vec<String>,
    /// IDs modified between from_version and to_version.
    pub modified: Vec<String>,
    /// IDs deleted between from_version and to_version.
    pub deleted: Vec<String>,
    /// Source version.
    pub from_version: Version,
    /// Target version.
    pub to_version: Version,
}

// ── Audit Entry ──────────────────────────────────────────────────────────────

/// Audit trail entry for a mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Version of this operation.
    pub version: Version,
    /// Operation type.
    pub operation: AuditOp,
    /// Affected vector ID.
    pub vector_id: String,
    /// Timestamp (epoch seconds).
    pub timestamp: u64,
}

/// Audit operation type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditOp {
    /// Vector inserted or updated.
    Upsert,
    /// Vector deleted.
    Delete,
    /// Snapshot created.
    Snapshot,
    /// Rollback performed.
    Rollback,
}

// ── Time-Travel Service ──────────────────────────────────────────────────────

/// Version store: maps vector ID → BTreeMap<version, record>.
type VersionStore = HashMap<String, BTreeMap<Version, VersionedVector>>;

/// Time-travel query service with MVCC version tracking.
pub struct TimeTravelService<'a> {
    db: &'a Database,
    collection_name: String,
    config: TimeTravelConfig,
    versions: VersionStore,
    current: Version,
    snapshots: Vec<Snapshot>,
    audit_log: Vec<AuditEntry>,
    ops_since_snapshot: u64,
}

impl<'a> TimeTravelService<'a> {
    /// Create a new time-travel service for a collection.
    pub fn new(
        db: &'a Database,
        collection: &str,
        config: TimeTravelConfig,
    ) -> Result<Self> {
        let _coll = db.collection(collection)?;
        Ok(Self {
            db,
            collection_name: collection.to_string(),
            config,
            versions: HashMap::new(),
            current: 0,
            snapshots: Vec::new(),
            audit_log: Vec::new(),
            ops_since_snapshot: 0,
        })
    }

    /// Insert or update a vector (creates a new version).
    pub fn insert(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<Version> {
        self.current += 1;
        let version = self.current;

        let record = VersionedVector {
            id: id.to_string(),
            version,
            vector: vector.to_vec(),
            metadata: metadata.clone(),
            deleted: false,
        };

        let history = self.versions.entry(id.to_string()).or_default();
        history.insert(version, record);

        // Trim old versions
        while history.len() > self.config.max_versions_per_key {
            if let Some(&oldest) = history.keys().next() {
                history.remove(&oldest);
            }
        }

        // Write to the actual collection (delete first if exists for update)
        let coll = self.db.collection(&self.collection_name)?;
        if coll.get(id).is_some() {
            let _ = coll.delete(id);
        }
        coll.insert(id, vector, metadata)?;

        self.record_audit(version, AuditOp::Upsert, id);
        self.maybe_auto_snapshot()?;

        Ok(version)
    }

    /// Delete a vector (creates a tombstone version).
    pub fn delete(&mut self, id: &str) -> Result<Version> {
        self.current += 1;
        let version = self.current;

        let tombstone = VersionedVector {
            id: id.to_string(),
            version,
            vector: Vec::new(),
            metadata: None,
            deleted: true,
        };

        let history = self.versions.entry(id.to_string()).or_default();
        history.insert(version, tombstone);

        let coll = self.db.collection(&self.collection_name)?;
        let _ = coll.delete(id);

        self.record_audit(version, AuditOp::Delete, id);
        self.maybe_auto_snapshot()?;

        Ok(version)
    }

    /// Get a vector at a specific version.
    pub fn get_at(&self, id: &str, version: Version) -> Result<Option<VersionedVector>> {
        let history = match self.versions.get(id) {
            Some(h) => h,
            None => return Ok(None),
        };

        // Find the latest version <= requested version
        let record = history
            .range(..=version)
            .next_back()
            .map(|(_, v)| v.clone());

        match record {
            Some(r) if r.deleted => Ok(None),
            other => Ok(other),
        }
    }

    /// Search at a specific version (reconstructs the collection state).
    pub fn search_at(
        &self,
        query: &[f32],
        k: usize,
        version: Version,
    ) -> Result<Vec<VersionedVector>> {
        let mut results: Vec<VersionedVector> = Vec::new();

        for (_id, history) in &self.versions {
            if let Some((_, record)) = history.range(..=version).next_back() {
                if !record.deleted {
                    results.push(record.clone());
                }
            }
        }

        // Sort by distance to query
        let dist_fn = crate::distance::DistanceFunction::Cosine;
        results.sort_by(|a, b| {
            let da = dist_fn.compute(query, &a.vector);
            let db_dist = dist_fn.compute(query, &b.vector);
            da.partial_cmp(&db_dist).unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(k);
        Ok(results)
    }

    /// Compute diff between two versions.
    pub fn diff(&self, from: Version, to: Version) -> VersionDiff {
        let mut diff = VersionDiff {
            from_version: from,
            to_version: to,
            ..Default::default()
        };

        for (id, history) in &self.versions {
            let at_from = history
                .range(..=from)
                .next_back()
                .map(|(_, r)| r.clone());
            let at_to = history
                .range(..=to)
                .next_back()
                .map(|(_, r)| r.clone());

            match (at_from, at_to) {
                (None, Some(r)) if !r.deleted => {
                    diff.added.push(id.clone());
                }
                (Some(r), None) if !r.deleted => {
                    diff.deleted.push(id.clone());
                }
                (Some(a), Some(b)) => {
                    if a.deleted && !b.deleted {
                        diff.added.push(id.clone());
                    } else if !a.deleted && b.deleted {
                        diff.deleted.push(id.clone());
                    } else if !a.deleted && !b.deleted && a.version != b.version {
                        diff.modified.push(id.clone());
                    }
                }
                _ => {}
            }
        }

        diff
    }

    /// Create a named snapshot at the current version.
    pub fn create_snapshot(&mut self, name: &str, description: Option<&str>) -> Result<Snapshot> {
        let snap = Snapshot {
            name: name.to_string(),
            version: self.current,
            description: description.map(|s| s.to_string()),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.snapshots.push(snap.clone());
        self.record_audit(self.current, AuditOp::Snapshot, name);

        // Trim old snapshots
        while self.snapshots.len() > self.config.max_snapshots {
            self.snapshots.remove(0);
        }

        Ok(snap)
    }

    /// List all snapshots.
    pub fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }

    /// Find a snapshot by name.
    pub fn find_snapshot(&self, name: &str) -> Option<&Snapshot> {
        self.snapshots.iter().find(|s| s.name == name)
    }

    /// Rollback to a specific version. Creates new versions that match the
    /// state at the rollback target. Does not delete history.
    pub fn rollback_to(&mut self, target_version: Version) -> Result<Version> {
        // Collect the state at target_version
        let mut state_at_target: Vec<(String, Option<VersionedVector>)> = Vec::new();
        for (id, history) in &self.versions {
            let record = history.range(..=target_version).next_back().map(|(_, r)| r.clone());
            state_at_target.push((id.clone(), record));
        }

        // Apply the target state as new versions
        for (id, record) in state_at_target {
            match record {
                Some(r) if !r.deleted => {
                    self.insert(&id, &r.vector, r.metadata)?;
                }
                _ => {
                    // Delete if it was deleted or didn't exist at target
                    if self.versions.contains_key(&id) {
                        let current_latest = self
                            .versions
                            .get(&id)
                            .and_then(|h| h.values().next_back())
                            .map(|r| r.deleted);
                        if current_latest == Some(false) {
                            self.delete(&id)?;
                        }
                    }
                }
            }
        }

        self.record_audit(self.current, AuditOp::Rollback, &format!("v{target_version}"));
        Ok(self.current)
    }

    /// Get version history for a specific vector.
    pub fn history(&self, id: &str) -> Vec<VersionedVector> {
        self.versions
            .get(id)
            .map(|h| h.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Get the full audit log.
    pub fn audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }

    /// Get the current (latest) version number.
    pub fn current_version(&self) -> Version {
        self.current
    }

    /// Total number of versioned records across all vectors.
    pub fn total_versions(&self) -> usize {
        self.versions.values().map(|h| h.len()).sum()
    }

    fn record_audit(&mut self, version: Version, op: AuditOp, id: &str) {
        self.audit_log.push(AuditEntry {
            version,
            operation: op,
            vector_id: id.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        });
    }

    fn maybe_auto_snapshot(&mut self) -> Result<()> {
        self.ops_since_snapshot += 1;
        if self.config.auto_snapshot && self.ops_since_snapshot >= self.config.snapshot_interval {
            self.ops_since_snapshot = 0;
            let name = format!("auto_v{}", self.current);
            self.create_snapshot(&name, Some("automatic snapshot"))?;
        }
        Ok(())
    }
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
    fn test_versioned_insert_and_get() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        let v1 = svc.insert("a", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
        let v2 = svc.insert("a", &[5.0, 6.0, 7.0, 8.0], None).unwrap();

        let at_v1 = svc.get_at("a", v1).unwrap().unwrap();
        assert_eq!(at_v1.vector, vec![1.0, 2.0, 3.0, 4.0]);

        let at_v2 = svc.get_at("a", v2).unwrap().unwrap();
        assert_eq!(at_v2.vector, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_delete_creates_tombstone() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        let v1 = svc.insert("a", &[1.0; 4], None).unwrap();
        let v2 = svc.delete("a").unwrap();

        assert!(svc.get_at("a", v1).unwrap().is_some());
        assert!(svc.get_at("a", v2).unwrap().is_none());
    }

    #[test]
    fn test_diff() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        let v0 = svc.current_version();
        svc.insert("a", &[1.0; 4], None).unwrap();
        svc.insert("b", &[2.0; 4], None).unwrap();
        let v1 = svc.current_version();

        let diff = svc.diff(v0, v1);
        assert_eq!(diff.added.len(), 2);
        assert!(diff.modified.is_empty());
        assert!(diff.deleted.is_empty());
    }

    #[test]
    fn test_diff_with_modifications() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        svc.insert("a", &[1.0; 4], None).unwrap();
        let v1 = svc.current_version();

        svc.insert("a", &[2.0; 4], None).unwrap();
        let v2 = svc.current_version();

        let diff = svc.diff(v1, v2);
        assert_eq!(diff.modified.len(), 1);
        assert_eq!(diff.modified[0], "a");
    }

    #[test]
    fn test_snapshots() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        svc.insert("a", &[1.0; 4], None).unwrap();
        let snap = svc
            .create_snapshot("release-1", Some("first release"))
            .unwrap();
        assert_eq!(snap.name, "release-1");

        assert_eq!(svc.snapshots().len(), 1);
        assert!(svc.find_snapshot("release-1").is_some());
        assert!(svc.find_snapshot("nonexistent").is_none());
    }

    #[test]
    fn test_rollback() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        svc.insert("a", &[1.0; 4], None).unwrap();
        let v1 = svc.current_version();

        svc.insert("a", &[9.0; 4], None).unwrap();
        svc.insert("b", &[5.0; 4], None).unwrap();

        svc.rollback_to(v1).unwrap();

        // After rollback, current state should match v1
        let latest_a = svc.get_at("a", svc.current_version()).unwrap().unwrap();
        assert_eq!(latest_a.vector, vec![1.0; 4]);
    }

    #[test]
    fn test_history() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        svc.insert("a", &[1.0; 4], None).unwrap();
        svc.insert("a", &[2.0; 4], None).unwrap();
        svc.insert("a", &[3.0; 4], None).unwrap();

        let hist = svc.history("a");
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0].vector, vec![1.0; 4]);
        assert_eq!(hist[2].vector, vec![3.0; 4]);
    }

    #[test]
    fn test_audit_log() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        svc.insert("a", &[1.0; 4], None).unwrap();
        svc.delete("a").unwrap();

        let log = svc.audit_log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].operation, AuditOp::Upsert);
        assert_eq!(log[1].operation, AuditOp::Delete);
    }

    #[test]
    fn test_search_at_version() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();

        svc.insert("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        svc.insert("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        let v1 = svc.current_version();

        svc.delete("b").unwrap();
        let v2 = svc.current_version();

        let results_v1 = svc.search_at(&[1.0, 0.0, 0.0, 0.0], 10, v1).unwrap();
        assert_eq!(results_v1.len(), 2);

        let results_v2 = svc.search_at(&[1.0, 0.0, 0.0, 0.0], 10, v2).unwrap();
        assert_eq!(results_v2.len(), 1);
    }

    #[test]
    fn test_version_trimming() {
        let db = test_db();
        let config = TimeTravelConfig {
            max_versions_per_key: 3,
            ..Default::default()
        };
        let mut svc = TimeTravelService::new(&db, "test", config).unwrap();

        for i in 0..10 {
            svc.insert("a", &[i as f32; 4], None).unwrap();
        }

        let hist = svc.history("a");
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_get_nonexistent() {
        let db = test_db();
        let svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();
        assert!(svc.get_at("nonexistent", 999).unwrap().is_none());
    }
}
