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
use crate::error::Result;

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
    /// Retention window in seconds. Versions older than this are eligible for pruning.
    /// None means no time-based pruning (only count-based limits apply).
    pub retention_window_seconds: Option<u64>,
    /// Automatically prune expired snapshots when creating new ones.
    pub auto_prune: bool,
}

impl Default for TimeTravelConfig {
    fn default() -> Self {
        Self {
            max_versions_per_key: 100,
            max_snapshots: 1000,
            auto_snapshot: true,
            snapshot_interval: 100,
            retention_window_seconds: None,
            auto_prune: true,
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
    pub fn new(db: &'a Database, collection: &str, config: TimeTravelConfig) -> Result<Self> {
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
    pub fn insert(&mut self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<Version> {
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
            coll.delete(id)?;
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
        coll.delete(id)?;

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
            let da = dist_fn.compute(query, &a.vector).unwrap_or(f32::MAX);
            let db_dist = dist_fn.compute(query, &b.vector).unwrap_or(f32::MAX);
            da.partial_cmp(&db_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
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
            let at_from = history.range(..=from).next_back().map(|(_, r)| r.clone());
            let at_to = history.range(..=to).next_back().map(|(_, r)| r.clone());

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
            let record = history
                .range(..=target_version)
                .next_back()
                .map(|(_, r)| r.clone());
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

        self.record_audit(
            self.current,
            AuditOp::Rollback,
            &format!("v{target_version}"),
        );
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

    /// Search at the version that was active at the given Unix timestamp.
    ///
    /// Finds the latest version committed before `unix_timestamp` and
    /// searches against that historical state.
    pub fn search_at_timestamp(
        &self,
        query: &[f32],
        k: usize,
        unix_timestamp: u64,
    ) -> Result<Vec<crate::collection::SearchResult>> {
        // Find the latest version that was committed at or before the timestamp
        let target_version = self
            .audit_log
            .iter()
            .filter(|entry| entry.timestamp <= unix_timestamp)
            .map(|entry| entry.version)
            .max()
            .unwrap_or(0);

        if target_version == 0 {
            return Ok(Vec::new());
        }

        self.search_at(query, k, target_version)
    }

    /// Get the state of a vector at the given Unix timestamp.
    pub fn get_at_timestamp(
        &self,
        id: &str,
        unix_timestamp: u64,
    ) -> Result<Option<VersionedVector>> {
        let target_version = self
            .audit_log
            .iter()
            .filter(|entry| entry.timestamp <= unix_timestamp && entry.vector_id == id)
            .map(|entry| entry.version)
            .max()
            .unwrap_or(0);

        if target_version == 0 {
            return Ok(None);
        }

        self.get_at(id, target_version)
    }

    /// Generate a changelog between two timestamps.
    pub fn changelog(
        &self,
        from_timestamp: u64,
        to_timestamp: u64,
    ) -> Vec<&AuditEntry> {
        self.audit_log
            .iter()
            .filter(|entry| entry.timestamp >= from_timestamp && entry.timestamp <= to_timestamp)
            .collect()
    }

    /// Garbage collect old versions to reclaim memory.
    ///
    /// Removes all version history entries older than `min_version`,
    /// keeping only the latest version for each vector.
    /// Returns the number of version entries removed.
    pub fn gc(&mut self, min_version: Version) -> usize {
        let mut removed = 0;

        for history in self.versions.values_mut() {
            let before = history.len();
            // Keep the latest version and any version >= min_version
            if history.len() > 1 {
                let latest = history.last().cloned();
                history.retain(|v| v.version >= min_version);
                // Always keep at least the latest version
                if history.is_empty() {
                    if let Some(latest) = latest {
                        history.push(latest);
                    }
                }
            }
            removed += before - history.len();
        }

        // Also trim audit log
        let before_audit = self.audit_log.len();
        self.audit_log.retain(|entry| entry.version >= min_version);
        removed += before_audit - self.audit_log.len();

        // Trim old snapshots
        let before_snap = self.snapshots.len();
        self.snapshots.retain(|s| s.version >= min_version);
        removed += before_snap - self.snapshots.len();

        removed
    }

    /// Get a detailed diff between two versions, including vector-level changes.
    pub fn detailed_diff(&self, from: Version, to: Version) -> DetailedVersionDiff {
        let basic = self.diff(from, to);

        let mut vector_changes = Vec::new();
        for id in &basic.modified {
            let old_vec = self.get_at(id, from).ok().flatten();
            let new_vec = self.get_at(id, to).ok().flatten();

            if let (Some(old), Some(new)) = (old_vec, new_vec) {
                // Compute L2 distance between old and new vectors
                let l2_dist: f32 = old.vector.iter().zip(new.vector.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();

                vector_changes.push(VectorChange {
                    id: id.clone(),
                    old_version: from,
                    new_version: to,
                    l2_distance: l2_dist,
                    dimension_changes: old.vector.iter().zip(new.vector.iter())
                        .enumerate()
                        .filter(|(_, (a, b))| (a - b).abs() > f32::EPSILON)
                        .count(),
                    metadata_changed: old.metadata != new.metadata,
                });
            }
        }

        DetailedVersionDiff {
            basic,
            vector_changes,
            total_operations: self.audit_log.iter()
                .filter(|e| e.version > from && e.version <= to)
                .count(),
        }
    }

    /// Get storage statistics for the version history.
    pub fn storage_stats(&self) -> VersionStorageStats {
        let total_versions: usize = self.versions.values().map(|h| h.len()).sum();
        let total_vectors: usize = self.versions.len();
        let avg_versions_per_vector = if total_vectors > 0 {
            total_versions as f64 / total_vectors as f64
        } else {
            0.0
        };
        let max_versions_per_vector = self.versions.values()
            .map(|h| h.len())
            .max()
            .unwrap_or(0);

        VersionStorageStats {
            total_vectors,
            total_versions,
            avg_versions_per_vector,
            max_versions_per_vector,
            snapshot_count: self.snapshots.len(),
            audit_log_entries: self.audit_log.len(),
        }
    }
}

/// Detailed diff between two versions including vector-level changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedVersionDiff {
    /// Basic added/modified/deleted summary.
    pub basic: VersionDiff,
    /// Per-vector change details for modified vectors.
    pub vector_changes: Vec<VectorChange>,
    /// Total number of operations between the two versions.
    pub total_operations: usize,
}

/// Details about how a specific vector changed between versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorChange {
    /// The vector ID.
    pub id: String,
    /// Old version number.
    pub old_version: Version,
    /// New version number.
    pub new_version: Version,
    /// L2 distance between old and new vector values.
    pub l2_distance: f32,
    /// Number of dimensions that changed.
    pub dimension_changes: usize,
    /// Whether metadata changed.
    pub metadata_changed: bool,
}

/// Storage statistics for the version history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionStorageStats {
    pub total_vectors: usize,
    pub total_versions: usize,
    pub avg_versions_per_vector: f64,
    pub max_versions_per_vector: usize,
    pub snapshot_count: usize,
    pub audit_log_entries: usize,
}

// ── Restore from Snapshot ───────────────────────────────────────────────────

/// Result of a snapshot restore operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreResult {
    /// The version restored to.
    pub restored_version: Version,
    /// Number of vectors in the restored state.
    pub vector_count: usize,
    /// Number of versions that were discarded (newer than snapshot).
    pub versions_discarded: usize,
}

impl<'a> TimeTravelService<'a> {
    /// Restore the collection to the state captured in a named snapshot.
    ///
    /// This discards all changes after the snapshot version and makes
    /// the snapshot's state the current state. Creates a new version
    /// for the restore operation itself.
    pub fn restore_snapshot(&mut self, snapshot_name: &str) -> Result<RestoreResult> {
        let snapshot = self.find_snapshot(snapshot_name)
            .ok_or_else(|| NeedleError::NotFound(format!("Snapshot '{snapshot_name}'")))?
            .clone();

        let target_version = snapshot.version;
        let result = self.rollback_to(target_version)?;

        // Count discarded versions
        let versions_discarded = self.versions.values()
            .flat_map(|h| h.iter())
            .filter(|v| v.version > target_version && v.version != result)
            .count();

        // Count vectors at restored state
        let vector_count = self.versions.values()
            .filter(|h| h.iter().any(|v| v.version <= target_version && !v.deleted))
            .count();

        Ok(RestoreResult {
            restored_version: result,
            vector_count,
            versions_discarded,
        })
    }

    /// Compact history by merging consecutive versions of the same vector
    /// that have identical content (dedup).
    ///
    /// Returns the number of redundant version entries removed.
    pub fn compact_history(&mut self) -> usize {
        let mut removed = 0;

        for history in self.versions.values_mut() {
            if history.len() <= 1 {
                continue;
            }

            let mut compacted = Vec::with_capacity(history.len());
            compacted.push(history[0].clone());

            for i in 1..history.len() {
                let prev = &compacted[compacted.len() - 1];
                let curr = &history[i];

                // Keep if vector changed or deletion status changed
                let vectors_differ = prev.vector != curr.vector;
                let deletion_changed = prev.deleted != curr.deleted;
                let metadata_changed = prev.metadata != curr.metadata;

                if vectors_differ || deletion_changed || metadata_changed {
                    compacted.push(curr.clone());
                } else {
                    removed += 1;
                }
            }

            *history = compacted;
        }

        removed
    }

    /// Get the list of version numbers where a specific vector was modified.
    pub fn version_history_for(&self, id: &str) -> Vec<Version> {
        self.versions.get(id)
            .map(|h| h.iter().map(|v| v.version).collect())
            .unwrap_or_default()
    }

    /// Prune versions and snapshots older than the retention window.
    /// Returns the total number of entries removed (versions + snapshots + audit entries).
    pub fn prune_expired(&mut self) -> usize {
        let retention = match self.config.retention_window_seconds {
            Some(r) => r,
            None => return 0,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let cutoff = now.saturating_sub(retention);
        let mut removed = 0;

        // Find the latest version that's still before the cutoff
        let cutoff_version = self.audit_log
            .iter()
            .filter(|e| e.timestamp <= cutoff)
            .map(|e| e.version)
            .max()
            .unwrap_or(0);

        if cutoff_version == 0 {
            return 0;
        }

        // Prune old version entries (keep at least the latest per vector)
        for history in self.versions.values_mut() {
            let before = history.len();
            if history.len() > 1 {
                let latest = history.last().cloned();
                history.retain(|v| v.version > cutoff_version);
                if history.is_empty() {
                    if let Some(latest) = latest {
                        history.push(latest);
                    }
                }
            }
            removed += before - history.len();
        }

        // Prune old snapshots
        let before_snap = self.snapshots.len();
        self.snapshots.retain(|s| s.created_at > cutoff);
        removed += before_snap - self.snapshots.len();

        // Prune old audit entries
        let before_audit = self.audit_log.len();
        self.audit_log.retain(|e| e.timestamp > cutoff);
        removed += before_audit - self.audit_log.len();

        removed
    }

    /// Get the retention window configuration.
    pub fn retention_window(&self) -> Option<u64> {
        self.config.retention_window_seconds
    }

    /// Set the retention window (in seconds).
    pub fn set_retention_window(&mut self, seconds: Option<u64>) {
        self.config.retention_window_seconds = seconds;
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

    #[test]
    fn test_changelog() {
        let db = test_db();
        let mut svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();
        svc.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        svc.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let log = svc.changelog(0, u64::MAX);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_search_at_timestamp_empty() {
        let db = test_db();
        let svc = TimeTravelService::new(&db, "test", TimeTravelConfig::default()).unwrap();
        let results = svc.search_at_timestamp(&[1.0, 0.0, 0.0, 0.0], 5, 0).unwrap();
        assert!(results.is_empty());
    }
}
