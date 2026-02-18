//! Snapshot-Based Time Travel
//!
//! Point-in-time read-only database views using version counters and
//! CoW-style snapshot chains. Each write increments the version; snapshots
//! capture the state at a specific version for later query.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::snapshot_time_travel::{
//!     VersionedStore, VersionSnapshot, VersionQuery,
//! };
//!
//! let mut store = VersionedStore::new();
//!
//! // Insert at version 1
//! store.insert("doc1", vec![1.0; 4], None);
//! let v1 = store.current_version();
//!
//! // Modify at version 2
//! store.insert("doc2", vec![2.0; 4], None);
//!
//! // Time travel to version 1
//! let snapshot = store.at_version(v1).unwrap();
//! assert!(snapshot.contains("doc1"));
//! assert!(!snapshot.contains("doc2"));
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

// ── Version Entry ────────────────────────────────────────────────────────────

/// A single versioned operation in the log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionOp {
    /// Insert a vector.
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Update a vector.
    Update {
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Delete a vector.
    Delete { id: String },
}

/// A timestamped version entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Version number.
    pub version: u64,
    /// Operation performed.
    pub operation: VersionOp,
    /// Timestamp.
    pub timestamp: u64,
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// A read-only snapshot at a specific version.
#[derive(Debug, Clone)]
pub struct VersionSnapshot {
    /// Version this snapshot represents.
    pub version: u64,
    /// Vectors present at this version.
    vectors: HashMap<String, (Vec<f32>, Option<Value>)>,
}

impl VersionSnapshot {
    /// Check if a vector ID exists in this snapshot.
    pub fn contains(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<(&[f32], Option<&Value>)> {
        self.vectors.get(id).map(|(v, m)| (v.as_slice(), m.as_ref()))
    }

    /// Number of vectors in this snapshot.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the snapshot is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// List all vector IDs.
    pub fn ids(&self) -> Vec<&str> {
        self.vectors.keys().map(|s| s.as_str()).collect()
    }

    /// Search by brute-force distance (for snapshot queries).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut distances: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, (vec, _))| {
                let dist = cosine_distance(query, vec);
                (id.clone(), dist)
            })
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
}

// ── Version Query ────────────────────────────────────────────────────────────

/// Query against a specific version.
#[derive(Debug, Clone)]
pub struct VersionQuery {
    /// Target version (None = latest).
    pub version: Option<u64>,
    /// Vector ID to retrieve.
    pub id: Option<String>,
    /// Search query vector.
    pub search: Option<Vec<f32>>,
    /// Number of results for search.
    pub k: usize,
}

// ── Version Diff ─────────────────────────────────────────────────────────────

/// Diff between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    /// From version.
    pub from: u64,
    /// To version.
    pub to: u64,
    /// IDs added.
    pub added: Vec<String>,
    /// IDs removed.
    pub removed: Vec<String>,
    /// IDs modified.
    pub modified: Vec<String>,
}

// ── Retention Policy ─────────────────────────────────────────────────────────

/// How long to retain version history.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum versions to retain.
    pub max_versions: usize,
    /// Maximum age in seconds.
    pub max_age_secs: Option<u64>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_versions: 10_000,
            max_age_secs: None,
        }
    }
}

// ── Versioned Store ──────────────────────────────────────────────────────────

/// Versioned vector store with time-travel support.
pub struct VersionedStore {
    /// Current state.
    current: HashMap<String, (Vec<f32>, Option<Value>)>,
    /// Version log (append-only).
    log: Vec<VersionEntry>,
    /// Current version counter.
    version: u64,
    /// Retention policy.
    retention: RetentionPolicy,
}

impl VersionedStore {
    /// Create a new versioned store.
    pub fn new() -> Self {
        Self {
            current: HashMap::new(),
            log: Vec::new(),
            version: 0,
            retention: RetentionPolicy::default(),
        }
    }

    /// Create with a retention policy.
    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    /// Insert a vector, creating a new version.
    pub fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: Option<Value>) -> u64 {
        self.version += 1;
        self.log.push(VersionEntry {
            version: self.version,
            operation: VersionOp::Insert {
                id: id.into(),
                vector: vector.clone(),
                metadata: metadata.clone(),
            },
            timestamp: now_secs(),
        });
        self.current.insert(id.into(), (vector, metadata));
        self.enforce_retention();
        self.version
    }

    /// Update a vector, creating a new version.
    pub fn update(&mut self, id: &str, vector: Vec<f32>, metadata: Option<Value>) -> Option<u64> {
        if !self.current.contains_key(id) {
            return None;
        }
        self.version += 1;
        self.log.push(VersionEntry {
            version: self.version,
            operation: VersionOp::Update {
                id: id.into(),
                vector: vector.clone(),
                metadata: metadata.clone(),
            },
            timestamp: now_secs(),
        });
        self.current.insert(id.into(), (vector, metadata));
        Some(self.version)
    }

    /// Delete a vector, creating a new version.
    pub fn delete(&mut self, id: &str) -> Option<u64> {
        if self.current.remove(id).is_none() {
            return None;
        }
        self.version += 1;
        self.log.push(VersionEntry {
            version: self.version,
            operation: VersionOp::Delete { id: id.into() },
            timestamp: now_secs(),
        });
        Some(self.version)
    }

    /// Get the current version number.
    pub fn current_version(&self) -> u64 {
        self.version
    }

    /// Get a read-only snapshot at a specific version.
    pub fn at_version(&self, version: u64) -> Result<VersionSnapshot> {
        if version > self.version {
            return Err(NeedleError::NotFound(format!("Version {version} does not exist (current: {})", self.version)));
        }
        let min_version = self.log.first().map_or(0, |e| e.version);
        if version < min_version && min_version > 0 {
            return Err(NeedleError::NotFound(format!(
                "Version {version} has been compacted (earliest: {min_version})"
            )));
        }

        // Replay log up to target version
        let mut state: HashMap<String, (Vec<f32>, Option<Value>)> = HashMap::new();
        for entry in &self.log {
            if entry.version > version {
                break;
            }
            match &entry.operation {
                VersionOp::Insert { id, vector, metadata } | VersionOp::Update { id, vector, metadata } => {
                    state.insert(id.clone(), (vector.clone(), metadata.clone()));
                }
                VersionOp::Delete { id } => {
                    state.remove(id);
                }
            }
        }

        Ok(VersionSnapshot { version, vectors: state })
    }

    /// Compute diff between two versions.
    pub fn diff(&self, from: u64, to: u64) -> Result<VersionDiff> {
        let snap_from = self.at_version(from)?;
        let snap_to = self.at_version(to)?;

        let from_ids: std::collections::HashSet<&str> = snap_from.ids().into_iter().collect();
        let to_ids: std::collections::HashSet<&str> = snap_to.ids().into_iter().collect();

        let added: Vec<String> = to_ids.difference(&from_ids).map(|s| s.to_string()).collect();
        let removed: Vec<String> = from_ids.difference(&to_ids).map(|s| s.to_string()).collect();
        let modified: Vec<String> = from_ids
            .intersection(&to_ids)
            .filter(|id| {
                let Some((v1, _)) = snap_from.get(id) else { return false };
                let Some((v2, _)) = snap_to.get(id) else { return false };
                v1 != v2
            })
            .map(|s| s.to_string())
            .collect();

        Ok(VersionDiff { from, to, added, removed, modified })
    }

    /// Current vector count.
    pub fn len(&self) -> usize {
        self.current.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty()
    }

    /// Version log length.
    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    /// Find the version that was current at a given Unix timestamp.
    pub fn version_at_timestamp(&self, timestamp: u64) -> Option<u64> {
        self.log
            .iter()
            .rev()
            .find(|entry| entry.timestamp <= timestamp)
            .map(|entry| entry.version)
    }

    /// Search at a given Unix timestamp (NeedleQL AT TIMESTAMP support).
    pub fn search_at_timestamp(
        &self,
        query: &[f32],
        k: usize,
        timestamp: u64,
    ) -> Result<Vec<(String, f32)>> {
        let version = self.version_at_timestamp(timestamp)
            .ok_or_else(|| NeedleError::InvalidArgument(
                format!("No data at timestamp {timestamp}")
            ))?;

        let snap = self.at_version(version)?;
        Ok(snap.search(query, k))
    }

    /// Parse and execute a NeedleQL-style AT TIMESTAMP query.
    /// Format: "AT TIMESTAMP <unix_epoch>"
    pub fn parse_at_timestamp(clause: &str) -> Option<u64> {
        let clause = clause.trim().to_uppercase();
        if let Some(ts_str) = clause.strip_prefix("AT TIMESTAMP") {
            ts_str.trim().trim_matches('\'').trim_matches('"').parse().ok()
        } else {
            None
        }
    }

    /// Get a summary of all versions with timestamps.
    pub fn version_timeline(&self) -> Vec<(u64, u64, String)> {
        self.log
            .iter()
            .map(|entry| {
                let op_type = match &entry.operation {
                    VersionOp::Insert { id, .. } => format!("insert:{id}"),
                    VersionOp::Update { id, .. } => format!("update:{id}"),
                    VersionOp::Delete { id } => format!("delete:{id}"),
                };
                (entry.version, entry.timestamp, op_type)
            })
            .collect()
    }

    fn enforce_retention(&mut self) {
        // Enforce max versions
        while self.log.len() > self.retention.max_versions {
            self.log.remove(0);
        }

        // Enforce max age
        if let Some(max_age) = self.retention.max_age_secs {
            let cutoff = now_secs().saturating_sub(max_age);
            self.log.retain(|entry| entry.timestamp >= cutoff);
        }
    }

    /// Manually prune snapshots older than a given timestamp.
    pub fn prune_before(&mut self, timestamp: u64) -> usize {
        let before = self.log.len();
        self.log.retain(|entry| entry.timestamp >= timestamp);
        before - self.log.len()
    }

    /// Prune snapshots keeping only the last N versions.
    pub fn prune_keep_last(&mut self, n: usize) -> usize {
        if self.log.len() <= n {
            return 0;
        }
        let to_remove = self.log.len() - n;
        self.log.drain(..to_remove);
        to_remove
    }

    /// Search at a specific version number.
    pub fn search_at_version(
        &self,
        query: &[f32],
        k: usize,
        version: u64,
    ) -> Result<Vec<(String, f32)>> {
        let snap = self.at_version(version)?;
        Ok(snap.search(query, k))
    }

    /// Parse an ISO 8601 or RFC 3339 date string to a Unix timestamp.
    pub fn parse_datetime(datetime: &str) -> Option<u64> {
        // Try parsing as ISO 8601 via chrono
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(datetime) {
            return Some(dt.timestamp() as u64);
        }
        // Try parsing as "YYYY-MM-DD"
        if let Ok(dt) = chrono::NaiveDate::parse_from_str(datetime, "%Y-%m-%d") {
            return dt.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc().timestamp() as u64);
        }
        // Try as raw Unix timestamp
        datetime.parse().ok()
    }

    /// Get retention policy.
    pub fn retention(&self) -> &RetentionPolicy {
        &self.retention
    }

    /// Set retention policy.
    pub fn set_retention(&mut self, policy: RetentionPolicy) {
        self.retention = policy;
        self.enforce_retention();
    }
}

impl Default for VersionedStore {
    fn default() -> Self {
        Self::new()
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_version() {
        let mut store = VersionedStore::new();
        let v1 = store.insert("a", vec![1.0; 4], None);
        assert_eq!(v1, 1);
        let v2 = store.insert("b", vec![2.0; 4], None);
        assert_eq!(v2, 2);
        assert_eq!(store.current_version(), 2);
    }

    #[test]
    fn test_time_travel() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0; 4], None);
        let v1 = store.current_version();
        store.insert("b", vec![2.0; 4], None);
        store.delete("a");

        let snap = store.at_version(v1).unwrap();
        assert!(snap.contains("a"));
        assert!(!snap.contains("b"));
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn test_snapshot_search() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0, 0.0, 0.0, 0.0], None);
        store.insert("b", vec![0.0, 1.0, 0.0, 0.0], None);

        let snap = store.at_version(store.current_version()).unwrap();
        let results = snap.search(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_diff() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0; 4], None);
        let v1 = store.current_version();
        store.insert("b", vec![2.0; 4], None);
        store.delete("a");
        let v2 = store.current_version();

        let diff = store.diff(v1, v2).unwrap();
        assert!(diff.added.contains(&"b".to_string()));
        assert!(diff.removed.contains(&"a".to_string()));
    }

    #[test]
    fn test_update_versioning() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0; 4], None);
        let v1 = store.current_version();
        store.update("a", vec![9.0; 4], None);
        let v2 = store.current_version();

        let snap1 = store.at_version(v1).unwrap();
        let snap2 = store.at_version(v2).unwrap();
        let (vec1, _) = snap1.get("a").unwrap();
        let (vec2, _) = snap2.get("a").unwrap();
        assert!((vec1[0] - 1.0).abs() < 0.01);
        assert!((vec2[0] - 9.0).abs() < 0.01);
    }

    #[test]
    fn test_version_not_found() {
        let store = VersionedStore::new();
        assert!(store.at_version(999).is_err());
    }

    #[test]
    fn test_retention() {
        let mut store = VersionedStore::new().with_retention(RetentionPolicy {
            max_versions: 3,
            max_age_secs: None,
        });
        for i in 0..10 {
            store.insert(&format!("v{i}"), vec![i as f32; 4], None);
        }
        assert!(store.log_len() <= 3);
    }

    #[test]
    fn test_search_at_timestamp() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0, 0.0, 0.0, 0.0], None);
        // All entries get `now_secs()` as timestamp, so searching at current time should work
        let ts = now_secs();
        let results = store.search_at_timestamp(&[1.0, 0.0, 0.0, 0.0], 5, ts).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_parse_at_timestamp() {
        assert_eq!(VersionedStore::parse_at_timestamp("AT TIMESTAMP 1700000000"), Some(1700000000));
        assert_eq!(VersionedStore::parse_at_timestamp("at timestamp '1700000000'"), Some(1700000000));
        assert_eq!(VersionedStore::parse_at_timestamp("invalid"), None);
    }

    #[test]
    fn test_version_timeline() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0; 4], None);
        store.insert("b", vec![2.0; 4], None);
        store.delete("a");

        let timeline = store.version_timeline();
        assert_eq!(timeline.len(), 3);
        assert!(timeline[0].2.starts_with("insert:"));
        assert!(timeline[2].2.starts_with("delete:"));
    }

    #[test]
    fn test_prune_keep_last() {
        let mut store = VersionedStore::new();
        for i in 0..10 {
            store.insert(&format!("v{i}"), vec![i as f32; 4], None);
        }
        let pruned = store.prune_keep_last(3);
        assert_eq!(pruned, 7);
        assert_eq!(store.log_len(), 3);
    }

    #[test]
    fn test_prune_before_timestamp() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0; 4], None);
        store.insert("b", vec![2.0; 4], None);
        // All entries have `now` timestamp, so pruning before future = 0
        let pruned = store.prune_before(now_secs() + 100);
        assert_eq!(pruned, 2); // all entries are before future
    }

    #[test]
    fn test_search_at_version() {
        let mut store = VersionedStore::new();
        store.insert("a", vec![1.0, 0.0, 0.0, 0.0], None);
        let v1 = store.current_version();
        store.insert("b", vec![0.0, 1.0, 0.0, 0.0], None);
        store.delete("a");

        // Search at v1 should find "a" but not "b"
        let results = store.search_at_version(&[1.0, 0.0, 0.0, 0.0], 5, v1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_parse_datetime() {
        // Unix timestamp
        assert_eq!(VersionedStore::parse_datetime("1700000000"), Some(1700000000));
        // ISO 8601 date
        assert!(VersionedStore::parse_datetime("2026-01-15").is_some());
        // Invalid
        assert!(VersionedStore::parse_datetime("not-a-date").is_none());
    }

    #[test]
    fn test_age_based_retention() {
        let mut store = VersionedStore::new().with_retention(RetentionPolicy {
            max_versions: 10_000,
            max_age_secs: Some(0), // expire immediately
        });
        store.insert("a", vec![1.0; 4], None);
        // After insertion with max_age=0, the entry may get pruned on next insert
        store.insert("b", vec![2.0; 4], None);
        // At least the latest should survive
        assert!(store.log_len() >= 1);
    }

    #[test]
    fn test_set_retention() {
        let mut store = VersionedStore::new();
        for i in 0..10 {
            store.insert(&format!("v{i}"), vec![i as f32; 4], None);
        }
        assert_eq!(store.log_len(), 10);

        store.set_retention(RetentionPolicy {
            max_versions: 3,
            max_age_secs: None,
        });
        assert!(store.log_len() <= 3);
    }
}
