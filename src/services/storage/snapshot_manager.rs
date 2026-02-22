#![allow(clippy::unwrap_used)]
//! Snapshot Manager
//!
//! Named snapshot management layer on top of VersionedStore: create named
//! snapshots, query at specific snapshot points, diff between snapshots,
//! and enforce retention policies.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::snapshot_manager::{
//!     SnapshotManager, NamedSnapshot, SnapshotQuery,
//! };
//!
//! let mut mgr = SnapshotManager::new(4); // 4 dimensions
//!
//! mgr.insert("doc1", vec![1.0; 4], None);
//! let snap_v1 = mgr.create_snapshot("v1.0").unwrap();
//!
//! mgr.insert("doc2", vec![2.0; 4], None);
//! let snap_v2 = mgr.create_snapshot("v2.0").unwrap();
//!
//! // Query at v1 — should only find doc1
//! let results = mgr.search_at_snapshot("v1.0", &[1.0; 4], 5).unwrap();
//! assert_eq!(results.len(), 1);
//!
//! // Diff between snapshots
//! let diff = mgr.diff_snapshots("v1.0", "v2.0").unwrap();
//! assert!(diff.added.contains(&"doc2".to_string()));
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

/// A named snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedSnapshot {
    pub name: String,
    pub version: u64,
    pub created_at: u64,
    pub vector_count: usize,
    pub description: Option<String>,
}

/// Diff between two snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotDiff {
    pub from_snapshot: String,
    pub to_snapshot: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub modified: Vec<String>,
}

/// A search result from a snapshot query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSearchResult {
    pub id: String,
    pub distance: f32,
    pub snapshot_name: String,
}

/// Retention policy for snapshots.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub max_snapshots: usize,
    pub max_age_secs: Option<u64>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self { max_snapshots: 100, max_age_secs: None }
    }
}

/// Versioned vector entry.
#[derive(Debug, Clone)]
struct VectorEntry {
    vector: Vec<f32>,
    metadata: Option<Value>,
    created_at_version: u64,
    deleted_at_version: Option<u64>,
}

/// Snapshot manager with named snapshots and time-travel queries.
pub struct SnapshotManager {
    dimensions: usize,
    vectors: HashMap<String, Vec<VectorEntry>>,
    snapshots: HashMap<String, NamedSnapshot>,
    snapshot_order: Vec<String>,
    current_version: u64,
    retention: RetentionPolicy,
}

impl SnapshotManager {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            vectors: HashMap::new(),
            snapshots: HashMap::new(),
            snapshot_order: Vec::new(),
            current_version: 0,
            retention: RetentionPolicy::default(),
        }
    }

    /// Set retention policy.
    #[must_use]
    pub fn with_retention(mut self, policy: RetentionPolicy) -> Self {
        self.retention = policy;
        self
    }

    /// Insert a vector (advances version).
    pub fn insert(&mut self, id: &str, vector: Vec<f32>, metadata: Option<Value>) {
        self.current_version += 1;
        self.vectors
            .entry(id.to_string())
            .or_default()
            .push(VectorEntry {
                vector,
                metadata,
                created_at_version: self.current_version,
                deleted_at_version: None,
            });
    }

    /// Delete a vector (marks as deleted at current version).
    pub fn delete(&mut self, id: &str) -> bool {
        if let Some(entries) = self.vectors.get_mut(id) {
            if let Some(last) = entries.last_mut() {
                if last.deleted_at_version.is_none() {
                    self.current_version += 1;
                    last.deleted_at_version = Some(self.current_version);
                    return true;
                }
            }
        }
        false
    }

    /// Create a named snapshot at the current version.
    pub fn create_snapshot(&mut self, name: &str) -> Result<NamedSnapshot> {
        if self.snapshots.contains_key(name) {
            return Err(NeedleError::Conflict(format!("Snapshot '{name}' already exists")));
        }

        let count = self.count_at_version(self.current_version);
        let snap = NamedSnapshot {
            name: name.to_string(),
            version: self.current_version,
            created_at: now_secs(),
            vector_count: count,
            description: None,
        };

        self.snapshots.insert(name.to_string(), snap.clone());
        self.snapshot_order.push(name.to_string());
        self.enforce_retention();
        Ok(snap)
    }

    /// Search at a named snapshot.
    pub fn search_at_snapshot(
        &self,
        snapshot_name: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SnapshotSearchResult>> {
        let snap = self.snapshots.get(snapshot_name)
            .ok_or_else(|| NeedleError::NotFound(format!("Snapshot '{snapshot_name}'")))?;

        let mut results: Vec<(String, f32)> = Vec::new();
        for (id, entries) in &self.vectors {
            if let Some(vec) = Self::vector_at_version(entries, snap.version) {
                let dist = cosine_distance(query, vec);
                results.push((id.clone(), dist));
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        Ok(results
            .into_iter()
            .map(|(id, distance)| SnapshotSearchResult {
                id,
                distance,
                snapshot_name: snapshot_name.to_string(),
            })
            .collect())
    }

    /// Diff between two named snapshots.
    pub fn diff_snapshots(&self, from: &str, to: &str) -> Result<SnapshotDiff> {
        let snap_from = self.snapshots.get(from)
            .ok_or_else(|| NeedleError::NotFound(format!("Snapshot '{from}'")))?;
        let snap_to = self.snapshots.get(to)
            .ok_or_else(|| NeedleError::NotFound(format!("Snapshot '{to}'")))?;

        let ids_from = self.ids_at_version(snap_from.version);
        let ids_to = self.ids_at_version(snap_to.version);

        let added: Vec<String> = ids_to.iter().filter(|id| !ids_from.contains(*id)).cloned().collect();
        let removed: Vec<String> = ids_from.iter().filter(|id| !ids_to.contains(*id)).cloned().collect();
        let modified: Vec<String> = ids_from
            .iter()
            .filter(|id| {
                ids_to.contains(*id) && {
                    let v1 = self.vectors.get(*id).and_then(|e| Self::vector_at_version(e, snap_from.version));
                    let v2 = self.vectors.get(*id).and_then(|e| Self::vector_at_version(e, snap_to.version));
                    match (v1, v2) {
                        (Some(a), Some(b)) => a != b,
                        _ => false,
                    }
                }
            })
            .cloned()
            .collect();

        Ok(SnapshotDiff {
            from_snapshot: from.to_string(),
            to_snapshot: to.to_string(),
            added, removed, modified,
        })
    }

    /// List all snapshots.
    pub fn list_snapshots(&self) -> Vec<&NamedSnapshot> {
        self.snapshot_order
            .iter()
            .filter_map(|name| self.snapshots.get(name))
            .collect()
    }

    /// Delete a snapshot.
    pub fn delete_snapshot(&mut self, name: &str) -> bool {
        if self.snapshots.remove(name).is_some() {
            self.snapshot_order.retain(|n| n != name);
            true
        } else {
            false
        }
    }

    /// Current version.
    pub fn current_version(&self) -> u64 {
        self.current_version
    }

    /// Count of live vectors at current version.
    pub fn len(&self) -> usize {
        self.count_at_version(self.current_version)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn count_at_version(&self, version: u64) -> usize {
        self.vectors
            .values()
            .filter(|entries| Self::vector_at_version(entries, version).is_some())
            .count()
    }

    fn ids_at_version(&self, version: u64) -> Vec<String> {
        self.vectors
            .iter()
            .filter(|(_, entries)| Self::vector_at_version(entries, version).is_some())
            .map(|(id, _)| id.clone())
            .collect()
    }

    fn vector_at_version<'a>(entries: &'a [VectorEntry], version: u64) -> Option<&'a Vec<f32>> {
        entries
            .iter()
            .rev()
            .find(|e| {
                e.created_at_version <= version
                    && e.deleted_at_version.map_or(true, |del| del > version)
            })
            .map(|e| &e.vector)
    }

    fn enforce_retention(&mut self) {
        while self.snapshot_order.len() > self.retention.max_snapshots {
            if let Some(oldest) = self.snapshot_order.first().cloned() {
                self.snapshots.remove(&oldest);
                self.snapshot_order.remove(0);
            }
        }

        if let Some(max_age) = self.retention.max_age_secs {
            let cutoff = now_secs().saturating_sub(max_age);
            let expired: Vec<String> = self
                .snapshots
                .iter()
                .filter(|(_, s)| s.created_at < cutoff)
                .map(|(name, _)| name.clone())
                .collect();
            for name in expired {
                self.snapshots.remove(&name);
                self.snapshot_order.retain(|n| n != &name);
            }
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON { return 1.0; }
    1.0 - dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_snapshot() {
        let mut mgr = SnapshotManager::new(4);
        mgr.insert("a", vec![1.0; 4], None);
        let snap = mgr.create_snapshot("v1").unwrap();
        assert_eq!(snap.vector_count, 1);
    }

    #[test]
    fn test_time_travel_search() {
        let mut mgr = SnapshotManager::new(4);
        mgr.insert("a", vec![1.0, 0.0, 0.0, 0.0], None);
        mgr.create_snapshot("v1").unwrap();

        mgr.insert("b", vec![0.0, 1.0, 0.0, 0.0], None);
        mgr.create_snapshot("v2").unwrap();

        // Search at v1 — only "a"
        let r1 = mgr.search_at_snapshot("v1", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r1[0].id, "a");

        // Search at v2 — both
        let r2 = mgr.search_at_snapshot("v2", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(r2.len(), 2);
    }

    #[test]
    fn test_diff_snapshots() {
        let mut mgr = SnapshotManager::new(4);
        mgr.insert("a", vec![1.0; 4], None);
        mgr.create_snapshot("v1").unwrap();

        mgr.insert("b", vec![2.0; 4], None);
        mgr.delete("a");
        mgr.create_snapshot("v2").unwrap();

        let diff = mgr.diff_snapshots("v1", "v2").unwrap();
        assert!(diff.added.contains(&"b".to_string()));
        assert!(diff.removed.contains(&"a".to_string()));
    }

    #[test]
    fn test_duplicate_snapshot_name() {
        let mut mgr = SnapshotManager::new(4);
        mgr.create_snapshot("v1").unwrap();
        assert!(mgr.create_snapshot("v1").is_err());
    }

    #[test]
    fn test_delete_snapshot() {
        let mut mgr = SnapshotManager::new(4);
        mgr.create_snapshot("v1").unwrap();
        assert!(mgr.delete_snapshot("v1"));
        assert_eq!(mgr.list_snapshots().len(), 0);
    }

    #[test]
    fn test_retention_policy() {
        let mut mgr = SnapshotManager::new(4)
            .with_retention(RetentionPolicy { max_snapshots: 3, max_age_secs: None });

        for i in 0..5 {
            mgr.insert(&format!("v{i}"), vec![i as f32; 4], None);
            mgr.create_snapshot(&format!("snap_{i}")).unwrap();
        }

        assert!(mgr.list_snapshots().len() <= 3);
    }

    #[test]
    fn test_snapshot_not_found() {
        let mgr = SnapshotManager::new(4);
        assert!(mgr.search_at_snapshot("nonexistent", &[1.0; 4], 5).is_err());
    }

    #[test]
    fn test_delete_and_snapshot() {
        let mut mgr = SnapshotManager::new(4);
        mgr.insert("a", vec![1.0; 4], None);
        mgr.create_snapshot("before_delete").unwrap();

        mgr.delete("a");
        mgr.create_snapshot("after_delete").unwrap();

        let r1 = mgr.search_at_snapshot("before_delete", &[1.0; 4], 5).unwrap();
        assert_eq!(r1.len(), 1);

        let r2 = mgr.search_at_snapshot("after_delete", &[1.0; 4], 5).unwrap();
        assert_eq!(r2.len(), 0);
    }
}
