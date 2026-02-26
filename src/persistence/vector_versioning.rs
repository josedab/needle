#![allow(dead_code)]

//! Vector Versioning & Time-Travel Queries
//!
//! Provides MVCC semantics for point-in-time vector retrieval with snapshot isolation.
//! Extends the WAL infrastructure with version timestamps and compaction/retention policies.
//!
//! # Features
//!
//! - **Version Storage Layer**: Each vector mutation is stamped with a monotonic version
//! - **MVCC Read Protocol**: Snapshot isolation reads via `as_of(timestamp)`
//! - **Compaction & Retention**: Configurable policies to prune old versions
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::persistence::vector_versioning::{VersionedStore, RetentionPolicy, VersionQuery};
//!
//! let mut store = VersionedStore::new(VersioningConfig::default());
//! store.insert("doc1", &[0.1; 128], None)?;
//! store.update("doc1", &[0.2; 128], None)?;
//!
//! // Query at a previous point in time
//! let old_version = store.get_as_of("doc1", some_past_timestamp)?;
//!
//! // Compact old versions
//! let result = store.compact(&RetentionPolicy::default())?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for vector versioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningConfig {
    /// Maximum number of versions to retain per vector.
    pub max_versions_per_vector: usize,
    /// Whether to enable automatic compaction.
    pub auto_compact: bool,
    /// Compaction trigger threshold (number of tombstones before compaction fires).
    pub compact_tombstone_threshold: usize,
    /// Default retention policy.
    pub retention: RetentionPolicy,
}

impl Default for VersioningConfig {
    fn default() -> Self {
        Self {
            max_versions_per_vector: 100,
            auto_compact: true,
            compact_tombstone_threshold: 1000,
            retention: RetentionPolicy::default(),
        }
    }
}

/// Retention policy for version compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Maximum age of versions to retain (seconds). Versions older than this are eligible for GC.
    pub max_age_seconds: u64,
    /// Minimum number of versions to always keep per vector (even if older than max_age).
    pub min_versions_keep: usize,
    /// Whether to preserve snapshot-referenced versions from compaction.
    pub preserve_snapshots: bool,
    /// Maximum total versions across all vectors before forced compaction.
    pub max_total_versions: Option<usize>,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age_seconds: 7 * 24 * 3600, // 1 week
            min_versions_keep: 1,
            preserve_snapshots: true,
            max_total_versions: Some(1_000_000),
        }
    }
}

/// A single versioned vector entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Monotonic version number.
    pub version: u64,
    /// Timestamp when this version was created.
    pub timestamp: u64,
    /// Vector data (empty for tombstones).
    pub vector: Vec<f32>,
    /// Associated metadata.
    pub metadata: Option<serde_json::Value>,
    /// Whether this entry is a deletion marker.
    pub is_tombstone: bool,
}

/// WAL entry extension for versioned operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionedWalEntry {
    /// Versioned insert with explicit version stamp.
    Insert {
        collection: String,
        id: String,
        version: u64,
        timestamp: u64,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    /// Versioned delete (tombstone).
    Delete {
        collection: String,
        id: String,
        version: u64,
        timestamp: u64,
    },
    /// Snapshot marker for MVCC read consistency.
    Snapshot {
        snapshot_id: String,
        version: u64,
        timestamp: u64,
    },
    /// Compaction completed marker.
    CompactionComplete {
        versions_removed: usize,
        oldest_retained_version: u64,
        timestamp: u64,
    },
}

/// Query specification for version-aware reads.
#[derive(Debug, Clone)]
pub enum VersionQuery {
    /// Read the latest version.
    Latest,
    /// Read the version visible at a specific timestamp.
    AsOf(u64),
    /// Read a specific version number.
    AtVersion(u64),
    /// Read versions within a time range.
    Range { start: u64, end: u64 },
}

/// Result of a compaction operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Number of versions removed.
    pub versions_removed: usize,
    /// Number of tombstones removed.
    pub tombstones_removed: usize,
    /// Number of vectors with remaining versions.
    pub vectors_remaining: usize,
    /// Total versions remaining.
    pub total_versions_remaining: usize,
    /// Duration of compaction in microseconds.
    pub duration_us: u64,
}

/// Versioned vector store with MVCC semantics.
///
/// Maintains a version chain per vector ID, enabling point-in-time reads
/// and snapshot isolation.
pub struct VersionedStore {
    config: VersioningConfig,
    /// Version chains: vector_id -> ordered versions (oldest first).
    versions: HashMap<String, VecDeque<VersionEntry>>,
    /// Snapshot registry: snapshot_id -> version at snapshot time.
    snapshots: BTreeMap<String, u64>,
    /// Protected versions (referenced by snapshots).
    protected_versions: std::collections::HashSet<u64>,
    /// Current version counter (monotonically increasing).
    current_version: u64,
    /// Total tombstone count for auto-compaction trigger.
    tombstone_count: usize,
    /// Statistics.
    stats: VersioningStats,
}

/// Versioning statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersioningStats {
    /// Total versioned vectors.
    pub total_vectors: usize,
    /// Total versions across all vectors.
    pub total_versions: usize,
    /// Total tombstones.
    pub total_tombstones: usize,
    /// Total compactions performed.
    pub compactions_performed: usize,
    /// Total versions removed by compaction.
    pub versions_compacted: usize,
    /// Current version counter.
    pub current_version: u64,
    /// Number of active snapshots.
    pub active_snapshots: usize,
}

impl VersionedStore {
    /// Create a new versioned store.
    pub fn new(config: VersioningConfig) -> Self {
        Self {
            config,
            versions: HashMap::new(),
            snapshots: BTreeMap::new(),
            protected_versions: std::collections::HashSet::new(),
            current_version: 0,
            tombstone_count: 0,
            stats: VersioningStats::default(),
        }
    }

    /// Get the next version number.
    fn next_version(&mut self) -> u64 {
        self.current_version += 1;
        self.current_version
    }

    /// Get current timestamp.
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Insert or update a vector, creating a new version.
    pub fn put(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<u64> {
        let version = self.next_version();
        let timestamp = Self::now();

        let entry = VersionEntry {
            version,
            timestamp,
            vector: vector.to_vec(),
            metadata,
            is_tombstone: false,
        };

        let chain = self.versions.entry(id.to_string()).or_default();
        chain.push_back(entry);

        // Enforce per-vector version limit
        while chain.len() > self.config.max_versions_per_vector {
            if let Some(removed) = chain.pop_front() {
                if removed.is_tombstone {
                    self.tombstone_count = self.tombstone_count.saturating_sub(1);
                }
            }
        }

        self.update_stats();
        self.maybe_auto_compact()?;

        Ok(version)
    }

    /// Delete a vector by creating a tombstone version.
    pub fn delete(&mut self, id: &str) -> Result<u64> {
        if !self.versions.contains_key(id) {
            return Err(NeedleError::VectorNotFound(id.to_string()));
        }

        let version = self.next_version();
        let timestamp = Self::now();

        let entry = VersionEntry {
            version,
            timestamp,
            vector: Vec::new(),
            metadata: None,
            is_tombstone: true,
        };

        if let Some(chain) = self.versions.get_mut(id) {
            chain.push_back(entry);
        }

        self.tombstone_count += 1;
        self.update_stats();
        self.maybe_auto_compact()?;

        Ok(version)
    }

    /// Get the version of a vector visible at a specific timestamp (MVCC read).
    pub fn get_as_of(&self, id: &str, timestamp: u64) -> Option<&VersionEntry> {
        let chain = self.versions.get(id)?;
        // Find the latest version created at or before the timestamp that isn't a tombstone
        chain
            .iter()
            .rev()
            .find(|v| v.timestamp <= timestamp && !v.is_tombstone)
    }

    /// Get the latest version of a vector.
    /// Returns None if the vector has been deleted (last entry is a tombstone).
    pub fn get_latest(&self, id: &str) -> Option<&VersionEntry> {
        let chain = self.versions.get(id)?;
        let last = chain.back()?;
        // If the most recent entry is a tombstone, the vector is deleted
        if last.is_tombstone {
            return None;
        }
        Some(last)
    }

    /// Get a specific version of a vector.
    pub fn get_at_version(&self, id: &str, version: u64) -> Option<&VersionEntry> {
        let chain = self.versions.get(id)?;
        chain.iter().find(|v| v.version == version)
    }

    /// Query a vector using a VersionQuery specification.
    pub fn query(&self, id: &str, query: &VersionQuery) -> Option<Vec<&VersionEntry>> {
        let chain = self.versions.get(id)?;

        match query {
            VersionQuery::Latest => self.get_latest(id).map(|v| vec![v]),
            VersionQuery::AsOf(ts) => self.get_as_of(id, *ts).map(|v| vec![v]),
            VersionQuery::AtVersion(ver) => self.get_at_version(id, *ver).map(|v| vec![v]),
            VersionQuery::Range { start, end } => {
                let results: Vec<&VersionEntry> = chain
                    .iter()
                    .filter(|v| v.timestamp >= *start && v.timestamp <= *end)
                    .collect();
                if results.is_empty() {
                    None
                } else {
                    Some(results)
                }
            }
        }
    }

    /// Get full version history for a vector.
    pub fn history(&self, id: &str) -> Option<Vec<&VersionEntry>> {
        self.versions.get(id).map(|chain| chain.iter().collect())
    }

    /// Create a named snapshot at the current version.
    pub fn create_snapshot(&mut self, name: &str) -> u64 {
        let version = self.current_version;
        self.snapshots.insert(name.to_string(), version);
        self.protected_versions.insert(version);
        self.stats.active_snapshots = self.snapshots.len();
        version
    }

    /// Delete a snapshot, potentially allowing its referenced versions to be compacted.
    pub fn delete_snapshot(&mut self, name: &str) -> Result<()> {
        let version = self
            .snapshots
            .remove(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Snapshot '{}' not found", name)))?;

        // Only unprotect if no other snapshot references this version
        if !self.snapshots.values().any(|&v| v == version) {
            self.protected_versions.remove(&version);
        }

        self.stats.active_snapshots = self.snapshots.len();
        Ok(())
    }

    /// Get all vector IDs visible at a given timestamp.
    pub fn visible_ids_at(&self, timestamp: u64) -> Vec<String> {
        self.versions
            .iter()
            .filter_map(|(id, _)| {
                self.get_as_of(id, timestamp).map(|_| id.clone())
            })
            .collect()
    }

    /// Compact old versions according to the given retention policy.
    pub fn compact(&mut self, policy: &RetentionPolicy) -> Result<CompactionResult> {
        let start = std::time::Instant::now();
        let now = Self::now();
        let cutoff = now.saturating_sub(policy.max_age_seconds);
        let mut versions_removed = 0;
        let mut tombstones_removed = 0;

        for chain in self.versions.values_mut() {
            let keep_count = policy.min_versions_keep.max(1);
            if chain.len() <= keep_count {
                continue;
            }

            let removable_count = chain.len() - keep_count;
            let mut to_remove = Vec::new();

            for (i, entry) in chain.iter().enumerate() {
                if i >= removable_count {
                    break;
                }
                // Only remove versions older than cutoff
                if entry.timestamp >= cutoff {
                    continue;
                }
                // Don't remove snapshot-protected versions
                if policy.preserve_snapshots
                    && self.protected_versions.contains(&entry.version)
                {
                    continue;
                }
                to_remove.push(i);
            }

            // Remove in reverse order
            for i in to_remove.into_iter().rev() {
                if let Some(removed) = chain.remove(i) {
                    if removed.is_tombstone {
                        tombstones_removed += 1;
                    } else {
                        versions_removed += 1;
                    }
                }
            }
        }

        // Remove empty chains
        self.versions.retain(|_, chain| !chain.is_empty());

        self.tombstone_count = self.tombstone_count.saturating_sub(tombstones_removed);
        self.stats.compactions_performed += 1;
        self.stats.versions_compacted += versions_removed + tombstones_removed;
        self.update_stats();

        Ok(CompactionResult {
            versions_removed,
            tombstones_removed,
            vectors_remaining: self.versions.len(),
            total_versions_remaining: self
                .versions
                .values()
                .map(|c| c.len())
                .sum(),
            duration_us: start.elapsed().as_micros() as u64,
        })
    }

    /// Trigger auto-compaction if thresholds are exceeded.
    fn maybe_auto_compact(&mut self) -> Result<()> {
        if !self.config.auto_compact {
            return Ok(());
        }
        if self.tombstone_count >= self.config.compact_tombstone_threshold {
            self.compact(&self.config.retention.clone())?;
        }
        Ok(())
    }

    /// Update internal statistics.
    fn update_stats(&mut self) {
        self.stats.total_vectors = self.versions.len();
        self.stats.total_versions = self.versions.values().map(|c| c.len()).sum();
        self.stats.total_tombstones = self.tombstone_count;
        self.stats.current_version = self.current_version;
    }

    /// Get current statistics.
    pub fn stats(&self) -> &VersioningStats {
        &self.stats
    }

    /// Get all vectors visible at a given snapshot, returning (id, vector, metadata).
    /// This provides snapshot isolation: you see exactly the state at that timestamp.
    pub fn snapshot_read(&self, timestamp: u64) -> Vec<(String, Vec<f32>, Option<serde_json::Value>)> {
        self.versions
            .iter()
            .filter_map(|(id, _)| {
                self.get_as_of(id, timestamp).map(|entry| {
                    (id.clone(), entry.vector.clone(), entry.metadata.clone())
                })
            })
            .collect()
    }

    /// Create a versioned search builder for fluent as_of queries.
    pub fn search_builder(&self) -> VersionedSearchBuilder<'_> {
        VersionedSearchBuilder::new(self)
    }

    /// Get the diff between two timestamps: inserted, updated, and deleted vector IDs.
    pub fn diff_between(&self, from_ts: u64, to_ts: u64) -> VersionDiff {
        let mut diff = VersionDiff::default();
        for (id, chain) in &self.versions {
            let was_visible = self.get_as_of(id, from_ts);
            let is_visible = self.get_as_of(id, to_ts);

            match (was_visible, is_visible) {
                (None, Some(_)) => diff.inserted.push(id.clone()),
                (Some(_), None) => diff.deleted.push(id.clone()),
                (Some(old), Some(new)) if old.version != new.version => {
                    diff.updated.push(id.clone());
                }
                _ => {}
            }
        }
        diff
    }

    /// Estimate storage overhead from versioning.
    pub fn storage_overhead(&self) -> VersioningOverhead {
        let mut total_vector_bytes: u64 = 0;
        let mut latest_vector_bytes: u64 = 0;

        for chain in self.versions.values() {
            for entry in chain {
                let bytes = (entry.vector.len() * 4) as u64;
                total_vector_bytes += bytes;
            }
            if let Some(last) = chain.back() {
                latest_vector_bytes += (last.vector.len() * 4) as u64;
            }
        }

        VersioningOverhead {
            total_bytes: total_vector_bytes,
            latest_only_bytes: latest_vector_bytes,
            overhead_ratio: if latest_vector_bytes > 0 {
                total_vector_bytes as f64 / latest_vector_bytes as f64
            } else {
                1.0
            },
        }
    }
}

/// Diff between two timestamps.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionDiff {
    /// Vector IDs inserted between the two timestamps.
    pub inserted: Vec<String>,
    /// Vector IDs updated between the two timestamps.
    pub updated: Vec<String>,
    /// Vector IDs deleted between the two timestamps.
    pub deleted: Vec<String>,
}

/// Storage overhead from versioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersioningOverhead {
    /// Total bytes used by all versions.
    pub total_bytes: u64,
    /// Bytes that would be used if only latest versions were kept.
    pub latest_only_bytes: u64,
    /// Overhead ratio (total / latest_only). 1.0 = no overhead.
    pub overhead_ratio: f64,
}

/// Fluent builder for versioned searches.
pub struct VersionedSearchBuilder<'a> {
    store: &'a VersionedStore,
    query_type: VersionQuery,
}

impl<'a> VersionedSearchBuilder<'a> {
    fn new(store: &'a VersionedStore) -> Self {
        Self {
            store,
            query_type: VersionQuery::Latest,
        }
    }

    /// Set the query to read as of a specific timestamp.
    #[must_use]
    pub fn as_of(mut self, timestamp: u64) -> Self {
        self.query_type = VersionQuery::AsOf(timestamp);
        self
    }

    /// Set the query to read a specific version.
    #[must_use]
    pub fn at_version(mut self, version: u64) -> Self {
        self.query_type = VersionQuery::AtVersion(version);
        self
    }

    /// Set the query to read the latest version.
    #[must_use]
    pub fn latest(mut self) -> Self {
        self.query_type = VersionQuery::Latest;
        self
    }

    /// Set the query to a time range.
    #[must_use]
    pub fn range(mut self, start: u64, end: u64) -> Self {
        self.query_type = VersionQuery::Range { start, end };
        self
    }

    /// Execute the query for a specific vector ID.
    pub fn get(&self, id: &str) -> Option<Vec<&VersionEntry>> {
        self.store.query(id, &self.query_type)
    }

    /// Get all visible vectors at the configured point in time.
    /// Returns (id, vector, metadata) tuples.
    pub fn get_all(&self) -> Vec<(String, Vec<f32>, Option<serde_json::Value>)> {
        match &self.query_type {
            VersionQuery::Latest => {
                self.store
                    .versions
                    .iter()
                    .filter_map(|(id, _)| {
                        self.store.get_latest(id).map(|e| {
                            (id.clone(), e.vector.clone(), e.metadata.clone())
                        })
                    })
                    .collect()
            }
            VersionQuery::AsOf(ts) => self.store.snapshot_read(*ts),
            VersionQuery::AtVersion(ver) => {
                self.store
                    .versions
                    .iter()
                    .filter_map(|(id, _)| {
                        self.store.get_at_version(id, *ver).map(|e| {
                            (id.clone(), e.vector.clone(), e.metadata.clone())
                        })
                    })
                    .collect()
            }
            VersionQuery::Range { .. } => {
                // For range, return latest within the range
                self.store
                    .versions
                    .iter()
                    .filter_map(|(id, _)| {
                        self.store.query(id, &self.query_type).and_then(|entries| {
                            entries.last().map(|e| {
                                (id.clone(), e.vector.clone(), e.metadata.clone())
                            })
                        })
                    })
                    .collect()
            }
        }
    }
}

/// Provenance record tracking the full lifecycle of a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// ID of the vector this record belongs to
    pub vector_id: String,
    /// Source document or data origin
    pub source_document: Option<String>,
    /// Embedding model used to generate this vector
    pub embedding_model: Option<String>,
    /// Model version
    pub model_version: Option<String>,
    /// Pipeline ID that produced this vector
    pub pipeline_id: Option<String>,
    /// Unix timestamp of creation
    pub created_at: u64,
    /// Unix timestamp of last update
    pub updated_at: u64,
    /// Parent vector ID (if derived from another vector)
    pub parent_vector_id: Option<String>,
    /// Additional provenance metadata
    pub extra: Option<serde_json::Value>,
}

impl ProvenanceRecord {
    /// Create a new provenance record with minimal info
    pub fn new(vector_id: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            vector_id: vector_id.into(),
            source_document: None,
            embedding_model: None,
            model_version: None,
            pipeline_id: None,
            created_at: now,
            updated_at: now,
            parent_vector_id: None,
            extra: None,
        }
    }

    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source_document = Some(source.into());
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>, version: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self.model_version = Some(version.into());
        self
    }

    #[must_use]
    pub fn with_pipeline(mut self, pipeline_id: impl Into<String>) -> Self {
        self.pipeline_id = Some(pipeline_id.into());
        self
    }

    #[must_use]
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_vector_id = Some(parent_id.into());
        self
    }

    #[must_use]
    pub fn with_extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = Some(extra);
        self
    }
}

/// In-memory provenance store keyed by vector ID.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProvenanceStore {
    records: HashMap<String, ProvenanceRecord>,
}

impl ProvenanceStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, record: ProvenanceRecord) {
        self.records.insert(record.vector_id.clone(), record);
    }

    pub fn get(&self, vector_id: &str) -> Option<&ProvenanceRecord> {
        self.records.get(vector_id)
    }

    pub fn remove(&mut self, vector_id: &str) -> Option<ProvenanceRecord> {
        self.records.remove(vector_id)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Find all vectors derived from a source document
    pub fn find_by_source(&self, source: &str) -> Vec<&ProvenanceRecord> {
        self.records
            .values()
            .filter(|r| r.source_document.as_deref() == Some(source))
            .collect()
    }

    /// Find all vectors produced by a specific model
    pub fn find_by_model(&self, model: &str) -> Vec<&ProvenanceRecord> {
        self.records
            .values()
            .filter(|r| r.embedding_model.as_deref() == Some(model))
            .collect()
    }

    /// Find all vectors derived from a parent vector
    pub fn find_children(&self, parent_id: &str) -> Vec<&ProvenanceRecord> {
        self.records
            .values()
            .filter(|r| r.parent_vector_id.as_deref() == Some(parent_id))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_versioned_put_and_get() {
        let mut store = VersionedStore::new(VersioningConfig::default());

        let v1 = store.put("doc1", &[1.0, 2.0], None).unwrap();
        let v2 = store.put("doc1", &[3.0, 4.0], None).unwrap();

        assert!(v2 > v1);

        let latest = store.get_latest("doc1").unwrap();
        assert_eq!(latest.vector, vec![3.0, 4.0]);
        assert_eq!(latest.version, v2);

        let at_v1 = store.get_at_version("doc1", v1).unwrap();
        assert_eq!(at_v1.vector, vec![1.0, 2.0]);
    }

    #[test]
    fn test_delete_creates_tombstone() {
        let mut store = VersionedStore::new(VersioningConfig::default());

        store.put("doc1", &[1.0, 2.0], None).unwrap();
        store.delete("doc1").unwrap();

        assert!(store.get_latest("doc1").is_none());
        let history = store.history("doc1").unwrap();
        assert_eq!(history.len(), 2);
        assert!(history[1].is_tombstone);
    }

    #[test]
    fn test_snapshot_and_query() {
        let mut store = VersionedStore::new(VersioningConfig::default());

        store.put("doc1", &[1.0, 2.0], None).unwrap();
        let snap_ver = store.create_snapshot("snap1");

        store.put("doc1", &[3.0, 4.0], None).unwrap();

        let at_snap = store.get_at_version("doc1", snap_ver);
        // snap_ver is the version counter at snapshot time, not a vector version
        // The vector was at version 1 when snap was created at version 1
        let history = store.history("doc1").unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_compaction() {
        let config = VersioningConfig {
            max_versions_per_vector: 1000,
            ..Default::default()
        };
        let mut store = VersionedStore::new(config);

        // Insert many versions
        for i in 0..10 {
            store.put("doc1", &[i as f32], None).unwrap();
        }

        // Set all version timestamps to the past so compaction can remove them
        if let Some(chain) = store.versions.get_mut("doc1") {
            for entry in chain.iter_mut() {
                entry.timestamp = 0; // epoch - definitely old
            }
        }

        let policy = RetentionPolicy {
            max_age_seconds: 1, // 1 second - everything at timestamp 0 is old enough
            min_versions_keep: 2,
            preserve_snapshots: false,
            max_total_versions: None,
        };

        let result = store.compact(&policy).unwrap();
        assert_eq!(result.versions_removed, 8);
        assert_eq!(store.history("doc1").unwrap().len(), 2);
    }

    #[test]
    fn test_version_query_range() {
        let mut store = VersionedStore::new(VersioningConfig::default());

        store.put("doc1", &[1.0], None).unwrap();
        store.put("doc1", &[2.0], None).unwrap();
        store.put("doc1", &[3.0], None).unwrap();

        let all = store
            .query("doc1", &VersionQuery::Range { start: 0, end: u64::MAX })
            .unwrap();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut store = VersionedStore::new(VersioningConfig::default());
        assert!(store.delete("nonexistent").is_err());
    }

    #[test]
    fn test_provenance_record_builder() {
        let record = ProvenanceRecord::new("vec1")
            .with_source("doc_abc")
            .with_model("text-embedding-3-small", "1.0")
            .with_pipeline("pipeline_42")
            .with_parent("vec0");
        assert_eq!(record.vector_id, "vec1");
        assert_eq!(record.source_document.as_deref(), Some("doc_abc"));
        assert_eq!(record.embedding_model.as_deref(), Some("text-embedding-3-small"));
        assert_eq!(record.model_version.as_deref(), Some("1.0"));
        assert_eq!(record.pipeline_id.as_deref(), Some("pipeline_42"));
        assert_eq!(record.parent_vector_id.as_deref(), Some("vec0"));
    }

    #[test]
    fn test_provenance_store_crud() {
        let mut store = ProvenanceStore::new();
        assert!(store.is_empty());

        store.insert(ProvenanceRecord::new("v1").with_source("doc1"));
        store.insert(ProvenanceRecord::new("v2").with_source("doc1"));
        store.insert(ProvenanceRecord::new("v3").with_source("doc2").with_model("embed", "1.0"));
        assert_eq!(store.len(), 3);

        // Get
        let r = store.get("v1").unwrap();
        assert_eq!(r.source_document.as_deref(), Some("doc1"));

        // Find by source
        let by_doc1 = store.find_by_source("doc1");
        assert_eq!(by_doc1.len(), 2);

        // Find by model
        let by_model = store.find_by_model("embed");
        assert_eq!(by_model.len(), 1);

        // Remove
        store.remove("v1");
        assert_eq!(store.len(), 2);
        assert!(store.get("v1").is_none());
    }

    #[test]
    fn test_provenance_find_children() {
        let mut store = ProvenanceStore::new();
        store.insert(ProvenanceRecord::new("parent"));
        store.insert(ProvenanceRecord::new("child1").with_parent("parent"));
        store.insert(ProvenanceRecord::new("child2").with_parent("parent"));
        store.insert(ProvenanceRecord::new("other").with_parent("different"));

        let children = store.find_children("parent");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_snapshot_read() {
        let mut store = VersionedStore::new(VersioningConfig::default());
        let _v1 = store.put("a", &[1.0], None).expect("ok");
        let _v2 = store.put("b", &[2.0], None).expect("ok");

        // Snapshot read at future time should see both
        let all = store.snapshot_read(u64::MAX);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_search_builder_as_of() {
        let mut store = VersionedStore::new(VersioningConfig::default());
        store.put("doc1", &[1.0], None).expect("ok");
        store.put("doc2", &[2.0], None).expect("ok");

        let results = store.search_builder().as_of(u64::MAX).get_all();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_builder_latest() {
        let mut store = VersionedStore::new(VersioningConfig::default());
        store.put("doc1", &[1.0], None).expect("ok");
        store.put("doc1", &[2.0], None).expect("ok");

        let builder = store.search_builder().latest();
        let results = builder.get("doc1");
        assert!(results.is_some());
        let entries = results.expect("entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].vector, vec![2.0]);
    }

    #[test]
    fn test_diff_between() {
        let mut store = VersionedStore::new(VersioningConfig::default());

        // All inserted after ts=0
        store.put("a", &[1.0], None).expect("ok");
        store.put("b", &[2.0], None).expect("ok");

        let diff = store.diff_between(0, u64::MAX);
        assert_eq!(diff.inserted.len(), 2);
        assert!(diff.updated.is_empty());
        assert!(diff.deleted.is_empty());
    }

    #[test]
    fn test_storage_overhead() {
        let mut store = VersionedStore::new(VersioningConfig::default());
        store.put("doc1", &[1.0, 2.0, 3.0], None).expect("ok");
        store.put("doc1", &[4.0, 5.0, 6.0], None).expect("ok");

        let overhead = store.storage_overhead();
        // 2 versions of 3 floats = 24 bytes total, 12 latest only
        assert_eq!(overhead.total_bytes, 24);
        assert_eq!(overhead.latest_only_bytes, 12);
        assert!((overhead.overhead_ratio - 2.0).abs() < 0.01);
    }
}
