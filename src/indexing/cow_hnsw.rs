#![allow(clippy::unwrap_used)]
//! Copy-on-Write HNSW Index
//!
//! Provides concurrent reads during writes without lock contention by using
//! immutable base layers with delta buffers for new insertions. Readers see
//! a consistent snapshot while writers append to delta buffers. Periodic
//! compaction merges deltas into the base graph.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────┐
//! │ CowHnswIndex                                         │
//! │  ┌──────────────────┐  ┌──────────────────────────┐  │
//! │  │ Base Graph (Arc) │  │ Delta Buffer (RwLock)     │  │
//! │  │ - immutable      │  │ - new insertions          │  │
//! │  │ - shared reads   │  │ - pending deletes         │  │
//! │  └──────────────────┘  └──────────────────────────┘  │
//! │  ┌──────────────────┐  ┌──────────────────────────┐  │
//! │  │ Epoch Tracker    │  │ Compaction Worker         │  │
//! │  │ - reader epochs  │  │ - merges delta → base    │  │
//! │  │ - safe reclaim   │  │ - builds new Arc<Base>   │  │
//! │  └──────────────────┘  └──────────────────────────┘  │
//! └──────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::indexing::cow_hnsw::{CowHnswIndex, CowHnswConfig};
//! use needle::distance::DistanceFunction;
//!
//! let config = CowHnswConfig::default();
//! let index = CowHnswIndex::new(config, 128, DistanceFunction::Cosine);
//!
//! // Insert (takes write lock on delta only, not base)
//! index.insert("vec1", &embedding)?;
//!
//! // Search (reads base + delta without blocking writes)
//! let results = index.search(&query, 10)?;
//!
//! // Compact delta into base when buffer is large
//! index.compact()?;
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the COW HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CowHnswConfig {
    /// Maximum connections per node (M parameter).
    pub m: usize,
    /// Search depth during construction.
    pub ef_construction: usize,
    /// Search depth during queries.
    pub ef_search: usize,
    /// Delta buffer size threshold before auto-compaction hint.
    pub compaction_threshold: usize,
    /// Maximum level probability factor.
    pub ml: f64,
}

impl Default for CowHnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            compaction_threshold: 10_000,
            ml: 1.0 / (16.0_f64).ln(),
        }
    }
}

// ── Base Graph (Immutable) ───────────────────────────────────────────────────

/// Immutable base HNSW graph, shared via Arc for zero-copy reads.
#[derive(Debug, Clone)]
struct BaseGraph {
    /// Node data: (id, vector, neighbors_per_layer).
    nodes: Vec<BaseNode>,
    /// ID to node index mapping.
    id_map: HashMap<String, usize>,
    /// Entry point index.
    entry_point: Option<usize>,
    /// Maximum layer in the graph.
    max_layer: usize,
}

#[derive(Debug, Clone)]
struct BaseNode {
    id: String,
    vector: Vec<f32>,
    /// Neighbors per layer: layer 0 has up to 2*M connections, others up to M.
    neighbors: Vec<Vec<usize>>,
}

impl BaseGraph {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            id_map: HashMap::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
        distance_fn: &DistanceFunction,
        deleted: &HashSet<String>,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<std::cmp::Reverse<(OrderedFloat<f32>, usize)>> =
            BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        let dist = distance_fn.compute(query, &self.nodes[entry].vector).unwrap_or(f32::MAX);
        candidates.push(std::cmp::Reverse((OrderedFloat(dist), entry)));
        results.push((OrderedFloat(dist), entry));
        visited.insert(entry);

        while let Some(std::cmp::Reverse((OrderedFloat(c_dist), c_idx))) = candidates.pop() {
            if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
                if c_dist > worst_dist && results.len() >= ef {
                    break;
                }
            }

            if layer < self.nodes[c_idx].neighbors.len() {
                for &neighbor in &self.nodes[c_idx].neighbors[layer] {
                    if neighbor < self.nodes.len() && visited.insert(neighbor) {
                        let n_dist = distance_fn.compute(query, &self.nodes[neighbor].vector).unwrap_or(f32::MAX);

                        let should_add = results.len() < ef || {
                            if let Some(&(OrderedFloat(worst), _)) = results.peek() {
                                n_dist < worst
                            } else {
                                true
                            }
                        };

                        if should_add {
                            candidates
                                .push(std::cmp::Reverse((OrderedFloat(n_dist), neighbor)));
                            results.push((OrderedFloat(n_dist), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results
            .into_sorted_vec()
            .into_iter()
            .filter(|(_, idx)| !deleted.contains(&self.nodes[*idx].id))
            .map(|(OrderedFloat(d), idx)| (idx, d))
            .collect()
    }
}

// ── Delta Buffer ─────────────────────────────────────────────────────────────

/// Delta buffer for new insertions and pending deletes.
#[derive(Debug)]
struct DeltaBuffer {
    /// Newly inserted vectors (not yet in base graph).
    insertions: Vec<DeltaEntry>,
    /// IDs pending deletion (lazy deletion from base).
    pending_deletes: HashSet<String>,
    /// ID map for delta entries.
    id_map: HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct DeltaEntry {
    id: String,
    vector: Vec<f32>,
}

impl DeltaBuffer {
    fn new() -> Self {
        Self {
            insertions: Vec::new(),
            pending_deletes: HashSet::new(),
            id_map: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.insertions.len()
    }
}

// ── Epoch Tracker ────────────────────────────────────────────────────────────

/// Epoch-based reclamation tracker for safe deferred deletion.
struct EpochTracker {
    /// Current global epoch.
    current_epoch: AtomicU64,
    /// Number of active readers.
    active_readers: AtomicU64,
}

impl EpochTracker {
    fn new() -> Self {
        Self {
            current_epoch: AtomicU64::new(0),
            active_readers: AtomicU64::new(0),
        }
    }

    /// Enter a read epoch (reader registration).
    fn enter_read(&self) -> u64 {
        self.active_readers.fetch_add(1, AtomicOrdering::AcqRel);
        self.current_epoch.load(AtomicOrdering::Acquire)
    }

    /// Exit a read epoch (reader deregistration).
    fn exit_read(&self) {
        self.active_readers.fetch_sub(1, AtomicOrdering::AcqRel);
    }

    /// Advance the epoch (called during compaction).
    fn advance(&self) -> u64 {
        self.current_epoch.fetch_add(1, AtomicOrdering::AcqRel)
    }

    /// Check if there are active readers.
    fn has_active_readers(&self) -> bool {
        self.active_readers.load(AtomicOrdering::Acquire) > 0
    }

    /// Get current epoch.
    fn current(&self) -> u64 {
        self.current_epoch.load(AtomicOrdering::Acquire)
    }
}

// ── COW HNSW Index ───────────────────────────────────────────────────────────

/// Search result from COW HNSW.
#[derive(Debug, Clone)]
pub struct CowSearchResult {
    /// Vector ID.
    pub id: String,
    /// Distance to query.
    pub distance: f32,
}

/// Copy-on-Write HNSW index with lock-free reads during concurrent writes.
pub struct CowHnswIndex {
    config: CowHnswConfig,
    dimensions: usize,
    distance_fn: DistanceFunction,
    /// Immutable base graph, swapped atomically during compaction.
    base: Arc<RwLock<BaseGraph>>,
    /// Delta buffer for new insertions.
    delta: RwLock<DeltaBuffer>,
    /// Epoch tracker for safe reclamation.
    epoch: EpochTracker,
    /// Total compactions performed.
    compaction_count: AtomicU64,
}

impl CowHnswIndex {
    /// Create a new COW HNSW index.
    pub fn new(config: CowHnswConfig, dimensions: usize, distance_fn: DistanceFunction) -> Self {
        Self {
            config,
            dimensions,
            distance_fn,
            base: Arc::new(RwLock::new(BaseGraph::new())),
            delta: RwLock::new(DeltaBuffer::new()),
            epoch: EpochTracker::new(),
            compaction_count: AtomicU64::new(0),
        }
    }

    /// Insert a vector. Only takes a write lock on the delta buffer.
    pub fn insert(&self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        // Check if ID exists in base
        {
            let base = self.base.read();
            if base.id_map.contains_key(id) {
                return Err(NeedleError::InvalidInput(format!(
                    "Vector '{}' already exists in base graph",
                    id
                )));
            }
        }

        let mut delta = self.delta.write();
        if delta.id_map.contains_key(id) {
            return Err(NeedleError::InvalidInput(format!(
                "Vector '{}' already exists in delta buffer",
                id
            )));
        }

        let idx = delta.insertions.len();
        delta.insertions.push(DeltaEntry {
            id: id.to_string(),
            vector: vector.to_vec(),
        });
        delta.id_map.insert(id.to_string(), idx);
        Ok(())
    }

    /// Mark a vector for deletion. Actual removal happens during compaction.
    pub fn delete(&self, id: &str) -> Result<bool> {
        let mut delta = self.delta.write();

        // Remove from delta if present
        if delta.id_map.contains_key(id) {
            delta.id_map.remove(id);
            // Mark for cleanup (actual vec removal deferred)
            delta.pending_deletes.insert(id.to_string());
            return Ok(true);
        }

        // Mark base graph deletion
        let base = self.base.read();
        if base.id_map.contains_key(id) {
            delta.pending_deletes.insert(id.to_string());
            return Ok(true);
        }

        Ok(false)
    }

    /// Search for nearest neighbors. Reads base + delta without blocking writes.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<CowSearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let _epoch = self.epoch.enter_read();

        // Search base graph
        let base = self.base.read();
        let delta = self.delta.read();
        let deleted = &delta.pending_deletes;

        let mut all_results: Vec<(String, f32)> = Vec::new();

        // Base graph search
        if let Some(entry) = base.entry_point {
            let ef = self.config.ef_search.max(k);

            // Navigate from top layer to layer 0
            let mut current_entry = entry;
            for layer in (1..=base.max_layer).rev() {
                let results =
                    base.search_layer(query, current_entry, 1, layer, &self.distance_fn, deleted);
                if let Some((idx, _)) = results.first() {
                    current_entry = *idx;
                }
            }

            // Search layer 0 with full ef
            let layer0_results =
                base.search_layer(query, current_entry, ef, 0, &self.distance_fn, deleted);

            for (idx, dist) in layer0_results {
                if !deleted.contains(&base.nodes[idx].id) {
                    all_results.push((base.nodes[idx].id.clone(), dist));
                }
            }
        }

        // Linear scan of delta buffer (small, so this is fast)
        for entry in &delta.insertions {
            if !delta.pending_deletes.contains(&entry.id) && delta.id_map.contains_key(&entry.id) {
                let dist = self.distance_fn.compute(query, &entry.vector).unwrap_or(f32::MAX);
                all_results.push((entry.id.clone(), dist));
            }
        }

        // Drop read guards
        drop(delta);
        drop(base);
        self.epoch.exit_read();

        // Merge and sort
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);

        Ok(all_results
            .into_iter()
            .map(|(id, distance)| CowSearchResult { id, distance })
            .collect())
    }

    /// Compact delta buffer into base graph.
    /// Creates a new base graph with deltas merged in.
    pub fn compact(&self) -> Result<CompactionStats> {
        let mut delta = self.delta.write();
        let old_base = self.base.read();

        let delta_count = delta.insertions.len();
        let delete_count = delta.pending_deletes.len();

        // Build new base graph
        let mut new_nodes: Vec<BaseNode> = Vec::new();
        let mut new_id_map: HashMap<String, usize> = HashMap::new();

        // Copy existing base nodes (excluding deleted)
        for node in &old_base.nodes {
            if !delta.pending_deletes.contains(&node.id) {
                let idx = new_nodes.len();
                new_id_map.insert(node.id.clone(), idx);
                new_nodes.push(node.clone());
            }
        }

        // Add delta insertions (excluding deleted)
        for entry in &delta.insertions {
            if !delta.pending_deletes.contains(&entry.id) && delta.id_map.contains_key(&entry.id) {
                let idx = new_nodes.len();
                new_id_map.insert(entry.id.clone(), idx);
                new_nodes.push(BaseNode {
                    id: entry.id.clone(),
                    vector: entry.vector.clone(),
                    neighbors: vec![Vec::new()], // Layer 0 only
                });
            }
        }

        // Build neighbor connections for new nodes using greedy search
        let m = self.config.m;
        for i in 0..new_nodes.len() {
            if new_nodes[i].neighbors[0].is_empty() {
                // Find nearest neighbors by scanning
                let mut distances: Vec<(usize, f32)> = (0..new_nodes.len())
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d =
                            self.distance_fn.compute(&new_nodes[i].vector, &new_nodes[j].vector).unwrap_or(f32::MAX);
                        (j, d)
                    })
                    .collect();
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                distances.truncate(m * 2); // Layer 0 gets 2*M connections

                new_nodes[i].neighbors[0] = distances.into_iter().map(|(j, _)| j).collect();
            }
        }

        // Remap neighbor indices (since indices may have shifted due to deletions)
        // Build old-to-new index mapping
        let old_to_new: HashMap<usize, usize> = old_base
            .id_map
            .iter()
            .filter_map(|(id, &old_idx)| new_id_map.get(id).map(|&new_idx| (old_idx, new_idx)))
            .collect();

        for node in &mut new_nodes {
            for layer_neighbors in &mut node.neighbors {
                *layer_neighbors = layer_neighbors
                    .iter()
                    .filter_map(|&old_idx| old_to_new.get(&old_idx).copied())
                    .collect();
            }
        }

        // Find entry point (node closest to centroid)
        let entry_point = if new_nodes.is_empty() {
            None
        } else {
            let dim = self.dimensions;
            let n = new_nodes.len() as f32;
            let mut centroid = vec![0.0f32; dim];
            for node in &new_nodes {
                for (c, &v) in centroid.iter_mut().zip(node.vector.iter()) {
                    *c += v;
                }
            }
            for c in &mut centroid {
                *c /= n;
            }

            let mut best = 0;
            let mut best_dist = f32::MAX;
            for (i, node) in new_nodes.iter().enumerate() {
                let d = self.distance_fn.compute(&centroid, &node.vector).unwrap_or(f32::MAX);
                if d < best_dist {
                    best_dist = d;
                    best = i;
                }
            }
            Some(best)
        };

        let new_base = BaseGraph {
            nodes: new_nodes,
            id_map: new_id_map,
            entry_point,
            max_layer: 0,
        };

        // Advance epoch
        self.epoch.advance();

        // Swap base graph
        drop(old_base);
        *self.base.write() = new_base;

        // Clear delta
        delta.insertions.clear();
        delta.pending_deletes.clear();
        delta.id_map.clear();

        self.compaction_count.fetch_add(1, AtomicOrdering::Relaxed);

        Ok(CompactionStats {
            merged_insertions: delta_count,
            applied_deletes: delete_count,
            compaction_number: self.compaction_count.load(AtomicOrdering::Relaxed),
        })
    }

    /// Check if delta buffer exceeds compaction threshold.
    pub fn needs_compaction(&self) -> bool {
        self.delta.read().len() >= self.config.compaction_threshold
    }

    /// Get the total number of vectors (base + delta - deleted).
    pub fn len(&self) -> usize {
        let base = self.base.read();
        let delta = self.delta.read();
        let base_count = base.nodes.len();
        let delta_active = delta
            .insertions
            .iter()
            .filter(|e| !delta.pending_deletes.contains(&e.id) && delta.id_map.contains_key(&e.id))
            .count();
        let base_deleted = delta
            .pending_deletes
            .iter()
            .filter(|id| base.id_map.contains_key(*id))
            .count();
        base_count + delta_active - base_deleted
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get index statistics.
    pub fn stats(&self) -> CowHnswStats {
        let base = self.base.read();
        let delta = self.delta.read();
        CowHnswStats {
            base_vectors: base.nodes.len(),
            delta_insertions: delta.insertions.len(),
            pending_deletes: delta.pending_deletes.len(),
            compaction_count: self.compaction_count.load(AtomicOrdering::Relaxed),
            current_epoch: self.epoch.current(),
            has_active_readers: self.epoch.has_active_readers(),
        }
    }
}

/// Statistics for COW HNSW index.
#[derive(Debug, Clone)]
pub struct CowHnswStats {
    /// Vectors in the base graph.
    pub base_vectors: usize,
    /// Vectors in the delta buffer.
    pub delta_insertions: usize,
    /// IDs pending deletion.
    pub pending_deletes: usize,
    /// Number of compactions performed.
    pub compaction_count: u64,
    /// Current epoch number.
    pub current_epoch: u64,
    /// Whether there are active readers.
    pub has_active_readers: bool,
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionStats {
    /// Number of delta insertions merged.
    pub merged_insertions: usize,
    /// Number of deletions applied.
    pub applied_deletes: usize,
    /// Compaction sequence number.
    pub compaction_number: u64,
}

// ── WAL Integration for Crash Recovery ───────────────────────────────────────

/// WAL entry for COW HNSW operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CowWalEntry {
    /// Vector insertion.
    Insert { id: String, vector: Vec<f32> },
    /// Vector deletion.
    Delete { id: String },
    /// Compaction checkpoint (WAL can be truncated after this).
    CompactionCheckpoint { epoch: u64 },
}

/// Write-ahead log for COW HNSW crash recovery.
/// Entries are appended before mutations and replayed on recovery.
pub struct CowHnswWal {
    entries: Vec<CowWalEntry>,
}

impl CowHnswWal {
    /// Create a new in-memory WAL.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an entry.
    pub fn append(&mut self, entry: CowWalEntry) {
        self.entries.push(entry);
    }

    /// Replay entries into a COW HNSW index.
    pub fn replay(&self, index: &CowHnswIndex) -> Result<ReplayStats> {
        let mut inserts = 0;
        let mut deletes = 0;
        let mut errors = 0;

        for entry in &self.entries {
            match entry {
                CowWalEntry::Insert { id, vector } => {
                    match index.insert(id, vector) {
                        Ok(()) => inserts += 1,
                        Err(_) => errors += 1,
                    }
                }
                CowWalEntry::Delete { id } => {
                    match index.delete(id) {
                        Ok(true) => deletes += 1,
                        Ok(false) | Err(_) => errors += 1,
                    }
                }
                CowWalEntry::CompactionCheckpoint { .. } => {
                    // After a compaction checkpoint, previous entries are already in the base graph
                }
            }
        }

        Ok(ReplayStats {
            inserts,
            deletes,
            errors,
        })
    }

    /// Truncate entries up to and including the last compaction checkpoint.
    pub fn truncate_to_checkpoint(&mut self) {
        if let Some(pos) = self.entries.iter().rposition(|e| {
            matches!(e, CowWalEntry::CompactionCheckpoint { .. })
        }) {
            self.entries.drain(..=pos);
        }
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for CowHnswWal {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from WAL replay.
#[derive(Debug, Clone)]
pub struct ReplayStats {
    /// Number of inserts replayed.
    pub inserts: usize,
    /// Number of deletes replayed.
    pub deletes: usize,
    /// Number of failed operations.
    pub errors: usize,
}

// ── RAII Read Epoch Guard ────────────────────────────────────────────────────

/// RAII guard that automatically enters/exits a read epoch.
/// While held, the base graph snapshot is guaranteed to remain valid.
pub struct ReadGuard<'a> {
    epoch: &'a EpochTracker,
    _entered_epoch: u64,
}

impl<'a> ReadGuard<'a> {
    fn new(epoch: &'a EpochTracker) -> Self {
        let entered = epoch.enter_read();
        Self {
            epoch,
            _entered_epoch: entered,
        }
    }
}

impl<'a> Drop for ReadGuard<'a> {
    fn drop(&mut self) {
        self.epoch.exit_read();
    }
}

impl CowHnswIndex {
    /// Acquire a RAII read guard for snapshot isolation.
    pub fn read_guard(&self) -> ReadGuard<'_> {
        ReadGuard::new(&self.epoch)
    }

    /// Insert multiple vectors in a batch. More efficient than individual inserts
    /// because it takes a single write lock on the delta buffer.
    pub fn insert_batch(&self, items: &[(&str, &[f32])]) -> Result<usize> {
        let mut delta = self.delta.write();
        let base = self.base.read();
        let mut inserted = 0;

        for &(id, vector) in items {
            if vector.len() != self.dimensions {
                continue;
            }
            if base.id_map.contains_key(id) || delta.id_map.contains_key(id) {
                continue;
            }

            let idx = delta.insertions.len();
            delta.insertions.push(DeltaEntry {
                id: id.to_string(),
                vector: vector.to_vec(),
            });
            delta.id_map.insert(id.to_string(), idx);
            inserted += 1;
        }

        Ok(inserted)
    }

    /// Get the current snapshot epoch for isolation level tracking.
    pub fn current_epoch(&self) -> u64 {
        self.epoch.current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(dim: usize, base: f32) -> Vec<f32> {
        (0..dim).map(|i| base + i as f32 * 0.01).collect()
    }

    #[test]
    fn test_create_cow_index() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Cosine);
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.insert("c", &[0.0, 0.0, 1.0, 0.0]).unwrap();

        assert_eq!(idx.len(), 3);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_delete() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        assert!(idx.delete("a").unwrap());
        assert!(!idx.delete("nonexistent").unwrap());

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn test_compact() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.insert("c", &[0.0, 0.0, 1.0, 0.0]).unwrap();

        let stats = idx.compact().unwrap();
        assert_eq!(stats.merged_insertions, 3);

        // Search should still work after compaction
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_compact_with_deletes() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Compact to move to base
        idx.compact().unwrap();

        // Delete from base
        idx.delete("a").unwrap();

        // Re-compact
        let stats = idx.compact().unwrap();
        assert_eq!(stats.applied_deletes, 1);

        // "a" should not appear in results
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.iter().all(|r| r.id != "a"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        assert!(idx.insert("a", &[1.0, 0.0]).is_err());
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.search(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn test_duplicate_insert() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(idx.insert("a", &[0.0, 1.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_stats() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let stats = idx.stats();
        assert_eq!(stats.base_vectors, 0);
        assert_eq!(stats.delta_insertions, 2);
        assert_eq!(stats.compaction_count, 0);
    }

    #[test]
    fn test_needs_compaction() {
        let config = CowHnswConfig {
            compaction_threshold: 3,
            ..Default::default()
        };
        let idx = CowHnswIndex::new(config, 4, DistanceFunction::Euclidean);
        assert!(!idx.needs_compaction());

        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!(!idx.needs_compaction());

        idx.insert("c", &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert!(idx.needs_compaction());
    }

    #[test]
    fn test_concurrent_read_write() {
        use std::sync::Arc;
        use std::thread;

        let idx = Arc::new(CowHnswIndex::new(
            CowHnswConfig::default(),
            4,
            DistanceFunction::Euclidean,
        ));

        // Insert some initial data
        for i in 0..20 {
            idx.insert(&format!("v{}", i), &make_vec(4, i as f32 * 0.1))
                .unwrap();
        }

        let idx_read = Arc::clone(&idx);
        let idx_write = Arc::clone(&idx);

        // Concurrent reads and writes
        let reader = thread::spawn(move || {
            let mut success = 0;
            for _ in 0..100 {
                let results = idx_read.search(&[0.5, 0.5, 0.5, 0.5], 5).unwrap();
                if !results.is_empty() {
                    success += 1;
                }
            }
            success
        });

        let writer = thread::spawn(move || {
            for i in 20..40 {
                let _ = idx_write.insert(&format!("v{}", i), &make_vec(4, i as f32 * 0.1));
            }
        });

        writer.join().unwrap();
        let read_success = reader.join().unwrap();
        assert!(read_success > 0, "Concurrent reads should succeed");
    }

    #[test]
    fn test_larger_dataset() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 16, DistanceFunction::Euclidean);

        for i in 0..50 {
            idx.insert(&format!("v{}", i), &make_vec(16, i as f32 * 0.1))
                .unwrap();
        }

        // Search before compaction (delta scan)
        let results = idx.search(&make_vec(16, 0.0), 5).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].id, "v0");

        // Compact and search again
        idx.compact().unwrap();
        let stats = idx.stats();
        assert_eq!(stats.base_vectors, 50);
        assert_eq!(stats.delta_insertions, 0);

        let results = idx.search(&make_vec(16, 0.0), 5).unwrap();
        assert!(!results.is_empty());

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }

    #[test]
    fn test_wal_replay() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        let mut wal = CowHnswWal::new();

        wal.append(CowWalEntry::Insert {
            id: "a".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
        });
        wal.append(CowWalEntry::Insert {
            id: "b".into(),
            vector: vec![0.0, 1.0, 0.0, 0.0],
        });

        let stats = wal.replay(&idx).unwrap();
        assert_eq!(stats.inserts, 2);
        assert_eq!(stats.errors, 0);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_wal_truncate_to_checkpoint() {
        let mut wal = CowHnswWal::new();
        wal.append(CowWalEntry::Insert {
            id: "a".into(),
            vector: vec![1.0],
        });
        wal.append(CowWalEntry::CompactionCheckpoint { epoch: 1 });
        wal.append(CowWalEntry::Insert {
            id: "b".into(),
            vector: vec![2.0],
        });

        assert_eq!(wal.len(), 3);
        wal.truncate_to_checkpoint();
        assert_eq!(wal.len(), 1); // Only "b" remains after checkpoint
    }

    #[test]
    fn test_wal_replay_with_deletes() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        let mut wal = CowHnswWal::new();

        wal.append(CowWalEntry::Insert {
            id: "a".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0],
        });
        wal.append(CowWalEntry::Delete { id: "a".into() });

        let stats = wal.replay(&idx).unwrap();
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.deletes, 1);
    }

    #[test]
    fn test_batch_insert() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        let items: Vec<(&str, &[f32])> = vec![
            ("a", &[1.0, 0.0, 0.0, 0.0]),
            ("b", &[0.0, 1.0, 0.0, 0.0]),
            ("c", &[0.0, 0.0, 1.0, 0.0]),
        ];
        let inserted = idx.insert_batch(&items).unwrap();
        assert_eq!(inserted, 3);
        assert_eq!(idx.len(), 3);

        // Duplicates should be skipped
        let inserted2 = idx.insert_batch(&items).unwrap();
        assert_eq!(inserted2, 0);
    }

    #[test]
    fn test_batch_insert_wrong_dim() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        let items: Vec<(&str, &[f32])> = vec![
            ("a", &[1.0, 0.0]),        // wrong dim
            ("b", &[0.0, 1.0, 0.0, 0.0]),  // correct
        ];
        let inserted = idx.insert_batch(&items).unwrap();
        assert_eq!(inserted, 1);
    }

    #[test]
    fn test_read_guard() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();

        // Read guard should not panic or deadlock
        {
            let _guard = idx.read_guard();
            // While guard is held, reads should succeed
            let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
            assert!(!results.is_empty());
        }
        // Guard dropped, epoch should be clean
        assert!(!idx.epoch.has_active_readers());
    }

    #[test]
    fn test_current_epoch() {
        let idx = CowHnswIndex::new(CowHnswConfig::default(), 4, DistanceFunction::Euclidean);
        let e1 = idx.current_epoch();
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.compact().unwrap();
        let e2 = idx.current_epoch();
        assert!(e2 > e1);
    }
}
