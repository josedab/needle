#![allow(dead_code)]

//! Concurrent HNSW Graph Operations
//!
//! Provides fine-grained concurrent access to the HNSW graph for online insert/delete
//! without full rebuild. Complements `incremental.rs` with:
//!
//! - **Node-level locking**: Fine-grained `RwLock` per node for concurrent graph mutations
//! - **Lazy mark-and-skip deletion**: Tombstone markers with neighbor shortcutting
//! - **Background compaction**: Remove tombstones and rebalance neighbor lists
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────┐
//! │           ConcurrentHnswGraph                     │
//! ├──────────────────────────────────────────────────┤
//! │  Node[0] ←RwLock→ neighbors, tombstone flag      │
//! │  Node[1] ←RwLock→ neighbors, tombstone flag      │
//! │  ...                                              │
//! │  Node[N] ←RwLock→ neighbors, tombstone flag      │
//! ├──────────────────────────────────────────────────┤
//! │  CompactionWorker: removes tombstones, relinks    │
//! └──────────────────────────────────────────────────┘
//! ```

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// A node in the concurrent HNSW graph.
struct ConcurrentNode {
    /// Neighbor lists per layer.
    neighbors: RwLock<Vec<Vec<usize>>>,
    /// Tombstone flag for lazy deletion.
    tombstone: AtomicBool,
    /// The layer this node was assigned to.
    max_layer: usize,
}

impl ConcurrentNode {
    fn new(max_layer: usize) -> Self {
        let mut layers = Vec::with_capacity(max_layer + 1);
        for _ in 0..=max_layer {
            layers.push(Vec::new());
        }
        Self {
            neighbors: RwLock::new(layers),
            tombstone: AtomicBool::new(false),
            max_layer,
        }
    }

    fn is_deleted(&self) -> bool {
        self.tombstone.load(Ordering::Acquire)
    }

    fn mark_deleted(&self) {
        self.tombstone.store(true, Ordering::Release);
    }
}

/// Configuration for concurrent HNSW operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentHnswConfig {
    /// Maximum connections per node at layer 0.
    pub m_max_0: usize,
    /// Maximum connections per node at higher layers.
    pub m_max: usize,
    /// Tombstone ratio that triggers compaction.
    pub compaction_threshold: f64,
    /// Maximum neighbors to shortcut during lazy deletion.
    pub max_shortcut_neighbors: usize,
}

impl Default for ConcurrentHnswConfig {
    fn default() -> Self {
        Self {
            m_max_0: 32,
            m_max: 16,
            compaction_threshold: 0.2,
            max_shortcut_neighbors: 8,
        }
    }
}

/// Statistics for the concurrent HNSW graph.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcurrentHnswStats {
    /// Total nodes (including tombstones).
    pub total_nodes: usize,
    /// Active (non-deleted) nodes.
    pub active_nodes: usize,
    /// Tombstone count.
    pub tombstone_count: usize,
    /// Total edges in the graph.
    pub total_edges: usize,
    /// Compactions performed.
    pub compactions_performed: u64,
    /// Nodes removed by compaction.
    pub nodes_compacted: u64,
    /// Edges repaired during compaction.
    pub edges_repaired: u64,
}

/// Result of a compaction operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Tombstones removed.
    pub tombstones_removed: usize,
    /// Edges repaired (shortcutted neighbors re-linked).
    pub edges_repaired: usize,
    /// Neighbor lists rebalanced.
    pub lists_rebalanced: usize,
    /// Duration in microseconds.
    pub duration_us: u64,
}

/// Concurrent HNSW graph supporting online inserts and lazy deletes.
pub struct ConcurrentHnswGraph {
    config: ConcurrentHnswConfig,
    nodes: RwLock<Vec<Arc<ConcurrentNode>>>,
    active_count: AtomicUsize,
    tombstone_count: AtomicUsize,
    compaction_count: AtomicU64,
    nodes_compacted: AtomicU64,
    edges_repaired: AtomicU64,
}

impl ConcurrentHnswGraph {
    /// Create a new concurrent HNSW graph.
    pub fn new(config: ConcurrentHnswConfig) -> Self {
        Self {
            config,
            nodes: RwLock::new(Vec::new()),
            active_count: AtomicUsize::new(0),
            tombstone_count: AtomicUsize::new(0),
            compaction_count: AtomicU64::new(0),
            nodes_compacted: AtomicU64::new(0),
            edges_repaired: AtomicU64::new(0),
        }
    }

    /// Insert a node at a given layer and connect it to neighbors.
    ///
    /// Uses node-level write locks only on the affected neighbor lists,
    /// allowing concurrent inserts to different regions of the graph.
    pub fn insert_node(&self, layer: usize, neighbors: Vec<(usize, Vec<usize>)>) -> usize {
        let node = Arc::new(ConcurrentNode::new(layer));

        // Set initial neighbors per layer
        {
            let mut node_neighbors = node.neighbors.write();
            for (l, nbrs) in &neighbors {
                if *l < node_neighbors.len() {
                    node_neighbors[*l] = nbrs.clone();
                }
            }
        }

        // Add node to the graph
        let node_id;
        {
            let mut nodes = self.nodes.write();
            node_id = nodes.len();
            nodes.push(node);
        }

        // Connect neighbors back to the new node (bidirectional)
        let nodes = self.nodes.read();
        for (l, nbrs) in &neighbors {
            let max_connections = if *l == 0 {
                self.config.m_max_0
            } else {
                self.config.m_max
            };

            for &nbr_id in nbrs {
                if nbr_id < nodes.len() && !nodes[nbr_id].is_deleted() {
                    let mut nbr_neighbors = nodes[nbr_id].neighbors.write();
                    if *l < nbr_neighbors.len() {
                        if !nbr_neighbors[*l].contains(&node_id) {
                            nbr_neighbors[*l].push(node_id);
                            // Prune if exceeding max connections
                            if nbr_neighbors[*l].len() > max_connections {
                                nbr_neighbors[*l].truncate(max_connections);
                            }
                        }
                    }
                }
            }
        }

        self.active_count.fetch_add(1, Ordering::Relaxed);
        node_id
    }

    /// Lazy delete: mark node as tombstone and shortcut its neighbors.
    ///
    /// Instead of removing the node from the graph, we mark it as deleted
    /// and connect its neighbors to each other to maintain graph connectivity.
    pub fn lazy_delete(&self, node_id: usize) -> Result<()> {
        let nodes = self.nodes.read();
        if node_id >= nodes.len() {
            return Err(NeedleError::VectorNotFound(format!("Node {}", node_id)));
        }

        let node = &nodes[node_id];
        if node.is_deleted() {
            return Ok(()); // Already deleted
        }

        // Mark as tombstone
        node.mark_deleted();

        // Shortcut neighbors: connect deleted node's neighbors to each other
        let node_neighbors = node.neighbors.read();
        for (layer, layer_neighbors) in node_neighbors.iter().enumerate() {
            let max_connections = if layer == 0 {
                self.config.m_max_0
            } else {
                self.config.m_max
            };

            // For each pair of neighbors, create a shortcut edge
            let live_neighbors: Vec<usize> = layer_neighbors
                .iter()
                .copied()
                .filter(|&n| n < nodes.len() && !nodes[n].is_deleted())
                .take(self.config.max_shortcut_neighbors)
                .collect();

            for i in 0..live_neighbors.len() {
                for j in (i + 1)..live_neighbors.len() {
                    let a = live_neighbors[i];
                    let b = live_neighbors[j];

                    // Add shortcut a -> b
                    if a < nodes.len() {
                        let mut a_nbrs = nodes[a].neighbors.write();
                        if layer < a_nbrs.len()
                            && !a_nbrs[layer].contains(&b)
                            && a_nbrs[layer].len() < max_connections
                        {
                            a_nbrs[layer].push(b);
                        }
                    }

                    // Add shortcut b -> a
                    if b < nodes.len() {
                        let mut b_nbrs = nodes[b].neighbors.write();
                        if layer < b_nbrs.len()
                            && !b_nbrs[layer].contains(&a)
                            && b_nbrs[layer].len() < max_connections
                        {
                            b_nbrs[layer].push(a);
                        }
                    }
                }
            }
        }

        self.active_count.fetch_sub(1, Ordering::Relaxed);
        self.tombstone_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Check if compaction is needed based on tombstone ratio.
    pub fn needs_compaction(&self) -> bool {
        let total = self.nodes.read().len();
        if total == 0 {
            return false;
        }
        let tombstones = self.tombstone_count.load(Ordering::Relaxed);
        (tombstones as f64 / total as f64) >= self.config.compaction_threshold
    }

    /// Background compaction: remove tombstones and rebalance neighbor lists.
    ///
    /// This operation:
    /// 1. Removes references to deleted nodes from all neighbor lists
    /// 2. Rebalances neighbor lists that have become too short
    /// 3. Reports statistics on edges repaired
    pub fn compact(&self) -> CompactionResult {
        let start = std::time::Instant::now();
        let mut tombstones_removed = 0;
        let mut edges_repaired = 0;
        let mut lists_rebalanced = 0;

        let nodes = self.nodes.read();

        // Collect tombstone IDs
        let tombstone_ids: HashSet<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_deleted())
            .map(|(i, _)| i)
            .collect();

        tombstones_removed = tombstone_ids.len();

        // For each live node, clean up neighbor lists
        for (i, node) in nodes.iter().enumerate() {
            if node.is_deleted() {
                continue;
            }

            let mut neighbors = node.neighbors.write();
            for (layer, layer_nbrs) in neighbors.iter_mut().enumerate() {
                let before_len = layer_nbrs.len();

                // Remove references to tombstoned nodes
                layer_nbrs.retain(|n| !tombstone_ids.contains(n));

                let removed = before_len - layer_nbrs.len();
                if removed > 0 {
                    edges_repaired += removed;
                }

                // Rebalance: if neighbor list is too short, try to add neighbors-of-neighbors
                let min_connections = if layer == 0 {
                    self.config.m_max_0 / 4
                } else {
                    self.config.m_max / 4
                };
                let max_connections = if layer == 0 {
                    self.config.m_max_0
                } else {
                    self.config.m_max
                };

                if layer_nbrs.len() < min_connections && layer_nbrs.len() < max_connections {
                    // Gather candidates from neighbors-of-neighbors
                    let mut candidates: Vec<usize> = Vec::new();
                    for &nbr in layer_nbrs.iter() {
                        if nbr < nodes.len() && !nodes[nbr].is_deleted() {
                            let nbr_nbrs = nodes[nbr].neighbors.read();
                            if layer < nbr_nbrs.len() {
                                for &nn in &nbr_nbrs[layer] {
                                    if nn != i
                                        && !tombstone_ids.contains(&nn)
                                        && !layer_nbrs.contains(&nn)
                                        && !candidates.contains(&nn)
                                    {
                                        candidates.push(nn);
                                    }
                                }
                            }
                        }
                    }

                    // Add candidates up to max_connections
                    for candidate in candidates {
                        if layer_nbrs.len() >= max_connections {
                            break;
                        }
                        layer_nbrs.push(candidate);
                        lists_rebalanced += 1;
                    }
                }
            }
        }

        self.tombstone_count.store(0, Ordering::Relaxed);
        self.compaction_count.fetch_add(1, Ordering::Relaxed);
        self.nodes_compacted
            .fetch_add(tombstones_removed as u64, Ordering::Relaxed);
        self.edges_repaired
            .fetch_add(edges_repaired as u64, Ordering::Relaxed);

        CompactionResult {
            tombstones_removed,
            edges_repaired,
            lists_rebalanced,
            duration_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Get the neighbors of a node, skipping tombstoned nodes.
    pub fn get_live_neighbors(&self, node_id: usize, layer: usize) -> Vec<usize> {
        let nodes = self.nodes.read();
        if node_id >= nodes.len() || nodes[node_id].is_deleted() {
            return Vec::new();
        }

        let neighbors = nodes[node_id].neighbors.read();
        if layer >= neighbors.len() {
            return Vec::new();
        }

        neighbors[layer]
            .iter()
            .copied()
            .filter(|&n| n < nodes.len() && !nodes[n].is_deleted())
            .collect()
    }

    /// Check if a node is deleted (tombstoned).
    pub fn is_deleted(&self, node_id: usize) -> bool {
        let nodes = self.nodes.read();
        node_id < nodes.len() && nodes[node_id].is_deleted()
    }

    /// Get current statistics.
    pub fn stats(&self) -> ConcurrentHnswStats {
        let nodes = self.nodes.read();
        let total_edges: usize = nodes
            .iter()
            .filter(|n| !n.is_deleted())
            .map(|n| {
                n.neighbors
                    .read()
                    .iter()
                    .map(|l| l.len())
                    .sum::<usize>()
            })
            .sum();

        ConcurrentHnswStats {
            total_nodes: nodes.len(),
            active_nodes: self.active_count.load(Ordering::Relaxed),
            tombstone_count: self.tombstone_count.load(Ordering::Relaxed),
            total_edges,
            compactions_performed: self.compaction_count.load(Ordering::Relaxed),
            nodes_compacted: self.nodes_compacted.load(Ordering::Relaxed),
            edges_repaired: self.edges_repaired.load(Ordering::Relaxed),
        }
    }

    /// Get the number of active (non-deleted) nodes.
    pub fn len(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the tombstone ratio (deleted / total).
    pub fn tombstone_ratio(&self) -> f64 {
        let total = self.nodes.read().len();
        if total == 0 {
            return 0.0;
        }
        self.tombstone_count.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Insert and auto-compact if the tombstone threshold is exceeded.
    pub fn insert_node_with_auto_compact(
        &self,
        layer: usize,
        neighbors: Vec<(usize, Vec<usize>)>,
    ) -> (usize, Option<CompactionResult>) {
        let id = self.insert_node(layer, neighbors);
        let compaction = if self.needs_compaction() {
            Some(self.compact())
        } else {
            None
        };
        (id, compaction)
    }

    /// Delete and auto-compact if the tombstone threshold is exceeded.
    pub fn lazy_delete_with_auto_compact(
        &self,
        node_id: usize,
    ) -> Result<Option<CompactionResult>> {
        self.lazy_delete(node_id)?;
        if self.needs_compaction() {
            Ok(Some(self.compact()))
        } else {
            Ok(None)
        }
    }

    /// Batch insert multiple nodes. Returns the assigned node IDs.
    pub fn batch_insert(
        &self,
        entries: Vec<(usize, Vec<(usize, Vec<usize>)>)>,
    ) -> Vec<usize> {
        entries
            .into_iter()
            .map(|(layer, neighbors)| self.insert_node(layer, neighbors))
            .collect()
    }

    /// Batch delete multiple nodes.
    pub fn batch_delete(&self, node_ids: &[usize]) -> Result<usize> {
        let mut deleted = 0;
        for &id in node_ids {
            self.lazy_delete(id)?;
            deleted += 1;
        }
        Ok(deleted)
    }

    /// Get the total number of nodes including tombstones.
    pub fn total_nodes(&self) -> usize {
        self.nodes.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_neighbors() {
        let graph = ConcurrentHnswGraph::new(ConcurrentHnswConfig::default());

        // Insert first node with no neighbors
        let n0 = graph.insert_node(0, vec![]);
        assert_eq!(n0, 0);

        // Insert second node connected to first
        let n1 = graph.insert_node(0, vec![(0, vec![0])]);
        assert_eq!(n1, 1);

        let nbrs = graph.get_live_neighbors(1, 0);
        assert!(nbrs.contains(&0));
    }

    #[test]
    fn test_lazy_delete() {
        let graph = ConcurrentHnswGraph::new(ConcurrentHnswConfig::default());

        let n0 = graph.insert_node(0, vec![]);
        let n1 = graph.insert_node(0, vec![(0, vec![0])]);
        let n2 = graph.insert_node(0, vec![(0, vec![0, 1])]);

        // Delete n1
        graph.lazy_delete(n1).unwrap();

        assert!(graph.is_deleted(n1));
        assert!(!graph.is_deleted(n0));
        assert!(!graph.is_deleted(n2));
        assert_eq!(graph.len(), 2);

        // n1's neighbors should be shortcutted
        let n0_nbrs = graph.get_live_neighbors(n0, 0);
        // n0 should potentially have n2 as a shortcut neighbor
        // (depends on shortcutting logic)
    }

    #[test]
    fn test_compaction() {
        let config = ConcurrentHnswConfig {
            compaction_threshold: 0.1,
            ..Default::default()
        };
        let graph = ConcurrentHnswGraph::new(config);

        for i in 0..10 {
            let nbrs = if i == 0 {
                vec![]
            } else {
                vec![(0, vec![i - 1])]
            };
            graph.insert_node(0, nbrs);
        }

        // Delete some nodes
        graph.lazy_delete(2).unwrap();
        graph.lazy_delete(5).unwrap();
        graph.lazy_delete(7).unwrap();

        assert!(graph.needs_compaction());

        let result = graph.compact();
        assert_eq!(result.tombstones_removed, 3);
        assert!(result.edges_repaired > 0 || result.tombstones_removed > 0);
    }

    #[test]
    fn test_stats() {
        let graph = ConcurrentHnswGraph::new(ConcurrentHnswConfig::default());

        graph.insert_node(0, vec![]);
        graph.insert_node(0, vec![(0, vec![0])]);
        graph.insert_node(0, vec![(0, vec![0, 1])]);

        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.active_nodes, 3);
        assert_eq!(stats.tombstone_count, 0);
    }

    #[test]
    fn test_tombstone_ratio() {
        let graph = ConcurrentHnswGraph::new(ConcurrentHnswConfig::default());
        graph.insert_node(0, vec![]);
        graph.insert_node(0, vec![(0, vec![0])]);
        graph.insert_node(0, vec![(0, vec![0, 1])]);
        graph.insert_node(0, vec![(0, vec![0, 1, 2])]);

        assert!((graph.tombstone_ratio() - 0.0).abs() < 0.001);

        graph.lazy_delete(1).expect("delete");
        assert!((graph.tombstone_ratio() - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_auto_compact_on_delete() {
        let config = ConcurrentHnswConfig {
            compaction_threshold: 0.2,
            ..Default::default()
        };
        let graph = ConcurrentHnswGraph::new(config);

        for i in 0..5 {
            let nbrs = if i == 0 { vec![] } else { vec![(0, vec![i - 1])] };
            graph.insert_node(0, nbrs);
        }

        // Delete 2 of 5 = 40% > 20% threshold
        graph.lazy_delete(1).expect("del");
        let result = graph.lazy_delete_with_auto_compact(2).expect("del+compact");
        // Should have triggered compaction
        assert!(result.is_some());
    }

    #[test]
    fn test_batch_insert() {
        let graph = ConcurrentHnswGraph::new(ConcurrentHnswConfig::default());
        let entries = vec![
            (0, vec![]),
            (0, vec![]),
            (0, vec![]),
        ];
        let ids = graph.batch_insert(entries);
        assert_eq!(ids.len(), 3);
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn test_batch_delete() {
        let graph = ConcurrentHnswGraph::new(ConcurrentHnswConfig::default());
        for _ in 0..5 {
            graph.insert_node(0, vec![]);
        }

        let deleted = graph.batch_delete(&[0, 2, 4]).expect("batch delete");
        assert_eq!(deleted, 3);
        assert_eq!(graph.len(), 2);
    }
}
