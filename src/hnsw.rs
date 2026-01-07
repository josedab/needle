//! Hierarchical Navigable Small World (HNSW) Index
//!
//! This module implements the HNSW algorithm for approximate nearest neighbor search,
//! providing sub-linear search complexity with high recall rates.
//!
//! # Overview
//!
//! HNSW is a graph-based algorithm that creates a hierarchical, navigable graph structure
//! where each node represents a vector. The graph has multiple layers, with sparser
//! connections at higher layers enabling fast traversal to the approximate neighborhood.
//!
//! # Key Parameters
//!
//! - **M**: Maximum number of connections per node (default: 16). Higher values improve
//!   recall but increase memory usage and index time.
//! - **ef_construction**: Search depth during index construction (default: 200). Higher
//!   values improve index quality but slow construction.
//! - **ef_search**: Search depth during queries (default: 50). Higher values improve
//!   recall but slow queries.
//!
//! # Example
//!
//! ```ignore
//! use needle::hnsw::{HnswConfig, HnswIndex};
//! use needle::DistanceFunction;
//!
//! // Create index with custom configuration
//! let config = HnswConfig::builder()
//!     .m(32)
//!     .ef_construction(400)
//!     .ef_search(100);
//!
//! let mut index = HnswIndex::new(config, DistanceFunction::Cosine);
//! ```
//!
//! # Performance Characteristics
//!
//! - **Search**: O(log n) average case
//! - **Insert**: O(log n) average case
//! - **Memory**: O(n * M) for connections + O(n * d) for vectors
//!
//! # Thread Safety
//!
//! The HNSW index is not thread-safe by itself. Thread safety is provided at the
//! `Collection` and `Database` levels using `parking_lot::RwLock`.

use crate::distance::DistanceFunction;
use crate::error::Result;
use ordered_float::OrderedFloat;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use tracing::debug;

/// Internal vector ID
pub type VectorId = usize;

/// Statistics from an HNSW search operation
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Number of nodes visited during the search
    pub visited_nodes: usize,
    /// Number of layers traversed (including layer 0)
    pub layers_traversed: usize,
}

/// HNSW configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node (default: 16)
    pub m: usize,
    /// Maximum connections for layer 0 (default: 2 * m)
    pub m_max_0: usize,
    /// Construction search depth (default: 200)
    pub ef_construction: usize,
    /// Query search depth (default: 50)
    pub ef_search: usize,
    /// Level multiplier (default: 1/ln(M))
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m_max_0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
        }
    }
}

impl HnswConfig {
    /// Create a new configuration with custom M value
    pub fn with_m(m: usize) -> Self {
        Self {
            m,
            m_max_0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    /// Create a new builder starting from defaults
    pub fn builder() -> Self {
        Self::default()
    }

    /// Set the M parameter (max connections per layer)
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max_0 = m * 2;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Set the M_max_0 parameter (max connections for layer 0)
    pub fn m_max_0(mut self, m_max_0: usize) -> Self {
        self.m_max_0 = m_max_0;
        self
    }

    /// Set ef_construction (construction search depth)
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search (query search depth)
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set the level multiplier
    pub fn ml(mut self, ml: f64) -> Self {
        self.ml = ml;
        self
    }
}

/// A layer in the HNSW graph
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Layer {
    /// Adjacency lists for each node in this layer
    connections: Vec<Vec<VectorId>>,
}

impl Layer {
    fn new() -> Self {
        Self {
            connections: Vec::new(),
        }
    }

    fn ensure_capacity(&mut self, id: VectorId) {
        if id >= self.connections.len() {
            self.connections.resize(id + 1, Vec::new());
        }
    }

    fn get_connections(&self, id: VectorId) -> &[VectorId] {
        self.connections
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    fn set_connections(&mut self, id: VectorId, neighbors: Vec<VectorId>) {
        self.ensure_capacity(id);
        self.connections[id] = neighbors;
    }

    fn add_connection(&mut self, from: VectorId, to: VectorId) {
        self.ensure_capacity(from);
        if !self.connections[from].contains(&to) {
            self.connections[from].push(to);
        }
    }

    /// Get the number of connections for a node without allocating
    fn connection_count(&self, id: VectorId) -> usize {
        self.connections.get(id).map(|v| v.len()).unwrap_or(0)
    }
}

/// Bit-packed set for efficient deleted vector tracking
///
/// Uses 1 bit per potential vector ID, providing O(1) lookups with
/// better cache performance than HashSet for dense ID spaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BitSet {
    /// Bit storage: each u64 holds 64 bits
    bits: Vec<u64>,
    /// Cached count of set bits
    count: usize,
}

impl BitSet {
    /// Create a new empty BitSet
    pub fn new() -> Self {
        Self {
            bits: Vec::new(),
            count: 0,
        }
    }

    /// Check if an ID is in the set
    #[inline]
    pub fn contains(&self, id: &usize) -> bool {
        let word_idx = *id / 64;
        let bit_idx = *id % 64;
        self.bits.get(word_idx).map_or(false, |word| (word >> bit_idx) & 1 == 1)
    }

    /// Insert an ID into the set, returns true if it was newly inserted
    pub fn insert(&mut self, id: usize) -> bool {
        let word_idx = id / 64;
        let bit_idx = id % 64;

        // Grow if necessary
        if word_idx >= self.bits.len() {
            self.bits.resize(word_idx + 1, 0);
        }

        let mask = 1u64 << bit_idx;
        let was_set = (self.bits[word_idx] & mask) != 0;
        if !was_set {
            self.bits[word_idx] |= mask;
            self.count += 1;
        }
        !was_set
    }

    /// Remove an ID from the set, returns true if it was present
    #[allow(dead_code)]
    pub fn remove(&mut self, id: &usize) -> bool {
        let word_idx = *id / 64;
        let bit_idx = *id % 64;

        if word_idx >= self.bits.len() {
            return false;
        }

        let mask = 1u64 << bit_idx;
        let was_set = (self.bits[word_idx] & mask) != 0;
        if was_set {
            self.bits[word_idx] &= !mask;
            self.count -= 1;
        }
        was_set
    }

    /// Get the number of elements in the set
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the set is empty
    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clear all elements from the set
    pub fn clear(&mut self) {
        self.bits.clear();
        self.count = 0;
    }
}

/// HNSW index for approximate nearest neighbor search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    /// Layers of the graph (layer 0 is densest)
    layers: Vec<Layer>,
    /// Entry point (node in highest layer)
    entry_point: Option<VectorId>,
    /// The level of the entry point
    entry_level: usize,
    /// Levels assigned to each vector
    node_levels: Vec<usize>,
    /// Configuration
    config: HnswConfig,
    /// Distance function
    distance: DistanceFunction,
    /// Number of vectors (not counting deleted)
    count: usize,
    /// Bit-packed set of deleted vector IDs for cache-efficient lookups
    #[serde(default)]
    deleted: BitSet,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(config: HnswConfig, distance: DistanceFunction) -> Self {
        Self {
            layers: vec![Layer::new()],
            entry_point: None,
            entry_level: 0,
            node_levels: Vec::new(),
            config,
            distance,
            count: 0,
            deleted: BitSet::new(),
        }
    }

    /// Create a new index with default configuration
    pub fn with_distance(distance: DistanceFunction) -> Self {
        Self::new(HnswConfig::default(), distance)
    }

    /// Get the number of vectors in the index
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the configuration
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Set the ef_search parameter for queries
    pub fn set_ef_search(&mut self, ef: usize) {
        self.config.ef_search = ef;
    }

    /// Generate a random level for a new node
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f64>() < self.config.ml && level < 32 {
            level += 1;
        }
        level
    }

    /// Get max connections for a given layer
    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max_0
        } else {
            self.config.m
        }
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: VectorId, vector: &[f32], vectors: &[Vec<f32>]) -> Result<()> {
        let level = self.random_level();

        // Ensure we have enough layers
        while self.layers.len() <= level {
            self.layers.push(Layer::new());
        }

        // Ensure node_levels has capacity
        if id >= self.node_levels.len() {
            self.node_levels.resize(id + 1, 0);
        }
        self.node_levels[id] = level;

        // Handle first insertion
        if self.entry_point.is_none() {
            self.entry_point = Some(id);
            self.entry_level = level;
            for l in 0..=level {
                self.layers[l].ensure_capacity(id);
            }
            self.count += 1;
            return Ok(());
        }

        // Safety: entry_point is Some because we just checked is_none() above
        let entry_point = self.entry_point.expect("entry_point should be Some after is_none check");
        let mut current = entry_point;

        // Traverse from top layer to one above insert level
        for l in (level + 1..=self.entry_level).rev() {
            let result = self.search_layer(vector, current, 1, l, vectors);
            if let Some((closest, _)) = result.first() {
                current = *closest;
            }
        }

        // Insert into layers from insert level (or entry level) down to 0
        let start_level = level.min(self.entry_level);
        for l in (0..=start_level).rev() {
            let candidates =
                self.search_layer(vector, current, self.config.ef_construction, l, vectors);
            let neighbors = self.select_neighbors(&candidates, self.max_connections(l));

            // Set connections for the new node
            self.layers[l].set_connections(id, neighbors.iter().map(|(n, _)| *n).collect());

            // Add bidirectional connections
            for (neighbor_id, _) in &neighbors {
                self.layers[l].add_connection(*neighbor_id, id);

                // Prune if too many connections - check count first to avoid allocation
                let max_conn = self.max_connections(l);
                if self.layers[l].connection_count(*neighbor_id) > max_conn {
                    // Only allocate when pruning is actually needed
                    let neighbor_connections = self.layers[l].get_connections(*neighbor_id).to_vec();
                    let neighbor_vec = &vectors[*neighbor_id];
                    let scored: Vec<(VectorId, f32)> = neighbor_connections
                        .iter()
                        .map(|&n| (n, self.distance.compute(neighbor_vec, &vectors[n])))
                        .collect();
                    let pruned = self.select_neighbors(&scored, max_conn);
                    self.layers[l]
                        .set_connections(*neighbor_id, pruned.iter().map(|(n, _)| *n).collect());
                }
            }

            if !candidates.is_empty() {
                current = candidates[0].0;
            }
        }

        // Update entry point if the new node has higher level
        if level > self.entry_level {
            self.entry_point = Some(id);
            self.entry_level = level;
        }

        self.count += 1;
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize, vectors: &[Vec<f32>]) -> Vec<(VectorId, f32)> {
        self.search_with_ef(query, k, self.config.ef_search, vectors)
    }

    /// Search for k nearest neighbors and return statistics
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[Vec<f32>],
    ) -> (Vec<(VectorId, f32)>, SearchStats) {
        self.search_with_ef_stats(query, k, self.config.ef_search, vectors)
    }

    /// Search for k nearest neighbors with a custom ef_search parameter
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &[Vec<f32>],
    ) -> Vec<(VectorId, f32)> {
        self.search_with_ef_stats(query, k, ef_search, vectors).0
    }

    /// Search for k nearest neighbors with custom ef_search and return statistics
    pub fn search_with_ef_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &[Vec<f32>],
    ) -> (Vec<(VectorId, f32)>, SearchStats) {
        let mut stats = SearchStats::default();

        if self.entry_point.is_none() {
            return (vec![], stats);
        }

        // Safety: entry_point is Some because we just checked is_none() above
        let mut current = self.entry_point.expect("entry_point should be Some after is_none check");

        // Traverse from top layer down to layer 1
        let layers_to_traverse = self.entry_level;
        for l in (1..=self.entry_level).rev() {
            let (result, visited) = self.search_layer_with_stats(query, current, 1, l, vectors);
            stats.visited_nodes += visited;
            if let Some((closest, _)) = result.first() {
                current = *closest;
            }
        }

        // Search layer 0 with custom ef_search
        let (candidates, visited) = self.search_layer_with_stats(query, current, ef_search, 0, vectors);
        stats.visited_nodes += visited;

        // Total layers traversed = upper layers + layer 0
        stats.layers_traversed = layers_to_traverse + 1;

        // Return top k
        (candidates.into_iter().take(k).collect(), stats)
    }

    /// Search a single layer using beam search
    fn search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: usize,
        vectors: &[Vec<f32>],
    ) -> Vec<(VectorId, f32)> {
        // Use Vec<u8> for O(1) visited checks with better cache behavior than Vec<bool>
        // (Vec<bool> uses bit-packing which causes more CPU operations)
        let mut visited = vec![0u8; vectors.len()];
        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>> = BinaryHeap::new();
        // Max-heap for results (farthest first, for easy pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();

        let entry_dist = self.distance.compute(query, &vectors[entry]);
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        // Only add to results if not deleted
        if !self.deleted.contains(&entry) {
            results.push((OrderedFloat(entry_dist), entry));
        }
        visited[entry] = 1;

        while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            // Get the worst distance in results
            let worst_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);

            // If the best candidate is worse than our worst result, we're done
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            let connections = self.layers[layer].get_connections(c_id);
            for &neighbor in connections {
                if visited[neighbor] == 0 {
                    visited[neighbor] = 1;
                    let dist = self.distance.compute(query, &vectors[neighbor]);

                    // Always add to candidates for graph traversal
                    if dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse((OrderedFloat(dist), neighbor)));

                        // Only add to results if not deleted
                        if !self.deleted.contains(&neighbor) {
                            results.push((OrderedFloat(dist), neighbor));

                            // Keep only top ef results
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert to sorted vector (closest first)
        let mut result_vec: Vec<_> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        // Use sort_unstable_by for 5-10% faster sorting (stable ordering not needed for results)
        result_vec.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result_vec
    }

    /// Search a single layer and return the number of visited nodes
    fn search_layer_with_stats(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: usize,
        vectors: &[Vec<f32>],
    ) -> (Vec<(VectorId, f32)>, usize) {
        let mut visited = vec![0u8; vectors.len()];
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();
        let mut visited_count = 0usize;

        let entry_dist = self.distance.compute(query, &vectors[entry]);
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        if !self.deleted.contains(&entry) {
            results.push((OrderedFloat(entry_dist), entry));
        }
        visited[entry] = 1;
        visited_count += 1;

        while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            let worst_dist = results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY);

            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            let connections = self.layers[layer].get_connections(c_id);
            for &neighbor in connections {
                if visited[neighbor] == 0 {
                    visited[neighbor] = 1;
                    visited_count += 1;
                    let dist = self.distance.compute(query, &vectors[neighbor]);

                    if dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse((OrderedFloat(dist), neighbor)));

                        if !self.deleted.contains(&neighbor) {
                            results.push((OrderedFloat(dist), neighbor));

                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<_> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        (result_vec, visited_count)
    }

    /// Select neighbors using the simple heuristic
    fn select_neighbors(
        &self,
        candidates: &[(VectorId, f32)],
        max_neighbors: usize,
    ) -> Vec<(VectorId, f32)> {
        // Simple selection: take the closest neighbors
        let mut sorted = candidates.to_vec();
        // Use sort_unstable_by for 5-10% faster sorting (stable ordering not needed)
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(max_neighbors);
        sorted
    }

    /// Delete a vector from the index (marks as deleted)
    pub fn delete(&mut self, id: VectorId) -> Result<bool> {
        if id >= self.node_levels.len() || self.deleted.contains(&id) {
            return Ok(false);
        }
        self.deleted.insert(id);
        self.count = self.count.saturating_sub(1);
        Ok(true)
    }

    /// Check if a vector is deleted
    pub fn is_deleted(&self, id: VectorId) -> bool {
        self.deleted.contains(&id)
    }

    /// Get the number of deleted vectors
    pub fn deleted_count(&self) -> usize {
        self.deleted.len()
    }

    /// Check if a vector exists in the index (and is not deleted)
    pub fn contains(&self, id: VectorId) -> bool {
        id < self.node_levels.len() && !self.deleted.contains(&id)
    }

    /// Compact the index by rebuilding without deleted vectors
    /// Returns a mapping from old IDs to new IDs
    pub fn compact(&mut self, vectors: &[Vec<f32>]) -> HashMap<VectorId, VectorId> {
        if self.deleted.is_empty() {
            debug!("Compact called but no deleted vectors");
            return HashMap::new();
        }

        debug!(
            deleted = self.deleted.len(),
            total = self.count + self.deleted.len(),
            "Starting index compaction"
        );

        // Build mapping from old IDs to new IDs
        let mut id_map: HashMap<VectorId, VectorId> = HashMap::new();
        let mut new_id = 0usize;

        for old_id in 0..self.node_levels.len() {
            if !self.deleted.contains(&old_id) {
                id_map.insert(old_id, new_id);
                new_id += 1;
            }
        }

        // Collect non-deleted vectors with their new IDs
        let active_vectors: Vec<(VectorId, &[f32])> = id_map
            .iter()
            .map(|(&old_id, &new_id)| (new_id, vectors[old_id].as_slice()))
            .collect();

        // Create a new index
        let mut new_index = HnswIndex::new(self.config.clone(), self.distance);

        // Collect vectors for the new index
        let new_vectors: Vec<Vec<f32>> = {
            let mut sorted: Vec<_> = active_vectors.iter().collect();
            sorted.sort_by_key(|(id, _)| *id);
            sorted.iter().map(|(_, v)| v.to_vec()).collect()
        };

        // Insert all vectors into new index
        for (new_id, vec) in new_vectors.iter().enumerate() {
            let _ = new_index.insert(new_id, vec, &new_vectors);
        }

        // Replace self with new index
        self.layers = new_index.layers;
        self.entry_point = new_index.entry_point;
        self.entry_level = new_index.entry_level;
        self.node_levels = new_index.node_levels;
        self.count = new_index.count;
        self.deleted.clear();

        debug!(
            new_count = self.count,
            remapped = id_map.len(),
            "Index compaction completed"
        );

        id_map
    }

    /// Check if compaction is recommended based on deleted ratio
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        if self.count == 0 && self.deleted.is_empty() {
            return false;
        }
        let total = self.count + self.deleted.len();
        let deleted_ratio = self.deleted.len() as f64 / total as f64;
        deleted_ratio >= threshold
    }

    /// Get index statistics
    pub fn stats(&self) -> HnswStats {
        let num_layers = self.layers.len();
        let total_edges: usize = self.layers.iter()
            .map(|l| l.connections.iter().map(|c| c.len()).sum::<usize>())
            .sum();
        let active_count = self.count;
        let avg_connections = if active_count > 0 {
            total_edges as f64 / active_count as f64
        } else {
            0.0
        };

        HnswStats {
            num_vectors: active_count,
            num_deleted: self.deleted.len(),
            num_layers,
            total_edges,
            avg_connections_per_node: avg_connections,
            entry_point: self.entry_point,
            entry_level: self.entry_level,
            m: self.config.m,
            ef_construction: self.config.ef_construction,
            ef_search: self.config.ef_search,
        }
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();

        // Layers memory
        let layer_memory: usize = self.layers.iter()
            .map(|l| {
                std::mem::size_of::<Layer>() +
                l.connections.iter()
                    .map(|c| std::mem::size_of::<Vec<VectorId>>() + c.len() * std::mem::size_of::<VectorId>())
                    .sum::<usize>()
            })
            .sum();

        // Node levels vector
        let node_levels_memory = self.node_levels.len() * std::mem::size_of::<usize>();

        base_size + layer_memory + node_levels_memory
    }
}

/// HNSW index statistics
#[derive(Debug, Clone)]
pub struct HnswStats {
    /// Number of active vectors in the index
    pub num_vectors: usize,
    /// Number of deleted (tombstoned) vectors
    pub num_deleted: usize,
    /// Number of layers in the graph
    pub num_layers: usize,
    /// Total number of edges (connections)
    pub total_edges: usize,
    /// Average connections per node
    pub avg_connections_per_node: f64,
    /// Entry point vector ID
    pub entry_point: Option<VectorId>,
    /// Level of the entry point
    pub entry_level: usize,
    /// M parameter (max connections per node)
    pub m: usize,
    /// ef_construction parameter
    pub ef_construction: usize,
    /// ef_search parameter
    pub ef_search: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn random_vector(dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_hnsw_insert_and_search() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 128;
        let n = 1000;

        // Generate random vectors
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(dim)).collect();

        // Insert all vectors
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        assert_eq!(index.len(), n);

        // Search for nearest neighbors of first vector
        let query = &vectors[0];
        let results = index.search(query, 10, &vectors);

        // The query vector itself should be the closest
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.001); // Should be ~0 distance to itself
    }

    #[test]
    fn test_hnsw_empty() {
        let index = HnswIndex::with_distance(DistanceFunction::Cosine);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let results = index.search(&[1.0, 2.0, 3.0], 10, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_recall() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 32;
        let n = 500;
        let k = 10;

        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        // Test recall on several queries
        let mut total_recall = 0.0;
        let num_queries = 10;

        for i in 0..num_queries {
            let query = &vectors[i * 10];

            // Get HNSW results
            let hnsw_results: HashSet<_> = index
                .search(query, k, &vectors)
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            // Compute brute force results
            let mut brute_force: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(id, v)| (id, DistanceFunction::Euclidean.compute(query, v)))
                .collect();
            brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let brute_force_results: HashSet<_> =
                brute_force.iter().take(k).map(|(id, _)| *id).collect();

            // Calculate recall
            let intersection = hnsw_results.intersection(&brute_force_results).count();
            total_recall += intersection as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.8,
            "Recall should be > 80%, got {}",
            avg_recall
        );
    }

    #[test]
    fn test_hnsw_delete() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 32;
        let n = 100;

        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        assert_eq!(index.len(), n);
        assert_eq!(index.deleted_count(), 0);

        // Delete some vectors
        index.delete(5).unwrap();
        index.delete(10).unwrap();
        index.delete(15).unwrap();

        assert_eq!(index.len(), n - 3);
        assert_eq!(index.deleted_count(), 3);
        assert!(index.is_deleted(5));
        assert!(index.is_deleted(10));
        assert!(!index.is_deleted(0));

        // Search should not return deleted vectors
        let results = index.search(&vectors[5], 10, &vectors);
        for (id, _) in &results {
            assert!(!index.is_deleted(*id));
        }
    }

    #[test]
    fn test_hnsw_compaction() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 32;
        let n = 50;

        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        // Delete half the vectors
        for i in (0..n).step_by(2) {
            index.delete(i).unwrap();
        }

        assert_eq!(index.len(), n / 2);
        assert_eq!(index.deleted_count(), n / 2);
        assert!(index.needs_compaction(0.3)); // 50% deleted > 30% threshold

        // Compact the index
        let id_map = index.compact(&vectors);

        assert_eq!(index.len(), n / 2);
        assert_eq!(index.deleted_count(), 0);
        assert!(!index.needs_compaction(0.3));
        assert!(!id_map.is_empty());

        // Verify remapping - odd IDs should be mapped
        for old_id in (1..n).step_by(2) {
            assert!(id_map.contains_key(&old_id));
        }
    }

    #[test]
    fn test_hnsw_stats() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let vectors: Vec<Vec<f32>> = (0..20).map(|_| random_vector(16)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 20);
        assert_eq!(stats.num_deleted, 0);

        index.delete(5).unwrap();
        index.delete(10).unwrap();

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 18);
        assert_eq!(stats.num_deleted, 2);
    }
}
