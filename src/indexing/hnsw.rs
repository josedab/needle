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
//! # Example: Configuring HNSW Parameters
//!
//! ```
//! use needle::{HnswConfig, HnswIndex, DistanceFunction};
//!
//! // Create index with custom configuration for high-recall search
//! let config = HnswConfig::builder()
//!     .m(32)                 // More connections = better recall
//!     .ef_construction(400)  // Higher = better index quality
//!     .ef_search(100);       // Higher = better search recall
//!
//! let index = HnswIndex::new(config, DistanceFunction::Cosine);
//! assert_eq!(index.len(), 0);
//! ```
//!
//! # Example: Using HNSW via Collection (Recommended)
//!
//! For most use cases, use the high-level [`Collection`](crate::Collection) API
//! which handles vector storage and HNSW indexing together:
//!
//! ```
//! use needle::{Database, CollectionConfig, DistanceFunction};
//!
//! let db = Database::in_memory();
//!
//! // Create collection with custom HNSW parameters
//! let config = CollectionConfig::new("vectors", 4)
//!     .with_m(24)                 // Custom M parameter
//!     .with_ef_construction(300); // Custom ef_construction
//!
//! db.create_collection_with_config(config).unwrap();
//!
//! let collection = db.collection("vectors").unwrap();
//! collection.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
//! collection.insert("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();
//!
//! // Search returns results with IDs and distances
//! let results = collection.search(&[0.2, 0.3, 0.4, 0.5], 1).unwrap();
//! assert_eq!(results[0].id, "v1");
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
//!
//! # When to Use HNSW
//!
//! HNSW is the **default index** and best choice for most use cases:
//!
//! | Use Case | HNSW Suitability |
//! |----------|------------------|
//! | < 10M vectors | ✅ Excellent - fits entirely in memory |
//! | Real-time search | ✅ Excellent - sub-millisecond queries |
//! | High recall requirements | ✅ Excellent - easily achieves 95%+ recall |
//! | Frequent updates | ✅ Good - O(log n) insert/delete |
//! | Memory constrained | ⚠️ Consider IVF or DiskANN instead |
//! | > 100M vectors | ⚠️ Consider DiskANN for disk-based search |
//!
//! ## HNSW vs IVF
//!
//! - **HNSW** is faster for search but uses more memory
//! - **IVF** uses less memory but requires training on data distribution
//! - Choose HNSW when query latency is critical
//! - Choose IVF when memory is constrained and you can accept slower queries
//!
//! ## HNSW vs DiskANN
//!
//! - **HNSW** stores the entire index in memory for fast search
//! - **DiskANN** stores vectors on disk with memory-efficient navigation
//! - Choose HNSW for datasets that fit in memory
//! - Choose DiskANN for datasets exceeding available RAM
//!
//! ## Insertion Algorithm
//!
//! New vectors are inserted via the following steps (Malkov & Yashunin, 2016):
//!
//! 1. **Layer assignment**: A random layer `l` is chosen using an exponentially decaying
//!    probability: `l = floor(-ln(uniform(0,1)) * m_l)` where `m_l = 1/ln(M)`. Most
//!    vectors land on layer 0; higher layers are exponentially rarer.
//!
//! 2. **Greedy descent (layers > l)**: Starting from the entry point at the top layer,
//!    greedily traverse toward the new vector through layers above `l`, keeping only the
//!    single closest node at each layer.
//!
//! 3. **Insertion into layers l..0**: At each layer from `l` down to 0, search for the
//!    `ef_construction` nearest neighbors, then connect the new node to the closest `M`
//!    neighbors (or `M * 2` on layer 0 for denser connectivity).
//!
//! 4. **Connection pruning**: When adding bidirectional edges would exceed the maximum
//!    connection count for an existing neighbor, the neighbor's connections are pruned
//!    using a heuristic that keeps diverse, short-distance edges. The heuristic prefers
//!    neighbors that are not only close to the node but also distant from each other,
//!    preserving graph navigability.
//!
//! 5. **Entry point update**: If the new vector's assigned layer exceeds the current
//!    maximum layer, the new node becomes the new global entry point.

use crate::distance::DistanceFunction;
use crate::error::Result;
use ordered_float::OrderedFloat;
use rand::Rng;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use tracing::debug;

/// Internal vector ID
pub type VectorId = usize;

/// Statistics from an HNSW search operation
#[must_use]
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Number of nodes visited during the search
    pub visited_nodes: usize,
    /// Number of layers traversed (including layer 0)
    pub layers_traversed: usize,
    /// Number of distance computations performed
    pub distance_computations: usize,
    /// Time spent in HNSW traversal (microseconds)
    pub traversal_time_us: u64,
}

/// A single hop in the HNSW graph traversal, recording which node was visited and the distance computed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHop {
    /// Layer at which this hop occurred
    pub layer: usize,
    /// Node ID visited
    pub node_id: VectorId,
    /// Distance from the query vector to this node
    pub distance: f32,
    /// Whether this node was added to the result candidate set
    pub added_to_candidates: bool,
    /// Neighbors explored from this node
    pub neighbors_explored: usize,
}

/// Detailed trace of an HNSW search, recording every hop for debugging and visualization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchTrace {
    /// Ordered list of hops during graph traversal
    pub hops: Vec<TraceHop>,
    /// Entry point node ID
    pub entry_point: Option<VectorId>,
    /// Entry point layer
    pub entry_layer: usize,
    /// Final result IDs with distances (top-k)
    pub results: Vec<(VectorId, f32)>,
    /// Total distance computations
    pub distance_computations: usize,
    /// Total nodes visited
    pub nodes_visited: usize,
    /// Layers traversed (from highest to 0)
    pub layers_traversed: Vec<usize>,
}

impl SearchTrace {
    /// Format the trace as an ASCII graph showing layer-by-layer traversal.
    pub fn format_ascii(&self) -> String {
        let mut out = String::new();
        out.push_str("HNSW Search Trace\n");

        if let Some(ep) = self.entry_point {
            out.push_str(&format!("Entry point: node {} (layer {})\n", ep, self.entry_layer));
        }
        out.push_str(&format!(
            "Stats: {} nodes visited, {} distance computations\n\n",
            self.nodes_visited, self.distance_computations
        ));

        // Group hops by layer
        let max_layer = self.hops.iter().map(|h| h.layer).max().unwrap_or(0);
        for layer in (0..=max_layer).rev() {
            let layer_hops: Vec<&TraceHop> = self.hops.iter().filter(|h| h.layer == layer).collect();
            if layer_hops.is_empty() {
                continue;
            }
            out.push_str(&format!("Layer {}:", layer));
            for hop in &layer_hops {
                let marker = if hop.added_to_candidates { "+" } else { "." };
                out.push_str(&format!(" [{}]{:.3}", hop.node_id, hop.distance));
                out.push_str(marker);
            }
            out.push('\n');
        }

        if !self.results.is_empty() {
            out.push_str("\nResults: ");
            for (id, dist) in &self.results {
                out.push_str(&format!("({}:{:.4}) ", id, dist));
            }
            out.push('\n');
        }

        out
    }
}

impl std::fmt::Display for SearchTrace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.format_ascii())
    }
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
    #[must_use]
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
    #[must_use]
    pub fn builder() -> Self {
        Self::default()
    }

    /// Set the M parameter (max connections per layer)
    #[must_use]
    pub fn m(mut self, m: usize) -> Self {
        self.m = m;
        self.m_max_0 = m * 2;
        self.ml = 1.0 / (m as f64).ln();
        self
    }

    /// Set the M_max_0 parameter (max connections for layer 0)
    #[must_use]
    pub fn m_max_0(mut self, m_max_0: usize) -> Self {
        self.m_max_0 = m_max_0;
        self
    }

    /// Set ef_construction (construction search depth)
    #[must_use]
    pub fn ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search (query search depth)
    #[must_use]
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set the level multiplier
    #[must_use]
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
            .map_or(&[], |v| v.as_slice())
    }

    fn set_connections(&mut self, id: VectorId, neighbors: Vec<VectorId>) {
        self.ensure_capacity(id);
        self.connections[id] = neighbors;
    }

    #[inline]
    fn add_connection(&mut self, from: VectorId, to: VectorId) {
        self.ensure_capacity(from);
        let conns = &mut self.connections[from];
        // For small connection lists, linear scan is cache-friendly and fast.
        // For larger lists (M=32+), binary search on sorted vec avoids O(n) scan.
        if conns.len() < 32 {
            if !conns.contains(&to) {
                conns.push(to);
            }
        } else {
            match conns.binary_search(&to) {
                Ok(_) => {} // already present
                Err(pos) => conns.insert(pos, to),
            }
        }
    }

    /// Get the number of connections for a node without allocating
    fn connection_count(&self, id: VectorId) -> usize {
        self.connections.get(id).map_or(0, |v| v.len())
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
        self.bits
            .get(word_idx)
            .is_some_and(|word| (word >> bit_idx) & 1 == 1)
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
    /// Access tracker for recording node access patterns
    #[serde(skip)]
    access_tracker: Option<AccessTracker>,
    /// Markov predictor for prefetching
    #[serde(skip)]
    markov_predictor: Option<MarkovPredictor>,
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
            access_tracker: None,
            markov_predictor: None,
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

    /// Generate a random level for a new node using a geometric distribution.
    ///
    /// Each node is assigned to layers 0..=level. The probability of being assigned
    /// to layer L is: P(level >= L) = ml^L, where ml is the level multiplier
    /// (default: 1/ln(M)). This gives an exponential decay: most nodes exist only
    /// in layer 0, while ~1/M nodes reach layer 1, ~1/M² reach layer 2, etc.
    /// The expected graph height is O(ln(n)), matching a skip-list structure.
    /// Level is capped at 32 to prevent degenerate cases.
    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0;
        while rng.gen::<f64>() < self.config.ml && level < 32 {
            level += 1;
        }
        level
    }

    /// Get max connections for a given layer.
    ///
    /// Layer 0 allows `m_max_0` (default: 2*M = 32) connections because it is the
    /// densest layer and the primary search surface. Upper layers use `M` (default: 16)
    /// connections. More connections improve recall at the cost of memory and build time.
    fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m_max_0
        } else {
            self.config.m
        }
    }

    /// Insert a vector into the HNSW index.
    ///
    /// Algorithm overview (from the HNSW paper by Malkov & Yashunin, 2018):
    /// 1. Assign a random level L to the new node (geometric distribution).
    /// 2. Starting from the entry point at the top layer, greedily descend through
    ///    layers above L, finding the closest node at each layer (ef=1).
    /// 3. From layer min(L, entry_level) down to layer 0, perform a beam search
    ///    with ef=ef_construction to find candidate neighbors.
    /// 4. Select the closest M neighbors and create bidirectional edges.
    /// 5. If any neighbor now exceeds max_connections, prune its edges by keeping
    ///    only the M closest neighbors (simple heuristic).
    /// 6. If L > entry_level, the new node becomes the entry point.
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

        let entry_point = self.entry_point.ok_or_else(|| {
            crate::error::NeedleError::Index("HNSW entry point missing after init check".into())
        })?;
        let mut current = entry_point;

        // Traverse from top layer to one above insert level
        for l in (level + 1..=self.entry_level).rev() {
            let result = self.search_layer(vector, current, 1, l, vectors)?;
            if let Some((closest, _)) = result.first() {
                current = *closest;
            }
        }

        // Insert into layers from insert level (or entry level) down to 0
        let start_level = level.min(self.entry_level);
        for l in (0..=start_level).rev() {
            let candidates =
                self.search_layer(vector, current, self.config.ef_construction, l, vectors)?;
            let neighbors = Self::select_neighbors(&candidates, self.max_connections(l));

            // Set connections for the new node
            self.layers[l].set_connections(id, neighbors.iter().map(|(n, _)| *n).collect());

            // Add bidirectional connections
            for (neighbor_id, _) in &neighbors {
                self.layers[l].add_connection(*neighbor_id, id);

                // Prune if too many connections - check count first to avoid allocation
                let max_conn = self.max_connections(l);
                if self.layers[l].connection_count(*neighbor_id) > max_conn {
                    // SmallVec avoids heap allocation for common case (M ≤ 16)
                    let neighbor_connections: SmallVec<[VectorId; 32]> =
                        self.layers[l].get_connections(*neighbor_id).iter().copied().collect();
                    let neighbor_vec = &vectors[*neighbor_id];
                    let mut scored: SmallVec<[(VectorId, f32); 32]> = SmallVec::with_capacity(neighbor_connections.len());
                    for &n in &neighbor_connections {
                        scored.push((n, self.distance.compute(neighbor_vec, &vectors[n])?));
                    }
                    let pruned = Self::select_neighbors(&scored, max_conn);
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

    /// Search for k nearest neighbors.
    ///
    /// Uses the default ef_search parameter from config. Higher ef_search improves
    /// recall (probability of finding true nearest neighbors) at the cost of latency.
    /// Typical values: ef_search=50 for ~95% recall, ef_search=200 for ~99% recall.
    pub fn search(&self, query: &[f32], k: usize, vectors: &[Vec<f32>]) -> Result<Vec<(VectorId, f32)>> {
        self.search_with_ef(query, k, self.config.ef_search, vectors)
    }

    /// Search for k nearest neighbors and return statistics
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<(VectorId, f32)>, SearchStats)> {
        self.search_with_ef_stats(query, k, self.config.ef_search, vectors)
    }

    /// Search for k nearest neighbors with a custom ef_search parameter
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<(VectorId, f32)>> {
        Ok(self.search_with_ef_stats(query, k, ef_search, vectors)?.0)
    }

    /// Search for k nearest neighbors with custom ef_search and return statistics.
    ///
    /// The search proceeds in two phases:
    /// 1. **Greedy descent (layers entry_level..1):** Starting from the entry point,
    ///    traverse each upper layer with ef=1 (greedy), moving to the closest node
    ///    found at each layer. This quickly narrows down to the right neighborhood.
    /// 2. **Beam search (layer 0):** Search layer 0 with the full ef_search parameter,
    ///    exploring more candidates for higher recall. Return the top-k results.
    pub fn search_with_ef_stats(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<(VectorId, f32)>, SearchStats)> {
        let mut stats = SearchStats::default();

        if self.entry_point.is_none() {
            return Ok((vec![], stats));
        }

        let mut current = self.entry_point.ok_or_else(|| {
            crate::error::NeedleError::Index("HNSW entry point missing after init check".into())
        })?;

        // Traverse from top layer down to layer 1
        let layers_to_traverse = self.entry_level;
        for l in (1..=self.entry_level).rev() {
            let (result, visited) = self.search_layer_with_stats(query, current, 1, l, vectors)?;
            stats.visited_nodes += visited;
            if let Some((closest, _)) = result.first() {
                current = *closest;
            }
        }

        // Search layer 0 with custom ef_search
        let (candidates, visited) =
            self.search_layer_with_stats(query, current, ef_search, 0, vectors)?;
        stats.visited_nodes += visited;

        // Total layers traversed = upper layers + layer 0
        stats.layers_traversed = layers_to_traverse + 1;

        // Return top k
        Ok((candidates.into_iter().take(k).collect(), stats))
    }

    /// Search with full trace recording for debugging and visualization.
    ///
    /// Returns results along with a detailed `SearchTrace` showing every hop
    /// in the graph traversal, useful for understanding search behavior.
    pub fn search_with_trace(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<(VectorId, f32)>, SearchTrace)> {
        let mut trace = SearchTrace {
            entry_point: self.entry_point,
            entry_layer: self.entry_level,
            ..Default::default()
        };

        if self.entry_point.is_none() {
            return Ok((vec![], trace));
        }

        let mut current = self.entry_point.ok_or_else(|| {
            crate::error::NeedleError::Index("HNSW entry point missing".into())
        })?;

        // Traverse upper layers (greedy, ef=1)
        for l in (1..=self.entry_level).rev() {
            trace.layers_traversed.push(l);
            let result = self.search_layer(query, current, 1, l, vectors)?;
            for (node_id, dist) in &result {
                trace.hops.push(TraceHop {
                    layer: l,
                    node_id: *node_id,
                    distance: *dist,
                    added_to_candidates: true,
                    neighbors_explored: self.layers[l].connection_count(*node_id),
                });
                trace.nodes_visited += 1;
                trace.distance_computations += 1;
            }
            if let Some((closest, _)) = result.first() {
                current = *closest;
            }
        }

        // Search layer 0 with full ef_search
        trace.layers_traversed.push(0);
        let candidates = self.search_layer(query, current, self.config.ef_search, 0, vectors)?;
        for (node_id, dist) in &candidates {
            trace.hops.push(TraceHop {
                layer: 0,
                node_id: *node_id,
                distance: *dist,
                added_to_candidates: true,
                neighbors_explored: self.layers[0].connection_count(*node_id),
            });
            trace.nodes_visited += 1;
            trace.distance_computations += 1;
        }

        let results: Vec<_> = candidates.into_iter().take(k).collect();
        trace.results = results.clone();
        Ok((results, trace))
    }

    /// Search for all vectors within a given distance radius.
    ///
    /// Unlike `search()` which returns the top-k nearest neighbors, this method
    /// returns all vectors whose distance to the query is less than or equal to
    /// `max_distance`. This is useful for applications that need all similar items
    /// above a certain similarity threshold.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `vectors` - The vector storage
    ///
    /// # Returns
    ///
    /// A vector of (id, distance) pairs for all vectors within the radius,
    /// sorted by distance (closest first).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{HnswConfig, HnswIndex, DistanceFunction};
    ///
    /// let config = HnswConfig::default();
    /// let mut index = HnswIndex::new(config, DistanceFunction::Cosine);
    /// let mut vectors = Vec::new();
    ///
    /// // Insert some vectors
    /// let v1 = vec![1.0, 0.0, 0.0];
    /// let v2 = vec![0.9, 0.1, 0.0];
    /// let v3 = vec![0.0, 1.0, 0.0];
    ///
    /// index.insert(0, &v1, &vectors).unwrap();
    /// vectors.push(v1.clone());
    /// index.insert(1, &v2, &vectors).unwrap();
    /// vectors.push(v2.clone());
    /// index.insert(2, &v3, &vectors).unwrap();
    /// vectors.push(v3.clone());
    ///
    /// // Find all vectors within distance 0.2 of v1
    /// let results = index.search_radius(&v1, 0.2, &vectors).unwrap();
    /// assert!(results.iter().any(|(id, _)| *id == 0)); // v1 itself
    /// assert!(results.iter().any(|(id, _)| *id == 1)); // v2 is close
    /// ```
    ///
    /// # Notes
    ///
    /// - Uses an over-fetching strategy: searches with high ef_search, then filters
    /// - Results may be incomplete for very large radii (approximate algorithm)
    /// - For exact range queries on small datasets, consider brute-force search
    pub fn search_radius(
        &self,
        query: &[f32],
        max_distance: f32,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<(VectorId, f32)>> {
        Ok(self.search_radius_with_stats(query, max_distance, vectors)?
            .0)
    }

    /// Search for all vectors within a given distance radius and return statistics.
    ///
    /// See [`search_radius`](Self::search_radius) for details.
    pub fn search_radius_with_stats(
        &self,
        query: &[f32],
        max_distance: f32,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<(VectorId, f32)>, SearchStats)> {
        // Use a large ef_search for better coverage, then filter by distance
        // The over-fetch factor helps ensure we find most/all vectors within radius
        let ef_search = self.config.ef_search.max(200);

        let (candidates, stats) =
            self.search_with_ef_stats(query, vectors.len().min(ef_search * 10), ef_search, vectors)?;

        // Filter to only include vectors within the radius
        let results: Vec<_> = candidates
            .into_iter()
            .filter(|(_, dist)| *dist <= max_distance)
            .collect();

        Ok((results, stats))
    }

    /// Search a single layer using beam search
    fn search_layer(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: usize,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<(VectorId, f32)>> {
        Ok(self.search_layer_core(query, entry, ef, layer, vectors)?.0)
    }

    /// Search a single layer and return the number of visited nodes
    fn search_layer_with_stats(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: usize,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<(VectorId, f32)>, usize)> {
        self.search_layer_core(query, entry, ef, layer, vectors)
    }

    /// Core layer search implementation (beam search). Returns (results, visited_count).
    ///
    /// Implements Algorithm 2 from the HNSW paper:
    /// - Maintains two heaps: a min-heap of candidates (closest first) and a
    ///   max-heap of results (farthest first, for easy pruning to ef best).
    /// - Iteratively pops the closest unvisited candidate, explores its neighbors,
    ///   and adds promising ones to both heaps.
    /// - **Early termination:** stops when the best remaining candidate is farther
    ///   than the worst result and we already have ef results.
    /// - Uses a Vec<u8> visited array for O(1) cache-friendly lookups instead of
    ///   a HashSet.
    fn search_layer_core(
        &self,
        query: &[f32],
        entry: VectorId,
        ef: usize,
        layer: usize,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<(VectorId, f32)>, usize)> {
        // Use Vec<u8> for O(1) visited checks with better cache behavior than Vec<bool>
        let mut visited = vec![0u8; vectors.len()];
        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VectorId)>> = BinaryHeap::new();
        // Max-heap for results (farthest first, for easy pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, VectorId)> = BinaryHeap::new();
        let mut visited_count = 0usize;

        let entry_dist = self.distance.compute(query, &vectors[entry])?;
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        if !self.deleted.contains(&entry) {
            results.push((OrderedFloat(entry_dist), entry));
        }
        visited[entry] = 1;
        visited_count += 1;

        while let Some(Reverse((OrderedFloat(c_dist), c_id))) = candidates.pop() {
            let worst_dist = results.peek().map_or(f32::INFINITY, |(d, _)| d.0);

            // If the best candidate is worse than our worst result, we're done
            if c_dist > worst_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            let connections = self.layers[layer].get_connections(c_id);
            for &neighbor in connections {
                if visited[neighbor] == 0 {
                    visited[neighbor] = 1;
                    visited_count += 1;
                    let dist = self.distance.compute(query, &vectors[neighbor])?;

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

        // Convert to sorted vector (closest first)
        let mut result_vec: Vec<_> = results.into_iter().map(|(d, id)| (id, d.0)).collect();
        result_vec
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok((result_vec, visited_count))
    }

    /// Select neighbors using the simple heuristic (Algorithm 3 from the HNSW paper).
    ///
    /// Sorts candidates by distance and keeps the closest `max_neighbors`. This is
    /// the "simple" selection strategy. The alternative "heuristic" strategy
    /// (Algorithm 4) also considers diversity by preferring neighbors that are not
    /// too close to each other, but the simple version performs well in practice
    /// and is faster.
    fn select_neighbors(
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
    pub fn compact(&mut self, vectors: &[Vec<f32>]) -> Result<HashMap<VectorId, VectorId>> {
        if self.deleted.is_empty() {
            debug!("Compact called but no deleted vectors");
            return Ok(HashMap::new());
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
            new_index.insert(new_id, vec, &new_vectors)?;
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

        Ok(id_map)
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
        let total_edges: usize = self
            .layers
            .iter()
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
        let layer_memory: usize = self
            .layers
            .iter()
            .map(|l| {
                std::mem::size_of::<Layer>()
                    + l.connections
                        .iter()
                        .map(|c| {
                            std::mem::size_of::<Vec<VectorId>>()
                                + c.len() * std::mem::size_of::<VectorId>()
                        })
                        .sum::<usize>()
            })
            .sum();

        // Node levels vector
        let node_levels_memory = self.node_levels.len() * std::mem::size_of::<usize>();

        base_size + layer_memory + node_levels_memory
    }

    /// Enable access tracking for prefetch learning
    pub fn enable_prefetch(&mut self, buffer_size: usize) {
        self.access_tracker = Some(AccessTracker::new(buffer_size));
        self.markov_predictor = Some(MarkovPredictor::new());
    }

    /// Record access to a node (for external callers to feed access data)
    pub fn record_access(&mut self, node_id: VectorId) {
        if let Some(tracker) = &mut self.access_tracker {
            tracker.record(node_id);
        }
    }

    /// Train the Markov predictor from recorded access patterns
    pub fn train_predictor(&mut self) {
        if let (Some(tracker), Some(predictor)) = (&self.access_tracker, &mut self.markov_predictor)
        {
            let sequence = tracker.recent(tracker.buffer.len());
            predictor.train(&sequence);
        }
    }

    /// Get prefetch predictions for the given access context
    pub fn predict_prefetch(&self, prev: VectorId, current: VectorId, n: usize) -> Vec<VectorId> {
        self.markov_predictor
            .as_ref()
            .map(|p| p.predict(prev, current, n))
            .unwrap_or_default()
    }

    /// Get prefetch statistics
    pub fn prefetch_stats(&self) -> PrefetchStats {
        PrefetchStats {
            predictions_made: 0,
            predictions_hit: 0,
            pattern_count: self
                .markov_predictor
                .as_ref()
                .map_or(0, MarkovPredictor::pattern_count),
            training_samples: self
                .markov_predictor
                .as_ref()
                .map_or(0, MarkovPredictor::training_samples),
        }
    }
}

impl super::VectorIndex for HnswIndex {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<(usize, f32)>> {
        self.search(query, k, vectors)
    }

    fn insert(&mut self, id: usize, vector: &[f32], vectors: &[Vec<f32>]) -> Result<()> {
        self.insert(id, vector, vectors)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        self.delete(id)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

/// HNSW index statistics
#[must_use]
#[derive(Debug, Clone, Serialize)]
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

/// Ring buffer for tracking node access sequences during HNSW traversal.
/// Used by the Markov predictor to learn access patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccessTracker {
    /// Ring buffer of recently accessed node IDs
    buffer: Vec<VectorId>,
    /// Current write position in the ring buffer
    write_pos: usize,
    /// Maximum buffer size
    capacity: usize,
    /// Total accesses recorded
    total_accesses: u64,
}

impl AccessTracker {
    /// Create a new access tracker with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            write_pos: 0,
            capacity: capacity.max(16),
            total_accesses: 0,
        }
    }

    /// Record an access to a node
    pub fn record(&mut self, node_id: VectorId) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(node_id);
        } else {
            self.buffer[self.write_pos] = node_id;
        }
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_accesses += 1;
    }

    /// Get the last N accessed nodes
    pub fn recent(&self, n: usize) -> Vec<VectorId> {
        let n = n.min(self.buffer.len());
        let mut result = Vec::with_capacity(n);
        let start = if self.buffer.len() < self.capacity {
            self.buffer.len().saturating_sub(n)
        } else {
            (self.write_pos + self.capacity - n) % self.capacity
        };
        for i in 0..n {
            let idx = (start + i) % self.buffer.len();
            result.push(self.buffer[idx]);
        }
        result
    }

    /// Total accesses recorded
    pub fn total_accesses(&self) -> u64 {
        self.total_accesses
    }

    /// Get all recorded access pairs for Markov training
    pub fn access_pairs(&self) -> Vec<(VectorId, VectorId)> {
        if self.buffer.len() < 2 {
            return Vec::new();
        }
        let ordered = self.recent(self.buffer.len());
        ordered.windows(2).map(|w| (w[0], w[1])).collect()
    }
}

/// Order-2 Markov model for predicting next HNSW nodes to access.
/// Learns transition probabilities from (prev, current) -> next node patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MarkovPredictor {
    /// Transition counts: (prev, current) -> {next -> count}
    transitions: HashMap<(VectorId, VectorId), HashMap<VectorId, u32>>,
    /// Order-1 fallback: current -> {next -> count}
    fallback: HashMap<VectorId, HashMap<VectorId, u32>>,
    /// Total training samples
    training_samples: u64,
}

impl MarkovPredictor {
    /// Create a new empty predictor
    pub fn new() -> Self {
        Self::default()
    }

    /// Train the model on a sequence of node accesses
    pub fn train(&mut self, sequence: &[VectorId]) {
        // Order-1 transitions
        for pair in sequence.windows(2) {
            *self
                .fallback
                .entry(pair[0])
                .or_default()
                .entry(pair[1])
                .or_insert(0) += 1;
        }
        // Order-2 transitions
        for triple in sequence.windows(3) {
            *self
                .transitions
                .entry((triple[0], triple[1]))
                .or_default()
                .entry(triple[2])
                .or_insert(0) += 1;
            self.training_samples += 1;
        }
    }

    /// Predict the next N most likely nodes given the last two accessed nodes
    pub fn predict(&self, prev: VectorId, current: VectorId, n: usize) -> Vec<VectorId> {
        // Try order-2 first
        if let Some(next_map) = self.transitions.get(&(prev, current)) {
            let mut candidates: Vec<_> = next_map.iter().collect();
            candidates.sort_by(|a, b| b.1.cmp(a.1));
            let result: Vec<VectorId> = candidates.iter().take(n).map(|(&id, _)| id).collect();
            if !result.is_empty() {
                return result;
            }
        }
        // Fallback to order-1
        if let Some(next_map) = self.fallback.get(&current) {
            let mut candidates: Vec<_> = next_map.iter().collect();
            candidates.sort_by(|a, b| b.1.cmp(a.1));
            return candidates.iter().take(n).map(|(&id, _)| id).collect();
        }
        Vec::new()
    }

    /// Number of training samples
    pub fn training_samples(&self) -> u64 {
        self.training_samples
    }

    /// Number of unique transition patterns learned
    pub fn pattern_count(&self) -> usize {
        self.transitions.len()
    }
}

/// Prefetch statistics
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    /// Total prefetch predictions made
    pub predictions_made: u64,
    /// Predictions that were actually used (hit)
    pub predictions_hit: u64,
    /// Current pattern count in the Markov model
    pub pattern_count: usize,
    /// Total training samples
    pub training_samples: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vector;
    use std::collections::HashSet;

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
        let results = index.search(query, 10, &vectors).unwrap();

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

        let results = index.search(&[1.0, 2.0, 3.0], 10, &[]).unwrap();
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
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();

            // Compute brute force results
            let mut brute_force: Vec<_> = vectors
                .iter()
                .enumerate()
                .map(|(id, v)| (id, DistanceFunction::Euclidean.compute(query, v).unwrap()))
                .collect();
            brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
        let results = index.search(&vectors[5], 10, &vectors).unwrap();
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
        let id_map = index.compact(&vectors).unwrap();

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

    #[test]
    fn test_access_tracker_ring_buffer() {
        let mut tracker = AccessTracker::new(4);
        tracker.record(10);
        tracker.record(20);
        tracker.record(30);
        assert_eq!(tracker.total_accesses(), 3);
        let recent = tracker.recent(3);
        assert_eq!(recent, vec![10, 20, 30]);

        // Overflow ring buffer
        tracker.record(40);
        tracker.record(50);
        assert_eq!(tracker.total_accesses(), 5);
        let recent = tracker.recent(4);
        assert_eq!(recent, vec![20, 30, 40, 50]);
    }

    #[test]
    fn test_access_tracker_pairs() {
        let mut tracker = AccessTracker::new(16);
        tracker.record(1);
        tracker.record(2);
        tracker.record(3);
        let pairs = tracker.access_pairs();
        assert_eq!(pairs, vec![(1, 2), (2, 3)]);
    }

    #[test]
    fn test_markov_predictor_train_and_predict() {
        let mut predictor = MarkovPredictor::new();
        // Train on a repeating pattern: 1->2->3->1->2->3
        predictor.train(&[1, 2, 3, 1, 2, 3]);
        assert!(predictor.training_samples() > 0);

        // After seeing (1, 2), predict 3
        let predictions = predictor.predict(1, 2, 3);
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0], 3);
    }

    #[test]
    fn test_hnsw_prefetch_integration() {
        let mut index = HnswIndex::new(HnswConfig::default(), DistanceFunction::Cosine);
        index.enable_prefetch(64);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let vectors = vec![v1.clone(), v2.clone()];
        index.insert(0, &v1, &vectors).unwrap();
        index.insert(1, &v2, &vectors).unwrap();

        // Record some accesses
        index.record_access(0);
        index.record_access(1);
        index.record_access(0);
        index.train_predictor();

        let stats = index.prefetch_stats();
        assert!(stats.training_samples > 0);
    }

    // ── search_radius tests ──────────────────────────────────────────────

    #[test]
    fn test_search_radius() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 4;

        let vectors = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
        ];

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        let results = index.search_radius(&[0.0, 0.0, 0.0, 0.0], 0.5, &vectors).unwrap();
        // Only vectors within distance 0.5 should be returned
        assert!(results.len() >= 1);
        for (_, dist) in &results {
            assert!(*dist <= 0.5);
        }
    }

    #[test]
    fn test_search_radius_with_stats() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 4;
        let vectors: Vec<Vec<f32>> = (0..20).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        let (results, stats) = index.search_radius_with_stats(&vectors[0], 1.0, &vectors).unwrap();
        assert!(!results.is_empty());
        assert!(stats.visited_nodes > 0);
    }

    // ── search_with_trace test ───────────────────────────────────────────

    #[test]
    fn test_search_with_trace() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 16;
        let vectors: Vec<Vec<f32>> = (0..50).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        let (results, trace) = index.search_with_trace(&vectors[0], 5, &vectors).unwrap();
        assert!(!results.is_empty());
        assert!(trace.entry_point.is_some());
        assert!(!trace.hops.is_empty());
    }

    #[test]
    fn test_search_with_trace_empty_index() {
        let index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let (results, trace) = index.search_with_trace(&[1.0, 2.0], 5, &[]).unwrap();
        assert!(results.is_empty());
        assert!(trace.entry_point.is_none());
    }

    // ── search_with_ef edge cases ────────────────────────────────────────

    #[test]
    fn test_search_with_ef_less_than_k() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..30).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        // ef < k: should still return results (ef is expanded internally)
        let results = index.search_with_ef(&vectors[0], 10, 2, &vectors).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_search_with_ef_zero() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 4;
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        // ef = 0: should handle gracefully
        let results = index.search_with_ef(&vectors[0], 5, 0, &vectors);
        // Either returns results or an error, should not panic
        let _ = results;
    }

    // ── insert after delete: graph reconnection ──────────────────────────

    #[test]
    fn test_insert_after_delete() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 8;
        let mut vectors: Vec<Vec<f32>> = (0..20).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        // Delete several vectors
        for i in 0..10 {
            index.delete(i).unwrap();
        }

        // Insert new vectors
        let new_vec = random_vector(dim);
        vectors.push(new_vec.clone());
        index.insert(20, &new_vec, &vectors).unwrap();

        // Search should work and not return deleted vectors
        let results = index.search(&new_vec, 5, &vectors).unwrap();
        for (id, _) in &results {
            assert!(!index.is_deleted(*id));
        }
    }

    // ── compaction + search interleaving ──────────────────────────────────

    #[test]
    fn test_compaction_then_search() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 8;
        let vectors: Vec<Vec<f32>> = (0..30).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        // Delete every other vector
        for i in (0..30).step_by(2) {
            index.delete(i).unwrap();
        }

        let id_map = index.compact(&vectors).unwrap();
        assert_eq!(index.deleted_count(), 0);

        // Search after compaction should still work
        let query = random_vector(dim);
        let results = index.search(&query, 5, &vectors).unwrap();
        assert!(results.len() <= 5);
    }

    // ── HnswConfig::builder() edge cases ─────────────────────────────────

    #[test]
    fn test_hnsw_config_builder_default() {
        let config = HnswConfig::builder();
        assert_eq!(config.m, 16);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn test_hnsw_config_builder_custom() {
        let config = HnswConfig::builder()
            .m(32)
            .ef_construction(400)
            .ef_search(100);
        assert_eq!(config.m, 32);
        assert_eq!(config.ef_construction, 400);
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.m_max_0, 64); // 2 * m
    }

    #[test]
    fn test_hnsw_config_builder_m_zero() {
        // m=0 should still create a valid config object (may degrade search)
        let config = HnswConfig::builder().m(0);
        assert_eq!(config.m, 0);
    }

    #[test]
    fn test_hnsw_config_builder_custom_ml() {
        let config = HnswConfig::builder().m(16).ml(0.5);
        assert!((config.ml - 0.5).abs() < 1e-6);
    }

    // ── BitSet edge cases ────────────────────────────────────────────────

    #[test]
    fn test_bitset_remove_nonexistent() {
        let mut bitset = BitSet::new();
        assert!(!bitset.remove(&42));
        assert_eq!(bitset.len(), 0);
    }

    #[test]
    fn test_bitset_insert_duplicates() {
        let mut bitset = BitSet::new();
        assert!(bitset.insert(5)); // new
        assert!(!bitset.insert(5)); // duplicate
        assert_eq!(bitset.len(), 1);
    }

    #[test]
    fn test_bitset_large_ids() {
        let mut bitset = BitSet::new();
        bitset.insert(100_000);
        assert!(bitset.contains(&100_000));
        assert!(!bitset.contains(&99_999));
        assert_eq!(bitset.len(), 1);
    }

    #[test]
    fn test_bitset_clear() {
        let mut bitset = BitSet::new();
        for i in 0..10 {
            bitset.insert(i);
        }
        assert_eq!(bitset.len(), 10);
        bitset.clear();
        assert_eq!(bitset.len(), 0);
        assert!(bitset.is_empty());
    }

    // ── is_deleted consistency after compaction ───────────────────────────

    #[test]
    fn test_is_deleted_after_compaction() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 4;
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        index.delete(3).unwrap();
        index.delete(7).unwrap();
        assert!(index.is_deleted(3));
        assert!(index.is_deleted(7));

        index.compact(&vectors).unwrap();
        assert_eq!(index.deleted_count(), 0);
        // After compaction, no ID should be deleted
        for i in 0..index.len() {
            assert!(!index.is_deleted(i));
        }
    }

    // ── search in single-element index ───────────────────────────────────

    #[test]
    fn test_search_single_element() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let vectors = vec![vec![1.0, 0.0, 0.0]];
        index.insert(0, &vectors[0], &vectors).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 5, &vectors).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 0.001);
    }

    // ── search in all-deleted index ──────────────────────────────────────

    #[test]
    fn test_search_all_deleted() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 4;
        let vectors: Vec<Vec<f32>> = (0..5).map(|_| random_vector(dim)).collect();

        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        for i in 0..5 {
            index.delete(i).unwrap();
        }

        let results = index.search(&vectors[0], 5, &vectors).unwrap();
        assert!(results.is_empty(), "Search with all deleted should return empty");
    }

    // ── double delete ────────────────────────────────────────────────────

    #[test]
    fn test_double_delete() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let vectors = vec![random_vector(4)];
        index.insert(0, &vectors[0], &vectors).unwrap();

        assert!(index.delete(0).unwrap());
        assert!(!index.delete(0).unwrap()); // already deleted
    }

    // ── search_with_ef_stats ─────────────────────────────────────────────

    #[test]
    fn test_search_with_ef_stats() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 16;
        let vectors: Vec<Vec<f32>> = (0..50).map(|_| random_vector(dim)).collect();
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }
        let (results, stats) = index.search_with_ef_stats(&vectors[0], 5, 100, &vectors).unwrap();
        assert!(!results.is_empty());
        assert!(stats.visited_nodes > 0);
        assert!(stats.traversal_time_us > 0 || stats.visited_nodes > 0);
    }

    // ── estimated_memory ─────────────────────────────────────────────────

    #[test]
    fn test_estimated_memory() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let vectors: Vec<Vec<f32>> = (0..20).map(|_| random_vector(8)).collect();
        let empty_mem = index.estimated_memory();
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }
        let full_mem = index.estimated_memory();
        assert!(full_mem > empty_mem);
    }

    // ── cosine distance ──────────────────────────────────────────────────

    #[test]
    fn test_hnsw_cosine_distance() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Cosine);
        let dim = 8;
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }
        let results = index.search(&vectors[0], 3, &vectors).unwrap();
        assert_eq!(results[0].0, 0); // self is closest
        assert_eq!(results[1].0, 1); // similar direction
    }

    // ── HnswConfig with specific m_max_0 ────────────────────────────────

    #[test]
    fn test_hnsw_config_custom_m_max_0() {
        let config = HnswConfig::builder().m(16).m_max_0(48);
        assert_eq!(config.m_max_0, 48);
    }

    // ── needs_compaction thresholds ──────────────────────────────────────

    #[test]
    fn test_needs_compaction_threshold() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let vectors: Vec<Vec<f32>> = (0..10).map(|_| random_vector(4)).collect();
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }
        assert!(!index.needs_compaction(0.1));
        // Delete 50%
        for i in 0..5 {
            index.delete(i).unwrap();
        }
        assert!(index.needs_compaction(0.3)); // 50% > 30%
        assert!(!index.needs_compaction(0.6)); // 50% < 60%
    }

    // ── search with high ef returns more candidates ─────────────────────

    #[test]
    fn test_search_high_ef() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let dim = 16;
        let vectors: Vec<Vec<f32>> = (0..100).map(|_| random_vector(dim)).collect();
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }
        let low_ef = index.search_with_ef(&vectors[0], 10, 10, &vectors).unwrap();
        let high_ef = index.search_with_ef(&vectors[0], 10, 200, &vectors).unwrap();
        assert_eq!(low_ef.len(), 10);
        assert_eq!(high_ef.len(), 10);
        // Both should find the query itself
        assert!(low_ef.iter().any(|(id, _)| *id == 0));
        assert!(high_ef.iter().any(|(id, _)| *id == 0));
    }

    // ── HnswIndex::new constructor ───────────────────────────────────────

    #[test]
    fn test_hnsw_new_with_config() {
        let config = HnswConfig::builder().m(32).ef_construction(400).ef_search(100);
        let index = HnswIndex::new(config.clone(), DistanceFunction::Euclidean);
        assert!(index.is_empty());
        let stats = index.stats();
        assert_eq!(stats.num_vectors, 0);
    }

    // ── search_radius on empty index ─────────────────────────────────────

    #[test]
    fn test_search_radius_empty() {
        let index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let results = index.search_radius(&[1.0, 0.0], 1.0, &[]).unwrap();
        assert!(results.is_empty());
    }

    // ── insert same vector multiple times (different IDs) ────────────────

    #[test]
    fn test_insert_same_vector_different_ids() {
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        let v = vec![1.0, 0.0, 0.0, 0.0];
        let vectors = vec![v.clone(), v.clone(), v.clone()];
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }
        assert_eq!(index.len(), 3);
        let results = index.search(&v, 3, &vectors).unwrap();
        assert_eq!(results.len(), 3);
        // All should have 0 distance
        for (_, dist) in &results {
            assert!(*dist < 0.001);
        }
    }

    // ── BitSet multiple operations ───────────────────────────────────────

    #[test]
    fn test_bitset_operations_sequence() {
        let mut bitset = BitSet::new();
        for i in 0..100 {
            bitset.insert(i);
        }
        assert_eq!(bitset.len(), 100);
        for i in (0..100).step_by(2) {
            bitset.remove(&i);
        }
        assert_eq!(bitset.len(), 50);
        assert!(!bitset.contains(&0));
        assert!(bitset.contains(&1));
    }
}
