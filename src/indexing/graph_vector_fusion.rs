#![allow(clippy::unwrap_used)]
//! Graph-Vector Fusion Index
//!
//! Unified index structure combining knowledge graph edges with vector
//! similarity, enabling single-hop semantic traversal without separate graph
//! and vector queries.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::indexing::graph_vector_fusion::{
//!     GraphVectorFusion, FusionConfig, FusionNode, FusionEdge,
//!     EdgeType, FusionSearchResult,
//! };
//!
//! let mut index = GraphVectorFusion::new(FusionConfig::new(4));
//!
//! // Add nodes with embeddings
//! index.add_node(FusionNode::new("rust", vec![0.9, 0.1, 0.0, 0.0])).unwrap();
//! index.add_node(FusionNode::new("cargo", vec![0.8, 0.2, 0.0, 0.0])).unwrap();
//! index.add_node(FusionNode::new("crate", vec![0.7, 0.3, 0.0, 0.0])).unwrap();
//!
//! // Add typed edges
//! index.add_edge(FusionEdge::new("rust", "cargo", EdgeType::Uses)).unwrap();
//! index.add_edge(FusionEdge::new("cargo", "crate", EdgeType::Contains)).unwrap();
//!
//! // Fused search: vector similarity + graph traversal
//! let results = index.fused_search(&[0.85, 0.15, 0.0, 0.0], 5, 2).unwrap();
//! assert!(!results.is_empty());
//! ```

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the graph-vector fusion index.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Vector dimensionality.
    pub dimensions: usize,
    /// Distance function for vector similarity.
    pub distance: DistanceFunction,
    /// Maximum edges per node.
    pub max_edges: usize,
    /// Weight for vector similarity in fused scoring (0.0–1.0).
    pub vector_weight: f32,
    /// Weight for graph proximity in fused scoring (0.0–1.0).
    pub graph_weight: f32,
    /// Decay factor per hop (multiplied each hop).
    pub hop_decay: f32,
    /// Maximum traversal depth.
    pub max_depth: usize,
}

impl FusionConfig {
    /// Create a new config with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            distance: DistanceFunction::Cosine,
            max_edges: 128,
            vector_weight: 0.6,
            graph_weight: 0.4,
            hop_decay: 0.7,
            max_depth: 3,
        }
    }

    /// Set distance function.
    #[must_use]
    pub fn with_distance(mut self, dist: DistanceFunction) -> Self {
        self.distance = dist;
        self
    }

    /// Set scoring weights.
    #[must_use]
    pub fn with_weights(mut self, vector_weight: f32, graph_weight: f32) -> Self {
        self.vector_weight = vector_weight;
        self.graph_weight = graph_weight;
        self
    }

    /// Set max traversal depth.
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

// ── Edge Types ───────────────────────────────────────────────────────────────

/// Type of relationship between nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Generic "related to" relationship.
    RelatedTo,
    /// "Uses" or "depends on" relationship.
    Uses,
    /// "Contains" or "has" relationship.
    Contains,
    /// "Is a" (type/subtype) relationship.
    IsA,
    /// "Part of" relationship.
    PartOf,
    /// Custom relationship type.
    Custom(String),
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RelatedTo => write!(f, "RELATED_TO"),
            Self::Uses => write!(f, "USES"),
            Self::Contains => write!(f, "CONTAINS"),
            Self::IsA => write!(f, "IS_A"),
            Self::PartOf => write!(f, "PART_OF"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

// ── Node ─────────────────────────────────────────────────────────────────────

/// A node in the graph-vector fusion index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionNode {
    /// Unique identifier.
    pub id: String,
    /// Vector embedding.
    pub vector: Vec<f32>,
    /// Optional label (e.g., entity type).
    pub label: Option<String>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

impl FusionNode {
    /// Create a new node.
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            label: None,
            metadata: None,
        }
    }

    /// Set the node label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

// ── Edge ─────────────────────────────────────────────────────────────────────

/// An edge in the graph-vector fusion index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionEdge {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Edge type.
    pub edge_type: EdgeType,
    /// Edge weight (default 1.0).
    pub weight: f32,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

impl FusionEdge {
    /// Create a new edge.
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        edge_type: EdgeType,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            edge_type,
            weight: 1.0,
            metadata: None,
        }
    }

    /// Set edge weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

// ── Search Result ────────────────────────────────────────────────────────────

/// Result from a fused graph-vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSearchResult {
    /// Node ID.
    pub id: String,
    /// Fused score (lower is better).
    pub score: f32,
    /// Vector similarity component.
    pub vector_distance: f32,
    /// Graph proximity component.
    pub graph_distance: f32,
    /// Hops from the entry point.
    pub hops: usize,
    /// Path from entry point (if via graph traversal).
    pub path: Vec<String>,
    /// Node label.
    pub label: Option<String>,
    /// Node metadata.
    pub metadata: Option<Value>,
}

// Internal scored candidate for the priority queue
#[derive(Debug)]
struct ScoredNode {
    id: String,
    score: f32,
    vector_distance: f32,
    graph_distance: f32,
    hops: usize,
    path: Vec<String>,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering so smallest score is popped first
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
    }
}

// ── Adjacency ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct AdjEntry {
    target: String,
    edge_type: EdgeType,
    weight: f32,
}

// ── Fusion Index ─────────────────────────────────────────────────────────────

/// Graph-vector fusion index combining vector similarity with graph structure.
pub struct GraphVectorFusion {
    config: FusionConfig,
    nodes: HashMap<String, FusionNode>,
    adjacency: HashMap<String, Vec<AdjEntry>>,
    reverse_adjacency: HashMap<String, Vec<AdjEntry>>,
}

impl GraphVectorFusion {
    /// Create a new fusion index.
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// Add a node to the index.
    pub fn add_node(&mut self, node: FusionNode) -> Result<()> {
        if node.vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: node.vector.len(),
            });
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    /// Add an edge to the index.
    pub fn add_edge(&mut self, edge: FusionEdge) -> Result<()> {
        if !self.nodes.contains_key(&edge.source) {
            return Err(NeedleError::NotFound(format!(
                "Source node '{}' not found",
                edge.source
            )));
        }
        if !self.nodes.contains_key(&edge.target) {
            return Err(NeedleError::NotFound(format!(
                "Target node '{}' not found",
                edge.target
            )));
        }

        let source_adj = self.adjacency.entry(edge.source.clone()).or_default();
        if source_adj.len() >= self.config.max_edges {
            return Err(NeedleError::CapacityExceeded(format!(
                "Node '{}' exceeds max edges ({})",
                edge.source, self.config.max_edges
            )));
        }

        source_adj.push(AdjEntry {
            target: edge.target.clone(),
            edge_type: edge.edge_type.clone(),
            weight: edge.weight,
        });

        self.reverse_adjacency
            .entry(edge.target.clone())
            .or_default()
            .push(AdjEntry {
                target: edge.source,
                edge_type: edge.edge_type,
                weight: edge.weight,
            });

        Ok(())
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: &str) -> Result<bool> {
        if self.nodes.remove(id).is_none() {
            return Ok(false);
        }
        self.adjacency.remove(id);
        self.reverse_adjacency.remove(id);
        // Remove edges pointing to this node
        for adj_list in self.adjacency.values_mut() {
            adj_list.retain(|e| e.target != id);
        }
        for adj_list in self.reverse_adjacency.values_mut() {
            adj_list.retain(|e| e.target != id);
        }
        Ok(true)
    }

    /// Fused search: combine vector similarity with graph traversal.
    ///
    /// 1. Find top vector-similar nodes
    /// 2. Expand via graph edges up to `max_hops`
    /// 3. Score each candidate with weighted vector + graph distance
    pub fn fused_search(
        &self,
        query: &[f32],
        k: usize,
        max_hops: usize,
    ) -> Result<Vec<FusionSearchResult>> {
        if query.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let effective_hops = max_hops.min(self.config.max_depth);

        // Phase 1: Vector similarity search (brute-force over all nodes)
        let mut candidates: BinaryHeap<ScoredNode> = BinaryHeap::new();
        let mut visited: HashSet<String> = HashSet::new();

        for (id, node) in &self.nodes {
            let dist = self.config.distance.compute(query, &node.vector)?;
            candidates.push(ScoredNode {
                id: id.clone(),
                score: dist * self.config.vector_weight,
                vector_distance: dist,
                graph_distance: 0.0,
                hops: 0,
                path: vec![id.clone()],
            });
        }

        // Phase 2: Graph expansion from top vector matches
        let initial_k = (k * 3).min(candidates.len()); // seed with top 3×k
        let mut seeds: Vec<ScoredNode> = Vec::new();
        for _ in 0..initial_k {
            if let Some(node) = candidates.pop() {
                seeds.push(node);
            }
        }

        // Re-insert seeds and expand
        let mut all_candidates: HashMap<String, ScoredNode> = HashMap::new();
        for seed in seeds {
            visited.insert(seed.id.clone());
            self.expand_graph(
                query,
                &seed,
                effective_hops,
                &mut all_candidates,
                &mut visited,
            );
            all_candidates.insert(seed.id.clone(), seed);
        }

        // Phase 3: Sort by fused score and return top-k
        let mut results: Vec<FusionSearchResult> = all_candidates
            .into_values()
            .map(|c| {
                let node = self.nodes.get(&c.id);
                FusionSearchResult {
                    id: c.id,
                    score: c.score,
                    vector_distance: c.vector_distance,
                    graph_distance: c.graph_distance,
                    hops: c.hops,
                    path: c.path,
                    label: node.and_then(|n| n.label.clone()),
                    metadata: node.and_then(|n| n.metadata.clone()),
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Vector-only search (no graph expansion).
    pub fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<FusionSearchResult>> {
        if query.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.len(),
            });
        }

        let mut scored: Vec<(String, f32)> = Vec::new();
        for (id, node) in &self.nodes {
            let dist = self.config.distance.compute(query, &node.vector)?;
            scored.push((id.clone(), dist));
        }

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(id, dist)| {
                let node = self.nodes.get(&id);
                FusionSearchResult {
                    id: id.clone(),
                    score: dist,
                    vector_distance: dist,
                    graph_distance: 0.0,
                    hops: 0,
                    path: vec![id],
                    label: node.and_then(|n| n.label.clone()),
                    metadata: node.and_then(|n| n.metadata.clone()),
                }
            })
            .collect())
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, id: &str) -> Vec<(&str, &EdgeType, f32)> {
        self.adjacency
            .get(id)
            .map(|adj| {
                adj.iter()
                    .map(|e| (e.target.as_str(), &e.edge_type, e.weight))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the total number of edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&FusionNode> {
        self.nodes.get(id)
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn expand_graph(
        &self,
        query: &[f32],
        seed: &ScoredNode,
        remaining_hops: usize,
        candidates: &mut HashMap<String, ScoredNode>,
        visited: &mut HashSet<String>,
    ) {
        if remaining_hops == 0 {
            return;
        }

        let neighbors = self.adjacency.get(&seed.id);
        if let Some(adj) = neighbors {
            for entry in adj {
                if visited.contains(&entry.target) {
                    continue;
                }
                visited.insert(entry.target.clone());

                if let Some(node) = self.nodes.get(&entry.target) {
                    let vector_dist = self.config.distance.compute(query, &node.vector).unwrap_or(f32::MAX);
                    let hop = seed.hops + 1;
                    let graph_dist = 1.0 / (entry.weight * self.config.hop_decay.powi(hop as i32));
                    let fused_score = vector_dist * self.config.vector_weight
                        + graph_dist * self.config.graph_weight;

                    let mut path = seed.path.clone();
                    path.push(entry.target.clone());

                    let candidate = ScoredNode {
                        id: entry.target.clone(),
                        score: fused_score,
                        vector_distance: vector_dist,
                        graph_distance: graph_dist,
                        hops: hop,
                        path,
                    };

                    // Recursively expand
                    self.expand_graph(query, &candidate, remaining_hops - 1, candidates, visited);

                    // Keep best score for each node
                    let insert = candidates
                        .get(&entry.target)
                        .map_or(true, |existing| candidate.score < existing.score);
                    if insert {
                        candidates.insert(entry.target.clone(), candidate);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn create_test_index() -> GraphVectorFusion {
        let config = FusionConfig::new(4);
        let mut index = GraphVectorFusion::new(config);

        index
            .add_node(
                FusionNode::new("rust", vec![0.9, 0.1, 0.0, 0.0])
                    .with_label("language"),
            )
            .unwrap();
        index
            .add_node(
                FusionNode::new("cargo", vec![0.8, 0.2, 0.0, 0.0])
                    .with_label("tool"),
            )
            .unwrap();
        index
            .add_node(
                FusionNode::new("crate", vec![0.7, 0.3, 0.0, 0.0])
                    .with_label("package"),
            )
            .unwrap();
        index
            .add_node(
                FusionNode::new("python", vec![0.1, 0.9, 0.0, 0.0])
                    .with_label("language"),
            )
            .unwrap();

        index
            .add_edge(FusionEdge::new("rust", "cargo", EdgeType::Uses))
            .unwrap();
        index
            .add_edge(FusionEdge::new("cargo", "crate", EdgeType::Contains))
            .unwrap();
        index
            .add_edge(FusionEdge::new("rust", "crate", EdgeType::Uses))
            .unwrap();

        index
    }

    #[test]
    fn test_add_and_count() {
        let index = create_test_index();
        assert_eq!(index.node_count(), 4);
        assert_eq!(index.edge_count(), 3);
    }

    #[test]
    fn test_vector_search() {
        let index = create_test_index();
        let results = index.vector_search(&[0.85, 0.15, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "rust"); // closest
    }

    #[test]
    fn test_fused_search() {
        let index = create_test_index();
        let results = index
            .fused_search(&[0.85, 0.15, 0.0, 0.0], 4, 2)
            .unwrap();
        assert!(!results.is_empty());
        // Rust should be top result (closest vector + graph hub)
        assert_eq!(results[0].id, "rust");
    }

    #[test]
    fn test_graph_traversal_discovers_related() {
        let index = create_test_index();
        let results = index
            .fused_search(&[0.9, 0.1, 0.0, 0.0], 4, 2)
            .unwrap();

        // Should find cargo and crate via graph edges from rust
        let ids: HashSet<&str> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains("rust"));
        assert!(ids.contains("cargo"));
    }

    #[test]
    fn test_neighbors() {
        let index = create_test_index();
        let neighbors = index.neighbors("rust");
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let index = create_test_index();
        assert!(index.fused_search(&[1.0, 2.0], 5, 2).is_err());
    }

    #[test]
    fn test_remove_node() {
        let mut index = create_test_index();
        assert!(index.remove_node("cargo").unwrap());
        assert_eq!(index.node_count(), 3);
        assert!(index.get_node("cargo").is_none());
    }

    #[test]
    fn test_edge_types() {
        assert_eq!(format!("{}", EdgeType::Uses), "USES");
        assert_eq!(
            format!("{}", EdgeType::Custom("WORKS_WITH".into())),
            "WORKS_WITH"
        );
    }

    #[test]
    fn test_fusion_result_has_path() {
        let index = create_test_index();
        let results = index
            .fused_search(&[0.9, 0.1, 0.0, 0.0], 4, 2)
            .unwrap();

        for result in &results {
            assert!(!result.path.is_empty());
            // First element in path should be the node itself (or seed)
        }
    }
}
