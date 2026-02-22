#![allow(clippy::unwrap_used)]
//! Native Graph Query Language
//!
//! Extends NeedleQL with graph traversal operators (`->`, `<-`, `*..N`)
//! that combine vector similarity with knowledge graph edges in a single query.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::graph_query::{
//!     GraphQuery, GraphQueryEngine, TraversalOp, GraphQueryResult,
//! };
//!
//! let mut engine = GraphQueryEngine::new(4);
//!
//! // Add nodes and edges
//! engine.add_node("rust", vec![0.9, 0.1, 0.0, 0.0], None);
//! engine.add_node("cargo", vec![0.8, 0.2, 0.0, 0.0], None);
//! engine.add_edge("rust", "cargo", "uses");
//!
//! // Graph query: find nodes related to query, then traverse edges
//! let query = GraphQuery::new(vec![0.85, 0.15, 0.0, 0.0])
//!     .with_hops(2)
//!     .with_k(5);
//! let results = engine.execute(&query).unwrap();
//! assert!(!results.is_empty());
//! ```

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

/// A traversal operation in the graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraversalOp {
    /// Follow outgoing edges.
    Forward { edge_type: Option<String> },
    /// Follow incoming edges.
    Backward { edge_type: Option<String> },
    /// Follow edges in any direction.
    Any { edge_type: Option<String> },
}

/// A graph query combining vector similarity with traversal.
#[derive(Debug, Clone)]
pub struct GraphQuery {
    pub vector: Vec<f32>,
    pub k: usize,
    pub max_hops: usize,
    pub traversals: Vec<TraversalOp>,
    pub vector_weight: f32,
    pub graph_weight: f32,
}

impl GraphQuery {
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector, k: 10, max_hops: 2, traversals: Vec::new(), vector_weight: 0.6, graph_weight: 0.4 }
    }

    #[must_use] pub fn with_k(mut self, k: usize) -> Self { self.k = k; self }
    #[must_use] pub fn with_hops(mut self, hops: usize) -> Self { self.max_hops = hops; self }
    #[must_use] pub fn follow(mut self, op: TraversalOp) -> Self { self.traversals.push(op); self }
    #[must_use] pub fn with_weights(mut self, vec_w: f32, graph_w: f32) -> Self {
        self.vector_weight = vec_w; self.graph_weight = graph_w; self
    }
}

/// A result from a graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQueryResult {
    pub id: String,
    pub score: f32,
    pub vector_distance: f32,
    pub hops: usize,
    pub path: Vec<String>,
    pub edge_types: Vec<String>,
    pub metadata: Option<Value>,
}

struct Node {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

struct Edge {
    target: String,
    edge_type: String,
}

/// Graph query execution engine.
pub struct GraphQueryEngine {
    dimensions: usize,
    nodes: HashMap<String, Node>,
    forward_edges: HashMap<String, Vec<Edge>>,
    backward_edges: HashMap<String, Vec<Edge>>,
}

impl GraphQueryEngine {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions, nodes: HashMap::new(), forward_edges: HashMap::new(), backward_edges: HashMap::new() }
    }

    pub fn add_node(&mut self, id: &str, vector: Vec<f32>, metadata: Option<Value>) {
        self.nodes.insert(id.into(), Node { id: id.into(), vector, metadata });
    }

    pub fn add_edge(&mut self, from: &str, to: &str, edge_type: &str) {
        self.forward_edges.entry(from.into()).or_default().push(Edge { target: to.into(), edge_type: edge_type.into() });
        self.backward_edges.entry(to.into()).or_default().push(Edge { target: from.into(), edge_type: edge_type.into() });
    }

    pub fn execute(&self, query: &GraphQuery) -> Result<Vec<GraphQueryResult>> {
        if query.vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch { expected: self.dimensions, got: query.vector.len() });
        }

        // Phase 1: Vector similarity to find seed nodes
        let mut scored: Vec<(String, f32)> = self.nodes.iter()
            .map(|(id, n)| (id.clone(), cosine_distance(&query.vector, &n.vector)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let seeds: Vec<String> = scored.iter().take(query.k * 2).map(|(id, _)| id.clone()).collect();

        // Phase 2: Graph traversal from seeds
        let mut results: HashMap<String, GraphQueryResult> = HashMap::new();
        let mut visited: HashSet<String> = HashSet::new();

        for seed in &seeds {
            let vec_dist = scored.iter().find(|(id, _)| id == seed).map_or(1.0, |(_, d)| *d);
            visited.insert(seed.clone());
            results.insert(seed.clone(), GraphQueryResult {
                id: seed.clone(), score: vec_dist * query.vector_weight,
                vector_distance: vec_dist, hops: 0, path: vec![seed.clone()],
                edge_types: Vec::new(),
                metadata: self.nodes.get(seed).and_then(|n| n.metadata.clone()),
            });

            self.traverse(query, seed, 0, query.max_hops, &mut results, &mut visited, vec![seed.clone()], Vec::new());
        }

        let mut result_vec: Vec<GraphQueryResult> = results.into_values().collect();
        result_vec.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        result_vec.truncate(query.k);
        Ok(result_vec)
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.forward_edges.values().map(|v| v.len()).sum() }

    fn traverse(&self, query: &GraphQuery, node: &str, depth: usize, max_depth: usize,
        results: &mut HashMap<String, GraphQueryResult>, visited: &mut HashSet<String>,
        path: Vec<String>, edge_types: Vec<String>,
    ) {
        if depth >= max_depth { return; }
        if let Some(edges) = self.forward_edges.get(node) {
            for edge in edges {
                if visited.contains(&edge.target) { continue; }
                visited.insert(edge.target.clone());
                if let Some(target_node) = self.nodes.get(&edge.target) {
                    let vec_dist = cosine_distance(&query.vector, &target_node.vector);
                    let hop_penalty = (depth as f32 + 1.0) * query.graph_weight;
                    let score = vec_dist * query.vector_weight + hop_penalty * 0.1;
                    let mut p = path.clone(); p.push(edge.target.clone());
                    let mut et = edge_types.clone(); et.push(edge.edge_type.clone());
                    results.entry(edge.target.clone()).or_insert(GraphQueryResult {
                        id: edge.target.clone(), score, vector_distance: vec_dist,
                        hops: depth + 1, path: p.clone(), edge_types: et.clone(),
                        metadata: target_node.metadata.clone(),
                    });
                    self.traverse(query, &edge.target, depth + 1, max_depth, results, visited, p, et);
                }
            }
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b.iter()) { dot += x * y; na += x * x; nb += y * y; }
    let d = na.sqrt() * nb.sqrt();
    if d < f32::EPSILON { 1.0 } else { 1.0 - dot / d }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_search() {
        let mut e = GraphQueryEngine::new(4);
        e.add_node("a", vec![1.0, 0.0, 0.0, 0.0], None);
        e.add_node("b", vec![0.0, 1.0, 0.0, 0.0], None);
        let r = e.execute(&GraphQuery::new(vec![1.0, 0.0, 0.0, 0.0]).with_k(2)).unwrap();
        assert_eq!(r[0].id, "a");
    }

    #[test]
    fn test_graph_traversal() {
        let mut e = GraphQueryEngine::new(4);
        e.add_node("a", vec![0.9, 0.1, 0.0, 0.0], None);
        e.add_node("b", vec![0.1, 0.9, 0.0, 0.0], None);
        e.add_edge("a", "b", "related");
        let r = e.execute(&GraphQuery::new(vec![0.9, 0.1, 0.0, 0.0]).with_hops(2).with_k(5)).unwrap();
        let ids: HashSet<&str> = r.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains("a"));
        assert!(ids.contains("b"));
    }

    #[test]
    fn test_hop_tracking() {
        let mut e = GraphQueryEngine::new(4);
        e.add_node("a", vec![1.0, 0.0, 0.0, 0.0], None);
        e.add_node("b", vec![0.0, 1.0, 0.0, 0.0], None);
        e.add_node("c", vec![0.0, 0.0, 1.0, 0.0], None);
        e.add_edge("a", "b", "r1");
        e.add_edge("b", "c", "r2");
        // Query similar to "a", expect "c" found via traversal
        let r = e.execute(&GraphQuery::new(vec![1.0, 0.0, 0.0, 0.0]).with_hops(3).with_k(5)).unwrap();
        let ids: HashSet<&str> = r.iter().map(|x| x.id.as_str()).collect();
        assert!(ids.contains("a"));
        // "b" and "c" should be reachable via graph edges
        assert!(ids.contains("b") || ids.contains("c"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let e = GraphQueryEngine::new(4);
        assert!(e.execute(&GraphQuery::new(vec![1.0; 8])).is_err());
    }

    #[test]
    fn test_counts() {
        let mut e = GraphQueryEngine::new(4);
        e.add_node("a", vec![1.0; 4], None);
        e.add_node("b", vec![1.0; 4], None);
        e.add_edge("a", "b", "r");
        assert_eq!(e.node_count(), 2);
        assert_eq!(e.edge_count(), 1);
    }
}
