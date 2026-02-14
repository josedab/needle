//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Semantic Graph Traversal
//!
//! Explore vector neighborhoods and relationships through graph-based operations:
//! - K-nearest neighbor graph construction
//! - Semantic path finding between vectors
//! - Community detection in vector space
//! - Graph-based exploration and visualization
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::graph::{SemanticGraph, GraphConfig};
//!
//! let vectors: Vec<Vec<f32>> = /* your vectors */;
//! let ids: Vec<String> = /* vector IDs */;
//!
//! let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::default())?;
//!
//! // Find semantic path between two vectors
//! let path = graph.find_path("doc1", "doc2");
//!
//! // Explore neighborhood
//! let neighbors = graph.get_neighborhood("doc1", 2);
//!
//! // Detect communities
//! let communities = graph.detect_communities();
//! ```

use crate::distance::DistanceFunction;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// Configuration for semantic graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// Number of nearest neighbors to connect
    pub k: usize,
    /// Distance function
    pub distance: DistanceFunction,
    /// Similarity threshold for edges (1 - distance)
    pub threshold: Option<f32>,
    /// Make graph symmetric (undirected)
    pub symmetric: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            k: 10,
            distance: DistanceFunction::Cosine,
            threshold: None,
            symmetric: true,
        }
    }
}

impl GraphConfig {
    /// Create config with specific k
    pub fn with_k(k: usize) -> Self {
        Self {
            k,
            ..Default::default()
        }
    }

    /// Set similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
}

/// Edge in the semantic graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub from: String,
    /// Target node ID
    pub to: String,
    /// Similarity score (1 - distance)
    pub similarity: f32,
}

/// Node in the semantic graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Node ID
    pub id: String,
    /// Outgoing edges
    pub edges: Vec<Edge>,
    /// Node degree (number of connections)
    pub degree: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f32,
}

/// Semantic graph for exploring vector relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticGraph {
    /// Configuration
    config: GraphConfig,
    /// Nodes indexed by ID
    nodes: HashMap<String, Node>,
    /// Vectors indexed by ID
    vectors: HashMap<String, Vec<f32>>,
    /// Number of edges
    num_edges: usize,
}

impl SemanticGraph {
    /// Build semantic graph from vectors
    pub fn build(
        vectors: &[Vec<f32>],
        ids: &[String],
        config: GraphConfig,
    ) -> Result<Self, String> {
        if vectors.len() != ids.len() {
            return Err("Vectors and IDs must have same length".to_string());
        }

        let n = vectors.len();
        if n == 0 {
            return Ok(Self {
                config,
                nodes: HashMap::new(),
                vectors: HashMap::new(),
                num_edges: 0,
            });
        }

        // Store vectors
        let stored_vectors: HashMap<String, Vec<f32>> = ids
            .iter()
            .zip(vectors.iter())
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();

        // Build k-NN graph
        let mut nodes: HashMap<String, Node> = ids
            .iter()
            .map(|id| {
                (
                    id.clone(),
                    Node {
                        id: id.clone(),
                        edges: Vec::new(),
                        degree: 0,
                        clustering_coefficient: 0.0,
                    },
                )
            })
            .collect();

        let mut num_edges = 0;

        for i in 0..n {
            // Find k nearest neighbors
            let mut neighbors: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let dist = config.distance.compute(&vectors[i], &vectors[j]);
                    (j, dist)
                })
                .collect();

            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            neighbors.truncate(config.k);

            // Add edges
            for (j, dist) in neighbors {
                let similarity = 1.0 - dist.min(1.0);

                // Check threshold
                if let Some(thresh) = config.threshold {
                    if similarity < thresh {
                        continue;
                    }
                }

                let edge = Edge {
                    from: ids[i].clone(),
                    to: ids[j].clone(),
                    similarity,
                };

                if let Some(node) = nodes.get_mut(&ids[i]) {
                    node.edges.push(edge);
                    num_edges += 1;
                }

                // Add reverse edge for symmetric graph
                if config.symmetric {
                    let reverse_edge = Edge {
                        from: ids[j].clone(),
                        to: ids[i].clone(),
                        similarity,
                    };

                    // Check if reverse edge already exists
                    if let Some(node) = nodes.get_mut(&ids[j]) {
                        if !node.edges.iter().any(|e| e.to == ids[i]) {
                            node.edges.push(reverse_edge);
                            num_edges += 1;
                        }
                    }
                }
            }
        }

        // Calculate node degrees and clustering coefficients
        for node in nodes.values_mut() {
            node.degree = node.edges.len();
        }

        // Calculate clustering coefficients
        let node_ids: Vec<String> = nodes.keys().cloned().collect();
        for id in &node_ids {
            let cc = Self::calculate_clustering_coefficient(&nodes, id);
            if let Some(node) = nodes.get_mut(id) {
                node.clustering_coefficient = cc;
            }
        }

        Ok(Self {
            config,
            nodes,
            vectors: stored_vectors,
            num_edges,
        })
    }

    /// Calculate clustering coefficient for a node
    fn calculate_clustering_coefficient(nodes: &HashMap<String, Node>, id: &str) -> f32 {
        let node = match nodes.get(id) {
            Some(n) => n,
            None => return 0.0,
        };

        let neighbors: HashSet<&str> = node.edges.iter().map(|e| e.to.as_str()).collect();
        let k = neighbors.len();

        if k < 2 {
            return 0.0;
        }

        // Count edges between neighbors
        let mut edges_between = 0;
        for neighbor_id in &neighbors {
            if let Some(neighbor) = nodes.get(*neighbor_id) {
                for edge in &neighbor.edges {
                    if neighbors.contains(edge.to.as_str()) {
                        edges_between += 1;
                    }
                }
            }
        }

        // Clustering coefficient = actual edges / possible edges
        let possible_edges = k * (k - 1);
        if possible_edges > 0 {
            edges_between as f32 / possible_edges as f32
        } else {
            0.0
        }
    }

    /// Get node by ID
    pub fn get_node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Get vector by ID
    pub fn get_vector(&self, id: &str) -> Option<&Vec<f32>> {
        self.vectors.get(id)
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    /// Number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, id: &str) -> Vec<(&str, f32)> {
        self.nodes
            .get(id)
            .map(|n| {
                n.edges
                    .iter()
                    .map(|e| (e.to.as_str(), e.similarity))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get extended neighborhood (BFS up to max_hops)
    pub fn get_neighborhood(&self, id: &str, max_hops: usize) -> NeighborhoodResult {
        let mut visited: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let mut edges: Vec<Edge> = Vec::new();

        if self.nodes.contains_key(id) {
            queue.push_back((id.to_string(), 0));
            visited.insert(id.to_string(), 0);
        }

        while let Some((current, hop)) = queue.pop_front() {
            if hop >= max_hops {
                continue;
            }

            if let Some(node) = self.nodes.get(&current) {
                for edge in &node.edges {
                    if !visited.contains_key(&edge.to) {
                        visited.insert(edge.to.clone(), hop + 1);
                        queue.push_back((edge.to.clone(), hop + 1));
                    }
                    edges.push(edge.clone());
                }
            }
        }

        // Deduplicate edges
        let mut seen: HashSet<(String, String)> = HashSet::new();
        edges.retain(|e| {
            let key = if e.from < e.to {
                (e.from.clone(), e.to.clone())
            } else {
                (e.to.clone(), e.from.clone())
            };
            seen.insert(key)
        });

        NeighborhoodResult {
            center: id.to_string(),
            nodes: visited.into_iter().collect(),
            edges,
        }
    }

    /// Find shortest path between two nodes (by edge count)
    pub fn find_path(&self, from: &str, to: &str) -> Option<GraphPath> {
        if from == to {
            return Some(GraphPath {
                nodes: vec![from.to_string()],
                edges: Vec::new(),
                total_similarity: 1.0,
            });
        }

        // BFS for shortest path
        let mut visited: HashMap<String, (String, f32)> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string(), (String::new(), 0.0));

        while let Some(current) = queue.pop_front() {
            if let Some(node) = self.nodes.get(&current) {
                for edge in &node.edges {
                    if !visited.contains_key(&edge.to) {
                        visited.insert(edge.to.clone(), (current.clone(), edge.similarity));
                        queue.push_back(edge.to.clone());

                        if edge.to == to {
                            // Reconstruct path
                            return Some(self.reconstruct_path(from, to, &visited));
                        }
                    }
                }
            }
        }

        None
    }

    /// Find highest-similarity path (Dijkstra variant)
    pub fn find_best_path(&self, from: &str, to: &str) -> Option<GraphPath> {
        if from == to {
            return Some(GraphPath {
                nodes: vec![from.to_string()],
                edges: Vec::new(),
                total_similarity: 1.0,
            });
        }

        // Max-heap using similarity (negate for min-heap behavior)
        let mut heap: BinaryHeap<(ordered_float::OrderedFloat<f32>, String)> = BinaryHeap::new();
        let mut best_sim: HashMap<String, f32> = HashMap::new();
        let mut parent: HashMap<String, (String, f32)> = HashMap::new();

        heap.push((ordered_float::OrderedFloat(1.0), from.to_string()));
        best_sim.insert(from.to_string(), 1.0);

        while let Some((sim, current)) = heap.pop() {
            let sim = sim.0;

            if current == to {
                return Some(self.reconstruct_path_sim(from, to, &parent));
            }

            if sim < best_sim.get(&current).copied().unwrap_or(0.0) {
                continue;
            }

            if let Some(node) = self.nodes.get(&current) {
                for edge in &node.edges {
                    let new_sim = sim * edge.similarity;

                    if new_sim > best_sim.get(&edge.to).copied().unwrap_or(0.0) {
                        best_sim.insert(edge.to.clone(), new_sim);
                        parent.insert(edge.to.clone(), (current.clone(), edge.similarity));
                        heap.push((ordered_float::OrderedFloat(new_sim), edge.to.clone()));
                    }
                }
            }
        }

        None
    }

    fn reconstruct_path(
        &self,
        _from: &str,
        to: &str,
        visited: &HashMap<String, (String, f32)>,
    ) -> GraphPath {
        let mut nodes = vec![to.to_string()];
        let mut edges = Vec::new();
        let mut current = to.to_string();
        let mut total_similarity = 1.0;

        while let Some((prev, sim)) = visited.get(&current) {
            if prev.is_empty() {
                break;
            }
            edges.push(Edge {
                from: prev.clone(),
                to: current.clone(),
                similarity: *sim,
            });
            total_similarity *= sim;
            nodes.push(prev.clone());
            current = prev.clone();
        }

        nodes.reverse();
        edges.reverse();

        GraphPath {
            nodes,
            edges,
            total_similarity,
        }
    }

    fn reconstruct_path_sim(
        &self,
        from: &str,
        to: &str,
        parent: &HashMap<String, (String, f32)>,
    ) -> GraphPath {
        let mut nodes = vec![to.to_string()];
        let mut edges = Vec::new();
        let mut current = to.to_string();
        let mut total_similarity = 1.0;

        while let Some((prev, sim)) = parent.get(&current) {
            edges.push(Edge {
                from: prev.clone(),
                to: current.clone(),
                similarity: *sim,
            });
            total_similarity *= sim;
            nodes.push(prev.clone());
            current = prev.clone();
            if current == from {
                break;
            }
        }

        nodes.reverse();
        edges.reverse();

        GraphPath {
            nodes,
            edges,
            total_similarity,
        }
    }

    /// Detect communities using label propagation
    pub fn detect_communities(&self) -> Vec<Community> {
        let mut labels: HashMap<String, usize> = HashMap::new();

        // Initialize each node with unique label
        for (i, id) in self.nodes.keys().enumerate() {
            labels.insert(id.clone(), i);
        }

        // Label propagation
        let max_iterations = 100;
        for _ in 0..max_iterations {
            let mut changed = false;

            for (id, node) in &self.nodes {
                if node.edges.is_empty() {
                    continue;
                }

                // Count neighbor labels (weighted by similarity)
                let mut label_weights: HashMap<usize, f32> = HashMap::new();
                for edge in &node.edges {
                    if let Some(&neighbor_label) = labels.get(&edge.to) {
                        *label_weights.entry(neighbor_label).or_insert(0.0) += edge.similarity;
                    }
                }

                // Find most common label
                if let Some((&best_label, _)) = label_weights
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    if labels.get(id) != Some(&best_label) {
                        labels.insert(id.clone(), best_label);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Group nodes by label
        let mut communities: HashMap<usize, Vec<String>> = HashMap::new();
        for (id, label) in &labels {
            communities.entry(*label).or_default().push(id.clone());
        }

        // Convert to Community structs
        communities
            .into_iter()
            .enumerate()
            .map(|(idx, (_, members))| {
                let internal_edges: usize = members
                    .iter()
                    .filter_map(|id| self.nodes.get(id))
                    .flat_map(|n| &n.edges)
                    .filter(|e| members.contains(&e.to))
                    .count();

                let avg_similarity = if internal_edges > 0 {
                    members
                        .iter()
                        .filter_map(|id| self.nodes.get(id))
                        .flat_map(|n| &n.edges)
                        .filter(|e| members.contains(&e.to))
                        .map(|e| e.similarity)
                        .sum::<f32>()
                        / internal_edges as f32
                } else {
                    0.0
                };

                Community {
                    id: idx,
                    members,
                    internal_edges,
                    avg_similarity,
                }
            })
            .collect()
    }

    /// Get graph statistics
    pub fn stats(&self) -> GraphStats {
        let num_nodes = self.nodes.len();
        let num_edges = self.num_edges;

        let degrees: Vec<usize> = self.nodes.values().map(|n| n.degree).collect();
        let avg_degree = if num_nodes > 0 {
            degrees.iter().sum::<usize>() as f32 / num_nodes as f32
        } else {
            0.0
        };

        let avg_clustering = if num_nodes > 0 {
            self.nodes
                .values()
                .map(|n| n.clustering_coefficient)
                .sum::<f32>()
                / num_nodes as f32
        } else {
            0.0
        };

        let avg_similarity = if num_edges > 0 {
            self.nodes
                .values()
                .flat_map(|n| &n.edges)
                .map(|e| e.similarity)
                .sum::<f32>()
                / num_edges as f32
        } else {
            0.0
        };

        GraphStats {
            num_nodes,
            num_edges,
            avg_degree,
            avg_clustering_coefficient: avg_clustering,
            avg_edge_similarity: avg_similarity,
            density: if num_nodes > 1 {
                2.0 * num_edges as f32 / (num_nodes * (num_nodes - 1)) as f32
            } else {
                0.0
            },
        }
    }

    /// Find most central nodes (by degree)
    pub fn top_nodes_by_degree(&self, k: usize) -> Vec<(&str, usize)> {
        let mut nodes: Vec<(&str, usize)> = self
            .nodes
            .iter()
            .map(|(id, n)| (id.as_str(), n.degree))
            .collect();

        nodes.sort_by(|a, b| b.1.cmp(&a.1));
        nodes.truncate(k);
        nodes
    }

    /// Find nodes similar to a given vector (not in graph)
    pub fn find_similar(&self, vector: &[f32], k: usize) -> Vec<(&str, f32)> {
        let mut similarities: Vec<(&str, f32)> = self
            .vectors
            .iter()
            .map(|(id, v)| {
                let dist = self.config.distance.compute(vector, v);
                (id.as_str(), 1.0 - dist.min(1.0))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }

    /// Export graph to edge list format
    pub fn to_edge_list(&self) -> Vec<(String, String, f32)> {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut edges = Vec::new();

        for node in self.nodes.values() {
            for edge in &node.edges {
                let key = if edge.from < edge.to {
                    (edge.from.clone(), edge.to.clone())
                } else {
                    (edge.to.clone(), edge.from.clone())
                };

                if !seen.contains(&key) {
                    seen.insert(key);
                    edges.push((edge.from.clone(), edge.to.clone(), edge.similarity));
                }
            }
        }

        edges
    }
}

/// Result of neighborhood exploration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborhoodResult {
    /// Center node ID
    pub center: String,
    /// Nodes with their hop distance from center
    pub nodes: Vec<(String, usize)>,
    /// Edges in the neighborhood
    pub edges: Vec<Edge>,
}

/// Path through the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    /// Nodes in path order
    pub nodes: Vec<String>,
    /// Edges in path
    pub edges: Vec<Edge>,
    /// Product of edge similarities
    pub total_similarity: f32,
}

/// Community detected in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Community ID
    pub id: usize,
    /// Member node IDs
    pub members: Vec<String>,
    /// Number of internal edges
    pub internal_edges: usize,
    /// Average similarity within community
    pub avg_similarity: f32,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of edges
    pub num_edges: usize,
    /// Average node degree
    pub avg_degree: f32,
    /// Average clustering coefficient
    pub avg_clustering_coefficient: f32,
    /// Average edge similarity
    pub avg_edge_similarity: f32,
    /// Graph density
    pub density: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Vec<Vec<f32>>, Vec<String>) {
        // Create two clusters
        let mut vectors = Vec::new();
        let mut ids = Vec::new();

        // Cluster 1: vectors near [1, 0, 0, ...]
        for i in 0..10 {
            let mut v = vec![0.0; 8];
            v[0] = 1.0 + (i as f32 * 0.05);
            v[1] = 0.1 * (i as f32 % 3.0);
            vectors.push(v);
            ids.push(format!("cluster1_{}", i));
        }

        // Cluster 2: vectors near [0, 1, 0, ...]
        for i in 0..10 {
            let mut v = vec![0.0; 8];
            v[1] = 1.0 + (i as f32 * 0.05);
            v[0] = 0.1 * (i as f32 % 3.0);
            vectors.push(v);
            ids.push(format!("cluster2_{}", i));
        }

        (vectors, ids)
    }

    #[test]
    fn test_graph_build() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        assert_eq!(graph.num_nodes(), 20);
        assert!(graph.num_edges() > 0);
    }

    #[test]
    fn test_neighbors() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        let neighbors = graph.neighbors("cluster1_0");
        assert!(!neighbors.is_empty());

        // Neighbors should be from same cluster mostly
        let same_cluster = neighbors
            .iter()
            .filter(|(id, _)| id.starts_with("cluster1_"))
            .count();
        assert!(same_cluster > 0);
    }

    #[test]
    fn test_find_path() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        // Path within same cluster should exist
        let path = graph.find_path("cluster1_0", "cluster1_5");
        assert!(path.is_some());

        let path = path.unwrap();
        assert!(!path.nodes.is_empty());
        assert_eq!(path.nodes.first().unwrap(), "cluster1_0");
        assert_eq!(path.nodes.last().unwrap(), "cluster1_5");
    }

    #[test]
    fn test_best_path() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        let path = graph.find_best_path("cluster1_0", "cluster1_5");
        assert!(path.is_some());
    }

    #[test]
    fn test_neighborhood() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        let neighborhood = graph.get_neighborhood("cluster1_0", 2);
        assert_eq!(neighborhood.center, "cluster1_0");
        assert!(!neighborhood.nodes.is_empty());
    }

    #[test]
    fn test_detect_communities() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(3)).unwrap();

        let communities = graph.detect_communities();
        assert!(!communities.is_empty());

        // Total members should equal total nodes
        let total_members: usize = communities.iter().map(|c| c.members.len()).sum();
        assert_eq!(total_members, 20);
    }

    #[test]
    fn test_graph_stats() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        let stats = graph.stats();
        assert_eq!(stats.num_nodes, 20);
        assert!(stats.avg_degree > 0.0);
        assert!(stats.density >= 0.0 && stats.density <= 1.0);
    }

    #[test]
    fn test_top_nodes_by_degree() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        let top = graph.top_nodes_by_degree(5);
        assert_eq!(top.len(), 5);

        // Should be sorted descending by degree
        for i in 1..top.len() {
            assert!(top[i - 1].1 >= top[i].1);
        }
    }

    #[test]
    fn test_find_similar() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        // Query similar to cluster 1
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let similar = graph.find_similar(&query, 5);

        assert_eq!(similar.len(), 5);
        // Most similar should be from cluster 1
        assert!(similar[0].0.starts_with("cluster1_"));
    }

    #[test]
    fn test_edge_list_export() {
        let (vectors, ids) = create_test_data();
        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::with_k(5)).unwrap();

        let edges = graph.to_edge_list();
        assert!(!edges.is_empty());

        // All similarities should be valid
        assert!(edges.iter().all(|(_, _, s)| *s >= 0.0 && *s <= 1.0));
    }

    #[test]
    fn test_empty_graph() {
        let vectors: Vec<Vec<f32>> = Vec::new();
        let ids: Vec<String> = Vec::new();

        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::default()).unwrap();

        assert_eq!(graph.num_nodes(), 0);
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_single_node() {
        let vectors = vec![vec![1.0, 0.0, 0.0]];
        let ids = vec!["single".to_string()];

        let graph = SemanticGraph::build(&vectors, &ids, GraphConfig::default()).unwrap();

        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.neighbors("single").len(), 0);
    }
}
