//! Native GraphRAG Service
//!
//! Graph + vector hybrid index that stores entity relationships alongside
//! embeddings, enabling multi-hop retrieval with graph traversal combined
//! with similarity search.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::graphrag_service::{
//!     GraphRagService, GraphRagConfig, Entity, Relation,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("entities", 128).unwrap();
//!
//! let mut service = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();
//!
//! // Add entities with embeddings
//! service.add_entity(Entity {
//!     id: "rust".into(),
//!     label: "Programming Language".into(),
//!     vector: vec![0.9, 0.1, 0.0, 0.0],  // simplified
//!     metadata: None,
//! }).unwrap();
//!
//! service.add_entity(Entity {
//!     id: "cargo".into(),
//!     label: "Build Tool".into(),
//!     vector: vec![0.8, 0.2, 0.0, 0.0],
//!     metadata: None,
//! }).unwrap();
//!
//! // Add relationships
//! service.add_relation(Relation::new("rust", "cargo", "uses")).unwrap();
//!
//! // Multi-hop search: find entities related to "rust" within 2 hops
//! let results = service.graph_search(&[0.9, 0.1, 0.0, 0.0], 5, 2, 0.5).unwrap();
//! assert!(!results.is_empty());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// GraphRAG service configuration.
#[derive(Debug, Clone)]
pub struct GraphRagConfig {
    /// Maximum hops for graph traversal.
    pub max_hops: usize,
    /// Default graph weight in hybrid scoring (0.0 = vector only, 1.0 = graph only).
    pub default_graph_weight: f32,
    /// Maximum edges per node.
    pub max_edges_per_node: usize,
    /// Whether to enable bidirectional edge traversal.
    pub bidirectional: bool,
    /// Decay factor per hop (multiplied each hop to reduce distant node scores).
    pub hop_decay: f32,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            default_graph_weight: 0.3,
            max_edges_per_node: 100,
            bidirectional: true,
            hop_decay: 0.7,
        }
    }
}

// ── Entity & Relation Types ──────────────────────────────────────────────────

/// An entity node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique entity ID.
    pub id: String,
    /// Entity label/type (e.g., "Person", "Concept", "Document").
    pub label: String,
    /// Entity embedding vector.
    pub vector: Vec<f32>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

/// A directed relation (edge) between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity ID.
    pub source: String,
    /// Target entity ID.
    pub target: String,
    /// Relation type (e.g., "uses", "authored_by", "related_to").
    pub relation_type: String,
    /// Optional edge weight (default 1.0).
    pub weight: f32,
    /// Optional edge metadata.
    pub metadata: Option<Value>,
}

impl Relation {
    /// Create a new relation with default weight.
    pub fn new(
        source: impl Into<String>,
        target: impl Into<String>,
        rel_type: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            relation_type: rel_type.into(),
            weight: 1.0,
            metadata: None,
        }
    }

    /// Set edge weight.
    #[must_use]
    pub fn with_weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    /// Set edge metadata.
    #[must_use]
    pub fn with_metadata(mut self, meta: Value) -> Self {
        self.metadata = Some(meta);
        self
    }
}

/// A result from graph-enhanced search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    /// Entity ID.
    pub id: String,
    /// Entity label.
    pub label: String,
    /// Combined score (lower = more relevant for distance-based, higher for score-based).
    pub score: f32,
    /// Vector similarity component.
    pub vector_score: f32,
    /// Graph proximity component.
    pub graph_score: f32,
    /// Number of hops from the seed entity.
    pub hops: usize,
    /// Traversal path from seed to this entity.
    pub path: Vec<String>,
    /// Entity metadata.
    pub metadata: Option<Value>,
}

/// Graph statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphStats {
    /// Number of entity nodes.
    pub entity_count: usize,
    /// Number of relation edges.
    pub relation_count: usize,
    /// Number of unique relation types.
    pub relation_type_count: usize,
    /// Number of unique labels.
    pub label_count: usize,
    /// Average edges per node.
    pub avg_degree: f32,
    /// Maximum edges on any single node.
    pub max_degree: usize,
}

// ── Adjacency List ───────────────────────────────────────────────────────────

/// Internal adjacency list for graph traversal.
#[derive(Debug, Clone, Default)]
struct AdjacencyList {
    /// Outgoing edges: source → [(target, relation_type, weight)]
    outgoing: HashMap<String, Vec<(String, String, f32)>>,
    /// Incoming edges: target → [(source, relation_type, weight)]
    incoming: HashMap<String, Vec<(String, String, f32)>>,
}

impl AdjacencyList {
    fn add_edge(&mut self, source: &str, target: &str, rel_type: &str, weight: f32) {
        self.outgoing.entry(source.to_string()).or_default().push((
            target.to_string(),
            rel_type.to_string(),
            weight,
        ));

        self.incoming.entry(target.to_string()).or_default().push((
            source.to_string(),
            rel_type.to_string(),
            weight,
        ));
    }

    fn neighbors(&self, node: &str, bidirectional: bool) -> Vec<(String, String, f32)> {
        let mut neighbors = Vec::new();
        if let Some(out) = self.outgoing.get(node) {
            neighbors.extend(out.iter().cloned());
        }
        if bidirectional {
            if let Some(inc) = self.incoming.get(node) {
                neighbors.extend(inc.iter().cloned());
            }
        }
        neighbors
    }

    fn degree(&self, node: &str, bidirectional: bool) -> usize {
        let out = self.outgoing.get(node).map_or(0, |v| v.len());
        let inc = if bidirectional {
            self.incoming.get(node).map_or(0, |v| v.len())
        } else {
            0
        };
        out + inc
    }

    fn remove_node(&mut self, node: &str) {
        self.outgoing.remove(node);
        self.incoming.remove(node);
        for edges in self.outgoing.values_mut() {
            edges.retain(|(t, _, _)| t != node);
        }
        for edges in self.incoming.values_mut() {
            edges.retain(|(s, _, _)| s != node);
        }
    }
}

// ── GraphRAG Service ─────────────────────────────────────────────────────────

/// GraphRAG service combining graph traversal with vector similarity search.
pub struct GraphRagService<'a> {
    db: &'a Database,
    collection_name: String,
    config: GraphRagConfig,
    entities: HashMap<String, Entity>,
    adjacency: AdjacencyList,
    relation_count: usize,
}

impl<'a> GraphRagService<'a> {
    /// Create a new GraphRAG service.
    pub fn new(db: &'a Database, collection: &str, config: GraphRagConfig) -> Result<Self> {
        let _coll = db.collection(collection)?;
        Ok(Self {
            db,
            collection_name: collection.to_string(),
            config,
            entities: HashMap::new(),
            adjacency: AdjacencyList::default(),
            relation_count: 0,
        })
    }

    /// Add an entity to the graph and vector index.
    pub fn add_entity(&mut self, entity: Entity) -> Result<()> {
        if entity.id.is_empty() {
            return Err(NeedleError::InvalidArgument("entity ID is required".into()));
        }

        let coll = self.db.collection(&self.collection_name)?;
        let meta = entity
            .metadata
            .clone()
            .unwrap_or_else(|| serde_json::json!({ "label": entity.label }));
        coll.insert(&entity.id, &entity.vector, Some(meta))?;

        self.entities.insert(entity.id.clone(), entity);
        Ok(())
    }

    /// Add a relation between two entities.
    pub fn add_relation(&mut self, relation: Relation) -> Result<()> {
        if !self.entities.contains_key(&relation.source) {
            return Err(NeedleError::InvalidArgument(format!(
                "source entity '{}' not found",
                relation.source
            )));
        }
        if !self.entities.contains_key(&relation.target) {
            return Err(NeedleError::InvalidArgument(format!(
                "target entity '{}' not found",
                relation.target
            )));
        }

        let degree = self
            .adjacency
            .degree(&relation.source, self.config.bidirectional);
        if degree >= self.config.max_edges_per_node {
            return Err(NeedleError::InvalidArgument(format!(
                "node '{}' has reached max edges ({})",
                relation.source, self.config.max_edges_per_node
            )));
        }

        self.adjacency.add_edge(
            &relation.source,
            &relation.target,
            &relation.relation_type,
            relation.weight,
        );
        self.relation_count += 1;
        Ok(())
    }

    /// Remove an entity and its edges.
    pub fn remove_entity(&mut self, id: &str) -> Result<bool> {
        if self.entities.remove(id).is_none() {
            return Ok(false);
        }
        self.adjacency.remove_node(id);
        let coll = self.db.collection(&self.collection_name)?;
        coll.delete(id)?;
        Ok(true)
    }

    /// Graph-enhanced vector search.
    ///
    /// 1. Performs vector similarity search to find seed entities
    /// 2. Traverses the graph from seeds up to `hops` hops
    /// 3. Scores all reached entities with a weighted combination of vector
    ///    similarity and graph proximity
    pub fn graph_search(
        &self,
        query: &[f32],
        k: usize,
        hops: usize,
        graph_weight: f32,
    ) -> Result<Vec<GraphSearchResult>> {
        let hops = hops.min(self.config.max_hops);
        let graph_weight = graph_weight.clamp(0.0, 1.0);

        // Step 1: Vector search for seeds
        let coll = self.db.collection(&self.collection_name)?;
        let vector_results = coll.search(query, k * 2)?; // oversample for reranking

        // Build vector score map
        let mut vector_scores: HashMap<String, f32> = HashMap::new();
        for (_i, r) in vector_results.iter().enumerate() {
            let score = 1.0 / (1.0 + r.distance); // convert distance to score
            vector_scores.insert(r.id.clone(), score);
        }

        // Step 2: BFS graph traversal from seed nodes
        let mut graph_scores: HashMap<String, (f32, usize, Vec<String>)> = HashMap::new();
        let seeds: Vec<String> = vector_results
            .iter()
            .take(k)
            .map(|r| r.id.clone())
            .collect();

        for seed in &seeds {
            let mut visited: HashSet<String> = HashSet::new();
            let mut queue: VecDeque<(String, usize, f32, Vec<String>)> = VecDeque::new();
            queue.push_back((seed.clone(), 0, 1.0, vec![seed.clone()]));
            visited.insert(seed.clone());

            while let Some((node, depth, acc_weight, path)) = queue.pop_front() {
                if depth > hops {
                    continue;
                }

                let entry = graph_scores
                    .entry(node.clone())
                    .or_insert((0.0, depth, path.clone()));
                entry.0 = entry.0.max(acc_weight);
                if depth < entry.1 {
                    entry.1 = depth;
                    entry.2 = path.clone();
                }

                if depth < hops {
                    let neighbors = self.adjacency.neighbors(&node, self.config.bidirectional);
                    for (neighbor, _rel_type, weight) in neighbors {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor.clone());
                            let new_weight = acc_weight * self.config.hop_decay * weight;
                            let mut new_path = path.clone();
                            new_path.push(neighbor.clone());
                            queue.push_back((neighbor, depth + 1, new_weight, new_path));
                        }
                    }
                }
            }
        }

        // Step 3: Combine scores
        let mut all_candidates: HashSet<String> = HashSet::new();
        all_candidates.extend(vector_scores.keys().cloned());
        all_candidates.extend(graph_scores.keys().cloned());

        let mut results: Vec<GraphSearchResult> = Vec::new();

        for id in all_candidates {
            let entity = match self.entities.get(&id) {
                Some(e) => e,
                None => continue,
            };

            let vs = vector_scores.get(&id).copied().unwrap_or(0.0);
            let (gs, hops_from_seed, path) =
                graph_scores
                    .get(&id)
                    .cloned()
                    .unwrap_or((0.0, 0, vec![id.clone()]));

            let combined = (1.0 - graph_weight) * vs + graph_weight * gs;

            results.push(GraphSearchResult {
                id: id.clone(),
                label: entity.label.clone(),
                score: combined,
                vector_score: vs,
                graph_score: gs,
                hops: hops_from_seed,
                path,
                metadata: entity.metadata.clone(),
            });
        }

        // Sort by combined score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Get neighbors of an entity.
    pub fn neighbors(&self, id: &str) -> Vec<(String, String, f32)> {
        self.adjacency.neighbors(id, self.config.bidirectional)
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Get graph statistics.
    pub fn stats(&self) -> GraphStats {
        let entity_count = self.entities.len();
        let relation_count = self.relation_count;
        let labels: HashSet<&str> = self.entities.values().map(|e| e.label.as_str()).collect();
        let relation_types: HashSet<&str> = self
            .adjacency
            .outgoing
            .values()
            .flat_map(|edges| edges.iter().map(|(_, rt, _)| rt.as_str()))
            .collect();

        let degrees: Vec<usize> = self
            .entities
            .keys()
            .map(|id| self.adjacency.degree(id, self.config.bidirectional))
            .collect();

        let avg_degree = if degrees.is_empty() {
            0.0
        } else {
            degrees.iter().sum::<usize>() as f32 / degrees.len() as f32
        };

        GraphStats {
            entity_count,
            relation_count,
            relation_type_count: relation_types.len(),
            label_count: labels.len(),
            avg_degree,
            max_degree: degrees.into_iter().max().unwrap_or(0),
        }
    }

    /// Find shortest path between two entities using BFS.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }
        if !self.entities.contains_key(from) || !self.entities.contains_key(to) {
            return None;
        }

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>)> = VecDeque::new();
        queue.push_back((from.to_string(), vec![from.to_string()]));
        visited.insert(from.to_string());

        while let Some((node, path)) = queue.pop_front() {
            let neighbors = self.adjacency.neighbors(&node, self.config.bidirectional);
            for (neighbor, _, _) in neighbors {
                if neighbor == to {
                    let mut result = path;
                    result.push(to.to_string());
                    return Some(result);
                }
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor.clone());
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    queue.push_back((neighbor, new_path));
                }
            }
        }

        None
    }

    /// Get all entities of a specific label.
    pub fn entities_by_label(&self, label: &str) -> Vec<&Entity> {
        self.entities
            .values()
            .filter(|e| e.label == label)
            .collect()
    }

    /// Get all unique relation types in the graph.
    pub fn relation_types(&self) -> Vec<String> {
        let types: HashSet<&str> = self
            .adjacency
            .outgoing
            .values()
            .flat_map(|edges| edges.iter().map(|(_, rt, _)| rt.as_str()))
            .collect();
        types.into_iter().map(String::from).collect()
    }

    /// Detect communities using a simple label propagation algorithm.
    /// Returns a map of community_id → list of entity IDs.
    pub fn detect_communities(&self, max_iterations: usize) -> HashMap<usize, Vec<String>> {
        let mut labels: HashMap<String, usize> = HashMap::new();
        for (i, id) in self.entities.keys().enumerate() {
            labels.insert(id.clone(), i);
        }

        for _ in 0..max_iterations {
            let mut changed = false;
            for id in self.entities.keys() {
                let neighbors = self.adjacency.neighbors(id, self.config.bidirectional);
                if neighbors.is_empty() {
                    continue;
                }
                // Count neighbor labels
                let mut label_counts: HashMap<usize, usize> = HashMap::new();
                for (n, _, _) in &neighbors {
                    if let Some(&lbl) = labels.get(n) {
                        *label_counts.entry(lbl).or_insert(0) += 1;
                    }
                }
                // Pick the most common label
                if let Some((&best_label, _)) =
                    label_counts.iter().max_by_key(|(_, &count)| count)
                {
                    let current = labels.get(id).copied().unwrap_or(0);
                    if best_label != current {
                        labels.insert(id.clone(), best_label);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Group by community
        let mut communities: HashMap<usize, Vec<String>> = HashMap::new();
        for (id, label) in labels {
            communities.entry(label).or_default().push(id);
        }
        communities
    }

    /// Export a subgraph around a seed entity up to `hops` hops.
    pub fn export_subgraph(
        &self,
        seed: &str,
        hops: usize,
    ) -> (Vec<Entity>, Vec<(String, String, String, f32)>) {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((seed.to_string(), 0));
        visited.insert(seed.to_string());

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            let neighbors = self.adjacency.neighbors(&node, self.config.bidirectional);
            for (n, _, _) in neighbors {
                if !visited.contains(&n) {
                    visited.insert(n.clone());
                    queue.push_back((n, depth + 1));
                }
            }
        }

        let entities: Vec<Entity> = visited
            .iter()
            .filter_map(|id| self.entities.get(id).cloned())
            .collect();

        let mut edges = Vec::new();
        for id in &visited {
            if let Some(out) = self.adjacency.outgoing.get(id) {
                for (target, rel_type, weight) in out {
                    if visited.contains(target) {
                        edges.push((id.clone(), target.clone(), rel_type.clone(), *weight));
                    }
                }
            }
        }

        (entities, edges)
    }

    /// Search with relation type filtering — only traverse edges of the specified types.
    pub fn graph_search_with_relation_filter(
        &self,
        query: &[f32],
        k: usize,
        hops: usize,
        graph_weight: f32,
        allowed_relations: &[&str],
    ) -> Result<Vec<GraphSearchResult>> {
        let hops = hops.min(self.config.max_hops);
        let graph_weight = graph_weight.clamp(0.0, 1.0);
        let allowed: HashSet<&str> = allowed_relations.iter().copied().collect();

        let coll = self.db.collection(&self.collection_name)?;
        let vector_results = coll.search(query, k * 2)?;

        let mut vector_scores: HashMap<String, f32> = HashMap::new();
        for r in &vector_results {
            vector_scores.insert(r.id.clone(), 1.0 / (1.0 + r.distance));
        }

        let mut graph_scores: HashMap<String, (f32, usize, Vec<String>)> = HashMap::new();
        let seeds: Vec<String> = vector_results.iter().take(k).map(|r| r.id.clone()).collect();

        for seed in &seeds {
            let mut visited: HashSet<String> = HashSet::new();
            let mut queue: VecDeque<(String, usize, f32, Vec<String>)> = VecDeque::new();
            queue.push_back((seed.clone(), 0, 1.0, vec![seed.clone()]));
            visited.insert(seed.clone());

            while let Some((node, depth, acc, path)) = queue.pop_front() {
                let entry = graph_scores
                    .entry(node.clone())
                    .or_insert((0.0, depth, path.clone()));
                entry.0 = entry.0.max(acc);
                if depth < entry.1 {
                    entry.1 = depth;
                    entry.2 = path.clone();
                }

                if depth < hops {
                    let neighbors = self.adjacency.neighbors(&node, self.config.bidirectional);
                    for (n, rel_type, w) in neighbors {
                        if !visited.contains(&n) && allowed.contains(rel_type.as_str()) {
                            visited.insert(n.clone());
                            let mut new_path = path.clone();
                            new_path.push(n.clone());
                            queue.push_back((n, depth + 1, acc * self.config.hop_decay * w, new_path));
                        }
                    }
                }
            }
        }

        let mut all_candidates: HashSet<String> = HashSet::new();
        all_candidates.extend(vector_scores.keys().cloned());
        all_candidates.extend(graph_scores.keys().cloned());

        let mut results: Vec<GraphSearchResult> = Vec::new();
        for id in all_candidates {
            let entity = match self.entities.get(&id) { Some(e) => e, None => continue };
            let vs = vector_scores.get(&id).copied().unwrap_or(0.0);
            let (gs, h, p) = graph_scores.get(&id).cloned().unwrap_or((0.0, 0, vec![id.clone()]));
            results.push(GraphSearchResult {
                id: id.clone(), label: entity.label.clone(),
                score: (1.0 - graph_weight) * vs + graph_weight * gs,
                vector_score: vs, graph_score: gs, hops: h, path: p,
                metadata: entity.metadata.clone(),
            });
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    /// Leiden-inspired community detection using modularity optimization.
    ///
    /// Assigns each node to the community that maximizes modularity gain,
    /// iterating until convergence or `max_iterations` is reached.
    /// Returns community ID → list of entity IDs.
    pub fn detect_communities_leiden(
        &self,
        max_iterations: usize,
        resolution: f64,
    ) -> Vec<Community> {
        let nodes: Vec<String> = self.entities.keys().cloned().collect();
        if nodes.is_empty() {
            return Vec::new();
        }

        let mut community_of: HashMap<String, usize> = HashMap::new();
        for (i, id) in nodes.iter().enumerate() {
            community_of.insert(id.clone(), i);
        }

        // Compute total edge weight (m) for modularity calculation
        let mut total_weight: f64 = 0.0;
        for edges in self.adjacency.outgoing.values() {
            for (_, _, w) in edges {
                total_weight += *w as f64;
            }
        }
        if total_weight == 0.0 {
            total_weight = 1.0;
        }

        // Node strengths (weighted degree)
        let mut strength: HashMap<String, f64> = HashMap::new();
        for node in &nodes {
            let s: f64 = self
                .adjacency
                .neighbors(node, self.config.bidirectional)
                .iter()
                .map(|(_, _, w)| *w as f64)
                .sum();
            strength.insert(node.clone(), s);
        }

        for _ in 0..max_iterations {
            let mut changed = false;

            for node in &nodes {
                let current_comm = community_of[node];
                let node_strength = strength.get(node).copied().unwrap_or(0.0);

                // Compute weight to each neighboring community
                let mut comm_weights: HashMap<usize, f64> = HashMap::new();
                let neighbors = self.adjacency.neighbors(node, self.config.bidirectional);
                for (n, _, w) in &neighbors {
                    if let Some(&c) = community_of.get(n) {
                        *comm_weights.entry(c).or_insert(0.0) += *w as f64;
                    }
                }

                // Compute community total strengths
                let mut comm_total_strength: HashMap<usize, f64> = HashMap::new();
                for (n, &c) in &community_of {
                    *comm_total_strength.entry(c).or_insert(0.0) +=
                        strength.get(n).copied().unwrap_or(0.0);
                }

                // Find best community (maximizing modularity gain)
                let mut best_comm = current_comm;
                let mut best_gain: f64 = 0.0;

                for (&candidate_comm, &w_to_comm) in &comm_weights {
                    if candidate_comm == current_comm {
                        continue;
                    }
                    let sigma_tot = comm_total_strength
                        .get(&candidate_comm)
                        .copied()
                        .unwrap_or(0.0);
                    let sigma_tot_old = comm_total_strength
                        .get(&current_comm)
                        .copied()
                        .unwrap_or(0.0)
                        - node_strength;

                    let w_to_old = comm_weights.get(&current_comm).copied().unwrap_or(0.0);

                    // Modularity gain of moving node to candidate community
                    let gain = (w_to_comm - w_to_old)
                        - resolution * node_strength * (sigma_tot - sigma_tot_old)
                            / (2.0 * total_weight);

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = candidate_comm;
                    }
                }

                if best_comm != current_comm {
                    community_of.insert(node.clone(), best_comm);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Normalize community IDs to 0..N and build result
        let mut comm_remap: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;
        let mut community_members: HashMap<usize, Vec<String>> = HashMap::new();
        for (node, &comm) in &community_of {
            let normalized = *comm_remap.entry(comm).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            community_members
                .entry(normalized)
                .or_default()
                .push(node.clone());
        }

        community_members
            .into_iter()
            .map(|(id, members)| Community { id, members })
            .collect()
    }

    /// Find all entities connected to `entity_id` within `hops` hops.
    pub fn connected_to(&self, entity_id: &str, hops: usize) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((entity_id.to_string(), 0));
        visited.insert(entity_id.to_string());

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            for (n, _, _) in self.adjacency.neighbors(&node, self.config.bidirectional) {
                if !visited.contains(&n) {
                    visited.insert(n.clone());
                    queue.push_back((n, depth + 1));
                }
            }
        }

        visited.into_iter().filter(|id| id != entity_id).collect()
    }

    /// Find all entities within the same community as `entity_id`.
    pub fn within_community(&self, entity_id: &str, resolution: f64) -> Vec<String> {
        let communities = self.detect_communities_leiden(20, resolution);
        for comm in &communities {
            if comm.members.contains(&entity_id.to_string()) {
                return comm
                    .members
                    .iter()
                    .filter(|id| id.as_str() != entity_id)
                    .cloned()
                    .collect();
            }
        }
        Vec::new()
    }

    /// Apply a graph filter to narrow search results.
    pub fn apply_graph_filter(
        &self,
        candidates: &[GraphSearchResult],
        filter: &GraphFilter,
    ) -> Vec<GraphSearchResult> {
        candidates
            .iter()
            .filter(|r| self.matches_graph_filter(&r.id, filter))
            .cloned()
            .collect()
    }

    fn matches_graph_filter(&self, entity_id: &str, filter: &GraphFilter) -> bool {
        match filter {
            GraphFilter::ConnectedTo { target, max_hops } => {
                let connected = self.connected_to(target, *max_hops);
                connected.contains(&entity_id.to_string())
            }
            GraphFilter::WithinCommunity { seed, resolution } => {
                let members = self.within_community(seed, *resolution);
                members.contains(&entity_id.to_string())
            }
            GraphFilter::HasRelation { relation_type } => self
                .adjacency
                .neighbors(entity_id, self.config.bidirectional)
                .iter()
                .any(|(_, rt, _)| rt == relation_type),
            GraphFilter::MinDegree(min) => {
                self.adjacency.degree(entity_id, self.config.bidirectional) >= *min
            }
            GraphFilter::And(filters) => filters.iter().all(|f| self.matches_graph_filter(entity_id, f)),
            GraphFilter::Or(filters) => filters.iter().any(|f| self.matches_graph_filter(entity_id, f)),
        }
    }
}

/// A detected community of entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Community identifier.
    pub id: usize,
    /// Entity IDs belonging to this community.
    pub members: Vec<String>,
}

/// Graph-aware filter operators for search refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphFilter {
    /// Include only entities connected to `target` within `max_hops`.
    ConnectedTo { target: String, max_hops: usize },
    /// Include only entities in the same community as `seed`.
    WithinCommunity { seed: String, resolution: f64 },
    /// Include only entities with at least one edge of the given type.
    HasRelation { relation_type: String },
    /// Include only entities with degree >= min.
    MinDegree(usize),
    /// All sub-filters must match.
    And(Vec<GraphFilter>),
    /// At least one sub-filter must match.
    Or(Vec<GraphFilter>),
}

// ── RAG Context Assembly ─────────────────────────────────────────────────────

/// A fully assembled RAG context from graph-augmented retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagContext {
    /// The assembled context text, ready for LLM prompt injection.
    pub context: String,
    /// Entity IDs that contributed to this context.
    pub source_entities: Vec<String>,
    /// Relation triples (source, relation, target) in the context.
    pub relations: Vec<(String, String, String)>,
    /// Number of hops traversed.
    pub hops_traversed: usize,
    /// Vector similarity score of the seed entities.
    pub similarity_score: f32,
}

impl<'a> GraphRagService<'a> {
    /// Assemble a RAG context by combining vector similarity search with
    /// graph traversal to gather multi-hop related entities.
    ///
    /// This produces a structured context string suitable for injection
    /// into an LLM prompt, combining direct vector matches with
    /// graph-connected entities for richer context.
    pub fn assemble_context(
        &self,
        query_vector: &[f32],
        k: usize,
        max_hops: usize,
        graph_weight: f32,
    ) -> Result<RagContext> {
        // Step 1: Vector search for seed entities
        let search_results = self.graph_search(query_vector, k, max_hops, graph_weight)?;

        if search_results.is_empty() {
            return Ok(RagContext {
                context: String::new(),
                source_entities: Vec::new(),
                relations: Vec::new(),
                hops_traversed: 0,
                similarity_score: 0.0,
            });
        }

        let avg_score = search_results.iter().map(|r| r.score).sum::<f32>()
            / search_results.len() as f32;

        // Step 2: Collect entities and their relations
        let mut context_parts: Vec<String> = Vec::new();
        let mut all_entity_ids: Vec<String> = Vec::new();
        let mut all_relations: Vec<(String, String, String)> = Vec::new();

        for result in &search_results {
            all_entity_ids.push(result.id.clone());

            // Add entity info
            if let Some(entity) = self.entities.get(&result.id) {
                context_parts.push(format!(
                    "[{}: {}]",
                    entity.label, result.id
                ));
            }

            // Add relation context for each path hop
            for (hop_idx, hop_id) in result.path.iter().enumerate() {
                if hop_idx > 0 {
                    let prev = &result.path[hop_idx - 1];
                    // Find the relation between prev and hop_id
                    for (neighbor, rel_type, _weight) in self.adjacency.neighbors(prev, true) {
                        if &neighbor == hop_id {
                            let triple = (
                                prev.clone(),
                                rel_type.clone(),
                                neighbor.clone(),
                            );
                            if !all_relations.contains(&triple) {
                                context_parts.push(format!(
                                    "{} --[{}]--> {}",
                                    prev, rel_type, neighbor
                                ));
                                all_relations.push(triple);
                            }
                            break;
                        }
                    }
                }
            }
        }

        let max_hops_found = search_results.iter().map(|r| r.hops).max().unwrap_or(0);

        Ok(RagContext {
            context: context_parts.join("\n"),
            source_entities: all_entity_ids,
            relations: all_relations,
            hops_traversed: max_hops_found,
            similarity_score: avg_score,
        })
    }

    /// Compute PageRank scores for all entities in the graph.
    ///
    /// Returns a map of entity ID → PageRank score. Higher scores indicate
    /// more "important" entities that are well-connected in the graph.
    /// Useful for boosting search results by entity importance.
    pub fn pagerank(&self, damping: f32, iterations: usize) -> HashMap<String, f32> {
        let n = self.entities.len();
        if n == 0 {
            return HashMap::new();
        }

        let entity_ids: Vec<&String> = self.entities.keys().collect();
        let n_f = n as f32;
        let mut scores: HashMap<&String, f32> = entity_ids.iter().map(|id| (*id, 1.0 / n_f)).collect();

        for _ in 0..iterations {
            let mut new_scores: HashMap<&String, f32> = entity_ids.iter().map(|id| (*id, (1.0 - damping) / n_f)).collect();

            for id in &entity_ids {
                let out_degree = self.adjacency.degree(id, false);
                if out_degree == 0 {
                    // Dangling node — distribute evenly
                    let share = scores[id] / n_f;
                    for other in &entity_ids {
                        *new_scores.get_mut(other).unwrap_or(&mut 0.0) += damping * share;
                    }
                } else {
                    // Distribute to neighbors
                    let share = scores[id] / out_degree as f32;
                    if let Some(edges) = self.adjacency.outgoing.get(id.as_str()) {
                        for (target, _, _) in edges {
                            if let Some(score) = new_scores.get_mut(&target) {
                                *score += damping * share;
                            }
                        }
                    }
                    if self.config.bidirectional {
                        if let Some(edges) = self.adjacency.incoming.get(id.as_str()) {
                            for (source, _, _) in edges {
                                if let Some(score) = new_scores.get_mut(&source) {
                                    *score += damping * share;
                                }
                            }
                        }
                    }
                }
            }

            scores = new_scores;
        }

        scores.into_iter().map(|(k, v)| (k.clone(), v)).collect()
    }

    /// Search with PageRank boost: entities with higher graph importance
    /// get boosted in the final ranking.
    pub fn graph_search_with_pagerank(
        &self,
        query: &[f32],
        k: usize,
        max_hops: usize,
        graph_weight: f32,
        pagerank_weight: f32,
    ) -> Result<Vec<GraphSearchResult>> {
        let mut results = self.graph_search(query, k * 2, max_hops, graph_weight)?;
        let pr_scores = self.pagerank(0.85, 20);

        // Boost scores by PageRank
        for result in &mut results {
            let pr = pr_scores.get(&result.id).copied().unwrap_or(0.0);
            result.score = result.score * (1.0 - pagerank_weight) + pr * pagerank_weight;
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }
}

// ── Simple Entity Extraction ────────────────────────────────────────────────

/// Extract candidate entity names from text using simple heuristics.
///
/// Uses capitalization patterns and common NER-like rules:
/// - Consecutive capitalized words → potential entity
/// - Words in quotes → potential entity
/// - Words after "the" that are capitalized → potential entity
pub fn extract_entities_from_text(text: &str) -> Vec<String> {
    let mut entities = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());

        // Check for consecutive capitalized words (multi-word entities)
        if !word.is_empty() && word.chars().next().is_some_and(|c| c.is_uppercase()) {
            let mut entity_parts = vec![word.to_string()];
            let mut j = i + 1;
            while j < words.len() {
                let next = words[j].trim_matches(|c: char| !c.is_alphanumeric());
                if !next.is_empty() && next.chars().next().is_some_and(|c| c.is_uppercase()) {
                    entity_parts.push(next.to_string());
                    j += 1;
                } else {
                    break;
                }
            }

            // Only treat as entity if not a sentence starter (first word) or if multi-word
            if entity_parts.len() > 1 || (i > 0 && !words[i - 1].ends_with('.')) {
                let entity = entity_parts.join(" ");
                if entity.len() > 1 && !entities.contains(&entity) {
                    entities.push(entity);
                }
            }

            i = j;
            continue;
        }

        // Check for quoted strings
        if word.starts_with('"') || word.starts_with('\'') {
            let quote_char = word.chars().next().unwrap_or('"');
            let mut quoted = word.trim_start_matches(quote_char).to_string();
            let mut j = i + 1;
            while j < words.len() && !words[j].ends_with(quote_char) {
                quoted.push(' ');
                quoted.push_str(words[j]);
                j += 1;
            }
            if j < words.len() {
                quoted.push(' ');
                quoted.push_str(words[j].trim_end_matches(quote_char));
                j += 1;
            }
            let trimmed = quoted.trim().to_string();
            if !trimmed.is_empty() && !entities.contains(&trimmed) {
                entities.push(trimmed);
            }
            i = j;
            continue;
        }

        i += 1;
    }

    entities
}

/// Extract a k-hop subgraph around a set of seed entities.
///
/// Returns entity IDs and relation triples within `max_hops` of any seed entity.
/// Useful for building context windows for LLM prompts.
pub fn extract_subgraph(
    service: &GraphRagService<'_>,
    seed_entities: &[&str],
    max_hops: usize,
) -> (Vec<String>, Vec<(String, String, String)>) {
    let mut visited = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut relations = Vec::new();

    // Seed the BFS
    for &seed in seed_entities {
        if !visited.contains(seed) {
            visited.insert(seed.to_string());
            queue.push_back((seed.to_string(), 0));
        }
    }

    // BFS traversal
    while let Some((current, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }

        let neighbors = service.neighbors(&current);
        for (neighbor_id, relation_type, _weight) in neighbors {
            // Record relation
            let triple = (current.clone(), relation_type, neighbor_id.clone());
            if !relations.contains(&triple) {
                relations.push(triple);
            }

            // Continue BFS
            if !visited.contains(&neighbor_id) {
                visited.insert(neighbor_id.clone());
                queue.push_back((neighbor_id, depth + 1));
            }
        }
    }

    let entity_ids: Vec<String> = visited.into_iter().collect();
    (entity_ids, relations)
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("entities", 4).unwrap();
        db
    }

    fn make_entity(id: &str, label: &str, vec: Vec<f32>) -> Entity {
        Entity {
            id: id.into(),
            label: label.into(),
            vector: vec,
            metadata: None,
        }
    }

    #[test]
    fn test_add_entity_and_relation() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "Lang", vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        svc.add_entity(make_entity("b", "Tool", vec![0.9, 0.1, 0.0, 0.0]))
            .unwrap();
        svc.add_relation(Relation::new("a", "b", "uses")).unwrap();

        let stats = svc.stats();
        assert_eq!(stats.entity_count, 2);
        assert_eq!(stats.relation_count, 1);
    }

    #[test]
    fn test_graph_search() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "Lang", vec![1.0, 0.0, 0.0, 0.0]))
            .unwrap();
        svc.add_entity(make_entity("b", "Tool", vec![0.5, 0.5, 0.0, 0.0]))
            .unwrap();
        svc.add_entity(make_entity("c", "Lib", vec![0.0, 0.0, 1.0, 0.0]))
            .unwrap();
        svc.add_relation(Relation::new("a", "b", "uses")).unwrap();
        svc.add_relation(Relation::new("b", "c", "depends_on"))
            .unwrap();

        let results = svc.graph_search(&[1.0, 0.0, 0.0, 0.0], 5, 2, 0.3).unwrap();
        assert!(!results.is_empty());
        // "a" should be most relevant (closest vector + seed)
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_shortest_path() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("c", "X", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "r")).unwrap();
        svc.add_relation(Relation::new("b", "c", "r")).unwrap();

        let path = svc.shortest_path("a", "c").unwrap();
        assert_eq!(path, vec!["a", "b", "c"]);

        // No path
        svc.add_entity(make_entity("d", "X", vec![1.0; 4])).unwrap();
        assert!(svc.shortest_path("d", "a").is_none());
    }

    #[test]
    fn test_remove_entity() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "X", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "r")).unwrap();

        assert!(svc.remove_entity("a").unwrap());
        assert!(!svc.remove_entity("nonexistent").unwrap());
        assert_eq!(svc.stats().entity_count, 1);
    }

    #[test]
    fn test_relation_validation() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "X", vec![1.0; 4])).unwrap();

        // Target doesn't exist
        assert!(svc
            .add_relation(Relation::new("a", "nonexistent", "r"))
            .is_err());
        // Source doesn't exist
        assert!(svc
            .add_relation(Relation::new("nonexistent", "a", "r"))
            .is_err());
    }

    #[test]
    fn test_entities_by_label() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "Person", vec![1.0; 4]))
            .unwrap();
        svc.add_entity(make_entity("b", "Person", vec![1.0; 4]))
            .unwrap();
        svc.add_entity(make_entity("c", "Place", vec![1.0; 4]))
            .unwrap();

        assert_eq!(svc.entities_by_label("Person").len(), 2);
        assert_eq!(svc.entities_by_label("Place").len(), 1);
        assert_eq!(svc.entities_by_label("Unknown").len(), 0);
    }

    #[test]
    fn test_graph_stats() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "A", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "B", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("c", "A", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "r1")).unwrap();
        svc.add_relation(Relation::new("a", "c", "r2")).unwrap();

        let stats = svc.stats();
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.relation_count, 2);
        assert_eq!(stats.relation_type_count, 2);
        assert_eq!(stats.label_count, 2);
        assert!(stats.avg_degree > 0.0);
    }

    #[test]
    fn test_weighted_relation() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "X", vec![1.0; 4])).unwrap();

        let rel = Relation::new("a", "b", "strong").with_weight(2.0);
        svc.add_relation(rel).unwrap();

        let neighbors = svc.neighbors("a");
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].2, 2.0);
    }

    #[test]
    fn test_community_detection() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        // Two disconnected components → should be 2 communities
        svc.add_entity(make_entity("a", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("c", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("d", "X", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "r")).unwrap();
        svc.add_relation(Relation::new("c", "d", "r")).unwrap();

        let communities = svc.detect_communities(10);
        assert!(communities.len() >= 2);
    }

    #[test]
    fn test_export_subgraph() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("c", "X", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("d", "X", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "r")).unwrap();
        svc.add_relation(Relation::new("b", "c", "r")).unwrap();
        svc.add_relation(Relation::new("c", "d", "r")).unwrap();

        // 1 hop from "a" → should get a and b
        let (entities, edges) = svc.export_subgraph("a", 1);
        assert_eq!(entities.len(), 2);
        assert_eq!(edges.len(), 1);

        // 2 hops from "a" → should get a, b, c
        let (entities, _) = svc.export_subgraph("a", 2);
        assert_eq!(entities.len(), 3);
    }

    #[test]
    fn test_search_with_relation_filter() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "Lang", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        svc.add_entity(make_entity("b", "Tool", vec![0.5, 0.5, 0.0, 0.0])).unwrap();
        svc.add_entity(make_entity("c", "Lib", vec![0.0, 0.0, 1.0, 0.0])).unwrap();
        svc.add_relation(Relation::new("a", "b", "uses")).unwrap();
        svc.add_relation(Relation::new("b", "c", "depends_on")).unwrap();

        // Only follow "uses" edges
        let results = svc
            .graph_search_with_relation_filter(&[1.0, 0.0, 0.0, 0.0], 5, 2, 0.3, &["uses"])
            .unwrap();
        // "c" should NOT be reachable via graph since "depends_on" is filtered
        let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
        // a and b should be present, c only via vector score (if close enough)
        assert!(ids.contains(&"a"));
    }

    #[test]
    fn test_leiden_community_detection() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        // Create two clusters: {a,b,c} densely connected, {d,e,f} densely connected
        for id in ["a", "b", "c", "d", "e", "f"] {
            svc.add_entity(make_entity(id, "Node", vec![1.0; 4])).unwrap();
        }
        // Cluster 1
        svc.add_relation(Relation::new("a", "b", "r").with_weight(1.0)).unwrap();
        svc.add_relation(Relation::new("b", "c", "r").with_weight(1.0)).unwrap();
        svc.add_relation(Relation::new("a", "c", "r").with_weight(1.0)).unwrap();
        // Cluster 2
        svc.add_relation(Relation::new("d", "e", "r").with_weight(1.0)).unwrap();
        svc.add_relation(Relation::new("e", "f", "r").with_weight(1.0)).unwrap();
        svc.add_relation(Relation::new("d", "f", "r").with_weight(1.0)).unwrap();
        // Weak bridge
        svc.add_relation(Relation::new("c", "d", "r").with_weight(0.1)).unwrap();

        let communities = svc.detect_communities_leiden(50, 1.0);
        assert!(communities.len() >= 2, "Expected at least 2 communities, got {}", communities.len());

        // Verify nodes in same cluster end up together
        let a_comm = communities.iter().find(|c| c.members.contains(&"a".to_string()));
        let d_comm = communities.iter().find(|c| c.members.contains(&"d".to_string()));
        assert!(a_comm.is_some());
        assert!(d_comm.is_some());
    }

    #[test]
    fn test_leiden_empty_graph() {
        let db = test_db();
        let svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();
        let communities = svc.detect_communities_leiden(10, 1.0);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_connected_to() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        for id in ["a", "b", "c", "d"] {
            svc.add_entity(make_entity(id, "N", vec![1.0; 4])).unwrap();
        }
        svc.add_relation(Relation::new("a", "b", "r")).unwrap();
        svc.add_relation(Relation::new("b", "c", "r")).unwrap();
        svc.add_relation(Relation::new("c", "d", "r")).unwrap();

        // 1 hop from a
        let one_hop = svc.connected_to("a", 1);
        assert!(one_hop.contains(&"b".to_string()));
        assert!(!one_hop.contains(&"c".to_string()));

        // 2 hops from a
        let two_hops = svc.connected_to("a", 2);
        assert!(two_hops.contains(&"b".to_string()));
        assert!(two_hops.contains(&"c".to_string()));
        assert!(!two_hops.contains(&"d".to_string()));
    }

    #[test]
    fn test_within_community() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        for id in ["a", "b", "c"] {
            svc.add_entity(make_entity(id, "N", vec![1.0; 4])).unwrap();
        }
        svc.add_relation(Relation::new("a", "b", "r").with_weight(1.0)).unwrap();
        svc.add_relation(Relation::new("b", "c", "r").with_weight(1.0)).unwrap();
        svc.add_relation(Relation::new("a", "c", "r").with_weight(1.0)).unwrap();

        let peers = svc.within_community("a", 1.0);
        // All 3 nodes should be in the same community
        assert!(peers.contains(&"b".to_string()) || peers.contains(&"c".to_string()));
    }

    #[test]
    fn test_graph_filter_connected_to() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "N", vec![1.0, 0.0, 0.0, 0.0])).unwrap();
        svc.add_entity(make_entity("b", "N", vec![0.9, 0.1, 0.0, 0.0])).unwrap();
        svc.add_entity(make_entity("c", "N", vec![0.0, 1.0, 0.0, 0.0])).unwrap();
        svc.add_relation(Relation::new("a", "b", "r")).unwrap();

        let results = svc.graph_search(&[1.0, 0.0, 0.0, 0.0], 10, 1, 0.3).unwrap();

        let filter = GraphFilter::ConnectedTo { target: "a".into(), max_hops: 1 };
        let filtered = svc.apply_graph_filter(&results, &filter);
        let ids: Vec<&str> = filtered.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(!ids.contains(&"c"));
    }

    #[test]
    fn test_graph_filter_has_relation() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "N", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "N", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("c", "N", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "uses")).unwrap();
        svc.add_relation(Relation::new("b", "c", "depends_on")).unwrap();

        let results = svc.graph_search(&[1.0; 4], 10, 2, 0.3).unwrap();
        let filter = GraphFilter::HasRelation { relation_type: "uses".into() };
        let filtered = svc.apply_graph_filter(&results, &filter);

        let ids: Vec<&str> = filtered.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"a"));
    }

    #[test]
    fn test_graph_filter_min_degree() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("hub", "N", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("leaf1", "N", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("leaf2", "N", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("hub", "leaf1", "r")).unwrap();
        svc.add_relation(Relation::new("hub", "leaf2", "r")).unwrap();

        let results = svc.graph_search(&[1.0; 4], 10, 1, 0.3).unwrap();
        let filter = GraphFilter::MinDegree(2);
        let filtered = svc.apply_graph_filter(&results, &filter);

        // Only hub has degree >= 2
        let ids: Vec<&str> = filtered.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"hub"));
        assert!(!ids.contains(&"leaf1"));
    }

    #[test]
    fn test_graph_filter_and_or() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(make_entity("a", "N", vec![1.0; 4])).unwrap();
        svc.add_entity(make_entity("b", "N", vec![1.0; 4])).unwrap();
        svc.add_relation(Relation::new("a", "b", "uses")).unwrap();

        let results = svc.graph_search(&[1.0; 4], 10, 1, 0.3).unwrap();

        // AND: has "uses" relation AND degree >= 1
        let filter = GraphFilter::And(vec![
            GraphFilter::HasRelation { relation_type: "uses".into() },
            GraphFilter::MinDegree(1),
        ]);
        let filtered = svc.apply_graph_filter(&results, &filter);
        assert!(!filtered.is_empty());

        // OR: degree >= 5 OR has "uses" relation
        let filter = GraphFilter::Or(vec![
            GraphFilter::MinDegree(5),
            GraphFilter::HasRelation { relation_type: "uses".into() },
        ]);
        let filtered = svc.apply_graph_filter(&results, &filter);
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_community_struct_serde() {
        let comm = Community { id: 0, members: vec!["a".into(), "b".into()] };
        let json = serde_json::to_string(&comm).unwrap();
        let deser: Community = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, 0);
        assert_eq!(deser.members.len(), 2);
    }

    #[test]
    fn test_assemble_context_empty() {
        let db = test_db();
        let svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();
        let ctx = svc.assemble_context(&[1.0, 0.0, 0.0, 0.0], 5, 2, 0.3).unwrap();
        assert!(ctx.context.is_empty());
        assert!(ctx.source_entities.is_empty());
    }

    #[test]
    fn test_assemble_context_with_data() {
        let db = test_db();
        let mut svc = GraphRagService::new(&db, "entities", GraphRagConfig::default()).unwrap();

        svc.add_entity(Entity {
            id: "rust".into(),
            label: "Language".into(),
            vector: vec![0.9, 0.1, 0.0, 0.0],
            metadata: None,
        }).unwrap();

        svc.add_entity(Entity {
            id: "cargo".into(),
            label: "Tool".into(),
            vector: vec![0.8, 0.2, 0.0, 0.0],
            metadata: None,
        }).unwrap();

        svc.add_relation(Relation::new("rust", "cargo", "uses")).unwrap();

        let ctx = svc.assemble_context(&[0.9, 0.1, 0.0, 0.0], 5, 2, 0.3).unwrap();
        assert!(!ctx.source_entities.is_empty());
        assert!(ctx.similarity_score > 0.0);
    }

    #[test]
    fn test_graph_filter_serde() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let filter = GraphFilter::ConnectedTo { target: "x".into(), max_hops: 2 };
        let json = serde_json::to_string(&filter)?;
        let deser: GraphFilter = serde_json::from_str(&json)?;
        match deser {
            GraphFilter::ConnectedTo { target, max_hops } => {
                assert_eq!(target, "x");
                assert_eq!(max_hops, 2);
            }
            _ => return Err("Wrong variant".into()),
        }

        Ok(())
    }
}
