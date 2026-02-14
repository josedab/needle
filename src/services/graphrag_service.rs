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
    pub fn with_weight(mut self, w: f32) -> Self {
        self.weight = w;
        self
    }

    /// Set edge metadata.
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
}
