//! Knowledge Graph Fusion - Combine vector search with graph relationships.
//!
//! Integrates vector similarity search with knowledge graph traversal
//! for enhanced semantic understanding and relationship-aware retrieval.
//!
//! # Features
//!
//! - **Entity linking**: Connect vectors to graph entities
//! - **Relationship traversal**: Follow edges during search
//! - **Hybrid ranking**: Combine vector similarity with graph distance
//! - **Path finding**: Find semantic paths between concepts
//! - **Subgraph extraction**: Get relevant graph context
//!
//! # Example
//!
//! ```ignore
//! use needle::knowledge_graph::{KnowledgeGraph, Entity, Relation};
//!
//! let mut kg = KnowledgeGraph::new();
//!
//! // Add entities with embeddings
//! kg.add_entity("rust", &embedding, HashMap::new())?;
//! kg.add_entity("programming", &embedding2, HashMap::new())?;
//!
//! // Add relationship
//! kg.add_relation("rust", "programming", "is_a")?;
//!
//! // Search with graph context
//! let results = kg.search_with_context(&query, 10)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// An entity in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity ID.
    pub id: String,
    /// Entity type.
    pub entity_type: String,
    /// Entity label.
    pub label: String,
    /// Vector embedding.
    pub embedding: Vec<f32>,
    /// Properties.
    pub properties: HashMap<String, String>,
    /// Creation timestamp.
    pub created_at: u64,
}

/// A relation between entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity ID.
    pub source: String,
    /// Target entity ID.
    pub target: String,
    /// Relation type.
    pub relation_type: String,
    /// Relation weight.
    pub weight: f32,
    /// Properties.
    pub properties: HashMap<String, String>,
}

/// Search result with graph context.
#[derive(Debug, Clone)]
pub struct GraphSearchResult {
    /// Entity.
    pub entity: Entity,
    /// Vector similarity score.
    pub similarity: f32,
    /// Graph relevance score.
    pub graph_score: f32,
    /// Combined score.
    pub combined_score: f32,
    /// Related entities (1-hop).
    pub related: Vec<RelatedEntity>,
    /// Path from query context (if any).
    pub path: Option<Vec<String>>,
}

/// A related entity with its relationship.
#[derive(Debug, Clone)]
pub struct RelatedEntity {
    /// Entity.
    pub entity: Entity,
    /// Relation type.
    pub relation: String,
    /// Direction.
    pub direction: RelationDirection,
}

/// Relation direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationDirection {
    /// Outgoing relation.
    Outgoing,
    /// Incoming relation.
    Incoming,
}

/// Path between entities.
#[derive(Debug, Clone)]
pub struct EntityPath {
    /// Path of entity IDs.
    pub entities: Vec<String>,
    /// Relations along the path.
    pub relations: Vec<String>,
    /// Total path weight.
    pub weight: f32,
    /// Path length.
    pub length: usize,
}

/// Configuration for knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphConfig {
    /// Weight for vector similarity in combined score.
    pub similarity_weight: f32,
    /// Weight for graph relevance in combined score.
    pub graph_weight: f32,
    /// Maximum hops for context.
    pub max_context_hops: usize,
    /// Maximum path length for path finding.
    pub max_path_length: usize,
    /// Enable bidirectional search.
    pub bidirectional_search: bool,
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.7,
            graph_weight: 0.3,
            max_context_hops: 2,
            max_path_length: 5,
            bidirectional_search: true,
        }
    }
}

/// Knowledge graph with vector embeddings.
pub struct KnowledgeGraph {
    /// Configuration.
    config: KnowledgeGraphConfig,
    /// Entities by ID.
    entities: HashMap<String, Entity>,
    /// Outgoing relations: source -> [(target, relation)]
    outgoing: HashMap<String, Vec<(String, Relation)>>,
    /// Incoming relations: target -> [(source, relation)]
    incoming: HashMap<String, Vec<(String, Relation)>>,
    /// Entity type index.
    type_index: HashMap<String, HashSet<String>>,
    /// Relation type index.
    relation_index: HashMap<String, Vec<Relation>>,
}

impl KnowledgeGraph {
    /// Create a new knowledge graph.
    pub fn new() -> Self {
        Self::with_config(KnowledgeGraphConfig::default())
    }

    /// Create with configuration.
    pub fn with_config(config: KnowledgeGraphConfig) -> Self {
        Self {
            config,
            entities: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            type_index: HashMap::new(),
            relation_index: HashMap::new(),
        }
    }

    /// Add an entity.
    pub fn add_entity(
        &mut self,
        id: &str,
        entity_type: &str,
        label: &str,
        embedding: &[f32],
        properties: HashMap<String, String>,
    ) -> Result<()> {
        if self.entities.contains_key(id) {
            return Err(NeedleError::InvalidInput(format!(
                "Entity '{}' already exists",
                id
            )));
        }

        let entity = Entity {
            id: id.to_string(),
            entity_type: entity_type.to_string(),
            label: label.to_string(),
            embedding: embedding.to_vec(),
            properties,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.type_index
            .entry(entity_type.to_string())
            .or_default()
            .insert(id.to_string());

        self.entities.insert(id.to_string(), entity);
        Ok(())
    }

    /// Get an entity.
    pub fn get_entity(&self, id: &str) -> Option<&Entity> {
        self.entities.get(id)
    }

    /// Update entity embedding.
    pub fn update_embedding(&mut self, id: &str, embedding: &[f32]) -> Result<()> {
        let entity = self.entities.get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Entity '{}' not found", id)))?;
        entity.embedding = embedding.to_vec();
        Ok(())
    }

    /// Add a relation.
    pub fn add_relation(
        &mut self,
        source: &str,
        target: &str,
        relation_type: &str,
    ) -> Result<()> {
        self.add_weighted_relation(source, target, relation_type, 1.0, HashMap::new())
    }

    /// Add a weighted relation.
    pub fn add_weighted_relation(
        &mut self,
        source: &str,
        target: &str,
        relation_type: &str,
        weight: f32,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        if !self.entities.contains_key(source) {
            return Err(NeedleError::NotFound(format!("Source entity '{}' not found", source)));
        }
        if !self.entities.contains_key(target) {
            return Err(NeedleError::NotFound(format!("Target entity '{}' not found", target)));
        }

        let relation = Relation {
            source: source.to_string(),
            target: target.to_string(),
            relation_type: relation_type.to_string(),
            weight,
            properties,
        };

        self.outgoing
            .entry(source.to_string())
            .or_default()
            .push((target.to_string(), relation.clone()));

        self.incoming
            .entry(target.to_string())
            .or_default()
            .push((source.to_string(), relation.clone()));

        self.relation_index
            .entry(relation_type.to_string())
            .or_default()
            .push(relation);

        Ok(())
    }

    /// Get outgoing relations.
    pub fn get_outgoing(&self, entity_id: &str) -> Vec<&Relation> {
        self.outgoing
            .get(entity_id)
            .map(|rels| rels.iter().map(|(_, r)| r).collect())
            .unwrap_or_default()
    }

    /// Get incoming relations.
    pub fn get_incoming(&self, entity_id: &str) -> Vec<&Relation> {
        self.incoming
            .get(entity_id)
            .map(|rels| rels.iter().map(|(_, r)| r).collect())
            .unwrap_or_default()
    }

    /// Get all neighbors (1-hop).
    pub fn get_neighbors(&self, entity_id: &str) -> Vec<RelatedEntity> {
        let mut neighbors = Vec::new();

        // Outgoing
        if let Some(rels) = self.outgoing.get(entity_id) {
            for (target_id, rel) in rels {
                if let Some(entity) = self.entities.get(target_id) {
                    neighbors.push(RelatedEntity {
                        entity: entity.clone(),
                        relation: rel.relation_type.clone(),
                        direction: RelationDirection::Outgoing,
                    });
                }
            }
        }

        // Incoming
        if let Some(rels) = self.incoming.get(entity_id) {
            for (source_id, rel) in rels {
                if let Some(entity) = self.entities.get(source_id) {
                    neighbors.push(RelatedEntity {
                        entity: entity.clone(),
                        relation: rel.relation_type.clone(),
                        direction: RelationDirection::Incoming,
                    });
                }
            }
        }

        neighbors
    }

    /// Vector similarity search.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<GraphSearchResult> {
        self.search_with_filter(query, k, None)
    }

    /// Search with entity type filter.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        entity_type: Option<&str>,
    ) -> Vec<GraphSearchResult> {
        let candidates: Vec<&Entity> = if let Some(etype) = entity_type {
            self.type_index
                .get(etype)
                .map(|ids| ids.iter().filter_map(|id| self.entities.get(id)).collect())
                .unwrap_or_default()
        } else {
            self.entities.values().collect()
        };

        let mut results: Vec<GraphSearchResult> = candidates
            .iter()
            .map(|entity| {
                let similarity = self.cosine_similarity(query, &entity.embedding);
                let graph_score = self.compute_graph_score(&entity.id);
                let combined_score = self.config.similarity_weight * similarity
                    + self.config.graph_weight * graph_score;

                let related = self.get_neighbors(&entity.id);

                GraphSearchResult {
                    entity: (*entity).clone(),
                    similarity,
                    graph_score,
                    combined_score,
                    related,
                    path: None,
                }
            })
            .collect();

        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        results.truncate(k);
        results
    }

    /// Search with graph context from anchor entities.
    pub fn search_with_context(
        &self,
        query: &[f32],
        k: usize,
        anchors: &[String],
    ) -> Vec<GraphSearchResult> {
        // Get reachable entities from anchors
        let reachable = self.get_reachable(anchors, self.config.max_context_hops);

        // Boost scores for entities near anchors
        let mut results: Vec<GraphSearchResult> = self.entities.values()
            .map(|entity| {
                let similarity = self.cosine_similarity(query, &entity.embedding);
                let graph_score = if reachable.contains(&entity.id) {
                    1.0
                } else {
                    self.compute_graph_score(&entity.id)
                };

                let combined_score = self.config.similarity_weight * similarity
                    + self.config.graph_weight * graph_score;

                let related = self.get_neighbors(&entity.id);

                // Find path to nearest anchor
                let path = anchors.iter()
                    .filter_map(|anchor| {
                        self.find_path(&entity.id, anchor)
                            .map(|p| p.entities)
                    })
                    .min_by_key(|p| p.len());

                GraphSearchResult {
                    entity: entity.clone(),
                    similarity,
                    graph_score,
                    combined_score,
                    related,
                    path,
                }
            })
            .collect();

        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        results.truncate(k);
        results
    }

    /// Get entities reachable within n hops.
    fn get_reachable(&self, start: &[String], max_hops: usize) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = start.iter()
            .map(|s| (s.clone(), 0))
            .collect();

        while let Some((entity_id, depth)) = queue.pop_front() {
            if visited.contains(&entity_id) || depth > max_hops {
                continue;
            }
            visited.insert(entity_id.clone());

            // Add neighbors
            if let Some(rels) = self.outgoing.get(&entity_id) {
                for (target, _) in rels {
                    queue.push_back((target.clone(), depth + 1));
                }
            }
            if let Some(rels) = self.incoming.get(&entity_id) {
                for (source, _) in rels {
                    queue.push_back((source.clone(), depth + 1));
                }
            }
        }

        visited
    }

    /// Find shortest path between two entities.
    pub fn find_path(&self, from: &str, to: &str) -> Option<EntityPath> {
        if !self.entities.contains_key(from) || !self.entities.contains_key(to) {
            return None;
        }

        if from == to {
            return Some(EntityPath {
                entities: vec![from.to_string()],
                relations: Vec::new(),
                weight: 0.0,
                length: 0,
            });
        }

        // BFS for shortest path
        let mut visited: HashSet<String> = HashSet::new();
        let mut parent: HashMap<String, (String, String)> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();

        queue.push_back(from.to_string());
        visited.insert(from.to_string());

        while let Some(current) = queue.pop_front() {
            if current == to {
                // Reconstruct path
                let mut path = vec![to.to_string()];
                let mut relations = Vec::new();
                let mut node = to.to_string();

                while let Some((parent_node, relation)) = parent.get(&node) {
                    path.push(parent_node.clone());
                    relations.push(relation.clone());
                    node = parent_node.clone();
                }

                path.reverse();
                relations.reverse();

                return Some(EntityPath {
                    length: path.len() - 1,
                    entities: path,
                    relations,
                    weight: 1.0,
                });
            }

            // Explore neighbors
            if let Some(rels) = self.outgoing.get(&current) {
                for (target, rel) in rels {
                    if !visited.contains(target) {
                        visited.insert(target.clone());
                        parent.insert(target.clone(), (current.clone(), rel.relation_type.clone()));
                        queue.push_back(target.clone());
                    }
                }
            }

            if self.config.bidirectional_search {
                if let Some(rels) = self.incoming.get(&current) {
                    for (source, rel) in rels {
                        if !visited.contains(source) {
                            visited.insert(source.clone());
                            parent.insert(source.clone(), (current.clone(), rel.relation_type.clone()));
                            queue.push_back(source.clone());
                        }
                    }
                }
            }

            // Check path length
            if parent.len() > self.config.max_path_length * self.entities.len() {
                break;
            }
        }

        None
    }

    /// Extract subgraph around entities.
    pub fn extract_subgraph(&self, center_ids: &[String], hops: usize) -> SubGraph {
        let reachable = self.get_reachable(center_ids, hops);

        let entities: Vec<Entity> = reachable.iter()
            .filter_map(|id| self.entities.get(id).cloned())
            .collect();

        let mut relations = Vec::new();
        for id in &reachable {
            if let Some(rels) = self.outgoing.get(id) {
                for (target, rel) in rels {
                    if reachable.contains(target) {
                        relations.push(rel.clone());
                    }
                }
            }
        }

        SubGraph {
            entities,
            relations,
            center_ids: center_ids.to_vec(),
        }
    }

    /// Compute graph score based on centrality.
    fn compute_graph_score(&self, entity_id: &str) -> f32 {
        let out_degree = self.outgoing.get(entity_id).map(|r| r.len()).unwrap_or(0);
        let in_degree = self.incoming.get(entity_id).map(|r| r.len()).unwrap_or(0);
        let total_degree = out_degree + in_degree;

        // Simple degree centrality normalized
        let max_degree = self.entities.len().max(1);
        (total_degree as f32 / max_degree as f32).min(1.0)
    }

    /// Cosine similarity.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
    }

    /// Get entities by type.
    pub fn get_by_type(&self, entity_type: &str) -> Vec<&Entity> {
        self.type_index
            .get(entity_type)
            .map(|ids| ids.iter().filter_map(|id| self.entities.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get relations by type.
    pub fn get_relations_by_type(&self, relation_type: &str) -> Vec<&Relation> {
        self.relation_index
            .get(relation_type)
            .map(|rels| rels.iter().collect())
            .unwrap_or_default()
    }

    /// Get graph statistics.
    pub fn stats(&self) -> GraphStats {
        let mut relation_counts: HashMap<String, usize> = HashMap::new();
        for rels in self.relation_index.values() {
            for rel in rels {
                *relation_counts.entry(rel.relation_type.clone()).or_default() += 1;
            }
        }

        GraphStats {
            entity_count: self.entities.len(),
            relation_count: relation_counts.values().sum(),
            entity_types: self.type_index.keys().cloned().collect(),
            relation_types: self.relation_index.keys().cloned().collect(),
            relation_counts,
        }
    }

    /// Entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Relation count.
    pub fn relation_count(&self) -> usize {
        self.outgoing.values().map(|r| r.len()).sum()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// A subgraph extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubGraph {
    /// Entities in subgraph.
    pub entities: Vec<Entity>,
    /// Relations in subgraph.
    pub relations: Vec<Relation>,
    /// Center entity IDs.
    pub center_ids: Vec<String>,
}

/// Graph statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Number of entities.
    pub entity_count: usize,
    /// Number of relations.
    pub relation_count: usize,
    /// Entity types.
    pub entity_types: Vec<String>,
    /// Relation types.
    pub relation_types: Vec<String>,
    /// Counts per relation type.
    pub relation_counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> KnowledgeGraph {
        let mut kg = KnowledgeGraph::new();

        kg.add_entity("rust", "language", "Rust", &[1.0, 0.0, 0.0], HashMap::new()).unwrap();
        kg.add_entity("python", "language", "Python", &[0.9, 0.1, 0.0], HashMap::new()).unwrap();
        kg.add_entity("programming", "concept", "Programming", &[0.5, 0.5, 0.0], HashMap::new()).unwrap();
        kg.add_entity("systems", "concept", "Systems", &[0.3, 0.0, 0.7], HashMap::new()).unwrap();

        kg.add_relation("rust", "programming", "is_a").unwrap();
        kg.add_relation("python", "programming", "is_a").unwrap();
        kg.add_relation("rust", "systems", "used_for").unwrap();

        kg
    }

    #[test]
    fn test_create_graph() {
        let kg = KnowledgeGraph::new();
        assert_eq!(kg.entity_count(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("e1", "type1", "Entity 1", &[1.0, 0.0], HashMap::new()).unwrap();

        assert_eq!(kg.entity_count(), 1);
        assert!(kg.get_entity("e1").is_some());
    }

    #[test]
    fn test_add_relation() {
        let kg = create_test_graph();
        assert_eq!(kg.relation_count(), 3);
    }

    #[test]
    fn test_get_neighbors() {
        let kg = create_test_graph();
        let neighbors = kg.get_neighbors("rust");

        assert_eq!(neighbors.len(), 2); // programming and systems
    }

    #[test]
    fn test_search() {
        let kg = create_test_graph();
        let query = vec![1.0, 0.0, 0.0];

        let results = kg.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].entity.id, "rust");
    }

    #[test]
    fn test_search_with_filter() {
        let kg = create_test_graph();
        let query = vec![1.0, 0.0, 0.0];

        let results = kg.search_with_filter(&query, 10, Some("language"));

        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.entity.entity_type, "language");
        }
    }

    #[test]
    fn test_find_path() {
        let kg = create_test_graph();

        let path = kg.find_path("rust", "python");
        assert!(path.is_some());

        let path = path.unwrap();
        assert!(path.entities.contains(&"rust".to_string()));
        assert!(path.entities.contains(&"python".to_string()));
    }

    #[test]
    fn test_no_path() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("a", "type", "A", &[1.0], HashMap::new()).unwrap();
        kg.add_entity("b", "type", "B", &[0.0], HashMap::new()).unwrap();

        let path = kg.find_path("a", "b");
        assert!(path.is_none());
    }

    #[test]
    fn test_extract_subgraph() {
        let kg = create_test_graph();

        let subgraph = kg.extract_subgraph(&["rust".to_string()], 1);

        assert!(subgraph.entities.len() >= 2); // rust and its neighbors
    }

    #[test]
    fn test_search_with_context() {
        let kg = create_test_graph();
        let query = vec![0.5, 0.5, 0.0];

        let results = kg.search_with_context(&query, 4, &["rust".to_string()]);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_stats() {
        let kg = create_test_graph();
        let stats = kg.stats();

        assert_eq!(stats.entity_count, 4);
        assert_eq!(stats.relation_count, 3);
        assert!(stats.entity_types.contains(&"language".to_string()));
        assert!(stats.relation_types.contains(&"is_a".to_string()));
    }

    #[test]
    fn test_get_by_type() {
        let kg = create_test_graph();

        let languages = kg.get_by_type("language");
        assert_eq!(languages.len(), 2);

        let concepts = kg.get_by_type("concept");
        assert_eq!(concepts.len(), 2);
    }

    #[test]
    fn test_get_relations_by_type() {
        let kg = create_test_graph();

        let is_a = kg.get_relations_by_type("is_a");
        assert_eq!(is_a.len(), 2);
    }

    #[test]
    fn test_update_embedding() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("e1", "type", "E1", &[1.0, 0.0], HashMap::new()).unwrap();

        kg.update_embedding("e1", &[0.0, 1.0]).unwrap();

        let entity = kg.get_entity("e1").unwrap();
        assert_eq!(entity.embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn test_weighted_relation() {
        let mut kg = KnowledgeGraph::new();
        kg.add_entity("a", "type", "A", &[1.0], HashMap::new()).unwrap();
        kg.add_entity("b", "type", "B", &[0.0], HashMap::new()).unwrap();

        kg.add_weighted_relation("a", "b", "strong", 0.9, HashMap::new()).unwrap();

        let rels = kg.get_outgoing("a");
        assert_eq!(rels[0].weight, 0.9);
    }

    #[test]
    fn test_graph_search_result_related() {
        let kg = create_test_graph();
        let query = vec![1.0, 0.0, 0.0];

        let results = kg.search(&query, 1);
        assert!(!results[0].related.is_empty());
    }

    #[test]
    fn test_self_path() {
        let kg = create_test_graph();
        let path = kg.find_path("rust", "rust");

        assert!(path.is_some());
        assert_eq!(path.unwrap().length, 0);
    }

    #[test]
    fn test_config() {
        let config = KnowledgeGraphConfig {
            similarity_weight: 0.5,
            graph_weight: 0.5,
            ..Default::default()
        };

        let kg = KnowledgeGraph::with_config(config);
        assert_eq!(kg.config.similarity_weight, 0.5);
    }
}
