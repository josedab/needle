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

        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap_or(std::cmp::Ordering::Equal));
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

        results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap_or(std::cmp::Ordering::Equal));
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

// ============================================================================
// GraphRAG Integration
// ============================================================================

/// Configuration for GraphRAG retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRAGConfig {
    /// Maximum hops for multi-hop reasoning
    pub max_hops: usize,
    /// Weight for vector similarity in final score
    pub similarity_weight: f32,
    /// Weight for graph relevance in final score
    pub graph_weight: f32,
    /// Weight for entity type matching
    pub type_weight: f32,
    /// Minimum similarity threshold for retrieval
    pub min_similarity: f32,
    /// Maximum context tokens (approximate)
    pub max_context_tokens: usize,
    /// Include relation descriptions in context
    pub include_relations: bool,
    /// Include entity properties in context
    pub include_properties: bool,
    /// Deduplicate entities in context
    pub deduplicate: bool,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            similarity_weight: 0.6,
            graph_weight: 0.4,
            type_weight: 0.1,
            min_similarity: 0.0,
            max_context_tokens: 4000,
            include_relations: true,
            include_properties: true,
            deduplicate: true,
        }
    }
}

/// A chunk of context for RAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkContext {
    /// Entity this context is about
    pub entity_id: String,
    /// Entity label
    pub label: String,
    /// Entity type
    pub entity_type: String,
    /// Text content
    pub content: String,
    /// Relevance score
    pub score: f32,
    /// Hop distance from query (0 = directly matched)
    pub hop_distance: usize,
    /// Related entity IDs
    pub related_entities: Vec<String>,
    /// Relation descriptions
    pub relation_descriptions: Vec<String>,
}

/// GraphRAG retriever for knowledge-enhanced RAG
pub struct GraphRAGRetriever<'a> {
    kg: &'a KnowledgeGraph,
    config: GraphRAGConfig,
}

impl<'a> GraphRAGRetriever<'a> {
    /// Create a new GraphRAG retriever
    pub fn new(kg: &'a KnowledgeGraph, config: GraphRAGConfig) -> Self {
        Self { kg, config }
    }

    /// Create with default config
    pub fn with_defaults(kg: &'a KnowledgeGraph) -> Self {
        Self::new(kg, GraphRAGConfig::default())
    }

    /// Retrieve context for a query embedding
    pub fn retrieve(&self, query: &[f32], k: usize) -> Vec<ChunkContext> {
        let mut contexts = Vec::new();
        let mut seen = HashSet::new();

        // Get initial results from vector search
        let initial_results = self.kg.search(query, k * 2);

        // Process each result with multi-hop expansion
        for (rank, result) in initial_results.iter().enumerate() {
            if seen.contains(&result.entity.id) {
                continue;
            }
            seen.insert(result.entity.id.clone());

            // Add the entity itself
            let context = self.build_chunk_context(&result.entity, result.combined_score, 0);
            contexts.push(context);

            // Multi-hop expansion
            if self.config.max_hops > 0 {
                let expanded = self.expand_hops(&result.entity.id, query, 1, &mut seen);
                contexts.extend(expanded);
            }

            // Check if we have enough context
            if contexts.len() >= k && rank > k / 2 {
                break;
            }
        }

        // Sort by score and truncate
        contexts.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        contexts.truncate(k);

        contexts
    }

    /// Expand from an entity through multiple hops
    fn expand_hops(
        &self,
        entity_id: &str,
        query: &[f32],
        current_hop: usize,
        seen: &mut HashSet<String>,
    ) -> Vec<ChunkContext> {
        if current_hop > self.config.max_hops {
            return vec![];
        }

        let mut contexts = Vec::new();

        // Get related entities (neighbors)
        let neighbors = self.kg.get_neighbors(entity_id);

        for related in neighbors {
            let related_id = &related.entity.id;
            if seen.contains(related_id) {
                continue;
            }
            seen.insert(related_id.clone());

            // Calculate score with decay for hops
            let similarity = self.cosine_similarity(query, &related.entity.embedding);
            let hop_decay = 1.0 / (1.0 + current_hop as f32 * 0.5);
            let score = similarity * hop_decay * self.config.graph_weight;

            if score >= self.config.min_similarity {
                let mut context = self.build_chunk_context(&related.entity, score, current_hop);
                context.relation_descriptions.push(format!(
                    "Related via '{}' from previous entity",
                    related.relation
                ));
                contexts.push(context);

                // Continue expanding if more hops allowed
                if current_hop < self.config.max_hops {
                    let deeper = self.expand_hops(related_id, query, current_hop + 1, seen);
                    contexts.extend(deeper);
                }
            }
        }

        contexts
    }

    /// Build a chunk context from an entity
    fn build_chunk_context(&self, entity: &Entity, score: f32, hop_distance: usize) -> ChunkContext {
        let mut content = format!("{} ({})", entity.label, entity.entity_type);

        // Add properties if configured
        if self.config.include_properties && !entity.properties.is_empty() {
            let props: Vec<String> = entity
                .properties
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect();
            content.push_str(&format!(". Properties: {}", props.join(", ")));
        }

        // Get related entities
        let related: Vec<String> = self
            .kg
            .get_neighbors(&entity.id)
            .iter()
            .map(|r| r.entity.id.clone())
            .collect();

        // Get relation descriptions
        let mut relation_descriptions = Vec::new();
        if self.config.include_relations {
            for rel in self.kg.get_outgoing(&entity.id) {
                if let Some(target) = self.kg.get_entity(&rel.target) {
                    relation_descriptions.push(format!(
                        "{} {} {}",
                        entity.label, rel.relation_type, target.label
                    ));
                }
            }
            for rel in self.kg.get_incoming(&entity.id) {
                if let Some(source) = self.kg.get_entity(&rel.source) {
                    relation_descriptions.push(format!(
                        "{} {} {}",
                        source.label, rel.relation_type, entity.label
                    ));
                }
            }
        }

        ChunkContext {
            entity_id: entity.id.clone(),
            label: entity.label.clone(),
            entity_type: entity.entity_type.clone(),
            content,
            score,
            hop_distance,
            related_entities: related,
            relation_descriptions,
        }
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
            return 0.0;
        }
        (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
    }

    /// Retrieve and format context as text
    pub fn retrieve_text(&self, query: &[f32], k: usize) -> String {
        let contexts = self.retrieve(query, k);
        self.format_context(&contexts)
    }

    /// Format contexts as text for LLM consumption
    pub fn format_context(&self, contexts: &[ChunkContext]) -> String {
        let mut text = String::new();

        for (i, ctx) in contexts.iter().enumerate() {
            text.push_str(&format!("{}. {}\n", i + 1, ctx.content));

            if !ctx.relation_descriptions.is_empty() {
                text.push_str("   Relationships:\n");
                for rel in &ctx.relation_descriptions {
                    text.push_str(&format!("   - {}\n", rel));
                }
            }
            text.push('\n');
        }

        text
    }
}

/// Entity linker for connecting query terms to graph entities
pub struct EntityLinker<'a> {
    kg: &'a KnowledgeGraph,
    type_preferences: HashMap<String, f32>,
}

impl<'a> EntityLinker<'a> {
    /// Create a new entity linker
    pub fn new(kg: &'a KnowledgeGraph) -> Self {
        Self {
            kg,
            type_preferences: HashMap::new(),
        }
    }

    /// Set preference weights for entity types
    pub fn with_type_preferences(mut self, prefs: HashMap<String, f32>) -> Self {
        self.type_preferences = prefs;
        self
    }

    /// Link a query embedding to the most relevant entity
    pub fn link(&self, query: &[f32]) -> Option<(String, f32)> {
        let results = self.kg.search(query, 1);
        results.first().map(|r| (r.entity.id.clone(), r.combined_score))
    }

    /// Link a query embedding to multiple entities
    pub fn link_multiple(&self, query: &[f32], k: usize) -> Vec<(String, f32)> {
        self.kg
            .search(query, k)
            .iter()
            .map(|r| (r.entity.id.clone(), r.combined_score))
            .collect()
    }

    /// Link by entity type
    pub fn link_by_type(&self, query: &[f32], entity_type: &str, k: usize) -> Vec<(String, f32)> {
        let entities = self.kg.get_by_type(entity_type);
        let mut scored: Vec<(String, f32)> = entities
            .iter()
            .map(|e| {
                let similarity = self.cosine_similarity(query, &e.embedding);
                (e.id.clone(), similarity)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a < f32::EPSILON || mag_b < f32::EPSILON {
            return 0.0;
        }
        (dot / (mag_a * mag_b)).clamp(0.0, 1.0)
    }
}

/// Multi-hop reasoning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPath {
    /// Starting entity
    pub start: String,
    /// Ending entity
    pub end: String,
    /// Path of entity IDs
    pub path: Vec<String>,
    /// Relation types along the path
    pub relations: Vec<String>,
    /// Confidence score
    pub confidence: f32,
    /// Explanation
    pub explanation: String,
}

/// Multi-hop reasoner for deeper context retrieval
pub struct MultiHopReasoner<'a> {
    kg: &'a KnowledgeGraph,
    max_depth: usize,
}

impl<'a> MultiHopReasoner<'a> {
    /// Create a new multi-hop reasoner
    pub fn new(kg: &'a KnowledgeGraph, max_depth: usize) -> Self {
        Self { kg, max_depth }
    }

    /// Find all paths between two entities
    pub fn find_paths(&self, start: &str, end: &str) -> Vec<ReasoningPath> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut current_path = vec![start.to_string()];
        let mut current_relations = Vec::new();

        self.dfs_paths(
            start,
            end,
            &mut visited,
            &mut current_path,
            &mut current_relations,
            &mut paths,
            0,
        );

        // Sort by shortest path
        paths.sort_by(|a, b| a.path.len().cmp(&b.path.len()));
        paths
    }

    fn dfs_paths(
        &self,
        current: &str,
        target: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        relations: &mut Vec<String>,
        paths: &mut Vec<ReasoningPath>,
        depth: usize,
    ) {
        if depth > self.max_depth {
            return;
        }

        if current == target {
            let explanation = self.generate_explanation(path, relations);
            let confidence = 1.0 / (1.0 + depth as f32 * 0.2);

            paths.push(ReasoningPath {
                start: path[0].clone(),
                end: current.to_string(),
                path: path.clone(),
                relations: relations.clone(),
                confidence,
                explanation,
            });
            return;
        }

        visited.insert(current.to_string());

        for rel in self.kg.get_outgoing(current) {
            if !visited.contains(&rel.target) {
                path.push(rel.target.clone());
                relations.push(rel.relation_type.clone());

                self.dfs_paths(
                    &rel.target,
                    target,
                    visited,
                    path,
                    relations,
                    paths,
                    depth + 1,
                );

                path.pop();
                relations.pop();
            }
        }

        visited.remove(current);
    }

    fn generate_explanation(&self, path: &[String], relations: &[String]) -> String {
        let mut explanation = String::new();

        for (i, entity_id) in path.iter().enumerate() {
            if let Some(entity) = self.kg.get_entity(entity_id) {
                if i > 0 {
                    explanation.push_str(&format!(" --[{}]--> ", relations[i - 1]));
                }
                explanation.push_str(&entity.label);
            }
        }

        explanation
    }

    /// Get reasoning chains starting from an entity
    pub fn reason_from(&self, start: &str, depth: usize) -> Vec<ReasoningPath> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start.to_string());

        let mut queue: VecDeque<(String, Vec<String>, Vec<String>, usize)> = VecDeque::new();
        queue.push_back((
            start.to_string(),
            vec![start.to_string()],
            vec![],
            0,
        ));

        while let Some((current, path, relations, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                if path.len() > 1 {
                    let explanation = self.generate_explanation(&path, &relations);
                    let confidence = 1.0 / (1.0 + current_depth as f32 * 0.2);

                    paths.push(ReasoningPath {
                        start: start.to_string(),
                        end: current.clone(),
                        path: path.clone(),
                        relations: relations.clone(),
                        confidence,
                        explanation,
                    });
                }
                continue;
            }

            for rel in self.kg.get_outgoing(&current) {
                if !visited.contains(&rel.target) {
                    visited.insert(rel.target.clone());

                    let mut new_path = path.clone();
                    new_path.push(rel.target.clone());

                    let mut new_relations = relations.clone();
                    new_relations.push(rel.relation_type.clone());

                    queue.push_back((
                        rel.target.clone(),
                        new_path,
                        new_relations,
                        current_depth + 1,
                    ));
                }
            }
        }

        paths
    }
}

/// GraphRAG query builder
pub struct GraphRAGQueryBuilder<'a> {
    retriever: GraphRAGRetriever<'a>,
    query: Vec<f32>,
    k: usize,
    entity_filter: Option<String>,
    include_paths: bool,
}

impl<'a> GraphRAGQueryBuilder<'a> {
    /// Create a new query builder
    pub fn new(kg: &'a KnowledgeGraph, query: Vec<f32>) -> Self {
        Self {
            retriever: GraphRAGRetriever::with_defaults(kg),
            query,
            k: 10,
            entity_filter: None,
            include_paths: false,
        }
    }

    /// Set number of results
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set GraphRAG config
    #[must_use]
    pub fn config(mut self, config: GraphRAGConfig) -> Self {
        self.retriever = GraphRAGRetriever::new(self.retriever.kg, config);
        self
    }

    /// Filter by entity type
    #[must_use]
    pub fn filter_type(mut self, entity_type: &str) -> Self {
        self.entity_filter = Some(entity_type.to_string());
        self
    }

    /// Include reasoning paths
    #[must_use]
    pub fn include_paths(mut self) -> Self {
        self.include_paths = true;
        self
    }

    /// Execute the query
    pub fn execute(self) -> Vec<ChunkContext> {
        let mut results = self.retriever.retrieve(&self.query, self.k);

        // Apply entity type filter if set
        if let Some(ref filter) = self.entity_filter {
            results.retain(|r| &r.entity_type == filter);
        }

        results
    }

    /// Execute and get formatted text
    pub fn execute_text(self) -> String {
        let mut results = self.retriever.retrieve(&self.query, self.k);

        // Apply entity type filter if set
        if let Some(ref filter) = self.entity_filter {
            results.retain(|r| &r.entity_type == filter);
        }

        self.retriever.format_context(&results)
    }
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

    // ==========================================================================
    // GraphRAG Tests
    // ==========================================================================

    #[test]
    fn test_graphrag_config_default() {
        let config = GraphRAGConfig::default();
        assert_eq!(config.max_hops, 2);
        assert_eq!(config.similarity_weight, 0.6);
        assert_eq!(config.graph_weight, 0.4);
    }

    #[test]
    fn test_graphrag_retriever_creation() {
        let kg = create_test_graph();
        let config = GraphRAGConfig::default();
        let _retriever = GraphRAGRetriever::new(&kg, config);
    }

    #[test]
    fn test_graphrag_retriever_retrieve() {
        let kg = create_test_graph();
        let retriever = GraphRAGRetriever::with_defaults(&kg);

        let query = vec![1.0, 0.0, 0.0];
        let contexts = retriever.retrieve(&query, 3);

        assert!(!contexts.is_empty());
        // First result should have hop_distance 0
        assert_eq!(contexts[0].hop_distance, 0);
    }

    #[test]
    fn test_graphrag_retrieve_text() {
        let kg = create_test_graph();
        let retriever = GraphRAGRetriever::with_defaults(&kg);

        let query = vec![1.0, 0.0, 0.0];
        let text = retriever.retrieve_text(&query, 3);

        assert!(!text.is_empty());
        // Should contain numbered items
        assert!(text.contains("1."));
    }

    #[test]
    fn test_chunk_context_fields() {
        let context = ChunkContext {
            entity_id: "test".to_string(),
            label: "Test Entity".to_string(),
            entity_type: "type".to_string(),
            content: "Test content".to_string(),
            score: 0.8,
            hop_distance: 1,
            related_entities: vec!["other".to_string()],
            relation_descriptions: vec!["test relation".to_string()],
        };

        assert_eq!(context.entity_id, "test");
        assert_eq!(context.hop_distance, 1);
        assert_eq!(context.related_entities.len(), 1);
    }

    #[test]
    fn test_entity_linker_link() {
        let kg = create_test_graph();
        let linker = EntityLinker::new(&kg);

        let query = vec![1.0, 0.0, 0.0];
        let linked = linker.link(&query);

        assert!(linked.is_some());
        let (id, _score) = linked.unwrap();
        assert_eq!(id, "rust"); // Should link to rust as it's closest to [1.0, 0.0, 0.0]
    }

    #[test]
    fn test_entity_linker_multiple() {
        let kg = create_test_graph();
        let linker = EntityLinker::new(&kg);

        let query = vec![0.5, 0.5, 0.0];
        let linked = linker.link_multiple(&query, 3);

        assert_eq!(linked.len(), 3);
    }

    #[test]
    fn test_entity_linker_by_type() {
        let kg = create_test_graph();
        let linker = EntityLinker::new(&kg);

        let query = vec![1.0, 0.0, 0.0];
        let linked = linker.link_by_type(&query, "language", 2);

        assert_eq!(linked.len(), 2);
        // All results should be languages
        for (id, _) in &linked {
            let entity = kg.get_entity(id).unwrap();
            assert_eq!(entity.entity_type, "language");
        }
    }

    #[test]
    fn test_multi_hop_reasoner_find_paths() {
        let kg = create_test_graph();
        let reasoner = MultiHopReasoner::new(&kg, 3);

        // Find paths from rust to programming
        let paths = reasoner.find_paths("rust", "programming");

        assert!(!paths.is_empty());
        assert_eq!(paths[0].start, "rust");
        assert_eq!(paths[0].end, "programming");
    }

    #[test]
    fn test_multi_hop_reasoner_reason_from() {
        let kg = create_test_graph();
        let reasoner = MultiHopReasoner::new(&kg, 3);

        // Use depth=1 to find direct neighbors
        let paths = reasoner.reason_from("rust", 1);

        // Should find paths to connected entities (programming and systems)
        assert!(!paths.is_empty());
        for path in &paths {
            assert_eq!(path.start, "rust");
            assert!(path.confidence > 0.0);
            assert!(path.path.len() >= 2); // At least start + end
        }
    }

    #[test]
    fn test_reasoning_path_fields() {
        let path = ReasoningPath {
            start: "a".to_string(),
            end: "b".to_string(),
            path: vec!["a".to_string(), "b".to_string()],
            relations: vec!["connected".to_string()],
            confidence: 0.8,
            explanation: "A --[connected]--> B".to_string(),
        };

        assert_eq!(path.path.len(), 2);
        assert_eq!(path.relations.len(), 1);
        assert!(path.confidence > 0.0);
    }

    #[test]
    fn test_graphrag_query_builder() {
        let kg = create_test_graph();
        let query = vec![1.0, 0.0, 0.0];

        let results = GraphRAGQueryBuilder::new(&kg, query)
            .k(5)
            .execute();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_graphrag_query_builder_filter() {
        let kg = create_test_graph();
        let query = vec![0.5, 0.5, 0.0];

        let results = GraphRAGQueryBuilder::new(&kg, query)
            .k(10)
            .filter_type("language")
            .execute();

        // All results should be of type "language"
        for ctx in &results {
            assert_eq!(ctx.entity_type, "language");
        }
    }

    #[test]
    fn test_graphrag_query_builder_text() {
        let kg = create_test_graph();
        let query = vec![1.0, 0.0, 0.0];

        let text = GraphRAGQueryBuilder::new(&kg, query)
            .k(3)
            .execute_text();

        assert!(!text.is_empty());
    }

    #[test]
    fn test_graphrag_config_custom() {
        let config = GraphRAGConfig {
            max_hops: 3,
            similarity_weight: 0.7,
            graph_weight: 0.3,
            min_similarity: 0.1,
            ..Default::default()
        };

        assert_eq!(config.max_hops, 3);
        assert_eq!(config.similarity_weight, 0.7);
    }

    #[test]
    fn test_graphrag_multi_hop_expansion() {
        let mut kg = KnowledgeGraph::new();

        // Create a chain: a -> b -> c -> d
        kg.add_entity("a", "type", "A", &[1.0, 0.0, 0.0], HashMap::new()).unwrap();
        kg.add_entity("b", "type", "B", &[0.8, 0.2, 0.0], HashMap::new()).unwrap();
        kg.add_entity("c", "type", "C", &[0.6, 0.4, 0.0], HashMap::new()).unwrap();
        kg.add_entity("d", "type", "D", &[0.4, 0.6, 0.0], HashMap::new()).unwrap();

        kg.add_relation("a", "b", "link").unwrap();
        kg.add_relation("b", "c", "link").unwrap();
        kg.add_relation("c", "d", "link").unwrap();

        let config = GraphRAGConfig {
            max_hops: 3,
            ..Default::default()
        };
        let retriever = GraphRAGRetriever::new(&kg, config);

        let query = vec![1.0, 0.0, 0.0];
        let contexts = retriever.retrieve(&query, 10);

        // Should find entities through multi-hop
        let ids: Vec<_> = contexts.iter().map(|c| c.entity_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        // Should also find b through the relation
    }
}
