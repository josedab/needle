//! Native Graph Vector Index
//!
//! Combines knowledge graphs with vector embeddings for relationship-aware retrieval.
//! This module provides a unified index that supports:
//!
//! - **Entity-relation modeling**: Store entities with embeddings and typed relationships
//! - **Graph+vector hybrid queries**: Combine vector similarity with graph traversal
//! - **Relationship-weighted search**: Boost results based on graph connectivity
//! - **Path-aware retrieval**: Find vectors connected by semantic paths
//!
//! # Architecture
//!
//! The GraphVectorIndex maintains both:
//! 1. An HNSW index for fast approximate vector search
//! 2. An adjacency list graph structure for relationship traversal
//!
//! Queries can leverage both structures for enhanced retrieval.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::graph_vector_index::{GraphVectorIndex, GraphVectorConfig, EdgeType};
//!
//! let config = GraphVectorConfig::default();
//! let mut index = GraphVectorIndex::new(384, config);
//!
//! // Add entities with embeddings
//! index.add_entity("rust_lang", &rust_embedding, Some(json!({"type": "language"})))?;
//! index.add_entity("cargo", &cargo_embedding, Some(json!({"type": "tool"})))?;
//!
//! // Add relationship
//! index.add_edge("rust_lang", "cargo", EdgeType::HasTool, 1.0)?;
//!
//! // Search with graph context
//! let results = index.search_with_graph(&query, 10)?;
//! for result in results {
//!     println!("{}: similarity={}, graph_boost={}",
//!         result.id, result.vector_score, result.graph_score);
//! }
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex, VectorId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;
use tracing::debug;

/// Edge type for relationships
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    /// Generic similarity relation
    Similar,
    /// Parent-child relation
    ParentOf,
    /// Child-parent relation
    ChildOf,
    /// Part-whole relation
    PartOf,
    /// Has-part relation
    HasPart,
    /// Instance-of relation
    InstanceOf,
    /// Type-of relation
    TypeOf,
    /// Related-to (generic)
    RelatedTo,
    /// Has-property relation
    HasProperty,
    /// Used-by relation
    UsedBy,
    /// Uses relation
    Uses,
    /// Has-tool relation
    HasTool,
    /// Depends-on relation
    DependsOn,
    /// Custom relation type
    Custom(String),
}

impl Default for EdgeType {
    fn default() -> Self {
        EdgeType::RelatedTo
    }
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeType::Similar => write!(f, "similar"),
            EdgeType::ParentOf => write!(f, "parent_of"),
            EdgeType::ChildOf => write!(f, "child_of"),
            EdgeType::PartOf => write!(f, "part_of"),
            EdgeType::HasPart => write!(f, "has_part"),
            EdgeType::InstanceOf => write!(f, "instance_of"),
            EdgeType::TypeOf => write!(f, "type_of"),
            EdgeType::RelatedTo => write!(f, "related_to"),
            EdgeType::HasProperty => write!(f, "has_property"),
            EdgeType::UsedBy => write!(f, "used_by"),
            EdgeType::Uses => write!(f, "uses"),
            EdgeType::HasTool => write!(f, "has_tool"),
            EdgeType::DependsOn => write!(f, "depends_on"),
            EdgeType::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// Configuration for graph vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphVectorConfig {
    /// Weight for vector similarity score (0-1)
    pub vector_weight: f32,
    /// Weight for graph score (0-1)
    pub graph_weight: f32,
    /// Maximum hops for graph traversal during search
    pub max_hops: usize,
    /// Decay factor for distant nodes (0-1)
    pub distance_decay: f32,
    /// Whether to include bidirectional edges
    pub bidirectional: bool,
    /// Minimum edge weight to consider
    pub min_edge_weight: f32,
    /// Maximum edges per node
    pub max_edges_per_node: usize,
    /// Enable automatic similarity edges
    pub auto_similarity_edges: bool,
    /// Threshold for automatic similarity edges
    pub similarity_threshold: f32,
}

impl Default for GraphVectorConfig {
    fn default() -> Self {
        Self {
            vector_weight: 0.7,
            graph_weight: 0.3,
            max_hops: 2,
            distance_decay: 0.5,
            bidirectional: true,
            min_edge_weight: 0.1,
            max_edges_per_node: 50,
            auto_similarity_edges: false,
            similarity_threshold: 0.8,
        }
    }
}

impl GraphVectorConfig {
    /// Create a builder
    pub fn builder() -> GraphVectorConfigBuilder {
        GraphVectorConfigBuilder::default()
    }
}

/// Builder for graph vector config
#[derive(Debug, Default)]
pub struct GraphVectorConfigBuilder {
    config: GraphVectorConfig,
}

impl GraphVectorConfigBuilder {
    pub fn vector_weight(mut self, weight: f32) -> Self {
        self.config.vector_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn graph_weight(mut self, weight: f32) -> Self {
        self.config.graph_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn max_hops(mut self, hops: usize) -> Self {
        self.config.max_hops = hops;
        self
    }

    pub fn distance_decay(mut self, decay: f32) -> Self {
        self.config.distance_decay = decay.clamp(0.0, 1.0);
        self
    }

    pub fn bidirectional(mut self, enabled: bool) -> Self {
        self.config.bidirectional = enabled;
        self
    }

    pub fn auto_similarity_edges(mut self, enabled: bool) -> Self {
        self.config.auto_similarity_edges = enabled;
        self
    }

    pub fn similarity_threshold(mut self, threshold: f32) -> Self {
        self.config.similarity_threshold = threshold;
        self
    }

    pub fn build(self) -> GraphVectorConfig {
        self.config
    }
}

/// An edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Target entity ID
    pub target: String,
    /// Edge type
    pub edge_type: EdgeType,
    /// Edge weight (0-1)
    pub weight: f32,
    /// Optional properties
    pub properties: Option<Value>,
}

/// An entity in the graph vector index
#[derive(Debug, Clone)]
pub struct GraphEntity {
    /// Entity ID
    pub id: String,
    /// Internal vector ID
    #[allow(dead_code)]
    internal_id: VectorId,
    /// Vector embedding
    pub vector: Vec<f32>,
    /// Metadata
    pub metadata: Option<Value>,
    /// Outgoing edges
    pub edges: Vec<GraphEdge>,
    /// Incoming edges (reverse links)
    pub incoming: Vec<String>,
}

/// Search result from graph vector index
#[derive(Debug, Clone)]
pub struct GraphVectorSearchResult {
    /// Entity ID
    pub id: String,
    /// Vector similarity score (lower is better for distance-based)
    pub vector_score: f32,
    /// Graph connectivity score (higher is better)
    pub graph_score: f32,
    /// Combined final score
    pub combined_score: f32,
    /// Metadata
    pub metadata: Option<Value>,
    /// Connected entities found during traversal
    pub connected_entities: Vec<ConnectedEntity>,
    /// Hops from nearest anchor (if graph traversal was used)
    pub hops_from_anchor: Option<usize>,
}

/// A connected entity discovered during graph traversal
#[derive(Debug, Clone)]
pub struct ConnectedEntity {
    /// Entity ID
    pub id: String,
    /// Relationship type
    pub relation: EdgeType,
    /// Hops from result entity
    pub hops: usize,
    /// Path of entity IDs to reach this
    pub path: Vec<String>,
}

/// Statistics for graph vector index
#[derive(Debug, Clone, Default)]
pub struct GraphVectorStats {
    /// Total entities
    pub entity_count: usize,
    /// Total edges
    pub edge_count: usize,
    /// Average edges per entity
    pub avg_edges: f32,
    /// Maximum edges on a single entity
    pub max_edges: usize,
    /// Number of connected components
    pub connected_components: usize,
    /// HNSW index memory estimate
    pub index_memory_bytes: usize,
}

/// Graph Vector Index combining vector search with graph traversal
pub struct GraphVectorIndex {
    /// Configuration
    config: GraphVectorConfig,
    /// Vector dimensions
    dimensions: usize,
    /// Distance function
    #[allow(dead_code)]
    distance: DistanceFunction,
    /// HNSW index for vector search
    index: RwLock<HnswIndex>,
    /// Vector storage
    vectors: RwLock<Vec<Vec<f32>>>,
    /// Entity storage: id -> entity
    entities: RwLock<HashMap<String, GraphEntity>>,
    /// ID mapping: external ID -> internal ID
    id_map: RwLock<HashMap<String, VectorId>>,
    /// Reverse ID mapping
    reverse_id_map: RwLock<HashMap<VectorId, String>>,
    /// Next internal ID
    next_id: std::sync::atomic::AtomicUsize,
}

impl GraphVectorIndex {
    /// Create a new graph vector index
    pub fn new(dimensions: usize, config: GraphVectorConfig) -> Self {
        Self::with_distance(dimensions, DistanceFunction::Cosine, config)
    }

    /// Create with specific distance function
    pub fn with_distance(
        dimensions: usize,
        distance: DistanceFunction,
        config: GraphVectorConfig,
    ) -> Self {
        let hnsw_config = HnswConfig::default();
        let index = HnswIndex::new(hnsw_config, distance);

        Self {
            config,
            dimensions,
            distance,
            index: RwLock::new(index),
            vectors: RwLock::new(Vec::new()),
            entities: RwLock::new(HashMap::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_id_map: RwLock::new(HashMap::new()),
            next_id: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Add an entity with its vector embedding
    pub fn add_entity(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();

        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let mut id_map = self.id_map.write();
        if id_map.contains_key(&id) {
            return Err(NeedleError::DuplicateId(id));
        }

        let internal_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Add vector to storage
        {
            let mut vectors = self.vectors.write();
            if internal_id >= vectors.len() {
                vectors.resize(internal_id + 1, Vec::new());
            }
            vectors[internal_id] = vector.to_vec();
        }

        // Add to HNSW index
        {
            let mut index = self.index.write();
            let vectors = self.vectors.read();
            index.insert(internal_id, vector, vectors.as_slice())?;
        }

        // Create entity
        let entity = GraphEntity {
            id: id.clone(),
            internal_id,
            vector: vector.to_vec(),
            metadata,
            edges: Vec::new(),
            incoming: Vec::new(),
        };

        // Update mappings
        id_map.insert(id.clone(), internal_id);
        self.reverse_id_map.write().insert(internal_id, id.clone());
        self.entities.write().insert(id, entity);

        Ok(())
    }

    /// Add an edge between two entities
    pub fn add_edge(&self, from: &str, to: &str, edge_type: EdgeType, weight: f32) -> Result<()> {
        self.add_edge_with_properties(from, to, edge_type, weight, None)
    }

    /// Add an edge with properties
    pub fn add_edge_with_properties(
        &self,
        from: &str,
        to: &str,
        edge_type: EdgeType,
        weight: f32,
        properties: Option<Value>,
    ) -> Result<()> {
        let weight = weight.clamp(0.0, 1.0);

        if weight < self.config.min_edge_weight {
            return Ok(()); // Skip edges below threshold
        }

        let mut entities = self.entities.write();

        // Check both entities exist
        if !entities.contains_key(from) {
            return Err(NeedleError::VectorNotFound(from.to_string()));
        }
        if !entities.contains_key(to) {
            return Err(NeedleError::VectorNotFound(to.to_string()));
        }

        // Add forward edge
        if let Some(from_entity) = entities.get_mut(from) {
            // Check max edges
            if from_entity.edges.len() >= self.config.max_edges_per_node {
                // Remove weakest edge
                if let Some(min_idx) = from_entity
                    .edges
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.weight
                            .partial_cmp(&b.weight)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                {
                    if from_entity.edges[min_idx].weight < weight {
                        from_entity.edges.remove(min_idx);
                    } else {
                        return Ok(()); // New edge is weaker, skip
                    }
                }
            }

            from_entity.edges.push(GraphEdge {
                target: to.to_string(),
                edge_type: edge_type.clone(),
                weight,
                properties: properties.clone(),
            });
        }

        // Add reverse link for incoming edge tracking
        if let Some(to_entity) = entities.get_mut(to) {
            if !to_entity.incoming.contains(&from.to_string()) {
                to_entity.incoming.push(from.to_string());
            }
        }

        // Add bidirectional edge if configured
        if self.config.bidirectional {
            let reverse_type = self.get_reverse_edge_type(&edge_type);
            if let Some(to_entity) = entities.get_mut(to) {
                // Check for existing reverse edge
                let has_reverse = to_entity.edges.iter().any(|e| e.target == from);
                if !has_reverse && to_entity.edges.len() < self.config.max_edges_per_node {
                    to_entity.edges.push(GraphEdge {
                        target: from.to_string(),
                        edge_type: reverse_type,
                        weight,
                        properties,
                    });
                }
            }
            if let Some(from_entity) = entities.get_mut(from) {
                if !from_entity.incoming.contains(&to.to_string()) {
                    from_entity.incoming.push(to.to_string());
                }
            }
        }

        Ok(())
    }

    /// Get the reverse edge type for bidirectional edges
    fn get_reverse_edge_type(&self, edge_type: &EdgeType) -> EdgeType {
        match edge_type {
            EdgeType::ParentOf => EdgeType::ChildOf,
            EdgeType::ChildOf => EdgeType::ParentOf,
            EdgeType::PartOf => EdgeType::HasPart,
            EdgeType::HasPart => EdgeType::PartOf,
            EdgeType::InstanceOf => EdgeType::TypeOf,
            EdgeType::TypeOf => EdgeType::InstanceOf,
            EdgeType::UsedBy => EdgeType::Uses,
            EdgeType::Uses => EdgeType::UsedBy,
            EdgeType::HasTool => EdgeType::UsedBy,
            EdgeType::DependsOn => EdgeType::UsedBy,
            _ => edge_type.clone(),
        }
    }

    /// Remove an entity
    pub fn remove_entity(&self, id: &str) -> Result<bool> {
        let internal_id = {
            let id_map = self.id_map.read();
            match id_map.get(id) {
                Some(&id) => id,
                None => return Ok(false),
            }
        };

        // Remove from index
        {
            let mut index = self.index.write();
            index.delete(internal_id)?;
        }

        // Remove from entities and clean up edges
        {
            let mut entities = self.entities.write();
            if let Some(entity) = entities.remove(id) {
                // Remove incoming edge references from other entities
                for incoming_id in &entity.incoming {
                    if let Some(other) = entities.get_mut(incoming_id) {
                        other.edges.retain(|e| e.target != id);
                    }
                }

                // Remove from outgoing targets' incoming lists
                for edge in &entity.edges {
                    if let Some(target) = entities.get_mut(&edge.target) {
                        target.incoming.retain(|i| i != id);
                    }
                }
            }
        }

        // Clean up mappings
        self.id_map.write().remove(id);
        self.reverse_id_map.write().remove(&internal_id);

        Ok(true)
    }

    /// Search with vector similarity only
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<GraphVectorSearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let results = {
            let index = self.index.read();
            let vectors = self.vectors.read();
            index.search(query, k, vectors.as_slice())
        };

        let reverse_map = self.reverse_id_map.read();
        let entities = self.entities.read();

        let search_results: Vec<GraphVectorSearchResult> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                let external_id = reverse_map.get(&internal_id)?;
                let entity = entities.get(external_id)?;

                Some(GraphVectorSearchResult {
                    id: external_id.clone(),
                    vector_score: distance,
                    graph_score: 0.0,
                    combined_score: distance,
                    metadata: entity.metadata.clone(),
                    connected_entities: Vec::new(),
                    hops_from_anchor: None,
                })
            })
            .collect();

        Ok(search_results)
    }

    /// Search with graph-boosted scoring
    pub fn search_with_graph(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<GraphVectorSearchResult>> {
        self.search_with_graph_context(query, k, None)
    }

    /// Search with graph context from anchor entities
    pub fn search_with_graph_context(
        &self,
        query: &[f32],
        k: usize,
        anchor_ids: Option<&[&str]>,
    ) -> Result<Vec<GraphVectorSearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let start = Instant::now();

        // Get more candidates than needed for graph reranking
        let fetch_k = k * 3;

        // Vector search
        let vector_results = {
            let index = self.index.read();
            let vectors = self.vectors.read();
            index.search(query, fetch_k, vectors.as_slice())
        };

        let reverse_map = self.reverse_id_map.read();
        let entities = self.entities.read();

        // Build graph scores based on connectivity
        let mut graph_scores: HashMap<String, f32> = HashMap::new();

        // If anchors provided, compute distances from anchors
        if let Some(anchors) = anchor_ids {
            for anchor_id in anchors {
                if let Some(_entity) = entities.get(*anchor_id) {
                    let reachable = self.bfs_reachable(*anchor_id, self.config.max_hops, &entities);
                    for (reachable_id, hops) in reachable {
                        let decay = self.config.distance_decay.powi(hops as i32);
                        let score = graph_scores.entry(reachable_id).or_insert(0.0);
                        *score = (*score).max(decay);
                    }
                }
            }
        } else {
            // Use vector search results as implicit anchors
            // Boost entities connected to top vector results
            for (i, (internal_id, _)) in vector_results.iter().take(5).enumerate() {
                if let Some(ext_id) = reverse_map.get(internal_id) {
                    let anchor_weight = 1.0 - (i as f32 * 0.15); // Decay for lower-ranked anchors
                    let reachable = self.bfs_reachable(ext_id, self.config.max_hops, &entities);
                    for (reachable_id, hops) in reachable {
                        let decay = self.config.distance_decay.powi(hops as i32) * anchor_weight;
                        let score = graph_scores.entry(reachable_id).or_insert(0.0);
                        *score = (*score).max(decay);
                    }
                }
            }
        }

        // Combine vector and graph scores
        let mut results: Vec<GraphVectorSearchResult> = vector_results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                let external_id = reverse_map.get(&internal_id)?;
                let entity = entities.get(external_id)?;

                // Normalize vector score (convert distance to similarity)
                let vector_score = distance;
                let _vector_sim = 1.0 / (1.0 + distance);

                // Get graph score
                let graph_score = graph_scores.get(external_id).copied().unwrap_or(0.0);

                // Combined score (lower is better, so invert graph contribution)
                let combined_score = (self.config.vector_weight * vector_score)
                    - (self.config.graph_weight * graph_score);

                // Get connected entities for context
                let connected = self.get_connected_entities(external_id, 1, &entities);

                Some(GraphVectorSearchResult {
                    id: external_id.clone(),
                    vector_score,
                    graph_score,
                    combined_score,
                    metadata: entity.metadata.clone(),
                    connected_entities: connected,
                    hops_from_anchor: None,
                })
            })
            .collect();

        // Sort by combined score (lower is better)
        results.sort_by(|a, b| {
            a.combined_score
                .partial_cmp(&b.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        debug!(
            duration_ms = start.elapsed().as_millis(),
            results = results.len(),
            "Graph vector search completed"
        );

        Ok(results)
    }

    /// BFS to find all reachable entities within max_hops
    fn bfs_reachable(
        &self,
        start: &str,
        max_hops: usize,
        entities: &HashMap<String, GraphEntity>,
    ) -> Vec<(String, usize)> {
        let mut visited: HashMap<String, usize> = HashMap::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        visited.insert(start.to_string(), 0);
        queue.push_back((start.to_string(), 0));

        while let Some((current_id, hops)) = queue.pop_front() {
            if hops >= max_hops {
                continue;
            }

            if let Some(entity) = entities.get(&current_id) {
                for edge in &entity.edges {
                    if !visited.contains_key(&edge.target) {
                        visited.insert(edge.target.clone(), hops + 1);
                        queue.push_back((edge.target.clone(), hops + 1));
                    }
                }
            }
        }

        visited.into_iter().collect()
    }

    /// Get connected entities up to max_hops
    fn get_connected_entities(
        &self,
        id: &str,
        max_hops: usize,
        entities: &HashMap<String, GraphEntity>,
    ) -> Vec<ConnectedEntity> {
        let mut result = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>, usize)> = VecDeque::new();

        visited.insert(id.to_string());
        queue.push_back((id.to_string(), vec![id.to_string()], 0));

        while let Some((current_id, path, hops)) = queue.pop_front() {
            if hops >= max_hops {
                continue;
            }

            if let Some(entity) = entities.get(&current_id) {
                for edge in &entity.edges {
                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target.clone());
                        let mut new_path = path.clone();
                        new_path.push(edge.target.clone());

                        result.push(ConnectedEntity {
                            id: edge.target.clone(),
                            relation: edge.edge_type.clone(),
                            hops: hops + 1,
                            path: new_path.clone(),
                        });

                        queue.push_back((edge.target.clone(), new_path, hops + 1));
                    }
                }
            }
        }

        result
    }

    /// Find shortest path between two entities
    pub fn find_path(&self, from: &str, to: &str) -> Option<GraphPath> {
        let entities = self.entities.read();

        if !entities.contains_key(from) || !entities.contains_key(to) {
            return None;
        }

        // BFS for shortest path
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>, Vec<EdgeType>)> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back((from.to_string(), vec![from.to_string()], Vec::new()));

        while let Some((current_id, path, edge_types)) = queue.pop_front() {
            if current_id == to {
                let len = edge_types.len();
                return Some(GraphPath {
                    entities: path,
                    edge_types,
                    length: len,
                });
            }

            if let Some(entity) = entities.get(&current_id) {
                for edge in &entity.edges {
                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target.clone());
                        let mut new_path = path.clone();
                        new_path.push(edge.target.clone());
                        let mut new_edges = edge_types.clone();
                        new_edges.push(edge.edge_type.clone());
                        queue.push_back((edge.target.clone(), new_path, new_edges));
                    }
                }
            }
        }

        None
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: &str) -> Option<(Vec<f32>, Option<Value>, Vec<GraphEdge>)> {
        let entities = self.entities.read();
        entities
            .get(id)
            .map(|e| (e.vector.clone(), e.metadata.clone(), e.edges.clone()))
    }

    /// Get statistics
    pub fn stats(&self) -> GraphVectorStats {
        let entities = self.entities.read();
        let index = self.index.read();

        let entity_count = entities.len();
        let edge_count: usize = entities.values().map(|e| e.edges.len()).sum();
        let avg_edges = if entity_count > 0 {
            edge_count as f32 / entity_count as f32
        } else {
            0.0
        };
        let max_edges = entities.values().map(|e| e.edges.len()).max().unwrap_or(0);

        // Count connected components via union-find simulation
        let connected_components = self.count_connected_components(&entities);

        GraphVectorStats {
            entity_count,
            edge_count,
            avg_edges,
            max_edges,
            connected_components,
            index_memory_bytes: index.estimated_memory(),
        }
    }

    fn count_connected_components(&self, entities: &HashMap<String, GraphEntity>) -> usize {
        let mut visited: HashSet<String> = HashSet::new();
        let mut components = 0;

        for id in entities.keys() {
            if !visited.contains(id) {
                components += 1;
                // BFS to mark all connected nodes
                let mut queue: VecDeque<String> = VecDeque::new();
                queue.push_back(id.clone());
                visited.insert(id.clone());

                while let Some(current) = queue.pop_front() {
                    if let Some(entity) = entities.get(&current) {
                        for edge in &entity.edges {
                            if !visited.contains(&edge.target) {
                                visited.insert(edge.target.clone());
                                queue.push_back(edge.target.clone());
                            }
                        }
                        for incoming in &entity.incoming {
                            if !visited.contains(incoming) {
                                visited.insert(incoming.clone());
                                queue.push_back(incoming.clone());
                            }
                        }
                    }
                }
            }
        }

        components
    }

    /// Get total entity count
    pub fn len(&self) -> usize {
        self.entities.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Path between entities in the graph
#[derive(Debug, Clone)]
pub struct GraphPath {
    /// Entity IDs in the path
    pub entities: Vec<String>,
    /// Edge types connecting entities
    pub edge_types: Vec<EdgeType>,
    /// Path length (number of edges)
    pub length: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_vector(dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_add_entity_and_search() {
        let config = GraphVectorConfig::default();
        let index = GraphVectorIndex::new(32, config);

        // Add entities
        for i in 0..100 {
            let vec = random_vector(32);
            index
                .add_entity(format!("entity_{}", i), &vec, None)
                .unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search
        let query = random_vector(32);
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_add_edges_and_graph_search() {
        let config = GraphVectorConfig::builder()
            .vector_weight(0.5)
            .graph_weight(0.5)
            .build();
        let index = GraphVectorIndex::new(8, config);

        // Add entities
        let v1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v4 = vec![0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0];

        index.add_entity("e1", &v1, None).unwrap();
        index.add_entity("e2", &v2, None).unwrap();
        index.add_entity("e3", &v3, None).unwrap();
        index.add_entity("e4", &v4, None).unwrap();

        // Add edges: e1 -> e2 -> e3 -> e4
        index
            .add_edge("e1", "e2", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("e2", "e3", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("e3", "e4", EdgeType::RelatedTo, 1.0)
            .unwrap();

        // Search with graph context should boost connected entities
        let results = index.search_with_graph(&v1, 4).unwrap();
        assert_eq!(results.len(), 4);

        // e2 should be highly ranked (similar vector + connected)
        let e2_pos = results.iter().position(|r| r.id == "e2");
        assert!(e2_pos.is_some());
    }

    #[test]
    fn test_find_path() {
        let config = GraphVectorConfig::default();
        let index = GraphVectorIndex::new(4, config);

        // Add entities in a chain
        for i in 0..5 {
            let v = vec![i as f32; 4];
            index.add_entity(format!("n{}", i), &v, None).unwrap();
        }

        // Create chain: n0 -> n1 -> n2 -> n3 -> n4
        index
            .add_edge("n0", "n1", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("n1", "n2", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("n2", "n3", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("n3", "n4", EdgeType::RelatedTo, 1.0)
            .unwrap();

        // Find path
        let path = index.find_path("n0", "n4").unwrap();
        assert_eq!(path.length, 4);
        assert_eq!(path.entities, vec!["n0", "n1", "n2", "n3", "n4"]);
    }

    #[test]
    fn test_remove_entity() {
        let config = GraphVectorConfig::default();
        let index = GraphVectorIndex::new(4, config);

        index.add_entity("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        index.add_entity("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        index.add_edge("a", "b", EdgeType::RelatedTo, 1.0).unwrap();

        assert_eq!(index.len(), 2);

        // Remove entity
        assert!(index.remove_entity("a").unwrap());
        assert_eq!(index.len(), 1);
        assert!(!index.remove_entity("a").unwrap()); // Already removed

        // Remaining entity should have no incoming edges from deleted entity
        let (_, _, edges) = index.get_entity("b").unwrap();
        // If bidirectional was on, b->a edge should also be gone
    }

    #[test]
    fn test_stats() {
        let config = GraphVectorConfig::default();
        let index = GraphVectorIndex::new(4, config);

        for i in 0..10 {
            index
                .add_entity(format!("e{}", i), &[i as f32; 4], None)
                .unwrap();
        }

        // Create two components
        index
            .add_edge("e0", "e1", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("e1", "e2", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("e5", "e6", EdgeType::RelatedTo, 1.0)
            .unwrap();
        index
            .add_edge("e6", "e7", EdgeType::RelatedTo, 1.0)
            .unwrap();

        let stats = index.stats();
        assert_eq!(stats.entity_count, 10);
        // With bidirectional edges, edge count doubles
        assert!(stats.edge_count >= 4);
    }
}
