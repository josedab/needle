//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! GraphRAG Native Index
//!
//! First-class knowledge graph construction from vectors and metadata with
//! entity extraction, multi-hop traversal, and graph-enhanced retrieval.
//!
//! Builds upon the existing `graphrag` module by adding:
//! - **Auto entity extraction** from text metadata
//! - **Graph-enhanced vector retrieval** that boosts results connected in the graph
//! - **Multi-hop reasoning** with configurable depth and path scoring
//! - **Community detection** for contextual chunking
//! - **Unified query interface** combining vector + graph signals
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::graphrag_index::*;
//!
//! let config = GraphRagConfig::default();
//! let mut index = GraphRagIndex::new(128, config);
//!
//! // Insert documents with automatic entity extraction
//! index.insert_document("doc1", &vec![0.1f32; 128],
//!     "Machine learning uses neural networks for pattern recognition."
//! ).unwrap();
//!
//! // Graph-enhanced search
//! let results = index.search_with_graph(&vec![0.1f32; 128], 10, 2).unwrap();
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::SearchResult;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};

// ---------------------------------------------------------------------------
// Entity Types
// ---------------------------------------------------------------------------

/// Category of an extracted entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityCategory {
    Person,
    Organization,
    Location,
    Concept,
    Technology,
    Event,
    Date,
    Metric,
    Custom(String),
}

/// An entity extracted from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub category: EntityCategory,
    pub confidence: f32,
    pub source_doc_id: String,
    pub span: Option<(usize, usize)>,
}

/// A relationship between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub weight: f32,
    pub source_doc_id: String,
}

// ---------------------------------------------------------------------------
// Graph Storage
// ---------------------------------------------------------------------------

/// A node in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub category: EntityCategory,
    pub embedding: Option<Vec<f32>>,
    pub properties: HashMap<String, Value>,
    pub document_ids: HashSet<String>,
    pub community_id: Option<u32>,
}

/// An edge in the knowledge graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub weight: f32,
}

/// The in-memory knowledge graph.
pub struct KnowledgeGraphStore {
    nodes: HashMap<String, GraphNode>,
    edges: Vec<GraphEdge>,
    adjacency: HashMap<String, Vec<usize>>, // node_id → edge indices
    communities: HashMap<u32, Vec<String>>,
    next_community_id: u32,
}

impl KnowledgeGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            communities: HashMap::new(),
            next_community_id: 0,
        }
    }

    /// Add or update a node.
    pub fn upsert_node(&mut self, id: &str, category: EntityCategory, doc_id: &str) {
        let node = self.nodes.entry(id.to_string()).or_insert_with(|| GraphNode {
            id: id.to_string(),
            category,
            embedding: None,
            properties: HashMap::new(),
            document_ids: HashSet::new(),
            community_id: None,
        });
        node.document_ids.insert(doc_id.to_string());
    }

    /// Add an edge.
    pub fn add_edge(&mut self, source: &str, target: &str, relation_type: &str, weight: f32) {
        let edge_idx = self.edges.len();
        self.edges.push(GraphEdge {
            source: source.to_string(),
            target: target.to_string(),
            relation_type: relation_type.to_string(),
            weight,
        });
        self.adjacency
            .entry(source.to_string())
            .or_default()
            .push(edge_idx);
        self.adjacency
            .entry(target.to_string())
            .or_default()
            .push(edge_idx);
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.nodes.get(id)
    }

    /// Get all edges connected to a node.
    pub fn get_edges(&self, node_id: &str) -> Vec<&GraphEdge> {
        self.adjacency
            .get(node_id)
            .map(|indices| indices.iter().filter_map(|&i| self.edges.get(i)).collect())
            .unwrap_or_default()
    }

    /// Get all neighbor node IDs.
    pub fn neighbors(&self, node_id: &str) -> Vec<String> {
        let edges = self.get_edges(node_id);
        let mut neighbors = HashSet::new();
        for edge in edges {
            if edge.source == node_id {
                neighbors.insert(edge.target.clone());
            } else {
                neighbors.insert(edge.source.clone());
            }
        }
        neighbors.into_iter().collect()
    }

    /// Multi-hop traversal from a starting node.
    pub fn traverse(&self, start: &str, max_depth: usize) -> Vec<TraversalPath> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, Vec<String>, f32)> = VecDeque::new();

        queue.push_back((start.to_string(), vec![start.to_string()], 1.0));
        visited.insert(start.to_string());

        while let Some((current, path, score)) = queue.pop_front() {
            if path.len() > 1 {
                paths.push(TraversalPath {
                    nodes: path.clone(),
                    total_weight: score,
                    depth: path.len() - 1,
                });
            }

            if path.len() > max_depth {
                continue;
            }

            for edge in self.get_edges(&current) {
                let next = if edge.source == current {
                    &edge.target
                } else {
                    &edge.source
                };

                if !visited.contains(next) {
                    visited.insert(next.clone());
                    let mut new_path = path.clone();
                    new_path.push(next.clone());
                    queue.push_back((next.clone(), new_path, score * edge.weight));
                }
            }
        }

        paths
    }

    /// Simple community detection using connected components with edge-weight thresholding.
    pub fn detect_communities(&mut self, min_weight: f32) -> usize {
        let mut visited = HashSet::new();
        let mut community_count = 0;

        for node_id in self.nodes.keys().cloned().collect::<Vec<_>>() {
            if visited.contains(&node_id) {
                continue;
            }

            let community_id = self.next_community_id;
            self.next_community_id += 1;
            community_count += 1;

            // BFS to find connected component
            let mut queue = VecDeque::new();
            queue.push_back(node_id.clone());
            visited.insert(node_id.clone());
            let mut members = Vec::new();

            while let Some(current) = queue.pop_front() {
                if let Some(node) = self.nodes.get_mut(&current) {
                    node.community_id = Some(community_id);
                }
                members.push(current.clone());

                for edge in self.get_edges(&current) {
                    if edge.weight < min_weight {
                        continue;
                    }
                    let next = if edge.source == current {
                        &edge.target
                    } else {
                        &edge.source
                    };
                    if !visited.contains(next) {
                        visited.insert(next.clone());
                        queue.push_back(next.clone());
                    }
                }
            }

            self.communities.insert(community_id, members);
        }

        community_count
    }

    /// Get all members of a community.
    pub fn community_members(&self, community_id: u32) -> Vec<&str> {
        self.communities
            .get(&community_id)
            .map(|members| members.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of detected communities.
    pub fn community_count(&self) -> usize {
        self.communities.len()
    }
}

impl Default for KnowledgeGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

/// A path discovered during graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPath {
    pub nodes: Vec<String>,
    pub total_weight: f32,
    pub depth: usize,
}

// ---------------------------------------------------------------------------
// Entity Extractor
// ---------------------------------------------------------------------------

/// Configuration for entity extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionConfig {
    /// Minimum confidence to accept an entity.
    pub min_confidence: f32,
    /// Maximum entities per document.
    pub max_entities_per_doc: usize,
    /// Custom entity patterns (regex-like simple patterns).
    pub custom_patterns: Vec<EntityPattern>,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_entities_per_doc: 50,
            custom_patterns: Vec::new(),
        }
    }
}

/// A pattern for custom entity extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityPattern {
    pub name: String,
    pub keywords: Vec<String>,
    pub category: EntityCategory,
}

/// Simple rule-based entity extractor. A production implementation would use
/// an NER model, but this provides useful extraction for structured text.
pub struct EntityExtractor {
    config: ExtractionConfig,
    concept_keywords: HashSet<String>,
    tech_keywords: HashSet<String>,
}

impl EntityExtractor {
    pub fn new(config: ExtractionConfig) -> Self {
        let concepts: HashSet<String> = [
            "algorithm", "model", "network", "learning", "data", "training",
            "inference", "optimization", "search", "retrieval", "embedding",
            "classification", "clustering", "regression", "attention",
            "transformer", "vector", "index", "query", "database",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let tech: HashSet<String> = [
            "python", "rust", "javascript", "pytorch", "tensorflow", "numpy",
            "cuda", "gpu", "cpu", "onnx", "docker", "kubernetes", "aws",
            "gcp", "azure", "linux", "macos", "windows", "sql", "nosql",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        Self {
            config,
            concept_keywords: concepts,
            tech_keywords: tech,
        }
    }

    /// Extract entities from text.
    pub fn extract(&self, text: &str, doc_id: &str) -> Vec<ExtractedEntity> {
        let mut entities = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();

        for (i, &word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            let clean = lower.trim_matches(|c: char| !c.is_alphanumeric());

            if clean.is_empty() {
                continue;
            }

            // Check concept keywords
            if self.concept_keywords.contains(clean) {
                entities.push(ExtractedEntity {
                    name: clean.to_string(),
                    category: EntityCategory::Concept,
                    confidence: 0.7,
                    source_doc_id: doc_id.to_string(),
                    span: Some((i, i + 1)),
                });
            }

            // Check technology keywords
            if self.tech_keywords.contains(clean) {
                entities.push(ExtractedEntity {
                    name: clean.to_string(),
                    category: EntityCategory::Technology,
                    confidence: 0.8,
                    source_doc_id: doc_id.to_string(),
                    span: Some((i, i + 1)),
                });
            }

            // Capitalised words might be proper nouns (names/orgs)
            if word.len() > 1
                && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                && !word.chars().all(|c| c.is_uppercase())
                && i > 0
            {
                entities.push(ExtractedEntity {
                    name: word.to_string(),
                    category: EntityCategory::Person,
                    confidence: 0.4,
                    source_doc_id: doc_id.to_string(),
                    span: Some((i, i + 1)),
                });
            }

            // Check custom patterns
            for pattern in &self.config.custom_patterns {
                if pattern.keywords.iter().any(|kw| clean == kw.to_lowercase()) {
                    entities.push(ExtractedEntity {
                        name: clean.to_string(),
                        category: pattern.category.clone(),
                        confidence: 0.9,
                        source_doc_id: doc_id.to_string(),
                        span: Some((i, i + 1)),
                    });
                }
            }
        }

        // Filter by confidence and limit
        entities.retain(|e| e.confidence >= self.config.min_confidence);
        entities.truncate(self.config.max_entities_per_doc);

        // Deduplicate by name
        let mut seen = HashSet::new();
        entities.retain(|e| seen.insert(e.name.clone()));

        entities
    }

    /// Extract co-occurrence relations between entities in the same document.
    pub fn extract_relations(
        &self,
        entities: &[ExtractedEntity],
        doc_id: &str,
    ) -> Vec<ExtractedRelation> {
        let mut relations = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let e1 = &entities[i];
                let e2 = &entities[j];

                // Co-occurrence weight decreases with distance
                let distance = match (e1.span, e2.span) {
                    (Some((s1, _)), Some((s2, _))) => (s2 as f32 - s1 as f32).abs(),
                    _ => 10.0,
                };
                let weight = 1.0 / (1.0 + distance / 5.0);

                if weight > 0.2 {
                    relations.push(ExtractedRelation {
                        source: e1.name.clone(),
                        target: e2.name.clone(),
                        relation_type: "co_occurs".to_string(),
                        weight,
                        source_doc_id: doc_id.to_string(),
                    });
                }
            }
        }

        relations
    }
}

// ---------------------------------------------------------------------------
// GraphRAG Config
// ---------------------------------------------------------------------------

/// Configuration for the GraphRAG index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagConfig {
    /// HNSW configuration for the vector index.
    pub hnsw_config: HnswConfig,
    /// Distance function.
    pub distance: DistanceFunction,
    /// Entity extraction configuration.
    pub extraction_config: ExtractionConfig,
    /// Weight of vector similarity in the final score (0-1).
    pub vector_weight: f32,
    /// Weight of graph connectivity in the final score (0-1).
    pub graph_weight: f32,
    /// Default max traversal depth for graph-enhanced search.
    pub default_max_depth: usize,
    /// Minimum edge weight for community detection.
    pub community_min_weight: f32,
}

impl Default for GraphRagConfig {
    fn default() -> Self {
        Self {
            hnsw_config: HnswConfig::default(),
            distance: DistanceFunction::Cosine,
            extraction_config: ExtractionConfig::default(),
            vector_weight: 0.7,
            graph_weight: 0.3,
            default_max_depth: 2,
            community_min_weight: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphRAG Index
// ---------------------------------------------------------------------------

/// Result from a graph-enhanced search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphRagResult {
    pub id: String,
    pub vector_distance: f32,
    pub graph_score: f32,
    pub combined_score: f32,
    pub metadata: Option<Value>,
    pub related_entities: Vec<String>,
    pub reasoning_paths: Vec<TraversalPath>,
}

/// Combined vector + knowledge graph index for GraphRAG retrieval.
pub struct GraphRagIndex {
    dimension: usize,
    config: GraphRagConfig,
    hnsw: RwLock<HnswIndex>,
    graph: RwLock<KnowledgeGraphStore>,
    extractor: EntityExtractor,
    vectors: RwLock<Vec<(String, Vec<f32>, Option<Value>)>>,
    doc_entities: RwLock<HashMap<String, Vec<String>>>, // doc_id → entity names
}

impl GraphRagIndex {
    /// Create a new GraphRAG index.
    pub fn new(dimension: usize, config: GraphRagConfig) -> Self {
        let hnsw = HnswIndex::new(config.hnsw_config.clone(), config.distance);
        let graph = KnowledgeGraphStore::new();
        let extractor = EntityExtractor::new(config.extraction_config.clone());

        Self {
            dimension,
            config,
            hnsw: RwLock::new(hnsw),
            graph: RwLock::new(graph),
            extractor,
            vectors: RwLock::new(Vec::new()),
            doc_entities: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a document with automatic entity extraction.
    pub fn insert_document(
        &self,
        id: &str,
        vector: &[f32],
        text: &str,
    ) -> Result<Vec<ExtractedEntity>> {
        self.insert_with_metadata(id, vector, text, None)
    }

    /// Insert a document with metadata and entity extraction.
    pub fn insert_with_metadata(
        &self,
        id: &str,
        vector: &[f32],
        text: &str,
        metadata: Option<Value>,
    ) -> Result<Vec<ExtractedEntity>> {
        if vector.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        // Insert vector
        let idx = {
            let mut vecs = self.vectors.write();
            let idx = vecs.len();
            let mut meta = metadata.unwrap_or(Value::Null);
            if meta.is_null() {
                meta = serde_json::json!({});
            }
            if let Value::Object(ref mut map) = meta {
                map.insert("_text".to_string(), Value::String(text.to_string()));
            }
            vecs.push((id.to_string(), vector.to_vec(), Some(meta)));
            idx
        };

        // Insert into HNSW
        {
            let vecs = self.vectors.read();
            let all_vecs: Vec<Vec<f32>> = vecs.iter().map(|(_, v, _)| v.clone()).collect();
            self.hnsw.write().insert(idx, vector, &all_vecs).map_err(|_| {
                NeedleError::Index("HNSW insert failed".into())
            })?;
        }

        // Extract entities and relations
        let entities = self.extractor.extract(text, id);
        let relations = self.extractor.extract_relations(&entities, id);

        // Add to knowledge graph
        {
            let mut graph = self.graph.write();
            for entity in &entities {
                graph.upsert_node(&entity.name, entity.category.clone(), id);
            }
            for relation in &relations {
                graph.add_edge(
                    &relation.source,
                    &relation.target,
                    &relation.relation_type,
                    relation.weight,
                );
            }
        }

        // Track doc → entity mapping
        {
            let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
            self.doc_entities
                .write()
                .insert(id.to_string(), entity_names);
        }

        Ok(entities)
    }

    /// Standard vector search without graph enhancement.
    pub fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let vecs = self.vectors.read();
        let all_vecs: Vec<Vec<f32>> = vecs.iter().map(|(_, v, _)| v.clone()).collect();
        let raw = self.hnsw.read().search(query, k, &all_vecs);

        Ok(raw
            .into_iter()
            .filter_map(|(idx, dist)| {
                vecs.get(idx).map(|(id, _, meta)| SearchResult {
                    id: id.clone(),
                    distance: dist,
                    metadata: meta.clone(),
                })
            })
            .collect())
    }

    /// Graph-enhanced search combining vector similarity with graph connectivity.
    pub fn search_with_graph(
        &self,
        query: &[f32],
        k: usize,
        max_depth: usize,
    ) -> Result<Vec<GraphRagResult>> {
        // Phase 1: Vector search (over-fetch)
        let vector_results = self.vector_search(query, k * 3)?;

        // Phase 2: Graph enhancement
        let graph = self.graph.read();
        let doc_entities = self.doc_entities.read();
        let mut scored_results = Vec::new();

        for vr in &vector_results {
            let entities = doc_entities
                .get(&vr.id)
                .cloned()
                .unwrap_or_default();

            // Compute graph score: how connected is this document to other results?
            let mut graph_score = 0.0f32;
            let mut related = Vec::new();
            let mut paths = Vec::new();

            for entity_name in &entities {
                let traversal = graph.traverse(entity_name, max_depth);
                for path in &traversal {
                    graph_score += path.total_weight * 0.1;
                    paths.push(path.clone());
                }

                let neighbors = graph.neighbors(entity_name);
                related.extend(neighbors);
            }

            related.sort();
            related.dedup();

            // Normalise graph score
            let max_graph_score = (entities.len() as f32 * max_depth as f32).max(1.0);
            graph_score = (graph_score / max_graph_score).min(1.0);

            // Combine scores
            let vector_score = 1.0 - vr.distance.min(1.0);
            let combined = self.config.vector_weight * vector_score
                + self.config.graph_weight * graph_score;

            scored_results.push(GraphRagResult {
                id: vr.id.clone(),
                vector_distance: vr.distance,
                graph_score,
                combined_score: combined,
                metadata: vr.metadata.clone(),
                related_entities: related,
                reasoning_paths: paths,
            });
        }

        // Sort by combined score (descending)
        scored_results.sort_by(|a, b| {
            OrderedFloat(b.combined_score).cmp(&OrderedFloat(a.combined_score))
        });
        scored_results.truncate(k);

        Ok(scored_results)
    }

    /// Run community detection on the knowledge graph.
    pub fn detect_communities(&self) -> usize {
        self.graph
            .write()
            .detect_communities(self.config.community_min_weight)
    }

    /// Get graph statistics.
    pub fn graph_stats(&self) -> GraphStats {
        let graph = self.graph.read();
        GraphStats {
            node_count: graph.node_count(),
            edge_count: graph.edge_count(),
            community_count: graph.community_count(),
            document_count: self.vectors.read().len(),
        }
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }
}

/// Statistics for the GraphRAG index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub community_count: usize,
    pub document_count: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_extraction() {
        let extractor = EntityExtractor::new(ExtractionConfig::default());
        let entities = extractor.extract(
            "Machine learning uses a neural network for training data models.",
            "doc1",
        );
        assert!(!entities.is_empty());
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"learning"));
        assert!(names.contains(&"network"));
    }

    #[test]
    fn test_entity_extraction_tech() {
        let extractor = EntityExtractor::new(ExtractionConfig::default());
        let entities = extractor.extract(
            "We use python and pytorch with cuda for gpu training.",
            "doc1",
        );
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"python"));
        assert!(names.contains(&"pytorch"));
        assert!(names.contains(&"cuda"));
    }

    #[test]
    fn test_relation_extraction() {
        let extractor = EntityExtractor::new(ExtractionConfig::default());
        let entities = extractor.extract("vector search algorithm optimization", "doc1");
        let relations = extractor.extract_relations(&entities, "doc1");
        assert!(!relations.is_empty());
        assert!(relations.iter().all(|r| r.weight > 0.0));
    }

    #[test]
    fn test_knowledge_graph_store() {
        let mut graph = KnowledgeGraphStore::new();
        graph.upsert_node("ml", EntityCategory::Concept, "doc1");
        graph.upsert_node("nn", EntityCategory::Concept, "doc1");
        graph.add_edge("ml", "nn", "uses", 0.8);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.neighbors("ml"), vec!["nn"]);
    }

    #[test]
    fn test_graph_traversal() {
        let mut graph = KnowledgeGraphStore::new();
        graph.upsert_node("a", EntityCategory::Concept, "d1");
        graph.upsert_node("b", EntityCategory::Concept, "d1");
        graph.upsert_node("c", EntityCategory::Concept, "d2");
        graph.add_edge("a", "b", "related", 0.9);
        graph.add_edge("b", "c", "related", 0.8);

        let paths = graph.traverse("a", 2);
        assert!(!paths.is_empty());

        // Should find path a→b and a→b→c
        let depths: Vec<usize> = paths.iter().map(|p| p.depth).collect();
        assert!(depths.contains(&1));
        assert!(depths.contains(&2));
    }

    #[test]
    fn test_community_detection() {
        let mut graph = KnowledgeGraphStore::new();
        // Component 1: a-b-c
        graph.upsert_node("a", EntityCategory::Concept, "d1");
        graph.upsert_node("b", EntityCategory::Concept, "d1");
        graph.upsert_node("c", EntityCategory::Concept, "d1");
        graph.add_edge("a", "b", "r", 0.9);
        graph.add_edge("b", "c", "r", 0.8);

        // Component 2: x-y (disconnected)
        graph.upsert_node("x", EntityCategory::Technology, "d2");
        graph.upsert_node("y", EntityCategory::Technology, "d2");
        graph.add_edge("x", "y", "r", 0.7);

        let num_communities = graph.detect_communities(0.5);
        assert_eq!(num_communities, 2);
        assert_eq!(graph.community_count(), 2);
    }

    #[test]
    fn test_graphrag_index_creation() {
        let index = GraphRagIndex::new(4, GraphRagConfig::default());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_graphrag_insert_and_search() {
        let index = GraphRagIndex::new(4, GraphRagConfig::default());

        let entities = index
            .insert_document(
                "doc1",
                &random_vector(4),
                "Machine learning uses neural network models for data training.",
            )
            .unwrap();
        assert!(!entities.is_empty());

        index
            .insert_document(
                "doc2",
                &random_vector(4),
                "Vector search with algorithm optimization for fast retrieval.",
            )
            .unwrap();

        assert_eq!(index.len(), 2);

        let results = index.vector_search(&random_vector(4), 2).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_graphrag_graph_enhanced_search() {
        let index = GraphRagIndex::new(4, GraphRagConfig::default());

        index
            .insert_document("d1", &[1.0, 0.0, 0.0, 0.0], "learning algorithm model")
            .unwrap();
        index
            .insert_document("d2", &[0.9, 0.1, 0.0, 0.0], "learning optimization training")
            .unwrap();
        index
            .insert_document("d3", &[0.0, 0.0, 1.0, 0.0], "python pytorch cuda")
            .unwrap();

        let results = index
            .search_with_graph(&[1.0, 0.0, 0.0, 0.0], 3, 2)
            .unwrap();
        assert!(!results.is_empty());

        // Results should have combined scores
        for result in &results {
            assert!(result.combined_score >= 0.0);
        }
    }

    #[test]
    fn test_graphrag_dimension_check() {
        let index = GraphRagIndex::new(4, GraphRagConfig::default());
        let result = index.insert_document("bad", &[1.0, 2.0], "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_graphrag_stats() {
        let index = GraphRagIndex::new(4, GraphRagConfig::default());
        index
            .insert_document("d1", &random_vector(4), "machine learning algorithm")
            .unwrap();

        let stats = index.graph_stats();
        assert!(stats.node_count > 0);
        assert_eq!(stats.document_count, 1);
    }

    #[test]
    fn test_graphrag_community_detection() {
        let index = GraphRagIndex::new(4, GraphRagConfig::default());
        index
            .insert_document("d1", &random_vector(4), "machine learning model training")
            .unwrap();
        index
            .insert_document("d2", &random_vector(4), "python pytorch cuda gpu")
            .unwrap();

        let communities = index.detect_communities();
        assert!(communities > 0);
    }

    #[test]
    fn test_custom_entity_pattern() {
        let config = ExtractionConfig {
            custom_patterns: vec![EntityPattern {
                name: "custom".into(),
                keywords: vec!["needle".into(), "haystack".into()],
                category: EntityCategory::Custom("VectorDB".into()),
            }],
            ..Default::default()
        };
        let extractor = EntityExtractor::new(config);
        let entities = extractor.extract("Using needle for vector search", "doc1");
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"needle"));
    }

    #[test]
    fn test_traversal_path_serialization() {
        let path = TraversalPath {
            nodes: vec!["a".into(), "b".into(), "c".into()],
            total_weight: 0.72,
            depth: 2,
        };
        let json = serde_json::to_string(&path).unwrap();
        let decoded: TraversalPath = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.depth, 2);
    }

    #[test]
    fn test_entity_category_variants() {
        let categories = vec![
            EntityCategory::Person,
            EntityCategory::Organization,
            EntityCategory::Location,
            EntityCategory::Concept,
            EntityCategory::Technology,
            EntityCategory::Event,
            EntityCategory::Date,
            EntityCategory::Metric,
            EntityCategory::Custom("test".into()),
        ];
        for cat in &categories {
            let json = serde_json::to_string(cat).unwrap();
            let _: EntityCategory = serde_json::from_str(&json).unwrap();
        }
    }
}
