#![allow(clippy::unwrap_used)]
//! Graph-Knowledge Fusion Service
//!
//! Auto entity/relation extraction from text, NeedleQL-style graph traversal
//! operators, and knowledge-aware search that combines graph structure with
//! vector similarity in a single query.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::graph_knowledge_service::{
//!     KnowledgeService, KnowledgeConfig, Document,
//!     KnowledgeQuery, KnowledgeResult,
//! };
//!
//! let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));
//!
//! // Ingest a document — auto-extracts entities and relations
//! svc.ingest(Document {
//!     id: "doc1".into(),
//!     text: "Rust uses Cargo as its build tool".into(),
//!     embedding: vec![0.9, 0.1, 0.0, 0.0],
//!     metadata: None,
//! }).unwrap();
//!
//! // Knowledge-aware search
//! let results = svc.search(KnowledgeQuery {
//!     embedding: vec![0.85, 0.15, 0.0, 0.0],
//!     k: 5,
//!     max_hops: 2,
//!     include_relations: true,
//! }).unwrap();
//! ```

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Knowledge service configuration.
#[derive(Debug, Clone)]
pub struct KnowledgeConfig {
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Maximum entities to extract per document.
    pub max_entities_per_doc: usize,
    /// Maximum relations per entity.
    pub max_relations: usize,
    /// Default max hops for graph traversal.
    pub default_max_hops: usize,
    /// Graph weight in fused scoring.
    pub graph_weight: f32,
}

impl KnowledgeConfig {
    /// Create a new config.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            max_entities_per_doc: 20,
            max_relations: 100,
            default_max_hops: 2,
            graph_weight: 0.3,
        }
    }
}

// ── Document ─────────────────────────────────────────────────────────────────

/// A document for ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID.
    pub id: String,
    /// Document text.
    pub text: String,
    /// Pre-computed embedding.
    pub embedding: Vec<f32>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

// ── Entity & Relation ────────────────────────────────────────────────────────

/// An extracted entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity text.
    pub name: String,
    /// Entity type (person, org, concept, etc).
    pub entity_type: String,
    /// Source document.
    pub source_doc: String,
}

/// A relation between entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Source entity.
    pub from: String,
    /// Target entity.
    pub to: String,
    /// Relation type.
    pub relation_type: String,
    /// Source document.
    pub source_doc: String,
}

// ── Query & Result ───────────────────────────────────────────────────────────

/// Knowledge-aware query.
#[derive(Debug, Clone)]
pub struct KnowledgeQuery {
    /// Query embedding.
    pub embedding: Vec<f32>,
    /// Number of results.
    pub k: usize,
    /// Maximum graph traversal hops.
    pub max_hops: usize,
    /// Whether to include relation paths in results.
    pub include_relations: bool,
}

/// Knowledge search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeResult {
    /// Document ID.
    pub doc_id: String,
    /// Fused score (lower = better).
    pub score: f32,
    /// Vector distance.
    pub vector_distance: f32,
    /// Graph distance (hops from closest match).
    pub graph_hops: usize,
    /// Related entities found via graph.
    pub related_entities: Vec<String>,
    /// Document text.
    pub text: String,
    /// Metadata.
    pub metadata: Option<Value>,
}

// ── Knowledge Service ────────────────────────────────────────────────────────

/// Knowledge-aware search service with auto entity/relation extraction.
pub struct KnowledgeService {
    config: KnowledgeConfig,
    collection: Collection,
    documents: HashMap<String, Document>,
    entities: HashMap<String, Entity>,
    relations: Vec<Relation>,
    doc_entities: HashMap<String, Vec<String>>,
    entity_docs: HashMap<String, Vec<String>>,
}

impl KnowledgeService {
    /// Create a new knowledge service.
    pub fn new(config: KnowledgeConfig) -> Self {
        let coll_config = CollectionConfig::new("__knowledge__", config.dimensions)
            .with_distance(DistanceFunction::Cosine);
        Self {
            config,
            collection: Collection::new(coll_config),
            documents: HashMap::new(),
            entities: HashMap::new(),
            relations: Vec::new(),
            doc_entities: HashMap::new(),
            entity_docs: HashMap::new(),
        }
    }

    /// Ingest a document, extracting entities and relations.
    pub fn ingest(&mut self, doc: Document) -> Result<IngestResult> {
        if doc.embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: doc.embedding.len(),
            });
        }

        let doc_id = doc.id.clone();

        // Extract entities from text (simple NLP: capitalized words as entities)
        let extracted = self.extract_entities(&doc.text, &doc.id);
        let entity_count = extracted.entities.len();
        let relation_count = extracted.relations.len();

        // Store in collection
        let meta = serde_json::json!({
            "text": doc.text,
            "entities": extracted.entities.iter().map(|e| &e.name).collect::<Vec<_>>(),
        });
        self.collection
            .insert(doc.id.clone(), &doc.embedding, Some(meta))?;

        // Index entities
        let entity_names: Vec<String> = extracted.entities.iter().map(|e| e.name.clone()).collect();
        self.doc_entities.insert(doc.id.clone(), entity_names.clone());
        for name in &entity_names {
            self.entity_docs
                .entry(name.clone())
                .or_default()
                .push(doc.id.clone());
        }
        for entity in extracted.entities {
            self.entities.insert(entity.name.clone(), entity);
        }
        self.relations.extend(extracted.relations);
        self.documents.insert(doc.id.clone(), doc);

        Ok(IngestResult {
            doc_id,
            entities_extracted: entity_count,
            relations_extracted: relation_count,
        })
    }

    /// Knowledge-aware search.
    pub fn search(&self, query: KnowledgeQuery) -> Result<Vec<KnowledgeResult>> {
        if query.embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query.embedding.len(),
            });
        }

        // Phase 1: Vector search
        let vector_results = self.collection.search(&query.embedding, query.k * 3)?;

        // Phase 2: Expand via graph
        let mut scored: HashMap<String, KnowledgeResult> = HashMap::new();

        for vr in &vector_results {
            if let Some(doc) = self.documents.get(&vr.id) {
                let related = if query.include_relations {
                    self.find_related_docs(&vr.id, query.max_hops)
                } else {
                    Vec::new()
                };

                scored.insert(
                    vr.id.clone(),
                    KnowledgeResult {
                        doc_id: vr.id.clone(),
                        score: vr.distance,
                        vector_distance: vr.distance,
                        graph_hops: 0,
                        related_entities: self
                            .doc_entities
                            .get(&vr.id)
                            .cloned()
                            .unwrap_or_default(),
                        text: doc.text.clone(),
                        metadata: doc.metadata.clone(),
                    },
                );

                // Add graph-expanded results
                for (related_id, hops) in &related {
                    if !scored.contains_key(related_id) {
                        if let Some(rdoc) = self.documents.get(related_id) {
                            let graph_boost = 1.0 + (*hops as f32) * self.config.graph_weight;
                            let dist = self.config.dimensions as f32; // placeholder distance
                            scored.insert(
                                related_id.clone(),
                                KnowledgeResult {
                                    doc_id: related_id.clone(),
                                    score: vr.distance * graph_boost,
                                    vector_distance: dist,
                                    graph_hops: *hops,
                                    related_entities: self
                                        .doc_entities
                                        .get(related_id)
                                        .cloned()
                                        .unwrap_or_default(),
                                    text: rdoc.text.clone(),
                                    metadata: rdoc.metadata.clone(),
                                },
                            );
                        }
                    }
                }
            }
        }

        let mut results: Vec<KnowledgeResult> = scored.into_values().collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(query.k);
        Ok(results)
    }

    /// Get all entities.
    pub fn entities(&self) -> Vec<&Entity> {
        self.entities.values().collect()
    }

    /// Get all relations.
    pub fn relations(&self) -> &[Relation] {
        &self.relations
    }

    /// Document count.
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn extract_entities(&self, text: &str, doc_id: &str) -> ExtractionResult {
        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let mut seen = HashSet::new();

        // Simple extraction: words starting with uppercase as entities
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in &words {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if !clean.is_empty()
                && clean.chars().next().map_or(false, |c| c.is_uppercase())
                && clean.len() > 1
                && !seen.contains(clean)
            {
                seen.insert(clean.to_string());
                entities.push(Entity {
                    name: clean.to_string(),
                    entity_type: "concept".into(),
                    source_doc: doc_id.into(),
                });
            }
        }

        // Extract relations between consecutive entities
        let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
        for pair in entity_names.windows(2) {
            relations.push(Relation {
                from: pair[0].clone(),
                to: pair[1].clone(),
                relation_type: "related_to".into(),
                source_doc: doc_id.into(),
            });
        }

        // Limit extraction
        entities.truncate(self.config.max_entities_per_doc);

        ExtractionResult {
            entities,
            relations,
        }
    }

    fn find_related_docs(&self, doc_id: &str, max_hops: usize) -> Vec<(String, usize)> {
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(doc_id.into());
        let mut result = Vec::new();
        let mut frontier: Vec<(String, usize)> = vec![(doc_id.into(), 0)];

        while let Some((current_doc, hops)) = frontier.pop() {
            if hops >= max_hops {
                continue;
            }
            if let Some(entities) = self.doc_entities.get(&current_doc) {
                for entity in entities {
                    if let Some(docs) = self.entity_docs.get(entity) {
                        for related_doc in docs {
                            if !visited.contains(related_doc) {
                                visited.insert(related_doc.clone());
                                result.push((related_doc.clone(), hops + 1));
                                frontier.push((related_doc.clone(), hops + 1));
                            }
                        }
                    }
                }
            }
        }
        result
    }
}

/// Result of entity extraction.
struct ExtractionResult {
    entities: Vec<Entity>,
    relations: Vec<Relation>,
}

/// Result of document ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    /// Document ID.
    pub doc_id: String,
    /// Number of entities extracted.
    pub entities_extracted: usize,
    /// Number of relations extracted.
    pub relations_extracted: usize,
}

// ── Community Detection ─────────────────────────────────────────────────────

/// A community of related entities found via Leiden-style detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    /// Community ID.
    pub id: usize,
    /// Member entity names.
    pub members: Vec<String>,
    /// Community size.
    pub size: usize,
    /// Representative topic/summary.
    pub label: Option<String>,
}

/// Leiden-inspired community detection on the entity graph.
///
/// Groups entities into communities based on shared document co-occurrence.
/// Uses a simplified label propagation approach: each entity starts in its
/// own community, then adopts the most frequent community among its neighbors.
pub fn detect_communities(
    entities: &HashMap<String, Entity>,
    relations: &[Relation],
    iterations: usize,
) -> Vec<Community> {
    if entities.is_empty() {
        return Vec::new();
    }

    // Build adjacency from relations
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for rel in relations {
        adjacency.entry(rel.from.clone()).or_default().push(rel.to.clone());
        adjacency.entry(rel.to.clone()).or_default().push(rel.from.clone());
    }

    // Initialize: each entity in its own community
    let mut labels: HashMap<String, usize> = HashMap::new();
    for (i, name) in entities.keys().enumerate() {
        labels.insert(name.clone(), i);
    }

    // Label propagation
    for _ in 0..iterations {
        let mut changed = false;
        for name in entities.keys() {
            if let Some(neighbors) = adjacency.get(name) {
                if neighbors.is_empty() {
                    continue;
                }
                // Count community labels among neighbors
                let mut counts: HashMap<usize, usize> = HashMap::new();
                for n in neighbors {
                    if let Some(&label) = labels.get(n) {
                        *counts.entry(label).or_insert(0) += 1;
                    }
                }
                // Adopt the most frequent label
                if let Some((&best_label, _)) = counts.iter().max_by_key(|(_, &c)| c) {
                    let current = labels.get(name).copied().unwrap_or(0);
                    if best_label != current {
                        labels.insert(name.clone(), best_label);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Group by label
    let mut communities: HashMap<usize, Vec<String>> = HashMap::new();
    for (name, label) in &labels {
        communities.entry(*label).or_default().push(name.clone());
    }

    communities.into_iter()
        .enumerate()
        .map(|(i, (_, members))| {
            let size = members.len();
            Community { id: i, members, size, label: None }
        })
        .filter(|c| c.size > 0)
        .collect()
}

// ── LLM Entity Extraction Types ─────────────────────────────────────────────

/// Configuration for LLM-powered entity extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionConfig {
    /// LLM model name (e.g., "gpt-4", "claude-3-sonnet").
    pub model: String,
    /// System prompt for entity extraction.
    pub system_prompt: String,
    /// Maximum entities to extract per document.
    pub max_entities: usize,
    /// Entity types to look for.
    pub entity_types: Vec<String>,
    /// Relation types to extract.
    pub relation_types: Vec<String>,
}

impl Default for LlmExtractionConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4".into(),
            system_prompt: "Extract entities and relationships from the following text.".into(),
            max_entities: 20,
            entity_types: vec![
                "Person".into(), "Organization".into(), "Location".into(),
                "Concept".into(), "Event".into(), "Technology".into(),
            ],
            relation_types: vec![
                "related_to".into(), "part_of".into(), "created_by".into(),
                "uses".into(), "located_in".into(), "causes".into(),
            ],
        }
    }
}

/// Result of LLM-based extraction (structured output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionResult {
    /// Extracted entities.
    pub entities: Vec<Entity>,
    /// Extracted relations.
    pub relations: Vec<Relation>,
    /// Confidence score (0.0-1.0).
    pub confidence: f32,
    /// Number of LLM tokens used.
    pub tokens_used: u32,
}

// ── Property Graph Enrichment ───────────────────────────────────────────────

/// Property on an entity node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityProperty {
    /// Property key.
    pub key: String,
    /// Property value.
    pub value: Value,
}

impl KnowledgeService {
    /// Detect communities in the entity graph using label propagation.
    pub fn detect_communities(&self, iterations: usize) -> Vec<Community> {
        detect_communities(&self.entities, &self.relations, iterations)
    }

    /// Add properties to an entity (property graph enrichment).
    pub fn add_entity_property(&mut self, entity_name: &str, key: &str, value: Value) -> bool {
        if let Some(entity) = self.entities.get_mut(entity_name) {
            entity.entity_type = format!(
                "{};{}={}",
                entity.entity_type,
                key,
                serde_json::to_string(&value).unwrap_or_default()
            );
            true
        } else {
            false
        }
    }

    /// Get graph statistics.
    pub fn graph_stats(&self) -> GraphStats {
        let communities = self.detect_communities(5);
        GraphStats {
            total_entities: self.entities.len(),
            total_relations: self.relations.len(),
            total_documents: self.documents.len(),
            community_count: communities.len(),
            avg_community_size: if communities.is_empty() {
                0.0
            } else {
                communities.iter().map(|c| c.size as f64).sum::<f64>() / communities.len() as f64
            },
        }
    }
}

/// Graph statistics summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Total entities in the graph.
    pub total_entities: usize,
    /// Total relations.
    pub total_relations: usize,
    /// Total documents indexed.
    pub total_documents: usize,
    /// Number of communities detected.
    pub community_count: usize,
    /// Average community size.
    pub avg_community_size: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_and_search() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));

        svc.ingest(Document {
            id: "d1".into(),
            text: "Rust uses Cargo for building".into(),
            embedding: vec![0.9, 0.1, 0.0, 0.0],
            metadata: None,
        })
        .unwrap();

        let results = svc
            .search(KnowledgeQuery {
                embedding: vec![0.85, 0.15, 0.0, 0.0],
                k: 5,
                max_hops: 2,
                include_relations: true,
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn test_entity_extraction() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));

        let result = svc
            .ingest(Document {
                id: "d1".into(),
                text: "Rust and Cargo are great tools by Mozilla".into(),
                embedding: vec![0.5; 4],
                metadata: None,
            })
            .unwrap();

        assert!(result.entities_extracted > 0);
        assert!(svc.entity_count() > 0);
    }

    #[test]
    fn test_graph_expansion() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));

        // Two docs sharing entity "Rust"
        svc.ingest(Document {
            id: "d1".into(),
            text: "Rust is fast".into(),
            embedding: vec![0.9, 0.1, 0.0, 0.0],
            metadata: None,
        })
        .unwrap();

        svc.ingest(Document {
            id: "d2".into(),
            text: "Rust is safe".into(),
            embedding: vec![0.1, 0.9, 0.0, 0.0],
            metadata: None,
        })
        .unwrap();

        // Search near d1, should find d2 via shared "Rust" entity
        let results = svc
            .search(KnowledgeQuery {
                embedding: vec![0.9, 0.1, 0.0, 0.0],
                k: 5,
                max_hops: 2,
                include_relations: true,
            })
            .unwrap();

        assert!(results.len() >= 2);
    }

    #[test]
    fn test_relations() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));
        svc.ingest(Document {
            id: "d1".into(),
            text: "Rust Cargo Crate".into(),
            embedding: vec![0.5; 4],
            metadata: None,
        })
        .unwrap();

        assert!(!svc.relations().is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));
        assert!(svc
            .ingest(Document {
                id: "bad".into(),
                text: "test".into(),
                embedding: vec![0.5; 8],
                metadata: None,
            })
            .is_err());
    }

    #[test]
    fn test_community_detection() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));

        svc.ingest(Document {
            id: "d1".into(),
            text: "Rust Cargo Crate".into(),
            embedding: vec![0.9, 0.1, 0.0, 0.0],
            metadata: None,
        }).unwrap();

        svc.ingest(Document {
            id: "d2".into(),
            text: "Python Pip Package".into(),
            embedding: vec![0.1, 0.9, 0.0, 0.0],
            metadata: None,
        }).unwrap();

        let communities = svc.detect_communities(5);
        assert!(!communities.is_empty());
    }

    #[test]
    fn test_graph_stats() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));
        svc.ingest(Document {
            id: "d1".into(),
            text: "Rust Cargo Crate".into(),
            embedding: vec![0.5; 4],
            metadata: None,
        }).unwrap();

        let stats = svc.graph_stats();
        assert_eq!(stats.total_documents, 1);
        assert!(stats.total_entities > 0);
    }

    #[test]
    fn test_entity_property() {
        let mut svc = KnowledgeService::new(KnowledgeConfig::new(4));
        svc.ingest(Document {
            id: "d1".into(),
            text: "Rust is great".into(),
            embedding: vec![0.5; 4],
            metadata: None,
        }).unwrap();

        let added = svc.add_entity_property("Rust", "category", serde_json::json!("language"));
        assert!(added);
    }

    #[test]
    fn test_llm_extraction_config_default() {
        let config = LlmExtractionConfig::default();
        assert_eq!(config.model, "gpt-4");
        assert!(!config.entity_types.is_empty());
        assert!(!config.relation_types.is_empty());
    }
}
