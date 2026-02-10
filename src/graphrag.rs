#[allow(unused_imports)]
use std::collections::{HashMap, HashSet, VecDeque};

use ordered_float::OrderedFloat;
#[allow(unused_imports)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::distance::DistanceFunction;
#[allow(unused_imports)]
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The type category of a knowledge-graph entity.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Concept,
    Event,
    Document,
    Custom(String),
}

/// A node in the knowledge graph.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub embedding: Option<Vec<f32>>,
    pub properties: HashMap<String, serde_json::Value>,
    pub community_id: Option<u32>,
}

/// The semantic label on an edge between two entities.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RelationType {
    RelatedTo,
    PartOf,
    HasProperty,
    Causes,
    Precedes,
    Contains,
    References,
    Custom(String),
}

/// A directed, weighted edge between two entities.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Relationship {
    pub source_id: String,
    pub target_id: String,
    pub relation_type: RelationType,
    pub weight: f32,
    pub properties: HashMap<String, serde_json::Value>,
}

/// A community discovered by label propagation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Community {
    pub id: u32,
    pub member_ids: Vec<String>,
    pub summary: Option<String>,
    pub level: u32,
}

/// Tuning knobs for the GraphRAG engine.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GraphRAGConfig {
    pub max_hops: usize,
    pub vector_weight: f32,
    pub graph_weight: f32,
    pub community_resolution: f32,
    pub min_community_size: usize,
    pub dimensions: usize,
}

impl Default for GraphRAGConfig {
    fn default() -> Self {
        Self {
            max_hops: 3,
            vector_weight: 0.6,
            graph_weight: 0.4,
            community_resolution: 1.0,
            min_community_size: 2,
            dimensions: 384,
        }
    }
}

/// A single result from a GraphRAG search.
#[derive(Clone, Debug)]
pub struct GraphRAGResult {
    pub entity: Entity,
    pub vector_score: f32,
    pub graph_score: f32,
    pub combined_score: f32,
    pub path: Vec<String>,
    pub hop_count: usize,
}

// ---------------------------------------------------------------------------
// GraphRAG engine
// ---------------------------------------------------------------------------

/// Knowledge-graph-augmented retrieval over vector embeddings.
pub struct GraphRAG {
    config: GraphRAGConfig,
    entities: HashMap<String, Entity>,
    relationships: Vec<Relationship>,
    adjacency: HashMap<String, Vec<(String, f32)>>,
    communities: Vec<Community>,
    // HNSW index for entity-embedding ANN search.
    hnsw: HnswIndex,
    // Parallel vector store consumed by the HNSW index.
    vectors: Vec<Vec<f32>>,
    // Maps entity-id → internal numeric id used by HNSW.
    id_to_idx: HashMap<String, usize>,
    // Reverse map: internal numeric id → entity-id.
    idx_to_id: Vec<String>,
}

impl GraphRAG {
    /// Create a new, empty GraphRAG engine.
    pub fn new(config: GraphRAGConfig) -> Self {
        let hnsw_config = HnswConfig::default();
        let hnsw = HnswIndex::new(hnsw_config, DistanceFunction::Cosine);
        Self {
            config,
            entities: HashMap::new(),
            relationships: Vec::new(),
            adjacency: HashMap::new(),
            communities: Vec::new(),
            hnsw,
            vectors: Vec::new(),
            id_to_idx: HashMap::new(),
            idx_to_id: Vec::new(),
        }
    }

    // -- Mutation ---------------------------------------------------------

    /// Add an entity to the knowledge graph.
    ///
    /// If the entity carries an embedding it is also inserted into the
    /// internal HNSW index so that it can be found via vector search.
    pub fn add_entity(&mut self, entity: Entity) -> Result<()> {
        let id = entity.id.clone();

        if let Some(ref emb) = entity.embedding {
            let idx = self.vectors.len();
            self.vectors.push(emb.clone());
            self.id_to_idx.insert(id.clone(), idx);
            self.idx_to_id.push(id.clone());
            self.hnsw.insert(idx, emb, &self.vectors)?;
        }

        self.adjacency.entry(id.clone()).or_default();
        self.entities.insert(id, entity);
        Ok(())
    }

    /// Add a directed relationship between two entities.
    pub fn add_relationship(&mut self, relationship: Relationship) -> Result<()> {
        let src = relationship.source_id.clone();
        let tgt = relationship.target_id.clone();
        let w = relationship.weight;

        self.adjacency.entry(src.clone()).or_default().push((tgt.clone(), w));
        self.adjacency.entry(tgt).or_default().push((src, w));
        self.relationships.push(relationship);
        Ok(())
    }

    // -- Entity extraction (rule-based) -----------------------------------

    /// Very simple rule-based entity extraction.
    ///
    /// Splits by `". "` to get sentences, then pulls out capitalised tokens
    /// that are at least two characters long and not the first word in a
    /// sentence.
    pub fn extract_entities_from_text(&self, text: &str) -> Vec<Entity> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for sentence in text.split(". ") {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            for (i, word) in words.iter().enumerate() {
                // Skip the first word (sentence-initial capital) and short tokens.
                if i == 0 || word.len() < 2 {
                    continue;
                }
                let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
                if clean.is_empty() {
                    continue;
                }
                let first = clean.chars().next().unwrap_or('a');
                if first.is_uppercase() && !seen.contains(clean) {
                    seen.insert(clean.to_string());
                    result.push(Entity {
                        id: format!("entity_{}", clean.to_lowercase()),
                        name: clean.to_string(),
                        entity_type: EntityType::Concept,
                        embedding: None,
                        properties: HashMap::new(),
                        community_id: None,
                    });
                }
            }
        }
        result
    }

    /// Derive co-occurrence relationships between entities that appear in
    /// the same sentence.
    pub fn extract_relationships_from_entities(&self, entities: &[Entity]) -> Vec<Relationship> {
        let mut rels = Vec::new();
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                rels.push(Relationship {
                    source_id: entities[i].id.clone(),
                    target_id: entities[j].id.clone(),
                    relation_type: RelationType::RelatedTo,
                    weight: 1.0,
                    properties: HashMap::new(),
                });
            }
        }
        rels
    }

    // -- Community detection (label propagation) --------------------------

    /// Run label-propagation community detection.
    ///
    /// Every node starts with a unique label. On each iteration every node
    /// adopts the most frequent label among its neighbours (weighted). The
    /// algorithm terminates when no label changes or after a bounded number
    /// of iterations.
    pub fn detect_communities(&mut self) -> Vec<Community> {
        let entity_ids: Vec<String> = self.entities.keys().cloned().collect();
        if entity_ids.is_empty() {
            self.communities = Vec::new();
            return Vec::new();
        }

        // Assign initial labels (each node gets its own index as label).
        let id_to_label_idx: HashMap<String, usize> = entity_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let mut labels: Vec<u32> = (0..entity_ids.len() as u32).collect();

        let max_iters = 100;
        for _ in 0..max_iters {
            let mut changed = false;
            for (i, eid) in entity_ids.iter().enumerate() {
                if let Some(neighbors) = self.adjacency.get(eid) {
                    if neighbors.is_empty() {
                        continue;
                    }
                    // Accumulate weighted votes per label.
                    let mut votes: HashMap<u32, f32> = HashMap::new();
                    for (nid, w) in neighbors {
                        if let Some(&ni) = id_to_label_idx.get(nid) {
                            *votes.entry(labels[ni]).or_default() += w * self.config.community_resolution;
                        }
                    }
                    if let Some((&best_label, _)) = votes.iter().max_by_key(|(_, v)| OrderedFloat(**v)) {
                        if labels[i] != best_label {
                            labels[i] = best_label;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Group entities by label.
        let mut groups: HashMap<u32, Vec<String>> = HashMap::new();
        for (i, eid) in entity_ids.iter().enumerate() {
            groups.entry(labels[i]).or_default().push(eid.clone());
        }

        let mut communities = Vec::new();
        let mut next_id: u32 = 0;
        for (_, members) in groups {
            if members.len() < self.config.min_community_size {
                continue;
            }
            let cid = next_id;
            next_id += 1;
            for mid in &members {
                if let Some(e) = self.entities.get_mut(mid) {
                    e.community_id = Some(cid);
                }
            }
            communities.push(Community {
                id: cid,
                member_ids: members,
                summary: None,
                level: 0,
            });
        }

        self.communities = communities.clone();
        communities
    }

    // -- Search -----------------------------------------------------------

    /// Combined vector + graph search.
    ///
    /// 1. Find the `k` nearest entity embeddings via HNSW.
    /// 2. For each result, run BFS up to `max_hops` to discover graph
    ///    neighbours.
    /// 3. Compute `combined_score = vector_weight * vector_score +
    ///    graph_weight * graph_score` where `graph_score = 1 / (1 + hops)`.
    pub fn search(
        &self,
        query_embedding: &[f32],
        k: usize,
        max_hops: Option<usize>,
    ) -> Vec<GraphRAGResult> {
        let hops = max_hops.unwrap_or(self.config.max_hops);

        if self.vectors.is_empty() {
            return Vec::new();
        }

        // Stage 1 – ANN vector search.
        let ann = self.hnsw.search(query_embedding, k, &self.vectors);

        let mut results_map: HashMap<String, GraphRAGResult> = HashMap::new();

        // Normalise distances to a 0..1 similarity score.
        let max_dist = ann.iter().map(|(_, d)| OrderedFloat(*d)).max().map(|d| d.0).unwrap_or(1.0).max(1e-6);

        for (vid, dist) in &ann {
            if let Some(eid) = self.idx_to_id.get(*vid) {
                if let Some(entity) = self.entities.get(eid) {
                    let vector_score = 1.0 - (dist / max_dist);
                    let graph_score = 1.0; // hop 0
                    let combined = self.config.vector_weight * vector_score
                        + self.config.graph_weight * graph_score;
                    results_map.insert(
                        eid.clone(),
                        GraphRAGResult {
                            entity: entity.clone(),
                            vector_score,
                            graph_score,
                            combined_score: combined,
                            path: vec![eid.clone()],
                            hop_count: 0,
                        },
                    );
                }
            }
        }

        // Stage 2 – BFS graph expansion from seed results.
        let seeds: Vec<String> = results_map.keys().cloned().collect();
        for seed in seeds {
            let seed_vs = results_map[&seed].vector_score;
            let mut visited: HashSet<String> = HashSet::new();
            visited.insert(seed.clone());
            let mut queue: VecDeque<(String, Vec<String>, usize)> = VecDeque::new();
            queue.push_back((seed.clone(), vec![seed.clone()], 0));

            while let Some((current, path, depth)) = queue.pop_front() {
                if depth >= hops {
                    continue;
                }
                if let Some(neighbors) = self.adjacency.get(&current) {
                    for (nid, _w) in neighbors {
                        if visited.contains(nid) {
                            continue;
                        }
                        visited.insert(nid.clone());
                        let hop = depth + 1;
                        let graph_score = 1.0 / (1.0 + hop as f32);
                        let combined = self.config.vector_weight * seed_vs
                            + self.config.graph_weight * graph_score;
                        let mut new_path = path.clone();
                        new_path.push(nid.clone());

                        if let Some(entity) = self.entities.get(nid) {
                            let entry = results_map
                                .entry(nid.clone())
                                .or_insert_with(|| GraphRAGResult {
                                    entity: entity.clone(),
                                    vector_score: seed_vs,
                                    graph_score,
                                    combined_score: combined,
                                    path: new_path.clone(),
                                    hop_count: hop,
                                });
                            if combined > entry.combined_score {
                                entry.combined_score = combined;
                                entry.graph_score = graph_score;
                                entry.path = new_path.clone();
                                entry.hop_count = hop;
                            }
                        }

                        queue.push_back((nid.clone(), new_path, hop));
                    }
                }
            }
        }

        let mut results: Vec<GraphRAGResult> = results_map.into_values().collect();
        results.sort_by(|a, b| OrderedFloat(b.combined_score).cmp(&OrderedFloat(a.combined_score)));
        results.truncate(k);
        results
    }

    /// Pure graph traversal starting from a known entity.
    ///
    /// Performs BFS from `start_entity_id` up to `max_hops`, scoring each
    /// discovered entity as `1 / (1 + hop_distance)`.
    pub fn multi_hop_search(
        &self,
        start_entity_id: &str,
        max_hops: usize,
        k: usize,
    ) -> Vec<GraphRAGResult> {
        if !self.entities.contains_key(start_entity_id) {
            return Vec::new();
        }

        let mut results: Vec<GraphRAGResult> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_entity_id.to_string());

        let mut queue: VecDeque<(String, Vec<String>, usize)> = VecDeque::new();
        queue.push_back((
            start_entity_id.to_string(),
            vec![start_entity_id.to_string()],
            0,
        ));

        while let Some((current, path, depth)) = queue.pop_front() {
            if depth > 0 {
                if let Some(entity) = self.entities.get(&current) {
                    let graph_score = 1.0 / (1.0 + depth as f32);
                    results.push(GraphRAGResult {
                        entity: entity.clone(),
                        vector_score: 0.0,
                        graph_score,
                        combined_score: graph_score,
                        path: path.clone(),
                        hop_count: depth,
                    });
                }
            }
            if depth >= max_hops {
                continue;
            }
            if let Some(neighbors) = self.adjacency.get(&current) {
                for (nid, _w) in neighbors {
                    if visited.contains(nid) {
                        continue;
                    }
                    visited.insert(nid.clone());
                    let mut new_path = path.clone();
                    new_path.push(nid.clone());
                    queue.push_back((nid.clone(), new_path, depth + 1));
                }
            }
        }

        results.sort_by(|a, b| OrderedFloat(b.combined_score).cmp(&OrderedFloat(a.combined_score)));
        results.truncate(k);
        results
    }

    // -- Accessors --------------------------------------------------------

    /// Return the community that the given entity belongs to, if any.
    pub fn get_community(&self, entity_id: &str) -> Option<&Community> {
        let cid = self.entities.get(entity_id)?.community_id?;
        self.communities.iter().find(|c| c.id == cid)
    }

    /// Return direct neighbours and their edge weights.
    pub fn get_neighbors(&self, entity_id: &str) -> Vec<(String, f32)> {
        self.adjacency
            .get(entity_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }

    pub fn community_count(&self) -> usize {
        self.communities.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entity(id: &str, name: &str, embedding: Option<Vec<f32>>) -> Entity {
        Entity {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: EntityType::Concept,
            embedding,
            properties: HashMap::new(),
            community_id: None,
        }
    }

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random vector.
        let mut v = Vec::with_capacity(dim);
        let mut s = seed;
        for _ in 0..dim {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.push(((s >> 33) as f32) / (u32::MAX as f32));
        }
        // Normalise so cosine distance is meaningful.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    #[test]
    fn test_add_entity_and_retrieve() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        let e = make_entity("e1", "Alice", None);
        g.add_entity(e).unwrap();
        assert_eq!(g.entity_count(), 1);
        assert!(g.entities.contains_key("e1"));
        assert_eq!(g.entities["e1"].name, "Alice");
    }

    #[test]
    fn test_add_relationship() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("a", "A", None)).unwrap();
        g.add_entity(make_entity("b", "B", None)).unwrap();
        g.add_relationship(Relationship {
            source_id: "a".into(),
            target_id: "b".into(),
            relation_type: RelationType::RelatedTo,
            weight: 0.9,
            properties: HashMap::new(),
        })
        .unwrap();
        assert_eq!(g.relationship_count(), 1);
        // Adjacency is bidirectional.
        assert_eq!(g.get_neighbors("a").len(), 1);
        assert_eq!(g.get_neighbors("b").len(), 1);
    }

    #[test]
    fn test_entity_extraction_from_text() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let text = "Alice works at Google. Last summer Bob visited Paris for vacation.";
        let entities = g.extract_entities_from_text(text);
        let names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"Google"));
        assert!(names.contains(&"Bob"));
        assert!(names.contains(&"Paris"));
    }

    #[test]
    fn test_relationship_extraction() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let entities = vec![
            make_entity("a", "A", None),
            make_entity("b", "B", None),
            make_entity("c", "C", None),
        ];
        let rels = g.extract_relationships_from_entities(&entities);
        // C(3,2) = 3 pairs
        assert_eq!(rels.len(), 3);
    }

    #[test]
    fn test_community_detection() {
        let mut g = GraphRAG::new(GraphRAGConfig {
            min_community_size: 2,
            ..Default::default()
        });
        // Create a small connected component: a--b--c
        for id in &["a", "b", "c"] {
            g.add_entity(make_entity(id, id, None)).unwrap();
        }
        g.add_relationship(Relationship {
            source_id: "a".into(),
            target_id: "b".into(),
            relation_type: RelationType::RelatedTo,
            weight: 1.0,
            properties: HashMap::new(),
        })
        .unwrap();
        g.add_relationship(Relationship {
            source_id: "b".into(),
            target_id: "c".into(),
            relation_type: RelationType::RelatedTo,
            weight: 1.0,
            properties: HashMap::new(),
        })
        .unwrap();

        let communities = g.detect_communities();
        // All three should converge to the same community.
        assert!(!communities.is_empty());
        let total_members: usize = communities.iter().map(|c| c.member_ids.len()).sum();
        assert!(total_members >= 2);
    }

    #[test]
    fn test_vector_search() {
        let dim = 32;
        let mut g = GraphRAG::new(GraphRAGConfig {
            dimensions: dim,
            ..Default::default()
        });
        for i in 0..5 {
            let emb = random_vec(dim, i);
            g.add_entity(make_entity(
                &format!("v{i}"),
                &format!("Vec{i}"),
                Some(emb),
            ))
            .unwrap();
        }

        let query = random_vec(dim, 0); // should be closest to v0
        let results = g.search(&query, 3, None);
        assert!(!results.is_empty());
        // The top result should be v0 itself (identical embedding).
        assert_eq!(results[0].entity.id, "v0");
    }

    #[test]
    fn test_multi_hop_search() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        // Chain: a -> b -> c -> d
        for id in &["a", "b", "c", "d"] {
            g.add_entity(make_entity(id, id, None)).unwrap();
        }
        for (s, t) in &[("a", "b"), ("b", "c"), ("c", "d")] {
            g.add_relationship(Relationship {
                source_id: s.to_string(),
                target_id: t.to_string(),
                relation_type: RelationType::RelatedTo,
                weight: 1.0,
                properties: HashMap::new(),
            })
            .unwrap();
        }

        let results = g.multi_hop_search("a", 3, 10);
        assert_eq!(results.len(), 3); // b, c, d
        // b is 1 hop => graph_score 0.5, c is 2 => 0.333, d is 3 => 0.25
        assert_eq!(results[0].entity.id, "b");
        assert_eq!(results[0].hop_count, 1);
    }

    #[test]
    fn test_combined_scoring() {
        let cfg = GraphRAGConfig {
            vector_weight: 0.6,
            graph_weight: 0.4,
            ..Default::default()
        };
        // combined = 0.6 * 0.8 + 0.4 * 0.5 = 0.68
        let combined = cfg.vector_weight * 0.8 + cfg.graph_weight * 0.5;
        assert!((combined - 0.68).abs() < 1e-6);
    }

    #[test]
    fn test_get_neighbors() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("x", "X", None)).unwrap();
        g.add_entity(make_entity("y", "Y", None)).unwrap();
        g.add_entity(make_entity("z", "Z", None)).unwrap();
        g.add_relationship(Relationship {
            source_id: "x".into(),
            target_id: "y".into(),
            relation_type: RelationType::Contains,
            weight: 0.7,
            properties: HashMap::new(),
        })
        .unwrap();
        g.add_relationship(Relationship {
            source_id: "x".into(),
            target_id: "z".into(),
            relation_type: RelationType::References,
            weight: 0.3,
            properties: HashMap::new(),
        })
        .unwrap();

        let nbrs = g.get_neighbors("x");
        assert_eq!(nbrs.len(), 2);
        let ids: HashSet<String> = nbrs.iter().map(|(id, _)| id.clone()).collect();
        assert!(ids.contains("y"));
        assert!(ids.contains("z"));
    }

    #[test]
    fn test_empty_graph() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        assert_eq!(g.entity_count(), 0);
        assert_eq!(g.relationship_count(), 0);
        assert_eq!(g.community_count(), 0);
        let query = random_vec(32, 42);
        let results = g.search(&query, 5, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let cfg = GraphRAGConfig::default();
        assert_eq!(cfg.max_hops, 3);
        assert!((cfg.vector_weight - 0.6).abs() < 1e-6);
        assert!((cfg.graph_weight - 0.4).abs() < 1e-6);
        assert!((cfg.community_resolution - 1.0).abs() < 1e-6);
        assert_eq!(cfg.min_community_size, 2);
        assert_eq!(cfg.dimensions, 384);
    }
}
