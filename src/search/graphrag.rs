#![allow(clippy::unwrap_used)]
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
    ) -> Result<Vec<GraphRAGResult>> {
        let hops = max_hops.unwrap_or(self.config.max_hops);

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Stage 1 – ANN vector search.
        let ann = self.hnsw.search(query_embedding, k, &self.vectors)?;

        let mut results_map: HashMap<String, GraphRAGResult> = HashMap::new();

        // Normalise distances to a 0..1 similarity score.
        let max_dist = ann.iter().map(|(_, d)| OrderedFloat(*d)).max().map_or(1.0, |d| d.0).max(1e-6);

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
        Ok(results)
    }
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

    /// Depth-first search traversal from a start entity.
    ///
    /// Returns entities reachable within `max_depth` hops, preferring deeper
    /// paths first (useful for finding long dependency chains).
    pub fn dfs_traverse(
        &self,
        start_entity_id: &str,
        max_depth: usize,
        k: usize,
    ) -> Vec<GraphRAGResult> {
        if !self.entities.contains_key(start_entity_id) {
            return Vec::new();
        }

        let mut results: Vec<GraphRAGResult> = Vec::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(start_entity_id.to_string());

        let mut stack: Vec<(String, Vec<String>, usize)> = vec![(
            start_entity_id.to_string(),
            vec![start_entity_id.to_string()],
            0,
        )];

        while let Some((current, path, depth)) = stack.pop() {
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
            if depth >= max_depth {
                continue;
            }
            if let Some(neighbors) = self.adjacency.get(&current) {
                for (nid, _w) in neighbors.iter().rev() {
                    if visited.contains(nid) {
                        continue;
                    }
                    visited.insert(nid.clone());
                    let mut new_path = path.clone();
                    new_path.push(nid.clone());
                    stack.push((nid.clone(), new_path, depth + 1));
                }
            }
        }

        results.sort_by(|a, b| OrderedFloat(b.combined_score).cmp(&OrderedFloat(a.combined_score)));
        results.truncate(k);
        results
    }

    /// Filter relationships by type, returning matching edges.
    pub fn filter_relationships(&self, relation_type: &RelationType) -> Vec<&Relationship> {
        self.relationships
            .iter()
            .filter(|r| &r.relation_type == relation_type)
            .collect()
    }

    /// Combined graph+vector search API.
    ///
    /// First performs vector ANN search for the query embedding, then expands
    /// results via graph traversal, and fuses scores with configurable weights.
    pub fn search_graphrag(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<GraphRAGResult>> {
        self.search(query_embedding, k, None)
    }

    /// Insert a document with automatic entity/relationship extraction.
    ///
    /// Extracts entities and relationships using rule-based NLP, optionally
    /// generates embeddings for each entity, and adds them to the graph.
    pub fn insert_document(
        &mut self,
        text: &str,
        doc_embedding: Option<&[f32]>,
    ) -> Result<(usize, usize)> {
        let entities = self.extract_entities_from_text(text);
        let relationships = self.extract_relationships_from_entities(&entities);

        let entity_count = entities.len();
        let rel_count = relationships.len();

        for mut entity in entities {
            entity.embedding = doc_embedding.map(|e| e.to_vec());
            self.add_entity(entity)?;
        }

        for rel in relationships {
            self.add_relationship(rel)?;
        }

        Ok((entity_count, rel_count))
    }

    /// Weighted graph traversal using edge weights for scoring.
    ///
    /// Unlike `multi_hop_search` which scores purely by hop distance,
    /// this method accumulates edge weights along paths, producing scores
    /// that reflect relationship strength.
    pub fn weighted_traversal(
        &self,
        start_entity_id: &str,
        max_hops: usize,
        k: usize,
    ) -> Vec<GraphRAGResult> {
        if !self.entities.contains_key(start_entity_id) {
            return Vec::new();
        }

        let mut results: Vec<GraphRAGResult> = Vec::new();
        let mut visited: HashMap<String, f32> = HashMap::new();
        // (node_id, path, depth, accumulated_weight)
        let mut queue: VecDeque<(String, Vec<String>, usize, f32)> = VecDeque::new();
        queue.push_back((
            start_entity_id.to_string(),
            vec![start_entity_id.to_string()],
            0,
            1.0,
        ));
        visited.insert(start_entity_id.to_string(), 1.0);

        while let Some((current, path, depth, acc_weight)) = queue.pop_front() {
            if depth > 0 {
                if let Some(entity) = self.entities.get(&current) {
                    // Score = accumulated weight decayed by depth
                    let graph_score = acc_weight / (1.0 + depth as f32);
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
                for (nid, edge_weight) in neighbors {
                    let new_weight = acc_weight * edge_weight;
                    let prev_weight = visited.get(nid).copied().unwrap_or(0.0);
                    if new_weight > prev_weight {
                        visited.insert(nid.clone(), new_weight);
                        let mut new_path = path.clone();
                        new_path.push(nid.clone());
                        queue.push_back((nid.clone(), new_path, depth + 1, new_weight));
                    }
                }
            }
        }

        results.sort_by(|a, b| OrderedFloat(b.combined_score).cmp(&OrderedFloat(a.combined_score)));
        results.truncate(k);
        results
    }

    /// Compute PageRank-inspired importance scores for all entities.
    /// Returns entity IDs sorted by importance (descending).
    pub fn compute_importance(&self, iterations: usize, damping: f32) -> Vec<(String, f32)> {
        let n = self.entities.len();
        if n == 0 {
            return Vec::new();
        }

        let ids: Vec<&String> = self.entities.keys().collect();
        let mut scores: HashMap<&String, f32> =
            ids.iter().map(|id| (*id, 1.0 / n as f32)).collect();

        for _ in 0..iterations {
            let mut new_scores: HashMap<&String, f32> =
                ids.iter().map(|id| (*id, (1.0 - damping) / n as f32)).collect();

            for id in &ids {
                if let Some(neighbors) = self.adjacency.get(*id) {
                    let out_degree = neighbors.len().max(1);
                    let share = scores[id] / out_degree as f32;
                    for (nid, _) in neighbors {
                        if let Some(score) = new_scores.get_mut(&nid) {
                            *score += damping * share;
                        }
                    }
                }
            }
            scores = new_scores;
        }

        let mut result: Vec<(String, f32)> = scores
            .into_iter()
            .map(|(id, score)| (id.clone(), score))
            .collect();
        result.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        result
    }

    /// Generate a summary string for a community.
    /// Returns the community members and their relationships.
    pub fn community_summary(&self, community_id: u32) -> Option<String> {
        let community = self.communities.iter().find(|c| c.id == community_id)?;
        let member_names: Vec<&str> = community
            .member_ids
            .iter()
            .filter_map(|id| self.entities.get(id).map(|e| e.name.as_str()))
            .collect();

        let internal_rels: Vec<String> = self
            .relationships
            .iter()
            .filter(|r| {
                community.member_ids.contains(&r.source_id)
                    && community.member_ids.contains(&r.target_id)
            })
            .map(|r| {
                let src = self.entities.get(&r.source_id).map_or(r.source_id.as_str(), |e| e.name.as_str());
                let tgt = self.entities.get(&r.target_id).map_or(r.target_id.as_str(), |e| e.name.as_str());
                format!("{} --[{:?}]--> {}", src, r.relation_type, tgt)
            })
            .collect();

        Some(format!(
            "Community {} ({} members): [{}]\nRelationships:\n{}",
            community_id,
            member_names.len(),
            member_names.join(", "),
            if internal_rels.is_empty() {
                "  (none)".to_string()
            } else {
                internal_rels.iter().map(|r| format!("  {r}")).collect::<Vec<_>>().join("\n")
            }
        ))
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
        let results = g.search(&query, 3, None).unwrap();
        assert!(!results.is_empty());
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
        let results = g.search(&query, 5, None).unwrap();
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

    #[test]
    fn test_dfs_traverse() {
        let dim = 32;
        let mut config = GraphRAGConfig::default();
        config.dimensions = dim;
        let mut g = GraphRAG::new(config);

        // Build a chain: A → B → C → D
        for (i, name) in ["A", "B", "C", "D"].iter().enumerate() {
            g.add_entity(make_entity(name, name, Some(random_vec(dim, i as u64 + 1))))
                .unwrap();
        }
        g.add_relationship(Relationship {
            source_id: "A".into(), target_id: "B".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "B".into(), target_id: "C".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "C".into(), target_id: "D".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        let results = g.dfs_traverse("A", 3, 10);
        assert_eq!(results.len(), 3); // B, C, D
        // All should have graph_score > 0
        assert!(results.iter().all(|r| r.graph_score > 0.0));
    }

    #[test]
    fn test_filter_relationships() {
        let dim = 32;
        let mut config = GraphRAGConfig::default();
        config.dimensions = dim;
        let mut g = GraphRAG::new(config);

        g.add_entity(make_entity("A", "A", Some(random_vec(dim, 1)))).unwrap();
        g.add_entity(make_entity("B", "B", Some(random_vec(dim, 2)))).unwrap();
        g.add_entity(make_entity("C", "C", Some(random_vec(dim, 3)))).unwrap();

        g.add_relationship(Relationship {
            source_id: "A".into(), target_id: "B".into(),
            relation_type: RelationType::PartOf, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "A".into(), target_id: "C".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        let part_of = g.filter_relationships(&RelationType::PartOf);
        assert_eq!(part_of.len(), 1);
        assert_eq!(part_of[0].target_id, "B");
    }

    // ── multi_hop_search: depth > graph diameter ─────────────────────────

    #[test]
    fn test_multi_hop_search_depth_exceeds_diameter() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        // Short chain: a -> b -> c
        for id in &["a", "b", "c"] {
            g.add_entity(make_entity(id, id, None)).unwrap();
        }
        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "b".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "b".into(), target_id: "c".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        // Depth 10 >> diameter 2
        let results = g.multi_hop_search("a", 10, 100);
        assert_eq!(results.len(), 2); // b, c (no more to find)
    }

    // ── multi_hop_search: graph with cycles ──────────────────────────────

    #[test]
    fn test_multi_hop_search_with_cycles() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        for id in &["a", "b", "c"] {
            g.add_entity(make_entity(id, id, None)).unwrap();
        }
        // Create cycle: a -> b -> c -> a
        for (s, t) in &[("a", "b"), ("b", "c"), ("c", "a")] {
            g.add_relationship(Relationship {
                source_id: s.to_string(), target_id: t.to_string(),
                relation_type: RelationType::RelatedTo, weight: 1.0,
                properties: HashMap::new(),
            }).unwrap();
        }

        let results = g.multi_hop_search("a", 5, 100);
        // Should not infinite loop; should find b and c
        assert!(results.len() >= 2);
        let ids: HashSet<String> = results.iter().map(|r| r.entity.id.clone()).collect();
        assert!(ids.contains("b"));
        assert!(ids.contains("c"));
    }

    // ── detect_communities: disconnected components ──────────────────────

    #[test]
    fn test_detect_communities_disconnected() {
        let mut g = GraphRAG::new(GraphRAGConfig {
            min_community_size: 2,
            ..Default::default()
        });

        // Component 1: a--b
        g.add_entity(make_entity("a", "A", None)).unwrap();
        g.add_entity(make_entity("b", "B", None)).unwrap();
        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "b".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        // Component 2: c--d
        g.add_entity(make_entity("c", "C", None)).unwrap();
        g.add_entity(make_entity("d", "D", None)).unwrap();
        g.add_relationship(Relationship {
            source_id: "c".into(), target_id: "d".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        let communities = g.detect_communities();
        // Should detect at least 2 communities
        assert!(communities.len() >= 2);
    }

    // ── detect_communities: single-node graph ────────────────────────────

    #[test]
    fn test_detect_communities_single_node() {
        let mut g = GraphRAG::new(GraphRAGConfig {
            min_community_size: 1,
            ..Default::default()
        });
        g.add_entity(make_entity("solo", "Solo", None)).unwrap();

        let communities = g.detect_communities();
        // Single node may or may not form a community depending on min_size
        let total_members: usize = communities.iter().map(|c| c.member_ids.len()).sum();
        assert!(total_members <= 1);
    }

    // ── get_neighbors: isolated node ─────────────────────────────────────

    #[test]
    fn test_get_neighbors_isolated_node() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("isolated", "Isolated", None)).unwrap();
        let neighbors = g.get_neighbors("isolated");
        assert!(neighbors.is_empty());
    }

    // ── get_neighbors: nonexistent node ──────────────────────────────────

    #[test]
    fn test_get_neighbors_nonexistent() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let neighbors = g.get_neighbors("nonexistent");
        assert!(neighbors.is_empty());
    }

    // ── add_relationship: nonexistent entity IDs ─────────────────────────

    #[test]
    fn test_add_relationship_nonexistent_source() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("b", "B", None)).unwrap();
        // add_relationship doesn't validate entity existence - it just adds to adjacency
        let result = g.add_relationship(Relationship {
            source_id: "nonexistent".into(),
            target_id: "b".into(),
            relation_type: RelationType::RelatedTo,
            weight: 1.0,
            properties: HashMap::new(),
        });
        // Should succeed (adjacency graph is permissive)
        assert!(result.is_ok());
    }

    #[test]
    fn test_add_relationship_nonexistent_target() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("a", "A", None)).unwrap();
        let result = g.add_relationship(Relationship {
            source_id: "a".into(),
            target_id: "nonexistent".into(),
            relation_type: RelationType::RelatedTo,
            weight: 1.0,
            properties: HashMap::new(),
        });
        assert!(result.is_ok());
    }

    // ── extract_entities_from_text: edge cases ───────────────────────────

    #[test]
    fn test_extract_entities_empty_text() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let entities = g.extract_entities_from_text("");
        assert!(entities.is_empty());
    }

    #[test]
    fn test_extract_entities_special_characters() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let entities = g.extract_entities_from_text("!@#$%^&*()");
        // Should not panic; may or may not extract entities
        let _ = entities;
    }

    // ── search combining vector + graph ──────────────────────────────────

    #[test]
    fn test_search_graphrag_combined() {
        let dim = 32;
        let mut g = GraphRAG::new(GraphRAGConfig {
            dimensions: dim,
            vector_weight: 0.5,
            graph_weight: 0.5,
            ..Default::default()
        });

        // Add entities with embeddings and relationships
        for i in 0..5 {
            g.add_entity(make_entity(
                &format!("e{i}"),
                &format!("Entity{i}"),
                Some(random_vec(dim, i)),
            )).unwrap();
        }
        g.add_relationship(Relationship {
            source_id: "e0".into(), target_id: "e1".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        let query = random_vec(dim, 0);
        let results = g.search(&query, 3, None).unwrap();
        assert!(!results.is_empty());
        // e0 should be closest by vector similarity
        assert_eq!(results[0].entity.id, "e0");
    }

    // ── multi_hop_search from nonexistent start ──────────────────────────

    #[test]
    fn test_multi_hop_search_nonexistent_start() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("a", "A", None)).unwrap();
        let results = g.multi_hop_search("nonexistent", 3, 10);
        assert!(results.is_empty());
    }

    // ── weighted_traversal ───────────────────────────────────────────────

    #[test]
    fn test_weighted_traversal_basic() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        for id in &["a", "b", "c", "d"] {
            g.add_entity(make_entity(id, id, None)).unwrap();
        }
        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "b".into(),
            relation_type: RelationType::RelatedTo, weight: 0.9,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "b".into(), target_id: "c".into(),
            relation_type: RelationType::RelatedTo, weight: 0.5,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "d".into(),
            relation_type: RelationType::RelatedTo, weight: 0.1,
            properties: HashMap::new(),
        }).unwrap();

        let results = g.weighted_traversal("a", 3, 10);
        assert!(!results.is_empty());
        // b should rank higher than d (higher weight)
        let b_score = results.iter().find(|r| r.entity.id == "b").map(|r| r.combined_score);
        let d_score = results.iter().find(|r| r.entity.id == "d").map(|r| r.combined_score);
        assert!(b_score > d_score);
    }

    #[test]
    fn test_weighted_traversal_nonexistent_start() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let results = g.weighted_traversal("nonexistent", 3, 10);
        assert!(results.is_empty());
    }

    // ── compute_importance ───────────────────────────────────────────────

    #[test]
    fn test_compute_importance() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        for id in &["hub", "spoke1", "spoke2", "spoke3"] {
            g.add_entity(make_entity(id, id, None)).unwrap();
        }
        // hub connects to all spokes
        for spoke in &["spoke1", "spoke2", "spoke3"] {
            g.add_relationship(Relationship {
                source_id: "hub".into(), target_id: spoke.to_string(),
                relation_type: RelationType::RelatedTo, weight: 1.0,
                properties: HashMap::new(),
            }).unwrap();
        }

        let importance = g.compute_importance(10, 0.85);
        assert_eq!(importance.len(), 4);
        // All scores should be positive
        for (_, score) in &importance {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_compute_importance_empty_graph() {
        let g = GraphRAG::new(GraphRAGConfig::default());
        let importance = g.compute_importance(10, 0.85);
        assert!(importance.is_empty());
    }

    // ── entity_type classification ───────────────────────────────────────

    #[test]
    fn test_entity_types() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        let person = Entity {
            id: "p1".to_string(),
            name: "Alice".to_string(),
            entity_type: EntityType::Person,
            embedding: None,
            properties: HashMap::new(),
            community_id: None,
        };
        g.add_entity(person).unwrap();
        assert_eq!(g.entities["p1"].entity_type, EntityType::Person);

        let org = Entity {
            id: "o1".to_string(),
            name: "Acme".to_string(),
            entity_type: EntityType::Organization,
            embedding: None,
            properties: HashMap::new(),
            community_id: None,
        };
        g.add_entity(org).unwrap();
        assert_eq!(g.entities["o1"].entity_type, EntityType::Organization);
    }

    // ── entity/relationship counts ───────────────────────────────────────

    #[test]
    fn test_counts() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        assert_eq!(g.entity_count(), 0);
        assert_eq!(g.relationship_count(), 0);
        assert_eq!(g.community_count(), 0);

        g.add_entity(make_entity("a", "A", None)).unwrap();
        g.add_entity(make_entity("b", "B", None)).unwrap();
        assert_eq!(g.entity_count(), 2);

        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "b".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        assert_eq!(g.relationship_count(), 1);
    }

    // ── search_graphrag combined ─────────────────────────────────────────

    #[test]
    fn test_search_graphrag_with_hops() {
        let dim = 32;
        let mut g = GraphRAG::new(GraphRAGConfig {
            dimensions: dim,
            max_hops: 2,
            vector_weight: 0.7,
            graph_weight: 0.3,
            ..Default::default()
        });

        for i in 0..5 {
            g.add_entity(make_entity(
                &format!("e{i}"),
                &format!("Entity{i}"),
                Some(random_vec(dim, i)),
            )).unwrap();
        }
        // Chain: e0 -> e1 -> e2
        g.add_relationship(Relationship {
            source_id: "e0".into(), target_id: "e1".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "e1".into(), target_id: "e2".into(),
            relation_type: RelationType::RelatedTo, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();

        let query = random_vec(dim, 0);
        let results = g.search_graphrag(&query, 5).unwrap();
        assert!(!results.is_empty());
    }

    // ── duplicate entity ID ──────────────────────────────────────────────

    #[test]
    fn test_add_duplicate_entity() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("e1", "First", None)).unwrap();
        let result = g.add_entity(make_entity("e1", "Duplicate", None));
        // Should either overwrite or error
        let _ = result;
        assert_eq!(g.entity_count(), 1);
    }

    // ── multiple relationship types ──────────────────────────────────────

    #[test]
    fn test_multiple_relationship_types() {
        let mut g = GraphRAG::new(GraphRAGConfig::default());
        g.add_entity(make_entity("a", "A", None)).unwrap();
        g.add_entity(make_entity("b", "B", None)).unwrap();

        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "b".into(),
            relation_type: RelationType::Contains, weight: 1.0,
            properties: HashMap::new(),
        }).unwrap();
        g.add_relationship(Relationship {
            source_id: "a".into(), target_id: "b".into(),
            relation_type: RelationType::References, weight: 0.5,
            properties: HashMap::new(),
        }).unwrap();

        assert_eq!(g.relationship_count(), 2);
        let contains = g.filter_relationships(&RelationType::Contains);
        assert_eq!(contains.len(), 1);
        let references = g.filter_relationships(&RelationType::References);
        assert_eq!(references.len(), 1);
    }
}
