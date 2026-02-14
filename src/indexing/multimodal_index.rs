//! Multi-Modal Unified Index
//!
//! Unified index supporting text, image, and audio embeddings with cross-modal
//! search, late fusion, and modality-aware distance functions.
//!
//! # Architecture
//!
//! ```text
//! Document (text + image + audio)
//!   ├── TextEmbedding   → shared HNSW
//!   ├── ImageEmbedding  → shared HNSW
//!   └── AudioEmbedding  → shared HNSW
//!         ↓
//!   LateFusion → combined ranking
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::multimodal_index::*;
//!
//! let config = MultiModalIndexConfig::default();
//! let mut index = MultiModalIndex::new(config);
//!
//! // Insert a document with text + image embeddings
//! let doc = MultiModalDocument::builder("doc1")
//!     .with_text_embedding(vec![0.1; 384])
//!     .with_image_embedding(vec![0.2; 512])
//!     .with_metadata(serde_json::json!({"title": "Example"}))
//!     .build();
//!
//! index.insert(doc).unwrap();
//!
//! // Cross-modal search: text query → find images
//! let results = index.cross_modal_search(
//!     &vec![0.1; 384],
//!     Modality::Text,
//!     Modality::Image,
//!     10,
//! ).unwrap();
//! ```

use std::collections::HashMap;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};

// ---------------------------------------------------------------------------
// Modality Types
// ---------------------------------------------------------------------------

/// Supported embedding modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    Code,
    Custom,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Image => write!(f, "image"),
            Self::Audio => write!(f, "audio"),
            Self::Video => write!(f, "video"),
            Self::Code => write!(f, "code"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Configuration for a single modality within the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfig {
    pub modality: Modality,
    pub dimension: usize,
    pub distance: DistanceFunction,
    pub weight: f32,
    pub hnsw_config: HnswConfig,
}

// ---------------------------------------------------------------------------
// Multi-Modal Document
// ---------------------------------------------------------------------------

/// A document with embeddings from multiple modalities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDoc {
    pub id: String,
    pub embeddings: HashMap<Modality, Vec<f32>>,
    pub metadata: Option<Value>,
}

/// Builder for constructing multi-modal documents.
pub struct MultiModalDocBuilder {
    id: String,
    embeddings: HashMap<Modality, Vec<f32>>,
    metadata: Option<Value>,
}

impl MultiModalDocBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            embeddings: HashMap::new(),
            metadata: None,
        }
    }

    pub fn with_text_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(Modality::Text, embedding);
        self
    }

    pub fn with_image_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(Modality::Image, embedding);
        self
    }

    pub fn with_audio_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(Modality::Audio, embedding);
        self
    }

    pub fn with_video_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(Modality::Video, embedding);
        self
    }

    pub fn with_code_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(Modality::Code, embedding);
        self
    }

    pub fn with_embedding(mut self, modality: Modality, embedding: Vec<f32>) -> Self {
        self.embeddings.insert(modality, embedding);
        self
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn build(self) -> MultiModalDoc {
        MultiModalDoc {
            id: self.id,
            embeddings: self.embeddings,
            metadata: self.metadata,
        }
    }
}

impl MultiModalDoc {
    pub fn builder(id: impl Into<String>) -> MultiModalDocBuilder {
        MultiModalDocBuilder::new(id)
    }

    /// Get the embedding for a specific modality.
    pub fn embedding(&self, modality: &Modality) -> Option<&Vec<f32>> {
        self.embeddings.get(modality)
    }

    /// List modalities present in this document.
    pub fn modalities(&self) -> Vec<Modality> {
        self.embeddings.keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Multi-Modal Index Configuration
// ---------------------------------------------------------------------------

/// Configuration for the multi-modal index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalIndexConfig {
    /// Per-modality configurations.
    pub modalities: Vec<ModalityConfig>,
    /// Fusion strategy for combining scores across modalities.
    pub fusion: FusionStrategy,
}

/// Strategy for combining scores from multiple modalities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FusionStrategy {
    /// Weighted average of normalised distances.
    WeightedAverage,
    /// Reciprocal Rank Fusion.
    ReciprocalRankFusion,
    /// Take the best score from any modality.
    BestOf,
    /// Require match in all queried modalities.
    Intersection,
}

impl Default for MultiModalIndexConfig {
    fn default() -> Self {
        Self {
            modalities: vec![
                ModalityConfig {
                    modality: Modality::Text,
                    dimension: 384,
                    distance: DistanceFunction::Cosine,
                    weight: 1.0,
                    hnsw_config: HnswConfig::default(),
                },
                ModalityConfig {
                    modality: Modality::Image,
                    dimension: 512,
                    distance: DistanceFunction::Cosine,
                    weight: 1.0,
                    hnsw_config: HnswConfig::default(),
                },
            ],
            fusion: FusionStrategy::WeightedAverage,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-Modality Index
// ---------------------------------------------------------------------------

struct ModalityIndex {
    config: ModalityConfig,
    hnsw: HnswIndex,
    /// (doc_index, vector) pairs for this modality.
    vectors: Vec<(usize, Vec<f32>)>,
}

impl ModalityIndex {
    fn new(config: ModalityConfig) -> Self {
        Self {
            hnsw: HnswIndex::new(config.hnsw_config.clone(), config.distance),
            config,
            vectors: Vec::new(),
        }
    }

    fn insert(&mut self, doc_index: usize, vector: &[f32]) -> Result<()> {
        if vector.len() != self.config.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimension,
                got: vector.len(),
            });
        }

        let idx = self.vectors.len();
        self.vectors.push((doc_index, vector.to_vec()));

        let all_vecs: Vec<Vec<f32>> = self.vectors.iter().map(|(_, v)| v.clone()).collect();
        self.hnsw
            .insert(idx, vector, &all_vecs)
            .map_err(|_| NeedleError::Index("HNSW insert failed".into()))?;

        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.config.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let all_vecs: Vec<Vec<f32>> = self.vectors.iter().map(|(_, v)| v.clone()).collect();
        let raw = self.hnsw.search(query, k, &all_vecs);

        Ok(raw
            .into_iter()
            .filter_map(|(idx, dist)| self.vectors.get(idx).map(|(doc_idx, _)| (*doc_idx, dist)))
            .collect())
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }
}

// ---------------------------------------------------------------------------
// Multi-Modal Index
// ---------------------------------------------------------------------------

/// Search result from the multi-modal index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalSearchResult {
    pub id: String,
    pub combined_score: f32,
    pub per_modality_scores: HashMap<Modality, f32>,
    pub metadata: Option<Value>,
    pub matching_modalities: Vec<Modality>,
}

/// The unified multi-modal index.
pub struct MultiModalUnifiedIndex {
    config: MultiModalIndexConfig,
    indices: HashMap<Modality, RwLock<ModalityIndex>>,
    documents: RwLock<Vec<MultiModalDoc>>,
    doc_id_map: RwLock<HashMap<String, usize>>,
}

impl MultiModalUnifiedIndex {
    /// Create a new multi-modal index.
    pub fn new(config: MultiModalIndexConfig) -> Self {
        let mut indices = HashMap::new();
        for mc in &config.modalities {
            indices.insert(mc.modality, RwLock::new(ModalityIndex::new(mc.clone())));
        }

        Self {
            config,
            indices,
            documents: RwLock::new(Vec::new()),
            doc_id_map: RwLock::new(HashMap::new()),
        }
    }

    /// Insert a multi-modal document.
    pub fn insert(&self, doc: MultiModalDoc) -> Result<()> {
        let doc_index = {
            let mut docs = self.documents.write();
            let idx = docs.len();
            self.doc_id_map.write().insert(doc.id.clone(), idx);
            docs.push(doc.clone());
            idx
        };

        for (modality, embedding) in &doc.embeddings {
            if let Some(index) = self.indices.get(modality) {
                index.write().insert(doc_index, embedding)?;
            }
        }

        Ok(())
    }

    /// Search within a single modality.
    pub fn search(
        &self,
        query: &[f32],
        modality: Modality,
        k: usize,
    ) -> Result<Vec<MultiModalSearchResult>> {
        let index = self.indices.get(&modality).ok_or_else(|| {
            NeedleError::InvalidInput(format!("No index for modality: {}", modality))
        })?;

        let raw = index.read().search(query, k)?;
        let docs = self.documents.read();

        Ok(raw
            .into_iter()
            .filter_map(|(doc_idx, dist)| {
                docs.get(doc_idx).map(|doc| {
                    let mut scores = HashMap::new();
                    scores.insert(modality, dist);
                    MultiModalSearchResult {
                        id: doc.id.clone(),
                        combined_score: 1.0 - dist.min(1.0),
                        per_modality_scores: scores,
                        metadata: doc.metadata.clone(),
                        matching_modalities: vec![modality],
                    }
                })
            })
            .collect())
    }

    /// Cross-modal search: query in one modality, find results from another.
    /// Requires documents that have embeddings in both modalities.
    pub fn cross_modal_search(
        &self,
        query: &[f32],
        query_modality: Modality,
        target_modality: Modality,
        k: usize,
    ) -> Result<Vec<MultiModalSearchResult>> {
        // Search in the query modality
        let candidates = self.search(query, query_modality, k * 3)?;

        // Filter to those that also have the target modality
        let docs = self.documents.read();
        let results: Vec<MultiModalSearchResult> = candidates
            .into_iter()
            .filter(|r| {
                if let Some(&idx) = self.doc_id_map.read().get(&r.id) {
                    if let Some(doc) = docs.get(idx) {
                        return doc.embeddings.contains_key(&target_modality);
                    }
                }
                false
            })
            .take(k)
            .map(|mut r| {
                r.matching_modalities.push(target_modality);
                r
            })
            .collect();

        Ok(results)
    }

    /// Multi-modal search with late fusion across specified modalities.
    pub fn fused_search(
        &self,
        queries: &HashMap<Modality, Vec<f32>>,
        k: usize,
    ) -> Result<Vec<MultiModalSearchResult>> {
        // Gather results from each modality
        let mut all_scores: HashMap<String, HashMap<Modality, f32>> = HashMap::new();

        for (modality, query) in queries {
            if let Ok(results) = self.search(query, *modality, k * 2) {
                for result in results {
                    all_scores
                        .entry(result.id.clone())
                        .or_default()
                        .insert(*modality, result.combined_score);
                }
            }
        }

        // Fuse scores
        let docs = self.documents.read();
        let weights: HashMap<Modality, f32> = self
            .config
            .modalities
            .iter()
            .map(|mc| (mc.modality, mc.weight))
            .collect();

        let mut fused_results: Vec<MultiModalSearchResult> = all_scores
            .into_iter()
            .filter_map(|(id, scores)| {
                let combined = match self.config.fusion {
                    FusionStrategy::WeightedAverage => {
                        let total_weight: f32 = scores.keys().filter_map(|m| weights.get(m)).sum();
                        if total_weight > 0.0 {
                            scores
                                .iter()
                                .map(|(m, s)| s * weights.get(m).unwrap_or(&1.0))
                                .sum::<f32>()
                                / total_weight
                        } else {
                            0.0
                        }
                    }
                    FusionStrategy::BestOf => scores.values().cloned().fold(0.0f32, f32::max),
                    FusionStrategy::ReciprocalRankFusion => {
                        // Simplified RRF: 1 / (k + rank)
                        scores
                            .values()
                            .map(|s| 1.0 / (60.0 + (1.0 - s) * 100.0))
                            .sum()
                    }
                    FusionStrategy::Intersection => {
                        if scores.len() == queries.len() {
                            scores.values().sum::<f32>() / scores.len() as f32
                        } else {
                            return None;
                        }
                    }
                };

                let metadata = self
                    .doc_id_map
                    .read()
                    .get(&id)
                    .and_then(|&idx| docs.get(idx))
                    .and_then(|doc| doc.metadata.clone());

                let matching: Vec<Modality> = scores.keys().copied().collect();

                Some(MultiModalSearchResult {
                    id,
                    combined_score: combined,
                    per_modality_scores: scores
                        .into_iter()
                        .map(|(m, s)| (m, 1.0 - s.min(1.0)))
                        .collect(),
                    metadata,
                    matching_modalities: matching,
                })
            })
            .collect();

        fused_results
            .sort_by(|a, b| OrderedFloat(b.combined_score).cmp(&OrderedFloat(a.combined_score)));
        fused_results.truncate(k);

        Ok(fused_results)
    }

    /// Get the number of documents.
    pub fn len(&self) -> usize {
        self.documents.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.read().is_empty()
    }

    /// Get statistics per modality.
    pub fn stats(&self) -> MultiModalStats {
        let per_modality: HashMap<Modality, usize> = self
            .indices
            .iter()
            .map(|(m, idx)| (*m, idx.read().len()))
            .collect();

        MultiModalStats {
            total_documents: self.len(),
            per_modality_vectors: per_modality,
            modality_count: self.indices.len(),
            fusion_strategy: self.config.fusion,
        }
    }
}

/// Statistics for the multi-modal index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalStats {
    pub total_documents: usize,
    pub per_modality_vectors: HashMap<Modality, usize>,
    pub modality_count: usize,
    pub fusion_strategy: FusionStrategy,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_document_builder() {
        let doc = MultiModalDoc::builder("doc1")
            .with_text_embedding(vec![0.1; 384])
            .with_image_embedding(vec![0.2; 512])
            .with_metadata(json!({"title": "Test"}))
            .build();

        assert_eq!(doc.id, "doc1");
        assert_eq!(doc.modalities().len(), 2);
        assert!(doc.embedding(&Modality::Text).is_some());
        assert!(doc.embedding(&Modality::Audio).is_none());
    }

    #[test]
    fn test_multimodal_index_creation() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_multimodal_insert_and_search() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());

        for i in 0..20 {
            let doc = MultiModalDoc::builder(format!("doc{}", i))
                .with_text_embedding(random_vector(384))
                .with_image_embedding(random_vector(512))
                .build();
            index.insert(doc).unwrap();
        }

        assert_eq!(index.len(), 20);

        let results = index
            .search(&random_vector(384), Modality::Text, 5)
            .unwrap();
        assert!(results.len() <= 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_multimodal_search_image() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());

        for i in 0..10 {
            let doc = MultiModalDoc::builder(format!("doc{}", i))
                .with_image_embedding(random_vector(512))
                .build();
            index.insert(doc).unwrap();
        }

        let results = index
            .search(&random_vector(512), Modality::Image, 5)
            .unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cross_modal_search() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());

        for i in 0..15 {
            let doc = MultiModalDoc::builder(format!("doc{}", i))
                .with_text_embedding(random_vector(384))
                .with_image_embedding(random_vector(512))
                .build();
            index.insert(doc).unwrap();
        }

        let results = index
            .cross_modal_search(&random_vector(384), Modality::Text, Modality::Image, 5)
            .unwrap();
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.matching_modalities.contains(&Modality::Image));
        }
    }

    #[test]
    fn test_fused_search_weighted() {
        let config = MultiModalIndexConfig {
            fusion: FusionStrategy::WeightedAverage,
            ..Default::default()
        };
        let index = MultiModalUnifiedIndex::new(config);

        for i in 0..10 {
            let doc = MultiModalDoc::builder(format!("doc{}", i))
                .with_text_embedding(random_vector(384))
                .with_image_embedding(random_vector(512))
                .build();
            index.insert(doc).unwrap();
        }

        let mut queries = HashMap::new();
        queries.insert(Modality::Text, random_vector(384));
        queries.insert(Modality::Image, random_vector(512));

        let results = index.fused_search(&queries, 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_fused_search_best_of() {
        let config = MultiModalIndexConfig {
            fusion: FusionStrategy::BestOf,
            ..Default::default()
        };
        let index = MultiModalUnifiedIndex::new(config);

        for i in 0..10 {
            let doc = MultiModalDoc::builder(format!("doc{}", i))
                .with_text_embedding(random_vector(384))
                .build();
            index.insert(doc).unwrap();
        }

        let mut queries = HashMap::new();
        queries.insert(Modality::Text, random_vector(384));

        let results = index.fused_search(&queries, 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());

        let doc = MultiModalDoc::builder("bad")
            .with_text_embedding(vec![0.1; 100]) // Wrong dim
            .build();

        assert!(index.insert(doc).is_err());
    }

    #[test]
    fn test_unknown_modality_search() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());
        let result = index.search(&random_vector(128), Modality::Audio, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_stats() {
        let index = MultiModalUnifiedIndex::new(MultiModalIndexConfig::default());

        index
            .insert(
                MultiModalDoc::builder("d1")
                    .with_text_embedding(random_vector(384))
                    .build(),
            )
            .unwrap();

        let stats = index.stats();
        assert_eq!(stats.total_documents, 1);
        assert_eq!(stats.modality_count, 2); // text + image configured
        assert_eq!(*stats.per_modality_vectors.get(&Modality::Text).unwrap(), 1);
    }

    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", Modality::Text), "text");
        assert_eq!(format!("{}", Modality::Image), "image");
        assert_eq!(format!("{}", Modality::Audio), "audio");
    }

    #[test]
    fn test_fusion_strategy_serialization() {
        let strategies = vec![
            FusionStrategy::WeightedAverage,
            FusionStrategy::ReciprocalRankFusion,
            FusionStrategy::BestOf,
            FusionStrategy::Intersection,
        ];
        for s in &strategies {
            let json = serde_json::to_string(s).unwrap();
            let decoded: FusionStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(*s, decoded);
        }
    }

    #[test]
    fn test_search_result_serialization() {
        let result = MultiModalSearchResult {
            id: "doc1".into(),
            combined_score: 0.85,
            per_modality_scores: {
                let mut m = HashMap::new();
                m.insert(Modality::Text, 0.1);
                m
            },
            metadata: Some(json!({"test": true})),
            matching_modalities: vec![Modality::Text],
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: MultiModalSearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "doc1");
    }
}
