//! Multi-Modal Late Fusion Index
//!
//! Cross-modal retrieval supporting image, text, and audio vectors in a single
//! index with late-fusion scoring and per-modality distance functions.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::indexing::multimodal_fusion::{
//!     MultiModalFusion, FusionConfig, Modality, ModalityConfig,
//!     MultiModalDocument, MultiModalQuery,
//! };
//! use needle::distance::DistanceFunction;
//!
//! let config = FusionConfig::builder()
//!     .add_modality(ModalityConfig::new(Modality::Text, 384))
//!     .add_modality(ModalityConfig::new(Modality::Image, 512))
//!     .build();
//!
//! let mut index = MultiModalFusion::new(config);
//!
//! // Insert a document with text and image embeddings
//! index.insert(MultiModalDocument {
//!     id: "doc1".into(),
//!     vectors: vec![
//!         (Modality::Text, vec![0.1f32; 384]),
//!         (Modality::Image, vec![0.2f32; 512]),
//!     ],
//!     metadata: None,
//! }).unwrap();
//!
//! // Cross-modal search: query with text, retrieve by text+image fusion
//! let results = index.search(MultiModalQuery {
//!     vectors: vec![(Modality::Text, vec![0.1f32; 384])],
//!     k: 5,
//!     modality_weights: None,
//! }).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Modality ─────────────────────────────────────────────────────────────────

/// Supported data modalities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text embeddings.
    Text,
    /// Image embeddings.
    Image,
    /// Audio embeddings.
    Audio,
    /// Video embeddings.
    Video,
    /// Code embeddings.
    Code,
    /// Custom modality.
    Custom(String),
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Image => write!(f, "image"),
            Self::Audio => write!(f, "audio"),
            Self::Video => write!(f, "video"),
            Self::Code => write!(f, "code"),
            Self::Custom(name) => write!(f, "{}", name),
        }
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a single modality.
#[derive(Debug, Clone)]
pub struct ModalityConfig {
    /// Modality type.
    pub modality: Modality,
    /// Embedding dimensions for this modality.
    pub dimensions: usize,
    /// Distance function for this modality.
    pub distance: DistanceFunction,
    /// Default weight in fusion scoring (0.0–1.0).
    pub default_weight: f32,
}

impl ModalityConfig {
    /// Create a new modality config.
    pub fn new(modality: Modality, dimensions: usize) -> Self {
        Self {
            modality,
            dimensions,
            distance: DistanceFunction::Cosine,
            default_weight: 1.0,
        }
    }

    /// Set distance function.
    #[must_use]
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance = distance;
        self
    }

    /// Set default weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.default_weight = weight;
        self
    }
}

/// Multi-modal fusion index configuration.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Registered modalities.
    pub modalities: HashMap<Modality, ModalityConfig>,
    /// Fusion strategy.
    pub fusion_strategy: FusionStrategy,
    /// Maximum documents.
    pub max_documents: usize,
}

impl FusionConfig {
    /// Create a builder.
    pub fn builder() -> FusionConfigBuilder {
        FusionConfigBuilder::default()
    }
}

/// Builder for `FusionConfig`.
#[derive(Debug, Default, Clone)]
pub struct FusionConfigBuilder {
    modalities: HashMap<Modality, ModalityConfig>,
    fusion_strategy: FusionStrategy,
}

impl FusionConfigBuilder {
    /// Add a modality.
    #[must_use]
    pub fn add_modality(mut self, config: ModalityConfig) -> Self {
        self.modalities.insert(config.modality.clone(), config);
        self
    }

    /// Set fusion strategy.
    #[must_use]
    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Build the config.
    pub fn build(self) -> FusionConfig {
        FusionConfig {
            modalities: self.modalities,
            fusion_strategy: self.fusion_strategy,
            max_documents: 10_000_000,
        }
    }
}

/// How to combine scores from multiple modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Weighted sum of per-modality distances.
    #[default]
    WeightedSum,
    /// Minimum distance across modalities.
    MinDistance,
    /// Maximum distance across modalities.
    MaxDistance,
    /// Reciprocal Rank Fusion.
    ReciprocalRankFusion,
}

// ── Document & Query ─────────────────────────────────────────────────────────

/// A multi-modal document with vectors for each modality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDocument {
    /// Unique identifier.
    pub id: String,
    /// Vectors keyed by modality.
    pub vectors: Vec<(Modality, Vec<f32>)>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

/// A multi-modal query.
#[derive(Debug, Clone)]
pub struct MultiModalQuery {
    /// Query vectors keyed by modality.
    pub vectors: Vec<(Modality, Vec<f32>)>,
    /// Number of results.
    pub k: usize,
    /// Optional per-modality weight overrides.
    pub modality_weights: Option<HashMap<Modality, f32>>,
}

// ── Search Result ────────────────────────────────────────────────────────────

/// Result from a multi-modal search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalSearchResult {
    /// Document ID.
    pub id: String,
    /// Fused score.
    pub score: f32,
    /// Per-modality distances.
    pub modality_scores: HashMap<String, f32>,
    /// Document metadata.
    pub metadata: Option<Value>,
    /// Which modalities this document has.
    pub modalities_present: Vec<String>,
}

// ── Internal storage ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct StoredDocument {
    id: String,
    vectors: HashMap<Modality, Vec<f32>>,
    metadata: Option<Value>,
}

// ── Multi-Modal Fusion Index ─────────────────────────────────────────────────

/// Multi-modal late fusion index.
pub struct MultiModalFusion {
    config: FusionConfig,
    documents: HashMap<String, StoredDocument>,
}

impl MultiModalFusion {
    /// Create a new multi-modal fusion index.
    pub fn new(config: FusionConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
        }
    }

    /// Insert a multi-modal document.
    pub fn insert(&mut self, doc: MultiModalDocument) -> Result<()> {
        if self.documents.len() >= self.config.max_documents {
            return Err(NeedleError::CapacityExceeded(
                "Max documents reached".into(),
            ));
        }

        // Validate dimensions for each modality
        for (modality, vector) in &doc.vectors {
            if let Some(config) = self.config.modalities.get(modality) {
                if vector.len() != config.dimensions {
                    return Err(NeedleError::DimensionMismatch {
                        expected: config.dimensions,
                        got: vector.len(),
                    });
                }
            }
            // Allow unregistered modalities (flexible schema)
        }

        let mut vectors = HashMap::new();
        for (modality, vector) in doc.vectors {
            vectors.insert(modality, vector);
        }

        self.documents.insert(
            doc.id.clone(),
            StoredDocument {
                id: doc.id,
                vectors,
                metadata: doc.metadata,
            },
        );

        Ok(())
    }

    /// Search across modalities with late fusion.
    pub fn search(&self, query: MultiModalQuery) -> Result<Vec<MultiModalSearchResult>> {
        let mut scored: Vec<MultiModalSearchResult> = Vec::new();

        for doc in self.documents.values() {
            let (score, modality_scores) = self.compute_fusion_score(doc, &query);
            if score.is_finite() {
                scored.push(MultiModalSearchResult {
                    id: doc.id.clone(),
                    score,
                    modality_scores,
                    metadata: doc.metadata.clone(),
                    modalities_present: doc
                        .vectors
                        .keys()
                        .map(|m| m.to_string())
                        .collect(),
                });
            }
        }

        scored.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(query.k);

        Ok(scored)
    }

    /// Delete a document.
    pub fn delete(&mut self, id: &str) -> bool {
        self.documents.remove(id).is_some()
    }

    /// Get document count.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// List registered modalities.
    pub fn modalities(&self) -> Vec<&Modality> {
        self.config.modalities.keys().collect()
    }

    /// Get a document by ID.
    pub fn get(&self, id: &str) -> Option<MultiModalDocument> {
        self.documents.get(id).map(|doc| MultiModalDocument {
            id: doc.id.clone(),
            vectors: doc
                .vectors
                .iter()
                .map(|(m, v)| (m.clone(), v.clone()))
                .collect(),
            metadata: doc.metadata.clone(),
        })
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn compute_fusion_score(
        &self,
        doc: &StoredDocument,
        query: &MultiModalQuery,
    ) -> (f32, HashMap<String, f32>) {
        let mut modality_scores: HashMap<String, f32> = HashMap::new();
        let mut total_weight: f32 = 0.0;
        let mut weighted_sum: f32 = 0.0;
        let mut min_dist: f32 = f32::MAX;
        let mut max_dist: f32 = f32::MIN;

        for (modality, query_vec) in &query.vectors {
            if let Some(doc_vec) = doc.vectors.get(modality) {
                let distance_fn = self
                    .config
                    .modalities
                    .get(modality)
                    .map(|c| c.distance)
                    .unwrap_or(DistanceFunction::Cosine);

                if query_vec.len() == doc_vec.len() {
                    let dist = distance_fn.compute(query_vec, doc_vec).unwrap_or(f32::MAX);

                    let weight = query
                        .modality_weights
                        .as_ref()
                        .and_then(|w| w.get(modality).copied())
                        .or_else(|| {
                            self.config
                                .modalities
                                .get(modality)
                                .map(|c| c.default_weight)
                        })
                        .unwrap_or(1.0);

                    modality_scores.insert(modality.to_string(), dist);
                    weighted_sum += dist * weight;
                    total_weight += weight;
                    min_dist = min_dist.min(dist);
                    max_dist = max_dist.max(dist);
                }
            }
        }

        let fused = match self.config.fusion_strategy {
            FusionStrategy::WeightedSum => {
                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    f32::INFINITY
                }
            }
            FusionStrategy::MinDistance => {
                if min_dist < f32::MAX {
                    min_dist
                } else {
                    f32::INFINITY
                }
            }
            FusionStrategy::MaxDistance => {
                if max_dist > f32::MIN {
                    max_dist
                } else {
                    f32::INFINITY
                }
            }
            FusionStrategy::ReciprocalRankFusion => {
                // RRF with k=60 (standard)
                if modality_scores.is_empty() {
                    f32::INFINITY
                } else {
                    let rrf: f32 = modality_scores
                        .values()
                        .map(|d| 1.0 / (60.0 + d))
                        .sum();
                    1.0 / rrf // invert so lower = better
                }
            }
        };

        (fused, modality_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_image_config() -> FusionConfig {
        FusionConfig::builder()
            .add_modality(ModalityConfig::new(Modality::Text, 8))
            .add_modality(ModalityConfig::new(Modality::Image, 8))
            .build()
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = MultiModalFusion::new(text_image_config());

        index
            .insert(MultiModalDocument {
                id: "doc1".into(),
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![0.5; 8]),
                ],
                metadata: None,
            })
            .unwrap();

        let results = index
            .search(MultiModalQuery {
                vectors: vec![(Modality::Text, vec![1.0; 8])],
                k: 5,
                modality_weights: None,
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
        assert!(results[0].modality_scores.contains_key("text"));
    }

    #[test]
    fn test_cross_modal_search() {
        let mut index = MultiModalFusion::new(text_image_config());

        index
            .insert(MultiModalDocument {
                id: "doc1".into(),
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ],
                metadata: None,
            })
            .unwrap();

        index
            .insert(MultiModalDocument {
                id: "doc2".into(),
                vectors: vec![
                    (Modality::Text, vec![0.0; 8]),
                    (Modality::Image, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                ],
                metadata: None,
            })
            .unwrap();

        // Search by image matching doc1's image direction
        let results = index
            .search(MultiModalQuery {
                vectors: vec![(Modality::Image, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])],
                k: 2,
                modality_weights: None,
            })
            .unwrap();

        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_dimension_validation() {
        let mut index = MultiModalFusion::new(text_image_config());

        let result = index.insert(MultiModalDocument {
            id: "bad".into(),
            vectors: vec![(Modality::Text, vec![1.0; 16])], // wrong dim
            metadata: None,
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_fusion() {
        let mut index = MultiModalFusion::new(text_image_config());

        index
            .insert(MultiModalDocument {
                id: "doc1".into(),
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![0.1; 8]),
                ],
                metadata: None,
            })
            .unwrap();

        // Weight image heavily
        let mut weights = HashMap::new();
        weights.insert(Modality::Image, 10.0);
        weights.insert(Modality::Text, 0.1);

        let results = index
            .search(MultiModalQuery {
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![1.0; 8]),
                ],
                k: 5,
                modality_weights: Some(weights),
            })
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_delete() {
        let mut index = MultiModalFusion::new(text_image_config());
        index
            .insert(MultiModalDocument {
                id: "doc1".into(),
                vectors: vec![(Modality::Text, vec![1.0; 8])],
                metadata: None,
            })
            .unwrap();

        assert!(index.delete("doc1"));
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_modalities_present() {
        let mut index = MultiModalFusion::new(text_image_config());
        index
            .insert(MultiModalDocument {
                id: "doc1".into(),
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![0.5; 8]),
                ],
                metadata: None,
            })
            .unwrap();

        let results = index
            .search(MultiModalQuery {
                vectors: vec![(Modality::Text, vec![1.0; 8])],
                k: 1,
                modality_weights: None,
            })
            .unwrap();

        assert_eq!(results[0].modalities_present.len(), 2);
    }

    #[test]
    fn test_min_distance_fusion() {
        let config = FusionConfig::builder()
            .add_modality(ModalityConfig::new(Modality::Text, 8))
            .add_modality(ModalityConfig::new(Modality::Image, 8))
            .fusion_strategy(FusionStrategy::MinDistance)
            .build();

        let mut index = MultiModalFusion::new(config);
        index
            .insert(MultiModalDocument {
                id: "doc1".into(),
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![0.0; 8]),
                ],
                metadata: None,
            })
            .unwrap();

        let results = index
            .search(MultiModalQuery {
                vectors: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![1.0; 8]),
                ],
                k: 1,
                modality_weights: None,
            })
            .unwrap();

        // Score should be min of text (0.0) and image (non-zero)
        assert_eq!(results.len(), 1);
    }
}
