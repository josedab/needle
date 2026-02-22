#![allow(clippy::unwrap_used)]
//! Multi-Modal Collection Service
//!
//! Modality-aware collection configuration with auto-modality detection,
//! unified multi-modal search API, and per-modality embedding management.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::multimodal_collection::{
//!     MultiModalCollectionService, ServiceConfig, ModalityDef,
//!     MultiModalDoc, MultiModalSearchQuery,
//! };
//! use needle::indexing::multimodal_fusion::Modality;
//!
//! let mut svc = MultiModalCollectionService::new(ServiceConfig::builder()
//!     .add_modality(ModalityDef::new(Modality::Text, 8))
//!     .add_modality(ModalityDef::new(Modality::Image, 8))
//!     .build());
//!
//! svc.insert(MultiModalDoc {
//!     id: "doc1".into(),
//!     modalities: vec![
//!         (Modality::Text, vec![1.0; 8]),
//!         (Modality::Image, vec![0.5; 8]),
//!     ],
//!     metadata: None,
//! }).unwrap();
//!
//! let results = svc.search(MultiModalSearchQuery {
//!     queries: vec![(Modality::Text, vec![1.0; 8])],
//!     k: 5,
//!     weights: None,
//! }).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::indexing::multimodal_fusion::{
    FusionConfig, FusionConfigBuilder, FusionStrategy, ModalityConfig, Modality,
    MultiModalDocument, MultiModalFusion, MultiModalQuery, MultiModalSearchResult,
};

// ── Modality Definition ──────────────────────────────────────────────────────

/// Definition of a modality for the collection.
#[derive(Debug, Clone)]
pub struct ModalityDef {
    /// Modality type.
    pub modality: Modality,
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Distance function.
    pub distance: DistanceFunction,
    /// Weight in fusion scoring.
    pub weight: f32,
}

impl ModalityDef {
    /// Create a new modality definition.
    pub fn new(modality: Modality, dimensions: usize) -> Self {
        Self {
            modality,
            dimensions,
            distance: DistanceFunction::Cosine,
            weight: 1.0,
        }
    }

    /// Set distance function.
    #[must_use]
    pub fn with_distance(mut self, dist: DistanceFunction) -> Self {
        self.distance = dist;
        self
    }

    /// Set weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }
}

// ── Service Configuration ────────────────────────────────────────────────────

/// Service configuration.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Registered modalities.
    pub modalities: Vec<ModalityDef>,
    /// Fusion strategy.
    pub fusion_strategy: FusionStrategy,
    /// Auto-detect modality from input.
    pub auto_detect: bool,
}

impl ServiceConfig {
    /// Create a builder.
    pub fn builder() -> ServiceConfigBuilder {
        ServiceConfigBuilder::default()
    }
}

/// Builder for service configuration.
#[derive(Debug, Default, Clone)]
pub struct ServiceConfigBuilder {
    modalities: Vec<ModalityDef>,
    fusion_strategy: FusionStrategy,
    auto_detect: bool,
}

impl ServiceConfigBuilder {
    /// Add a modality.
    #[must_use]
    pub fn add_modality(mut self, def: ModalityDef) -> Self {
        self.modalities.push(def);
        self
    }

    /// Set fusion strategy.
    #[must_use]
    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.fusion_strategy = strategy;
        self
    }

    /// Enable auto-modality detection.
    #[must_use]
    pub fn auto_detect(mut self, enabled: bool) -> Self {
        self.auto_detect = enabled;
        self
    }

    /// Build the config.
    pub fn build(self) -> ServiceConfig {
        ServiceConfig {
            modalities: self.modalities,
            fusion_strategy: self.fusion_strategy,
            auto_detect: self.auto_detect,
        }
    }
}

// ── Document & Query ─────────────────────────────────────────────────────────

/// A document with per-modality vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDoc {
    /// Document ID.
    pub id: String,
    /// Per-modality vectors.
    pub modalities: Vec<(Modality, Vec<f32>)>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

/// Search query across modalities.
#[derive(Debug, Clone)]
pub struct MultiModalSearchQuery {
    /// Per-modality query vectors.
    pub queries: Vec<(Modality, Vec<f32>)>,
    /// Number of results.
    pub k: usize,
    /// Optional weight overrides.
    pub weights: Option<HashMap<Modality, f32>>,
}

// ── Service Result ───────────────────────────────────────────────────────────

/// Enriched search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceSearchResult {
    /// Document ID.
    pub id: String,
    /// Fused score.
    pub score: f32,
    /// Per-modality scores.
    pub modality_scores: HashMap<String, f32>,
    /// Modalities present in the document.
    pub modalities_present: Vec<String>,
    /// Metadata.
    pub metadata: Option<Value>,
    /// Detected modality (if auto-detect enabled).
    pub detected_modality: Option<String>,
}

// ── Modality Stats ───────────────────────────────────────────────────────────

/// Statistics per modality.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModalityStats {
    /// Number of documents with this modality.
    pub doc_count: usize,
    /// Dimensions.
    pub dimensions: usize,
}

// ── Service ──────────────────────────────────────────────────────────────────

/// Multi-modal collection service.
pub struct MultiModalCollectionService {
    config: ServiceConfig,
    index: MultiModalFusion,
    modality_stats: HashMap<String, ModalityStats>,
}

impl MultiModalCollectionService {
    /// Create a new service.
    pub fn new(config: ServiceConfig) -> Self {
        let mut builder = FusionConfig::builder()
            .fusion_strategy(config.fusion_strategy);

        for def in &config.modalities {
            builder = builder.add_modality(
                ModalityConfig::new(def.modality.clone(), def.dimensions)
                    .with_distance(def.distance)
                    .with_weight(def.weight),
            );
        }

        let fusion_config = builder.build();
        Self {
            config,
            index: MultiModalFusion::new(fusion_config),
            modality_stats: HashMap::new(),
        }
    }

    /// Insert a multi-modal document.
    pub fn insert(&mut self, doc: MultiModalDoc) -> Result<()> {
        // Update modality stats
        for (modality, _) in &doc.modalities {
            let entry = self
                .modality_stats
                .entry(modality.to_string())
                .or_default();
            entry.doc_count += 1;
        }

        self.index.insert(MultiModalDocument {
            id: doc.id,
            vectors: doc.modalities,
            metadata: doc.metadata,
        })
    }

    /// Search across modalities.
    pub fn search(&self, query: MultiModalSearchQuery) -> Result<Vec<ServiceSearchResult>> {
        let results = self.index.search(MultiModalQuery {
            vectors: query.queries,
            k: query.k,
            modality_weights: query.weights,
        })?;

        Ok(results
            .into_iter()
            .map(|r| ServiceSearchResult {
                id: r.id,
                score: r.score,
                modality_scores: r.modality_scores,
                modalities_present: r.modalities_present,
                metadata: r.metadata,
                detected_modality: None,
            })
            .collect())
    }

    /// Delete a document.
    pub fn delete(&mut self, id: &str) -> bool {
        self.index.delete(id)
    }

    /// Get modality statistics.
    pub fn modality_stats(&self) -> &HashMap<String, ModalityStats> {
        &self.modality_stats
    }

    /// Total documents.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Detect modality from data characteristics.
    pub fn detect_modality(data: &[f32]) -> Modality {
        // Heuristic: determine modality from vector dimensions
        match data.len() {
            0..=128 => Modality::Text,
            129..=512 => Modality::Text,
            513..=1024 => Modality::Image,
            _ => Modality::Custom("unknown".into()),
        }
    }

    /// List registered modalities.
    pub fn registered_modalities(&self) -> Vec<&ModalityDef> {
        self.config.modalities.iter().collect()
    }

    /// Batch insert multiple documents.
    pub fn batch_insert(&mut self, docs: Vec<MultiModalDoc>) -> Result<usize> {
        let mut count = 0;
        for doc in docs {
            self.insert(doc)?;
            count += 1;
        }
        Ok(count)
    }

    /// Cross-modal search: query with one modality, find results across all modalities.
    /// This searches the single specified modality and returns results including
    /// information about other modalities present in each matching document.
    pub fn cross_modal_search(
        &self,
        query_modality: Modality,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<ServiceSearchResult>> {
        self.search(MultiModalSearchQuery {
            queries: vec![(query_modality, query_vector.to_vec())],
            k,
            weights: None,
        })
    }

    /// Get summary statistics for the entire collection.
    pub fn collection_stats(&self) -> MultiModalCollectionStats {
        MultiModalCollectionStats {
            total_documents: self.len(),
            modality_stats: self.modality_stats.clone(),
            registered_modalities: self
                .config
                .modalities
                .iter()
                .map(|m| m.modality.to_string())
                .collect(),
        }
    }
}

/// Summary statistics for a multi-modal collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalCollectionStats {
    /// Total document count.
    pub total_documents: usize,
    /// Per-modality statistics.
    pub modality_stats: HashMap<String, ModalityStats>,
    /// List of registered modality names.
    pub registered_modalities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_image_svc() -> MultiModalCollectionService {
        MultiModalCollectionService::new(
            ServiceConfig::builder()
                .add_modality(ModalityDef::new(Modality::Text, 8))
                .add_modality(ModalityDef::new(Modality::Image, 8))
                .build(),
        )
    }

    #[test]
    fn test_insert_and_search() {
        let mut svc = text_image_svc();
        svc.insert(MultiModalDoc {
            id: "d1".into(),
            modalities: vec![(Modality::Text, vec![1.0; 8])],
            metadata: None,
        })
        .unwrap();

        let results = svc
            .search(MultiModalSearchQuery {
                queries: vec![(Modality::Text, vec![1.0; 8])],
                k: 5,
                weights: None,
            })
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_multi_modality_insert() {
        let mut svc = text_image_svc();
        svc.insert(MultiModalDoc {
            id: "d1".into(),
            modalities: vec![
                (Modality::Text, vec![1.0; 8]),
                (Modality::Image, vec![0.5; 8]),
            ],
            metadata: None,
        })
        .unwrap();

        assert_eq!(svc.len(), 1);
        assert_eq!(svc.modality_stats().len(), 2);
    }

    #[test]
    fn test_delete() {
        let mut svc = text_image_svc();
        svc.insert(MultiModalDoc {
            id: "d1".into(),
            modalities: vec![(Modality::Text, vec![1.0; 8])],
            metadata: None,
        })
        .unwrap();

        assert!(svc.delete("d1"));
        assert_eq!(svc.len(), 0);
    }

    #[test]
    fn test_modality_detection() {
        assert!(matches!(
            MultiModalCollectionService::detect_modality(&vec![0.0; 64]),
            Modality::Text
        ));
        assert!(matches!(
            MultiModalCollectionService::detect_modality(&vec![0.0; 768]),
            Modality::Image
        ));
    }

    #[test]
    fn test_registered_modalities() {
        let svc = text_image_svc();
        assert_eq!(svc.registered_modalities().len(), 2);
    }

    #[test]
    fn test_weighted_search() {
        let mut svc = text_image_svc();
        svc.insert(MultiModalDoc {
            id: "d1".into(),
            modalities: vec![
                (Modality::Text, vec![1.0; 8]),
                (Modality::Image, vec![0.0; 8]),
            ],
            metadata: None,
        })
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(Modality::Image, 10.0);

        let results = svc
            .search(MultiModalSearchQuery {
                queries: vec![
                    (Modality::Text, vec![1.0; 8]),
                    (Modality::Image, vec![1.0; 8]),
                ],
                k: 1,
                weights: Some(weights),
            })
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_batch_insert() {
        let mut svc = text_image_svc();
        let docs = vec![
            MultiModalDoc {
                id: "d1".into(),
                modalities: vec![(Modality::Text, vec![1.0; 8])],
                metadata: None,
            },
            MultiModalDoc {
                id: "d2".into(),
                modalities: vec![(Modality::Text, vec![2.0; 8])],
                metadata: None,
            },
        ];
        let count = svc.batch_insert(docs).unwrap();
        assert_eq!(count, 2);
        assert_eq!(svc.len(), 2);
    }

    #[test]
    fn test_cross_modal_search() {
        let mut svc = text_image_svc();
        svc.insert(MultiModalDoc {
            id: "d1".into(),
            modalities: vec![
                (Modality::Text, vec![1.0; 8]),
                (Modality::Image, vec![0.5; 8]),
            ],
            metadata: None,
        })
        .unwrap();

        // Search with text, should find the document
        let results = svc
            .cross_modal_search(Modality::Text, &vec![1.0; 8], 5)
            .unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_collection_stats() {
        let mut svc = text_image_svc();
        svc.insert(MultiModalDoc {
            id: "d1".into(),
            modalities: vec![
                (Modality::Text, vec![1.0; 8]),
                (Modality::Image, vec![0.5; 8]),
            ],
            metadata: None,
        })
        .unwrap();

        let stats = svc.collection_stats();
        assert_eq!(stats.total_documents, 1);
        assert_eq!(stats.registered_modalities.len(), 2);
    }
}
