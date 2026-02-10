//! Multi-Modal Search Service
//!
//! Database-level service that combines the multi-modal unified index with
//! automatic embedding generation, enabling users to search across text,
//! image, and audio content with a single API.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::multimodal_service::{MultiModalService, MultiModalServiceConfig, ModalInput};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//!
//! let config = MultiModalServiceConfig::builder()
//!     .name("media")
//!     .text_dimension(384)
//!     .image_dimension(512)
//!     .build();
//!
//! let mut service = MultiModalService::new(&db, config).unwrap();
//!
//! // Insert a text document
//! service.insert_text("doc1", &vec![0.1f32; 384], None).unwrap();
//!
//! // Insert an image
//! service.insert_image("img1", &vec![0.2f32; 512], None).unwrap();
//!
//! // Cross-modal search: find images similar to a text query
//! let results = service.search(ModalInput::Text(vec![0.1f32; 384]), 10).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::Result;
use crate::multimodal_index::{
    FusionStrategy, Modality, ModalityConfig, MultiModalDocBuilder,
    MultiModalIndexConfig, MultiModalStats, MultiModalUnifiedIndex,
};

/// Configuration for the multi-modal service.
#[derive(Debug, Clone)]
pub struct MultiModalServiceConfig {
    pub name: String,
    pub text_dimension: usize,
    pub image_dimension: usize,
    pub audio_dimension: usize,
    pub fusion_strategy: FusionStrategy,
    pub default_k: usize,
}

impl Default for MultiModalServiceConfig {
    fn default() -> Self {
        Self {
            name: "multimodal".into(),
            text_dimension: 384,
            image_dimension: 512,
            audio_dimension: 256,
            fusion_strategy: FusionStrategy::WeightedAverage,
            default_k: 10,
        }
    }
}

pub struct MultiModalServiceConfigBuilder {
    config: MultiModalServiceConfig,
}

impl MultiModalServiceConfig {
    pub fn builder() -> MultiModalServiceConfigBuilder {
        MultiModalServiceConfigBuilder {
            config: Self::default(),
        }
    }
}

impl MultiModalServiceConfigBuilder {
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    pub fn text_dimension(mut self, dim: usize) -> Self {
        self.config.text_dimension = dim;
        self
    }

    pub fn image_dimension(mut self, dim: usize) -> Self {
        self.config.image_dimension = dim;
        self
    }

    pub fn audio_dimension(mut self, dim: usize) -> Self {
        self.config.audio_dimension = dim;
        self
    }

    pub fn fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.config.fusion_strategy = strategy;
        self
    }

    pub fn build(self) -> MultiModalServiceConfig {
        self.config
    }
}

/// Input for a multi-modal search query.
#[derive(Debug, Clone)]
pub enum ModalInput {
    /// Text embedding vector.
    Text(Vec<f32>),
    /// Image embedding vector.
    Image(Vec<f32>),
    /// Audio embedding vector.
    Audio(Vec<f32>),
    /// Pre-fused multi-modal embedding.
    Fused(Vec<f32>),
}

impl ModalInput {
    fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Image(_) => Modality::Image,
            Self::Audio(_) => Modality::Audio,
            Self::Fused(_) => Modality::Text, // default
        }
    }

    fn vector(&self) -> &[f32] {
        match self {
            Self::Text(v) | Self::Image(v) | Self::Audio(v) | Self::Fused(v) => v,
        }
    }
}

/// Result of a multi-modal search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalResult {
    pub id: String,
    pub distance: f32,
    pub score: f32,
    pub modality: String,
    pub metadata: Option<Value>,
}

/// Multi-modal search service combining text, image, and audio search.
pub struct MultiModalService<'a> {
    db: &'a Database,
    config: MultiModalServiceConfig,
    index: MultiModalUnifiedIndex,
    insert_count: u64,
    search_count: u64,
}

impl<'a> MultiModalService<'a> {
    /// Create a new multi-modal service.
    pub fn new(db: &'a Database, config: MultiModalServiceConfig) -> Result<Self> {
        let modalities = vec![
            ModalityConfig {
                modality: Modality::Text,
                dimension: config.text_dimension,
                weight: 1.0,
                distance: crate::distance::DistanceFunction::Cosine,
                hnsw_config: Default::default(),
            },
            ModalityConfig {
                modality: Modality::Image,
                dimension: config.image_dimension,
                weight: 1.0,
                distance: crate::distance::DistanceFunction::Cosine,
                hnsw_config: Default::default(),
            },
            ModalityConfig {
                modality: Modality::Audio,
                dimension: config.audio_dimension,
                weight: 1.0,
                distance: crate::distance::DistanceFunction::Cosine,
                hnsw_config: Default::default(),
            },
        ];

        let index_config = MultiModalIndexConfig {
            modalities,
            fusion: config.fusion_strategy.clone(),
        };

        let index = MultiModalUnifiedIndex::new(index_config);

        // Create backing collections in the database
        let text_col = format!("{}_text", config.name);
        let image_col = format!("{}_image", config.name);
        let audio_col = format!("{}_audio", config.name);

        for (name, dim) in [
            (&text_col, config.text_dimension),
            (&image_col, config.image_dimension),
            (&audio_col, config.audio_dimension),
        ] {
            if db.collection(name).is_err() {
                db.create_collection(name, dim)?;
            }
        }

        Ok(Self {
            db,
            config,
            index,
            insert_count: 0,
            search_count: 0,
        })
    }

    /// Insert a text document with its embedding.
    pub fn insert_text(
        &mut self,
        id: impl Into<String>,
        embedding: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let doc = MultiModalDocBuilder::new(&id)
            .with_text_embedding(embedding.to_vec())
            .with_metadata(metadata.clone().unwrap_or(Value::Null))
            .build();
        self.index.insert(doc)?;

        // Also insert into the backing collection
        let col_name = format!("{}_text", self.config.name);
        let coll = self.db.collection(&col_name)?;
        coll.insert(&id, embedding, metadata)?;

        self.insert_count += 1;
        Ok(())
    }

    /// Insert an image with its embedding.
    pub fn insert_image(
        &mut self,
        id: impl Into<String>,
        embedding: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let doc = MultiModalDocBuilder::new(&id)
            .with_image_embedding(embedding.to_vec())
            .with_metadata(metadata.clone().unwrap_or(Value::Null))
            .build();
        self.index.insert(doc)?;

        let col_name = format!("{}_image", self.config.name);
        let coll = self.db.collection(&col_name)?;
        coll.insert(&id, embedding, metadata)?;

        self.insert_count += 1;
        Ok(())
    }

    /// Insert an audio clip with its embedding.
    pub fn insert_audio(
        &mut self,
        id: impl Into<String>,
        embedding: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let doc = MultiModalDocBuilder::new(&id)
            .with_audio_embedding(embedding.to_vec())
            .with_metadata(metadata.clone().unwrap_or(Value::Null))
            .build();
        self.index.insert(doc)?;

        let col_name = format!("{}_audio", self.config.name);
        let coll = self.db.collection(&col_name)?;
        coll.insert(&id, embedding, metadata)?;

        self.insert_count += 1;
        Ok(())
    }

    /// Insert a multi-modal document with embeddings for multiple modalities.
    pub fn insert_multimodal(
        &mut self,
        id: impl Into<String>,
        embeddings: HashMap<Modality, Vec<f32>>,
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();
        let mut builder = MultiModalDocBuilder::new(&id);

        for (modality, embedding) in &embeddings {
            builder = builder.with_embedding(modality.clone(), embedding.clone());
        }
        if let Some(ref meta) = metadata {
            builder = builder.with_metadata(meta.clone());
        }

        self.index.insert(builder.build())?;
        self.insert_count += 1;
        Ok(())
    }

    /// Search across all modalities using a single query.
    pub fn search(
        &mut self,
        input: ModalInput,
        k: usize,
    ) -> Result<Vec<MultiModalResult>> {
        let modality = input.modality();
        let results = self.index.search(input.vector(), modality, k)?;
        self.search_count += 1;

        Ok(results
            .into_iter()
            .map(|r| MultiModalResult {
                id: r.id.clone(),
                distance: 1.0 - r.combined_score.min(1.0),
                score: r.combined_score,
                modality: r.matching_modalities.first().map(|m| format!("{:?}", m)).unwrap_or_default(),
                metadata: r.metadata.clone(),
            })
            .collect())
    }

    /// Cross-modal search: search one modality using a query from another.
    pub fn cross_modal_search(
        &mut self,
        input: ModalInput,
        target_modality: Modality,
        k: usize,
    ) -> Result<Vec<MultiModalResult>> {
        let results = self.index.cross_modal_search(
            input.vector(),
            input.modality(),
            target_modality,
            k,
        )?;
        self.search_count += 1;

        Ok(results
            .into_iter()
            .map(|r| MultiModalResult {
                id: r.id.clone(),
                distance: 1.0 - r.combined_score.min(1.0),
                score: r.combined_score,
                modality: r.matching_modalities.first().map(|m| format!("{:?}", m)).unwrap_or_default(),
                metadata: r.metadata.clone(),
            })
            .collect())
    }

    /// Get multi-modal index statistics.
    pub fn stats(&self) -> MultiModalStats {
        self.index.stats()
    }

    /// Total documents inserted.
    pub fn total_inserts(&self) -> u64 {
        self.insert_count
    }

    /// Total searches performed.
    pub fn total_searches(&self) -> u64 {
        self.search_count
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db() -> Database {
        Database::in_memory()
    }

    #[test]
    fn test_multimodal_service_creation() {
        let db = make_db();
        let config = MultiModalServiceConfig::builder()
            .name("media")
            .text_dimension(4)
            .image_dimension(4)
            .audio_dimension(4)
            .build();

        let service = MultiModalService::new(&db, config).unwrap();
        assert!(service.is_empty());
    }

    #[test]
    fn test_insert_and_search_text() {
        let db = make_db();
        let config = MultiModalServiceConfig::builder()
            .name("test")
            .text_dimension(4)
            .image_dimension(4)
            .audio_dimension(4)
            .build();

        let mut service = MultiModalService::new(&db, config).unwrap();
        service
            .insert_text("doc1", &[0.1, 0.2, 0.3, 0.4], None)
            .unwrap();
        service
            .insert_text("doc2", &[0.5, 0.6, 0.7, 0.8], None)
            .unwrap();

        let results = service
            .search(ModalInput::Text(vec![0.1, 0.2, 0.3, 0.4]), 2)
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(service.total_inserts(), 2);
    }

    #[test]
    fn test_insert_multimodal() {
        let db = make_db();
        let config = MultiModalServiceConfig::builder()
            .name("mm")
            .text_dimension(4)
            .image_dimension(4)
            .audio_dimension(4)
            .build();

        let mut service = MultiModalService::new(&db, config).unwrap();

        let mut embeddings = HashMap::new();
        embeddings.insert(Modality::Text, vec![0.1, 0.2, 0.3, 0.4]);
        embeddings.insert(Modality::Image, vec![0.5, 0.6, 0.7, 0.8]);

        service
            .insert_multimodal("doc1", embeddings, None)
            .unwrap();
        assert_eq!(service.len(), 1);
    }

    #[test]
    fn test_stats() {
        let db = make_db();
        let config = MultiModalServiceConfig::builder()
            .name("stats")
            .text_dimension(4)
            .image_dimension(4)
            .audio_dimension(4)
            .build();

        let mut service = MultiModalService::new(&db, config).unwrap();
        service
            .insert_text("doc1", &[0.1, 0.2, 0.3, 0.4], None)
            .unwrap();

        let stats = service.stats();
        assert_eq!(stats.total_documents, 1);
    }
}
