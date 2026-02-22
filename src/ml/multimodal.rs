#![allow(clippy::unwrap_used)]
//! # Multi-Modal Embeddings
//!
//! This module provides unified embedding support for multiple modalities:
//! text, images, and audio. It enables cross-modal search and retrieval
//! within a single vector space.
//!
//! ## Features
//!
//! - **Unified API**: Single interface for all modalities
//! - **Cross-Modal Search**: Search images with text queries and vice versa
//! - **Late Fusion**: Combine embeddings from different modalities
//! - **Model Abstraction**: Support for CLIP, ImageBind, and custom models
//! - **Batch Processing**: Efficient batch embedding for all modalities
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use needle::multimodal::{MultiModalEmbedder, Modality, EmbedderBackend};
//!
//! // Create a multi-modal embedder
//! let embedder = MultiModalEmbedder::new(EmbedderBackend::Clip)?;
//!
//! // Embed text
//! let text_emb = embedder.embed_text("A photo of a cat")?;
//!
//! // Embed image (from bytes)
//! let image_emb = embedder.embed_image(&image_bytes)?;
//!
//! // Cross-modal search: text query against image embeddings
//! // The embeddings are in the same vector space!
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::error::{NeedleError, Result};

/// Supported modalities for embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Text content
    Text,
    /// Image content (PNG, JPEG, etc.)
    Image,
    /// Audio content (WAV, MP3, etc.)
    Audio,
    /// Video content (frames + audio)
    Video,
    /// Source code content
    Code,
    /// Generic binary content
    Binary,
}

impl Modality {
    /// Get the modality name
    pub fn name(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Audio => "audio",
            Modality::Video => "video",
            Modality::Code => "code",
            Modality::Binary => "binary",
        }
    }

    /// Detect modality from MIME type
    pub fn from_mime_type(mime: &str) -> Option<Self> {
        let lower = mime.to_lowercase();
        if lower.starts_with("text/x-") || lower.starts_with("application/x-") {
            Some(Modality::Code)
        } else if lower.starts_with("text/") {
            Some(Modality::Text)
        } else if lower.starts_with("image/") {
            Some(Modality::Image)
        } else if lower.starts_with("audio/") {
            Some(Modality::Audio)
        } else if lower.starts_with("video/") {
            Some(Modality::Video)
        } else if lower == "application/octet-stream" {
            Some(Modality::Binary)
        } else {
            None
        }
    }
}

/// Backend embedding model/service
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbedderBackend {
    /// OpenAI CLIP model (text + image)
    Clip,
    /// Meta's ImageBind model (text + image + audio + video)
    ImageBind,
    /// OpenAI's text embedding models
    OpenAI,
    /// Cohere's embedding models
    Cohere,
    /// Custom ONNX model
    OnnxCustom(String),
    /// Mock backend for testing
    Mock,
}

impl EmbedderBackend {
    /// Get supported modalities for this backend
    pub fn supported_modalities(&self) -> Vec<Modality> {
        match self {
            EmbedderBackend::Clip => vec![Modality::Text, Modality::Image],
            EmbedderBackend::ImageBind => vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
            ],
            EmbedderBackend::OpenAI => vec![Modality::Text],
            EmbedderBackend::Cohere => vec![Modality::Text],
            EmbedderBackend::OnnxCustom(_) => vec![Modality::Text],
            EmbedderBackend::Mock => vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
                Modality::Binary,
            ],
        }
    }

    /// Get default embedding dimension for this backend
    pub fn default_dimension(&self) -> usize {
        match self {
            EmbedderBackend::Clip => 512,
            EmbedderBackend::ImageBind => 1024,
            EmbedderBackend::OpenAI => 1536,
            EmbedderBackend::Cohere => 1024,
            EmbedderBackend::OnnxCustom(_) => 384,
            EmbedderBackend::Mock => 128,
        }
    }
}

/// Configuration for multi-modal embedder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Backend to use
    pub backend: EmbedderBackend,
    /// Output embedding dimension
    pub dimension: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Maximum text length
    pub max_text_length: usize,
    /// Maximum image size (width or height)
    pub max_image_size: u32,
    /// Audio sample rate for resampling
    pub audio_sample_rate: u32,
    /// Batch size for processing
    pub batch_size: usize,
    /// Cache embeddings
    pub enable_cache: bool,
    /// Maximum cache entries
    pub cache_size: usize,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            backend: EmbedderBackend::Mock,
            dimension: 512,
            normalize: true,
            max_text_length: 512,
            max_image_size: 512,
            audio_sample_rate: 16000,
            batch_size: 32,
            enable_cache: true,
            cache_size: 1000,
        }
    }
}

impl MultiModalConfig {
    /// Create config with specific backend
    #[must_use]
    pub fn with_backend(mut self, backend: EmbedderBackend) -> Self {
        self.dimension = backend.default_dimension();
        self.backend = backend;
        self
    }

    /// Set output dimension
    #[must_use]
    pub fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = dim;
        self
    }

    /// Set normalization
    #[must_use]
    pub fn with_normalization(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set batch size
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
}

/// Multi-modal embedding with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalEmbedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Source modality
    pub modality: Modality,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MultiModalEmbedding {
    /// Create a new embedding
    pub fn new(vector: Vec<f32>, modality: Modality) -> Self {
        Self {
            vector,
            modality,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get embedding dimension
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

/// Input content for embedding
#[derive(Debug, Clone)]
pub enum EmbedInput {
    /// Text content
    Text(String),
    /// Image bytes with format hint
    Image {
        data: Vec<u8>,
        format: Option<String>,
    },
    /// Audio bytes with format hint
    Audio {
        data: Vec<u8>,
        format: Option<String>,
        sample_rate: Option<u32>,
    },
    /// Video bytes with format hint
    Video {
        data: Vec<u8>,
        format: Option<String>,
    },
    /// Pre-computed embedding (for late fusion)
    Embedding(Vec<f32>),
}

impl EmbedInput {
    /// Create text input
    pub fn text(s: impl Into<String>) -> Self {
        EmbedInput::Text(s.into())
    }

    /// Create image input
    pub fn image(data: Vec<u8>) -> Self {
        EmbedInput::Image { data, format: None }
    }

    /// Create audio input
    pub fn audio(data: Vec<u8>) -> Self {
        EmbedInput::Audio {
            data,
            format: None,
            sample_rate: None,
        }
    }

    /// Get the modality of this input
    pub fn modality(&self) -> Modality {
        match self {
            EmbedInput::Text(_) => Modality::Text,
            EmbedInput::Image { .. } => Modality::Image,
            EmbedInput::Audio { .. } => Modality::Audio,
            EmbedInput::Video { .. } => Modality::Video,
            EmbedInput::Embedding(_) => Modality::Binary,
        }
    }
}

/// Statistics for the embedder
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EmbedderStats {
    /// Total embeddings generated
    pub total_embeddings: u64,
    /// Embeddings by modality
    pub by_modality: HashMap<String, u64>,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average embedding time in milliseconds
    pub avg_embed_time_ms: f32,
    /// Total batch operations
    pub batch_operations: u64,
}

/// Cache entry for embeddings
struct CacheEntry {
    embedding: Vec<f32>,
    _created_at: u64,
}

/// Multi-modal embedder
pub struct MultiModalEmbedder {
    config: MultiModalConfig,
    /// Embedding cache (hash -> embedding)
    cache: RwLock<HashMap<u64, CacheEntry>>,
    /// Statistics
    stats: RwLock<EmbedderStats>,
}

impl MultiModalEmbedder {
    /// Create a new multi-modal embedder
    pub fn new(config: MultiModalConfig) -> Result<Self> {
        info!(
            backend = ?config.backend,
            dimension = config.dimension,
            "Creating multi-modal embedder"
        );

        Ok(Self {
            config,
            cache: RwLock::new(HashMap::new()),
            stats: RwLock::new(EmbedderStats::default()),
        })
    }

    /// Create with default config and specified backend
    pub fn with_backend(backend: EmbedderBackend) -> Result<Self> {
        let config = MultiModalConfig::default().with_backend(backend);
        Self::new(config)
    }

    /// Embed text content
    pub fn embed_text(&self, text: &str) -> Result<MultiModalEmbedding> {
        self.embed(EmbedInput::Text(text.to_string()))
    }

    /// Embed image bytes
    pub fn embed_image(&self, data: &[u8]) -> Result<MultiModalEmbedding> {
        self.embed(EmbedInput::Image {
            data: data.to_vec(),
            format: None,
        })
    }

    /// Embed audio bytes
    pub fn embed_audio(&self, data: &[u8]) -> Result<MultiModalEmbedding> {
        self.embed(EmbedInput::Audio {
            data: data.to_vec(),
            format: None,
            sample_rate: None,
        })
    }

    /// Embed any supported input
    pub fn embed(&self, input: EmbedInput) -> Result<MultiModalEmbedding> {
        let modality = input.modality();

        // Check if modality is supported
        if !self
            .config
            .backend
            .supported_modalities()
            .contains(&modality)
        {
            return Err(NeedleError::InvalidInput(format!(
                "Backend {:?} does not support modality {:?}",
                self.config.backend, modality
            )));
        }

        // Check cache
        if self.config.enable_cache {
            let hash = Self::hash_input(&input);
            if let Some(entry) = self.cache.read().get(&hash) {
                self.stats.write().cache_hits += 1;
                return Ok(MultiModalEmbedding::new(entry.embedding.clone(), modality));
            }
            self.stats.write().cache_misses += 1;
        }

        // Generate embedding
        let start = std::time::Instant::now();
        let vector = self.generate_embedding(&input)?;
        let elapsed = start.elapsed().as_secs_f32() * 1000.0;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_embeddings += 1;
            *stats
                .by_modality
                .entry(modality.name().to_string())
                .or_default() += 1;

            // Update average time
            let n = stats.total_embeddings as f32;
            stats.avg_embed_time_ms = stats.avg_embed_time_ms * (n - 1.0) / n + elapsed / n;
        }

        // Normalize if configured
        let vector = if self.config.normalize {
            normalize_vector(&vector)
        } else {
            vector
        };

        // Cache result
        if self.config.enable_cache {
            let hash = Self::hash_input(&input);
            let mut cache = self.cache.write();

            // Evict if cache is full
            if cache.len() >= self.config.cache_size {
                // Simple eviction: remove first entry
                if let Some(&key) = cache.keys().next() {
                    cache.remove(&key);
                }
            }

            cache.insert(
                hash,
                CacheEntry {
                    embedding: vector.clone(),
                    _created_at: current_timestamp(),
                },
            );
        }

        debug!(modality = ?modality, elapsed_ms = elapsed, "Generated embedding");

        Ok(MultiModalEmbedding::new(vector, modality))
    }

    /// Embed a batch of inputs
    pub fn embed_batch(&self, inputs: &[EmbedInput]) -> Result<Vec<MultiModalEmbedding>> {
        self.stats.write().batch_operations += 1;

        // For now, process sequentially
        // In production, this would use batched inference
        inputs
            .iter()
            .map(|input| self.embed(input.clone()))
            .collect()
    }

    /// Generate embedding for an input (backend-specific)
    fn generate_embedding(&self, input: &EmbedInput) -> Result<Vec<f32>> {
        match &self.config.backend {
            EmbedderBackend::Mock => self.generate_mock_embedding(input),
            EmbedderBackend::Clip => self.generate_clip_embedding(input),
            EmbedderBackend::ImageBind => self.generate_imagebind_embedding(input),
            EmbedderBackend::OpenAI => self.generate_openai_embedding(input),
            EmbedderBackend::Cohere => self.generate_cohere_embedding(input),
            EmbedderBackend::OnnxCustom(model_path) => {
                self.generate_onnx_embedding(input, model_path)
            }
        }
    }

    /// Generate mock embedding for testing
    fn generate_mock_embedding(&self, input: &EmbedInput) -> Result<Vec<f32>> {
        // Generate deterministic embedding based on input hash
        let hash = Self::hash_input(input);
        let mut embedding = Vec::with_capacity(self.config.dimension);

        let mut state = hash;
        for _ in 0..self.config.dimension {
            // Simple LCG for reproducible random values
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let value = (state as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0;
            embedding.push(value);
        }

        Ok(embedding)
    }

    /// Generate CLIP embedding (stub)
    fn generate_clip_embedding(&self, input: &EmbedInput) -> Result<Vec<f32>> {
        // In production, this would call the CLIP model
        // For now, return mock embedding
        match input {
            EmbedInput::Text(_) | EmbedInput::Image { .. } => self.generate_mock_embedding(input),
            _ => Err(NeedleError::InvalidInput(
                "CLIP only supports text and image modalities".to_string(),
            )),
        }
    }

    /// Generate ImageBind embedding (stub)
    fn generate_imagebind_embedding(&self, input: &EmbedInput) -> Result<Vec<f32>> {
        // In production, this would call the ImageBind model
        // For now, return mock embedding
        self.generate_mock_embedding(input)
    }

    /// Generate OpenAI embedding (stub)
    fn generate_openai_embedding(&self, input: &EmbedInput) -> Result<Vec<f32>> {
        match input {
            EmbedInput::Text(_) => self.generate_mock_embedding(input),
            _ => Err(NeedleError::InvalidInput(
                "OpenAI only supports text modality".to_string(),
            )),
        }
    }

    /// Generate Cohere embedding (stub)
    fn generate_cohere_embedding(&self, input: &EmbedInput) -> Result<Vec<f32>> {
        match input {
            EmbedInput::Text(_) => self.generate_mock_embedding(input),
            _ => Err(NeedleError::InvalidInput(
                "Cohere only supports text modality".to_string(),
            )),
        }
    }

    /// Generate ONNX embedding (stub)
    fn generate_onnx_embedding(&self, input: &EmbedInput, _model_path: &str) -> Result<Vec<f32>> {
        // In production, this would load and run the ONNX model
        self.generate_mock_embedding(input)
    }

    /// Hash input for caching
    fn hash_input(input: &EmbedInput) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        match input {
            EmbedInput::Text(text) => {
                "text".hash(&mut hasher);
                text.hash(&mut hasher);
            }
            EmbedInput::Image { data, format } => {
                "image".hash(&mut hasher);
                data.hash(&mut hasher);
                format.hash(&mut hasher);
            }
            EmbedInput::Audio {
                data,
                format,
                sample_rate,
            } => {
                "audio".hash(&mut hasher);
                data.hash(&mut hasher);
                format.hash(&mut hasher);
                sample_rate.hash(&mut hasher);
            }
            EmbedInput::Video { data, format } => {
                "video".hash(&mut hasher);
                data.hash(&mut hasher);
                format.hash(&mut hasher);
            }
            EmbedInput::Embedding(vec) => {
                "embedding".hash(&mut hasher);
                for &v in vec {
                    v.to_bits().hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }

    /// Get current statistics
    pub fn stats(&self) -> EmbedderStats {
        self.stats.read().clone()
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
        info!("Cleared embedding cache");
    }

    /// Get configuration
    pub fn config(&self) -> &MultiModalConfig {
        &self.config
    }
}

/// Late fusion combiner for multi-modal embeddings
pub struct LateFusion {
    /// Weights for each modality
    weights: HashMap<Modality, f32>,
    /// Output dimension
    dimension: usize,
}

impl LateFusion {
    /// Create a new late fusion combiner
    pub fn new(dimension: usize) -> Self {
        let mut weights = HashMap::new();
        weights.insert(Modality::Text, 1.0);
        weights.insert(Modality::Image, 1.0);
        weights.insert(Modality::Audio, 1.0);
        weights.insert(Modality::Video, 1.0);

        Self { weights, dimension }
    }

    /// Set weight for a modality
    #[must_use]
    pub fn with_weight(mut self, modality: Modality, weight: f32) -> Self {
        self.weights.insert(modality, weight);
        self
    }

    /// Combine multiple embeddings into one
    pub fn combine(&self, embeddings: &[MultiModalEmbedding]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Err(NeedleError::InvalidInput(
                "No embeddings to combine".to_string(),
            ));
        }

        // Check dimensions match
        for emb in embeddings {
            if emb.dimension() != self.dimension {
                return Err(NeedleError::InvalidInput(format!(
                    "Embedding dimension {} doesn't match expected {}",
                    emb.dimension(),
                    self.dimension
                )));
            }
        }

        // Weighted average
        let mut result = vec![0.0f32; self.dimension];
        let mut total_weight = 0.0f32;

        for emb in embeddings {
            let weight = self.weights.get(&emb.modality).copied().unwrap_or(1.0);
            total_weight += weight;

            for (i, &v) in emb.vector.iter().enumerate() {
                result[i] += v * weight;
            }
        }

        // Normalize by total weight
        if total_weight > 0.0 {
            for v in &mut result {
                *v /= total_weight;
            }
        }

        // L2 normalize the result
        Ok(normalize_vector(&result))
    }
}

/// Cross-modal search helper
pub struct CrossModalSearch {
    embedder: Arc<MultiModalEmbedder>,
}

impl CrossModalSearch {
    /// Create a new cross-modal search helper
    pub fn new(embedder: Arc<MultiModalEmbedder>) -> Self {
        Self { embedder }
    }

    /// Embed a text query for searching images
    pub fn text_to_image_query(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.embedder.embed_text(text)?;
        Ok(embedding.vector)
    }

    /// Embed an image for searching similar images or related text
    pub fn image_to_query(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        let embedding = self.embedder.embed_image(image_data)?;
        Ok(embedding.vector)
    }

    /// Embed audio for searching related content
    pub fn audio_to_query(&self, audio_data: &[u8]) -> Result<Vec<f32>> {
        let embedding = self.embedder.embed_audio(audio_data)?;
        Ok(embedding.vector)
    }
}

/// Normalize a vector to unit length
fn normalize_vector(vector: &[f32]) -> Vec<f32> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        vector.iter().map(|x| x / norm).collect()
    } else {
        vector.to_vec()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// =============================================================================
// Advanced Multi-Modal Features (Next-Gen)
// =============================================================================

/// Cross-modal search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Enable cross-modal search
    pub enabled: bool,
    /// Weight for text modality in unified search
    pub text_weight: f32,
    /// Weight for image modality in unified search
    pub image_weight: f32,
    /// Weight for audio modality in unified search
    pub audio_weight: f32,
    /// Fusion strategy for multi-modal queries
    pub fusion_strategy: FusionStrategy,
    /// Normalize scores across modalities
    pub normalize_scores: bool,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            text_weight: 1.0,
            image_weight: 1.0,
            audio_weight: 1.0,
            fusion_strategy: FusionStrategy::WeightedSum,
            normalize_scores: true,
        }
    }
}

/// Strategy for fusing results from multiple modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple weighted sum of similarities
    WeightedSum,
    /// Reciprocal Rank Fusion
    RRF,
    /// Maximum similarity across modalities
    Max,
    /// Minimum similarity (conservative)
    Min,
    /// Late fusion with learnable weights
    LearnedWeights,
}

/// A multi-modal document with embeddings for different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalDocument {
    /// Document ID
    pub id: String,
    /// Text embedding (if available)
    pub text_embedding: Option<Vec<f32>>,
    /// Image embedding (if available)
    pub image_embedding: Option<Vec<f32>>,
    /// Audio embedding (if available)
    pub audio_embedding: Option<Vec<f32>>,
    /// Original text content
    pub text_content: Option<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Available modalities
    pub modalities: Vec<Modality>,
}

impl MultiModalDocument {
    /// Create a new document
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text_embedding: None,
            image_embedding: None,
            audio_embedding: None,
            text_content: None,
            metadata: None,
            modalities: Vec::new(),
        }
    }

    /// Add text content and embedding
    pub fn with_text(mut self, content: &str, embedding: Vec<f32>) -> Self {
        self.text_content = Some(content.to_string());
        self.text_embedding = Some(embedding);
        if !self.modalities.contains(&Modality::Text) {
            self.modalities.push(Modality::Text);
        }
        self
    }

    /// Add image embedding
    pub fn with_image(mut self, embedding: Vec<f32>) -> Self {
        self.image_embedding = Some(embedding);
        if !self.modalities.contains(&Modality::Image) {
            self.modalities.push(Modality::Image);
        }
        self
    }

    /// Add audio embedding
    pub fn with_audio(mut self, embedding: Vec<f32>) -> Self {
        self.audio_embedding = Some(embedding);
        if !self.modalities.contains(&Modality::Audio) {
            self.modalities.push(Modality::Audio);
        }
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get embedding for a specific modality
    pub fn embedding_for(&self, modality: Modality) -> Option<&Vec<f32>> {
        match modality {
            Modality::Text => self.text_embedding.as_ref(),
            Modality::Image => self.image_embedding.as_ref(),
            Modality::Audio => self.audio_embedding.as_ref(),
            _ => None,
        }
    }

    /// Check if document has a specific modality
    pub fn has_modality(&self, modality: Modality) -> bool {
        self.modalities.contains(&modality)
    }
}

/// Multi-modal query for unified search
#[derive(Debug, Clone)]
pub struct MultiModalQuery {
    /// Text query embedding
    pub text_embedding: Option<Vec<f32>>,
    /// Image query embedding
    pub image_embedding: Option<Vec<f32>>,
    /// Audio query embedding
    pub audio_embedding: Option<Vec<f32>>,
    /// Original text query
    pub text_query: Option<String>,
    /// Target modalities to search
    pub target_modalities: Vec<Modality>,
    /// Number of results
    pub k: usize,
    /// Fusion configuration
    pub config: CrossModalConfig,
}

impl MultiModalQuery {
    /// Create a text query
    pub fn text(text: &str, embedding: Vec<f32>) -> Self {
        Self {
            text_embedding: Some(embedding),
            image_embedding: None,
            audio_embedding: None,
            text_query: Some(text.to_string()),
            target_modalities: vec![Modality::Text, Modality::Image, Modality::Audio],
            k: 10,
            config: CrossModalConfig::default(),
        }
    }

    /// Create an image query
    pub fn image(embedding: Vec<f32>) -> Self {
        Self {
            text_embedding: None,
            image_embedding: Some(embedding),
            audio_embedding: None,
            text_query: None,
            target_modalities: vec![Modality::Text, Modality::Image],
            k: 10,
            config: CrossModalConfig::default(),
        }
    }

    /// Set target modalities
    pub fn with_targets(mut self, modalities: Vec<Modality>) -> Self {
        self.target_modalities = modalities;
        self
    }

    /// Set number of results
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set fusion configuration
    pub fn with_config(mut self, config: CrossModalConfig) -> Self {
        self.config = config;
        self
    }
}

/// Result from multi-modal search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalSearchResult {
    /// Document ID
    pub id: String,
    /// Overall similarity score
    pub score: f32,
    /// Scores by modality
    pub modality_scores: HashMap<String, f32>,
    /// Matching modalities
    pub matched_modalities: Vec<Modality>,
    /// Document metadata
    pub metadata: Option<serde_json::Value>,
}

/// Unified multi-modal index for cross-modal retrieval
pub struct UnifiedMultiModalIndex {
    documents: RwLock<HashMap<String, MultiModalDocument>>,
    config: CrossModalConfig,
    dimension: usize,
}

impl UnifiedMultiModalIndex {
    /// Create a new unified multi-modal index
    pub fn new(dimension: usize) -> Self {
        Self {
            documents: RwLock::new(HashMap::new()),
            config: CrossModalConfig::default(),
            dimension,
        }
    }

    /// Set cross-modal configuration
    pub fn with_config(mut self, config: CrossModalConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a document to the index
    pub fn add(&self, document: MultiModalDocument) -> Result<()> {
        // Validate dimensions
        if let Some(ref emb) = document.text_embedding {
            if emb.len() != self.dimension {
                return Err(NeedleError::DimensionMismatch {
                    expected: self.dimension,
                    got: emb.len(),
                });
            }
        }
        if let Some(ref emb) = document.image_embedding {
            if emb.len() != self.dimension {
                return Err(NeedleError::DimensionMismatch {
                    expected: self.dimension,
                    got: emb.len(),
                });
            }
        }
        if let Some(ref emb) = document.audio_embedding {
            if emb.len() != self.dimension {
                return Err(NeedleError::DimensionMismatch {
                    expected: self.dimension,
                    got: emb.len(),
                });
            }
        }

        self.documents.write().insert(document.id.clone(), document);
        Ok(())
    }

    /// Search across modalities
    pub fn search(&self, query: &MultiModalQuery) -> Vec<MultiModalSearchResult> {
        let documents = self.documents.read();
        let mut results = Vec::new();

        for (id, doc) in documents.iter() {
            let mut modality_scores = HashMap::new();
            let mut matched = Vec::new();

            // Calculate similarity for each modality
            if let Some(ref q_emb) = query.text_embedding {
                if let Some(ref d_emb) = doc.text_embedding {
                    let sim = cosine_similarity(q_emb, d_emb);
                    modality_scores.insert("text".to_string(), sim);
                    matched.push(Modality::Text);
                }
                // Cross-modal: text query vs image embedding
                if query.target_modalities.contains(&Modality::Image) {
                    if let Some(ref d_emb) = doc.image_embedding {
                        let sim = cosine_similarity(q_emb, d_emb);
                        modality_scores.insert("text_to_image".to_string(), sim);
                        if !matched.contains(&Modality::Image) {
                            matched.push(Modality::Image);
                        }
                    }
                }
            }

            if let Some(ref q_emb) = query.image_embedding {
                if let Some(ref d_emb) = doc.image_embedding {
                    let sim = cosine_similarity(q_emb, d_emb);
                    modality_scores.insert("image".to_string(), sim);
                    if !matched.contains(&Modality::Image) {
                        matched.push(Modality::Image);
                    }
                }
                // Cross-modal: image query vs text embedding
                if query.target_modalities.contains(&Modality::Text) {
                    if let Some(ref d_emb) = doc.text_embedding {
                        let sim = cosine_similarity(q_emb, d_emb);
                        modality_scores.insert("image_to_text".to_string(), sim);
                        if !matched.contains(&Modality::Text) {
                            matched.push(Modality::Text);
                        }
                    }
                }
            }

            if let Some(ref q_emb) = query.audio_embedding {
                if let Some(ref d_emb) = doc.audio_embedding {
                    let sim = cosine_similarity(q_emb, d_emb);
                    modality_scores.insert("audio".to_string(), sim);
                    matched.push(Modality::Audio);
                }
            }

            if modality_scores.is_empty() {
                continue;
            }

            // Fuse scores based on strategy
            let fused_score = self.fuse_scores(&modality_scores, &query.config);

            results.push(MultiModalSearchResult {
                id: id.clone(),
                score: fused_score,
                modality_scores,
                matched_modalities: matched,
                metadata: doc.metadata.clone(),
            });
        }

        // Sort by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.k);

        results
    }

    fn fuse_scores(&self, scores: &HashMap<String, f32>, config: &CrossModalConfig) -> f32 {
        match config.fusion_strategy {
            FusionStrategy::WeightedSum => {
                let mut total = 0.0f32;
                let mut weight_sum = 0.0f32;

                for (key, &score) in scores {
                    let weight = if key.contains("text") {
                        config.text_weight
                    } else if key.contains("image") {
                        config.image_weight
                    } else if key.contains("audio") {
                        config.audio_weight
                    } else {
                        1.0
                    };
                    total += score * weight;
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    total / weight_sum
                } else {
                    0.0
                }
            }
            FusionStrategy::Max => scores.values().cloned().fold(0.0f32, f32::max),
            FusionStrategy::Min => scores.values().cloned().fold(1.0f32, f32::min),
            FusionStrategy::RRF => {
                // Reciprocal Rank Fusion (simplified for single query)
                let k = 60.0f32; // RRF constant
                let mut rrf_sum = 0.0f32;
                for &score in scores.values() {
                    // Convert similarity to pseudo-rank (higher similarity = lower rank)
                    let rank = 1.0 + (1.0 - score) * 100.0;
                    rrf_sum += 1.0 / (k + rank);
                }
                rrf_sum
            }
            FusionStrategy::LearnedWeights => {
                // Fall back to weighted sum (learned weights would require training)
                self.fuse_scores(
                    scores,
                    &CrossModalConfig {
                        fusion_strategy: FusionStrategy::WeightedSum,
                        ..config.clone()
                    },
                )
            }
        }
    }

    /// Get document by ID
    pub fn get(&self, id: &str) -> Option<MultiModalDocument> {
        self.documents.read().get(id).cloned()
    }

    /// Delete document by ID
    pub fn delete(&self, id: &str) -> bool {
        self.documents.write().remove(id).is_some()
    }

    /// Get total document count
    pub fn len(&self) -> usize {
        self.documents.read().len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.documents.read().is_empty()
    }

    /// Get modality statistics
    pub fn modality_stats(&self) -> ModalityStats {
        let docs = self.documents.read();
        let mut stats = ModalityStats::default();

        for doc in docs.values() {
            if doc.text_embedding.is_some() {
                stats.text_count += 1;
            }
            if doc.image_embedding.is_some() {
                stats.image_count += 1;
            }
            if doc.audio_embedding.is_some() {
                stats.audio_count += 1;
            }
            stats.total_documents += 1;
        }

        stats
    }
}

/// Statistics about modalities in the index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModalityStats {
    /// Total documents
    pub total_documents: usize,
    /// Documents with text embeddings
    pub text_count: usize,
    /// Documents with image embeddings
    pub image_count: usize,
    /// Documents with audio embeddings
    pub audio_count: usize,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 1e-10 && norm_b > 1e-10 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Model Pipeline Infrastructure
// ---------------------------------------------------------------------------

/// Describes a model that can produce embeddings from raw inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    pub model_id: String,
    pub model_family: String,
    pub supported_modalities: Vec<Modality>,
    pub output_dimensions: usize,
    pub max_sequence_length: Option<usize>,
    pub max_image_size: Option<(u32, u32)>,
    pub quantized: bool,
    pub size_bytes: u64,
}

/// Image preprocessing steps before model inference.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    pub target_size: (u32, u32),
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        // CLIP defaults
        Self {
            target_size: (224, 224),
            normalize_mean: [0.48145466, 0.4578275, 0.40821073],
            normalize_std: [0.26862954, 0.26130258, 0.27577711],
        }
    }
}

impl ImagePreprocessor {
    /// Preprocess raw image bytes into a normalized float tensor.
    /// Returns flattened [C, H, W] tensor.
    pub fn preprocess(&self, _raw_bytes: &[u8]) -> Result<Vec<f32>> {
        let (h, w) = self.target_size;
        let c = 3usize;
        let size = c * h as usize * w as usize;

        // Create a deterministic placeholder tensor from raw bytes hash.
        // Real implementation would decode image, resize, and normalize.
        let hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in _raw_bytes.iter().take(256) {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };

        let mut tensor = Vec::with_capacity(size);
        let mut state = hash;
        for _ in 0..size {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let raw = (state >> 33) as f32 / (u32::MAX as f32); // 0..1
            tensor.push(raw);
        }

        // Apply normalization
        for ch in 0..c {
            let start = ch * (h as usize * w as usize);
            let end = start + (h as usize * w as usize);
            for val in &mut tensor[start..end] {
                *val = (*val - self.normalize_mean[ch]) / self.normalize_std[ch];
            }
        }

        Ok(tensor)
    }
}

/// Text preprocessing for embedding models.
#[derive(Debug, Clone)]
pub struct TextPreprocessor {
    pub max_tokens: usize,
    pub lowercase: bool,
    pub strip_punctuation: bool,
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            lowercase: true,
            strip_punctuation: false,
        }
    }
}

impl TextPreprocessor {
    /// Preprocess text into token IDs (simplified whitespace tokenizer).
    pub fn preprocess(&self, text: &str) -> Vec<String> {
        let text = if self.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        let text = if self.strip_punctuation {
            text.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect()
        } else {
            text
        };

        text.split_whitespace()
            .take(self.max_tokens)
            .map(String::from)
            .collect()
    }
}

/// Registry of available embedding models with lazy loading.
pub struct EmbeddingModelRegistry {
    manifests: parking_lot::RwLock<Vec<ModelManifest>>,
    image_preprocessor: ImagePreprocessor,
    text_preprocessor: TextPreprocessor,
}

impl EmbeddingModelRegistry {
    pub fn new() -> Self {
        let mut manifests = Vec::new();

        // Register known models
        manifests.push(ModelManifest {
            model_id: "openai/clip-vit-base-patch32".into(),
            model_family: "CLIP".into(),
            supported_modalities: vec![Modality::Text, Modality::Image],
            output_dimensions: 512,
            max_sequence_length: Some(77),
            max_image_size: Some((224, 224)),
            quantized: false,
            size_bytes: 600 * 1024 * 1024,
        });
        manifests.push(ModelManifest {
            model_id: "openai/clip-vit-large-patch14".into(),
            model_family: "CLIP".into(),
            supported_modalities: vec![Modality::Text, Modality::Image],
            output_dimensions: 768,
            max_sequence_length: Some(77),
            max_image_size: Some((224, 224)),
            quantized: false,
            size_bytes: 1_800 * 1024 * 1024,
        });
        manifests.push(ModelManifest {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            model_family: "SentenceTransformer".into(),
            supported_modalities: vec![Modality::Text],
            output_dimensions: 384,
            max_sequence_length: Some(256),
            max_image_size: None,
            quantized: false,
            size_bytes: 80 * 1024 * 1024,
        });

        Self {
            manifests: parking_lot::RwLock::new(manifests),
            image_preprocessor: ImagePreprocessor::default(),
            text_preprocessor: TextPreprocessor::default(),
        }
    }

    /// Register a custom model.
    pub fn register(&self, manifest: ModelManifest) {
        self.manifests.write().push(manifest);
    }

    /// Find the best model for given modalities and dimension requirements.
    pub fn find_model(
        &self,
        modalities: &[Modality],
        preferred_dimensions: Option<usize>,
    ) -> Option<ModelManifest> {
        let manifests = self.manifests.read();
        let mut candidates: Vec<&ModelManifest> = manifests
            .iter()
            .filter(|m| {
                modalities
                    .iter()
                    .all(|mod_| m.supported_modalities.contains(mod_))
            })
            .collect();

        if let Some(dim) = preferred_dimensions {
            candidates.sort_by_key(|m| (m.output_dimensions as i64 - dim as i64).unsigned_abs());
        } else {
            candidates.sort_by_key(|m| m.size_bytes);
        }

        candidates.first().cloned().cloned()
    }

    /// List all registered models.
    pub fn list_models(&self) -> Vec<ModelManifest> {
        self.manifests.read().clone()
    }

    /// Get image preprocessor.
    pub fn image_preprocessor(&self) -> &ImagePreprocessor {
        &self.image_preprocessor
    }

    /// Get text preprocessor.
    pub fn text_preprocessor(&self) -> &TextPreprocessor {
        &self.text_preprocessor
    }
}

impl Default for EmbeddingModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Lazy Model Download & Caching ────────────────────────────────────────────

/// Model download status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model not yet downloaded.
    NotDownloaded,
    /// Model is being downloaded.
    Downloading { progress_pct: u8 },
    /// Model is downloaded and ready.
    Ready,
    /// Download failed.
    Failed(String),
}

/// Configuration for a pre-built model adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAdapter {
    /// Human-readable name.
    pub name: String,
    /// Model family (CLIP, SigLIP, ImageBind, Whisper).
    pub family: String,
    /// Supported input modalities.
    pub modalities: Vec<Modality>,
    /// Output embedding dimension.
    pub dimensions: usize,
    /// Model download URL.
    pub download_url: Option<String>,
    /// Expected file size in bytes.
    pub size_bytes: u64,
    /// Local cache directory.
    pub cache_dir: String,
    /// Current status.
    pub status: ModelStatus,
}

/// Registry of pre-built model adapters with lazy download.
pub struct ModelAdapterRegistry {
    adapters: parking_lot::RwLock<Vec<ModelAdapter>>,
    cache_dir: String,
}

impl ModelAdapterRegistry {
    /// Create a new registry with default adapters.
    pub fn new(cache_dir: &str) -> Self {
        let registry = Self {
            adapters: parking_lot::RwLock::new(Vec::new()),
            cache_dir: cache_dir.into(),
        };
        registry.register_defaults();
        registry
    }

    fn register_defaults(&self) {
        let defaults = vec![
            ModelAdapter {
                name: "CLIP ViT-B/32".into(),
                family: "clip".into(),
                modalities: vec![Modality::Text, Modality::Image],
                dimensions: 512,
                download_url: Some("https://huggingface.co/openai/clip-vit-base-patch32".into()),
                size_bytes: 350_000_000,
                cache_dir: format!("{}/clip-vit-b32", self.cache_dir),
                status: ModelStatus::NotDownloaded,
            },
            ModelAdapter {
                name: "SigLIP Base".into(),
                family: "siglip".into(),
                modalities: vec![Modality::Text, Modality::Image],
                dimensions: 768,
                download_url: Some("https://huggingface.co/google/siglip-base-patch16-224".into()),
                size_bytes: 400_000_000,
                cache_dir: format!("{}/siglip-base", self.cache_dir),
                status: ModelStatus::NotDownloaded,
            },
            ModelAdapter {
                name: "ImageBind".into(),
                family: "imagebind".into(),
                modalities: vec![
                    Modality::Text,
                    Modality::Image,
                    Modality::Audio,
                    Modality::Video,
                ],
                dimensions: 1024,
                download_url: Some("https://huggingface.co/facebook/imagebind".into()),
                size_bytes: 1_200_000_000,
                cache_dir: format!("{}/imagebind", self.cache_dir),
                status: ModelStatus::NotDownloaded,
            },
            ModelAdapter {
                name: "Whisper Base".into(),
                family: "whisper".into(),
                modalities: vec![Modality::Audio],
                dimensions: 512,
                download_url: Some("https://huggingface.co/openai/whisper-base".into()),
                size_bytes: 140_000_000,
                cache_dir: format!("{}/whisper-base", self.cache_dir),
                status: ModelStatus::NotDownloaded,
            },
        ];
        *self.adapters.write() = defaults;
    }

    /// List all available adapters.
    pub fn list(&self) -> Vec<ModelAdapter> {
        self.adapters.read().clone()
    }

    /// Find an adapter by family name.
    pub fn find_by_family(&self, family: &str) -> Option<ModelAdapter> {
        self.adapters
            .read()
            .iter()
            .find(|a| a.family == family)
            .cloned()
    }

    /// Find adapters that support a given modality.
    pub fn find_for_modality(&self, modality: Modality) -> Vec<ModelAdapter> {
        self.adapters
            .read()
            .iter()
            .filter(|a| a.modalities.contains(&modality))
            .cloned()
            .collect()
    }

    /// Check if a model is downloaded and ready.
    pub fn is_ready(&self, family: &str) -> bool {
        self.adapters
            .read()
            .iter()
            .any(|a| a.family == family && a.status == ModelStatus::Ready)
    }

    /// Mark a model as ready (after external download completes).
    pub fn mark_ready(&self, family: &str) {
        if let Some(adapter) = self
            .adapters
            .write()
            .iter_mut()
            .find(|a| a.family == family)
        {
            adapter.status = ModelStatus::Ready;
        }
    }

    /// Register a custom adapter.
    pub fn register(&self, adapter: ModelAdapter) {
        self.adapters.write().push(adapter);
    }
}

// =============================================================================
// Dimension Alignment via Projection Matrices
// =============================================================================

/// Learned projection matrix for aligning embeddings from different modalities
/// into a shared space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityProjection {
    /// Source modality.
    pub source_modality: Modality,
    /// Source dimensions.
    pub source_dims: usize,
    /// Target (shared space) dimensions.
    pub target_dims: usize,
    /// Row-major projection matrix (source_dims x target_dims).
    pub matrix: Vec<f32>,
}

impl ModalityProjection {
    /// Create a projection from a pre-computed matrix.
    pub fn new(
        source_modality: Modality,
        source_dims: usize,
        target_dims: usize,
        matrix: Vec<f32>,
    ) -> Result<Self> {
        if matrix.len() != source_dims * target_dims {
            return Err(NeedleError::InvalidConfig(format!(
                "Projection matrix size mismatch: expected {}x{}={}, got {}",
                source_dims,
                target_dims,
                source_dims * target_dims,
                matrix.len()
            )));
        }
        Ok(Self {
            source_modality,
            source_dims,
            target_dims,
            matrix,
        })
    }

    /// Create a random projection matrix (Gaussian random projection).
    pub fn random(source_modality: Modality, source_dims: usize, target_dims: usize, seed: u64) -> Self {
        let scale = 1.0 / (target_dims as f32).sqrt();
        let mut matrix = Vec::with_capacity(source_dims * target_dims);
        let mut state = seed;
        for _ in 0..source_dims * target_dims {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let sign = if (state >> 63) == 0 { 1.0 } else { -1.0 };
            matrix.push(sign * scale);
        }
        Self {
            source_modality,
            source_dims,
            target_dims,
            matrix,
        }
    }

    /// Project a vector from source to target space.
    pub fn project(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.source_dims {
            return Err(NeedleError::DimensionMismatch {
                expected: self.source_dims,
                got: vector.len(),
            });
        }
        let mut result = vec![0.0f32; self.target_dims];
        for (j, &v) in vector.iter().enumerate() {
            for (i, val) in result.iter_mut().enumerate() {
                *val += v * self.matrix[j * self.target_dims + i];
            }
        }
        Ok(result)
    }
}

/// Configuration for multi-modal cross-search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalSearchConfig {
    /// Per-modality weights for RRF fusion.
    pub modality_weights: HashMap<Modality, f32>,
    /// RRF constant k (default: 60).
    pub rrf_k: f32,
    /// Maximum results per modality before fusion.
    pub per_modality_k: usize,
    /// Final result count after fusion.
    pub final_k: usize,
}

impl Default for MultiModalSearchConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(Modality::Text, 1.0);
        weights.insert(Modality::Image, 1.0);
        weights.insert(Modality::Audio, 0.8);
        weights.insert(Modality::Code, 0.9);
        Self {
            modality_weights: weights,
            rrf_k: 60.0,
            per_modality_k: 50,
            final_k: 10,
        }
    }
}

/// Result from a multi-modal cross-search with modality-weighted RRF fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalFusionResult {
    /// Document ID.
    pub id: String,
    /// Fused score (higher is better).
    pub score: f32,
    /// Per-modality scores.
    pub modality_scores: HashMap<Modality, f32>,
    /// Per-modality ranks.
    pub modality_ranks: HashMap<Modality, usize>,
}

/// Perform modality-weighted RRF fusion over per-modality search results.
///
/// Each input is `(modality, results)` where results are `(id, distance)` pairs
/// sorted by distance (ascending). Returns fused results sorted by score (descending).
pub fn search_multimodal(
    per_modality_results: &[(Modality, Vec<(String, f32)>)],
    config: &MultiModalSearchConfig,
) -> Vec<CrossModalFusionResult> {
    let mut id_scores: HashMap<String, CrossModalFusionResult> = HashMap::new();

    for (modality, results) in per_modality_results {
        let weight = config.modality_weights.get(modality).copied().unwrap_or(1.0);

        for (rank, (id, distance)) in results.iter().enumerate() {
            let rrf_score = weight / (config.rrf_k + rank as f32 + 1.0);

            let entry = id_scores.entry(id.clone()).or_insert_with(|| {
                CrossModalFusionResult {
                    id: id.clone(),
                    score: 0.0,
                    modality_scores: HashMap::new(),
                    modality_ranks: HashMap::new(),
                }
            });
            entry.score += rrf_score;
            entry.modality_scores.insert(*modality, 1.0 - distance); // similarity
            entry.modality_ranks.insert(*modality, rank + 1);
        }
    }

    let mut results: Vec<CrossModalFusionResult> = id_scores.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(config.final_k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_creation() {
        let config = MultiModalConfig::default();
        let embedder = MultiModalEmbedder::new(config).unwrap();

        assert_eq!(embedder.config().dimension, 512);
    }

    #[test]
    fn test_text_embedding() {
        let embedder = MultiModalEmbedder::with_backend(EmbedderBackend::Mock).unwrap();
        let embedding = embedder.embed_text("Hello, world!").unwrap();

        assert_eq!(embedding.modality, Modality::Text);
        assert_eq!(embedding.dimension(), 128); // Mock dimension
    }

    #[test]
    fn test_image_embedding() {
        let embedder = MultiModalEmbedder::with_backend(EmbedderBackend::Mock).unwrap();
        let embedding = embedder.embed_image(&[0u8; 100]).unwrap();

        assert_eq!(embedding.modality, Modality::Image);
    }

    #[test]
    fn test_batch_embedding() {
        let embedder = MultiModalEmbedder::with_backend(EmbedderBackend::Mock).unwrap();

        let inputs = vec![
            EmbedInput::text("Hello"),
            EmbedInput::text("World"),
            EmbedInput::image(vec![0u8; 50]),
        ];

        let embeddings = embedder.embed_batch(&inputs).unwrap();
        assert_eq!(embeddings.len(), 3);
    }

    #[test]
    fn test_cache() {
        let config = MultiModalConfig::default().with_backend(EmbedderBackend::Mock);
        let embedder = MultiModalEmbedder::new(config).unwrap();

        // First call - cache miss
        let _ = embedder.embed_text("test").unwrap();
        let stats = embedder.stats();
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hits, 0);

        // Second call - cache hit
        let _ = embedder.embed_text("test").unwrap();
        let stats = embedder.stats();
        assert_eq!(stats.cache_hits, 1);
    }

    #[test]
    fn test_late_fusion() {
        let embedder = MultiModalEmbedder::with_backend(EmbedderBackend::Mock).unwrap();

        let text_emb = embedder.embed_text("A cat").unwrap();
        let image_emb = embedder.embed_image(&[1, 2, 3]).unwrap();

        let fusion = LateFusion::new(128)
            .with_weight(Modality::Text, 0.7)
            .with_weight(Modality::Image, 0.3);

        let combined = fusion.combine(&[text_emb, image_emb]).unwrap();
        assert_eq!(combined.len(), 128);

        // Should be normalized
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_modality_detection() {
        assert_eq!(Modality::from_mime_type("text/plain"), Some(Modality::Text));
        assert_eq!(Modality::from_mime_type("image/png"), Some(Modality::Image));
        assert_eq!(Modality::from_mime_type("audio/wav"), Some(Modality::Audio));
        assert_eq!(Modality::from_mime_type("video/mp4"), Some(Modality::Video));
    }

    #[test]
    fn test_cross_modal_search() {
        let embedder = Arc::new(MultiModalEmbedder::with_backend(EmbedderBackend::Mock).unwrap());
        let search = CrossModalSearch::new(embedder);

        let text_query = search.text_to_image_query("A beautiful sunset").unwrap();
        assert_eq!(text_query.len(), 128);

        let image_query = search.image_to_query(&[0u8; 100]).unwrap();
        assert_eq!(image_query.len(), 128);
    }

    #[test]
    fn test_unsupported_modality() {
        let config = MultiModalConfig::default().with_backend(EmbedderBackend::OpenAI);
        let embedder = MultiModalEmbedder::new(config).unwrap();

        // OpenAI doesn't support images
        let result = embedder.embed_image(&[0u8; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_embedding_normalization() {
        let config = MultiModalConfig::default()
            .with_backend(EmbedderBackend::Mock)
            .with_normalization(true);
        let embedder = MultiModalEmbedder::new(config).unwrap();

        let embedding = embedder.embed_text("test").unwrap();

        let norm: f32 = embedding.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    fn test_stats_tracking() {
        let embedder = MultiModalEmbedder::with_backend(EmbedderBackend::Mock).unwrap();

        for i in 0..10 {
            embedder.embed_text(&format!("text {}", i)).unwrap();
        }
        for i in 0..5 {
            embedder.embed_image(&[i as u8; 50]).unwrap();
        }

        let stats = embedder.stats();
        assert_eq!(stats.total_embeddings, 15);
        assert_eq!(stats.by_modality.get("text"), Some(&10));
        assert_eq!(stats.by_modality.get("image"), Some(&5));
    }

    #[test]
    fn test_image_preprocessor() {
        let preprocessor = ImagePreprocessor::default();
        let fake_image = vec![128u8; 1024];
        let tensor = preprocessor.preprocess(&fake_image).unwrap();
        let expected_size = 3 * 224 * 224;
        assert_eq!(tensor.len(), expected_size);
    }

    #[test]
    fn test_text_preprocessor() {
        let preprocessor = TextPreprocessor::default();
        let tokens = preprocessor.preprocess("Hello World! This is a test.");
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[0], "hello"); // lowercased
    }

    #[test]
    fn test_model_registry() {
        let registry = EmbeddingModelRegistry::new();
        let models = registry.list_models();
        assert!(models.len() >= 3);

        // Find best for text+image
        let model = registry.find_model(&[Modality::Text, Modality::Image], Some(512));
        assert!(model.is_some());
        assert_eq!(model.unwrap().model_family, "CLIP");
    }

    #[test]
    fn test_model_registry_custom() {
        let registry = EmbeddingModelRegistry::new();
        registry.register(ModelManifest {
            model_id: "custom/my-model".into(),
            model_family: "Custom".into(),
            supported_modalities: vec![Modality::Text, Modality::Audio],
            output_dimensions: 256,
            max_sequence_length: Some(128),
            max_image_size: None,
            quantized: true,
            size_bytes: 50 * 1024 * 1024,
        });

        let model = registry.find_model(&[Modality::Audio], None);
        assert!(model.is_some());
        assert_eq!(model.unwrap().model_id, "custom/my-model");
    }

    #[test]
    fn test_model_adapter_registry() {
        let registry = ModelAdapterRegistry::new("/tmp/needle-models");
        let adapters = registry.list();
        assert!(adapters.len() >= 4);

        // CLIP should support text and image
        let clip = registry.find_by_family("clip").unwrap();
        assert!(clip.modalities.contains(&Modality::Text));
        assert!(clip.modalities.contains(&Modality::Image));
        assert_eq!(clip.dimensions, 512);

        // Whisper should support audio
        let whisper = registry.find_by_family("whisper").unwrap();
        assert!(whisper.modalities.contains(&Modality::Audio));
    }

    #[test]
    fn test_model_adapter_find_for_modality() {
        let registry = ModelAdapterRegistry::new("/tmp/needle-models");
        let audio_models = registry.find_for_modality(Modality::Audio);
        assert!(audio_models.len() >= 2); // ImageBind and Whisper
    }

    #[test]
    fn test_model_adapter_lifecycle() {
        let registry = ModelAdapterRegistry::new("/tmp/needle-models");
        assert!(!registry.is_ready("clip"));

        registry.mark_ready("clip");
        assert!(registry.is_ready("clip"));
    }

    #[test]
    fn test_code_modality() {
        assert_eq!(Modality::Code.name(), "code");
        assert_eq!(Modality::from_mime_type("text/x-python"), Some(Modality::Code));
        assert_eq!(Modality::from_mime_type("application/x-rust"), Some(Modality::Code));
    }

    #[test]
    fn test_modality_projection() {
        let proj = ModalityProjection::random(Modality::Image, 512, 384, 42);
        let vector = vec![0.1; 512];
        let projected = proj.project(&vector).expect("project");
        assert_eq!(projected.len(), 384);
    }

    #[test]
    fn test_modality_projection_dimension_mismatch() {
        let proj = ModalityProjection::random(Modality::Image, 512, 384, 42);
        let bad_vector = vec![0.1; 256];
        assert!(proj.project(&bad_vector).is_err());
    }

    #[test]
    fn test_search_multimodal_rrf_fusion() {
        let text_results = vec![
            ("doc1".to_string(), 0.1),
            ("doc2".to_string(), 0.3),
            ("doc3".to_string(), 0.5),
        ];
        let image_results = vec![
            ("doc2".to_string(), 0.05),
            ("doc4".to_string(), 0.2),
            ("doc1".to_string(), 0.4),
        ];

        let config = MultiModalSearchConfig {
            final_k: 3,
            ..Default::default()
        };

        let results = search_multimodal(
            &[(Modality::Text, text_results), (Modality::Image, image_results)],
            &config,
        );
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // doc1 and doc2 appear in both, should have higher scores
        let doc2 = results.iter().find(|r| r.id == "doc2");
        assert!(doc2.is_some());
        let doc2 = doc2.expect("doc2");
        assert!(doc2.modality_scores.contains_key(&Modality::Text));
        assert!(doc2.modality_scores.contains_key(&Modality::Image));
    }
}
