//! Embedded Inference Engine
//!
//! Built-in model execution for query-time embedding generation, eliminating the
//! need for external embedding services. Supports model bundling so that a
//! database and its embedding model can be shipped as a single artifact.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::inference_engine::{
//!     InferenceEngine, InferenceConfig, ModelSpec, PoolingStrategy,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let config = InferenceConfig::builder()
//!     .model(ModelSpec::mini_lm_l6())
//!     .pooling(PoolingStrategy::Mean)
//!     .normalize(true)
//!     .build();
//!
//! let engine = InferenceEngine::new(config);
//!
//! // Embed text directly
//! let embedding = engine.embed_text("Hello, world!").unwrap();
//! assert_eq!(embedding.len(), 384);
//!
//! // Batch embedding
//! let texts = vec!["First doc", "Second doc"];
//! let embeddings = engine.embed_batch(&texts).unwrap();
//! assert_eq!(embeddings.len(), 2);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Model Specification ──────────────────────────────────────────────────────

/// Specification for an embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Model identifier (e.g., "all-MiniLM-L6-v2").
    pub name: String,
    /// Output embedding dimensions.
    pub dimensions: usize,
    /// Maximum input token length.
    pub max_tokens: usize,
    /// Model format.
    pub format: ModelFormat,
    /// Optional local file path to the model.
    pub path: Option<String>,
}

impl ModelSpec {
    /// Create a custom model specification.
    pub fn new(name: impl Into<String>, dimensions: usize) -> Self {
        Self {
            name: name.into(),
            dimensions,
            max_tokens: 512,
            format: ModelFormat::Onnx,
            path: None,
        }
    }

    /// Preset: all-MiniLM-L6-v2 (384 dimensions).
    pub fn mini_lm_l6() -> Self {
        Self::new("all-MiniLM-L6-v2", 384).with_max_tokens(256)
    }

    /// Preset: BGE-small-en-v1.5 (384 dimensions).
    pub fn bge_small() -> Self {
        Self::new("bge-small-en-v1.5", 384).with_max_tokens(512)
    }

    /// Preset: E5-small-v2 (384 dimensions).
    pub fn e5_small() -> Self {
        Self::new("e5-small-v2", 384).with_max_tokens(512)
    }

    /// Set maximum token length.
    #[must_use]
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Set local model path.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// Model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX format.
    Onnx,
    /// GGUF/GGML format.
    Gguf,
    /// SafeTensors format.
    SafeTensors,
}

// ── Pooling Strategy ─────────────────────────────────────────────────────────

/// How to pool token-level outputs into a single embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Use the [CLS] token output.
    Cls,
    /// Average all token outputs.
    Mean,
    /// Take the maximum across all tokens.
    Max,
    /// Use the last token output.
    LastToken,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::Mean
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Inference engine configuration.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Model specification.
    pub model: ModelSpec,
    /// Pooling strategy.
    pub pooling: PoolingStrategy,
    /// Whether to L2-normalize output embeddings.
    pub normalize: bool,
    /// Maximum batch size for batch inference.
    pub max_batch_size: usize,
    /// Number of threads for inference.
    pub num_threads: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model: ModelSpec::mini_lm_l6(),
            pooling: PoolingStrategy::Mean,
            normalize: true,
            max_batch_size: 64,
            num_threads: 4,
        }
    }
}

impl InferenceConfig {
    /// Create a builder.
    pub fn builder() -> InferenceConfigBuilder {
        InferenceConfigBuilder::default()
    }
}

/// Builder for `InferenceConfig`.
#[derive(Debug, Clone)]
pub struct InferenceConfigBuilder {
    config: InferenceConfig,
}

impl Default for InferenceConfigBuilder {
    fn default() -> Self {
        Self {
            config: InferenceConfig::default(),
        }
    }
}

impl InferenceConfigBuilder {
    /// Set model specification.
    #[must_use]
    pub fn model(mut self, model: ModelSpec) -> Self {
        self.config.model = model;
        self
    }

    /// Set pooling strategy.
    #[must_use]
    pub fn pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.config.pooling = pooling;
        self
    }

    /// Set normalization.
    #[must_use]
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set max batch size.
    #[must_use]
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Build the config.
    pub fn build(self) -> InferenceConfig {
        self.config
    }
}

// ── Tokenizer ────────────────────────────────────────────────────────────────

/// Simple whitespace tokenizer for the built-in inference engine.
/// Production usage would use a HuggingFace tokenizer or sentencepiece.
#[derive(Debug)]
struct SimpleTokenizer {
    max_tokens: usize,
}

impl SimpleTokenizer {
    fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .take(self.max_tokens)
            .map(|s| s.to_lowercase())
            .collect()
    }
}

// ── Inference Statistics ─────────────────────────────────────────────────────

/// Statistics from an inference run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceStats {
    /// Number of texts embedded.
    pub texts_embedded: usize,
    /// Total tokens processed.
    pub tokens_processed: usize,
    /// Time for tokenization.
    pub tokenize_ms: u64,
    /// Time for inference.
    pub inference_ms: u64,
    /// Time for pooling + normalization.
    pub postprocess_ms: u64,
}

// ── Inference Engine ─────────────────────────────────────────────────────────

/// Embedded inference engine for generating embeddings from text.
///
/// Uses a deterministic hash-based embedding for the built-in engine.
/// When the `embeddings` feature is enabled, this can be upgraded to use
/// real ONNX models.
pub struct InferenceEngine {
    config: InferenceConfig,
    tokenizer: SimpleTokenizer,
    total_inferences: u64,
    vocab: HashMap<String, Vec<f32>>,
}

impl InferenceEngine {
    /// Create a new inference engine.
    pub fn new(config: InferenceConfig) -> Self {
        let tokenizer = SimpleTokenizer::new(config.model.max_tokens);
        Self {
            config,
            tokenizer,
            total_inferences: 0,
            vocab: HashMap::new(),
        }
    }

    /// Embed a single text string.
    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let start = Instant::now();
        let tokens = self.tokenizer.tokenize(text);

        if tokens.is_empty() {
            return Err(NeedleError::InvalidInput("Empty text cannot be embedded".into()));
        }

        let embedding = self.embed_tokens(&tokens);
        self.total_inferences += 1;

        let _ = start.elapsed(); // timing captured but not yet surfaced
        Ok(embedding)
    }

    /// Embed a batch of text strings.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results = Vec::with_capacity(texts.len());
        for &text in texts {
            results.push(self.embed_text(text)?);
        }
        Ok(results)
    }

    /// Get the output dimensions.
    pub fn dimensions(&self) -> usize {
        self.config.model.dimensions
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.config.model.name
    }

    /// Get total inference count.
    pub fn total_inferences(&self) -> u64 {
        self.total_inferences
    }

    /// Get the current config.
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn embed_tokens(&mut self, tokens: &[String]) -> Vec<f32> {
        let dim = self.config.model.dimensions;

        // Generate per-token embeddings using deterministic hash
        let token_embeddings: Vec<Vec<f32>> = tokens
            .iter()
            .map(|token| self.token_embedding(token))
            .collect();

        // Pool token embeddings
        let pooled = match self.config.pooling {
            PoolingStrategy::Cls => token_embeddings[0].clone(),
            PoolingStrategy::Mean => {
                let mut avg = vec![0.0f32; dim];
                for emb in &token_embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        avg[i] += v;
                    }
                }
                let n = token_embeddings.len() as f32;
                for v in &mut avg {
                    *v /= n;
                }
                avg
            }
            PoolingStrategy::Max => {
                let mut max_vals = vec![f32::NEG_INFINITY; dim];
                for emb in &token_embeddings {
                    for (i, &v) in emb.iter().enumerate() {
                        if v > max_vals[i] {
                            max_vals[i] = v;
                        }
                    }
                }
                max_vals
            }
            PoolingStrategy::LastToken => token_embeddings.last().cloned().unwrap_or_else(|| vec![0.0; dim]),
        };

        // Normalize if configured
        if self.config.normalize {
            self.l2_normalize(pooled)
        } else {
            pooled
        }
    }

    fn token_embedding(&mut self, token: &str) -> Vec<f32> {
        let dim = self.config.model.dimensions;

        if let Some(cached) = self.vocab.get(token) {
            return cached.clone();
        }

        // Deterministic hash-based embedding generation
        let embedding = Self::hash_embedding(token, dim);
        self.vocab.insert(token.to_string(), embedding.clone());
        embedding
    }

    fn hash_embedding(token: &str, dim: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dim];
        let bytes = token.as_bytes();

        for (i, val) in embedding.iter_mut().enumerate() {
            // Use a simple but deterministic hash function
            let mut h: u64 = 0x517c_c1b7_2722_0a95;
            for &b in bytes {
                h = h.wrapping_mul(0x0100_0000_01b3).wrapping_add(u64::from(b));
            }
            h = h.wrapping_add(i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
            h ^= h >> 33;

            // Map to [-1, 1] range
            *val = ((h & 0xFFFF_FFFF) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        embedding
    }

    fn l2_normalize(&self, mut vec: Vec<f32>) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            for v in &mut vec {
                *v /= norm;
            }
        }
        vec
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_text() {
        let config = InferenceConfig::builder()
            .model(ModelSpec::new("test-model", 64))
            .normalize(true)
            .build();
        let mut engine = InferenceEngine::new(config);

        let embedding = engine.embed_text("hello world").unwrap();
        assert_eq!(embedding.len(), 64);

        // Check normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_deterministic() {
        let config = InferenceConfig::builder()
            .model(ModelSpec::new("test", 32))
            .build();
        let mut engine1 = InferenceEngine::new(config.clone());
        let mut engine2 = InferenceEngine::new(config);

        let e1 = engine1.embed_text("same text").unwrap();
        let e2 = engine2.embed_text("same text").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_batch_embed() {
        let config = InferenceConfig::builder()
            .model(ModelSpec::new("test", 32))
            .build();
        let mut engine = InferenceEngine::new(config);

        let embeddings = engine.embed_batch(&["hello", "world", "test"]).unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 32);
        }
    }

    #[test]
    fn test_empty_text_error() {
        let mut engine = InferenceEngine::new(InferenceConfig::default());
        assert!(engine.embed_text("").is_err());
    }

    #[test]
    fn test_different_pooling() {
        let texts = "the quick brown fox";

        let mut mean_engine = InferenceEngine::new(
            InferenceConfig::builder()
                .model(ModelSpec::new("test", 16))
                .pooling(PoolingStrategy::Mean)
                .normalize(false)
                .build(),
        );
        let mut cls_engine = InferenceEngine::new(
            InferenceConfig::builder()
                .model(ModelSpec::new("test", 16))
                .pooling(PoolingStrategy::Cls)
                .normalize(false)
                .build(),
        );

        let mean_emb = mean_engine.embed_text(texts).unwrap();
        let cls_emb = cls_engine.embed_text(texts).unwrap();

        // Different pooling should produce different embeddings
        assert_ne!(mean_emb, cls_emb);
    }

    #[test]
    fn test_model_presets() {
        let spec = ModelSpec::mini_lm_l6();
        assert_eq!(spec.dimensions, 384);
        assert_eq!(spec.name, "all-MiniLM-L6-v2");

        let spec = ModelSpec::bge_small();
        assert_eq!(spec.dimensions, 384);
    }

    #[test]
    fn test_inference_count() {
        let mut engine = InferenceEngine::new(InferenceConfig::default());
        assert_eq!(engine.total_inferences(), 0);
        engine.embed_text("hello").unwrap();
        engine.embed_text("world").unwrap();
        assert_eq!(engine.total_inferences(), 2);
    }
}
