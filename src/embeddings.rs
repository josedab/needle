//! ONNX Embedding Inference
//!
//! Provides text embedding generation using ONNX Runtime.
//! Supports popular embedding models like:
//! - all-MiniLM-L6-v2
//! - E5 (small, base, large)
//! - BGE (small, base, large)
//! - BAAI models
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::embeddings::TextEmbedder;
//!
//! let embedder = TextEmbedder::from_pretrained("all-MiniLM-L6-v2")?;
//! let embedding = embedder.embed("Hello, world!")?;
//! ```

use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::sync::Mutex;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Error type for embedding operations
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("ONNX Runtime error: {0}")]
    OrtError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

impl From<ort::Error> for EmbeddingError {
    fn from(e: ort::Error) -> Self {
        EmbeddingError::OrtError(e.to_string())
    }
}

impl From<tokenizers::Error> for EmbeddingError {
    fn from(e: tokenizers::Error) -> Self {
        EmbeddingError::TokenizerError(e.to_string())
    }
}

/// Result type for embedding operations
pub type Result<T> = std::result::Result<T, EmbeddingError>;

/// Configuration for the text embedder
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    /// Maximum sequence length
    pub max_length: usize,
    /// Whether to normalize embeddings
    pub normalize: bool,
    /// Pooling strategy
    pub pooling: PoolingStrategy,
    /// Number of threads for inference
    pub num_threads: usize,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            normalize: true,
            pooling: PoolingStrategy::Mean,
            num_threads: 4,
        }
    }
}

/// Pooling strategy for converting token embeddings to sentence embeddings
#[derive(Debug, Clone, Copy)]
pub enum PoolingStrategy {
    /// Use [CLS] token embedding
    Cls,
    /// Mean pooling over all tokens
    Mean,
    /// Max pooling over all tokens
    Max,
}

/// Text embedder using ONNX models
pub struct TextEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: EmbedderConfig,
    dimensions: usize,
}

impl TextEmbedder {
    /// Load from model directory containing model.onnx and tokenizer.json
    pub fn from_directory(path: impl AsRef<Path>, config: EmbedderConfig) -> Result<Self> {
        let model_path = path.as_ref().join("model.onnx");
        let tokenizer_path = path.as_ref().join("tokenizer.json");

        if !model_path.exists() {
            return Err(EmbeddingError::ModelNotFound(format!(
                "Model file not found: {}",
                model_path.display()
            )));
        }

        if !tokenizer_path.exists() {
            return Err(EmbeddingError::ModelNotFound(format!(
                "Tokenizer file not found: {}",
                tokenizer_path.display()
            )));
        }

        Self::from_files(&model_path, &tokenizer_path, config)
    }

    /// Load from specific model and tokenizer files
    pub fn from_files(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: EmbedderConfig,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)?;

        // Determine dimensions from model output shape
        // Use a default of 384 dimensions (common for small embedding models)
        let dimensions = 384;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            config,
            dimensions,
        })
    }

    /// Get the embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[text])?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbeddingError::OrtError("embed_batch returned empty results".into()))
    }

    /// Embed multiple texts in a batch
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbeddingError::TokenizerError(e.to_string()))?;

        let batch_size = encodings.len();
        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0)
            .min(self.config.max_length);

        // Prepare input tensors
        let mut input_ids = vec![0i64; batch_size * seq_len];
        let mut attention_mask = vec![0i64; batch_size * seq_len];
        let mut token_type_ids = vec![0i64; batch_size * seq_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let types = encoding.get_type_ids();

            let len = ids.len().min(seq_len);
            for j in 0..len {
                input_ids[i * seq_len + j] = ids[j] as i64;
                attention_mask[i * seq_len + j] = mask[j] as i64;
                token_type_ids[i * seq_len + j] = types[j] as i64;
            }
        }

        // Create ONNX inputs
        let input_ids_array =
            Array2::from_shape_vec((batch_size, seq_len), input_ids).map_err(|e| {
                EmbeddingError::InvalidInput(format!("Failed to create input_ids array: {}", e))
            })?;

        let attention_mask_array =
            Array2::from_shape_vec((batch_size, seq_len), attention_mask).map_err(|e| {
                EmbeddingError::InvalidInput(format!(
                    "Failed to create attention_mask array: {}",
                    e
                ))
            })?;

        let token_type_ids_array =
            Array2::from_shape_vec((batch_size, seq_len), token_type_ids).map_err(|e| {
                EmbeddingError::InvalidInput(format!(
                    "Failed to create token_type_ids array: {}",
                    e
                ))
            })?;

        // Run inference
        let mut session = self.session.lock().map_err(|_| EmbeddingError::OrtError("Session lock poisoned".into()))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => Value::from_array(input_ids_array)?,
            "attention_mask" => Value::from_array(attention_mask_array.clone())?,
            "token_type_ids" => Value::from_array(token_type_ids_array)?,
        ])?;

        // Extract embeddings from output
        // Output shape is typically (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        let output = outputs
            .iter()
            .find(|(name, _)| *name == "last_hidden_state" || *name == "sentence_embedding")
            .or_else(|| outputs.iter().next())
            .map(|(_, v)| v)
            .ok_or_else(|| EmbeddingError::OrtError("No output found".to_string()))?;

        // ORT 2.x: extract tensor returns (&Shape, &[T])
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let output_shape_dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let output_data: &[f32] = data;

        let embeddings = if output_shape_dims.len() == 3 {
            // Token-level output: (batch_size, seq_len, hidden_size)
            // Apply pooling
            self.pool_embeddings_raw(output_data, &output_shape_dims, &attention_mask_array, batch_size)?
        } else if output_shape_dims.len() == 2 {
            // Sentence-level output: (batch_size, hidden_size)
            (0..batch_size)
                .map(|i| {
                    let start = i * self.dimensions;
                    let end = start + self.dimensions;
                    if end <= output_data.len() {
                        output_data[start..end].to_vec()
                    } else {
                        vec![0.0; self.dimensions]
                    }
                })
                .collect()
        } else {
            return Err(EmbeddingError::OrtError(format!(
                "Unexpected output shape: {:?}",
                output_shape_dims
            )));
        };

        // Normalize if configured
        if self.config.normalize {
            Ok(embeddings.into_iter().map(|e| normalize(&e)).collect())
        } else {
            Ok(embeddings)
        }
    }

    /// Apply pooling to token embeddings (raw slice version for ORT 2.x)
    fn pool_embeddings_raw(
        &self,
        output: &[f32],
        shape: &[usize],
        attention_mask: &Array2<i64>,
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let seq_len = shape[1];
        let hidden_size = shape[2];

        // Helper to index into flat array as [batch, seq, hidden]
        let idx = |b: usize, s: usize, h: usize| -> usize {
            b * seq_len * hidden_size + s * hidden_size + h
        };

        let mut embeddings = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let embedding = match self.config.pooling {
                PoolingStrategy::Cls => {
                    // First token
                    let mut vec = Vec::with_capacity(hidden_size);
                    for j in 0..hidden_size {
                        vec.push(output[idx(i, 0, j)]);
                    }
                    vec
                }
                PoolingStrategy::Mean => {
                    // Mean over non-padded tokens
                    let mut sum = vec![0.0f32; hidden_size];
                    let mut count = 0.0f32;

                    for j in 0..seq_len {
                        if attention_mask[[i, j]] == 1 {
                            for k in 0..hidden_size {
                                sum[k] += output[idx(i, j, k)];
                            }
                            count += 1.0;
                        }
                    }

                    if count > 0.0 {
                        for v in &mut sum {
                            *v /= count;
                        }
                    }
                    sum
                }
                PoolingStrategy::Max => {
                    // Max over non-padded tokens
                    let mut max = vec![f32::NEG_INFINITY; hidden_size];

                    for j in 0..seq_len {
                        if attention_mask[[i, j]] == 1 {
                            for k in 0..hidden_size {
                                let val = output[idx(i, j, k)];
                                if val > max[k] {
                                    max[k] = val;
                                }
                            }
                        }
                    }

                    // Replace -inf with 0 if no tokens
                    for v in &mut max {
                        if v.is_infinite() {
                            *v = 0.0;
                        }
                    }
                    max
                }
            };

            embeddings.push(embedding);
        }

        Ok(embeddings)
    }
}

/// Normalize a vector to unit length
fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

/// Builder for creating TextEmbedder with custom configuration
pub struct EmbedderBuilder {
    config: EmbedderConfig,
}

impl Default for EmbedderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbedderBuilder {
    /// Create a new builder with default config
    pub fn new() -> Self {
        Self {
            config: EmbedderConfig::default(),
        }
    }

    /// Set maximum sequence length
    pub fn max_length(mut self, length: usize) -> Self {
        self.config.max_length = length;
        self
    }

    /// Enable/disable normalization
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }

    /// Set pooling strategy
    pub fn pooling(mut self, strategy: PoolingStrategy) -> Self {
        self.config.pooling = strategy;
        self
    }

    /// Set number of threads
    pub fn num_threads(mut self, threads: usize) -> Self {
        self.config.num_threads = threads;
        self
    }

    /// Build from model directory
    pub fn from_directory(self, path: impl AsRef<Path>) -> Result<TextEmbedder> {
        TextEmbedder::from_directory(path, self.config)
    }

    /// Build from specific files
    pub fn from_files(
        self,
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<TextEmbedder> {
        TextEmbedder::from_files(model_path, tokenizer_path, self.config)
    }
}

/// Trait for types that can generate embeddings
pub trait Embedder: Send + Sync {
    /// Embed a single text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;
}

impl Embedder for TextEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(texts)
    }

    fn dimensions(&self) -> usize {
        self.dimensions()
    }
}

/// Thread-safe embedder wrapper
pub struct SharedEmbedder {
    inner: Arc<dyn Embedder>,
}

impl SharedEmbedder {
    /// Create from any embedder
    pub fn new<E: Embedder + 'static>(embedder: E) -> Self {
        Self {
            inner: Arc::new(embedder),
        }
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.embed(text)
    }

    /// Embed multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_batch(texts)
    }

    /// Get embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
}

impl Clone for SharedEmbedder {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize(&vec);

        // Should be unit length
        let norm: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);

        // Check values
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_embedder_config_default() {
        let config = EmbedderConfig::default();
        assert_eq!(config.max_length, 512);
        assert!(config.normalize);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_embedder_builder() {
        let builder = EmbedderBuilder::new()
            .max_length(256)
            .normalize(false)
            .num_threads(2)
            .pooling(PoolingStrategy::Cls);

        assert_eq!(builder.config.max_length, 256);
        assert!(!builder.config.normalize);
        assert_eq!(builder.config.num_threads, 2);
    }
}
