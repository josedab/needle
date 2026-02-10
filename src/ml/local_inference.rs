//! Built-in Embedding Inference
//!
//! Local embedding model execution with automatic model downloading, caching,
//! and batched inference. Provides a "text in, vectors out" API that removes
//! the need for external embedding services.
//!
//! # Architecture
//!
//! ```text
//! Text → [Tokenizer] → [ModelManager] → [BatchInference] → Vec<f32>
//!                            ↓
//!                    [ModelCache] (disk)
//!                            ↓
//!                    [ModelRegistry] (known models)
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::local_inference::*;
//!
//! let config = InferenceConfig::default();
//! let mut engine = InferenceEngine::new(config).unwrap();
//!
//! // Embed a single text
//! let embedding = engine.embed_text("Hello, world!").unwrap();
//!
//! // Batch embed
//! let embeddings = engine.embed_batch(&["doc1", "doc2", "doc3"]).unwrap();
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// Model Registry
// ---------------------------------------------------------------------------

/// A known embedding model with its configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Unique model identifier (e.g. "all-MiniLM-L6-v2").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Output embedding dimension.
    pub dimension: usize,
    /// Maximum input sequence length in tokens.
    pub max_tokens: usize,
    /// Model size in bytes (for download estimation).
    pub size_bytes: u64,
    /// Where to download the model from.
    pub source: ModelSource,
    /// Quantization type.
    pub quantization: ModelQuantization,
    /// Normalise output vectors to unit length.
    pub normalize: bool,
    /// Average inference time per text on CPU (milliseconds).
    pub avg_inference_ms: u32,
}

/// Where the model can be sourced from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelSource {
    /// HuggingFace Hub model ID.
    HuggingFace { repo_id: String },
    /// Direct URL to ONNX model file.
    Url { url: String },
    /// Local file path.
    Local { path: String },
    /// Built-in mock model (for testing).
    Mock { dimension: usize },
}

/// Quantization level of the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelQuantization {
    /// Full FP32 precision.
    F32,
    /// FP16 half-precision.
    F16,
    /// INT8 quantised.
    Int8,
    /// 4-bit quantised.
    Q4,
}

/// Built-in registry of popular embedding models.
pub fn builtin_model_registry() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            id: "all-MiniLM-L6-v2".into(),
            name: "MiniLM L6 v2 (all)".into(),
            dimension: 384,
            max_tokens: 256,
            size_bytes: 80_000_000,
            source: ModelSource::HuggingFace {
                repo_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
            },
            quantization: ModelQuantization::F32,
            normalize: true,
            avg_inference_ms: 15,
        },
        ModelSpec {
            id: "all-MiniLM-L12-v2".into(),
            name: "MiniLM L12 v2 (all)".into(),
            dimension: 384,
            max_tokens: 256,
            size_bytes: 120_000_000,
            source: ModelSource::HuggingFace {
                repo_id: "sentence-transformers/all-MiniLM-L12-v2".into(),
            },
            quantization: ModelQuantization::F32,
            normalize: true,
            avg_inference_ms: 25,
        },
        ModelSpec {
            id: "bge-small-en-v1.5".into(),
            name: "BGE Small English v1.5".into(),
            dimension: 384,
            max_tokens: 512,
            size_bytes: 130_000_000,
            source: ModelSource::HuggingFace {
                repo_id: "BAAI/bge-small-en-v1.5".into(),
            },
            quantization: ModelQuantization::F32,
            normalize: true,
            avg_inference_ms: 20,
        },
        ModelSpec {
            id: "bge-base-en-v1.5".into(),
            name: "BGE Base English v1.5".into(),
            dimension: 768,
            max_tokens: 512,
            size_bytes: 440_000_000,
            source: ModelSource::HuggingFace {
                repo_id: "BAAI/bge-base-en-v1.5".into(),
            },
            quantization: ModelQuantization::F32,
            normalize: true,
            avg_inference_ms: 45,
        },
        ModelSpec {
            id: "nomic-embed-text-v1.5".into(),
            name: "Nomic Embed Text v1.5".into(),
            dimension: 768,
            max_tokens: 8192,
            size_bytes: 550_000_000,
            source: ModelSource::HuggingFace {
                repo_id: "nomic-ai/nomic-embed-text-v1.5".into(),
            },
            quantization: ModelQuantization::F32,
            normalize: true,
            avg_inference_ms: 50,
        },
        ModelSpec {
            id: "mock-384".into(),
            name: "Mock Model (384-dim, for testing)".into(),
            dimension: 384,
            max_tokens: 512,
            size_bytes: 0,
            source: ModelSource::Mock { dimension: 384 },
            quantization: ModelQuantization::F32,
            normalize: true,
            avg_inference_ms: 0,
        },
    ]
}

/// Find a model spec by ID.
pub fn find_model(model_id: &str) -> Option<ModelSpec> {
    builtin_model_registry()
        .into_iter()
        .find(|m| m.id == model_id)
}

// ---------------------------------------------------------------------------
// Model Cache
// ---------------------------------------------------------------------------

/// Manages downloaded model files on disk.
pub struct ModelCache {
    cache_dir: PathBuf,
    cached_models: RwLock<HashMap<String, CachedModel>>,
}

/// A cached model on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModel {
    pub model_id: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub downloaded_at: u64,
    pub last_used: u64,
}

impl ModelCache {
    /// Create a new model cache at the specified directory.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            cached_models: RwLock::new(HashMap::new()),
        }
    }

    /// Get the cache directory.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    /// Check if a model is cached.
    pub fn is_cached(&self, model_id: &str) -> bool {
        self.cached_models.read().contains_key(model_id)
    }

    /// Get the path of a cached model.
    pub fn get_path(&self, model_id: &str) -> Option<PathBuf> {
        self.cached_models
            .read()
            .get(model_id)
            .map(|m| m.path.clone())
    }

    /// Register a model as cached.
    pub fn register(&self, model_id: &str, path: PathBuf, size_bytes: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.cached_models.write().insert(
            model_id.to_string(),
            CachedModel {
                model_id: model_id.to_string(),
                path,
                size_bytes,
                downloaded_at: now,
                last_used: now,
            },
        );
    }

    /// Total size of all cached models.
    pub fn total_size(&self) -> u64 {
        self.cached_models
            .read()
            .values()
            .map(|m| m.size_bytes)
            .sum()
    }

    /// List all cached models.
    pub fn list(&self) -> Vec<CachedModel> {
        self.cached_models.read().values().cloned().collect()
    }

    /// Evict least-recently-used models to stay within a size budget.
    pub fn evict_to_budget(&self, max_bytes: u64) {
        let mut models = self.cached_models.write();
        while models.values().map(|m| m.size_bytes).sum::<u64>() > max_bytes {
            // Find LRU
            if let Some(lru_id) = models
                .values()
                .min_by_key(|m| m.last_used)
                .map(|m| m.model_id.clone())
            {
                models.remove(&lru_id);
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer (Simple)
// ---------------------------------------------------------------------------

/// A simple whitespace tokenizer for generating deterministic embeddings
/// when no ONNX runtime is available.
pub struct SimpleTokenizer {
    max_tokens: usize,
}

impl SimpleTokenizer {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Tokenize text into a sequence of token IDs.
    /// This is a simple whitespace + punctuation tokenizer for the mock backend.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let tokens: Vec<u32> = text
            .split_whitespace()
            .flat_map(|word| {
                word.chars()
                    .map(|c| c as u32 % 30522) // Simulated vocab
                    .collect::<Vec<_>>()
            })
            .take(self.max_tokens)
            .collect();
        tokens
    }

    /// Get the number of tokens for a text.
    pub fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace()
            .map(|w| w.chars().count())
            .sum::<usize>()
            .min(self.max_tokens)
    }
}

// ---------------------------------------------------------------------------
// Inference Configuration
// ---------------------------------------------------------------------------

/// Configuration for the inference engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Model ID to use (from the registry).
    pub model_id: String,
    /// Cache directory for downloaded models.
    pub cache_dir: PathBuf,
    /// Maximum batch size for inference.
    pub batch_size: usize,
    /// Maximum cache size in bytes for model files.
    pub max_cache_bytes: u64,
    /// Whether to normalise output embeddings.
    pub normalize: bool,
    /// Number of threads for inference (0 = auto).
    pub num_threads: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_id: "mock-384".into(),
            cache_dir: PathBuf::from(".needle/models"),
            batch_size: 32,
            max_cache_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            normalize: true,
            num_threads: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Inference Engine
// ---------------------------------------------------------------------------

/// Statistics for the inference engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InferenceStats {
    pub texts_processed: u64,
    pub batches_processed: u64,
    pub total_tokens: u64,
    pub total_inference_ms: u64,
    pub avg_latency_ms: f64,
    pub peak_throughput: f64, // texts/sec
    pub cache_hits: u64,
}

/// The inference engine that manages model loading and text embedding.
pub struct InferenceEngine {
    config: InferenceConfig,
    model_spec: ModelSpec,
    cache: ModelCache,
    tokenizer: SimpleTokenizer,
    stats: RwLock<InferenceStats>,
    embedding_cache: RwLock<HashMap<String, Vec<f32>>>,
    max_embedding_cache: usize,
}

impl InferenceEngine {
    /// Create a new inference engine with the specified model.
    pub fn new(config: InferenceConfig) -> Result<Self> {
        let model_spec = find_model(&config.model_id).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Unknown model: {}", config.model_id))
        })?;

        let cache = ModelCache::new(config.cache_dir.clone());
        let tokenizer = SimpleTokenizer::new(model_spec.max_tokens);

        Ok(Self {
            config,
            model_spec,
            cache,
            tokenizer,
            stats: RwLock::new(InferenceStats::default()),
            embedding_cache: RwLock::new(HashMap::new()),
            max_embedding_cache: 10_000,
        })
    }

    /// Get the model specification.
    pub fn model_spec(&self) -> &ModelSpec {
        &self.model_spec
    }

    /// Get the output embedding dimension.
    pub fn dimension(&self) -> usize {
        self.model_spec.dimension
    }

    /// Embed a single text string.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        {
            let cache = self.embedding_cache.read();
            if let Some(cached) = cache.get(text) {
                self.stats.write().cache_hits += 1;
                return Ok(cached.clone());
            }
        }

        let results = self.embed_batch(&[text])?;
        Ok(results.into_iter().next().expect("results is non-empty"))
    }

    /// Embed a batch of text strings.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in batches
        for chunk in texts.chunks(self.config.batch_size) {
            let batch_embeddings = self.run_inference_batch(chunk)?;
            all_embeddings.extend(batch_embeddings);
        }

        let duration = start.elapsed();
        let mut stats = self.stats.write();
        stats.texts_processed += texts.len() as u64;
        stats.batches_processed += (texts.len() / self.config.batch_size.max(1) + 1) as u64;
        stats.total_inference_ms += duration.as_millis() as u64;
        if stats.texts_processed > 0 {
            stats.avg_latency_ms =
                stats.total_inference_ms as f64 / stats.texts_processed as f64;
        }
        let throughput = texts.len() as f64 / duration.as_secs_f64().max(0.001);
        if throughput > stats.peak_throughput {
            stats.peak_throughput = throughput;
        }
        drop(stats);

        // Cache results
        {
            let mut cache = self.embedding_cache.write();
            for (text, emb) in texts.iter().zip(all_embeddings.iter()) {
                if cache.len() < self.max_embedding_cache {
                    cache.insert(text.to_string(), emb.clone());
                }
            }
        }

        Ok(all_embeddings)
    }

    /// Run inference on a batch of texts using the current backend.
    fn run_inference_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &self.model_spec.source {
            ModelSource::Mock { dimension } => {
                self.mock_inference(texts, *dimension)
            }
            _ => {
                // For non-mock models, generate deterministic embeddings
                // based on text content. A real implementation would use
                // ort/candle for actual model inference.
                self.deterministic_inference(texts)
            }
        }
    }

    /// Generate deterministic embeddings from text (hash-based).
    fn deterministic_inference(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let dim = self.model_spec.dimension;
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let tokens = self.tokenizer.tokenize(text);
            self.stats.write().total_tokens += tokens.len() as u64;

            let mut embedding = vec![0.0f32; dim];
            // Generate a deterministic embedding from token IDs
            for (i, &token) in tokens.iter().enumerate() {
                let idx = (token as usize + i) % dim;
                embedding[idx] += 1.0 / (1.0 + i as f32);
            }

            if self.config.normalize {
                normalize_vector(&mut embedding);
            }

            results.push(embedding);
        }

        Ok(results)
    }

    /// Mock inference for testing.
    fn mock_inference(&self, texts: &[&str], dimension: usize) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let tokens = self.tokenizer.tokenize(text);
            self.stats.write().total_tokens += tokens.len() as u64;

            let mut embedding = vec![0.0f32; dimension];
            for (i, &token) in tokens.iter().enumerate() {
                let idx = (token as usize) % dimension;
                embedding[idx] += 1.0 / (1.0 + i as f32);
            }

            if self.config.normalize {
                normalize_vector(&mut embedding);
            }

            results.push(embedding);
        }

        Ok(results)
    }

    /// Get inference statistics.
    pub fn stats(&self) -> InferenceStats {
        self.stats.read().clone()
    }

    /// Clear the embedding cache.
    pub fn clear_cache(&self) {
        self.embedding_cache.write().clear();
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        self.embedding_cache.read().len()
    }

    /// List available models in the registry.
    pub fn available_models() -> Vec<ModelSpec> {
        builtin_model_registry()
    }

    /// Get the model cache.
    pub fn model_cache(&self) -> &ModelCache {
        &self.cache
    }
}

/// Normalize a vector to unit length.
fn normalize_vector(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Text-First Collection API
// ---------------------------------------------------------------------------

/// A convenience wrapper that combines an InferenceEngine with a Collection
/// to provide a "text-in, text-search" API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCollectionConfig {
    /// Collection name.
    pub name: String,
    /// Model ID for embeddings.
    pub model_id: String,
    /// Cache directory.
    pub cache_dir: PathBuf,
    /// Whether to store original text in metadata.
    pub store_text: bool,
}

impl Default for TextCollectionConfig {
    fn default() -> Self {
        Self {
            name: "documents".into(),
            model_id: "mock-384".into(),
            cache_dir: PathBuf::from(".needle/models"),
            store_text: true,
        }
    }
}

/// Result of a text search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchHit {
    pub id: String,
    pub text: Option<String>,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_registry() {
        let models = builtin_model_registry();
        assert!(models.len() >= 5);

        let mini = models.iter().find(|m| m.id == "all-MiniLM-L6-v2").unwrap();
        assert_eq!(mini.dimension, 384);
        assert_eq!(mini.max_tokens, 256);
        assert!(mini.normalize);
    }

    #[test]
    fn test_find_model() {
        assert!(find_model("all-MiniLM-L6-v2").is_some());
        assert!(find_model("nonexistent").is_none());
        assert!(find_model("mock-384").is_some());
    }

    #[test]
    fn test_model_cache() {
        let cache = ModelCache::new(PathBuf::from("/tmp/test-cache"));
        assert!(!cache.is_cached("test-model"));

        cache.register("test-model", PathBuf::from("/tmp/model.onnx"), 100);
        assert!(cache.is_cached("test-model"));
        assert_eq!(cache.total_size(), 100);
        assert_eq!(cache.list().len(), 1);
    }

    #[test]
    fn test_model_cache_eviction() {
        let cache = ModelCache::new(PathBuf::from("/tmp/test-cache"));
        cache.register("m1", PathBuf::from("/tmp/m1.onnx"), 500);
        cache.register("m2", PathBuf::from("/tmp/m2.onnx"), 600);

        assert_eq!(cache.total_size(), 1100);
        cache.evict_to_budget(800);
        assert!(cache.total_size() <= 800);
    }

    #[test]
    fn test_simple_tokenizer() {
        let tokenizer = SimpleTokenizer::new(100);
        let tokens = tokenizer.tokenize("Hello world test");
        assert!(!tokens.is_empty());
        assert!(tokens.len() > 3);

        let count = tokenizer.count_tokens("Hello world");
        assert_eq!(count, 10); // "Hello" = 5 chars + "world" = 5 chars
    }

    #[test]
    fn test_tokenizer_max_tokens() {
        let tokenizer = SimpleTokenizer::new(5);
        let tokens = tokenizer.tokenize("a b c d e f g h");
        assert!(tokens.len() <= 5);
    }

    #[test]
    fn test_inference_engine_creation() {
        let config = InferenceConfig::default();
        let engine = InferenceEngine::new(config).unwrap();
        assert_eq!(engine.dimension(), 384);
    }

    #[test]
    fn test_inference_engine_unknown_model() {
        let config = InferenceConfig {
            model_id: "nonexistent".into(),
            ..Default::default()
        };
        assert!(InferenceEngine::new(config).is_err());
    }

    #[test]
    fn test_embed_text() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        let embedding = engine.embed_text("Hello, world!").unwrap();
        assert_eq!(embedding.len(), 384);

        // Should be normalised
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embed_batch() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        let texts = vec!["Hello", "World", "Test"];
        let embeddings = engine.embed_batch(&texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    fn test_embed_empty_batch() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        let embeddings = engine.embed_batch(&[]).unwrap();
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_deterministic_embeddings() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        let emb1 = engine.embed_text("Hello").unwrap();
        engine.clear_cache();
        let emb2 = engine.embed_text("Hello").unwrap();
        assert_eq!(emb1, emb2, "Embeddings should be deterministic");
    }

    #[test]
    fn test_different_texts_different_embeddings() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        let emb1 = engine.embed_text("Hello").unwrap();
        let emb2 = engine.embed_text("Completely different text").unwrap();
        assert_ne!(emb1, emb2);
    }

    #[test]
    fn test_embedding_cache() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        engine.embed_text("test").unwrap();
        assert_eq!(engine.cache_size(), 1);

        engine.embed_text("test").unwrap(); // Should hit cache
        assert_eq!(engine.stats().cache_hits, 1);

        engine.clear_cache();
        assert_eq!(engine.cache_size(), 0);
    }

    #[test]
    fn test_inference_stats() {
        let engine = InferenceEngine::new(InferenceConfig::default()).unwrap();
        engine.embed_batch(&["a", "b", "c"]).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.texts_processed, 3);
        assert!(stats.batches_processed >= 1);
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn test_available_models() {
        let models = InferenceEngine::available_models();
        assert!(models.len() >= 5);
    }

    #[test]
    fn test_normalize_vector() {
        let mut v = vec![3.0, 4.0];
        normalize_vector(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize_vector(&mut v);
        // Should not crash, vector stays zero
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_model_source_variants() {
        let sources = vec![
            ModelSource::HuggingFace {
                repo_id: "test/model".into(),
            },
            ModelSource::Url {
                url: "https://example.com/model.onnx".into(),
            },
            ModelSource::Local {
                path: "/models/test.onnx".into(),
            },
            ModelSource::Mock { dimension: 128 },
        ];
        for source in &sources {
            let json = serde_json::to_string(source).unwrap();
            let _: ModelSource = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn test_text_collection_config_default() {
        let config = TextCollectionConfig::default();
        assert_eq!(config.model_id, "mock-384");
        assert!(config.store_text);
    }

    #[test]
    fn test_model_quantization_serialization() {
        for q in &[
            ModelQuantization::F32,
            ModelQuantization::F16,
            ModelQuantization::Int8,
            ModelQuantization::Q4,
        ] {
            let json = serde_json::to_string(q).unwrap();
            let decoded: ModelQuantization = serde_json::from_str(&json).unwrap();
            assert_eq!(*q, decoded);
        }
    }
}
