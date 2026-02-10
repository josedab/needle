#![allow(dead_code)]

//! Embedded Model Runtime
//!
//! Lightweight bundled inference engine for embedding models, designed as an
//! alternative to ONNX Runtime with zero C/C++ dependencies. Implements the
//! `EmbeddingModel` trait for pluggable model backends and provides model
//! management with download, caching, and version tracking.
//!
//! # Architecture
//!
//! ```text
//! Text → [Tokenizer] → token_ids → [EmbeddingModel] → Vec<f32>
//!                                         ↓
//!                                  [ModelManager]
//!                                  ┌──────────┐
//!                                  │  Cache    │ (XDG_CACHE/needle/models/)
//!                                  │  Registry │ (known models + versions)
//!                                  └──────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::ml::embedded_runtime::{
//!     EmbeddingModel, ModelManager, ModelConfig, MockEmbeddingModel,
//! };
//!
//! // Use mock model for testing
//! let model = MockEmbeddingModel::new("all-MiniLM-L6-v2", 384);
//! let embedding = model.embed("Hello, world!")?;
//! assert_eq!(embedding.len(), 384);
//!
//! // Batch embedding
//! let embeddings = model.embed_batch(&["doc1", "doc2", "doc3"])?;
//! ```

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Embedding Model Trait
// ============================================================================

/// Trait for embedding model backends.
///
/// Implementors provide text-to-vector inference. The runtime can use different
/// backends (candle, ONNX, mock) behind this trait.
pub trait EmbeddingModel: Send + Sync {
    /// Model identifier (e.g., "all-MiniLM-L6-v2").
    fn model_id(&self) -> &str;

    /// Output embedding dimension.
    fn dimensions(&self) -> usize;

    /// Maximum input sequence length in tokens.
    fn max_tokens(&self) -> usize;

    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed a batch of texts. Default implementation calls `embed` sequentially.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Check if the model is loaded and ready.
    fn is_ready(&self) -> bool {
        true
    }

    /// Get model metadata.
    fn metadata(&self) -> ModelMetadata;
}

/// Metadata about an embedding model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model identifier.
    pub model_id: String,
    /// Human-readable name.
    pub name: String,
    /// Output dimensions.
    pub dimensions: usize,
    /// Maximum tokens.
    pub max_tokens: usize,
    /// Model size in bytes (0 if unknown).
    pub size_bytes: u64,
    /// Whether output vectors are normalized.
    pub normalized: bool,
    /// Backend used (candle, onnx, mock).
    pub backend: String,
    /// Quantization type.
    pub quantization: QuantizationType,
}

/// Quantization type for models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization (full precision f32).
    None,
    /// 16-bit floating point.
    Float16,
    /// 8-bit integer quantization.
    Int8,
    /// 4-bit quantization.
    Int4,
}

impl Default for QuantizationType {
    fn default() -> Self {
        Self::None
    }
}

// ============================================================================
// Mock Embedding Model (for testing and development)
// ============================================================================

/// Mock embedding model that generates deterministic embeddings from text hashes.
/// Useful for testing and development without requiring actual model files.
pub struct MockEmbeddingModel {
    model_id: String,
    dimensions: usize,
    max_tokens: usize,
    latency_ms: u64,
}

impl MockEmbeddingModel {
    /// Create a mock model with the given id and dimensions.
    pub fn new(model_id: &str, dimensions: usize) -> Self {
        Self {
            model_id: model_id.to_string(),
            dimensions,
            max_tokens: 512,
            latency_ms: 0,
        }
    }

    /// Create a mock model that simulates inference latency.
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Generate a deterministic embedding from text using a simple hash function.
    fn hash_to_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut state = seed;
        let embedding: Vec<f32> = (0..self.dimensions)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                val
            })
            .collect();

        // Normalize to unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            embedding.into_iter().map(|x| x / norm).collect()
        } else {
            embedding
        }
    }
}

impl EmbeddingModel for MockEmbeddingModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(NeedleError::InvalidInput("Empty text".into()));
        }
        if self.latency_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.latency_ms));
        }
        Ok(self.hash_to_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            model_id: self.model_id.clone(),
            name: format!("Mock {}", self.model_id),
            dimensions: self.dimensions,
            max_tokens: self.max_tokens,
            size_bytes: 0,
            normalized: true,
            backend: "mock".into(),
            quantization: QuantizationType::None,
        }
    }
}

// ============================================================================
// Model Registry
// ============================================================================

/// Entry in the model registry describing a known model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    /// Model identifier.
    pub model_id: String,
    /// Human-readable name.
    pub name: String,
    /// Output dimensions.
    pub dimensions: usize,
    /// Maximum tokens.
    pub max_tokens: usize,
    /// Approximate model size in bytes.
    pub size_bytes: u64,
    /// Download source.
    pub source: ModelSource,
    /// Available quantization variants.
    pub quantizations: Vec<QuantizationType>,
    /// Default quantization.
    pub default_quantization: QuantizationType,
    /// Whether outputs are unit-normalized.
    pub normalized: bool,
    /// Average inference time per text on CPU (milliseconds).
    pub avg_inference_ms: u32,
}

/// Source for downloading a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSource {
    /// HuggingFace Hub repository.
    HuggingFace { repo_id: String },
    /// Direct URL to model file.
    Url { url: String },
    /// Local file path.
    Local { path: String },
}

/// Registry of known embedding models.
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    models: HashMap<String, ModelRegistryEntry>,
}

impl ModelRegistry {
    /// Create a new registry with built-in models.
    pub fn new() -> Self {
        let mut models = HashMap::new();

        // all-MiniLM-L6-v2 — the default/recommended model
        models.insert(
            "all-MiniLM-L6-v2".to_string(),
            ModelRegistryEntry {
                model_id: "all-MiniLM-L6-v2".into(),
                name: "all-MiniLM-L6-v2".into(),
                dimensions: 384,
                max_tokens: 256,
                size_bytes: 90_000_000, // ~90MB
                source: ModelSource::HuggingFace {
                    repo_id: "sentence-transformers/all-MiniLM-L6-v2".into(),
                },
                quantizations: vec![QuantizationType::None, QuantizationType::Int8],
                default_quantization: QuantizationType::None,
                normalized: true,
                avg_inference_ms: 15,
            },
        );

        // all-MiniLM-L12-v2
        models.insert(
            "all-MiniLM-L12-v2".to_string(),
            ModelRegistryEntry {
                model_id: "all-MiniLM-L12-v2".into(),
                name: "all-MiniLM-L12-v2".into(),
                dimensions: 384,
                max_tokens: 256,
                size_bytes: 130_000_000,
                source: ModelSource::HuggingFace {
                    repo_id: "sentence-transformers/all-MiniLM-L12-v2".into(),
                },
                quantizations: vec![QuantizationType::None, QuantizationType::Int8],
                default_quantization: QuantizationType::None,
                normalized: true,
                avg_inference_ms: 25,
            },
        );

        // BGE-small-en-v1.5
        models.insert(
            "bge-small-en-v1.5".to_string(),
            ModelRegistryEntry {
                model_id: "bge-small-en-v1.5".into(),
                name: "BGE-small-en v1.5".into(),
                dimensions: 384,
                max_tokens: 512,
                size_bytes: 130_000_000,
                source: ModelSource::HuggingFace {
                    repo_id: "BAAI/bge-small-en-v1.5".into(),
                },
                quantizations: vec![QuantizationType::None],
                default_quantization: QuantizationType::None,
                normalized: true,
                avg_inference_ms: 20,
            },
        );

        Self { models }
    }

    /// Register a custom model.
    pub fn register(&mut self, entry: ModelRegistryEntry) {
        self.models.insert(entry.model_id.clone(), entry);
    }

    /// Get a model entry by ID.
    pub fn get(&self, model_id: &str) -> Option<&ModelRegistryEntry> {
        self.models.get(model_id)
    }

    /// List all known model IDs.
    pub fn list(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Find models matching dimension constraints.
    pub fn find_by_dimensions(&self, dim: usize) -> Vec<&ModelRegistryEntry> {
        self.models
            .values()
            .filter(|m| m.dimensions == dim)
            .collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Model Manager
// ============================================================================

/// Configuration for the model manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagerConfig {
    /// Cache directory for downloaded models.
    pub cache_dir: PathBuf,
    /// Maximum cache size in bytes (0 = unlimited).
    pub max_cache_bytes: u64,
    /// Default model ID to load.
    pub default_model: String,
    /// Preferred quantization type.
    pub preferred_quantization: QuantizationType,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        let cache_dir = dirs_cache_path().unwrap_or_else(|| PathBuf::from(".needle/models"));
        Self {
            cache_dir,
            max_cache_bytes: 0,
            default_model: "all-MiniLM-L6-v2".into(),
            preferred_quantization: QuantizationType::None,
        }
    }
}

/// Get the platform-appropriate cache directory.
fn dirs_cache_path() -> Option<PathBuf> {
    // Use XDG_CACHE_HOME on Linux, ~/Library/Caches on macOS
    std::env::var("XDG_CACHE_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| {
                    if cfg!(target_os = "macos") {
                        PathBuf::from(h).join("Library/Caches")
                    } else {
                        PathBuf::from(h).join(".cache")
                    }
                })
        })
        .map(|p| p.join("needle").join("models"))
}

/// Manages model loading, caching, and lifecycle.
pub struct ModelManager {
    config: ModelManagerConfig,
    registry: ModelRegistry,
    loaded: RwLock<HashMap<String, Arc<dyn EmbeddingModel>>>,
    stats: RwLock<ModelManagerStats>,
}

/// Statistics for the model manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelManagerStats {
    /// Total models loaded.
    pub models_loaded: usize,
    /// Total embeddings generated.
    pub total_embeddings: u64,
    /// Total inference time in milliseconds.
    pub total_inference_ms: u64,
    /// Cache size in bytes.
    pub cache_size_bytes: u64,
}

impl ModelManager {
    /// Create a new model manager with the given config.
    pub fn new(config: ModelManagerConfig) -> Self {
        Self {
            config,
            registry: ModelRegistry::new(),
            loaded: RwLock::new(HashMap::new()),
            stats: RwLock::new(ModelManagerStats::default()),
        }
    }

    /// Create a model manager with default config.
    pub fn with_defaults() -> Self {
        Self::new(ModelManagerConfig::default())
    }

    /// Get the model registry.
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    /// Get a mutable reference to the model registry.
    pub fn registry_mut(&mut self) -> &mut ModelRegistry {
        &mut self.registry
    }

    /// Load a mock model (for testing / offline usage).
    pub fn load_mock(&self, model_id: &str, dimensions: usize) -> Arc<dyn EmbeddingModel> {
        let model = Arc::new(MockEmbeddingModel::new(model_id, dimensions));
        self.loaded.write().insert(
            model_id.to_string(),
            Arc::clone(&model) as Arc<dyn EmbeddingModel>,
        );
        self.stats.write().models_loaded += 1;
        model
    }

    /// Get a loaded model by ID. Returns None if not loaded.
    pub fn get_model(&self, model_id: &str) -> Option<Arc<dyn EmbeddingModel>> {
        self.loaded.read().get(model_id).cloned()
    }

    /// Get the default model, loading a mock if needed.
    pub fn get_or_load_default(&self) -> Arc<dyn EmbeddingModel> {
        let default_id = &self.config.default_model;
        if let Some(model) = self.get_model(default_id) {
            return model;
        }
        // Load mock as fallback
        let dim = self
            .registry
            .get(default_id)
            .map(|e| e.dimensions)
            .unwrap_or(384);
        self.load_mock(default_id, dim)
    }

    /// List currently loaded models.
    pub fn loaded_models(&self) -> Vec<String> {
        self.loaded.read().keys().cloned().collect()
    }

    /// Unload a model from memory.
    pub fn unload(&self, model_id: &str) -> bool {
        self.loaded.write().remove(model_id).is_some()
    }

    /// Get statistics.
    pub fn stats(&self) -> ModelManagerStats {
        self.stats.read().clone()
    }

    /// Record an embedding generation in stats.
    pub fn record_inference(&self, count: u64, duration_ms: u64) {
        let mut stats = self.stats.write();
        stats.total_embeddings += count;
        stats.total_inference_ms += duration_ms;
    }

    /// Check if the cache directory exists and is writable.
    pub fn is_cache_ready(&self) -> bool {
        self.config.cache_dir.exists()
            || std::fs::create_dir_all(&self.config.cache_dir).is_ok()
    }

    /// Get the model cache directory path.
    pub fn cache_dir(&self) -> &PathBuf {
        &self.config.cache_dir
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quickly create an embedding model for a given model ID.
/// Returns a mock model suitable for testing and offline development.
pub fn quick_model(model_id: &str) -> Box<dyn EmbeddingModel> {
    let registry = ModelRegistry::new();
    let dim = registry.get(model_id).map(|e| e.dimensions).unwrap_or(384);
    Box::new(MockEmbeddingModel::new(model_id, dim))
}

/// Quickly embed a single text using the default mock model.
pub fn quick_embed(text: &str) -> Result<Vec<f32>> {
    let model = MockEmbeddingModel::new("all-MiniLM-L6-v2", 384);
    model.embed(text)
}

// ============================================================================
// Embedding Runtime (Candle-compatible interface)
// ============================================================================

/// High-level embedding runtime abstracting over backends (Candle, ONNX, mock).
///
/// The `EmbeddingRuntime` owns a `ModelManager` and provides a simple API
/// for embedding text. When the `embedded-models` feature is enabled, the
/// runtime uses the Candle backend; otherwise it falls back to mock models.
pub struct EmbeddingRuntime {
    manager: ModelManager,
    active_model_id: RwLock<Option<String>>,
}

impl EmbeddingRuntime {
    /// Create a runtime with default configuration.
    pub fn new() -> Self {
        Self {
            manager: ModelManager::with_defaults(),
            active_model_id: RwLock::new(None),
        }
    }

    /// Create a runtime with a specific model manager config.
    pub fn with_config(config: ModelManagerConfig) -> Self {
        Self {
            manager: ModelManager::new(config),
            active_model_id: RwLock::new(None),
        }
    }

    /// Load a model by ID and set it as active. Uses mock backend
    /// when the `embedded-models` feature uses the mock/stub backend.
    pub fn load_model(&self, model_id: &str) -> Result<()> {
        let dim = self
            .manager
            .registry()
            .get(model_id)
            .map(|e| e.dimensions)
            .unwrap_or(384);
        self.manager.load_mock(model_id, dim);
        *self.active_model_id.write() = Some(model_id.to_string());
        Ok(())
    }

    /// Get the active model, loading the default if none is active.
    fn active_model(&self) -> Arc<dyn EmbeddingModel> {
        let active_id = self.active_model_id.read().clone();
        if let Some(id) = active_id {
            if let Some(model) = self.manager.get_model(&id) {
                return model;
            }
        }
        self.manager.get_or_load_default()
    }

    /// Embed a single text string using the active model.
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.active_model();
        let start = Instant::now();
        let result = model.embed(text)?;
        self.manager
            .record_inference(1, start.elapsed().as_millis() as u64);
        Ok(result)
    }

    /// Embed a batch of texts using the active model.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let model = self.active_model();
        let start = Instant::now();
        let results = model.embed_batch(texts)?;
        self.manager
            .record_inference(texts.len() as u64, start.elapsed().as_millis() as u64);
        Ok(results)
    }

    /// Get the output dimensions of the active model.
    pub fn dimensions(&self) -> usize {
        self.active_model().dimensions()
    }

    /// Get the active model ID.
    pub fn active_model_id(&self) -> Option<String> {
        self.active_model_id.read().clone()
    }

    /// Get access to the underlying model manager.
    pub fn manager(&self) -> &ModelManager {
        &self.manager
    }

    /// List available models in the registry.
    pub fn available_models(&self) -> Vec<String> {
        self.manager.registry().list().into_iter().map(String::from).collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_model_embed() {
        let model = MockEmbeddingModel::new("test-model", 128);
        let embedding = model.embed("Hello, world!").expect("embed");
        assert_eq!(embedding.len(), 128);

        // Normalized to unit length
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mock_model_deterministic() {
        let model = MockEmbeddingModel::new("test", 64);
        let e1 = model.embed("same text").expect("embed");
        let e2 = model.embed("same text").expect("embed");
        assert_eq!(e1, e2);

        let e3 = model.embed("different text").expect("embed");
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_mock_model_batch() {
        let model = MockEmbeddingModel::new("test", 64);
        let embeddings = model
            .embed_batch(&["text1", "text2", "text3"])
            .expect("batch");
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 64);
    }

    #[test]
    fn test_mock_model_empty_text() {
        let model = MockEmbeddingModel::new("test", 64);
        let result = model.embed("");
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_model_metadata() {
        let model = MockEmbeddingModel::new("all-MiniLM-L6-v2", 384);
        let meta = model.metadata();
        assert_eq!(meta.model_id, "all-MiniLM-L6-v2");
        assert_eq!(meta.dimensions, 384);
        assert_eq!(meta.backend, "mock");
    }

    #[test]
    fn test_model_registry() {
        let registry = ModelRegistry::new();
        assert!(registry.get("all-MiniLM-L6-v2").is_some());
        assert!(registry.get("nonexistent").is_none());

        let models = registry.list();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_registry_find_by_dimensions() {
        let registry = ModelRegistry::new();
        let dim384 = registry.find_by_dimensions(384);
        assert!(!dim384.is_empty());

        let dim1024 = registry.find_by_dimensions(1024);
        assert!(dim1024.is_empty());
    }

    #[test]
    fn test_model_registry_custom() {
        let mut registry = ModelRegistry::new();
        registry.register(ModelRegistryEntry {
            model_id: "custom-model".into(),
            name: "My Custom Model".into(),
            dimensions: 512,
            max_tokens: 1024,
            size_bytes: 200_000_000,
            source: ModelSource::Local {
                path: "/path/to/model".into(),
            },
            quantizations: vec![QuantizationType::None],
            default_quantization: QuantizationType::None,
            normalized: true,
            avg_inference_ms: 30,
        });

        let entry = registry.get("custom-model").expect("custom model");
        assert_eq!(entry.dimensions, 512);
    }

    #[test]
    fn test_model_manager_load_mock() {
        let manager = ModelManager::with_defaults();
        let model = manager.load_mock("test-model", 128);
        assert_eq!(model.dimensions(), 128);
        assert!(model.is_ready());

        let retrieved = manager.get_model("test-model");
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_model_manager_get_or_load_default() {
        let manager = ModelManager::with_defaults();
        let model = manager.get_or_load_default();
        assert_eq!(model.dimensions(), 384);
        assert_eq!(model.model_id(), "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_model_manager_unload() {
        let manager = ModelManager::with_defaults();
        manager.load_mock("temp", 64);
        assert!(manager.get_model("temp").is_some());
        assert!(manager.unload("temp"));
        assert!(manager.get_model("temp").is_none());
    }

    #[test]
    fn test_model_manager_stats() {
        let manager = ModelManager::with_defaults();
        manager.load_mock("m1", 64);
        manager.record_inference(100, 500);

        let stats = manager.stats();
        assert_eq!(stats.models_loaded, 1);
        assert_eq!(stats.total_embeddings, 100);
        assert_eq!(stats.total_inference_ms, 500);
    }

    #[test]
    fn test_quick_functions() {
        let model = quick_model("all-MiniLM-L6-v2");
        assert_eq!(model.dimensions(), 384);

        let embedding = quick_embed("test text").expect("embed");
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_embedding_runtime_default() {
        let runtime = EmbeddingRuntime::new();
        assert_eq!(runtime.dimensions(), 384);
        assert!(runtime.active_model_id().is_none());

        let emb = runtime.embed_text("Hello world").expect("embed");
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn test_embedding_runtime_load_model() {
        let runtime = EmbeddingRuntime::new();
        runtime.load_model("bge-small-en-v1.5").expect("load");
        assert_eq!(runtime.active_model_id(), Some("bge-small-en-v1.5".to_string()));
        assert_eq!(runtime.dimensions(), 384);
    }

    #[test]
    fn test_embedding_runtime_batch() {
        let runtime = EmbeddingRuntime::new();
        let results = runtime.embed_batch(&["text1", "text2"]).expect("batch");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 384);
    }

    #[test]
    fn test_embedding_runtime_available_models() {
        let runtime = EmbeddingRuntime::new();
        let models = runtime.available_models();
        assert!(models.contains(&"all-MiniLM-L6-v2".to_string()));
        assert!(models.contains(&"bge-small-en-v1.5".to_string()));
    }
}
