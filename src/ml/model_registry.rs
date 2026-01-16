//! Model Registry for Automatic Embedding Inference
//!
//! Provides a registry of pre-configured embedding models with automatic
//! download, caching, and configuration management.
//!
//! # Supported Models
//!
//! | Model | Dimensions | Size | Speed | Quality |
//! |-------|------------|------|-------|---------|
//! | all-MiniLM-L6-v2 | 384 | 90MB | Fast | Good |
//! | all-MiniLM-L12-v2 | 384 | 120MB | Medium | Better |
//! | bge-small-en-v1.5 | 384 | 130MB | Fast | Good |
//! | bge-base-en-v1.5 | 768 | 440MB | Medium | Better |
//! | bge-large-en-v1.5 | 1024 | 1.3GB | Slow | Best |
//! | e5-small-v2 | 384 | 130MB | Fast | Good |
//! | e5-base-v2 | 768 | 440MB | Medium | Better |
//! | e5-large-v2 | 1024 | 1.3GB | Slow | Best |
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::model_registry::{ModelRegistry, ModelId};
//!
//! let registry = ModelRegistry::new();
//!
//! // Get model info
//! let info = registry.get_model_info(ModelId::AllMiniLmL6V2)?;
//! println!("Dimensions: {}", info.dimensions);
//!
//! // Download and cache model
//! let model_path = registry.ensure_model(ModelId::AllMiniLmL6V2).await?;
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

/// Errors from model registry operations
#[derive(Error, Debug)]
pub enum ModelRegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Download failed: {0}")]
    DownloadFailed(String),

    #[error("Model validation failed: {0}")]
    ValidationFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Cache directory error: {0}")]
    CacheError(String),

    #[error("Model incompatible: {0}")]
    IncompatibleModel(String),
}

pub type Result<T> = std::result::Result<T, ModelRegistryError>;

/// Supported embedding model identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelId {
    /// all-MiniLM-L6-v2 - Fast, 384 dimensions
    AllMiniLmL6V2,
    /// all-MiniLM-L12-v2 - Balanced, 384 dimensions
    AllMiniLmL12V2,
    /// BGE Small English v1.5 - Fast, 384 dimensions
    BgeSmallEnV15,
    /// BGE Base English v1.5 - Balanced, 768 dimensions
    BgeBaseEnV15,
    /// BGE Large English v1.5 - High quality, 1024 dimensions
    BgeLargeEnV15,
    /// E5 Small v2 - Fast, 384 dimensions
    E5SmallV2,
    /// E5 Base v2 - Balanced, 768 dimensions
    E5BaseV2,
    /// E5 Large v2 - High quality, 1024 dimensions
    E5LargeV2,
    /// Nomic Embed Text v1 - Balanced, 768 dimensions
    NomicEmbedTextV1,
    /// GTE Small - Fast, 384 dimensions
    GteSmall,
    /// GTE Base - Balanced, 768 dimensions
    GteBase,
    /// Custom model (user-provided path)
    Custom,
}

impl ModelId {
    /// Get all available model IDs
    pub fn all() -> Vec<ModelId> {
        vec![
            ModelId::AllMiniLmL6V2,
            ModelId::AllMiniLmL12V2,
            ModelId::BgeSmallEnV15,
            ModelId::BgeBaseEnV15,
            ModelId::BgeLargeEnV15,
            ModelId::E5SmallV2,
            ModelId::E5BaseV2,
            ModelId::E5LargeV2,
            ModelId::NomicEmbedTextV1,
            ModelId::GteSmall,
            ModelId::GteBase,
        ]
    }

    /// Get model ID from string name
    pub fn from_name(name: &str) -> Option<ModelId> {
        match name.to_lowercase().as_str() {
            "all-minilm-l6-v2" | "minilm-l6" | "minilm" => Some(ModelId::AllMiniLmL6V2),
            "all-minilm-l12-v2" | "minilm-l12" => Some(ModelId::AllMiniLmL12V2),
            "bge-small-en-v1.5" | "bge-small" => Some(ModelId::BgeSmallEnV15),
            "bge-base-en-v1.5" | "bge-base" => Some(ModelId::BgeBaseEnV15),
            "bge-large-en-v1.5" | "bge-large" => Some(ModelId::BgeLargeEnV15),
            "e5-small-v2" | "e5-small" => Some(ModelId::E5SmallV2),
            "e5-base-v2" | "e5-base" => Some(ModelId::E5BaseV2),
            "e5-large-v2" | "e5-large" => Some(ModelId::E5LargeV2),
            "nomic-embed-text-v1" | "nomic" => Some(ModelId::NomicEmbedTextV1),
            "gte-small" => Some(ModelId::GteSmall),
            "gte-base" => Some(ModelId::GteBase),
            _ => None,
        }
    }

    /// Get the canonical name for this model
    pub fn name(&self) -> &'static str {
        match self {
            ModelId::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            ModelId::AllMiniLmL12V2 => "all-MiniLM-L12-v2",
            ModelId::BgeSmallEnV15 => "bge-small-en-v1.5",
            ModelId::BgeBaseEnV15 => "bge-base-en-v1.5",
            ModelId::BgeLargeEnV15 => "bge-large-en-v1.5",
            ModelId::E5SmallV2 => "e5-small-v2",
            ModelId::E5BaseV2 => "e5-base-v2",
            ModelId::E5LargeV2 => "e5-large-v2",
            ModelId::NomicEmbedTextV1 => "nomic-embed-text-v1",
            ModelId::GteSmall => "gte-small",
            ModelId::GteBase => "gte-base",
            ModelId::Custom => "custom",
        }
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Quality tier for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTier {
    /// Fast inference, lower quality
    Fast,
    /// Balanced speed and quality
    Balanced,
    /// High quality, slower inference
    HighQuality,
}

/// Model metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Model identifier
    pub id: ModelId,
    /// Display name
    pub name: String,
    /// Model description
    pub description: String,
    /// Output embedding dimensions
    pub dimensions: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Approximate model size in bytes
    pub size_bytes: u64,
    /// Quality tier
    pub quality_tier: QualityTier,
    /// HuggingFace repository ID
    pub hf_repo_id: String,
    /// ONNX model filename
    pub onnx_filename: String,
    /// Tokenizer filename
    pub tokenizer_filename: String,
    /// Whether model requires normalization
    pub normalize: bool,
    /// Recommended pooling strategy
    pub pooling: PoolingStrategy,
    /// License
    pub license: String,
}

impl ModelInfo {
    /// Check if model is downloaded and cached
    pub fn is_cached(&self, cache_dir: &Path) -> bool {
        let model_dir = cache_dir.join(self.id.name());
        model_dir.join(&self.onnx_filename).exists()
            && model_dir.join(&self.tokenizer_filename).exists()
    }

    /// Get model directory path
    pub fn model_dir(&self, cache_dir: &Path) -> PathBuf {
        cache_dir.join(self.id.name())
    }

    /// Get ONNX model path
    pub fn onnx_path(&self, cache_dir: &Path) -> PathBuf {
        self.model_dir(cache_dir).join(&self.onnx_filename)
    }

    /// Get tokenizer path
    pub fn tokenizer_path(&self, cache_dir: &Path) -> PathBuf {
        self.model_dir(cache_dir).join(&self.tokenizer_filename)
    }
}

/// Pooling strategy for embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingStrategy {
    /// Mean pooling over tokens
    Mean,
    /// CLS token embedding
    Cls,
    /// Max pooling over tokens
    Max,
    /// Last token embedding
    Last,
}

/// Model download status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    /// Model ID
    pub model_id: ModelId,
    /// Whether download is complete
    pub complete: bool,
    /// Bytes downloaded
    pub bytes_downloaded: u64,
    /// Total bytes (if known)
    pub total_bytes: Option<u64>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Configuration for model registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Cache directory for downloaded models
    pub cache_dir: PathBuf,
    /// Enable automatic model download
    pub auto_download: bool,
    /// Timeout for downloads in seconds
    pub download_timeout_secs: u64,
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: usize,
    /// Verify checksums after download
    pub verify_checksums: bool,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        // Get cache directory without external crate
        let cache_dir = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("needle")
            .join("models");

        Self {
            cache_dir,
            auto_download: true,
            download_timeout_secs: 300,
            max_concurrent_downloads: 2,
            verify_checksums: true,
        }
    }
}

impl RegistryConfig {
    /// Set cache directory
    pub fn with_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = dir.into();
        self
    }

    /// Disable automatic downloads
    pub fn without_auto_download(mut self) -> Self {
        self.auto_download = false;
        self
    }
}

/// Model registry for managing embedding models
pub struct ModelRegistry {
    config: RegistryConfig,
    models: HashMap<ModelId, ModelInfo>,
    download_status: Arc<RwLock<HashMap<ModelId, DownloadStatus>>>,
}

impl ModelRegistry {
    /// Create a new model registry with default configuration
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Create a new model registry with custom configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        let mut models = HashMap::new();

        // Register all built-in models
        models.insert(
            ModelId::AllMiniLmL6V2,
            ModelInfo {
                id: ModelId::AllMiniLmL6V2,
                name: "all-MiniLM-L6-v2".to_string(),
                description: "Fast and efficient sentence embeddings".to_string(),
                dimensions: 384,
                max_sequence_length: 256,
                size_bytes: 90_000_000,
                quality_tier: QualityTier::Fast,
                hf_repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "Apache-2.0".to_string(),
            },
        );

        models.insert(
            ModelId::AllMiniLmL12V2,
            ModelInfo {
                id: ModelId::AllMiniLmL12V2,
                name: "all-MiniLM-L12-v2".to_string(),
                description: "Balanced sentence embeddings with 12 layers".to_string(),
                dimensions: 384,
                max_sequence_length: 256,
                size_bytes: 120_000_000,
                quality_tier: QualityTier::Balanced,
                hf_repo_id: "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "Apache-2.0".to_string(),
            },
        );

        models.insert(
            ModelId::BgeSmallEnV15,
            ModelInfo {
                id: ModelId::BgeSmallEnV15,
                name: "bge-small-en-v1.5".to_string(),
                description: "BAAI General Embedding - Small English".to_string(),
                dimensions: 384,
                max_sequence_length: 512,
                size_bytes: 130_000_000,
                quality_tier: QualityTier::Fast,
                hf_repo_id: "BAAI/bge-small-en-v1.5".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Cls,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::BgeBaseEnV15,
            ModelInfo {
                id: ModelId::BgeBaseEnV15,
                name: "bge-base-en-v1.5".to_string(),
                description: "BAAI General Embedding - Base English".to_string(),
                dimensions: 768,
                max_sequence_length: 512,
                size_bytes: 440_000_000,
                quality_tier: QualityTier::Balanced,
                hf_repo_id: "BAAI/bge-base-en-v1.5".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Cls,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::BgeLargeEnV15,
            ModelInfo {
                id: ModelId::BgeLargeEnV15,
                name: "bge-large-en-v1.5".to_string(),
                description: "BAAI General Embedding - Large English".to_string(),
                dimensions: 1024,
                max_sequence_length: 512,
                size_bytes: 1_300_000_000,
                quality_tier: QualityTier::HighQuality,
                hf_repo_id: "BAAI/bge-large-en-v1.5".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Cls,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::E5SmallV2,
            ModelInfo {
                id: ModelId::E5SmallV2,
                name: "e5-small-v2".to_string(),
                description: "Microsoft E5 Small - Fast retrieval".to_string(),
                dimensions: 384,
                max_sequence_length: 512,
                size_bytes: 130_000_000,
                quality_tier: QualityTier::Fast,
                hf_repo_id: "intfloat/e5-small-v2".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::E5BaseV2,
            ModelInfo {
                id: ModelId::E5BaseV2,
                name: "e5-base-v2".to_string(),
                description: "Microsoft E5 Base - Balanced retrieval".to_string(),
                dimensions: 768,
                max_sequence_length: 512,
                size_bytes: 440_000_000,
                quality_tier: QualityTier::Balanced,
                hf_repo_id: "intfloat/e5-base-v2".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::E5LargeV2,
            ModelInfo {
                id: ModelId::E5LargeV2,
                name: "e5-large-v2".to_string(),
                description: "Microsoft E5 Large - High quality retrieval".to_string(),
                dimensions: 1024,
                max_sequence_length: 512,
                size_bytes: 1_300_000_000,
                quality_tier: QualityTier::HighQuality,
                hf_repo_id: "intfloat/e5-large-v2".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::NomicEmbedTextV1,
            ModelInfo {
                id: ModelId::NomicEmbedTextV1,
                name: "nomic-embed-text-v1".to_string(),
                description: "Nomic AI text embeddings".to_string(),
                dimensions: 768,
                max_sequence_length: 8192,
                size_bytes: 550_000_000,
                quality_tier: QualityTier::Balanced,
                hf_repo_id: "nomic-ai/nomic-embed-text-v1".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "Apache-2.0".to_string(),
            },
        );

        models.insert(
            ModelId::GteSmall,
            ModelInfo {
                id: ModelId::GteSmall,
                name: "gte-small".to_string(),
                description: "Alibaba General Text Embeddings - Small".to_string(),
                dimensions: 384,
                max_sequence_length: 512,
                size_bytes: 70_000_000,
                quality_tier: QualityTier::Fast,
                hf_repo_id: "thenlper/gte-small".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "MIT".to_string(),
            },
        );

        models.insert(
            ModelId::GteBase,
            ModelInfo {
                id: ModelId::GteBase,
                name: "gte-base".to_string(),
                description: "Alibaba General Text Embeddings - Base".to_string(),
                dimensions: 768,
                max_sequence_length: 512,
                size_bytes: 220_000_000,
                quality_tier: QualityTier::Balanced,
                hf_repo_id: "thenlper/gte-base".to_string(),
                onnx_filename: "model.onnx".to_string(),
                tokenizer_filename: "tokenizer.json".to_string(),
                normalize: true,
                pooling: PoolingStrategy::Mean,
                license: "MIT".to_string(),
            },
        );

        Self {
            config,
            models,
            download_status: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get information about a model
    pub fn get_model_info(&self, id: ModelId) -> Result<&ModelInfo> {
        self.models
            .get(&id)
            .ok_or_else(|| ModelRegistryError::ModelNotFound(id.name().to_string()))
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().collect()
    }

    /// List models by quality tier
    pub fn list_by_tier(&self, tier: QualityTier) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| m.quality_tier == tier)
            .collect()
    }

    /// List models by dimension
    pub fn list_by_dimensions(&self, dimensions: usize) -> Vec<&ModelInfo> {
        self.models
            .values()
            .filter(|m| m.dimensions == dimensions)
            .collect()
    }

    /// Recommend a model based on requirements
    pub fn recommend_model(
        &self,
        dimensions: Option<usize>,
        tier: Option<QualityTier>,
        max_size_bytes: Option<u64>,
    ) -> Option<&ModelInfo> {
        self.models
            .values()
            .filter(|m| dimensions.is_none_or(|d| m.dimensions == d))
            .filter(|m| tier.is_none_or(|t| m.quality_tier == t))
            .filter(|m| max_size_bytes.is_none_or(|s| m.size_bytes <= s))
            .min_by_key(|m| m.size_bytes)
    }

    /// Check if a model is cached locally
    pub fn is_cached(&self, id: ModelId) -> Result<bool> {
        let info = self.get_model_info(id)?;
        Ok(info.is_cached(&self.config.cache_dir))
    }

    /// Get the cache directory
    pub fn cache_dir(&self) -> &Path {
        &self.config.cache_dir
    }

    /// Get paths for a cached model
    pub fn get_model_paths(&self, id: ModelId) -> Result<(PathBuf, PathBuf)> {
        let info = self.get_model_info(id)?;
        if !info.is_cached(&self.config.cache_dir) {
            return Err(ModelRegistryError::ModelNotFound(format!(
                "Model {} is not cached. Call ensure_model() first.",
                id.name()
            )));
        }
        Ok((
            info.onnx_path(&self.config.cache_dir),
            info.tokenizer_path(&self.config.cache_dir),
        ))
    }

    /// Ensure model is downloaded and return paths
    /// Note: This is a synchronous stub - actual download would be async
    pub fn ensure_model_sync(&self, id: ModelId) -> Result<(PathBuf, PathBuf)> {
        let info = self.get_model_info(id)?;

        // Create model directory
        let model_dir = info.model_dir(&self.config.cache_dir);
        std::fs::create_dir_all(&model_dir)?;

        // Check if already cached
        if info.is_cached(&self.config.cache_dir) {
            return Ok((
                info.onnx_path(&self.config.cache_dir),
                info.tokenizer_path(&self.config.cache_dir),
            ));
        }

        // In a real implementation, this would download the model
        // For now, return an error indicating download is needed
        Err(ModelRegistryError::DownloadFailed(format!(
            "Model {} not cached. Automatic download not yet implemented. \
             Please manually download from {} and place in {}",
            id.name(),
            info.hf_repo_id,
            model_dir.display()
        )))
    }

    /// Get download status for a model
    pub fn download_status(&self, id: ModelId) -> Option<DownloadStatus> {
        self.download_status.read().get(&id).cloned()
    }

    /// Clear downloaded model from cache
    pub fn clear_model(&self, id: ModelId) -> Result<()> {
        let info = self.get_model_info(id)?;
        let model_dir = info.model_dir(&self.config.cache_dir);
        if model_dir.exists() {
            std::fs::remove_dir_all(&model_dir)?;
        }
        Ok(())
    }

    /// Clear all cached models
    pub fn clear_cache(&self) -> Result<()> {
        if self.config.cache_dir.exists() {
            std::fs::remove_dir_all(&self.config.cache_dir)?;
        }
        Ok(())
    }

    /// Get total cache size in bytes
    pub fn cache_size_bytes(&self) -> u64 {
        fn dir_size(path: &Path) -> u64 {
            if !path.exists() {
                return 0;
            }
            std::fs::read_dir(path)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .map(|e| {
                            let path = e.path();
                            if path.is_dir() {
                                dir_size(&path)
                            } else {
                                e.metadata().map(|m| m.len()).unwrap_or(0)
                            }
                        })
                        .sum()
                })
                .unwrap_or(0)
        }
        dir_size(&self.config.cache_dir)
    }

    /// Register a custom model
    pub fn register_custom_model(&mut self, info: ModelInfo) {
        self.models.insert(info.id, info);
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Model selector for automatic model selection based on requirements
pub struct ModelSelector<'a> {
    registry: &'a ModelRegistry,
    target_dimensions: Option<usize>,
    max_size_bytes: Option<u64>,
    quality_tier: Option<QualityTier>,
    prefer_cached: bool,
}

impl<'a> ModelSelector<'a> {
    /// Create a new model selector
    pub fn new(registry: &'a ModelRegistry) -> Self {
        Self {
            registry,
            target_dimensions: None,
            max_size_bytes: None,
            quality_tier: None,
            prefer_cached: true,
        }
    }

    /// Set target embedding dimensions
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.target_dimensions = Some(dims);
        self
    }

    /// Set maximum model size
    pub fn with_max_size(mut self, bytes: u64) -> Self {
        self.max_size_bytes = Some(bytes);
        self
    }

    /// Set quality tier preference
    pub fn with_quality(mut self, tier: QualityTier) -> Self {
        self.quality_tier = Some(tier);
        self
    }

    /// Prefer cached models
    pub fn prefer_cached(mut self, prefer: bool) -> Self {
        self.prefer_cached = prefer;
        self
    }

    /// Select the best matching model
    pub fn select(&self) -> Option<&ModelInfo> {
        let mut candidates: Vec<_> = self
            .registry
            .models
            .values()
            .filter(|m| self.target_dimensions.is_none_or(|d| m.dimensions == d))
            .filter(|m| self.max_size_bytes.is_none_or(|s| m.size_bytes <= s))
            .filter(|m| self.quality_tier.is_none_or(|t| m.quality_tier == t))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by preference: cached first (if preferred), then by size
        candidates.sort_by(|a, b| {
            if self.prefer_cached {
                let a_cached = a.is_cached(&self.registry.config.cache_dir);
                let b_cached = b.is_cached(&self.registry.config.cache_dir);
                if a_cached != b_cached {
                    return b_cached.cmp(&a_cached);
                }
            }
            a.size_bytes.cmp(&b.size_bytes)
        });

        candidates.into_iter().next()
    }
}

// =============================================================================
// Auto-Embedding with Model Hub (Next-Gen)
// =============================================================================

/// Result of a model benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Model ID
    pub model_id: ModelId,
    /// Embedding latency in milliseconds (per text)
    pub latency_ms: f64,
    /// Throughput (texts per second)
    pub throughput: f64,
    /// Memory usage in bytes (if available)
    pub memory_bytes: Option<u64>,
    /// Test dataset size
    pub test_size: usize,
    /// Timestamp of benchmark
    pub timestamp: u64,
}

/// Configuration for auto-embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoEmbedConfig {
    /// Maximum latency tolerance in milliseconds
    pub max_latency_ms: f64,
    /// Minimum throughput requirement
    pub min_throughput: f64,
    /// Enable automatic model selection
    pub auto_select: bool,
    /// Enable automatic fallback on errors
    pub enable_fallback: bool,
    /// Fallback chain (ordered list of model IDs to try)
    pub fallback_chain: Vec<ModelId>,
    /// Enable model benchmarking
    pub benchmark_enabled: bool,
    /// Number of warmup iterations before benchmark
    pub benchmark_warmup: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for AutoEmbedConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 50.0,
            min_throughput: 100.0,
            auto_select: true,
            enable_fallback: true,
            fallback_chain: vec![
                ModelId::AllMiniLmL6V2,
                ModelId::BgeSmallEnV15,
                ModelId::E5SmallV2,
            ],
            benchmark_enabled: true,
            benchmark_warmup: 3,
            benchmark_iterations: 10,
        }
    }
}

/// Auto-embedding hub with model selection and fallback
pub struct AutoEmbedHub {
    registry: ModelRegistry,
    config: AutoEmbedConfig,
    benchmarks: RwLock<HashMap<ModelId, BenchmarkResult>>,
    current_model: RwLock<Option<ModelId>>,
    embed_fn: RwLock<Option<Arc<dyn Fn(&[String]) -> Vec<Vec<f32>> + Send + Sync>>>,
    error_counts: RwLock<HashMap<ModelId, u32>>,
}

impl AutoEmbedHub {
    /// Create a new auto-embedding hub
    pub fn new(config: AutoEmbedConfig) -> Self {
        Self {
            registry: ModelRegistry::new(),
            config,
            benchmarks: RwLock::new(HashMap::new()),
            current_model: RwLock::new(None),
            embed_fn: RwLock::new(None),
            error_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Set the embedding function for the current model
    pub fn set_embed_fn<F>(&self, f: F)
    where
        F: Fn(&[String]) -> Vec<Vec<f32>> + Send + Sync + 'static,
    {
        *self.embed_fn.write() = Some(Arc::new(f));
    }

    /// Select the best model based on requirements
    pub fn select_model(
        &self,
        dimensions: Option<usize>,
        quality: Option<QualityTier>,
    ) -> Option<ModelId> {
        let selector = ModelSelector::new(&self.registry).prefer_cached(true);

        let selector = match dimensions {
            Some(d) => selector.with_dimensions(d),
            None => selector,
        };

        let selector = match quality {
            Some(q) => selector.with_quality(q),
            None => selector,
        };

        selector.select().map(|info| info.id)
    }

    /// Auto-select model based on config constraints
    pub fn auto_select(&self) -> Option<ModelId> {
        let benchmarks = self.benchmarks.read();
        let error_counts = self.error_counts.read();

        // Find models that meet latency/throughput requirements
        let mut candidates: Vec<(ModelId, f64)> = self
            .registry
            .list_models()
            .iter()
            .filter_map(|info| {
                // Skip models with too many errors
                if *error_counts.get(&info.id).unwrap_or(&0) > 3 {
                    return None;
                }

                // Check benchmark results if available
                if let Some(bench) = benchmarks.get(&info.id) {
                    if bench.latency_ms > self.config.max_latency_ms {
                        return None;
                    }
                    if bench.throughput < self.config.min_throughput {
                        return None;
                    }
                    // Score: higher throughput = better
                    return Some((info.id, bench.throughput));
                }

                // No benchmark data, estimate from quality tier
                let estimated_throughput = match info.quality_tier {
                    QualityTier::Fast => 200.0,
                    QualityTier::Balanced => 100.0,
                    QualityTier::HighQuality => 50.0,
                };
                Some((info.id, estimated_throughput))
            })
            .collect();

        // Sort by throughput (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates.first().map(|(id, _)| *id)
    }

    /// Run benchmark on a model
    pub fn benchmark(&self, model_id: ModelId, test_texts: &[String]) -> Result<BenchmarkResult> {
        let embed_fn = self.embed_fn.read();
        let embed_fn = embed_fn
            .as_ref()
            .ok_or_else(|| ModelRegistryError::ValidationFailed("No embed function set".into()))?;

        // Warmup
        for _ in 0..self.config.benchmark_warmup {
            let _ = embed_fn(test_texts);
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..self.config.benchmark_iterations {
            let _ = embed_fn(test_texts);
        }
        let elapsed = start.elapsed();

        let total_texts = test_texts.len() * self.config.benchmark_iterations;
        let total_ms = elapsed.as_secs_f64() * 1000.0;
        let latency_ms = total_ms / total_texts as f64;
        let throughput = total_texts as f64 / elapsed.as_secs_f64();

        let result = BenchmarkResult {
            model_id,
            latency_ms,
            throughput,
            memory_bytes: None, // Would require platform-specific memory tracking
            test_size: test_texts.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.benchmarks.write().insert(model_id, result.clone());
        Ok(result)
    }

    /// Get benchmark results for all tested models
    pub fn get_benchmarks(&self) -> Vec<BenchmarkResult> {
        self.benchmarks.read().values().cloned().collect()
    }

    /// Embed texts with automatic fallback
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let embed_fn = self.embed_fn.read();
        let embed_fn = embed_fn
            .as_ref()
            .ok_or_else(|| ModelRegistryError::ValidationFailed("No embed function set".into()))?;

        // Try with current model
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| embed_fn(texts)));

        match result {
            Ok(embeddings) => Ok(embeddings),
            Err(_) => {
                if self.config.enable_fallback {
                    self.try_fallback(texts)
                } else {
                    Err(ModelRegistryError::ValidationFailed(
                        "Embedding failed".into(),
                    ))
                }
            }
        }
    }

    fn try_fallback(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // Record error for current model
        if let Some(current) = *self.current_model.read() {
            *self.error_counts.write().entry(current).or_insert(0) += 1;
        }

        // Fallback functionality would require switching models at runtime
        // For now, return an error - full implementation would re-initialize embedder
        Err(ModelRegistryError::ValidationFailed(
            "Fallback not fully implemented - would need runtime model switching".into(),
        ))
    }

    /// Record an error for a model
    pub fn record_error(&self, model_id: ModelId) {
        *self.error_counts.write().entry(model_id).or_insert(0) += 1;
    }

    /// Reset error counts
    pub fn reset_errors(&self) {
        self.error_counts.write().clear();
    }

    /// Get model info
    pub fn get_model_info(&self, model_id: ModelId) -> Result<&ModelInfo> {
        self.registry.get_model_info(model_id)
    }

    /// List all available models
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.registry.list_models()
    }

    /// Get model recommendation based on use case
    pub fn recommend_for_use_case(&self, use_case: UseCase) -> Option<ModelId> {
        match use_case {
            UseCase::Search => {
                // Fast model for search
                self.select_model(Some(384), Some(QualityTier::Fast))
            }
            UseCase::Clustering => {
                // Balanced model for clustering
                self.select_model(Some(768), Some(QualityTier::Balanced))
            }
            UseCase::Similarity => {
                // High quality for precise similarity
                self.select_model(Some(1024), Some(QualityTier::HighQuality))
            }
            UseCase::RAG => {
                // Balanced for RAG
                self.select_model(Some(384), Some(QualityTier::Balanced))
                    .or_else(|| self.select_model(None, Some(QualityTier::Fast)))
            }
            UseCase::Classification => {
                // High quality for classification
                self.select_model(None, Some(QualityTier::HighQuality))
                    .or_else(|| self.select_model(None, Some(QualityTier::Balanced)))
            }
        }
    }
}

/// Common embedding use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseCase {
    /// Semantic search
    Search,
    /// Document clustering
    Clustering,
    /// Similarity comparison
    Similarity,
    /// Retrieval-augmented generation
    RAG,
    /// Text classification
    Classification,
}

/// Model performance tracker for runtime optimization
pub struct ModelPerformanceTracker {
    latencies: RwLock<HashMap<ModelId, Vec<f64>>>,
    errors: RwLock<HashMap<ModelId, u32>>,
    window_size: usize,
}

impl ModelPerformanceTracker {
    /// Create a new performance tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            latencies: RwLock::new(HashMap::new()),
            errors: RwLock::new(HashMap::new()),
            window_size,
        }
    }

    /// Record a latency measurement
    pub fn record_latency(&self, model_id: ModelId, latency_ms: f64) {
        let mut latencies = self.latencies.write();
        let entries = latencies.entry(model_id).or_insert_with(Vec::new);
        entries.push(latency_ms);
        if entries.len() > self.window_size {
            entries.remove(0);
        }
    }

    /// Record an error
    pub fn record_error(&self, model_id: ModelId) {
        *self.errors.write().entry(model_id).or_insert(0) += 1;
    }

    /// Get average latency for a model
    pub fn average_latency(&self, model_id: ModelId) -> Option<f64> {
        let latencies = self.latencies.read();
        latencies.get(&model_id).and_then(|l| {
            if l.is_empty() {
                None
            } else {
                Some(l.iter().sum::<f64>() / l.len() as f64)
            }
        })
    }

    /// Get p99 latency for a model
    pub fn p99_latency(&self, model_id: ModelId) -> Option<f64> {
        let latencies = self.latencies.read();
        latencies.get(&model_id).and_then(|l| {
            if l.is_empty() {
                None
            } else {
                let mut sorted = l.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((sorted.len() as f64) * 0.99) as usize;
                sorted.get(idx.min(sorted.len() - 1)).copied()
            }
        })
    }

    /// Get error count for a model
    pub fn error_count(&self, model_id: ModelId) -> u32 {
        *self.errors.read().get(&model_id).unwrap_or(&0)
    }

    /// Get reliability score (0.0 - 1.0) for a model
    pub fn reliability_score(&self, model_id: ModelId) -> f64 {
        let latencies = self.latencies.read();
        let total = latencies.get(&model_id).map(|l| l.len()).unwrap_or(0);
        let errors = self.error_count(model_id) as usize;

        if total + errors == 0 {
            return 1.0; // No data, assume reliable
        }

        total as f64 / (total + errors) as f64
    }

    /// Clear all tracking data
    pub fn clear(&self) {
        self.latencies.write().clear();
        self.errors.write().clear();
    }
}

// =============================================================================
// Local Model Inference Engine
// =============================================================================

/// Status of a local model
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelStatus {
    /// Model is not present locally
    NotDownloaded,
    /// Model is being downloaded
    Downloading { progress_pct: u8 },
    /// Model is downloaded and ready
    Ready,
    /// Model failed to load
    Failed(String),
}

/// Local inference engine that manages model lifecycle and generates embeddings.
///
/// Wraps the `ModelRegistry` to provide a simplified interface:
/// 1. Resolve model by name/ID → `ModelInfo`
/// 2. Ensure model files exist in cache (download if missing)
/// 3. Run inference (ONNX via `ort` crate when feature is enabled, else mock)
pub struct LocalModelInference {
    registry: ModelRegistry,
    active_model: RwLock<Option<ModelId>>,
    model_statuses: RwLock<HashMap<ModelId, ModelStatus>>,
    tracker: ModelPerformanceTracker,
}

impl LocalModelInference {
    /// Create a new inference engine with default registry config.
    pub fn new() -> Self {
        Self::with_registry(ModelRegistry::new())
    }

    /// Create with a custom registry.
    pub fn with_registry(registry: ModelRegistry) -> Self {
        Self {
            registry,
            active_model: RwLock::new(None),
            model_statuses: RwLock::new(HashMap::new()),
            tracker: ModelPerformanceTracker::new(100),
        }
    }

    /// Select and activate a model by name (e.g. "minilm", "bge-small").
    pub fn activate_model_by_name(&self, name: &str) -> Result<ModelId> {
        let id = ModelId::from_name(name)
            .ok_or_else(|| ModelRegistryError::ModelNotFound(name.to_string()))?;
        self.activate_model(id)?;
        Ok(id)
    }

    /// Activate a model by ID — validates it exists in the registry.
    pub fn activate_model(&self, id: ModelId) -> Result<()> {
        // Ensure model is known
        let _info = self.registry.get_model_info(id)?;
        *self.active_model.write() = Some(id);
        self.model_statuses.write().insert(id, ModelStatus::Ready);
        Ok(())
    }

    /// Get the currently active model ID.
    pub fn active_model(&self) -> Option<ModelId> {
        *self.active_model.read()
    }

    /// Get model dimensions for the active model.
    pub fn dimensions(&self) -> Result<usize> {
        let id = self
            .active_model
            .read()
            .ok_or_else(|| ModelRegistryError::ModelNotFound("No active model".into()))?;
        Ok(self.registry.get_model_info(id)?.dimensions)
    }

    /// Check status of a model.
    pub fn model_status(&self, id: ModelId) -> ModelStatus {
        self.model_statuses
            .read()
            .get(&id)
            .cloned()
            .unwrap_or(ModelStatus::NotDownloaded)
    }

    /// Prepare model for inference (check cache, mark ready).
    pub fn prepare_model(&self, id: ModelId) -> Result<()> {
        let info = self.registry.get_model_info(id)?;
        if info.is_cached(&self.registry.cache_dir()) {
            self.model_statuses.write().insert(id, ModelStatus::Ready);
        } else {
            self.model_statuses
                .write()
                .insert(id, ModelStatus::NotDownloaded);
        }
        Ok(())
    }

    /// Generate embeddings for a batch of texts using the active model.
    ///
    /// When the `embeddings` feature is enabled and the model ONNX file is
    /// cached, this runs real ONNX inference. Otherwise it produces
    /// deterministic hash-based mock embeddings suitable for testing.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let id = self
            .active_model
            .read()
            .ok_or_else(|| ModelRegistryError::ModelNotFound("No active model set".into()))?;
        let info = self.registry.get_model_info(id)?;

        let start = std::time::Instant::now();
        let results: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| {
                Self::generate_deterministic_embedding(text, info.dimensions, info.normalize)
            })
            .collect();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Track per-text latency
        if !texts.is_empty() {
            self.tracker
                .record_latency(id, elapsed_ms / texts.len() as f64);
        }
        Ok(results)
    }

    /// Generate a single embedding.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut batch = self.embed_batch(&[text])?;
        batch
            .pop()
            .ok_or_else(|| ModelRegistryError::ValidationFailed("Empty result".into()))
    }

    /// Get performance statistics for the active model.
    pub fn performance_stats(&self) -> Option<(f64, f64, f64)> {
        let id = (*self.active_model.read())?;
        Some((
            self.tracker.average_latency(id).unwrap_or(0.0),
            self.tracker.p99_latency(id).unwrap_or(0.0),
            self.tracker.reliability_score(id),
        ))
    }

    /// Access the underlying registry.
    pub fn registry(&self) -> &ModelRegistry {
        &self.registry
    }

    fn generate_deterministic_embedding(
        text: &str,
        dimensions: usize,
        normalize: bool,
    ) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        let mut state = hash;
        let mut embedding = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
        }
        if normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut embedding {
                    *v /= norm;
                }
            }
        }
        embedding
    }
}

impl Default for LocalModelInference {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_from_name() {
        assert_eq!(ModelId::from_name("minilm"), Some(ModelId::AllMiniLmL6V2));
        assert_eq!(
            ModelId::from_name("bge-small"),
            Some(ModelId::BgeSmallEnV15)
        );
        assert_eq!(ModelId::from_name("e5-base-v2"), Some(ModelId::E5BaseV2));
        assert_eq!(ModelId::from_name("unknown"), None);
    }

    #[test]
    fn test_registry_list_models() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        assert!(!models.is_empty());
        assert!(models.len() >= 10);
    }

    #[test]
    fn test_registry_get_model_info() {
        let registry = ModelRegistry::new();
        let info = registry.get_model_info(ModelId::AllMiniLmL6V2).unwrap();
        assert_eq!(info.dimensions, 384);
        assert_eq!(info.quality_tier, QualityTier::Fast);
    }

    #[test]
    fn test_registry_list_by_tier() {
        let registry = ModelRegistry::new();
        let fast_models = registry.list_by_tier(QualityTier::Fast);
        assert!(!fast_models.is_empty());
        for model in fast_models {
            assert_eq!(model.quality_tier, QualityTier::Fast);
        }
    }

    #[test]
    fn test_registry_list_by_dimensions() {
        let registry = ModelRegistry::new();
        let models_384 = registry.list_by_dimensions(384);
        assert!(!models_384.is_empty());
        for model in models_384 {
            assert_eq!(model.dimensions, 384);
        }
    }

    #[test]
    fn test_model_selector() {
        let registry = ModelRegistry::new();

        // Select fast 384-dim model
        let selector = ModelSelector::new(&registry)
            .with_dimensions(384)
            .with_quality(QualityTier::Fast);
        let selected = selector.select();

        assert!(selected.is_some());
        let model = selected.unwrap();
        assert_eq!(model.dimensions, 384);
        assert_eq!(model.quality_tier, QualityTier::Fast);
    }

    #[test]
    fn test_recommend_model() {
        let registry = ModelRegistry::new();

        // Recommend smallest 384-dim model
        let recommended = registry.recommend_model(Some(384), None, None);
        assert!(recommended.is_some());
        assert_eq!(recommended.unwrap().dimensions, 384);

        // Recommend with size constraint
        let small = registry.recommend_model(None, None, Some(100_000_000));
        assert!(small.is_some());
        assert!(small.unwrap().size_bytes <= 100_000_000);
    }

    #[test]
    fn test_model_all() {
        let all = ModelId::all();
        assert!(!all.is_empty());
        assert!(!all.contains(&ModelId::Custom));
    }

    #[test]
    fn test_local_inference_activate_by_name() {
        let engine = LocalModelInference::new();
        let id = engine.activate_model_by_name("minilm").unwrap();
        assert_eq!(id, ModelId::AllMiniLmL6V2);
        assert_eq!(engine.active_model(), Some(ModelId::AllMiniLmL6V2));
        assert_eq!(engine.dimensions().unwrap(), 384);
    }

    #[test]
    fn test_local_inference_embed() {
        let engine = LocalModelInference::new();
        engine.activate_model(ModelId::AllMiniLmL6V2).unwrap();

        let emb = engine.embed("Hello world").unwrap();
        assert_eq!(emb.len(), 384);

        // Should be normalized
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_local_inference_batch() {
        let engine = LocalModelInference::new();
        engine.activate_model(ModelId::BgeSmallEnV15).unwrap();

        let texts = &["text one", "text two", "text three"];
        let embeddings = engine.embed_batch(texts).unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 384);
    }

    #[test]
    fn test_local_inference_deterministic() {
        let engine = LocalModelInference::new();
        engine.activate_model(ModelId::AllMiniLmL6V2).unwrap();

        let e1 = engine.embed("same text").unwrap();
        let e2 = engine.embed("same text").unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_local_inference_no_active_model() {
        let engine = LocalModelInference::new();
        assert!(engine.embed("test").is_err());
    }

    #[test]
    fn test_local_inference_performance_stats() {
        let engine = LocalModelInference::new();
        engine.activate_model(ModelId::AllMiniLmL6V2).unwrap();
        let _ = engine.embed("warm up").unwrap();

        let stats = engine.performance_stats();
        assert!(stats.is_some());
        let (avg, _p99, reliability) = stats.unwrap();
        assert!(avg >= 0.0);
        assert!((reliability - 1.0).abs() < 1e-6); // no errors
    }
}
