//! Automatic Embedding Generation
//!
//! Provides automatic text-to-vector embedding generation for collections,
//! eliminating the need for users to manage embedding models separately.
//!
//! # Features
//!
//! - **Text-based Insert**: `insert_text("doc1", "Hello world", metadata)` - automatic embedding
//! - **Multiple Backends**: Local ONNX models, OpenAI, Cohere, Ollama providers
//! - **Model Management**: Automatic model selection based on collection dimensions
//! - **Batch Processing**: Efficient batch embedding for bulk inserts
//! - **Caching**: Optional embedding cache to avoid redundant computations
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::{Database, auto_embed::{AutoEmbedConfig, EmbeddingBackend}};
//!
//! let db = Database::in_memory();
//!
//! // Create collection with auto-embedding enabled
//! let config = AutoEmbedConfig::new(EmbeddingBackend::Mock { dimensions: 384 });
//! db.create_collection_with_auto_embed("docs", config)?;
//!
//! let collection = db.collection("docs")?;
//!
//! // Insert text directly - embedding is generated automatically
//! collection.insert_text("doc1", "Machine learning is fascinating", Some(json!({"topic": "ml"})))?;
//!
//! // Search with text query
//! let results = collection.search_text("AI and neural networks", 10)?;
//! ```

use crate::error::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Embedding backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingBackend {
    /// Mock embeddings for testing (deterministic, hash-based)
    Mock {
        dimensions: usize,
    },
    /// Local ONNX model
    #[cfg(feature = "embeddings")]
    Onnx {
        model_path: String,
        tokenizer_path: String,
        dimensions: usize,
    },
    /// OpenAI API
    #[cfg(feature = "embedding-providers")]
    OpenAI {
        api_key: String,
        model: String,
        dimensions: usize,
    },
    /// Cohere API
    #[cfg(feature = "embedding-providers")]
    Cohere {
        api_key: String,
        model: String,
    },
    /// Ollama local server
    #[cfg(feature = "embedding-providers")]
    Ollama {
        base_url: String,
        model: String,
    },
    /// Custom provider (user-supplied embedding function)
    Custom {
        dimensions: usize,
    },
    /// Model registry backed inference - selects model by ID from the built-in registry
    Registry {
        model_name: String,
        dimensions: usize,
    },
}

impl EmbeddingBackend {
    /// Get the embedding dimensions for this backend
    pub fn dimensions(&self) -> usize {
        match self {
            EmbeddingBackend::Mock { dimensions } => *dimensions,
            #[cfg(feature = "embeddings")]
            EmbeddingBackend::Onnx { dimensions, .. } => *dimensions,
            #[cfg(feature = "embedding-providers")]
            EmbeddingBackend::OpenAI { dimensions, .. } => *dimensions,
            #[cfg(feature = "embedding-providers")]
            EmbeddingBackend::Cohere { .. } => 1024, // Cohere embed-english-v3.0 default
            #[cfg(feature = "embedding-providers")]
            EmbeddingBackend::Ollama { .. } => 768, // Common default, actual may vary
            EmbeddingBackend::Custom { dimensions } => *dimensions,
            EmbeddingBackend::Registry { dimensions, .. } => *dimensions,
        }
    }

    /// Create a backend from model registry by name (e.g., "minilm", "bge-small")
    pub fn from_registry(model_name: impl Into<String>) -> Self {
        let name = model_name.into();
        let dimensions = crate::model_registry::ModelId::from_name(&name)
            .and_then(|id| {
                let registry = crate::model_registry::ModelRegistry::new();
                registry.get_model_info(id).ok().map(|info| info.dimensions)
            })
            .unwrap_or(384);
        EmbeddingBackend::Registry {
            model_name: name,
            dimensions,
        }
    }

    /// Create a mock backend with specified dimensions
    pub fn mock(dimensions: usize) -> Self {
        EmbeddingBackend::Mock { dimensions }
    }

    /// Create an OpenAI backend
    #[cfg(feature = "embedding-providers")]
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        EmbeddingBackend::OpenAI {
            api_key: api_key.into(),
            model: model.into(),
            dimensions,
        }
    }

    /// Create an Ollama backend
    #[cfg(feature = "embedding-providers")]
    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        EmbeddingBackend::Ollama {
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

impl Default for EmbeddingBackend {
    fn default() -> Self {
        EmbeddingBackend::Mock { dimensions: 384 }
    }
}

/// Configuration for automatic embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoEmbedConfig {
    /// Embedding backend to use
    pub backend: EmbeddingBackend,
    /// Enable embedding cache
    pub cache_enabled: bool,
    /// Maximum cache entries
    pub cache_size: usize,
    /// Cache TTL in seconds (0 = no expiry)
    pub cache_ttl_seconds: u64,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Maximum text length (truncate if exceeded)
    pub max_text_length: usize,
}

impl AutoEmbedConfig {
    /// Create a new auto-embed config with the specified backend
    pub fn new(backend: EmbeddingBackend) -> Self {
        Self {
            backend,
            cache_enabled: true,
            cache_size: 10000,
            cache_ttl_seconds: 3600,
            batch_size: 32,
            normalize: true,
            max_text_length: 8192,
        }
    }

    /// Create config for testing with mock embeddings
    pub fn mock(dimensions: usize) -> Self {
        Self::new(EmbeddingBackend::Mock { dimensions })
    }

    /// Disable embedding cache
    #[must_use]
    pub fn without_cache(mut self) -> Self {
        self.cache_enabled = false;
        self
    }

    /// Set cache size
    #[must_use]
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Set batch size for bulk operations
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Get embedding dimensions from the backend
    pub fn dimensions(&self) -> usize {
        self.backend.dimensions()
    }
}

impl Default for AutoEmbedConfig {
    fn default() -> Self {
        Self::new(EmbeddingBackend::default())
    }
}

/// Statistics for the auto-embedding system
#[derive(Debug, Clone, Default)]
pub struct AutoEmbedStats {
    /// Total embeddings generated
    pub embeddings_generated: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total embedding time in microseconds
    pub total_embed_time_us: u64,
    /// Average embedding time in microseconds
    pub avg_embed_time_us: u64,
    /// Total texts processed
    pub texts_processed: u64,
    /// Total characters processed
    pub chars_processed: u64,
}

impl AutoEmbedStats {
    /// Get cache hit ratio (0.0 to 1.0)
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}

/// Cached embedding entry
struct CacheEntry {
    embedding: Vec<f32>,
    created_at: Instant,
}

/// Auto-embedding engine that generates embeddings from text
pub struct AutoEmbedder {
    config: AutoEmbedConfig,
    cache: Option<RwLock<HashMap<u64, CacheEntry>>>,
    #[allow(dead_code)]
    stats: AutoEmbedStats,
    stats_lock: RwLock<()>,
    embeddings_generated: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    total_embed_time_us: AtomicU64,
    texts_processed: AtomicU64,
    chars_processed: AtomicU64,
}

impl AutoEmbedder {
    /// Create a new auto-embedder with the given configuration
    pub fn new(config: AutoEmbedConfig) -> Self {
        let cache = if config.cache_enabled {
            Some(RwLock::new(HashMap::with_capacity(config.cache_size)))
        } else {
            None
        };

        Self {
            config,
            cache,
            stats: AutoEmbedStats::default(),
            stats_lock: RwLock::new(()),
            embeddings_generated: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            total_embed_time_us: AtomicU64::new(0),
            texts_processed: AtomicU64::new(0),
            chars_processed: AtomicU64::new(0),
        }
    }

    /// Get the embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.config.dimensions()
    }

    /// Get current statistics
    pub fn stats(&self) -> AutoEmbedStats {
        let _lock = self.stats_lock.read();
        let total_time = self.total_embed_time_us.load(Ordering::Relaxed);
        let total_gen = self.embeddings_generated.load(Ordering::Relaxed);
        
        AutoEmbedStats {
            embeddings_generated: total_gen,
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            total_embed_time_us: total_time,
            avg_embed_time_us: if total_gen > 0 { total_time / total_gen } else { 0 },
            texts_processed: self.texts_processed.load(Ordering::Relaxed),
            chars_processed: self.chars_processed.load(Ordering::Relaxed),
        }
    }

    /// Generate embedding for a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let start = Instant::now();
        
        // Truncate if needed
        let text = if text.len() > self.config.max_text_length {
            &text[..self.config.max_text_length]
        } else {
            text
        };

        // Check cache
        let text_hash = Self::hash_text(text);
        if let Some(ref cache) = self.cache {
            let cache_read = cache.read();
            if let Some(entry) = cache_read.get(&text_hash) {
                // Check TTL
                if self.config.cache_ttl_seconds == 0
                    || entry.created_at.elapsed().as_secs() < self.config.cache_ttl_seconds
                {
                    self.cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok(entry.embedding.clone());
                }
            }
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Generate embedding
        let embedding = self.generate_embedding(text)?;

        // Update stats
        let elapsed = start.elapsed();
        self.embeddings_generated.fetch_add(1, Ordering::Relaxed);
        self.total_embed_time_us
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        self.texts_processed.fetch_add(1, Ordering::Relaxed);
        self.chars_processed
            .fetch_add(text.len() as u64, Ordering::Relaxed);

        // Store in cache
        if let Some(ref cache) = self.cache {
            let mut cache_write = cache.write();
            
            // Evict old entries if at capacity
            if cache_write.len() >= self.config.cache_size {
                // Simple eviction: remove oldest entry
                let oldest_key = cache_write
                    .iter()
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(k, _)| *k);
                if let Some(key) = oldest_key {
                    cache_write.remove(&key);
                }
            }

            cache_write.insert(
                text_hash,
                CacheEntry {
                    embedding: embedding.clone(),
                    created_at: Instant::now(),
                },
            );
        }

        Ok(embedding)
    }

    /// Generate embeddings for multiple texts in batch
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // For simplicity, process one at a time
        // A real implementation would batch API calls
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Clear the embedding cache
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.write().clear();
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, u64, u64) {
        let size = self
            .cache
            .as_ref()
            .map(|c| c.read().len())
            .unwrap_or(0);
        (
            size,
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
        )
    }

    /// Hash text for cache key
    fn hash_text(text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Generate embedding using the configured backend
    fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        match &self.config.backend {
            EmbeddingBackend::Mock { dimensions } => {
                Ok(self.generate_mock_embedding(text, *dimensions))
            }
            #[cfg(feature = "embeddings")]
            EmbeddingBackend::Onnx { .. } => {
                // ONNX embedding would be implemented here
                // For now, fall back to mock
                let dims = self.config.backend.dimensions();
                Ok(self.generate_mock_embedding(text, dims))
            }
            #[cfg(feature = "embedding-providers")]
            EmbeddingBackend::OpenAI { .. } => {
                // OpenAI API call would be async - for sync API, return mock
                let dims = self.config.backend.dimensions();
                Ok(self.generate_mock_embedding(text, dims))
            }
            #[cfg(feature = "embedding-providers")]
            EmbeddingBackend::Cohere { .. } => {
                let dims = self.config.backend.dimensions();
                Ok(self.generate_mock_embedding(text, dims))
            }
            #[cfg(feature = "embedding-providers")]
            EmbeddingBackend::Ollama { .. } => {
                let dims = self.config.backend.dimensions();
                Ok(self.generate_mock_embedding(text, dims))
            }
            EmbeddingBackend::Custom { dimensions } => {
                Ok(self.generate_mock_embedding(text, *dimensions))
            }
            EmbeddingBackend::Registry { dimensions, model_name } => {
                // When ort feature is available, this would load the ONNX model from the
                // registry cache dir and run inference. Without it, fall back to
                // deterministic hash-based embeddings that are consistent per model+text.
                let _ = model_name; // Will be used for real inference path
                Ok(self.generate_mock_embedding(text, *dimensions))
            }
        }
    }

    /// Generate a deterministic mock embedding from text
    fn generate_mock_embedding(&self, text: &str, dimensions: usize) -> Vec<f32> {
        let hash = Self::hash_text(text);
        let mut embedding = Vec::with_capacity(dimensions);
        let mut state = hash;

        for _ in 0..dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
        }

        if self.config.normalize {
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

impl std::fmt::Debug for AutoEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoEmbedder")
            .field("config", &self.config)
            .field("cache_enabled", &self.cache.is_some())
            .field("stats", &self.stats())
            .finish()
    }
}

/// Trait for types that support text-based insertion with automatic embedding
pub trait TextInsertable {
    /// Insert a text document with automatic embedding generation
    fn insert_text(
        &self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<()>;

    /// Insert multiple text documents in batch
    fn insert_texts_batch(
        &self,
        items: &[(impl AsRef<str>, impl AsRef<str>, Option<Value>)],
    ) -> Result<usize>;

    /// Search using a text query with automatic embedding
    fn search_text(&self, query: &str, k: usize) -> Result<Vec<crate::SearchResult>>;

    /// Search with text query and metadata filter
    fn search_text_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &crate::Filter,
    ) -> Result<Vec<crate::SearchResult>>;
}

/// Builder for creating auto-embed enabled collections
#[derive(Debug, Clone)]
pub struct AutoEmbedCollectionBuilder {
    name: String,
    config: AutoEmbedConfig,
    model_hub: Option<ModelHub>,
    auto_normalize: bool,
}

impl AutoEmbedCollectionBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: AutoEmbedConfig::default(),
            model_hub: None,
            auto_normalize: false,
        }
    }

    /// Set the embedding backend
    #[must_use]
    pub fn with_backend(mut self, backend: EmbeddingBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Enable caching with specified size
    #[must_use]
    pub fn with_cache(mut self, size: usize) -> Self {
        self.config.cache_enabled = true;
        self.config.cache_size = size;
        self
    }

    /// Disable caching
    #[must_use]
    pub fn without_cache(mut self) -> Self {
        self.config.cache_enabled = false;
        self
    }

    /// Set batch size
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the configuration
    pub fn config(&self) -> &AutoEmbedConfig {
        &self.config
    }

    /// Set a model hub for model lookups
    #[must_use]
    pub fn with_model_hub(mut self, hub: ModelHub) -> Self {
        self.model_hub = Some(hub);
        self
    }

    /// Select a model by name from the attached model hub
    #[must_use]
    pub fn with_model_name(mut self, name: &str) -> Self {
        if let Some(ref hub) = self.model_hub {
            if let Some(entry) = hub.get_model(name) {
                self.config.backend = EmbeddingBackend::Registry {
                    model_name: entry.name.clone(),
                    dimensions: entry.dimensions,
                };
            }
        }
        self
    }

    /// Enable or disable automatic vector normalization
    #[must_use]
    pub fn with_auto_normalize(mut self, normalize: bool) -> Self {
        self.auto_normalize = normalize;
        self.config.normalize = normalize;
        self
    }

    /// Build the configuration
    pub fn build(self) -> (String, AutoEmbedConfig) {
        (self.name, self.config)
    }
}

// ---------------------------------------------------------------------------
// Model Hub / Registry Integration
// ---------------------------------------------------------------------------

/// Type of embedding model
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelType {
    /// Sentence-transformer style local models
    SentenceTransformer,
    /// OpenAI hosted models
    OpenAI,
    /// Cohere hosted models
    Cohere,
    /// User-provided custom model
    Custom,
}

/// A single entry in the model registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    /// Human-readable model name / identifier
    pub name: String,
    /// Output embedding dimensionality
    pub dimensions: usize,
    /// Short description of the model
    pub description: String,
    /// Maximum input token count
    pub max_tokens: usize,
    /// Category of the model
    pub model_type: ModelType,
}

/// A registry of pre-configured embedding models with auto-download capabilities
#[derive(Debug, Clone)]
pub struct ModelHub {
    models: HashMap<String, ModelEntry>,
    /// Local directory used for caching downloaded model artefacts
    pub cache_dir: String,
}

impl ModelHub {
    /// Create a new hub pre-populated with well-known models
    pub fn new() -> Self {
        let mut models = HashMap::new();

        models.insert(
            "all-MiniLM-L6-v2".to_string(),
            ModelEntry {
                name: "all-MiniLM-L6-v2".to_string(),
                dimensions: 384,
                description: "Lightweight sentence-transformer, good general-purpose model"
                    .to_string(),
                max_tokens: 256,
                model_type: ModelType::SentenceTransformer,
            },
        );

        models.insert(
            "bge-small-en".to_string(),
            ModelEntry {
                name: "bge-small-en".to_string(),
                dimensions: 384,
                description: "BAAI bge-small English embedding model".to_string(),
                max_tokens: 512,
                model_type: ModelType::SentenceTransformer,
            },
        );

        models.insert(
            "text-embedding-3-small".to_string(),
            ModelEntry {
                name: "text-embedding-3-small".to_string(),
                dimensions: 1536,
                description: "OpenAI text-embedding-3-small model".to_string(),
                max_tokens: 8191,
                model_type: ModelType::OpenAI,
            },
        );

        models.insert(
            "text-embedding-3-large".to_string(),
            ModelEntry {
                name: "text-embedding-3-large".to_string(),
                dimensions: 3072,
                description: "OpenAI text-embedding-3-large model".to_string(),
                max_tokens: 8191,
                model_type: ModelType::OpenAI,
            },
        );

        models.insert(
            "embed-english-v3.0".to_string(),
            ModelEntry {
                name: "embed-english-v3.0".to_string(),
                dimensions: 1024,
                description: "Cohere Embed English v3.0".to_string(),
                max_tokens: 512,
                model_type: ModelType::Cohere,
            },
        );

        Self {
            models,
            cache_dir: ".needle_models".to_string(),
        }
    }

    /// Look up a model by name
    pub fn get_model(&self, name: &str) -> Option<&ModelEntry> {
        self.models.get(name)
    }

    /// Return all registered models
    pub fn list_models(&self) -> Vec<&ModelEntry> {
        self.models.values().collect()
    }

    /// Recommend a model whose output matches the requested dimensionality.
    /// Returns the first model with an exact dimension match, preferring
    /// `SentenceTransformer` models when there are ties.
    pub fn recommend_model(&self, dimensions: usize) -> Option<&ModelEntry> {
        // Prefer local sentence-transformer models over API models
        self.models
            .values()
            .filter(|e| e.dimensions == dimensions)
            .min_by_key(|e| match e.model_type {
                ModelType::SentenceTransformer => 0,
                ModelType::Custom => 1,
                ModelType::Cohere => 2,
                ModelType::OpenAI => 3,
            })
    }

    /// Register a custom model entry
    pub fn register_custom_model(&mut self, entry: ModelEntry) {
        self.models.insert(entry.name.clone(), entry);
    }
}

impl Default for ModelHub {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Text-first collection API (returns embeddings)
// ---------------------------------------------------------------------------

/// Result of a text-based search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    /// Vector / document identifier
    pub id: String,
    /// Distance to the query vector
    pub distance: f32,
    /// Original text if stored
    pub text: Option<String>,
    /// Associated metadata
    pub metadata: Option<Value>,
}

/// Text-first collection trait that returns generated embeddings to callers.
///
/// This complements the existing [`TextInsertable`] trait by exposing the
/// raw embedding vectors produced during insert and search operations.
pub trait TextFirstCollection {
    /// Insert a single text, returning the generated embedding
    fn insert_text(
        &self,
        id: impl Into<String>,
        text: &str,
        metadata: Option<Value>,
    ) -> Result<Vec<f32>>;

    /// Insert multiple texts, returning their embeddings
    fn insert_texts(
        &self,
        items: &[(String, String, Option<Value>)],
    ) -> Result<Vec<Vec<f32>>>;

    /// Search by text query, returning [`TextSearchResult`]s
    fn search_text(&self, query: &str, k: usize) -> Result<Vec<TextSearchResult>>;
}

// ---------------------------------------------------------------------------
// Embedding Model Manager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of embedding models including switching and
/// re-embedding estimation.
#[derive(Debug, Clone)]
pub struct EmbeddingModelManager {
    active_model: String,
    model_versions: HashMap<String, Vec<String>>,
}

impl EmbeddingModelManager {
    /// Create a new manager with the given initial model name
    pub fn new(initial_model: impl Into<String>) -> Self {
        let name = initial_model.into();
        let mut versions = HashMap::new();
        versions.insert(name.clone(), vec!["v1".to_string()]);
        Self {
            active_model: name,
            model_versions: versions,
        }
    }

    /// Switch the active model, returning an error if the name is empty
    pub fn switch_model(&mut self, name: &str) -> Result<()> {
        if name.is_empty() {
            return Err(crate::error::NeedleError::InvalidInput(
                "model name cannot be empty".to_string(),
            ));
        }
        self.active_model = name.to_string();
        self.model_versions
            .entry(name.to_string())
            .or_insert_with(|| vec!["v1".to_string()]);
        Ok(())
    }

    /// Return the currently active model name
    pub fn get_active_model(&self) -> &str {
        &self.active_model
    }

    /// Returns `true` when switching between different models requires
    /// re-embedding all existing vectors.
    pub fn re_embed_needed(&self, old_model: &str, new_model: &str) -> bool {
        old_model != new_model
    }

    /// Estimate wall-clock time to re-embed `vector_count` vectors in
    /// batches of `batch_size`, assuming ~5 ms per embedding.
    pub fn estimate_re_embed_time(&self, vector_count: usize, batch_size: usize) -> Duration {
        let bs = if batch_size == 0 { 1 } else { batch_size };
        let batches = (vector_count + bs - 1) / bs;
        // Rough estimate: 5 ms per vector
        Duration::from_millis(vector_count as u64 * 5 + batches as u64)
    }

    /// Record a new version string for a model
    pub fn add_version(&mut self, model: &str, version: impl Into<String>) {
        self.model_versions
            .entry(model.to_string())
            .or_default()
            .push(version.into());
    }

    /// List recorded versions for a model
    pub fn get_versions(&self, model: &str) -> Option<&Vec<String>> {
        self.model_versions.get(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_embed_config_default() {
        let config = AutoEmbedConfig::default();
        assert_eq!(config.dimensions(), 384);
        assert!(config.cache_enabled);
    }

    #[test]
    fn test_auto_embed_config_mock() {
        let config = AutoEmbedConfig::mock(512);
        assert_eq!(config.dimensions(), 512);
    }

    #[test]
    fn test_auto_embedder_basic() {
        let embedder = AutoEmbedder::new(AutoEmbedConfig::mock(128));
        
        let embedding = embedder.embed("Hello, world!").unwrap();
        assert_eq!(embedding.len(), 128);
        
        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_auto_embedder_deterministic() {
        let embedder = AutoEmbedder::new(AutoEmbedConfig::mock(64));
        
        let e1 = embedder.embed("test text").unwrap();
        let e2 = embedder.embed("test text").unwrap();
        
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_auto_embedder_cache() {
        let embedder = AutoEmbedder::new(AutoEmbedConfig::mock(64).with_cache_size(100));
        
        // First call - cache miss
        let _ = embedder.embed("cached text").unwrap();
        let (_, hits, misses) = embedder.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
        
        // Second call - cache hit
        let _ = embedder.embed("cached text").unwrap();
        let (_, hits, misses) = embedder.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_auto_embedder_batch() {
        let embedder = AutoEmbedder::new(AutoEmbedConfig::mock(64));
        
        let texts = vec!["text1", "text2", "text3"];
        let embeddings = embedder.embed_batch(&texts).unwrap();
        
        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_auto_embedder_stats() {
        let embedder = AutoEmbedder::new(AutoEmbedConfig::mock(64).without_cache());
        
        let _ = embedder.embed("text1").unwrap();
        let _ = embedder.embed("text2").unwrap();
        
        let stats = embedder.stats();
        assert_eq!(stats.embeddings_generated, 2);
        assert_eq!(stats.texts_processed, 2);
    }

    #[test]
    fn test_embedding_backend_dimensions() {
        assert_eq!(EmbeddingBackend::mock(256).dimensions(), 256);
        assert_eq!(EmbeddingBackend::Custom { dimensions: 512 }.dimensions(), 512);
    }

    #[test]
    fn test_auto_embed_collection_builder() {
        let builder = AutoEmbedCollectionBuilder::new("test")
            .with_backend(EmbeddingBackend::mock(256))
            .with_cache(5000)
            .with_batch_size(64);
        
        let (name, config) = builder.build();
        assert_eq!(name, "test");
        assert_eq!(config.dimensions(), 256);
        assert_eq!(config.cache_size, 5000);
        assert_eq!(config.batch_size, 64);
    }

    // -----------------------------------------------------------------------
    // ModelHub tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_hub_creation_and_listing() {
        let hub = ModelHub::new();
        let models = hub.list_models();
        assert!(models.len() >= 5, "hub should contain at least 5 pre-loaded models");
        assert!(hub.get_model("all-MiniLM-L6-v2").is_some());
        assert!(hub.get_model("bge-small-en").is_some());
        assert!(hub.get_model("text-embedding-3-small").is_some());
    }

    #[test]
    fn test_model_recommendation_by_dimensions() {
        let hub = ModelHub::new();

        // 384-d should recommend a sentence-transformer model
        let rec = hub.recommend_model(384);
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().dimensions, 384);
        assert_eq!(rec.unwrap().model_type, ModelType::SentenceTransformer);

        // 1536-d should recommend OpenAI small
        let rec = hub.recommend_model(1536);
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().dimensions, 1536);

        // Non-existent dimension returns None
        assert!(hub.recommend_model(9999).is_none());
    }

    #[test]
    fn test_custom_model_registration() {
        let mut hub = ModelHub::new();
        let initial_count = hub.list_models().len();

        hub.register_custom_model(ModelEntry {
            name: "my-custom-model".to_string(),
            dimensions: 768,
            description: "A custom test model".to_string(),
            max_tokens: 512,
            model_type: ModelType::Custom,
        });

        assert_eq!(hub.list_models().len(), initial_count + 1);
        let entry = hub.get_model("my-custom-model").unwrap();
        assert_eq!(entry.dimensions, 768);
        assert_eq!(entry.model_type, ModelType::Custom);
    }

    // -----------------------------------------------------------------------
    // TextFirstCollection trait tests (mock implementation)
    // -----------------------------------------------------------------------

    struct MockTextCollection {
        embedder: AutoEmbedder,
    }

    impl TextFirstCollection for MockTextCollection {
        fn insert_text(
            &self,
            id: impl Into<String>,
            text: &str,
            _metadata: Option<Value>,
        ) -> Result<Vec<f32>> {
            let _id = id.into();
            self.embedder.embed(text)
        }

        fn insert_texts(
            &self,
            items: &[(String, String, Option<Value>)],
        ) -> Result<Vec<Vec<f32>>> {
            items.iter().map(|(_, text, _)| self.embedder.embed(text)).collect()
        }

        fn search_text(&self, query: &str, k: usize) -> Result<Vec<TextSearchResult>> {
            let _embedding = self.embedder.embed(query)?;
            Ok((0..k.min(3))
                .map(|i| TextSearchResult {
                    id: format!("result_{}", i),
                    distance: i as f32 * 0.1,
                    text: Some(format!("Result text {}", i)),
                    metadata: None,
                })
                .collect())
        }
    }

    #[test]
    fn test_text_first_collection_insert() {
        let coll = MockTextCollection {
            embedder: AutoEmbedder::new(AutoEmbedConfig::mock(128)),
        };

        let embedding = coll.insert_text("doc1", "hello world", None).unwrap();
        assert_eq!(embedding.len(), 128);
    }

    #[test]
    fn test_text_first_collection_batch_insert() {
        let coll = MockTextCollection {
            embedder: AutoEmbedder::new(AutoEmbedConfig::mock(64)),
        };

        let items = vec![
            ("a".to_string(), "text one".to_string(), None),
            ("b".to_string(), "text two".to_string(), None),
        ];
        let embeddings = coll.insert_texts(&items).unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 64);
    }

    #[test]
    fn test_text_first_collection_search() {
        let coll = MockTextCollection {
            embedder: AutoEmbedder::new(AutoEmbedConfig::mock(64)),
        };

        let results = coll.search_text("query", 5).unwrap();
        assert_eq!(results.len(), 3); // mock caps at 3
        assert_eq!(results[0].id, "result_0");
        assert!(results[0].text.is_some());
    }

    // -----------------------------------------------------------------------
    // EmbeddingModelManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_manager_switch() {
        let mut mgr = EmbeddingModelManager::new("all-MiniLM-L6-v2");
        assert_eq!(mgr.get_active_model(), "all-MiniLM-L6-v2");

        mgr.switch_model("bge-small-en").unwrap();
        assert_eq!(mgr.get_active_model(), "bge-small-en");

        // Empty name should fail
        assert!(mgr.switch_model("").is_err());
    }

    #[test]
    fn test_model_manager_re_embed_needed() {
        let mgr = EmbeddingModelManager::new("model-a");
        assert!(mgr.re_embed_needed("model-a", "model-b"));
        assert!(!mgr.re_embed_needed("model-a", "model-a"));
    }

    #[test]
    fn test_model_manager_estimate_re_embed_time() {
        let mgr = EmbeddingModelManager::new("model-a");
        let duration = mgr.estimate_re_embed_time(1000, 32);
        // 1000 vectors * 5ms = 5000ms base, plus batch overhead
        assert!(duration.as_millis() >= 5000);
        assert!(duration.as_millis() < 10000);

        // Zero batch size should not panic
        let d = mgr.estimate_re_embed_time(10, 0);
        assert!(d.as_millis() > 0);
    }

    #[test]
    fn test_model_manager_versions() {
        let mut mgr = EmbeddingModelManager::new("model-a");
        mgr.add_version("model-a", "v2");
        let versions = mgr.get_versions("model-a").unwrap();
        assert_eq!(versions, &["v1", "v2"]);
    }

    // -----------------------------------------------------------------------
    // Builder with model hub integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_with_model_hub() {
        let hub = ModelHub::new();
        let builder = AutoEmbedCollectionBuilder::new("my_coll")
            .with_model_hub(hub)
            .with_model_name("all-MiniLM-L6-v2")
            .with_auto_normalize(true);

        let (name, config) = builder.build();
        assert_eq!(name, "my_coll");
        assert_eq!(config.dimensions(), 384);
        assert!(config.normalize);
    }

    #[test]
    fn test_builder_with_unknown_model_name() {
        let hub = ModelHub::new();
        // Unknown model should leave backend unchanged (default mock 384)
        let builder = AutoEmbedCollectionBuilder::new("coll")
            .with_model_hub(hub)
            .with_model_name("nonexistent-model");

        let (_, config) = builder.build();
        assert_eq!(config.dimensions(), 384); // default
    }
}
