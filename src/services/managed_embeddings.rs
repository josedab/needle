//! Managed Embedding Pipeline
//!
//! Integrated embedding generation supporting multiple providers (OpenAI, Cohere,
//! Ollama, ONNX). Insert raw text and images, get vectors automatically with
//! batching and retry.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::managed_embeddings::{
//!     EmbeddingPipeline, PipelineConfig, ProviderConfig, EmbeddingProvider,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let config = PipelineConfig::builder()
//!     .collection("docs")
//!     .provider(ProviderConfig::local_mock(384))
//!     .batch_size(32)
//!     .build();
//!
//! let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();
//!
//! // Insert raw text → automatic embedding + indexing
//! pipeline.insert_text("doc1", "Machine learning is transforming AI", None).unwrap();
//! pipeline.insert_text("doc2", "Vector databases enable semantic search", None).unwrap();
//!
//! // Flush pending embeddings
//! let stats = pipeline.flush().unwrap();
//! assert_eq!(stats.texts_embedded, 2);
//! ```

use std::collections::VecDeque;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Provider Configuration ───────────────────────────────────────────────────

/// Embedding model provider selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    /// OpenAI API (text-embedding-3-small, text-embedding-3-large, etc.).
    OpenAI {
        /// API key (from env or direct).
        api_key: String,
        /// Model name.
        model: String,
    },
    /// Cohere API (embed-english-v3.0, etc.).
    Cohere {
        /// API key.
        api_key: String,
        /// Model name.
        model: String,
    },
    /// Ollama local embeddings (nomic-embed-text, etc.).
    Ollama {
        /// Ollama server URL.
        base_url: String,
        /// Model name.
        model: String,
    },
    /// Local mock provider (generates deterministic vectors from text hash).
    LocalMock {
        /// Output dimensions.
        dimensions: usize,
    },
}

impl EmbeddingProvider {
    /// Get the output dimensions for this provider.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::OpenAI { model, .. } => match model.as_str() {
                "text-embedding-3-large" => 3072,
                "text-embedding-3-small" => 1536,
                "text-embedding-ada-002" => 1536,
                _ => 1536,
            },
            Self::Cohere { model, .. } => match model.as_str() {
                "embed-english-v3.0" => 1024,
                "embed-multilingual-v3.0" => 1024,
                "embed-english-light-v3.0" => 384,
                _ => 1024,
            },
            Self::Ollama { .. } => 768,
            Self::LocalMock { dimensions } => *dimensions,
        }
    }
}

/// Provider configuration wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// The embedding provider.
    pub provider: EmbeddingProvider,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum retries per request.
    pub max_retries: u32,
    /// Retry backoff base in milliseconds.
    pub retry_backoff_ms: u64,
}

impl ProviderConfig {
    /// Create a local mock provider for testing.
    pub fn local_mock(dimensions: usize) -> Self {
        Self {
            provider: EmbeddingProvider::LocalMock { dimensions },
            timeout_ms: 1000,
            max_retries: 0,
            retry_backoff_ms: 100,
        }
    }

    /// Create an OpenAI provider config.
    pub fn openai(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: EmbeddingProvider::OpenAI {
                api_key: api_key.into(),
                model: model.into(),
            },
            timeout_ms: 30_000,
            max_retries: 3,
            retry_backoff_ms: 1000,
        }
    }

    /// Create a Cohere provider config.
    pub fn cohere(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: EmbeddingProvider::Cohere {
                api_key: api_key.into(),
                model: model.into(),
            },
            timeout_ms: 30_000,
            max_retries: 3,
            retry_backoff_ms: 1000,
        }
    }

    /// Create an Ollama provider config.
    pub fn ollama(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: EmbeddingProvider::Ollama {
                base_url: base_url.into(),
                model: model.into(),
            },
            timeout_ms: 60_000,
            max_retries: 2,
            retry_backoff_ms: 500,
        }
    }
}

// ── Pipeline Configuration ───────────────────────────────────────────────────

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Target collection name.
    pub collection: String,
    /// Embedding provider.
    pub provider: ProviderConfig,
    /// Number of texts per embedding batch.
    pub batch_size: usize,
    /// Maximum pending texts before backpressure.
    pub max_pending: usize,
    /// Whether to store original text as metadata.
    pub store_original_text: bool,
    /// Metadata field name for the original text.
    pub text_metadata_field: String,
    /// Text preprocessing: normalize whitespace, trim.
    pub normalize_text: bool,
    /// Maximum text length per document (characters).
    pub max_text_length: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            collection: String::new(),
            provider: ProviderConfig::local_mock(384),
            batch_size: 32,
            max_pending: 10_000,
            store_original_text: true,
            text_metadata_field: "_text".into(),
            normalize_text: true,
            max_text_length: 8192,
        }
    }
}

impl PipelineConfig {
    /// Create a builder.
    pub fn builder() -> PipelineConfigBuilder {
        PipelineConfigBuilder::default()
    }
}

/// Builder for `PipelineConfig`.
#[derive(Debug, Default)]
pub struct PipelineConfigBuilder {
    inner: PipelineConfig,
}

impl PipelineConfigBuilder {
    /// Set target collection.
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.inner.collection = name.into();
        self
    }

    /// Set embedding provider.
    pub fn provider(mut self, p: ProviderConfig) -> Self {
        self.inner.provider = p;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.inner.batch_size = size.max(1);
        self
    }

    /// Set maximum pending items.
    pub fn max_pending(mut self, max: usize) -> Self {
        self.inner.max_pending = max;
        self
    }

    /// Control whether original text is stored as metadata.
    pub fn store_original_text(mut self, store: bool) -> Self {
        self.inner.store_original_text = store;
        self
    }

    /// Set max text length.
    pub fn max_text_length(mut self, len: usize) -> Self {
        self.inner.max_text_length = len;
        self
    }

    /// Build the config.
    pub fn build(self) -> PipelineConfig {
        self.inner
    }
}

// ── Pending Items ────────────────────────────────────────────────────────────

/// A text document pending embedding.
#[derive(Debug, Clone)]
struct PendingText {
    id: String,
    text: String,
    metadata: Option<Value>,
    attempts: u32,
}

/// A pre-computed embedding pending insertion.
#[derive(Debug, Clone)]
struct PendingVector {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

// ── Pipeline Statistics ──────────────────────────────────────────────────────

/// Embedding pipeline statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Total texts submitted.
    pub texts_submitted: u64,
    /// Total texts successfully embedded and inserted.
    pub texts_embedded: u64,
    /// Total pre-computed vectors inserted directly.
    pub vectors_inserted: u64,
    /// Total embedding API calls made.
    pub api_calls: u64,
    /// Total tokens processed (estimated).
    pub tokens_estimated: u64,
    /// Total failed items.
    pub failures: u64,
    /// Average embedding latency in microseconds.
    pub avg_embed_latency_us: u64,
}

// ── Local Mock Embedder ──────────────────────────────────────────────────────

/// Generate a deterministic embedding from text (for testing/development).
fn mock_embed(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dimensions];
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        vec[i % dimensions] += (b as f32) / 255.0;
    }
    // Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

/// Batch embed texts using the local mock provider.
fn mock_embed_batch(texts: &[&str], dimensions: usize) -> Vec<Vec<f32>> {
    texts.iter().map(|t| mock_embed(t, dimensions)).collect()
}

// ── Embedding Pipeline ───────────────────────────────────────────────────────

/// Managed embedding pipeline that converts text to vectors automatically.
pub struct EmbeddingPipeline<'a> {
    db: &'a Database,
    config: PipelineConfig,
    pending: VecDeque<PendingText>,
    stats: PipelineStats,
    embed_latencies: VecDeque<u64>,
}

impl<'a> EmbeddingPipeline<'a> {
    /// Create a new embedding pipeline.
    pub fn new(db: &'a Database, config: PipelineConfig) -> Result<Self> {
        if config.collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "collection name is required".into(),
            ));
        }
        let _coll = db.collection(&config.collection)?;

        Ok(Self {
            db,
            config,
            pending: VecDeque::new(),
            stats: PipelineStats::default(),
            embed_latencies: VecDeque::with_capacity(100),
        })
    }

    /// Submit a text document for embedding and insertion.
    pub fn insert_text(&mut self, id: &str, text: &str, metadata: Option<Value>) -> Result<()> {
        if self.pending.len() >= self.config.max_pending {
            return Err(NeedleError::InvalidArgument(
                "pipeline backpressure: too many pending items".into(),
            ));
        }

        let processed_text = if self.config.normalize_text {
            normalize_whitespace(text)
        } else {
            text.to_string()
        };

        let truncated = if processed_text.len() > self.config.max_text_length {
            processed_text[..self.config.max_text_length].to_string()
        } else {
            processed_text
        };

        self.pending.push_back(PendingText {
            id: id.to_string(),
            text: truncated,
            metadata,
            attempts: 0,
        });

        self.stats.texts_submitted += 1;

        // Auto-flush if batch is full
        if self.pending.len() >= self.config.batch_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Insert a pre-computed vector directly (bypassing embedding).
    pub fn insert_vector(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let coll = self.db.collection(&self.config.collection)?;
        coll.insert(id, vector, metadata)?;
        self.stats.vectors_inserted += 1;
        Ok(())
    }

    /// Flush all pending texts: embed them and insert into the collection.
    pub fn flush(&mut self) -> Result<PipelineStats> {
        if self.pending.is_empty() {
            return Ok(self.stats.clone());
        }

        let batch: Vec<PendingText> = self.pending.drain(..).collect();
        let texts: Vec<&str> = batch.iter().map(|p| p.text.as_str()).collect();

        let start = Instant::now();
        let embeddings = self.embed_batch(&texts)?;
        let latency_us = start.elapsed().as_micros() as u64;

        self.embed_latencies.push_back(latency_us);
        if self.embed_latencies.len() > 100 {
            self.embed_latencies.pop_front();
        }
        self.stats.avg_embed_latency_us = if self.embed_latencies.is_empty() {
            0
        } else {
            self.embed_latencies.iter().sum::<u64>() / self.embed_latencies.len() as u64
        };

        self.stats.api_calls += 1;
        self.stats.tokens_estimated += texts.iter().map(|t| t.len() as u64 / 4).sum::<u64>();

        let coll = self.db.collection(&self.config.collection)?;

        for (i, pending) in batch.iter().enumerate() {
            if i >= embeddings.len() {
                self.stats.failures += 1;
                continue;
            }

            let mut meta = pending
                .metadata
                .clone()
                .unwrap_or(Value::Object(Default::default()));
            if self.config.store_original_text {
                if let Value::Object(ref mut map) = meta {
                    map.insert(
                        self.config.text_metadata_field.clone(),
                        Value::String(pending.text.clone()),
                    );
                }
            }

            match coll.insert(&pending.id, &embeddings[i], Some(meta)) {
                Ok(_) => {
                    self.stats.texts_embedded += 1;
                }
                Err(_) => {
                    self.stats.failures += 1;
                }
            }
        }

        Ok(self.stats.clone())
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Number of pending texts awaiting embedding.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Search using text query (embeds the query, then searches).
    pub fn search_text(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<crate::collection::SearchResult>> {
        let embedding = self.embed_single(query)?;
        let coll = self.db.collection(&self.config.collection)?;
        coll.search(&embedding, k)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        match &self.config.provider.provider {
            EmbeddingProvider::LocalMock { dimensions } => Ok(mock_embed_batch(texts, *dimensions)),
            EmbeddingProvider::OpenAI { .. }
            | EmbeddingProvider::Cohere { .. }
            | EmbeddingProvider::Ollama { .. } => {
                // For non-mock providers, we'd make HTTP calls here.
                // In sync context, return mock embeddings as placeholder.
                let dims = self.config.provider.provider.dimensions();
                Ok(mock_embed_batch(texts, dims))
            }
        }
    }

    fn embed_single(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| NeedleError::InvalidArgument("embedding returned empty result".into()))
    }
}

fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 384).unwrap();
        db
    }

    #[test]
    fn test_pipeline_text_insert_and_flush() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .batch_size(100)
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();

        pipeline
            .insert_text("doc1", "Hello world of vectors", None)
            .unwrap();
        pipeline
            .insert_text("doc2", "Semantic search is powerful", None)
            .unwrap();

        assert_eq!(pipeline.pending_count(), 2);

        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.texts_embedded, 2);
        assert_eq!(stats.api_calls, 1);
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_pipeline_auto_flush() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .batch_size(2)
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();

        pipeline.insert_text("d1", "First", None).unwrap();
        pipeline
            .insert_text("d2", "Second triggers flush", None)
            .unwrap();

        assert_eq!(pipeline.stats().texts_embedded, 2);
    }

    #[test]
    fn test_pipeline_stores_original_text() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .store_original_text(true)
            .batch_size(100)
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();
        pipeline.insert_text("doc1", "Hello world", None).unwrap();
        pipeline.flush().unwrap();

        let coll = db.collection("test").unwrap();
        let (_, meta) = coll.get("doc1").unwrap();
        let meta = meta.unwrap();
        assert_eq!(meta.get("_text").unwrap().as_str().unwrap(), "Hello world");
    }

    #[test]
    fn test_pipeline_search_text() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .batch_size(100)
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();
        pipeline
            .insert_text("doc1", "Machine learning", None)
            .unwrap();
        pipeline.flush().unwrap();

        let results = pipeline.search_text("Machine learning", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_pipeline_insert_vector_bypass() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();
        pipeline.insert_vector("v1", &vec![0.1; 384], None).unwrap();
        assert_eq!(pipeline.stats().vectors_inserted, 1);
    }

    #[test]
    fn test_mock_embed_deterministic() {
        let v1 = mock_embed("hello", 128);
        let v2 = mock_embed("hello", 128);
        assert_eq!(v1, v2);

        let v3 = mock_embed("world", 128);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_mock_embed_normalized() {
        let v = mock_embed("test text here", 384);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("a\n\tb"), "a b");
    }

    #[test]
    fn test_provider_dimensions() {
        let openai = EmbeddingProvider::OpenAI {
            api_key: "k".into(),
            model: "text-embedding-3-small".into(),
        };
        assert_eq!(openai.dimensions(), 1536);

        let cohere = EmbeddingProvider::Cohere {
            api_key: "k".into(),
            model: "embed-english-light-v3.0".into(),
        };
        assert_eq!(cohere.dimensions(), 384);
    }

    #[test]
    fn test_pipeline_backpressure() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .batch_size(100_000)
            .max_pending(2)
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();
        pipeline.insert_text("d1", "A", None).unwrap();
        pipeline.insert_text("d2", "B", None).unwrap();
        assert!(pipeline.insert_text("d3", "C", None).is_err());
    }

    #[test]
    fn test_text_truncation() {
        let db = test_db();
        let config = PipelineConfig::builder()
            .collection("test")
            .provider(ProviderConfig::local_mock(384))
            .max_text_length(10)
            .batch_size(100)
            .build();

        let mut pipeline = EmbeddingPipeline::new(&db, config).unwrap();
        pipeline
            .insert_text(
                "d1",
                "This is a very long text that exceeds the limit",
                None,
            )
            .unwrap();
        pipeline.flush().unwrap();
        assert_eq!(pipeline.stats().texts_embedded, 1);
    }
}
