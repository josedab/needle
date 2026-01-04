//! Embedding Providers
//!
//! Abstraction layer for various embedding model providers including OpenAI, Cohere,
//! Ollama, and a mock provider for testing. This module provides a unified interface
//! for generating embeddings from different sources with support for batching, caching,
//! and rate limiting.
//!
//! # Features
//!
//! - **Unified API**: Common trait for all embedding providers
//! - **Batching**: Efficient batch processing of embedding requests
//! - **Caching**: LRU cache layer to avoid redundant API calls
//! - **Rate Limiting**: Token bucket rate limiter for API compliance
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::embeddings_provider::{
//!     EmbeddingProvider, OpenAIProvider, OpenAIConfig, CachedProvider,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = OpenAIConfig::new("your-api-key".to_string());
//!     let provider = OpenAIProvider::new(config);
//!     let cached = CachedProvider::new(provider, 1000);
//!
//!     let embedding = cached.embed("Hello, world!".to_string()).await?;
//!     println!("Embedding dimensions: {}", embedding.len());
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock, Semaphore};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during embedding operations
#[derive(Error, Debug)]
pub enum EmbeddingProviderError {
    #[error("API error: {message} (status: {status_code:?})")] ApiError { message: String, status_code: Option<u16> },
    #[error("Rate limit exceeded: {0}")] RateLimitExceeded(String),
    #[error("Authentication failed: {0}")] AuthenticationError(String),
    #[error("Invalid input: {0}")] InvalidInput(String),
    #[error("Network error: {0}")] NetworkError(String),
    #[error("Request timeout after {0:?}")] Timeout(Duration),
    #[error("Provider unavailable: {0}")] ProviderUnavailable(String),
    #[error("Configuration error: {0}")] ConfigurationError(String),
    #[error("Serialization error: {0}")] SerializationError(String),
    #[error("Model not found: {0}")] ModelNotFound(String),
    #[error("Batch size {size} exceeds maximum {max_size}")] BatchTooLarge { size: usize, max_size: usize },
    #[error("Initialization error: {0}")] InitializationError(String),
}

pub type Result<T> = std::result::Result<T, EmbeddingProviderError>;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for batching embedding requests
#[derive(Debug, Clone)]
pub struct BatchConfig { pub max_batch_size: usize, pub batch_delay: Duration }

impl Default for BatchConfig {
    fn default() -> Self { Self { max_batch_size: 100, batch_delay: Duration::from_millis(50) } }
}

/// OpenAI provider configuration
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String, pub model: String, pub base_url: String,
    pub timeout: Duration, pub dimensions: Option<usize>,
}

impl OpenAIConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "text-embedding-3-small".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            timeout: Duration::from_secs(30),
            dimensions: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }
}

/// Cohere provider configuration
#[derive(Debug, Clone)]
pub struct CohereConfig {
    pub api_key: String, pub model: String, pub base_url: String,
    pub timeout: Duration, pub input_type: CohereInputType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohereInputType { SearchQuery, SearchDocument, Classification, Clustering }

impl CohereInputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SearchQuery => "search_query", Self::SearchDocument => "search_document",
            Self::Classification => "classification", Self::Clustering => "clustering",
        }
    }
}

impl CohereConfig {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            model: "embed-english-v3.0".to_string(),
            base_url: "https://api.cohere.ai/v1".to_string(),
            timeout: Duration::from_secs(30),
            input_type: CohereInputType::SearchDocument,
        }
    }
}

/// Ollama provider configuration
#[derive(Debug, Clone)]
pub struct OllamaConfig { pub model: String, pub base_url: String, pub timeout: Duration }

impl Default for OllamaConfig {
    fn default() -> Self {
        Self { model: "nomic-embed-text".into(), base_url: "http://localhost:11434".into(), timeout: Duration::from_secs(60) }
    }
}

/// Mock provider configuration for testing
#[derive(Debug, Clone)]
pub struct MockConfig {
    pub dimensions: usize,
    pub latency: Duration,
    pub normalize: bool,
    pub seed: Option<u64>,
}

impl Default for MockConfig {
    fn default() -> Self {
        Self { dimensions: 384, latency: Duration::from_millis(10), normalize: true, seed: Some(42) }
    }
}

impl MockConfig {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions, ..Default::default() }
    }
}

// ============================================================================
// Core Trait
// ============================================================================

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &str;
    fn dimensions(&self) -> usize;
    fn max_batch_size(&self) -> usize;

    async fn embed(&self, text: String) -> Result<Vec<f32>> {
        let results = self.embed_batch(vec![text]).await?;
        results.into_iter().next()
            .ok_or_else(|| EmbeddingProviderError::InvalidInput("Empty result".into()))
    }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    async fn health_check(&self) -> Result<()> {
        self.embed("test".to_string()).await.map(|_| ())
    }
}

// ============================================================================
// OpenAI Provider
// ============================================================================

pub struct OpenAIProvider {
    config: OpenAIConfig,
    client: reqwest::Client,
    dimensions: usize,
}

impl OpenAIProvider {
    pub fn new(config: OpenAIConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");
        let dimensions = config.dimensions.unwrap_or(1536);
        Self { config, client, dimensions }
    }

    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| EmbeddingProviderError::ConfigurationError("OPENAI_API_KEY not set".into()))?;
        Ok(Self::new(OpenAIConfig::new(api_key)))
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str { "openai" }
    fn dimensions(&self) -> usize { self.dimensions }
    fn max_batch_size(&self) -> usize { 100 }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(Vec::new()); }
        let mut body = serde_json::json!({ "model": self.config.model, "input": texts });
        if let Some(dims) = self.config.dimensions {
            body["dimensions"] = serde_json::json!(dims);
        }

        let resp = self.client
            .post(format!("{}/embeddings", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&body)
            .send().await
            .map_err(|e| EmbeddingProviderError::NetworkError(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(EmbeddingProviderError::ApiError {
                message: resp.text().await.unwrap_or_default(),
                status_code: Some(status.as_u16()),
            });
        }

        let result: serde_json::Value = resp.json().await
            .map_err(|e| EmbeddingProviderError::SerializationError(e.to_string()))?;

        result["data"].as_array()
            .ok_or_else(|| EmbeddingProviderError::SerializationError("Missing data".into()))?
            .iter()
            .map(|item| {
                item["embedding"].as_array()
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                    .ok_or_else(|| EmbeddingProviderError::SerializationError("Missing embedding".into()))
            })
            .collect()
    }
}

// ============================================================================
// Cohere Provider
// ============================================================================

pub struct CohereProvider {
    config: CohereConfig,
    client: reqwest::Client,
}

impl CohereProvider {
    pub fn new(config: CohereConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbeddingProviderError::InitializationError(format!("Failed to create HTTP client: {}", e)))?;
        Ok(Self { config, client })
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    fn name(&self) -> &str { "cohere" }
    fn dimensions(&self) -> usize { 1024 }
    fn max_batch_size(&self) -> usize { 96 }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(Vec::new()); }
        let body = serde_json::json!({
            "model": self.config.model,
            "texts": texts,
            "input_type": self.config.input_type.as_str(),
        });

        let resp = self.client
            .post(format!("{}/embed", self.config.base_url))
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&body)
            .send().await
            .map_err(|e| EmbeddingProviderError::NetworkError(e.to_string()))?;

        if !resp.status().is_success() {
            let status_code = resp.status().as_u16();
            let message = resp.text().await.unwrap_or_default();
            return Err(EmbeddingProviderError::ApiError {
                message,
                status_code: Some(status_code),
            });
        }

        let result: serde_json::Value = resp.json().await
            .map_err(|e| EmbeddingProviderError::SerializationError(e.to_string()))?;

        Ok(result["embeddings"].as_array().unwrap_or(&vec![]).iter()
            .map(|arr| arr.as_array().unwrap_or(&vec![]).iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .collect())
    }
}

// ============================================================================
// Ollama Provider
// ============================================================================

pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
    dimensions: Arc<RwLock<Option<usize>>>,
}

impl OllamaProvider {
    pub fn new(config: OllamaConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| EmbeddingProviderError::InitializationError(format!("Failed to create HTTP client: {}", e)))?;
        Ok(Self { config, client, dimensions: Arc::new(RwLock::new(None)) })
    }
}

#[async_trait]
impl EmbeddingProvider for OllamaProvider {
    fn name(&self) -> &str { "ollama" }
    fn dimensions(&self) -> usize { self.dimensions.try_read().ok().and_then(|d| *d).unwrap_or(768) }
    fn max_batch_size(&self) -> usize { 32 }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(Vec::new()); }
        let mut embeddings = Vec::with_capacity(texts.len());

        for text in texts {
            let body = serde_json::json!({ "model": self.config.model, "prompt": text });
            let resp = self.client
                .post(format!("{}/api/embeddings", self.config.base_url))
                .json(&body)
                .send().await
                .map_err(|e| EmbeddingProviderError::NetworkError(e.to_string()))?;

            if !resp.status().is_success() {
                let status_code = resp.status().as_u16();
                let message = resp.text().await.unwrap_or_default();
                return Err(EmbeddingProviderError::ApiError {
                    message,
                    status_code: Some(status_code),
                });
            }

            let result: serde_json::Value = resp.json().await
                .map_err(|e| EmbeddingProviderError::SerializationError(e.to_string()))?;

            let emb: Vec<f32> = result["embedding"].as_array()
                .ok_or_else(|| EmbeddingProviderError::SerializationError("Missing embedding".into()))?
                .iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();

            if self.dimensions.read().await.is_none() {
                *self.dimensions.write().await = Some(emb.len());
            }
            embeddings.push(emb);
        }
        Ok(embeddings)
    }
}

// ============================================================================
// Mock Provider
// ============================================================================

pub struct MockProvider {
    config: MockConfig,
    call_count: AtomicU64,
}

impl MockProvider {
    pub fn new(config: MockConfig) -> Self {
        Self { config, call_count: AtomicU64::new(0) }
    }

    pub fn call_count(&self) -> u64 { self.call_count.load(Ordering::Relaxed) }

    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish().wrapping_add(self.config.seed.unwrap_or(0));

        let mut emb = Vec::with_capacity(self.config.dimensions);
        let mut state = seed;
        for _ in 0..self.config.dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            emb.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
        }

        if self.config.normalize {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 { emb.iter_mut().for_each(|v| *v /= norm); }
        }
        emb
    }
}

#[async_trait]
impl EmbeddingProvider for MockProvider {
    fn name(&self) -> &str { "mock" }
    fn dimensions(&self) -> usize { self.config.dimensions }
    fn max_batch_size(&self) -> usize { 1000 }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        if self.config.latency > Duration::ZERO {
            tokio::time::sleep(self.config.latency).await;
        }
        Ok(texts.iter().map(|t| self.generate_embedding(t)).collect())
    }
}

// ============================================================================
// Caching Layer
// ============================================================================

struct CacheEntry { embedding: Vec<f32>, created_at: Instant }

pub struct EmbeddingCache {
    entries: Mutex<HashMap<String, CacheEntry>>,
    max_entries: usize, ttl: Option<Duration>, hits: AtomicU64, misses: AtomicU64,
}

impl EmbeddingCache {
    pub fn new(max_entries: usize) -> Self {
        Self { entries: Mutex::new(HashMap::new()), max_entries, ttl: None,
               hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self { self.ttl = Some(ttl); self }

    pub async fn get(&self, text: &str) -> Option<Vec<f32>> {
        let entries = self.entries.lock().await;
        if let Some(e) = entries.get(text) {
            if self.ttl.is_none_or(|ttl| e.created_at.elapsed() <= ttl) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(e.embedding.clone());
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    pub async fn put(&self, text: String, embedding: Vec<f32>) {
        let mut entries = self.entries.lock().await;
        while entries.len() >= self.max_entries {
            if let Some(k) = entries.iter().min_by_key(|(_, e)| e.created_at).map(|(k, _)| k.clone()) {
                entries.remove(&k);
            } else { break; }
        }
        entries.insert(text, CacheEntry { embedding, created_at: Instant::now() });
    }

    pub fn stats(&self) -> (u64, u64) { (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed)) }
}

pub struct CachedProvider<P: EmbeddingProvider> {
    inner: P,
    cache: EmbeddingCache,
}

impl<P: EmbeddingProvider> CachedProvider<P> {
    pub fn new(provider: P, max_entries: usize) -> Self {
        Self { inner: provider, cache: EmbeddingCache::new(max_entries) }
    }

    pub fn cache_stats(&self) -> (u64, u64) { self.cache.stats() }
}

#[async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for CachedProvider<P> {
    fn name(&self) -> &str { self.inner.name() }
    fn dimensions(&self) -> usize { self.inner.dimensions() }
    fn max_batch_size(&self) -> usize { self.inner.max_batch_size() }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut results = vec![None; texts.len()];
        let mut uncached = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(emb) = self.cache.get(text).await { results[i] = Some(emb); }
            else { uncached.push((i, text.clone())); }
        }

        if !uncached.is_empty() {
            let uncached_texts: Vec<_> = uncached.iter().map(|(_, t)| t.clone()).collect();
            let new_embs = self.inner.embed_batch(uncached_texts).await?;
            for ((i, text), emb) in uncached.into_iter().zip(new_embs) {
                self.cache.put(text, emb.clone()).await;
                results[i] = Some(emb);
            }
        }
        results.into_iter()
            .enumerate()
            .map(|(i, r)| r.ok_or_else(|| EmbeddingProviderError::InvalidInput(
                format!("Missing embedding result at index {}", i)
            )))
            .collect()
    }
}

// ============================================================================
// Rate Limiting
// ============================================================================

pub struct RateLimiter {
    tokens: Mutex<f64>, max_tokens: f64, refill_rate: f64,
    last_refill: Mutex<Instant>, semaphore: Semaphore,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32, burst_size: usize) -> Self {
        let max = burst_size as f64;
        Self {
            tokens: Mutex::new(max), max_tokens: max,
            refill_rate: requests_per_minute as f64 / 60.0,
            last_refill: Mutex::new(Instant::now()),
            semaphore: Semaphore::new(burst_size),
        }
    }

    pub async fn acquire(&self) -> Result<()> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| EmbeddingProviderError::RateLimitExceeded("Semaphore closed".into()))?;
        loop {
            let mut last = self.last_refill.lock().await;
            let mut tokens = self.tokens.lock().await;
            *tokens = (*tokens + last.elapsed().as_secs_f64() * self.refill_rate).min(self.max_tokens);
            *last = Instant::now();
            if *tokens >= 1.0 { *tokens -= 1.0; return Ok(()); }
            drop(tokens); drop(last);
            tokio::time::sleep(Duration::from_secs_f64(1.0 / self.refill_rate)).await;
        }
    }
}

pub struct RateLimitedProvider<P: EmbeddingProvider> {
    inner: P,
    limiter: RateLimiter,
}

impl<P: EmbeddingProvider> RateLimitedProvider<P> {
    pub fn new(provider: P, requests_per_minute: u32, burst_size: usize) -> Self {
        Self { inner: provider, limiter: RateLimiter::new(requests_per_minute, burst_size) }
    }
}

#[async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for RateLimitedProvider<P> {
    fn name(&self) -> &str { self.inner.name() }
    fn dimensions(&self) -> usize { self.inner.dimensions() }
    fn max_batch_size(&self) -> usize { self.inner.max_batch_size() }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.limiter.acquire().await?;
        self.inner.embed_batch(texts).await
    }
}

// ============================================================================
// Batching Provider
// ============================================================================

pub struct BatchingProvider<P: EmbeddingProvider> {
    inner: P,
    config: BatchConfig,
}

impl<P: EmbeddingProvider> BatchingProvider<P> {
    pub fn new(provider: P, config: BatchConfig) -> Self {
        Self { inner: provider, config }
    }
}

#[async_trait]
impl<P: EmbeddingProvider> EmbeddingProvider for BatchingProvider<P> {
    fn name(&self) -> &str { self.inner.name() }
    fn dimensions(&self) -> usize { self.inner.dimensions() }
    fn max_batch_size(&self) -> usize { self.config.max_batch_size }

    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(Vec::new()); }
        let mut all = Vec::with_capacity(texts.len());

        for (i, chunk) in texts.chunks(self.config.max_batch_size).enumerate() {
            if i > 0 && self.config.batch_delay > Duration::ZERO {
                tokio::time::sleep(self.config.batch_delay).await;
            }
            all.extend(self.inner.embed_batch(chunk.to_vec()).await?);
        }
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_provider_basic() {
        let provider = MockProvider::new(MockConfig::new(128));
        let emb = provider.embed("hello".to_string()).await.unwrap();
        assert_eq!(emb.len(), 128);
        assert_eq!(provider.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_provider_deterministic() {
        let p = MockProvider::new(MockConfig::new(64));
        let e1 = p.embed("test".to_string()).await.unwrap();
        let e2 = p.embed("test".to_string()).await.unwrap();
        assert_eq!(e1, e2);
    }

    #[tokio::test]
    async fn test_mock_normalized() {
        let p = MockProvider::new(MockConfig::new(64));
        let emb = p.embed("test".to_string()).await.unwrap();
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_cache_hit_miss() {
        let cache = EmbeddingCache::new(100);
        assert!(cache.get("test").await.is_none());
        cache.put("test".to_string(), vec![1.0, 2.0]).await;
        assert_eq!(cache.get("test").await, Some(vec![1.0, 2.0]));
        assert_eq!(cache.stats(), (1, 1));
    }

    #[tokio::test]
    async fn test_cache_eviction() {
        let cache = EmbeddingCache::new(2);
        cache.put("a".to_string(), vec![1.0]).await;
        cache.put("b".to_string(), vec![2.0]).await;
        cache.put("c".to_string(), vec![3.0]).await;
        assert!(cache.get("a").await.is_none());
    }

    #[tokio::test]
    async fn test_cached_provider() {
        let cached = CachedProvider::new(MockProvider::new(MockConfig::new(32)), 100);
        let e1 = cached.embed("hello".to_string()).await.unwrap();
        let e2 = cached.embed("hello".to_string()).await.unwrap();
        assert_eq!(e1, e2);
        assert_eq!(cached.cache_stats(), (1, 1));
    }

    #[tokio::test]
    async fn test_batching_provider() {
        let batch = BatchingProvider::new(
            MockProvider::new(MockConfig::new(32)),
            BatchConfig { max_batch_size: 2, batch_delay: Duration::ZERO },
        );
        let texts: Vec<_> = (0..5).map(|i| format!("t{}", i)).collect();
        assert_eq!(batch.embed_batch(texts).await.unwrap().len(), 5);
    }

    #[tokio::test]
    async fn test_provider_chain() {
        let chain = BatchingProvider::new(
            RateLimitedProvider::new(
                CachedProvider::new(MockProvider::new(MockConfig::new(64)), 1000),
                100, 10,
            ),
            BatchConfig { max_batch_size: 10, batch_delay: Duration::ZERO },
        );
        let texts: Vec<_> = (0..15).map(|i| format!("text{}", i)).collect();
        let e1 = chain.embed_batch(texts.clone()).await.unwrap();
        let e2 = chain.embed_batch(texts).await.unwrap();
        assert_eq!(e1, e2);
    }

    #[tokio::test]
    async fn test_trait_object() {
        let p: Box<dyn EmbeddingProvider> = Box::new(MockProvider::new(MockConfig::new(128)));
        assert_eq!(p.name(), "mock");
        assert_eq!(p.dimensions(), 128);
    }

    #[test]
    fn test_configs() {
        let o = OpenAIConfig::new("k".into()).with_model("lg").with_dimensions(3072);
        assert_eq!(o.dimensions, Some(3072));
        assert_eq!(CohereConfig::new("k".into()).input_type.as_str(), "search_document");
        assert_eq!(OllamaConfig::default().model, "nomic-embed-text");
    }
}
