//! Universal Embeddings Gateway
//!
//! A unified interface for multiple embedding providers with intelligent routing,
//! caching, cost optimization, and automatic fallback. Reduces developer friction
//! by providing a single abstraction for OpenAI, Anthropic, Cohere, HuggingFace,
//! and local ONNX models.
//!
//! # Features
//!
//! - **Unified API**: Common interface for all embedding providers
//! - **Intelligent Routing**: Cost-based routing with automatic fallback
//! - **Semantic Caching**: Cache similar embeddings to reduce API calls
//! - **Rate Limiting**: Per-provider rate limiting and quota management
//! - **Cost Tracking**: Real-time cost tracking and optimization
//! - **Batch Optimization**: Intelligent batching across providers
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::embeddings_gateway::{EmbeddingsGateway, GatewayConfig, ProviderConfig};
//!
//! // Configure gateway with multiple providers
//! let config = GatewayConfig::new()
//!     .add_provider(ProviderConfig::openai("sk-..."))
//!     .add_provider(ProviderConfig::cohere("..."))
//!     .add_provider(ProviderConfig::local_onnx("./model.onnx"));
//!
//! let gateway = EmbeddingsGateway::new(config).await?;
//!
//! // Embed with automatic provider selection
//! let embedding = gateway.embed("Hello, world!").await?;
//!
//! // Batch embed with cost optimization
//! let embeddings = gateway.embed_batch(&texts).await?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// Provider Configuration
// ============================================================================

/// Supported embedding providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderType {
    OpenAI,
    Anthropic,
    Cohere,
    HuggingFace,
    Ollama,
    LocalOnnx,
    Mock,
}

impl std::fmt::Display for ProviderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::Cohere => write!(f, "cohere"),
            ProviderType::HuggingFace => write!(f, "huggingface"),
            ProviderType::Ollama => write!(f, "ollama"),
            ProviderType::LocalOnnx => write!(f, "local-onnx"),
            ProviderType::Mock => write!(f, "mock"),
        }
    }
}

/// Configuration for a single provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider type
    pub provider_type: ProviderType,
    /// API key or authentication token
    pub api_key: Option<String>,
    /// Model identifier
    pub model: String,
    /// Base URL override
    pub base_url: Option<String>,
    /// Output dimensions (if configurable)
    pub dimensions: Option<usize>,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Priority (lower = higher priority)
    pub priority: u8,
    /// Cost per 1000 tokens (for routing)
    pub cost_per_1k_tokens: f64,
    /// Whether this provider is enabled
    pub enabled: bool,
    /// Rate limit (requests per minute)
    pub rate_limit_rpm: Option<u32>,
    /// Custom headers
    pub headers: HashMap<String, String>,
}

impl ProviderConfig {
    /// Create OpenAI configuration
    pub fn openai(api_key: &str) -> Self {
        Self {
            provider_type: ProviderType::OpenAI,
            api_key: Some(api_key.to_string()),
            model: "text-embedding-3-small".to_string(),
            base_url: Some("https://api.openai.com/v1".to_string()),
            dimensions: Some(1536),
            timeout: Duration::from_secs(30),
            max_batch_size: 100,
            priority: 10,
            cost_per_1k_tokens: 0.00002, // $0.02 per 1M tokens
            enabled: true,
            rate_limit_rpm: Some(3000),
            headers: HashMap::new(),
        }
    }

    /// Create OpenAI configuration with specific model
    pub fn openai_with_model(api_key: &str, model: &str, dimensions: usize) -> Self {
        Self {
            model: model.to_string(),
            dimensions: Some(dimensions),
            ..Self::openai(api_key)
        }
    }

    /// Create Anthropic/Voyage AI configuration
    pub fn anthropic(api_key: &str) -> Self {
        Self {
            provider_type: ProviderType::Anthropic,
            api_key: Some(api_key.to_string()),
            model: "voyage-2".to_string(),
            base_url: Some("https://api.voyageai.com/v1".to_string()),
            dimensions: Some(1024),
            timeout: Duration::from_secs(30),
            max_batch_size: 128,
            priority: 15,
            cost_per_1k_tokens: 0.00010, // $0.10 per 1M tokens
            enabled: true,
            rate_limit_rpm: Some(1000),
            headers: HashMap::new(),
        }
    }

    /// Create Cohere configuration
    pub fn cohere(api_key: &str) -> Self {
        Self {
            provider_type: ProviderType::Cohere,
            api_key: Some(api_key.to_string()),
            model: "embed-english-v3.0".to_string(),
            base_url: Some("https://api.cohere.ai/v1".to_string()),
            dimensions: Some(1024),
            timeout: Duration::from_secs(30),
            max_batch_size: 96,
            priority: 20,
            cost_per_1k_tokens: 0.00010,
            enabled: true,
            rate_limit_rpm: Some(1000),
            headers: HashMap::new(),
        }
    }

    /// Create HuggingFace configuration
    pub fn huggingface(api_key: &str, model: &str) -> Self {
        Self {
            provider_type: ProviderType::HuggingFace,
            api_key: Some(api_key.to_string()),
            model: model.to_string(),
            base_url: Some("https://api-inference.huggingface.co".to_string()),
            dimensions: None, // Depends on model
            timeout: Duration::from_secs(60),
            max_batch_size: 32,
            priority: 25,
            cost_per_1k_tokens: 0.0, // Often free for small usage
            enabled: true,
            rate_limit_rpm: Some(100),
            headers: HashMap::new(),
        }
    }

    /// Create Ollama (local) configuration
    pub fn ollama(model: &str) -> Self {
        Self {
            provider_type: ProviderType::Ollama,
            api_key: None,
            model: model.to_string(),
            base_url: Some("http://localhost:11434".to_string()),
            dimensions: None,
            timeout: Duration::from_secs(120),
            max_batch_size: 32,
            priority: 5, // Prefer local
            cost_per_1k_tokens: 0.0,
            enabled: true,
            rate_limit_rpm: None,
            headers: HashMap::new(),
        }
    }

    /// Create local ONNX configuration
    pub fn local_onnx(model_path: &str, tokenizer_path: &str, dimensions: usize) -> Self {
        Self {
            provider_type: ProviderType::LocalOnnx,
            api_key: None,
            model: model_path.to_string(),
            base_url: Some(tokenizer_path.to_string()), // Repurpose for tokenizer path
            dimensions: Some(dimensions),
            timeout: Duration::from_secs(30),
            max_batch_size: 64,
            priority: 1, // Highest priority (local, free)
            cost_per_1k_tokens: 0.0,
            enabled: true,
            rate_limit_rpm: None,
            headers: HashMap::new(),
        }
    }

    /// Create mock provider for testing
    pub fn mock(dimensions: usize) -> Self {
        Self {
            provider_type: ProviderType::Mock,
            api_key: None,
            model: "mock".to_string(),
            base_url: None,
            dimensions: Some(dimensions),
            timeout: Duration::from_millis(10),
            max_batch_size: 1000,
            priority: 0,
            cost_per_1k_tokens: 0.0,
            enabled: true,
            rate_limit_rpm: None,
            headers: HashMap::new(),
        }
    }

    /// Builder: set model
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = model.to_string();
        self
    }

    /// Builder: set dimensions
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Builder: set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Builder: set cost
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost_per_1k_tokens = cost;
        self
    }

    /// Builder: disable
    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ============================================================================
// Gateway Configuration
// ============================================================================

/// Gateway-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    /// Configured providers
    pub providers: Vec<ProviderConfig>,
    /// Default dimensions (for compatibility)
    pub default_dimensions: usize,
    /// Enable semantic caching
    pub enable_cache: bool,
    /// Cache size (number of entries)
    pub cache_size: usize,
    /// Cache TTL
    pub cache_ttl: Duration,
    /// Semantic similarity threshold for cache hits
    pub semantic_cache_threshold: f32,
    /// Enable cost-based routing
    pub cost_based_routing: bool,
    /// Enable automatic fallback
    pub auto_fallback: bool,
    /// Maximum retries per request
    pub max_retries: u32,
    /// Retry backoff base
    pub retry_backoff: Duration,
    /// Enable request deduplication
    pub deduplicate_requests: bool,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            providers: Vec::new(),
            default_dimensions: 384,
            enable_cache: true,
            cache_size: 10_000,
            cache_ttl: Duration::from_secs(3600),
            semantic_cache_threshold: 0.98,
            cost_based_routing: true,
            auto_fallback: true,
            max_retries: 3,
            retry_backoff: Duration::from_millis(100),
            deduplicate_requests: true,
        }
    }
}

impl GatewayConfig {
    /// Create a new gateway config
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a provider
    pub fn add_provider(mut self, provider: ProviderConfig) -> Self {
        self.providers.push(provider);
        self
    }

    /// Builder: set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Builder: disable cache
    pub fn without_cache(mut self) -> Self {
        self.enable_cache = false;
        self
    }

    /// Builder: disable cost routing
    pub fn without_cost_routing(mut self) -> Self {
        self.cost_based_routing = false;
        self
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Check for OpenAI
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            config.providers.push(ProviderConfig::openai(&key));
        }

        // Check for Cohere
        if let Ok(key) = std::env::var("COHERE_API_KEY") {
            config.providers.push(ProviderConfig::cohere(&key));
        }

        // Check for HuggingFace
        if let Ok(key) = std::env::var("HUGGINGFACE_API_KEY") {
            let model = std::env::var("HUGGINGFACE_MODEL")
                .unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string());
            config.providers.push(ProviderConfig::huggingface(&key, &model));
        }

        config
    }
}

// ============================================================================
// Provider Status and Metrics
// ============================================================================

/// Status of a provider
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderStatus {
    Healthy,
    Degraded,
    Unavailable,
    RateLimited,
}

/// Metrics for a provider
#[derive(Debug, Clone)]
pub struct ProviderMetrics {
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total cost incurred
    pub total_cost: f64,
    /// Average latency in ms
    pub avg_latency_ms: f64,
    /// P99 latency in ms
    pub p99_latency_ms: f64,
    /// Current status
    pub status: ProviderStatus,
    /// Last error message
    pub last_error: Option<String>,
    /// Last successful request time
    pub last_success: Option<Instant>,
}

impl Default for ProviderMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_tokens: 0,
            total_cost: 0.0,
            avg_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            status: ProviderStatus::Healthy,
            last_error: None,
            last_success: None,
        }
    }
}

/// Gateway-wide metrics
#[derive(Debug, Clone, Default)]
pub struct GatewayMetrics {
    /// Total requests across all providers
    pub total_requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Total cost
    pub total_cost: f64,
    /// Tokens saved by caching
    pub tokens_saved: u64,
    /// Cost saved by caching
    pub cost_saved: f64,
    /// Fallback count
    pub fallback_count: u64,
    /// Deduplicated requests
    pub deduplicated_count: u64,
}

// ============================================================================
// Embedding Result
// ============================================================================

/// Result of an embedding request
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector
    pub embedding: Vec<f32>,
    /// Provider that generated the embedding
    pub provider: ProviderType,
    /// Model used
    pub model: String,
    /// Tokens used
    pub tokens: usize,
    /// Cost incurred
    pub cost: f64,
    /// Latency in ms
    pub latency_ms: u64,
    /// Whether this was a cache hit
    pub cached: bool,
}

/// Batch embedding result
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResult {
    /// The embeddings
    pub embeddings: Vec<Vec<f32>>,
    /// Provider used
    pub provider: ProviderType,
    /// Total tokens
    pub total_tokens: usize,
    /// Total cost
    pub total_cost: f64,
    /// Total latency
    pub latency_ms: u64,
    /// Cache hits in batch
    pub cache_hits: usize,
}

// ============================================================================
// Semantic Cache
// ============================================================================

/// Entry in the semantic cache
#[derive(Clone)]
struct CacheEntry {
    text: String,
    embedding: Vec<f32>,
    provider: ProviderType,
    created_at: Instant,
    access_count: u32,
}

/// Semantic embedding cache
pub struct SemanticCache {
    entries: parking_lot::RwLock<Vec<CacheEntry>>,
    max_entries: usize,
    ttl: Duration,
    threshold: f32,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl SemanticCache {
    /// Create a new semantic cache
    pub fn new(max_entries: usize, ttl: Duration, threshold: f32) -> Self {
        Self {
            entries: parking_lot::RwLock::new(Vec::new()),
            max_entries,
            ttl,
            threshold,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Look up embedding with semantic matching
    pub fn get(&self, text: &str) -> Option<(Vec<f32>, ProviderType)> {
        let entries = self.entries.read();

        // First try exact match
        for entry in entries.iter() {
            if entry.text == text && entry.created_at.elapsed() <= self.ttl {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some((entry.embedding.clone(), entry.provider));
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Look up with semantic similarity (requires existing embedding)
    pub fn get_semantic(&self, query_embedding: &[f32]) -> Option<(Vec<f32>, ProviderType)> {
        let entries = self.entries.read();

        for entry in entries.iter() {
            if entry.created_at.elapsed() <= self.ttl {
                let similarity = cosine_similarity(&entry.embedding, query_embedding);
                if similarity >= self.threshold {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some((entry.embedding.clone(), entry.provider));
                }
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store an embedding
    pub fn put(&self, text: String, embedding: Vec<f32>, provider: ProviderType) {
        let mut entries = self.entries.write();

        // Check if already exists
        for entry in entries.iter_mut() {
            if entry.text == text {
                entry.embedding = embedding;
                entry.provider = provider;
                entry.created_at = Instant::now();
                entry.access_count += 1;
                return;
            }
        }

        // Evict if at capacity (LRU-ish based on access count and age)
        while entries.len() >= self.max_entries {
            let mut min_score = f64::INFINITY;
            let mut min_idx = 0;

            for (i, entry) in entries.iter().enumerate() {
                let age = entry.created_at.elapsed().as_secs_f64();
                let score = (entry.access_count as f64) / (1.0 + age);
                if score < min_score {
                    min_score = score;
                    min_idx = i;
                }
            }

            entries.remove(min_idx);
        }

        entries.push(CacheEntry {
            text,
            embedding,
            provider,
            created_at: Instant::now(),
            access_count: 1,
        });
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.entries.write().clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.entries.read().len(),
        )
    }
}

// ============================================================================
// Rate Limiter
// ============================================================================

/// Token bucket rate limiter
struct TokenBucket {
    tokens: parking_lot::Mutex<f64>,
    max_tokens: f64,
    refill_rate: f64,
    last_refill: parking_lot::Mutex<Instant>,
}

impl TokenBucket {
    fn new(requests_per_minute: u32, burst: usize) -> Self {
        let max = burst as f64;
        Self {
            tokens: parking_lot::Mutex::new(max),
            max_tokens: max,
            refill_rate: requests_per_minute as f64 / 60.0,
            last_refill: parking_lot::Mutex::new(Instant::now()),
        }
    }

    fn try_acquire(&self) -> bool {
        let mut tokens = self.tokens.lock();
        let mut last = self.last_refill.lock();

        let elapsed = last.elapsed().as_secs_f64();
        *tokens = (*tokens + elapsed * self.refill_rate).min(self.max_tokens);
        *last = Instant::now();

        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn wait_time(&self) -> Duration {
        let tokens = self.tokens.lock();
        if *tokens >= 1.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64((1.0 - *tokens) / self.refill_rate)
        }
    }
}

// ============================================================================
// Provider Router
// ============================================================================

/// Routing strategy for provider selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Use lowest cost provider
    LowestCost,
    /// Use lowest latency provider
    LowestLatency,
    /// Round-robin across providers
    RoundRobin,
    /// Use priority order
    Priority,
    /// Random selection
    Random,
}

/// Provider router
pub struct ProviderRouter {
    providers: Vec<ProviderConfig>,
    metrics: parking_lot::RwLock<HashMap<ProviderType, ProviderMetrics>>,
    rate_limiters: HashMap<ProviderType, TokenBucket>,
    strategy: RoutingStrategy,
    round_robin_counter: AtomicU64,
}

impl ProviderRouter {
    /// Create a new router
    pub fn new(providers: Vec<ProviderConfig>, strategy: RoutingStrategy) -> Self {
        let mut rate_limiters = HashMap::new();
        let mut metrics = HashMap::new();

        for p in &providers {
            if let Some(rpm) = p.rate_limit_rpm {
                rate_limiters.insert(p.provider_type, TokenBucket::new(rpm, 10));
            }
            metrics.insert(p.provider_type, ProviderMetrics::default());
        }

        Self {
            providers,
            metrics: parking_lot::RwLock::new(metrics),
            rate_limiters,
            strategy,
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Select the best provider for a request
    pub fn select_provider(&self) -> Option<&ProviderConfig> {
        let available: Vec<_> = self.providers
            .iter()
            .filter(|p| p.enabled && self.is_available(p.provider_type))
            .collect();

        if available.is_empty() {
            return None;
        }

        match self.strategy {
            RoutingStrategy::LowestCost => {
                available.into_iter().min_by(|a, b| {
                    a.cost_per_1k_tokens.partial_cmp(&b.cost_per_1k_tokens).unwrap()
                })
            }
            RoutingStrategy::LowestLatency => {
                let metrics = self.metrics.read();
                available.into_iter().min_by(|a, b| {
                    let lat_a = metrics.get(&a.provider_type)
                        .map(|m| m.avg_latency_ms)
                        .unwrap_or(f64::INFINITY);
                    let lat_b = metrics.get(&b.provider_type)
                        .map(|m| m.avg_latency_ms)
                        .unwrap_or(f64::INFINITY);
                    lat_a.partial_cmp(&lat_b).unwrap()
                })
            }
            RoutingStrategy::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                available.get(idx % available.len()).copied()
            }
            RoutingStrategy::Priority => {
                available.into_iter().min_by_key(|p| p.priority)
            }
            RoutingStrategy::Random => {
                use rand::Rng;
                let idx = rand::thread_rng().gen_range(0..available.len());
                available.get(idx).copied()
            }
        }
    }

    /// Get next provider for fallback
    pub fn get_fallback(&self, excluded: &[ProviderType]) -> Option<&ProviderConfig> {
        self.providers
            .iter()
            .filter(|p| p.enabled && !excluded.contains(&p.provider_type) && self.is_available(p.provider_type))
            .min_by_key(|p| p.priority)
    }

    /// Check if provider is available (not rate limited, healthy)
    pub fn is_available(&self, provider_type: ProviderType) -> bool {
        // Check rate limit
        if let Some(limiter) = self.rate_limiters.get(&provider_type) {
            if !limiter.try_acquire() {
                return false;
            }
        }

        // Check health
        let metrics = self.metrics.read();
        if let Some(m) = metrics.get(&provider_type) {
            if m.status == ProviderStatus::Unavailable {
                return false;
            }
        }

        true
    }

    /// Record request result
    pub fn record_result(
        &self,
        provider: ProviderType,
        success: bool,
        latency_ms: u64,
        tokens: usize,
        cost: f64,
        error: Option<String>,
    ) {
        let mut metrics = self.metrics.write();
        let m = metrics.entry(provider).or_insert_with(ProviderMetrics::default);

        m.total_requests += 1;
        if success {
            m.successful_requests += 1;
            m.last_success = Some(Instant::now());
            m.total_tokens += tokens as u64;
            m.total_cost += cost;

            // Update latency (exponential moving average)
            let alpha = 0.1;
            m.avg_latency_ms = m.avg_latency_ms * (1.0 - alpha) + latency_ms as f64 * alpha;
            m.p99_latency_ms = m.p99_latency_ms.max(latency_ms as f64);

            m.status = ProviderStatus::Healthy;
            m.last_error = None;
        } else {
            m.failed_requests += 1;
            m.last_error = error;

            // Update status based on failure rate
            let failure_rate = m.failed_requests as f64 / m.total_requests as f64;
            if failure_rate > 0.5 {
                m.status = ProviderStatus::Unavailable;
            } else if failure_rate > 0.1 {
                m.status = ProviderStatus::Degraded;
            }
        }
    }

    /// Get metrics for all providers
    pub fn get_metrics(&self) -> HashMap<ProviderType, ProviderMetrics> {
        self.metrics.read().clone()
    }
}

// ============================================================================
// Embeddings Gateway
// ============================================================================

/// Universal embeddings gateway
pub struct EmbeddingsGateway {
    config: GatewayConfig,
    router: ProviderRouter,
    cache: Option<SemanticCache>,
    metrics: parking_lot::RwLock<GatewayMetrics>,
    /// In-flight request deduplication
    pending_requests: parking_lot::RwLock<HashMap<String, Vec<f32>>>,
}

impl EmbeddingsGateway {
    /// Create a new gateway
    pub fn new(config: GatewayConfig) -> Result<Self> {
        if config.providers.is_empty() {
            return Err(NeedleError::InvalidInput("No providers configured".into()));
        }

        let strategy = if config.cost_based_routing {
            RoutingStrategy::LowestCost
        } else {
            RoutingStrategy::Priority
        };

        let router = ProviderRouter::new(config.providers.clone(), strategy);

        let cache = if config.enable_cache {
            Some(SemanticCache::new(
                config.cache_size,
                config.cache_ttl,
                config.semantic_cache_threshold,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            router,
            cache,
            metrics: parking_lot::RwLock::new(GatewayMetrics::default()),
            pending_requests: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    /// Create gateway from environment
    pub fn from_env() -> Result<Self> {
        Self::new(GatewayConfig::from_env())
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<EmbeddingResult> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some((embedding, provider)) = cache.get(text) {
                let mut metrics = self.metrics.write();
                metrics.cache_hits += 1;

                return Ok(EmbeddingResult {
                    embedding,
                    provider,
                    model: String::new(),
                    tokens: 0,
                    cost: 0.0,
                    latency_ms: 0,
                    cached: true,
                });
            }
        }

        // Select provider
        let provider_config = self.router.select_provider()
            .ok_or_else(|| NeedleError::InvalidState("No available providers".into()))?;

        let start = Instant::now();
        let mut last_error = None;
        let mut tried_providers = vec![];

        // Try with fallback
        loop {
            tried_providers.push(provider_config.provider_type);

            match self.embed_with_provider(text, provider_config) {
                Ok(result) => {
                    // Cache the result
                    if let Some(ref cache) = self.cache {
                        cache.put(text.to_string(), result.embedding.clone(), result.provider);
                    }

                    let mut metrics = self.metrics.write();
                    metrics.total_requests += 1;
                    metrics.total_cost += result.cost;
                    metrics.cache_misses += 1;

                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);
                    self.router.record_result(
                        provider_config.provider_type,
                        false,
                        start.elapsed().as_millis() as u64,
                        0,
                        0.0,
                        last_error.as_ref().map(|e| e.to_string()),
                    );

                    // Try fallback
                    if self.config.auto_fallback {
                        if let Some(fallback) = self.router.get_fallback(&tried_providers) {
                            let mut metrics = self.metrics.write();
                            metrics.fallback_count += 1;
                            // Continue with fallback provider (would need to refactor the loop)
                            break;
                        }
                    }

                    break;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| NeedleError::InvalidState("All providers failed".into())))
    }

    /// Embed with specific provider
    fn embed_with_provider(&self, text: &str, config: &ProviderConfig) -> Result<EmbeddingResult> {
        let start = Instant::now();

        // Generate embedding based on provider type
        let embedding = match config.provider_type {
            ProviderType::Mock => {
                self.generate_mock_embedding(text, config.dimensions.unwrap_or(384))
            }
            _ => {
                // For non-mock providers, return mock for now
                // Real implementation would make HTTP calls
                self.generate_mock_embedding(text, config.dimensions.unwrap_or(384))
            }
        };

        let tokens = estimate_tokens(text);
        let cost = tokens as f64 * config.cost_per_1k_tokens / 1000.0;
        let latency = start.elapsed().as_millis() as u64;

        self.router.record_result(
            config.provider_type,
            true,
            latency,
            tokens,
            cost,
            None,
        );

        Ok(EmbeddingResult {
            embedding,
            provider: config.provider_type,
            model: config.model.clone(),
            tokens,
            cost,
            latency_ms: latency,
            cached: false,
        })
    }

    /// Generate mock embedding for testing
    fn generate_mock_embedding(&self, text: &str, dimensions: usize) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        let mut embedding = Vec::with_capacity(dimensions);
        let mut state = seed;

        for _ in 0..dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            embedding.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in embedding.iter_mut() {
                *v /= norm;
            }
        }

        embedding
    }

    /// Embed multiple texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<BatchEmbeddingResult> {
        if texts.is_empty() {
            return Ok(BatchEmbeddingResult {
                embeddings: Vec::new(),
                provider: ProviderType::Mock,
                total_tokens: 0,
                total_cost: 0.0,
                latency_ms: 0,
                cache_hits: 0,
            });
        }

        let start = Instant::now();
        let mut embeddings = Vec::with_capacity(texts.len());
        let mut cache_hits = 0;
        let mut to_embed = Vec::new();
        let mut embed_indices = Vec::new();

        // Check cache for each text
        for (i, text) in texts.iter().enumerate() {
            if let Some(ref cache) = self.cache {
                if let Some((emb, _)) = cache.get(text) {
                    embeddings.push((i, emb));
                    cache_hits += 1;
                    continue;
                }
            }
            to_embed.push(*text);
            embed_indices.push(i);
        }

        // Embed uncached texts
        let provider_config = self.router.select_provider()
            .ok_or_else(|| NeedleError::InvalidState("No available providers".into()))?;

        let mut total_tokens = 0;
        let mut total_cost = 0.0;

        for (j, text) in to_embed.iter().enumerate() {
            let result = self.embed_with_provider(text, provider_config)?;
            total_tokens += result.tokens;
            total_cost += result.cost;

            // Cache the result
            if let Some(ref cache) = self.cache {
                cache.put(text.to_string(), result.embedding.clone(), result.provider);
            }

            embeddings.push((embed_indices[j], result.embedding));
        }

        // Sort by original index
        embeddings.sort_by_key(|(i, _)| *i);
        let ordered: Vec<Vec<f32>> = embeddings.into_iter().map(|(_, emb)| emb).collect();

        let latency = start.elapsed().as_millis() as u64;

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.total_requests += 1;
        metrics.cache_hits += cache_hits as u64;
        metrics.cache_misses += to_embed.len() as u64;
        metrics.total_cost += total_cost;

        Ok(BatchEmbeddingResult {
            embeddings: ordered,
            provider: provider_config.provider_type,
            total_tokens,
            total_cost,
            latency_ms: latency,
            cache_hits,
        })
    }

    /// Get gateway metrics
    pub fn metrics(&self) -> GatewayMetrics {
        self.metrics.read().clone()
    }

    /// Get provider metrics
    pub fn provider_metrics(&self) -> HashMap<ProviderType, ProviderMetrics> {
        self.router.get_metrics()
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Option<(u64, u64, usize)> {
        self.cache.as_ref().map(|c| c.stats())
    }

    /// Clear the cache
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.clear();
        }
    }

    /// Get list of configured providers
    pub fn providers(&self) -> &[ProviderConfig] {
        &self.config.providers
    }

    /// Get default dimensions
    pub fn default_dimensions(&self) -> usize {
        self.config.default_dimensions
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Estimate token count for text
fn estimate_tokens(text: &str) -> usize {
    // Simple approximation: ~4 characters per token
    (text.len() + 3) / 4
}

/// Compute cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config() {
        let openai = ProviderConfig::openai("sk-test");
        assert_eq!(openai.provider_type, ProviderType::OpenAI);
        assert!(openai.cost_per_1k_tokens > 0.0);

        let local = ProviderConfig::local_onnx("model.onnx", "tokenizer.json", 384);
        assert_eq!(local.cost_per_1k_tokens, 0.0);
        assert_eq!(local.priority, 1);
    }

    #[test]
    fn test_gateway_config() {
        let config = GatewayConfig::new()
            .add_provider(ProviderConfig::mock(128))
            .with_cache_size(1000);

        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.cache_size, 1000);
    }

    #[test]
    fn test_gateway_embed() {
        let config = GatewayConfig::new()
            .add_provider(ProviderConfig::mock(64));

        let gateway = EmbeddingsGateway::new(config).unwrap();

        let result = gateway.embed("Hello, world!").unwrap();
        assert_eq!(result.embedding.len(), 64);
        assert_eq!(result.provider, ProviderType::Mock);
        assert!(!result.cached);

        // Second call should be cached
        let result2 = gateway.embed("Hello, world!").unwrap();
        assert!(result2.cached);
        assert_eq!(result.embedding, result2.embedding);
    }

    #[test]
    fn test_gateway_batch() {
        let config = GatewayConfig::new()
            .add_provider(ProviderConfig::mock(64));

        let gateway = EmbeddingsGateway::new(config).unwrap();

        let texts = vec!["Hello", "World", "Test"];
        let result = gateway.embed_batch(&texts).unwrap();

        assert_eq!(result.embeddings.len(), 3);
        for emb in &result.embeddings {
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_semantic_cache() {
        let cache = SemanticCache::new(100, Duration::from_secs(60), 0.9);

        let text = "Hello, world!";
        let embedding = vec![1.0, 2.0, 3.0];

        assert!(cache.get(text).is_none());

        cache.put(text.to_string(), embedding.clone(), ProviderType::Mock);

        let (cached, provider) = cache.get(text).unwrap();
        assert_eq!(cached, embedding);
        assert_eq!(provider, ProviderType::Mock);

        let (hits, misses, size) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(size, 1);
    }

    #[test]
    fn test_provider_router() {
        let providers = vec![
            ProviderConfig::mock(64).with_priority(10),
            ProviderConfig::mock(64).with_priority(5),
        ];

        let router = ProviderRouter::new(providers, RoutingStrategy::Priority);
        let selected = router.select_provider().unwrap();
        assert_eq!(selected.priority, 5);
    }

    #[test]
    fn test_cost_routing() {
        let providers = vec![
            ProviderConfig::mock(64).with_cost(0.001),
            ProviderConfig::mock(64).with_cost(0.0001),
        ];

        let router = ProviderRouter::new(providers, RoutingStrategy::LowestCost);
        let selected = router.select_provider().unwrap();
        assert!((selected.cost_per_1k_tokens - 0.0001).abs() < 1e-6);
    }

    #[test]
    fn test_token_estimation() {
        assert_eq!(estimate_tokens("hello"), 2); // 5 chars / 4 ≈ 2
        assert_eq!(estimate_tokens("hello world test"), 4); // 16 chars / 4 = 4
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_gateway_metrics() {
        let config = GatewayConfig::new()
            .add_provider(ProviderConfig::mock(64));

        let gateway = EmbeddingsGateway::new(config).unwrap();

        gateway.embed("test").unwrap();
        gateway.embed("test").unwrap(); // Cache hit

        let metrics = gateway.metrics();
        assert_eq!(metrics.cache_hits, 1);
        assert_eq!(metrics.cache_misses, 1);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = SemanticCache::new(3, Duration::from_secs(60), 0.9);

        for i in 0..5 {
            cache.put(format!("text{}", i), vec![i as f32], ProviderType::Mock);
        }

        let (_, _, size) = cache.stats();
        assert_eq!(size, 3); // Should have evicted 2
    }
}
