//! Semantic Caching Layer
//!
//! Similarity-based LLM response cache: before querying an LLM, search for
//! semantically similar past queries and return cached responses if the
//! distance is below a configurable threshold.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::semantic_cache::{
//!     SemanticCache, CacheConfig, CacheEntry, CacheHit,
//! };
//!
//! let mut cache = SemanticCache::new(CacheConfig::new(384));
//!
//! // Store a query-response pair
//! let query_embedding = vec![0.1f32; 384];
//! cache.put(
//!     &query_embedding,
//!     "What is Rust?",
//!     "Rust is a systems programming language...",
//!     None,
//! ).unwrap();
//!
//! // Check cache for a similar query
//! let similar_query = vec![0.1001f32; 384];
//! if let Some(hit) = cache.get(&similar_query, None).unwrap() {
//!     println!("Cache hit! Distance: {}", hit.distance);
//!     println!("Cached response: {}", hit.response);
//! }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Semantic cache configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Similarity threshold (distance below which a cache hit is returned).
    /// Lower = more strict matching. Default: 0.15 (cosine distance).
    pub similarity_threshold: f32,
    /// Default TTL for cache entries.
    pub default_ttl: Option<Duration>,
    /// Maximum number of cached entries.
    pub max_entries: usize,
    /// Distance function.
    pub distance: DistanceFunction,
    /// Whether to track hit/miss analytics.
    pub enable_analytics: bool,
}

impl CacheConfig {
    /// Create a new config with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            similarity_threshold: 0.15,
            default_ttl: None,
            max_entries: 100_000,
            distance: DistanceFunction::Cosine,
            enable_analytics: true,
        }
    }

    /// Set similarity threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set default TTL.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = Some(ttl);
        self
    }

    /// Set max entries.
    #[must_use]
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }
}

// ── Cache Entry ──────────────────────────────────────────────────────────────

/// A cached query-response pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Original query text.
    pub query: String,
    /// Cached response text.
    pub response: String,
    /// Optional model name that generated the response.
    pub model: Option<String>,
    /// Timestamp when this entry was created.
    pub created_at: u64,
    /// Expiration timestamp (epoch seconds), if set.
    pub expires_at: Option<u64>,
    /// Number of times this entry was served as a hit.
    pub hit_count: u64,
    /// Optional additional metadata.
    pub metadata: Option<Value>,
}

// ── Cache Hit ────────────────────────────────────────────────────────────────

/// Result returned when a cache hit occurs.
#[derive(Debug, Clone)]
pub struct CacheHit {
    /// The cached response.
    pub response: String,
    /// Distance between query and cached entry.
    pub distance: f32,
    /// The original cached query.
    pub cached_query: String,
    /// The entry ID.
    pub entry_id: String,
    /// Model that generated this response.
    pub model: Option<String>,
}

// ── Analytics ────────────────────────────────────────────────────────────────

/// Cache analytics and statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheAnalytics {
    /// Total cache lookups.
    pub total_lookups: u64,
    /// Total cache hits.
    pub total_hits: u64,
    /// Total cache misses.
    pub total_misses: u64,
    /// Total entries stored.
    pub total_entries: usize,
    /// Total entries evicted.
    pub total_evictions: u64,
    /// Total entries expired.
    pub total_expirations: u64,
    /// Average hit distance.
    pub avg_hit_distance: f32,
}

impl CacheAnalytics {
    /// Cache hit rate (0.0–1.0).
    pub fn hit_rate(&self) -> f32 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.total_hits as f32 / self.total_lookups as f32
    }

    /// Estimated cost savings (assuming $0.002 per LLM query).
    pub fn estimated_savings_usd(&self, cost_per_query: f32) -> f32 {
        self.total_hits as f32 * cost_per_query
    }
}

// ── Semantic Cache ───────────────────────────────────────────────────────────

/// Similarity-based semantic cache for LLM responses.
pub struct SemanticCache {
    config: CacheConfig,
    collection: Collection,
    entries: HashMap<String, CacheEntry>,
    analytics: CacheAnalytics,
    next_id: u64,
}

impl SemanticCache {
    /// Create a new semantic cache.
    pub fn new(config: CacheConfig) -> Self {
        let coll_config = CollectionConfig::new("__semantic_cache__", config.dimensions)
            .with_distance(config.distance);
        let collection = Collection::new(coll_config);

        Self {
            config,
            collection,
            entries: HashMap::new(),
            analytics: CacheAnalytics::default(),
            next_id: 0,
        }
    }

    /// Store a query-response pair in the cache.
    pub fn put(
        &mut self,
        query_embedding: &[f32],
        query_text: &str,
        response: &str,
        model: Option<&str>,
    ) -> Result<String> {
        if query_embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query_embedding.len(),
            });
        }

        // Evict if at capacity
        if self.entries.len() >= self.config.max_entries {
            self.evict_oldest();
        }

        let id = format!("cache_{}", self.next_id);
        self.next_id += 1;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expires_at = self
            .config
            .default_ttl
            .map(|ttl| now + ttl.as_secs());

        let entry = CacheEntry {
            query: query_text.to_string(),
            response: response.to_string(),
            model: model.map(String::from),
            created_at: now,
            expires_at,
            hit_count: 0,
            metadata: None,
        };

        let metadata = serde_json::to_value(&entry).ok();
        self.collection
            .insert(id.clone(), query_embedding, metadata)?;
        self.entries.insert(id.clone(), entry);
        self.analytics.total_entries = self.entries.len();

        Ok(id)
    }

    /// Look up a semantically similar cached response.
    ///
    /// Returns `Some(CacheHit)` if a cached entry is within the similarity threshold.
    pub fn get(
        &mut self,
        query_embedding: &[f32],
        threshold_override: Option<f32>,
    ) -> Result<Option<CacheHit>> {
        self.analytics.total_lookups += 1;

        if self.entries.is_empty() {
            self.analytics.total_misses += 1;
            return Ok(None);
        }

        let threshold = threshold_override.unwrap_or(self.config.similarity_threshold);
        let results = self.collection.search(query_embedding, 1)?;

        if let Some(result) = results.first() {
            if result.distance <= threshold {
                if let Some(entry) = self.entries.get_mut(&result.id) {
                    // Check expiration
                    let now = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();

                    if let Some(expires) = entry.expires_at {
                        if now > expires {
                            self.analytics.total_expirations += 1;
                            self.analytics.total_misses += 1;
                            // Entry expired — remove it
                            let _ = self.collection.delete(&result.id);
                            self.entries.remove(&result.id);
                            return Ok(None);
                        }
                    }

                    entry.hit_count += 1;
                    self.analytics.total_hits += 1;

                    // Update running average distance
                    let hits = self.analytics.total_hits as f32;
                    self.analytics.avg_hit_distance = self.analytics.avg_hit_distance
                        * ((hits - 1.0) / hits)
                        + result.distance / hits;

                    return Ok(Some(CacheHit {
                        response: entry.response.clone(),
                        distance: result.distance,
                        cached_query: entry.query.clone(),
                        entry_id: result.id.clone(),
                        model: entry.model.clone(),
                    }));
                }
            }
        }

        self.analytics.total_misses += 1;
        Ok(None)
    }

    /// Invalidate a specific cache entry.
    pub fn invalidate(&mut self, entry_id: &str) -> Result<bool> {
        if self.entries.remove(entry_id).is_some() {
            let _ = self.collection.delete(entry_id);
            self.analytics.total_entries = self.entries.len();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all cache entries.
    pub fn clear(&mut self) {
        let ids: Vec<String> = self.entries.keys().cloned().collect();
        for id in ids {
            let _ = self.collection.delete(&id);
        }
        self.entries.clear();
        self.analytics.total_entries = 0;
    }

    /// Get cache analytics.
    pub fn analytics(&self) -> &CacheAnalytics {
        &self.analytics
    }

    /// Get the number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Sweep expired entries.
    pub fn sweep_expired(&mut self) -> usize {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| e.expires_at.map_or(false, |exp| now > exp))
            .map(|(id, _)| id.clone())
            .collect();

        let count = expired.len();
        for id in expired {
            let _ = self.collection.delete(&id);
            self.entries.remove(&id);
        }
        self.analytics.total_expirations += count as u64;
        self.analytics.total_entries = self.entries.len();
        count
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn evict_oldest(&mut self) {
        // Find the entry with the lowest hit_count (LFU-style)
        if let Some((id, _)) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.hit_count)
        {
            let id = id.clone();
            let _ = self.collection.delete(&id);
            self.entries.remove(&id);
            self.analytics.total_evictions += 1;
        }
    }

    /// Invalidate all cache entries that reference a specific vector ID.
    /// Call this when vectors are inserted/deleted/updated to keep cache consistent.
    pub fn invalidate_for_vector(&mut self, vector_id: &str) -> usize {
        let affected: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, e)| {
                e.response.contains(vector_id)
                    || e.query.contains(vector_id)
                    || e.metadata
                        .as_ref()
                        .map_or(false, |m| m.to_string().contains(vector_id))
            })
            .map(|(id, _)| id.clone())
            .collect();

        let count = affected.len();
        for id in affected {
            let _ = self.collection.delete(&id);
            self.entries.remove(&id);
        }
        self.analytics.total_entries = self.entries.len();
        count
    }

    /// Warm up the cache from a list of pre-computed query-response pairs.
    pub fn warm_up(
        &mut self,
        entries: Vec<(Vec<f32>, String, String, Option<String>)>,
    ) -> Result<usize> {
        let mut count = 0;
        for (embedding, query, response, model) in entries {
            self.put(&embedding, &query, &response, model.as_deref())?;
            count += 1;
        }
        Ok(count)
    }

    /// Export analytics in Prometheus exposition format.
    pub fn prometheus_metrics(&self) -> String {
        let a = &self.analytics;
        format!(
            "# HELP needle_cache_lookups_total Total cache lookups\n\
             # TYPE needle_cache_lookups_total counter\n\
             needle_cache_lookups_total {}\n\
             # HELP needle_cache_hits_total Total cache hits\n\
             # TYPE needle_cache_hits_total counter\n\
             needle_cache_hits_total {}\n\
             # HELP needle_cache_misses_total Total cache misses\n\
             # TYPE needle_cache_misses_total counter\n\
             needle_cache_misses_total {}\n\
             # HELP needle_cache_entries Current cached entries\n\
             # TYPE needle_cache_entries gauge\n\
             needle_cache_entries {}\n\
             # HELP needle_cache_evictions_total Total evictions\n\
             # TYPE needle_cache_evictions_total counter\n\
             needle_cache_evictions_total {}\n\
             # HELP needle_cache_hit_rate Cache hit rate\n\
             # TYPE needle_cache_hit_rate gauge\n\
             needle_cache_hit_rate {:.4}\n\
             # HELP needle_cache_avg_hit_distance Average hit distance\n\
             # TYPE needle_cache_avg_hit_distance gauge\n\
             needle_cache_avg_hit_distance {:.6}\n",
            a.total_lookups, a.total_hits, a.total_misses, a.total_entries,
            a.total_evictions, a.hit_rate(), a.avg_hit_distance,
        )
    }
}

// ── LLM Middleware ───────────────────────────────────────────────────────────

/// An LLM provider that can be wrapped with caching middleware.
pub trait LlmProvider {
    /// Generate a response for the given prompt.
    fn generate(&self, prompt: &str, model: Option<&str>) -> Result<String>;

    /// Embed a text string into a vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Cache middleware that wraps any `LlmProvider` with semantic caching.
///
/// Intercepts LLM calls, checks the cache, and only calls the underlying
/// provider on cache miss. Designed for integration with LangChain,
/// LlamaIndex, or any custom LLM pipeline.
pub struct CacheMiddleware<P: LlmProvider> {
    /// The underlying LLM provider.
    provider: P,
    /// The semantic cache.
    cache: SemanticCache,
    /// Model name for cache namespace isolation.
    model_name: Option<String>,
}

impl<P: LlmProvider> CacheMiddleware<P> {
    /// Create a new cache middleware wrapping the given provider.
    pub fn new(provider: P, cache_config: CacheConfig) -> Self {
        Self {
            provider,
            cache: SemanticCache::new(cache_config),
            model_name: None,
        }
    }

    /// Set the model name for cache namespace isolation.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model_name = Some(model.into());
        self
    }

    /// Generate a response, using cache when possible.
    ///
    /// 1. Embeds the prompt
    /// 2. Checks the semantic cache
    /// 3. On hit: returns cached response
    /// 4. On miss: calls the underlying provider, caches the result
    pub fn generate(&mut self, prompt: &str) -> Result<CachedResponse> {
        let embedding = self.provider.embed(prompt)?;

        // Check cache
        if let Some(hit) = self.cache.get(&embedding, None)? {
            // If model isolation is configured, verify the model matches
            if let Some(ref expected_model) = self.model_name {
                if hit.model.as_deref() != Some(expected_model.as_str()) {
                    // Different model — treat as miss
                    let response = self.provider.generate(prompt, self.model_name.as_deref())?;
                    self.cache
                        .put(&embedding, prompt, &response, self.model_name.as_deref())?;
                    return Ok(CachedResponse {
                        response,
                        from_cache: false,
                        distance: None,
                    });
                }
            }
            return Ok(CachedResponse {
                response: hit.response,
                from_cache: true,
                distance: Some(hit.distance),
            });
        }

        // Cache miss — call provider
        let response = self.provider.generate(prompt, self.model_name.as_deref())?;
        self.cache
            .put(&embedding, prompt, &response, self.model_name.as_deref())?;

        Ok(CachedResponse {
            response,
            from_cache: false,
            distance: None,
        })
    }

    /// Access the underlying cache for analytics, warming, etc.
    pub fn cache(&self) -> &SemanticCache {
        &self.cache
    }

    /// Access the underlying cache mutably.
    pub fn cache_mut(&mut self) -> &mut SemanticCache {
        &mut self.cache
    }

    /// Access the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }
}

/// Response from the cache middleware.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The generated (or cached) response.
    pub response: String,
    /// Whether this was a cache hit.
    pub from_cache: bool,
    /// Similarity distance (only present for cache hits).
    pub distance: Option<f32>,
}

// ── Batch Cache Warming ──────────────────────────────────────────────────────

/// Configuration for batch cache warming from a dataset.
#[derive(Debug, Clone)]
pub struct WarmUpConfig {
    /// Maximum number of entries to warm.
    pub max_entries: usize,
    /// Model name to associate with warmed entries.
    pub model: Option<String>,
}

impl Default for WarmUpConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            model: None,
        }
    }
}

/// Batch warm-up from query-response pairs with an embedding provider.
pub fn warm_cache_from_pairs<P: LlmProvider>(
    cache: &mut SemanticCache,
    provider: &P,
    pairs: &[(&str, &str)],
    config: &WarmUpConfig,
) -> Result<WarmUpResult> {
    let mut embedded = 0;
    let mut failed = 0;
    let limit = config.max_entries.min(pairs.len());

    for (query, response) in pairs.iter().take(limit) {
        match provider.embed(query) {
            Ok(embedding) => {
                cache.put(&embedding, query, response, config.model.as_deref())?;
                embedded += 1;
            }
            Err(_) => {
                failed += 1;
            }
        }
    }

    Ok(WarmUpResult { embedded, failed })
}

/// Result of a cache warming operation.
#[derive(Debug, Clone)]
pub struct WarmUpResult {
    /// Number of entries successfully warmed.
    pub embedded: usize,
    /// Number of entries that failed to embed.
    pub failed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(base: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| base + (i as f32) * 0.001).collect()
    }

    #[test]
    fn test_put_and_get() {
        let mut cache = SemanticCache::new(CacheConfig::new(32));

        let emb = make_embedding(0.5, 32);
        cache
            .put(&emb, "What is Rust?", "Rust is a language", Some("gpt-4"))
            .unwrap();

        // Exact same query should hit
        let hit = cache.get(&emb, None).unwrap();
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert_eq!(hit.response, "Rust is a language");
        assert_eq!(hit.model, Some("gpt-4".into()));
    }

    #[test]
    fn test_similar_query_hits() {
        let mut cache = SemanticCache::new(CacheConfig::new(32).with_threshold(0.2));

        let emb = make_embedding(0.5, 32);
        cache.put(&emb, "What is Rust?", "Rust answer", None).unwrap();

        // Slightly different embedding
        let similar = make_embedding(0.501, 32);
        let hit = cache.get(&similar, None).unwrap();
        assert!(hit.is_some());
    }

    #[test]
    fn test_dissimilar_query_misses() {
        let mut cache = SemanticCache::new(CacheConfig::new(32).with_threshold(0.01));

        let emb = make_embedding(0.5, 32);
        cache.put(&emb, "What is Rust?", "Rust answer", None).unwrap();

        // Very different embedding
        let different = make_embedding(-0.5, 32);
        let hit = cache.get(&different, None).unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn test_analytics() {
        let mut cache = SemanticCache::new(CacheConfig::new(16));
        let emb = make_embedding(0.5, 16);
        cache.put(&emb, "q", "r", None).unwrap();

        cache.get(&emb, None).unwrap(); // hit
        cache.get(&make_embedding(-1.0, 16), Some(0.001)).unwrap(); // miss

        let analytics = cache.analytics();
        assert_eq!(analytics.total_lookups, 2);
        assert_eq!(analytics.total_hits, 1);
        assert_eq!(analytics.total_misses, 1);
        assert!((analytics.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_invalidate() {
        let mut cache = SemanticCache::new(CacheConfig::new(8));
        let emb = make_embedding(0.5, 8);
        let id = cache.put(&emb, "q", "r", None).unwrap();

        assert!(cache.invalidate(&id).unwrap());
        assert!(!cache.invalidate(&id).unwrap()); // already gone
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = SemanticCache::new(CacheConfig::new(8));
        for i in 0..5 {
            let emb = make_embedding(i as f32 * 0.1, 8);
            cache.put(&emb, &format!("q{i}"), &format!("r{i}"), None).unwrap();
        }
        assert_eq!(cache.len(), 5);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut cache = SemanticCache::new(CacheConfig::new(8).with_max_entries(3));

        for i in 0..4 {
            let emb = make_embedding(i as f32 * 0.3, 8);
            cache.put(&emb, &format!("q{i}"), &format!("r{i}"), None).unwrap();
        }

        assert_eq!(cache.len(), 3); // one evicted
        assert_eq!(cache.analytics().total_evictions, 1);
    }

    #[test]
    fn test_cost_savings() {
        let mut analytics = CacheAnalytics::default();
        analytics.total_hits = 1000;
        assert!((analytics.estimated_savings_usd(0.002) - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_invalidate_for_vector() {
        let mut cache = SemanticCache::new(CacheConfig::new(8));
        let emb = make_embedding(0.5, 8);
        // Store a response that references vector "doc42"
        cache.put(&emb, "about doc42", "doc42 is important", None).unwrap();

        let evicted = cache.invalidate_for_vector("doc42");
        assert_eq!(evicted, 1);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_warm_up() {
        let mut cache = SemanticCache::new(CacheConfig::new(8));
        let entries = vec![
            (make_embedding(0.1, 8), "q1".into(), "r1".into(), None),
            (make_embedding(0.2, 8), "q2".into(), "r2".into(), Some("gpt-4".into())),
        ];

        let count = cache.warm_up(entries).unwrap();
        assert_eq!(count, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_prometheus_metrics() {
        let mut cache = SemanticCache::new(CacheConfig::new(8));
        let emb = make_embedding(0.5, 8);
        cache.put(&emb, "q", "r", None).unwrap();
        cache.get(&emb, None).unwrap();

        let metrics = cache.prometheus_metrics();
        assert!(metrics.contains("needle_cache_lookups_total 1"));
        assert!(metrics.contains("needle_cache_hits_total 1"));
        assert!(metrics.contains("needle_cache_entries 1"));
    }

    // ── Middleware Tests ──

    /// Mock LLM provider for testing.
    struct MockLlm {
        response: String,
        dimensions: usize,
    }

    impl MockLlm {
        fn new(response: &str, dims: usize) -> Self {
            Self {
                response: response.into(),
                dimensions: dims,
            }
        }
    }

    impl LlmProvider for MockLlm {
        fn generate(&self, _prompt: &str, _model: Option<&str>) -> Result<String> {
            Ok(self.response.clone())
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Simple deterministic embedding based on first char
            let seed = text.bytes().next().unwrap_or(0) as f32 / 255.0;
            Ok(vec![seed; self.dimensions])
        }
    }

    #[test]
    fn test_middleware_cache_miss() {
        let provider = MockLlm::new("Hello!", 8);
        let config = CacheConfig::new(8);
        let mut mw = CacheMiddleware::new(provider, config);

        let resp = mw.generate("test query").unwrap();
        assert_eq!(resp.response, "Hello!");
        assert!(!resp.from_cache);
        assert!(resp.distance.is_none());
        assert_eq!(mw.cache().len(), 1);
    }

    #[test]
    fn test_middleware_cache_hit() {
        let provider = MockLlm::new("Hello!", 8);
        let config = CacheConfig::new(8).with_threshold(0.5);
        let mut mw = CacheMiddleware::new(provider, config);

        // First call: miss
        mw.generate("test query").unwrap();

        // Second call with same query: hit
        let resp = mw.generate("test query").unwrap();
        assert!(resp.from_cache);
        assert!(resp.distance.is_some());
        assert_eq!(resp.response, "Hello!");
    }

    #[test]
    fn test_middleware_with_model() {
        let provider = MockLlm::new("model response", 8);
        let config = CacheConfig::new(8);
        let mut mw = CacheMiddleware::new(provider, config).with_model("gpt-4");

        let resp = mw.generate("hello").unwrap();
        assert!(!resp.from_cache);

        // Model-namespaced cache
        assert_eq!(mw.cache().len(), 1);
    }

    #[test]
    fn test_middleware_analytics() {
        let provider = MockLlm::new("resp", 8);
        let config = CacheConfig::new(8).with_threshold(0.5);
        let mut mw = CacheMiddleware::new(provider, config);

        mw.generate("query1").unwrap();
        mw.generate("query1").unwrap(); // hit
        mw.generate("query1").unwrap(); // hit

        let analytics = mw.cache().analytics();
        assert_eq!(analytics.total_hits, 2);
        assert_eq!(analytics.total_misses, 1);
        assert!(analytics.hit_rate() > 0.5);
    }

    #[test]
    fn test_warm_cache_from_pairs() {
        let provider = MockLlm::new("", 8);
        let config = CacheConfig::new(8);
        let mut cache = SemanticCache::new(config);

        let pairs = vec![
            ("what is rust?", "A systems programming language"),
            ("what is python?", "A high-level scripting language"),
            ("what is java?", "A cross-platform language"),
        ];

        let warm_config = WarmUpConfig {
            max_entries: 10,
            model: Some("test-model".into()),
        };

        let result = warm_cache_from_pairs(&mut cache, &provider, &pairs, &warm_config).unwrap();
        assert_eq!(result.embedded, 3);
        assert_eq!(result.failed, 0);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_warm_cache_max_entries() {
        let provider = MockLlm::new("", 8);
        let config = CacheConfig::new(8);
        let mut cache = SemanticCache::new(config);

        let pairs = vec![
            ("q1", "r1"),
            ("q2", "r2"),
            ("q3", "r3"),
        ];

        let warm_config = WarmUpConfig {
            max_entries: 2,
            model: None,
        };

        let result = warm_cache_from_pairs(&mut cache, &provider, &pairs, &warm_config).unwrap();
        assert_eq!(result.embedded, 2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cached_response_fields() {
        let hit = CachedResponse {
            response: "test".into(),
            from_cache: true,
            distance: Some(0.05),
        };
        assert!(hit.from_cache);
        assert_eq!(hit.distance, Some(0.05));

        let miss = CachedResponse {
            response: "test".into(),
            from_cache: false,
            distance: None,
        };
        assert!(!miss.from_cache);
        assert!(miss.distance.is_none());
    }
}
