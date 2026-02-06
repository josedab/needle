//! Semantic Caching for LLM Applications
//!
//! Provides intelligent caching of LLM responses based on semantic similarity
//! of input queries. This reduces LLM API costs and latency for similar queries.
//!
//! # Features
//!
//! - **Semantic Matching**: Find cached responses for semantically similar queries
//! - **Configurable Threshold**: Control cache hit sensitivity
//! - **TTL-based Expiration**: Automatic cache entry expiration
//! - **Cost Tracking**: Monitor savings from cache hits
//! - **Query Normalization**: Normalize queries for better hit rates
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::llm_cache::{LlmCache, LlmCacheConfig};
//!
//! let config = LlmCacheConfig::new(128)  // embedding dimensions
//!     .with_max_entries(10000)
//!     .with_similarity_threshold(0.95)
//!     .with_ttl(3600);
//!
//! let cache = LlmCache::new(config);
//!
//! // Store a response
//! cache.put("What is the capital of France?", &query_embedding, "Paris is the capital of France.");
//!
//! // Later, a similar query
//! if let Some(cached) = cache.get_semantic("What's France's capital city?", &similar_embedding) {
//!     println!("Cached response: {}", cached.response);
//! }
//! ```

use crate::distance::DistanceFunction;
use crate::error::Result;

use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Configuration for LLM semantic cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmCacheConfig {
    /// Embedding dimensions
    pub dimensions: usize,
    /// Maximum cache entries
    pub max_entries: usize,
    /// Similarity threshold (0.0 - 1.0)
    pub similarity_threshold: f32,
    /// Time-to-live in seconds
    pub ttl_seconds: u64,
    /// Distance function for similarity
    pub distance_function: DistanceFunction,
    /// Enable query normalization
    pub normalize_queries: bool,
    /// Track cost savings
    pub track_costs: bool,
    /// Estimated cost per LLM query (for savings calculation)
    pub cost_per_query: f64,
}

impl LlmCacheConfig {
    /// Create a new config with embedding dimensions
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            max_entries: 10000,
            similarity_threshold: 0.95,
            ttl_seconds: 3600,
            distance_function: DistanceFunction::Cosine,
            normalize_queries: true,
            track_costs: true,
            cost_per_query: 0.001, // $0.001 per query
        }
    }

    /// Set maximum entries
    #[must_use]
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set similarity threshold
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set TTL in seconds
    #[must_use]
    pub fn with_ttl(mut self, seconds: u64) -> Self {
        self.ttl_seconds = seconds;
        self
    }

    /// Set distance function
    #[must_use]
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_function = distance;
        self
    }

    /// Set cost per query
    #[must_use]
    pub fn with_cost_per_query(mut self, cost: f64) -> Self {
        self.cost_per_query = cost;
        self
    }
}

/// A cached LLM response
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Original query text
    query: String,
    /// Query embedding
    embedding: Vec<f32>,
    /// LLM response
    response: String,
    /// Optional metadata (model used, tokens, etc.)
    metadata: Option<Value>,
    /// Creation time
    created_at: Instant,
    /// Access count
    access_count: u64,
    /// Last access time
    last_accessed: Instant,
}

/// Result of a cache lookup
#[derive(Debug, Clone)]
pub struct CacheHit {
    /// Cached response
    pub response: String,
    /// Original cached query
    pub cached_query: String,
    /// Similarity score
    pub similarity: f32,
    /// Optional metadata
    pub metadata: Option<Value>,
    /// Age of cache entry in seconds
    pub age_seconds: u64,
}

/// Cache statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LlmCacheStats {
    /// Total queries
    pub total_queries: u64,
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Exact matches (query text match)
    pub exact_matches: u64,
    /// Semantic matches (embedding similarity)
    pub semantic_matches: u64,
    /// Current cache size
    pub size: usize,
    /// Evictions due to size limit
    pub evictions: u64,
    /// Expirations due to TTL
    pub expirations: u64,
    /// Estimated cost savings
    pub cost_savings: f64,
    /// Hit rate
    pub hit_rate: f32,
    /// Average similarity for semantic hits
    pub avg_semantic_similarity: f32,
}

/// Semantic cache for LLM responses
pub struct LlmCache {
    config: LlmCacheConfig,
    entries: RwLock<Vec<CacheEntry>>,
    /// Index for exact query matches
    query_index: RwLock<HashMap<String, usize>>,
    /// LRU order for eviction
    lru_order: RwLock<VecDeque<String>>,
    /// Statistics
    stats: LlmCacheStatsInternal,
}

struct LlmCacheStatsInternal {
    total_queries: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    exact_matches: AtomicU64,
    semantic_matches: AtomicU64,
    evictions: AtomicU64,
    expirations: AtomicU64,
    semantic_similarity_sum: RwLock<f64>,
}

impl Default for LlmCacheStatsInternal {
    fn default() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            exact_matches: AtomicU64::new(0),
            semantic_matches: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
            semantic_similarity_sum: RwLock::new(0.0),
        }
    }
}

impl LlmCache {
    /// Create a new LLM cache
    pub fn new(config: LlmCacheConfig) -> Self {
        Self {
            config,
            entries: RwLock::new(Vec::new()),
            query_index: RwLock::new(HashMap::new()),
            lru_order: RwLock::new(VecDeque::new()),
            stats: LlmCacheStatsInternal::default(),
        }
    }

    /// Normalize a query string for better matching
    fn normalize_query(&self, query: &str) -> String {
        if !self.config.normalize_queries {
            return query.to_string();
        }

        query
            .to_lowercase()
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Store a query-response pair
    pub fn put(
        &self,
        query: &str,
        embedding: &[f32],
        response: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let normalized = self.normalize_query(query);

        // Check for existing entry
        {
            let index = self.query_index.read();
            if index.contains_key(&normalized) {
                // Update existing entry
                let mut entries = self.entries.write();
                if let Some(&idx) = index.get(&normalized) {
                    if idx < entries.len() {
                        entries[idx].response = response.to_string();
                        entries[idx].embedding = embedding.to_vec();
                        entries[idx].metadata = metadata;
                        entries[idx].created_at = Instant::now();
                        return Ok(());
                    }
                }
            }
        }

        // Evict if necessary
        self.maybe_evict();

        // Add new entry
        let entry = CacheEntry {
            query: normalized.clone(),
            embedding: embedding.to_vec(),
            response: response.to_string(),
            metadata,
            created_at: Instant::now(),
            access_count: 0,
            last_accessed: Instant::now(),
        };

        let mut entries = self.entries.write();
        let idx = entries.len();
        entries.push(entry);

        self.query_index.write().insert(normalized.clone(), idx);
        self.lru_order.write().push_back(normalized);

        Ok(())
    }

    /// Get cached response by exact query match
    pub fn get_exact(&self, query: &str) -> Option<CacheHit> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        let normalized = self.normalize_query(query);
        let index = self.query_index.read();

        if let Some(&idx) = index.get(&normalized) {
            let mut entries = self.entries.write();
            if idx < entries.len() {
                let entry = &mut entries[idx];

                // Check TTL
                if entry.created_at.elapsed() > Duration::from_secs(self.config.ttl_seconds) {
                    drop(entries);
                    drop(index);
                    self.remove_entry(&normalized);
                    self.stats.expirations.fetch_add(1, Ordering::Relaxed);
                    self.stats.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }

                // Update access
                entry.access_count += 1;
                entry.last_accessed = Instant::now();

                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.exact_matches.fetch_add(1, Ordering::Relaxed);

                return Some(CacheHit {
                    response: entry.response.clone(),
                    cached_query: entry.query.clone(),
                    similarity: 1.0,
                    metadata: entry.metadata.clone(),
                    age_seconds: entry.created_at.elapsed().as_secs(),
                });
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Get cached response by semantic similarity
    pub fn get_semantic(&self, query: &str, query_embedding: &[f32]) -> Option<CacheHit> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        // First try exact match
        if let Some(hit) = self.get_exact_internal(query) {
            return Some(hit);
        }

        let entries = self.entries.read();
        let now = Instant::now();
        let ttl = Duration::from_secs(self.config.ttl_seconds);

        // Find best semantic match
        let mut best_match: Option<(usize, f32)> = None;

        for (idx, entry) in entries.iter().enumerate() {
            // Skip expired entries
            if entry.created_at.elapsed() > ttl {
                continue;
            }

            let similarity = self.compute_similarity(query_embedding, &entry.embedding);

            if similarity >= self.config.similarity_threshold {
                if best_match.is_none() || similarity > best_match.unwrap().1 {
                    best_match = Some((idx, similarity));
                }
            }
        }

        drop(entries);

        if let Some((idx, similarity)) = best_match {
            let mut entries = self.entries.write();
            if idx < entries.len() {
                let entry = &mut entries[idx];
                entry.access_count += 1;
                entry.last_accessed = now;

                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.semantic_matches.fetch_add(1, Ordering::Relaxed);
                *self.stats.semantic_similarity_sum.write() += similarity as f64;

                return Some(CacheHit {
                    response: entry.response.clone(),
                    cached_query: entry.query.clone(),
                    similarity,
                    metadata: entry.metadata.clone(),
                    age_seconds: entry.created_at.elapsed().as_secs(),
                });
            }
        }

        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Internal exact match without incrementing total queries
    fn get_exact_internal(&self, query: &str) -> Option<CacheHit> {
        let normalized = self.normalize_query(query);
        let index = self.query_index.read();

        if let Some(&idx) = index.get(&normalized) {
            let mut entries = self.entries.write();
            if idx < entries.len() {
                let entry = &mut entries[idx];

                if entry.created_at.elapsed() > Duration::from_secs(self.config.ttl_seconds) {
                    return None;
                }

                entry.access_count += 1;
                entry.last_accessed = Instant::now();

                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                self.stats.exact_matches.fetch_add(1, Ordering::Relaxed);

                return Some(CacheHit {
                    response: entry.response.clone(),
                    cached_query: entry.query.clone(),
                    similarity: 1.0,
                    metadata: entry.metadata.clone(),
                    age_seconds: entry.created_at.elapsed().as_secs(),
                });
            }
        }

        None
    }

    /// Compute similarity between two embeddings
    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let distance = self.config.distance_function.compute(a, b);

        // Convert distance to similarity (0-1)
        match self.config.distance_function {
            DistanceFunction::Cosine | DistanceFunction::CosineNormalized => 1.0 - distance,
            DistanceFunction::Euclidean => 1.0 / (1.0 + distance),
            DistanceFunction::DotProduct => (1.0 + distance) / 2.0,
            DistanceFunction::Manhattan => 1.0 / (1.0 + distance),
        }
    }

    /// Evict entries if over capacity
    fn maybe_evict(&self) {
        let entries_len = self.entries.read().len();
        if entries_len < self.config.max_entries {
            return;
        }

        // Evict LRU entry
        let to_remove = self.lru_order.write().pop_front();
        if let Some(query) = to_remove {
            self.remove_entry(&query);
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Remove an entry by query
    fn remove_entry(&self, query: &str) {
        let mut index = self.query_index.write();
        if let Some(idx) = index.remove(query) {
            let mut entries = self.entries.write();
            if idx < entries.len() {
                entries.remove(idx);
                // Reindex
                for (new_idx, entry) in entries.iter().enumerate() {
                    index.insert(entry.query.clone(), new_idx);
                }
            }
        }
        self.lru_order.write().retain(|q| q != query);
    }

    /// Clear expired entries
    pub fn clear_expired(&self) -> usize {
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        let mut expired = Vec::new();

        {
            let entries = self.entries.read();
            for entry in entries.iter() {
                if entry.created_at.elapsed() > ttl {
                    expired.push(entry.query.clone());
                }
            }
        }

        let count = expired.len();
        for query in expired {
            self.remove_entry(&query);
        }

        self.stats.expirations.fetch_add(count as u64, Ordering::Relaxed);
        count
    }

    /// Get cache statistics
    pub fn stats(&self) -> LlmCacheStats {
        let total = self.stats.total_queries.load(Ordering::Relaxed);
        let hits = self.stats.hits.load(Ordering::Relaxed);
        let semantic = self.stats.semantic_matches.load(Ordering::Relaxed);

        let hit_rate = if total > 0 {
            hits as f32 / total as f32
        } else {
            0.0
        };

        let avg_similarity = if semantic > 0 {
            (*self.stats.semantic_similarity_sum.read() / semantic as f64) as f32
        } else {
            0.0
        };

        let cost_savings = if self.config.track_costs {
            hits as f64 * self.config.cost_per_query
        } else {
            0.0
        };

        LlmCacheStats {
            total_queries: total,
            hits,
            misses: self.stats.misses.load(Ordering::Relaxed),
            exact_matches: self.stats.exact_matches.load(Ordering::Relaxed),
            semantic_matches: semantic,
            size: self.entries.read().len(),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            expirations: self.stats.expirations.load(Ordering::Relaxed),
            cost_savings,
            hit_rate,
            avg_semantic_similarity: avg_similarity,
        }
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.entries.write().clear();
        self.query_index.write().clear();
        self.lru_order.write().clear();
    }

    /// Get all cached queries (for debugging)
    pub fn cached_queries(&self) -> Vec<String> {
        self.entries.read().iter().map(|e| e.query.clone()).collect()
    }

    /// Get similar queries above threshold
    pub fn find_similar(&self, query_embedding: &[f32], limit: usize) -> Vec<(String, f32)> {
        let entries = self.entries.read();
        let ttl = Duration::from_secs(self.config.ttl_seconds);

        let mut results: Vec<(String, f32)> = entries
            .iter()
            .filter(|e| e.created_at.elapsed() <= ttl)
            .map(|e| {
                let similarity = self.compute_similarity(query_embedding, &e.embedding);
                (e.query.clone(), similarity)
            })
            .filter(|(_, sim)| *sim >= self.config.similarity_threshold)
            .collect();

        results.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        results.truncate(limit);
        results
    }
}

/// Builder for LLM cache queries
pub struct LlmCacheQueryBuilder<'a> {
    cache: &'a LlmCache,
    query: String,
    embedding: Option<Vec<f32>>,
    allow_semantic: bool,
    min_similarity: Option<f32>,
}

impl<'a> LlmCacheQueryBuilder<'a> {
    /// Create a new query builder
    pub fn new(cache: &'a LlmCache, query: &str) -> Self {
        Self {
            cache,
            query: query.to_string(),
            embedding: None,
            allow_semantic: true,
            min_similarity: None,
        }
    }

    /// Set query embedding for semantic matching
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Disable semantic matching (exact only)
    pub fn exact_only(mut self) -> Self {
        self.allow_semantic = false;
        self
    }

    /// Set minimum similarity threshold
    pub fn min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = Some(threshold);
        self
    }

    /// Execute the cache lookup
    pub fn get(self) -> Option<CacheHit> {
        if self.allow_semantic {
            if let Some(embedding) = self.embedding {
                return self.cache.get_semantic(&self.query, &embedding);
            }
        }
        self.cache.get_exact(&self.query)
    }
}

/// Wrapper for caching LLM function calls
pub struct CachedLlm<F> {
    cache: LlmCache,
    llm_fn: F,
    embed_fn: Box<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
}

impl<F> CachedLlm<F>
where
    F: Fn(&str) -> Result<String> + Send + Sync,
{
    /// Create a new cached LLM wrapper
    pub fn new<E>(config: LlmCacheConfig, llm_fn: F, embed_fn: E) -> Self
    where
        E: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
    {
        Self {
            cache: LlmCache::new(config),
            llm_fn,
            embed_fn: Box::new(embed_fn),
        }
    }

    /// Query with caching
    pub fn query(&self, prompt: &str) -> Result<String> {
        let embedding = (self.embed_fn)(prompt);

        // Check cache
        if let Some(hit) = self.cache.get_semantic(prompt, &embedding) {
            return Ok(hit.response);
        }

        // Call LLM
        let response = (self.llm_fn)(prompt)?;

        // Cache result
        self.cache.put(prompt, &embedding, &response, None)?;

        Ok(response)
    }

    /// Get cache statistics
    pub fn stats(&self) -> LlmCacheStats {
        self.cache.stats()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
    }
}

// =============================================================================
// Enhanced Semantic Caching Features (Next-Gen)
// =============================================================================

/// Cache warming strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheWarmingConfig {
    /// Queries to pre-populate the cache with
    pub seed_queries: Vec<WarmingQuery>,
    /// Enable background refresh of popular queries
    pub enable_refresh: bool,
    /// Refresh interval in seconds
    pub refresh_interval_seconds: u64,
    /// Minimum access count to be eligible for refresh
    pub min_access_count: u64,
}

impl Default for CacheWarmingConfig {
    fn default() -> Self {
        Self {
            seed_queries: Vec::new(),
            enable_refresh: false,
            refresh_interval_seconds: 3600,
            min_access_count: 5,
        }
    }
}

/// A query to warm the cache with
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingQuery {
    pub query: String,
    pub embedding: Vec<f32>,
    pub response: String,
    pub metadata: Option<Value>,
}

/// Adaptive threshold configuration
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdConfig {
    /// Initial threshold
    pub initial_threshold: f32,
    /// Minimum threshold (floor)
    pub min_threshold: f32,
    /// Maximum threshold (ceiling)
    pub max_threshold: f32,
    /// Target hit rate (0.0 - 1.0)
    pub target_hit_rate: f32,
    /// Adjustment step size
    pub adjustment_step: f32,
    /// Window size for hit rate calculation
    pub window_size: usize,
}

impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            initial_threshold: 0.95,
            min_threshold: 0.80,
            max_threshold: 0.99,
            target_hit_rate: 0.30,
            adjustment_step: 0.01,
            window_size: 100,
        }
    }
}

/// Enhanced LLM cache with adaptive thresholds and warming
pub struct EnhancedLlmCache {
    inner: LlmCache,
    adaptive_config: Option<AdaptiveThresholdConfig>,
    current_threshold: RwLock<f32>,
    recent_results: RwLock<VecDeque<bool>>, // true = hit, false = miss
    popular_queries: RwLock<Vec<PopularQuery>>,
}

#[derive(Debug, Clone)]
struct PopularQuery {
    query: String,
    embedding: Vec<f32>,
    access_count: u64,
    last_refreshed: Instant,
}

impl EnhancedLlmCache {
    /// Create an enhanced cache with adaptive thresholds
    pub fn new(config: LlmCacheConfig) -> Self {
        let threshold = config.similarity_threshold;
        Self {
            inner: LlmCache::new(config),
            adaptive_config: None,
            current_threshold: RwLock::new(threshold),
            recent_results: RwLock::new(VecDeque::with_capacity(100)),
            popular_queries: RwLock::new(Vec::new()),
        }
    }

    /// Enable adaptive threshold tuning
    #[must_use]
    pub fn with_adaptive_threshold(mut self, config: AdaptiveThresholdConfig) -> Self {
        *self.current_threshold.write() = config.initial_threshold;
        self.adaptive_config = Some(config);
        self
    }

    /// Warm the cache with seed queries
    pub fn warm(&self, warming_config: &CacheWarmingConfig) -> Result<usize> {
        let mut warmed = 0;
        for query in &warming_config.seed_queries {
            self.inner.put(
                &query.query,
                &query.embedding,
                &query.response,
                query.metadata.clone(),
            )?;
            warmed += 1;
        }
        Ok(warmed)
    }

    /// Store a query-response pair
    pub fn put(
        &self,
        query: &str,
        embedding: &[f32],
        response: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.inner.put(query, embedding, response, metadata)
    }

    /// Get cached response with adaptive threshold
    pub fn get(&self, query: &str, embedding: &[f32]) -> Option<CacheHit> {
        let threshold = *self.current_threshold.read();

        // Try exact match first
        if let Some(hit) = self.inner.get_exact(query) {
            self.record_result(true);
            self.update_popularity(query, embedding);
            return Some(hit);
        }

        // Try semantic match with current threshold
        let entries = self.inner.entries.read();
        let ttl = Duration::from_secs(self.inner.config.ttl_seconds);

        let mut best_match: Option<(usize, f32)> = None;
        for (idx, entry) in entries.iter().enumerate() {
            if entry.created_at.elapsed() > ttl {
                continue;
            }

            let similarity = self.inner.compute_similarity(embedding, &entry.embedding);
            if similarity >= threshold {
                if best_match.is_none() || similarity > best_match.unwrap().1 {
                    best_match = Some((idx, similarity));
                }
            }
        }

        drop(entries);

        if let Some((idx, similarity)) = best_match {
            self.record_result(true);
            self.update_popularity(query, embedding);

            let entries = self.inner.entries.read();
            let entry = &entries[idx];
            return Some(CacheHit {
                response: entry.response.clone(),
                cached_query: entry.query.clone(),
                similarity,
                metadata: entry.metadata.clone(),
                age_seconds: entry.created_at.elapsed().as_secs(),
            });
        }

        self.record_result(false);
        None
    }

    fn record_result(&self, hit: bool) {
        let mut recent = self.recent_results.write();
        let window_size = self
            .adaptive_config
            .as_ref()
            .map(|c| c.window_size)
            .unwrap_or(100);

        recent.push_back(hit);
        while recent.len() > window_size {
            recent.pop_front();
        }

        // Adjust threshold if adaptive mode is enabled
        if let Some(ref config) = self.adaptive_config {
            if recent.len() >= window_size / 2 {
                let hit_rate = recent.iter().filter(|&&h| h).count() as f32 / recent.len() as f32;
                let mut threshold = self.current_threshold.write();

                if hit_rate < config.target_hit_rate - 0.05 {
                    // Hit rate too low, lower threshold
                    *threshold = (*threshold - config.adjustment_step).max(config.min_threshold);
                } else if hit_rate > config.target_hit_rate + 0.10 {
                    // Hit rate comfortably high, raise threshold for quality
                    *threshold = (*threshold + config.adjustment_step).min(config.max_threshold);
                }
            }
        }
    }

    fn update_popularity(&self, query: &str, embedding: &[f32]) {
        let mut popular = self.popular_queries.write();

        if let Some(pq) = popular.iter_mut().find(|p| p.query == query) {
            pq.access_count += 1;
        } else if popular.len() < 1000 {
            popular.push(PopularQuery {
                query: query.to_string(),
                embedding: embedding.to_vec(),
                access_count: 1,
                last_refreshed: Instant::now(),
            });
        }
    }

    /// Get current adaptive threshold
    pub fn current_threshold(&self) -> f32 {
        *self.current_threshold.read()
    }

    /// Get current hit rate
    pub fn current_hit_rate(&self) -> f32 {
        let recent = self.recent_results.read();
        if recent.is_empty() {
            return 0.0;
        }
        recent.iter().filter(|&&h| h).count() as f32 / recent.len() as f32
    }

    /// Get most popular queries for cache warming
    pub fn get_popular_queries(&self, limit: usize) -> Vec<(String, u64)> {
        let mut popular: Vec<_> = self
            .popular_queries
            .read()
            .iter()
            .map(|p| (p.query.clone(), p.access_count))
            .collect();

        popular.sort_by(|a, b| b.1.cmp(&a.1));
        popular.truncate(limit);
        popular
    }

    /// Get statistics
    pub fn stats(&self) -> LlmCacheStats {
        self.inner.stats()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.inner.clear();
        self.recent_results.write().clear();
    }
}

/// OpenAI-compatible chat completion request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// OpenAI-compatible chat completion response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: ChatUsage,
    /// Extension: whether this came from cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached: Option<bool>,
    /// Extension: similarity score if cached
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_similarity: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI API Proxy with semantic caching
///
/// This proxy intercepts chat completion requests and serves cached responses
/// when semantically similar queries are found.
pub struct OpenAIProxy {
    cache: EnhancedLlmCache,
    embed_fn: Box<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
    llm_fn: Box<dyn Fn(&ChatCompletionRequest) -> Result<ChatCompletionResponse> + Send + Sync>,
}

impl OpenAIProxy {
    /// Create a new OpenAI proxy with caching
    pub fn new<E, L>(config: LlmCacheConfig, embed_fn: E, llm_fn: L) -> Self
    where
        E: Fn(&str) -> Vec<f32> + Send + Sync + 'static,
        L: Fn(&ChatCompletionRequest) -> Result<ChatCompletionResponse> + Send + Sync + 'static,
    {
        Self {
            cache: EnhancedLlmCache::new(config),
            embed_fn: Box::new(embed_fn),
            llm_fn: Box::new(llm_fn),
        }
    }

    /// Enable adaptive thresholds
    pub fn with_adaptive_threshold(mut self, config: AdaptiveThresholdConfig) -> Self {
        self.cache = self.cache.with_adaptive_threshold(config);
        self
    }

    /// Process a chat completion request
    pub fn chat_completion(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        // Don't cache streaming requests
        if request.stream.unwrap_or(false) {
            return (self.llm_fn)(request);
        }

        // Create cache key from messages
        let cache_key = self.create_cache_key(&request.messages);
        let embedding = (self.embed_fn)(&cache_key);

        // Check cache
        if let Some(hit) = self.cache.get(&cache_key, &embedding) {
            // Parse cached response
            if let Ok(mut response) = serde_json::from_str::<ChatCompletionResponse>(&hit.response)
            {
                response.cached = Some(true);
                response.cache_similarity = Some(hit.similarity);
                return Ok(response);
            }
        }

        // Call actual LLM
        let response = (self.llm_fn)(request)?;

        // Cache the response
        let response_json = serde_json::to_string(&response)?;
        self.cache.put(&cache_key, &embedding, &response_json, None)?;

        Ok(response)
    }

    fn create_cache_key(&self, messages: &[ChatMessage]) -> String {
        // Use the last user message as the primary cache key
        // Include system message for context if present
        let mut key_parts = Vec::new();

        for msg in messages {
            if msg.role == "system" || msg.role == "user" {
                key_parts.push(format!("[{}] {}", msg.role, msg.content));
            }
        }

        key_parts.join(" | ")
    }

    /// Get cache statistics
    pub fn stats(&self) -> LlmCacheStats {
        self.cache.stats()
    }

    /// Get current adaptive threshold
    pub fn current_threshold(&self) -> f32 {
        self.cache.current_threshold()
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.clear();
    }
}

/// Multi-tier cache for different latency/accuracy tradeoffs
pub struct MultiTierCache {
    /// L1: In-memory exact match (fastest)
    l1_exact: RwLock<HashMap<String, CacheEntry>>,
    /// L2: Semantic cache (fast)
    l2_semantic: LlmCache,
    /// Stats
    l1_hits: AtomicU64,
    l2_hits: AtomicU64,
    misses: AtomicU64,
}

impl MultiTierCache {
    /// Create a new multi-tier cache
    pub fn new(config: LlmCacheConfig) -> Self {
        Self {
            l1_exact: RwLock::new(HashMap::new()),
            l2_semantic: LlmCache::new(config),
            l1_hits: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Store in both tiers
    pub fn put(
        &self,
        query: &str,
        embedding: &[f32],
        response: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        // L1: exact key
        let normalized = query.to_lowercase().trim().to_string();
        self.l1_exact.write().insert(
            normalized.clone(),
            CacheEntry {
                query: normalized,
                embedding: embedding.to_vec(),
                response: response.to_string(),
                metadata: metadata.clone(),
                created_at: Instant::now(),
                access_count: 0,
                last_accessed: Instant::now(),
            },
        );

        // L2: semantic
        self.l2_semantic.put(query, embedding, response, metadata)
    }

    /// Get from cache, checking L1 first, then L2
    pub fn get(&self, query: &str, embedding: &[f32]) -> Option<CacheHit> {
        let normalized = query.to_lowercase().trim().to_string();

        // L1: exact match
        if let Some(entry) = self.l1_exact.read().get(&normalized) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(CacheHit {
                response: entry.response.clone(),
                cached_query: entry.query.clone(),
                similarity: 1.0,
                metadata: entry.metadata.clone(),
                age_seconds: entry.created_at.elapsed().as_secs(),
            });
        }

        // L2: semantic match
        if let Some(hit) = self.l2_semantic.get_semantic(query, embedding) {
            self.l2_hits.fetch_add(1, Ordering::Relaxed);
            return Some(hit);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Get tier-specific statistics
    pub fn tier_stats(&self) -> MultiTierStats {
        let l1 = self.l1_hits.load(Ordering::Relaxed);
        let l2 = self.l2_hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = l1 + l2 + misses;

        MultiTierStats {
            l1_hits: l1,
            l2_hits: l2,
            misses,
            l1_hit_rate: if total > 0 { l1 as f64 / total as f64 } else { 0.0 },
            l2_hit_rate: if total > 0 { l2 as f64 / total as f64 } else { 0.0 },
            total_hit_rate: if total > 0 { (l1 + l2) as f64 / total as f64 } else { 0.0 },
        }
    }

    /// Clear all tiers
    pub fn clear(&self) {
        self.l1_exact.write().clear();
        self.l2_semantic.clear();
    }
}

/// Statistics for multi-tier cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTierStats {
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub misses: u64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub total_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * seed).sin()).collect()
    }

    #[test]
    fn test_cache_config() {
        let config = LlmCacheConfig::new(128)
            .with_max_entries(5000)
            .with_similarity_threshold(0.9)
            .with_ttl(7200);

        assert_eq!(config.dimensions, 128);
        assert_eq!(config.max_entries, 5000);
        assert!((config.similarity_threshold - 0.9).abs() < 0.01);
        assert_eq!(config.ttl_seconds, 7200);
    }

    #[test]
    fn test_exact_match() {
        let config = LlmCacheConfig::new(64);
        let cache = LlmCache::new(config);

        let embedding = test_embedding(64, 1.0);
        cache
            .put("What is AI?", &embedding, "AI is artificial intelligence.", None)
            .unwrap();

        let hit = cache.get_exact("What is AI?").unwrap();
        assert_eq!(hit.response, "AI is artificial intelligence.");
        assert!((hit.similarity - 1.0).abs() < 0.01);

        assert!(cache.get_exact("Different query").is_none());
    }

    #[test]
    fn test_semantic_match() {
        let config = LlmCacheConfig::new(64).with_similarity_threshold(0.8);
        let cache = LlmCache::new(config);

        let embedding1 = test_embedding(64, 1.0);
        cache
            .put("What is machine learning?", &embedding1, "ML is a subset of AI.", None)
            .unwrap();

        // Similar embedding should hit
        let embedding2 = test_embedding(64, 1.01); // Slightly different
        let hit = cache.get_semantic("What is ML?", &embedding2).unwrap();
        assert_eq!(hit.response, "ML is a subset of AI.");
        assert!(hit.similarity >= 0.8);

        // Very different embedding should miss
        let embedding3 = test_embedding(64, 10.0);
        assert!(cache.get_semantic("Different topic", &embedding3).is_none());
    }

    #[test]
    fn test_normalization() {
        let config = LlmCacheConfig::new(64);
        let cache = LlmCache::new(config);

        let embedding = test_embedding(64, 1.0);
        cache.put("  What is AI?  ", &embedding, "Response", None).unwrap();

        // Should match with different spacing/case
        assert!(cache.get_exact("what is ai?").is_some());
        assert!(cache.get_exact("WHAT   IS   AI?").is_some());
    }

    #[test]
    fn test_eviction() {
        let config = LlmCacheConfig::new(64).with_max_entries(3);
        let cache = LlmCache::new(config);

        for i in 0..5 {
            let embedding = test_embedding(64, i as f32);
            cache.put(&format!("query{}", i), &embedding, "response", None).unwrap();
        }

        assert_eq!(cache.len(), 3);

        let stats = cache.stats();
        assert_eq!(stats.evictions, 2);
    }

    #[test]
    fn test_statistics() {
        let config = LlmCacheConfig::new(64).with_cost_per_query(0.01);
        let cache = LlmCache::new(config);

        let embedding = test_embedding(64, 1.0);
        cache.put("query1", &embedding, "response1", None).unwrap();

        // Hit
        cache.get_exact("query1");
        cache.get_exact("query1");

        // Miss
        cache.get_exact("query2");

        let stats = cache.stats();
        assert_eq!(stats.total_queries, 3);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
        assert!((stats.cost_savings - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_find_similar() {
        let config = LlmCacheConfig::new(64).with_similarity_threshold(0.0); // Accept any similarity
        let cache = LlmCache::new(config);

        for i in 0..5 {
            let embedding = test_embedding(64, i as f32 + 0.1); // Avoid zero embedding
            cache.put(&format!("query{}", i), &embedding, "response", None).unwrap();
        }

        let search_embedding = test_embedding(64, 2.5);
        let similar = cache.find_similar(&search_embedding, 3);

        assert!(!similar.is_empty());
        // Results should be sorted by similarity
        for i in 1..similar.len() {
            assert!(similar[i].1 <= similar[i - 1].1);
        }
    }

    #[test]
    fn test_query_builder() {
        let config = LlmCacheConfig::new(64).with_similarity_threshold(0.5); // Lower threshold
        let cache = LlmCache::new(config);

        let embedding = test_embedding(64, 1.0);
        cache.put("test query", &embedding, "test response", None).unwrap();

        // Exact only
        let hit = LlmCacheQueryBuilder::new(&cache, "test query")
            .exact_only()
            .get()
            .unwrap();
        assert_eq!(hit.response, "test response");

        // With very similar embedding (same seed should give high similarity)
        let hit2 = LlmCacheQueryBuilder::new(&cache, "similar")
            .with_embedding(test_embedding(64, 1.0)) // Same embedding
            .get();
        assert!(hit2.is_some());
    }

    #[test]
    fn test_clear() {
        let config = LlmCacheConfig::new(64);
        let cache = LlmCache::new(config);

        let embedding = test_embedding(64, 1.0);
        cache.put("query", &embedding, "response", None).unwrap();
        assert_eq!(cache.len(), 1);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cached_llm() {
        let config = LlmCacheConfig::new(64);
        let call_count = std::sync::atomic::AtomicU64::new(0);

        let cached = CachedLlm::new(
            config,
            |_prompt: &str| {
                call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok("LLM response".to_string())
            },
            |_text: &str| test_embedding(64, 1.0),
        );

        // First call - cache miss
        let response1 = cached.query("test prompt").unwrap();
        assert_eq!(response1, "LLM response");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1);

        // Second call - cache hit
        let response2 = cached.query("test prompt").unwrap();
        assert_eq!(response2, "LLM response");
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 1); // No additional call

        let stats = cached.stats();
        assert_eq!(stats.hits, 1);
    }
}
