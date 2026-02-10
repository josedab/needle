//! Semantic Caching Layer
//!
//! A multi-tier LLM response cache that uses vector similarity to match
//! semantically equivalent prompts. Provides L1 (in-memory) and L2 (larger)
//! cache tiers with automatic promotion and eviction.
//!
//! # Overview
//!
//! Instead of requiring exact string matches, the semantic cache uses HNSW-based
//! nearest neighbor search to find cached responses for prompts that are
//! semantically similar to previous queries.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::semantic_cache::{SemanticCache, SemanticCacheConfig};
//!
//! let config = SemanticCacheConfig::default();
//! let mut cache = SemanticCache::new(config);
//!
//! // Store a response
//! let embedding = vec![0.1; 384];
//! cache.store("What is Rust?", "Rust is a systems programming language.", embedding.clone(), None).unwrap();
//!
//! // Look up by similar embedding
//! if let Some(entry) = cache.lookup(&embedding) {
//!     println!("Cache hit: {}", entry.response);
//! }
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Cache tier indicating where an entry is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTier {
    /// In-memory tier with HNSW index for fast similarity lookup
    L1Memory,
    /// Larger capacity tier for less frequently accessed entries
    L2Disk,
    /// Remote/distributed tier (reserved for future use)
    L3Remote,
}

/// Eviction strategy for cache tiers when capacity is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Evict expired entries first, then fall back to LRU
    TTL,
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        EvictionStrategy::LRU
    }
}

/// A single cached entry containing the prompt key, response, and embedding.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Original prompt or cache key
    pub key: String,
    /// Cached LLM response
    pub response: String,
    /// Vector embedding of the prompt
    pub embedding: Vec<f32>,
    /// When this entry was created
    pub created_at: Instant,
    /// When this entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry has been accessed
    pub access_count: u64,
    /// Time-to-live for this entry
    pub ttl: Option<Duration>,
    /// Which cache tier this entry resides in
    pub tier: CacheTier,
    /// Optional metadata associated with this entry
    pub metadata: Option<serde_json::Value>,
}

impl CacheEntry {
    /// Check whether this entry has expired based on its TTL
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() >= ttl
        } else {
            false
        }
    }
}

/// Configuration for the semantic cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Minimum cosine similarity to consider a cache hit (0.0–1.0).
    /// For cosine distance, a distance of `1.0 - similarity_threshold` is used.
    pub similarity_threshold: f32,
    /// Maximum number of entries in the L1 (memory) tier
    pub l1_max_entries: usize,
    /// Maximum number of entries in the L2 tier
    pub l2_max_entries: usize,
    /// Default TTL applied to new entries when no explicit TTL is given
    #[serde(
        serialize_with = "serialize_opt_duration",
        deserialize_with = "deserialize_opt_duration"
    )]
    pub default_ttl: Option<Duration>,
    /// Embedding dimensionality
    pub dimensions: usize,
    /// Access count threshold: entries exceeding this in L2 are promoted to L1
    pub promotion_threshold: u64,
    /// Strategy used when a tier is full and a new entry must be inserted
    pub eviction_strategy: EvictionStrategy,
}

fn serialize_opt_duration<S>(dur: &Option<Duration>, s: S) -> std::result::Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match dur {
        Some(d) => s.serialize_some(&d.as_secs()),
        None => s.serialize_none(),
    }
}

fn deserialize_opt_duration<'de, D>(d: D) -> std::result::Result<Option<Duration>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<u64> = Option::deserialize(d)?;
    Ok(opt.map(Duration::from_secs))
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            l1_max_entries: 1000,
            l2_max_entries: 10_000,
            default_ttl: None,
            dimensions: 384,
            promotion_threshold: 5,
            eviction_strategy: EvictionStrategy::LRU,
        }
    }
}

/// Aggregated cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Current number of L1 entries
    pub l1_size: usize,
    /// Current number of L2 entries
    pub l2_size: usize,
    /// Hit ratio (hits / (hits + misses)), 0.0 if no lookups
    pub hit_ratio: f64,
    /// Average lookup latency in microseconds
    pub avg_lookup_us: f64,
    /// Total evictions performed
    pub evictions: u64,
    /// Total promotions from L2 to L1
    pub promotions: u64,
}

/// Internal mapping from HNSW vector id to cache key.
struct VectorIdMap {
    id_to_key: Vec<String>,
}

impl VectorIdMap {
    fn new() -> Self {
        Self {
            id_to_key: Vec::new(),
        }
    }

    fn insert(&mut self, key: String) -> usize {
        let id = self.id_to_key.len();
        self.id_to_key.push(key);
        id
    }

    fn get_key(&self, id: usize) -> Option<&str> {
        self.id_to_key.get(id).map(|s| s.as_str())
    }
}

/// Multi-tier semantic cache with HNSW-backed similarity lookup.
pub struct SemanticCache {
    config: SemanticCacheConfig,
    /// L1 in-memory entries keyed by prompt key
    l1_entries: HashMap<String, CacheEntry>,
    /// L2 larger-capacity entries keyed by prompt key
    l2_entries: HashMap<String, CacheEntry>,
    /// HNSW index over L1 embeddings for fast nearest-neighbor lookup
    hnsw_index: HnswIndex,
    /// All vectors stored in insertion order (parallel to HNSW vector IDs)
    vectors: Vec<Vec<f32>>,
    /// Maps HNSW vector IDs back to cache keys
    id_map: VectorIdMap,
    /// Reverse map from cache key to HNSW vector ID
    key_to_vid: HashMap<String, usize>,
    // Statistics
    hits: u64,
    misses: u64,
    total_lookup_us: u64,
    lookup_count: u64,
    evictions: u64,
    promotions: u64,
}

impl SemanticCache {
    /// Create a new semantic cache with the given configuration.
    pub fn new(config: SemanticCacheConfig) -> Self {
        let hnsw_config = HnswConfig::builder()
            .m(16)
            .ef_construction(100)
            .ef_search(50);
        let hnsw_index = HnswIndex::new(hnsw_config, DistanceFunction::Cosine);

        Self {
            config,
            l1_entries: HashMap::new(),
            l2_entries: HashMap::new(),
            hnsw_index,
            vectors: Vec::new(),
            id_map: VectorIdMap::new(),
            key_to_vid: HashMap::new(),
            hits: 0,
            misses: 0,
            total_lookup_us: 0,
            lookup_count: 0,
            evictions: 0,
            promotions: 0,
        }
    }

    /// Look up the cache for a semantically similar entry.
    ///
    /// Uses the HNSW index to find the nearest cached embedding, checks whether
    /// the similarity meets the configured threshold, and returns the entry if so.
    pub fn lookup(&mut self, query_embedding: &[f32]) -> Option<CacheEntry> {
        let start = Instant::now();

        if self.hnsw_index.is_empty() {
            self.misses += 1;
            self.record_lookup(start);
            return None;
        }

        let results = self.hnsw_index.search(query_embedding, 1, &self.vectors);

        if let Some(&(vid, distance)) = results.first() {
            // Cosine distance: 0 = identical. similarity = 1 - distance.
            let similarity = 1.0 - distance;
            if similarity >= self.config.similarity_threshold {
                if let Some(key) = self.id_map.get_key(vid) {
                    let key = key.to_string();
                    // Check L1 first, then L2
                    if let Some(entry) = self.l1_entries.get_mut(&key) {
                        if entry.is_expired() {
                            self.misses += 1;
                            self.record_lookup(start);
                            return None;
                        }
                        entry.last_accessed = Instant::now();
                        entry.access_count += 1;
                        self.hits += 1;
                        let result = entry.clone();
                        self.record_lookup(start);
                        return Some(result);
                    }
                    if let Some(entry) = self.l2_entries.get_mut(&key) {
                        if entry.is_expired() {
                            self.misses += 1;
                            self.record_lookup(start);
                            return None;
                        }
                        entry.last_accessed = Instant::now();
                        entry.access_count += 1;
                        self.hits += 1;
                        let result = entry.clone();
                        // Auto-promote if access count exceeds threshold
                        if result.access_count >= self.config.promotion_threshold {
                            self.promote_internal(&key);
                        }
                        self.record_lookup(start);
                        return Some(result);
                    }
                }
            }
        }

        self.misses += 1;
        self.record_lookup(start);
        None
    }

    /// Store a new entry in the cache.
    ///
    /// The entry is placed in L1 if there is capacity, otherwise in L2.
    /// When a tier is full, the eviction strategy is applied.
    pub fn store(
        &mut self,
        key: &str,
        response: &str,
        embedding: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: embedding.len(),
            });
        }

        let now = Instant::now();
        let entry = CacheEntry {
            key: key.to_string(),
            response: response.to_string(),
            embedding: embedding.clone(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl: self.config.default_ttl,
            tier: CacheTier::L1Memory,
            metadata,
        };

        // Insert into HNSW index
        let vid = self.id_map.insert(key.to_string());
        self.vectors.push(embedding);
        self.hnsw_index
            .insert(vid, &self.vectors[vid], &self.vectors)?;
        self.key_to_vid.insert(key.to_string(), vid);

        // Place in L1 if room, otherwise L2
        if self.l1_entries.len() < self.config.l1_max_entries {
            self.l1_entries.insert(key.to_string(), entry);
        } else if self.l2_entries.len() < self.config.l2_max_entries {
            let mut entry = entry;
            entry.tier = CacheTier::L2Disk;
            self.l2_entries.insert(key.to_string(), entry);
        } else {
            // Evict from L2 to make room
            self.evict_one(CacheTier::L2Disk);
            let mut entry = entry;
            entry.tier = CacheTier::L2Disk;
            self.l2_entries.insert(key.to_string(), entry);
        }

        Ok(())
    }

    /// Invalidate (remove) a specific cache entry by key.
    pub fn invalidate(&mut self, key: &str) {
        self.l1_entries.remove(key);
        self.l2_entries.remove(key);
    }

    /// Invalidate all entries whose embeddings are within `threshold` similarity
    /// of the given embedding.
    pub fn invalidate_similar(&mut self, embedding: &[f32], threshold: f32) {
        if self.hnsw_index.is_empty() {
            return;
        }

        // Search for many candidates
        let k = self.l1_entries.len() + self.l2_entries.len();
        if k == 0 {
            return;
        }
        let results = self.hnsw_index.search(embedding, k, &self.vectors);

        let keys_to_remove: Vec<String> = results
            .iter()
            .filter(|(_, distance)| {
                let similarity = 1.0 - distance;
                similarity >= threshold
            })
            .filter_map(|(vid, _)| self.id_map.get_key(*vid).map(|k| k.to_string()))
            .collect();

        for key in keys_to_remove {
            self.l1_entries.remove(&key);
            self.l2_entries.remove(&key);
        }
    }

    /// Return current cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total = self.hits + self.misses;
        let hit_ratio = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        let avg_lookup_us = if self.lookup_count > 0 {
            self.total_lookup_us as f64 / self.lookup_count as f64
        } else {
            0.0
        };

        CacheStats {
            hits: self.hits,
            misses: self.misses,
            l1_size: self.l1_entries.len(),
            l2_size: self.l2_entries.len(),
            hit_ratio,
            avg_lookup_us,
            evictions: self.evictions,
            promotions: self.promotions,
        }
    }

    /// Clear all entries from both tiers and reset the HNSW index.
    pub fn clear(&mut self) {
        self.l1_entries.clear();
        self.l2_entries.clear();
        let hnsw_config = HnswConfig::builder()
            .m(16)
            .ef_construction(100)
            .ef_search(50);
        self.hnsw_index = HnswIndex::new(hnsw_config, DistanceFunction::Cosine);
        self.vectors.clear();
        self.id_map = VectorIdMap::new();
        self.key_to_vid.clear();
    }

    /// Explicitly promote an entry from L2 to L1.
    pub fn promote(&mut self, key: &str) {
        self.promote_internal(key);
    }

    /// Evict all expired entries from both tiers.
    pub fn evict_expired(&mut self) {
        let expired_l1: Vec<String> = self
            .l1_entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        for key in &expired_l1 {
            self.l1_entries.remove(key);
            self.evictions += 1;
        }

        let expired_l2: Vec<String> = self
            .l2_entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        for key in &expired_l2 {
            self.l2_entries.remove(key);
            self.evictions += 1;
        }
    }

    // --- Internal helpers ---

    fn record_lookup(&mut self, start: Instant) {
        self.total_lookup_us += start.elapsed().as_micros() as u64;
        self.lookup_count += 1;
    }

    fn promote_internal(&mut self, key: &str) {
        if let Some(mut entry) = self.l2_entries.remove(key) {
            // Evict from L1 if full
            if self.l1_entries.len() >= self.config.l1_max_entries {
                self.evict_one(CacheTier::L1Memory);
            }
            entry.tier = CacheTier::L1Memory;
            self.l1_entries.insert(key.to_string(), entry);
            self.promotions += 1;
        }
    }

    fn evict_one(&mut self, tier: CacheTier) {
        let entries = match tier {
            CacheTier::L1Memory => &mut self.l1_entries,
            CacheTier::L2Disk => &mut self.l2_entries,
            CacheTier::L3Remote => return,
        };

        if entries.is_empty() {
            return;
        }

        let victim_key = match self.config.eviction_strategy {
            EvictionStrategy::LRU => entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone()),
            EvictionStrategy::LFU => entries
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),
            EvictionStrategy::TTL => {
                // Evict expired first, then fall back to LRU
                let expired = entries
                    .iter()
                    .filter(|(_, e)| e.is_expired())
                    .map(|(k, _)| k.clone())
                    .next();
                expired.or_else(|| {
                    entries
                        .iter()
                        .min_by_key(|(_, e)| e.last_accessed)
                        .map(|(k, _)| k.clone())
                })
            }
        };

        if let Some(key) = victim_key {
            entries.remove(&key);
            self.evictions += 1;
        }
    }
}

/// Response returned from the [`CachedLlmWrapper`].
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The LLM response text
    pub response: String,
    /// Whether this response was served from cache
    pub cache_hit: bool,
    /// Similarity score if served from cache
    pub similarity: Option<f32>,
    /// Latency of the operation in microseconds
    pub latency_us: u64,
}

/// Wrapper that combines a [`SemanticCache`] with user-provided embedding and
/// generation functions to transparently cache LLM responses.
pub struct CachedLlmWrapper<E, G>
where
    E: Fn(&str) -> Vec<f32>,
    G: Fn(&str) -> String,
{
    cache: SemanticCache,
    embed_fn: E,
    generate_fn: G,
}

impl<E, G> CachedLlmWrapper<E, G>
where
    E: Fn(&str) -> Vec<f32>,
    G: Fn(&str) -> String,
{
    /// Create a new cached LLM wrapper.
    pub fn new(cache: SemanticCache, embed_fn: E, generate_fn: G) -> Self {
        Self {
            cache,
            embed_fn,
            generate_fn,
        }
    }

    /// Query the LLM with automatic caching.
    ///
    /// Checks the cache first; on a miss, calls the generation function,
    /// stores the result, and returns it.
    pub fn query(&mut self, prompt: &str) -> CachedResponse {
        let start = Instant::now();
        let embedding = (self.embed_fn)(prompt);

        // Try cache lookup
        if let Some(entry) = self.cache.lookup(&embedding) {
            let similarity = {
                let dist = DistanceFunction::Cosine.compute(&embedding, &entry.embedding);
                1.0 - dist
            };
            return CachedResponse {
                response: entry.response,
                cache_hit: true,
                similarity: Some(similarity),
                latency_us: start.elapsed().as_micros() as u64,
            };
        }

        // Cache miss — generate and store
        let response = (self.generate_fn)(prompt);
        let _ = self.cache.store(prompt, &response, embedding, None);

        CachedResponse {
            response,
            cache_hit: false,
            similarity: None,
            latency_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Get a reference to the underlying cache.
    pub fn cache(&self) -> &SemanticCache {
        &self.cache
    }

    /// Get a mutable reference to the underlying cache.
    pub fn cache_mut(&mut self) -> &mut SemanticCache {
        &mut self.cache
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    /// Generate a deterministic mock embedding from a string key.
    fn mock_embedding(text: &str, dimensions: usize) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        let mut embedding = Vec::with_capacity(dimensions);
        let mut state = hash;

        for _ in 0..dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        embedding
    }

    fn test_config() -> SemanticCacheConfig {
        SemanticCacheConfig {
            similarity_threshold: 0.92,
            l1_max_entries: 5,
            l2_max_entries: 10,
            default_ttl: None,
            dimensions: 32,
            promotion_threshold: 3,
            eviction_strategy: EvictionStrategy::LRU,
        }
    }

    #[test]
    fn test_store_and_lookup() {
        let mut cache = SemanticCache::new(test_config());
        let emb = mock_embedding("hello world", 32);
        cache
            .store("hello world", "greeting response", emb.clone(), None)
            .unwrap();

        let result = cache.lookup(&emb);
        assert!(result.is_some());
        let entry = result.unwrap();
        assert_eq!(entry.response, "greeting response");
        assert_eq!(entry.key, "hello world");
    }

    #[test]
    fn test_similarity_threshold() {
        let config = SemanticCacheConfig {
            similarity_threshold: 0.99,
            dimensions: 32,
            ..test_config()
        };
        let mut cache = SemanticCache::new(config);
        let emb = mock_embedding("hello world", 32);
        cache
            .store("hello world", "greeting response", emb.clone(), None)
            .unwrap();

        // A very different embedding should not match at 0.99 threshold
        let different = mock_embedding("quantum physics equations", 32);
        let result = cache.lookup(&different);
        assert!(result.is_none());
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = SemanticCache::new(test_config());
        let emb = mock_embedding("nonexistent query", 32);
        let result = cache.lookup(&emb);
        assert!(result.is_none());

        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_ttl_expiry() {
        let config = SemanticCacheConfig {
            default_ttl: Some(Duration::from_millis(1)),
            dimensions: 32,
            ..test_config()
        };
        let mut cache = SemanticCache::new(config);
        let emb = mock_embedding("ephemeral", 32);
        cache
            .store("ephemeral", "short-lived", emb.clone(), None)
            .unwrap();

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(10));

        let result = cache.lookup(&emb);
        assert!(result.is_none(), "expected cache miss after TTL expiry");
    }

    #[test]
    fn test_l1_eviction() {
        let config = SemanticCacheConfig {
            l1_max_entries: 3,
            l2_max_entries: 10,
            dimensions: 32,
            ..test_config()
        };
        let mut cache = SemanticCache::new(config);

        // Fill L1
        for i in 0..3 {
            let key = format!("entry_{}", i);
            let emb = mock_embedding(&key, 32);
            cache
                .store(&key, &format!("resp_{}", i), emb, None)
                .unwrap();
        }
        assert_eq!(cache.stats().l1_size, 3);

        // Next entry should go to L2 since L1 is full
        let emb = mock_embedding("entry_3", 32);
        cache.store("entry_3", "resp_3", emb, None).unwrap();
        assert_eq!(cache.stats().l1_size, 3);
        assert_eq!(cache.stats().l2_size, 1);
    }

    #[test]
    fn test_invalidation() {
        let mut cache = SemanticCache::new(test_config());
        let emb = mock_embedding("to_remove", 32);
        cache
            .store("to_remove", "will be gone", emb.clone(), None)
            .unwrap();

        assert!(cache.lookup(&emb).is_some());

        cache.invalidate("to_remove");

        // The HNSW index still points to the vector, but the entry map is empty
        // so lookup should return None (key no longer in l1 or l2)
        let result = cache.lookup(&emb);
        assert!(result.is_none());
    }

    #[test]
    fn test_similar_invalidation() {
        let mut cache = SemanticCache::new(test_config());
        let emb = mock_embedding("target", 32);
        cache
            .store("target", "target response", emb.clone(), None)
            .unwrap();

        // Invalidate anything with similarity >= 0.5 to this embedding
        cache.invalidate_similar(&emb, 0.5);
        assert!(cache.l1_entries.is_empty() || !cache.l1_entries.contains_key("target"));
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = SemanticCache::new(test_config());
        let emb = mock_embedding("stats_test", 32);
        cache
            .store("stats_test", "resp", emb.clone(), None)
            .unwrap();

        // One hit
        let _ = cache.lookup(&emb);
        // One miss
        let _ = cache.lookup(&mock_embedding("nonexistent", 32));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_ratio - 0.5).abs() < 1e-6);
        assert_eq!(stats.l1_size, 1);
    }

    #[test]
    fn test_promotion() {
        let config = SemanticCacheConfig {
            l1_max_entries: 2,
            l2_max_entries: 10,
            promotion_threshold: 3,
            dimensions: 32,
            ..test_config()
        };
        let mut cache = SemanticCache::new(config);

        // Fill L1
        for i in 0..2 {
            let key = format!("l1_{}", i);
            let emb = mock_embedding(&key, 32);
            cache.store(&key, &format!("r_{}", i), emb, None).unwrap();
        }

        // This goes to L2
        let key = "promote_me";
        let emb = mock_embedding(key, 32);
        cache
            .store(key, "promoted response", emb.clone(), None)
            .unwrap();
        assert_eq!(cache.stats().l2_size, 1);

        // Access it enough times to trigger promotion
        for _ in 0..4 {
            let _ = cache.lookup(&emb);
        }

        // After promotion, it should be in L1
        let stats = cache.stats();
        assert!(stats.promotions >= 1);
    }

    #[test]
    fn test_cached_wrapper_hit_miss() {
        let config = SemanticCacheConfig {
            dimensions: 32,
            similarity_threshold: 0.90,
            ..test_config()
        };
        let cache = SemanticCache::new(config);

        let mut wrapper = CachedLlmWrapper::new(
            cache,
            |text: &str| mock_embedding(text, 32),
            |prompt: &str| format!("Generated: {}", prompt),
        );

        // First call — miss
        let resp1 = wrapper.query("What is Rust?");
        assert!(!resp1.cache_hit);
        assert_eq!(resp1.response, "Generated: What is Rust?");

        // Same query — hit
        let resp2 = wrapper.query("What is Rust?");
        assert!(resp2.cache_hit);
        assert_eq!(resp2.response, "Generated: What is Rust?");
        assert!(resp2.similarity.is_some());
    }

    #[test]
    fn test_eviction_strategy() {
        // Test LFU eviction
        let config = SemanticCacheConfig {
            l1_max_entries: 2,
            l2_max_entries: 2,
            eviction_strategy: EvictionStrategy::LFU,
            dimensions: 32,
            ..test_config()
        };
        let mut cache = SemanticCache::new(config);

        // Fill L1
        let emb_a = mock_embedding("a", 32);
        cache.store("a", "resp_a", emb_a.clone(), None).unwrap();
        let emb_b = mock_embedding("b", 32);
        cache.store("b", "resp_b", emb_b.clone(), None).unwrap();

        // Access 'a' several times so it has a higher access count
        for _ in 0..5 {
            let _ = cache.lookup(&emb_a);
        }

        // Fill L2
        let emb_c = mock_embedding("c", 32);
        cache.store("c", "resp_c", emb_c.clone(), None).unwrap();
        let emb_d = mock_embedding("d", 32);
        cache.store("d", "resp_d", emb_d.clone(), None).unwrap();

        // L2 is now full (2 entries). Store another; should evict least-frequently-used from L2.
        let emb_e = mock_embedding("e", 32);
        cache.store("e", "resp_e", emb_e, None).unwrap();

        assert!(cache.stats().evictions >= 1);
        // L2 should still be at capacity
        assert!(cache.stats().l2_size <= 2);
    }

    #[test]
    fn test_clear() {
        let mut cache = SemanticCache::new(test_config());
        let emb = mock_embedding("clear_test", 32);
        cache.store("clear_test", "resp", emb, None).unwrap();

        cache.clear();
        let stats = cache.stats();
        assert_eq!(stats.l1_size, 0);
        assert_eq!(stats.l2_size, 0);
    }

    #[test]
    fn test_evict_expired() {
        let config = SemanticCacheConfig {
            default_ttl: Some(Duration::from_millis(1)),
            dimensions: 32,
            ..test_config()
        };
        let mut cache = SemanticCache::new(config);
        let emb = mock_embedding("expire_me", 32);
        cache.store("expire_me", "resp", emb, None).unwrap();

        std::thread::sleep(Duration::from_millis(10));
        cache.evict_expired();

        assert_eq!(cache.stats().l1_size, 0);
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut cache = SemanticCache::new(test_config());
        let wrong_dim = vec![0.0f32; 64]; // config expects 32
        let result = cache.store("bad", "resp", wrong_dim, None);
        assert!(result.is_err());
    }
}
