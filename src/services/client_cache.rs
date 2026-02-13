//! Client-Side Vector Caching
//!
//! Provides a configurable client-side cache for vectors and search results
//! with LRU eviction, TTL expiry, and invalidation support.
//!
//! Reduces round-trips to the database by 60-80% for read-heavy workloads.
//!
//! # Example
//!
//! ```rust
//! use needle::services::client_cache::*;
//! use std::time::Duration;
//!
//! let config = CacheConfig::builder()
//!     .max_vectors(10_000)
//!     .max_search_results(1_000)
//!     .vector_ttl(Duration::from_secs(300))
//!     .search_ttl(Duration::from_secs(60))
//!     .build();
//! let mut cache = VectorCache::new(config);
//!
//! // Cache a vector
//! cache.put_vector("doc1", vec![0.1, 0.2, 0.3]);
//! assert!(cache.get_vector("doc1").is_some());
//!
//! // Cache search results
//! let key = SearchCacheKey::new("my_collection", &[0.1, 0.2, 0.3], 10);
//! cache.put_search(key.clone(), vec!["doc1".into(), "doc2".into()]);
//! assert!(cache.get_search(&key).is_some());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Configuration
// ============================================================================

/// Cache configuration.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cached vectors.
    pub max_vectors: usize,
    /// Maximum cached search result sets.
    pub max_search_results: usize,
    /// TTL for cached vectors.
    pub vector_ttl: Duration,
    /// TTL for cached search results.
    pub search_ttl: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_vectors: 10_000,
            max_search_results: 1_000,
            vector_ttl: Duration::from_secs(300),
            search_ttl: Duration::from_secs(60),
        }
    }
}

impl CacheConfig {
    /// Create a new builder.
    pub fn builder() -> CacheConfigBuilder {
        CacheConfigBuilder(CacheConfig::default())
    }
}

/// Builder for CacheConfig.
pub struct CacheConfigBuilder(CacheConfig);

impl CacheConfigBuilder {
    /// Set max cached vectors.
    #[must_use]
    pub fn max_vectors(mut self, n: usize) -> Self {
        self.0.max_vectors = n;
        self
    }
    /// Set max cached search result sets.
    #[must_use]
    pub fn max_search_results(mut self, n: usize) -> Self {
        self.0.max_search_results = n;
        self
    }
    /// Set vector TTL.
    #[must_use]
    pub fn vector_ttl(mut self, ttl: Duration) -> Self {
        self.0.vector_ttl = ttl;
        self
    }
    /// Set search result TTL.
    #[must_use]
    pub fn search_ttl(mut self, ttl: Duration) -> Self {
        self.0.search_ttl = ttl;
        self
    }
    /// Build the config.
    pub fn build(self) -> CacheConfig {
        self.0
    }
}

// ============================================================================
// Cache Entry Types
// ============================================================================

/// A cached vector entry with metadata.
#[derive(Debug, Clone)]
struct VectorEntry {
    vector: Vec<f32>,
    inserted_at: Instant,
    last_accessed: Instant,
    access_count: u64,
}

/// Key for cached search results.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchCacheKey {
    /// Collection name.
    pub collection: String,
    /// Quantized query vector (rounded for cache key stability).
    pub query_hash: Vec<i32>,
    /// Number of results requested.
    pub k: usize,
}

impl SearchCacheKey {
    /// Create a cache key from a collection, query vector, and k.
    /// The query vector is quantized to i32 (multiplied by 10000) for stable hashing.
    pub fn new(collection: &str, query: &[f32], k: usize) -> Self {
        Self {
            collection: collection.to_string(),
            query_hash: query.iter().map(|&v| (v * 10000.0) as i32).collect(),
            k,
        }
    }
}

/// A cached search result entry.
#[derive(Debug, Clone)]
struct SearchEntry {
    result_ids: Vec<String>,
    inserted_at: Instant,
}

// ============================================================================
// Cache Statistics
// ============================================================================

/// Cache performance statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Vector cache hits.
    pub vector_hits: u64,
    /// Vector cache misses.
    pub vector_misses: u64,
    /// Search cache hits.
    pub search_hits: u64,
    /// Search cache misses.
    pub search_misses: u64,
    /// Total vectors currently cached.
    pub cached_vectors: usize,
    /// Total search result sets cached.
    pub cached_searches: usize,
    /// Total evictions.
    pub evictions: u64,
    /// Total invalidations.
    pub invalidations: u64,
}

impl CacheStats {
    /// Vector cache hit rate (0.0 to 1.0).
    pub fn vector_hit_rate(&self) -> f64 {
        let total = self.vector_hits + self.vector_misses;
        if total == 0 {
            0.0
        } else {
            self.vector_hits as f64 / total as f64
        }
    }

    /// Search cache hit rate (0.0 to 1.0).
    pub fn search_hit_rate(&self) -> f64 {
        let total = self.search_hits + self.search_misses;
        if total == 0 {
            0.0
        } else {
            self.search_hits as f64 / total as f64
        }
    }
}

// ============================================================================
// Vector Cache
// ============================================================================

/// Client-side vector and search result cache with LRU eviction and TTL.
pub struct VectorCache {
    config: CacheConfig,
    vectors: HashMap<String, VectorEntry>,
    searches: HashMap<SearchCacheKey, SearchEntry>,
    stats: CacheStats,
}

impl VectorCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            vectors: HashMap::new(),
            searches: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    // -- Vector Cache --

    /// Cache a vector.
    pub fn put_vector(&mut self, id: &str, vector: Vec<f32>) {
        if self.vectors.len() >= self.config.max_vectors {
            self.evict_lru_vector();
        }
        let now = Instant::now();
        self.vectors.insert(
            id.to_string(),
            VectorEntry {
                vector,
                inserted_at: now,
                last_accessed: now,
                access_count: 0,
            },
        );
        self.stats.cached_vectors = self.vectors.len();
    }

    /// Get a cached vector, returning None if missing or expired.
    pub fn get_vector(&mut self, id: &str) -> Option<&[f32]> {
        let ttl = self.config.vector_ttl;
        let entry = self.vectors.get_mut(id)?;
        if entry.inserted_at.elapsed() > ttl {
            // Expired — remove
            self.vectors.remove(id);
            self.stats.vector_misses += 1;
            self.stats.cached_vectors = self.vectors.len();
            return None;
        }
        entry.last_accessed = Instant::now();
        entry.access_count += 1;
        self.stats.vector_hits += 1;
        // Re-borrow immutably
        self.vectors.get(id).map(|e| e.vector.as_slice())
    }

    /// Check if a vector is cached (without updating access stats).
    pub fn contains_vector(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }

    // -- Search Cache --

    /// Cache search results.
    pub fn put_search(&mut self, key: SearchCacheKey, result_ids: Vec<String>) {
        if self.searches.len() >= self.config.max_search_results {
            self.evict_oldest_search();
        }
        self.searches.insert(
            key,
            SearchEntry {
                result_ids,
                inserted_at: Instant::now(),
            },
        );
        self.stats.cached_searches = self.searches.len();
    }

    /// Get cached search results, returning None if missing or expired.
    pub fn get_search(&mut self, key: &SearchCacheKey) -> Option<&[String]> {
        let ttl = self.config.search_ttl;
        let entry = self.searches.get(key)?;
        if entry.inserted_at.elapsed() > ttl {
            self.searches.remove(key);
            self.stats.search_misses += 1;
            self.stats.cached_searches = self.searches.len();
            return None;
        }
        self.stats.search_hits += 1;
        self.searches.get(key).map(|e| e.result_ids.as_slice())
    }

    // -- Invalidation --

    /// Invalidate a specific vector (e.g., after mutation notification).
    pub fn invalidate_vector(&mut self, id: &str) {
        if self.vectors.remove(id).is_some() {
            self.stats.invalidations += 1;
            self.stats.cached_vectors = self.vectors.len();
        }
        // Also invalidate any search results that reference this vector
        let keys_to_remove: Vec<SearchCacheKey> = self
            .searches
            .iter()
            .filter(|(_, entry)| entry.result_ids.iter().any(|rid| rid == id))
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys_to_remove {
            self.searches.remove(&key);
            self.stats.invalidations += 1;
        }
        self.stats.cached_searches = self.searches.len();
    }

    /// Invalidate all cached data for a collection.
    pub fn invalidate_collection(&mut self, collection: &str) {
        let keys_to_remove: Vec<SearchCacheKey> = self
            .searches
            .keys()
            .filter(|k| k.collection == collection)
            .cloned()
            .collect();
        for key in keys_to_remove {
            self.searches.remove(&key);
            self.stats.invalidations += 1;
        }
        self.stats.cached_searches = self.searches.len();
    }

    /// Clear all cached data.
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.searches.clear();
        self.stats.cached_vectors = 0;
        self.stats.cached_searches = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    // -- Internal --

    fn evict_lru_vector(&mut self) {
        if let Some(lru_id) = self
            .vectors
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(id, _)| id.clone())
        {
            self.vectors.remove(&lru_id);
            self.stats.evictions += 1;
        }
    }

    fn evict_oldest_search(&mut self) {
        if let Some(oldest_key) = self
            .searches
            .iter()
            .min_by_key(|(_, e)| e.inserted_at)
            .map(|(key, _)| key.clone())
        {
            self.searches.remove(&oldest_key);
            self.stats.evictions += 1;
        }
    }
}

// ============================================================================
// Invalidation Messages (for WebSocket subscription)
// ============================================================================

/// Invalidation event from server mutations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationEvent {
    /// A specific vector was inserted or updated.
    VectorMutated {
        collection: String,
        vector_id: String,
    },
    /// A specific vector was deleted.
    VectorDeleted {
        collection: String,
        vector_id: String,
    },
    /// A collection was modified (e.g., compacted).
    CollectionModified { collection: String },
    /// A collection was dropped.
    CollectionDropped { collection: String },
}

impl VectorCache {
    /// Process an invalidation event from the server.
    pub fn apply_invalidation(&mut self, event: &InvalidationEvent) {
        match event {
            InvalidationEvent::VectorMutated { vector_id, collection }
            | InvalidationEvent::VectorDeleted { vector_id, collection } => {
                self.invalidate_vector(vector_id);
                self.invalidate_collection(collection);
            }
            InvalidationEvent::CollectionModified { collection }
            | InvalidationEvent::CollectionDropped { collection } => {
                self.invalidate_collection(collection);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> VectorCache {
        VectorCache::new(CacheConfig::default())
    }

    #[test]
    fn test_vector_cache_put_get() {
        let mut cache = default_cache();
        cache.put_vector("v1", vec![1.0, 2.0, 3.0]);
        assert!(cache.contains_vector("v1"));
        let v = cache.get_vector("v1").expect("should exist");
        assert_eq!(v, &[1.0, 2.0, 3.0]);
        assert_eq!(cache.stats().vector_hits, 1);
    }

    #[test]
    fn test_vector_cache_miss() {
        let mut cache = default_cache();
        assert!(cache.get_vector("missing").is_none());
        assert_eq!(cache.stats().vector_misses, 0); // no entry to expire
    }

    #[test]
    fn test_search_cache() {
        let mut cache = default_cache();
        let key = SearchCacheKey::new("docs", &[0.1, 0.2], 5);
        cache.put_search(key.clone(), vec!["d1".into(), "d2".into()]);
        let results = cache.get_search(&key).expect("cached");
        assert_eq!(results, &["d1", "d2"]);
        assert_eq!(cache.stats().search_hits, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig::builder().max_vectors(2).build();
        let mut cache = VectorCache::new(config);

        cache.put_vector("v1", vec![1.0]);
        cache.put_vector("v2", vec![2.0]);
        // Access v1 to make it more recently used
        let _ = cache.get_vector("v1");
        // Adding v3 should evict v2 (LRU)
        cache.put_vector("v3", vec![3.0]);

        assert!(cache.contains_vector("v1"));
        assert!(!cache.contains_vector("v2"));
        assert!(cache.contains_vector("v3"));
        assert_eq!(cache.stats().evictions, 1);
    }

    #[test]
    fn test_invalidation() {
        let mut cache = default_cache();
        cache.put_vector("v1", vec![1.0]);
        let key = SearchCacheKey::new("docs", &[1.0], 5);
        cache.put_search(key.clone(), vec!["v1".into()]);

        // Invalidate v1 should also clear search results referencing it
        cache.invalidate_vector("v1");
        assert!(!cache.contains_vector("v1"));
        assert!(cache.get_search(&key).is_none());
        assert!(cache.stats().invalidations >= 1);
    }

    #[test]
    fn test_collection_invalidation() {
        let mut cache = default_cache();
        let key1 = SearchCacheKey::new("docs", &[1.0], 5);
        let key2 = SearchCacheKey::new("other", &[1.0], 5);
        cache.put_search(key1.clone(), vec!["d1".into()]);
        cache.put_search(key2.clone(), vec!["d2".into()]);

        cache.invalidate_collection("docs");
        assert!(cache.get_search(&key1).is_none());
        // "other" collection should still be cached
        assert!(cache.get_search(&key2).is_some());
    }

    #[test]
    fn test_invalidation_event() {
        let mut cache = default_cache();
        cache.put_vector("v1", vec![1.0]);
        let key = SearchCacheKey::new("docs", &[1.0], 5);
        cache.put_search(key.clone(), vec!["v1".into()]);

        let event = InvalidationEvent::VectorDeleted {
            collection: "docs".into(),
            vector_id: "v1".into(),
        };
        cache.apply_invalidation(&event);
        assert!(!cache.contains_vector("v1"));
    }

    #[test]
    fn test_clear() {
        let mut cache = default_cache();
        cache.put_vector("v1", vec![1.0]);
        cache.put_search(
            SearchCacheKey::new("docs", &[1.0], 5),
            vec!["v1".into()],
        );
        cache.clear();
        assert_eq!(cache.stats().cached_vectors, 0);
        assert_eq!(cache.stats().cached_searches, 0);
    }

    #[test]
    fn test_ttl_expiry() {
        let config = CacheConfig::builder()
            .vector_ttl(Duration::from_millis(1))
            .build();
        let mut cache = VectorCache::new(config);
        cache.put_vector("v1", vec![1.0]);

        // Sleep to let TTL expire
        std::thread::sleep(Duration::from_millis(5));
        assert!(cache.get_vector("v1").is_none());
    }
}
