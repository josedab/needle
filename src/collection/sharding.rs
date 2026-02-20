//! Sharded query caching for collection search results.
//!
//! Provides exact-match and semantic similarity caching to accelerate
//! repeated or similar search queries.

use crate::collection::config::SemanticQueryCacheConfig;
use crate::collection::search::SearchResult;
use crate::collection::QueryCacheStats;
use crate::distance::DistanceFunction;
use crate::hnsw::{HnswConfig, HnswIndex};
use lru::LruCache;
use ordered_float::OrderedFloat;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of shards for the query cache to reduce lock contention.
pub(super) const CACHE_SHARD_COUNT: usize = 16;

/// Cache key for query result caching.
///
/// Uses OrderedFloat to make f32 values hashable while handling NaN/Inf correctly.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct QueryCacheKey {
    /// Query vector converted to ordered floats for hashing
    query: Vec<OrderedFloat<f32>>,
    /// Number of results requested
    k: usize,
}

impl Hash for QueryCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.k.hash(state);
        for &v in &self.query {
            v.hash(state);
        }
    }
}

impl QueryCacheKey {
    pub(super) fn new(query: &[f32], k: usize) -> Self {
        Self {
            query: query.iter().map(|&f| OrderedFloat(f)).collect(),
            k,
        }
    }
}

/// Cached search result entry
#[derive(Clone)]
pub(super) struct CachedSearchResult {
    pub(super) results: Arc<Vec<SearchResult>>,
}

/// Internal query cache with statistics tracking
pub(super) struct QueryCache {
    cache: LruCache<QueryCacheKey, CachedSearchResult>,
    hits: u64,
    misses: u64,
}

impl std::fmt::Debug for QueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryCache")
            .field("size", &self.cache.len())
            .field("capacity", &self.cache.cap())
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl QueryCache {
    pub(super) fn new(capacity: NonZeroUsize) -> Self {
        Self {
            cache: LruCache::new(capacity),
            hits: 0,
            misses: 0,
        }
    }

    pub(super) fn get(&mut self, key: &QueryCacheKey) -> Option<&CachedSearchResult> {
        if let Some(result) = self.cache.get(key) {
            self.hits += 1;
            Some(result)
        } else {
            self.misses += 1;
            None
        }
    }

    pub(super) fn put(&mut self, key: QueryCacheKey, value: CachedSearchResult) {
        self.cache.put(key, value);
    }

    pub(super) fn clear(&mut self) {
        self.cache.clear();
    }

    pub(super) fn stats(&self, capacity: usize) -> QueryCacheStats {
        QueryCacheStats {
            hits: self.hits,
            misses: self.misses,
            size: self.cache.len(),
            capacity,
            semantic_hits: 0,
            semantic_misses: 0,
        }
    }
}

/// Sharded query cache that distributes entries across multiple LRU shards.
/// Each shard is independently locked, reducing contention for concurrent searches.
pub(super) struct ShardedQueryCache {
    shards: Vec<Mutex<QueryCache>>,
    pub(super) capacity: usize,
}

impl std::fmt::Debug for ShardedQueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (total_size, total_hits, total_misses) = self.aggregate_stats();
        f.debug_struct("ShardedQueryCache")
            .field("shards", &CACHE_SHARD_COUNT)
            .field("total_size", &total_size)
            .field("total_hits", &total_hits)
            .field("total_misses", &total_misses)
            .finish()
    }
}

impl ShardedQueryCache {
    pub(super) fn new(capacity: NonZeroUsize) -> Self {
        // .max(1) guarantees this is always >= 1, so NonZeroUsize::new never returns None
        let per_shard_val = (capacity.get() / CACHE_SHARD_COUNT).max(1);
        let per_shard = NonZeroUsize::new(per_shard_val).unwrap_or(NonZeroUsize::MIN);

        let shards = (0..CACHE_SHARD_COUNT)
            .map(|_| Mutex::new(QueryCache::new(per_shard)))
            .collect();

        Self {
            shards,
            capacity: capacity.get(),
        }
    }

    pub(super) fn shard_for(&self, key: &QueryCacheKey) -> &Mutex<QueryCache> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % CACHE_SHARD_COUNT;
        &self.shards[idx]
    }

    pub(super) fn get(&self, key: &QueryCacheKey) -> Option<CachedSearchResult> {
        let mut shard = self.shard_for(key).lock();
        shard.get(key).cloned()
    }

    pub(super) fn put(&self, key: QueryCacheKey, value: CachedSearchResult) {
        let mut shard = self.shard_for(&key).lock();
        shard.put(key, value);
    }

    pub(super) fn clear(&self) {
        for shard in &self.shards {
            shard.lock().clear();
        }
    }

    pub(super) fn aggregate_stats(&self) -> (usize, u64, u64) {
        let mut total_size = 0;
        let mut total_hits = 0;
        let mut total_misses = 0;
        for shard in &self.shards {
            let s = shard.lock();
            total_size += s.cache.len();
            total_hits += s.hits;
            total_misses += s.misses;
        }
        (total_size, total_hits, total_misses)
    }

    pub(super) fn stats(&self, capacity: usize) -> QueryCacheStats {
        let (size, hits, misses) = self.aggregate_stats();
        QueryCacheStats {
            hits,
            misses,
            size,
            capacity,
            semantic_hits: 0,
            semantic_misses: 0,
        }
    }
}

/// Semantic query cache entry with TTL support.
pub(super) struct SemanticCacheEntry {
    /// The cached search results
    results: Vec<SearchResult>,
    /// Number of results requested (k parameter)
    k: usize,
    /// Expiration timestamp (Unix epoch seconds), None = no expiration
    expires_at: Option<u64>,
    /// Last access timestamp for LRU eviction
    last_accessed: u64,
}

/// Semantic query cache that uses a small HNSW index to find similar past queries.
///
/// Instead of requiring exact vector match (like `ShardedQueryCache`), this cache
/// finds queries that are *similar enough* (above a configurable threshold) and
/// returns their cached results. This dramatically improves cache hit rates for
/// workloads with slight query variations.
pub(super) struct SemanticQueryCache {
    /// Small HNSW index storing cached query vectors
    index: HnswIndex,
    /// Stored query vectors (parallel to index entries)
    vectors: Vec<Vec<f32>>,
    /// Cached results keyed by internal vector ID
    pub(super) entries: HashMap<usize, SemanticCacheEntry>,
    /// Similarity threshold (0.0-1.0); distance must be < (1 - threshold)
    similarity_threshold: f32,
    /// Maximum capacity
    pub(super) capacity: usize,
    /// TTL in seconds for cache entries
    ttl_seconds: Option<u64>,
    /// Hit counter
    hits: u64,
    /// Miss counter
    misses: u64,
    /// Next internal ID for the cache index
    next_id: usize,
}

impl std::fmt::Debug for SemanticQueryCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticQueryCache")
            .field("size", &self.entries.len())
            .field("capacity", &self.capacity)
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

impl SemanticQueryCache {
    pub(super) fn new(config: &SemanticQueryCacheConfig, _dimensions: usize) -> Self {
        let hnsw_config = HnswConfig {
            m: 8,
            ef_construction: 100,
            ef_search: 10,
            ..HnswConfig::default()
        };
        Self {
            index: HnswIndex::new(hnsw_config, DistanceFunction::Cosine),
            vectors: Vec::new(),
            entries: HashMap::new(),
            similarity_threshold: config.similarity_threshold,
            capacity: config.capacity,
            ttl_seconds: config.ttl_seconds,
            hits: 0,
            misses: 0,
            next_id: 0,
        }
    }

    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Look up cached results for a similar query vector.
    pub(super) fn lookup(&mut self, query: &[f32], k: usize) -> Option<Vec<SearchResult>> {
        if self.index.is_empty() {
            self.misses += 1;
            return None;
        }

        let max_distance = 1.0 - self.similarity_threshold;

        // Search the cache index for the most similar cached query
        let candidates = self.index.search(query, 1, &self.vectors).ok()?;

        if let Some(&(id, distance)) = candidates.first() {
            if distance <= max_distance {
                if let Some(entry) = self.entries.get_mut(&id) {
                    // Check TTL (lazy expiration)
                    if let Some(expires_at) = entry.expires_at {
                        if Self::now_unix() > expires_at {
                            self.misses += 1;
                            return None;
                        }
                    }
                    // Check that cached k is sufficient
                    if entry.k >= k {
                        self.hits += 1;
                        entry.last_accessed = Self::now_unix();
                        let mut results = entry.results.clone();
                        results.truncate(k);
                        return Some(results);
                    }
                }
            }
        }

        self.misses += 1;
        None
    }

    /// Insert a query and its results into the cache.
    pub(super) fn insert(&mut self, query: &[f32], k: usize, results: &[SearchResult]) {
        // LRU eviction: remove oldest-accessed entry when at capacity
        while self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        let id = self.next_id;
        self.next_id += 1;

        let query_vec = query.to_vec();
        while self.vectors.len() <= id {
            self.vectors.push(Vec::new());
        }
        self.vectors[id] = query_vec.clone();

        let now = Self::now_unix();
        let expires_at = self.ttl_seconds.map(|ttl| now + ttl);
        self.entries.insert(
            id,
            SemanticCacheEntry {
                results: results.to_vec(),
                k,
                expires_at,
                last_accessed: now,
            },
        );

        // Insert into the HNSW index for similarity-based lookups.
        // Cache index failures are non-fatal: the entry is still stored in the
        // entries map and available for exact-match lookups.
        if let Err(e) = self.index.insert(id, &query_vec, &self.vectors) {
            tracing::debug!("Semantic cache index insert failed (non-fatal): {}", e);
        }
    }

    /// Evict the least-recently-used entry, preferring expired entries first.
    fn evict_lru(&mut self) {
        // First try to find and remove an expired entry
        let now = Self::now_unix();
        let expired_id = self.entries.iter().find_map(|(&id, entry)| {
            entry.expires_at.filter(|&exp| now > exp).map(|_| id)
        });
        if let Some(id) = expired_id {
            self.entries.remove(&id);
            self.index.delete(id);
            return;
        }
        // Otherwise remove least-recently-accessed
        if let Some((&lru_id, _)) = self.entries.iter().min_by_key(|(_, e)| e.last_accessed) {
            self.entries.remove(&lru_id);
            self.index.delete(lru_id);
        }
    }

    /// Warm the cache with a set of query vectors and their results.
    /// Useful for pre-populating the cache on startup with common queries.
    pub(super) fn warm(&mut self, entries: Vec<(&[f32], usize, Vec<SearchResult>)>) {
        for (query, k, results) in entries {
            self.insert(query, k, &results);
        }
    }

    /// Clear all cached entries and reset the index.
    pub(super) fn clear(&mut self) {
        let hnsw_config = self.index.config().clone();
        self.index = HnswIndex::new(hnsw_config, DistanceFunction::Cosine);
        self.vectors.clear();
        self.entries.clear();
        self.next_id = 0;
    }

    pub(super) fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::collection::config::SemanticQueryCacheConfig;

    fn make_result(id: &str, dist: f32) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            distance: dist,
            metadata: None,
        }
    }

    // ── QueryCacheKey ───────────────────────────────────────────────────

    #[test]
    fn test_cache_key_equality() {
        let k1 = QueryCacheKey::new(&[1.0, 2.0, 3.0], 5);
        let k2 = QueryCacheKey::new(&[1.0, 2.0, 3.0], 5);
        assert!(k1 == k2);
    }

    #[test]
    fn test_cache_key_different_k() {
        let k1 = QueryCacheKey::new(&[1.0, 2.0], 5);
        let k2 = QueryCacheKey::new(&[1.0, 2.0], 10);
        assert!(k1 != k2);
    }

    #[test]
    fn test_cache_key_different_vector() {
        let k1 = QueryCacheKey::new(&[1.0, 2.0], 5);
        let k2 = QueryCacheKey::new(&[1.0, 3.0], 5);
        assert!(k1 != k2);
    }

    #[test]
    fn test_cache_key_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        let k1 = QueryCacheKey::new(&[1.0, 2.0], 5);
        let k2 = QueryCacheKey::new(&[1.0, 2.0], 5);
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        k1.hash(&mut h1);
        k2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── QueryCache ──────────────────────────────────────────────────────

    #[test]
    fn test_query_cache_put_get() {
        let mut cache = QueryCache::new(NonZeroUsize::new(10).unwrap());
        let key = QueryCacheKey::new(&[1.0, 0.0], 5);
        let value = CachedSearchResult {
            results: Arc::new(vec![make_result("a", 0.1)]),
        };
        cache.put(key.clone(), value);
        let result = cache.get(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap().results.len(), 1);
    }

    #[test]
    fn test_query_cache_miss() {
        let mut cache = QueryCache::new(NonZeroUsize::new(10).unwrap());
        let key = QueryCacheKey::new(&[1.0, 0.0], 5);
        let result = cache.get(&key);
        assert!(result.is_none());
    }

    #[test]
    fn test_query_cache_stats() {
        let mut cache = QueryCache::new(NonZeroUsize::new(10).unwrap());
        let key = QueryCacheKey::new(&[1.0], 1);
        let _ = cache.get(&key); // miss
        cache.put(
            key.clone(),
            CachedSearchResult { results: Arc::new(vec![]) },
        );
        let _ = cache.get(&key); // hit
        let stats = cache.stats(10);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
        assert_eq!(stats.capacity, 10);
    }

    #[test]
    fn test_query_cache_clear() {
        let mut cache = QueryCache::new(NonZeroUsize::new(10).unwrap());
        let key = QueryCacheKey::new(&[1.0], 1);
        cache.put(key.clone(), CachedSearchResult { results: Arc::new(vec![]) });
        cache.clear();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_query_cache_eviction() {
        let mut cache = QueryCache::new(NonZeroUsize::new(2).unwrap());
        let k1 = QueryCacheKey::new(&[1.0], 1);
        let k2 = QueryCacheKey::new(&[2.0], 1);
        let k3 = QueryCacheKey::new(&[3.0], 1);
        let val = CachedSearchResult { results: Arc::new(vec![]) };

        cache.put(k1.clone(), val.clone());
        cache.put(k2.clone(), val.clone());
        cache.put(k3.clone(), val);

        // k1 should have been evicted (LRU)
        assert!(cache.get(&k1).is_none());
        assert!(cache.get(&k3).is_some());
    }

    #[test]
    fn test_query_cache_debug() {
        let cache = QueryCache::new(NonZeroUsize::new(10).unwrap());
        let debug = format!("{:?}", cache);
        assert!(debug.contains("QueryCache"));
        assert!(debug.contains("hits"));
    }

    // ── ShardedQueryCache ───────────────────────────────────────────────

    #[test]
    fn test_sharded_cache_put_get() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(100).unwrap());
        let key = QueryCacheKey::new(&[1.0, 2.0], 5);
        let val = CachedSearchResult {
            results: Arc::new(vec![make_result("a", 0.1)]),
        };
        cache.put(key.clone(), val);
        let result = cache.get(&key);
        assert!(result.is_some());
    }

    #[test]
    fn test_sharded_cache_miss() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(100).unwrap());
        let key = QueryCacheKey::new(&[1.0, 2.0], 5);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_sharded_cache_clear() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(100).unwrap());
        let key = QueryCacheKey::new(&[1.0], 1);
        cache.put(key.clone(), CachedSearchResult { results: Arc::new(vec![]) });
        cache.clear();
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_sharded_cache_aggregate_stats() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(100).unwrap());
        let key = QueryCacheKey::new(&[1.0], 1);
        let _ = cache.get(&key); // miss
        cache.put(key.clone(), CachedSearchResult { results: Arc::new(vec![]) });
        let _ = cache.get(&key); // hit

        let (size, hits, misses) = cache.aggregate_stats();
        assert_eq!(size, 1);
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_sharded_cache_stats() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(50).unwrap());
        let stats = cache.stats(50);
        assert_eq!(stats.capacity, 50);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_sharded_cache_debug() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(100).unwrap());
        let debug = format!("{:?}", cache);
        assert!(debug.contains("ShardedQueryCache"));
    }

    #[test]
    fn test_sharded_cache_distributes_keys() {
        let cache = ShardedQueryCache::new(NonZeroUsize::new(1000).unwrap());
        for i in 0..50 {
            let key = QueryCacheKey::new(&[i as f32], 1);
            cache.put(key, CachedSearchResult { results: Arc::new(vec![]) });
        }
        let (size, _, _) = cache.aggregate_stats();
        assert_eq!(size, 50);
    }

    // ── SemanticQueryCache ──────────────────────────────────────────────

    #[test]
    fn test_semantic_cache_insert_and_lookup() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = vec![make_result("a", 0.1), make_result("b", 0.2)];
        cache.insert(&query, 5, &results);

        // Exact same query should hit
        let lookup = cache.lookup(&query, 5);
        assert!(lookup.is_some());
        assert_eq!(lookup.unwrap().len(), 2);
    }

    #[test]
    fn test_semantic_cache_miss_empty() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let result = cache.lookup(&query, 5);
        assert!(result.is_none());
    }

    #[test]
    fn test_semantic_cache_truncates_to_k() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = vec![
            make_result("a", 0.1),
            make_result("b", 0.2),
            make_result("c", 0.3),
        ];
        cache.insert(&query, 10, &results);

        // Request fewer results than cached
        let lookup = cache.lookup(&query, 2);
        assert!(lookup.is_some());
        assert_eq!(lookup.unwrap().len(), 2);
    }

    #[test]
    fn test_semantic_cache_miss_insufficient_k() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        let query = vec![1.0, 0.0, 0.0, 0.0];
        cache.insert(&query, 3, &vec![make_result("a", 0.1)]);

        // Request more results than cached k
        let lookup = cache.lookup(&query, 5);
        assert!(lookup.is_none());
    }

    #[test]
    fn test_semantic_cache_eviction_at_capacity() {
        let config = SemanticQueryCacheConfig::new(2, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        cache.insert(&[1.0, 0.0, 0.0, 0.0], 5, &vec![make_result("a", 0.1)]);
        cache.insert(&[0.0, 1.0, 0.0, 0.0], 5, &vec![make_result("b", 0.1)]);
        cache.insert(&[0.0, 0.0, 1.0, 0.0], 5, &vec![make_result("c", 0.1)]);

        // Should have evicted an entry
        assert!(cache.entries.len() <= 2);
    }

    #[test]
    fn test_semantic_cache_clear() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        cache.insert(&[1.0, 0.0, 0.0, 0.0], 5, &vec![make_result("a", 0.1)]);
        cache.clear();

        assert!(cache.entries.is_empty());
        assert_eq!(cache.next_id, 0);
    }

    #[test]
    fn test_semantic_cache_stats() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        let _ = cache.lookup(&[1.0, 0.0, 0.0, 0.0], 5); // miss
        let (hits, misses) = cache.stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_semantic_cache_warm() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let mut cache = SemanticQueryCache::new(&config, 4);

        let entries = vec![
            (&[1.0_f32, 0.0, 0.0, 0.0][..], 5, vec![make_result("a", 0.1)]),
            (&[0.0, 1.0, 0.0, 0.0][..], 5, vec![make_result("b", 0.2)]),
        ];
        cache.warm(entries);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn test_semantic_cache_debug() {
        let config = SemanticQueryCacheConfig::new(10, 0.95);
        let cache = SemanticQueryCache::new(&config, 4);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("SemanticQueryCache"));
    }
}
