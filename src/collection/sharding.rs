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
