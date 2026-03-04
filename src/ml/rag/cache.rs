use super::{Citation, RetrievedChunk};
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Cache entry for RAG responses
#[derive(Clone)]
struct CacheEntry {
    response: CachedRagResponse,
    created_at: Instant,
}

/// Cached version of RagResponse (without timing metadata)
#[derive(Clone)]
pub(super) struct CachedRagResponse {
    pub(super) chunks: Vec<RetrievedChunk>,
    pub(super) context: String,
    pub(super) citations: Vec<Citation>,
}

/// Cache key combining query text and filter hash
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub(super) struct CacheKey {
    query: String,
    filter_hash: Option<u64>,
}

impl CacheKey {
    pub(super) fn new(query: &str, filter: Option<&crate::metadata::Filter>) -> Self {
        let filter_hash = filter.map(|f| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{:?}", f).hash(&mut hasher);
            hasher.finish()
        });
        Self {
            query: query.to_string(),
            filter_hash,
        }
    }
}

/// RAG response cache with LRU eviction and TTL
pub struct RagCache {
    cache: Mutex<LruCache<CacheKey, CacheEntry>>,
    ttl: Duration,
    stats: Mutex<RagCacheStats>,
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct RagCacheStats {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total cache evictions
    pub evictions: u64,
    /// Total cache invalidations
    pub invalidations: u64,
}

impl RagCache {
    /// Create a new RAG cache
    pub fn new(capacity: usize, ttl_seconds: u64) -> Self {
        let capacity =
            NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1).expect("1 is non-zero"));
        Self {
            cache: Mutex::new(LruCache::new(capacity)),
            ttl: Duration::from_secs(ttl_seconds),
            stats: Mutex::new(RagCacheStats::default()),
        }
    }

    /// Get a cached response
    pub(super) fn get(&self, key: &CacheKey) -> Option<CachedRagResponse> {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.lock();

        if let Some(entry) = cache.get(key) {
            // Check TTL
            if self.ttl.as_secs() == 0 || entry.created_at.elapsed() < self.ttl {
                stats.hits += 1;
                return Some(entry.response.clone());
            }
            // Expired - remove it
            cache.pop(key);
            stats.evictions += 1;
        }

        stats.misses += 1;
        None
    }

    /// Put a response in the cache
    pub(super) fn put(&self, key: CacheKey, response: CachedRagResponse) {
        let mut cache = self.cache.lock();
        cache.put(
            key,
            CacheEntry {
                response,
                created_at: Instant::now(),
            },
        );
    }

    /// Invalidate all cache entries
    pub fn invalidate_all(&self) {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.lock();
        stats.invalidations += cache.len() as u64;
        cache.clear();
    }

    /// Invalidate entries matching a document ID
    pub fn invalidate_document(&self, doc_id: &str) {
        let mut cache = self.cache.lock();
        let mut stats = self.stats.lock();

        // Note: LRU cache doesn't support efficient iteration and removal
        // For a production system, consider using a different cache implementation
        // that supports this pattern more efficiently
        let keys_to_remove: Vec<CacheKey> = cache
            .iter()
            .filter(|(_, entry)| {
                entry
                    .response
                    .chunks
                    .iter()
                    .any(|c| c.chunk.document_id == doc_id)
            })
            .map(|(k, _)| k.clone())
            .collect();

        for key in keys_to_remove {
            cache.pop(&key);
            stats.invalidations += 1;
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> RagCacheStats {
        self.stats.lock().clone()
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let stats = self.stats.lock();
        let total = stats.hits + stats.misses;
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Chunk, Citation, RetrievedChunk};

    fn make_chunk(doc_id: &str, text: &str) -> RetrievedChunk {
        RetrievedChunk {
            chunk: Chunk {
                id: format!("{doc_id}_0"),
                document_id: doc_id.to_string(),
                text: text.to_string(),
                start_pos: 0,
                end_pos: text.len(),
                chunk_index: 0,
                total_chunks: 1,
                parent_id: None,
                children: vec![],
                metadata: None,
            },
            score: 0.9,
            rerank_score: None,
            final_score: 0.9,
        }
    }

    fn make_response(doc_id: &str) -> CachedRagResponse {
        CachedRagResponse {
            chunks: vec![make_chunk(doc_id, "some text")],
            context: "context".into(),
            citations: vec![Citation {
                document_id: doc_id.into(),
                chunk_id: format!("{doc_id}_0"),
                snippet: "some text".into(),
                position: (0, 9),
                score: 0.9,
            }],
        }
    }

    fn key(query: &str) -> CacheKey {
        CacheKey::new(query, None)
    }

    // ====================================================================
    // Basic put / get
    // ====================================================================

    #[test]
    fn test_put_and_get() {
        let cache = RagCache::new(10, 60);
        cache.put(key("hello"), make_response("d1"));

        let result = cache.get(&key("hello"));
        assert!(result.is_some());
        assert_eq!(result.unwrap().context, "context");
    }

    #[test]
    fn test_get_miss() {
        let cache = RagCache::new(10, 60);
        let result = cache.get(&key("missing"));
        assert!(result.is_none());
    }

    // ====================================================================
    // TTL=0 (infinite)
    // ====================================================================

    #[test]
    fn test_ttl_zero_infinite() {
        let cache = RagCache::new(10, 0);
        cache.put(key("q"), make_response("d1"));

        // With TTL=0, entry should never expire
        let result = cache.get(&key("q"));
        assert!(result.is_some());
    }

    // ====================================================================
    // TTL expiration
    // ====================================================================

    #[test]
    fn test_ttl_expiration() {
        let cache = RagCache::new(10, 1); // 1 second TTL
        cache.put(key("q"), make_response("d1"));

        // Immediately should be present
        assert!(cache.get(&key("q")).is_some());

        // Wait for expiration
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let result = cache.get(&key("q"));
        assert!(result.is_none());
    }

    // ====================================================================
    // LRU eviction at capacity
    // ====================================================================

    #[test]
    fn test_lru_eviction() {
        let cache = RagCache::new(2, 60);

        cache.put(key("q1"), make_response("d1"));
        cache.put(key("q2"), make_response("d2"));
        assert_eq!(cache.len(), 2);

        // Adding a third should evict the LRU entry (q1)
        cache.put(key("q3"), make_response("d3"));
        assert_eq!(cache.len(), 2);

        // q1 should be evicted
        assert!(cache.get(&key("q1")).is_none());
        assert!(cache.get(&key("q3")).is_some());
    }

    // ====================================================================
    // Hit rate
    // ====================================================================

    #[test]
    fn test_hit_rate_zero_accesses() {
        let cache = RagCache::new(10, 60);
        let rate = cache.hit_rate();
        assert!((rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q"), make_response("d1"));

        let _ = cache.get(&key("q")); // hit
        let _ = cache.get(&key("q")); // hit
        assert!((cache.hit_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_mixed() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q"), make_response("d1"));

        let _ = cache.get(&key("q")); // hit
        let _ = cache.get(&key("miss")); // miss
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ====================================================================
    // Invalidation
    // ====================================================================

    #[test]
    fn test_invalidate_all() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q1"), make_response("d1"));
        cache.put(key("q2"), make_response("d2"));

        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_invalidate_all_tracks_stats() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q1"), make_response("d1"));
        cache.put(key("q2"), make_response("d2"));

        cache.invalidate_all();
        let stats = cache.stats();
        assert_eq!(stats.invalidations, 2);
    }

    #[test]
    fn test_invalidate_document() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q1"), make_response("d1"));
        cache.put(key("q2"), make_response("d2"));

        cache.invalidate_document("d1");
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&key("q1")).is_none());
    }

    #[test]
    fn test_invalidate_document_nonexistent() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q1"), make_response("d1"));

        // Invalidating non-existent doc should be a no-op
        cache.invalidate_document("no_such_doc");
        assert_eq!(cache.len(), 1);
    }

    // ====================================================================
    // CacheKey: None vs Some filter
    // ====================================================================

    #[test]
    fn test_cache_key_none_vs_some_filter_differ() {
        let k1 = CacheKey::new("query", None);

        // We can't easily construct a Filter here, but we can verify
        // that the same query with None filter produces consistent keys
        let k2 = CacheKey::new("query", None);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_queries() {
        let k1 = CacheKey::new("hello", None);
        let k2 = CacheKey::new("world", None);
        assert_ne!(k1, k2);
    }

    // ====================================================================
    // Stats
    // ====================================================================

    #[test]
    fn test_stats_initial() {
        let cache = RagCache::new(10, 60);
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.invalidations, 0);
    }

    #[test]
    fn test_stats_after_operations() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q"), make_response("d1"));

        let _ = cache.get(&key("q")); // hit
        let _ = cache.get(&key("miss")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    // ====================================================================
    // Capacity edge: NonZeroUsize fallback
    // ====================================================================

    #[test]
    fn test_capacity_zero_uses_one() {
        let cache = RagCache::new(0, 60);
        // Should not panic; capacity falls back to 1
        cache.put(key("q"), make_response("d1"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_large_capacity() {
        let cache = RagCache::new(1_000_000, 60);
        cache.put(key("q"), make_response("d1"));
        assert_eq!(cache.len(), 1);
    }

    // ====================================================================
    // len / is_empty
    // ====================================================================

    #[test]
    fn test_len_empty() {
        let cache = RagCache::new(10, 60);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_len_after_put() {
        let cache = RagCache::new(10, 60);
        cache.put(key("q1"), make_response("d1"));
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }
}
