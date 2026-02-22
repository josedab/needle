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
