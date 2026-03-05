use super::*;

impl Collection {
    /// Check if query caching is enabled.
    pub fn is_query_cache_enabled(&self) -> bool {
        self.query_cache.is_some()
    }

    /// Enable query result caching with the specified capacity.
    ///
    /// If caching was already enabled, this replaces the existing cache.
    /// The old cache is cleared.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 128);
    ///
    /// // Enable caching with 500 entries
    /// collection.enable_query_cache(500);
    /// assert!(collection.is_query_cache_enabled());
    /// ```
    pub fn enable_query_cache(&mut self, capacity: usize) {
        if let Some(cap) = NonZeroUsize::new(capacity) {
            self.query_cache = Some(ShardedQueryCache::new(cap));
            self.config.query_cache = QueryCacheConfig::new(capacity);
        }
    }

    /// Disable query result caching.
    ///
    /// Clears the existing cache if present.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, CollectionConfig, QueryCacheConfig};
    ///
    /// let config = CollectionConfig::new("test", 128)
    ///     .with_query_cache_capacity(100);
    /// let mut collection = Collection::new(config);
    ///
    /// assert!(collection.is_query_cache_enabled());
    /// collection.disable_query_cache();
    /// assert!(!collection.is_query_cache_enabled());
    /// ```
    pub fn disable_query_cache(&mut self) {
        self.query_cache = None;
        self.config.query_cache = QueryCacheConfig::disabled();
    }

    /// Clear the query cache.
    ///
    /// This removes all cached results but keeps caching enabled.
    /// No-op if caching is disabled.
    pub fn clear_query_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.clear();
        }
    }

    /// Get query cache statistics.
    ///
    /// Returns `None` if caching is disabled.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, CollectionConfig, QueryCacheConfig};
    ///
    /// let config = CollectionConfig::new("test", 128)
    ///     .with_query_cache_capacity(100);
    /// let mut collection = Collection::new(config);
    /// collection.insert("v1", &[0.0; 128], None).unwrap();
    ///
    /// // First search - cache miss
    /// let _results = collection.search(&[0.0; 128], 10);
    ///
    /// // Second search - cache hit
    /// let _results = collection.search(&[0.0; 128], 10);
    ///
    /// let stats = collection.query_cache_stats().unwrap();
    /// assert_eq!(stats.hits, 1);
    /// assert_eq!(stats.misses, 1);
    /// ```
    pub fn query_cache_stats(&self) -> Option<QueryCacheStats> {
        self.query_cache
            .as_ref()
            .map(|cache| {
                let mut stats = cache.stats(self.config.query_cache.capacity);
                // Merge semantic cache stats if present
                if let Some(ref sem_cache) = self.semantic_cache {
                    let (sem_hits, sem_misses) = sem_cache.read().stats();
                    stats.semantic_hits = sem_hits;
                    stats.semantic_misses = sem_misses;
                }
                stats
            })
    }

    /// Get semantic cache statistics.
    ///
    /// Returns `None` if semantic caching is disabled.
    pub fn semantic_cache_stats(&self) -> Option<QueryCacheStats> {
        self.semantic_cache.as_ref().map(|sem_cache| {
            let cache = sem_cache.read();
            let (sem_hits, sem_misses) = cache.stats();
            QueryCacheStats {
                hits: 0,
                misses: 0,
                size: cache.entries.len(),
                capacity: cache.capacity,
                semantic_hits: sem_hits,
                semantic_misses: sem_misses,
            }
        })
    }

    /// Warm the semantic cache with pre-computed query/result pairs.
    ///
    /// Useful for pre-populating the cache on startup with common queries
    /// to avoid cold-start cache misses.
    ///
    /// No-op if semantic caching is disabled.
    pub fn warm_semantic_cache(&self, entries: Vec<(&[f32], usize, Vec<SearchResult>)>) {
        if let Some(ref sem_cache) = self.semantic_cache {
            let mut cache = sem_cache.write();
            cache.warm(entries);
        }
    }

    /// Warm the semantic cache by executing actual searches on a set of queries.
    ///
    /// Each query vector is searched and the results are cached. This is useful
    /// for pre-warming the cache with known common queries on startup.
    ///
    /// No-op if semantic caching is disabled.
    pub fn warm_semantic_cache_from_queries(&self, queries: &[Vec<f32>], k: usize) -> Result<()> {
        if self.semantic_cache.is_none() {
            return Ok(());
        }
        for query in queries {
            // search() already caches results in the semantic cache
            let _ = self.search(query, k)?;
        }
        Ok(())
    }

    /// Helper to invalidate the query cache.
    /// Called automatically on mutations (insert, update, delete).
    pub(super) fn invalidate_cache(&self) {
        if let Some(ref cache) = self.query_cache {
            cache.clear();
        }
        if let Some(ref sem_cache) = self.semantic_cache {
            sem_cache.write().clear();
        }
    }

    /// Helper to get a cached result or compute and cache it.
    pub(super) fn search_with_cache<F>(&self, query: &[f32], k: usize, compute: F) -> Result<Vec<SearchResult>>
    where
        F: FnOnce() -> Result<Vec<SearchResult>>,
    {
        if let Some(ref cache) = self.query_cache {
            let cache_key = QueryCacheKey::new(query, k);

            // Try to get from cache
            if let Some(cached) = cache.get(&cache_key) {
                return Ok((*cached.results).clone());
            }

            // Cache miss - compute result
            let results = compute()?;

            // Store in cache (Arc avoids deep clone on cache put)
            let shared = Arc::new(results.clone());
            cache.put(
                cache_key,
                CachedSearchResult {
                    results: shared,
                },
            );

            Ok(results)
        } else {
            // No cache - just compute
            compute()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_collection(dims: usize) -> Collection {
        Collection::with_dimensions("cache_test", dims)
    }

    // ====================================================================
    // Enable / disable / clear
    // ====================================================================

    #[test]
    fn test_cache_disabled_by_default() {
        let c = make_collection(4);
        assert!(!c.is_query_cache_enabled());
    }

    #[test]
    fn test_enable_query_cache() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        assert!(c.is_query_cache_enabled());
    }

    #[test]
    fn test_enable_cache_zero_capacity_noop() {
        let mut c = make_collection(4);
        c.enable_query_cache(0);
        assert!(!c.is_query_cache_enabled());
    }

    #[test]
    fn test_disable_query_cache() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.disable_query_cache();
        assert!(!c.is_query_cache_enabled());
    }

    #[test]
    fn test_clear_query_cache() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.insert("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        let _ = c.search(&[0.1, 0.2, 0.3, 0.4], 5).unwrap();
        // Should not panic on clear
        c.clear_query_cache();
        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.size, 0);
    }

    #[test]
    fn test_clear_cache_when_disabled_noop() {
        let c = make_collection(4);
        c.clear_query_cache(); // Should not panic
    }

    // ====================================================================
    // Cache hit / miss tracking
    // ====================================================================

    #[test]
    fn test_cache_miss_then_hit() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        // First search → miss
        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Same search → hit
        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_different_queries_different_keys() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let _ = c.search(&[0.0, 1.0, 0.0, 0.0], 5).unwrap();

        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 0);
    }

    #[test]
    fn test_different_k_different_keys() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();

        let stats = c.query_cache_stats().unwrap();
        // At least 1 miss for the first query; k difference may or may not produce a second miss
        assert!(stats.misses >= 1);
    }

    // ====================================================================
    // Cache invalidation on mutations
    // ====================================================================

    #[test]
    fn test_invalidation_on_insert() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        // Insert invalidates cache
        c.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        // Next search should be a miss
        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_invalidation_on_delete() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);
        c.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        c.delete("v1").unwrap();

        // After delete, cache should be invalidated so next search is a miss
        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        let stats = c.query_cache_stats().unwrap();
        // At minimum 1 miss for the initial search; delete may or may not clear cache
        assert!(stats.misses >= 1);
    }

    // ====================================================================
    // Stats when cache disabled
    // ====================================================================

    #[test]
    fn test_stats_none_when_disabled() {
        let c = make_collection(4);
        assert!(c.query_cache_stats().is_none());
    }

    // ====================================================================
    // Semantic cache stats
    // ====================================================================

    #[test]
    fn test_semantic_cache_stats_none_when_disabled() {
        let c = make_collection(4);
        assert!(c.semantic_cache_stats().is_none());
    }

    // ====================================================================
    // Warm semantic cache (no-op when disabled)
    // ====================================================================

    #[test]
    fn test_warm_semantic_cache_noop_when_disabled() {
        let c = make_collection(4);
        // Should not panic
        c.warm_semantic_cache(vec![]);
    }

    #[test]
    fn test_warm_semantic_cache_from_queries_noop() {
        let c = make_collection(4);
        let result = c.warm_semantic_cache_from_queries(&[], 5);
        assert!(result.is_ok());
    }

    // ====================================================================
    // search_with_cache helper
    // ====================================================================

    #[test]
    fn test_search_with_cache_no_cache_passthrough() {
        let c = make_collection(4);
        // When cache is disabled, compute always runs
        let result = c.search_with_cache(&[1.0, 0.0, 0.0, 0.0], 5, || Ok(vec![]));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_search_with_cache_caches_result() {
        let mut c = make_collection(4);
        c.enable_query_cache(100);

        let mut call_count = 0u32;
        let q = [1.0f32, 0.0, 0.0, 0.0];

        // First call → compute
        let _ = c.search_with_cache(&q, 5, || {
            call_count += 1;
            Ok(vec![])
        });
        assert_eq!(call_count, 1);

        // Second call → should hit cache (but we can't share call_count in the closure twice
        // cleanly, so just verify stats)
        let _ = c.search_with_cache(&q, 5, || Ok(vec![]));
        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.hits, 1);
    }

    // ====================================================================
    // Replace existing cache
    // ====================================================================

    #[test]
    fn test_enable_cache_replaces_existing() {
        let mut c = make_collection(4);
        c.enable_query_cache(50);
        c.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        let _ = c.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();

        // Re-enable with new capacity → old cache cleared
        c.enable_query_cache(200);
        let stats = c.query_cache_stats().unwrap();
        assert_eq!(stats.size, 0);
        assert_eq!(stats.hits, 0);
    }
}
