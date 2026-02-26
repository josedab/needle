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
                    let (sem_hits, sem_misses) = sem_cache.lock().stats();
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
            let cache = sem_cache.lock();
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
            let mut cache = sem_cache.lock();
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
            sem_cache.lock().clear();
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
}
