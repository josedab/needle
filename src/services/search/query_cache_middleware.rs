#![allow(clippy::unwrap_used)]
//! Semantic Query Cache HTTP Integration
//!
//! Wires the SemanticCache into the HTTP search path as transparent middleware.
//! Auto-invalidates on collection mutations. Provides a stats dashboard endpoint.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::query_cache_middleware::{
//!     QueryCacheMiddleware, CacheMiddlewareConfig, CacheDashboard,
//! };
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 4).unwrap();
//!
//! let mut mw = QueryCacheMiddleware::new(CacheMiddlewareConfig::new(4));
//!
//! // Search goes through cache
//! let results = mw.cached_search(&db, "docs", &[1.0; 4], 10).unwrap();
//!
//! // On mutation, invalidate
//! mw.on_insert("docs", "v1");
//!
//! // Dashboard stats
//! let dashboard = mw.dashboard();
//! println!("Hit rate: {:.1}%", dashboard.hit_rate * 100.0);
//! ```

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

use crate::collection::SearchResult;
use crate::database::Database;
use crate::error::Result;
use crate::services::semantic_cache::{CacheConfig, SemanticCache};

/// Configuration for cache middleware.
#[derive(Debug, Clone)]
pub struct CacheMiddlewareConfig {
    pub dimensions: usize,
    pub similarity_threshold: f32,
    pub max_entries: usize,
    pub enabled: bool,
}

impl CacheMiddlewareConfig {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            similarity_threshold: 0.05,
            max_entries: 50_000,
            enabled: true,
        }
    }

    #[must_use]
    pub fn with_threshold(mut self, t: f32) -> Self { self.similarity_threshold = t; self }

    #[must_use]
    pub fn disabled(mut self) -> Self { self.enabled = false; self }
}

/// Cache dashboard metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheDashboard {
    pub enabled: bool,
    pub total_queries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f32,
    pub avg_hit_latency_us: u64,
    pub avg_miss_latency_us: u64,
    pub cached_entries: usize,
    pub estimated_savings_usd: f32,
    pub invalidations: u64,
}

/// Transparent query cache middleware for the HTTP search path.
pub struct QueryCacheMiddleware {
    config: CacheMiddlewareConfig,
    cache: SemanticCache,
    hit_latencies: Vec<u64>,
    miss_latencies: Vec<u64>,
    total_queries: u64,
    invalidations: u64,
    collection_vectors: HashMap<String, Vec<String>>,
}

impl QueryCacheMiddleware {
    pub fn new(config: CacheMiddlewareConfig) -> Self {
        let cache_config = CacheConfig::new(config.dimensions)
            .with_threshold(config.similarity_threshold)
            .with_max_entries(config.max_entries);
        Self {
            config,
            cache: SemanticCache::new(cache_config),
            hit_latencies: Vec::new(),
            miss_latencies: Vec::new(),
            total_queries: 0,
            invalidations: 0,
            collection_vectors: HashMap::new(),
        }
    }

    /// Execute a search with transparent caching.
    pub fn cached_search(
        &mut self,
        db: &Database,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        self.total_queries += 1;

        if !self.config.enabled {
            let coll = db.collection(collection)?;
            return coll.search(query, k);
        }

        let start = Instant::now();

        // Check cache
        if let Ok(Some(hit)) = self.cache.get(query, None) {
            let latency = start.elapsed().as_micros() as u64;
            self.hit_latencies.push(latency);
            // Deserialize cached results
            if let Ok(results) = serde_json::from_str::<Vec<CachedResult>>(&hit.response) {
                return Ok(results
                    .into_iter()
                    .map(|r| SearchResult {
                        id: r.id,
                        distance: r.distance,
                        metadata: r.metadata,
                    })
                    .collect());
            }
        }

        // Cache miss — execute real search
        let coll = db.collection(collection)?;
        let results = coll.search(query, k)?;

        let latency = start.elapsed().as_micros() as u64;
        self.miss_latencies.push(latency);

        // Store in cache
        let cached: Vec<CachedResult> = results
            .iter()
            .map(|r| CachedResult {
                id: r.id.clone(),
                distance: r.distance,
                metadata: r.metadata.clone(),
            })
            .collect();
        if let Ok(json) = serde_json::to_string(&cached) {
            if let Err(e) = self.cache.put(query, collection, &json, None) {
                debug!(collection, error = %e, "Failed to store search results in query cache");
            }
        }

        Ok(results)
    }

    /// Called when a vector is inserted — invalidate affected cache entries.
    pub fn on_insert(&mut self, collection: &str, vector_id: &str) {
        self.collection_vectors
            .entry(collection.to_string())
            .or_default()
            .push(vector_id.to_string());
        // Invalidate cache entries that reference this collection
        let count = self.cache.invalidate_for_vector(collection);
        self.invalidations += count as u64;
    }

    /// Called when a vector is deleted.
    pub fn on_delete(&mut self, _collection: &str, vector_id: &str) {
        let count = self.cache.invalidate_for_vector(vector_id);
        self.invalidations += count as u64;
    }

    /// Get cache dashboard stats.
    pub fn dashboard(&self) -> CacheDashboard {
        let analytics = self.cache.analytics();
        CacheDashboard {
            enabled: self.config.enabled,
            total_queries: self.total_queries,
            cache_hits: analytics.total_hits,
            cache_misses: analytics.total_misses,
            hit_rate: analytics.hit_rate(),
            avg_hit_latency_us: avg(&self.hit_latencies),
            avg_miss_latency_us: avg(&self.miss_latencies),
            cached_entries: self.cache.len(),
            estimated_savings_usd: analytics.estimated_savings_usd(0.002),
            invalidations: self.invalidations,
        }
    }

    /// Enable or disable the cache at runtime.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Whether the cache is enabled.
    pub fn is_enabled(&self) -> bool { self.config.enabled }
}

#[derive(Serialize, Deserialize)]
struct CachedResult {
    id: String,
    distance: f32,
    metadata: Option<Value>,
}

fn avg(values: &[u64]) -> u64 {
    if values.is_empty() { 0 } else { values.iter().sum::<u64>() / values.len() as u64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Database {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let coll = db.collection("docs").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        db
    }

    #[test]
    fn test_cached_search() {
        let db = setup();
        let mut mw = QueryCacheMiddleware::new(CacheMiddlewareConfig::new(4));

        let r1 = mw.cached_search(&db, "docs", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(!r1.is_empty());

        // Second identical query should hit cache
        let r2 = mw.cached_search(&db, "docs", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(r2.len(), r1.len());

        let dash = mw.dashboard();
        assert_eq!(dash.total_queries, 2);
        assert!(dash.cache_hits >= 1);
    }

    #[test]
    fn test_invalidation_on_insert() {
        let db = setup();
        let mut mw = QueryCacheMiddleware::new(CacheMiddlewareConfig::new(4));

        mw.cached_search(&db, "docs", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(mw.dashboard().cached_entries > 0);

        mw.on_insert("docs", "v3");
        // Cache entry referencing "docs" should be invalidated
        assert_eq!(mw.dashboard().invalidations, 1);
    }

    #[test]
    fn test_disabled_cache() {
        let db = setup();
        let mut mw = QueryCacheMiddleware::new(CacheMiddlewareConfig::new(4).disabled());

        mw.cached_search(&db, "docs", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(mw.dashboard().cached_entries, 0);
    }

    #[test]
    fn test_runtime_toggle() {
        let db = setup();
        let mut mw = QueryCacheMiddleware::new(CacheMiddlewareConfig::new(4));
        assert!(mw.is_enabled());

        mw.set_enabled(false);
        assert!(!mw.is_enabled());

        mw.cached_search(&db, "docs", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(mw.dashboard().cached_entries, 0);
    }

    #[test]
    fn test_clear_cache() {
        let db = setup();
        let mut mw = QueryCacheMiddleware::new(CacheMiddlewareConfig::new(4));
        mw.cached_search(&db, "docs", &[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(mw.dashboard().cached_entries > 0);

        mw.clear();
        assert_eq!(mw.dashboard().cached_entries, 0);
    }
}
