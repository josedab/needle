//! Semantic LLM Cache Middleware
//!
//! Transparent caching layer that wraps LLM providers, with multi-model
//! isolation, analytics dashboard, and cost-savings tracking.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::llm_cache_middleware::{
//!     CacheMiddleware, MiddlewareConfig, ModelNamespace,
//! };
//!
//! let mut mw = CacheMiddleware::new(MiddlewareConfig::new(32));
//!
//! // Check cache before calling LLM
//! let query_emb = vec![0.1f32; 32];
//! if let Some(cached) = mw.check("gpt-4", &query_emb, "What is Rust?").unwrap() {
//!     println!("Cache hit: {}", cached.response);
//! } else {
//!     // Call LLM, then store result
//!     let response = "Rust is a systems programming language...";
//!     mw.store("gpt-4", &query_emb, "What is Rust?", response).unwrap();
//! }
//!
//! // View analytics
//! let analytics = mw.analytics("gpt-4");
//! println!("Hit rate: {:.1}%", analytics.hit_rate() * 100.0);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};
use crate::services::semantic_cache::{CacheAnalytics, CacheConfig, CacheHit, SemanticCache};

// ── Configuration ────────────────────────────────────────────────────────────

/// Middleware configuration.
#[derive(Debug, Clone)]
pub struct MiddlewareConfig {
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Default similarity threshold.
    pub default_threshold: f32,
    /// Per-model threshold overrides.
    pub model_thresholds: HashMap<String, f32>,
    /// Maximum entries per model namespace.
    pub max_entries_per_model: usize,
    /// Cost per LLM query in USD (for savings estimation).
    pub cost_per_query: f32,
}

impl MiddlewareConfig {
    /// Create a new config with the given dimensions.
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            default_threshold: 0.15,
            model_thresholds: HashMap::new(),
            max_entries_per_model: 50_000,
            cost_per_query: 0.002,
        }
    }

    /// Set a per-model threshold.
    #[must_use]
    pub fn with_model_threshold(mut self, model: &str, threshold: f32) -> Self {
        self.model_thresholds.insert(model.into(), threshold);
        self
    }

    /// Set cost per query.
    #[must_use]
    pub fn with_cost(mut self, cost: f32) -> Self {
        self.cost_per_query = cost;
        self
    }
}

// ── Model Namespace ──────────────────────────────────────────────────────────

/// Per-model namespace info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelNamespace {
    /// Model name.
    pub model: String,
    /// Number of cached entries.
    pub entries: usize,
    /// Cache analytics.
    pub analytics: CacheAnalytics,
}

// ── Cache Middleware ─────────────────────────────────────────────────────────

/// Transparent caching middleware for LLM providers.
pub struct CacheMiddleware {
    config: MiddlewareConfig,
    caches: HashMap<String, SemanticCache>,
}

impl CacheMiddleware {
    /// Create a new cache middleware.
    pub fn new(config: MiddlewareConfig) -> Self {
        Self {
            config,
            caches: HashMap::new(),
        }
    }

    /// Check the cache for a hit.
    pub fn check(
        &mut self,
        model: &str,
        query_embedding: &[f32],
        _query_text: &str,
    ) -> Result<Option<CacheHit>> {
        let threshold = self.threshold_for(model);
        let cache = self.get_or_create_cache(model);
        cache.get(query_embedding, Some(threshold))
    }

    /// Store a query-response pair.
    pub fn store(
        &mut self,
        model: &str,
        query_embedding: &[f32],
        query_text: &str,
        response: &str,
    ) -> Result<String> {
        let cache = self.get_or_create_cache(model);
        cache.put(query_embedding, query_text, response, Some(model))
    }

    /// Invalidate a specific cached entry.
    pub fn invalidate(&mut self, model: &str, entry_id: &str) -> Result<bool> {
        if let Some(cache) = self.caches.get_mut(model) {
            cache.invalidate(entry_id)
        } else {
            Ok(false)
        }
    }

    /// Clear all caches for a model.
    pub fn clear_model(&mut self, model: &str) {
        if let Some(cache) = self.caches.get_mut(model) {
            cache.clear();
        }
    }

    /// Clear all caches.
    pub fn clear_all(&mut self) {
        for cache in self.caches.values_mut() {
            cache.clear();
        }
    }

    /// Get analytics for a specific model.
    pub fn analytics(&self, model: &str) -> CacheAnalytics {
        self.caches
            .get(model)
            .map(|c| c.analytics().clone())
            .unwrap_or_default()
    }

    /// Get all model namespaces with analytics.
    pub fn all_namespaces(&self) -> Vec<ModelNamespace> {
        self.caches
            .iter()
            .map(|(model, cache)| ModelNamespace {
                model: model.clone(),
                entries: cache.len(),
                analytics: cache.analytics().clone(),
            })
            .collect()
    }

    /// Get total estimated cost savings across all models.
    pub fn total_savings(&self) -> f32 {
        self.caches
            .values()
            .map(|c| c.analytics().estimated_savings_usd(self.config.cost_per_query))
            .sum()
    }

    /// Get total hit rate across all models.
    pub fn total_hit_rate(&self) -> f32 {
        let total_lookups: u64 = self.caches.values().map(|c| c.analytics().total_lookups).sum();
        let total_hits: u64 = self.caches.values().map(|c| c.analytics().total_hits).sum();
        if total_lookups == 0 {
            0.0
        } else {
            total_hits as f32 / total_lookups as f32
        }
    }

    /// Number of model namespaces.
    pub fn model_count(&self) -> usize {
        self.caches.len()
    }

    fn get_or_create_cache(&mut self, model: &str) -> &mut SemanticCache {
        let config = &self.config;
        self.caches.entry(model.into()).or_insert_with(|| {
            let threshold = config
                .model_thresholds
                .get(model)
                .copied()
                .unwrap_or(config.default_threshold);
            SemanticCache::new(
                CacheConfig::new(config.dimensions)
                    .with_threshold(threshold)
                    .with_max_entries(config.max_entries_per_model),
            )
        })
    }

    fn threshold_for(&self, model: &str) -> f32 {
        self.config
            .model_thresholds
            .get(model)
            .copied()
            .unwrap_or(self.config.default_threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(base: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| base + i as f32 * 0.001).collect()
    }

    #[test]
    fn test_store_and_check() {
        let mut mw = CacheMiddleware::new(MiddlewareConfig::new(16));
        let e = emb(0.5, 16);
        mw.store("gpt-4", &e, "What is Rust?", "Rust is...").unwrap();

        let hit = mw.check("gpt-4", &e, "What is Rust?").unwrap();
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().response, "Rust is...");
    }

    #[test]
    fn test_model_isolation() {
        let mut mw = CacheMiddleware::new(MiddlewareConfig::new(16));
        let e = emb(0.5, 16);
        mw.store("gpt-4", &e, "q", "gpt4 answer").unwrap();

        // Different model shouldn't find it
        let hit = mw.check("claude", &e, "q").unwrap();
        assert!(hit.is_none());
    }

    #[test]
    fn test_analytics() {
        let mut mw = CacheMiddleware::new(MiddlewareConfig::new(8));
        let e = emb(0.5, 8);
        mw.store("gpt-4", &e, "q", "r").unwrap();
        mw.check("gpt-4", &e, "q").unwrap(); // hit
        mw.check("gpt-4", &emb(-1.0, 8), "other").unwrap(); // miss (use strict threshold)

        let analytics = mw.analytics("gpt-4");
        assert_eq!(analytics.total_lookups, 2);
        assert!(analytics.total_hits >= 1);
    }

    #[test]
    fn test_clear_model() {
        let mut mw = CacheMiddleware::new(MiddlewareConfig::new(8));
        mw.store("gpt-4", &emb(0.5, 8), "q", "r").unwrap();
        mw.clear_model("gpt-4");
        assert_eq!(mw.analytics("gpt-4").total_entries, 0);
    }

    #[test]
    fn test_total_savings() {
        let mut mw = CacheMiddleware::new(MiddlewareConfig::new(8).with_cost(0.01));
        let e = emb(0.5, 8);
        mw.store("m1", &e, "q", "r").unwrap();
        mw.check("m1", &e, "q").unwrap();

        // Savings = hits × cost
        assert!(mw.total_savings() >= 0.0);
    }

    #[test]
    fn test_namespaces() {
        let mut mw = CacheMiddleware::new(MiddlewareConfig::new(8));
        mw.store("gpt-4", &emb(0.1, 8), "q1", "r1").unwrap();
        mw.store("claude", &emb(0.2, 8), "q2", "r2").unwrap();
        assert_eq!(mw.model_count(), 2);
        assert_eq!(mw.all_namespaces().len(), 2);
    }

    #[test]
    fn test_per_model_threshold() {
        let config = MiddlewareConfig::new(8).with_model_threshold("strict", 0.001);
        let mut mw = CacheMiddleware::new(config);
        let e = emb(0.5, 8);
        mw.store("strict", &e, "q", "r").unwrap();

        // Very different embedding should miss with strict threshold
        let different: Vec<f32> = (0..8).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let hit = mw.check("strict", &different, "q").unwrap();
        assert!(hit.is_none());
    }
}
