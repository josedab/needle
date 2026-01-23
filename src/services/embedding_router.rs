//! Embedding Model Router
//!
//! Smart routing across multiple embedding providers with fallback chains,
//! per-collection model pinning, health-check-based failover, and cost tracking.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::embedding_router::{EmbeddingRouter, RouterConfig, ProviderEntry};
//!
//! let config = RouterConfig::default();
//! let router = EmbeddingRouter::new(config);
//!
//! // Register providers in priority order
//! router.register("ollama", ProviderEntry::new("ollama", 768, 0.0));
//! router.register("openai", ProviderEntry::new("openai", 1536, 0.0001));
//!
//! // Pin a collection to a specific provider
//! router.pin_collection("docs", "openai");
//!
//! // Route returns the best available provider name
//! let provider = router.route("docs");
//! assert_eq!(provider, Some("openai".to_string()));
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Configuration for the embedding router.
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Health check interval
    pub health_check_interval: Duration,
    /// Maximum consecutive failures before marking unhealthy
    pub max_failures: u32,
    /// Whether to cache embeddings across providers
    pub enable_cache: bool,
    /// Default provider (first healthy in chain)
    pub default_strategy: RoutingStrategy,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(60),
            max_failures: 3,
            enable_cache: true,
            default_strategy: RoutingStrategy::PriorityChain,
        }
    }
}

/// How to select among available providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Use providers in registered priority order (first healthy wins)
    PriorityChain,
    /// Use the cheapest healthy provider
    LowestCost,
    /// Use the lowest-latency healthy provider
    LowestLatency,
    /// Round-robin across healthy providers
    RoundRobin,
}

/// A registered embedding provider entry.
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    pub name: String,
    pub dimensions: usize,
    pub cost_per_token: f64,
    pub avg_latency_ms: f64,
    pub healthy: bool,
    pub consecutive_failures: u32,
    pub total_calls: u64,
    pub total_errors: u64,
    pub last_health_check: Option<Instant>,
}

impl ProviderEntry {
    pub fn new(name: impl Into<String>, dimensions: usize, cost_per_token: f64) -> Self {
        Self {
            name: name.into(),
            dimensions,
            cost_per_token,
            avg_latency_ms: 0.0,
            healthy: true,
            consecutive_failures: 0,
            total_calls: 0,
            total_errors: 0,
            last_health_check: None,
        }
    }

    pub fn record_success(&mut self, latency_ms: f64) {
        self.total_calls += 1;
        self.consecutive_failures = 0;
        self.healthy = true;
        // Exponential moving average
        if self.avg_latency_ms == 0.0 {
            self.avg_latency_ms = latency_ms;
        } else {
            self.avg_latency_ms = self.avg_latency_ms * 0.9 + latency_ms * 0.1;
        }
    }

    pub fn record_failure(&mut self, max_failures: u32) {
        self.total_calls += 1;
        self.total_errors += 1;
        self.consecutive_failures += 1;
        if self.consecutive_failures >= max_failures {
            self.healthy = false;
        }
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.total_errors as f64 / self.total_calls as f64
        }
    }
}

/// Embedding model router with fallback chains and health tracking.
pub struct EmbeddingRouter {
    config: RouterConfig,
    /// Providers in priority order
    providers: Vec<ProviderEntry>,
    /// Collection → provider name pinning
    collection_pins: HashMap<String, String>,
    /// Round-robin counter
    rr_counter: usize,
}

impl EmbeddingRouter {
    pub fn new(config: RouterConfig) -> Self {
        Self {
            config,
            providers: Vec::new(),
            collection_pins: HashMap::new(),
            rr_counter: 0,
        }
    }

    /// Register a provider in the fallback chain.
    pub fn register(&mut self, entry: ProviderEntry) {
        self.providers.push(entry);
    }

    /// Pin a collection to a specific provider.
    pub fn pin_collection(&mut self, collection: &str, provider: &str) {
        self.collection_pins
            .insert(collection.to_string(), provider.to_string());
    }

    /// Remove collection pinning.
    pub fn unpin_collection(&mut self, collection: &str) {
        self.collection_pins.remove(collection);
    }

    /// Route a request: returns the provider name to use.
    pub fn route(&mut self, collection: Option<&str>) -> Option<String> {
        // Check collection pinning first
        if let Some(coll) = collection {
            if let Some(pinned) = self.collection_pins.get(coll) {
                let provider = self.providers.iter().find(|p| &p.name == pinned && p.healthy);
                if let Some(p) = provider {
                    return Some(p.name.clone());
                }
                // Pinned provider unhealthy — fall through to strategy
            }
        }

        let healthy: Vec<&ProviderEntry> = self.providers.iter().filter(|p| p.healthy).collect();
        if healthy.is_empty() {
            return None;
        }

        match self.config.default_strategy {
            RoutingStrategy::PriorityChain => Some(healthy[0].name.clone()),
            RoutingStrategy::LowestCost => healthy
                .iter()
                .min_by(|a, b| {
                    a.cost_per_token
                        .partial_cmp(&b.cost_per_token)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p.name.clone()),
            RoutingStrategy::LowestLatency => healthy
                .iter()
                .min_by(|a, b| {
                    a.avg_latency_ms
                        .partial_cmp(&b.avg_latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p.name.clone()),
            RoutingStrategy::RoundRobin => {
                let idx = self.rr_counter % healthy.len();
                self.rr_counter += 1;
                Some(healthy[idx].name.clone())
            }
        }
    }

    /// Record a successful call for a provider.
    pub fn record_success(&mut self, provider: &str, latency_ms: f64) {
        if let Some(p) = self.providers.iter_mut().find(|p| p.name == provider) {
            p.record_success(latency_ms);
        }
    }

    /// Record a failed call for a provider.
    pub fn record_failure(&mut self, provider: &str) {
        let max = self.config.max_failures;
        if let Some(p) = self.providers.iter_mut().find(|p| p.name == provider) {
            p.record_failure(max);
        }
    }

    /// Get stats for all providers.
    pub fn stats(&self) -> Vec<ProviderStats> {
        self.providers
            .iter()
            .map(|p| ProviderStats {
                name: p.name.clone(),
                dimensions: p.dimensions,
                healthy: p.healthy,
                cost_per_token: p.cost_per_token,
                avg_latency_ms: p.avg_latency_ms,
                error_rate: p.error_rate(),
                total_calls: p.total_calls,
            })
            .collect()
    }

    /// Get the collection pinning map.
    pub fn pins(&self) -> &HashMap<String, String> {
        &self.collection_pins
    }
}

/// Provider statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderStats {
    pub name: String,
    pub dimensions: usize,
    pub healthy: bool,
    pub cost_per_token: f64,
    pub avg_latency_ms: f64,
    pub error_rate: f64,
    pub total_calls: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_chain_routing() {
        let mut router = EmbeddingRouter::new(RouterConfig::default());
        router.register(ProviderEntry::new("ollama", 768, 0.0));
        router.register(ProviderEntry::new("openai", 1536, 0.0001));

        assert_eq!(router.route(None), Some("ollama".to_string()));
    }

    #[test]
    fn test_collection_pinning() {
        let mut router = EmbeddingRouter::new(RouterConfig::default());
        router.register(ProviderEntry::new("ollama", 768, 0.0));
        router.register(ProviderEntry::new("openai", 1536, 0.0001));
        router.pin_collection("premium_docs", "openai");

        assert_eq!(
            router.route(Some("premium_docs")),
            Some("openai".to_string())
        );
        assert_eq!(
            router.route(Some("other")),
            Some("ollama".to_string())
        );
    }

    #[test]
    fn test_failover_on_unhealthy() {
        let mut router = EmbeddingRouter::new(RouterConfig {
            max_failures: 2,
            ..RouterConfig::default()
        });
        router.register(ProviderEntry::new("primary", 768, 0.0));
        router.register(ProviderEntry::new("backup", 768, 0.001));

        // Mark primary as failed
        router.record_failure("primary");
        router.record_failure("primary");

        assert_eq!(router.route(None), Some("backup".to_string()));
    }

    #[test]
    fn test_lowest_cost_strategy() {
        let mut router = EmbeddingRouter::new(RouterConfig {
            default_strategy: RoutingStrategy::LowestCost,
            ..RouterConfig::default()
        });
        router.register(ProviderEntry::new("expensive", 1536, 0.01));
        router.register(ProviderEntry::new("cheap", 768, 0.001));

        assert_eq!(router.route(None), Some("cheap".to_string()));
    }

    #[test]
    fn test_stats_tracking() {
        let mut router = EmbeddingRouter::new(RouterConfig::default());
        router.register(ProviderEntry::new("test", 768, 0.0));

        router.record_success("test", 50.0);
        router.record_success("test", 30.0);
        router.record_failure("test");

        let stats = router.stats();
        assert_eq!(stats[0].total_calls, 3);
        assert!(stats[0].error_rate > 0.0);
        assert!(stats[0].avg_latency_ms > 0.0);
    }

    #[test]
    fn test_round_robin() {
        let mut router = EmbeddingRouter::new(RouterConfig {
            default_strategy: RoutingStrategy::RoundRobin,
            ..RouterConfig::default()
        });
        router.register(ProviderEntry::new("a", 768, 0.0));
        router.register(ProviderEntry::new("b", 768, 0.0));

        assert_eq!(router.route(None), Some("a".to_string()));
        assert_eq!(router.route(None), Some("b".to_string()));
        assert_eq!(router.route(None), Some("a".to_string()));
    }
}
