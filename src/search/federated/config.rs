use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Federation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationConfig {
    /// Query timeout
    pub query_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Number of retries for failed requests
    pub max_retries: usize,
    /// Minimum instances that must respond
    pub min_instances: usize,
    /// Allow partial results
    pub allow_partial: bool,
    /// Routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Result merge strategy
    pub merge_strategy: MergeStrategy,
    /// Enable adaptive routing based on latency
    pub adaptive_routing: bool,
    /// Latency weight for routing decisions (0-1)
    pub latency_weight: f32,
    /// Unhealthy threshold (consecutive failures)
    pub unhealthy_threshold: usize,
    /// Recovery threshold (consecutive successes)
    pub recovery_threshold: usize,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            query_timeout: Duration::from_secs(10),
            health_check_interval: Duration::from_secs(30),
            max_retries: 2,
            min_instances: 1,
            allow_partial: true,
            routing_strategy: RoutingStrategy::LatencyAware,
            merge_strategy: MergeStrategy::DistanceBased,
            adaptive_routing: true,
            latency_weight: 0.7,
            unhealthy_threshold: 3,
            recovery_threshold: 2,
        }
    }
}

impl FederationConfig {
    /// Set query timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Set minimum instances
    pub fn with_min_instances(mut self, min: usize) -> Self {
        self.min_instances = min;
        self
    }

    /// Set routing strategy
    pub fn with_routing(mut self, strategy: RoutingStrategy) -> Self {
        self.routing_strategy = strategy;
        self
    }

    /// Set merge strategy
    pub fn with_merge(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = strategy;
        self
    }

    /// Enable/disable partial results
    pub fn with_partial_results(mut self, allow: bool) -> Self {
        self.allow_partial = allow;
        self
    }
}

/// Routing strategy for selecting instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Query all instances (scatter-gather)
    Broadcast,
    /// Select based on latency
    LatencyAware,
    /// Select based on geographic proximity
    GeographicProximity,
    /// Round-robin selection
    RoundRobin,
    /// Random selection
    Random,
    /// Priority-based (prefer higher priority instances)
    PriorityBased,
    /// Quorum-based (wait for N responses)
    Quorum(usize),
}

/// Strategy for merging results from multiple instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge by distance (default for vector search)
    DistanceBased,
    /// Use reciprocal rank fusion
    ReciprocalRankFusion,
    /// First response wins (for latency-critical queries)
    FirstResponse,
    /// Weighted by instance priority
    PriorityWeighted,
    /// Consensus-based (deduplicate by ID)
    Consensus,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests needed: see docs/TODO-test-coverage.md
}
