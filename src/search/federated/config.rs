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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_config() {
        let config = FederationConfig::default();
        assert_eq!(config.query_timeout, Duration::from_secs(10));
        assert_eq!(config.health_check_interval, Duration::from_secs(30));
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.min_instances, 1);
        assert!(config.allow_partial);
        assert_eq!(config.routing_strategy, RoutingStrategy::LatencyAware);
        assert_eq!(config.merge_strategy, MergeStrategy::DistanceBased);
        assert!(config.adaptive_routing);
        assert!((config.latency_weight - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.unhealthy_threshold, 3);
        assert_eq!(config.recovery_threshold, 2);
    }

    #[test]
    fn test_builder_with_timeout() {
        let config = FederationConfig::default().with_timeout(Duration::from_secs(30));
        assert_eq!(config.query_timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_builder_with_min_instances() {
        let config = FederationConfig::default().with_min_instances(3);
        assert_eq!(config.min_instances, 3);
    }

    #[test]
    fn test_builder_with_routing() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Broadcast);
        assert_eq!(config.routing_strategy, RoutingStrategy::Broadcast);
    }

    #[test]
    fn test_builder_with_merge() {
        let config =
            FederationConfig::default().with_merge(MergeStrategy::ReciprocalRankFusion);
        assert_eq!(config.merge_strategy, MergeStrategy::ReciprocalRankFusion);
    }

    #[test]
    fn test_builder_with_partial_results() {
        let config = FederationConfig::default().with_partial_results(false);
        assert!(!config.allow_partial);
    }

    #[test]
    fn test_builder_chaining() {
        let config = FederationConfig::default()
            .with_timeout(Duration::from_secs(5))
            .with_min_instances(2)
            .with_routing(RoutingStrategy::RoundRobin)
            .with_merge(MergeStrategy::Consensus)
            .with_partial_results(false);

        assert_eq!(config.query_timeout, Duration::from_secs(5));
        assert_eq!(config.min_instances, 2);
        assert_eq!(config.routing_strategy, RoutingStrategy::RoundRobin);
        assert_eq!(config.merge_strategy, MergeStrategy::Consensus);
        assert!(!config.allow_partial);
    }

    #[test]
    fn test_routing_strategy_variants() {
        let strategies = vec![
            RoutingStrategy::Broadcast,
            RoutingStrategy::LatencyAware,
            RoutingStrategy::GeographicProximity,
            RoutingStrategy::RoundRobin,
            RoutingStrategy::Random,
            RoutingStrategy::PriorityBased,
            RoutingStrategy::Quorum(3),
        ];
        // Verify all variants are distinct
        for (i, a) in strategies.iter().enumerate() {
            for (j, b) in strategies.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_merge_strategy_variants() {
        let strategies = vec![
            MergeStrategy::DistanceBased,
            MergeStrategy::ReciprocalRankFusion,
            MergeStrategy::FirstResponse,
            MergeStrategy::PriorityWeighted,
            MergeStrategy::Consensus,
        ];
        for (i, a) in strategies.iter().enumerate() {
            for (j, b) in strategies.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = FederationConfig::default()
            .with_routing(RoutingStrategy::GeographicProximity)
            .with_merge(MergeStrategy::PriorityWeighted);

        let json = serde_json::to_string(&config).expect("serialize");
        let deserialized: FederationConfig =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(
            deserialized.routing_strategy,
            RoutingStrategy::GeographicProximity
        );
        assert_eq!(deserialized.merge_strategy, MergeStrategy::PriorityWeighted);
        assert_eq!(deserialized.max_retries, config.max_retries);
    }

    #[test]
    fn test_quorum_routing_with_different_sizes() {
        let q1 = RoutingStrategy::Quorum(1);
        let q3 = RoutingStrategy::Quorum(3);
        let q5 = RoutingStrategy::Quorum(5);

        assert_ne!(q1, q3);
        assert_ne!(q3, q5);
        assert_eq!(q1, RoutingStrategy::Quorum(1));
    }

    #[test]
    fn test_merge_strategy_serialization() {
        for strategy in &[
            MergeStrategy::DistanceBased,
            MergeStrategy::ReciprocalRankFusion,
            MergeStrategy::FirstResponse,
            MergeStrategy::PriorityWeighted,
            MergeStrategy::Consensus,
        ] {
            let json = serde_json::to_string(strategy).expect("serialize");
            let deserialized: MergeStrategy =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(*strategy, deserialized);
        }
    }
}
