//! # Federated Multi-Instance Search
//!
//! Enables cross-datacenter queries with latency-aware routing and distributed
//! result merging for horizontal scaling across multiple Needle instances.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Federation Coordinator                        │
//! │                                                                  │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │Instance Registry│ │Health Monitor│ │Result Merger │          │
//! │  └──────────────┘  └──────────────┘  └──────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//!                              │
//!           ┌──────────────────┼──────────────────┐
//!           │                  │                  │
//!           ▼                  ▼                  ▼
//!     ┌──────────┐      ┌──────────┐      ┌──────────┐
//!     │Instance A│      │Instance B│      │Instance C│
//!     │ (US-West)│      │ (US-East)│      │ (EU)     │
//!     └──────────┘      └──────────┘      └──────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::federated::{Federation, FederationConfig, InstanceConfig};
//!
//! // Create federation
//! let config = FederationConfig::default();
//! let federation = Federation::new(config);
//!
//! // Register instances
//! federation.register_instance(InstanceConfig {
//!     id: "us-west-1".to_string(),
//!     endpoint: "http://needle-west.example.com:8080".to_string(),
//!     region: "us-west".to_string(),
//!     priority: 1,
//! });
//!
//! // Federated search
//! let results = federation.search("my_collection", &query_vector, 10).await?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

pub mod config;
pub mod coordinator;
pub mod discovery;
pub mod health;
pub mod instance;
pub mod merger;
pub mod sharding;

pub use config::{FederationConfig, MergeStrategy, RoutingStrategy};
pub use coordinator::{Federation, FederationStats, FederationStatsSnapshot};
pub use discovery::{
    DiscoveryConfig, DiscoveryService, GossipConfig, GossipMemberState, GossipMessage,
    GossipProtocol, GossipTickResult, PeerInfo, PeerState,
};
pub use health::{FederationHealth, HealthCheckResult, HealthMonitor};
pub use instance::{HealthStatus, InstanceConfig, InstanceInfo, InstanceRegistry};
pub use merger::{FederatedSearchResponse, FederatedSearchResult, ResultMerger};
pub use sharding::{
    ConsistencyLevel, CrossInstanceDedup, DedupStrategy, HashRing, QueryPlan, QueryPlanner,
    RebalanceResult, ShardManager, ShardMove,
};

// ============================================================================
// Error Types
// ============================================================================

/// Federation errors
#[derive(Error, Debug)]
pub enum FederationError {
    #[error("Instance not found: {0}")]
    InstanceNotFound(String),

    #[error("No healthy instances available")]
    NoHealthyInstances,

    #[error("Query timeout after {0:?}")]
    Timeout(Duration),

    #[error("Partial results: {success} of {total} instances responded")]
    PartialResults { success: usize, total: usize },

    #[error("All instances failed: {0}")]
    AllInstancesFailed(String),

    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Instance unhealthy: {0}")]
    UnhealthyInstance(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Quorum not reached: need {required}, got {available}")]
    QuorumNotReached { required: usize, available: usize },
}

pub type FederationResult<T> = std::result::Result<T, FederationError>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_federation() -> Federation {
        let config = FederationConfig::default();
        let federation = Federation::new(config);

        // Register test instances
        federation.register_instance(
            InstanceConfig::new("us-west-1", "http://localhost:8081")
                .with_region("us-west")
                .with_priority(2),
        );

        federation.register_instance(
            InstanceConfig::new("us-east-1", "http://localhost:8082")
                .with_region("us-east")
                .with_priority(1),
        );

        federation.register_instance(
            InstanceConfig::new("eu-west-1", "http://localhost:8083")
                .with_region("eu-west")
                .with_priority(1),
        );

        // Mark all as healthy
        federation
            .registry
            .update_health("us-west-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("us-east-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("eu-west-1", HealthStatus::Healthy);

        federation
    }

    #[test]
    fn test_instance_registry() {
        let registry = InstanceRegistry::new();

        registry.register(InstanceConfig::new("test-1", "http://localhost:8080"));

        assert_eq!(registry.len(), 1);
        assert!(registry.get("test-1").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_instance_registration() {
        let federation = create_test_federation();

        assert_eq!(federation.list_instances().len(), 3);
        assert!(federation.get_instance("us-west-1").is_some());
    }

    #[test]
    fn test_instance_health_tracking() {
        let federation = create_test_federation();

        // Record some queries
        federation.registry.record_query("us-west-1", 10.0, true);
        federation.registry.record_query("us-west-1", 15.0, true);
        federation.registry.record_query("us-west-1", 20.0, false);

        let info = federation.get_instance("us-west-1").unwrap();
        assert_eq!(info.total_queries, 3);
        assert_eq!(info.failed_queries, 1);
        assert!(info.avg_latency_ms > 0.0);
    }

    #[test]
    fn test_select_instances_broadcast() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Broadcast);
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("inst-1", "http://localhost:8081"));
        federation.register_instance(InstanceConfig::new("inst-2", "http://localhost:8082"));
        federation
            .registry
            .update_health("inst-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-2", HealthStatus::Healthy);

        let instances = federation.select_instances("test_collection").unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_select_instances_latency_aware() {
        let federation = create_test_federation();

        // Simulate different latencies
        federation.registry.record_query("us-west-1", 100.0, true);
        federation.registry.record_query("us-east-1", 50.0, true);
        federation.registry.record_query("eu-west-1", 200.0, true);

        let instances = federation.select_instances("test_collection").unwrap();

        // Should prefer us-east-1 (lowest latency) but also consider priority
        assert!(!instances.is_empty());
    }

    #[test]
    fn test_select_instances_no_healthy() {
        let config = FederationConfig::default();
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("unhealthy", "http://localhost:8080"));
        federation
            .registry
            .update_health("unhealthy", HealthStatus::Unhealthy);

        let result = federation.select_instances("test_collection");
        assert!(matches!(result, Err(FederationError::NoHealthyInstances)));
    }

    #[test]
    fn test_federated_search() {
        let federation = create_test_federation();

        let query = vec![0.1; 128];
        let result = federation.search("test_collection", &query, 10).unwrap();

        assert!(!result.results.is_empty());
        assert!(!result.instances_responded.is_empty());
        assert!(result.execution_time_ms > 0.0);
    }

    #[test]
    fn test_result_merger_distance_based() {
        let merger = ResultMerger::new(MergeStrategy::DistanceBased);

        let results = vec![
            (
                "inst-1".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "a".to_string(),
                        distance: 0.5,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "b".to_string(),
                        distance: 0.8,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
            (
                "inst-2".to_string(),
                vec![FederatedSearchResult {
                    id: "c".to_string(),
                    distance: 0.3,
                    metadata: None,
                    source_instance: "inst-2".to_string(),
                    collection: "test".to_string(),
                }],
            ),
        ];

        let merged = merger.merge(results, 2);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].id, "c"); // Lowest distance
        assert_eq!(merged[1].id, "a");
    }

    #[test]
    fn test_result_merger_rrf() {
        let merger = ResultMerger::new(MergeStrategy::ReciprocalRankFusion);

        let results = vec![
            (
                "inst-1".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "a".to_string(),
                        distance: 0.1,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "b".to_string(),
                        distance: 0.2,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
            (
                "inst-2".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "a".to_string(),
                        distance: 0.15,
                        metadata: None,
                        source_instance: "inst-2".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "c".to_string(),
                        distance: 0.25,
                        metadata: None,
                        source_instance: "inst-2".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
        ];

        let merged = merger.merge(results, 3);

        // 'a' should be first (appears in both lists)
        assert_eq!(merged[0].id, "a");
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_result_merger_consensus() {
        let merger = ResultMerger::new(MergeStrategy::Consensus);

        let results = vec![
            (
                "inst-1".to_string(),
                vec![
                    FederatedSearchResult {
                        id: "common".to_string(),
                        distance: 0.1,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                    FederatedSearchResult {
                        id: "unique-1".to_string(),
                        distance: 0.05,
                        metadata: None,
                        source_instance: "inst-1".to_string(),
                        collection: "test".to_string(),
                    },
                ],
            ),
            (
                "inst-2".to_string(),
                vec![FederatedSearchResult {
                    id: "common".to_string(),
                    distance: 0.15,
                    metadata: None,
                    source_instance: "inst-2".to_string(),
                    collection: "test".to_string(),
                }],
            ),
            (
                "inst-3".to_string(),
                vec![FederatedSearchResult {
                    id: "common".to_string(),
                    distance: 0.12,
                    metadata: None,
                    source_instance: "inst-3".to_string(),
                    collection: "test".to_string(),
                }],
            ),
        ];

        let merged = merger.merge(results, 2);

        // 'common' should be first (appears in all 3 lists)
        assert_eq!(merged[0].id, "common");
    }

    #[test]
    fn test_federation_health() {
        let federation = create_test_federation();

        let health = federation.health();

        assert_eq!(health.total_instances, 3);
        assert_eq!(health.healthy_instances, 3);
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_federation_health_degraded() {
        let federation = create_test_federation();

        // Mark one instance as unhealthy
        federation
            .registry
            .update_health("eu-west-1", HealthStatus::Unhealthy);

        let health = federation.health();

        assert_eq!(health.healthy_instances, 2);
        assert_eq!(health.unhealthy_instances, 1);
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_routing_score() {
        let mut info = InstanceInfo::new(InstanceConfig::new("test", "http://localhost:8080"));

        info.status = HealthStatus::Healthy;
        info.avg_latency_ms = 100.0;
        info.config.priority = 2;

        let score = info.routing_score(0.7);

        // Score should be reasonable
        assert!(score > 0.0);
        assert!(score < 10.0);
    }

    #[test]
    fn test_instance_success_rate() {
        let mut info = InstanceInfo::new(InstanceConfig::new("test", "http://localhost:8080"));

        info.total_queries = 100;
        info.failed_queries = 10;

        assert!((info.success_rate() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_quorum_routing() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Quorum(2));
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("inst-1", "http://localhost:8081"));
        federation.register_instance(InstanceConfig::new("inst-2", "http://localhost:8082"));
        federation.register_instance(InstanceConfig::new("inst-3", "http://localhost:8083"));

        federation
            .registry
            .update_health("inst-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-2", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-3", HealthStatus::Healthy);

        let instances = federation.select_instances("test_collection").unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_quorum_not_reached() {
        let config = FederationConfig::default().with_routing(RoutingStrategy::Quorum(3));
        let federation = Federation::new(config);

        federation.register_instance(InstanceConfig::new("inst-1", "http://localhost:8081"));
        federation.register_instance(InstanceConfig::new("inst-2", "http://localhost:8082"));

        federation
            .registry
            .update_health("inst-1", HealthStatus::Healthy);
        federation
            .registry
            .update_health("inst-2", HealthStatus::Healthy);

        let result = federation.select_instances("test_collection");
        assert!(matches!(
            result,
            Err(FederationError::QuorumNotReached { .. })
        ));
    }

    #[test]
    fn test_federation_stats() {
        let federation = create_test_federation();

        let query = vec![0.1; 128];
        let _ = federation.search("test_collection", &query, 10);
        let _ = federation.search("test_collection", &query, 10);

        let stats = federation.stats();
        assert_eq!(stats.total_queries, 2);
    }

    #[test]
    fn test_health_check() {
        let federation = create_test_federation();

        let result = federation.check_instance_health("us-west-1");

        assert_eq!(result.instance_id, "us-west-1");
        assert_eq!(result.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_instances_by_region() {
        let federation = create_test_federation();

        let us_west = federation.registry.instances_by_region("us-west");
        let us_east = federation.registry.instances_by_region("us-east");

        assert_eq!(us_west.len(), 1);
        assert_eq!(us_east.len(), 1);
    }

    #[test]
    fn test_round_robin_selection() {
        let registry = InstanceRegistry::new();

        registry.register(InstanceConfig::new("inst-1", "http://localhost:8081"));
        registry.register(InstanceConfig::new("inst-2", "http://localhost:8082"));

        registry.update_health("inst-1", HealthStatus::Healthy);
        registry.update_health("inst-2", HealthStatus::Healthy);

        let first = registry.select_round_robin();
        let second = registry.select_round_robin();

        assert!(first.is_some());
        assert!(second.is_some());
        // Should alternate (though order depends on internal storage)
    }

    #[test]
    fn test_federation_config_builder() {
        let config = FederationConfig::default()
            .with_timeout(Duration::from_secs(5))
            .with_min_instances(2)
            .with_routing(RoutingStrategy::PriorityBased)
            .with_merge(MergeStrategy::Consensus)
            .with_partial_results(false);

        assert_eq!(config.query_timeout, Duration::from_secs(5));
        assert_eq!(config.min_instances, 2);
        assert_eq!(config.routing_strategy, RoutingStrategy::PriorityBased);
        assert_eq!(config.merge_strategy, MergeStrategy::Consensus);
        assert!(!config.allow_partial);
    }

    // ---- Discovery Service tests ----

    #[test]
    fn test_discovery_heartbeat() {
        let registry = Arc::new(InstanceRegistry::new());
        let svc = DiscoveryService::new(registry.clone(), DiscoveryConfig::default());

        svc.heartbeat("inst-1", "http://localhost:8081", HashMap::new());
        assert_eq!(svc.tracked_count(), 1);
        assert!(registry.get("inst-1").is_some());
        assert_eq!(
            registry.get("inst-1").unwrap().status,
            HealthStatus::Healthy
        );
    }

    #[test]
    fn test_discovery_with_region() {
        let registry = Arc::new(InstanceRegistry::new());
        let svc = DiscoveryService::new(registry.clone(), DiscoveryConfig::default());

        let mut meta = HashMap::new();
        meta.insert("region".to_string(), "eu-west".to_string());
        svc.heartbeat("eu-1", "http://eu:8081", meta);

        let eu = registry.instances_by_region("eu-west");
        assert_eq!(eu.len(), 1);
    }

    #[test]
    fn test_discovery_check_heartbeats() {
        let registry = Arc::new(InstanceRegistry::new());
        let config = DiscoveryConfig {
            heartbeat_interval: Duration::from_millis(1),
            missed_heartbeat_threshold: 1,
            auto_remove_stale: false,
        };
        let svc = DiscoveryService::new(registry.clone(), config);

        svc.heartbeat("inst-1", "http://localhost:8081", HashMap::new());
        std::thread::sleep(Duration::from_millis(5));

        let stale = svc.check_heartbeats();
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], "inst-1");
    }

    // ---- Cross-Instance Dedup tests ----

    #[test]
    fn test_dedup_best_distance() {
        let dedup = CrossInstanceDedup::new(DedupStrategy::BestDistance);
        let r1 = vec![
            FederatedSearchResult {
                id: "v1".into(),
                distance: 0.5,
                metadata: None,
                source_instance: "inst-1".into(),
                collection: "test".into(),
            },
            FederatedSearchResult {
                id: "v2".into(),
                distance: 0.3,
                metadata: None,
                source_instance: "inst-1".into(),
                collection: "test".into(),
            },
        ];
        let r2 = vec![FederatedSearchResult {
            id: "v1".into(),
            distance: 0.2,
            metadata: None,
            source_instance: "inst-2".into(),
            collection: "test".into(),
        }];

        let merged = dedup.dedup(&[r1, r2], 10);
        assert_eq!(merged.len(), 2);
        // v1 should have distance 0.2 (best)
        let v1 = merged.iter().find(|r| r.id == "v1").unwrap();
        assert!((v1.distance - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_average_distance() {
        let dedup = CrossInstanceDedup::new(DedupStrategy::AverageDistance);
        let r1 = vec![FederatedSearchResult {
            id: "v1".into(),
            distance: 0.2,
            metadata: None,
            source_instance: "inst-1".into(),
            collection: "test".into(),
        }];
        let r2 = vec![FederatedSearchResult {
            id: "v1".into(),
            distance: 0.4,
            metadata: None,
            source_instance: "inst-2".into(),
            collection: "test".into(),
        }];

        let merged = dedup.dedup(&[r1, r2], 10);
        assert_eq!(merged.len(), 1);
        assert!((merged[0].distance - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_dedup_truncates() {
        let dedup = CrossInstanceDedup::new(DedupStrategy::BestDistance);
        let results: Vec<Vec<FederatedSearchResult>> = (0..20)
            .map(|i| {
                vec![FederatedSearchResult {
                    id: format!("v{}", i),
                    distance: i as f32 * 0.1,
                    metadata: None,
                    source_instance: "inst-1".into(),
                    collection: "test".into(),
                }]
            })
            .collect();

        let merged = dedup.dedup(&results, 5);
        assert_eq!(merged.len(), 5);
    }

    // ---- Query Planner tests ----

    #[test]
    fn test_query_planner_basic() {
        let registry = Arc::new(InstanceRegistry::new());
        registry
            .register(InstanceConfig::new("i1", "http://localhost:8081").with_collection("docs"));
        registry
            .register(InstanceConfig::new("i2", "http://localhost:8082").with_collection("docs"));
        registry.update_health("i1", HealthStatus::Healthy);
        registry.update_health("i2", HealthStatus::Healthy);

        let planner = QueryPlanner::new(registry);
        let plan = planner
            .plan("docs", 10, ConsistencyLevel::One, Duration::from_secs(5))
            .unwrap();
        assert_eq!(plan.target_instances.len(), 1);
        assert_eq!(plan.k, 10);
    }

    #[test]
    fn test_query_planner_quorum() {
        let registry = Arc::new(InstanceRegistry::new());
        for i in 0..5 {
            registry.register(
                InstanceConfig::new(format!("i{}", i), format!("http://localhost:{}", 8080 + i))
                    .with_collection("data"),
            );
            registry.update_health(&format!("i{}", i), HealthStatus::Healthy);
        }

        let planner = QueryPlanner::new(registry);
        let plan = planner
            .plan("data", 10, ConsistencyLevel::Quorum, Duration::from_secs(5))
            .unwrap();
        assert_eq!(plan.target_instances.len(), 3); // 5/2 + 1 = 3
    }

    #[test]
    fn test_query_planner_validate() {
        let registry = Arc::new(InstanceRegistry::new());
        registry
            .register(InstanceConfig::new("i1", "http://localhost:8081").with_collection("docs"));
        registry.update_health("i1", HealthStatus::Healthy);

        let planner = QueryPlanner::new(registry.clone());
        let plan = planner
            .plan("docs", 10, ConsistencyLevel::One, Duration::from_secs(5))
            .unwrap();
        assert!(planner.validate(&plan));

        registry.update_health("i1", HealthStatus::Unhealthy);
        assert!(!planner.validate(&plan));
    }

    // ---- Gossip Protocol tests ----

    #[test]
    fn test_gossip_add_seed() {
        let registry = Arc::new(InstanceRegistry::new());
        let gossip = GossipProtocol::new("local", "http://localhost:8080", registry.clone(), GossipConfig::default());
        gossip.add_seed("peer1", "http://peer1:8080");

        assert_eq!(gossip.alive_count(), 1);
        assert!(registry.get("peer1").is_some());
    }

    #[test]
    fn test_gossip_create_ping() -> Result<(), Box<dyn std::error::Error>> {
        let registry = Arc::new(InstanceRegistry::new());
        let gossip = GossipProtocol::new("local", "http://localhost:8080", registry, GossipConfig::default());
        gossip.add_seed("peer1", "http://peer1:8080");

        let msg = gossip.create_ping();
        match msg {
            GossipMessage::Ping { sender, members } => {
                assert_eq!(sender, "local");
                assert!(members.len() >= 1); // at least peer1 + self
            }
            _ => return Err("Expected Ping".into()),
        }

        Ok(())
    }

    #[test]
    fn test_gossip_handle_ping() {
        let reg1 = Arc::new(InstanceRegistry::new());
        let gossip1 = GossipProtocol::new("node1", "http://n1:8080", reg1.clone(), GossipConfig::default());

        let reg2 = Arc::new(InstanceRegistry::new());
        let gossip2 = GossipProtocol::new("node2", "http://n2:8080", reg2.clone(), GossipConfig::default());

        // node1 creates a ping
        gossip1.add_seed("node3", "http://n3:8080");
        let ping = gossip1.create_ping();

        // node2 handles the ping — should learn about node1 and node3
        let ack = gossip2.handle_message(ping);
        assert!(ack.is_some()); // Should return Ack

        let peers = gossip2.peers();
        let peer_ids: Vec<&str> = peers.iter().map(|p| p.id.as_str()).collect();
        assert!(peer_ids.contains(&"node1"));
        assert!(peer_ids.contains(&"node3"));
    }

    #[test]
    fn test_gossip_transitive_discovery() {
        let reg_a = Arc::new(InstanceRegistry::new());
        let gossip_a = GossipProtocol::new("A", "http://a:8080", reg_a, GossipConfig::default());

        let reg_b = Arc::new(InstanceRegistry::new());
        let gossip_b = GossipProtocol::new("B", "http://b:8080", reg_b, GossipConfig::default());

        let reg_c = Arc::new(InstanceRegistry::new());
        let gossip_c = GossipProtocol::new("C", "http://c:8080", reg_c, GossipConfig::default());

        // A knows B
        gossip_a.add_seed("B", "http://b:8080");
        // B knows C
        gossip_b.add_seed("C", "http://c:8080");

        // A pings B → B learns about A
        let ping_a = gossip_a.create_ping();
        let ack_b = gossip_b.handle_message(ping_a).unwrap();

        // B acks with its full membership (includes C) → A learns about C
        gossip_a.handle_message(ack_b);

        // A should now know about C (transitive discovery)
        let a_peers = gossip_a.peers();
        let a_ids: Vec<&str> = a_peers.iter().map(|p| p.id.as_str()).collect();
        assert!(a_ids.contains(&"B"));
        assert!(a_ids.contains(&"C"));
    }

    #[test]
    fn test_gossip_leave() {
        let registry = Arc::new(InstanceRegistry::new());
        let gossip = GossipProtocol::new("local", "http://localhost:8080", registry.clone(), GossipConfig::default());
        gossip.add_seed("peer1", "http://peer1:8080");
        assert_eq!(gossip.alive_count(), 1);

        let leave = GossipMessage::Leave { sender: "peer1".into() };
        gossip.handle_message(leave);

        let peers = gossip.peers();
        let peer1 = peers.iter().find(|p| p.id == "peer1").unwrap();
        assert_eq!(peer1.state, PeerState::Left);
        assert!(registry.get("peer1").is_none());
    }

    #[test]
    fn test_gossip_tick_suspect() {
        let config = GossipConfig {
            suspect_timeout: Duration::from_millis(1), // Immediate for testing
            dead_timeout: Duration::from_secs(30),
            ..GossipConfig::default()
        };
        let registry = Arc::new(InstanceRegistry::new());
        let gossip = GossipProtocol::new("local", "http://localhost:8080", registry, config);
        gossip.add_seed("peer1", "http://peer1:8080");

        // Wait for suspect timeout
        std::thread::sleep(Duration::from_millis(10));

        let result = gossip.tick();
        assert!(result.newly_suspect.contains(&"peer1".to_string()));
        assert_eq!(result.newly_dead.len(), 0); // Not dead yet
    }

    #[test]
    fn test_gossip_tick_dead() {
        let config = GossipConfig {
            suspect_timeout: Duration::from_millis(1),
            dead_timeout: Duration::from_millis(2),
            ..GossipConfig::default()
        };
        let registry = Arc::new(InstanceRegistry::new());
        let gossip = GossipProtocol::new("local", "http://localhost:8080", registry.clone(), config);
        gossip.add_seed("peer1", "http://peer1:8080");

        std::thread::sleep(Duration::from_millis(5));
        gossip.tick(); // → suspect
        std::thread::sleep(Duration::from_millis(5));
        let result = gossip.tick(); // → dead

        assert!(result.newly_dead.contains(&"peer1".to_string()));
        assert!(registry.get("peer1").is_none());
    }

    #[test]
    fn test_gossip_select_targets() {
        let registry = Arc::new(InstanceRegistry::new());
        let gossip = GossipProtocol::new("local", "http://localhost:8080", registry, GossipConfig {
            fanout: 2,
            ..GossipConfig::default()
        });

        for i in 0..5 {
            gossip.add_seed(format!("peer{i}"), format!("http://peer{i}:8080"));
        }

        let targets = gossip.select_gossip_targets();
        assert_eq!(targets.len(), 2);
    }

    #[test]
    fn test_gossip_config_serde() {
        let config = GossipConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: GossipConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.fanout, 3);
    }

    #[test]
    fn test_peer_state_serde() {
        let states = vec![PeerState::Alive, PeerState::Suspect, PeerState::Dead, PeerState::Left];
        for s in states {
            let json = serde_json::to_string(&s).unwrap();
            let deser: PeerState = serde_json::from_str(&json).unwrap();
            assert_eq!(s, deser);
        }
    }

    // ---- Hash Ring tests ----

    #[test]
    fn test_hash_ring_basic() {
        let mut ring = HashRing::new(1);
        ring.add_instance("node-a");
        ring.add_instance("node-b");
        ring.add_instance("node-c");

        assert_eq!(ring.instance_count(), 3);

        // Every key should map to some instance
        for i in 0..100 {
            let key = format!("key-{i}");
            assert!(ring.get_instance(&key).is_some());
        }
    }

    #[test]
    fn test_hash_ring_replication() {
        let mut ring = HashRing::new(3);
        ring.add_instance("node-a");
        ring.add_instance("node-b");
        ring.add_instance("node-c");

        let replicas = ring.get_instances("test-key");
        assert_eq!(replicas.len(), 3);
        // All should be distinct
        let unique: std::collections::HashSet<&str> = replicas.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_hash_ring_remove_instance() {
        let mut ring = HashRing::new(1);
        ring.add_instance("node-a");
        ring.add_instance("node-b");
        assert_eq!(ring.instance_count(), 2);

        ring.remove_instance("node-a");
        assert_eq!(ring.instance_count(), 1);

        // All keys should now map to node-b
        for i in 0..10 {
            assert_eq!(ring.get_instance(&format!("k{i}")), Some("node-b"));
        }
    }

    #[test]
    fn test_hash_ring_balance() {
        let mut ring = HashRing::new(1);
        ring.add_instance("a");
        ring.add_instance("b");
        ring.add_instance("c");

        let keys: Vec<String> = (0..300).map(|i| format!("key-{i}")).collect();
        let key_refs: Vec<&str> = keys.iter().map(|s| s.as_str()).collect();
        let (min, max) = ring.check_balance(&key_refs);
        // With 128 vnodes and 300 keys, distribution should be reasonable
        assert!(min > 0);
        assert!(max < 200); // no single node gets more than 200/300
    }

    // ---- Shard Manager tests ----

    #[test]
    fn test_shard_manager_assign() {
        let mut sm = ShardManager::new(2, 2.0);
        sm.add_instance("node-1");
        sm.add_instance("node-2");
        sm.add_instance("node-3");

        let assigned = sm.assign_shard("collection-a");
        assert_eq!(assigned.len(), 2); // replica_factor=2
    }

    #[test]
    fn test_shard_manager_rebalance_on_add() {
        let mut sm = ShardManager::new(1, 2.0);
        sm.add_instance("node-1");
        sm.assign_shard("shard-0");
        sm.assign_shard("shard-1");
        sm.assign_shard("shard-2");

        // All shards on node-1
        let result = sm.add_instance("node-2");
        // Some shards should have moved
        assert!(result.total_shards >= 3);
    }

    #[test]
    fn test_shard_manager_load_distribution() {
        let mut sm = ShardManager::new(1, 2.0);
        sm.add_instance("n1");
        sm.add_instance("n2");
        for i in 0..10 {
            sm.assign_shard(&format!("s{i}"));
        }
        let dist = sm.load_distribution();
        assert!(!dist.is_empty());
        let total: usize = dist.values().sum();
        assert_eq!(total, 10);
    }
}
