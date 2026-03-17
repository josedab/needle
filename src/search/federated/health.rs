use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::config::FederationConfig;
use super::instance::{HealthStatus, InstanceRegistry};

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub instance_id: String,
    pub status: HealthStatus,
    pub latency_ms: f64,
    pub error: Option<String>,
    pub timestamp: u64,
}

/// Health monitor for tracking instance health
pub struct HealthMonitor {
    registry: Arc<InstanceRegistry>,
    config: FederationConfig,
    check_results: RwLock<HashMap<String, Vec<HealthCheckResult>>>,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(registry: Arc<InstanceRegistry>, config: FederationConfig) -> Self {
        Self {
            registry,
            config,
            check_results: RwLock::new(HashMap::new()),
        }
    }

    /// Record a health check result
    pub fn record_check(&self, result: HealthCheckResult) {
        let instance_id = result.instance_id.clone();

        // Update registry
        self.registry.update_health(&instance_id, result.status);

        // Store result history
        let mut results = self.check_results.write();
        let history = results.entry(instance_id).or_default();
        history.push(result);

        // Keep only recent history (last 100 checks)
        if history.len() > 100 {
            history.remove(0);
        }
    }

    /// Perform a simulated health check (in real impl, this would make HTTP calls)
    pub fn check_instance(&self, instance_id: &str) -> HealthCheckResult {
        let start = Instant::now();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Simulate health check - in real implementation this would ping the instance
        let info = self.registry.get(instance_id);

        match info {
            Some(info) => {
                let latency = start.elapsed().as_secs_f64() * 1000.0;

                // Determine health based on consecutive failures
                let status = if info.consecutive_failures >= self.config.unhealthy_threshold {
                    HealthStatus::Unhealthy
                } else if info.consecutive_failures > 0 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };

                HealthCheckResult {
                    instance_id: instance_id.to_string(),
                    status,
                    latency_ms: latency,
                    error: None,
                    timestamp,
                }
            }
            None => HealthCheckResult {
                instance_id: instance_id.to_string(),
                status: HealthStatus::Unknown,
                latency_ms: 0.0,
                error: Some("Instance not found".to_string()),
                timestamp,
            },
        }
    }

    /// Get health check history for an instance
    pub fn get_history(&self, instance_id: &str) -> Vec<HealthCheckResult> {
        self.check_results
            .read()
            .get(instance_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get overall federation health status
    pub fn federation_health(&self) -> FederationHealth {
        let instances = self.registry.list();
        let total = instances.len();
        let healthy = instances
            .iter()
            .filter(|i| i.status == HealthStatus::Healthy)
            .count();
        let degraded = instances
            .iter()
            .filter(|i| i.status == HealthStatus::Degraded)
            .count();
        let unhealthy = instances
            .iter()
            .filter(|i| i.status == HealthStatus::Unhealthy)
            .count();

        let status = if healthy == total {
            HealthStatus::Healthy
        } else if healthy + degraded >= self.config.min_instances {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };

        FederationHealth {
            status,
            total_instances: total,
            healthy_instances: healthy,
            degraded_instances: degraded,
            unhealthy_instances: unhealthy,
            avg_latency_ms: instances.iter().map(|i| i.avg_latency_ms).sum::<f64>()
                / total.max(1) as f64,
        }
    }
}

/// Overall federation health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationHealth {
    pub status: HealthStatus,
    pub total_instances: usize,
    pub healthy_instances: usize,
    pub degraded_instances: usize,
    pub unhealthy_instances: usize,
    pub avg_latency_ms: f64,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use super::super::instance::InstanceConfig;

    fn make_registry_with_instances(
        configs: Vec<(InstanceConfig, HealthStatus)>,
    ) -> Arc<InstanceRegistry> {
        let registry = Arc::new(InstanceRegistry::new());
        for (config, status) in configs {
            let id = config.id.clone();
            registry.register(config);
            registry.update_health(&id, status);
        }
        registry
    }

    #[test]
    fn test_federation_health_all_healthy() {
        let registry = make_registry_with_instances(vec![
            (
                InstanceConfig::new("i1", "http://a:8080"),
                HealthStatus::Healthy,
            ),
            (
                InstanceConfig::new("i2", "http://b:8080"),
                HealthStatus::Healthy,
            ),
        ]);

        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let health = monitor.federation_health();

        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.total_instances, 2);
        assert_eq!(health.healthy_instances, 2);
        assert_eq!(health.degraded_instances, 0);
        assert_eq!(health.unhealthy_instances, 0);
    }

    #[test]
    fn test_federation_health_mixed_status() {
        let registry = make_registry_with_instances(vec![
            (
                InstanceConfig::new("i1", "http://a:8080"),
                HealthStatus::Healthy,
            ),
            (
                InstanceConfig::new("i2", "http://b:8080"),
                HealthStatus::Degraded,
            ),
            (
                InstanceConfig::new("i3", "http://c:8080"),
                HealthStatus::Unhealthy,
            ),
        ]);

        let config = FederationConfig::default().with_min_instances(1);
        let monitor = HealthMonitor::new(registry, config);
        let health = monitor.federation_health();

        // 1 healthy + 1 degraded >= min_instances(1), so Degraded overall
        assert_eq!(health.status, HealthStatus::Degraded);
        assert_eq!(health.healthy_instances, 1);
        assert_eq!(health.degraded_instances, 1);
        assert_eq!(health.unhealthy_instances, 1);
    }

    #[test]
    fn test_federation_health_all_unhealthy() {
        let registry = make_registry_with_instances(vec![
            (
                InstanceConfig::new("i1", "http://a:8080"),
                HealthStatus::Unhealthy,
            ),
            (
                InstanceConfig::new("i2", "http://b:8080"),
                HealthStatus::Unhealthy,
            ),
        ]);

        let config = FederationConfig::default().with_min_instances(1);
        let monitor = HealthMonitor::new(registry, config);
        let health = monitor.federation_health();

        assert_eq!(health.status, HealthStatus::Unhealthy);
        assert_eq!(health.healthy_instances, 0);
        assert_eq!(health.unhealthy_instances, 2);
    }

    #[test]
    fn test_federation_health_empty_registry() {
        let registry = Arc::new(InstanceRegistry::new());
        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let health = monitor.federation_health();

        assert_eq!(health.total_instances, 0);
        assert_eq!(health.healthy_instances, 0);
        // 0 healthy == 0 total → Healthy status (all match)
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_record_check_stores_history() {
        let registry = Arc::new(InstanceRegistry::new());
        registry.register(InstanceConfig::new("i1", "http://a:8080"));

        let monitor = HealthMonitor::new(registry, FederationConfig::default());

        monitor.record_check(HealthCheckResult {
            instance_id: "i1".to_string(),
            status: HealthStatus::Healthy,
            latency_ms: 5.0,
            error: None,
            timestamp: 1000,
        });

        monitor.record_check(HealthCheckResult {
            instance_id: "i1".to_string(),
            status: HealthStatus::Degraded,
            latency_ms: 50.0,
            error: None,
            timestamp: 2000,
        });

        let history = monitor.get_history("i1");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].status, HealthStatus::Healthy);
        assert_eq!(history[1].status, HealthStatus::Degraded);
    }

    #[test]
    fn test_history_capped_at_100() {
        let registry = Arc::new(InstanceRegistry::new());
        registry.register(InstanceConfig::new("i1", "http://a:8080"));

        let monitor = HealthMonitor::new(registry, FederationConfig::default());

        for i in 0..110 {
            monitor.record_check(HealthCheckResult {
                instance_id: "i1".to_string(),
                status: HealthStatus::Healthy,
                latency_ms: i as f64,
                error: None,
                timestamp: i as u64,
            });
        }

        let history = monitor.get_history("i1");
        assert_eq!(history.len(), 100);
        // First entry should have been evicted (started at 0, now starts at 10)
        assert!((history[0].latency_ms - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_history_unknown_instance() {
        let registry = Arc::new(InstanceRegistry::new());
        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let history = monitor.get_history("nonexistent");
        assert!(history.is_empty());
    }

    #[test]
    fn test_check_instance_not_found() {
        let registry = Arc::new(InstanceRegistry::new());
        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let result = monitor.check_instance("nonexistent");

        assert_eq!(result.status, HealthStatus::Unknown);
        assert!(result.error.is_some());
        assert_eq!(result.error.as_deref(), Some("Instance not found"));
    }

    #[test]
    fn test_check_instance_healthy() {
        let registry = Arc::new(InstanceRegistry::new());
        registry.register(InstanceConfig::new("i1", "http://a:8080"));
        // Reset consecutive_failures to 0 by marking healthy
        registry.update_health("i1", HealthStatus::Healthy);

        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let result = monitor.check_instance("i1");

        assert_eq!(result.status, HealthStatus::Healthy);
        assert!(result.error.is_none());
        assert!(result.timestamp > 0);
    }

    #[test]
    fn test_check_instance_unhealthy_threshold() {
        let registry = Arc::new(InstanceRegistry::new());
        registry.register(InstanceConfig::new("i1", "http://a:8080"));
        // Simulate consecutive failures reaching threshold (default 3)
        registry.update_health("i1", HealthStatus::Unhealthy);
        registry.update_health("i1", HealthStatus::Unhealthy);
        registry.update_health("i1", HealthStatus::Unhealthy);

        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let result = monitor.check_instance("i1");

        assert_eq!(result.status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_avg_latency_calculation() {
        let registry = make_registry_with_instances(vec![
            (
                InstanceConfig::new("i1", "http://a:8080"),
                HealthStatus::Healthy,
            ),
            (
                InstanceConfig::new("i2", "http://b:8080"),
                HealthStatus::Healthy,
            ),
        ]);
        // Record latency to set avg_latency_ms
        registry.record_query("i1", 10.0, true);
        registry.record_query("i2", 20.0, true);

        let monitor = HealthMonitor::new(registry, FederationConfig::default());
        let health = monitor.federation_health();

        // avg_latency_ms should be the average of all instances' avg_latency_ms
        assert!(health.avg_latency_ms > 0.0);
    }
}
