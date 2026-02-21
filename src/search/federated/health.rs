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
