#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::control_plane::current_timestamp;

/// Policy for automatic instance recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRecoveryConfig {
    /// Maximum number of automatic restarts before escalating.
    pub max_restarts: u32,
    /// Cooldown between restarts in seconds.
    pub restart_cooldown_secs: u64,
    /// Health check interval in seconds.
    pub health_check_interval_secs: u64,
    /// Number of consecutive failures before restarting.
    pub failure_threshold: u32,
}

impl Default for AutoRecoveryConfig {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            restart_cooldown_secs: 30,
            health_check_interval_secs: 10,
            failure_threshold: 3,
        }
    }
}

/// Tracks instance health and manages automatic recovery.
pub struct InstanceRecoveryManager {
    config: AutoRecoveryConfig,
    instances: RwLock<HashMap<String, InstanceHealth>>,
}

/// Health state tracked per instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceHealth {
    pub instance_id: String,
    pub consecutive_failures: u32,
    pub total_restarts: u32,
    pub last_healthy_at: u64,
    pub last_restart_at: Option<u64>,
    pub status: RecoveryStatus,
}

/// Recovery status of an instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStatus {
    Healthy,
    Degraded,
    Restarting,
    Exhausted,
}

impl InstanceRecoveryManager {
    pub fn new(config: AutoRecoveryConfig) -> Self {
        Self {
            config,
            instances: RwLock::new(HashMap::new()),
        }
    }

    /// Register an instance for health tracking.
    pub fn register(&self, instance_id: &str) {
        let mut instances = self.instances.write();
        instances.insert(
            instance_id.to_string(),
            InstanceHealth {
                instance_id: instance_id.to_string(),
                consecutive_failures: 0,
                total_restarts: 0,
                last_healthy_at: current_timestamp(),
                last_restart_at: None,
                status: RecoveryStatus::Healthy,
            },
        );
    }

    /// Record a successful health check.
    pub fn record_healthy(&self, instance_id: &str) {
        let mut instances = self.instances.write();
        if let Some(health) = instances.get_mut(instance_id) {
            health.consecutive_failures = 0;
            health.last_healthy_at = current_timestamp();
            health.status = RecoveryStatus::Healthy;
        }
    }

    /// Record a failed health check. Returns recovery action if threshold exceeded.
    pub fn record_failure(&self, instance_id: &str) -> Option<RecoveryAction> {
        let mut instances = self.instances.write();
        let health = instances.get_mut(instance_id)?;

        health.consecutive_failures += 1;

        if health.consecutive_failures < self.config.failure_threshold {
            health.status = RecoveryStatus::Degraded;
            return None;
        }

        if health.total_restarts >= self.config.max_restarts {
            health.status = RecoveryStatus::Exhausted;
            return Some(RecoveryAction::Escalate {
                instance_id: instance_id.to_string(),
                reason: format!("Exhausted {} restart attempts", self.config.max_restarts),
            });
        }

        // Check cooldown
        if let Some(last_restart) = health.last_restart_at {
            let elapsed = current_timestamp() - last_restart;
            if elapsed < self.config.restart_cooldown_secs {
                return None;
            }
        }

        health.total_restarts += 1;
        health.consecutive_failures = 0;
        health.last_restart_at = Some(current_timestamp());
        health.status = RecoveryStatus::Restarting;

        Some(RecoveryAction::Restart {
            instance_id: instance_id.to_string(),
            attempt: health.total_restarts,
        })
    }

    /// Get the health status of an instance.
    pub fn get_health(&self, instance_id: &str) -> Option<InstanceHealth> {
        self.instances.read().get(instance_id).cloned()
    }

    /// List all instances and their health.
    pub fn all_instances(&self) -> Vec<InstanceHealth> {
        self.instances.read().values().cloned().collect()
    }
}

/// Action to take when an instance needs recovery.
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Restart { instance_id: String, attempt: u32 },
    Escalate { instance_id: String, reason: String },
}
