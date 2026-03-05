use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};


/// Configuration for a remote instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceConfig {
    /// Unique instance identifier
    pub id: String,
    /// HTTP endpoint
    pub endpoint: String,
    /// Geographic region
    pub region: String,
    /// Priority (higher = preferred)
    pub priority: u32,
    /// Collections available on this instance
    pub collections: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl InstanceConfig {
    /// Create a new instance config
    pub fn new(id: impl Into<String>, endpoint: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            endpoint: endpoint.into(),
            region: "default".to_string(),
            priority: 1,
            collections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set region
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Add a collection
    pub fn with_collection(mut self, collection: impl Into<String>) -> Self {
        self.collections.push(collection.into());
        self
    }
}

/// Health status of an instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Instance is healthy and responding
    Healthy,
    /// Instance has degraded performance
    Degraded,
    /// Instance is unhealthy (not responding)
    Unhealthy,
    /// Instance status is unknown
    Unknown,
}

/// Detailed instance information
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    /// Configuration
    pub config: InstanceConfig,
    /// Current health status
    pub status: HealthStatus,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Total queries processed
    pub total_queries: u64,
    /// Failed queries
    pub failed_queries: u64,
    /// Last successful health check
    pub last_healthy: Option<u64>,
    /// Consecutive failures
    pub consecutive_failures: usize,
    /// Consecutive successes
    pub consecutive_successes: usize,
}

impl InstanceInfo {
    pub(crate) fn new(config: InstanceConfig) -> Self {
        Self {
            config,
            status: HealthStatus::Unknown,
            avg_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            total_queries: 0,
            failed_queries: 0,
            last_healthy: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
        }
    }

    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_queries == 0 {
            1.0
        } else {
            1.0 - (self.failed_queries as f64 / self.total_queries as f64)
        }
    }

    /// Calculate routing score (lower is better)
    /// Compute routing score (lower is better for routing priority).
    /// Returns a value >= 0.01 to ensure positive scores.
    pub fn routing_score(&self, latency_weight: f32) -> f64 {
        let latency_score = self.avg_latency_ms / 1000.0; // Normalize to seconds
        let health_score = match self.status {
            HealthStatus::Healthy => 0.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Unhealthy => 10.0,
            HealthStatus::Unknown => 1.0,
        };
        let failure_score = self.consecutive_failures as f64 * 0.1;
        // Priority multiplier: higher priority = lower score (better)
        let priority_multiplier = 1.0 / (self.config.priority as f64 + 1.0);

        let base_score = latency_weight as f64 * latency_score
            + (1.0 - latency_weight as f64) * health_score
            + failure_score
            + 0.1; // Base offset to ensure positive scores

        // Apply priority as a multiplier (higher priority reduces score)
        (base_score * priority_multiplier).max(0.01)
    }
}

/// Instance registry for tracking remote instances
pub struct InstanceRegistry {
    instances: RwLock<HashMap<String, InstanceInfo>>,
    regions: RwLock<HashMap<String, Vec<String>>>,
    round_robin_counter: AtomicUsize,
}

impl Default for InstanceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl InstanceRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            instances: RwLock::new(HashMap::new()),
            regions: RwLock::new(HashMap::new()),
            round_robin_counter: AtomicUsize::new(0),
        }
    }

    /// Register an instance
    pub fn register(&self, config: InstanceConfig) {
        let region = config.region.clone();
        let id = config.id.clone();

        let mut instances = self.instances.write();
        instances.insert(id.clone(), InstanceInfo::new(config));

        let mut regions = self.regions.write();
        regions.entry(region).or_default().push(id);
    }

    /// Unregister an instance
    pub fn unregister(&self, id: &str) -> Option<InstanceInfo> {
        let mut instances = self.instances.write();
        let info = instances.remove(id)?;

        let mut regions = self.regions.write();
        if let Some(region_instances) = regions.get_mut(&info.config.region) {
            region_instances.retain(|i| i != id);
        }

        Some(info)
    }

    /// Get instance info
    pub fn get(&self, id: &str) -> Option<InstanceInfo> {
        self.instances.read().get(id).cloned()
    }

    /// Get all instances
    pub fn list(&self) -> Vec<InstanceInfo> {
        self.instances.read().values().cloned().collect()
    }

    /// Get healthy instances
    pub fn healthy_instances(&self) -> Vec<InstanceInfo> {
        self.instances
            .read()
            .values()
            .filter(|i| i.status == HealthStatus::Healthy || i.status == HealthStatus::Degraded)
            .cloned()
            .collect()
    }

    /// Get instances by region
    pub fn instances_by_region(&self, region: &str) -> Vec<InstanceInfo> {
        let regions = self.regions.read();
        let instances = self.instances.read();

        regions
            .get(region)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| instances.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get instances with a specific collection
    pub fn instances_with_collection(&self, collection: &str) -> Vec<InstanceInfo> {
        self.instances
            .read()
            .values()
            .filter(|i| {
                i.config.collections.is_empty()
                    || i.config.collections.contains(&collection.to_string())
            })
            .cloned()
            .collect()
    }

    /// Update instance health
    pub fn update_health(&self, id: &str, status: HealthStatus) {
        let mut instances = self.instances.write();
        if let Some(info) = instances.get_mut(id) {
            info.status = status;
            match status {
                HealthStatus::Healthy => {
                    info.consecutive_successes += 1;
                    info.consecutive_failures = 0;
                    info.last_healthy = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                    );
                }
                HealthStatus::Unhealthy => {
                    info.consecutive_failures += 1;
                    info.consecutive_successes = 0;
                }
                _ => {}
            }
        }
    }

    /// Record query result
    pub fn record_query(&self, id: &str, latency_ms: f64, success: bool) {
        let mut instances = self.instances.write();
        if let Some(info) = instances.get_mut(id) {
            info.total_queries += 1;

            if success {
                // Update average latency with exponential moving average
                let alpha = 0.1;
                info.avg_latency_ms = alpha * latency_ms + (1.0 - alpha) * info.avg_latency_ms;

                // Update P99 (simplified - just track max recent)
                if latency_ms > info.p99_latency_ms {
                    info.p99_latency_ms = latency_ms;
                } else {
                    info.p99_latency_ms = 0.99 * info.p99_latency_ms + 0.01 * latency_ms;
                }

                info.consecutive_successes += 1;
                info.consecutive_failures = 0;
            } else {
                info.failed_queries += 1;
                info.consecutive_failures += 1;
                info.consecutive_successes = 0;
            }
        }
    }

    /// Select instance using round-robin
    pub fn select_round_robin(&self) -> Option<String> {
        let instances = self.instances.read();
        let healthy: Vec<_> = instances
            .iter()
            .filter(|(_, i)| i.status == HealthStatus::Healthy)
            .map(|(id, _)| id.clone())
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % healthy.len();
        Some(healthy[idx].clone())
    }

    /// Select instance by lowest latency
    pub fn select_by_latency(&self, latency_weight: f32) -> Option<String> {
        let instances = self.instances.read();
        instances
            .iter()
            .filter(|(_, i)| {
                i.status == HealthStatus::Healthy || i.status == HealthStatus::Degraded
            })
            .min_by(|(_, a), (_, b)| {
                a.routing_score(latency_weight)
                    .partial_cmp(&b.routing_score(latency_weight))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, _)| id.clone())
    }

    /// Get instance count
    pub fn len(&self) -> usize {
        self.instances.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.instances.read().is_empty()
    }
}

#[cfg(test)]
mod tests {}
