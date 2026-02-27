use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::config::{FederationConfig, RoutingStrategy};
use super::health::{FederationHealth, HealthCheckResult, HealthMonitor};
use super::instance::{HealthStatus, InstanceConfig, InstanceInfo, InstanceRegistry};
use super::merger::{FederatedSearchResponse, FederatedSearchResult, ResultMerger};
use super::FederationError;

/// Main federation coordinator
pub struct Federation {
    config: FederationConfig,
    pub(crate) registry: Arc<InstanceRegistry>,
    merger: ResultMerger,
    monitor: HealthMonitor,
    stats: FederationStats,
}

impl Federation {
    /// Create a new federation
    pub fn new(config: FederationConfig) -> Self {
        let registry = Arc::new(InstanceRegistry::new());
        let merger = ResultMerger::new(config.merge_strategy);
        let monitor = HealthMonitor::new(registry.clone(), config.clone());

        Self {
            config,
            registry,
            merger,
            monitor,
            stats: FederationStats::default(),
        }
    }

    /// Register an instance
    pub fn register_instance(&self, config: InstanceConfig) {
        self.registry.register(config);
    }

    /// Unregister an instance
    pub fn unregister_instance(&self, id: &str) -> Option<InstanceInfo> {
        self.registry.unregister(id)
    }

    /// Get instance info
    pub fn get_instance(&self, id: &str) -> Option<InstanceInfo> {
        self.registry.get(id)
    }

    /// List all instances
    pub fn list_instances(&self) -> Vec<InstanceInfo> {
        self.registry.list()
    }

    /// Select instances for a query based on routing strategy
    pub fn select_instances(&self, collection: &str) -> super::FederationResult<Vec<String>> {
        let candidates = self.registry.instances_with_collection(collection);

        if candidates.is_empty() {
            return Err(FederationError::NoHealthyInstances);
        }

        let healthy: Vec<_> = candidates
            .into_iter()
            .filter(|i| i.status == HealthStatus::Healthy || i.status == HealthStatus::Degraded)
            .collect();

        if healthy.is_empty() {
            return Err(FederationError::NoHealthyInstances);
        }

        match self.config.routing_strategy {
            RoutingStrategy::Broadcast => Ok(healthy.iter().map(|i| i.config.id.clone()).collect()),

            RoutingStrategy::LatencyAware => {
                let mut sorted = healthy;
                sorted.sort_by(|a, b| {
                    a.routing_score(self.config.latency_weight)
                        .partial_cmp(&b.routing_score(self.config.latency_weight))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(sorted.iter().map(|i| i.config.id.clone()).collect())
            }

            RoutingStrategy::GeographicProximity => {
                // In real implementation, would sort by geographic distance
                Ok(healthy.iter().map(|i| i.config.id.clone()).collect())
            }

            RoutingStrategy::RoundRobin => self
                .registry
                .select_round_robin()
                .map(|id| vec![id])
                .ok_or(FederationError::NoHealthyInstances),

            RoutingStrategy::Random => {
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let mut ids: Vec<_> = healthy.iter().map(|i| i.config.id.clone()).collect();
                ids.shuffle(&mut rng);
                Ok(ids)
            }

            RoutingStrategy::PriorityBased => {
                let mut sorted = healthy;
                sorted.sort_by(|a, b| b.config.priority.cmp(&a.config.priority));
                Ok(sorted.iter().map(|i| i.config.id.clone()).collect())
            }

            RoutingStrategy::Quorum(n) => {
                if healthy.len() < n {
                    return Err(FederationError::QuorumNotReached {
                        required: n,
                        available: healthy.len(),
                    });
                }
                Ok(healthy
                    .iter()
                    .take(n)
                    .map(|i| i.config.id.clone())
                    .collect())
            }
        }
    }

    /// Execute a federated search (simulated - returns mock results)
    pub fn search(
        &self,
        collection: &str,
        _query: &[f32],
        k: usize,
    ) -> super::FederationResult<FederatedSearchResponse> {
        let start = Instant::now();

        // Select instances
        let instances = self.select_instances(collection)?;

        if instances.is_empty() {
            return Err(FederationError::NoHealthyInstances);
        }

        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        // Simulate parallel queries to instances
        let mut instance_results: Vec<(String, Vec<FederatedSearchResult>)> = Vec::new();
        let mut instance_latencies: HashMap<String, f64> = HashMap::new();
        let mut instances_failed: Vec<String> = Vec::new();

        for instance_id in &instances {
            let query_start = Instant::now();

            // Simulate query - in real implementation, this would make HTTP calls
            let info = self.registry.get(instance_id);

            if let Some(info) = info {
                if info.status == HealthStatus::Unhealthy {
                    instances_failed.push(instance_id.clone());
                    self.registry.record_query(instance_id, 0.0, false);
                    continue;
                }

                // Generate mock results
                let results: Vec<FederatedSearchResult> = (0..k.min(5))
                    .map(|i| FederatedSearchResult {
                        id: format!("{}_{}", instance_id, i),
                        distance: 0.1 + (i as f32 * 0.1),
                        metadata: None,
                        source_instance: instance_id.clone(),
                        collection: collection.to_string(),
                    })
                    .collect();

                let latency = query_start.elapsed().as_secs_f64() * 1000.0;
                instance_latencies.insert(instance_id.clone(), latency);
                self.registry.record_query(instance_id, latency, true);

                instance_results.push((instance_id.clone(), results));
            } else {
                instances_failed.push(instance_id.clone());
            }
        }

        // Check minimum instances
        if instance_results.len() < self.config.min_instances && !self.config.allow_partial {
            self.stats.failed_queries.fetch_add(1, Ordering::Relaxed);
            return Err(FederationError::PartialResults {
                success: instance_results.len(),
                total: instances.len(),
            });
        }

        // Merge results
        let total_found: usize = instance_results.iter().map(|(_, r)| r.len()).sum();
        let instances_responded: Vec<_> =
            instance_results.iter().map(|(id, _)| id.clone()).collect();
        let merged = self.merger.merge(instance_results, k);

        let execution_time = start.elapsed();

        let is_partial = !instances_failed.is_empty();
        if is_partial {
            self.stats.partial_results.fetch_add(1, Ordering::Relaxed);
        }

        Ok(FederatedSearchResponse {
            results: merged,
            total_found,
            instances_responded,
            instances_failed,
            execution_time_ms: execution_time.as_secs_f64() * 1000.0,
            instance_latencies,
            is_partial,
        })
    }

    /// Get federation health
    pub fn health(&self) -> FederationHealth {
        self.monitor.federation_health()
    }

    /// Get statistics
    pub fn stats(&self) -> FederationStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get configuration
    pub fn config(&self) -> &FederationConfig {
        &self.config
    }

    /// Perform health check on an instance
    pub fn check_instance_health(&self, id: &str) -> HealthCheckResult {
        let result = self.monitor.check_instance(id);
        self.monitor.record_check(result.clone());
        result
    }
}

/// Federation statistics
#[derive(Debug, Default)]
pub struct FederationStats {
    total_queries: AtomicU64,
    failed_queries: AtomicU64,
    partial_results: AtomicU64,
    timeouts: AtomicU64,
}

impl FederationStats {
    fn snapshot(&self) -> FederationStatsSnapshot {
        FederationStatsSnapshot {
            total_queries: self.total_queries.load(Ordering::Relaxed),
            failed_queries: self.failed_queries.load(Ordering::Relaxed),
            partial_results: self.partial_results.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of federation stats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStatsSnapshot {
    pub total_queries: u64,
    pub failed_queries: u64,
    pub partial_results: u64,
    pub timeouts: u64,
}

#[cfg(test)]
mod tests {}
