#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use super::control_plane::current_timestamp;

/// Identifier for a geographic region.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Region {
    UsEast1,
    UsWest2,
    EuWest1,
    EuCentral1,
    ApSoutheast1,
    ApNortheast1,
    Custom(String),
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Region::UsEast1 => write!(f, "us-east-1"),
            Region::UsWest2 => write!(f, "us-west-2"),
            Region::EuWest1 => write!(f, "eu-west-1"),
            Region::EuCentral1 => write!(f, "eu-central-1"),
            Region::ApSoutheast1 => write!(f, "ap-southeast-1"),
            Region::ApNortheast1 => write!(f, "ap-northeast-1"),
            Region::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// Health status of a regional deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// A regional Needle deployment endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionalEndpoint {
    pub region: Region,
    pub endpoint_url: String,
    pub health: RegionHealth,
    pub latency_ms: f64,
    pub capacity_pct: f64,
    pub last_health_check: u64,
}

/// Strategy for routing requests across regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route to nearest healthy region by latency
    LatencyBased,
    /// Round-robin across healthy regions
    RoundRobin,
    /// Route to the region with most available capacity
    CapacityBased,
    /// Always use primary, failover on unhealthy
    PrimaryWithFailover,
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        RoutingStrategy::LatencyBased
    }
}

/// Multi-region router that directs traffic to the best endpoint.
pub struct RegionRouter {
    endpoints: RwLock<Vec<RegionalEndpoint>>,
    strategy: RoutingStrategy,
    primary_region: Region,
    round_robin_counter: AtomicU64,
}

impl RegionRouter {
    pub fn new(primary_region: Region, strategy: RoutingStrategy) -> Self {
        Self {
            endpoints: RwLock::new(Vec::new()),
            strategy,
            primary_region,
            round_robin_counter: AtomicU64::new(0),
        }
    }

    /// Register a regional endpoint.
    pub fn add_endpoint(&self, endpoint: RegionalEndpoint) {
        self.endpoints.write().push(endpoint);
    }

    /// Update the health status of a region.
    pub fn update_health(&self, region: &Region, health: RegionHealth, latency_ms: f64) {
        let mut eps = self.endpoints.write();
        if let Some(ep) = eps.iter_mut().find(|e| &e.region == region) {
            ep.health = health;
            ep.latency_ms = latency_ms;
            ep.last_health_check = current_timestamp();
        }
    }

    /// Select the best endpoint for a request based on the routing strategy.
    pub fn route(&self) -> Option<RegionalEndpoint> {
        let eps = self.endpoints.read();
        let healthy: Vec<&RegionalEndpoint> = eps
            .iter()
            .filter(|e| e.health == RegionHealth::Healthy || e.health == RegionHealth::Degraded)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        match self.strategy {
            RoutingStrategy::LatencyBased => healthy
                .iter()
                .min_by(|a, b| {
                    a.latency_ms
                        .partial_cmp(&b.latency_ms)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| (*e).clone()),

            RoutingStrategy::RoundRobin => {
                let idx = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) as usize;
                Some(healthy[idx % healthy.len()].clone())
            }

            RoutingStrategy::CapacityBased => healthy
                .iter()
                .min_by(|a, b| {
                    a.capacity_pct
                        .partial_cmp(&b.capacity_pct)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| (*e).clone()),

            RoutingStrategy::PrimaryWithFailover => {
                if let Some(primary) = healthy.iter().find(|e| e.region == self.primary_region) {
                    Some((*primary).clone())
                } else {
                    healthy.first().map(|e| (*e).clone())
                }
            }
        }
    }

    /// Return all endpoints with their health status.
    pub fn list_endpoints(&self) -> Vec<RegionalEndpoint> {
        self.endpoints.read().clone()
    }
}
