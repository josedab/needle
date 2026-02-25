#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::control_plane::current_timestamp;

/// Tracks SLA compliance for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaPolicy {
    /// Target availability percentage (e.g. 99.95)
    pub target_availability_pct: f64,
    /// Maximum p99 query latency in milliseconds
    pub max_p99_latency_ms: f64,
    /// Maximum data loss window in seconds (RPO)
    pub max_data_loss_seconds: u64,
    /// Maximum recovery time in seconds (RTO)
    pub max_recovery_seconds: u64,
}

impl Default for SlaPolicy {
    fn default() -> Self {
        Self {
            target_availability_pct: 99.9,
            max_p99_latency_ms: 50.0,
            max_data_loss_seconds: 60,
            max_recovery_seconds: 300,
        }
    }
}

/// A recorded SLA breach event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaBreach {
    pub timestamp: u64,
    pub breach_type: SlaBreachType,
    pub actual_value: f64,
    pub threshold: f64,
    pub duration_seconds: u64,
}

/// Types of SLA breaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlaBreachType {
    Availability,
    Latency,
    DataLoss,
    RecoveryTime,
}

/// Sliding-window SLA health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlaReport {
    pub window_seconds: u64,
    pub availability_pct: f64,
    pub p99_latency_ms: f64,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub breaches: Vec<SlaBreach>,
    pub compliant: bool,
}

/// Monitors SLA compliance using a rolling window of request outcomes.
pub struct SlaMonitor {
    policy: SlaPolicy,
    window_seconds: u64,
    request_log: RwLock<Vec<RequestOutcome>>,
    breaches: RwLock<Vec<SlaBreach>>,
}

#[derive(Debug, Clone)]
struct RequestOutcome {
    timestamp: u64,
    success: bool,
    latency_ms: f64,
}

impl SlaMonitor {
    pub fn new(policy: SlaPolicy, window_seconds: u64) -> Self {
        Self {
            policy,
            window_seconds,
            request_log: RwLock::new(Vec::new()),
            breaches: RwLock::new(Vec::new()),
        }
    }

    /// Record a request outcome for SLA tracking.
    pub fn record_request(&self, success: bool, latency_ms: f64) {
        let outcome = RequestOutcome {
            timestamp: current_timestamp(),
            success,
            latency_ms,
        };
        self.request_log.write().push(outcome);
    }

    /// Generate an SLA compliance report for the current window.
    pub fn report(&self) -> SlaReport {
        let now = current_timestamp();
        let cutoff = now.saturating_sub(self.window_seconds);

        let log = self.request_log.read();
        let windowed: Vec<&RequestOutcome> = log.iter().filter(|r| r.timestamp >= cutoff).collect();

        let total = windowed.len() as u64;
        let failed = windowed.iter().filter(|r| !r.success).count() as u64;
        let availability_pct = if total > 0 {
            ((total - failed) as f64 / total as f64) * 100.0
        } else {
            100.0
        };

        let p99_latency_ms = {
            let mut latencies: Vec<f64> = windowed.iter().map(|r| r.latency_ms).collect();
            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if latencies.is_empty() {
                0.0
            } else {
                let idx = ((latencies.len() as f64) * 0.99).ceil() as usize;
                latencies[idx.min(latencies.len() - 1)]
            }
        };

        let mut new_breaches = Vec::new();
        if availability_pct < self.policy.target_availability_pct && total > 0 {
            new_breaches.push(SlaBreach {
                timestamp: now,
                breach_type: SlaBreachType::Availability,
                actual_value: availability_pct,
                threshold: self.policy.target_availability_pct,
                duration_seconds: self.window_seconds,
            });
        }
        if p99_latency_ms > self.policy.max_p99_latency_ms && total > 0 {
            new_breaches.push(SlaBreach {
                timestamp: now,
                breach_type: SlaBreachType::Latency,
                actual_value: p99_latency_ms,
                threshold: self.policy.max_p99_latency_ms,
                duration_seconds: self.window_seconds,
            });
        }

        if !new_breaches.is_empty() {
            self.breaches.write().extend(new_breaches.clone());
        }

        let all_breaches = self.breaches.read().clone();
        SlaReport {
            window_seconds: self.window_seconds,
            availability_pct,
            p99_latency_ms,
            total_requests: total,
            failed_requests: failed,
            breaches: all_breaches,
            compliant: availability_pct >= self.policy.target_availability_pct
                && p99_latency_ms <= self.policy.max_p99_latency_ms,
        }
    }

    /// Prune old entries outside the monitoring window.
    pub fn prune(&self) {
        let cutoff = current_timestamp().saturating_sub(self.window_seconds * 2);
        self.request_log.write().retain(|r| r.timestamp >= cutoff);
    }
}
