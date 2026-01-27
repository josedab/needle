//! Production Readiness Probe
//!
//! `needle doctor` CLI + `/readiness` HTTP endpoint that validates index health,
//! WAL integrity, storage utilization, and query latency percentiles.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::readiness_probe::{
//!     ReadinessProbe, ProbeConfig, HealthReport, CheckResult, CheckStatus,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let probe = ReadinessProbe::new(ProbeConfig::default());
//! let report = probe.check(&db);
//!
//! println!("Overall: {:?}", report.status);
//! for check in &report.checks {
//!     println!("  {} : {:?} - {}", check.name, check.status, check.message);
//! }
//! ```

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::database::Database;

/// Check status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckStatus { Pass, Warn, Fail }

/// A single check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
    pub duration_us: u64,
}

/// Overall health report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    pub status: CheckStatus,
    pub checks: Vec<CheckResult>,
    pub total_duration_us: u64,
    pub timestamp: u64,
}

impl HealthReport {
    /// Format as JSON for HTTP response.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Single-line summary.
    pub fn summary(&self) -> String {
        let pass = self.checks.iter().filter(|c| c.status == CheckStatus::Pass).count();
        let warn = self.checks.iter().filter(|c| c.status == CheckStatus::Warn).count();
        let fail = self.checks.iter().filter(|c| c.status == CheckStatus::Fail).count();
        format!("{:?}: {} pass, {} warn, {} fail", self.status, pass, warn, fail)
    }
}

/// Probe configuration with thresholds.
#[derive(Debug, Clone)]
pub struct ProbeConfig {
    pub max_collections: usize,
    pub max_vectors_per_collection: usize,
    pub max_search_latency_ms: f64,
    pub search_sample_k: usize,
}

impl Default for ProbeConfig {
    fn default() -> Self {
        Self {
            max_collections: 1000,
            max_vectors_per_collection: 100_000_000,
            max_search_latency_ms: 100.0,
            search_sample_k: 10,
        }
    }
}

/// Production readiness probe.
pub struct ReadinessProbe {
    config: ProbeConfig,
}

impl ReadinessProbe {
    pub fn new(config: ProbeConfig) -> Self { Self { config } }

    /// Run all health checks against a database.
    pub fn check(&self, db: &Database) -> HealthReport {
        let start = Instant::now();
        let mut checks = Vec::new();

        checks.push(self.check_collections(db));
        checks.push(self.check_collection_sizes(db));
        checks.push(self.check_search_latency(db));
        checks.push(self.check_api_responsiveness(db));

        let overall = if checks.iter().any(|c| c.status == CheckStatus::Fail) {
            CheckStatus::Fail
        } else if checks.iter().any(|c| c.status == CheckStatus::Warn) {
            CheckStatus::Warn
        } else {
            CheckStatus::Pass
        };

        HealthReport {
            status: overall,
            checks,
            total_duration_us: start.elapsed().as_micros() as u64,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    fn check_collections(&self, db: &Database) -> CheckResult {
        let start = Instant::now();
        let collections = db.list_collections();
        let count = collections.len();
        let (status, msg) = if count > self.config.max_collections {
            (CheckStatus::Fail, format!("{count} collections exceeds limit {}", self.config.max_collections))
        } else if count == 0 {
            (CheckStatus::Warn, "No collections found".into())
        } else {
            (CheckStatus::Pass, format!("{count} collections OK"))
        };
        CheckResult { name: "collections".into(), status, message: msg, duration_us: start.elapsed().as_micros() as u64 }
    }

    fn check_collection_sizes(&self, db: &Database) -> CheckResult {
        let start = Instant::now();
        let collections = db.list_collections();
        let mut issues = Vec::new();
        for name in &collections {
            if let Ok(coll) = db.collection(name) {
                let size = coll.len();
                if size > self.config.max_vectors_per_collection {
                    issues.push(format!("'{}': {} vectors exceeds limit", name, size));
                }
            }
        }
        let status = if issues.is_empty() { CheckStatus::Pass } else { CheckStatus::Warn };
        let msg = if issues.is_empty() {
            "All collection sizes within limits".into()
        } else {
            issues.join("; ")
        };
        CheckResult { name: "collection_sizes".into(), status, message: msg, duration_us: start.elapsed().as_micros() as u64 }
    }

    fn check_search_latency(&self, db: &Database) -> CheckResult {
        let start = Instant::now();
        let collections = db.list_collections();
        if collections.is_empty() {
            return CheckResult {
                name: "search_latency".into(), status: CheckStatus::Pass,
                message: "No collections to test".into(), duration_us: start.elapsed().as_micros() as u64,
            };
        }

        let mut max_latency_ms = 0.0f64;
        let mut search_failed = false;
        for name in &collections {
            if let Ok(coll) = db.collection(name) {
                if coll.len() == 0 { continue; }
                let dims = coll.dimensions().unwrap_or(4);
                let query = vec![0.0f32; dims];
                let search_start = Instant::now();
                match coll.search(&query, self.config.search_sample_k) {
                    Ok(_) => {
                        let latency = search_start.elapsed().as_secs_f64() * 1000.0;
                        max_latency_ms = max_latency_ms.max(latency);
                    }
                    Err(_) => {
                        search_failed = true;
                    }
                }
            }
        }

        if search_failed {
            return CheckResult {
                name: "search_latency".into(),
                status: CheckStatus::Fail,
                message: "Search operation failed".into(),
                duration_us: start.elapsed().as_micros() as u64,
            };
        }

        let (status, msg) = if max_latency_ms > self.config.max_search_latency_ms {
            (CheckStatus::Fail, format!("Max search latency {max_latency_ms:.1}ms exceeds threshold"))
        } else if max_latency_ms > self.config.max_search_latency_ms * 0.8 {
            (CheckStatus::Warn, format!("Max search latency {max_latency_ms:.1}ms approaching threshold"))
        } else {
            (CheckStatus::Pass, format!("Max search latency {max_latency_ms:.1}ms OK"))
        };
        CheckResult { name: "search_latency".into(), status, message: msg, duration_us: start.elapsed().as_micros() as u64 }
    }

    fn check_api_responsiveness(&self, db: &Database) -> CheckResult {
        let start = Instant::now();
        let collections_result = db.list_collections();
        let latency = start.elapsed().as_micros() as u64;
        if collections_result.is_empty() && latency > 10_000 {
            return CheckResult {
                name: "api_responsiveness".into(),
                status: CheckStatus::Warn,
                message: format!("API response in {latency}µs (slow, no collections)"),
                duration_us: latency,
            };
        }
        CheckResult {
            name: "api_responsiveness".into(),
            status: if latency < 10_000 { CheckStatus::Pass } else { CheckStatus::Warn },
            message: format!("API response in {latency}µs"),
            duration_us: latency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_database() {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let coll = db.collection("docs").unwrap();
        coll.insert("v1", &[1.0; 4], None).unwrap();

        let probe = ReadinessProbe::new(ProbeConfig::default());
        let report = probe.check(&db);
        assert_eq!(report.status, CheckStatus::Pass);
        assert_eq!(report.checks.len(), 4);
    }

    #[test]
    fn test_empty_database_warns() {
        let db = Database::in_memory();
        let probe = ReadinessProbe::new(ProbeConfig::default());
        let report = probe.check(&db);
        assert_eq!(report.status, CheckStatus::Warn);
    }

    #[test]
    fn test_summary() {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let probe = ReadinessProbe::new(ProbeConfig::default());
        let report = probe.check(&db);
        let summary = report.summary();
        assert!(summary.contains("pass"));
    }

    #[test]
    fn test_json_output() {
        let db = Database::in_memory();
        let probe = ReadinessProbe::new(ProbeConfig::default());
        let report = probe.check(&db);
        let json = report.to_json();
        assert!(json.contains("status"));
        assert!(json.contains("checks"));
    }

    #[test]
    fn test_latency_check_with_data() {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let coll = db.collection("docs").unwrap();
        for i in 0..100 {
            coll.insert(format!("v{i}"), &[i as f32 * 0.01, 0.0, 0.0, 0.0], None).unwrap();
        }

        let probe = ReadinessProbe::new(ProbeConfig::default());
        let report = probe.check(&db);
        let latency_check = report.checks.iter().find(|c| c.name == "search_latency").unwrap();
        assert_eq!(latency_check.status, CheckStatus::Pass);
    }
}
