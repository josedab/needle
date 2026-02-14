//! OpenTelemetry Observability Service
//!
//! Auto-instrumentation layer for Needle database operations, providing
//! distributed tracing spans, latency histograms, throughput counters,
//! and structured metrics following OTel semantic conventions.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::otel_service::{ObservabilityService, ObservabilityConfig};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let config = ObservabilityConfig::builder()
//!     .service_name("my-app")
//!     .enable_latency_histograms(true)
//!     .enable_throughput_counters(true)
//!     .build();
//!
//! let mut obs = ObservabilityService::new(config);
//!
//! // Instrument an operation
//! let coll = db.collection("docs").unwrap();
//! obs.record_insert("docs", std::time::Duration::from_micros(500), true);
//! obs.record_search("docs", std::time::Duration::from_millis(3), 10, true);
//!
//! // Get metrics snapshot
//! let metrics = obs.snapshot();
//! println!("p50 search latency: {:.2}ms", metrics.search_latency_p50_ms);
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Configuration for the observability service.
#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub enable_latency_histograms: bool,
    pub enable_throughput_counters: bool,
    pub histogram_buckets_ms: Vec<f64>,
    pub max_span_buffer: usize,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            service_name: "needle".into(),
            service_version: env!("CARGO_PKG_VERSION").into(),
            environment: "development".into(),
            enable_latency_histograms: true,
            enable_throughput_counters: true,
            histogram_buckets_ms: vec![0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0],
            max_span_buffer: 10_000,
        }
    }
}

pub struct ObservabilityConfigBuilder {
    config: ObservabilityConfig,
}

impl ObservabilityConfig {
    pub fn builder() -> ObservabilityConfigBuilder {
        ObservabilityConfigBuilder {
            config: Self::default(),
        }
    }
}

impl ObservabilityConfigBuilder {
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.config.service_name = name.into();
        self
    }

    pub fn service_version(mut self, version: impl Into<String>) -> Self {
        self.config.service_version = version.into();
        self
    }

    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = env.into();
        self
    }

    pub fn enable_latency_histograms(mut self, enable: bool) -> Self {
        self.config.enable_latency_histograms = enable;
        self
    }

    pub fn enable_throughput_counters(mut self, enable: bool) -> Self {
        self.config.enable_throughput_counters = enable;
        self
    }

    pub fn build(self) -> ObservabilityConfig {
        self.config
    }
}

/// Type of operation being observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    Insert,
    Search,
    BatchSearch,
    Delete,
    Export,
    Compact,
    Save,
}

impl std::fmt::Display for OperationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Insert => write!(f, "insert"),
            Self::Search => write!(f, "search"),
            Self::BatchSearch => write!(f, "batch_search"),
            Self::Delete => write!(f, "delete"),
            Self::Export => write!(f, "export"),
            Self::Compact => write!(f, "compact"),
            Self::Save => write!(f, "save"),
        }
    }
}

/// A recorded span (trace segment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanRecord {
    pub operation: OperationType,
    pub collection: String,
    pub duration_us: u64,
    pub success: bool,
    pub attributes: HashMap<String, String>,
    pub timestamp_ms: u64,
}

/// Latency histogram for a specific operation.
#[derive(Debug, Clone, Default)]
struct LatencyHistogram {
    buckets: Vec<f64>,
    counts: Vec<u64>,
    total_count: u64,
    total_sum_us: u64,
    min_us: u64,
    max_us: u64,
}

impl LatencyHistogram {
    fn new(bucket_boundaries_ms: &[f64]) -> Self {
        Self {
            buckets: bucket_boundaries_ms.to_vec(),
            counts: vec![0; bucket_boundaries_ms.len() + 1], // +1 for overflow
            total_count: 0,
            total_sum_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    fn record(&mut self, duration: Duration) {
        let us = duration.as_micros() as u64;
        let ms = duration.as_secs_f64() * 1000.0;

        self.total_count += 1;
        self.total_sum_us += us;
        self.min_us = self.min_us.min(us);
        self.max_us = self.max_us.max(us);

        // Find bucket
        let mut placed = false;
        for (i, boundary) in self.buckets.iter().enumerate() {
            if ms <= *boundary {
                self.counts[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            *self.counts.last_mut().expect("counts is non-empty") += 1;
        }
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        let target = (self.total_count as f64 * p / 100.0).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                if i < self.buckets.len() {
                    return self.buckets[i];
                } else {
                    return self.max_us as f64 / 1000.0;
                }
            }
        }
        self.max_us as f64 / 1000.0
    }

    fn mean_ms(&self) -> f64 {
        if self.total_count == 0 {
            0.0
        } else {
            (self.total_sum_us as f64 / self.total_count as f64) / 1000.0
        }
    }
}

/// Metrics snapshot for export.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub service_name: String,
    pub total_operations: u64,
    pub total_errors: u64,
    pub error_rate: f64,

    // Search metrics
    pub search_count: u64,
    pub search_latency_p50_ms: f64,
    pub search_latency_p95_ms: f64,
    pub search_latency_p99_ms: f64,
    pub search_latency_mean_ms: f64,
    pub search_throughput_qps: f64,

    // Insert metrics
    pub insert_count: u64,
    pub insert_latency_p50_ms: f64,
    pub insert_latency_p95_ms: f64,
    pub insert_latency_mean_ms: f64,
    pub insert_throughput_ops: f64,

    // Per-collection breakdown
    pub collections: HashMap<String, CollectionMetrics>,
}

/// Per-collection metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionMetrics {
    pub search_count: u64,
    pub insert_count: u64,
    pub delete_count: u64,
    pub error_count: u64,
}

/// The observability service — tracks all database operations.
pub struct ObservabilityService {
    config: ObservabilityConfig,
    search_histogram: RwLock<LatencyHistogram>,
    insert_histogram: RwLock<LatencyHistogram>,
    operation_counts: RwLock<HashMap<OperationType, u64>>,
    error_counts: RwLock<HashMap<OperationType, u64>>,
    collection_metrics: RwLock<HashMap<String, CollectionMetrics>>,
    spans: RwLock<Vec<SpanRecord>>,
    started_at: Instant,
}

impl ObservabilityService {
    /// Create a new observability service.
    pub fn new(config: ObservabilityConfig) -> Self {
        let buckets = config.histogram_buckets_ms.clone();
        Self {
            config,
            search_histogram: RwLock::new(LatencyHistogram::new(&buckets)),
            insert_histogram: RwLock::new(LatencyHistogram::new(&buckets)),
            operation_counts: RwLock::new(HashMap::new()),
            error_counts: RwLock::new(HashMap::new()),
            collection_metrics: RwLock::new(HashMap::new()),
            spans: RwLock::new(Vec::new()),
            started_at: Instant::now(),
        }
    }

    /// Record an insert operation.
    pub fn record_insert(&self, collection: &str, duration: Duration, success: bool) {
        self.record_operation(
            OperationType::Insert,
            collection,
            duration,
            success,
            HashMap::new(),
        );
        if self.config.enable_latency_histograms {
            self.insert_histogram.write().record(duration);
        }
    }

    /// Record a search operation.
    pub fn record_search(&self, collection: &str, duration: Duration, k: usize, success: bool) {
        let mut attrs = HashMap::new();
        attrs.insert("k".into(), k.to_string());
        self.record_operation(OperationType::Search, collection, duration, success, attrs);
        if self.config.enable_latency_histograms {
            self.search_histogram.write().record(duration);
        }
    }

    /// Record a delete operation.
    pub fn record_delete(&self, collection: &str, duration: Duration, success: bool) {
        self.record_operation(
            OperationType::Delete,
            collection,
            duration,
            success,
            HashMap::new(),
        );
    }

    /// Record a generic operation.
    pub fn record_operation(
        &self,
        op: OperationType,
        collection: &str,
        duration: Duration,
        success: bool,
        attributes: HashMap<String, String>,
    ) {
        // Update counters
        *self.operation_counts.write().entry(op).or_insert(0) += 1;
        if !success {
            *self.error_counts.write().entry(op).or_insert(0) += 1;
        }

        // Update collection metrics
        let mut coll_metrics = self.collection_metrics.write();
        let cm = coll_metrics.entry(collection.to_string()).or_default();
        match op {
            OperationType::Search | OperationType::BatchSearch => cm.search_count += 1,
            OperationType::Insert => cm.insert_count += 1,
            OperationType::Delete => cm.delete_count += 1,
            _ => {}
        }
        if !success {
            cm.error_count += 1;
        }

        // Record span
        let mut spans = self.spans.write();
        if spans.len() < self.config.max_span_buffer {
            spans.push(SpanRecord {
                operation: op,
                collection: collection.to_string(),
                duration_us: duration.as_micros() as u64,
                success,
                attributes,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
            });
        }
    }

    /// Get a metrics snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let op_counts = self.operation_counts.read();
        let err_counts = self.error_counts.read();
        let search_hist = self.search_histogram.read();
        let insert_hist = self.insert_histogram.read();
        let uptime_secs = self.started_at.elapsed().as_secs_f64().max(1.0);

        let total_ops: u64 = op_counts.values().sum();
        let total_errors: u64 = err_counts.values().sum();
        let search_count = *op_counts.get(&OperationType::Search).unwrap_or(&0);
        let insert_count = *op_counts.get(&OperationType::Insert).unwrap_or(&0);

        MetricsSnapshot {
            service_name: self.config.service_name.clone(),
            total_operations: total_ops,
            total_errors,
            error_rate: if total_ops > 0 {
                total_errors as f64 / total_ops as f64
            } else {
                0.0
            },
            search_count,
            search_latency_p50_ms: search_hist.percentile(50.0),
            search_latency_p95_ms: search_hist.percentile(95.0),
            search_latency_p99_ms: search_hist.percentile(99.0),
            search_latency_mean_ms: search_hist.mean_ms(),
            search_throughput_qps: search_count as f64 / uptime_secs,
            insert_count,
            insert_latency_p50_ms: insert_hist.percentile(50.0),
            insert_latency_p95_ms: insert_hist.percentile(95.0),
            insert_latency_mean_ms: insert_hist.mean_ms(),
            insert_throughput_ops: insert_count as f64 / uptime_secs,
            collections: self.collection_metrics.read().clone(),
        }
    }

    /// Export metrics in Prometheus exposition format.
    pub fn prometheus_export(&self) -> String {
        let snap = self.snapshot();
        let mut out = String::new();
        let svc = &snap.service_name;

        out.push_str(&format!(
            "# HELP needle_operations_total Total operations\n\
             # TYPE needle_operations_total counter\n\
             needle_operations_total{{service=\"{}\"}} {}\n\n",
            svc, snap.total_operations
        ));
        out.push_str(&format!(
            "# HELP needle_errors_total Total errors\n\
             # TYPE needle_errors_total counter\n\
             needle_errors_total{{service=\"{}\"}} {}\n\n",
            svc, snap.total_errors
        ));
        out.push_str(&format!(
            "# HELP needle_search_latency_ms Search latency\n\
             # TYPE needle_search_latency_ms summary\n\
             needle_search_latency_ms{{service=\"{}\",quantile=\"0.5\"}} {:.4}\n\
             needle_search_latency_ms{{service=\"{}\",quantile=\"0.95\"}} {:.4}\n\
             needle_search_latency_ms{{service=\"{}\",quantile=\"0.99\"}} {:.4}\n\n",
            svc,
            snap.search_latency_p50_ms,
            svc,
            snap.search_latency_p95_ms,
            svc,
            snap.search_latency_p99_ms,
        ));
        out.push_str(&format!(
            "# HELP needle_insert_latency_ms Insert latency\n\
             # TYPE needle_insert_latency_ms summary\n\
             needle_insert_latency_ms{{service=\"{}\",quantile=\"0.5\"}} {:.4}\n\
             needle_insert_latency_ms{{service=\"{}\",quantile=\"0.95\"}} {:.4}\n\n",
            svc, snap.insert_latency_p50_ms, svc, snap.insert_latency_p95_ms,
        ));
        out.push_str(&format!(
            "# HELP needle_search_qps Search queries per second\n\
             # TYPE needle_search_qps gauge\n\
             needle_search_qps{{service=\"{}\"}} {:.2}\n\n",
            svc, snap.search_throughput_qps
        ));

        for (coll, cm) in &snap.collections {
            out.push_str(&format!(
                "needle_collection_searches{{service=\"{}\",collection=\"{}\"}} {}\n",
                svc, coll, cm.search_count
            ));
            out.push_str(&format!(
                "needle_collection_inserts{{service=\"{}\",collection=\"{}\"}} {}\n",
                svc, coll, cm.insert_count
            ));
        }

        out
    }

    /// Get recorded spans (for distributed tracing export).
    pub fn drain_spans(&self) -> Vec<SpanRecord> {
        let mut spans = self.spans.write();
        std::mem::take(&mut *spans)
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        *self.search_histogram.write() = LatencyHistogram::new(&self.config.histogram_buckets_ms);
        *self.insert_histogram.write() = LatencyHistogram::new(&self.config.histogram_buckets_ms);
        self.operation_counts.write().clear();
        self.error_counts.write().clear();
        self.collection_metrics.write().clear();
        self.spans.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_snapshot() {
        let config = ObservabilityConfig::builder().service_name("test").build();
        let obs = ObservabilityService::new(config);

        obs.record_insert("col1", Duration::from_millis(1), true);
        obs.record_insert("col1", Duration::from_millis(2), true);
        obs.record_search("col1", Duration::from_millis(5), 10, true);
        obs.record_search("col1", Duration::from_millis(3), 10, false);

        let snap = obs.snapshot();
        assert_eq!(snap.insert_count, 2);
        assert_eq!(snap.search_count, 2);
        assert_eq!(snap.total_errors, 1);
        assert!(snap.error_rate > 0.0);
    }

    #[test]
    fn test_latency_percentiles() {
        let config = ObservabilityConfig::default();
        let obs = ObservabilityService::new(config);

        for i in 0..100 {
            obs.record_search("col", Duration::from_millis(i), 10, true);
        }

        let snap = obs.snapshot();
        assert!(snap.search_latency_p50_ms > 0.0);
        assert!(snap.search_latency_p95_ms > snap.search_latency_p50_ms);
        assert!(snap.search_latency_p99_ms >= snap.search_latency_p95_ms);
    }

    #[test]
    fn test_prometheus_export() {
        let config = ObservabilityConfig::builder()
            .service_name("needle-test")
            .build();
        let obs = ObservabilityService::new(config);

        obs.record_search("docs", Duration::from_millis(5), 10, true);
        obs.record_insert("docs", Duration::from_millis(1), true);

        let prom = obs.prometheus_export();
        assert!(prom.contains("needle_operations_total"));
        assert!(prom.contains("needle_search_latency_ms"));
        assert!(prom.contains("needle-test"));
    }

    #[test]
    fn test_per_collection_metrics() {
        let config = ObservabilityConfig::default();
        let obs = ObservabilityService::new(config);

        obs.record_search("col_a", Duration::from_millis(1), 5, true);
        obs.record_search("col_b", Duration::from_millis(2), 10, true);
        obs.record_insert("col_a", Duration::from_millis(1), true);

        let snap = obs.snapshot();
        assert_eq!(snap.collections["col_a"].search_count, 1);
        assert_eq!(snap.collections["col_a"].insert_count, 1);
        assert_eq!(snap.collections["col_b"].search_count, 1);
    }

    #[test]
    fn test_drain_spans() {
        let config = ObservabilityConfig::default();
        let obs = ObservabilityService::new(config);

        obs.record_search("col", Duration::from_millis(1), 5, true);
        obs.record_insert("col", Duration::from_millis(1), true);

        let spans = obs.drain_spans();
        assert_eq!(spans.len(), 2);

        // After drain, spans should be empty
        let spans = obs.drain_spans();
        assert!(spans.is_empty());
    }

    #[test]
    fn test_reset() {
        let config = ObservabilityConfig::default();
        let obs = ObservabilityService::new(config);

        obs.record_search("col", Duration::from_millis(1), 5, true);
        obs.reset();

        let snap = obs.snapshot();
        assert_eq!(snap.total_operations, 0);
    }
}
