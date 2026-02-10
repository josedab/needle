//! Vector Observability Suite
//!
//! Comprehensive monitoring for the Needle vector database with query latency
//! histograms, recall estimation, index health metrics, drift detection
//! integration, and pre-built Grafana dashboard generation.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Default histogram bucket boundaries in microseconds.
const DEFAULT_BUCKET_BOUNDARIES: &[f64] = &[
    100.0, 500.0, 1_000.0, 2_500.0, 5_000.0, 10_000.0, 25_000.0, 50_000.0, 100_000.0,
];

// ---------------------------------------------------------------------------
// LatencyHistogram
// ---------------------------------------------------------------------------

/// Cumulative histogram tracking query latency in microseconds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// (upper_bound_us, cumulative_count)
    pub buckets: Vec<(f64, u64)>,
    pub total_count: u64,
    pub sum_us: u64,
    pub min_us: u64,
    pub max_us: u64,
}

impl LatencyHistogram {
    /// Create a new histogram with the given bucket upper-bounds (in µs).
    pub fn new(bucket_boundaries: &[f64]) -> Self {
        let buckets = bucket_boundaries.iter().map(|&b| (b, 0u64)).collect();
        Self {
            buckets,
            total_count: 0,
            sum_us: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    /// Record an observed latency value (in µs).
    pub fn observe(&mut self, latency_us: u64) {
        self.total_count += 1;
        self.sum_us += latency_us;
        if latency_us < self.min_us {
            self.min_us = latency_us;
        }
        if latency_us > self.max_us {
            self.max_us = latency_us;
        }
        let val = latency_us as f64;
        for bucket in self.buckets.iter_mut() {
            if val <= bucket.0 {
                bucket.1 += 1;
            }
        }
    }

    /// Estimate the value at the given percentile `p` (0.0–1.0).
    ///
    /// Uses linear interpolation across cumulative bucket counts.
    pub fn percentile(&self, p: f64) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        let target = p * self.total_count as f64;

        let mut prev_count: u64 = 0;
        let mut prev_bound: f64 = 0.0;

        for &(bound, count) in &self.buckets {
            if count as f64 >= target {
                let count_in_bucket = count - prev_count;
                if count_in_bucket == 0 {
                    return prev_bound;
                }
                let fraction = (target - prev_count as f64) / count_in_bucket as f64;
                return prev_bound + fraction * (bound - prev_bound);
            }
            prev_count = count;
            prev_bound = bound;
        }

        // Beyond the last bucket – return max observed value.
        self.max_us as f64
    }

    /// Arithmetic mean latency in µs.
    pub fn mean(&self) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.sum_us as f64 / self.total_count as f64
    }

    /// Reset all counters to zero.
    pub fn reset(&mut self) {
        for bucket in self.buckets.iter_mut() {
            bucket.1 = 0;
        }
        self.total_count = 0;
        self.sum_us = 0;
        self.min_us = u64::MAX;
        self.max_us = 0;
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new(DEFAULT_BUCKET_BOUNDARIES)
    }
}

// ---------------------------------------------------------------------------
// RecallEstimator
// ---------------------------------------------------------------------------

/// Estimates ANN recall by comparing approximate results with exact results.
#[derive(Debug, Clone)]
pub struct RecallEstimator {
    pub sample_size: usize,
    pub estimated_recall: f32,
    pub confidence: f32,
    pub last_measured: Option<Instant>,
}

impl RecallEstimator {
    pub fn new(sample_size: usize) -> Self {
        Self {
            sample_size,
            estimated_recall: 0.0,
            confidence: 0.0,
            last_measured: None,
        }
    }

    /// Compute recall as Jaccard similarity between approximate and exact result sets.
    pub fn estimate(&mut self, approximate_results: &[String], exact_results: &[String]) -> f32 {
        if exact_results.is_empty() && approximate_results.is_empty() {
            self.estimated_recall = 1.0;
            self.confidence = 1.0;
            self.last_measured = Some(Instant::now());
            return 1.0;
        }
        if exact_results.is_empty() || approximate_results.is_empty() {
            self.estimated_recall = 0.0;
            self.confidence = 1.0;
            self.last_measured = Some(Instant::now());
            return 0.0;
        }

        let approx_set: std::collections::HashSet<&String> = approximate_results.iter().collect();
        let exact_set: std::collections::HashSet<&String> = exact_results.iter().collect();

        let intersection = approx_set.intersection(&exact_set).count();
        let union = approx_set.union(&exact_set).count();

        let recall = if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        };

        self.estimated_recall = recall;
        self.confidence = (approximate_results.len().min(exact_results.len()) as f32
            / self.sample_size as f32)
            .min(1.0);
        self.last_measured = Some(Instant::now());
        recall
    }

    /// Returns `true` when the last measurement is older than `max_age`.
    pub fn is_stale(&self, max_age: Duration) -> bool {
        match self.last_measured {
            Some(ts) => ts.elapsed() > max_age,
            None => true,
        }
    }
}

// ---------------------------------------------------------------------------
// IndexHealthMetrics
// ---------------------------------------------------------------------------

/// Health snapshot for a single collection index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealthMetrics {
    /// Ratio of deleted vectors to total vectors (0.0–1.0).
    pub fragmentation_ratio: f32,
    /// Balance score of the HNSW graph (0.0–1.0, higher is better).
    pub balance_score: f32,
    pub memory_usage_bytes: u64,
    pub disk_usage_bytes: u64,
    pub vector_count: usize,
    pub deleted_count: usize,
    pub avg_connections_per_node: f32,
    pub last_compaction: Option<String>,
}

impl Default for IndexHealthMetrics {
    fn default() -> Self {
        Self {
            fragmentation_ratio: 0.0,
            balance_score: 1.0,
            memory_usage_bytes: 0,
            disk_usage_bytes: 0,
            vector_count: 0,
            deleted_count: 0,
            avg_connections_per_node: 0.0,
            last_compaction: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Alerting types
// ---------------------------------------------------------------------------

/// Severity level for drift / anomaly alerts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// A triggered alert produced by the observability suite.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DriftAlert {
    pub alert_id: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub triggered_at: String,
    pub acknowledged: bool,
}

/// Which metric an alert rule evaluates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MetricType {
    QueryLatencyP99,
    RecallEstimate,
    Fragmentation,
    MemoryUsage,
    DriftScore,
    ErrorRate,
}

/// Condition that must hold for an alert to fire.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AlertCondition {
    GreaterThan(f64),
    LessThan(f64),
    OutOfRange(f64, f64),
}

/// A user-defined alert rule evaluated on every `check_alerts` call.
pub struct AlertRule {
    pub name: String,
    pub metric: MetricType,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub cooldown: Duration,
    pub last_fired: Option<Instant>,
}

// ---------------------------------------------------------------------------
// ObservabilitySuite
// ---------------------------------------------------------------------------

/// Central observability hub — one instance per `Database`.
pub struct ObservabilitySuite {
    latency_histograms: RwLock<HashMap<String, LatencyHistogram>>,
    recall_estimator: RwLock<RecallEstimator>,
    index_health: RwLock<HashMap<String, IndexHealthMetrics>>,
    alerts: RwLock<Vec<DriftAlert>>,
    alert_rules: RwLock<Vec<AlertRule>>,
    error_counts: RwLock<HashMap<String, u64>>,
}

impl ObservabilitySuite {
    /// Create a new, empty observability suite.
    pub fn new() -> Self {
        Self {
            latency_histograms: RwLock::new(HashMap::new()),
            recall_estimator: RwLock::new(RecallEstimator::new(100)),
            index_health: RwLock::new(HashMap::new()),
            alerts: RwLock::new(Vec::new()),
            alert_rules: RwLock::new(Vec::new()),
            error_counts: RwLock::new(HashMap::new()),
        }
    }

    /// Record a query latency observation for the given collection.
    pub fn record_query_latency(&self, collection: &str, latency_us: u64) {
        let mut histograms = self.latency_histograms.write();
        histograms
            .entry(collection.to_string())
            .or_insert_with(LatencyHistogram::default)
            .observe(latency_us);
    }

    /// Increment an error counter keyed by `"collection:error_type"`.
    pub fn record_error(&self, collection: &str, error_type: &str) {
        let key = format!("{}:{}", collection, error_type);
        let mut counts = self.error_counts.write();
        *counts.entry(key).or_insert(0) += 1;
    }

    /// Replace the health snapshot for the given collection.
    pub fn update_index_health(&self, collection: &str, health: IndexHealthMetrics) {
        let mut map = self.index_health.write();
        map.insert(collection.to_string(), health);
    }

    /// Estimate recall for a collection and store the result.
    pub fn estimate_recall(&self, _collection: &str, approx: &[String], exact: &[String]) -> f32 {
        let mut estimator = self.recall_estimator.write();
        estimator.estimate(approx, exact)
    }

    /// Register an alert rule to be evaluated on each `check_alerts` call.
    pub fn add_alert_rule(&self, rule: AlertRule) {
        let mut rules = self.alert_rules.write();
        rules.push(rule);
    }

    /// Evaluate all alert rules against current metrics and return newly fired alerts.
    pub fn check_alerts(&self) -> Vec<DriftAlert> {
        let mut rules = self.alert_rules.write();
        let histograms = self.latency_histograms.read();
        let recall = self.recall_estimator.read();
        let health_map = self.index_health.read();
        let error_counts = self.error_counts.read();

        let now = Instant::now();
        let mut new_alerts: Vec<DriftAlert> = Vec::new();

        for rule in rules.iter_mut() {
            // Cooldown check
            if let Some(last) = rule.last_fired {
                if now.duration_since(last) < rule.cooldown {
                    continue;
                }
            }

            let current_value: Option<f64> = match &rule.metric {
                MetricType::QueryLatencyP99 => {
                    // Aggregate p99 across all collections (take max).
                    histograms
                        .values()
                        .map(|h| h.percentile(0.99))
                        .fold(None, |acc: Option<f64>, v| {
                            Some(acc.map_or(v, |a: f64| a.max(v)))
                        })
                }
                MetricType::RecallEstimate => Some(recall.estimated_recall as f64),
                MetricType::Fragmentation => health_map
                    .values()
                    .map(|h| h.fragmentation_ratio as f64)
                    .fold(None, |acc: Option<f64>, v| {
                        Some(acc.map_or(v, |a: f64| a.max(v)))
                    }),
                MetricType::MemoryUsage => health_map
                    .values()
                    .map(|h| h.memory_usage_bytes as f64)
                    .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a + v))),
                MetricType::DriftScore => None, // external integration
                MetricType::ErrorRate => {
                    let total: u64 = error_counts.values().sum();
                    Some(total as f64)
                }
            };

            let current_value = match current_value {
                Some(v) => v,
                None => continue,
            };

            let triggered = match &rule.condition {
                AlertCondition::GreaterThan(t) => current_value > *t,
                AlertCondition::LessThan(t) => current_value < *t,
                AlertCondition::OutOfRange(lo, hi) => current_value < *lo || current_value > *hi,
            };

            if triggered {
                let threshold = match &rule.condition {
                    AlertCondition::GreaterThan(t) | AlertCondition::LessThan(t) => *t,
                    AlertCondition::OutOfRange(lo, hi) => {
                        if current_value < *lo {
                            *lo
                        } else {
                            *hi
                        }
                    }
                };

                let alert = DriftAlert {
                    alert_id: format!("{}-{}", rule.name, chrono::Utc::now().timestamp_millis()),
                    severity: rule.severity.clone(),
                    message: format!(
                        "Rule '{}' triggered: current value {:.4} crossed threshold {:.4}",
                        rule.name, current_value, threshold
                    ),
                    metric_name: format!("{:?}", rule.metric),
                    current_value,
                    threshold,
                    triggered_at: chrono::Utc::now().to_rfc3339(),
                    acknowledged: false,
                };

                rule.last_fired = Some(now);
                new_alerts.push(alert);
            }
        }

        // Persist new alerts.
        let mut stored = self.alerts.write();
        stored.extend(new_alerts.clone());
        new_alerts
    }

    /// Acknowledge (silence) an alert by its ID.
    pub fn acknowledge_alert(&self, alert_id: &str) {
        let mut alerts = self.alerts.write();
        for alert in alerts.iter_mut() {
            if alert.alert_id == alert_id {
                alert.acknowledged = true;
            }
        }
    }

    /// Return a clone of the latency histogram for the given collection.
    pub fn get_latency_histogram(&self, collection: &str) -> Option<LatencyHistogram> {
        let histograms = self.latency_histograms.read();
        histograms.get(collection).cloned()
    }

    /// Return a clone of the health metrics for the given collection.
    pub fn get_health(&self, collection: &str) -> Option<IndexHealthMetrics> {
        let health = self.index_health.read();
        health.get(collection).cloned()
    }

    /// Return all unacknowledged alerts.
    pub fn active_alerts(&self) -> Vec<DriftAlert> {
        let alerts = self.alerts.read();
        alerts.iter().filter(|a| !a.acknowledged).cloned().collect()
    }

    /// Generate a Grafana dashboard JSON model with panels for each collection.
    pub fn generate_grafana_dashboard(&self, collections: &[String]) -> String {
        let mut panels: Vec<serde_json::Value> = Vec::new();
        let mut panel_id = 1u64;
        let mut y_pos = 0u64;

        // Per-collection latency histogram panels
        for coll in collections {
            panels.push(serde_json::json!({
                "id": panel_id,
                "type": "histogram",
                "title": format!("Query Latency — {}", coll),
                "gridPos": { "x": 0, "y": y_pos, "w": 12, "h": 8 },
                "targets": [{
                    "expr": format!("needle_query_latency_us_bucket{{collection=\"{}\"}}", coll),
                    "legendFormat": "{{le}}"
                }],
                "fieldConfig": {
                    "defaults": { "unit": "µs" }
                }
            }));
            panel_id += 1;
            y_pos += 8;
        }

        // Recall estimate gauge
        panels.push(serde_json::json!({
            "id": panel_id,
            "type": "gauge",
            "title": "Recall Estimate",
            "gridPos": { "x": 12, "y": 0, "w": 6, "h": 8 },
            "targets": [{
                "expr": "needle_recall_estimate",
                "legendFormat": "recall"
            }],
            "fieldConfig": {
                "defaults": {
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "steps": [
                            { "value": 0, "color": "red" },
                            { "value": 0.8, "color": "yellow" },
                            { "value": 0.95, "color": "green" }
                        ]
                    }
                }
            }
        }));
        panel_id += 1;

        // Memory usage graph
        panels.push(serde_json::json!({
            "id": panel_id,
            "type": "timeseries",
            "title": "Memory Usage",
            "gridPos": { "x": 18, "y": 0, "w": 6, "h": 8 },
            "targets": [{
                "expr": "needle_memory_usage_bytes",
                "legendFormat": "{{collection}}"
            }],
            "fieldConfig": {
                "defaults": { "unit": "bytes" }
            }
        }));
        panel_id += 1;

        // Active alerts list
        panels.push(serde_json::json!({
            "id": panel_id,
            "type": "table",
            "title": "Active Alerts",
            "gridPos": { "x": 0, "y": y_pos, "w": 24, "h": 6 },
            "targets": [{
                "expr": "needle_active_alerts",
                "format": "table"
            }],
            "fieldConfig": {
                "defaults": {}
            }
        }));

        let dashboard = serde_json::json!({
            "uid": "needle-observability",
            "title": "Needle Vector Database — Observability",
            "tags": ["needle", "vector-database", "observability"],
            "timezone": "browser",
            "schemaVersion": 39,
            "version": 1,
            "refresh": "10s",
            "panels": panels
        });

        serde_json::to_string_pretty(&dashboard).unwrap_or_default()
    }
}

impl Default for ObservabilitySuite {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// =============================================================================
// OpenTelemetry Export Layer
// =============================================================================

/// OpenTelemetry-compatible metric point for export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelMetricPoint {
    /// Metric name (e.g. "needle.query.latency_us").
    pub name: String,
    /// Description.
    pub description: String,
    /// Metric type: "histogram", "counter", "gauge".
    pub metric_type: String,
    /// Resource attributes (service.name, etc.).
    pub resource: HashMap<String, String>,
    /// Data points.
    pub data_points: Vec<OtelDataPoint>,
}

/// A single OpenTelemetry data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelDataPoint {
    /// Attribute labels (e.g. collection="docs").
    pub attributes: HashMap<String, String>,
    /// Timestamp in nanoseconds since epoch.
    pub time_unix_nano: u64,
    /// For counters/gauges: the value.
    pub value: Option<f64>,
    /// For histograms: bucket counts.
    pub bucket_counts: Option<Vec<u64>>,
    /// For histograms: explicit bounds.
    pub explicit_bounds: Option<Vec<f64>>,
    /// For histograms: sum.
    pub sum: Option<f64>,
    /// For histograms: count.
    pub count: Option<u64>,
}

/// Configuration for the OpenTelemetry exporter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtelExportConfig {
    /// OTLP endpoint (e.g. "http://localhost:4317").
    pub endpoint: String,
    /// Service name resource attribute.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// Additional resource attributes.
    pub resource_attributes: HashMap<String, String>,
    /// Export format: "otlp_json" or "otlp_proto" (only json implemented).
    pub format: String,
}

impl Default for OtelExportConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:4317".into(),
            service_name: "needle".into(),
            service_version: env!("CARGO_PKG_VERSION").into(),
            resource_attributes: HashMap::new(),
            format: "otlp_json".into(),
        }
    }
}

impl OtelExportConfig {
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }
}

/// Exports metrics from `ObservabilitySuite` in OpenTelemetry-compatible format.
pub struct OtelExporter {
    config: OtelExportConfig,
}

impl OtelExporter {
    pub fn new(config: OtelExportConfig) -> Self {
        Self { config }
    }

    /// Collect all current metrics from the suite as OTel metric points.
    pub fn collect(&self, suite: &ObservabilitySuite) -> Vec<OtelMetricPoint> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let resource = self.resource_attributes();
        let mut metrics = Vec::new();

        // Latency histograms
        let histograms = suite.latency_histograms.read();
        for (collection, hist) in histograms.iter() {
            let mut attrs = HashMap::new();
            attrs.insert("collection".into(), collection.clone());

            metrics.push(OtelMetricPoint {
                name: "needle.query.latency_us".into(),
                description: "Query latency in microseconds".into(),
                metric_type: "histogram".into(),
                resource: resource.clone(),
                data_points: vec![OtelDataPoint {
                    attributes: attrs,
                    time_unix_nano: now_ns,
                    value: None,
                    bucket_counts: Some(hist.buckets.iter().map(|(_, c)| *c).collect()),
                    explicit_bounds: Some(hist.buckets.iter().map(|(b, _)| *b).collect()),
                    sum: Some(hist.sum_us as f64),
                    count: Some(hist.total_count),
                }],
            });
        }

        // Error counters
        let errors = suite.error_counts.read();
        for (key, count) in errors.iter() {
            let parts: Vec<&str> = key.splitn(2, ':').collect();
            let mut attrs = HashMap::new();
            if parts.len() == 2 {
                attrs.insert("collection".into(), parts[0].to_string());
                attrs.insert("error_type".into(), parts[1].to_string());
            } else {
                attrs.insert("key".into(), key.clone());
            }

            metrics.push(OtelMetricPoint {
                name: "needle.errors.total".into(),
                description: "Total error count".into(),
                metric_type: "counter".into(),
                resource: resource.clone(),
                data_points: vec![OtelDataPoint {
                    attributes: attrs,
                    time_unix_nano: now_ns,
                    value: Some(*count as f64),
                    bucket_counts: None,
                    explicit_bounds: None,
                    sum: None,
                    count: None,
                }],
            });
        }

        // Index health gauges
        let health = suite.index_health.read();
        for (collection, h) in health.iter() {
            let mut attrs = HashMap::new();
            attrs.insert("collection".into(), collection.clone());

            metrics.push(OtelMetricPoint {
                name: "needle.index.balance_score".into(),
                description: "Balance score of the index graph".into(),
                metric_type: "gauge".into(),
                resource: resource.clone(),
                data_points: vec![OtelDataPoint {
                    attributes: attrs.clone(),
                    time_unix_nano: now_ns,
                    value: Some(h.balance_score as f64),
                    bucket_counts: None,
                    explicit_bounds: None,
                    sum: None,
                    count: None,
                }],
            });

            metrics.push(OtelMetricPoint {
                name: "needle.index.vector_count".into(),
                description: "Number of vectors in the index".into(),
                metric_type: "gauge".into(),
                resource: resource.clone(),
                data_points: vec![OtelDataPoint {
                    attributes: attrs,
                    time_unix_nano: now_ns,
                    value: Some(h.vector_count as f64),
                    bucket_counts: None,
                    explicit_bounds: None,
                    sum: None,
                    count: None,
                }],
            });
        }

        metrics
    }

    /// Export metrics as OTLP JSON string.
    pub fn export_json(&self, suite: &ObservabilitySuite) -> String {
        let metrics = self.collect(suite);
        serde_json::to_string_pretty(&metrics).unwrap_or_default()
    }

    fn resource_attributes(&self) -> HashMap<String, String> {
        let mut attrs = self.config.resource_attributes.clone();
        attrs.insert("service.name".into(), self.config.service_name.clone());
        attrs.insert("service.version".into(), self.config.service_version.clone());
        attrs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_histogram_observe() {
        let mut h = LatencyHistogram::default();
        h.observe(250);
        h.observe(750);
        h.observe(1500);

        assert_eq!(h.total_count, 3);
        assert_eq!(h.sum_us, 2500);
        assert_eq!(h.min_us, 250);
        assert_eq!(h.max_us, 1500);

        // bucket 100 µs → 0 (250 > 100)
        assert_eq!(h.buckets[0].1, 0);
        // bucket 500 µs → 1 (250 ≤ 500)
        assert_eq!(h.buckets[1].1, 1);
        // bucket 1000 µs → 2 (250,750 ≤ 1000)
        assert_eq!(h.buckets[2].1, 2);
        // bucket 2500 µs → 3
        assert_eq!(h.buckets[3].1, 3);
    }

    #[test]
    fn test_latency_histogram_percentile() {
        let mut h = LatencyHistogram::new(&[100.0, 500.0, 1000.0]);
        // Insert 100 values ≤ 100 µs
        for _ in 0..100 {
            h.observe(50);
        }
        // p50 should be within the first bucket
        let p50 = h.percentile(0.50);
        assert!(p50 <= 100.0, "p50={} should be <= 100", p50);

        // Insert 100 values in (500, 1000] range
        for _ in 0..100 {
            h.observe(800);
        }
        // p99 should be in the higher range
        let p99 = h.percentile(0.99);
        assert!(p99 > 100.0, "p99={} should be > 100", p99);
    }

    #[test]
    fn test_recall_estimation() {
        let mut est = RecallEstimator::new(10);
        let approx = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let exact = vec!["a".into(), "b".into(), "e".into(), "f".into()];
        let recall = est.estimate(&approx, &exact);
        // Jaccard: intersection={a,b}=2, union={a,b,c,d,e,f}=6 → 2/6 ≈ 0.333
        assert!((recall - 2.0 / 6.0).abs() < 1e-5, "recall={}", recall);
        assert!(est.last_measured.is_some());
    }

    #[test]
    fn test_recall_perfect() {
        let mut est = RecallEstimator::new(10);
        let results: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let recall = est.estimate(&results, &results);
        assert!((recall - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_recall_zero_overlap() {
        let mut est = RecallEstimator::new(10);
        let approx: Vec<String> = vec!["a".into(), "b".into()];
        let exact: Vec<String> = vec!["c".into(), "d".into()];
        let recall = est.estimate(&approx, &exact);
        assert!((recall - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_index_health_defaults() {
        let health = IndexHealthMetrics::default();
        assert_eq!(health.fragmentation_ratio, 0.0);
        assert_eq!(health.balance_score, 1.0);
        assert_eq!(health.memory_usage_bytes, 0);
        assert_eq!(health.vector_count, 0);
        assert_eq!(health.deleted_count, 0);
        assert!(health.last_compaction.is_none());
    }

    #[test]
    fn test_alert_rule_greater_than() {
        let suite = ObservabilitySuite::new();
        // Record high latency
        suite.record_query_latency("coll", 200_000);

        suite.add_alert_rule(AlertRule {
            name: "high_latency".into(),
            metric: MetricType::QueryLatencyP99,
            condition: AlertCondition::GreaterThan(50_000.0),
            severity: AlertSeverity::Critical,
            cooldown: Duration::from_secs(0),
            last_fired: None,
        });

        let alerts = suite.check_alerts();
        assert!(!alerts.is_empty(), "expected an alert to fire");
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_rule_less_than() {
        let suite = ObservabilitySuite::new();
        // Set low recall
        let approx: Vec<String> = vec!["a".into()];
        let exact: Vec<String> = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        suite.estimate_recall("coll", &approx, &exact);

        suite.add_alert_rule(AlertRule {
            name: "low_recall".into(),
            metric: MetricType::RecallEstimate,
            condition: AlertCondition::LessThan(0.9),
            severity: AlertSeverity::Warning,
            cooldown: Duration::from_secs(0),
            last_fired: None,
        });

        let alerts = suite.check_alerts();
        assert!(!alerts.is_empty(), "expected a low-recall alert");
        assert_eq!(alerts[0].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_alert_acknowledged() {
        let suite = ObservabilitySuite::new();
        suite.record_query_latency("coll", 200_000);

        suite.add_alert_rule(AlertRule {
            name: "latency_ack_test".into(),
            metric: MetricType::QueryLatencyP99,
            condition: AlertCondition::GreaterThan(1.0),
            severity: AlertSeverity::Info,
            cooldown: Duration::from_secs(0),
            last_fired: None,
        });

        let alerts = suite.check_alerts();
        assert!(!alerts.is_empty());

        let alert_id = alerts[0].alert_id.clone();
        suite.acknowledge_alert(&alert_id);

        let active = suite.active_alerts();
        assert!(
            active.iter().all(|a| a.alert_id != alert_id),
            "acknowledged alert should not appear as active"
        );
    }

    #[test]
    fn test_record_query_latency() {
        let suite = ObservabilitySuite::new();
        suite.record_query_latency("my_coll", 500);
        suite.record_query_latency("my_coll", 1500);

        let h = suite.get_latency_histogram("my_coll").unwrap();
        assert_eq!(h.total_count, 2);
        assert_eq!(h.min_us, 500);
        assert_eq!(h.max_us, 1500);

        assert!(suite.get_latency_histogram("nonexistent").is_none());
    }

    #[test]
    fn test_grafana_dashboard_generation() {
        let suite = ObservabilitySuite::new();
        let collections = vec!["docs".to_string(), "images".to_string()];
        let json = suite.generate_grafana_dashboard(&collections);

        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["uid"], "needle-observability");
        assert_eq!(parsed["title"], "Needle Vector Database — Observability");

        let panels = parsed["panels"].as_array().unwrap();
        // 2 histogram + 1 gauge + 1 timeseries + 1 table = 5
        assert_eq!(panels.len(), 5);
        assert_eq!(panels[0]["type"], "histogram");
        assert!(panels[0]["title"].as_str().unwrap().contains("docs"));
        assert_eq!(panels[2]["type"], "gauge");
        assert_eq!(panels[3]["type"], "timeseries");
        assert_eq!(panels[4]["type"], "table");
    }

    #[test]
    fn test_error_counting() {
        let suite = ObservabilitySuite::new();
        suite.record_error("coll1", "timeout");
        suite.record_error("coll1", "timeout");
        suite.record_error("coll1", "dimension_mismatch");
        suite.record_error("coll2", "timeout");

        let counts = suite.error_counts.read();
        assert_eq!(*counts.get("coll1:timeout").unwrap(), 2);
        assert_eq!(*counts.get("coll1:dimension_mismatch").unwrap(), 1);
        assert_eq!(*counts.get("coll2:timeout").unwrap(), 1);
    }

    #[test]
    fn test_otel_exporter_collect() {
        let suite = ObservabilitySuite::new();
        suite.record_query_latency("docs", 500);
        suite.record_query_latency("docs", 2000);
        suite.record_error("docs", "timeout");
        suite.update_index_health("docs", IndexHealthMetrics {
            vector_count: 1000,
            fragmentation_ratio: 0.1,
            balance_score: 0.9,
            memory_usage_bytes: 50000,
            disk_usage_bytes: 0,
            deleted_count: 10,
            avg_connections_per_node: 12.0,
            last_compaction: None,
        });

        let exporter = OtelExporter::new(OtelExportConfig::default());
        let metrics = exporter.collect(&suite);

        // Should have latency histogram + error counter + recall gauge + vector_count gauge
        assert!(metrics.len() >= 4);

        let latency = metrics.iter().find(|m| m.name == "needle.query.latency_us").unwrap();
        assert_eq!(latency.metric_type, "histogram");
        assert_eq!(latency.data_points[0].count, Some(2));

        let errors = metrics.iter().find(|m| m.name == "needle.errors.total").unwrap();
        assert_eq!(errors.metric_type, "counter");
        assert_eq!(errors.data_points[0].value, Some(1.0));

        let recall = metrics.iter().find(|m| m.name == "needle.index.balance_score").unwrap();
        assert_eq!(recall.metric_type, "gauge");
    }

    #[test]
    fn test_otel_export_json() {
        let suite = ObservabilitySuite::new();
        suite.record_query_latency("test", 100);

        let exporter = OtelExporter::new(
            OtelExportConfig::new("http://localhost:4317")
                .with_service_name("needle-test"),
        );
        let json = exporter.export_json(&suite);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed.is_array());

        let arr = parsed.as_array().unwrap();
        assert!(!arr.is_empty());
        assert!(arr[0]["resource"]["service.name"].as_str().unwrap().contains("needle-test"));
    }

    #[test]
    fn test_otel_config_defaults() {
        let config = OtelExportConfig::default();
        assert_eq!(config.endpoint, "http://localhost:4317");
        assert_eq!(config.service_name, "needle");
        assert_eq!(config.format, "otlp_json");
    }
}
