//! Prometheus-compatible metrics for Needle
//!
//! This module provides observability into database operations using Prometheus metrics,
//! enabling monitoring and alerting for production deployments.
//!
//! # Metrics
//!
//! - **operations_total**: Counter for total operations by type (insert, search, delete)
//! - **errors_total**: Counter for errors by type
//! - **operation_duration_seconds**: Histogram of operation latencies
//! - **collection_vectors_total**: Gauge of vectors per collection
//! - **collection_dimensions**: Gauge of dimensions per collection
//! - **index_health**: Gauge indicating index health status
//!
//! # Usage
//!
//! ```rust,ignore
//! use needle::metrics::{metrics, MetricsGuard};
//!
//! // Record an operation
//! let _guard = MetricsGuard::new("search", "my_collection");
//! // ... perform search ...
//! // Guard records duration on drop
//!
//! // Export metrics for Prometheus scraping
//! let output = metrics().export();
//! ```
//!
//! # Integration
//!
//! When using the HTTP server (feature: server), metrics are automatically exposed
//! at the `/metrics` endpoint for Prometheus scraping.

use prometheus::{
    register_counter_vec, register_gauge_vec, register_histogram_vec,
    CounterVec, GaugeVec, HistogramVec, Encoder, TextEncoder,
};
use std::sync::OnceLock;
use std::time::Instant;

static METRICS: OnceLock<NeedleMetrics> = OnceLock::new();
static HTTP_METRICS: OnceLock<HttpMetrics> = OnceLock::new();

/// Get or initialize the global metrics instance
pub fn metrics() -> &'static NeedleMetrics {
    METRICS.get_or_init(NeedleMetrics::new)
}

/// Get or initialize the HTTP metrics instance
pub fn http_metrics() -> &'static HttpMetrics {
    HTTP_METRICS.get_or_init(HttpMetrics::new)
}

/// HTTP server metrics for request tracking
pub struct HttpMetrics {
    /// Total HTTP requests by method, path, and status
    pub requests_total: CounterVec,
    /// HTTP request duration histogram by method and path
    pub request_duration_seconds: HistogramVec,
    /// Currently active HTTP requests
    pub requests_in_flight: GaugeVec,
}

impl HttpMetrics {
    fn new() -> Self {
        let requests_total = register_counter_vec!(
            "needle_http_requests_total",
            "Total number of HTTP requests",
            &["method", "path", "status"]
        )
        .expect("Failed to create HTTP requests counter");

        let request_duration_seconds = register_histogram_vec!(
            "needle_http_request_duration_seconds",
            "HTTP request duration in seconds",
            &["method", "path"],
            vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        .expect("Failed to create HTTP duration histogram");

        let requests_in_flight = register_gauge_vec!(
            "needle_http_requests_in_flight",
            "Number of HTTP requests currently being processed",
            &["method"]
        )
        .expect("Failed to create requests in flight gauge");

        Self {
            requests_total,
            request_duration_seconds,
            requests_in_flight,
        }
    }

    /// Start timing an HTTP request, returns a guard that records metrics on drop
    pub fn start_request(&self, method: &str, path: &str) -> HttpRequestTimer<'_> {
        self.requests_in_flight
            .with_label_values(&[method])
            .inc();

        HttpRequestTimer {
            method: method.to_string(),
            path: normalize_path(path),
            start: Instant::now(),
            status: None,
            metrics: self,
        }
    }
}

/// Normalize path for metrics to avoid cardinality explosion
/// Replaces dynamic segments like UUIDs and IDs with placeholders
fn normalize_path(path: &str) -> String {
    let parts: Vec<&str> = path.split('/').collect();
    let mut normalized = Vec::with_capacity(parts.len());

    // Known static segments that should not be replaced
    const STATIC_SEGMENTS: &[&str] = &["batch", "search", "upsert", "metadata", "compact", "export"];

    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        // Check if this looks like a collection name or vector ID (after /collections or /vectors)
        let prev = if i > 0 { parts.get(i - 1) } else { None };
        if prev == Some(&"collections") && !part.starts_with(':') {
            normalized.push(":name");
        } else if prev == Some(&"vectors") && !part.starts_with(':') && !STATIC_SEGMENTS.contains(part) {
            normalized.push(":id");
        } else {
            normalized.push(part);
        }
    }

    format!("/{}", normalized.join("/"))
}

/// Timer for HTTP requests that records duration and status on drop
pub struct HttpRequestTimer<'a> {
    method: String,
    path: String,
    start: Instant,
    status: Option<u16>,
    metrics: &'a HttpMetrics,
}

impl<'a> HttpRequestTimer<'a> {
    /// Set the response status code
    pub fn set_status(&mut self, status: u16) {
        self.status = Some(status);
    }
}

impl<'a> Drop for HttpRequestTimer<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();
        let status = self.status.unwrap_or(0).to_string();

        // Record request count
        self.metrics.requests_total
            .with_label_values(&[&self.method, &self.path, &status])
            .inc();

        // Record duration
        self.metrics.request_duration_seconds
            .with_label_values(&[&self.method, &self.path])
            .observe(duration);

        // Decrement in-flight counter
        self.metrics.requests_in_flight
            .with_label_values(&[&self.method])
            .dec();
    }
}

/// All metrics for Needle
pub struct NeedleMetrics {
    // Counters
    pub operations_total: CounterVec,
    pub errors_total: CounterVec,

    // Histograms
    pub operation_duration_seconds: HistogramVec,
    pub search_result_count: HistogramVec,
    /// Number of nodes visited during HNSW search
    pub hnsw_visited_nodes: HistogramVec,
    /// Number of layers traversed during HNSW search
    pub hnsw_layers_traversed: HistogramVec,

    // Gauges
    pub collection_vectors: GaugeVec,
    pub collection_deleted_vectors: GaugeVec,
    pub collection_dimensions: GaugeVec,
    pub index_levels: GaugeVec,
    pub memory_bytes: GaugeVec,
}

impl NeedleMetrics {
    fn new() -> Self {
        let operations_total = register_counter_vec!(
            "needle_operations_total",
            "Total number of operations",
            &["collection", "operation"]
        )
        .expect("Failed to create operations counter");

        let errors_total = register_counter_vec!(
            "needle_errors_total",
            "Total number of errors",
            &["collection", "operation", "error_type"]
        )
        .expect("Failed to create errors counter");

        let operation_duration_seconds = register_histogram_vec!(
            "needle_operation_duration_seconds",
            "Operation duration in seconds",
            &["collection", "operation"],
            vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        .expect("Failed to create duration histogram");

        let search_result_count = register_histogram_vec!(
            "needle_search_result_count",
            "Number of results returned by search",
            &["collection"],
            vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0]
        )
        .expect("Failed to create result count histogram");

        let hnsw_visited_nodes = register_histogram_vec!(
            "needle_hnsw_visited_nodes",
            "Number of nodes visited during HNSW search",
            &["collection"],
            vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0]
        )
        .expect("Failed to create HNSW visited nodes histogram");

        let hnsw_layers_traversed = register_histogram_vec!(
            "needle_hnsw_layers_traversed",
            "Number of HNSW layers traversed during search",
            &["collection"],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0]
        )
        .expect("Failed to create HNSW layers traversed histogram");

        let collection_vectors = register_gauge_vec!(
            "needle_collection_vectors",
            "Number of vectors in collection",
            &["collection"]
        )
        .expect("Failed to create vectors gauge");

        let collection_deleted_vectors = register_gauge_vec!(
            "needle_collection_deleted_vectors",
            "Number of soft-deleted vectors pending compaction",
            &["collection"]
        )
        .expect("Failed to create deleted vectors gauge");

        let collection_dimensions = register_gauge_vec!(
            "needle_collection_dimensions",
            "Vector dimensions for collection",
            &["collection"]
        )
        .expect("Failed to create dimensions gauge");

        let index_levels = register_gauge_vec!(
            "needle_index_levels",
            "Number of HNSW index levels",
            &["collection"]
        )
        .expect("Failed to create index levels gauge");

        let memory_bytes = register_gauge_vec!(
            "needle_memory_bytes",
            "Estimated memory usage in bytes",
            &["collection", "component"]
        )
        .expect("Failed to create memory gauge");

        Self {
            operations_total,
            errors_total,
            operation_duration_seconds,
            search_result_count,
            hnsw_visited_nodes,
            hnsw_layers_traversed,
            collection_vectors,
            collection_deleted_vectors,
            collection_dimensions,
            index_levels,
            memory_bytes,
        }
    }

    /// Record an operation start, returns a guard that records duration on drop
    pub fn operation(&self, collection: &str, operation: &str) -> OperationTimer {
        self.operations_total
            .with_label_values(&[collection, operation])
            .inc();

        OperationTimer {
            collection: collection.to_string(),
            operation: operation.to_string(),
            start: Instant::now(),
            histogram: self.operation_duration_seconds.clone(),
        }
    }

    /// Record an error
    pub fn error(&self, collection: &str, operation: &str, error_type: &str) {
        self.errors_total
            .with_label_values(&[collection, operation, error_type])
            .inc();
    }

    /// Record search result count
    pub fn record_search_results(&self, collection: &str, count: usize) {
        self.search_result_count
            .with_label_values(&[collection])
            .observe(count as f64);
    }

    /// Record HNSW search metrics
    pub fn record_hnsw_search(&self, collection: &str, visited_nodes: usize, layers_traversed: usize) {
        self.hnsw_visited_nodes
            .with_label_values(&[collection])
            .observe(visited_nodes as f64);
        self.hnsw_layers_traversed
            .with_label_values(&[collection])
            .observe(layers_traversed as f64);
    }

    /// Update collection metrics
    pub fn update_collection(&self, collection: &str, vectors: usize, deleted: usize, dims: usize) {
        self.collection_vectors
            .with_label_values(&[collection])
            .set(vectors as f64);
        self.collection_deleted_vectors
            .with_label_values(&[collection])
            .set(deleted as f64);
        self.collection_dimensions
            .with_label_values(&[collection])
            .set(dims as f64);
    }

    /// Update index metrics
    pub fn update_index(&self, collection: &str, levels: usize) {
        self.index_levels
            .with_label_values(&[collection])
            .set(levels as f64);
    }

    /// Update memory metrics
    pub fn update_memory(&self, collection: &str, vectors: usize, metadata: usize, index: usize) {
        self.memory_bytes
            .with_label_values(&[collection, "vectors"])
            .set(vectors as f64);
        self.memory_bytes
            .with_label_values(&[collection, "metadata"])
            .set(metadata as f64);
        self.memory_bytes
            .with_label_values(&[collection, "index"])
            .set(index as f64);
    }

    /// Export metrics in Prometheus text format
    pub fn export(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }
}

/// Timer that records duration when dropped
pub struct OperationTimer {
    collection: String,
    operation: String,
    start: Instant,
    histogram: HistogramVec,
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed().as_secs_f64();
        self.histogram
            .with_label_values(&[&self.collection, &self.operation])
            .observe(duration);
    }
}

/// Metrics snapshot for a collection
#[derive(Debug, Clone)]
pub struct CollectionMetrics {
    pub name: String,
    pub vector_count: usize,
    pub deleted_count: usize,
    pub dimensions: usize,
    pub index_levels: usize,
    pub vector_memory_bytes: usize,
    pub metadata_memory_bytes: usize,
    pub index_memory_bytes: usize,
    pub total_memory_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let m = NeedleMetrics::new();

        // Record some operations
        {
            let _timer = m.operation("test_collection", "insert");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        m.error("test_collection", "insert", "dimension_mismatch");
        m.record_search_results("test_collection", 10);
        m.update_collection("test_collection", 100, 5, 128);

        let output = m.export();
        assert!(output.contains("needle_operations_total"));
        assert!(output.contains("needle_errors_total"));
    }

    #[test]
    fn test_normalize_path() {
        // Static paths should remain unchanged
        assert_eq!(normalize_path("/health"), "/health");
        assert_eq!(normalize_path("/collections"), "/collections");
        assert_eq!(normalize_path("/save"), "/save");

        // Collection names should be normalized
        assert_eq!(
            normalize_path("/collections/my_collection"),
            "/collections/:name"
        );
        assert_eq!(
            normalize_path("/collections/test-collection/search"),
            "/collections/:name/search"
        );
        assert_eq!(
            normalize_path("/collections/docs/vectors"),
            "/collections/:name/vectors"
        );

        // Vector IDs should be normalized
        assert_eq!(
            normalize_path("/collections/docs/vectors/vec-123"),
            "/collections/:name/vectors/:id"
        );
        assert_eq!(
            normalize_path("/collections/my_collection/vectors/some-id/metadata"),
            "/collections/:name/vectors/:id/metadata"
        );

        // Batch endpoints
        assert_eq!(
            normalize_path("/collections/test/vectors/batch"),
            "/collections/:name/vectors/batch"
        );
        assert_eq!(
            normalize_path("/collections/test/search/batch"),
            "/collections/:name/search/batch"
        );
    }

    #[test]
    fn test_http_metrics_creation() {
        let m = HttpMetrics::new();

        // Start a request timer
        {
            let mut timer = m.start_request("GET", "/collections/test/vectors/123");
            timer.set_status(200);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        // Export should include HTTP metrics
        let encoder = TextEncoder::new();
        let metric_families = prometheus::gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        let output = String::from_utf8(buffer).unwrap();

        assert!(output.contains("needle_http_requests_total"));
        assert!(output.contains("needle_http_request_duration_seconds"));
    }
}

// ============================================================================
// Grafana Dashboard Generation
// ============================================================================

/// Generate a Grafana dashboard JSON for Needle metrics
pub fn generate_grafana_dashboard(config: GrafanaDashboardConfig) -> String {
    let panels = vec![
        // Operations panel
        grafana_panel(
            1,
            "Operations per Second",
            "rate(needle_operations_total[5m])",
            "timeseries",
            GridPos { x: 0, y: 0, w: 12, h: 8 },
        ),
        // Errors panel
        grafana_panel(
            2,
            "Error Rate",
            "rate(needle_errors_total[5m])",
            "timeseries",
            GridPos { x: 12, y: 0, w: 12, h: 8 },
        ),
        // Latency panel
        grafana_panel(
            3,
            "Operation Latency (p95)",
            "histogram_quantile(0.95, rate(needle_operation_duration_seconds_bucket[5m]))",
            "timeseries",
            GridPos { x: 0, y: 8, w: 12, h: 8 },
        ),
        // Vector count panel
        grafana_panel(
            4,
            "Total Vectors",
            "sum(needle_collection_vectors_total)",
            "stat",
            GridPos { x: 12, y: 8, w: 6, h: 8 },
        ),
        // Memory usage panel
        grafana_panel(
            5,
            "Memory Usage",
            "sum(needle_collection_memory_bytes)",
            "gauge",
            GridPos { x: 18, y: 8, w: 6, h: 8 },
        ),
        // Search results panel
        grafana_panel(
            6,
            "Search Results per Query",
            "rate(needle_search_results_total[5m]) / rate(needle_operations_total{operation=\"search\"}[5m])",
            "timeseries",
            GridPos { x: 0, y: 16, w: 12, h: 8 },
        ),
        // HTTP requests panel
        grafana_panel(
            7,
            "HTTP Requests per Second",
            "rate(needle_http_requests_total[5m])",
            "timeseries",
            GridPos { x: 12, y: 16, w: 12, h: 8 },
        ),
    ];

    let dashboard = serde_json::json!({
        "annotations": {
            "list": []
        },
        "editable": true,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": null,
        "links": [],
        "liveNow": false,
        "panels": panels,
        "refresh": config.refresh_interval,
        "schemaVersion": 38,
        "tags": ["needle", "vector-db"],
        "templating": {
            "list": [
                {
                    "current": { "selected": false, "text": "All", "value": "$__all" },
                    "datasource": { "type": "prometheus", "uid": config.datasource_uid },
                    "definition": "label_values(needle_operations_total, collection)",
                    "hide": 0,
                    "includeAll": true,
                    "label": "Collection",
                    "multi": true,
                    "name": "collection",
                    "options": [],
                    "query": { "query": "label_values(needle_operations_total, collection)", "refId": "PrometheusVariableQueryEditor-VariableQuery" },
                    "refresh": 1,
                    "regex": "",
                    "skipUrlSync": false,
                    "sort": 1,
                    "type": "query"
                }
            ]
        },
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "timepicker": {},
        "timezone": "",
        "title": config.title,
        "uid": config.uid,
        "version": 1,
        "weekStart": ""
    });

    serde_json::to_string_pretty(&dashboard).unwrap_or_default()
}

/// Configuration for Grafana dashboard generation
#[derive(Debug, Clone)]
pub struct GrafanaDashboardConfig {
    /// Dashboard title
    pub title: String,
    /// Dashboard UID (for unique identification)
    pub uid: String,
    /// Prometheus datasource UID
    pub datasource_uid: String,
    /// Refresh interval (e.g., "5s", "30s", "1m")
    pub refresh_interval: String,
}

impl Default for GrafanaDashboardConfig {
    fn default() -> Self {
        Self {
            title: "Needle Vector Database".to_string(),
            uid: "needle-dashboard".to_string(),
            datasource_uid: "prometheus".to_string(),
            refresh_interval: "30s".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GridPos {
    x: u8,
    y: u8,
    w: u8,
    h: u8,
}

fn grafana_panel(id: u8, title: &str, expr: &str, panel_type: &str, grid: GridPos) -> serde_json::Value {
    serde_json::json!({
        "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
        "fieldConfig": {
            "defaults": {
                "color": { "mode": "palette-classic" },
                "custom": {
                    "axisCenteredZero": false,
                    "axisColorMode": "text",
                    "axisLabel": "",
                    "axisPlacement": "auto",
                    "barAlignment": 0,
                    "drawStyle": "line",
                    "fillOpacity": 10,
                    "gradientMode": "none",
                    "hideFrom": { "legend": false, "tooltip": false, "viz": false },
                    "lineInterpolation": "linear",
                    "lineWidth": 1,
                    "pointSize": 5,
                    "scaleDistribution": { "type": "linear" },
                    "showPoints": "never",
                    "spanNulls": false,
                    "stacking": { "group": "A", "mode": "none" },
                    "thresholdsStyle": { "mode": "off" }
                },
                "mappings": [],
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        { "color": "green", "value": null },
                        { "color": "red", "value": 80 }
                    ]
                },
                "unit": "short"
            },
            "overrides": []
        },
        "gridPos": { "h": grid.h, "w": grid.w, "x": grid.x, "y": grid.y },
        "id": id,
        "options": {
            "legend": { "calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": true },
            "tooltip": { "mode": "single", "sort": "none" }
        },
        "pluginVersion": "10.0.0",
        "targets": [{
            "datasource": { "type": "prometheus", "uid": "${DS_PROMETHEUS}" },
            "editorMode": "code",
            "expr": expr,
            "legendFormat": "__auto",
            "range": true,
            "refId": "A"
        }],
        "title": title,
        "type": panel_type
    })
}

// ============================================================================
// Alerting Rules Generation
// ============================================================================

/// Generate Prometheus alerting rules for Needle
pub fn generate_alerting_rules(config: AlertingConfig) -> String {
    let rules = vec![
        AlertRule {
            name: "NeedleHighErrorRate".to_string(),
            expr: format!(
                "rate(needle_errors_total[5m]) / rate(needle_operations_total[5m]) > {}",
                config.error_rate_threshold
            ),
            for_duration: "5m".to_string(),
            severity: "warning".to_string(),
            summary: "High error rate in Needle operations".to_string(),
            description: "Error rate is above {{ $value | printf \"%.2f\" }}% of operations".to_string(),
        },
        AlertRule {
            name: "NeedleHighLatency".to_string(),
            expr: format!(
                "histogram_quantile(0.95, rate(needle_operation_duration_seconds_bucket[5m])) > {}",
                config.latency_threshold_ms / 1000.0
            ),
            for_duration: "5m".to_string(),
            severity: "warning".to_string(),
            summary: "High latency in Needle operations".to_string(),
            description: "P95 latency is {{ $value | printf \"%.2f\" }}s".to_string(),
        },
        AlertRule {
            name: "NeedleHighMemoryUsage".to_string(),
            expr: format!(
                "sum(needle_collection_memory_bytes) > {}",
                config.memory_threshold_bytes
            ),
            for_duration: "10m".to_string(),
            severity: "warning".to_string(),
            summary: "High memory usage in Needle".to_string(),
            description: "Total memory usage is {{ $value | humanize1024 }}".to_string(),
        },
        AlertRule {
            name: "NeedleNoOperations".to_string(),
            expr: "rate(needle_operations_total[10m]) == 0".to_string(),
            for_duration: "15m".to_string(),
            severity: "info".to_string(),
            summary: "No Needle operations detected".to_string(),
            description: "No operations have been recorded for 15 minutes".to_string(),
        },
        AlertRule {
            name: "NeedleIndexUnhealthy".to_string(),
            expr: "needle_index_health < 1".to_string(),
            for_duration: "5m".to_string(),
            severity: "critical".to_string(),
            summary: "Needle index is unhealthy".to_string(),
            description: "Index health check is failing for collection {{ $labels.collection }}".to_string(),
        },
    ];

    let yaml = format!(
        r#"groups:
  - name: needle_alerts
    rules:
{}
"#,
        rules
            .iter()
            .map(|r| r.to_yaml())
            .collect::<Vec<_>>()
            .join("\n")
    );

    yaml
}

/// Configuration for alerting rules
#[derive(Debug, Clone)]
pub struct AlertingConfig {
    /// Error rate threshold (0.0 to 1.0)
    pub error_rate_threshold: f64,
    /// Latency threshold in milliseconds
    pub latency_threshold_ms: f64,
    /// Memory threshold in bytes
    pub memory_threshold_bytes: u64,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            error_rate_threshold: 0.05, // 5% error rate
            latency_threshold_ms: 100.0, // 100ms p95 latency
            memory_threshold_bytes: 8 * 1024 * 1024 * 1024, // 8GB
        }
    }
}

struct AlertRule {
    name: String,
    expr: String,
    for_duration: String,
    severity: String,
    summary: String,
    description: String,
}

impl AlertRule {
    fn to_yaml(&self) -> String {
        format!(
            r#"      - alert: {}
        expr: {}
        for: {}
        labels:
          severity: {}
        annotations:
          summary: "{}"
          description: "{}""#,
            self.name, self.expr, self.for_duration, self.severity, self.summary, self.description
        )
    }
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Simple anomaly detector for metrics
pub struct AnomalyDetector {
    /// Rolling window of values
    window: std::collections::VecDeque<f64>,
    /// Window size
    window_size: usize,
    /// Number of standard deviations for anomaly threshold
    threshold_sigmas: f64,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(window_size: usize, threshold_sigmas: f64) -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(window_size),
            window_size,
            threshold_sigmas,
        }
    }

    /// Add a value and check if it's anomalous
    pub fn check(&mut self, value: f64) -> AnomalyResult {
        if self.window.len() < self.window_size {
            self.window.push_back(value);
            return AnomalyResult {
                is_anomaly: false,
                value,
                mean: value,
                std_dev: 0.0,
                z_score: 0.0,
            };
        }

        let mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
        let variance = self.window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.window.len() as f64;
        let std_dev = variance.sqrt();

        let z_score = if std_dev > 0.0 {
            (value - mean) / std_dev
        } else {
            0.0
        };

        let is_anomaly = z_score.abs() > self.threshold_sigmas;

        // Update window
        self.window.pop_front();
        self.window.push_back(value);

        AnomalyResult {
            is_anomaly,
            value,
            mean,
            std_dev,
            z_score,
        }
    }

    /// Reset the detector
    pub fn reset(&mut self) {
        self.window.clear();
    }
}

/// Result of anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Whether the value is anomalous
    pub is_anomaly: bool,
    /// The value checked
    pub value: f64,
    /// Mean of the window
    pub mean: f64,
    /// Standard deviation of the window
    pub std_dev: f64,
    /// Z-score of the value
    pub z_score: f64,
}

#[cfg(test)]
mod dashboard_tests {
    use super::*;

    #[test]
    fn test_grafana_dashboard_generation() {
        let config = GrafanaDashboardConfig::default();
        let dashboard = generate_grafana_dashboard(config);
        
        assert!(dashboard.contains("Needle Vector Database"));
        assert!(dashboard.contains("needle_operations_total"));
        assert!(dashboard.contains("timeseries"));
    }

    #[test]
    fn test_alerting_rules_generation() {
        let config = AlertingConfig::default();
        let rules = generate_alerting_rules(config);
        
        assert!(rules.contains("NeedleHighErrorRate"));
        assert!(rules.contains("NeedleHighLatency"));
        assert!(rules.contains("severity: warning"));
    }

    #[test]
    fn test_anomaly_detector() {
        let mut detector = AnomalyDetector::new(10, 2.0);
        
        // Add normal values
        for i in 0..15 {
            let result = detector.check(100.0 + (i as f64 % 5.0));
            if i >= 10 {
                assert!(!result.is_anomaly);
            }
        }
        
        // Add anomalous value
        let result = detector.check(200.0);
        assert!(result.is_anomaly);
        assert!(result.z_score.abs() > 2.0);
    }
}
