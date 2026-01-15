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
