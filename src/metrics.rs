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

/// Get or initialize the global metrics instance
pub fn metrics() -> &'static NeedleMetrics {
    METRICS.get_or_init(NeedleMetrics::new)
}

/// All metrics for Needle
pub struct NeedleMetrics {
    // Counters
    pub operations_total: CounterVec,
    pub errors_total: CounterVec,

    // Histograms
    pub operation_duration_seconds: HistogramVec,
    pub search_result_count: HistogramVec,

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
}
