//! Metric types for telemetry.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metric types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter (monotonically increasing).
    Counter(u64),
    /// Gauge (can go up or down).
    Gauge(f64),
    /// Histogram (distribution).
    Histogram(Vec<f64>),
}

/// A metric measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name.
    pub name: String,
    /// Metric value.
    pub value: MetricValue,
    /// Metric labels.
    pub labels: HashMap<String, String>,
    /// Timestamp.
    pub timestamp: u64,
}

/// Histogram statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    /// Sample count.
    pub count: usize,
    /// Sum of all values.
    pub sum: f64,
    /// Mean value.
    pub mean: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// 50th percentile.
    pub p50: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}
