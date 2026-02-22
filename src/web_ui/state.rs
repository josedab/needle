#![cfg_attr(test, allow(clippy::unwrap_used))]

use crate::database::Database;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Web UI server
#[derive(Debug, Clone)]
pub struct WebUiConfig {
    /// Address to bind the server to
    pub addr: SocketAddr,
    /// Application title shown in the browser
    pub title: String,
    /// Whether to enable the query playground
    pub enable_query_playground: bool,
    /// Refresh interval for auto-updating dashboards (in seconds)
    pub refresh_interval: u64,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self {
            addr: std::net::SocketAddr::from(([127, 0, 0, 1], 8081)),
            title: "Needle Dashboard".to_string(),
            enable_query_playground: true,
            refresh_interval: 30,
        }
    }
}

impl WebUiConfig {
    /// Create a new configuration with the specified address
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.parse().unwrap_or_else(|_| {
                std::net::SocketAddr::from(([127, 0, 0, 1], 8081))
            }),
            ..Default::default()
        }
    }

    /// Set a custom title for the dashboard
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Enable or disable the query playground
    pub fn with_query_playground(mut self, enabled: bool) -> Self {
        self.enable_query_playground = enabled;
        self
    }
}

// ============================================================================
// Application State
// ============================================================================

/// Shared state for the Web UI application
pub struct WebUiState {
    /// Reference to the Needle database
    pub db: RwLock<Database>,
    /// Configuration
    pub config: WebUiConfig,
    /// Server start time for uptime calculation
    pub start_time: u64,
}

impl WebUiState {
    /// Create a new Web UI state
    pub fn new(db: Database, config: WebUiConfig) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            db: RwLock::new(db),
            config,
            start_time,
        }
    }

    /// Get the server uptime in seconds
    pub fn uptime(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now - self.start_time
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Query parameters for search requests
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    /// Collection to search in
    pub collection: Option<String>,
    /// Query vector as comma-separated values
    pub vector: Option<String>,
    /// Number of results to return
    pub k: Option<usize>,
}

/// API statistics response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    /// Server health status
    pub healthy: bool,
    /// Server uptime in seconds
    pub uptime_seconds: u64,
    /// Total number of collections
    pub total_collections: usize,
    /// Total number of vectors across all collections
    pub total_vectors: usize,
    /// Per-collection statistics
    pub collections: Vec<CollectionStatsResponse>,
    /// Server version
    pub version: String,
}

/// Per-collection statistics
#[derive(Debug, Serialize)]
pub struct CollectionStatsResponse {
    /// Collection name
    pub name: String,
    /// Number of vectors
    pub vector_count: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Number of deleted vectors pending compaction
    pub deleted_count: usize,
    /// Whether compaction is needed
    pub needs_compaction: bool,
}

// ============================================================================
// Visual Query Builder & Admin Dashboard (Next-Gen)
// ============================================================================

/// Visual query builder state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualQueryState {
    /// Selected collection
    pub collection: Option<String>,
    /// Query vector (can be pasted or generated)
    pub query_vector: Option<Vec<f32>>,
    /// Number of results
    pub limit: usize,
    /// Filter conditions
    pub filters: Vec<FilterCondition>,
    /// Distance function
    pub distance: String,
    /// Include metadata in results
    pub include_metadata: bool,
    /// Use HNSW index
    pub use_index: bool,
}

impl Default for VisualQueryState {
    fn default() -> Self {
        Self {
            collection: None,
            query_vector: None,
            limit: 10,
            filters: Vec::new(),
            distance: "cosine".to_string(),
            include_metadata: true,
            use_index: true,
        }
    }
}

/// Filter condition for the visual builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field name
    pub field: String,
    /// Operator (eq, ne, gt, gte, lt, lte, in, contains)
    pub operator: String,
    /// Value
    pub value: serde_json::Value,
}

impl FilterCondition {
    /// Convert to Filter
    pub fn to_filter(&self) -> Option<crate::metadata::Filter> {
        Some(match self.operator.as_str() {
            "eq" => crate::metadata::Filter::eq(self.field.clone(), self.value.clone()),
            "ne" => crate::metadata::Filter::ne(self.field.clone(), self.value.clone()),
            "gt" => crate::metadata::Filter::gt(self.field.clone(), self.value.clone()),
            "gte" => crate::metadata::Filter::gte(self.field.clone(), self.value.clone()),
            "lt" => crate::metadata::Filter::lt(self.field.clone(), self.value.clone()),
            "lte" => crate::metadata::Filter::lte(self.field.clone(), self.value.clone()),
            "in" => {
                if let serde_json::Value::Array(arr) = &self.value {
                    crate::metadata::Filter::is_in(self.field.clone(), arr.clone())
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    }
}

/// Admin dashboard section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminSection {
    /// Section name
    pub name: String,
    /// Section description
    pub description: String,
    /// Available actions
    pub actions: Vec<AdminAction>,
}

/// Admin action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminAction {
    /// Action ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// HTTP method
    pub method: String,
    /// Endpoint
    pub endpoint: String,
    /// Required parameters
    pub params: Vec<ActionParam>,
    /// Whether action is dangerous
    pub dangerous: bool,
}

/// Action parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionParam {
    /// Parameter name
    pub name: String,
    /// Parameter type (string, number, boolean, array)
    pub param_type: String,
    /// Whether required
    pub required: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Description
    pub description: String,
}

// ============================================================================
// Alerting Configuration
// ============================================================================

/// Alerting configuration for observability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerting
    pub enabled: bool,
    /// Latency threshold in milliseconds (alert if p99 exceeds this)
    pub latency_threshold_ms: f64,
    /// Minimum recall threshold (alert if below this)
    pub min_recall: f32,
    /// Maximum error rate (fraction, alert if above this)
    pub max_error_rate: f32,
    /// Webhook URL for notifications
    pub webhook_url: Option<String>,
    /// Check interval in seconds
    pub check_interval_secs: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            latency_threshold_ms: 100.0,
            min_recall: 0.85,
            max_error_rate: 0.05,
            webhook_url: None,
            check_interval_secs: 60,
        }
    }
}

/// An alert that has been triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Timestamp (unix seconds)
    pub timestamp: u64,
    /// Whether this alert has been acknowledged
    pub acknowledged: bool,
}

/// Alert severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

// ============================================================================
// Real-Time Index Monitoring Dashboard
// ============================================================================

/// Health score for a single collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionHealthScore {
    /// Collection name.
    pub name: String,
    /// Overall health score (0.0-1.0, higher is better).
    pub score: f64,
    /// Index fragmentation (fraction of deleted vectors).
    pub fragmentation: f64,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
    /// Vector count.
    pub vector_count: usize,
    /// Whether compaction is recommended.
    pub needs_compaction: bool,
    /// Dimensional density (vectors per dimension).
    pub density: f64,
}

/// Aggregated metrics snapshot for the monitoring dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSnapshot {
    /// Timestamp of this snapshot.
    pub timestamp: u64,
    /// Total collections.
    pub total_collections: usize,
    /// Total vectors across all collections.
    pub total_vectors: usize,
    /// Total estimated memory usage.
    pub total_memory_bytes: usize,
    /// Per-collection health scores.
    pub health_scores: Vec<CollectionHealthScore>,
    /// Overall system health (0.0-1.0).
    pub system_health: f64,
    /// Server uptime in seconds.
    pub uptime_secs: u64,
}

/// Latency histogram bucket for heatmap visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBucket {
    /// Bucket label (e.g., "0-1ms", "1-5ms").
    pub label: String,
    /// Upper bound in milliseconds.
    pub upper_bound_ms: f64,
    /// Number of operations in this bucket.
    pub count: u64,
}

/// Latency heatmap data for dashboard visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHeatmap {
    /// Histogram buckets.
    pub buckets: Vec<LatencyBucket>,
    /// Total recorded operations.
    pub total_ops: u64,
    /// P50 latency estimate in ms.
    pub p50_ms: f64,
    /// P95 latency estimate in ms.
    pub p95_ms: f64,
    /// P99 latency estimate in ms.
    pub p99_ms: f64,
}

impl LatencyHeatmap {
    /// Create a new heatmap with default buckets.
    pub fn new() -> Self {
        let bucket_bounds = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0];
        let buckets = bucket_bounds
            .windows(2)
            .map(|w| LatencyBucket {
                label: format!("{}-{}ms", w[0], w[1]),
                upper_bound_ms: w[1],
                count: 0,
            })
            .collect();
        Self {
            buckets,
            total_ops: 0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
        }
    }

    /// Record a latency observation.
    pub fn record(&mut self, latency_ms: f64) {
        self.total_ops += 1;
        for bucket in &mut self.buckets {
            if latency_ms <= bucket.upper_bound_ms {
                bucket.count += 1;
                return;
            }
        }
        // Overflow: add to last bucket
        if let Some(last) = self.buckets.last_mut() {
            last.count += 1;
        }
    }

    /// Estimate percentile from histogram.
    pub fn compute_percentiles(&mut self) {
        if self.total_ops == 0 {
            return;
        }

        let percentiles = [0.50, 0.95, 0.99];
        let mut results = [0.0f64; 3];

        for (i, &pct) in percentiles.iter().enumerate() {
            let target_count = (self.total_ops as f64 * pct).ceil() as u64;
            let mut running = 0u64;
            for bucket in &self.buckets {
                running += bucket.count;
                if running >= target_count {
                    results[i] = bucket.upper_bound_ms;
                    break;
                }
            }
        }

        self.p50_ms = results[0];
        self.p95_ms = results[1];
        self.p99_ms = results[2];
    }
}

impl Default for LatencyHeatmap {
    fn default() -> Self {
        Self::new()
    }
}
