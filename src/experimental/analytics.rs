//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Real-Time Analytics Dashboard
//!
//! Comprehensive analytics and monitoring for production Needle deployments.
//! Provides query pattern analysis, performance insights, slow query logging,
//! and alerting infrastructure.
//!
//! # Features
//!
//! - **Query Pattern Analysis**: Track common query patterns and their performance
//! - **Slow Query Log**: Automatically log queries exceeding latency thresholds
//! - **Performance Insights**: Identify bottlenecks and optimization opportunities
//! - **Real-Time Streaming**: Push analytics events to subscribers
//! - **Historical Aggregations**: Hourly, daily, and weekly rollups
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::analytics::{AnalyticsDashboard, AnalyticsConfig, QueryTracker};
//!
//! let config = AnalyticsConfig::default()
//!     .with_slow_query_threshold_ms(100)
//!     .with_retention_hours(168);
//!
//! let dashboard = AnalyticsDashboard::new(config);
//!
//! // Track a query
//! dashboard.track_query(QueryEvent {
//!     collection: "documents".to_string(),
//!     operation: "search",
//!     latency_ms: 45.0,
//!     result_count: 10,
//!     ..Default::default()
//! });
//!
//! // Get insights
//! let insights = dashboard.get_insights();
//! println!("QPS: {}", insights.current_qps);
//! println!("P99 latency: {}ms", insights.p99_latency_ms);
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the analytics dashboard
#[derive(Debug, Clone)]
pub struct AnalyticsConfig {
    /// Threshold in milliseconds for slow query logging
    pub slow_query_threshold_ms: f64,
    /// Maximum number of slow queries to retain
    pub max_slow_queries: usize,
    /// Maximum number of query patterns to track
    pub max_query_patterns: usize,
    /// Retention period for detailed events in hours
    pub retention_hours: u64,
    /// Enable query sampling (0.0-1.0)
    pub sample_rate: f64,
    /// Window size for rate calculations in seconds
    pub rate_window_seconds: u64,
    /// Enable detailed pattern analysis
    pub enable_pattern_analysis: bool,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            slow_query_threshold_ms: 100.0,
            max_slow_queries: 1000,
            max_query_patterns: 500,
            retention_hours: 24,
            sample_rate: 1.0,
            rate_window_seconds: 60,
            enable_pattern_analysis: true,
        }
    }
}

impl AnalyticsConfig {
    /// Set slow query threshold
    #[must_use]
    pub fn with_slow_query_threshold_ms(mut self, threshold: f64) -> Self {
        self.slow_query_threshold_ms = threshold;
        self
    }

    /// Set retention period in hours
    #[must_use]
    pub fn with_retention_hours(mut self, hours: u64) -> Self {
        self.retention_hours = hours;
        self
    }

    /// Set sample rate (0.0-1.0)
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }
}

// ============================================================================
// Query Events
// ============================================================================

/// A query event for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvent {
    /// Collection name
    pub collection: String,
    /// Operation type (search, insert, delete, etc.)
    pub operation: String,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Number of results returned
    pub result_count: usize,
    /// Top-k parameter if applicable
    pub k: Option<usize>,
    /// Whether a filter was used
    pub has_filter: bool,
    /// Filter complexity (number of conditions)
    pub filter_complexity: Option<usize>,
    /// HNSW nodes visited
    pub nodes_visited: Option<usize>,
    /// Timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: Option<Value>,
}

impl Default for QueryEvent {
    fn default() -> Self {
        Self {
            collection: String::new(),
            operation: String::new(),
            latency_ms: 0.0,
            result_count: 0,
            k: None,
            has_filter: false,
            filter_complexity: None,
            nodes_visited: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time before UNIX epoch")
                .as_secs(),
            metadata: None,
        }
    }
}

/// A slow query entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQuery {
    /// The query event
    pub event: QueryEvent,
    /// Why it was flagged as slow
    pub reason: String,
    /// Suggestions for optimization
    pub suggestions: Vec<String>,
}

// ============================================================================
// Query Patterns
// ============================================================================

/// A pattern of queries for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Pattern identifier (hash of collection + operation + filter shape)
    pub pattern_id: String,
    /// Collection name
    pub collection: String,
    /// Operation type
    pub operation: String,
    /// Whether this pattern uses filters
    pub uses_filter: bool,
    /// Typical k value
    pub typical_k: usize,
    /// Total count of queries matching this pattern
    pub count: u64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// P50 latency
    pub p50_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
    /// Average result count
    pub avg_result_count: f64,
    /// First seen timestamp
    pub first_seen: u64,
    /// Last seen timestamp
    pub last_seen: u64,
}

impl QueryPattern {
    fn new(event: &QueryEvent) -> Self {
        let pattern_id = Self::compute_pattern_id(event);
        Self {
            pattern_id,
            collection: event.collection.clone(),
            operation: event.operation.clone(),
            uses_filter: event.has_filter,
            typical_k: event.k.unwrap_or(10),
            count: 1,
            avg_latency_ms: event.latency_ms,
            p50_latency_ms: event.latency_ms,
            p95_latency_ms: event.latency_ms,
            p99_latency_ms: event.latency_ms,
            avg_result_count: event.result_count as f64,
            first_seen: event.timestamp,
            last_seen: event.timestamp,
        }
    }

    fn compute_pattern_id(event: &QueryEvent) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        event.collection.hash(&mut hasher);
        event.operation.hash(&mut hasher);
        event.has_filter.hash(&mut hasher);
        event.k.hash(&mut hasher);
        format!("pat_{:016x}", hasher.finish())
    }

    fn update(&mut self, event: &QueryEvent, latencies: &[f64]) {
        self.count += 1;
        self.last_seen = event.timestamp;
        
        // Update running average
        let n = self.count as f64;
        self.avg_latency_ms = self.avg_latency_ms * (n - 1.0) / n + event.latency_ms / n;
        self.avg_result_count = self.avg_result_count * (n - 1.0) / n + event.result_count as f64 / n;

        // Update percentiles from sampled latencies
        if !latencies.is_empty() {
            let mut sorted: Vec<f64> = latencies.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let len = sorted.len();
            self.p50_latency_ms = sorted[len * 50 / 100];
            self.p95_latency_ms = sorted[len * 95 / 100];
            self.p99_latency_ms = sorted[len.saturating_sub(1).max(len * 99 / 100)];
        }
    }
}

// ============================================================================
// Analytics Dashboard
// ============================================================================

/// Real-time analytics dashboard
pub struct AnalyticsDashboard {
    config: AnalyticsConfig,
    /// Recent events for rate calculation
    recent_events: RwLock<VecDeque<(Instant, QueryEvent)>>,
    /// Slow query log
    slow_queries: RwLock<VecDeque<SlowQuery>>,
    /// Query patterns
    patterns: RwLock<HashMap<String, (QueryPattern, Vec<f64>)>>,
    /// Aggregate counters
    counters: DashboardCounters,
    /// Per-collection stats
    collection_stats: RwLock<HashMap<String, CollectionAnalytics>>,
    /// Start time
    start_time: Instant,
}

struct DashboardCounters {
    total_queries: AtomicU64,
    total_inserts: AtomicU64,
    total_deletes: AtomicU64,
    total_errors: AtomicU64,
    slow_queries: AtomicU64,
    filtered_queries: AtomicU64,
}

impl Default for DashboardCounters {
    fn default() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            total_inserts: AtomicU64::new(0),
            total_deletes: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            slow_queries: AtomicU64::new(0),
            filtered_queries: AtomicU64::new(0),
        }
    }
}

/// Analytics for a single collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionAnalytics {
    /// Collection name
    pub name: String,
    /// Total queries
    pub total_queries: u64,
    /// Total inserts
    pub total_inserts: u64,
    /// Total deletes
    pub total_deletes: u64,
    /// Average search latency
    pub avg_search_latency_ms: f64,
    /// Average insert latency
    pub avg_insert_latency_ms: f64,
    /// Queries in last hour
    pub queries_last_hour: u64,
    /// Peak QPS observed
    pub peak_qps: f64,
}

impl AnalyticsDashboard {
    /// Create a new analytics dashboard
    pub fn new(config: AnalyticsConfig) -> Self {
        Self {
            config,
            recent_events: RwLock::new(VecDeque::new()),
            slow_queries: RwLock::new(VecDeque::new()),
            patterns: RwLock::new(HashMap::new()),
            counters: DashboardCounters::default(),
            collection_stats: RwLock::new(HashMap::new()),
            start_time: Instant::now(),
        }
    }

    /// Track a query event
    pub fn track_query(&self, event: QueryEvent) {
        // Update counters
        match event.operation.as_str() {
            "search" | "query" => {
                self.counters.total_queries.fetch_add(1, Ordering::Relaxed);
                if event.has_filter {
                    self.counters.filtered_queries.fetch_add(1, Ordering::Relaxed);
                }
            }
            "insert" | "upsert" => {
                self.counters.total_inserts.fetch_add(1, Ordering::Relaxed);
            }
            "delete" => {
                self.counters.total_deletes.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        // Check for slow query
        if event.latency_ms > self.config.slow_query_threshold_ms {
            self.counters.slow_queries.fetch_add(1, Ordering::Relaxed);
            self.log_slow_query(&event);
        }

        // Add to recent events
        {
            let mut recent = self.recent_events.write();
            recent.push_back((Instant::now(), event.clone()));
            
            // Trim old events
            let cutoff = Instant::now() - Duration::from_secs(self.config.rate_window_seconds);
            while recent.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
                recent.pop_front();
            }
        }

        // Update collection stats
        self.update_collection_stats(&event);

        // Update patterns if enabled
        if self.config.enable_pattern_analysis {
            self.update_pattern(&event);
        }
    }

    /// Log a slow query
    fn log_slow_query(&self, event: &QueryEvent) {
        let mut suggestions = Vec::new();

        // Generate suggestions based on the query
        if event.has_filter && event.filter_complexity.unwrap_or(0) > 5 {
            suggestions.push("Consider simplifying the filter or creating a specialized index".to_string());
        }
        if event.k.unwrap_or(0) > 100 {
            suggestions.push("High k value increases search time - consider pagination".to_string());
        }
        if event.nodes_visited.unwrap_or(0) > 1000 {
            suggestions.push("Many nodes visited - consider increasing ef_construction or M parameter".to_string());
        }
        if event.result_count == 0 {
            suggestions.push("No results returned - check if filter is too restrictive".to_string());
        }

        let slow = SlowQuery {
            event: event.clone(),
            reason: format!("Latency {}ms exceeds threshold {}ms", 
                event.latency_ms, self.config.slow_query_threshold_ms),
            suggestions,
        };

        let mut slow_queries = self.slow_queries.write();
        slow_queries.push_back(slow);
        
        // Trim to max size
        while slow_queries.len() > self.config.max_slow_queries {
            slow_queries.pop_front();
        }
    }

    /// Update collection statistics
    fn update_collection_stats(&self, event: &QueryEvent) {
        let mut stats = self.collection_stats.write();
        let entry = stats.entry(event.collection.clone()).or_insert_with(|| {
            CollectionAnalytics {
                name: event.collection.clone(),
                ..Default::default()
            }
        });

        match event.operation.as_str() {
            "search" | "query" => {
                entry.total_queries += 1;
                let n = entry.total_queries as f64;
                entry.avg_search_latency_ms = 
                    entry.avg_search_latency_ms * (n - 1.0) / n + event.latency_ms / n;
            }
            "insert" | "upsert" => {
                entry.total_inserts += 1;
                let n = entry.total_inserts as f64;
                entry.avg_insert_latency_ms = 
                    entry.avg_insert_latency_ms * (n - 1.0) / n + event.latency_ms / n;
            }
            "delete" => {
                entry.total_deletes += 1;
            }
            _ => {}
        }
    }

    /// Update query pattern
    fn update_pattern(&self, event: &QueryEvent) {
        let pattern_id = QueryPattern::compute_pattern_id(event);
        let mut patterns = self.patterns.write();

        // Keep max pattern limit
        if patterns.len() >= self.config.max_query_patterns && !patterns.contains_key(&pattern_id) {
            // Remove oldest pattern
            let oldest = patterns
                .iter()
                .min_by_key(|(_, (p, _))| p.last_seen)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest {
                patterns.remove(&key);
            }
        }

        patterns
            .entry(pattern_id.clone())
            .and_modify(|(pattern, latencies)| {
                latencies.push(event.latency_ms);
                // Keep only last 1000 latencies for percentile calculation
                if latencies.len() > 1000 {
                    latencies.remove(0);
                }
                pattern.update(event, latencies);
            })
            .or_insert_with(|| (QueryPattern::new(event), vec![event.latency_ms]));
    }

    /// Track an error
    pub fn track_error(&self, collection: &str, operation: &str, _error: &str) {
        self.counters.total_errors.fetch_add(1, Ordering::Relaxed);
        
        // Also track as a query event with high latency marker
        self.track_query(QueryEvent {
            collection: collection.to_string(),
            operation: format!("{}_error", operation),
            latency_ms: 0.0,
            ..Default::default()
        });
    }

    // =========================================================================
    // Insights and Reports
    // =========================================================================

    /// Get current dashboard insights
    pub fn get_insights(&self) -> DashboardInsights {
        let recent = self.recent_events.read();
        let now = Instant::now();
        let window = Duration::from_secs(self.config.rate_window_seconds);

        // Calculate current rates
        let events_in_window: Vec<_> = recent
            .iter()
            .filter(|(t, _)| now.duration_since(*t) < window)
            .collect();

        let current_qps = events_in_window.len() as f64 / self.config.rate_window_seconds as f64;

        // Calculate latency percentiles
        let mut latencies: Vec<f64> = events_in_window
            .iter()
            .filter(|(_, e)| e.operation == "search" || e.operation == "query")
            .map(|(_, e)| e.latency_ms)
            .collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (p50, p95, p99) = if latencies.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let len = latencies.len();
            (
                latencies[len * 50 / 100],
                latencies[len * 95 / 100],
                latencies[len.saturating_sub(1).max(len * 99 / 100)],
            )
        };

        let uptime = self.start_time.elapsed();

        DashboardInsights {
            current_qps,
            avg_latency_ms: if latencies.is_empty() { 0.0 } else { 
                latencies.iter().sum::<f64>() / latencies.len() as f64 
            },
            p50_latency_ms: p50,
            p95_latency_ms: p95,
            p99_latency_ms: p99,
            total_queries: self.counters.total_queries.load(Ordering::Relaxed),
            total_inserts: self.counters.total_inserts.load(Ordering::Relaxed),
            total_deletes: self.counters.total_deletes.load(Ordering::Relaxed),
            total_errors: self.counters.total_errors.load(Ordering::Relaxed),
            slow_query_count: self.counters.slow_queries.load(Ordering::Relaxed),
            filtered_query_ratio: {
                let total = self.counters.total_queries.load(Ordering::Relaxed);
                let filtered = self.counters.filtered_queries.load(Ordering::Relaxed);
                if total > 0 { filtered as f64 / total as f64 } else { 0.0 }
            },
            uptime_seconds: uptime.as_secs(),
            error_rate: {
                let total = self.counters.total_queries.load(Ordering::Relaxed) 
                    + self.counters.total_inserts.load(Ordering::Relaxed)
                    + self.counters.total_deletes.load(Ordering::Relaxed);
                let errors = self.counters.total_errors.load(Ordering::Relaxed);
                if total > 0 { errors as f64 / total as f64 } else { 0.0 }
            },
        }
    }

    /// Get slow queries
    pub fn get_slow_queries(&self, limit: usize) -> Vec<SlowQuery> {
        self.slow_queries
            .read()
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get query patterns sorted by frequency
    pub fn get_top_patterns(&self, limit: usize) -> Vec<QueryPattern> {
        let patterns = self.patterns.read();
        let mut sorted: Vec<_> = patterns.values().map(|(p, _)| p.clone()).collect();
        sorted.sort_by(|a, b| b.count.cmp(&a.count));
        sorted.truncate(limit);
        sorted
    }

    /// Get slowest patterns by P99 latency
    pub fn get_slowest_patterns(&self, limit: usize) -> Vec<QueryPattern> {
        let patterns = self.patterns.read();
        let mut sorted: Vec<_> = patterns.values().map(|(p, _)| p.clone()).collect();
        sorted.sort_by(|a, b| {
            b.p99_latency_ms.partial_cmp(&a.p99_latency_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.truncate(limit);
        sorted
    }

    /// Get collection analytics
    pub fn get_collection_analytics(&self) -> Vec<CollectionAnalytics> {
        self.collection_stats.read().values().cloned().collect()
    }

    /// Get analytics for a specific collection
    pub fn get_collection_analytics_by_name(&self, name: &str) -> Option<CollectionAnalytics> {
        self.collection_stats.read().get(name).cloned()
    }

    /// Generate a JSON report
    pub fn generate_report(&self) -> Value {
        serde_json::json!({
            "insights": self.get_insights(),
            "slow_queries": self.get_slow_queries(10),
            "top_patterns": self.get_top_patterns(10),
            "slowest_patterns": self.get_slowest_patterns(5),
            "collections": self.get_collection_analytics(),
            "generated_at": SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time before UNIX epoch")
                .as_secs(),
        })
    }

    /// Clear all analytics data
    pub fn clear(&self) {
        self.recent_events.write().clear();
        self.slow_queries.write().clear();
        self.patterns.write().clear();
        self.collection_stats.write().clear();
        self.counters.total_queries.store(0, Ordering::Relaxed);
        self.counters.total_inserts.store(0, Ordering::Relaxed);
        self.counters.total_deletes.store(0, Ordering::Relaxed);
        self.counters.total_errors.store(0, Ordering::Relaxed);
        self.counters.slow_queries.store(0, Ordering::Relaxed);
        self.counters.filtered_queries.store(0, Ordering::Relaxed);
    }
}

/// Dashboard insights snapshot
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DashboardInsights {
    /// Current queries per second
    pub current_qps: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// P50 (median) latency
    pub p50_latency_ms: f64,
    /// P95 latency
    pub p95_latency_ms: f64,
    /// P99 latency
    pub p99_latency_ms: f64,
    /// Total search queries
    pub total_queries: u64,
    /// Total inserts
    pub total_inserts: u64,
    /// Total deletes
    pub total_deletes: u64,
    /// Total errors
    pub total_errors: u64,
    /// Count of slow queries
    pub slow_query_count: u64,
    /// Ratio of queries using filters
    pub filtered_query_ratio: f64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
}

// ============================================================================
// Query Tracker (Integration Helper)
// ============================================================================

/// Helper for tracking queries with automatic timing
pub struct QueryTracker {
    dashboard: Arc<AnalyticsDashboard>,
    collection: String,
    operation: String,
    start: Instant,
    k: Option<usize>,
    has_filter: bool,
    filter_complexity: Option<usize>,
    result_count: Option<usize>,
    nodes_visited: Option<usize>,
}

impl QueryTracker {
    /// Create a new query tracker
    pub fn new(dashboard: Arc<AnalyticsDashboard>, collection: &str, operation: &str) -> Self {
        Self {
            dashboard,
            collection: collection.to_string(),
            operation: operation.to_string(),
            start: Instant::now(),
            k: None,
            has_filter: false,
            filter_complexity: None,
            result_count: None,
            nodes_visited: None,
        }
    }

    /// Set the k parameter
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = Some(k);
        self
    }

    /// Mark as using a filter
    pub fn with_filter(mut self, complexity: usize) -> Self {
        self.has_filter = true;
        self.filter_complexity = Some(complexity);
        self
    }

    /// Set the result count
    pub fn set_result_count(&mut self, count: usize) {
        self.result_count = Some(count);
    }

    /// Set nodes visited
    pub fn set_nodes_visited(&mut self, count: usize) {
        self.nodes_visited = Some(count);
    }

    /// Finish tracking (called automatically on drop)
    pub fn finish(self) {
        drop(self);
    }
}

impl Drop for QueryTracker {
    fn drop(&mut self) {
        let latency = self.start.elapsed();
        
        self.dashboard.track_query(QueryEvent {
            collection: self.collection.clone(),
            operation: self.operation.clone(),
            latency_ms: latency.as_secs_f64() * 1000.0,
            result_count: self.result_count.unwrap_or(0),
            k: self.k,
            has_filter: self.has_filter,
            filter_complexity: self.filter_complexity,
            nodes_visited: self.nodes_visited,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time before UNIX epoch")
                .as_secs(),
            metadata: None,
        });
    }
}

// ============================================================================
// Alert Rules
// ============================================================================

/// Alert condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule name
    pub name: String,
    /// Condition type
    pub condition: AlertCondition,
    /// Threshold value
    pub threshold: f64,
    /// Duration the condition must hold
    pub duration_seconds: u64,
    /// Severity level
    pub severity: AlertSeverity,
    /// Description
    pub description: String,
}

/// Types of alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// QPS above threshold
    QpsAbove,
    /// QPS below threshold
    QpsBelow,
    /// Latency P99 above threshold (ms)
    LatencyP99Above,
    /// Error rate above threshold (0.0-1.0)
    ErrorRateAbove,
    /// Slow query rate above threshold
    SlowQueryRateAbove,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// An active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// The rule that triggered
    pub rule: AlertRule,
    /// Current value that triggered the alert
    pub current_value: f64,
    /// When the alert started
    pub started_at: u64,
    /// Whether the alert is currently firing
    pub is_firing: bool,
}

/// Alert manager for evaluating rules
pub struct AlertManager {
    rules: RwLock<Vec<AlertRule>>,
    active_alerts: RwLock<HashMap<String, Alert>>,
    dashboard: Arc<AnalyticsDashboard>,
}

impl AlertManager {
    /// Create a new alert manager
    pub fn new(dashboard: Arc<AnalyticsDashboard>) -> Self {
        Self {
            rules: RwLock::new(Vec::new()),
            active_alerts: RwLock::new(HashMap::new()),
            dashboard,
        }
    }

    /// Add an alert rule
    pub fn add_rule(&self, rule: AlertRule) {
        self.rules.write().push(rule);
    }

    /// Add default alert rules
    pub fn add_default_rules(&self) {
        self.add_rule(AlertRule {
            name: "HighLatency".to_string(),
            condition: AlertCondition::LatencyP99Above,
            threshold: 500.0,
            duration_seconds: 300,
            severity: AlertSeverity::Warning,
            description: "P99 latency exceeds 500ms".to_string(),
        });

        self.add_rule(AlertRule {
            name: "HighErrorRate".to_string(),
            condition: AlertCondition::ErrorRateAbove,
            threshold: 0.05,
            duration_seconds: 300,
            severity: AlertSeverity::Critical,
            description: "Error rate exceeds 5%".to_string(),
        });

        self.add_rule(AlertRule {
            name: "LowThroughput".to_string(),
            condition: AlertCondition::QpsBelow,
            threshold: 1.0,
            duration_seconds: 600,
            severity: AlertSeverity::Info,
            description: "QPS below 1 for 10 minutes".to_string(),
        });
    }

    /// Evaluate all rules and return active alerts
    pub fn evaluate(&self) -> Vec<Alert> {
        let insights = self.dashboard.get_insights();
        let rules = self.rules.read();
        let mut active = self.active_alerts.write();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX epoch")
            .as_secs();

        for rule in rules.iter() {
            let (is_firing, current_value) = match rule.condition {
                AlertCondition::QpsAbove => (insights.current_qps > rule.threshold, insights.current_qps),
                AlertCondition::QpsBelow => (insights.current_qps < rule.threshold, insights.current_qps),
                AlertCondition::LatencyP99Above => (insights.p99_latency_ms > rule.threshold, insights.p99_latency_ms),
                AlertCondition::ErrorRateAbove => (insights.error_rate > rule.threshold, insights.error_rate),
                AlertCondition::SlowQueryRateAbove => {
                    let total = insights.total_queries.max(1);
                    let rate = insights.slow_query_count as f64 / total as f64;
                    (rate > rule.threshold, rate)
                }
            };

            if is_firing {
                active.entry(rule.name.clone())
                    .and_modify(|a| {
                        a.current_value = current_value;
                        a.is_firing = true;
                    })
                    .or_insert_with(|| Alert {
                        rule: rule.clone(),
                        current_value,
                        started_at: now,
                        is_firing: true,
                    });
            } else {
                if let Some(alert) = active.get_mut(&rule.name) {
                    alert.is_firing = false;
                }
            }
        }

        active.values().filter(|a| a.is_firing).cloned().collect()
    }

    /// Get all active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts
            .read()
            .values()
            .filter(|a| a.is_firing)
            .cloned()
            .collect()
    }

    /// Clear resolved alerts
    pub fn clear_resolved(&self) {
        self.active_alerts.write().retain(|_, a| a.is_firing);
    }
}

// ---------------------------------------------------------------------------
// Time-Series Metrics Store for Dashboards
// ---------------------------------------------------------------------------

/// A data point in a time-series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: u64,
    pub value: f64,
}

/// A named time-series with configurable retention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub name: String,
    pub points: Vec<TimeSeriesPoint>,
    pub max_points: usize,
}

impl TimeSeries {
    pub fn new(name: &str, max_points: usize) -> Self {
        Self {
            name: name.to_string(),
            points: Vec::new(),
            max_points,
        }
    }

    pub fn push(&mut self, value: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.points.push(TimeSeriesPoint {
            timestamp: now,
            value,
        });
        if self.points.len() > self.max_points {
            self.points.remove(0);
        }
    }

    pub fn latest(&self) -> Option<f64> {
        self.points.last().map(|p| p.value)
    }

    pub fn average(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|p| p.value).sum();
        sum / self.points.len() as f64
    }

    pub fn max(&self) -> f64 {
        self.points
            .iter()
            .map(|p| p.value)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Stores multiple time-series for the dashboard.
pub struct MetricsStore {
    series: RwLock<HashMap<String, TimeSeries>>,
    default_max_points: usize,
}

impl MetricsStore {
    pub fn new(default_max_points: usize) -> Self {
        Self {
            series: RwLock::new(HashMap::new()),
            default_max_points,
        }
    }

    /// Record a metric value.
    pub fn record(&self, name: &str, value: f64) {
        let mut series = self.series.write();
        let ts = series
            .entry(name.to_string())
            .or_insert_with(|| TimeSeries::new(name, self.default_max_points));
        ts.push(value);
    }

    /// Get a time-series by name.
    pub fn get(&self, name: &str) -> Option<TimeSeries> {
        self.series.read().get(name).cloned()
    }

    /// List all metric names.
    pub fn list_metrics(&self) -> Vec<String> {
        self.series.read().keys().cloned().collect()
    }

    /// Generate a JSON snapshot of all metrics for the dashboard.
    pub fn snapshot(&self) -> serde_json::Value {
        let series = self.series.read();
        let mut map = serde_json::Map::new();
        for (name, ts) in series.iter() {
            map.insert(
                name.clone(),
                serde_json::json!({
                    "latest": ts.latest(),
                    "average": ts.average(),
                    "max": ts.max(),
                    "points": ts.points.len(),
                }),
            );
        }
        serde_json::Value::Object(map)
    }
}

/// Generates an HTML dashboard page from the metrics store.
pub fn generate_dashboard_html(
    insights: &DashboardInsights,
    metrics: &MetricsStore,
) -> String {
    let snapshot = metrics.snapshot();
    let insights_json = serde_json::to_string_pretty(insights).unwrap_or_default();
    let metrics_json = serde_json::to_string_pretty(&snapshot).unwrap_or_default();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Needle Analytics Dashboard</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }}
    .card {{ background: white; border-radius: 8px; padding: 20px; margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .metric {{ text-align: center; }}
    .metric .value {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
    .metric .label {{ color: #6b7280; font-size: 0.9em; }}
    pre {{ background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 8px; overflow-x: auto; }}
    h1 {{ color: #1e293b; }}
    h2 {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }}
  </style>
</head>
<body>
  <h1>📊 Needle Analytics Dashboard</h1>

  <div class="grid">
    <div class="card metric"><div class="value">{qps}</div><div class="label">Queries / sec (avg)</div></div>
    <div class="card metric"><div class="value">{latency}</div><div class="label">Avg Latency (ms)</div></div>
    <div class="card metric"><div class="value">{slow}</div><div class="label">Slow Queries</div></div>
    <div class="card metric"><div class="value">{error_rate}</div><div class="label">Error Rate</div></div>
  </div>

  <div class="card">
    <h2>Insights</h2>
    <pre>{insights}</pre>
  </div>

  <div class="card">
    <h2>Metrics Snapshot</h2>
    <pre>{metrics}</pre>
  </div>
</body>
</html>"#,
        qps = format!("{:.1}", insights.current_qps),
        latency = format!("{:.2}", insights.avg_latency_ms),
        slow = insights.slow_query_count,
        error_rate = format!("{:.2}%", insights.error_rate * 100.0),
        insights = insights_json,
        metrics = metrics_json,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let config = AnalyticsConfig::default();
        let dashboard = AnalyticsDashboard::new(config);
        
        let insights = dashboard.get_insights();
        assert_eq!(insights.total_queries, 0);
        assert_eq!(insights.current_qps, 0.0);
    }

    #[test]
    fn test_query_tracking() {
        let config = AnalyticsConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        for i in 0..100 {
            dashboard.track_query(QueryEvent {
                collection: "test".to_string(),
                operation: "search".to_string(),
                latency_ms: 10.0 + (i as f64),
                result_count: 10,
                k: Some(10),
                has_filter: i % 2 == 0,
                ..Default::default()
            });
        }

        let insights = dashboard.get_insights();
        assert_eq!(insights.total_queries, 100);
        assert!(insights.filtered_query_ratio > 0.4);
    }

    #[test]
    fn test_slow_query_logging() {
        let config = AnalyticsConfig {
            slow_query_threshold_ms: 50.0,
            ..Default::default()
        };
        let dashboard = AnalyticsDashboard::new(config);

        // Fast query
        dashboard.track_query(QueryEvent {
            collection: "test".to_string(),
            operation: "search".to_string(),
            latency_ms: 10.0,
            ..Default::default()
        });

        // Slow query
        dashboard.track_query(QueryEvent {
            collection: "test".to_string(),
            operation: "search".to_string(),
            latency_ms: 100.0,
            ..Default::default()
        });

        let slow = dashboard.get_slow_queries(10);
        assert_eq!(slow.len(), 1);
        assert!(slow[0].event.latency_ms > 50.0);
    }

    #[test]
    fn test_pattern_analysis() {
        let config = AnalyticsConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        // Same pattern multiple times
        for _ in 0..50 {
            dashboard.track_query(QueryEvent {
                collection: "docs".to_string(),
                operation: "search".to_string(),
                latency_ms: 20.0,
                k: Some(10),
                has_filter: true,
                ..Default::default()
            });
        }

        // Different pattern
        for _ in 0..30 {
            dashboard.track_query(QueryEvent {
                collection: "docs".to_string(),
                operation: "search".to_string(),
                latency_ms: 15.0,
                k: Some(5),
                has_filter: false,
                ..Default::default()
            });
        }

        let patterns = dashboard.get_top_patterns(10);
        assert!(patterns.len() >= 2);
        assert!(patterns[0].count >= patterns[1].count);
    }

    #[test]
    fn test_collection_analytics() {
        let config = AnalyticsConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        // Queries to different collections
        for _ in 0..10 {
            dashboard.track_query(QueryEvent {
                collection: "docs".to_string(),
                operation: "search".to_string(),
                latency_ms: 20.0,
                ..Default::default()
            });
        }

        for _ in 0..5 {
            dashboard.track_query(QueryEvent {
                collection: "images".to_string(),
                operation: "search".to_string(),
                latency_ms: 30.0,
                ..Default::default()
            });
        }

        let analytics = dashboard.get_collection_analytics();
        assert_eq!(analytics.len(), 2);

        let docs = dashboard.get_collection_analytics_by_name("docs").unwrap();
        assert_eq!(docs.total_queries, 10);
    }

    #[test]
    fn test_query_tracker() {
        let config = AnalyticsConfig::default();
        let dashboard = Arc::new(AnalyticsDashboard::new(config));

        {
            let mut tracker = QueryTracker::new(dashboard.clone(), "test", "search")
                .with_k(10)
                .with_filter(3);
            tracker.set_result_count(5);
            // tracker drops here, recording the event
        }

        let insights = dashboard.get_insights();
        assert_eq!(insights.total_queries, 1);
    }

    #[test]
    fn test_alert_manager() {
        let config = AnalyticsConfig::default();
        let dashboard = Arc::new(AnalyticsDashboard::new(config));
        let alerts = AlertManager::new(dashboard.clone());

        alerts.add_rule(AlertRule {
            name: "TestAlert".to_string(),
            condition: AlertCondition::QpsBelow,
            threshold: 100.0,
            duration_seconds: 0,
            severity: AlertSeverity::Warning,
            description: "Test".to_string(),
        });

        let active = alerts.evaluate();
        // QPS is 0, which is below 100
        assert!(!active.is_empty());
    }

    #[test]
    fn test_report_generation() {
        let config = AnalyticsConfig::default();
        let dashboard = AnalyticsDashboard::new(config);

        dashboard.track_query(QueryEvent {
            collection: "test".to_string(),
            operation: "search".to_string(),
            latency_ms: 20.0,
            ..Default::default()
        });

        let report = dashboard.generate_report();
        assert!(report.get("insights").is_some());
        assert!(report.get("collections").is_some());
    }

    #[test]
    fn test_metrics_store() {
        let store = MetricsStore::new(100);
        store.record("qps", 150.0);
        store.record("qps", 200.0);
        store.record("latency_ms", 5.5);

        let qps = store.get("qps").unwrap();
        assert_eq!(qps.points.len(), 2);
        assert_eq!(qps.average(), 175.0);
        assert_eq!(qps.max(), 200.0);

        let names = store.list_metrics();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_metrics_store_retention() {
        let store = MetricsStore::new(3);
        for i in 0..5 {
            store.record("x", i as f64);
        }
        let ts = store.get("x").unwrap();
        assert_eq!(ts.points.len(), 3); // oldest evicted
    }

    #[test]
    fn test_dashboard_html_generation() {
        let insights = DashboardInsights {
            current_qps: 42.5,
            avg_latency_ms: 3.2,
            slow_query_count: 2,
            error_rate: 0.01,
            ..Default::default()
        };
        let store = MetricsStore::new(100);
        store.record("qps", 42.5);
        let html = generate_dashboard_html(&insights, &store);
        assert!(html.contains("Needle Analytics Dashboard"));
        assert!(html.contains("42.5"));
    }
}
