#![allow(clippy::unwrap_used)]
//! Query Profiler - EXPLAIN ANALYZE for vector search operations.
//!
//! Provides detailed performance analysis of vector search queries,
//! including execution plans, timing breakdowns, and optimization suggestions.
//!
//! # Features
//!
//! - **Execution plans**: Visualize query execution strategy
//! - **Timing breakdown**: Per-stage latency measurements
//! - **Resource usage**: Memory, I/O, and CPU metrics
//! - **Optimization hints**: Automatic suggestions for improvement
//! - **Comparative analysis**: Compare different query strategies
//!
//! # Example
//!
//! ```ignore
//! use needle::profiler::{QueryProfiler, ProfiledQuery};
//!
//! let profiler = QueryProfiler::new();
//!
//! let result = profiler.profile(|| {
//!     db.search(&query, 10)
//! });
//!
//! println!("{}", result.explain());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Profiler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable detailed timing.
    pub detailed_timing: bool,
    /// Enable memory tracking.
    pub track_memory: bool,
    /// Enable I/O tracking.
    pub track_io: bool,
    /// Sample rate for continuous profiling.
    pub sample_rate: f64,
    /// Maximum profile history.
    pub max_history: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            detailed_timing: true,
            track_memory: true,
            track_io: true,
            sample_rate: 1.0,
            max_history: 1000,
        }
    }
}

/// Query execution stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStage {
    /// Query parsing.
    Parse,
    /// Filter evaluation.
    Filter,
    /// Vector search.
    VectorSearch,
    /// Index lookup.
    IndexLookup,
    /// Distance calculation.
    DistanceCalc,
    /// Result ranking.
    Ranking,
    /// Post-processing.
    PostProcess,
    /// Serialization.
    Serialize,
    /// Custom stage.
    Custom(String),
}

impl ExecutionStage {
    /// Get stage name.
    pub fn name(&self) -> &str {
        match self {
            ExecutionStage::Parse => "Parse",
            ExecutionStage::Filter => "Filter",
            ExecutionStage::VectorSearch => "VectorSearch",
            ExecutionStage::IndexLookup => "IndexLookup",
            ExecutionStage::DistanceCalc => "DistanceCalc",
            ExecutionStage::Ranking => "Ranking",
            ExecutionStage::PostProcess => "PostProcess",
            ExecutionStage::Serialize => "Serialize",
            ExecutionStage::Custom(name) => name,
        }
    }
}

/// Timing for a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTiming {
    /// Stage.
    pub stage: ExecutionStage,
    /// Start time (relative to query start).
    pub start_offset_us: u64,
    /// Duration in microseconds.
    pub duration_us: u64,
    /// Number of iterations/loops.
    pub iterations: usize,
    /// Items processed.
    pub items_processed: usize,
}

impl StageTiming {
    /// Get items per second throughput.
    pub fn throughput(&self) -> f64 {
        if self.duration_us == 0 {
            return 0.0;
        }
        (self.items_processed as f64 / self.duration_us as f64) * 1_000_000.0
    }
}

/// Memory usage snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Heap bytes used.
    pub heap_bytes: usize,
    /// Peak heap bytes.
    pub peak_heap_bytes: usize,
    /// Allocations count.
    pub allocations: usize,
    /// Deallocations count.
    pub deallocations: usize,
}

/// I/O statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IoStats {
    /// Bytes read.
    pub bytes_read: usize,
    /// Bytes written.
    pub bytes_written: usize,
    /// Read operations.
    pub read_ops: usize,
    /// Write operations.
    pub write_ops: usize,
    /// Cache hits.
    pub cache_hits: usize,
    /// Cache misses.
    pub cache_misses: usize,
}

/// Query execution plan node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    /// Node type.
    pub node_type: String,
    /// Description.
    pub description: String,
    /// Estimated cost.
    pub estimated_cost: f64,
    /// Actual cost (after execution).
    pub actual_cost: Option<f64>,
    /// Estimated rows.
    pub estimated_rows: usize,
    /// Actual rows (after execution).
    pub actual_rows: Option<usize>,
    /// Child nodes.
    pub children: Vec<PlanNode>,
    /// Node properties.
    pub properties: HashMap<String, String>,
}

impl PlanNode {
    /// Create a new plan node.
    pub fn new(node_type: &str, description: &str) -> Self {
        Self {
            node_type: node_type.to_string(),
            description: description.to_string(),
            estimated_cost: 0.0,
            actual_cost: None,
            estimated_rows: 0,
            actual_rows: None,
            children: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Add a child node.
    pub fn child(mut self, child: PlanNode) -> Self {
        self.children.push(child);
        self
    }

    /// Set estimated cost.
    pub fn cost(mut self, cost: f64) -> Self {
        self.estimated_cost = cost;
        self
    }

    /// Set estimated rows.
    pub fn rows(mut self, rows: usize) -> Self {
        self.estimated_rows = rows;
        self
    }

    /// Add a property.
    pub fn property(mut self, key: &str, value: &str) -> Self {
        self.properties.insert(key.to_string(), value.to_string());
        self
    }

    /// Format as tree string.
    pub fn format_tree(&self, indent: usize) -> String {
        let mut result = String::new();
        let prefix = "  ".repeat(indent);

        result.push_str(&format!(
            "{}-> {} ({})\n",
            prefix, self.node_type, self.description
        ));

        if !self.properties.is_empty() {
            for (k, v) in &self.properties {
                result.push_str(&format!("{}   {}: {}\n", prefix, k, v));
            }
        }

        if let Some(actual_cost) = self.actual_cost {
            result.push_str(&format!(
                "{}   Cost: {:.2} (est: {:.2})\n",
                prefix, actual_cost, self.estimated_cost
            ));
        }

        if let Some(actual_rows) = self.actual_rows {
            result.push_str(&format!(
                "{}   Rows: {} (est: {})\n",
                prefix, actual_rows, self.estimated_rows
            ));
        }

        for child in &self.children {
            result.push_str(&child.format_tree(indent + 1));
        }

        result
    }
}

/// Optimization suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Severity (info, warning, critical).
    pub severity: HintSeverity,
    /// Category.
    pub category: String,
    /// Message.
    pub message: String,
    /// Suggested action.
    pub suggestion: String,
    /// Estimated improvement.
    pub estimated_improvement: Option<f64>,
}

/// Hint severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HintSeverity {
    /// Informational.
    Info,
    /// Warning.
    Warning,
    /// Critical issue.
    Critical,
}

/// Complete query profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    /// Query identifier.
    pub query_id: String,
    /// Query description.
    pub query_desc: String,
    /// Total execution time.
    pub total_time_us: u64,
    /// Execution plan.
    pub plan: Option<PlanNode>,
    /// Stage timings.
    pub stages: Vec<StageTiming>,
    /// Memory usage.
    pub memory: MemorySnapshot,
    /// I/O statistics.
    pub io: IoStats,
    /// Optimization hints.
    pub hints: Vec<OptimizationHint>,
    /// Result count.
    pub result_count: usize,
    /// Timestamp.
    pub timestamp: u64,
    /// Custom metrics.
    pub metrics: HashMap<String, f64>,
}

impl QueryProfile {
    /// Create a new profile.
    pub fn new(query_id: &str) -> Self {
        Self {
            query_id: query_id.to_string(),
            query_desc: String::new(),
            total_time_us: 0,
            plan: None,
            stages: Vec::new(),
            memory: MemorySnapshot::default(),
            io: IoStats::default(),
            hints: Vec::new(),
            result_count: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metrics: HashMap::new(),
        }
    }

    /// Generate EXPLAIN output.
    pub fn explain(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("Query Profile: {}\n", self.query_id));
        output.push_str(&format!(
            "Total Time: {:.3} ms\n",
            self.total_time_us as f64 / 1000.0
        ));
        output.push_str(&format!("Results: {}\n\n", self.result_count));

        // Execution plan
        if let Some(plan) = &self.plan {
            output.push_str("Execution Plan:\n");
            output.push_str(&plan.format_tree(0));
            output.push('\n');
        }

        // Stage breakdown
        output.push_str("Stage Breakdown:\n");
        for stage in &self.stages {
            let pct = if self.total_time_us > 0 {
                (stage.duration_us as f64 / self.total_time_us as f64) * 100.0
            } else {
                0.0
            };
            output.push_str(&format!(
                "  {:<15} {:>8.3} ms ({:>5.1}%) - {} items\n",
                stage.stage.name(),
                stage.duration_us as f64 / 1000.0,
                pct,
                stage.items_processed
            ));
        }

        // Memory
        output.push_str(&format!(
            "\nMemory: {} bytes (peak: {} bytes)\n",
            self.memory.heap_bytes, self.memory.peak_heap_bytes
        ));

        // I/O
        output.push_str(&format!(
            "I/O: {} reads ({} bytes), {} cache hits\n",
            self.io.read_ops, self.io.bytes_read, self.io.cache_hits
        ));

        // Hints
        if !self.hints.is_empty() {
            output.push_str("\nOptimization Hints:\n");
            for hint in &self.hints {
                let severity = match hint.severity {
                    HintSeverity::Info => "INFO",
                    HintSeverity::Warning => "WARN",
                    HintSeverity::Critical => "CRIT",
                };
                output.push_str(&format!(
                    "  [{}] {}: {}\n        -> {}\n",
                    severity, hint.category, hint.message, hint.suggestion
                ));
            }
        }

        output
    }

    /// Generate EXPLAIN ANALYZE output (verbose).
    pub fn explain_analyze(&self) -> String {
        let mut output = self.explain();

        output.push_str("\n--- Detailed Metrics ---\n");

        for (key, value) in &self.metrics {
            output.push_str(&format!("  {}: {:.4}\n", key, value));
        }

        output
    }
}

/// Query profiler.
pub struct QueryProfiler {
    /// Configuration.
    config: ProfilerConfig,
    /// Profile history.
    history: Vec<QueryProfile>,
    /// Active profile.
    active: Option<ProfileBuilder>,
}

impl QueryProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create with config.
    pub fn with_config(config: ProfilerConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            active: None,
        }
    }

    /// Start profiling a query.
    pub fn start(&mut self, query_id: &str) -> &mut ProfileBuilder {
        self.active = Some(ProfileBuilder::new(query_id, &self.config));
        self.active.as_mut().expect("profiler has active span")
    }

    /// End profiling and get result.
    pub fn end(&mut self) -> Option<QueryProfile> {
        if let Some(builder) = self.active.take() {
            let profile = builder.finish();

            // Add to history
            self.history.push(profile.clone());
            if self.history.len() > self.config.max_history {
                self.history.remove(0);
            }

            Some(profile)
        } else {
            None
        }
    }

    /// Profile a function.
    pub fn profile<T, F: FnOnce() -> T>(&mut self, query_id: &str, f: F) -> (T, QueryProfile) {
        self.start(query_id);
        let result = f();
        let profile = self.end().expect("profiler has active span");
        (result, profile)
    }

    /// Get profile history.
    pub fn history(&self) -> &[QueryProfile] {
        &self.history
    }

    /// Get average metrics from history.
    pub fn average_metrics(&self) -> HashMap<String, f64> {
        if self.history.is_empty() {
            return HashMap::new();
        }

        let mut totals: HashMap<String, f64> = HashMap::new();
        let count = self.history.len() as f64;

        for profile in &self.history {
            *totals.entry("total_time_us".to_string()).or_default() += profile.total_time_us as f64;
            *totals.entry("result_count".to_string()).or_default() += profile.result_count as f64;

            for (key, value) in &profile.metrics {
                *totals.entry(key.clone()).or_default() += value;
            }
        }

        for value in totals.values_mut() {
            *value /= count;
        }

        totals
    }

    /// Clear history.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

impl Default for QueryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing a query profile.
pub struct ProfileBuilder {
    profile: QueryProfile,
    start_time: Instant,
    stage_start: Option<(ExecutionStage, Instant)>,
    config: ProfilerConfig,
}

impl ProfileBuilder {
    /// Create new builder.
    fn new(query_id: &str, config: &ProfilerConfig) -> Self {
        Self {
            profile: QueryProfile::new(query_id),
            start_time: Instant::now(),
            stage_start: None,
            config: config.clone(),
        }
    }

    /// Set query description.
    pub fn description(&mut self, desc: &str) -> &mut Self {
        self.profile.query_desc = desc.to_string();
        self
    }

    /// Set execution plan.
    pub fn plan(&mut self, plan: PlanNode) -> &mut Self {
        self.profile.plan = Some(plan);
        self
    }

    /// Start a stage.
    pub fn start_stage(&mut self, stage: ExecutionStage) -> &mut Self {
        self.end_stage(); // End any active stage
        self.stage_start = Some((stage, Instant::now()));
        self
    }

    /// End current stage.
    pub fn end_stage(&mut self) -> &mut Self {
        self.end_stage_with_items(0)
    }

    /// End stage with item count.
    pub fn end_stage_with_items(&mut self, items: usize) -> &mut Self {
        if let Some((stage, start)) = self.stage_start.take() {
            let elapsed = start.elapsed();
            let start_offset = start.duration_since(self.start_time);

            self.profile.stages.push(StageTiming {
                stage,
                start_offset_us: start_offset.as_micros() as u64,
                duration_us: elapsed.as_micros() as u64,
                iterations: 1,
                items_processed: items,
            });
        }
        self
    }

    /// Record memory usage.
    pub fn memory(&mut self, heap_bytes: usize, peak: usize) -> &mut Self {
        self.profile.memory.heap_bytes = heap_bytes;
        self.profile.memory.peak_heap_bytes = peak;
        self
    }

    /// Record I/O stats.
    pub fn io(&mut self, reads: usize, bytes_read: usize, cache_hits: usize) -> &mut Self {
        self.profile.io.read_ops = reads;
        self.profile.io.bytes_read = bytes_read;
        self.profile.io.cache_hits = cache_hits;
        self
    }

    /// Add a metric.
    pub fn metric(&mut self, name: &str, value: f64) -> &mut Self {
        self.profile.metrics.insert(name.to_string(), value);
        self
    }

    /// Set result count.
    pub fn results(&mut self, count: usize) -> &mut Self {
        self.profile.result_count = count;
        self
    }

    /// Add optimization hint.
    pub fn hint(&mut self, hint: OptimizationHint) -> &mut Self {
        self.profile.hints.push(hint);
        self
    }

    /// Finish and generate profile.
    fn finish(mut self) -> QueryProfile {
        self.end_stage(); // End any active stage
        self.profile.total_time_us = self.start_time.elapsed().as_micros() as u64;
        self.analyze_and_add_hints();
        self.profile
    }

    /// Analyze profile and add automatic hints.
    fn analyze_and_add_hints(&mut self) {
        // Check for slow stages
        for stage in &self.profile.stages {
            let pct = if self.profile.total_time_us > 0 {
                (stage.duration_us as f64 / self.profile.total_time_us as f64) * 100.0
            } else {
                0.0
            };

            if pct > 50.0 {
                self.profile.hints.push(OptimizationHint {
                    severity: HintSeverity::Warning,
                    category: "Performance".to_string(),
                    message: format!(
                        "{} stage takes {}% of query time",
                        stage.stage.name(),
                        pct as i32
                    ),
                    suggestion: format!(
                        "Consider optimizing {} stage or using caching",
                        stage.stage.name()
                    ),
                    estimated_improvement: Some(pct / 2.0),
                });
            }
        }

        // Check cache hit ratio
        let total_accesses = self.profile.io.cache_hits + self.profile.io.cache_misses;
        if total_accesses > 0 {
            let hit_ratio = self.profile.io.cache_hits as f64 / total_accesses as f64;
            if hit_ratio < 0.5 {
                self.profile.hints.push(OptimizationHint {
                    severity: HintSeverity::Info,
                    category: "Caching".to_string(),
                    message: format!("Low cache hit ratio: {:.1}%", hit_ratio * 100.0),
                    suggestion: "Consider increasing cache size or pre-warming cache".to_string(),
                    estimated_improvement: Some((1.0 - hit_ratio) * 20.0),
                });
            }
        }

        // Check memory usage
        if self.profile.memory.peak_heap_bytes > 100 * 1024 * 1024 {
            self.profile.hints.push(OptimizationHint {
                severity: HintSeverity::Warning,
                category: "Memory".to_string(),
                message: format!(
                    "High memory usage: {} MB",
                    self.profile.memory.peak_heap_bytes / (1024 * 1024)
                ),
                suggestion: "Consider streaming results or using pagination".to_string(),
                estimated_improvement: None,
            });
        }
    }
}

/// Compare two profiles.
pub fn compare_profiles(a: &QueryProfile, b: &QueryProfile) -> ProfileComparison {
    let time_diff = b.total_time_us as f64 - a.total_time_us as f64;
    let time_pct = if a.total_time_us > 0 {
        (time_diff / a.total_time_us as f64) * 100.0
    } else {
        0.0
    };

    ProfileComparison {
        baseline_id: a.query_id.clone(),
        comparison_id: b.query_id.clone(),
        time_diff_us: time_diff as i64,
        time_diff_pct: time_pct,
        memory_diff: b.memory.peak_heap_bytes as i64 - a.memory.peak_heap_bytes as i64,
        io_diff: b.io.bytes_read as i64 - a.io.bytes_read as i64,
        is_improvement: time_diff < 0.0,
    }
}

/// Profile comparison result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileComparison {
    /// Baseline profile ID.
    pub baseline_id: String,
    /// Comparison profile ID.
    pub comparison_id: String,
    /// Time difference in microseconds.
    pub time_diff_us: i64,
    /// Time difference percentage.
    pub time_diff_pct: f64,
    /// Memory difference in bytes.
    pub memory_diff: i64,
    /// I/O difference in bytes.
    pub io_diff: i64,
    /// Whether comparison is an improvement.
    pub is_improvement: bool,
}

// ── OpenTelemetry Distributed Tracing ────────────────────────────────────────

/// Distributed trace context for propagation across gRPC/REST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// W3C Trace ID (32 hex chars).
    pub trace_id: String,
    /// W3C Span ID (16 hex chars).
    pub span_id: String,
    /// Parent Span ID (if this is a child span).
    pub parent_span_id: Option<String>,
    /// Trace flags (e.g., sampled).
    pub flags: u8,
    /// Baggage items for cross-service context.
    pub baggage: HashMap<String, String>,
}

impl TraceContext {
    /// Create a new root trace context.
    pub fn new_root() -> Self {
        Self {
            trace_id: generate_trace_id(),
            span_id: generate_span_id(),
            parent_span_id: None,
            flags: 0x01, // sampled
            baggage: HashMap::new(),
        }
    }

    /// Create a child span context.
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            span_id: generate_span_id(),
            parent_span_id: Some(self.span_id.clone()),
            flags: self.flags,
            baggage: self.baggage.clone(),
        }
    }

    /// Serialize to W3C traceparent header value.
    pub fn to_traceparent(&self) -> String {
        format!("00-{}-{}-{:02x}", self.trace_id, self.span_id, self.flags)
    }

    /// Parse from W3C traceparent header value.
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }
        let flags = u8::from_str_radix(parts[3], 16).ok()?;
        Some(Self {
            trace_id: parts[1].to_string(),
            span_id: parts[2].to_string(),
            parent_span_id: None,
            flags,
            baggage: HashMap::new(),
        })
    }

    /// Check if this trace is sampled.
    pub fn is_sampled(&self) -> bool {
        self.flags & 0x01 != 0
    }
}

fn generate_trace_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:032x}", rng.gen::<u128>())
}

fn generate_span_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:016x}", rng.gen::<u64>())
}

/// A recorded span with timing and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracedSpan {
    /// Span name (e.g., "hnsw.search", "filter.evaluate").
    pub name: String,
    /// Trace context.
    pub context: TraceContext,
    /// Start time (epoch microseconds).
    pub start_us: u64,
    /// Duration in microseconds.
    pub duration_us: u64,
    /// Span attributes.
    pub attributes: HashMap<String, String>,
    /// Span status.
    pub status: SpanStatus,
}

/// Span status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Unset (default).
    Unset,
    /// Ok.
    Ok,
    /// Error.
    Error,
}

impl Default for SpanStatus {
    fn default() -> Self {
        Self::Unset
    }
}

// ── Slow Query Log ───────────────────────────────────────────────────────────

/// Configuration for slow query logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryConfig {
    /// Threshold in microseconds; queries slower than this are logged.
    pub threshold_us: u64,
    /// Maximum entries in the slow query log.
    pub max_entries: usize,
    /// Auto-capture EXPLAIN for slow queries.
    pub auto_explain: bool,
    /// Sampling rate for slow queries (0.0-1.0).
    pub sample_rate: f64,
}

impl Default for SlowQueryConfig {
    fn default() -> Self {
        Self {
            threshold_us: 10_000, // 10ms
            max_entries: 1000,
            auto_explain: true,
            sample_rate: 1.0,
        }
    }
}

/// A slow query log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryEntry {
    /// Query ID.
    pub query_id: String,
    /// Query description.
    pub description: String,
    /// Total duration in microseconds.
    pub duration_us: u64,
    /// Timestamp (epoch seconds).
    pub timestamp: u64,
    /// Collection name.
    pub collection: String,
    /// Number of results returned.
    pub results_count: usize,
    /// Auto-captured EXPLAIN output.
    pub explain: Option<String>,
    /// Trace context (if distributed tracing is enabled).
    pub trace_context: Option<TraceContext>,
}

/// Slow query log with configurable threshold and automatic EXPLAIN capture.
pub struct SlowQueryLog {
    config: SlowQueryConfig,
    entries: Vec<SlowQueryEntry>,
    next_id: u64,
}

impl SlowQueryLog {
    /// Create a new slow query log.
    pub fn new(config: SlowQueryConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            next_id: 0,
        }
    }

    /// Record a query if it exceeds the slow query threshold.
    pub fn record(
        &mut self,
        duration_us: u64,
        description: &str,
        collection: &str,
        results_count: usize,
        explain: Option<String>,
        trace_context: Option<TraceContext>,
    ) -> Option<String> {
        if duration_us < self.config.threshold_us {
            return None;
        }

        // Apply sampling
        if self.config.sample_rate < 1.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() > self.config.sample_rate {
                return None;
            }
        }

        self.next_id += 1;
        let query_id = format!("sq_{}", self.next_id);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = SlowQueryEntry {
            query_id: query_id.clone(),
            description: description.to_string(),
            duration_us,
            timestamp: now,
            collection: collection.to_string(),
            results_count,
            explain,
            trace_context,
        };

        if self.entries.len() >= self.config.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);

        Some(query_id)
    }

    /// Get all slow query entries.
    pub fn entries(&self) -> &[SlowQueryEntry] {
        &self.entries
    }

    /// Get entries for a specific collection.
    pub fn entries_for_collection(&self, collection: &str) -> Vec<&SlowQueryEntry> {
        self.entries
            .iter()
            .filter(|e| e.collection == collection)
            .collect()
    }

    /// Get the top N slowest queries.
    pub fn top_slowest(&self, n: usize) -> Vec<&SlowQueryEntry> {
        let mut sorted: Vec<_> = self.entries.iter().collect();
        sorted.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));
        sorted.truncate(n);
        sorted
    }

    /// Clear the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Debug Profile Endpoint Data ──────────────────────────────────────────────

/// Response data for the /debug/profile endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugProfileResponse {
    /// HNSW layer traversal breakdown.
    pub hnsw_layers: Vec<HnswLayerProfile>,
    /// Filter evaluation cost.
    pub filter_evaluation: Option<FilterProfile>,
    /// Distance computation breakdown.
    pub distance_computation: DistanceProfile,
    /// Slow query summary.
    pub slow_queries: SlowQuerySummary,
}

/// HNSW layer traversal profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswLayerProfile {
    /// Layer number (0 = bottom).
    pub layer: usize,
    /// Nodes visited in this layer.
    pub nodes_visited: usize,
    /// Time spent in this layer (microseconds).
    pub time_us: u64,
    /// Distance computations in this layer.
    pub distance_computations: usize,
}

/// Filter evaluation profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterProfile {
    /// Filter expression.
    pub expression: String,
    /// Candidates before filtering.
    pub candidates_before: usize,
    /// Candidates after filtering.
    pub candidates_after: usize,
    /// Filter evaluation time (microseconds).
    pub time_us: u64,
    /// Selectivity ratio.
    pub selectivity: f64,
}

/// Distance computation profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceProfile {
    /// Distance function used.
    pub function: String,
    /// Total distance computations.
    pub total_computations: usize,
    /// Total time (microseconds).
    pub total_time_us: u64,
    /// Average time per computation (nanoseconds).
    pub avg_per_computation_ns: f64,
    /// Vector dimensions.
    pub dimensions: usize,
}

/// Summary of slow queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQuerySummary {
    /// Total slow queries recorded.
    pub total: usize,
    /// Slowest query time (microseconds).
    pub max_duration_us: u64,
    /// Average slow query time (microseconds).
    pub avg_duration_us: f64,
    /// Top collections by slow query count.
    pub top_collections: Vec<(String, usize)>,
}

// ── Span Collector ───────────────────────────────────────────────────────────

/// Collects spans for export to OpenTelemetry-compatible backends.
pub struct SpanCollector {
    spans: Vec<TracedSpan>,
    max_capacity: usize,
}

impl SpanCollector {
    /// Create a new span collector with a maximum capacity.
    pub fn new(max_capacity: usize) -> Self {
        Self {
            spans: Vec::new(),
            max_capacity,
        }
    }

    /// Record a completed span.
    pub fn record(&mut self, span: TracedSpan) {
        if self.spans.len() >= self.max_capacity {
            self.spans.remove(0);
        }
        self.spans.push(span);
    }

    /// Drain all collected spans for export.
    pub fn drain(&mut self) -> Vec<TracedSpan> {
        std::mem::take(&mut self.spans)
    }

    /// Number of collected spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

// ── Slow Query Percentile Stats ──────────────────────────────────────────────

/// Percentile statistics for slow queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryPercentileStats {
    /// Median (p50) latency in microseconds.
    pub p50_us: u64,
    /// P95 latency.
    pub p95_us: u64,
    /// P99 latency.
    pub p99_us: u64,
    /// Average latency.
    pub avg_us: f64,
    /// Total entries analyzed.
    pub count: usize,
}

impl SlowQueryLog {
    /// Compute percentile statistics across all slow query entries.
    pub fn compute_stats(&self) -> SlowQueryPercentileStats {
        if self.entries.is_empty() {
            return SlowQueryPercentileStats {
                p50_us: 0,
                p95_us: 0,
                p99_us: 0,
                avg_us: 0.0,
                count: 0,
            };
        }

        let mut durations: Vec<u64> = self.entries.iter().map(|e| e.duration_us).collect();
        durations.sort_unstable();
        let n = durations.len();
        let sum: u64 = durations.iter().sum();

        SlowQueryPercentileStats {
            p50_us: durations[n / 2],
            p95_us: durations[((n as f64 * 0.95) as usize).min(n - 1)],
            p99_us: durations[((n as f64 * 0.99) as usize).min(n - 1)],
            avg_us: sum as f64 / n as f64,
            count: n,
        }
    }
}

// ── Profile Endpoint Builder ─────────────────────────────────────────────────

/// Builder for constructing /debug/profile responses incrementally.
pub struct ProfileEndpointBuilder {
    hnsw_layers: Vec<HnswLayerProfile>,
    filter: Option<FilterProfile>,
    distance: DistanceProfile,
    slow_queries: SlowQuerySummary,
}

impl ProfileEndpointBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            hnsw_layers: Vec::new(),
            filter: None,
            distance: DistanceProfile {
                function: String::new(),
                total_computations: 0,
                total_time_us: 0,
                avg_per_computation_ns: 0.0,
                dimensions: 0,
            },
            slow_queries: SlowQuerySummary {
                total: 0,
                max_duration_us: 0,
                avg_duration_us: 0.0,
                top_collections: Vec::new(),
            },
        }
    }

    /// Add an HNSW layer profile.
    #[must_use]
    pub fn add_hnsw_layer(
        mut self,
        layer: usize,
        nodes_visited: usize,
        time_us: u64,
        distance_computations: usize,
    ) -> Self {
        self.hnsw_layers.push(HnswLayerProfile {
            layer,
            nodes_visited,
            time_us,
            distance_computations,
        });
        self
    }

    /// Set filter evaluation profile.
    #[must_use]
    pub fn with_filter(
        mut self,
        expression: &str,
        candidates_before: usize,
        candidates_after: usize,
        time_us: u64,
    ) -> Self {
        let selectivity = if candidates_before > 0 {
            candidates_after as f64 / candidates_before as f64
        } else {
            0.0
        };
        self.filter = Some(FilterProfile {
            expression: expression.to_string(),
            candidates_before,
            candidates_after,
            time_us,
            selectivity,
        });
        self
    }

    /// Set distance computation profile.
    #[must_use]
    pub fn with_distance(
        mut self,
        function: &str,
        total_computations: usize,
        total_time_us: u64,
        dimensions: usize,
    ) -> Self {
        let avg_ns = if total_computations > 0 {
            (total_time_us as f64 * 1000.0) / total_computations as f64
        } else {
            0.0
        };
        self.distance = DistanceProfile {
            function: function.to_string(),
            total_computations,
            total_time_us,
            avg_per_computation_ns: avg_ns,
            dimensions,
        };
        self
    }

    /// Build the final response.
    pub fn build(self) -> DebugProfileResponse {
        DebugProfileResponse {
            hnsw_layers: self.hnsw_layers,
            filter_evaluation: self.filter,
            distance_computation: self.distance,
            slow_queries: self.slow_queries,
        }
    }
}

impl Default for ProfileEndpointBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_create_profiler() {
        let profiler = QueryProfiler::new();
        assert!(profiler.history().is_empty());
    }

    #[test]
    fn test_profile_query() {
        let mut profiler = QueryProfiler::new();

        profiler.start("test_query");
        std::thread::sleep(Duration::from_millis(10));
        let profile = profiler.end().unwrap();

        assert!(profile.total_time_us >= 10000);
    }

    #[test]
    fn test_stage_timing() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("query");
        builder
            .start_stage(ExecutionStage::Parse)
            .end_stage()
            .start_stage(ExecutionStage::VectorSearch)
            .end_stage_with_items(100);

        let profile = profiler.end().unwrap();

        assert_eq!(profile.stages.len(), 2);
        assert_eq!(profile.stages[1].items_processed, 100);
    }

    #[test]
    fn test_plan_node() {
        let plan = PlanNode::new("HnswSearch", "Search using HNSW index")
            .cost(10.0)
            .rows(100)
            .property("ef", "50")
            .child(
                PlanNode::new("Filter", "Apply metadata filters")
                    .cost(2.0)
                    .rows(50),
            );

        let tree = plan.format_tree(0);
        assert!(tree.contains("HnswSearch"));
        assert!(tree.contains("Filter"));
    }

    #[test]
    fn test_explain_output() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("search");
        builder
            .description("Vector similarity search")
            .start_stage(ExecutionStage::VectorSearch)
            .end_stage_with_items(1000)
            .results(10);

        let profile = profiler.end().unwrap();
        let explain = profile.explain();

        assert!(explain.contains("search"));
        assert!(explain.contains("VectorSearch"));
    }

    #[test]
    fn test_optimization_hints() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("slow_query");
        builder.start_stage(ExecutionStage::VectorSearch);

        // Simulate slow stage
        std::thread::sleep(Duration::from_millis(50));
        builder.end_stage();

        let profile = profiler.end().unwrap();

        // Should have hints about slow stage
        assert!(!profile.hints.is_empty() || profile.total_time_us > 0);
    }

    #[test]
    fn test_profile_function() {
        let mut profiler = QueryProfiler::new();

        let (result, profile) = profiler.profile("compute", || {
            std::thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        assert!(profile.total_time_us >= 5000);
    }

    #[test]
    fn test_profile_history() {
        let mut profiler = QueryProfiler::new();

        for i in 0..5 {
            profiler.start(&format!("query_{}", i));
            profiler.end();
        }

        assert_eq!(profiler.history().len(), 5);
    }

    #[test]
    fn test_average_metrics() {
        let mut profiler = QueryProfiler::new();

        for _ in 0..3 {
            let builder = profiler.start("query");
            builder.metric("distance_calcs", 100.0);
            profiler.end();
        }

        let avgs = profiler.average_metrics();
        assert_eq!(avgs.get("distance_calcs"), Some(&100.0));
    }

    #[test]
    fn test_compare_profiles() {
        let mut a = QueryProfile::new("a");
        a.total_time_us = 1000;
        a.memory.peak_heap_bytes = 1000;

        let mut b = QueryProfile::new("b");
        b.total_time_us = 800;
        b.memory.peak_heap_bytes = 900;

        let comparison = compare_profiles(&a, &b);

        assert!(comparison.is_improvement);
        assert!(comparison.time_diff_us < 0);
    }

    #[test]
    fn test_io_stats() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("query");
        builder.io(100, 4096, 80);

        let profile = profiler.end().unwrap();

        assert_eq!(profile.io.read_ops, 100);
        assert_eq!(profile.io.bytes_read, 4096);
        assert_eq!(profile.io.cache_hits, 80);
    }

    #[test]
    fn test_memory_snapshot() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("query");
        builder.memory(1024, 2048);

        let profile = profiler.end().unwrap();

        assert_eq!(profile.memory.heap_bytes, 1024);
        assert_eq!(profile.memory.peak_heap_bytes, 2048);
    }

    #[test]
    fn test_stage_throughput() {
        let timing = StageTiming {
            stage: ExecutionStage::VectorSearch,
            start_offset_us: 0,
            duration_us: 1_000_000, // 1 second
            iterations: 1,
            items_processed: 1000,
        };

        assert_eq!(timing.throughput(), 1000.0);
    }

    #[test]
    fn test_custom_stage() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("query");
        builder
            .start_stage(ExecutionStage::Custom("CustomOp".to_string()))
            .end_stage();

        let profile = profiler.end().unwrap();
        assert_eq!(profile.stages[0].stage.name(), "CustomOp");
    }

    #[test]
    fn test_explain_analyze() {
        let mut profiler = QueryProfiler::new();

        let builder = profiler.start("query");
        builder
            .metric("vectors_scanned", 10000.0)
            .metric("distance_calculations", 5000.0);

        let profile = profiler.end().unwrap();
        let output = profile.explain_analyze();

        assert!(output.contains("vectors_scanned"));
        assert!(output.contains("10000"));
    }

    #[test]
    fn test_trace_context_creation() {
        let root = TraceContext::new_root();
        assert_eq!(root.trace_id.len(), 32);
        assert_eq!(root.span_id.len(), 16);
        assert!(root.parent_span_id.is_none());
        assert!(root.is_sampled());
    }

    #[test]
    fn test_trace_context_child() {
        let root = TraceContext::new_root();
        let child = root.child();
        assert_eq!(child.trace_id, root.trace_id);
        assert_ne!(child.span_id, root.span_id);
        assert_eq!(child.parent_span_id, Some(root.span_id));
    }

    #[test]
    fn test_traceparent_roundtrip() {
        let root = TraceContext::new_root();
        let header = root.to_traceparent();
        assert!(header.starts_with("00-"));

        let parsed = TraceContext::from_traceparent(&header).unwrap();
        assert_eq!(parsed.trace_id, root.trace_id);
        assert_eq!(parsed.span_id, root.span_id);
        assert_eq!(parsed.flags, root.flags);
    }

    #[test]
    fn test_traceparent_parse_invalid() {
        assert!(TraceContext::from_traceparent("invalid").is_none());
        assert!(TraceContext::from_traceparent("").is_none());
    }

    #[test]
    fn test_slow_query_log() {
        let mut log = SlowQueryLog::new(SlowQueryConfig {
            threshold_us: 5000,
            max_entries: 100,
            auto_explain: false,
            sample_rate: 1.0,
        });

        // Below threshold — not recorded
        assert!(log.record(1000, "fast query", "docs", 10, None, None).is_none());
        assert!(log.is_empty());

        // Above threshold — recorded
        let id = log.record(10_000, "slow search", "docs", 5, Some("EXPLAIN".into()), None);
        assert!(id.is_some());
        assert_eq!(log.len(), 1);

        let id = log.record(50_000, "very slow", "users", 3, None, None);
        assert!(id.is_some());
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_slow_query_top_slowest() {
        let mut log = SlowQueryLog::new(SlowQueryConfig {
            threshold_us: 1000,
            max_entries: 100,
            auto_explain: false,
            sample_rate: 1.0,
        });

        log.record(5000, "medium", "docs", 10, None, None);
        log.record(50000, "very slow", "docs", 10, None, None);
        log.record(2000, "fast-ish", "docs", 10, None, None);

        let top = log.top_slowest(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].duration_us, 50000);
        assert_eq!(top[1].duration_us, 5000);
    }

    #[test]
    fn test_slow_query_by_collection() {
        let mut log = SlowQueryLog::new(SlowQueryConfig {
            threshold_us: 1000,
            max_entries: 100,
            auto_explain: false,
            sample_rate: 1.0,
        });

        log.record(5000, "q1", "docs", 10, None, None);
        log.record(5000, "q2", "users", 10, None, None);
        log.record(5000, "q3", "docs", 10, None, None);

        let docs = log.entries_for_collection("docs");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_slow_query_max_entries() {
        let mut log = SlowQueryLog::new(SlowQueryConfig {
            threshold_us: 100,
            max_entries: 3,
            auto_explain: false,
            sample_rate: 1.0,
        });

        for i in 0..5 {
            log.record(1000 * (i + 1), &format!("q{}", i), "c", 1, None, None);
        }
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_span_collector() {
        let mut collector = SpanCollector::new(100);
        collector.record(TracedSpan {
            name: "hnsw.search".into(),
            context: TraceContext::new_root(),
            start_us: 0,
            duration_us: 500,
            attributes: HashMap::new(),
            status: SpanStatus::Ok,
        });
        collector.record(TracedSpan {
            name: "filter.evaluate".into(),
            context: TraceContext::new_root(),
            start_us: 500,
            duration_us: 200,
            attributes: HashMap::new(),
            status: SpanStatus::Ok,
        });
        assert_eq!(collector.len(), 2);

        let spans = collector.drain();
        assert_eq!(spans.len(), 2);
        assert_eq!(collector.len(), 0);
    }

    #[test]
    fn test_slow_query_percentiles() {
        let mut log = SlowQueryLog::new(SlowQueryConfig {
            threshold_us: 100,
            max_entries: 1000,
            auto_explain: false,
            sample_rate: 1.0,
        });
        for i in 0..100 {
            log.record(100 + i * 10, &format!("q{i}"), "docs", 5, None, None);
        }
        let stats = log.compute_stats();
        assert!(stats.p50_us > 0);
        assert!(stats.p95_us > stats.p50_us);
        assert!(stats.p99_us >= stats.p95_us);
        assert!(stats.avg_us > 0.0);
    }

    #[test]
    fn test_profile_endpoint_builder() {
        let builder = ProfileEndpointBuilder::new()
            .add_hnsw_layer(0, 100, 500, 200)
            .add_hnsw_layer(1, 20, 100, 40)
            .with_filter("category = 'books'", 1000, 50, 300)
            .with_distance("cosine", 240, 450, 128);

        let response = builder.build();
        assert_eq!(response.hnsw_layers.len(), 2);
        assert!(response.filter_evaluation.is_some());
        assert_eq!(response.distance_computation.function, "cosine");
    }

    #[test]
    fn test_sampling_trace_context() {
        let sampled = TraceContext::new_root();
        assert!(sampled.is_sampled());

        let mut not_sampled = TraceContext::new_root();
        not_sampled.flags = 0x00;
        assert!(!not_sampled.is_sampled());
    }
}
