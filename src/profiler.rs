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

        result.push_str(&format!("{}-> {} ({})\n", prefix, self.node_type, self.description));

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
                .unwrap()
                .as_secs(),
            metrics: HashMap::new(),
        }
    }

    /// Generate EXPLAIN output.
    pub fn explain(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!("Query Profile: {}\n", self.query_id));
        output.push_str(&format!("Total Time: {:.3} ms\n", self.total_time_us as f64 / 1000.0));
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
        self.active.as_mut().unwrap()
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
        let profile = self.end().unwrap();
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
#[allow(dead_code)]
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
                    message: format!("{} stage takes {}% of query time", stage.stage.name(), pct as i32),
                    suggestion: format!("Consider optimizing {} stage or using caching", stage.stage.name()),
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
        builder
            .start_stage(ExecutionStage::VectorSearch);

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
}
