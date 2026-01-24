//! Integrated Evaluation & Benchmarking Suite
//!
//! Built-in recall@k, precision, latency, throughput benchmarking with support
//! for standard datasets (SIFT1M, GloVe, GIST). Includes HTML report generation
//! and regression detection.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::benchmark_suite::{
//!     BenchmarkEngine, BenchmarkConfig, BenchmarkRun, DatasetConfig,
//! };
//!
//! let mut engine = BenchmarkEngine::new(BenchmarkConfig::default());
//!
//! // Add ground truth for recall computation
//! engine.add_ground_truth("q0", vec!["v1".into(), "v2".into(), "v3".into()]);
//!
//! // Record search results
//! engine.record_search("q0", vec!["v1".into(), "v3".into(), "v5".into()], 1200);
//!
//! // Compute metrics
//! let metrics = engine.compute_metrics();
//! println!("Recall@3: {:.3}", metrics.recall_at_k);
//! println!("P99 latency: {}μs", metrics.p99_latency_us);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Dataset Configuration ───────────────────────────────────────────────────

/// Standard benchmark dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StandardDataset {
    /// SIFT1M: 1M 128-dim SIFT descriptors.
    Sift1M,
    /// GloVe: 1.2M 100-dim word vectors.
    Glove100,
    /// GloVe 200-dim.
    Glove200,
    /// GIST: 1M 960-dim GIST descriptors.
    Gist1M,
    /// Deep1M: 1M 96-dim deep descriptors.
    Deep1M,
    /// Random: synthetic random vectors.
    Random,
}

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset name.
    pub name: String,
    /// Standard dataset type.
    pub dataset_type: StandardDataset,
    /// Number of base vectors.
    pub num_vectors: usize,
    /// Number of query vectors.
    pub num_queries: usize,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Number of ground-truth nearest neighbors per query.
    pub ground_truth_k: usize,
}

impl DatasetConfig {
    /// Get config for a standard dataset.
    pub fn standard(dataset: StandardDataset) -> Self {
        match dataset {
            StandardDataset::Sift1M => Self {
                name: "SIFT1M".into(),
                dataset_type: dataset,
                num_vectors: 1_000_000,
                num_queries: 10_000,
                dimensions: 128,
                ground_truth_k: 100,
            },
            StandardDataset::Glove100 => Self {
                name: "GloVe-100".into(),
                dataset_type: dataset,
                num_vectors: 1_183_514,
                num_queries: 10_000,
                dimensions: 100,
                ground_truth_k: 100,
            },
            StandardDataset::Glove200 => Self {
                name: "GloVe-200".into(),
                dataset_type: dataset,
                num_vectors: 1_183_514,
                num_queries: 10_000,
                dimensions: 200,
                ground_truth_k: 100,
            },
            StandardDataset::Gist1M => Self {
                name: "GIST1M".into(),
                dataset_type: dataset,
                num_vectors: 1_000_000,
                num_queries: 1_000,
                dimensions: 960,
                ground_truth_k: 100,
            },
            StandardDataset::Deep1M => Self {
                name: "Deep1M".into(),
                dataset_type: dataset,
                num_vectors: 1_000_000,
                num_queries: 10_000,
                dimensions: 96,
                ground_truth_k: 100,
            },
            StandardDataset::Random => Self {
                name: "Random".into(),
                dataset_type: dataset,
                num_vectors: 100_000,
                num_queries: 1_000,
                dimensions: 128,
                ground_truth_k: 10,
            },
        }
    }
}

// ── Metrics ─────────────────────────────────────────────────────────────────

/// Computed benchmark metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Recall@k: fraction of true neighbors found.
    pub recall_at_k: f64,
    /// Precision@k: fraction of returned results that are true neighbors.
    pub precision_at_k: f64,
    /// k value used.
    pub k: usize,
    /// Mean query latency (microseconds).
    pub mean_latency_us: f64,
    /// Median query latency (microseconds).
    pub median_latency_us: u64,
    /// P95 latency (microseconds).
    pub p95_latency_us: u64,
    /// P99 latency (microseconds).
    pub p99_latency_us: u64,
    /// Minimum latency (microseconds).
    pub min_latency_us: u64,
    /// Maximum latency (microseconds).
    pub max_latency_us: u64,
    /// Queries per second.
    pub qps: f64,
    /// Total queries evaluated.
    pub total_queries: usize,
    /// Total time (microseconds).
    pub total_time_us: u64,
}

impl Default for BenchmarkMetrics {
    fn default() -> Self {
        Self {
            recall_at_k: 0.0,
            precision_at_k: 0.0,
            k: 10,
            mean_latency_us: 0.0,
            median_latency_us: 0,
            p95_latency_us: 0,
            p99_latency_us: 0,
            min_latency_us: 0,
            max_latency_us: 0,
            qps: 0.0,
            total_queries: 0,
            total_time_us: 0,
        }
    }
}

// ── Benchmark Run ───────────────────────────────────────────────────────────

/// A completed benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRun {
    /// Run identifier.
    pub run_id: String,
    /// Run label/description.
    pub label: String,
    /// Dataset used.
    pub dataset: String,
    /// HNSW parameters used.
    pub hnsw_params: HnswParams,
    /// Computed metrics.
    pub metrics: BenchmarkMetrics,
    /// Timestamp of run.
    pub timestamp: String,
}

/// HNSW parameters for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

// ── Regression Detection ────────────────────────────────────────────────────

/// Regression detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Whether a regression was detected.
    pub regression_detected: bool,
    /// Specific regressions found.
    pub regressions: Vec<Regression>,
    /// Improvements found.
    pub improvements: Vec<Improvement>,
}

/// A detected regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Regression {
    /// Metric name.
    pub metric: String,
    /// Baseline value.
    pub baseline: f64,
    /// Current value.
    pub current: f64,
    /// Percentage change.
    pub change_pct: f64,
    /// Severity.
    pub severity: Severity,
}

/// A detected improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Improvement {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
    pub change_pct: f64,
}

/// Regression severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

// ── Config ──────────────────────────────────────────────────────────────────

/// Benchmark engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Default k for recall/precision.
    pub default_k: usize,
    /// Regression threshold for latency (percentage increase).
    pub latency_regression_threshold: f64,
    /// Regression threshold for recall (percentage decrease).
    pub recall_regression_threshold: f64,
    /// Maximum benchmark history to retain.
    pub max_history: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            default_k: 10,
            latency_regression_threshold: 10.0,
            recall_regression_threshold: 2.0,
            max_history: 100,
        }
    }
}

// ── Benchmark Engine ────────────────────────────────────────────────────────

/// Search result for a query.
struct QueryResult {
    query_id: String,
    result_ids: Vec<String>,
    latency_us: u64,
}

/// Benchmark engine for evaluation and regression detection.
pub struct BenchmarkEngine {
    config: BenchmarkConfig,
    ground_truth: HashMap<String, Vec<String>>,
    results: Vec<QueryResult>,
    history: Vec<BenchmarkRun>,
    run_count: u64,
}

impl BenchmarkEngine {
    /// Create a new benchmark engine.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            ground_truth: HashMap::new(),
            results: Vec::new(),
            history: Vec::new(),
            run_count: 0,
        }
    }

    /// Add ground truth neighbors for a query.
    pub fn add_ground_truth(&mut self, query_id: &str, neighbor_ids: Vec<String>) {
        self.ground_truth
            .insert(query_id.to_string(), neighbor_ids);
    }

    /// Record a search result.
    pub fn record_search(&mut self, query_id: &str, result_ids: Vec<String>, latency_us: u64) {
        self.results.push(QueryResult {
            query_id: query_id.to_string(),
            result_ids,
            latency_us,
        });
    }

    /// Clear recorded results (but keep ground truth).
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Compute metrics from recorded results.
    pub fn compute_metrics(&self) -> BenchmarkMetrics {
        if self.results.is_empty() {
            return BenchmarkMetrics::default();
        }

        let k = self.config.default_k;
        let mut recall_sum = 0.0;
        let mut precision_sum = 0.0;
        let mut recall_count = 0;

        for result in &self.results {
            if let Some(gt) = self.ground_truth.get(&result.query_id) {
                let gt_k: std::collections::HashSet<_> = gt.iter().take(k).collect();
                let result_k: std::collections::HashSet<_> = result.result_ids.iter().take(k).collect();

                let hits = gt_k.intersection(&result_k).count();
                if !gt_k.is_empty() {
                    recall_sum += hits as f64 / gt_k.len() as f64;
                }
                if !result_k.is_empty() {
                    precision_sum += hits as f64 / result_k.len() as f64;
                }
                recall_count += 1;
            }
        }

        let recall_at_k = if recall_count > 0 {
            recall_sum / recall_count as f64
        } else {
            0.0
        };
        let precision_at_k = if recall_count > 0 {
            precision_sum / recall_count as f64
        } else {
            0.0
        };

        // Latency statistics
        let mut latencies: Vec<u64> = self.results.iter().map(|r| r.latency_us).collect();
        latencies.sort_unstable();
        let total_time_us: u64 = latencies.iter().sum();
        let n = latencies.len();

        let mean_latency_us = total_time_us as f64 / n as f64;
        let median_latency_us = latencies[n / 2];
        let p95_latency_us = latencies[(n as f64 * 0.95) as usize];
        let p99_latency_us = latencies[(n as f64 * 0.99).min((n - 1) as f64) as usize];
        let min_latency_us = latencies[0];
        let max_latency_us = latencies[n - 1];
        let qps = if total_time_us > 0 {
            n as f64 / (total_time_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        BenchmarkMetrics {
            recall_at_k,
            precision_at_k,
            k,
            mean_latency_us,
            median_latency_us,
            p95_latency_us,
            p99_latency_us,
            min_latency_us,
            max_latency_us,
            qps,
            total_queries: n,
            total_time_us,
        }
    }

    /// Save a benchmark run to history.
    pub fn save_run(&mut self, label: &str, dataset: &str, hnsw_params: HnswParams) -> String {
        let metrics = self.compute_metrics();
        self.run_count += 1;
        let run_id = format!("run-{:04}", self.run_count);

        let run = BenchmarkRun {
            run_id: run_id.clone(),
            label: label.to_string(),
            dataset: dataset.to_string(),
            hnsw_params,
            metrics,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        if self.history.len() >= self.config.max_history {
            self.history.remove(0);
        }
        self.history.push(run);
        run_id
    }

    /// Get benchmark history.
    pub fn history(&self) -> &[BenchmarkRun] {
        &self.history
    }

    /// Compare current metrics against a baseline run.
    pub fn detect_regression(&self, baseline_run_id: &str) -> Result<RegressionResult> {
        let baseline = self
            .history
            .iter()
            .find(|r| r.run_id == baseline_run_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Run '{baseline_run_id}'")))?;

        let current = self.compute_metrics();
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        // Check latency regression (increase is bad)
        let latency_change = if baseline.metrics.mean_latency_us > 0.0 {
            ((current.mean_latency_us - baseline.metrics.mean_latency_us)
                / baseline.metrics.mean_latency_us)
                * 100.0
        } else {
            0.0
        };
        if latency_change > self.config.latency_regression_threshold {
            regressions.push(Regression {
                metric: "mean_latency_us".into(),
                baseline: baseline.metrics.mean_latency_us,
                current: current.mean_latency_us,
                change_pct: latency_change,
                severity: if latency_change > 50.0 {
                    Severity::Critical
                } else if latency_change > 25.0 {
                    Severity::High
                } else {
                    Severity::Medium
                },
            });
        } else if latency_change < -self.config.latency_regression_threshold {
            improvements.push(Improvement {
                metric: "mean_latency_us".into(),
                baseline: baseline.metrics.mean_latency_us,
                current: current.mean_latency_us,
                change_pct: latency_change,
            });
        }

        // Check recall regression (decrease is bad)
        let recall_change = if baseline.metrics.recall_at_k > 0.0 {
            ((current.recall_at_k - baseline.metrics.recall_at_k) / baseline.metrics.recall_at_k)
                * 100.0
        } else {
            0.0
        };
        if recall_change < -self.config.recall_regression_threshold {
            regressions.push(Regression {
                metric: "recall_at_k".into(),
                baseline: baseline.metrics.recall_at_k,
                current: current.recall_at_k,
                change_pct: recall_change,
                severity: if recall_change < -10.0 {
                    Severity::Critical
                } else if recall_change < -5.0 {
                    Severity::High
                } else {
                    Severity::Medium
                },
            });
        } else if recall_change > self.config.recall_regression_threshold {
            improvements.push(Improvement {
                metric: "recall_at_k".into(),
                baseline: baseline.metrics.recall_at_k,
                current: current.recall_at_k,
                change_pct: recall_change,
            });
        }

        // Check QPS regression
        let qps_change = if baseline.metrics.qps > 0.0 {
            ((current.qps - baseline.metrics.qps) / baseline.metrics.qps) * 100.0
        } else {
            0.0
        };
        if qps_change < -self.config.latency_regression_threshold {
            regressions.push(Regression {
                metric: "qps".into(),
                baseline: baseline.metrics.qps,
                current: current.qps,
                change_pct: qps_change,
                severity: Severity::Medium,
            });
        } else if qps_change > self.config.latency_regression_threshold {
            improvements.push(Improvement {
                metric: "qps".into(),
                baseline: baseline.metrics.qps,
                current: current.qps,
                change_pct: qps_change,
            });
        }

        Ok(RegressionResult {
            regression_detected: !regressions.is_empty(),
            regressions,
            improvements,
        })
    }

    /// Generate an HTML benchmark report.
    pub fn generate_html_report(&self) -> String {
        let metrics = self.compute_metrics();
        let mut html = String::from(
            r#"<!DOCTYPE html>
<html><head><title>Needle Benchmark Report</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
h1 { color: #1a1a2e; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
th { background: #1a1a2e; color: white; }
tr:nth-child(even) { background: #f9f9f9; }
.good { color: #28a745; } .bad { color: #dc3545; }
.metric-card { display: inline-block; padding: 20px; margin: 10px; border-radius: 8px; background: #f0f0f0; min-width: 150px; }
.metric-value { font-size: 2em; font-weight: bold; color: #1a1a2e; }
.metric-label { color: #666; }
</style></head><body>
<h1>🎯 Needle Benchmark Report</h1>
"#,
        );

        html.push_str("<div>\n");
        html.push_str(&format!(
            "<div class='metric-card'><div class='metric-value'>{:.3}</div><div class='metric-label'>Recall@{}</div></div>\n",
            metrics.recall_at_k, metrics.k
        ));
        html.push_str(&format!(
            "<div class='metric-card'><div class='metric-value'>{:.3}</div><div class='metric-label'>Precision@{}</div></div>\n",
            metrics.precision_at_k, metrics.k
        ));
        html.push_str(&format!(
            "<div class='metric-card'><div class='metric-value'>{:.0}μs</div><div class='metric-label'>Mean Latency</div></div>\n",
            metrics.mean_latency_us
        ));
        html.push_str(&format!(
            "<div class='metric-card'><div class='metric-value'>{:.0}</div><div class='metric-label'>QPS</div></div>\n",
            metrics.qps
        ));
        html.push_str("</div>\n\n");

        html.push_str("<h2>Latency Distribution</h2>\n<table>\n");
        html.push_str("<tr><th>Metric</th><th>Value (μs)</th></tr>\n");
        html.push_str(&format!("<tr><td>Min</td><td>{}</td></tr>\n", metrics.min_latency_us));
        html.push_str(&format!("<tr><td>Median</td><td>{}</td></tr>\n", metrics.median_latency_us));
        html.push_str(&format!("<tr><td>Mean</td><td>{:.0}</td></tr>\n", metrics.mean_latency_us));
        html.push_str(&format!("<tr><td>P95</td><td>{}</td></tr>\n", metrics.p95_latency_us));
        html.push_str(&format!("<tr><td>P99</td><td>{}</td></tr>\n", metrics.p99_latency_us));
        html.push_str(&format!("<tr><td>Max</td><td>{}</td></tr>\n", metrics.max_latency_us));
        html.push_str("</table>\n");

        if !self.history.is_empty() {
            html.push_str("<h2>Run History</h2>\n<table>\n");
            html.push_str("<tr><th>Run</th><th>Dataset</th><th>Recall</th><th>P99 (μs)</th><th>QPS</th></tr>\n");
            for run in &self.history {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.3}</td><td>{}</td><td>{:.0}</td></tr>\n",
                    run.label,
                    run.dataset,
                    run.metrics.recall_at_k,
                    run.metrics.p99_latency_us,
                    run.metrics.qps
                ));
            }
            html.push_str("</table>\n");
        }

        html.push_str("</body></html>");
        html
    }

    /// Get config.
    pub fn config(&self) -> &BenchmarkConfig {
        &self.config
    }

    /// Get ground truth count.
    pub fn ground_truth_count(&self) -> usize {
        self.ground_truth.len()
    }

    /// Get recorded result count.
    pub fn result_count(&self) -> usize {
        self.results.len()
    }
}

impl Default for BenchmarkEngine {
    fn default() -> Self {
        Self::new(BenchmarkConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> BenchmarkEngine {
        BenchmarkEngine::new(BenchmarkConfig::default())
    }

    #[test]
    fn test_recall_perfect() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into(), "v1".into(), "v2".into()]);
        e.record_search("q0", vec!["v0".into(), "v1".into(), "v2".into()], 100);
        let m = e.compute_metrics();
        assert!((m.recall_at_k - 1.0).abs() < 0.01);
        assert!((m.precision_at_k - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_recall_partial() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into(), "v1".into(), "v2".into(), "v3".into()]);
        e.record_search("q0", vec!["v0".into(), "v1".into(), "vX".into(), "vY".into()], 100);
        let m = e.compute_metrics();
        assert!((m.recall_at_k - 0.5).abs() < 0.01); // 2 out of 4
    }

    #[test]
    fn test_latency_stats() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        for i in 0..100 {
            e.record_search("q0", vec!["v0".into()], (i + 1) * 100);
        }
        let m = e.compute_metrics();
        assert!(m.p99_latency_us > m.p95_latency_us);
        assert!(m.p95_latency_us > m.median_latency_us);
        assert!(m.max_latency_us >= m.p99_latency_us);
        assert!(m.min_latency_us <= m.median_latency_us);
    }

    #[test]
    fn test_qps_calculation() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        for _ in 0..1000 {
            e.record_search("q0", vec!["v0".into()], 1000); // 1ms each
        }
        let m = e.compute_metrics();
        assert!(m.qps > 0.0);
    }

    #[test]
    fn test_empty_metrics() {
        let e = make_engine();
        let m = e.compute_metrics();
        assert_eq!(m.total_queries, 0);
        assert_eq!(m.recall_at_k, 0.0);
    }

    #[test]
    fn test_save_and_history() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 100);
        let run_id = e.save_run("baseline", "SIFT1M", HnswParams::default());
        assert!(run_id.starts_with("run-"));
        assert_eq!(e.history().len(), 1);
    }

    #[test]
    fn test_regression_detection_no_regression() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 100);
        let run_id = e.save_run("baseline", "test", HnswParams::default());

        e.clear_results();
        e.record_search("q0", vec!["v0".into()], 100);
        let result = e.detect_regression(&run_id).unwrap();
        assert!(!result.regression_detected);
    }

    #[test]
    fn test_regression_detection_latency() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 100);
        let run_id = e.save_run("baseline", "test", HnswParams::default());

        e.clear_results();
        e.record_search("q0", vec!["v0".into()], 500); // 5x slower
        let result = e.detect_regression(&run_id).unwrap();
        assert!(result.regression_detected);
    }

    #[test]
    fn test_html_report() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 100);
        e.save_run("test", "Random", HnswParams::default());

        let html = e.generate_html_report();
        assert!(html.contains("Needle Benchmark Report"));
        assert!(html.contains("Recall"));
        assert!(html.contains("Latency"));
    }

    #[test]
    fn test_dataset_configs() {
        let sift = DatasetConfig::standard(StandardDataset::Sift1M);
        assert_eq!(sift.dimensions, 128);
        assert_eq!(sift.num_vectors, 1_000_000);

        let glove = DatasetConfig::standard(StandardDataset::Glove100);
        assert_eq!(glove.dimensions, 100);

        let gist = DatasetConfig::standard(StandardDataset::Gist1M);
        assert_eq!(gist.dimensions, 960);
    }

    #[test]
    fn test_clear_results() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 100);
        assert_eq!(e.result_count(), 1);
        e.clear_results();
        assert_eq!(e.result_count(), 0);
        assert_eq!(e.ground_truth_count(), 1); // preserved
    }

    #[test]
    fn test_regression_missing_baseline() {
        let e = make_engine();
        assert!(e.detect_regression("nonexistent").is_err());
    }

    #[test]
    fn test_multiple_queries() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into(), "v1".into()]);
        e.add_ground_truth("q1", vec!["v2".into(), "v3".into()]);
        e.record_search("q0", vec!["v0".into(), "v1".into()], 100);
        e.record_search("q1", vec!["v2".into(), "vX".into()], 200);
        let m = e.compute_metrics();
        assert!((m.recall_at_k - 0.75).abs() < 0.01); // (1.0 + 0.5) / 2
    }
}
