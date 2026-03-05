#![allow(clippy::unwrap_used)]
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
    /// Peak memory usage in bytes (if tracked).
    pub peak_memory_bytes: u64,
    /// Average memory usage in bytes (if tracked).
    pub avg_memory_bytes: f64,
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
            peak_memory_bytes: 0,
            avg_memory_bytes: 0.0,
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
    memory_bytes: Option<u64>,
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
            memory_bytes: None,
        });
    }

    /// Record a search result with memory usage.
    pub fn record_search_with_memory(
        &mut self,
        query_id: &str,
        result_ids: Vec<String>,
        latency_us: u64,
        memory_bytes: u64,
    ) {
        self.results.push(QueryResult {
            query_id: query_id.to_string(),
            result_ids,
            latency_us,
            memory_bytes: Some(memory_bytes),
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

        // Memory tracking
        let memory_samples: Vec<u64> = self.results.iter().filter_map(|r| r.memory_bytes).collect();
        let peak_memory_bytes = memory_samples.iter().copied().max().unwrap_or(0);
        let avg_memory_bytes = if memory_samples.is_empty() {
            0.0
        } else {
            memory_samples.iter().sum::<u64>() as f64 / memory_samples.len() as f64
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
            peak_memory_bytes,
            avg_memory_bytes,
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

// ── Competitor Comparison ───────────────────────────────────────────────────

/// Result from a competitor system (Qdrant, ChromaDB, LanceDB, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorResult {
    /// Competitor name.
    pub name: String,
    /// Dataset used.
    pub dataset: String,
    /// Metrics from the competitor.
    pub metrics: BenchmarkMetrics,
    /// Competitor version.
    pub version: String,
}

/// Comparison between Needle and a competitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorComparison {
    /// Name of the competitor.
    pub competitor_name: String,
    /// Recall difference (Needle - competitor).
    pub recall_diff: f64,
    /// Latency ratio (Needle / competitor). < 1.0 means Needle is faster.
    pub latency_ratio: f64,
    /// QPS ratio (Needle / competitor). > 1.0 means Needle is faster.
    pub qps_ratio: f64,
    /// Overall verdict.
    pub verdict: String,
}

/// Compare Needle metrics against a competitor.
pub fn compare_with_competitor(
    needle: &BenchmarkMetrics,
    competitor: &CompetitorResult,
) -> CompetitorComparison {
    let recall_diff = needle.recall_at_k - competitor.metrics.recall_at_k;
    let latency_ratio = if competitor.metrics.mean_latency_us > 0.0 {
        needle.mean_latency_us / competitor.metrics.mean_latency_us
    } else {
        0.0
    };
    let qps_ratio = if competitor.metrics.qps > 0.0 {
        needle.qps / competitor.metrics.qps
    } else {
        0.0
    };

    let verdict = if recall_diff >= 0.0 && latency_ratio <= 1.0 {
        "Needle wins on both recall and latency".into()
    } else if recall_diff >= 0.0 {
        "Needle has better recall, higher latency".into()
    } else if latency_ratio <= 1.0 {
        "Needle has lower latency, lower recall".into()
    } else {
        format!(
            "Competitor {} is ahead on measured metrics",
            competitor.name
        )
    };

    CompetitorComparison {
        competitor_name: competitor.name.clone(),
        recall_diff,
        latency_ratio,
        qps_ratio,
        verdict,
    }
}

// ── Badge Generation ────────────────────────────────────────────────────────

/// SVG badge for embedding in README or website.
#[derive(Debug, Clone)]
pub struct Badge {
    /// Badge label.
    pub label: String,
    /// Badge value.
    pub value: String,
    /// Badge color.
    pub color: String,
    /// SVG content.
    pub svg: String,
}

/// Generate SVG badges for benchmark results.
pub fn generate_badges(metrics: &BenchmarkMetrics, dataset: &str) -> Vec<Badge> {
    let mut badges = Vec::new();

    // Recall badge
    let recall_color = if metrics.recall_at_k >= 0.95 {
        "#4c1"
    } else if metrics.recall_at_k >= 0.90 {
        "#97ca00"
    } else {
        "#dfb317"
    };
    badges.push(make_badge(
        &format!("recall@{}", metrics.k),
        &format!("{:.1}%", metrics.recall_at_k * 100.0),
        recall_color,
    ));

    // P99 latency badge
    let latency_color = if metrics.p99_latency_us < 1000 {
        "#4c1"
    } else if metrics.p99_latency_us < 5000 {
        "#97ca00"
    } else {
        "#dfb317"
    };
    badges.push(make_badge(
        "p99 latency",
        &format!("{}μs", metrics.p99_latency_us),
        latency_color,
    ));

    // QPS badge
    let qps_color = if metrics.qps > 10_000.0 {
        "#4c1"
    } else if metrics.qps > 1_000.0 {
        "#97ca00"
    } else {
        "#dfb317"
    };
    badges.push(make_badge("QPS", &format!("{:.0}", metrics.qps), qps_color));

    // Dataset badge
    badges.push(make_badge("dataset", dataset, "#007ec6"));

    badges
}

fn make_badge(label: &str, value: &str, color: &str) -> Badge {
    let label_width = label.len() * 7 + 10;
    let value_width = value.len() * 7 + 10;
    let total_width = label_width + value_width;
    let lx = label_width / 2;
    let vx = label_width + value_width / 2;
    let fill_color = color;

    let svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width}\" height=\"20\">\
         <rect width=\"{label_width}\" height=\"20\" fill=\"#555\"/>\
         <rect x=\"{label_width}\" width=\"{value_width}\" height=\"20\" fill=\"{fill_color}\"/>\
         <text x=\"{lx}\" y=\"14\" fill=\"white\" font-size=\"11\" text-anchor=\"middle\">{label}</text>\
         <text x=\"{vx}\" y=\"14\" fill=\"white\" font-size=\"11\" text-anchor=\"middle\">{value}</text>\
         </svg>"
    );

    Badge {
        label: label.to_string(),
        value: value.to_string(),
        color: color.to_string(),
        svg,
    }
}

// ── ann-benchmarks Compatible Export ────────────────────────────────────────

impl BenchmarkEngine {
    /// Export results in ann-benchmarks compatible JSON format.
    pub fn export_ann_benchmarks_format(&self, algorithm: &str, version: &str) -> String {
        let metrics = self.compute_metrics();
        let runs: Vec<serde_json::Value> = self
            .history
            .iter()
            .map(|run| {
                serde_json::json!({
                    "algorithm": algorithm,
                    "version": version,
                    "dataset": run.dataset,
                    "parameters": {
                        "M": run.hnsw_params.m,
                        "ef_construction": run.hnsw_params.ef_construction,
                        "ef_search": run.hnsw_params.ef_search,
                    },
                    "recall": run.metrics.recall_at_k,
                    "qps": run.metrics.qps,
                    "mean_latency_us": run.metrics.mean_latency_us,
                    "p99_latency_us": run.metrics.p99_latency_us,
                    "k": run.metrics.k,
                })
            })
            .collect();

        let output = serde_json::json!({
            "algorithm": algorithm,
            "version": version,
            "current_metrics": {
                "recall": metrics.recall_at_k,
                "qps": metrics.qps,
                "mean_latency_us": metrics.mean_latency_us,
                "p99_latency_us": metrics.p99_latency_us,
            },
            "runs": runs,
        });

        serde_json::to_string_pretty(&output).unwrap_or_default()
    }

    /// Generate comparison HTML report including competitor data.
    pub fn generate_comparison_report(
        &self,
        competitors: &[CompetitorResult],
    ) -> String {
        let metrics = self.compute_metrics();
        let mut html = self.generate_html_report();

        // Insert competitor comparison table before closing body
        if !competitors.is_empty() {
            let mut table = String::from("<h2>🏁 Competitor Comparison</h2>\n<table>\n");
            table.push_str("<tr><th>System</th><th>Recall</th><th>Mean Latency (μs)</th><th>QPS</th><th>Verdict</th></tr>\n");

            // Add Needle row
            table.push_str(&format!(
                "<tr><td><strong>Needle</strong></td><td>{:.3}</td><td>{:.0}</td><td>{:.0}</td><td>-</td></tr>\n",
                metrics.recall_at_k, metrics.mean_latency_us, metrics.qps
            ));

            for comp in competitors {
                let comparison = compare_with_competitor(&metrics, comp);
                table.push_str(&format!(
                    "<tr><td>{}</td><td>{:.3}</td><td>{:.0}</td><td>{:.0}</td><td>{}</td></tr>\n",
                    comp.name,
                    comp.metrics.recall_at_k,
                    comp.metrics.mean_latency_us,
                    comp.metrics.qps,
                    comparison.verdict
                ));
            }
            table.push_str("</table>\n");

            html = html.replace("</body>", &format!("{table}</body>"));
        }

        html
    }
}

// ── Docker-based Benchmark Isolation ────────────────────────────────────────

/// Configuration for Docker-based benchmark isolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerBenchmarkConfig {
    /// CPU limit per container.
    pub cpu_limit: f64,
    /// Memory limit per container.
    pub memory_limit: String,
    /// Dataset mount path.
    pub dataset_path: String,
    /// Results output path.
    pub results_path: String,
    /// Competitors to benchmark.
    pub competitors: Vec<CompetitorDockerConfig>,
}

/// Docker config for a single competitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompetitorDockerConfig {
    /// Competitor name.
    pub name: String,
    /// Docker image.
    pub image: String,
    /// Port mapping.
    pub port: u16,
    /// Environment variables.
    pub env: HashMap<String, String>,
}

impl Default for DockerBenchmarkConfig {
    fn default() -> Self {
        Self {
            cpu_limit: 4.0,
            memory_limit: "8g".into(),
            dataset_path: "./benchmark-data".into(),
            results_path: "./benchmark-results".into(),
            competitors: vec![
                CompetitorDockerConfig {
                    name: "qdrant".into(),
                    image: "qdrant/qdrant:latest".into(),
                    port: 6333,
                    env: HashMap::new(),
                },
                CompetitorDockerConfig {
                    name: "chromadb".into(),
                    image: "chromadb/chroma:latest".into(),
                    port: 8000,
                    env: HashMap::new(),
                },
                CompetitorDockerConfig {
                    name: "lancedb".into(),
                    image: "lancedb/lancedb:latest".into(),
                    port: 8080,
                    env: HashMap::new(),
                },
            ],
        }
    }
}

impl DockerBenchmarkConfig {
    /// Generate a docker-compose.yml for benchmark isolation.
    pub fn generate_docker_compose(&self) -> String {
        let mut compose = String::from("services:\n");

        // Needle service
        compose.push_str(&format!(
            "  needle-bench:\n    build: .\n    cpus: {}\n    mem_limit: {}\n    volumes:\n      - {}:/data\n      - {}:/results\n",
            self.cpu_limit, self.memory_limit, self.dataset_path, self.results_path
        ));

        // Competitor services
        for comp in &self.competitors {
            compose.push_str(&format!(
                "  {}-bench:\n    image: {}\n    cpus: {}\n    mem_limit: {}\n    ports:\n      - \"{}:{}\"\n    volumes:\n      - {}:/data\n",
                comp.name, comp.image, self.cpu_limit, self.memory_limit,
                comp.port, comp.port, self.dataset_path
            ));
            if !comp.env.is_empty() {
                compose.push_str("    environment:\n");
                for (k, v) in &comp.env {
                    compose.push_str(&format!("      - {}={}\n", k, v));
                }
            }
        }

        compose
    }
}

// ── Pareto Frontier Analysis ─────────────────────────────────────────────────

/// Compute the Pareto frontier from a set of benchmark runs.
/// A run is Pareto-optimal if no other run has both better recall AND better QPS.
pub fn compute_pareto_frontier(runs: &[BenchmarkRun]) -> Vec<&BenchmarkRun> {
    let mut frontier: Vec<&BenchmarkRun> = Vec::new();

    for run in runs {
        let dominated = runs.iter().any(|other| {
            other.run_id != run.run_id
                && other.metrics.recall_at_k >= run.metrics.recall_at_k
                && other.metrics.qps >= run.metrics.qps
                && (other.metrics.recall_at_k > run.metrics.recall_at_k
                    || other.metrics.qps > run.metrics.qps)
        });
        if !dominated {
            frontier.push(run);
        }
    }

    frontier.sort_by(|a, b| {
        a.metrics
            .recall_at_k
            .partial_cmp(&b.metrics.recall_at_k)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    frontier
}

/// Estimate concurrent QPS from single-threaded metrics.
/// Uses Amdahl's law approximation: concurrent_qps ≈ single_qps × threads × efficiency.
pub fn estimate_concurrent_qps(metrics: &BenchmarkMetrics, num_threads: usize) -> f64 {
    let efficiency = 0.85; // Typical HNSW concurrency efficiency
    metrics.qps * num_threads as f64 * efficiency
}

// ── Markdown Report Generator ────────────────────────────────────────────────

impl BenchmarkEngine {
    /// Generate a markdown benchmark report.
    pub fn generate_markdown_report(&self) -> String {
        let metrics = self.compute_metrics();
        let mut md = String::new();

        md.push_str("## Needle Benchmark Report\n\n");

        md.push_str("### Key Metrics\n\n");
        md.push_str(&format!("| Metric | Value |\n|--------|-------|\n"));
        md.push_str(&format!("| Recall@{} | {:.3} |\n", metrics.k, metrics.recall_at_k));
        md.push_str(&format!("| Precision@{} | {:.3} |\n", metrics.k, metrics.precision_at_k));
        md.push_str(&format!("| QPS | {:.0} |\n", metrics.qps));
        md.push_str(&format!("| Mean Latency | {:.0}μs |\n", metrics.mean_latency_us));
        md.push_str(&format!("| P99 Latency | {}μs |\n", metrics.p99_latency_us));
        md.push_str(&format!("| Total Queries | {} |\n", metrics.total_queries));
        if metrics.peak_memory_bytes > 0 {
            md.push_str(&format!(
                "| Peak Memory | {:.1} MB |\n",
                metrics.peak_memory_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        md.push('\n');

        md.push_str("### Latency Distribution\n\n");
        md.push_str("| Percentile | Latency (μs) |\n|------------|-------------|\n");
        md.push_str(&format!("| Min | {} |\n", metrics.min_latency_us));
        md.push_str(&format!("| P50 | {} |\n", metrics.median_latency_us));
        md.push_str(&format!("| P95 | {} |\n", metrics.p95_latency_us));
        md.push_str(&format!("| P99 | {} |\n", metrics.p99_latency_us));
        md.push_str(&format!("| Max | {} |\n", metrics.max_latency_us));
        md.push('\n');

        if !self.history.is_empty() {
            md.push_str("### Run History\n\n");
            md.push_str("| Run | Dataset | Recall | P99 (μs) | QPS |\n");
            md.push_str("|-----|---------|--------|----------|-----|\n");
            for run in &self.history {
                md.push_str(&format!(
                    "| {} | {} | {:.3} | {} | {:.0} |\n",
                    run.label, run.dataset, run.metrics.recall_at_k,
                    run.metrics.p99_latency_us, run.metrics.qps
                ));
            }
            md.push('\n');

            // Pareto frontier
            let frontier = compute_pareto_frontier(&self.history);
            if frontier.len() > 1 {
                md.push_str("### Pareto Frontier (Recall vs QPS)\n\n");
                md.push_str("| Run | Recall | QPS |\n|-----|--------|-----|\n");
                for run in &frontier {
                    md.push_str(&format!(
                        "| {} | {:.3} | {:.0} |\n",
                        run.label, run.metrics.recall_at_k, run.metrics.qps
                    ));
                }
                md.push('\n');
            }
        }

        md
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

    #[test]
    fn test_competitor_comparison() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into(), "v1".into()]);
        e.record_search("q0", vec!["v0".into(), "v1".into()], 100);

        let needle_metrics = e.compute_metrics();
        let competitor = CompetitorResult {
            name: "Qdrant".into(),
            dataset: "SIFT1M".into(),
            metrics: BenchmarkMetrics {
                recall_at_k: 0.95,
                precision_at_k: 0.95,
                k: 10,
                mean_latency_us: 500.0,
                median_latency_us: 450,
                p95_latency_us: 800,
                p99_latency_us: 1200,
                min_latency_us: 100,
                max_latency_us: 5000,
                qps: 2000.0,
                total_queries: 10_000,
                total_time_us: 5_000_000,
                avg_memory_bytes: 0.0,
                peak_memory_bytes: 0,
            },
            version: "1.8.0".into(),
        };

        let comparison = compare_with_competitor(&needle_metrics, &competitor);
        assert_eq!(comparison.competitor_name, "Qdrant");
        assert!(comparison.latency_ratio != 0.0);
    }

    #[test]
    fn test_generate_badge() {
        let metrics = BenchmarkMetrics {
            recall_at_k: 0.95,
            p99_latency_us: 1200,
            qps: 5000.0,
            ..Default::default()
        };
        let badges = generate_badges(&metrics, "SIFT1M");
        assert!(badges.len() >= 3);
        for badge in &badges {
            assert!(badge.svg.contains("svg"));
        }
    }

    #[test]
    fn test_ann_benchmarks_export() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 500);
        e.save_run("test", "SIFT1M", HnswParams::default());

        let export = e.export_ann_benchmarks_format("needle", "0.1.0");
        assert!(export.contains("needle"));
        assert!(export.contains("recall"));
    }

    #[test]
    fn test_docker_config_generation() {
        let config = DockerBenchmarkConfig::default();
        let compose = config.generate_docker_compose();
        assert!(compose.contains("services:"));
        assert!(compose.contains("needle-bench"));
    }

    #[test]
    fn test_memory_tracking() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search_with_memory("q0", vec!["v0".into()], 100, 1024 * 1024);
        e.record_search_with_memory("q0", vec!["v0".into()], 200, 2 * 1024 * 1024);
        let m = e.compute_metrics();
        assert!(m.peak_memory_bytes > 0);
        assert!(m.avg_memory_bytes > 0.0);
    }

    #[test]
    fn test_pareto_frontier() {
        let runs = vec![
            BenchmarkRun {
                run_id: "r1".into(), label: "low-ef".into(), dataset: "t".into(),
                hnsw_params: HnswParams { m: 16, ef_construction: 100, ef_search: 10 },
                metrics: BenchmarkMetrics { recall_at_k: 0.80, qps: 10000.0, ..Default::default() },
                timestamp: "2024-01-01T00:00:00Z".into(),
            },
            BenchmarkRun {
                run_id: "r2".into(), label: "high-ef".into(), dataset: "t".into(),
                hnsw_params: HnswParams { m: 16, ef_construction: 100, ef_search: 200 },
                metrics: BenchmarkMetrics { recall_at_k: 0.99, qps: 1000.0, ..Default::default() },
                timestamp: "2024-01-01T00:00:00Z".into(),
            },
            BenchmarkRun {
                run_id: "r3".into(), label: "dominated".into(), dataset: "t".into(),
                hnsw_params: HnswParams { m: 16, ef_construction: 100, ef_search: 50 },
                metrics: BenchmarkMetrics { recall_at_k: 0.75, qps: 500.0, ..Default::default() },
                timestamp: "2024-01-01T00:00:00Z".into(),
            },
        ];
        let frontier = compute_pareto_frontier(&runs);
        assert_eq!(frontier.len(), 2); // r3 is dominated by r1
    }

    #[test]
    fn test_markdown_report() {
        let mut e = make_engine();
        e.add_ground_truth("q0", vec!["v0".into()]);
        e.record_search("q0", vec!["v0".into()], 100);
        e.save_run("test", "Random", HnswParams::default());
        let md = e.generate_markdown_report();
        assert!(md.contains("## Needle Benchmark Report"));
        assert!(md.contains("Recall"));
        assert!(md.contains("|"));
    }

    #[test]
    fn test_concurrent_qps() {
        let m = BenchmarkMetrics {
            total_queries: 1000,
            total_time_us: 1_000_000, // 1 second
            ..Default::default()
        };
        let cqps = estimate_concurrent_qps(&m, 4);
        // With 4 threads and 1000 QPS single-threaded, should estimate ~3200-4000
        assert!(cqps > 2000.0);
    }
}
