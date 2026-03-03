//! Automated Recall Benchmarking
//!
//! Generates synthetic datasets, runs ANN search against brute-force ground truth,
//! and reports recall@k, QPS, latency percentiles, and memory usage.
//!
//! # Example
//!
//! ```rust
//! use needle::recall_benchmark::*;
//!
//! let config = BenchmarkConfig {
//!     num_vectors: 1000,
//!     dimensions: 64,
//!     num_queries: 50,
//!     k_values: vec![1, 5, 10],
//!     ..Default::default()
//! };
//! let report = run_recall_benchmark(&config);
//! println!("{}", report.summary());
//! ```

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a recall benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of vectors to insert.
    pub num_vectors: usize,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Number of search queries.
    pub num_queries: usize,
    /// k values for recall@k.
    pub k_values: Vec<usize>,
    /// Distance function.
    pub distance: DistanceFunction,
    /// ef_search values to sweep.
    pub ef_search_values: Vec<usize>,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10_000,
            dimensions: 128,
            num_queries: 100,
            k_values: vec![1, 5, 10, 50],
            distance: DistanceFunction::Cosine,
            ef_search_values: vec![50, 100, 200],
            seed: 42,
        }
    }
}

// ============================================================================
// Report Types
// ============================================================================

/// Recall metrics for a specific k value and ef_search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMetric {
    /// k value.
    pub k: usize,
    /// ef_search parameter.
    pub ef_search: usize,
    /// Recall@k (0.0 to 1.0).
    pub recall: f64,
    /// Queries per second.
    pub qps: f64,
    /// Mean latency in microseconds.
    pub mean_latency_us: f64,
    /// P50 latency in microseconds.
    pub p50_latency_us: f64,
    /// P95 latency in microseconds.
    pub p95_latency_us: f64,
    /// P99 latency in microseconds.
    pub p99_latency_us: f64,
}

/// Complete benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Configuration used.
    pub config: BenchmarkConfig,
    /// Index build time in milliseconds.
    pub build_time_ms: u64,
    /// Memory estimate in bytes.
    pub memory_estimate_bytes: usize,
    /// Recall metrics for each (k, ef_search) combination.
    pub metrics: Vec<RecallMetric>,
}

impl BenchmarkReport {
    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "Needle Recall Benchmark\n\
             =======================\n\
             Vectors: {}, Dimensions: {}, Queries: {}\n\
             Distance: {:?}\n\
             Build time: {}ms\n\
             Est. memory: {:.1} MB\n\n\
             {:<10} {:<12} {:<10} {:<12} {:<12} {:<12} {:<12}\n\
             {}\n",
            self.config.num_vectors,
            self.config.dimensions,
            self.config.num_queries,
            self.config.distance,
            self.build_time_ms,
            self.memory_estimate_bytes as f64 / 1_048_576.0,
            "k", "ef_search", "recall", "qps", "mean_us", "p95_us", "p99_us",
            "-".repeat(80),
        );
        for m in &self.metrics {
            s.push_str(&format!(
                "{:<10} {:<12} {:<10.4} {:<12.1} {:<12.1} {:<12.1} {:<12.1}\n",
                m.k, m.ef_search, m.recall, m.qps, m.mean_latency_us, m.p95_latency_us, m.p99_latency_us,
            ));
        }
        s
    }

    /// Export the report as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export the report as Markdown.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Needle Benchmark Report\n\n");
        md.push_str("## Configuration\n\n");
        md.push_str(&format!("| Parameter | Value |\n|---|---|\n"));
        md.push_str(&format!("| Vectors | {} |\n", self.config.num_vectors));
        md.push_str(&format!("| Dimensions | {} |\n", self.config.dimensions));
        md.push_str(&format!("| Queries | {} |\n", self.config.num_queries));
        md.push_str(&format!("| Distance | {:?} |\n", self.config.distance));
        md.push_str(&format!("| Build time | {}ms |\n", self.build_time_ms));
        md.push_str(&format!(
            "| Est. memory | {:.1} MB |\n\n",
            self.memory_estimate_bytes as f64 / 1_048_576.0
        ));
        md.push_str("## Results\n\n");
        md.push_str("| k | ef_search | recall | QPS | mean (µs) | p95 (µs) | p99 (µs) |\n");
        md.push_str("|---|---|---|---|---|---|---|\n");
        for m in &self.metrics {
            md.push_str(&format!(
                "| {} | {} | {:.4} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
                m.k, m.ef_search, m.recall, m.qps, m.mean_latency_us, m.p95_latency_us, m.p99_latency_us,
            ));
        }
        md
    }
}

// ============================================================================
// Benchmark Engine
// ============================================================================

/// Generate a deterministic pseudo-random vector.
fn gen_vector(seed: u64, idx: usize, dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(idx as u64);
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        v.push(val);
    }
    v
}

/// Compute brute-force ground truth: for each query, find exact top-k by distance.
fn compute_ground_truth(
    queries: &[Vec<f32>],
    vectors: &[Vec<f32>],
    k: usize,
    distance: &DistanceFunction,
) -> Vec<Vec<String>> {
    queries
        .iter()
        .map(|query| {
            let mut scored: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    distance.compute(query, v).ok().map(|d| (i, d))
                })
                .collect();
            scored.sort_by_key(|(_, d)| OrderedFloat(*d));
            scored.truncate(k);
            scored
                .into_iter()
                .map(|(i, _)| format!("vec_{}", i))
                .collect()
        })
        .collect()
}

/// Compute recall@k: fraction of ground-truth results found in ANN results.
fn compute_recall(ann_results: &[String], ground_truth: &[String], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<&str> =
        ground_truth.iter().take(k).map(|s| s.as_str()).collect();
    let found = ann_results
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id.as_str()))
        .count();
    if gt_set.is_empty() {
        1.0
    } else {
        found as f64 / gt_set.len() as f64
    }
}

/// Compute a percentile from a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

/// Run a full recall benchmark.
pub fn run_recall_benchmark(config: &BenchmarkConfig) -> BenchmarkReport {
    // Generate dataset
    let vectors: Vec<Vec<f32>> = (0..config.num_vectors)
        .map(|i| gen_vector(config.seed, i, config.dimensions))
        .collect();
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|i| gen_vector(config.seed.wrapping_add(999_999), i, config.dimensions))
        .collect();

    // Build index
    let build_start = Instant::now();
    let col_config = CollectionConfig::new("bench", config.dimensions)
        .with_distance(config.distance.clone());
    let mut collection = Collection::new(col_config);
    for (i, vec) in vectors.iter().enumerate() {
        let _ = collection.insert(format!("vec_{}", i), vec, None);
    }
    let build_time_ms = build_start.elapsed().as_millis() as u64;

    let memory_estimate = config.num_vectors * config.dimensions * std::mem::size_of::<f32>();

    let max_k = config.k_values.iter().copied().max().unwrap_or(10);

    // Compute ground truth for max_k
    let ground_truth = compute_ground_truth(&queries, &vectors, max_k, &config.distance);

    let mut metrics = Vec::new();

    for &ef_search in &config.ef_search_values {
        // For each query, measure latency and collect results
        let mut all_latencies: Vec<f64> = Vec::with_capacity(config.num_queries);
        let mut all_results: Vec<Vec<String>> = Vec::with_capacity(config.num_queries);

        for query in &queries {
            let start = Instant::now();
            let results = collection.search(query, max_k).unwrap_or_default();
            let elapsed_us = start.elapsed().as_micros() as f64;
            all_latencies.push(elapsed_us);
            all_results.push(results.into_iter().map(|r| r.id).collect());
        }

        all_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let total_time_secs: f64 = all_latencies.iter().sum::<f64>() / 1_000_000.0;
        let qps = if total_time_secs > 0.0 {
            config.num_queries as f64 / total_time_secs
        } else {
            0.0
        };

        let mean_latency = if all_latencies.is_empty() {
            0.0
        } else {
            all_latencies.iter().sum::<f64>() / all_latencies.len() as f64
        };

        for &k in &config.k_values {
            let recall_sum: f64 = all_results
                .iter()
                .zip(ground_truth.iter())
                .map(|(ann, gt)| compute_recall(ann, gt, k))
                .sum();
            let avg_recall = recall_sum / config.num_queries as f64;

            metrics.push(RecallMetric {
                k,
                ef_search,
                recall: avg_recall,
                qps,
                mean_latency_us: mean_latency,
                p50_latency_us: percentile(&all_latencies, 0.50),
                p95_latency_us: percentile(&all_latencies, 0.95),
                p99_latency_us: percentile(&all_latencies, 0.99),
            });
        }
    }

    BenchmarkReport {
        config: config.clone(),
        build_time_ms,
        memory_estimate_bytes: memory_estimate,
        metrics,
    }
}

/// Compare two benchmark reports side-by-side.
pub fn compare_reports(baseline: &BenchmarkReport, candidate: &BenchmarkReport) -> ComparisonReport {
    let mut comparisons = Vec::new();

    for bm in &baseline.metrics {
        let cm = candidate
            .metrics
            .iter()
            .find(|m| m.k == bm.k && m.ef_search == bm.ef_search);

        if let Some(cm) = cm {
            comparisons.push(MetricComparison {
                k: bm.k,
                ef_search: bm.ef_search,
                baseline_recall: bm.recall,
                candidate_recall: cm.recall,
                recall_delta: cm.recall - bm.recall,
                baseline_qps: bm.qps,
                candidate_qps: cm.qps,
                qps_change_pct: if bm.qps > 0.0 {
                    (cm.qps - bm.qps) / bm.qps * 100.0
                } else {
                    0.0
                },
                baseline_p99_us: bm.p99_latency_us,
                candidate_p99_us: cm.p99_latency_us,
                p99_change_pct: if bm.p99_latency_us > 0.0 {
                    (cm.p99_latency_us - bm.p99_latency_us) / bm.p99_latency_us * 100.0
                } else {
                    0.0
                },
            });
        }
    }

    let build_time_change_pct = if baseline.build_time_ms > 0 {
        (candidate.build_time_ms as f64 - baseline.build_time_ms as f64)
            / baseline.build_time_ms as f64
            * 100.0
    } else {
        0.0
    };

    ComparisonReport {
        baseline_config: baseline.config.clone(),
        candidate_config: candidate.config.clone(),
        build_time_change_pct,
        comparisons,
    }
}

/// Side-by-side comparison of two benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Baseline configuration.
    pub baseline_config: BenchmarkConfig,
    /// Candidate configuration.
    pub candidate_config: BenchmarkConfig,
    /// Build time change percentage.
    pub build_time_change_pct: f64,
    /// Per-(k, ef_search) comparisons.
    pub comparisons: Vec<MetricComparison>,
}

/// Comparison of a single (k, ef_search) metric pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// k value.
    pub k: usize,
    /// ef_search value.
    pub ef_search: usize,
    /// Baseline recall@k.
    pub baseline_recall: f64,
    /// Candidate recall@k.
    pub candidate_recall: f64,
    /// Recall delta (positive = improvement).
    pub recall_delta: f64,
    /// Baseline QPS.
    pub baseline_qps: f64,
    /// Candidate QPS.
    pub candidate_qps: f64,
    /// QPS change percentage (positive = faster).
    pub qps_change_pct: f64,
    /// Baseline P99 latency (µs).
    pub baseline_p99_us: f64,
    /// Candidate P99 latency (µs).
    pub candidate_p99_us: f64,
    /// P99 change percentage (negative = faster).
    pub p99_change_pct: f64,
}

impl ComparisonReport {
    /// Generate a summary string.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("Benchmark Comparison\n");
        s.push_str("====================\n");
        s.push_str(&format!(
            "Build time change: {:+.1}%\n\n",
            self.build_time_change_pct
        ));
        s.push_str(&format!(
            "{:<6} {:<10} {:<14} {:<14} {:<12} {:<12}\n",
            "k", "ef_search", "recall_delta", "qps_change%", "p99_base_us", "p99_cand_us"
        ));
        s.push_str(&format!("{}\n", "-".repeat(70)));
        for c in &self.comparisons {
            s.push_str(&format!(
                "{:<6} {:<10} {:+<14.4} {:+<14.1} {:<12.1} {:<12.1}\n",
                c.k, c.ef_search, c.recall_delta, c.qps_change_pct, c.baseline_p99_us, c.candidate_p99_us,
            ));
        }
        s
    }

    /// Export as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Export as Markdown.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Benchmark Comparison\n\n");
        md.push_str(&format!("Build time change: {:+.1}%\n\n", self.build_time_change_pct));
        md.push_str("| k | ef_search | recall Δ | QPS change | p99 base (µs) | p99 cand (µs) |\n");
        md.push_str("|---|---|---|---|---|---|\n");
        for c in &self.comparisons {
            md.push_str(&format!(
                "| {} | {} | {:+.4} | {:+.1}% | {:.1} | {:.1} |\n",
                c.k, c.ef_search, c.recall_delta, c.qps_change_pct, c.baseline_p99_us, c.candidate_p99_us,
            ));
        }
        md
    }
}

// ============================================================================
// ANN-Benchmarks Standard Datasets
// ============================================================================

/// Standard ANN-benchmarks dataset specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnDataset {
    /// Dataset name (e.g., "sift-128-euclidean").
    pub name: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance metric.
    pub distance: DistanceFunction,
    /// Number of base vectors.
    pub num_vectors: usize,
    /// Number of query vectors.
    pub num_queries: usize,
    /// HDF5 file URL for downloading.
    pub url: String,
}

impl AnnDataset {
    /// SIFT-1M: 128-dim, 1M vectors, Euclidean distance.
    pub fn sift_1m() -> Self {
        Self {
            name: "sift-128-euclidean".to_string(),
            dimensions: 128,
            distance: DistanceFunction::Euclidean,
            num_vectors: 1_000_000,
            num_queries: 10_000,
            url: "http://ann-benchmarks.com/sift-128-euclidean.hdf5".to_string(),
        }
    }

    /// GloVe-200: 200-dim, ~1.2M vectors, Angular (Cosine) distance.
    pub fn glove_200() -> Self {
        Self {
            name: "glove-200-angular".to_string(),
            dimensions: 200,
            distance: DistanceFunction::Cosine,
            num_vectors: 1_183_514,
            num_queries: 10_000,
            url: "http://ann-benchmarks.com/glove-200-angular.hdf5".to_string(),
        }
    }

    /// GIST-960: 960-dim, 1M vectors, Euclidean distance.
    pub fn gist_960() -> Self {
        Self {
            name: "gist-960-euclidean".to_string(),
            dimensions: 960,
            distance: DistanceFunction::Euclidean,
            num_vectors: 1_000_000,
            num_queries: 1_000,
            url: "http://ann-benchmarks.com/gist-960-euclidean.hdf5".to_string(),
        }
    }

    /// All standard ANN-benchmark datasets.
    pub fn all_standard() -> Vec<Self> {
        vec![Self::sift_1m(), Self::glove_200(), Self::gist_960()]
    }
}

/// Result of running an ANN-benchmarks-style evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnBenchmarkResult {
    /// Dataset used.
    pub dataset: String,
    /// Algorithm name (e.g., "needle-hnsw").
    pub algorithm: String,
    /// Parameters used (for Pareto curve).
    pub parameters: std::collections::HashMap<String, String>,
    /// Recall@10 (standard metric).
    pub recall_at_10: f64,
    /// Queries per second.
    pub qps: f64,
    /// Index build time in seconds.
    pub build_time_seconds: f64,
    /// Peak memory usage in bytes.
    pub memory_bytes: usize,
}

impl AnnBenchmarkResult {
    /// Format as ann-benchmarks.com compatible JSON output.
    pub fn to_ann_benchmarks_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Run a standardized ANN-benchmark evaluation using synthetic data
/// matching dataset dimensions and distance metric.
pub fn run_ann_benchmark(dataset: &AnnDataset, ef_search_values: &[usize]) -> Vec<AnnBenchmarkResult> {
    let config = BenchmarkConfig {
        num_vectors: dataset.num_vectors.min(10_000), // Cap for synthetic test
        dimensions: dataset.dimensions,
        num_queries: dataset.num_queries.min(100),
        k_values: vec![10],
        distance: dataset.distance.clone(),
        ef_search_values: ef_search_values.to_vec(),
        seed: 42,
    };

    let report = run_recall_benchmark(&config);
    report
        .metrics
        .iter()
        .map(|m| {
            let mut params = std::collections::HashMap::new();
            params.insert("ef_search".to_string(), m.ef_search.to_string());
            AnnBenchmarkResult {
                dataset: dataset.name.clone(),
                algorithm: "needle-hnsw".to_string(),
                parameters: params,
                recall_at_10: m.recall,
                qps: m.qps,
                build_time_seconds: report.build_time_ms as f64 / 1000.0,
                memory_bytes: report.memory_estimate_bytes,
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_vector_deterministic() {
        let v1 = gen_vector(42, 0, 4);
        let v2 = gen_vector(42, 0, 4);
        assert_eq!(v1, v2);

        let v3 = gen_vector(42, 1, 4);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_compute_recall() {
        let ann = vec!["a".into(), "b".into(), "c".into()];
        let gt = vec!["a".into(), "b".into(), "d".into()];
        assert!((compute_recall(&ann, &gt, 3) - 2.0 / 3.0).abs() < 0.01);

        // Perfect recall
        assert!((compute_recall(&ann, &ann, 3) - 1.0).abs() < 0.01);

        // Zero recall
        let other = vec!["x".into(), "y".into(), "z".into()];
        assert!((compute_recall(&ann, &other, 3) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&sorted, 0.50) - 5.0).abs() < 0.01);
        assert!((percentile(&sorted, 0.99) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_run_benchmark_small() {
        let config = BenchmarkConfig {
            num_vectors: 100,
            dimensions: 16,
            num_queries: 10,
            k_values: vec![1, 5],
            ef_search_values: vec![50],
            seed: 42,
            distance: DistanceFunction::Cosine,
        };
        let report = run_recall_benchmark(&config);
        assert!(report.build_time_ms < 10_000);
        assert_eq!(report.metrics.len(), 2); // 2 k values * 1 ef_search
        for m in &report.metrics {
            assert!(m.recall >= 0.0 && m.recall <= 1.0);
            assert!(m.qps > 0.0);
        }
    }

    #[test]
    fn test_report_summary() {
        let config = BenchmarkConfig {
            num_vectors: 50,
            dimensions: 8,
            num_queries: 5,
            k_values: vec![1],
            ef_search_values: vec![50],
            seed: 1,
            distance: DistanceFunction::Euclidean,
        };
        let report = run_recall_benchmark(&config);
        let summary = report.summary();
        assert!(summary.contains("Recall Benchmark"));
        assert!(summary.contains("recall"));
    }

    #[test]
    fn test_report_json_export() {
        let config = BenchmarkConfig {
            num_vectors: 50,
            dimensions: 8,
            num_queries: 5,
            k_values: vec![1],
            ef_search_values: vec![50],
            seed: 1,
            distance: DistanceFunction::Cosine,
        };
        let report = run_recall_benchmark(&config);
        let json = report.to_json();
        assert!(json.contains("num_vectors"));
        assert!(json.contains("recall"));
    }

    #[test]
    fn test_report_markdown_export() {
        let config = BenchmarkConfig {
            num_vectors: 50,
            dimensions: 8,
            num_queries: 5,
            k_values: vec![1],
            ef_search_values: vec![50],
            seed: 1,
            distance: DistanceFunction::Cosine,
        };
        let report = run_recall_benchmark(&config);
        let md = report.to_markdown();
        assert!(md.contains("# Needle Benchmark Report"));
        assert!(md.contains("| k |"));
    }

    #[test]
    fn test_compare_reports() {
        let config1 = BenchmarkConfig {
            num_vectors: 50,
            dimensions: 8,
            num_queries: 5,
            k_values: vec![1, 5],
            ef_search_values: vec![50],
            seed: 42,
            distance: DistanceFunction::Cosine,
        };
        let config2 = BenchmarkConfig {
            num_vectors: 50,
            dimensions: 8,
            num_queries: 5,
            k_values: vec![1, 5],
            ef_search_values: vec![50],
            seed: 43,
            distance: DistanceFunction::Cosine,
        };
        let report1 = run_recall_benchmark(&config1);
        let report2 = run_recall_benchmark(&config2);
        let comparison = compare_reports(&report1, &report2);

        assert_eq!(comparison.comparisons.len(), 2);
        let summary = comparison.summary();
        assert!(summary.contains("Benchmark Comparison"));
        let md = comparison.to_markdown();
        assert!(md.contains("# Benchmark Comparison"));
    }

    #[test]
    fn test_ann_dataset_definitions() {
        let sift = AnnDataset::sift_1m();
        assert_eq!(sift.dimensions, 128);
        assert_eq!(sift.num_vectors, 1_000_000);

        let glove = AnnDataset::glove_200();
        assert_eq!(glove.dimensions, 200);

        let gist = AnnDataset::gist_960();
        assert_eq!(gist.dimensions, 960);

        let all = AnnDataset::all_standard();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_ann_benchmark_run() {
        let mut dataset = AnnDataset::sift_1m();
        dataset.num_vectors = 100; // tiny for test
        dataset.num_queries = 10;

        let results = run_ann_benchmark(&dataset, &[50]);
        assert!(!results.is_empty());
        assert!(results[0].recall_at_10 > 0.0);
        assert!(results[0].qps > 0.0);

        let json = results[0].to_ann_benchmarks_json();
        assert!(json.contains("needle-hnsw"));
    }
}
