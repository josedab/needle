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
}
