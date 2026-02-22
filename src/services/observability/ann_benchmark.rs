#![allow(clippy::unwrap_used)]
//! ann-benchmarks Integration
//!
//! Benchmark harness compatible with ann-benchmarks.com format. Loads standard
//! datasets (SIFT, GloVe, Fashion-MNIST), runs recall/QPS measurements, and
//! generates reports for submission.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::ann_benchmark::{
//!     BenchmarkSuite, BenchmarkConfig, DatasetConfig, BenchmarkResult,
//! };
//!
//! let mut suite = BenchmarkSuite::new(BenchmarkConfig::default());
//!
//! // Add vectors and ground truth
//! suite.add_vectors(vec![vec![1.0; 128]; 1000]);
//! suite.add_queries(vec![vec![0.5; 128]; 100], vec![vec![0usize; 10]; 100]);
//!
//! let result = suite.run().unwrap();
//! println!("Recall@10: {:.4}, QPS: {:.0}", result.recall_at_k, result.qps);
//! ```

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::indexing::hnsw::HnswConfig;

// ── Configuration ────────────────────────────────────────────────────────────

/// Dataset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Dataset name.
    pub name: String,
    /// Dimensions.
    pub dimensions: usize,
    /// Distance metric.
    pub distance: String,
    /// Number of neighbors for recall calculation.
    pub k: usize,
}

impl DatasetConfig {
    /// SIFT-1M dataset config.
    pub fn sift_1m() -> Self {
        Self { name: "sift-128-euclidean".into(), dimensions: 128, distance: "euclidean".into(), k: 10 }
    }

    /// GloVe-200 dataset config.
    pub fn glove_200() -> Self {
        Self { name: "glove-200-angular".into(), dimensions: 200, distance: "angular".into(), k: 10 }
    }

    /// Fashion-MNIST dataset config.
    pub fn fashion_mnist() -> Self {
        Self { name: "fashion-mnist-784-euclidean".into(), dimensions: 784, distance: "euclidean".into(), k: 10 }
    }
}

/// Benchmark configuration.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// HNSW M parameter values to test.
    pub m_values: Vec<usize>,
    /// ef_construction values to test.
    pub ef_construction_values: Vec<usize>,
    /// ef_search values to test (for QPS/recall tradeoff).
    pub ef_search_values: Vec<usize>,
    /// K for recall calculation.
    pub k: usize,
    /// Number of warmup queries.
    pub warmup_queries: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            m_values: vec![16],
            ef_construction_values: vec![200],
            ef_search_values: vec![10, 20, 50, 100, 200, 500],
            k: 10,
            warmup_queries: 100,
        }
    }
}

// ── Benchmark Result ─────────────────────────────────────────────────────────

/// Result from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Algorithm name.
    pub algorithm: String,
    /// Parameters used.
    pub params: BenchmarkParams,
    /// Recall@k.
    pub recall_at_k: f64,
    /// Queries per second.
    pub qps: f64,
    /// Build time in seconds.
    pub build_time_secs: f64,
    /// Index memory in bytes.
    pub index_memory_bytes: usize,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// P99 latency.
    pub p99_latency_ms: f64,
    /// Number of queries.
    pub num_queries: usize,
    /// Number of vectors.
    pub num_vectors: usize,
}

/// Parameters for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub k: usize,
}

// ── Parameter Sweep Result ───────────────────────────────────────────────────

/// Results from sweeping ef_search values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepResult {
    /// Algorithm name.
    pub algorithm: String,
    /// Per-ef_search results (recall, qps).
    pub points: Vec<(f64, f64)>,
    /// Build parameters.
    pub build_params: String,
}

// ── Benchmark Suite ──────────────────────────────────────────────────────────

/// Benchmark harness for ann-benchmarks compatibility.
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    ground_truth: Vec<Vec<usize>>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite.
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            vectors: Vec::new(),
            queries: Vec::new(),
            ground_truth: Vec::new(),
        }
    }

    /// Add dataset vectors.
    pub fn add_vectors(&mut self, vectors: Vec<Vec<f32>>) {
        self.vectors = vectors;
    }

    /// Add queries with ground truth.
    pub fn add_queries(&mut self, queries: Vec<Vec<f32>>, ground_truth: Vec<Vec<usize>>) {
        self.queries = queries;
        self.ground_truth = ground_truth;
    }

    /// Run benchmark with current configuration.
    pub fn run(&self) -> Result<BenchmarkResult> {
        if self.vectors.is_empty() || self.queries.is_empty() {
            return Err(NeedleError::InvalidInput("Vectors and queries must be provided".into()));
        }

        let dims = self.vectors[0].len();
        let m = self.config.m_values[0];
        let ef_construction = self.config.ef_construction_values[0];
        let ef_search = *self.config.ef_search_values.last().unwrap_or(&50);

        // Build index
        let build_start = Instant::now();
        let config = CollectionConfig::new("benchmark", dims)
            .with_distance(DistanceFunction::Cosine)
            .with_hnsw_config(HnswConfig::with_m(m).ef_construction(ef_construction));
        let mut collection = Collection::new(config);

        for (i, vec) in self.vectors.iter().enumerate() {
            collection.insert(format!("v{i}"), vec, None)?;
        }
        let build_time = build_start.elapsed().as_secs_f64();

        // Set ef_search
        collection.set_ef_search(ef_search);

        // Warmup
        for q in self.queries.iter().take(self.config.warmup_queries.min(self.queries.len())) {
            let _ = collection.search(q, self.config.k);
        }

        // Benchmark queries
        let mut latencies = Vec::with_capacity(self.queries.len());
        let mut total_recall = 0.0f64;

        for (qi, query) in self.queries.iter().enumerate() {
            let start = Instant::now();
            let results = collection.search(query, self.config.k)?;
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            latencies.push(latency);

            // Calculate recall
            if qi < self.ground_truth.len() {
                let gt = &self.ground_truth[qi];
                let result_ids: std::collections::HashSet<usize> = results.iter()
                    .filter_map(|r| r.id.strip_prefix('v').and_then(|s| s.parse().ok()))
                    .collect();
                let relevant: usize = gt.iter().take(self.config.k).filter(|id| result_ids.contains(id)).count();
                total_recall += relevant as f64 / gt.len().min(self.config.k) as f64;
            }
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p99_idx = (latencies.len() as f64 * 0.99) as usize;
        let p99 = latencies.get(p99_idx.min(latencies.len().saturating_sub(1))).copied().unwrap_or(0.0);
        let total_time: f64 = latencies.iter().sum();
        let qps = self.queries.len() as f64 / (total_time / 1000.0);
        let recall = total_recall / self.queries.len() as f64;

        let mem_estimate = Collection::estimate_memory(self.vectors.len(), dims, 256);

        Ok(BenchmarkResult {
            algorithm: format!("needle-hnsw(M={m},ef_c={ef_construction},ef_s={ef_search})"),
            params: BenchmarkParams { m, ef_construction, ef_search, k: self.config.k },
            recall_at_k: recall,
            qps,
            build_time_secs: build_time,
            index_memory_bytes: mem_estimate,
            avg_latency_ms: avg_latency,
            p99_latency_ms: p99,
            num_queries: self.queries.len(),
            num_vectors: self.vectors.len(),
        })
    }

    /// Run parameter sweep across ef_search values.
    pub fn sweep_ef_search(&self) -> Result<SweepResult> {
        if self.vectors.is_empty() || self.queries.is_empty() {
            return Err(NeedleError::InvalidInput("Data required".into()));
        }

        let dims = self.vectors[0].len();
        let m = self.config.m_values[0];
        let ef_c = self.config.ef_construction_values[0];

        // Build index once
        let config = CollectionConfig::new("sweep", dims)
            .with_distance(DistanceFunction::Cosine)
            .with_hnsw_config(HnswConfig::with_m(m).ef_construction(ef_c));
        let mut collection = Collection::new(config);

        for (i, vec) in self.vectors.iter().enumerate() {
            collection.insert(format!("v{i}"), vec, None)?;
        }

        let mut points = Vec::new();
        for &ef_s in &self.config.ef_search_values {
            collection.set_ef_search(ef_s);

            let mut total_time = 0.0f64;
            let mut total_recall = 0.0f64;

            for (qi, query) in self.queries.iter().enumerate() {
                let start = Instant::now();
                let results = collection.search(query, self.config.k)?;
                total_time += start.elapsed().as_secs_f64();

                if qi < self.ground_truth.len() {
                    let gt = &self.ground_truth[qi];
                    let ids: std::collections::HashSet<usize> = results.iter()
                        .filter_map(|r| r.id.strip_prefix('v').and_then(|s| s.parse().ok()))
                        .collect();
                    let relevant = gt.iter().take(self.config.k).filter(|id| ids.contains(id)).count();
                    total_recall += relevant as f64 / gt.len().min(self.config.k) as f64;
                }
            }

            let recall = total_recall / self.queries.len() as f64;
            let qps = self.queries.len() as f64 / total_time;
            points.push((recall, qps));
        }

        Ok(SweepResult {
            algorithm: "needle-hnsw".into(),
            points,
            build_params: format!("M={m}, ef_construction={ef_c}"),
        })
    }

    /// Generate ann-benchmarks compatible JSON output.
    pub fn to_ann_benchmarks_json(&self, result: &BenchmarkResult) -> String {
        serde_json::json!({
            "algorithm": result.algorithm,
            "dataset": format!("custom-{}", result.num_vectors),
            "count": result.num_vectors,
            "dims": if self.vectors.is_empty() { 0 } else { self.vectors[0].len() },
            "k": result.params.k,
            "recall": result.recall_at_k,
            "qps": result.qps,
            "build_time": result.build_time_secs,
            "params": {
                "M": result.params.m,
                "efConstruction": result.params.ef_construction,
                "efSearch": result.params.ef_search,
            }
        }).to_string()
    }
}

/// Generate a comparison report across multiple benchmark results.
pub fn comparison_report(results: &[BenchmarkResult]) -> String {
    let mut report = String::new();
    report.push_str("┌─────────────────────────────────────────────────────────────────────┐\n");
    report.push_str("│ ANN Benchmark Comparison Report                                    │\n");
    report.push_str("├──────────────────┬──────────┬──────────┬──────────┬─────────────────┤\n");
    report.push_str("│ Algorithm        │ Recall@K │ QPS      │ Latency  │ Memory          │\n");
    report.push_str("├──────────────────┼──────────┼──────────┼──────────┼─────────────────┤\n");

    for r in results {
        let mem_mb = r.index_memory_bytes as f64 / (1024.0 * 1024.0);
        report.push_str(&format!(
            "│ {:<16} │ {:<8.4} │ {:<8.0} │ {:<8.2}ms│ {:<13.1}MB │\n",
            truncate_str(&r.algorithm, 16),
            r.recall_at_k,
            r.qps,
            r.avg_latency_ms,
            mem_mb,
        ));
    }

    report.push_str("└──────────────────┴──────────┴──────────┴──────────┴─────────────────┘\n");

    // Find Pareto-optimal points (highest recall for given QPS range)
    if results.len() > 1 {
        report.push_str("\nPareto Analysis:\n");
        let mut by_recall = results.to_vec();
        by_recall.sort_by(|a, b| b.recall_at_k.partial_cmp(&a.recall_at_k).unwrap_or(std::cmp::Ordering::Equal));
        let best_recall_name = by_recall[0].algorithm.clone();
        let best_recall_val = by_recall[0].recall_at_k;
        by_recall.sort_by(|a, b| b.qps.partial_cmp(&a.qps).unwrap_or(std::cmp::Ordering::Equal));
        let best_qps_name = by_recall[0].algorithm.clone();
        let best_qps_val = by_recall[0].qps;
        report.push_str(&format!("  Best Recall: {} ({:.4})\n", best_recall_name, best_recall_val));
        report.push_str(&format!("  Best QPS:    {} ({:.0})\n", best_qps_name, best_qps_val));
    }

    report
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() <= max { s.to_string() } else { format!("{}…", &s[..max - 1]) }
}

/// Generate a Python ann-benchmarks harness definition file.
/// This produces the YAML config that ann-benchmarks uses to define an algorithm.
pub fn generate_harness_definition(algorithm_name: &str) -> String {
    format!(
        r#"# ann-benchmarks harness definition for Needle
# Place in ann_benchmarks/algorithms/{algo}/config.yml

{algo}:
  docker-tag: needle-ann-bench
  module: ann_benchmarks.algorithms.{algo}
  constructor: NeedleHNSW
  base-args: ["@metric"]
  run-groups:
    base:
      args: [[16, 32]]
      query-args: [[10, 20, 50, 100, 200, 500]]
"#,
        algo = algorithm_name
    )
}

/// Generate a Python module stub for ann-benchmarks integration.
pub fn generate_python_module(algorithm_name: &str) -> String {
    format!(
        r#""""ann-benchmarks algorithm module for Needle Vector Database."""

from ann_benchmarks.algorithms.base import BaseANN
import subprocess
import json
import tempfile
import os


class {class_name}(BaseANN):
    def __init__(self, metric, m=16):
        self.metric = metric
        self.m = m
        self._name = "needle-hnsw"
        self._db_path = None

    def fit(self, X):
        self._db_path = tempfile.mktemp(suffix=".needle")
        # Use needle CLI to build index
        data = [{{"id": f"v{{i}}", "vector": row.tolist()}} for i, row in enumerate(X)]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
            jsonl_path = f.name
        subprocess.run([
            "needle", "create", self._db_path,
        ], check=True)
        subprocess.run([
            "needle", "create-collection", self._db_path,
            "-n", "bench", "-d", str(X.shape[1]),
            "--distance", "cosine" if self.metric == "angular" else "euclidean",
        ], check=True)
        subprocess.run([
            "needle", "import", self._db_path, "-c", "bench", "-f", jsonl_path,
        ], check=True)
        os.unlink(jsonl_path)

    def query(self, v, n):
        query_str = ",".join(str(x) for x in v.tolist())
        result = subprocess.run([
            "needle", "search", self._db_path,
            "-c", "bench", "-q", query_str, "-k", str(n),
            "--output", "json",
        ], capture_output=True, text=True, check=True)
        ids = [int(r["id"].lstrip("v")) for r in json.loads(result.stdout)]
        return ids

    def __str__(self):
        return f"Needle(m={{self.m}})"
"#,
        class_name = algorithm_name
            .chars()
            .enumerate()
            .map(|(i, c)| if i == 0 {{ c.to_uppercase().next().unwrap_or(c) }} else {{ c }})
            .collect::<String>()
    )
}

/// All supported standard datasets with their configurations.
pub fn standard_datasets() -> Vec<DatasetConfig> {
    vec![
        DatasetConfig::sift_1m(),
        DatasetConfig::glove_200(),
        DatasetConfig::fashion_mnist(),
        DatasetConfig {
            name: "nytimes-256-angular".into(),
            dimensions: 256,
            distance: "angular".into(),
            k: 10,
        },
        DatasetConfig {
            name: "random-128-euclidean".into(),
            dimensions: 128,
            distance: "euclidean".into(),
            k: 10,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vectors;

    fn brute_force_neighbors(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = vectors.iter().enumerate()
            .map(|(i, v)| {
                let dot: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
                let na = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                (i, 1.0 - dot / (na * nb + f32::EPSILON))
            }).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.into_iter().take(k).map(|(i, _)| i).collect()
    }

    #[test]
    fn test_benchmark_run() {
        let vectors = random_vectors(200, 32);
        let queries = random_vectors(20, 32);
        let gt: Vec<Vec<usize>> = queries.iter()
            .map(|q| brute_force_neighbors(&vectors, q, 10))
            .collect();

        let mut suite = BenchmarkSuite::new(BenchmarkConfig {
            ef_search_values: vec![50],
            k: 10,
            warmup_queries: 5,
            ..Default::default()
        });
        suite.add_vectors(vectors);
        suite.add_queries(queries, gt);

        let result = suite.run().unwrap();
        assert!(result.recall_at_k > 0.0);
        assert!(result.qps > 0.0);
        assert!(result.build_time_secs > 0.0);
        assert_eq!(result.num_vectors, 200);
    }

    #[test]
    fn test_ef_sweep() {
        let vectors = random_vectors(100, 16);
        let queries = random_vectors(10, 16);
        let gt: Vec<Vec<usize>> = queries.iter()
            .map(|q| brute_force_neighbors(&vectors, q, 5))
            .collect();

        let mut suite = BenchmarkSuite::new(BenchmarkConfig {
            ef_search_values: vec![10, 50],
            k: 5,
            warmup_queries: 2,
            ..Default::default()
        });
        suite.add_vectors(vectors);
        suite.add_queries(queries, gt);

        let sweep = suite.sweep_ef_search().unwrap();
        assert_eq!(sweep.points.len(), 2);
        // Higher ef_search should give better recall
        assert!(sweep.points[1].0 >= sweep.points[0].0 - 0.1);
    }

    #[test]
    fn test_json_output() {
        let suite = BenchmarkSuite::new(BenchmarkConfig::default());
        let result = BenchmarkResult {
            algorithm: "test".into(),
            params: BenchmarkParams { m: 16, ef_construction: 200, ef_search: 50, k: 10 },
            recall_at_k: 0.95, qps: 1000.0, build_time_secs: 1.5,
            index_memory_bytes: 1_000_000, avg_latency_ms: 1.0, p99_latency_ms: 3.0,
            num_queries: 100, num_vectors: 10000,
        };
        let json = suite.to_ann_benchmarks_json(&result);
        assert!(json.contains("recall"));
        assert!(json.contains("qps"));
    }

    #[test]
    fn test_dataset_configs() {
        let sift = DatasetConfig::sift_1m();
        assert_eq!(sift.dimensions, 128);
        let glove = DatasetConfig::glove_200();
        assert_eq!(glove.dimensions, 200);
    }

    #[test]
    fn test_empty_data_error() {
        let suite = BenchmarkSuite::new(BenchmarkConfig::default());
        assert!(suite.run().is_err());
    }

    #[test]
    fn test_comparison_report() {
        let r1 = BenchmarkResult {
            algorithm: "needle-hnsw".into(),
            params: BenchmarkParams { m: 16, ef_construction: 200, ef_search: 50, k: 10 },
            recall_at_k: 0.95, qps: 1000.0, build_time_secs: 1.5,
            index_memory_bytes: 1_000_000, avg_latency_ms: 1.0, p99_latency_ms: 3.0,
            num_queries: 100, num_vectors: 10000,
        };
        let r2 = BenchmarkResult {
            algorithm: "brute-force".into(),
            params: BenchmarkParams { m: 0, ef_construction: 0, ef_search: 0, k: 10 },
            recall_at_k: 1.0, qps: 100.0, build_time_secs: 0.0,
            index_memory_bytes: 0, avg_latency_ms: 10.0, p99_latency_ms: 15.0,
            num_queries: 100, num_vectors: 10000,
        };

        let report = comparison_report(&[r1, r2]);
        assert!(report.contains("needle-hnsw"));
        assert!(report.contains("brute-force"));
        assert!(report.contains("Recall"));
    }

    #[test]
    fn test_harness_definition() {
        let yaml = generate_harness_definition("needle_hnsw");
        assert!(yaml.contains("needle_hnsw"));
        assert!(yaml.contains("docker-tag"));
        assert!(yaml.contains("run-groups"));
    }

    #[test]
    fn test_python_module() {
        let py = generate_python_module("needle_hnsw");
        assert!(py.contains("BaseANN"));
        assert!(py.contains("def fit"));
        assert!(py.contains("def query"));
    }

    #[test]
    fn test_standard_datasets() {
        let datasets = standard_datasets();
        assert!(datasets.len() >= 5);
        assert!(datasets.iter().any(|d| d.name.contains("sift")));
        assert!(datasets.iter().any(|d| d.name.contains("glove")));
    }
}
