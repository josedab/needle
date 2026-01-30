//! ann-benchmarks Docker Runner
//!
//! Docker-based benchmark configuration for ann-benchmarks.com submission.
//! Generates Dockerfile, runner script, and benchmark configuration.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::benchmark_runner::{BenchRunner, BenchDataset, RunnerConfig};
//!
//! let runner = BenchRunner::new(RunnerConfig::default());
//! let dockerfile = runner.generate_dockerfile();
//! let config = runner.generate_config(BenchDataset::Sift1M);
//! ```

use serde::{Deserialize, Serialize};

/// Benchmark dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchDataset { Sift1M, Glove200, FashionMnist, Nytimes, Random1M }

impl BenchDataset {
    pub fn name(&self) -> &str {
        match self { Self::Sift1M => "sift-128-euclidean", Self::Glove200 => "glove-200-angular",
            Self::FashionMnist => "fashion-mnist-784-euclidean", Self::Nytimes => "nytimes-256-angular",
            Self::Random1M => "random-xs-20-euclidean" }
    }
    pub fn dimensions(&self) -> usize {
        match self { Self::Sift1M => 128, Self::Glove200 => 200, Self::FashionMnist => 784, Self::Nytimes => 256, Self::Random1M => 20 }
    }
    pub fn metric(&self) -> &str {
        match self { Self::Sift1M | Self::FashionMnist | Self::Random1M => "euclidean", Self::Glove200 | Self::Nytimes => "angular" }
    }
}

/// Runner configuration.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub rust_version: String,
    pub target: String,
    pub features: Vec<String>,
    pub m_values: Vec<usize>,
    pub ef_construction_values: Vec<usize>,
    pub ef_search_values: Vec<usize>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            rust_version: "1.85".into(), target: "x86_64-unknown-linux-gnu".into(),
            features: vec!["simd".into()],
            m_values: vec![12, 16, 24], ef_construction_values: vec![100, 200, 500],
            ef_search_values: vec![10, 20, 50, 100, 200, 400, 800],
        }
    }
}

/// Benchmark runner.
pub struct BenchRunner { config: RunnerConfig }

impl BenchRunner {
    pub fn new(config: RunnerConfig) -> Self { Self { config } }

    /// Generate Dockerfile for ann-benchmarks.
    pub fn generate_dockerfile(&self) -> String {
        format!(r#"FROM rust:{rust} as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features "{features}"

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/needle /usr/local/bin/
COPY --from=builder /app/scripts/ann-benchmark.sh /usr/local/bin/
ENTRYPOINT ["needle"]
"#, rust = self.config.rust_version, features = self.config.features.join(","))
    }

    /// Generate ann-benchmarks algorithm config YAML.
    pub fn generate_config(&self, dataset: BenchDataset) -> String {
        let mut configs = Vec::new();
        for &m in &self.config.m_values {
            for &ef_c in &self.config.ef_construction_values {
                let ef_searches: Vec<String> = self.config.ef_search_values.iter().map(|e| e.to_string()).collect();
                configs.push(format!(
                    "  needle-hnsw-M{m}-efc{ef_c}:\n    docker-tag: needle\n    module: needle\n    constructor: NeedleHNSW\n    base-args: [\"@metric\", {m}, {ef_c}]\n    run-groups:\n      {dataset}:\n        args: [[{ef_searches}]]",
                    m = m, ef_c = ef_c, dataset = dataset.name(), ef_searches = ef_searches.join(", ")
                ));
            }
        }
        format!("float:\n  any:\n{}", configs.join("\n"))
    }

    /// Generate Python wrapper for ann-benchmarks.
    pub fn generate_python_wrapper(&self) -> String {
        r#""""Needle wrapper for ann-benchmarks."""
import subprocess, json, numpy as np
from ann_benchmarks.algorithms.base import BaseANN

class NeedleHNSW(BaseANN):
    def __init__(self, metric, m, ef_construction):
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = 50

    def fit(self, X):
        # Index via Needle CLI
        self.n, self.d = X.shape
        subprocess.run(["needle", "create", "/tmp/bench.needle"], check=True)
        subprocess.run(["needle", "create-collection", "/tmp/bench.needle",
                        "-n", "bench", "-d", str(self.d)], check=True)

    def set_query_arguments(self, ef_search):
        self.ef_search = ef_search

    def query(self, v, n):
        # Returns indices of n nearest neighbors
        return list(range(n))  # Placeholder

    def __str__(self):
        return f"NeedleHNSW(M={self.m}, efc={self.ef_construction}, efs={self.ef_search})"
"#.into()
    }

    /// List all supported datasets.
    pub fn datasets() -> Vec<BenchDataset> {
        vec![BenchDataset::Sift1M, BenchDataset::Glove200, BenchDataset::FashionMnist, BenchDataset::Nytimes, BenchDataset::Random1M]
    }

    /// Estimate benchmark runtime for a dataset.
    pub fn estimated_runtime_minutes(&self, dataset: BenchDataset) -> f32 {
        let configs = self.config.m_values.len() * self.config.ef_construction_values.len();
        let sweeps = self.config.ef_search_values.len();
        let base = match dataset {
            BenchDataset::Sift1M => 5.0, BenchDataset::Glove200 => 8.0,
            BenchDataset::FashionMnist => 15.0, _ => 3.0,
        };
        base * configs as f32 * sweeps as f32 / 10.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dockerfile() {
        let runner = BenchRunner::new(RunnerConfig::default());
        let df = runner.generate_dockerfile();
        assert!(df.contains("rust:1.85"));
        assert!(df.contains("simd"));
    }

    #[test]
    fn test_config() {
        let runner = BenchRunner::new(RunnerConfig::default());
        let cfg = runner.generate_config(BenchDataset::Sift1M);
        assert!(cfg.contains("sift-128-euclidean"));
        assert!(cfg.contains("needle-hnsw"));
    }

    #[test]
    fn test_python_wrapper() {
        let runner = BenchRunner::new(RunnerConfig::default());
        let py = runner.generate_python_wrapper();
        assert!(py.contains("NeedleHNSW"));
        assert!(py.contains("BaseANN"));
    }

    #[test]
    fn test_datasets() {
        assert_eq!(BenchRunner::datasets().len(), 5);
        assert_eq!(BenchDataset::Sift1M.dimensions(), 128);
    }

    #[test]
    fn test_runtime_estimate() {
        let runner = BenchRunner::new(RunnerConfig::default());
        let mins = runner.estimated_runtime_minutes(BenchDataset::Sift1M);
        assert!(mins > 0.0);
    }
}
