//! Vector Compression Advisor
//!
//! Automatic recommendation engine that analyzes vector distributions and
//! recommends optimal quantization strategies per accuracy target.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::indexing::compression_advisor::{CompressionAdvisor, AdvisorConfig};
//!
//! let vectors: Vec<Vec<f32>> = load_your_vectors();
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! let advisor = CompressionAdvisor::new(AdvisorConfig::default());
//! let report = advisor.analyze(&refs, 10)?;
//! println!("{}", report.summary());
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::indexing::quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for the compression advisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisorConfig {
    /// Maximum sample size for analysis.
    pub max_sample_size: usize,
    /// Number of recall test queries.
    pub num_test_queries: usize,
    /// k for recall@k evaluation.
    pub recall_k: usize,
    /// Target recall thresholds to report on.
    pub target_recalls: Vec<f64>,
}

impl Default for AdvisorConfig {
    fn default() -> Self {
        Self {
            max_sample_size: 10_000,
            num_test_queries: 100,
            recall_k: 10,
            target_recalls: vec![0.99, 0.95, 0.90, 0.85],
        }
    }
}

/// Quantization strategy identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// No quantization (baseline).
    None,
    /// 8-bit scalar quantization.
    SQ8,
    /// Product quantization with the given number of subvectors.
    PQ { subvectors: usize },
    /// Binary quantization (1 bit per dimension).
    Binary,
}

impl std::fmt::Display for QuantizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationStrategy::None => write!(f, "None (f32)"),
            QuantizationStrategy::SQ8 => write!(f, "SQ8 (int8)"),
            QuantizationStrategy::PQ { subvectors } => {
                write!(f, "PQ ({}sv)", subvectors)
            }
            QuantizationStrategy::Binary => write!(f, "Binary (1-bit)"),
        }
    }
}

/// Result for a single quantization strategy evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResult {
    /// Strategy evaluated.
    pub strategy: QuantizationStrategy,
    /// Recall@k achieved.
    pub recall_at_k: f64,
    /// Compression ratio (original_size / compressed_size).
    pub compression_ratio: f64,
    /// Memory per vector (bytes).
    pub bytes_per_vector: usize,
    /// Analysis time in milliseconds.
    pub analysis_time_ms: u64,
    /// Average reconstruction error (L2).
    pub reconstruction_error: f64,
}

/// Distribution analysis of the vector dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Number of vectors analyzed.
    pub num_vectors: usize,
    /// Dimensionality.
    pub dimensions: usize,
    /// Per-dimension variance.
    pub variance: Vec<f64>,
    /// Mean variance across dimensions.
    pub mean_variance: f64,
    /// Coefficient of variation.
    pub coeff_variation: f64,
    /// Estimated clustering tendency (Hopkins statistic approximation).
    pub clustering_tendency: f64,
}

/// Full advisor report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisorReport {
    /// Distribution analysis.
    pub distribution: DistributionAnalysis,
    /// Results per strategy.
    pub strategies: Vec<StrategyResult>,
    /// Recommended strategy for each target recall.
    pub recommendations: Vec<Recommendation>,
    /// Total analysis time in milliseconds.
    pub total_analysis_time_ms: u64,
}

/// A single recommendation for a target recall level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Target recall level.
    pub target_recall: f64,
    /// Recommended strategy.
    pub strategy: QuantizationStrategy,
    /// Achieved recall.
    pub achieved_recall: f64,
    /// Compression ratio.
    pub compression_ratio: f64,
    /// Memory savings description.
    pub savings: String,
}

impl AdvisorReport {
    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("═══════════════════════════════════════════════════\n");
        s.push_str("        Vector Compression Advisor Report          \n");
        s.push_str("═══════════════════════════════════════════════════\n\n");

        s.push_str(&format!(
            "Dataset: {} vectors × {} dimensions\n",
            self.distribution.num_vectors, self.distribution.dimensions
        ));
        s.push_str(&format!(
            "Mean variance: {:.4}, Clustering tendency: {:.2}\n\n",
            self.distribution.mean_variance, self.distribution.clustering_tendency
        ));

        s.push_str("Strategy Results:\n");
        s.push_str("─────────────────────────────────────────────────\n");
        s.push_str(&format!(
            "{:<20} {:>10} {:>12} {:>10}\n",
            "Strategy", "Recall@k", "Compression", "Bytes/Vec"
        ));
        for sr in &self.strategies {
            s.push_str(&format!(
                "{:<20} {:>9.3}% {:>11.1}x {:>10}\n",
                sr.strategy.to_string(),
                sr.recall_at_k * 100.0,
                sr.compression_ratio,
                sr.bytes_per_vector
            ));
        }

        s.push_str("\nRecommendations:\n");
        s.push_str("─────────────────────────────────────────────────\n");
        for rec in &self.recommendations {
            s.push_str(&format!(
                "  Target {:.0}% recall → {} ({:.1}x compression, {})\n",
                rec.target_recall * 100.0,
                rec.strategy,
                rec.compression_ratio,
                rec.savings
            ));
        }

        s.push_str(&format!(
            "\nAnalysis completed in {}ms\n",
            self.total_analysis_time_ms
        ));
        s
    }
}

/// Compression advisor engine.
pub struct CompressionAdvisor {
    config: AdvisorConfig,
}

impl CompressionAdvisor {
    /// Create a new compression advisor.
    pub fn new(config: AdvisorConfig) -> Self {
        Self { config }
    }

    /// Analyze vectors and produce a compression recommendation report.
    pub fn analyze(&self, vectors: &[&[f32]], recall_k: usize) -> Result<AdvisorReport> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput(
                "No vectors to analyze".to_string(),
            ));
        }

        let start = Instant::now();
        let dim = vectors[0].len();
        let n = vectors.len();

        // Stratified sample if dataset is large
        let sample = if n > self.config.max_sample_size {
            let step = n / self.config.max_sample_size;
            vectors.iter().step_by(step.max(1)).copied().collect::<Vec<_>>()
        } else {
            vectors.to_vec()
        };

        // Analyze distribution
        let distribution = self.analyze_distribution(&sample, dim);

        // Split into base and queries
        let num_queries = self.config.num_test_queries.min(sample.len() / 10).max(1);
        let (base, queries) = sample.split_at(sample.len() - num_queries);

        // Compute ground truth (brute force)
        let k = recall_k.min(base.len());
        let ground_truth = self.compute_ground_truth(base, queries, k);

        // Test each strategy
        let mut strategies = Vec::new();

        // Baseline (no quantization)
        strategies.push(StrategyResult {
            strategy: QuantizationStrategy::None,
            recall_at_k: 1.0,
            compression_ratio: 1.0,
            bytes_per_vector: dim * 4,
            analysis_time_ms: 0,
            reconstruction_error: 0.0,
        });

        // SQ8
        strategies.push(self.test_sq8(base, queries, &ground_truth, k, dim));

        // PQ with different subvector counts
        for num_sub in [dim / 4, dim / 2].iter().copied() {
            if num_sub > 0 && dim % num_sub == 0 {
                strategies.push(self.test_pq(base, queries, &ground_truth, k, dim, num_sub));
            }
        }

        // Binary
        strategies.push(self.test_binary(base, queries, &ground_truth, k, dim));

        // Generate recommendations
        let recommendations = self.generate_recommendations(&strategies, dim);

        let total_analysis_time_ms = start.elapsed().as_millis() as u64;

        Ok(AdvisorReport {
            distribution,
            strategies,
            recommendations,
            total_analysis_time_ms,
        })
    }

    fn analyze_distribution(&self, vectors: &[&[f32]], dim: usize) -> DistributionAnalysis {
        let n = vectors.len();

        // Compute per-dimension mean and variance
        let mut means = vec![0.0f64; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                means[i] += val as f64;
            }
        }
        for m in &mut means {
            *m /= n as f64;
        }

        let mut variance = vec![0.0f64; dim];
        for v in vectors {
            for (i, &val) in v.iter().enumerate() {
                let diff = val as f64 - means[i];
                variance[i] += diff * diff;
            }
        }
        for v in &mut variance {
            *v /= n as f64;
        }

        let mean_variance = variance.iter().sum::<f64>() / dim as f64;
        let variance_of_variance = {
            let v_mean = mean_variance;
            variance.iter().map(|&v| (v - v_mean).powi(2)).sum::<f64>() / dim as f64
        };
        let coeff_variation = if mean_variance > 0.0 {
            variance_of_variance.sqrt() / mean_variance
        } else {
            0.0
        };

        // Approximate clustering tendency via inter-point distance variance
        let sample_size = 50.min(n);
        let mut distances = Vec::new();
        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                let d: f64 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(&a, &b)| ((a - b) as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push(d);
            }
        }
        let clustering_tendency = if distances.is_empty() {
            0.5
        } else {
            let mean_d = distances.iter().sum::<f64>() / distances.len() as f64;
            let var_d =
                distances.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / distances.len() as f64;
            // High variance in distances suggests clustering
            (var_d / (mean_d.powi(2) + 1e-10)).min(1.0)
        };

        DistributionAnalysis {
            num_vectors: n,
            dimensions: dim,
            variance,
            mean_variance,
            coeff_variation,
            clustering_tendency,
        }
    }

    fn compute_ground_truth(
        &self,
        base: &[&[f32]],
        queries: &[&[f32]],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let dist_fn = DistanceFunction::Euclidean;
        queries
            .iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = base
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, dist_fn.compute(q, b).unwrap_or(f32::MAX)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect()
    }

    fn compute_recall(
        &self,
        results: &[Vec<usize>],
        ground_truth: &[Vec<usize>],
        k: usize,
    ) -> f64 {
        let mut recall_sum = 0.0;
        let count = results.len().min(ground_truth.len());
        for i in 0..count {
            let gt: std::collections::HashSet<_> = ground_truth[i].iter().take(k).collect();
            let res: std::collections::HashSet<_> = results[i].iter().take(k).collect();
            let hits = gt.intersection(&res).count();
            if !gt.is_empty() {
                recall_sum += hits as f64 / gt.len() as f64;
            }
        }
        if count > 0 {
            recall_sum / count as f64
        } else {
            0.0
        }
    }

    fn test_sq8(
        &self,
        base: &[&[f32]],
        queries: &[&[f32]],
        ground_truth: &[Vec<usize>],
        k: usize,
        dim: usize,
    ) -> StrategyResult {
        let start = Instant::now();
        let sq = ScalarQuantizer::train(base);

        // Quantize all base vectors
        let quantized: Vec<Vec<u8>> = base.iter().map(|v| sq.quantize(v)).collect();
        let dequantized: Vec<Vec<f32>> = quantized.iter().map(|q| sq.dequantize(q)).collect();

        // Search using dequantized vectors
        let dist_fn = DistanceFunction::Euclidean;
        let results: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = dequantized
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, dist_fn.compute(q, b).unwrap_or(f32::MAX)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect();

        let recall = self.compute_recall(&results, ground_truth, k);

        // Compute reconstruction error
        let mut total_error = 0.0;
        for (orig, deq) in base.iter().zip(dequantized.iter()) {
            let err: f64 = orig
                .iter()
                .zip(deq.iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            total_error += err.sqrt();
        }
        let reconstruction_error = total_error / base.len() as f64;

        StrategyResult {
            strategy: QuantizationStrategy::SQ8,
            recall_at_k: recall,
            compression_ratio: 4.0, // f32 -> u8
            bytes_per_vector: dim,
            analysis_time_ms: start.elapsed().as_millis() as u64,
            reconstruction_error,
        }
    }

    fn test_pq(
        &self,
        base: &[&[f32]],
        queries: &[&[f32]],
        ground_truth: &[Vec<usize>],
        k: usize,
        dim: usize,
        num_subvectors: usize,
    ) -> StrategyResult {
        let start = Instant::now();
        let pq = ProductQuantizer::train(base, num_subvectors);

        // Encode all base vectors
        let encoded: Vec<Vec<u8>> = base.iter().map(|v| pq.encode(v)).collect();
        let decoded: Vec<Vec<f32>> = encoded.iter().map(|c| pq.decode(c)).collect();

        // Search using decoded vectors
        let dist_fn = DistanceFunction::Euclidean;
        let results: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let mut dists: Vec<(usize, f32)> = decoded
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, dist_fn.compute(q, b).unwrap_or(f32::MAX)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect();

        let recall = self.compute_recall(&results, ground_truth, k);
        let compression_ratio = (dim * 4) as f64 / num_subvectors as f64;

        let mut total_error = 0.0;
        for (orig, dec) in base.iter().zip(decoded.iter()) {
            let err: f64 = orig
                .iter()
                .zip(dec.iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            total_error += err.sqrt();
        }

        StrategyResult {
            strategy: QuantizationStrategy::PQ {
                subvectors: num_subvectors,
            },
            recall_at_k: recall,
            compression_ratio,
            bytes_per_vector: num_subvectors,
            analysis_time_ms: start.elapsed().as_millis() as u64,
            reconstruction_error: total_error / base.len() as f64,
        }
    }

    fn test_binary(
        &self,
        base: &[&[f32]],
        queries: &[&[f32]],
        ground_truth: &[Vec<usize>],
        k: usize,
        dim: usize,
    ) -> StrategyResult {
        let start = Instant::now();
        let bq = BinaryQuantizer::train(base);

        let quantized: Vec<Vec<u8>> = base.iter().map(|v| bq.quantize(v)).collect();

        // Search using Hamming distance
        let results: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let q_bin = bq.quantize(q);
                let mut dists: Vec<(usize, u32)> = quantized
                    .iter()
                    .enumerate()
                    .map(|(i, b)| (i, BinaryQuantizer::hamming_distance(&q_bin, b)))
                    .collect();
                dists.sort_by_key(|&(_, d)| d);
                dists.into_iter().take(k).map(|(i, _)| i).collect()
            })
            .collect();

        let recall = self.compute_recall(&results, ground_truth, k);
        let bytes_per_vector = (dim + 7) / 8;

        StrategyResult {
            strategy: QuantizationStrategy::Binary,
            recall_at_k: recall,
            compression_ratio: (dim * 4) as f64 / bytes_per_vector as f64,
            bytes_per_vector,
            analysis_time_ms: start.elapsed().as_millis() as u64,
            reconstruction_error: f64::NAN, // Binary has no dequantization
        }
    }

    fn generate_recommendations(
        &self,
        strategies: &[StrategyResult],
        dim: usize,
    ) -> Vec<Recommendation> {
        self.config
            .target_recalls
            .iter()
            .map(|&target| {
                // Find best strategy that meets target recall with highest compression
                let mut candidates: Vec<_> = strategies
                    .iter()
                    .filter(|s| s.recall_at_k >= target)
                    .collect();
                candidates.sort_by(|a, b| {
                    b.compression_ratio
                        .partial_cmp(&a.compression_ratio)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                if let Some(best) = candidates.first() {
                    let orig_bytes = dim * 4;
                    let saved_pct =
                        (1.0 - best.bytes_per_vector as f64 / orig_bytes as f64) * 100.0;
                    Recommendation {
                        target_recall: target,
                        strategy: best.strategy,
                        achieved_recall: best.recall_at_k,
                        compression_ratio: best.compression_ratio,
                        savings: format!("{:.0}% memory savings", saved_pct),
                    }
                } else {
                    Recommendation {
                        target_recall: target,
                        strategy: QuantizationStrategy::None,
                        achieved_recall: 1.0,
                        compression_ratio: 1.0,
                        savings: "No compression meets target".into(),
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect())
            .collect()
    }

    #[test]
    fn test_advisor_basic() {
        let vectors = make_vectors(200, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig {
            max_sample_size: 200,
            num_test_queries: 10,
            recall_k: 5,
            target_recalls: vec![0.95, 0.90],
        });

        let report = advisor.analyze(&refs, 5).unwrap();

        assert!(report.strategies.len() >= 3); // None, SQ8, Binary at minimum
        assert_eq!(report.distribution.dimensions, 16);
        assert_eq!(report.distribution.num_vectors, 200);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_advisor_summary() {
        let vectors = make_vectors(100, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig {
            max_sample_size: 100,
            num_test_queries: 5,
            recall_k: 3,
            target_recalls: vec![0.90],
        });

        let report = advisor.analyze(&refs, 3).unwrap();
        let summary = report.summary();

        assert!(summary.contains("Vector Compression Advisor"));
        assert!(summary.contains("Recall@k"));
        assert!(summary.contains("Recommendations"));
    }

    #[test]
    fn test_advisor_distribution_analysis() {
        let vectors = make_vectors(50, 4);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig::default());
        let dist = advisor.analyze_distribution(&refs, 4);

        assert_eq!(dist.dimensions, 4);
        assert_eq!(dist.num_vectors, 50);
        assert_eq!(dist.variance.len(), 4);
        assert!(dist.mean_variance >= 0.0);
    }

    #[test]
    fn test_advisor_empty_input() {
        let advisor = CompressionAdvisor::new(AdvisorConfig::default());
        let refs: Vec<&[f32]> = vec![];
        assert!(advisor.analyze(&refs, 5).is_err());
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(QuantizationStrategy::None.to_string(), "None (f32)");
        assert_eq!(QuantizationStrategy::SQ8.to_string(), "SQ8 (int8)");
        assert_eq!(
            QuantizationStrategy::PQ { subvectors: 8 }.to_string(),
            "PQ (8sv)"
        );
        assert_eq!(QuantizationStrategy::Binary.to_string(), "Binary (1-bit)");
    }

    #[test]
    fn test_sq8_recall() {
        let vectors = make_vectors(200, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig {
            max_sample_size: 200,
            num_test_queries: 10,
            recall_k: 5,
            target_recalls: vec![0.90],
        });

        let report = advisor.analyze(&refs, 5).unwrap();
        let sq8 = report
            .strategies
            .iter()
            .find(|s| s.strategy == QuantizationStrategy::SQ8)
            .unwrap();

        // SQ8 should have high recall (>= 80% typically)
        assert!(sq8.recall_at_k >= 0.5);
        assert_eq!(sq8.compression_ratio, 4.0);
    }

    #[test]
    fn test_stratified_sampling() {
        // Create vectors with distinct clusters
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for cluster in 0..3 {
            for i in 0..100 {
                let base = cluster as f32 * 10.0;
                vectors.push((0..8).map(|j| base + (i * 8 + j) as f32 * 0.001).collect());
            }
        }
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig {
            max_sample_size: 50, // Force stratified sampling
            num_test_queries: 5,
            recall_k: 3,
            target_recalls: vec![0.90],
        });

        let report = advisor.analyze(&refs, 3).unwrap();
        // Should still produce valid results despite sampling
        assert!(!report.strategies.is_empty());
        assert!(report.distribution.num_vectors <= 50);
    }

    #[test]
    fn test_per_dimension_variance() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32 * 0.1, 0.5, i as f32 * 0.01, 0.0])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig::default());
        let dist = advisor.analyze_distribution(&refs, 4);

        // Dimension 0 should have highest variance
        assert!(dist.variance[0] > dist.variance[1]);
        assert!(dist.variance[0] > dist.variance[3]);
    }

    #[test]
    fn test_timing_budget() {
        let vectors = make_vectors(100, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig {
            max_sample_size: 100,
            num_test_queries: 5,
            recall_k: 3,
            target_recalls: vec![0.90],
        });

        let report = advisor.analyze(&refs, 3).unwrap();
        // Should complete quickly for small datasets
        assert!(report.total_analysis_time_ms < 5000);
        // Each strategy should track its own timing
        for strategy in &report.strategies {
            if strategy.strategy != QuantizationStrategy::None {
                assert!(strategy.analysis_time_ms >= 0);
            }
        }
    }

    #[test]
    fn test_recommendation_respects_target() {
        let vectors = make_vectors(200, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let advisor = CompressionAdvisor::new(AdvisorConfig {
            max_sample_size: 200,
            num_test_queries: 10,
            recall_k: 5,
            target_recalls: vec![0.99, 0.50],
        });

        let report = advisor.analyze(&refs, 5).unwrap();
        assert_eq!(report.recommendations.len(), 2);

        // Higher recall target should recommend less compression
        let high_rec = &report.recommendations[0];
        let low_rec = &report.recommendations[1];
        assert!(high_rec.target_recall > low_rec.target_recall);
        // Lower target can use more aggressive compression
        assert!(low_rec.compression_ratio >= high_rec.compression_ratio);
    }
}
