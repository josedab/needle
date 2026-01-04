//! Anomaly Detection
//!
//! Detect outliers and anomalies in vector space using various methods:
//! - Local Outlier Factor (LOF)
//! - Isolation Forest
//! - Distance-based outlier detection
//! - Statistical methods (z-score, IQR)
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::anomaly::{LocalOutlierFactor, IsolationForest};
//!
//! let vectors: Vec<Vec<f32>> = /* your vectors */;
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! // LOF-based detection
//! let lof = LocalOutlierFactor::fit(&refs, 20);
//! let outliers = lof.find_outliers(1.5); // threshold
//!
//! // Isolation Forest
//! let forest = IsolationForest::fit(&refs, 100, 256);
//! let scores = forest.score(&refs);
//! ```

use crate::distance::DistanceFunction;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Local Outlier Factor for density-based anomaly detection
#[derive(Debug, Clone)]
pub struct LocalOutlierFactor {
    /// Training vectors
    vectors: Vec<Vec<f32>>,
    /// Number of neighbors
    k: usize,
    /// Distance function
    distance: DistanceFunction,
    /// Precomputed k-distances
    k_distances: Vec<f32>,
    /// Precomputed LOF scores
    lof_scores: Vec<f32>,
}

impl LocalOutlierFactor {
    /// Fit LOF model to data
    pub fn fit(vectors: &[&[f32]], k: usize) -> Self {
        Self::fit_with_distance(vectors, k, DistanceFunction::Euclidean)
    }

    /// Fit with specific distance function
    pub fn fit_with_distance(
        vectors: &[&[f32]],
        k: usize,
        distance: DistanceFunction,
    ) -> Self {
        let n = vectors.len();
        let k = k.min(n.saturating_sub(1)).max(1);

        let stored: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();

        if n <= 1 {
            return Self {
                vectors: stored,
                k,
                distance,
                k_distances: vec![0.0; n],
                lof_scores: vec![1.0; n],
            };
        }

        // Compute all pairwise distances and k-nearest neighbors
        let mut neighbors: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        let mut k_distances = Vec::with_capacity(n);

        for i in 0..n {
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distance.compute(vectors[i], vectors[j])))
                .collect();

            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);

            k_distances.push(dists.last().map(|(_, d)| *d).unwrap_or(0.0));
            neighbors.push(dists);
        }

        // Compute local reachability density (LRD)
        let lrd: Vec<f32> = (0..n)
            .map(|i| {
                let sum: f32 = neighbors[i]
                    .iter()
                    .map(|(j, d)| d.max(k_distances[*j]))
                    .sum();

                if sum > 0.0 {
                    neighbors[i].len() as f32 / sum
                } else {
                    f32::MAX
                }
            })
            .collect();

        // Compute LOF scores
        let lof_scores: Vec<f32> = (0..n)
            .map(|i| {
                if lrd[i] == 0.0 || lrd[i] == f32::MAX {
                    return 1.0;
                }

                let lof_sum: f32 = neighbors[i]
                    .iter()
                    .map(|(j, _)| {
                        if lrd[*j] == f32::MAX {
                            0.0
                        } else {
                            lrd[*j]
                        }
                    })
                    .sum();

                if neighbors[i].is_empty() {
                    1.0
                } else {
                    lof_sum / (neighbors[i].len() as f32 * lrd[i])
                }
            })
            .collect();

        Self {
            vectors: stored,
            k,
            distance,
            k_distances,
            lof_scores,
        }
    }

    /// Get LOF scores for training data
    pub fn scores(&self) -> &[f32] {
        &self.lof_scores
    }

    /// Find outliers above threshold
    pub fn find_outliers(&self, threshold: f32) -> Vec<usize> {
        self.lof_scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Score new vectors (predict outlier scores)
    pub fn score_new(&self, vectors: &[&[f32]]) -> Vec<f32> {
        if self.vectors.is_empty() {
            return vec![1.0; vectors.len()];
        }

        vectors
            .iter()
            .map(|v| self.score_single(v))
            .collect()
    }

    /// Score a single vector
    fn score_single(&self, vector: &[f32]) -> f32 {
        let n = self.vectors.len();
        let k = self.k.min(n);

        // Find k nearest neighbors
        let mut dists: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, self.distance.compute(vector, v)))
            .collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);

        if dists.is_empty() {
            return 1.0;
        }

        // Compute reachability distances
        let reach_sum: f32 = dists
            .iter()
            .map(|(j, d)| d.max(self.k_distances[*j]))
            .sum();

        let lrd_new = if reach_sum > 0.0 {
            k as f32 / reach_sum
        } else {
            return 1.0;
        };

        // Compute LOF
        let lof_sum: f32 = dists
            .iter()
            .map(|(j, _)| {
                let lrd_j = {
                    let sum: f32 = self.vectors
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| *i != *j)
                        .map(|(i, v)| {
                            let d = self.distance.compute(&self.vectors[*j], v);
                            d.max(self.k_distances[i])
                        })
                        .take(self.k)
                        .sum();
                    if sum > 0.0 { self.k as f32 / sum } else { f32::MAX }
                };
                if lrd_j == f32::MAX { 0.0 } else { lrd_j }
            })
            .sum();

        if lrd_new > 0.0 {
            lof_sum / (k as f32 * lrd_new)
        } else {
            1.0
        }
    }

    /// Get number of neighbors
    pub fn k(&self) -> usize {
        self.k
    }
}

/// Isolation Forest for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    /// Isolation trees
    trees: Vec<IsolationTree>,
    /// Sample size per tree
    sample_size: usize,
    /// Average path length for normalization
    avg_path_length: f32,
}

impl IsolationForest {
    /// Fit an isolation forest
    pub fn fit(vectors: &[&[f32]], n_trees: usize, sample_size: usize) -> Self {
        let n = vectors.len();
        let sample_size = sample_size.min(n);

        if n == 0 || vectors[0].is_empty() {
            return Self {
                trees: Vec::new(),
                sample_size,
                avg_path_length: 1.0,
            };
        }

        let dims = vectors[0].len();
        let max_depth = (sample_size as f32).log2().ceil() as usize;

        let mut rng = rand::thread_rng();
        let trees: Vec<IsolationTree> = (0..n_trees)
            .map(|_| {
                // Sample subset
                let indices: Vec<usize> = (0..n).collect();
                let sample: Vec<usize> = indices
                    .choose_multiple(&mut rng, sample_size)
                    .copied()
                    .collect();

                let sample_vecs: Vec<&[f32]> = sample.iter().map(|&i| vectors[i]).collect();
                IsolationTree::build(&sample_vecs, dims, max_depth, &mut rng)
            })
            .collect();

        let avg_path_length = Self::average_path_length(sample_size);

        Self {
            trees,
            sample_size,
            avg_path_length,
        }
    }

    /// Compute anomaly scores (higher = more anomalous)
    pub fn score(&self, vectors: &[&[f32]]) -> Vec<f32> {
        vectors.iter().map(|v| self.score_single(v)).collect()
    }

    /// Score a single vector
    pub fn score_single(&self, vector: &[f32]) -> f32 {
        if self.trees.is_empty() {
            return 0.5;
        }

        let avg_path: f32 = self.trees.iter().map(|t| t.path_length(vector) as f32).sum();
        let avg_path = avg_path / self.trees.len() as f32;

        // Anomaly score: 2^(-avg_path / c(n))
        2.0_f32.powf(-avg_path / self.avg_path_length)
    }

    /// Find outliers above threshold
    pub fn find_outliers(&self, vectors: &[&[f32]], threshold: f32) -> Vec<usize> {
        self.score(vectors)
            .iter()
            .enumerate()
            .filter(|(_, &score)| score > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Average path length for sample size n
    fn average_path_length(n: usize) -> f32 {
        if n <= 1 {
            return 1.0;
        }
        let n = n as f32;
        2.0 * (n.ln() + 0.577_215_7) - (2.0 * (n - 1.0) / n)
    }

    /// Get number of trees
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

/// Single isolation tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationTree {
    root: Option<IsolationNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum IsolationNode {
    Internal {
        feature: usize,
        threshold: f32,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    Leaf {
        size: usize,
    },
}

impl IsolationTree {
    fn build(vectors: &[&[f32]], dims: usize, max_depth: usize, rng: &mut impl Rng) -> Self {
        let root = Self::build_node(vectors, dims, 0, max_depth, rng);
        Self { root: Some(root) }
    }

    fn build_node(
        vectors: &[&[f32]],
        dims: usize,
        depth: usize,
        max_depth: usize,
        rng: &mut impl Rng,
    ) -> IsolationNode {
        if depth >= max_depth || vectors.len() <= 1 {
            return IsolationNode::Leaf {
                size: vectors.len(),
            };
        }

        // Random feature
        let feature = rng.gen_range(0..dims);

        // Find min/max for this feature
        let values: Vec<f32> = vectors.iter().map(|v| v[feature]).collect();
        let min = values.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max = values.iter().fold(f32::MIN, |a, &b| a.max(b));

        if (max - min).abs() < 1e-10 {
            return IsolationNode::Leaf {
                size: vectors.len(),
            };
        }

        // Random threshold
        let threshold = rng.gen::<f32>() * (max - min) + min;

        // Split
        let (left_vecs, right_vecs): (Vec<&[f32]>, Vec<&[f32]>) = vectors
            .iter()
            .partition(|v| v[feature] < threshold);

        if left_vecs.is_empty() || right_vecs.is_empty() {
            return IsolationNode::Leaf {
                size: vectors.len(),
            };
        }

        IsolationNode::Internal {
            feature,
            threshold,
            left: Box::new(Self::build_node(&left_vecs, dims, depth + 1, max_depth, rng)),
            right: Box::new(Self::build_node(&right_vecs, dims, depth + 1, max_depth, rng)),
        }
    }

    fn path_length(&self, vector: &[f32]) -> usize {
        match &self.root {
            Some(node) => Self::path_length_node(node, vector, 0),
            None => 0,
        }
    }

    fn path_length_node(node: &IsolationNode, vector: &[f32], depth: usize) -> usize {
        match node {
            IsolationNode::Leaf { size } => {
                depth + IsolationForest::average_path_length(*size) as usize
            }
            IsolationNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if vector.get(*feature).copied().unwrap_or(0.0) < *threshold {
                    Self::path_length_node(left, vector, depth + 1)
                } else {
                    Self::path_length_node(right, vector, depth + 1)
                }
            }
        }
    }
}

/// Distance-based outlier detection
#[derive(Debug, Clone)]
pub struct DistanceOutlierDetector {
    /// Reference vectors
    vectors: Vec<Vec<f32>>,
    /// Distance function
    distance: DistanceFunction,
    /// Number of neighbors for distance calculation
    k: usize,
}

impl DistanceOutlierDetector {
    /// Create new detector
    pub fn new(vectors: &[&[f32]], k: usize, distance: DistanceFunction) -> Self {
        Self {
            vectors: vectors.iter().map(|v| v.to_vec()).collect(),
            distance,
            k: k.min(vectors.len().saturating_sub(1)).max(1),
        }
    }

    /// Get average distance to k nearest neighbors
    pub fn avg_knn_distances(&self) -> Vec<f32> {
        let n = self.vectors.len();

        (0..n)
            .map(|i| {
                let mut dists: Vec<f32> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| self.distance.compute(&self.vectors[i], &self.vectors[j]))
                    .collect();

                dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(self.k);

                if dists.is_empty() {
                    0.0
                } else {
                    dists.iter().sum::<f32>() / dists.len() as f32
                }
            })
            .collect()
    }

    /// Find outliers based on distance threshold
    pub fn find_outliers(&self, threshold: f32) -> Vec<usize> {
        self.avg_knn_distances()
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Find outliers using percentile
    pub fn find_outliers_percentile(&self, percentile: f32) -> Vec<usize> {
        let distances = self.avg_knn_distances();
        let mut sorted = distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((percentile / 100.0) * sorted.len() as f32) as usize;
        let threshold = sorted.get(idx).copied().unwrap_or(f32::MAX);

        distances
            .iter()
            .enumerate()
            .filter(|(_, &d)| d > threshold)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Statistical outlier detection for individual dimensions
#[derive(Debug, Clone)]
pub struct StatisticalOutlierDetector {
    /// Mean per dimension
    means: Vec<f32>,
    /// Std dev per dimension
    stds: Vec<f32>,
    /// Q1 per dimension
    q1: Vec<f32>,
    /// Q3 per dimension
    q3: Vec<f32>,
}

impl StatisticalOutlierDetector {
    /// Fit detector to data
    pub fn fit(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            return Self {
                means: Vec::new(),
                stds: Vec::new(),
                q1: Vec::new(),
                q3: Vec::new(),
            };
        }

        let dims = vectors[0].len();
        let n = vectors.len() as f32;

        let mut means = vec![0.0; dims];
        let mut stds = vec![0.0; dims];
        let mut q1 = vec![0.0; dims];
        let mut q3 = vec![0.0; dims];

        for d in 0..dims {
            let mut values: Vec<f32> = vectors.iter().map(|v| v[d]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Mean
            means[d] = values.iter().sum::<f32>() / n;

            // Std
            let variance: f32 = values.iter().map(|&x| (x - means[d]).powi(2)).sum::<f32>() / n;
            stds[d] = variance.sqrt();

            // Quartiles
            let n = values.len();
            q1[d] = values[n / 4];
            q3[d] = values[(3 * n) / 4];
        }

        Self { means, stds, q1, q3 }
    }

    /// Z-score based outlier detection
    pub fn zscore_outliers(&self, vectors: &[&[f32]], threshold: f32) -> Vec<usize> {
        vectors
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                v.iter().zip(self.means.iter().zip(self.stds.iter())).any(
                    |(&val, (&mean, &std))| {
                        if std > 0.0 {
                            ((val - mean) / std).abs() > threshold
                        } else {
                            false
                        }
                    },
                )
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// IQR-based outlier detection
    pub fn iqr_outliers(&self, vectors: &[&[f32]], multiplier: f32) -> Vec<usize> {
        vectors
            .iter()
            .enumerate()
            .filter(|(_, v)| {
                v.iter()
                    .zip(self.q1.iter().zip(self.q3.iter()))
                    .any(|(&val, (&q1, &q3))| {
                        let iqr = q3 - q1;
                        val < q1 - multiplier * iqr || val > q3 + multiplier * iqr
                    })
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Get z-scores for all vectors
    pub fn zscores(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        vectors
            .iter()
            .map(|v| {
                v.iter()
                    .zip(self.means.iter().zip(self.stds.iter()))
                    .map(|(&val, (&mean, &std))| {
                        if std > 0.0 {
                            (val - mean) / std
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

/// Combined anomaly detector using multiple methods
#[derive(Debug)]
pub struct EnsembleAnomalyDetector {
    lof: LocalOutlierFactor,
    iforest: IsolationForest,
    distance_detector: DistanceOutlierDetector,
}

impl EnsembleAnomalyDetector {
    /// Create ensemble detector
    pub fn fit(vectors: &[&[f32]], lof_k: usize, n_trees: usize) -> Self {
        let sample_size = 256.min(vectors.len());

        Self {
            lof: LocalOutlierFactor::fit(vectors, lof_k),
            iforest: IsolationForest::fit(vectors, n_trees, sample_size),
            distance_detector: DistanceOutlierDetector::new(
                vectors,
                lof_k,
                DistanceFunction::Euclidean,
            ),
        }
    }

    /// Get combined anomaly scores
    pub fn scores(&self) -> Vec<f32> {
        let lof_scores = self.lof.scores();
        let n = lof_scores.len();

        // Normalize LOF scores (typically > 1 for outliers)
        let lof_max = lof_scores.iter().fold(1.0_f32, |a, &b| a.max(b));
        let lof_normalized: Vec<f32> = lof_scores.iter().map(|&s| s / lof_max).collect();

        // Get isolation forest scores
        let iforest_scores: Vec<f32> = (0..n)
            .map(|i| self.iforest.score_single(&self.lof.vectors[i]))
            .collect();

        // Get distance-based scores
        let dist_scores = self.distance_detector.avg_knn_distances();
        let dist_max = dist_scores.iter().fold(1.0_f32, |a, &b| a.max(b));
        let dist_normalized: Vec<f32> = if dist_max > 0.0 {
            dist_scores.iter().map(|&s| s / dist_max).collect()
        } else {
            vec![0.0; n]
        };

        // Combine scores (average)
        (0..n)
            .map(|i| (lof_normalized[i] + iforest_scores[i] + dist_normalized[i]) / 3.0)
            .collect()
    }

    /// Find consensus outliers (detected by multiple methods)
    pub fn find_consensus_outliers(
        &self,
        lof_threshold: f32,
        iforest_threshold: f32,
        min_votes: usize,
    ) -> Vec<usize> {
        let n = self.lof.scores().len();
        let lof_outliers: HashSet<usize> = self.lof.find_outliers(lof_threshold).into_iter().collect();

        let iforest_scores: Vec<f32> = (0..n)
            .map(|i| self.iforest.score_single(&self.lof.vectors[i]))
            .collect();
        let iforest_outliers: HashSet<usize> = iforest_scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > iforest_threshold)
            .map(|(i, _)| i)
            .collect();

        let dist_outliers: HashSet<usize> = self
            .distance_detector
            .find_outliers_percentile(95.0)
            .into_iter()
            .collect();

        (0..n)
            .filter(|&i| {
                let votes = [
                    lof_outliers.contains(&i),
                    iforest_outliers.contains(&i),
                    dist_outliers.contains(&i),
                ]
                .iter()
                .filter(|&&v| v)
                .count();
                votes >= min_votes
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    fn create_data_with_outliers() -> (Vec<Vec<f32>>, Vec<usize>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut vectors = Vec::new();
        let mut outlier_indices = Vec::new();

        // Normal points clustered around origin
        for _ in 0..95 {
            let v: Vec<f32> = (0..8)
                .map(|_| (rng.gen::<f32>() - 0.5) * 2.0)
                .collect();
            vectors.push(v);
        }

        // Outliers far from cluster
        for _ in 0..5 {
            let v: Vec<f32> = (0..8)
                .map(|_| (rng.gen::<f32>() - 0.5) * 20.0 + 10.0)
                .collect();
            outlier_indices.push(vectors.len());
            vectors.push(v);
        }

        (vectors, outlier_indices)
    }

    #[test]
    fn test_lof_basic() {
        let (vectors, _) = create_data_with_outliers();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let lof = LocalOutlierFactor::fit(&refs, 10);
        let scores = lof.scores();

        assert_eq!(scores.len(), vectors.len());
        assert!(scores.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn test_lof_finds_outliers() {
        let (vectors, true_outliers) = create_data_with_outliers();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let lof = LocalOutlierFactor::fit(&refs, 10);
        let detected = lof.find_outliers(1.5);

        // Should detect at least some of the true outliers
        let detected_set: HashSet<usize> = detected.into_iter().collect();
        let true_set: HashSet<usize> = true_outliers.into_iter().collect();
        let overlap = detected_set.intersection(&true_set).count();

        assert!(overlap > 0, "LOF should detect at least some outliers");
    }

    #[test]
    fn test_isolation_forest_basic() {
        let vectors: Vec<Vec<f32>> = (0..100).map(|_| random_vector(8)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let forest = IsolationForest::fit(&refs, 50, 64);
        let scores = forest.score(&refs);

        assert_eq!(scores.len(), vectors.len());
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_isolation_forest_finds_outliers() {
        let (vectors, _) = create_data_with_outliers();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let forest = IsolationForest::fit(&refs, 100, 64);
        let scores = forest.score(&refs);

        // Outliers should have higher scores
        let normal_avg: f32 = scores[..95].iter().sum::<f32>() / 95.0;
        let outlier_avg: f32 = scores[95..].iter().sum::<f32>() / 5.0;

        assert!(
            outlier_avg > normal_avg,
            "Outliers should have higher anomaly scores"
        );
    }

    #[test]
    fn test_distance_outlier_detector() {
        let (vectors, _) = create_data_with_outliers();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let detector = DistanceOutlierDetector::new(&refs, 10, DistanceFunction::Euclidean);
        let distances = detector.avg_knn_distances();

        assert_eq!(distances.len(), vectors.len());

        // Outliers should have larger distances
        let normal_avg: f32 = distances[..95].iter().sum::<f32>() / 95.0;
        let outlier_avg: f32 = distances[95..].iter().sum::<f32>() / 5.0;

        assert!(outlier_avg > normal_avg);
    }

    #[test]
    fn test_statistical_detector() {
        let (vectors, _) = create_data_with_outliers();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let detector = StatisticalOutlierDetector::fit(&refs);
        let zscore_outliers = detector.zscore_outliers(&refs, 2.0);
        let iqr_outliers = detector.iqr_outliers(&refs, 1.5);

        // Should detect some outliers
        assert!(!zscore_outliers.is_empty() || !iqr_outliers.is_empty());
    }

    #[test]
    fn test_ensemble_detector() {
        let (vectors, _) = create_data_with_outliers();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let ensemble = EnsembleAnomalyDetector::fit(&refs, 10, 50);
        let scores = ensemble.scores();

        assert_eq!(scores.len(), vectors.len());
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_lof_score_new() {
        let vectors: Vec<Vec<f32>> = (0..50).map(|_| random_vector(8)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let lof = LocalOutlierFactor::fit(&refs, 10);

        // Score new vectors
        let new_vectors: Vec<Vec<f32>> = (0..5).map(|_| random_vector(8)).collect();
        let new_refs: Vec<&[f32]> = new_vectors.iter().map(|v| v.as_slice()).collect();

        let scores = lof.score_new(&new_refs);
        assert_eq!(scores.len(), 5);
    }

    #[test]
    fn test_empty_data() {
        let empty: Vec<&[f32]> = Vec::new();

        let lof = LocalOutlierFactor::fit(&empty, 10);
        assert!(lof.scores().is_empty());

        let forest = IsolationForest::fit(&empty, 10, 32);
        assert_eq!(forest.n_trees(), 0);
    }
}
