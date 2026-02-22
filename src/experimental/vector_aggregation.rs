#![allow(clippy::unwrap_used)]
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Vector Aggregation Primitives
//!
//! Native query operators for computing aggregations over vector collections:
//! CENTROID, MEDOID, SPREAD, CLUSTER(k), DISTRIBUTION, OUTLIERS.
//! Uses Rayon for parallel computation.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::vector_aggregation::*;
//!
//! let vectors: Vec<Vec<f32>> = get_vectors();
//! let centroid = compute_centroid(&vectors, 128);
//! let medoid = compute_medoid(&vectors, DistanceFn::Cosine);
//! let outliers = detect_outliers(&vectors, 2.0);
//! ```

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Distance function for aggregation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggDistanceFn {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Dot product distance (negative dot product)
    DotProduct,
}

impl Default for AggDistanceFn {
    fn default() -> Self {
        Self::Euclidean
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x - *y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| (*x as f64) * (*y as f64)).sum();
    let mag_a: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    if mag_a < 1e-10 || mag_b < 1e-10 {
        1.0
    } else {
        1.0 - (dot / (mag_a * mag_b))
    }
}

fn compute_distance(a: &[f32], b: &[f32], method: AggDistanceFn) -> f64 {
    match method {
        AggDistanceFn::Euclidean => euclidean_distance(a, b),
        AggDistanceFn::Cosine => cosine_distance(a, b),
        AggDistanceFn::DotProduct => {
            -a.iter()
                .zip(b.iter())
                .map(|(x, y)| (*x as f64) * (*y as f64))
                .sum::<f64>()
        }
    }
}

/// Result of an aggregation operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    /// Type of aggregation performed
    pub operation: String,
    /// Result vectors (single for centroid/medoid, multiple for clusters)
    pub vectors: Vec<Vec<f32>>,
    /// Scalar metrics
    pub metrics: HashMap<String, f64>,
    /// Optional cluster assignments (vector index -> cluster index)
    pub assignments: Option<Vec<usize>>,
    /// Computation time in microseconds
    pub compute_time_us: u64,
}

/// Compute the centroid (mean) of a set of vectors using Rayon parallelism.
pub fn compute_centroid(vectors: &[&[f32]], dimensions: usize) -> Vec<f32> {
    if vectors.is_empty() {
        return vec![0.0; dimensions];
    }

    let n = vectors.len() as f32;

    // Use parallel fold for large vector sets
    if vectors.len() > 1000 {
        let sums: Vec<f64> = (0..dimensions)
            .into_par_iter()
            .map(|d| vectors.iter().map(|v| v[d] as f64).sum::<f64>())
            .collect();
        sums.iter().map(|s| (*s / n as f64) as f32).collect()
    } else {
        let mut result = vec![0.0f64; dimensions];
        for v in vectors {
            for (i, val) in v.iter().enumerate() {
                result[i] += *val as f64;
            }
        }
        result.iter().map(|s| (*s / n as f64) as f32).collect()
    }
}

/// Compute the medoid (actual vector closest to centroid) of a set of vectors.
pub fn compute_medoid(vectors: &[&[f32]], distance_fn: AggDistanceFn) -> Option<(usize, Vec<f32>)> {
    if vectors.is_empty() {
        return None;
    }

    // Find vector with minimum sum of distances to all others
    let best_idx = if vectors.len() > 500 {
        // Parallel for large sets
        (0..vectors.len())
            .into_par_iter()
            .map(|i| {
                let total_dist: f64 = vectors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, v)| compute_distance(vectors[i], v, distance_fn))
                    .sum();
                (i, total_dist)
            })
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    } else {
        (0..vectors.len())
            .map(|i| {
                let total_dist: f64 = vectors
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, v)| compute_distance(vectors[i], v, distance_fn))
                    .sum();
                (i, total_dist)
            })
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    };

    best_idx.map(|idx| (idx, vectors[idx].to_vec()))
}

/// Compute spread (variance) of vectors around their centroid.
pub fn compute_spread(vectors: &[&[f32]], dimensions: usize, distance_fn: AggDistanceFn) -> f64 {
    if vectors.len() <= 1 {
        return 0.0;
    }

    let centroid = compute_centroid(vectors, dimensions);

    let total_dist: f64 = if vectors.len() > 1000 {
        vectors
            .par_iter()
            .map(|v| compute_distance(v, &centroid, distance_fn).powi(2))
            .sum()
    } else {
        vectors
            .iter()
            .map(|v| compute_distance(v, &centroid, distance_fn).powi(2))
            .sum()
    };

    total_dist / vectors.len() as f64
}

/// Cluster vectors using k-means with Rayon parallelism.
pub fn cluster_vectors(
    vectors: &[&[f32]],
    k: usize,
    dimensions: usize,
    max_iterations: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    if vectors.is_empty() || k == 0 {
        return (Vec::new(), Vec::new());
    }

    let k = k.min(vectors.len());

    // Initialize centroids using k-means++ style
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let mut rng = rand::thread_rng();
    use rand::Rng;

    centroids.push(vectors[rng.gen_range(0..vectors.len())].to_vec());

    for _ in 1..k {
        let distances: Vec<f64> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| euclidean_distance(v, c))
                    .fold(f64::MAX, f64::min)
            })
            .collect();

        let total: f64 = distances.iter().sum();
        if total < 1e-10 {
            break;
        }

        let threshold = rng.gen::<f64>() * total;
        let mut cum = 0.0;
        let mut selected = 0;
        for (i, d) in distances.iter().enumerate() {
            cum += d;
            if cum >= threshold {
                selected = i;
                break;
            }
        }
        centroids.push(vectors[selected].to_vec());
    }

    let mut assignments = vec![0usize; vectors.len()];

    for _iter in 0..max_iterations {
        // Assign step (parallel)
        let new_assignments: Vec<usize> = if vectors.len() > 500 {
            vectors
                .par_iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, euclidean_distance(v, c)))
                        .min_by(|a, b| {
                            a.1.partial_cmp(&b.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect()
        } else {
            vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, euclidean_distance(v, c)))
                        .min_by(|a, b| {
                            a.1.partial_cmp(&b.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect()
        };

        if new_assignments == assignments {
            break; // Converged
        }
        assignments = new_assignments;

        // Update centroids
        for c_idx in 0..centroids.len() {
            let cluster_vecs: Vec<&[f32]> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &a)| a == c_idx)
                .map(|(i, _)| vectors[i])
                .collect();

            if !cluster_vecs.is_empty() {
                centroids[c_idx] = compute_centroid(&cluster_vecs, dimensions);
            }
        }
    }

    (centroids, assignments)
}

/// Compute the distribution of vectors (histogram of distances from centroid).
pub fn compute_distribution(
    vectors: &[&[f32]],
    dimensions: usize,
    num_bins: usize,
    distance_fn: AggDistanceFn,
) -> DistributionResult {
    if vectors.is_empty() {
        return DistributionResult {
            bin_edges: Vec::new(),
            bin_counts: Vec::new(),
            mean_distance: 0.0,
            std_distance: 0.0,
            min_distance: 0.0,
            max_distance: 0.0,
        };
    }

    let centroid = compute_centroid(vectors, dimensions);
    let distances: Vec<f64> = vectors
        .iter()
        .map(|v| compute_distance(v, &centroid, distance_fn))
        .collect();

    let min_d = distances.iter().cloned().fold(f64::MAX, f64::min);
    let max_d = distances.iter().cloned().fold(f64::MIN, f64::max);
    let mean_d: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
    let variance: f64 = distances.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>()
        / distances.len() as f64;
    let std_d = variance.sqrt();

    let bin_width = if (max_d - min_d).abs() < 1e-10 {
        1.0
    } else {
        (max_d - min_d) / num_bins as f64
    };

    let mut bin_counts = vec![0u64; num_bins];
    let bin_edges: Vec<f64> = (0..=num_bins)
        .map(|i| min_d + i as f64 * bin_width)
        .collect();

    for d in &distances {
        let bin = (((d - min_d) / bin_width) as usize).min(num_bins - 1);
        bin_counts[bin] += 1;
    }

    DistributionResult {
        bin_edges,
        bin_counts,
        mean_distance: mean_d,
        std_distance: std_d,
        min_distance: min_d,
        max_distance: max_d,
    }
}

/// Distribution analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionResult {
    /// Bin edge values
    pub bin_edges: Vec<f64>,
    /// Count of vectors in each bin
    pub bin_counts: Vec<u64>,
    /// Mean distance from centroid
    pub mean_distance: f64,
    /// Standard deviation of distances
    pub std_distance: f64,
    /// Minimum distance
    pub min_distance: f64,
    /// Maximum distance
    pub max_distance: f64,
}

/// An outlier detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierResult {
    /// Index of the outlier vector
    pub index: usize,
    /// Distance from centroid
    pub distance: f64,
    /// Z-score
    pub z_score: f64,
}

/// Detect outlier vectors that are far from the centroid.
pub fn detect_outliers(
    vectors: &[&[f32]],
    dimensions: usize,
    threshold: f64,
    distance_fn: AggDistanceFn,
) -> Vec<OutlierResult> {
    if vectors.len() <= 2 {
        return Vec::new();
    }

    let centroid = compute_centroid(vectors, dimensions);
    let distances: Vec<f64> = vectors
        .iter()
        .map(|v| compute_distance(v, &centroid, distance_fn))
        .collect();

    let mean_d: f64 = distances.iter().sum::<f64>() / distances.len() as f64;
    let variance: f64 = distances.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>()
        / distances.len() as f64;
    let std_d = variance.sqrt();

    if std_d < 1e-10 {
        return Vec::new();
    }

    distances
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| {
            let z_score = (d - mean_d) / std_d;
            if z_score > threshold {
                Some(OutlierResult {
                    index: i,
                    distance: d,
                    z_score,
                })
            } else {
                None
            }
        })
        .collect()
}

/// Aggregation operation types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationOp {
    /// Compute centroid (mean vector)
    Centroid,
    /// Compute medoid (actual closest to center)
    Medoid,
    /// Compute spread (variance)
    Spread,
    /// Cluster into k groups
    Cluster { k: usize },
    /// Compute distance distribution histogram
    Distribution { bins: usize },
    /// Detect outliers beyond threshold standard deviations
    Outliers { threshold: f64 },
}

/// Execute an aggregation operation on a set of vectors.
pub fn execute_aggregation(
    vectors: &[&[f32]],
    dimensions: usize,
    op: &AggregationOp,
    distance_fn: AggDistanceFn,
) -> AggregationResult {
    let start = std::time::Instant::now();

    let result = match op {
        AggregationOp::Centroid => {
            let centroid = compute_centroid(vectors, dimensions);
            AggregationResult {
                operation: "centroid".to_string(),
                vectors: vec![centroid],
                metrics: HashMap::new(),
                assignments: None,
                compute_time_us: 0,
            }
        }
        AggregationOp::Medoid => {
            let medoid = compute_medoid(vectors, distance_fn);
            let mut metrics = HashMap::new();
            let vecs = if let Some((idx, v)) = medoid {
                metrics.insert("medoid_index".to_string(), idx as f64);
                vec![v]
            } else {
                Vec::new()
            };
            AggregationResult {
                operation: "medoid".to_string(),
                vectors: vecs,
                metrics,
                assignments: None,
                compute_time_us: 0,
            }
        }
        AggregationOp::Spread => {
            let spread = compute_spread(vectors, dimensions, distance_fn);
            let mut metrics = HashMap::new();
            metrics.insert("spread".to_string(), spread);
            metrics.insert("std_dev".to_string(), spread.sqrt());
            AggregationResult {
                operation: "spread".to_string(),
                vectors: Vec::new(),
                metrics,
                assignments: None,
                compute_time_us: 0,
            }
        }
        AggregationOp::Cluster { k } => {
            let (centroids, assignments) = cluster_vectors(vectors, *k, dimensions, 50);
            let mut metrics = HashMap::new();
            metrics.insert("num_clusters".to_string(), centroids.len() as f64);
            AggregationResult {
                operation: "cluster".to_string(),
                vectors: centroids,
                metrics,
                assignments: Some(assignments),
                compute_time_us: 0,
            }
        }
        AggregationOp::Distribution { bins } => {
            let dist = compute_distribution(vectors, dimensions, *bins, distance_fn);
            let mut metrics = HashMap::new();
            metrics.insert("mean_distance".to_string(), dist.mean_distance);
            metrics.insert("std_distance".to_string(), dist.std_distance);
            metrics.insert("min_distance".to_string(), dist.min_distance);
            metrics.insert("max_distance".to_string(), dist.max_distance);
            AggregationResult {
                operation: "distribution".to_string(),
                vectors: Vec::new(),
                metrics,
                assignments: None,
                compute_time_us: 0,
            }
        }
        AggregationOp::Outliers { threshold } => {
            let outliers = detect_outliers(vectors, dimensions, *threshold, distance_fn);
            let mut metrics = HashMap::new();
            metrics.insert("outlier_count".to_string(), outliers.len() as f64);
            let outlier_indices: Vec<f64> = outliers.iter().map(|o| o.index as f64).collect();
            AggregationResult {
                operation: "outliers".to_string(),
                vectors: Vec::new(),
                metrics,
                assignments: None,
                compute_time_us: 0,
            }
        }
    };

    AggregationResult {
        compute_time_us: start.elapsed().as_micros() as u64,
        ..result
    }
}

/// Cached aggregation result with TTL.
#[derive(Debug, Clone)]
pub struct CachedAggregation {
    /// The cached result
    pub result: AggregationResult,
    /// When it was computed
    pub computed_at: SystemTime,
    /// Time-to-live
    pub ttl: Duration,
}

impl CachedAggregation {
    /// Check if the cached result has expired.
    pub fn is_expired(&self) -> bool {
        SystemTime::now()
            .duration_since(self.computed_at)
            .unwrap_or(Duration::MAX)
            > self.ttl
    }
}

/// Cache for aggregation results.
pub struct AggregationCache {
    cache: HashMap<String, CachedAggregation>,
    default_ttl: Duration,
}

impl AggregationCache {
    /// Create a new aggregation cache.
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            default_ttl,
        }
    }

    /// Get a cached result if available and not expired.
    pub fn get(&self, key: &str) -> Option<&AggregationResult> {
        self.cache
            .get(key)
            .filter(|c| !c.is_expired())
            .map(|c| &c.result)
    }

    /// Store a result in the cache.
    pub fn put(&mut self, key: String, result: AggregationResult) {
        self.cache.insert(
            key,
            CachedAggregation {
                result,
                computed_at: SystemTime::now(),
                ttl: self.default_ttl,
            },
        );
    }

    /// Store a result with a custom TTL.
    pub fn put_with_ttl(&mut self, key: String, result: AggregationResult, ttl: Duration) {
        self.cache.insert(
            key,
            CachedAggregation {
                result,
                computed_at: SystemTime::now(),
                ttl,
            },
        );
    }

    /// Evict expired entries.
    pub fn evict_expired(&mut self) {
        self.cache.retain(|_, v| !v.is_expired());
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vectors() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
        ]
    }

    fn as_slices(vecs: &[Vec<f32>]) -> Vec<&[f32]> {
        vecs.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn test_centroid() {
        let vecs = test_vectors();
        let slices = as_slices(&vecs);
        let centroid = compute_centroid(&slices, 3);
        assert_eq!(centroid.len(), 3);
        // Mean of all vectors
        assert!((centroid[0] - 0.4).abs() < 1e-6);
        assert!((centroid[1] - 0.6).abs() < 1e-6);
        assert!((centroid[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_medoid() {
        let vecs = test_vectors();
        let slices = as_slices(&vecs);
        let (idx, medoid) = compute_medoid(&slices, AggDistanceFn::Euclidean).unwrap();
        assert!(idx < vecs.len());
        assert_eq!(medoid.len(), 3);
    }

    #[test]
    fn test_spread() {
        let vecs = test_vectors();
        let slices = as_slices(&vecs);
        let spread = compute_spread(&slices, 3, AggDistanceFn::Euclidean);
        assert!(spread > 0.0);
    }

    #[test]
    fn test_cluster() {
        let vecs = test_vectors();
        let slices = as_slices(&vecs);
        let (centroids, assignments) = cluster_vectors(&slices, 2, 3, 50);
        assert_eq!(centroids.len(), 2);
        assert_eq!(assignments.len(), vecs.len());
        for &a in &assignments {
            assert!(a < 2);
        }
    }

    #[test]
    fn test_distribution() {
        let vecs = test_vectors();
        let slices = as_slices(&vecs);
        let dist = compute_distribution(&slices, 3, 5, AggDistanceFn::Euclidean);
        assert_eq!(dist.bin_counts.len(), 5);
        assert!(dist.mean_distance > 0.0);
    }

    #[test]
    fn test_outliers() {
        let mut vecs = test_vectors();
        // Add an outlier
        vecs.push(vec![10.0, 10.0, 10.0]);
        let slices = as_slices(&vecs);
        let outliers = detect_outliers(&slices, 3, 1.5, AggDistanceFn::Euclidean);
        assert!(!outliers.is_empty());
        // The far-away vector should be an outlier
        assert!(outliers.iter().any(|o| o.index == 5));
    }

    #[test]
    fn test_execute_aggregation() {
        let vecs = test_vectors();
        let slices = as_slices(&vecs);
        let result =
            execute_aggregation(&slices, 3, &AggregationOp::Centroid, AggDistanceFn::Euclidean);
        assert_eq!(result.operation, "centroid");
        assert_eq!(result.vectors.len(), 1);
    }

    #[test]
    fn test_aggregation_cache() {
        let mut cache = AggregationCache::new(Duration::from_secs(60));

        let result = AggregationResult {
            operation: "centroid".to_string(),
            vectors: vec![vec![0.5; 3]],
            metrics: HashMap::new(),
            assignments: None,
            compute_time_us: 42,
        };

        cache.put("test_key".to_string(), result);
        assert!(cache.get("test_key").is_some());
        assert!(cache.get("missing").is_none());
        assert_eq!(cache.len(), 1);
    }
}
