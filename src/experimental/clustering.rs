//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Vector Clustering
//!
//! Provides clustering algorithms for organizing and exploring vector data:
//! - K-means clustering
//! - Hierarchical/Agglomerative clustering
//! - Mini-batch K-means for large datasets
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::clustering::{KMeans, ClusteringConfig};
//!
//! let vectors: Vec<Vec<f32>> = /* your vectors */;
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! let kmeans = KMeans::fit(&refs, 5, ClusteringConfig::default())?;
//! let labels = kmeans.predict(&refs);
//! let centroids = kmeans.centroids();
//! ```

use crate::distance::DistanceFunction;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringConfig {
    /// Maximum iterations for convergence
    pub max_iterations: usize,
    /// Convergence threshold (centroid movement)
    pub tolerance: f32,
    /// Distance function to use
    pub distance: DistanceFunction,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Number of random initializations (best result kept)
    pub n_init: usize,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            max_iterations: 300,
            tolerance: 1e-4,
            distance: DistanceFunction::Euclidean,
            seed: None,
            n_init: 10,
        }
    }
}

impl ClusteringConfig {
    /// Use cosine distance
    pub fn with_cosine(mut self) -> Self {
        self.distance = DistanceFunction::Cosine;
        self
    }

    /// Set max iterations
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// K-Means clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans {
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,
    /// Configuration used
    config: ClusteringConfig,
    /// Number of iterations until convergence
    iterations: usize,
    /// Final inertia (sum of squared distances to centroids)
    inertia: f32,
}

impl KMeans {
    /// Fit K-means clustering to vectors
    pub fn fit(vectors: &[&[f32]], k: usize, config: ClusteringConfig) -> Result<Self, String> {
        if vectors.is_empty() {
            return Err("Cannot cluster empty dataset".to_string());
        }
        if k == 0 {
            return Err("k must be at least 1".to_string());
        }
        if k > vectors.len() {
            return Err(format!(
                "k ({}) cannot exceed number of vectors ({})",
                k,
                vectors.len()
            ));
        }

        let dims = vectors[0].len();
        for (i, v) in vectors.iter().enumerate() {
            if v.len() != dims {
                return Err(format!(
                    "Vector {} has {} dimensions, expected {}",
                    i,
                    v.len(),
                    dims
                ));
            }
        }

        let mut best_result: Option<KMeans> = None;

        for init in 0..config.n_init {
            let seed = config.seed.map(|s| s + init as u64);
            match Self::fit_single(vectors, k, &config, seed) {
                Ok(result) => {
                    if best_result.is_none()
                        || result.inertia < best_result.as_ref().expect("checked is_none above").inertia
                    {
                        best_result = Some(result);
                    }
                }
                Err(e) => {
                    if init == 0 {
                        return Err(e);
                    }
                }
            }
        }

        best_result.ok_or_else(|| "All clustering attempts failed".to_string())
    }

    /// Single K-means run
    fn fit_single(
        vectors: &[&[f32]],
        k: usize,
        config: &ClusteringConfig,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        let dims = vectors[0].len();

        // Initialize centroids using K-means++
        let mut centroids = Self::kmeans_plus_plus_init(vectors, k, config, seed);

        let mut iterations = 0;
        let mut prev_inertia = f32::MAX;

        for iter in 0..config.max_iterations {
            iterations = iter + 1;

            // Assign points to nearest centroid
            let assignments: Vec<usize> = vectors
                .iter()
                .map(|v| Self::nearest_centroid(v, &centroids, &config.distance))
                .collect();

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dims]; k];
            let mut counts = vec![0usize; k];

            for (vec, &cluster) in vectors.iter().zip(assignments.iter()) {
                counts[cluster] += 1;
                for (j, &val) in vec.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }

            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[i] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[i] as f32;
                    }
                } else {
                    // Empty cluster: reinitialize with random point
                    let mut rng = Self::get_rng(seed.map(|s| s + iter as u64));
                    if let Some(vec) = vectors.choose(&mut rng) {
                        *centroid = vec.to_vec();
                    }
                }
            }

            // Calculate inertia
            let inertia: f32 = vectors
                .iter()
                .zip(assignments.iter())
                .map(|(v, &c)| {
                    let dist = config.distance.compute(v, &new_centroids[c]);
                    dist * dist
                })
                .sum();

            // Check convergence
            if (prev_inertia - inertia).abs() < config.tolerance {
                centroids = new_centroids;
                return Ok(Self {
                    centroids,
                    config: config.clone(),
                    iterations,
                    inertia,
                });
            }

            prev_inertia = inertia;
            centroids = new_centroids;
        }

        let inertia: f32 = vectors
            .iter()
            .map(|v| {
                let c = Self::nearest_centroid(v, &centroids, &config.distance);
                let dist = config.distance.compute(v, &centroids[c]);
                dist * dist
            })
            .sum();

        Ok(Self {
            centroids,
            config: config.clone(),
            iterations,
            inertia,
        })
    }

    /// K-means++ initialization
    fn kmeans_plus_plus_init(
        vectors: &[&[f32]],
        k: usize,
        config: &ClusteringConfig,
        seed: Option<u64>,
    ) -> Vec<Vec<f32>> {
        let mut rng = Self::get_rng(seed);
        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        let first_idx = rng.gen_range(0..vectors.len());
        centroids.push(vectors[first_idx].to_vec());

        // Remaining centroids: weighted by distance squared
        for _ in 1..k {
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| {
                            let d = config.distance.compute(v, c);
                            d * d
                        })
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total == 0.0 {
                // All points are at centroids, pick random
                let idx = rng.gen_range(0..vectors.len());
                centroids.push(vectors[idx].to_vec());
                continue;
            }

            let threshold = rng.gen::<f32>() * total;
            let mut cumsum = 0.0;
            let mut selected = vectors.len() - 1;

            for (i, &d) in distances.iter().enumerate() {
                cumsum += d;
                if cumsum >= threshold {
                    selected = i;
                    break;
                }
            }

            centroids.push(vectors[selected].to_vec());
        }

        centroids
    }

    /// Find nearest centroid
    fn nearest_centroid(
        vector: &[f32],
        centroids: &[Vec<f32>],
        distance: &DistanceFunction,
    ) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, distance.compute(vector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn get_rng(seed: Option<u64>) -> impl Rng {
        use rand::SeedableRng;
        match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Get cluster centroids
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Get number of clusters
    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    /// Get number of iterations until convergence
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Get final inertia
    pub fn inertia(&self) -> f32 {
        self.inertia
    }

    /// Predict cluster labels for vectors
    pub fn predict(&self, vectors: &[&[f32]]) -> Vec<usize> {
        vectors
            .iter()
            .map(|v| Self::nearest_centroid(v, &self.centroids, &self.config.distance))
            .collect()
    }

    /// Predict single vector
    pub fn predict_one(&self, vector: &[f32]) -> usize {
        Self::nearest_centroid(vector, &self.centroids, &self.config.distance)
    }

    /// Get distances to each centroid
    pub fn distances(&self, vector: &[f32]) -> Vec<f32> {
        self.centroids
            .iter()
            .map(|c| self.config.distance.compute(vector, c))
            .collect()
    }

    /// Get cluster sizes from predictions
    pub fn cluster_sizes(&self, labels: &[usize]) -> Vec<usize> {
        let mut sizes = vec![0; self.centroids.len()];
        for &label in labels {
            if label < sizes.len() {
                sizes[label] += 1;
            }
        }
        sizes
    }
}

/// Mini-batch K-means for large datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBatchKMeans {
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,
    /// Configuration
    config: ClusteringConfig,
    /// Batch size
    batch_size: usize,
    /// Counts per centroid (for incremental updates)
    counts: Vec<usize>,
}

impl MiniBatchKMeans {
    /// Create new mini-batch K-means
    pub fn new(k: usize, dims: usize, batch_size: usize, config: ClusteringConfig) -> Self {
        Self {
            centroids: vec![vec![0.0; dims]; k],
            config,
            batch_size,
            counts: vec![0; k],
        }
    }

    /// Initialize centroids from sample
    pub fn init(&mut self, sample: &[&[f32]]) {
        if sample.is_empty() {
            return;
        }
        self.centroids =
            KMeans::kmeans_plus_plus_init(sample, self.centroids.len(), &self.config, None);
    }

    /// Partial fit on a batch
    pub fn partial_fit(&mut self, batch: &[&[f32]]) {
        if batch.is_empty() {
            return;
        }

        // Assign points to centroids
        let assignments: Vec<usize> = batch
            .iter()
            .map(|v| KMeans::nearest_centroid(v, &self.centroids, &self.config.distance))
            .collect();

        // Update centroids incrementally
        for (vec, &cluster) in batch.iter().zip(assignments.iter()) {
            self.counts[cluster] += 1;
            let eta = 1.0 / self.counts[cluster] as f32;

            for (j, &val) in vec.iter().enumerate() {
                self.centroids[cluster][j] += eta * (val - self.centroids[cluster][j]);
            }
        }
    }

    /// Fit on full dataset using batches
    pub fn fit(vectors: &[&[f32]], k: usize, batch_size: usize, config: ClusteringConfig) -> Self {
        if vectors.is_empty() {
            return Self::new(k, 0, batch_size, config);
        }

        let dims = vectors[0].len();
        let mut kmeans = Self::new(k, dims, batch_size, config);

        // Initialize from sample
        let sample_size = (batch_size * 3).min(vectors.len());
        let sample: Vec<&[f32]> = vectors.iter().take(sample_size).copied().collect();
        kmeans.init(&sample);

        // Process batches
        for chunk in vectors.chunks(batch_size) {
            let batch: Vec<&[f32]> = chunk.to_vec();
            kmeans.partial_fit(&batch);
        }

        kmeans
    }

    /// Get centroids
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Predict cluster labels
    pub fn predict(&self, vectors: &[&[f32]]) -> Vec<usize> {
        vectors
            .iter()
            .map(|v| KMeans::nearest_centroid(v, &self.centroids, &self.config.distance))
            .collect()
    }
}

/// Hierarchical clustering linkage methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Linkage {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage (mean distance)
    Average,
    /// Ward's method (minimum variance)
    Ward,
}

/// Hierarchical clustering result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalClustering {
    /// Merge history: (cluster_a, cluster_b, distance, new_size)
    merges: Vec<(usize, usize, f32, usize)>,
    /// Number of original points
    n_samples: usize,
    /// Distance function used
    distance: DistanceFunction,
    /// Linkage method
    linkage: Linkage,
}

impl HierarchicalClustering {
    /// Perform agglomerative hierarchical clustering
    pub fn fit(vectors: &[&[f32]], linkage: Linkage, distance: DistanceFunction) -> Self {
        let n = vectors.len();
        if n == 0 {
            return Self {
                merges: Vec::new(),
                n_samples: 0,
                distance,
                linkage,
            };
        }

        // Compute initial distance matrix
        let mut dist_matrix: Vec<Vec<f32>> = vec![vec![f32::MAX; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = distance.compute(vectors[i], vectors[j]);
                dist_matrix[i][j] = d;
                dist_matrix[j][i] = d;
            }
            dist_matrix[i][i] = 0.0;
        }

        // Track active clusters and their sizes
        let mut active: Vec<bool> = vec![true; n];
        let mut sizes: Vec<usize> = vec![1; n];
        let mut merges = Vec::with_capacity(n - 1);

        // Cluster centroids for Ward's method
        let mut centroids: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();

        for _ in 0..(n - 1) {
            // Find minimum distance pair
            let mut min_dist = f32::MAX;
            let mut min_i = 0;
            let mut min_j = 0;

            for i in 0..n {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !active[j] {
                        continue;
                    }
                    if dist_matrix[i][j] < min_dist {
                        min_dist = dist_matrix[i][j];
                        min_i = i;
                        min_j = j;
                    }
                }
            }

            if min_dist == f32::MAX {
                break;
            }

            // Merge clusters
            let new_size = sizes[min_i] + sizes[min_j];
            merges.push((min_i, min_j, min_dist, new_size));

            // Update centroid for Ward's method
            if matches!(linkage, Linkage::Ward) {
                let dims = centroids[0].len();
                let mut new_centroid = vec![0.0; dims];
                for d in 0..dims {
                    new_centroid[d] = (centroids[min_i][d] * sizes[min_i] as f32
                        + centroids[min_j][d] * sizes[min_j] as f32)
                        / new_size as f32;
                }
                centroids[min_i] = new_centroid;
            }

            // Deactivate j, keep i as merged cluster
            active[min_j] = false;
            sizes[min_i] = new_size;

            // Update distances from merged cluster to all others
            for k in 0..n {
                if !active[k] || k == min_i {
                    continue;
                }

                let new_dist = match linkage {
                    Linkage::Single => dist_matrix[min_i][k].min(dist_matrix[min_j][k]),
                    Linkage::Complete => dist_matrix[min_i][k].max(dist_matrix[min_j][k]),
                    Linkage::Average => {
                        let n_i = (sizes[min_i] - sizes[min_j]) as f32;
                        let n_j = sizes[min_j] as f32;
                        (n_i * dist_matrix[min_i][k] + n_j * dist_matrix[min_j][k]) / (n_i + n_j)
                    }
                    Linkage::Ward => {
                        // Ward's formula
                        let n_i = (sizes[min_i] - sizes[min_j]) as f32;
                        let n_j = sizes[min_j] as f32;
                        let n_k = sizes[k] as f32;
                        let n_total = n_i + n_j + n_k;

                        let d_ik = dist_matrix[min_i][k];
                        let d_jk = dist_matrix[min_j][k];
                        let d_ij = min_dist;

                        (((n_i + n_k) * d_ik * d_ik + (n_j + n_k) * d_jk * d_jk
                            - n_k * d_ij * d_ij)
                            / n_total)
                            .sqrt()
                    }
                };

                dist_matrix[min_i][k] = new_dist;
                dist_matrix[k][min_i] = new_dist;
            }
        }

        Self {
            merges,
            n_samples: n,
            distance,
            linkage,
        }
    }

    /// Get cluster labels for k clusters
    pub fn cut(&self, k: usize) -> Vec<usize> {
        if k == 0 || k > self.n_samples {
            return (0..self.n_samples).collect();
        }

        let n = self.n_samples;
        let mut labels: Vec<usize> = (0..n).collect();

        // Apply merges up to n - k (leaving k clusters)
        let n_merges = (n - k).min(self.merges.len());

        for (i, j, _, _) in self.merges.iter().take(n_merges) {
            let label_i = labels[*i];
            let label_j = labels[*j];

            // All points with label_j get label_i
            for label in labels.iter_mut() {
                if *label == label_j {
                    *label = label_i;
                }
            }
        }

        // Renumber labels to be contiguous 0..k
        let mut label_map: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0;

        for label in &mut labels {
            if let Some(&new_label) = label_map.get(label) {
                *label = new_label;
            } else {
                label_map.insert(*label, next_label);
                *label = next_label;
                next_label += 1;
            }
        }

        labels
    }

    /// Get merge distances (for dendrogram)
    pub fn distances(&self) -> Vec<f32> {
        self.merges.iter().map(|(_, _, d, _)| *d).collect()
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

/// Find optimal number of clusters using elbow method
pub fn elbow_method(vectors: &[&[f32]], max_k: usize, config: &ClusteringConfig) -> Vec<f32> {
    let max_k = max_k.min(vectors.len());
    let mut inertias = Vec::with_capacity(max_k);

    for k in 1..=max_k {
        match KMeans::fit(vectors, k, config.clone()) {
            Ok(kmeans) => inertias.push(kmeans.inertia()),
            Err(_) => inertias.push(f32::MAX),
        }
    }

    inertias
}

/// Silhouette score for clustering quality
pub fn silhouette_score(
    vectors: &[&[f32]],
    labels: &[usize],
    distance: &DistanceFunction,
) -> f32 {
    if vectors.len() != labels.len() || vectors.is_empty() {
        return 0.0;
    }

    let n = vectors.len();
    let k = labels.iter().max().copied().unwrap_or(0) + 1;

    if k <= 1 {
        return 0.0;
    }

    let mut total_silhouette = 0.0;

    for i in 0..n {
        let label_i = labels[i];

        // a(i) = average distance to points in same cluster
        let mut same_cluster_dist = 0.0;
        let mut same_cluster_count = 0;

        // b(i) = minimum average distance to points in other clusters
        let mut other_cluster_dists = vec![0.0; k];
        let mut other_cluster_counts = vec![0; k];

        for j in 0..n {
            if i == j {
                continue;
            }

            let dist = distance.compute(vectors[i], vectors[j]);

            if labels[j] == label_i {
                same_cluster_dist += dist;
                same_cluster_count += 1;
            } else {
                other_cluster_dists[labels[j]] += dist;
                other_cluster_counts[labels[j]] += 1;
            }
        }

        let a = if same_cluster_count > 0 {
            same_cluster_dist / same_cluster_count as f32
        } else {
            0.0
        };

        let b = (0..k)
            .filter(|&c| c != label_i && other_cluster_counts[c] > 0)
            .map(|c| other_cluster_dists[c] / other_cluster_counts[c] as f32)
            .fold(f32::MAX, f32::min);

        let silhouette = if a.max(b) > 0.0 {
            (b - a) / a.max(b)
        } else {
            0.0
        };

        total_silhouette += silhouette;
    }

    total_silhouette / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    fn create_clustered_data(
        k: usize,
        points_per_cluster: usize,
        dims: usize,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut vectors = Vec::new();
        let mut labels = Vec::new();

        for cluster in 0..k {
            // Random centroid
            let centroid: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>() * 10.0).collect();

            for _ in 0..points_per_cluster {
                // Add noise around centroid
                let point: Vec<f32> = centroid
                    .iter()
                    .map(|&c| c + (rng.gen::<f32>() - 0.5) * 0.5)
                    .collect();
                vectors.push(point);
                labels.push(cluster);
            }
        }

        (vectors, labels)
    }

    #[test]
    fn test_kmeans_basic() {
        let (vectors, _) = create_clustered_data(3, 20, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default().with_seed(42);
        let kmeans = KMeans::fit(&refs, 3, config).unwrap();

        assert_eq!(kmeans.k(), 3);
        assert_eq!(kmeans.centroids().len(), 3);
        assert!(kmeans.inertia() > 0.0);

        let labels = kmeans.predict(&refs);
        assert_eq!(labels.len(), vectors.len());
    }

    #[test]
    fn test_kmeans_convergence() {
        let vectors: Vec<Vec<f32>> = (0..50).map(|_| random_vector(16)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default()
            .with_seed(42)
            .with_max_iterations(100);
        let kmeans = KMeans::fit(&refs, 5, config).unwrap();

        assert!(kmeans.iterations() <= 100);
    }

    #[test]
    fn test_minibatch_kmeans() {
        let vectors: Vec<Vec<f32>> = (0..200).map(|_| random_vector(16)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default().with_seed(42);
        let kmeans = MiniBatchKMeans::fit(&refs, 5, 32, config);

        assert_eq!(kmeans.centroids().len(), 5);

        let labels = kmeans.predict(&refs);
        assert_eq!(labels.len(), vectors.len());
    }

    #[test]
    fn test_hierarchical_clustering() {
        let vectors = [vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 5.1]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let hc = HierarchicalClustering::fit(&refs, Linkage::Single, DistanceFunction::Euclidean);

        // Cut into 2 clusters
        let labels = hc.cut(2);
        assert_eq!(labels.len(), 4);

        // Points 0,1 should be in same cluster, 2,3 in another
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_elbow_method() {
        let (vectors, _) = create_clustered_data(3, 30, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default().with_seed(42);
        let inertias = elbow_method(&refs, 6, &config);

        assert_eq!(inertias.len(), 6);
        // Inertia should generally decrease as k increases
        assert!(inertias[0] >= inertias[5]);
    }

    #[test]
    fn test_silhouette_score() {
        let (vectors, true_labels) = create_clustered_data(3, 30, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let score = silhouette_score(&refs, &true_labels, &DistanceFunction::Euclidean);

        // Well-separated clusters should have positive silhouette score
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_kmeans_predict_one() {
        let vectors: Vec<Vec<f32>> = (0..20).map(|_| random_vector(8)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default().with_seed(42);
        let kmeans = KMeans::fit(&refs, 3, config).unwrap();

        let new_vector = random_vector(8);
        let label = kmeans.predict_one(&new_vector);
        assert!(label < 3);
    }

    #[test]
    fn test_kmeans_distances() {
        let vectors: Vec<Vec<f32>> = (0..20).map(|_| random_vector(8)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default().with_seed(42);
        let kmeans = KMeans::fit(&refs, 3, config).unwrap();

        let distances = kmeans.distances(&vectors[0]);
        assert_eq!(distances.len(), 3);
        assert!(distances.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_cluster_sizes() {
        let vectors: Vec<Vec<f32>> = (0..30).map(|_| random_vector(8)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let config = ClusteringConfig::default().with_seed(42);
        let kmeans = KMeans::fit(&refs, 3, config).unwrap();
        let labels = kmeans.predict(&refs);
        let sizes = kmeans.cluster_sizes(&labels);

        assert_eq!(sizes.len(), 3);
        assert_eq!(sizes.iter().sum::<usize>(), 30);
    }
}
