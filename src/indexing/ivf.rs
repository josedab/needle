//! # IVF (Inverted File) Index
//!
//! IVF indexing for approximate nearest neighbor search. This module provides
//! cluster-based indexing that partitions vectors into cells based on proximity
//! to learned centroids.
//!
//! ## Index Types
//!
//! - **IvfFlat**: Exact distance computation within clusters
//! - **IvfPQ**: Product quantization for compressed vector storage
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::ivf::{IvfIndex, IvfConfig};
//!
//! // Create an IVF index
//! let config = IvfConfig::new(128)  // 128 clusters
//!     .with_nprobe(16);  // Search 16 nearest clusters
//!
//! let mut index = IvfIndex::new(384, config);
//!
//! // Train on sample vectors
//! index.train(&training_vectors);
//!
//! // Insert vectors
//! index.insert(0, &vector);
//!
//! // Search
//! let results = index.search(&query, 10);
//! ```
//!
//! # When to Use IVF
//!
//! IVF is best suited for memory-constrained scenarios with large, static datasets:
//!
//! | Use Case | IVF Suitability |
//! |----------|-----------------|
//! | Memory constrained | ✅ Excellent - lower memory than HNSW |
//! | Static datasets | ✅ Excellent - train once, query many times |
//! | Batch processing | ✅ Good - efficient for bulk operations |
//! | Very high dimensions | ✅ Good - PQ compression helps |
//! | Real-time search | ⚠️ Slower than HNSW |
//! | Frequent updates | ⚠️ May require retraining |
//! | Small datasets (<100K) | ⚠️ HNSW is simpler and faster |
//!
//! ## IVF vs HNSW
//!
//! - **IVF** requires training but uses less memory
//! - **HNSW** is faster but uses more memory
//! - Choose IVF when memory is limited and you can tolerate slower queries
//! - Choose HNSW when query latency is critical
//!
//! ## IVF vs DiskANN
//!
//! - **IVF** keeps centroids in memory, vectors can be on disk with PQ
//! - **DiskANN** is purpose-built for disk-based search
//! - Choose IVF for moderate-scale datasets with memory constraints
//! - Choose DiskANN for very large datasets (>100M vectors)
//!
//! ## Configuration Guidelines
//!
//! - **n_clusters**: sqrt(n) is a good starting point
//! - **n_probe**: 1-5% of n_clusters for speed/recall tradeoff
//! - **use_pq**: Enable for 4-16x memory reduction at slight accuracy cost

use crate::distance::euclidean_distance;
use crate::quantization::ProductQuantizer;
use ordered_float::OrderedFloat;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use thiserror::Error;

/// IVF index errors
#[derive(Error, Debug)]
pub enum IvfError {
    #[error("Index not trained")]
    NotTrained,

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Training failed: {0}")]
    TrainingFailed(String),

    #[error("Empty training data")]
    EmptyTrainingData,
}

pub type IvfResult<T> = std::result::Result<T, IvfError>;

/// Configuration for IVF index
#[derive(Debug, Clone)]
pub struct IvfConfig {
    /// Number of clusters (centroids)
    pub n_clusters: usize,
    /// Number of clusters to search (nprobe)
    pub n_probe: usize,
    /// Maximum iterations for k-means training
    pub max_iterations: usize,
    /// Convergence threshold
    pub tolerance: f32,
    /// Use product quantization
    pub use_pq: bool,
    /// PQ subvectors (if use_pq is true)
    pub pq_subvectors: usize,
    /// PQ bits per subvector
    pub pq_bits: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: 256,
            n_probe: 8,
            max_iterations: 100,
            tolerance: 1e-4,
            use_pq: false,
            pq_subvectors: 8,
            pq_bits: 8,
        }
    }
}

impl IvfConfig {
    /// Create a new config with specified number of clusters
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            ..Default::default()
        }
    }

    /// Set number of clusters to probe during search
    pub fn with_nprobe(mut self, n_probe: usize) -> Self {
        self.n_probe = n_probe;
        self
    }

    /// Set maximum training iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Enable product quantization
    pub fn with_pq(mut self, subvectors: usize, bits: usize) -> Self {
        self.use_pq = true;
        self.pq_subvectors = subvectors;
        self.pq_bits = bits;
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// A single cluster in the IVF index
#[derive(Debug, Clone)]
struct IvfCluster {
    /// Centroid vector
    centroid: Vec<f32>,
    /// IDs of vectors in this cluster
    ids: Vec<usize>,
    /// Full vectors (for IVF_FLAT)
    vectors: Vec<Vec<f32>>,
    /// PQ codes (for IVF_PQ)
    pq_codes: Vec<Vec<u8>>,
}

impl IvfCluster {
    fn new(centroid: Vec<f32>) -> Self {
        Self {
            centroid,
            ids: Vec::new(),
            vectors: Vec::new(),
            pq_codes: Vec::new(),
        }
    }

    fn clear_vectors(&mut self) {
        self.ids.clear();
        self.vectors.clear();
        self.pq_codes.clear();
    }
}

/// IVF index for approximate nearest neighbor search
pub struct IvfIndex {
    /// Vector dimensions
    dimensions: usize,
    /// Configuration
    config: IvfConfig,
    /// Clusters (after training)
    clusters: Vec<IvfCluster>,
    /// Product quantizer (if enabled)
    pq: Option<ProductQuantizer>,
    /// Whether the index has been trained
    trained: bool,
    /// Total number of vectors
    n_vectors: usize,
}

impl IvfIndex {
    /// Create a new IVF index
    pub fn new(dimensions: usize, config: IvfConfig) -> Self {
        Self {
            dimensions,
            config,
            clusters: Vec::new(),
            pq: None,
            trained: false,
            n_vectors: 0,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(dimensions: usize) -> Self {
        Self::new(dimensions, IvfConfig::default())
    }

    /// Check if index is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get number of vectors in index
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Train the index on a set of vectors using k-means
    pub fn train(&mut self, vectors: &[&[f32]]) -> IvfResult<()> {
        if vectors.is_empty() {
            return Err(IvfError::EmptyTrainingData);
        }

        // Validate dimensions
        for v in vectors {
            if v.len() != self.dimensions {
                return Err(IvfError::DimensionMismatch {
                    expected: self.dimensions,
                    got: v.len(),
                });
            }
        }

        // Run k-means to find centroids
        let centroids = self.kmeans(vectors)?;

        // Initialize clusters
        self.clusters = centroids
            .into_iter()
            .map(IvfCluster::new)
            .collect();

        // Train PQ if enabled
        if self.config.use_pq {
            self.pq = Some(ProductQuantizer::train(
                vectors,
                self.config.pq_subvectors,
            ));
        }

        self.trained = true;
        Ok(())
    }

    /// K-means clustering implementation
    fn kmeans(&self, vectors: &[&[f32]]) -> IvfResult<Vec<Vec<f32>>> {
        let n = vectors.len();
        let k = self.config.n_clusters.min(n);

        // Initialize centroids using k-means++
        let mut centroids = self.kmeans_pp_init(vectors, k);
        let mut assignments = vec![0usize; n];

        for _iter in 0..self.config.max_iterations {
            // Assign vectors to nearest centroid
            let mut changed = false;
            for (i, v) in vectors.iter().enumerate() {
                let nearest = self.find_nearest_centroid(v, &centroids);
                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0f32; self.dimensions]; k];
            let mut counts = vec![0usize; k];

            for (i, v) in vectors.iter().enumerate() {
                let cluster = assignments[i];
                counts[cluster] += 1;
                for (j, &val) in v.iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }

            // Compute mean
            for (c, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[c] as f32;
                    }
                } else {
                    // Empty cluster: reinitialize randomly
                    let random_idx = rand::random::<usize>() % n;
                    *centroid = vectors[random_idx].to_vec();
                }
            }

            // Check convergence
            let max_shift: f32 = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_distance(old, new))
                .fold(0.0f32, |a, b| a.max(b));

            centroids = new_centroids;

            if max_shift < self.config.tolerance {
                break;
            }
        }

        Ok(centroids)
    }

    /// K-means++ initialization
    fn kmeans_pp_init(&self, vectors: &[&[f32]], k: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();
        let n = vectors.len();

        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        let first_idx = rand::random::<usize>() % n;
        centroids.push(vectors[first_idx].to_vec());

        // Remaining centroids: weighted by distance squared
        let mut distances = vec![f32::MAX; n];

        for _ in 1..k {
            // Update distances to nearest centroid
            for (i, v) in vectors.iter().enumerate() {
                let d = euclidean_distance(v, centroids.last().expect("centroids is non-empty"));
                distances[i] = distances[i].min(d);
            }

            // Weighted random selection
            let total: f32 = distances.iter().map(|d| d * d).sum();
            if total == 0.0 {
                // All remaining points are at centroids
                let remaining: Vec<usize> = (0..n)
                    .filter(|&i| !centroids.iter().any(|c| euclidean_distance(vectors[i], c) < 1e-10))
                    .collect();
                if remaining.is_empty() {
                    break;
                }
                let idx = *remaining.choose(&mut rng).expect("remaining is non-empty");
                centroids.push(vectors[idx].to_vec());
            } else {
                let threshold = rand::random::<f32>() * total;
                let mut cumsum = 0.0;
                for (i, d) in distances.iter().enumerate() {
                    cumsum += d * d;
                    if cumsum >= threshold {
                        centroids.push(vectors[i].to_vec());
                        break;
                    }
                }
            }
        }

        centroids
    }

    /// Find nearest centroid index
    fn find_nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        centroids
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| OrderedFloat(euclidean_distance(vector, c)))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Insert a vector into the index
    pub fn insert(&mut self, id: usize, vector: &[f32]) -> IvfResult<()> {
        if !self.trained {
            return Err(IvfError::NotTrained);
        }

        if vector.len() != self.dimensions {
            return Err(IvfError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        // Find nearest cluster
        let cluster_idx = self.find_nearest_centroid(
            vector,
            &self.clusters.iter().map(|c| c.centroid.clone()).collect::<Vec<_>>(),
        );

        let cluster = &mut self.clusters[cluster_idx];
        cluster.ids.push(id);

        if self.config.use_pq {
            if let Some(ref pq) = self.pq {
                cluster.pq_codes.push(pq.encode(vector));
            }
        } else {
            cluster.vectors.push(vector.to_vec());
        }

        self.n_vectors += 1;
        Ok(())
    }

    /// Insert multiple vectors
    pub fn insert_batch(&mut self, vectors: &[(usize, &[f32])]) -> IvfResult<()> {
        for &(id, vector) in vectors {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &[f32], k: usize) -> IvfResult<Vec<(usize, f32)>> {
        if !self.trained {
            return Err(IvfError::NotTrained);
        }

        if query.len() != self.dimensions {
            return Err(IvfError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        // Find n_probe nearest clusters
        let centroids: Vec<Vec<f32>> = self.clusters.iter().map(|c| c.centroid.clone()).collect();
        let mut cluster_distances: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, euclidean_distance(query, c)))
            .collect();

        cluster_distances.sort_by_key(|(_, d)| OrderedFloat(*d));

        // Search in nearest clusters
        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();

        for &(cluster_idx, _) in cluster_distances.iter().take(self.config.n_probe) {
            let cluster = &self.clusters[cluster_idx];

            if self.config.use_pq {
                // Asymmetric distance computation with PQ
                if let Some(ref pq) = self.pq {
                    for (i, code) in cluster.pq_codes.iter().enumerate() {
                        let distance = pq.asymmetric_distance(query, code);
                        let id = cluster.ids[i];
                        heap.push(Reverse((OrderedFloat(distance), id)));
                        if heap.len() > k {
                            heap.pop();
                        }
                    }
                }
            } else {
                // Exact distance computation
                for (i, vector) in cluster.vectors.iter().enumerate() {
                    let distance = euclidean_distance(query, vector);
                    let id = cluster.ids[i];
                    heap.push(Reverse((OrderedFloat(distance), id)));
                    if heap.len() > k {
                        heap.pop();
                    }
                }
            }
        }

        // Extract results
        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|Reverse((d, id))| (id, d.0))
            .collect();

        results.sort_by_key(|(_, d)| OrderedFloat(*d));
        Ok(results)
    }

    /// Get cluster statistics
    pub fn cluster_stats(&self) -> Vec<ClusterStats> {
        self.clusters
            .iter()
            .enumerate()
            .map(|(i, c)| ClusterStats {
                id: i,
                size: c.ids.len(),
                centroid_norm: c.centroid.iter().map(|x| x * x).sum::<f32>().sqrt(),
            })
            .collect()
    }

    /// Remove all vectors from a specific cluster
    pub fn clear_cluster(&mut self, cluster_id: usize) -> IvfResult<usize> {
        if cluster_id >= self.clusters.len() {
            return Err(IvfError::InvalidConfig(format!(
                "Cluster {} does not exist",
                cluster_id
            )));
        }

        let removed = self.clusters[cluster_id].ids.len();
        self.clusters[cluster_id].clear_vectors();
        self.n_vectors -= removed;
        Ok(removed)
    }

    /// Clear all vectors (keep trained centroids)
    pub fn clear(&mut self) {
        for cluster in &mut self.clusters {
            cluster.clear_vectors();
        }
        self.n_vectors = 0;
    }
}

/// Statistics for a cluster
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Cluster ID
    pub id: usize,
    /// Number of vectors in cluster
    pub size: usize,
    /// L2 norm of centroid
    pub centroid_norm: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut rng = thread_rng();
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_ivf_train() {
        let config = IvfConfig::new(4).with_max_iterations(10);
        let mut index = IvfIndex::new(64, config);

        let vectors = random_vectors(100, 64);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        index.train(&refs).unwrap();
        assert!(index.is_trained());
        assert_eq!(index.clusters.len(), 4);
    }

    #[test]
    fn test_ivf_insert_and_search() {
        let config = IvfConfig::new(4).with_nprobe(4); // Search all clusters
        let mut index = IvfIndex::new(32, config);

        // Generate training data
        let train_vectors = random_vectors(100, 32);
        let train_refs: Vec<&[f32]> = train_vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&train_refs).unwrap();

        // Insert vectors
        for (i, v) in train_vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Search - get more results to ensure we find what we're looking for
        let query = &train_vectors[0];
        let results = index.search(query, 100).unwrap();

        // Check we get results
        assert!(!results.is_empty(), "Should return results");

        // Results should contain the query itself (id 0) with low distance
        let query_result = results.iter().find(|(id, _)| *id == 0);
        assert!(query_result.is_some(), "Query vector should be in results");
        assert!(query_result.unwrap().1 < 0.1, "Query distance should be near zero");

        // First result should have low distance (could be query or very similar vector)
        assert!(results[0].1 < 1.0, "Top result should have reasonable distance");
    }

    #[test]
    fn test_ivf_pq() {
        let config = IvfConfig::new(4)
            .with_nprobe(2)
            .with_pq(4, 8);
        let mut index = IvfIndex::new(32, config);

        let vectors = random_vectors(100, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        index.train(&refs).unwrap();
        assert!(index.pq.is_some());

        // Insert and search
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        let results = index.search(&vectors[0], 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_ivf_cluster_stats() {
        let config = IvfConfig::new(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        index.train(&refs).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        let stats = index.cluster_stats();
        assert_eq!(stats.len(), 4);

        let total_size: usize = stats.iter().map(|s| s.size).sum();
        assert_eq!(total_size, 50);
    }

    #[test]
    fn test_ivf_dimension_mismatch() {
        let config = IvfConfig::new(4);
        let mut index = IvfIndex::new(32, config);

        let vectors = random_vectors(50, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        // Try to insert wrong dimension
        let wrong_dim = vec![1.0; 64];
        let result = index.insert(0, &wrong_dim);
        assert!(matches!(result, Err(IvfError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ivf_not_trained() {
        let index = IvfIndex::new(32, IvfConfig::default());

        let result = index.search(&[1.0; 32], 5);
        assert!(matches!(result, Err(IvfError::NotTrained)));
    }
}
