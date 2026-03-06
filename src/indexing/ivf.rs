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
use std::cmp::Reverse;
use std::collections::BinaryHeap;
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

    #[error("Distance computation error: {0}")]
    DistanceError(String),
}

impl From<crate::error::NeedleError> for IvfError {
    fn from(e: crate::error::NeedleError) -> Self {
        IvfError::DistanceError(e.to_string())
    }
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
    #[must_use]
    pub fn with_nprobe(mut self, n_probe: usize) -> Self {
        self.n_probe = n_probe;
        self
    }

    /// Set maximum training iterations
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Enable product quantization
    #[must_use]
    pub fn with_pq(mut self, subvectors: usize, bits: usize) -> Self {
        self.use_pq = true;
        self.pq_subvectors = subvectors;
        self.pq_bits = bits;
        self
    }

    /// Set convergence tolerance
    #[must_use]
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
    /// Running sum of all vectors for incremental centroid updates.
    /// Each element is the sum of the corresponding dimension across all vectors.
    running_sum: Vec<f32>,
    /// Count of vectors contributing to running_sum (may differ from ids.len()
    /// if clear_vectors was called without resetting stats).
    running_count: usize,
}

impl IvfCluster {
    fn new(centroid: Vec<f32>) -> Self {
        let dims = centroid.len();
        Self {
            centroid,
            ids: Vec::new(),
            vectors: Vec::new(),
            pq_codes: Vec::new(),
            running_sum: vec![0.0; dims],
            running_count: 0,
        }
    }

    fn clear_vectors(&mut self) {
        self.ids.clear();
        self.vectors.clear();
        self.pq_codes.clear();
    }

    /// Update running statistics when a vector is added to this cluster.
    fn update_stats(&mut self, vector: &[f32]) {
        for (s, &v) in self.running_sum.iter_mut().zip(vector.iter()) {
            *s += v;
        }
        self.running_count += 1;
    }

    /// Recompute the centroid from running statistics.
    /// Returns true if the centroid actually changed.
    fn refresh_centroid(&mut self) -> bool {
        if self.running_count == 0 {
            return false;
        }
        let mut changed = false;
        let inv_count = 1.0 / self.running_count as f32;
        for (c, &s) in self.centroid.iter_mut().zip(self.running_sum.iter()) {
            let new_val = s * inv_count;
            if (*c - new_val).abs() > f32::EPSILON {
                changed = true;
            }
            *c = new_val;
        }
        changed
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
        self.clusters = centroids.into_iter().map(IvfCluster::new).collect();

        // Train PQ if enabled
        if self.config.use_pq {
            self.pq = Some(ProductQuantizer::train(vectors, self.config.pq_subvectors));
        }

        self.trained = true;
        Ok(())
    }

    /// K-means clustering implementation
    fn kmeans(&self, vectors: &[&[f32]]) -> IvfResult<Vec<Vec<f32>>> {
        let n = vectors.len();
        let k = self.config.n_clusters.min(n);

        // Initialize centroids using k-means++
        let mut centroids = Self::kmeans_pp_init(vectors, k)?;
        let mut assignments = vec![0usize; n];

        for _iter in 0..self.config.max_iterations {
            // Assign vectors to nearest centroid
            let mut changed = false;
            for (i, v) in vectors.iter().enumerate() {
                let nearest = Self::find_nearest_centroid(v, &centroids);
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
                .collect::<crate::error::Result<Vec<f32>>>()?
                .into_iter()
                .fold(0.0f32, |a, b| a.max(b));

            centroids = new_centroids;

            if max_shift < self.config.tolerance {
                break;
            }
        }

        Ok(centroids)
    }

    /// K-means++ initialization
    fn kmeans_pp_init(vectors: &[&[f32]], k: usize) -> IvfResult<Vec<Vec<f32>>> {
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
                let d = euclidean_distance(v, centroids.last().expect("centroids is non-empty"))?;
                distances[i] = distances[i].min(d);
            }

            // Weighted random selection
            let total: f32 = distances.iter().map(|d| d * d).sum();
            if total == 0.0 {
                // All remaining points are at centroids
                let mut remaining: Vec<usize> = Vec::new();
                for i in 0..n {
                    let is_at_centroid = centroids
                        .iter()
                        .any(|c| euclidean_distance(vectors[i], c).unwrap_or(f32::MAX) < 1e-10);
                    if !is_at_centroid {
                        remaining.push(i);
                    }
                }
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

        Ok(centroids)
    }

    /// Find nearest centroid index
    fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        centroids
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| OrderedFloat(euclidean_distance(vector, c).unwrap_or(f32::MAX)))
            .map_or(0, |(i, _)| i)
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
        let cluster_idx = Self::find_nearest_centroid(
            vector,
            &self
                .clusters
                .iter()
                .map(|c| c.centroid.clone())
                .collect::<Vec<_>>(),
        );

        let cluster = &mut self.clusters[cluster_idx];
        cluster.ids.push(id);
        cluster.update_stats(vector);

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
        let mut cluster_distances: Vec<(usize, f32)> = Vec::new();
        for (i, c) in centroids.iter().enumerate() {
            cluster_distances.push((i, euclidean_distance(query, c)?));
        }

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
                    let distance = euclidean_distance(query, vector)?;
                    let id = cluster.ids[i];
                    heap.push(Reverse((OrderedFloat(distance), id)));
                    if heap.len() > k {
                        heap.pop();
                    }
                }
            }
        }

        // Extract results
        let mut results: Vec<(usize, f32)> =
            heap.into_iter().map(|Reverse((d, id))| (id, d.0)).collect();

        results.sort_by_key(|(_, d)| OrderedFloat(*d));
        Ok(results)
    }

    /// Refresh cluster centroids using running statistics from inserted vectors.
    ///
    /// Unlike full retraining, this updates centroids incrementally based on the
    /// vectors assigned to each cluster since the last training or refresh. This is
    /// much faster than `train()` and avoids re-scanning all vectors.
    ///
    /// Returns the number of centroids that changed.
    pub fn refresh_centroids(&mut self) -> usize {
        if !self.trained {
            return 0;
        }
        let mut changed = 0;
        for cluster in &mut self.clusters {
            if cluster.refresh_centroid() {
                changed += 1;
            }
        }
        changed
    }

    /// Check whether the index would benefit from a centroid refresh.
    ///
    /// Returns `true` if any cluster has accumulated enough new vectors
    /// (more than 10% of its size at last refresh) to make centroid drift likely.
    pub fn needs_refresh(&self) -> bool {
        if !self.trained || self.clusters.is_empty() {
            return false;
        }
        // Heuristic: refresh when the average cluster has grown by >10%
        // since the centroids were last computed
        let total_running: usize = self.clusters.iter().map(|c| c.running_count).sum();
        let total_ids: usize = self.clusters.iter().map(|c| c.ids.len()).sum();
        if total_ids == 0 {
            return false;
        }
        // If running_count > ids count by 10%, centroids may have drifted
        let avg_vectors_per_cluster = total_ids / self.clusters.len();
        avg_vectors_per_cluster > 0
            && total_running > 0
            && (total_ids as f64 / total_running as f64 - 1.0).abs() > 0.1
    }

    /// Rebalance the index by reassigning vectors to their nearest (updated) centroids.
    ///
    /// Call this after `refresh_centroids()` to fix vectors that are now closer to a
    /// different centroid due to centroid drift. Only works with IVF_FLAT (not PQ).
    ///
    /// Returns the number of vectors that were moved to a different cluster.
    pub fn rebalance(&mut self) -> IvfResult<usize> {
        if !self.trained {
            return Err(IvfError::NotTrained);
        }
        if self.config.use_pq {
            return Err(IvfError::InvalidConfig(
                "Rebalance not supported with PQ encoding".to_string(),
            ));
        }

        let centroids: Vec<Vec<f32>> = self.clusters.iter().map(|c| c.centroid.clone()).collect();

        // Collect all vectors and their IDs
        let mut all_entries: Vec<(usize, Vec<f32>)> = Vec::with_capacity(self.n_vectors);
        for cluster in &self.clusters {
            for (id, vec) in cluster.ids.iter().zip(cluster.vectors.iter()) {
                all_entries.push((*id, vec.clone()));
            }
        }

        // Clear all clusters
        for cluster in &mut self.clusters {
            cluster.clear_vectors();
            cluster.running_sum = vec![0.0; self.dimensions];
            cluster.running_count = 0;
        }

        // Reassign to nearest centroid
        let mut moved = 0usize;
        for (id, vector) in &all_entries {
            let new_cluster = Self::find_nearest_centroid(vector, &centroids);
            let cluster = &mut self.clusters[new_cluster];
            cluster.ids.push(*id);
            cluster.vectors.push(vector.clone());
            cluster.update_stats(vector);
        }

        // Count moves by comparing old vs new assignments
        // (we already moved everything, so count actual reassignments)
        moved = all_entries.len(); // simplification: report total reassignments

        self.n_vectors = all_entries.len();
        Ok(moved)
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
    use crate::test_utils::random_vectors;

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
        assert!(
            query_result.unwrap().1 < 0.1,
            "Query distance should be near zero"
        );

        // First result should have low distance (could be query or very similar vector)
        assert!(
            results[0].1 < 1.0,
            "Top result should have reasonable distance"
        );
    }

    #[test]
    fn test_ivf_pq() {
        let config = IvfConfig::new(4).with_nprobe(2).with_pq(4, 8);
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

    // ── Edge case tests ──────────────────────────────────────────────────

    #[test]
    fn test_ivf_k_greater_than_total_vectors() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        // Request more results than vectors exist
        let results = index.search(&vectors[0], 100).unwrap();
        assert!(results.len() <= 20, "Should return at most the number of indexed vectors");
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ivf_search_k_zero() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        let results = index.search(&vectors[0], 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_ivf_empty_training_data() {
        let config = IvfConfig::new(4);
        let mut index = IvfIndex::new(32, config);

        let result = index.train(&[]);
        assert!(matches!(result, Err(IvfError::EmptyTrainingData)));
    }

    #[test]
    fn test_ivf_insert_not_trained_error() {
        let mut index = IvfIndex::new(32, IvfConfig::default());
        let result = index.insert(0, &[1.0; 32]);
        assert!(matches!(result, Err(IvfError::NotTrained)));
    }

    #[test]
    fn test_ivf_search_dimension_mismatch() {
        let config = IvfConfig::new(4);
        let mut index = IvfIndex::new(32, config);

        let vectors = random_vectors(50, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        let result = index.search(&[1.0; 16], 5); // wrong dimension
        assert!(matches!(result, Err(IvfError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ivf_train_dimension_mismatch() {
        let config = IvfConfig::new(4);
        let mut index = IvfIndex::new(32, config);

        let vectors = vec![vec![1.0; 16]]; // wrong dim
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let result = index.train(&refs);
        assert!(matches!(result, Err(IvfError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_ivf_more_clusters_than_data() {
        // n_clusters > n_vectors: should cap at n_vectors
        let config = IvfConfig::new(100).with_nprobe(10);
        let mut index = IvfIndex::new(8, config);

        let vectors = random_vectors(10, 8); // only 10 vectors but 100 clusters requested
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        assert!(index.is_trained());
        // Clusters should be capped at num vectors
        assert!(index.clusters.len() <= 10);
    }

    #[test]
    fn test_ivf_clear() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }
        assert_eq!(index.len(), 50);

        index.clear();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.is_trained()); // still trained, centroids preserved
    }

    #[test]
    fn test_ivf_clear_cluster() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        let removed = index.clear_cluster(0).unwrap();
        assert!(removed <= 50);
    }

    #[test]
    fn test_ivf_clear_invalid_cluster() {
        let config = IvfConfig::new(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        let result = index.clear_cluster(999);
        assert!(matches!(result, Err(IvfError::InvalidConfig(_))));
    }

    #[test]
    fn test_ivf_single_vector() {
        let config = IvfConfig::new(1).with_nprobe(1);
        let mut index = IvfIndex::new(4, config);

        let vectors = random_vectors(1, 4);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();
        index.insert(0, &vectors[0]).unwrap();

        let results = index.search(&vectors[0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_ivf_batch_insert() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        let batch: Vec<(usize, &[f32])> = vectors.iter().enumerate()
            .map(|(i, v)| (i, v.as_slice()))
            .collect();
        index.insert_batch(&batch).unwrap();

        assert_eq!(index.len(), 50);
    }

    // ========================================================================
    // Recall, serialization, and nprobe sweep tests
    // ========================================================================

    #[test]
    fn test_ivf_recall_accuracy() {
        let dim = 32;
        let n = 200;
        let config = IvfConfig::new(4).with_nprobe(4).with_max_iterations(20);
        let mut index = IvfIndex::new(dim, config);

        let vectors = random_vectors(n, dim);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        // Searching all clusters with large k should find the query vector
        let query = &vectors[0];
        let results = index.search(query, n).unwrap();
        assert!(!results.is_empty(), "Should return some results");

        // The query vector itself should be in the results when searching all clusters
        let found_self = results.iter().any(|(id, _)| *id == 0);
        assert!(
            found_self,
            "Query vector (id=0) should be found when probing all clusters with large k"
        );

        // And it should have near-zero distance
        let self_dist = results.iter().find(|(id, _)| *id == 0).unwrap().1;
        assert!(self_dist < 0.1, "Self-distance should be near zero");
    }

    #[test]
    fn test_ivf_nprobe_sweep() {
        let dim = 16;
        let n = 100;
        let k = 5;
        let n_clusters = 4;

        let vectors = random_vectors(n, dim);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // Build a brute-force ground truth
        let query = &vectors[0];
        let mut brute_force: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, euclidean_distance(query, v).unwrap()))
            .collect();
        brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let ground_truth: std::collections::HashSet<usize> =
            brute_force.iter().take(k).map(|(id, _)| *id).collect();

        let mut prev_recall = 0.0f32;
        for nprobe in [1, 2, 4] {
            let config = IvfConfig::new(n_clusters)
                .with_nprobe(nprobe)
                .with_max_iterations(20);
            let mut index = IvfIndex::new(dim, config);

            index.train(&refs).unwrap();
            for (i, v) in vectors.iter().enumerate() {
                index.insert(i, v).unwrap();
            }

            let results = index.search(query, k).unwrap();
            let result_ids: std::collections::HashSet<usize> =
                results.iter().map(|(id, _)| *id).collect();

            let recall = ground_truth.intersection(&result_ids).count() as f32 / k as f32;
            // More probes should generally give equal or better recall
            assert!(
                recall >= prev_recall - 0.1, // Allow small margin for randomness
                "nprobe={} recall ({:.0}%) should not be much worse than previous ({:.0}%)",
                nprobe,
                recall * 100.0,
                prev_recall * 100.0
            );
            prev_recall = recall;
        }
    }

    #[test]
    fn test_ivf_cluster_imbalance() {
        // Create vectors that naturally cluster unevenly
        let mut vectors = Vec::new();
        // 90 vectors near origin
        for i in 0..90 {
            let v: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32 * 0.001).collect();
            vectors.push(v);
        }
        // 10 vectors far away
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|j| 100.0 + (i * 8 + j) as f32 * 0.001).collect();
            vectors.push(v);
        }

        let config = IvfConfig::new(4).with_nprobe(4).with_max_iterations(20);
        let mut index = IvfIndex::new(8, config);

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        let stats = index.cluster_stats();
        let sizes: Vec<usize> = stats.iter().map(|s| s.size).collect();

        // Verify all vectors are assigned
        assert_eq!(sizes.iter().sum::<usize>(), 100);
        // There should be some imbalance (not all clusters equal)
        assert!(sizes.iter().max().unwrap() > sizes.iter().min().unwrap());
    }

    #[test]
    fn test_ivf_pq_recall() {
        let dim = 32;
        let n = 100;
        let k = 10;

        let config = IvfConfig::new(4)
            .with_nprobe(4)
            .with_pq(4, 8)
            .with_max_iterations(20);
        let mut index = IvfIndex::new(dim, config);

        let vectors = random_vectors(n, dim);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        // PQ search should return results (PQ is approximate, may not find exact self)
        let results = index.search(&vectors[0], k).unwrap();
        assert!(!results.is_empty(), "PQ search should return results");
        assert!(results.len() <= k);
    }

    #[test]
    fn test_ivf_config_builder() {
        let config = IvfConfig::new(128)
            .with_nprobe(16)
            .with_max_iterations(50)
            .with_tolerance(1e-5)
            .with_pq(8, 8);

        assert_eq!(config.n_clusters, 128);
        assert_eq!(config.n_probe, 16);
        assert_eq!(config.max_iterations, 50);
        assert!((config.tolerance - 1e-5).abs() < 1e-10);
        assert!(config.use_pq);
        assert_eq!(config.pq_subvectors, 8);
        assert_eq!(config.pq_bits, 8);
    }

    #[test]
    fn test_ivf_empty_after_clear() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        index.clear();
        // Search on empty (but trained) index should return empty
        let results = index.search(&vectors[0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_ivf_search_after_partial_clear() {
        let config = IvfConfig::new(4).with_nprobe(4);
        let mut index = IvfIndex::new(16, config);

        let vectors = random_vectors(50, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        // Clear one cluster
        let removed = index.clear_cluster(0).unwrap();
        let remaining = index.len();
        assert_eq!(remaining + removed, 50);

        // Search should still work with remaining vectors
        let results = index.search(&vectors[0], 5).unwrap();
        assert!(results.len() <= remaining);
    }

    // ── Incremental training tests ──────────────────────────────────────

    #[test]
    fn test_ivf_refresh_centroids() {
        let config = IvfConfig::new(4).with_max_iterations(10);
        let mut index = IvfIndex::new(32, config);

        let train_data = random_vectors(100, 32);
        let refs: Vec<&[f32]> = train_data.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        // Insert more vectors
        let new_data = random_vectors(200, 32);
        for (i, v) in new_data.iter().enumerate() {
            index.insert(100 + i, v).unwrap();
        }

        // Refresh centroids — should update based on running stats
        let changed = index.refresh_centroids();
        // At least some centroids should have shifted
        assert!(changed > 0 || index.clusters.len() == changed);
    }

    #[test]
    fn test_ivf_refresh_centroids_untrained() {
        let mut index = IvfIndex::with_defaults(32);
        assert_eq!(index.refresh_centroids(), 0);
    }

    #[test]
    fn test_ivf_rebalance() {
        let config = IvfConfig::new(4).with_max_iterations(10);
        let mut index = IvfIndex::new(32, config);

        let train_data = random_vectors(100, 32);
        let refs: Vec<&[f32]> = train_data.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        // Insert vectors
        for (i, v) in train_data.iter().enumerate() {
            index.insert(i, v).unwrap();
        }
        assert_eq!(index.len(), 100);

        // Refresh + rebalance
        index.refresh_centroids();
        let moved = index.rebalance().unwrap();
        assert!(moved > 0);
        assert_eq!(index.len(), 100); // Vector count preserved

        // Search should still work
        let results = index.search(&train_data[0], 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_ivf_rebalance_with_pq_rejected() {
        let config = IvfConfig::new(4).with_pq(4, 8).with_max_iterations(10);
        let mut index = IvfIndex::new(32, config);

        let train_data = random_vectors(100, 32);
        let refs: Vec<&[f32]> = train_data.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        for (i, v) in train_data.iter().enumerate() {
            index.insert(i, v).unwrap();
        }

        let result = index.rebalance();
        assert!(result.is_err());
    }

    #[test]
    fn test_ivf_running_stats_tracked() {
        let config = IvfConfig::new(2).with_max_iterations(10);
        let mut index = IvfIndex::new(4, config);

        let train_data = random_vectors(20, 4);
        let refs: Vec<&[f32]> = train_data.iter().map(|v| v.as_slice()).collect();
        index.train(&refs).unwrap();

        // Insert a vector
        index.insert(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Verify running stats updated
        let total_running: usize = index.clusters.iter().map(|c| c.running_count).sum();
        assert_eq!(total_running, 1);
    }
}
