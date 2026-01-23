//! Edge Data Partitioning
//!
//! Provides intelligent data partitioning strategies for edge environments
//! with limited memory and storage. Enables efficient loading of only the
//! relevant portions of the index for a given query.
//!
//! # Strategies
//!
//! - **Cluster-Based**: Partition by k-means clusters for locality-aware loading
//! - **Geographic**: Partition by geographic region for geo-distributed data
//! - **Hash-Based**: Simple consistent hashing for uniform distribution
//! - **Hierarchical**: Multi-level partitioning for very large datasets
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::edge_partitioning::{PartitionManager, ClusterPartitioner, PartitionConfig};
//!
//! // Create cluster-based partitioner
//! let config = PartitionConfig::for_edge_memory(128 * 1024 * 1024); // 128MB
//! let mut partitioner = ClusterPartitioner::new(config);
//!
//! // Train on vectors
//! partitioner.train(&vectors)?;
//!
//! // Get partition for a query
//! let partition_ids = partitioner.route_query(&query_vector);
//!
//! // Load only relevant partitions
//! for pid in partition_ids {
//!     runtime.load_partition(pid, storage)?;
//! }
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Partition Configuration
// ============================================================================

/// Configuration for edge partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Maximum vectors per partition
    pub max_vectors_per_partition: usize,
    /// Target number of partitions
    pub target_partitions: usize,
    /// Number of partitions to probe during search
    pub search_probe_count: usize,
    /// Enable partition preloading hints
    pub enable_preload_hints: bool,
    /// Overlap factor for partition boundaries
    pub boundary_overlap: f32,
    /// Maximum partition size in bytes
    pub max_partition_size_bytes: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_partition: 10_000,
            target_partitions: 16,
            search_probe_count: 3,
            enable_preload_hints: true,
            boundary_overlap: 0.1,
            max_partition_size_bytes: 10 * 1024 * 1024, // 10 MB
        }
    }
}

impl PartitionConfig {
    /// Create config optimized for given memory budget
    pub fn for_edge_memory(memory_bytes: usize) -> Self {
        // Assume ~2KB per vector (384 dims * 4 bytes + overhead)
        let vectors_per_mb = 1024 * 1024 / 2048;
        let max_vectors = (memory_bytes / 1024 / 1024) * vectors_per_mb / 2;

        Self {
            max_vectors_per_partition: max_vectors.min(20_000),
            target_partitions: ((memory_bytes / 1024 / 1024) / 10).max(4).min(64) as usize,
            search_probe_count: 3,
            enable_preload_hints: true,
            boundary_overlap: 0.1,
            max_partition_size_bytes: memory_bytes / 4,
        }
    }

    /// Builder: set max vectors per partition
    pub fn with_max_vectors(mut self, count: usize) -> Self {
        self.max_vectors_per_partition = count;
        self
    }

    /// Builder: set target partition count
    pub fn with_target_partitions(mut self, count: usize) -> Self {
        self.target_partitions = count;
        self
    }

    /// Builder: set search probe count
    pub fn with_probe_count(mut self, count: usize) -> Self {
        self.search_probe_count = count;
        self
    }
}

// ============================================================================
// Partition Metadata
// ============================================================================

/// Unique partition identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartitionId(pub u32);

impl PartitionId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

impl std::fmt::Display for PartitionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "partition_{}", self.0)
    }
}

/// Metadata for a single partition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionMetadata {
    /// Partition ID
    pub id: PartitionId,
    /// Number of vectors in partition
    pub vector_count: usize,
    /// Centroid of partition (for cluster-based routing)
    pub centroid: Option<Vec<f32>>,
    /// Bounding box min (for range-based routing)
    pub bounds_min: Option<Vec<f32>>,
    /// Bounding box max
    pub bounds_max: Option<Vec<f32>>,
    /// Partition size in bytes
    pub size_bytes: usize,
    /// Storage key for partition data
    pub storage_key: String,
    /// Related partitions for multi-probe
    pub related_partitions: Vec<PartitionId>,
    /// Access frequency (for preloading hints)
    pub access_count: u64,
}

impl PartitionMetadata {
    /// Create new partition metadata
    pub fn new(id: PartitionId) -> Self {
        Self {
            id,
            vector_count: 0,
            centroid: None,
            bounds_min: None,
            bounds_max: None,
            size_bytes: 0,
            storage_key: id.to_string(),
            related_partitions: Vec::new(),
            access_count: 0,
        }
    }

    /// Calculate distance from query to partition centroid
    pub fn distance_to_query(&self, query: &[f32]) -> f32 {
        match &self.centroid {
            Some(centroid) => euclidean_distance(query, centroid),
            None => f32::INFINITY,
        }
    }

    /// Check if query might be in partition bounds
    pub fn might_contain(&self, query: &[f32]) -> bool {
        match (&self.bounds_min, &self.bounds_max) {
            (Some(min), Some(max)) => {
                query.iter().enumerate().all(|(i, &v)| {
                    v >= min.get(i).copied().unwrap_or(f32::NEG_INFINITY)
                        && v <= max.get(i).copied().unwrap_or(f32::INFINITY)
                })
            }
            _ => true, // No bounds, assume might contain
        }
    }
}

/// Complete partition manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManifest {
    /// Collection name
    pub collection_name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Total vector count across all partitions
    pub total_vectors: usize,
    /// Partition metadata
    pub partitions: Vec<PartitionMetadata>,
    /// Partitioning strategy used
    pub strategy: PartitionStrategy,
    /// Configuration used
    pub config: PartitionConfig,
    /// Creation timestamp
    pub created_at: u64,
    /// Global entry points for search
    pub global_entry_points: Vec<(PartitionId, usize)>,
}

impl PartitionManifest {
    /// Create a new manifest
    pub fn new(collection_name: &str, dimensions: usize, strategy: PartitionStrategy) -> Self {
        Self {
            collection_name: collection_name.to_string(),
            dimensions,
            total_vectors: 0,
            partitions: Vec::new(),
            strategy,
            config: PartitionConfig::default(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            global_entry_points: Vec::new(),
        }
    }

    /// Add a partition
    pub fn add_partition(&mut self, partition: PartitionMetadata) {
        self.total_vectors += partition.vector_count;
        self.partitions.push(partition);
    }

    /// Get partition by ID
    pub fn get_partition(&self, id: PartitionId) -> Option<&PartitionMetadata> {
        self.partitions.iter().find(|p| p.id == id)
    }

    /// Find closest partitions to a query
    pub fn find_closest_partitions(&self, query: &[f32], count: usize) -> Vec<PartitionId> {
        let mut distances: Vec<(PartitionId, f32)> = self.partitions
            .iter()
            .map(|p| (p.id, p.distance_to_query(query)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances.into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(NeedleError::Serialization)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(NeedleError::Serialization)
    }
}

/// Partitioning strategy identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// K-means cluster-based partitioning
    Cluster,
    /// Consistent hash-based partitioning
    Hash,
    /// Geographic/spatial partitioning
    Geographic,
    /// Hierarchical multi-level partitioning
    Hierarchical,
    /// Simple round-robin distribution
    RoundRobin,
}

// ============================================================================
// Partition Routing
// ============================================================================

/// Routes queries to appropriate partitions
pub trait PartitionRouter: Send + Sync {
    /// Route a query to relevant partitions
    fn route(&self, query: &[f32], max_partitions: usize) -> Vec<PartitionId>;

    /// Assign a vector to a partition
    fn assign(&self, vector: &[f32]) -> PartitionId;

    /// Get all partition IDs
    fn all_partitions(&self) -> Vec<PartitionId>;

    /// Get partition metadata
    fn get_metadata(&self, id: PartitionId) -> Option<&PartitionMetadata>;
}

// ============================================================================
// Cluster-Based Partitioner
// ============================================================================

/// K-means cluster-based partitioner
///
/// Partitions vectors into clusters based on similarity. Queries are
/// routed to the nearest cluster centroids.
pub struct ClusterPartitioner {
    config: PartitionConfig,
    /// Cluster centroids
    centroids: Vec<Vec<f32>>,
    /// Partition metadata
    partitions: Vec<PartitionMetadata>,
    /// Trained flag
    trained: bool,
}

impl ClusterPartitioner {
    /// Create a new cluster partitioner
    pub fn new(config: PartitionConfig) -> Self {
        Self {
            config,
            centroids: Vec::new(),
            partitions: Vec::new(),
            trained: false,
        }
    }

    /// Train the partitioner on a set of vectors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput("No vectors to train on".into()));
        }

        let k = self.config.target_partitions.min(vectors.len());

        // Simple k-means++ initialization
        self.centroids = Self::kmeans_plusplus_init(vectors, k);

        // Run k-means iterations
        for _ in 0..20 {
            let assignments = self.assign_to_clusters(vectors);
            self.update_centroids(vectors, &assignments);
        }

        // Create partition metadata
        self.partitions = self.centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| {
                let mut meta = PartitionMetadata::new(PartitionId::new(i as u32));
                meta.centroid = Some(centroid.clone());
                meta
            })
            .collect();

        // Find related partitions (nearest neighbors)
        self.compute_related_partitions();

        self.trained = true;
        Ok(())
    }

    /// K-means++ initialization
    fn kmeans_plusplus_init(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut centroids = Vec::with_capacity(k);

        // First centroid: random
        centroids.push(vectors[rng.gen_range(0..vectors.len())].clone());

        // Remaining centroids: weighted by distance
        for _ in 1..k {
            let distances: Vec<f32> = vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| euclidean_distance(v, c))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap_or(f32::INFINITY)
                })
                .collect();

            let total: f32 = distances.iter().sum();
            if total <= 0.0 {
                break;
            }

            let threshold: f32 = rng.gen::<f32>() * total;
            let mut cumulative = 0.0;

            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist;
                if cumulative >= threshold {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }
        }

        centroids
    }

    /// Assign vectors to nearest clusters
    fn assign_to_clusters(&self, vectors: &[Vec<f32>]) -> Vec<usize> {
        vectors
            .iter()
            .map(|v| {
                self.centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        euclidean_distance(v, a)
                            .partial_cmp(&euclidean_distance(v, b))
                            .unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Update centroids based on assignments
    fn update_centroids(&mut self, vectors: &[Vec<f32>], assignments: &[usize]) {
        let dims = vectors.first().map(|v| v.len()).unwrap_or(0);

        for (i, centroid) in self.centroids.iter_mut().enumerate() {
            let cluster_vectors: Vec<&Vec<f32>> = vectors
                .iter()
                .zip(assignments.iter())
                .filter(|(_, &a)| a == i)
                .map(|(v, _)| v)
                .collect();

            if !cluster_vectors.is_empty() {
                *centroid = vec![0.0; dims];
                for v in &cluster_vectors {
                    for (j, val) in v.iter().enumerate() {
                        centroid[j] += val;
                    }
                }
                for val in centroid.iter_mut() {
                    *val /= cluster_vectors.len() as f32;
                }
            }
        }
    }

    /// Compute related partitions for multi-probe
    fn compute_related_partitions(&mut self) {
        for i in 0..self.partitions.len() {
            let centroid = &self.centroids[i];
            let mut distances: Vec<(usize, f32)> = self.centroids
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, c)| (j, euclidean_distance(centroid, c)))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            self.partitions[i].related_partitions = distances
                .iter()
                .take(self.config.search_probe_count - 1)
                .map(|(j, _)| PartitionId::new(*j as u32))
                .collect();
        }
    }
}

impl PartitionRouter for ClusterPartitioner {
    fn route(&self, query: &[f32], max_partitions: usize) -> Vec<PartitionId> {
        let mut distances: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, euclidean_distance(query, c)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        distances
            .into_iter()
            .take(max_partitions)
            .map(|(i, _)| PartitionId::new(i as u32))
            .collect()
    }

    fn assign(&self, vector: &[f32]) -> PartitionId {
        let best = self.centroids
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                euclidean_distance(vector, a)
                    .partial_cmp(&euclidean_distance(vector, b))
                    .unwrap()
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        PartitionId::new(best as u32)
    }

    fn all_partitions(&self) -> Vec<PartitionId> {
        (0..self.partitions.len())
            .map(|i| PartitionId::new(i as u32))
            .collect()
    }

    fn get_metadata(&self, id: PartitionId) -> Option<&PartitionMetadata> {
        self.partitions.get(id.0 as usize)
    }
}

// ============================================================================
// Hash-Based Partitioner
// ============================================================================

/// Consistent hash-based partitioner
///
/// Uses consistent hashing for uniform distribution. Useful when
/// vectors don't have clear clusters or for ID-based lookups.
pub struct HashPartitioner {
    config: PartitionConfig,
    partitions: Vec<PartitionMetadata>,
    /// Hash ring for consistent hashing
    ring: Vec<(u64, PartitionId)>,
}

impl HashPartitioner {
    /// Create a new hash partitioner
    pub fn new(config: PartitionConfig) -> Self {
        let mut partitioner = Self {
            config: config.clone(),
            partitions: Vec::new(),
            ring: Vec::new(),
        };

        partitioner.initialize_partitions();
        partitioner
    }

    fn initialize_partitions(&mut self) {
        // Create partitions
        for i in 0..self.config.target_partitions {
            self.partitions.push(PartitionMetadata::new(PartitionId::new(i as u32)));
        }

        // Build hash ring with virtual nodes
        let virtual_nodes_per_partition = 100;
        for (i, _) in self.partitions.iter().enumerate() {
            for vn in 0..virtual_nodes_per_partition {
                let hash = Self::hash_string(&format!("partition_{}_{}", i, vn));
                self.ring.push((hash, PartitionId::new(i as u32)));
            }
        }

        self.ring.sort_by_key(|(h, _)| *h);
    }

    fn hash_string(s: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    fn hash_vector(v: &[f32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for val in v {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn find_partition(&self, hash: u64) -> PartitionId {
        // Binary search in ring
        let pos = self.ring.binary_search_by_key(&hash, |(h, _)| *h);
        let index = match pos {
            Ok(i) => i,
            Err(i) => i % self.ring.len(),
        };
        self.ring[index].1
    }
}

impl PartitionRouter for HashPartitioner {
    fn route(&self, query: &[f32], max_partitions: usize) -> Vec<PartitionId> {
        let hash = Self::hash_vector(query);
        let primary = self.find_partition(hash);

        // For hash-based, probe neighbors in ring
        let mut result = vec![primary];
        let pos = self.ring.iter().position(|(_, id)| *id == primary).unwrap_or(0);

        for i in 1..max_partitions {
            let next_pos = (pos + i * 100) % self.ring.len();
            let next_id = self.ring[next_pos].1;
            if !result.contains(&next_id) {
                result.push(next_id);
            }
            if result.len() >= max_partitions {
                break;
            }
        }

        result
    }

    fn assign(&self, vector: &[f32]) -> PartitionId {
        let hash = Self::hash_vector(vector);
        self.find_partition(hash)
    }

    fn all_partitions(&self) -> Vec<PartitionId> {
        (0..self.partitions.len())
            .map(|i| PartitionId::new(i as u32))
            .collect()
    }

    fn get_metadata(&self, id: PartitionId) -> Option<&PartitionMetadata> {
        self.partitions.get(id.0 as usize)
    }
}

// ============================================================================
// Hierarchical Partitioner
// ============================================================================

/// Hierarchical tree-based partitioner
///
/// Builds a tree of partitions for very large datasets. Each level
/// of the tree provides more granular partitioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalNode {
    /// Node ID
    pub id: u32,
    /// Level in hierarchy (0 = root)
    pub level: u8,
    /// Centroid of this node
    pub centroid: Vec<f32>,
    /// Child node IDs (empty for leaf nodes)
    pub children: Vec<u32>,
    /// Partition ID (only for leaf nodes)
    pub partition_id: Option<PartitionId>,
}

pub struct HierarchicalPartitioner {
    config: PartitionConfig,
    /// Tree nodes
    nodes: Vec<HierarchicalNode>,
    /// Root node ID
    root_id: u32,
    /// Leaf partitions
    partitions: Vec<PartitionMetadata>,
    /// Max tree depth
    max_depth: u8,
}

impl HierarchicalPartitioner {
    /// Create a new hierarchical partitioner
    pub fn new(config: PartitionConfig, max_depth: u8) -> Self {
        Self {
            config,
            nodes: Vec::new(),
            root_id: 0,
            partitions: Vec::new(),
            max_depth,
        }
    }

    /// Build tree from vectors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput("No vectors to train on".into()));
        }

        // Calculate global centroid for root
        let dims = vectors[0].len();
        let mut root_centroid = vec![0.0; dims];
        for v in vectors {
            for (i, val) in v.iter().enumerate() {
                root_centroid[i] += val;
            }
        }
        for val in root_centroid.iter_mut() {
            *val /= vectors.len() as f32;
        }

        // Build tree recursively
        let indices: Vec<usize> = (0..vectors.len()).collect();
        self.build_tree(vectors, &indices, root_centroid, 0);

        // Create partition metadata from leaf nodes
        self.partitions = self.nodes
            .iter()
            .filter(|n| n.partition_id.is_some())
            .map(|n| {
                let mut meta = PartitionMetadata::new(n.partition_id.unwrap());
                meta.centroid = Some(n.centroid.clone());
                meta
            })
            .collect();

        Ok(())
    }

    fn build_tree(
        &mut self,
        vectors: &[Vec<f32>],
        indices: &[usize],
        centroid: Vec<f32>,
        depth: u8,
    ) -> u32 {
        let node_id = self.nodes.len() as u32;

        // Check if we should create a leaf
        if depth >= self.max_depth || indices.len() <= self.config.max_vectors_per_partition {
            let partition_id = PartitionId::new(self.partitions.len() as u32);
            self.nodes.push(HierarchicalNode {
                id: node_id,
                level: depth,
                centroid,
                children: Vec::new(),
                partition_id: Some(partition_id),
            });
            return node_id;
        }

        // Split into children (binary split along highest variance dimension)
        let split_dim = self.find_split_dimension(vectors, indices);
        let (left_indices, right_indices) = self.split_by_dimension(vectors, indices, split_dim);

        // Calculate child centroids
        let left_centroid = self.compute_centroid(vectors, &left_indices);
        let right_centroid = self.compute_centroid(vectors, &right_indices);

        // Placeholder node
        self.nodes.push(HierarchicalNode {
            id: node_id,
            level: depth,
            centroid,
            children: Vec::new(),
            partition_id: None,
        });

        // Build children
        let left_id = self.build_tree(vectors, &left_indices, left_centroid, depth + 1);
        let right_id = self.build_tree(vectors, &right_indices, right_centroid, depth + 1);

        // Update children
        self.nodes[node_id as usize].children = vec![left_id, right_id];

        node_id
    }

    fn find_split_dimension(&self, vectors: &[Vec<f32>], indices: &[usize]) -> usize {
        if indices.is_empty() {
            return 0;
        }

        let dims = vectors[0].len();
        let mut max_variance = 0.0;
        let mut split_dim = 0;

        for d in 0..dims {
            let values: Vec<f32> = indices.iter().map(|&i| vectors[i][d]).collect();
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
            let variance: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

            if variance > max_variance {
                max_variance = variance;
                split_dim = d;
            }
        }

        split_dim
    }

    fn split_by_dimension(
        &self,
        vectors: &[Vec<f32>],
        indices: &[usize],
        dim: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            vectors[a][dim].partial_cmp(&vectors[b][dim]).unwrap()
        });

        let mid = sorted.len() / 2;
        (sorted[..mid].to_vec(), sorted[mid..].to_vec())
    }

    fn compute_centroid(&self, vectors: &[Vec<f32>], indices: &[usize]) -> Vec<f32> {
        if indices.is_empty() {
            return Vec::new();
        }

        let dims = vectors[0].len();
        let mut centroid = vec![0.0; dims];

        for &i in indices {
            for (j, val) in vectors[i].iter().enumerate() {
                centroid[j] += val;
            }
        }

        for val in centroid.iter_mut() {
            *val /= indices.len() as f32;
        }

        centroid
    }

    /// Find leaf nodes for a query by traversing tree
    fn traverse(&self, query: &[f32], node_id: u32, results: &mut Vec<PartitionId>, max: usize) {
        if results.len() >= max {
            return;
        }

        let node = &self.nodes[node_id as usize];

        if let Some(partition_id) = node.partition_id {
            results.push(partition_id);
            return;
        }

        // Sort children by distance to query
        let mut child_distances: Vec<(u32, f32)> = node.children
            .iter()
            .map(|&child_id| {
                let child = &self.nodes[child_id as usize];
                (child_id, euclidean_distance(query, &child.centroid))
            })
            .collect();

        child_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Traverse closest children first
        for (child_id, _) in child_distances {
            self.traverse(query, child_id, results, max);
        }
    }
}

impl PartitionRouter for HierarchicalPartitioner {
    fn route(&self, query: &[f32], max_partitions: usize) -> Vec<PartitionId> {
        let mut results = Vec::new();
        self.traverse(query, self.root_id, &mut results, max_partitions);
        results
    }

    fn assign(&self, vector: &[f32]) -> PartitionId {
        let mut current = self.root_id;

        loop {
            let node = &self.nodes[current as usize];

            if let Some(partition_id) = node.partition_id {
                return partition_id;
            }

            // Find closest child
            current = node.children
                .iter()
                .min_by(|&&a, &&b| {
                    let dist_a = euclidean_distance(vector, &self.nodes[a as usize].centroid);
                    let dist_b = euclidean_distance(vector, &self.nodes[b as usize].centroid);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .copied()
                .unwrap_or(current);
        }
    }

    fn all_partitions(&self) -> Vec<PartitionId> {
        self.nodes
            .iter()
            .filter_map(|n| n.partition_id)
            .collect()
    }

    fn get_metadata(&self, id: PartitionId) -> Option<&PartitionMetadata> {
        self.partitions.iter().find(|p| p.id == id)
    }
}

// ============================================================================
// Partition Manager
// ============================================================================

/// Manages partitioned data for edge runtime
pub struct PartitionManager<R: PartitionRouter> {
    router: R,
    config: PartitionConfig,
    /// Currently loaded partitions
    loaded: HashSet<PartitionId>,
    /// Partition data cache
    cache: HashMap<PartitionId, Vec<u8>>,
    /// Access statistics
    access_stats: HashMap<PartitionId, u64>,
}

impl<R: PartitionRouter> PartitionManager<R> {
    /// Create a new partition manager
    pub fn new(router: R, config: PartitionConfig) -> Self {
        Self {
            router,
            config,
            loaded: HashSet::new(),
            cache: HashMap::new(),
            access_stats: HashMap::new(),
        }
    }

    /// Route a query to partitions
    pub fn route_query(&self, query: &[f32]) -> Vec<PartitionId> {
        self.router.route(query, self.config.search_probe_count)
    }

    /// Assign a vector to a partition
    pub fn assign_vector(&self, vector: &[f32]) -> PartitionId {
        self.router.assign(vector)
    }

    /// Check if partition is loaded
    pub fn is_loaded(&self, partition_id: PartitionId) -> bool {
        self.loaded.contains(&partition_id)
    }

    /// Mark partition as loaded
    pub fn mark_loaded(&mut self, partition_id: PartitionId, data: Vec<u8>) {
        self.loaded.insert(partition_id);
        self.cache.insert(partition_id, data);
    }

    /// Evict partition from cache
    pub fn evict(&mut self, partition_id: PartitionId) {
        self.loaded.remove(&partition_id);
        self.cache.remove(&partition_id);
    }

    /// Record access for statistics
    pub fn record_access(&mut self, partition_id: PartitionId) {
        *self.access_stats.entry(partition_id).or_insert(0) += 1;
    }

    /// Get partition data from cache
    pub fn get_cached(&self, partition_id: PartitionId) -> Option<&[u8]> {
        self.cache.get(&partition_id).map(|v| v.as_slice())
    }

    /// Get partitions to preload based on access patterns
    pub fn preload_hints(&self) -> Vec<PartitionId> {
        let mut sorted: Vec<_> = self.access_stats.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));

        sorted
            .into_iter()
            .take(self.config.search_probe_count)
            .map(|(id, _)| *id)
            .filter(|id| !self.loaded.contains(id))
            .collect()
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        self.cache.values().map(|v| v.len()).sum()
    }

    /// Get loaded partition count
    pub fn loaded_count(&self) -> usize {
        self.loaded.len()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(count: usize, dims: usize) -> Vec<Vec<f32>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..count)
            .map(|_| (0..dims).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_partition_config() {
        let config = PartitionConfig::for_edge_memory(128 * 1024 * 1024);
        assert!(config.max_vectors_per_partition > 0);
        assert!(config.target_partitions > 0);
    }

    #[test]
    fn test_cluster_partitioner() {
        let config = PartitionConfig::default()
            .with_target_partitions(4);

        let mut partitioner = ClusterPartitioner::new(config);

        let vectors = random_vectors(100, 64);
        partitioner.train(&vectors).unwrap();

        // Route a query
        let query = &vectors[0];
        let partitions = partitioner.route(query, 2);
        assert!(!partitions.is_empty());
        assert!(partitions.len() <= 2);

        // Assign vectors
        let assigned = partitioner.assign(&vectors[50]);
        assert!(assigned.0 < 4);
    }

    #[test]
    fn test_hash_partitioner() {
        let config = PartitionConfig::default()
            .with_target_partitions(8);

        let partitioner = HashPartitioner::new(config);

        let vectors = random_vectors(100, 64);

        // All vectors should be assignable
        for v in &vectors {
            let partition = partitioner.assign(v);
            assert!(partition.0 < 8);
        }

        // Same vector should always go to same partition
        let v = &vectors[0];
        let p1 = partitioner.assign(v);
        let p2 = partitioner.assign(v);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_hierarchical_partitioner() {
        let config = PartitionConfig::default()
            .with_max_vectors(20);

        let mut partitioner = HierarchicalPartitioner::new(config, 3);

        let vectors = random_vectors(100, 32);
        partitioner.train(&vectors).unwrap();

        // Should have created leaf partitions
        assert!(!partitioner.all_partitions().is_empty());

        // Route queries
        let query = &vectors[0];
        let partitions = partitioner.route(query, 3);
        assert!(!partitions.is_empty());
    }

    #[test]
    fn test_partition_manager() {
        let config = PartitionConfig::default().with_target_partitions(4);
        let partitioner = HashPartitioner::new(config.clone());
        let mut manager = PartitionManager::new(partitioner, config);

        let query = vec![0.1, 0.2, 0.3];
        let partitions = manager.route_query(&query);
        assert!(!partitions.is_empty());

        // Mark partition as loaded
        manager.mark_loaded(partitions[0], vec![1, 2, 3]);
        assert!(manager.is_loaded(partitions[0]));
        assert_eq!(manager.loaded_count(), 1);

        // Record access
        manager.record_access(partitions[0]);
        assert_eq!(manager.access_stats.get(&partitions[0]), Some(&1));
    }

    #[test]
    fn test_partition_manifest() {
        let mut manifest = PartitionManifest::new("test", 64, PartitionStrategy::Cluster);

        let mut partition = PartitionMetadata::new(PartitionId::new(0));
        partition.vector_count = 100;
        partition.centroid = Some(vec![0.0; 64]);

        manifest.add_partition(partition);
        assert_eq!(manifest.total_vectors, 100);
        assert_eq!(manifest.partitions.len(), 1);

        // Serialize and deserialize
        let json = manifest.to_json().unwrap();
        let restored = PartitionManifest::from_json(&json).unwrap();
        assert_eq!(restored.total_vectors, 100);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }
}
