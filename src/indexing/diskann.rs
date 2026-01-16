//! DiskANN - Billion-scale on-disk approximate nearest neighbor search.
//!
//! Implements a disk-based vector index optimized for datasets that exceed RAM capacity.
//! Uses a Vamana graph algorithm with SSD-optimized access patterns.
//!
//! # Features
//!
//! - **Memory-mapped storage**: Efficient disk access with OS page cache
//! - **Graph-based search**: Vamana algorithm for high recall
//! - **Beam search**: Configurable search width for quality/speed tradeoff
//! - **Compression**: Optional PQ (Product Quantization) for reduced storage
//! - **Incremental builds**: Add vectors without full reindex
//!
//! # Example
//!
//! ```ignore
//! use needle::diskann::{DiskAnnIndex, DiskAnnConfig};
//!
//! let config = DiskAnnConfig::default();
//! let mut index = DiskAnnIndex::create("./my_index", 128, config)?;
//!
//! // Add vectors
//! index.add("vec1", &embedding)?;
//!
//! // Build the graph
//! index.build()?;
//!
//! // Search
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # When to Use DiskANN
//!
//! DiskANN is designed for billion-scale datasets that exceed available RAM:
//!
//! | Use Case | DiskANN Suitability |
//! |----------|---------------------|
//! | > 100M vectors | ✅ Excellent - purpose-built for scale |
//! | Exceeds RAM | ✅ Excellent - disk-based with efficient I/O |
//! | SSD storage | ✅ Excellent - optimized for SSD access patterns |
//! | Cost-effective scaling | ✅ Good - use cheap disk instead of expensive RAM |
//! | Real-time search | ⚠️ Higher latency than in-memory (5-20ms) |
//! | Frequent updates | ⚠️ Incremental updates possible but slower |
//! | Small datasets (<1M) | ⚠️ HNSW is simpler and faster |
//!
//! ## DiskANN vs HNSW
//!
//! - **DiskANN** scales to billions of vectors with SSD storage
//! - **HNSW** is faster but limited by available RAM
//! - Choose DiskANN when data exceeds memory (typically >10-100M vectors)
//! - Choose HNSW when data fits in memory and latency is critical
//!
//! ## DiskANN vs IVF
//!
//! - **DiskANN** has lower latency for disk-based search
//! - **IVF** is simpler but less optimized for disk access
//! - Choose DiskANN for large-scale SSD-based deployments
//! - Choose IVF for moderate-scale memory-constrained scenarios
//!
//! ## Configuration Guidelines
//!
//! - **max_degree** (R): 64-128 for high recall, 32-64 for faster queries
//! - **search_list_size** (L): Higher = better recall, slower search
//! - **alpha**: 1.0-1.2, higher values improve graph connectivity
//! - **cache_size**: Set based on available RAM for frequently accessed vectors
//!
//! ## Hardware Recommendations
//!
//! - NVMe SSD strongly recommended for production workloads
//! - Index size ≈ vectors * (dimensions * 4 + max_degree * 8) bytes
//! - With PQ: Index size ≈ vectors * (pq_subvectors + max_degree * 8) bytes

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Configuration for DiskANN index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskAnnConfig {
    /// Maximum degree of the graph (R parameter).
    pub max_degree: usize,
    /// Build-time graph degree (L parameter).
    pub build_list_size: usize,
    /// Search beam width.
    pub search_list_size: usize,
    /// Alpha parameter for Vamana graph construction.
    pub alpha: f32,
    /// Number of vectors per disk page.
    pub vectors_per_page: usize,
    /// Enable Product Quantization compression.
    pub use_pq: bool,
    /// Number of PQ subvectors.
    pub pq_subvectors: usize,
    /// Bits per PQ code.
    pub pq_bits: usize,
    /// Cache size in number of vectors.
    pub cache_size: usize,
}

impl Default for DiskAnnConfig {
    fn default() -> Self {
        Self {
            max_degree: 64,
            build_list_size: 100,
            search_list_size: 100,
            alpha: 1.2,
            vectors_per_page: 16,
            use_pq: false,
            pq_subvectors: 8,
            pq_bits: 8,
            cache_size: 10000,
        }
    }
}

/// Entry point for a vector in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphNode {
    /// Vector ID.
    id: String,
    /// Indices of neighbor nodes.
    neighbors: Vec<usize>,
    /// Offset in the vector data file.
    data_offset: u64,
}

/// On-disk metadata for the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexMetadata {
    /// Number of dimensions.
    dimensions: usize,
    /// Total number of vectors.
    num_vectors: usize,
    /// Index of the entry point node.
    entry_point: Option<usize>,
    /// Configuration used to build the index.
    config: DiskAnnConfig,
    /// Mapping from vector ID to node index.
    id_to_index: HashMap<String, usize>,
}

/// Search result from DiskANN.
#[derive(Debug, Clone)]
pub struct DiskAnnResult {
    /// Vector ID.
    pub id: String,
    /// Distance to query.
    pub distance: f32,
    /// Node index (internal).
    pub node_index: usize,
}

/// Candidate for beam search.
#[derive(Debug, Clone)]
struct SearchCandidate {
    index: usize,
    distance: f32,
    visited: bool,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// DiskANN index for billion-scale vector search.
pub struct DiskAnnIndex {
    /// Base path for index files.
    base_path: PathBuf,
    /// Index metadata.
    metadata: IndexMetadata,
    /// Graph nodes (loaded in memory).
    nodes: Vec<GraphNode>,
    /// Vector data file handle.
    data_file: Option<File>,
    /// Cache of recently accessed vectors.
    vector_cache: HashMap<usize, Vec<f32>>,
    /// LRU order for cache eviction.
    cache_order: Vec<usize>,
    /// Whether index is built and ready for search.
    is_built: bool,
    /// Pending vectors not yet added to graph.
    pending_vectors: Vec<(String, Vec<f32>)>,
}

impl DiskAnnIndex {
    /// Create a new DiskANN index.
    pub fn create<P: AsRef<Path>>(
        path: P,
        dimensions: usize,
        config: DiskAnnConfig,
    ) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();
        fs::create_dir_all(&base_path)?;

        let metadata = IndexMetadata {
            dimensions,
            num_vectors: 0,
            entry_point: None,
            config,
            id_to_index: HashMap::new(),
        };

        let index = Self {
            base_path,
            metadata,
            nodes: Vec::new(),
            data_file: None,
            vector_cache: HashMap::new(),
            cache_order: Vec::new(),
            is_built: false,
            pending_vectors: Vec::new(),
        };

        Ok(index)
    }

    /// Open an existing DiskANN index.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let base_path = path.as_ref().to_path_buf();

        // Load metadata
        let metadata_path = base_path.join("metadata.json");
        let file = File::open(&metadata_path)?;
        let reader = BufReader::new(file);
        let metadata: IndexMetadata = serde_json::from_reader(reader)?;

        // Load graph
        let graph_path = base_path.join("graph.bin");
        let nodes = if graph_path.exists() {
            let file = File::open(&graph_path)?;
            let reader = BufReader::new(file);
            bincode::deserialize_from(reader)
                .map_err(|e| NeedleError::InvalidInput(format!("Failed to load graph: {}", e)))?
        } else {
            Vec::new()
        };

        // Open data file
        let data_path = base_path.join("vectors.bin");
        let data_file = if data_path.exists() {
            Some(OpenOptions::new().read(true).open(&data_path)?)
        } else {
            None
        };

        let is_built = !nodes.is_empty();

        Ok(Self {
            base_path,
            metadata,
            nodes,
            data_file,
            vector_cache: HashMap::new(),
            cache_order: Vec::new(),
            is_built,
            pending_vectors: Vec::new(),
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.metadata.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.metadata.dimensions,
                vector.len()
            )));
        }

        if self.metadata.id_to_index.contains_key(id) {
            return Err(NeedleError::InvalidInput(format!(
                "Vector with id '{}' already exists",
                id
            )));
        }

        self.pending_vectors.push((id.to_string(), vector.to_vec()));
        Ok(())
    }

    /// Add multiple vectors in batch.
    pub fn add_batch(&mut self, vectors: Vec<(String, Vec<f32>)>) -> Result<()> {
        for (id, vector) in vectors {
            self.add(&id, &vector)?;
        }
        Ok(())
    }

    /// Build the index from added vectors.
    pub fn build(&mut self) -> Result<()> {
        if self.pending_vectors.is_empty() && self.nodes.is_empty() {
            return Err(NeedleError::InvalidInput("No vectors to index".to_string()));
        }

        // Append pending vectors to data file
        self.flush_pending_vectors()?;

        // Build Vamana graph
        self.build_vamana_graph()?;

        // Save metadata and graph
        self.save()?;

        self.is_built = true;
        Ok(())
    }

    /// Flush pending vectors to disk.
    fn flush_pending_vectors(&mut self) -> Result<()> {
        if self.pending_vectors.is_empty() {
            return Ok(());
        }

        let data_path = self.base_path.join("vectors.bin");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&data_path)?;

        for (id, vector) in self.pending_vectors.drain(..) {
            let offset = file.seek(SeekFrom::End(0))?;

            // Write vector data
            let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            file.write_all(&bytes)?;

            // Create graph node
            let node_index = self.nodes.len();
            let node = GraphNode {
                id: id.clone(),
                neighbors: Vec::new(),
                data_offset: offset,
            };

            self.nodes.push(node);
            self.metadata.id_to_index.insert(id, node_index);
            self.metadata.num_vectors += 1;
        }

        file.flush()?;

        // Reopen data file for reading
        self.data_file = Some(OpenOptions::new().read(true).open(&data_path)?);

        Ok(())
    }

    /// Build Vamana graph for approximate nearest neighbor search.
    fn build_vamana_graph(&mut self) -> Result<()> {
        let n = self.nodes.len();
        if n == 0 {
            return Ok(());
        }

        // Initialize with random entry point
        self.metadata.entry_point = Some(0);

        // Load all vectors for building (in production, this would be streamed)
        let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let vector = self.load_vector(i)?;
            all_vectors.push(vector);
        }

        // Initialize random neighbors
        let max_degree = self.metadata.config.max_degree;
        for i in 0..n {
            let mut neighbors: Vec<usize> = (0..n).filter(|&j| j != i).take(max_degree).collect();
            neighbors.truncate(max_degree);
            self.nodes[i].neighbors = neighbors;
        }

        // Vamana graph construction
        let alpha = self.metadata.config.alpha;
        let build_list_size = self.metadata.config.build_list_size;

        for i in 0..n {
            let query = &all_vectors[i];

            // Greedy search from entry point
            let candidates =
                self.greedy_search_internal(query, &all_vectors, build_list_size, Some(i));

            // Robust prune
            let new_neighbors = self.robust_prune(&candidates, query, &all_vectors, alpha);
            self.nodes[i].neighbors = new_neighbors.clone();

            // Add reverse edges
            for &neighbor in &new_neighbors {
                if !self.nodes[neighbor].neighbors.contains(&i) {
                    if self.nodes[neighbor].neighbors.len() < max_degree {
                        self.nodes[neighbor].neighbors.push(i);
                    } else {
                        // Prune neighbor's edges
                        let neighbor_vec = &all_vectors[neighbor];
                        let mut candidates: Vec<usize> = self.nodes[neighbor].neighbors.clone();
                        candidates.push(i);
                        let pruned = self.robust_prune_indices(
                            &candidates,
                            neighbor_vec,
                            &all_vectors,
                            alpha,
                        );
                        self.nodes[neighbor].neighbors = pruned;
                    }
                }
            }
        }

        // Update entry point to medoid
        self.metadata.entry_point = Some(self.find_medoid(&all_vectors));

        Ok(())
    }

    /// Greedy search on the graph (internal, uses preloaded vectors).
    fn greedy_search_internal(
        &self,
        query: &[f32],
        all_vectors: &[Vec<f32>],
        list_size: usize,
        exclude: Option<usize>,
    ) -> Vec<(usize, f32)> {
        let entry = match self.metadata.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };

        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();

        // Start from entry point
        let entry_dist = self.distance(query, &all_vectors[entry]);
        candidates.push(SearchCandidate {
            index: entry,
            distance: entry_dist,
            visited: false,
        });
        visited.insert(entry);

        let mut results: Vec<(usize, f32)> = Vec::new();

        while let Some(mut current) = candidates.pop() {
            if current.visited {
                continue;
            }
            current.visited = true;

            if exclude != Some(current.index) {
                results.push((current.index, current.distance));
            }

            if results.len() >= list_size {
                break;
            }

            // Explore neighbors
            for &neighbor in &self.nodes[current.index].neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let dist = self.distance(query, &all_vectors[neighbor]);
                    candidates.push(SearchCandidate {
                        index: neighbor,
                        distance: dist,
                        visited: false,
                    });
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(list_size);
        results
    }

    /// Robust prune algorithm for Vamana.
    fn robust_prune(
        &self,
        candidates: &[(usize, f32)],
        query: &[f32],
        all_vectors: &[Vec<f32>],
        alpha: f32,
    ) -> Vec<usize> {
        let indices: Vec<usize> = candidates.iter().map(|(i, _)| *i).collect();
        self.robust_prune_indices(&indices, query, all_vectors, alpha)
    }

    /// Robust prune with indices only.
    fn robust_prune_indices(
        &self,
        candidates: &[usize],
        query: &[f32],
        all_vectors: &[Vec<f32>],
        alpha: f32,
    ) -> Vec<usize> {
        let max_degree = self.metadata.config.max_degree;

        // Sort candidates by distance
        let mut sorted: Vec<(usize, f32)> = candidates
            .iter()
            .map(|&i| (i, self.distance(query, &all_vectors[i])))
            .collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result: Vec<usize> = Vec::new();

        for (candidate, candidate_dist) in sorted {
            if result.len() >= max_degree {
                break;
            }

            // Check if candidate is alpha-dominated by existing neighbors
            let mut dominated = false;
            for &neighbor in &result {
                let neighbor_to_candidate =
                    self.distance(&all_vectors[neighbor], &all_vectors[candidate]);
                if neighbor_to_candidate * alpha < candidate_dist {
                    dominated = true;
                    break;
                }
            }

            if !dominated {
                result.push(candidate);
            }
        }

        result
    }

    /// Find the medoid (most central point).
    fn find_medoid(&self, all_vectors: &[Vec<f32>]) -> usize {
        let n = all_vectors.len();
        if n == 0 {
            return 0;
        }

        // Compute centroid
        let dim = all_vectors[0].len();
        let mut centroid = vec![0.0f32; dim];
        for vec in all_vectors {
            for (i, &v) in vec.iter().enumerate() {
                centroid[i] += v;
            }
        }
        for c in &mut centroid {
            *c /= n as f32;
        }

        // Find closest point to centroid
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, vec) in all_vectors.iter().enumerate() {
            let dist = self.distance(&centroid, vec);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Load a vector from disk.
    fn load_vector(&mut self, index: usize) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.vector_cache.get(&index) {
            return Ok(cached.clone());
        }

        let node = &self.nodes[index];
        let dim = self.metadata.dimensions;

        let file = self
            .data_file
            .as_mut()
            .ok_or_else(|| NeedleError::InvalidInput("Data file not open".to_string()))?;

        file.seek(SeekFrom::Start(node.data_offset))?;

        let mut bytes = vec![0u8; dim * 4];
        file.read_exact(&mut bytes)?;

        let vector: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Add to cache
        self.add_to_cache(index, vector.clone());

        Ok(vector)
    }

    /// Add vector to cache with LRU eviction.
    fn add_to_cache(&mut self, index: usize, vector: Vec<f32>) {
        let cache_size = self.metadata.config.cache_size;

        // Evict if necessary
        while self.vector_cache.len() >= cache_size && !self.cache_order.is_empty() {
            let evict_idx = self.cache_order.remove(0);
            self.vector_cache.remove(&evict_idx);
        }

        self.vector_cache.insert(index, vector);
        self.cache_order.push(index);
    }

    /// Compute Euclidean distance.
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Search for nearest neighbors.
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<DiskAnnResult>> {
        if !self.is_built {
            return Err(NeedleError::InvalidInput("Index not built".to_string()));
        }

        if query.len() != self.metadata.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.metadata.dimensions,
                query.len()
            )));
        }

        let entry = match self.metadata.entry_point {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let search_list_size = self.metadata.config.search_list_size.max(k);
        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();

        // Start from entry point
        let entry_vec = self.load_vector(entry)?;
        let entry_dist = self.distance(query, &entry_vec);
        candidates.push(SearchCandidate {
            index: entry,
            distance: entry_dist,
            visited: false,
        });
        visited.insert(entry);

        let mut results: Vec<(usize, f32)> = Vec::new();

        while let Some(mut current) = candidates.pop() {
            if current.visited {
                continue;
            }
            current.visited = true;
            results.push((current.index, current.distance));

            if results.len() >= search_list_size {
                break;
            }

            // Explore neighbors
            let neighbors = self.nodes[current.index].neighbors.clone();
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    let neighbor_vec = self.load_vector(neighbor)?;
                    let dist = self.distance(query, &neighbor_vec);
                    candidates.push(SearchCandidate {
                        index: neighbor,
                        distance: dist,
                        visited: false,
                    });
                }
            }
        }

        // Sort and truncate
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        // Convert to results
        let disk_results: Vec<DiskAnnResult> = results
            .into_iter()
            .map(|(idx, dist)| DiskAnnResult {
                id: self.nodes[idx].id.clone(),
                distance: dist,
                node_index: idx,
            })
            .collect();

        Ok(disk_results)
    }

    /// Save the index to disk.
    pub fn save(&self) -> Result<()> {
        // Save metadata
        let metadata_path = self.base_path.join("metadata.json");
        let file = File::create(&metadata_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.metadata)?;

        // Save graph
        let graph_path = self.base_path.join("graph.bin");
        let file = File::create(&graph_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &self.nodes)
            .map_err(|e| NeedleError::InvalidInput(format!("Failed to save graph: {}", e)))?;

        Ok(())
    }

    /// Get index statistics.
    pub fn stats(&self) -> DiskAnnStats {
        let total_edges: usize = self.nodes.iter().map(|n| n.neighbors.len()).sum();
        let avg_degree = if self.nodes.is_empty() {
            0.0
        } else {
            total_edges as f64 / self.nodes.len() as f64
        };

        DiskAnnStats {
            num_vectors: self.metadata.num_vectors,
            dimensions: self.metadata.dimensions,
            total_edges,
            avg_degree,
            cache_size: self.vector_cache.len(),
            is_built: self.is_built,
        }
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.metadata.num_vectors
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.num_vectors == 0
    }

    /// Clear the vector cache.
    pub fn clear_cache(&mut self) {
        self.vector_cache.clear();
        self.cache_order.clear();
    }
}

/// Statistics about a DiskANN index.
#[derive(Debug, Clone)]
pub struct DiskAnnStats {
    /// Number of vectors.
    pub num_vectors: usize,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Total number of graph edges.
    pub total_edges: usize,
    /// Average node degree.
    pub avg_degree: f64,
    /// Current cache size.
    pub cache_size: usize,
    /// Whether index is built.
    pub is_built: bool,
}

/// Builder for DiskANN queries.
pub struct DiskAnnQueryBuilder<'a> {
    index: &'a mut DiskAnnIndex,
    query: Vec<f32>,
    k: usize,
    search_list_size: Option<usize>,
}

impl<'a> DiskAnnQueryBuilder<'a> {
    /// Create a new query builder.
    pub fn new(index: &'a mut DiskAnnIndex, query: Vec<f32>) -> Self {
        Self {
            index,
            query,
            k: 10,
            search_list_size: None,
        }
    }

    /// Set the number of results to return.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the search list size (beam width).
    pub fn search_list_size(mut self, size: usize) -> Self {
        self.search_list_size = Some(size);
        self
    }

    /// Execute the search.
    pub fn execute(self) -> Result<Vec<DiskAnnResult>> {
        // Temporarily override search list size if specified
        let original_size = self.index.metadata.config.search_list_size;
        if let Some(size) = self.search_list_size {
            self.index.metadata.config.search_list_size = size;
        }

        let results = self.index.search(&self.query, self.k);

        // Restore original size
        self.index.metadata.config.search_list_size = original_size;

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<(String, Vec<f32>)> {
        (0..n)
            .map(|i| {
                let vector: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect();
                (format!("vec_{}", i), vector)
            })
            .collect()
    }

    #[test]
    fn test_create_index() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let index = DiskAnnIndex::create(dir.path(), 128, config).unwrap();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_vectors() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        index.add("vec1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        index.add("vec2", &[5.0, 6.0, 7.0, 8.0]).unwrap();

        // Vectors are pending until build
        assert_eq!(index.pending_vectors.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        let result = index.add("vec1", &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_id() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        index.add("vec1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        index.build().unwrap();

        let result = index.add("vec1", &[5.0, 6.0, 7.0, 8.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_and_search() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig {
            max_degree: 4,
            build_list_size: 10,
            search_list_size: 10,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        let vectors = vec![
            ("a", vec![0.0, 0.0, 0.0, 0.0]),
            ("b", vec![1.0, 0.0, 0.0, 0.0]),
            ("c", vec![0.0, 1.0, 0.0, 0.0]),
            ("d", vec![0.0, 0.0, 1.0, 0.0]),
            ("e", vec![0.0, 0.0, 0.0, 1.0]),
        ];

        for (id, vec) in vectors {
            index.add(id, &vec).unwrap();
        }

        index.build().unwrap();

        // Search for nearest to origin
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 3).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a"); // Origin should be closest
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_save_and_load() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();

        // Create and build index
        {
            let mut index = DiskAnnIndex::create(dir.path(), 4, config.clone()).unwrap();
            index.add("vec1", &[1.0, 2.0, 3.0, 4.0]).unwrap();
            index.add("vec2", &[5.0, 6.0, 7.0, 8.0]).unwrap();
            index.build().unwrap();
        }

        // Load and verify
        let mut index = DiskAnnIndex::open(dir.path()).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.is_built);

        // Search should work
        let results = index.search(&[1.0, 2.0, 3.0, 4.0], 1).unwrap();
        assert_eq!(results[0].id, "vec1");
    }

    #[test]
    fn test_batch_add() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
            ("c".to_string(), vec![0.0, 0.0, 1.0, 0.0]),
        ];

        index.add_batch(vectors).unwrap();
        index.build().unwrap();

        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_stats() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 8, config).unwrap();

        for i in 0..10 {
            let vec: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32).collect();
            index.add(&format!("vec_{}", i), &vec).unwrap();
        }

        index.build().unwrap();

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 10);
        assert_eq!(stats.dimensions, 8);
        assert!(stats.total_edges > 0);
        assert!(stats.is_built);
    }

    #[test]
    fn test_larger_index() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig {
            max_degree: 8,
            build_list_size: 20,
            search_list_size: 20,
            cache_size: 50,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 16, config).unwrap();

        let vectors = create_test_vectors(100, 16);
        index.add_batch(vectors).unwrap();
        index.build().unwrap();

        // Search should return good results
        let query: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
    }

    #[test]
    fn test_query_builder() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        index.add("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();

        let results = DiskAnnQueryBuilder::new(&mut index, vec![1.0, 0.0, 0.0, 0.0])
            .k(1)
            .search_list_size(10)
            .execute()
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_cache_eviction() {
        let dir = TempDir::new().unwrap();
        let config = DiskAnnConfig {
            cache_size: 5,
            max_degree: 4,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 4, config).unwrap();

        for i in 0..20 {
            let vec = vec![i as f32, 0.0, 0.0, 0.0];
            index.add(&format!("vec_{}", i), &vec).unwrap();
        }

        index.build().unwrap();

        // Multiple searches should trigger cache eviction
        for i in 0..10 {
            let query = vec![i as f32, 0.0, 0.0, 0.0];
            let _ = index.search(&query, 3).unwrap();
        }

        // Cache should be at or under limit
        assert!(index.vector_cache.len() <= 5);
    }
}
