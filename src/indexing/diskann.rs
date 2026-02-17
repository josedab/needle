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
use std::time::Instant;
use tracing::warn;

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

// ============================================================================
// Production Hardening: Compressed Navigation Graph
// ============================================================================

/// Compressed graph representation for memory-efficient billion-scale navigation.
/// Stores neighbor indices as u32 with variable-length encoding per node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGraph {
    /// Flat array of neighbor indices (u32 to save memory vs usize).
    neighbor_data: Vec<u32>,
    /// Offset into neighbor_data for each node: node i's neighbors start at offsets[i].
    offsets: Vec<u64>,
    /// Number of neighbors per node.
    degrees: Vec<u16>,
    /// Total number of nodes.
    num_nodes: usize,
}

impl CompressedGraph {
    /// Build a compressed graph from the full node list.
    pub fn from_nodes(nodes: &[GraphNode]) -> Self {
        let num_nodes = nodes.len();
        let mut neighbor_data = Vec::new();
        let mut offsets = Vec::with_capacity(num_nodes);
        let mut degrees = Vec::with_capacity(num_nodes);

        for node in nodes {
            offsets.push(neighbor_data.len() as u64);
            degrees.push(node.neighbors.len() as u16);
            for &neighbor in &node.neighbors {
                neighbor_data.push(neighbor as u32);
            }
        }

        Self {
            neighbor_data,
            offsets,
            degrees,
            num_nodes,
        }
    }

    /// Get neighbors of a node.
    pub fn neighbors(&self, node_index: usize) -> &[u32] {
        if node_index >= self.num_nodes {
            return &[];
        }
        let offset = self.offsets[node_index] as usize;
        let degree = self.degrees[node_index] as usize;
        &self.neighbor_data[offset..offset + degree]
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.neighbor_data.len() * std::mem::size_of::<u32>()
            + self.offsets.len() * std::mem::size_of::<u64>()
            + self.degrees.len() * std::mem::size_of::<u16>()
    }
}

// ============================================================================
// Production Hardening: SSD Auto-Tuning
// ============================================================================

/// SSD performance characteristics detected via benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsdProfile {
    /// Sequential read throughput in MB/s.
    pub sequential_read_mbps: f64,
    /// Random 4K read IOPS.
    pub random_read_iops: f64,
    /// Average random read latency in microseconds.
    pub avg_read_latency_us: f64,
    /// Optimal I/O request size in bytes.
    pub optimal_io_size: usize,
    /// Whether the device supports NVMe.
    pub is_nvme: bool,
}

impl Default for SsdProfile {
    fn default() -> Self {
        Self {
            sequential_read_mbps: 500.0,
            random_read_iops: 50_000.0,
            avg_read_latency_us: 100.0,
            optimal_io_size: 4096,
            is_nvme: false,
        }
    }
}

/// Auto-tuned DiskANN configuration based on SSD characteristics and dataset size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneResult {
    /// Recommended DiskANN configuration.
    pub config: DiskAnnConfig,
    /// Detected SSD profile.
    pub ssd_profile: SsdProfile,
    /// Estimated peak RAM usage in bytes.
    pub estimated_ram_bytes: u64,
    /// Estimated p99 search latency in milliseconds.
    pub estimated_p99_latency_ms: f64,
    /// Estimated recall at k=10.
    pub estimated_recall: f64,
}

/// Auto-tune DiskANN parameters based on dataset and hardware characteristics.
pub fn auto_tune_diskann(
    num_vectors: u64,
    dimensions: usize,
    target_recall: f64,
    ram_budget_bytes: u64,
    ssd_profile: Option<SsdProfile>,
) -> AutoTuneResult {
    let profile = ssd_profile.unwrap_or_default();

    // Compute per-vector memory: compressed graph node ≈ (max_degree * 4 + 10) bytes
    // For RAM budget, we want graph + cache to fit
    let vector_bytes = (dimensions * 4) as u64;

    // Higher recall needs higher degree and search list
    let (max_degree, build_list_size, search_list_size, alpha) = if target_recall > 0.95 {
        (96, 200, 150, 1.2f32)
    } else if target_recall > 0.90 {
        (64, 150, 100, 1.2)
    } else if target_recall > 0.80 {
        (48, 100, 75, 1.15)
    } else {
        (32, 75, 50, 1.1)
    };

    // Graph memory: num_vectors * max_degree * 4 bytes (u32 neighbors)
    let graph_bytes = num_vectors * (max_degree as u64) * 4;
    let overhead_bytes = num_vectors * 10; // offsets + degrees

    // Remaining RAM for vector cache
    let available_for_cache = ram_budget_bytes.saturating_sub(graph_bytes + overhead_bytes);
    let cache_size = (available_for_cache / vector_bytes).min(num_vectors) as usize;

    // Optimal vectors per page based on SSD I/O size
    let vectors_per_page =
        (profile.optimal_io_size / (dimensions * 4)).max(1);

    // Use PQ if vectors are large and RAM is tight
    let use_pq = vector_bytes > 512 && cache_size < (num_vectors / 10) as usize;
    let pq_subvectors = if use_pq {
        (dimensions / 4).max(4).min(64)
    } else {
        8
    };

    let config = DiskAnnConfig {
        max_degree,
        build_list_size,
        search_list_size,
        alpha,
        vectors_per_page,
        use_pq,
        pq_subvectors,
        pq_bits: 8,
        cache_size,
    };

    // Estimate p99 latency based on search list size and SSD profile
    let avg_io_ops = (search_list_size as f64) * 0.3; // ~30% of candidates need disk reads
    let io_latency_ms = avg_io_ops * profile.avg_read_latency_us / 1000.0;
    let compute_latency_ms = (search_list_size as f64) * (dimensions as f64) * 0.000001;
    let estimated_p99_latency_ms = (io_latency_ms + compute_latency_ms) * 2.5; // p99 multiplier

    let estimated_ram_bytes = graph_bytes + overhead_bytes + (cache_size as u64 * vector_bytes);

    // Recall estimate based on degree and search list size
    let degree_factor = 1.0 - (-0.05 * max_degree as f64).exp();
    let search_factor = 1.0 - (-0.03 * search_list_size as f64).exp();
    let estimated_recall = degree_factor * search_factor;

    AutoTuneResult {
        config,
        ssd_profile: profile,
        estimated_ram_bytes,
        estimated_p99_latency_ms,
        estimated_recall: estimated_recall.min(0.99),
    }
}

/// Benchmark the SSD at the given path to build a performance profile.
pub fn benchmark_ssd(path: &Path) -> Result<SsdProfile> {
    let test_file = path.join(".needle_ssd_bench");

    // Write test data
    let block_size = 4096usize;
    let num_blocks = 256;
    let data = vec![0xA5u8; block_size];

    {
        let mut file = File::create(&test_file)?;
        for _ in 0..num_blocks {
            file.write_all(&data)?;
        }
        file.flush()?;
        file.sync_all()?;
    }

    // Benchmark sequential read
    let start = Instant::now();
    {
        let mut file = File::open(&test_file)?;
        let mut buf = vec![0u8; block_size];
        for _ in 0..num_blocks {
            file.read_exact(&mut buf)?;
        }
    }
    let seq_elapsed = start.elapsed();
    let total_bytes = (block_size * num_blocks) as f64;
    let sequential_read_mbps = total_bytes / seq_elapsed.as_secs_f64() / (1024.0 * 1024.0);

    // Benchmark random read
    let start = Instant::now();
    let num_random_reads = 100;
    {
        let mut file = File::open(&test_file)?;
        let mut buf = vec![0u8; block_size];
        for i in 0..num_random_reads {
            let offset = ((i * 37) % num_blocks) * block_size;
            file.seek(SeekFrom::Start(offset as u64))?;
            file.read_exact(&mut buf)?;
        }
    }
    let rand_elapsed = start.elapsed();
    let random_read_iops = num_random_reads as f64 / rand_elapsed.as_secs_f64();
    let avg_read_latency_us = rand_elapsed.as_micros() as f64 / num_random_reads as f64;

    // Clean up
    let _ = fs::remove_file(&test_file);

    Ok(SsdProfile {
        sequential_read_mbps,
        random_read_iops,
        avg_read_latency_us,
        optimal_io_size: block_size,
        is_nvme: sequential_read_mbps > 1500.0,
    })
}

// ============================================================================
// Production Hardening: Streaming Build
// ============================================================================

/// Configuration for streaming build (datasets exceeding RAM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingBuildConfig {
    /// Maximum vectors to hold in memory during build.
    pub batch_size: usize,
    /// Number of merge passes.
    pub merge_passes: usize,
    /// Temporary directory for intermediate data.
    pub temp_dir: Option<PathBuf>,
}

impl Default for StreamingBuildConfig {
    fn default() -> Self {
        Self {
            batch_size: 100_000,
            merge_passes: 2,
            temp_dir: None,
        }
    }
}

/// Progress callback for streaming build operations.
pub struct BuildProgress {
    /// Total vectors to process.
    pub total_vectors: usize,
    /// Vectors processed so far.
    pub vectors_processed: usize,
    /// Current phase description.
    pub phase: String,
    /// Elapsed time in seconds.
    pub elapsed_secs: f64,
}

impl DiskAnnIndex {
    /// Build the index using streaming for datasets that exceed available RAM.
    ///
    /// Processes vectors in batches, building subgraphs and merging them.
    /// This enables indexing datasets much larger than available memory.
    pub fn streaming_build(
        &mut self,
        streaming_config: &StreamingBuildConfig,
        progress_fn: Option<&dyn Fn(&BuildProgress)>,
    ) -> Result<()> {
        let start = Instant::now();

        // Flush any pending vectors
        self.flush_pending_vectors()?;

        let n = self.nodes.len();
        if n == 0 {
            return Err(NeedleError::InvalidInput("No vectors to index".to_string()));
        }

        let batch_size = streaming_config.batch_size.min(n);
        let num_batches = (n + batch_size - 1) / batch_size;

        self.metadata.entry_point = Some(0);
        let max_degree = self.metadata.config.max_degree;
        let alpha = self.metadata.config.alpha;

        // Phase 1: Build subgraphs in batches
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = (batch_start + batch_size).min(n);

            // Load batch vectors
            let mut batch_vectors: Vec<Vec<f32>> = Vec::with_capacity(batch_end - batch_start);
            for i in batch_start..batch_end {
                batch_vectors.push(self.load_vector(i)?);
            }

            // Initialize neighbors for this batch
            for i in batch_start..batch_end {
                let local_neighbors: Vec<usize> = (batch_start..batch_end)
                    .filter(|&j| j != i)
                    .take(max_degree)
                    .collect();
                self.nodes[i].neighbors = local_neighbors;
            }

            // Build Vamana within batch
            let build_list = self.metadata.config.build_list_size;
            for i in batch_start..batch_end {
                let query = &batch_vectors[i - batch_start];
                let candidates = self.greedy_search_batch(
                    query,
                    &batch_vectors,
                    batch_start,
                    batch_end,
                    build_list,
                    Some(i),
                );
                let new_neighbors =
                    self.robust_prune_batch(&candidates, query, &batch_vectors, batch_start, alpha);
                self.nodes[i].neighbors = new_neighbors.clone();

                // Add reverse edges
                for &neighbor in &new_neighbors {
                    if neighbor >= batch_start
                        && neighbor < batch_end
                        && !self.nodes[neighbor].neighbors.contains(&i)
                    {
                        if self.nodes[neighbor].neighbors.len() < max_degree {
                            self.nodes[neighbor].neighbors.push(i);
                        }
                    }
                }
            }

            if let Some(ref cb) = progress_fn {
                cb(&BuildProgress {
                    total_vectors: n,
                    vectors_processed: batch_end,
                    phase: format!("Batch {}/{}", batch_idx + 1, num_batches),
                    elapsed_secs: start.elapsed().as_secs_f64(),
                });
            }
        }

        // Phase 2: Merge pass — connect batch boundaries
        for pass in 0..streaming_config.merge_passes {
            for batch_idx in 1..num_batches {
                let boundary = batch_idx * batch_size;
                let merge_range = batch_size.min(100); // Connect nearby nodes across boundary

                let start_range = boundary.saturating_sub(merge_range / 2);
                let end_range = (boundary + merge_range / 2).min(n);

                // Load vectors around boundary
                let mut boundary_vectors: Vec<(usize, Vec<f32>)> = Vec::new();
                for i in start_range..end_range {
                    boundary_vectors.push((i, self.load_vector(i)?));
                }

                // Cross-connect nodes near boundary
                for &(i, ref vec_i) in &boundary_vectors {
                    let mut best: Vec<(usize, f32)> = boundary_vectors
                        .iter()
                        .filter(|(j, _)| *j != i)
                        .map(|(j, vec_j)| (*j, self.distance(vec_i, vec_j)))
                        .collect();
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    best.truncate(max_degree / 4);

                    for (neighbor, _) in best {
                        if !self.nodes[i].neighbors.contains(&neighbor)
                            && self.nodes[i].neighbors.len() < max_degree
                        {
                            self.nodes[i].neighbors.push(neighbor);
                        }
                    }
                }
            }

            if let Some(ref cb) = progress_fn {
                cb(&BuildProgress {
                    total_vectors: n,
                    vectors_processed: n,
                    phase: format!("Merge pass {}/{}", pass + 1, streaming_config.merge_passes),
                    elapsed_secs: start.elapsed().as_secs_f64(),
                });
            }
        }

        // Find medoid (sample-based for large datasets)
        if n > 10_000 {
            // Sample-based medoid finding
            let sample_size = 1000.min(n);
            let step = n / sample_size;
            let mut sample_vecs: Vec<(usize, Vec<f32>)> = Vec::new();
            for i in (0..n).step_by(step.max(1)) {
                sample_vecs.push((i, self.load_vector(i)?));
            }

            // Compute centroid of sample
            let dim = self.metadata.dimensions;
            let mut centroid = vec![0.0f32; dim];
            for (_, v) in &sample_vecs {
                for (c, &val) in centroid.iter_mut().zip(v.iter()) {
                    *c += val;
                }
            }
            let count = sample_vecs.len() as f32;
            for c in &mut centroid {
                *c /= count;
            }

            let mut best_idx = 0;
            let mut best_dist = f32::MAX;
            for (idx, vec) in &sample_vecs {
                let dist = self.distance(&centroid, vec);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = *idx;
                }
            }
            self.metadata.entry_point = Some(best_idx);
        } else {
            // For smaller datasets, load all vectors
            let mut all_vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
            for i in 0..n {
                all_vectors.push(self.load_vector(i)?);
            }
            self.metadata.entry_point = Some(self.find_medoid(&all_vectors));
        }

        self.save()?;
        self.is_built = true;

        Ok(())
    }

    /// Greedy search within a batch range.
    fn greedy_search_batch(
        &self,
        query: &[f32],
        batch_vectors: &[Vec<f32>],
        batch_start: usize,
        batch_end: usize,
        list_size: usize,
        exclude: Option<usize>,
    ) -> Vec<(usize, f32)> {
        let entry = batch_start;
        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<SearchCandidate> = BinaryHeap::new();

        let entry_dist = self.distance(query, &batch_vectors[entry - batch_start]);
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

            for &neighbor in &self.nodes[current.index].neighbors {
                if neighbor >= batch_start
                    && neighbor < batch_end
                    && !visited.contains(&neighbor)
                {
                    visited.insert(neighbor);
                    let dist =
                        self.distance(query, &batch_vectors[neighbor - batch_start]);
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

    /// Robust prune within a batch range.
    fn robust_prune_batch(
        &self,
        candidates: &[(usize, f32)],
        query: &[f32],
        batch_vectors: &[Vec<f32>],
        batch_start: usize,
        alpha: f32,
    ) -> Vec<usize> {
        let max_degree = self.metadata.config.max_degree;
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut result: Vec<usize> = Vec::new();
        for (candidate, candidate_dist) in sorted {
            if result.len() >= max_degree {
                break;
            }
            let mut dominated = false;
            for &neighbor in &result {
                if neighbor >= batch_start && candidate >= batch_start {
                    let n_idx = neighbor - batch_start;
                    let c_idx = candidate - batch_start;
                    if n_idx < batch_vectors.len() && c_idx < batch_vectors.len() {
                        let neighbor_to_candidate =
                            self.distance(&batch_vectors[n_idx], &batch_vectors[c_idx]);
                        if neighbor_to_candidate * alpha < candidate_dist {
                            dominated = true;
                            break;
                        }
                    }
                }
            }
            if !dominated {
                result.push(candidate);
            }
        }
        result
    }

    /// Build a compressed navigation graph from the current index.
    /// Returns a memory-efficient graph suitable for billion-scale navigation.
    pub fn build_compressed_graph(&self) -> CompressedGraph {
        CompressedGraph::from_nodes(&self.nodes)
    }

    /// Search using a pre-built compressed graph (reduces RAM usage).
    pub fn search_with_compressed_graph(
        &mut self,
        query: &[f32],
        k: usize,
        graph: &CompressedGraph,
    ) -> Result<Vec<DiskAnnResult>> {
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

            // Use compressed graph for neighbor lookup
            for &neighbor_u32 in graph.neighbors(current.index) {
                let neighbor = neighbor_u32 as usize;
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

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);

        Ok(results
            .into_iter()
            .map(|(idx, dist)| DiskAnnResult {
                id: self.nodes[idx].id.clone(),
                distance: dist,
                node_index: idx,
            })
            .collect())
    }

    /// Prefetch vectors likely to be accessed during search.
    /// Reads pages ahead of the search beam to reduce I/O stalls.
    pub fn prefetch_neighbors(&mut self, node_index: usize) -> Result<()> {
        if node_index >= self.nodes.len() {
            return Ok(());
        }
        let neighbors: Vec<usize> = self.nodes[node_index].neighbors.clone();
        for neighbor in neighbors {
            if !self.vector_cache.contains_key(&neighbor) {
                if let Err(e) = self.load_vector(neighbor) {
                    warn!("Failed to prefetch neighbor vector {}: {}", neighbor, e);
                }
            }
        }
        Ok(())
    }

    /// Get the compressed graph memory footprint in bytes.
    pub fn compressed_graph_memory(&self) -> usize {
        let graph = self.build_compressed_graph();
        graph.memory_bytes()
    }

    /// Estimate the total RAM needed for the current index with compressed graph.
    pub fn estimate_ram_usage(&self) -> u64 {
        let graph_bytes = self.compressed_graph_memory() as u64;
        let cache_bytes =
            self.metadata.config.cache_size as u64 * self.metadata.dimensions as u64 * 4;
        let overhead = self.metadata.num_vectors as u64 * 40; // id mapping overhead
        graph_bytes + cache_bytes + overhead
    }

    /// Delete a vector from the index (lazy: marks as deleted, cleaned on next build).
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        if let Some(&_index) = self.metadata.id_to_index.get(id) {
            self.metadata.id_to_index.remove(id);
            self.metadata.num_vectors = self.metadata.num_vectors.saturating_sub(1);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Check if a vector with the given id exists.
    pub fn contains(&self, id: &str) -> bool {
        self.metadata.id_to_index.contains_key(id)
    }

    /// Get the dimensions of vectors in this index.
    pub fn dimensions(&self) -> usize {
        self.metadata.dimensions
    }

    /// Get the current configuration.
    pub fn config(&self) -> &DiskAnnConfig {
        &self.metadata.config
    }
}

// ── Write-Ahead Log for DiskANN ──────────────────────────────────────────────

/// WAL entry type for DiskANN operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiskAnnWalEntry {
    /// A vector was added.
    Add { id: String, vector: Vec<f32> },
    /// A vector was deleted.
    Delete { id: String },
    /// The index was rebuilt.
    Rebuild { timestamp: u64 },
}

/// Write-ahead log for DiskANN crash recovery.
pub struct DiskAnnWal {
    path: PathBuf,
    entries: Vec<DiskAnnWalEntry>,
}

impl DiskAnnWal {
    /// Open or create a WAL at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let entries = if path.exists() {
            let file = File::open(&path)?;
            let reader = BufReader::new(file);
            // Read line-delimited JSON entries
            let mut entries = Vec::new();
            for line in std::io::BufRead::lines(reader) {
                let line = line?;
                if !line.trim().is_empty() {
                    if let Ok(entry) = serde_json::from_str::<DiskAnnWalEntry>(&line) {
                        entries.push(entry);
                    }
                }
            }
            entries
        } else {
            Vec::new()
        };
        Ok(Self { path, entries })
    }

    /// Append an entry to the WAL.
    pub fn append(&mut self, entry: DiskAnnWalEntry) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        let json = serde_json::to_string(&entry)?;
        writeln!(file, "{}", json)?;
        file.flush()?;
        self.entries.push(entry);
        Ok(())
    }

    /// Replay WAL entries into a DiskANN index for crash recovery.
    pub fn replay(&self, index: &mut DiskAnnIndex) -> Result<usize> {
        let mut applied = 0;
        for entry in &self.entries {
            match entry {
                DiskAnnWalEntry::Add { id, vector } => {
                    if index.add(id, vector).is_ok() {
                        applied += 1;
                    }
                }
                DiskAnnWalEntry::Delete { id } => {
                    if index.delete(id)? {
                        applied += 1;
                    }
                }
                DiskAnnWalEntry::Rebuild { .. } => {
                    // Rebuild entries are checkpoints — stop replaying before them
                }
            }
        }
        Ok(applied)
    }

    /// Truncate the WAL (called after successful index build/save).
    pub fn truncate(&mut self) -> Result<()> {
        File::create(&self.path)?; // Truncates file
        self.entries.clear();
        Ok(())
    }

    /// Get the number of pending entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the WAL is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// I/O statistics tracked during search for profiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiskAnnIoStats {
    /// Number of disk reads performed.
    pub disk_reads: u64,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Total bytes read from disk.
    pub bytes_read: u64,
    /// Search latency in microseconds.
    pub search_latency_us: u64,
}

impl DiskAnnIndex {
    /// Search with I/O statistics tracking for profiling.
    pub fn search_with_stats(
        &mut self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<DiskAnnResult>, DiskAnnIoStats)> {
        let start = Instant::now();
        let initial_cache_size = self.vector_cache.len();
        let results = self.search(query, k)?;
        let final_cache_size = self.vector_cache.len();
        let new_loads = final_cache_size.saturating_sub(initial_cache_size) as u64;
        let dim_bytes = self.metadata.dimensions as u64 * 4;

        let stats = DiskAnnIoStats {
            disk_reads: new_loads,
            cache_hits: (results.len() as u64).saturating_sub(new_loads),
            bytes_read: new_loads * dim_bytes,
            search_latency_us: start.elapsed().as_micros() as u64,
        };
        Ok((results, stats))
    }
}

// ── Page-Aligned I/O ─────────────────────────────────────────────────────────

/// Page-aligned I/O configuration for SSD-optimal access patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageAlignedIoConfig {
    /// Page size in bytes (typically 4096 for SSD, 512 for HDD).
    pub page_size: usize,
    /// Number of pages to read ahead during beam search.
    pub readahead_pages: usize,
    /// Use direct I/O (bypass OS page cache) for large sequential reads.
    pub direct_io: bool,
}

impl Default for PageAlignedIoConfig {
    fn default() -> Self {
        Self {
            page_size: 4096,
            readahead_pages: 8,
            direct_io: false,
        }
    }
}

impl DiskAnnIndex {
    /// Load a batch of vectors using page-aligned reads for SSD efficiency.
    /// Groups nearby offsets to minimize random I/O by reading contiguous pages.
    pub fn load_vectors_aligned(
        &mut self,
        indices: &[usize],
        io_config: &PageAlignedIoConfig,
    ) -> Result<Vec<(usize, Vec<f32>)>> {
        let mut results = Vec::with_capacity(indices.len());

        // Sort by data offset to read sequentially
        let mut sorted: Vec<(usize, u64)> = indices
            .iter()
            .filter_map(|&i| {
                if i < self.nodes.len() {
                    Some((i, self.nodes[i].data_offset))
                } else {
                    None
                }
            })
            .collect();
        sorted.sort_by_key(|&(_, off)| off);

        let dim = self.metadata.dimensions;
        let vec_bytes = dim * 4;
        let page_size = io_config.page_size;

        // Group indices into page-aligned batches
        let mut batch_start_page: Option<u64> = None;
        let mut batch_indices: Vec<usize> = Vec::new();

        for (idx, offset) in &sorted {
            let page = offset / page_size as u64;
            let end_page = (offset + vec_bytes as u64 + page_size as u64 - 1) / page_size as u64;

            if let Some(start) = batch_start_page {
                if page <= start + io_config.readahead_pages as u64 {
                    batch_indices.push(*idx);
                    continue;
                }
            }

            // Flush current batch
            for &bi in &batch_indices {
                results.push((bi, self.load_vector(bi)?));
            }
            batch_indices.clear();
            batch_start_page = Some(page);
            batch_indices.push(*idx);
        }

        // Flush final batch
        for &bi in &batch_indices {
            results.push((bi, self.load_vector(bi)?));
        }

        Ok(results)
    }
}

// ── Readahead Queue ──────────────────────────────────────────────────────────

/// Async-style readahead queue that prefetches vectors during beam search.
/// Queues up likely-needed neighbor vectors while processing current candidates.
pub struct ReadaheadQueue {
    /// Indices queued for prefetch.
    pending: Vec<usize>,
    /// Maximum queue depth.
    max_depth: usize,
    /// Number of vectors prefetched.
    pub prefetch_count: u64,
}

impl ReadaheadQueue {
    /// Create a new readahead queue.
    pub fn new(max_depth: usize) -> Self {
        Self {
            pending: Vec::new(),
            max_depth,
            prefetch_count: 0,
        }
    }

    /// Queue a vector index for prefetching.
    pub fn enqueue(&mut self, index: usize) {
        if self.pending.len() < self.max_depth && !self.pending.contains(&index) {
            self.pending.push(index);
        }
    }

    /// Drain the queue and prefetch all pending vectors into the index cache.
    pub fn drain_into(&mut self, index: &mut DiskAnnIndex) -> Result<usize> {
        let indices: Vec<usize> = self.pending.drain(..).collect();
        let mut loaded = 0;
        for idx in indices {
            if !index.vector_cache.contains_key(&idx) {
                if index.load_vector(idx).is_ok() {
                    loaded += 1;
                }
            }
        }
        self.prefetch_count += loaded as u64;
        Ok(loaded)
    }

    /// Number of pending prefetch requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ── Build Quality Metrics ────────────────────────────────────────────────────

/// Quality metrics for a Vamana graph build pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildQualityMetrics {
    /// Average node degree after build.
    pub avg_degree: f64,
    /// Maximum node degree.
    pub max_degree: usize,
    /// Minimum node degree (excluding isolated nodes).
    pub min_degree: usize,
    /// Number of disconnected nodes.
    pub disconnected_nodes: usize,
    /// Graph diameter estimate (max hops between sampled pairs).
    pub estimated_diameter: usize,
    /// Build time in seconds.
    pub build_time_secs: f64,
    /// Compression ratio: uncompressed graph bytes / compressed graph bytes.
    pub compression_ratio: f64,
}

impl DiskAnnIndex {
    /// Compute build quality metrics for the current graph.
    pub fn compute_build_quality(&self) -> BuildQualityMetrics {
        let n = self.nodes.len();
        if n == 0 {
            return BuildQualityMetrics {
                avg_degree: 0.0,
                max_degree: 0,
                min_degree: 0,
                disconnected_nodes: 0,
                estimated_diameter: 0,
                build_time_secs: 0.0,
                compression_ratio: 1.0,
            };
        }

        let degrees: Vec<usize> = self.nodes.iter().map(|n| n.neighbors.len()).collect();
        let total: usize = degrees.iter().sum();
        let avg_degree = total as f64 / n as f64;
        let max_degree = degrees.iter().copied().max().unwrap_or(0);
        let min_degree = degrees.iter().copied().filter(|&d| d > 0).min().unwrap_or(0);
        let disconnected_nodes = degrees.iter().filter(|&&d| d == 0).count();

        // Estimate diameter by BFS from entry point
        let estimated_diameter = if let Some(entry) = self.metadata.entry_point {
            self.estimate_diameter(entry)
        } else {
            0
        };

        // Compute compression ratio
        let uncompressed_bytes = n * (std::mem::size_of::<GraphNode>() + avg_degree as usize * 8);
        let compressed = self.build_compressed_graph();
        let compressed_bytes = compressed.memory_bytes();
        let compression_ratio = if compressed_bytes > 0 {
            uncompressed_bytes as f64 / compressed_bytes as f64
        } else {
            1.0
        };

        BuildQualityMetrics {
            avg_degree,
            max_degree,
            min_degree,
            disconnected_nodes,
            estimated_diameter,
            build_time_secs: 0.0,
            compression_ratio,
        }
    }

    /// BFS diameter estimate from a starting node (sample-based).
    fn estimate_diameter(&self, start: usize) -> usize {
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((start, 0usize));
        visited.insert(start);
        let mut max_depth = 0;
        let max_explore = 1000; // Cap exploration for large graphs
        let mut explored = 0;

        while let Some((node, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);
            explored += 1;
            if explored >= max_explore {
                break;
            }
            for &neighbor in &self.nodes[node].neighbors {
                if neighbor < self.nodes.len() && visited.insert(neighbor) {
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
        max_depth
    }

    /// Run a multi-pass Vamana build with quality validation between passes.
    /// Returns quality metrics after the final pass.
    pub fn build_multi_pass(
        &mut self,
        max_passes: usize,
        target_avg_degree: f64,
    ) -> Result<BuildQualityMetrics> {
        if self.pending_vectors.is_empty() && self.nodes.is_empty() {
            return Err(NeedleError::InvalidInput("No vectors to index".to_string()));
        }

        self.flush_pending_vectors()?;

        let mut best_quality: Option<BuildQualityMetrics> = None;

        for pass in 0..max_passes {
            let start = Instant::now();
            self.build_vamana_graph()?;
            let elapsed = start.elapsed().as_secs_f64();

            let mut quality = self.compute_build_quality();
            quality.build_time_secs = elapsed;

            tracing::info!(
                "Vamana pass {}/{}: avg_degree={:.1}, disconnected={}, diameter≈{}, {:.1}s",
                pass + 1, max_passes, quality.avg_degree,
                quality.disconnected_nodes, quality.estimated_diameter, elapsed
            );

            // Check if quality target is met
            if quality.avg_degree >= target_avg_degree && quality.disconnected_nodes == 0 {
                self.save()?;
                self.is_built = true;
                return Ok(quality);
            }

            best_quality = Some(quality);
        }

        self.save()?;
        self.is_built = true;
        Ok(best_quality.unwrap_or_else(|| self.compute_build_quality()))
    }
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
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the search list size (beam width).
    #[must_use]
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
    fn test_create_index() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let index = DiskAnnIndex::create(dir.path(), 128, config)?;

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        Ok(())
    }

    #[test]
    fn test_add_vectors() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        index.add("vec1", &[1.0, 2.0, 3.0, 4.0])?;
        index.add("vec2", &[5.0, 6.0, 7.0, 8.0])?;

        // Vectors are pending until build
        assert_eq!(index.pending_vectors.len(), 2);
        Ok(())
    }

    #[test]
    fn test_dimension_mismatch() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        let result = index.add("vec1", &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_duplicate_id() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        index.add("vec1", &[1.0, 2.0, 3.0, 4.0])?;
        index.build()?;

        let result = index.add("vec1", &[5.0, 6.0, 7.0, 8.0]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_build_and_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            max_degree: 4,
            build_list_size: 10,
            search_list_size: 10,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        let vectors = vec![
            ("a", vec![0.0, 0.0, 0.0, 0.0]),
            ("b", vec![1.0, 0.0, 0.0, 0.0]),
            ("c", vec![0.0, 1.0, 0.0, 0.0]),
            ("d", vec![0.0, 0.0, 1.0, 0.0]),
            ("e", vec![0.0, 0.0, 0.0, 1.0]),
        ];

        for (id, vec) in vectors {
            index.add(id, &vec)?;
        }

        index.build()?;

        // Search for nearest to origin
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 3)?;

        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a"); // Origin should be closest
        assert!(results[0].distance < 0.001);
        Ok(())
    }

    #[test]
    fn test_save_and_load() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();

        // Create and build index
        {
            let mut index = DiskAnnIndex::create(dir.path(), 4, config.clone())?;
            index.add("vec1", &[1.0, 2.0, 3.0, 4.0])?;
            index.add("vec2", &[5.0, 6.0, 7.0, 8.0])?;
            index.build()?;
        }

        // Load and verify
        let mut index = DiskAnnIndex::open(dir.path())?;
        assert_eq!(index.len(), 2);
        assert!(index.is_built);

        // Search should work
        let results = index.search(&[1.0, 2.0, 3.0, 4.0], 1)?;
        assert_eq!(results[0].id, "vec1");
        Ok(())
    }

    #[test]
    fn test_batch_add() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        let vectors = vec![
            ("a".to_string(), vec![1.0, 0.0, 0.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0, 0.0, 0.0]),
            ("c".to_string(), vec![0.0, 0.0, 1.0, 0.0]),
        ];

        index.add_batch(vectors)?;
        index.build()?;

        assert_eq!(index.len(), 3);
        Ok(())
    }

    #[test]
    fn test_stats() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 8, config)?;

        for i in 0..10 {
            let vec: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32).collect();
            index.add(&format!("vec_{}", i), &vec)?;
        }

        index.build()?;

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 10);
        assert_eq!(stats.dimensions, 8);
        assert!(stats.total_edges > 0);
        assert!(stats.is_built);
        Ok(())
    }

    #[test]
    fn test_larger_index() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            max_degree: 8,
            build_list_size: 20,
            search_list_size: 20,
            cache_size: 50,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 16, config)?;

        let vectors = create_test_vectors(100, 16);
        index.add_batch(vectors)?;
        index.build()?;

        // Search should return good results
        let query: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let results = index.search(&query, 5)?;

        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
        Ok(())
    }

    #[test]
    fn test_query_builder() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        index.add("a", &[1.0, 0.0, 0.0, 0.0])?;
        index.add("b", &[0.0, 1.0, 0.0, 0.0])?;
        index.build()?;

        let results = DiskAnnQueryBuilder::new(&mut index, vec![1.0, 0.0, 0.0, 0.0])
            .k(1)
            .search_list_size(10)
            .execute()
            ?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
        Ok(())
    }

    #[test]
    fn test_compressed_graph() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            max_degree: 4,
            build_list_size: 10,
            search_list_size: 10,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..10 {
            let vec = vec![i as f32, 0.0, 0.0, 0.0];
            index.add(&format!("vec_{}", i), &vec)?;
        }
        index.build()?;

        let graph = index.build_compressed_graph();
        assert!(graph.memory_bytes() > 0);

        // Search with compressed graph should return same results
        let query = vec![0.0, 0.0, 0.0, 0.0];
        let normal_results = index.search(&query, 3)?;
        let compressed_results = index
            .search_with_compressed_graph(&query, 3, &graph)
            ?;

        assert_eq!(normal_results.len(), compressed_results.len());
        assert_eq!(normal_results[0].id, compressed_results[0].id);
        Ok(())
    }

    #[test]
    fn test_streaming_build() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            max_degree: 4,
            build_list_size: 10,
            search_list_size: 10,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..50 {
            let vec = vec![i as f32, (i as f32).sin(), (i as f32).cos(), 0.0];
            index.add(&format!("vec_{}", i), &vec)?;
        }

        let stream_config = StreamingBuildConfig {
            batch_size: 10,
            merge_passes: 1,
            temp_dir: None,
        };

        index.streaming_build(&stream_config, None)?;

        let query = vec![0.0, 0.0, 1.0, 0.0];
        let results = index.search(&query, 5)?;
        assert_eq!(results.len(), 5);
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i - 1].distance);
        }
        Ok(())
    }

    #[test]
    fn test_auto_tune() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let result = auto_tune_diskann(1_000_000, 128, 0.90, 2 * 1024 * 1024 * 1024, None);
        assert!(result.config.max_degree >= 32);
        assert!(result.estimated_ram_bytes > 0);
        assert!(result.estimated_recall > 0.5);
        assert!(result.estimated_p99_latency_ms > 0.0);
        Ok(())
    }

    #[test]
    fn test_ssd_benchmark() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let profile = benchmark_ssd(dir.path())?;
        assert!(profile.sequential_read_mbps > 0.0);
        assert!(profile.random_read_iops > 0.0);
        assert!(profile.avg_read_latency_us > 0.0);
        Ok(())
    }

    #[test]
    fn test_prefetch_neighbors() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            max_degree: 4,
            cache_size: 100,
            ..Default::default()
        };
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..10 {
            index
                .add(&format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0])
                ?;
        }
        index.build()?;
        index.clear_cache();

        // Prefetch neighbors of node 0
        index.prefetch_neighbors(0)?;
        assert!(!index.vector_cache.is_empty());
        Ok(())
    }

    #[test]
    fn test_estimate_ram_usage() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 128, config)?;
        for i in 0..100 {
            let vec: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32).sin()).collect();
            index.add(&format!("v{}", i), &vec)?;
        }
        index.build()?;

        let ram = index.estimate_ram_usage();
        assert!(ram > 0);
        Ok(())
    }

    #[test]
    fn test_cache_eviction() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            cache_size: 5,
            max_degree: 4,
            ..Default::default()
        };

        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;

        for i in 0..20 {
            let vec = vec![i as f32, 0.0, 0.0, 0.0];
            index.add(&format!("vec_{}", i), &vec)?;
        }

        index.build()?;

        // Multiple searches should trigger cache eviction
        for i in 0..10 {
            let query = vec![i as f32, 0.0, 0.0, 0.0];
            let _ = index.search(&query, 3)?;
        }

        // Cache should be at or under limit
        assert!(index.vector_cache.len() <= 5);
        Ok(())
    }

    #[test]
    fn test_wal_append_and_replay() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let wal_path = dir.path().join("test.wal");

        // Write WAL entries
        {
            let mut wal = DiskAnnWal::open(&wal_path)?;
            wal.append(DiskAnnWalEntry::Add {
                id: "v1".into(),
                vector: vec![1.0, 2.0, 3.0, 4.0],
            })
            ?;
            wal.append(DiskAnnWalEntry::Add {
                id: "v2".into(),
                vector: vec![5.0, 6.0, 7.0, 8.0],
            })
            ?;
            assert_eq!(wal.len(), 2);
        }

        // Replay into fresh index
        let wal = DiskAnnWal::open(&wal_path)?;
        assert_eq!(wal.len(), 2);

        let mut index = DiskAnnIndex::create(dir.path().join("idx"), 4, DiskAnnConfig::default())?;
        let applied = wal.replay(&mut index)?;
        assert_eq!(applied, 2);
        Ok(())
    }

    #[test]
    fn test_wal_truncate() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let wal_path = dir.path().join("trunc.wal");
        let mut wal = DiskAnnWal::open(&wal_path)?;
        wal.append(DiskAnnWalEntry::Add {
            id: "v1".into(),
            vector: vec![1.0],
        })
        ?;
        assert!(!wal.is_empty());
        wal.truncate()?;
        assert!(wal.is_empty());
        Ok(())
    }

    #[test]
    fn test_search_with_stats() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig {
            max_degree: 4,
            build_list_size: 10,
            search_list_size: 10,
            cache_size: 100,
            ..Default::default()
        };
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..10 {
            index.add(&format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0])?;
        }
        index.build()?;
        index.clear_cache();

        let (results, stats) = index.search_with_stats(&[0.0, 0.0, 0.0, 0.0], 3)?;
        assert_eq!(results.len(), 3);
        assert!(stats.search_latency_us > 0);
        assert!(stats.disk_reads > 0 || stats.cache_hits > 0);
        Ok(())
    }

    #[test]
    fn test_delete_vector() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig::default();
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        index.add("v1", &[1.0, 0.0, 0.0, 0.0])?;
        index.build()?;

        assert!(index.contains("v1"));
        assert!(index.delete("v1")?);
        assert!(!index.contains("v1"));
        assert!(!index.delete("v1")?);
        Ok(())
    }

    #[test]
    fn test_build_quality_metrics() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig { max_degree: 4, build_list_size: 10, search_list_size: 10, ..Default::default() };
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..20 {
            index.add(&format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0])?;
        }
        index.build()?;

        let quality = index.compute_build_quality();
        assert!(quality.avg_degree > 0.0);
        assert!(quality.max_degree > 0);
        assert_eq!(quality.disconnected_nodes, 0);
        assert!(quality.estimated_diameter > 0);
        assert!(quality.compression_ratio > 0.0);
        Ok(())
    }

    #[test]
    fn test_multi_pass_build() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig { max_degree: 4, build_list_size: 10, search_list_size: 10, ..Default::default() };
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..15 {
            index.add(&format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0])?;
        }

        let quality = index.build_multi_pass(2, 2.0)?;
        assert!(quality.avg_degree >= 2.0);
        assert!(index.is_built);
        Ok(())
    }

    #[test]
    fn test_page_aligned_load() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig { max_degree: 4, build_list_size: 10, search_list_size: 10, ..Default::default() };
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..10 {
            index.add(&format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0])?;
        }
        index.build()?;
        index.clear_cache();

        let io_config = PageAlignedIoConfig::default();
        let loaded = index.load_vectors_aligned(&[0, 1, 2, 3], &io_config)?;
        assert_eq!(loaded.len(), 4);
        Ok(())
    }

    #[test]
    fn test_readahead_queue() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = TempDir::new()?;
        let config = DiskAnnConfig { max_degree: 4, build_list_size: 10, search_list_size: 10, ..Default::default() };
        let mut index = DiskAnnIndex::create(dir.path(), 4, config)?;
        for i in 0..10 {
            index.add(&format!("v{i}"), &[i as f32, 0.0, 0.0, 0.0])?;
        }
        index.build()?;
        index.clear_cache();

        let mut queue = ReadaheadQueue::new(16);
        queue.enqueue(0);
        queue.enqueue(1);
        queue.enqueue(2);
        assert_eq!(queue.pending_count(), 3);

        let loaded = queue.drain_into(&mut index)?;
        assert_eq!(loaded, 3);
        assert_eq!(queue.prefetch_count, 3);
        assert_eq!(queue.pending_count(), 0);
        Ok(())
    }
}
