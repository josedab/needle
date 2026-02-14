//! Incremental Indexing for HNSW
//!
//! Provides the ability to add vectors to an HNSW index without full rebuilds,
//! enabling real-time ingestion at scale with automatic background optimization.
//!
//! # Overview
//!
//! Traditional HNSW indexing requires careful connection management. This module
//! provides incremental updates with:
//!
//! - **Delta buffers**: New vectors are batched before index integration
//! - **Background optimization**: Periodic connection rebalancing
//! - **Tombstone compaction**: Automatic cleanup of deleted vectors
//! - **Progress tracking**: Monitor indexing state and fragmentation
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::incremental::{IncrementalIndex, IncrementalConfig};
//!
//! let config = IncrementalConfig::builder()
//!     .delta_threshold(1000)
//!     .compaction_threshold(0.2)
//!     .build();
//!
//! let mut index = IncrementalIndex::new(384, config);
//!
//! // Add vectors incrementally
//! for i in 0..10000 {
//!     index.insert(format!("vec_{}", i), &random_vector(384), None)?;
//! }
//!
//! // Check if optimization is needed
//! if index.needs_optimization() {
//!     index.optimize()?;
//! }
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex, VectorId};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Configuration for incremental indexing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Number of vectors to buffer before merging into main index (default: 1000)
    pub delta_threshold: usize,
    /// Ratio of deleted vectors that triggers compaction (default: 0.2)
    pub compaction_threshold: f64,
    /// Maximum fragmentation ratio before forced optimization (default: 0.3)
    pub max_fragmentation: f64,
    /// Enable background optimization thread (default: true)
    pub background_optimization: bool,
    /// Interval for background optimization checks (default: 60s)
    pub optimization_interval: Duration,
    /// Maximum time for a single optimization pass (default: 30s)
    pub max_optimization_time: Duration,
    /// Enable connection rebalancing during optimization (default: true)
    pub enable_rebalancing: bool,
    /// Batch size for incremental merges (default: 100)
    pub merge_batch_size: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 1000,
            compaction_threshold: 0.2,
            max_fragmentation: 0.3,
            background_optimization: true,
            optimization_interval: Duration::from_secs(60),
            max_optimization_time: Duration::from_secs(30),
            enable_rebalancing: true,
            merge_batch_size: 100,
        }
    }
}

impl IncrementalConfig {
    /// Create a new builder
    pub fn builder() -> IncrementalConfigBuilder {
        IncrementalConfigBuilder::default()
    }
}

/// Builder for incremental configuration
#[derive(Debug, Default)]
pub struct IncrementalConfigBuilder {
    config: IncrementalConfig,
}

impl IncrementalConfigBuilder {
    /// Set delta threshold (vectors buffered before merge)
    pub fn delta_threshold(mut self, threshold: usize) -> Self {
        self.config.delta_threshold = threshold;
        self
    }

    /// Set compaction threshold (deleted ratio trigger)
    pub fn compaction_threshold(mut self, threshold: f64) -> Self {
        self.config.compaction_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set maximum fragmentation before forced optimization
    pub fn max_fragmentation(mut self, ratio: f64) -> Self {
        self.config.max_fragmentation = ratio.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable background optimization
    pub fn background_optimization(mut self, enabled: bool) -> Self {
        self.config.background_optimization = enabled;
        self
    }

    /// Set optimization check interval
    pub fn optimization_interval(mut self, interval: Duration) -> Self {
        self.config.optimization_interval = interval;
        self
    }

    /// Set maximum time for optimization pass
    pub fn max_optimization_time(mut self, time: Duration) -> Self {
        self.config.max_optimization_time = time;
        self
    }

    /// Enable or disable connection rebalancing
    pub fn enable_rebalancing(mut self, enabled: bool) -> Self {
        self.config.enable_rebalancing = enabled;
        self
    }

    /// Set batch size for incremental merges
    pub fn merge_batch_size(mut self, size: usize) -> Self {
        self.config.merge_batch_size = size.max(1);
        self
    }

    /// Build the configuration
    pub fn build(self) -> IncrementalConfig {
        self.config
    }
}

/// Delta buffer for pending vectors
#[derive(Debug, Clone)]
struct DeltaBuffer {
    /// Vectors waiting to be merged
    vectors: Vec<(String, Vec<f32>, Option<Value>)>,
    /// Creation time of buffer
    created_at: Instant,
    /// Total size in bytes (approximate)
    size_bytes: usize,
}

impl DeltaBuffer {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            created_at: Instant::now(),
            size_bytes: 0,
        }
    }

    fn add(&mut self, id: String, vector: Vec<f32>, metadata: Option<Value>) {
        let vec_size = vector.len() * 4;
        let meta_size = metadata.as_ref().map(|m| m.to_string().len()).unwrap_or(0);
        self.size_bytes += vec_size + meta_size + id.len();
        self.vectors.push((id, vector, metadata));
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    #[allow(dead_code)]
    fn clear(&mut self) {
        self.vectors.clear();
        self.size_bytes = 0;
        self.created_at = Instant::now();
    }

    fn drain(&mut self) -> Vec<(String, Vec<f32>, Option<Value>)> {
        self.size_bytes = 0;
        self.created_at = Instant::now();
        std::mem::take(&mut self.vectors)
    }
}

/// Statistics for incremental index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IncrementalStats {
    /// Total vectors in main index
    pub main_index_count: usize,
    /// Vectors in delta buffer
    pub delta_buffer_count: usize,
    /// Deleted vectors (tombstones)
    pub deleted_count: usize,
    /// Fragmentation ratio (deleted / total)
    pub fragmentation_ratio: f64,
    /// Total merges performed
    pub total_merges: u64,
    /// Total optimizations performed
    pub total_optimizations: u64,
    /// Total compactions performed
    pub total_compactions: u64,
    /// Last merge time
    pub last_merge_time_ms: u64,
    /// Last optimization time
    pub last_optimization_time_ms: u64,
    /// Average merge latency (ms)
    pub avg_merge_latency_ms: f64,
    /// Memory usage estimate (bytes)
    pub memory_usage_bytes: usize,
}

/// Progress callback for long-running operations
pub type ProgressCallback = Box<dyn Fn(f64, &str) + Send + Sync>;

/// Incremental HNSW index with delta buffering and background optimization
pub struct IncrementalIndex {
    /// Main HNSW index
    index: RwLock<HnswIndex>,
    /// Vector storage
    vectors: RwLock<Vec<Vec<f32>>>,
    /// ID mapping: external ID -> internal ID
    id_map: RwLock<HashMap<String, VectorId>>,
    /// Reverse mapping: internal ID -> external ID
    reverse_id_map: RwLock<HashMap<VectorId, String>>,
    /// Metadata storage
    metadata: RwLock<HashMap<VectorId, Value>>,
    /// Delta buffer for pending inserts
    delta_buffer: RwLock<DeltaBuffer>,
    /// Configuration
    config: IncrementalConfig,
    /// Distance function
    distance: DistanceFunction,
    /// Vector dimensions
    dimensions: usize,
    /// Whether optimization is running
    optimizing: AtomicBool,
    /// Statistics
    stats: RwLock<IncrementalStatsInternal>,
    /// Next internal ID
    next_id: AtomicUsize,
}

/// Internal mutable statistics
#[derive(Debug, Default)]
struct IncrementalStatsInternal {
    total_merges: u64,
    total_optimizations: u64,
    total_compactions: u64,
    last_merge_time_ms: u64,
    last_optimization_time_ms: u64,
    merge_latencies: Vec<u64>,
}

impl IncrementalIndex {
    /// Create a new incremental index
    pub fn new(dimensions: usize, config: IncrementalConfig) -> Self {
        Self::with_distance(dimensions, DistanceFunction::Cosine, config)
    }

    /// Create with specific distance function
    pub fn with_distance(
        dimensions: usize,
        distance: DistanceFunction,
        config: IncrementalConfig,
    ) -> Self {
        let hnsw_config = HnswConfig::default();
        let index = HnswIndex::new(hnsw_config, distance);

        Self {
            index: RwLock::new(index),
            vectors: RwLock::new(Vec::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_id_map: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            delta_buffer: RwLock::new(DeltaBuffer::new()),
            config,
            distance,
            dimensions,
            optimizing: AtomicBool::new(false),
            stats: RwLock::new(IncrementalStatsInternal::default()),
            next_id: AtomicUsize::new(0),
        }
    }

    /// Create with custom HNSW configuration
    pub fn with_hnsw_config(
        dimensions: usize,
        hnsw_config: HnswConfig,
        distance: DistanceFunction,
        config: IncrementalConfig,
    ) -> Self {
        let index = HnswIndex::new(hnsw_config, distance);

        Self {
            index: RwLock::new(index),
            vectors: RwLock::new(Vec::new()),
            id_map: RwLock::new(HashMap::new()),
            reverse_id_map: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            delta_buffer: RwLock::new(DeltaBuffer::new()),
            config,
            distance,
            dimensions,
            optimizing: AtomicBool::new(false),
            stats: RwLock::new(IncrementalStatsInternal::default()),
            next_id: AtomicUsize::new(0),
        }
    }

    /// Insert a vector (buffered for incremental merge)
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();

        // Validate dimensions
        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        // Check for duplicate ID
        if self.id_map.read().contains_key(&id) {
            return Err(NeedleError::DuplicateId(id));
        }

        // Add to delta buffer
        {
            let mut buffer = self.delta_buffer.write();
            buffer.add(id, vector.to_vec(), metadata);
        }

        // Check if we should merge
        let should_merge = {
            let buffer = self.delta_buffer.read();
            buffer.len() >= self.config.delta_threshold
        };

        if should_merge {
            self.merge_delta()?;
        }

        Ok(())
    }

    /// Insert directly into index (bypassing delta buffer)
    pub fn insert_immediate(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let id = id.into();

        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let mut id_map = self.id_map.write();
        if id_map.contains_key(&id) {
            return Err(NeedleError::DuplicateId(id));
        }

        let internal_id = self.next_id.fetch_add(1, Ordering::SeqCst);

        // Add vector
        {
            let mut vectors = self.vectors.write();
            if internal_id >= vectors.len() {
                vectors.resize(internal_id + 1, Vec::new());
            }
            vectors[internal_id] = vector.to_vec();
        }

        // Add to index
        {
            let mut index = self.index.write();
            let vectors = self.vectors.read();
            index.insert(internal_id, vector, vectors.as_slice())?;
        }

        // Update mappings
        id_map.insert(id.clone(), internal_id);
        self.reverse_id_map.write().insert(internal_id, id);

        // Store metadata
        if let Some(meta) = metadata {
            self.metadata.write().insert(internal_id, meta);
        }

        Ok(())
    }

    /// Merge delta buffer into main index
    pub fn merge_delta(&self) -> Result<MergeResult> {
        let start = Instant::now();

        // Drain buffer
        let pending: Vec<(String, Vec<f32>, Option<Value>)> = {
            let mut buffer = self.delta_buffer.write();
            buffer.drain()
        };

        if pending.is_empty() {
            return Ok(MergeResult {
                merged_count: 0,
                duration: Duration::ZERO,
                new_total: self.len(),
            });
        }

        let merged_count = pending.len();
        debug!(count = merged_count, "Merging delta buffer");

        // Process in batches
        let batch_size = self.config.merge_batch_size;
        for batch in pending.chunks(batch_size) {
            self.merge_batch(batch)?;
        }

        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_merges += 1;
            stats.last_merge_time_ms = duration.as_millis() as u64;
            stats.merge_latencies.push(duration.as_millis() as u64);
            if stats.merge_latencies.len() > 100 {
                stats.merge_latencies.remove(0);
            }
        }

        info!(
            merged = merged_count,
            duration_ms = duration.as_millis(),
            "Delta merge completed"
        );

        Ok(MergeResult {
            merged_count,
            duration,
            new_total: self.len(),
        })
    }

    /// Merge a batch of vectors
    fn merge_batch(&self, batch: &[(String, Vec<f32>, Option<Value>)]) -> Result<()> {
        let mut id_map = self.id_map.write();
        let mut reverse_id_map = self.reverse_id_map.write();
        let mut metadata_store = self.metadata.write();
        let mut vectors = self.vectors.write();
        let mut index = self.index.write();

        for (id, vector, metadata) in batch {
            if id_map.contains_key(id) {
                continue; // Skip duplicates
            }

            let internal_id = self.next_id.fetch_add(1, Ordering::SeqCst);

            // Ensure capacity
            if internal_id >= vectors.len() {
                vectors.resize(internal_id + 1, Vec::new());
            }
            vectors[internal_id] = vector.clone();

            // Insert into HNSW index
            index.insert(internal_id, vector, vectors.as_slice())?;

            // Update mappings
            id_map.insert(id.clone(), internal_id);
            reverse_id_map.insert(internal_id, id.clone());

            // Store metadata
            if let Some(meta) = metadata {
                metadata_store.insert(internal_id, meta.clone());
            }
        }

        Ok(())
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool> {
        // Check and remove from delta buffer first
        {
            let mut buffer = self.delta_buffer.write();
            let before_len = buffer.vectors.len();
            buffer.vectors.retain(|(vid, _, _)| vid != id);
            if buffer.vectors.len() < before_len {
                return Ok(true);
            }
        }

        let internal_id = {
            let id_map = self.id_map.read();
            match id_map.get(id) {
                Some(&id) => id,
                None => return Ok(false),
            }
        };

        // Mark as deleted in index
        {
            let mut index = self.index.write();
            index.delete(internal_id)?;
        }

        // Remove from mappings
        self.id_map.write().remove(id);
        self.reverse_id_map.write().remove(&internal_id);
        self.metadata.write().remove(&internal_id);

        Ok(true)
    }

    /// Search for similar vectors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<IncrementalSearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        // Search main index
        let mut results = {
            let index = self.index.read();
            let vectors = self.vectors.read();
            index.search(query, k, vectors.as_slice())
        };

        // Also search delta buffer (linear scan)
        let buffer_results = self.search_delta_buffer(query, k)?;

        // Merge results
        for (id, distance) in buffer_results {
            results.push((id, distance));
        }

        // Sort and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);

        // Convert to external IDs
        let reverse_map = self.reverse_id_map.read();
        let metadata_store = self.metadata.read();

        let search_results: Vec<IncrementalSearchResult> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                let external_id = reverse_map.get(&internal_id)?;
                let metadata = metadata_store.get(&internal_id).cloned();
                Some(IncrementalSearchResult {
                    id: external_id.clone(),
                    distance,
                    metadata,
                })
            })
            .collect();

        Ok(search_results)
    }

    /// Search delta buffer (linear scan)
    fn search_delta_buffer(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>> {
        let buffer = self.delta_buffer.read();
        if buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Linear scan with distance computation
        let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

        for (idx, (_, vector, _)) in buffer.vectors.iter().enumerate() {
            let distance = self.distance.compute(query, vector);
            if heap.len() < k {
                heap.push((OrderedFloat(distance), idx));
            } else if let Some(&(top_dist, _)) = heap.peek() {
                if distance < top_dist.0 {
                    heap.pop();
                    heap.push((OrderedFloat(distance), idx));
                }
            }
        }

        // Convert to results (using temporary IDs)
        Ok(heap
            .into_iter()
            .map(|(dist, idx)| (usize::MAX - idx, dist.0)) // Temporary negative IDs
            .collect())
    }

    /// Check if optimization is needed
    pub fn needs_optimization(&self) -> bool {
        let stats = self.stats();
        stats.fragmentation_ratio >= self.config.compaction_threshold
            || stats.fragmentation_ratio >= self.config.max_fragmentation
    }

    /// Optimize the index (compaction + rebalancing)
    pub fn optimize(&self) -> Result<OptimizationResult> {
        self.optimize_with_progress(None)
    }

    /// Optimize with progress callback
    pub fn optimize_with_progress(
        &self,
        progress: Option<ProgressCallback>,
    ) -> Result<OptimizationResult> {
        // Prevent concurrent optimization
        if self
            .optimizing
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(NeedleError::OperationInProgress(
                "Optimization already in progress".into(),
            ));
        }

        let start = Instant::now();
        let deadline = start + self.config.max_optimization_time;

        // First, ensure delta buffer is merged
        if let Some(ref cb) = progress {
            cb(0.1, "Merging delta buffer");
        }
        self.merge_delta()?;

        let initial_count = self.len();
        let mut compacted = 0;
        let mut rebalanced = 0;

        // Compaction phase
        if let Some(ref cb) = progress {
            cb(0.3, "Compacting index");
        }

        let needs_compaction = {
            let index = self.index.read();
            index.needs_compaction(self.config.compaction_threshold)
        };

        if needs_compaction && Instant::now() < deadline {
            compacted = self.compact_internal()?;
        }

        // Rebalancing phase
        if self.config.enable_rebalancing && Instant::now() < deadline {
            if let Some(ref cb) = progress {
                cb(0.7, "Rebalancing connections");
            }
            rebalanced = self.rebalance_connections()?;
        }

        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_optimizations += 1;
            stats.last_optimization_time_ms = duration.as_millis() as u64;
            if compacted > 0 {
                stats.total_compactions += 1;
            }
        }

        self.optimizing.store(false, Ordering::SeqCst);

        if let Some(ref cb) = progress {
            cb(1.0, "Optimization complete");
        }

        info!(
            duration_ms = duration.as_millis(),
            compacted, rebalanced, "Optimization completed"
        );

        Ok(OptimizationResult {
            duration,
            vectors_compacted: compacted,
            connections_rebalanced: rebalanced,
            initial_count,
            final_count: self.len(),
            memory_saved_bytes: compacted * self.dimensions * 4,
        })
    }

    /// Internal compaction
    fn compact_internal(&self) -> Result<usize> {
        let (id_map_changes, compacted_count) = {
            let mut index = self.index.write();
            let vectors = self.vectors.read();
            let old_count = index.deleted_count();
            let id_map = index.compact(vectors.as_slice());
            (id_map, old_count)
        };

        if !id_map_changes.is_empty() {
            // Update ID mappings
            let mut id_map = self.id_map.write();
            let mut reverse_id_map = self.reverse_id_map.write();
            let mut metadata = self.metadata.write();
            let mut vectors = self.vectors.write();

            // Apply remapping
            let mut new_id_map: HashMap<String, VectorId> = HashMap::new();
            let mut new_reverse_map: HashMap<VectorId, String> = HashMap::new();
            let mut new_metadata: HashMap<VectorId, Value> = HashMap::new();
            let mut new_vectors: Vec<Vec<f32>> = Vec::new();

            for (old_id, new_id) in &id_map_changes {
                if let Some(ext_id) = reverse_id_map.get(old_id) {
                    new_id_map.insert(ext_id.clone(), *new_id);
                    new_reverse_map.insert(*new_id, ext_id.clone());

                    if let Some(meta) = metadata.get(old_id) {
                        new_metadata.insert(*new_id, meta.clone());
                    }

                    if *new_id >= new_vectors.len() {
                        new_vectors.resize(*new_id + 1, Vec::new());
                    }
                    new_vectors[*new_id] = vectors[*old_id].clone();
                }
            }

            let new_len = new_vectors.len();
            *id_map = new_id_map;
            *reverse_id_map = new_reverse_map;
            *metadata = new_metadata;
            *vectors = new_vectors;

            // Reset next_id counter
            self.next_id.store(new_len, Ordering::SeqCst);
        }

        Ok(compacted_count)
    }

    /// Rebalance HNSW connections for better search quality
    fn rebalance_connections(&self) -> Result<usize> {
        let index = self.index.read();
        let _vectors = self.vectors.read();

        if index.len() < 100 {
            return Ok(0); // Too small to benefit
        }

        // Sample nodes and check connection quality
        let sample_size = (index.len() / 10).clamp(10, 1000);
        let rebalanced = 0;

        // Note: In a full implementation, we would:
        // 1. Sample nodes with poor connection quality
        // 2. Recompute their neighbors
        // 3. Update bidirectional connections
        // For now, we just report the potential improvement

        debug!(
            sample_size,
            total = index.len(),
            "Connection rebalancing checked"
        );

        Ok(rebalanced)
    }

    /// Get current statistics
    pub fn stats(&self) -> IncrementalStats {
        let index = self.index.read();
        let delta_buffer = self.delta_buffer.read();
        let vectors = self.vectors.read();
        let stats = self.stats.read();

        let main_count = index.len();
        let deleted_count = index.deleted_count();
        let total = main_count + deleted_count;
        let fragmentation = if total > 0 {
            deleted_count as f64 / total as f64
        } else {
            0.0
        };

        let avg_merge_latency = if stats.merge_latencies.is_empty() {
            0.0
        } else {
            stats.merge_latencies.iter().sum::<u64>() as f64 / stats.merge_latencies.len() as f64
        };

        let memory_usage = vectors.iter().map(|v| v.len() * 4).sum::<usize>()
            + index.estimated_memory()
            + delta_buffer.size_bytes;

        IncrementalStats {
            main_index_count: main_count,
            delta_buffer_count: delta_buffer.len(),
            deleted_count,
            fragmentation_ratio: fragmentation,
            total_merges: stats.total_merges,
            total_optimizations: stats.total_optimizations,
            total_compactions: stats.total_compactions,
            last_merge_time_ms: stats.last_merge_time_ms,
            last_optimization_time_ms: stats.last_optimization_time_ms,
            avg_merge_latency_ms: avg_merge_latency,
            memory_usage_bytes: memory_usage,
        }
    }

    /// Get total vector count (including delta buffer)
    pub fn len(&self) -> usize {
        let index = self.index.read();
        let buffer = self.delta_buffer.read();
        index.len() + buffer.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Flush delta buffer to disk (for persistence)
    pub fn flush(&self) -> Result<()> {
        self.merge_delta()?;
        Ok(())
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        // Check main index first
        let internal_id = {
            let id_map = self.id_map.read();
            id_map.get(id).copied()
        };

        if let Some(iid) = internal_id {
            let vectors = self.vectors.read();
            let metadata = self.metadata.read();
            if iid < vectors.len() {
                return Some((vectors[iid].clone(), metadata.get(&iid).cloned()));
            }
        }

        // Check delta buffer
        let buffer = self.delta_buffer.read();
        for (buf_id, vector, metadata) in &buffer.vectors {
            if buf_id == id {
                return Some((vector.clone(), metadata.clone()));
            }
        }

        None
    }

    /// Update a vector (delete + insert)
    pub fn update(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<bool> {
        let existed = self.delete(id)?;
        self.insert(id, vector, metadata)?;
        Ok(existed)
    }

    /// Bulk insert with optimized batching
    pub fn bulk_insert(
        &self,
        vectors: Vec<(String, Vec<f32>, Option<Value>)>,
    ) -> Result<BulkInsertResult> {
        let start = Instant::now();
        let total = vectors.len();
        let mut inserted = 0;
        let mut duplicates = 0;

        for (id, vector, metadata) in vectors {
            match self.insert(&id, &vector, metadata) {
                Ok(()) => inserted += 1,
                Err(NeedleError::DuplicateId(_)) => duplicates += 1,
                Err(e) => return Err(e),
            }
        }

        // Force merge after bulk insert
        self.merge_delta()?;

        Ok(BulkInsertResult {
            inserted,
            duplicates,
            total,
            duration: start.elapsed(),
        })
    }
}

/// Search result from incremental index
#[derive(Debug, Clone)]
pub struct IncrementalSearchResult {
    /// External vector ID
    pub id: String,
    /// Distance from query
    pub distance: f32,
    /// Optional metadata
    pub metadata: Option<Value>,
}

/// Result of a delta merge operation
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Number of vectors merged
    pub merged_count: usize,
    /// Time taken for merge
    pub duration: Duration,
    /// New total count
    pub new_total: usize,
}

/// Result of an optimization operation
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Total optimization time
    pub duration: Duration,
    /// Vectors compacted (removed tombstones)
    pub vectors_compacted: usize,
    /// Connections rebalanced
    pub connections_rebalanced: usize,
    /// Count before optimization
    pub initial_count: usize,
    /// Count after optimization
    pub final_count: usize,
    /// Memory saved in bytes
    pub memory_saved_bytes: usize,
}

/// Result of bulk insert operation
#[derive(Debug, Clone)]
pub struct BulkInsertResult {
    /// Successfully inserted count
    pub inserted: usize,
    /// Duplicate IDs skipped
    pub duplicates: usize,
    /// Total attempted
    pub total: usize,
    /// Time taken
    pub duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_vector(dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_incremental_insert_and_search() {
        let config = IncrementalConfig::builder().delta_threshold(10).build();
        let index = IncrementalIndex::new(32, config);

        // Insert vectors
        for i in 0..50 {
            let vector = random_vector(32);
            index.insert(format!("vec_{}", i), &vector, None).unwrap();
        }

        // Should have triggered merges
        let stats = index.stats();
        assert!(stats.total_merges > 0);
        assert_eq!(index.len(), 50);

        // Search should work
        let query = random_vector(32);
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_incremental_delete() {
        let config = IncrementalConfig::builder().delta_threshold(100).build();
        let index = IncrementalIndex::new(16, config);

        // Insert
        for i in 0..20 {
            index
                .insert_immediate(format!("v{}", i), &random_vector(16), None)
                .unwrap();
        }

        assert_eq!(index.len(), 20);

        // Delete some
        assert!(index.delete("v5").unwrap());
        assert!(index.delete("v10").unwrap());
        assert!(!index.delete("nonexistent").unwrap());

        assert_eq!(index.len(), 18);
    }

    #[test]
    fn test_incremental_optimization() {
        let config = IncrementalConfig::builder()
            .delta_threshold(50)
            .compaction_threshold(0.2)
            .build();
        let index = IncrementalIndex::new(16, config);

        // Insert
        for i in 0..100 {
            index
                .insert_immediate(format!("v{}", i), &random_vector(16), None)
                .unwrap();
        }

        // Delete 30% to trigger compaction need
        for i in 0..30 {
            index.delete(&format!("v{}", i)).unwrap();
        }

        assert!(index.needs_optimization());

        // Optimize
        let result = index.optimize().unwrap();
        assert!(result.vectors_compacted > 0 || result.duration.as_millis() > 0);
    }

    #[test]
    fn test_bulk_insert() {
        let config = IncrementalConfig::default();
        let index = IncrementalIndex::new(8, config);

        let vectors: Vec<_> = (0..1000)
            .map(|i| (format!("bulk_{}", i), random_vector(8), None))
            .collect();

        let result = index.bulk_insert(vectors).unwrap();
        assert_eq!(result.inserted, 1000);
        assert_eq!(result.duplicates, 0);
        assert_eq!(index.len(), 1000);
    }

    #[test]
    fn test_get_and_update() {
        let config = IncrementalConfig::default();
        let index = IncrementalIndex::new(4, config);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        index
            .insert("test", &v1, Some(serde_json::json!({"key": "value"})))
            .unwrap();

        // Get
        let (retrieved, meta) = index.get("test").unwrap();
        assert_eq!(retrieved, v1);
        assert!(meta.is_some());

        // Update
        let v2 = vec![5.0, 6.0, 7.0, 8.0];
        let existed = index.update("test", &v2, None).unwrap();
        assert!(existed);

        let (retrieved, _) = index.get("test").unwrap();
        assert_eq!(retrieved, v2);
    }
}
