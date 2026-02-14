//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Optimised Serverless Edge Runtime
//!
//! Minimal-footprint WASM runtime for deploying vector search on serverless
//! edge platforms (Cloudflare Workers, Deno Deploy, Vercel Edge).
//!
//! Goals:
//! - Bundle size <5MB WASM
//! - Cold start <50ms
//! - Sub-5ms search on 100K vectors
//! - Automatic index partitioning for CDN distribution
//!
//! # Architecture
//!
//! ```text
//! CDN Edge Node
//!   ├── PartitionRouter  → routes query to correct partition
//!   ├── CompactIndex     → minimal HNSW with quantised vectors
//!   └── SearchCache      → LRU cache for hot queries
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::Duration;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// Platform Configuration
// ---------------------------------------------------------------------------

/// Target edge platform with its resource constraints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EdgePlatform {
    CloudflareWorkers,
    DenoKv,
    VercelEdge,
    FastlyCompute,
    LambdaEdge,
    GenericWasm,
}

/// Resource limits for an edge platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformLimits {
    /// Maximum memory in bytes.
    pub max_memory_bytes: usize,
    /// Maximum execution time.
    pub max_execution_time: Duration,
    /// Maximum request body size in bytes.
    pub max_request_body: usize,
    /// Maximum response size in bytes.
    pub max_response_size: usize,
    /// Whether KV storage is available.
    pub has_kv_store: bool,
}

impl PlatformLimits {
    /// Get the limits for a known platform.
    pub fn for_platform(platform: &EdgePlatform) -> Self {
        match platform {
            EdgePlatform::CloudflareWorkers => Self {
                max_memory_bytes: 128 * 1024 * 1024,
                max_execution_time: Duration::from_millis(50),
                max_request_body: 100 * 1024 * 1024,
                max_response_size: 100 * 1024 * 1024,
                has_kv_store: true,
            },
            EdgePlatform::DenoKv => Self {
                max_memory_bytes: 512 * 1024 * 1024,
                max_execution_time: Duration::from_secs(5),
                max_request_body: 10 * 1024 * 1024,
                max_response_size: 10 * 1024 * 1024,
                has_kv_store: true,
            },
            EdgePlatform::VercelEdge => Self {
                max_memory_bytes: 128 * 1024 * 1024,
                max_execution_time: Duration::from_millis(25),
                max_request_body: 4 * 1024 * 1024,
                max_response_size: 4 * 1024 * 1024,
                has_kv_store: false,
            },
            EdgePlatform::FastlyCompute => Self {
                max_memory_bytes: 256 * 1024 * 1024,
                max_execution_time: Duration::from_secs(2),
                max_request_body: 50 * 1024 * 1024,
                max_response_size: 50 * 1024 * 1024,
                has_kv_store: true,
            },
            EdgePlatform::LambdaEdge => Self {
                max_memory_bytes: 128 * 1024 * 1024,
                max_execution_time: Duration::from_secs(5),
                max_request_body: 1024 * 1024,
                max_response_size: 1024 * 1024,
                has_kv_store: false,
            },
            EdgePlatform::GenericWasm => Self {
                max_memory_bytes: 256 * 1024 * 1024,
                max_execution_time: Duration::from_secs(30),
                max_request_body: 10 * 1024 * 1024,
                max_response_size: 10 * 1024 * 1024,
                has_kv_store: false,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Compact Index
// ---------------------------------------------------------------------------

/// A minimal, memory-efficient vector index optimised for edge deployment.
/// Uses scalar-quantised vectors (u8) and a simplified HNSW graph.
pub struct CompactIndex {
    dimension: usize,
    distance: DistanceFunction,
    vectors: Vec<CompactVector>,
    graph: Vec<Vec<u32>>, // adjacency list, u32 indices
    entry_point: Option<u32>,
    m: usize,
    ef_search: usize,
}

/// A scalar-quantised vector with ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompactVector {
    id: String,
    quantized: Vec<u8>,
    min_val: f32,
    max_val: f32,
    metadata: Option<serde_json::Value>,
}

impl CompactIndex {
    /// Create a new compact index.
    pub fn new(dimension: usize, distance: DistanceFunction, m: usize, ef_search: usize) -> Self {
        Self {
            dimension,
            distance,
            vectors: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            m,
            ef_search,
        }
    }

    /// Insert a vector with scalar quantisation.
    pub fn insert(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let (quantized, min_val, max_val) = scalar_quantize(vector);

        let idx = self.vectors.len() as u32;
        self.vectors.push(CompactVector {
            id: id.to_string(),
            quantized,
            min_val,
            max_val,
            metadata,
        });

        // Build graph connections (simplified greedy)
        let mut neighbors = Vec::new();
        if !self.graph.is_empty() {
            let mut distances: Vec<(u32, f32)> = self
                .graph
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let dist = self.quantized_distance(idx as usize, i);
                    (i as u32, dist)
                })
                .collect();
            distances.sort_by_key(|(_, d)| OrderedFloat(*d));
            neighbors = distances.iter().take(self.m).map(|(i, _)| *i).collect();
        }

        // Add reverse edges
        for &neighbor in &neighbors {
            if let Some(adj) = self.graph.get_mut(neighbor as usize) {
                if adj.len() < self.m * 2 {
                    adj.push(idx);
                }
            }
        }

        self.graph.push(neighbors);
        if self.entry_point.is_none() {
            self.entry_point = Some(0);
        }

        Ok(())
    }

    /// Search for k nearest neighbours using the compact index.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<EdgeSearchResult>> {
        if query.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let (q_quant, q_min, q_max) = scalar_quantize(query);

        // Greedy beam search on the compact graph
        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        let entry = self.entry_point.unwrap_or(0) as usize;
        let entry_dist = self.quantized_distance_raw(&q_quant, q_min, q_max, entry);
        candidates.push((entry as u32, entry_dist));
        visited[entry] = true;

        let mut results: Vec<(u32, f32)> = vec![(entry as u32, entry_dist)];

        for _ in 0..self.ef_search.min(self.vectors.len()) {
            // Pick best unvisited candidate
            candidates.sort_by_key(|(_, d)| OrderedFloat(*d));
            let current = match candidates.first() {
                Some(&(idx, _)) => {
                    candidates.remove(0);
                    idx
                }
                None => break,
            };

            // Explore neighbors
            if let Some(neighbors) = self.graph.get(current as usize) {
                for &neighbor in neighbors {
                    let n = neighbor as usize;
                    if n < visited.len() && !visited[n] {
                        visited[n] = true;
                        let dist = self.quantized_distance_raw(&q_quant, q_min, q_max, n);
                        candidates.push((neighbor, dist));
                        results.push((neighbor, dist));
                    }
                }
            }
        }

        results.sort_by_key(|(_, d)| OrderedFloat(*d));
        results.truncate(k);

        Ok(results
            .into_iter()
            .filter_map(|(idx, dist)| {
                self.vectors.get(idx as usize).map(|v| EdgeSearchResult {
                    id: v.id.clone(),
                    distance: dist,
                    metadata: v.metadata.clone(),
                })
            })
            .collect())
    }

    fn quantized_distance(&self, a: usize, b: usize) -> f32 {
        let va = &self.vectors[a];
        let vb = &self.vectors[b];
        quantized_cosine_distance(
            &va.quantized,
            va.min_val,
            va.max_val,
            &vb.quantized,
            vb.min_val,
            vb.max_val,
        )
    }

    fn quantized_distance_raw(&self, q: &[u8], q_min: f32, q_max: f32, b: usize) -> f32 {
        let vb = &self.vectors[b];
        quantized_cosine_distance(q, q_min, q_max, &vb.quantized, vb.min_val, vb.max_val)
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let vec_mem = self.vectors.len() * (self.dimension + 32); // quantized + overhead
        let graph_mem = self.graph.iter().map(|adj| adj.len() * 4).sum::<usize>();
        vec_mem + graph_mem
    }
}

/// Search result from the edge runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Quantisation Helpers
// ---------------------------------------------------------------------------

/// Scalar-quantize a float vector to u8 (0-255 range).
fn scalar_quantize(vector: &[f32]) -> (Vec<u8>, f32, f32) {
    let min_val = vector.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = vector.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(1e-10);

    let quantized: Vec<u8> = vector
        .iter()
        .map(|&v| ((v - min_val) / range * 255.0).round() as u8)
        .collect();

    (quantized, min_val, max_val)
}

/// Approximate cosine distance between two quantised vectors.
fn quantized_cosine_distance(
    a: &[u8],
    a_min: f32,
    a_max: f32,
    b: &[u8],
    b_min: f32,
    b_max: f32,
) -> f32 {
    let a_range = (a_max - a_min).max(1e-10);
    let b_range = (b_max - b_min).max(1e-10);

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (&qa, &qb) in a.iter().zip(b.iter()) {
        let va = (qa as f32 / 255.0) * a_range + a_min;
        let vb = (qb as f32 / 255.0) * b_range + b_min;
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    1.0 - (dot / denom)
}

// ---------------------------------------------------------------------------
// Partition Manager
// ---------------------------------------------------------------------------

/// Configuration for index partitioning across CDN edge nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Maximum vectors per partition.
    pub max_vectors_per_partition: usize,
    /// Maximum memory per partition in bytes.
    pub max_memory_per_partition: usize,
    /// Number of overlap vectors between partitions for recall.
    pub overlap_count: usize,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            max_vectors_per_partition: 50_000,
            max_memory_per_partition: 64 * 1024 * 1024, // 64MB
            overlap_count: 100,
        }
    }
}

/// A partition of the index for edge deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionManifest {
    pub partition_id: String,
    pub num_vectors: usize,
    pub dimension: usize,
    pub memory_bytes: usize,
    pub region: Option<String>,
    pub vector_ids: Vec<String>,
}

/// Manages partitioning of a vector index for CDN distribution.
pub struct EdgePartitionManager {
    config: PartitionConfig,
    partitions: Vec<PartitionManifest>,
}

impl EdgePartitionManager {
    pub fn new(config: PartitionConfig) -> Self {
        Self {
            config,
            partitions: Vec::new(),
        }
    }

    /// Compute partitioning for a set of vectors.
    pub fn compute_partitions(
        &mut self,
        ids: &[String],
        dimension: usize,
    ) -> Vec<PartitionManifest> {
        let bytes_per_vec = dimension + 32; // quantised + overhead
        let max_by_memory = self.config.max_memory_per_partition / bytes_per_vec.max(1);
        let partition_size = self
            .config
            .max_vectors_per_partition
            .min(max_by_memory)
            .max(1);

        let mut partitions = Vec::new();
        let mut partition_idx = 0;

        for chunk in ids.chunks(partition_size) {
            let manifest = PartitionManifest {
                partition_id: format!("partition-{}", partition_idx),
                num_vectors: chunk.len(),
                dimension,
                memory_bytes: chunk.len() * bytes_per_vec,
                region: None,
                vector_ids: chunk.to_vec(),
            };
            partitions.push(manifest);
            partition_idx += 1;
        }

        self.partitions = partitions.clone();
        partitions
    }

    /// Get the partition containing a specific vector ID.
    pub fn find_partition(&self, vector_id: &str) -> Option<&PartitionManifest> {
        self.partitions
            .iter()
            .find(|p| p.vector_ids.contains(&vector_id.to_string()))
    }

    /// Total number of partitions.
    pub fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    /// List all partition manifests.
    pub fn manifests(&self) -> &[PartitionManifest] {
        &self.partitions
    }
}

// ---------------------------------------------------------------------------
// Edge Runtime Stats
// ---------------------------------------------------------------------------

/// Runtime metrics for the edge deployment.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EdgeRuntimeStats {
    pub searches: u64,
    pub avg_latency_us: u64,
    pub p99_latency_us: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_used_bytes: usize,
    pub vectors_loaded: usize,
}

// ---------------------------------------------------------------------------
// Edge Persistence Layer
// ---------------------------------------------------------------------------

/// A storage backend trait for edge persistence (IndexedDB, KV stores, etc.).
/// The in-memory implementation is used for testing and non-WASM targets.
pub trait EdgePersistence: Send + Sync {
    /// Store a chunk of data under a key.
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;
    /// Retrieve data by key.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    /// Delete data by key.
    fn delete(&self, key: &str) -> Result<bool>;
    /// List all keys with a given prefix.
    fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
    /// Get total storage used in bytes.
    fn storage_used(&self) -> usize;
}

/// In-memory persistence backend for testing.
pub struct InMemoryPersistence {
    data: parking_lot::RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryPersistence {
    pub fn new() -> Self {
        Self {
            data: parking_lot::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryPersistence {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgePersistence for InMemoryPersistence {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.data.write().insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.read().get(key).cloned())
    }

    fn delete(&self, key: &str) -> Result<bool> {
        Ok(self.data.write().remove(key).is_some())
    }

    fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self
            .data
            .read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn storage_used(&self) -> usize {
        self.data.read().values().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Offline Search Cache
// ---------------------------------------------------------------------------

/// Configuration for the offline-first search cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineCacheConfig {
    /// Max number of cached queries.
    pub max_entries: usize,
    /// Max age of cached results in seconds.
    pub ttl_secs: u64,
}

impl Default for OfflineCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_secs: 3600,
        }
    }
}

/// Cached search result entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedQuery {
    query_hash: u64,
    results: Vec<EdgeSearchResult>,
    timestamp: u64,
}

/// Offline-first search cache that stores recent queries for instant offline retrieval.
pub struct OfflineSearchCache {
    config: OfflineCacheConfig,
    entries: parking_lot::RwLock<VecDeque<CachedQuery>>,
}

impl OfflineSearchCache {
    pub fn new(config: OfflineCacheConfig) -> Self {
        Self {
            config,
            entries: parking_lot::RwLock::new(VecDeque::new()),
        }
    }

    /// Store a query result in the cache.
    pub fn put(&self, query: &[f32], results: Vec<EdgeSearchResult>) {
        let hash = hash_query(query);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut entries = self.entries.write();

        // Evict expired and over-capacity entries
        entries.retain(|e| now - e.timestamp < self.config.ttl_secs);
        while entries.len() >= self.config.max_entries {
            entries.pop_front();
        }

        entries.push_back(CachedQuery {
            query_hash: hash,
            results,
            timestamp: now,
        });
    }

    /// Look up a cached result for a query.
    pub fn get(&self, query: &[f32]) -> Option<Vec<EdgeSearchResult>> {
        let hash = hash_query(query);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entries = self.entries.read();
        entries
            .iter()
            .rev()
            .find(|e| e.query_hash == hash && now - e.timestamp < self.config.ttl_secs)
            .map(|e| e.results.clone())
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.entries.write().clear();
    }
}

fn hash_query(query: &[f32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for &v in query {
        v.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// CompactIndex Serialization
// ---------------------------------------------------------------------------

impl CompactIndex {
    /// Serialize the index to bytes for persistence.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(&self.to_snapshot()).unwrap_or_default()
    }

    /// Deserialize an index from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let snap: CompactIndexSnapshot = serde_json::from_slice(data).map_err(|e| {
            NeedleError::InvalidInput(format!("Failed to deserialize index: {}", e))
        })?;
        Ok(snap.into_index())
    }

    fn to_snapshot(&self) -> CompactIndexSnapshot {
        CompactIndexSnapshot {
            dimension: self.dimension,
            m: self.m,
            ef_search: self.ef_search,
            vectors: self.vectors.clone(),
            graph: self.graph.clone(),
            entry_point: self.entry_point,
        }
    }

    /// Save the index to a persistence backend.
    pub fn save_to(&self, store: &dyn EdgePersistence, key: &str) -> Result<()> {
        let data = self.to_bytes();
        store.put(key, &data)
    }

    /// Load the index from a persistence backend.
    pub fn load_from(store: &dyn EdgePersistence, key: &str) -> Result<Option<Self>> {
        match store.get(key)? {
            Some(data) => Ok(Some(Self::from_bytes(&data)?)),
            None => Ok(None),
        }
    }
}

/// Serializable snapshot of a CompactIndex.
#[derive(Serialize, Deserialize)]
struct CompactIndexSnapshot {
    dimension: usize,
    m: usize,
    ef_search: usize,
    vectors: Vec<CompactVector>,
    graph: Vec<Vec<u32>>,
    entry_point: Option<u32>,
}

impl CompactIndexSnapshot {
    fn into_index(self) -> CompactIndex {
        CompactIndex {
            dimension: self.dimension,
            distance: DistanceFunction::Cosine, // default; serialization doesn't track distance
            vectors: self.vectors,
            graph: self.graph,
            entry_point: self.entry_point,
            m: self.m,
            ef_search: self.ef_search,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_platform_limits() {
        let limits = PlatformLimits::for_platform(&EdgePlatform::CloudflareWorkers);
        assert_eq!(limits.max_memory_bytes, 128 * 1024 * 1024);
        assert!(limits.has_kv_store);

        let vercel = PlatformLimits::for_platform(&EdgePlatform::VercelEdge);
        assert!(!vercel.has_kv_store);
    }

    #[test]
    fn test_scalar_quantize() {
        let v = vec![0.0, 0.5, 1.0];
        let (q, min, max) = scalar_quantize(&v);
        assert_eq!(q.len(), 3);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 128); // ~half
        assert_eq!(q[2], 255);
        assert!((min - 0.0).abs() < 0.001);
        assert!((max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_quantized_cosine_same_vector() {
        let v = vec![1.0, 2.0, 3.0];
        let (q, min, max) = scalar_quantize(&v);
        let dist = quantized_cosine_distance(&q, min, max, &q, min, max);
        assert!(
            dist < 0.01,
            "Same vector should have ~0 distance, got {}",
            dist
        );
    }

    #[test]
    fn test_compact_index_insert_and_search() {
        let mut index = CompactIndex::new(4, DistanceFunction::Cosine, 8, 50);

        for i in 0..50 {
            index
                .insert(&format!("v{}", i), &random_vector(4), None)
                .unwrap();
        }

        assert_eq!(index.len(), 50);
        assert!(!index.is_empty());

        let results = index.search(&random_vector(4), 5).unwrap();
        assert!(results.len() <= 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_compact_index_with_metadata() {
        let mut index = CompactIndex::new(3, DistanceFunction::Cosine, 4, 20);
        index
            .insert("v1", &[1.0, 0.0, 0.0], Some(json!({"label": "x"})))
            .unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
        assert!(results[0].metadata.is_some());
    }

    #[test]
    fn test_compact_index_dimension_check() {
        let mut index = CompactIndex::new(4, DistanceFunction::Cosine, 4, 20);
        assert!(index.insert("bad", &[1.0, 2.0], None).is_err());
    }

    #[test]
    fn test_compact_index_empty_search() {
        let index = CompactIndex::new(4, DistanceFunction::Cosine, 4, 20);
        let results = index.search(&random_vector(4), 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_compact_index_memory_usage() {
        let mut index = CompactIndex::new(128, DistanceFunction::Cosine, 8, 50);
        for i in 0..100 {
            index
                .insert(&format!("v{}", i), &random_vector(128), None)
                .unwrap();
        }
        let mem = index.memory_usage();
        assert!(mem > 0);
        assert!(mem < 1024 * 1024); // Should be very compact
    }

    #[test]
    fn test_partition_manager() {
        let config = PartitionConfig {
            max_vectors_per_partition: 10,
            max_memory_per_partition: 64 * 1024 * 1024,
            overlap_count: 0,
        };
        let mut manager = EdgePartitionManager::new(config);

        let ids: Vec<String> = (0..25).map(|i| format!("v{}", i)).collect();
        let partitions = manager.compute_partitions(&ids, 128);

        assert_eq!(partitions.len(), 3); // 10 + 10 + 5
        assert_eq!(partitions[0].num_vectors, 10);
        assert_eq!(partitions[2].num_vectors, 5);
    }

    #[test]
    fn test_partition_find() {
        let config = PartitionConfig {
            max_vectors_per_partition: 5,
            ..Default::default()
        };
        let mut manager = EdgePartitionManager::new(config);

        let ids: Vec<String> = (0..12).map(|i| format!("v{}", i)).collect();
        manager.compute_partitions(&ids, 64);

        let partition = manager.find_partition("v7");
        assert!(partition.is_some());
        assert_eq!(partition.unwrap().partition_id, "partition-1");
    }

    #[test]
    fn test_partition_count() {
        let mut manager = EdgePartitionManager::new(PartitionConfig::default());
        assert_eq!(manager.partition_count(), 0);

        let ids: Vec<String> = (0..100).map(|i| format!("v{}", i)).collect();
        manager.compute_partitions(&ids, 384);
        assert!(manager.partition_count() >= 1);
    }

    #[test]
    fn test_edge_platform_serialization() {
        let platforms = vec![
            EdgePlatform::CloudflareWorkers,
            EdgePlatform::DenoKv,
            EdgePlatform::VercelEdge,
            EdgePlatform::FastlyCompute,
            EdgePlatform::LambdaEdge,
            EdgePlatform::GenericWasm,
        ];
        for p in &platforms {
            let json = serde_json::to_string(p).unwrap();
            let decoded: EdgePlatform = serde_json::from_str(&json).unwrap();
            assert_eq!(*p, decoded);
        }
    }

    #[test]
    fn test_edge_search_result_serialization() {
        let result = EdgeSearchResult {
            id: "v1".into(),
            distance: 0.15,
            metadata: Some(json!({"key": "val"})),
        };
        let json = serde_json::to_string(&result).unwrap();
        let decoded: EdgeSearchResult = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "v1");
    }

    #[test]
    fn test_edge_runtime_stats_default() {
        let stats = EdgeRuntimeStats::default();
        assert_eq!(stats.searches, 0);
        assert_eq!(stats.memory_used_bytes, 0);
    }

    #[test]
    fn test_in_memory_persistence() {
        let store = InMemoryPersistence::new();
        store.put("key1", b"hello").unwrap();
        store.put("key2", b"world").unwrap();

        assert_eq!(store.get("key1").unwrap().unwrap(), b"hello");
        assert_eq!(store.storage_used(), 10);

        let keys = store.list_keys("key").unwrap();
        assert_eq!(keys.len(), 2);

        assert!(store.delete("key1").unwrap());
        assert!(store.get("key1").unwrap().is_none());
        assert!(!store.delete("nonexistent").unwrap());
    }

    #[test]
    fn test_compact_index_serialization() {
        let mut index = CompactIndex::new(4, DistanceFunction::Cosine, 4, 20);
        index.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        index.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let bytes = index.to_bytes();
        assert!(!bytes.is_empty());

        let restored = CompactIndex::from_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 2);

        let results = restored.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_compact_index_persistence_roundtrip() {
        let store = InMemoryPersistence::new();
        let mut index = CompactIndex::new(3, DistanceFunction::Cosine, 4, 20);
        index
            .insert("a", &[1.0, 0.0, 0.0], Some(json!({"tag": "x"})))
            .unwrap();

        index.save_to(&store, "my_index").unwrap();

        let loaded = CompactIndex::load_from(&store, "my_index")
            .unwrap()
            .unwrap();
        assert_eq!(loaded.len(), 1);

        assert!(CompactIndex::load_from(&store, "nonexistent")
            .unwrap()
            .is_none());
    }

    #[test]
    fn test_offline_search_cache() {
        let cache = OfflineSearchCache::new(OfflineCacheConfig::default());
        assert!(cache.is_empty());

        let query = vec![1.0, 0.0, 0.0];
        let results = vec![EdgeSearchResult {
            id: "v1".into(),
            distance: 0.1,
            metadata: None,
        }];

        cache.put(&query, results.clone());
        assert_eq!(cache.len(), 1);

        let cached = cache.get(&query).unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].id, "v1");

        // Different query should miss
        assert!(cache.get(&[0.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn test_offline_cache_eviction() {
        let config = OfflineCacheConfig {
            max_entries: 3,
            ttl_secs: 3600,
        };
        let cache = OfflineSearchCache::new(config);

        for i in 0..5 {
            cache.put(&[i as f32], vec![]);
        }

        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_offline_cache_clear() {
        let cache = OfflineSearchCache::new(OfflineCacheConfig::default());
        cache.put(&[1.0], vec![]);
        cache.put(&[2.0], vec![]);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert!(cache.is_empty());
    }
}
