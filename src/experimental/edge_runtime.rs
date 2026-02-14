//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Serverless Edge Runtime
//!
//! Provides optimized WASM-based runtime for serverless edge environments like
//! Cloudflare Workers, Deno Deploy, and Vercel Edge Functions. This module
//! focuses on minimal bundle size, fast cold starts, and efficient memory usage.
//!
//! # Features
//!
//! - **Compact Binary**: Optimized for <500KB gzipped bundle size
//! - **Streaming Index**: Lazy loading of index segments for reduced memory
//! - **Platform Adapters**: Native support for edge platforms (Workers, Deno, Vercel)
//! - **Cold Start Optimization**: Sub-50ms initialization
//! - **Data Partitioning**: Automatic sharding for edge storage limits
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::edge_runtime::{EdgeRuntime, EdgeConfig, Platform};
//!
//! // Initialize with platform-specific adapter
//! let runtime = EdgeRuntime::new(EdgeConfig::for_platform(Platform::CloudflareWorkers));
//!
//! // Load collection from edge storage
//! runtime.load_collection("vectors", storage_backend).await?;
//!
//! // Search with minimal memory footprint
//! let results = runtime.search(&query_vector, 10).await?;
//! ```

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::HnswConfig;
use crate::metadata::Filter;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ============================================================================
// Platform Detection and Configuration
// ============================================================================

/// Supported edge platforms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Platform {
    /// Cloudflare Workers with KV/R2 storage
    CloudflareWorkers,
    /// Deno Deploy with KV storage
    DenoKv,
    /// Vercel Edge Functions
    VercelEdge,
    /// Fastly Compute@Edge
    FastlyCompute,
    /// AWS Lambda@Edge
    LambdaEdge,
    /// Generic WASM runtime
    GenericWasm,
}

impl Platform {
    /// Get platform-specific memory limit in bytes
    pub fn memory_limit(&self) -> usize {
        match self {
            Platform::CloudflareWorkers => 128 * 1024 * 1024, // 128 MB
            Platform::DenoKv => 512 * 1024 * 1024,            // 512 MB
            Platform::VercelEdge => 128 * 1024 * 1024,        // 128 MB
            Platform::FastlyCompute => 128 * 1024 * 1024,     // 128 MB
            Platform::LambdaEdge => 128 * 1024 * 1024,        // 128 MB
            Platform::GenericWasm => 256 * 1024 * 1024,       // 256 MB
        }
    }

    /// Get platform-specific max execution time in ms
    pub fn max_execution_time_ms(&self) -> u64 {
        match self {
            Platform::CloudflareWorkers => 30_000, // 30s (paid)
            Platform::DenoKv => 60_000,            // 60s
            Platform::VercelEdge => 30_000,        // 30s
            Platform::FastlyCompute => 60_000,     // 60s
            Platform::LambdaEdge => 5_000,         // 5s (viewer request)
            Platform::GenericWasm => 120_000,      // 2 min
        }
    }

    /// Get platform-specific max storage size per key in bytes
    pub fn max_storage_value_size(&self) -> usize {
        match self {
            Platform::CloudflareWorkers => 25 * 1024 * 1024, // 25 MB (R2)
            Platform::DenoKv => 64 * 1024,                   // 64 KB
            Platform::VercelEdge => 4 * 1024 * 1024,         // 4 MB
            Platform::FastlyCompute => 8 * 1024 * 1024,      // 8 MB
            Platform::LambdaEdge => 40 * 1024,               // 40 KB (response)
            Platform::GenericWasm => 100 * 1024 * 1024,      // 100 MB
        }
    }

    /// Check if platform supports streaming responses
    pub fn supports_streaming(&self) -> bool {
        matches!(
            self,
            Platform::CloudflareWorkers | Platform::DenoKv | Platform::VercelEdge
        )
    }

    /// Detect current platform from environment
    pub fn detect() -> Self {
        // In WASM, we can't reliably detect platform, so default to generic
        Platform::GenericWasm
    }
}

/// Edge runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// Target platform
    pub platform: Platform,
    /// Maximum vectors to keep in memory
    pub max_vectors_in_memory: usize,
    /// Enable lazy index loading
    pub lazy_loading: bool,
    /// Segment size for index partitioning (number of vectors)
    pub segment_size: usize,
    /// Enable compression for storage
    pub compress_storage: bool,
    /// Use quantization to reduce memory
    pub use_quantization: bool,
    /// HNSW parameters optimized for edge
    pub hnsw_config: EdgeHnswConfig,
    /// Warmup vectors to preload on cold start
    pub warmup_vectors: usize,
    /// Cache configuration
    pub cache_config: EdgeCacheConfig,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self::for_platform(Platform::GenericWasm)
    }
}

impl EdgeConfig {
    /// Create configuration optimized for a specific platform
    pub fn for_platform(platform: Platform) -> Self {
        let memory_limit = platform.memory_limit();

        // Calculate optimal vector count based on memory
        // Assume 384-dimensional vectors (common for small models)
        // Each vector ~1.5KB (384 * 4 bytes + overhead)
        let vectors_per_mb = 1024 * 1024 / 1536;
        let max_vectors = (memory_limit / 1024 / 1024 / 2) * vectors_per_mb; // Use half memory

        Self {
            platform,
            max_vectors_in_memory: max_vectors.min(100_000),
            lazy_loading: true,
            segment_size: 10_000,
            compress_storage: true,
            use_quantization: memory_limit < 256 * 1024 * 1024,
            hnsw_config: EdgeHnswConfig::for_platform(platform),
            warmup_vectors: 100,
            cache_config: EdgeCacheConfig::default(),
        }
    }

    /// Builder: set maximum vectors in memory
    pub fn with_max_vectors(mut self, count: usize) -> Self {
        self.max_vectors_in_memory = count;
        self
    }

    /// Builder: enable/disable lazy loading
    pub fn with_lazy_loading(mut self, enabled: bool) -> Self {
        self.lazy_loading = enabled;
        self
    }

    /// Builder: set segment size
    pub fn with_segment_size(mut self, size: usize) -> Self {
        self.segment_size = size;
        self
    }
}

/// HNSW configuration optimized for edge environments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeHnswConfig {
    /// Number of connections per node
    pub m: usize,
    /// Size of dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of dynamic candidate list during search
    pub ef_search: usize,
}

impl Default for EdgeHnswConfig {
    fn default() -> Self {
        Self {
            m: 12,                // Reduced from 16 for memory
            ef_construction: 100, // Reduced from 200
            ef_search: 30,        // Reduced from 50 for speed
        }
    }
}

impl EdgeHnswConfig {
    /// Create optimized config for platform
    pub fn for_platform(platform: Platform) -> Self {
        match platform {
            Platform::LambdaEdge => Self {
                m: 8,
                ef_construction: 50,
                ef_search: 20,
            },
            Platform::CloudflareWorkers | Platform::VercelEdge => Self {
                m: 12,
                ef_construction: 100,
                ef_search: 30,
            },
            _ => Self::default(),
        }
    }

    /// Convert to standard HNSW config
    pub fn to_hnsw_config(&self) -> HnswConfig {
        HnswConfig::default()
            .m(self.m)
            .ef_construction(self.ef_construction)
            .ef_search(self.ef_search)
    }
}

/// Cache configuration for edge runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCacheConfig {
    /// Maximum cached search results
    pub max_cached_results: usize,
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
    /// Enable semantic caching (cache similar queries)
    pub semantic_caching: bool,
    /// Similarity threshold for semantic cache hits
    pub semantic_threshold: f32,
}

impl Default for EdgeCacheConfig {
    fn default() -> Self {
        Self {
            max_cached_results: 100,
            ttl_seconds: 60,
            semantic_caching: true,
            semantic_threshold: 0.95,
        }
    }
}

// ============================================================================
// Index Segment for Partitioned Loading
// ============================================================================

/// Metadata for an index segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetadata {
    /// Segment ID
    pub id: u32,
    /// Number of vectors in segment
    pub vector_count: usize,
    /// Starting vector ID offset
    pub id_offset: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Uncompressed size in bytes
    pub uncompressed_size: usize,
    /// Checksum for integrity
    pub checksum: u64,
    /// HNSW layer info for this segment
    pub max_layer: usize,
}

/// A segment of the index that can be loaded independently
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSegment {
    /// Segment metadata
    pub metadata: SegmentMetadata,
    /// Vector IDs in this segment
    pub vector_ids: Vec<String>,
    /// Vector data (may be quantized)
    pub vectors: Vec<Vec<f32>>,
    /// HNSW graph edges for this segment
    pub edges: HashMap<usize, Vec<Vec<usize>>>,
    /// Optional metadata for vectors
    #[serde(default)]
    pub vector_metadata: HashMap<String, serde_json::Value>,
}

impl IndexSegment {
    /// Create a new empty segment
    pub fn new(id: u32, id_offset: usize) -> Self {
        Self {
            metadata: SegmentMetadata {
                id,
                vector_count: 0,
                id_offset,
                compressed_size: 0,
                uncompressed_size: 0,
                checksum: 0,
                max_layer: 0,
            },
            vector_ids: Vec::new(),
            vectors: Vec::new(),
            edges: HashMap::new(),
            vector_metadata: HashMap::new(),
        }
    }

    /// Add a vector to the segment
    pub fn add_vector(
        &mut self,
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) {
        self.vector_ids.push(id.clone());
        self.vectors.push(vector);
        if let Some(meta) = metadata {
            self.vector_metadata.insert(id, meta);
        }
        self.metadata.vector_count += 1;
    }

    /// Check if segment is full
    pub fn is_full(&self, max_size: usize) -> bool {
        self.metadata.vector_count >= max_size
    }

    /// Serialize segment to bytes (using JSON to support serde_json::Value metadata)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(NeedleError::Serialization)
    }

    /// Deserialize segment from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(NeedleError::Serialization)
    }

    /// Compress segment data (placeholder - returns uncompressed bytes)
    #[allow(dead_code)]
    pub fn compress(&self) -> Result<Vec<u8>> {
        let bytes = self.to_bytes()?;
        // TODO: Use LZ4 or similar compression when compression feature is implemented
        Ok(bytes)
    }

    /// Decompress segment data (placeholder - assumes uncompressed bytes)
    #[allow(dead_code)]
    pub fn decompress(bytes: &[u8]) -> Result<Self> {
        // TODO: Add decompression when compression feature is implemented
        Self::from_bytes(bytes)
    }
}

/// Collection manifest for edge storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeManifest {
    /// Collection name
    pub name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function
    pub distance: String,
    /// Total vector count
    pub total_vectors: usize,
    /// Segment metadata list
    pub segments: Vec<SegmentMetadata>,
    /// Creation timestamp
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: u64,
    /// HNSW configuration used
    pub hnsw_config: EdgeHnswConfig,
    /// Entry point for HNSW graph
    pub entry_point: Option<usize>,
}

impl EdgeManifest {
    /// Create a new manifest
    pub fn new(name: &str, dimensions: usize, distance: DistanceFunction) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            name: name.to_string(),
            dimensions,
            distance: format!("{:?}", distance),
            total_vectors: 0,
            segments: Vec::new(),
            created_at: now,
            updated_at: now,
            hnsw_config: EdgeHnswConfig::default(),
            entry_point: None,
        }
    }

    /// Add segment metadata
    pub fn add_segment(&mut self, segment: SegmentMetadata) {
        self.total_vectors += segment.vector_count;
        self.segments.push(segment);
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
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

// ============================================================================
// Edge Storage Backend Trait
// ============================================================================

/// Storage backend for edge platforms
pub trait EdgeStorage: Send + Sync {
    /// Get a value by key
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Put a value
    fn put(&self, key: &str, value: &[u8]) -> Result<()>;

    /// Delete a value
    fn delete(&self, key: &str) -> Result<()>;

    /// List keys with prefix
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Check if key exists
    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }
}

/// In-memory storage for testing and simple use cases
#[derive(Default)]
pub struct InMemoryEdgeStorage {
    data: parking_lot::RwLock<HashMap<String, Vec<u8>>>,
}

impl InMemoryEdgeStorage {
    /// Create new in-memory storage
    pub fn new() -> Self {
        Self::default()
    }
}

impl EdgeStorage for InMemoryEdgeStorage {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.read().get(key).cloned())
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        self.data.write().insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.data.write().remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self
            .data
            .read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }
}

// ============================================================================
// Search Result Cache
// ============================================================================

/// Cached search result with expiration
#[derive(Clone)]
#[allow(dead_code)]
struct CachedResult {
    results: Vec<SearchResult>,
    query_hash: u64,
    created_at: std::time::Instant,
}

/// LRU cache for search results
pub struct SearchCache {
    entries: parking_lot::RwLock<HashMap<u64, CachedResult>>,
    max_entries: usize,
    ttl: std::time::Duration,
    /// Optional: cached query vectors for semantic matching
    query_vectors: parking_lot::RwLock<HashMap<u64, Vec<f32>>>,
    semantic_threshold: f32,
}

impl SearchCache {
    /// Create a new search cache
    pub fn new(config: &EdgeCacheConfig) -> Self {
        Self {
            entries: parking_lot::RwLock::new(HashMap::new()),
            max_entries: config.max_cached_results,
            ttl: std::time::Duration::from_secs(config.ttl_seconds),
            query_vectors: parking_lot::RwLock::new(HashMap::new()),
            semantic_threshold: config.semantic_threshold,
        }
    }

    /// Compute hash for a query vector
    fn hash_query(query: &[f32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &v in query {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Get cached result for exact query match
    pub fn get_exact(&self, query: &[f32], k: usize) -> Option<Vec<SearchResult>> {
        let hash = Self::hash_query(query);
        let entries = self.entries.read();

        if let Some(cached) = entries.get(&hash) {
            if cached.created_at.elapsed() <= self.ttl && cached.results.len() >= k {
                return Some(cached.results.iter().take(k).cloned().collect());
            }
        }
        None
    }

    /// Get cached result with semantic matching
    pub fn get_semantic(&self, query: &[f32], k: usize) -> Option<Vec<SearchResult>> {
        // First try exact match
        if let Some(results) = self.get_exact(query, k) {
            return Some(results);
        }

        // Try semantic match
        let query_vectors = self.query_vectors.read();
        let entries = self.entries.read();

        for (hash, cached_query) in query_vectors.iter() {
            // Compute cosine similarity
            let similarity = cosine_similarity(query, cached_query);
            if similarity >= self.semantic_threshold {
                if let Some(cached) = entries.get(hash) {
                    if cached.created_at.elapsed() <= self.ttl && cached.results.len() >= k {
                        return Some(cached.results.iter().take(k).cloned().collect());
                    }
                }
            }
        }
        None
    }

    /// Store result in cache
    pub fn put(&self, query: &[f32], results: Vec<SearchResult>) {
        let hash = Self::hash_query(query);

        let mut entries = self.entries.write();

        // Evict oldest if at capacity
        while entries.len() >= self.max_entries {
            let oldest = entries
                .iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| *k);
            if let Some(key) = oldest {
                entries.remove(&key);
                self.query_vectors.write().remove(&key);
            } else {
                break;
            }
        }

        entries.insert(
            hash,
            CachedResult {
                results,
                query_hash: hash,
                created_at: std::time::Instant::now(),
            },
        );

        // Store query vector for semantic matching
        self.query_vectors.write().insert(hash, query.to_vec());
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        self.entries.write().clear();
        self.query_vectors.write().clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let entries = self.entries.read();
        CacheStats {
            entry_count: entries.len(),
            max_entries: self.max_entries,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entry_count: usize,
    pub max_entries: usize,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ============================================================================
// Edge Runtime Core
// ============================================================================

/// Main edge runtime for serverless vector search
pub struct EdgeRuntime {
    config: EdgeConfig,
    /// In-memory collection (for loaded segments)
    collection: Option<Collection>,
    /// Manifest for the collection
    manifest: Option<EdgeManifest>,
    /// Loaded segment IDs
    loaded_segments: Vec<u32>,
    /// Search result cache
    cache: SearchCache,
    /// Statistics
    stats: EdgeRuntimeStats,
}

/// Runtime statistics
#[derive(Debug, Clone, Default)]
pub struct EdgeRuntimeStats {
    /// Total searches performed
    pub search_count: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Segments loaded
    pub segments_loaded: usize,
    /// Current memory estimate in bytes
    pub memory_estimate: usize,
}

impl EdgeRuntime {
    /// Create a new edge runtime
    pub fn new(config: EdgeConfig) -> Self {
        Self {
            cache: SearchCache::new(&config.cache_config),
            config,
            collection: None,
            manifest: None,
            loaded_segments: Vec::new(),
            stats: EdgeRuntimeStats::default(),
        }
    }

    /// Create runtime with default config for detected platform
    pub fn auto() -> Self {
        Self::new(EdgeConfig::for_platform(Platform::detect()))
    }

    /// Get current configuration
    pub fn config(&self) -> &EdgeConfig {
        &self.config
    }

    /// Get runtime statistics
    pub fn stats(&self) -> &EdgeRuntimeStats {
        &self.stats
    }

    /// Initialize an empty collection
    pub fn create_collection(
        &mut self,
        name: &str,
        dimensions: usize,
        distance: DistanceFunction,
    ) -> Result<()> {
        let collection_config = CollectionConfig::new(name, dimensions)
            .with_distance(distance)
            .with_m(self.config.hnsw_config.m)
            .with_ef_construction(self.config.hnsw_config.ef_construction);

        self.collection = Some(Collection::new(collection_config));
        self.manifest = Some(EdgeManifest::new(name, dimensions, distance));
        self.loaded_segments.clear();

        Ok(())
    }

    /// Load collection from edge storage
    pub fn load_collection<S: EdgeStorage>(&mut self, name: &str, storage: &S) -> Result<()> {
        // Load manifest
        let manifest_key = format!("{}/manifest.json", name);
        let manifest_bytes = storage
            .get(&manifest_key)?
            .ok_or_else(|| NeedleError::NotFound(format!("Collection '{}' not found", name)))?;

        let manifest_str = String::from_utf8(manifest_bytes)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;
        let manifest = EdgeManifest::from_json(&manifest_str)?;

        // Parse distance function
        let distance = match manifest.distance.as_str() {
            "Cosine" => DistanceFunction::Cosine,
            "Euclidean" => DistanceFunction::Euclidean,
            "DotProduct" => DistanceFunction::DotProduct,
            "Manhattan" => DistanceFunction::Manhattan,
            _ => DistanceFunction::Cosine,
        };

        // Create collection config
        let collection_config = CollectionConfig::new(&manifest.name, manifest.dimensions)
            .with_distance(distance)
            .with_m(manifest.hnsw_config.m)
            .with_ef_construction(manifest.hnsw_config.ef_construction);

        self.collection = Some(Collection::new(collection_config));
        self.manifest = Some(manifest);
        self.loaded_segments.clear();

        // Load segments based on lazy loading config
        if !self.config.lazy_loading {
            self.load_all_segments(storage)?;
        }

        Ok(())
    }

    /// Load all segments from storage
    fn load_all_segments<S: EdgeStorage>(&mut self, storage: &S) -> Result<()> {
        let segment_ids: Vec<u32> = {
            let manifest = self
                .manifest
                .as_ref()
                .ok_or_else(|| NeedleError::InvalidState("No manifest loaded".into()))?;
            manifest.segments.iter().map(|s| s.id).collect()
        };

        for segment_id in segment_ids {
            self.load_segment(segment_id, storage)?;
        }

        Ok(())
    }

    /// Load a specific segment
    fn load_segment<S: EdgeStorage>(&mut self, segment_id: u32, storage: &S) -> Result<()> {
        if self.loaded_segments.contains(&segment_id) {
            return Ok(()); // Already loaded
        }

        let manifest = self
            .manifest
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No manifest loaded".into()))?;

        let segment_key = format!("{}/segment_{}.bin", manifest.name, segment_id);
        let segment_bytes = storage
            .get(&segment_key)?
            .ok_or_else(|| NeedleError::NotFound(format!("Segment {} not found", segment_id)))?;

        let segment = IndexSegment::from_bytes(&segment_bytes)?;

        // Add vectors to collection
        let collection = self
            .collection
            .as_mut()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        for (i, id) in segment.vector_ids.iter().enumerate() {
            let vector = &segment.vectors[i];
            let metadata = segment.vector_metadata.get(id).cloned();
            collection.insert(id, vector, metadata)?;
        }

        self.loaded_segments.push(segment_id);
        self.stats.segments_loaded = self.loaded_segments.len();
        self.stats.memory_estimate += segment.metadata.uncompressed_size;

        Ok(())
    }

    /// Save collection to edge storage
    pub fn save_collection<S: EdgeStorage>(&mut self, storage: &S) -> Result<()> {
        let collection = self
            .collection
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        let manifest = self
            .manifest
            .as_mut()
            .ok_or_else(|| NeedleError::InvalidState("No manifest initialized".into()))?;

        // Partition vectors into segments
        let mut segments: Vec<IndexSegment> = Vec::new();
        let mut current_segment = IndexSegment::new(0, 0);

        for (id, vector, metadata) in collection.iter() {
            current_segment.add_vector(id.to_string(), vector.to_vec(), metadata.cloned());

            if current_segment.is_full(self.config.segment_size) {
                segments.push(current_segment);
                let next_id = segments.len() as u32;
                let next_offset = segments.iter().map(|s| s.metadata.vector_count).sum();
                current_segment = IndexSegment::new(next_id, next_offset);
            }
        }

        // Don't forget the last segment
        if !current_segment.vector_ids.is_empty() {
            segments.push(current_segment);
        }

        // Save segments and update manifest
        manifest.segments.clear();
        manifest.total_vectors = 0;

        for segment in &segments {
            let segment_bytes = segment.to_bytes()?;
            let segment_key = format!("{}/segment_{}.bin", manifest.name, segment.metadata.id);

            let mut segment_meta = segment.metadata.clone();
            segment_meta.uncompressed_size = segment_bytes.len();
            segment_meta.compressed_size = segment_bytes.len(); // TODO: compression

            storage.put(&segment_key, &segment_bytes)?;
            manifest.add_segment(segment_meta);
        }

        // Save manifest
        let manifest_key = format!("{}/manifest.json", manifest.name);
        let manifest_json = manifest.to_json()?;
        storage.put(&manifest_key, manifest_json.as_bytes())?;

        Ok(())
    }

    /// Insert a vector
    pub fn insert(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let collection = self
            .collection
            .as_mut()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        collection.insert(id, vector, metadata)?;

        // Invalidate cache
        self.cache.clear();

        Ok(())
    }

    /// Search for similar vectors
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.stats.search_count += 1;

        // Check cache first
        if let Some(results) = self.cache.get_semantic(query, k) {
            self.stats.cache_hits += 1;
            return Ok(results);
        }
        self.stats.cache_misses += 1;

        let collection = self
            .collection
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        let results = collection.search(query, k)?;

        // Cache results
        self.cache.put(query, results.clone());

        Ok(results)
    }

    /// Search with filter
    pub fn search_with_filter(
        &mut self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.stats.search_count += 1;
        self.stats.cache_misses += 1; // Filtered searches bypass cache

        let collection = self
            .collection
            .as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        collection.search_with_filter(query, k, filter)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<serde_json::Value>)> {
        self.collection
            .as_ref()?
            .get(id)
            .map(|(v, m)| (v.to_vec(), m.cloned()))
    }

    /// Delete a vector
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let collection = self
            .collection
            .as_mut()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        let deleted = collection.delete(id)?;

        // Invalidate cache
        if deleted {
            self.cache.clear();
        }

        Ok(deleted)
    }

    /// Get collection size
    pub fn len(&self) -> usize {
        self.collection.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// Check if collection is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        self.stats.memory_estimate
    }
}

// ============================================================================
// Compact Serialization for WASM
// ============================================================================

/// Compact vector representation for minimal bundle size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactVector {
    /// Vector ID (interned)
    pub id_index: u32,
    /// Quantized vector data
    pub data: Vec<u8>,
}

/// String interning table for compact IDs
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StringTable {
    strings: Vec<String>,
    index: HashMap<String, u32>,
}

impl StringTable {
    /// Create a new string table
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a string and get its index
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.index.get(s) {
            idx
        } else {
            let idx = self.strings.len() as u32;
            self.strings.push(s.to_string());
            self.index.insert(s.to_string(), idx);
            idx
        }
    }

    /// Get string by index
    pub fn get(&self, idx: u32) -> Option<&str> {
        self.strings.get(idx as usize).map(|s| s.as_str())
    }

    /// Number of interned strings
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

// ============================================================================
// Cold Start Optimization
// ============================================================================

/// Warmup hints for cold start optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupHints {
    /// Most frequently accessed vector IDs
    pub hot_vectors: Vec<String>,
    /// Common query patterns (cluster centroids)
    pub query_patterns: Vec<Vec<f32>>,
    /// Entry points for graph traversal
    pub entry_points: Vec<usize>,
}

impl WarmupHints {
    /// Create empty warmup hints
    pub fn new() -> Self {
        Self {
            hot_vectors: Vec::new(),
            query_patterns: Vec::new(),
            entry_points: Vec::new(),
        }
    }

    /// Add a hot vector ID
    pub fn add_hot_vector(&mut self, id: String) {
        if !self.hot_vectors.contains(&id) {
            self.hot_vectors.push(id);
        }
    }

    /// Add a query pattern
    pub fn add_query_pattern(&mut self, pattern: Vec<f32>) {
        self.query_patterns.push(pattern);
    }
}

impl Default for WarmupHints {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Lazy Segment Loading
// ============================================================================

/// Statistics for lazy segment loading
#[derive(Debug, Clone, Default)]
pub struct LazyLoadStats {
    pub segments_loaded: usize,
    pub total_loads: u64,
    pub total_load_time_us: u64,
    pub evictions: u64,
    pub cache_hit_rate: f64,
}

/// Demand-loads index segments with LRU eviction
pub struct LazySegmentLoader {
    manifest: EdgeManifest,
    loaded_segments: HashMap<String, IndexSegment>,
    storage: Box<dyn EdgeStorage>,
    load_order: Vec<String>, // LRU order (most-recently-used at end)
    max_loaded_segments: usize,
    total_loads: u64,
    total_load_time_us: u64,
    cache_hits: u64,
    cache_requests: u64,
    evictions: u64,
}

impl LazySegmentLoader {
    /// Create a new lazy segment loader
    pub fn new(manifest: EdgeManifest, storage: Box<dyn EdgeStorage>, max_segments: usize) -> Self {
        Self {
            manifest,
            loaded_segments: HashMap::new(),
            storage,
            load_order: Vec::new(),
            max_loaded_segments: max_segments,
            total_loads: 0,
            total_load_time_us: 0,
            cache_hits: 0,
            cache_requests: 0,
            evictions: 0,
        }
    }

    /// Get a segment by ID, loading on demand and evicting LRU if at capacity
    pub fn get_segment(&mut self, segment_id: &str) -> Result<&IndexSegment> {
        self.cache_requests += 1;

        if self.loaded_segments.contains_key(segment_id) {
            self.cache_hits += 1;
            // Move to end of LRU order (most recently used)
            self.load_order.retain(|id| id != segment_id);
            self.load_order.push(segment_id.to_string());
            return Ok(self
                .loaded_segments
                .get(segment_id)
                .expect("segment was just loaded"));
        }

        // Evict LRU segment if at capacity
        while self.loaded_segments.len() >= self.max_loaded_segments {
            if let Some(lru_id) = self.load_order.first().cloned() {
                self.evict_segment(&lru_id);
            } else {
                break;
            }
        }

        // Load segment from storage
        let start = Instant::now();
        let key = format!("{}/segment_{}.bin", self.manifest.name, segment_id);
        let bytes = self
            .storage
            .get(&key)?
            .ok_or_else(|| NeedleError::NotFound(format!("Segment '{}' not found", segment_id)))?;
        let segment = IndexSegment::from_bytes(&bytes)?;
        let elapsed_us = start.elapsed().as_micros() as u64;

        self.total_loads += 1;
        self.total_load_time_us += elapsed_us;

        self.loaded_segments.insert(segment_id.to_string(), segment);
        self.load_order.push(segment_id.to_string());

        Ok(self
            .loaded_segments
            .get(segment_id)
            .expect("segment was just loaded"))
    }

    /// Preload specific segments, returns number successfully loaded
    pub fn preload_segments(&mut self, segment_ids: &[&str]) -> Result<usize> {
        let mut loaded = 0;
        for &id in segment_ids {
            if self.loaded_segments.contains_key(id) {
                loaded += 1;
                continue;
            }
            match self.get_segment(id) {
                Ok(_) => loaded += 1,
                Err(_) => {}
            }
        }
        Ok(loaded)
    }

    /// Evict a specific segment, returns true if it was loaded
    pub fn evict_segment(&mut self, segment_id: &str) -> bool {
        if self.loaded_segments.remove(segment_id).is_some() {
            self.load_order.retain(|id| id != segment_id);
            self.evictions += 1;
            true
        } else {
            false
        }
    }

    /// Number of currently loaded segments
    pub fn loaded_count(&self) -> usize {
        self.loaded_segments.len()
    }

    /// Get loading statistics
    pub fn stats(&self) -> LazyLoadStats {
        let hit_rate = if self.cache_requests > 0 {
            self.cache_hits as f64 / self.cache_requests as f64
        } else {
            0.0
        };
        LazyLoadStats {
            segments_loaded: self.loaded_segments.len(),
            total_loads: self.total_loads,
            total_load_time_us: self.total_load_time_us,
            evictions: self.evictions,
            cache_hit_rate: hit_rate,
        }
    }
}

// ============================================================================
// Cold Start Optimization
// ============================================================================

/// Strategy for warming up the runtime on cold start
#[derive(Debug, Clone)]
pub enum WarmupStrategy {
    /// No warmup
    None,
    /// Preload the N most frequently accessed segments
    PreloadFrequent { top_n: usize },
    /// Preload smallest segments up to a byte budget
    PreloadSmallest { max_bytes: usize },
    /// Preload a custom list of segment IDs
    Custom(Vec<String>),
}

/// Timing information for cold start phases
#[derive(Debug, Clone, Default)]
pub struct ColdStartTiming {
    pub init_time_us: u64,
    pub first_query_time_us: u64,
    pub warmup_time_us: u64,
    pub total_cold_start_us: u64,
}

/// Optimizes cold start behavior for edge platforms
pub struct ColdStartOptimizer {
    platform: Platform,
    warmup_strategy: WarmupStrategy,
    preload_top_k: usize,
    timing: ColdStartTiming,
}

impl ColdStartOptimizer {
    /// Create a new cold start optimizer for the given platform
    pub fn new(platform: Platform) -> Self {
        Self {
            platform,
            warmup_strategy: WarmupStrategy::None,
            preload_top_k: 3,
            timing: ColdStartTiming::default(),
        }
    }

    /// Set the warmup strategy (builder pattern)
    pub fn with_strategy(mut self, strategy: WarmupStrategy) -> Self {
        self.warmup_strategy = strategy;
        self
    }

    /// Plan which segments to preload based on the strategy and manifest
    pub fn plan_warmup(&self, manifest: &EdgeManifest) -> Vec<String> {
        match &self.warmup_strategy {
            WarmupStrategy::None => Vec::new(),
            WarmupStrategy::PreloadFrequent { top_n } => {
                // Preload the first N segments (heuristic: lower IDs are more frequently accessed)
                manifest
                    .segments
                    .iter()
                    .take(*top_n)
                    .map(|s| s.id.to_string())
                    .collect()
            }
            WarmupStrategy::PreloadSmallest { max_bytes } => {
                let mut segments: Vec<_> = manifest.segments.iter().collect();
                segments.sort_by_key(|s| s.compressed_size);
                let mut budget = *max_bytes;
                let mut result = Vec::new();
                for s in segments {
                    if s.compressed_size <= budget {
                        budget -= s.compressed_size;
                        result.push(s.id.to_string());
                    } else {
                        break;
                    }
                }
                result
            }
            WarmupStrategy::Custom(ids) => ids.clone(),
        }
    }

    /// Record cold start timing
    pub fn record_timing(&mut self, timing: ColdStartTiming) {
        self.timing = timing;
    }

    /// Get recorded timing
    pub fn get_timing(&self) -> &ColdStartTiming {
        &self.timing
    }

    /// Check if cold start is within the platform's execution time budget
    pub fn is_within_budget(&self) -> bool {
        let budget_us = self.platform.max_execution_time_ms() * 1000;
        self.timing.total_cold_start_us < budget_us
    }
}

// ============================================================================
// Enhanced Search Cache
// ============================================================================

/// A single cached search result entry
pub struct CachedSearchResult {
    pub results: Vec<EdgeSearchResult>,
    pub created_at: Instant,
    pub access_count: u64,
}

/// A search result from the edge runtime
#[derive(Debug, Clone)]
pub struct EdgeSearchResult {
    pub id: String,
    pub distance: f32,
    pub segment_id: String,
}

/// Enhanced search result cache with TTL and hit/miss tracking
pub struct EdgeSearchCache {
    cache: HashMap<u64, CachedSearchResult>,
    max_entries: usize,
    ttl: Duration,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Statistics for EdgeSearchCache
#[derive(Debug, Clone)]
pub struct EdgeCacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

impl EdgeSearchCache {
    /// Create a new edge search cache
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_entries,
            ttl,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Get cached results for a query hash, returning None on miss or TTL expiry
    pub fn get(&mut self, query_hash: u64) -> Option<&[EdgeSearchResult]> {
        // Check if entry exists and is not expired
        let expired = self
            .cache
            .get(&query_hash)
            .map(|entry| entry.created_at.elapsed() > self.ttl)
            .unwrap_or(true);

        if expired {
            // Remove expired entry if it existed
            if self.cache.remove(&query_hash).is_some() {
                self.evictions += 1;
            }
            self.misses += 1;
            return None;
        }

        self.hits += 1;
        let entry = self
            .cache
            .get_mut(&query_hash)
            .expect("entry was just inserted");
        entry.access_count += 1;
        Some(&entry.results)
    }

    /// Store search results for a query hash, evicting least-accessed entry if at capacity
    pub fn put(&mut self, query_hash: u64, results: Vec<EdgeSearchResult>) {
        // Evict expired entries first
        let now = Instant::now();
        let expired_keys: Vec<u64> = self
            .cache
            .iter()
            .filter(|(_, v)| now.duration_since(v.created_at) > self.ttl)
            .map(|(&k, _)| k)
            .collect();
        for key in &expired_keys {
            self.cache.remove(key);
            self.evictions += 1;
        }

        // Evict least-accessed entry if still at capacity
        while self.cache.len() >= self.max_entries {
            let lru_key = self
                .cache
                .iter()
                .min_by_key(|(_, v)| v.access_count)
                .map(|(&k, _)| k);
            if let Some(key) = lru_key {
                self.cache.remove(&key);
                self.evictions += 1;
            } else {
                break;
            }
        }

        self.cache.insert(
            query_hash,
            CachedSearchResult {
                results,
                created_at: Instant::now(),
                access_count: 0,
            },
        );
    }

    /// Invalidate a specific cache entry, returns true if it existed
    pub fn invalidate(&mut self, query_hash: u64) -> bool {
        self.cache.remove(&query_hash).is_some()
    }

    /// Clear all cache entries
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Deterministic hash for a query vector and k value
    pub fn hash_query(query: &[f32], k: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &v in query {
            v.to_bits().hash(&mut hasher);
        }
        k.hash(&mut hasher);
        hasher.finish()
    }

    /// Get cache statistics
    pub fn stats(&self) -> EdgeCacheStats {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        EdgeCacheStats {
            entries: self.cache.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            hit_rate,
        }
    }
}

// ============================================================================
// Edge Runtime Builder and Enhanced Runtime
// ============================================================================

/// Health status for the enhanced edge runtime
#[derive(Debug, Clone)]
pub struct EdgeHealthStatus {
    pub is_healthy: bool,
    pub segments_loaded: usize,
    pub cache_entries: usize,
    pub memory_usage_estimate: usize,
}

/// Builder for fluent construction of EdgeRuntimeEnhanced
pub struct EdgeRuntimeBuilder {
    platform: Platform,
    config: EdgeConfig,
    warmup_strategy: WarmupStrategy,
    cache_max_entries: usize,
    cache_ttl: Duration,
    max_segments: usize,
}

impl EdgeRuntimeBuilder {
    /// Create a new builder for the given platform
    pub fn new(platform: Platform) -> Self {
        Self {
            platform,
            config: EdgeConfig::for_platform(platform),
            warmup_strategy: WarmupStrategy::None,
            cache_max_entries: 100,
            cache_ttl: Duration::from_secs(60),
            max_segments: 10,
        }
    }

    /// Set the edge config
    pub fn with_config(mut self, config: EdgeConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the warmup strategy
    pub fn with_warmup(mut self, strategy: WarmupStrategy) -> Self {
        self.warmup_strategy = strategy;
        self
    }

    /// Set cache parameters
    pub fn with_cache(mut self, max_entries: usize, ttl: Duration) -> Self {
        self.cache_max_entries = max_entries;
        self.cache_ttl = ttl;
        self
    }

    /// Set maximum loaded segments
    pub fn with_max_segments(mut self, n: usize) -> Self {
        self.max_segments = n;
        self
    }

    /// Build the enhanced runtime
    pub fn build(
        self,
        manifest: EdgeManifest,
        storage: Box<dyn EdgeStorage>,
    ) -> Result<EdgeRuntimeEnhanced> {
        let optimizer = ColdStartOptimizer::new(self.platform).with_strategy(self.warmup_strategy);
        let cache = EdgeSearchCache::new(self.cache_max_entries, self.cache_ttl);
        let loader = LazySegmentLoader::new(manifest, storage, self.max_segments);

        Ok(EdgeRuntimeEnhanced {
            loader,
            cache,
            optimizer,
        })
    }
}

/// Enhanced edge runtime combining lazy loading, caching, and cold start optimization
pub struct EdgeRuntimeEnhanced {
    pub loader: LazySegmentLoader,
    pub cache: EdgeSearchCache,
    pub optimizer: ColdStartOptimizer,
}

impl EdgeRuntimeEnhanced {
    /// Search for nearest neighbors with caching and lazy segment loading
    pub fn search(&mut self, query: &[f32], k: usize) -> Result<Vec<EdgeSearchResult>> {
        let query_hash = EdgeSearchCache::hash_query(query, k);

        // Check cache first
        if let Some(results) = self.cache.get(query_hash) {
            return Ok(results.to_vec());
        }

        // Load segments and search
        let mut all_results: Vec<EdgeSearchResult> = Vec::new();
        let segment_ids: Vec<String> = self
            .loader
            .manifest
            .segments
            .iter()
            .map(|s| s.id.to_string())
            .collect();

        for seg_id in &segment_ids {
            let segment = self.loader.get_segment(seg_id)?;
            // Linear scan within segment (simplified search)
            for (i, vec) in segment.vectors.iter().enumerate() {
                if vec.len() == query.len() {
                    let distance = cosine_distance(query, vec);
                    if let Some(id) = segment.vector_ids.get(i) {
                        all_results.push(EdgeSearchResult {
                            id: id.clone(),
                            distance,
                            segment_id: seg_id.clone(),
                        });
                    }
                }
            }
        }

        // Sort by distance and take top-k
        all_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(k);

        // Cache results
        self.cache.put(query_hash, all_results.clone());

        Ok(all_results)
    }

    /// Get health status of the runtime
    pub fn health_check(&self) -> EdgeHealthStatus {
        let loader_stats = self.loader.stats();
        let cache_stats = self.cache.stats();

        EdgeHealthStatus {
            is_healthy: true,
            segments_loaded: loader_stats.segments_loaded,
            cache_entries: cache_stats.entries,
            memory_usage_estimate: loader_stats.segments_loaded * 1024 * 1024, // rough estimate
        }
    }
}

/// Compute cosine distance (1 - similarity) between two vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_platform_config() {
        let config = EdgeConfig::for_platform(Platform::CloudflareWorkers);
        assert!(config.max_vectors_in_memory > 0);
        assert!(config.lazy_loading);

        let config = EdgeConfig::for_platform(Platform::LambdaEdge);
        assert!(config.hnsw_config.m < 16); // Reduced for memory
    }

    #[test]
    fn test_edge_runtime_basic() {
        let mut runtime = EdgeRuntime::new(EdgeConfig::default());

        runtime
            .create_collection("test", 128, DistanceFunction::Cosine)
            .unwrap();

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(128);
            runtime.insert(format!("vec_{}", i), &vec, None).unwrap();
        }

        assert_eq!(runtime.len(), 100);

        // Search
        let query = random_vector(128);
        let results = runtime.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_cache() {
        let config = EdgeCacheConfig::default();
        let cache = SearchCache::new(&config);

        let query = vec![1.0, 2.0, 3.0];
        let results = vec![
            SearchResult {
                id: "a".into(),
                distance: 0.1,
                metadata: None,
            },
            SearchResult {
                id: "b".into(),
                distance: 0.2,
                metadata: None,
            },
        ];

        // Initially no cache
        assert!(cache.get_exact(&query, 2).is_none());

        // Put in cache
        cache.put(&query, results.clone());

        // Should hit cache
        let cached = cache.get_exact(&query, 2).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].id, "a");
    }

    #[test]
    fn test_semantic_cache() {
        let config = EdgeCacheConfig {
            semantic_threshold: 0.99,
            ..Default::default()
        };
        let cache = SearchCache::new(&config);

        let query1 = vec![1.0, 0.0, 0.0];
        let query2 = vec![0.999, 0.001, 0.0]; // Very similar
        let query3 = vec![0.0, 1.0, 0.0]; // Different

        let results = vec![SearchResult {
            id: "a".into(),
            distance: 0.1,
            metadata: None,
        }];

        cache.put(&query1, results);

        // Similar query should hit
        assert!(cache.get_semantic(&query2, 1).is_some());

        // Different query should miss
        assert!(cache.get_semantic(&query3, 1).is_none());
    }

    #[test]
    fn test_index_segment() {
        let mut segment = IndexSegment::new(0, 0);

        segment.add_vector("v1".into(), vec![1.0, 2.0], None);
        segment.add_vector(
            "v2".into(),
            vec![3.0, 4.0],
            Some(serde_json::json!({"key": "value"})),
        );

        assert_eq!(segment.metadata.vector_count, 2);
        assert!(!segment.is_full(10));

        // Serialize and deserialize
        let bytes = segment.to_bytes().unwrap();
        let restored = IndexSegment::from_bytes(&bytes).unwrap();

        assert_eq!(restored.vector_ids, segment.vector_ids);
        assert_eq!(restored.vectors, segment.vectors);
    }

    #[test]
    fn test_edge_storage_roundtrip() {
        let storage = InMemoryEdgeStorage::new();
        let mut runtime = EdgeRuntime::new(EdgeConfig::default());

        // Create and populate collection
        runtime
            .create_collection("test", 64, DistanceFunction::Cosine)
            .unwrap();
        for i in 0..50 {
            runtime
                .insert(format!("v{}", i), &random_vector(64), None)
                .unwrap();
        }

        // Save to storage
        runtime.save_collection(&storage).unwrap();

        // Load in new runtime
        let mut runtime2 = EdgeRuntime::new(EdgeConfig::default().with_lazy_loading(false));
        runtime2.load_collection("test", &storage).unwrap();

        assert_eq!(runtime2.len(), 50);
    }

    #[test]
    fn test_string_table() {
        let mut table = StringTable::new();

        let idx1 = table.intern("hello");
        let idx2 = table.intern("world");
        let idx3 = table.intern("hello"); // Duplicate

        assert_eq!(idx1, idx3); // Same string, same index
        assert_ne!(idx1, idx2);

        assert_eq!(table.get(idx1), Some("hello"));
        assert_eq!(table.get(idx2), Some("world"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_stats() {
        let cache = SearchCache::new(&EdgeCacheConfig::default());

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 0);

        cache.put(&[1.0], vec![]);
        let stats = cache.stats();
        assert_eq!(stats.entry_count, 1);
    }

    // ========================================================================
    // New tests for enhanced edge runtime features
    // ========================================================================

    /// Helper: create a test manifest with N segments and store them
    fn setup_test_segments(
        storage: &InMemoryEdgeStorage,
        collection_name: &str,
        num_segments: usize,
        vectors_per_segment: usize,
        dim: usize,
    ) -> EdgeManifest {
        let mut manifest = EdgeManifest::new(collection_name, dim, DistanceFunction::Cosine);

        for seg_idx in 0..num_segments {
            let mut segment = IndexSegment::new(seg_idx as u32, seg_idx * vectors_per_segment);
            for v in 0..vectors_per_segment {
                let id = format!("seg{}_{}", seg_idx, v);
                segment.add_vector(id, random_vector(dim), None);
            }
            let bytes = segment.to_bytes().unwrap();
            let key = format!("{}/segment_{}.bin", collection_name, seg_idx);
            let meta = SegmentMetadata {
                id: seg_idx as u32,
                vector_count: vectors_per_segment,
                id_offset: seg_idx * vectors_per_segment,
                compressed_size: bytes.len(),
                uncompressed_size: bytes.len(),
                checksum: 0,
                max_layer: 0,
            };
            manifest.add_segment(meta);
            storage.put(&key, &bytes).unwrap();
        }
        manifest
    }

    #[test]
    fn test_lazy_segment_loader_load_on_demand() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "lazy", 3, 5, 4);
        let mut loader = LazySegmentLoader::new(manifest, Box::new(storage), 10);

        assert_eq!(loader.loaded_count(), 0);

        let segment = loader.get_segment("0").unwrap();
        assert_eq!(segment.metadata.vector_count, 5);
        assert_eq!(loader.loaded_count(), 1);

        let _ = loader.get_segment("1").unwrap();
        assert_eq!(loader.loaded_count(), 2);

        // Requesting same segment again should be a cache hit
        let _ = loader.get_segment("0").unwrap();
        assert_eq!(loader.loaded_count(), 2);
        assert!(loader.stats().cache_hit_rate > 0.0);
    }

    #[test]
    fn test_lazy_segment_loader_lru_eviction() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "lru", 4, 3, 4);
        let mut loader = LazySegmentLoader::new(manifest, Box::new(storage), 2);

        // Load segments 0 and 1 (fills capacity)
        let _ = loader.get_segment("0").unwrap();
        let _ = loader.get_segment("1").unwrap();
        assert_eq!(loader.loaded_count(), 2);

        // Loading segment 2 should evict segment 0 (LRU)
        let _ = loader.get_segment("2").unwrap();
        assert_eq!(loader.loaded_count(), 2);
        assert!(loader.stats().evictions >= 1);

        // Segment 0 should need reloading
        let _ = loader.get_segment("0").unwrap();
        assert_eq!(loader.stats().total_loads, 4); // 0,1,2,0
    }

    #[test]
    fn test_lazy_segment_loader_preload() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "preload", 3, 5, 4);
        let mut loader = LazySegmentLoader::new(manifest, Box::new(storage), 10);

        let loaded = loader.preload_segments(&["0", "2"]).unwrap();
        assert_eq!(loaded, 2);
        assert_eq!(loader.loaded_count(), 2);

        // Preloading already-loaded segment counts as loaded
        let loaded = loader.preload_segments(&["0"]).unwrap();
        assert_eq!(loaded, 1);
    }

    #[test]
    fn test_lazy_segment_loader_evict() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "evict", 2, 3, 4);
        let mut loader = LazySegmentLoader::new(manifest, Box::new(storage), 10);

        let _ = loader.get_segment("0").unwrap();
        assert_eq!(loader.loaded_count(), 1);

        assert!(loader.evict_segment("0"));
        assert_eq!(loader.loaded_count(), 0);

        // Evicting non-loaded segment returns false
        assert!(!loader.evict_segment("99"));
    }

    #[test]
    fn test_cold_start_optimizer_plan_warmup_none() {
        let manifest = EdgeManifest::new("test", 4, DistanceFunction::Cosine);
        let optimizer = ColdStartOptimizer::new(Platform::CloudflareWorkers)
            .with_strategy(WarmupStrategy::None);

        let plan = optimizer.plan_warmup(&manifest);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_cold_start_optimizer_plan_warmup_frequent() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "freq", 5, 3, 4);

        let optimizer = ColdStartOptimizer::new(Platform::CloudflareWorkers)
            .with_strategy(WarmupStrategy::PreloadFrequent { top_n: 2 });

        let plan = optimizer.plan_warmup(&manifest);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0], "0");
        assert_eq!(plan[1], "1");
    }

    #[test]
    fn test_cold_start_optimizer_plan_warmup_smallest() {
        let mut manifest = EdgeManifest::new("test", 4, DistanceFunction::Cosine);
        manifest.add_segment(SegmentMetadata {
            id: 0,
            vector_count: 10,
            id_offset: 0,
            compressed_size: 500,
            uncompressed_size: 1000,
            checksum: 0,
            max_layer: 0,
        });
        manifest.add_segment(SegmentMetadata {
            id: 1,
            vector_count: 10,
            id_offset: 10,
            compressed_size: 100,
            uncompressed_size: 200,
            checksum: 0,
            max_layer: 0,
        });
        manifest.add_segment(SegmentMetadata {
            id: 2,
            vector_count: 10,
            id_offset: 20,
            compressed_size: 300,
            uncompressed_size: 600,
            checksum: 0,
            max_layer: 0,
        });

        let optimizer = ColdStartOptimizer::new(Platform::GenericWasm)
            .with_strategy(WarmupStrategy::PreloadSmallest { max_bytes: 450 });

        let plan = optimizer.plan_warmup(&manifest);
        // Sorted by size: 100, 300, 500 → fits 100 + 300 = 400 ≤ 450
        assert_eq!(plan.len(), 2);
        assert!(plan.contains(&"1".to_string()));
        assert!(plan.contains(&"2".to_string()));
    }

    #[test]
    fn test_cold_start_optimizer_plan_warmup_custom() {
        let manifest = EdgeManifest::new("test", 4, DistanceFunction::Cosine);
        let optimizer = ColdStartOptimizer::new(Platform::VercelEdge)
            .with_strategy(WarmupStrategy::Custom(vec!["seg_a".into(), "seg_b".into()]));

        let plan = optimizer.plan_warmup(&manifest);
        assert_eq!(plan, vec!["seg_a", "seg_b"]);
    }

    #[test]
    fn test_cold_start_optimizer_timing() {
        let mut optimizer = ColdStartOptimizer::new(Platform::CloudflareWorkers);
        assert!(optimizer.is_within_budget()); // zero timing

        optimizer.record_timing(ColdStartTiming {
            init_time_us: 1000,
            first_query_time_us: 2000,
            warmup_time_us: 500,
            total_cold_start_us: 3500,
        });

        let timing = optimizer.get_timing();
        assert_eq!(timing.init_time_us, 1000);
        assert!(optimizer.is_within_budget());
    }

    #[test]
    fn test_edge_search_cache_put_get() {
        let mut cache = EdgeSearchCache::new(10, Duration::from_secs(60));

        let hash = EdgeSearchCache::hash_query(&[1.0, 2.0], 5);
        assert!(cache.get(hash).is_none());

        let results = vec![
            EdgeSearchResult {
                id: "a".into(),
                distance: 0.1,
                segment_id: "s0".into(),
            },
            EdgeSearchResult {
                id: "b".into(),
                distance: 0.2,
                segment_id: "s0".into(),
            },
        ];
        cache.put(hash, results);

        let cached = cache.get(hash).unwrap();
        assert_eq!(cached.len(), 2);
        assert_eq!(cached[0].id, "a");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1); // first get was a miss
    }

    #[test]
    fn test_edge_search_cache_eviction_at_capacity() {
        let mut cache = EdgeSearchCache::new(2, Duration::from_secs(60));

        let h1 = EdgeSearchCache::hash_query(&[1.0], 1);
        let h2 = EdgeSearchCache::hash_query(&[2.0], 1);
        let h3 = EdgeSearchCache::hash_query(&[3.0], 1);

        cache.put(h1, vec![]);
        cache.put(h2, vec![]);
        assert_eq!(cache.stats().entries, 2);

        // Inserting a third should evict one
        cache.put(h3, vec![]);
        assert_eq!(cache.stats().entries, 2);
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_edge_search_cache_ttl_expiry() {
        let mut cache = EdgeSearchCache::new(10, Duration::from_millis(1));

        let hash = EdgeSearchCache::hash_query(&[1.0], 1);
        cache.put(
            hash,
            vec![EdgeSearchResult {
                id: "x".into(),
                distance: 0.0,
                segment_id: "s0".into(),
            }],
        );

        // Sleep to let TTL expire
        std::thread::sleep(Duration::from_millis(5));

        assert!(cache.get(hash).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_edge_search_cache_invalidate() {
        let mut cache = EdgeSearchCache::new(10, Duration::from_secs(60));

        let hash = EdgeSearchCache::hash_query(&[1.0], 5);
        cache.put(hash, vec![]);
        assert!(cache.invalidate(hash));
        assert!(!cache.invalidate(hash)); // already removed
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn test_edge_search_cache_clear() {
        let mut cache = EdgeSearchCache::new(10, Duration::from_secs(60));
        cache.put(1, vec![]);
        cache.put(2, vec![]);
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn test_hash_query_determinism() {
        let query = vec![1.0f32, 2.0, 3.0, 4.5];
        let h1 = EdgeSearchCache::hash_query(&query, 10);
        let h2 = EdgeSearchCache::hash_query(&query, 10);
        assert_eq!(h1, h2);

        // Different k yields different hash
        let h3 = EdgeSearchCache::hash_query(&query, 5);
        assert_ne!(h1, h3);

        // Different vector yields different hash
        let other = vec![1.0f32, 2.0, 3.0, 4.6];
        let h4 = EdgeSearchCache::hash_query(&other, 10);
        assert_ne!(h1, h4);
    }

    #[test]
    fn test_edge_runtime_builder_fluent_api() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "builder", 2, 3, 4);

        let enhanced = EdgeRuntimeBuilder::new(Platform::CloudflareWorkers)
            .with_config(EdgeConfig::for_platform(Platform::CloudflareWorkers))
            .with_warmup(WarmupStrategy::PreloadFrequent { top_n: 1 })
            .with_cache(50, Duration::from_secs(30))
            .with_max_segments(5)
            .build(manifest, Box::new(storage))
            .unwrap();

        let health = enhanced.health_check();
        assert!(health.is_healthy);
        assert_eq!(health.segments_loaded, 0);
        assert_eq!(health.cache_entries, 0);
    }

    #[test]
    fn test_edge_runtime_enhanced_search_with_caching() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "search", 2, 5, 4);

        let mut enhanced = EdgeRuntimeBuilder::new(Platform::GenericWasm)
            .with_cache(100, Duration::from_secs(60))
            .with_max_segments(10)
            .build(manifest, Box::new(storage))
            .unwrap();

        let query = vec![0.5, 0.5, 0.5, 0.5];

        // First search → cache miss
        let results = enhanced.search(&query, 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(enhanced.cache.stats().misses, 1);
        assert_eq!(enhanced.cache.stats().hits, 0);

        // Second search with same query → cache hit
        let results2 = enhanced.search(&query, 3).unwrap();
        assert_eq!(results2.len(), 3);
        assert_eq!(enhanced.cache.stats().hits, 1);

        // Results should match
        assert_eq!(results[0].id, results2[0].id);
    }

    #[test]
    fn test_edge_runtime_enhanced_health_check() {
        let storage = InMemoryEdgeStorage::new();
        let manifest = setup_test_segments(&storage, "health", 3, 4, 4);

        let mut enhanced = EdgeRuntimeBuilder::new(Platform::VercelEdge)
            .with_max_segments(10)
            .build(manifest, Box::new(storage))
            .unwrap();

        // Before any search
        let health = enhanced.health_check();
        assert!(health.is_healthy);
        assert_eq!(health.segments_loaded, 0);

        // After searching (triggers segment loading)
        let _ = enhanced.search(&[0.1, 0.2, 0.3, 0.4], 2).unwrap();
        let health = enhanced.health_check();
        assert!(health.segments_loaded > 0);
        assert_eq!(health.cache_entries, 1);
    }
}
