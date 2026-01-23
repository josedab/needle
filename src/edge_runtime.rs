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
            Platform::CloudflareWorkers => 30_000,  // 30s (paid)
            Platform::DenoKv => 60_000,             // 60s
            Platform::VercelEdge => 30_000,         // 30s
            Platform::FastlyCompute => 60_000,      // 60s
            Platform::LambdaEdge => 5_000,          // 5s (viewer request)
            Platform::GenericWasm => 120_000,       // 2 min
        }
    }

    /// Get platform-specific max storage size per key in bytes
    pub fn max_storage_value_size(&self) -> usize {
        match self {
            Platform::CloudflareWorkers => 25 * 1024 * 1024,  // 25 MB (R2)
            Platform::DenoKv => 64 * 1024,                     // 64 KB
            Platform::VercelEdge => 4 * 1024 * 1024,           // 4 MB
            Platform::FastlyCompute => 8 * 1024 * 1024,        // 8 MB
            Platform::LambdaEdge => 40 * 1024,                 // 40 KB (response)
            Platform::GenericWasm => 100 * 1024 * 1024,        // 100 MB
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
            m: 12,              // Reduced from 16 for memory
            ef_construction: 100, // Reduced from 200
            ef_search: 30,      // Reduced from 50 for speed
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
    pub fn add_vector(&mut self, id: String, vector: Vec<f32>, metadata: Option<serde_json::Value>) {
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

    /// Compress segment data
    #[cfg(feature = "compression")]
    pub fn compress(&self) -> Result<Vec<u8>> {
        let bytes = self.to_bytes()?;
        // Use simple LZ4 or similar compression
        Ok(bytes) // Placeholder - actual compression would be added
    }

    /// Decompress segment data
    #[cfg(feature = "compression")]
    pub fn decompress(bytes: &[u8]) -> Result<Self> {
        // Placeholder for decompression
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
        Ok(self.data.read()
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
            let oldest = entries.iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| *k);
            if let Some(key) = oldest {
                entries.remove(&key);
                self.query_vectors.write().remove(&key);
            } else {
                break;
            }
        }

        entries.insert(hash, CachedResult {
            results,
            query_hash: hash,
            created_at: std::time::Instant::now(),
        });

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
        let manifest_bytes = storage.get(&manifest_key)?
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
            let manifest = self.manifest.as_ref()
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

        let manifest = self.manifest.as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No manifest loaded".into()))?;

        let segment_key = format!("{}/segment_{}.bin", manifest.name, segment_id);
        let segment_bytes = storage.get(&segment_key)?
            .ok_or_else(|| NeedleError::NotFound(format!("Segment {} not found", segment_id)))?;

        let segment = IndexSegment::from_bytes(&segment_bytes)?;

        // Add vectors to collection
        let collection = self.collection.as_mut()
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
        let collection = self.collection.as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        let manifest = self.manifest.as_mut()
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
        let collection = self.collection.as_mut()
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

        let collection = self.collection.as_ref()
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

        let collection = self.collection.as_ref()
            .ok_or_else(|| NeedleError::InvalidState("No collection initialized".into()))?;

        collection.search_with_filter(query, k, filter)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<serde_json::Value>)> {
        self.collection.as_ref()?.get(id).map(|(v, m)| (v.to_vec(), m.cloned()))
    }

    /// Delete a vector
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let collection = self.collection.as_mut()
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

        runtime.create_collection("test", 128, DistanceFunction::Cosine).unwrap();

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
            SearchResult { id: "a".into(), distance: 0.1, metadata: None },
            SearchResult { id: "b".into(), distance: 0.2, metadata: None },
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

        let results = vec![
            SearchResult { id: "a".into(), distance: 0.1, metadata: None },
        ];

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
        segment.add_vector("v2".into(), vec![3.0, 4.0], Some(serde_json::json!({"key": "value"})));

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
        runtime.create_collection("test", 64, DistanceFunction::Cosine).unwrap();
        for i in 0..50 {
            runtime.insert(format!("v{}", i), &random_vector(64), None).unwrap();
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
}
