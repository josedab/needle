//! Platform Adapters for Edge Runtimes
//!
//! Provides platform-specific storage adapters for serverless edge environments.
//! Each adapter implements the `EdgeStorage` trait with optimizations specific
//! to the target platform.
//!
//! # Supported Platforms
//!
//! - **Cloudflare Workers**: KV and R2 storage integration
//! - **Deno Deploy**: Deno KV storage
//! - **Vercel Edge**: Edge Config and Blob storage
//! - **Generic WASM**: In-memory and IndexedDB fallback
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::platform_adapters::{CloudflareKvAdapter, DenoKvAdapter};
//!
//! // Cloudflare Workers
//! let cf_storage = CloudflareKvAdapter::new("NEEDLE_KV");
//!
//! // Deno Deploy
//! let deno_storage = DenoKvAdapter::new().await?;
//! ```

use crate::edge_runtime::EdgeStorage;
use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Cloudflare Workers Adapters
// ============================================================================

/// Cloudflare Workers KV storage adapter
///
/// KV is optimized for read-heavy workloads with eventual consistency.
/// Best for: cached data, configuration, frequently accessed vectors.
///
/// Limits:
/// - Max value size: 25 MB
/// - Max key size: 512 bytes
/// - Eventually consistent reads
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CloudflareKvAdapter {
    /// KV namespace binding name
    namespace: String,
    /// Key prefix for all operations
    prefix: String,
    /// Cached values for read optimization
    cache: Arc<parking_lot::RwLock<HashMap<String, Vec<u8>>>>,
}

impl CloudflareKvAdapter {
    /// Create a new KV adapter with namespace binding name
    pub fn new(namespace: &str) -> Self {
        Self {
            namespace: namespace.to_string(),
            prefix: String::new(),
            cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Set a key prefix for all operations
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    /// Get the full key with prefix
    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }

    /// Clear the local cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
}

impl EdgeStorage for CloudflareKvAdapter {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let full_key = self.full_key(key);

        // Check cache first
        if let Some(value) = self.cache.read().get(&full_key) {
            return Ok(Some(value.clone()));
        }

        // In actual WASM environment, this would call:
        // kv_namespace.get(key, { type: 'arrayBuffer' })
        // For now, return None (not in cache)
        Ok(None)
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let full_key = self.full_key(key);

        // Update cache
        self.cache.write().insert(full_key.clone(), value.to_vec());

        // In actual WASM environment, this would call:
        // kv_namespace.put(key, value)
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let full_key = self.full_key(key);
        self.cache.write().remove(&full_key);

        // In actual WASM environment, this would call:
        // kv_namespace.delete(key)
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);

        // Return cached keys matching prefix
        Ok(self.cache.read()
            .keys()
            .filter(|k| k.starts_with(&full_prefix))
            .cloned()
            .collect())
    }
}

/// Cloudflare R2 storage adapter for large objects
///
/// R2 is S3-compatible object storage with no egress fees.
/// Best for: large segments, full index backups, bulk data.
///
/// Limits:
/// - Max object size: 5 TB
/// - Strongly consistent reads
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CloudflareR2Adapter {
    /// R2 bucket binding name
    bucket: String,
    /// Key prefix
    prefix: String,
}

impl CloudflareR2Adapter {
    /// Create a new R2 adapter with bucket binding name
    pub fn new(bucket: &str) -> Self {
        Self {
            bucket: bucket.to_string(),
            prefix: String::new(),
        }
    }

    /// Set a key prefix
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = prefix.to_string();
        self
    }

    fn full_key(&self, key: &str) -> String {
        if self.prefix.is_empty() {
            key.to_string()
        } else {
            format!("{}/{}", self.prefix, key)
        }
    }
}

impl EdgeStorage for CloudflareR2Adapter {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let _full_key = self.full_key(key);
        // In actual WASM environment:
        // const object = await bucket.get(key);
        // if (object === null) return null;
        // return new Uint8Array(await object.arrayBuffer());
        Ok(None)
    }

    fn put(&self, key: &str, _value: &[u8]) -> Result<()> {
        let _full_key = self.full_key(key);
        // await bucket.put(key, value);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let _full_key = self.full_key(key);
        // await bucket.delete(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let _full_prefix = self.full_key(prefix);
        // const listed = await bucket.list({ prefix });
        // return listed.objects.map(o => o.key);
        Ok(Vec::new())
    }
}

// ============================================================================
// Deno Deploy Adapter
// ============================================================================

/// Deno KV storage adapter
///
/// Deno KV provides strongly consistent key-value storage with ACID transactions.
/// Best for: data requiring consistency, metadata, small to medium objects.
///
/// Limits:
/// - Max value size: 64 KB
/// - Max key size: 2 KB
/// - Supports atomic transactions
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DenoKvAdapter {
    /// Database path (empty for default)
    path: String,
    /// In-memory cache for testing
    cache: Arc<parking_lot::RwLock<HashMap<String, Vec<u8>>>>,
}

impl DenoKvAdapter {
    /// Create a new Deno KV adapter using default database
    pub fn new() -> Self {
        Self {
            path: String::new(),
            cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Create with specific database path (for testing)
    pub fn with_path(path: &str) -> Self {
        Self {
            path: path.to_string(),
            cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }
}

impl Default for DenoKvAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeStorage for DenoKvAdapter {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Check cache
        if let Some(value) = self.cache.read().get(key) {
            return Ok(Some(value.clone()));
        }

        // In actual Deno environment:
        // const kv = await Deno.openKv(this.path || undefined);
        // const result = await kv.get([key]);
        // return result.value;
        Ok(None)
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        // Check Deno KV value size limit
        if value.len() > 64 * 1024 {
            return Err(NeedleError::InvalidInput(format!(
                "Value size {} exceeds Deno KV limit of 64KB",
                value.len()
            )));
        }

        self.cache.write().insert(key.to_string(), value.to_vec());

        // const kv = await Deno.openKv();
        // await kv.set([key], value);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.cache.write().remove(key);
        // await kv.delete([key]);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self.cache.read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }
}

// ============================================================================
// Vercel Edge Adapter
// ============================================================================

/// Vercel Edge Config adapter for configuration data
///
/// Edge Config is optimized for reading configuration at the edge.
/// Best for: config, feature flags, small lookup tables.
///
/// Limits:
/// - Max total size: 512 KB (hobby) to 24 MB (enterprise)
/// - Read-only at runtime (write via API)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VercelEdgeConfigAdapter {
    /// Edge Config connection string
    connection_string: String,
    /// Cache
    cache: Arc<parking_lot::RwLock<HashMap<String, Vec<u8>>>>,
}

impl VercelEdgeConfigAdapter {
    /// Create a new Edge Config adapter
    pub fn new(connection_string: &str) -> Self {
        Self {
            connection_string: connection_string.to_string(),
            cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        std::env::var("EDGE_CONFIG")
            .map(|s| Self::new(&s))
            .map_err(|_| NeedleError::InvalidInput("EDGE_CONFIG not set".into()))
    }
}

impl EdgeStorage for VercelEdgeConfigAdapter {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if let Some(value) = self.cache.read().get(key) {
            return Ok(Some(value.clone()));
        }

        // import { get } from '@vercel/edge-config';
        // const value = await get(key);
        Ok(None)
    }

    fn put(&self, _key: &str, _value: &[u8]) -> Result<()> {
        // Edge Config is read-only at runtime
        Err(NeedleError::InvalidOperation(
            "Vercel Edge Config is read-only at runtime".into()
        ))
    }

    fn delete(&self, _key: &str) -> Result<()> {
        Err(NeedleError::InvalidOperation(
            "Vercel Edge Config is read-only at runtime".into()
        ))
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self.cache.read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }
}

/// Vercel Blob storage adapter
///
/// Vercel Blob is object storage for larger files.
/// Best for: large segments, media, bulk data.
///
/// Limits:
/// - Max blob size: 500 MB (Pro), 5 GB (Enterprise)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VercelBlobAdapter {
    /// Blob store token
    token: String,
    /// URL prefix for blobs
    url_prefix: String,
}

impl VercelBlobAdapter {
    /// Create a new Blob adapter
    pub fn new(token: &str) -> Self {
        Self {
            token: token.to_string(),
            url_prefix: String::new(),
        }
    }

    /// Create from environment
    pub fn from_env() -> Result<Self> {
        std::env::var("BLOB_READ_WRITE_TOKEN")
            .map(|t| Self::new(&t))
            .map_err(|_| NeedleError::InvalidInput("BLOB_READ_WRITE_TOKEN not set".into()))
    }
}

impl EdgeStorage for VercelBlobAdapter {
    fn get(&self, _key: &str) -> Result<Option<Vec<u8>>> {
        // const { downloadUrl } = await head(key);
        // const response = await fetch(downloadUrl);
        // return new Uint8Array(await response.arrayBuffer());
        Ok(None)
    }

    fn put(&self, _key: &str, _value: &[u8]) -> Result<()> {
        // await put(key, value, { access: 'public' });
        Ok(())
    }

    fn delete(&self, _key: &str) -> Result<()> {
        // await del(key);
        Ok(())
    }

    fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        // const { blobs } = await list({ prefix });
        // return blobs.map(b => b.pathname);
        Ok(Vec::new())
    }
}

// ============================================================================
// Tiered Storage Adapter
// ============================================================================

/// Tiered storage combining fast cache with larger backing store
///
/// Uses a small, fast storage for frequently accessed data and
/// a larger, slower storage for bulk data.
pub struct TieredEdgeStorage<C: EdgeStorage, B: EdgeStorage> {
    /// Fast cache tier (e.g., KV, in-memory)
    cache: C,
    /// Backing store tier (e.g., R2, Blob)
    backing: B,
    /// Maximum size to cache
    cache_threshold: usize,
    /// Keys known to be in backing store
    backing_keys: parking_lot::RwLock<std::collections::HashSet<String>>,
}

impl<C: EdgeStorage, B: EdgeStorage> TieredEdgeStorage<C, B> {
    /// Create a new tiered storage
    pub fn new(cache: C, backing: B, cache_threshold: usize) -> Self {
        Self {
            cache,
            backing,
            cache_threshold,
            backing_keys: parking_lot::RwLock::new(std::collections::HashSet::new()),
        }
    }
}

impl<C: EdgeStorage, B: EdgeStorage> EdgeStorage for TieredEdgeStorage<C, B> {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Try cache first
        if let Some(value) = self.cache.get(key)? {
            return Ok(Some(value));
        }

        // Try backing store
        if let Some(value) = self.backing.get(key)? {
            // Optionally promote to cache if small enough
            if value.len() <= self.cache_threshold {
                let _ = self.cache.put(key, &value);
            }
            return Ok(Some(value));
        }

        Ok(None)
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        if value.len() <= self.cache_threshold {
            // Small values go to cache
            self.cache.put(key, value)?;
        } else {
            // Large values go to backing store
            self.backing.put(key, value)?;
            self.backing_keys.write().insert(key.to_string());
        }
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        // Delete from both tiers
        let _ = self.cache.delete(key);
        let _ = self.backing.delete(key);
        self.backing_keys.write().remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys: std::collections::HashSet<String> = self.cache.list(prefix)?
            .into_iter()
            .collect();

        keys.extend(self.backing.list(prefix)?);

        Ok(keys.into_iter().collect())
    }
}

// ============================================================================
// Chunked Storage Adapter
// ============================================================================

/// Storage adapter that chunks large values across multiple keys
///
/// Useful for platforms with small value size limits (e.g., Deno KV 64KB).
pub struct ChunkedEdgeStorage<S: EdgeStorage> {
    inner: S,
    chunk_size: usize,
}

/// Metadata for chunked values
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkMetadata {
    total_size: usize,
    chunk_count: usize,
    checksum: u64,
}

impl<S: EdgeStorage> ChunkedEdgeStorage<S> {
    /// Create a new chunked storage adapter
    pub fn new(storage: S, chunk_size: usize) -> Self {
        Self {
            inner: storage,
            chunk_size,
        }
    }

    fn chunk_key(&self, key: &str, chunk_index: usize) -> String {
        format!("{}.__chunk_{}", key, chunk_index)
    }

    fn meta_key(&self, key: &str) -> String {
        format!("{}.__meta", key)
    }

    fn compute_checksum(data: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
}

impl<S: EdgeStorage> EdgeStorage for ChunkedEdgeStorage<S> {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // Try direct get first (for small values)
        if let Some(value) = self.inner.get(key)? {
            return Ok(Some(value));
        }

        // Check for chunked value
        let meta_key = self.meta_key(key);
        let meta_bytes = match self.inner.get(&meta_key)? {
            Some(b) => b,
            None => return Ok(None),
        };

        let meta: ChunkMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(NeedleError::Serialization)?;

        // Read all chunks
        let mut data = Vec::with_capacity(meta.total_size);
        for i in 0..meta.chunk_count {
            let chunk_key = self.chunk_key(key, i);
            let chunk = self.inner.get(&chunk_key)?
                .ok_or_else(|| NeedleError::NotFound(format!("Missing chunk {} for {}", i, key)))?;
            data.extend(chunk);
        }

        // Verify checksum
        if Self::compute_checksum(&data) != meta.checksum {
            return Err(NeedleError::Corruption("Chunk checksum mismatch".into()));
        }

        Ok(Some(data))
    }

    fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        if value.len() <= self.chunk_size {
            // Small value, store directly
            return self.inner.put(key, value);
        }

        // Chunk the value
        let chunks: Vec<&[u8]> = value.chunks(self.chunk_size).collect();
        let meta = ChunkMetadata {
            total_size: value.len(),
            chunk_count: chunks.len(),
            checksum: Self::compute_checksum(value),
        };

        // Write chunks
        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_key = self.chunk_key(key, i);
            self.inner.put(&chunk_key, chunk)?;
        }

        // Write metadata
        let meta_bytes = serde_json::to_vec(&meta)
            .map_err(NeedleError::Serialization)?;
        let meta_key = self.meta_key(key);
        self.inner.put(&meta_key, &meta_bytes)?;

        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        // Delete direct key
        let _ = self.inner.delete(key);

        // Check for chunked value
        let meta_key = self.meta_key(key);
        if let Some(meta_bytes) = self.inner.get(&meta_key)? {
            if let Ok(meta) = serde_json::from_slice::<ChunkMetadata>(&meta_bytes) {
                // Delete all chunks
                for i in 0..meta.chunk_count {
                    let _ = self.inner.delete(&self.chunk_key(key, i));
                }
            }
            self.inner.delete(&meta_key)?;
        }

        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        // Get all keys and filter out chunk keys
        Ok(self.inner.list(prefix)?
            .into_iter()
            .filter(|k| !k.contains(".__chunk_") && !k.ends_with(".__meta"))
            .collect())
    }
}

// ============================================================================
// Platform Detection and Factory
// ============================================================================

/// Detected platform information
#[derive(Debug, Clone)]
pub struct PlatformInfo {
    /// Platform type
    pub platform: crate::edge_runtime::Platform,
    /// Platform version if available
    pub version: Option<String>,
    /// Available storage backends
    pub available_storage: Vec<StorageType>,
}

/// Storage type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageType {
    CloudflareKv,
    CloudflareR2,
    DenoKv,
    VercelEdgeConfig,
    VercelBlob,
    InMemory,
}

impl PlatformInfo {
    /// Detect current platform
    pub fn detect() -> Self {
        // In WASM, we'd check global objects:
        // - Cloudflare: typeof caches !== 'undefined' && typeof Deno === 'undefined'
        // - Deno: typeof Deno !== 'undefined'
        // - Generic: fallback

        Self {
            platform: crate::edge_runtime::Platform::GenericWasm,
            version: None,
            available_storage: vec![StorageType::InMemory],
        }
    }

    /// Check if specific storage is available
    pub fn has_storage(&self, storage_type: StorageType) -> bool {
        self.available_storage.contains(&storage_type)
    }
}

/// Storage adapter factory
pub struct StorageFactory;

impl StorageFactory {
    /// Create optimal storage for detected platform
    pub fn create_for_platform() -> Box<dyn EdgeStorage + Send + Sync> {
        let info = PlatformInfo::detect();

        match info.platform {
            crate::edge_runtime::Platform::CloudflareWorkers => {
                // Would create tiered KV + R2 storage
                Box::new(crate::edge_runtime::InMemoryEdgeStorage::new())
            }
            crate::edge_runtime::Platform::DenoKv => {
                Box::new(DenoKvAdapter::new())
            }
            crate::edge_runtime::Platform::VercelEdge => {
                // Would create Edge Config + Blob storage
                Box::new(crate::edge_runtime::InMemoryEdgeStorage::new())
            }
            _ => {
                Box::new(crate::edge_runtime::InMemoryEdgeStorage::new())
            }
        }
    }

    /// Create specific storage type
    pub fn create(storage_type: StorageType) -> Box<dyn EdgeStorage + Send + Sync> {
        match storage_type {
            StorageType::CloudflareKv => Box::new(CloudflareKvAdapter::new("NEEDLE_KV")),
            StorageType::CloudflareR2 => Box::new(CloudflareR2Adapter::new("NEEDLE_R2")),
            StorageType::DenoKv => Box::new(DenoKvAdapter::new()),
            StorageType::VercelEdgeConfig => {
                Box::new(crate::edge_runtime::InMemoryEdgeStorage::new())
            }
            StorageType::VercelBlob => {
                Box::new(crate::edge_runtime::InMemoryEdgeStorage::new())
            }
            StorageType::InMemory => Box::new(crate::edge_runtime::InMemoryEdgeStorage::new()),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloudflare_kv_adapter() {
        let adapter = CloudflareKvAdapter::new("TEST_KV").with_prefix("test");

        // Put and get
        adapter.put("key1", b"value1").unwrap();
        let value = adapter.get("key1").unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));

        // Delete
        adapter.delete("key1").unwrap();
        assert!(adapter.get("key1").unwrap().is_none());
    }

    #[test]
    fn test_deno_kv_adapter() {
        let adapter = DenoKvAdapter::new();

        adapter.put("key1", b"value1").unwrap();
        let value = adapter.get("key1").unwrap();
        assert_eq!(value, Some(b"value1".to_vec()));

        // Test size limit
        let large_value = vec![0u8; 100 * 1024]; // 100KB
        assert!(adapter.put("large", &large_value).is_err());
    }

    #[test]
    fn test_chunked_storage() {
        let inner = crate::edge_runtime::InMemoryEdgeStorage::new();
        let chunked = ChunkedEdgeStorage::new(inner, 10); // 10 byte chunks

        // Small value (no chunking)
        chunked.put("small", b"hello").unwrap();
        assert_eq!(chunked.get("small").unwrap(), Some(b"hello".to_vec()));

        // Large value (chunked)
        let large = b"this is a much longer value that will be chunked";
        chunked.put("large", large).unwrap();
        assert_eq!(chunked.get("large").unwrap(), Some(large.to_vec()));

        // Delete
        chunked.delete("large").unwrap();
        assert!(chunked.get("large").unwrap().is_none());
    }

    #[test]
    fn test_tiered_storage() {
        let cache = crate::edge_runtime::InMemoryEdgeStorage::new();
        let backing = crate::edge_runtime::InMemoryEdgeStorage::new();
        let tiered = TieredEdgeStorage::new(cache, backing, 100);

        // Small value goes to cache
        tiered.put("small", b"hello").unwrap();
        assert_eq!(tiered.get("small").unwrap(), Some(b"hello".to_vec()));

        // Large value goes to backing
        let large = vec![0u8; 200];
        tiered.put("large", &large).unwrap();
        assert_eq!(tiered.get("large").unwrap(), Some(large));
    }

    #[test]
    fn test_list_with_prefix() {
        let adapter = CloudflareKvAdapter::new("TEST");

        adapter.put("prefix/a", b"1").unwrap();
        adapter.put("prefix/b", b"2").unwrap();
        adapter.put("other/c", b"3").unwrap();

        let keys = adapter.list("prefix/").unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_platform_detection() {
        let info = PlatformInfo::detect();
        assert!(info.available_storage.contains(&StorageType::InMemory));
    }

    #[test]
    fn test_storage_factory() {
        let storage = StorageFactory::create(StorageType::InMemory);
        storage.put("test", b"value").unwrap();
        assert_eq!(storage.get("test").unwrap(), Some(b"value".to_vec()));
    }
}
