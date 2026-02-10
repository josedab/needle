//! Cloud Storage Backends
//!
//! Abstract storage layer supporting local and cloud backends for Needle vector database.
//! Provides a unified interface for storing vector data across different storage providers
//! with features like connection pooling, retry logic, and caching.
//!
//! # Supported Backends
//!
//! - **LocalBackend**: File system storage for development and small deployments
//! - **S3Backend**: AWS S3 compatible storage (mock/stub)
//! - **GCSBackend**: Google Cloud Storage (mock/stub)
//! - **AzureBlobBackend**: Azure Blob Storage (mock/stub)
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::cloud_storage::{StorageBackend, LocalBackend, StorageConfig};
//!
//! #[tokio::main]
//! async fn main() -> needle::Result<()> {
//!     // Create a local backend
//!     let backend = LocalBackend::new("/path/to/storage")?;
//!
//!     // Write data
//!     backend.write("vectors/collection1/chunk_001", b"vector data").await?;
//!
//!     // Read data
//!     let data = backend.read("vectors/collection1/chunk_001").await?;
//!
//!     // List objects
//!     let keys = backend.list("vectors/collection1/").await?;
//!
//!     // Check existence
//!     let exists = backend.exists("vectors/collection1/chunk_001").await?;
//!
//!     // Delete
//!     backend.delete("vectors/collection1/chunk_001").await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Caching
//!
//! ```rust,ignore
//! use needle::cloud_storage::{CachedBackend, LocalBackend, CacheConfig};
//!
//! let backend = LocalBackend::new("/path/to/storage")?;
//! let cached = CachedBackend::new(backend, CacheConfig::default());
//!
//! // Reads are cached automatically
//! let data = cached.read("key").await?;
//! ```
//!
//! # Retry Logic
//!
//! All cloud backends include automatic retry with exponential backoff for transient failures.

mod common;
mod config;
mod local;
mod cached;
mod s3;
mod gcs;
mod azure;

pub use config::{
    CacheConfig, ConnectionHandle, ConnectionPool, PoolStats, RetryPolicy, StorageBackend,
    StorageConfig,
};
pub use local::LocalBackend;
pub use cached::{
    CacheStats, CacheTier, CachedBackend, TieredCacheBackend, TieredCacheConfig, TieredCacheStats,
};
pub use s3::{S3Backend, S3Config};
pub use gcs::{GCSBackend, GCSConfig};
pub use azure::{AzureBlobBackend, AzureBlobConfig};

use crate::error::Result;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Streaming Support
// ============================================================================

/// Chunk for streaming uploads/downloads.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Chunk index.
    pub index: usize,
    /// Chunk data.
    pub data: Vec<u8>,
    /// Is this the last chunk.
    pub is_last: bool,
}

/// Streaming reader for large objects.
pub struct StreamingReader<'a> {
    backend: &'a dyn StorageBackend,
    key: String,
    chunk_size: usize,
    current_offset: usize,
    total_size: Option<usize>,
}

impl<'a> StreamingReader<'a> {
    /// Create a new streaming reader.
    pub fn new(backend: &'a dyn StorageBackend, key: &str, chunk_size: usize) -> Self {
        Self {
            backend,
            key: key.to_string(),
            chunk_size,
            current_offset: 0,
            total_size: None,
        }
    }

    /// Read the next chunk.
    pub async fn next_chunk(&mut self) -> Result<Option<StreamChunk>> {
        // For simplicity, read all data and chunk it
        // In production, use range requests
        if self.total_size.is_none() {
            let data = self.backend.read(&self.key).await?;
            self.total_size = Some(data.len());
        }

        // Safe: we just set total_size above if it was None
        let total = self.total_size.expect("total_size was just set");
        if self.current_offset >= total {
            return Ok(None);
        }

        let data = self.backend.read(&self.key).await?;
        let end = std::cmp::min(self.current_offset + self.chunk_size, total);
        let chunk_data = data[self.current_offset..end].to_vec();
        let is_last = end >= total;

        let chunk = StreamChunk {
            index: self.current_offset / self.chunk_size,
            data: chunk_data,
            is_last,
        };

        self.current_offset = end;
        Ok(Some(chunk))
    }
}

/// Streaming writer for large objects.
#[allow(dead_code)]
pub struct StreamingWriter<'a> {
    backend: &'a dyn StorageBackend,
    key: String,
    buffer: Vec<u8>,
    chunk_size: usize,
    chunks_written: usize,
}

impl<'a> StreamingWriter<'a> {
    /// Create a new streaming writer.
    pub fn new(backend: &'a dyn StorageBackend, key: &str, chunk_size: usize) -> Self {
        Self {
            backend,
            key: key.to_string(),
            buffer: Vec::new(),
            chunk_size,
            chunks_written: 0,
        }
    }

    /// Write a chunk of data.
    pub fn write_chunk(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.chunks_written += 1;
    }

    /// Finalize and flush all data.
    pub async fn finalize(self) -> Result<usize> {
        self.backend.write(&self.key, &self.buffer).await?;
        Ok(self.buffer.len())
    }

    /// Get number of chunks written.
    pub fn chunks_written(&self) -> usize {
        self.chunks_written
    }

    /// Get current buffer size.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================================
// Multipart Upload
// ============================================================================

/// Multipart upload state.
#[derive(Debug, Clone)]
pub struct MultipartUpload {
    /// Upload ID.
    pub upload_id: String,
    /// Target key.
    pub key: String,
    /// Parts uploaded.
    pub parts: Vec<MultipartPart>,
    /// Creation time.
    pub created_at: u64,
}

/// Part of a multipart upload.
#[derive(Debug, Clone)]
pub struct MultipartPart {
    /// Part number (1-indexed).
    pub part_number: usize,
    /// Part ETag (hash).
    pub etag: String,
    /// Part size.
    pub size: usize,
}

/// Multipart upload manager.
#[allow(dead_code)]
pub struct MultipartUploader<'a> {
    backend: &'a dyn StorageBackend,
    upload: MultipartUpload,
    part_size: usize,
    buffer: Vec<u8>,
}

impl<'a> MultipartUploader<'a> {
    /// Start a new multipart upload.
    pub fn new(backend: &'a dyn StorageBackend, key: &str, part_size: usize) -> Self {
        let upload_id = format!("mpu_{}", now_timestamp());

        Self {
            backend,
            upload: MultipartUpload {
                upload_id,
                key: key.to_string(),
                parts: Vec::new(),
                created_at: now_timestamp(),
            },
            part_size,
            buffer: Vec::new(),
        }
    }

    /// Upload a part.
    pub async fn upload_part(&mut self, data: &[u8]) -> Result<MultipartPart> {
        let part_number = self.upload.parts.len() + 1;
        let part_key = format!("{}._part_{}", self.upload.key, part_number);

        // Write part
        self.backend.write(&part_key, data).await?;

        // Calculate ETag (simple hash)
        let etag = simple_hash(data);

        let part = MultipartPart {
            part_number,
            etag,
            size: data.len(),
        };

        self.upload.parts.push(part.clone());
        Ok(part)
    }

    /// Complete the multipart upload.
    pub async fn complete(self) -> Result<()> {
        // Concatenate all parts
        let mut final_data = Vec::new();

        for part in &self.upload.parts {
            let part_key = format!("{}._part_{}", self.upload.key, part.part_number);
            let part_data = self.backend.read(&part_key).await?;
            final_data.extend_from_slice(&part_data);

            // Clean up part
            self.backend.delete(&part_key).await?;
        }

        // Write final object
        self.backend.write(&self.upload.key, &final_data).await?;
        Ok(())
    }

    /// Abort the multipart upload.
    pub async fn abort(self) -> Result<()> {
        // Clean up all parts
        for part in &self.upload.parts {
            let part_key = format!("{}._part_{}", self.upload.key, part.part_number);
            self.backend.delete(&part_key).await?;
        }
        Ok(())
    }

    /// Get upload ID.
    pub fn upload_id(&self) -> &str {
        &self.upload.upload_id
    }

    /// Get parts count.
    pub fn parts_count(&self) -> usize {
        self.upload.parts.len()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current timestamp.
fn now_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Simple hash function for ETags.
fn simple_hash(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use cached::key_to_filename;
    use tempfile::TempDir;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use crate::error::NeedleError;

    // Helper to create a test backend
    fn create_test_backend() -> (TempDir, LocalBackend) {
        let temp_dir = TempDir::new().unwrap();
        let backend = LocalBackend::new(temp_dir.path()).unwrap();
        (temp_dir, backend)
    }

    #[test]
    fn test_local_backend_write_read() {
        let (_temp_dir, backend) = create_test_backend();

        // Use a simple blocking approach for tests
        let write_future = backend.write("test/key1", b"hello world");
        futures::executor::block_on(write_future).unwrap();

        let read_future = backend.read("test/key1");
        let data = futures::executor::block_on(read_future).unwrap();

        assert_eq!(data, b"hello world");
    }

    #[test]
    fn test_local_backend_exists() {
        let (_temp_dir, backend) = create_test_backend();

        // Key doesn't exist initially
        let exists_future = backend.exists("test/nonexistent");
        assert!(!futures::executor::block_on(exists_future).unwrap());

        // Write and check again
        let write_future = backend.write("test/exists_key", b"data");
        futures::executor::block_on(write_future).unwrap();

        let exists_future = backend.exists("test/exists_key");
        assert!(futures::executor::block_on(exists_future).unwrap());
    }

    #[test]
    fn test_local_backend_delete() {
        let (_temp_dir, backend) = create_test_backend();

        // Write data
        let write_future = backend.write("test/to_delete", b"data");
        futures::executor::block_on(write_future).unwrap();

        // Delete
        let delete_future = backend.delete("test/to_delete");
        futures::executor::block_on(delete_future).unwrap();

        // Should not exist
        let exists_future = backend.exists("test/to_delete");
        assert!(!futures::executor::block_on(exists_future).unwrap());
    }

    #[test]
    fn test_local_backend_list() {
        let (_temp_dir, backend) = create_test_backend();

        // Write multiple keys
        for i in 0..5 {
            let key = format!("prefix/key_{}", i);
            let write_future = backend.write(&key, b"data");
            futures::executor::block_on(write_future).unwrap();
        }

        // Write with different prefix
        let write_future = backend.write("other/key", b"data");
        futures::executor::block_on(write_future).unwrap();

        // List with prefix
        let list_future = backend.list("prefix/");
        let keys = futures::executor::block_on(list_future).unwrap();

        assert_eq!(keys.len(), 5);
        for key in &keys {
            assert!(key.starts_with("prefix/"));
        }
    }

    #[test]
    fn test_local_backend_read_nonexistent() {
        let (_temp_dir, backend) = create_test_backend();

        let read_future = backend.read("nonexistent_key");
        let result = futures::executor::block_on(read_future);

        assert!(result.is_err());
        match result {
            Err(NeedleError::NotFound(_)) => {}
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_s3_backend_basic_operations() {
        let config = S3Config {
            bucket: "test-bucket".to_string(),
            region: "us-west-2".to_string(),
            ..Default::default()
        };
        let backend = S3Backend::new(config);

        // Write
        let write_future = backend.write("vectors/chunk_1", b"vector data");
        futures::executor::block_on(write_future).unwrap();

        // Read
        let read_future = backend.read("vectors/chunk_1");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"vector data");

        // Exists
        let exists_future = backend.exists("vectors/chunk_1");
        assert!(futures::executor::block_on(exists_future).unwrap());

        // List
        let list_future = backend.list("vectors/");
        let keys = futures::executor::block_on(list_future).unwrap();
        assert_eq!(keys.len(), 1);

        // Delete
        let delete_future = backend.delete("vectors/chunk_1");
        futures::executor::block_on(delete_future).unwrap();

        let exists_future = backend.exists("vectors/chunk_1");
        assert!(!futures::executor::block_on(exists_future).unwrap());
    }

    #[test]
    fn test_gcs_backend_basic_operations() {
        let config = GCSConfig {
            project_id: "test-project".to_string(),
            bucket: "test-bucket".to_string(),
            ..Default::default()
        };
        let backend = GCSBackend::new(config);

        // Write and read
        let write_future = backend.write("data/file1", b"test content");
        futures::executor::block_on(write_future).unwrap();

        let read_future = backend.read("data/file1");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"test content");
    }

    #[test]
    fn test_azure_backend_basic_operations() {
        let config = AzureBlobConfig {
            account_name: "testaccount".to_string(),
            container: "testcontainer".to_string(),
            ..Default::default()
        };
        let backend = AzureBlobBackend::new(config);

        // Write and read
        let write_future = backend.write("blobs/data1", b"azure data");
        futures::executor::block_on(write_future).unwrap();

        let read_future = backend.read("blobs/data1");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"azure data");
    }

    #[test]
    fn test_cached_backend() {
        let (_temp_dir, inner) = create_test_backend();
        let cached = CachedBackend::new(inner, CacheConfig::default());

        // Write data (this also populates the cache)
        let write_future = cached.write("cache/key1", b"cached data");
        futures::executor::block_on(write_future).unwrap();

        // First read - cache hit (cached by write)
        let read_future = cached.read("cache/key1");
        let _ = futures::executor::block_on(read_future).unwrap();

        // Second read - also cache hit
        let read_future = cached.read("cache/key1");
        let _ = futures::executor::block_on(read_future).unwrap();

        // Check stats - both reads were hits since write populates cache
        assert_eq!(cached.stats().hits(), 2);
        assert_eq!(cached.stats().misses(), 0);
        assert_eq!(cached.stats().hit_rate(), 1.0);
    }

    #[test]
    fn test_cache_invalidation() {
        let (_temp_dir, inner) = create_test_backend();
        let cached = CachedBackend::new(inner, CacheConfig::default());

        // Write and read to populate cache
        let write_future = cached.write("cache/key2", b"data");
        futures::executor::block_on(write_future).unwrap();

        let read_future = cached.read("cache/key2");
        let _ = futures::executor::block_on(read_future).unwrap();

        assert!(cached.is_cached("cache/key2"));

        // Invalidate
        cached.invalidate("cache/key2");
        assert!(!cached.is_cached("cache/key2"));
    }

    #[test]
    fn test_cache_clear() {
        let (_temp_dir, inner) = create_test_backend();
        let cached = CachedBackend::new(inner, CacheConfig::default());

        // Populate cache
        for i in 0..5 {
            let key = format!("cache/clear_test_{}", i);
            let write_future = cached.write(&key, b"data");
            futures::executor::block_on(write_future).unwrap();
        }

        assert!(cached.stats().bytes_cached() > 0);

        // Clear cache
        cached.clear_cache();
        assert_eq!(cached.stats().bytes_cached(), 0);
    }

    #[test]
    fn test_connection_pool() {
        let pool = ConnectionPool::new(10, 2, Duration::from_secs(30));

        assert_eq!(pool.max_size(), 10);
        assert_eq!(pool.min_size(), 2);

        // Acquire connection
        let _conn = pool.acquire().unwrap();

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.requests_served, 1);
    }

    #[test]
    fn test_connection_pool_release() {
        let pool = ConnectionPool::new(10, 2, Duration::from_secs(30));

        let initial_idle = pool.stats().idle_connections;

        {
            let _conn = pool.acquire().unwrap();
            assert_eq!(pool.stats().active_connections, 1);
        }

        // Connection released
        assert_eq!(pool.stats().active_connections, 0);
        assert_eq!(pool.stats().idle_connections, initial_idle);
    }

    #[test]
    fn test_retry_policy() {
        let policy = RetryPolicy::default();

        // Test successful operation
        let result = futures::executor::block_on(policy.execute(|| async { Ok::<_, NeedleError>(42) }));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_streaming_writer() {
        let (_temp_dir, backend) = create_test_backend();

        let mut writer = StreamingWriter::new(&backend, "stream/large_file", 1024);

        // Write chunks
        writer.write_chunk(b"chunk 1 data ");
        writer.write_chunk(b"chunk 2 data ");
        writer.write_chunk(b"chunk 3 data");

        assert_eq!(writer.chunks_written(), 3);
        assert!(writer.buffer_size() > 0);

        // Finalize
        let size = futures::executor::block_on(writer.finalize()).unwrap();
        assert!(size > 0);

        // Verify data
        let read_future = backend.read("stream/large_file");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"chunk 1 data chunk 2 data chunk 3 data");
    }

    #[test]
    fn test_multipart_uploader() {
        let (_temp_dir, backend) = create_test_backend();

        let mut uploader = MultipartUploader::new(&backend, "multipart/file", 1024);

        // Upload parts
        let part1 = futures::executor::block_on(uploader.upload_part(b"part 1 data ")).unwrap();
        assert_eq!(part1.part_number, 1);

        let part2 = futures::executor::block_on(uploader.upload_part(b"part 2 data")).unwrap();
        assert_eq!(part2.part_number, 2);

        assert_eq!(uploader.parts_count(), 2);

        // Complete
        futures::executor::block_on(uploader.complete()).unwrap();

        // Verify final data
        let read_future = backend.read("multipart/file");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"part 1 data part 2 data");
    }

    #[test]
    fn test_multipart_abort() {
        let (_temp_dir, backend) = create_test_backend();

        let mut uploader = MultipartUploader::new(&backend, "multipart/abort_test", 1024);

        // Upload a part
        futures::executor::block_on(uploader.upload_part(b"part data")).unwrap();

        // Abort
        futures::executor::block_on(uploader.abort()).unwrap();

        // Final file should not exist
        let exists_future = backend.exists("multipart/abort_test");
        assert!(!futures::executor::block_on(exists_future).unwrap());
    }

    #[test]
    fn test_storage_config_defaults() {
        let config = StorageConfig::default();

        assert_eq!(config.max_retries, 3);
        assert_eq!(config.connection_timeout, Duration::from_secs(30));
        assert_eq!(config.read_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_s3_config() {
        let config = S3Config {
            region: "eu-west-1".to_string(),
            bucket: "my-bucket".to_string(),
            endpoint: Some("http://localhost:9000".to_string()),
            path_style: true,
            ..Default::default()
        };

        let backend = S3Backend::new(config);
        assert_eq!(backend.bucket(), "my-bucket");
        assert_eq!(backend.region(), "eu-west-1");
    }

    #[test]
    fn test_gcs_config() {
        let config = GCSConfig {
            project_id: "my-gcp-project".to_string(),
            bucket: "my-gcs-bucket".to_string(),
            credentials_path: Some("/path/to/creds.json".to_string()),
            ..Default::default()
        };

        let backend = GCSBackend::new(config);
        assert_eq!(backend.bucket(), "my-gcs-bucket");
        assert_eq!(backend.project_id(), "my-gcp-project");
    }

    #[test]
    fn test_azure_config() {
        let config = AzureBlobConfig {
            account_name: "myaccount".to_string(),
            container: "mycontainer".to_string(),
            account_key: Some("secret_key".to_string()),
            ..Default::default()
        };

        let backend = AzureBlobBackend::new(config);
        assert_eq!(backend.container(), "mycontainer");
        assert_eq!(backend.account_name(), "myaccount");
    }

    #[test]
    fn test_cache_config() {
        let config = CacheConfig {
            max_size: 50 * 1024 * 1024,
            default_ttl: Duration::from_secs(600),
            enable_stats: true,
        };

        assert_eq!(config.max_size, 50 * 1024 * 1024);
        assert_eq!(config.default_ttl, Duration::from_secs(600));
    }

    #[test]
    fn test_simple_hash() {
        let hash1 = simple_hash(b"hello world");
        let hash2 = simple_hash(b"hello world");
        let hash3 = simple_hash(b"different data");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_pool_stats() {
        let pool = ConnectionPool::new(20, 5, Duration::from_secs(60));

        let stats = pool.stats();
        assert_eq!(stats.idle_connections, 5);
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.requests_served, 0);
    }

    #[test]
    fn test_nested_key_paths() {
        let (_temp_dir, backend) = create_test_backend();

        // Write to deeply nested path
        let key = "a/b/c/d/e/file.dat";
        let write_future = backend.write(key, b"nested data");
        futures::executor::block_on(write_future).unwrap();

        // Read back
        let read_future = backend.read(key);
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"nested data");
    }

    #[test]
    fn test_empty_prefix_list() {
        let (_temp_dir, backend) = create_test_backend();

        // Write some data
        let write_future = backend.write("file1", b"data1");
        futures::executor::block_on(write_future).unwrap();

        let write_future = backend.write("file2", b"data2");
        futures::executor::block_on(write_future).unwrap();

        // List with empty prefix should return all files
        let list_future = backend.list("");
        let keys = futures::executor::block_on(list_future).unwrap();
        assert!(keys.len() >= 2);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let (_temp_dir, backend) = create_test_backend();

        let key = "overwrite/test";

        // Write initial data
        let write_future = backend.write(key, b"initial data");
        futures::executor::block_on(write_future).unwrap();

        // Overwrite
        let write_future = backend.write(key, b"new data");
        futures::executor::block_on(write_future).unwrap();

        // Read should return new data
        let read_future = backend.read(key);
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"new data");
    }

    #[test]
    fn test_delete_nonexistent_key() {
        let (_temp_dir, backend) = create_test_backend();

        // Delete should be idempotent
        let delete_future = backend.delete("nonexistent/key");
        let result = futures::executor::block_on(delete_future);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Tiered Cache Tests
    // ========================================================================

    #[test]
    fn test_tiered_cache_config_defaults() {
        let config = TieredCacheConfig::default();

        assert_eq!(config.memory_max_size, 100 * 1024 * 1024);
        assert_eq!(config.ssd_max_size, 1024 * 1024 * 1024);
        assert_eq!(config.memory_ttl, Duration::from_secs(300));
        assert_eq!(config.ssd_ttl, Duration::from_secs(3600));
        assert!(config.enable_prefetch);
        assert_eq!(config.promotion_threshold, 3);
    }

    #[test]
    fn test_tiered_cache_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let inner = LocalBackend::new(temp_dir.path().join("storage")).unwrap();

        let cache_config = TieredCacheConfig {
            ssd_cache_path: temp_dir.path().join("cache"),
            ..Default::default()
        };

        let cached = TieredCacheBackend::new(inner, cache_config).unwrap();

        // Write data
        let write_future = cached.write("tiered/key1", b"tiered cache data");
        futures::executor::block_on(write_future).unwrap();

        // First read - should be a hit (cached on write)
        let read_future = cached.read("tiered/key1");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, b"tiered cache data");

        // Check stats
        assert!(cached.stats().memory_hits.load(Ordering::Relaxed) > 0
            || cached.stats().ssd_hits.load(Ordering::Relaxed) > 0
            || cached.stats().origin_fetches.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_tiered_cache_memory_tier() {
        let temp_dir = TempDir::new().unwrap();
        let inner = LocalBackend::new(temp_dir.path().join("storage")).unwrap();

        let cache_config = TieredCacheConfig {
            ssd_cache_path: temp_dir.path().join("cache"),
            memory_max_size: 10 * 1024 * 1024, // 10MB
            ..Default::default()
        };

        let cached = TieredCacheBackend::new(inner, cache_config).unwrap();

        // Write small data (should go to memory)
        let small_data = b"small data for memory tier";
        let write_future = cached.write("memory/small", small_data);
        futures::executor::block_on(write_future).unwrap();

        // Read should hit memory
        let read_future = cached.read("memory/small");
        let data = futures::executor::block_on(read_future).unwrap();
        assert_eq!(data, small_data);

        // Memory bytes should be tracked
        assert!(cached.stats().memory_bytes.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_tiered_cache_stats() {
        let stats = TieredCacheStats::default();

        assert_eq!(stats.memory_hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.ssd_hits.load(Ordering::Relaxed), 0);
        assert_eq!(stats.origin_fetches.load(Ordering::Relaxed), 0);
        assert_eq!(stats.hit_rate(), 0.0);

        // Simulate some hits
        stats.memory_hits.fetch_add(8, Ordering::Relaxed);
        stats.ssd_hits.fetch_add(2, Ordering::Relaxed);

        // Hit rate should be 100% (no origin fetches)
        assert_eq!(stats.hit_rate(), 1.0);

        // Add an origin fetch
        stats.origin_fetches.fetch_add(1, Ordering::Relaxed);

        // Hit rate should now be ~90.9%
        let hit_rate = stats.hit_rate();
        assert!(hit_rate > 0.9 && hit_rate < 0.92);
    }

    #[test]
    fn test_tiered_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let inner = LocalBackend::new(temp_dir.path().join("storage")).unwrap();

        let cache_config = TieredCacheConfig {
            ssd_cache_path: temp_dir.path().join("cache"),
            ..Default::default()
        };

        let cached = TieredCacheBackend::new(inner, cache_config).unwrap();

        // Write some data
        let write_future = cached.write("clear/test1", b"data1");
        futures::executor::block_on(write_future).unwrap();

        let write_future = cached.write("clear/test2", b"data2");
        futures::executor::block_on(write_future).unwrap();

        assert!(cached.stats().memory_bytes.load(Ordering::Relaxed) > 0);

        // Clear all caches
        cached.clear_all().unwrap();

        assert_eq!(cached.stats().memory_bytes.load(Ordering::Relaxed), 0);
        assert_eq!(cached.stats().ssd_bytes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_tiered_cache_delete() {
        let temp_dir = TempDir::new().unwrap();
        let inner = LocalBackend::new(temp_dir.path().join("storage")).unwrap();

        let cache_config = TieredCacheConfig {
            ssd_cache_path: temp_dir.path().join("cache"),
            ..Default::default()
        };

        let cached = TieredCacheBackend::new(inner, cache_config).unwrap();

        // Write data
        let write_future = cached.write("delete/test", b"to be deleted");
        futures::executor::block_on(write_future).unwrap();

        let initial_bytes = cached.stats().memory_bytes.load(Ordering::Relaxed);
        assert!(initial_bytes > 0);

        // Delete
        let delete_future = cached.delete("delete/test");
        futures::executor::block_on(delete_future).unwrap();

        // Cache should be updated
        let final_bytes = cached.stats().memory_bytes.load(Ordering::Relaxed);
        assert!(final_bytes < initial_bytes);

        // Data should not exist
        let read_future = cached.read("delete/test");
        let result = futures::executor::block_on(read_future);
        assert!(result.is_err());
    }

    #[test]
    fn test_key_to_filename() {
        assert_eq!(key_to_filename("simple"), "simple");
        assert_eq!(key_to_filename("path/to/file"), "path_to_file");
        assert_eq!(key_to_filename("a\\b:c*d?e"), "a_b_c_d_e");
        assert_eq!(key_to_filename("<test>|file"), "_test__file");
    }

    #[test]
    fn test_cache_tier_enum() {
        let memory_tier = CacheTier::Memory;
        let ssd_tier = CacheTier::Ssd;
        let origin_tier = CacheTier::Origin;

        assert_eq!(memory_tier, CacheTier::Memory);
        assert_ne!(memory_tier, ssd_tier);
        assert_ne!(ssd_tier, origin_tier);
    }

    #[test]
    fn test_s3_backend_is_connected() {
        let config = S3Config::default();
        let backend = S3Backend::new(config);

        // Mock backend should not be connected
        assert!(!backend.is_connected());
    }

    #[test]
    fn test_gcs_backend_is_connected() {
        let config = GCSConfig::default();
        let backend = GCSBackend::new(config);

        // Mock backend should not be connected
        assert!(!backend.is_connected());
    }

    #[test]
    fn test_azure_backend_is_connected() {
        let config = AzureBlobConfig::default();
        let backend = AzureBlobBackend::new(config);

        // Mock backend should not be connected
        assert!(!backend.is_connected());
    }
}
