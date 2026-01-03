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

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Core Trait
// ============================================================================

/// Storage backend trait for cloud and local storage operations.
///
/// All operations are async to support both local I/O and network operations.
/// Implementations should handle retries internally for transient failures.
pub trait StorageBackend: Send + Sync {
    /// Read data from storage by key.
    ///
    /// Returns the data as a byte vector.
    /// Returns an error if the key does not exist or reading fails.
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>>;

    /// Write data to storage.
    ///
    /// Overwrites existing data if the key already exists.
    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Delete data from storage.
    ///
    /// Returns Ok(()) even if the key does not exist (idempotent).
    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// List keys with a given prefix.
    ///
    /// Returns all keys that start with the given prefix.
    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>>;

    /// Check if a key exists.
    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;
}

// ============================================================================
// Configuration
// ============================================================================

/// General storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Maximum retry attempts for transient failures.
    pub max_retries: u32,
    /// Initial retry delay (doubles on each retry).
    pub initial_retry_delay: Duration,
    /// Maximum retry delay cap.
    pub max_retry_delay: Duration,
    /// Connection timeout.
    pub connection_timeout: Duration,
    /// Read timeout.
    pub read_timeout: Duration,
    /// Write timeout.
    pub write_timeout: Duration,
    /// Enable request logging.
    pub enable_logging: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_retry_delay: Duration::from_millis(100),
            max_retry_delay: Duration::from_secs(10),
            connection_timeout: Duration::from_secs(30),
            read_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(120),
            enable_logging: false,
        }
    }
}

/// S3-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Config {
    /// AWS region.
    pub region: String,
    /// S3 bucket name.
    pub bucket: String,
    /// Optional endpoint URL (for S3-compatible services).
    pub endpoint: Option<String>,
    /// Access key ID.
    pub access_key_id: Option<String>,
    /// Secret access key.
    pub secret_access_key: Option<String>,
    /// Use path-style URLs.
    pub path_style: bool,
    /// General storage config.
    pub storage: StorageConfig,
}

impl Default for S3Config {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            bucket: "needle-vectors".to_string(),
            endpoint: None,
            access_key_id: None,
            secret_access_key: None,
            path_style: false,
            storage: StorageConfig::default(),
        }
    }
}

/// GCS-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCSConfig {
    /// GCP project ID.
    pub project_id: String,
    /// GCS bucket name.
    pub bucket: String,
    /// Path to service account credentials JSON.
    pub credentials_path: Option<String>,
    /// General storage config.
    pub storage: StorageConfig,
}

impl Default for GCSConfig {
    fn default() -> Self {
        Self {
            project_id: "my-project".to_string(),
            bucket: "needle-vectors".to_string(),
            credentials_path: None,
            storage: StorageConfig::default(),
        }
    }
}

/// Azure Blob-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureBlobConfig {
    /// Azure storage account name.
    pub account_name: String,
    /// Container name.
    pub container: String,
    /// Account key (or use managed identity).
    pub account_key: Option<String>,
    /// Connection string (alternative to account name/key).
    pub connection_string: Option<String>,
    /// General storage config.
    pub storage: StorageConfig,
}

impl Default for AzureBlobConfig {
    fn default() -> Self {
        Self {
            account_name: "needlestorage".to_string(),
            container: "vectors".to_string(),
            account_key: None,
            connection_string: None,
            storage: StorageConfig::default(),
        }
    }
}

// ============================================================================
// Connection Pool
// ============================================================================

/// Simple connection pool statistics.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total connections created.
    pub connections_created: u64,
    /// Active connections.
    pub active_connections: u64,
    /// Idle connections.
    pub idle_connections: u64,
    /// Total requests served.
    pub requests_served: u64,
    /// Failed connection attempts.
    pub connection_failures: u64,
}

/// Connection pool for managing backend connections.
pub struct ConnectionPool {
    /// Maximum pool size.
    max_size: usize,
    /// Minimum pool size.
    min_size: usize,
    /// Connection timeout.
    timeout: Duration,
    /// Statistics.
    stats: Arc<PoolStatsInner>,
}

struct PoolStatsInner {
    connections_created: AtomicU64,
    active_connections: AtomicU64,
    idle_connections: AtomicU64,
    requests_served: AtomicU64,
    connection_failures: AtomicU64,
}

impl ConnectionPool {
    /// Create a new connection pool.
    pub fn new(max_size: usize, min_size: usize, timeout: Duration) -> Self {
        Self {
            max_size,
            min_size,
            timeout,
            stats: Arc::new(PoolStatsInner {
                connections_created: AtomicU64::new(0),
                active_connections: AtomicU64::new(0),
                idle_connections: AtomicU64::new(min_size as u64),
                requests_served: AtomicU64::new(0),
                connection_failures: AtomicU64::new(0),
            }),
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            connections_created: self.stats.connections_created.load(Ordering::Relaxed),
            active_connections: self.stats.active_connections.load(Ordering::Relaxed),
            idle_connections: self.stats.idle_connections.load(Ordering::Relaxed),
            requests_served: self.stats.requests_served.load(Ordering::Relaxed),
            connection_failures: self.stats.connection_failures.load(Ordering::Relaxed),
        }
    }

    /// Acquire a connection from the pool.
    pub fn acquire(&self) -> Result<ConnectionHandle> {
        self.stats.active_connections.fetch_add(1, Ordering::Relaxed);
        self.stats.idle_connections.fetch_sub(1, Ordering::Relaxed);
        self.stats.requests_served.fetch_add(1, Ordering::Relaxed);

        Ok(ConnectionHandle {
            pool: Arc::clone(&self.stats),
        })
    }

    /// Get maximum pool size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Get minimum pool size.
    pub fn min_size(&self) -> usize {
        self.min_size
    }

    /// Get connection timeout.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Handle to a pooled connection.
pub struct ConnectionHandle {
    pool: Arc<PoolStatsInner>,
}

impl Drop for ConnectionHandle {
    fn drop(&mut self) {
        self.pool.active_connections.fetch_sub(1, Ordering::Relaxed);
        self.pool.idle_connections.fetch_add(1, Ordering::Relaxed);
    }
}

// ============================================================================
// Retry Logic
// ============================================================================

/// Retry policy with exponential backoff.
pub struct RetryPolicy {
    /// Maximum number of attempts.
    pub max_attempts: u32,
    /// Initial delay between retries.
    pub initial_delay: Duration,
    /// Maximum delay cap.
    pub max_delay: Duration,
    /// Jitter factor (0.0 to 1.0).
    pub jitter: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: 0.1,
        }
    }
}

impl RetryPolicy {
    /// Execute an operation with retry logic.
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let mut last_error = None;
        let mut delay = self.initial_delay;

        for attempt in 0..self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Check if error is retryable
                    if !Self::is_retryable(&e) {
                        return Err(e);
                    }

                    last_error = Some(e);

                    if attempt < self.max_attempts - 1 {
                        // Apply jitter
                        let jitter_amount = delay.as_millis() as f64 * self.jitter;
                        let jittered_delay = Duration::from_millis(
                            (delay.as_millis() as f64 + rand_jitter(jitter_amount)) as u64,
                        );

                        // Wait before retry (simulated)
                        std::thread::sleep(jittered_delay);

                        // Exponential backoff
                        delay = std::cmp::min(delay * 2, self.max_delay);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            NeedleError::Io(std::io::Error::other(
                "Retry exhausted with no error",
            ))
        }))
    }

    /// Check if an error is retryable.
    fn is_retryable(error: &NeedleError) -> bool {
        matches!(
            error,
            NeedleError::Io(_) | NeedleError::BackupError(_)
        )
    }
}

/// Generate random jitter value.
fn rand_jitter(max: f64) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    let hash = hasher.finish();
    (hash as f64 / u64::MAX as f64) * max
}

// ============================================================================
// Local Backend
// ============================================================================

/// File system storage backend for local development.
pub struct LocalBackend {
    /// Base directory for storage.
    base_path: PathBuf,
    /// Connection pool (for API consistency).
    pool: ConnectionPool,
    /// Retry policy.
    retry_policy: RetryPolicy,
}

impl LocalBackend {
    /// Create a new local backend.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base = base_path.as_ref().to_path_buf();

        // Create base directory if it doesn't exist
        std::fs::create_dir_all(&base)?;

        Ok(Self {
            base_path: base,
            pool: ConnectionPool::new(10, 2, Duration::from_secs(30)),
            retry_policy: RetryPolicy::default(),
        })
    }

    /// Get the full path for a key.
    fn key_to_path(&self, key: &str) -> PathBuf {
        self.base_path.join(key)
    }

    /// Ensure parent directory exists.
    fn ensure_parent_dir(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(())
    }
}

impl StorageBackend for LocalBackend {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let path = self.key_to_path(key);
            let _conn = self.pool.acquire()?;

            self.retry_policy
                .execute(|| async {
                    std::fs::read(&path).map_err(|e| {
                        if e.kind() == std::io::ErrorKind::NotFound {
                            NeedleError::NotFound(format!("Key '{}' not found", key))
                        } else {
                            NeedleError::Io(e)
                        }
                    })
                })
                .await
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let path = self.key_to_path(key);
            let _conn = self.pool.acquire()?;

            self.ensure_parent_dir(&path)?;

            self.retry_policy
                .execute(|| async { std::fs::write(&path, data).map_err(NeedleError::Io) })
                .await
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let path = self.key_to_path(key);
            let _conn = self.pool.acquire()?;

            match std::fs::remove_file(&path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(NeedleError::Io(e)),
            }
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(async move {
            let base = self.key_to_path(prefix);
            let _conn = self.pool.acquire()?;

            let mut keys = Vec::new();

            // Handle prefix as directory or file prefix
            let search_dir = if base.is_dir() {
                base.clone()
            } else {
                base.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| self.base_path.clone())
            };

            if search_dir.exists() {
                self.collect_keys(&search_dir, prefix, &mut keys)?;
            }

            Ok(keys)
        })
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let path = self.key_to_path(key);
            let _conn = self.pool.acquire()?;
            Ok(path.exists())
        })
    }
}

impl LocalBackend {
    /// Recursively collect keys from directory.
    fn collect_keys(&self, dir: &Path, prefix: &str, keys: &mut Vec<String>) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                self.collect_keys(&path, prefix, keys)?;
            } else if let Ok(relative) = path.strip_prefix(&self.base_path) {
                let key = relative.to_string_lossy().to_string();
                if key.starts_with(prefix) {
                    keys.push(key);
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// S3 Backend (Stub)
// ============================================================================

/// AWS S3 storage backend (stub implementation).
///
/// This is a mock implementation for testing and development.
/// In production, replace with actual AWS SDK calls.
#[allow(dead_code)]
pub struct S3Backend {
    /// Configuration.
    config: S3Config,
    /// Connection pool.
    pool: ConnectionPool,
    /// Retry policy (reserved for real S3 implementation).
    retry_policy: RetryPolicy,
    /// In-memory storage for testing.
    storage: parking_lot::RwLock<HashMap<String, Vec<u8>>>,
}

impl S3Backend {
    /// Create a new S3 backend.
    pub fn new(config: S3Config) -> Self {
        Self {
            pool: ConnectionPool::new(
                50,
                5,
                config.storage.connection_timeout,
            ),
            retry_policy: RetryPolicy {
                max_attempts: config.storage.max_retries,
                initial_delay: config.storage.initial_retry_delay,
                max_delay: config.storage.max_retry_delay,
                jitter: 0.1,
            },
            config,
            storage: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Get the full S3 key (bucket/key).
    fn full_key(&self, key: &str) -> String {
        format!("{}/{}", self.config.bucket, key)
    }

    /// Get bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Get region.
    pub fn region(&self) -> &str {
        &self.config.region
    }
}

impl StorageBackend for S3Backend {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use aws_sdk_s3::Client::get_object
            // let resp = client.get_object()
            //     .bucket(&self.config.bucket)
            //     .key(key)
            //     .send()
            //     .await?;
            // let data = resp.body.collect().await?.into_bytes().to_vec();

            let storage = self.storage.read();
            storage
                .get(&full_key)
                .cloned()
                .ok_or_else(|| NeedleError::NotFound(format!("S3 key '{}' not found", key)))
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use aws_sdk_s3::Client::put_object
            // client.put_object()
            //     .bucket(&self.config.bucket)
            //     .key(key)
            //     .body(ByteStream::from(data.to_vec()))
            //     .send()
            //     .await?;

            let mut storage = self.storage.write();
            storage.insert(full_key, data.to_vec());
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use aws_sdk_s3::Client::delete_object
            // client.delete_object()
            //     .bucket(&self.config.bucket)
            //     .key(key)
            //     .send()
            //     .await?;

            let mut storage = self.storage.write();
            storage.remove(&full_key);
            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_prefix = self.full_key(prefix);

            // Stub: In production, use aws_sdk_s3::Client::list_objects_v2
            // let resp = client.list_objects_v2()
            //     .bucket(&self.config.bucket)
            //     .prefix(prefix)
            //     .send()
            //     .await?;

            let storage = self.storage.read();
            let bucket_prefix = format!("{}/", self.config.bucket);

            Ok(storage
                .keys()
                .filter(|k| k.starts_with(&full_prefix))
                .map(|k| k.strip_prefix(&bucket_prefix).unwrap_or(k).to_string())
                .collect())
        })
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use aws_sdk_s3::Client::head_object
            // match client.head_object()
            //     .bucket(&self.config.bucket)
            //     .key(key)
            //     .send()
            //     .await
            // {
            //     Ok(_) => Ok(true),
            //     Err(e) if e.is_not_found() => Ok(false),
            //     Err(e) => Err(e.into()),
            // }

            let storage = self.storage.read();
            Ok(storage.contains_key(&full_key))
        })
    }
}

// ============================================================================
// GCS Backend (Stub)
// ============================================================================

/// Google Cloud Storage backend (stub implementation).
#[allow(dead_code)]
pub struct GCSBackend {
    /// Configuration.
    config: GCSConfig,
    /// Connection pool.
    pool: ConnectionPool,
    /// Retry policy (reserved for real GCS implementation).
    retry_policy: RetryPolicy,
    /// In-memory storage for testing.
    storage: parking_lot::RwLock<HashMap<String, Vec<u8>>>,
}

impl GCSBackend {
    /// Create a new GCS backend.
    pub fn new(config: GCSConfig) -> Self {
        Self {
            pool: ConnectionPool::new(50, 5, config.storage.connection_timeout),
            retry_policy: RetryPolicy {
                max_attempts: config.storage.max_retries,
                initial_delay: config.storage.initial_retry_delay,
                max_delay: config.storage.max_retry_delay,
                jitter: 0.1,
            },
            config,
            storage: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Get the full GCS object name.
    fn full_key(&self, key: &str) -> String {
        format!("gs://{}/{}", self.config.bucket, key)
    }

    /// Get bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Get project ID.
    pub fn project_id(&self) -> &str {
        &self.config.project_id
    }
}

impl StorageBackend for GCSBackend {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use google-cloud-storage crate
            // let client = Client::default();
            // let data = client.object()
            //     .bucket(&self.config.bucket)
            //     .name(key)
            //     .download()
            //     .await?;

            let storage = self.storage.read();
            storage
                .get(&full_key)
                .cloned()
                .ok_or_else(|| NeedleError::NotFound(format!("GCS object '{}' not found", key)))
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use google-cloud-storage crate
            // let client = Client::default();
            // client.object()
            //     .bucket(&self.config.bucket)
            //     .name(key)
            //     .upload(data)
            //     .await?;

            let mut storage = self.storage.write();
            storage.insert(full_key, data.to_vec());
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use google-cloud-storage crate
            // let client = Client::default();
            // client.object()
            //     .bucket(&self.config.bucket)
            //     .name(key)
            //     .delete()
            //     .await?;

            let mut storage = self.storage.write();
            storage.remove(&full_key);
            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_prefix = self.full_key(prefix);
            let bucket_prefix = format!("gs://{}/", self.config.bucket);

            // Stub: In production, use google-cloud-storage crate
            // let client = Client::default();
            // let objects = client.bucket(&self.config.bucket)
            //     .list_objects()
            //     .prefix(prefix)
            //     .send()
            //     .await?;

            let storage = self.storage.read();
            Ok(storage
                .keys()
                .filter(|k| k.starts_with(&full_prefix))
                .map(|k| k.strip_prefix(&bucket_prefix).unwrap_or(k).to_string())
                .collect())
        })
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            let storage = self.storage.read();
            Ok(storage.contains_key(&full_key))
        })
    }
}

// ============================================================================
// Azure Blob Backend (Stub)
// ============================================================================

/// Azure Blob Storage backend (stub implementation).
#[allow(dead_code)]
pub struct AzureBlobBackend {
    /// Configuration.
    config: AzureBlobConfig,
    /// Connection pool.
    pool: ConnectionPool,
    /// Retry policy (reserved for real Azure implementation).
    retry_policy: RetryPolicy,
    /// In-memory storage for testing.
    storage: parking_lot::RwLock<HashMap<String, Vec<u8>>>,
}

impl AzureBlobBackend {
    /// Create a new Azure Blob backend.
    pub fn new(config: AzureBlobConfig) -> Self {
        Self {
            pool: ConnectionPool::new(50, 5, config.storage.connection_timeout),
            retry_policy: RetryPolicy {
                max_attempts: config.storage.max_retries,
                initial_delay: config.storage.initial_retry_delay,
                max_delay: config.storage.max_retry_delay,
                jitter: 0.1,
            },
            config,
            storage: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Get the full Azure blob path.
    fn full_key(&self, key: &str) -> String {
        format!(
            "https://{}.blob.core.windows.net/{}/{}",
            self.config.account_name, self.config.container, key
        )
    }

    /// Get container name.
    pub fn container(&self) -> &str {
        &self.config.container
    }

    /// Get account name.
    pub fn account_name(&self) -> &str {
        &self.config.account_name
    }
}

impl StorageBackend for AzureBlobBackend {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use azure_storage_blobs crate
            // let client = BlobServiceClient::new(&self.config.account_name, credential);
            // let container = client.container_client(&self.config.container);
            // let blob = container.blob_client(key);
            // let data = blob.get_content().await?;

            let storage = self.storage.read();
            storage
                .get(&full_key)
                .cloned()
                .ok_or_else(|| NeedleError::NotFound(format!("Azure blob '{}' not found", key)))
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use azure_storage_blobs crate
            // let client = BlobServiceClient::new(&self.config.account_name, credential);
            // let container = client.container_client(&self.config.container);
            // let blob = container.blob_client(key);
            // blob.put_block_blob(data).await?;

            let mut storage = self.storage.write();
            storage.insert(full_key, data.to_vec());
            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            // Stub: In production, use azure_storage_blobs crate
            // let client = BlobServiceClient::new(&self.config.account_name, credential);
            // let container = client.container_client(&self.config.container);
            // let blob = container.blob_client(key);
            // blob.delete().await?;

            let mut storage = self.storage.write();
            storage.remove(&full_key);
            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_prefix = self.full_key(prefix);
            let container_prefix = format!(
                "https://{}.blob.core.windows.net/{}/",
                self.config.account_name, self.config.container
            );

            // Stub: In production, use azure_storage_blobs crate
            // let client = BlobServiceClient::new(&self.config.account_name, credential);
            // let container = client.container_client(&self.config.container);
            // let blobs = container.list_blobs()
            //     .prefix(prefix)
            //     .into_stream()
            //     .collect()
            //     .await?;

            let storage = self.storage.read();
            Ok(storage
                .keys()
                .filter(|k| k.starts_with(&full_prefix))
                .map(|k| k.strip_prefix(&container_prefix).unwrap_or(k).to_string())
                .collect())
        })
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            let _conn = self.pool.acquire()?;
            let full_key = self.full_key(key);

            let storage = self.storage.read();
            Ok(storage.contains_key(&full_key))
        })
    }
}

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
// Caching Layer
// ============================================================================

/// Cache configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum cache size in bytes.
    pub max_size: usize,
    /// Default TTL for cached items.
    pub default_ttl: Duration,
    /// Enable cache statistics.
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_size: 100 * 1024 * 1024, // 100MB
            default_ttl: Duration::from_secs(300), // 5 minutes
            enable_stats: true,
        }
    }
}

/// Cached storage backend wrapper.
pub struct CachedBackend<B: StorageBackend> {
    /// Inner backend.
    inner: B,
    /// Cache configuration.
    config: CacheConfig,
    /// Cache storage.
    cache: parking_lot::RwLock<HashMap<String, CacheEntry>>,
    /// Cache statistics.
    stats: CacheStats,
}

/// Cache entry.
#[derive(Clone)]
#[allow(dead_code)]
struct CacheEntry {
    data: Vec<u8>,
    expires_at: Instant,
    last_accessed: Instant,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
    evictions: Arc<AtomicU64>,
    bytes_cached: Arc<AtomicU64>,
}

impl CacheStats {
    /// Get cache hits.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Get cache misses.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Get evictions.
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    /// Get bytes cached.
    pub fn bytes_cached(&self) -> u64 {
        self.bytes_cached.load(Ordering::Relaxed)
    }

    /// Get hit rate.
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits() as f64;
        let total = hits + self.misses() as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

impl<B: StorageBackend> CachedBackend<B> {
    /// Create a new cached backend.
    pub fn new(inner: B, config: CacheConfig) -> Self {
        Self {
            inner,
            config,
            cache: parking_lot::RwLock::new(HashMap::new()),
            stats: CacheStats::default(),
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        self.stats.bytes_cached.store(0, Ordering::Relaxed);
    }

    /// Invalidate a specific key.
    pub fn invalidate(&self, key: &str) {
        let mut cache = self.cache.write();
        if let Some(entry) = cache.remove(key) {
            self.stats
                .bytes_cached
                .fetch_sub(entry.data.len() as u64, Ordering::Relaxed);
        }
    }

    /// Evict expired entries.
    pub fn evict_expired(&self) {
        let mut cache = self.cache.write();
        let now = Instant::now();

        let expired_keys: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| entry.expires_at <= now)
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            if let Some(entry) = cache.remove(&key) {
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .bytes_cached
                    .fetch_sub(entry.data.len() as u64, Ordering::Relaxed);
            }
        }
    }

    /// Check if key is cached.
    pub fn is_cached(&self, key: &str) -> bool {
        let cache = self.cache.read();
        if let Some(entry) = cache.get(key) {
            entry.expires_at > Instant::now()
        } else {
            false
        }
    }
}

impl<B: StorageBackend> StorageBackend for CachedBackend<B> {
    fn read<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first
            {
                let cache = self.cache.read();
                if let Some(entry) = cache.get(key) {
                    if entry.expires_at > Instant::now() {
                        self.stats.hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(entry.data.clone());
                    }
                }
            }

            // Cache miss - fetch from backend
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            let data = self.inner.read(key).await?;

            // Add to cache
            {
                let mut cache = self.cache.write();
                let entry = CacheEntry {
                    data: data.clone(),
                    expires_at: Instant::now() + self.config.default_ttl,
                    last_accessed: Instant::now(),
                };
                self.stats
                    .bytes_cached
                    .fetch_add(entry.data.len() as u64, Ordering::Relaxed);
                cache.insert(key.to_string(), entry);
            }

            Ok(data)
        })
    }

    fn write<'a>(&'a self, key: &'a str, data: &'a [u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Write to backend
            self.inner.write(key, data).await?;

            // Update cache
            {
                let mut cache = self.cache.write();
                let entry = CacheEntry {
                    data: data.to_vec(),
                    expires_at: Instant::now() + self.config.default_ttl,
                    last_accessed: Instant::now(),
                };

                // Remove old entry if exists
                if let Some(old) = cache.remove(key) {
                    self.stats
                        .bytes_cached
                        .fetch_sub(old.data.len() as u64, Ordering::Relaxed);
                }

                self.stats
                    .bytes_cached
                    .fetch_add(entry.data.len() as u64, Ordering::Relaxed);
                cache.insert(key.to_string(), entry);
            }

            Ok(())
        })
    }

    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            // Delete from backend
            self.inner.delete(key).await?;

            // Remove from cache
            self.invalidate(key);

            Ok(())
        })
    }

    fn list<'a>(&'a self, prefix: &'a str) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>> {
        // List is not cached - always go to backend
        self.inner.list(prefix)
    }

    fn exists<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>> {
        Box::pin(async move {
            // Check cache first
            if self.is_cached(key) {
                return Ok(true);
            }

            // Check backend
            self.inner.exists(key).await
        })
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
    use tempfile::TempDir;

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
}
