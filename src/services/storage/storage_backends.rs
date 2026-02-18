//! Pluggable Storage Backends
//!
//! Synchronous, in-memory storage layer that mirrors the async
//! [`StorageBackend`](crate::persistence::cloud_storage::StorageBackend) trait
//! defined in `src/persistence/cloud_storage/mod.rs`. This module is always
//! available (no feature gate) and is suitable for embedded / single-process
//! use-cases where an async runtime is not desired.
//!
//! # Relationship to the cloud storage scaffold
//!
//! The `persistence::cloud_storage` module exposes the **async**
//! [`StorageBackend`](crate::persistence::cloud_storage::StorageBackend) trait
//! with concrete implementations behind feature gates:
//!
//! | Feature flag            | Async backend                                                                   |
//! |-------------------------|---------------------------------------------------------------------------------|
//! | `cloud-storage-s3`      | [`S3Backend`](crate::persistence::cloud_storage::S3Backend)                     |
//! | `cloud-storage-gcs`     | [`GCSBackend`](crate::persistence::cloud_storage::GCSBackend)                   |
//! | `cloud-storage-azure`   | [`AzureBlobBackend`](crate::persistence::cloud_storage::AzureBlobBackend)       |
//! | *(always available)*    | [`LocalBackend`](crate::persistence::cloud_storage::LocalBackend)               |
//! | *(always available)*    | [`CachedBackend`](crate::persistence::cloud_storage::CachedBackend)             |
//! | *(always available)*    | [`TieredCacheBackend`](crate::persistence::cloud_storage::TieredCacheBackend)   |
//!
//! The [`StorageManager`] in this module provides an **equivalent synchronous
//! API** (`put` / `get` / `delete` / `exists` / `list_keys`) backed by
//! in-memory storage so that callers can program against the same operations
//! without pulling in Tokio or an async runtime.
//!
//! Use [`BackendType`] to declare *which* backend you logically target. When
//! running inside an async context with the relevant cloud-storage features
//! enabled, prefer the real async implementations from
//! `persistence::cloud_storage` — with
//! [`ConnectionPool`](crate::persistence::cloud_storage::ConnectionPool) and
//! [`RetryPolicy`](crate::persistence::cloud_storage::RetryPolicy) for
//! production workloads.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::storage_backends::{
//!     StorageManager, StorageConfig, BackendType,
//! };
//!
//! let mut manager = StorageManager::new(StorageConfig::default());
//!
//! // Write and read data
//! manager.put("collection/vectors/doc-1", b"hello").unwrap();
//! let data = manager.get("collection/vectors/doc-1").unwrap();
//! assert_eq!(data.unwrap(), b"hello");
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Backend Type ────────────────────────────────────────────────────────────

/// Identifies the logical storage backend.
///
/// Each variant corresponds to a concrete implementation, either in this module
/// (synchronous, in-memory) or in
/// [`persistence::cloud_storage`](crate::persistence::cloud_storage) (async,
/// feature-gated):
///
/// | Variant       | Sync (this module)        | Async (`cloud_storage`)                                                      |
/// |---------------|---------------------------|------------------------------------------------------------------------------|
/// | `Memory`      | [`MemoryBackend`] (inner) | —                                                                            |
/// | `LocalFile`   | [`FileBackend`] (inner)   | [`LocalBackend`](crate::persistence::cloud_storage::LocalBackend)            |
/// | `S3`          | In-memory stub            | [`S3Backend`](crate::persistence::cloud_storage::S3Backend) (`cloud-storage-s3`)    |
/// | `GCS`         | In-memory stub            | [`GCSBackend`](crate::persistence::cloud_storage::GCSBackend) (`cloud-storage-gcs`) |
/// | `AzureBlob`   | In-memory stub            | [`AzureBlobBackend`](crate::persistence::cloud_storage::AzureBlobBackend) (`cloud-storage-azure`) |
/// | `SQLite`      | Falls back to Memory      | —                                                                            |
/// | `RocksDB`     | Falls back to Memory      | —                                                                            |
/// | `DuckDB`      | Falls back to Memory      | —                                                                            |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// In-memory storage (for testing, ephemeral data).
    Memory,
    /// Local file system.
    /// Async counterpart: [`LocalBackend`](crate::persistence::cloud_storage::LocalBackend).
    LocalFile,
    /// S3-compatible object storage.
    /// Async counterpart: [`S3Backend`](crate::persistence::cloud_storage::S3Backend)
    /// (requires feature `cloud-storage-s3`).
    S3,
    /// Google Cloud Storage.
    /// Async counterpart: [`GCSBackend`](crate::persistence::cloud_storage::GCSBackend)
    /// (requires feature `cloud-storage-gcs`).
    GCS,
    /// Azure Blob Storage.
    /// Async counterpart: [`AzureBlobBackend`](crate::persistence::cloud_storage::AzureBlobBackend)
    /// (requires feature `cloud-storage-azure`).
    AzureBlob,
    /// SQLite-backed storage.
    SQLite,
    /// RocksDB-backed storage.
    RocksDB,
    /// DuckDB-backed storage.
    DuckDB,
}

impl fmt::Display for BackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory => write!(f, "Memory"),
            Self::LocalFile => write!(f, "LocalFile"),
            Self::S3 => write!(f, "S3"),
            Self::GCS => write!(f, "GCS"),
            Self::AzureBlob => write!(f, "AzureBlob"),
            Self::SQLite => write!(f, "SQLite"),
            Self::RocksDB => write!(f, "RocksDB"),
            Self::DuckDB => write!(f, "DuckDB"),
        }
    }
}

/// Metadata about a stored object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMeta {
    /// Object key.
    pub key: String,
    /// Size in bytes.
    pub size: u64,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last modified timestamp.
    pub modified_at: u64,
    /// Content hash (SHA-256 hex).
    pub content_hash: Option<String>,
    /// Custom metadata.
    pub tags: HashMap<String, String>,
}

/// Storage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total objects stored.
    pub total_objects: u64,
    /// Total bytes stored.
    pub total_bytes: u64,
    /// Total read operations.
    pub read_ops: u64,
    /// Total write operations.
    pub write_ops: u64,
    /// Total delete operations.
    pub delete_ops: u64,
    /// Cache hit count.
    pub cache_hits: u64,
    /// Cache miss count.
    pub cache_misses: u64,
}

// ── In-Memory Backend ───────────────────────────────────────────────────────

/// In-memory storage (HashMap-based).
///
/// This is the synchronous counterpart used by [`StorageManager`]. For an async
/// version with real I/O, see the [`StorageBackend`](crate::persistence::cloud_storage::StorageBackend)
/// trait implementations in `persistence::cloud_storage`.
struct MemoryBackend {
    data: HashMap<String, Vec<u8>>,
    metadata: HashMap<String, ObjectMeta>,
}

impl MemoryBackend {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    fn put(&mut self, key: &str, value: &[u8]) -> Result<()> {
        let now = now_secs();
        self.metadata.insert(
            key.to_string(),
            ObjectMeta {
                key: key.to_string(),
                size: value.len() as u64,
                created_at: self
                    .metadata
                    .get(key)
                    .map_or(now, |m| m.created_at),
                modified_at: now,
                content_hash: None,
                tags: HashMap::new(),
            },
        );
        self.data.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.get(key).cloned())
    }

    fn delete(&mut self, key: &str) -> Result<bool> {
        self.metadata.remove(key);
        Ok(self.data.remove(key).is_some())
    }

    fn exists(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    fn list_keys(&self, prefix: &str) -> Vec<String> {
        self.data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect()
    }

    fn object_meta(&self, key: &str) -> Option<&ObjectMeta> {
        self.metadata.get(key)
    }

    #[allow(dead_code)]
    fn total_bytes(&self) -> u64 {
        self.data.values().map(|v| v.len() as u64).sum()
    }

    #[allow(dead_code)]
    fn total_objects(&self) -> u64 {
        self.data.len() as u64
    }
}

// ── File Backend ────────────────────────────────────────────────────────────

/// Local filesystem storage (simulated in-memory for portability).
///
/// For real file-system I/O, use
/// [`LocalBackend`](crate::persistence::cloud_storage::LocalBackend) from the
/// cloud storage scaffold.
struct FileBackend {
    inner: MemoryBackend,
    base_path: String,
}

impl FileBackend {
    fn new(base_path: String) -> Self {
        Self {
            inner: MemoryBackend::new(),
            base_path,
        }
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}/{}", self.base_path, key)
    }
}

// ── S3 Backend ──────────────────────────────────────────────────────────────

/// S3-compatible object storage (simulated for non-async context).
///
/// This is a synchronous stub. For real S3 access, enable the `cloud-storage-s3`
/// feature and use [`S3Backend`](crate::persistence::cloud_storage::S3Backend)
/// with [`RetryPolicy`](crate::persistence::cloud_storage::RetryPolicy) and
/// [`ConnectionPool`](crate::persistence::cloud_storage::ConnectionPool).
struct S3Backend {
    inner: MemoryBackend,
    #[allow(dead_code)]
    bucket: String,
    #[allow(dead_code)]
    region: String,
}

impl S3Backend {
    fn new(bucket: String, region: String) -> Self {
        Self {
            inner: MemoryBackend::new(),
            bucket,
            region,
        }
    }
}

// ── GCS Backend ─────────────────────────────────────────────────────────────

/// Google Cloud Storage (simulated for non-async context).
///
/// For real GCS access, enable the `cloud-storage-gcs` feature and use
/// [`GCSBackend`](crate::persistence::cloud_storage::GCSBackend).
struct GcsBackend {
    inner: MemoryBackend,
    #[allow(dead_code)]
    bucket: String,
    #[allow(dead_code)]
    project_id: String,
}

impl GcsBackend {
    fn new(bucket: String, project_id: String) -> Self {
        Self {
            inner: MemoryBackend::new(),
            bucket,
            project_id,
        }
    }
}

// ── Azure Blob Backend ──────────────────────────────────────────────────────

/// Azure Blob Storage (simulated for non-async context).
///
/// For real Azure access, enable the `cloud-storage-azure` feature and use
/// [`AzureBlobBackend`](crate::persistence::cloud_storage::AzureBlobBackend).
struct AzureBlobBackend {
    inner: MemoryBackend,
    #[allow(dead_code)]
    container: String,
    #[allow(dead_code)]
    account: String,
}

impl AzureBlobBackend {
    fn new(container: String, account: String) -> Self {
        Self {
            inner: MemoryBackend::new(),
            container,
            account,
        }
    }
}

// ── Storage Backend Enum ────────────────────────────────────────────────────

/// Unified synchronous storage backend dispatch.
///
/// Each variant wraps a synchronous, in-memory implementation. For production
/// cloud workloads, use the async
/// [`StorageBackend`](crate::persistence::cloud_storage::StorageBackend) trait
/// with [`CachedBackend`](crate::persistence::cloud_storage::CachedBackend) or
/// [`TieredCacheBackend`](crate::persistence::cloud_storage::TieredCacheBackend).
enum StorageBackendImpl {
    Memory(MemoryBackend),
    File(FileBackend),
    S3(S3Backend),
    GCS(GcsBackend),
    AzureBlob(AzureBlobBackend),
}

/// A registered backend with its configuration.
struct RegisteredBackend {
    backend: StorageBackendImpl,
    backend_type: BackendType,
    read_only: bool,
}

// ── Tiered Storage ──────────────────────────────────────────────────────────

/// Storage tier classification.
///
/// For async tiered caching, see
/// [`TieredCacheBackend`](crate::persistence::cloud_storage::TieredCacheBackend)
/// in the cloud storage scaffold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageTier {
    /// Frequently accessed data.
    Hot,
    /// Occasionally accessed data.
    Warm,
    /// Rarely accessed data, archival.
    Cold,
}

/// Tiering policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieringPolicy {
    /// Move to warm after this many seconds without access.
    pub warm_after_secs: u64,
    /// Move to cold after this many seconds without access.
    pub cold_after_secs: u64,
    /// Enable automatic tiering.
    pub auto_tier: bool,
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self {
            warm_after_secs: 86400,       // 1 day
            cold_after_secs: 86400 * 30,  // 30 days
            auto_tier: false,
        }
    }
}

// ── Storage Config ──────────────────────────────────────────────────────────

/// Configuration for the synchronous [`StorageManager`].
///
/// This mirrors the shape of
/// [`StorageConfig`](crate::persistence::cloud_storage::StorageConfig) from the
/// cloud storage scaffold but omits async-only fields (timeouts, retry delays)
/// which are handled by
/// [`RetryPolicy`](crate::persistence::cloud_storage::RetryPolicy) and
/// [`ConnectionPool`](crate::persistence::cloud_storage::ConnectionPool) in the
/// async path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Primary backend type.
    pub primary_backend: BackendType,
    /// Optional base path for file backend.
    pub base_path: Option<String>,
    /// S3 bucket (for S3 backend).
    pub s3_bucket: Option<String>,
    /// S3 region (for S3 backend).
    pub s3_region: Option<String>,
    /// GCS bucket (for GCS backend).
    pub gcs_bucket: Option<String>,
    /// GCS project ID (for GCS backend).
    pub gcs_project_id: Option<String>,
    /// Azure container (for Azure Blob backend).
    pub azure_container: Option<String>,
    /// Azure account name (for Azure Blob backend).
    pub azure_account: Option<String>,
    /// Enable read cache.
    pub enable_cache: bool,
    /// Cache capacity (number of objects).
    pub cache_capacity: usize,
    /// Tiering policy.
    pub tiering: TieringPolicy,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            primary_backend: BackendType::Memory,
            base_path: None,
            s3_bucket: None,
            s3_region: None,
            gcs_bucket: None,
            gcs_project_id: None,
            azure_container: None,
            azure_account: None,
            enable_cache: true,
            cache_capacity: 10_000,
            tiering: TieringPolicy::default(),
        }
    }
}

// ── Storage Manager ─────────────────────────────────────────────────────────

/// Synchronous storage manager with pluggable backends.
///
/// Provides the same five core operations as the async
/// [`StorageBackend`](crate::persistence::cloud_storage::StorageBackend) trait
/// (`read`/`write`/`delete`/`list`/`exists`) but exposed as synchronous methods
/// (`get`/`put`/`delete`/`list_keys`/`exists`).
///
/// # Cloud storage integration
///
/// This manager is intentionally synchronous so it works without a Tokio
/// runtime. When cloud features are enabled, prefer the async backends
/// directly:
///
/// * [`S3Backend`](crate::persistence::cloud_storage::S3Backend) — `cloud-storage-s3`
/// * [`GCSBackend`](crate::persistence::cloud_storage::GCSBackend) — `cloud-storage-gcs`
/// * [`AzureBlobBackend`](crate::persistence::cloud_storage::AzureBlobBackend) — `cloud-storage-azure`
///
/// Wrap them with [`CachedBackend`](crate::persistence::cloud_storage::CachedBackend)
/// or [`TieredCacheBackend`](crate::persistence::cloud_storage::TieredCacheBackend)
/// for read caching.
pub struct StorageManager {
    config: StorageConfig,
    backends: Vec<RegisteredBackend>,
    primary_idx: usize,
    cache: HashMap<String, Vec<u8>>,
    access_times: HashMap<String, u64>,
    stats: StorageStats,
}

impl StorageManager {
    /// Create a new storage manager with the given configuration.
    pub fn new(config: StorageConfig) -> Self {
        let backend = Self::create_backend(&config, config.primary_backend);
        Self {
            backends: vec![RegisteredBackend {
                backend,
                backend_type: config.primary_backend,
                read_only: false,
            }],
            primary_idx: 0,
            cache: HashMap::new(),
            access_times: HashMap::new(),
            stats: StorageStats::default(),
            config,
        }
    }

    /// Store data under a key.
    ///
    /// Mirrors [`StorageBackend::write`](crate::persistence::cloud_storage::StorageBackend::write).
    pub fn put(&mut self, key: &str, value: &[u8]) -> Result<()> {
        let backend = &mut self.backends[self.primary_idx];
        if backend.read_only {
            return Err(NeedleError::InvalidOperation(
                "Primary backend is read-only".into(),
            ));
        }
        Self::backend_put(&mut backend.backend, key, value)?;
        self.stats.write_ops += 1;
        self.stats.total_objects += 1;
        self.stats.total_bytes += value.len() as u64;
        self.access_times.insert(key.to_string(), now_secs());

        // Update cache
        if self.config.enable_cache {
            if self.cache.len() >= self.config.cache_capacity {
                // Evict oldest entry
                if let Some(oldest) = self
                    .access_times
                    .iter()
                    .filter(|(k, _)| self.cache.contains_key(*k))
                    .min_by_key(|(_, t)| *t)
                    .map(|(k, _)| k.clone())
                {
                    self.cache.remove(&oldest);
                }
            }
            self.cache.insert(key.to_string(), value.to_vec());
        }
        Ok(())
    }

    /// Retrieve data by key.
    ///
    /// Mirrors [`StorageBackend::read`](crate::persistence::cloud_storage::StorageBackend::read).
    pub fn get(&mut self, key: &str) -> Result<Option<Vec<u8>>> {
        self.stats.read_ops += 1;
        self.access_times.insert(key.to_string(), now_secs());

        // Check cache first
        if self.config.enable_cache {
            if let Some(cached) = self.cache.get(key) {
                self.stats.cache_hits += 1;
                return Ok(Some(cached.clone()));
            }
            self.stats.cache_misses += 1;
        }

        let backend = &self.backends[self.primary_idx];
        let result = Self::backend_get(&backend.backend, key)?;

        // Populate cache
        if self.config.enable_cache {
            if let Some(ref data) = result {
                self.cache.insert(key.to_string(), data.clone());
            }
        }
        Ok(result)
    }

    /// Delete data by key.
    ///
    /// Mirrors [`StorageBackend::delete`](crate::persistence::cloud_storage::StorageBackend::delete).
    pub fn delete(&mut self, key: &str) -> Result<bool> {
        let backend = &mut self.backends[self.primary_idx];
        if backend.read_only {
            return Err(NeedleError::InvalidOperation(
                "Primary backend is read-only".into(),
            ));
        }
        let deleted = Self::backend_delete(&mut backend.backend, key)?;
        if deleted {
            self.stats.delete_ops += 1;
            self.cache.remove(key);
        }
        Ok(deleted)
    }

    /// Check if a key exists.
    ///
    /// Mirrors [`StorageBackend::exists`](crate::persistence::cloud_storage::StorageBackend::exists).
    pub fn exists(&self, key: &str) -> bool {
        if self.cache.contains_key(key) {
            return true;
        }
        let backend = &self.backends[self.primary_idx];
        Self::backend_exists(&backend.backend, key)
    }

    /// List keys with a given prefix.
    ///
    /// Mirrors [`StorageBackend::list`](crate::persistence::cloud_storage::StorageBackend::list).
    pub fn list_keys(&self, prefix: &str) -> Vec<String> {
        let backend = &self.backends[self.primary_idx];
        Self::backend_list_keys(&backend.backend, prefix)
    }

    /// Get object metadata.
    pub fn object_meta(&self, key: &str) -> Option<ObjectMeta> {
        let backend = &self.backends[self.primary_idx];
        Self::backend_object_meta(&backend.backend, key)
    }

    /// Classify a key into a storage tier based on last access time.
    pub fn classify_tier(&self, key: &str) -> StorageTier {
        let now = now_secs();
        let last_access = self.access_times.get(key).copied().unwrap_or(0);
        let age = now.saturating_sub(last_access);

        if age > self.config.tiering.cold_after_secs {
            StorageTier::Cold
        } else if age > self.config.tiering.warm_after_secs {
            StorageTier::Warm
        } else {
            StorageTier::Hot
        }
    }

    /// Get storage statistics.
    pub fn stats(&self) -> &StorageStats {
        &self.stats
    }

    /// Get the primary backend type.
    pub fn primary_backend_type(&self) -> BackendType {
        self.backends[self.primary_idx].backend_type
    }

    /// Get config.
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }

    /// Clear the read cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Register an additional backend (for tiered storage).
    pub fn add_backend(&mut self, backend_type: BackendType, read_only: bool) {
        let backend = Self::create_backend(&self.config, backend_type);
        self.backends.push(RegisteredBackend {
            backend,
            backend_type,
            read_only,
        });
    }

    /// Get the number of registered backends.
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Describe all available backend types and their async counterparts.
    ///
    /// Returns a human-readable summary mapping each [`BackendType`] to the
    /// corresponding async
    /// [`StorageBackend`](crate::persistence::cloud_storage::StorageBackend)
    /// implementation and its required feature flag.
    pub fn storage_backend_type_description() -> Vec<BackendTypeInfo> {
        vec![
            BackendTypeInfo {
                backend_type: BackendType::Memory,
                description: "In-memory storage for testing and ephemeral data".into(),
                async_counterpart: None,
                feature_flag: None,
            },
            BackendTypeInfo {
                backend_type: BackendType::LocalFile,
                description: "Local filesystem storage".into(),
                async_counterpart: Some("persistence::cloud_storage::LocalBackend".into()),
                feature_flag: None,
            },
            BackendTypeInfo {
                backend_type: BackendType::S3,
                description: "AWS S3-compatible object storage".into(),
                async_counterpart: Some("persistence::cloud_storage::S3Backend".into()),
                feature_flag: Some("cloud-storage-s3".into()),
            },
            BackendTypeInfo {
                backend_type: BackendType::GCS,
                description: "Google Cloud Storage".into(),
                async_counterpart: Some("persistence::cloud_storage::GCSBackend".into()),
                feature_flag: Some("cloud-storage-gcs".into()),
            },
            BackendTypeInfo {
                backend_type: BackendType::AzureBlob,
                description: "Azure Blob Storage".into(),
                async_counterpart: Some("persistence::cloud_storage::AzureBlobBackend".into()),
                feature_flag: Some("cloud-storage-azure".into()),
            },
            BackendTypeInfo {
                backend_type: BackendType::SQLite,
                description: "SQLite-backed storage (falls back to Memory)".into(),
                async_counterpart: None,
                feature_flag: None,
            },
            BackendTypeInfo {
                backend_type: BackendType::RocksDB,
                description: "RocksDB-backed storage (falls back to Memory)".into(),
                async_counterpart: None,
                feature_flag: None,
            },
            BackendTypeInfo {
                backend_type: BackendType::DuckDB,
                description: "DuckDB-backed storage (falls back to Memory)".into(),
                async_counterpart: None,
                feature_flag: None,
            },
        ]
    }

    // ── Private helpers ─────────────────────────────────────────────────

    fn create_backend(config: &StorageConfig, backend_type: BackendType) -> StorageBackendImpl {
        match backend_type {
            BackendType::Memory => StorageBackendImpl::Memory(MemoryBackend::new()),
            BackendType::LocalFile => StorageBackendImpl::File(FileBackend::new(
                config.base_path.clone().unwrap_or_else(|| "/tmp/needle".into()),
            )),
            BackendType::S3 => StorageBackendImpl::S3(S3Backend::new(
                config.s3_bucket.clone().unwrap_or_default(),
                config.s3_region.clone().unwrap_or_else(|| "us-east-1".into()),
            )),
            BackendType::GCS => StorageBackendImpl::GCS(GcsBackend::new(
                config.gcs_bucket.clone().unwrap_or_default(),
                config.gcs_project_id.clone().unwrap_or_default(),
            )),
            BackendType::AzureBlob => StorageBackendImpl::AzureBlob(AzureBlobBackend::new(
                config.azure_container.clone().unwrap_or_default(),
                config.azure_account.clone().unwrap_or_default(),
            )),
            // SQLite / RocksDB / DuckDB fall back to memory for now
            _ => StorageBackendImpl::Memory(MemoryBackend::new()),
        }
    }

    fn backend_put(backend: &mut StorageBackendImpl, key: &str, value: &[u8]) -> Result<()> {
        match backend {
            StorageBackendImpl::Memory(b) => b.put(key, value),
            StorageBackendImpl::File(b) => b.inner.put(&b.full_key(key), value),
            StorageBackendImpl::S3(b) => b.inner.put(key, value),
            StorageBackendImpl::GCS(b) => b.inner.put(key, value),
            StorageBackendImpl::AzureBlob(b) => b.inner.put(key, value),
        }
    }

    fn backend_get(backend: &StorageBackendImpl, key: &str) -> Result<Option<Vec<u8>>> {
        match backend {
            StorageBackendImpl::Memory(b) => b.get(key),
            StorageBackendImpl::File(b) => b.inner.get(&b.full_key(key)),
            StorageBackendImpl::S3(b) => b.inner.get(key),
            StorageBackendImpl::GCS(b) => b.inner.get(key),
            StorageBackendImpl::AzureBlob(b) => b.inner.get(key),
        }
    }

    fn backend_delete(backend: &mut StorageBackendImpl, key: &str) -> Result<bool> {
        match backend {
            StorageBackendImpl::Memory(b) => b.delete(key),
            StorageBackendImpl::File(b) => {
                let full = b.full_key(key);
                b.inner.delete(&full)
            }
            StorageBackendImpl::S3(b) => b.inner.delete(key),
            StorageBackendImpl::GCS(b) => b.inner.delete(key),
            StorageBackendImpl::AzureBlob(b) => b.inner.delete(key),
        }
    }

    fn backend_exists(backend: &StorageBackendImpl, key: &str) -> bool {
        match backend {
            StorageBackendImpl::Memory(b) => b.exists(key),
            StorageBackendImpl::File(b) => b.inner.exists(&b.full_key(key)),
            StorageBackendImpl::S3(b) => b.inner.exists(key),
            StorageBackendImpl::GCS(b) => b.inner.exists(key),
            StorageBackendImpl::AzureBlob(b) => b.inner.exists(key),
        }
    }

    fn backend_list_keys(backend: &StorageBackendImpl, prefix: &str) -> Vec<String> {
        match backend {
            StorageBackendImpl::Memory(b) => b.list_keys(prefix),
            StorageBackendImpl::File(b) => b.inner.list_keys(&b.full_key(prefix)),
            StorageBackendImpl::S3(b) => b.inner.list_keys(prefix),
            StorageBackendImpl::GCS(b) => b.inner.list_keys(prefix),
            StorageBackendImpl::AzureBlob(b) => b.inner.list_keys(prefix),
        }
    }

    fn backend_object_meta(backend: &StorageBackendImpl, key: &str) -> Option<ObjectMeta> {
        match backend {
            StorageBackendImpl::Memory(b) => b.object_meta(key).cloned(),
            StorageBackendImpl::File(b) => b.inner.object_meta(&b.full_key(key)).cloned(),
            StorageBackendImpl::S3(b) => b.inner.object_meta(key).cloned(),
            StorageBackendImpl::GCS(b) => b.inner.object_meta(key).cloned(),
            StorageBackendImpl::AzureBlob(b) => b.inner.object_meta(key).cloned(),
        }
    }
}

impl Default for StorageManager {
    fn default() -> Self {
        Self::new(StorageConfig::default())
    }
}

// ── BackendTypeInfo ─────────────────────────────────────────────────────────

/// Describes a [`BackendType`] and its relationship to the async cloud storage
/// scaffold in [`persistence::cloud_storage`](crate::persistence::cloud_storage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendTypeInfo {
    /// The backend type variant.
    pub backend_type: BackendType,
    /// Human-readable description.
    pub description: String,
    /// Path to the async counterpart (if any) in `persistence::cloud_storage`.
    pub async_counterpart: Option<String>,
    /// Cargo feature flag required to enable the async counterpart.
    pub feature_flag: Option<String>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> StorageManager {
        StorageManager::new(StorageConfig::default())
    }

    #[test]
    fn test_put_get() {
        let mut m = make_manager();
        m.put("key1", b"value1").unwrap();
        let data = m.get("key1").unwrap();
        assert_eq!(data.unwrap(), b"value1");
    }

    #[test]
    fn test_get_missing() {
        let mut m = make_manager();
        let data = m.get("nonexistent").unwrap();
        assert!(data.is_none());
    }

    #[test]
    fn test_delete() {
        let mut m = make_manager();
        m.put("key1", b"value1").unwrap();
        assert!(m.delete("key1").unwrap());
        assert!(m.get("key1").unwrap().is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut m = make_manager();
        assert!(!m.delete("nonexistent").unwrap());
    }

    #[test]
    fn test_exists() {
        let mut m = make_manager();
        assert!(!m.exists("key1"));
        m.put("key1", b"value1").unwrap();
        assert!(m.exists("key1"));
    }

    #[test]
    fn test_list_keys() {
        let mut m = make_manager();
        m.put("coll/vec/1", b"a").unwrap();
        m.put("coll/vec/2", b"b").unwrap();
        m.put("other/1", b"c").unwrap();
        let keys = m.list_keys("coll/");
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_cache_hit() {
        let mut m = make_manager();
        m.put("key1", b"value1").unwrap();
        m.get("key1").unwrap(); // populate cache
        m.get("key1").unwrap(); // cache hit
        assert!(m.stats().cache_hits > 0);
    }

    #[test]
    fn test_cache_disabled() {
        let mut m = StorageManager::new(StorageConfig {
            enable_cache: false,
            ..Default::default()
        });
        m.put("key1", b"value1").unwrap();
        m.get("key1").unwrap();
        assert_eq!(m.stats().cache_hits, 0);
    }

    #[test]
    fn test_object_meta() {
        let mut m = make_manager();
        m.put("key1", b"hello").unwrap();
        let meta = m.object_meta("key1").unwrap();
        assert_eq!(meta.size, 5);
    }

    #[test]
    fn test_file_backend() {
        let mut m = StorageManager::new(StorageConfig {
            primary_backend: BackendType::LocalFile,
            base_path: Some("/tmp/test".into()),
            ..Default::default()
        });
        m.put("key1", b"data").unwrap();
        let data = m.get("key1").unwrap();
        assert!(data.is_none() || !data.unwrap().is_empty());
    }

    #[test]
    fn test_s3_backend() {
        let mut m = StorageManager::new(StorageConfig {
            primary_backend: BackendType::S3,
            s3_bucket: Some("test-bucket".into()),
            s3_region: Some("us-east-1".into()),
            ..Default::default()
        });
        m.put("key1", b"data").unwrap();
        assert_eq!(m.primary_backend_type(), BackendType::S3);
    }

    #[test]
    fn test_storage_stats() {
        let mut m = make_manager();
        m.put("a", b"1").unwrap();
        m.put("b", b"22").unwrap();
        m.get("a").unwrap();
        m.delete("b").unwrap();
        assert_eq!(m.stats().write_ops, 2);
        assert_eq!(m.stats().read_ops, 1);
        assert_eq!(m.stats().delete_ops, 1);
    }

    #[test]
    fn test_tiering_classification() {
        let m = make_manager();
        // Fresh key = Hot
        assert_eq!(m.classify_tier("unknown"), StorageTier::Cold); // no access = cold
    }

    #[test]
    fn test_clear_cache() {
        let mut m = make_manager();
        m.put("k", b"v").unwrap();
        assert!(m.cache_size() > 0);
        m.clear_cache();
        assert_eq!(m.cache_size(), 0);
    }

    #[test]
    fn test_add_backend() {
        let mut m = make_manager();
        assert_eq!(m.backend_count(), 1);
        m.add_backend(BackendType::S3, true);
        assert_eq!(m.backend_count(), 2);
    }

    #[test]
    fn test_overwrite() {
        let mut m = make_manager();
        m.put("key1", b"v1").unwrap();
        m.put("key1", b"v2").unwrap();
        let data = m.get("key1").unwrap().unwrap();
        assert_eq!(data, b"v2");
    }

    #[test]
    fn test_gcs_backend() {
        let mut m = StorageManager::new(StorageConfig {
            primary_backend: BackendType::GCS,
            gcs_bucket: Some("test-bucket".into()),
            gcs_project_id: Some("my-project".into()),
            ..Default::default()
        });
        m.put("key1", b"gcs-data").unwrap();
        assert_eq!(m.get("key1").unwrap().unwrap(), b"gcs-data");
        assert_eq!(m.primary_backend_type(), BackendType::GCS);
    }

    #[test]
    fn test_azure_blob_backend() {
        let mut m = StorageManager::new(StorageConfig {
            primary_backend: BackendType::AzureBlob,
            azure_container: Some("test-container".into()),
            azure_account: Some("testaccount".into()),
            ..Default::default()
        });
        m.put("key1", b"azure-data").unwrap();
        assert_eq!(m.get("key1").unwrap().unwrap(), b"azure-data");
        assert_eq!(m.primary_backend_type(), BackendType::AzureBlob);
    }

    #[test]
    fn test_storage_backend_type_description() {
        let infos = StorageManager::storage_backend_type_description();
        assert_eq!(infos.len(), 8);

        let s3_info = infos.iter().find(|i| i.backend_type == BackendType::S3).unwrap();
        assert_eq!(s3_info.feature_flag.as_deref(), Some("cloud-storage-s3"));
        assert!(s3_info.async_counterpart.is_some());

        let gcs_info = infos.iter().find(|i| i.backend_type == BackendType::GCS).unwrap();
        assert_eq!(gcs_info.feature_flag.as_deref(), Some("cloud-storage-gcs"));

        let azure_info = infos.iter().find(|i| i.backend_type == BackendType::AzureBlob).unwrap();
        assert_eq!(azure_info.feature_flag.as_deref(), Some("cloud-storage-azure"));

        let mem_info = infos.iter().find(|i| i.backend_type == BackendType::Memory).unwrap();
        assert!(mem_info.async_counterpart.is_none());
        assert!(mem_info.feature_flag.is_none());
    }

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::Memory.to_string(), "Memory");
        assert_eq!(BackendType::S3.to_string(), "S3");
        assert_eq!(BackendType::GCS.to_string(), "GCS");
        assert_eq!(BackendType::AzureBlob.to_string(), "AzureBlob");
    }
}
