//! Core traits, configuration types, connection pooling, and retry logic.

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

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
    fn read<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>>;

    /// Write data to storage.
    ///
    /// Overwrites existing data if the key already exists.
    fn write<'a>(
        &'a self,
        key: &'a str,
        data: &'a [u8],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// Delete data from storage.
    ///
    /// Returns Ok(()) even if the key does not exist (idempotent).
    fn delete<'a>(&'a self, key: &'a str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

    /// List keys with a given prefix.
    ///
    /// Returns all keys that start with the given prefix.
    fn list<'a>(
        &'a self,
        prefix: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + 'a>>;

    /// Check if a key exists.
    fn exists<'a>(
        &'a self,
        key: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;
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
            max_size: 100 * 1024 * 1024,           // 100MB
            default_ttl: Duration::from_secs(300), // 5 minutes
            enable_stats: true,
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

pub(super) struct PoolStatsInner {
    connections_created: AtomicU64,
    active_connections: AtomicU64,
    idle_connections: AtomicU64,
    requests_served: AtomicU64,
    connection_failures: AtomicU64,
}

impl ConnectionPool {
    /// Create a connection pool with default sizing from a storage config.
    pub fn from_storage_config(config: &StorageConfig) -> Self {
        Self::new(50, 5, config.connection_timeout)
    }

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
        self.stats
            .active_connections
            .fetch_add(1, Ordering::Relaxed);
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
    /// Create a retry policy from a storage config.
    pub fn from_storage_config(config: &StorageConfig) -> Self {
        Self {
            max_attempts: config.max_retries,
            initial_delay: config.initial_retry_delay,
            max_delay: config.max_retry_delay,
            jitter: 0.1,
        }
    }

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
            NeedleError::Io(std::io::Error::other("Retry exhausted with no error"))
        }))
    }

    /// Check if an error is retryable.
    fn is_retryable(error: &NeedleError) -> bool {
        matches!(error, NeedleError::Io(_) | NeedleError::BackupError(_))
    }
}

/// Generate random jitter value.
pub(super) fn rand_jitter(max: f64) -> f64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    std::time::Instant::now().hash(&mut hasher);
    let hash = hasher.finish();
    (hash as f64 / u64::MAX as f64) * max
}
