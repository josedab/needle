//! Local file system storage backend.

use crate::error::{NeedleError, Result};
use super::config::{ConnectionPool, RetryPolicy, StorageBackend};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

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
