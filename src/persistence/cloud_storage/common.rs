//! Common utilities shared across cloud storage backends.
//!
//! All three cloud backends (S3, GCS, Azure) use identical in-memory mock
//! storage for testing/fallback. This module extracts that shared behavior.

use crate::error::{NeedleError, Result};
use std::collections::HashMap;

/// In-memory mock storage used as a fallback when cloud SDKs are not available.
pub(super) struct MockStorage {
    /// Label for error messages (e.g., "S3 key", "GCS object", "Azure blob").
    label: String,
    /// Base prefix for internal key mapping (e.g., "bucket/", "gs://bucket/").
    base_prefix: String,
    /// In-memory key-value storage.
    storage: parking_lot::RwLock<HashMap<String, Vec<u8>>>,
}

impl MockStorage {
    /// Create a new mock storage.
    ///
    /// - `label`: used in NotFound error messages (e.g., "S3 key")
    /// - `base_prefix`: prepended to keys for internal storage (e.g., "my-bucket/")
    pub fn new(label: impl Into<String>, base_prefix: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            base_prefix: base_prefix.into(),
            storage: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}{}", self.base_prefix, key)
    }

    pub fn read(&self, key: &str) -> Result<Vec<u8>> {
        let full_key = self.full_key(key);
        let storage = self.storage.read();
        storage
            .get(&full_key)
            .cloned()
            .ok_or_else(|| NeedleError::NotFound(format!("{} '{}' not found", self.label, key)))
    }

    pub fn write(&self, key: &str, data: &[u8]) {
        let full_key = self.full_key(key);
        let mut storage = self.storage.write();
        storage.insert(full_key, data.to_vec());
    }

    pub fn delete(&self, key: &str) {
        let full_key = self.full_key(key);
        let mut storage = self.storage.write();
        storage.remove(&full_key);
    }

    pub fn list(&self, prefix: &str) -> Vec<String> {
        let full_prefix = self.full_key(prefix);
        let storage = self.storage.read();
        storage
            .keys()
            .filter(|k| k.starts_with(&full_prefix))
            .map(|k| k.strip_prefix(&self.base_prefix).unwrap_or(k).to_string())
            .collect()
    }

    pub fn exists(&self, key: &str) -> bool {
        let full_key = self.full_key(key);
        let storage = self.storage.read();
        storage.contains_key(&full_key)
    }
}
