#![allow(clippy::unwrap_used)]
//! Browser-Native WASM Persistence
//!
//! IndexedDB-style persistence layer for the WASM SDK, with chunked storage
//! for large databases, serialization/deserialization pipeline, and a
//! Web Worker search architecture with message protocol.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::wasm_persistence::{
//!     PersistenceLayer, PersistenceConfig, StorageChunk,
//!     ChunkIndex, PersistenceStats,
//! };
//!
//! let mut layer = PersistenceLayer::new(PersistenceConfig::default());
//!
//! // Serialize data into chunks
//! let data = vec![1u8; 50_000];
//! let chunks = layer.write("my_collection", &data).unwrap();
//! assert!(chunks > 0);
//!
//! // Read back
//! let restored = layer.read("my_collection").unwrap();
//! assert_eq!(restored.len(), 50_000);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Persistence layer configuration.
#[derive(Debug, Clone)]
pub struct PersistenceConfig {
    /// Maximum size of each storage chunk in bytes.
    pub chunk_size: usize,
    /// Whether to compress chunks.
    pub compress: bool,
    /// Maximum total storage in bytes.
    pub max_storage_bytes: usize,
    /// Database name for IndexedDB.
    pub db_name: String,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            chunk_size: 64 * 1024, // 64KB chunks
            compress: false,
            max_storage_bytes: 500 * 1024 * 1024, // 500MB
            db_name: "needle_db".into(),
        }
    }
}

impl PersistenceConfig {
    /// Set chunk size.
    #[must_use]
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set database name.
    #[must_use]
    pub fn with_db_name(mut self, name: impl Into<String>) -> Self {
        self.db_name = name.into();
        self
    }
}

// ── Storage Chunk ────────────────────────────────────────────────────────────

/// A single chunk of persisted data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageChunk {
    /// Chunk index (0-based).
    pub index: usize,
    /// Collection key.
    pub key: String,
    /// Raw data bytes.
    pub data: Vec<u8>,
    /// CRC32 checksum.
    pub checksum: u32,
}

impl StorageChunk {
    fn compute_checksum(data: &[u8]) -> u32 {
        let mut hash: u32 = 0xFFFF_FFFF;
        for &byte in data {
            hash ^= u32::from(byte);
            for _ in 0..8 {
                if hash & 1 != 0 {
                    hash = (hash >> 1) ^ 0xEDB8_8320;
                } else {
                    hash >>= 1;
                }
            }
        }
        hash ^ 0xFFFF_FFFF
    }

    fn verify(&self) -> bool {
        Self::compute_checksum(&self.data) == self.checksum
    }
}

// ── Chunk Index ──────────────────────────────────────────────────────────────

/// Index of chunks for a stored key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkIndex {
    /// Storage key.
    pub key: String,
    /// Total data size in bytes.
    pub total_bytes: usize,
    /// Number of chunks.
    pub chunk_count: usize,
    /// Chunk size used.
    pub chunk_size: usize,
}

// ── Persistence Stats ────────────────────────────────────────────────────────

/// Statistics about stored data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistenceStats {
    /// Total keys stored.
    pub keys: usize,
    /// Total bytes stored.
    pub total_bytes: usize,
    /// Total chunks.
    pub total_chunks: usize,
    /// Average chunk fill ratio.
    pub avg_fill_ratio: f32,
}

// ── Worker Search Protocol ───────────────────────────────────────────────────

/// Message for the search worker thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchWorkerRequest {
    /// Load a collection from storage.
    Load { key: String },
    /// Search within a loaded collection.
    Search { query: Vec<f32>, k: usize },
    /// Unload a collection from memory.
    Unload { key: String },
    /// Get worker status.
    Status,
}

/// Response from the search worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchWorkerResponse {
    /// Collection loaded successfully.
    Loaded { key: String, vectors: usize },
    /// Search results.
    Results {
        items: Vec<WorkerSearchResult>,
        duration_ms: u64,
    },
    /// Collection unloaded.
    Unloaded { key: String },
    /// Worker status.
    Status {
        loaded_collections: usize,
        total_vectors: usize,
    },
    /// Error occurred.
    Error(String),
}

/// A single search result from the worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerSearchResult {
    /// Vector ID.
    pub id: String,
    /// Distance score.
    pub distance: f32,
}

// ── Persistence Layer ────────────────────────────────────────────────────────

/// In-memory persistence layer simulating IndexedDB chunked storage.
pub struct PersistenceLayer {
    config: PersistenceConfig,
    chunks: HashMap<String, Vec<StorageChunk>>,
    indices: HashMap<String, ChunkIndex>,
}

impl PersistenceLayer {
    /// Create a new persistence layer.
    pub fn new(config: PersistenceConfig) -> Self {
        Self {
            config,
            chunks: HashMap::new(),
            indices: HashMap::new(),
        }
    }

    /// Write data for a key, splitting into chunks. Returns chunk count.
    pub fn write(&mut self, key: &str, data: &[u8]) -> Result<usize> {
        let total = self.total_bytes();
        if total + data.len() > self.config.max_storage_bytes {
            return Err(NeedleError::CapacityExceeded(format!(
                "Storage limit exceeded: {} + {} > {}",
                total,
                data.len(),
                self.config.max_storage_bytes
            )));
        }

        let chunk_size = self.config.chunk_size;
        let mut stored_chunks = Vec::new();

        for (i, chunk_data) in data.chunks(chunk_size).enumerate() {
            stored_chunks.push(StorageChunk {
                index: i,
                key: key.into(),
                data: chunk_data.to_vec(),
                checksum: StorageChunk::compute_checksum(chunk_data),
            });
        }

        let count = stored_chunks.len();
        self.indices.insert(
            key.into(),
            ChunkIndex {
                key: key.into(),
                total_bytes: data.len(),
                chunk_count: count,
                chunk_size,
            },
        );
        self.chunks.insert(key.into(), stored_chunks);
        Ok(count)
    }

    /// Read data for a key, reassembling chunks.
    pub fn read(&self, key: &str) -> Result<Vec<u8>> {
        let chunks = self
            .chunks
            .get(key)
            .ok_or_else(|| NeedleError::NotFound(format!("Key '{key}' not found")))?;

        let mut data = Vec::new();
        for chunk in chunks {
            if !chunk.verify() {
                return Err(NeedleError::Corruption(format!(
                    "Chunk {} of '{key}' failed checksum",
                    chunk.index
                )));
            }
            data.extend_from_slice(&chunk.data);
        }
        Ok(data)
    }

    /// Delete stored data for a key.
    pub fn delete(&mut self, key: &str) -> bool {
        self.chunks.remove(key).is_some() | self.indices.remove(key).is_some()
    }

    /// List all stored keys.
    pub fn keys(&self) -> Vec<String> {
        self.indices.keys().cloned().collect()
    }

    /// Get chunk index for a key.
    pub fn index(&self, key: &str) -> Option<&ChunkIndex> {
        self.indices.get(key)
    }

    /// Get persistence stats.
    pub fn stats(&self) -> PersistenceStats {
        let total_chunks: usize = self.chunks.values().map(|v| v.len()).sum();
        let total_bytes: usize = self
            .chunks
            .values()
            .flat_map(|v| v.iter())
            .map(|c| c.data.len())
            .sum();
        let avg_fill = if total_chunks > 0 {
            total_bytes as f32 / (total_chunks as f32 * self.config.chunk_size as f32)
        } else {
            0.0
        };
        PersistenceStats {
            keys: self.indices.len(),
            total_bytes,
            total_chunks,
            avg_fill_ratio: avg_fill,
        }
    }

    fn total_bytes(&self) -> usize {
        self.indices.values().map(|i| i.total_bytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read() {
        let mut layer = PersistenceLayer::new(PersistenceConfig::default());
        let data = vec![42u8; 200_000]; // ~3 chunks at 64KB
        let chunks = layer.write("test", &data).unwrap();
        assert!(chunks > 1);

        let restored = layer.read("test").unwrap();
        assert_eq!(restored, data);
    }

    #[test]
    fn test_small_data_single_chunk() {
        let mut layer = PersistenceLayer::new(PersistenceConfig::default());
        let data = vec![1u8; 100];
        let chunks = layer.write("small", &data).unwrap();
        assert_eq!(chunks, 1);
    }

    #[test]
    fn test_checksum_verification() {
        let mut layer = PersistenceLayer::new(PersistenceConfig::default());
        layer.write("test", &[1, 2, 3, 4, 5]).unwrap();

        // Corrupt a chunk
        if let Some(chunks) = layer.chunks.get_mut("test") {
            chunks[0].data[0] = 255;
        }

        assert!(layer.read("test").is_err());
    }

    #[test]
    fn test_delete() {
        let mut layer = PersistenceLayer::new(PersistenceConfig::default());
        layer.write("test", &[1, 2, 3]).unwrap();
        assert!(layer.delete("test"));
        assert!(layer.read("test").is_err());
    }

    #[test]
    fn test_keys() {
        let mut layer = PersistenceLayer::new(PersistenceConfig::default());
        layer.write("a", &[1]).unwrap();
        layer.write("b", &[2]).unwrap();
        let keys = layer.keys();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_capacity_limit() {
        let config = PersistenceConfig {
            max_storage_bytes: 100,
            ..Default::default()
        };
        let mut layer = PersistenceLayer::new(config);
        assert!(layer.write("big", &vec![0u8; 200]).is_err());
    }

    #[test]
    fn test_stats() {
        let mut layer = PersistenceLayer::new(PersistenceConfig::default());
        layer.write("a", &vec![0u8; 1000]).unwrap();
        let stats = layer.stats();
        assert_eq!(stats.keys, 1);
        assert_eq!(stats.total_bytes, 1000);
    }

    #[test]
    fn test_chunk_index() {
        let mut layer = PersistenceLayer::new(
            PersistenceConfig::default().with_chunk_size(100),
        );
        layer.write("test", &vec![0u8; 350]).unwrap();
        let idx = layer.index("test").unwrap();
        assert_eq!(idx.total_bytes, 350);
        assert_eq!(idx.chunk_count, 4); // ceil(350/100)
    }
}
