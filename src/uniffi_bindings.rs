//! UniFFI bindings for Swift and Kotlin
//!
//! This module provides cross-platform mobile bindings using UniFFI's proc-macro approach,
//! enabling iOS (Swift) and Android (Kotlin) applications to use Needle.
//!
//! # Features
//!
//! - Thread-safe collection management with RwLock and Arc
//! - Vector insertion with optional JSON metadata
//! - Similarity search with configurable k and distance functions
//! - Metadata filtering using MongoDB-style query syntax
//! - Batch operations for efficient bulk processing
//! - Serialization/deserialization for persistence
//!
//! # Building
//!
//! Build the shared library:
//! ```bash
//! cargo build --release --features uniffi-bindings
//! ```
//!
//! # Generating Bindings
//!
//! Generate Swift bindings (macOS/iOS):
//! ```bash
//! cargo run --features uniffi-bindings --bin uniffi-bindgen generate \
//!     --library target/release/libneedle.dylib \
//!     --language swift \
//!     --out-dir ./bindings/swift
//! ```
//!
//! Generate Kotlin bindings (Android):
//! ```bash
//! cargo run --features uniffi-bindings --bin uniffi-bindgen generate \
//!     --library target/release/libneedle.so \
//!     --language kotlin \
//!     --out-dir ./bindings/kotlin
//! ```
//!
//! # Swift Usage Example
//!
//! ```swift
//! let collection = try NeedleCollection(name: "vectors", dimensions: 128, distance: "cosine")
//! try collection.insert(id: "id1", vector: Array(repeating: 0.1, count: 128), metadata: nil)
//! let results = try collection.search(query: Array(repeating: 0.1, count: 128), k: 5)
//! ```
//!
//! # Kotlin Usage Example
//!
//! ```kotlin
//! val collection = NeedleCollection("vectors", 128, "cosine")
//! collection.insert("id1", FloatArray(128) { 0.1f }, null)
//! val results = collection.search(FloatArray(128) { 0.1f }, 5)
//! ```

use crate::collection::{Collection, CollectionConfig, SearchResult as RustSearchResult};
use crate::distance::DistanceFunction;
use crate::metadata::Filter;
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::Arc;

/// Get the library version
#[uniffi::export]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Error type for UniFFI
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum NeedleError {
    #[error("IO error: {msg}")]
    Io { msg: String },
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: u32, got: u32 },
    #[error("Collection not found: {name}")]
    CollectionNotFound { name: String },
    #[error("Collection already exists: {name}")]
    CollectionAlreadyExists { name: String },
    #[error("Vector already exists: {id}")]
    VectorAlreadyExists { id: String },
    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },
    #[error("Invalid configuration: {msg}")]
    InvalidConfig { msg: String },
    #[error("Index error: {msg}")]
    Index { msg: String },
    #[error("Data corruption: {msg}")]
    Corruption { msg: String },
    #[error("Serialization error: {msg}")]
    Serialization { msg: String },
    #[error("Invalid database: {msg}")]
    InvalidDatabase { msg: String },
    #[error("Capacity exceeded: {msg}")]
    CapacityExceeded { msg: String },
    #[error("Invalid vector: {msg}")]
    InvalidVector { msg: String },
    #[error("Invalid input: {msg}")]
    InvalidInput { msg: String },
    #[error("Quota exceeded: {msg}")]
    QuotaExceeded { msg: String },
    #[error("Backup error: {msg}")]
    BackupError { msg: String },
    #[error("Not found: {msg}")]
    NotFound { msg: String },
    #[error("Conflict: {msg}")]
    Conflict { msg: String },
    #[error("Encryption error: {msg}")]
    EncryptionError { msg: String },
    #[error("Consensus error: {msg}")]
    ConsensusError { msg: String },
    #[error("Lock error")]
    LockError,
    #[error("Timeout: {msg}")]
    Timeout { msg: String },
    #[error("Lock timeout: {msg}")]
    LockTimeout { msg: String },
}

impl From<crate::error::NeedleError> for NeedleError {
    fn from(err: crate::error::NeedleError) -> Self {
        match err {
            crate::error::NeedleError::Io(e) => NeedleError::Io { msg: e.to_string() },
            crate::error::NeedleError::DimensionMismatch { expected, got } => {
                NeedleError::DimensionMismatch {
                    expected: expected as u32,
                    got: got as u32,
                }
            }
            crate::error::NeedleError::CollectionNotFound(name) => {
                NeedleError::CollectionNotFound { name }
            }
            crate::error::NeedleError::CollectionAlreadyExists(name) => {
                NeedleError::CollectionAlreadyExists { name }
            }
            crate::error::NeedleError::VectorAlreadyExists(id) => {
                NeedleError::VectorAlreadyExists { id }
            }
            crate::error::NeedleError::VectorNotFound(id) => NeedleError::VectorNotFound { id },
            crate::error::NeedleError::InvalidConfig(msg) => NeedleError::InvalidConfig { msg },
            crate::error::NeedleError::Index(msg) => NeedleError::Index { msg },
            crate::error::NeedleError::Corruption(msg) => NeedleError::Corruption { msg },
            crate::error::NeedleError::Serialization(e) => {
                NeedleError::Serialization { msg: e.to_string() }
            }
            crate::error::NeedleError::InvalidDatabase(msg) => NeedleError::InvalidDatabase { msg },
            crate::error::NeedleError::CapacityExceeded(msg) => {
                NeedleError::CapacityExceeded { msg }
            }
            crate::error::NeedleError::InvalidVector(msg) => NeedleError::InvalidVector { msg },
            crate::error::NeedleError::InvalidInput(msg) => NeedleError::InvalidInput { msg },
            crate::error::NeedleError::QuotaExceeded(msg) => NeedleError::QuotaExceeded { msg },
            crate::error::NeedleError::BackupError(msg) => NeedleError::BackupError { msg },
            crate::error::NeedleError::NotFound(msg) => NeedleError::NotFound { msg },
            crate::error::NeedleError::Conflict(msg) => NeedleError::Conflict { msg },
            crate::error::NeedleError::EncryptionError(msg) => NeedleError::EncryptionError { msg },
            crate::error::NeedleError::ConsensusError(msg) => NeedleError::ConsensusError { msg },
            crate::error::NeedleError::LockError => NeedleError::LockError,
            crate::error::NeedleError::Timeout(duration) => NeedleError::Timeout {
                msg: format!("Operation timed out after {:?}", duration),
            },
            crate::error::NeedleError::LockTimeout(duration) => NeedleError::LockTimeout {
                msg: format!("Lock acquisition timed out after {:?}", duration),
            },
            // Catch remaining variants
            other => NeedleError::InvalidInput {
                msg: other.to_string(),
            },
        }
    }
}

/// Search result for UniFFI
#[derive(Debug, Clone, uniffi::Record)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata_json: Option<String>,
}

impl From<RustSearchResult> for SearchResult {
    fn from(result: RustSearchResult) -> Self {
        Self {
            id: result.id,
            distance: result.distance,
            metadata_json: result.metadata.map(|v| v.to_string()),
        }
    }
}

/// Collection wrapper for UniFFI
#[derive(uniffi::Object)]
pub struct NeedleCollection {
    inner: RwLock<Collection>,
}

#[uniffi::export]
impl NeedleCollection {
    /// Create a new collection
    #[uniffi::constructor]
    pub fn new(name: String, dimensions: u32, distance: String) -> Result<Arc<Self>, NeedleError> {
        let dist_fn = match distance.to_lowercase().as_str() {
            "cosine" => DistanceFunction::Cosine,
            "euclidean" | "l2" => DistanceFunction::Euclidean,
            "dot" | "dotproduct" | "inner_product" => DistanceFunction::DotProduct,
            "manhattan" | "l1" => DistanceFunction::Manhattan,
            _ => {
                return Err(NeedleError::InvalidConfig {
                    msg: format!("Unknown distance function: {}", distance),
                })
            }
        };

        let config = CollectionConfig::new(name, dimensions as usize).with_distance(dist_fn);
        Ok(Arc::new(Self {
            inner: RwLock::new(Collection::new(config)),
        }))
    }

    /// Create from bytes
    #[uniffi::constructor]
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Arc<Self>, NeedleError> {
        let collection = Collection::from_bytes(&bytes)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(collection),
        }))
    }

    /// Get the collection name
    pub fn name(&self) -> String {
        self.inner.read().name().to_string()
    }

    /// Get the vector dimensions
    pub fn dimensions(&self) -> u32 {
        self.inner.read().dimensions() as u32
    }

    /// Get the number of vectors
    pub fn len(&self) -> u64 {
        self.inner.read().len() as u64
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Insert a vector with ID and optional metadata (as JSON string)
    pub fn insert(
        &self,
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
    ) -> Result<(), NeedleError> {
        let meta_value: Option<Value> = if let Some(json) = metadata_json {
            Some(
                serde_json::from_str(&json).map_err(|e| NeedleError::Serialization {
                    msg: format!("Invalid JSON metadata: {}", e),
                })?,
            )
        } else {
            None
        };

        self.inner.write().insert(&id, &vector, meta_value)?;
        Ok(())
    }

    /// Insert multiple vectors in batch
    pub fn insert_batch(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata_json_array: Option<Vec<String>>,
    ) -> Result<(), NeedleError> {
        if ids.len() != vectors.len() {
            return Err(NeedleError::InvalidConfig {
                msg: "ids and vectors must have the same length".to_string(),
            });
        }

        let meta_values: Vec<Option<Value>> = if let Some(meta_list) = metadata_json_array {
            if meta_list.len() != ids.len() {
                return Err(NeedleError::InvalidConfig {
                    msg: "metadata must have the same length as ids".to_string(),
                });
            }
            meta_list
                .into_iter()
                .map(|json| {
                    serde_json::from_str(&json)
                        .map(Some)
                        .map_err(|e| NeedleError::Serialization {
                            msg: format!("Invalid JSON metadata: {}", e),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; ids.len()]
        };

        self.inner.write().insert_batch(ids, vectors, meta_values)?;
        Ok(())
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: Vec<f32>, k: u32) -> Result<Vec<SearchResult>, NeedleError> {
        let results = self.inner.read().search(&query, k as usize)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Search with a metadata filter (JSON string)
    pub fn search_with_filter(
        &self,
        query: Vec<f32>,
        k: u32,
        filter_json: String,
    ) -> Result<Vec<SearchResult>, NeedleError> {
        let filter_value: Value =
            serde_json::from_str(&filter_json).map_err(|e| NeedleError::Serialization {
                msg: format!("Invalid filter JSON: {}", e),
            })?;

        let filter = Filter::parse(&filter_value).map_err(|e| NeedleError::InvalidConfig {
            msg: format!("Invalid filter format: {}", e),
        })?;

        let results = self
            .inner
            .read()
            .search_with_filter(&query, k as usize, &filter)?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Batch search for multiple queries
    pub fn batch_search(
        &self,
        queries: Vec<Vec<f32>>,
        k: u32,
    ) -> Result<Vec<Vec<SearchResult>>, NeedleError> {
        let results = self.inner.read().batch_search(&queries, k as usize)?;
        Ok(results
            .into_iter()
            .map(|batch| batch.into_iter().map(SearchResult::from).collect())
            .collect())
    }

    /// Get a vector by ID (returns JSON string with vector and metadata)
    pub fn get_vector_json(&self, id: String) -> Option<String> {
        let coll = self.inner.read();
        coll.get(&id).map(|(vector, metadata)| {
            serde_json::json!({
                "vector": vector,
                "metadata": metadata
            })
            .to_string()
        })
    }

    /// Check if a vector ID exists
    pub fn contains(&self, id: String) -> bool {
        self.inner.read().contains(&id)
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: String) -> Result<bool, NeedleError> {
        Ok(self.inner.write().delete(&id)?)
    }

    /// Set ef_search parameter
    pub fn set_ef_search(&self, ef: u32) {
        self.inner.write().set_ef_search(ef as usize);
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, NeedleError> {
        Ok(self.inner.read().to_bytes()?)
    }
}

// ── Offline-First Sync Queue ─────────────────────────────────────────────────
// These types support mobile clients that need to operate offline and
// synchronize changes when connectivity is restored.

/// Connection state for mobile sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Device is online and can sync.
    Online,
    /// Device is offline; operations are queued.
    Offline,
    /// Actively syncing queued operations.
    Syncing,
    /// Sync failed; will retry.
    RetryPending,
}

/// A queued operation for offline-first sync.
#[derive(Debug, Clone)]
pub enum SyncOperation {
    /// Insert a vector.
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
    },
    /// Update a vector.
    Update {
        id: String,
        vector: Vec<f32>,
        metadata_json: Option<String>,
    },
    /// Delete a vector.
    Delete { id: String },
}

/// Conflict resolution strategy when local and remote changes collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Local changes win.
    LocalWins,
    /// Remote/server changes win.
    RemoteWins,
    /// Keep both versions (suffixed with timestamp).
    KeepBoth,
}

/// Result of a sync attempt.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of operations successfully synced.
    pub synced: usize,
    /// Number of operations that failed.
    pub failed: usize,
    /// Number of conflicts resolved.
    pub conflicts: usize,
    /// Operations that remain in the queue (failed or not attempted).
    pub remaining: usize,
}

/// Offline-first sync queue for mobile clients.
///
/// Queues write operations when the device is offline and replays them
/// when connectivity is restored. Supports configurable conflict resolution
/// and retry with exponential backoff.
pub struct SyncQueue {
    /// Pending operations in FIFO order.
    queue: std::collections::VecDeque<SyncOperation>,
    /// Current connection state.
    state: ConnectionState,
    /// Conflict resolution strategy.
    conflict_strategy: ConflictStrategy,
    /// Number of retry attempts for the current batch.
    retry_count: u32,
    /// Maximum retries before giving up on a batch.
    max_retries: u32,
    /// Base retry delay in milliseconds.
    base_retry_delay_ms: u64,
}

impl SyncQueue {
    /// Create a new sync queue.
    pub fn new(conflict_strategy: ConflictStrategy) -> Self {
        Self {
            queue: std::collections::VecDeque::new(),
            state: ConnectionState::Offline,
            conflict_strategy,
            retry_count: 0,
            max_retries: 5,
            base_retry_delay_ms: 1000,
        }
    }

    /// Set maximum retry attempts.
    #[must_use]
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Enqueue an operation (stored while offline, applied immediately if online).
    pub fn enqueue(&mut self, op: SyncOperation) {
        self.queue.push_back(op);
    }

    /// Get current connection state.
    pub fn connection_state(&self) -> ConnectionState {
        self.state
    }

    /// Set connection state.
    pub fn set_connection_state(&mut self, state: ConnectionState) {
        if state == ConnectionState::Online {
            self.retry_count = 0;
        }
        self.state = state;
    }

    /// Number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Attempt to sync pending operations to the local collection.
    ///
    /// In a real mobile app this would communicate with a remote server;
    /// here we apply directly to the local collection as a reference
    /// implementation.
    pub fn sync_to_local(
        &mut self,
        collection: &NeedleCollection,
    ) -> Result<SyncResult, NeedleError> {
        if self.state == ConnectionState::Offline {
            return Ok(SyncResult {
                synced: 0,
                failed: 0,
                conflicts: 0,
                remaining: self.queue.len(),
            });
        }

        self.state = ConnectionState::Syncing;
        let mut synced = 0;
        let mut failed = 0;
        let mut conflicts = 0;
        let mut remaining_ops = std::collections::VecDeque::new();

        while let Some(op) = self.queue.pop_front() {
            match self.apply_op(collection, &op) {
                Ok(had_conflict) => {
                    synced += 1;
                    if had_conflict {
                        conflicts += 1;
                    }
                }
                Err(_) => {
                    failed += 1;
                    remaining_ops.push_back(op);
                }
            }
        }

        self.queue = remaining_ops;
        self.state = if self.queue.is_empty() {
            ConnectionState::Online
        } else if self.retry_count < self.max_retries {
            self.retry_count += 1;
            ConnectionState::RetryPending
        } else {
            ConnectionState::Online
        };

        Ok(SyncResult {
            synced,
            failed,
            conflicts,
            remaining: self.queue.len(),
        })
    }

    fn apply_op(
        &self,
        collection: &NeedleCollection,
        op: &SyncOperation,
    ) -> Result<bool, NeedleError> {
        match op {
            SyncOperation::Insert {
                id,
                vector,
                metadata_json,
            } => {
                let has_conflict = collection.contains(id.clone());
                if has_conflict && self.conflict_strategy == ConflictStrategy::RemoteWins {
                    return Ok(true);
                }
                if has_conflict && self.conflict_strategy == ConflictStrategy::KeepBoth {
                    let new_id = format!("{id}_local");
                    collection.insert(new_id, vector.clone(), metadata_json.clone())?;
                    return Ok(true);
                }
                if has_conflict {
                    // LocalWins: delete existing and re-insert
                    let _ = collection.delete(id.clone());
                }
                collection.insert(id.clone(), vector.clone(), metadata_json.clone())?;
                Ok(has_conflict)
            }
            SyncOperation::Update {
                id,
                vector,
                metadata_json,
            } => {
                let has_conflict = !collection.contains(id.clone());
                if has_conflict {
                    collection.insert(id.clone(), vector.clone(), metadata_json.clone())?;
                } else {
                    // Delete and re-insert (Collection API pattern)
                    let _ = collection.delete(id.clone());
                    collection.insert(id.clone(), vector.clone(), metadata_json.clone())?;
                }
                Ok(has_conflict)
            }
            SyncOperation::Delete { id } => {
                let existed = collection.contains(id.clone());
                if existed {
                    collection.delete(id.clone())?;
                }
                Ok(!existed)
            }
        }
    }

    /// Calculate the current retry delay using exponential backoff.
    pub fn current_retry_delay_ms(&self) -> u64 {
        self.base_retry_delay_ms * 2u64.pow(self.retry_count)
    }

    /// Drain all pending operations without syncing.
    pub fn drain(&mut self) -> Vec<SyncOperation> {
        self.queue.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_collection() -> Arc<NeedleCollection> {
        NeedleCollection::new("test".into(), 4, "cosine".into()).unwrap()
    }

    #[test]
    fn test_sync_queue_offline_enqueue() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        assert_eq!(queue.connection_state(), ConnectionState::Offline);

        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata_json: None,
        });
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn test_sync_queue_offline_no_sync() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata_json: None,
        });

        let coll = test_collection();
        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.synced, 0);
        assert_eq!(result.remaining, 1);
    }

    #[test]
    fn test_sync_queue_online_sync() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        queue.set_connection_state(ConnectionState::Online);

        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata_json: None,
        });
        queue.enqueue(SyncOperation::Insert {
            id: "v2".into(),
            vector: vec![2.0; 4],
            metadata_json: Some(r#"{"key":"value"}"#.into()),
        });

        let coll = test_collection();
        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.synced, 2);
        assert_eq!(result.remaining, 0);
        assert_eq!(coll.len(), 2);
    }

    #[test]
    fn test_sync_queue_conflict_local_wins() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        let coll = test_collection();

        // Pre-insert a vector
        coll.insert("v1".into(), vec![1.0; 4], None).unwrap();

        // Enqueue conflicting insert
        queue.set_connection_state(ConnectionState::Online);
        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![2.0; 4],
            metadata_json: None,
        });

        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.conflicts, 1);
        assert_eq!(result.synced, 1);
    }

    #[test]
    fn test_sync_queue_conflict_remote_wins() {
        let mut queue = SyncQueue::new(ConflictStrategy::RemoteWins);
        let coll = test_collection();

        coll.insert("v1".into(), vec![1.0; 4], None).unwrap();

        queue.set_connection_state(ConnectionState::Online);
        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![2.0; 4],
            metadata_json: None,
        });

        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.conflicts, 1);
        // Remote wins = don't overwrite
        assert_eq!(coll.len(), 1);
    }

    #[test]
    fn test_sync_queue_conflict_keep_both() {
        let mut queue = SyncQueue::new(ConflictStrategy::KeepBoth);
        let coll = test_collection();

        coll.insert("v1".into(), vec![1.0; 4], None).unwrap();

        queue.set_connection_state(ConnectionState::Online);
        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![2.0; 4],
            metadata_json: None,
        });

        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.conflicts, 1);
        assert_eq!(coll.len(), 2); // both v1 and v1_local
    }

    #[test]
    fn test_sync_queue_delete() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        let coll = test_collection();

        coll.insert("v1".into(), vec![1.0; 4], None).unwrap();

        queue.set_connection_state(ConnectionState::Online);
        queue.enqueue(SyncOperation::Delete { id: "v1".into() });

        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.synced, 1);
        assert_eq!(coll.len(), 0);
    }

    #[test]
    fn test_sync_queue_update() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        let coll = test_collection();

        coll.insert("v1".into(), vec![1.0; 4], None).unwrap();

        queue.set_connection_state(ConnectionState::Online);
        queue.enqueue(SyncOperation::Update {
            id: "v1".into(),
            vector: vec![9.0; 4],
            metadata_json: None,
        });

        let result = queue.sync_to_local(&coll).unwrap();
        assert_eq!(result.synced, 1);
        assert_eq!(coll.len(), 1);
    }

    #[test]
    fn test_sync_queue_retry_delay() {
        let queue = SyncQueue::new(ConflictStrategy::LocalWins);
        assert_eq!(queue.current_retry_delay_ms(), 1000);
    }

    #[test]
    fn test_sync_queue_drain() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        queue.enqueue(SyncOperation::Insert {
            id: "v1".into(),
            vector: vec![1.0; 4],
            metadata_json: None,
        });
        queue.enqueue(SyncOperation::Delete { id: "v2".into() });

        let ops = queue.drain();
        assert_eq!(ops.len(), 2);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_connection_state_transitions() {
        let mut queue = SyncQueue::new(ConflictStrategy::LocalWins);
        assert_eq!(queue.connection_state(), ConnectionState::Offline);

        queue.set_connection_state(ConnectionState::Online);
        assert_eq!(queue.connection_state(), ConnectionState::Online);

        queue.set_connection_state(ConnectionState::Syncing);
        assert_eq!(queue.connection_state(), ConnectionState::Syncing);
    }
}
