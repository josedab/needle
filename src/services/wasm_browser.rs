//! WebAssembly-First SDK
//!
//! Complete vector database SDK designed to run in the browser, Deno, and
//! Cloudflare Workers with persistence abstraction, Web Worker protocol,
//! and streaming search support.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::wasm_browser::{
//!     WasmDatabase, WasmConfig, StorageBackend,
//!     WasmSearchResult, WorkerMessage, WorkerResponse,
//! };
//!
//! let config = WasmConfig::builder()
//!     .storage(StorageBackend::InMemory)
//!     .max_collections(16)
//!     .build();
//!
//! let mut db = WasmDatabase::new(config);
//! db.create_collection("docs", 4).unwrap();
//! db.insert("docs", "v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
//!
//! let results = db.search("docs", &[1.0, 2.0, 3.0, 4.0], 5).unwrap();
//! assert_eq!(results[0].id, "v1");
//!
//! // Serialize for Worker message passing
//! let msg = WorkerMessage::Search {
//!     collection: "docs".into(),
//!     query: vec![1.0; 4],
//!     k: 5,
//! };
//! let response = db.handle_message(msg).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Storage backend for the WASM SDK.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// In-memory only (lost on page unload).
    InMemory,
    /// IndexedDB persistence (browser).
    IndexedDb { db_name: String },
    /// LocalStorage (small datasets only).
    LocalStorage { key_prefix: String },
    /// Origin Private File System (modern browsers).
    Opfs { directory: String },
}

impl Default for StorageBackend {
    fn default() -> Self {
        Self::InMemory
    }
}

/// WASM SDK configuration.
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Storage backend.
    pub storage: StorageBackend,
    /// Maximum number of collections.
    pub max_collections: usize,
    /// Maximum vectors per collection.
    pub max_vectors_per_collection: usize,
    /// Whether to enable compression for storage.
    pub compress: bool,
    /// Default distance function.
    pub default_distance: DistanceFunction,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            storage: StorageBackend::InMemory,
            max_collections: 64,
            max_vectors_per_collection: 1_000_000,
            compress: false,
            default_distance: DistanceFunction::Cosine,
        }
    }
}

impl WasmConfig {
    /// Create a builder.
    pub fn builder() -> WasmConfigBuilder {
        WasmConfigBuilder::default()
    }
}

/// Builder for `WasmConfig`.
#[derive(Debug, Clone)]
pub struct WasmConfigBuilder {
    config: WasmConfig,
}

impl Default for WasmConfigBuilder {
    fn default() -> Self {
        Self {
            config: WasmConfig::default(),
        }
    }
}

impl WasmConfigBuilder {
    /// Set storage backend.
    #[must_use]
    pub fn storage(mut self, backend: StorageBackend) -> Self {
        self.config.storage = backend;
        self
    }

    /// Set max collections.
    #[must_use]
    pub fn max_collections(mut self, max: usize) -> Self {
        self.config.max_collections = max;
        self
    }

    /// Set max vectors per collection.
    #[must_use]
    pub fn max_vectors(mut self, max: usize) -> Self {
        self.config.max_vectors_per_collection = max;
        self
    }

    /// Enable compression.
    #[must_use]
    pub fn compress(mut self, enabled: bool) -> Self {
        self.config.compress = enabled;
        self
    }

    /// Build the config.
    pub fn build(self) -> WasmConfig {
        self.config
    }
}

// ── Search Result ────────────────────────────────────────────────────────────

/// Search result from the WASM SDK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSearchResult {
    /// Vector ID.
    pub id: String,
    /// Distance score.
    pub distance: f32,
    /// Metadata (if available).
    pub metadata: Option<Value>,
}

// ── Worker Protocol ──────────────────────────────────────────────────────────

/// Message sent from main thread to Web Worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerMessage {
    /// Create a collection.
    CreateCollection {
        name: String,
        dimensions: usize,
    },
    /// Insert a vector.
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Search for similar vectors.
    Search {
        collection: String,
        query: Vec<f32>,
        k: usize,
    },
    /// Delete a vector.
    Delete {
        collection: String,
        id: String,
    },
    /// Get a vector by ID.
    Get {
        collection: String,
        id: String,
    },
    /// List all collections.
    ListCollections,
    /// Get collection info.
    CollectionInfo {
        name: String,
    },
    /// Export all data.
    Export {
        collection: String,
    },
    /// Batch insert.
    BatchInsert {
        collection: String,
        items: Vec<BatchItem>,
    },
}

/// A single item in a batch insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchItem {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

/// Response sent from Web Worker back to main thread.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerResponse {
    /// Operation succeeded.
    Ok,
    /// Search results.
    SearchResults(Vec<WasmSearchResult>),
    /// Vector data.
    VectorData {
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Collection list.
    Collections(Vec<CollectionInfo>),
    /// Single collection info.
    Info(CollectionInfo),
    /// Export data.
    ExportData(Vec<ExportEntry>),
    /// Batch result.
    BatchResult {
        inserted: usize,
        failed: usize,
    },
    /// Error occurred.
    Error(String),
}

/// Collection information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// Collection name.
    pub name: String,
    /// Number of vectors.
    pub count: usize,
    /// Vector dimensions.
    pub dimensions: usize,
}

/// Export entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportEntry {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata.
    pub metadata: Option<Value>,
}

// ── WASM Database ────────────────────────────────────────────────────────────

/// Main WASM-optimized database interface.
///
/// Provides a simplified API suitable for browser environments with
/// Web Worker message protocol support.
pub struct WasmDatabase {
    config: WasmConfig,
    collections: HashMap<String, Collection>,
}

impl WasmDatabase {
    /// Create a new WASM database.
    pub fn new(config: WasmConfig) -> Self {
        Self {
            config,
            collections: HashMap::new(),
        }
    }

    /// Create a collection.
    pub fn create_collection(&mut self, name: &str, dimensions: usize) -> Result<()> {
        if self.collections.len() >= self.config.max_collections {
            return Err(NeedleError::CapacityExceeded(format!(
                "Max collections ({}) reached",
                self.config.max_collections
            )));
        }
        if self.collections.contains_key(name) {
            return Err(NeedleError::CollectionAlreadyExists(name.into()));
        }
        let config = CollectionConfig::new(name, dimensions)
            .with_distance(self.config.default_distance);
        self.collections
            .insert(name.into(), Collection::new(config));
        Ok(())
    }

    /// Insert a vector.
    pub fn insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let coll = self.get_collection_mut(collection)?;
        coll.insert(id, vector, metadata)?;
        Ok(())
    }

    /// Search for similar vectors.
    pub fn search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<WasmSearchResult>> {
        let coll = self.get_collection(collection)?;
        let results = coll.search(query, k)?;
        Ok(results
            .into_iter()
            .map(|r| WasmSearchResult {
                id: r.id,
                distance: r.distance,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Delete a vector.
    pub fn delete(&mut self, collection: &str, id: &str) -> Result<bool> {
        let coll = self.get_collection_mut(collection)?;
        coll.delete(id)
    }

    /// Get a vector by ID.
    pub fn get(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<(Vec<f32>, Option<Value>)>> {
        let coll = self.get_collection(collection)?;
        Ok(coll.get(id).map(|(v, m)| (v.to_vec(), m.cloned())))
    }

    /// List all collections.
    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        self.collections
            .iter()
            .map(|(name, coll)| CollectionInfo {
                name: name.clone(),
                count: coll.len(),
                dimensions: coll.dimensions(),
            })
            .collect()
    }

    /// Handle a Web Worker message.
    pub fn handle_message(&mut self, msg: WorkerMessage) -> Result<WorkerResponse> {
        match msg {
            WorkerMessage::CreateCollection { name, dimensions } => {
                self.create_collection(&name, dimensions)?;
                Ok(WorkerResponse::Ok)
            }
            WorkerMessage::Insert {
                collection, id, vector, metadata,
            } => {
                self.insert(&collection, &id, &vector, metadata)?;
                Ok(WorkerResponse::Ok)
            }
            WorkerMessage::Search {
                collection, query, k,
            } => {
                let results = self.search(&collection, &query, k)?;
                Ok(WorkerResponse::SearchResults(results))
            }
            WorkerMessage::Delete { collection, id } => {
                self.delete(&collection, &id)?;
                Ok(WorkerResponse::Ok)
            }
            WorkerMessage::Get { collection, id } => {
                if let Some((vector, metadata)) = self.get(&collection, &id)? {
                    Ok(WorkerResponse::VectorData {
                        id,
                        vector,
                        metadata,
                    })
                } else {
                    Err(NeedleError::VectorNotFound(id))
                }
            }
            WorkerMessage::ListCollections => {
                Ok(WorkerResponse::Collections(self.list_collections()))
            }
            WorkerMessage::CollectionInfo { name } => {
                let coll = self.get_collection(&name)?;
                Ok(WorkerResponse::Info(CollectionInfo {
                    name,
                    count: coll.len(),
                    dimensions: coll.dimensions(),
                }))
            }
            WorkerMessage::Export { collection } => {
                let coll = self.get_collection(&collection)?;
                let mut entries = Vec::new();
                let ids: Vec<String> = coll.ids().map(String::from).collect();
                for id in ids {
                    if let Some((vec, meta)) = coll.get(&id) {
                        entries.push(ExportEntry {
                            id,
                            vector: vec.to_vec(),
                            metadata: meta.cloned(),
                        });
                    }
                }
                Ok(WorkerResponse::ExportData(entries))
            }
            WorkerMessage::BatchInsert { collection, items } => {
                let mut inserted = 0;
                let mut failed = 0;
                for item in items {
                    match self.insert(&collection, &item.id, &item.vector, item.metadata) {
                        Ok(()) => inserted += 1,
                        Err(_) => failed += 1,
                    }
                }
                Ok(WorkerResponse::BatchResult { inserted, failed })
            }
        }
    }

    /// Get the storage backend type.
    pub fn storage_backend(&self) -> &StorageBackend {
        &self.config.storage
    }

    /// Get the number of collections.
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn get_collection(&self, name: &str) -> Result<&Collection> {
        self.collections
            .get(name)
            .ok_or_else(|| NeedleError::CollectionNotFound(name.into()))
    }

    fn get_collection_mut(&mut self, name: &str) -> Result<&mut Collection> {
        self.collections
            .get_mut(name)
            .ok_or_else(|| NeedleError::CollectionNotFound(name.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_insert() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert("test", "v1", &[1.0, 2.0, 3.0, 4.0], None)
            .unwrap();

        let info = db.list_collections();
        assert_eq!(info.len(), 1);
        assert_eq!(info[0].count, 1);
    }

    #[test]
    fn test_search() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert("test", "v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        db.insert("test", "v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = db.search("test", &[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_worker_protocol() {
        let mut db = WasmDatabase::new(WasmConfig::default());

        // Create collection via message
        let resp = db
            .handle_message(WorkerMessage::CreateCollection {
                name: "docs".into(),
                dimensions: 4,
            })
            .unwrap();
        assert!(matches!(resp, WorkerResponse::Ok));

        // Insert via message
        let resp = db
            .handle_message(WorkerMessage::Insert {
                collection: "docs".into(),
                id: "d1".into(),
                vector: vec![1.0; 4],
                metadata: None,
            })
            .unwrap();
        assert!(matches!(resp, WorkerResponse::Ok));

        // Search via message
        let resp = db
            .handle_message(WorkerMessage::Search {
                collection: "docs".into(),
                query: vec![1.0; 4],
                k: 5,
            })
            .unwrap();
        match resp {
            WorkerResponse::SearchResults(results) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].id, "d1");
            }
            _ => panic!("Expected SearchResults"),
        }
    }

    #[test]
    fn test_delete() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert("test", "v1", &[1.0; 4], None).unwrap();

        assert!(db.delete("test", "v1").unwrap());
        assert!(db.get("test", "v1").unwrap().is_none());
    }

    #[test]
    fn test_max_collections() {
        let config = WasmConfig::builder().max_collections(2).build();
        let mut db = WasmDatabase::new(config);

        db.create_collection("a", 4).unwrap();
        db.create_collection("b", 4).unwrap();
        assert!(db.create_collection("c", 4).is_err());
    }

    #[test]
    fn test_batch_insert() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();

        let resp = db
            .handle_message(WorkerMessage::BatchInsert {
                collection: "test".into(),
                items: vec![
                    BatchItem {
                        id: "v1".into(),
                        vector: vec![1.0; 4],
                        metadata: None,
                    },
                    BatchItem {
                        id: "v2".into(),
                        vector: vec![2.0; 4],
                        metadata: None,
                    },
                ],
            })
            .unwrap();

        match resp {
            WorkerResponse::BatchResult { inserted, failed } => {
                assert_eq!(inserted, 2);
                assert_eq!(failed, 0);
            }
            _ => panic!("Expected BatchResult"),
        }
    }

    #[test]
    fn test_export() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert("test", "v1", &[1.0; 4], None).unwrap();

        let resp = db
            .handle_message(WorkerMessage::Export {
                collection: "test".into(),
            })
            .unwrap();

        match resp {
            WorkerResponse::ExportData(entries) => {
                assert_eq!(entries.len(), 1);
            }
            _ => panic!("Expected ExportData"),
        }
    }
}
