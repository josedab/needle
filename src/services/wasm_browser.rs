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
    /// Search with a metadata filter.
    SearchFiltered {
        collection: String,
        query: Vec<f32>,
        k: usize,
        filter: Value,
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
    /// Delete a collection.
    DeleteCollection {
        name: String,
    },
    /// Import data into a collection.
    Import {
        collection: String,
        entries: Vec<BatchItem>,
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

    /// Delete a collection.
    pub fn delete_collection(&mut self, name: &str) -> Result<bool> {
        Ok(self.collections.remove(name).is_some())
    }

    /// Search with a JSON filter (parsed from a serde_json::Value).
    pub fn search_filtered(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter_json: &Value,
    ) -> Result<Vec<WasmSearchResult>> {
        let coll = self.get_collection(collection)?;
        let filter = crate::metadata::Filter::parse(filter_json)
            .map_err(|e| NeedleError::InvalidArgument(e))?;
        let results = coll.search_with_filter(query, k, &filter)?;
        Ok(results
            .into_iter()
            .map(|r| WasmSearchResult {
                id: r.id,
                distance: r.distance,
                metadata: r.metadata,
            })
            .collect())
    }

    /// Serialize the entire database for persistence (IndexedDB / localStorage).
    pub fn serialize_state(&self) -> Result<Value> {
        let mut collections = serde_json::Map::new();
        for (name, coll) in &self.collections {
            let mut entries = Vec::new();
            let ids: Vec<String> = coll.ids().map(String::from).collect();
            for id in ids {
                if let Some((vec, meta)) = coll.get(&id) {
                    entries.push(serde_json::json!({
                        "id": id,
                        "vector": vec.to_vec(),
                        "metadata": meta.cloned(),
                    }));
                }
            }
            collections.insert(
                name.clone(),
                serde_json::json!({
                    "dimensions": coll.dimensions(),
                    "entries": entries,
                }),
            );
        }
        Ok(Value::Object(collections))
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
            WorkerMessage::SearchFiltered {
                collection, query, k, filter,
            } => {
                let results = self.search_filtered(&collection, &query, k, &filter)?;
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
            WorkerMessage::DeleteCollection { name } => {
                self.delete_collection(&name)?;
                Ok(WorkerResponse::Ok)
            }
            WorkerMessage::Import { collection, entries } => {
                let mut inserted = 0;
                let mut failed = 0;
                for item in entries {
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

// ── Sync Protocol ───────────────────────────────────────────────────────────

/// Sync direction for server reconciliation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncDirection {
    /// Client pushes changes to server.
    Push,
    /// Client pulls changes from server.
    Pull,
    /// Bidirectional sync.
    Bidirectional,
}

/// A change entry for sync protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncChange {
    /// Operation type.
    pub op: SyncOp,
    /// Collection name.
    pub collection: String,
    /// Timestamp (epoch ms).
    pub timestamp: u64,
}

/// Sync operation type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SyncOp {
    /// Insert a new vector.
    Insert { id: String, vector: Vec<f32>, metadata: Option<Value> },
    /// Update an existing vector.
    Update { id: String, vector: Vec<f32>, metadata: Option<Value> },
    /// Delete a vector.
    Delete { id: String },
    /// Create a collection.
    CreateCollection { name: String, dimensions: usize },
    /// Delete a collection.
    DeleteCollection { name: String },
}

/// Sync status between local and remote.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncState {
    /// Local version counter.
    pub local_version: u64,
    /// Remote version counter.
    pub remote_version: u64,
    /// Pending changes to sync.
    pub pending_changes: usize,
    /// Last sync timestamp.
    pub last_sync_at: Option<u64>,
    /// Sync direction.
    pub direction: SyncDirection,
}

/// Sync result after reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResult {
    /// Changes pushed to server.
    pub pushed: usize,
    /// Changes pulled from server.
    pub pulled: usize,
    /// Conflicts detected.
    pub conflicts: usize,
    /// New local version after sync.
    pub new_version: u64,
}

// ── React Hook Types ────────────────────────────────────────────────────────

/// TypeScript-compatible hook return type for useNeedleSearch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHookState {
    /// Search results.
    pub results: Vec<WasmSearchResult>,
    /// Whether a search is in progress.
    pub loading: bool,
    /// Error message if search failed.
    pub error: Option<String>,
    /// Time taken in milliseconds.
    pub latency_ms: f64,
}

impl Default for SearchHookState {
    fn default() -> Self {
        Self {
            results: Vec::new(),
            loading: false,
            error: None,
            latency_ms: 0.0,
        }
    }
}

/// TypeScript-compatible hook return type for useCollection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionHookState {
    /// Collection information.
    pub info: Option<CollectionInfo>,
    /// Whether the collection is loaded.
    pub ready: bool,
    /// Error message if loading failed.
    pub error: Option<String>,
}

impl Default for CollectionHookState {
    fn default() -> Self {
        Self {
            info: None,
            ready: false,
            error: None,
        }
    }
}

/// TypeScript type definition export for the npm package.
pub fn typescript_definitions() -> &'static str {
    r#"
export interface NeedleConfig {
  storage?: 'memory' | 'indexeddb' | 'localstorage' | 'opfs';
  maxCollections?: number;
  maxVectorsPerCollection?: number;
  compress?: boolean;
}

export interface SearchResult {
  id: string;
  distance: number;
  metadata?: Record<string, unknown>;
}

export interface CollectionInfo {
  name: string;
  count: number;
  dimensions: number;
}

export function useNeedleSearch(
  collection: string,
  query: number[],
  k?: number
): {
  results: SearchResult[];
  loading: boolean;
  error: string | null;
  latency_ms: number;
};

export function useCollection(
  name: string
): {
  info: CollectionInfo | null;
  ready: boolean;
  error: string | null;
  insert: (id: string, vector: number[], metadata?: Record<string, unknown>) => Promise<void>;
  search: (query: number[], k?: number) => Promise<SearchResult[]>;
  delete: (id: string) => Promise<boolean>;
};
"#
}

// ── IndexedDB Restore ───────────────────────────────────────────────────────

impl WasmDatabase {
    /// Restore database state from a serialized JSON (from IndexedDB/localStorage).
    pub fn restore_state(&mut self, state: &Value) -> Result<RestoreResult> {
        let obj = state.as_object()
            .ok_or_else(|| NeedleError::InvalidArgument("State must be a JSON object".into()))?;

        let mut restored_collections = 0usize;
        let mut restored_vectors = 0usize;
        let mut errors = 0usize;

        for (name, coll_data) in obj {
            let dims = coll_data.get("dimensions")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            if dims == 0 {
                errors += 1;
                continue;
            }

            if !self.collections.contains_key(name) {
                if let Err(_) = self.create_collection(name, dims) {
                    errors += 1;
                    continue;
                }
            }
            restored_collections += 1;

            if let Some(entries) = coll_data.get("entries").and_then(|v| v.as_array()) {
                for entry in entries {
                    let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let vector: Vec<f32> = entry.get("vector")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                        .unwrap_or_default();
                    let metadata = entry.get("metadata").cloned();

                    if !id.is_empty() && !vector.is_empty() {
                        match self.insert(name, id, &vector, metadata) {
                            Ok(()) => restored_vectors += 1,
                            Err(_) => errors += 1,
                        }
                    }
                }
            }
        }

        Ok(RestoreResult { restored_collections, restored_vectors, errors })
    }
}

/// Result of restoring state from persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreResult {
    /// Number of collections restored.
    pub restored_collections: usize,
    /// Number of vectors restored.
    pub restored_vectors: usize,
    /// Number of errors encountered during restore.
    pub errors: usize,
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

    #[test]
    fn test_delete_collection() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        assert_eq!(db.collection_count(), 1);

        assert!(db.delete_collection("test").unwrap());
        assert_eq!(db.collection_count(), 0);
    }

    #[test]
    fn test_search_filtered() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert(
            "test",
            "v1",
            &[1.0; 4],
            Some(serde_json::json!({"color": "red"})),
        )
        .unwrap();
        db.insert(
            "test",
            "v2",
            &[2.0; 4],
            Some(serde_json::json!({"color": "blue"})),
        )
        .unwrap();

        let filter = serde_json::json!({"color": {"$eq": "red"}});
        let results = db.search_filtered("test", &[1.0; 4], 5, &filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_serialize_state() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert("test", "v1", &[1.0; 4], None).unwrap();

        let state = db.serialize_state().unwrap();
        assert!(state.get("test").is_some());
    }

    #[test]
    fn test_worker_delete_collection() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();

        let resp = db
            .handle_message(WorkerMessage::DeleteCollection {
                name: "test".into(),
            })
            .unwrap();
        assert!(matches!(resp, WorkerResponse::Ok));
        assert_eq!(db.collection_count(), 0);
    }

    #[test]
    fn test_serialize_and_restore() {
        let mut db = WasmDatabase::new(WasmConfig::default());
        db.create_collection("test", 4).unwrap();
        db.insert("test", "v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
        db.insert("test", "v2", &[0.5; 4], Some(serde_json::json!({"tag": "x"}))).unwrap();

        let state = db.serialize_state().unwrap();

        let mut db2 = WasmDatabase::new(WasmConfig::default());
        let result = db2.restore_state(&state).unwrap();
        assert_eq!(result.restored_collections, 1);
        assert_eq!(result.restored_vectors, 2);
        assert_eq!(result.errors, 0);

        let results = db2.search("test", &[1.0, 2.0, 3.0, 4.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_sync_types() {
        let state = SyncState {
            local_version: 5,
            remote_version: 3,
            pending_changes: 2,
            last_sync_at: None,
            direction: SyncDirection::Push,
        };
        assert_eq!(state.pending_changes, 2);
        assert_eq!(state.direction, SyncDirection::Push);
    }

    #[test]
    fn test_search_hook_state_default() {
        let state = SearchHookState::default();
        assert!(state.results.is_empty());
        assert!(!state.loading);
        assert!(state.error.is_none());
    }

    #[test]
    fn test_typescript_definitions() {
        let defs = typescript_definitions();
        assert!(defs.contains("useNeedleSearch"));
        assert!(defs.contains("useCollection"));
        assert!(defs.contains("SearchResult"));
    }
}
