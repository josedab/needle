//! Database Management
//!
//! The database is the top-level container for collections in Needle.
//! It provides thread-safe access to collections and handles persistence.
//!
//! # Overview
//!
//! A `Database` can be either:
//! - **File-backed**: Persisted to a single `.needle` file on disk
//! - **In-memory**: Ephemeral storage for testing or temporary use
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::{Database, CollectionConfig, DistanceFunction};
//! use serde_json::json;
//!
//! // File-backed database
//! let mut db = Database::open("vectors.needle")?;
//!
//! // Create a collection
//! db.create_collection("documents", 384)?;
//!
//! // Get a thread-safe reference
//! let collection = db.collection("documents")?;
//!
//! // Insert vectors
//! collection.insert("doc1", &vec![0.1; 384], Some(json!({"title": "Hello"})))?;
//!
//! // Save to disk
//! db.save()?;
//! # Ok::<(), needle::NeedleError>(())
//! ```
//!
//! # Thread Safety
//!
//! The `Database` uses `parking_lot::RwLock` internally for concurrent access.
//! Multiple readers can access collections simultaneously, while writers get
//! exclusive access. The `CollectionRef` type provides a safe handle for
//! concurrent collection operations.
//!
//! # Persistence
//!
//! Changes are not automatically persisted. Call `save()` to write changes to disk.
//! For applications requiring durability, consider using the WAL (Write-Ahead Log)
//! feature or calling `save()` after critical operations.

use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::storage::{crc32, StorageEngine, HEADER_SIZE};
use crate::tuning::{AdaptiveTuner, RecommendedIndex, WorkloadObservation};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// Export entry type: (id, vector, metadata)
pub type ExportEntry = (String, Vec<f32>, Option<Value>);

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to the database file
    pub path: PathBuf,
    /// Whether to create if not exists
    pub create_if_missing: bool,
    /// Read-only mode
    pub read_only: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("needle.db"),
            create_if_missing: true,
            read_only: false,
        }
    }
}

impl DatabaseConfig {
    /// Create a new config with the given path
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }
}

/// Database state stored in the file
#[derive(Debug, Default, Serialize, Deserialize)]
struct DatabaseState {
    /// All collections
    collections: HashMap<String, Collection>,
    /// Aliases mapping alias name -> canonical collection name
    #[serde(default)]
    aliases: HashMap<String, String>,
}

/// The main database handle for managing vector collections.
///
/// A `Database` is the top-level container in Needle, providing:
/// - Collection management (create, drop, list)
/// - Thread-safe access to collections via [`CollectionRef`]
/// - Persistence to a single `.needle` file
/// - Atomic saves with crash protection
///
/// # Storage Modes
///
/// - **File-backed**: Created with [`Database::open()`], persists to disk
/// - **In-memory**: Created with [`Database::in_memory()`], no persistence
///
/// # Thread Safety
///
/// `Database` is fully thread-safe. Access collections through [`collection()`](Self::collection)
/// which returns a [`CollectionRef`] with proper locking.
///
/// # Example
///
/// ```
/// use needle::Database;
/// use serde_json::json;
///
/// let db = Database::in_memory();
///
/// // Create a collection
/// db.create_collection("embeddings", 384)?;
///
/// // Get a thread-safe reference
/// let coll = db.collection("embeddings")?;
///
/// // Insert vectors
/// coll.insert("doc1", &vec![0.1; 384], Some(json!({"title": "Hello"})))?;
///
/// // Search
/// let results = coll.search(&vec![0.1; 384], 10)?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
pub struct Database {
    /// Database configuration
    config: DatabaseConfig,
    /// Storage engine
    storage: Option<StorageEngine>,
    /// Collections (thread-safe)
    state: Arc<RwLock<DatabaseState>>,
    /// Whether there are unsaved changes (AtomicBool for lock-free access)
    dirty: AtomicBool,
    /// Modification generation counter for race-free dirty tracking.
    /// Incremented on every modification. Used during save to detect
    /// concurrent modifications and avoid clearing dirty flag prematurely.
    modification_gen: AtomicU64,
    /// Last saved generation. If modification_gen > saved_gen, database is dirty.
    saved_gen: AtomicU64,
    /// Optional adaptive tuner for online index-tuning feedback
    adaptive_tuner: Option<Arc<AdaptiveTuner>>,
}

impl Database {
    /// Open or create a database at the given path.
    ///
    /// This creates a new database file if it doesn't exist, or opens an
    /// existing one. The database uses a single-file storage format that
    /// can be easily backed up or distributed.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file (typically ending in `.needle`)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file exists but is corrupted or invalid
    /// - The parent directory doesn't exist
    /// - Permission denied (file or directory is read-only)
    /// - I/O error during file operations
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use needle::Database;
    ///
    /// let db = Database::open("my_vectors.needle")?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = DatabaseConfig::new(path.as_ref());
        Self::open_with_config(config)
    }

    /// Open or create a database with custom configuration.
    ///
    /// This provides more control over database behavior through the
    /// [`DatabaseConfig`] struct.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `create_if_missing` is false and the database doesn't exist
    /// - The file exists but is corrupted (checksum mismatch)
    /// - The file exists but contains invalid JSON data
    /// - I/O error during file operations
    #[instrument(skip(config), fields(path = ?config.path))]
    pub fn open_with_config(config: DatabaseConfig) -> Result<Self> {
        let exists = config.path.exists();
        info!(exists = exists, "Opening database");

        if !exists && !config.create_if_missing {
            warn!("Database not found and create_if_missing is false");
            return Err(NeedleError::InvalidDatabase(format!(
                "Database not found: {:?}",
                config.path
            )));
        }

        let mut storage = StorageEngine::open(&config.path)?;

        // Load existing state or create new
        let state = if exists {
            // Read state from file if metadata exists
            let header = storage.header().clone();
            if header.metadata_offset >= HEADER_SIZE as u64 {
                // Prefer state_size from header if available, otherwise compute from file size
                let state_len = if header.state_size > 0 {
                    header.state_size as usize
                } else {
                    (storage.file_size()? - header.metadata_offset) as usize
                };

                if state_len > 0 {
                    let state_bytes = storage.read_at(header.metadata_offset, state_len)?;

                    // Verify state checksum if available (state_checksum > 0 means it was set)
                    if header.state_checksum > 0 {
                        let computed_checksum = crc32(&state_bytes);
                        if computed_checksum != header.state_checksum {
                            warn!(
                                expected = header.state_checksum,
                                computed = computed_checksum,
                                "Checksum mismatch detected - database may be corrupted"
                            );
                            return Err(NeedleError::Corruption(
                                "State data checksum mismatch. Database file may be corrupted."
                                    .into(),
                            ));
                        }
                    }

                    serde_json::from_slice(&state_bytes).map_err(|e| {
                        warn!(error = %e, "Failed to deserialize database state");
                        NeedleError::Corruption(format!(
                            "Failed to deserialize database state: {}. Database file may be corrupted.",
                            e
                        ))
                    })?
                } else {
                    DatabaseState::default()
                }
            } else {
                DatabaseState::default()
            }
        } else {
            DatabaseState::default()
        };

        let collection_count = state.collections.len();
        let total_vectors: usize = state.collections.values().map(|c| c.len()).sum();
        info!(
            collections = collection_count,
            vectors = total_vectors,
            "Database opened successfully"
        );

        Ok(Self {
            config,
            storage: Some(storage),
            state: Arc::new(RwLock::new(state)),
            dirty: AtomicBool::new(false),
            modification_gen: AtomicU64::new(0),
            saved_gen: AtomicU64::new(0),
            adaptive_tuner: None,
        })
    }

    /// Create an in-memory database (not persisted).
    ///
    /// In-memory databases are useful for:
    /// - Testing without file I/O
    /// - Temporary caches that don't need persistence
    /// - Prototyping and experimentation
    ///
    /// Calling [`save()`](Self::save) on an in-memory database is a no-op.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("test", 128).unwrap();
    /// // Data is lost when `db` is dropped
    /// ```
    pub fn in_memory() -> Self {
        Self {
            config: DatabaseConfig::default(),
            storage: None,
            state: Arc::new(RwLock::new(DatabaseState::default())),
            dirty: AtomicBool::new(false),
            modification_gen: AtomicU64::new(0),
            saved_gen: AtomicU64::new(0),
            adaptive_tuner: None,
        }
    }

    /// Attach an adaptive tuner for online index-parameter learning.
    /// After each search, the tuner receives latency feedback.
    pub fn set_adaptive_tuner(&mut self, tuner: Arc<AdaptiveTuner>) {
        self.adaptive_tuner = Some(tuner);
    }

    /// Get a reference to the adaptive tuner, if one is attached.
    pub fn adaptive_tuner(&self) -> Option<&Arc<AdaptiveTuner>> {
        self.adaptive_tuner.as_ref()
    }

    /// Get the database file path.
    ///
    /// Returns `Some(&Path)` for file-backed databases, `None` for in-memory databases.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let memory_db = Database::in_memory();
    /// assert!(memory_db.path().is_none());
    /// ```
    ///
    /// ```rust,no_run
    /// use needle::Database;
    /// use std::path::Path;
    ///
    /// let file_db = Database::open("vectors.needle")?;
    /// assert_eq!(file_db.path(), Some(Path::new("vectors.needle")));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn path(&self) -> Option<&Path> {
        if self.storage.is_some() {
            Some(&self.config.path)
        } else {
            None
        }
    }

    /// Create a new collection with default settings.
    ///
    /// Creates a new collection for storing vectors of the specified dimensionality.
    /// Uses cosine distance and default HNSW parameters (M=16, ef_construction=200).
    ///
    /// # Arguments
    ///
    /// * `name` - Unique name for the collection
    /// * `dimensions` - Number of dimensions for vectors (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionAlreadyExists`] if a collection with the
    /// same name already exists.
    ///
    /// # Panics
    ///
    /// Panics if `dimensions` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("embeddings", 384)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn create_collection(&self, name: impl Into<String>, dimensions: usize) -> Result<()> {
        self.create_collection_with_config(CollectionConfig::new(name, dimensions))
    }

    /// Create a new collection with custom configuration.
    ///
    /// Provides full control over collection settings including distance function
    /// and HNSW parameters.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionAlreadyExists`] if a collection with the
    /// same name already exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Database, CollectionConfig, DistanceFunction};
    ///
    /// let db = Database::in_memory();
    /// let config = CollectionConfig::new("embeddings", 384)
    ///     .with_distance(DistanceFunction::Euclidean)
    ///     .with_m(32)
    ///     .with_ef_construction(400);
    /// db.create_collection_with_config(config)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn create_collection_with_config(&self, config: CollectionConfig) -> Result<()> {
        let mut state = self.state.write();

        if state.collections.contains_key(&config.name) {
            debug!(collection = %config.name, "Collection already exists");
            return Err(NeedleError::CollectionAlreadyExists(config.name));
        }

        info!(
            collection = %config.name,
            dimensions = config.dimensions,
            "Creating collection"
        );
        let collection = Collection::new(config.clone());
        state.collections.insert(config.name, collection);

        self.mark_modified();
        Ok(())
    }

    /// Get a thread-safe reference to a collection.
    ///
    /// Returns a [`CollectionRef`] that provides safe concurrent access to
    /// the collection. Multiple threads can hold references simultaneously
    /// for read operations; write operations are serialized.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the collection to access
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if no collection with
    /// the given name exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 128)?;
    ///
    /// let collection = db.collection("docs")?;
    /// collection.insert("v1", &vec![0.1; 128], None)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn collection(&self, name: &str) -> Result<CollectionRef<'_>> {
        let state = self.state.read();
        // Check if name is a direct collection name
        if state.collections.contains_key(name) {
            return Ok(CollectionRef::new(self, name.to_string()));
        }
        // Check if name is an alias
        if let Some(canonical_name) = state.aliases.get(name) {
            if state.collections.contains_key(canonical_name) {
                return Ok(CollectionRef::new(self, canonical_name.clone()));
            }
        }
        Err(NeedleError::CollectionNotFound(name.to_string()))
    }

    /// List all collection names in the database.
    ///
    /// Returns the names of all collections in no particular order.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 128)?;
    /// db.create_collection("images", 512)?;
    ///
    /// let names = db.list_collections();
    /// assert!(names.contains(&"docs".to_string()));
    /// assert!(names.contains(&"images".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn list_collections(&self) -> Vec<String> {
        self.state.read().collections.keys().cloned().collect()
    }

    /// Drop a collection and all its data.
    ///
    /// Removes the collection and all vectors it contains. This operation
    /// is immediate but changes are not persisted until [`save()`](Self::save)
    /// is called.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the collection to drop
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the collection was dropped, `Ok(false)` if no
    /// collection with that name existed.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("temp", 64)?;
    ///
    /// assert!(db.delete_collection("temp")?);
    /// assert!(!db.delete_collection("nonexistent")?);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete_collection(&self, name: &str) -> Result<bool> {
        let mut state = self.state.write();

        // Check if any aliases point to this collection
        let aliases_pointing: Vec<String> = state
            .aliases
            .iter()
            .filter(|(_, target)| *target == name)
            .map(|(alias, _)| alias.clone())
            .collect();

        if !aliases_pointing.is_empty() {
            return Err(NeedleError::CollectionHasAliases(name.to_string()));
        }

        let removed = state.collections.remove(name).is_some();
        if removed {
            info!(collection = %name, "Collection deleted");
            self.mark_modified();
        } else {
            debug!(collection = %name, "Collection not found for deletion");
        }
        Ok(removed)
    }

    /// Delete a collection and all its data.
    #[deprecated(since = "0.2.0", note = "renamed to `delete_collection`")]
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        self.delete_collection(name)
    }

    /// Check if a collection exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 128)?;
    ///
    /// assert!(db.has_collection("docs"));
    /// assert!(!db.has_collection("nonexistent"));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn has_collection(&self, name: &str) -> bool {
        self.state.read().collections.contains_key(name)
    }

    /// Export the entire database (all collections) as a JSON string.
    ///
    /// The output format is a JSON object mapping collection names to arrays of
    /// `[id, vector, metadata]` tuples.
    pub fn export_all_json(&self) -> Result<String> {
        let mut result: serde_json::Map<String, Value> = serde_json::Map::new();
        let state = self.state.read();

        for (name, collection) in &state.collections {
            let mut entries = Vec::new();
            let ids: Vec<String> = collection.ids().map(|s| s.to_string()).collect();
            for id in &ids {
                if let Some((vec, meta)) = collection.get(id) {
                    entries.push(serde_json::json!({
                        "id": id,
                        "vector": vec,
                        "metadata": meta,
                    }));
                }
            }
            let col_meta = serde_json::json!({
                "dimensions": collection.dimensions(),
                "entries": entries,
            });
            result.insert(name.clone(), col_meta);
        }
        serde_json::to_string(&Value::Object(result))
            .map_err(|e| NeedleError::InvalidInput(format!("JSON serialization: {}", e)))
    }

    /// Import collections from a JSON string produced by `export_all_json`.
    ///
    /// Creates collections that don't exist and inserts all vectors. Existing
    /// vectors with the same ID are overwritten.
    pub fn import_all_json(&self, json: &str) -> Result<()> {
        let obj: serde_json::Map<String, Value> = serde_json::from_str(json)
            .map_err(|e| NeedleError::InvalidInput(format!("JSON parse: {}", e)))?;

        for (name, col_data) in &obj {
            let dimensions = col_data["dimensions"]
                .as_u64()
                .ok_or_else(|| NeedleError::InvalidInput("Missing dimensions".into()))?
                as usize;

            if !self.has_collection(name) {
                self.create_collection(name, dimensions)?;
            }

            let col = self.collection(name)?;
            if let Some(entries) = col_data["entries"].as_array() {
                for entry in entries {
                    let id = entry["id"]
                        .as_str()
                        .ok_or_else(|| NeedleError::InvalidInput("Missing id".into()))?;
                    let vector: Vec<f32> = entry["vector"]
                        .as_array()
                        .ok_or_else(|| NeedleError::InvalidInput("Missing vector".into()))?
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    let metadata = if entry["metadata"].is_null() {
                        None
                    } else {
                        Some(entry["metadata"].clone())
                    };
                    col.insert(id, &vector, metadata)?;
                }
            }
        }
        Ok(())
    }

    /// Create an alias for a collection.
    ///
    /// Aliases provide alternative names for collections, useful for blue-green
    /// deployments where you can switch the "production" alias from one collection
    /// to another atomically.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to create
    /// * `collection` - The target collection name
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::AliasAlreadyExists`] if the alias already exists.
    /// Returns [`NeedleError::CollectionNotFound`] if the target collection doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    ///
    /// // Create alias pointing to the new version
    /// db.create_alias("docs", "docs_v2")?;
    ///
    /// // Now we can access via alias
    /// let coll = db.collection("docs")?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn create_alias(&self, alias: &str, collection: &str) -> Result<()> {
        let mut state = self.state.write();

        // Check if collection exists
        if !state.collections.contains_key(collection) {
            return Err(NeedleError::CollectionNotFound(collection.to_string()));
        }

        // Check if alias already exists (as alias or collection name)
        if state.aliases.contains_key(alias) {
            return Err(NeedleError::AliasAlreadyExists(alias.to_string()));
        }
        if state.collections.contains_key(alias) {
            return Err(NeedleError::AliasAlreadyExists(format!(
                "{} (conflicts with collection name)",
                alias
            )));
        }

        info!(alias = %alias, collection = %collection, "Creating alias");
        state
            .aliases
            .insert(alias.to_string(), collection.to_string());
        self.mark_modified();
        Ok(())
    }

    /// Delete an alias.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to delete
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the alias was deleted, `Ok(false)` if no alias
    /// with that name existed.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    ///
    /// assert!(db.delete_alias("docs")?);
    /// assert!(!db.delete_alias("nonexistent")?);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete_alias(&self, alias: &str) -> Result<bool> {
        let mut state = self.state.write();
        let removed = state.aliases.remove(alias).is_some();
        if removed {
            info!(alias = %alias, "Alias deleted");
            self.mark_modified();
        } else {
            debug!(alias = %alias, "Alias not found for delete");
        }
        Ok(removed)
    }

    /// List all aliases.
    ///
    /// Returns a list of `(alias_name, collection_name)` tuples.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v1", 128)?;
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    /// db.create_alias("old_docs", "docs_v1")?;
    ///
    /// let aliases = db.list_aliases();
    /// assert_eq!(aliases.len(), 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn list_aliases(&self) -> Vec<(String, String)> {
        self.state
            .read()
            .aliases
            .iter()
            .map(|(alias, collection)| (alias.clone(), collection.clone()))
            .collect()
    }

    /// Get the canonical collection name for an alias.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to resolve
    ///
    /// # Returns
    ///
    /// Returns `Some(collection_name)` if the alias exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    ///
    /// assert_eq!(db.get_canonical_name("docs"), Some("docs_v2".to_string()));
    /// assert_eq!(db.get_canonical_name("nonexistent"), None);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn get_canonical_name(&self, alias: &str) -> Option<String> {
        self.state.read().aliases.get(alias).cloned()
    }

    /// Get all aliases that point to a specific collection.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection name to find aliases for
    ///
    /// # Returns
    ///
    /// A vector of alias names that reference the given collection.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    /// db.create_alias("production", "docs_v2")?;
    ///
    /// let aliases = db.aliases_for_collection("docs_v2");
    /// assert_eq!(aliases.len(), 2);
    /// assert!(aliases.contains(&"docs".to_string()));
    /// assert!(aliases.contains(&"production".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn aliases_for_collection(&self, collection: &str) -> Vec<String> {
        self.state
            .read()
            .aliases
            .iter()
            .filter(|(_, target)| *target == collection)
            .map(|(alias, _)| alias.clone())
            .collect()
    }

    /// Update an existing alias to point to a different collection.
    ///
    /// This is useful for blue-green deployments where you want to atomically
    /// switch an alias from one collection to another.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to update
    /// * `collection` - The new target collection name
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::AliasNotFound`] if the alias doesn't exist.
    /// Returns [`NeedleError::CollectionNotFound`] if the target collection doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v1", 128)?;
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("production", "docs_v1")?;
    ///
    /// // Switch production to v2
    /// db.update_alias("production", "docs_v2")?;
    ///
    /// assert_eq!(db.get_canonical_name("production"), Some("docs_v2".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn update_alias(&self, alias: &str, collection: &str) -> Result<()> {
        let mut state = self.state.write();

        // Check if collection exists
        if !state.collections.contains_key(collection) {
            return Err(NeedleError::CollectionNotFound(collection.to_string()));
        }

        // Check if alias exists
        if !state.aliases.contains_key(alias) {
            return Err(NeedleError::AliasNotFound(alias.to_string()));
        }

        info!(alias = %alias, collection = %collection, "Updating alias");
        state
            .aliases
            .insert(alias.to_string(), collection.to_string());
        self.mark_modified();
        Ok(())
    }

    /// Save changes to disk.
    ///
    /// Persists all collections and their data to the database file. Uses
    /// atomic writes to prevent corruption on crash. For in-memory databases,
    /// this is a no-op.
    ///
    /// Uses generation-based tracking to avoid race conditions where
    /// concurrent modifications could be marked as saved when they weren't.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The database file cannot be written (permission denied, disk full)
    /// - I/O error during write operations
    /// - The database is in-memory (returns `Ok(())` instead)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use needle::Database;
    ///
    /// let mut db = Database::open("data.needle")?;
    /// db.create_collection("docs", 128)?;
    ///
    /// let collection = db.collection("docs")?;
    /// collection.insert("v1", &vec![0.1; 128], None)?;
    ///
    /// // Persist changes to disk
    /// db.save()?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    #[instrument(skip(self))]
    pub fn save(&mut self) -> Result<()> {
        let storage = match &mut self.storage {
            Some(s) => s,
            None => {
                debug!("In-memory database, skipping save");
                return Ok(());
            }
        };

        // Capture the modification generation BEFORE serializing state.
        // This ensures we only mark as saved up to this point.
        let gen_at_save_start = self.modification_gen.load(Ordering::Acquire);
        debug!(generation = gen_at_save_start, "Starting save");

        let state = self.state.read();
        let state_bytes = serde_json::to_vec(&*state)?;

        // Build updated header
        let mut header = storage.header().clone();
        header.metadata_offset = HEADER_SIZE as u64;
        header.vector_count = state.collections.values().map(|c| c.len() as u64).sum();

        // Use atomic save to prevent corruption on crash
        storage.atomic_save(&header, &state_bytes)?;

        // Only update saved_gen if no concurrent modifications happened.
        // Use compare-and-swap loop to safely update.
        loop {
            let current_saved = self.saved_gen.load(Ordering::Acquire);
            // Only update if we're moving forward
            if gen_at_save_start <= current_saved {
                break; // Another save already covered our changes
            }
            match self.saved_gen.compare_exchange_weak(
                current_saved,
                gen_at_save_start,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue, // Retry
            }
        }

        // Clear dirty flag only if no modifications happened during save
        let current_mod_gen = self.modification_gen.load(Ordering::Acquire);
        if current_mod_gen == gen_at_save_start {
            self.dirty.store(false, Ordering::Release);
        }

        info!(
            vectors = header.vector_count,
            bytes = state_bytes.len(),
            "Database saved successfully"
        );

        Ok(())
    }

    /// Check if there are unsaved changes.
    ///
    /// Returns `true` if any modifications have been made since the last
    /// [`save()`](Self::save) call. For in-memory databases, always returns
    /// `false` after creation since there's nothing to persist.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use needle::Database;
    ///
    /// let mut db = Database::open("test.needle")?;
    /// assert!(!db.is_dirty()); // Just opened, no changes
    ///
    /// db.create_collection("docs", 128)?;
    /// assert!(db.is_dirty()); // Has unsaved changes
    ///
    /// db.save()?;
    /// assert!(!db.is_dirty()); // Changes saved
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn is_dirty(&self) -> bool {
        // Use generation counters for accurate dirty tracking
        let mod_gen = self.modification_gen.load(Ordering::Acquire);
        let saved_gen = self.saved_gen.load(Ordering::Acquire);
        mod_gen > saved_gen || self.dirty.load(Ordering::Acquire)
    }

    /// Mark the database as modified. Thread-safe and race-condition free.
    fn mark_modified(&self) {
        self.modification_gen.fetch_add(1, Ordering::Release);
        self.dirty.store(true, Ordering::Release);
    }

    /// Get total number of vectors across all collections.
    ///
    /// Returns the sum of active (non-deleted) vectors in all collections.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// db.create_collection("images", 4)?;
    ///
    /// let docs = db.collection("docs")?;
    /// let images = db.collection("images")?;
    ///
    /// docs.insert("d1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// images.insert("i1", &[0.0, 1.0, 0.0, 0.0], None)?;
    /// images.insert("i2", &[0.0, 0.0, 1.0, 0.0], None)?;
    ///
    /// assert_eq!(db.total_vectors(), 3);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn total_vectors(&self) -> usize {
        self.state
            .read()
            .collections
            .values()
            .map(|c| c.len())
            .sum()
    }

    // Internal methods for CollectionRef

    fn insert_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert(id, vector, metadata)?;
        self.mark_modified();
        Ok(())
    }

    fn insert_vec_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_vec(id, vector, metadata)?;
        self.mark_modified();
        Ok(())
    }

    fn insert_with_ttl_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_with_ttl(id, vector, metadata, ttl_seconds)?;
        self.mark_modified();
        Ok(())
    }

    fn insert_vec_with_ttl_internal(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.insert_vec_with_ttl(id, vector, metadata, ttl_seconds)?;
        self.mark_modified();
        Ok(())
    }

    fn update_internal(
        &self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.update(id, vector, metadata)?;
        self.mark_modified();
        Ok(())
    }

    fn search_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let start = std::time::Instant::now();

        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let results = coll.search(query, k)?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Feed latency data to adaptive tuner if attached
        if let Some(tuner) = &self.adaptive_tuner {
            let stats = coll.stats();
            let obs = WorkloadObservation {
                vector_count: stats.vector_count,
                dimensions: stats.dimensions,
                qps: 0.0,
                insert_rate: 0.0,
                avg_latency_ms: latency_ms,
                measured_recall: 1.0, // unknown in production; assume best-case
                memory_bytes: stats.total_memory_bytes as u64,
                current_index: RecommendedIndex::Hnsw,
                current_config: None,
            };
            tuner.feedback(&obs, 1.0, latency_ms);
        }

        Ok(results)
    }

    fn search_with_filter_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_with_filter(query, k, filter)
    }

    fn search_with_options_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        distance_override: Option<crate::DistanceFunction>,
        filter: Option<&Filter>,
        post_filter: Option<&Filter>,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let mut builder = coll.search_builder(query).k(k);

        if let Some(dist) = distance_override {
            builder = builder.distance(dist);
        }

        if let Some(f) = filter {
            builder = builder.filter(f);
        }

        if let Some(pf) = post_filter {
            builder = builder
                .post_filter(pf)
                .post_filter_factor(post_filter_factor);
        }

        builder.execute()
    }

    fn search_explain_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_explain(query, k)
    }

    fn search_with_filter_explain_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_with_filter_explain(query, k, filter)
    }

    fn search_with_post_filter_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
        pre_filter: Option<&Filter>,
        post_filter: &Filter,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let mut builder = coll
            .search_builder(query)
            .k(k)
            .post_filter(post_filter)
            .post_filter_factor(post_filter_factor);

        if let Some(pf) = pre_filter {
            builder = builder.filter(pf);
        }

        builder.execute()
    }

    fn search_radius_internal(
        &self,
        collection: &str,
        query: &[f32],
        max_distance: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_radius(query, max_distance, limit)
    }

    fn search_radius_with_filter_internal(
        &self,
        collection: &str,
        query: &[f32],
        max_distance: f32,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_radius_with_filter(query, max_distance, limit, filter)
    }

    fn get_internal(&self, collection: &str, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        let state = self.state.read();
        let coll = state.collections.get(collection)?;
        let (vec, meta) = coll.get(id)?;
        Some((vec.to_vec(), meta.cloned()))
    }

    fn delete_internal(&self, collection: &str, id: &str) -> Result<bool> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let deleted = coll.delete(id)?;
        if deleted {
            self.mark_modified();
        }
        Ok(deleted)
    }

    fn collection_len(&self, collection: &str) -> usize {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.len())
            .unwrap_or(0)
    }

    fn collection_dimensions(&self, collection: &str) -> Option<usize> {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.dimensions())
    }

    fn collection_stats_internal(
        &self,
        collection: &str,
    ) -> Result<crate::collection::CollectionStats> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        Ok(coll.stats())
    }

    fn export_internal(&self, collection: &str) -> Result<Vec<ExportEntry>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        Ok(coll
            .iter()
            .map(|(id, vec, meta)| (id.to_string(), vec.to_vec(), meta.cloned()))
            .collect())
    }

    fn ids_internal(&self, collection: &str) -> Result<Vec<String>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        Ok(coll.ids().map(|s| s.to_string()).collect())
    }

    fn compact_internal(&self, collection: &str) -> Result<usize> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let deleted = coll.compact()?;
        if deleted > 0 {
            self.mark_modified();
        }
        Ok(deleted)
    }

    fn needs_compaction_internal(&self, collection: &str, threshold: f64) -> bool {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.needs_compaction(threshold))
            .unwrap_or(false)
    }

    // ============ TTL Internal Methods ============

    fn expire_vectors_internal(&self, collection: &str) -> Result<usize> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        let count = coll.expire_vectors()?;
        if count > 0 {
            self.mark_modified();
        }
        Ok(count)
    }

    fn needs_expiration_sweep_internal(&self, collection: &str, threshold: f64) -> bool {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.needs_expiration_sweep(threshold))
            .unwrap_or(false)
    }

    fn ttl_stats_internal(&self, collection: &str) -> (usize, usize, Option<u64>, Option<u64>) {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.ttl_stats())
            .unwrap_or((0, 0, None, None))
    }

    fn get_ttl_internal(&self, collection: &str, id: &str) -> Option<u64> {
        self.state
            .read()
            .collections
            .get(collection)
            .and_then(|c| c.get_ttl(id))
    }

    fn set_ttl_internal(&self, collection: &str, id: &str, ttl_seconds: Option<u64>) -> Result<()> {
        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.set_ttl(id, ttl_seconds)?;
        self.mark_modified();
        Ok(())
    }

    fn count_internal(&self, collection: &str, filter: Option<&Filter>) -> Result<usize> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        Ok(coll.count(filter))
    }

    fn deleted_count_internal(&self, collection: &str) -> usize {
        self.state
            .read()
            .collections
            .get(collection)
            .map(|c| c.deleted_count())
            .unwrap_or(0)
    }

    fn search_ids_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search_ids(query, k)
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Warn about unsaved changes but do NOT perform I/O in Drop.
        // Performing I/O in Drop is unexpected behavior and can cause issues:
        // - Panics during Drop are hard to handle
        // - I/O errors in Drop cannot be properly propagated
        // - May cause deadlocks in async contexts
        // Users should explicitly call save() before dropping.
        if self.is_dirty() {
            eprintln!(
                "needle: warning: database dropped with unsaved changes. \
                 Call save() explicitly before dropping to persist data."
            );
        }
    }
}

pub mod collection_ref;
pub use collection_ref::{CollectionRef, SearchParams};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_database_in_memory() {
        let db = Database::in_memory();

        db.create_collection("documents", 128).unwrap();

        let coll = db.collection("documents").unwrap();
        let vec = random_vector(128);
        coll.insert("doc1", &vec, Some(json!({"title": "Test"})))
            .unwrap();

        assert_eq!(coll.len(), 1);
    }

    #[test]
    fn test_database_persistence() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.needle");

        // Create and populate database
        {
            let mut db = Database::open(&path).unwrap();
            db.create_collection("documents", 64).unwrap();

            let coll = db.collection("documents").unwrap();
            for i in 0..10 {
                let vec = random_vector(64);
                coll.insert(format!("doc{}", i), &vec, Some(json!({"index": i})))
                    .unwrap();
            }

            db.save().unwrap();
        }

        // Reopen and verify
        {
            let db = Database::open(&path).unwrap();
            let coll = db.collection("documents").unwrap();
            assert_eq!(coll.len(), 10);

            let (_, meta) = coll.get("doc5").unwrap();
            assert_eq!(meta.unwrap()["index"], 5);
        }
    }

    #[test]
    fn test_database_multiple_collections() {
        let db = Database::in_memory();

        db.create_collection("images", 512).unwrap();
        db.create_collection("text", 384).unwrap();

        let collections = db.list_collections();
        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"images".to_string()));
        assert!(collections.contains(&"text".to_string()));
    }

    #[test]
    fn test_database_drop_collection() {
        let db = Database::in_memory();

        db.create_collection("temp", 128).unwrap();
        assert!(db.has_collection("temp"));

        db.drop_collection("temp").unwrap();
        assert!(!db.has_collection("temp"));
    }

    #[test]
    fn test_database_search() {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();

        let coll = db.collection("test").unwrap();

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(32);
            coll.insert(format!("doc{}", i), &vec, Some(json!({"index": i})))
                .unwrap();
        }

        // Search
        let query = random_vector(32);
        let results = coll.search(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_collection_not_found() {
        let db = Database::in_memory();
        let result = db.collection("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_collection_already_exists() {
        let db = Database::in_memory();
        db.create_collection("test", 128).unwrap();

        let result = db.create_collection("test", 128);
        assert!(result.is_err());
    }

    // ========== Alias Tests ==========

    #[test]
    fn test_alias_create_and_resolve() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();

        // Create alias
        db.create_alias("prod", "v1").unwrap();

        // Resolve alias
        assert_eq!(db.get_canonical_name("prod"), Some("v1".to_string()));

        // Access collection via alias
        let coll = db.collection("prod").unwrap();
        assert_eq!(coll.count(None).unwrap(), 0);
    }

    #[test]
    fn test_alias_list() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();
        db.create_collection("v2", 128).unwrap();

        db.create_alias("prod", "v1").unwrap();
        db.create_alias("staging", "v2").unwrap();

        let aliases = db.list_aliases();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&("prod".to_string(), "v1".to_string())));
        assert!(aliases.contains(&("staging".to_string(), "v2".to_string())));
    }

    #[test]
    fn test_alias_delete() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();
        db.create_alias("prod", "v1").unwrap();

        assert!(db.get_canonical_name("prod").is_some());

        // Delete alias
        let deleted = db.delete_alias("prod").unwrap();
        assert!(deleted);

        // Alias no longer exists
        assert!(db.get_canonical_name("prod").is_none());

        // Deleting again returns false
        let deleted = db.delete_alias("prod").unwrap();
        assert!(!deleted);
    }

    #[test]
    fn test_alias_prevents_collection_drop() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();
        db.create_alias("prod", "v1").unwrap();

        // Trying to drop collection with alias should fail
        let result = db.drop_collection("v1");
        assert!(result.is_err());

        // Delete alias first
        db.delete_alias("prod").unwrap();

        // Now drop should succeed
        let dropped = db.drop_collection("v1").unwrap();
        assert!(dropped);
    }

    #[test]
    fn test_alias_update() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();
        db.create_collection("v2", 128).unwrap();

        db.create_alias("prod", "v1").unwrap();
        assert_eq!(db.get_canonical_name("prod"), Some("v1".to_string()));

        // Update alias to point to v2
        db.update_alias("prod", "v2").unwrap();
        assert_eq!(db.get_canonical_name("prod"), Some("v2".to_string()));
    }

    #[test]
    fn test_aliases_for_collection() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();

        db.create_alias("prod", "v1").unwrap();
        db.create_alias("latest", "v1").unwrap();

        let aliases = db.aliases_for_collection("v1");
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&"prod".to_string()));
        assert!(aliases.contains(&"latest".to_string()));
    }

    #[test]
    fn test_alias_already_exists() {
        let db = Database::in_memory();
        db.create_collection("v1", 128).unwrap();

        db.create_alias("prod", "v1").unwrap();

        // Creating same alias again should fail
        let result = db.create_alias("prod", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_alias_to_nonexistent_collection() {
        let db = Database::in_memory();

        // Creating alias to nonexistent collection should fail
        let result = db.create_alias("prod", "nonexistent");
        assert!(result.is_err());
    }

    // ========== TTL Tests ==========

    #[test]
    fn test_ttl_insert_and_stats() {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();

        let coll = db.collection("test").unwrap();

        // Insert with TTL
        let vec = random_vector(32);
        coll.insert_with_ttl("doc1", &vec, None, Some(3600))
            .unwrap();

        // Insert without TTL
        let vec2 = random_vector(32);
        coll.insert("doc2", &vec2, None).unwrap();

        let (total_with_ttl, _expired, _earliest, _latest) = coll.ttl_stats();
        assert_eq!(total_with_ttl, 1);
        assert_eq!(coll.count(None).unwrap(), 2);
    }

    #[test]
    fn test_ttl_get_and_set() {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();

        let coll = db.collection("test").unwrap();

        let vec = random_vector(32);
        coll.insert("doc1", &vec, None).unwrap();

        // Initially no TTL
        assert!(coll.get_ttl("doc1").is_none());

        // Set TTL
        coll.set_ttl("doc1", Some(3600)).unwrap();
        assert!(coll.get_ttl("doc1").is_some());

        // Remove TTL
        coll.set_ttl("doc1", None).unwrap();
        assert!(coll.get_ttl("doc1").is_none());
    }

    #[test]
    fn test_ttl_expiration_sweep() {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();

        let coll = db.collection("test").unwrap();

        // Insert with very short TTL (already expired - TTL of 0 means expire immediately)
        let vec = random_vector(32);
        // Use a TTL in the past by setting expiration directly
        coll.insert_with_ttl("doc1", &vec, None, Some(0)).unwrap();

        // Wait a tiny bit and expire - but since TTL=0 sets expiration to now,
        // the vector should be expired immediately on the next sweep
        std::thread::sleep(std::time::Duration::from_millis(10));

        let expired = coll.expire_vectors().unwrap();
        assert_eq!(expired, 1);

        // Vector should be gone
        assert!(coll.get("doc1").is_none());
    }

    #[test]
    fn test_ttl_lazy_expiration_in_search() {
        use crate::CollectionConfig;

        let db = Database::in_memory();

        // Create collection with lazy expiration enabled
        let config = CollectionConfig::new("test", 32).with_lazy_expiration(true);
        db.create_collection_with_config(config).unwrap();

        let coll = db.collection("test").unwrap();

        // Insert expired vector
        let vec1 = random_vector(32);
        coll.insert_with_ttl("expired", &vec1, None, Some(0))
            .unwrap();

        // Insert non-expired vector
        let vec2 = random_vector(32);
        coll.insert("valid", &vec2, None).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));

        // Search should only return the valid vector (lazy expiration filters out expired)
        let results = coll.search(&vec1, 10).unwrap();

        // The expired vector should not appear in results
        assert!(results.iter().all(|r| r.id != "expired"));
    }

    // ========== Distance Override Tests ==========

    #[test]
    fn test_distance_override_search() {
        use crate::DistanceFunction;

        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();

        let coll = db.collection("test").unwrap();

        // Insert some vectors
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        coll.insert("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        coll.insert("c", &[0.5, 0.5, 0.0, 0.0], None).unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Search with default distance (cosine)
        let results_default = coll.search(&query, 3).unwrap();
        assert_eq!(results_default.len(), 3);
        assert_eq!(results_default[0].id, "a"); // Exact match

        // Search with euclidean distance override (triggers brute-force)
        let results_euclidean = coll
            .search_with_options(&query, 3, Some(DistanceFunction::Euclidean), None, None, 3)
            .unwrap();
        assert_eq!(results_euclidean.len(), 3);
        assert_eq!(results_euclidean[0].id, "a"); // Still exact match
        assert!(results_euclidean[0].distance < 0.001); // Euclidean distance to itself is 0
    }

    #[test]
    fn test_distance_override_produces_correct_ordering() {
        use crate::DistanceFunction;

        let db = Database::in_memory();

        // Create collection with cosine distance
        let config =
            crate::CollectionConfig::new("test", 3).with_distance(DistanceFunction::Cosine);
        db.create_collection_with_config(config).unwrap();

        let coll = db.collection("test").unwrap();

        // Insert vectors with different magnitudes but same direction
        coll.insert("small", &[1.0, 0.0, 0.0], None).unwrap();
        coll.insert("large", &[10.0, 0.0, 0.0], None).unwrap();

        let query = vec![5.0, 0.0, 0.0];

        // With cosine, both should have same distance (direction matters, not magnitude)
        let cosine_results = coll.search(&query, 2).unwrap();
        let d1 = cosine_results[0].distance;
        let d2 = cosine_results[1].distance;
        assert!(
            (d1 - d2).abs() < 0.001,
            "Cosine distances should be equal for same direction"
        );

        // With euclidean override, magnitude matters
        let euclidean_results = coll
            .search_with_options(&query, 2, Some(DistanceFunction::Euclidean), None, None, 3)
            .unwrap();

        // "small" (1.0) is closer to query (5.0) than "large" (10.0) in euclidean
        // distance(5,1) = 4, distance(5,10) = 5
        assert_eq!(euclidean_results[0].id, "small");
        assert_eq!(euclidean_results[1].id, "large");
    }

    #[test]
    fn test_distance_override_same_as_index_uses_hnsw() {
        use crate::DistanceFunction;

        let db = Database::in_memory();

        // Create collection with euclidean distance
        let config =
            crate::CollectionConfig::new("test", 32).with_distance(DistanceFunction::Euclidean);
        db.create_collection_with_config(config).unwrap();

        let coll = db.collection("test").unwrap();

        for i in 0..100 {
            let vec = random_vector(32);
            coll.insert(format!("doc{}", i), &vec, None).unwrap();
        }

        let query = random_vector(32);

        // Override with same distance as index - should use HNSW (not brute force)
        let results = coll
            .search_with_options(&query, 10, Some(DistanceFunction::Euclidean), None, None, 3)
            .unwrap();

        assert_eq!(results.len(), 10);
    }
}
