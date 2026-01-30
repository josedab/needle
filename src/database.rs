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
use crate::tuning::{AdaptiveTuner, WorkloadObservation, RecommendedIndex};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn, instrument};

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
                                "State data checksum mismatch. Database file may be corrupted.".into(),
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
            return Ok(CollectionRef {
                db: self,
                name: name.to_string(),
            });
        }
        // Check if name is an alias
        if let Some(canonical_name) = state.aliases.get(name) {
            if state.collections.contains_key(canonical_name) {
                return Ok(CollectionRef {
                    db: self,
                    name: canonical_name.clone(),
                });
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
    /// assert!(db.drop_collection("temp")?);
    /// assert!(!db.drop_collection("nonexistent")?);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
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
            info!(collection = %name, "Collection dropped");
            self.mark_modified();
        } else {
            debug!(collection = %name, "Collection not found for drop");
        }
        Ok(removed)
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
        state.aliases.insert(alias.to_string(), collection.to_string());
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
        state.aliases.insert(alias.to_string(), collection.to_string());
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
            builder = builder.post_filter(pf).post_filter_factor(post_filter_factor);
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

        let mut builder = coll.search_builder(query)
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

/// A thread-safe reference to a collection for concurrent access.
///
/// `CollectionRef` is the primary way to interact with collections when using
/// `Database`. It wraps collection operations with proper locking, enabling
/// safe concurrent read and write access from multiple threads.
///
/// # Thread Safety
///
/// - Multiple readers can access the collection simultaneously
/// - Writers get exclusive access (blocking other readers and writers)
/// - All operations are atomic at the method level
///
/// # Obtaining a CollectionRef
///
/// Use [`Database::collection()`] to get a reference:
///
/// ```
/// use needle::Database;
///
/// let db = Database::in_memory();
/// db.create_collection("embeddings", 128)?;
///
/// let collection = db.collection("embeddings")?;
/// // Use collection for insert, search, delete operations
/// # Ok::<(), needle::NeedleError>(())
/// ```
///
/// # Example: Concurrent Access
///
/// ```
/// use std::sync::Arc;
/// use std::thread;
/// use needle::Database;
///
/// let db = Arc::new(Database::in_memory());
/// db.create_collection("docs", 4).unwrap();
///
/// // Insert from main thread
/// let coll = db.collection("docs").unwrap();
/// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
///
/// // Search from multiple threads
/// let handles: Vec<_> = (0..4).map(|_| {
///     let db = Arc::clone(&db);
///     thread::spawn(move || {
///         let coll = db.collection("docs").unwrap();
///         coll.search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap()
///     })
/// }).collect();
///
/// for h in handles {
///     let results = h.join().unwrap();
///     assert!(!results.is_empty());
/// }
/// ```
pub struct CollectionRef<'a> {
    db: &'a Database,
    name: String,
}

impl<'a> CollectionRef<'a> {
    /// Get the collection name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of vectors
    pub fn len(&self) -> usize {
        self.db.collection_len(&self.name)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the vector dimensions
    pub fn dimensions(&self) -> Option<usize> {
        self.db.collection_dimensions(&self.name)
    }

    /// Get collection statistics (vector count, dimensions, memory usage, etc.)
    pub fn stats(&self) -> Result<crate::collection::CollectionStats> {
        self.db.collection_stats_internal(&self.name)
    }

    /// Insert a vector into the collection.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The vector data (must match collection dimensions)
    /// * `metadata` - Optional JSON metadata to associate with the vector
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::DimensionMismatch`] if the vector length doesn't match
    /// the collection's configured dimensions.
    ///
    /// Returns [`NeedleError::InvalidVector`] if the vector contains NaN or Infinity values.
    ///
    /// Returns [`NeedleError::DuplicateId`] if a vector with this ID already exists.
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_internal(&self.name, id, vector, metadata)
    }

    /// Insert a vector, taking ownership (more efficient when you have a Vec).
    ///
    /// This variant avoids an allocation when you already have a `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Same as [`insert`](Self::insert).
    pub fn insert_vec(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_vec_internal(&self.name, id, vector, metadata)
    }

    /// Insert a vector with explicit TTL (time-to-live).
    ///
    /// The vector will automatically expire after the specified TTL.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the vector
    /// * `vector` - The vector data
    /// * `metadata` - Optional JSON metadata
    /// * `ttl_seconds` - TTL in seconds; if `None`, uses collection default
    pub fn insert_with_ttl(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        self.db.insert_with_ttl_internal(&self.name, id, vector, metadata, ttl_seconds)
    }

    /// Insert a vector with TTL, taking ownership (more efficient).
    pub fn insert_vec_with_ttl(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        self.db.insert_vec_with_ttl_internal(&self.name, id, vector, metadata, ttl_seconds)
    }

    /// Search for the k most similar vectors to the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Number of nearest neighbors to return
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::DimensionMismatch`] if the query length doesn't match
    /// the collection's configured dimensions.
    ///
    /// Returns [`NeedleError::InvalidVector`] if the query contains NaN or Infinity values.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.db.search_internal(&self.name, query, k)
    }

    /// Search for similar vectors with metadata filtering.
    ///
    /// Applies the filter before searching, potentially reducing the search space.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::DimensionMismatch`] if the query length doesn't match
    /// the collection's configured dimensions.
    ///
    /// Returns [`NeedleError::InvalidVector`] if the query contains NaN or Infinity values.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_with_filter_internal(&self.name, query, k, filter)
    }

    /// Search with full options including distance override, filters, and post-filter.
    ///
    /// This method provides access to all search options without using the builder pattern.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of results to return
    /// * `distance_override` - Optional distance function override (falls back to brute-force)
    /// * `filter` - Optional pre-filter (applied during ANN search)
    /// * `post_filter` - Optional post-filter (applied after ANN search)
    /// * `post_filter_factor` - Over-fetch factor for post-filtering (default: 3)
    pub fn search_with_options(
        &self,
        query: &[f32],
        k: usize,
        distance_override: Option<crate::DistanceFunction>,
        filter: Option<&Filter>,
        post_filter: Option<&Filter>,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db.search_with_options_internal(
            &self.name,
            query,
            k,
            distance_override,
            filter,
            post_filter,
            post_filter_factor,
        )
    }

    /// Search with detailed query execution profiling.
    ///
    /// Returns both the search results and a [`SearchExplain`](crate::SearchExplain)
    /// struct containing detailed timing and statistics.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let collection = db.collection("docs")?;
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// let (results, explain) = collection.search_explain(&[1.0, 0.0, 0.0, 0.0], 10)?;
    /// println!("Search took {}μs, visited {} nodes",
    ///          explain.total_time_us, explain.hnsw_stats.visited_nodes);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_explain(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        self.db.search_explain_internal(&self.name, query, k)
    }

    /// Search with metadata filter and detailed profiling.
    ///
    /// Combines filtered search with query execution profiling.
    pub fn search_with_filter_explain(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<(Vec<SearchResult>, crate::collection::SearchExplain)> {
        self.db
            .search_with_filter_explain_internal(&self.name, query, k, filter)
    }

    /// Find all vectors within a given distance from the query.
    ///
    /// Unlike top-k search which returns exactly k results regardless of distance,
    /// range queries return all vectors within `max_distance` from the query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let collection = db.collection("docs")?;
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// // Find all vectors within distance 0.5
    /// let results = collection.search_radius(&[1.0, 0.0, 0.0, 0.0], 0.5, 100)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_radius(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_radius_internal(&self.name, query, max_distance, limit)
    }

    /// Find all vectors within a given distance with metadata filtering.
    ///
    /// Combines range queries with metadata pre-filtering.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return
    /// * `filter` - MongoDB-style filter for metadata
    pub fn search_radius_with_filter(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_radius_with_filter_internal(&self.name, query, max_distance, limit, filter)
    }

    /// Retrieve a vector and its metadata by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The external ID of the vector to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some((vector, metadata))` if found, `None` otherwise.
    /// The vector data is cloned for thread-safety.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"title": "Hello"})))?;
    ///
    /// if let Some((vector, metadata)) = coll.get("v1") {
    ///     assert_eq!(vector, vec![1.0, 2.0, 3.0, 4.0]);
    ///     assert!(metadata.is_some());
    /// }
    ///
    /// assert!(coll.get("nonexistent").is_none());
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        self.db.get_internal(&self.name, id)
    }

    /// Delete a vector by its ID.
    ///
    /// Removes the vector from the index. Storage space is not immediately
    /// reclaimed; call [`compact()`](Self::compact) after many deletions.
    ///
    /// # Arguments
    ///
    /// * `id` - The external ID of the vector to delete
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the vector was deleted, `Ok(false)` if no vector
    /// with that ID existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(coll.delete("v1")?);       // Returns true
    /// assert!(!coll.delete("v1")?);      // Already deleted, returns false
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete(&self, id: &str) -> Result<bool> {
        self.db.delete_internal(&self.name, id)
    }

    /// Export all vectors from the collection.
    ///
    /// Returns all vectors with their IDs and metadata, useful for backup
    /// or migration purposes.
    ///
    /// # Returns
    ///
    /// A vector of `(id, vector, metadata)` tuples for all vectors in the collection.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"x": 1})))?;
    /// coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// let exported = coll.export_all()?;
    /// assert_eq!(exported.len(), 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn export_all(&self) -> Result<Vec<ExportEntry>> {
        self.db.export_internal(&self.name)
    }

    /// Get all vector IDs in the collection.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("doc1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// coll.insert("doc2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// let ids = coll.ids()?;
    /// assert!(ids.contains(&"doc1".to_string()));
    /// assert!(ids.contains(&"doc2".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn ids(&self) -> Result<Vec<String>> {
        self.db.ids_internal(&self.name)
    }

    /// Compact the collection, removing deleted vectors from storage.
    ///
    /// After many deletions, storage space is not immediately reclaimed.
    /// Calling `compact()` rebuilds internal structures to reclaim space.
    ///
    /// # Returns
    ///
    /// The number of deleted vectors that were removed from storage.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    ///
    /// // Insert and delete vectors
    /// for i in 0..100 {
    ///     coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
    /// }
    /// for i in 0..50 {
    ///     coll.delete(&format!("v{}", i))?;
    /// }
    ///
    /// // Reclaim storage space
    /// let removed = coll.compact()?;
    /// assert_eq!(removed, 50);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn compact(&self) -> Result<usize> {
        self.db.compact_internal(&self.name)
    }

    /// Check if the collection needs compaction.
    ///
    /// Returns `true` if the ratio of deleted vectors to total vectors exceeds
    /// the given threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The deleted/total ratio above which compaction is needed (0.0-1.0)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    ///
    /// for i in 0..10 {
    ///     coll.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
    /// }
    /// for i in 0..8 {
    ///     coll.delete(&format!("v{}", i))?;
    /// }
    ///
    /// // 8 deleted out of 10 = 80% deleted, so threshold 0.5 should trigger
    /// assert!(coll.needs_compaction(0.5));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.db.needs_compaction_internal(&self.name, threshold)
    }

    // ============ TTL Methods ============

    /// Sweep and delete all expired vectors.
    ///
    /// Returns the number of vectors that were expired and deleted.
    pub fn expire_vectors(&self) -> Result<usize> {
        self.db.expire_vectors_internal(&self.name)
    }

    /// Check if an expiration sweep is needed based on a threshold.
    ///
    /// Returns true if the ratio of expired vectors to total vectors
    /// exceeds the given threshold (0.0-1.0).
    pub fn needs_expiration_sweep(&self, threshold: f64) -> bool {
        self.db.needs_expiration_sweep_internal(&self.name, threshold)
    }

    /// Get TTL statistics for the collection.
    ///
    /// Returns (total_with_ttl, expired_count, earliest_expiration, latest_expiration).
    pub fn ttl_stats(&self) -> (usize, usize, Option<u64>, Option<u64>) {
        self.db.ttl_stats_internal(&self.name)
    }

    /// Get the expiration timestamp for a vector by external ID.
    pub fn get_ttl(&self, id: &str) -> Option<u64> {
        self.db.get_ttl_internal(&self.name, id)
    }

    /// Set or update the TTL for an existing vector.
    pub fn set_ttl(&self, id: &str, ttl_seconds: Option<u64>) -> Result<()> {
        self.db.set_ttl_internal(&self.name, id, ttl_seconds)
    }

    /// Count vectors in the collection, optionally matching a filter.
    ///
    /// # Arguments
    ///
    /// * `filter` - Optional metadata filter; if `None`, counts all vectors
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if the collection no longer exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Database, Filter};
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"})))?;
    /// coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"type": "b"})))?;
    /// coll.insert("v3", &[0.0, 0.0, 1.0, 0.0], Some(json!({"type": "a"})))?;
    ///
    /// assert_eq!(coll.count(None)?, 3);
    ///
    /// let filter = Filter::eq("type", "a");
    /// assert_eq!(coll.count(Some(&filter))?, 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn count(&self, filter: Option<&Filter>) -> Result<usize> {
        self.db.count_internal(&self.name, filter)
    }

    /// Get the number of deleted vectors pending compaction.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// assert_eq!(coll.deleted_count(), 0);
    ///
    /// coll.delete("v1")?;
    /// assert_eq!(coll.deleted_count(), 1);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn deleted_count(&self) -> usize {
        self.db.deleted_count_internal(&self.name)
    }

    /// Search and return only IDs with distances (faster than full search).
    ///
    /// Skips metadata lookup, making this faster when you only need vector IDs.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity
    /// - [`NeedleError::CollectionNotFound`] - Collection no longer exists
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// coll.insert("v2", &[0.9, 0.1, 0.0, 0.0], None)?;
    ///
    /// let results = coll.search_ids(&[1.0, 0.0, 0.0, 0.0], 10)?;
    /// for (id, distance) in results {
    ///     println!("{}: {:.4}", id, distance);
    /// }
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_ids(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.db.search_ids_internal(&self.name, query, k)
    }

    /// Check if a vector with the given ID exists in the collection.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(coll.contains("v1"));
    /// assert!(!coll.contains("v2"));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Update an existing vector and its metadata.
    ///
    /// Replaces both the vector data and metadata. The vector is re-indexed
    /// after the update.
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the vector to update
    /// * `vector` - New vector data (must match collection dimensions)
    /// * `metadata` - New metadata (replaces existing metadata)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::VectorNotFound`] - No vector with the given ID exists
    /// - [`NeedleError::DimensionMismatch`] - Vector dimensions don't match
    /// - [`NeedleError::CollectionNotFound`] - Collection no longer exists
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    /// use serde_json::json;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"version": 1})))?;
    ///
    /// // Update the vector and metadata
    /// coll.update("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"version": 2})))?;
    ///
    /// let (vec, _meta) = coll.get("v1").unwrap();
    /// assert_eq!(vec, vec![0.0, 1.0, 0.0, 0.0]);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn update(
        &self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.update_internal(&self.name, id, vector, metadata)
    }

    /// Search with post-filter support.
    ///
    /// Post-filtering applies the filter after ANN search, which is useful when:
    /// - You need to guarantee k candidates before filtering
    /// - The filter involves expensive computation
    /// - The filter is highly selective and pre-filtering would miss results
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `k` - Number of results to return
    /// * `pre_filter` - Optional filter applied during ANN search (efficient)
    /// * `post_filter` - Filter applied after ANN search
    /// * `post_filter_factor` - Over-fetch factor (search fetches k * factor candidates)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Database, Filter};
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4)?;
    /// let coll = db.collection("docs")?;
    ///
    /// // Insert vectors
    /// coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(serde_json::json!({"score": 10})))?;
    /// coll.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(serde_json::json!({"score": 20})))?;
    ///
    /// // Search with post-filter: find similar vectors, then filter by score
    /// let post_filter = Filter::gt("score", 15);
    /// let results = coll.search_with_post_filter(
    ///     &[1.0, 0.0, 0.0, 0.0],
    ///     10,
    ///     None,
    ///     &post_filter,
    ///     3,
    /// )?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_with_post_filter(
        &self,
        query: &[f32],
        k: usize,
        pre_filter: Option<&Filter>,
        post_filter: &Filter,
        post_filter_factor: usize,
    ) -> Result<Vec<SearchResult>> {
        self.db.search_with_post_filter_internal(
            &self.name,
            query,
            k,
            pre_filter,
            post_filter,
            post_filter_factor,
        )
    }
}

// ============================================================================
// Replicated Database - Raft Integration
// ============================================================================

use crate::raft::{
    Command as RaftCommand, NodeId, RaftConfig, RaftError, RaftMessage,
    RaftNode, RaftState,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A database command that can be replicated across the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplicatedCommand {
    /// Create a new collection.
    CreateCollection {
        name: String,
        dimensions: usize,
        config: Option<CollectionConfig>,
    },
    /// Drop a collection.
    DropCollection { name: String },
    /// Insert a vector into a collection.
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Update a vector in a collection.
    Update {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Delete a vector from a collection.
    Delete { collection: String, id: String },
    /// Compact a collection.
    Compact { collection: String },
    /// Clear a collection.
    Clear { collection: String },
    /// No-op command for Raft leader establishment.
    Noop,
}

impl ReplicatedCommand {
    /// Convert to a Raft command for serialization.
    fn to_raft_command(&self) -> RaftCommand {
        // Serialize the replicated command and store in the metadata field
        let serialized = serde_json::to_string(self).unwrap_or_default();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("cmd".to_string(), serialized);

        // Use the Insert variant to carry our custom command
        RaftCommand::Insert {
            id: self.command_id(),
            vector: Vec::new(),
            metadata,
        }
    }

    /// Generate a unique command ID for deduplication.
    fn command_id(&self) -> String {
        let mut hasher = DefaultHasher::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        now.hash(&mut hasher);
        format!("cmd_{:016x}", hasher.finish())
    }

    /// Parse a replicated command from a Raft command.
    fn from_raft_command(cmd: &RaftCommand) -> Option<Self> {
        match cmd {
            RaftCommand::Insert { metadata, .. } => {
                metadata.get("cmd")
                    .and_then(|s| serde_json::from_str(s).ok())
            }
            RaftCommand::Noop => Some(ReplicatedCommand::Noop),
            _ => None,
        }
    }
}

/// Configuration for replicated database.
#[derive(Debug, Clone)]
pub struct ReplicatedDatabaseConfig {
    /// Node ID for this replica.
    pub node_id: NodeId,
    /// Raft configuration.
    pub raft_config: RaftConfig,
    /// Peer nodes in the cluster.
    pub peers: Vec<NodeId>,
    /// Allow reads from followers (eventually consistent).
    pub allow_follower_reads: bool,
}

impl Default for ReplicatedDatabaseConfig {
    fn default() -> Self {
        Self {
            node_id: NodeId(1),
            raft_config: RaftConfig::default(),
            peers: Vec::new(),
            allow_follower_reads: true,
        }
    }
}

impl ReplicatedDatabaseConfig {
    /// Create a new config with the given node ID.
    pub fn new(node_id: u64) -> Self {
        Self {
            node_id: NodeId(node_id),
            ..Default::default()
        }
    }

    /// Add peer nodes to the cluster.
    pub fn with_peers(mut self, peers: Vec<u64>) -> Self {
        self.peers = peers.into_iter().map(NodeId).collect();
        self
    }

    /// Set Raft configuration.
    pub fn with_raft_config(mut self, config: RaftConfig) -> Self {
        self.raft_config = config;
        self
    }

    /// Enable or disable follower reads.
    pub fn allow_follower_reads(mut self, allow: bool) -> Self {
        self.allow_follower_reads = allow;
        self
    }
}

/// A replicated database that uses Raft for consensus.
///
/// This wraps a `Database` and ensures all write operations are
/// replicated across the cluster before being applied locally.
///
/// # Example
///
/// ```ignore
/// use needle::database::{Database, ReplicatedDatabase, ReplicatedDatabaseConfig};
///
/// // Create the underlying database
/// let db = Database::in_memory();
///
/// // Configure replication
/// let config = ReplicatedDatabaseConfig::new(1)
///     .with_peers(vec![2, 3]);
///
/// // Create replicated database
/// let mut replicated_db = ReplicatedDatabase::new(db, config);
///
/// // Operations are now replicated
/// replicated_db.propose_create_collection("documents", 384)?;
/// ```
pub struct ReplicatedDatabase {
    /// The underlying database.
    db: Database,
    /// Raft node for consensus.
    raft: RaftNode,
    /// Configuration.
    config: ReplicatedDatabaseConfig,
    /// Message handler for network communication.
    message_handler: Option<Box<dyn MessageHandler>>,
}

/// Trait for handling Raft messages (network communication).
pub trait MessageHandler: Send + Sync {
    /// Send a message to another node.
    fn send(&self, to: NodeId, message: RaftMessage) -> Result<()>;

    /// Receive messages from other nodes.
    fn receive(&self) -> Vec<(NodeId, RaftMessage)>;
}

impl ReplicatedDatabase {
    /// Create a new replicated database.
    pub fn new(db: Database, config: ReplicatedDatabaseConfig) -> Self {
        let mut raft = RaftNode::new(config.node_id, config.raft_config.clone());
        raft.initialize(config.peers.clone());

        Self {
            db,
            raft,
            config,
            message_handler: None,
        }
    }

    /// Set the message handler for network communication.
    pub fn with_message_handler<H: MessageHandler + 'static>(mut self, handler: H) -> Self {
        self.message_handler = Some(Box::new(handler));
        self
    }

    /// Get the underlying database reference.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.raft.is_leader()
    }

    /// Get the current leader's node ID.
    pub fn leader(&self) -> Option<NodeId> {
        self.raft.leader()
    }

    /// Get the current Raft state.
    pub fn state(&self) -> RaftState {
        self.raft.state()
    }

    /// Get the node ID.
    pub fn node_id(&self) -> NodeId {
        self.raft.id()
    }

    /// Tick the Raft node (should be called periodically).
    ///
    /// This handles election timeouts and heartbeats.
    pub fn tick(&mut self) {
        self.raft.tick();

        // Send outgoing messages
        if let Some(handler) = &self.message_handler {
            for msg in self.raft.take_messages() {
                if let Err(e) = handler.send(msg.to, msg.message) {
                    warn!(target = ?msg.to, error = %e, "Failed to send Raft message");
                }
            }
        }

        // Receive and process incoming messages
        if let Some(handler) = &self.message_handler {
            for (from, message) in handler.receive() {
                self.raft.handle_message(from, message);
            }
        }

        // Apply committed commands
        self.apply_committed();
    }

    /// Apply committed commands to the database.
    fn apply_committed(&mut self) {
        for cmd in self.raft.take_committed() {
            if let Some(replicated_cmd) = ReplicatedCommand::from_raft_command(&cmd) {
                if let Err(e) = self.apply_command(&replicated_cmd) {
                    warn!(error = %e, "Failed to apply committed command");
                }
            }
        }
    }

    /// Apply a single command to the database.
    fn apply_command(&self, cmd: &ReplicatedCommand) -> Result<()> {
        match cmd {
            ReplicatedCommand::CreateCollection { name, dimensions, config } => {
                if let Some(cfg) = config {
                    self.db.create_collection_with_config(cfg.clone())
                } else {
                    self.db.create_collection(name, *dimensions)
                }
            }
            ReplicatedCommand::DropCollection { name } => {
                self.db.drop_collection(name).map(|_| ())
            }
            ReplicatedCommand::Insert { collection, id, vector, metadata } => {
                let coll = self.db.collection(collection)?;
                coll.insert(id, vector, metadata.clone())
            }
            ReplicatedCommand::Update { collection, id, vector, metadata } => {
                let coll = self.db.collection(collection)?;
                coll.update(id, vector, metadata.clone())
            }
            ReplicatedCommand::Delete { collection, id } => {
                let coll = self.db.collection(collection)?;
                coll.delete(id).map(|_| ())
            }
            ReplicatedCommand::Compact { collection } => {
                let coll = self.db.collection(collection)?;
                coll.compact().map(|_| ())
            }
            ReplicatedCommand::Clear { collection } => {
                // Clear by iterating and deleting all
                let coll = self.db.collection(collection)?;
                let ids = coll.ids()?;
                for id in ids {
                    let _ = coll.delete(&id);
                }
                Ok(())
            }
            ReplicatedCommand::Noop => Ok(()),
        }
    }

    /// Propose creating a new collection.
    pub fn propose_create_collection(
        &mut self,
        name: &str,
        dimensions: usize,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::CreateCollection {
            name: name.to_string(),
            dimensions,
            config: None,
        })
    }

    /// Propose creating a collection with custom configuration.
    pub fn propose_create_collection_with_config(
        &mut self,
        config: CollectionConfig,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::CreateCollection {
            name: config.name.clone(),
            dimensions: config.dimensions,
            config: Some(config),
        })
    }

    /// Propose dropping a collection.
    pub fn propose_drop_collection(
        &mut self,
        name: &str,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::DropCollection {
            name: name.to_string(),
        })
    }

    /// Propose inserting a vector.
    pub fn propose_insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::Insert {
            collection: collection.to_string(),
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
        })
    }

    /// Propose updating a vector.
    pub fn propose_update(
        &mut self,
        collection: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::Update {
            collection: collection.to_string(),
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
        })
    }

    /// Propose deleting a vector.
    pub fn propose_delete(
        &mut self,
        collection: &str,
        id: &str,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        self.propose_command(ReplicatedCommand::Delete {
            collection: collection.to_string(),
            id: id.to_string(),
        })
    }

    /// Propose a command to the Raft cluster.
    fn propose_command(
        &mut self,
        cmd: ReplicatedCommand,
    ) -> std::result::Result<(), ReplicatedDatabaseError> {
        if !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        let raft_cmd = cmd.to_raft_command();
        self.raft.propose(raft_cmd).map_err(|e| match e {
            RaftError::NotLeader(leader) => ReplicatedDatabaseError::NotLeader(leader),
            RaftError::InvalidOperation(msg) => ReplicatedDatabaseError::InvalidOperation(msg),
        })?;

        Ok(())
    }

    /// Read from the local database.
    ///
    /// If `allow_follower_reads` is false and this node is not the leader,
    /// this will return an error.
    pub fn read_collection(&self, name: &str) -> std::result::Result<CollectionRef<'_>, ReplicatedDatabaseError> {
        if !self.config.allow_follower_reads && !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        self.db.collection(name).map_err(ReplicatedDatabaseError::Database)
    }

    /// List all collections (local read).
    pub fn list_collections(&self) -> std::result::Result<Vec<String>, ReplicatedDatabaseError> {
        if !self.config.allow_follower_reads && !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        Ok(self.db.list_collections())
    }

    /// Check if a collection exists (local read).
    pub fn has_collection(&self, name: &str) -> std::result::Result<bool, ReplicatedDatabaseError> {
        if !self.config.allow_follower_reads && !self.is_leader() {
            return Err(ReplicatedDatabaseError::NotLeader(self.leader()));
        }

        Ok(self.db.has_collection(name))
    }

    /// Get cluster status.
    pub fn status(&self) -> ReplicatedDatabaseStatus {
        let raft_status = self.raft.status();
        ReplicatedDatabaseStatus {
            node_id: raft_status.id,
            state: raft_status.state,
            term: raft_status.term,
            leader_id: raft_status.leader_id,
            commit_index: raft_status.commit_index,
            last_applied: raft_status.last_applied,
            cluster_size: raft_status.cluster_size,
            collections: self.db.list_collections(),
            total_vectors: self.db.total_vectors(),
        }
    }
}

/// Errors from replicated database operations.
#[derive(Debug)]
pub enum ReplicatedDatabaseError {
    /// Not the leader; redirect to the leader.
    NotLeader(Option<NodeId>),
    /// Invalid operation.
    InvalidOperation(String),
    /// Database error.
    Database(NeedleError),
}

impl std::fmt::Display for ReplicatedDatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplicatedDatabaseError::NotLeader(Some(leader)) => {
                write!(f, "Not the leader; redirect to node {:?}", leader)
            }
            ReplicatedDatabaseError::NotLeader(None) => {
                write!(f, "Not the leader; leader unknown")
            }
            ReplicatedDatabaseError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
            ReplicatedDatabaseError::Database(e) => {
                write!(f, "Database error: {}", e)
            }
        }
    }
}

impl std::error::Error for ReplicatedDatabaseError {}

/// Status of a replicated database node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatedDatabaseStatus {
    /// This node's ID.
    pub node_id: NodeId,
    /// Current Raft state.
    pub state: RaftState,
    /// Current term.
    pub term: u64,
    /// Known leader ID.
    pub leader_id: Option<NodeId>,
    /// Commit index.
    pub commit_index: u64,
    /// Last applied index.
    pub last_applied: u64,
    /// Number of nodes in cluster.
    pub cluster_size: usize,
    /// List of collections.
    pub collections: Vec<String>,
    /// Total vectors across all collections.
    pub total_vectors: usize,
}

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
        coll.insert_with_ttl("doc1", &vec, None, Some(3600)).unwrap();

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
        coll.insert_with_ttl("expired", &vec1, None, Some(0)).unwrap();

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
        let config = crate::CollectionConfig::new("test", 3)
            .with_distance(DistanceFunction::Cosine);
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
        assert!((d1 - d2).abs() < 0.001, "Cosine distances should be equal for same direction");

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
        let config = crate::CollectionConfig::new("test", 32)
            .with_distance(DistanceFunction::Euclidean);
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
