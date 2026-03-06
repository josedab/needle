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

use crate::collection::{Collection, CollectionConfig};
use crate::error::{NeedleError, Result};
use crate::storage::{crc32, StorageEngine, HEADER_SIZE};
use crate::tuning::AdaptiveTuner;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// Export entry type: (id, vector, metadata)
pub type ExportEntry = (String, Vec<f32>, Option<Value>);

mod aliases;
mod config;
mod internal;
mod snapshots;

pub use config::DatabaseConfig;

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
    /// Storage engine (Arc-wrapped Mutex for shared save access, including auto-flush thread)
    storage: Arc<parking_lot::Mutex<Option<StorageEngine>>>,
    /// Collections (thread-safe)
    state: Arc<RwLock<DatabaseState>>,
    /// Whether there are unsaved changes (Arc'd for auto-flush thread sharing)
    dirty: Arc<AtomicBool>,
    /// Modification generation counter for race-free dirty tracking.
    /// Incremented on every modification. Used during save to detect
    /// concurrent modifications and avoid clearing dirty flag prematurely.
    modification_gen: Arc<AtomicU64>,
    /// Last saved generation. If modification_gen > saved_gen, database is dirty.
    saved_gen: Arc<AtomicU64>,
    /// Optional adaptive tuner for online index-tuning feedback
    adaptive_tuner: Option<Arc<AdaptiveTuner>>,
    /// Optional adaptive index manager for workload-driven index selection
    adaptive_index_manager: Option<Arc<crate::indexing::adaptive_index_manager::AdaptiveIndexManager>>,
    /// Optional metrics aggregator for the observability dashboard
    #[cfg(feature = "observability")]
    dashboard_metrics: Option<Arc<crate::observe::dashboard::MetricsAggregator>>,
    /// Optional replica manager for snapshot-based replication
    replica_manager: Option<Arc<crate::persistence::replica_manager::ReplicaManager>>,
    /// Versioned stores per collection for MVCC time-travel queries
    versioned_stores: Arc<RwLock<HashMap<String, crate::persistence::vector_versioning::VersionedStore>>>,
    /// MVCC transaction manager for snapshot isolation
    transaction_manager: crate::persistence::transaction::TransactionManager,
    /// Maps transaction IDs to their collection-aware operations.
    /// This bridges the gap between the generic TransactionManager (which doesn't
    /// know about collections) and the Database (which does).
    tx_collection_ops: Arc<RwLock<HashMap<crate::persistence::transaction::TransactionId, Vec<(String, crate::persistence::transaction::WriteOperation)>>>>,
    /// Handle for the auto-flush background thread (if active).
    /// Set to `Some` when `auto_flush_interval_secs > 0`.
    auto_flush_shutdown: Option<Arc<AtomicBool>>,
    auto_flush_handle: Option<std::thread::JoinHandle<()>>,
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
                let file_size = storage.file_size()?;
                let state_len = if header.state_size > 0 {
                    header.state_size as usize
                } else {
                    (file_size - header.metadata_offset) as usize
                };

                // Validate that the claimed state region fits within the file
                if header.metadata_offset.saturating_add(state_len as u64) > file_size {
                    return Err(NeedleError::Corruption(
                        format!(
                            "State region (offset={}, size={}) exceeds file size ({})",
                            header.metadata_offset, state_len, file_size,
                        ),
                    ));
                }

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

        let auto_flush_interval = config.auto_flush_interval_secs;

        let mut db = Self {
            config,
            storage: Arc::new(parking_lot::Mutex::new(Some(storage))),
            state: Arc::new(RwLock::new(state)),
            dirty: Arc::new(AtomicBool::new(false)),
            modification_gen: Arc::new(AtomicU64::new(0)),
            saved_gen: Arc::new(AtomicU64::new(0)),
            adaptive_tuner: None, adaptive_index_manager: None, replica_manager: None,
            versioned_stores: Arc::new(RwLock::new(HashMap::new())),
            transaction_manager: crate::persistence::transaction::TransactionManager::new(),
            tx_collection_ops: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "observability")]
            dashboard_metrics: None,
            auto_flush_shutdown: None,
            auto_flush_handle: None,
        };

        if auto_flush_interval > 0 {
            db.start_auto_flush(auto_flush_interval);
        }

        Ok(db)
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
            storage: Arc::new(parking_lot::Mutex::new(None)),
            state: Arc::new(RwLock::new(DatabaseState::default())),
            dirty: Arc::new(AtomicBool::new(false)),
            modification_gen: Arc::new(AtomicU64::new(0)),
            saved_gen: Arc::new(AtomicU64::new(0)),
            adaptive_tuner: None, adaptive_index_manager: None, replica_manager: None,
            versioned_stores: Arc::new(RwLock::new(HashMap::new())),
            transaction_manager: crate::persistence::transaction::TransactionManager::new(),
            tx_collection_ops: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "observability")]
            dashboard_metrics: None,
            auto_flush_shutdown: None,
            auto_flush_handle: None,
        }
    }

    /// Create a lightweight handle that shares the same internal state (collections,
    /// versioned stores, etc.) as this database. The returned instance has no
    /// storage engine and is intended for read-heavy access patterns such as the
    /// MCP HTTP transport where `Arc<Database>` ownership is required.
    pub fn shared_handle(&self) -> Self {
        Self {
            config: self.config.clone(),
            storage: Arc::new(parking_lot::Mutex::new(None)),
            state: Arc::clone(&self.state),
            dirty: Arc::new(AtomicBool::new(false)),
            modification_gen: Arc::new(AtomicU64::new(0)),
            saved_gen: Arc::new(AtomicU64::new(0)),
            adaptive_tuner: self.adaptive_tuner.clone(),
            adaptive_index_manager: self.adaptive_index_manager.clone(),
            replica_manager: self.replica_manager.clone(),
            versioned_stores: Arc::clone(&self.versioned_stores),
            transaction_manager: crate::persistence::transaction::TransactionManager::new(),
            tx_collection_ops: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "observability")]
            dashboard_metrics: self.dashboard_metrics.clone(),
            auto_flush_shutdown: None,
            auto_flush_handle: None,
        }
    }

    /// Start a background thread that periodically saves dirty data to disk.
    ///
    /// The thread wakes every `interval_secs` seconds, checks if the database
    /// has unsaved changes, and calls `save()` if needed. The thread is cleanly
    /// shut down when the `Database` is dropped.
    fn start_auto_flush(&mut self, interval_secs: u64) {
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = Arc::clone(&shutdown);

        let db_state = Arc::clone(&self.state);
        let db_storage = Arc::clone(&self.storage);
        let db_mod_gen = Arc::clone(&self.modification_gen);
        let db_saved_gen = Arc::clone(&self.saved_gen);
        let db_dirty = Arc::clone(&self.dirty);

        let handle = std::thread::Builder::new()
            .name("needle-auto-flush".to_string())
            .spawn(move || {
                let interval = std::time::Duration::from_secs(interval_secs);
                loop {
                    // Sleep in small increments for responsive shutdown
                    let sleep_step = std::time::Duration::from_millis(500);
                    let mut slept = std::time::Duration::ZERO;
                    while slept < interval {
                        if shutdown_clone.load(Ordering::Relaxed) {
                            return;
                        }
                        std::thread::sleep(sleep_step.min(interval - slept));
                        slept += sleep_step;
                    }

                    if shutdown_clone.load(Ordering::Relaxed) {
                        return;
                    }

                    // Check if dirty
                    let current_mod = db_mod_gen.load(Ordering::Acquire);
                    let current_saved = db_saved_gen.load(Ordering::Acquire);
                    if current_mod <= current_saved && !db_dirty.load(Ordering::Acquire) {
                        continue;
                    }

                    // Perform save
                    let mut storage_guard = db_storage.lock();
                    let storage = match storage_guard.as_mut() {
                        Some(s) => s,
                        None => continue,
                    };

                    let gen_at_save_start = db_mod_gen.load(Ordering::Acquire);

                    let (state_bytes, vector_count) = {
                        let state = db_state.read();
                        let vc: u64 = state.collections.values().map(|c| c.len() as u64).sum();
                        match serde_json::to_vec(&*state) {
                            Ok(bytes) => (bytes, vc),
                            Err(e) => {
                                eprintln!("needle: auto-flush serialization error: {e}");
                                continue;
                            }
                        }
                    };

                    let mut header = storage.header().clone();
                    header.metadata_offset = crate::storage::HEADER_SIZE as u64;
                    header.vector_count = vector_count;

                    if let Err(e) = storage.atomic_save(&header, &state_bytes) {
                        eprintln!("needle: auto-flush save error: {e}");
                        continue;
                    }

                    // Update saved_gen
                    loop {
                        let cs = db_saved_gen.load(Ordering::Acquire);
                        if gen_at_save_start <= cs {
                            break;
                        }
                        match db_saved_gen.compare_exchange_weak(
                            cs, gen_at_save_start, Ordering::Release, Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(_) => continue,
                        }
                    }

                    let cm = db_mod_gen.load(Ordering::Acquire);
                    if cm == gen_at_save_start {
                        db_dirty.store(false, Ordering::Release);
                    }
                }
            })
            .ok();

        self.auto_flush_shutdown = Some(shutdown);
        self.auto_flush_handle = handle;
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

    /// Attach an adaptive index manager for workload-driven index selection.
    pub fn set_adaptive_index_manager(
        &mut self,
        manager: Arc<crate::indexing::adaptive_index_manager::AdaptiveIndexManager>,
    ) {
        self.adaptive_index_manager = Some(manager);
    }

    /// Get a reference to the adaptive index manager, if one is attached.
    pub fn adaptive_index_manager(
        &self,
    ) -> Option<&Arc<crate::indexing::adaptive_index_manager::AdaptiveIndexManager>> {
        self.adaptive_index_manager.as_ref()
    }

    /// Attach a metrics aggregator for the observability dashboard.
    #[cfg(feature = "observability")]
    pub fn set_dashboard_metrics(
        &mut self,
        metrics: Arc<crate::observe::dashboard::MetricsAggregator>,
    ) {
        self.dashboard_metrics = Some(metrics);
    }

    /// Get a reference to the dashboard metrics aggregator.
    #[cfg(feature = "observability")]
    pub fn dashboard_metrics(
        &self,
    ) -> Option<&Arc<crate::observe::dashboard::MetricsAggregator>> {
        self.dashboard_metrics.as_ref()
    }

    /// Attach a replica manager for snapshot-based replication.
    pub fn set_replica_manager(
        &mut self,
        manager: Arc<crate::persistence::replica_manager::ReplicaManager>,
    ) {
        self.replica_manager = Some(manager);
    }

    /// Get a reference to the replica manager.
    pub fn replica_manager(
        &self,
    ) -> Option<&Arc<crate::persistence::replica_manager::ReplicaManager>> {
        self.replica_manager.as_ref()
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
        if self.storage.lock().is_some() {
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
        self.state
            .read()
            .collections
            .keys()
            .filter(|k| !k.starts_with("__snapshot__"))
            .cloned()
            .collect()
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
        // Check aliases under read lock (O(n) iteration without blocking writers)
        {
            let state = self.state.read();
            let has_aliases = state
                .aliases
                .iter()
                .any(|(_, target)| *target == name);
            if has_aliases {
                return Err(NeedleError::CollectionHasAliases(name.to_string()));
            }
        }

        // Re-acquire write lock for the actual deletion
        let mut state = self.state.write();

        // Re-check aliases under write lock to avoid TOCTOU race
        let has_aliases = state
            .aliases
            .iter()
            .any(|(_, target)| *target == name);
        if has_aliases {
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

    /// Rename an existing collection.
    ///
    /// Updates the collection's internal name, moves it under the new key in the
    /// database, and rewrites any aliases that pointed to the old name.
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::CollectionNotFound`] if `old_name` does not exist.
    /// Returns [`NeedleError::CollectionAlreadyExists`] if `new_name` is already taken.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("drafts", 128)?;
    /// db.rename_collection("drafts", "published")?;
    ///
    /// assert!(!db.has_collection("drafts"));
    /// assert!(db.has_collection("published"));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn rename_collection(&self, old_name: &str, new_name: &str) -> Result<()> {
        crate::collection::config::validate_collection_name(new_name)?;

        let mut state = self.state.write();

        if !state.collections.contains_key(old_name) {
            return Err(NeedleError::CollectionNotFound(old_name.to_string()));
        }
        if state.collections.contains_key(new_name) {
            return Err(NeedleError::CollectionAlreadyExists(new_name.to_string()));
        }

        if let Some(mut collection) = state.collections.remove(old_name) {
            collection.set_name(new_name.to_string());
            state.collections.insert(new_name.to_string(), collection);

            // Update any aliases pointing to the old name
            for target in state.aliases.values_mut() {
                if *target == old_name {
                    *target = new_name.to_string();
                }
            }

            info!(old_name = %old_name, new_name = %new_name, "Collection renamed");
            self.mark_modified();
        }
        Ok(())
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

    /// Enable vector versioning on a collection for MVCC time-travel queries.
    ///
    /// Creates a `VersionedStore` for the collection with the given configuration.
    pub fn enable_versioning(
        &self,
        collection: &str,
        config: crate::persistence::vector_versioning::VersioningConfig,
    ) -> Result<()> {
        if !self.has_collection(collection) {
            return Err(NeedleError::CollectionNotFound(collection.to_string()));
        }
        let store = crate::persistence::vector_versioning::VersionedStore::new(config);
        self.versioned_stores
            .write()
            .insert(collection.to_string(), store);
        Ok(())
    }

    /// Access the versioned store for a collection (if versioning is enabled).
    pub fn versioned_store(
        &self,
        collection: &str,
    ) -> Result<parking_lot::MappedRwLockReadGuard<'_, crate::persistence::vector_versioning::VersionedStore>> {
        let stores = self.versioned_stores.read();
        if !stores.contains_key(collection) {
            return Err(NeedleError::InvalidOperation(format!(
                "Versioning not enabled for collection '{collection}'"
            )));
        }
        Ok(parking_lot::RwLockReadGuard::map(stores, |s| {
            s.get(collection).expect("checked above")
        }))
    }

    /// Access the versioned store mutably for put/delete operations.
    pub fn versioned_store_mut(
        &self,
        collection: &str,
    ) -> Result<parking_lot::MappedRwLockWriteGuard<'_, crate::persistence::vector_versioning::VersionedStore>> {
        let stores = self.versioned_stores.write();
        if !stores.contains_key(collection) {
            return Err(NeedleError::InvalidOperation(format!(
                "Versioning not enabled for collection '{collection}'"
            )));
        }
        Ok(parking_lot::RwLockWriteGuard::map(stores, |s| {
            s.get_mut(collection).expect("checked above")
        }))
    }

    /// Alter a collection's dimensions, migrating all existing vectors.
    ///
    /// Applies the given [`DimensionStrategy`] to re-dimension every vector
    /// in the collection and rebuilds the HNSW index. The operation is atomic:
    /// on failure, the collection is left unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the collection doesn't exist, target dimensions
    /// are zero, or the strategy is incompatible with the dimension change.
    pub fn alter_collection_dimensions(
        &self,
        name: &str,
        new_dimensions: usize,
        strategy: crate::persistence::schema_evolution::DimensionStrategy,
    ) -> Result<()> {
        use crate::persistence::schema_evolution::adapt_dimensions;

        if new_dimensions == 0 {
            return Err(NeedleError::InvalidConfig(
                "Target dimensions must be > 0".into(),
            ));
        }

        let mut state = self.state.write();
        let coll = state
            .collections
            .get_mut(name)
            .ok_or_else(|| NeedleError::CollectionNotFound(name.to_string()))?;

        let old_dimensions = coll.dimensions();
        if old_dimensions == new_dimensions {
            return Ok(());
        }

        // Collect all vectors and metadata
        let ids: Vec<String> = coll.ids().map(|s| s.to_string()).collect();
        let mut entries: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some((vec, meta)) = coll.get(id) {
                let new_vec = adapt_dimensions(vec, new_dimensions, &strategy)?;
                entries.push((id.clone(), new_vec, meta.cloned()));
            }
        }

        // Rebuild collection with new dimensions
        let mut new_config = coll.config().clone();
        new_config.dimensions = new_dimensions;
        let mut new_coll = Collection::new(new_config);

        for (id, vec, meta) in entries {
            new_coll.insert_vec(id, vec, meta)?;
        }

        // Atomic swap
        *coll = new_coll;
        self.mark_modified();

        Ok(())
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
    pub fn save(&self) -> Result<()> {
        let mut storage_guard = self.storage.lock();
        let storage = match storage_guard.as_mut() {
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

        // Serialize state outside the read lock to minimize lock hold time.
        // Clone the state under the lock, release it, then do the expensive
        // JSON serialization without blocking writers.
        let (state_bytes, vector_count) = {
            let state = self.state.read();
            let vector_count: u64 = state.collections.values().map(|c| c.len() as u64).sum();
            let bytes = serde_json::to_vec(&*state)?;
            (bytes, vector_count)
        };

        // Build updated header
        let mut header = storage.header().clone();
        header.metadata_offset = HEADER_SIZE as u64;
        header.vector_count = vector_count;

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
        let gen = self.modification_gen.fetch_add(1, Ordering::Release);
        self.dirty.store(true, Ordering::Release);

        // Advance replica manager LSN if attached
        if let Some(rm) = &self.replica_manager {
            rm.advance_leader_lsn(gen + 1);
        }
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

    // ── MVCC Transaction API ────────────────────────────────────────────

    /// Begin a new MVCC transaction with snapshot isolation.
    ///
    /// Returns a `TransactionId` that must be passed to subsequent operations.
    /// The transaction sees a consistent snapshot of data committed before this call.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs", 4).unwrap();
    ///
    /// let tx = db.begin_transaction().unwrap();
    /// // ... perform operations within the transaction ...
    /// db.commit_transaction(tx).unwrap();
    /// ```
    pub fn begin_transaction(
        &self,
    ) -> Result<crate::persistence::transaction::TransactionId> {
        self.transaction_manager.begin()
    }

    /// Buffer an insert operation within an active transaction.
    ///
    /// The insert is not visible to other readers until the transaction is committed.
    pub fn transaction_insert(
        &self,
        tx_id: crate::persistence::transaction::TransactionId,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let id_str = id.into();
        // Validate collection exists and dimensions match
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;
        let expected_dims = coll.config().dimensions;
        if vector.len() != expected_dims {
            return Err(NeedleError::DimensionMismatch {
                expected: expected_dims,
                got: vector.len(),
            });
        }
        drop(state);

        // Record the collection name in our local mapping
        {
            let mut tx_ops = self.tx_collection_ops.write();
            tx_ops
                .entry(tx_id)
                .or_insert_with(Vec::new)
                .push((collection.to_string(), crate::persistence::transaction::WriteOperation::Insert {
                    id: id_str.clone(),
                    vector: vector.clone(),
                    metadata: metadata.clone(),
                }));
        }

        self.transaction_manager.with_transaction_mut(tx_id, |txn| {
            txn.insert(id_str, vector, metadata)
        })
    }

    /// Buffer a delete operation within an active transaction.
    pub fn transaction_delete(
        &self,
        tx_id: crate::persistence::transaction::TransactionId,
        collection: &str,
        id: impl Into<String>,
    ) -> Result<()> {
        let id_str = id.into();

        {
            let mut tx_ops = self.tx_collection_ops.write();
            tx_ops
                .entry(tx_id)
                .or_insert_with(Vec::new)
                .push((collection.to_string(), crate::persistence::transaction::WriteOperation::Delete {
                    id: id_str.clone(),
                }));
        }

        self.transaction_manager.with_transaction_mut(tx_id, |txn| {
            txn.delete(id_str)
        })
    }

    /// Commit a transaction, applying all buffered operations atomically.
    ///
    /// The commit protocol:
    /// 1. Acquire the database write lock (prevents concurrent modifications)
    /// 2. Run conflict detection via `TransactionManager::commit()`
    /// 3. Apply all buffered operations to collections
    /// 4. Release locks
    ///
    /// If conflict detection fails, no operations are applied.
    /// If an individual operation fails (e.g., duplicate ID), the transaction
    /// is still considered committed but the failing operation is logged.
    pub fn commit_transaction(
        &self,
        tx_id: crate::persistence::transaction::TransactionId,
    ) -> Result<()> {
        // Extract the collection-aware operations
        let collection_ops = {
            let mut tx_ops = self.tx_collection_ops.write();
            tx_ops.remove(&tx_id).unwrap_or_default()
        };

        // Hold the write lock BEFORE conflict detection to ensure atomicity:
        // no other transaction can commit between our check and our apply.
        let mut state = self.state.write();

        // Conflict detection (may fail with Conflict error)
        self.transaction_manager.commit(tx_id)?;

        // Apply operations — the transaction is now committed in the manager,
        // so we must apply all operations. Failures are collected and reported.
        let mut apply_errors: Vec<String> = Vec::new();
        for (collection_name, op) in &collection_ops {
            let coll = match state.collections.get_mut(collection_name) {
                Some(c) => c,
                None => {
                    apply_errors.push(format!("collection '{collection_name}' not found"));
                    continue;
                }
            };
            match op {
                crate::persistence::transaction::WriteOperation::Insert { id, vector, metadata } => {
                    if let Err(e) = coll.insert(id, vector, metadata.clone()) {
                        apply_errors.push(format!("insert '{id}': {e}"));
                    }
                }
                crate::persistence::transaction::WriteOperation::Delete { id } => {
                    if let Err(e) = coll.delete(id) {
                        apply_errors.push(format!("delete '{id}': {e}"));
                    }
                }
                crate::persistence::transaction::WriteOperation::Update { id, vector, metadata } => {
                    if vector.is_some() || metadata.is_some() {
                        let _ = coll.delete(id);
                        if let Some(vec) = vector {
                            if let Err(e) = coll.insert(id, vec, metadata.clone()) {
                                apply_errors.push(format!("update '{id}': {e}"));
                            }
                        }
                    }
                }
            }
        }
        drop(state);

        self.mark_modified();

        if apply_errors.is_empty() {
            Ok(())
        } else {
            tracing::warn!(
                "Transaction {tx_id}: {} operation(s) failed during apply",
                apply_errors.len()
            );
            Err(NeedleError::InvalidOperation(format!(
                "Transaction committed but {} operation(s) failed: {}",
                apply_errors.len(),
                apply_errors.join("; ")
            )))
        }
    }

    /// Roll back a transaction, discarding all buffered operations.
    pub fn rollback_transaction(
        &self,
        tx_id: crate::persistence::transaction::TransactionId,
    ) -> Result<()> {
        // Clean up our collection mapping
        {
            let mut tx_ops = self.tx_collection_ops.write();
            tx_ops.remove(&tx_id);
        }
        self.transaction_manager.rollback(tx_id)
    }

    /// Get transaction statistics (active count, total committed/aborted/conflicts).
    pub fn transaction_stats(
        &self,
    ) -> crate::persistence::transaction::TransactionStats {
        self.transaction_manager.stats()
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Signal auto-flush thread to shut down and wait for it
        if let Some(shutdown) = self.auto_flush_shutdown.take() {
            shutdown.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.auto_flush_handle.take() {
            let _ = handle.join();
        }

        if self.is_dirty() {
            if self.config.auto_save {
                if let Err(e) = self.save() {
                    eprintln!(
                        "needle: error: auto-save failed during drop: {e}. \
                         Data may not have been persisted."
                    );
                }
            } else {
                eprintln!(
                    "needle: warning: database dropped with unsaved changes. \
                     Call save() explicitly before dropping to persist data."
                );
            }
        }
    }
}

pub mod collection_ref;
pub use collection_ref::{CollectionRef, SearchParams};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vector;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_database_in_memory() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();

        db.create_collection("documents", 128)?;

        let coll = db.collection("documents")?;
        let vec = random_vector(128);
        coll.insert("doc1", &vec, Some(json!({"title": "Test"})))?;

        assert_eq!(coll.len(), 1);
        Ok(())
    }

    #[test]
    fn test_database_persistence() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let path = dir.path().join("test.needle");

        // Create and populate database
        {
            let mut db = Database::open(&path)?;
            db.create_collection("documents", 64)?;

            let coll = db.collection("documents")?;
            for i in 0..10 {
                let vec = random_vector(64);
                coll.insert(format!("doc{}", i), &vec, Some(json!({"index": i})))?;
            }

            db.save()?;
        }

        // Reopen and verify
        {
            let db = Database::open(&path)?;
            let coll = db.collection("documents")?;
            assert_eq!(coll.len(), 10);

            let (_, meta) = coll.get("doc5").ok_or("doc5 not found")?;
            assert_eq!(meta.ok_or("missing metadata")?["index"], 5);
        }
        Ok(())
    }

    #[test]
    fn test_database_multiple_collections() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();

        db.create_collection("images", 512)?;
        db.create_collection("text", 384)?;

        let collections = db.list_collections();
        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"images".to_string()));
        assert!(collections.contains(&"text".to_string()));
        Ok(())
    }

    #[test]
    fn test_database_drop_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();

        db.create_collection("temp", 128)?;
        assert!(db.has_collection("temp"));

        db.drop_collection("temp")?;
        assert!(!db.has_collection("temp"));
        Ok(())
    }

    #[test]
    fn test_database_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("test", 32)?;

        let coll = db.collection("test")?;

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(32);
            coll.insert(format!("doc{}", i), &vec, Some(json!({"index": i})))?;
        }

        // Search
        let query = random_vector(32);
        let results = coll.search(&query, 10)?;

        assert_eq!(results.len(), 10);
        Ok(())
    }

    #[test]
    fn test_collection_not_found() {
        let db = Database::in_memory();
        let result = db.collection("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_collection_already_exists() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("test", 128)?;

        let result = db.create_collection("test", 128);
        assert!(result.is_err());
        Ok(())
    }

    // ========== Alias Tests ==========

    #[test]
    fn test_alias_create_and_resolve() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;

        // Create alias
        db.create_alias("prod", "v1")?;

        // Resolve alias
        assert_eq!(db.get_canonical_name("prod"), Some("v1".to_string()));

        // Access collection via alias
        let coll = db.collection("prod")?;
        assert_eq!(coll.count(None)?, 0);
        Ok(())
    }

    #[test]
    fn test_alias_list() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;
        db.create_collection("v2", 128)?;

        db.create_alias("prod", "v1")?;
        db.create_alias("staging", "v2")?;

        let aliases = db.list_aliases();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&("prod".to_string(), "v1".to_string())));
        assert!(aliases.contains(&("staging".to_string(), "v2".to_string())));
        Ok(())
    }

    #[test]
    fn test_alias_delete() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;
        db.create_alias("prod", "v1")?;

        assert!(db.get_canonical_name("prod").is_some());

        // Delete alias
        let deleted = db.delete_alias("prod")?;
        assert!(deleted);

        // Alias no longer exists
        assert!(db.get_canonical_name("prod").is_none());

        // Deleting again returns false
        let deleted = db.delete_alias("prod")?;
        assert!(!deleted);
        Ok(())
    }

    #[test]
    fn test_alias_prevents_collection_drop() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;
        db.create_alias("prod", "v1")?;

        // Trying to drop collection with alias should fail
        let result = db.drop_collection("v1");
        assert!(result.is_err());

        // Delete alias first
        db.delete_alias("prod")?;

        // Now drop should succeed
        let dropped = db.drop_collection("v1")?;
        assert!(dropped);
        Ok(())
    }

    #[test]
    fn test_alias_update() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;
        db.create_collection("v2", 128)?;

        db.create_alias("prod", "v1")?;
        assert_eq!(db.get_canonical_name("prod"), Some("v1".to_string()));

        // Update alias to point to v2
        db.update_alias("prod", "v2")?;
        assert_eq!(db.get_canonical_name("prod"), Some("v2".to_string()));
        Ok(())
    }

    #[test]
    fn test_aliases_for_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;

        db.create_alias("prod", "v1")?;
        db.create_alias("latest", "v1")?;

        let aliases = db.aliases_for_collection("v1");
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&"prod".to_string()));
        assert!(aliases.contains(&"latest".to_string()));
        Ok(())
    }

    #[test]
    fn test_alias_already_exists() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("v1", 128)?;

        db.create_alias("prod", "v1")?;

        // Creating same alias again should fail
        let result = db.create_alias("prod", "v1");
        assert!(result.is_err());
        Ok(())
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
    fn test_ttl_insert_and_stats() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("test", 32)?;

        let coll = db.collection("test")?;

        // Insert with TTL
        let vec = random_vector(32);
        coll.insert_with_ttl("doc1", &vec, None, Some(3600))?;

        // Insert without TTL
        let vec2 = random_vector(32);
        coll.insert("doc2", &vec2, None)?;

        let (total_with_ttl, _expired, _earliest, _latest) = coll.ttl_stats();
        assert_eq!(total_with_ttl, 1);
        assert_eq!(coll.count(None)?, 2);
        Ok(())
    }

    #[test]
    fn test_ttl_get_and_set() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("test", 32)?;

        let coll = db.collection("test")?;

        let vec = random_vector(32);
        coll.insert("doc1", &vec, None)?;

        // Initially no TTL
        assert!(coll.get_ttl("doc1").is_none());

        // Set TTL
        coll.set_ttl("doc1", Some(3600))?;
        assert!(coll.get_ttl("doc1").is_some());

        // Remove TTL
        coll.set_ttl("doc1", None)?;
        assert!(coll.get_ttl("doc1").is_none());
        Ok(())
    }

    #[test]
    fn test_ttl_expiration_sweep() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("test", 32)?;

        let coll = db.collection("test")?;

        // Insert with very short TTL (already expired - TTL of 0 means expire immediately)
        let vec = random_vector(32);
        // Use a TTL in the past by setting expiration directly
        coll.insert_with_ttl("doc1", &vec, None, Some(0))?;

        // Wait a tiny bit and expire - but since TTL=0 sets expiration to now,
        // the vector should be expired immediately on the next sweep
        std::thread::sleep(std::time::Duration::from_millis(10));

        let expired = coll.expire_vectors()?;
        assert_eq!(expired, 1);

        // Vector should be gone
        assert!(coll.get("doc1").is_none());
        Ok(())
    }

    #[test]
    fn test_ttl_lazy_expiration_in_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::CollectionConfig;

        let db = Database::in_memory();

        // Create collection with lazy expiration enabled
        let config = CollectionConfig::new("test", 32).with_lazy_expiration(true);
        db.create_collection_with_config(config)?;

        let coll = db.collection("test")?;

        // Insert expired vector
        let vec1 = random_vector(32);
        coll.insert_with_ttl("expired", &vec1, None, Some(0))?;

        // Insert non-expired vector
        let vec2 = random_vector(32);
        coll.insert("valid", &vec2, None)?;

        std::thread::sleep(std::time::Duration::from_millis(10));

        // Search should only return the valid vector (lazy expiration filters out expired)
        let results = coll.search(&vec1, 10)?;

        // The expired vector should not appear in results
        assert!(results.iter().all(|r| r.id != "expired"));
        Ok(())
    }

    // ========== Distance Override Tests ==========

    #[test]
    fn test_distance_override_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::DistanceFunction;

        let db = Database::in_memory();
        db.create_collection("test", 4)?;

        let coll = db.collection("test")?;

        // Insert some vectors
        coll.insert("a", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("b", &[0.0, 1.0, 0.0, 0.0], None)?;
        coll.insert("c", &[0.5, 0.5, 0.0, 0.0], None)?;

        let query = vec![1.0, 0.0, 0.0, 0.0];

        // Search with default distance (cosine)
        let results_default = coll.search(&query, 3)?;
        assert_eq!(results_default.len(), 3);
        assert_eq!(results_default[0].id, "a"); // Exact match

        // Search with euclidean distance override (triggers brute-force)
        let results_euclidean = coll
            .search_with_options(&query, 3, Some(DistanceFunction::Euclidean), None, None, 3)?;
        assert_eq!(results_euclidean.len(), 3);
        assert_eq!(results_euclidean[0].id, "a"); // Still exact match
        assert!(results_euclidean[0].distance < 0.001); // Euclidean distance to itself is 0
        Ok(())
    }

    #[test]
    fn test_distance_override_produces_correct_ordering() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::DistanceFunction;

        let db = Database::in_memory();

        // Create collection with cosine distance
        let config =
            crate::CollectionConfig::new("test", 3).with_distance(DistanceFunction::Cosine);
        db.create_collection_with_config(config)?;

        let coll = db.collection("test")?;

        // Insert vectors with different magnitudes but same direction
        coll.insert("small", &[1.0, 0.0, 0.0], None)?;
        coll.insert("large", &[10.0, 0.0, 0.0], None)?;

        let query = vec![5.0, 0.0, 0.0];

        // With cosine, both should have same distance (direction matters, not magnitude)
        let cosine_results = coll.search(&query, 2)?;
        let d1 = cosine_results[0].distance;
        let d2 = cosine_results[1].distance;
        assert!(
            (d1 - d2).abs() < 0.001,
            "Cosine distances should be equal for same direction"
        );

        // With euclidean override, magnitude matters
        let euclidean_results = coll
            .search_with_options(&query, 2, Some(DistanceFunction::Euclidean), None, None, 3)?;

        // "small" (1.0) is closer to query (5.0) than "large" (10.0) in euclidean
        // distance(5,1) = 4, distance(5,10) = 5
        assert_eq!(euclidean_results[0].id, "small");
        assert_eq!(euclidean_results[1].id, "large");
        Ok(())
    }

    #[test]
    fn test_distance_override_same_as_index_uses_hnsw() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::DistanceFunction;

        let db = Database::in_memory();

        // Create collection with euclidean distance
        let config =
            crate::CollectionConfig::new("test", 32).with_distance(DistanceFunction::Euclidean);
        db.create_collection_with_config(config)?;

        let coll = db.collection("test")?;

        for i in 0..100 {
            let vec = random_vector(32);
            coll.insert(format!("doc{}", i), &vec, None)?;
        }

        let query = random_vector(32);

        // Override with same distance as index - should use HNSW (not brute force)
        let results = coll
            .search_with_options(&query, 10, Some(DistanceFunction::Euclidean), None, None, 3)?;

        assert_eq!(results.len(), 10);
        Ok(())
    }

    // ── Next-Gen Feature Tests ──────────────────────────────────────────

    #[test]
    fn test_matryoshka_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 128)?;
        let coll = db.collection("docs")?;

        for i in 0..50 {
            let vec = random_vector(128);
            coll.insert(format!("doc{}", i), &vec, None)?;
        }

        let query = random_vector(128);
        let results = coll.search_matryoshka(&query, 10, 64, 4)?;

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
        // Verify results are sorted by distance
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
        Ok(())
    }

    #[test]
    fn test_matryoshka_search_fallback_to_full() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 64)?;
        let coll = db.collection("docs")?;

        for i in 0..20 {
            let vec = random_vector(64);
            coll.insert(format!("doc{}", i), &vec, None)?;
        }

        let query = random_vector(64);
        // coarse_dims >= dims should fall back to normal search
        let results = coll.search_matryoshka(&query, 5, 128, 4)?;
        assert_eq!(results.len(), 5);
        Ok(())
    }

    #[test]
    fn test_snapshot_create_list_restore() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 32)?;
        let coll = db.collection("docs")?;

        // Insert data
        for i in 0..10 {
            coll.insert(format!("v{}", i), &random_vector(32), None)?;
        }
        assert_eq!(coll.len(), 10);

        // Create snapshot
        coll.create_snapshot("snap1")?;

        // List snapshots
        let snapshots = coll.list_snapshots();
        assert!(snapshots.contains(&"snap1".to_string()));

        // Delete some vectors
        for i in 0..5 {
            coll.delete(&format!("v{}", i))?;
        }
        assert_eq!(coll.len(), 5);

        // Restore snapshot
        coll.restore_snapshot("snap1")?;
        assert_eq!(coll.len(), 10);
        Ok(())
    }

    #[test]
    fn test_memory_store_and_recall() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("memories", 32)?;
        let coll = db.collection("memories")?;

        // Store a "memory" with metadata
        let vec = random_vector(32);
        let metadata = json!({
            "_memory_content": "The user prefers dark mode",
            "_memory_tier": "semantic",
            "_memory_importance": 0.8,
        });
        coll.insert("mem_1", &vec, Some(metadata))?;

        // Recall by similarity
        let results = coll.search(&vec, 5)?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "mem_1");

        let meta = results[0].metadata.as_ref().ok_or("missing metadata")?;
        assert_eq!(meta["_memory_tier"], "semantic");
        assert_eq!(meta["_memory_content"], "The user prefers dark mode");

        // Forget
        assert!(coll.delete("mem_1")?);
        assert_eq!(coll.len(), 0);
        Ok(())
    }

    #[test]
    fn test_vector_diff_between_collections() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("a", 32)?;
        db.create_collection("b", 32)?;

        let coll_a = db.collection("a")?;
        let coll_b = db.collection("b")?;

        // Shared vectors (same ID, same data)
        for i in 0..5 {
            let vec = random_vector(32);
            coll_a.insert(format!("shared_{}", i), &vec, None)?;
            coll_b.insert(format!("shared_{}", i), &vec, None)?;
        }

        // Only in A
        for i in 0..3 {
            coll_a.insert(format!("only_a_{}", i), &random_vector(32), None)?;
        }

        // Only in B
        for i in 0..2 {
            coll_b.insert(format!("only_b_{}", i), &random_vector(32), None)?;
        }

        let ids_a: std::collections::HashSet<String> = coll_a.ids()?.into_iter().collect();
        let ids_b: std::collections::HashSet<String> = coll_b.ids()?.into_iter().collect();

        assert_eq!(ids_a.difference(&ids_b).count(), 3); // only in A
        assert_eq!(ids_b.difference(&ids_a).count(), 2); // only in B
        assert_eq!(ids_a.intersection(&ids_b).count(), 5); // shared
        Ok(())
    }

    #[test]
    fn test_cost_estimator_integration() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::search::cost_estimator::{CostEstimator, CollectionStatistics};

        let db = Database::in_memory();
        db.create_collection("bench", 128)?;
        let coll = db.collection("bench")?;

        for i in 0..100 {
            coll.insert(format!("v{}", i), &random_vector(128), None)?;
        }

        let stats = coll.stats()?;
        let col_stats = CollectionStatistics::new(stats.vector_count, stats.dimensions, 0.0);

        let estimator = CostEstimator::default();
        let plan = estimator.plan(&col_stats, 10, None);

        assert!(plan.cost.estimated_latency_ms > 0.0);
        assert!(plan.cost.distance_computations > 0);
        assert!(!plan.rationale.is_empty());
        Ok(())
    }

    #[test]
    fn test_quantized_index_persistence() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::indexing::quantization::{ScalarQuantizer, QuantizedIndex};

        let vectors: Vec<Vec<f32>> = (0..50).map(|_| random_vector(32)).collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let sq = ScalarQuantizer::train(&refs);
        let idx = QuantizedIndex::Scalar(sq);

        // Serialize
        let bytes = idx.to_bytes();
        assert!(!bytes.is_empty());

        // Deserialize
        let restored = QuantizedIndex::from_bytes(&bytes)?;
        assert_eq!(restored.dimensions(), 32);
        assert_eq!(restored.compression_label(), "4x (scalar u8)");
        Ok(())
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_embedding_router_integration() {
        use crate::services::embedding_router::{EmbeddingRouter, RouterConfig, ProviderEntry};

        let mut router = EmbeddingRouter::new(RouterConfig::default());
        router.register(ProviderEntry::new("local", 384, 0.0));
        router.register(ProviderEntry::new("openai", 1536, 0.0001));
        router.pin_collection("premium", "openai");

        // Regular collection → first provider (local)
        assert_eq!(router.route(Some("docs")), Some("local".to_string()));

        // Pinned collection → pinned provider
        assert_eq!(router.route(Some("premium")), Some("openai".to_string()));

        // Record failure to test failover
        router.record_failure("local");
        router.record_failure("local");
        router.record_failure("local");
        assert_eq!(router.route(Some("docs")), Some("openai".to_string()));

        // Stats tracking
        let stats = router.stats();
        assert_eq!(stats.len(), 2);
    }

    #[test]
    #[cfg(feature = "experimental")]
    fn test_webhook_service() {
        use crate::services::webhook_delivery::{
            WebhookService, WebhookConfig, WebhookSubscription, EventFilter
        };

        let mut svc = WebhookService::new(WebhookConfig::default());
        let sub = WebhookSubscription::new("https://example.com/hook", EventFilter::all());
        svc.subscribe(sub);

        svc.enqueue("docs", "insert", "v1");
        svc.enqueue("docs", "delete", "v2");

        let stats = svc.process_queue();
        assert_eq!(stats.delivered, 2);

        let (delivered, failed, pending) = svc.total_stats();
        assert_eq!(delivered, 2);
        assert_eq!(failed, 0);
        assert_eq!(pending, 0);
    }

    #[test]
    fn test_rbac_path_extraction() {
        // Test the path extraction logic used by RBAC middleware
        let path = "/collections/my_docs/vectors/v1";
        let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect();
        assert_eq!(parts[0], "collections");
        assert_eq!(parts[1], "my_docs");

        let path2 = "/health";
        let parts2: Vec<&str> = path2.trim_start_matches('/').split('/').collect();
        assert_ne!(parts2[0], "collections");
    }

    #[test]
    fn test_auto_save_config_defaults() {
        let config = DatabaseConfig::default();
        assert!(!config.auto_save, "auto_save should default to false");

        let config = DatabaseConfig::new("/tmp/test.needle").with_auto_save(true);
        assert!(config.auto_save);
    }

    #[test]
    fn test_auto_save_on_drop() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let path = dir.path().join("auto_save_test.needle");

        // Create database with auto_save, insert data, then drop
        {
            let mut db = Database::open_with_config(DatabaseConfig::new(&path).with_auto_save(true))?;
            db.create_collection("test", 4)?;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], None)?;
            assert!(db.is_dirty());
            // Drop triggers auto-save
        }

        // Reopen and verify data was persisted
        let db2 = Database::open(&path)?;
        let coll2 = db2.collection("test")?;
        assert_eq!(coll2.len(), 1);
        Ok(())
    }

    #[test]
    fn test_auto_flush_interval_config() {
        let config = DatabaseConfig::new("/tmp/test.needle")
            .with_auto_flush_interval_secs(30);
        assert_eq!(config.auto_flush_interval_secs, 30);
    }

    #[test]
    fn test_auto_flush_thread_starts_and_stops() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let path = dir.path().join("auto_flush_test.needle");

        {
            let config = DatabaseConfig::new(&path)
                .with_auto_flush_interval_secs(1);
            let db = Database::open_with_config(config)?;
            db.create_collection("test", 4)?;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], None)?;
            assert!(db.is_dirty());

            // Wait for auto-flush to trigger (interval is 1 second)
            std::thread::sleep(std::time::Duration::from_secs(2));

            // After auto-flush, should no longer be dirty
            assert!(!db.is_dirty(), "auto-flush should have saved dirty changes");
            // Drop shuts down the thread cleanly
        }

        // Verify data was persisted by the auto-flush
        let db2 = Database::open(&path)?;
        let coll2 = db2.collection("test")?;
        assert_eq!(coll2.len(), 1);
        Ok(())
    }

    // ── Schema Evolution Tests ──────────────────────────────────────────

    #[test]
    fn test_alter_collection_dimensions_zero_pad() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::schema_evolution::DimensionStrategy;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;

        let coll = db.collection("docs")?;
        coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"tag": "a"})))?;
        coll.insert("v2", &[5.0, 6.0, 7.0, 8.0], None)?;

        db.alter_collection_dimensions("docs", 6, DimensionStrategy::ZeroPad)?;

        let coll = db.collection("docs")?;
        assert_eq!(coll.dimensions(), Some(6));
        assert_eq!(coll.len(), 2);

        // Verify vector was zero-padded
        let state = db.state.read();
        let c = state.collections.get("docs").expect("collection exists");
        let (vec, meta) = c.get("v1").expect("vector exists");
        assert_eq!(vec.len(), 6);
        assert_eq!(&vec[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&vec[4..], &[0.0, 0.0]);
        assert!(meta.is_some());
        Ok(())
    }

    #[test]
    fn test_alter_collection_dimensions_truncate() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::schema_evolution::DimensionStrategy;

        let db = Database::in_memory();
        db.create_collection("docs", 8)?;
        let coll = db.collection("docs")?;
        coll.insert("v1", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], None)?;

        db.alter_collection_dimensions("docs", 4, DimensionStrategy::Truncate)?;

        let coll = db.collection("docs")?;
        assert_eq!(coll.dimensions(), Some(4));

        let state = db.state.read();
        let c = state.collections.get("docs").expect("exists");
        let (vec, _) = c.get("v1").expect("exists");
        assert_eq!(vec, &[1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn test_alter_collection_dimensions_noop_same() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::schema_evolution::DimensionStrategy;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        let coll = db.collection("docs")?;
        coll.insert("v1", &[1.0, 2.0, 3.0, 4.0], None)?;

        // Same dimensions → no-op
        db.alter_collection_dimensions("docs", 4, DimensionStrategy::ZeroPad)?;
        let coll = db.collection("docs")?;
        assert_eq!(coll.len(), 1);
        Ok(())
    }

    #[test]
    fn test_alter_collection_dimensions_zero_rejects() {
        use crate::persistence::schema_evolution::DimensionStrategy;

        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let result = db.alter_collection_dimensions("docs", 0, DimensionStrategy::ZeroPad);
        assert!(result.is_err());
    }

    #[test]
    fn test_alter_collection_dimensions_not_found() {
        use crate::persistence::schema_evolution::DimensionStrategy;

        let db = Database::in_memory();
        let result = db.alter_collection_dimensions("nonexistent", 4, DimensionStrategy::ZeroPad);
        assert!(result.is_err());
    }

    #[test]
    fn test_alter_collection_search_after() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::schema_evolution::DimensionStrategy;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        let coll = db.collection("docs")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;

        db.alter_collection_dimensions("docs", 6, DimensionStrategy::ZeroPad)?;

        // Search should work with new dimensions
        let coll = db.collection("docs")?;
        let results = coll.search(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1"); // closest
        Ok(())
    }

    // ── Vector Versioning Tests ─────────────────────────────────────────

    #[test]
    fn test_enable_versioning() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::vector_versioning::VersioningConfig;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        db.enable_versioning("docs", VersioningConfig::default())?;

        // Should be able to access the store
        let store = db.versioned_store("docs")?;
        assert_eq!(store.stats().total_vectors, 0);
        Ok(())
    }

    #[test]
    fn test_versioning_not_found() {
        let db = Database::in_memory();
        let result = db.enable_versioning("nonexistent", Default::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_versioning_not_enabled() {
        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();
        let result = db.versioned_store("docs");
        assert!(result.is_err());
    }

    #[test]
    fn test_versioning_records_on_insert() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::vector_versioning::VersioningConfig;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        db.enable_versioning("docs", VersioningConfig::default())?;

        let coll = db.collection("docs")?;
        coll.insert_vec("v1", vec![1.0, 0.0, 0.0, 0.0], None)?;
        coll.insert_vec("v2", vec![0.0, 1.0, 0.0, 0.0], None)?;

        let store = db.versioned_store("docs")?;
        assert_eq!(store.stats().total_vectors, 2);
        assert_eq!(store.stats().total_versions, 2);

        // Verify specific version exists
        let latest = store.get_latest("v1");
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().vector, vec![1.0, 0.0, 0.0, 0.0]);
        Ok(())
    }

    #[test]
    fn test_versioning_records_on_delete() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::vector_versioning::VersioningConfig;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        db.enable_versioning("docs", VersioningConfig::default())?;

        let coll = db.collection("docs")?;
        coll.insert_vec("v1", vec![1.0, 0.0, 0.0, 0.0], None)?;
        coll.delete("v1")?;

        let store = db.versioned_store("docs")?;
        // Version store should have the insert + tombstone
        assert_eq!(store.stats().total_versions, 2);
        // Latest should be None (tombstone)
        assert!(store.get_latest("v1").is_none());
        Ok(())
    }

    #[test]
    fn test_versioning_time_travel() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::vector_versioning::VersioningConfig;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        db.enable_versioning("docs", VersioningConfig::default())?;

        let coll = db.collection("docs")?;
        coll.insert_vec("v1", vec![1.0, 0.0, 0.0, 0.0], None)?;

        // Get current time
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Vector should be visible at current timestamp
        let store = db.versioned_store("docs")?;
        let version = store.get_as_of("v1", now);
        assert!(version.is_some());

        // Vector should not be visible before it was inserted
        let version = store.get_as_of("v1", 0);
        assert!(version.is_none());
        Ok(())
    }

    #[test]
    fn test_versioning_records_on_slice_insert() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::vector_versioning::VersioningConfig;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        db.enable_versioning("docs", VersioningConfig::default())?;

        let coll = db.collection("docs")?;
        // Use insert (slice) not insert_vec (owned)
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;

        let store = db.versioned_store("docs")?;
        assert_eq!(store.stats().total_vectors, 1);
        let latest = store.get_latest("v1");
        assert!(latest.is_some());
        Ok(())
    }

    #[test]
    fn test_versioning_records_on_update() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::persistence::vector_versioning::VersioningConfig;

        let db = Database::in_memory();
        db.create_collection("docs", 4)?;
        db.enable_versioning("docs", VersioningConfig::default())?;

        let coll = db.collection("docs")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        coll.update("v1", &[0.0, 1.0, 0.0, 0.0], None)?;

        let store = db.versioned_store("docs")?;
        // Should have 2 versions: insert + update
        assert_eq!(store.stats().total_versions, 2);

        // Latest should be the updated vector
        let latest = store.get_latest("v1").unwrap();
        assert_eq!(latest.vector, vec![0.0, 1.0, 0.0, 0.0]);

        // History should have 2 entries
        let history = store.history("v1").unwrap();
        assert_eq!(history.len(), 2);
        Ok(())
    }

    #[test]
    fn test_mvcc_transaction_commit_applies() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 4)?;

        // Begin a transaction and insert
        let tx = db.begin_transaction()?;
        db.transaction_insert(tx, "docs", "v1", vec![1.0, 0.0, 0.0, 0.0], None)?;
        db.transaction_insert(tx, "docs", "v2", vec![0.0, 1.0, 0.0, 0.0], Some(json!({"key": "val"})))?;

        // Before commit: collection should be empty (snapshot isolation)
        let coll = db.collection("docs")?;
        assert_eq!(coll.len(), 0);

        // Commit applies the operations
        db.commit_transaction(tx)?;
        assert_eq!(coll.len(), 2);

        // Verify vectors are searchable
        let results = coll.search(&[1.0, 0.0, 0.0, 0.0], 2)?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1");
        Ok(())
    }

    #[test]
    fn test_mvcc_transaction_rollback_discards() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 4)?;

        let tx = db.begin_transaction()?;
        db.transaction_insert(tx, "docs", "v1", vec![1.0, 0.0, 0.0, 0.0], None)?;
        db.rollback_transaction(tx)?;

        let coll = db.collection("docs")?;
        assert_eq!(coll.len(), 0);
        Ok(())
    }

    #[test]
    fn test_mvcc_transaction_delete() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 4)?;

        // Pre-populate
        let coll = db.collection("docs")?;
        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        assert_eq!(coll.len(), 1);

        // Delete in a transaction
        let tx = db.begin_transaction()?;
        db.transaction_delete(tx, "docs", "v1")?;

        // Before commit, vector still visible
        assert_eq!(coll.len(), 1);

        db.commit_transaction(tx)?;
        assert_eq!(coll.len(), 0);
        Ok(())
    }

    #[test]
    fn test_mvcc_conflict_detection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Database::in_memory();
        db.create_collection("docs", 4)?;

        let tx1 = db.begin_transaction()?;
        let tx2 = db.begin_transaction()?;

        // Both write to the same ID
        db.transaction_insert(tx1, "docs", "v1", vec![1.0, 0.0, 0.0, 0.0], None)?;
        db.transaction_insert(tx2, "docs", "v1", vec![0.0, 1.0, 0.0, 0.0], None)?;

        // First commit succeeds
        db.commit_transaction(tx1)?;

        // Second commit should fail with conflict
        let result = db.commit_transaction(tx2);
        assert!(result.is_err());
        assert!(format!("{:?}", result).contains("Conflict"));
        Ok(())
    }

    #[test]
    fn test_concurrent_insert_and_search() -> Result<()> {
        use std::sync::Arc;
        use std::thread;

        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 4)?;

        let num_writers = 4;
        let inserts_per_writer = 50;
        let num_readers = 4;
        let reads_per_reader = 100;

        // Seed a few vectors so readers always have something to find
        let coll = db.collection("test")?;
        for i in 0..10 {
            let v = vec![(i as f32) * 0.1, 0.0, 0.0, 1.0];
            coll.insert(format!("seed-{i}"), &v, None)?;
        }

        let mut handles = Vec::new();

        // Writer threads
        for w in 0..num_writers {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                let coll = db.collection("test").unwrap();
                for i in 0..inserts_per_writer {
                    let id = format!("w{w}-{i}");
                    let v = vec![w as f32 * 0.1, i as f32 * 0.01, 0.0, 1.0];
                    let _ = coll.insert(&id, &v, None);
                }
            }));
        }

        // Reader threads
        for _r in 0..num_readers {
            let db = Arc::clone(&db);
            handles.push(thread::spawn(move || {
                let coll = db.collection("test").unwrap();
                for _ in 0..reads_per_reader {
                    let query = vec![0.5_f32, 0.5, 0.0, 1.0];
                    let results = coll.search(&query, 5);
                    // Should never error or panic
                    assert!(results.is_ok());
                }
            }));
        }

        for h in handles {
            h.join().expect("Thread panicked during concurrent test");
        }

        // Verify final state is consistent
        let coll = db.collection("test")?;
        let count = coll.len();
        // At least seed vectors + some inserts (exact count depends on timing)
        assert!(count >= 10, "Expected at least 10 vectors, got {}", count);
        assert!(
            count <= 10 + num_writers * inserts_per_writer,
            "Too many vectors: {}",
            count
        );

        Ok(())
    }

    #[test]
    fn test_concurrent_insert_and_delete() -> Result<()> {
        use std::sync::Arc;
        use std::thread;

        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 3)?;

        // Insert vectors that will be concurrently deleted
        let coll = db.collection("test")?;
        for i in 0..100 {
            coll.insert(format!("v{i}"), &[i as f32, 0.0, 1.0], None)?;
        }

        let mut handles = Vec::new();

        // Deleter thread: delete even IDs
        let db_d = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            let coll = db_d.collection("test").unwrap();
            for i in (0..100).step_by(2) {
                let id = format!("v{i}");
                let _ = coll.delete(&id);
            }
        }));

        // Inserter thread: insert new vectors concurrently
        let db_i = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            let coll = db_i.collection("test").unwrap();
            for i in 100..150 {
                let _ = coll.insert(format!("v{i}"), &[i as f32, 1.0, 0.0], None);
            }
        }));

        // Reader thread: continuously search
        let db_r = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            let coll = db_r.collection("test").unwrap();
            for _ in 0..200 {
                let _ = coll.search(&[50.0, 0.5, 0.5], 10);
            }
        }));

        for h in handles {
            h.join().expect("Thread panicked during concurrent delete test");
        }

        Ok(())
    }

    // ── rename_collection tests ──────────────────────────────────────────

    #[test]
    fn test_rename_collection_basic() {
        let db = Database::in_memory();
        db.create_collection("old", 4).unwrap();
        db.collection("old").unwrap().insert("v1", &[1.0; 4], None).unwrap();

        db.rename_collection("old", "new").unwrap();

        assert!(!db.has_collection("old"));
        assert!(db.has_collection("new"));
        // Data survived the rename
        let coll = db.collection("new").unwrap();
        assert_eq!(coll.len(), 1);
    }

    #[test]
    fn test_rename_collection_not_found() {
        let db = Database::in_memory();
        let err = db.rename_collection("nope", "new").unwrap_err();
        assert!(matches!(err, NeedleError::CollectionNotFound(_)));
    }

    #[test]
    fn test_rename_collection_conflict() {
        let db = Database::in_memory();
        db.create_collection("a", 4).unwrap();
        db.create_collection("b", 4).unwrap();

        let err = db.rename_collection("a", "b").unwrap_err();
        assert!(matches!(err, NeedleError::CollectionAlreadyExists(_)));
    }

    #[test]
    fn test_rename_collection_updates_aliases() {
        let db = Database::in_memory();
        db.create_collection("v1", 4).unwrap();
        db.create_alias("prod", "v1").unwrap();

        db.rename_collection("v1", "v2").unwrap();

        // Alias now points to the new name
        let coll = db.collection("prod").unwrap();
        assert_eq!(coll.name(), "v2");
    }
}
