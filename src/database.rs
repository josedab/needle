use crate::collection::{Collection, CollectionConfig, SearchResult};
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::storage::{StorageEngine, HEADER_SIZE};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
}

/// The main database handle
pub struct Database {
    /// Database configuration
    config: DatabaseConfig,
    /// Storage engine
    storage: Option<StorageEngine>,
    /// Collections (thread-safe)
    state: Arc<RwLock<DatabaseState>>,
    /// Whether there are unsaved changes (AtomicBool for lock-free access)
    dirty: AtomicBool,
}

impl Database {
    /// Open or create a database at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = DatabaseConfig::new(path.as_ref());
        Self::open_with_config(config)
    }

    /// Open or create a database with custom configuration
    pub fn open_with_config(config: DatabaseConfig) -> Result<Self> {
        let exists = config.path.exists();

        if !exists && !config.create_if_missing {
            return Err(NeedleError::InvalidDatabase(format!(
                "Database not found: {:?}",
                config.path
            )));
        }

        let mut storage = StorageEngine::open(&config.path)?;

        // Load existing state or create new
        let state = if exists {
            // Read state from file if metadata exists
            let header = storage.header();
            if header.metadata_offset >= HEADER_SIZE as u64 {
                let state_len = storage.file_size()? - header.metadata_offset;
                if state_len > 0 {
                    let state_bytes =
                        storage.read_at(header.metadata_offset, state_len as usize)?;
                    serde_json::from_slice(&state_bytes).unwrap_or_default()
                } else {
                    DatabaseState::default()
                }
            } else {
                DatabaseState::default()
            }
        } else {
            DatabaseState::default()
        };

        Ok(Self {
            config,
            storage: Some(storage),
            state: Arc::new(RwLock::new(state)),
            dirty: AtomicBool::new(false),
        })
    }

    /// Create an in-memory database (not persisted)
    pub fn in_memory() -> Self {
        Self {
            config: DatabaseConfig::default(),
            storage: None,
            state: Arc::new(RwLock::new(DatabaseState::default())),
            dirty: AtomicBool::new(false),
        }
    }

    /// Get the database path
    pub fn path(&self) -> Option<&Path> {
        if self.storage.is_some() {
            Some(&self.config.path)
        } else {
            None
        }
    }

    /// Create a new collection
    pub fn create_collection(&self, name: impl Into<String>, dimensions: usize) -> Result<()> {
        self.create_collection_with_config(CollectionConfig::new(name, dimensions))
    }

    /// Create a new collection with custom configuration
    pub fn create_collection_with_config(&self, config: CollectionConfig) -> Result<()> {
        let mut state = self.state.write();

        if state.collections.contains_key(&config.name) {
            return Err(NeedleError::CollectionAlreadyExists(config.name));
        }

        let collection = Collection::new(config.clone());
        state.collections.insert(config.name, collection);

        self.dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Get a collection by name (read-only access)
    pub fn collection(&self, name: &str) -> Result<CollectionRef<'_>> {
        let state = self.state.read();
        if !state.collections.contains_key(name) {
            return Err(NeedleError::CollectionNotFound(name.to_string()));
        }
        Ok(CollectionRef {
            db: self,
            name: name.to_string(),
        })
    }

    /// List all collection names
    pub fn list_collections(&self) -> Vec<String> {
        self.state.read().collections.keys().cloned().collect()
    }

    /// Drop a collection
    pub fn drop_collection(&self, name: &str) -> Result<bool> {
        let mut state = self.state.write();
        let removed = state.collections.remove(name).is_some();
        if removed {
            self.dirty.store(true, Ordering::Release);
        }
        Ok(removed)
    }

    /// Check if a collection exists
    pub fn has_collection(&self, name: &str) -> bool {
        self.state.read().collections.contains_key(name)
    }

    /// Save changes to disk
    pub fn save(&mut self) -> Result<()> {
        let storage = match &mut self.storage {
            Some(s) => s,
            None => return Ok(()), // In-memory database, nothing to save
        };

        let state = self.state.read();
        let state_bytes = serde_json::to_vec(&*state)?;

        // Write state after header
        let offset = HEADER_SIZE as u64;
        storage.write_at(offset, &state_bytes)?;

        // Update header
        let header = storage.header_mut();
        header.metadata_offset = offset;
        header.vector_count = state.collections.values().map(|c| c.len() as u64).sum();
        storage.write_header()?;

        storage.sync()?;
        self.dirty.store(false, Ordering::Release);

        Ok(())
    }

    /// Check if there are unsaved changes
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Get total number of vectors across all collections
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
        self.dirty.store(true, Ordering::Release);
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
        self.dirty.store(true, Ordering::Release);
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
        self.dirty.store(true, Ordering::Release);
        Ok(())
    }

    fn search_internal(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let state = self.state.read();
        let coll = state
            .collections
            .get(collection)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection.to_string()))?;

        coll.search(query, k)
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
            self.dirty.store(true, Ordering::Release);
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
            self.dirty.store(true, Ordering::Release);
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
        // Auto-save on drop if dirty
        if self.is_dirty() {
            let _ = self.save();
        }
    }
}

/// A reference to a collection for convenient access
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

    /// Insert a vector
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_internal(&self.name, id, vector, metadata)
    }

    /// Insert a vector, taking ownership (more efficient when you have a Vec)
    pub fn insert_vec(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.insert_vec_internal(&self.name, id, vector, metadata)
    }

    /// Search for similar vectors
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.db.search_internal(&self.name, query, k)
    }

    /// Search with metadata filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        self.db
            .search_with_filter_internal(&self.name, query, k, filter)
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        self.db.get_internal(&self.name, id)
    }

    /// Delete a vector
    pub fn delete(&self, id: &str) -> Result<bool> {
        self.db.delete_internal(&self.name, id)
    }

    /// Export all vectors from the collection
    /// Returns a vector of (id, vector, metadata) tuples
    pub fn export_all(&self) -> Result<Vec<ExportEntry>> {
        self.db.export_internal(&self.name)
    }

    /// Get all vector IDs in the collection
    pub fn ids(&self) -> Result<Vec<String>> {
        self.db.ids_internal(&self.name)
    }

    /// Compact the collection, removing deleted vectors
    /// Returns the number of vectors removed
    pub fn compact(&self) -> Result<usize> {
        self.db.compact_internal(&self.name)
    }

    /// Check if the collection needs compaction
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.db.needs_compaction_internal(&self.name, threshold)
    }

    /// Count vectors, optionally matching a filter
    pub fn count(&self, filter: Option<&Filter>) -> Result<usize> {
        self.db.count_internal(&self.name, filter)
    }

    /// Get the number of deleted vectors pending compaction
    pub fn deleted_count(&self) -> usize {
        self.db.deleted_count_internal(&self.name)
    }

    /// Search and return only IDs with distances (faster than full search)
    pub fn search_ids(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.db.search_ids_internal(&self.name, query, k)
    }

    /// Check if a vector with the given ID exists in the collection
    pub fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Update a vector by ID
    pub fn update(
        &self,
        id: &str,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        self.db.update_internal(&self.name, id, vector, metadata)
    }
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
}
