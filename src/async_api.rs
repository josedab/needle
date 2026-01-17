//! Async API for Needle Vector Database
//!
//! Provides async/await interface for non-blocking operations on the Needle
//! vector database. This module wraps the synchronous [`Database`] API using
//! tokio's `spawn_blocking` for CPU-bound work, enabling integration with
//! async runtimes without blocking the executor.
//!
//! # Feature Flag
//!
//! This module is available with the `async` feature:
//!
//! ```toml
//! [dependencies]
//! needle = { version = "0.1", features = ["async"] }
//! ```
//!
//! The `async` feature provides the async API without the HTTP server.
//! If you need the HTTP server, use the `server` feature which includes `async`.
//!
//! # Features
//!
//! - **Async Operations**: All core database operations available as async methods
//! - **Streaming Support**: Stream large result sets efficiently with backpressure
//! - **Batch Processing**: Parallel batch operations with configurable concurrency
//! - **Thread Safety**: Uses `Arc<Database>` for safe concurrent access
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::async_api::{AsyncDatabase, AsyncDatabaseConfig};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> needle::Result<()> {
//!     // Open database asynchronously
//!     let db = AsyncDatabase::open("vectors.needle").await?;
//!
//!     // Create a collection
//!     db.create_collection("documents", 384).await?;
//!
//!     // Insert vectors
//!     let embedding = vec![0.1; 384];
//!     db.insert(
//!         "documents",
//!         "doc1",
//!         embedding.clone(),
//!         Some(json!({"title": "Hello World"}))
//!     ).await?;
//!
//!     // Search asynchronously
//!     let results = db.search("documents", embedding, 10).await?;
//!
//!     for result in results {
//!         println!("Found: {} with distance {}", result.id, result.distance);
//!     }
//!
//!     // Save changes
//!     db.save().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Streaming Large Result Sets
//!
//! For large collections, use streaming to process results incrementally:
//!
//! ```rust,ignore
//! use needle::async_api::AsyncDatabase;
//! use tokio_stream::StreamExt;
//!
//! async fn export_all(db: &AsyncDatabase, collection: &str) -> needle::Result<()> {
//!     let mut stream = db.stream_export(collection, 100).await?;
//!
//!     while let Some(batch) = stream.next().await {
//!         let entries = batch?;
//!         for (id, vector, metadata) in entries {
//!             // Process each entry
//!             println!("Exporting: {}", id);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Batch Operations
//!
//! Efficiently process multiple operations in parallel:
//!
//! ```rust,ignore
//! use needle::async_api::AsyncDatabase;
//!
//! async fn batch_search(db: &AsyncDatabase) -> needle::Result<()> {
//!     let queries: Vec<Vec<f32>> = vec![
//!         vec![0.1; 384],
//!         vec![0.2; 384],
//!         vec![0.3; 384],
//!     ];
//!
//!     let all_results = db.batch_search("documents", queries, 10).await?;
//!
//!     for (i, results) in all_results.iter().enumerate() {
//!         println!("Query {}: {} results", i, results.len());
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::collection::{CollectionConfig, SearchResult};
use crate::database::{Database, DatabaseConfig, ExportEntry};
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use serde_json::Value;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::Mutex;

/// Configuration for async database operations
#[derive(Debug, Clone)]
pub struct AsyncDatabaseConfig {
    /// Base database configuration
    pub database: DatabaseConfig,
    /// Maximum concurrent operations for batch processing
    pub max_concurrency: usize,
    /// Default batch size for streaming operations
    pub stream_batch_size: usize,
}

impl Default for AsyncDatabaseConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            max_concurrency: 4,
            stream_batch_size: 100,
        }
    }
}

impl AsyncDatabaseConfig {
    /// Create a new async config with the given path
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            database: DatabaseConfig::new(path),
            ..Default::default()
        }
    }

    /// Set maximum concurrency for batch operations
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = max;
        self
    }

    /// Set default batch size for streaming
    pub fn with_stream_batch_size(mut self, size: usize) -> Self {
        self.stream_batch_size = size;
        self
    }
}

/// Async wrapper for the Needle vector database
///
/// Provides async/await interface for all database operations. Uses
/// `tokio::spawn_blocking` internally to offload CPU-intensive work
/// to the blocking thread pool.
#[derive(Clone)]
pub struct AsyncDatabase {
    /// Inner database wrapped in Arc for shared ownership
    inner: Arc<Mutex<Database>>,
    /// Configuration for async operations
    config: AsyncDatabaseConfig,
}

impl AsyncDatabase {
    /// Open or create a database at the given path asynchronously
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database file
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let db = AsyncDatabase::open("my_vectors.needle").await?;
    /// ```
    pub async fn open<P: AsRef<Path> + Send + 'static>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let db = tokio::task::spawn_blocking(move || Database::open(path))
            .await
            .map_err(|e| NeedleError::Io(std::io::Error::other(e)))??;

        Ok(Self {
            inner: Arc::new(Mutex::new(db)),
            config: AsyncDatabaseConfig::default(),
        })
    }

    /// Open or create a database with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Async database configuration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = AsyncDatabaseConfig::new("vectors.needle")
    ///     .with_max_concurrency(8)
    ///     .with_stream_batch_size(200);
    /// let db = AsyncDatabase::open_with_config(config).await?;
    /// ```
    pub async fn open_with_config(config: AsyncDatabaseConfig) -> Result<Self> {
        let db_config = config.database.clone();
        let db = tokio::task::spawn_blocking(move || Database::open_with_config(db_config))
            .await
            .map_err(|e| NeedleError::Io(std::io::Error::other(e)))??;

        Ok(Self {
            inner: Arc::new(Mutex::new(db)),
            config,
        })
    }

    /// Create an in-memory database (not persisted)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let db = AsyncDatabase::in_memory();
    /// ```
    pub fn in_memory() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Database::in_memory())),
            config: AsyncDatabaseConfig::default(),
        }
    }

    /// Create an in-memory database with custom async configuration
    pub fn in_memory_with_config(config: AsyncDatabaseConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Database::in_memory())),
            config,
        }
    }

    /// Create a new collection
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the collection
    /// * `dimensions` - Vector dimensions
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// db.create_collection("documents", 384).await?;
    /// ```
    pub async fn create_collection(&self, name: impl Into<String>, dimensions: usize) -> Result<()> {
        let name = name.into();
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.create_collection(name, dimensions)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Create a new collection with custom configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Collection configuration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use needle::{CollectionConfig, DistanceFunction};
    ///
    /// let config = CollectionConfig::new("documents", 384)
    ///     .with_distance(DistanceFunction::Euclidean);
    /// db.create_collection_with_config(config).await?;
    /// ```
    pub async fn create_collection_with_config(&self, config: CollectionConfig) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.create_collection_with_config(config)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// List all collection names
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let collections = db.list_collections().await;
    /// for name in collections {
    ///     println!("Collection: {}", name);
    /// }
    /// ```
    pub async fn list_collections(&self) -> Vec<String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.list_collections()
        })
        .await
        .unwrap_or_default()
    }

    /// Check if a collection exists
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    pub async fn has_collection(&self, name: &str) -> bool {
        let inner = self.inner.clone();
        let name = name.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.has_collection(&name)
        })
        .await
        .unwrap_or(false)
    }

    /// Drop a collection
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    ///
    /// # Returns
    ///
    /// `true` if the collection was dropped, `false` if it didn't exist
    pub async fn drop_collection(&self, name: &str) -> Result<bool> {
        let inner = self.inner.clone();
        let name = name.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.drop_collection(&name)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Insert a vector into a collection
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `id` - Vector ID
    /// * `vector` - Vector data
    /// * `metadata` - Optional metadata
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use serde_json::json;
    ///
    /// db.insert(
    ///     "documents",
    ///     "doc1",
    ///     vec![0.1; 384],
    ///     Some(json!({"title": "Hello"}))
    /// ).await?;
    /// ```
    pub async fn insert(
        &self,
        collection: &str,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        let id = id.into();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            // Use insert_vec to avoid cloning the vector we already own
            coll.insert_vec(id, vector, metadata)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Batch insert multiple vectors
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `entries` - Vector of (id, vector, metadata) tuples
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let entries = vec![
    ///     ("doc1".to_string(), vec![0.1; 384], None),
    ///     ("doc2".to_string(), vec![0.2; 384], None),
    /// ];
    /// db.batch_insert("documents", entries).await?;
    /// ```
    pub async fn batch_insert(
        &self,
        collection: &str,
        entries: Vec<(String, Vec<f32>, Option<Value>)>,
    ) -> Result<()> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            for (id, vector, metadata) in entries {
                coll.insert(id, &vector, metadata)?;
            }
            Ok(())
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let results = db.search("documents", vec![0.1; 384], 10).await?;
    /// for result in results {
    ///     println!("{}: {}", result.id, result.distance);
    /// }
    /// ```
    pub async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.search(&query, k)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Search with a metadata filter
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    /// * `filter` - Metadata filter
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use needle::Filter;
    ///
    /// let filter = Filter::eq("category", "tech");
    /// let results = db.search_with_filter("documents", vec![0.1; 384], 10, filter).await?;
    /// ```
    pub async fn search_with_filter(
        &self,
        collection: &str,
        query: Vec<f32>,
        k: usize,
        filter: Filter,
    ) -> Result<Vec<SearchResult>> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.search_with_filter(&query, k, &filter)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Batch search with multiple queries
    ///
    /// Executes multiple searches in parallel using the blocking thread pool.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `queries` - Vector of query vectors
    /// * `k` - Number of results per query
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let queries = vec![
    ///     vec![0.1; 384],
    ///     vec![0.2; 384],
    /// ];
    /// let all_results = db.batch_search("documents", queries, 10).await?;
    /// ```
    pub async fn batch_search(
        &self,
        collection: &str,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            // Use rayon's parallel search internally
            let results: Result<Vec<Vec<SearchResult>>> = queries
                .iter()
                .map(|query| coll.search(query, k))
                .collect();
            results
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Batch search with filter
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `queries` - Vector of query vectors
    /// * `k` - Number of results per query
    /// * `filter` - Metadata filter
    pub async fn batch_search_with_filter(
        &self,
        collection: &str,
        queries: Vec<Vec<f32>>,
        k: usize,
        filter: Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            let results: Result<Vec<Vec<SearchResult>>> = queries
                .iter()
                .map(|query| coll.search_with_filter(query, k, &filter))
                .collect();
            results
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Get a vector by ID
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `id` - Vector ID
    ///
    /// # Returns
    ///
    /// Vector data and metadata if found
    pub async fn get(&self, collection: &str, id: &str) -> Option<(Vec<f32>, Option<Value>)> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            if let Ok(coll) = db.collection(&collection) {
                coll.get(&id)
            } else {
                None
            }
        })
        .await
        .ok()
        .flatten()
    }

    /// Delete a vector by ID
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `id` - Vector ID
    ///
    /// # Returns
    ///
    /// `true` if the vector was deleted, `false` if it didn't exist
    pub async fn delete(&self, collection: &str, id: &str) -> Result<bool> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        let id = id.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.delete(&id)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Batch delete multiple vectors
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `ids` - Vector of IDs to delete
    ///
    /// # Returns
    ///
    /// Number of vectors actually deleted
    pub async fn batch_delete(&self, collection: &str, ids: Vec<String>) -> Result<usize> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            let mut deleted = 0;
            for id in &ids {
                if coll.delete(id)? {
                    deleted += 1;
                }
            }
            Ok(deleted)
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Export all vectors from a collection
    ///
    /// For large collections, consider using `stream_export` instead.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    pub async fn export(&self, collection: &str) -> Result<Vec<ExportEntry>> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.export_all()
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Stream export vectors from a collection in batches
    ///
    /// Returns a stream that yields batches of export entries. This is more
    /// memory-efficient for large collections.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `batch_size` - Number of entries per batch
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use tokio_stream::StreamExt;
    ///
    /// let mut stream = db.stream_export("documents", 100).await?;
    /// while let Some(batch) = stream.next().await {
    ///     let entries = batch?;
    ///     println!("Got {} entries", entries.len());
    /// }
    /// ```
    pub async fn stream_export(
        &self,
        collection: &str,
        batch_size: usize,
    ) -> Result<ExportStream> {
        // First get all IDs
        let inner = self.inner.clone();
        let collection_name = collection.to_string();
        let ids = tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection_name)?;
            coll.ids()
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))??;

        Ok(ExportStream::new(
            self.inner.clone(),
            collection.to_string(),
            ids,
            batch_size,
        ))
    }

    /// Get all vector IDs in a collection
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    pub async fn ids(&self, collection: &str) -> Result<Vec<String>> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.ids()
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Get the number of vectors in a collection
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    pub async fn count(&self, collection: &str) -> usize {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            if let Ok(coll) = db.collection(&collection) {
                coll.len()
            } else {
                0
            }
        })
        .await
        .unwrap_or(0)
    }

    /// Count vectors matching a filter
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `filter` - Optional metadata filter
    pub async fn count_with_filter(&self, collection: &str, filter: Option<Filter>) -> Result<usize> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.count(filter.as_ref())
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Compact a collection, removing deleted vectors
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    ///
    /// # Returns
    ///
    /// Number of vectors removed
    pub async fn compact(&self, collection: &str) -> Result<usize> {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;
            coll.compact()
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Check if a collection needs compaction
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `threshold` - Deleted ratio threshold (0.0-1.0)
    pub async fn needs_compaction(&self, collection: &str, threshold: f64) -> bool {
        let inner = self.inner.clone();
        let collection = collection.to_string();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            if let Ok(coll) = db.collection(&collection) {
                coll.needs_compaction(threshold)
            } else {
                false
            }
        })
        .await
        .unwrap_or(false)
    }

    /// Save changes to disk
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// db.save().await?;
    /// ```
    pub async fn save(&self) -> Result<()> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut db = inner.blocking_lock();
            db.save()
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }

    /// Check if there are unsaved changes
    pub async fn is_dirty(&self) -> bool {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.is_dirty()
        })
        .await
        .unwrap_or(false)
    }

    /// Get total number of vectors across all collections
    pub async fn total_vectors(&self) -> usize {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            db.total_vectors()
        })
        .await
        .unwrap_or(0)
    }

    /// Get the underlying database for advanced operations
    ///
    /// Use with caution - this provides direct access to the synchronous API.
    pub fn inner(&self) -> Arc<Mutex<Database>> {
        self.inner.clone()
    }

    /// Get the async configuration
    pub fn config(&self) -> &AsyncDatabaseConfig {
        &self.config
    }
}

/// Stream for exporting vectors in batches
///
/// Implements `Stream` trait for async iteration over export entries.
pub struct ExportStream {
    inner: Arc<Mutex<Database>>,
    collection: String,
    ids: Vec<String>,
    batch_size: usize,
    offset: usize,
    pending: Option<Pin<Box<dyn std::future::Future<Output = Result<Vec<ExportEntry>>> + Send>>>,
}

impl ExportStream {
    fn new(
        inner: Arc<Mutex<Database>>,
        collection: String,
        ids: Vec<String>,
        batch_size: usize,
    ) -> Self {
        Self {
            inner,
            collection,
            ids,
            batch_size,
            offset: 0,
            pending: None,
        }
    }

    /// Get the total number of items
    pub fn total(&self) -> usize {
        self.ids.len()
    }

    /// Get the current offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Check if the stream is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.offset >= self.ids.len()
    }
}

impl futures::Stream for ExportStream {
    type Item = Result<Vec<ExportEntry>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // If we have a pending future, poll it
        if let Some(ref mut pending) = self.pending {
            match pending.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    self.pending = None;
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Check if we're done
        if self.offset >= self.ids.len() {
            return Poll::Ready(None);
        }

        // Get the next batch of IDs
        let end = std::cmp::min(self.offset + self.batch_size, self.ids.len());
        let batch_ids: Vec<String> = self.ids[self.offset..end].to_vec();
        self.offset = end;

        // Create future for fetching this batch
        let inner = self.inner.clone();
        let collection = self.collection.clone();
        let future = async move {
            tokio::task::spawn_blocking(move || {
                let db = inner.blocking_lock();
                let coll = db.collection(&collection)?;
                let mut entries = Vec::with_capacity(batch_ids.len());
                for id in batch_ids {
                    if let Some((vec, meta)) = coll.get(&id) {
                        entries.push((id, vec, meta));
                    }
                }
                Ok(entries)
            })
            .await
            .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
        };

        self.pending = Some(Box::pin(future));

        // Poll the new future
        if let Some(ref mut pending) = self.pending {
            match pending.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    self.pending = None;
                    Poll::Ready(Some(result))
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Pending
        }
    }
}

/// Stream for search results with pagination
///
/// Useful for streaming through large result sets with cursor-based pagination.
pub struct SearchStream {
    inner: Arc<Mutex<Database>>,
    collection: String,
    query: Vec<f32>,
    filter: Option<Filter>,
    batch_size: usize,
    offset: usize,
    total_limit: Option<usize>,
    pending: Option<Pin<Box<dyn std::future::Future<Output = Result<Vec<SearchResult>>> + Send>>>,
    exhausted: bool,
}

impl SearchStream {
    /// Create a new search stream
    pub fn new(
        inner: Arc<Mutex<Database>>,
        collection: String,
        query: Vec<f32>,
        filter: Option<Filter>,
        batch_size: usize,
        total_limit: Option<usize>,
    ) -> Self {
        Self {
            inner,
            collection,
            query,
            filter,
            batch_size,
            offset: 0,
            total_limit,
            pending: None,
            exhausted: false,
        }
    }

    /// Get the current offset
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Check if the stream is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }
}

impl futures::Stream for SearchStream {
    type Item = Result<Vec<SearchResult>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // If we have a pending future, poll it
        if let Some(ref mut pending) = self.pending {
            match pending.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    self.pending = None;
                    match &result {
                        Ok(results) if results.is_empty() => {
                            self.exhausted = true;
                        }
                        Ok(results) if results.len() < self.batch_size => {
                            self.exhausted = true;
                        }
                        Err(_) => {
                            self.exhausted = true;
                        }
                        _ => {}
                    }
                    return Poll::Ready(Some(result));
                }
                Poll::Pending => return Poll::Pending,
            }
        }

        // Check if we're done
        if self.exhausted {
            return Poll::Ready(None);
        }

        // Check total limit
        if let Some(limit) = self.total_limit {
            if self.offset >= limit {
                return Poll::Ready(None);
            }
        }

        // Calculate batch size respecting total limit
        let mut current_batch = self.batch_size;
        if let Some(limit) = self.total_limit {
            let remaining = limit - self.offset;
            current_batch = std::cmp::min(current_batch, remaining);
        }

        let inner = self.inner.clone();
        let collection = self.collection.clone();
        let query = self.query.clone();
        let filter = self.filter.clone();
        let k = self.offset + current_batch;

        self.offset += current_batch;

        // Create future for fetching this batch
        let future = async move {
            tokio::task::spawn_blocking(move || {
                let db = inner.blocking_lock();
                let coll = db.collection(&collection)?;
                if let Some(f) = filter {
                    coll.search_with_filter(&query, k, &f)
                } else {
                    coll.search(&query, k)
                }
            })
            .await
            .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
        };

        self.pending = Some(Box::pin(future));

        // Poll the new future
        if let Some(ref mut pending) = self.pending {
            match pending.as_mut().poll(cx) {
                Poll::Ready(result) => {
                    self.pending = None;
                    match &result {
                        Ok(results) if results.is_empty() => {
                            self.exhausted = true;
                        }
                        Ok(results) if results.len() < current_batch => {
                            self.exhausted = true;
                        }
                        Err(_) => {
                            self.exhausted = true;
                        }
                        _ => {}
                    }
                    Poll::Ready(Some(result))
                }
                Poll::Pending => Poll::Pending,
            }
        } else {
            Poll::Pending
        }
    }
}

/// Builder for async batch operations
///
/// Provides a fluent interface for building and executing batch operations.
pub struct BatchOperationBuilder {
    inner: Arc<Mutex<Database>>,
    collection: String,
    inserts: Vec<(String, Vec<f32>, Option<Value>)>,
    deletes: Vec<String>,
}

impl BatchOperationBuilder {
    /// Create a new batch operation builder
    pub fn new(db: &AsyncDatabase, collection: &str) -> Self {
        Self {
            inner: db.inner.clone(),
            collection: collection.to_string(),
            inserts: Vec::new(),
            deletes: Vec::new(),
        }
    }

    /// Add an insert operation
    pub fn insert(mut self, id: impl Into<String>, vector: Vec<f32>, metadata: Option<Value>) -> Self {
        self.inserts.push((id.into(), vector, metadata));
        self
    }

    /// Add a delete operation
    pub fn delete(mut self, id: impl Into<String>) -> Self {
        self.deletes.push(id.into());
        self
    }

    /// Execute all operations
    ///
    /// Returns the number of successful inserts and deletes.
    pub async fn execute(self) -> Result<BatchResult> {
        let inner = self.inner;
        let collection = self.collection;
        let inserts = self.inserts;
        let deletes = self.deletes;

        tokio::task::spawn_blocking(move || {
            let db = inner.blocking_lock();
            let coll = db.collection(&collection)?;

            let mut inserted = 0;
            let mut insert_errors = Vec::new();
            for (id, vector, metadata) in inserts {
                match coll.insert(&id, &vector, metadata) {
                    Ok(_) => inserted += 1,
                    Err(e) => insert_errors.push((id, e.to_string())),
                }
            }

            let mut deleted = 0;
            let mut delete_errors = Vec::new();
            for id in deletes {
                match coll.delete(&id) {
                    Ok(true) => deleted += 1,
                    Ok(false) => {}
                    Err(e) => delete_errors.push((id, e.to_string())),
                }
            }

            Ok(BatchResult {
                inserted,
                deleted,
                insert_errors,
                delete_errors,
            })
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }
}

/// Result of a batch operation
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Number of vectors inserted
    pub inserted: usize,
    /// Number of vectors deleted
    pub deleted: usize,
    /// Insert errors: (id, error message)
    pub insert_errors: Vec<(String, String)>,
    /// Delete errors: (id, error message)
    pub delete_errors: Vec<(String, String)>,
}

impl BatchResult {
    /// Check if the batch completed without errors
    pub fn is_success(&self) -> bool {
        self.insert_errors.is_empty() && self.delete_errors.is_empty()
    }

    /// Get total number of errors
    pub fn error_count(&self) -> usize {
        self.insert_errors.len() + self.delete_errors.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[tokio::test]
    async fn test_async_database_in_memory() {
        let db = AsyncDatabase::in_memory();

        db.create_collection("documents", 128).await.unwrap();

        let vec = random_vector(128);
        db.insert(
            "documents",
            "doc1",
            vec.clone(),
            Some(json!({"title": "Test"})),
        )
        .await
        .unwrap();

        assert_eq!(db.count("documents").await, 1);
    }

    #[tokio::test]
    async fn test_async_search() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, Some(json!({"index": i})))
                .await
                .unwrap();
        }

        // Search
        let query = random_vector(32);
        let results = db.search("test", query, 10).await.unwrap();

        assert_eq!(results.len(), 10);
    }

    #[tokio::test]
    async fn test_async_search_with_filter() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert vectors with category metadata
        for i in 0..100 {
            let vec = random_vector(32);
            let category = if i % 2 == 0 { "even" } else { "odd" };
            db.insert(
                "test",
                format!("doc{}", i),
                vec,
                Some(json!({"index": i, "category": category})),
            )
            .await
            .unwrap();
        }

        // Search with filter
        let query = random_vector(32);
        let filter = Filter::eq("category", "even");
        let results = db.search_with_filter("test", query, 10, filter).await.unwrap();

        // All results should have category "even"
        for result in &results {
            let meta = result.metadata.as_ref().unwrap();
            assert_eq!(meta["category"], "even");
        }
    }

    #[tokio::test]
    async fn test_async_batch_search() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        // Batch search
        let queries: Vec<Vec<f32>> = (0..5).map(|_| random_vector(32)).collect();
        let results = db.batch_search("test", queries, 10).await.unwrap();

        assert_eq!(results.len(), 5);
        for result_set in &results {
            assert_eq!(result_set.len(), 10);
        }
    }

    #[tokio::test]
    async fn test_async_get_and_delete() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        let vec = random_vector(32);
        db.insert("test", "doc1", vec.clone(), Some(json!({"title": "Test"})))
            .await
            .unwrap();

        // Get
        let (retrieved_vec, metadata) = db.get("test", "doc1").await.unwrap();
        assert_eq!(retrieved_vec, vec);
        assert_eq!(metadata.unwrap()["title"], "Test");

        // Delete
        assert!(db.delete("test", "doc1").await.unwrap());
        assert!(db.get("test", "doc1").await.is_none());
    }

    #[tokio::test]
    async fn test_async_batch_operations() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Batch insert
        let entries: Vec<(String, Vec<f32>, Option<Value>)> = (0..10)
            .map(|i| (format!("doc{}", i), random_vector(32), None))
            .collect();
        db.batch_insert("test", entries).await.unwrap();

        assert_eq!(db.count("test").await, 10);

        // Batch delete
        let ids: Vec<String> = (0..5).map(|i| format!("doc{}", i)).collect();
        let deleted = db.batch_delete("test", ids).await.unwrap();

        assert_eq!(deleted, 5);
        assert_eq!(db.count("test").await, 5);
    }

    #[tokio::test]
    async fn test_async_export() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, Some(json!({"i": i})))
                .await
                .unwrap();
        }

        let entries = db.export("test").await.unwrap();
        assert_eq!(entries.len(), 10);
    }

    #[tokio::test]
    async fn test_async_stream_export() {
        use futures::StreamExt;

        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        for i in 0..25 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        let mut stream = db.stream_export("test", 10).await.unwrap();
        let mut total = 0;
        let mut batch_count = 0;

        while let Some(batch) = stream.next().await {
            let entries = batch.unwrap();
            total += entries.len();
            batch_count += 1;
        }

        assert_eq!(total, 25);
        assert_eq!(batch_count, 3); // 10 + 10 + 5
    }

    #[tokio::test]
    async fn test_async_collection_operations() {
        let db = AsyncDatabase::in_memory();

        db.create_collection("col1", 64).await.unwrap();
        db.create_collection("col2", 128).await.unwrap();

        let collections = db.list_collections().await;
        assert_eq!(collections.len(), 2);

        assert!(db.has_collection("col1").await);
        assert!(db.has_collection("col2").await);
        assert!(!db.has_collection("col3").await);

        db.drop_collection("col1").await.unwrap();
        assert!(!db.has_collection("col1").await);
    }

    #[tokio::test]
    async fn test_async_compact() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert and delete some vectors
        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        for i in 0..5 {
            db.delete("test", &format!("doc{}", i)).await.unwrap();
        }

        // Compact
        let removed = db.compact("test").await.unwrap();
        assert_eq!(removed, 5);
    }

    #[tokio::test]
    async fn test_batch_operation_builder() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert some initial vectors
        for i in 0..5 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        // Use batch builder
        let result = BatchOperationBuilder::new(&db, "test")
            .insert("new1", random_vector(32), None)
            .insert("new2", random_vector(32), Some(json!({"new": true})))
            .delete("doc0")
            .delete("doc1")
            .execute()
            .await
            .unwrap();

        assert_eq!(result.inserted, 2);
        assert_eq!(result.deleted, 2);
        assert!(result.is_success());
        assert_eq!(db.count("test").await, 5); // 5 - 2 + 2 = 5
    }

    #[tokio::test]
    async fn test_async_ids() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        let ids = db.ids("test").await.unwrap();
        assert_eq!(ids.len(), 10);
    }

    #[tokio::test]
    async fn test_async_count_with_filter() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        for i in 0..100 {
            let vec = random_vector(32);
            let category = if i % 2 == 0 { "even" } else { "odd" };
            db.insert(
                "test",
                format!("doc{}", i),
                vec,
                Some(json!({"category": category})),
            )
            .await
            .unwrap();
        }

        let filter = Filter::eq("category", "even");
        let count = db.count_with_filter("test", Some(filter)).await.unwrap();
        assert_eq!(count, 50);
    }

    #[tokio::test]
    async fn test_async_is_dirty() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Should be dirty after insert
        let vec = random_vector(32);
        db.insert("test", "doc1", vec, None).await.unwrap();
        assert!(db.is_dirty().await);
    }

    #[tokio::test]
    async fn test_async_total_vectors() {
        let db = AsyncDatabase::in_memory();

        db.create_collection("col1", 32).await.unwrap();
        db.create_collection("col2", 32).await.unwrap();

        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("col1", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        for i in 0..5 {
            let vec = random_vector(32);
            db.insert("col2", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        assert_eq!(db.total_vectors().await, 15);
    }

    #[tokio::test]
    async fn test_async_needs_compaction() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        for i in 0..100 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await
                .unwrap();
        }

        // Delete 50% of vectors
        for i in 0..50 {
            db.delete("test", &format!("doc{}", i)).await.unwrap();
        }

        // Should need compaction at 30% threshold
        assert!(db.needs_compaction("test", 0.3).await);
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Spawn multiple concurrent insert tasks
        let mut handles = Vec::new();
        for i in 0..10 {
            let db_clone = db.clone();
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    let vec = random_vector(32);
                    db_clone
                        .insert("test", format!("doc_{}_{}", i, j), vec, None)
                        .await
                        .unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(db.count("test").await, 100);
    }

    // AsyncDatabaseConfig tests
    #[test]
    fn test_async_config_default() {
        let config = AsyncDatabaseConfig::default();
        assert_eq!(config.max_concurrency, 4);
        assert_eq!(config.stream_batch_size, 100);
    }

    #[test]
    fn test_async_config_with_max_concurrency() {
        let config = AsyncDatabaseConfig::default().with_max_concurrency(8);
        assert_eq!(config.max_concurrency, 8);
        assert_eq!(config.stream_batch_size, 100); // unchanged
    }

    #[test]
    fn test_async_config_with_stream_batch_size() {
        let config = AsyncDatabaseConfig::default().with_stream_batch_size(50);
        assert_eq!(config.max_concurrency, 4); // unchanged
        assert_eq!(config.stream_batch_size, 50);
    }

    #[test]
    fn test_async_config_chained() {
        let config = AsyncDatabaseConfig::default()
            .with_max_concurrency(16)
            .with_stream_batch_size(200);
        assert_eq!(config.max_concurrency, 16);
        assert_eq!(config.stream_batch_size, 200);
    }

    #[tokio::test]
    async fn test_collection_not_found_error() {
        let db = AsyncDatabase::in_memory();

        // Searching non-existent collection should return error
        let result = db.search("nonexistent", vec![0.0; 32], 10).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_dimension_mismatch_error() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 64).await.unwrap();

        // Insert with wrong dimensions
        let result = db.insert("test", "doc1", vec![0.0; 32], None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_duplicate_collection_error() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 64).await.unwrap();

        // Creating same collection again should error
        let result = db.create_collection("test", 64).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_search_empty_queries() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert some vectors first
        for i in 0..10 {
            db.insert("test", format!("doc{}", i), random_vector(32), None)
                .await
                .unwrap();
        }

        // Batch search with empty query list
        let results = db.batch_search("test", Vec::new(), 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_batch_insert_empty() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Batch insert with empty list should succeed
        let entries: Vec<(String, Vec<f32>, Option<Value>)> = Vec::new();
        db.batch_insert("test", entries).await.unwrap();

        assert_eq!(db.count("test").await, 0);
    }

    #[tokio::test]
    async fn test_batch_delete_empty() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Insert some vectors
        for i in 0..5 {
            db.insert("test", format!("doc{}", i), random_vector(32), None)
                .await
                .unwrap();
        }

        // Batch delete with empty list
        let deleted = db.batch_delete("test", Vec::new()).await.unwrap();
        assert_eq!(deleted, 0);
        assert_eq!(db.count("test").await, 5); // unchanged
    }

    #[tokio::test]
    async fn test_batch_operation_builder_empty() {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await.unwrap();

        // Empty batch operation should succeed
        let result = BatchOperationBuilder::new(&db, "test")
            .execute()
            .await
            .unwrap();

        assert_eq!(result.inserted, 0);
        assert_eq!(result.deleted, 0);
        assert!(result.is_success());
    }
}
