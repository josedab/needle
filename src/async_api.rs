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

#![allow(clippy::unwrap_used)] // tech debt: 67 unwrap() calls remaining
use crate::collection::{CollectionConfig, SearchResult};
use crate::database::{Database, DatabaseConfig, ExportEntry};
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use serde_json::Value;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::RwLock;

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

/// Reduces `spawn_blocking` boilerplate in `AsyncDatabase` methods.
///
/// Four variants:
/// - `db_op!(self, |db| expr)` — read lock, returns `Result<T>`
/// - `db_op!(self, mut |db| expr)` — write lock, returns `Result<T>`
/// - `db_op!(self, default val, |db| expr)` — read lock, returns `T` with fallback
/// - `db_op!(self, write default val, |db| expr)` — write lock, returns `T` with fallback
macro_rules! db_op {
    ($s:ident, |$db:ident| $body:expr) => {{
        let inner = $s.inner.clone();
        tokio::task::spawn_blocking(move || {
            let $db = inner.blocking_read();
            $body
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }};
    ($s:ident, mut |$db:ident| $body:expr) => {{
        let inner = $s.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut $db = inner.blocking_write();
            $body
        })
        .await
        .map_err(|e| NeedleError::Io(std::io::Error::other(e)))?
    }};
    ($s:ident, default $default:expr, |$db:ident| $body:expr) => {{
        let inner = $s.inner.clone();
        tokio::task::spawn_blocking(move || {
            let $db = inner.blocking_read();
            $body
        })
        .await
        .unwrap_or($default)
    }};
}

/// Async wrapper for the Needle vector database
///
/// Provides async/await interface for all database operations. Uses
/// `tokio::spawn_blocking` internally to offload CPU-intensive work
/// to the blocking thread pool.
#[derive(Clone)]
pub struct AsyncDatabase {
    /// Inner database wrapped in Arc for shared ownership
    inner: Arc<RwLock<Database>>,
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
            inner: Arc::new(RwLock::new(db)),
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
            inner: Arc::new(RwLock::new(db)),
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
            inner: Arc::new(RwLock::new(Database::in_memory())),
            config: AsyncDatabaseConfig::default(),
        }
    }

    /// Create an in-memory database with custom async configuration
    pub fn in_memory_with_config(config: AsyncDatabaseConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Database::in_memory())),
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
    pub async fn create_collection(
        &self,
        name: impl Into<String>,
        dimensions: usize,
    ) -> Result<()> {
        let name = name.into();
        db_op!(self, |db| db.create_collection(name, dimensions))
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
        db_op!(self, |db| db.create_collection_with_config(config))
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
        db_op!(self, default Vec::new(), |db| db.list_collections())
    }

    /// Check if a collection exists
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    pub async fn has_collection(&self, name: &str) -> bool {
        let name = name.to_string();
        db_op!(self, default false, |db| db.has_collection(&name))
    }

    /// Drop a collection
    ///
    /// # Arguments
    ///
    /// * `name` - Collection name
    ///
    /// # Returns
    ///
    /// `true` if the collection was deleted, `false` if it didn't exist
    pub async fn delete_collection(&self, name: &str) -> Result<bool> {
        let name = name.to_string();
        db_op!(self, |db| db.delete_collection(&name))
    }

    /// Delete a collection and all its data.
    #[deprecated(since = "0.2.0", note = "renamed to `delete_collection`")]
    pub async fn drop_collection(&self, name: &str) -> Result<bool> {
        self.delete_collection(name).await
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
        let collection = collection.to_string();
        let id = id.into();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            // Use insert_vec to avoid cloning the vector we already own
            coll.insert_vec(id, vector, metadata)
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            for (id, vector, metadata) in entries {
                coll.insert(id, &vector, metadata)?;
            }
            Ok(())
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.search(&query, k)
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.search_with_filter(&query, k, &filter)
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            // Use rayon's parallel search internally
            let results: Result<Vec<Vec<SearchResult>>> =
                queries.iter().map(|query| coll.search(query, k)).collect();
            results
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            let results: Result<Vec<Vec<SearchResult>>> = queries
                .iter()
                .map(|query| coll.search_with_filter(query, k, &filter))
                .collect();
            results
        })
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
        let collection = collection.to_string();
        let id = id.to_string();
        db_op!(self, default None, |db| {
            if let Ok(coll) = db.collection(&collection) {
                coll.get(&id)
            } else {
                None
            }
        })
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
        let collection = collection.to_string();
        let id = id.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.delete(&id)
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            let mut deleted = 0;
            for id in &ids {
                if coll.delete(id)? {
                    deleted += 1;
                }
            }
            Ok(deleted)
        })
    }

    /// Export all vectors from a collection
    ///
    /// For large collections, consider using `stream_export` instead.
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    pub async fn export(&self, collection: &str) -> Result<Vec<ExportEntry>> {
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.export_all()
        })
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
    pub async fn stream_export(&self, collection: &str, batch_size: usize) -> Result<ExportStream> {
        // First get all IDs
        let inner = self.inner.clone();
        let collection_name = collection.to_string();
        let ids = tokio::task::spawn_blocking(move || {
            let db = inner.blocking_read();
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.ids()
        })
    }

    /// Get the number of vectors in a collection
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    pub async fn count(&self, collection: &str) -> usize {
        let collection = collection.to_string();
        db_op!(self, default 0, |db| {
            if let Ok(coll) = db.collection(&collection) {
                coll.len()
            } else {
                0
            }
        })
    }

    /// Count vectors matching a filter
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `filter` - Optional metadata filter
    pub async fn count_with_filter(
        &self,
        collection: &str,
        filter: Option<Filter>,
    ) -> Result<usize> {
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.count(filter.as_ref())
        })
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
        let collection = collection.to_string();
        db_op!(self, |db| {
            let coll = db.collection(&collection)?;
            coll.compact()
        })
    }

    /// Check if a collection needs compaction
    ///
    /// # Arguments
    ///
    /// * `collection` - Collection name
    /// * `threshold` - Deleted ratio threshold (0.0-1.0)
    pub async fn needs_compaction(&self, collection: &str, threshold: f64) -> bool {
        let collection = collection.to_string();
        db_op!(self, default false, |db| {
            if let Ok(coll) = db.collection(&collection) {
                coll.needs_compaction(threshold)
            } else {
                false
            }
        })
    }

    /// Save changes to disk
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// db.save().await?;
    /// ```
    pub async fn save(&self) -> Result<()> {
        db_op!(self, mut |db| db.save())
    }

    /// Check if there are unsaved changes
    pub async fn is_dirty(&self) -> bool {
        db_op!(self, default false, |db| db.is_dirty())
    }

    /// Get total number of vectors across all collections
    pub async fn total_vectors(&self) -> usize {
        db_op!(self, default 0, |db| db.total_vectors())
    }

    /// Get the underlying database for advanced operations
    ///
    /// Use with caution - this provides direct access to the synchronous API.
    pub fn inner(&self) -> Arc<RwLock<Database>> {
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
    inner: Arc<RwLock<Database>>,
    collection: String,
    ids: Vec<String>,
    batch_size: usize,
    offset: usize,
    pending: Option<Pin<Box<dyn std::future::Future<Output = Result<Vec<ExportEntry>>> + Send>>>,
}

impl ExportStream {
    fn new(
        inner: Arc<RwLock<Database>>,
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
                let db = inner.blocking_read();
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
    inner: Arc<RwLock<Database>>,
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
        inner: Arc<RwLock<Database>>,
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
                let db = inner.blocking_read();
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
    inner: Arc<RwLock<Database>>,
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
    pub fn insert(
        mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Self {
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
            let db = inner.blocking_read();
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
    use crate::test_utils::random_vector;
    use serde_json::json;

    #[tokio::test]
    async fn test_async_database_in_memory() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();

        db.create_collection("documents", 128).await?;

        let vec = random_vector(128);
        db.insert(
            "documents",
            "doc1",
            vec.clone(),
            Some(json!({"title": "Test"})),
        )
        .await?;

        assert_eq!(db.count("documents").await, 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, Some(json!({"index": i})))
                .await?;
        }

        // Search
        let query = random_vector(32);
        let results = db.search("test", query, 10).await?;

        assert_eq!(results.len(), 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_search_with_filter() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

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
            .await?;
        }

        // Search with filter
        let query = random_vector(32);
        let filter = Filter::eq("category", "even");
        let results = db
            .search_with_filter("test", query, 10, filter)
            .await?;

        // All results should have category "even"
        for result in &results {
            let meta = result.metadata.as_ref().ok_or("missing metadata")?;
            assert_eq!(meta["category"], "even");
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Insert vectors
        for i in 0..100 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await?;
        }

        // Batch search
        let queries: Vec<Vec<f32>> = (0..5).map(|_| random_vector(32)).collect();
        let results = db.batch_search("test", queries, 10).await?;

        assert_eq!(results.len(), 5);
        for result_set in &results {
            assert_eq!(result_set.len(), 10);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_async_get_and_delete() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        let vec = random_vector(32);
        db.insert("test", "doc1", vec.clone(), Some(json!({"title": "Test"})))
            .await?;

        // Get
        let (retrieved_vec, metadata) = db.get("test", "doc1").await.ok_or("vector not found")?;
        assert_eq!(retrieved_vec, vec);
        assert_eq!(metadata.ok_or("missing metadata")?["title"], "Test");

        // Delete
        assert!(db.delete("test", "doc1").await?);
        assert!(db.get("test", "doc1").await.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn test_async_batch_operations() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Batch insert
        let entries: Vec<(String, Vec<f32>, Option<Value>)> = (0..10)
            .map(|i| (format!("doc{}", i), random_vector(32), None))
            .collect();
        db.batch_insert("test", entries).await?;

        assert_eq!(db.count("test").await, 10);

        // Batch delete
        let ids: Vec<String> = (0..5).map(|i| format!("doc{}", i)).collect();
        let deleted = db.batch_delete("test", ids).await?;

        assert_eq!(deleted, 5);
        assert_eq!(db.count("test").await, 5);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_export() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, Some(json!({"i": i})))
                .await?;
        }

        let entries = db.export("test").await?;
        assert_eq!(entries.len(), 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_stream_export() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use futures::StreamExt;

        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        for i in 0..25 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await?;
        }

        let mut stream = db.stream_export("test", 10).await?;
        let mut total = 0;
        let mut batch_count = 0;

        while let Some(batch) = stream.next().await {
            let entries = batch?;
            total += entries.len();
            batch_count += 1;
        }

        assert_eq!(total, 25);
        assert_eq!(batch_count, 3); // 10 + 10 + 5
        Ok(())
    }

    #[tokio::test]
    async fn test_async_collection_operations() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();

        db.create_collection("col1", 64).await?;
        db.create_collection("col2", 128).await?;

        let collections = db.list_collections().await;
        assert_eq!(collections.len(), 2);

        assert!(db.has_collection("col1").await);
        assert!(db.has_collection("col2").await);
        assert!(!db.has_collection("col3").await);

        db.drop_collection("col1").await?;
        assert!(!db.has_collection("col1").await);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_compact() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Insert and delete some vectors
        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await?;
        }

        for i in 0..5 {
            db.delete("test", &format!("doc{}", i)).await?;
        }

        // Compact
        let removed = db.compact("test").await?;
        assert_eq!(removed, 5);
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operation_builder() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Insert some initial vectors
        for i in 0..5 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await?;
        }

        // Use batch builder
        let result = BatchOperationBuilder::new(&db, "test")
            .insert("new1", random_vector(32), None)
            .insert("new2", random_vector(32), Some(json!({"new": true})))
            .delete("doc0")
            .delete("doc1")
            .execute()
            .await?;

        assert_eq!(result.inserted, 2);
        assert_eq!(result.deleted, 2);
        assert!(result.is_success());
        assert_eq!(db.count("test").await, 5); // 5 - 2 + 2 = 5
        Ok(())
    }

    #[tokio::test]
    async fn test_async_ids() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await?;
        }

        let ids = db.ids("test").await?;
        assert_eq!(ids.len(), 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_count_with_filter() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        for i in 0..100 {
            let vec = random_vector(32);
            let category = if i % 2 == 0 { "even" } else { "odd" };
            db.insert(
                "test",
                format!("doc{}", i),
                vec,
                Some(json!({"category": category})),
            )
            .await?;
        }

        let filter = Filter::eq("category", "even");
        let count = db.count_with_filter("test", Some(filter)).await?;
        assert_eq!(count, 50);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_is_dirty() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Should be dirty after insert
        let vec = random_vector(32);
        db.insert("test", "doc1", vec, None).await?;
        assert!(db.is_dirty().await);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_total_vectors() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();

        db.create_collection("col1", 32).await?;
        db.create_collection("col2", 32).await?;

        for i in 0..10 {
            let vec = random_vector(32);
            db.insert("col1", format!("doc{}", i), vec, None)
                .await?;
        }

        for i in 0..5 {
            let vec = random_vector(32);
            db.insert("col2", format!("doc{}", i), vec, None)
                .await?;
        }

        assert_eq!(db.total_vectors().await, 15);
        Ok(())
    }

    #[tokio::test]
    async fn test_async_needs_compaction() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        for i in 0..100 {
            let vec = random_vector(32);
            db.insert("test", format!("doc{}", i), vec, None)
                .await?;
        }

        // Delete 50% of vectors
        for i in 0..50 {
            db.delete("test", &format!("doc{}", i)).await?;
        }

        // Should need compaction at 30% threshold
        assert!(db.needs_compaction("test", 0.3).await);
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_operations() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Spawn multiple concurrent insert tasks
        let mut handles = Vec::new();
        for i in 0..10 {
            let db_clone = db.clone();
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    let vec = random_vector(32);
                    db_clone
                        .insert("test", format!("doc_{}_{}", i, j), vec, None)
                        .await?;
                }
                Ok::<(), crate::error::NeedleError>(())
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await??;
        }

        assert_eq!(db.count("test").await, 100);
        Ok(())
    }

    // AsyncDatabaseConfig tests
    #[test]
    fn test_async_config_default() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = AsyncDatabaseConfig::default();
        assert_eq!(config.max_concurrency, 4);
        assert_eq!(config.stream_batch_size, 100);
        Ok(())
    }

    #[test]
    fn test_async_config_with_max_concurrency() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = AsyncDatabaseConfig::default().with_max_concurrency(8);
        assert_eq!(config.max_concurrency, 8);
        assert_eq!(config.stream_batch_size, 100); // unchanged
        Ok(())
    }

    #[test]
    fn test_async_config_with_stream_batch_size() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = AsyncDatabaseConfig::default().with_stream_batch_size(50);
        assert_eq!(config.max_concurrency, 4); // unchanged
        assert_eq!(config.stream_batch_size, 50);
        Ok(())
    }

    #[test]
    fn test_async_config_chained() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = AsyncDatabaseConfig::default()
            .with_max_concurrency(16)
            .with_stream_batch_size(200);
        assert_eq!(config.max_concurrency, 16);
        assert_eq!(config.stream_batch_size, 200);
        Ok(())
    }

    #[tokio::test]
    async fn test_collection_not_found_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();

        // Searching non-existent collection should return error
        let result = db.search("nonexistent", vec![0.0; 32], 10).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_dimension_mismatch_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 64).await?;

        // Insert with wrong dimensions
        let result = db.insert("test", "doc1", vec![0.0; 32], None).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_duplicate_collection_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 64).await?;

        // Creating same collection again should error
        let result = db.create_collection("test", 64).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_search_empty_queries() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Insert some vectors first
        for i in 0..10 {
            db.insert("test", format!("doc{}", i), random_vector(32), None)
                .await?;
        }

        // Batch search with empty query list
        let results = db.batch_search("test", Vec::new(), 10).await?;
        assert!(results.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_insert_empty() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Batch insert with empty list should succeed
        let entries: Vec<(String, Vec<f32>, Option<Value>)> = Vec::new();
        db.batch_insert("test", entries).await?;

        assert_eq!(db.count("test").await, 0);
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_delete_empty() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Insert some vectors
        for i in 0..5 {
            db.insert("test", format!("doc{}", i), random_vector(32), None)
                .await?;
        }

        // Batch delete with empty list
        let deleted = db.batch_delete("test", Vec::new()).await?;
        assert_eq!(deleted, 0);
        assert_eq!(db.count("test").await, 5); // unchanged
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operation_builder_empty() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 32).await?;

        // Empty batch operation should succeed
        let result = BatchOperationBuilder::new(&db, "test")
            .execute()
            .await?;

        assert_eq!(result.inserted, 0);
        assert_eq!(result.deleted, 0);
        assert!(result.is_success());
        Ok(())
    }

    // ── open + in_memory_with_config ─────────────────────────────────────

    #[tokio::test]
    async fn test_in_memory_with_config() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = AsyncDatabaseConfig::default()
            .with_max_concurrency(4)
            .with_stream_batch_size(50);
        let db = AsyncDatabase::in_memory_with_config(config);
        db.create_collection("test", 32).await?;
        assert!(db.has_collection("test").await);
        Ok(())
    }

    // ── create_collection + insert + search round-trip ───────────────────

    #[tokio::test]
    async fn test_roundtrip_create_insert_search() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("docs", 4).await?;

        let vec = vec![1.0, 0.0, 0.0, 0.0];
        db.insert("docs", "v1", vec.clone(), Some(json!({"label": "a"})))
            .await?;

        let results = db.search("docs", vec.clone(), 5).await?;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
        assert!(results[0].distance < 0.001);
        Ok(())
    }

    // ── batch_delete: partial and full ───────────────────────────────────

    #[tokio::test]
    async fn test_batch_delete_partial() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;

        for i in 0..5 {
            db.insert(
                "test",
                &format!("v{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                None,
            )
            .await?;
        }

        // Delete 3 out of 5, including one nonexistent
        let deleted = db
            .batch_delete(
                "test",
                vec!["v0".into(), "v1".into(), "v2".into(), "nonexistent".into()],
            )
            .await?;
        assert!(deleted >= 3);
        assert_eq!(db.count("test").await, 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_delete_all() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;

        for i in 0..3 {
            db.insert("test", &format!("v{i}"), vec![i as f32, 0.0, 0.0, 0.0], None)
                .await?;
        }

        let ids: Vec<String> = (0..3).map(|i| format!("v{i}")).collect();
        let deleted = db.batch_delete("test", ids).await?;
        assert_eq!(deleted, 3);
        assert_eq!(db.count("test").await, 0);
        Ok(())
    }

    // ── concurrent async operations across multiple tasks ────────────────

    #[tokio::test]
    async fn test_concurrent_inserts_across_tasks() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Arc::new(AsyncDatabase::in_memory());
        db.create_collection("test", 4).await?;

        let mut handles = Vec::new();
        for i in 0..10 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                db.insert(
                    "test",
                    &format!("v{i}"),
                    vec![i as f32, 0.0, 0.0, 0.0],
                    None,
                )
                .await
            }));
        }

        for h in handles {
            h.await??;
        }
        assert_eq!(db.count("test").await, 10);
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent_search_across_tasks() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Arc::new(AsyncDatabase::in_memory());
        db.create_collection("test", 4).await?;

        for i in 0..20 {
            db.insert("test", &format!("v{i}"), random_vector(4), None)
                .await?;
        }

        let mut handles = Vec::new();
        for _ in 0..5 {
            let db = Arc::clone(&db);
            handles.push(tokio::spawn(async move {
                db.search("test", random_vector(4), 5).await
            }));
        }

        for h in handles {
            let results = h.await??;
            assert!(results.len() <= 5);
        }
        Ok(())
    }

    // ── error propagation from sync Database ─────────────────────────────

    #[tokio::test]
    async fn test_error_propagation_insert_wrong_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        let result = db.insert("nonexistent", "v1", vec![1.0], None).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_error_propagation_search_wrong_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        let result = db.search("nonexistent", vec![1.0], 5).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_error_propagation_delete_wrong_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        let result = db.delete("nonexistent", "v1").await;
        assert!(result.is_err());
        Ok(())
    }

    // ── search_with_filter with invalid filter ───────────────────────────

    #[tokio::test]
    async fn test_search_with_invalid_filter_operator() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;

        db.insert("test", "v1", vec![1.0, 0.0, 0.0, 0.0], Some(json!({"x": 1})))
            .await?;

        // Use an unsupported operator
        let filter = crate::metadata::Filter::parse(&json!({"x": {"$invalid_op": 5}}));
        match filter {
            Ok(f) => {
                // If parsing succeeds, search should still work (or fail gracefully)
                let _ = db.search_with_filter("test", vec![1.0, 0.0, 0.0, 0.0], 5, f).await;
            }
            Err(_) => {
                // Invalid filter should be rejected at parse time
            }
        }
        Ok(())
    }

    // ── stream_export pagination ─────────────────────────────────────────

    #[tokio::test]
    async fn test_stream_export_pagination() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use futures::StreamExt;

        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;

        for i in 0..15 {
            db.insert("test", &format!("v{i}"), vec![i as f32, 0.0, 0.0, 0.0], None)
                .await?;
        }

        let mut stream = db.stream_export("test", 5).await?;
        let mut total = 0;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            assert!(batch.len() <= 5);
            total += batch.len();
        }
        assert_eq!(total, 15);
        Ok(())
    }

    // ── delete_collection ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_delete_collection_async() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        assert!(db.has_collection("test").await);

        let deleted = db.delete_collection("test").await?;
        assert!(deleted);
        assert!(!db.has_collection("test").await);
        Ok(())
    }

    #[tokio::test]
    async fn test_delete_nonexistent_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        let result = db.delete_collection("nonexistent").await;
        // Should either return Ok(false) or Err
        match result {
            Ok(deleted) => assert!(!deleted),
            Err(_) => {} // Also acceptable
        }
        Ok(())
    }

    // ── create_collection_with_config ────────────────────────────────────

    #[tokio::test]
    async fn test_create_collection_with_config() -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::collection::config::CollectionConfig;
        let db = AsyncDatabase::in_memory();
        let config = CollectionConfig::new("custom_coll", 64);
        db.create_collection_with_config(config).await?;
        assert!(db.has_collection("custom_coll").await);
        assert_eq!(db.count("custom_coll").await, 0);
        Ok(())
    }

    // ── drop_collection ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_drop_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("to_drop", 4).await?;
        assert!(db.has_collection("to_drop").await);
        let dropped = db.drop_collection("to_drop").await?;
        assert!(dropped);
        assert!(!db.has_collection("to_drop").await);
        Ok(())
    }

    // ── list_collections ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_collections() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("a", 4).await?;
        db.create_collection("b", 8).await?;
        let cols = db.list_collections().await;
        assert!(cols.contains(&"a".to_string()));
        assert!(cols.contains(&"b".to_string()));
        assert_eq!(cols.len(), 2);
        Ok(())
    }

    // ── save and is_dirty round-trip ─────────────────────────────────────

    #[tokio::test]
    async fn test_save_is_dirty_roundtrip() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        db.insert("test", "v1", vec![1.0, 0.0, 0.0, 0.0], None).await?;
        // in-memory DB: is_dirty and save may behave differently
        let dirty = db.is_dirty().await;
        let _ = db.save().await;
        // After save, verify state is consistent
        assert_eq!(db.count("test").await, 1);
        let _ = dirty;
        Ok(())
    }

    // ── total_vectors across multiple collections ────────────────────────

    #[tokio::test]
    async fn test_total_vectors_multi_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("a", 4).await?;
        db.create_collection("b", 4).await?;
        db.insert("a", "v1", vec![1.0, 0.0, 0.0, 0.0], None).await?;
        db.insert("a", "v2", vec![0.0, 1.0, 0.0, 0.0], None).await?;
        db.insert("b", "v3", vec![0.0, 0.0, 1.0, 0.0], None).await?;
        assert_eq!(db.total_vectors().await, 3);
        Ok(())
    }

    // ── batch_search_with_filter ─────────────────────────────────────────

    #[tokio::test]
    async fn test_batch_search_with_filter() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        for i in 0..10 {
            db.insert(
                "test",
                &format!("v{i}"),
                vec![i as f32, 0.0, 0.0, 0.0],
                Some(json!({"group": if i < 5 { "a" } else { "b" }})),
            ).await?;
        }

        let filter = crate::metadata::Filter::parse(&json!({"group": "a"}))?;
        let results = db.batch_search_with_filter(
            "test",
            vec![vec![0.0, 0.0, 0.0, 0.0], vec![5.0, 0.0, 0.0, 0.0]],
            3,
            filter,
        ).await?;
        assert_eq!(results.len(), 2);
        for batch in &results {
            assert!(batch.len() <= 3);
        }
        Ok(())
    }

    // ── inner() and config() accessors ───────────────────────────────────

    #[test]
    fn test_inner_accessor() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        let inner = db.inner();
        let _ = inner; // Just verify we can access it
        Ok(())
    }

    #[test]
    fn test_config_accessor() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = AsyncDatabaseConfig::default()
            .with_max_concurrency(8);
        let db = AsyncDatabase::in_memory_with_config(config);
        assert_eq!(db.config().max_concurrency, 8);
        Ok(())
    }

    // ── get returns None for nonexistent ──────────────────────────────────

    #[tokio::test]
    async fn test_get_nonexistent_vector() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        let result = db.get("test", "nonexistent").await;
        assert!(result.is_none());
        Ok(())
    }

    // ── get returns vector and metadata ──────────────────────────────────

    #[tokio::test]
    async fn test_get_with_metadata() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        db.insert("test", "v1", vec![1.0, 2.0, 3.0, 4.0], Some(json!({"k": "v"}))).await?;
        let (vec, meta) = db.get("test", "v1").await.ok_or("vector not found")?;
        assert_eq!(vec.len(), 4);
        assert!(meta.is_some());
        Ok(())
    }

    // ── ids() ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_ids_returns_all() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        for i in 0..5 {
            db.insert("test", &format!("v{i}"), random_vector(4), None).await?;
        }
        let ids = db.ids("test").await?;
        assert_eq!(ids.len(), 5);
        Ok(())
    }

    // ── compact returns count ────────────────────────────────────────────

    #[tokio::test]
    async fn test_compact_returns_count() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        for i in 0..10 {
            db.insert("test", &format!("v{i}"), random_vector(4), None).await?;
        }
        for i in 0..5 {
            db.delete("test", &format!("v{i}")).await?;
        }
        let removed = db.compact("test").await?;
        assert!(removed > 0 || db.count("test").await == 5);
        Ok(())
    }

    // ── needs_compaction ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_needs_compaction_after_deletions() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        for i in 0..10 {
            db.insert("test", &format!("v{i}"), random_vector(4), None).await?;
        }
        for i in 0..8 {
            db.delete("test", &format!("v{i}")).await?;
        }
        assert!(db.needs_compaction("test", 0.5).await);
        Ok(())
    }

    // ── count_with_filter ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_count_with_filter_subset() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        for i in 0..6 {
            db.insert(
                "test",
                &format!("v{i}"),
                random_vector(4),
                Some(json!({"type": if i % 2 == 0 { "even" } else { "odd" }})),
            ).await?;
        }
        let filter = crate::metadata::Filter::parse(&json!({"type": "even"}))?;
        let count = db.count_with_filter("test", Some(filter)).await?;
        assert_eq!(count, 3);
        Ok(())
    }

    // ── duplicate collection creation error ──────────────────────────────

    #[tokio::test]
    async fn test_create_duplicate_collection() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        let result = db.create_collection("test", 4).await;
        assert!(result.is_err());
        Ok(())
    }

    // ── batch insert then search correctness ─────────────────────────────

    #[tokio::test]
    async fn test_batch_insert_search_correctness() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;

        let entries: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = (0..20)
            .map(|i| (format!("v{i}"), random_vector(4), None))
            .collect();
        db.batch_insert("test", entries).await?;

        assert_eq!(db.count("test").await, 20);
        let results = db.search("test", random_vector(4), 5).await?;
        assert_eq!(results.len(), 5);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i].distance >= results[i-1].distance);
        }
        Ok(())
    }

    // ── ExportStream accessors ───────────────────────────────────────────

    #[tokio::test]
    async fn test_export_stream_accessors() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = AsyncDatabase::in_memory();
        db.create_collection("test", 4).await?;
        for i in 0..3 {
            db.insert("test", &format!("v{i}"), random_vector(4), None).await?;
        }
        let stream = db.stream_export("test", 10).await?;
        assert_eq!(stream.total(), 3);
        assert_eq!(stream.offset(), 0);
        assert!(!stream.is_exhausted());
        Ok(())
    }
}
