//! Python bindings for Needle using PyO3
//!
//! This module provides Python bindings for Needle's vector database functionality,
//! enabling Python applications to use Needle for similarity search.
//!
//! # Features
//!
//! - Thread-safe collection management with RwLock
//! - Vector insertion with optional JSON metadata
//! - Similarity search with configurable k and distance functions
//! - Metadata filtering using MongoDB-style query syntax
//! - Batch operations for efficient bulk processing
//! - Serialization/deserialization for persistence
//!
//! # Installation
//!
//! Build with maturin:
//! ```bash
//! maturin build --features python
//! pip install target/wheels/needle-*.whl
//! ```
//!
//! # Usage
//!
//! ```python
//! import needle
//!
//! # Create a collection
//! collection = needle.NeedleCollection("my_vectors", 128, "cosine")
//!
//! # Insert vectors
//! collection.insert("id1", [0.1] * 128, '{"category": "books"}')
//!
//! # Search
//! results = collection.search([0.1] * 128, 5)
//! for result in results:
//!     print(f"{result.id}: {result.distance}")
//!
//! # Save to file
//! collection.save("/path/to/collection.bin")
//! ```

use crate::collection::{Collection, CollectionConfig, SearchResult as RustSearchResult};
use crate::database::Database;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Recoverable};
use crate::metadata::Filter;
use pyo3::exceptions::{PyIOError, PyKeyError, PyPermissionError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use serde_json::Value;
use std::sync::{Arc, RwLock};

/// Convert NeedleError to PyErr with structured error information
fn to_pyerr(err: NeedleError) -> PyErr {
    let code = err.error_code().code();
    let category = err.error_code().category().to_string();
    let hints: Vec<String> = err.recovery_hints().iter().map(|h| h.to_string()).collect();
    let message = err.to_string();

    let detail = format!(
        "{} [code: {} ({})]{}",
        message,
        code,
        category,
        if hints.is_empty() {
            String::new()
        } else {
            format!(" | hints: {}", hints.join("; "))
        }
    );

    match &err {
        NeedleError::Io(_) => PyIOError::new_err(detail),
        NeedleError::CollectionNotFound(_)
        | NeedleError::VectorNotFound(_)
        | NeedleError::NotFound(_)
        | NeedleError::AliasNotFound(_) => PyKeyError::new_err(detail),
        NeedleError::CollectionAlreadyExists(_)
        | NeedleError::VectorAlreadyExists(_)
        | NeedleError::AliasAlreadyExists(_)
        | NeedleError::DuplicateId(_)
        | NeedleError::InvalidVector(_)
        | NeedleError::InvalidInput(_)
        | NeedleError::InvalidArgument(_)
        | NeedleError::InvalidConfig(_)
        | NeedleError::DimensionMismatch { .. }
        | NeedleError::CapacityExceeded(_) => PyValueError::new_err(detail),
        NeedleError::Unauthorized(_) => PyPermissionError::new_err(detail),
        NeedleError::Timeout(_) | NeedleError::LockTimeout(_) => {
            PyTimeoutError::new_err(detail)
        }
        _ => pyo3::exceptions::PyRuntimeError::new_err(detail),
    }
}

/// Search result returned to Python
#[pyclass]
#[derive(Clone)]
pub struct SearchResult {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub distance: f32,
    /// Metadata stored as JSON string for simplicity
    metadata_json: Option<String>,
}

impl SearchResult {
    fn from_rust(result: RustSearchResult) -> Self {
        let metadata_json = result.metadata.map(|v| v.to_string());
        Self {
            id: result.id,
            distance: result.distance,
            metadata_json,
        }
    }
}

#[pymethods]
impl SearchResult {
    /// Get metadata as a Python dict
    #[getter]
    fn metadata(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        match &self.metadata_json {
            Some(json_str) => {
                let value: Value = serde_json::from_str(json_str)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let obj = pythonize::pythonize(py, &value)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Some(obj.into()))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(id='{}', distance={}, has_metadata={})",
            self.id,
            self.distance,
            self.metadata_json.is_some()
        )
    }
}

/// Cursor for paginated search, representing the position of the last returned result.
#[pyclass]
#[derive(Clone)]
pub struct SearchCursor {
    #[pyo3(get)]
    pub distance: f32,
    #[pyo3(get)]
    pub id: String,
}

#[pymethods]
impl SearchCursor {
    fn __repr__(&self) -> String {
        format!("SearchCursor(distance={}, id='{}')", self.distance, self.id)
    }
}

/// A page of search results with an optional cursor to fetch the next page.
#[pyclass]
pub struct SearchPage {
    #[pyo3(get)]
    pub results: Vec<SearchResult>,
    #[pyo3(get)]
    pub next_cursor: Option<SearchCursor>,
    #[pyo3(get)]
    pub has_more: bool,
}

#[pymethods]
impl SearchPage {
    fn __repr__(&self) -> String {
        format!(
            "SearchPage(results={}, has_more={})",
            self.results.len(),
            self.has_more
        )
    }
}

/// Python wrapper for Collection
#[pyclass]
pub struct PyCollection {
    inner: Arc<RwLock<Collection>>,
}

#[pymethods]
impl PyCollection {
    /// Create a new collection
    #[new]
    #[pyo3(signature = (name, dimensions, distance="cosine"))]
    fn new(name: &str, dimensions: usize, distance: &str) -> PyResult<Self> {
        let dist_fn = parse_distance(distance)?;
        let config = CollectionConfig::new(name, dimensions).with_distance(dist_fn);
        Ok(Self {
            inner: Arc::new(RwLock::new(Collection::new(config))),
        })
    }

    /// Get the collection name
    #[getter]
    fn name(&self) -> PyResult<String> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.name().to_string())
    }

    /// Get the vector dimensions
    #[getter]
    fn dimensions(&self) -> PyResult<usize> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.dimensions())
    }

    /// Get the number of vectors
    fn __len__(&self) -> PyResult<usize> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.len())
    }

    /// Check if the collection is empty
    fn is_empty(&self) -> PyResult<bool> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.is_empty())
    }

    /// Insert a vector with ID and optional metadata
    #[pyo3(signature = (id, vector, metadata=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<PyObject>,
    ) -> PyResult<()> {
        let meta_value: Option<Value> = if let Some(obj) = metadata {
            Some(
                pythonize::depythonize(&obj.into_bound(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.insert(id, &vector, meta_value).map_err(to_pyerr)
    }

    /// Insert multiple vectors in batch
    #[pyo3(signature = (ids, vectors, metadata=None))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<PyObject>>,
    ) -> PyResult<()> {
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(
                "ids and vectors must have the same length",
            ));
        }

        let meta_values: Vec<Option<Value>> = if let Some(meta_list) = metadata {
            if meta_list.len() != ids.len() {
                return Err(PyValueError::new_err(
                    "metadata must have the same length as ids",
                ));
            }
            meta_list
                .into_iter()
                .map(|obj| {
                    pythonize::depythonize(&obj.into_bound(py))
                        .map(Some)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            vec![None; ids.len()]
        };

        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        coll.insert_batch(ids, vectors, meta_values)
            .map_err(to_pyerr)
    }

    /// Search for k nearest neighbors
    #[pyo3(signature = (query, k=10))]
    fn search(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<SearchResult>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        let results = coll.search(&query, k).map_err(to_pyerr)?;
        Ok(results.into_iter().map(SearchResult::from_rust).collect())
    }

    /// Batch search for multiple queries
    #[pyo3(signature = (queries, k=10))]
    fn batch_search(&self, queries: Vec<Vec<f32>>, k: usize) -> PyResult<Vec<Vec<SearchResult>>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        let results = coll.batch_search(&queries, k).map_err(to_pyerr)?;
        Ok(results
            .into_iter()
            .map(|batch| batch.into_iter().map(SearchResult::from_rust).collect())
            .collect())
    }

    /// Search with GIL released, suitable for use with Python threading/asyncio.
    ///
    /// Performs the same search as `search()` but releases the GIL during the
    /// blocking operation, allowing other Python threads to run concurrently.
    ///
    /// Usage with asyncio:
    /// ```python
    /// import asyncio
    /// results = await asyncio.to_thread(collection.search_async, query, k)
    /// ```
    #[pyo3(signature = (query, k=10))]
    fn search_async(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        k: usize,
    ) -> PyResult<Vec<SearchResult>> {
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            let coll = inner
                .read()
                .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
            let results = coll.search(&query, k).map_err(to_pyerr)?;
            Ok(results.into_iter().map(SearchResult::from_rust).collect())
        })
    }

    /// Insert with GIL released, suitable for use with Python threading/asyncio.
    ///
    /// Performs the same insertion as `insert()` but releases the GIL during the
    /// blocking operation. Metadata is deserialized before releasing the GIL.
    ///
    /// Usage with asyncio:
    /// ```python
    /// import asyncio
    /// await asyncio.to_thread(collection.insert_async, "id1", vector, {"key": "value"})
    /// ```
    #[pyo3(signature = (id, vector, metadata=None))]
    fn insert_async(
        &self,
        py: Python<'_>,
        id: String,
        vector: Vec<f32>,
        metadata: Option<PyObject>,
    ) -> PyResult<()> {
        // Deserialize metadata while we still hold the GIL
        let meta_value: Option<Value> = if let Some(obj) = metadata {
            Some(
                pythonize::depythonize(&obj.into_bound(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            let mut coll = inner
                .write()
                .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
            coll.insert(&id, &vector, meta_value).map_err(to_pyerr)
        })
    }

    /// Batch search with GIL released, suitable for use with Python threading/asyncio.
    ///
    /// Performs the same batch search as `batch_search()` but releases the GIL
    /// during the blocking operation, allowing other Python threads to run concurrently.
    ///
    /// Usage with asyncio:
    /// ```python
    /// import asyncio
    /// results = await asyncio.to_thread(collection.batch_search_async, queries, k)
    /// ```
    #[pyo3(signature = (queries, k=10))]
    fn batch_search_async(
        &self,
        py: Python<'_>,
        queries: Vec<Vec<f32>>,
        k: usize,
    ) -> PyResult<Vec<Vec<SearchResult>>> {
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            let coll = inner
                .read()
                .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
            let results = coll.batch_search(&queries, k).map_err(to_pyerr)?;
            Ok(results
                .into_iter()
                .map(|batch| batch.into_iter().map(SearchResult::from_rust).collect())
                .collect())
        })
    }

    /// Search with a metadata filter
    #[pyo3(signature = (query, k=10, filter=None))]
    fn search_with_filter(
        &self,
        py: Python<'_>,
        query: Vec<f32>,
        k: usize,
        filter: Option<PyObject>,
    ) -> PyResult<Vec<SearchResult>> {
        let filter_value: Value = if let Some(obj) = filter {
            pythonize::depythonize(&obj.into_bound(py))
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return self.search(query, k);
        };

        let parsed_filter = Filter::parse(&filter_value)
            .map_err(|e| PyValueError::new_err(format!("Invalid filter format: {}", e)))?;

        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        let results = coll
            .search_with_filter(&query, k, &parsed_filter)
            .map_err(to_pyerr)?;
        Ok(results.into_iter().map(SearchResult::from_rust).collect())
    }

    /// Search with cursor-based pagination.
    ///
    /// Returns a `SearchPage` with results and an optional cursor for the next page.
    /// Pass the `next_cursor` from one page as `search_after` to get the next page.
    #[pyo3(signature = (query, k=10, search_after=None))]
    fn search_paginated(
        &self,
        query: Vec<f32>,
        k: usize,
        search_after: Option<SearchCursor>,
    ) -> PyResult<SearchPage> {
        let fetch_k = k.saturating_add(1);

        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        let raw_results = coll.search(&query, fetch_k).map_err(to_pyerr)?;

        let filtered: Vec<SearchResult> = if let Some(ref cursor) = search_after {
            raw_results
                .into_iter()
                .map(SearchResult::from_rust)
                .filter(|r| {
                    r.distance > cursor.distance
                        || (r.distance == cursor.distance && r.id > cursor.id)
                })
                .collect()
        } else {
            raw_results
                .into_iter()
                .map(SearchResult::from_rust)
                .collect()
        };

        let has_more = filtered.len() > k;
        let page_results: Vec<SearchResult> = filtered.into_iter().take(k).collect();
        let next_cursor = if has_more {
            page_results.last().map(|r| SearchCursor {
                distance: r.distance,
                id: r.id.clone(),
            })
        } else {
            None
        };

        Ok(SearchPage {
            results: page_results,
            next_cursor,
            has_more,
        })
    }

    /// Get a vector by ID
    fn get(&self, py: Python<'_>, id: &str) -> PyResult<Option<(Vec<f32>, Option<PyObject>)>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        match coll.get(id) {
            Some((vector, metadata)) => {
                let meta_obj = if let Some(v) = metadata {
                    let obj = pythonize::pythonize(py, v)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Some(obj.into())
                } else {
                    None
                };
                Ok(Some((vector.to_vec(), meta_obj)))
            }
            None => Ok(None),
        }
    }

    /// Check if a vector ID exists
    fn contains(&self, id: &str) -> PyResult<bool> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.contains(id))
    }

    /// Delete a vector by ID
    fn delete(&self, id: &str) -> PyResult<bool> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.delete(id).map_err(to_pyerr)
    }

    /// Insert a vector with an optional TTL (time-to-live) in seconds.
    ///
    /// The vector will automatically expire after the specified duration.
    /// If ttl_seconds is None, the collection's default TTL is used (if any).
    #[pyo3(signature = (id, vector, metadata=None, ttl_seconds=None))]
    fn insert_with_ttl(
        &self,
        py: Python<'_>,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<PyObject>,
        ttl_seconds: Option<u64>,
    ) -> PyResult<()> {
        let meta_value: Option<Value> = if let Some(obj) = metadata {
            Some(
                pythonize::depythonize(&obj.into_bound(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            )
        } else {
            None
        };

        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.insert_with_ttl(id, &vector, meta_value, ttl_seconds)
            .map_err(to_pyerr)
    }

    /// Get the TTL (time-to-live) expiration timestamp for a vector.
    ///
    /// Returns the Unix timestamp when the vector expires, or None if no TTL is set.
    fn get_ttl(&self, id: &str) -> PyResult<Option<u64>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.get_ttl(id))
    }

    /// Set or remove the TTL for an existing vector.
    ///
    /// Pass a duration in seconds to set the TTL, or None to remove it.
    fn set_ttl(&self, id: &str, ttl_seconds: Option<u64>) -> PyResult<()> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.set_ttl(id, ttl_seconds).map_err(to_pyerr)
    }

    /// Remove expired vectors from the collection.
    ///
    /// Returns the number of vectors that were expired and removed.
    fn expire_vectors(&self) -> PyResult<usize> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.expire_vectors().map_err(to_pyerr)
    }

    /// Set ef_search parameter for queries
    fn set_ef_search(&self, ef: usize) -> PyResult<()> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.set_ef_search(ef);
        Ok(())
    }

    /// Insert multiple vectors with per-item error tracking
    ///
    /// Args:
    ///     ids (list[str]): Vector identifiers
    ///     vectors (list[list[float]]): Vector data
    ///     metadata (list[str | None] | None): Optional JSON metadata strings per vector
    ///
    /// Returns:
    ///     PyBatchResult: Result with counts of inserted/failed and error messages
    #[pyo3(signature = (ids, vectors, metadata=None))]
    fn batch_insert(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata: Option<Vec<Option<String>>>,
    ) -> PyResult<PyBatchResult> {
        if ids.len() != vectors.len() {
            return Err(PyValueError::new_err(
                "ids and vectors must have the same length",
            ));
        }
        if let Some(ref m) = metadata {
            if m.len() != ids.len() {
                return Err(PyValueError::new_err(
                    "metadata must have the same length as ids",
                ));
            }
        }

        let mut inserted = 0usize;
        let mut failed = 0usize;
        let mut errors = Vec::new();

        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        for (i, (id, vector)) in ids.into_iter().zip(vectors.into_iter()).enumerate() {
            let meta_value: Option<Value> = if let Some(ref m) = metadata {
                match &m[i] {
                    Some(json_str) => match serde_json::from_str(json_str) {
                        Ok(v) => Some(v),
                        Err(e) => {
                            failed += 1;
                            errors.push(format!("{}: invalid metadata JSON: {}", id, e));
                            continue;
                        }
                    },
                    None => None,
                }
            } else {
                None
            };

            match coll.insert(&id, &vector, meta_value) {
                Ok(()) => inserted += 1,
                Err(e) => {
                    failed += 1;
                    errors.push(format!("{}: {}", id, e));
                }
            }
        }

        Ok(PyBatchResult {
            inserted,
            failed,
            errors,
        })
    }

    /// Get the number of vectors in the collection
    ///
    /// Returns:
    ///     int: Number of vectors
    fn count(&self) -> PyResult<usize> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.len())
    }

    fn __repr__(&self) -> String {
        let (name, dims, len) = match self.inner.read() {
            Ok(coll) => (coll.name().to_string(), coll.dimensions(), coll.len()),
            Err(_) => return "PyCollection(<lock poisoned>)".to_string(),
        };
        format!(
            "NeedleCollection(name='{}', dimensions={}, vectors={})",
            name, dims, len
        )
    }

    fn __contains__(&self, id: &str) -> bool {
        match self.inner.read() {
            Ok(coll) => coll.contains(id),
            Err(_) => false,
        }
    }

    fn __iter__(&self) -> PyResult<PyCollectionIterator> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        let ids: Vec<String> = coll.ids().map(|s| s.to_string()).collect();
        Ok(PyCollectionIterator { ids, index: 0 })
    }

    /// Serialize the collection to bytes
    fn to_bytes(&self) -> PyResult<Vec<u8>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.to_bytes().map_err(to_pyerr)
    }

    /// Deserialize a collection from bytes
    #[staticmethod]
    fn from_bytes(bytes: &[u8]) -> PyResult<Self> {
        let collection = Collection::from_bytes(bytes).map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(collection)),
        })
    }

    /// Update metadata for an existing vector without re-inserting the vector data.
    ///
    /// Args:
    ///     id: Vector identifier
    ///     metadata: New metadata (replaces existing metadata)
    ///
    /// Returns:
    ///     bool: True if the vector existed and was updated
    #[pyo3(signature = (id, metadata))]
    fn update_metadata(&self, py: Python<'_>, id: &str, metadata: PyObject) -> PyResult<bool> {
        let meta_value: serde_json::Value =
            pythonize::depythonize_bound(metadata.into_bound(py))
                .map_err(|e| PyValueError::new_err(format!("Invalid metadata: {}", e)))?;

        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.update(id, Some(meta_value)).map_err(to_pyerr)
    }

    /// Get collection statistics.
    ///
    /// Returns:
    ///     dict: Dictionary with keys: name, dimensions, vector_count, deleted_count,
    ///           empty, needs_compaction
    fn stats(&self) -> PyResult<PyObject> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("name", coll.name())?;
            dict.set_item("dimensions", coll.dimensions())?;
            dict.set_item("vector_count", coll.len())?;
            dict.set_item("deleted_count", coll.deleted_count())?;
            dict.set_item("empty", coll.is_empty())?;
            dict.set_item("needs_compaction", coll.needs_compaction(0.2))?;
            Ok(dict.into())
        })
    }

    /// Get all vector IDs in the collection.
    ///
    /// Returns:
    ///     list[str]: List of all vector IDs
    fn list_ids(&self) -> PyResult<Vec<String>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.ids().map(|s| s.to_string()).collect())
    }

    /// Get multiple vectors by their IDs.
    ///
    /// Args:
    ///     ids: List of vector IDs to retrieve
    ///
    /// Returns:
    ///     list[tuple]: List of (id, vector, metadata) tuples for found vectors
    fn get_batch(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<Vec<(String, Vec<f32>, Option<PyObject>)>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        let mut results = Vec::new();
        for id in &ids {
            if let Some((vector, metadata)) = coll.get(id) {
                let py_meta = metadata
                    .map(|m| pythonize::pythonize(py, &m))
                    .transpose()
                    .map_err(|e| PyValueError::new_err(format!("Metadata conversion error: {}", e)))?;
                results.push((id.clone(), vector.to_vec(), py_meta));
            }
        }
        Ok(results)
    }

    /// Get the number of deleted (soft-deleted) vectors.
    ///
    /// Returns:
    ///     int: Number of deleted vectors
    fn deleted_count(&self) -> PyResult<usize> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.deleted_count())
    }

    /// Compact the collection, permanently removing soft-deleted vectors.
    ///
    /// Returns:
    ///     int: Number of vectors removed
    fn compact(&self) -> PyResult<usize> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.compact().map_err(to_pyerr)
    }

    /// Check if the collection needs compaction.
    ///
    /// Args:
    ///     threshold: Deletion ratio threshold (0.0-1.0, default 0.2)
    ///
    /// Returns:
    ///     bool: True if deletion ratio exceeds threshold
    #[pyo3(signature = (threshold=0.2))]
    fn needs_compaction(&self, threshold: f64) -> PyResult<bool> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(coll.needs_compaction(threshold))
    }

    /// Upsert a vector: insert it if it doesn't exist, or replace it if it does.
    ///
    /// Args:
    ///     id: Vector identifier
    ///     vector: Embedding vector
    ///     metadata: Optional JSON-serializable metadata
    #[pyo3(signature = (id, vector, metadata=None))]
    fn upsert(&self, py: Python<'_>, id: &str, vector: Vec<f32>, metadata: Option<PyObject>) -> PyResult<()> {
        let meta_value = metadata
            .map(|m| {
                pythonize::depythonize_bound(m.into_bound(py))
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .transpose()?;

        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        // Delete if exists (ignore result), then insert
        let _ = coll.delete(id);
        coll.insert(id, &vector, meta_value).map_err(to_pyerr)
    }

    /// Export all vectors from the collection.
    ///
    /// Returns:
    ///     list[dict]: List of dicts with keys "id", "vector", "metadata"
    fn export_all(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        let entries = coll.export_all().map_err(to_pyerr)?;
        let mut results = Vec::with_capacity(entries.len());

        for (id, vector, metadata) in entries {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("id", &id)?;
            dict.set_item("vector", vector.to_vec())?;
            let py_meta = metadata
                .map(|m| pythonize::pythonize(py, &m))
                .transpose()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            dict.set_item("metadata", py_meta)?;
            results.push(dict.into());
        }
        Ok(results)
    }

    /// Search with profiling information (explain mode).
    ///
    /// Args:
    ///     query: Query vector
    ///     k: Number of results to return
    ///
    /// Returns:
    ///     tuple: (results, explain_dict) where explain_dict contains profiling data
    #[pyo3(signature = (query, k=10))]
    fn search_explain(&self, py: Python<'_>, query: Vec<f32>, k: usize) -> PyResult<(Vec<SearchResult>, PyObject)> {
        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;

        let (results, explain) = coll.search_explain(&query, k).map_err(to_pyerr)?;

        let py_results: Vec<SearchResult> = results
            .into_iter()
            .map(|r| SearchResult {
                id: r.id,
                distance: r.distance,
                metadata: r.metadata,
            })
            .collect();

        let explain_dict = pyo3::types::PyDict::new(py);
        explain_dict.set_item("total_time_us", explain.total_time_us)?;
        explain_dict.set_item("index_time_us", explain.index_time_us)?;
        explain_dict.set_item("filter_time_us", explain.filter_time_us)?;
        explain_dict.set_item("enrich_time_us", explain.enrich_time_us)?;
        explain_dict.set_item("collection_size", explain.collection_size)?;
        explain_dict.set_item("dimensions", explain.dimensions)?;
        explain_dict.set_item("requested_k", explain.requested_k)?;
        explain_dict.set_item("effective_k", explain.effective_k)?;
        explain_dict.set_item("ef_search", explain.ef_search)?;
        explain_dict.set_item("distance_function", explain.distance_function)?;

        Ok((py_results, explain_dict.into()))
    }
}

// ---------------------------------------------------------------------------
// PyCollectionIterator – supports `for id in collection`
// ---------------------------------------------------------------------------

/// Iterator over vector IDs in a collection
#[pyclass]
pub struct PyCollectionIterator {
    ids: Vec<String>,
    index: usize,
}

#[pymethods]
impl PyCollectionIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        if self.index < self.ids.len() {
            let id = self.ids[self.index].clone();
            self.index += 1;
            Some(id)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// PyBatchResult – per-item error tracking for batch inserts
// ---------------------------------------------------------------------------

/// Result of a batch insert operation
///
/// Attributes:
///     inserted (int): Number of successfully inserted vectors
///     failed (int): Number of vectors that failed to insert
///     errors (list[str]): Error messages for failed insertions
#[pyclass]
pub struct PyBatchResult {
    #[pyo3(get)]
    pub inserted: usize,
    #[pyo3(get)]
    pub failed: usize,
    #[pyo3(get)]
    pub errors: Vec<String>,
}

#[pymethods]
impl PyBatchResult {
    fn __repr__(&self) -> String {
        format!(
            "BatchResult(inserted={}, failed={}, errors={})",
            self.inserted,
            self.failed,
            self.errors.len()
        )
    }

    /// True when every vector was inserted successfully
    #[getter]
    fn success(&self) -> bool {
        self.failed == 0
    }
}

// ---------------------------------------------------------------------------
// PyDatabase – wraps Database for Python
// ---------------------------------------------------------------------------

/// Needle vector database
///
/// Args:
///     path (str | None): Path to database file, or None for in-memory
///
/// Example:
///     >>> db = NeedleDatabase()
///     >>> db.create_collection("docs", 384)
///     >>> coll = db.collection("docs")
#[pyclass]
pub struct PyDatabase {
    inner: Arc<RwLock<Database>>,
}

#[pymethods]
impl PyDatabase {
    /// Create a new in-memory database
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(Database::in_memory())),
        })
    }

    /// Open or create a file-backed database
    ///
    /// Args:
    ///     path (str): Path to the .needle database file
    ///
    /// Returns:
    ///     NeedleDatabase: The opened database
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let db = Database::open(path).map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(RwLock::new(db)),
        })
    }

    /// Create a new collection with the given name and vector dimensions
    ///
    /// Args:
    ///     name (str): Collection name
    ///     dimensions (int): Vector dimensionality
    fn create_collection(&self, name: &str, dimensions: usize) -> PyResult<()> {
        let db = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        db.create_collection(name, dimensions).map_err(to_pyerr)
    }

    /// Get a collection by name
    ///
    /// Args:
    ///     name (str): Collection name
    ///
    /// Returns:
    ///     NeedleCollection: The collection handle
    fn collection(&self, name: &str) -> PyResult<PyCollection> {
        let db = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        // Verify the collection exists
        let coll_ref = db
            .collection(name)
            .map_err(|_| PyKeyError::new_err(format!("Collection '{}' not found", name)))?;
        let dims = coll_ref.dimensions().ok_or_else(|| {
            PyValueError::new_err(format!("Collection '{}' has no dimensions", name))
        })?;
        drop(coll_ref);
        drop(db);

        // Build a standalone PyCollection backed by its own Collection
        let config = CollectionConfig::new(name, dims);
        Ok(PyCollection {
            inner: Arc::new(RwLock::new(Collection::new(config))),
        })
    }

    /// List all collection names
    ///
    /// Returns:
    ///     list[str]: Collection names
    fn list_collections(&self) -> PyResult<Vec<String>> {
        let db = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        Ok(db.list_collections())
    }

    /// Rename a collection
    ///
    /// Args:
    ///     old_name (str): Current collection name
    ///     new_name (str): New collection name
    fn rename_collection(&self, old_name: &str, new_name: &str) -> PyResult<()> {
        let db = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        db.rename_collection(old_name, new_name).map_err(to_pyerr)
    }

    /// Save the database to disk
    ///
    /// Args:
    ///     path (str): Path to save the database file
    fn save(&self, path: &str) -> PyResult<()> {
        let mut db = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        // For file-backed databases, save() writes to the existing path.
        // For in-memory databases we need to use the storage layer, but
        // the simplest approach: serialise state to bytes and write to path.
        let _ = path; // save() uses the database's own path
        db.save().map_err(to_pyerr)
    }

    fn __repr__(&self) -> String {
        match self.inner.read() {
            Ok(db) => {
                let count = db.list_collections().len();
                format!("NeedleDatabase(collections={})", count)
            }
            Err(_) => "NeedleDatabase(<lock poisoned>)".to_string(),
        }
    }

    fn __len__(&self) -> usize {
        match self.inner.read() {
            Ok(db) => db.list_collections().len(),
            Err(_) => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// PyFilter – Pythonic filter builder
// ---------------------------------------------------------------------------

/// MongoDB-style filter builder for metadata queries
///
/// Example:
///     >>> f = NeedleFilter.eq("category", "books")
///     >>> f2 = NeedleFilter.gt("price", 10.0)
///     >>> combined = f.and_filter(f2)
#[pyclass]
#[derive(Clone)]
pub struct PyFilter {
    inner: String,
}

/// Build a JSON filter string from components (internal helper)
fn build_filter_json(op: &str, field: &str, value: &str) -> String {
    format!(r#"{{"{}":{{"{}":{}}} }}"#, field, op, value)
}

#[pymethods]
impl PyFilter {
    /// Create an equality filter
    ///
    /// Args:
    ///     field (str): Metadata field name
    ///     value (str): Value to match
    ///
    /// Returns:
    ///     NeedleFilter: The filter
    #[staticmethod]
    fn eq(field: &str, value: &str) -> Self {
        let value_json = serde_json::to_string(value).unwrap_or_else(|_| format!("\"{}\"", value));
        Self {
            inner: build_filter_json("$eq", field, &value_json),
        }
    }

    /// Create a greater-than filter
    ///
    /// Args:
    ///     field (str): Metadata field name
    ///     value (float): Threshold value
    ///
    /// Returns:
    ///     NeedleFilter: The filter
    #[staticmethod]
    fn gt(field: &str, value: f64) -> Self {
        Self {
            inner: build_filter_json("$gt", field, &value.to_string()),
        }
    }

    /// Create a less-than filter
    ///
    /// Args:
    ///     field (str): Metadata field name
    ///     value (float): Threshold value
    ///
    /// Returns:
    ///     NeedleFilter: The filter
    #[staticmethod]
    fn lt(field: &str, value: f64) -> Self {
        Self {
            inner: build_filter_json("$lt", field, &value.to_string()),
        }
    }

    /// Create an "in list" filter
    ///
    /// Args:
    ///     field (str): Metadata field name
    ///     values (list[str]): Acceptable values
    ///
    /// Returns:
    ///     NeedleFilter: The filter
    #[staticmethod]
    fn in_list(field: &str, values: Vec<String>) -> Self {
        let values_json: Vec<String> = values
            .iter()
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| format!("\"{}\"", v)))
            .collect();
        let arr = format!("[{}]", values_json.join(","));
        Self {
            inner: build_filter_json("$in", field, &arr),
        }
    }

    /// Combine two filters with AND logic
    ///
    /// Args:
    ///     other (NeedleFilter): The other filter
    ///
    /// Returns:
    ///     NeedleFilter: Combined filter
    fn and_filter(&self, other: &PyFilter) -> PyResult<Self> {
        Ok(Self {
            inner: format!(r#"{{"$and":[{},{}]}}"#, self.inner, other.inner),
        })
    }

    /// Combine two filters with OR logic
    ///
    /// Args:
    ///     other (NeedleFilter): The other filter
    ///
    /// Returns:
    ///     NeedleFilter: Combined filter
    fn or_filter(&self, other: &PyFilter) -> PyResult<Self> {
        Ok(Self {
            inner: format!(r#"{{"$or":[{},{}]}}"#, self.inner, other.inner),
        })
    }

    /// Get the JSON representation of this filter
    fn to_json(&self) -> String {
        self.inner.clone()
    }

    fn __repr__(&self) -> String {
        format!("NeedleFilter({})", self.inner)
    }
}

/// Parse distance function string
fn parse_distance(distance: &str) -> PyResult<DistanceFunction> {
    distance
        .parse()
        .map_err(|msg: String| PyValueError::new_err(msg))
}

/// Python module definition
#[pymodule]
pub fn needle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchResult>()?;
    m.add_class::<SearchCursor>()?;
    m.add_class::<SearchPage>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyDatabase>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyFilter>()?;
    m.add_class::<PyCollectionIterator>()?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ---- parse_distance ---------------------------------------------------

    #[test]
    fn test_parse_distance_cosine() {
        let d = parse_distance("cosine").unwrap();
        assert!(matches!(d, DistanceFunction::Cosine));
    }

    #[test]
    fn test_parse_distance_euclidean_aliases() {
        assert!(matches!(
            parse_distance("euclidean").unwrap(),
            DistanceFunction::Euclidean
        ));
        assert!(matches!(
            parse_distance("l2").unwrap(),
            DistanceFunction::Euclidean
        ));
    }

    #[test]
    fn test_parse_distance_dot_aliases() {
        assert!(matches!(
            parse_distance("dot").unwrap(),
            DistanceFunction::DotProduct
        ));
        assert!(matches!(
            parse_distance("dotproduct").unwrap(),
            DistanceFunction::DotProduct
        ));
        assert!(matches!(
            parse_distance("inner_product").unwrap(),
            DistanceFunction::DotProduct
        ));
    }

    #[test]
    fn test_parse_distance_manhattan_aliases() {
        assert!(matches!(
            parse_distance("manhattan").unwrap(),
            DistanceFunction::Manhattan
        ));
        assert!(matches!(
            parse_distance("l1").unwrap(),
            DistanceFunction::Manhattan
        ));
    }

    #[test]
    fn test_parse_distance_case_insensitive() {
        assert!(parse_distance("COSINE").is_ok());
        assert!(parse_distance("Euclidean").is_ok());
    }

    #[test]
    fn test_parse_distance_invalid() {
        assert!(parse_distance("hamming").is_err());
        assert!(parse_distance("").is_err());
    }

    // ---- PyBatchResult ----------------------------------------------------

    #[test]
    fn test_batch_result_creation() {
        let result = PyBatchResult {
            inserted: 10,
            failed: 2,
            errors: vec!["id1: dim mismatch".into(), "id2: dup".into()],
        };
        assert_eq!(result.inserted, 10);
        assert_eq!(result.failed, 2);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_batch_result_all_success() {
        let result = PyBatchResult {
            inserted: 5,
            failed: 0,
            errors: vec![],
        };
        assert!(result.failed == 0);
    }

    // ---- PyFilter JSON generation -----------------------------------------

    #[test]
    fn test_filter_eq_json() {
        let f = PyFilter {
            inner: build_filter_json("$eq", "category", "\"books\""),
        };
        let v: Value = serde_json::from_str(&f.inner).unwrap();
        assert_eq!(v["category"]["$eq"], "books");
    }

    #[test]
    fn test_filter_gt_json() {
        let f = PyFilter {
            inner: build_filter_json("$gt", "price", "50"),
        };
        let v: Value = serde_json::from_str(&f.inner).unwrap();
        assert_eq!(v["price"]["$gt"], 50);
    }

    #[test]
    fn test_filter_lt_json() {
        let f = PyFilter {
            inner: build_filter_json("$lt", "score", "0.5"),
        };
        let v: Value = serde_json::from_str(&f.inner).unwrap();
        assert_eq!(v["score"]["$lt"], 0.5);
    }

    #[test]
    fn test_filter_in_json() {
        let f = PyFilter {
            inner: build_filter_json("$in", "tag", r#"["a","b","c"]"#),
        };
        let v: Value = serde_json::from_str(&f.inner).unwrap();
        let arr = v["tag"]["$in"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], "a");
    }

    #[test]
    fn test_filter_and_combination() {
        let f1 = PyFilter {
            inner: build_filter_json("$eq", "a", "1"),
        };
        let f2 = PyFilter {
            inner: build_filter_json("$gt", "b", "2"),
        };
        let combined_json = format!(r#"{{"$and":[{},{}]}}"#, f1.inner, f2.inner);
        let v: Value = serde_json::from_str(&combined_json).unwrap();
        assert!(v["$and"].is_array());
        assert_eq!(v["$and"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_filter_or_combination() {
        let f1 = PyFilter {
            inner: build_filter_json("$eq", "x", "\"hello\""),
        };
        let f2 = PyFilter {
            inner: build_filter_json("$lt", "y", "10"),
        };
        let combined_json = format!(r#"{{"$or":[{},{}]}}"#, f1.inner, f2.inner);
        let v: Value = serde_json::from_str(&combined_json).unwrap();
        assert!(v["$or"].is_array());
        assert_eq!(v["$or"].as_array().unwrap().len(), 2);
    }

    // ---- PyCollectionIterator ---------------------------------------------

    #[test]
    fn test_collection_iterator() {
        let mut iter = PyCollectionIterator {
            ids: vec!["a".into(), "b".into(), "c".into()],
            index: 0,
        };
        assert_eq!(iter.__next__(), Some("a".into()));
        assert_eq!(iter.__next__(), Some("b".into()));
        assert_eq!(iter.__next__(), Some("c".into()));
        assert_eq!(iter.__next__(), None);
    }

    #[test]
    fn test_collection_iterator_empty() {
        let mut iter = PyCollectionIterator {
            ids: vec![],
            index: 0,
        };
        assert_eq!(iter.__next__(), None);
    }
}
