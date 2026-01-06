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
use crate::distance::DistanceFunction;
use crate::error::NeedleError;
use crate::metadata::parse_filter;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use serde_json::Value;
use std::sync::{Arc, RwLock};

/// Convert NeedleError to PyErr
fn to_pyerr(err: NeedleError) -> PyErr {
    match err {
        NeedleError::Io(_) => PyIOError::new_err(err.to_string()),
        _ => PyValueError::new_err(err.to_string()),
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
            return Err(PyValueError::new_err("ids and vectors must have the same length"));
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

        coll.insert_batch(ids, vectors, meta_values).map_err(to_pyerr)
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

        let parsed_filter = parse_filter(&filter_value)
            .ok_or_else(|| PyValueError::new_err("Invalid filter format"))?;

        let coll = self
            .inner
            .read()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        let results = coll
            .search_with_filter(&query, k, &parsed_filter)
            .map_err(to_pyerr)?;
        Ok(results.into_iter().map(SearchResult::from_rust).collect())
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

    /// Set ef_search parameter for queries
    fn set_ef_search(&self, ef: usize) -> PyResult<()> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| PyValueError::new_err("Lock poisoned"))?;
        coll.set_ef_search(ef);
        Ok(())
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
}

/// Parse distance function string
fn parse_distance(distance: &str) -> PyResult<DistanceFunction> {
    match distance.to_lowercase().as_str() {
        "cosine" => Ok(DistanceFunction::Cosine),
        "euclidean" | "l2" => Ok(DistanceFunction::Euclidean),
        "dot" | "dotproduct" | "inner_product" => Ok(DistanceFunction::DotProduct),
        "manhattan" | "l1" => Ok(DistanceFunction::Manhattan),
        _ => Err(PyValueError::new_err(format!(
            "Unknown distance function: {}. Use: cosine, euclidean, dot, manhattan",
            distance
        ))),
    }
}

/// Python module definition
#[pymodule]
pub fn needle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SearchResult>()?;
    m.add_class::<PyCollection>()?;

    // Version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
