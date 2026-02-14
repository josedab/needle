//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Python SDK v2 — Zero-Copy Arrow Bridge
//!
//! Provides a high-performance bridge between Python numpy arrays and Needle's
//! internal vector storage using Apache Arrow-compatible memory layouts. This
//! module is the foundation for the "Python SDK 2.0" that delivers:
//!
//! - **Zero-copy numpy interop**: numpy arrays → Rust slices without copying
//! - **Arrow IPC compatibility**: Columnar exchange for batch operations
//! - **Pydantic-compatible output**: Structured results as typed dictionaries
//! - **Batch operation support**: Efficient bulk insert/search via columnar format
//!
//! # Architecture
//!
//! ```text
//! Python (numpy)  →  Arrow C Data Interface  →  Rust &[f32]  →  Collection
//!                                                              ↓
//! Python (dict)   ←  Arrow IPC / JSON         ←  SearchResult ←
//! ```
//!
//! This module does NOT depend on PyO3 — it provides the data conversion layer
//! that the `python` feature module can use. This keeps it testable without
//! a Python interpreter.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// Memory Layout for Zero-Copy
// ---------------------------------------------------------------------------

/// Describes how vector data is laid out in memory for zero-copy access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryOrder {
    /// Row-major (C-contiguous): vectors are contiguous.
    RowMajor,
    /// Column-major (Fortran-contiguous): dimensions are contiguous.
    ColumnMajor,
}

/// A view into a contiguous f32 buffer representing multiple vectors.
/// Designed for zero-copy access from numpy / Arrow arrays.
#[derive(Debug)]
pub struct VectorArrayView {
    /// Raw f32 data in row-major order.
    pub data: Vec<f32>,
    /// Number of vectors (rows).
    pub num_vectors: usize,
    /// Dimensionality of each vector.
    pub dimension: usize,
    /// Memory layout.
    pub order: MemoryOrder,
}

impl VectorArrayView {
    /// Create a view from a flat buffer with validation.
    pub fn new(data: Vec<f32>, num_vectors: usize, dimension: usize) -> Result<Self> {
        let expected = num_vectors * dimension;
        if data.len() != expected {
            return Err(NeedleError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            num_vectors,
            dimension,
            order: MemoryOrder::RowMajor,
        })
    }

    /// Get the i-th vector as a slice (zero-copy for row-major).
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        if index >= self.num_vectors {
            return None;
        }
        let start = index * self.dimension;
        Some(&self.data[start..start + self.dimension])
    }

    /// Iterate over all vectors.
    pub fn iter(&self) -> VectorArrayIter<'_> {
        VectorArrayIter { view: self, pos: 0 }
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Iterator over vectors in a VectorArrayView.
pub struct VectorArrayIter<'a> {
    view: &'a VectorArrayView,
    pos: usize,
}

impl<'a> Iterator for VectorArrayIter<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.view.get_vector(self.pos)?;
        self.pos += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.num_vectors - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for VectorArrayIter<'a> {}

// ---------------------------------------------------------------------------
// Batch Insert Request
// ---------------------------------------------------------------------------

/// A batch insert request in columnar format for maximum throughput.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertRequest {
    /// Collection name to insert into.
    pub collection: String,
    /// Vector IDs, one per vector.
    pub ids: Vec<String>,
    /// Flat f32 vector data in row-major order.
    pub vectors_flat: Vec<f32>,
    /// Dimensionality.
    pub dimension: usize,
    /// Optional metadata per vector (keyed by index).
    pub metadata: Vec<Option<Value>>,
}

impl BatchInsertRequest {
    /// Validate the request.
    pub fn validate(&self) -> Result<()> {
        if self.ids.is_empty() {
            return Err(NeedleError::InvalidInput("Empty batch".into()));
        }
        let expected_floats = self.ids.len() * self.dimension;
        if self.vectors_flat.len() != expected_floats {
            return Err(NeedleError::DimensionMismatch {
                expected: expected_floats,
                got: self.vectors_flat.len(),
            });
        }
        if self.metadata.len() != self.ids.len() {
            return Err(NeedleError::InvalidInput(format!(
                "Metadata count ({}) must match ID count ({})",
                self.metadata.len(),
                self.ids.len()
            )));
        }
        Ok(())
    }

    /// Get the i-th vector as a slice.
    pub fn get_vector(&self, index: usize) -> Option<&[f32]> {
        if index >= self.ids.len() {
            return None;
        }
        let start = index * self.dimension;
        Some(&self.vectors_flat[start..start + self.dimension])
    }

    /// Number of vectors in this batch.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Batch Insert Result
// ---------------------------------------------------------------------------

/// Result of a batch insert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertResponse {
    /// Number of successfully inserted vectors.
    pub inserted: usize,
    /// Number of failed insertions.
    pub failed: usize,
    /// Errors keyed by vector ID.
    pub errors: HashMap<String, String>,
    /// Wall-clock duration.
    pub duration_ms: u64,
    /// Throughput in vectors per second.
    pub throughput: f64,
}

// ---------------------------------------------------------------------------
// Batch Search Request
// ---------------------------------------------------------------------------

/// A batch search request in columnar format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchRequest {
    /// Collection to search.
    pub collection: String,
    /// Flat f32 query vectors in row-major order.
    pub queries_flat: Vec<f32>,
    /// Number of query vectors.
    pub num_queries: usize,
    /// Dimensionality.
    pub dimension: usize,
    /// Number of results per query.
    pub k: usize,
    /// Optional JSON filter to apply.
    pub filter: Option<Value>,
    /// Whether to include metadata in results.
    pub include_metadata: bool,
}

impl BatchSearchRequest {
    /// Validate the request.
    pub fn validate(&self) -> Result<()> {
        if self.num_queries == 0 {
            return Err(NeedleError::InvalidInput("No queries".into()));
        }
        let expected = self.num_queries * self.dimension;
        if self.queries_flat.len() != expected {
            return Err(NeedleError::DimensionMismatch {
                expected,
                got: self.queries_flat.len(),
            });
        }
        if self.k == 0 {
            return Err(NeedleError::InvalidInput("k must be > 0".into()));
        }
        Ok(())
    }

    /// Get the i-th query vector as a slice.
    pub fn get_query(&self, index: usize) -> Option<&[f32]> {
        if index >= self.num_queries {
            return None;
        }
        let start = index * self.dimension;
        Some(&self.queries_flat[start..start + self.dimension])
    }
}

// ---------------------------------------------------------------------------
// Batch Search Result
// ---------------------------------------------------------------------------

/// A single search result in a format optimised for Python consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonSearchResult {
    pub id: String,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<Value>,
}

/// Results from a batch search, one entry per query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSearchResponse {
    /// Results grouped by query index.
    pub results: Vec<Vec<PythonSearchResult>>,
    /// Total search duration.
    pub duration_ms: u64,
    /// Average latency per query.
    pub avg_latency_ms: f64,
}

// ---------------------------------------------------------------------------
// Type Conversion Helpers
// ---------------------------------------------------------------------------

/// Convert a distance to a similarity score (0-1 range).
pub fn distance_to_similarity(distance: f32, metric: &str) -> f32 {
    match metric {
        "cosine" => 1.0 - distance,
        "euclidean" => 1.0 / (1.0 + distance),
        "dot" | "dotproduct" => {
            // Dot product distances are negative similarities
            1.0 / (1.0 + (-distance).exp())
        }
        "manhattan" => 1.0 / (1.0 + distance),
        _ => 1.0 - distance.min(1.0),
    }
}

/// Convert metadata Value to a Python-friendly typed dictionary representation.
pub fn metadata_to_python_dict(value: &Value) -> HashMap<String, TypedValue> {
    let mut result = HashMap::new();
    if let Value::Object(map) = value {
        for (k, v) in map {
            result.insert(k.clone(), TypedValue::from_json(v));
        }
    }
    result
}

/// A typed value representation for Python interop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum TypedValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<TypedValue>),
    Dict(HashMap<String, TypedValue>),
    Null,
}

impl TypedValue {
    pub fn from_json(value: &Value) -> Self {
        match value {
            Value::String(s) => TypedValue::String(s.clone()),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    TypedValue::Integer(i)
                } else {
                    TypedValue::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            Value::Bool(b) => TypedValue::Boolean(*b),
            Value::Array(arr) => TypedValue::List(arr.iter().map(TypedValue::from_json).collect()),
            Value::Object(obj) => {
                let map = obj
                    .iter()
                    .map(|(k, v)| (k.clone(), TypedValue::from_json(v)))
                    .collect();
                TypedValue::Dict(map)
            }
            Value::Null => TypedValue::Null,
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            TypedValue::String(s) => Value::String(s.clone()),
            TypedValue::Integer(i) => Value::Number((*i).into()),
            TypedValue::Float(f) => {
                Value::Number(serde_json::Number::from_f64(*f).unwrap_or(0.into()))
            }
            TypedValue::Boolean(b) => Value::Bool(*b),
            TypedValue::List(arr) => Value::Array(arr.iter().map(|v| v.to_json()).collect()),
            TypedValue::Dict(map) => {
                let obj: serde_json::Map<String, Value> =
                    map.iter().map(|(k, v)| (k.clone(), v.to_json())).collect();
                Value::Object(obj)
            }
            TypedValue::Null => Value::Null,
        }
    }
}

// ---------------------------------------------------------------------------
// Collection Metadata for Python SDK
// ---------------------------------------------------------------------------

/// Collection info exposed to the Python SDK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub distance_metric: String,
    pub vector_count: usize,
    pub index_type: String,
    pub memory_usage_bytes: usize,
}

/// Database info exposed to the Python SDK.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    pub path: Option<String>,
    pub collections: Vec<CollectionInfo>,
    pub total_vectors: usize,
    pub total_memory_bytes: usize,
    pub version: String,
}

// ---------------------------------------------------------------------------
// High-Level SDK Operations
// ---------------------------------------------------------------------------

/// Timing and throughput stats for an operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub operation: String,
    pub duration_ms: u64,
    pub items_processed: usize,
    pub throughput: f64,
}

impl OperationStats {
    pub fn new(operation: &str, start: Instant, items: usize) -> Self {
        let duration = start.elapsed();
        let ms = duration.as_millis() as u64;
        let throughput = if ms > 0 {
            items as f64 / (ms as f64 / 1000.0)
        } else {
            items as f64
        };
        Self {
            operation: operation.to_string(),
            duration_ms: ms,
            items_processed: items,
            throughput,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_vector_array_view() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = VectorArrayView::new(data, 2, 3).unwrap();

        assert_eq!(view.num_vectors, 2);
        assert_eq!(view.dimension, 3);
        assert_eq!(view.get_vector(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(view.get_vector(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(view.get_vector(2), None);
    }

    #[test]
    fn test_vector_array_view_invalid_size() {
        let data = vec![1.0, 2.0, 3.0];
        let result = VectorArrayView::new(data, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_vector_array_iter() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let view = VectorArrayView::new(data, 2, 2).unwrap();
        let vecs: Vec<&[f32]> = view.iter().collect();
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0], &[1.0, 2.0]);
        assert_eq!(vecs[1], &[3.0, 4.0]);
    }

    #[test]
    fn test_vector_array_byte_size() {
        let data = vec![0.0f32; 100];
        let view = VectorArrayView::new(data, 10, 10).unwrap();
        assert_eq!(view.byte_size(), 400); // 100 floats * 4 bytes
    }

    #[test]
    fn test_batch_insert_request_validate() {
        let req = BatchInsertRequest {
            collection: "test".into(),
            ids: vec!["a".into(), "b".into()],
            vectors_flat: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dimension: 3,
            metadata: vec![None, None],
        };
        assert!(req.validate().is_ok());
        assert_eq!(req.len(), 2);
    }

    #[test]
    fn test_batch_insert_request_validate_mismatch() {
        let req = BatchInsertRequest {
            collection: "test".into(),
            ids: vec!["a".into()],
            vectors_flat: vec![1.0, 2.0], // Should be 3
            dimension: 3,
            metadata: vec![None],
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_batch_insert_request_empty() {
        let req = BatchInsertRequest {
            collection: "test".into(),
            ids: vec![],
            vectors_flat: vec![],
            dimension: 3,
            metadata: vec![],
        };
        assert!(req.validate().is_err());
        assert!(req.is_empty());
    }

    #[test]
    fn test_batch_search_request_validate() {
        let req = BatchSearchRequest {
            collection: "test".into(),
            queries_flat: vec![1.0, 2.0, 3.0],
            num_queries: 1,
            dimension: 3,
            k: 10,
            filter: None,
            include_metadata: true,
        };
        assert!(req.validate().is_ok());
        assert_eq!(req.get_query(0), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn test_batch_search_request_k_zero() {
        let req = BatchSearchRequest {
            collection: "test".into(),
            queries_flat: vec![1.0, 2.0, 3.0],
            num_queries: 1,
            dimension: 3,
            k: 0,
            filter: None,
            include_metadata: true,
        };
        assert!(req.validate().is_err());
    }

    #[test]
    fn test_distance_to_similarity() {
        assert!((distance_to_similarity(0.0, "cosine") - 1.0).abs() < 0.001);
        assert!((distance_to_similarity(1.0, "cosine") - 0.0).abs() < 0.001);
        assert!((distance_to_similarity(0.0, "euclidean") - 1.0).abs() < 0.001);
        assert!(distance_to_similarity(10.0, "euclidean") < 0.1);
    }

    #[test]
    fn test_typed_value_roundtrip() {
        let original = json!({
            "name": "test",
            "count": 42,
            "score": 0.95,
            "active": true,
            "tags": ["a", "b"],
            "nested": {"key": "val"}
        });

        let typed = metadata_to_python_dict(&original);
        assert!(matches!(typed.get("name"), Some(TypedValue::String(_))));
        assert!(matches!(typed.get("count"), Some(TypedValue::Integer(42))));
        assert!(matches!(
            typed.get("active"),
            Some(TypedValue::Boolean(true))
        ));

        // Round-trip
        for (key, tv) in &typed {
            let json_val = tv.to_json();
            let tv2 = TypedValue::from_json(&json_val);
            assert_eq!(
                format!("{:?}", tv),
                format!("{:?}", tv2),
                "Round-trip failed for key: {}",
                key
            );
        }
    }

    #[test]
    fn test_typed_value_null() {
        let tv = TypedValue::from_json(&Value::Null);
        assert!(matches!(tv, TypedValue::Null));
        assert_eq!(tv.to_json(), Value::Null);
    }

    #[test]
    fn test_operation_stats() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let stats = OperationStats::new("insert", start, 100);
        assert_eq!(stats.operation, "insert");
        assert!(stats.duration_ms >= 10);
        assert_eq!(stats.items_processed, 100);
    }

    #[test]
    fn test_collection_info_serialization() {
        let info = CollectionInfo {
            name: "docs".into(),
            dimension: 384,
            distance_metric: "cosine".into(),
            vector_count: 1000,
            index_type: "hnsw".into(),
            memory_usage_bytes: 1024 * 1024,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: CollectionInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "docs");
        assert_eq!(decoded.dimension, 384);
    }

    #[test]
    fn test_batch_insert_get_vector() {
        let req = BatchInsertRequest {
            collection: "test".into(),
            ids: vec!["a".into(), "b".into()],
            vectors_flat: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dimension: 3,
            metadata: vec![None, None],
        };
        assert_eq!(req.get_vector(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(req.get_vector(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(req.get_vector(2), None);
    }

    #[test]
    fn test_python_search_result() {
        let result = PythonSearchResult {
            id: "doc1".into(),
            distance: 0.15,
            score: 0.85,
            metadata: Some(json!({"title": "Hello"})),
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["id"], "doc1");
        assert!((json["score"].as_f64().unwrap() - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_memory_order() {
        let view = VectorArrayView {
            data: vec![1.0, 2.0],
            num_vectors: 1,
            dimension: 2,
            order: MemoryOrder::ColumnMajor,
        };
        assert_eq!(view.order, MemoryOrder::ColumnMajor);
    }
}
