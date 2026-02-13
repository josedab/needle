//! Typed Vector Namespaces
//!
//! Allows collections to define multiple named vector fields (namespaces),
//! each with independent dimensions and distance functions. Enables
//! multi-vector-per-document storage and multi-field fusion queries.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::services::collection::vector_namespace::*;
//! use needle::DistanceFunction;
//!
//! let mut schema = NamespaceSchema::new("products");
//! schema.add_field(VectorFieldDef {
//!     name: "title_embedding".into(),
//!     dimensions: 384,
//!     distance: DistanceFunction::Cosine,
//!     required: true,
//!     description: Some("Title text embedding".into()),
//! })?;
//! schema.add_field(VectorFieldDef {
//!     name: "image_embedding".into(),
//!     dimensions: 512,
//!     distance: DistanceFunction::Cosine,
//!     required: false,
//!     description: Some("Product image embedding".into()),
//! })?;
//!
//! let mut store = NamespaceStore::new(schema);
//! store.insert("prod1", &[("title_embedding", &title_vec), ("image_embedding", &img_vec)])?;
//!
//! // Fusion query across fields
//! let results = store.fusion_search(&[
//!     FieldQuery { field: "title_embedding", vector: &q1, weight: 0.7 },
//!     FieldQuery { field: "image_embedding", vector: &q2, weight: 0.3 },
//! ], 10)?;
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Schema Definition
// ============================================================================

/// Definition of a single named vector field within a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorFieldDef {
    /// Field name (e.g., "title_embedding", "image_embedding").
    pub name: String,
    /// Dimensionality of vectors in this field.
    pub dimensions: usize,
    /// Distance function for this field.
    pub distance: DistanceFunction,
    /// Whether this field is required on every document.
    pub required: bool,
    /// Optional human-readable description.
    pub description: Option<String>,
}

/// Schema defining the vector namespaces for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceSchema {
    /// Collection name this schema belongs to.
    pub collection: String,
    /// Ordered list of vector field definitions.
    pub fields: Vec<VectorFieldDef>,
}

impl NamespaceSchema {
    /// Create a new namespace schema for the given collection.
    pub fn new(collection: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            fields: Vec::new(),
        }
    }

    /// Add a vector field definition. Returns error if name is duplicate.
    pub fn add_field(&mut self, field: VectorFieldDef) -> Result<()> {
        if field.name.is_empty() {
            return Err(NeedleError::InvalidInput(
                "Vector field name cannot be empty".to_string(),
            ));
        }
        if field.dimensions == 0 {
            return Err(NeedleError::InvalidInput(format!(
                "Vector field '{}' must have dimensions > 0",
                field.name
            )));
        }
        if self.fields.iter().any(|f| f.name == field.name) {
            return Err(NeedleError::InvalidInput(format!(
                "Duplicate vector field name: '{}'",
                field.name
            )));
        }
        self.fields.push(field);
        Ok(())
    }

    /// Get a field definition by name.
    pub fn get_field(&self, name: &str) -> Option<&VectorFieldDef> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// List all field names.
    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Validate that a set of provided fields matches the schema.
    pub fn validate_fields(&self, provided: &[&str]) -> Result<()> {
        for field in &self.fields {
            if field.required && !provided.contains(&field.name.as_str()) {
                return Err(NeedleError::InvalidInput(format!(
                    "Required vector field '{}' is missing",
                    field.name
                )));
            }
        }
        for name in provided {
            if !self.fields.iter().any(|f| f.name == *name) {
                return Err(NeedleError::InvalidInput(format!(
                    "Unknown vector field: '{}'",
                    name
                )));
            }
        }
        Ok(())
    }
}

// ============================================================================
// Storage
// ============================================================================

/// A document's vector data across all namespaces.
#[derive(Debug, Clone)]
struct DocumentVectors {
    /// field_name -> vector data
    fields: HashMap<String, Vec<f32>>,
}

/// In-memory store for multi-namespace vector documents.
pub struct NamespaceStore {
    schema: NamespaceSchema,
    /// doc_id -> document vectors
    documents: HashMap<String, DocumentVectors>,
}

impl NamespaceStore {
    /// Create a new store with the given schema.
    pub fn new(schema: NamespaceSchema) -> Self {
        Self {
            schema,
            documents: HashMap::new(),
        }
    }

    /// Insert a document with vectors for one or more fields.
    pub fn insert(
        &mut self,
        doc_id: &str,
        vectors: &[(&str, &[f32])],
    ) -> Result<()> {
        let field_names: Vec<&str> = vectors.iter().map(|(name, _)| *name).collect();
        self.schema.validate_fields(&field_names)?;

        // Validate dimensions
        for (field_name, vector) in vectors {
            let field_def = self.schema.get_field(field_name).ok_or_else(|| {
                NeedleError::InvalidInput(format!("Unknown field: '{}'", field_name))
            })?;
            if vector.len() != field_def.dimensions {
                return Err(NeedleError::DimensionMismatch {
                    expected: field_def.dimensions,
                    got: vector.len(),
                });
            }
        }

        let mut fields = HashMap::new();
        for (name, vec) in vectors {
            fields.insert(name.to_string(), vec.to_vec());
        }

        self.documents
            .insert(doc_id.to_string(), DocumentVectors { fields });
        Ok(())
    }

    /// Get all vectors for a document.
    pub fn get(&self, doc_id: &str) -> Option<HashMap<&str, &[f32]>> {
        let doc = self.documents.get(doc_id)?;
        let mut result = HashMap::new();
        for (name, vec) in &doc.fields {
            result.insert(name.as_str(), vec.as_slice());
        }
        Some(result)
    }

    /// Get a specific field's vector for a document.
    pub fn get_field(&self, doc_id: &str, field: &str) -> Option<&[f32]> {
        self.documents
            .get(doc_id)
            .and_then(|doc| doc.fields.get(field).map(|v| v.as_slice()))
    }

    /// Delete a document.
    pub fn delete(&mut self, doc_id: &str) -> bool {
        self.documents.remove(doc_id).is_some()
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Search a single field using brute-force distance computation.
    pub fn search_field(
        &self,
        field: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<NamespaceSearchResult>> {
        let field_def = self.schema.get_field(field).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Unknown field: '{}'", field))
        })?;
        if query.len() != field_def.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: field_def.dimensions,
                got: query.len(),
            });
        }

        let mut scored: Vec<(String, f32)> = self
            .documents
            .iter()
            .filter_map(|(doc_id, doc)| {
                let vec = doc.fields.get(field)?;
                let dist = field_def.distance.compute(query, vec).ok()?;
                Some((doc_id.clone(), dist))
            })
            .collect();

        scored.sort_by_key(|(_, d)| OrderedFloat(*d));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(id, distance)| NamespaceSearchResult {
                id,
                distance,
                field: field.to_string(),
            })
            .collect())
    }

    /// Fusion search across multiple fields using weighted score combination.
    ///
    /// Each query specifies a field, query vector, and weight. Results are
    /// combined using reciprocal rank fusion with the specified weights.
    pub fn fusion_search(
        &self,
        queries: &[FieldQuery<'_>],
        k: usize,
    ) -> Result<Vec<FusionSearchResult>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }

        // Per-field search, gather top results
        let per_field_k = k * 3; // over-fetch for better fusion
        let mut doc_scores: HashMap<String, f64> = HashMap::new();

        for query in queries {
            let results = self.search_field(query.field, query.vector, per_field_k)?;
            for (rank, result) in results.iter().enumerate() {
                // Reciprocal rank fusion: score = weight / (rank + 60)
                let rrf_score = query.weight as f64 / (rank as f64 + 60.0);
                *doc_scores.entry(result.id.clone()).or_insert(0.0) += rrf_score;
            }
        }

        let mut scored: Vec<(String, f64)> = doc_scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(id, score)| FusionSearchResult { id, score })
            .collect())
    }

    /// Get the schema.
    pub fn schema(&self) -> &NamespaceSchema {
        &self.schema
    }
}

// ============================================================================
// Query & Result Types
// ============================================================================

/// A single-field search result.
#[derive(Debug, Clone)]
pub struct NamespaceSearchResult {
    /// Document ID.
    pub id: String,
    /// Distance from query.
    pub distance: f32,
    /// Field that was searched.
    pub field: String,
}

/// A query against a specific vector field.
pub struct FieldQuery<'a> {
    /// Field name to search.
    pub field: &'a str,
    /// Query vector.
    pub vector: &'a [f32],
    /// Weight for fusion (0.0-1.0).
    pub weight: f32,
}

/// Result from a multi-field fusion search.
#[derive(Debug, Clone)]
pub struct FusionSearchResult {
    /// Document ID.
    pub id: String,
    /// Combined fusion score (higher is more relevant).
    pub score: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema() -> NamespaceSchema {
        let mut schema = NamespaceSchema::new("products");
        schema
            .add_field(VectorFieldDef {
                name: "title".into(),
                dimensions: 4,
                distance: DistanceFunction::Cosine,
                required: true,
                description: None,
            })
            .expect("add title field");
        schema
            .add_field(VectorFieldDef {
                name: "image".into(),
                dimensions: 3,
                distance: DistanceFunction::Euclidean,
                required: false,
                description: Some("Image embedding".into()),
            })
            .expect("add image field");
        schema
    }

    #[test]
    fn test_schema_validation() {
        let schema = make_schema();
        // title is required
        assert!(schema.validate_fields(&["title"]).is_ok());
        assert!(schema.validate_fields(&["title", "image"]).is_ok());
        assert!(schema.validate_fields(&[]).is_err()); // missing required
        assert!(schema.validate_fields(&["image"]).is_err()); // missing required
        assert!(schema.validate_fields(&["title", "unknown"]).is_err());
    }

    #[test]
    fn test_duplicate_field_name() {
        let mut schema = NamespaceSchema::new("test");
        schema
            .add_field(VectorFieldDef {
                name: "f1".into(),
                dimensions: 4,
                distance: DistanceFunction::Cosine,
                required: true,
                description: None,
            })
            .expect("first field");
        let result = schema.add_field(VectorFieldDef {
            name: "f1".into(),
            dimensions: 8,
            distance: DistanceFunction::Euclidean,
            required: false,
            description: None,
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_and_get() {
        let schema = make_schema();
        let mut store = NamespaceStore::new(schema);

        let title_vec = [0.1, 0.2, 0.3, 0.4];
        let image_vec = [1.0, 2.0, 3.0];

        store
            .insert("doc1", &[("title", &title_vec), ("image", &image_vec)])
            .expect("insert");
        assert_eq!(store.len(), 1);

        let vecs = store.get("doc1").expect("get");
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs["title"], &title_vec[..]);
        assert_eq!(vecs["image"], &image_vec[..]);

        // Get single field
        assert_eq!(store.get_field("doc1", "title"), Some(&title_vec[..]));
        assert!(store.get_field("doc1", "unknown").is_none());
    }

    #[test]
    fn test_dimension_mismatch() {
        let schema = make_schema();
        let mut store = NamespaceStore::new(schema);

        let wrong_dims = [0.1, 0.2]; // title needs 4 dims
        let result = store.insert("doc1", &[("title", &wrong_dims)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_field_search() {
        let schema = make_schema();
        let mut store = NamespaceStore::new(schema);

        store
            .insert("d1", &[("title", &[1.0, 0.0, 0.0, 0.0])])
            .expect("insert d1");
        store
            .insert("d2", &[("title", &[0.0, 1.0, 0.0, 0.0])])
            .expect("insert d2");
        store
            .insert("d3", &[("title", &[1.0, 0.1, 0.0, 0.0])])
            .expect("insert d3");

        let query = [1.0, 0.0, 0.0, 0.0];
        let results = store.search_field("title", &query, 2).expect("search");
        assert_eq!(results.len(), 2);
        // d1 should be closest (cosine)
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_fusion_search() {
        let schema = make_schema();
        let mut store = NamespaceStore::new(schema);

        // d1: good title match, bad image match
        store
            .insert(
                "d1",
                &[
                    ("title", &[1.0, 0.0, 0.0, 0.0]),
                    ("image", &[0.0, 0.0, 1.0]),
                ],
            )
            .expect("d1");
        // d2: bad title match, good image match
        store
            .insert(
                "d2",
                &[
                    ("title", &[0.0, 0.0, 0.0, 1.0]),
                    ("image", &[1.0, 0.0, 0.0]),
                ],
            )
            .expect("d2");
        // d3: moderate both
        store
            .insert(
                "d3",
                &[
                    ("title", &[0.5, 0.5, 0.0, 0.0]),
                    ("image", &[0.5, 0.5, 0.0]),
                ],
            )
            .expect("d3");

        let results = store
            .fusion_search(
                &[
                    FieldQuery {
                        field: "title",
                        vector: &[1.0, 0.0, 0.0, 0.0],
                        weight: 0.5,
                    },
                    FieldQuery {
                        field: "image",
                        vector: &[1.0, 0.0, 0.0],
                        weight: 0.5,
                    },
                ],
                3,
            )
            .expect("fusion search");

        assert_eq!(results.len(), 3);
        // All docs should appear with positive scores
        for r in &results {
            assert!(r.score > 0.0);
        }
    }

    #[test]
    fn test_delete() {
        let schema = make_schema();
        let mut store = NamespaceStore::new(schema);
        store
            .insert("d1", &[("title", &[1.0, 0.0, 0.0, 0.0])])
            .expect("insert");
        assert!(store.delete("d1"));
        assert!(!store.delete("d1")); // already deleted
        assert!(store.is_empty());
    }
}
