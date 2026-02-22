#![allow(clippy::unwrap_used)]
//! Schema-Aware Typed Collections
//!
//! Define typed metadata schemas per collection with validation, secondary indexes
//! on metadata fields, and auto-generated filter helpers for sub-millisecond
//! filtered search on high-cardinality fields.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::typed_schema::{
//!     SchemaService, SchemaDefinition, FieldDef, FieldType,
//! };
//!
//! let mut svc = SchemaService::new();
//!
//! // Define a schema
//! let schema = SchemaDefinition::new("documents")
//!     .field("title", FieldType::String, true)
//!     .field("category", FieldType::String, false)
//!     .field("price", FieldType::Float, false)
//!     .field("in_stock", FieldType::Bool, false)
//!     .indexed("category")
//!     .indexed("price");
//!
//! svc.register_schema(schema).unwrap();
//!
//! // Validate metadata
//! let meta = serde_json::json!({"title": "Hello", "category": "books", "price": 9.99});
//! assert!(svc.validate("documents", &meta).is_ok());
//! ```

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Field Types ─────────────────────────────────────────────────────────────

/// Metadata field type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Int,
    Float,
    Bool,
    StringArray,
    IntArray,
}

/// A field definition in a schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    /// Field name.
    pub name: String,
    /// Field type.
    pub field_type: FieldType,
    /// Whether the field is required.
    pub required: bool,
    /// Default value (if any).
    pub default: Option<serde_json::Value>,
    /// Description.
    pub description: Option<String>,
}

impl FieldDef {
    pub fn new(name: impl Into<String>, field_type: FieldType, required: bool) -> Self {
        Self {
            name: name.into(),
            field_type,
            required,
            default: None,
            description: None,
        }
    }

    /// Set a default value.
    #[must_use]
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Set a description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

// ── Schema Definition ───────────────────────────────────────────────────────

/// A schema definition for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaDefinition {
    /// Collection name this schema applies to.
    pub collection: String,
    /// Field definitions.
    pub fields: Vec<FieldDef>,
    /// Fields that have secondary indexes.
    pub indexed_fields: HashSet<String>,
    /// Whether to allow extra fields not in the schema.
    pub allow_extra_fields: bool,
    /// Schema version.
    pub version: u32,
}

impl SchemaDefinition {
    /// Create a new schema for a collection.
    pub fn new(collection: impl Into<String>) -> Self {
        Self {
            collection: collection.into(),
            fields: Vec::new(),
            indexed_fields: HashSet::new(),
            allow_extra_fields: true,
            version: 1,
        }
    }

    /// Add a field to the schema.
    #[must_use]
    pub fn field(mut self, name: impl Into<String>, field_type: FieldType, required: bool) -> Self {
        self.fields.push(FieldDef::new(name, field_type, required));
        self
    }

    /// Add a field with default value.
    #[must_use]
    pub fn field_with_default(
        mut self,
        name: impl Into<String>,
        field_type: FieldType,
        default: serde_json::Value,
    ) -> Self {
        self.fields
            .push(FieldDef::new(name, field_type, false).with_default(default));
        self
    }

    /// Mark a field as indexed for fast filtered search.
    #[must_use]
    pub fn indexed(mut self, field_name: impl Into<String>) -> Self {
        self.indexed_fields.insert(field_name.into());
        self
    }

    /// Set whether extra fields are allowed.
    #[must_use]
    pub fn strict(mut self) -> Self {
        self.allow_extra_fields = false;
        self
    }
}

// ── Secondary Index ─────────────────────────────────────────────────────────

/// A secondary index on a metadata field.
#[derive(Debug, Clone)]
struct SecondaryIndex {
    field_name: String,
    field_type: FieldType,
    // String index: value -> set of vector IDs
    string_index: HashMap<String, HashSet<String>>,
    // Numeric index: BTreeMap for range queries
    numeric_index: BTreeMap<ordered_float::OrderedFloat<f64>, HashSet<String>>,
    // Bool index
    bool_index: HashMap<bool, HashSet<String>>,
    entry_count: usize,
}

impl SecondaryIndex {
    fn new(field_name: String, field_type: FieldType) -> Self {
        Self {
            field_name,
            field_type,
            string_index: HashMap::new(),
            numeric_index: BTreeMap::new(),
            bool_index: HashMap::new(),
            entry_count: 0,
        }
    }

    fn insert(&mut self, vector_id: &str, value: &serde_json::Value) {
        match self.field_type {
            FieldType::String | FieldType::StringArray => {
                if let Some(s) = value.as_str() {
                    self.string_index
                        .entry(s.to_string())
                        .or_default()
                        .insert(vector_id.to_string());
                    self.entry_count += 1;
                }
                if let Some(arr) = value.as_array() {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            self.string_index
                                .entry(s.to_string())
                                .or_default()
                                .insert(vector_id.to_string());
                            self.entry_count += 1;
                        }
                    }
                }
            }
            FieldType::Int | FieldType::Float | FieldType::IntArray => {
                if let Some(n) = value.as_f64() {
                    self.numeric_index
                        .entry(ordered_float::OrderedFloat(n))
                        .or_default()
                        .insert(vector_id.to_string());
                    self.entry_count += 1;
                }
            }
            FieldType::Bool => {
                if let Some(b) = value.as_bool() {
                    self.bool_index
                        .entry(b)
                        .or_default()
                        .insert(vector_id.to_string());
                    self.entry_count += 1;
                }
            }
        }
    }

    fn remove(&mut self, vector_id: &str) {
        for set in self.string_index.values_mut() {
            set.remove(vector_id);
        }
        for set in self.numeric_index.values_mut() {
            set.remove(vector_id);
        }
        for set in self.bool_index.values_mut() {
            set.remove(vector_id);
        }
    }

    fn lookup_eq(&self, value: &serde_json::Value) -> HashSet<String> {
        match self.field_type {
            FieldType::String | FieldType::StringArray => {
                if let Some(s) = value.as_str() {
                    self.string_index.get(s).cloned().unwrap_or_default()
                } else {
                    HashSet::new()
                }
            }
            FieldType::Int | FieldType::Float | FieldType::IntArray => {
                if let Some(n) = value.as_f64() {
                    self.numeric_index
                        .get(&ordered_float::OrderedFloat(n))
                        .cloned()
                        .unwrap_or_default()
                } else {
                    HashSet::new()
                }
            }
            FieldType::Bool => {
                if let Some(b) = value.as_bool() {
                    self.bool_index.get(&b).cloned().unwrap_or_default()
                } else {
                    HashSet::new()
                }
            }
        }
    }

    fn lookup_range(&self, min: f64, max: f64) -> HashSet<String> {
        let mut result = HashSet::new();
        let range = self.numeric_index.range(
            ordered_float::OrderedFloat(min)..=ordered_float::OrderedFloat(max),
        );
        for (_, ids) in range {
            result.extend(ids.iter().cloned());
        }
        result
    }

    fn distinct_values(&self) -> usize {
        match self.field_type {
            FieldType::String | FieldType::StringArray => self.string_index.len(),
            FieldType::Int | FieldType::Float | FieldType::IntArray => self.numeric_index.len(),
            FieldType::Bool => self.bool_index.len(),
        }
    }
}

// ── Index Stats ─────────────────────────────────────────────────────────────

/// Statistics for a secondary index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub field_name: String,
    pub field_type: FieldType,
    pub entry_count: usize,
    pub distinct_values: usize,
}

// ── Validation Result ───────────────────────────────────────────────────────

/// Validation error detail.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

// ── Schema Service ──────────────────────────────────────────────────────────

/// Schema service managing typed collection schemas.
pub struct SchemaService {
    schemas: HashMap<String, SchemaDefinition>,
    indexes: HashMap<String, HashMap<String, SecondaryIndex>>,
}

impl SchemaService {
    /// Create a new schema service.
    pub fn new() -> Self {
        Self {
            schemas: HashMap::new(),
            indexes: HashMap::new(),
        }
    }

    /// Register a schema for a collection.
    pub fn register_schema(&mut self, schema: SchemaDefinition) -> Result<()> {
        // Validate that indexed fields exist in the schema
        for idx_field in &schema.indexed_fields {
            if !schema.fields.iter().any(|f| f.name == *idx_field) {
                return Err(NeedleError::InvalidArgument(format!(
                    "Indexed field '{idx_field}' not found in schema"
                )));
            }
        }

        // Create secondary indexes
        let mut field_indexes = HashMap::new();
        for idx_field in &schema.indexed_fields {
            if let Some(field) = schema.fields.iter().find(|f| f.name == *idx_field) {
                field_indexes.insert(
                    idx_field.clone(),
                    SecondaryIndex::new(idx_field.clone(), field.field_type),
                );
            }
        }

        self.indexes
            .insert(schema.collection.clone(), field_indexes);
        self.schemas.insert(schema.collection.clone(), schema);
        Ok(())
    }

    /// Get a schema for a collection.
    pub fn schema(&self, collection: &str) -> Option<&SchemaDefinition> {
        self.schemas.get(collection)
    }

    /// List all registered schemas.
    pub fn list_schemas(&self) -> Vec<&SchemaDefinition> {
        self.schemas.values().collect()
    }

    /// Validate metadata against a collection's schema.
    pub fn validate(
        &self,
        collection: &str,
        metadata: &serde_json::Value,
    ) -> Result<Vec<ValidationError>> {
        let schema = self.schemas.get(collection).ok_or_else(|| {
            NeedleError::NotFound(format!("Schema for collection '{collection}'"))
        })?;

        let mut errors = Vec::new();
        let obj = metadata.as_object();

        // Check required fields
        for field in &schema.fields {
            if field.required {
                let has_field = obj
                    .is_some_and(|o| o.contains_key(&field.name));
                if !has_field && field.default.is_none() {
                    errors.push(ValidationError {
                        field: field.name.clone(),
                        message: format!("Required field '{}' is missing", field.name),
                    });
                }
            }
        }

        // Check field types
        if let Some(obj) = obj {
            for (key, value) in obj {
                if let Some(field) = schema.fields.iter().find(|f| f.name == *key) {
                    if !Self::type_matches(field.field_type, value) {
                        errors.push(ValidationError {
                            field: key.clone(),
                            message: format!(
                                "Field '{}' expected type {:?}, got {}",
                                key,
                                field.field_type,
                                value_type_name(value)
                            ),
                        });
                    }
                } else if !schema.allow_extra_fields {
                    errors.push(ValidationError {
                        field: key.clone(),
                        message: format!("Unknown field '{key}' (strict mode)"),
                    });
                }
            }
        }

        Ok(errors)
    }

    /// Index a vector's metadata.
    pub fn index_metadata(
        &mut self,
        collection: &str,
        vector_id: &str,
        metadata: &serde_json::Value,
    ) {
        if let Some(indexes) = self.indexes.get_mut(collection) {
            if let Some(obj) = metadata.as_object() {
                for (field, value) in obj {
                    if let Some(idx) = indexes.get_mut(field) {
                        idx.insert(vector_id, value);
                    }
                }
            }
        }
    }

    /// Remove a vector from all indexes.
    pub fn remove_from_indexes(&mut self, collection: &str, vector_id: &str) {
        if let Some(indexes) = self.indexes.get_mut(collection) {
            for idx in indexes.values_mut() {
                idx.remove(vector_id);
            }
        }
    }

    /// Lookup vectors by exact field value using secondary index.
    pub fn lookup_eq(
        &self,
        collection: &str,
        field: &str,
        value: &serde_json::Value,
    ) -> Result<HashSet<String>> {
        let indexes = self.indexes.get(collection).ok_or_else(|| {
            NeedleError::NotFound(format!("Indexes for collection '{collection}'"))
        })?;
        let idx = indexes.get(field).ok_or_else(|| {
            NeedleError::NotFound(format!("Index on field '{field}'"))
        })?;
        Ok(idx.lookup_eq(value))
    }

    /// Lookup vectors by numeric range using secondary index.
    pub fn lookup_range(
        &self,
        collection: &str,
        field: &str,
        min: f64,
        max: f64,
    ) -> Result<HashSet<String>> {
        let indexes = self.indexes.get(collection).ok_or_else(|| {
            NeedleError::NotFound(format!("Indexes for collection '{collection}'"))
        })?;
        let idx = indexes.get(field).ok_or_else(|| {
            NeedleError::NotFound(format!("Index on field '{field}'"))
        })?;
        Ok(idx.lookup_range(min, max))
    }

    /// Get index statistics for a collection.
    pub fn index_stats(&self, collection: &str) -> Vec<IndexStats> {
        self.indexes
            .get(collection)
            .map(|indexes| {
                indexes
                    .values()
                    .map(|idx| IndexStats {
                        field_name: idx.field_name.clone(),
                        field_type: idx.field_type,
                        entry_count: idx.entry_count,
                        distinct_values: idx.distinct_values(),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Apply defaults to metadata based on schema.
    pub fn apply_defaults(
        &self,
        collection: &str,
        metadata: &mut serde_json::Value,
    ) -> Result<()> {
        let schema = self.schemas.get(collection).ok_or_else(|| {
            NeedleError::NotFound(format!("Schema for collection '{collection}'"))
        })?;

        if let Some(obj) = metadata.as_object_mut() {
            for field in &schema.fields {
                if !obj.contains_key(&field.name) {
                    if let Some(default) = &field.default {
                        obj.insert(field.name.clone(), default.clone());
                    }
                }
            }
        }
        Ok(())
    }

    /// Remove a schema.
    pub fn remove_schema(&mut self, collection: &str) -> Result<()> {
        self.schemas.remove(collection).ok_or_else(|| {
            NeedleError::NotFound(format!("Schema for collection '{collection}'"))
        })?;
        self.indexes.remove(collection);
        Ok(())
    }

    fn type_matches(expected: FieldType, value: &serde_json::Value) -> bool {
        match expected {
            FieldType::String => value.is_string(),
            FieldType::Int => value.is_i64() || value.is_u64(),
            FieldType::Float => value.is_number(),
            FieldType::Bool => value.is_boolean(),
            FieldType::StringArray => {
                value
                    .as_array()
                    .is_some_and(|arr| arr.iter().all(|v| v.is_string()))
            }
            FieldType::IntArray => {
                value
                    .as_array()
                    .is_some_and(|arr| arr.iter().all(|v| v.is_i64() || v.is_u64()))
            }
        }
    }
}

impl Default for SchemaService {
    fn default() -> Self {
        Self::new()
    }
}

fn value_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_service() -> SchemaService {
        let mut svc = SchemaService::new();
        let schema = SchemaDefinition::new("docs")
            .field("title", FieldType::String, true)
            .field("category", FieldType::String, false)
            .field("price", FieldType::Float, false)
            .field("in_stock", FieldType::Bool, false)
            .field("tags", FieldType::StringArray, false)
            .indexed("category")
            .indexed("price");
        svc.register_schema(schema).unwrap();
        svc
    }

    #[test]
    fn test_register_schema() {
        let svc = make_service();
        assert!(svc.schema("docs").is_some());
        assert_eq!(svc.schema("docs").unwrap().fields.len(), 5);
    }

    #[test]
    fn test_validate_valid() {
        let svc = make_service();
        let meta = json!({"title": "Hello", "category": "books", "price": 9.99});
        let errors = svc.validate("docs", &meta).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_missing_required() {
        let svc = make_service();
        let meta = json!({"category": "books"});
        let errors = svc.validate("docs", &meta).unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "title");
    }

    #[test]
    fn test_validate_wrong_type() {
        let svc = make_service();
        let meta = json!({"title": 123}); // should be string
        let errors = svc.validate("docs", &meta).unwrap();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("expected type"));
    }

    #[test]
    fn test_validate_strict_mode() {
        let mut svc = SchemaService::new();
        let schema = SchemaDefinition::new("strict_coll")
            .field("name", FieldType::String, true)
            .strict();
        svc.register_schema(schema).unwrap();

        let meta = json!({"name": "test", "extra": "field"});
        let errors = svc.validate("strict_coll", &meta).unwrap();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("Unknown field"));
    }

    #[test]
    fn test_secondary_index_eq() {
        let mut svc = make_service();
        svc.index_metadata("docs", "v1", &json!({"category": "books", "price": 9.99}));
        svc.index_metadata("docs", "v2", &json!({"category": "games", "price": 59.99}));
        svc.index_metadata("docs", "v3", &json!({"category": "books", "price": 19.99}));

        let results = svc.lookup_eq("docs", "category", &json!("books")).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains("v1"));
        assert!(results.contains("v3"));
    }

    #[test]
    fn test_secondary_index_range() {
        let mut svc = make_service();
        svc.index_metadata("docs", "v1", &json!({"price": 9.99}));
        svc.index_metadata("docs", "v2", &json!({"price": 59.99}));
        svc.index_metadata("docs", "v3", &json!({"price": 19.99}));

        let results = svc.lookup_range("docs", "price", 5.0, 25.0).unwrap();
        assert_eq!(results.len(), 2); // v1 and v3
    }

    #[test]
    fn test_remove_from_index() {
        let mut svc = make_service();
        svc.index_metadata("docs", "v1", &json!({"category": "books"}));
        svc.remove_from_indexes("docs", "v1");
        let results = svc.lookup_eq("docs", "category", &json!("books")).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_index_stats() {
        let mut svc = make_service();
        svc.index_metadata("docs", "v1", &json!({"category": "a", "price": 1.0}));
        svc.index_metadata("docs", "v2", &json!({"category": "b", "price": 2.0}));
        svc.index_metadata("docs", "v3", &json!({"category": "a", "price": 3.0}));

        let stats = svc.index_stats("docs");
        assert_eq!(stats.len(), 2);
        let cat_stats = stats.iter().find(|s| s.field_name == "category").unwrap();
        assert_eq!(cat_stats.distinct_values, 2);
    }

    #[test]
    fn test_apply_defaults() {
        let mut svc = SchemaService::new();
        let schema = SchemaDefinition::new("coll")
            .field("title", FieldType::String, true)
            .field_with_default("status", FieldType::String, json!("draft"));
        svc.register_schema(schema).unwrap();

        let mut meta = json!({"title": "Hello"});
        svc.apply_defaults("coll", &mut meta).unwrap();
        assert_eq!(meta["status"], "draft");
    }

    #[test]
    fn test_remove_schema() {
        let mut svc = make_service();
        svc.remove_schema("docs").unwrap();
        assert!(svc.schema("docs").is_none());
    }

    #[test]
    fn test_invalid_indexed_field() {
        let mut svc = SchemaService::new();
        let schema = SchemaDefinition::new("bad")
            .field("name", FieldType::String, true)
            .indexed("nonexistent");
        assert!(svc.register_schema(schema).is_err());
    }

    #[test]
    fn test_bool_index() {
        let mut svc = SchemaService::new();
        let schema = SchemaDefinition::new("coll")
            .field("active", FieldType::Bool, false)
            .indexed("active");
        svc.register_schema(schema).unwrap();

        svc.index_metadata("coll", "v1", &json!({"active": true}));
        svc.index_metadata("coll", "v2", &json!({"active": false}));
        svc.index_metadata("coll", "v3", &json!({"active": true}));

        let results = svc.lookup_eq("coll", "active", &json!(true)).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_string_array_field() {
        let svc = make_service();
        let meta = json!({"title": "Test", "tags": ["rust", "database"]});
        let errors = svc.validate("docs", &meta).unwrap();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_list_schemas() {
        let svc = make_service();
        assert_eq!(svc.list_schemas().len(), 1);
    }

    #[test]
    fn test_validate_unknown_collection() {
        let svc = make_service();
        assert!(svc.validate("unknown", &json!({})).is_err());
    }
}
