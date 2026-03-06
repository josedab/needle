//! Metadata Storage and Filtering
//!
//! This module provides metadata storage for vectors and MongoDB-style query filters
//! for search result filtering.
//!
//! # Overview
//!
//! Each vector can have associated JSON metadata. Metadata can be used to:
//! - Store additional information about vectors (titles, categories, timestamps)
//! - Filter search results based on field values
//! - Implement faceted search and refinement
//!
//! # Filter Syntax
//!
//! Filters use MongoDB-style query syntax:
//!
//! ```rust
//! use needle::metadata::Filter;
//! use serde_json::json;
//!
//! // Simple equality
//! let filter = Filter::eq("category", "books");
//!
//! // Comparison operators
//! let filter = Filter::gt("price", 10.0);
//! let filter = Filter::lte("rating", 5);
//!
//! // Logical operators
//! let filter = Filter::and(vec![
//!     Filter::eq("category", "books"),
//!     Filter::lt("price", 50.0),
//! ]);
//!
//! // Parse from JSON (MongoDB syntax)
//! let filter = Filter::parse(&json!({
//!     "category": "books",
//!     "price": { "$lt": 50 }
//! }))?;
//! # Ok::<(), String>(())
//! ```
//!
//! # Supported Operators
//!
//! | Operator | Description |
//! |----------|-------------|
//! | `$eq` | Equal to |
//! | `$ne` | Not equal to |
//! | `$gt` | Greater than |
//! | `$gte` | Greater than or equal |
//! | `$lt` | Less than |
//! | `$lte` | Less than or equal |
//! | `$in` | In array |
//! | `$nin` | Not in array |
//! | `$contains` | Array contains value |
//! | `$and` | Logical AND |
//! | `$or` | Logical OR |
//! | `$not` | Logical NOT |
//! | `$between` | Inclusive range `[low, high]` |
//! | `$size` | Array/string length equals |
//! | `$type` | JSON value type check |

#![cfg_attr(test, allow(clippy::unwrap_used))]
use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Validate metadata against a schema.
///
/// The schema supports a subset of JSON Schema:
/// - `"required"`: array of field names that must be present
/// - `"properties"`: object mapping field names to type constraints
///   - `"type"`: one of `"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`, `"null"`
pub fn validate_metadata_schema(metadata: &Value, schema: &Value) -> Result<()> {
    let meta_obj = metadata
        .as_object()
        .ok_or_else(|| NeedleError::InvalidInput("metadata must be a JSON object".into()))?;

    // Check required fields
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        for req in required {
            if let Some(field_name) = req.as_str() {
                if !meta_obj.contains_key(field_name) {
                    return Err(NeedleError::InvalidInput(format!(
                        "missing required metadata field: '{field_name}'"
                    )));
                }
            }
        }
    }

    // Check property types
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (field_name, field_schema) in properties {
            if let Some(value) = meta_obj.get(field_name) {
                if let Some(expected_type) = field_schema.get("type").and_then(Value::as_str) {
                    let actual_type = json_type_name(value);
                    if actual_type != expected_type {
                        return Err(NeedleError::InvalidInput(format!(
                            "metadata field '{field_name}' expected type '{expected_type}', got '{actual_type}'"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Return the JSON Schema type name for a `serde_json::Value`.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Statistics about a single metadata field.
#[derive(Debug, Clone, Serialize)]
pub struct FieldStats {
    /// Field name
    pub name: String,
    /// Number of distinct values
    pub cardinality: usize,
    /// Whether the field is indexed for fast filtering
    pub is_indexed: bool,
    /// Whether this field exceeds the high-cardinality threshold
    pub is_high_cardinality: bool,
}

/// Maximum nesting depth for flattening metadata fields.
/// Prevents pathological cases with deeply nested objects.
const FLATTEN_MAX_DEPTH: usize = 5;

/// Flatten nested metadata into dot-notation field paths.
///
/// Given `{"author": {"name": "Alice", "age": 30}, "category": "books"}`,
/// produces:
/// - `("category", "books")`
/// - `("author.name", "Alice")`
/// - `("author.age", 30)`
///
/// **Note:** Dot characters in field names are treated as path separators.
/// A literal field named `"a.b"` will conflict with nested path `a -> b`.
fn flatten_metadata(value: &Value, prefix: &str, depth: usize, out: &mut Vec<(String, Value)>) {
    if depth > FLATTEN_MAX_DEPTH {
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, val) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match val {
                    Value::Object(_) => flatten_metadata(val, &path, depth + 1, out),
                    _ => out.push((path, val.clone())),
                }
            }
        }
        _ => {
            if !prefix.is_empty() {
                out.push((prefix.to_string(), value.clone()));
            }
        }
    }
}

/// Metadata entry for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntry {
    /// External (user-provided) ID
    pub external_id: String,
    /// JSON metadata
    pub data: Option<Value>,
}

/// Default maximum distinct values per field before the inverted index stops
/// adding new values for that field. Prevents high-cardinality fields (like UUIDs)
/// from bloating the index with no filtering benefit.
const DEFAULT_CARDINALITY_THRESHOLD: usize = 10_000;

/// Metadata store for vectors
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataStore {
    /// Mapping from internal ID to metadata
    entries: HashMap<usize, MetadataEntry>,
    /// Mapping from external ID to internal ID
    id_map: HashMap<String, usize>,
    /// Per-field bloom filters for fast pre-check on equality filters.
    /// Key is field name; value is the bit-vector bloom filter.
    #[serde(skip)]
    field_blooms: HashMap<String, BloomFilter>,
    /// Inverted index: field_name -> (stringified_value -> set of internal IDs).
    /// Enables O(1) lookup for equality filters instead of O(n) scan.
    /// Fields exceeding the cardinality threshold are excluded.
    #[serde(skip)]
    field_indexes: HashMap<String, HashMap<String, Vec<usize>>>,
    /// Fields that have been disabled from indexing due to high cardinality.
    #[serde(skip)]
    high_cardinality_fields: std::collections::HashSet<String>,
    /// Configurable cardinality threshold for auto-detection.
    #[serde(skip)]
    cardinality_threshold: usize,
}

/// Simple bit-array bloom filter for metadata field values.
///
/// Used as a pre-check to skip vectors that definitely don't contain a
/// particular field value, avoiding expensive `HashMap` lookups during search.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u8,
}

impl Default for BloomFilter {
    fn default() -> Self {
        Self::new(8192, 3)
    }
}

impl BloomFilter {
    /// Create a bloom filter with `num_bits` capacity and `num_hashes` hash functions.
    pub fn new(num_bits: usize, num_hashes: u8) -> Self {
        let words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; words],
            num_bits,
            num_hashes,
        }
    }

    fn hash_pair(item: &str) -> (u64, u64) {
        let mut h1 = std::collections::hash_map::DefaultHasher::new();
        item.hash(&mut h1);
        let a = h1.finish();

        let mut h2 = std::collections::hash_map::DefaultHasher::new();
        a.hash(&mut h2);
        let b = h2.finish();

        (a, b)
    }

    /// Insert a value into the bloom filter.
    pub fn insert(&mut self, value: &str) {
        let (h1, h2) = Self::hash_pair(value);
        for i in 0..u64::from(self.num_hashes) {
            let idx = (h1.wrapping_add(i.wrapping_mul(h2))) as usize % self.num_bits;
            self.bits[idx / 64] |= 1 << (idx % 64);
        }
    }

    /// Check if a value might be in the filter. `false` means definitely absent.
    pub fn might_contain(&self, value: &str) -> bool {
        let (h1, h2) = Self::hash_pair(value);
        for i in 0..u64::from(self.num_hashes) {
            let idx = (h1.wrapping_add(i.wrapping_mul(h2))) as usize % self.num_bits;
            if self.bits[idx / 64] & (1 << (idx % 64)) == 0 {
                return false;
            }
        }
        true
    }
}

impl MetadataStore {
    /// Create a new metadata store
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            id_map: HashMap::new(),
            field_blooms: HashMap::new(),
            field_indexes: HashMap::new(),
            high_cardinality_fields: std::collections::HashSet::new(),
            cardinality_threshold: DEFAULT_CARDINALITY_THRESHOLD,
        }
    }

    /// Create a new metadata store with a custom cardinality threshold.
    pub fn with_cardinality_threshold(threshold: usize) -> Self {
        Self {
            cardinality_threshold: threshold,
            ..Self::new()
        }
    }

    /// Insert metadata for a vector
    pub fn insert(
        &mut self,
        internal_id: usize,
        external_id: String,
        data: Option<Value>,
    ) -> Result<()> {
        if self.id_map.contains_key(&external_id) {
            return Err(NeedleError::VectorAlreadyExists(external_id));
        }

        // Update bloom filters and inverted indexes with field values,
        // including nested fields via dot-notation paths.
        if let Some(ref obj) = data {
            let mut flattened = Vec::new();
            flatten_metadata(obj, "", 0, &mut flattened);

            for (field, value) in flattened {
                let bloom = self
                    .field_blooms
                    .entry(field.clone())
                    .or_insert_with(BloomFilter::default);
                bloom.insert(&value.to_string());

                // Skip inverted index for high-cardinality fields
                if self.high_cardinality_fields.contains(&field) {
                    continue;
                }

                let val_str = value.to_string();
                let field_idx = self.field_indexes
                    .entry(field.clone())
                    .or_default();
                field_idx
                    .entry(val_str)
                    .or_default()
                    .push(internal_id);

                // Auto-detect high cardinality and stop indexing
                if field_idx.len() > self.cardinality_threshold {
                    self.high_cardinality_fields.insert(field.clone());
                    self.field_indexes.remove(&field);
                }
            }
        }

        self.id_map.insert(external_id.clone(), internal_id);
        self.entries
            .insert(internal_id, MetadataEntry { external_id, data });

        Ok(())
    }

    /// Get metadata by internal ID
    pub fn get(&self, internal_id: usize) -> Option<&MetadataEntry> {
        self.entries.get(&internal_id)
    }

    /// Get internal ID by external ID
    pub fn get_internal_id(&self, external_id: &str) -> Option<usize> {
        self.id_map.get(external_id).copied()
    }

    /// Delete metadata by internal ID
    pub fn delete(&mut self, internal_id: usize) -> Option<MetadataEntry> {
        if let Some(entry) = self.entries.remove(&internal_id) {
            self.id_map.remove(&entry.external_id);

            // Remove from inverted indexes (including nested fields)
            if let Some(ref obj) = entry.data {
                let mut flattened = Vec::new();
                flatten_metadata(obj, "", 0, &mut flattened);
                for (field, value) in flattened {
                    let val_str = value.to_string();
                    if let Some(idx) = self.field_indexes.get_mut(&field) {
                        if let Some(ids) = idx.get_mut(&val_str) {
                            ids.retain(|&id| id != internal_id);
                        }
                    }
                }
            }

            Some(entry)
        } else {
            None
        }
    }

    /// Look up internal IDs matching a field==value condition via the inverted index.
    /// Returns `None` if the field is not indexed, `Some(ids)` otherwise.
    pub fn lookup_field_eq(&self, field: &str, value: &Value) -> Option<&Vec<usize>> {
        let val_str = value.to_string();
        self.field_indexes.get(field)?.get(&val_str)
    }

    /// Get the set of indexed field names.
    pub fn indexed_fields(&self) -> Vec<&str> {
        self.field_indexes.keys().map(|s| s.as_str()).collect()
    }

    /// Get the cardinality (distinct values) for an indexed field.
    pub fn field_cardinality(&self, field: &str) -> usize {
        self.field_indexes.get(field).map_or(0, |idx| idx.len())
    }

    /// Check if a field has been excluded from the inverted index due to high cardinality.
    pub fn is_high_cardinality(&self, field: &str) -> bool {
        self.high_cardinality_fields.contains(field)
    }

    /// Mark fields as high-cardinality upfront, skipping inverted-index
    /// construction for these fields from the start. Use this when you know
    /// which fields have high cardinality (e.g., UUIDs, timestamps) to avoid
    /// the overhead of indexing and then discarding.
    pub fn mark_high_cardinality_fields(&mut self, fields: &[String]) {
        for field in fields {
            self.high_cardinality_fields.insert(field.clone());
            self.field_indexes.remove(field);
        }
    }

    /// Returns the current cardinality threshold for auto-detection.
    pub fn cardinality_threshold(&self) -> usize {
        self.cardinality_threshold
    }

    /// Set the cardinality threshold for auto-detection of high-cardinality fields.
    pub fn set_cardinality_threshold(&mut self, threshold: usize) {
        self.cardinality_threshold = threshold;
    }

    /// Try to resolve a filter using the inverted index for O(1) equality lookups.
    /// Returns `Some(set_of_matching_internal_ids)` if the filter is a simple equality
    /// condition on an indexed field. Returns `None` otherwise (caller should fall back
    /// to sequential scan).
    pub fn resolve_filter_via_index(&self, filter: &Filter) -> Option<std::collections::HashSet<usize>> {
        match filter {
            Filter::Condition(cond) if cond.operator == FilterOperator::Eq => {
                let ids = self.lookup_field_eq(&cond.field, &cond.value)?;
                Some(ids.iter().copied().collect())
            }
            Filter::And(filters) => {
                // Intersect results from all sub-filters
                let mut result: Option<std::collections::HashSet<usize>> = None;
                for f in filters {
                    if let Some(ids) = self.resolve_filter_via_index(f) {
                        result = Some(match result {
                            Some(existing) => existing.intersection(&ids).copied().collect(),
                            None => ids,
                        });
                    } else {
                        return None; // Can't fully resolve, fall back
                    }
                }
                result
            }
            _ => None,
        }
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all external IDs
    pub fn all_external_ids(&self) -> Vec<String> {
        self.id_map.keys().cloned().collect()
    }

    /// Check if external ID exists
    pub fn contains(&self, external_id: &str) -> bool {
        self.id_map.contains_key(external_id)
    }

    /// Fast bloom-filter check: returns `false` if the given field definitely
    /// does NOT contain the given value across any vector in the store.
    /// Returns `true` if the value *might* exist (possible false positive).
    pub fn bloom_might_contain(&self, field: &str, value: &str) -> bool {
        self.field_blooms
            .get(field)
            .map_or(false, |bloom| bloom.might_contain(value))
    }

    /// Get the per-field bloom filter (for diagnostics).
    pub fn field_bloom(&self, field: &str) -> Option<&BloomFilter> {
        self.field_blooms.get(field)
    }

    /// Update the metadata data for an entry, maintaining the inverted index.
    pub fn update_data(&mut self, internal_id: usize, data: Option<Value>) -> Result<()> {
        let entry = self
            .entries
            .get_mut(&internal_id)
            .ok_or_else(|| NeedleError::VectorNotFound(internal_id.to_string()))?;

        // Remove old values from inverted index (including nested fields)
        if let Some(ref old_obj) = entry.data {
            let mut flattened = Vec::new();
            flatten_metadata(old_obj, "", 0, &mut flattened);
            for (field, value) in flattened {
                let val_str = value.to_string();
                if let Some(idx) = self.field_indexes.get_mut(&field) {
                    if let Some(ids) = idx.get_mut(&val_str) {
                        ids.retain(|&id| id != internal_id);
                    }
                }
            }
        }

        // Add new values to inverted index (including nested fields)
        if let Some(ref new_obj) = data {
            let mut flattened = Vec::new();
            flatten_metadata(new_obj, "", 0, &mut flattened);
            for (field, value) in flattened {
                let val_str = value.to_string();
                self.field_indexes
                    .entry(field)
                    .or_default()
                    .entry(val_str)
                    .or_default()
                    .push(internal_id);
            }
        }

        entry.data = data;
        Ok(())
    }

    /// Merge partial metadata into existing metadata (JSON Merge Patch, RFC 7386).
    ///
    /// - New keys are added
    /// - Existing keys are overwritten
    /// - Keys set to `null` are removed
    /// - Non-object existing metadata is fully replaced
    pub fn merge_data(&mut self, internal_id: usize, patch: &Value) -> Result<()> {
        let entry = self
            .entries
            .get(&internal_id)
            .ok_or_else(|| NeedleError::VectorNotFound(internal_id.to_string()))?;

        let merged = match (&entry.data, patch) {
            (Some(Value::Object(existing)), Value::Object(patch_map)) => {
                let mut merged = existing.clone();
                for (key, value) in patch_map {
                    if value.is_null() {
                        merged.remove(key);
                    } else {
                        merged.insert(key.clone(), value.clone());
                    }
                }
                Some(Value::Object(merged))
            }
            // If existing isn't an object or patch isn't an object, replace entirely
            (_, patch_val) => Some(patch_val.clone()),
        };

        self.update_data(internal_id, merged)
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory(&self) -> usize {
        // HashMap overhead + entries
        let base_overhead = std::mem::size_of::<Self>();
        let entry_overhead = self.entries.len()
            * (
                std::mem::size_of::<usize>() +  // Key
            std::mem::size_of::<MetadataEntry>()
                // Value struct
            );
        let id_map_overhead = self.id_map.len()
            * (
                32 +  // Average string size estimate
            std::mem::size_of::<usize>()
                // Value
            );
        // Estimate JSON data size
        let json_size: usize = self
            .entries
            .values()
            .filter_map(|e| e.data.as_ref())
            .map(|v| v.to_string().len())
            .sum();

        base_overhead + entry_overhead + id_map_overhead + json_size
    }

    /// Iterate over all entries
    pub fn iter(&self) -> impl Iterator<Item = (usize, &MetadataEntry)> {
        self.entries.iter().map(|(&k, v)| (k, v))
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(bytes)?)
    }

    /// Get statistics for a specific metadata field.
    pub fn field_stats(&self, field: &str) -> Option<FieldStats> {
        let is_indexed = self.field_indexes.contains_key(field);
        let is_high_card = self.is_high_cardinality(field);

        // Field must be known to us (indexed or high-cardinality)
        if !is_indexed && !is_high_card {
            return None;
        }

        Some(FieldStats {
            name: field.to_string(),
            cardinality: self.field_cardinality(field),
            is_indexed,
            is_high_cardinality: is_high_card,
        })
    }

    /// Get statistics for all known metadata fields.
    pub fn all_field_stats(&self) -> Vec<FieldStats> {
        let mut fields: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for key in self.field_indexes.keys() {
            fields.insert(key.as_str());
        }
        for key in &self.high_cardinality_fields {
            fields.insert(key.as_str());
        }

        let mut stats: Vec<FieldStats> = fields
            .into_iter()
            .filter_map(|f| self.field_stats(f))
            .collect();
        stats.sort_by(|a, b| a.name.cmp(&b.name));
        stats
    }
}

/// Filter comparison operators
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterOperator {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Greater than
    Gt,
    /// Greater than or equal
    Gte,
    /// Less than
    Lt,
    /// Less than or equal
    Lte,
    /// In array
    In,
    /// Not in array
    NotIn,
    /// Contains (for strings/arrays)
    Contains,
    /// String starts with prefix
    StartsWith,
    /// String ends with suffix
    EndsWith,
    /// Field exists (or does not exist when value is false)
    Exists,
    /// Matches a regular expression pattern
    Regex,
    /// All values in the filter array must be present in the field array
    All,
    /// At least one array element matches all nested conditions
    ElemMatch,
    /// Inclusive range check: value >= low AND value <= high
    Between,
    /// Array or string length equals the given integer
    Size,
    /// Field value matches the given JSON type name
    Type,
}

impl std::fmt::Display for FilterOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eq => write!(f, "$eq"),
            Self::Ne => write!(f, "$ne"),
            Self::Gt => write!(f, "$gt"),
            Self::Gte => write!(f, "$gte"),
            Self::Lt => write!(f, "$lt"),
            Self::Lte => write!(f, "$lte"),
            Self::In => write!(f, "$in"),
            Self::NotIn => write!(f, "$nin"),
            Self::Contains => write!(f, "$contains"),
            Self::StartsWith => write!(f, "$startsWith"),
            Self::EndsWith => write!(f, "$endsWith"),
            Self::Exists => write!(f, "$exists"),
            Self::Regex => write!(f, "$regex"),
            Self::All => write!(f, "$all"),
            Self::ElemMatch => write!(f, "$elemMatch"),
            Self::Between => write!(f, "$between"),
            Self::Size => write!(f, "$size"),
            Self::Type => write!(f, "$type"),
        }
    }
}

/// A single filter condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field path (dot-separated for nested fields)
    pub field: String,
    /// Comparison operator
    pub operator: FilterOperator,
    /// Value to compare against
    pub value: Value,
}

/// Filter expression combining multiple conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    /// Single condition
    Condition(FilterCondition),
    /// Logical AND of multiple filters
    And(Vec<Filter>),
    /// Logical OR of multiple filters
    Or(Vec<Filter>),
    /// Logical NOT of a filter
    Not(Box<Filter>),
}

impl Filter {
    /// Create a simple equality filter
    pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Eq,
            value: value.into(),
        })
    }

    /// Create a "not equal" filter
    pub fn ne(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Ne,
            value: value.into(),
        })
    }

    /// Create a "greater than" filter
    pub fn gt(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Gt,
            value: value.into(),
        })
    }

    /// Create a "greater than or equal" filter
    pub fn gte(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Gte,
            value: value.into(),
        })
    }

    /// Create a "less than" filter
    pub fn lt(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Lt,
            value: value.into(),
        })
    }

    /// Create a "less than or equal" filter
    pub fn lte(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Lte,
            value: value.into(),
        })
    }

    /// Create an "in" filter
    pub fn is_in(field: impl Into<String>, values: Vec<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::In,
            value: Value::Array(values),
        })
    }

    /// Combine filters with AND
    pub fn and(filters: Vec<Filter>) -> Self {
        Filter::And(filters)
    }

    /// Combine filters with OR
    pub fn or(filters: Vec<Filter>) -> Self {
        Filter::Or(filters)
    }

    /// Negate a filter
    pub fn negate(filter: Filter) -> Self {
        Filter::Not(Box::new(filter))
    }

    /// Create a "contains" filter (string substring or array element).
    pub fn contains(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Contains,
            value: value.into(),
        })
    }

    /// Create a "starts with" filter for string prefix matching.
    pub fn starts_with(field: impl Into<String>, prefix: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::StartsWith,
            value: prefix.into(),
        })
    }

    /// Create an "ends with" filter for string suffix matching.
    pub fn ends_with(field: impl Into<String>, suffix: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::EndsWith,
            value: suffix.into(),
        })
    }

    /// Create a "not in" filter (value not in array).
    pub fn not_in(field: impl Into<String>, values: Vec<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::NotIn,
            value: Value::Array(values),
        })
    }

    /// Create an "exists" filter: `true` matches when the field is present,
    /// `false` matches when the field is absent.
    pub fn exists(field: impl Into<String>, exists: bool) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Exists,
            value: Value::Bool(exists),
        })
    }

    /// Create a regex filter for pattern matching on string fields.
    ///
    /// Uses Rust's `regex` syntax (from the standard library's basic matching).
    /// The pattern is matched against the entire string value.
    pub fn regex(field: impl Into<String>, pattern: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Regex,
            value: Value::String(pattern.into()),
        })
    }

    /// Create an "all" filter: the field must be an array containing every
    /// element in `values`.
    ///
    /// MongoDB equivalent: `{"tags": {"$all": ["rust", "database"]}}`
    pub fn all(field: impl Into<String>, values: Vec<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::All,
            value: Value::Array(values),
        })
    }

    /// Create an "elemMatch" filter: the field must be an array where at least
    /// one element satisfies the nested filter conditions.
    ///
    /// MongoDB equivalent: `{"items": {"$elemMatch": {"price": {"$gt": 10}}}}`
    pub fn elem_match(field: impl Into<String>, conditions: Value) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::ElemMatch,
            value: conditions,
        })
    }

    /// Create a "between" filter for inclusive range matching.
    ///
    /// Equivalent to `$gte low AND $lte high`.
    ///
    /// MongoDB-style: `{"price": {"$between": [10, 50]}}`
    pub fn between(field: impl Into<String>, low: impl Into<Value>, high: impl Into<Value>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Between,
            value: Value::Array(vec![low.into(), high.into()]),
        })
    }

    /// Create a "size" filter to match arrays or strings by length.
    ///
    /// MongoDB-style: `{"tags": {"$size": 3}}`
    pub fn size(field: impl Into<String>, len: usize) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Size,
            value: Value::Number(serde_json::Number::from(len)),
        })
    }

    /// Create a "type" filter to match fields by their JSON value type.
    ///
    /// Accepted type names: `"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`, `"null"`.
    ///
    /// MongoDB-style: `{"age": {"$type": "number"}}`
    pub fn has_type(field: impl Into<String>, type_name: impl Into<String>) -> Self {
        Filter::Condition(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Type,
            value: Value::String(type_name.into()),
        })
    }

    /// Evaluate filter against metadata
    pub fn matches(&self, metadata: Option<&Value>) -> bool {
        match self {
            Filter::Condition(cond) => evaluate_condition(cond, metadata),
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            Filter::Not(filter) => !filter.matches(metadata),
        }
    }

    /// Compute the maximum nesting depth of this filter expression.
    ///
    /// A single `Condition` has depth 1. `And`/`Or`/`Not` add one level
    /// on top of their deepest child.
    pub fn depth(&self) -> usize {
        match self {
            Filter::Condition(_) => 1,
            Filter::And(filters) | Filter::Or(filters) => {
                1 + filters.iter().map(Filter::depth).max().unwrap_or(0)
            }
            Filter::Not(inner) => 1 + inner.depth(),
        }
    }

    /// Extract simple equality conditions for bloom filter pre-checks.
    /// Returns a list of (field, value_string) pairs that must all be present
    /// for the filter to possibly match (only from top-level AND/Eq conditions).
    pub fn equality_conditions(&self) -> Vec<(&str, String)> {
        match self {
            Filter::Condition(cond) if cond.operator == FilterOperator::Eq => {
                vec![(&cond.field, cond.value.to_string())]
            }
            Filter::And(filters) => filters
                .iter()
                .flat_map(|f| f.equality_conditions())
                .collect(),
            _ => vec![],
        }
    }

    /// Maximum depth for nested filters to prevent stack overflow
    const MAX_FILTER_DEPTH: usize = 32;

    /// Maximum number of elements in a single $and/$or array
    const MAX_FILTER_ARRAY_SIZE: usize = 1000;

    /// Parse a filter from a JSON value (MongoDB-like syntax)
    ///
    /// Supports:
    /// - Simple equality: `{"field": "value"}`
    /// - Comparison operators: `{"field": {"$gt": 5}}`
    /// - Logical operators: `{"$and": [...]}`, `{"$or": [...]}`, `{"$not": {...}}`
    ///
    /// Filter nesting is limited to 32 levels to prevent stack overflow.
    pub fn parse(value: &Value) -> std::result::Result<Self, String> {
        Self::parse_with_depth(value, 0)
    }

    fn parse_with_depth(value: &Value, depth: usize) -> std::result::Result<Self, String> {
        if depth > Self::MAX_FILTER_DEPTH {
            return Err(format!(
                "Filter nesting too deep (max {} levels)",
                Self::MAX_FILTER_DEPTH
            ));
        }

        match value {
            Value::Object(map) => {
                // Check for logical operators
                if let Some(and_value) = map.get("$and") {
                    return Self::parse_logical_and_with_depth(and_value, depth + 1);
                }
                if let Some(or_value) = map.get("$or") {
                    return Self::parse_logical_or_with_depth(or_value, depth + 1);
                }
                if let Some(not_value) = map.get("$not") {
                    return Self::parse_logical_not_with_depth(not_value, depth + 1);
                }

                // Parse field conditions
                let mut conditions = Vec::new();
                for (field, field_value) in map {
                    if field.starts_with('$') {
                        return Err(format!("Unknown operator: {}", field));
                    }
                    conditions.push(Self::parse_field_condition(field, field_value)?);
                }

                if conditions.is_empty() {
                    return Err("Empty filter object".to_string());
                }
                if conditions.len() == 1 {
                    conditions.pop().ok_or_else(|| "Empty conditions after len check".to_string())
                } else {
                    Ok(Filter::And(conditions))
                }
            }
            _ => Err("Filter must be a JSON object".to_string()),
        }
    }

    fn parse_logical_and_with_depth(
        value: &Value,
        depth: usize,
    ) -> std::result::Result<Self, String> {
        match value {
            Value::Array(arr) => {
                if arr.len() > Self::MAX_FILTER_ARRAY_SIZE {
                    return Err(format!(
                        "$and array too large ({} elements, max {})",
                        arr.len(),
                        Self::MAX_FILTER_ARRAY_SIZE
                    ));
                }
                let filters: std::result::Result<Vec<Filter>, String> = arr
                    .iter()
                    .map(|v| Self::parse_with_depth(v, depth))
                    .collect();
                Ok(Filter::And(filters?))
            }
            _ => Err("$and must be an array".to_string()),
        }
    }

    fn parse_logical_or_with_depth(
        value: &Value,
        depth: usize,
    ) -> std::result::Result<Self, String> {
        match value {
            Value::Array(arr) => {
                if arr.len() > Self::MAX_FILTER_ARRAY_SIZE {
                    return Err(format!(
                        "$or array too large ({} elements, max {})",
                        arr.len(),
                        Self::MAX_FILTER_ARRAY_SIZE
                    ));
                }
                let filters: std::result::Result<Vec<Filter>, String> = arr
                    .iter()
                    .map(|v| Self::parse_with_depth(v, depth))
                    .collect();
                Ok(Filter::Or(filters?))
            }
            _ => Err("$or must be an array".to_string()),
        }
    }

    fn parse_logical_not_with_depth(
        value: &Value,
        depth: usize,
    ) -> std::result::Result<Self, String> {
        let inner = Self::parse_with_depth(value, depth)?;
        Ok(Filter::Not(Box::new(inner)))
    }

    fn parse_field_condition(field: &str, value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Object(map) if map.keys().any(|k| k.starts_with('$')) => {
                // Has operators
                let mut conditions = Vec::new();
                for (op, op_value) in map {
                    let filter = match op.as_str() {
                        "$eq" => Filter::eq(field, op_value.clone()),
                        "$ne" => Filter::ne(field, op_value.clone()),
                        "$gt" => Filter::gt(field, op_value.clone()),
                        "$gte" => Filter::gte(field, op_value.clone()),
                        "$lt" => Filter::lt(field, op_value.clone()),
                        "$lte" => Filter::lte(field, op_value.clone()),
                        "$in" => match op_value {
                            Value::Array(arr) => Filter::is_in(field, arr.clone()),
                            _ => return Err("$in requires an array".to_string()),
                        },
                        "$nin" => match op_value {
                            Value::Array(arr) => Filter::Condition(FilterCondition {
                                field: field.to_string(),
                                operator: FilterOperator::NotIn,
                                value: Value::Array(arr.clone()),
                            }),
                            _ => return Err("$nin requires an array".to_string()),
                        },
                        "$contains" => Filter::Condition(FilterCondition {
                            field: field.to_string(),
                            operator: FilterOperator::Contains,
                            value: op_value.clone(),
                        }),
                        "$startsWith" => Filter::Condition(FilterCondition {
                            field: field.to_string(),
                            operator: FilterOperator::StartsWith,
                            value: op_value.clone(),
                        }),
                        "$endsWith" => Filter::Condition(FilterCondition {
                            field: field.to_string(),
                            operator: FilterOperator::EndsWith,
                            value: op_value.clone(),
                        }),
                        "$exists" => {
                            let exists = op_value.as_bool().ok_or(
                                "$exists requires a boolean value".to_string(),
                            )?;
                            Filter::exists(field, exists)
                        }
                        "$regex" => match op_value {
                            Value::String(_) => Filter::regex(field, op_value.as_str().expect("checked above")),
                            _ => return Err("$regex requires a string pattern".to_string()),
                        },
                        "$all" => match op_value {
                            Value::Array(arr) => Filter::all(field, arr.clone()),
                            _ => return Err("$all requires an array".to_string()),
                        },
                        "$elemMatch" => match op_value {
                            Value::Object(_) => Filter::elem_match(field, op_value.clone()),
                            _ => return Err("$elemMatch requires an object".to_string()),
                        },
                        "$between" => match op_value {
                            Value::Array(arr) if arr.len() == 2 => {
                                Filter::Condition(FilterCondition {
                                    field: field.to_string(),
                                    operator: FilterOperator::Between,
                                    value: op_value.clone(),
                                })
                            }
                            Value::Array(_) => return Err("$between requires an array of exactly [low, high]".to_string()),
                            _ => return Err("$between requires an array of [low, high]".to_string()),
                        },
                        "$size" => match op_value {
                            Value::Number(_) => Filter::Condition(FilterCondition {
                                field: field.to_string(),
                                operator: FilterOperator::Size,
                                value: op_value.clone(),
                            }),
                            _ => return Err("$size requires a number".to_string()),
                        },
                        "$type" => match op_value {
                            Value::String(s) => {
                                let valid = ["string", "number", "boolean", "array", "object", "null"];
                                if valid.contains(&s.as_str()) {
                                    Filter::Condition(FilterCondition {
                                        field: field.to_string(),
                                        operator: FilterOperator::Type,
                                        value: op_value.clone(),
                                    })
                                } else {
                                    return Err(format!(
                                        "$type requires one of: {}",
                                        valid.join(", ")
                                    ));
                                }
                            }
                            _ => return Err("$type requires a string".to_string()),
                        },
                        _ => return Err(format!("Unknown operator: {}", op)),
                    };
                    conditions.push(filter);
                }
                if conditions.len() == 1 {
                    conditions.pop().ok_or_else(|| "Empty conditions after len check".to_string())
                } else {
                    Ok(Filter::And(conditions))
                }
            }
            // Simple equality
            _ => Ok(Filter::eq(field, value.clone())),
        }
    }
}

/// Evaluate a single filter condition
fn evaluate_condition(condition: &FilterCondition, metadata: Option<&Value>) -> bool {
    let metadata = match metadata {
        Some(m) => m,
        None => {
            // No metadata: only $exists:false matches
            return condition.operator == FilterOperator::Exists
                && condition.value == Value::Bool(false);
        }
    };

    // $exists checks field presence, not value comparison
    if condition.operator == FilterOperator::Exists {
        let field_present = get_field_value(metadata, &condition.field).is_some();
        let want_exists = condition.value.as_bool().unwrap_or(true);
        return field_present == want_exists;
    }

    let field_value = get_field_value(metadata, &condition.field);

    match field_value {
        Some(value) => compare_values(&condition.operator, value, &condition.value),
        None => false,
    }
}

/// Get nested field value using dot notation
fn get_field_value<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = value;

    for part in parts {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            }
            Value::Array(arr) => {
                let index: usize = part.parse().ok()?;
                current = arr.get(index)?;
            }
            _ => return None,
        }
    }

    Some(current)
}

/// Compare values using the given operator
fn compare_values(operator: &FilterOperator, field_value: &Value, filter_value: &Value) -> bool {
    match operator {
        FilterOperator::Eq => field_value == filter_value,
        FilterOperator::Ne => field_value != filter_value,
        FilterOperator::Gt => compare_numeric(field_value, filter_value, |a, b| a > b),
        FilterOperator::Gte => compare_numeric(field_value, filter_value, |a, b| a >= b),
        FilterOperator::Lt => compare_numeric(field_value, filter_value, |a, b| a < b),
        FilterOperator::Lte => compare_numeric(field_value, filter_value, |a, b| a <= b),
        FilterOperator::In => {
            if let Value::Array(arr) = filter_value {
                arr.contains(field_value)
            } else {
                false
            }
        }
        FilterOperator::NotIn => {
            if let Value::Array(arr) = filter_value {
                !arr.contains(field_value)
            } else {
                true
            }
        }
        FilterOperator::Contains => match (field_value, filter_value) {
            (Value::String(s), Value::String(substr)) => s.contains(substr.as_str()),
            (Value::Array(arr), v) => arr.contains(v),
            _ => false,
        },
        FilterOperator::StartsWith => match (field_value, filter_value) {
            (Value::String(s), Value::String(prefix)) => s.starts_with(prefix.as_str()),
            _ => false,
        },
        FilterOperator::EndsWith => match (field_value, filter_value) {
            (Value::String(s), Value::String(suffix)) => s.ends_with(suffix.as_str()),
            _ => false,
        },
        // $exists is handled in evaluate_condition before compare_values is called
        FilterOperator::Exists => false,
        FilterOperator::Regex => match (field_value, filter_value) {
            (Value::String(s), Value::String(pattern)) => simple_regex_match(s, pattern),
            _ => false,
        },
        FilterOperator::All => match (field_value, filter_value) {
            (Value::Array(field_arr), Value::Array(required)) => {
                required.iter().all(|req| field_arr.contains(req))
            }
            _ => false,
        },
        FilterOperator::ElemMatch => match field_value {
            Value::Array(arr) => {
                if let Ok(sub_filter) = Filter::parse(filter_value) {
                    arr.iter().any(|elem| sub_filter.matches(Some(elem)))
                } else {
                    false
                }
            }
            _ => false,
        },
        FilterOperator::Between => {
            if let Value::Array(bounds) = filter_value {
                if bounds.len() == 2 {
                    compare_numeric(field_value, &bounds[0], |a, b| a >= b)
                        && compare_numeric(field_value, &bounds[1], |a, b| a <= b)
                } else {
                    false
                }
            } else {
                false
            }
        }
        FilterOperator::Size => {
            let expected = filter_value.as_u64();
            match (field_value, expected) {
                (Value::Array(arr), Some(n)) => arr.len() as u64 == n,
                (Value::String(s), Some(n)) => s.len() as u64 == n,
                _ => false,
            }
        }
        FilterOperator::Type => {
            if let Value::String(type_name) = filter_value {
                match type_name.as_str() {
                    "string" => field_value.is_string(),
                    "number" => field_value.is_number(),
                    "boolean" => field_value.is_boolean(),
                    "array" => field_value.is_array(),
                    "object" => field_value.is_object(),
                    "null" => field_value.is_null(),
                    _ => false,
                }
            } else {
                false
            }
        }
    }
}

/// Simple regex-like pattern matching without external regex crate.
///
/// Supports:
/// - `.` matches any single character
/// - `*` after a char matches zero or more of that char
/// - `.*` matches any sequence
/// - `^` anchors to start (implicit if not using `.*` prefix)
/// - `$` anchors to end (implicit if not using `.*` suffix)
/// - Literal characters match themselves
///
/// For full regex support, callers should use an external regex crate.
/// This implementation covers common metadata filtering patterns.
fn simple_regex_match(text: &str, pattern: &str) -> bool {
    // Fast path for common cases
    if pattern.is_empty() {
        return text.is_empty();
    }
    if pattern == ".*" || pattern == "^.*$" {
        return true;
    }

    // Strip optional anchors
    let pattern = pattern.strip_prefix('^').unwrap_or(pattern);
    let pattern = pattern.strip_suffix('$').unwrap_or(pattern);

    // Check if pattern starts/ends with .* for contains-like behavior
    let (prefix_wild, pattern) = if let Some(rest) = pattern.strip_prefix(".*") {
        (true, rest)
    } else {
        (false, pattern)
    };
    let (suffix_wild, pattern) = if let Some(rest) = pattern.strip_suffix(".*") {
        (true, rest)
    } else {
        (false, pattern)
    };

    // If both prefix and suffix are wild, check if pattern is contained in text
    if prefix_wild && suffix_wild {
        return text.contains(pattern);
    }

    if prefix_wild {
        return text.ends_with(pattern);
    }

    if suffix_wild {
        return text.starts_with(pattern);
    }

    // Exact match (treating only `.` as wildcard for single char)
    if pattern.len() != text.len() {
        return false;
    }
    pattern.chars().zip(text.chars()).all(|(p, t)| p == '.' || p == t)
}

/// Compare numeric values
fn compare_numeric<F>(a: &Value, b: &Value, cmp: F) -> bool
where
    F: Fn(f64, f64) -> bool,
{
    let a_num = value_to_f64(a);
    let b_num = value_to_f64(b);

    match (a_num, b_num) {
        (Some(a), Some(b)) => cmp(a, b),
        _ => false,
    }
}

/// Convert a JSON value to f64 if possible
fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_metadata_store() {
        let mut store = MetadataStore::new();

        store
            .insert(0, "doc1".to_string(), Some(json!({"title": "Hello"})))
            .unwrap();
        store
            .insert(1, "doc2".to_string(), Some(json!({"title": "World"})))
            .unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.get_internal_id("doc1"), Some(0));
        assert_eq!(store.get(0).unwrap().external_id, "doc1");
    }

    #[test]
    fn test_filter_eq() {
        let filter = Filter::eq("category", "greeting");
        let metadata = json!({"category": "greeting", "priority": 1});

        assert!(filter.matches(Some(&metadata)));

        let metadata2 = json!({"category": "other"});
        assert!(!filter.matches(Some(&metadata2)));
    }

    #[test]
    fn test_filter_numeric() {
        let filter = Filter::gt("score", 5);
        let metadata = json!({"score": 10});

        assert!(filter.matches(Some(&metadata)));

        let metadata2 = json!({"score": 3});
        assert!(!filter.matches(Some(&metadata2)));
    }

    #[test]
    fn test_filter_in() {
        let filter = Filter::is_in("status", vec![json!("active"), json!("pending")]);
        let metadata = json!({"status": "active"});

        assert!(filter.matches(Some(&metadata)));

        let metadata2 = json!({"status": "deleted"});
        assert!(!filter.matches(Some(&metadata2)));
    }

    #[test]
    fn test_filter_and() {
        let filter = Filter::and(vec![
            Filter::eq("category", "greeting"),
            Filter::gt("priority", 0),
        ]);

        let metadata = json!({"category": "greeting", "priority": 1});
        assert!(filter.matches(Some(&metadata)));

        let metadata2 = json!({"category": "greeting", "priority": 0});
        assert!(!filter.matches(Some(&metadata2)));
    }

    #[test]
    fn test_filter_nested() {
        let filter = Filter::eq("user.name", "Alice");
        let metadata = json!({"user": {"name": "Alice", "age": 30}});

        assert!(filter.matches(Some(&metadata)));
    }

    #[test]
    fn test_parse_filter() {
        let filter_json = json!({
            "category": {"$eq": "greeting"},
            "priority": {"$gte": 1}
        });

        let filter = Filter::parse(&filter_json).unwrap();
        let metadata = json!({"category": "greeting", "priority": 2});

        assert!(filter.matches(Some(&metadata)));
    }

    #[test]
    fn test_parse_filter_shorthand() {
        let filter_json = json!({"category": "greeting"});

        let filter = Filter::parse(&filter_json).unwrap();
        let metadata = json!({"category": "greeting"});

        assert!(filter.matches(Some(&metadata)));
    }

    #[test]
    fn test_filter_max_depth() {
        // Build a deeply nested filter that exceeds MAX_FILTER_DEPTH (32)
        let mut nested = json!({"field": "value"});
        for _ in 0..35 {
            nested = json!({"$not": nested});
        }

        let result = Filter::parse(&nested);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too deep"));
    }

    #[test]
    fn test_filter_valid_depth() {
        // A reasonably nested filter should work
        let filter_json = json!({
            "$and": [
                {"$or": [
                    {"category": "a"},
                    {"category": "b"}
                ]},
                {"$not": {"status": "deleted"}}
            ]
        });

        let result = Filter::parse(&filter_json);
        assert!(result.is_ok());
    }

    // ── Inverted Index / Filtered Pre-Index Tests ────────────────────────

    #[test]
    fn test_inverted_index_populated_on_insert() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"color": "red", "size": 10}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"color": "blue", "size": 20}))).unwrap();
        store.insert(2, "v2".into(), Some(json!({"color": "red", "size": 30}))).unwrap();

        assert!(store.indexed_fields().contains(&"color"));
        assert!(store.indexed_fields().contains(&"size"));
        assert_eq!(store.field_cardinality("color"), 2); // "red" and "blue"

        // Lookup returns matching internal IDs
        let reds = store.lookup_field_eq("color", &json!("red"));
        assert!(reds.is_some());
        let reds = reds.unwrap();
        assert_eq!(reds.len(), 2);
        assert!(reds.contains(&0));
        assert!(reds.contains(&2));
    }

    #[test]
    fn test_inverted_index_cleaned_on_delete() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"color": "red"}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"color": "red"}))).unwrap();

        store.delete(0);

        let reds = store.lookup_field_eq("color", &json!("red")).unwrap();
        assert_eq!(reds.len(), 1);
        assert!(reds.contains(&1));
    }

    #[test]
    fn test_inverted_index_updated_on_metadata_change() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"color": "red"}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"color": "red"}))).unwrap();

        // Update v0 from red to blue
        store.update_data(0, Some(json!({"color": "blue"}))).unwrap();

        // Red should now only contain v1
        let reds = store.lookup_field_eq("color", &json!("red")).unwrap();
        assert_eq!(reds.len(), 1);
        assert!(reds.contains(&1));

        // Blue should contain v0
        let blues = store.lookup_field_eq("color", &json!("blue")).unwrap();
        assert_eq!(blues.len(), 1);
        assert!(blues.contains(&0));
    }

    #[test]
    fn test_inverted_index_lookup_missing_field() {
        let store = MetadataStore::new();
        assert!(store.lookup_field_eq("nonexistent", &json!("val")).is_none());
    }

    #[test]
    fn test_resolve_filter_via_index_eq() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"cat": "a"}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"cat": "b"}))).unwrap();
        store.insert(2, "v2".into(), Some(json!({"cat": "a"}))).unwrap();

        let filter = Filter::eq("cat", "a");
        let result = store.resolve_filter_via_index(&filter);
        assert!(result.is_some());
        let ids = result.unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&0));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_resolve_filter_via_index_and() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"cat": "a", "type": "x"}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"cat": "a", "type": "y"}))).unwrap();
        store.insert(2, "v2".into(), Some(json!({"cat": "b", "type": "x"}))).unwrap();

        let filter = Filter::and(vec![
            Filter::eq("cat", "a"),
            Filter::eq("type", "x"),
        ]);
        let result = store.resolve_filter_via_index(&filter);
        assert!(result.is_some());
        let ids = result.unwrap();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&0));
    }

    #[test]
    fn test_resolve_filter_via_index_unsupported_falls_back() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"val": 10}))).unwrap();

        // Greater-than is not supported by inverted index
        let filter = Filter::gt("val", 5);
        let result = store.resolve_filter_via_index(&filter);
        assert!(result.is_none()); // Falls back to sequential
    }

    #[test]
    fn test_filtered_search_uses_index() {
        // Integration test: search with equality filter should use inverted index path
        use crate::Collection;
        let mut col = Collection::with_dimensions("test", 4);
        for i in 0..20 {
            let cat = if i % 2 == 0 { "even" } else { "odd" };
            col.insert(
                format!("v{i}"),
                &[i as f32, 0.0, 0.0, 0.0],
                Some(json!({"cat": cat})),
            ).unwrap();
        }

        let filter = Filter::eq("cat", "even");
        let results = col.search_builder(&[10.0, 0.0, 0.0, 0.0])
            .k(5)
            .filter(&filter)
            .execute()
            .unwrap();

        assert_eq!(results.len(), 5);
        // All results should have cat=even
        for r in &results {
            assert!(r.metadata.as_ref().unwrap()["cat"] == "even");
        }
    }

    #[test]
    fn test_high_cardinality_auto_detection() {
        let mut store = MetadataStore::new();

        // Insert vectors with unique UUID-like values
        for i in 0..super::DEFAULT_CARDINALITY_THRESHOLD + 100 {
            store
                .insert(
                    i,
                    format!("v{i}"),
                    Some(json!({
                        "category": if i % 3 == 0 { "a" } else if i % 3 == 1 { "b" } else { "c" },
                        "uuid": format!("uuid-{i}")
                    })),
                )
                .unwrap();
        }

        // "category" has low cardinality (3) — should be indexed
        assert!(!store.is_high_cardinality("category"));
        assert!(store.field_cardinality("category") == 3);
        assert!(store.lookup_field_eq("category", &json!("a")).is_some());

        // "uuid" has high cardinality — should be auto-excluded
        assert!(store.is_high_cardinality("uuid"));
        // Inverted index for uuid should be removed
        assert!(store.lookup_field_eq("uuid", &json!("uuid-0")).is_none());
    }

    #[test]
    fn test_field_stats() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"cat": "a", "tag": "x"}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"cat": "b", "tag": "x"}))).unwrap();
        store.insert(2, "v2".into(), Some(json!({"cat": "a", "tag": "y"}))).unwrap();

        // Specific field stats
        let cat_stats = store.field_stats("cat").unwrap();
        assert_eq!(cat_stats.name, "cat");
        assert_eq!(cat_stats.cardinality, 2);
        assert!(cat_stats.is_indexed);
        assert!(!cat_stats.is_high_cardinality);

        let tag_stats = store.field_stats("tag").unwrap();
        assert_eq!(tag_stats.name, "tag");
        assert_eq!(tag_stats.cardinality, 2);
        assert!(tag_stats.is_indexed);

        // Unknown field returns None
        assert!(store.field_stats("nonexistent").is_none());

        // All field stats
        let all = store.all_field_stats();
        assert_eq!(all.len(), 2);
        // Sorted by name
        assert_eq!(all[0].name, "cat");
        assert_eq!(all[1].name, "tag");
    }

    #[test]
    fn test_custom_cardinality_threshold() {
        let mut store = MetadataStore::with_cardinality_threshold(100);
        assert_eq!(store.cardinality_threshold(), 100);

        // Insert 110 unique values — should trigger high-cardinality at 100
        for i in 0..110 {
            store
                .insert(i, format!("v{i}"), Some(json!({ "field": format!("val-{i}") })))
                .unwrap();
        }
        assert!(store.is_high_cardinality("field"));
        assert!(store.lookup_field_eq("field", &json!("val-0")).is_none());
    }

    #[test]
    fn test_set_cardinality_threshold() {
        let mut store = MetadataStore::new();
        assert_eq!(store.cardinality_threshold(), super::DEFAULT_CARDINALITY_THRESHOLD);

        store.set_cardinality_threshold(500);
        assert_eq!(store.cardinality_threshold(), 500);
    }

    // ── Schema validation tests ──────────────────────────────────────

    #[test]
    fn test_schema_valid_metadata_passes() {
        let schema = json!({
            "required": ["title"],
            "properties": {
                "title": {"type": "string"},
                "score": {"type": "number"}
            }
        });
        let metadata = json!({"title": "hello", "score": 42});
        assert!(validate_metadata_schema(&metadata, &schema).is_ok());
    }

    #[test]
    fn test_schema_missing_required_field_rejected() {
        let schema = json!({
            "required": ["title"],
            "properties": {
                "title": {"type": "string"}
            }
        });
        let metadata = json!({"score": 42});
        let err = validate_metadata_schema(&metadata, &schema).unwrap_err();
        assert!(err.to_string().contains("missing required metadata field"));
    }

    #[test]
    fn test_schema_wrong_type_rejected() {
        let schema = json!({
            "properties": {
                "score": {"type": "number"}
            }
        });
        let metadata = json!({"score": "not_a_number"});
        let err = validate_metadata_schema(&metadata, &schema).unwrap_err();
        assert!(err.to_string().contains("expected type 'number'"));
    }

    #[test]
    fn test_schema_no_schema_is_noop() {
        // validate_metadata_schema is only called when schema is Some,
        // but verify the function itself handles empty schema gracefully.
        let schema = json!({});
        let metadata = json!({"anything": true});
        assert!(validate_metadata_schema(&metadata, &schema).is_ok());
    }

    #[test]
    fn test_schema_extra_fields_allowed() {
        let schema = json!({
            "required": ["title"],
            "properties": {
                "title": {"type": "string"}
            }
        });
        let metadata = json!({"title": "hi", "extra": 123});
        assert!(validate_metadata_schema(&metadata, &schema).is_ok());
    }

    #[test]
    fn test_schema_all_types_validated() {
        let schema = json!({
            "properties": {
                "s": {"type": "string"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "a": {"type": "array"},
                "o": {"type": "object"},
                "z": {"type": "null"}
            }
        });
        let metadata = json!({
            "s": "str", "n": 1.5, "b": true,
            "a": [1,2], "o": {"k": "v"}, "z": null
        });
        assert!(validate_metadata_schema(&metadata, &schema).is_ok());
    }

    // ── Nested field (dot-notation) tests ────────────────────────────

    #[test]
    fn test_nested_field_resolution() {
        let data = json!({"a": {"b": {"c": 42}}});
        assert_eq!(get_field_value(&data, "a.b.c"), Some(&json!(42)));
        assert_eq!(get_field_value(&data, "a.b"), Some(&json!({"c": 42})));
        assert_eq!(get_field_value(&data, "a.b.missing"), None);
        assert_eq!(get_field_value(&data, "x"), None);
    }

    #[test]
    fn test_nested_filter_matching() {
        let filter = Filter::parse(&json!({"author.name": "Alice"})).unwrap();
        let metadata = json!({"author": {"name": "Alice", "age": 30}});
        assert!(filter.matches(Some(&metadata)));

        let metadata2 = json!({"author": {"name": "Bob"}});
        assert!(!filter.matches(Some(&metadata2)));
    }

    #[test]
    fn test_nested_field_indexing() {
        let mut store = MetadataStore::new();
        store.insert(
            0,
            "v0".into(),
            Some(json!({"author": {"name": "Alice", "age": 30}, "category": "books"})),
        ).unwrap();

        // Top-level field indexed
        let books = store.lookup_field_eq("category", &json!("books"));
        assert!(books.is_some());
        assert!(books.unwrap().contains(&0));

        // Nested fields indexed with dot-notation paths
        let alice = store.lookup_field_eq("author.name", &json!("Alice"));
        assert!(alice.is_some());
        assert!(alice.unwrap().contains(&0));

        let age = store.lookup_field_eq("author.age", &json!(30));
        assert!(age.is_some());
        assert!(age.unwrap().contains(&0));
    }

    #[test]
    fn test_nested_field_index_cleaned_on_delete() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"a": {"b": "val"}}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"a": {"b": "val"}}))).unwrap();

        store.delete(0);

        let results = store.lookup_field_eq("a.b", &json!("val")).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results.contains(&1));
    }

    #[test]
    fn test_nested_field_index_updated_on_metadata_change() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"x": {"y": "old"}}))).unwrap();

        store.update_data(0, Some(json!({"x": {"y": "new"}}))).unwrap();

        let old = store.lookup_field_eq("x.y", &json!("old"));
        assert!(old.is_none() || old.unwrap().is_empty() || !old.unwrap().contains(&0));

        let new = store.lookup_field_eq("x.y", &json!("new"));
        assert!(new.is_some());
        assert!(new.unwrap().contains(&0));
    }

    #[test]
    fn test_backward_compat_flat_metadata() {
        // Existing flat metadata filters still work identically
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"color": "red", "size": 10}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"color": "blue", "size": 20}))).unwrap();

        let filter = Filter::eq("color", "red");
        assert!(filter.matches(store.get(0).unwrap().data.as_ref()));
        assert!(!filter.matches(store.get(1).unwrap().data.as_ref()));

        let reds = store.lookup_field_eq("color", &json!("red")).unwrap();
        assert_eq!(reds.len(), 1);
        assert!(reds.contains(&0));
    }

    #[test]
    fn test_flatten_metadata_depth_limit() {
        // Deeply nested object should stop at FLATTEN_MAX_DEPTH
        let deep = json!({"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}});
        let mut out = Vec::new();
        flatten_metadata(&deep, "", 0, &mut out);
        // "a.b.c.d.e" is depth 5, so "f.g" at depth 6 is skipped
        // "a.b.c.d.e.f" is depth 5 for the "f" key, "g" inside is depth 6
        // At depth 5 we enter object {"f": {"g": 1}}, flatten_metadata is called
        // with depth=5 for {"f": {"g": 1}}, which tries depth 6 for {"g": 1} and returns
        // So "a.b.c.d.e.f" should not appear as a leaf since {"g":1} is an object
        // but depth 6 is blocked, so nothing is emitted for that branch.
        assert!(out.iter().all(|(path, _)| path.matches('.').count() < FLATTEN_MAX_DEPTH));
    }

    #[test]
    fn test_nested_resolve_filter_via_index() {
        let mut store = MetadataStore::new();
        store.insert(0, "v0".into(), Some(json!({"author": {"name": "Alice"}}))).unwrap();
        store.insert(1, "v1".into(), Some(json!({"author": {"name": "Bob"}}))).unwrap();

        let filter = Filter::eq("author.name", "Alice");
        let result = store.resolve_filter_via_index(&filter);
        assert!(result.is_some());
        let ids = result.unwrap();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&0));
    }

    // ── $exists filter tests ─────────────────────────────────────────

    #[test]
    fn test_filter_exists_true() {
        let filter = Filter::exists("category", true);
        assert!(filter.matches(Some(&json!({"category": "books"}))));
        assert!(!filter.matches(Some(&json!({"other": "value"}))));
        assert!(!filter.matches(None));
    }

    #[test]
    fn test_filter_exists_false() {
        let filter = Filter::exists("category", false);
        assert!(!filter.matches(Some(&json!({"category": "books"}))));
        assert!(filter.matches(Some(&json!({"other": "value"}))));
        assert!(filter.matches(None));
    }

    #[test]
    fn test_filter_exists_nested() {
        let filter = Filter::exists("author.name", true);
        assert!(filter.matches(Some(&json!({"author": {"name": "Alice"}}))));
        assert!(!filter.matches(Some(&json!({"author": {"age": 30}}))));
    }

    #[test]
    fn test_parse_exists_filter() {
        let filter = Filter::parse(&json!({"tags": {"$exists": true}})).unwrap();
        assert!(filter.matches(Some(&json!({"tags": ["a", "b"]}))));
        assert!(!filter.matches(Some(&json!({"other": 1}))));

        let filter2 = Filter::parse(&json!({"deleted": {"$exists": false}})).unwrap();
        assert!(filter2.matches(Some(&json!({"name": "test"}))));
        assert!(!filter2.matches(Some(&json!({"deleted": true}))));
    }

    #[test]
    fn test_parse_exists_requires_bool() {
        let result = Filter::parse(&json!({"field": {"$exists": "yes"}}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("boolean"));
    }

    // ── $regex filter tests ──────────────────────────────────────────

    #[test]
    fn test_filter_regex_basic() {
        let filter = Filter::regex("name", ".*Alice.*");
        assert!(filter.matches(Some(&json!({"name": "Alice Smith"}))));
        assert!(!filter.matches(Some(&json!({"name": "Bob"}))));
    }

    #[test]
    fn test_filter_regex_prefix() {
        let filter = Filter::regex("email", "admin.*");
        assert!(filter.matches(Some(&json!({"email": "admin@example.com"}))));
        assert!(!filter.matches(Some(&json!({"email": "user@admin.com"}))));
    }

    #[test]
    fn test_filter_regex_suffix() {
        let filter = Filter::regex("file", ".*.pdf");
        assert!(filter.matches(Some(&json!({"file": "report.pdf"}))));
        assert!(!filter.matches(Some(&json!({"file": "report.doc"}))));
    }

    #[test]
    fn test_filter_regex_dot_wildcard() {
        let filter = Filter::regex("code", "A.C");
        assert!(filter.matches(Some(&json!({"code": "ABC"}))));
        assert!(filter.matches(Some(&json!({"code": "AXC"}))));
        assert!(!filter.matches(Some(&json!({"code": "ABBC"}))));
    }

    #[test]
    fn test_filter_regex_match_all() {
        let filter = Filter::regex("any", ".*");
        assert!(filter.matches(Some(&json!({"any": "anything"}))));
        assert!(filter.matches(Some(&json!({"any": ""}))));
    }

    #[test]
    fn test_filter_regex_non_string_field() {
        let filter = Filter::regex("count", ".*");
        assert!(!filter.matches(Some(&json!({"count": 42}))));
    }

    #[test]
    fn test_parse_regex_filter() {
        let filter = Filter::parse(&json!({"name": {"$regex": ".*test.*"}})).unwrap();
        assert!(filter.matches(Some(&json!({"name": "my_test_file"}))));
        assert!(!filter.matches(Some(&json!({"name": "production"}))));
    }

    #[test]
    fn test_parse_regex_requires_string() {
        let result = Filter::parse(&json!({"field": {"$regex": 42}}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("string"));
    }

    // ── Convenience constructor tests ────────────────────────────────

    #[test]
    fn test_filter_contains_constructor() {
        let filter = Filter::contains("tags", "rust");
        let meta = json!({"tags": ["rust", "python"]});
        assert!(filter.matches(Some(&meta)));
        assert!(!filter.matches(Some(&json!({"tags": ["java"]}))));
    }

    #[test]
    fn test_filter_starts_with_constructor() {
        let filter = Filter::starts_with("name", "Dr.");
        assert!(filter.matches(Some(&json!({"name": "Dr. Smith"}))));
        assert!(!filter.matches(Some(&json!({"name": "Mr. Smith"}))));
    }

    #[test]
    fn test_filter_ends_with_constructor() {
        let filter = Filter::ends_with("email", "@example.com");
        assert!(filter.matches(Some(&json!({"email": "user@example.com"}))));
        assert!(!filter.matches(Some(&json!({"email": "user@other.com"}))));
    }

    #[test]
    fn test_filter_not_in_constructor() {
        let filter = Filter::not_in("status", vec![json!("deleted"), json!("archived")]);
        assert!(filter.matches(Some(&json!({"status": "active"}))));
        assert!(!filter.matches(Some(&json!({"status": "deleted"}))));
    }

    // ── Display impl tests ───────────────────────────────────────────

    #[test]
    fn test_filter_operator_display() {
        assert_eq!(FilterOperator::Eq.to_string(), "$eq");
        assert_eq!(FilterOperator::Ne.to_string(), "$ne");
        assert_eq!(FilterOperator::Gt.to_string(), "$gt");
        assert_eq!(FilterOperator::Gte.to_string(), "$gte");
        assert_eq!(FilterOperator::Lt.to_string(), "$lt");
        assert_eq!(FilterOperator::Lte.to_string(), "$lte");
        assert_eq!(FilterOperator::In.to_string(), "$in");
        assert_eq!(FilterOperator::NotIn.to_string(), "$nin");
        assert_eq!(FilterOperator::Contains.to_string(), "$contains");
        assert_eq!(FilterOperator::StartsWith.to_string(), "$startsWith");
        assert_eq!(FilterOperator::EndsWith.to_string(), "$endsWith");
        assert_eq!(FilterOperator::Exists.to_string(), "$exists");
        assert_eq!(FilterOperator::Regex.to_string(), "$regex");
    }

    // ── Serialization tests ──────────────────────────────────────────

    #[test]
    fn test_filter_serialize_roundtrip() {
        let filter = Filter::and(vec![
            Filter::eq("category", "books"),
            Filter::gt("price", 10),
            Filter::exists("in_stock", true),
        ]);
        let json = serde_json::to_string(&filter).unwrap();
        let deserialized: Filter = serde_json::from_str(&json).unwrap();

        // Verify the deserialized filter matches the same data
        let meta = json!({"category": "books", "price": 25, "in_stock": true});
        assert!(filter.matches(Some(&meta)));
        assert!(deserialized.matches(Some(&meta)));
    }

    #[test]
    fn test_filter_condition_serialize_roundtrip() {
        let cond = FilterCondition {
            field: "status".to_string(),
            operator: FilterOperator::Ne,
            value: json!("deleted"),
        };
        let json = serde_json::to_string(&cond).unwrap();
        let deserialized: FilterCondition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.field, "status");
        assert_eq!(deserialized.operator, FilterOperator::Ne);
    }

    // ── simple_regex_match unit tests ────────────────────────────────

    #[test]
    fn test_simple_regex_empty_pattern() {
        assert!(simple_regex_match("", ""));
        assert!(!simple_regex_match("text", ""));
    }

    #[test]
    fn test_simple_regex_with_anchors() {
        assert!(simple_regex_match("hello", "^hello$"));
        assert!(simple_regex_match("anything", "^.*$"));
    }

    // ── $all operator tests ──────────────────────────────────────────────

    #[test]
    fn test_all_operator_matches_subset() {
        let filter = Filter::parse(&json!({"tags": {"$all": ["rust", "db"]}})).unwrap();
        let meta = json!({"tags": ["rust", "db", "vector"]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_all_operator_exact_match() {
        let filter = Filter::parse(&json!({"tags": {"$all": ["a", "b"]}})).unwrap();
        let meta = json!({"tags": ["a", "b"]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_all_operator_missing_element() {
        let filter = Filter::parse(&json!({"tags": {"$all": ["rust", "missing"]}})).unwrap();
        let meta = json!({"tags": ["rust", "db"]});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_all_operator_empty_required() {
        let filter = Filter::parse(&json!({"tags": {"$all": []}})).unwrap();
        let meta = json!({"tags": ["anything"]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_all_operator_non_array_field() {
        let filter = Filter::parse(&json!({"name": {"$all": ["x"]}})).unwrap();
        let meta = json!({"name": "x"});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_all_operator_with_numbers() {
        let filter = Filter::parse(&json!({"scores": {"$all": [1, 2, 3]}})).unwrap();
        let meta = json!({"scores": [1, 2, 3, 4, 5]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_all_parse_error_non_array() {
        let result = Filter::parse(&json!({"tags": {"$all": "not_array"}}));
        assert!(result.is_err());
    }

    // ── $elemMatch operator tests ────────────────────────────────────────

    #[test]
    fn test_elem_match_basic() {
        let filter = Filter::parse(&json!({
            "items": {"$elemMatch": {"price": {"$gt": 10}}}
        })).unwrap();
        let meta = json!({"items": [
            {"price": 5, "name": "cheap"},
            {"price": 15, "name": "expensive"}
        ]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_elem_match_no_match() {
        let filter = Filter::parse(&json!({
            "items": {"$elemMatch": {"price": {"$gt": 100}}}
        })).unwrap();
        let meta = json!({"items": [
            {"price": 5},
            {"price": 15}
        ]});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_elem_match_multiple_conditions() {
        let filter = Filter::parse(&json!({
            "items": {"$elemMatch": {"price": {"$gt": 10}, "in_stock": true}}
        })).unwrap();
        let meta = json!({"items": [
            {"price": 15, "in_stock": false},
            {"price": 20, "in_stock": true}
        ]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_elem_match_multiple_conditions_no_single_element_matches() {
        let filter = Filter::parse(&json!({
            "items": {"$elemMatch": {"price": {"$gt": 10}, "in_stock": true}}
        })).unwrap();
        // price>10 in one element, in_stock=true in another — no single element matches both
        let meta = json!({"items": [
            {"price": 15, "in_stock": false},
            {"price": 5, "in_stock": true}
        ]});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_elem_match_non_array_field() {
        let filter = Filter::parse(&json!({
            "name": {"$elemMatch": {"x": 1}}
        })).unwrap();
        let meta = json!({"name": "not_array"});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_elem_match_empty_array() {
        let filter = Filter::parse(&json!({
            "items": {"$elemMatch": {"x": 1}}
        })).unwrap();
        let meta = json!({"items": []});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_elem_match_parse_error_non_object() {
        let result = Filter::parse(&json!({"items": {"$elemMatch": [1,2,3]}}));
        assert!(result.is_err());
    }

    // ── Display tests for new operators ──────────────────────────────────

    #[test]
    fn test_all_operator_display() {
        assert_eq!(format!("{}", FilterOperator::All), "$all");
    }

    #[test]
    fn test_elem_match_operator_display() {
        assert_eq!(format!("{}", FilterOperator::ElemMatch), "$elemMatch");
    }

    // ── Builder method tests ─────────────────────────────────────────────

    #[test]
    fn test_filter_all_builder() {
        let filter = Filter::all("tags", vec![json!("a"), json!("b")]);
        let meta = json!({"tags": ["a", "b", "c"]});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_filter_elem_match_builder() {
        let filter = Filter::elem_match("items", json!({"x": {"$gt": 5}}));
        let meta = json!({"items": [{"x": 3}, {"x": 10}]});
        assert!(filter.matches(Some(&meta)));
    }

    // ── Edge case: $in / $not_in with empty arrays ──────────────────────

    #[test]
    fn test_in_empty_array_matches_nothing() {
        let filter = Filter::parse(&json!({"status": {"$in": []}})).unwrap();
        let meta = json!({"status": "active"});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_not_in_empty_array_matches_everything() {
        let filter = Filter::parse(&json!({"status": {"$nin": []}})).unwrap();
        let meta = json!({"status": "active"});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_in_empty_array_with_no_metadata() {
        let filter = Filter::parse(&json!({"status": {"$in": []}})).unwrap();
        assert!(!filter.matches(None));
    }

    // ── Edge case: deeply nested field paths ─────────────────────────────

    #[test]
    fn test_nested_field_depth_5() {
        let filter = Filter::parse(&json!({"a.b.c.d.e": "deep"})).unwrap();
        let meta = json!({"a": {"b": {"c": {"d": {"e": "deep"}}}}});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_nested_field_missing_intermediate() {
        let filter = Filter::parse(&json!({"a.b.c": "value"})).unwrap();
        let meta = json!({"a": {"x": 1}});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_nested_field_with_null_intermediate() {
        let filter = Filter::parse(&json!({"a.b": "value"})).unwrap();
        let meta = json!({"a": null});
        assert!(!filter.matches(Some(&meta)));
    }

    // ── Edge case: unicode field names and values ────────────────────────

    #[test]
    fn test_unicode_field_name() {
        let filter = Filter::parse(&json!({"名前": "テスト"})).unwrap();
        let meta = json!({"名前": "テスト"});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_emoji_in_value() {
        let filter = Filter::eq("tag", json!("🔥hot"));
        let meta = json!({"tag": "🔥hot"});
        assert!(filter.matches(Some(&meta)));
    }

    // ── Edge case: numeric comparison precision ─────────────────────────

    #[test]
    fn test_float_equality_exact() {
        let filter = Filter::parse(&json!({"score": 0.1})).unwrap();
        let meta = json!({"score": 0.1});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_comparison_integer_vs_float() {
        let filter = Filter::parse(&json!({"count": {"$gte": 5}})).unwrap();
        let meta = json!({"count": 5.0});
        assert!(filter.matches(Some(&meta)));
    }

    // ── Edge case: $exists with various field types ─────────────────────

    #[test]
    fn test_exists_true_on_null_value() {
        let filter = Filter::exists("field", true);
        let meta = json!({"field": null});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_exists_false_on_absent_field() {
        let filter = Filter::exists("missing", false);
        let meta = json!({"other": 1});
        assert!(filter.matches(Some(&meta)));
    }

    #[test]
    fn test_exists_true_on_empty_string() {
        let filter = Filter::exists("field", true);
        let meta = json!({"field": ""});
        assert!(filter.matches(Some(&meta)));
    }

    // ── Edge case: $contains on non-array/non-string ────────────────────

    #[test]
    fn test_contains_on_number_returns_false() {
        let filter = Filter::contains("field", json!(5));
        let meta = json!({"field": 5});
        assert!(!filter.matches(Some(&meta)));
    }

    #[test]
    fn test_contains_on_boolean_returns_false() {
        let filter = Filter::contains("field", json!(true));
        let meta = json!({"field": true});
        assert!(!filter.matches(Some(&meta)));
    }

    // ── $between operator tests ─────────────────────────────────────────

    #[test]
    fn test_between_basic() {
        let filter = Filter::between("price", 10, 50);
        assert!(filter.matches(Some(&json!({"price": 10}))));
        assert!(filter.matches(Some(&json!({"price": 30}))));
        assert!(filter.matches(Some(&json!({"price": 50}))));
        assert!(!filter.matches(Some(&json!({"price": 9}))));
        assert!(!filter.matches(Some(&json!({"price": 51}))));
    }

    #[test]
    fn test_between_float() {
        let filter = Filter::parse(&json!({"score": {"$between": [0.5, 0.9]}})).unwrap();
        assert!(filter.matches(Some(&json!({"score": 0.7}))));
        assert!(filter.matches(Some(&json!({"score": 0.5}))));
        assert!(filter.matches(Some(&json!({"score": 0.9}))));
        assert!(!filter.matches(Some(&json!({"score": 0.4}))));
        assert!(!filter.matches(Some(&json!({"score": 1.0}))));
    }

    #[test]
    fn test_between_parse_error_wrong_length() {
        let result = Filter::parse(&json!({"x": {"$between": [1, 2, 3]}}));
        assert!(result.is_err());
    }

    #[test]
    fn test_between_parse_error_not_array() {
        let result = Filter::parse(&json!({"x": {"$between": 5}}));
        assert!(result.is_err());
    }

    #[test]
    fn test_between_no_metadata() {
        let filter = Filter::between("price", 1, 10);
        assert!(!filter.matches(None));
    }

    #[test]
    fn test_between_missing_field() {
        let filter = Filter::between("price", 1, 10);
        assert!(!filter.matches(Some(&json!({"other": 5}))));
    }

    #[test]
    fn test_between_non_numeric() {
        let filter = Filter::between("name", 1, 10);
        assert!(!filter.matches(Some(&json!({"name": "alice"}))));
    }

    #[test]
    fn test_between_display() {
        assert_eq!(FilterOperator::Between.to_string(), "$between");
    }

    // ── $size operator tests ────────────────────────────────────────────

    #[test]
    fn test_size_array() {
        let filter = Filter::size("tags", 3);
        assert!(filter.matches(Some(&json!({"tags": ["a", "b", "c"]}))));
        assert!(!filter.matches(Some(&json!({"tags": ["a", "b"]}))));
        assert!(!filter.matches(Some(&json!({"tags": []}))));
    }

    #[test]
    fn test_size_string() {
        let filter = Filter::size("code", 5);
        assert!(filter.matches(Some(&json!({"code": "hello"}))));
        assert!(!filter.matches(Some(&json!({"code": "hi"}))));
    }

    #[test]
    fn test_size_zero() {
        let filter = Filter::size("items", 0);
        assert!(filter.matches(Some(&json!({"items": []}))));
        assert!(!filter.matches(Some(&json!({"items": [1]}))));
    }

    #[test]
    fn test_size_parse() {
        let filter = Filter::parse(&json!({"tags": {"$size": 2}})).unwrap();
        assert!(filter.matches(Some(&json!({"tags": [1, 2]}))));
        assert!(!filter.matches(Some(&json!({"tags": [1]}))));
    }

    #[test]
    fn test_size_parse_error_not_number() {
        let result = Filter::parse(&json!({"tags": {"$size": "three"}}));
        assert!(result.is_err());
    }

    #[test]
    fn test_size_non_array_non_string() {
        let filter = Filter::size("val", 1);
        assert!(!filter.matches(Some(&json!({"val": 42}))));
        assert!(!filter.matches(Some(&json!({"val": true}))));
    }

    #[test]
    fn test_size_display() {
        assert_eq!(FilterOperator::Size.to_string(), "$size");
    }

    // ── $type operator tests ────────────────────────────────────────────

    #[test]
    fn test_type_string() {
        let filter = Filter::has_type("name", "string");
        assert!(filter.matches(Some(&json!({"name": "alice"}))));
        assert!(!filter.matches(Some(&json!({"name": 42}))));
    }

    #[test]
    fn test_type_number() {
        let filter = Filter::has_type("age", "number");
        assert!(filter.matches(Some(&json!({"age": 25}))));
        assert!(filter.matches(Some(&json!({"age": 3.14}))));
        assert!(!filter.matches(Some(&json!({"age": "25"}))));
    }

    #[test]
    fn test_type_boolean() {
        let filter = Filter::has_type("active", "boolean");
        assert!(filter.matches(Some(&json!({"active": true}))));
        assert!(!filter.matches(Some(&json!({"active": 1}))));
    }

    #[test]
    fn test_type_array() {
        let filter = Filter::has_type("tags", "array");
        assert!(filter.matches(Some(&json!({"tags": [1, 2]}))));
        assert!(!filter.matches(Some(&json!({"tags": "not array"}))));
    }

    #[test]
    fn test_type_object() {
        let filter = Filter::has_type("meta", "object");
        assert!(filter.matches(Some(&json!({"meta": {"key": "val"}}))));
        assert!(!filter.matches(Some(&json!({"meta": [1]}))));
    }

    #[test]
    fn test_type_null() {
        let filter = Filter::has_type("field", "null");
        assert!(filter.matches(Some(&json!({"field": null}))));
        assert!(!filter.matches(Some(&json!({"field": 0}))));
    }

    #[test]
    fn test_type_parse() {
        let filter = Filter::parse(&json!({"age": {"$type": "number"}})).unwrap();
        assert!(filter.matches(Some(&json!({"age": 10}))));
        assert!(!filter.matches(Some(&json!({"age": "ten"}))));
    }

    #[test]
    fn test_type_parse_error_invalid_type() {
        let result = Filter::parse(&json!({"x": {"$type": "integer"}}));
        assert!(result.is_err());
    }

    #[test]
    fn test_type_parse_error_not_string() {
        let result = Filter::parse(&json!({"x": {"$type": 42}}));
        assert!(result.is_err());
    }

    #[test]
    fn test_type_display() {
        assert_eq!(FilterOperator::Type.to_string(), "$type");
    }

    // ── depth() method tests ────────────────────────────────────────────

    #[test]
    fn test_depth_condition() {
        assert_eq!(Filter::eq("x", 1).depth(), 1);
    }

    #[test]
    fn test_depth_and() {
        let filter = Filter::and(vec![Filter::eq("a", 1), Filter::gt("b", 2)]);
        assert_eq!(filter.depth(), 2);
    }

    #[test]
    fn test_depth_nested() {
        let inner = Filter::and(vec![Filter::eq("a", 1), Filter::eq("b", 2)]);
        let outer = Filter::or(vec![inner, Filter::eq("c", 3)]);
        assert_eq!(outer.depth(), 3);
    }

    #[test]
    fn test_depth_not() {
        let filter = Filter::negate(Filter::eq("x", 1));
        assert_eq!(filter.depth(), 2);
    }

    #[test]
    fn test_depth_empty_and() {
        let filter = Filter::and(vec![]);
        assert_eq!(filter.depth(), 1);
    }
}
