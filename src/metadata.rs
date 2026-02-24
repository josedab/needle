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

#![cfg_attr(test, allow(clippy::unwrap_used))]
use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Metadata entry for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntry {
    /// External (user-provided) ID
    pub external_id: String,
    /// JSON metadata
    pub data: Option<Value>,
}

/// Maximum distinct values per field before the inverted index stops adding
/// new values for that field. Prevents high-cardinality fields (like UUIDs)
/// from bloating the index with no filtering benefit.
const FIELD_INDEX_CARDINALITY_THRESHOLD: usize = 10_000;

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
    /// Fields exceeding FIELD_INDEX_CARDINALITY_THRESHOLD are excluded.
    #[serde(skip)]
    field_indexes: HashMap<String, HashMap<String, Vec<usize>>>,
    /// Fields that have been disabled from indexing due to high cardinality.
    #[serde(skip)]
    high_cardinality_fields: std::collections::HashSet<String>,
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

        // Update bloom filters and inverted indexes with field values
        if let Some(Value::Object(ref map)) = data {
            for (field, value) in map {
                let bloom = self
                    .field_blooms
                    .entry(field.clone())
                    .or_insert_with(BloomFilter::default);
                bloom.insert(&value.to_string());

                // Skip inverted index for high-cardinality fields
                if self.high_cardinality_fields.contains(field) {
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
                if field_idx.len() > FIELD_INDEX_CARDINALITY_THRESHOLD {
                    self.high_cardinality_fields.insert(field.clone());
                    self.field_indexes.remove(field);
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

            // Remove from inverted indexes
            if let Some(Value::Object(ref map)) = entry.data {
                for (field, value) in map {
                    let val_str = value.to_string();
                    if let Some(idx) = self.field_indexes.get_mut(field) {
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

        // Remove old values from inverted index
        if let Some(Value::Object(ref old_map)) = entry.data {
            for (field, value) in old_map {
                let val_str = value.to_string();
                if let Some(idx) = self.field_indexes.get_mut(field) {
                    if let Some(ids) = idx.get_mut(&val_str) {
                        ids.retain(|&id| id != internal_id);
                    }
                }
            }
        }

        // Add new values to inverted index
        if let Some(Value::Object(ref new_map)) = data {
            for (field, value) in new_map {
                let val_str = value.to_string();
                self.field_indexes
                    .entry(field.clone())
                    .or_default()
                    .entry(val_str)
                    .or_default()
                    .push(internal_id);
            }
        }

        entry.data = data;
        Ok(())
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
}

/// Filter comparison operators
#[derive(Debug, Clone, PartialEq)]
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
}

/// A single filter condition
#[derive(Debug, Clone)]
pub struct FilterCondition {
    /// Field path (dot-separated for nested fields)
    pub field: String,
    /// Comparison operator
    pub operator: FilterOperator,
    /// Value to compare against
    pub value: Value,
}

/// Filter expression combining multiple conditions
#[derive(Debug, Clone)]
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

    /// Evaluate filter against metadata
    pub fn matches(&self, metadata: Option<&Value>) -> bool {
        match self {
            Filter::Condition(cond) => evaluate_condition(cond, metadata),
            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),
            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),
            Filter::Not(filter) => !filter.matches(metadata),
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
        None => return false,
    };

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
    }
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
        for i in 0..super::FIELD_INDEX_CARDINALITY_THRESHOLD + 100 {
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
}
