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

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Metadata entry for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataEntry {
    /// External (user-provided) ID
    pub external_id: String,
    /// JSON metadata
    pub data: Option<Value>,
}

/// Metadata store for vectors
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataStore {
    /// Mapping from internal ID to metadata
    entries: HashMap<usize, MetadataEntry>,
    /// Mapping from external ID to internal ID
    id_map: HashMap<String, usize>,
}

impl MetadataStore {
    /// Create a new metadata store
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            id_map: HashMap::new(),
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
            Some(entry)
        } else {
            None
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

    /// Update the metadata data for an entry
    pub fn update_data(&mut self, internal_id: usize, data: Option<Value>) -> Result<()> {
        let entry = self.entries.get_mut(&internal_id)
            .ok_or_else(|| NeedleError::VectorNotFound(internal_id.to_string()))?;
        entry.data = data;
        Ok(())
    }

    /// Estimate memory usage in bytes
    pub fn estimated_memory(&self) -> usize {
        // HashMap overhead + entries
        let base_overhead = std::mem::size_of::<Self>();
        let entry_overhead = self.entries.len() * (
            std::mem::size_of::<usize>() +  // Key
            std::mem::size_of::<MetadataEntry>()  // Value struct
        );
        let id_map_overhead = self.id_map.len() * (
            32 +  // Average string size estimate
            std::mem::size_of::<usize>()  // Value
        );
        // Estimate JSON data size
        let json_size: usize = self.entries.values()
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

    /// Parse a filter from a JSON value (MongoDB-like syntax)
    ///
    /// Supports:
    /// - Simple equality: `{"field": "value"}`
    /// - Comparison operators: `{"field": {"$gt": 5}}`
    /// - Logical operators: `{"$and": [...]}`, `{"$or": [...]}`, `{"$not": {...}}`
    pub fn parse(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Object(map) => {
                // Check for logical operators
                if let Some(and_value) = map.get("$and") {
                    return Self::parse_logical_and(and_value);
                }
                if let Some(or_value) = map.get("$or") {
                    return Self::parse_logical_or(or_value);
                }
                if let Some(not_value) = map.get("$not") {
                    return Self::parse_logical_not(not_value);
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
                    // Safe: we just checked conditions.len() == 1
                    Ok(conditions.pop().expect("checked len == 1"))
                } else {
                    Ok(Filter::And(conditions))
                }
            }
            _ => Err("Filter must be a JSON object".to_string()),
        }
    }

    fn parse_logical_and(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Array(arr) => {
                let filters: std::result::Result<Vec<Filter>, String> = arr.iter().map(Self::parse).collect();
                Ok(Filter::And(filters?))
            }
            _ => Err("$and must be an array".to_string()),
        }
    }

    fn parse_logical_or(value: &Value) -> std::result::Result<Self, String> {
        match value {
            Value::Array(arr) => {
                let filters: std::result::Result<Vec<Filter>, String> = arr.iter().map(Self::parse).collect();
                Ok(Filter::Or(filters?))
            }
            _ => Err("$or must be an array".to_string()),
        }
    }

    fn parse_logical_not(value: &Value) -> std::result::Result<Self, String> {
        let inner = Self::parse(value)?;
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
                    // Safe: we just checked conditions.len() == 1
                    Ok(conditions.pop().expect("checked len == 1"))
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

/// Parse a filter from JSON-like syntax
/// Example: {"category": {"$eq": "greeting"}}
pub fn parse_filter(value: &Value) -> Option<Filter> {
    match value {
        Value::Object(map) => {
            // Check for logical operators
            if let Some(Value::Array(arr)) = map.get("$and") {
                let filters: Vec<Filter> = arr.iter().filter_map(parse_filter).collect();
                return Some(Filter::And(filters));
            }
            if let Some(Value::Array(arr)) = map.get("$or") {
                let filters: Vec<Filter> = arr.iter().filter_map(parse_filter).collect();
                return Some(Filter::Or(filters));
            }
            if let Some(not_val) = map.get("$not") {
                if let Some(filter) = parse_filter(not_val) {
                    return Some(Filter::Not(Box::new(filter)));
                }
            }

            // Field conditions
            let conditions: Vec<Filter> = map
                .iter()
                .filter(|(k, _)| !k.starts_with('$'))
                .filter_map(|(field, cond)| parse_field_condition(field, cond))
                .collect();

            if conditions.len() == 1 {
                conditions.into_iter().next()
            } else if conditions.len() > 1 {
                Some(Filter::And(conditions))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Parse a field condition
fn parse_field_condition(field: &str, condition: &Value) -> Option<Filter> {
    match condition {
        Value::Object(map) => {
            let conditions: Vec<FilterCondition> = map
                .iter()
                .filter_map(|(op, val)| {
                    let operator = match op.as_str() {
                        "$eq" => Some(FilterOperator::Eq),
                        "$ne" => Some(FilterOperator::Ne),
                        "$gt" => Some(FilterOperator::Gt),
                        "$gte" => Some(FilterOperator::Gte),
                        "$lt" => Some(FilterOperator::Lt),
                        "$lte" => Some(FilterOperator::Lte),
                        "$in" => Some(FilterOperator::In),
                        "$nin" => Some(FilterOperator::NotIn),
                        "$contains" => Some(FilterOperator::Contains),
                        _ => None,
                    }?;

                    Some(FilterCondition {
                        field: field.to_string(),
                        operator,
                        value: val.clone(),
                    })
                })
                .collect();

            if conditions.len() == 1 {
                Some(Filter::Condition(conditions.into_iter().next()?))
            } else if conditions.len() > 1 {
                Some(Filter::And(
                    conditions.into_iter().map(Filter::Condition).collect(),
                ))
            } else {
                None
            }
        }
        // Shorthand: {"field": value} means {"field": {"$eq": value}}
        _ => Some(Filter::Condition(FilterCondition {
            field: field.to_string(),
            operator: FilterOperator::Eq,
            value: condition.clone(),
        })),
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

        let filter = parse_filter(&filter_json).unwrap();
        let metadata = json!({"category": "greeting", "priority": 2});

        assert!(filter.matches(Some(&metadata)));
    }

    #[test]
    fn test_parse_filter_shorthand() {
        let filter_json = json!({"category": "greeting"});

        let filter = parse_filter(&filter_json).unwrap();
        let metadata = json!({"category": "greeting"});

        assert!(filter.matches(Some(&metadata)));
    }
}
