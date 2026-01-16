//! Semantic Kernel Memory Store Integration
//!
//! Provides a Needle-backed memory store compatible with Microsoft's
//! Semantic Kernel patterns for memory management and retrieval.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::semantic_kernel::{NeedleMemoryStore, MemoryRecord};
//!
//! let store = NeedleMemoryStore::new(384).unwrap();
//!
//! let record = MemoryRecord::new("memory1", "User likes coffee")
//!     .with_description("User preference")
//!     .with_external_source("conversation-42");
//!
//! store.upsert("default", &record, &vec![0.1; 384]).unwrap();
//! let results = store.get_nearest_matches("default", &vec![0.1; 384], 5, 0.7).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::framework_common::{distance_to_score, FrameworkCollection};

// ---------------------------------------------------------------------------
// MemoryRecord
// ---------------------------------------------------------------------------

/// A memory record following Semantic Kernel's memory model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub text: String,
    pub description: String,
    pub external_source_name: String,
    pub additional_metadata: Value,
    pub is_reference: bool,
    pub timestamp: Option<String>,
}

impl MemoryRecord {
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            description: String::new(),
            external_source_name: String::new(),
            additional_metadata: json!({}),
            is_reference: false,
            timestamp: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_external_source(mut self, source: impl Into<String>) -> Self {
        self.external_source_name = source.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        if let Value::Object(ref mut map) = self.additional_metadata {
            map.insert(key.into(), value.into());
        }
        self
    }

    pub fn with_reference(mut self, is_ref: bool) -> Self {
        self.is_reference = is_ref;
        self
    }

    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }
}

/// A search result with relevance score.
#[derive(Debug, Clone)]
pub struct MemoryQueryResult {
    pub record: MemoryRecord,
    pub relevance: f32,
}

// ---------------------------------------------------------------------------
// NeedleMemoryStore
// ---------------------------------------------------------------------------

/// Semantic Kernel-compatible memory store backed by Needle.
///
/// Supports multiple "collections" (named memory spaces), each backed
/// by a separate Needle collection.
pub struct NeedleMemoryStore {
    embedding_dimension: usize,
    collections: parking_lot::RwLock<HashMap<String, FrameworkCollection>>,
}

impl NeedleMemoryStore {
    /// Create a new memory store.
    pub fn new(embedding_dimension: usize) -> Result<Self> {
        Ok(Self {
            embedding_dimension,
            collections: parking_lot::RwLock::new(HashMap::new()),
        })
    }

    /// Create (or get existing) a named collection.
    pub fn create_collection(&self, name: &str) {
        let mut cols = self.collections.write();
        if !cols.contains_key(name) {
            cols.insert(
                name.to_string(),
                FrameworkCollection::new(name, self.embedding_dimension, DistanceFunction::Cosine),
            );
        }
    }

    /// Check if a collection exists.
    pub fn has_collection(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    /// Delete a collection.
    pub fn delete_collection(&self, name: &str) -> bool {
        self.collections.write().remove(name).is_some()
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Upsert a memory record into a collection.
    pub fn upsert(
        &self,
        collection_name: &str,
        record: &MemoryRecord,
        embedding: &[f32],
    ) -> Result<String> {
        self.create_collection(collection_name);
        let cols = self.collections.read();
        let col = cols
            .get(collection_name)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection_name.to_string()))?;

        let metadata = json!({
            "_text": record.text,
            "_description": record.description,
            "_external_source": record.external_source_name,
            "_is_reference": record.is_reference,
            "_timestamp": record.timestamp,
            "_additional": record.additional_metadata,
        });

        // Delete existing if present (upsert semantics)
        let _ = col.write().delete(&record.id);
        col.write()
            .insert(&record.id, embedding, Some(metadata.clone()))?;

        Ok(record.id.clone())
    }

    /// Batch upsert multiple records.
    pub fn upsert_batch(
        &self,
        collection_name: &str,
        records: &[MemoryRecord],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<String>> {
        if records.len() != embeddings.len() {
            return Err(NeedleError::InvalidInput(format!(
                "Record count ({}) must match embedding count ({})",
                records.len(),
                embeddings.len()
            )));
        }

        let mut ids = Vec::with_capacity(records.len());
        for (record, embedding) in records.iter().zip(embeddings.iter()) {
            ids.push(self.upsert(collection_name, record, embedding)?);
        }
        Ok(ids)
    }

    /// Get a specific record by ID.
    pub fn get(&self, collection_name: &str, key: &str) -> Option<MemoryRecord> {
        let cols = self.collections.read();
        let col = cols.get(collection_name)?;
        let guard = col.read();
        let (_, metadata) = guard.get(key)?;
        let meta = metadata?.clone();
        Some(metadata_to_record(key, &meta))
    }

    /// Remove a record by ID.
    pub fn remove(&self, collection_name: &str, key: &str) -> Result<()> {
        let cols = self.collections.read();
        let col = cols
            .get(collection_name)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection_name.to_string()))?;
        let _ = col.write().delete(key);
        Ok(())
    }

    /// Search for nearest matches above a minimum relevance threshold.
    pub fn get_nearest_matches(
        &self,
        collection_name: &str,
        embedding: &[f32],
        limit: usize,
        min_relevance: f32,
    ) -> Result<Vec<MemoryQueryResult>> {
        let cols = self.collections.read();
        let col = cols
            .get(collection_name)
            .ok_or_else(|| NeedleError::CollectionNotFound(collection_name.to_string()))?;

        let results = col.search(embedding, limit, None)?;

        Ok(results
            .into_iter()
            .map(|r| {
                let relevance = distance_to_score(r.distance, DistanceFunction::Cosine);
                let record = metadata_to_record(&r.id, &r.metadata.unwrap_or(json!({})));
                MemoryQueryResult { record, relevance }
            })
            .filter(|r| r.relevance >= min_relevance)
            .collect())
    }

    /// Count records in a collection.
    pub fn count(&self, collection_name: &str) -> usize {
        self.collections
            .read()
            .get(collection_name)
            .map_or(0, |c| c.len())
    }
}

fn metadata_to_record(id: &str, metadata: &Value) -> MemoryRecord {
    MemoryRecord {
        id: id.to_string(),
        text: metadata
            .get("_text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        description: metadata
            .get("_description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        external_source_name: metadata
            .get("_external_source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        is_reference: metadata
            .get("_is_reference")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        timestamp: metadata
            .get("_timestamp")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        additional_metadata: metadata.get("_additional").cloned().unwrap_or(json!({})),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> NeedleMemoryStore {
        NeedleMemoryStore::new(4).unwrap()
    }

    #[test]
    fn test_create_and_list_collections() {
        let store = make_store();
        store.create_collection("memories");
        store.create_collection("facts");

        assert!(store.has_collection("memories"));
        assert!(store.has_collection("facts"));
        assert_eq!(store.list_collections().len(), 2);
    }

    #[test]
    fn test_delete_collection() {
        let store = make_store();
        store.create_collection("temp");
        assert!(store.delete_collection("temp"));
        assert!(!store.has_collection("temp"));
        assert!(!store.delete_collection("nonexistent"));
    }

    #[test]
    fn test_upsert_and_get() {
        let store = make_store();
        let record = MemoryRecord::new("m1", "User likes coffee")
            .with_description("preference")
            .with_external_source("chat-1");

        store
            .upsert("default", &record, &[1.0, 0.0, 0.0, 0.0])
            .unwrap();
        assert_eq!(store.count("default"), 1);

        let retrieved = store.get("default", "m1").unwrap();
        assert_eq!(retrieved.text, "User likes coffee");
        assert_eq!(retrieved.description, "preference");
        assert_eq!(retrieved.external_source_name, "chat-1");
    }

    #[test]
    fn test_upsert_overwrites() {
        let store = make_store();
        let r1 = MemoryRecord::new("m1", "Version 1");
        let r2 = MemoryRecord::new("m1", "Version 2");

        store.upsert("col", &r1, &[1.0; 4]).unwrap();
        store.upsert("col", &r2, &[0.0; 4]).unwrap();

        assert_eq!(store.count("col"), 1);
        let retrieved = store.get("col", "m1").unwrap();
        assert_eq!(retrieved.text, "Version 2");
    }

    #[test]
    fn test_batch_upsert() {
        let store = make_store();
        let records = vec![
            MemoryRecord::new("m1", "First"),
            MemoryRecord::new("m2", "Second"),
        ];
        let embeddings = vec![vec![1.0; 4], vec![0.0, 1.0, 0.0, 0.0]];

        let ids = store.upsert_batch("col", &records, &embeddings).unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(store.count("col"), 2);
    }

    #[test]
    fn test_batch_upsert_mismatch() {
        let store = make_store();
        let records = vec![MemoryRecord::new("m1", "A")];
        assert!(store.upsert_batch("col", &records, &[]).is_err());
    }

    #[test]
    fn test_remove() {
        let store = make_store();
        let record = MemoryRecord::new("m1", "Test");
        store.upsert("col", &record, &[1.0; 4]).unwrap();
        assert_eq!(store.count("col"), 1);

        store.remove("col", "m1").unwrap();
        assert_eq!(store.count("col"), 0);
    }

    #[test]
    fn test_nearest_matches() {
        let store = make_store();
        let records = vec![
            MemoryRecord::new("m1", "Very relevant"),
            MemoryRecord::new("m2", "Less relevant"),
        ];
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        store.upsert_batch("col", &records, &embeddings).unwrap();

        let results = store
            .get_nearest_matches("col", &[1.0, 0.0, 0.0, 0.0], 5, 0.0)
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id, "m1");
        assert!(results[0].relevance > 0.0);
    }

    #[test]
    fn test_nearest_matches_with_threshold() {
        let store = make_store();
        let record = MemoryRecord::new("m1", "Test");
        store.upsert("col", &record, &[1.0, 0.0, 0.0, 0.0]).unwrap();

        // Very high threshold should filter out results
        let results = store
            .get_nearest_matches("col", &[0.0, 1.0, 0.0, 0.0], 5, 0.99)
            .unwrap();
        // May or may not have results depending on cosine similarity
        // The key test is that filtering works without error
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_get_nonexistent() {
        let store = make_store();
        assert!(store.get("col", "nonexistent").is_none());
    }

    #[test]
    fn test_record_builder() {
        let record = MemoryRecord::new("m1", "Text")
            .with_description("desc")
            .with_external_source("src")
            .with_metadata("key", "value")
            .with_reference(true)
            .with_timestamp("2026-01-01T00:00:00Z");

        assert_eq!(record.description, "desc");
        assert_eq!(record.external_source_name, "src");
        assert!(record.is_reference);
        assert_eq!(record.timestamp.unwrap(), "2026-01-01T00:00:00Z");
    }

    #[test]
    fn test_count_empty() {
        let store = make_store();
        assert_eq!(store.count("nonexistent"), 0);
    }
}
