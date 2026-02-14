//! Haystack Document Store Integration
//!
//! Provides a Needle-backed document store compatible with the Haystack
//! framework's retrieval pipeline patterns.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::haystack::{NeedleDocumentStore, HaystackDocument};
//!
//! let store = NeedleDocumentStore::new("documents", 384).unwrap();
//!
//! let doc = HaystackDocument::new("doc1", "Hello world")
//!     .with_metadata("source", "greeting");
//! store.write_documents(&[doc], &[vec![0.1; 384]]).unwrap();
//!
//! let results = store.query_by_embedding(&vec![0.1; 384], 10).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collection::SearchResult;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::framework_common::{distance_to_score, FrameworkCollection};
use crate::metadata::Filter;

// ---------------------------------------------------------------------------
// HaystackDocument
// ---------------------------------------------------------------------------

/// A document in the Haystack framework style.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HaystackDocument {
    pub id: String,
    pub content: String,
    pub content_type: ContentType,
    pub metadata: Value,
    pub score: Option<f32>,
}

/// Content type of a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Text,
    Table,
    Image,
}

impl Default for ContentType {
    fn default() -> Self {
        Self::Text
    }
}

impl HaystackDocument {
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            content_type: ContentType::Text,
            metadata: json!({}),
            score: None,
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        if let Value::Object(ref mut map) = self.metadata {
            map.insert(key.into(), value.into());
        }
        self
    }

    pub fn with_content_type(mut self, ct: ContentType) -> Self {
        self.content_type = ct;
        self
    }

    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }
}

// ---------------------------------------------------------------------------
// NeedleDocumentStore
// ---------------------------------------------------------------------------

/// Configuration for the Haystack document store.
#[derive(Debug, Clone)]
pub struct DocumentStoreConfig {
    pub collection_name: String,
    pub embedding_dimension: usize,
    pub distance_function: DistanceFunction,
    pub store_content: bool,
    pub content_key: String,
    pub duplicate_policy: DuplicatePolicy,
}

/// Policy for handling duplicate document IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DuplicatePolicy {
    /// Overwrite existing documents with the same ID.
    Overwrite,
    /// Skip documents that already exist.
    Skip,
    /// Return an error if a duplicate is found.
    Error,
}

impl Default for DuplicatePolicy {
    fn default() -> Self {
        Self::Overwrite
    }
}

impl DocumentStoreConfig {
    pub fn new(collection_name: impl Into<String>, embedding_dimension: usize) -> Self {
        Self {
            collection_name: collection_name.into(),
            embedding_dimension,
            distance_function: DistanceFunction::Cosine,
            store_content: true,
            content_key: "_content".to_string(),
            duplicate_policy: DuplicatePolicy::default(),
        }
    }
}

/// Haystack-compatible document store backed by Needle.
pub struct NeedleDocumentStore {
    inner: FrameworkCollection,
    config: DocumentStoreConfig,
}

impl NeedleDocumentStore {
    /// Create a new document store with default cosine distance.
    pub fn new(collection_name: &str, embedding_dimension: usize) -> Result<Self> {
        let config = DocumentStoreConfig::new(collection_name, embedding_dimension);
        let inner = FrameworkCollection::new(
            collection_name,
            embedding_dimension,
            config.distance_function,
        );
        Ok(Self { inner, config })
    }

    /// Create from an existing collection.
    pub fn from_collection(
        collection: crate::collection::Collection,
        config: DocumentStoreConfig,
    ) -> Self {
        let inner = FrameworkCollection::from_collection(collection);
        Self { inner, config }
    }

    /// Write documents with their embeddings into the store.
    pub fn write_documents(
        &self,
        documents: &[HaystackDocument],
        embeddings: &[Vec<f32>],
    ) -> Result<usize> {
        if documents.len() != embeddings.len() {
            return Err(NeedleError::InvalidInput(format!(
                "Document count ({}) must match embedding count ({})",
                documents.len(),
                embeddings.len()
            )));
        }

        let mut written = 0;
        for (doc, embedding) in documents.iter().zip(embeddings.iter()) {
            let mut metadata = doc.metadata.clone();
            if self.config.store_content {
                if let Value::Object(ref mut map) = metadata {
                    map.insert(
                        self.config.content_key.clone(),
                        Value::String(doc.content.clone()),
                    );
                    map.insert("_content_type".to_string(), json!(doc.content_type));
                }
            }

            match self.config.duplicate_policy {
                DuplicatePolicy::Overwrite => {
                    let _ = self.inner.write().delete(&doc.id);
                    self.inner
                        .write()
                        .insert(&doc.id, embedding, Some(metadata))?;
                    written += 1;
                }
                DuplicatePolicy::Skip => {
                    let exists = self.inner.read().get(&doc.id).is_some();
                    if !exists {
                        self.inner
                            .write()
                            .insert(&doc.id, embedding, Some(metadata))?;
                        written += 1;
                    }
                }
                DuplicatePolicy::Error => {
                    let exists = self.inner.read().get(&doc.id).is_some();
                    if exists {
                        return Err(NeedleError::DuplicateId(doc.id.clone()));
                    }
                    self.inner
                        .write()
                        .insert(&doc.id, embedding, Some(metadata))?;
                    written += 1;
                }
            }
        }

        Ok(written)
    }

    /// Delete documents by ID.
    pub fn delete_documents(&self, ids: &[String]) -> Result<usize> {
        let mut deleted = 0;
        for id in ids {
            if self.inner.write().delete(id)? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Query by embedding vector.
    pub fn query_by_embedding(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<HaystackDocument>> {
        let results = self.inner.search(query_embedding, top_k, None)?;
        Ok(results.into_iter().map(|r| self.result_to_doc(r)).collect())
    }

    /// Query by embedding with metadata filter.
    pub fn query_by_embedding_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
    ) -> Result<Vec<HaystackDocument>> {
        let results = self.inner.search(query_embedding, top_k, Some(filter))?;
        Ok(results.into_iter().map(|r| self.result_to_doc(r)).collect())
    }

    /// Count documents in the store.
    pub fn count_documents(&self) -> usize {
        self.inner.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get a specific document by ID.
    pub fn get_document(&self, id: &str) -> Option<HaystackDocument> {
        let col = self.inner.read();
        let (_, metadata) = col.get(id)?;
        let content = metadata
            .and_then(|m| m.get(&self.config.content_key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Some(HaystackDocument {
            id: id.to_string(),
            content,
            content_type: ContentType::Text,
            metadata: metadata.cloned().unwrap_or(json!({})),
            score: None,
        })
    }

    fn result_to_doc(&self, result: SearchResult) -> HaystackDocument {
        let content = result
            .metadata
            .as_ref()
            .and_then(|m| m.get(&self.config.content_key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let score = distance_to_score(result.distance, self.config.distance_function);

        HaystackDocument {
            id: result.id,
            content,
            content_type: ContentType::Text,
            metadata: result.metadata.unwrap_or(json!({})),
            score: Some(score),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> NeedleDocumentStore {
        NeedleDocumentStore::new("test_haystack", 4).unwrap()
    }

    #[test]
    fn test_write_and_query() {
        let store = make_store();
        let docs = vec![
            HaystackDocument::new("d1", "Hello world"),
            HaystackDocument::new("d2", "Goodbye world"),
        ];
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

        let written = store.write_documents(&docs, &embeddings).unwrap();
        assert_eq!(written, 2);
        assert_eq!(store.count_documents(), 2);

        let results = store.query_by_embedding(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1");
        assert!(results[0].score.unwrap() > 0.0);
    }

    #[test]
    fn test_write_with_metadata() {
        let store = make_store();
        let doc = HaystackDocument::new("d1", "Test content")
            .with_metadata("source", "test")
            .with_metadata("page", 42);

        store.write_documents(&[doc], &[vec![1.0; 4]]).unwrap();

        let retrieved = store.get_document("d1").unwrap();
        assert_eq!(retrieved.content, "Test content");
    }

    #[test]
    fn test_delete_documents() {
        let store = make_store();
        let docs = vec![
            HaystackDocument::new("d1", "A"),
            HaystackDocument::new("d2", "B"),
        ];
        store
            .write_documents(&docs, &[vec![1.0; 4], vec![0.0; 4]])
            .unwrap();
        assert_eq!(store.count_documents(), 2);

        let deleted = store.delete_documents(&["d1".to_string()]).unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(store.count_documents(), 1);
    }

    #[test]
    fn test_duplicate_policy_skip() {
        let config = DocumentStoreConfig {
            duplicate_policy: DuplicatePolicy::Skip,
            ..DocumentStoreConfig::new("test", 4)
        };
        let collection =
            crate::collection::Collection::new(crate::collection::CollectionConfig::new("test", 4));
        let store = NeedleDocumentStore::from_collection(collection, config);

        let doc = HaystackDocument::new("d1", "First");
        store.write_documents(&[doc], &[vec![1.0; 4]]).unwrap();

        let doc2 = HaystackDocument::new("d1", "Second");
        let written = store.write_documents(&[doc2], &[vec![0.0; 4]]).unwrap();
        assert_eq!(written, 0); // skipped
        assert_eq!(store.count_documents(), 1);
    }

    #[test]
    fn test_duplicate_policy_error() {
        let config = DocumentStoreConfig {
            duplicate_policy: DuplicatePolicy::Error,
            ..DocumentStoreConfig::new("test", 4)
        };
        let collection =
            crate::collection::Collection::new(crate::collection::CollectionConfig::new("test", 4));
        let store = NeedleDocumentStore::from_collection(collection, config);

        let doc = HaystackDocument::new("d1", "Content");
        store.write_documents(&[doc], &[vec![1.0; 4]]).unwrap();

        let doc2 = HaystackDocument::new("d1", "Content2");
        assert!(store.write_documents(&[doc2], &[vec![0.0; 4]]).is_err());
    }

    #[test]
    fn test_mismatched_counts() {
        let store = make_store();
        let docs = vec![HaystackDocument::new("d1", "A")];
        assert!(store.write_documents(&docs, &[]).is_err());
    }

    #[test]
    fn test_get_nonexistent() {
        let store = make_store();
        assert!(store.get_document("nonexistent").is_none());
    }

    #[test]
    fn test_content_type() {
        let doc = HaystackDocument::new("d1", "A").with_content_type(ContentType::Table);
        assert_eq!(doc.content_type, ContentType::Table);
    }

    #[test]
    fn test_empty_store() {
        let store = make_store();
        assert!(store.is_empty());
        assert_eq!(store.count_documents(), 0);
    }
}
