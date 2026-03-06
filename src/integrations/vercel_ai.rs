//! Vercel AI SDK Integration
//!
//! Provides a vector store adapter compatible with the Vercel AI SDK ecosystem.
//! This enables using Needle as a retrieval backend for AI applications built
//! with the Vercel AI SDK, Next.js, and edge runtimes.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::vercel_ai::{VercelAiStore, VercelAiConfig};
//!
//! let config = VercelAiConfig::default();
//! let store = VercelAiStore::new("documents", 384, config).unwrap();
//!
//! store.add_document("doc1", &vec![0.1; 384], "Hello world", None).unwrap();
//!
//! let results = store.retrieve(&vec![0.1; 384], None).unwrap();
//! for result in results {
//!     println!("Score: {:.4}, Content: {}", result.score, result.content);
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collection::Collection;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::framework_common::{distance_to_score, FrameworkCollection};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Vercel AI adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VercelAiConfig {
    /// Maximum number of results to return.
    pub max_results: usize,
    /// Minimum similarity score threshold (0.0 – 1.0).
    pub score_threshold: Option<f32>,
    /// Whether to include metadata in results.
    pub include_metadata: bool,
    /// Distance function used for scoring.
    #[serde(default)]
    pub distance_function: DistanceFunction,
    /// Key in metadata where content is stored.
    #[serde(default = "default_content_key")]
    pub content_key: String,
}

fn default_content_key() -> String {
    "_content".to_string()
}

impl Default for VercelAiConfig {
    fn default() -> Self {
        Self {
            max_results: 10,
            score_threshold: None,
            include_metadata: true,
            distance_function: DistanceFunction::Cosine,
            content_key: default_content_key(),
        }
    }
}

// ---------------------------------------------------------------------------
// RetrievalResult
// ---------------------------------------------------------------------------

/// A retrieval result compatible with the Vercel AI SDK.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievalResult {
    /// Unique identifier of the document.
    pub id: String,
    /// Text content of the document.
    pub content: String,
    /// Similarity score (0.0 – 1.0).
    pub score: f32,
    /// Optional metadata associated with the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------------
// VercelAiStore
// ---------------------------------------------------------------------------

/// Vercel AI SDK compatible vector store backed by Needle.
pub struct VercelAiStore {
    inner: FrameworkCollection,
    config: VercelAiConfig,
}

impl VercelAiStore {
    /// Create a new store with default cosine distance.
    pub fn new(
        collection_name: &str,
        embedding_dimension: usize,
        config: VercelAiConfig,
    ) -> Result<Self> {
        let inner = FrameworkCollection::new(
            collection_name,
            embedding_dimension,
            config.distance_function,
        );
        Ok(Self { inner, config })
    }

    /// Create from an existing [`Collection`].
    pub fn from_collection(collection: Collection, config: VercelAiConfig) -> Self {
        Self {
            inner: FrameworkCollection::from_collection(collection),
            config,
        }
    }

    /// Retrieve relevant documents for a query vector.
    ///
    /// Returns results in the format expected by Vercel AI SDK retrieval tools.
    pub fn retrieve(
        &self,
        query: &[f32],
        top_k: Option<usize>,
    ) -> Result<Vec<RetrievalResult>> {
        let k = top_k.unwrap_or(self.config.max_results);
        let results = self.inner.search(query, k, None)?;

        let mut out: Vec<RetrievalResult> = results
            .into_iter()
            .map(|r| self.result_to_retrieval(r))
            .collect();

        if let Some(threshold) = self.config.score_threshold {
            out.retain(|r| r.score >= threshold);
        }

        Ok(out)
    }

    /// Add a document with embedding to the store.
    pub fn add_document(
        &self,
        id: &str,
        embedding: &[f32],
        content: &str,
        metadata: Option<Value>,
    ) -> Result<()> {
        let mut meta = metadata.unwrap_or_else(|| json!({}));
        if let Value::Object(ref mut map) = meta {
            map.insert(
                self.config.content_key.clone(),
                Value::String(content.to_string()),
            );
        }
        self.inner.write().insert(id, embedding, Some(meta))?;
        Ok(())
    }

    /// Add multiple documents with embeddings in one call.
    pub fn add_documents(
        &self,
        ids: &[&str],
        embeddings: &[Vec<f32>],
        contents: &[&str],
        metadatas: Option<&[Value]>,
    ) -> Result<usize> {
        if ids.len() != embeddings.len() || ids.len() != contents.len() {
            return Err(NeedleError::InvalidInput(format!(
                "ids ({}), embeddings ({}), and contents ({}) must have the same length",
                ids.len(),
                embeddings.len(),
                contents.len()
            )));
        }

        for (i, (id, embedding)) in ids.iter().zip(embeddings.iter()).enumerate() {
            let user_meta = metadatas.and_then(|m| m.get(i)).cloned();
            self.add_document(id, embedding, contents[i], user_meta)?;
        }
        Ok(ids.len())
    }

    /// Delete a document by ID.
    pub fn delete_document(&self, id: &str) -> Result<bool> {
        self.inner.write().delete(id)
    }

    /// Number of documents in the store.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn result_to_retrieval(&self, result: crate::collection::SearchResult) -> RetrievalResult {
        let content = result
            .metadata
            .as_ref()
            .and_then(|m| m.get(&self.config.content_key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let score = distance_to_score(result.distance, self.config.distance_function);

        let metadata = if self.config.include_metadata {
            result.metadata
        } else {
            None
        };

        RetrievalResult {
            id: result.id,
            content,
            score,
            metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> VercelAiStore {
        VercelAiStore::new("test_vercel", 4, VercelAiConfig::default()).unwrap()
    }

    #[test]
    fn test_add_and_retrieve() {
        let store = make_store();

        store
            .add_document("d1", &[1.0, 0.0, 0.0, 0.0], "Hello world", None)
            .unwrap();
        store
            .add_document("d2", &[0.0, 1.0, 0.0, 0.0], "Goodbye world", None)
            .unwrap();

        assert_eq!(store.len(), 2);

        let results = store.retrieve(&[1.0, 0.0, 0.0, 0.0], Some(2)).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1");
        assert_eq!(results[0].content, "Hello world");
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_score_threshold_filtering() {
        let config = VercelAiConfig {
            score_threshold: Some(0.99),
            ..Default::default()
        };
        let store = VercelAiStore::new("test_threshold", 4, config).unwrap();

        store
            .add_document("d1", &[1.0, 0.0, 0.0, 0.0], "Close match", None)
            .unwrap();
        store
            .add_document("d2", &[0.0, 1.0, 0.0, 0.0], "Far match", None)
            .unwrap();

        let results = store.retrieve(&[1.0, 0.0, 0.0, 0.0], None).unwrap();
        // Only the very close match should survive the 0.99 threshold
        assert!(results.len() <= 1);
    }

    #[test]
    fn test_delete_document() {
        let store = make_store();
        store
            .add_document("d1", &[1.0; 4], "Content", None)
            .unwrap();
        assert_eq!(store.len(), 1);

        store.delete_document("d1").unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn test_add_documents_batch() {
        let store = make_store();
        let ids = vec!["d1", "d2"];
        let embeddings = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let contents = vec!["First", "Second"];

        let count = store
            .add_documents(&ids, &embeddings, &contents, None)
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_add_documents_mismatch() {
        let store = make_store();
        let ids = vec!["d1"];
        let embeddings: Vec<Vec<f32>> = vec![];
        let contents = vec!["A"];

        assert!(store
            .add_documents(&ids, &embeddings, &contents, None)
            .is_err());
    }

    #[test]
    fn test_metadata_included() {
        let config = VercelAiConfig {
            include_metadata: true,
            ..Default::default()
        };
        let store = VercelAiStore::new("test_meta", 4, config).unwrap();

        store
            .add_document(
                "d1",
                &[1.0; 4],
                "Content",
                Some(json!({"source": "test"})),
            )
            .unwrap();

        let results = store.retrieve(&[1.0; 4], Some(1)).unwrap();
        assert!(results[0].metadata.is_some());
    }

    #[test]
    fn test_metadata_excluded() {
        let config = VercelAiConfig {
            include_metadata: false,
            ..Default::default()
        };
        let store = VercelAiStore::new("test_no_meta", 4, config).unwrap();

        store
            .add_document(
                "d1",
                &[1.0; 4],
                "Content",
                Some(json!({"source": "test"})),
            )
            .unwrap();

        let results = store.retrieve(&[1.0; 4], Some(1)).unwrap();
        assert!(results[0].metadata.is_none());
    }

    #[test]
    fn test_empty_store() {
        let store = make_store();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_from_collection() {
        let collection =
            crate::collection::Collection::new(crate::collection::CollectionConfig::new("test", 4));
        let store = VercelAiStore::from_collection(collection, VercelAiConfig::default());
        assert!(store.is_empty());
    }
}
