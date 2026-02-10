// Allow dead_code for this public API module - types are exported for library users
#![allow(dead_code)]
#![allow(clippy::wrong_self_convention)]

//! LangChain Integration
//!
//! Compatible interface for LangChain and similar frameworks.
//!
//! This module provides a VectorStore-like interface that follows LangChain's
//! patterns and naming conventions, making it easy to integrate Needle with
//! LangChain Python applications via FFI or as a drop-in replacement.
//!
//! # Features
//!
//! - LangChain-compatible `Document` abstraction
//! - `NeedleVectorStore` implementing VectorStore-like interface
//! - Synchronous and async variants of all methods
//! - Metadata filtering support
//! - Maximum Marginal Relevance (MMR) search
//! - Similarity search with relevance scores
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::langchain::{Document, NeedleVectorStore, NeedleVectorStoreConfig};
//! use serde_json::json;
//!
//! fn main() -> needle::Result<()> {
//!     // Create a vector store
//!     let config = NeedleVectorStoreConfig::new("documents", 384);
//!     let store = NeedleVectorStore::new(config)?;
//!
//!     // Add documents with text content
//!     let docs = vec![
//!         Document::new("Machine learning is a subset of AI.")
//!             .with_metadata(json!({"source": "intro.txt"})),
//!         Document::new("Deep learning uses neural networks.")
//!             .with_metadata(json!({"source": "deep.txt"})),
//!     ];
//!
//!     // Assuming you have embeddings from an embedding model
//!     let embeddings: Vec<Vec<f32>> = vec![vec![0.1; 384], vec![0.2; 384]];
//!     store.add_documents(&docs, &embeddings)?;
//!
//!     // Search for similar documents
//!     let query_embedding = vec![0.15; 384];
//!     let results = store.similarity_search(&query_embedding, 5)?;
//!
//!     for (doc, score) in results {
//!         println!("Score: {:.4}, Content: {}", score, doc.page_content);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! # Async Usage
//!
//! ```rust,ignore
//! use needle::langchain::{Document, NeedleVectorStore, NeedleVectorStoreConfig};
//!
//! #[tokio::main]
//! async fn main() -> needle::Result<()> {
//!     let config = NeedleVectorStoreConfig::new("documents", 384);
//!     let store = NeedleVectorStore::new(config)?;
//!
//!     // Async similarity search
//!     let results = store.asimilarity_search(&query_embedding, 5).await?;
//!
//!     Ok(())
//! }
//! ```

use crate::collection::{Collection, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::framework_common::FrameworkCollection;
use crate::metadata::Filter;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use uuid::Uuid;

/// A document abstraction compatible with LangChain's Document class.
///
/// Documents consist of page content (the text) and associated metadata.
/// This is the primary unit for storing and retrieving information in
/// the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// The main text content of the document
    pub page_content: String,
    /// Metadata associated with the document
    pub metadata: Value,
    /// Unique identifier for the document
    #[serde(default = "generate_doc_id")]
    pub id: String,
}

fn generate_doc_id() -> String {
    Uuid::new_v4().to_string()
}

impl Document {
    /// Create a new document with the given content
    pub fn new(page_content: impl Into<String>) -> Self {
        Self {
            page_content: page_content.into(),
            metadata: Value::Object(serde_json::Map::new()),
            id: generate_doc_id(),
        }
    }

    /// Create a document with content and metadata
    pub fn with_content_and_metadata(page_content: impl Into<String>, metadata: Value) -> Self {
        Self {
            page_content: page_content.into(),
            metadata,
            id: generate_doc_id(),
        }
    }

    /// Set the document's metadata
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the document's ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Add a single metadata field
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        if let Value::Object(ref mut map) = self.metadata {
            map.insert(key.into(), value.into());
        }
    }

    /// Get a metadata field by key
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        if let Value::Object(ref map) = self.metadata {
            map.get(key)
        } else {
            None
        }
    }
}

impl Default for Document {
    fn default() -> Self {
        Self::new("")
    }
}

/// Configuration for NeedleVectorStore
#[derive(Debug, Clone)]
pub struct NeedleVectorStoreConfig {
    /// Name of the collection
    pub collection_name: String,
    /// Dimension of the embedding vectors
    pub embedding_dimension: usize,
    /// Distance function to use
    pub distance_function: DistanceFunction,
    /// Relevance score function (how to convert distance to relevance)
    pub relevance_score_fn: RelevanceScoreFunction,
    /// Whether to store the page content in metadata
    pub store_content: bool,
    /// Key in metadata where page content is stored
    pub content_key: String,
}

impl NeedleVectorStoreConfig {
    /// Create a new configuration with required parameters
    pub fn new(collection_name: impl Into<String>, embedding_dimension: usize) -> Self {
        Self {
            collection_name: collection_name.into(),
            embedding_dimension,
            distance_function: DistanceFunction::Cosine,
            relevance_score_fn: RelevanceScoreFunction::Cosine,
            store_content: true,
            content_key: "_page_content".to_string(),
        }
    }

    /// Set the distance function
    pub fn with_distance_function(mut self, distance_function: DistanceFunction) -> Self {
        self.distance_function = distance_function;
        self
    }

    /// Set the relevance score function
    pub fn with_relevance_score_fn(mut self, relevance_score_fn: RelevanceScoreFunction) -> Self {
        self.relevance_score_fn = relevance_score_fn;
        self
    }

    /// Set whether to store page content in metadata
    pub fn with_store_content(mut self, store_content: bool) -> Self {
        self.store_content = store_content;
        self
    }

    /// Set the content key in metadata
    pub fn with_content_key(mut self, content_key: impl Into<String>) -> Self {
        self.content_key = content_key.into();
        self
    }
}

/// Function to convert distance to relevance score
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelevanceScoreFunction {
    /// Cosine: 1 - distance (for cosine distance)
    Cosine,
    /// Euclidean: 1 / (1 + distance)
    Euclidean,
    /// Dot Product: (distance + 1) / 2 (assuming normalized vectors)
    DotProduct,
    /// Max inner product: direct negative distance
    MaxInnerProduct,
}

impl RelevanceScoreFunction {
    /// Convert a distance value to a relevance score
    pub fn to_relevance_score(&self, distance: f32) -> f32 {
        match self {
            Self::Cosine => 1.0 - distance,
            Self::Euclidean => 1.0 / (1.0 + distance),
            Self::DotProduct => (distance + 1.0) / 2.0,
            Self::MaxInnerProduct => -distance,
        }
    }
}

/// Result from a similarity search with score
pub type DocumentWithScore = (Document, f32);

/// A LangChain-compatible vector store backed by Needle.
///
/// This provides the same interface as LangChain's VectorStore class,
/// allowing easy integration with LangChain-based applications.
pub struct NeedleVectorStore {
    /// The underlying collection
    collection: FrameworkCollection,
    /// Configuration
    config: NeedleVectorStoreConfig,
}

impl NeedleVectorStore {
    /// Create a new vector store with the given configuration
    pub fn new(config: NeedleVectorStoreConfig) -> Result<Self> {
        let collection = FrameworkCollection::new(
            &config.collection_name,
            config.embedding_dimension,
            config.distance_function,
        );

        Ok(Self {
            collection,
            config,
        })
    }

    /// Create a vector store from an existing collection
    pub fn from_collection(collection: Collection, config: NeedleVectorStoreConfig) -> Self {
        Self {
            collection: FrameworkCollection::from_collection(collection),
            config,
        }
    }

    /// Get the collection name
    pub fn collection_name(&self) -> &str {
        &self.config.collection_name
    }

    /// Get the embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.config.embedding_dimension
    }

    /// Get the number of documents in the store
    pub fn len(&self) -> usize {
        self.collection.len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.collection.is_empty()
    }

    // =========================================================================
    // Core VectorStore Interface - Synchronous Methods
    // =========================================================================

    /// Add texts with embeddings to the vector store.
    ///
    /// This is the fundamental method for adding content. The texts will be
    /// stored as metadata alongside their embeddings.
    ///
    /// # Arguments
    /// * `texts` - The text contents to add
    /// * `embeddings` - The embedding vectors for each text
    /// * `metadatas` - Optional metadata for each text
    /// * `ids` - Optional IDs for each text (auto-generated if not provided)
    ///
    /// # Returns
    /// The IDs of the added texts
    pub fn add_texts(
        &self,
        texts: &[String],
        embeddings: &[Vec<f32>],
        metadatas: Option<&[Value]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        if texts.len() != embeddings.len() {
            return Err(NeedleError::InvalidInput(
                "Number of texts must match number of embeddings".to_string(),
            ));
        }

        let mut collection = self.collection.write();
        let mut result_ids = Vec::with_capacity(texts.len());

        for (i, (text, embedding)) in texts.iter().zip(embeddings.iter()).enumerate() {
            // Generate or use provided ID
            let id = ids
                .and_then(|ids| ids.get(i).cloned())
                .unwrap_or_else(generate_doc_id);

            // Build metadata
            let mut metadata = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_else(|| Value::Object(serde_json::Map::new()));

            // Store page content in metadata if configured
            if self.config.store_content {
                if let Value::Object(ref mut map) = metadata {
                    map.insert(self.config.content_key.clone(), Value::String(text.clone()));
                }
            }

            collection.insert(&id, embedding, Some(metadata))?;
            result_ids.push(id);
        }

        Ok(result_ids)
    }

    /// Add documents with embeddings to the vector store.
    ///
    /// # Arguments
    /// * `documents` - The documents to add
    /// * `embeddings` - The embedding vectors for each document
    ///
    /// # Returns
    /// The IDs of the added documents
    pub fn add_documents(
        &self,
        documents: &[Document],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<String>> {
        if documents.len() != embeddings.len() {
            return Err(NeedleError::InvalidInput(
                "Number of documents must match number of embeddings".to_string(),
            ));
        }

        let texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let metadatas: Vec<Value> = documents.iter().map(|d| d.metadata.clone()).collect();
        let ids: Vec<String> = documents.iter().map(|d| d.id.clone()).collect();

        self.add_texts(&texts, embeddings, Some(&metadatas), Some(&ids))
    }

    /// Perform a similarity search and return documents.
    ///
    /// # Arguments
    /// * `query_embedding` - The embedding vector to search with
    /// * `k` - The number of results to return
    ///
    /// # Returns
    /// A list of (Document, score) tuples, sorted by relevance
    pub fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<DocumentWithScore>> {
        self.similarity_search_with_filter(query_embedding, k, None)
    }

    /// Perform a similarity search with a metadata filter.
    ///
    /// # Arguments
    /// * `query_embedding` - The embedding vector to search with
    /// * `k` - The number of results to return
    /// * `filter` - Optional metadata filter
    ///
    /// # Returns
    /// A list of (Document, score) tuples, sorted by relevance
    pub fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<DocumentWithScore>> {
        let results = self.collection.search(query_embedding, k, filter)?;
        self.results_to_documents(results)
    }

    /// Perform a similarity search and return scores.
    ///
    /// Returns documents with their relevance scores (higher is better).
    pub fn similarity_search_with_score(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<DocumentWithScore>> {
        self.similarity_search(query_embedding, k)
    }

    /// Perform a similarity search by vector with relevance scores.
    ///
    /// This is an alias for similarity_search_with_score for LangChain compatibility.
    pub fn similarity_search_with_relevance_scores(
        &self,
        query_embedding: &[f32],
        k: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<DocumentWithScore>> {
        let results = self.similarity_search_with_score(query_embedding, k)?;

        if let Some(threshold) = score_threshold {
            Ok(results.into_iter().filter(|(_, score)| *score >= threshold).collect())
        } else {
            Ok(results)
        }
    }

    /// Maximum Marginal Relevance (MMR) search.
    ///
    /// MMR balances relevance and diversity in the results. It selects documents
    /// that are both similar to the query and dissimilar to already-selected documents.
    ///
    /// # Arguments
    /// * `query_embedding` - The embedding vector to search with
    /// * `k` - The number of results to return
    /// * `fetch_k` - The number of candidates to fetch before reranking
    /// * `lambda_mult` - Balance between relevance (1.0) and diversity (0.0). Default: 0.5
    ///
    /// # Returns
    /// A list of (Document, score) tuples selected by MMR
    pub fn max_marginal_relevance_search(
        &self,
        query_embedding: &[f32],
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<DocumentWithScore>> {
        self.max_marginal_relevance_search_with_filter(
            query_embedding,
            k,
            fetch_k,
            lambda_mult,
            None,
        )
    }

    /// Maximum Marginal Relevance search with metadata filter.
    ///
    /// # Arguments
    /// * `query_embedding` - The embedding vector to search with
    /// * `k` - The number of results to return
    /// * `fetch_k` - The number of candidates to fetch before reranking
    /// * `lambda_mult` - Balance between relevance (1.0) and diversity (0.0)
    /// * `filter` - Optional metadata filter
    ///
    /// # Returns
    /// A list of (Document, score) tuples selected by MMR
    pub fn max_marginal_relevance_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
        filter: Option<&Filter>,
    ) -> Result<Vec<DocumentWithScore>> {
        // Fetch more candidates than needed
        let candidates = self.collection.search(query_embedding, fetch_k, filter)?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Get embeddings for candidates
        let collection = self.collection.read();
        let candidate_embeddings: Vec<(SearchResult, Vec<f32>)> = candidates
            .into_iter()
            .filter_map(|result| {
                collection.get(&result.id).map(|(vec, _): (&[f32], _)| (result, vec.to_vec()))
            })
            .collect();

        // Apply MMR
        let selected = self.mmr_select(
            query_embedding,
            &candidate_embeddings,
            k,
            lambda_mult,
        );

        // Convert to documents
        selected
            .into_iter()
            .map(|(result, _)| self.result_to_document(result))
            .collect()
    }

    /// Delete documents by IDs.
    ///
    /// # Arguments
    /// * `ids` - The IDs of documents to delete
    ///
    /// # Returns
    /// true if all documents were deleted successfully
    pub fn delete(&self, ids: &[String]) -> Result<bool> {
        let mut collection = self.collection.write();
        let mut all_deleted = true;

        for id in ids {
            if !collection.delete(id)? {
                all_deleted = false;
            }
        }

        Ok(all_deleted)
    }

    /// Get a document by ID.
    ///
    /// # Arguments
    /// * `id` - The document ID
    ///
    /// # Returns
    /// The document if found
    pub fn get(&self, id: &str) -> Option<Document> {
        let collection = self.collection.read();
        let (_, metadata): (&[f32], Option<&Value>) = collection.get(id)?;

        let page_content = metadata
            .and_then(|m: &Value| {
                if let Value::Object(ref map) = m {
                    map.get(&self.config.content_key)
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        Some(Document {
            id: id.to_string(),
            page_content,
            metadata: metadata.cloned().unwrap_or(Value::Object(serde_json::Map::new())),
        })
    }

    /// Update a document's metadata.
    ///
    /// # Arguments
    /// * `id` - The document ID
    /// * `metadata` - New metadata to set
    pub fn update_metadata(&self, id: &str, metadata: Value) -> Result<()> {
        self.collection.update_metadata(id, Some(metadata))
    }

    // =========================================================================
    // Async Variants (for compatibility with async frameworks)
    // =========================================================================

    /// Async version of add_texts.
    pub async fn aadd_texts(
        &self,
        texts: &[String],
        embeddings: &[Vec<f32>],
        metadatas: Option<&[Value]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        // For now, we just wrap the sync version
        // In a real async implementation, this would use tokio::spawn_blocking
        self.add_texts(texts, embeddings, metadatas, ids)
    }

    /// Async version of add_documents.
    pub async fn aadd_documents(
        &self,
        documents: &[Document],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<String>> {
        self.add_documents(documents, embeddings)
    }

    /// Async version of similarity_search.
    pub async fn asimilarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<DocumentWithScore>> {
        self.similarity_search(query_embedding, k)
    }

    /// Async version of similarity_search_with_filter.
    pub async fn asimilarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<DocumentWithScore>> {
        self.similarity_search_with_filter(query_embedding, k, filter)
    }

    /// Async version of similarity_search_with_score.
    pub async fn asimilarity_search_with_score(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<DocumentWithScore>> {
        self.similarity_search_with_score(query_embedding, k)
    }

    /// Async version of similarity_search_with_relevance_scores.
    pub async fn asimilarity_search_with_relevance_scores(
        &self,
        query_embedding: &[f32],
        k: usize,
        score_threshold: Option<f32>,
    ) -> Result<Vec<DocumentWithScore>> {
        self.similarity_search_with_relevance_scores(query_embedding, k, score_threshold)
    }

    /// Async version of max_marginal_relevance_search.
    pub async fn amax_marginal_relevance_search(
        &self,
        query_embedding: &[f32],
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<DocumentWithScore>> {
        self.max_marginal_relevance_search(query_embedding, k, fetch_k, lambda_mult)
    }

    /// Async version of max_marginal_relevance_search_with_filter.
    pub async fn amax_marginal_relevance_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
        filter: Option<&Filter>,
    ) -> Result<Vec<DocumentWithScore>> {
        self.max_marginal_relevance_search_with_filter(
            query_embedding,
            k,
            fetch_k,
            lambda_mult,
            filter,
        )
    }

    /// Async version of delete.
    pub async fn adelete(&self, ids: &[String]) -> Result<bool> {
        self.delete(ids)
    }

    // =========================================================================
    // Batch Operations
    // =========================================================================

    /// Batch similarity search for multiple queries.
    ///
    /// Runs searches in parallel using Rayon.
    pub fn batch_similarity_search(
        &self,
        query_embeddings: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<Vec<DocumentWithScore>>> {
        let collection = self.collection.read();

        let results: Vec<Result<Vec<DocumentWithScore>>> = query_embeddings
            .par_iter()
            .map(|query| {
                let search_results = collection.search(query, k)?;
                self.results_to_documents(search_results)
            })
            .collect();

        results.into_iter().collect()
    }

    /// Batch similarity search with filter.
    pub fn batch_similarity_search_with_filter(
        &self,
        query_embeddings: &[Vec<f32>],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<DocumentWithScore>>> {
        let collection = self.collection.read();

        let results: Vec<Result<Vec<DocumentWithScore>>> = query_embeddings
            .par_iter()
            .map(|query| {
                let search_results = collection.search_with_filter(query, k, filter)?;
                self.results_to_documents(search_results)
            })
            .collect();

        results.into_iter().collect()
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /// Get collection statistics.
    pub fn stats(&self) -> CollectionStats {
        let needle_stats = self.collection.stats();

        CollectionStats {
            name: needle_stats.name,
            document_count: needle_stats.vector_count,
            dimensions: needle_stats.dimensions,
            total_memory_bytes: needle_stats.total_memory_bytes,
        }
    }

    /// Compact the store, removing deleted documents.
    pub fn compact(&self) -> Result<usize> {
        self.collection.compact()
    }

    /// Serialize the store to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.collection.to_bytes()
    }

    /// Deserialize a store from bytes.
    pub fn from_bytes(bytes: &[u8], config: NeedleVectorStoreConfig) -> Result<Self> {
        let collection = FrameworkCollection::from_bytes(bytes)?;
        Ok(Self { collection, config })
    }

    // =========================================================================
    // Private Helper Methods
    // =========================================================================

    /// Convert search results to documents with scores.
    fn results_to_documents(&self, results: Vec<SearchResult>) -> Result<Vec<DocumentWithScore>> {
        results
            .into_iter()
            .map(|result| self.result_to_document(result))
            .collect()
    }

    /// Convert a single search result to a document with score.
    fn result_to_document(&self, result: SearchResult) -> Result<DocumentWithScore> {
        let page_content = result
            .metadata
            .as_ref()
            .and_then(|m| {
                if let Value::Object(ref map) = m {
                    map.get(&self.config.content_key)
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();

        let score = self.config.relevance_score_fn.to_relevance_score(result.distance);

        Ok((
            Document {
                id: result.id,
                page_content,
                metadata: result.metadata.unwrap_or(Value::Object(serde_json::Map::new())),
            },
            score,
        ))
    }

    /// MMR selection algorithm.
    fn mmr_select(
        &self,
        query_embedding: &[f32],
        candidates: &[(SearchResult, Vec<f32>)],
        k: usize,
        lambda_mult: f32,
    ) -> Vec<(SearchResult, Vec<f32>)> {
        if candidates.is_empty() || k == 0 {
            return vec![];
        }

        let mut selected: Vec<(SearchResult, Vec<f32>)> = Vec::with_capacity(k);
        let mut remaining: HashSet<usize> = (0..candidates.len()).collect();

        // Select the first document (most similar to query)
        let first_idx = 0;
        selected.push(candidates[first_idx].clone());
        remaining.remove(&first_idx);

        // Select remaining documents using MMR
        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx = None;
            let mut best_score = f32::NEG_INFINITY;

            for &idx in &remaining {
                let (_, embedding) = &candidates[idx];

                // Relevance to query
                let query_sim = self.cosine_similarity(query_embedding, embedding);

                // Maximum similarity to already selected documents
                let max_selected_sim = selected
                    .iter()
                    .map(|(_, sel_emb)| self.cosine_similarity(embedding, sel_emb))
                    .fold(f32::NEG_INFINITY, f32::max);

                // MMR score
                let mmr_score = lambda_mult * query_sim - (1.0 - lambda_mult) * max_selected_sim;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                selected.push(candidates[idx].clone());
                remaining.remove(&idx);
            } else {
                break;
            }
        }

        selected
    }

    /// Compute cosine similarity between two vectors.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Collection statistics for the vector store.
#[derive(Debug, Clone)]
pub struct CollectionStats {
    /// Collection name
    pub name: String,
    /// Number of documents
    pub document_count: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
}

/// Builder for creating metadata filters with a fluent API.
///
/// This provides a more ergonomic way to build filters that follows
/// LangChain's patterns.
#[derive(Debug, Clone, Default)]
pub struct FilterBuilder {
    conditions: Vec<Filter>,
}

impl FilterBuilder {
    /// Create a new filter builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an equality condition.
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.conditions.push(Filter::eq(field, value));
        self
    }

    /// Add a not-equal condition.
    pub fn ne(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.conditions.push(Filter::ne(field, value));
        self
    }

    /// Add a greater-than condition.
    pub fn gt(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.conditions.push(Filter::gt(field, value));
        self
    }

    /// Add a greater-than-or-equal condition.
    pub fn gte(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.conditions.push(Filter::gte(field, value));
        self
    }

    /// Add a less-than condition.
    pub fn lt(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.conditions.push(Filter::lt(field, value));
        self
    }

    /// Add a less-than-or-equal condition.
    pub fn lte(mut self, field: impl Into<String>, value: impl Into<Value>) -> Self {
        self.conditions.push(Filter::lte(field, value));
        self
    }

    /// Add an "in" condition.
    pub fn is_in(mut self, field: impl Into<String>, values: Vec<Value>) -> Self {
        self.conditions.push(Filter::is_in(field, values));
        self
    }

    /// Build the filter as an AND of all conditions.
    pub fn build(self) -> Option<Filter> {
        match self.conditions.len() {
            0 => None,
            1 => Some(self.conditions.into_iter().next().expect("length checked above")),
            _ => Some(Filter::and(self.conditions)),
        }
    }

    /// Build the filter as an OR of all conditions.
    pub fn build_or(self) -> Option<Filter> {
        match self.conditions.len() {
            0 => None,
            1 => Some(self.conditions.into_iter().next().expect("length checked above")),
            _ => Some(Filter::or(self.conditions)),
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    fn normalize_vector(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Hello, world!")
            .with_metadata(json!({"source": "test.txt"}))
            .with_id("doc-1");

        assert_eq!(doc.page_content, "Hello, world!");
        assert_eq!(doc.id, "doc-1");
        assert_eq!(doc.get_metadata("source"), Some(&json!("test.txt")));
    }

    #[test]
    fn test_document_add_metadata() {
        let mut doc = Document::new("Test content");
        doc.add_metadata("key1", "value1");
        doc.add_metadata("key2", 42);

        assert_eq!(doc.get_metadata("key1"), Some(&json!("value1")));
        assert_eq!(doc.get_metadata("key2"), Some(&json!(42)));
    }

    #[test]
    fn test_vector_store_creation() {
        let config = NeedleVectorStoreConfig::new("test", 128);
        let store = NeedleVectorStore::new(config).unwrap();

        assert_eq!(store.collection_name(), "test");
        assert_eq!(store.embedding_dimension(), 128);
        assert!(store.is_empty());
    }

    #[test]
    fn test_add_texts() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let texts = vec![
            "First document".to_string(),
            "Second document".to_string(),
        ];
        let embeddings = vec![random_vector(32), random_vector(32)];
        let metadatas = vec![json!({"index": 0}), json!({"index": 1})];

        let ids = store
            .add_texts(&texts, &embeddings, Some(&metadatas), None)
            .unwrap();

        assert_eq!(ids.len(), 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_add_documents() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let docs = vec![
            Document::new("First document").with_metadata(json!({"source": "a.txt"})),
            Document::new("Second document").with_metadata(json!({"source": "b.txt"})),
        ];
        let embeddings = vec![random_vector(32), random_vector(32)];

        let ids = store.add_documents(&docs, &embeddings).unwrap();

        assert_eq!(ids.len(), 2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_similarity_search() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        // Add some documents
        let mut embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(32)).collect();
        for emb in &mut embeddings {
            normalize_vector(emb);
        }

        let texts: Vec<String> = (0..10).map(|i| format!("Document {}", i)).collect();
        let metadatas: Vec<Value> = (0..10).map(|i| json!({"index": i})).collect();

        store
            .add_texts(&texts, &embeddings, Some(&metadatas), None)
            .unwrap();

        // Search
        let query = embeddings[0].clone();
        let results = store.similarity_search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be the most similar (same as query)
        assert!(results[0].1 > 0.99, "First result should have high score");
    }

    #[test]
    fn test_similarity_search_with_filter() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..20).map(|_| random_vector(32)).collect();
        let texts: Vec<String> = (0..20).map(|i| format!("Document {}", i)).collect();
        let metadatas: Vec<Value> = (0..20)
            .map(|i| {
                json!({
                    "index": i,
                    "category": if i % 2 == 0 { "even" } else { "odd" }
                })
            })
            .collect();

        store
            .add_texts(&texts, &embeddings, Some(&metadatas), None)
            .unwrap();

        // Search with filter
        let query = random_vector(32);
        let filter = Filter::eq("category", "even");
        let results = store
            .similarity_search_with_filter(&query, 5, Some(&filter))
            .unwrap();

        assert!(!results.is_empty());
        for (doc, _) in &results {
            assert_eq!(doc.get_metadata("category"), Some(&json!("even")));
        }
    }

    #[test]
    fn test_similarity_search_with_relevance_scores() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(32)).collect();
        let texts: Vec<String> = (0..10).map(|i| format!("Document {}", i)).collect();

        store.add_texts(&texts, &embeddings, None, None).unwrap();

        let query = random_vector(32);

        // Without threshold
        let results = store
            .similarity_search_with_relevance_scores(&query, 5, None)
            .unwrap();
        assert_eq!(results.len(), 5);

        // With threshold (may filter some results)
        let high_threshold_results = store
            .similarity_search_with_relevance_scores(&query, 5, Some(0.9))
            .unwrap();
        assert!(high_threshold_results.len() <= 5);
    }

    #[test]
    fn test_mmr_search() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        // Create embeddings with some diversity
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        for i in 0..20 {
            let mut v = random_vector(32);
            // Make some vectors similar to each other
            if i > 0 && i % 5 != 0 {
                for j in 0..32 {
                    v[j] = embeddings[i - 1][j] * 0.9 + v[j] * 0.1;
                }
            }
            normalize_vector(&mut v);
            embeddings.push(v);
        }

        let texts: Vec<String> = (0..20).map(|i| format!("Document {}", i)).collect();
        store.add_texts(&texts, &embeddings, None, None).unwrap();

        // MMR search should return diverse results
        let query = embeddings[0].clone();
        let results = store
            .max_marginal_relevance_search(&query, 5, 10, 0.5)
            .unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_delete() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings = vec![random_vector(32), random_vector(32)];
        let texts = vec!["Doc 1".to_string(), "Doc 2".to_string()];
        let ids = vec!["id1".to_string(), "id2".to_string()];

        store
            .add_texts(&texts, &embeddings, None, Some(&ids))
            .unwrap();
        assert_eq!(store.len(), 2);

        let deleted = store.delete(&["id1".to_string()]).unwrap();
        assert!(deleted);
        assert_eq!(store.len(), 1);

        // id1 should be gone
        assert!(store.get("id1").is_none());
        // id2 should still exist
        assert!(store.get("id2").is_some());
    }

    #[test]
    fn test_get() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embedding = random_vector(32);
        let doc = Document::new("Test content")
            .with_id("test-id")
            .with_metadata(json!({"key": "value"}));

        store.add_documents(std::slice::from_ref(&doc), &[embedding]).unwrap();

        let retrieved = store.get("test-id").unwrap();
        assert_eq!(retrieved.id, "test-id");
        assert_eq!(retrieved.get_metadata("key"), Some(&json!("value")));
    }

    #[test]
    fn test_batch_similarity_search() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..50).map(|_| random_vector(32)).collect();
        let texts: Vec<String> = (0..50).map(|i| format!("Document {}", i)).collect();

        store.add_texts(&texts, &embeddings, None, None).unwrap();

        let queries = vec![random_vector(32), random_vector(32), random_vector(32)];
        let results = store.batch_similarity_search(&queries, 5).unwrap();

        assert_eq!(results.len(), 3);
        for result_set in &results {
            assert_eq!(result_set.len(), 5);
        }
    }

    #[test]
    fn test_filter_builder() {
        let filter = FilterBuilder::new()
            .eq("category", "books")
            .gt("price", 10)
            .lte("price", 50)
            .build();

        assert!(filter.is_some());

        let metadata = json!({
            "category": "books",
            "price": 30
        });

        assert!(filter.unwrap().matches(Some(&metadata)));
    }

    #[test]
    fn test_filter_builder_or() {
        let filter = FilterBuilder::new()
            .eq("status", "active")
            .eq("status", "pending")
            .build_or();

        assert!(filter.is_some());

        let active = json!({"status": "active"});
        let pending = json!({"status": "pending"});
        let completed = json!({"status": "completed"});

        let f = filter.unwrap();
        assert!(f.matches(Some(&active)));
        assert!(f.matches(Some(&pending)));
        assert!(!f.matches(Some(&completed)));
    }

    #[test]
    fn test_relevance_score_functions() {
        // Cosine
        let cosine = RelevanceScoreFunction::Cosine;
        assert!((cosine.to_relevance_score(0.0) - 1.0).abs() < 0.001);
        assert!((cosine.to_relevance_score(1.0) - 0.0).abs() < 0.001);
        assert!((cosine.to_relevance_score(0.5) - 0.5).abs() < 0.001);

        // Euclidean
        let euclidean = RelevanceScoreFunction::Euclidean;
        assert!((euclidean.to_relevance_score(0.0) - 1.0).abs() < 0.001);
        assert!((euclidean.to_relevance_score(1.0) - 0.5).abs() < 0.001);

        // DotProduct
        let dot = RelevanceScoreFunction::DotProduct;
        assert!((dot.to_relevance_score(1.0) - 1.0).abs() < 0.001);
        assert!((dot.to_relevance_score(-1.0) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_stats() {
        let config = NeedleVectorStoreConfig::new("test", 64);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(64)).collect();
        let texts: Vec<String> = (0..10).map(|i| format!("Document {}", i)).collect();

        store.add_texts(&texts, &embeddings, None, None).unwrap();

        let stats = store.stats();
        assert_eq!(stats.name, "test");
        assert_eq!(stats.document_count, 10);
        assert_eq!(stats.dimensions, 64);
        assert!(stats.total_memory_bytes > 0);
    }

    #[test]
    fn test_compact() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(32)).collect();
        let texts: Vec<String> = (0..10).map(|i| format!("Document {}", i)).collect();
        let ids: Vec<String> = (0..10).map(|i| format!("id{}", i)).collect();

        store
            .add_texts(&texts, &embeddings, None, Some(&ids))
            .unwrap();

        // Delete some documents
        store.delete(&["id0".to_string(), "id1".to_string(), "id2".to_string()]).unwrap();

        // Compact
        let removed = store.compact().unwrap();
        assert_eq!(removed, 3);
        assert_eq!(store.len(), 7);
    }

    #[test]
    fn test_serialization() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config.clone()).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..5).map(|_| random_vector(32)).collect();
        let texts: Vec<String> = (0..5).map(|i| format!("Document {}", i)).collect();

        store.add_texts(&texts, &embeddings, None, None).unwrap();

        // Serialize
        let bytes = store.to_bytes().unwrap();

        // Deserialize
        let restored = NeedleVectorStore::from_bytes(&bytes, config).unwrap();

        assert_eq!(store.len(), restored.len());
        assert_eq!(store.embedding_dimension(), restored.embedding_dimension());
    }

    #[tokio::test]
    async fn test_async_methods() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(32)).collect();
        let texts: Vec<String> = (0..10).map(|i| format!("Document {}", i)).collect();

        // Async add
        let ids = store.aadd_texts(&texts, &embeddings, None, None).await.unwrap();
        assert_eq!(ids.len(), 10);

        // Async search
        let query = random_vector(32);
        let results = store.asimilarity_search(&query, 5).await.unwrap();
        assert_eq!(results.len(), 5);

        // Async delete
        let deleted = store.adelete(&[ids[0].clone()]).await.unwrap();
        assert!(deleted);
    }

    #[test]
    fn test_empty_store_search() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let query = random_vector(32);
        let results = store.similarity_search(&query, 5).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_mmr_empty_store() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let query = random_vector(32);
        let results = store
            .max_marginal_relevance_search(&query, 5, 10, 0.5)
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_mismatched_dimensions() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let texts = vec!["Test".to_string()];
        let wrong_dim_embeddings = vec![random_vector(64)]; // Wrong dimension

        let result = store.add_texts(&texts, &wrong_dim_embeddings, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_counts() {
        let config = NeedleVectorStoreConfig::new("test", 32);
        let store = NeedleVectorStore::new(config).unwrap();

        let texts = vec!["Test 1".to_string(), "Test 2".to_string()];
        let embeddings = vec![random_vector(32)]; // Only one embedding

        let result = store.add_texts(&texts, &embeddings, None, None);
        assert!(result.is_err());
    }
}
