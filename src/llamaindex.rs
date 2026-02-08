//! LlamaIndex Integration
//!
//! Compatible interface for LlamaIndex and similar frameworks.
//!
//! This module provides a VectorStoreIndex-like interface that follows LlamaIndex's
//! patterns and naming conventions, making it easy to integrate Needle with
//! LlamaIndex Python applications via FFI or as a drop-in replacement.
//!
//! # Features
//!
//! - LlamaIndex-compatible `TextNode` abstraction
//! - `NeedleVectorStoreIndex` implementing VectorStoreIndex-like interface
//! - Conversation memory backends for chat applications
//! - Document chunking with overlap support
//! - Retrieval query engine patterns
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::llamaindex::{TextNode, NeedleVectorStoreIndex, NeedleIndexConfig};
//! use serde_json::json;
//!
//! fn main() -> needle::Result<()> {
//!     // Create a vector store index
//!     let config = NeedleIndexConfig::new("documents", 384);
//!     let index = NeedleVectorStoreIndex::new(config)?;
//!
//!     // Add nodes with embeddings
//!     let nodes = vec![
//!         TextNode::new("Machine learning is a subset of AI.")
//!             .with_metadata("source", "intro.txt"),
//!         TextNode::new("Deep learning uses neural networks.")
//!             .with_metadata("source", "deep.txt"),
//!     ];
//!
//!     let embeddings: Vec<Vec<f32>> = vec![vec![0.1; 384], vec![0.2; 384]];
//!     index.insert_nodes(&nodes, &embeddings)?;
//!
//!     // Query the index
//!     let query_embedding = vec![0.15; 384];
//!     let results = index.query(&query_embedding, 5)?;
//!
//!     for result in results {
//!         println!("Score: {:.4}, Text: {}", result.score, result.node.text);
//!     }
//!
//!     Ok(())
//! }
//! ```

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

// ============================================================================
// Text Node (LlamaIndex BaseNode equivalent)
// ============================================================================

/// A text node abstraction compatible with LlamaIndex's BaseNode/TextNode class.
///
/// Nodes are the atomic unit of data in LlamaIndex. They contain text content,
/// embeddings, and relationships to other nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextNode {
    /// Unique identifier for the node
    pub id: String,
    /// The text content of the node
    pub text: String,
    /// Metadata associated with the node
    pub metadata: HashMap<String, Value>,
    /// Relationships to other nodes
    pub relationships: HashMap<NodeRelationship, RelatedNode>,
    /// Start character position in source document
    pub start_char_idx: Option<usize>,
    /// End character position in source document
    pub end_char_idx: Option<usize>,
    /// Hash of the text content for deduplication
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

/// Types of relationships between nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRelationship {
    /// Source document this node came from
    Source,
    /// Previous node in sequence
    Previous,
    /// Next node in sequence
    Next,
    /// Parent node in hierarchy
    Parent,
    /// Child nodes in hierarchy
    Child,
}

/// A related node reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedNode {
    /// ID of the related node
    pub node_id: String,
    /// Optional metadata about the relationship
    pub metadata: Option<Value>,
}

impl TextNode {
    /// Create a new text node with the given content
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        let hash = Self::compute_hash(&text);
        Self {
            id: Uuid::new_v4().to_string(),
            text,
            metadata: HashMap::new(),
            relationships: HashMap::new(),
            start_char_idx: None,
            end_char_idx: None,
            hash: Some(hash),
        }
    }

    /// Create a node with a specific ID
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Add metadata to the node
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set character indices
    pub fn with_char_indices(mut self, start: usize, end: usize) -> Self {
        self.start_char_idx = Some(start);
        self.end_char_idx = Some(end);
        self
    }

    /// Add a relationship to another node
    pub fn with_relationship(mut self, rel_type: NodeRelationship, node_id: impl Into<String>) -> Self {
        self.relationships.insert(
            rel_type,
            RelatedNode {
                node_id: node_id.into(),
                metadata: None,
            },
        );
        self
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Get the source document ID if available
    pub fn source_node_id(&self) -> Option<&str> {
        self.relationships
            .get(&NodeRelationship::Source)
            .map(|r| r.node_id.as_str())
    }

    /// Get the previous node ID if available
    pub fn prev_node_id(&self) -> Option<&str> {
        self.relationships
            .get(&NodeRelationship::Previous)
            .map(|r| r.node_id.as_str())
    }

    /// Get the next node ID if available
    pub fn next_node_id(&self) -> Option<&str> {
        self.relationships
            .get(&NodeRelationship::Next)
            .map(|r| r.node_id.as_str())
    }

    /// Compute a simple hash of the text content
    fn compute_hash(text: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Convert to JSON metadata for storage
    pub fn to_metadata(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("_text".to_string(), json!(self.text));
        map.insert("_node_id".to_string(), json!(self.id));
        
        if let Some(hash) = &self.hash {
            map.insert("_hash".to_string(), json!(hash));
        }
        if let Some(start) = self.start_char_idx {
            map.insert("_start_char_idx".to_string(), json!(start));
        }
        if let Some(end) = self.end_char_idx {
            map.insert("_end_char_idx".to_string(), json!(end));
        }
        
        // Add user metadata
        for (k, v) in &self.metadata {
            map.insert(k.clone(), v.clone());
        }
        
        // Add relationships
        if !self.relationships.is_empty() {
            let rels: HashMap<String, String> = self
                .relationships
                .iter()
                .map(|(k, v)| (format!("{:?}", k), v.node_id.clone()))
                .collect();
            map.insert("_relationships".to_string(), json!(rels));
        }
        
        Value::Object(map)
    }

    /// Create a TextNode from stored metadata
    pub fn from_metadata(id: &str, metadata: &Value) -> Option<Self> {
        let map = metadata.as_object()?;
        let text = map.get("_text")?.as_str()?.to_string();
        
        let mut node = Self::new(text).with_id(id);
        
        if let Some(hash) = map.get("_hash").and_then(|v| v.as_str()) {
            node.hash = Some(hash.to_string());
        }
        if let Some(start) = map.get("_start_char_idx").and_then(|v| v.as_u64()) {
            node.start_char_idx = Some(start as usize);
        }
        if let Some(end) = map.get("_end_char_idx").and_then(|v| v.as_u64()) {
            node.end_char_idx = Some(end as usize);
        }
        
        // Extract user metadata (non-underscore prefixed keys)
        for (k, v) in map {
            if !k.starts_with('_') {
                node.metadata.insert(k.clone(), v.clone());
            }
        }
        
        Some(node)
    }
}

// ============================================================================
// Query Result
// ============================================================================

/// Result from a vector store query
#[derive(Debug, Clone)]
pub struct NodeWithScore {
    /// The retrieved node
    pub node: TextNode,
    /// Similarity score (higher is better)
    pub score: f32,
}

// ============================================================================
// Index Configuration
// ============================================================================

/// Configuration for NeedleVectorStoreIndex
#[derive(Debug, Clone)]
pub struct NeedleIndexConfig {
    /// Name of the collection
    pub collection_name: String,
    /// Dimension of the embedding vectors
    pub embed_dim: usize,
    /// Distance function to use
    pub distance_function: DistanceFunction,
    /// Whether to store text content in metadata
    pub store_text: bool,
}

impl NeedleIndexConfig {
    /// Create a new configuration with required parameters
    pub fn new(collection_name: impl Into<String>, embed_dim: usize) -> Self {
        Self {
            collection_name: collection_name.into(),
            embed_dim,
            distance_function: DistanceFunction::Cosine,
            store_text: true,
        }
    }

    /// Set the distance function
    #[must_use]
    pub fn with_distance_function(mut self, distance: DistanceFunction) -> Self {
        self.distance_function = distance;
        self
    }

    /// Set whether to store text content
    #[must_use]
    pub fn with_store_text(mut self, store: bool) -> Self {
        self.store_text = store;
        self
    }
}

// ============================================================================
// Vector Store Index
// ============================================================================

/// A LlamaIndex-compatible vector store index backed by Needle.
///
/// This provides the same interface as LlamaIndex's VectorStoreIndex class,
/// allowing easy integration with LlamaIndex-based applications.
pub struct NeedleVectorStoreIndex {
    collection: Arc<RwLock<Collection>>,
    config: NeedleIndexConfig,
}

impl NeedleVectorStoreIndex {
    /// Create a new vector store index with the given configuration
    pub fn new(config: NeedleIndexConfig) -> Result<Self> {
        let collection_config = CollectionConfig::new(&config.collection_name, config.embed_dim)
            .with_distance(config.distance_function);
        let collection = Collection::new(collection_config);
        
        Ok(Self {
            collection: Arc::new(RwLock::new(collection)),
            config,
        })
    }

    /// Create an index from an existing collection
    pub fn from_collection(collection: Collection, config: NeedleIndexConfig) -> Self {
        Self {
            collection: Arc::new(RwLock::new(collection)),
            config,
        }
    }

    /// Get the number of nodes in the index
    pub fn len(&self) -> usize {
        self.collection.read().len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.collection.read().is_empty()
    }

    // =========================================================================
    // Node Operations
    // =========================================================================

    /// Insert nodes with their embeddings
    pub fn insert_nodes(&self, nodes: &[TextNode], embeddings: &[Vec<f32>]) -> Result<Vec<String>> {
        if nodes.len() != embeddings.len() {
            return Err(NeedleError::InvalidInput(
                "Number of nodes must match number of embeddings".to_string(),
            ));
        }

        let mut collection = self.collection.write();
        let mut ids = Vec::with_capacity(nodes.len());

        for (node, embedding) in nodes.iter().zip(embeddings.iter()) {
            let metadata = if self.config.store_text {
                Some(node.to_metadata())
            } else {
                None
            };
            
            collection.insert(&node.id, embedding, metadata)?;
            ids.push(node.id.clone());
        }

        Ok(ids)
    }

    /// Delete nodes by IDs
    pub fn delete_nodes(&self, node_ids: &[String]) -> Result<usize> {
        let mut collection = self.collection.write();
        let mut deleted = 0;
        
        for id in node_ids {
            if collection.delete(id)? {
                deleted += 1;
            }
        }
        
        Ok(deleted)
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: &str) -> Option<TextNode> {
        let collection = self.collection.read();
        let (_, metadata) = collection.get(node_id)?;
        metadata.and_then(|m| TextNode::from_metadata(node_id, m))
    }

    // =========================================================================
    // Query Operations
    // =========================================================================

    /// Query the index with an embedding vector
    pub fn query(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<NodeWithScore>> {
        self.query_with_filter(query_embedding, top_k, None)
    }

    /// Query the index with a filter
    pub fn query_with_filter(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<NodeWithScore>> {
        let collection = self.collection.read();
        
        let results = if let Some(f) = filter {
            collection.search_with_filter(query_embedding, top_k, f)?
        } else {
            collection.search(query_embedding, top_k)?
        };

        let nodes: Vec<NodeWithScore> = results
            .into_iter()
            .filter_map(|result| {
                let node = result
                    .metadata
                    .as_ref()
                    .and_then(|m| TextNode::from_metadata(&result.id, m))?;
                let score = self.distance_to_score(result.distance);
                Some(NodeWithScore { node, score })
            })
            .collect();

        Ok(nodes)
    }

    /// Retrieve nodes by IDs
    pub fn retrieve(&self, node_ids: &[String]) -> Vec<TextNode> {
        let collection = self.collection.read();
        node_ids
            .iter()
            .filter_map(|id| {
                let (_, metadata) = collection.get(id)?;
                metadata.and_then(|m| TextNode::from_metadata(id, m))
            })
            .collect()
    }

    // =========================================================================
    // Utility Methods
    // =========================================================================

    /// Convert distance to similarity score
    fn distance_to_score(&self, distance: f32) -> f32 {
        match self.config.distance_function {
            DistanceFunction::Cosine | DistanceFunction::CosineNormalized => 1.0 - distance,
            DistanceFunction::Euclidean => 1.0 / (1.0 + distance),
            DistanceFunction::DotProduct => (distance + 1.0) / 2.0,
            DistanceFunction::Manhattan => 1.0 / (1.0 + distance),
        }
    }

    /// Serialize the index to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.collection.read().to_bytes()
    }

    /// Deserialize an index from bytes
    pub fn from_bytes(bytes: &[u8], config: NeedleIndexConfig) -> Result<Self> {
        let collection = Collection::from_bytes(bytes)?;
        Ok(Self::from_collection(collection, config))
    }
}

// ============================================================================
// Document Chunking
// ============================================================================

/// Configuration for document chunking
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Target chunk size in characters
    pub chunk_size: usize,
    /// Overlap between chunks in characters
    pub chunk_overlap: usize,
    /// Separator to split on (default: paragraph)
    pub separator: String,
    /// Include metadata from parent document
    pub include_metadata: bool,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1024,
            chunk_overlap: 200,
            separator: "\n\n".to_string(),
            include_metadata: true,
        }
    }
}

impl ChunkConfig {
    /// Create a new chunk configuration
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            ..Default::default()
        }
    }

    /// Set the separator
    #[must_use]
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }
}

/// Document chunker for splitting documents into nodes
pub struct DocumentChunker {
    config: ChunkConfig,
}

impl DocumentChunker {
    /// Create a new document chunker
    pub fn new(config: ChunkConfig) -> Self {
        Self { config }
    }

    /// Chunk a document into text nodes
    pub fn chunk_document(
        &self,
        document_id: &str,
        text: &str,
        metadata: Option<HashMap<String, Value>>,
    ) -> Vec<TextNode> {
        let chunks = self.split_text(text);
        let mut nodes = Vec::with_capacity(chunks.len());
        let mut prev_node_id: Option<String> = None;

        for (i, (chunk_text, start_idx, end_idx)) in chunks.into_iter().enumerate() {
            let mut node = TextNode::new(chunk_text)
                .with_char_indices(start_idx, end_idx)
                .with_relationship(NodeRelationship::Source, document_id)
                .with_metadata("chunk_index", i as i64);

            // Add link to previous node
            if let Some(prev_id) = &prev_node_id {
                node = node.with_relationship(NodeRelationship::Previous, prev_id.clone());
            }

            // Add parent metadata
            if self.config.include_metadata {
                if let Some(ref meta) = metadata {
                    for (k, v) in meta {
                        node.metadata.insert(k.clone(), v.clone());
                    }
                }
            }

            prev_node_id = Some(node.id.clone());
            nodes.push(node);
        }

        // Add next relationships
        for i in 0..nodes.len().saturating_sub(1) {
            let next_id = nodes[i + 1].id.clone();
            nodes[i].relationships.insert(
                NodeRelationship::Next,
                RelatedNode {
                    node_id: next_id,
                    metadata: None,
                },
            );
        }

        nodes
    }

    /// Split text into chunks with overlap
    fn split_text(&self, text: &str) -> Vec<(String, usize, usize)> {
        let mut chunks = Vec::new();
        let paragraphs: Vec<&str> = text.split(&self.config.separator).collect();
        
        let mut current_chunk = String::new();
        let mut current_start = 0;
        let mut char_pos = 0;

        for para in paragraphs {
            let para_len = para.len();
            
            if current_chunk.len() + para_len > self.config.chunk_size && !current_chunk.is_empty() {
                // Save current chunk
                let chunk_end = char_pos;
                chunks.push((current_chunk.clone(), current_start, chunk_end));
                
                // Start new chunk with overlap
                let overlap_start = if current_chunk.len() > self.config.chunk_overlap {
                    current_chunk.len() - self.config.chunk_overlap
                } else {
                    0
                };
                current_chunk = current_chunk[overlap_start..].to_string();
                current_start = chunk_end - (current_chunk.len());
            }

            if !current_chunk.is_empty() {
                current_chunk.push_str(&self.config.separator);
            }
            current_chunk.push_str(para);
            char_pos += para_len + self.config.separator.len();
        }

        // Don't forget the last chunk
        if !current_chunk.is_empty() {
            chunks.push((current_chunk, current_start, text.len()));
        }

        chunks
    }
}

// ============================================================================
// Conversation Memory
// ============================================================================

/// A message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: MessageRole,
    /// Content of the message
    pub content: String,
    /// Timestamp
    pub timestamp: u64,
    /// Optional metadata
    pub metadata: Option<Value>,
}

/// Role in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    /// System prompt
    System,
    /// User message
    User,
    /// Assistant response
    Assistant,
    /// Function/tool call result
    Function,
}

impl ChatMessage {
    /// Create a new message
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metadata: None,
        }
    }

    /// Add metadata to the message
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Configuration for conversation memory
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum messages to keep
    pub max_messages: usize,
    /// Maximum tokens to keep (approximate)
    pub max_tokens: usize,
    /// Embedding dimension for semantic retrieval
    pub embed_dim: Option<usize>,
    /// Whether to include system messages in retrieval
    pub include_system: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_messages: 100,
            max_tokens: 4000,
            embed_dim: None,
            include_system: false,
        }
    }
}

/// Conversation memory for chat applications
///
/// Supports both sliding window and semantic retrieval memory patterns.
pub struct ConversationMemory {
    messages: RwLock<VecDeque<ChatMessage>>,
    vector_store: Option<Arc<NeedleVectorStoreIndex>>,
    config: MemoryConfig,
    message_count: AtomicU64,
}

impl ConversationMemory {
    /// Create a new conversation memory
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let vector_store = if let Some(dim) = config.embed_dim {
            let index_config = NeedleIndexConfig::new("_conversation_memory", dim);
            Some(Arc::new(NeedleVectorStoreIndex::new(index_config)?))
        } else {
            None
        };

        Ok(Self {
            messages: RwLock::new(VecDeque::new()),
            vector_store,
            config,
            message_count: AtomicU64::new(0),
        })
    }

    /// Add a message to memory
    pub fn add_message(&self, message: ChatMessage, embedding: Option<&[f32]>) -> Result<()> {
        let mut messages = self.messages.write();
        
        // Add to vector store if enabled
        if let (Some(store), Some(emb)) = (&self.vector_store, embedding) {
            let msg_id = self.message_count.fetch_add(1, Ordering::SeqCst);
            let node = TextNode::new(&message.content)
                .with_id(format!("msg_{}", msg_id))
                .with_metadata("role", format!("{:?}", message.role))
                .with_metadata("timestamp", message.timestamp as i64);
            
            store.insert_nodes(&[node], &[emb.to_vec()])?;
        }

        messages.push_back(message);

        // Enforce max messages
        while messages.len() > self.config.max_messages {
            messages.pop_front();
        }

        Ok(())
    }

    /// Add a user message
    pub fn add_user_message(&self, content: impl Into<String>, embedding: Option<&[f32]>) -> Result<()> {
        self.add_message(ChatMessage::new(MessageRole::User, content), embedding)
    }

    /// Add an assistant message
    pub fn add_assistant_message(&self, content: impl Into<String>, embedding: Option<&[f32]>) -> Result<()> {
        self.add_message(ChatMessage::new(MessageRole::Assistant, content), embedding)
    }

    /// Get recent messages (sliding window)
    pub fn get_recent(&self, count: usize) -> Vec<ChatMessage> {
        let messages = self.messages.read();
        messages
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Get all messages
    pub fn get_all(&self) -> Vec<ChatMessage> {
        self.messages.read().iter().cloned().collect()
    }

    /// Retrieve relevant messages by semantic similarity
    pub fn retrieve_relevant(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<ChatMessage>> {
        let store = self.vector_store.as_ref().ok_or_else(|| {
            NeedleError::InvalidInput("Semantic retrieval requires embed_dim in config".to_string())
        })?;

        let results = store.query(query_embedding, top_k)?;
        
        // Map back to chat messages
        let messages = self.messages.read();
        let relevant: Vec<ChatMessage> = results
            .into_iter()
            .filter_map(|r| {
                let timestamp = r.node.get_metadata("timestamp")?.as_i64()? as u64;
                messages.iter().find(|m| m.timestamp == timestamp).cloned()
            })
            .collect();

        Ok(relevant)
    }

    /// Get message count
    pub fn len(&self) -> usize {
        self.messages.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.messages.read().is_empty()
    }

    /// Clear all messages
    pub fn clear(&self) {
        self.messages.write().clear();
    }

    /// Get formatted history for LLM context
    pub fn get_formatted_history(&self, max_tokens: Option<usize>) -> String {
        let messages = self.messages.read();
        let max_tokens = max_tokens.unwrap_or(self.config.max_tokens);
        let mut result = String::new();
        let mut token_estimate = 0;

        for msg in messages.iter().rev() {
            let formatted = format!(
                "{}: {}\n",
                match msg.role {
                    MessageRole::System => "System",
                    MessageRole::User => "User",
                    MessageRole::Assistant => "Assistant",
                    MessageRole::Function => "Function",
                },
                msg.content
            );
            
            // Rough token estimate (4 chars per token)
            let msg_tokens = formatted.len() / 4;
            if token_estimate + msg_tokens > max_tokens {
                break;
            }
            
            result = formatted + &result;
            token_estimate += msg_tokens;
        }

        result
    }
}

// ============================================================================
// Retrieval Query Engine
// ============================================================================

/// A simple retrieval-augmented query engine
pub struct RetrieverQueryEngine {
    index: Arc<NeedleVectorStoreIndex>,
    top_k: usize,
    score_threshold: Option<f32>,
}

impl RetrieverQueryEngine {
    /// Create a new query engine
    pub fn new(index: Arc<NeedleVectorStoreIndex>, top_k: usize) -> Self {
        Self {
            index,
            top_k,
            score_threshold: None,
        }
    }

    /// Set minimum score threshold
    #[must_use]
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = Some(threshold);
        self
    }

    /// Retrieve relevant nodes
    pub fn retrieve(&self, query_embedding: &[f32]) -> Result<Vec<NodeWithScore>> {
        let mut results = self.index.query(query_embedding, self.top_k)?;
        
        if let Some(threshold) = self.score_threshold {
            results.retain(|r| r.score >= threshold);
        }
        
        Ok(results)
    }

    /// Retrieve and format context for LLM
    pub fn retrieve_context(&self, query_embedding: &[f32]) -> Result<String> {
        let nodes = self.retrieve(query_embedding)?;
        
        let context = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| format!("[{}] {}", i + 1, n.node.text))
            .collect::<Vec<_>>()
            .join("\n\n");
        
        Ok(context)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_text_node_creation() {
        let node = TextNode::new("Hello, world!")
            .with_id("node-1")
            .with_metadata("source", "test.txt")
            .with_char_indices(0, 13);

        assert_eq!(node.id, "node-1");
        assert_eq!(node.text, "Hello, world!");
        assert_eq!(node.get_metadata("source"), Some(&json!("test.txt")));
        assert_eq!(node.start_char_idx, Some(0));
        assert_eq!(node.end_char_idx, Some(13));
    }

    #[test]
    fn test_text_node_relationships() {
        let node = TextNode::new("Content")
            .with_relationship(NodeRelationship::Source, "doc-1")
            .with_relationship(NodeRelationship::Previous, "node-0")
            .with_relationship(NodeRelationship::Next, "node-2");

        assert_eq!(node.source_node_id(), Some("doc-1"));
        assert_eq!(node.prev_node_id(), Some("node-0"));
        assert_eq!(node.next_node_id(), Some("node-2"));
    }

    #[test]
    fn test_text_node_serialization() {
        let node = TextNode::new("Test content")
            .with_id("test-id")
            .with_metadata("key", "value");

        let metadata = node.to_metadata();
        let restored = TextNode::from_metadata("test-id", &metadata).unwrap();

        assert_eq!(restored.id, "test-id");
        assert_eq!(restored.text, "Test content");
        assert_eq!(restored.get_metadata("key"), Some(&json!("value")));
    }

    #[test]
    fn test_vector_store_index() {
        let config = NeedleIndexConfig::new("test", 32);
        let index = NeedleVectorStoreIndex::new(config).unwrap();

        let nodes: Vec<TextNode> = (0..10)
            .map(|i| TextNode::new(format!("Document {}", i)).with_metadata("index", i as i64))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(32)).collect();

        let ids = index.insert_nodes(&nodes, &embeddings).unwrap();
        assert_eq!(ids.len(), 10);
        assert_eq!(index.len(), 10);
    }

    #[test]
    fn test_query() {
        let config = NeedleIndexConfig::new("test", 32);
        let index = NeedleVectorStoreIndex::new(config).unwrap();

        let nodes: Vec<TextNode> = (0..10)
            .map(|i| TextNode::new(format!("Document {}", i)))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..10).map(|_| random_vector(32)).collect();

        index.insert_nodes(&nodes, &embeddings).unwrap();

        let results = index.query(&embeddings[0], 5).unwrap();
        assert_eq!(results.len(), 5);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn test_document_chunker() {
        let config = ChunkConfig::new(100, 20);
        let chunker = DocumentChunker::new(config);

        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\nFourth paragraph.";
        let nodes = chunker.chunk_document("doc-1", text, None);

        assert!(!nodes.is_empty());
        for node in &nodes {
            assert_eq!(node.source_node_id(), Some("doc-1"));
        }
    }

    #[test]
    fn test_chunker_relationships() {
        let config = ChunkConfig::new(50, 10);
        let chunker = DocumentChunker::new(config);

        let text = "A\n\nB\n\nC\n\nD";
        let nodes = chunker.chunk_document("doc", text, None);

        // Check that nodes are linked
        if nodes.len() > 1 {
            assert!(nodes[0].next_node_id().is_some());
            assert!(nodes[1].prev_node_id().is_some());
        }
    }

    #[test]
    fn test_conversation_memory() {
        let config = MemoryConfig::default();
        let memory = ConversationMemory::new(config).unwrap();

        memory.add_user_message("Hello!", None).unwrap();
        memory.add_assistant_message("Hi there!", None).unwrap();

        assert_eq!(memory.len(), 2);

        let recent = memory.get_recent(10);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].role, MessageRole::User);
        assert_eq!(recent[1].role, MessageRole::Assistant);
    }

    #[test]
    fn test_memory_max_messages() {
        let config = MemoryConfig {
            max_messages: 5,
            ..Default::default()
        };
        let memory = ConversationMemory::new(config).unwrap();

        for i in 0..10 {
            memory.add_user_message(format!("Message {}", i), None).unwrap();
        }

        assert_eq!(memory.len(), 5);
    }

    #[test]
    fn test_formatted_history() {
        let config = MemoryConfig::default();
        let memory = ConversationMemory::new(config).unwrap();

        memory.add_user_message("What is AI?", None).unwrap();
        memory.add_assistant_message("AI is artificial intelligence.", None).unwrap();

        let history = memory.get_formatted_history(None);
        assert!(history.contains("User:"));
        assert!(history.contains("Assistant:"));
    }

    #[test]
    fn test_retriever_query_engine() {
        let config = NeedleIndexConfig::new("test", 32);
        let index = Arc::new(NeedleVectorStoreIndex::new(config).unwrap());

        let nodes: Vec<TextNode> = (0..5)
            .map(|i| TextNode::new(format!("Content {}", i)))
            .collect();
        let embeddings: Vec<Vec<f32>> = (0..5).map(|_| random_vector(32)).collect();

        index.insert_nodes(&nodes, &embeddings).unwrap();

        let engine = RetrieverQueryEngine::new(index, 3);
        let results = engine.retrieve(&embeddings[0]).unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_retriever_context() {
        let config = NeedleIndexConfig::new("test", 32);
        let index = Arc::new(NeedleVectorStoreIndex::new(config).unwrap());

        let nodes = vec![
            TextNode::new("Machine learning basics"),
            TextNode::new("Deep learning fundamentals"),
        ];
        let embeddings: Vec<Vec<f32>> = (0..2).map(|_| random_vector(32)).collect();

        index.insert_nodes(&nodes, &embeddings).unwrap();

        let engine = RetrieverQueryEngine::new(index, 2);
        let context = engine.retrieve_context(&embeddings[0]).unwrap();

        assert!(context.contains("[1]"));
        assert!(context.contains("[2]"));
    }
}
