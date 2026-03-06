//! CrewAI Integration
//!
//! Provides a memory store adapter for CrewAI agents, enabling persistent
//! vector-based memory for multi-agent systems. Each memory is tagged with
//! the originating agent name for scoped recall.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::crewai::{CrewAiMemoryStore, CrewAiConfig};
//!
//! let config = CrewAiConfig::default();
//! let store = CrewAiMemoryStore::new("agent_memory", 384, config).unwrap();
//!
//! store.remember("m1", &vec![0.1; 384], "User likes coffee", "researcher").unwrap();
//!
//! let memories = store.recall(&vec![0.1; 384], 5).unwrap();
//! for mem in memories {
//!     println!("[{}] {:.4}: {}", mem.agent, mem.relevance, mem.content);
//! }
//! ```

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collection::Collection;
use crate::distance::DistanceFunction;
use crate::error::Result;
use crate::framework_common::{distance_to_score, FrameworkCollection};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the CrewAI memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrewAiConfig {
    /// Maximum number of memories to return from a recall query.
    pub max_memories: usize,
    /// Minimum relevance score (0.0 – 1.0) for recalled memories.
    pub relevance_threshold: f32,
    /// Distance function used for scoring.
    #[serde(default)]
    pub distance_function: DistanceFunction,
    /// Key in metadata where content is stored.
    #[serde(default = "default_content_key")]
    pub content_key: String,
    /// Key in metadata where agent name is stored.
    #[serde(default = "default_agent_key")]
    pub agent_key: String,
}

fn default_content_key() -> String {
    "_content".to_string()
}

fn default_agent_key() -> String {
    "_agent".to_string()
}

impl Default for CrewAiConfig {
    fn default() -> Self {
        Self {
            max_memories: 10,
            relevance_threshold: 0.0,
            distance_function: DistanceFunction::Cosine,
            content_key: default_content_key(),
            agent_key: default_agent_key(),
        }
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// A single memory recalled from the store.
#[derive(Debug, Clone, Serialize)]
pub struct Memory {
    /// Unique identifier.
    pub id: String,
    /// Text content of the memory.
    pub content: String,
    /// Name of the agent that created this memory.
    pub agent: String,
    /// Relevance score (0.0 – 1.0).
    pub relevance: f32,
    /// Optional extra metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

// ---------------------------------------------------------------------------
// CrewAiMemoryStore
// ---------------------------------------------------------------------------

/// CrewAI memory store backed by Needle.
pub struct CrewAiMemoryStore {
    inner: FrameworkCollection,
    config: CrewAiConfig,
}

impl CrewAiMemoryStore {
    /// Create a new memory store.
    pub fn new(
        collection_name: &str,
        embedding_dimension: usize,
        config: CrewAiConfig,
    ) -> Result<Self> {
        let inner = FrameworkCollection::new(
            collection_name,
            embedding_dimension,
            config.distance_function,
        );
        Ok(Self { inner, config })
    }

    /// Create from an existing [`Collection`].
    pub fn from_collection(collection: Collection, config: CrewAiConfig) -> Self {
        Self {
            inner: FrameworkCollection::from_collection(collection),
            config,
        }
    }

    /// Store a memory with embedding.
    pub fn remember(
        &self,
        id: &str,
        embedding: &[f32],
        content: &str,
        agent: &str,
    ) -> Result<()> {
        let metadata = json!({
            self.config.content_key.clone(): content,
            self.config.agent_key.clone(): agent,
        });
        // Upsert semantics: remove existing then insert
        let _ = self.inner.write().delete(id);
        self.inner.write().insert(id, embedding, Some(metadata))?;
        Ok(())
    }

    /// Store a memory with embedding and additional metadata.
    pub fn remember_with_metadata(
        &self,
        id: &str,
        embedding: &[f32],
        content: &str,
        agent: &str,
        extra: Value,
    ) -> Result<()> {
        let mut metadata = if let Value::Object(map) = extra {
            Value::Object(map)
        } else {
            json!({})
        };
        if let Value::Object(ref mut map) = metadata {
            map.insert(
                self.config.content_key.clone(),
                Value::String(content.to_string()),
            );
            map.insert(
                self.config.agent_key.clone(),
                Value::String(agent.to_string()),
            );
        }
        let _ = self.inner.write().delete(id);
        self.inner.write().insert(id, embedding, Some(metadata))?;
        Ok(())
    }

    /// Recall relevant memories for a query vector.
    pub fn recall(&self, query: &[f32], top_k: usize) -> Result<Vec<Memory>> {
        let k = top_k.min(self.config.max_memories);
        let results = self.inner.search(query, k, None)?;

        let memories: Vec<Memory> = results
            .into_iter()
            .map(|r| self.result_to_memory(r))
            .filter(|m| m.relevance >= self.config.relevance_threshold)
            .collect();

        Ok(memories)
    }

    /// Forget (delete) a specific memory by ID.
    pub fn forget(&self, id: &str) -> Result<bool> {
        self.inner.write().delete(id)
    }

    /// Number of memories in the store.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn result_to_memory(&self, result: crate::collection::SearchResult) -> Memory {
        let meta = result.metadata.as_ref();

        let content = meta
            .and_then(|m| m.get(&self.config.content_key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let agent = meta
            .and_then(|m| m.get(&self.config.agent_key))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let relevance = distance_to_score(result.distance, self.config.distance_function);

        Memory {
            id: result.id,
            content,
            agent,
            relevance,
            metadata: result.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> CrewAiMemoryStore {
        CrewAiMemoryStore::new("test_crewai", 4, CrewAiConfig::default()).unwrap()
    }

    #[test]
    fn test_remember_and_recall() {
        let store = make_store();

        store
            .remember("m1", &[1.0, 0.0, 0.0, 0.0], "User likes coffee", "researcher")
            .unwrap();
        store
            .remember("m2", &[0.0, 1.0, 0.0, 0.0], "Project deadline is Friday", "planner")
            .unwrap();

        assert_eq!(store.len(), 2);

        let memories = store.recall(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert_eq!(memories.len(), 2);
        assert_eq!(memories[0].id, "m1");
        assert_eq!(memories[0].content, "User likes coffee");
        assert_eq!(memories[0].agent, "researcher");
        assert!(memories[0].relevance > 0.0);
    }

    #[test]
    fn test_relevance_threshold() {
        let config = CrewAiConfig {
            relevance_threshold: 0.99,
            ..Default::default()
        };
        let store = CrewAiMemoryStore::new("test_threshold", 4, config).unwrap();

        store
            .remember("m1", &[1.0, 0.0, 0.0, 0.0], "Close", "agent1")
            .unwrap();
        store
            .remember("m2", &[0.0, 1.0, 0.0, 0.0], "Far", "agent2")
            .unwrap();

        let memories = store.recall(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        // Only the very close match should survive the 0.99 threshold
        assert!(memories.len() <= 1);
    }

    #[test]
    fn test_forget() {
        let store = make_store();
        store
            .remember("m1", &[1.0; 4], "Remember me", "agent1")
            .unwrap();
        assert_eq!(store.len(), 1);

        store.forget("m1").unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn test_upsert_semantics() {
        let store = make_store();
        store
            .remember("m1", &[1.0; 4], "Version 1", "agent1")
            .unwrap();
        store
            .remember("m1", &[0.0; 4], "Version 2", "agent1")
            .unwrap();

        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_remember_with_metadata() {
        let store = make_store();
        store
            .remember_with_metadata(
                "m1",
                &[1.0; 4],
                "Content",
                "agent1",
                json!({"priority": "high"}),
            )
            .unwrap();

        let memories = store.recall(&[1.0; 4], 1).unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].content, "Content");
        assert_eq!(memories[0].agent, "agent1");
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
        let store = CrewAiMemoryStore::from_collection(collection, CrewAiConfig::default());
        assert!(store.is_empty());
    }
}
