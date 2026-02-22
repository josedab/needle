#![allow(clippy::unwrap_used)]
//! Agentic Memory Protocol
//!
//! MCP-optimized memory layer with three tiers: episodic (conversation turns),
//! semantic (extracted facts/knowledge), and procedural (tool-use patterns).
//! Supports auto-consolidation, importance scoring, and forgetting curves.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::agentic_memory_protocol::{
//!     AgentMemory, MemoryConfig, MemoryTier, MemoryEntry,
//! };
//!
//! let mut memory = AgentMemory::new(MemoryConfig::default());
//!
//! // Store an episodic memory (conversation turn)
//! memory.remember(MemoryEntry::episodic(
//!     "user asked about Rust ownership",
//!     &[0.1f32; 32],
//! )).unwrap();
//!
//! // Recall semantically similar memories
//! let recalled = memory.recall(&[0.1f32; 32], 5, None).unwrap();
//! assert!(!recalled.is_empty());
//!
//! // Consolidate: promote important episodic → semantic
//! let stats = memory.consolidate().unwrap();
//! ```

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

// ── Memory Tiers ─────────────────────────────────────────────────────────────

/// Memory tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    /// Short-lived conversation turns and observations.
    Episodic,
    /// Extracted facts and knowledge (promoted from episodic).
    Semantic,
    /// Tool-use patterns and learned procedures.
    Procedural,
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Episodic => write!(f, "episodic"),
            Self::Semantic => write!(f, "semantic"),
            Self::Procedural => write!(f, "procedural"),
        }
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Agent memory configuration.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Embedding dimensions.
    pub dimensions: usize,
    /// Default TTL for episodic memories.
    pub episodic_ttl: Option<Duration>,
    /// Importance threshold for promotion to semantic tier (0.0–1.0).
    pub promotion_threshold: f32,
    /// Maximum memories per tier.
    pub max_per_tier: usize,
    /// Forgetting curve decay rate (higher = faster forgetting).
    pub decay_rate: f32,
    /// Session identifier for scoping.
    pub session_id: Option<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            dimensions: 32,
            episodic_ttl: Some(Duration::from_secs(3600)),
            promotion_threshold: 0.7,
            max_per_tier: 10_000,
            decay_rate: 0.01,
            session_id: None,
        }
    }
}

impl MemoryConfig {
    /// Set dimensions.
    #[must_use]
    pub fn with_dimensions(mut self, dim: usize) -> Self {
        self.dimensions = dim;
        self
    }

    /// Set session ID.
    #[must_use]
    pub fn with_session(mut self, session: impl Into<String>) -> Self {
        self.session_id = Some(session.into());
        self
    }

    /// Set promotion threshold.
    #[must_use]
    pub fn with_promotion_threshold(mut self, threshold: f32) -> Self {
        self.promotion_threshold = threshold;
        self
    }
}

// ── Memory Entry ─────────────────────────────────────────────────────────────

/// A single memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Memory content text.
    pub content: String,
    /// Embedding vector.
    pub embedding: Vec<f32>,
    /// Memory tier.
    pub tier: MemoryTier,
    /// Importance score (0.0–1.0).
    pub importance: f32,
    /// Access count (increases with recall).
    pub access_count: u32,
    /// Creation timestamp (epoch seconds).
    pub created_at: u64,
    /// Last accessed timestamp.
    pub last_accessed: u64,
    /// Optional metadata.
    pub metadata: Option<Value>,
    /// Session ID.
    pub session_id: Option<String>,
}

impl MemoryEntry {
    /// Create an episodic memory.
    pub fn episodic(content: &str, embedding: &[f32]) -> Self {
        let now = now_secs();
        Self {
            content: content.into(),
            embedding: embedding.to_vec(),
            tier: MemoryTier::Episodic,
            importance: 0.5,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            metadata: None,
            session_id: None,
        }
    }

    /// Create a semantic memory.
    pub fn semantic(content: &str, embedding: &[f32]) -> Self {
        let now = now_secs();
        Self {
            content: content.into(),
            embedding: embedding.to_vec(),
            tier: MemoryTier::Semantic,
            importance: 0.8,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            metadata: None,
            session_id: None,
        }
    }

    /// Create a procedural memory.
    pub fn procedural(content: &str, embedding: &[f32]) -> Self {
        let now = now_secs();
        Self {
            content: content.into(),
            embedding: embedding.to_vec(),
            tier: MemoryTier::Procedural,
            importance: 0.9,
            access_count: 0,
            created_at: now,
            last_accessed: now,
            metadata: None,
            session_id: None,
        }
    }

    /// Set importance score.
    #[must_use]
    pub fn with_importance(mut self, score: f32) -> Self {
        self.importance = score.clamp(0.0, 1.0);
        self
    }

    /// Set metadata.
    #[must_use]
    pub fn with_metadata(mut self, meta: Value) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Compute effective importance with forgetting curve.
    pub fn effective_importance(&self, decay_rate: f32) -> f32 {
        let now = now_secs();
        let age_hours = (now.saturating_sub(self.last_accessed)) as f32 / 3600.0;
        let recency_boost = (-decay_rate * age_hours).exp();
        let access_boost = (self.access_count as f32).ln_1p() * 0.1;
        (self.importance * recency_boost + access_boost).clamp(0.0, 1.0)
    }
}

// ── Recall Result ────────────────────────────────────────────────────────────

/// Result from a memory recall operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallResult {
    /// Memory ID.
    pub id: String,
    /// Memory content.
    pub content: String,
    /// Similarity distance.
    pub distance: f32,
    /// Effective importance.
    pub importance: f32,
    /// Memory tier.
    pub tier: MemoryTier,
    /// Access count.
    pub access_count: u32,
    /// Metadata.
    pub metadata: Option<Value>,
}

// ── Consolidation Stats ──────────────────────────────────────────────────────

/// Statistics from a consolidation pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Memories promoted from episodic to semantic.
    pub promoted: usize,
    /// Memories forgotten (expired or below threshold).
    pub forgotten: usize,
    /// Total memories scanned.
    pub scanned: usize,
}

// ── MCP Tool Definition ──────────────────────────────────────────────────────

/// Definition of an MCP tool exposed by the memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    /// Tool name.
    pub name: String,
    /// Tool description.
    pub description: String,
    /// Parameters: (name, type, required).
    pub parameters: Vec<(String, String, bool)>,
}

// ── Agent Memory ─────────────────────────────────────────────────────────────

/// MCP-compatible agent memory system.
pub struct AgentMemory {
    config: MemoryConfig,
    episodic: Collection,
    semantic: Collection,
    procedural: Collection,
    entries: HashMap<String, MemoryEntry>,
    next_id: u64,
}

impl AgentMemory {
    /// Create a new agent memory system.
    pub fn new(config: MemoryConfig) -> Self {
        let dim = config.dimensions;
        let episodic = Collection::new(
            CollectionConfig::new("__memory_episodic__", dim)
                .with_distance(DistanceFunction::Cosine),
        );
        let semantic = Collection::new(
            CollectionConfig::new("__memory_semantic__", dim)
                .with_distance(DistanceFunction::Cosine),
        );
        let procedural = Collection::new(
            CollectionConfig::new("__memory_procedural__", dim)
                .with_distance(DistanceFunction::Cosine),
        );
        Self {
            config,
            episodic,
            semantic,
            procedural,
            entries: HashMap::new(),
            next_id: 0,
        }
    }

    /// Store a memory.
    pub fn remember(&mut self, mut entry: MemoryEntry) -> Result<String> {
        if entry.embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: entry.embedding.len(),
            });
        }

        let id = format!("mem_{}", self.next_id);
        self.next_id += 1;

        entry.session_id = self.config.session_id.clone();

        let meta = serde_json::to_value(&entry).ok();
        let coll = self.tier_collection_mut(entry.tier);
        coll.insert(id.clone(), &entry.embedding, meta)?;
        self.entries.insert(id.clone(), entry);
        Ok(id)
    }

    /// Recall memories similar to a query embedding.
    pub fn recall(
        &mut self,
        query: &[f32],
        k: usize,
        tier_filter: Option<MemoryTier>,
    ) -> Result<Vec<RecallResult>> {
        let mut results = Vec::new();

        let tiers = match tier_filter {
            Some(t) => vec![t],
            None => vec![MemoryTier::Episodic, MemoryTier::Semantic, MemoryTier::Procedural],
        };

        for tier in tiers {
            let coll = self.tier_collection(tier);
            if let Ok(search_results) = coll.search(query, k) {
                for sr in search_results {
                    if let Some(entry) = self.entries.get_mut(&sr.id) {
                        entry.access_count += 1;
                        entry.last_accessed = now_secs();
                        let importance = entry.effective_importance(self.config.decay_rate);
                        results.push(RecallResult {
                            id: sr.id,
                            content: entry.content.clone(),
                            distance: sr.distance,
                            importance,
                            tier: entry.tier,
                            access_count: entry.access_count,
                            metadata: entry.metadata.clone(),
                        });
                    }
                }
            }
        }

        // Sort by combined score (low distance + high importance)
        results.sort_by(|a, b| {
            let score_a = a.distance - a.importance * 0.5;
            let score_b = b.distance - b.importance * 0.5;
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        Ok(results)
    }

    /// Consolidate memories: promote important episodic → semantic, forget expired.
    pub fn consolidate(&mut self) -> Result<ConsolidationStats> {
        let mut stats = ConsolidationStats::default();
        let now = now_secs();

        let entries: Vec<(String, MemoryEntry)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.tier == MemoryTier::Episodic)
            .map(|(id, e)| (id.clone(), e.clone()))
            .collect();

        stats.scanned = entries.len();

        for (id, entry) in entries {
            let eff_importance = entry.effective_importance(self.config.decay_rate);

            // Promote high-importance episodic → semantic
            if eff_importance >= self.config.promotion_threshold {
                if let Err(e) = self.episodic.delete(&id) {
                    tracing::warn!("Failed to delete episodic memory '{id}' during promotion: {e}");
                }
                let meta = serde_json::to_value(&entry).ok();
                let new_id = format!("mem_{}", self.next_id);
                self.next_id += 1;

                self.semantic
                    .insert(new_id.clone(), &entry.embedding, meta)?;
                let mut promoted = entry.clone();
                promoted.tier = MemoryTier::Semantic;
                self.entries.remove(&id);
                self.entries.insert(new_id, promoted);
                stats.promoted += 1;
                continue;
            }

            // Forget expired episodic memories
            if let Some(ttl) = self.config.episodic_ttl {
                if now.saturating_sub(entry.created_at) > ttl.as_secs() && eff_importance < 0.3
                {
                    if let Err(e) = self.episodic.delete(&id) {
                        tracing::warn!("Failed to delete expired episodic memory '{id}': {e}");
                    }
                    self.entries.remove(&id);
                    stats.forgotten += 1;
                }
            }
        }

        Ok(stats)
    }

    /// Forget a specific memory.
    pub fn forget(&mut self, id: &str) -> Result<bool> {
        if let Some(entry) = self.entries.remove(id) {
            let coll = self.tier_collection_mut(entry.tier);
            if let Err(e) = coll.delete(id) {
                tracing::warn!("Failed to delete memory '{id}' from collection: {e}");
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get memory count per tier.
    pub fn counts(&self) -> HashMap<MemoryTier, usize> {
        let mut counts = HashMap::new();
        counts.insert(MemoryTier::Episodic, self.episodic.len());
        counts.insert(MemoryTier::Semantic, self.semantic.len());
        counts.insert(MemoryTier::Procedural, self.procedural.len());
        counts
    }

    /// Total memory count.
    pub fn total(&self) -> usize {
        self.entries.len()
    }

    /// Export all memories as a serializable list (for persistence or cross-agent sharing).
    pub fn export(&self) -> Vec<MemoryEntry> {
        self.entries.values().cloned().collect()
    }

    /// Import memories from another agent or a previous session.
    pub fn import(&mut self, memories: Vec<MemoryEntry>) -> Result<usize> {
        let mut imported = 0;
        for entry in memories {
            if entry.embedding.len() != self.config.dimensions {
                continue; // skip dimension mismatch silently
            }
            self.remember(entry)?;
            imported += 1;
        }
        Ok(imported)
    }

    /// Recall memories scoped to a specific session ID.
    pub fn recall_session(
        &mut self,
        query: &[f32],
        k: usize,
        session_id: &str,
    ) -> Result<Vec<RecallResult>> {
        let mut all = self.recall(query, k * 3, None)?;
        all.retain(|r| {
            self.entries
                .get(&r.id)
                .and_then(|e| e.session_id.as_deref())
                .map_or(false, |sid| sid == session_id)
        });
        all.truncate(k);
        Ok(all)
    }

    /// Return MCP tool definitions for this memory system.
    pub fn mcp_tool_definitions() -> Vec<McpToolDef> {
        vec![
            McpToolDef {
                name: "memory_remember".into(),
                description: "Store a memory with content, tier (episodic/semantic/procedural), and importance score".into(),
                parameters: vec![
                    ("content".into(), "string".into(), true),
                    ("tier".into(), "string".into(), false),
                    ("importance".into(), "number".into(), false),
                ],
            },
            McpToolDef {
                name: "memory_recall".into(),
                description: "Recall memories similar to a query, optionally filtered by tier".into(),
                parameters: vec![
                    ("query".into(), "string".into(), true),
                    ("k".into(), "number".into(), false),
                    ("tier".into(), "string".into(), false),
                ],
            },
            McpToolDef {
                name: "memory_forget".into(),
                description: "Forget a specific memory by ID".into(),
                parameters: vec![("id".into(), "string".into(), true)],
            },
            McpToolDef {
                name: "memory_consolidate".into(),
                description: "Consolidate memories: promote important episodic to semantic, forget expired".into(),
                parameters: vec![],
            },
        ]
    }

    fn tier_collection(&self, tier: MemoryTier) -> &Collection {
        match tier {
            MemoryTier::Episodic => &self.episodic,
            MemoryTier::Semantic => &self.semantic,
            MemoryTier::Procedural => &self.procedural,
        }
    }

    fn tier_collection_mut(&mut self, tier: MemoryTier) -> &mut Collection {
        match tier {
            MemoryTier::Episodic => &mut self.episodic,
            MemoryTier::Semantic => &mut self.semantic,
            MemoryTier::Procedural => &mut self.procedural,
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Framework Adapters ───────────────────────────────────────────────────────

/// LangChain-compatible MemoryStore adapter.
///
/// Wraps `AgentMemory` to expose a `BaseMemory`-style interface matching
/// LangChain's memory abstractions (add_message, get_relevant, clear).
pub struct LangChainMemoryStore {
    inner: AgentMemory,
}

impl LangChainMemoryStore {
    /// Create a new LangChain memory store.
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            inner: AgentMemory::new(config),
        }
    }

    /// Add a chat message as an episodic memory.
    pub fn add_message(&mut self, role: &str, content: &str, embedding: &[f32]) -> Result<String> {
        let entry = MemoryEntry::episodic(content, embedding)
            .with_metadata(serde_json::json!({
                "role": role,
                "framework": "langchain",
            }));
        self.inner.remember(entry)
    }

    /// Get relevant memories for a query (LangChain `get_relevant_documents`).
    pub fn get_relevant(&mut self, query_embedding: &[f32], k: usize) -> Result<Vec<RecallResult>> {
        self.inner.recall(query_embedding, k, None)
    }

    /// Clear all memories (LangChain `clear`).
    pub fn clear(&mut self) {
        let ids: Vec<String> = self.inner.entries.keys().cloned().collect();
        for id in ids {
            if let Err(e) = self.inner.forget(&id) {
                tracing::warn!("Failed to forget memory '{id}' during clear: {e}");
            }
        }
    }

    /// Get conversation history formatted as role/content pairs.
    pub fn load_memory_variables(&self) -> Vec<(String, String)> {
        let mut messages: Vec<_> = self.inner.entries.values()
            .filter(|e| e.tier == MemoryTier::Episodic)
            .map(|e| {
                let role = e.metadata.as_ref()
                    .and_then(|m| m.get("role"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("user")
                    .to_string();
                (e.created_at, role, e.content.clone())
            })
            .collect();
        messages.sort_by_key(|(ts, _, _)| *ts);
        messages.into_iter().map(|(_, role, content)| (role, content)).collect()
    }

    /// Access the underlying memory.
    pub fn inner(&mut self) -> &mut AgentMemory {
        &mut self.inner
    }
}

/// LlamaIndex-compatible ChatMemoryBuffer adapter.
///
/// Wraps `AgentMemory` to match LlamaIndex's `ChatMemoryBuffer` interface
/// with sliding-window conversation memory and semantic retrieval.
pub struct LlamaIndexMemoryBuffer {
    inner: AgentMemory,
    /// Maximum number of recent messages to keep in the buffer.
    pub max_messages: usize,
    /// Conversation history (role, content, timestamp).
    history: Vec<(String, String, u64)>,
}

impl LlamaIndexMemoryBuffer {
    /// Create a new LlamaIndex memory buffer.
    pub fn new(config: MemoryConfig, max_messages: usize) -> Self {
        Self {
            inner: AgentMemory::new(config),
            max_messages,
            history: Vec::new(),
        }
    }

    /// Put a chat message into the buffer (LlamaIndex `put`).
    pub fn put(&mut self, role: &str, content: &str, embedding: &[f32]) -> Result<String> {
        let now = now_secs();
        self.history.push((role.to_string(), content.to_string(), now));

        // Trim history to window size
        if self.history.len() > self.max_messages {
            self.history.drain(..self.history.len() - self.max_messages);
        }

        let entry = MemoryEntry::episodic(content, embedding)
            .with_metadata(serde_json::json!({
                "role": role,
                "framework": "llamaindex",
                "turn_index": self.history.len() - 1,
            }));
        self.inner.remember(entry)
    }

    /// Get recent messages in the buffer window (LlamaIndex `get`).
    pub fn get(&self) -> Vec<(String, String)> {
        self.history.iter()
            .map(|(role, content, _)| (role.clone(), content.clone()))
            .collect()
    }

    /// Get semantically relevant context (LlamaIndex `get_all`).
    pub fn get_all(&mut self, query_embedding: &[f32], k: usize) -> Result<Vec<RecallResult>> {
        self.inner.recall(query_embedding, k, None)
    }

    /// Reset the buffer.
    pub fn reset(&mut self) {
        self.history.clear();
        let ids: Vec<String> = self.inner.entries.keys().cloned().collect();
        for id in ids {
            if let Err(e) = self.inner.forget(&id) {
                tracing::warn!("Failed to forget memory '{id}' during reset: {e}");
            }
        }
    }

    /// Number of messages in buffer.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Access the underlying memory.
    pub fn inner(&mut self) -> &mut AgentMemory {
        &mut self.inner
    }
}

// ── Ebbinghaus Forgetting Curves ─────────────────────────────────────────────

/// Ebbinghaus forgetting curve model for spaced repetition memory strength.
/// Models how memory retention decreases over time without reinforcement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EbbinghausCurve {
    /// Memory stability (higher = slower forgetting). Increases with repetition.
    pub stability: f64,
    /// Number of successful retrievals.
    pub repetitions: u32,
    /// Time of last retrieval (epoch seconds).
    pub last_retrieval: u64,
}

impl EbbinghausCurve {
    /// Create a new curve for a freshly created memory.
    pub fn new() -> Self {
        Self {
            stability: 1.0,
            repetitions: 0,
            last_retrieval: now_secs(),
        }
    }

    /// Compute retention probability (0.0 - 1.0) at the current time.
    pub fn retention(&self) -> f64 {
        let elapsed_hours =
            (now_secs().saturating_sub(self.last_retrieval)) as f64 / 3600.0;
        (-elapsed_hours / (self.stability * 24.0)).exp()
    }

    /// Record a retrieval, strengthening the memory.
    pub fn record_retrieval(&mut self) {
        self.repetitions += 1;
        self.last_retrieval = now_secs();
        // Each retrieval increases stability (spaced repetition effect)
        self.stability *= 1.0 + 0.5 * (self.repetitions as f64).ln_1p();
    }

    /// Get the optimal review time (when retention drops to 90%).
    pub fn optimal_review_hours(&self) -> f64 {
        let target_retention = 0.9_f64;
        -target_retention.ln() * self.stability * 24.0
    }
}

impl Default for EbbinghausCurve {
    fn default() -> Self {
        Self::new()
    }
}

// ── Importance Scoring ───────────────────────────────────────────────────────

/// Composite importance scorer using recency × frequency × relevance.
#[derive(Debug, Clone)]
pub struct ImportanceScorer {
    /// Weight for recency factor (0.0 - 1.0).
    pub recency_weight: f32,
    /// Weight for frequency factor (0.0 - 1.0).
    pub frequency_weight: f32,
    /// Weight for relevance factor (0.0 - 1.0).
    pub relevance_weight: f32,
    /// Decay half-life in hours for recency computation.
    pub recency_half_life_hours: f32,
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self {
            recency_weight: 0.4,
            frequency_weight: 0.3,
            relevance_weight: 0.3,
            recency_half_life_hours: 24.0,
        }
    }
}

impl ImportanceScorer {
    /// Score a memory entry. `relevance` is 1.0 - distance (semantic similarity).
    pub fn score(&self, entry: &MemoryEntry, relevance: f32) -> f32 {
        let age_hours =
            (now_secs().saturating_sub(entry.last_accessed)) as f32 / 3600.0;
        let recency = (-age_hours * (2.0f32.ln()) / self.recency_half_life_hours).exp();
        let frequency = (entry.access_count as f32 + 1.0).ln() / 5.0f32.ln();
        let relevance_clamped = relevance.clamp(0.0, 1.0);

        (self.recency_weight * recency
            + self.frequency_weight * frequency.min(1.0)
            + self.relevance_weight * relevance_clamped)
            .clamp(0.0, 1.0)
    }
}

// ── Conversation-Aware RAG ───────────────────────────────────────────────────

/// A conversation turn for context-aware retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// Speaker role (user, assistant, system).
    pub role: String,
    /// Content text.
    pub content: String,
    /// Embedding (optional, lazily computed).
    pub embedding: Option<Vec<f32>>,
    /// Timestamp.
    pub timestamp: u64,
}

/// Conversation-aware RAG context builder.
/// Combines conversation history with semantic memory for richer retrieval context.
pub struct ConversationRag {
    memory: AgentMemory,
    history: Vec<ConversationTurn>,
    max_context_turns: usize,
    scorer: ImportanceScorer,
}

impl ConversationRag {
    /// Create a new conversation-aware RAG system.
    pub fn new(config: MemoryConfig, max_context_turns: usize) -> Self {
        Self {
            memory: AgentMemory::new(config),
            history: Vec::new(),
            max_context_turns,
            scorer: ImportanceScorer::default(),
        }
    }

    /// Add a conversation turn.
    pub fn add_turn(&mut self, role: &str, content: &str, embedding: Option<&[f32]>) -> Result<()> {
        let turn = ConversationTurn {
            role: role.into(),
            content: content.into(),
            embedding: embedding.map(|e| e.to_vec()),
            timestamp: now_secs(),
        };

        if let Some(emb) = embedding {
            let entry = MemoryEntry::episodic(content, emb)
                .with_metadata(serde_json::json!({
                    "role": role,
                    "turn_index": self.history.len(),
                }));
            self.memory.remember(entry)?;
        }

        self.history.push(turn);
        Ok(())
    }

    /// Retrieve context for the next response, combining recent turns with semantic memories.
    pub fn build_context(
        &mut self,
        query_embedding: &[f32],
        k_memories: usize,
    ) -> Result<(Vec<ConversationTurn>, Vec<RecallResult>)> {
        let start = self.history.len().saturating_sub(self.max_context_turns);
        let recent_turns = self.history[start..].to_vec();
        let memories = self.memory.recall(query_embedding, k_memories, None)?;
        Ok((recent_turns, memories))
    }

    /// Store a knowledge fact as semantic memory for future RAG retrieval.
    pub fn add_knowledge(&mut self, content: &str, embedding: &[f32]) -> Result<String> {
        let entry = MemoryEntry::semantic(content, embedding);
        self.memory.remember(entry)
    }

    /// Get conversation history length.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get total memory count.
    pub fn memory_count(&self) -> usize {
        self.memory.total()
    }

    /// Access the underlying memory.
    pub fn memory(&mut self) -> &mut AgentMemory {
        &mut self.memory
    }

    /// Consolidate memories.
    pub fn consolidate(&mut self) -> Result<ConsolidationStats> {
        self.memory.consolidate()
    }
}

/// AutoGen-compatible memory provider adapter.
///
/// Wraps `AgentMemory` to match AutoGen's memory interfaces for
/// multi-agent systems with shared memory pools.
pub struct AutoGenMemoryProvider {
    inner: AgentMemory,
    agent_id: String,
}

impl AutoGenMemoryProvider {
    /// Create a new AutoGen memory provider for a specific agent.
    pub fn new(config: MemoryConfig, agent_id: impl Into<String>) -> Self {
        Self {
            inner: AgentMemory::new(config),
            agent_id: agent_id.into(),
        }
    }

    /// Store a memory scoped to this agent.
    pub fn add(&mut self, content: &str, embedding: &[f32]) -> Result<String> {
        let entry = MemoryEntry::episodic(content, embedding)
            .with_metadata(serde_json::json!({
                "agent_id": self.agent_id,
                "framework": "autogen",
            }));
        self.inner.remember(entry)
    }

    /// Query memories for this agent.
    pub fn query(&mut self, embedding: &[f32], k: usize) -> Result<Vec<RecallResult>> {
        let mut results = self.inner.recall(embedding, k * 2, None)?;
        results.retain(|r| {
            self.inner.entries.get(&r.id)
                .and_then(|e| e.metadata.as_ref())
                .and_then(|m| m.get("agent_id"))
                .and_then(|v| v.as_str())
                .map_or(false, |aid| aid == self.agent_id)
        });
        results.truncate(k);
        Ok(results)
    }

    /// Query memories from any agent (shared pool).
    pub fn query_shared(&mut self, embedding: &[f32], k: usize) -> Result<Vec<RecallResult>> {
        self.inner.recall(embedding, k, None)
    }

    /// Get this agent's ID.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Access the underlying memory.
    pub fn inner(&mut self) -> &mut AgentMemory {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emb(base: f32) -> Vec<f32> {
        (0..32).map(|i| base + i as f32 * 0.01).collect()
    }

    #[test]
    fn test_remember_and_recall() {
        let mut mem = AgentMemory::new(MemoryConfig::default());
        mem.remember(MemoryEntry::episodic("rust ownership", &emb(0.1))).unwrap();
        mem.remember(MemoryEntry::episodic("python GIL", &emb(0.9))).unwrap();

        let results = mem.recall(&emb(0.1), 5, None).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "rust ownership");
    }

    #[test]
    fn test_tier_filter() {
        let mut mem = AgentMemory::new(MemoryConfig::default());
        mem.remember(MemoryEntry::episodic("ep", &emb(0.1))).unwrap();
        mem.remember(MemoryEntry::semantic("sem", &emb(0.2))).unwrap();

        let ep_only = mem.recall(&emb(0.1), 5, Some(MemoryTier::Episodic)).unwrap();
        assert!(ep_only.iter().all(|r| r.tier == MemoryTier::Episodic));
    }

    #[test]
    fn test_consolidate_promotion() {
        let mut mem = AgentMemory::new(
            MemoryConfig::default().with_promotion_threshold(0.4),
        );
        // High importance → should be promoted
        mem.remember(MemoryEntry::episodic("important", &emb(0.5)).with_importance(0.9)).unwrap();
        // Low importance → stays
        mem.remember(MemoryEntry::episodic("trivial", &emb(0.1)).with_importance(0.1)).unwrap();

        let stats = mem.consolidate().unwrap();
        assert_eq!(stats.promoted, 1);
        assert_eq!(*mem.counts().get(&MemoryTier::Semantic).unwrap(), 1);
    }

    #[test]
    fn test_forget() {
        let mut mem = AgentMemory::new(MemoryConfig::default());
        let id = mem.remember(MemoryEntry::episodic("forget me", &emb(0.1))).unwrap();
        assert!(mem.forget(&id).unwrap());
        assert_eq!(mem.total(), 0);
    }

    #[test]
    fn test_procedural_memory() {
        let mut mem = AgentMemory::new(MemoryConfig::default());
        mem.remember(MemoryEntry::procedural("use cargo build", &emb(0.5))).unwrap();
        let results = mem.recall(&emb(0.5), 1, Some(MemoryTier::Procedural)).unwrap();
        assert_eq!(results[0].tier, MemoryTier::Procedural);
    }

    #[test]
    fn test_importance_decay() {
        let entry = MemoryEntry::episodic("old", &emb(0.1)).with_importance(0.5);
        let imp = entry.effective_importance(0.01);
        // Recently created, so importance should be close to base
        assert!(imp > 0.4);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut mem = AgentMemory::new(MemoryConfig::default());
        assert!(mem.remember(MemoryEntry::episodic("bad", &[1.0; 8])).is_err());
    }

    #[test]
    fn test_export_import() {
        let mut mem1 = AgentMemory::new(MemoryConfig::default());
        mem1.remember(MemoryEntry::episodic("fact A", &emb(0.1))).unwrap();
        mem1.remember(MemoryEntry::semantic("fact B", &emb(0.5))).unwrap();

        let exported = mem1.export();
        assert_eq!(exported.len(), 2);

        let mut mem2 = AgentMemory::new(MemoryConfig::default());
        let imported = mem2.import(exported).unwrap();
        assert_eq!(imported, 2);
        assert_eq!(mem2.total(), 2);
    }

    #[test]
    fn test_session_scoped_recall() {
        let config = MemoryConfig::default().with_session("session-1");
        let mut mem = AgentMemory::new(config);
        mem.remember(MemoryEntry::episodic("scoped fact", &emb(0.2))).unwrap();

        // Memory should be findable in session-1
        let results = mem.recall_session(&emb(0.2), 5, "session-1").unwrap();
        assert_eq!(results.len(), 1);

        // Not in session-2
        let results = mem.recall_session(&emb(0.2), 5, "session-2").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_mcp_tool_definitions() {
        let tools = AgentMemory::mcp_tool_definitions();
        assert_eq!(tools.len(), 4);
        assert!(tools.iter().any(|t| t.name == "memory_remember"));
        assert!(tools.iter().any(|t| t.name == "memory_recall"));
        assert!(tools.iter().any(|t| t.name == "memory_forget"));
        assert!(tools.iter().any(|t| t.name == "memory_consolidate"));
    }

    #[test]
    fn test_langchain_adapter() {
        let mut store = LangChainMemoryStore::new(MemoryConfig::default());
        store.add_message("user", "Hello", &emb(0.1)).unwrap();
        store.add_message("assistant", "Hi there", &emb(0.2)).unwrap();

        let results = store.get_relevant(&emb(0.1), 5).unwrap();
        assert!(!results.is_empty());

        let vars = store.load_memory_variables();
        assert_eq!(vars.len(), 2);
        // Both created at same second, so just verify both exist
        let roles: Vec<&str> = vars.iter().map(|(r, _)| r.as_str()).collect();
        assert!(roles.contains(&"user"));
        assert!(roles.contains(&"assistant"));

        store.clear();
        assert_eq!(store.inner().total(), 0);
    }

    #[test]
    fn test_llamaindex_adapter() {
        let mut buf = LlamaIndexMemoryBuffer::new(MemoryConfig::default(), 3);
        buf.put("user", "msg1", &emb(0.1)).unwrap();
        buf.put("assistant", "msg2", &emb(0.2)).unwrap();
        buf.put("user", "msg3", &emb(0.3)).unwrap();
        buf.put("assistant", "msg4", &emb(0.4)).unwrap();

        // Window should be 3 messages
        assert_eq!(buf.len(), 3);
        let history = buf.get();
        assert_eq!(history[0].1, "msg2");

        buf.reset();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_autogen_adapter() {
        let mut provider = AutoGenMemoryProvider::new(MemoryConfig::default(), "agent-1");
        provider.add("fact for agent-1", &emb(0.1)).unwrap();
        assert_eq!(provider.agent_id(), "agent-1");

        let results = provider.query(&emb(0.1), 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ebbinghaus_curve() {
        let mut curve = EbbinghausCurve::new();
        // Fresh memory should have high retention
        assert!(curve.retention() > 0.99);

        // Retrieval should strengthen
        let old_stability = curve.stability;
        curve.record_retrieval();
        assert!(curve.stability > old_stability);
        assert_eq!(curve.repetitions, 1);

        // Optimal review should be positive
        assert!(curve.optimal_review_hours() > 0.0);
    }

    #[test]
    fn test_importance_scorer() {
        let scorer = ImportanceScorer::default();
        let entry = MemoryEntry::episodic("test", &emb(0.1));
        let score = scorer.score(&entry, 0.9);
        assert!(score > 0.0 && score <= 1.0);

        // High relevance should give higher score
        let high = scorer.score(&entry, 1.0);
        let low = scorer.score(&entry, 0.0);
        assert!(high > low);
    }

    #[test]
    fn test_conversation_rag() {
        let mut rag = ConversationRag::new(MemoryConfig::default(), 5);
        rag.add_turn("user", "What is Rust?", Some(&emb(0.1))).unwrap();
        rag.add_turn("assistant", "Rust is a systems language", Some(&emb(0.2))).unwrap();
        rag.add_knowledge("Rust was created by Mozilla", &emb(0.15)).unwrap();

        assert_eq!(rag.history_len(), 2);
        assert_eq!(rag.memory_count(), 3);

        let (turns, memories) = rag.build_context(&emb(0.12), 5).unwrap();
        assert_eq!(turns.len(), 2);
        assert!(!memories.is_empty());
    }

    #[test]
    fn test_conversation_rag_window() {
        let mut rag = ConversationRag::new(MemoryConfig::default(), 2);
        for i in 0..5 {
            rag.add_turn("user", &format!("msg {i}"), Some(&emb(i as f32 * 0.1))).unwrap();
        }

        let (turns, _) = rag.build_context(&emb(0.1), 3).unwrap();
        assert_eq!(turns.len(), 2); // max_context_turns = 2
    }

    #[test]
    fn test_batch_consolidation() {
        let mut mem = AgentMemory::new(
            MemoryConfig::default().with_promotion_threshold(0.3),
        );
        // Add many episodic memories
        for i in 0..10 {
            let imp = if i % 2 == 0 { 0.9 } else { 0.1 };
            mem.remember(
                MemoryEntry::episodic(&format!("fact {i}"), &emb(i as f32 * 0.1))
                    .with_importance(imp),
            )
            .unwrap();
        }

        let stats = mem.consolidate().unwrap();
        // Should promote high-importance ones
        assert!(stats.promoted > 0);
        assert!(stats.scanned > 0);
    }

    #[test]
    fn test_importance_scorer_recency_decay() {
        let scorer = ImportanceScorer {
            recency_weight: 1.0,
            frequency_weight: 0.0,
            relevance_weight: 0.0,
            recency_half_life_hours: 24.0,
        };
        let entry = MemoryEntry::episodic("test", &emb(0.1));
        // Just created, so recency should be close to 1.0
        let score = scorer.score(&entry, 0.5);
        assert!(score > 0.9);
    }

    #[test]
    fn test_importance_scorer_frequency() {
        let scorer = ImportanceScorer {
            recency_weight: 0.0,
            frequency_weight: 1.0,
            relevance_weight: 0.0,
            recency_half_life_hours: 24.0,
        };
        let mut entry = MemoryEntry::episodic("test", &emb(0.1));
        let score_0 = scorer.score(&entry, 0.5);
        entry.access_count = 10;
        let score_10 = scorer.score(&entry, 0.5);
        assert!(score_10 > score_0);
    }

    #[test]
    fn test_conversation_rag_knowledge_retrieval() {
        let mut rag = ConversationRag::new(MemoryConfig::default(), 10);
        // Add knowledge
        rag.add_knowledge("Rust was first released in 2015", &emb(0.3)).unwrap();
        rag.add_knowledge("Python was created by Guido van Rossum", &emb(0.7)).unwrap();

        // Add conversation turns
        rag.add_turn("user", "Tell me about Rust", Some(&emb(0.31))).unwrap();

        let (turns, memories) = rag.build_context(&emb(0.3), 5).unwrap();
        assert_eq!(turns.len(), 1);
        // Should find the Rust knowledge first
        assert!(!memories.is_empty());
    }

    #[test]
    fn test_ebbinghaus_multiple_retrievals() {
        let mut curve = EbbinghausCurve::new();
        let initial_stability = curve.stability;

        // Multiple retrievals should compound stability
        for _ in 0..5 {
            curve.record_retrieval();
        }
        assert!(curve.stability > initial_stability * 3.0);
        assert_eq!(curve.repetitions, 5);
        // Should retain very well with many repetitions
        assert!(curve.retention() > 0.99);
    }

    #[test]
    fn test_memory_export_import_preserves_tiers() {
        let mut mem1 = AgentMemory::new(MemoryConfig::default());
        mem1.remember(MemoryEntry::episodic("ep", &emb(0.1))).unwrap();
        mem1.remember(MemoryEntry::semantic("sem", &emb(0.5))).unwrap();
        mem1.remember(MemoryEntry::procedural("proc", &emb(0.9))).unwrap();

        let exported = mem1.export();
        assert_eq!(exported.len(), 3);

        let mut mem2 = AgentMemory::new(MemoryConfig::default());
        mem2.import(exported).unwrap();
        assert_eq!(mem2.total(), 3);

        let counts = mem2.counts();
        assert_eq!(*counts.get(&MemoryTier::Episodic).unwrap(), 1);
        assert_eq!(*counts.get(&MemoryTier::Semantic).unwrap(), 1);
        assert_eq!(*counts.get(&MemoryTier::Procedural).unwrap(), 1);
    }
}
