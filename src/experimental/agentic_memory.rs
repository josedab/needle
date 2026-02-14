//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Agentic Memory System
//!
//! A specialized memory store for AI agents that provides long-term memory
//! with temporal decay, importance scoring, and associative retrieval.
//!
//! # Memory Types
//!
//! - **Short-term Memory**: Session-scoped, high-frequency access, auto-expiring
//! - **Long-term Memory**: Persistent, importance-weighted, semantic retrieval
//! - **Working Memory**: Active context window for current reasoning
//!
//! # Features
//!
//! - Temporal decay functions (exponential, linear, step)
//! - Importance scoring based on access patterns
//! - Associative retrieval chains for related memories
//! - Automatic summarization and consolidation
//! - Memory capacity management with smart eviction
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::agentic_memory::{AgentMemory, MemoryConfig, MemoryType};
//!
//! let config = MemoryConfig::new(384)
//!     .with_short_term_capacity(100)
//!     .with_long_term_capacity(10000)
//!     .with_decay_function(DecayFunction::Exponential { half_life_hours: 24.0 });
//!
//! let memory = AgentMemory::new(config);
//!
//! // Store a memory
//! memory.remember("user_preference", &embedding, MemoryType::LongTerm, json!({
//!     "content": "User prefers concise responses",
//!     "source": "conversation",
//! }))?;
//!
//! // Recall relevant memories
//! let memories = memory.recall(&query_embedding, 5)?;
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Memory type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Short-term memory - session-scoped, auto-expiring
    ShortTerm,
    /// Long-term memory - persistent, importance-weighted
    LongTerm,
    /// Working memory - active context for current task
    Working,
    /// Episodic memory - specific events/experiences
    Episodic,
    /// Semantic memory - facts and concepts
    Semantic,
    /// Procedural memory - how-to knowledge
    Procedural,
}

impl Default for MemoryType {
    fn default() -> Self {
        MemoryType::LongTerm
    }
}

/// Temporal decay function for memory importance
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DecayFunction {
    /// No decay - memories maintain importance
    None,
    /// Exponential decay with half-life
    Exponential { half_life_hours: f64 },
    /// Linear decay over time
    Linear { decay_rate_per_hour: f64 },
    /// Step decay - discrete drops at intervals
    Step {
        interval_hours: f64,
        decay_per_step: f64,
    },
    /// Power law decay (forgetting curve)
    PowerLaw { exponent: f64 },
}

impl Default for DecayFunction {
    fn default() -> Self {
        DecayFunction::Exponential {
            half_life_hours: 168.0,
        } // 1 week
    }
}

impl DecayFunction {
    /// Calculate decay factor for a given age in hours
    pub fn decay_factor(&self, age_hours: f64) -> f64 {
        match self {
            DecayFunction::None => 1.0,
            DecayFunction::Exponential { half_life_hours } => {
                0.5_f64.powf(age_hours / half_life_hours)
            }
            DecayFunction::Linear {
                decay_rate_per_hour,
            } => (1.0 - decay_rate_per_hour * age_hours).max(0.0),
            DecayFunction::Step {
                interval_hours,
                decay_per_step,
            } => {
                let steps = (age_hours / interval_hours).floor();
                (1.0 - decay_per_step * steps).max(0.0)
            }
            DecayFunction::PowerLaw { exponent } => 1.0 / (1.0 + age_hours).powf(*exponent),
        }
    }
}

/// Configuration for agent memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Embedding dimensions
    pub dimensions: usize,
    /// Short-term memory capacity
    pub short_term_capacity: usize,
    /// Long-term memory capacity
    pub long_term_capacity: usize,
    /// Working memory capacity
    pub working_memory_capacity: usize,
    /// Decay function for importance
    pub decay_function: DecayFunction,
    /// Distance function for similarity
    pub distance_function: DistanceFunction,
    /// Minimum similarity for recall
    pub recall_threshold: f32,
    /// Enable associative linking
    pub enable_associations: bool,
    /// Maximum associations per memory
    pub max_associations: usize,
    /// Association similarity threshold
    pub association_threshold: f32,
    /// Short-term TTL in seconds
    pub short_term_ttl_secs: u64,
    /// Base importance for new memories
    pub base_importance: f64,
    /// Importance boost per access
    pub access_boost: f64,
    /// Enable automatic consolidation
    pub auto_consolidate: bool,
    /// Consolidation threshold (importance)
    pub consolidation_threshold: f64,
}

impl MemoryConfig {
    /// Create a new config with dimensions
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            short_term_capacity: 100,
            long_term_capacity: 10000,
            working_memory_capacity: 20,
            decay_function: DecayFunction::default(),
            distance_function: DistanceFunction::Cosine,
            recall_threshold: 0.5,
            enable_associations: true,
            max_associations: 5,
            association_threshold: 0.7,
            short_term_ttl_secs: 3600, // 1 hour
            base_importance: 1.0,
            access_boost: 0.1,
            auto_consolidate: true,
            consolidation_threshold: 2.0,
        }
    }

    /// Set short-term capacity
    pub fn with_short_term_capacity(mut self, capacity: usize) -> Self {
        self.short_term_capacity = capacity;
        self
    }

    /// Set long-term capacity
    pub fn with_long_term_capacity(mut self, capacity: usize) -> Self {
        self.long_term_capacity = capacity;
        self
    }

    /// Set decay function
    pub fn with_decay_function(mut self, decay: DecayFunction) -> Self {
        self.decay_function = decay;
        self
    }

    /// Set recall threshold
    pub fn with_recall_threshold(mut self, threshold: f32) -> Self {
        self.recall_threshold = threshold;
        self
    }

    /// Enable/disable associations
    pub fn with_associations(mut self, enable: bool) -> Self {
        self.enable_associations = enable;
        self
    }
}

/// A single memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    /// Unique memory ID
    pub id: String,
    /// Memory type
    pub memory_type: MemoryType,
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Memory content/metadata
    pub content: Value,
    /// Creation timestamp
    pub created_at: u64,
    /// Last access timestamp
    pub last_accessed: u64,
    /// Access count
    pub access_count: u64,
    /// Base importance score
    pub importance: f64,
    /// Associated memory IDs
    pub associations: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Source context
    pub source: Option<String>,
}

impl Memory {
    /// Create a new memory
    pub fn new(
        id: impl Into<String>,
        memory_type: MemoryType,
        embedding: Vec<f32>,
        content: Value,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: id.into(),
            memory_type,
            embedding,
            content,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            importance: 1.0,
            associations: Vec::new(),
            tags: Vec::new(),
            source: None,
        }
    }

    /// Calculate age in hours
    pub fn age_hours(&self) -> f64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        (now.saturating_sub(self.created_at)) as f64 / 3600.0
    }

    /// Calculate effective importance with decay
    pub fn effective_importance(&self, decay: &DecayFunction) -> f64 {
        let decay_factor = decay.decay_factor(self.age_hours());
        self.importance * decay_factor
    }

    /// Mark as accessed
    pub fn mark_accessed(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.last_accessed = now;
        self.access_count += 1;
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// Result of a memory recall operation
#[derive(Debug, Clone)]
pub struct RecallResult {
    /// Retrieved memory
    pub memory: Memory,
    /// Similarity score
    pub similarity: f32,
    /// Effective importance (with decay)
    pub effective_importance: f64,
    /// Combined relevance score
    pub relevance: f64,
}

/// Statistics for agent memory
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total memories stored
    pub total_memories: usize,
    /// Short-term memories
    pub short_term_count: usize,
    /// Long-term memories
    pub long_term_count: usize,
    /// Working memory count
    pub working_memory_count: usize,
    /// Total recalls
    pub total_recalls: u64,
    /// Cache hits (found relevant memory)
    pub recall_hits: u64,
    /// Average recall time in microseconds
    pub avg_recall_time_us: u64,
    /// Memories consolidated
    pub consolidations: u64,
    /// Memories evicted
    pub evictions: u64,
}

/// Agent memory system
pub struct AgentMemory {
    config: MemoryConfig,
    /// Short-term memories
    short_term: RwLock<HashMap<String, Memory>>,
    /// Long-term memories
    long_term: RwLock<HashMap<String, Memory>>,
    /// Working memory (ordered by recency)
    working: RwLock<VecDeque<String>>,
    /// All memory IDs by type
    by_type: RwLock<HashMap<MemoryType, HashSet<String>>>,
    /// Memory ID counter
    id_counter: AtomicU64,
    /// Statistics
    stats: RwLock<MemoryStats>,
}

impl AgentMemory {
    /// Create a new agent memory system
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            short_term: RwLock::new(HashMap::new()),
            long_term: RwLock::new(HashMap::new()),
            working: RwLock::new(VecDeque::new()),
            by_type: RwLock::new(HashMap::new()),
            id_counter: AtomicU64::new(0),
            stats: RwLock::new(MemoryStats::default()),
        }
    }

    /// Generate a new memory ID
    fn generate_id(&self) -> String {
        let counter = self.id_counter.fetch_add(1, Ordering::SeqCst);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        format!("mem_{}_{}", timestamp, counter)
    }

    /// Store a new memory
    pub fn remember(
        &self,
        embedding: &[f32],
        memory_type: MemoryType,
        content: Value,
    ) -> Result<String> {
        if embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: embedding.len(),
            });
        }

        let id = self.generate_id();
        let mut memory = Memory::new(&id, memory_type, embedding.to_vec(), content);
        memory.importance = self.config.base_importance;

        // Find associations if enabled
        if self.config.enable_associations {
            let associations = self.find_associations(embedding, &id);
            memory.associations = associations;
        }

        // Store based on type
        match memory_type {
            MemoryType::ShortTerm => {
                self.maybe_evict_short_term();
                self.short_term.write().insert(id.clone(), memory);
            }
            MemoryType::Working => {
                self.add_to_working(&id);
                self.short_term.write().insert(id.clone(), memory);
            }
            _ => {
                self.maybe_evict_long_term();
                self.long_term.write().insert(id.clone(), memory);
            }
        }

        // Track by type
        self.by_type
            .write()
            .entry(memory_type)
            .or_default()
            .insert(id.clone());

        // Update stats
        let mut stats = self.stats.write();
        stats.total_memories += 1;
        match memory_type {
            MemoryType::ShortTerm => stats.short_term_count += 1,
            MemoryType::Working => stats.working_memory_count += 1,
            _ => stats.long_term_count += 1,
        }

        Ok(id)
    }

    /// Recall memories similar to a query
    pub fn recall(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<RecallResult>> {
        if query_embedding.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: query_embedding.len(),
            });
        }

        let start = Instant::now();
        let mut results = Vec::new();

        // Search short-term memories
        {
            let short_term = self.short_term.read();
            for memory in short_term.values() {
                let similarity = self.compute_similarity(query_embedding, &memory.embedding);
                if similarity >= self.config.recall_threshold {
                    let effective_importance =
                        memory.effective_importance(&self.config.decay_function);
                    let relevance = similarity as f64 * 0.7 + effective_importance * 0.3;
                    results.push(RecallResult {
                        memory: memory.clone(),
                        similarity,
                        effective_importance,
                        relevance,
                    });
                }
            }
        }

        // Search long-term memories
        {
            let long_term = self.long_term.read();
            for memory in long_term.values() {
                let similarity = self.compute_similarity(query_embedding, &memory.embedding);
                if similarity >= self.config.recall_threshold {
                    let effective_importance =
                        memory.effective_importance(&self.config.decay_function);
                    let relevance = similarity as f64 * 0.7 + effective_importance * 0.3;
                    results.push(RecallResult {
                        memory: memory.clone(),
                        similarity,
                        effective_importance,
                        relevance,
                    });
                }
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| OrderedFloat(b.relevance).cmp(&OrderedFloat(a.relevance)));
        results.truncate(limit);

        // Update access counts for returned memories
        for result in &results {
            self.mark_accessed(&result.memory.id);
        }

        // Update stats
        let elapsed = start.elapsed();
        let mut stats = self.stats.write();
        stats.total_recalls += 1;
        if !results.is_empty() {
            stats.recall_hits += 1;
        }
        stats.avg_recall_time_us = (stats.avg_recall_time_us * (stats.total_recalls - 1)
            + elapsed.as_micros() as u64)
            / stats.total_recalls;

        Ok(results)
    }

    /// Recall with specific memory types
    pub fn recall_by_type(
        &self,
        query_embedding: &[f32],
        memory_type: MemoryType,
        limit: usize,
    ) -> Result<Vec<RecallResult>> {
        let all = self.recall(query_embedding, limit * 2)?;
        Ok(all
            .into_iter()
            .filter(|r| r.memory.memory_type == memory_type)
            .take(limit)
            .collect())
    }

    /// Get associated memories
    pub fn get_associations(&self, memory_id: &str) -> Vec<Memory> {
        let memory = self.get(memory_id);
        match memory {
            Some(mem) => mem
                .associations
                .iter()
                .filter_map(|id| self.get(id))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get a specific memory by ID
    pub fn get(&self, id: &str) -> Option<Memory> {
        if let Some(memory) = self.short_term.read().get(id) {
            return Some(memory.clone());
        }
        self.long_term.read().get(id).cloned()
    }

    /// Forget a memory
    pub fn forget(&self, id: &str) -> bool {
        let removed = self.short_term.write().remove(id).is_some()
            || self.long_term.write().remove(id).is_some();

        if removed {
            // Remove from type index
            let mut by_type = self.by_type.write();
            for ids in by_type.values_mut() {
                ids.remove(id);
            }

            // Remove from working memory
            self.working.write().retain(|i| i != id);
        }

        removed
    }

    /// Consolidate short-term to long-term memory
    pub fn consolidate(&self) -> usize {
        let mut consolidated = 0;
        let threshold = self.config.consolidation_threshold;

        let to_consolidate: Vec<_> = {
            let short_term = self.short_term.read();
            short_term
                .values()
                .filter(|m| m.effective_importance(&self.config.decay_function) >= threshold)
                .filter(|m| m.memory_type == MemoryType::ShortTerm)
                .cloned()
                .collect()
        };

        for mut memory in to_consolidate {
            memory.memory_type = MemoryType::LongTerm;

            self.short_term.write().remove(&memory.id);
            self.long_term
                .write()
                .insert(memory.id.clone(), memory.clone());

            // Update type tracking
            let mut by_type = self.by_type.write();
            by_type
                .entry(MemoryType::ShortTerm)
                .or_default()
                .remove(&memory.id);
            by_type
                .entry(MemoryType::LongTerm)
                .or_default()
                .insert(memory.id);

            consolidated += 1;
        }

        self.stats.write().consolidations += consolidated as u64;
        consolidated
    }

    /// Clear expired short-term memories
    pub fn clear_expired(&self) -> usize {
        let ttl = self.config.short_term_ttl_secs;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let expired: Vec<_> = {
            let short_term = self.short_term.read();
            short_term
                .values()
                .filter(|m| now - m.created_at > ttl)
                .map(|m| m.id.clone())
                .collect()
        };

        let count = expired.len();
        for id in expired {
            self.forget(&id);
        }

        count
    }

    /// Get working memory content
    pub fn get_working_memory(&self) -> Vec<Memory> {
        let working = self.working.read();
        working.iter().filter_map(|id| self.get(id)).collect()
    }

    /// Clear working memory
    pub fn clear_working_memory(&self) {
        let ids: Vec<_> = self.working.write().drain(..).collect();
        for id in ids {
            if let Some(mut memory) = self.short_term.write().remove(&id) {
                // Move to short-term if important enough
                if memory.effective_importance(&self.config.decay_function) > 0.5 {
                    memory.memory_type = MemoryType::ShortTerm;
                    self.short_term.write().insert(id, memory);
                }
            }
        }
        self.stats.write().working_memory_count = 0;
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryStats {
        self.stats.read().clone()
    }

    /// Get total memory count
    pub fn len(&self) -> usize {
        self.short_term.read().len() + self.long_term.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.short_term.read().is_empty() && self.long_term.read().is_empty()
    }

    // === Private helpers ===

    fn compute_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let distance = self.config.distance_function.compute(a, b);
        match self.config.distance_function {
            DistanceFunction::Cosine | DistanceFunction::CosineNormalized => 1.0 - distance,
            DistanceFunction::Euclidean => 1.0 / (1.0 + distance),
            DistanceFunction::DotProduct => (1.0 + distance) / 2.0,
            DistanceFunction::Manhattan => 1.0 / (1.0 + distance),
        }
    }

    fn find_associations(&self, embedding: &[f32], exclude_id: &str) -> Vec<String> {
        let threshold = self.config.association_threshold;
        let max = self.config.max_associations;

        let mut associations = Vec::new();

        // Check long-term memories
        {
            let long_term = self.long_term.read();
            for (id, memory) in long_term.iter() {
                if id != exclude_id {
                    let similarity = self.compute_similarity(embedding, &memory.embedding);
                    if similarity >= threshold {
                        associations.push((id.clone(), similarity));
                    }
                }
            }
        }

        // Sort and limit
        associations.sort_by(|a, b| OrderedFloat(b.1).cmp(&OrderedFloat(a.1)));
        associations.truncate(max);
        associations.into_iter().map(|(id, _)| id).collect()
    }

    fn mark_accessed(&self, id: &str) {
        if let Some(memory) = self.short_term.write().get_mut(id) {
            memory.mark_accessed();
            memory.importance += self.config.access_boost;
        } else if let Some(memory) = self.long_term.write().get_mut(id) {
            memory.mark_accessed();
            memory.importance += self.config.access_boost;
        }
    }

    fn add_to_working(&self, id: &str) {
        let mut working = self.working.write();

        // Remove if already present
        working.retain(|i| i != id);

        // Add to front
        working.push_front(id.to_string());

        // Enforce capacity
        while working.len() > self.config.working_memory_capacity {
            working.pop_back();
        }
    }

    fn maybe_evict_short_term(&self) {
        let capacity = self.config.short_term_capacity;
        let mut short_term = self.short_term.write();

        while short_term.len() >= capacity {
            // Find lowest importance memory
            let lowest = short_term
                .iter()
                .min_by(|a, b| {
                    let imp_a = a.1.effective_importance(&self.config.decay_function);
                    let imp_b = b.1.effective_importance(&self.config.decay_function);
                    OrderedFloat(imp_a).cmp(&OrderedFloat(imp_b))
                })
                .map(|(id, _)| id.clone());

            if let Some(id) = lowest {
                short_term.remove(&id);
                self.stats.write().evictions += 1;
            } else {
                break;
            }
        }
    }

    fn maybe_evict_long_term(&self) {
        let capacity = self.config.long_term_capacity;
        let mut long_term = self.long_term.write();

        while long_term.len() >= capacity {
            // Find lowest importance memory
            let lowest = long_term
                .iter()
                .min_by(|a, b| {
                    let imp_a = a.1.effective_importance(&self.config.decay_function);
                    let imp_b = b.1.effective_importance(&self.config.decay_function);
                    OrderedFloat(imp_a).cmp(&OrderedFloat(imp_b))
                })
                .map(|(id, _)| id.clone());

            if let Some(id) = lowest {
                long_term.remove(&id);
                self.stats.write().evictions += 1;
            } else {
                break;
            }
        }
    }
}

/// Builder for creating agent memory with fluent API
pub struct AgentMemoryBuilder {
    config: MemoryConfig,
}

impl AgentMemoryBuilder {
    /// Create a new builder
    pub fn new(dimensions: usize) -> Self {
        Self {
            config: MemoryConfig::new(dimensions),
        }
    }

    /// Set short-term capacity
    pub fn short_term_capacity(mut self, capacity: usize) -> Self {
        self.config.short_term_capacity = capacity;
        self
    }

    /// Set long-term capacity
    pub fn long_term_capacity(mut self, capacity: usize) -> Self {
        self.config.long_term_capacity = capacity;
        self
    }

    /// Set working memory capacity
    pub fn working_memory_capacity(mut self, capacity: usize) -> Self {
        self.config.working_memory_capacity = capacity;
        self
    }

    /// Set decay function
    pub fn decay_function(mut self, decay: DecayFunction) -> Self {
        self.config.decay_function = decay;
        self
    }

    /// Set recall threshold
    pub fn recall_threshold(mut self, threshold: f32) -> Self {
        self.config.recall_threshold = threshold;
        self
    }

    /// Enable associations
    pub fn with_associations(mut self) -> Self {
        self.config.enable_associations = true;
        self
    }

    /// Build the agent memory
    pub fn build(self) -> AgentMemory {
        AgentMemory::new(self.config)
    }
}

// ---------------------------------------------------------------------------
// Conversation History Tracker
// ---------------------------------------------------------------------------

/// Role in a conversation turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConversationRole {
    User,
    Assistant,
    System,
    Tool,
}

/// A single turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub id: String,
    pub role: ConversationRole,
    pub content: String,
    pub timestamp: u64,
    pub token_count: usize,
    pub metadata: Option<Value>,
}

/// Configuration for conversation tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    /// Maximum turns to keep in history
    pub max_turns: usize,
    /// Maximum total tokens in the context window
    pub max_context_tokens: usize,
    /// Embed and persist turns to long-term memory automatically
    pub auto_persist: bool,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_turns: 100,
            max_context_tokens: 8192,
            auto_persist: false,
        }
    }
}

/// Tracks conversation history with sliding window support.
pub struct ConversationTracker {
    config: ConversationConfig,
    sessions: RwLock<HashMap<String, ConversationSession>>,
}

struct ConversationSession {
    turns: VecDeque<ConversationTurn>,
    total_tokens: usize,
    #[allow(dead_code)]
    created_at: u64,
}

impl ConversationTracker {
    pub fn new(config: ConversationConfig) -> Self {
        Self {
            config,
            sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Start or get a session by ID.
    fn ensure_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write();
        sessions.entry(session_id.to_string()).or_insert_with(|| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            ConversationSession {
                turns: VecDeque::new(),
                total_tokens: 0,
                created_at: now,
            }
        });
    }

    /// Add a turn to the session.
    pub fn add_turn(
        &self,
        session_id: &str,
        role: ConversationRole,
        content: &str,
        token_count: usize,
        metadata: Option<Value>,
    ) -> String {
        self.ensure_session(session_id);
        let turn_id = format!("turn_{:016x}", {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            session_id.hash(&mut h);
            content.hash(&mut h);
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
                .hash(&mut h);
            h.finish()
        });
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let turn = ConversationTurn {
            id: turn_id.clone(),
            role,
            content: content.to_string(),
            timestamp: now,
            token_count,
            metadata,
        };

        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.get_mut(session_id) {
            session.total_tokens += token_count;
            session.turns.push_back(turn);

            // Evict oldest turns if over limits
            while session.turns.len() > self.config.max_turns
                || session.total_tokens > self.config.max_context_tokens
            {
                if let Some(removed) = session.turns.pop_front() {
                    session.total_tokens = session.total_tokens.saturating_sub(removed.token_count);
                } else {
                    break;
                }
            }
        }
        turn_id
    }

    /// Get the current context window for a session.
    pub fn get_context_window(&self, session_id: &str, max_tokens: usize) -> Vec<ConversationTurn> {
        let sessions = self.sessions.read();
        let session = match sessions.get(session_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut result = Vec::new();
        let mut tokens = 0;
        for turn in session.turns.iter().rev() {
            if tokens + turn.token_count > max_tokens {
                break;
            }
            tokens += turn.token_count;
            result.push(turn.clone());
        }
        result.reverse();
        result
    }

    /// Get full history for a session.
    pub fn get_history(&self, session_id: &str) -> Vec<ConversationTurn> {
        self.sessions
            .read()
            .get(session_id)
            .map(|s| s.turns.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Delete a session.
    pub fn delete_session(&self, session_id: &str) {
        self.sessions.write().remove(session_id);
    }

    /// List active session IDs.
    pub fn list_sessions(&self) -> Vec<String> {
        self.sessions.read().keys().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// Tool-Call Cache
// ---------------------------------------------------------------------------

/// A cached tool-call result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedToolCall {
    pub tool_name: String,
    pub arguments_hash: u64,
    pub result: Value,
    pub cached_at: u64,
    pub ttl_seconds: u64,
    pub hit_count: u64,
}

/// Caches tool-call results to avoid redundant LLM-driven tool invocations.
pub struct ToolCallCache {
    cache: RwLock<HashMap<u64, CachedToolCall>>,
    max_entries: usize,
    default_ttl_seconds: u64,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl ToolCallCache {
    pub fn new(max_entries: usize, default_ttl_seconds: u64) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            max_entries,
            default_ttl_seconds,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn hash_call(tool_name: &str, arguments: &Value) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        tool_name.hash(&mut hasher);
        arguments.to_string().hash(&mut hasher);
        hasher.finish()
    }

    /// Look up a cached tool-call result.
    pub fn get(&self, tool_name: &str, arguments: &Value) -> Option<Value> {
        let key = Self::hash_call(tool_name, arguments);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut cache = self.cache.write();
        if let Some(entry) = cache.get_mut(&key) {
            if now.saturating_sub(entry.cached_at) < entry.ttl_seconds {
                entry.hit_count += 1;
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(entry.result.clone());
            }
            // Expired
            cache.remove(&key);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store a tool-call result in the cache.
    pub fn put(&self, tool_name: &str, arguments: &Value, result: Value) {
        self.put_with_ttl(tool_name, arguments, result, self.default_ttl_seconds);
    }

    /// Store with a custom TTL.
    pub fn put_with_ttl(
        &self,
        tool_name: &str,
        arguments: &Value,
        result: Value,
        ttl_seconds: u64,
    ) {
        let key = Self::hash_call(tool_name, arguments);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let entry = CachedToolCall {
            tool_name: tool_name.to_string(),
            arguments_hash: key,
            result,
            cached_at: now,
            ttl_seconds,
            hit_count: 0,
        };

        let mut cache = self.cache.write();
        // Evict LRU if over capacity
        if cache.len() >= self.max_entries && !cache.contains_key(&key) {
            if let Some(oldest_key) = cache
                .iter()
                .min_by_key(|(_, v)| v.cached_at)
                .map(|(k, _)| *k)
            {
                cache.remove(&oldest_key);
            }
        }
        cache.insert(key, entry);
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.cache.write().clear();
    }

    /// Cache statistics.
    pub fn stats(&self) -> ToolCallCacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        ToolCallCacheStats {
            entries: self.cache.read().len(),
            hits,
            misses,
            hit_rate: if hits + misses > 0 {
                hits as f64 / (hits + misses) as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for the tool-call cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallCacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

// ---------------------------------------------------------------------------
// Session Context Window Manager
// ---------------------------------------------------------------------------

/// Manages dynamic context windowing for LLM prompts, combining conversation
/// history, relevant memories, and tool results within token budget.
pub struct ContextWindowManager {
    max_tokens: usize,
    system_prompt_tokens: usize,
    reserved_for_response: usize,
}

/// A section of the context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSection {
    pub label: String,
    pub content: String,
    pub token_count: usize,
    pub priority: u32,
}

impl ContextWindowManager {
    pub fn new(
        max_tokens: usize,
        system_prompt_tokens: usize,
        reserved_for_response: usize,
    ) -> Self {
        Self {
            max_tokens,
            system_prompt_tokens,
            reserved_for_response,
        }
    }

    /// Available tokens after system prompt and response reservation.
    pub fn available_tokens(&self) -> usize {
        self.max_tokens
            .saturating_sub(self.system_prompt_tokens)
            .saturating_sub(self.reserved_for_response)
    }

    /// Build a context window from sections, fitting as many as possible
    /// in priority order within the token budget.
    pub fn build_window(&self, mut sections: Vec<ContextSection>) -> Vec<ContextSection> {
        let budget = self.available_tokens();
        sections.sort_by(|a, b| b.priority.cmp(&a.priority));

        let mut used = 0;
        let mut result = Vec::new();
        for section in sections {
            if used + section.token_count <= budget {
                used += section.token_count;
                result.push(section);
            }
        }
        // Restore insertion order by label (conversation turns should appear chronologically)
        result.sort_by(|a, b| a.label.cmp(&b.label));
        result
    }

    /// Convenience: combine conversation history and recalled memories into a window.
    pub fn compose(
        &self,
        conversation: &[ConversationTurn],
        memories: &[RecallResult],
    ) -> Vec<ContextSection> {
        let mut sections = Vec::new();

        // Recent conversation turns (higher priority for recent)
        for (i, turn) in conversation.iter().enumerate() {
            sections.push(ContextSection {
                label: format!("turn_{:04}", i),
                content: format!(
                    "[{}] {}",
                    match turn.role {
                        ConversationRole::User => "user",
                        ConversationRole::Assistant => "assistant",
                        ConversationRole::System => "system",
                        ConversationRole::Tool => "tool",
                    },
                    turn.content
                ),
                token_count: turn.token_count,
                priority: 100 + i as u32, // more recent = higher
            });
        }

        // Recalled memories (priority based on relevance score)
        for (i, recall) in memories.iter().enumerate() {
            let priority = (recall.relevance * 50.0) as u32;
            sections.push(ContextSection {
                label: format!("memory_{:04}", i),
                content: recall.memory.content.to_string(),
                token_count: recall.memory.content.to_string().len() / 4, // rough estimate
                priority,
            });
        }

        self.build_window(sections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn random_embedding(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            emb.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
        }
        // Normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut emb {
                *v /= norm;
            }
        }
        emb
    }

    #[test]
    fn test_decay_functions() {
        let exp = DecayFunction::Exponential {
            half_life_hours: 24.0,
        };
        assert!((exp.decay_factor(0.0) - 1.0).abs() < 0.01);
        assert!((exp.decay_factor(24.0) - 0.5).abs() < 0.01);
        assert!((exp.decay_factor(48.0) - 0.25).abs() < 0.01);

        let linear = DecayFunction::Linear {
            decay_rate_per_hour: 0.1,
        };
        assert!((linear.decay_factor(0.0) - 1.0).abs() < 0.01);
        assert!((linear.decay_factor(5.0) - 0.5).abs() < 0.01);
        assert!((linear.decay_factor(10.0)).abs() < 0.01);

        let none = DecayFunction::None;
        assert!((none.decay_factor(1000.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_creation() {
        let memory = AgentMemory::new(MemoryConfig::new(128));
        assert!(memory.is_empty());
        assert_eq!(memory.len(), 0);
    }

    #[test]
    fn test_remember_and_recall() {
        let memory = AgentMemory::new(MemoryConfig::new(64).with_recall_threshold(0.0));

        let emb1 = random_embedding(64, 1);
        let emb2 = random_embedding(64, 2);

        memory
            .remember(&emb1, MemoryType::LongTerm, json!({"content": "test1"}))
            .unwrap();
        memory
            .remember(&emb2, MemoryType::LongTerm, json!({"content": "test2"}))
            .unwrap();

        assert_eq!(memory.len(), 2);

        // Recall similar to emb1
        let results = memory.recall(&emb1, 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].memory.content["content"], "test1");
    }

    #[test]
    fn test_memory_types() {
        let config = MemoryConfig::new(64)
            .with_short_term_capacity(10)
            .with_long_term_capacity(100);
        let memory = AgentMemory::new(config);

        let emb = random_embedding(64, 1);

        memory
            .remember(&emb, MemoryType::ShortTerm, json!({}))
            .unwrap();
        memory
            .remember(&emb, MemoryType::LongTerm, json!({}))
            .unwrap();
        memory
            .remember(&emb, MemoryType::Episodic, json!({}))
            .unwrap();

        assert_eq!(memory.len(), 3);

        let stats = memory.stats();
        assert_eq!(stats.short_term_count, 1);
        assert_eq!(stats.long_term_count, 2);
    }

    #[test]
    fn test_working_memory() {
        let config = MemoryConfig::new(64).with_recall_threshold(0.0);
        let mut config = config;
        config.working_memory_capacity = 3;
        let memory = AgentMemory::new(config);

        for i in 0..5 {
            let emb = random_embedding(64, i);
            memory
                .remember(&emb, MemoryType::Working, json!({"i": i}))
                .unwrap();
        }

        let working = memory.get_working_memory();
        assert_eq!(working.len(), 3);
    }

    #[test]
    fn test_forget() {
        let memory = AgentMemory::new(MemoryConfig::new(64));

        let emb = random_embedding(64, 1);
        let id = memory
            .remember(&emb, MemoryType::LongTerm, json!({}))
            .unwrap();

        assert_eq!(memory.len(), 1);
        assert!(memory.get(&id).is_some());

        assert!(memory.forget(&id));
        assert_eq!(memory.len(), 0);
        assert!(memory.get(&id).is_none());
    }

    #[test]
    fn test_eviction() {
        let config = MemoryConfig::new(64).with_short_term_capacity(3);
        let memory = AgentMemory::new(config);

        for i in 0..5 {
            let emb = random_embedding(64, i);
            memory
                .remember(&emb, MemoryType::ShortTerm, json!({"i": i}))
                .unwrap();
        }

        // Should have evicted 2
        assert!(memory.short_term.read().len() <= 3);

        let stats = memory.stats();
        assert!(stats.evictions >= 2);
    }

    #[test]
    fn test_builder() {
        let memory = AgentMemoryBuilder::new(128)
            .short_term_capacity(50)
            .long_term_capacity(5000)
            .decay_function(DecayFunction::Exponential {
                half_life_hours: 48.0,
            })
            .recall_threshold(0.6)
            .with_associations()
            .build();

        assert!(memory.is_empty());
    }

    #[test]
    fn test_conversation_tracker() {
        let tracker = ConversationTracker::new(ConversationConfig {
            max_turns: 5,
            max_context_tokens: 1000,
            ..Default::default()
        });

        let session = "session_1";
        tracker.add_turn(session, ConversationRole::User, "Hello", 10, None);
        tracker.add_turn(session, ConversationRole::Assistant, "Hi there!", 15, None);
        tracker.add_turn(session, ConversationRole::User, "Search for docs", 20, None);

        let history = tracker.get_history(session);
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].role, ConversationRole::User);
        assert_eq!(history[1].role, ConversationRole::Assistant);
    }

    #[test]
    fn test_conversation_window_eviction() {
        let tracker = ConversationTracker::new(ConversationConfig {
            max_turns: 3,
            max_context_tokens: 10000,
            ..Default::default()
        });
        let s = "s1";
        for i in 0..6 {
            tracker.add_turn(s, ConversationRole::User, &format!("msg {}", i), 10, None);
        }
        let history = tracker.get_history(s);
        assert_eq!(history.len(), 3); // oldest evicted
    }

    #[test]
    fn test_tool_call_cache() {
        let cache = ToolCallCache::new(100, 3600);

        let args = json!({"query": "hello world"});
        assert!(cache.get("search", &args).is_none());

        cache.put("search", &args, json!({"results": [1, 2, 3]}));
        let hit = cache.get("search", &args).unwrap();
        assert_eq!(hit, json!({"results": [1, 2, 3]}));

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_rate > 0.0);
    }

    #[test]
    fn test_context_window_manager() {
        let mgr = ContextWindowManager::new(4096, 200, 800);
        assert_eq!(mgr.available_tokens(), 3096);

        let sections = vec![
            ContextSection {
                label: "a".into(),
                content: "low".into(),
                token_count: 100,
                priority: 10,
            },
            ContextSection {
                label: "b".into(),
                content: "high".into(),
                token_count: 100,
                priority: 50,
            },
            ContextSection {
                label: "c".into(),
                content: "huge".into(),
                token_count: 5000,
                priority: 99,
            },
        ];
        let result = mgr.build_window(sections);
        // "c" doesn't fit (5000 > 3096). "b" and "a" fit.
        assert_eq!(result.len(), 2);
    }
}
