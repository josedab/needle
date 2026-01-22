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
                let _ = self.episodic.delete(&id);
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
                    let _ = self.episodic.delete(&id);
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
            let _ = coll.delete(id);
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
}
