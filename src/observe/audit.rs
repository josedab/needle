#![allow(dead_code)]

//! Vector Lineage & Audit Trail
//!
//! Track full provenance and access history for compliance. Provides lineage
//! tracking, search audit events, and compliance reporting.
//!
//! # Features
//!
//! - **VectorLineage**: Track source documents, embedding models, transformations
//! - **SearchAuditEvent**: Log who searched what, when, and what was returned
//! - **Audit Log**: Append-only log with rotation and retention
//! - **Compliance Reporting**: Export audit data as JSON/CSV
//! - **GDPR Forget**: Delete all traces of a vector/document
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::observe::audit::{
//!     AuditLog, AuditConfig, SearchAuditEvent, VectorLineage, ComplianceReport,
//! };
//!
//! let mut log = AuditLog::new(AuditConfig::default());
//!
//! // Track vector lineage
//! log.record_lineage(VectorLineage {
//!     vector_id: "vec1".into(),
//!     source_document_id: Some("doc1".into()),
//!     embedding_model: "all-MiniLM-L6-v2".into(),
//!     ..Default::default()
//! });
//!
//! // Log a search event
//! log.record_search(SearchAuditEvent {
//!     query_id: "q1".into(),
//!     user_id: Some("user@example.com".into()),
//!     collection: "documents".into(),
//!     result_ids: vec!["vec1".into(), "vec2".into()],
//!     ..Default::default()
//! });
//!
//! // Generate compliance report
//! let report = log.generate_report("vec1")?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Audit Configuration
// ============================================================================

/// Configuration for the audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Audit level: controls what events are recorded.
    pub level: AuditLevel,
    /// Maximum events to retain in memory.
    pub max_events: usize,
    /// Enable rotation (discard oldest events when full).
    pub rotation_enabled: bool,
    /// Retention period in seconds (0 = forever).
    pub retention_seconds: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            level: AuditLevel::Standard,
            max_events: 100_000,
            rotation_enabled: true,
            retention_seconds: 30 * 24 * 3600, // 30 days
        }
    }
}

/// Audit level controlling event granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditLevel {
    /// No auditing.
    Off,
    /// Record only write operations (inserts, deletes).
    Minimal,
    /// Record writes and searches.
    Standard,
    /// Record everything including metadata access.
    Verbose,
}

// ============================================================================
// Vector Lineage
// ============================================================================

/// Lineage record for a vector, tracking its full provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLineage {
    /// Vector ID.
    pub vector_id: String,
    /// Source document ID.
    pub source_document_id: Option<String>,
    /// Embedding model used.
    pub embedding_model: String,
    /// Model version.
    pub model_version: Option<String>,
    /// Chain of transformations applied.
    pub transformation_chain: Vec<Transformation>,
    /// Timestamp when the vector was created.
    pub created_at: u64,
    /// Collection the vector belongs to.
    pub collection: Option<String>,
    /// Additional metadata.
    pub extra: Option<serde_json::Value>,
}

impl Default for VectorLineage {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            vector_id: String::new(),
            source_document_id: None,
            embedding_model: String::new(),
            model_version: None,
            transformation_chain: Vec::new(),
            created_at: now,
            collection: None,
            extra: None,
        }
    }
}

/// A transformation applied to a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformation {
    /// Transformation type.
    pub kind: TransformationType,
    /// Timestamp when applied.
    pub applied_at: u64,
    /// Description.
    pub description: Option<String>,
}

/// Types of transformations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformationType {
    /// Unit normalization.
    Normalize,
    /// Dimensionality reduction.
    DimensionReduce { from: usize, to: usize },
    /// Quantization.
    Quantize { method: String },
    /// PCA projection.
    PcaProject,
    /// Re-embedding with a different model.
    ReEmbed { model: String },
    /// Custom transformation.
    Custom { name: String },
}

// ============================================================================
// Search Audit Event
// ============================================================================

/// An audit event recording a search operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchAuditEvent {
    /// Unique query ID.
    pub query_id: String,
    /// User or API key that initiated the search.
    pub user_id: Option<String>,
    /// Collection searched.
    pub collection: String,
    /// Number of results requested.
    pub top_k: usize,
    /// IDs of vectors returned.
    pub result_ids: Vec<String>,
    /// Timestamp.
    pub timestamp: u64,
    /// Query latency in milliseconds.
    pub latency_ms: u64,
    /// Whether a filter was used.
    pub had_filter: bool,
    /// IP address or client identifier.
    pub client_id: Option<String>,
}

impl Default for SearchAuditEvent {
    fn default() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            query_id: String::new(),
            user_id: None,
            collection: String::new(),
            top_k: 0,
            result_ids: Vec::new(),
            timestamp: now,
            latency_ms: 0,
            had_filter: false,
            client_id: None,
        }
    }
}

/// General audit event wrapping different event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEvent {
    /// Vector inserted.
    Insert {
        vector_id: String,
        collection: String,
        user_id: Option<String>,
        timestamp: u64,
    },
    /// Vector deleted.
    Delete {
        vector_id: String,
        collection: String,
        user_id: Option<String>,
        timestamp: u64,
    },
    /// Search performed.
    Search(SearchAuditEvent),
    /// Lineage recorded.
    Lineage(VectorLineage),
    /// GDPR forget request.
    Forget {
        vector_id: String,
        user_id: Option<String>,
        timestamp: u64,
    },
    /// Vector accessed (read).
    Access {
        vector_id: String,
        collection: String,
        user_id: Option<String>,
        timestamp: u64,
    },
}

impl AuditEvent {
    fn timestamp(&self) -> u64 {
        match self {
            AuditEvent::Insert { timestamp, .. }
            | AuditEvent::Delete { timestamp, .. }
            | AuditEvent::Forget { timestamp, .. }
            | AuditEvent::Access { timestamp, .. } => *timestamp,
            AuditEvent::Search(e) => e.timestamp,
            AuditEvent::Lineage(l) => l.created_at,
        }
    }
}

// ============================================================================
// Audit Log
// ============================================================================

/// Append-only audit log with rotation and retention.
pub struct AuditLog {
    config: AuditConfig,
    events: VecDeque<AuditEvent>,
    lineage_store: HashMap<String, VectorLineage>,
    /// Cumulative stats.
    stats: AuditStats,
}

/// Audit log statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStats {
    /// Total events recorded.
    pub total_events: u64,
    /// Total searches recorded.
    pub total_searches: u64,
    /// Total inserts recorded.
    pub total_inserts: u64,
    /// Total deletes recorded.
    pub total_deletes: u64,
    /// Total forget requests.
    pub total_forgets: u64,
    /// Events discarded by rotation.
    pub events_rotated: u64,
}

impl AuditLog {
    /// Create a new audit log.
    pub fn new(config: AuditConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
            lineage_store: HashMap::new(),
            stats: AuditStats::default(),
        }
    }

    /// Record a vector lineage.
    pub fn record_lineage(&mut self, lineage: VectorLineage) {
        if self.config.level == AuditLevel::Off {
            return;
        }
        let id = lineage.vector_id.clone();
        self.append_event(AuditEvent::Lineage(lineage.clone()));
        self.lineage_store.insert(id, lineage);
    }

    /// Record a search event.
    pub fn record_search(&mut self, event: SearchAuditEvent) {
        if self.config.level == AuditLevel::Off || self.config.level == AuditLevel::Minimal {
            return;
        }
        self.stats.total_searches += 1;
        self.append_event(AuditEvent::Search(event));
    }

    /// Record a vector insertion.
    pub fn record_insert(
        &mut self,
        vector_id: &str,
        collection: &str,
        user_id: Option<&str>,
    ) {
        if self.config.level == AuditLevel::Off {
            return;
        }
        self.stats.total_inserts += 1;
        let now = Self::now();
        self.append_event(AuditEvent::Insert {
            vector_id: vector_id.to_string(),
            collection: collection.to_string(),
            user_id: user_id.map(String::from),
            timestamp: now,
        });
    }

    /// Record a vector deletion.
    pub fn record_delete(
        &mut self,
        vector_id: &str,
        collection: &str,
        user_id: Option<&str>,
    ) {
        if self.config.level == AuditLevel::Off {
            return;
        }
        self.stats.total_deletes += 1;
        let now = Self::now();
        self.append_event(AuditEvent::Delete {
            vector_id: vector_id.to_string(),
            collection: collection.to_string(),
            user_id: user_id.map(String::from),
            timestamp: now,
        });
    }

    /// Record a vector access.
    pub fn record_access(
        &mut self,
        vector_id: &str,
        collection: &str,
        user_id: Option<&str>,
    ) {
        if self.config.level != AuditLevel::Verbose {
            return;
        }
        let now = Self::now();
        self.append_event(AuditEvent::Access {
            vector_id: vector_id.to_string(),
            collection: collection.to_string(),
            user_id: user_id.map(String::from),
            timestamp: now,
        });
    }

    /// GDPR forget: remove all traces of a vector.
    pub fn forget(&mut self, vector_id: &str, user_id: Option<&str>) -> ForgetResult {
        let now = Self::now();
        let mut events_removed = 0;

        // Remove lineage
        let had_lineage = self.lineage_store.remove(vector_id).is_some();

        // Remove all events referencing this vector
        let before = self.events.len();
        self.events.retain(|event| {
            !matches_vector_id(event, vector_id)
        });
        events_removed = before - self.events.len();

        // Record the forget event itself
        self.stats.total_forgets += 1;
        self.append_event(AuditEvent::Forget {
            vector_id: vector_id.to_string(),
            user_id: user_id.map(String::from),
            timestamp: now,
        });

        ForgetResult {
            vector_id: vector_id.to_string(),
            events_removed,
            lineage_removed: had_lineage,
        }
    }

    /// Get lineage for a vector.
    pub fn get_lineage(&self, vector_id: &str) -> Option<&VectorLineage> {
        self.lineage_store.get(vector_id)
    }

    /// Find all vectors from a source document.
    pub fn vectors_from_source(&self, source_document_id: &str) -> Vec<&VectorLineage> {
        self.lineage_store
            .values()
            .filter(|l| l.source_document_id.as_deref() == Some(source_document_id))
            .collect()
    }

    /// Find all search events that accessed a specific vector.
    pub fn searches_accessing(&self, vector_id: &str) -> Vec<&SearchAuditEvent> {
        self.events
            .iter()
            .filter_map(|event| {
                if let AuditEvent::Search(search) = event {
                    if search.result_ids.contains(&vector_id.to_string()) {
                        return Some(search);
                    }
                }
                None
            })
            .collect()
    }

    /// Find all events for a specific user.
    pub fn events_by_user(&self, user_id: &str) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|event| event_user_id(event) == Some(user_id))
            .collect()
    }

    /// Generate a compliance report for a vector.
    pub fn generate_report(&self, vector_id: &str) -> Result<ComplianceReport> {
        let lineage = self.lineage_store.get(vector_id).cloned();
        let access_events: Vec<AuditEvent> = self
            .events
            .iter()
            .filter(|e| matches_vector_id(e, vector_id))
            .cloned()
            .collect();

        let search_count = access_events
            .iter()
            .filter(|e| matches!(e, AuditEvent::Search(_)))
            .count();

        let users: Vec<String> = access_events
            .iter()
            .filter_map(|e| event_user_id(e).map(String::from))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        Ok(ComplianceReport {
            vector_id: vector_id.to_string(),
            lineage,
            total_events: access_events.len(),
            search_count,
            users_who_accessed: users,
            events: access_events,
        })
    }

    /// Export all events as JSON.
    pub fn export_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.events.iter().collect::<Vec<_>>())
            .map_err(|e| NeedleError::Serialization(e))
    }

    /// Get audit statistics.
    pub fn stats(&self) -> &AuditStats {
        &self.stats
    }

    /// Get the number of events in the log.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Apply retention policy: remove events older than retention_seconds.
    pub fn apply_retention(&mut self) -> usize {
        if self.config.retention_seconds == 0 {
            return 0;
        }
        let cutoff = Self::now().saturating_sub(self.config.retention_seconds);
        let before = self.events.len();
        self.events.retain(|e| e.timestamp() >= cutoff);
        let removed = before - self.events.len();
        self.stats.events_rotated += removed as u64;
        removed
    }

    // Internal helpers

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn append_event(&mut self, event: AuditEvent) {
        self.stats.total_events += 1;
        self.events.push_back(event);

        // Rotation
        if self.config.rotation_enabled && self.events.len() > self.config.max_events {
            let excess = self.events.len() - self.config.max_events;
            for _ in 0..excess {
                self.events.pop_front();
            }
            self.stats.events_rotated += excess as u64;
        }
    }
}

/// Result of a GDPR forget operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgetResult {
    /// Vector ID that was forgotten.
    pub vector_id: String,
    /// Number of audit events removed.
    pub events_removed: usize,
    /// Whether lineage data was removed.
    pub lineage_removed: bool,
}

/// Compliance report for a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    /// Vector ID.
    pub vector_id: String,
    /// Lineage information (if available).
    pub lineage: Option<VectorLineage>,
    /// Total events related to this vector.
    pub total_events: usize,
    /// Number of search events that returned this vector.
    pub search_count: usize,
    /// Users who accessed this vector.
    pub users_who_accessed: Vec<String>,
    /// All events related to this vector.
    pub events: Vec<AuditEvent>,
}

// Helper functions

fn matches_vector_id(event: &AuditEvent, vector_id: &str) -> bool {
    match event {
        AuditEvent::Insert { vector_id: vid, .. }
        | AuditEvent::Delete { vector_id: vid, .. }
        | AuditEvent::Forget { vector_id: vid, .. }
        | AuditEvent::Access { vector_id: vid, .. } => vid == vector_id,
        AuditEvent::Search(search) => search.result_ids.iter().any(|id| id == vector_id),
        AuditEvent::Lineage(l) => l.vector_id == vector_id,
    }
}

fn event_user_id(event: &AuditEvent) -> Option<&str> {
    match event {
        AuditEvent::Insert { user_id, .. }
        | AuditEvent::Delete { user_id, .. }
        | AuditEvent::Forget { user_id, .. }
        | AuditEvent::Access { user_id, .. } => user_id.as_deref(),
        AuditEvent::Search(search) => search.user_id.as_deref(),
        AuditEvent::Lineage(_) => None,
    }
}

// ============================================================================
// Thread-Safe Audit Log Wrapper
// ============================================================================

/// Thread-safe wrapper around `AuditLog` for concurrent access.
///
/// Uses `parking_lot::RwLock` to allow multiple concurrent readers
/// (report generation, queries) with exclusive write access for recording.
pub struct SharedAuditLog {
    inner: parking_lot::RwLock<AuditLog>,
}

impl SharedAuditLog {
    /// Create a new shared audit log.
    pub fn new(config: AuditConfig) -> Self {
        Self {
            inner: parking_lot::RwLock::new(AuditLog::new(config)),
        }
    }

    /// Record a vector lineage entry.
    pub fn record_lineage(&self, lineage: VectorLineage) {
        self.inner.write().record_lineage(lineage);
    }

    /// Record a search audit event.
    pub fn record_search(&self, event: SearchAuditEvent) {
        self.inner.write().record_search(event);
    }

    /// Record a vector insertion.
    pub fn record_insert(&self, vector_id: &str, collection: &str, user_id: Option<&str>) {
        self.inner.write().record_insert(vector_id, collection, user_id);
    }

    /// Record a vector deletion.
    pub fn record_delete(&self, vector_id: &str, collection: &str, user_id: Option<&str>) {
        self.inner.write().record_delete(vector_id, collection, user_id);
    }

    /// Record a vector access.
    pub fn record_access(&self, vector_id: &str, collection: &str, user_id: Option<&str>) {
        self.inner.write().record_access(vector_id, collection, user_id);
    }

    /// GDPR forget.
    pub fn forget(&self, vector_id: &str, user_id: Option<&str>) -> ForgetResult {
        self.inner.write().forget(vector_id, user_id)
    }

    /// Generate a compliance report (read-only).
    pub fn generate_report(&self, vector_id: &str) -> Result<ComplianceReport> {
        self.inner.read().generate_report(vector_id)
    }

    /// Get lineage for a vector (read-only).
    pub fn get_lineage(&self, vector_id: &str) -> Option<VectorLineage> {
        self.inner.read().get_lineage(vector_id).cloned()
    }

    /// Get audit statistics (read-only).
    pub fn stats(&self) -> AuditStats {
        self.inner.read().stats().clone()
    }

    /// Export events as JSON (read-only).
    pub fn export_json(&self) -> Result<String> {
        self.inner.read().export_json()
    }

    /// Event count (read-only).
    pub fn event_count(&self) -> usize {
        self.inner.read().event_count()
    }

    /// Apply retention policy.
    pub fn apply_retention(&self) -> usize {
        self.inner.write().apply_retention()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log() -> AuditLog {
        AuditLog::new(AuditConfig {
            level: AuditLevel::Verbose,
            ..Default::default()
        })
    }

    #[test]
    fn test_record_lineage() {
        let mut log = make_log();
        log.record_lineage(VectorLineage {
            vector_id: "v1".into(),
            source_document_id: Some("doc1".into()),
            embedding_model: "miniLM".into(),
            ..Default::default()
        });

        let lineage = log.get_lineage("v1").expect("lineage");
        assert_eq!(lineage.embedding_model, "miniLM");
        assert_eq!(lineage.source_document_id.as_deref(), Some("doc1"));
    }

    #[test]
    fn test_record_search() {
        let mut log = make_log();
        log.record_search(SearchAuditEvent {
            query_id: "q1".into(),
            user_id: Some("alice".into()),
            collection: "docs".into(),
            top_k: 10,
            result_ids: vec!["v1".into(), "v2".into()],
            ..Default::default()
        });

        assert_eq!(log.stats().total_searches, 1);

        let searches = log.searches_accessing("v1");
        assert_eq!(searches.len(), 1);
        assert_eq!(searches[0].query_id, "q1");
    }

    #[test]
    fn test_record_insert_delete() {
        let mut log = make_log();
        log.record_insert("v1", "docs", Some("alice"));
        log.record_delete("v1", "docs", Some("alice"));

        assert_eq!(log.stats().total_inserts, 1);
        assert_eq!(log.stats().total_deletes, 1);
    }

    #[test]
    fn test_gdpr_forget() {
        let mut log = make_log();
        log.record_lineage(VectorLineage {
            vector_id: "v1".into(),
            embedding_model: "model".into(),
            ..Default::default()
        });
        log.record_insert("v1", "docs", Some("alice"));

        let result = log.forget("v1", Some("alice"));
        assert!(result.lineage_removed);
        assert!(result.events_removed > 0);
        assert!(log.get_lineage("v1").is_none());
    }

    #[test]
    fn test_vectors_from_source() {
        let mut log = make_log();
        log.record_lineage(VectorLineage {
            vector_id: "v1".into(),
            source_document_id: Some("doc1".into()),
            embedding_model: "m".into(),
            ..Default::default()
        });
        log.record_lineage(VectorLineage {
            vector_id: "v2".into(),
            source_document_id: Some("doc1".into()),
            embedding_model: "m".into(),
            ..Default::default()
        });
        log.record_lineage(VectorLineage {
            vector_id: "v3".into(),
            source_document_id: Some("doc2".into()),
            embedding_model: "m".into(),
            ..Default::default()
        });

        let from_doc1 = log.vectors_from_source("doc1");
        assert_eq!(from_doc1.len(), 2);
    }

    #[test]
    fn test_events_by_user() {
        let mut log = make_log();
        log.record_insert("v1", "docs", Some("alice"));
        log.record_insert("v2", "docs", Some("bob"));
        log.record_insert("v3", "docs", Some("alice"));

        let alice_events = log.events_by_user("alice");
        assert_eq!(alice_events.len(), 2);
    }

    #[test]
    fn test_compliance_report() {
        let mut log = make_log();
        log.record_lineage(VectorLineage {
            vector_id: "v1".into(),
            embedding_model: "model".into(),
            ..Default::default()
        });
        log.record_insert("v1", "docs", Some("alice"));
        log.record_search(SearchAuditEvent {
            query_id: "q1".into(),
            user_id: Some("bob".into()),
            collection: "docs".into(),
            result_ids: vec!["v1".into()],
            ..Default::default()
        });

        let report = log.generate_report("v1").expect("report");
        assert!(report.lineage.is_some());
        assert_eq!(report.search_count, 1);
        assert!(report.users_who_accessed.contains(&"alice".to_string()));
        assert!(report.users_who_accessed.contains(&"bob".to_string()));
    }

    #[test]
    fn test_audit_level_off() {
        let mut log = AuditLog::new(AuditConfig {
            level: AuditLevel::Off,
            ..Default::default()
        });
        log.record_insert("v1", "docs", Some("alice"));
        log.record_search(SearchAuditEvent::default());
        assert_eq!(log.event_count(), 0);
    }

    #[test]
    fn test_audit_level_minimal() {
        let mut log = AuditLog::new(AuditConfig {
            level: AuditLevel::Minimal,
            ..Default::default()
        });
        log.record_insert("v1", "docs", Some("alice"));
        log.record_search(SearchAuditEvent::default());
        log.record_access("v1", "docs", Some("alice"));

        // Minimal: only inserts/deletes, no searches or access
        assert_eq!(log.event_count(), 1);
    }

    #[test]
    fn test_rotation() {
        let mut log = AuditLog::new(AuditConfig {
            level: AuditLevel::Verbose,
            max_events: 5,
            rotation_enabled: true,
            ..Default::default()
        });

        for i in 0..10 {
            log.record_insert(&format!("v{i}"), "docs", None);
        }

        assert_eq!(log.event_count(), 5);
        assert!(log.stats().events_rotated > 0);
    }

    #[test]
    fn test_export_json() {
        let mut log = make_log();
        log.record_insert("v1", "docs", None);
        let json = log.export_json().expect("json");
        assert!(json.contains("v1"));
    }

    #[test]
    fn test_shared_audit_log() {
        let shared = SharedAuditLog::new(AuditConfig {
            level: AuditLevel::Verbose,
            ..Default::default()
        });

        shared.record_insert("v1", "docs", Some("alice"));
        shared.record_search(SearchAuditEvent {
            query_id: "q1".into(),
            user_id: Some("bob".into()),
            collection: "docs".into(),
            result_ids: vec!["v1".into()],
            ..Default::default()
        });

        assert_eq!(shared.event_count(), 2);
        let stats = shared.stats();
        assert_eq!(stats.total_inserts, 1);
        assert_eq!(stats.total_searches, 1);

        let report = shared.generate_report("v1").expect("report");
        assert_eq!(report.total_events, 2);
    }

    #[test]
    fn test_shared_audit_log_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let shared = Arc::new(SharedAuditLog::new(AuditConfig {
            level: AuditLevel::Standard,
            ..Default::default()
        }));

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let log = Arc::clone(&shared);
                thread::spawn(move || {
                    for i in 0..50 {
                        log.record_insert(
                            &format!("v{t}_{i}"),
                            "docs",
                            Some(&format!("user{t}")),
                        );
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread join");
        }

        assert_eq!(shared.event_count(), 200);
    }
}
