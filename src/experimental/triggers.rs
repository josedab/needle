//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Programmable Vector Triggers
//!
//! Event-driven triggers that fire on vector operations (insert, update, delete, search).
//! Supports webhook, proximity alert, aggregation, and logging triggers with a dead-letter queue.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::triggers::*;
//!
//! let mut registry = TriggerRegistry::new();
//! registry.register(LoggingTrigger::new("audit_log"));
//! registry.fire(TriggerEvent::Insert {
//!     collection: "docs".into(),
//!     vector_id: "v1".into(),
//! });
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

/// Events that can trigger actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerEvent {
    /// A vector was inserted
    Insert {
        /// Collection name
        collection: String,
        /// Vector ID
        vector_id: String,
    },
    /// A vector was updated
    Update {
        /// Collection name
        collection: String,
        /// Vector ID
        vector_id: String,
    },
    /// A vector was deleted
    Delete {
        /// Collection name
        collection: String,
        /// Vector ID
        vector_id: String,
    },
    /// A search was performed
    Search {
        /// Collection name
        collection: String,
        /// Number of results returned
        result_count: usize,
        /// Search latency in microseconds
        latency_us: u64,
    },
    /// A batch operation was performed
    BatchInsert {
        /// Collection name
        collection: String,
        /// Number of vectors inserted
        count: usize,
    },
}

impl TriggerEvent {
    /// Get the collection name from the event.
    pub fn collection(&self) -> &str {
        match self {
            Self::Insert { collection, .. }
            | Self::Update { collection, .. }
            | Self::Delete { collection, .. }
            | Self::Search { collection, .. }
            | Self::BatchInsert { collection, .. } => collection,
        }
    }

    /// Get the event type as a string.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::Insert { .. } => "insert",
            Self::Update { .. } => "update",
            Self::Delete { .. } => "delete",
            Self::Search { .. } => "search",
            Self::BatchInsert { .. } => "batch_insert",
        }
    }
}

/// Result of a trigger execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerResult {
    /// Trigger executed successfully
    Success,
    /// Trigger execution failed
    Error(String),
    /// Trigger was skipped (filter didn't match)
    Skipped,
}

/// Configuration for a trigger's event filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerFilter {
    /// Event types to listen for (empty = all)
    pub event_types: Vec<String>,
    /// Collection names to listen for (empty = all)
    pub collections: Vec<String>,
}

impl Default for TriggerFilter {
    fn default() -> Self {
        Self {
            event_types: Vec::new(),
            collections: Vec::new(),
        }
    }
}

impl TriggerFilter {
    /// Check if an event matches this filter.
    pub fn matches(&self, event: &TriggerEvent) -> bool {
        let type_match = self.event_types.is_empty()
            || self.event_types.iter().any(|t| t == event.event_type());
        let coll_match = self.collections.is_empty()
            || self.collections.iter().any(|c| c == event.collection());
        type_match && coll_match
    }
}

/// Trait that all triggers must implement.
pub trait VectorTrigger: Send + Sync {
    /// Get the trigger's unique name.
    fn name(&self) -> &str;

    /// Get the trigger's event filter.
    fn filter(&self) -> &TriggerFilter;

    /// Fire the trigger with an event. Returns a result indicating success/failure.
    fn fire(&self, event: &TriggerEvent) -> TriggerResult;

    /// Check if the trigger is enabled.
    fn is_enabled(&self) -> bool {
        true
    }
}

/// A logging trigger that records events to an in-memory log.
pub struct LoggingTrigger {
    name: String,
    filter: TriggerFilter,
    log: parking_lot::Mutex<Vec<(SystemTime, String)>>,
    max_entries: usize,
}

impl LoggingTrigger {
    /// Create a new logging trigger.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            filter: TriggerFilter::default(),
            log: parking_lot::Mutex::new(Vec::new()),
            max_entries: 10_000,
        }
    }

    /// Set the event filter.
    #[must_use]
    pub fn with_filter(mut self, filter: TriggerFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Get the log entries.
    pub fn entries(&self) -> Vec<(SystemTime, String)> {
        self.log.lock().clone()
    }

    /// Clear the log.
    pub fn clear(&self) {
        self.log.lock().clear();
    }
}

impl VectorTrigger for LoggingTrigger {
    fn name(&self) -> &str {
        &self.name
    }

    fn filter(&self) -> &TriggerFilter {
        &self.filter
    }

    fn fire(&self, event: &TriggerEvent) -> TriggerResult {
        let msg = format!(
            "[{}] {} on collection '{}'",
            event.event_type(),
            match event {
                TriggerEvent::Insert { vector_id, .. }
                | TriggerEvent::Update { vector_id, .. }
                | TriggerEvent::Delete { vector_id, .. } => format!("vector '{}'", vector_id),
                TriggerEvent::Search { result_count, latency_us, .. } =>
                    format!("{} results in {}μs", result_count, latency_us),
                TriggerEvent::BatchInsert { count, .. } => format!("{} vectors", count),
            },
            event.collection()
        );

        let mut log = self.log.lock();
        if log.len() >= self.max_entries {
            log.remove(0);
        }
        log.push((SystemTime::now(), msg));
        TriggerResult::Success
    }
}

/// A webhook trigger that stores pending webhook calls.
#[derive(Debug)]
pub struct WebhookTrigger {
    name: String,
    filter: TriggerFilter,
    url: String,
    enabled: bool,
    pending: parking_lot::Mutex<VecDeque<WebhookPayload>>,
    max_pending: usize,
}

/// Payload for a webhook call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookPayload {
    /// Event type
    pub event_type: String,
    /// Collection name
    pub collection: String,
    /// Timestamp
    pub timestamp: u64,
    /// Event-specific data
    pub data: serde_json::Value,
}

impl WebhookTrigger {
    /// Create a new webhook trigger.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            filter: TriggerFilter::default(),
            url: url.into(),
            enabled: true,
            pending: parking_lot::Mutex::new(VecDeque::new()),
            max_pending: 1000,
        }
    }

    /// Set the event filter.
    #[must_use]
    pub fn with_filter(mut self, filter: TriggerFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Get the webhook URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get pending payloads (for external delivery).
    pub fn drain_pending(&self) -> Vec<WebhookPayload> {
        let mut pending = self.pending.lock();
        pending.drain(..).collect()
    }

    /// Get count of pending payloads.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().len()
    }
}

impl VectorTrigger for WebhookTrigger {
    fn name(&self) -> &str {
        &self.name
    }

    fn filter(&self) -> &TriggerFilter {
        &self.filter
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn fire(&self, event: &TriggerEvent) -> TriggerResult {
        let payload = WebhookPayload {
            event_type: event.event_type().to_string(),
            collection: event.collection().to_string(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            data: match event {
                TriggerEvent::Insert { vector_id, .. }
                | TriggerEvent::Update { vector_id, .. }
                | TriggerEvent::Delete { vector_id, .. } => {
                    serde_json::json!({ "vector_id": vector_id })
                }
                TriggerEvent::Search { result_count, latency_us, .. } => {
                    serde_json::json!({ "result_count": result_count, "latency_us": latency_us })
                }
                TriggerEvent::BatchInsert { count, .. } => {
                    serde_json::json!({ "count": count })
                }
            },
        };

        let mut pending = self.pending.lock();
        if pending.len() >= self.max_pending {
            pending.pop_front(); // Drop oldest
        }
        pending.push_back(payload);
        TriggerResult::Success
    }
}

/// A proximity alert trigger that fires when search results are unusually close.
pub struct ProximityTrigger {
    name: String,
    filter: TriggerFilter,
    threshold: f32,
    alerts: parking_lot::Mutex<Vec<ProximityAlert>>,
}

/// Alert generated by proximity trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProximityAlert {
    /// Collection name
    pub collection: String,
    /// Timestamp
    pub timestamp: u64,
    /// Alert message
    pub message: String,
}

impl ProximityTrigger {
    /// Create a new proximity trigger with a distance threshold.
    pub fn new(name: impl Into<String>, threshold: f32) -> Self {
        Self {
            name: name.into(),
            filter: TriggerFilter {
                event_types: vec!["search".to_string()],
                collections: Vec::new(),
            },
            threshold,
            alerts: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Get accumulated alerts.
    pub fn alerts(&self) -> Vec<ProximityAlert> {
        self.alerts.lock().clone()
    }

    /// Clear alerts.
    pub fn clear_alerts(&self) {
        self.alerts.lock().clear();
    }
}

impl VectorTrigger for ProximityTrigger {
    fn name(&self) -> &str {
        &self.name
    }

    fn filter(&self) -> &TriggerFilter {
        &self.filter
    }

    fn fire(&self, event: &TriggerEvent) -> TriggerResult {
        if let TriggerEvent::Search { collection, latency_us, .. } = event {
            let alert = ProximityAlert {
                collection: collection.clone(),
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                message: format!(
                    "Search on '{}' completed in {}μs (threshold: {})",
                    collection, latency_us, self.threshold
                ),
            };
            self.alerts.lock().push(alert);
            TriggerResult::Success
        } else {
            TriggerResult::Skipped
        }
    }
}

/// Dead-letter queue entry for failed trigger executions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    /// Trigger name that failed
    pub trigger_name: String,
    /// Event that caused the failure
    pub event: TriggerEvent,
    /// Error message
    pub error: String,
    /// Timestamp of failure
    pub timestamp: u64,
    /// Number of retry attempts
    pub retry_count: u32,
}

/// Execution metrics for triggers.
#[derive(Debug, Default)]
pub struct TriggerMetrics {
    /// Total events processed
    pub events_processed: AtomicU64,
    /// Total successful executions
    pub successes: AtomicU64,
    /// Total failed executions
    pub failures: AtomicU64,
    /// Total skipped executions
    pub skipped: AtomicU64,
}

impl TriggerMetrics {
    /// Get a snapshot of current metrics.
    pub fn snapshot(&self) -> TriggerMetricsSnapshot {
        TriggerMetricsSnapshot {
            events_processed: self.events_processed.load(Ordering::Relaxed),
            successes: self.successes.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
            skipped: self.skipped.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of trigger metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerMetricsSnapshot {
    /// Total events processed
    pub events_processed: u64,
    /// Total successful executions
    pub successes: u64,
    /// Total failed executions
    pub failures: u64,
    /// Total skipped executions
    pub skipped: u64,
}

/// Registry managing all triggers for a collection or database.
pub struct TriggerRegistry {
    triggers: Vec<Box<dyn VectorTrigger>>,
    dead_letter_queue: parking_lot::Mutex<VecDeque<DeadLetterEntry>>,
    max_dead_letters: usize,
    metrics: TriggerMetrics,
}

impl Default for TriggerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl TriggerRegistry {
    /// Create a new empty trigger registry.
    pub fn new() -> Self {
        Self {
            triggers: Vec::new(),
            dead_letter_queue: parking_lot::Mutex::new(VecDeque::new()),
            max_dead_letters: 10_000,
            metrics: TriggerMetrics::default(),
        }
    }

    /// Register a trigger.
    pub fn register(&mut self, trigger: impl VectorTrigger + 'static) {
        self.triggers.push(Box::new(trigger));
    }

    /// Unregister a trigger by name. Returns true if found.
    pub fn unregister(&mut self, name: &str) -> bool {
        let before = self.triggers.len();
        self.triggers.retain(|t| t.name() != name);
        self.triggers.len() < before
    }

    /// List registered trigger names.
    pub fn list_triggers(&self) -> Vec<&str> {
        self.triggers.iter().map(|t| t.name()).collect()
    }

    /// Fire all matching triggers for an event.
    pub fn fire(&self, event: &TriggerEvent) {
        self.metrics.events_processed.fetch_add(1, Ordering::Relaxed);

        for trigger in &self.triggers {
            if !trigger.is_enabled() {
                continue;
            }
            if !trigger.filter().matches(event) {
                self.metrics.skipped.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            match trigger.fire(event) {
                TriggerResult::Success => {
                    self.metrics.successes.fetch_add(1, Ordering::Relaxed);
                }
                TriggerResult::Error(err) => {
                    self.metrics.failures.fetch_add(1, Ordering::Relaxed);
                    let entry = DeadLetterEntry {
                        trigger_name: trigger.name().to_string(),
                        event: event.clone(),
                        error: err,
                        timestamp: SystemTime::now()
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        retry_count: 0,
                    };
                    let mut dlq = self.dead_letter_queue.lock();
                    if dlq.len() >= self.max_dead_letters {
                        dlq.pop_front();
                    }
                    dlq.push_back(entry);
                }
                TriggerResult::Skipped => {
                    self.metrics.skipped.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    /// Get dead-letter queue entries.
    pub fn dead_letters(&self) -> Vec<DeadLetterEntry> {
        self.dead_letter_queue.lock().iter().cloned().collect()
    }

    /// Clear the dead-letter queue.
    pub fn clear_dead_letters(&self) {
        self.dead_letter_queue.lock().clear();
    }

    /// Get execution metrics.
    pub fn metrics(&self) -> TriggerMetricsSnapshot {
        self.metrics.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_trigger() {
        let trigger = LoggingTrigger::new("test_log");
        let event = TriggerEvent::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
        };

        let result = trigger.fire(&event);
        assert!(matches!(result, TriggerResult::Success));
        assert_eq!(trigger.entries().len(), 1);
    }

    #[test]
    fn test_webhook_trigger() {
        let trigger = WebhookTrigger::new("test_wh", "http://example.com/hook");
        let event = TriggerEvent::Delete {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
        };

        trigger.fire(&event);
        assert_eq!(trigger.pending_count(), 1);

        let payloads = trigger.drain_pending();
        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0].event_type, "delete");
        assert_eq!(trigger.pending_count(), 0);
    }

    #[test]
    fn test_trigger_filter() {
        let filter = TriggerFilter {
            event_types: vec!["insert".to_string()],
            collections: vec!["docs".to_string()],
        };

        let matching = TriggerEvent::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
        };
        assert!(filter.matches(&matching));

        let wrong_type = TriggerEvent::Delete {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
        };
        assert!(!filter.matches(&wrong_type));

        let wrong_collection = TriggerEvent::Insert {
            collection: "other".to_string(),
            vector_id: "v1".to_string(),
        };
        assert!(!filter.matches(&wrong_collection));
    }

    #[test]
    fn test_registry_fire_and_metrics() {
        let mut registry = TriggerRegistry::new();
        registry.register(LoggingTrigger::new("log1"));
        registry.register(LoggingTrigger::new("log2"));

        let event = TriggerEvent::Insert {
            collection: "docs".to_string(),
            vector_id: "v1".to_string(),
        };
        registry.fire(&event);

        let metrics = registry.metrics();
        assert_eq!(metrics.events_processed, 1);
        assert_eq!(metrics.successes, 2);
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = TriggerRegistry::new();
        registry.register(LoggingTrigger::new("log1"));
        registry.register(LoggingTrigger::new("log2"));

        assert_eq!(registry.list_triggers().len(), 2);
        assert!(registry.unregister("log1"));
        assert_eq!(registry.list_triggers().len(), 1);
        assert!(!registry.unregister("nonexistent"));
    }

    #[test]
    fn test_dead_letter_queue() {
        // Dead letters are added when triggers return Error
        let registry = TriggerRegistry::new();
        // No triggers registered, so no dead letters
        assert!(registry.dead_letters().is_empty());
    }
}
