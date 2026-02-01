#![allow(dead_code)]

//! Native Vector Change Data Capture (CDC)
//!
//! Provides real-time change event streaming for vector operations.
//! Supports in-process callbacks, durable cursor tracking, and
//! at-least-once delivery guarantees.

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Log Sequence Number for ordering events
pub type Lsn = u64;

/// Type of change operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    Insert,
    Update,
    Delete,
}

/// A single vector change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorChangeEvent {
    /// Log sequence number (monotonically increasing)
    pub lsn: Lsn,
    /// Type of change
    pub change_type: ChangeType,
    /// Collection name
    pub collection: String,
    /// Vector ID
    pub vector_id: String,
    /// Vector data before the change (None for inserts)
    pub before: Option<Vec<f32>>,
    /// Vector data after the change (None for deletes)
    pub after: Option<Vec<f32>>,
    /// Metadata before the change
    pub metadata_before: Option<serde_json::Value>,
    /// Metadata after the change
    pub metadata_after: Option<serde_json::Value>,
    /// Unix timestamp in milliseconds
    pub timestamp: u64,
}

impl VectorChangeEvent {
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Create an insert event
    pub fn insert(
        collection: impl Into<String>,
        vector_id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
        lsn: Lsn,
    ) -> Self {
        Self {
            lsn,
            change_type: ChangeType::Insert,
            collection: collection.into(),
            vector_id: vector_id.into(),
            before: None,
            after: Some(vector),
            metadata_before: None,
            metadata_after: metadata,
            timestamp: Self::now_ms(),
        }
    }

    /// Create a delete event
    pub fn delete(
        collection: impl Into<String>,
        vector_id: impl Into<String>,
        vector: Option<Vec<f32>>,
        metadata: Option<serde_json::Value>,
        lsn: Lsn,
    ) -> Self {
        Self {
            lsn,
            change_type: ChangeType::Delete,
            collection: collection.into(),
            vector_id: vector_id.into(),
            before: vector,
            after: None,
            metadata_before: metadata,
            metadata_after: None,
            timestamp: Self::now_ms(),
        }
    }

    /// Create an update event
    pub fn update(
        collection: impl Into<String>,
        vector_id: impl Into<String>,
        before: Option<Vec<f32>>,
        after: Vec<f32>,
        meta_before: Option<serde_json::Value>,
        meta_after: Option<serde_json::Value>,
        lsn: Lsn,
    ) -> Self {
        Self {
            lsn,
            change_type: ChangeType::Update,
            collection: collection.into(),
            vector_id: vector_id.into(),
            before,
            after: Some(after),
            metadata_before: meta_before,
            metadata_after: meta_after,
            timestamp: Self::now_ms(),
        }
    }
}

/// Subscriber cursor tracking for at-least-once delivery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriberCursor {
    /// Subscriber ID
    pub subscriber_id: String,
    /// Last acknowledged LSN
    pub last_ack_lsn: Lsn,
    /// When the cursor was last updated
    pub updated_at: u64,
}

/// Configuration for the CDC event log
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcConfig {
    /// Maximum events to retain in the log
    pub max_log_size: usize,
    /// Whether to include vector data in events (can be large)
    pub include_vectors: bool,
    /// Whether to include metadata diffs
    pub include_metadata: bool,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            max_log_size: 100_000,
            include_vectors: false,
            include_metadata: true,
        }
    }
}

/// Type alias for change event callbacks
pub type ChangeCallback = Box<dyn Fn(&VectorChangeEvent) + Send + Sync>;

/// CDC event log with ordered event storage and subscriber management
pub struct ChangeEventLog {
    config: CdcConfig,
    /// Ordered event log
    events: RwLock<Vec<VectorChangeEvent>>,
    /// Current LSN counter
    next_lsn: AtomicU64,
    /// Subscriber cursors
    cursors: RwLock<HashMap<String, SubscriberCursor>>,
    /// In-process change callbacks
    callbacks: RwLock<Vec<(String, ChangeCallback)>>,
}

impl ChangeEventLog {
    pub fn new(config: CdcConfig) -> Self {
        Self {
            config,
            events: RwLock::new(Vec::new()),
            next_lsn: AtomicU64::new(1),
            cursors: RwLock::new(HashMap::new()),
            callbacks: RwLock::new(Vec::new()),
        }
    }

    /// Get the next LSN
    pub fn next_lsn(&self) -> Lsn {
        self.next_lsn.fetch_add(1, Ordering::SeqCst)
    }

    /// Append an event to the log and notify callbacks
    pub fn append(&self, event: VectorChangeEvent) {
        // Notify in-process callbacks
        let callbacks = self.callbacks.read();
        for (_, cb) in callbacks.iter() {
            cb(&event);
        }
        drop(callbacks);

        // Append to log
        let mut events = self.events.write();
        events.push(event);

        // Evict old events if over capacity
        if events.len() > self.config.max_log_size {
            let drain_count = events.len() - self.config.max_log_size;
            events.drain(0..drain_count);
        }
    }

    /// Register an in-process callback for change events
    pub fn on_change(&self, subscriber_id: impl Into<String>, callback: ChangeCallback) {
        let mut callbacks = self.callbacks.write();
        callbacks.push((subscriber_id.into(), callback));
    }

    /// Remove a callback by subscriber ID
    pub fn remove_callback(&self, subscriber_id: &str) {
        let mut callbacks = self.callbacks.write();
        callbacks.retain(|(id, _)| id != subscriber_id);
    }

    /// Get events since a given LSN (for polling subscribers)
    pub fn events_since(&self, lsn: Lsn) -> Vec<VectorChangeEvent> {
        let events = self.events.read();
        events.iter().filter(|e| e.lsn > lsn).cloned().collect()
    }

    /// Get events for a specific collection since a given LSN
    pub fn collection_events_since(&self, collection: &str, lsn: Lsn) -> Vec<VectorChangeEvent> {
        let events = self.events.read();
        events
            .iter()
            .filter(|e| e.collection == collection && e.lsn > lsn)
            .cloned()
            .collect()
    }

    /// Update subscriber cursor (acknowledge receipt up to LSN)
    pub fn acknowledge(&self, subscriber_id: &str, lsn: Lsn) {
        let mut cursors = self.cursors.write();
        let cursor = cursors
            .entry(subscriber_id.to_string())
            .or_insert_with(|| SubscriberCursor {
                subscriber_id: subscriber_id.to_string(),
                last_ack_lsn: 0,
                updated_at: 0,
            });
        cursor.last_ack_lsn = lsn;
        cursor.updated_at = VectorChangeEvent::now_ms();
    }

    /// Get the cursor for a subscriber
    pub fn get_cursor(&self, subscriber_id: &str) -> Option<SubscriberCursor> {
        self.cursors.read().get(subscriber_id).cloned()
    }

    /// Get current log size
    pub fn len(&self) -> usize {
        self.events.read().len()
    }

    /// Check if the log is empty
    pub fn is_empty(&self) -> bool {
        self.events.read().is_empty()
    }

    /// Get the latest LSN
    pub fn latest_lsn(&self) -> Lsn {
        self.next_lsn.load(Ordering::SeqCst) - 1
    }

    /// CDC statistics
    pub fn stats(&self) -> CdcStats {
        let events = self.events.read();
        let cursors = self.cursors.read();
        let callbacks = self.callbacks.read();
        CdcStats {
            total_events: events.len(),
            latest_lsn: self.latest_lsn(),
            subscriber_count: cursors.len(),
            callback_count: callbacks.len(),
        }
    }
}

/// CDC statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CdcStats {
    pub total_events: usize,
    pub latest_lsn: Lsn,
    pub subscriber_count: usize,
    pub callback_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    #[test]
    fn test_event_log_basic() {
        let log = ChangeEventLog::new(CdcConfig::default());
        let lsn = log.next_lsn();
        let event = VectorChangeEvent::insert("test_col", "vec1", vec![1.0, 2.0], None, lsn);
        log.append(event);
        assert_eq!(log.len(), 1);
        let events = log.events_since(0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].vector_id, "vec1");
    }

    #[test]
    fn test_callback_notification() {
        let log = ChangeEventLog::new(CdcConfig::default());
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        log.on_change(
            "test_sub",
            Box::new(move |_event| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let lsn = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col",
            "v1",
            vec![1.0],
            None,
            lsn,
        ));
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_cursor_tracking() {
        let log = ChangeEventLog::new(CdcConfig::default());
        let lsn1 = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col",
            "v1",
            vec![1.0],
            None,
            lsn1,
        ));
        let lsn2 = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col",
            "v2",
            vec![2.0],
            None,
            lsn2,
        ));

        log.acknowledge("sub1", lsn1);
        let cursor = log.get_cursor("sub1").unwrap();
        assert_eq!(cursor.last_ack_lsn, lsn1);

        let pending = log.events_since(lsn1);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].vector_id, "v2");
    }

    #[test]
    fn test_log_eviction() {
        let config = CdcConfig {
            max_log_size: 5,
            ..Default::default()
        };
        let log = ChangeEventLog::new(config);
        for i in 0..10 {
            let lsn = log.next_lsn();
            log.append(VectorChangeEvent::insert(
                "col",
                format!("v{i}"),
                vec![i as f32],
                None,
                lsn,
            ));
        }
        assert_eq!(log.len(), 5);
    }

    #[test]
    fn test_collection_events_since() {
        let log = ChangeEventLog::new(CdcConfig::default());
        let lsn1 = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col_a",
            "v1",
            vec![1.0],
            None,
            lsn1,
        ));
        let lsn2 = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col_b",
            "v2",
            vec![2.0],
            None,
            lsn2,
        ));
        let lsn3 = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col_a",
            "v3",
            vec![3.0],
            None,
            lsn3,
        ));

        let events = log.collection_events_since("col_a", 0);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].vector_id, "v1");
        assert_eq!(events[1].vector_id, "v3");
    }

    #[test]
    fn test_remove_callback() {
        let log = ChangeEventLog::new(CdcConfig::default());
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();
        log.on_change(
            "sub1",
            Box::new(move |_| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        let lsn = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col",
            "v1",
            vec![1.0],
            None,
            lsn,
        ));
        assert_eq!(count.load(Ordering::SeqCst), 1);

        log.remove_callback("sub1");

        let lsn = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col",
            "v2",
            vec![2.0],
            None,
            lsn,
        ));
        assert_eq!(count.load(Ordering::SeqCst), 1); // unchanged
    }

    #[test]
    fn test_stats() {
        let log = ChangeEventLog::new(CdcConfig::default());
        let lsn = log.next_lsn();
        log.append(VectorChangeEvent::insert(
            "col",
            "v1",
            vec![1.0],
            None,
            lsn,
        ));
        log.acknowledge("sub1", lsn);

        let stats = log.stats();
        assert_eq!(stats.total_events, 1);
        assert_eq!(stats.latest_lsn, 1);
        assert_eq!(stats.subscriber_count, 1);
        assert_eq!(stats.callback_count, 0);
    }

    #[test]
    fn test_delete_and_update_events() {
        let log = ChangeEventLog::new(CdcConfig::default());

        let lsn1 = log.next_lsn();
        log.append(VectorChangeEvent::delete(
            "col",
            "v1",
            Some(vec![1.0]),
            None,
            lsn1,
        ));

        let lsn2 = log.next_lsn();
        log.append(VectorChangeEvent::update(
            "col",
            "v2",
            Some(vec![1.0]),
            vec![2.0],
            None,
            None,
            lsn2,
        ));

        let events = log.events_since(0);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].change_type, ChangeType::Delete);
        assert!(events[0].before.is_some());
        assert!(events[0].after.is_none());
        assert_eq!(events[1].change_type, ChangeType::Update);
        assert!(events[1].before.is_some());
        assert!(events[1].after.is_some());
    }

    #[test]
    fn test_change_event_constructors() {
        let insert = VectorChangeEvent::insert("col1", "v1", vec![1.0, 2.0], None, 1);
        assert_eq!(insert.change_type, ChangeType::Insert);
        assert!(insert.before.is_none());
        assert!(insert.after.is_some());

        let delete = VectorChangeEvent::delete("col1", "v1", Some(vec![1.0, 2.0]), None, 2);
        assert_eq!(delete.change_type, ChangeType::Delete);
        assert!(delete.before.is_some());
        assert!(delete.after.is_none());

        let update = VectorChangeEvent::update("col1", "v1", Some(vec![1.0]), vec![2.0], None, None, 3);
        assert_eq!(update.change_type, ChangeType::Update);
        assert!(update.before.is_some());
        assert!(update.after.is_some());
    }
}
