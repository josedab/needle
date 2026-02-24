//! Collection-level Change Data Capture (CDC)
//!
//! Provides an append-only event log per collection with sequence numbers
//! and cursor-based resumption for reactive architectures and real-time sync.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::collection::cdc::{CdcLog, CdcEventType};
//!
//! let mut log = CdcLog::new(1000);
//! log.append(CdcEventType::Insert, "doc1", None);
//!
//! // Resume from a cursor
//! let events = log.events_since(0, 100);
//! let cursor = events.last().map(|e| e.sequence);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single CDC event representing a mutation in the collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcEvent {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Unix epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Type of operation.
    pub event_type: CdcEventType,
    /// Affected vector ID.
    pub vector_id: String,
    /// Optional metadata snapshot (e.g., for inserts/updates).
    pub metadata: Option<serde_json::Value>,
}

/// Types of CDC events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdcEventType {
    /// A new vector was inserted.
    Insert,
    /// An existing vector was updated.
    Update,
    /// A vector was deleted.
    Delete,
}

/// Configuration for the CDC log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcConfig {
    /// Maximum number of events to retain (0 = unlimited).
    pub max_events: usize,
    /// Whether CDC is enabled.
    pub enabled: bool,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            max_events: 10_000,
            enabled: false,
        }
    }
}

/// Append-only event log for collection-level CDC.
///
/// Tracks all mutations with monotonic sequence numbers, supporting
/// cursor-based resumption and configurable retention.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcLog {
    events: VecDeque<CdcEvent>,
    next_sequence: u64,
    max_events: usize,
}

impl CdcLog {
    /// Create a new CDC log with the given retention limit.
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::new(),
            next_sequence: 1,
            max_events,
        }
    }

    /// Append a new event to the log.
    pub fn append(
        &mut self,
        event_type: CdcEventType,
        vector_id: &str,
        metadata: Option<serde_json::Value>,
    ) -> u64 {
        let seq = self.next_sequence;
        self.next_sequence += 1;

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.events.push_back(CdcEvent {
            sequence: seq,
            timestamp_ms,
            event_type,
            vector_id: vector_id.to_string(),
            metadata,
        });

        // Enforce retention
        if self.max_events > 0 {
            while self.events.len() > self.max_events {
                self.events.pop_front();
            }
        }

        seq
    }

    /// Get all events with sequence > `after_sequence`, up to `limit`.
    pub fn events_since(&self, after_sequence: u64, limit: usize) -> Vec<&CdcEvent> {
        self.events
            .iter()
            .filter(|e| e.sequence > after_sequence)
            .take(limit)
            .collect()
    }

    /// Get the current head sequence number (the latest event's sequence).
    pub fn head_sequence(&self) -> u64 {
        self.next_sequence.saturating_sub(1)
    }

    /// Get the earliest available sequence number (for cursor validation).
    pub fn earliest_sequence(&self) -> u64 {
        self.events.front().map_or(0, |e| e.sequence)
    }

    /// Total number of events currently retained.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Compact the log, removing events older than `before_sequence`.
    pub fn compact(&mut self, before_sequence: u64) {
        while let Some(front) = self.events.front() {
            if front.sequence < before_sequence {
                self.events.pop_front();
            } else {
                break;
            }
        }
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for CdcLog {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_cdc_log_append_and_query() {
        let mut log = CdcLog::new(100);
        let s1 = log.append(CdcEventType::Insert, "v1", None);
        let s2 = log.append(CdcEventType::Update, "v1", None);
        let s3 = log.append(CdcEventType::Delete, "v2", None);

        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        assert_eq!(log.len(), 3);

        // Resume from cursor
        let events = log.events_since(1, 10);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].sequence, 2);
        assert_eq!(events[1].sequence, 3);
    }

    #[test]
    fn test_cdc_log_retention() {
        let mut log = CdcLog::new(3);
        for i in 0..5 {
            log.append(CdcEventType::Insert, &format!("v{i}"), None);
        }
        assert_eq!(log.len(), 3);
        assert_eq!(log.earliest_sequence(), 3);
    }

    #[test]
    fn test_cdc_log_compact() {
        let mut log = CdcLog::new(100);
        for i in 0..10 {
            log.append(CdcEventType::Insert, &format!("v{i}"), None);
        }
        log.compact(5);
        assert_eq!(log.len(), 6); // sequences 5..10
        assert_eq!(log.earliest_sequence(), 5);
    }

    #[test]
    fn test_cdc_log_head_sequence() {
        let mut log = CdcLog::new(100);
        assert_eq!(log.head_sequence(), 0);
        log.append(CdcEventType::Insert, "v1", None);
        assert_eq!(log.head_sequence(), 1);
    }

    /// Integration test: verify CDC events are recorded on Collection insert/update/delete
    #[test]
    fn test_cdc_collection_integration() {
        use crate::Collection;

        let mut col = Collection::with_dimensions("test", 4);
        col.enable_cdc(1000);
        assert!(col.cdc_enabled());
        assert_eq!(col.cdc_head_sequence(), 0);

        // Insert triggers CDC
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.cdc_head_sequence(), 1);

        let events = col.cdc_events_since(0, 10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, CdcEventType::Insert);
        assert_eq!(events[0].vector_id, "v1");

        // Update triggers CDC
        col.update("v1", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.cdc_head_sequence(), 2);

        let events = col.cdc_events_since(1, 10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, CdcEventType::Update);

        // Delete triggers CDC
        col.delete("v1").unwrap();
        assert_eq!(col.cdc_head_sequence(), 3);

        let events = col.cdc_events_since(2, 10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, CdcEventType::Delete);

        // Total events
        let all = col.cdc_events_since(0, 100);
        assert_eq!(all.len(), 3);
    }

    /// Test CDC with disabled state
    #[test]
    fn test_cdc_disabled_noop() {
        use crate::Collection;

        let mut col = Collection::with_dimensions("test", 4);
        // CDC not enabled - operations should work without CDC
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(col.cdc_head_sequence(), 0);
        assert!(col.cdc_events_since(0, 10).is_empty());
    }

    /// Test CDC via CollectionRef (Database-level)
    #[test]
    fn test_cdc_via_collection_ref() {
        use crate::Database;

        let db = Database::in_memory();
        db.create_collection("docs", 4).unwrap();

        let coll = db.collection("docs").unwrap();
        coll.enable_cdc(100).unwrap();

        coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert_eq!(coll.cdc_head_sequence(), 1);

        let events = coll.cdc_events_since(0, 10);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, CdcEventType::Insert);
    }
}
