#![allow(clippy::unwrap_used)]
//! Collection Change Streams (SSE)
//!
//! Server-Sent Events endpoint for real-time collection change notifications.
//! Clients subscribe to insert/update/delete events with filtering and
//! offset-based resume support.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::change_stream::{
//!     ChangeStreamService, StreamConfig, ChangeEvent, EventFilter,
//!     Subscriber, SubscriberHandle,
//! };
//!
//! let mut svc = ChangeStreamService::new(StreamConfig::default());
//!
//! // Subscribe to a collection
//! let handle = svc.subscribe("docs", EventFilter::all());
//!
//! // Emit events (called by insert/delete handlers)
//! svc.emit_insert("docs", "v1");
//! svc.emit_delete("docs", "v2");
//!
//! // Poll events for subscriber
//! let events = svc.poll(&handle.id);
//! assert_eq!(events.len(), 2);
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

/// Event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    Insert,
    Update,
    Delete,
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Insert => write!(f, "insert"),
            Self::Update => write!(f, "update"),
            Self::Delete => write!(f, "delete"),
        }
    }
}

/// A single change event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    pub id: u64,
    pub kind: EventKind,
    pub collection: String,
    pub vector_id: String,
    pub timestamp: u64,
}

impl ChangeEvent {
    /// Format as SSE data line.
    pub fn to_sse(&self) -> String {
        format!(
            "id: {}\nevent: {}\ndata: {{\"collection\":\"{}\",\"vector_id\":\"{}\",\"timestamp\":{}}}\n\n",
            self.id, self.kind, self.collection, self.vector_id, self.timestamp
        )
    }
}

/// Filter for which events a subscriber receives.
#[derive(Debug, Clone)]
pub struct EventFilter {
    pub event_kinds: Option<Vec<EventKind>>,
    pub id_prefix: Option<String>,
}

impl EventFilter {
    /// Accept all events.
    pub fn all() -> Self {
        Self { event_kinds: None, id_prefix: None }
    }

    /// Only specific event types.
    pub fn kinds(kinds: Vec<EventKind>) -> Self {
        Self { event_kinds: Some(kinds), id_prefix: None }
    }

    /// Only events matching an ID prefix.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.id_prefix = Some(prefix.into());
        self
    }

    fn matches(&self, event: &ChangeEvent) -> bool {
        if let Some(ref kinds) = self.event_kinds {
            if !kinds.contains(&event.kind) {
                return false;
            }
        }
        if let Some(ref prefix) = self.id_prefix {
            if !event.vector_id.starts_with(prefix) {
                return false;
            }
        }
        true
    }
}

/// Handle returned to a subscriber.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriberHandle {
    pub id: String,
    pub collection: String,
    pub from_offset: u64,
}

/// Internal subscriber state.
struct Subscriber {
    handle: SubscriberHandle,
    filter: EventFilter,
    last_seen: u64,
}

/// Stream configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub max_subscribers: usize,
    pub max_event_log: usize,
    pub heartbeat_interval_secs: u64,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self { max_subscribers: 100, max_event_log: 100_000, heartbeat_interval_secs: 15 }
    }
}

/// Change stream service managing subscribers and event dispatch.
pub struct ChangeStreamService {
    config: StreamConfig,
    subscribers: HashMap<String, Subscriber>,
    events: VecDeque<ChangeEvent>,
    next_event_id: u64,
    next_sub_id: u64,
}

impl ChangeStreamService {
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config,
            subscribers: HashMap::new(),
            events: VecDeque::new(),
            next_event_id: 1,
            next_sub_id: 0,
        }
    }

    /// Subscribe to changes on a collection.
    pub fn subscribe(&mut self, collection: &str, filter: EventFilter) -> Result<SubscriberHandle> {
        if self.subscribers.len() >= self.config.max_subscribers {
            return Err(NeedleError::CapacityExceeded(
                format!("Max subscribers ({}) reached", self.config.max_subscribers)
            ));
        }
        let id = format!("sub_{}", self.next_sub_id);
        self.next_sub_id += 1;
        let handle = SubscriberHandle {
            id: id.clone(),
            collection: collection.to_string(),
            from_offset: self.next_event_id.saturating_sub(1),
        };
        self.subscribers.insert(id.clone(), Subscriber {
            handle: handle.clone(),
            filter,
            last_seen: self.next_event_id.saturating_sub(1),
        });
        Ok(handle)
    }

    /// Subscribe with resume from a specific offset.
    pub fn subscribe_from(
        &mut self,
        collection: &str,
        filter: EventFilter,
        from_offset: u64,
    ) -> Result<SubscriberHandle> {
        let mut handle = self.subscribe(collection, filter)?;
        handle.from_offset = from_offset;
        if let Some(sub) = self.subscribers.get_mut(&handle.id) {
            sub.last_seen = from_offset;
        }
        Ok(handle)
    }

    /// Unsubscribe.
    pub fn unsubscribe(&mut self, sub_id: &str) -> bool {
        self.subscribers.remove(sub_id).is_some()
    }

    /// Emit an insert event.
    pub fn emit_insert(&mut self, collection: &str, vector_id: &str) {
        self.emit(EventKind::Insert, collection, vector_id);
    }

    /// Emit an update event.
    pub fn emit_update(&mut self, collection: &str, vector_id: &str) {
        self.emit(EventKind::Update, collection, vector_id);
    }

    /// Emit a delete event.
    pub fn emit_delete(&mut self, collection: &str, vector_id: &str) {
        self.emit(EventKind::Delete, collection, vector_id);
    }

    /// Poll pending events for a subscriber.
    pub fn poll(&mut self, sub_id: &str) -> Vec<ChangeEvent> {
        let sub = match self.subscribers.get(sub_id) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let last_seen = sub.last_seen;
        let collection = sub.handle.collection.clone();

        let events: Vec<ChangeEvent> = self.events.iter()
            .filter(|e| e.id > last_seen && e.collection == collection && sub.filter.matches(e))
            .cloned()
            .collect();

        if let Some(last) = events.last() {
            if let Some(s) = self.subscribers.get_mut(sub_id) {
                s.last_seen = last.id;
            }
        }
        events
    }

    /// Format pending events as SSE text for HTTP response.
    pub fn poll_sse(&mut self, sub_id: &str) -> String {
        let events = self.poll(sub_id);
        events.iter().map(|e| e.to_sse()).collect()
    }

    /// Current event offset.
    pub fn current_offset(&self) -> u64 { self.next_event_id.saturating_sub(1) }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize { self.subscribers.len() }

    /// Number of events in the log.
    pub fn event_count(&self) -> usize { self.events.len() }

    fn emit(&mut self, kind: EventKind, collection: &str, vector_id: &str) {
        let event = ChangeEvent {
            id: self.next_event_id,
            kind,
            collection: collection.to_string(),
            vector_id: vector_id.to_string(),
            timestamp: now_secs(),
        };
        self.next_event_id += 1;
        self.events.push_back(event);
        while self.events.len() > self.config.max_event_log {
            self.events.pop_front();
        }
    }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_and_poll() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let handle = svc.subscribe("docs", EventFilter::all()).unwrap();

        svc.emit_insert("docs", "v1");
        svc.emit_delete("docs", "v2");

        let events = svc.poll(&handle.id);
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, EventKind::Insert);
        assert_eq!(events[1].kind, EventKind::Delete);
    }

    #[test]
    fn test_collection_scoping() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let handle = svc.subscribe("docs", EventFilter::all()).unwrap();

        svc.emit_insert("docs", "v1");
        svc.emit_insert("other", "v2");

        let events = svc.poll(&handle.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].vector_id, "v1");
    }

    #[test]
    fn test_event_filter_kinds() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let handle = svc.subscribe("docs", EventFilter::kinds(vec![EventKind::Insert])).unwrap();

        svc.emit_insert("docs", "v1");
        svc.emit_delete("docs", "v2");

        let events = svc.poll(&handle.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, EventKind::Insert);
    }

    #[test]
    fn test_id_prefix_filter() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let filter = EventFilter::all().with_prefix("doc_");
        let handle = svc.subscribe("docs", filter).unwrap();

        svc.emit_insert("docs", "doc_1");
        svc.emit_insert("docs", "img_1");

        let events = svc.poll(&handle.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].vector_id, "doc_1");
    }

    #[test]
    fn test_offset_resume() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        svc.emit_insert("docs", "v1");
        svc.emit_insert("docs", "v2");
        let offset = svc.current_offset();

        // Subscribe from current offset — should miss v1 and v2
        let handle = svc.subscribe_from("docs", EventFilter::all(), offset).unwrap();
        svc.emit_insert("docs", "v3");

        let events = svc.poll(&handle.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].vector_id, "v3");
    }

    #[test]
    fn test_sse_format() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let handle = svc.subscribe("docs", EventFilter::all()).unwrap();
        svc.emit_insert("docs", "v1");

        let sse = svc.poll_sse(&handle.id);
        assert!(sse.contains("event: insert"));
        assert!(sse.contains("data:"));
        assert!(sse.contains("v1"));
    }

    #[test]
    fn test_unsubscribe() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let handle = svc.subscribe("docs", EventFilter::all()).unwrap();
        assert_eq!(svc.subscriber_count(), 1);
        assert!(svc.unsubscribe(&handle.id));
        assert_eq!(svc.subscriber_count(), 0);
    }

    #[test]
    fn test_max_subscribers() {
        let mut svc = ChangeStreamService::new(StreamConfig { max_subscribers: 2, ..Default::default() });
        svc.subscribe("a", EventFilter::all()).unwrap();
        svc.subscribe("b", EventFilter::all()).unwrap();
        assert!(svc.subscribe("c", EventFilter::all()).is_err());
    }

    #[test]
    fn test_poll_idempotent() {
        let mut svc = ChangeStreamService::new(StreamConfig::default());
        let handle = svc.subscribe("docs", EventFilter::all()).unwrap();
        svc.emit_insert("docs", "v1");

        let first = svc.poll(&handle.id);
        assert_eq!(first.len(), 1);

        // Second poll should return nothing (already advanced)
        let second = svc.poll(&handle.id);
        assert!(second.is_empty());
    }
}
