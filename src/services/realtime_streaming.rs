//! Real-Time Vector Streaming
//!
//! Change feed with collection subscriptions. Clients receive insert/update/delete
//! events as they happen, enabling live dashboards and agent-to-agent coordination.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::vector_streaming::{
//!     StreamServer, StreamConfig, ChangeEvent, Subscription,
//! };
//!
//! let mut server = StreamServer::new(StreamConfig::default());
//! let sub_id = server.subscribe("docs", None);
//!
//! // Publish a change
//! server.publish(ChangeEvent::insert("docs", "v1", &[1.0; 4]));
//!
//! // Poll for events
//! let events = server.poll(sub_id, 100);
//! assert_eq!(events.len(), 1);
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Change event type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeEvent {
    Insert { collection: String, id: String, vector_dims: usize, metadata: Option<Value>, timestamp: u64 },
    Update { collection: String, id: String, vector_dims: usize, metadata: Option<Value>, timestamp: u64 },
    Delete { collection: String, id: String, timestamp: u64 },
}

impl ChangeEvent {
    pub fn insert(collection: &str, id: &str, vector: &[f32]) -> Self {
        Self::Insert { collection: collection.into(), id: id.into(), vector_dims: vector.len(), metadata: None, timestamp: now_secs() }
    }
    pub fn update(collection: &str, id: &str, vector: &[f32]) -> Self {
        Self::Update { collection: collection.into(), id: id.into(), vector_dims: vector.len(), metadata: None, timestamp: now_secs() }
    }
    pub fn delete(collection: &str, id: &str) -> Self {
        Self::Delete { collection: collection.into(), id: id.into(), timestamp: now_secs() }
    }
    pub fn collection(&self) -> &str {
        match self { Self::Insert { collection, .. } | Self::Update { collection, .. } | Self::Delete { collection, .. } => collection }
    }
}

/// Subscription filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionFilter {
    pub event_types: Option<Vec<String>>,
    pub id_prefix: Option<String>,
}

/// A subscription handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Subscription(pub u64);

/// Stream configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub max_buffer_per_sub: usize,
    pub max_subscriptions: usize,
}

impl Default for StreamConfig {
    fn default() -> Self { Self { max_buffer_per_sub: 10_000, max_subscriptions: 1_000 } }
}

struct SubState {
    collection: String,
    filter: Option<SubscriptionFilter>,
    events: VecDeque<ChangeEvent>,
    created_at: u64,
}

/// Real-time change feed server.
pub struct StreamServer {
    config: StreamConfig,
    subscriptions: HashMap<u64, SubState>,
    next_id: u64,
    total_events: u64,
}

impl StreamServer {
    pub fn new(config: StreamConfig) -> Self {
        Self { config, subscriptions: HashMap::new(), next_id: 1, total_events: 0 }
    }

    /// Subscribe to changes on a collection.
    pub fn subscribe(&mut self, collection: &str, filter: Option<SubscriptionFilter>) -> Subscription {
        let id = self.next_id;
        self.next_id += 1;
        self.subscriptions.insert(id, SubState {
            collection: collection.into(), filter, events: VecDeque::new(), created_at: now_secs(),
        });
        Subscription(id)
    }

    /// Unsubscribe.
    pub fn unsubscribe(&mut self, sub: Subscription) -> bool {
        self.subscriptions.remove(&sub.0).is_some()
    }

    /// Publish a change event to matching subscribers.
    pub fn publish(&mut self, event: ChangeEvent) {
        self.total_events += 1;
        let collection = event.collection().to_string();
        for sub in self.subscriptions.values_mut() {
            if sub.collection == collection {
                if sub.events.len() >= self.config.max_buffer_per_sub {
                    sub.events.pop_front();
                }
                sub.events.push_back(event.clone());
            }
        }
    }

    /// Poll events for a subscription (drains buffer up to limit).
    pub fn poll(&mut self, sub: Subscription, limit: usize) -> Vec<ChangeEvent> {
        let state = match self.subscriptions.get_mut(&sub.0) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let n = limit.min(state.events.len());
        state.events.drain(..n).collect()
    }

    /// Number of active subscriptions.
    pub fn subscription_count(&self) -> usize { self.subscriptions.len() }
    /// Total events published.
    pub fn total_events(&self) -> u64 { self.total_events }
    /// Pending events for a subscription.
    pub fn pending(&self, sub: Subscription) -> usize {
        self.subscriptions.get(&sub.0).map_or(0, |s| s.events.len())
    }
}

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_and_publish() {
        let mut srv = StreamServer::new(StreamConfig::default());
        let sub = srv.subscribe("docs", None);
        srv.publish(ChangeEvent::insert("docs", "v1", &[1.0; 4]));
        let events = srv.poll(sub, 10);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_collection_filter() {
        let mut srv = StreamServer::new(StreamConfig::default());
        let sub_docs = srv.subscribe("docs", None);
        let sub_imgs = srv.subscribe("images", None);
        srv.publish(ChangeEvent::insert("docs", "v1", &[1.0; 4]));
        assert_eq!(srv.poll(sub_docs, 10).len(), 1);
        assert_eq!(srv.poll(sub_imgs, 10).len(), 0);
    }

    #[test]
    fn test_unsubscribe() {
        let mut srv = StreamServer::new(StreamConfig::default());
        let sub = srv.subscribe("docs", None);
        assert!(srv.unsubscribe(sub));
        assert_eq!(srv.subscription_count(), 0);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut srv = StreamServer::new(StreamConfig { max_buffer_per_sub: 3, max_subscriptions: 10 });
        let sub = srv.subscribe("docs", None);
        for i in 0..5 { srv.publish(ChangeEvent::insert("docs", &format!("v{i}"), &[1.0; 4])); }
        assert_eq!(srv.pending(sub), 3);
    }

    #[test]
    fn test_multiple_subscribers() {
        let mut srv = StreamServer::new(StreamConfig::default());
        let s1 = srv.subscribe("docs", None);
        let s2 = srv.subscribe("docs", None);
        srv.publish(ChangeEvent::insert("docs", "v1", &[1.0; 4]));
        assert_eq!(srv.poll(s1, 10).len(), 1);
        assert_eq!(srv.poll(s2, 10).len(), 1);
    }
}
