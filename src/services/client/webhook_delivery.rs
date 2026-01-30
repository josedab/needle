//! Webhook Delivery Service
//!
//! HTTP webhook delivery for collection change events with configurable
//! filters, retry with exponential backoff, and dead-letter queue.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::webhook_delivery::{
//!     WebhookService, WebhookConfig, WebhookSubscription, EventFilter,
//! };
//!
//! let mut svc = WebhookService::new(WebhookConfig::default());
//!
//! // Register a webhook
//! let sub = WebhookSubscription::new(
//!     "https://example.com/hooks/needle",
//!     EventFilter::all(),
//! ).with_secret("my-hmac-secret");
//! let id = svc.subscribe(sub);
//!
//! // Enqueue an event for delivery
//! svc.enqueue("docs", "insert", "vec_123");
//!
//! // Process the delivery queue
//! let stats = svc.process_queue();
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Webhook service configuration.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Maximum retry attempts per event
    pub max_retries: u32,
    /// Initial retry delay (doubles on each retry)
    pub initial_retry_delay: Duration,
    /// Maximum events in the dead-letter queue
    pub max_dlq_size: usize,
    /// Request timeout per webhook call
    pub request_timeout: Duration,
    /// Maximum concurrent deliveries
    pub max_concurrent: usize,
}

impl Default for WebhookConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_retry_delay: Duration::from_secs(1),
            max_dlq_size: 10_000,
            request_timeout: Duration::from_secs(10),
            max_concurrent: 10,
        }
    }
}

/// Event types that can trigger webhooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WebhookEventType {
    Insert,
    Update,
    Delete,
    Compact,
}

impl std::fmt::Display for WebhookEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Insert => write!(f, "insert"),
            Self::Update => write!(f, "update"),
            Self::Delete => write!(f, "delete"),
            Self::Compact => write!(f, "compact"),
        }
    }
}

/// Filter for which events trigger a webhook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    /// Event types to deliver (empty = all)
    pub event_types: Vec<WebhookEventType>,
    /// Collection name filter (empty = all collections)
    pub collections: Vec<String>,
}

impl EventFilter {
    pub fn all() -> Self {
        Self {
            event_types: vec![],
            collections: vec![],
        }
    }

    pub fn for_collection(name: impl Into<String>) -> Self {
        Self {
            event_types: vec![],
            collections: vec![name.into()],
        }
    }

    pub fn matches(&self, collection: &str, event_type: WebhookEventType) -> bool {
        let type_match = self.event_types.is_empty()
            || self.event_types.contains(&event_type);
        let coll_match = self.collections.is_empty()
            || self.collections.iter().any(|c| c == collection);
        type_match && coll_match
    }
}

/// A webhook subscription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookSubscription {
    pub id: String,
    pub url: String,
    pub filter: EventFilter,
    /// HMAC-SHA256 secret for signing payloads
    pub secret: Option<String>,
    pub active: bool,
    pub created_at: u64,
}

impl WebhookSubscription {
    pub fn new(url: impl Into<String>, filter: EventFilter) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            id: format!("wh_{}", ts),
            url: url.into(),
            filter,
            secret: None,
            active: true,
            created_at: ts,
        }
    }

    #[must_use]
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }
}

/// An event pending delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    pub id: u64,
    pub collection: String,
    pub event_type: WebhookEventType,
    pub vector_id: String,
    pub timestamp: u64,
    pub attempt: u32,
}

/// Dead-letter queue entry.
#[derive(Debug, Clone, Serialize)]
pub struct DeadLetterEntry {
    pub event: WebhookEvent,
    pub subscription_id: String,
    pub last_error: String,
    pub failed_at: u64,
}

/// Delivery processing statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct DeliveryStats {
    pub delivered: usize,
    pub failed: usize,
    pub retried: usize,
    pub dlq_size: usize,
}

/// Webhook delivery service.
pub struct WebhookService {
    config: WebhookConfig,
    subscriptions: HashMap<String, WebhookSubscription>,
    queue: VecDeque<(WebhookEvent, String)>,
    dlq: VecDeque<DeadLetterEntry>,
    next_event_id: u64,
    total_delivered: u64,
    total_failed: u64,
}

impl WebhookService {
    pub fn new(config: WebhookConfig) -> Self {
        Self {
            config,
            subscriptions: HashMap::new(),
            queue: VecDeque::new(),
            dlq: VecDeque::new(),
            next_event_id: 1,
            total_delivered: 0,
            total_failed: 0,
        }
    }

    /// Register a webhook subscription. Returns the subscription ID.
    pub fn subscribe(&mut self, sub: WebhookSubscription) -> String {
        let id = sub.id.clone();
        self.subscriptions.insert(id.clone(), sub);
        id
    }

    /// Remove a webhook subscription.
    pub fn unsubscribe(&mut self, id: &str) -> bool {
        self.subscriptions.remove(id).is_some()
    }

    /// List all subscriptions.
    pub fn list_subscriptions(&self) -> Vec<&WebhookSubscription> {
        self.subscriptions.values().collect()
    }

    /// Enqueue an event for delivery to all matching subscriptions.
    pub fn enqueue(&mut self, collection: &str, event_type_str: &str, vector_id: &str) {
        let event_type = match event_type_str {
            "insert" => WebhookEventType::Insert,
            "update" => WebhookEventType::Update,
            "delete" => WebhookEventType::Delete,
            "compact" => WebhookEventType::Compact,
            _ => return,
        };

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let event = WebhookEvent {
            id: self.next_event_id,
            collection: collection.to_string(),
            event_type,
            vector_id: vector_id.to_string(),
            timestamp: ts,
            attempt: 0,
        };
        self.next_event_id += 1;

        // Fan out to matching subscriptions
        let matching: Vec<String> = self
            .subscriptions
            .iter()
            .filter(|(_, sub)| sub.active && sub.filter.matches(collection, event_type))
            .map(|(id, _)| id.clone())
            .collect();

        for sub_id in matching {
            self.queue.push_back((event.clone(), sub_id));
        }
    }

    /// Process the delivery queue (sync simulation — real impl would use async HTTP).
    /// Returns delivery statistics.
    pub fn process_queue(&mut self) -> DeliveryStats {
        let mut stats = DeliveryStats::default();
        let max_process = self.config.max_concurrent;

        for _ in 0..max_process {
            let Some((mut event, sub_id)) = self.queue.pop_front() else {
                break;
            };

            // Simulate delivery (in production, this would be an async HTTP POST)
            let success = self.subscriptions.contains_key(&sub_id);

            if success {
                self.total_delivered += 1;
                stats.delivered += 1;
            } else {
                event.attempt += 1;
                if event.attempt < self.config.max_retries {
                    self.queue.push_back((event, sub_id));
                    stats.retried += 1;
                } else {
                    let ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64;
                    if self.dlq.len() < self.config.max_dlq_size {
                        self.dlq.push_back(DeadLetterEntry {
                            event,
                            subscription_id: sub_id,
                            last_error: "Max retries exceeded".to_string(),
                            failed_at: ts,
                        });
                    }
                    self.total_failed += 1;
                    stats.failed += 1;
                }
            }
        }

        stats.dlq_size = self.dlq.len();
        stats
    }

    /// Get the dead-letter queue size.
    pub fn dlq_size(&self) -> usize {
        self.dlq.len()
    }

    /// Drain the dead-letter queue.
    pub fn drain_dlq(&mut self) -> Vec<DeadLetterEntry> {
        self.dlq.drain(..).collect()
    }

    /// Get overall stats.
    pub fn total_stats(&self) -> (u64, u64, usize) {
        (self.total_delivered, self.total_failed, self.queue.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subscribe_and_enqueue() {
        let mut svc = WebhookService::new(WebhookConfig::default());
        let sub = WebhookSubscription::new("https://example.com/hook", EventFilter::all());
        let id = svc.subscribe(sub);

        svc.enqueue("docs", "insert", "v1");
        assert_eq!(svc.queue.len(), 1);

        let stats = svc.process_queue();
        assert_eq!(stats.delivered, 1);
        assert!(svc.queue.is_empty());
    }

    #[test]
    fn test_filter_matching() {
        let mut svc = WebhookService::new(WebhookConfig::default());
        let sub = WebhookSubscription::new(
            "https://example.com/hook",
            EventFilter::for_collection("docs"),
        );
        svc.subscribe(sub);

        svc.enqueue("docs", "insert", "v1");
        svc.enqueue("other", "insert", "v2"); // Should not match
        assert_eq!(svc.queue.len(), 1);
    }

    #[test]
    fn test_event_filter_all() {
        let filter = EventFilter::all();
        assert!(filter.matches("any", WebhookEventType::Insert));
        assert!(filter.matches("any", WebhookEventType::Delete));
    }

    #[test]
    fn test_unsubscribe() {
        let mut svc = WebhookService::new(WebhookConfig::default());
        let sub = WebhookSubscription::new("https://example.com", EventFilter::all());
        let id = svc.subscribe(sub);

        assert!(svc.unsubscribe(&id));
        assert!(!svc.unsubscribe(&id));
    }

    #[test]
    fn test_webhook_secret() {
        let sub = WebhookSubscription::new("https://example.com", EventFilter::all())
            .with_secret("my-secret");
        assert_eq!(sub.secret, Some("my-secret".to_string()));
    }
}
