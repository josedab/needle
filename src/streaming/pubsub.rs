use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc, Mutex, RwLock};

use super::core::{
    ChangeEvent, ChangeEventFilter, StreamError, StreamResult, DEFAULT_BUFFER_SIZE,
    DEFAULT_CHANNEL_CAPACITY,
};

// ============================================================================
// Subscriber
// ============================================================================

/// Subscriber handle for receiving changes via pub/sub
pub struct Subscriber {
    /// Unique subscriber ID
    pub id: u64,
    /// Collection being watched
    pub collection: String,
    /// Receiver for changes
    receiver: mpsc::Receiver<ChangeEvent>,
    /// Active flag
    active: Arc<AtomicBool>,
    /// Event filter
    filter: ChangeEventFilter,
}

impl Subscriber {
    /// Receive the next change event
    pub async fn recv(&mut self) -> Option<ChangeEvent> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }

        loop {
            match self.receiver.recv().await {
                Some(event) => {
                    if self.filter.matches(&event) {
                        return Some(event);
                    }
                }
                None => return None,
            }
        }
    }

    /// Receive with timeout
    pub async fn recv_timeout(&mut self, timeout: Duration) -> StreamResult<Option<ChangeEvent>> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(None);
        }

        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(event) => Ok(event),
            Err(_) => Err(StreamError::Timeout),
        }
    }

    /// Try to receive without blocking
    pub fn try_recv(&mut self) -> Option<ChangeEvent> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }

        match self.receiver.try_recv() {
            Ok(event) if self.filter.matches(&event) => Some(event),
            _ => None,
        }
    }

    /// Unsubscribe from changes
    pub fn unsubscribe(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if still active
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Get the filter for this subscriber
    pub fn filter(&self) -> &ChangeEventFilter {
        &self.filter
    }
}

// ============================================================================
// Subscription Info (Internal)
// ============================================================================

/// Internal subscription info stored in the pub/sub system
struct SubscriptionInfo {
    sender: mpsc::Sender<ChangeEvent>,
    active: Arc<AtomicBool>,
    filter: ChangeEventFilter,
}

// ============================================================================
// PubSub
// ============================================================================

/// Publish/Subscribe system for real-time updates with backpressure handling
pub struct PubSub {
    /// Subscriptions by collection
    subscriptions: Arc<RwLock<HashMap<String, Vec<SubscriptionInfo>>>>,
    /// Global subscriptions (all collections)
    global_subscriptions: Arc<RwLock<Vec<SubscriptionInfo>>>,
    /// Next subscriber ID
    next_subscriber_id: AtomicU64,
    /// Broadcast channel for all events
    broadcast_tx: broadcast::Sender<ChangeEvent>,
    /// Buffer for backpressure handling
    buffer: Arc<Mutex<VecDeque<ChangeEvent>>>,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Channel capacity for subscribers
    channel_capacity: usize,
}

impl PubSub {
    /// Create a new pub/sub system
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(DEFAULT_CHANNEL_CAPACITY);
        Self {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            global_subscriptions: Arc::new(RwLock::new(Vec::new())),
            next_subscriber_id: AtomicU64::new(1),
            broadcast_tx,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_buffer_size: DEFAULT_BUFFER_SIZE,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
        }
    }

    /// Create with custom configuration
    pub fn with_config(max_buffer_size: usize, channel_capacity: usize) -> Self {
        let (broadcast_tx, _) = broadcast::channel(channel_capacity);
        Self {
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            global_subscriptions: Arc::new(RwLock::new(Vec::new())),
            next_subscriber_id: AtomicU64::new(1),
            broadcast_tx,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_buffer_size,
            channel_capacity,
        }
    }

    /// Subscribe to changes for a specific collection
    pub async fn subscribe(&self, collection: &str) -> Subscriber {
        self.subscribe_with_filter(collection, ChangeEventFilter::default())
            .await
    }

    /// Subscribe with a custom filter
    pub async fn subscribe_with_filter(
        &self,
        collection: &str,
        filter: ChangeEventFilter,
    ) -> Subscriber {
        let id = self.next_subscriber_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(self.channel_capacity);
        let active = Arc::new(AtomicBool::new(true));

        let info = SubscriptionInfo {
            sender: tx,
            active: Arc::clone(&active),
            filter: filter.clone(),
        };

        let mut subs = self.subscriptions.write().await;
        subs.entry(collection.to_string())
            .or_insert_with(Vec::new)
            .push(info);

        Subscriber {
            id,
            collection: collection.to_string(),
            receiver: rx,
            active,
            filter,
        }
    }

    /// Subscribe to all collections
    pub async fn subscribe_all(&self) -> Subscriber {
        self.subscribe_all_with_filter(ChangeEventFilter::default())
            .await
    }

    /// Subscribe to all collections with a filter
    pub async fn subscribe_all_with_filter(&self, filter: ChangeEventFilter) -> Subscriber {
        let id = self.next_subscriber_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = mpsc::channel(self.channel_capacity);
        let active = Arc::new(AtomicBool::new(true));

        let info = SubscriptionInfo {
            sender: tx,
            active: Arc::clone(&active),
            filter: filter.clone(),
        };

        let mut subs = self.global_subscriptions.write().await;
        subs.push(info);

        Subscriber {
            id,
            collection: "*".to_string(),
            receiver: rx,
            active,
            filter,
        }
    }

    /// Publish a change event to all matching subscribers
    pub async fn publish(&self, event: ChangeEvent) -> StreamResult<()> {
        // Send to broadcast channel
        let _ = self.broadcast_tx.send(event.clone());

        // Send to collection-specific subscribers
        let subs = self.subscriptions.read().await;
        if let Some(subscribers) = subs.get(&event.collection) {
            for sub in subscribers {
                if sub.active.load(Ordering::Relaxed)
                    && sub.filter.matches(&event)
                    && sub.sender.try_send(event.clone()).is_err()
                {
                    // Buffer the event for backpressure handling
                    self.buffer_event(event.clone()).await?;
                }
            }
        }

        // Send to global subscribers
        let global_subs = self.global_subscriptions.read().await;
        for sub in global_subs.iter() {
            if sub.active.load(Ordering::Relaxed)
                && sub.filter.matches(&event)
                && sub.sender.try_send(event.clone()).is_err()
            {
                self.buffer_event(event.clone()).await?;
            }
        }

        Ok(())
    }

    /// Buffer an event when channels are full (backpressure handling)
    async fn buffer_event(&self, event: ChangeEvent) -> StreamResult<()> {
        let mut buffer = self.buffer.lock().await;
        if buffer.len() >= self.max_buffer_size {
            return Err(StreamError::BufferOverflow);
        }
        buffer.push_back(event);
        Ok(())
    }

    /// Flush buffered events to subscribers
    pub async fn flush_buffer(&self) -> StreamResult<usize> {
        let mut buffer = self.buffer.lock().await;
        let mut flushed = 0;

        while let Some(event) = buffer.pop_front() {
            let subs = self.subscriptions.read().await;
            if let Some(subscribers) = subs.get(&event.collection) {
                for sub in subscribers {
                    if sub.active.load(Ordering::Relaxed) && sub.filter.matches(&event) {
                        if sub.sender.try_send(event.clone()).is_ok() {
                            flushed += 1;
                        } else {
                            // Put it back and stop
                            buffer.push_front(event);
                            return Ok(flushed);
                        }
                    }
                }
            }
        }

        Ok(flushed)
    }

    /// Get subscriber count for a collection
    pub async fn subscriber_count(&self, collection: &str) -> usize {
        let subs = self.subscriptions.read().await;
        subs.get(collection)
            .map(|s| {
                s.iter()
                    .filter(|sub| sub.active.load(Ordering::Relaxed))
                    .count()
            })
            .unwrap_or(0)
    }

    /// Get total subscriber count
    pub async fn total_subscriber_count(&self) -> usize {
        let subs = self.subscriptions.read().await;
        let global = self.global_subscriptions.read().await;

        let collection_count: usize = subs
            .values()
            .map(|s| {
                s.iter()
                    .filter(|sub| sub.active.load(Ordering::Relaxed))
                    .count()
            })
            .sum();

        let global_count = global
            .iter()
            .filter(|sub| sub.active.load(Ordering::Relaxed))
            .count();

        collection_count + global_count
    }

    /// Get buffered event count
    pub async fn buffer_count(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Clean up inactive subscriptions
    pub async fn cleanup_inactive(&self) {
        let mut subs = self.subscriptions.write().await;
        for subscribers in subs.values_mut() {
            subscribers.retain(|sub| sub.active.load(Ordering::Relaxed));
        }
        // Remove empty collections
        subs.retain(|_, v| !v.is_empty());

        let mut global = self.global_subscriptions.write().await;
        global.retain(|sub| sub.active.load(Ordering::Relaxed));
    }

    /// Get a broadcast receiver for all events
    pub fn broadcast_receiver(&self) -> broadcast::Receiver<ChangeEvent> {
        self.broadcast_tx.subscribe()
    }
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
    }
}
