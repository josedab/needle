use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, RwLock};

use super::core::{
    current_timestamp_millis, ChangeEvent, ChangeEventFilter, ResumeToken, StreamError,
    StreamResult, COMPACTION_THRESHOLD, DEFAULT_BUFFER_SIZE, DEFAULT_CHANNEL_CAPACITY,
};
use super::event_log::{EventLog, ReplayOptions};
use super::pubsub::{PubSub, Subscriber};

// ============================================================================
// Change Stream
// ============================================================================

/// Change stream for receiving real-time updates with async iteration
pub struct ChangeStream {
    /// Collection being watched (None for all)
    collection: Option<String>,
    /// Event filter
    filter: ChangeEventFilter,
    /// Receiver for change events
    receiver: mpsc::Receiver<ChangeEvent>,
    /// Current position in the stream
    position: AtomicU64,
    /// Whether the stream is closed
    closed: Arc<AtomicBool>,
    /// Buffer for pending events
    buffer: VecDeque<ChangeEvent>,
    /// Maximum buffer size
    max_buffer_size: usize,
}

impl ChangeStream {
    /// Create a new change stream
    pub fn new(receiver: mpsc::Receiver<ChangeEvent>) -> Self {
        Self {
            collection: None,
            filter: ChangeEventFilter::default(),
            receiver,
            position: AtomicU64::new(0),
            closed: Arc::new(AtomicBool::new(false)),
            buffer: VecDeque::new(),
            max_buffer_size: DEFAULT_BUFFER_SIZE,
        }
    }

    /// Create a change stream for a specific collection
    pub fn for_collection(collection: &str, receiver: mpsc::Receiver<ChangeEvent>) -> Self {
        let mut stream = Self::new(receiver);
        stream.collection = Some(collection.to_string());
        stream.filter = ChangeEventFilter::collections(&[collection]);
        stream
    }

    /// Set the filter for this stream
    pub fn with_filter(mut self, filter: ChangeEventFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Set maximum buffer size
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Get the next change event (async)
    pub async fn next(&mut self) -> Option<ChangeEvent> {
        if self.closed.load(Ordering::Relaxed) {
            return None;
        }

        // First check buffer
        if let Some(event) = self.buffer.pop_front() {
            self.position.store(event.id, Ordering::Relaxed);
            return Some(event);
        }

        // Wait for next event from receiver
        loop {
            match self.receiver.recv().await {
                Some(event) => {
                    if self.filter.matches(&event) {
                        self.position.store(event.id, Ordering::Relaxed);
                        return Some(event);
                    }
                    // Event doesn't match filter, continue waiting
                }
                None => {
                    self.closed.store(true, Ordering::Relaxed);
                    return None;
                }
            }
        }
    }

    /// Get the next event with timeout
    pub async fn next_timeout(&mut self, timeout: Duration) -> StreamResult<Option<ChangeEvent>> {
        if self.closed.load(Ordering::Relaxed) {
            return Ok(None);
        }

        // First check buffer
        if let Some(event) = self.buffer.pop_front() {
            self.position.store(event.id, Ordering::Relaxed);
            return Ok(Some(event));
        }

        match tokio::time::timeout(timeout, self.next()).await {
            Ok(event) => Ok(event),
            Err(_) => Err(StreamError::Timeout),
        }
    }

    /// Try to get the next event without blocking
    pub fn try_next(&mut self) -> Option<ChangeEvent> {
        if self.closed.load(Ordering::Relaxed) {
            return None;
        }

        // First check buffer
        if let Some(event) = self.buffer.pop_front() {
            self.position.store(event.id, Ordering::Relaxed);
            return Some(event);
        }

        // Try to receive without blocking
        match self.receiver.try_recv() {
            Ok(event) => {
                if self.filter.matches(&event) {
                    self.position.store(event.id, Ordering::Relaxed);
                    Some(event)
                } else {
                    None
                }
            }
            Err(_) => None,
        }
    }

    /// Get the current position in the stream
    pub fn position(&self) -> u64 {
        self.position.load(Ordering::Relaxed)
    }

    /// Get the current resume token
    pub fn resume_token(&self) -> ResumeToken {
        let pos = self.position.load(Ordering::Relaxed);
        let timestamp = current_timestamp_millis();
        ResumeToken::new(pos, timestamp)
    }

    /// Close the stream
    pub fn close(&self) {
        self.closed.store(true, Ordering::Relaxed);
    }

    /// Check if the stream is closed
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    /// Get the collection this stream is watching
    pub fn collection(&self) -> Option<&str> {
        self.collection.as_deref()
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

// ============================================================================
// Stream Manager
// ============================================================================

/// Manager for change streams and subscriptions
///
/// Coordinates event logging, publishing, and stream creation
pub struct StreamManager {
    /// Event log for persistence
    event_log: Arc<EventLog>,
    /// Pub/sub system
    pubsub: Arc<PubSub>,
    /// Active change stream senders
    stream_senders: Arc<RwLock<Vec<mpsc::Sender<ChangeEvent>>>>,
    /// Configuration
    pub(crate) config: StreamManagerConfig,
}

/// Configuration for StreamManager
#[derive(Debug, Clone)]
pub struct StreamManagerConfig {
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Channel capacity
    pub channel_capacity: usize,
    /// Compaction threshold
    pub compaction_threshold: usize,
    /// Auto-cleanup interval in seconds
    pub cleanup_interval_secs: u64,
}

impl Default for StreamManagerConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: DEFAULT_BUFFER_SIZE,
            channel_capacity: DEFAULT_CHANNEL_CAPACITY,
            compaction_threshold: COMPACTION_THRESHOLD,
            cleanup_interval_secs: 60,
        }
    }
}

impl StreamManager {
    /// Create a new stream manager
    pub fn new() -> Self {
        Self {
            event_log: Arc::new(EventLog::new()),
            pubsub: Arc::new(PubSub::new()),
            stream_senders: Arc::new(RwLock::new(Vec::new())),
            config: StreamManagerConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StreamManagerConfig) -> Self {
        Self {
            event_log: Arc::new(EventLog::with_compaction_threshold(
                config.compaction_threshold,
            )),
            pubsub: Arc::new(PubSub::with_config(
                config.max_buffer_size,
                config.channel_capacity,
            )),
            stream_senders: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Create with custom event log and pubsub
    pub fn with_components(event_log: EventLog, pubsub: PubSub) -> Self {
        Self {
            event_log: Arc::new(event_log),
            pubsub: Arc::new(pubsub),
            stream_senders: Arc::new(RwLock::new(Vec::new())),
            config: StreamManagerConfig::default(),
        }
    }

    /// Record and broadcast a change event
    pub async fn record_change(&self, event: ChangeEvent) -> StreamResult<u64> {
        // Append to event log
        let position = self.event_log.append(event.clone()).await?;

        // Publish to subscribers
        self.pubsub.publish(event.clone()).await?;

        // Send to active change streams
        let senders = self.stream_senders.read().await;
        for sender in senders.iter() {
            let _ = sender.try_send(event.clone());
        }

        Ok(position)
    }

    /// Record multiple events atomically
    pub async fn record_changes(&self, events: Vec<ChangeEvent>) -> StreamResult<Vec<u64>> {
        let positions = self.event_log.append_batch(events.clone()).await?;

        for event in events {
            self.pubsub.publish(event.clone()).await?;

            let senders = self.stream_senders.read().await;
            for sender in senders.iter() {
                let _ = sender.try_send(event.clone());
            }
        }

        Ok(positions)
    }

    /// Create a new change stream
    pub async fn create_stream(&self) -> ChangeStream {
        let (tx, rx) = mpsc::channel(self.config.channel_capacity);

        let mut senders = self.stream_senders.write().await;
        senders.push(tx);

        ChangeStream::new(rx)
    }

    /// Create a change stream for a specific collection
    pub async fn create_stream_for_collection(&self, collection: &str) -> ChangeStream {
        let (tx, rx) = mpsc::channel(self.config.channel_capacity);

        let mut senders = self.stream_senders.write().await;
        senders.push(tx);

        ChangeStream::for_collection(collection, rx)
    }

    /// Create a change stream with resume from token
    pub async fn create_stream_with_resume(
        &self,
        resume_token: &ResumeToken,
    ) -> StreamResult<ChangeStream> {
        let (tx, rx) = mpsc::channel(self.config.channel_capacity);

        // Replay events from the resume position
        let events = self
            .event_log
            .replay(ReplayOptions::new().from(resume_token.position + 1))
            .await?;

        // Send replayed events
        for event in events {
            let _ = tx.send(event).await;
        }

        let mut senders = self.stream_senders.write().await;
        senders.push(tx);

        Ok(ChangeStream::new(rx))
    }

    /// Subscribe to a collection
    pub async fn subscribe(&self, collection: &str) -> Subscriber {
        self.pubsub.subscribe(collection).await
    }

    /// Subscribe with filter
    pub async fn subscribe_with_filter(
        &self,
        collection: &str,
        filter: ChangeEventFilter,
    ) -> Subscriber {
        self.pubsub.subscribe_with_filter(collection, filter).await
    }

    /// Subscribe to all collections
    pub async fn subscribe_all(&self) -> Subscriber {
        self.pubsub.subscribe_all().await
    }

    /// Get the event log
    pub fn event_log(&self) -> &Arc<EventLog> {
        &self.event_log
    }

    /// Get the pubsub system
    pub fn pubsub(&self) -> &Arc<PubSub> {
        &self.pubsub
    }

    /// Cleanup inactive resources
    pub async fn cleanup(&self) {
        // Cleanup inactive subscriptions
        self.pubsub.cleanup_inactive().await;

        // Remove closed stream senders
        let mut senders = self.stream_senders.write().await;
        senders.retain(|sender| !sender.is_closed());
    }

    /// Check if compaction is needed and perform it
    pub async fn maybe_compact(&self, keep_from: u64) -> StreamResult<Option<usize>> {
        if self.event_log.needs_compaction().await {
            let removed = self.event_log.compact(keep_from).await?;
            Ok(Some(removed))
        } else {
            Ok(None)
        }
    }

    /// Get statistics about the streaming system
    pub async fn stats(&self) -> StreamStats {
        StreamStats {
            event_log_size: self.event_log.len().await,
            current_position: self.event_log.current_position(),
            last_compacted_position: self.event_log.last_compacted_position(),
            total_subscribers: self.pubsub.total_subscriber_count().await,
            buffered_events: self.pubsub.buffer_count().await,
            active_streams: self.stream_senders.read().await.len(),
        }
    }
}

impl Default for StreamManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stream Stats
// ============================================================================

/// Statistics about the streaming system
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Number of events in the log
    pub event_log_size: usize,
    /// Current position in the log
    pub current_position: u64,
    /// Last compacted position
    pub last_compacted_position: u64,
    /// Total number of subscribers
    pub total_subscribers: usize,
    /// Number of buffered events (backpressure)
    pub buffered_events: usize,
    /// Number of active change streams
    pub active_streams: usize,
}
