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
            if sender.try_send(event.clone()).is_err() {
                tracing::debug!("Change stream subscriber lagging, event dropped");
            }
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
                if sender.try_send(event.clone()).is_err() {
                    tracing::debug!("Change stream subscriber lagging, event dropped");
                }
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
            if tx.send(event).await.is_err() {
                tracing::warn!("Failed to replay event to change stream");
                break;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::core::OperationType;

    fn insert_event(collection: &str) -> ChangeEvent {
        ChangeEvent::insert(collection, "key1", vec![1, 2, 3], 0)
    }

    // ====================================================================
    // Record change
    // ====================================================================

    #[tokio::test]
    async fn test_record_change() {
        let mgr = StreamManager::new();
        let pos = mgr.record_change(insert_event("docs")).await.unwrap();
        assert_eq!(pos, 0);

        let stats = mgr.stats().await;
        assert_eq!(stats.event_log_size, 1);
        assert_eq!(stats.current_position, 1);
    }

    #[tokio::test]
    async fn test_record_changes_batch() {
        let mgr = StreamManager::new();
        let events = vec![
            insert_event("docs"),
            insert_event("docs"),
            insert_event("docs"),
        ];

        let positions = mgr.record_changes(events).await.unwrap();
        assert_eq!(positions.len(), 3);
        assert_eq!(mgr.stats().await.event_log_size, 3);
    }

    // ====================================================================
    // Multi-destination publishing
    // ====================================================================

    #[tokio::test]
    async fn test_record_publishes_to_pubsub() {
        let mgr = StreamManager::new();
        let mut sub = mgr.subscribe("docs").await;

        mgr.record_change(insert_event("docs")).await.unwrap();

        let received = sub.try_recv();
        assert!(received.is_some());
    }

    #[tokio::test]
    async fn test_record_publishes_to_stream() {
        let mgr = StreamManager::new();
        let mut stream = mgr.create_stream().await;

        mgr.record_change(insert_event("docs")).await.unwrap();

        let event = stream.try_next();
        assert!(event.is_some());
    }

    // ====================================================================
    // Stream creation
    // ====================================================================

    #[tokio::test]
    async fn test_create_stream() {
        let mgr = StreamManager::new();
        let stream = mgr.create_stream().await;
        assert!(stream.collection().is_none());
        assert!(!stream.is_closed());
    }

    #[tokio::test]
    async fn test_create_stream_for_collection() {
        let mgr = StreamManager::new();
        let stream = mgr.create_stream_for_collection("docs").await;
        assert_eq!(stream.collection(), Some("docs"));
    }

    #[tokio::test]
    async fn test_stream_position_and_resume_token() {
        let mgr = StreamManager::new();
        let mut stream = mgr.create_stream().await;

        mgr.record_change(insert_event("docs")).await.unwrap();

        if let Some(event) = stream.try_next() {
            assert_eq!(stream.position(), event.id);
        }

        let token = stream.resume_token();
        assert!(token.position <= mgr.event_log().current_position());
    }

    #[tokio::test]
    async fn test_stream_close() {
        let mgr = StreamManager::new();
        let stream = mgr.create_stream().await;
        stream.close();
        assert!(stream.is_closed());
    }

    // ====================================================================
    // Resume token recovery
    // ====================================================================

    #[tokio::test]
    async fn test_create_stream_with_resume() {
        let mgr = StreamManager::new();

        // Record some events
        for _ in 0..5 {
            mgr.record_change(insert_event("docs")).await.unwrap();
        }

        // Resume from position 2 → should get events 3,4
        let token = ResumeToken::new(2, 0);
        let mut stream = mgr.create_stream_with_resume(&token).await.unwrap();

        let mut count = 0;
        while stream.try_next().is_some() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    // ====================================================================
    // Subscribe helpers
    // ====================================================================

    #[tokio::test]
    async fn test_subscribe_via_manager() {
        let mgr = StreamManager::new();
        let sub = mgr.subscribe("docs").await;
        assert!(sub.is_active());
    }

    #[tokio::test]
    async fn test_subscribe_with_filter_via_manager() {
        let mgr = StreamManager::new();
        let filter = ChangeEventFilter::operations(&[OperationType::Insert]);
        let sub = mgr.subscribe_with_filter("docs", filter).await;
        assert!(sub.is_active());
    }

    #[tokio::test]
    async fn test_subscribe_all_via_manager() {
        let mgr = StreamManager::new();
        let sub = mgr.subscribe_all().await;
        assert!(sub.is_active());
    }

    // ====================================================================
    // Cleanup
    // ====================================================================

    #[tokio::test]
    async fn test_cleanup() {
        let mgr = StreamManager::new();
        let sub = mgr.subscribe("docs").await;
        sub.unsubscribe();

        mgr.cleanup().await;
        let stats = mgr.stats().await;
        assert_eq!(stats.total_subscribers, 0);
    }

    // ====================================================================
    // Compaction
    // ====================================================================

    #[tokio::test]
    async fn test_maybe_compact_not_needed() {
        let mgr = StreamManager::new();
        let result = mgr.maybe_compact(0).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_maybe_compact_needed() {
        let config = StreamManagerConfig {
            compaction_threshold: 5,
            ..Default::default()
        };
        let mgr = StreamManager::with_config(config);

        for _ in 0..6 {
            mgr.record_change(insert_event("docs")).await.unwrap();
        }

        let result = mgr.maybe_compact(3).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), 3);
    }

    // ====================================================================
    // Stats
    // ====================================================================

    #[tokio::test]
    async fn test_stats_initial() {
        let mgr = StreamManager::new();
        let stats = mgr.stats().await;

        assert_eq!(stats.event_log_size, 0);
        assert_eq!(stats.current_position, 0);
        assert_eq!(stats.last_compacted_position, 0);
        assert_eq!(stats.total_subscribers, 0);
        assert_eq!(stats.buffered_events, 0);
        assert_eq!(stats.active_streams, 0);
    }

    #[tokio::test]
    async fn test_stats_with_activity() {
        let mgr = StreamManager::new();
        let _sub = mgr.subscribe("docs").await;
        let _stream = mgr.create_stream().await;
        mgr.record_change(insert_event("docs")).await.unwrap();

        let stats = mgr.stats().await;
        assert_eq!(stats.event_log_size, 1);
        assert_eq!(stats.total_subscribers, 1);
        assert_eq!(stats.active_streams, 1);
    }

    // ====================================================================
    // With config / components
    // ====================================================================

    #[tokio::test]
    async fn test_with_config() {
        let config = StreamManagerConfig {
            max_buffer_size: 512,
            channel_capacity: 128,
            compaction_threshold: 500,
            cleanup_interval_secs: 30,
        };
        let mgr = StreamManager::with_config(config);
        assert_eq!(mgr.config.max_buffer_size, 512);
    }

    #[tokio::test]
    async fn test_with_components() {
        let log = EventLog::new();
        let pubsub = PubSub::new();
        let mgr = StreamManager::with_components(log, pubsub);
        let stats = mgr.stats().await;
        assert_eq!(stats.event_log_size, 0);
    }

    #[tokio::test]
    async fn test_default() {
        let mgr = StreamManager::default();
        assert_eq!(mgr.stats().await.event_log_size, 0);
    }

    // ====================================================================
    // ChangeStream buffer
    // ====================================================================

    #[tokio::test]
    async fn test_stream_buffer_size() {
        let mgr = StreamManager::new();
        let stream = mgr.create_stream().await;
        assert_eq!(stream.buffer_size(), 0);
    }

    #[tokio::test]
    async fn test_stream_with_filter() {
        let (_, rx) = mpsc::channel(16);
        let filter = ChangeEventFilter::collections(&["docs"]);
        let stream = ChangeStream::new(rx).with_filter(filter);
        assert!(stream.collection().is_none()); // collection is set separately from filter
    }

    #[tokio::test]
    async fn test_stream_try_next_when_closed() {
        let mgr = StreamManager::new();
        let mut stream = mgr.create_stream().await;
        stream.close();
        assert!(stream.try_next().is_none());
    }

    #[tokio::test]
    async fn test_stream_next_timeout() {
        let mgr = StreamManager::new();
        let mut stream = mgr.create_stream().await;

        let result = stream.next_timeout(Duration::from_millis(10)).await;
        assert!(result.is_err()); // Timeout
    }

    #[tokio::test]
    async fn test_stream_next_timeout_when_closed() {
        let mgr = StreamManager::new();
        let mut stream = mgr.create_stream().await;
        stream.close();

        let result = stream.next_timeout(Duration::from_millis(10)).await;
        assert!(result.unwrap().is_none());
    }
}
