//! Streaming - Change Streams and Real-time Updates
//!
//! Real-time change notification and event streaming for Needle database.
//!
//! This module provides comprehensive support for:
//! - Change streams with async iteration and filtering
//! - Publish/subscribe pattern for real-time updates
//! - Event sourcing with append-only logs
//! - Change replay and compaction
//! - Backpressure handling and buffering
//!
//! # Examples
//!
//! ## Basic Change Stream
//!
//! ```rust,ignore
//! use needle::streaming::{ChangeStream, ChangeEventFilter, OperationType, StreamManager};
//!
//! // Create a stream manager
//! let manager = StreamManager::new();
//!
//! // Create a change stream for a collection
//! let mut stream = manager.create_stream_for_collection("users").await;
//!
//! // Filter for insert and update operations only
//! let filter = ChangeEventFilter::operations(&[
//!     OperationType::Insert,
//!     OperationType::Update,
//! ]);
//! let mut stream = stream.with_filter(filter);
//!
//! // Iterate over changes asynchronously
//! while let Some(event) = stream.next().await {
//!     match event.operation {
//!         OperationType::Insert => println!("New document: {:?}", event.document_key),
//!         OperationType::Update => println!("Updated: {:?}", event.document_key),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! ## Pub/Sub Pattern
//!
//! ```rust,ignore
//! use needle::streaming::PubSub;
//!
//! let pubsub = PubSub::new();
//!
//! // Subscribe to a collection
//! let mut subscriber = pubsub.subscribe("orders").await;
//!
//! // Receive updates
//! while let Some(change) = subscriber.recv().await {
//!     process_order_change(change);
//! }
//! ```
//!
//! ## Event Sourcing with Replay
//!
//! ```rust,ignore
//! use needle::streaming::{EventLog, ReplayOptions};
//!
//! let log = EventLog::new();
//!
//! // Append events
//! log.append(ChangeEvent::insert("users", "user_1", vec![1, 2, 3], 0)).await?;
//!
//! // Replay events from a specific position
//! let events = log.replay(ReplayOptions::new()
//!     .from(1000)
//!     .collection("users")
//! ).await?;
//!
//! // Compact old events
//! log.compact(5000).await?;
//! ```
//!
//! ## Resume from Token
//!
//! ```rust,ignore
//! use needle::streaming::{StreamManager, ResumeToken};
//!
//! let manager = StreamManager::new();
//!
//! // Save resume token before disconnecting
//! let token = stream.resume_token();
//! let token_str = token.as_str().to_string();
//!
//! // Later, resume from token
//! let resume_token = ResumeToken::parse(&token_str)?;
//! let mut stream = manager.create_stream_with_resume(&resume_token).await?;
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::{broadcast, mpsc, Mutex, RwLock};

// ============================================================================
// Constants
// ============================================================================

/// Maximum buffer size for change events
const DEFAULT_BUFFER_SIZE: usize = 1024;

/// Default channel capacity for subscribers
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// Maximum events before compaction is recommended
const COMPACTION_THRESHOLD: usize = 10000;

/// Default timeout for stream operations in milliseconds
#[allow(dead_code)]
const DEFAULT_TIMEOUT_MS: u64 = 30000;

// ============================================================================
// Error Types
// ============================================================================

/// Result type for streaming operations
pub type StreamResult<T> = Result<T, StreamError>;

/// Errors that can occur during streaming operations
#[derive(Debug, Clone)]
pub enum StreamError {
    /// Stream has been closed
    StreamClosed,
    /// Buffer overflow occurred
    BufferOverflow,
    /// Invalid resume token
    InvalidResumeToken(String),
    /// Subscription error
    SubscriptionError(String),
    /// Event log error
    EventLogError(String),
    /// Channel send error
    SendError(String),
    /// Channel receive error
    ReceiveError(String),
    /// Position not found in log
    PositionNotFound(u64),
    /// Compaction in progress
    CompactionInProgress,
    /// Timeout occurred
    Timeout,
    /// Operation cancelled
    Cancelled,
}

impl fmt::Display for StreamError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamError::StreamClosed => write!(f, "Stream has been closed"),
            StreamError::BufferOverflow => write!(f, "Buffer overflow - backpressure limit reached"),
            StreamError::InvalidResumeToken(t) => write!(f, "Invalid resume token: {}", t),
            StreamError::SubscriptionError(e) => write!(f, "Subscription error: {}", e),
            StreamError::EventLogError(e) => write!(f, "Event log error: {}", e),
            StreamError::SendError(e) => write!(f, "Send error: {}", e),
            StreamError::ReceiveError(e) => write!(f, "Receive error: {}", e),
            StreamError::PositionNotFound(p) => write!(f, "Position {} not found in log", p),
            StreamError::CompactionInProgress => write!(f, "Compaction is in progress"),
            StreamError::Timeout => write!(f, "Operation timed out"),
            StreamError::Cancelled => write!(f, "Operation was cancelled"),
        }
    }
}

impl std::error::Error for StreamError {}

// ============================================================================
// Operation Types
// ============================================================================

/// Types of operations that can be tracked in change streams
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    /// Document/vector insertion
    Insert,
    /// Document/vector update
    Update,
    /// Document/vector deletion
    Delete,
    /// Collection drop
    Drop,
    /// Collection rename
    Rename,
    /// Index creation
    CreateIndex,
    /// Index deletion
    DropIndex,
    /// Batch operation
    Batch,
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OperationType::Insert => write!(f, "insert"),
            OperationType::Update => write!(f, "update"),
            OperationType::Delete => write!(f, "delete"),
            OperationType::Drop => write!(f, "drop"),
            OperationType::Rename => write!(f, "rename"),
            OperationType::CreateIndex => write!(f, "createIndex"),
            OperationType::DropIndex => write!(f, "dropIndex"),
            OperationType::Batch => write!(f, "batch"),
        }
    }
}

impl FromStr for OperationType {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "insert" => Ok(OperationType::Insert),
            "update" => Ok(OperationType::Update),
            "delete" => Ok(OperationType::Delete),
            "drop" => Ok(OperationType::Drop),
            "rename" => Ok(OperationType::Rename),
            "createindex" | "create_index" => Ok(OperationType::CreateIndex),
            "dropindex" | "drop_index" => Ok(OperationType::DropIndex),
            "batch" => Ok(OperationType::Batch),
            _ => Err(()),
        }
    }
}

// ============================================================================
// Change Event
// ============================================================================

/// A change event representing a single modification in the database
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    /// Unique identifier/position for this event
    pub id: u64,
    /// Type of operation
    pub operation: OperationType,
    /// Name of the affected collection
    pub collection: String,
    /// Unique key of the affected document (if applicable)
    pub document_key: Option<String>,
    /// The full document/vector data after the change (for insert/update)
    pub full_document: Option<Vec<u8>>,
    /// Fields that were updated (for update operations)
    pub updated_fields: Option<HashMap<String, Vec<u8>>>,
    /// Fields that were removed (for update operations)
    pub removed_fields: Option<Vec<String>>,
    /// Timestamp when the change occurred (milliseconds since UNIX epoch)
    pub timestamp: u64,
    /// Resume token for this position
    pub resume_token: ResumeToken,
    /// Previous document state (if requested and available)
    pub full_document_before_change: Option<Vec<u8>>,
    /// Metadata associated with the change
    pub metadata: Option<HashMap<String, String>>,
}

impl ChangeEvent {
    /// Create a new insert event
    pub fn insert(collection: &str, document_key: &str, document: Vec<u8>, position: u64) -> Self {
        let timestamp = current_timestamp_millis();

        Self {
            id: position,
            operation: OperationType::Insert,
            collection: collection.to_string(),
            document_key: Some(document_key.to_string()),
            full_document: Some(document),
            updated_fields: None,
            removed_fields: None,
            timestamp,
            resume_token: ResumeToken::new(position, timestamp),
            full_document_before_change: None,
            metadata: None,
        }
    }

    /// Create a new update event
    pub fn update(
        collection: &str,
        document_key: &str,
        full_document: Option<Vec<u8>>,
        updated_fields: HashMap<String, Vec<u8>>,
        removed_fields: Vec<String>,
        position: u64,
    ) -> Self {
        let timestamp = current_timestamp_millis();

        Self {
            id: position,
            operation: OperationType::Update,
            collection: collection.to_string(),
            document_key: Some(document_key.to_string()),
            full_document,
            updated_fields: Some(updated_fields),
            removed_fields: Some(removed_fields),
            timestamp,
            resume_token: ResumeToken::new(position, timestamp),
            full_document_before_change: None,
            metadata: None,
        }
    }

    /// Create a new delete event
    pub fn delete(collection: &str, document_key: &str, position: u64) -> Self {
        let timestamp = current_timestamp_millis();

        Self {
            id: position,
            operation: OperationType::Delete,
            collection: collection.to_string(),
            document_key: Some(document_key.to_string()),
            full_document: None,
            updated_fields: None,
            removed_fields: None,
            timestamp,
            resume_token: ResumeToken::new(position, timestamp),
            full_document_before_change: None,
            metadata: None,
        }
    }

    /// Create a collection drop event
    pub fn drop_collection(collection: &str, position: u64) -> Self {
        let timestamp = current_timestamp_millis();

        Self {
            id: position,
            operation: OperationType::Drop,
            collection: collection.to_string(),
            document_key: None,
            full_document: None,
            updated_fields: None,
            removed_fields: None,
            timestamp,
            resume_token: ResumeToken::new(position, timestamp),
            full_document_before_change: None,
            metadata: None,
        }
    }

    /// Set the full document before the change
    pub fn with_before_change(mut self, document: Vec<u8>) -> Self {
        self.full_document_before_change = Some(document);
        self
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata
            .get_or_insert_with(HashMap::new)
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Set multiple metadata entries
    pub fn with_metadata_map(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

// ============================================================================
// Resume Token
// ============================================================================

/// Token for resuming a change stream from a specific position
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResumeToken {
    /// Position in the event log
    pub position: u64,
    /// Timestamp when this token was created
    pub timestamp: u64,
    /// Encoded token string
    encoded: String,
}

impl ResumeToken {
    /// Create a new resume token
    pub fn new(position: u64, timestamp: u64) -> Self {
        let encoded = format!("{}:{}", position, timestamp);
        Self {
            position,
            timestamp,
            encoded,
        }
    }

    /// Parse a resume token from a string
    pub fn parse(s: &str) -> StreamResult<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(StreamError::InvalidResumeToken(format!(
                "Expected format 'position:timestamp', got '{}'",
                s
            )));
        }

        let position = parts[0]
            .parse::<u64>()
            .map_err(|_| StreamError::InvalidResumeToken(format!("Invalid position: {}", parts[0])))?;
        let timestamp = parts[1]
            .parse::<u64>()
            .map_err(|_| StreamError::InvalidResumeToken(format!("Invalid timestamp: {}", parts[1])))?;

        Ok(Self {
            position,
            timestamp,
            encoded: s.to_string(),
        })
    }

    /// Get the encoded token string
    pub fn as_str(&self) -> &str {
        &self.encoded
    }
}

impl fmt::Display for ResumeToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.encoded)
    }
}

// ============================================================================
// Change Event Filter
// ============================================================================

/// Filter for change events in a stream
#[derive(Debug, Clone, Default)]
pub struct ChangeEventFilter {
    /// Filter by collections (None means all collections)
    pub collections: Option<Vec<String>>,
    /// Filter by operation types (None means all operations)
    pub operations: Option<Vec<OperationType>>,
    /// Filter by document key pattern (substring match)
    pub document_key_pattern: Option<String>,
    /// Include full document in events
    pub full_document: bool,
    /// Include document before change
    pub full_document_before_change: bool,
    /// Minimum timestamp filter
    pub min_timestamp: Option<u64>,
    /// Maximum timestamp filter
    pub max_timestamp: Option<u64>,
}

impl ChangeEventFilter {
    /// Create a new empty filter (matches all events)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a filter for specific collections
    pub fn collections(collections: &[&str]) -> Self {
        Self {
            collections: Some(collections.iter().map(|s| s.to_string()).collect()),
            ..Default::default()
        }
    }

    /// Create a filter for specific operation types
    pub fn operations(ops: &[OperationType]) -> Self {
        Self {
            operations: Some(ops.to_vec()),
            ..Default::default()
        }
    }

    /// Add collection filter
    pub fn with_collections(mut self, collections: &[&str]) -> Self {
        self.collections = Some(collections.iter().map(|s| s.to_string()).collect());
        self
    }

    /// Add operation type filter
    pub fn with_operations(mut self, ops: &[OperationType]) -> Self {
        self.operations = Some(ops.to_vec());
        self
    }

    /// Set document key pattern filter
    pub fn with_document_key_pattern(mut self, pattern: &str) -> Self {
        self.document_key_pattern = Some(pattern.to_string());
        self
    }

    /// Request full documents in events
    pub fn with_full_document(mut self) -> Self {
        self.full_document = true;
        self
    }

    /// Request document before change
    pub fn with_full_document_before_change(mut self) -> Self {
        self.full_document_before_change = true;
        self
    }

    /// Set minimum timestamp filter
    pub fn with_min_timestamp(mut self, timestamp: u64) -> Self {
        self.min_timestamp = Some(timestamp);
        self
    }

    /// Set maximum timestamp filter
    pub fn with_max_timestamp(mut self, timestamp: u64) -> Self {
        self.max_timestamp = Some(timestamp);
        self
    }

    /// Check if an event matches this filter
    pub fn matches(&self, event: &ChangeEvent) -> bool {
        // Check collection filter
        if let Some(ref collections) = self.collections {
            if !collections.contains(&event.collection) {
                return false;
            }
        }

        // Check operation filter
        if let Some(ref operations) = self.operations {
            if !operations.contains(&event.operation) {
                return false;
            }
        }

        // Check document key pattern
        if let Some(ref pattern) = self.document_key_pattern {
            if let Some(ref key) = event.document_key {
                if !key.contains(pattern) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check timestamp filters
        if let Some(min_ts) = self.min_timestamp {
            if event.timestamp < min_ts {
                return false;
            }
        }

        if let Some(max_ts) = self.max_timestamp {
            if event.timestamp > max_ts {
                return false;
            }
        }

        true
    }
}

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
                if sub.active.load(Ordering::Relaxed) && sub.filter.matches(&event)
                    && sub.sender.try_send(event.clone()).is_err() {
                        // Buffer the event for backpressure handling
                        self.buffer_event(event.clone()).await?;
                    }
            }
        }

        // Send to global subscribers
        let global_subs = self.global_subscriptions.read().await;
        for sub in global_subs.iter() {
            if sub.active.load(Ordering::Relaxed) && sub.filter.matches(&event)
                && sub.sender.try_send(event.clone()).is_err() {
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
            .map(|s| s.iter().filter(|sub| sub.active.load(Ordering::Relaxed)).count())
            .unwrap_or(0)
    }

    /// Get total subscriber count
    pub async fn total_subscriber_count(&self) -> usize {
        let subs = self.subscriptions.read().await;
        let global = self.global_subscriptions.read().await;

        let collection_count: usize = subs
            .values()
            .map(|s| s.iter().filter(|sub| sub.active.load(Ordering::Relaxed)).count())
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

// ============================================================================
// Replay Options
// ============================================================================

/// Options for replaying events from the event log
#[derive(Debug, Clone, Default)]
pub struct ReplayOptions {
    /// Start position (inclusive)
    pub from_position: Option<u64>,
    /// End position (exclusive)
    pub to_position: Option<u64>,
    /// Filter by collection
    pub collection: Option<String>,
    /// Filter by operation types
    pub operations: Option<Vec<OperationType>>,
    /// Maximum number of events to return
    pub limit: Option<usize>,
    /// Skip first N events after filtering
    pub offset: Option<usize>,
}

impl ReplayOptions {
    /// Create new replay options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set start position
    pub fn from(mut self, position: u64) -> Self {
        self.from_position = Some(position);
        self
    }

    /// Set end position
    pub fn to(mut self, position: u64) -> Self {
        self.to_position = Some(position);
        self
    }

    /// Filter by collection
    pub fn collection(mut self, collection: &str) -> Self {
        self.collection = Some(collection.to_string());
        self
    }

    /// Filter by operations
    pub fn operations(mut self, ops: &[OperationType]) -> Self {
        self.operations = Some(ops.to_vec());
        self
    }

    /// Set limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

// ============================================================================
// Event Log (Event Sourcing)
// ============================================================================

/// Append-only event log for event sourcing
///
/// Provides persistent storage of all change events with support for:
/// - Appending new events
/// - Replaying events from any position
/// - Compacting old events
/// - Snapshots and restoration
pub struct EventLog {
    /// Events stored in memory (would be persisted in production)
    events: Arc<RwLock<Vec<ChangeEvent>>>,
    /// Current position counter
    position: AtomicU64,
    /// Whether compaction is in progress
    compacting: Arc<AtomicBool>,
    /// Compaction threshold
    compaction_threshold: usize,
    /// Last compacted position
    last_compacted_position: AtomicU64,
}

impl EventLog {
    /// Create a new event log
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            position: AtomicU64::new(0),
            compacting: Arc::new(AtomicBool::new(false)),
            compaction_threshold: COMPACTION_THRESHOLD,
            last_compacted_position: AtomicU64::new(0),
        }
    }

    /// Create with custom compaction threshold
    pub fn with_compaction_threshold(threshold: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            position: AtomicU64::new(0),
            compacting: Arc::new(AtomicBool::new(false)),
            compaction_threshold: threshold,
            last_compacted_position: AtomicU64::new(0),
        }
    }

    /// Append an event to the log
    pub async fn append(&self, mut event: ChangeEvent) -> StreamResult<u64> {
        if self.compacting.load(Ordering::Relaxed) {
            return Err(StreamError::CompactionInProgress);
        }

        let position = self.position.fetch_add(1, Ordering::SeqCst);
        event.id = position;
        event.resume_token = ResumeToken::new(position, event.timestamp);

        let mut events = self.events.write().await;
        events.push(event);

        Ok(position)
    }

    /// Append multiple events atomically
    pub async fn append_batch(&self, mut events_batch: Vec<ChangeEvent>) -> StreamResult<Vec<u64>> {
        if self.compacting.load(Ordering::Relaxed) {
            return Err(StreamError::CompactionInProgress);
        }

        let mut positions = Vec::with_capacity(events_batch.len());
        let mut events = self.events.write().await;

        for event in events_batch.iter_mut() {
            let position = self.position.fetch_add(1, Ordering::SeqCst);
            event.id = position;
            event.resume_token = ResumeToken::new(position, event.timestamp);
            events.push(event.clone());
            positions.push(position);
        }

        Ok(positions)
    }

    /// Get an event by position
    pub async fn get(&self, position: u64) -> Option<ChangeEvent> {
        let events = self.events.read().await;
        let last_compacted = self.last_compacted_position.load(Ordering::Relaxed);

        if position < last_compacted {
            return None; // Event was compacted
        }

        let index = (position - last_compacted) as usize;
        events.get(index).cloned()
    }

    /// Replay events based on options
    pub async fn replay(&self, options: ReplayOptions) -> StreamResult<Vec<ChangeEvent>> {
        let events = self.events.read().await;
        let last_compacted = self.last_compacted_position.load(Ordering::Relaxed);

        let from = options.from_position.unwrap_or(last_compacted);
        let to = options.to_position.unwrap_or(u64::MAX);

        if from < last_compacted {
            return Err(StreamError::PositionNotFound(from));
        }

        let start_index = (from.saturating_sub(last_compacted)) as usize;

        let mut result: Vec<ChangeEvent> = events
            .iter()
            .skip(start_index)
            .filter(|e| e.id >= from && e.id < to)
            .filter(|e| {
                if let Some(ref collection) = options.collection {
                    e.collection == *collection
                } else {
                    true
                }
            })
            .filter(|e| {
                if let Some(ref ops) = options.operations {
                    ops.contains(&e.operation)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        // Apply offset
        if let Some(offset) = options.offset {
            result = result.into_iter().skip(offset).collect();
        }

        // Apply limit
        if let Some(limit) = options.limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    /// Get the current position
    pub fn current_position(&self) -> u64 {
        self.position.load(Ordering::Relaxed)
    }

    /// Get the number of events in the log
    pub async fn len(&self) -> usize {
        self.events.read().await.len()
    }

    /// Check if the log is empty
    pub async fn is_empty(&self) -> bool {
        self.events.read().await.is_empty()
    }

    /// Check if compaction is needed
    pub async fn needs_compaction(&self) -> bool {
        self.len().await >= self.compaction_threshold
    }

    /// Check if compaction is in progress
    pub fn is_compacting(&self) -> bool {
        self.compacting.load(Ordering::Relaxed)
    }

    /// Get the last compacted position
    pub fn last_compacted_position(&self) -> u64 {
        self.last_compacted_position.load(Ordering::Relaxed)
    }

    /// Compact the log by removing old events
    ///
    /// This keeps only events at or after the specified position
    pub async fn compact(&self, keep_from: u64) -> StreamResult<usize> {
        if self
            .compacting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(StreamError::CompactionInProgress);
        }

        let removed = {
            let mut events = self.events.write().await;
            let last_compacted = self.last_compacted_position.load(Ordering::Relaxed);

            if keep_from <= last_compacted {
                self.compacting.store(false, Ordering::Relaxed);
                return Ok(0);
            }

            let remove_index = (keep_from - last_compacted) as usize;
            let removed_count = remove_index.min(events.len());

            events.drain(0..removed_count);
            self.last_compacted_position
                .store(keep_from, Ordering::Relaxed);

            removed_count
        };

        self.compacting.store(false, Ordering::Relaxed);
        Ok(removed)
    }

    /// Create a snapshot of the current state
    pub async fn snapshot(&self) -> EventLogSnapshot {
        let events = self.events.read().await;
        EventLogSnapshot {
            events: events.clone(),
            position: self.position.load(Ordering::Relaxed),
            last_compacted: self.last_compacted_position.load(Ordering::Relaxed),
        }
    }

    /// Restore from a snapshot
    pub async fn restore(&self, snapshot: EventLogSnapshot) -> StreamResult<()> {
        if self.compacting.load(Ordering::Relaxed) {
            return Err(StreamError::CompactionInProgress);
        }

        let mut events = self.events.write().await;
        *events = snapshot.events;
        self.position.store(snapshot.position, Ordering::Relaxed);
        self.last_compacted_position
            .store(snapshot.last_compacted, Ordering::Relaxed);

        Ok(())
    }

    /// Get events in a range
    pub async fn range(&self, from: u64, to: u64) -> StreamResult<Vec<ChangeEvent>> {
        self.replay(ReplayOptions::new().from(from).to(to)).await
    }

    /// Count events matching criteria
    pub async fn count(&self, options: ReplayOptions) -> StreamResult<usize> {
        Ok(self.replay(options).await?.len())
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Event Log Snapshot
// ============================================================================

/// Snapshot of an event log for backup/restore
#[derive(Debug, Clone)]
pub struct EventLogSnapshot {
    /// Events in the snapshot
    pub events: Vec<ChangeEvent>,
    /// Position at snapshot time
    pub position: u64,
    /// Last compacted position
    pub last_compacted: u64,
}

impl EventLogSnapshot {
    /// Get the number of events in the snapshot
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the snapshot is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
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
    config: StreamManagerConfig,
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current timestamp in milliseconds since UNIX epoch
fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_change_event_insert() {
        let event = ChangeEvent::insert("users", "user_1", vec![1, 2, 3], 0);

        assert_eq!(event.operation, OperationType::Insert);
        assert_eq!(event.collection, "users");
        assert_eq!(event.document_key, Some("user_1".to_string()));
        assert_eq!(event.full_document, Some(vec![1, 2, 3]));
        assert_eq!(event.id, 0);
    }

    #[test]
    fn test_change_event_update() {
        let mut updated_fields = HashMap::new();
        updated_fields.insert("name".to_string(), vec![4, 5, 6]);

        let event = ChangeEvent::update(
            "users",
            "user_1",
            Some(vec![1, 2, 3]),
            updated_fields.clone(),
            vec!["old_field".to_string()],
            1,
        );

        assert_eq!(event.operation, OperationType::Update);
        assert_eq!(event.updated_fields, Some(updated_fields));
        assert_eq!(event.removed_fields, Some(vec!["old_field".to_string()]));
    }

    #[test]
    fn test_change_event_delete() {
        let event = ChangeEvent::delete("users", "user_1", 2);

        assert_eq!(event.operation, OperationType::Delete);
        assert!(event.full_document.is_none());
    }

    #[test]
    fn test_change_event_with_metadata() {
        let event = ChangeEvent::insert("users", "user_1", vec![1], 0)
            .with_metadata("source", "api")
            .with_metadata("version", "1.0");

        assert!(event.metadata.is_some());
        let meta = event.metadata.unwrap();
        assert_eq!(meta.get("source"), Some(&"api".to_string()));
        assert_eq!(meta.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_resume_token() {
        let token = ResumeToken::new(100, 1234567890);

        assert_eq!(token.position, 100);
        assert_eq!(token.timestamp, 1234567890);
        assert_eq!(token.as_str(), "100:1234567890");

        let parsed = ResumeToken::parse("100:1234567890").unwrap();
        assert_eq!(parsed.position, 100);
        assert_eq!(parsed.timestamp, 1234567890);
    }

    #[test]
    fn test_resume_token_invalid() {
        assert!(ResumeToken::parse("invalid").is_err());
        assert!(ResumeToken::parse("100").is_err());
        assert!(ResumeToken::parse("100:abc").is_err());
        assert!(ResumeToken::parse("abc:123").is_err());
    }

    #[test]
    fn test_change_event_filter_collections() {
        let filter = ChangeEventFilter::collections(&["users", "orders"]);

        let users_event = ChangeEvent::insert("users", "1", vec![], 0);
        let orders_event = ChangeEvent::insert("orders", "1", vec![], 0);
        let products_event = ChangeEvent::insert("products", "1", vec![], 0);

        assert!(filter.matches(&users_event));
        assert!(filter.matches(&orders_event));
        assert!(!filter.matches(&products_event));
    }

    #[test]
    fn test_change_event_filter_operations() {
        let filter = ChangeEventFilter::operations(&[OperationType::Insert, OperationType::Update]);

        let insert_event = ChangeEvent::insert("users", "1", vec![], 0);
        let delete_event = ChangeEvent::delete("users", "1", 0);

        assert!(filter.matches(&insert_event));
        assert!(!filter.matches(&delete_event));
    }

    #[test]
    fn test_change_event_filter_combined() {
        let filter = ChangeEventFilter::new()
            .with_collections(&["users"])
            .with_operations(&[OperationType::Insert]);

        let matching_event = ChangeEvent::insert("users", "1", vec![], 0);
        let wrong_collection = ChangeEvent::insert("orders", "1", vec![], 0);
        let wrong_operation = ChangeEvent::delete("users", "1", 0);

        assert!(filter.matches(&matching_event));
        assert!(!filter.matches(&wrong_collection));
        assert!(!filter.matches(&wrong_operation));
    }

    #[test]
    fn test_change_event_filter_document_key_pattern() {
        let filter = ChangeEventFilter::new().with_document_key_pattern("user_");

        let matching = ChangeEvent::insert("users", "user_123", vec![], 0);
        let not_matching = ChangeEvent::insert("users", "admin_1", vec![], 0);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&not_matching));
    }

    #[test]
    fn test_operation_type_from_str() {
        assert_eq!("insert".parse::<OperationType>(), Ok(OperationType::Insert));
        assert_eq!("UPDATE".parse::<OperationType>(), Ok(OperationType::Update));
        assert_eq!("Delete".parse::<OperationType>(), Ok(OperationType::Delete));
        assert_eq!("createIndex".parse::<OperationType>(), Ok(OperationType::CreateIndex));
        assert_eq!("create_index".parse::<OperationType>(), Ok(OperationType::CreateIndex));
        assert_eq!("invalid".parse::<OperationType>(), Err(()));
    }

    #[tokio::test]
    async fn test_event_log_append_and_get() {
        let log = EventLog::new();

        let event1 = ChangeEvent::insert("users", "1", vec![1], 0);
        let event2 = ChangeEvent::insert("users", "2", vec![2], 0);

        let pos1 = log.append(event1).await.unwrap();
        let pos2 = log.append(event2).await.unwrap();

        assert_eq!(pos1, 0);
        assert_eq!(pos2, 1);

        let retrieved1 = log.get(0).await.unwrap();
        assert_eq!(retrieved1.document_key, Some("1".to_string()));

        let retrieved2 = log.get(1).await.unwrap();
        assert_eq!(retrieved2.document_key, Some("2".to_string()));
    }

    #[tokio::test]
    async fn test_event_log_append_batch() {
        let log = EventLog::new();

        let events: Vec<ChangeEvent> = (0..5)
            .map(|i| ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0))
            .collect();

        let positions = log.append_batch(events).await.unwrap();

        assert_eq!(positions.len(), 5);
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
        assert_eq!(log.len().await, 5);
    }

    #[tokio::test]
    async fn test_event_log_replay() {
        let log = EventLog::new();

        for i in 0..10 {
            let collection = if i % 2 == 0 { "users" } else { "orders" };
            let event = ChangeEvent::insert(collection, &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        // Replay all
        let all = log.replay(ReplayOptions::new()).await.unwrap();
        assert_eq!(all.len(), 10);

        // Replay from position
        let from_5 = log.replay(ReplayOptions::new().from(5)).await.unwrap();
        assert_eq!(from_5.len(), 5);

        // Replay with collection filter
        let users_only = log
            .replay(ReplayOptions::new().collection("users"))
            .await
            .unwrap();
        assert_eq!(users_only.len(), 5);

        // Replay with limit
        let limited = log.replay(ReplayOptions::new().limit(3)).await.unwrap();
        assert_eq!(limited.len(), 3);

        // Replay with offset
        let with_offset = log
            .replay(ReplayOptions::new().offset(2).limit(3))
            .await
            .unwrap();
        assert_eq!(with_offset.len(), 3);
        assert_eq!(with_offset[0].id, 2);
    }

    #[tokio::test]
    async fn test_event_log_compaction() {
        let log = EventLog::with_compaction_threshold(5);

        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        assert_eq!(log.len().await, 10);
        assert!(log.needs_compaction().await);

        // Compact, keeping from position 5
        let removed = log.compact(5).await.unwrap();
        assert_eq!(removed, 5);
        assert_eq!(log.len().await, 5);

        // Old events should be gone
        assert!(log.get(0).await.is_none());
        assert!(log.get(4).await.is_none());

        // New events should still be accessible
        let event5 = log.get(5).await.unwrap();
        assert_eq!(event5.document_key, Some("5".to_string()));
    }

    #[tokio::test]
    async fn test_event_log_snapshot_and_restore() {
        let log = EventLog::new();

        for i in 0..5 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        let snapshot = log.snapshot().await;
        assert_eq!(snapshot.events.len(), 5);
        assert_eq!(snapshot.position, 5);

        // Create new log and restore
        let new_log = EventLog::new();
        new_log.restore(snapshot).await.unwrap();

        assert_eq!(new_log.len().await, 5);
        assert_eq!(new_log.current_position(), 5);

        let event = new_log.get(2).await.unwrap();
        assert_eq!(event.document_key, Some("2".to_string()));
    }

    #[tokio::test]
    async fn test_pubsub_subscribe_and_publish() {
        let pubsub = PubSub::new();

        let mut subscriber = pubsub.subscribe("users").await;

        let event = ChangeEvent::insert("users", "1", vec![1, 2, 3], 0);
        pubsub.publish(event.clone()).await.unwrap();

        let received = subscriber.recv().await.unwrap();
        assert_eq!(received.document_key, Some("1".to_string()));
    }

    #[tokio::test]
    async fn test_pubsub_multiple_subscribers() {
        let pubsub = PubSub::new();

        let mut sub1 = pubsub.subscribe("users").await;
        let mut sub2 = pubsub.subscribe("users").await;

        let event = ChangeEvent::insert("users", "1", vec![1], 0);
        pubsub.publish(event).await.unwrap();

        let recv1 = sub1.recv().await.unwrap();
        let recv2 = sub2.recv().await.unwrap();

        assert_eq!(recv1.document_key, recv2.document_key);
    }

    #[tokio::test]
    async fn test_pubsub_filtered_subscription() {
        let pubsub = PubSub::new();

        let filter = ChangeEventFilter::operations(&[OperationType::Insert]);
        let mut subscriber = pubsub.subscribe_with_filter("users", filter).await;

        // Publish insert - should be received
        let insert_event = ChangeEvent::insert("users", "1", vec![1], 0);
        pubsub.publish(insert_event).await.unwrap();

        // Publish delete - should not be received (filtered out)
        let delete_event = ChangeEvent::delete("users", "1", 1);
        pubsub.publish(delete_event).await.unwrap();

        let received = subscriber.try_recv().unwrap();
        assert_eq!(received.operation, OperationType::Insert);

        // Delete should not have been received
        assert!(subscriber.try_recv().is_none());
    }

    #[tokio::test]
    async fn test_pubsub_global_subscription() {
        let pubsub = PubSub::new();

        let mut global_sub = pubsub.subscribe_all().await;

        let users_event = ChangeEvent::insert("users", "1", vec![1], 0);
        let orders_event = ChangeEvent::insert("orders", "1", vec![2], 1);

        pubsub.publish(users_event).await.unwrap();
        pubsub.publish(orders_event).await.unwrap();

        let recv1 = global_sub.recv().await.unwrap();
        let recv2 = global_sub.recv().await.unwrap();

        assert_eq!(recv1.collection, "users");
        assert_eq!(recv2.collection, "orders");
    }

    #[tokio::test]
    async fn test_pubsub_unsubscribe() {
        let pubsub = PubSub::new();

        let subscriber = pubsub.subscribe("users").await;
        assert_eq!(pubsub.subscriber_count("users").await, 1);

        subscriber.unsubscribe();
        pubsub.cleanup_inactive().await;

        assert_eq!(pubsub.subscriber_count("users").await, 0);
    }

    #[tokio::test]
    async fn test_change_stream() {
        let (tx, rx) = mpsc::channel(10);
        let mut stream = ChangeStream::new(rx);

        // Send events
        let event1 = ChangeEvent::insert("users", "1", vec![1], 0);
        let event2 = ChangeEvent::insert("users", "2", vec![2], 1);

        tx.send(event1).await.unwrap();
        tx.send(event2).await.unwrap();

        // Receive events
        let recv1 = stream.next().await.unwrap();
        assert_eq!(recv1.document_key, Some("1".to_string()));
        assert_eq!(stream.position(), 0);

        let recv2 = stream.next().await.unwrap();
        assert_eq!(recv2.document_key, Some("2".to_string()));
        assert_eq!(stream.position(), 1);
    }

    #[tokio::test]
    async fn test_change_stream_with_filter() {
        let (tx, rx) = mpsc::channel(10);
        let filter = ChangeEventFilter::collections(&["users"]);
        let mut stream = ChangeStream::new(rx).with_filter(filter);

        // Send events
        let users_event = ChangeEvent::insert("users", "1", vec![1], 0);
        let orders_event = ChangeEvent::insert("orders", "1", vec![2], 1);
        let users_event2 = ChangeEvent::insert("users", "2", vec![3], 2);

        tx.send(users_event).await.unwrap();
        tx.send(orders_event).await.unwrap();
        tx.send(users_event2).await.unwrap();

        // Should only receive users events
        let recv1 = stream.next().await.unwrap();
        assert_eq!(recv1.collection, "users");
        assert_eq!(recv1.document_key, Some("1".to_string()));

        let recv2 = stream.next().await.unwrap();
        assert_eq!(recv2.collection, "users");
        assert_eq!(recv2.document_key, Some("2".to_string()));
    }

    #[tokio::test]
    async fn test_stream_manager() {
        let manager = StreamManager::new();

        // Create subscriber
        let mut subscriber = manager.subscribe("users").await;

        // Record change
        let event = ChangeEvent::insert("users", "1", vec![1, 2, 3], 0);
        let position = manager.record_change(event).await.unwrap();

        assert_eq!(position, 0);

        // Subscriber should receive the event
        let received = subscriber.recv().await.unwrap();
        assert_eq!(received.document_key, Some("1".to_string()));

        // Event should be in the log
        let from_log = manager.event_log().get(0).await.unwrap();
        assert_eq!(from_log.document_key, Some("1".to_string()));
    }

    #[tokio::test]
    async fn test_stream_manager_record_multiple() {
        let manager = StreamManager::new();

        let events: Vec<ChangeEvent> = (0..5)
            .map(|i| ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0))
            .collect();

        let positions = manager.record_changes(events).await.unwrap();

        assert_eq!(positions.len(), 5);
        assert_eq!(manager.event_log().len().await, 5);
    }

    #[tokio::test]
    async fn test_stream_manager_resume() {
        let manager = StreamManager::new();

        // Record some events
        for i in 0..5 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            manager.record_change(event).await.unwrap();
        }

        // Create stream with resume from position 2
        let resume_token = ResumeToken::new(2, 0);
        let mut stream = manager
            .create_stream_with_resume(&resume_token)
            .await
            .unwrap();

        // Should receive events starting from position 3
        let event = stream.next().await.unwrap();
        assert_eq!(event.id, 3);

        let event = stream.next().await.unwrap();
        assert_eq!(event.id, 4);
    }

    #[tokio::test]
    async fn test_stream_manager_cleanup() {
        let manager = StreamManager::new();

        // Create subscriber and unsubscribe
        let subscriber = manager.subscribe("users").await;
        subscriber.unsubscribe();

        // Cleanup
        manager.cleanup().await;

        // Subscriber count should be 0
        assert_eq!(manager.pubsub().subscriber_count("users").await, 0);
    }

    #[tokio::test]
    async fn test_stream_manager_stats() {
        let manager = StreamManager::new();

        // Record some events
        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            manager.record_change(event).await.unwrap();
        }

        // Subscribe
        let _sub = manager.subscribe("users").await;

        let stats = manager.stats().await;
        assert_eq!(stats.event_log_size, 10);
        assert_eq!(stats.current_position, 10);
        assert_eq!(stats.total_subscribers, 1);
    }

    #[tokio::test]
    async fn test_backpressure_handling() {
        let pubsub = PubSub::with_config(10, 2); // Small channel capacity

        let mut subscriber = pubsub.subscribe("users").await;

        // Fill up the channel and buffer
        for i in 0..15 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], i as u64);
            let result = pubsub.publish(event).await;
            if i >= 12 {
                // Should start failing due to buffer overflow
                assert!(result.is_err());
            }
        }

        // Drain some events
        for _ in 0..2 {
            subscriber.recv().await;
        }

        // Flush buffer should work now
        let flushed = pubsub.flush_buffer().await.unwrap();
        assert!(flushed > 0);
    }

    #[tokio::test]
    async fn test_operation_type_display() {
        assert_eq!(format!("{}", OperationType::Insert), "insert");
        assert_eq!(format!("{}", OperationType::Update), "update");
        assert_eq!(format!("{}", OperationType::Delete), "delete");
        assert_eq!(format!("{}", OperationType::Drop), "drop");
        assert_eq!(format!("{}", OperationType::Rename), "rename");
        assert_eq!(format!("{}", OperationType::CreateIndex), "createIndex");
        assert_eq!(format!("{}", OperationType::DropIndex), "dropIndex");
        assert_eq!(format!("{}", OperationType::Batch), "batch");
    }

    #[test]
    fn test_stream_error_display() {
        assert_eq!(
            format!("{}", StreamError::StreamClosed),
            "Stream has been closed"
        );
        assert_eq!(
            format!("{}", StreamError::BufferOverflow),
            "Buffer overflow - backpressure limit reached"
        );
        assert_eq!(
            format!("{}", StreamError::InvalidResumeToken("abc".to_string())),
            "Invalid resume token: abc"
        );
        assert_eq!(format!("{}", StreamError::Timeout), "Operation timed out");
    }

    #[test]
    fn test_with_before_change() {
        let event = ChangeEvent::delete("users", "1", 0).with_before_change(vec![1, 2, 3]);

        assert_eq!(event.full_document_before_change, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_drop_collection_event() {
        let event = ChangeEvent::drop_collection("users", 0);

        assert_eq!(event.operation, OperationType::Drop);
        assert_eq!(event.collection, "users");
        assert!(event.document_key.is_none());
    }

    #[tokio::test]
    async fn test_concurrent_compaction() {
        let log = Arc::new(EventLog::new());

        // Append events
        for i in 0..100 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        // Try to compact twice simultaneously
        let log1 = Arc::clone(&log);
        let log2 = Arc::clone(&log);

        let handle1 = tokio::spawn(async move { log1.compact(50).await });
        let handle2 = tokio::spawn(async move { log2.compact(50).await });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        // One should succeed, one should fail or return 0
        let success_count = [&result1, &result2]
            .iter()
            .filter(|r| matches!(r, Ok(n) if *n > 0))
            .count();

        assert!(
            success_count <= 1,
            "Only one compaction should remove events"
        );
    }

    #[tokio::test]
    async fn test_replay_after_compaction() {
        let log = EventLog::new();

        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        // Compact first 5 events
        log.compact(5).await.unwrap();

        // Try to replay from compacted position - should fail
        let result = log.replay(ReplayOptions::new().from(0)).await;
        assert!(matches!(result, Err(StreamError::PositionNotFound(0))));

        // Replay from valid position should work
        let events = log.replay(ReplayOptions::new().from(5)).await.unwrap();
        assert_eq!(events.len(), 5);
        assert_eq!(events[0].id, 5);
    }

    #[tokio::test]
    async fn test_event_log_range() {
        let log = EventLog::new();

        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        let range = log.range(3, 7).await.unwrap();
        assert_eq!(range.len(), 4);
        assert_eq!(range[0].id, 3);
        assert_eq!(range[3].id, 6);
    }

    #[tokio::test]
    async fn test_replay_with_operations_filter() {
        let log = EventLog::new();

        log.append(ChangeEvent::insert("users", "1", vec![], 0))
            .await
            .unwrap();
        log.append(ChangeEvent::delete("users", "1", 0))
            .await
            .unwrap();
        log.append(ChangeEvent::insert("users", "2", vec![], 0))
            .await
            .unwrap();

        let inserts = log
            .replay(ReplayOptions::new().operations(&[OperationType::Insert]))
            .await
            .unwrap();

        assert_eq!(inserts.len(), 2);
        assert!(inserts.iter().all(|e| e.operation == OperationType::Insert));
    }

    #[test]
    fn test_stream_manager_config_default() {
        let config = StreamManagerConfig::default();

        assert_eq!(config.max_buffer_size, DEFAULT_BUFFER_SIZE);
        assert_eq!(config.channel_capacity, DEFAULT_CHANNEL_CAPACITY);
        assert_eq!(config.compaction_threshold, COMPACTION_THRESHOLD);
        assert_eq!(config.cleanup_interval_secs, 60);
    }

    #[tokio::test]
    async fn test_stream_manager_with_config() {
        let config = StreamManagerConfig {
            max_buffer_size: 512,
            channel_capacity: 128,
            compaction_threshold: 5000,
            cleanup_interval_secs: 30,
        };

        let manager = StreamManager::with_config(config.clone());

        // Verify the manager was created with custom config
        assert_eq!(manager.config.max_buffer_size, 512);
        assert_eq!(manager.config.channel_capacity, 128);
    }

    #[tokio::test]
    async fn test_subscriber_is_active() {
        let pubsub = PubSub::new();
        let subscriber = pubsub.subscribe("test").await;

        assert!(subscriber.is_active());

        subscriber.unsubscribe();

        assert!(!subscriber.is_active());
    }

    #[tokio::test]
    async fn test_change_stream_close() {
        let (tx, rx) = mpsc::channel(10);
        let stream = ChangeStream::new(rx);

        assert!(!stream.is_closed());

        stream.close();

        assert!(stream.is_closed());
        drop(tx);
    }

    #[tokio::test]
    async fn test_event_log_count() {
        let log = EventLog::new();

        for i in 0..10 {
            let collection = if i % 2 == 0 { "users" } else { "orders" };
            let event = ChangeEvent::insert(collection, &format!("{}", i), vec![], 0);
            log.append(event).await.unwrap();
        }

        let total_count = log.count(ReplayOptions::new()).await.unwrap();
        assert_eq!(total_count, 10);

        let users_count = log
            .count(ReplayOptions::new().collection("users"))
            .await
            .unwrap();
        assert_eq!(users_count, 5);
    }

    #[test]
    fn test_event_log_snapshot_is_empty() {
        let snapshot = EventLogSnapshot {
            events: vec![],
            position: 0,
            last_compacted: 0,
        };

        assert!(snapshot.is_empty());
        assert_eq!(snapshot.len(), 0);
    }
}
