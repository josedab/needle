use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Constants
// ============================================================================

/// Maximum buffer size for change events
pub(crate) const DEFAULT_BUFFER_SIZE: usize = 1024;

/// Default channel capacity for subscribers
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// Maximum events before compaction is recommended
pub(crate) const COMPACTION_THRESHOLD: usize = 10000;

/// Default timeout for stream operations in milliseconds
#[allow(dead_code)]
pub(crate) const DEFAULT_TIMEOUT_MS: u64 = 30000;

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current timestamp in milliseconds since UNIX epoch
pub(crate) fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

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
            StreamError::BufferOverflow => {
                write!(f, "Buffer overflow - backpressure limit reached")
            }
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

        let position = parts[0].parse::<u64>().map_err(|_| {
            StreamError::InvalidResumeToken(format!("Invalid position: {}", parts[0]))
        })?;
        let timestamp = parts[1].parse::<u64>().map_err(|_| {
            StreamError::InvalidResumeToken(format!("Invalid timestamp: {}", parts[1]))
        })?;

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
