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
pub(crate) const _DEFAULT_TIMEOUT_MS: u64 = 30000;

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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── OperationType ───────────────────────────────────────────────────

    #[test]
    fn test_operation_type_display() {
        assert_eq!(OperationType::Insert.to_string(), "insert");
        assert_eq!(OperationType::Update.to_string(), "update");
        assert_eq!(OperationType::Delete.to_string(), "delete");
        assert_eq!(OperationType::Drop.to_string(), "drop");
        assert_eq!(OperationType::Rename.to_string(), "rename");
        assert_eq!(OperationType::CreateIndex.to_string(), "createIndex");
        assert_eq!(OperationType::DropIndex.to_string(), "dropIndex");
        assert_eq!(OperationType::Batch.to_string(), "batch");
    }

    #[test]
    fn test_operation_type_from_str() {
        assert_eq!(OperationType::from_str("insert"), Ok(OperationType::Insert));
        assert_eq!(OperationType::from_str("UPDATE"), Ok(OperationType::Update));
        assert_eq!(OperationType::from_str("Delete"), Ok(OperationType::Delete));
        assert_eq!(OperationType::from_str("createindex"), Ok(OperationType::CreateIndex));
        assert_eq!(OperationType::from_str("create_index"), Ok(OperationType::CreateIndex));
        assert_eq!(OperationType::from_str("dropindex"), Ok(OperationType::DropIndex));
        assert_eq!(OperationType::from_str("drop_index"), Ok(OperationType::DropIndex));
    }

    #[test]
    fn test_operation_type_from_str_invalid() {
        assert!(OperationType::from_str("unknown").is_err());
        assert!(OperationType::from_str("").is_err());
    }

    // ── StreamError ─────────────────────────────────────────────────────

    #[test]
    fn test_stream_error_display() {
        assert!(StreamError::StreamClosed.to_string().contains("closed"));
        assert!(StreamError::BufferOverflow.to_string().contains("overflow"));
        assert!(StreamError::Timeout.to_string().contains("timed out"));
        assert!(StreamError::Cancelled.to_string().contains("cancelled"));
        assert!(StreamError::CompactionInProgress.to_string().contains("Compaction"));
        assert!(StreamError::InvalidResumeToken("bad".into()).to_string().contains("bad"));
        assert!(StreamError::PositionNotFound(42).to_string().contains("42"));
        assert!(StreamError::SendError("err".into()).to_string().contains("err"));
        assert!(StreamError::ReceiveError("err".into()).to_string().contains("err"));
        assert!(StreamError::SubscriptionError("err".into()).to_string().contains("err"));
        assert!(StreamError::EventLogError("err".into()).to_string().contains("err"));
    }

    // ── ResumeToken ─────────────────────────────────────────────────────

    #[test]
    fn test_resume_token_new() {
        let token = ResumeToken::new(42, 1000);
        assert_eq!(token.position, 42);
        assert_eq!(token.timestamp, 1000);
        assert_eq!(token.as_str(), "42:1000");
    }

    #[test]
    fn test_resume_token_parse_valid() {
        let token = ResumeToken::parse("100:5000").unwrap();
        assert_eq!(token.position, 100);
        assert_eq!(token.timestamp, 5000);
    }

    #[test]
    fn test_resume_token_parse_invalid_format() {
        assert!(ResumeToken::parse("invalid").is_err());
        assert!(ResumeToken::parse("1:2:3").is_err());
        assert!(ResumeToken::parse("").is_err());
    }

    #[test]
    fn test_resume_token_parse_invalid_numbers() {
        assert!(ResumeToken::parse("abc:123").is_err());
        assert!(ResumeToken::parse("123:abc").is_err());
    }

    #[test]
    fn test_resume_token_display() {
        let token = ResumeToken::new(10, 20);
        assert_eq!(format!("{}", token), "10:20");
    }

    #[test]
    fn test_resume_token_equality() {
        let t1 = ResumeToken::new(1, 100);
        let t2 = ResumeToken::new(1, 100);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_resume_token_roundtrip() {
        let original = ResumeToken::new(42, 9999);
        let parsed = ResumeToken::parse(original.as_str()).unwrap();
        assert_eq!(original, parsed);
    }

    // ── ChangeEvent ─────────────────────────────────────────────────────

    #[test]
    fn test_change_event_insert() {
        let event = ChangeEvent::insert("docs", "doc1", vec![1, 2, 3], 1);
        assert_eq!(event.operation, OperationType::Insert);
        assert_eq!(event.collection, "docs");
        assert_eq!(event.document_key, Some("doc1".to_string()));
        assert!(event.full_document.is_some());
        assert_eq!(event.id, 1);
        assert!(event.timestamp > 0);
    }

    #[test]
    fn test_change_event_update() {
        let mut fields = HashMap::new();
        fields.insert("field1".to_string(), vec![4, 5]);
        let event = ChangeEvent::update(
            "docs", "doc1", Some(vec![1, 2, 3]), fields, vec!["old_field".into()], 2,
        );
        assert_eq!(event.operation, OperationType::Update);
        assert!(event.updated_fields.is_some());
        assert!(event.removed_fields.is_some());
    }

    #[test]
    fn test_change_event_delete() {
        let event = ChangeEvent::delete("docs", "doc1", 3);
        assert_eq!(event.operation, OperationType::Delete);
        assert!(event.full_document.is_none());
    }

    #[test]
    fn test_change_event_drop_collection() {
        let event = ChangeEvent::drop_collection("docs", 4);
        assert_eq!(event.operation, OperationType::Drop);
        assert!(event.document_key.is_none());
    }

    #[test]
    fn test_change_event_with_before_change() {
        let event = ChangeEvent::insert("docs", "doc1", vec![1], 1)
            .with_before_change(vec![0]);
        assert!(event.full_document_before_change.is_some());
    }

    #[test]
    fn test_change_event_with_metadata() {
        let event = ChangeEvent::insert("docs", "doc1", vec![1], 1)
            .with_metadata("source", "api")
            .with_metadata("version", "1");
        let meta = event.metadata.unwrap();
        assert_eq!(meta.get("source").unwrap(), "api");
        assert_eq!(meta.get("version").unwrap(), "1");
    }

    #[test]
    fn test_change_event_with_metadata_map() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), "val".to_string());
        let event = ChangeEvent::insert("docs", "d1", vec![], 1)
            .with_metadata_map(map);
        assert_eq!(event.metadata.unwrap().get("key").unwrap(), "val");
    }

    // ── ChangeEventFilter ───────────────────────────────────────────────

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = ChangeEventFilter::new();
        let event = ChangeEvent::insert("any", "any", vec![], 1);
        assert!(filter.matches(&event));
    }

    #[test]
    fn test_filter_by_collection() {
        let filter = ChangeEventFilter::collections(&["docs"]);
        let match_event = ChangeEvent::insert("docs", "d1", vec![], 1);
        let no_match = ChangeEvent::insert("other", "d1", vec![], 2);
        assert!(filter.matches(&match_event));
        assert!(!filter.matches(&no_match));
    }

    #[test]
    fn test_filter_by_operations() {
        let filter = ChangeEventFilter::operations(&[OperationType::Insert, OperationType::Update]);
        let insert = ChangeEvent::insert("c", "d1", vec![], 1);
        let delete = ChangeEvent::delete("c", "d1", 2);
        assert!(filter.matches(&insert));
        assert!(!filter.matches(&delete));
    }

    #[test]
    fn test_filter_by_document_key_pattern() {
        let filter = ChangeEventFilter::new()
            .with_document_key_pattern("user");
        let match_event = ChangeEvent::insert("c", "user_123", vec![], 1);
        let no_match = ChangeEvent::insert("c", "post_456", vec![], 2);
        let no_key = ChangeEvent::drop_collection("c", 3);
        assert!(filter.matches(&match_event));
        assert!(!filter.matches(&no_match));
        assert!(!filter.matches(&no_key));
    }

    #[test]
    fn test_filter_by_timestamp_range() {
        let filter = ChangeEventFilter::new()
            .with_min_timestamp(100)
            .with_max_timestamp(200);
        let mut in_range = ChangeEvent::insert("c", "d", vec![], 1);
        in_range.timestamp = 150;
        let mut before = ChangeEvent::insert("c", "d", vec![], 2);
        before.timestamp = 50;
        let mut after = ChangeEvent::insert("c", "d", vec![], 3);
        after.timestamp = 300;

        assert!(filter.matches(&in_range));
        assert!(!filter.matches(&before));
        assert!(!filter.matches(&after));
    }

    #[test]
    fn test_filter_combined() {
        let filter = ChangeEventFilter::new()
            .with_collections(&["docs"])
            .with_operations(&[OperationType::Insert]);

        let match_event = ChangeEvent::insert("docs", "d1", vec![], 1);
        let wrong_collection = ChangeEvent::insert("other", "d1", vec![], 2);
        let wrong_operation = ChangeEvent::delete("docs", "d1", 3);

        assert!(filter.matches(&match_event));
        assert!(!filter.matches(&wrong_collection));
        assert!(!filter.matches(&wrong_operation));
    }

    // ── Constants ───────────────────────────────────────────────────────

    #[test]
    fn test_constants() {
        assert!(DEFAULT_BUFFER_SIZE > 0);
        assert!(DEFAULT_CHANNEL_CAPACITY > 0);
        assert!(COMPACTION_THRESHOLD > 0);
    }

    // ── current_timestamp_millis ─────────────────────────────────────────

    #[test]
    fn test_current_timestamp_millis() {
        let ts = current_timestamp_millis();
        assert!(ts > 0);
        // Should be after 2020-01-01
        assert!(ts > 1_577_836_800_000);
    }
}
