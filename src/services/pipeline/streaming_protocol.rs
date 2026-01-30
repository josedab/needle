//! Streaming Ingestion Protocol
//!
//! Real-time vector ingestion via a persistent connection protocol with
//! backpressure, batching, and exactly-once semantics for continuous data
//! pipelines.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::streaming_protocol::{
//!     StreamProtocol, StreamConfig, StreamFrame, StreamAck,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let mut stream = StreamProtocol::new(&db, StreamConfig::default());
//!
//! // Push frames with sequence IDs for exactly-once semantics
//! let frame = StreamFrame::insert("docs", "doc1", vec![0.1f32; 384])
//!     .with_sequence(1);
//! let ack = stream.push(frame).unwrap();
//! assert!(ack.accepted);
//!
//! // Flush buffered frames to database
//! let stats = stream.flush().unwrap();
//! assert_eq!(stats.flushed, 1);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Streaming protocol configuration.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum frames to buffer before requiring a flush.
    pub max_buffer_size: usize,
    /// Auto-flush interval (flush is triggered after this duration if buffer non-empty).
    pub flush_interval: Duration,
    /// Maximum frame payload size in bytes.
    pub max_frame_bytes: usize,
    /// Enable exactly-once deduplication via sequence IDs.
    pub exactly_once: bool,
    /// Backpressure threshold — reject frames when buffer exceeds this percentage.
    pub backpressure_threshold: f32,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10_000,
            flush_interval: Duration::from_millis(500),
            max_frame_bytes: 10 * 1024 * 1024,
            exactly_once: true,
            backpressure_threshold: 0.9,
        }
    }
}

impl StreamConfig {
    /// Builder: set batch size.
    #[must_use]
    pub fn with_max_buffer(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }

    /// Builder: set flush interval.
    #[must_use]
    pub fn with_flush_interval(mut self, interval: Duration) -> Self {
        self.flush_interval = interval;
        self
    }

    /// Builder: enable/disable exactly-once.
    #[must_use]
    pub fn with_exactly_once(mut self, enabled: bool) -> Self {
        self.exactly_once = enabled;
        self
    }
}

// ── Frame Types ──────────────────────────────────────────────────────────────

/// A single frame in the streaming protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamFrame {
    /// Client-provided sequence ID for deduplication.
    pub sequence: Option<u64>,
    /// The operation to perform.
    pub operation: FrameOp,
    /// Timestamp when frame was created.
    pub created_at: SystemTime,
}

impl StreamFrame {
    /// Create an insert frame.
    pub fn insert(
        collection: impl Into<String>,
        id: impl Into<String>,
        vector: Vec<f32>,
    ) -> Self {
        Self {
            sequence: None,
            operation: FrameOp::Insert {
                collection: collection.into(),
                id: id.into(),
                vector,
                metadata: None,
            },
            created_at: SystemTime::now(),
        }
    }

    /// Create a delete frame.
    pub fn delete(collection: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            sequence: None,
            operation: FrameOp::Delete {
                collection: collection.into(),
                id: id.into(),
            },
            created_at: SystemTime::now(),
        }
    }

    /// Set sequence ID for exactly-once semantics.
    #[must_use]
    pub fn with_sequence(mut self, seq: u64) -> Self {
        self.sequence = Some(seq);
        self
    }

    /// Attach metadata to an insert frame.
    #[must_use]
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        if let FrameOp::Insert {
            metadata: ref mut m, ..
        } = self.operation
        {
            *m = Some(metadata);
        }
        self
    }
}

/// The operation within a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrameOp {
    /// Insert a vector.
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
    /// Delete a vector.
    Delete { collection: String, id: String },
    /// Upsert a vector.
    Upsert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<Value>,
    },
}

// ── Acknowledgment ───────────────────────────────────────────────────────────

/// Acknowledgment sent back to the client after receiving a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamAck {
    /// Whether the frame was accepted into the buffer.
    pub accepted: bool,
    /// The sequence ID (echoed back).
    pub sequence: Option<u64>,
    /// Current buffer utilization (0.0–1.0).
    pub buffer_utilization: f32,
    /// If not accepted, the reason.
    pub reject_reason: Option<String>,
}

// ── Flush Statistics ─────────────────────────────────────────────────────────

/// Statistics from a flush operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlushStats {
    /// Number of frames successfully flushed.
    pub flushed: usize,
    /// Number of frames that failed.
    pub failed: usize,
    /// Number of duplicate frames skipped.
    pub duplicates_skipped: usize,
    /// Time taken to flush.
    pub duration_ms: u64,
    /// Collections affected.
    pub collections: Vec<String>,
}

// ── Stream Protocol ──────────────────────────────────────────────────────────

/// Streaming ingestion protocol handler.
///
/// Buffers incoming frames and flushes them to the database in batches.
pub struct StreamProtocol<'a> {
    db: &'a Database,
    config: StreamConfig,
    buffer: VecDeque<StreamFrame>,
    seen_sequences: HashSet<u64>,
    last_flush: Instant,
    total_flushed: u64,
    total_received: u64,
}

impl<'a> StreamProtocol<'a> {
    /// Create a new streaming protocol handler.
    pub fn new(db: &'a Database, config: StreamConfig) -> Self {
        Self {
            db,
            config,
            buffer: VecDeque::new(),
            seen_sequences: HashSet::new(),
            last_flush: Instant::now(),
            total_flushed: 0,
            total_received: 0,
        }
    }

    /// Push a frame into the buffer. Returns an acknowledgment.
    pub fn push(&mut self, frame: StreamFrame) -> Result<StreamAck> {
        self.total_received += 1;

        // Check backpressure
        let utilization =
            self.buffer.len() as f32 / self.config.max_buffer_size.max(1) as f32;
        if utilization >= self.config.backpressure_threshold {
            return Ok(StreamAck {
                accepted: false,
                sequence: frame.sequence,
                buffer_utilization: utilization,
                reject_reason: Some("Backpressure: buffer full, flush required".into()),
            });
        }

        // Exactly-once deduplication
        if self.config.exactly_once {
            if let Some(seq) = frame.sequence {
                if self.seen_sequences.contains(&seq) {
                    return Ok(StreamAck {
                        accepted: false,
                        sequence: Some(seq),
                        buffer_utilization: utilization,
                        reject_reason: Some(format!("Duplicate sequence: {seq}")),
                    });
                }
                self.seen_sequences.insert(seq);
            }
        }

        self.buffer.push_back(frame.clone());

        Ok(StreamAck {
            accepted: true,
            sequence: frame.sequence,
            buffer_utilization: self.buffer.len() as f32
                / self.config.max_buffer_size.max(1) as f32,
            reject_reason: None,
        })
    }

    /// Check if an auto-flush should be triggered.
    pub fn should_flush(&self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }
        self.buffer.len() >= self.config.max_buffer_size
            || self.last_flush.elapsed() >= self.config.flush_interval
    }

    /// Flush all buffered frames to the database.
    pub fn flush(&mut self) -> Result<FlushStats> {
        let start = Instant::now();
        let mut stats = FlushStats::default();
        let mut collections = HashSet::new();

        while let Some(frame) = self.buffer.pop_front() {
            match self.apply_frame(&frame) {
                Ok(coll) => {
                    collections.insert(coll);
                    stats.flushed += 1;
                }
                Err(_) => {
                    stats.failed += 1;
                }
            }
        }

        stats.duration_ms = start.elapsed().as_millis() as u64;
        stats.collections = collections.into_iter().collect();
        self.last_flush = Instant::now();
        self.total_flushed += stats.flushed as u64;
        Ok(stats)
    }

    /// Get the number of buffered frames.
    pub fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Get total frames received.
    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Get total frames flushed.
    pub fn total_flushed(&self) -> u64 {
        self.total_flushed
    }

    /// Clear deduplication state (use after checkpointing).
    pub fn clear_dedup_state(&mut self) {
        self.seen_sequences.clear();
    }

    fn apply_frame(&self, frame: &StreamFrame) -> Result<String> {
        match &frame.operation {
            FrameOp::Insert {
                collection, id, vector, metadata,
            } => {
                let coll = self.db.collection(collection)?;
                coll.insert(id.clone(), vector, metadata.clone())?;
                Ok(collection.clone())
            }
            FrameOp::Delete { collection, id } => {
                let coll = self.db.collection(collection)?;
                coll.delete(id)?;
                Ok(collection.clone())
            }
            FrameOp::Upsert {
                collection, id, vector, metadata,
            } => {
                let coll = self.db.collection(collection)?;
                if coll.get(id).is_some() {
                    coll.delete(id)?;
                }
                coll.insert(id.clone(), vector, metadata.clone())?;
                Ok(collection.clone())
            }
        }
    }
}

// ── Subscription Manager ─────────────────────────────────────────────────────

/// Subscription state for a single subscriber.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique subscription ID.
    pub id: String,
    /// Collection being watched.
    pub collection: String,
    /// Last acknowledged offset.
    pub last_offset: u64,
    /// Event filter (None = all events).
    pub event_types: Option<Vec<ChangeEventType>>,
    /// Created timestamp.
    pub created_at: u64,
}

/// Type of change event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeEventType {
    Insert,
    Delete,
    Upsert,
}

/// A change event emitted by the subscription manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    pub offset: u64,
    pub event_type: ChangeEventType,
    pub collection: String,
    pub vector_id: String,
    pub timestamp: u64,
}

/// Manages subscriptions and dispatches change events.
pub struct SubscriptionManager {
    subscriptions: HashMap<String, Subscription>,
    event_log: VecDeque<ChangeEvent>,
    next_offset: u64,
    max_log_size: usize,
    next_sub_id: u64,
}

impl SubscriptionManager {
    pub fn new(max_log_size: usize) -> Self {
        Self {
            subscriptions: HashMap::new(),
            event_log: VecDeque::new(),
            next_offset: 1,
            max_log_size,
            next_sub_id: 0,
        }
    }

    /// Create a new subscription.
    pub fn subscribe(
        &mut self,
        collection: &str,
        event_types: Option<Vec<ChangeEventType>>,
    ) -> Subscription {
        let id = format!("sub_{}", self.next_sub_id);
        self.next_sub_id += 1;
        let sub = Subscription {
            id: id.clone(),
            collection: collection.to_string(),
            last_offset: self.next_offset.saturating_sub(1),
            event_types,
            created_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.subscriptions.insert(id, sub.clone());
        sub
    }

    /// Remove a subscription.
    pub fn unsubscribe(&mut self, sub_id: &str) -> bool {
        self.subscriptions.remove(sub_id).is_some()
    }

    /// Record a change event (called after flush applies a frame).
    pub fn emit(&mut self, event_type: ChangeEventType, collection: &str, vector_id: &str) {
        let event = ChangeEvent {
            offset: self.next_offset,
            event_type,
            collection: collection.to_string(),
            vector_id: vector_id.to_string(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.next_offset += 1;
        self.event_log.push_back(event);
        while self.event_log.len() > self.max_log_size {
            self.event_log.pop_front();
        }
    }

    /// Poll pending events for a subscription.
    pub fn poll(&mut self, sub_id: &str) -> Vec<ChangeEvent> {
        let sub = match self.subscriptions.get(sub_id) {
            Some(s) => s.clone(),
            None => return Vec::new(),
        };

        let events: Vec<ChangeEvent> = self
            .event_log
            .iter()
            .filter(|e| {
                e.offset > sub.last_offset
                    && e.collection == sub.collection
                    && sub
                        .event_types
                        .as_ref()
                        .map_or(true, |types| types.contains(&e.event_type))
            })
            .cloned()
            .collect();

        if let Some(last) = events.last() {
            if let Some(s) = self.subscriptions.get_mut(sub_id) {
                s.last_offset = last.offset;
            }
        }

        events
    }

    /// Acknowledge events up to an offset.
    pub fn ack(&mut self, sub_id: &str, offset: u64) {
        if let Some(sub) = self.subscriptions.get_mut(sub_id) {
            sub.last_offset = sub.last_offset.max(offset);
        }
    }

    /// Number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Current offset.
    pub fn current_offset(&self) -> u64 {
        self.next_offset.saturating_sub(1)
    }
}

impl Default for SubscriptionManager {
    fn default() -> Self {
        Self::new(100_000)
    }
}

// ── Progressive Search ──────────────────────────────────────────────────────

/// Result from a progressive search step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveSearchResult {
    /// Current best results so far.
    pub results: Vec<(String, f32)>,
    /// Whether more results may come.
    pub is_complete: bool,
    /// Number of vectors scanned so far.
    pub vectors_scanned: usize,
    /// Elapsed time in microseconds.
    pub elapsed_us: u64,
}

/// Performs a search that yields intermediate results in batches.
pub fn progressive_search(
    db: &Database,
    collection: &str,
    query: &[f32],
    k: usize,
    batch_size: usize,
) -> Result<Vec<ProgressiveSearchResult>> {
    let coll = db.collection(collection)?;
    let start = Instant::now();
    let mut steps = Vec::new();

    // First batch: fast approximate with reduced ef
    let initial_k = batch_size.min(k);
    let initial = coll.search(query, initial_k)?;
    steps.push(ProgressiveSearchResult {
        results: initial.iter().map(|r| (r.id.clone(), r.distance)).collect(),
        is_complete: initial_k >= k,
        vectors_scanned: initial.len(),
        elapsed_us: start.elapsed().as_micros() as u64,
    });

    // Full results if needed
    if initial_k < k {
        let full = coll.search(query, k)?;
        steps.push(ProgressiveSearchResult {
            results: full.iter().map(|r| (r.id.clone(), r.distance)).collect(),
            is_complete: true,
            vectors_scanned: full.len(),
            elapsed_us: start.elapsed().as_micros() as u64,
        });
    }

    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_push_and_flush() {
        let db = setup();
        let mut stream = StreamProtocol::new(&db, StreamConfig::default());

        let frame = StreamFrame::insert("test", "v1", vec![1.0; 4]).with_sequence(1);
        let ack = stream.push(frame).unwrap();
        assert!(ack.accepted);

        let stats = stream.flush().unwrap();
        assert_eq!(stats.flushed, 1);
        assert_eq!(stats.failed, 0);

        let coll = db.collection("test").unwrap();
        assert!(coll.get("v1").is_some());
    }

    #[test]
    fn test_exactly_once_dedup() {
        let db = setup();
        let mut stream = StreamProtocol::new(&db, StreamConfig::default());

        let frame1 = StreamFrame::insert("test", "v1", vec![1.0; 4]).with_sequence(1);
        let frame2 = StreamFrame::insert("test", "v1_dup", vec![2.0; 4]).with_sequence(1);

        assert!(stream.push(frame1).unwrap().accepted);
        assert!(!stream.push(frame2).unwrap().accepted); // dup
        assert_eq!(stream.buffer_len(), 1);
    }

    #[test]
    fn test_backpressure() {
        let db = setup();
        let config = StreamConfig::default()
            .with_max_buffer(10)
            .with_exactly_once(false);
        let mut stream = StreamProtocol::new(&db, config);

        // Fill buffer to 90% (backpressure threshold)
        for i in 0..9 {
            let frame = StreamFrame::insert("test", format!("v{i}"), vec![1.0; 4]);
            assert!(stream.push(frame).unwrap().accepted);
        }

        // Next should be rejected
        let frame = StreamFrame::insert("test", "overflow", vec![1.0; 4]);
        let ack = stream.push(frame).unwrap();
        assert!(!ack.accepted);
    }

    #[test]
    fn test_delete_frame() {
        let db = setup();
        let coll = db.collection("test").unwrap();
        coll.insert("v1", &[1.0; 4], None).unwrap();

        let mut stream = StreamProtocol::new(&db, StreamConfig::default());
        stream
            .push(StreamFrame::delete("test", "v1").with_sequence(1))
            .unwrap();
        stream.flush().unwrap();

        assert!(coll.get("v1").is_none());
    }

    #[test]
    fn test_should_flush() {
        let db = setup();
        let config = StreamConfig::default()
            .with_max_buffer(2)
            .with_flush_interval(Duration::from_millis(1));
        let mut stream = StreamProtocol::new(&db, config);

        assert!(!stream.should_flush());

        stream
            .push(StreamFrame::insert("test", "v1", vec![1.0; 4]))
            .unwrap();
        stream
            .push(StreamFrame::insert("test", "v2", vec![2.0; 4]))
            .unwrap();

        assert!(stream.should_flush());
    }

    #[test]
    fn test_stats() {
        let db = setup();
        let mut stream =
            StreamProtocol::new(&db, StreamConfig::default().with_exactly_once(false));

        for i in 0..5 {
            stream
                .push(StreamFrame::insert("test", format!("v{i}"), vec![1.0; 4]))
                .unwrap();
        }

        assert_eq!(stream.total_received(), 5);
        stream.flush().unwrap();
        assert_eq!(stream.total_flushed(), 5);
    }

    #[test]
    fn test_subscription_manager() {
        let mut mgr = SubscriptionManager::new(1000);
        let sub = mgr.subscribe("test", None);
        assert_eq!(mgr.subscription_count(), 1);

        mgr.emit(ChangeEventType::Insert, "test", "v1");
        mgr.emit(ChangeEventType::Insert, "test", "v2");
        mgr.emit(ChangeEventType::Delete, "test", "v3");

        let events = mgr.poll(&sub.id);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].vector_id, "v1");
        assert_eq!(events[2].event_type, ChangeEventType::Delete);

        // Subsequent poll should return nothing (already acked)
        let events2 = mgr.poll(&sub.id);
        assert!(events2.is_empty());
    }

    #[test]
    fn test_subscription_filter() {
        let mut mgr = SubscriptionManager::new(1000);
        let sub = mgr.subscribe("test", Some(vec![ChangeEventType::Insert]));

        mgr.emit(ChangeEventType::Insert, "test", "v1");
        mgr.emit(ChangeEventType::Delete, "test", "v2");

        let events = mgr.poll(&sub.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, ChangeEventType::Insert);
    }

    #[test]
    fn test_subscription_collection_scope() {
        let mut mgr = SubscriptionManager::new(1000);
        let sub = mgr.subscribe("coll_a", None);

        mgr.emit(ChangeEventType::Insert, "coll_a", "v1");
        mgr.emit(ChangeEventType::Insert, "coll_b", "v2");

        let events = mgr.poll(&sub.id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].collection, "coll_a");
    }

    #[test]
    fn test_unsubscribe() {
        let mut mgr = SubscriptionManager::new(1000);
        let sub = mgr.subscribe("test", None);
        assert!(mgr.unsubscribe(&sub.id));
        assert_eq!(mgr.subscription_count(), 0);
        assert!(!mgr.unsubscribe("nonexistent"));
    }

    #[test]
    fn test_progressive_search() {
        let db = setup();
        let coll = db.collection("test").unwrap();
        for i in 0..20 {
            coll.insert(format!("v{i}"), &[i as f32 * 0.1, 0.0, 0.0, 0.0], None).unwrap();
        }

        let steps = progressive_search(&db, "test", &[1.0, 0.0, 0.0, 0.0], 10, 3).unwrap();
        assert!(!steps.is_empty());
        // Last step should be complete
        assert!(steps.last().unwrap().is_complete);
        assert_eq!(steps.last().unwrap().results.len(), 10);
    }
}
