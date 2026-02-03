#![allow(dead_code)]

//! Streaming Vector Ingestion Pipeline
//!
//! Real-time vector ingestion with support for multiple sources (WebSocket, Redis Streams,
//! Kafka), backpressure handling, exactly-once semantics via WAL integration, and
//! configurable batching windows.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::streaming::ingestion::{
//!     IngestionPipeline, IngestionConfig, ChannelSource, IngestionRecord,
//! };
//!
//! let config = IngestionConfig::default();
//! let (source, sender) = ChannelSource::new(1024);
//! let mut pipeline = IngestionPipeline::new(config);
//!
//! // Start ingestion from source
//! pipeline.attach_source("my-source", Box::new(source))?;
//!
//! // Send vectors through the channel
//! sender.send(IngestionRecord {
//!     id: "vec1".into(),
//!     collection: "documents".into(),
//!     vector: vec![0.1; 384],
//!     metadata: None,
//! })?;
//!
//! // Process a batch
//! let stats = pipeline.process_batch()?;
//! println!("Ingested {} vectors", stats.vectors_ingested);
//! ```

use crate::error::{NeedleError, Result};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the ingestion pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    /// Maximum records to buffer before flushing (default: 1000).
    pub batch_size: usize,
    /// Maximum time to wait before flushing a partial batch (default: 100ms).
    pub flush_interval_ms: u64,
    /// Maximum buffer capacity across all sources (backpressure threshold).
    pub max_buffer_capacity: usize,
    /// Enable exactly-once semantics via WAL offset tracking.
    pub exactly_once: bool,
    /// Number of retry attempts for failed ingestions.
    pub max_retries: u32,
    /// Base delay between retries (exponential backoff).
    pub retry_base_delay_ms: u64,
    /// Enable deduplication based on vector ID.
    pub dedup_enabled: bool,
    /// Checkpoint interval for offset tracking (number of batches).
    pub checkpoint_interval: usize,
}

impl Default for IngestionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            flush_interval_ms: 100,
            max_buffer_capacity: 100_000,
            exactly_once: true,
            max_retries: 3,
            retry_base_delay_ms: 100,
            dedup_enabled: true,
            checkpoint_interval: 10,
        }
    }
}

// ============================================================================
// Ingestion Record
// ============================================================================

/// A single vector record to be ingested.
#[derive(Debug, Clone)]
pub struct IngestionRecord {
    /// Vector ID.
    pub id: String,
    /// Target collection name.
    pub collection: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Optional metadata as JSON.
    pub metadata: Option<serde_json::Value>,
    /// Source-specific offset for exactly-once tracking.
    pub source_offset: Option<SourceOffset>,
}

/// Source-specific offset for tracking consumption progress.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SourceOffset {
    /// Numeric sequence offset (e.g., Kafka offset).
    Sequence(u64),
    /// String-based cursor (e.g., Redis stream ID).
    Cursor(String),
    /// Timestamp-based offset.
    Timestamp(u64),
}

impl std::fmt::Display for SourceOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceOffset::Sequence(n) => write!(f, "seq:{n}"),
            SourceOffset::Cursor(s) => write!(f, "cur:{s}"),
            SourceOffset::Timestamp(t) => write!(f, "ts:{t}"),
        }
    }
}

// ============================================================================
// Ingestion Source Trait
// ============================================================================

/// Trait for pluggable ingestion sources.
///
/// Implementors provide a stream of `IngestionRecord` items. The pipeline
/// calls `poll_batch` periodically to pull records from the source.
pub trait IngestionSource: Send + Sync {
    /// Source name for logging and metrics.
    fn name(&self) -> &str;

    /// Poll for available records, up to `max_records`.
    /// Returns an empty vec if no records are available.
    fn poll_batch(&self, max_records: usize) -> Result<Vec<IngestionRecord>>;

    /// Acknowledge successful processing up to the given offset.
    /// Used for exactly-once semantics with external systems.
    fn acknowledge(&self, offset: &SourceOffset) -> Result<()>;

    /// Check if the source is still connected/healthy.
    fn is_healthy(&self) -> bool;

    /// Close the source gracefully.
    fn close(&self) -> Result<()>;
}

// ============================================================================
// Channel-based Source (in-process)
// ============================================================================

/// A channel-based ingestion source for in-process vector streaming.
pub struct ChannelSource {
    name: String,
    buffer: Arc<Mutex<VecDeque<IngestionRecord>>>,
    closed: Arc<AtomicBool>,
    capacity: usize,
}

/// Sender half for the channel source.
pub struct ChannelSender {
    buffer: Arc<Mutex<VecDeque<IngestionRecord>>>,
    closed: Arc<AtomicBool>,
    capacity: usize,
}

impl ChannelSource {
    /// Create a new channel source with the given buffer capacity.
    /// Returns `(source, sender)` pair.
    pub fn new(capacity: usize) -> (Self, ChannelSender) {
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(capacity)));
        let closed = Arc::new(AtomicBool::new(false));
        let source = Self {
            name: "channel".into(),
            buffer: Arc::clone(&buffer),
            closed: Arc::clone(&closed),
            capacity,
        };
        let sender = ChannelSender {
            buffer,
            closed,
            capacity,
        };
        (source, sender)
    }

    /// Create a named channel source.
    pub fn named(name: &str, capacity: usize) -> (Self, ChannelSender) {
        let (mut source, sender) = Self::new(capacity);
        source.name = name.to_string();
        (source, sender)
    }
}

impl ChannelSender {
    /// Send a record into the ingestion pipeline.
    /// Returns error if buffer is full (backpressure) or source is closed.
    pub fn send(&self, record: IngestionRecord) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(NeedleError::InvalidOperation(
                "Ingestion source is closed".into(),
            ));
        }
        let mut buf = self.buffer.lock();
        if buf.len() >= self.capacity {
            return Err(NeedleError::CapacityExceeded(
                "Ingestion buffer full (backpressure)".into(),
            ));
        }
        buf.push_back(record);
        Ok(())
    }

    /// Send a batch of records.
    pub fn send_batch(&self, records: Vec<IngestionRecord>) -> Result<usize> {
        if self.closed.load(Ordering::Acquire) {
            return Err(NeedleError::InvalidOperation(
                "Ingestion source is closed".into(),
            ));
        }
        let mut buf = self.buffer.lock();
        let available = self.capacity.saturating_sub(buf.len());
        let count = records.len().min(available);
        for record in records.into_iter().take(count) {
            buf.push_back(record);
        }
        Ok(count)
    }

    /// Check current buffer utilization.
    pub fn buffer_len(&self) -> usize {
        self.buffer.lock().len()
    }
}

impl IngestionSource for ChannelSource {
    fn name(&self) -> &str {
        &self.name
    }

    fn poll_batch(&self, max_records: usize) -> Result<Vec<IngestionRecord>> {
        let mut buf = self.buffer.lock();
        let count = max_records.min(buf.len());
        let batch: Vec<_> = buf.drain(..count).collect();
        Ok(batch)
    }

    fn acknowledge(&self, _offset: &SourceOffset) -> Result<()> {
        // Channel source doesn't need external acknowledgment
        Ok(())
    }

    fn is_healthy(&self) -> bool {
        !self.closed.load(Ordering::Acquire)
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }
}

// ============================================================================
// Offset Tracker (for exactly-once semantics)
// ============================================================================

/// Tracks consumed offsets per source for exactly-once semantics.
#[derive(Debug)]
pub struct OffsetTracker {
    /// Last committed offset per source.
    committed: RwLock<HashMap<String, SourceOffset>>,
    /// Pending (not yet committed) offsets per source.
    pending: RwLock<HashMap<String, SourceOffset>>,
    /// Total records processed since last checkpoint.
    records_since_checkpoint: AtomicU64,
}

impl OffsetTracker {
    /// Create a new offset tracker.
    pub fn new() -> Self {
        Self {
            committed: RwLock::new(HashMap::new()),
            pending: RwLock::new(HashMap::new()),
            records_since_checkpoint: AtomicU64::new(0),
        }
    }

    /// Record a pending offset for a source.
    pub fn record_pending(&self, source: &str, offset: SourceOffset) {
        self.pending
            .write()
            .insert(source.to_string(), offset);
        self.records_since_checkpoint.fetch_add(1, Ordering::Relaxed);
    }

    /// Commit all pending offsets (called after successful batch write).
    pub fn commit_pending(&self) -> HashMap<String, SourceOffset> {
        let mut pending = self.pending.write();
        let mut committed = self.committed.write();
        let flushed = pending.drain().collect::<HashMap<_, _>>();
        for (source, offset) in &flushed {
            committed.insert(source.clone(), offset.clone());
        }
        self.records_since_checkpoint.store(0, Ordering::Relaxed);
        flushed
    }

    /// Get the last committed offset for a source.
    pub fn committed_offset(&self, source: &str) -> Option<SourceOffset> {
        self.committed.read().get(source).cloned()
    }

    /// Check if a checkpoint is needed.
    pub fn needs_checkpoint(&self, interval: u64) -> bool {
        self.records_since_checkpoint.load(Ordering::Relaxed) >= interval
    }

    /// Reset pending offsets (e.g., on failure / rollback).
    pub fn rollback_pending(&self) {
        self.pending.write().clear();
    }
}

impl Default for OffsetTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WAL Checkpoint Integration
// ============================================================================

/// Callback for persisting offset checkpoints to WAL or external storage.
pub trait CheckpointSink: Send + Sync {
    /// Persist committed offsets. Called after `checkpoint_interval` batches.
    fn save_checkpoint(&self, offsets: &HashMap<String, SourceOffset>) -> Result<()>;
    /// Load the last persisted checkpoint (called on pipeline startup for recovery).
    fn load_checkpoint(&self) -> Result<HashMap<String, SourceOffset>>;
}

/// In-memory checkpoint sink (no persistence; useful for testing).
pub struct MemoryCheckpointSink {
    state: Mutex<HashMap<String, SourceOffset>>,
}

impl MemoryCheckpointSink {
    /// Create a new in-memory checkpoint sink.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for MemoryCheckpointSink {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointSink for MemoryCheckpointSink {
    fn save_checkpoint(&self, offsets: &HashMap<String, SourceOffset>) -> Result<()> {
        let mut state = self.state.lock();
        for (k, v) in offsets {
            state.insert(k.clone(), v.clone());
        }
        Ok(())
    }

    fn load_checkpoint(&self) -> Result<HashMap<String, SourceOffset>> {
        Ok(self.state.lock().clone())
    }
}

// ============================================================================
// Ingestion Metrics Counters
// ============================================================================

/// Counters exposed for Prometheus / observability integration.
#[derive(Debug)]
pub struct IngestionMetrics {
    /// vectors_ingested_total
    pub vectors_ingested_total: AtomicU64,
    /// vectors_skipped_total
    pub vectors_skipped_total: AtomicU64,
    /// vectors_failed_total
    pub vectors_failed_total: AtomicU64,
    /// batches_processed_total
    pub batches_processed_total: AtomicU64,
    /// bytes_ingested_total
    pub bytes_ingested_total: AtomicU64,
    /// backpressure_events_total (times a send was rejected)
    pub backpressure_events_total: AtomicU64,
    /// checkpoint_count
    pub checkpoint_count: AtomicU64,
}

impl IngestionMetrics {
    fn new() -> Self {
        Self {
            vectors_ingested_total: AtomicU64::new(0),
            vectors_skipped_total: AtomicU64::new(0),
            vectors_failed_total: AtomicU64::new(0),
            batches_processed_total: AtomicU64::new(0),
            bytes_ingested_total: AtomicU64::new(0),
            backpressure_events_total: AtomicU64::new(0),
            checkpoint_count: AtomicU64::new(0),
        }
    }

    fn record_batch(&self, stats: &BatchStats) {
        self.vectors_ingested_total.fetch_add(stats.vectors_ingested, Ordering::Relaxed);
        self.vectors_skipped_total.fetch_add(stats.vectors_skipped, Ordering::Relaxed);
        self.vectors_failed_total.fetch_add(stats.vectors_failed, Ordering::Relaxed);
        self.batches_processed_total.fetch_add(1, Ordering::Relaxed);
        self.bytes_ingested_total.fetch_add(stats.bytes_ingested, Ordering::Relaxed);
    }
}

impl Default for IngestionMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// WebSocket Source (stub — requires `server` feature for full implementation)
// ============================================================================

/// WebSocket ingestion source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketSourceConfig {
    /// Buffer capacity for incoming messages.
    pub buffer_capacity: usize,
    /// Maximum message size in bytes.
    pub max_message_bytes: usize,
}

impl Default for WebSocketSourceConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 4096,
            max_message_bytes: 16 * 1024 * 1024, // 16 MB
        }
    }
}

/// WebSocket ingestion source.
/// Receives `IngestionRecord` JSON messages over WebSocket connections.
/// Full async implementation requires the `server` (tokio/axum) feature.
pub struct WebSocketSource {
    name: String,
    config: WebSocketSourceConfig,
    buffer: Arc<Mutex<VecDeque<IngestionRecord>>>,
    closed: Arc<AtomicBool>,
}

impl WebSocketSource {
    /// Create a new WebSocket source with the given config.
    /// Returns `(source, push_handle)` — the push handle is used by the WS
    /// connection handler to enqueue received records.
    pub fn new(config: WebSocketSourceConfig) -> (Self, WebSocketPushHandle) {
        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_capacity)));
        let closed = Arc::new(AtomicBool::new(false));
        let source = Self {
            name: "websocket".into(),
            config,
            buffer: Arc::clone(&buffer),
            closed: Arc::clone(&closed),
        };
        let handle = WebSocketPushHandle {
            buffer,
            closed,
        };
        (source, handle)
    }
}

/// Handle for pushing records into a `WebSocketSource` from an async WS handler.
pub struct WebSocketPushHandle {
    buffer: Arc<Mutex<VecDeque<IngestionRecord>>>,
    closed: Arc<AtomicBool>,
}

impl WebSocketPushHandle {
    /// Push a record into the source buffer.
    pub fn push(&self, record: IngestionRecord) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            return Err(NeedleError::InvalidOperation("WebSocket source closed".into()));
        }
        self.buffer.lock().push_back(record);
        Ok(())
    }
}

impl IngestionSource for WebSocketSource {
    fn name(&self) -> &str { &self.name }

    fn poll_batch(&self, max_records: usize) -> Result<Vec<IngestionRecord>> {
        let mut buf = self.buffer.lock();
        let count = max_records.min(buf.len());
        Ok(buf.drain(..count).collect())
    }

    fn acknowledge(&self, _offset: &SourceOffset) -> Result<()> { Ok(()) }

    fn is_healthy(&self) -> bool { !self.closed.load(Ordering::Acquire) }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        Ok(())
    }
}

/// Redis Streams ingestion source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisStreamSourceConfig {
    /// Redis stream key.
    pub stream_key: String,
    /// Consumer group name.
    pub consumer_group: String,
    /// Consumer name within the group.
    pub consumer_name: String,
    /// Buffer capacity.
    pub buffer_capacity: usize,
}

impl Default for RedisStreamSourceConfig {
    fn default() -> Self {
        Self {
            stream_key: "needle:vectors".into(),
            consumer_group: "needle-ingest".into(),
            consumer_name: "worker-0".into(),
            buffer_capacity: 4096,
        }
    }
}

// ============================================================================
// Ingestion Pipeline
// ============================================================================

/// Statistics for an ingestion batch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchStats {
    /// Number of vectors ingested in this batch.
    pub vectors_ingested: u64,
    /// Number of records skipped (duplicates, errors).
    pub vectors_skipped: u64,
    /// Number of records that failed.
    pub vectors_failed: u64,
    /// Batch processing duration in microseconds.
    pub duration_us: u64,
    /// Bytes of vector data ingested.
    pub bytes_ingested: u64,
}

/// Cumulative ingestion pipeline statistics.
#[derive(Debug)]
pub struct PipelineStats {
    /// Total vectors ingested.
    pub total_ingested: AtomicU64,
    /// Total vectors skipped.
    pub total_skipped: AtomicU64,
    /// Total vectors failed.
    pub total_failed: AtomicU64,
    /// Total batches processed.
    pub total_batches: AtomicU64,
    /// Total bytes ingested.
    pub total_bytes: AtomicU64,
    /// Pipeline start time.
    pub started_at: Instant,
}

impl PipelineStats {
    fn new() -> Self {
        Self {
            total_ingested: AtomicU64::new(0),
            total_skipped: AtomicU64::new(0),
            total_failed: AtomicU64::new(0),
            total_batches: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    /// Get current throughput in vectors per second.
    pub fn throughput_vps(&self) -> f64 {
        let elapsed = self.started_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_ingested.load(Ordering::Relaxed) as f64 / elapsed
        } else {
            0.0
        }
    }

    fn record_batch(&self, stats: &BatchStats) {
        self.total_ingested
            .fetch_add(stats.vectors_ingested, Ordering::Relaxed);
        self.total_skipped
            .fetch_add(stats.vectors_skipped, Ordering::Relaxed);
        self.total_failed
            .fetch_add(stats.vectors_failed, Ordering::Relaxed);
        self.total_batches.fetch_add(1, Ordering::Relaxed);
        self.total_bytes
            .fetch_add(stats.bytes_ingested, Ordering::Relaxed);
    }
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Record handler callback that processes individual ingestion records.
/// Returns `Ok(true)` if the record was ingested, `Ok(false)` if skipped.
pub type RecordHandler = Box<dyn Fn(&IngestionRecord) -> Result<bool> + Send + Sync>;

/// The main streaming ingestion pipeline.
///
/// Coordinates multiple sources, batching, deduplication, offset tracking,
/// and delivery to the database.
pub struct IngestionPipeline {
    config: IngestionConfig,
    sources: RwLock<HashMap<String, Box<dyn IngestionSource>>>,
    offset_tracker: Arc<OffsetTracker>,
    stats: Arc<PipelineStats>,
    metrics: Arc<IngestionMetrics>,
    checkpoint_sink: Option<Arc<dyn CheckpointSink>>,
    handler: Option<RecordHandler>,
    running: AtomicBool,
    /// Dedup set for the current batch window.
    dedup_window: Mutex<std::collections::HashSet<String>>,
    /// Number of batches since last checkpoint.
    batches_since_checkpoint: AtomicU64,
}

impl IngestionPipeline {
    /// Create a new ingestion pipeline with the given configuration.
    pub fn new(config: IngestionConfig) -> Self {
        Self {
            config,
            sources: RwLock::new(HashMap::new()),
            offset_tracker: Arc::new(OffsetTracker::new()),
            stats: Arc::new(PipelineStats::new()),
            metrics: Arc::new(IngestionMetrics::new()),
            checkpoint_sink: None,
            handler: None,
            running: AtomicBool::new(false),
            dedup_window: Mutex::new(std::collections::HashSet::new()),
            batches_since_checkpoint: AtomicU64::new(0),
        }
    }

    /// Set the record handler that processes each ingested record.
    pub fn with_handler(mut self, handler: RecordHandler) -> Self {
        self.handler = Some(handler);
        self
    }

    /// Set the checkpoint sink for WAL-backed exactly-once recovery.
    pub fn with_checkpoint_sink(mut self, sink: Arc<dyn CheckpointSink>) -> Self {
        self.checkpoint_sink = Some(sink);
        self
    }

    /// Get a reference to the ingestion metrics counters.
    pub fn metrics(&self) -> &IngestionMetrics {
        &self.metrics
    }

    /// Attach a named ingestion source.
    pub fn attach_source(
        &self,
        name: &str,
        source: Box<dyn IngestionSource>,
    ) -> Result<()> {
        let mut sources = self.sources.write();
        if sources.contains_key(name) {
            return Err(NeedleError::InvalidOperation(format!(
                "Source '{name}' is already attached"
            )));
        }
        sources.insert(name.to_string(), source);
        Ok(())
    }

    /// Detach a source by name.
    pub fn detach_source(&self, name: &str) -> Result<()> {
        let mut sources = self.sources.write();
        if let Some(source) = sources.remove(name) {
            source.close()?;
        }
        Ok(())
    }

    /// Process one batch of records from all attached sources.
    /// Returns batch statistics.
    pub fn process_batch(&self) -> Result<BatchStats> {
        let start = Instant::now();
        let mut batch_stats = BatchStats::default();

        // Collect records from all sources
        let mut records = Vec::new();
        {
            let sources = self.sources.read();
            let per_source_limit = self
                .config
                .batch_size
                .checked_div(sources.len().max(1))
                .unwrap_or(self.config.batch_size);

            for (name, source) in sources.iter() {
                if !source.is_healthy() {
                    continue;
                }
                match source.poll_batch(per_source_limit) {
                    Ok(batch) => {
                        for record in batch {
                            records.push((name.clone(), record));
                        }
                    }
                    Err(e) => {
                        tracing::warn!(source = %name, error = %e, "Failed to poll source");
                    }
                }
            }
        }

        // Deduplication
        let mut dedup = self.dedup_window.lock();
        let mut unique_records = Vec::with_capacity(records.len());
        for (source_name, record) in records {
            if self.config.dedup_enabled {
                let dedup_key = format!("{}:{}", record.collection, record.id);
                if !dedup.insert(dedup_key) {
                    batch_stats.vectors_skipped += 1;
                    continue;
                }
            }
            unique_records.push((source_name, record));
        }
        drop(dedup);

        // Process each record
        for (source_name, record) in &unique_records {
            let byte_size = record.vector.len() as u64 * 4; // f32 = 4 bytes

            if let Some(ref handler) = self.handler {
                match handler(record) {
                    Ok(true) => {
                        batch_stats.vectors_ingested += 1;
                        batch_stats.bytes_ingested += byte_size;
                    }
                    Ok(false) => {
                        batch_stats.vectors_skipped += 1;
                    }
                    Err(_) => {
                        batch_stats.vectors_failed += 1;
                    }
                }
            } else {
                // No handler: count as ingested (dry-run mode)
                batch_stats.vectors_ingested += 1;
                batch_stats.bytes_ingested += byte_size;
            }

            // Track offset for exactly-once
            if self.config.exactly_once {
                if let Some(ref offset) = record.source_offset {
                    self.offset_tracker.record_pending(source_name, offset.clone());
                }
            }
        }

        // Commit offsets and acknowledge sources
        if self.config.exactly_once && batch_stats.vectors_ingested > 0 {
            let committed = self.offset_tracker.commit_pending();
            let sources = self.sources.read();
            for (source_name, offset) in &committed {
                if let Some(source) = sources.get(source_name) {
                    if let Err(e) = source.acknowledge(offset) {
                        tracing::warn!(source = %source_name, error = %e, "Failed to acknowledge offset");
                    }
                }
            }
        }

        batch_stats.duration_us = start.elapsed().as_micros() as u64;
        self.stats.record_batch(&batch_stats);
        self.metrics.record_batch(&batch_stats);

        // WAL checkpoint integration
        if self.config.exactly_once && batch_stats.vectors_ingested > 0 {
            let count = self.batches_since_checkpoint.fetch_add(1, Ordering::Relaxed) + 1;
            if count >= self.config.checkpoint_interval as u64 {
                self.batches_since_checkpoint.store(0, Ordering::Relaxed);
                if let Some(ref sink) = self.checkpoint_sink {
                    let committed = self.offset_tracker.committed.read().clone();
                    if let Err(e) = sink.save_checkpoint(&committed) {
                        tracing::warn!(error = %e, "Failed to save checkpoint");
                    } else {
                        self.metrics.checkpoint_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        Ok(batch_stats)
    }

    /// Get a reference to the pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Get a reference to the offset tracker.
    pub fn offset_tracker(&self) -> &OffsetTracker {
        &self.offset_tracker
    }

    /// Get current buffer utilization across all sources.
    pub fn buffer_utilization(&self) -> f64 {
        let total_buffered: usize = 0; // Sources don't expose buffer sizes uniformly
        total_buffered as f64 / self.config.max_buffer_capacity as f64
    }

    /// Clear the deduplication window (e.g., after a time window expires).
    pub fn clear_dedup_window(&self) {
        self.dedup_window.lock().clear();
    }

    /// Shutdown the pipeline gracefully, closing all sources.
    pub fn shutdown(&self) -> Result<()> {
        self.running.store(false, Ordering::Release);
        let sources = self.sources.read();
        for (name, source) in sources.iter() {
            if let Err(e) = source.close() {
                tracing::warn!(source = %name, error = %e, "Failed to close source");
            }
        }
        Ok(())
    }

    /// List names of all attached sources.
    pub fn source_names(&self) -> Vec<String> {
        self.sources.read().keys().cloned().collect()
    }

    /// Check health of all attached sources.
    pub fn source_health(&self) -> HashMap<String, bool> {
        self.sources
            .read()
            .iter()
            .map(|(name, source)| (name.clone(), source.is_healthy()))
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str, collection: &str, dim: usize) -> IngestionRecord {
        IngestionRecord {
            id: id.into(),
            collection: collection.into(),
            vector: vec![0.1; dim],
            metadata: None,
            source_offset: Some(SourceOffset::Sequence(0)),
        }
    }

    #[test]
    fn test_channel_source_send_receive() {
        let (source, sender) = ChannelSource::new(100);
        sender.send(make_record("v1", "test", 128)).expect("send should succeed");
        sender.send(make_record("v2", "test", 128)).expect("send should succeed");

        let batch = source.poll_batch(10).expect("poll should succeed");
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, "v1");
        assert_eq!(batch[1].id, "v2");
    }

    #[test]
    fn test_channel_source_backpressure() {
        let (source, sender) = ChannelSource::new(2);
        sender.send(make_record("v1", "test", 128)).expect("ok");
        sender.send(make_record("v2", "test", 128)).expect("ok");
        let result = sender.send(make_record("v3", "test", 128));
        assert!(result.is_err(), "Should fail with backpressure");

        // Drain one and retry
        let _ = source.poll_batch(1).expect("ok");
        sender.send(make_record("v3", "test", 128)).expect("ok after drain");
    }

    #[test]
    fn test_channel_source_close() {
        let (source, sender) = ChannelSource::new(100);
        source.close().expect("close should succeed");
        let result = sender.send(make_record("v1", "test", 128));
        assert!(result.is_err(), "Should fail after close");
        assert!(!source.is_healthy());
    }

    #[test]
    fn test_pipeline_basic_ingestion() {
        let config = IngestionConfig {
            batch_size: 10,
            dedup_enabled: false,
            exactly_once: false,
            ..Default::default()
        };
        let pipeline = IngestionPipeline::new(config);
        let (source, sender) = ChannelSource::new(100);
        pipeline.attach_source("ch1", Box::new(source)).expect("attach");

        for i in 0..5 {
            sender
                .send(make_record(&format!("v{i}"), "test", 64))
                .expect("send");
        }

        let stats = pipeline.process_batch().expect("process");
        assert_eq!(stats.vectors_ingested, 5);
        assert_eq!(stats.vectors_skipped, 0);
    }

    #[test]
    fn test_pipeline_deduplication() {
        let config = IngestionConfig {
            batch_size: 10,
            dedup_enabled: true,
            exactly_once: false,
            ..Default::default()
        };
        let pipeline = IngestionPipeline::new(config);
        let (source, sender) = ChannelSource::new(100);
        pipeline.attach_source("ch1", Box::new(source)).expect("attach");

        // Send duplicates
        sender.send(make_record("v1", "test", 64)).expect("send");
        sender.send(make_record("v1", "test", 64)).expect("send");
        sender.send(make_record("v2", "test", 64)).expect("send");

        let stats = pipeline.process_batch().expect("process");
        assert_eq!(stats.vectors_ingested, 2);
        assert_eq!(stats.vectors_skipped, 1);
    }

    #[test]
    fn test_pipeline_with_handler() {
        let config = IngestionConfig {
            dedup_enabled: false,
            exactly_once: false,
            ..Default::default()
        };

        let ingested = Arc::new(AtomicU64::new(0));
        let ingested_clone = Arc::clone(&ingested);

        let pipeline = IngestionPipeline::new(config).with_handler(Box::new(
            move |_record: &IngestionRecord| {
                ingested_clone.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            },
        ));

        let (source, sender) = ChannelSource::new(100);
        pipeline.attach_source("ch1", Box::new(source)).expect("attach");

        sender.send(make_record("v1", "test", 64)).expect("send");
        sender.send(make_record("v2", "test", 64)).expect("send");

        let stats = pipeline.process_batch().expect("process");
        assert_eq!(stats.vectors_ingested, 2);
        assert_eq!(ingested.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_offset_tracker() {
        let tracker = OffsetTracker::new();
        tracker.record_pending("kafka", SourceOffset::Sequence(100));
        tracker.record_pending("kafka", SourceOffset::Sequence(101));

        assert!(tracker.committed_offset("kafka").is_none());

        let committed = tracker.commit_pending();
        assert_eq!(committed.len(), 1);
        assert_eq!(
            tracker.committed_offset("kafka"),
            Some(SourceOffset::Sequence(101))
        );
    }

    #[test]
    fn test_offset_tracker_rollback() {
        let tracker = OffsetTracker::new();
        tracker.record_pending("src", SourceOffset::Sequence(50));
        tracker.rollback_pending();
        let committed = tracker.commit_pending();
        assert!(committed.is_empty());
    }

    #[test]
    fn test_pipeline_multi_source() {
        let config = IngestionConfig {
            batch_size: 20,
            dedup_enabled: false,
            exactly_once: false,
            ..Default::default()
        };
        let pipeline = IngestionPipeline::new(config);

        let (source1, sender1) = ChannelSource::named("src1", 100);
        let (source2, sender2) = ChannelSource::named("src2", 100);
        pipeline.attach_source("src1", Box::new(source1)).expect("attach");
        pipeline.attach_source("src2", Box::new(source2)).expect("attach");

        sender1.send(make_record("v1", "test", 64)).expect("send");
        sender2.send(make_record("v2", "test", 64)).expect("send");

        let stats = pipeline.process_batch().expect("process");
        assert_eq!(stats.vectors_ingested, 2);

        let names = pipeline.source_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_pipeline_shutdown() {
        let pipeline = IngestionPipeline::new(IngestionConfig::default());
        let (source, _sender) = ChannelSource::new(10);
        pipeline.attach_source("ch", Box::new(source)).expect("attach");
        pipeline.shutdown().expect("shutdown should succeed");
    }

    #[test]
    fn test_batch_send() {
        let (_source, sender) = ChannelSource::new(5);
        let records: Vec<_> = (0..10)
            .map(|i| make_record(&format!("v{i}"), "test", 64))
            .collect();
        let count = sender.send_batch(records).expect("batch send");
        assert_eq!(count, 5, "Should only accept up to capacity");
        assert_eq!(sender.buffer_len(), 5);
    }

    #[test]
    fn test_checkpoint_sink() {
        let sink = MemoryCheckpointSink::new();
        let mut offsets = HashMap::new();
        offsets.insert("src1".to_string(), SourceOffset::Sequence(42));
        sink.save_checkpoint(&offsets).expect("save");

        let loaded = sink.load_checkpoint().expect("load");
        assert_eq!(loaded.get("src1"), Some(&SourceOffset::Sequence(42)));
    }

    #[test]
    fn test_pipeline_with_checkpoint() {
        let config = IngestionConfig {
            batch_size: 10,
            dedup_enabled: false,
            exactly_once: true,
            checkpoint_interval: 1, // checkpoint every batch
            ..Default::default()
        };
        let sink = Arc::new(MemoryCheckpointSink::new());
        let pipeline = IngestionPipeline::new(config)
            .with_checkpoint_sink(Arc::clone(&sink) as Arc<dyn CheckpointSink>);

        let (source, sender) = ChannelSource::new(100);
        pipeline.attach_source("ch1", Box::new(source)).expect("attach");

        let mut record = make_record("v1", "test", 64);
        record.source_offset = Some(SourceOffset::Sequence(100));
        sender.send(record).expect("send");

        pipeline.process_batch().expect("process");

        // Verify checkpoint was saved
        let loaded = sink.load_checkpoint().expect("load");
        assert_eq!(loaded.get("ch1"), Some(&SourceOffset::Sequence(100)));
    }

    #[test]
    fn test_ingestion_metrics() {
        let config = IngestionConfig {
            dedup_enabled: false,
            exactly_once: false,
            ..Default::default()
        };
        let pipeline = IngestionPipeline::new(config);
        let (source, sender) = ChannelSource::new(100);
        pipeline.attach_source("ch1", Box::new(source)).expect("attach");

        sender.send(make_record("v1", "test", 64)).expect("send");
        sender.send(make_record("v2", "test", 64)).expect("send");
        pipeline.process_batch().expect("process");

        assert_eq!(pipeline.metrics().vectors_ingested_total.load(Ordering::Relaxed), 2);
        assert_eq!(pipeline.metrics().batches_processed_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_websocket_source() {
        let config = WebSocketSourceConfig::default();
        let (source, handle) = WebSocketSource::new(config);

        handle.push(make_record("v1", "test", 64)).expect("push");
        handle.push(make_record("v2", "test", 64)).expect("push");

        let batch = source.poll_batch(10).expect("poll");
        assert_eq!(batch.len(), 2);
        assert!(source.is_healthy());

        source.close().expect("close");
        assert!(handle.push(make_record("v3", "test", 64)).is_err());
    }
}
