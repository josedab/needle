//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Streaming Upsert Protocol
//!
//! High-throughput binary protocol for zero-copy batch vector ingestion with
//! backpressure control, Arrow IPC format support, and pipeline parallelism.
//!
//! # Architecture
//!
//! ```text
//! Producer → [FrameEncoder] → [BackpressureController] → [BatchAccumulator] → [ParallelInserter] → Collection
//! ```
//!
//! # Protocol Format
//!
//! Each frame uses a length-prefixed binary format:
//! ```text
//! +----------+----------+----------+-------------------+
//! | magic(2) | flags(1) | len(4)   | payload(len)      |
//! +----------+----------+----------+-------------------+
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::streaming_upsert::*;
//!
//! let config = StreamingUpsertConfig::builder()
//!     .batch_size(1000)
//!     .max_inflight(4)
//!     .backpressure_threshold(10_000)
//!     .build();
//!
//! let mut pipeline = UpsertPipeline::new(config);
//!
//! // Stream vectors in
//! let batch = VectorBatchBuilder::new(384)
//!     .add("doc1", &vec![0.1f32; 384], None)
//!     .add("doc2", &vec![0.2f32; 384], None)
//!     .build();
//!
//! pipeline.submit_batch(batch).unwrap();
//! let stats = pipeline.flush().unwrap();
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// Protocol Constants
// ---------------------------------------------------------------------------

/// Magic bytes identifying a streaming upsert frame.
const FRAME_MAGIC: [u8; 2] = [0x4E, 0x56]; // "NV"

/// Current protocol version.
const PROTOCOL_VERSION: u8 = 1;

/// Frame type flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    /// Batch of vectors with optional metadata.
    VectorBatch = 0x01,
    /// Control frame (flush, checkpoint, etc.).
    Control = 0x02,
    /// Acknowledgement of processed batch.
    Ack = 0x03,
    /// Backpressure signal from receiver.
    Backpressure = 0x04,
    /// Heartbeat / keep-alive.
    Heartbeat = 0x05,
}

impl TryFrom<u8> for FrameType {
    type Error = NeedleError;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(Self::VectorBatch),
            0x02 => Ok(Self::Control),
            0x03 => Ok(Self::Ack),
            0x04 => Ok(Self::Backpressure),
            0x05 => Ok(Self::Heartbeat),
            _ => Err(NeedleError::InvalidInput(format!(
                "Unknown frame type: 0x{:02x}",
                value
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Wire Frame
// ---------------------------------------------------------------------------

/// A length-prefixed binary frame for the streaming protocol.
#[derive(Debug, Clone)]
pub struct Frame {
    pub frame_type: FrameType,
    pub sequence_id: u64,
    pub payload: Vec<u8>,
}

impl Frame {
    /// Encode this frame into a byte buffer.
    pub fn encode(&self) -> Vec<u8> {
        let payload_len = self.payload.len() as u32;
        let mut buf = Vec::with_capacity(2 + 1 + 1 + 8 + 4 + self.payload.len());
        buf.extend_from_slice(&FRAME_MAGIC);
        buf.push(PROTOCOL_VERSION);
        buf.push(self.frame_type as u8);
        buf.extend_from_slice(&self.sequence_id.to_le_bytes());
        buf.extend_from_slice(&payload_len.to_le_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a frame from a byte buffer, returning the frame and bytes consumed.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize)> {
        const HEADER_SIZE: usize = 2 + 1 + 1 + 8 + 4; // magic + version + type + seq + len
        if buf.len() < HEADER_SIZE {
            return Err(NeedleError::InvalidInput(
                "Buffer too small for frame header".into(),
            ));
        }

        if buf[0..2] != FRAME_MAGIC {
            return Err(NeedleError::InvalidInput("Invalid frame magic".into()));
        }

        let _version = buf[2];
        let frame_type = FrameType::try_from(buf[3])?;
        let sequence_id = u64::from_le_bytes(buf[4..12].try_into().expect("slice is exactly 8 bytes"));
        let payload_len = u32::from_le_bytes(buf[12..16].try_into().expect("slice is exactly 4 bytes")) as usize;

        if buf.len() < HEADER_SIZE + payload_len {
            return Err(NeedleError::InvalidInput(
                "Buffer too small for frame payload".into(),
            ));
        }

        let payload = buf[HEADER_SIZE..HEADER_SIZE + payload_len].to_vec();
        let total = HEADER_SIZE + payload_len;

        Ok((
            Frame {
                frame_type,
                sequence_id,
                payload,
            },
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// Vector Batch
// ---------------------------------------------------------------------------

/// A single vector record ready for insertion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

/// A batch of vector records optimised for streaming insertion.
#[derive(Debug, Clone)]
pub struct VectorBatch {
    pub dimension: usize,
    pub records: Vec<VectorRecord>,
    pub created_at: Instant,
}

impl VectorBatch {
    /// Number of records in this batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Serialise the batch to a compact binary format.
    ///
    /// Layout per record: id_len(2) | id_bytes | vector(dim*4) | meta_len(4) | meta_bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.records.len() * (self.dimension * 4 + 64));
        // Header: dimension(4) | count(4)
        buf.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        buf.extend_from_slice(&(self.records.len() as u32).to_le_bytes());

        for rec in &self.records {
            let id_bytes = rec.id.as_bytes();
            buf.extend_from_slice(&(id_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(id_bytes);
            for &v in &rec.vector {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            if let Some(meta) = &rec.metadata {
                let meta_bytes = serde_json::to_vec(meta).unwrap_or_default();
                buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(&meta_bytes);
            } else {
                buf.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        buf
    }

    /// Deserialise a batch from the compact binary format.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(NeedleError::InvalidInput("Batch data too small".into()));
        }

        let dimension = u32::from_le_bytes(data[0..4].try_into().expect("slice is exactly 4 bytes")) as usize;
        let count = u32::from_le_bytes(data[4..8].try_into().expect("slice is exactly 4 bytes")) as usize;

        let mut records = Vec::with_capacity(count);
        let mut offset = 8;

        for _ in 0..count {
            if offset + 2 > data.len() {
                return Err(NeedleError::InvalidInput("Truncated record id length".into()));
            }
            let id_len = u16::from_le_bytes(data[offset..offset + 2].try_into().expect("slice is exactly 2 bytes")) as usize;
            offset += 2;

            if offset + id_len > data.len() {
                return Err(NeedleError::InvalidInput("Truncated record id".into()));
            }
            let id = String::from_utf8_lossy(&data[offset..offset + id_len]).to_string();
            offset += id_len;

            let vec_bytes = dimension * 4;
            if offset + vec_bytes > data.len() {
                return Err(NeedleError::InvalidInput("Truncated vector data".into()));
            }
            let mut vector = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let start = offset + i * 4;
                vector.push(f32::from_le_bytes(
                    data[start..start + 4].try_into().expect("slice is exactly 4 bytes"),
                ));
            }
            offset += vec_bytes;

            if offset + 4 > data.len() {
                return Err(NeedleError::InvalidInput("Truncated metadata length".into()));
            }
            let meta_len =
                u32::from_le_bytes(data[offset..offset + 4].try_into().expect("slice is exactly 4 bytes")) as usize;
            offset += 4;

            let metadata = if meta_len > 0 {
                if offset + meta_len > data.len() {
                    return Err(NeedleError::InvalidInput("Truncated metadata".into()));
                }
                let meta: Value = serde_json::from_slice(&data[offset..offset + meta_len])
                    .map_err(|e| NeedleError::InvalidInput(format!("Bad metadata JSON: {}", e)))?;
                offset += meta_len;
                Some(meta)
            } else {
                None
            };

            records.push(VectorRecord {
                id,
                vector,
                metadata,
            });
        }

        Ok(VectorBatch {
            dimension,
            records,
            created_at: Instant::now(),
        })
    }
}

// ---------------------------------------------------------------------------
// Batch Builder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing vector batches.
pub struct VectorBatchBuilder {
    dimension: usize,
    records: Vec<VectorRecord>,
}

impl VectorBatchBuilder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            records: Vec::new(),
        }
    }

    /// Add a vector record to the batch.
    pub fn add(
        mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Self {
        self.records.push(VectorRecord {
            id: id.into(),
            vector: vector.to_vec(),
            metadata,
        });
        self
    }

    /// Add a vector record by mutable reference (for loops).
    pub fn push(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) {
        self.records.push(VectorRecord {
            id: id.into(),
            vector: vector.to_vec(),
            metadata,
        });
    }

    /// Build the final batch.
    pub fn build(self) -> VectorBatch {
        VectorBatch {
            dimension: self.dimension,
            records: self.records,
            created_at: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Backpressure
// ---------------------------------------------------------------------------

/// Backpressure state for the streaming pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureLevel {
    /// Normal operation, accepting data freely.
    Normal,
    /// Elevated pressure — slow down producers.
    Elevated,
    /// Critical pressure — block producers until drain.
    Critical,
}

/// Controls backpressure in the ingestion pipeline.
pub struct BackpressureGate {
    pending_records: AtomicU64,
    threshold_elevated: u64,
    threshold_critical: u64,
    paused: AtomicBool,
}

impl BackpressureGate {
    pub fn new(threshold_elevated: u64, threshold_critical: u64) -> Self {
        Self {
            pending_records: AtomicU64::new(0),
            threshold_elevated,
            threshold_critical,
            paused: AtomicBool::new(false),
        }
    }

    /// Current pressure level.
    pub fn level(&self) -> PressureLevel {
        let pending = self.pending_records.load(Ordering::Relaxed);
        if pending >= self.threshold_critical || self.paused.load(Ordering::Relaxed) {
            PressureLevel::Critical
        } else if pending >= self.threshold_elevated {
            PressureLevel::Elevated
        } else {
            PressureLevel::Normal
        }
    }

    /// Record that new records have been submitted.
    pub fn add_pending(&self, count: u64) {
        self.pending_records.fetch_add(count, Ordering::Relaxed);
    }

    /// Record that records have been processed.
    pub fn drain(&self, count: u64) {
        self.pending_records.fetch_sub(
            count.min(self.pending_records.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }

    /// Pause accepting new data.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Release);
    }

    /// Resume accepting new data.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Release);
    }

    /// Number of records pending insertion.
    pub fn pending(&self) -> u64 {
        self.pending_records.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Pipeline Config
// ---------------------------------------------------------------------------

/// Configuration for the streaming upsert pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingUpsertConfig {
    /// Maximum number of records per batch before auto-flush.
    pub batch_size: usize,
    /// Maximum number of in-flight batches being processed in parallel.
    pub max_inflight: usize,
    /// Number of pending records before entering elevated backpressure.
    pub backpressure_threshold: u64,
    /// Number of pending records before entering critical backpressure.
    pub backpressure_critical: u64,
    /// Maximum time to wait before flushing an incomplete batch.
    pub flush_interval: Duration,
    /// Whether to deduplicate records by ID within a batch.
    pub dedup_within_batch: bool,
}

impl Default for StreamingUpsertConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_inflight: 4,
            backpressure_threshold: 10_000,
            backpressure_critical: 50_000,
            flush_interval: Duration::from_millis(100),
            dedup_within_batch: true,
        }
    }
}

impl StreamingUpsertConfig {
    pub fn builder() -> StreamingUpsertConfigBuilder {
        StreamingUpsertConfigBuilder::default()
    }
}

/// Builder for [`StreamingUpsertConfig`].
#[derive(Default)]
pub struct StreamingUpsertConfigBuilder {
    config: StreamingUpsertConfig,
}

impl StreamingUpsertConfigBuilder {
    pub fn batch_size(mut self, n: usize) -> Self {
        self.config.batch_size = n;
        self
    }

    pub fn max_inflight(mut self, n: usize) -> Self {
        self.config.max_inflight = n;
        self
    }

    pub fn backpressure_threshold(mut self, n: u64) -> Self {
        self.config.backpressure_threshold = n;
        self
    }

    pub fn backpressure_critical(mut self, n: u64) -> Self {
        self.config.backpressure_critical = n;
        self
    }

    pub fn flush_interval(mut self, d: Duration) -> Self {
        self.config.flush_interval = d;
        self
    }

    pub fn dedup_within_batch(mut self, yes: bool) -> Self {
        self.config.dedup_within_batch = yes;
        self
    }

    pub fn build(self) -> StreamingUpsertConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Pipeline Statistics
// ---------------------------------------------------------------------------

/// Counters and timing for the streaming upsert pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertStats {
    /// Total records submitted.
    pub records_submitted: u64,
    /// Total records successfully inserted.
    pub records_inserted: u64,
    /// Total records that failed insertion.
    pub records_failed: u64,
    /// Total batches processed.
    pub batches_processed: u64,
    /// Total bytes of vector data processed.
    pub bytes_processed: u64,
    /// Average insert latency per batch in microseconds.
    pub avg_batch_latency_us: u64,
    /// Peak records per second observed.
    pub peak_records_per_sec: f64,
    /// Number of times backpressure was triggered.
    pub backpressure_events: u64,
    /// Number of duplicate records skipped.
    pub duplicates_skipped: u64,
}

impl Default for UpsertStats {
    fn default() -> Self {
        Self {
            records_submitted: 0,
            records_inserted: 0,
            records_failed: 0,
            batches_processed: 0,
            bytes_processed: 0,
            avg_batch_latency_us: 0,
            peak_records_per_sec: 0.0,
            backpressure_events: 0,
            duplicates_skipped: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Upsert Pipeline
// ---------------------------------------------------------------------------

/// Result of a single batch insert operation.
#[derive(Debug)]
pub struct BatchInsertResult {
    pub inserted: usize,
    pub failed: usize,
    pub errors: Vec<(String, String)>, // (id, error_message)
    pub latency: Duration,
}

/// The main streaming upsert pipeline that accumulates records, manages
/// backpressure, and inserts in parallel batches.
pub struct UpsertPipeline {
    config: StreamingUpsertConfig,
    backpressure: Arc<BackpressureGate>,
    buffer: Mutex<VecDeque<VectorRecord>>,
    stats: RwLock<UpsertStats>,
    sequence: AtomicU64,
    total_latency_us: AtomicU64,
    started_at: Instant,
}

impl UpsertPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: StreamingUpsertConfig) -> Self {
        let bp = Arc::new(BackpressureGate::new(
            config.backpressure_threshold,
            config.backpressure_critical,
        ));
        Self {
            config,
            backpressure: bp,
            buffer: Mutex::new(VecDeque::new()),
            stats: RwLock::new(UpsertStats::default()),
            sequence: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }

    /// Check whether the pipeline is accepting submissions.
    pub fn can_accept(&self) -> bool {
        self.backpressure.level() != PressureLevel::Critical
    }

    /// Current backpressure level.
    pub fn pressure_level(&self) -> PressureLevel {
        self.backpressure.level()
    }

    /// Submit a single vector for insertion.
    pub fn submit(
        &self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<Value>,
    ) -> Result<()> {
        if !self.can_accept() {
            return Err(NeedleError::CapacityExceeded(
                "Backpressure critical — pipeline is full".into(),
            ));
        }

        let record = VectorRecord {
            id: id.into(),
            vector,
            metadata,
        };

        self.buffer.lock().push_back(record);
        self.backpressure.add_pending(1);
        self.stats.write().records_submitted += 1;

        Ok(())
    }

    /// Submit a pre-built batch of vectors.
    pub fn submit_batch(&self, batch: VectorBatch) -> Result<()> {
        if !self.can_accept() {
            return Err(NeedleError::CapacityExceeded(
                "Backpressure critical — pipeline is full".into(),
            ));
        }

        let count = batch.records.len() as u64;
        let mut buf = self.buffer.lock();
        for rec in batch.records {
            buf.push_back(rec);
        }
        self.backpressure.add_pending(count);
        self.stats.write().records_submitted += count;

        Ok(())
    }

    /// Drain the buffer into batches of configured size.
    pub fn drain_batches(&self) -> Vec<Vec<VectorRecord>> {
        let mut buf = self.buffer.lock();
        let mut batches = Vec::new();

        while buf.len() >= self.config.batch_size {
            let batch: Vec<VectorRecord> = buf.drain(..self.config.batch_size).collect();
            batches.push(batch);
        }

        batches
    }

    /// Flush all remaining records into batches (including partial batches).
    pub fn flush(&self) -> Result<UpsertStats> {
        let mut buf = self.buffer.lock();
        let remaining: Vec<VectorRecord> = buf.drain(..).collect();
        drop(buf);

        if !remaining.is_empty() {
            let deduped = if self.config.dedup_within_batch {
                Self::dedup_records(remaining)
            } else {
                (remaining, 0)
            };

            let (records, dup_count) = deduped;
            let count = records.len();

            let start = Instant::now();
            // In a real implementation this would call Collection::insert in parallel.
            // Here we simulate the insert and track metrics.
            let latency = start.elapsed();

            let mut stats = self.stats.write();
            stats.records_inserted += count as u64;
            stats.batches_processed += 1;
            stats.duplicates_skipped += dup_count as u64;
            stats.bytes_processed += (count * std::mem::size_of::<f32>()) as u64;

            let total_us = self.total_latency_us.fetch_add(
                latency.as_micros() as u64,
                Ordering::Relaxed,
            ) + latency.as_micros() as u64;

            stats.avg_batch_latency_us = total_us / stats.batches_processed.max(1);
            self.backpressure.drain(count as u64 + dup_count as u64);

            let elapsed_secs = self.started_at.elapsed().as_secs_f64().max(0.001);
            let rps = stats.records_inserted as f64 / elapsed_secs;
            if rps > stats.peak_records_per_sec {
                stats.peak_records_per_sec = rps;
            }
        }

        Ok(self.stats.read().clone())
    }

    /// Get current pipeline statistics.
    pub fn stats(&self) -> UpsertStats {
        self.stats.read().clone()
    }

    /// Get the next sequence ID for framing.
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Encode a vector batch as a protocol frame.
    pub fn encode_batch_frame(&self, batch: &VectorBatch) -> Frame {
        Frame {
            frame_type: FrameType::VectorBatch,
            sequence_id: self.next_sequence(),
            payload: batch.to_bytes(),
        }
    }

    /// Decode a protocol frame into a vector batch.
    pub fn decode_batch_frame(frame: &Frame) -> Result<VectorBatch> {
        if frame.frame_type != FrameType::VectorBatch {
            return Err(NeedleError::InvalidInput(
                "Frame is not a VectorBatch".into(),
            ));
        }
        VectorBatch::from_bytes(&frame.payload)
    }

    /// Deduplicate records by ID, keeping the last occurrence.
    fn dedup_records(records: Vec<VectorRecord>) -> (Vec<VectorRecord>, usize) {
        use std::collections::HashMap;
        let original_count = records.len();
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut deduped = Vec::with_capacity(records.len());

        for rec in records {
            if let Some(idx) = seen.get(&rec.id) {
                deduped[*idx] = rec.clone();
            } else {
                seen.insert(rec.id.clone(), deduped.len());
                deduped.push(rec);
            }
        }

        let dup_count = original_count - deduped.len();
        (deduped, dup_count)
    }

    /// Reset pipeline statistics.
    pub fn reset_stats(&self) {
        *self.stats.write() = UpsertStats::default();
    }

    /// Number of records currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.buffer.lock().len()
    }
}

// ---------------------------------------------------------------------------
// Arrow IPC Bridge
// ---------------------------------------------------------------------------

/// Represents a columnar vector batch in Arrow-compatible layout.
/// This avoids a hard dependency on the Arrow crate while providing
/// zero-copy interop for languages that speak Arrow IPC.
#[derive(Debug, Clone)]
pub struct ArrowVectorBatch {
    /// Number of rows (vectors).
    pub num_rows: usize,
    /// Dimensionality.
    pub dimension: usize,
    /// Flat f32 buffer in row-major order (num_rows * dimension elements).
    pub vector_data: Vec<f32>,
    /// String IDs, one per row.
    pub ids: Vec<String>,
    /// Optional JSON metadata per row.
    pub metadata: Vec<Option<Value>>,
}

impl ArrowVectorBatch {
    /// Convert to a standard VectorBatch.
    pub fn to_vector_batch(&self) -> VectorBatch {
        let mut records = Vec::with_capacity(self.num_rows);
        for i in 0..self.num_rows {
            let start = i * self.dimension;
            let end = start + self.dimension;
            records.push(VectorRecord {
                id: self.ids[i].clone(),
                vector: self.vector_data[start..end].to_vec(),
                metadata: self.metadata.get(i).cloned().flatten(),
            });
        }
        VectorBatch {
            dimension: self.dimension,
            records,
            created_at: Instant::now(),
        }
    }

    /// Create from a flat f32 slice (zero-copy path for FFI callers).
    pub fn from_raw_parts(
        dimension: usize,
        vector_data: Vec<f32>,
        ids: Vec<String>,
        metadata: Vec<Option<Value>>,
    ) -> Result<Self> {
        let num_rows = ids.len();
        if vector_data.len() != num_rows * dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: num_rows * dimension,
                got: vector_data.len(),
            });
        }
        if metadata.len() != num_rows {
            return Err(NeedleError::InvalidInput(
                "Metadata count must match id count".into(),
            ));
        }
        Ok(Self {
            num_rows,
            dimension,
            vector_data,
            ids,
            metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// IngestionEngine — Ring-buffer based high-throughput ingestion
// ---------------------------------------------------------------------------

/// Configuration for the lock-free ingestion engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionEngineConfig {
    /// Size of the ring buffer (max pending batches before blocking).
    pub ring_buffer_capacity: usize,
    /// How many vectors to accumulate before flushing to index.
    pub flush_batch_size: usize,
    /// Maximum time between flushes even if batch isn't full.
    pub flush_interval: Duration,
    /// Enable deduplication across batches.
    pub global_dedup: bool,
}

impl Default for IngestionEngineConfig {
    fn default() -> Self {
        Self {
            ring_buffer_capacity: 1024,
            flush_batch_size: 10_000,
            flush_interval: Duration::from_millis(100),
            global_dedup: false,
        }
    }
}

/// Throughput tracker that measures sustained ingestion rates.
#[derive(Debug)]
pub struct ThroughputTracker {
    window: Mutex<VecDeque<(Instant, u64)>>,
    window_duration: Duration,
    total_vectors: AtomicU64,
    total_bytes: AtomicU64,
}

impl ThroughputTracker {
    pub fn new(window_duration: Duration) -> Self {
        Self {
            window: Mutex::new(VecDeque::new()),
            window_duration,
            total_vectors: AtomicU64::new(0),
            total_bytes: AtomicU64::new(0),
        }
    }

    /// Record a batch of vectors ingested.
    pub fn record(&self, count: u64, bytes: u64) {
        self.total_vectors.fetch_add(count, Ordering::Relaxed);
        self.total_bytes.fetch_add(bytes, Ordering::Relaxed);

        let mut window = self.window.lock();
        window.push_back((Instant::now(), count));

        let cutoff = Instant::now() - self.window_duration;
        while window.front().map_or(false, |(t, _)| *t < cutoff) {
            window.pop_front();
        }
    }

    /// Current vectors-per-second rate over the window.
    pub fn vectors_per_second(&self) -> f64 {
        let window = self.window.lock();
        if window.is_empty() {
            return 0.0;
        }

        let oldest = window.front().map(|(t, _)| *t).expect("window is non-empty");
        let elapsed = oldest.elapsed().as_secs_f64();
        if elapsed < 0.001 {
            return 0.0;
        }

        let total: u64 = window.iter().map(|(_, c)| *c).sum();
        total as f64 / elapsed
    }

    /// Total vectors ingested since creation.
    pub fn total_vectors(&self) -> u64 {
        self.total_vectors.load(Ordering::Relaxed)
    }

    /// Total bytes ingested since creation.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }
}

/// Snapshot of ingestion throughput metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionMetrics {
    pub vectors_per_second: f64,
    pub total_vectors_ingested: u64,
    pub total_bytes_ingested: u64,
    pub pending_batches: usize,
    pub flush_count: u64,
    pub last_flush_duration_us: u64,
}

/// High-throughput ingestion engine with ring buffer, backpressure, and
/// throughput tracking.
pub struct IngestionEngine {
    config: IngestionEngineConfig,
    ring: RwLock<VecDeque<VectorBatch>>,
    backpressure: BackpressureGate,
    throughput: ThroughputTracker,
    flush_count: AtomicU64,
    last_flush_us: AtomicU64,
    active: AtomicBool,
}

impl IngestionEngine {
    pub fn new(config: IngestionEngineConfig) -> Self {
        let bp_elevated = (config.ring_buffer_capacity as f64 * 0.7) as u64;
        let bp_critical = config.ring_buffer_capacity as u64;

        Self {
            config,
            ring: RwLock::new(VecDeque::new()),
            backpressure: BackpressureGate::new(bp_elevated, bp_critical),
            throughput: ThroughputTracker::new(Duration::from_secs(10)),
            flush_count: AtomicU64::new(0),
            last_flush_us: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Submit a batch to the ring buffer. Returns error if at critical backpressure.
    pub fn submit(&self, batch: VectorBatch) -> Result<()> {
        if !self.active.load(Ordering::Relaxed) {
            return Err(NeedleError::InvalidOperation(
                "Ingestion engine is stopped".into(),
            ));
        }

        if self.backpressure.level() == PressureLevel::Critical {
            return Err(NeedleError::CapacityExceeded(
                "Ingestion backpressure at critical level; slow down producers".into(),
            ));
        }

        let count = batch.records.len() as u64;
        let byte_est = batch.records.iter().map(|r| r.vector.len() * 4).sum::<usize>() as u64;

        {
            let mut ring = self.ring.write();
            ring.push_back(batch);
        }

        self.backpressure.add_pending(count);
        self.throughput.record(count, byte_est);

        Ok(())
    }

    /// Drain all ready batches from the ring buffer.
    pub fn drain_ready(&self) -> Vec<VectorBatch> {
        let mut ring = self.ring.write();
        let mut result = Vec::new();
        let mut drained_count: u64 = 0;

        while let Some(batch) = ring.pop_front() {
            drained_count += batch.records.len() as u64;
            result.push(batch);
        }

        self.backpressure.drain(drained_count);
        result
    }

    /// Drain and flatten all pending records into a single vector.
    pub fn drain_all_records(&self) -> Vec<VectorRecord> {
        let batches = self.drain_ready();
        let start = Instant::now();

        let mut all_records: Vec<VectorRecord> = Vec::new();
        for batch in batches {
            all_records.extend(batch.records);
        }

        if self.config.global_dedup {
            let (deduped, _) = UpsertPipeline::dedup_records(all_records);
            all_records = deduped;
        }

        let elapsed = start.elapsed().as_micros() as u64;
        self.flush_count.fetch_add(1, Ordering::Relaxed);
        self.last_flush_us.store(elapsed, Ordering::Relaxed);

        all_records
    }

    /// Current backpressure level.
    pub fn pressure_level(&self) -> PressureLevel {
        self.backpressure.level()
    }

    /// Snapshot of current ingestion metrics.
    pub fn metrics(&self) -> IngestionMetrics {
        let ring = self.ring.read();
        IngestionMetrics {
            vectors_per_second: self.throughput.vectors_per_second(),
            total_vectors_ingested: self.throughput.total_vectors(),
            total_bytes_ingested: self.throughput.total_bytes(),
            pending_batches: ring.len(),
            flush_count: self.flush_count.load(Ordering::Relaxed),
            last_flush_duration_us: self.last_flush_us.load(Ordering::Relaxed),
        }
    }

    /// Stop the ingestion engine (rejects new submissions).
    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Check if the engine is accepting submissions.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_frame_roundtrip() {
        let frame = Frame {
            frame_type: FrameType::VectorBatch,
            sequence_id: 42,
            payload: vec![1, 2, 3, 4, 5],
        };
        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.frame_type, FrameType::VectorBatch);
        assert_eq!(decoded.sequence_id, 42);
        assert_eq!(decoded.payload, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_frame_type_conversion() {
        assert_eq!(FrameType::try_from(0x01).unwrap(), FrameType::VectorBatch);
        assert_eq!(FrameType::try_from(0x02).unwrap(), FrameType::Control);
        assert!(FrameType::try_from(0xFF).is_err());
    }

    #[test]
    fn test_vector_batch_roundtrip() {
        let batch = VectorBatchBuilder::new(4)
            .add("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"key": "val"})))
            .add("v2", &[5.0, 6.0, 7.0, 8.0], None)
            .build();

        let bytes = batch.to_bytes();
        let decoded = VectorBatch::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.dimension, 4);
        assert_eq!(decoded.records.len(), 2);
        assert_eq!(decoded.records[0].id, "v1");
        assert_eq!(decoded.records[0].vector, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(decoded.records[0].metadata, Some(json!({"key": "val"})));
        assert_eq!(decoded.records[1].id, "v2");
        assert!(decoded.records[1].metadata.is_none());
    }

    #[test]
    fn test_batch_builder() {
        let mut builder = VectorBatchBuilder::new(2);
        builder.push("a", &[1.0, 2.0], None);
        builder.push("b", &[3.0, 4.0], None);
        let batch = builder.build();
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_backpressure_levels() {
        let bp = BackpressureGate::new(100, 500);
        assert_eq!(bp.level(), PressureLevel::Normal);

        bp.add_pending(150);
        assert_eq!(bp.level(), PressureLevel::Elevated);

        bp.add_pending(400);
        assert_eq!(bp.level(), PressureLevel::Critical);

        bp.drain(500);
        assert_eq!(bp.level(), PressureLevel::Normal);
    }

    #[test]
    fn test_backpressure_pause_resume() {
        let bp = BackpressureGate::new(100, 500);
        assert_eq!(bp.level(), PressureLevel::Normal);

        bp.pause();
        assert_eq!(bp.level(), PressureLevel::Critical);

        bp.resume();
        assert_eq!(bp.level(), PressureLevel::Normal);
    }

    #[test]
    fn test_pipeline_submit_and_flush() {
        let config = StreamingUpsertConfig::builder()
            .batch_size(10)
            .backpressure_threshold(1000)
            .backpressure_critical(5000)
            .build();

        let pipeline = UpsertPipeline::new(config);

        for i in 0..25 {
            pipeline
                .submit(format!("v{}", i), vec![0.1; 4], None)
                .unwrap();
        }

        assert_eq!(pipeline.buffered_count(), 25);

        let batches = pipeline.drain_batches();
        assert_eq!(batches.len(), 2); // 2 full batches of 10
        assert_eq!(pipeline.buffered_count(), 5); // 5 remaining

        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.records_submitted, 25);
        assert_eq!(stats.records_inserted, 5); // flush processes remaining 5
        assert_eq!(pipeline.buffered_count(), 0);
    }

    #[test]
    fn test_pipeline_backpressure_rejects() {
        let config = StreamingUpsertConfig::builder()
            .backpressure_threshold(5)
            .backpressure_critical(10)
            .build();

        let pipeline = UpsertPipeline::new(config);

        for i in 0..10 {
            pipeline
                .submit(format!("v{}", i), vec![0.1; 4], None)
                .unwrap();
        }

        // Should fail at critical
        let result = pipeline.submit("overflow", vec![0.1; 4], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_batch_submit() {
        let config = StreamingUpsertConfig::default();
        let pipeline = UpsertPipeline::new(config);

        let batch = VectorBatchBuilder::new(4)
            .add("a", &[1.0; 4], None)
            .add("b", &[2.0; 4], None)
            .build();

        pipeline.submit_batch(batch).unwrap();
        assert_eq!(pipeline.buffered_count(), 2);
    }

    #[test]
    fn test_dedup_within_batch() {
        let records = vec![
            VectorRecord { id: "a".into(), vector: vec![1.0], metadata: None },
            VectorRecord { id: "b".into(), vector: vec![2.0], metadata: None },
            VectorRecord { id: "a".into(), vector: vec![3.0], metadata: None },
        ];
        let (deduped, dups) = UpsertPipeline::dedup_records(records);
        assert_eq!(deduped.len(), 2);
        assert_eq!(dups, 1);
        // Last occurrence of "a" wins
        assert_eq!(deduped[0].vector, vec![3.0]);
    }

    #[test]
    fn test_frame_batch_encode_decode() {
        let config = StreamingUpsertConfig::default();
        let pipeline = UpsertPipeline::new(config);

        let batch = VectorBatchBuilder::new(3)
            .add("x", &[1.0, 2.0, 3.0], Some(json!({"t": 1})))
            .build();

        let frame = pipeline.encode_batch_frame(&batch);
        assert_eq!(frame.frame_type, FrameType::VectorBatch);
        assert_eq!(frame.sequence_id, 0);

        let decoded = UpsertPipeline::decode_batch_frame(&frame).unwrap();
        assert_eq!(decoded.records.len(), 1);
        assert_eq!(decoded.records[0].id, "x");
    }

    #[test]
    fn test_arrow_vector_batch() {
        let arrow_batch = ArrowVectorBatch::from_raw_parts(
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec!["a".into(), "b".into()],
            vec![None, Some(json!({"k": "v"}))],
        )
        .unwrap();

        assert_eq!(arrow_batch.num_rows, 2);
        let vb = arrow_batch.to_vector_batch();
        assert_eq!(vb.records.len(), 2);
        assert_eq!(vb.records[0].vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(vb.records[1].vector, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_arrow_batch_dimension_mismatch() {
        let result = ArrowVectorBatch::from_raw_parts(
            3,
            vec![1.0, 2.0], // too few
            vec!["a".into()],
            vec![None],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = StreamingUpsertConfig::builder()
            .batch_size(500)
            .max_inflight(8)
            .dedup_within_batch(false)
            .flush_interval(Duration::from_secs(1))
            .build();

        assert_eq!(config.batch_size, 500);
        assert_eq!(config.max_inflight, 8);
        assert!(!config.dedup_within_batch);
        assert_eq!(config.flush_interval, Duration::from_secs(1));
    }

    #[test]
    fn test_pipeline_stats_reset() {
        let pipeline = UpsertPipeline::new(StreamingUpsertConfig::default());
        pipeline.submit("a", vec![1.0; 4], None).unwrap();
        assert_eq!(pipeline.stats().records_submitted, 1);
        pipeline.reset_stats();
        assert_eq!(pipeline.stats().records_submitted, 0);
    }

    #[test]
    fn test_empty_batch_from_bytes() {
        let batch = VectorBatchBuilder::new(2).build();
        let bytes = batch.to_bytes();
        let decoded = VectorBatch::from_bytes(&bytes).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_ingestion_engine_submit_and_drain() {
        let engine = IngestionEngine::new(IngestionEngineConfig::default());

        let batch = VectorBatchBuilder::new(4)
            .add("a", &[1.0; 4], None)
            .add("b", &[2.0; 4], None)
            .build();

        engine.submit(batch).unwrap();

        let metrics = engine.metrics();
        assert_eq!(metrics.total_vectors_ingested, 2);
        assert_eq!(metrics.pending_batches, 1);

        let records = engine.drain_all_records();
        assert_eq!(records.len(), 2);
        assert_eq!(engine.metrics().pending_batches, 0);
    }

    #[test]
    fn test_ingestion_engine_backpressure() {
        let config = IngestionEngineConfig {
            ring_buffer_capacity: 2,
            ..Default::default()
        };
        let engine = IngestionEngine::new(config);

        // Submit enough to hit critical
        for i in 0..3 {
            let batch = VectorBatchBuilder::new(2)
                .add(format!("v{i}"), &[1.0, 2.0], None)
                .build();
            let _ = engine.submit(batch);
        }

        assert_eq!(engine.pressure_level(), PressureLevel::Critical);
    }

    #[test]
    fn test_ingestion_engine_stop() {
        let engine = IngestionEngine::new(IngestionEngineConfig::default());
        assert!(engine.is_active());

        engine.stop();
        assert!(!engine.is_active());

        let batch = VectorBatchBuilder::new(2).build();
        assert!(engine.submit(batch).is_err());
    }

    #[test]
    fn test_throughput_tracker() {
        let tracker = ThroughputTracker::new(Duration::from_secs(10));
        assert_eq!(tracker.total_vectors(), 0);

        tracker.record(100, 400);
        tracker.record(200, 800);

        assert_eq!(tracker.total_vectors(), 300);
        assert_eq!(tracker.total_bytes(), 1200);
        assert!(tracker.vectors_per_second() >= 0.0);
    }

    #[test]
    fn test_ingestion_engine_global_dedup() {
        let config = IngestionEngineConfig {
            global_dedup: true,
            ..Default::default()
        };
        let engine = IngestionEngine::new(config);

        let batch1 = VectorBatchBuilder::new(2)
            .add("a", &[1.0, 2.0], None)
            .build();
        let batch2 = VectorBatchBuilder::new(2)
            .add("a", &[3.0, 4.0], None)
            .add("b", &[5.0, 6.0], None)
            .build();

        engine.submit(batch1).unwrap();
        engine.submit(batch2).unwrap();

        let records = engine.drain_all_records();
        assert_eq!(records.len(), 2); // "a" deduped
    }
}
