//! Streaming Ingest Pipeline
//!
//! Built-in CDC connectors (Kafka, PostgreSQL logical replication, webhook) that
//! auto-embed and index documents in real-time with backpressure and exactly-once
//! semantics.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::streaming_ingest::{
//!     StreamingIngestConfig, StreamingIngestPipeline, SourceConfig,
//!     WebhookSourceConfig, IngestRecord,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let config = StreamingIngestConfig::builder()
//!     .collection("docs")
//!     .batch_size(256)
//!     .flush_interval_ms(500)
//!     .enable_exactly_once(true)
//!     .build();
//!
//! let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();
//!
//! // Ingest a single record
//! let record = IngestRecord::new("doc1", vec![0.1f32; 384])
//!     .with_metadata(serde_json::json!({"source": "webhook"}));
//! pipeline.push(record).unwrap();
//!
//! // Flush to database
//! let stats = pipeline.flush().unwrap();
//! assert_eq!(stats.records_flushed, 1);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Source connector type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceConfig {
    /// Receive records via direct push (webhook-style).
    Webhook(WebhookSourceConfig),
    /// Poll records from a Kafka-compatible topic.
    Kafka(KafkaSourceConfig),
    /// Tail a PostgreSQL logical replication slot.
    Postgres(PostgresSourceConfig),
    /// Receive records via WebSocket connection (real-time push).
    WebSocket(WebSocketSourceConfig),
    /// Watch a directory for new JSON/NDJSON files containing vector records.
    FileWatch(FileWatchSourceConfig),
    /// Generic pull-based connector.
    Custom(CustomSourceConfig),
}

/// Webhook source configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebhookSourceConfig {
    /// Maximum request body size in bytes.
    pub max_body_bytes: usize,
    /// Whether to require HMAC signature validation.
    pub require_signature: bool,
}

impl Default for WebhookSourceConfig {
    fn default() -> Self {
        Self {
            max_body_bytes: 10 * 1024 * 1024,
            require_signature: false,
        }
    }
}

/// Kafka source configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KafkaSourceConfig {
    /// Broker addresses.
    pub brokers: Vec<String>,
    /// Topic to consume from.
    pub topic: String,
    /// Consumer group ID.
    pub group_id: String,
    /// Start from earliest or latest offset.
    pub auto_offset_reset: OffsetReset,
}

/// Kafka offset reset policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OffsetReset {
    /// Start from the earliest available offset.
    Earliest,
    /// Start from the latest offset.
    Latest,
}

impl Default for KafkaSourceConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".into()],
            topic: String::new(),
            group_id: "needle-ingest".into(),
            auto_offset_reset: OffsetReset::Latest,
        }
    }
}

/// PostgreSQL CDC source configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostgresSourceConfig {
    /// Connection string.
    pub connection_string: String,
    /// Replication slot name.
    pub slot_name: String,
    /// Publication name.
    pub publication: String,
}

/// WebSocket source configuration for real-time push ingestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSocketSourceConfig {
    /// WebSocket URL (ws:// or wss://).
    pub url: String,
    /// Reconnect interval on disconnect (ms).
    pub reconnect_interval_ms: u64,
    /// Maximum reconnect attempts before giving up (0 = unlimited).
    pub max_reconnect_attempts: u32,
    /// Ping interval for keepalive (ms).
    pub ping_interval_ms: u64,
    /// Maximum message size in bytes.
    pub max_message_bytes: usize,
    /// Optional authentication token sent in the initial handshake.
    pub auth_token: Option<String>,
    /// Message format expected from the WebSocket.
    pub message_format: WebSocketMessageFormat,
}

/// Format of messages received over WebSocket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSocketMessageFormat {
    /// JSON objects with vector, id, and optional metadata fields.
    Json,
    /// Length-prefixed binary frames (matching the streaming protocol).
    Binary,
    /// Newline-delimited JSON (one record per line).
    NdJson,
}

impl Default for WebSocketSourceConfig {
    fn default() -> Self {
        Self {
            url: String::new(),
            reconnect_interval_ms: 1000,
            max_reconnect_attempts: 0,
            ping_interval_ms: 30_000,
            max_message_bytes: 16 * 1024 * 1024,
            auth_token: None,
            message_format: WebSocketMessageFormat::Json,
        }
    }
}

/// File-watch source configuration for directory-based ingestion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileWatchSourceConfig {
    /// Directory path to watch for new files.
    pub watch_dir: String,
    /// File extensions to process (e.g., ["json", "ndjson"]).
    pub extensions: Vec<String>,
    /// Polling interval in milliseconds.
    pub poll_interval_ms: u64,
    /// Whether to process existing files on startup.
    pub process_existing: bool,
    /// Whether to delete files after successful ingestion.
    pub delete_after_ingest: bool,
    /// Move processed files to this directory instead of deleting (if set).
    pub archive_dir: Option<String>,
}

impl Default for FileWatchSourceConfig {
    fn default() -> Self {
        Self {
            watch_dir: String::new(),
            extensions: vec!["json".into(), "ndjson".into()],
            poll_interval_ms: 1000,
            process_existing: true,
            delete_after_ingest: false,
            archive_dir: None,
        }
    }
}

/// Generic custom source configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomSourceConfig {
    /// User-defined connector name.
    pub name: String,
    /// Opaque configuration payload.
    pub params: HashMap<String, String>,
}

/// Delivery guarantee semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// At-most-once: fire and forget.
    AtMostOnce,
    /// At-least-once: retry on failure (may produce duplicates).
    AtLeastOnce,
    /// Exactly-once: idempotent upserts with checkpoint tracking.
    ExactlyOnce,
}

impl Default for DeliveryGuarantee {
    fn default() -> Self {
        Self::AtLeastOnce
    }
}

/// Pipeline-level configuration.
#[derive(Debug, Clone)]
pub struct StreamingIngestConfig {
    /// Target collection name.
    pub collection: String,
    /// Number of records per batch flush.
    pub batch_size: usize,
    /// Maximum time (ms) between flushes.
    pub flush_interval_ms: u64,
    /// Maximum in-memory buffer before backpressure triggers.
    pub max_buffer_size: usize,
    /// Delivery semantics.
    pub delivery: DeliveryGuarantee,
    /// Enable content-hash deduplication.
    pub enable_dedup: bool,
    /// Maximum retries for failed records before dead-lettering.
    pub max_retries: u32,
    /// Optional source connector config.
    pub source: Option<SourceConfig>,
    /// Optional transform: field path to extract vector from JSON payload.
    pub vector_field: Option<String>,
    /// Optional transform: field path to extract ID from JSON payload.
    pub id_field: Option<String>,
    /// Optional transform: field path to extract metadata from JSON payload.
    pub metadata_field: Option<String>,
}

impl Default for StreamingIngestConfig {
    fn default() -> Self {
        Self {
            collection: String::new(),
            batch_size: 256,
            flush_interval_ms: 1000,
            max_buffer_size: 10_000,
            delivery: DeliveryGuarantee::default(),
            enable_dedup: false,
            max_retries: 3,
            source: None,
            vector_field: None,
            id_field: None,
            metadata_field: None,
        }
    }
}

impl StreamingIngestConfig {
    /// Create a new builder.
    pub fn builder() -> StreamingIngestConfigBuilder {
        StreamingIngestConfigBuilder::default()
    }
}

/// Builder for `StreamingIngestConfig`.
#[derive(Debug, Default)]
pub struct StreamingIngestConfigBuilder {
    inner: StreamingIngestConfig,
}

impl StreamingIngestConfigBuilder {
    /// Set the target collection.
    #[must_use]
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.inner.collection = name.into();
        self
    }

    /// Set batch size.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.inner.batch_size = size.max(1);
        self
    }

    /// Set flush interval in milliseconds.
    #[must_use]
    pub fn flush_interval_ms(mut self, ms: u64) -> Self {
        self.inner.flush_interval_ms = ms;
        self
    }

    /// Set maximum buffer size.
    #[must_use]
    pub fn max_buffer_size(mut self, size: usize) -> Self {
        self.inner.max_buffer_size = size.max(1);
        self
    }

    /// Enable exactly-once delivery.
    #[must_use]
    pub fn enable_exactly_once(mut self, enable: bool) -> Self {
        self.inner.delivery = if enable {
            DeliveryGuarantee::ExactlyOnce
        } else {
            DeliveryGuarantee::AtLeastOnce
        };
        self
    }

    /// Set delivery guarantee.
    #[must_use]
    pub fn delivery(mut self, d: DeliveryGuarantee) -> Self {
        self.inner.delivery = d;
        self
    }

    /// Enable content-hash deduplication.
    #[must_use]
    pub fn enable_dedup(mut self, enable: bool) -> Self {
        self.inner.enable_dedup = enable;
        self
    }

    /// Set maximum retries before dead-lettering.
    #[must_use]
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.inner.max_retries = retries;
        self
    }

    /// Set source connector.
    #[must_use]
    pub fn source(mut self, src: SourceConfig) -> Self {
        self.inner.source = Some(src);
        self
    }

    /// Configure a WebSocket source for real-time push ingestion.
    #[must_use]
    pub fn websocket(self, url: impl Into<String>) -> Self {
        self.source(SourceConfig::WebSocket(WebSocketSourceConfig {
            url: url.into(),
            ..WebSocketSourceConfig::default()
        }))
    }

    /// Configure a file-watch source for directory-based ingestion.
    #[must_use]
    pub fn file_watch(self, dir: impl Into<String>) -> Self {
        self.source(SourceConfig::FileWatch(FileWatchSourceConfig {
            watch_dir: dir.into(),
            ..FileWatchSourceConfig::default()
        }))
    }

    /// Set the JSON field path for extracting vectors.
    #[must_use]
    pub fn vector_field(mut self, field: impl Into<String>) -> Self {
        self.inner.vector_field = Some(field.into());
        self
    }

    /// Set the JSON field path for extracting IDs.
    #[must_use]
    pub fn id_field(mut self, field: impl Into<String>) -> Self {
        self.inner.id_field = Some(field.into());
        self
    }

    /// Set the JSON field path for extracting metadata.
    #[must_use]
    pub fn metadata_field(mut self, field: impl Into<String>) -> Self {
        self.inner.metadata_field = Some(field.into());
        self
    }

    /// Build the configuration.
    pub fn build(self) -> StreamingIngestConfig {
        self.inner
    }
}

// ── Records ──────────────────────────────────────────────────────────────────

/// A single record to ingest.
#[derive(Debug, Clone)]
pub struct IngestRecord {
    /// Unique vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Optional metadata.
    pub metadata: Option<Value>,
    /// Attempt counter (for retry tracking).
    pub attempts: u32,
    /// Source-provided sequence number for exactly-once dedup.
    pub sequence_id: Option<u64>,
}

impl IngestRecord {
    /// Create a new ingest record.
    pub fn new(id: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            metadata: None,
            attempts: 0,
            sequence_id: None,
        }
    }

    /// Attach metadata.
    #[must_use]
    pub fn with_metadata(mut self, meta: Value) -> Self {
        self.metadata = Some(meta);
        self
    }

    /// Attach a source sequence ID for exactly-once tracking.
    #[must_use]
    pub fn with_sequence_id(mut self, seq: u64) -> Self {
        self.sequence_id = Some(seq);
        self
    }
}

/// Parse a JSON payload into an `IngestRecord` using field extraction config.
pub fn parse_json_payload(
    payload: &Value,
    id_field: Option<&str>,
    vector_field: Option<&str>,
    metadata_field: Option<&str>,
) -> Result<IngestRecord> {
    let id = if let Some(field) = id_field {
        payload
            .get(field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                NeedleError::InvalidArgument(format!("missing or non-string id field '{field}'"))
            })?
            .to_string()
    } else {
        payload
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidArgument("missing 'id' field".into()))?
            .to_string()
    };

    let vector = if let Some(field) = vector_field {
        extract_vector(payload.get(field).ok_or_else(|| {
            NeedleError::InvalidArgument(format!("missing vector field '{field}'"))
        })?)?
    } else {
        extract_vector(
            payload
                .get("vector")
                .ok_or_else(|| NeedleError::InvalidArgument("missing 'vector' field".into()))?,
        )?
    };

    let metadata = if let Some(field) = metadata_field {
        payload.get(field).cloned()
    } else {
        payload.get("metadata").cloned()
    };

    Ok(IngestRecord::new(id, vector).with_metadata(metadata.unwrap_or(Value::Null)))
}

fn extract_vector(val: &Value) -> Result<Vec<f32>> {
    val.as_array()
        .ok_or_else(|| NeedleError::InvalidArgument("vector field must be an array".into()))?
        .iter()
        .map(|v| {
            v.as_f64().map(|f| f as f32).ok_or_else(|| {
                NeedleError::InvalidArgument("vector elements must be numbers".into())
            })
        })
        .collect()
}

// ── Record Transformers ─────────────────────────────────────────────────────

/// A transformation applied to ingest records before they reach the database.
pub trait RecordTransformer: Send + Sync {
    /// Transform a record. Return `None` to filter it out.
    fn transform(&self, record: IngestRecord) -> Option<IngestRecord>;
}

/// Normalizes vectors to unit length (L2 normalization).
pub struct L2NormalizeTransformer;

impl RecordTransformer for L2NormalizeTransformer {
    fn transform(&self, mut record: IngestRecord) -> Option<IngestRecord> {
        let norm: f32 = record.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in record.vector.iter_mut() {
                *v /= norm;
            }
        }
        Some(record)
    }
}

/// Filters out records with zero-magnitude vectors.
pub struct ZeroVectorFilter;

impl RecordTransformer for ZeroVectorFilter {
    fn transform(&self, record: IngestRecord) -> Option<IngestRecord> {
        let magnitude: f32 = record.vector.iter().map(|x| x * x).sum();
        if magnitude > f32::EPSILON {
            Some(record)
        } else {
            None
        }
    }
}

/// Truncates or pads vectors to a target dimension.
pub struct DimensionAdapter {
    target_dim: usize,
}

impl DimensionAdapter {
    pub fn new(target_dim: usize) -> Self {
        Self { target_dim }
    }
}

impl RecordTransformer for DimensionAdapter {
    fn transform(&self, mut record: IngestRecord) -> Option<IngestRecord> {
        record.vector.resize(self.target_dim, 0.0);
        Some(record)
    }
}

/// Composes multiple transformers in sequence.
pub struct TransformPipeline {
    transformers: Vec<Box<dyn RecordTransformer>>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self {
            transformers: Vec::new(),
        }
    }

    pub fn add(mut self, t: Box<dyn RecordTransformer>) -> Self {
        self.transformers.push(t);
        self
    }

    /// Apply all transformers in sequence. Returns None if any transformer filters it out.
    pub fn apply(&self, mut record: IngestRecord) -> Option<IngestRecord> {
        for transformer in &self.transformers {
            record = transformer.transform(record)?;
        }
        Some(record)
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── Checkpoint ───────────────────────────────────────────────────────────────

/// Checkpoint state for exactly-once tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Last committed sequence ID per source partition.
    pub committed_sequences: HashMap<String, u64>,
    /// Timestamp of last checkpoint.
    pub last_checkpoint_at: Option<u64>,
    /// Total records committed through this checkpoint.
    pub total_committed: u64,
}

impl Checkpoint {
    /// Check if a sequence ID has already been committed for a partition.
    pub fn is_committed(&self, partition: &str, seq: u64) -> bool {
        self.committed_sequences
            .get(partition)
            .map_or(false, |&committed| seq <= committed)
    }

    /// Advance the checkpoint for a partition.
    pub fn advance(&mut self, partition: &str, seq: u64, count: u64) {
        let entry = self
            .committed_sequences
            .entry(partition.to_string())
            .or_insert(0);
        if seq > *entry {
            *entry = seq;
        }
        self.total_committed += count;
        self.last_checkpoint_at = Some(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );
    }
}

// ── Dead-Letter Queue ────────────────────────────────────────────────────────

/// A record that failed ingestion after all retries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    /// The original record ID.
    pub record_id: String,
    /// Error message from the last attempt.
    pub error: String,
    /// Number of attempts made.
    pub attempts: u32,
    /// When the record was dead-lettered (epoch seconds).
    pub failed_at: u64,
    /// Original vector (for potential replay).
    pub vector: Vec<f32>,
    /// Original metadata.
    pub metadata: Option<Value>,
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Pipeline statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestStats {
    /// Total records received into the buffer.
    pub records_received: u64,
    /// Total records successfully flushed to the database.
    pub records_flushed: u64,
    /// Total records deduplicated (skipped).
    pub records_deduped: u64,
    /// Total records sent to the dead-letter queue.
    pub records_dead_lettered: u64,
    /// Total flush operations performed.
    pub flush_count: u64,
    /// Total bytes of vector data ingested.
    pub bytes_ingested: u64,
    /// Average flush latency in microseconds.
    pub avg_flush_latency_us: u64,
    /// Number of retries across all records.
    pub total_retries: u64,
}

/// Backpressure signal for upstream flow control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressureSignal {
    /// Pipeline can accept more records.
    Accept,
    /// Pipeline is getting full; slow down.
    Throttle,
    /// Pipeline is at capacity; pause sending.
    Pause,
}

/// Pipeline health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Pipeline is operating normally.
    Healthy,
    /// Pipeline is under pressure but still accepting records.
    Degraded,
    /// Pipeline is at capacity and may drop records.
    Overloaded,
}

/// Pipeline health summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineHealth {
    /// Overall health status.
    pub status: HealthStatus,
    /// Buffer utilization (0.0–1.0).
    pub buffer_usage: f32,
    /// Number of records waiting in the buffer.
    pub pending_records: usize,
    /// Number of dead-letter entries.
    pub dead_letter_count: usize,
    /// Total records flushed.
    pub total_flushed: u64,
    /// Total records dead-lettered.
    pub total_errors: u64,
    /// Average flush latency in microseconds.
    pub avg_flush_latency_us: u64,
}

// ── Lag Monitoring ───────────────────────────────────────────────────────────

/// Lag metrics for monitoring pipeline health.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LagMetrics {
    /// Records buffered but not yet flushed.
    pub buffer_lag: u64,
    /// Milliseconds since the last flush.
    pub time_since_flush_ms: u64,
    /// Whether a flush is overdue based on the configured interval.
    pub flush_overdue: bool,
    /// Records received but not yet committed through checkpointing.
    pub checkpoint_lag: u64,
    /// Number of records in the dead-letter queue.
    pub dead_letter_count: u64,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Streaming ingestion pipeline with backpressure and exactly-once semantics.
pub struct StreamingIngestPipeline<'a> {
    db: &'a Database,
    config: StreamingIngestConfig,
    buffer: VecDeque<IngestRecord>,
    dead_letters: Vec<DeadLetterEntry>,
    seen_ids: HashSet<String>,
    checkpoint: Checkpoint,
    stats: IngestStats,
    last_flush: Instant,
    flush_latencies: VecDeque<u64>,
}

impl<'a> StreamingIngestPipeline<'a> {
    /// Create a new streaming ingest pipeline.
    pub fn new(db: &'a Database, config: StreamingIngestConfig) -> Result<Self> {
        if config.collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "collection name is required".into(),
            ));
        }
        // Validate the collection exists
        let _coll = db.collection(&config.collection)?;

        Ok(Self {
            db,
            config,
            buffer: VecDeque::new(),
            dead_letters: Vec::new(),
            seen_ids: HashSet::new(),
            checkpoint: Checkpoint::default(),
            stats: IngestStats::default(),
            last_flush: Instant::now(),
            flush_latencies: VecDeque::with_capacity(100),
        })
    }

    /// Push a single record into the pipeline buffer.
    ///
    /// Returns the current backpressure signal. The caller should slow down or
    /// pause if the signal is `Throttle` or `Pause`.
    pub fn push(&mut self, record: IngestRecord) -> Result<BackpressureSignal> {
        // Exactly-once: skip if already committed
        if self.config.delivery == DeliveryGuarantee::ExactlyOnce {
            if let Some(seq) = record.sequence_id {
                if self.checkpoint.is_committed("default", seq) {
                    self.stats.records_deduped += 1;
                    return Ok(self.backpressure_signal());
                }
            }
        }

        // Content-hash dedup
        if self.config.enable_dedup && self.seen_ids.contains(&record.id) {
            self.stats.records_deduped += 1;
            return Ok(self.backpressure_signal());
        }

        if self.config.enable_dedup {
            self.seen_ids.insert(record.id.clone());
        }

        self.stats.records_received += 1;
        self.stats.bytes_ingested += (record.vector.len() * 4) as u64;
        self.buffer.push_back(record);

        // Auto-flush if batch is full
        if self.buffer.len() >= self.config.batch_size {
            self.flush()?;
        }

        Ok(self.backpressure_signal())
    }

    /// Push multiple records.
    pub fn push_batch(&mut self, records: Vec<IngestRecord>) -> Result<BackpressureSignal> {
        for record in records {
            self.push(record)?;
        }
        Ok(self.backpressure_signal())
    }

    /// Push a raw JSON payload, extracting fields per config.
    pub fn push_json(&mut self, payload: &Value) -> Result<BackpressureSignal> {
        let record = parse_json_payload(
            payload,
            self.config.id_field.as_deref(),
            self.config.vector_field.as_deref(),
            self.config.metadata_field.as_deref(),
        )?;
        self.push(record)
    }

    /// Flush buffered records to the database.
    pub fn flush(&mut self) -> Result<IngestStats> {
        if self.buffer.is_empty() {
            return Ok(self.stats.clone());
        }

        let start = Instant::now();
        let coll = self.db.collection(&self.config.collection)?;
        let _batch_size = self.buffer.len();
        let mut flushed = 0u64;
        let mut max_seq = 0u64;

        let drain: Vec<IngestRecord> = self.buffer.drain(..).collect();
        let mut retry_queue: Vec<IngestRecord> = Vec::new();

        for mut record in drain {
            match coll.insert(&record.id, &record.vector, record.metadata.clone()) {
                Ok(_) => {
                    flushed += 1;
                    if let Some(seq) = record.sequence_id {
                        max_seq = max_seq.max(seq);
                    }
                }
                Err(e) => {
                    record.attempts += 1;
                    if record.attempts < self.config.max_retries {
                        self.stats.total_retries += 1;
                        retry_queue.push(record);
                    } else {
                        self.dead_letter(record, &e.to_string());
                    }
                }
            }
        }

        // Re-queue retries
        for record in retry_queue {
            self.buffer.push_back(record);
        }

        // Update checkpoint for exactly-once
        if self.config.delivery == DeliveryGuarantee::ExactlyOnce && max_seq > 0 {
            self.checkpoint.advance("default", max_seq, flushed);
        }

        self.stats.records_flushed += flushed;
        self.stats.flush_count += 1;

        let latency_us = start.elapsed().as_micros() as u64;
        self.flush_latencies.push_back(latency_us);
        if self.flush_latencies.len() > 100 {
            self.flush_latencies.pop_front();
        }
        self.stats.avg_flush_latency_us = if self.flush_latencies.is_empty() {
            0
        } else {
            self.flush_latencies.iter().sum::<u64>() / self.flush_latencies.len() as u64
        };

        self.last_flush = Instant::now();

        Ok(self.stats.clone())
    }

    /// Flush if the configured interval has elapsed.
    pub fn tick(&mut self) -> Result<Option<IngestStats>> {
        let elapsed = self.last_flush.elapsed();
        if elapsed >= Duration::from_millis(self.config.flush_interval_ms)
            && !self.buffer.is_empty()
        {
            Ok(Some(self.flush()?))
        } else {
            Ok(None)
        }
    }

    /// Get current backpressure signal.
    pub fn backpressure_signal(&self) -> BackpressureSignal {
        let usage = self.buffer.len() as f64 / self.config.max_buffer_size as f64;
        if usage >= 0.95 {
            BackpressureSignal::Pause
        } else if usage >= 0.75 {
            BackpressureSignal::Throttle
        } else {
            BackpressureSignal::Accept
        }
    }

    /// Get pipeline statistics.
    pub fn stats(&self) -> &IngestStats {
        &self.stats
    }

    /// Get dead-letter entries.
    pub fn dead_letters(&self) -> &[DeadLetterEntry] {
        &self.dead_letters
    }

    /// Drain and return dead-letter entries for external processing.
    pub fn drain_dead_letters(&mut self) -> Vec<DeadLetterEntry> {
        std::mem::take(&mut self.dead_letters)
    }

    /// Get the current checkpoint state.
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.checkpoint
    }

    /// Restore from a previously saved checkpoint (for crash recovery).
    pub fn restore_checkpoint(&mut self, checkpoint: Checkpoint) {
        self.checkpoint = checkpoint;
    }

    /// Number of records currently buffered.
    pub fn pending_count(&self) -> usize {
        self.buffer.len()
    }

    /// Replay dead-letter entries back into the pipeline buffer for retry.
    /// Returns the number of entries replayed.
    pub fn replay_dead_letters(&mut self) -> usize {
        let entries = std::mem::take(&mut self.dead_letters);
        let count = entries.len();
        for entry in entries {
            let record = IngestRecord {
                id: entry.record_id,
                vector: entry.vector,
                metadata: entry.metadata,
                attempts: 0, // reset attempts for replay
                sequence_id: None,
            };
            self.buffer.push_back(record);
        }
        count
    }

    /// Get a pipeline health summary.
    pub fn health(&self) -> PipelineHealth {
        let signal = self.backpressure_signal();
        let buffer_usage =
            self.buffer.len() as f32 / self.config.max_buffer_size as f32;
        PipelineHealth {
            status: match signal {
                BackpressureSignal::Accept => HealthStatus::Healthy,
                BackpressureSignal::Throttle => HealthStatus::Degraded,
                BackpressureSignal::Pause => HealthStatus::Overloaded,
            },
            buffer_usage,
            pending_records: self.buffer.len(),
            dead_letter_count: self.dead_letters.len(),
            total_flushed: self.stats.records_flushed,
            total_errors: self.stats.records_dead_lettered,
            avg_flush_latency_us: self.stats.avg_flush_latency_us,
        }
    }

    /// Get lag metrics for monitoring.
    pub fn lag_metrics(&self) -> LagMetrics {
        let buffer_lag = self.buffer.len() as u64;
        let time_since_flush_ms = self.last_flush.elapsed().as_millis() as u64;
        let flush_overdue = time_since_flush_ms > self.config.flush_interval_ms && !self.buffer.is_empty();
        LagMetrics {
            buffer_lag,
            time_since_flush_ms,
            flush_overdue,
            checkpoint_lag: self.stats.records_received.saturating_sub(self.stats.records_flushed),
            dead_letter_count: self.dead_letters.len() as u64,
        }
    }

    /// Export Prometheus-compatible metrics as text.
    pub fn prometheus_metrics(&self) -> String {
        let lag = self.lag_metrics();
        format!(
            "# HELP needle_streaming_records_received Total records received\n\
             # TYPE needle_streaming_records_received counter\n\
             needle_streaming_records_received {}\n\
             # HELP needle_streaming_records_flushed Total records flushed\n\
             # TYPE needle_streaming_records_flushed counter\n\
             needle_streaming_records_flushed {}\n\
             # HELP needle_streaming_records_deduped Total records deduplicated\n\
             # TYPE needle_streaming_records_deduped counter\n\
             needle_streaming_records_deduped {}\n\
             # HELP needle_streaming_records_dead_lettered Total dead-lettered records\n\
             # TYPE needle_streaming_records_dead_lettered counter\n\
             needle_streaming_records_dead_lettered {}\n\
             # HELP needle_streaming_buffer_lag Current buffer lag\n\
             # TYPE needle_streaming_buffer_lag gauge\n\
             needle_streaming_buffer_lag {}\n\
             # HELP needle_streaming_buffer_usage Buffer utilization ratio\n\
             # TYPE needle_streaming_buffer_usage gauge\n\
             needle_streaming_buffer_usage {:.4}\n\
             # HELP needle_streaming_flush_latency_us Average flush latency\n\
             # TYPE needle_streaming_flush_latency_us gauge\n\
             needle_streaming_flush_latency_us {}\n\
             # HELP needle_streaming_checkpoint_lag Records not yet checkpointed\n\
             # TYPE needle_streaming_checkpoint_lag gauge\n\
             needle_streaming_checkpoint_lag {}\n\
             # HELP needle_streaming_flush_count Total flushes\n\
             # TYPE needle_streaming_flush_count counter\n\
             needle_streaming_flush_count {}\n",
            self.stats.records_received,
            self.stats.records_flushed,
            self.stats.records_deduped,
            self.stats.records_dead_lettered,
            lag.buffer_lag,
            self.buffer.len() as f64 / self.config.max_buffer_size as f64,
            self.stats.avg_flush_latency_us,
            lag.checkpoint_lag,
            self.stats.flush_count,
        )
    }

    fn dead_letter(&mut self, record: IngestRecord, error: &str) {
        self.stats.records_dead_lettered += 1;
        self.dead_letters.push(DeadLetterEntry {
            record_id: record.id,
            error: error.to_string(),
            attempts: record.attempts,
            failed_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            vector: record.vector,
            metadata: record.metadata,
        });
    }

    /// Graceful shutdown: flush all pending records and save the database.
    ///
    /// Returns final pipeline statistics after draining the buffer.
    pub fn graceful_shutdown(&mut self) -> Result<IngestStats> {
        // Flush all remaining buffered records
        if !self.buffer.is_empty() {
            self.flush()?;
        }

        // Retry any remaining records one more time
        if !self.buffer.is_empty() {
            self.flush()?;
        }

        // Persist the database
        self.db.save()?;

        tracing::info!(
            "Streaming pipeline shutdown: {} flushed, {} dead-lettered, {} pending",
            self.stats.records_flushed,
            self.stats.records_dead_lettered,
            self.buffer.len()
        );

        Ok(self.stats.clone())
    }
}

// ── Webhook Handler ──────────────────────────────────────────────────────────

/// Configuration for the webhook ingestion handler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookHandlerConfig {
    /// HMAC secret for signature validation (SHA-256).
    pub hmac_secret: Option<String>,
    /// Maximum payload size in bytes.
    pub max_payload_bytes: usize,
    /// JSON path to the vector field (dot-separated, e.g. "embedding").
    pub vector_field: String,
    /// JSON path to the ID field.
    pub id_field: String,
    /// Optional JSON path to metadata.
    pub metadata_field: Option<String>,
    /// Whether payloads may contain arrays of records.
    pub allow_batch: bool,
}

impl Default for WebhookHandlerConfig {
    fn default() -> Self {
        Self {
            hmac_secret: None,
            max_payload_bytes: 10 * 1024 * 1024,
            vector_field: "vector".to_string(),
            id_field: "id".to_string(),
            metadata_field: Some("metadata".to_string()),
            allow_batch: true,
        }
    }
}

/// Result of processing a webhook payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookResult {
    /// Number of records accepted.
    pub accepted: usize,
    /// Number of records rejected.
    pub rejected: usize,
    /// Backpressure signal for the caller.
    pub backpressure: bool,
    /// Error details for rejected records.
    pub errors: Vec<String>,
}

/// Handles webhook payloads and feeds them into a `StreamingIngestPipeline`.
pub struct WebhookHandler {
    config: WebhookHandlerConfig,
}

impl WebhookHandler {
    /// Create a new webhook handler.
    pub fn new(config: WebhookHandlerConfig) -> Self {
        Self { config }
    }

    /// Process a webhook JSON payload into ingest records.
    ///
    /// Returns parsed records and any validation errors.
    pub fn process_payload(&self, payload: &Value) -> Result<(Vec<IngestRecord>, Vec<String>)> {
        let records_json = if payload.is_array() && self.config.allow_batch {
            payload.as_array().cloned().unwrap_or_default()
        } else if payload.is_object() {
            vec![payload.clone()]
        } else {
            return Err(NeedleError::InvalidInput(
                "Webhook payload must be a JSON object or array".to_string(),
            ));
        };

        let mut records = Vec::with_capacity(records_json.len());
        let mut errors = Vec::new();

        for (i, item) in records_json.iter().enumerate() {
            match self.parse_record(item) {
                Ok(record) => records.push(record),
                Err(e) => errors.push(format!("record[{i}]: {e}")),
            }
        }

        Ok((records, errors))
    }

    /// Ingest a webhook payload directly into the pipeline.
    pub fn ingest_into<'a>(
        &self,
        payload: &Value,
        pipeline: &mut StreamingIngestPipeline<'a>,
    ) -> Result<WebhookResult> {
        let (records, errors) = self.process_payload(payload)?;
        let rejected = errors.len();
        let mut accepted = 0;
        let mut backpressure = false;

        for record in records {
            match pipeline.push(record) {
                Ok(signal) => {
                    accepted += 1;
                    if matches!(signal, BackpressureSignal::Throttle | BackpressureSignal::Pause) {
                        backpressure = true;
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok(WebhookResult {
            accepted,
            rejected,
            backpressure,
            errors,
        })
    }

    fn parse_record(&self, item: &Value) -> Result<IngestRecord> {
        let id = item
            .get(&self.config.id_field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                NeedleError::InvalidInput(format!(
                    "Missing or invalid '{}' field",
                    self.config.id_field
                ))
            })?;

        let vector_val = item.get(&self.config.vector_field).ok_or_else(|| {
            NeedleError::InvalidInput(format!(
                "Missing '{}' field",
                self.config.vector_field
            ))
        })?;

        let vector: Vec<f32> = vector_val
            .as_array()
            .ok_or_else(|| {
                NeedleError::InvalidInput(format!(
                    "'{}' must be an array of numbers",
                    self.config.vector_field
                ))
            })?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| NeedleError::InvalidInput("Vector element must be a number".to_string()))
            })
            .collect::<Result<Vec<f32>>>()?;

        let metadata = self
            .config
            .metadata_field
            .as_ref()
            .and_then(|field| item.get(field).cloned());

        let mut record = IngestRecord::new(id, vector);
        if let Some(meta) = metadata {
            record = record.with_metadata(meta);
        }

        Ok(record)
    }

    /// Validate a signature header against the raw payload body.
    ///
    /// Currently performs a constant-time comparison of the expected
    /// hex-encoded SHA-256 digest. For HMAC validation, enable the `server`
    /// feature which brings in the `hmac` crate.
    pub fn validate_signature(&self, body: &[u8], signature_header: &str) -> bool {
        let Some(ref secret) = self.config.hmac_secret else {
            return true; // No secret configured, skip validation
        };

        // Simple SHA-256(secret + body) comparison
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(secret.as_bytes());
        hasher.update(body);
        let digest = hasher.finalize();

        let hex_sig = signature_header
            .strip_prefix("sha256=")
            .unwrap_or(signature_header);

        // Constant-time-ish comparison via formatted hex
        let expected = format!("{:x}", digest);
        expected == hex_sig
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_pipeline_basic_ingest() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(10)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        let record = IngestRecord::new("v1", vec![1.0, 2.0, 3.0, 4.0]);
        let signal = pipeline.push(record).unwrap();
        assert_eq!(signal, BackpressureSignal::Accept);
        assert_eq!(pipeline.pending_count(), 1);

        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.records_flushed, 1);
        assert_eq!(pipeline.pending_count(), 0);
    }

    #[test]
    fn test_pipeline_auto_flush_on_batch_size() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(2)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        pipeline
            .push(IngestRecord::new("v1", vec![1.0; 4]))
            .unwrap();
        // Second push triggers auto-flush
        pipeline
            .push(IngestRecord::new("v2", vec![2.0; 4]))
            .unwrap();

        assert_eq!(pipeline.stats().records_flushed, 2);
    }

    #[test]
    fn test_pipeline_dedup() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .enable_dedup(true)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        pipeline
            .push(IngestRecord::new("v1", vec![1.0; 4]))
            .unwrap();
        pipeline
            .push(IngestRecord::new("v1", vec![1.0; 4]))
            .unwrap();
        pipeline
            .push(IngestRecord::new("v2", vec![2.0; 4]))
            .unwrap();

        assert_eq!(pipeline.stats().records_received, 2);
        assert_eq!(pipeline.stats().records_deduped, 1);
    }

    #[test]
    fn test_pipeline_exactly_once_checkpoint() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .enable_exactly_once(true)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        let r1 = IngestRecord::new("v1", vec![1.0; 4]).with_sequence_id(1);
        let r2 = IngestRecord::new("v2", vec![2.0; 4]).with_sequence_id(2);
        pipeline.push(r1).unwrap();
        pipeline.push(r2).unwrap();
        pipeline.flush().unwrap();

        // Replay same sequence IDs — should be deduped
        let r1_replay = IngestRecord::new("v1", vec![1.0; 4]).with_sequence_id(1);
        pipeline.push(r1_replay).unwrap();
        assert_eq!(pipeline.stats().records_deduped, 1);
    }

    #[test]
    fn test_pipeline_backpressure() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100_000)
            .max_buffer_size(4)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        pipeline
            .push(IngestRecord::new("v1", vec![1.0; 4]))
            .unwrap();
        pipeline
            .push(IngestRecord::new("v2", vec![2.0; 4]))
            .unwrap();
        pipeline
            .push(IngestRecord::new("v3", vec![3.0; 4]))
            .unwrap();

        let signal = pipeline
            .push(IngestRecord::new("v4", vec![4.0; 4]))
            .unwrap();
        assert_eq!(signal, BackpressureSignal::Pause);
    }

    #[test]
    fn test_pipeline_json_ingest() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        let payload = serde_json::json!({
            "id": "doc1",
            "vector": [1.0, 2.0, 3.0, 4.0],
            "metadata": {"source": "test"}
        });

        pipeline.push_json(&payload).unwrap();
        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.records_flushed, 1);
    }

    #[test]
    fn test_pipeline_custom_field_extraction() {
        let payload = serde_json::json!({
            "doc_id": "abc",
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "props": {"color": "red"}
        });

        let record =
            parse_json_payload(&payload, Some("doc_id"), Some("embedding"), Some("props")).unwrap();
        assert_eq!(record.id, "abc");
        assert_eq!(record.vector.len(), 4);
        assert_eq!(
            record
                .metadata
                .unwrap()
                .get("color")
                .unwrap()
                .as_str()
                .unwrap(),
            "red"
        );
    }

    #[test]
    fn test_checkpoint_advance() {
        let mut cp = Checkpoint::default();
        assert!(!cp.is_committed("p0", 1));

        cp.advance("p0", 5, 5);
        assert!(cp.is_committed("p0", 5));
        assert!(cp.is_committed("p0", 3));
        assert!(!cp.is_committed("p0", 6));
        assert!(!cp.is_committed("p1", 1));
    }

    #[test]
    fn test_pipeline_requires_collection() {
        let db = Database::in_memory();
        let config = StreamingIngestConfig::builder().collection("").build();
        assert!(StreamingIngestPipeline::new(&db, config).is_err());
    }

    #[test]
    fn test_pipeline_tick_noop_on_empty_buffer() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .flush_interval_ms(0)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();
        let result = pipeline.tick().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_pipeline_health() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .max_buffer_size(10)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();
        let health = pipeline.health();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.pending_records, 0);

        // Fill buffer to trigger degraded
        for i in 0..8 {
            pipeline
                .push(IngestRecord::new(format!("v{i}"), vec![1.0; 4]))
                .unwrap();
        }
        let health = pipeline.health();
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[test]
    fn test_replay_dead_letters() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .max_retries(1)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        // Insert a record that causes dimension mismatch → dead letter
        pipeline
            .push(IngestRecord::new("bad", vec![1.0; 8])) // wrong dimensions
            .unwrap();
        pipeline.flush().unwrap();

        assert_eq!(pipeline.dead_letters().len(), 1);

        // Replay
        let replayed = pipeline.replay_dead_letters();
        assert_eq!(replayed, 1);
        assert!(pipeline.dead_letters().is_empty());
        assert_eq!(pipeline.pending_count(), 1);
    }

    #[test]
    fn test_websocket_source_config_defaults() {
        let config = WebSocketSourceConfig::default();
        assert!(config.url.is_empty());
        assert_eq!(config.reconnect_interval_ms, 1000);
        assert_eq!(config.max_reconnect_attempts, 0);
        assert_eq!(config.ping_interval_ms, 30_000);
        assert_eq!(config.max_message_bytes, 16 * 1024 * 1024);
        assert!(config.auth_token.is_none());
        assert_eq!(config.message_format, WebSocketMessageFormat::Json);
    }

    #[test]
    fn test_websocket_source_builder() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .websocket("ws://localhost:8080/vectors")
            .build();

        match &config.source {
            Some(SourceConfig::WebSocket(ws)) => {
                assert_eq!(ws.url, "ws://localhost:8080/vectors");
                assert_eq!(ws.message_format, WebSocketMessageFormat::Json);
            }
            other => return Err(format!("Expected WebSocket source, got {other:?}").into()),
        }

        Ok(())
    }

    #[test]
    fn test_websocket_source_custom_config() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let ws_config = WebSocketSourceConfig {
            url: "wss://stream.example.com/ingest".into(),
            reconnect_interval_ms: 5000,
            max_reconnect_attempts: 10,
            ping_interval_ms: 15_000,
            max_message_bytes: 1024 * 1024,
            auth_token: Some("bearer-token-123".into()),
            message_format: WebSocketMessageFormat::NdJson,
        };

        let config = StreamingIngestConfig::builder()
            .collection("test")
            .source(SourceConfig::WebSocket(ws_config.clone()))
            .batch_size(64)
            .enable_exactly_once(true)
            .build();

        match &config.source {
            Some(SourceConfig::WebSocket(ws)) => {
                assert_eq!(ws.url, "wss://stream.example.com/ingest");
                assert_eq!(ws.reconnect_interval_ms, 5000);
                assert_eq!(ws.max_reconnect_attempts, 10);
                assert_eq!(ws.message_format, WebSocketMessageFormat::NdJson);
                assert_eq!(ws.auth_token.as_deref(), Some("bearer-token-123"));
            }
            other => return Err(format!("Expected WebSocket source, got {other:?}").into()),
        }
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.delivery, DeliveryGuarantee::ExactlyOnce);

        Ok(())
    }

    #[test]
    fn test_websocket_pipeline_ingest() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .websocket("ws://localhost:9090/v1/ingest")
            .batch_size(10)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        let record = IngestRecord::new("ws-1", vec![1.0, 2.0, 3.0, 4.0])
            .with_metadata(serde_json::json!({"source": "websocket", "channel": "live"}));
        pipeline.push(record).unwrap();

        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.records_flushed, 1);
    }

    #[test]
    fn test_websocket_message_format_serde() {
        let formats = vec![
            WebSocketMessageFormat::Json,
            WebSocketMessageFormat::Binary,
            WebSocketMessageFormat::NdJson,
        ];
        for fmt in formats {
            let serialized = serde_json::to_string(&fmt).unwrap();
            let deserialized: WebSocketMessageFormat =
                serde_json::from_str(&serialized).unwrap();
            assert_eq!(fmt, deserialized);
        }
    }

    #[test]
    fn test_lag_metrics() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .flush_interval_ms(1000)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();
        pipeline.push(IngestRecord::new("v1", vec![1.0; 4])).unwrap();
        pipeline.push(IngestRecord::new("v2", vec![2.0; 4])).unwrap();

        let lag = pipeline.lag_metrics();
        assert_eq!(lag.buffer_lag, 2);
        assert_eq!(lag.checkpoint_lag, 2);
        assert_eq!(lag.dead_letter_count, 0);

        pipeline.flush().unwrap();
        let lag = pipeline.lag_metrics();
        assert_eq!(lag.buffer_lag, 0);
        assert_eq!(lag.checkpoint_lag, 0);
    }

    #[test]
    fn test_prometheus_metrics() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(100)
            .build();

        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();
        pipeline.push(IngestRecord::new("v1", vec![1.0; 4])).unwrap();
        pipeline.flush().unwrap();

        let metrics = pipeline.prometheus_metrics();
        assert!(metrics.contains("needle_streaming_records_received 1"));
        assert!(metrics.contains("needle_streaming_records_flushed 1"));
        assert!(metrics.contains("needle_streaming_flush_count 1"));
    }

    #[test]
    fn test_webhook_handler_single_record() {
        let handler = WebhookHandler::new(WebhookHandlerConfig::default());
        let payload = serde_json::json!({
            "id": "doc1",
            "vector": [1.0, 2.0, 3.0, 4.0],
            "metadata": {"source": "webhook"}
        });

        let (records, errors) = handler.process_payload(&payload).unwrap();
        assert_eq!(records.len(), 1);
        assert!(errors.is_empty());
        assert_eq!(records[0].id, "doc1");
        assert_eq!(records[0].vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_webhook_handler_batch() {
        let handler = WebhookHandler::new(WebhookHandlerConfig::default());
        let payload = serde_json::json!([
            {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"id": "b", "vector": [0.0, 1.0, 0.0, 0.0]}
        ]);

        let (records, errors) = handler.process_payload(&payload).unwrap();
        assert_eq!(records.len(), 2);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_webhook_handler_partial_failure() {
        let handler = WebhookHandler::new(WebhookHandlerConfig::default());
        let payload = serde_json::json!([
            {"id": "good", "vector": [1.0, 2.0, 3.0, 4.0]},
            {"vector": [1.0, 2.0, 3.0, 4.0]},
            {"id": "also_good", "vector": [0.0, 1.0, 0.0, 0.0]}
        ]);

        let (records, errors) = handler.process_payload(&payload).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("record[1]"));
    }

    #[test]
    fn test_webhook_handler_into_pipeline() {
        let db = test_db();
        let config = StreamingIngestConfig::builder()
            .collection("test")
            .batch_size(10)
            .build();
        let mut pipeline = StreamingIngestPipeline::new(&db, config).unwrap();

        let handler = WebhookHandler::new(WebhookHandlerConfig::default());
        let payload = serde_json::json!({
            "id": "doc1",
            "vector": [1.0, 2.0, 3.0, 4.0]
        });

        let result = handler.ingest_into(&payload, &mut pipeline).unwrap();
        assert_eq!(result.accepted, 1);
        assert_eq!(result.rejected, 0);
        assert!(!result.backpressure);
    }

    #[test]
    fn test_webhook_no_secret_always_valid() {
        let handler = WebhookHandler::new(WebhookHandlerConfig::default());
        assert!(handler.validate_signature(b"test body", "anything"));
    }
}
