//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Vector Streaming API
//!
//! **DEPRECATED**: This module overlaps with `crate::streaming` and will be merged in a
//! future release. Prefer using `crate::streaming` for new code.
//!
//! Real-time CDC connectors for ingesting vectors from message queues like Kafka and Pulsar
//! with exactly-once semantics and automatic batching.
//!
//! # Features
//!
//! - **Kafka Consumer**: Consume vectors from Kafka topics with offset management
//! - **Pulsar Consumer**: Consume vectors from Pulsar topics with acknowledgments
//! - **Exactly-Once Semantics**: Transactional ingestion with deduplication
//! - **Automatic Batching**: Configurable batch sizes for optimal throughput
//! - **Backpressure Handling**: Pause/resume based on collection capacity
//! - **Schema Registry**: Support for Avro, JSON, and binary vector formats
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::vector_streaming::{VectorConsumer, ConsumerConfig, VectorMessage};
//!
//! let config = ConsumerConfig::kafka("localhost:9092", "vectors-topic")
//!     .group_id("needle-consumer")
//!     .batch_size(100)
//!     .auto_commit(false);
//!
//! let consumer = VectorConsumer::new(config).await?;
//!
//! // Process vectors
//! while let Some(batch) = consumer.poll().await? {
//!     for msg in batch {
//!         collection.insert(&msg.id, &msg.vector, msg.metadata)?;
//!     }
//!     consumer.commit().await?;
//! }
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use parking_lot::RwLock;
use tracing::{debug, info};

/// Message source type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageSource {
    /// Apache Kafka
    Kafka,
    /// Apache Pulsar
    Pulsar,
    /// PostgreSQL CDC
    Postgres,
    /// MongoDB Change Streams
    MongoDB,
    /// Custom/Mock source for testing
    Mock,
}

/// Vector format in messages
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorFormat {
    /// JSON array of floats
    Json,
    /// Binary float32 little-endian
    BinaryF32LE,
    /// Binary float32 big-endian
    BinaryF32BE,
    /// Base64-encoded binary
    Base64,
    /// Avro schema-encoded
    Avro,
}

impl Default for VectorFormat {
    fn default() -> Self {
        VectorFormat::Json
    }
}

/// Consumer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerConfig {
    /// Message source type
    pub source: MessageSource,
    /// Bootstrap servers (Kafka) or service URL (Pulsar)
    pub brokers: String,
    /// Topic to consume from
    pub topic: String,
    /// Consumer group ID
    pub group_id: String,
    /// Batch size for polling
    pub batch_size: usize,
    /// Polling timeout in milliseconds
    pub poll_timeout_ms: u64,
    /// Auto-commit offsets
    pub auto_commit: bool,
    /// Auto-commit interval in milliseconds
    pub auto_commit_interval_ms: u64,
    /// Vector format in messages
    pub vector_format: VectorFormat,
    /// Field name for vector ID in JSON messages
    pub id_field: String,
    /// Field name for vector data in JSON messages
    pub vector_field: String,
    /// Field name for metadata in JSON messages
    pub metadata_field: Option<String>,
    /// Enable exactly-once semantics
    pub exactly_once: bool,
    /// Maximum retries for failed messages
    pub max_retries: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Enable deduplication
    pub deduplication: bool,
    /// Deduplication window size
    pub dedup_window_size: usize,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            source: MessageSource::Mock,
            brokers: "localhost:9092".to_string(),
            topic: "vectors".to_string(),
            group_id: "needle-consumer".to_string(),
            batch_size: 100,
            poll_timeout_ms: 1000,
            auto_commit: false,
            auto_commit_interval_ms: 5000,
            vector_format: VectorFormat::Json,
            id_field: "id".to_string(),
            vector_field: "vector".to_string(),
            metadata_field: Some("metadata".to_string()),
            exactly_once: true,
            max_retries: 3,
            retry_delay_ms: 1000,
            deduplication: true,
            dedup_window_size: 10000,
        }
    }
}

impl ConsumerConfig {
    /// Create Kafka consumer config
    pub fn kafka(brokers: &str, topic: &str) -> Self {
        Self {
            source: MessageSource::Kafka,
            brokers: brokers.to_string(),
            topic: topic.to_string(),
            ..Default::default()
        }
    }

    /// Create Pulsar consumer config
    pub fn pulsar(service_url: &str, topic: &str) -> Self {
        Self {
            source: MessageSource::Pulsar,
            brokers: service_url.to_string(),
            topic: topic.to_string(),
            ..Default::default()
        }
    }

    /// Create mock consumer config for testing
    pub fn mock() -> Self {
        Self {
            source: MessageSource::Mock,
            ..Default::default()
        }
    }

    /// Set consumer group ID
    pub fn group_id(mut self, group_id: &str) -> Self {
        self.group_id = group_id.to_string();
        self
    }

    /// Set batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set auto-commit behavior
    pub fn auto_commit(mut self, enabled: bool) -> Self {
        self.auto_commit = enabled;
        self
    }

    /// Set vector format
    pub fn vector_format(mut self, format: VectorFormat) -> Self {
        self.vector_format = format;
        self
    }

    /// Enable exactly-once semantics
    pub fn exactly_once(mut self, enabled: bool) -> Self {
        self.exactly_once = enabled;
        self
    }

    /// Enable deduplication
    pub fn deduplication(mut self, enabled: bool) -> Self {
        self.deduplication = enabled;
        self
    }

    /// Create PostgreSQL CDC config
    pub fn postgres(connection_string: &str, table: &str) -> Self {
        Self {
            source: MessageSource::Postgres,
            brokers: connection_string.to_string(),
            topic: table.to_string(),
            group_id: "needle-pg-cdc".to_string(),
            auto_commit: true,
            ..Default::default()
        }
    }

    /// Create MongoDB change stream config
    pub fn mongodb(connection_string: &str, collection: &str) -> Self {
        Self {
            source: MessageSource::MongoDB,
            brokers: connection_string.to_string(),
            topic: collection.to_string(),
            group_id: "needle-mongo-cdc".to_string(),
            auto_commit: true,
            ..Default::default()
        }
    }
}

/// A vector message from the stream
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMessage {
    /// Vector ID
    pub id: String,
    /// Vector data
    pub vector: Vec<f32>,
    /// Optional metadata
    pub metadata: Option<Value>,
    /// Message offset/position
    pub offset: u64,
    /// Partition (for Kafka)
    pub partition: Option<i32>,
    /// Timestamp
    pub timestamp: u64,
    /// Message key (optional)
    pub key: Option<String>,
}

/// Offset tracking for exactly-once semantics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OffsetState {
    /// Committed offsets per partition
    pub committed: HashMap<i32, u64>,
    /// Pending offsets (processed but not committed)
    pub pending: HashMap<i32, u64>,
    /// Last commit timestamp
    pub last_commit: u64,
}

/// Consumer statistics
#[derive(Debug, Clone, Default)]
pub struct ConsumerStats {
    /// Total messages consumed
    pub messages_consumed: u64,
    /// Total bytes consumed
    pub bytes_consumed: u64,
    /// Messages per second (recent average)
    pub messages_per_second: f64,
    /// Bytes per second (recent average)
    pub bytes_per_second: f64,
    /// Total errors encountered
    pub errors: u64,
    /// Duplicates filtered
    pub duplicates_filtered: u64,
    /// Current lag (messages behind latest)
    pub lag: u64,
    /// Consumer uptime in seconds
    pub uptime_seconds: u64,
}

/// Vector consumer for streaming ingestion
pub struct VectorConsumer {
    /// Configuration
    config: ConsumerConfig,
    /// Running state
    running: AtomicBool,
    /// Offset tracking
    offsets: RwLock<OffsetState>,
    /// Deduplication cache (recent IDs)
    dedup_cache: RwLock<VecDeque<String>>,
    /// Dedup set for O(1) lookup
    dedup_set: RwLock<HashSet<String>>,
    /// Statistics
    stats: RwLock<ConsumerStatsInternal>,
    /// Start time
    start_time: Instant,
    /// Message buffer for mock source
    mock_buffer: RwLock<VecDeque<VectorMessage>>,
}

#[derive(Debug, Default)]
struct ConsumerStatsInternal {
    messages_consumed: AtomicU64,
    bytes_consumed: AtomicU64,
    errors: AtomicU64,
    duplicates_filtered: AtomicU64,
    #[allow(dead_code)]
    recent_messages: VecDeque<(Instant, u64)>,
}

impl VectorConsumer {
    /// Create a new vector consumer
    pub fn new(config: ConsumerConfig) -> Result<Self> {
        info!(
            source = ?config.source,
            brokers = %config.brokers,
            topic = %config.topic,
            "Creating vector consumer"
        );

        Ok(Self {
            config,
            running: AtomicBool::new(false),
            offsets: RwLock::new(OffsetState::default()),
            dedup_cache: RwLock::new(VecDeque::new()),
            dedup_set: RwLock::new(HashSet::new()),
            stats: RwLock::new(ConsumerStatsInternal::default()),
            start_time: Instant::now(),
            mock_buffer: RwLock::new(VecDeque::new()),
        })
    }

    /// Start the consumer
    pub fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(NeedleError::InvalidOperation(
                "Consumer already running".to_string(),
            ));
        }

        info!("Vector consumer started");
        Ok(())
    }

    /// Stop the consumer
    pub fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        info!("Vector consumer stopped");
        Ok(())
    }

    /// Check if consumer is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Poll for a batch of messages
    pub fn poll(&self) -> Result<Option<Vec<VectorMessage>>> {
        if !self.is_running() {
            return Ok(None);
        }

        let messages = match self.config.source {
            MessageSource::Mock => self.poll_mock()?,
            MessageSource::Kafka => self.poll_kafka()?,
            MessageSource::Pulsar => self.poll_pulsar()?,
            MessageSource::Postgres => self.poll_postgres()?,
            MessageSource::MongoDB => self.poll_mongodb()?,
        };

        if messages.is_empty() {
            return Ok(None);
        }

        // Apply deduplication if enabled
        let messages = if self.config.deduplication {
            self.deduplicate(messages)?
        } else {
            messages
        };

        // Update stats
        {
            let stats = self.stats.read();
            stats.messages_consumed.fetch_add(messages.len() as u64, Ordering::Relaxed);
            let bytes: usize = messages.iter().map(|m| m.vector.len() * 4).sum();
            stats.bytes_consumed.fetch_add(bytes as u64, Ordering::Relaxed);
        }

        if messages.is_empty() {
            Ok(None)
        } else {
            Ok(Some(messages))
        }
    }

    /// Poll from mock source (for testing)
    fn poll_mock(&self) -> Result<Vec<VectorMessage>> {
        let mut buffer = self.mock_buffer.write();
        let batch_size = self.config.batch_size.min(buffer.len());
        let messages: Vec<VectorMessage> = buffer.drain(..batch_size).collect();
        Ok(messages)
    }

    /// Poll from Kafka
    #[cfg(not(feature = "cdc-kafka"))]
    fn poll_kafka(&self) -> Result<Vec<VectorMessage>> {
        debug!("Kafka poll (stub - enable cdc-kafka feature for real implementation)");
        Ok(Vec::new())
    }

    /// Poll from Pulsar
    #[cfg(not(feature = "cdc-pulsar"))]
    fn poll_pulsar(&self) -> Result<Vec<VectorMessage>> {
        debug!("Pulsar poll (stub - enable cdc-pulsar feature for real implementation)");
        Ok(Vec::new())
    }

    /// Poll from PostgreSQL CDC
    #[cfg(not(feature = "cdc-postgres"))]
    fn poll_postgres(&self) -> Result<Vec<VectorMessage>> {
        debug!("PostgreSQL CDC poll (stub - enable cdc-postgres feature for real implementation)");
        Ok(Vec::new())
    }

    /// Poll from MongoDB change streams
    #[cfg(not(feature = "cdc-mongodb"))]
    fn poll_mongodb(&self) -> Result<Vec<VectorMessage>> {
        debug!("MongoDB CDC poll (stub - enable cdc-mongodb feature for real implementation)");
        Ok(Vec::new())
    }

    /// Deduplicate messages
    fn deduplicate(&self, messages: Vec<VectorMessage>) -> Result<Vec<VectorMessage>> {
        let mut cache = self.dedup_cache.write();
        let mut set = self.dedup_set.write();
        let stats = self.stats.read();

        let mut result = Vec::with_capacity(messages.len());

        for msg in messages {
            if set.contains(&msg.id) {
                stats.duplicates_filtered.fetch_add(1, Ordering::Relaxed);
                continue;
            }

            // Add to cache
            if cache.len() >= self.config.dedup_window_size {
                if let Some(old_id) = cache.pop_front() {
                    set.remove(&old_id);
                }
            }
            cache.push_back(msg.id.clone());
            set.insert(msg.id.clone());

            result.push(msg);
        }

        Ok(result)
    }

    /// Commit current offsets
    pub fn commit(&self) -> Result<()> {
        let mut offsets = self.offsets.write();

        // Move pending to committed - collect first to avoid borrow conflict
        let pending: Vec<_> = offsets.pending.drain().collect();
        for (partition, offset) in pending {
            offsets.committed.insert(partition, offset);
        }

        offsets.last_commit = current_timestamp();

        debug!(offsets = ?offsets.committed, "Offsets committed");
        Ok(())
    }

    /// Commit specific offset for partition
    pub fn commit_offset(&self, partition: i32, offset: u64) -> Result<()> {
        let mut offsets = self.offsets.write();
        offsets.committed.insert(partition, offset);
        offsets.last_commit = current_timestamp();
        Ok(())
    }

    /// Get current offset state
    pub fn offset_state(&self) -> OffsetState {
        self.offsets.read().clone()
    }

    /// Add mock messages for testing
    pub fn add_mock_messages(&self, messages: Vec<VectorMessage>) {
        let mut buffer = self.mock_buffer.write();
        for msg in messages {
            buffer.push_back(msg);
        }
    }

    /// Parse a JSON message into a VectorMessage
    pub fn parse_json_message(&self, json: &str, offset: u64) -> Result<VectorMessage> {
        let value: Value = serde_json::from_str(json)
            .map_err(|e| NeedleError::InvalidInput(format!("JSON parse error: {}", e)))?;

        let id = value
            .get(&self.config.id_field)
            .and_then(|v| v.as_str())
            .ok_or_else(|| NeedleError::InvalidInput("Missing id field".to_string()))?
            .to_string();

        let vector: Vec<f32> = value
            .get(&self.config.vector_field)
            .and_then(|v| v.as_array())
            .ok_or_else(|| NeedleError::InvalidInput("Missing vector field".to_string()))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if vector.is_empty() {
            return Err(NeedleError::InvalidInput("Empty vector".to_string()));
        }

        let metadata = self
            .config
            .metadata_field
            .as_ref()
            .and_then(|field| value.get(field).cloned());

        Ok(VectorMessage {
            id,
            vector,
            metadata,
            offset,
            partition: None,
            timestamp: current_timestamp(),
            key: None,
        })
    }

    /// Parse binary vector data
    pub fn parse_binary_vector(&self, data: &[u8]) -> Result<Vec<f32>> {
        match self.config.vector_format {
            VectorFormat::BinaryF32LE => {
                if data.len() % 4 != 0 {
                    return Err(NeedleError::InvalidInput(
                        "Invalid binary vector length".to_string(),
                    ));
                }
                Ok(data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            VectorFormat::BinaryF32BE => {
                if data.len() % 4 != 0 {
                    return Err(NeedleError::InvalidInput(
                        "Invalid binary vector length".to_string(),
                    ));
                }
                Ok(data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            }
            VectorFormat::Base64 => {
                // Decode base64 then parse as LE
                let decoded = base64_decode(data)?;
                self.parse_binary_vector_bytes(&decoded, true)
            }
            _ => Err(NeedleError::InvalidInput(
                "Unsupported binary format".to_string(),
            )),
        }
    }

    fn parse_binary_vector_bytes(&self, data: &[u8], little_endian: bool) -> Result<Vec<f32>> {
        if data.len() % 4 != 0 {
            return Err(NeedleError::InvalidInput(
                "Invalid binary vector length".to_string(),
            ));
        }
        Ok(data
            .chunks_exact(4)
            .map(|chunk| {
                if little_endian {
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                } else {
                    f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                }
            })
            .collect())
    }

    /// Get consumer statistics
    pub fn stats(&self) -> ConsumerStats {
        let stats = self.stats.read();
        let uptime = self.start_time.elapsed().as_secs();

        // Calculate rates
        let messages = stats.messages_consumed.load(Ordering::Relaxed);
        let bytes = stats.bytes_consumed.load(Ordering::Relaxed);
        let (mps, bps) = if uptime > 0 {
            (messages as f64 / uptime as f64, bytes as f64 / uptime as f64)
        } else {
            (0.0, 0.0)
        };

        ConsumerStats {
            messages_consumed: messages,
            bytes_consumed: bytes,
            messages_per_second: mps,
            bytes_per_second: bps,
            errors: stats.errors.load(Ordering::Relaxed),
            duplicates_filtered: stats.duplicates_filtered.load(Ordering::Relaxed),
            lag: 0, // Would be populated from broker
            uptime_seconds: uptime,
        }
    }
}

/// Producer for writing vectors to message queues
pub struct VectorProducer {
    /// Configuration
    config: ProducerConfig,
    /// Running state
    #[allow(dead_code)]
    running: AtomicBool,
    /// Statistics
    stats: RwLock<ProducerStats>,
    /// Mock output buffer for testing
    mock_output: RwLock<VecDeque<VectorMessage>>,
}

/// Producer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerConfig {
    /// Message source type
    pub source: MessageSource,
    /// Bootstrap servers
    pub brokers: String,
    /// Topic to produce to
    pub topic: String,
    /// Vector format
    pub vector_format: VectorFormat,
    /// Batch size for sending
    pub batch_size: usize,
    /// Linger time in milliseconds (wait for batch)
    pub linger_ms: u64,
    /// Compression type
    pub compression: CompressionType,
    /// Enable idempotent producer
    pub idempotent: bool,
    /// Acknowledgment mode
    pub acks: AckMode,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            source: MessageSource::Mock,
            brokers: "localhost:9092".to_string(),
            topic: "vectors".to_string(),
            vector_format: VectorFormat::Json,
            batch_size: 100,
            linger_ms: 5,
            compression: CompressionType::None,
            idempotent: true,
            acks: AckMode::All,
        }
    }
}

impl ProducerConfig {
    /// Create Kafka producer config
    pub fn kafka(brokers: &str, topic: &str) -> Self {
        Self {
            source: MessageSource::Kafka,
            brokers: brokers.to_string(),
            topic: topic.to_string(),
            ..Default::default()
        }
    }

    /// Create Pulsar producer config
    pub fn pulsar(service_url: &str, topic: &str) -> Self {
        Self {
            source: MessageSource::Pulsar,
            brokers: service_url.to_string(),
            topic: topic.to_string(),
            ..Default::default()
        }
    }
}

/// Compression type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Gzip,
    Snappy,
    Lz4,
    Zstd,
}

/// Acknowledgment mode
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AckMode {
    /// No acknowledgment (fire and forget)
    None,
    /// Leader acknowledgment only
    Leader,
    /// All replicas must acknowledge
    All,
}

/// Producer statistics
#[derive(Debug, Clone, Default)]
pub struct ProducerStats {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Errors encountered
    pub errors: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

impl VectorProducer {
    /// Create a new producer
    pub fn new(config: ProducerConfig) -> Result<Self> {
        Ok(Self {
            config,
            running: AtomicBool::new(true),
            stats: RwLock::new(ProducerStats::default()),
            mock_output: RwLock::new(VecDeque::new()),
        })
    }

    /// Send a single vector
    pub fn send(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        let msg = VectorMessage {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
            offset: 0,
            partition: None,
            timestamp: current_timestamp(),
            key: Some(id.to_string()),
        };

        self.send_message(msg)
    }

    /// Send a batch of vectors
    pub fn send_batch(
        &self,
        vectors: &[(String, Vec<f32>, Option<Value>)],
    ) -> Result<usize> {
        let mut sent = 0;
        for (id, vector, metadata) in vectors {
            self.send(id, vector, metadata.clone())?;
            sent += 1;
        }
        Ok(sent)
    }

    fn send_message(&self, msg: VectorMessage) -> Result<()> {
        match self.config.source {
            MessageSource::Mock => {
                self.mock_output.write().push_back(msg);
            }
            MessageSource::Kafka => {
                // Would use rdkafka producer
                debug!("Kafka send (stub)");
            }
            MessageSource::Pulsar => {
                // Would use pulsar producer
                debug!("Pulsar send (stub)");
            }
            _ => {}
        }

        // Update stats
        let mut stats = self.stats.write();
        stats.messages_sent += 1;

        Ok(())
    }

    /// Get mock output (for testing)
    pub fn get_mock_output(&self) -> Vec<VectorMessage> {
        self.mock_output.write().drain(..).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> ProducerStats {
        self.stats.read().clone()
    }

    /// Flush pending messages
    pub fn flush(&self) -> Result<()> {
        // Would flush internal buffers
        Ok(())
    }
}

/// Stream processor for transforming vectors
pub struct StreamProcessor {
    /// Input consumer
    consumer: Arc<VectorConsumer>,
    /// Processing function
    processor: Box<dyn Fn(VectorMessage) -> Option<VectorMessage> + Send + Sync>,
    /// Output producer (optional)
    producer: Option<Arc<VectorProducer>>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new<F>(consumer: Arc<VectorConsumer>, processor: F) -> Self
    where
        F: Fn(VectorMessage) -> Option<VectorMessage> + Send + Sync + 'static,
    {
        Self {
            consumer,
            processor: Box::new(processor),
            producer: None,
        }
    }

    /// Set output producer
    pub fn with_producer(mut self, producer: Arc<VectorProducer>) -> Self {
        self.producer = Some(producer);
        self
    }

    /// Process one batch
    pub fn process_batch(&self) -> Result<usize> {
        let batch = match self.consumer.poll()? {
            Some(b) => b,
            None => return Ok(0),
        };

        let mut processed = 0;
        for msg in batch {
            if let Some(transformed) = (self.processor)(msg) {
                if let Some(ref producer) = self.producer {
                    producer.send(
                        &transformed.id,
                        &transformed.vector,
                        transformed.metadata,
                    )?;
                }
                processed += 1;
            }
        }

        self.consumer.commit()?;
        Ok(processed)
    }
}

// Helper functions

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn base64_decode(data: &[u8]) -> Result<Vec<u8>> {
    // Simple base64 decode (would use base64 crate in production)
    let s = std::str::from_utf8(data)
        .map_err(|e| NeedleError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

    // Basic base64 decode
    let mut result = Vec::new();
    let chars: Vec<u8> = s.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();

    for chunk in chars.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let values: Vec<u8> = chunk
            .iter()
            .map(|&c| {
                match c {
                    b'A'..=b'Z' => c - b'A',
                    b'a'..=b'z' => c - b'a' + 26,
                    b'0'..=b'9' => c - b'0' + 52,
                    b'+' => 62,
                    b'/' => 63,
                    b'=' => 0,
                    _ => 0,
                }
            })
            .collect();

        result.push((values[0] << 2) | (values[1] >> 4));
        if chunk[2] != b'=' {
            result.push((values[1] << 4) | (values[2] >> 2));
        }
        if chunk[3] != b'=' {
            result.push((values[2] << 6) | values[3]);
        }
    }

    Ok(result)
}

// ============================================================================
// Exactly-Once Semantics Enhancements
// ============================================================================

/// Checkpoint storage for exactly-once recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Consumer group ID
    pub group_id: String,
    /// Topic name
    pub topic: String,
    /// Partition offsets
    pub offsets: HashMap<u32, u64>,
    /// Processed message IDs (idempotency tokens)
    pub processed_ids: HashSet<String>,
    /// Watermark (latest fully processed timestamp)
    pub watermark: u64,
    /// Checkpoint timestamp
    pub timestamp: u64,
    /// Checkpoint sequence number
    pub sequence: u64,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(group_id: &str, topic: &str) -> Self {
        Self {
            group_id: group_id.to_string(),
            topic: topic.to_string(),
            offsets: HashMap::new(),
            processed_ids: HashSet::new(),
            watermark: 0,
            timestamp: current_timestamp(),
            sequence: 0,
        }
    }

    /// Update offset for a partition
    pub fn update_offset(&mut self, partition: u32, offset: u64) {
        self.offsets.insert(partition, offset);
        self.timestamp = current_timestamp();
        self.sequence += 1;
    }

    /// Mark a message as processed
    pub fn mark_processed(&mut self, id: &str) {
        self.processed_ids.insert(id.to_string());
    }

    /// Check if a message was already processed
    pub fn is_processed(&self, id: &str) -> bool {
        self.processed_ids.contains(id)
    }

    /// Update watermark
    pub fn update_watermark(&mut self, watermark: u64) {
        if watermark > self.watermark {
            self.watermark = watermark;
        }
    }

    /// Trim old processed IDs to prevent unbounded growth
    pub fn trim_processed_ids(&mut self, max_size: usize) {
        if self.processed_ids.len() > max_size {
            // Keep only the most recent entries
            // In practice, would use a bounded LRU set
            let excess = self.processed_ids.len() - max_size;
            let to_remove: Vec<_> = self.processed_ids.iter().take(excess).cloned().collect();
            for id in to_remove {
                self.processed_ids.remove(&id);
            }
        }
    }
}

/// Checkpoint storage trait for persistence
pub trait CheckpointStore: Send + Sync {
    /// Save checkpoint
    fn save(&self, checkpoint: &Checkpoint) -> Result<()>;
    /// Load checkpoint
    fn load(&self, group_id: &str, topic: &str) -> Result<Option<Checkpoint>>;
    /// Delete checkpoint
    fn delete(&self, group_id: &str, topic: &str) -> Result<()>;
}

/// In-memory checkpoint store (for testing)
#[derive(Debug, Default)]
pub struct InMemoryCheckpointStore {
    checkpoints: RwLock<HashMap<String, Checkpoint>>,
}

impl InMemoryCheckpointStore {
    /// Create new in-memory store
    pub fn new() -> Self {
        Self::default()
    }

    fn key(group_id: &str, topic: &str) -> String {
        format!("{}:{}", group_id, topic)
    }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let key = Self::key(&checkpoint.group_id, &checkpoint.topic);
        self.checkpoints.write().insert(key, checkpoint.clone());
        Ok(())
    }

    fn load(&self, group_id: &str, topic: &str) -> Result<Option<Checkpoint>> {
        let key = Self::key(group_id, topic);
        Ok(self.checkpoints.read().get(&key).cloned())
    }

    fn delete(&self, group_id: &str, topic: &str) -> Result<()> {
        let key = Self::key(group_id, topic);
        self.checkpoints.write().remove(&key);
        Ok(())
    }
}

/// Dead letter queue for failed messages
#[derive(Debug)]
pub struct DeadLetterQueue {
    /// Failed messages
    messages: RwLock<VecDeque<FailedMessage>>,
    /// Maximum queue size
    max_size: usize,
    /// Statistics
    stats: RwLock<DlqStats>,
}

/// A message that failed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedMessage {
    /// Original message
    pub message: VectorMessage,
    /// Failure reason
    pub error: String,
    /// Number of retry attempts
    pub retry_count: u32,
    /// First failure timestamp
    pub first_failure: u64,
    /// Last failure timestamp
    pub last_failure: u64,
}

/// DLQ statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DlqStats {
    /// Total messages sent to DLQ
    pub total_failed: u64,
    /// Messages retried successfully
    pub retried_success: u64,
    /// Messages permanently failed
    pub permanently_failed: u64,
    /// Current queue size
    pub queue_size: usize,
}

impl DeadLetterQueue {
    /// Create a new DLQ
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: RwLock::new(VecDeque::with_capacity(max_size)),
            max_size,
            stats: RwLock::new(DlqStats::default()),
        }
    }

    /// Add a failed message
    pub fn push(&self, message: VectorMessage, error: &str, retry_count: u32) {
        let now = current_timestamp();
        let failed = FailedMessage {
            message,
            error: error.to_string(),
            retry_count,
            first_failure: now,
            last_failure: now,
        };

        let mut queue = self.messages.write();
        let mut stats = self.stats.write();

        // Evict oldest if at capacity
        if queue.len() >= self.max_size {
            queue.pop_front();
            stats.permanently_failed += 1;
        }

        queue.push_back(failed);
        stats.total_failed += 1;
        stats.queue_size = queue.len();
    }

    /// Pop a message for retry
    pub fn pop(&self) -> Option<FailedMessage> {
        let mut queue = self.messages.write();
        let msg = queue.pop_front();
        if msg.is_some() {
            self.stats.write().queue_size = queue.len();
        }
        msg
    }

    /// Get all messages (for inspection)
    pub fn peek_all(&self) -> Vec<FailedMessage> {
        self.messages.read().iter().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> DlqStats {
        self.stats.read().clone()
    }

    /// Mark a retry as successful
    pub fn mark_retry_success(&self) {
        self.stats.write().retried_success += 1;
    }

    /// Clear the queue
    pub fn clear(&self) {
        self.messages.write().clear();
        self.stats.write().queue_size = 0;
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.messages.read().len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.messages.read().is_empty()
    }
}

/// Transactional batch processor with exactly-once semantics
pub struct TransactionalProcessor {
    /// Consumer
    consumer: Arc<VectorConsumer>,
    /// Checkpoint store
    checkpoint_store: Arc<dyn CheckpointStore>,
    /// Dead letter queue
    dlq: DeadLetterQueue,
    /// Current checkpoint
    checkpoint: RwLock<Checkpoint>,
    /// Processing function
    processor: Box<dyn Fn(&VectorMessage) -> Result<()> + Send + Sync>,
    /// Maximum retries before DLQ
    max_retries: u32,
    /// Checkpoint interval (number of messages)
    checkpoint_interval: usize,
    /// Messages since last checkpoint
    messages_since_checkpoint: AtomicU64,
    /// Optional backpressure controller
    backpressure: Option<Arc<BackpressureController>>,
}

impl TransactionalProcessor {
    /// Create a new transactional processor
    pub fn new<F>(
        consumer: Arc<VectorConsumer>,
        checkpoint_store: Arc<dyn CheckpointStore>,
        processor: F,
    ) -> Result<Self>
    where
        F: Fn(&VectorMessage) -> Result<()> + Send + Sync + 'static,
    {
        let config = &consumer.config;

        // Load or create checkpoint
        let checkpoint = checkpoint_store
            .load(&config.group_id, &config.topic)?
            .unwrap_or_else(|| Checkpoint::new(&config.group_id, &config.topic));

        Ok(Self {
            consumer,
            checkpoint_store,
            dlq: DeadLetterQueue::new(10000),
            checkpoint: RwLock::new(checkpoint),
            processor: Box::new(processor),
            max_retries: 3,
            checkpoint_interval: 1000,
            messages_since_checkpoint: AtomicU64::new(0),
            backpressure: None,
        })
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set checkpoint interval
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Attach a backpressure controller to throttle ingestion automatically.
    pub fn with_backpressure(mut self, controller: Arc<BackpressureController>) -> Self {
        self.backpressure = Some(controller);
        self
    }

    /// Process a batch with exactly-once semantics
    pub fn process_batch(&self) -> Result<BatchResult> {
        // Check backpressure before polling
        if let Some(bp) = &self.backpressure {
            if !bp.should_poll() {
                return Ok(BatchResult::default());
            }
        }

        let batch = match self.consumer.poll()? {
            Some(b) => b,
            None => return Ok(BatchResult::default()),
        };

        // Apply backpressure-aware batch limiting
        let batch: Vec<_> = if let Some(bp) = &self.backpressure {
            let effective = bp.effective_batch_size(batch.len());
            if effective == 0 {
                return Ok(BatchResult::default());
            }
            bp.on_receive(effective as u64);
            batch.into_iter().take(effective).collect()
        } else {
            batch
        };

        let mut processed = 0;
        let mut skipped = 0;
        let mut failed = 0;

        for msg in batch {
            // Check if already processed (idempotency)
            if self.checkpoint.read().is_processed(&msg.id) {
                skipped += 1;
                continue;
            }

            // Process with retry
            let result = self.process_with_retry(&msg);

            match result {
                Ok(()) => {
                    // Mark as processed
                    self.checkpoint.write().mark_processed(&msg.id);

                    // Update offset
                    if let Some(partition) = msg.partition {
                        self.checkpoint.write().update_offset(partition as u32, msg.offset);
                    }

                    // Update watermark
                    self.checkpoint.write().update_watermark(msg.timestamp);

                    processed += 1;
                }
                Err(e) => {
                    // Send to DLQ
                    self.dlq.push(msg, &e.to_string(), self.max_retries);
                    failed += 1;
                }
            }
        }

        // Maybe checkpoint
        let count = self.messages_since_checkpoint.fetch_add(processed as u64, Ordering::Relaxed);
        if count + processed as u64 >= self.checkpoint_interval as u64 {
            self.save_checkpoint()?;
            self.messages_since_checkpoint.store(0, Ordering::Relaxed);
        }

        // Commit offsets
        self.consumer.commit()?;

        // Notify backpressure controller of committed messages
        if let Some(bp) = &self.backpressure {
            bp.on_commit(processed as u64);
        }

        Ok(BatchResult {
            processed,
            skipped,
            failed,
        })
    }

    /// Process a single message with retry
    fn process_with_retry(&self, msg: &VectorMessage) -> Result<()> {
        let mut attempts = 0;
        loop {
            match (self.processor)(msg) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_retries {
                        return Err(e);
                    }
                    // Simple backoff
                    std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
                }
            }
        }
    }

    /// Save checkpoint to store
    pub fn save_checkpoint(&self) -> Result<()> {
        let checkpoint = self.checkpoint.read().clone();
        self.checkpoint_store.save(&checkpoint)?;
        debug!(sequence = checkpoint.sequence, "Checkpoint saved");
        Ok(())
    }

    /// Get current checkpoint
    pub fn checkpoint(&self) -> Checkpoint {
        self.checkpoint.read().clone()
    }

    /// Get DLQ reference
    pub fn dlq(&self) -> &DeadLetterQueue {
        &self.dlq
    }

    /// Retry messages from DLQ
    pub fn retry_dlq(&self) -> Result<usize> {
        let mut retried = 0;

        while let Some(failed) = self.dlq.pop() {
            if failed.retry_count >= self.max_retries {
                // Re-add to DLQ as permanently failed
                self.dlq.push(
                    failed.message,
                    &format!("Max retries exceeded: {}", failed.error),
                    failed.retry_count,
                );
                continue;
            }

            match (self.processor)(&failed.message) {
                Ok(()) => {
                    self.dlq.mark_retry_success();
                    self.checkpoint.write().mark_processed(&failed.message.id);
                    retried += 1;
                }
                Err(e) => {
                    self.dlq.push(failed.message, &e.to_string(), failed.retry_count + 1);
                }
            }
        }

        Ok(retried)
    }
}

/// Result of batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchResult {
    /// Successfully processed messages
    pub processed: usize,
    /// Skipped (already processed) messages
    pub skipped: usize,
    /// Failed messages (sent to DLQ)
    pub failed: usize,
}

impl BatchResult {
    /// Total messages in batch
    pub fn total(&self) -> usize {
        self.processed + self.skipped + self.failed
    }
}

/// Watermark tracker for windowed operations
#[derive(Debug)]
pub struct WatermarkTracker {
    /// Current watermark per partition
    watermarks: RwLock<HashMap<u32, u64>>,
    /// Global watermark (minimum across partitions)
    global_watermark: AtomicU64,
    /// Allowed lateness in milliseconds
    allowed_lateness: u64,
}

impl WatermarkTracker {
    /// Create a new watermark tracker
    pub fn new(allowed_lateness_ms: u64) -> Self {
        Self {
            watermarks: RwLock::new(HashMap::new()),
            global_watermark: AtomicU64::new(0),
            allowed_lateness: allowed_lateness_ms,
        }
    }

    /// Update watermark for a partition
    pub fn update(&self, partition: u32, timestamp: u64) {
        let mut watermarks = self.watermarks.write();
        let current = watermarks.entry(partition).or_insert(0);
        if timestamp > *current {
            *current = timestamp;
        }

        // Update global watermark
        if let Some(min) = watermarks.values().min() {
            self.global_watermark.store(*min, Ordering::SeqCst);
        }
    }

    /// Get current global watermark
    pub fn watermark(&self) -> u64 {
        self.global_watermark.load(Ordering::SeqCst)
    }

    /// Get watermark for a specific partition
    pub fn partition_watermark(&self, partition: u32) -> Option<u64> {
        self.watermarks.read().get(&partition).copied()
    }

    /// Check if a timestamp is late
    pub fn is_late(&self, timestamp: u64) -> bool {
        let watermark = self.watermark();
        watermark > 0 && timestamp + self.allowed_lateness < watermark
    }

    /// Get all partition watermarks
    pub fn all_watermarks(&self) -> HashMap<u32, u64> {
        self.watermarks.read().clone()
    }
}

// ============================================================================
// Feature-Gated CDC Client Implementations
// ============================================================================

/// Kafka CDC implementation using rdkafka
#[cfg(feature = "cdc-kafka")]
pub mod kafka_cdc {
    use super::*;
    use rdkafka::config::ClientConfig;
    use rdkafka::consumer::{Consumer, StreamConsumer, CommitMode};
    use rdkafka::message::Message;
    use std::time::Duration;

    /// Kafka consumer wrapper
    pub struct KafkaVectorConsumer {
        consumer: StreamConsumer,
        config: ConsumerConfig,
    }

    impl KafkaVectorConsumer {
        /// Create a new Kafka consumer
        pub fn new(config: ConsumerConfig) -> Result<Self> {
            let consumer: StreamConsumer = ClientConfig::new()
                .set("bootstrap.servers", &config.brokers)
                .set("group.id", &config.group_id)
                .set("enable.auto.commit", config.auto_commit.to_string())
                .set("auto.offset.reset", "earliest")
                .set("session.timeout.ms", "6000")
                .create()
                .map_err(|e| NeedleError::InvalidOperation(format!("Kafka config error: {}", e)))?;

            consumer.subscribe(&[&config.topic])
                .map_err(|e| NeedleError::InvalidOperation(format!("Kafka subscribe error: {}", e)))?;

            Ok(Self { consumer, config })
        }

        /// Poll for messages (blocking)
        pub fn poll(&self, timeout: Duration) -> Result<Vec<VectorMessage>> {
            let mut messages = Vec::new();

            for _ in 0..self.config.batch_size {
                match self.consumer.poll(timeout) {
                    Some(Ok(msg)) => {
                        if let Some(payload) = msg.payload() {
                            if let Ok(vec_msg) = self.parse_message(payload, &msg) {
                                messages.push(vec_msg);
                            }
                        }
                    }
                    Some(Err(e)) => {
                        debug!("Kafka poll error: {}", e);
                        break;
                    }
                    None => break,
                }
            }

            Ok(messages)
        }

        /// Parse a Kafka message into VectorMessage
        fn parse_message(&self, payload: &[u8], msg: &rdkafka::message::BorrowedMessage) -> Result<VectorMessage> {
            match self.config.vector_format {
                VectorFormat::Json => {
                    let json: serde_json::Value = serde_json::from_slice(payload)
                        .map_err(|e| NeedleError::InvalidFormat(format!("JSON parse error: {}", e)))?;

                    let id = json.get(&self.config.id_field)
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| NeedleError::InvalidFormat("Missing id field".to_string()))?
                        .to_string();

                    let vector: Vec<f32> = json.get(&self.config.vector_field)
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| NeedleError::InvalidFormat("Missing vector field".to_string()))?
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();

                    let metadata = self.config.metadata_field.as_ref()
                        .and_then(|field| json.get(field).cloned());

                    Ok(VectorMessage {
                        id,
                        vector,
                        metadata,
                        offset: msg.offset() as u64,
                        partition: Some(msg.partition()),
                        timestamp: msg.timestamp().to_millis().unwrap_or(0) as u64,
                        key: msg.key().map(|k| String::from_utf8_lossy(k).to_string()),
                    })
                }
                VectorFormat::BinaryF32LE => {
                    // Binary format: first 8 bytes = id length, then id, then vector
                    if payload.len() < 8 {
                        return Err(NeedleError::InvalidFormat("Payload too short".to_string()));
                    }
                    let id_len = u64::from_le_bytes(payload[..8].try_into().expect("slice is exactly 8 bytes")) as usize;
                    if payload.len() < 8 + id_len {
                        return Err(NeedleError::InvalidFormat("Invalid id length".to_string()));
                    }
                    let id = String::from_utf8_lossy(&payload[8..8 + id_len]).to_string();
                    let vector_bytes = &payload[8 + id_len..];
                    let vector: Vec<f32> = vector_bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk is exactly 4 bytes")))
                        .collect();

                    Ok(VectorMessage {
                        id,
                        vector,
                        metadata: None,
                        offset: msg.offset() as u64,
                        partition: Some(msg.partition()),
                        timestamp: msg.timestamp().to_millis().unwrap_or(0) as u64,
                        key: msg.key().map(|k| String::from_utf8_lossy(k).to_string()),
                    })
                }
                _ => Err(NeedleError::InvalidFormat(format!(
                    "Unsupported format: {:?}",
                    self.config.vector_format
                ))),
            }
        }

        /// Commit offsets
        pub fn commit(&self) -> Result<()> {
            self.consumer.commit_consumer_state(CommitMode::Sync)
                .map_err(|e| NeedleError::InvalidOperation(format!("Kafka commit error: {}", e)))
        }
    }
}

/// Pulsar CDC implementation
#[cfg(feature = "cdc-pulsar")]
pub mod pulsar_cdc {
    use super::*;
    use pulsar::{Pulsar, TokioExecutor, Consumer as PulsarConsumer, SubType, consumer::InitialPosition};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Pulsar consumer wrapper
    pub struct PulsarVectorConsumer {
        consumer: Arc<Mutex<PulsarConsumer<String, TokioExecutor>>>,
        config: ConsumerConfig,
    }

    impl PulsarVectorConsumer {
        /// Create a new Pulsar consumer (async)
        pub async fn new(config: ConsumerConfig) -> Result<Self> {
            let pulsar: Pulsar<TokioExecutor> = Pulsar::builder(&config.brokers, TokioExecutor)
                .build()
                .await
                .map_err(|e| NeedleError::InvalidOperation(format!("Pulsar connection error: {}", e)))?;

            let consumer: PulsarConsumer<String, TokioExecutor> = pulsar
                .consumer()
                .with_topic(&config.topic)
                .with_subscription(&config.group_id)
                .with_subscription_type(SubType::Shared)
                .with_options(pulsar::consumer::ConsumerOptions {
                    initial_position: InitialPosition::Earliest,
                    ..Default::default()
                })
                .build()
                .await
                .map_err(|e| NeedleError::InvalidOperation(format!("Pulsar consumer error: {}", e)))?;

            Ok(Self {
                consumer: Arc::new(Mutex::new(consumer)),
                config,
            })
        }

        /// Poll for messages (async)
        pub async fn poll(&self) -> Result<Vec<VectorMessage>> {
            let mut messages = Vec::new();
            let mut consumer = self.consumer.lock().await;

            for _ in 0..self.config.batch_size {
                match tokio::time::timeout(
                    std::time::Duration::from_millis(self.config.poll_timeout_ms),
                    consumer.try_next()
                ).await {
                    Ok(Ok(Some(msg))) => {
                        if let Ok(vec_msg) = self.parse_message(&msg) {
                            messages.push(vec_msg);
                            if !self.config.auto_commit {
                                let _ = consumer.ack(&msg).await;
                            }
                        }
                    }
                    _ => break,
                }
            }

            Ok(messages)
        }

        fn parse_message(&self, msg: &pulsar::consumer::Message<String>) -> Result<VectorMessage> {
            let payload = msg.payload.data.as_slice();
            let json: serde_json::Value = serde_json::from_slice(payload)
                .map_err(|e| NeedleError::InvalidFormat(format!("JSON parse error: {}", e)))?;

            let id = json.get(&self.config.id_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| NeedleError::InvalidFormat("Missing id field".to_string()))?
                .to_string();

            let vector: Vec<f32> = json.get(&self.config.vector_field)
                .and_then(|v| v.as_array())
                .ok_or_else(|| NeedleError::InvalidFormat("Missing vector field".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            let metadata = self.config.metadata_field.as_ref()
                .and_then(|field| json.get(field).cloned());

            Ok(VectorMessage {
                id,
                vector,
                metadata,
                offset: msg.message_id().entry_id(),
                partition: Some(msg.message_id().partition()),
                timestamp: msg.metadata().publish_time,
                key: msg.key().map(|k| k.to_string()),
            })
        }
    }
}

/// PostgreSQL CDC implementation using logical replication
#[cfg(feature = "cdc-postgres")]
pub mod postgres_cdc {
    use super::*;
    use tokio_postgres::{Client, NoTls, Row};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// PostgreSQL CDC consumer
    pub struct PostgresVectorConsumer {
        client: Arc<Mutex<Client>>,
        config: ConsumerConfig,
        last_lsn: Arc<Mutex<u64>>,
    }

    impl PostgresVectorConsumer {
        /// Create a new PostgreSQL CDC consumer
        pub async fn new(config: ConsumerConfig) -> Result<Self> {
            let (client, connection) = tokio_postgres::connect(&config.brokers, NoTls)
                .await
                .map_err(|e| NeedleError::InvalidOperation(format!("PostgreSQL connection error: {}", e)))?;

            // Spawn connection task
            tokio::spawn(async move {
                if let Err(e) = connection.await {
                    tracing::error!("PostgreSQL connection error: {}", e);
                }
            });

            Ok(Self {
                client: Arc::new(Mutex::new(client)),
                config,
                last_lsn: Arc::new(Mutex::new(0)),
            })
        }

        /// Poll for changes using a polling approach (for simplicity)
        /// In production, you'd want to use logical replication with pg_logical or wal2json
        pub async fn poll(&self) -> Result<Vec<VectorMessage>> {
            let client = self.client.lock().await;
            let last_lsn = *self.last_lsn.lock().await;

            // Query for new/updated vectors since last poll
            // This assumes a table structure with id, vector (as array), metadata (jsonb), and updated_at
            let query = format!(
                "SELECT id, vector, metadata, EXTRACT(EPOCH FROM updated_at)::bigint as ts 
                 FROM {} 
                 WHERE EXTRACT(EPOCH FROM updated_at)::bigint > $1 
                 ORDER BY updated_at ASC 
                 LIMIT $2",
                self.config.topic // topic is used as table name
            );

            let rows = client
                .query(&query, &[&(last_lsn as i64), &(self.config.batch_size as i64)])
                .await
                .map_err(|e| NeedleError::InvalidOperation(format!("PostgreSQL query error: {}", e)))?;

            let mut messages = Vec::with_capacity(rows.len());
            let mut max_ts = last_lsn;

            for row in rows {
                if let Ok(msg) = self.row_to_message(&row) {
                    max_ts = max_ts.max(msg.timestamp);
                    messages.push(msg);
                }
            }

            // Update last LSN
            if max_ts > last_lsn {
                *self.last_lsn.lock().await = max_ts;
            }

            Ok(messages)
        }

        fn row_to_message(&self, row: &Row) -> Result<VectorMessage> {
            let id: String = row.try_get("id")
                .map_err(|e| NeedleError::InvalidFormat(format!("Missing id: {}", e)))?;

            // PostgreSQL array to Vec<f32>
            let vector: Vec<f32> = row.try_get::<_, Vec<f32>>("vector")
                .or_else(|_| {
                    // Try as f64 array and convert
                    row.try_get::<_, Vec<f64>>("vector")
                        .map(|v| v.into_iter().map(|f| f as f32).collect())
                })
                .map_err(|e| NeedleError::InvalidFormat(format!("Invalid vector: {}", e)))?;

            let metadata: Option<serde_json::Value> = row.try_get("metadata").ok();
            let timestamp: i64 = row.try_get("ts").unwrap_or(0);

            Ok(VectorMessage {
                id,
                vector,
                metadata,
                offset: timestamp as u64,
                partition: None,
                timestamp: timestamp as u64,
                key: None,
            })
        }
    }
}

/// MongoDB change streams implementation
#[cfg(feature = "cdc-mongodb")]
pub mod mongodb_cdc {
    use super::*;
    use mongodb::{Client, options::ClientOptions, Collection, bson::{doc, Document}};
    use mongodb::change_stream::event::ChangeStreamEvent;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use futures::StreamExt;

    /// MongoDB change stream consumer
    pub struct MongoVectorConsumer {
        collection: Collection<Document>,
        config: ConsumerConfig,
        resume_token: Arc<Mutex<Option<mongodb::bson::Document>>>,
    }

    impl MongoVectorConsumer {
        /// Create a new MongoDB change stream consumer
        pub async fn new(config: ConsumerConfig) -> Result<Self> {
            let client_options = ClientOptions::parse(&config.brokers)
                .await
                .map_err(|e| NeedleError::InvalidOperation(format!("MongoDB connection error: {}", e)))?;

            let client = Client::with_options(client_options)
                .map_err(|e| NeedleError::InvalidOperation(format!("MongoDB client error: {}", e)))?;

            // Parse database and collection from topic (format: "database.collection")
            let parts: Vec<&str> = config.topic.split('.').collect();
            let (db_name, coll_name) = if parts.len() >= 2 {
                (parts[0], parts[1])
            } else {
                ("needle", config.topic.as_str())
            };

            let collection = client.database(db_name).collection::<Document>(coll_name);

            Ok(Self {
                collection,
                config,
                resume_token: Arc::new(Mutex::new(None)),
            })
        }

        /// Poll for changes using change streams
        pub async fn poll(&self) -> Result<Vec<VectorMessage>> {
            let resume_token = self.resume_token.lock().await.clone();

            let mut change_stream = if let Some(token) = resume_token {
                self.collection.watch()
                    .resume_after(token)
                    .await
            } else {
                self.collection.watch().await
            }
            .map_err(|e| NeedleError::InvalidOperation(format!("MongoDB change stream error: {}", e)))?;

            let mut messages = Vec::new();

            // Poll for up to batch_size changes with timeout
            let timeout = tokio::time::Duration::from_millis(self.config.poll_timeout_ms);
            let deadline = tokio::time::Instant::now() + timeout;

            while messages.len() < self.config.batch_size && tokio::time::Instant::now() < deadline {
                match tokio::time::timeout_at(deadline, change_stream.next()).await {
                    Ok(Some(Ok(event))) => {
                        // Update resume token
                        if let Some(token) = event.id.clone() {
                            *self.resume_token.lock().await = Some(token);
                        }

                        if let Ok(msg) = self.event_to_message(&event) {
                            messages.push(msg);
                        }
                    }
                    _ => break,
                }
            }

            Ok(messages)
        }

        fn event_to_message(&self, event: &ChangeStreamEvent<Document>) -> Result<VectorMessage> {
            let doc = event.full_document.as_ref()
                .ok_or_else(|| NeedleError::InvalidFormat("No full document in change event".to_string()))?;

            let id = doc.get_str("_id")
                .or_else(|_| doc.get_object_id("_id").map(|oid| oid.to_hex()))
                .map_err(|_| NeedleError::InvalidFormat("Missing _id field".to_string()))?
                .to_string();

            let vector: Vec<f32> = doc.get_array(&self.config.vector_field)
                .map_err(|_| NeedleError::InvalidFormat("Missing vector field".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            let metadata = self.config.metadata_field.as_ref().and_then(|field| {
                doc.get_document(field).ok().map(|d| {
                    serde_json::to_value(d).unwrap_or(serde_json::Value::Null)
                })
            });

            let timestamp = event.cluster_time
                .map(|t| t.time as u64)
                .unwrap_or(0);

            Ok(VectorMessage {
                id,
                vector,
                metadata,
                offset: timestamp,
                partition: None,
                timestamp,
                key: None,
            })
        }
    }
}

// Re-export feature-gated types
#[cfg(feature = "cdc-kafka")]
pub use kafka_cdc::KafkaVectorConsumer;

#[cfg(feature = "cdc-pulsar")]
pub use pulsar_cdc::PulsarVectorConsumer;

#[cfg(feature = "cdc-postgres")]
pub use postgres_cdc::PostgresVectorConsumer;

#[cfg(feature = "cdc-mongodb")]
pub use mongodb_cdc::MongoVectorConsumer;

// =============================================================================
// Advanced Stream Processing (Next-Gen)
// =============================================================================

/// Stream processing operation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOp {
    /// Filter messages
    Filter,
    /// Transform vectors
    Transform,
    /// Aggregate over window
    Aggregate,
    /// Join with another stream
    Join,
    /// Route to different outputs
    Route,
}

/// Stream processing pipeline for vector data
pub struct VectorStreamPipeline {
    stages: Vec<StreamStage>,
    error_handler: Box<dyn Fn(&VectorMessage, &str) + Send + Sync>,
    metrics: StreamMetrics,
}

struct StreamStage {
    #[allow(dead_code)]
    name: String,
    op: StreamOp,
    processor: Box<dyn Fn(VectorMessage) -> Option<VectorMessage> + Send + Sync>,
}

/// Metrics for stream processing
#[derive(Debug, Default)]
pub struct StreamMetrics {
    pub messages_processed: AtomicU64,
    pub messages_filtered: AtomicU64,
    pub messages_transformed: AtomicU64,
    pub messages_errored: AtomicU64,
    pub processing_time_us: AtomicU64,
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_processed(&self) {
        self.messages_processed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_filtered(&self) {
        self.messages_filtered.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_transformed(&self) {
        self.messages_transformed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.messages_errored.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_processing_time(&self, micros: u64) {
        self.processing_time_us.fetch_add(micros, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StreamMetricsSnapshot {
        StreamMetricsSnapshot {
            messages_processed: self.messages_processed.load(Ordering::Relaxed),
            messages_filtered: self.messages_filtered.load(Ordering::Relaxed),
            messages_transformed: self.messages_transformed.load(Ordering::Relaxed),
            messages_errored: self.messages_errored.load(Ordering::Relaxed),
            total_processing_time_us: self.processing_time_us.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetricsSnapshot {
    pub messages_processed: u64,
    pub messages_filtered: u64,
    pub messages_transformed: u64,
    pub messages_errored: u64,
    pub total_processing_time_us: u64,
}

impl VectorStreamPipeline {
    /// Create a new stream pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            error_handler: Box::new(|_msg, _err| {}),
            metrics: StreamMetrics::new(),
        }
    }

    /// Add a filter stage
    pub fn filter<F>(mut self, name: &str, predicate: F) -> Self
    where
        F: Fn(&VectorMessage) -> bool + Send + Sync + 'static,
    {
        self.stages.push(StreamStage {
            name: name.to_string(),
            op: StreamOp::Filter,
            processor: Box::new(move |msg| {
                if predicate(&msg) {
                    Some(msg)
                } else {
                    None
                }
            }),
        });
        self
    }

    /// Add a transform stage for vectors
    pub fn transform<F>(mut self, name: &str, transformer: F) -> Self
    where
        F: Fn(Vec<f32>) -> Vec<f32> + Send + Sync + 'static,
    {
        self.stages.push(StreamStage {
            name: name.to_string(),
            op: StreamOp::Transform,
            processor: Box::new(move |mut msg| {
                msg.vector = transformer(msg.vector);
                Some(msg)
            }),
        });
        self
    }

    /// Add a metadata transform stage
    pub fn map_metadata<F>(mut self, name: &str, mapper: F) -> Self
    where
        F: Fn(Option<Value>) -> Option<Value> + Send + Sync + 'static,
    {
        self.stages.push(StreamStage {
            name: name.to_string(),
            op: StreamOp::Transform,
            processor: Box::new(move |mut msg| {
                msg.metadata = mapper(msg.metadata);
                Some(msg)
            }),
        });
        self
    }

    /// Set error handler
    pub fn on_error<F>(mut self, handler: F) -> Self
    where
        F: Fn(&VectorMessage, &str) + Send + Sync + 'static,
    {
        self.error_handler = Box::new(handler);
        self
    }

    /// Process a single message through the pipeline
    pub fn process(&self, mut message: VectorMessage) -> Option<VectorMessage> {
        let start = Instant::now();

        for stage in &self.stages {
            match (stage.processor)(message.clone()) {
                Some(processed) => {
                    message = processed;
                    match stage.op {
                        StreamOp::Filter => self.metrics.record_filtered(),
                        StreamOp::Transform => self.metrics.record_transformed(),
                        _ => {}
                    }
                }
                None => {
                    self.metrics.record_filtered();
                    return None;
                }
            }
        }

        self.metrics.record_processed();
        self.metrics
            .record_processing_time(start.elapsed().as_micros() as u64);

        Some(message)
    }

    /// Process a batch of messages
    pub fn process_batch(&self, messages: Vec<VectorMessage>) -> Vec<VectorMessage> {
        messages.into_iter().filter_map(|m| self.process(m)).collect()
    }

    /// Get metrics snapshot
    pub fn metrics(&self) -> StreamMetricsSnapshot {
        self.metrics.snapshot()
    }
}

impl Default for VectorStreamPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// CDC event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdcEventType {
    /// Insert/create
    Insert,
    /// Update
    Update,
    /// Delete
    Delete,
    /// Snapshot (initial load)
    Snapshot,
}

/// CDC event for vector changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcEvent {
    /// Event type
    pub event_type: CdcEventType,
    /// Vector ID
    pub id: String,
    /// Collection name
    pub collection: String,
    /// New vector value (None for deletes)
    pub vector: Option<Vec<f32>>,
    /// Previous vector value (for updates)
    pub previous_vector: Option<Vec<f32>>,
    /// Metadata
    pub metadata: Option<Value>,
    /// Timestamp of the change
    pub timestamp: u64,
    /// Source transaction ID
    pub transaction_id: Option<String>,
    /// Log sequence number
    pub lsn: Option<u64>,
}

/// CDC stream for change data capture
pub struct CdcStream {
    events: RwLock<VecDeque<CdcEvent>>,
    subscribers: RwLock<Vec<Box<dyn Fn(&CdcEvent) + Send + Sync>>>,
    last_lsn: AtomicU64,
    enabled: AtomicBool,
}

impl CdcStream {
    /// Create a new CDC stream
    pub fn new() -> Self {
        Self {
            events: RwLock::new(VecDeque::new()),
            subscribers: RwLock::new(Vec::new()),
            last_lsn: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Publish a CDC event
    pub fn publish(&self, event: CdcEvent) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Update LSN
        if let Some(lsn) = event.lsn {
            self.last_lsn.fetch_max(lsn, Ordering::Relaxed);
        }

        // Store event
        self.events.write().push_back(event.clone());

        // Notify subscribers
        for subscriber in self.subscribers.read().iter() {
            subscriber(&event);
        }
    }

    /// Subscribe to CDC events
    pub fn subscribe<F>(&self, callback: F)
    where
        F: Fn(&CdcEvent) + Send + Sync + 'static,
    {
        self.subscribers.write().push(Box::new(callback));
    }

    /// Get events since LSN
    pub fn get_events_since(&self, lsn: u64) -> Vec<CdcEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.lsn.unwrap_or(0) > lsn)
            .cloned()
            .collect()
    }

    /// Get last LSN
    pub fn last_lsn(&self) -> u64 {
        self.last_lsn.load(Ordering::Relaxed)
    }

    /// Enable/disable CDC
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Clear old events (retention)
    pub fn prune(&self, keep_count: usize) {
        let mut events = self.events.write();
        while events.len() > keep_count {
            events.pop_front();
        }
    }

    /// Get pending event count
    pub fn pending_count(&self) -> usize {
        self.events.read().len()
    }
}

impl Default for CdcStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Replay manager for stream recovery
pub struct ReplayManager {
    snapshots: RwLock<HashMap<String, StreamSnapshot>>,
    retention_seconds: u64,
}

/// Stream snapshot for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSnapshot {
    /// Snapshot ID
    pub id: String,
    /// Consumer group
    pub group_id: String,
    /// Topic/stream name
    pub topic: String,
    /// Partition offsets at snapshot time
    pub offsets: HashMap<u32, u64>,
    /// Watermark at snapshot time
    pub watermark: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Processed message IDs (for dedup)
    pub processed_ids: Vec<String>,
}

impl ReplayManager {
    /// Create a new replay manager
    pub fn new(retention_seconds: u64) -> Self {
        Self {
            snapshots: RwLock::new(HashMap::new()),
            retention_seconds,
        }
    }

    /// Take a snapshot
    pub fn take_snapshot(
        &self,
        group_id: &str,
        topic: &str,
        offsets: HashMap<u32, u64>,
        watermark: u64,
        processed_ids: Vec<String>,
    ) -> String {
        let id = format!(
            "{}-{}-{}",
            group_id,
            topic,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let snapshot = StreamSnapshot {
            id: id.clone(),
            group_id: group_id.to_string(),
            topic: topic.to_string(),
            offsets,
            watermark,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            processed_ids,
        };

        self.snapshots.write().insert(id.clone(), snapshot);
        id
    }

    /// Get a snapshot
    pub fn get_snapshot(&self, id: &str) -> Option<StreamSnapshot> {
        self.snapshots.read().get(id).cloned()
    }

    /// List snapshots for a group/topic
    pub fn list_snapshots(&self, group_id: &str, topic: &str) -> Vec<StreamSnapshot> {
        self.snapshots
            .read()
            .values()
            .filter(|s| s.group_id == group_id && s.topic == topic)
            .cloned()
            .collect()
    }

    /// Clean up old snapshots
    pub fn cleanup(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.snapshots
            .write()
            .retain(|_, s| now - s.timestamp < self.retention_seconds);
    }
}

// ---------------------------------------------------------------------------
// Backpressure Controller
// ---------------------------------------------------------------------------

/// Backpressure state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressureState {
    /// Normal throughput
    Flowing,
    /// Reduced throughput — consumer should slow down
    Throttled,
    /// Consumer should stop polling until pressure eases
    Paused,
}

/// Configuration for the backpressure controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Maximum in-flight (un-committed) messages before throttling
    pub max_in_flight: usize,
    /// Maximum in-flight before pausing entirely
    pub max_in_flight_pause: usize,
    /// Percentage of collection capacity that triggers throttling (0..100)
    pub capacity_throttle_pct: f64,
    /// Percentage of collection capacity that triggers pause (0..100)
    pub capacity_pause_pct: f64,
    /// When throttled, reduce batch size by this factor
    pub throttle_factor: f64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 10_000,
            max_in_flight_pause: 50_000,
            capacity_throttle_pct: 80.0,
            capacity_pause_pct: 95.0,
            throttle_factor: 0.5,
        }
    }
}

/// Controls backpressure between a streaming source and Needle ingestion.
pub struct BackpressureController {
    config: BackpressureConfig,
    in_flight: AtomicU64,
    state: RwLock<BackpressureState>,
    throttle_count: AtomicU64,
    pause_count: AtomicU64,
}

impl BackpressureController {
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            in_flight: AtomicU64::new(0),
            state: RwLock::new(BackpressureState::Flowing),
            throttle_count: AtomicU64::new(0),
            pause_count: AtomicU64::new(0),
        }
    }

    /// Notify the controller that messages were received.
    pub fn on_receive(&self, count: u64) {
        self.in_flight.fetch_add(count, Ordering::Relaxed);
        self.evaluate();
    }

    /// Notify the controller that messages were committed/acked.
    pub fn on_commit(&self, count: u64) {
        let prev = self.in_flight.load(Ordering::Relaxed);
        self.in_flight.store(prev.saturating_sub(count), Ordering::Relaxed);
        self.evaluate();
    }

    /// Evaluate the current state based on in-flight count.
    fn evaluate(&self) {
        let in_flight = self.in_flight.load(Ordering::Relaxed) as usize;
        let mut state = self.state.write();

        if in_flight >= self.config.max_in_flight_pause {
            if *state != BackpressureState::Paused {
                self.pause_count.fetch_add(1, Ordering::Relaxed);
                info!(in_flight, "Backpressure: pausing consumer");
            }
            *state = BackpressureState::Paused;
        } else if in_flight >= self.config.max_in_flight {
            if *state == BackpressureState::Flowing {
                self.throttle_count.fetch_add(1, Ordering::Relaxed);
                debug!(in_flight, "Backpressure: throttling consumer");
            }
            *state = BackpressureState::Throttled;
        } else {
            *state = BackpressureState::Flowing;
        }
    }

    /// Evaluate based on collection capacity (vector count vs limit).
    pub fn evaluate_capacity(&self, current_vectors: u64, max_vectors: u64) {
        if max_vectors == 0 {
            return;
        }
        let pct = (current_vectors as f64 / max_vectors as f64) * 100.0;
        let mut state = self.state.write();

        if pct >= self.config.capacity_pause_pct {
            if *state != BackpressureState::Paused {
                self.pause_count.fetch_add(1, Ordering::Relaxed);
            }
            *state = BackpressureState::Paused;
        } else if pct >= self.config.capacity_throttle_pct {
            if *state == BackpressureState::Flowing {
                self.throttle_count.fetch_add(1, Ordering::Relaxed);
            }
            *state = BackpressureState::Throttled;
        }
    }

    /// Current backpressure state.
    pub fn state(&self) -> BackpressureState {
        *self.state.read()
    }

    /// Get the effective batch size given current pressure.
    pub fn effective_batch_size(&self, base_batch_size: usize) -> usize {
        match self.state() {
            BackpressureState::Flowing => base_batch_size,
            BackpressureState::Throttled => {
                (base_batch_size as f64 * self.config.throttle_factor).max(1.0) as usize
            }
            BackpressureState::Paused => 0,
        }
    }

    /// Returns true if the consumer should poll.
    pub fn should_poll(&self) -> bool {
        self.state() != BackpressureState::Paused
    }

    /// Backpressure statistics.
    pub fn stats(&self) -> BackpressureStats {
        BackpressureStats {
            state: self.state(),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            throttle_events: self.throttle_count.load(Ordering::Relaxed),
            pause_events: self.pause_count.load(Ordering::Relaxed),
        }
    }
}

/// Statistics about backpressure events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureStats {
    pub state: BackpressureState,
    pub in_flight: u64,
    pub throttle_events: u64,
    pub pause_events: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consumer_config() {
        let config = ConsumerConfig::kafka("localhost:9092", "test-topic")
            .group_id("test-group")
            .batch_size(50)
            .exactly_once(true);

        assert_eq!(config.source, MessageSource::Kafka);
        assert_eq!(config.brokers, "localhost:9092");
        assert_eq!(config.topic, "test-topic");
        assert_eq!(config.group_id, "test-group");
        assert_eq!(config.batch_size, 50);
        assert!(config.exactly_once);
    }

    #[test]
    fn test_mock_consumer() {
        let config = ConsumerConfig::mock();
        let consumer = VectorConsumer::new(config).unwrap();
        consumer.start().unwrap();

        // Add mock messages
        let messages = vec![
            VectorMessage {
                id: "v1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v2".to_string(),
                vector: vec![4.0, 5.0, 6.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        // Poll
        let batch = consumer.poll().unwrap().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, "v1");
        assert_eq!(batch[1].id, "v2");

        consumer.stop().unwrap();
    }

    #[test]
    fn test_deduplication() {
        let config = ConsumerConfig::mock().deduplication(true);
        let consumer = VectorConsumer::new(config).unwrap();
        consumer.start().unwrap();

        // Add duplicate messages
        let messages = vec![
            VectorMessage {
                id: "v1".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v1".to_string(), // Duplicate
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v2".to_string(),
                vector: vec![3.0],
                metadata: None,
                offset: 2,
                partition: None,
                timestamp: 0,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        let batch = consumer.poll().unwrap().unwrap();
        assert_eq!(batch.len(), 2); // Duplicate filtered
        assert_eq!(batch[0].id, "v1");
        assert_eq!(batch[1].id, "v2");

        let stats = consumer.stats();
        assert_eq!(stats.duplicates_filtered, 1);
    }

    #[test]
    fn test_json_parsing() {
        let config = ConsumerConfig::mock();
        let consumer = VectorConsumer::new(config).unwrap();

        let json = r#"{"id": "vec1", "vector": [1.0, 2.0, 3.0], "metadata": {"key": "value"}}"#;
        let msg = consumer.parse_json_message(json, 0).unwrap();

        assert_eq!(msg.id, "vec1");
        assert_eq!(msg.vector, vec![1.0, 2.0, 3.0]);
        assert!(msg.metadata.is_some());
    }

    #[test]
    fn test_binary_parsing() {
        let config = ConsumerConfig::mock().vector_format(VectorFormat::BinaryF32LE);
        let consumer = VectorConsumer::new(config).unwrap();

        // Create binary data for [1.0, 2.0]
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());

        let vector = consumer.parse_binary_vector(&data).unwrap();
        assert_eq!(vector, vec![1.0, 2.0]);
    }

    #[test]
    fn test_producer() {
        let config = ProducerConfig {
            source: MessageSource::Mock,
            ..Default::default()
        };
        let producer = VectorProducer::new(config).unwrap();

        producer.send("v1", &[1.0, 2.0], None).unwrap();
        producer.send("v2", &[3.0, 4.0], Some(serde_json::json!({"key": "value"}))).unwrap();

        let output = producer.get_mock_output();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].id, "v1");
        assert_eq!(output[1].id, "v2");
    }

    #[test]
    fn test_offset_commit() {
        let config = ConsumerConfig::mock();
        let consumer = VectorConsumer::new(config).unwrap();

        consumer.commit_offset(0, 100).unwrap();
        consumer.commit_offset(1, 200).unwrap();

        let state = consumer.offset_state();
        assert_eq!(state.committed.get(&0), Some(&100));
        assert_eq!(state.committed.get(&1), Some(&200));
    }

    // Exactly-once enhancement tests

    #[test]
    fn test_checkpoint() {
        let mut checkpoint = Checkpoint::new("test-group", "test-topic");

        checkpoint.update_offset(0, 100);
        checkpoint.update_offset(1, 200);
        checkpoint.mark_processed("msg-1");
        checkpoint.mark_processed("msg-2");
        checkpoint.update_watermark(1000);

        assert_eq!(checkpoint.offsets.get(&0), Some(&100));
        assert_eq!(checkpoint.offsets.get(&1), Some(&200));
        assert!(checkpoint.is_processed("msg-1"));
        assert!(checkpoint.is_processed("msg-2"));
        assert!(!checkpoint.is_processed("msg-3"));
        assert_eq!(checkpoint.watermark, 1000);
    }

    #[test]
    fn test_in_memory_checkpoint_store() {
        let store = InMemoryCheckpointStore::new();

        let mut checkpoint = Checkpoint::new("group1", "topic1");
        checkpoint.update_offset(0, 500);

        store.save(&checkpoint).unwrap();

        let loaded = store.load("group1", "topic1").unwrap().unwrap();
        assert_eq!(loaded.group_id, "group1");
        assert_eq!(loaded.offsets.get(&0), Some(&500));

        store.delete("group1", "topic1").unwrap();
        assert!(store.load("group1", "topic1").unwrap().is_none());
    }

    #[test]
    fn test_dead_letter_queue() {
        let dlq = DeadLetterQueue::new(100);

        let msg = VectorMessage {
            id: "failed-msg".to_string(),
            vector: vec![1.0, 2.0],
            metadata: None,
            offset: 0,
            partition: None,
            timestamp: 0,
            key: None,
        };

        dlq.push(msg.clone(), "Processing failed", 1);
        assert_eq!(dlq.len(), 1);

        let stats = dlq.stats();
        assert_eq!(stats.total_failed, 1);
        assert_eq!(stats.queue_size, 1);

        let popped = dlq.pop().unwrap();
        assert_eq!(popped.message.id, "failed-msg");
        assert_eq!(popped.error, "Processing failed");
        assert_eq!(popped.retry_count, 1);

        assert!(dlq.is_empty());
    }

    #[test]
    fn test_watermark_tracker() {
        let tracker = WatermarkTracker::new(1000);

        tracker.update(0, 5000);
        tracker.update(1, 3000);
        tracker.update(2, 7000);

        // Global watermark should be minimum
        assert_eq!(tracker.watermark(), 3000);

        // Partition watermarks
        assert_eq!(tracker.partition_watermark(0), Some(5000));
        assert_eq!(tracker.partition_watermark(1), Some(3000));
        assert_eq!(tracker.partition_watermark(2), Some(7000));

        // Late message check (with 1000ms allowed lateness)
        assert!(tracker.is_late(1000)); // Too late
        assert!(!tracker.is_late(2500)); // Within lateness window
        assert!(!tracker.is_late(4000)); // Ahead of watermark
    }

    #[test]
    fn test_batch_result() {
        let result = BatchResult {
            processed: 10,
            skipped: 2,
            failed: 1,
        };

        assert_eq!(result.total(), 13);
    }

    #[test]
    fn test_transactional_processor() {
        let config = ConsumerConfig::mock();
        let consumer = Arc::new(VectorConsumer::new(config).unwrap());
        let store = Arc::new(InMemoryCheckpointStore::new());

        let processed_ids = Arc::new(RwLock::new(Vec::new()));
        let ids_clone = processed_ids.clone();

        let processor = TransactionalProcessor::new(
            consumer.clone(),
            store,
            move |msg: &VectorMessage| {
                ids_clone.write().push(msg.id.clone());
                Ok(())
            },
        )
        .unwrap()
        .with_max_retries(3)
        .with_checkpoint_interval(10);

        consumer.start().unwrap();

        // Add messages
        let messages = vec![
            VectorMessage {
                id: "t1".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: Some(0),
                timestamp: 1000,
                key: None,
            },
            VectorMessage {
                id: "t2".to_string(),
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: Some(0),
                timestamp: 2000,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        // Process
        let result = processor.process_batch().unwrap();
        assert_eq!(result.processed, 2);
        assert_eq!(result.failed, 0);

        // Check idempotency - same messages should be skipped
        let checkpoint = processor.checkpoint();
        assert!(checkpoint.is_processed("t1"));
        assert!(checkpoint.is_processed("t2"));
    }

    #[test]
    fn test_backpressure_flowing() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 100,
            max_in_flight_pause: 500,
            ..Default::default()
        });
        assert_eq!(controller.state(), BackpressureState::Flowing);
        assert!(controller.should_poll());
        assert_eq!(controller.effective_batch_size(50), 50);
    }

    #[test]
    fn test_backpressure_throttle() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 100,
            max_in_flight_pause: 500,
            throttle_factor: 0.25,
            ..Default::default()
        });
        controller.on_receive(150);
        assert_eq!(controller.state(), BackpressureState::Throttled);
        assert!(controller.should_poll());
        assert_eq!(controller.effective_batch_size(100), 25);
    }

    #[test]
    fn test_backpressure_pause_and_recover() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 100,
            max_in_flight_pause: 500,
            ..Default::default()
        });
        controller.on_receive(600);
        assert_eq!(controller.state(), BackpressureState::Paused);
        assert!(!controller.should_poll());
        assert_eq!(controller.effective_batch_size(100), 0);

        // Commit enough to recover
        controller.on_commit(550);
        assert_eq!(controller.state(), BackpressureState::Flowing);
        assert!(controller.should_poll());
    }

    #[test]
    fn test_backpressure_capacity() {
        let controller = BackpressureController::new(BackpressureConfig {
            capacity_throttle_pct: 80.0,
            capacity_pause_pct: 95.0,
            ..Default::default()
        });
        controller.evaluate_capacity(85_000, 100_000);
        assert_eq!(controller.state(), BackpressureState::Throttled);

        controller.evaluate_capacity(96_000, 100_000);
        assert_eq!(controller.state(), BackpressureState::Paused);
    }
}
