#![allow(clippy::unwrap_used)]

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, info};

use super::{base64_decode, current_timestamp};

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
    #[must_use]
    pub fn group_id(mut self, group_id: &str) -> Self {
        self.group_id = group_id.to_string();
        self
    }

    /// Set batch size
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set auto-commit behavior
    #[must_use]
    pub fn auto_commit(mut self, enabled: bool) -> Self {
        self.auto_commit = enabled;
        self
    }

    /// Set vector format
    #[must_use]
    pub fn vector_format(mut self, format: VectorFormat) -> Self {
        self.vector_format = format;
        self
    }

    /// Enable exactly-once semantics
    #[must_use]
    pub fn exactly_once(mut self, enabled: bool) -> Self {
        self.exactly_once = enabled;
        self
    }

    /// Enable deduplication
    #[must_use]
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
    pub(crate) config: ConsumerConfig,
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
            MessageSource::Kafka => Self::poll_kafka()?,
            MessageSource::Pulsar => Self::poll_pulsar()?,
            MessageSource::Postgres => Self::poll_postgres()?,
            MessageSource::MongoDB => Self::poll_mongodb()?,
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
            stats
                .messages_consumed
                .fetch_add(messages.len() as u64, Ordering::Relaxed);
            let bytes: usize = messages.iter().map(|m| m.vector.len() * 4).sum();
            stats
                .bytes_consumed
                .fetch_add(bytes as u64, Ordering::Relaxed);
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
    fn poll_kafka() -> Result<Vec<VectorMessage>> {
        debug!("Kafka poll (stub - enable cdc-kafka feature for real implementation)");
        Ok(Vec::new())
    }

    /// Poll from Pulsar
    #[cfg(not(feature = "cdc-pulsar"))]
    fn poll_pulsar() -> Result<Vec<VectorMessage>> {
        debug!("Pulsar poll (stub - enable cdc-pulsar feature for real implementation)");
        Ok(Vec::new())
    }

    /// Poll from PostgreSQL CDC
    #[cfg(not(feature = "cdc-postgres"))]
    fn poll_postgres() -> Result<Vec<VectorMessage>> {
        debug!("PostgreSQL CDC poll (stub - enable cdc-postgres feature for real implementation)");
        Ok(Vec::new())
    }

    /// Poll from MongoDB change streams
    #[cfg(not(feature = "cdc-mongodb"))]
    fn poll_mongodb() -> Result<Vec<VectorMessage>> {
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
                Self::parse_binary_vector_bytes(&decoded, true)
            }
            _ => Err(NeedleError::InvalidInput(
                "Unsupported binary format".to_string(),
            )),
        }
    }

    fn parse_binary_vector_bytes(data: &[u8], little_endian: bool) -> Result<Vec<f32>> {
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
            (
                messages as f64 / uptime as f64,
                bytes as f64 / uptime as f64,
            )
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
