#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::debug;

use crate::error::Result;

use super::consumer::{MessageSource, VectorFormat, VectorMessage};
use super::current_timestamp;

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
    pub fn send_batch(&self, vectors: &[(String, Vec<f32>, Option<Value>)]) -> Result<usize> {
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
