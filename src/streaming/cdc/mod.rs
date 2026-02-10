pub mod debezium;
pub mod kafka;
pub mod mongodb;
pub mod postgres;
pub mod pulsar;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;

use super::core::{
    ChangeEvent, StreamError, StreamResult, current_timestamp_millis,
};
use super::stream_manager::StreamManager;

pub use debezium::{DebeziumParser, DebeziumSourceType};
pub use kafka::{KafkaConnector, KafkaConnectorConfig};
pub use mongodb::{MongoCdcConfig, MongoCdcConnector};
pub use postgres::{PostgresCdcConfig, PostgresCdcConnector};
pub use pulsar::{PulsarConnector, PulsarConnectorConfig, PulsarSubscriptionPosition};

// ============================================================================
// CDC Connector Trait
// ============================================================================

/// CDC connector trait for database change capture
#[allow(async_fn_in_trait)]
pub trait CdcConnector: Send + Sync {
    /// Connect to the data source
    async fn connect(&mut self) -> StreamResult<()>;

    /// Start capturing changes
    async fn start_capture(&mut self) -> StreamResult<()>;

    /// Stop capturing changes
    async fn stop_capture(&mut self) -> StreamResult<()>;

    /// Get the next change event (blocking)
    async fn next_change(&mut self) -> StreamResult<Option<ChangeEvent>>;

    /// Get current position/offset for checkpointing
    fn current_position(&self) -> CdcPosition;

    /// Resume from a specific position
    async fn seek(&mut self, position: &CdcPosition) -> StreamResult<()>;

    /// Check if connector is connected
    fn is_connected(&self) -> bool;

    /// Get connector statistics
    fn stats(&self) -> CdcConnectorStats;
}

// ============================================================================
// CDC Position
// ============================================================================

/// CDC position for checkpointing and resumption
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CdcPosition {
    /// Source-specific position identifier
    pub position: String,
    /// Timestamp when this position was recorded
    pub timestamp: u64,
    /// Source identifier (topic, table, etc.)
    pub source: String,
    /// Partition/shard identifier (if applicable)
    pub partition: Option<i32>,
}

impl CdcPosition {
    /// Create a new CDC position
    pub fn new(position: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            position: position.into(),
            timestamp: current_timestamp_millis(),
            source: source.into(),
            partition: None,
        }
    }

    /// With partition
    pub fn with_partition(mut self, partition: i32) -> Self {
        self.partition = Some(partition);
        self
    }

    /// Serialize to string for storage/transmission
    pub fn serialize(&self) -> String {
        if let Some(partition) = self.partition {
            format!(
                "{}:{}:{}:{}",
                self.source, partition, self.position, self.timestamp
            )
        } else {
            format!("{}::{}:{}", self.source, self.position, self.timestamp)
        }
    }
    /// Parse from string
    pub fn parse(s: &str) -> StreamResult<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 4 {
            return Err(StreamError::InvalidResumeToken(format!(
                "Invalid CDC position format: {}",
                s
            )));
        }

        let source = parts[0].to_string();
        let partition = if parts[1].is_empty() {
            None
        } else {
            parts[1]
                .parse::<i32>()
                .map(Some)
                .map_err(|_| StreamError::InvalidResumeToken("Invalid partition".to_string()))?
        };
        let position = parts[2].to_string();
        let timestamp = parts[3]
            .parse::<u64>()
            .map_err(|_| StreamError::InvalidResumeToken("Invalid timestamp".to_string()))?;

        Ok(Self {
            position,
            timestamp,
            source,
            partition,
        })
    }
}

impl std::fmt::Display for CdcPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.serialize())
    }
}

// ============================================================================
// CDC Connector Stats
// ============================================================================

/// Statistics for CDC connectors
#[derive(Debug, Clone, Default)]
pub struct CdcConnectorStats {
    /// Total messages received
    pub messages_received: u64,
    /// Messages processed successfully
    pub messages_processed: u64,
    /// Messages that failed processing
    pub messages_failed: u64,
    /// Current lag (messages behind)
    pub lag: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Average processing latency in milliseconds
    pub avg_latency_ms: f64,
    /// Last error message (if any)
    pub last_error: Option<String>,
    /// Connection uptime in seconds
    pub uptime_secs: u64,
}

// ============================================================================
// CDC Config
// ============================================================================

/// Configuration for CDC connectors
#[derive(Debug, Clone)]
pub struct CdcConfig {
    /// Batch size for fetching messages
    pub batch_size: usize,
    /// Timeout for fetch operations in milliseconds
    pub fetch_timeout_ms: u64,
    /// Auto-commit interval in milliseconds (0 to disable)
    pub auto_commit_interval_ms: u64,
    /// Maximum retries on failure
    pub max_retries: u32,
    /// Retry backoff base in milliseconds
    pub retry_backoff_ms: u64,
    /// Enable exactly-once semantics (if supported)
    pub exactly_once: bool,
    /// Dead letter queue topic/collection (if supported)
    pub dlq_destination: Option<String>,
}

impl Default for CdcConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            fetch_timeout_ms: 5000,
            auto_commit_interval_ms: 5000,
            max_retries: 3,
            retry_backoff_ms: 1000,
            exactly_once: false,
            dlq_destination: None,
        }
    }
}

// ============================================================================
// CDC Ingestion Pipeline
// ============================================================================

/// Pipeline for ingesting CDC events into Needle
pub struct CdcIngestionPipeline {
    /// Stream manager for recording events
    stream_manager: Arc<StreamManager>,
    /// Transformation function
    transformer: Option<Box<dyn Fn(ChangeEvent) -> Option<ChangeEvent> + Send + Sync>>,
    /// Checkpoint interval (events)
    checkpoint_interval: usize,
    /// Events since last checkpoint
    events_since_checkpoint: AtomicU64,
    /// Last checkpoint position
    last_checkpoint: RwLock<Option<CdcPosition>>,
    /// Pipeline statistics
    stats: RwLock<CdcPipelineStats>,
}

/// Statistics for the CDC pipeline
#[derive(Debug, Clone, Default)]
pub struct CdcPipelineStats {
    /// Total events ingested
    pub events_ingested: u64,
    /// Events transformed
    pub events_transformed: u64,
    /// Events filtered out
    pub events_filtered: u64,
    /// Last checkpoint time
    pub last_checkpoint_time: Option<u64>,
    /// Errors encountered
    pub errors: u64,
}

impl CdcIngestionPipeline {
    /// Create a new ingestion pipeline
    pub fn new(stream_manager: Arc<StreamManager>) -> Self {
        Self {
            stream_manager,
            transformer: None,
            checkpoint_interval: 1000,
            events_since_checkpoint: AtomicU64::new(0),
            last_checkpoint: RwLock::new(None),
            stats: RwLock::new(CdcPipelineStats::default()),
        }
    }

    /// Set a transformation function
    pub fn with_transformer<F>(mut self, f: F) -> Self
    where
        F: Fn(ChangeEvent) -> Option<ChangeEvent> + Send + Sync + 'static,
    {
        self.transformer = Some(Box::new(f));
        self
    }

    /// Set checkpoint interval
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Ingest a single event
    pub async fn ingest(&self, event: ChangeEvent) -> StreamResult<Option<u64>> {
        let mut stats = self.stats.write().await;

        // Apply transformation
        let event = if let Some(ref transformer) = self.transformer {
            match transformer(event) {
                Some(e) => {
                    stats.events_transformed += 1;
                    e
                }
                None => {
                    stats.events_filtered += 1;
                    return Ok(None);
                }
            }
        } else {
            event
        };

        // Record the event
        match self.stream_manager.record_change(event).await {
            Ok(position) => {
                stats.events_ingested += 1;
                self.events_since_checkpoint.fetch_add(1, Ordering::Relaxed);
                Ok(Some(position))
            }
            Err(e) => {
                stats.errors += 1;
                Err(e)
            }
        }
    }

    /// Ingest events from a CDC connector
    pub async fn ingest_from_connector<C: CdcConnector>(
        &self,
        connector: &mut C,
    ) -> StreamResult<usize> {
        let mut ingested = 0;

        loop {
            match connector.next_change().await {
                Ok(Some(event)) => {
                    self.ingest(event).await?;
                    ingested += 1;

                    // Check if we need to checkpoint
                    let events_since = self.events_since_checkpoint.load(Ordering::Relaxed);
                    if events_since >= self.checkpoint_interval as u64 {
                        self.checkpoint(connector.current_position()).await?;
                    }
                }
                Ok(None) => break,
                Err(StreamError::Timeout) => break,
                Err(e) => return Err(e),
            }
        }

        Ok(ingested)
    }

    /// Create a checkpoint
    pub async fn checkpoint(&self, position: CdcPosition) -> StreamResult<()> {
        *self.last_checkpoint.write().await = Some(position);
        self.events_since_checkpoint.store(0, Ordering::Relaxed);

        let mut stats = self.stats.write().await;
        stats.last_checkpoint_time = Some(current_timestamp_millis());

        Ok(())
    }

    /// Get the last checkpoint position
    pub async fn last_checkpoint(&self) -> Option<CdcPosition> {
        self.last_checkpoint.read().await.clone()
    }

    /// Get pipeline statistics
    pub async fn stats(&self) -> CdcPipelineStats {
        self.stats.read().await.clone()
    }
}
