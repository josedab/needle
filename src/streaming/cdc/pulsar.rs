use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use super::{
    CdcConfig, CdcConnector, CdcConnectorStats, CdcPosition,
};
use super::debezium::DebeziumParser;
use crate::streaming::core::{ChangeEvent, StreamError, StreamResult};

// ============================================================================
// Pulsar Connector Config
// ============================================================================

/// Apache Pulsar connector configuration
#[derive(Debug, Clone)]
pub struct PulsarConnectorConfig {
    /// Pulsar service URL
    pub service_url: String,
    /// Topic to consume from
    pub topic: String,
    /// Subscription name
    pub subscription: String,
    /// Consumer name
    pub consumer_name: String,
    /// Batch receive settings
    pub batch_size: usize,
    /// Receive timeout in milliseconds
    pub receive_timeout_ms: u64,
    /// General CDC config
    pub cdc_config: CdcConfig,
    /// Initial subscription position
    pub subscription_initial_position: PulsarSubscriptionPosition,
    /// Enable dead letter queue
    pub enable_dead_letter: bool,
    /// Dead letter topic (if enabled)
    pub dead_letter_topic: Option<String>,
    /// Max redelivery count before dead letter
    pub max_redelivery_count: u32,
}

/// Pulsar subscription initial position
#[derive(Debug, Clone, Copy, Default)]
pub enum PulsarSubscriptionPosition {
    #[default]
    Latest,
    Earliest,
}

impl Default for PulsarConnectorConfig {
    fn default() -> Self {
        Self {
            service_url: "pulsar://localhost:6650".to_string(),
            topic: "persistent://public/default/needle-cdc".to_string(),
            subscription: "needle-cdc-subscription".to_string(),
            consumer_name: "needle-cdc-consumer".to_string(),
            batch_size: 100,
            receive_timeout_ms: 5000,
            cdc_config: CdcConfig::default(),
            subscription_initial_position: PulsarSubscriptionPosition::Latest,
            enable_dead_letter: false,
            dead_letter_topic: None,
            max_redelivery_count: 3,
        }
    }
}

impl PulsarConnectorConfig {
    /// Create a new Pulsar connector config with custom settings
    pub fn new(service_url: impl Into<String>, topic: impl Into<String>) -> Self {
        Self {
            service_url: service_url.into(),
            topic: topic.into(),
            ..Default::default()
        }
    }

    /// Set subscription name
    pub fn with_subscription(mut self, subscription: impl Into<String>) -> Self {
        self.subscription = subscription.into();
        self
    }

    /// Set consumer name
    pub fn with_consumer_name(mut self, name: impl Into<String>) -> Self {
        self.consumer_name = name.into();
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set initial subscription position
    pub fn with_initial_position(mut self, position: PulsarSubscriptionPosition) -> Self {
        self.subscription_initial_position = position;
        self
    }

    /// Enable dead letter queue
    pub fn with_dead_letter(mut self, topic: impl Into<String>, max_redelivery: u32) -> Self {
        self.enable_dead_letter = true;
        self.dead_letter_topic = Some(topic.into());
        self.max_redelivery_count = max_redelivery;
        self
    }
}

// ============================================================================
// Pulsar Connector (feature-gated)
// ============================================================================

/// Apache Pulsar CDC connector
#[cfg(feature = "cdc-pulsar")]
pub struct PulsarConnector {
    config: PulsarConnectorConfig,
    client: Option<::pulsar::Pulsar<::pulsar::TokioExecutor>>,
    consumer: Option<::pulsar::Consumer<Vec<u8>, ::pulsar::TokioExecutor>>,
    connected: Arc<AtomicBool>,
    stats: Arc<RwLock<CdcConnectorStats>>,
    parser: DebeziumParser,
    current_message_id: Arc<RwLock<Option<String>>>,
}

#[cfg(feature = "cdc-pulsar")]
impl PulsarConnector {
    /// Create a new Pulsar connector
    pub fn new(config: PulsarConnectorConfig, parser: DebeziumParser) -> Self {
        Self {
            config,
            client: None,
            consumer: None,
            connected: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(CdcConnectorStats::default())),
            parser,
            current_message_id: Arc::new(RwLock::new(None)),
        }
    }

    /// Parse message payload to ChangeEvent
    fn parse_message(&self, payload: &[u8]) -> StreamResult<Option<ChangeEvent>> {
        // Try to parse as JSON
        let json: serde_json::Value = serde_json::from_slice(payload)
            .map_err(|e| StreamError::EventLogError(format!("JSON parse error: {}", e)))?;

        // Use Debezium parser
        self.parser.parse(&json)
    }

    /// Get connector statistics
    pub fn get_stats(&self) -> CdcConnectorStats {
        self.stats.blocking_read().clone()
    }
}

#[cfg(feature = "cdc-pulsar")]
impl CdcConnector for PulsarConnector {
    async fn connect(&mut self) -> StreamResult<()> {
        use ::pulsar::{Pulsar, TokioExecutor};

        let client = Pulsar::builder(&self.config.service_url, TokioExecutor)
            .build()
            .await
            .map_err(|e| StreamError::SubscriptionError(format!("Pulsar connect error: {}", e)))?;

        self.client = Some(client);
        self.connected.store(true, Ordering::Relaxed);

        Ok(())
    }

    async fn start_capture(&mut self) -> StreamResult<()> {
        use ::pulsar::SubType;

        if !self.is_connected() {
            self.connect().await?;
        }

        let client = self.client.as_ref().ok_or(StreamError::StreamClosed)?;

        // Build consumer options
        let mut consumer_builder = client
            .consumer()
            .with_topic(&self.config.topic)
            .with_subscription(&self.config.subscription)
            .with_subscription_type(SubType::Exclusive)
            .with_consumer_name(&self.config.consumer_name);

        // Set initial position
        // Note: Pulsar crate may need different API for initial position
        // This is a simplified version

        let consumer: ::pulsar::Consumer<Vec<u8>, ::pulsar::TokioExecutor> = consumer_builder
            .build()
            .await
            .map_err(|e| StreamError::SubscriptionError(format!("Consumer build error: {}", e)))?;

        self.consumer = Some(consumer);

        Ok(())
    }

    async fn stop_capture(&mut self) -> StreamResult<()> {
        self.consumer = None;
        self.connected.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn next_change(&mut self) -> StreamResult<Option<ChangeEvent>> {
        use futures_util::StreamExt;

        let consumer = self.consumer.as_mut().ok_or(StreamError::StreamClosed)?;

        let timeout = Duration::from_millis(self.config.receive_timeout_ms);

        match tokio::time::timeout(timeout, consumer.next()).await {
            Ok(Some(Ok(msg))) => {
                let mut stats = self.stats.write().await;
                stats.messages_received += 1;
                stats.bytes_received += msg.payload.data.len() as u64;

                // Store message ID for checkpointing
                let msg_id = format!("{:?}", msg.message_id());
                *self.current_message_id.write().await = Some(msg_id);

                match self.parse_message(&msg.payload.data) {
                    Ok(Some(event)) => {
                        // Acknowledge the message
                        if let Err(e) = consumer.ack(&msg).await {
                            stats.last_error = Some(format!("Ack error: {}", e));
                        }
                        stats.messages_processed += 1;
                        Ok(Some(event))
                    }
                    Ok(None) => {
                        // Filtered out, still ack
                        let _ = consumer.ack(&msg).await;
                        Ok(None)
                    }
                    Err(e) => {
                        stats.messages_failed += 1;
                        stats.last_error = Some(e.to_string());
                        // Negative ack for redelivery
                        let _ = consumer.nack(&msg).await;
                        Err(e)
                    }
                }
            }
            Ok(Some(Err(e))) => {
                let mut stats = self.stats.write().await;
                stats.last_error = Some(e.to_string());
                Err(StreamError::ReceiveError(format!("Pulsar error: {}", e)))
            }
            Ok(None) => Ok(None),
            Err(_) => Err(StreamError::Timeout),
        }
    }

    fn current_position(&self) -> CdcPosition {
        let msg_id = self.current_message_id.blocking_read();
        let position = msg_id.clone().unwrap_or_else(|| "0".to_string());
        CdcPosition::new(position, &self.config.topic)
    }

    async fn seek(&mut self, position: &CdcPosition) -> StreamResult<()> {
        // Pulsar seek requires message ID, which is complex
        // For now, log the intent
        tracing::info!("Pulsar seek requested to position: {:?}", position);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn stats(&self) -> CdcConnectorStats {
        self.stats.blocking_read().clone()
    }
}

// ============================================================================
// Mock Pulsar Connector (when feature is disabled)
// ============================================================================

/// Mock Pulsar connector for when feature is disabled
#[cfg(not(feature = "cdc-pulsar"))]
pub struct PulsarConnector {
    _config: PulsarConnectorConfig,
}

#[cfg(not(feature = "cdc-pulsar"))]
impl PulsarConnector {
    pub fn new(config: PulsarConnectorConfig, _parser: DebeziumParser) -> Self {
        Self { _config: config }
    }
}
