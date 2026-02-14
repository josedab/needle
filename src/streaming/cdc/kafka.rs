use super::debezium::DebeziumParser;
use super::CdcConfig;

// ============================================================================
// Kafka Connector Config
// ============================================================================

/// Kafka CDC connector configuration
#[derive(Debug, Clone)]
pub struct KafkaConnectorConfig {
    /// Kafka broker addresses
    pub brokers: Vec<String>,
    /// Topic to consume from
    pub topic: String,
    /// Consumer group ID
    pub group_id: String,
    /// General CDC config
    pub cdc_config: CdcConfig,
    /// Security protocol (PLAINTEXT, SSL, SASL_SSL, etc.)
    pub security_protocol: String,
    /// SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)
    pub sasl_mechanism: Option<String>,
    /// SASL username
    pub sasl_username: Option<String>,
    /// SASL password
    pub sasl_password: Option<String>,
    /// SSL CA certificate path
    pub ssl_ca_path: Option<String>,
    /// Consumer offset reset policy (earliest, latest)
    pub offset_reset: String,
}

impl Default for KafkaConnectorConfig {
    fn default() -> Self {
        Self {
            brokers: vec!["localhost:9092".to_string()],
            topic: "".to_string(),
            group_id: "needle-cdc".to_string(),
            cdc_config: CdcConfig::default(),
            security_protocol: "PLAINTEXT".to_string(),
            sasl_mechanism: None,
            sasl_username: None,
            sasl_password: None,
            ssl_ca_path: None,
            offset_reset: "earliest".to_string(),
        }
    }
}

// ============================================================================
// Kafka Connector (feature-gated)
// ============================================================================

/// Kafka CDC connector for consuming Debezium messages
#[cfg(feature = "cdc-kafka")]
pub struct KafkaConnector {
    config: KafkaConnectorConfig,
    consumer: Option<rdkafka::consumer::StreamConsumer>,
    parser: DebeziumParser,
    connected: Arc<AtomicBool>,
    stats: Arc<RwLock<CdcConnectorStats>>,
    current_offset: Arc<AtomicU64>,
    start_time: Option<std::time::Instant>,
}

#[cfg(feature = "cdc-kafka")]
impl KafkaConnector {
    /// Create a new Kafka connector
    pub fn new(config: KafkaConnectorConfig, parser: DebeziumParser) -> Self {
        Self {
            config,
            consumer: None,
            parser,
            connected: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(CdcConnectorStats::default())),
            current_offset: Arc::new(AtomicU64::new(0)),
            start_time: None,
        }
    }

    /// Build rdkafka consumer config
    fn build_consumer_config(&self) -> rdkafka::ClientConfig {
        use rdkafka::ClientConfig;

        let mut config = ClientConfig::new();
        config
            .set("bootstrap.servers", self.config.brokers.join(","))
            .set("group.id", &self.config.group_id)
            .set("enable.auto.commit", "true")
            .set(
                "auto.commit.interval.ms",
                self.config.cdc_config.auto_commit_interval_ms.to_string(),
            )
            .set("auto.offset.reset", &self.config.offset_reset)
            .set("security.protocol", &self.config.security_protocol);

        if let Some(ref mechanism) = self.config.sasl_mechanism {
            config.set("sasl.mechanism", mechanism);
        }
        if let Some(ref username) = self.config.sasl_username {
            config.set("sasl.username", username);
        }
        if let Some(ref password) = self.config.sasl_password {
            config.set("sasl.password", password);
        }
        if let Some(ref ca_path) = self.config.ssl_ca_path {
            config.set("ssl.ca.location", ca_path);
        }

        config
    }
}

#[cfg(feature = "cdc-kafka")]
impl CdcConnector for KafkaConnector {
    async fn connect(&mut self) -> StreamResult<()> {
        use rdkafka::consumer::Consumer;

        let consumer_config = self.build_consumer_config();
        let consumer: rdkafka::consumer::StreamConsumer = consumer_config
            .create()
            .map_err(|e| StreamError::SubscriptionError(format!("Kafka create error: {}", e)))?;

        consumer
            .subscribe(&[&self.config.topic])
            .map_err(|e| StreamError::SubscriptionError(format!("Kafka subscribe error: {}", e)))?;

        self.consumer = Some(consumer);
        self.connected.store(true, Ordering::Relaxed);
        self.start_time = Some(std::time::Instant::now());

        Ok(())
    }

    async fn start_capture(&mut self) -> StreamResult<()> {
        if !self.is_connected() {
            self.connect().await?;
        }
        Ok(())
    }

    async fn stop_capture(&mut self) -> StreamResult<()> {
        self.connected.store(false, Ordering::Relaxed);
        self.consumer = None;
        Ok(())
    }

    async fn next_change(&mut self) -> StreamResult<Option<ChangeEvent>> {
        use rdkafka::consumer::StreamConsumer;
        use rdkafka::Message;

        let consumer = self.consumer.as_ref().ok_or(StreamError::StreamClosed)?;

        let timeout = Duration::from_millis(self.config.cdc_config.fetch_timeout_ms);

        match tokio::time::timeout(timeout, consumer.recv()).await {
            Ok(Ok(msg)) => {
                let mut stats = self.stats.write().await;
                stats.messages_received += 1;

                if let Some(payload) = msg.payload() {
                    stats.bytes_received += payload.len() as u64;

                    let json_str = String::from_utf8_lossy(payload);
                    match self.parser.parse_json(&json_str) {
                        Ok(event) => {
                            stats.messages_processed += 1;
                            self.current_offset
                                .store(msg.offset() as u64, Ordering::Relaxed);
                            Ok(Some(event))
                        }
                        Err(e) => {
                            stats.messages_failed += 1;
                            stats.last_error = Some(e.to_string());
                            Err(e)
                        }
                    }
                } else {
                    Ok(None)
                }
            }
            Ok(Err(e)) => {
                let mut stats = self.stats.write().await;
                stats.last_error = Some(e.to_string());
                Err(StreamError::ReceiveError(format!("Kafka error: {}", e)))
            }
            Err(_) => Err(StreamError::Timeout),
        }
    }

    fn current_position(&self) -> CdcPosition {
        CdcPosition::new(
            self.current_offset.load(Ordering::Relaxed).to_string(),
            &self.config.topic,
        )
    }

    async fn seek(&mut self, position: &CdcPosition) -> StreamResult<()> {
        use rdkafka::consumer::Consumer;
        use rdkafka::TopicPartitionList;

        let consumer = self.consumer.as_ref().ok_or(StreamError::StreamClosed)?;

        let offset: i64 = position
            .position
            .parse()
            .map_err(|_| StreamError::InvalidResumeToken("Invalid offset".to_string()))?;

        let partition = position.partition.unwrap_or(0);

        let mut tpl = TopicPartitionList::new();
        tpl.add_partition_offset(
            &self.config.topic,
            partition,
            rdkafka::Offset::Offset(offset),
        )
        .map_err(|e| StreamError::EventLogError(format!("TPL error: {}", e)))?;

        consumer
            .seek(
                &self.config.topic,
                partition,
                rdkafka::Offset::Offset(offset),
                Duration::from_secs(10),
            )
            .map_err(|e| StreamError::EventLogError(format!("Seek error: {}", e)))?;

        self.current_offset.store(offset as u64, Ordering::Relaxed);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    fn stats(&self) -> CdcConnectorStats {
        let stats = self.stats.blocking_read();
        let mut result = stats.clone();
        if let Some(start) = self.start_time {
            result.uptime_secs = start.elapsed().as_secs();
        }
        result
    }
}

// ============================================================================
// Mock Kafka Connector (when feature is disabled)
// ============================================================================

/// Mock Kafka connector for when feature is disabled
#[cfg(not(feature = "cdc-kafka"))]
pub struct KafkaConnector {
    _config: KafkaConnectorConfig,
}

#[cfg(not(feature = "cdc-kafka"))]
impl KafkaConnector {
    pub fn new(config: KafkaConnectorConfig, _parser: DebeziumParser) -> Self {
        Self { _config: config }
    }
}
