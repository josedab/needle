#![allow(clippy::unwrap_used)]

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::consumer::{ConsumerConfig, VectorFormat, VectorMessage};

// ============================================================================
// Feature-Gated CDC Client Implementations
// ============================================================================

/// Kafka CDC implementation using rdkafka
#[cfg(feature = "cdc-kafka")]
pub mod kafka_cdc {
    use super::*;
    use rdkafka::config::ClientConfig;
    use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
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

            consumer.subscribe(&[&config.topic]).map_err(|e| {
                NeedleError::InvalidOperation(format!("Kafka subscribe error: {}", e))
            })?;

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
        fn parse_message(
            &self,
            payload: &[u8],
            msg: &rdkafka::message::BorrowedMessage,
        ) -> Result<VectorMessage> {
            match self.config.vector_format {
                VectorFormat::Json => {
                    let json: serde_json::Value = serde_json::from_slice(payload).map_err(|e| {
                        NeedleError::InvalidFormat(format!("JSON parse error: {}", e))
                    })?;

                    let id = json
                        .get(&self.config.id_field)
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| NeedleError::InvalidFormat("Missing id field".to_string()))?
                        .to_string();

                    let vector: Vec<f32> = json
                        .get(&self.config.vector_field)
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| {
                            NeedleError::InvalidFormat("Missing vector field".to_string())
                        })?
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();

                    let metadata = self
                        .config
                        .metadata_field
                        .as_ref()
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
                    let id_len = u64::from_le_bytes(
                        payload[..8].try_into().expect("slice is exactly 8 bytes"),
                    ) as usize;
                    if payload.len() < 8 + id_len {
                        return Err(NeedleError::InvalidFormat("Invalid id length".to_string()));
                    }
                    let id = String::from_utf8_lossy(&payload[8..8 + id_len]).to_string();
                    let vector_bytes = &payload[8 + id_len..];
                    let vector: Vec<f32> = vector_bytes
                        .chunks_exact(4)
                        .map(|b| {
                            f32::from_le_bytes(b.try_into().expect("chunk is exactly 4 bytes"))
                        })
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
            self.consumer
                .commit_consumer_state(CommitMode::Sync)
                .map_err(|e| NeedleError::InvalidOperation(format!("Kafka commit error: {}", e)))
        }
    }
}

/// Pulsar CDC implementation
#[cfg(feature = "cdc-pulsar")]
pub mod pulsar_cdc {
    use super::*;
    use pulsar::{
        consumer::InitialPosition, Consumer as PulsarConsumer, Pulsar, SubType, TokioExecutor,
    };
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
                .map_err(|e| {
                    NeedleError::InvalidOperation(format!("Pulsar connection error: {}", e))
                })?;

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
                .map_err(|e| {
                    NeedleError::InvalidOperation(format!("Pulsar consumer error: {}", e))
                })?;

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
                    consumer.try_next(),
                )
                .await
                {
                    Ok(Ok(Some(msg))) => {
                        if let Ok(vec_msg) = self.parse_message(&msg) {
                            messages.push(vec_msg);
                            if !self.config.auto_commit {
                                if let Err(e) = consumer.ack(&msg).await {
                                    tracing::warn!("Failed to ack Pulsar message: {}", e);
                                }
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

            let id = json
                .get(&self.config.id_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| NeedleError::InvalidFormat("Missing id field".to_string()))?
                .to_string();

            let vector: Vec<f32> = json
                .get(&self.config.vector_field)
                .and_then(|v| v.as_array())
                .ok_or_else(|| NeedleError::InvalidFormat("Missing vector field".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            let metadata = self
                .config
                .metadata_field
                .as_ref()
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
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio_postgres::{Client, NoTls, Row};

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
                .map_err(|e| {
                    NeedleError::InvalidOperation(format!("PostgreSQL connection error: {}", e))
                })?;

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
                .query(
                    &query,
                    &[&(last_lsn as i64), &(self.config.batch_size as i64)],
                )
                .await
                .map_err(|e| {
                    NeedleError::InvalidOperation(format!("PostgreSQL query error: {}", e))
                })?;

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
            let id: String = row
                .try_get("id")
                .map_err(|e| NeedleError::InvalidFormat(format!("Missing id: {}", e)))?;

            // PostgreSQL array to Vec<f32>
            let vector: Vec<f32> = row
                .try_get::<_, Vec<f32>>("vector")
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
    use futures::StreamExt;
    use mongodb::change_stream::event::ChangeStreamEvent;
    use mongodb::{
        bson::{doc, Document},
        options::ClientOptions,
        Client, Collection,
    };
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// MongoDB change stream consumer
    pub struct MongoVectorConsumer {
        collection: Collection<Document>,
        config: ConsumerConfig,
        resume_token: Arc<Mutex<Option<mongodb::bson::Document>>>,
    }

    impl MongoVectorConsumer {
        /// Create a new MongoDB change stream consumer
        pub async fn new(config: ConsumerConfig) -> Result<Self> {
            let client_options = ClientOptions::parse(&config.brokers).await.map_err(|e| {
                NeedleError::InvalidOperation(format!("MongoDB connection error: {}", e))
            })?;

            let client = Client::with_options(client_options).map_err(|e| {
                NeedleError::InvalidOperation(format!("MongoDB client error: {}", e))
            })?;

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
                self.collection.watch().resume_after(token).await
            } else {
                self.collection.watch().await
            }
            .map_err(|e| {
                NeedleError::InvalidOperation(format!("MongoDB change stream error: {}", e))
            })?;

            let mut messages = Vec::new();

            // Poll for up to batch_size changes with timeout
            let timeout = tokio::time::Duration::from_millis(self.config.poll_timeout_ms);
            let deadline = tokio::time::Instant::now() + timeout;

            while messages.len() < self.config.batch_size && tokio::time::Instant::now() < deadline
            {
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
            let doc = event.full_document.as_ref().ok_or_else(|| {
                NeedleError::InvalidFormat("No full document in change event".to_string())
            })?;

            let id = doc
                .get_str("_id")
                .or_else(|_| doc.get_object_id("_id").map(|oid| oid.to_hex()))
                .map_err(|_| NeedleError::InvalidFormat("Missing _id field".to_string()))?
                .to_string();

            let vector: Vec<f32> = doc
                .get_array(&self.config.vector_field)
                .map_err(|_| NeedleError::InvalidFormat("Missing vector field".to_string()))?
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();

            let metadata = self.config.metadata_field.as_ref().and_then(|field| {
                doc.get_document(field)
                    .ok()
                    .map(|d| serde_json::to_value(d).unwrap_or(serde_json::Value::Null))
            });

            let timestamp = event.cluster_time.map(|t| t.time as u64).unwrap_or(0);

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
