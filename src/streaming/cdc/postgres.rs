use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use super::{
    CdcConfig, CdcConnector, CdcConnectorStats, CdcPosition,
};
use crate::streaming::core::{
    ChangeEvent, OperationType, ResumeToken, StreamError, StreamResult,
    current_timestamp_millis,
};

// ============================================================================
// PostgreSQL CDC Config
// ============================================================================

/// PostgreSQL CDC connector configuration
#[derive(Debug, Clone)]
pub struct PostgresCdcConfig {
    /// PostgreSQL connection string
    pub connection_string: String,
    /// Replication slot name
    pub slot_name: String,
    /// Publication name
    pub publication_name: String,
    /// Tables to capture changes from
    pub tables: Vec<String>,
    /// General CDC config
    pub cdc_config: CdcConfig,
}

impl Default for PostgresCdcConfig {
    fn default() -> Self {
        Self {
            connection_string: "postgres://localhost/needle".to_string(),
            slot_name: "needle_slot".to_string(),
            publication_name: "needle_publication".to_string(),
            tables: vec![],
            cdc_config: CdcConfig::default(),
        }
    }
}

// ============================================================================
// PostgreSQL CDC Connector (feature-gated)
// ============================================================================

/// PostgreSQL CDC connector using logical replication
#[cfg(feature = "cdc-postgres")]
pub struct PostgresCdcConnector {
    config: PostgresCdcConfig,
    client: Option<tokio_postgres::Client>,
    connected: Arc<AtomicBool>,
    stats: Arc<RwLock<CdcConnectorStats>>,
    current_lsn: Arc<AtomicU64>,
    collection_mapping: HashMap<String, String>,
}

#[cfg(feature = "cdc-postgres")]
impl PostgresCdcConnector {
    /// Create a new PostgreSQL CDC connector
    pub fn new(config: PostgresCdcConfig) -> Self {
        Self {
            config,
            client: None,
            connected: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(CdcConnectorStats::default())),
            current_lsn: Arc::new(AtomicU64::new(0)),
            collection_mapping: HashMap::new(),
        }
    }

    /// Add table to collection mapping
    pub fn with_mapping(mut self, table: impl Into<String>, collection: impl Into<String>) -> Self {
        self.collection_mapping.insert(table.into(), collection.into());
        self
    }

    /// Parse a PostgreSQL logical replication message
    fn parse_message(&self, data: &[u8]) -> StreamResult<Option<ChangeEvent>> {
        // Simplified parsing - real implementation would use pgoutput protocol
        if data.is_empty() {
            return Ok(None);
        }

        let msg_type = data[0] as char;
        let timestamp = current_timestamp_millis();

        match msg_type {
            'I' => {
                // Insert
                let json_str = String::from_utf8_lossy(&data[1..]);
                let value: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| StreamError::EventLogError(format!("Parse error: {}", e)))?;

                let table = value.get("table").and_then(|v| v.as_str()).unwrap_or("unknown");
                let collection = self.collection_mapping.get(table).cloned().unwrap_or_else(|| table.to_string());
                let id = value.get("id").map(|v| v.to_string());

                Ok(Some(ChangeEvent::insert(
                    &collection,
                    &id.unwrap_or_default(),
                    json_str.as_bytes().to_vec(),
                    0,
                )))
            }
            'U' => {
                // Update
                let json_str = String::from_utf8_lossy(&data[1..]);
                let value: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| StreamError::EventLogError(format!("Parse error: {}", e)))?;

                let table = value.get("table").and_then(|v| v.as_str()).unwrap_or("unknown");
                let collection = self.collection_mapping.get(table).cloned().unwrap_or_else(|| table.to_string());
                let id = value.get("id").map(|v| v.to_string()).unwrap_or_default();

                let event = ChangeEvent {
                    id: 0,
                    operation: OperationType::Update,
                    collection,
                    document_key: Some(id),
                    full_document: Some(json_str.as_bytes().to_vec()),
                    updated_fields: None,
                    removed_fields: None,
                    timestamp,
                    resume_token: ResumeToken::new(0, timestamp),
                    full_document_before_change: None,
                    metadata: None,
                };

                Ok(Some(event))
            }
            'D' => {
                // Delete
                let json_str = String::from_utf8_lossy(&data[1..]);
                let value: serde_json::Value = serde_json::from_str(&json_str)
                    .map_err(|e| StreamError::EventLogError(format!("Parse error: {}", e)))?;

                let table = value.get("table").and_then(|v| v.as_str()).unwrap_or("unknown");
                let collection = self.collection_mapping.get(table).cloned().unwrap_or_else(|| table.to_string());
                let id = value.get("id").map(|v| v.to_string()).unwrap_or_default();

                Ok(Some(ChangeEvent::delete(&collection, &id, 0)))
            }
            _ => Ok(None),
        }
    }
}

#[cfg(feature = "cdc-postgres")]
impl CdcConnector for PostgresCdcConnector {
    async fn connect(&mut self) -> StreamResult<()> {
        let (client, connection) = tokio_postgres::connect(&self.config.connection_string, tokio_postgres::NoTls)
            .await
            .map_err(|e| StreamError::SubscriptionError(format!("PostgreSQL connect error: {}", e)))?;

        // Spawn connection handler
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("PostgreSQL connection error: {}", e);
            }
        });

        self.client = Some(client);
        self.connected.store(true, Ordering::Relaxed);

        Ok(())
    }

    async fn start_capture(&mut self) -> StreamResult<()> {
        if !self.is_connected() {
            self.connect().await?;
        }

        let client = self.client.as_ref().ok_or(StreamError::StreamClosed)?;

        // Create replication slot if it doesn't exist
        let slot_query = format!(
            "SELECT pg_create_logical_replication_slot('{}', 'pgoutput') WHERE NOT EXISTS (
                SELECT 1 FROM pg_replication_slots WHERE slot_name = '{}'
            )",
            self.config.slot_name, self.config.slot_name
        );

        let _ = client.execute(&slot_query, &[]).await;

        Ok(())
    }

    async fn stop_capture(&mut self) -> StreamResult<()> {
        self.connected.store(false, Ordering::Relaxed);
        self.client = None;
        Ok(())
    }

    async fn next_change(&mut self) -> StreamResult<Option<ChangeEvent>> {
        let client = self.client.as_ref().ok_or(StreamError::StreamClosed)?;

        // Poll for changes using pg_logical_slot_get_changes
        let query = format!(
            "SELECT lsn, xid, data FROM pg_logical_slot_get_changes('{}', NULL, {}, 'proto_version', '1', 'publication_names', '{}')",
            self.config.slot_name,
            self.config.cdc_config.batch_size,
            self.config.publication_name
        );

        let rows = client
            .query(&query, &[])
            .await
            .map_err(|e| StreamError::ReceiveError(format!("PostgreSQL query error: {}", e)))?;

        if let Some(row) = rows.first() {
            let data: &[u8] = row.get(2);
            let mut stats = self.stats.write().await;
            stats.messages_received += 1;
            stats.bytes_received += data.len() as u64;

            match self.parse_message(data) {
                Ok(Some(event)) => {
                    stats.messages_processed += 1;
                    Ok(Some(event))
                }
                Ok(None) => Ok(None),
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

    fn current_position(&self) -> CdcPosition {
        CdcPosition::new(
            self.current_lsn.load(Ordering::Relaxed).to_string(),
            &self.config.slot_name,
        )
    }

    async fn seek(&mut self, position: &CdcPosition) -> StreamResult<()> {
        let lsn: u64 = position
            .position
            .parse()
            .map_err(|_| StreamError::InvalidResumeToken("Invalid LSN".to_string()))?;

        self.current_lsn.store(lsn, Ordering::Relaxed);
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
// Mock PostgreSQL CDC Connector (when feature is disabled)
// ============================================================================

/// Mock PostgreSQL CDC connector for when feature is disabled
#[cfg(not(feature = "cdc-postgres"))]
pub struct PostgresCdcConnector {
    _config: PostgresCdcConfig,
}

#[cfg(not(feature = "cdc-postgres"))]
impl PostgresCdcConnector {
    pub fn new(config: PostgresCdcConfig) -> Self {
        Self { _config: config }
    }
}
