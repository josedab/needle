use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
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
// MongoDB CDC Config
// ============================================================================

/// MongoDB change stream configuration
#[derive(Debug, Clone)]
pub struct MongoCdcConfig {
    /// MongoDB connection string
    pub connection_string: String,
    /// Database name
    pub database: String,
    /// Collections to watch (empty = all)
    pub collections: Vec<String>,
    /// General CDC config
    pub cdc_config: CdcConfig,
    /// Full document lookup on update
    pub full_document: String,
    /// Full document before change (MongoDB 6.0+)
    pub full_document_before_change: String,
}

impl Default for MongoCdcConfig {
    fn default() -> Self {
        Self {
            connection_string: "mongodb://localhost:27017".to_string(),
            database: "needle".to_string(),
            collections: vec![],
            cdc_config: CdcConfig::default(),
            full_document: "updateLookup".to_string(),
            full_document_before_change: "off".to_string(),
        }
    }
}

// ============================================================================
// MongoDB CDC Connector (feature-gated)
// ============================================================================

/// MongoDB change stream connector
#[cfg(feature = "cdc-mongodb")]
pub struct MongoCdcConnector {
    config: MongoCdcConfig,
    client: Option<::mongodb::Client>,
    connected: Arc<AtomicBool>,
    stats: Arc<RwLock<CdcConnectorStats>>,
    resume_token: Arc<RwLock<Option<::mongodb::bson::Document>>>,
}

#[cfg(feature = "cdc-mongodb")]
impl MongoCdcConnector {
    /// Create a new MongoDB CDC connector
    pub fn new(config: MongoCdcConfig) -> Self {
        Self {
            config,
            client: None,
            connected: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(CdcConnectorStats::default())),
            resume_token: Arc::new(RwLock::new(None)),
        }
    }

    /// Convert MongoDB change event to ChangeEvent
    fn convert_change_event(&self, doc: ::mongodb::bson::Document) -> StreamResult<ChangeEvent> {
        use ::mongodb::bson::Bson;

        let op_type = doc
            .get_str("operationType")
            .map_err(|_| StreamError::EventLogError("Missing operationType".to_string()))?;

        let operation = match op_type {
            "insert" => OperationType::Insert,
            "update" | "replace" => OperationType::Update,
            "delete" => OperationType::Delete,
            "drop" => OperationType::Drop,
            "rename" => OperationType::Rename,
            "dropDatabase" => OperationType::Drop,
            "invalidate" => OperationType::Drop,
            _ => {
                return Err(StreamError::EventLogError(format!(
                    "Unknown operation type: {}",
                    op_type
                )))
            }
        };

        let ns = doc.get_document("ns").ok();
        let collection = ns
            .and_then(|n| n.get_str("coll").ok())
            .unwrap_or("unknown")
            .to_string();

        let document_key = doc
            .get_document("documentKey")
            .ok()
            .and_then(|dk| dk.get("_id"))
            .map(|id| match id {
                Bson::ObjectId(oid) => oid.to_hex(),
                _ => id.to_string(),
            });

        let timestamp = doc
            .get_timestamp("clusterTime")
            .map(|t| (t.time as u64) * 1000)
            .unwrap_or_else(|_| current_timestamp_millis());

        let full_document = doc
            .get_document("fullDocument")
            .ok()
            .map(|d| serde_json::to_vec(d).unwrap_or_default());

        let full_document_before = doc
            .get_document("fullDocumentBeforeChange")
            .ok()
            .map(|d| serde_json::to_vec(d).unwrap_or_default());

        let mut event = ChangeEvent {
            id: 0,
            operation,
            collection,
            document_key,
            full_document,
            updated_fields: None,
            removed_fields: None,
            timestamp,
            resume_token: ResumeToken::new(0, timestamp),
            full_document_before_change: full_document_before,
            metadata: None,
        };

        // Extract update description
        if let Ok(update_desc) = doc.get_document("updateDescription") {
            if let Ok(updated_fields) = update_desc.get_document("updatedFields") {
                let mut fields = HashMap::new();
                for (key, value) in updated_fields {
                    fields.insert(key.clone(), serde_json::to_vec(value).unwrap_or_default());
                }
                event.updated_fields = Some(fields);
            }

            if let Ok(removed) = update_desc.get_array("removedFields") {
                event.removed_fields = Some(
                    removed
                        .iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect(),
                );
            }
        }

        Ok(event)
    }
}

#[cfg(feature = "cdc-mongodb")]
impl CdcConnector for MongoCdcConnector {
    async fn connect(&mut self) -> StreamResult<()> {
        let client = ::mongodb::Client::with_uri_str(&self.config.connection_string)
            .await
            .map_err(|e| StreamError::SubscriptionError(format!("MongoDB connect error: {}", e)))?;

        self.client = Some(client);
        self.connected.store(true, Ordering::Relaxed);

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
        self.client = None;
        Ok(())
    }

    async fn next_change(&mut self) -> StreamResult<Option<ChangeEvent>> {
        use futures_util::StreamExt;
        use ::mongodb::options::ChangeStreamOptions;

        let client = self.client.as_ref().ok_or(StreamError::StreamClosed)?;
        let db = client.database(&self.config.database);

        let mut options = ChangeStreamOptions::default();
        options.full_document = Some(::mongodb::options::FullDocumentType::UpdateLookup);

        // Resume from token if available
        let resume_token = self.resume_token.read().await.clone();
        if let Some(token) = resume_token {
            options.resume_after = Some(::mongodb::change_stream::ResumeToken::from(token));
        }

        let mut change_stream = db
            .watch()
            .await
            .map_err(|e| StreamError::SubscriptionError(format!("Watch error: {}", e)))?;

        let timeout = Duration::from_millis(self.config.cdc_config.fetch_timeout_ms);

        match tokio::time::timeout(timeout, change_stream.next()).await {
            Ok(Some(Ok(change))) => {
                let mut stats = self.stats.write().await;
                stats.messages_received += 1;

                // Store resume token
                if let Some(token) = change_stream.resume_token() {
                    *self.resume_token.write().await = Some(token.to_raw_value().as_document()
                        .cloned()
                        .unwrap_or_default());
                }

                // Convert to raw document for processing
                let doc = ::mongodb::bson::to_document(&change)
                    .map_err(|e| StreamError::EventLogError(format!("Bson error: {}", e)))?;

                match self.convert_change_event(doc) {
                    Ok(event) => {
                        stats.messages_processed += 1;
                        Ok(Some(event))
                    }
                    Err(e) => {
                        stats.messages_failed += 1;
                        stats.last_error = Some(e.to_string());
                        Err(e)
                    }
                }
            }
            Ok(Some(Err(e))) => {
                let mut stats = self.stats.write().await;
                stats.last_error = Some(e.to_string());
                Err(StreamError::ReceiveError(format!("MongoDB error: {}", e)))
            }
            Ok(None) => Ok(None),
            Err(_) => Err(StreamError::Timeout),
        }
    }

    fn current_position(&self) -> CdcPosition {
        let token = self.resume_token.blocking_read();
        let position = token
            .as_ref()
            .map(|t| t.to_string())
            .unwrap_or_else(|| "0".to_string());
        CdcPosition::new(position, &self.config.database)
    }

    async fn seek(&mut self, _position: &CdcPosition) -> StreamResult<()> {
        // MongoDB uses resume tokens, which are opaque
        // For now, just reset
        *self.resume_token.write().await = None;
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
// Mock MongoDB CDC Connector (when feature is disabled)
// ============================================================================

/// Mock MongoDB CDC connector for when feature is disabled
#[cfg(not(feature = "cdc-mongodb"))]
pub struct MongoCdcConnector {
    _config: MongoCdcConfig,
}

#[cfg(not(feature = "cdc-mongodb"))]
impl MongoCdcConnector {
    pub fn new(config: MongoCdcConfig) -> Self {
        Self { _config: config }
    }
}
