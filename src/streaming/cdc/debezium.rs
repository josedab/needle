use std::collections::HashMap;

use super::{StreamError, StreamResult};
use crate::streaming::core::{current_timestamp_millis, ChangeEvent, OperationType, ResumeToken};

// ============================================================================
// Debezium Format Parser
// ============================================================================

/// Parser for Debezium CDC format
///
/// Debezium is a popular CDC tool that produces a standardized JSON format
/// for database changes. This parser converts Debezium messages to ChangeEvents.
pub struct DebeziumParser {
    /// Source database type
    pub source_type: DebeziumSourceType,
    /// Collection name mapping (table -> collection)
    pub collection_mapping: HashMap<String, String>,
    /// Whether to include full document before change
    pub include_before: bool,
    /// Schema registry URL (for Avro format)
    pub schema_registry_url: Option<String>,
}

/// Debezium source database types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebeziumSourceType {
    PostgreSQL,
    MySQL,
    MongoDB,
    SQLServer,
    Oracle,
    Cassandra,
}

impl DebeziumParser {
    /// Create a new Debezium parser
    pub fn new(source_type: DebeziumSourceType) -> Self {
        Self {
            source_type,
            collection_mapping: HashMap::new(),
            include_before: true,
            schema_registry_url: None,
        }
    }

    /// Add a collection mapping
    pub fn with_mapping(mut self, table: impl Into<String>, collection: impl Into<String>) -> Self {
        self.collection_mapping
            .insert(table.into(), collection.into());
        self
    }

    /// Set schema registry URL for Avro
    pub fn with_schema_registry(mut self, url: impl Into<String>) -> Self {
        self.schema_registry_url = Some(url.into());
        self
    }

    /// Parse a Debezium JSON message
    pub fn parse_json(&self, json: &str) -> StreamResult<ChangeEvent> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| StreamError::EventLogError(format!("JSON parse error: {}", e)))?;

        self.parse_value(&value)
    }

    /// Parse a Debezium JSON value
    pub fn parse_value(&self, value: &serde_json::Value) -> StreamResult<ChangeEvent> {
        // Extract payload (Debezium wraps in "payload" for Kafka Connect)
        let payload = value.get("payload").unwrap_or(value);

        // Get operation type
        let op = payload
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| StreamError::EventLogError("Missing 'op' field".to_string()))?;

        let operation = match op {
            "c" | "r" => OperationType::Insert, // create or read (snapshot)
            "u" => OperationType::Update,
            "d" => OperationType::Delete,
            "t" => OperationType::Drop, // truncate
            _ => {
                return Err(StreamError::EventLogError(format!(
                    "Unknown operation: {}",
                    op
                )))
            }
        };

        // Extract source metadata
        let source = payload.get("source");
        let table = source
            .and_then(|s| s.get("table"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let collection = self
            .collection_mapping
            .get(table)
            .cloned()
            .unwrap_or_else(|| table.to_string());

        // Extract timestamp
        let ts_ms = payload
            .get("ts_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(current_timestamp_millis);

        // Extract document key
        let key = payload.get("key");
        let document_key = key
            .and_then(|k| {
                if k.is_string() {
                    k.as_str().map(String::from)
                } else {
                    Some(k.to_string())
                }
            })
            .or_else(|| {
                // Try to extract from after/before document
                payload
                    .get("after")
                    .or_else(|| payload.get("before"))
                    .and_then(|doc| doc.get("id").or_else(|| doc.get("_id")))
                    .map(|id| id.to_string())
            });

        // Extract full documents
        let after_doc = payload.get("after").map(|v| v.to_string().into_bytes());
        let before_doc = if self.include_before {
            payload.get("before").map(|v| v.to_string().into_bytes())
        } else {
            None
        };

        // Build change event
        let mut event = ChangeEvent {
            id: 0, // Will be set by event log
            operation,
            collection,
            document_key,
            full_document: after_doc,
            updated_fields: None,
            removed_fields: None,
            timestamp: ts_ms,
            resume_token: ResumeToken::new(0, ts_ms),
            full_document_before_change: before_doc,
            metadata: None,
        };

        // Extract update description for updates
        if operation == OperationType::Update {
            if let Some(update_desc) = payload.get("updateDescription") {
                let mut updated_fields = HashMap::new();
                if let Some(updated) = update_desc.get("updatedFields") {
                    if let Some(obj) = updated.as_object() {
                        for (key, value) in obj {
                            updated_fields.insert(key.clone(), value.to_string().into_bytes());
                        }
                    }
                }
                event.updated_fields = Some(updated_fields);

                if let Some(removed) = update_desc.get("removedFields") {
                    if let Some(arr) = removed.as_array() {
                        event.removed_fields = Some(
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect(),
                        );
                    }
                }
            }
        }

        // Add source metadata
        if let Some(source) = source {
            let mut metadata = HashMap::new();
            if let Some(db) = source.get("db").and_then(|v| v.as_str()) {
                metadata.insert("database".to_string(), db.to_string());
            }
            if let Some(schema) = source.get("schema").and_then(|v| v.as_str()) {
                metadata.insert("schema".to_string(), schema.to_string());
            }
            if let Some(connector) = source.get("connector").and_then(|v| v.as_str()) {
                metadata.insert("connector".to_string(), connector.to_string());
            }
            if let Some(lsn) = source.get("lsn").and_then(|v| v.as_u64()) {
                metadata.insert("lsn".to_string(), lsn.to_string());
            }
            if !metadata.is_empty() {
                event.metadata = Some(metadata);
            }
        }

        Ok(event)
    }
}
