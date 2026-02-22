#![allow(clippy::unwrap_used)]
//! WebSocket Change Feed Protocol
//!
//! Formal message protocol for real-time vector change notifications.
//! Defines wire format, handshake, subscription management, and offset tracking.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::ws_protocol::{
//!     WsMessage, WsSubscribe, WsEvent, WsAck, ProtocolVersion,
//! };
//!
//! let sub = WsMessage::Subscribe(WsSubscribe { collection: "docs".into(), from_offset: None, filter: None });
//! let json = serde_json::to_string(&sub).unwrap();
//! assert!(json.contains("docs"));
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Protocol version.
pub const PROTOCOL_VERSION: &str = "1.0";

/// Top-level WebSocket message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Client → Server: Subscribe to collection changes.
    Subscribe(WsSubscribe),
    /// Client → Server: Unsubscribe.
    Unsubscribe(WsUnsubscribe),
    /// Server → Client: Change event.
    Event(WsEvent),
    /// Client → Server: Acknowledge receipt (for offset tracking).
    Ack(WsAck),
    /// Server → Client: Subscription confirmed.
    Subscribed(WsSubscribed),
    /// Server → Client: Error.
    Error(WsError),
    /// Bidirectional: Keep-alive ping.
    Ping { timestamp: u64 },
    /// Bidirectional: Keep-alive pong.
    Pong { timestamp: u64 },
}

/// Subscribe request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsSubscribe {
    pub collection: String,
    pub from_offset: Option<u64>,
    pub filter: Option<EventFilter>,
}

/// Unsubscribe request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsUnsubscribe { pub collection: String }

/// Subscription confirmed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsSubscribed {
    pub collection: String,
    pub subscription_id: String,
    pub current_offset: u64,
}

/// Change event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsEvent {
    pub subscription_id: String,
    pub offset: u64,
    pub event_type: EventType,
    pub collection: String,
    pub vector_id: String,
    pub timestamp: u64,
    pub metadata: Option<Value>,
}

/// Event type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType { Insert, Update, Delete }

/// Client acknowledgment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsAck { pub subscription_id: String, pub offset: u64 }

/// Error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsError { pub code: u32, pub message: String }

/// Event filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    pub event_types: Option<Vec<EventType>>,
    pub id_prefix: Option<String>,
}

/// Protocol version info for handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub version: String,
    pub min_supported: String,
    pub features: Vec<String>,
}

impl Default for ProtocolVersion {
    fn default() -> Self {
        Self {
            version: PROTOCOL_VERSION.into(), min_supported: "1.0".into(),
            features: vec!["offset-tracking".into(), "filtering".into(), "ping-pong".into()],
        }
    }
}

/// Validate a message for protocol compliance.
pub fn validate_message(msg: &WsMessage) -> Result<(), String> {
    match msg {
        WsMessage::Subscribe(s) => {
            if s.collection.is_empty() { return Err("Collection name required".into()); }
            Ok(())
        }
        WsMessage::Ack(a) => {
            if a.subscription_id.is_empty() { return Err("Subscription ID required".into()); }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_subscribe() {
        let msg = WsMessage::Subscribe(WsSubscribe { collection: "docs".into(), from_offset: Some(42), filter: None });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Subscribe"));
        assert!(json.contains("docs"));
    }

    #[test]
    fn test_serialize_event() {
        let msg = WsMessage::Event(WsEvent {
            subscription_id: "sub1".into(), offset: 100, event_type: EventType::Insert,
            collection: "docs".into(), vector_id: "v1".into(), timestamp: 12345, metadata: None,
        });
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("Insert"));
    }

    #[test]
    fn test_validate_empty_collection() {
        let msg = WsMessage::Subscribe(WsSubscribe { collection: "".into(), from_offset: None, filter: None });
        assert!(validate_message(&msg).is_err());
    }

    #[test]
    fn test_validate_valid() {
        let msg = WsMessage::Subscribe(WsSubscribe { collection: "docs".into(), from_offset: None, filter: None });
        assert!(validate_message(&msg).is_ok());
    }

    #[test]
    fn test_protocol_version() {
        let v = ProtocolVersion::default();
        assert_eq!(v.version, "1.0");
        assert!(v.features.contains(&"offset-tracking".to_string()));
    }

    #[test]
    fn test_roundtrip() {
        let msg = WsMessage::Ping { timestamp: 999 };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: WsMessage = serde_json::from_str(&json).unwrap();
        matches!(parsed, WsMessage::Ping { timestamp: 999 });
    }
}
