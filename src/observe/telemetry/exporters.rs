//! Export formats: OTLP, Jaeger, Zipkin, Console.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::spans::{AttributeValue, Span, SpanKind, SpanStatus};
use super::{TelemetryConfig, Telemetry};

/// Export format for telemetry data
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// OpenTelemetry Protocol (OTLP)
    #[default]
    Otlp,
    /// Jaeger format
    Jaeger,
    /// Zipkin format
    Zipkin,
    /// JSON format
    Json,
    /// Console/stdout (for debugging)
    Console,
}

// ============================================================================
// OTLP Export Format
// ============================================================================

/// OTLP span representation (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpSpan {
    /// Trace ID (hex string)
    pub trace_id: String,
    /// Span ID (hex string)
    pub span_id: String,
    /// Parent span ID (hex string)
    pub parent_span_id: String,
    /// Span name
    pub name: String,
    /// Start time (Unix nanoseconds)
    pub start_time_unix_nano: u64,
    /// End time (Unix nanoseconds)
    pub end_time_unix_nano: u64,
    /// Attributes
    pub attributes: Vec<OtlpAttribute>,
    /// Events
    pub events: Vec<OtlpEvent>,
    /// Status
    pub status: OtlpStatus,
    /// Kind
    pub kind: u8,
}

/// OTLP attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpAttribute {
    /// Key
    pub key: String,
    /// Value
    pub value: OtlpAnyValue,
}

/// OTLP any value (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OtlpAnyValue {
    /// String value
    String { string_value: String },
    /// Integer value
    Int { int_value: i64 },
    /// Double value
    Double { double_value: f64 },
    /// Boolean value
    Bool { bool_value: bool },
}

/// OTLP event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpEvent {
    /// Event name
    pub name: String,
    /// Timestamp
    pub time_unix_nano: u64,
    /// Attributes
    pub attributes: Vec<OtlpAttribute>,
}

/// OTLP status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpStatus {
    /// Status code (0=Unset, 1=Ok, 2=Error)
    pub code: u8,
    /// Status message
    pub message: String,
}

/// OTLP resource spans (top-level export structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpResourceSpans {
    /// Resource attributes
    pub resource: OtlpResource,
    /// Scope spans
    pub scope_spans: Vec<OtlpScopeSpans>,
}

/// OTLP resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpResource {
    /// Attributes
    pub attributes: Vec<OtlpAttribute>,
}

/// OTLP scope spans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpScopeSpans {
    /// Instrumentation scope
    pub scope: OtlpScope,
    /// Spans
    pub spans: Vec<OtlpSpan>,
}

/// OTLP instrumentation scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtlpScope {
    /// Name
    pub name: String,
    /// Version
    pub version: String,
}

impl From<&AttributeValue> for OtlpAnyValue {
    fn from(value: &AttributeValue) -> Self {
        match value {
            AttributeValue::String(s) => OtlpAnyValue::String {
                string_value: s.clone(),
            },
            AttributeValue::Int(i) => OtlpAnyValue::Int { int_value: *i },
            AttributeValue::Float(f) => OtlpAnyValue::Double { double_value: *f },
            AttributeValue::Bool(b) => OtlpAnyValue::Bool { bool_value: *b },
            AttributeValue::StringArray(arr) => OtlpAnyValue::String {
                string_value: arr.join(","),
            },
            AttributeValue::IntArray(arr) => OtlpAnyValue::String {
                string_value: arr
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            },
        }
    }
}

impl From<&Span> for OtlpSpan {
    fn from(span: &Span) -> Self {
        let attributes: Vec<OtlpAttribute> = span
            .attributes
            .iter()
            .map(|(k, v)| OtlpAttribute {
                key: k.clone(),
                value: v.into(),
            })
            .collect();

        let events: Vec<OtlpEvent> = span
            .events
            .iter()
            .map(|e| OtlpEvent {
                name: e.name.clone(),
                time_unix_nano: e.timestamp,
                attributes: e
                    .attributes
                    .iter()
                    .map(|(k, v)| OtlpAttribute {
                        key: k.clone(),
                        value: v.into(),
                    })
                    .collect(),
            })
            .collect();

        let status = OtlpStatus {
            code: match span.status {
                SpanStatus::Unset => 0,
                SpanStatus::Ok => 1,
                SpanStatus::Error => 2,
            },
            message: span.status_message.clone().unwrap_or_default(),
        };

        let kind = match span.kind {
            SpanKind::Internal => 1,
            SpanKind::Server => 2,
            SpanKind::Client => 3,
            SpanKind::Producer => 4,
            SpanKind::Consumer => 5,
        };

        OtlpSpan {
            trace_id: span.context.trace_id.clone(),
            span_id: span.context.span_id.clone(),
            parent_span_id: span.context.parent_span_id.clone().unwrap_or_default(),
            name: span.name.clone(),
            start_time_unix_nano: span.start_time,
            end_time_unix_nano: span.end_time.unwrap_or(span.start_time),
            attributes,
            events,
            status,
            kind,
        }
    }
}

// ============================================================================
// Export Implementations
// ============================================================================

impl Telemetry {
    /// Export spans in OTLP JSON format.
    pub fn export_otlp(&self) -> OtlpResourceSpans {
        let spans = self.get_spans();
        let otlp_spans: Vec<OtlpSpan> = spans.iter().map(OtlpSpan::from).collect();

        let resource_attrs: Vec<OtlpAttribute> = self
            .config()
            .resource
            .to_map()
            .into_iter()
            .map(|(k, v)| OtlpAttribute {
                key: k,
                value: OtlpAnyValue::String { string_value: v },
            })
            .collect();

        OtlpResourceSpans {
            resource: OtlpResource {
                attributes: resource_attrs,
            },
            scope_spans: vec![OtlpScopeSpans {
                scope: OtlpScope {
                    name: "needle".to_string(),
                    version: env!("CARGO_PKG_VERSION").to_string(),
                },
                spans: otlp_spans,
            }],
        }
    }

    /// Export spans in OTLP JSON string format.
    pub fn export_otlp_json(&self) -> String {
        let otlp = self.export_otlp();
        serde_json::to_string_pretty(&otlp).unwrap_or_default()
    }

    /// Export spans in Jaeger format (Thrift JSON).
    pub fn export_jaeger(&self) -> JaegerExport {
        let spans = self.get_spans();
        let jaeger_spans: Vec<JaegerSpan> = spans.iter().map(JaegerSpan::from).collect();

        JaegerExport {
            data: vec![JaegerTrace {
                trace_id: spans
                    .first()
                    .map(|s| s.context.trace_id.clone())
                    .unwrap_or_default(),
                spans: jaeger_spans,
                processes: {
                    let mut map = HashMap::new();
                    map.insert(
                        "p1".to_string(),
                        JaegerProcess {
                            service_name: self.config().service_name.clone(),
                            tags: vec![
                                JaegerTag {
                                    key: "service.version".to_string(),
                                    tag_type: "string".to_string(),
                                    value: JaegerTagValue::String(
                                        self.config().service_version.clone(),
                                    ),
                                },
                                JaegerTag {
                                    key: "deployment.environment".to_string(),
                                    tag_type: "string".to_string(),
                                    value: JaegerTagValue::String(self.config().environment.clone()),
                                },
                            ],
                        },
                    );
                    map
                },
            }],
        }
    }

    /// Export spans in Jaeger JSON string format.
    pub fn export_jaeger_json(&self) -> String {
        let jaeger = self.export_jaeger();
        serde_json::to_string_pretty(&jaeger).unwrap_or_default()
    }

    /// Export spans in Zipkin format.
    pub fn export_zipkin(&self) -> Vec<ZipkinSpan> {
        let spans = self.get_spans();
        spans
            .iter()
            .map(|s| ZipkinSpan::from_span(s, self.config()))
            .collect()
    }

    /// Export spans in Zipkin JSON string format.
    pub fn export_zipkin_json(&self) -> String {
        let zipkin = self.export_zipkin();
        serde_json::to_string_pretty(&zipkin).unwrap_or_default()
    }

    /// Export in configured format.
    pub fn export(&self) -> String {
        match self.config().export_format {
            ExportFormat::Otlp => self.export_otlp_json(),
            ExportFormat::Jaeger => self.export_jaeger_json(),
            ExportFormat::Zipkin => self.export_zipkin_json(),
            ExportFormat::Json => self.export_spans_json(),
            ExportFormat::Console => self.export_console(),
        }
    }

    /// Export spans in human-readable console format.
    pub fn export_console(&self) -> String {
        let spans = self.get_spans();
        let mut output = String::new();

        for span in spans {
            let duration_ms = span.duration_ms().unwrap_or(0.0);
            let status_icon = match span.status {
                SpanStatus::Ok => "✓",
                SpanStatus::Error => "✗",
                SpanStatus::Unset => "○",
            };

            output.push_str(&format!(
                "{} {} [{:.2}ms] trace={} span={}\n",
                status_icon,
                span.name,
                duration_ms,
                &span.context.trace_id[..8],
                &span.context.span_id[..8]
            ));

            for (key, value) in &span.attributes {
                let value_str = match value {
                    AttributeValue::String(s) => s.clone(),
                    AttributeValue::Int(i) => i.to_string(),
                    AttributeValue::Float(f) => f.to_string(),
                    AttributeValue::Bool(b) => b.to_string(),
                    AttributeValue::StringArray(arr) => arr.join(", "),
                    AttributeValue::IntArray(arr) => arr
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                };
                output.push_str(&format!("    {}: {}\n", key, value_str));
            }

            for event in &span.events {
                output.push_str(&format!("    • {}\n", event.name));
            }

            if let Some(ref msg) = span.status_message {
                output.push_str(&format!("    error: {}\n", msg));
            }
        }

        output
    }
}

// ============================================================================
// Jaeger Export Format
// ============================================================================

/// Jaeger export format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerExport {
    /// Data array
    pub data: Vec<JaegerTrace>,
}

/// Jaeger trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerTrace {
    /// Trace ID
    #[serde(rename = "traceID")]
    pub trace_id: String,
    /// Spans
    pub spans: Vec<JaegerSpan>,
    /// Processes
    pub processes: HashMap<String, JaegerProcess>,
}

/// Jaeger span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerSpan {
    /// Trace ID
    #[serde(rename = "traceID")]
    pub trace_id: String,
    /// Span ID
    #[serde(rename = "spanID")]
    pub span_id: String,
    /// Operation name
    #[serde(rename = "operationName")]
    pub operation_name: String,
    /// References (parent spans)
    pub references: Vec<JaegerReference>,
    /// Start time (microseconds)
    #[serde(rename = "startTime")]
    pub start_time: u64,
    /// Duration (microseconds)
    pub duration: u64,
    /// Tags
    pub tags: Vec<JaegerTag>,
    /// Logs (events)
    pub logs: Vec<JaegerLog>,
    /// Process ID
    #[serde(rename = "processID")]
    pub process_id: String,
}

/// Jaeger reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerReference {
    /// Reference type
    #[serde(rename = "refType")]
    pub ref_type: String,
    /// Trace ID
    #[serde(rename = "traceID")]
    pub trace_id: String,
    /// Span ID
    #[serde(rename = "spanID")]
    pub span_id: String,
}

/// Jaeger tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerTag {
    /// Key
    pub key: String,
    /// Type
    #[serde(rename = "type")]
    pub tag_type: String,
    /// Value
    pub value: JaegerTagValue,
}

/// Jaeger tag value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum JaegerTagValue {
    /// String value
    String(String),
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Bool(bool),
}

/// Jaeger log (event)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerLog {
    /// Timestamp (microseconds)
    pub timestamp: u64,
    /// Fields
    pub fields: Vec<JaegerTag>,
}

/// Jaeger process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerProcess {
    /// Service name
    #[serde(rename = "serviceName")]
    pub service_name: String,
    /// Tags
    pub tags: Vec<JaegerTag>,
}

impl From<&Span> for JaegerSpan {
    fn from(span: &Span) -> Self {
        let tags: Vec<JaegerTag> = span
            .attributes
            .iter()
            .map(|(k, v)| JaegerTag {
                key: k.clone(),
                tag_type: match v {
                    AttributeValue::String(_) => "string",
                    AttributeValue::Int(_) => "int64",
                    AttributeValue::Float(_) => "float64",
                    AttributeValue::Bool(_) => "bool",
                    AttributeValue::StringArray(_) => "string",
                    AttributeValue::IntArray(_) => "string",
                }
                .to_string(),
                value: match v {
                    AttributeValue::String(s) => JaegerTagValue::String(s.clone()),
                    AttributeValue::Int(i) => JaegerTagValue::Int(*i),
                    AttributeValue::Float(f) => JaegerTagValue::Float(*f),
                    AttributeValue::Bool(b) => JaegerTagValue::Bool(*b),
                    AttributeValue::StringArray(arr) => JaegerTagValue::String(arr.join(",")),
                    AttributeValue::IntArray(arr) => JaegerTagValue::String(
                        arr.iter()
                            .map(|i| i.to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    ),
                },
            })
            .collect();

        let logs: Vec<JaegerLog> = span
            .events
            .iter()
            .map(|e| JaegerLog {
                timestamp: e.timestamp / 1000, // Convert nanos to micros
                fields: vec![JaegerTag {
                    key: "event".to_string(),
                    tag_type: "string".to_string(),
                    value: JaegerTagValue::String(e.name.clone()),
                }],
            })
            .collect();

        let references = if let Some(ref parent) = span.context.parent_span_id {
            vec![JaegerReference {
                ref_type: "CHILD_OF".to_string(),
                trace_id: span.context.trace_id.clone(),
                span_id: parent.clone(),
            }]
        } else {
            vec![]
        };

        let duration = span.end_time.unwrap_or(span.start_time) - span.start_time;

        JaegerSpan {
            trace_id: span.context.trace_id.clone(),
            span_id: span.context.span_id.clone(),
            operation_name: span.name.clone(),
            references,
            start_time: span.start_time / 1000, // Convert nanos to micros
            duration: duration / 1000,
            tags,
            logs,
            process_id: "p1".to_string(),
        }
    }
}

// ============================================================================
// Zipkin Export Format
// ============================================================================

/// Zipkin span format (v2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinSpan {
    /// Trace ID
    #[serde(rename = "traceId")]
    pub trace_id: String,
    /// Span ID
    pub id: String,
    /// Parent ID
    #[serde(rename = "parentId", skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// Operation name
    pub name: String,
    /// Timestamp (microseconds)
    pub timestamp: u64,
    /// Duration (microseconds)
    pub duration: u64,
    /// Kind
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Local endpoint
    #[serde(rename = "localEndpoint")]
    pub local_endpoint: ZipkinEndpoint,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Annotations (events)
    pub annotations: Vec<ZipkinAnnotation>,
}

/// Zipkin endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinEndpoint {
    /// Service name
    #[serde(rename = "serviceName")]
    pub service_name: String,
    /// IPv4 address
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ipv4: Option<String>,
    /// Port
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
}

/// Zipkin annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZipkinAnnotation {
    /// Timestamp (microseconds)
    pub timestamp: u64,
    /// Value
    pub value: String,
}

impl ZipkinSpan {
    /// Convert from internal span
    pub fn from_span(span: &Span, config: &TelemetryConfig) -> Self {
        let kind = match span.kind {
            SpanKind::Server => Some("SERVER"),
            SpanKind::Client => Some("CLIENT"),
            SpanKind::Producer => Some("PRODUCER"),
            SpanKind::Consumer => Some("CONSUMER"),
            SpanKind::Internal => None,
        };

        let tags: HashMap<String, String> = span
            .attributes
            .iter()
            .map(|(k, v)| {
                let value_str = match v {
                    AttributeValue::String(s) => s.clone(),
                    AttributeValue::Int(i) => i.to_string(),
                    AttributeValue::Float(f) => f.to_string(),
                    AttributeValue::Bool(b) => b.to_string(),
                    AttributeValue::StringArray(arr) => arr.join(","),
                    AttributeValue::IntArray(arr) => arr
                        .iter()
                        .map(|i| i.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                };
                (k.clone(), value_str)
            })
            .collect();

        let annotations: Vec<ZipkinAnnotation> = span
            .events
            .iter()
            .map(|e| ZipkinAnnotation {
                timestamp: e.timestamp / 1000,
                value: e.name.clone(),
            })
            .collect();

        let duration = span.end_time.unwrap_or(span.start_time) - span.start_time;

        ZipkinSpan {
            trace_id: span.context.trace_id.clone(),
            id: span.context.span_id.clone(),
            parent_id: span.context.parent_span_id.clone(),
            name: span.name.clone(),
            timestamp: span.start_time / 1000,
            duration: duration / 1000,
            kind: kind.map(|s| s.to_string()),
            local_endpoint: ZipkinEndpoint {
                service_name: config.service_name.clone(),
                ipv4: None,
                port: None,
            },
            tags,
            annotations,
        }
    }
}
