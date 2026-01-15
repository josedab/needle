//! OpenTelemetry Integration - Observability for vector database operations.
//!
//! Provides distributed tracing, metrics, and logging following OpenTelemetry standards.
//!
//! # Features
//!
//! - **Distributed tracing**: Track requests across services
//! - **Metrics**: Latency, throughput, error rates
//! - **Custom spans**: Instrument search and indexing operations
//! - **Exporters**: Support for Jaeger, Zipkin, OTLP
//! - **Context propagation**: W3C Trace Context support
//!
//! # Example
//!
//! ```ignore
//! use needle::telemetry::{Telemetry, TelemetryConfig, SpanBuilder};
//!
//! let telemetry = Telemetry::new(TelemetryConfig::default())?;
//!
//! // Create a span for a search operation
//! let span = telemetry.span("vector_search")
//!     .attribute("collection", "embeddings")
//!     .attribute("k", 10)
//!     .start();
//!
//! // Do work...
//!
//! span.end();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Global telemetry instance for easy access
static GLOBAL_TELEMETRY: OnceLock<Arc<Telemetry>> = OnceLock::new();

/// Initialize the global telemetry provider
pub fn init_telemetry(config: TelemetryConfig) -> Arc<Telemetry> {
    let telemetry = Arc::new(Telemetry::new(config));
    let _ = GLOBAL_TELEMETRY.set(telemetry.clone());
    telemetry
}

/// Get the global telemetry instance
pub fn global_telemetry() -> Option<Arc<Telemetry>> {
    GLOBAL_TELEMETRY.get().cloned()
}

/// Convenience macro for creating instrumented spans
#[macro_export]
macro_rules! span {
    ($name:expr) => {
        $crate::telemetry::global_telemetry()
            .map(|t| t.span($name).start())
    };
    ($name:expr, $($key:expr => $value:expr),* $(,)?) => {
        $crate::telemetry::global_telemetry()
            .map(|t| {
                let mut builder = t.span($name);
                $(
                    builder = builder.attribute($key, $value);
                )*
                builder.start()
            })
    };
}

/// Semantic conventions for database operations (OpenTelemetry standard)
pub mod semantic {
    /// Database system name
    pub const DB_SYSTEM: &str = "db.system";
    /// Database name
    pub const DB_NAME: &str = "db.name";
    /// Database operation (e.g., search, insert)
    pub const DB_OPERATION: &str = "db.operation";
    /// Database statement/query
    pub const DB_STATEMENT: &str = "db.statement";
    /// Collection name
    pub const DB_COLLECTION: &str = "db.collection";

    /// Vector database specific attributes
    pub const DB_VECTOR_DIMENSION: &str = "db.vector.dimension";
    pub const DB_VECTOR_K: &str = "db.vector.k";
    pub const DB_VECTOR_DISTANCE: &str = "db.vector.distance_function";
    pub const DB_VECTOR_FILTER: &str = "db.vector.filter";
    pub const DB_VECTOR_RESULTS_COUNT: &str = "db.vector.results_count";
    pub const DB_VECTOR_SEARCH_TYPE: &str = "db.vector.search_type";

    /// Network attributes
    pub const NET_PEER_NAME: &str = "net.peer.name";
    pub const NET_PEER_PORT: &str = "net.peer.port";

    /// Error attributes
    pub const EXCEPTION_TYPE: &str = "exception.type";
    pub const EXCEPTION_MESSAGE: &str = "exception.message";
    pub const EXCEPTION_STACKTRACE: &str = "exception.stacktrace";
}

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

/// Resource attributes describing the service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAttributes {
    /// Service name
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Service namespace
    pub service_namespace: Option<String>,
    /// Service instance ID
    pub service_instance_id: Option<String>,
    /// Deployment environment
    pub deployment_environment: String,
    /// Host name
    pub host_name: Option<String>,
    /// Process runtime
    pub process_runtime: String,
    /// Additional attributes
    pub extra: HashMap<String, String>,
}

impl Default for ResourceAttributes {
    fn default() -> Self {
        Self {
            service_name: "needle".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            service_namespace: None,
            service_instance_id: None,
            deployment_environment: "development".to_string(),
            host_name: std::env::var("HOSTNAME").ok(),
            process_runtime: "rust".to_string(),
            extra: HashMap::new(),
        }
    }
}

impl ResourceAttributes {
    /// Create new resource attributes
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            ..Default::default()
        }
    }

    /// Set service version
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.service_version = version.into();
        self
    }

    /// Set environment
    pub fn with_environment(mut self, env: impl Into<String>) -> Self {
        self.deployment_environment = env.into();
        self
    }

    /// Set namespace
    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.service_namespace = Some(ns.into());
        self
    }

    /// Set instance ID
    pub fn with_instance_id(mut self, id: impl Into<String>) -> Self {
        self.service_instance_id = Some(id.into());
        self
    }

    /// Add extra attribute
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }

    /// Convert to HashMap for export
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("service.name".to_string(), self.service_name.clone());
        map.insert("service.version".to_string(), self.service_version.clone());
        map.insert("deployment.environment".to_string(), self.deployment_environment.clone());
        map.insert("process.runtime.name".to_string(), self.process_runtime.clone());

        if let Some(ref ns) = self.service_namespace {
            map.insert("service.namespace".to_string(), ns.clone());
        }
        if let Some(ref id) = self.service_instance_id {
            map.insert("service.instance.id".to_string(), id.clone());
        }
        if let Some(ref host) = self.host_name {
            map.insert("host.name".to_string(), host.clone());
        }

        map.extend(self.extra.clone());
        map
    }
}

/// Configuration for telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Service name.
    pub service_name: String,
    /// Service version.
    pub service_version: String,
    /// Environment (dev, staging, prod).
    pub environment: String,
    /// Enable tracing.
    pub tracing_enabled: bool,
    /// Enable metrics.
    pub metrics_enabled: bool,
    /// Sample rate (0.0 - 1.0).
    pub sample_rate: f64,
    /// Exporter endpoint.
    pub exporter_endpoint: Option<String>,
    /// Maximum spans to buffer.
    pub max_buffer_size: usize,
    /// Flush interval in milliseconds.
    pub flush_interval_ms: u64,
    /// Export format
    pub export_format: ExportFormat,
    /// Resource attributes
    pub resource: ResourceAttributes,
    /// Propagation format (W3C, B3, Jaeger)
    pub propagation_format: PropagationFormat,
    /// Enable baggage propagation
    pub baggage_enabled: bool,
}

/// Propagation format for distributed tracing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PropagationFormat {
    /// W3C Trace Context (default)
    #[default]
    W3C,
    /// B3 format (Zipkin)
    B3,
    /// B3 Single Header
    B3Single,
    /// Jaeger format
    Jaeger,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "needle".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "development".to_string(),
            tracing_enabled: true,
            metrics_enabled: true,
            sample_rate: 1.0,
            exporter_endpoint: None,
            max_buffer_size: 1000,
            flush_interval_ms: 5000,
            export_format: ExportFormat::default(),
            resource: ResourceAttributes::default(),
            propagation_format: PropagationFormat::default(),
            baggage_enabled: true,
        }
    }
}

impl TelemetryConfig {
    /// Create new configuration with service name
    pub fn new(service_name: impl Into<String>) -> Self {
        let name = service_name.into();
        Self {
            service_name: name.clone(),
            resource: ResourceAttributes::new(name),
            ..Default::default()
        }
    }

    /// Set environment
    pub fn with_environment(mut self, env: impl Into<String>) -> Self {
        let e = env.into();
        self.environment = e.clone();
        self.resource.deployment_environment = e;
        self
    }

    /// Set exporter endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.exporter_endpoint = Some(endpoint.into());
        self
    }

    /// Set export format
    pub fn with_format(mut self, format: ExportFormat) -> Self {
        self.export_format = format;
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Disable tracing
    pub fn disable_tracing(mut self) -> Self {
        self.tracing_enabled = false;
        self
    }

    /// Disable metrics
    pub fn disable_metrics(mut self) -> Self {
        self.metrics_enabled = false;
        self
    }
}

/// Baggage for propagating key-value pairs across service boundaries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Baggage {
    /// Key-value pairs
    pub items: HashMap<String, String>,
}

impl Baggage {
    /// Create new empty baggage
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a baggage item
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.items.insert(key.into(), value.into());
    }

    /// Get a baggage item
    pub fn get(&self, key: &str) -> Option<&String> {
        self.items.get(key)
    }

    /// Remove a baggage item
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.items.remove(key)
    }

    /// Parse from baggage header (key1=value1,key2=value2)
    pub fn from_header(header: &str) -> Self {
        let mut baggage = Self::new();
        for pair in header.split(',') {
            let parts: Vec<&str> = pair.trim().splitn(2, '=').collect();
            if parts.len() == 2 {
                baggage.set(parts[0].trim(), parts[1].trim());
            }
        }
        baggage
    }

    /// Convert to baggage header
    pub fn to_header(&self) -> String {
        self.items
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Trace context for distributed tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Trace ID (128-bit as hex string).
    pub trace_id: String,
    /// Span ID (64-bit as hex string).
    pub span_id: String,
    /// Parent span ID.
    pub parent_span_id: Option<String>,
    /// Trace flags.
    pub trace_flags: u8,
    /// Baggage items
    #[serde(default)]
    pub baggage: Baggage,
}

impl TraceContext {
    /// Generate new trace context.
    pub fn new() -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();

        let mut hasher = DefaultHasher::new();
        now.hash(&mut hasher);
        let trace_id = format!("{:032x}", hasher.finish() as u128);

        let mut hasher = DefaultHasher::new();
        (now + 1).hash(&mut hasher);
        let span_id = format!("{:016x}", hasher.finish());

        Self {
            trace_id,
            span_id,
            parent_span_id: None,
            trace_flags: 1, // Sampled
            baggage: Baggage::default(),
        }
    }

    /// Create child context.
    pub fn child(&self) -> Self {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos();

        let mut hasher = DefaultHasher::new();
        now.hash(&mut hasher);
        let span_id = format!("{:016x}", hasher.finish());

        Self {
            trace_id: self.trace_id.clone(),
            span_id,
            parent_span_id: Some(self.span_id.clone()),
            trace_flags: self.trace_flags,
            baggage: self.baggage.clone(),
        }
    }

    /// Parse from W3C traceparent header.
    pub fn from_traceparent(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() != 4 {
            return None;
        }

        Some(Self {
            trace_id: parts[1].to_string(),
            span_id: parts[2].to_string(),
            parent_span_id: None,
            trace_flags: u8::from_str_radix(parts[3], 16).unwrap_or(0),
            baggage: Baggage::default(),
        })
    }

    /// Convert to W3C traceparent header.
    pub fn to_traceparent(&self) -> String {
        format!("00-{}-{}-{:02x}", self.trace_id, self.span_id, self.trace_flags)
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Span status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Unset status.
    Unset,
    /// Operation succeeded.
    Ok,
    /// Operation failed.
    Error,
}

/// A span representing a unit of work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Span name.
    pub name: String,
    /// Trace context.
    pub context: TraceContext,
    /// Start time (Unix timestamp in nanoseconds).
    pub start_time: u64,
    /// End time (Unix timestamp in nanoseconds).
    pub end_time: Option<u64>,
    /// Span attributes.
    pub attributes: HashMap<String, AttributeValue>,
    /// Span events.
    pub events: Vec<SpanEvent>,
    /// Span status.
    pub status: SpanStatus,
    /// Status message.
    pub status_message: Option<String>,
    /// Span kind.
    pub kind: SpanKind,
}

impl Span {
    /// Create a new span.
    pub fn new(name: &str, context: TraceContext) -> Self {
        Self {
            name: name.to_string(),
            context,
            start_time: Self::now_nanos(),
            end_time: None,
            attributes: HashMap::new(),
            events: Vec::new(),
            status: SpanStatus::Unset,
            status_message: None,
            kind: SpanKind::Internal,
        }
    }

    fn now_nanos() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after Unix epoch")
            .as_nanos() as u64
    }

    /// Set an attribute.
    pub fn set_attribute(&mut self, key: &str, value: impl Into<AttributeValue>) {
        self.attributes.insert(key.to_string(), value.into());
    }

    /// Add an event.
    pub fn add_event(&mut self, name: &str) {
        self.events.push(SpanEvent {
            name: name.to_string(),
            timestamp: Self::now_nanos(),
            attributes: HashMap::new(),
        });
    }

    /// Add an event with attributes.
    pub fn add_event_with_attributes(&mut self, name: &str, attrs: HashMap<String, AttributeValue>) {
        self.events.push(SpanEvent {
            name: name.to_string(),
            timestamp: Self::now_nanos(),
            attributes: attrs,
        });
    }

    /// Set status to OK.
    pub fn set_ok(&mut self) {
        self.status = SpanStatus::Ok;
    }

    /// Set status to Error.
    pub fn set_error(&mut self, message: &str) {
        self.status = SpanStatus::Error;
        self.status_message = Some(message.to_string());
    }

    /// End the span.
    pub fn end(&mut self) {
        self.end_time = Some(Self::now_nanos());
    }

    /// Get duration in milliseconds.
    pub fn duration_ms(&self) -> Option<f64> {
        self.end_time.map(|end| (end - self.start_time) as f64 / 1_000_000.0)
    }
}

/// Span event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Event name.
    pub name: String,
    /// Timestamp.
    pub timestamp: u64,
    /// Event attributes.
    pub attributes: HashMap<String, AttributeValue>,
}

/// Span kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Internal operation.
    Internal,
    /// Server handling request.
    Server,
    /// Client making request.
    Client,
    /// Producer sending message.
    Producer,
    /// Consumer receiving message.
    Consumer,
}

/// Attribute value types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// String array.
    StringArray(Vec<String>),
    /// Integer array.
    IntArray(Vec<i64>),
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        AttributeValue::String(s.to_string())
    }
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        AttributeValue::String(s)
    }
}

impl From<i64> for AttributeValue {
    fn from(v: i64) -> Self {
        AttributeValue::Int(v)
    }
}

impl From<i32> for AttributeValue {
    fn from(v: i32) -> Self {
        AttributeValue::Int(v as i64)
    }
}

impl From<usize> for AttributeValue {
    fn from(v: usize) -> Self {
        AttributeValue::Int(v as i64)
    }
}

impl From<f64> for AttributeValue {
    fn from(v: f64) -> Self {
        AttributeValue::Float(v)
    }
}

impl From<bool> for AttributeValue {
    fn from(v: bool) -> Self {
        AttributeValue::Bool(v)
    }
}

/// Metric types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter (monotonically increasing).
    Counter(u64),
    /// Gauge (can go up or down).
    Gauge(f64),
    /// Histogram (distribution).
    Histogram(Vec<f64>),
}

/// A metric measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name.
    pub name: String,
    /// Metric value.
    pub value: MetricValue,
    /// Metric labels.
    pub labels: HashMap<String, String>,
    /// Timestamp.
    pub timestamp: u64,
}

/// Telemetry system.
pub struct Telemetry {
    /// Configuration.
    config: TelemetryConfig,
    /// Completed spans buffer.
    spans: Arc<RwLock<Vec<Span>>>,
    /// Metrics buffer.
    metrics: Arc<RwLock<Vec<Metric>>>,
    /// Counter metrics.
    counters: Arc<RwLock<HashMap<String, u64>>>,
    /// Histogram metrics.
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl Telemetry {
    /// Create new telemetry system.
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            config,
            spans: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(Vec::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Start a new span.
    pub fn span(&self, name: &str) -> SpanBuilder<'_> {
        SpanBuilder::new(self, name)
    }

    /// Start a span with existing context.
    pub fn span_with_context(&self, name: &str, context: TraceContext) -> SpanBuilder<'_> {
        SpanBuilder::with_context(self, name, context)
    }

    /// Record a completed span.
    pub fn record_span(&self, span: Span) {
        if !self.config.tracing_enabled {
            return;
        }

        if let Ok(mut spans) = self.spans.write() {
            spans.push(span);
            if spans.len() > self.config.max_buffer_size {
                spans.remove(0);
            }
        }
    }

    /// Increment a counter.
    pub fn counter(&self, name: &str, value: u64, labels: HashMap<String, String>) {
        if !self.config.metrics_enabled {
            return;
        }

        if let Ok(mut counters) = self.counters.write() {
            *counters.entry(name.to_string()).or_default() += value;
        }

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.push(Metric {
                name: name.to_string(),
                value: MetricValue::Counter(value),
                labels,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock should be after Unix epoch")
                    .as_nanos() as u64,
            });
        }
    }

    /// Record a gauge value.
    pub fn gauge(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        if !self.config.metrics_enabled {
            return;
        }

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.push(Metric {
                name: name.to_string(),
                value: MetricValue::Gauge(value),
                labels,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock should be after Unix epoch")
                    .as_nanos() as u64,
            });
        }
    }

    /// Record a histogram value.
    pub fn histogram(&self, name: &str, value: f64, labels: HashMap<String, String>) {
        if !self.config.metrics_enabled {
            return;
        }

        if let Ok(mut histograms) = self.histograms.write() {
            histograms.entry(name.to_string()).or_default().push(value);
        }

        if let Ok(mut metrics) = self.metrics.write() {
            metrics.push(Metric {
                name: name.to_string(),
                value: MetricValue::Histogram(vec![value]),
                labels,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock should be after Unix epoch")
                    .as_nanos() as u64,
            });
        }
    }

    /// Record latency (convenience method).
    pub fn record_latency(&self, operation: &str, duration: Duration, success: bool) {
        let mut labels = HashMap::new();
        labels.insert("operation".to_string(), operation.to_string());
        labels.insert("success".to_string(), success.to_string());

        self.histogram(
            &format!("{}_latency_ms", operation),
            duration.as_secs_f64() * 1000.0,
            labels,
        );
    }

    /// Get recent spans.
    pub fn get_spans(&self) -> Vec<Span> {
        self.spans.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Get counter value.
    pub fn get_counter(&self, name: &str) -> u64 {
        self.counters.read()
            .map(|c| *c.get(name).unwrap_or(&0))
            .unwrap_or(0)
    }

    /// Get histogram values.
    pub fn get_histogram(&self, name: &str) -> Vec<f64> {
        self.histograms.read()
            .map(|h| h.get(name).cloned().unwrap_or_default())
            .unwrap_or_default()
    }

    /// Get histogram statistics.
    pub fn histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        let values = self.get_histogram(name);
        if values.is_empty() {
            return None;
        }

        let count = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / count as f64;

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[count - 1];
        let p50 = sorted[count / 2];
        let p95 = sorted[(count as f64 * 0.95) as usize];
        let p99 = sorted[(count as f64 * 0.99) as usize];

        Some(HistogramStats {
            count,
            sum,
            mean,
            min,
            max,
            p50,
            p95,
            p99,
        })
    }

    /// Clear all recorded data.
    pub fn clear(&self) {
        if let Ok(mut spans) = self.spans.write() {
            spans.clear();
        }
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.clear();
        }
        if let Ok(mut counters) = self.counters.write() {
            counters.clear();
        }
        if let Ok(mut histograms) = self.histograms.write() {
            histograms.clear();
        }
    }

    /// Export spans as JSON.
    pub fn export_spans_json(&self) -> String {
        let spans = self.get_spans();
        serde_json::to_string_pretty(&spans).unwrap_or_default()
    }

    /// Get telemetry configuration.
    pub fn config(&self) -> &TelemetryConfig {
        &self.config
    }
}

impl Default for Telemetry {
    fn default() -> Self {
        Self::new(TelemetryConfig::default())
    }
}

/// Histogram statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    /// Sample count.
    pub count: usize,
    /// Sum of all values.
    pub sum: f64,
    /// Mean value.
    pub mean: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// 50th percentile.
    pub p50: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

/// Builder for creating spans.
pub struct SpanBuilder<'a> {
    telemetry: &'a Telemetry,
    name: String,
    context: TraceContext,
    attributes: HashMap<String, AttributeValue>,
    kind: SpanKind,
}

impl<'a> SpanBuilder<'a> {
    /// Create new span builder.
    fn new(telemetry: &'a Telemetry, name: &str) -> Self {
        Self {
            telemetry,
            name: name.to_string(),
            context: TraceContext::new(),
            attributes: HashMap::new(),
            kind: SpanKind::Internal,
        }
    }

    /// Create span builder with context.
    fn with_context(telemetry: &'a Telemetry, name: &str, context: TraceContext) -> Self {
        Self {
            telemetry,
            name: name.to_string(),
            context: context.child(),
            attributes: HashMap::new(),
            kind: SpanKind::Internal,
        }
    }

    /// Add an attribute.
    pub fn attribute(mut self, key: &str, value: impl Into<AttributeValue>) -> Self {
        self.attributes.insert(key.to_string(), value.into());
        self
    }

    /// Set span kind.
    pub fn kind(mut self, kind: SpanKind) -> Self {
        self.kind = kind;
        self
    }

    /// Start the span.
    pub fn start(self) -> ActiveSpan<'a> {
        let mut span = Span::new(&self.name, self.context);
        span.attributes = self.attributes;
        span.kind = self.kind;

        ActiveSpan {
            telemetry: self.telemetry,
            span,
            start: Instant::now(),
        }
    }
}

/// An active span that records itself on drop.
pub struct ActiveSpan<'a> {
    telemetry: &'a Telemetry,
    span: Span,
    start: Instant,
}

impl<'a> ActiveSpan<'a> {
    /// Get the trace context.
    pub fn context(&self) -> &TraceContext {
        &self.span.context
    }

    /// Add an attribute.
    pub fn set_attribute(&mut self, key: &str, value: impl Into<AttributeValue>) {
        self.span.set_attribute(key, value);
    }

    /// Add an event.
    pub fn add_event(&mut self, name: &str) {
        self.span.add_event(name);
    }

    /// Set status to OK.
    pub fn set_ok(&mut self) {
        self.span.set_ok();
    }

    /// Set status to Error.
    pub fn set_error(&mut self, message: &str) {
        self.span.set_error(message);
    }

    /// End the span and record it.
    pub fn end(mut self) {
        self.span.end();
        self.telemetry.record_span(self.span.clone());
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl<'a> Drop for ActiveSpan<'a> {
    fn drop(&mut self) {
        if self.span.end_time.is_none() {
            self.span.end();
            self.telemetry.record_span(self.span.clone());
        }
    }
}

// ============================================================================
// Propagation Format Support (B3, Jaeger)
// ============================================================================

impl TraceContext {
    /// Parse from B3 multi-header format.
    /// Headers: X-B3-TraceId, X-B3-SpanId, X-B3-ParentSpanId, X-B3-Sampled, X-B3-Flags
    pub fn from_b3_headers(
        trace_id: Option<&str>,
        span_id: Option<&str>,
        parent_span_id: Option<&str>,
        sampled: Option<&str>,
    ) -> Option<Self> {
        let trace_id = trace_id?;
        let span_id = span_id?;

        // Pad trace_id to 32 chars if it's 16 (short form)
        let trace_id = if trace_id.len() == 16 {
            format!("0000000000000000{}", trace_id)
        } else {
            trace_id.to_string()
        };

        let trace_flags = match sampled {
            Some("1") | Some("true") => 1,
            Some("0") | Some("false") => 0,
            _ => 1,
        };

        Some(Self {
            trace_id,
            span_id: span_id.to_string(),
            parent_span_id: parent_span_id.map(|s| s.to_string()),
            trace_flags,
            baggage: Baggage::default(),
        })
    }

    /// Parse from B3 single-header format.
    /// Format: {TraceId}-{SpanId}-{SamplingState}-{ParentSpanId}
    pub fn from_b3_single(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split('-').collect();
        if parts.len() < 2 {
            return None;
        }

        let trace_id = if parts[0].len() == 16 {
            format!("0000000000000000{}", parts[0])
        } else {
            parts[0].to_string()
        };

        let span_id = parts[1].to_string();

        let trace_flags = if parts.len() > 2 {
            match parts[2] {
                "1" | "d" => 1,
                _ => 0,
            }
        } else {
            1
        };

        let parent_span_id = if parts.len() > 3 {
            Some(parts[3].to_string())
        } else {
            None
        };

        Some(Self {
            trace_id,
            span_id,
            parent_span_id,
            trace_flags,
            baggage: Baggage::default(),
        })
    }

    /// Convert to B3 headers.
    pub fn to_b3_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert("X-B3-TraceId".to_string(), self.trace_id.clone());
        headers.insert("X-B3-SpanId".to_string(), self.span_id.clone());
        if let Some(ref parent) = self.parent_span_id {
            headers.insert("X-B3-ParentSpanId".to_string(), parent.clone());
        }
        headers.insert(
            "X-B3-Sampled".to_string(),
            if self.trace_flags & 1 == 1 { "1" } else { "0" }.to_string(),
        );
        headers
    }

    /// Convert to B3 single header.
    pub fn to_b3_single(&self) -> String {
        let sampled = if self.trace_flags & 1 == 1 { "1" } else { "0" };
        match &self.parent_span_id {
            Some(parent) => format!("{}-{}-{}-{}", self.trace_id, self.span_id, sampled, parent),
            None => format!("{}-{}-{}", self.trace_id, self.span_id, sampled),
        }
    }

    /// Parse from Jaeger header format.
    /// Format: {trace-id}:{span-id}:{parent-span-id}:{flags}
    pub fn from_jaeger_header(header: &str) -> Option<Self> {
        let parts: Vec<&str> = header.split(':').collect();
        if parts.len() != 4 {
            return None;
        }

        let trace_id = if parts[0].len() == 16 {
            format!("0000000000000000{}", parts[0])
        } else {
            parts[0].to_string()
        };

        let parent_span_id = if parts[2] == "0" {
            None
        } else {
            Some(parts[2].to_string())
        };

        let trace_flags = u8::from_str_radix(parts[3], 16).unwrap_or(0);

        Some(Self {
            trace_id,
            span_id: parts[1].to_string(),
            parent_span_id,
            trace_flags,
            baggage: Baggage::default(),
        })
    }

    /// Convert to Jaeger header format.
    pub fn to_jaeger_header(&self) -> String {
        let parent = self.parent_span_id.as_deref().unwrap_or("0");
        format!(
            "{}:{}:{}:{:x}",
            self.trace_id, self.span_id, parent, self.trace_flags
        )
    }

    /// Parse from headers using the specified propagation format.
    pub fn from_headers(
        headers: &HashMap<String, String>,
        format: PropagationFormat,
    ) -> Option<Self> {
        match format {
            PropagationFormat::W3C => {
                headers.get("traceparent").and_then(|h| Self::from_traceparent(h))
            }
            PropagationFormat::B3 => Self::from_b3_headers(
                headers.get("X-B3-TraceId").map(|s| s.as_str()),
                headers.get("X-B3-SpanId").map(|s| s.as_str()),
                headers.get("X-B3-ParentSpanId").map(|s| s.as_str()),
                headers.get("X-B3-Sampled").map(|s| s.as_str()),
            ),
            PropagationFormat::B3Single => {
                headers.get("b3").and_then(|h| Self::from_b3_single(h))
            }
            PropagationFormat::Jaeger => {
                headers.get("uber-trace-id").and_then(|h| Self::from_jaeger_header(h))
            }
        }
    }

    /// Convert to headers using the specified propagation format.
    pub fn to_headers(&self, format: PropagationFormat) -> HashMap<String, String> {
        match format {
            PropagationFormat::W3C => {
                let mut headers = HashMap::new();
                headers.insert("traceparent".to_string(), self.to_traceparent());
                headers
            }
            PropagationFormat::B3 => self.to_b3_headers(),
            PropagationFormat::B3Single => {
                let mut headers = HashMap::new();
                headers.insert("b3".to_string(), self.to_b3_single());
                headers
            }
            PropagationFormat::Jaeger => {
                let mut headers = HashMap::new();
                headers.insert("uber-trace-id".to_string(), self.to_jaeger_header());
                headers
            }
        }
    }
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
                string_value: arr.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
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
            .config
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
                trace_id: spans.first().map(|s| s.context.trace_id.clone()).unwrap_or_default(),
                spans: jaeger_spans,
                processes: {
                    let mut map = HashMap::new();
                    map.insert(
                        "p1".to_string(),
                        JaegerProcess {
                            service_name: self.config.service_name.clone(),
                            tags: vec![
                                JaegerTag {
                                    key: "service.version".to_string(),
                                    tag_type: "string".to_string(),
                                    value: JaegerTagValue::String(self.config.service_version.clone()),
                                },
                                JaegerTag {
                                    key: "deployment.environment".to_string(),
                                    tag_type: "string".to_string(),
                                    value: JaegerTagValue::String(self.config.environment.clone()),
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
        spans.iter().map(|s| ZipkinSpan::from_span(s, &self.config)).collect()
    }

    /// Export spans in Zipkin JSON string format.
    pub fn export_zipkin_json(&self) -> String {
        let zipkin = self.export_zipkin();
        serde_json::to_string_pretty(&zipkin).unwrap_or_default()
    }

    /// Export in configured format.
    pub fn export(&self) -> String {
        match self.config.export_format {
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
                status_icon, span.name, duration_ms,
                &span.context.trace_id[..8], &span.context.span_id[..8]
            ));

            for (key, value) in &span.attributes {
                let value_str = match value {
                    AttributeValue::String(s) => s.clone(),
                    AttributeValue::Int(i) => i.to_string(),
                    AttributeValue::Float(f) => f.to_string(),
                    AttributeValue::Bool(b) => b.to_string(),
                    AttributeValue::StringArray(arr) => arr.join(", "),
                    AttributeValue::IntArray(arr) => arr.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "),
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
                    AttributeValue::IntArray(arr) => {
                        JaegerTagValue::String(arr.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","))
                    }
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
                    AttributeValue::IntArray(arr) => {
                        arr.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
                    }
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

/// Instrumented operation timer.
pub struct OperationTimer<'a> {
    telemetry: &'a Telemetry,
    operation: String,
    start: Instant,
    success: bool,
}

impl<'a> OperationTimer<'a> {
    /// Create a new timer.
    pub fn new(telemetry: &'a Telemetry, operation: &str) -> Self {
        Self {
            telemetry,
            operation: operation.to_string(),
            start: Instant::now(),
            success: true,
        }
    }

    /// Mark as failed.
    pub fn fail(&mut self) {
        self.success = false;
    }

    /// Record and return elapsed time.
    pub fn stop(self) -> Duration {
        let duration = self.start.elapsed();
        self.telemetry.record_latency(&self.operation, duration, self.success);
        duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_telemetry() {
        let telemetry = Telemetry::new(TelemetryConfig::default());
        assert!(telemetry.config.tracing_enabled);
    }

    #[test]
    fn test_trace_context() {
        let ctx = TraceContext::new();
        assert_eq!(ctx.trace_id.len(), 32);
        assert_eq!(ctx.span_id.len(), 16);

        let child = ctx.child();
        assert_eq!(child.trace_id, ctx.trace_id);
        assert_eq!(child.parent_span_id, Some(ctx.span_id));
    }

    #[test]
    fn test_traceparent() {
        let ctx = TraceContext::new();
        let header = ctx.to_traceparent();

        assert!(header.starts_with("00-"));

        let parsed = TraceContext::from_traceparent(&header).unwrap();
        assert_eq!(parsed.trace_id, ctx.trace_id);
    }

    #[test]
    fn test_span_creation() {
        let telemetry = Telemetry::default();

        let span = telemetry.span("test_op")
            .attribute("key", "value")
            .attribute("count", 42i64)
            .start();

        span.end();

        let spans = telemetry.get_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "test_op");
    }

    #[test]
    fn test_span_events() {
        let telemetry = Telemetry::default();

        let mut span = telemetry.span("test").start();
        span.add_event("started");
        span.add_event("processing");
        span.add_event("completed");
        span.end();

        let spans = telemetry.get_spans();
        assert_eq!(spans[0].events.len(), 3);
    }

    #[test]
    fn test_span_status() {
        let telemetry = Telemetry::default();

        let mut span = telemetry.span("test").start();
        span.set_error("Something went wrong");
        span.end();

        let spans = telemetry.get_spans();
        assert_eq!(spans[0].status, SpanStatus::Error);
        assert_eq!(spans[0].status_message, Some("Something went wrong".to_string()));
    }

    #[test]
    fn test_counter() {
        let telemetry = Telemetry::default();

        telemetry.counter("requests", 1, HashMap::new());
        telemetry.counter("requests", 1, HashMap::new());
        telemetry.counter("requests", 1, HashMap::new());

        assert_eq!(telemetry.get_counter("requests"), 3);
    }

    #[test]
    fn test_histogram() {
        let telemetry = Telemetry::default();

        for i in 0..100 {
            telemetry.histogram("latency", i as f64, HashMap::new());
        }

        let values = telemetry.get_histogram("latency");
        assert_eq!(values.len(), 100);

        let stats = telemetry.histogram_stats("latency").unwrap();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 99.0);
    }

    #[test]
    fn test_record_latency() {
        let telemetry = Telemetry::default();

        telemetry.record_latency("search", Duration::from_millis(50), true);
        telemetry.record_latency("search", Duration::from_millis(100), true);

        let values = telemetry.get_histogram("search_latency_ms");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_span_auto_end() {
        let telemetry = Telemetry::default();

        {
            let _span = telemetry.span("auto_end").start();
            // Span ends automatically when dropped
        }

        let spans = telemetry.get_spans();
        assert_eq!(spans.len(), 1);
        assert!(spans[0].end_time.is_some());
    }

    #[test]
    fn test_child_span() {
        let telemetry = Telemetry::default();

        let parent = telemetry.span("parent").start();
        let ctx = parent.context().clone();
        parent.end();

        let child = telemetry.span_with_context("child", ctx).start();
        child.end();

        let spans = telemetry.get_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].context.trace_id, spans[1].context.trace_id);
    }

    #[test]
    fn test_clear() {
        let telemetry = Telemetry::default();

        telemetry.span("test").start().end();
        telemetry.counter("count", 1, HashMap::new());

        assert!(!telemetry.get_spans().is_empty());

        telemetry.clear();

        assert!(telemetry.get_spans().is_empty());
        assert_eq!(telemetry.get_counter("count"), 0);
    }

    #[test]
    fn test_operation_timer() {
        let telemetry = Telemetry::default();

        let timer = OperationTimer::new(&telemetry, "operation");
        std::thread::sleep(Duration::from_millis(10));
        let duration = timer.stop();

        assert!(duration.as_millis() >= 10);

        let values = telemetry.get_histogram("operation_latency_ms");
        assert_eq!(values.len(), 1);
    }

    #[test]
    fn test_span_kind() {
        let telemetry = Telemetry::default();

        let span = telemetry.span("server")
            .kind(SpanKind::Server)
            .start();
        span.end();

        let spans = telemetry.get_spans();
        assert_eq!(spans[0].kind, SpanKind::Server);
    }

    #[test]
    fn test_disabled_tracing() {
        let config = TelemetryConfig {
            tracing_enabled: false,
            ..Default::default()
        };
        let telemetry = Telemetry::new(config);

        telemetry.span("test").start().end();

        let spans = telemetry.get_spans();
        assert!(spans.is_empty());
    }

    #[test]
    fn test_export_json() {
        let telemetry = Telemetry::default();

        telemetry.span("test")
            .attribute("key", "value")
            .start()
            .end();

        let json = telemetry.export_spans_json();
        assert!(json.contains("test"));
        assert!(json.contains("key"));
    }

    // ============================================================================
    // Propagation Format Tests
    // ============================================================================

    #[test]
    fn test_b3_single_header() {
        let ctx = TraceContext::new();
        let header = ctx.to_b3_single();

        // Should be in format: trace_id-span_id-sampled
        assert!(header.contains('-'));

        let parsed = TraceContext::from_b3_single(&header).unwrap();
        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.span_id, ctx.span_id);
    }

    #[test]
    fn test_b3_multi_headers() {
        let ctx = TraceContext::new();
        let headers = ctx.to_b3_headers();

        assert!(headers.contains_key("X-B3-TraceId"));
        assert!(headers.contains_key("X-B3-SpanId"));
        assert!(headers.contains_key("X-B3-Sampled"));

        let parsed = TraceContext::from_b3_headers(
            headers.get("X-B3-TraceId").map(|s| s.as_str()),
            headers.get("X-B3-SpanId").map(|s| s.as_str()),
            headers.get("X-B3-ParentSpanId").map(|s| s.as_str()),
            headers.get("X-B3-Sampled").map(|s| s.as_str()),
        ).unwrap();

        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.span_id, ctx.span_id);
    }

    #[test]
    fn test_jaeger_header() {
        let ctx = TraceContext::new();
        let header = ctx.to_jaeger_header();

        // Should be in format: trace_id:span_id:parent_span_id:flags
        assert!(header.contains(':'));

        let parsed = TraceContext::from_jaeger_header(&header).unwrap();
        assert_eq!(parsed.trace_id, ctx.trace_id);
        assert_eq!(parsed.span_id, ctx.span_id);
    }

    #[test]
    fn test_propagation_formats() {
        let ctx = TraceContext::new();

        // Test W3C format
        let w3c_headers = ctx.to_headers(PropagationFormat::W3C);
        assert!(w3c_headers.contains_key("traceparent"));
        let parsed_w3c = TraceContext::from_headers(&w3c_headers, PropagationFormat::W3C).unwrap();
        assert_eq!(parsed_w3c.trace_id, ctx.trace_id);

        // Test B3 format
        let b3_headers = ctx.to_headers(PropagationFormat::B3);
        let parsed_b3 = TraceContext::from_headers(&b3_headers, PropagationFormat::B3).unwrap();
        assert_eq!(parsed_b3.trace_id, ctx.trace_id);

        // Test B3 Single format
        let b3s_headers = ctx.to_headers(PropagationFormat::B3Single);
        let parsed_b3s = TraceContext::from_headers(&b3s_headers, PropagationFormat::B3Single).unwrap();
        assert_eq!(parsed_b3s.trace_id, ctx.trace_id);

        // Test Jaeger format
        let jaeger_headers = ctx.to_headers(PropagationFormat::Jaeger);
        let parsed_jaeger = TraceContext::from_headers(&jaeger_headers, PropagationFormat::Jaeger).unwrap();
        assert_eq!(parsed_jaeger.trace_id, ctx.trace_id);
    }

    // ============================================================================
    // Baggage Tests
    // ============================================================================

    #[test]
    fn test_baggage() {
        let mut baggage = Baggage::new();
        baggage.set("user_id", "12345");
        baggage.set("tenant", "acme");

        assert_eq!(baggage.get("user_id"), Some(&"12345".to_string()));
        assert_eq!(baggage.get("tenant"), Some(&"acme".to_string()));

        baggage.remove("user_id");
        assert!(baggage.get("user_id").is_none());
    }

    #[test]
    fn test_baggage_header() {
        let mut baggage = Baggage::new();
        baggage.set("key1", "value1");
        baggage.set("key2", "value2");

        let header = baggage.to_header();
        assert!(header.contains("key1=value1"));
        assert!(header.contains("key2=value2"));

        let parsed = Baggage::from_header(&header);
        assert_eq!(parsed.get("key1"), Some(&"value1".to_string()));
        assert_eq!(parsed.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_baggage_propagation() {
        let mut ctx = TraceContext::new();
        ctx.baggage.set("request_id", "abc-123");
        ctx.baggage.set("user", "john");

        let child = ctx.child();
        // Baggage should be propagated to child
        assert_eq!(child.baggage.get("request_id"), Some(&"abc-123".to_string()));
        assert_eq!(child.baggage.get("user"), Some(&"john".to_string()));
    }

    // ============================================================================
    // Export Format Tests
    // ============================================================================

    #[test]
    fn test_export_otlp() {
        let telemetry = Telemetry::default();

        telemetry.span("test_operation")
            .attribute("db.system", "needle")
            .attribute("db.operation", "search")
            .kind(SpanKind::Server)
            .start()
            .end();

        let otlp = telemetry.export_otlp();
        assert!(!otlp.scope_spans.is_empty());
        assert!(!otlp.scope_spans[0].spans.is_empty());
        assert_eq!(otlp.scope_spans[0].spans[0].name, "test_operation");
        assert_eq!(otlp.scope_spans[0].spans[0].kind, 2); // Server = 2
    }

    #[test]
    fn test_export_otlp_json() {
        let telemetry = Telemetry::default();

        telemetry.span("search")
            .attribute("k", 10i64)
            .start()
            .end();

        let json = telemetry.export_otlp_json();
        assert!(json.contains("search"));
        assert!(json.contains("scope_spans"));
        assert!(json.contains("resource"));
    }

    #[test]
    fn test_export_jaeger() {
        let telemetry = Telemetry::default();

        let mut span = telemetry.span("jaeger_test")
            .attribute("service", "needle")
            .start();
        span.add_event("processing");
        span.end();

        let jaeger = telemetry.export_jaeger();
        assert!(!jaeger.data.is_empty());
        assert!(!jaeger.data[0].spans.is_empty());
        assert_eq!(jaeger.data[0].spans[0].operation_name, "jaeger_test");
        assert!(!jaeger.data[0].spans[0].logs.is_empty()); // Has events
    }

    #[test]
    fn test_export_zipkin() {
        let telemetry = Telemetry::default();

        telemetry.span("zipkin_test")
            .attribute("http.method", "GET")
            .kind(SpanKind::Client)
            .start()
            .end();

        let zipkin = telemetry.export_zipkin();
        assert!(!zipkin.is_empty());
        assert_eq!(zipkin[0].name, "zipkin_test");
        assert_eq!(zipkin[0].kind, Some("CLIENT".to_string()));
    }

    #[test]
    fn test_export_console() {
        let telemetry = Telemetry::default();

        let mut span = telemetry.span("console_test")
            .attribute("key", "value")
            .start();
        span.set_ok();
        span.end();

        let output = telemetry.export_console();
        assert!(output.contains("console_test"));
        assert!(output.contains("key: value"));
        assert!(output.contains("✓")); // OK status
    }

    #[test]
    fn test_export_by_format() {
        let config = TelemetryConfig::new("test")
            .with_format(ExportFormat::Zipkin);
        let telemetry = Telemetry::new(config);

        telemetry.span("format_test").start().end();

        let export = telemetry.export();
        // Zipkin format returns array
        assert!(export.contains("["));
        assert!(export.contains("format_test"));
    }

    // ============================================================================
    // Resource Attributes Tests
    // ============================================================================

    #[test]
    fn test_resource_attributes() {
        let resource = ResourceAttributes::new("my-service")
            .with_version("1.2.3")
            .with_environment("production")
            .with_namespace("backend")
            .with_instance_id("instance-1")
            .with_attribute("custom.key", "custom.value");

        let map = resource.to_map();
        assert_eq!(map.get("service.name"), Some(&"my-service".to_string()));
        assert_eq!(map.get("service.version"), Some(&"1.2.3".to_string()));
        assert_eq!(map.get("deployment.environment"), Some(&"production".to_string()));
        assert_eq!(map.get("service.namespace"), Some(&"backend".to_string()));
        assert_eq!(map.get("service.instance.id"), Some(&"instance-1".to_string()));
        assert_eq!(map.get("custom.key"), Some(&"custom.value".to_string()));
    }

    #[test]
    fn test_telemetry_config_builder() {
        let config = TelemetryConfig::new("my-app")
            .with_environment("staging")
            .with_endpoint("http://localhost:4317")
            .with_format(ExportFormat::Otlp)
            .with_sample_rate(0.5)
            .disable_metrics();

        assert_eq!(config.service_name, "my-app");
        assert_eq!(config.environment, "staging");
        assert_eq!(config.exporter_endpoint, Some("http://localhost:4317".to_string()));
        assert_eq!(config.export_format, ExportFormat::Otlp);
        assert_eq!(config.sample_rate, 0.5);
        assert!(!config.metrics_enabled);
    }

    #[test]
    fn test_semantic_conventions() {
        // Verify semantic constants are correct OpenTelemetry standards
        assert_eq!(semantic::DB_SYSTEM, "db.system");
        assert_eq!(semantic::DB_NAME, "db.name");
        assert_eq!(semantic::DB_OPERATION, "db.operation");
        assert_eq!(semantic::DB_COLLECTION, "db.collection");

        // Vector DB specific
        assert_eq!(semantic::DB_VECTOR_DIMENSION, "db.vector.dimension");
        assert_eq!(semantic::DB_VECTOR_K, "db.vector.k");
    }
}
