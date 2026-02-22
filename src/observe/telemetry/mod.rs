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

pub mod exporters;
pub mod metrics;
pub mod spans;

// Re-export all public types from submodules
pub use exporters::{
    ExportFormat, JaegerExport, JaegerLog, JaegerProcess, JaegerReference, JaegerSpan, JaegerTag,
    JaegerTagValue, JaegerTrace, OtlpAnyValue, OtlpAttribute, OtlpEvent, OtlpResource,
    OtlpResourceSpans, OtlpScope, OtlpScopeSpans, OtlpSpan, OtlpStatus, ZipkinAnnotation,
    ZipkinEndpoint, ZipkinSpan,
};
pub use metrics::{HistogramStats, Metric, MetricValue};
pub use spans::{
    ActiveSpan, AttributeValue, OperationTimer, Span, SpanBuilder, SpanEvent, SpanKind,
    SpanStatus,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Global telemetry instance for easy access
static GLOBAL_TELEMETRY: OnceLock<Arc<Telemetry>> = OnceLock::new();

/// Initialize the global telemetry provider
pub fn init_telemetry(config: TelemetryConfig) -> Arc<Telemetry> {
    let telemetry = Arc::new(Telemetry::new(config));
    if GLOBAL_TELEMETRY.set(telemetry.clone()).is_err() {
        tracing::debug!("Global telemetry already initialized, ignoring re-init");
    }
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
        map.insert(
            "deployment.environment".to_string(),
            self.deployment_environment.clone(),
        );
        map.insert(
            "process.runtime.name".to_string(),
            self.process_runtime.clone(),
        );

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
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let trace_id = format!("{:032x}", rng.gen::<u128>());
        let span_id = format!("{:016x}", rng.gen::<u64>());

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
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
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
        format!(
            "00-{}-{}-{:02x}",
            self.trace_id, self.span_id, self.trace_flags
        )
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
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
            PropagationFormat::W3C => headers
                .get("traceparent")
                .and_then(|h| Self::from_traceparent(h)),
            PropagationFormat::B3 => Self::from_b3_headers(
                headers.get("X-B3-TraceId").map(|s| s.as_str()),
                headers.get("X-B3-SpanId").map(|s| s.as_str()),
                headers.get("X-B3-ParentSpanId").map(|s| s.as_str()),
                headers.get("X-B3-Sampled").map(|s| s.as_str()),
            ),
            PropagationFormat::B3Single => headers.get("b3").and_then(|h| Self::from_b3_single(h)),
            PropagationFormat::Jaeger => headers
                .get("uber-trace-id")
                .and_then(|h| Self::from_jaeger_header(h)),
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
                    .unwrap_or_default()
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
                    .unwrap_or_default()
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
                    .unwrap_or_default()
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
        self.counters
            .read()
            .map(|c| *c.get(name).unwrap_or(&0))
            .unwrap_or(0)
    }

    /// Get histogram values.
    pub fn get_histogram(&self, name: &str) -> Vec<f64> {
        self.histograms
            .read()
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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

        let span = telemetry
            .span("test_op")
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
        assert_eq!(
            spans[0].status_message,
            Some("Something went wrong".to_string())
        );
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

        let span = telemetry.span("server").kind(SpanKind::Server).start();
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

        telemetry
            .span("test")
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
        )
        .unwrap();

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
        let parsed_b3s =
            TraceContext::from_headers(&b3s_headers, PropagationFormat::B3Single).unwrap();
        assert_eq!(parsed_b3s.trace_id, ctx.trace_id);

        // Test Jaeger format
        let jaeger_headers = ctx.to_headers(PropagationFormat::Jaeger);
        let parsed_jaeger =
            TraceContext::from_headers(&jaeger_headers, PropagationFormat::Jaeger).unwrap();
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
        assert_eq!(
            child.baggage.get("request_id"),
            Some(&"abc-123".to_string())
        );
        assert_eq!(child.baggage.get("user"), Some(&"john".to_string()));
    }

    // ============================================================================
    // Export Format Tests
    // ============================================================================

    #[test]
    fn test_export_otlp() {
        let telemetry = Telemetry::default();

        telemetry
            .span("test_operation")
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

        telemetry.span("search").attribute("k", 10i64).start().end();

        let json = telemetry.export_otlp_json();
        assert!(json.contains("search"));
        assert!(json.contains("scope_spans"));
        assert!(json.contains("resource"));
    }

    #[test]
    fn test_export_jaeger() {
        let telemetry = Telemetry::default();

        let mut span = telemetry
            .span("jaeger_test")
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

        telemetry
            .span("zipkin_test")
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

        let mut span = telemetry
            .span("console_test")
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
        let config = TelemetryConfig::new("test").with_format(ExportFormat::Zipkin);
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
        assert_eq!(
            map.get("deployment.environment"),
            Some(&"production".to_string())
        );
        assert_eq!(map.get("service.namespace"), Some(&"backend".to_string()));
        assert_eq!(
            map.get("service.instance.id"),
            Some(&"instance-1".to_string())
        );
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
        assert_eq!(
            config.exporter_endpoint,
            Some("http://localhost:4317".to_string())
        );
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
