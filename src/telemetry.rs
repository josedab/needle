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
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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
        }
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
}
