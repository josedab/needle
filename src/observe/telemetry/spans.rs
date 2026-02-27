//! Span types and builders for distributed tracing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::{Telemetry, TraceContext};

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
            .unwrap_or_default()
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
    pub fn add_event_with_attributes(
        &mut self,
        name: &str,
        attrs: HashMap<String, AttributeValue>,
    ) {
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
        self.end_time
            .map(|end| (end - self.start_time) as f64 / 1_000_000.0)
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
    pub(crate) fn new(telemetry: &'a Telemetry, name: &str) -> Self {
        Self {
            telemetry,
            name: name.to_string(),
            context: TraceContext::new(),
            attributes: HashMap::new(),
            kind: SpanKind::Internal,
        }
    }

    /// Create span builder with context.
    pub(crate) fn with_context(telemetry: &'a Telemetry, name: &str, context: TraceContext) -> Self {
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
        self.telemetry
            .record_latency(&self.operation, duration, self.success);
        duration
    }
}

#[cfg(test)]
mod tests {}
