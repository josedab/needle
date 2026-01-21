//! OpenTelemetry-Native Tracing
//!
//! Span instrumentation for the search pipeline: embed → traverse → filter →
//! rerank, with structured attributes and OTLP-compatible export.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::otel_tracing::{
//!     SearchTracer, TraceConfig, SearchSpan, SpanKind,
//! };
//!
//! let mut tracer = SearchTracer::new(TraceConfig::default());
//!
//! // Instrument a search pipeline
//! let trace_id = tracer.start_trace("search", "docs");
//! tracer.start_span(trace_id, SpanKind::Embed, "embed_query");
//! tracer.end_span(trace_id, "embed_query", 1.2);
//! tracer.start_span(trace_id, SpanKind::HnswTraversal, "hnsw_search");
//! tracer.add_attribute(trace_id, "hnsw_search", "ef_search", "50");
//! tracer.add_attribute(trace_id, "hnsw_search", "nodes_visited", "847");
//! tracer.end_span(trace_id, "hnsw_search", 3.5);
//! tracer.end_trace(trace_id);
//!
//! let trace = tracer.get_trace(trace_id).unwrap();
//! assert_eq!(trace.spans.len(), 2);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Span Kind ────────────────────────────────────────────────────────────────

/// The type of operation a span represents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanKind {
    /// Embedding generation.
    Embed,
    /// HNSW graph traversal.
    HnswTraversal,
    /// Metadata filter evaluation.
    Filter,
    /// Post-search reranking.
    Rerank,
    /// Quantized distance computation.
    QuantizedSearch,
    /// Result assembly and scoring.
    ResultAssembly,
    /// Full search pipeline.
    SearchPipeline,
    /// Custom operation.
    Custom(String),
}

impl std::fmt::Display for SpanKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embed => write!(f, "embed"),
            Self::HnswTraversal => write!(f, "hnsw.traversal"),
            Self::Filter => write!(f, "filter"),
            Self::Rerank => write!(f, "rerank"),
            Self::QuantizedSearch => write!(f, "quantized.search"),
            Self::ResultAssembly => write!(f, "result.assembly"),
            Self::SearchPipeline => write!(f, "search.pipeline"),
            Self::Custom(name) => write!(f, "custom.{name}"),
        }
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Tracing configuration.
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Service name for traces.
    pub service_name: String,
    /// Sampling rate (0.0 to 1.0).
    pub sample_rate: f32,
    /// Maximum traces to retain.
    pub max_traces: usize,
    /// Whether to include vector data in spans (privacy consideration).
    pub include_vectors: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            service_name: "needle".into(),
            sample_rate: 1.0,
            max_traces: 10_000,
            include_vectors: false,
        }
    }
}

impl TraceConfig {
    /// Set service name.
    #[must_use]
    pub fn with_service(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set sampling rate.
    #[must_use]
    pub fn with_sample_rate(mut self, rate: f32) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }
}

// ── Search Span ──────────────────────────────────────────────────────────────

/// A single span in a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpan {
    /// Span name.
    pub name: String,
    /// Span kind.
    pub kind: SpanKind,
    /// Start time (epoch microseconds).
    pub start_us: u64,
    /// End time (epoch microseconds), None if still open.
    pub end_us: Option<u64>,
    /// Duration in milliseconds.
    pub duration_ms: Option<f64>,
    /// Key-value attributes.
    pub attributes: HashMap<String, String>,
    /// Status (ok, error).
    pub status: SpanStatus,
}

/// Span status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Operation succeeded.
    Ok,
    /// Operation failed.
    Error(String),
    /// Still in progress.
    InProgress,
}

// ── Search Trace ─────────────────────────────────────────────────────────────

/// A complete trace containing multiple spans.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchTrace {
    /// Trace ID.
    pub trace_id: u64,
    /// Operation name.
    pub operation: String,
    /// Collection name.
    pub collection: String,
    /// Service name.
    pub service: String,
    /// All spans in this trace.
    pub spans: Vec<SearchSpan>,
    /// Total duration in milliseconds.
    pub total_duration_ms: Option<f64>,
    /// Start time.
    pub start_us: u64,
    /// End time.
    pub end_us: Option<u64>,
}

// ── Trace Summary ────────────────────────────────────────────────────────────

/// Summary statistics across traces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceSummary {
    /// Total traces.
    pub total_traces: usize,
    /// Average total duration.
    pub avg_duration_ms: f64,
    /// P50 duration.
    pub p50_duration_ms: f64,
    /// P99 duration.
    pub p99_duration_ms: f64,
    /// Average spans per trace.
    pub avg_spans: f64,
    /// Per-span-kind average duration.
    pub span_kind_avg_ms: HashMap<String, f64>,
}

// ── Search Tracer ────────────────────────────────────────────────────────────

/// Tracer for search pipeline instrumentation.
pub struct SearchTracer {
    config: TraceConfig,
    traces: HashMap<u64, SearchTrace>,
    next_id: u64,
    completed_durations: Vec<f64>,
}

impl SearchTracer {
    /// Create a new tracer.
    pub fn new(config: TraceConfig) -> Self {
        Self {
            config,
            traces: HashMap::new(),
            next_id: 1,
            completed_durations: Vec::new(),
        }
    }

    /// Start a new trace. Returns the trace ID.
    pub fn start_trace(&mut self, operation: &str, collection: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        self.traces.insert(id, SearchTrace {
            trace_id: id,
            operation: operation.into(),
            collection: collection.into(),
            service: self.config.service_name.clone(),
            spans: Vec::new(),
            total_duration_ms: None,
            start_us: now_us(),
            end_us: None,
        });
        id
    }

    /// Start a span within a trace.
    pub fn start_span(&mut self, trace_id: u64, kind: SpanKind, name: &str) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            trace.spans.push(SearchSpan {
                name: name.into(),
                kind,
                start_us: now_us(),
                end_us: None,
                duration_ms: None,
                attributes: HashMap::new(),
                status: SpanStatus::InProgress,
            });
        }
    }

    /// End a span with its duration.
    pub fn end_span(&mut self, trace_id: u64, name: &str, duration_ms: f64) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            if let Some(span) = trace.spans.iter_mut().find(|s| s.name == name) {
                span.end_us = Some(now_us());
                span.duration_ms = Some(duration_ms);
                span.status = SpanStatus::Ok;
            }
        }
    }

    /// Mark a span as errored.
    pub fn error_span(&mut self, trace_id: u64, name: &str, error: &str) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            if let Some(span) = trace.spans.iter_mut().find(|s| s.name == name) {
                span.end_us = Some(now_us());
                span.status = SpanStatus::Error(error.into());
            }
        }
    }

    /// Add an attribute to a span.
    pub fn add_attribute(&mut self, trace_id: u64, span_name: &str, key: &str, value: &str) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            if let Some(span) = trace.spans.iter_mut().find(|s| s.name == span_name) {
                span.attributes.insert(key.into(), value.into());
            }
        }
    }

    /// End a trace.
    pub fn end_trace(&mut self, trace_id: u64) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            let end = now_us();
            trace.end_us = Some(end);
            let duration = (end - trace.start_us) as f64 / 1000.0;
            trace.total_duration_ms = Some(duration);
            self.completed_durations.push(duration);
        }
        // Enforce retention
        while self.traces.len() > self.config.max_traces {
            if let Some(&oldest) = self.traces.keys().min() {
                self.traces.remove(&oldest);
            }
        }
    }

    /// Get a trace by ID.
    pub fn get_trace(&self, trace_id: u64) -> Option<&SearchTrace> {
        self.traces.get(&trace_id)
    }

    /// Get summary statistics.
    pub fn summary(&self) -> TraceSummary {
        let completed: Vec<&SearchTrace> = self.traces.values()
            .filter(|t| t.end_us.is_some())
            .collect();

        if completed.is_empty() {
            return TraceSummary::default();
        }

        let mut durations: Vec<f64> = completed.iter()
            .filter_map(|t| t.total_duration_ms)
            .collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let avg = durations.iter().sum::<f64>() / durations.len() as f64;
        let p50 = durations.get(durations.len() / 2).copied().unwrap_or(0.0);
        let p99_idx = (durations.len() as f64 * 0.99) as usize;
        let p99 = durations.get(p99_idx.min(durations.len().saturating_sub(1))).copied().unwrap_or(0.0);

        let total_spans: usize = completed.iter().map(|t| t.spans.len()).sum();
        let avg_spans = total_spans as f64 / completed.len() as f64;

        // Per-kind averages
        let mut kind_totals: HashMap<String, (f64, usize)> = HashMap::new();
        for trace in &completed {
            for span in &trace.spans {
                if let Some(dur) = span.duration_ms {
                    let entry = kind_totals.entry(span.kind.to_string()).or_default();
                    entry.0 += dur;
                    entry.1 += 1;
                }
            }
        }
        let span_kind_avg: HashMap<String, f64> = kind_totals.into_iter()
            .map(|(k, (total, count))| (k, total / count as f64))
            .collect();

        TraceSummary {
            total_traces: completed.len(),
            avg_duration_ms: avg,
            p50_duration_ms: p50,
            p99_duration_ms: p99,
            avg_spans,
            span_kind_avg_ms: span_kind_avg,
        }
    }

    /// Total trace count.
    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }

    /// Clear all traces.
    pub fn clear(&mut self) {
        self.traces.clear();
        self.completed_durations.clear();
    }
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tracing() {
        let mut tracer = SearchTracer::new(TraceConfig::default());
        let tid = tracer.start_trace("search", "docs");

        tracer.start_span(tid, SpanKind::Embed, "embed");
        tracer.end_span(tid, "embed", 1.5);

        tracer.start_span(tid, SpanKind::HnswTraversal, "hnsw");
        tracer.add_attribute(tid, "hnsw", "ef_search", "50");
        tracer.end_span(tid, "hnsw", 3.2);

        tracer.end_trace(tid);

        let trace = tracer.get_trace(tid).unwrap();
        assert_eq!(trace.spans.len(), 2);
        assert!(trace.total_duration_ms.is_some());
    }

    #[test]
    fn test_span_attributes() {
        let mut tracer = SearchTracer::new(TraceConfig::default());
        let tid = tracer.start_trace("search", "docs");
        tracer.start_span(tid, SpanKind::HnswTraversal, "hnsw");
        tracer.add_attribute(tid, "hnsw", "nodes", "847");
        tracer.end_span(tid, "hnsw", 2.0);
        tracer.end_trace(tid);

        let trace = tracer.get_trace(tid).unwrap();
        assert_eq!(trace.spans[0].attributes.get("nodes").unwrap(), "847");
    }

    #[test]
    fn test_error_span() {
        let mut tracer = SearchTracer::new(TraceConfig::default());
        let tid = tracer.start_trace("search", "docs");
        tracer.start_span(tid, SpanKind::Filter, "filter");
        tracer.error_span(tid, "filter", "invalid predicate");
        tracer.end_trace(tid);

        let trace = tracer.get_trace(tid).unwrap();
        assert!(matches!(trace.spans[0].status, SpanStatus::Error(_)));
    }

    #[test]
    fn test_summary() {
        let mut tracer = SearchTracer::new(TraceConfig::default());

        for _ in 0..5 {
            let tid = tracer.start_trace("search", "docs");
            tracer.start_span(tid, SpanKind::Embed, "embed");
            tracer.end_span(tid, "embed", 1.0);
            tracer.start_span(tid, SpanKind::HnswTraversal, "hnsw");
            tracer.end_span(tid, "hnsw", 3.0);
            tracer.end_trace(tid);
        }

        let summary = tracer.summary();
        assert_eq!(summary.total_traces, 5);
        assert!(summary.avg_spans > 1.0);
        assert!(summary.span_kind_avg_ms.contains_key("embed"));
    }

    #[test]
    fn test_retention() {
        let config = TraceConfig { max_traces: 3, ..Default::default() };
        let mut tracer = SearchTracer::new(config);

        for _ in 0..5 {
            let tid = tracer.start_trace("s", "c");
            tracer.end_trace(tid);
        }
        assert!(tracer.trace_count() <= 3);
    }

    #[test]
    fn test_span_kinds() {
        assert_eq!(format!("{}", SpanKind::Embed), "embed");
        assert_eq!(format!("{}", SpanKind::HnswTraversal), "hnsw.traversal");
        assert_eq!(format!("{}", SpanKind::Custom("foo".into())), "custom.foo");
    }

    #[test]
    fn test_clear() {
        let mut tracer = SearchTracer::new(TraceConfig::default());
        let tid = tracer.start_trace("s", "c");
        tracer.end_trace(tid);
        tracer.clear();
        assert_eq!(tracer.trace_count(), 0);
    }
}
