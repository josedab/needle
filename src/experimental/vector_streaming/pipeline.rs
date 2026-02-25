#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use tracing::{debug, info};

use super::consumer::VectorMessage;

// =============================================================================
// Advanced Stream Processing (Next-Gen)
// =============================================================================

/// Stream processing operation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamOp {
    /// Filter messages
    Filter,
    /// Transform vectors
    Transform,
    /// Aggregate over window
    Aggregate,
    /// Join with another stream
    Join,
    /// Route to different outputs
    Route,
}

/// Stream processing pipeline for vector data
pub struct VectorStreamPipeline {
    stages: Vec<StreamStage>,
    error_handler: Box<dyn Fn(&VectorMessage, &str) + Send + Sync>,
    metrics: StreamMetrics,
}

struct StreamStage {
    #[allow(dead_code)]
    name: String,
    op: StreamOp,
    processor: Box<dyn Fn(VectorMessage) -> Option<VectorMessage> + Send + Sync>,
}

/// Metrics for stream processing
#[derive(Debug, Default)]
pub struct StreamMetrics {
    pub messages_processed: AtomicU64,
    pub messages_filtered: AtomicU64,
    pub messages_transformed: AtomicU64,
    pub messages_errored: AtomicU64,
    pub processing_time_us: AtomicU64,
}

impl StreamMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_processed(&self) {
        self.messages_processed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_filtered(&self) {
        self.messages_filtered.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_transformed(&self) {
        self.messages_transformed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.messages_errored.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_processing_time(&self, micros: u64) {
        self.processing_time_us.fetch_add(micros, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StreamMetricsSnapshot {
        StreamMetricsSnapshot {
            messages_processed: self.messages_processed.load(Ordering::Relaxed),
            messages_filtered: self.messages_filtered.load(Ordering::Relaxed),
            messages_transformed: self.messages_transformed.load(Ordering::Relaxed),
            messages_errored: self.messages_errored.load(Ordering::Relaxed),
            total_processing_time_us: self.processing_time_us.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetricsSnapshot {
    pub messages_processed: u64,
    pub messages_filtered: u64,
    pub messages_transformed: u64,
    pub messages_errored: u64,
    pub total_processing_time_us: u64,
}

impl VectorStreamPipeline {
    /// Create a new stream pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            error_handler: Box::new(|_msg, _err| {}),
            metrics: StreamMetrics::new(),
        }
    }

    /// Add a filter stage
    pub fn filter<F>(mut self, name: &str, predicate: F) -> Self
    where
        F: Fn(&VectorMessage) -> bool + Send + Sync + 'static,
    {
        self.stages.push(StreamStage {
            name: name.to_string(),
            op: StreamOp::Filter,
            processor: Box::new(move |msg| if predicate(&msg) { Some(msg) } else { None }),
        });
        self
    }

    /// Add a transform stage for vectors
    pub fn transform<F>(mut self, name: &str, transformer: F) -> Self
    where
        F: Fn(Vec<f32>) -> Vec<f32> + Send + Sync + 'static,
    {
        self.stages.push(StreamStage {
            name: name.to_string(),
            op: StreamOp::Transform,
            processor: Box::new(move |mut msg| {
                msg.vector = transformer(msg.vector);
                Some(msg)
            }),
        });
        self
    }

    /// Add a metadata transform stage
    pub fn map_metadata<F>(mut self, name: &str, mapper: F) -> Self
    where
        F: Fn(Option<Value>) -> Option<Value> + Send + Sync + 'static,
    {
        self.stages.push(StreamStage {
            name: name.to_string(),
            op: StreamOp::Transform,
            processor: Box::new(move |mut msg| {
                msg.metadata = mapper(msg.metadata);
                Some(msg)
            }),
        });
        self
    }

    /// Set error handler
    pub fn on_error<F>(mut self, handler: F) -> Self
    where
        F: Fn(&VectorMessage, &str) + Send + Sync + 'static,
    {
        self.error_handler = Box::new(handler);
        self
    }

    /// Process a single message through the pipeline
    pub fn process(&self, mut message: VectorMessage) -> Option<VectorMessage> {
        let start = Instant::now();

        for stage in &self.stages {
            match (stage.processor)(message.clone()) {
                Some(processed) => {
                    message = processed;
                    match stage.op {
                        StreamOp::Filter => self.metrics.record_filtered(),
                        StreamOp::Transform => self.metrics.record_transformed(),
                        _ => {}
                    }
                }
                None => {
                    self.metrics.record_filtered();
                    return None;
                }
            }
        }

        self.metrics.record_processed();
        self.metrics
            .record_processing_time(start.elapsed().as_micros() as u64);

        Some(message)
    }

    /// Process a batch of messages
    pub fn process_batch(&self, messages: Vec<VectorMessage>) -> Vec<VectorMessage> {
        messages
            .into_iter()
            .filter_map(|m| self.process(m))
            .collect()
    }

    /// Get metrics snapshot
    pub fn metrics(&self) -> StreamMetricsSnapshot {
        self.metrics.snapshot()
    }
}

impl Default for VectorStreamPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// CDC event types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CdcEventType {
    /// Insert/create
    Insert,
    /// Update
    Update,
    /// Delete
    Delete,
    /// Snapshot (initial load)
    Snapshot,
}

/// CDC event for vector changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcEvent {
    /// Event type
    pub event_type: CdcEventType,
    /// Vector ID
    pub id: String,
    /// Collection name
    pub collection: String,
    /// New vector value (None for deletes)
    pub vector: Option<Vec<f32>>,
    /// Previous vector value (for updates)
    pub previous_vector: Option<Vec<f32>>,
    /// Metadata
    pub metadata: Option<Value>,
    /// Timestamp of the change
    pub timestamp: u64,
    /// Source transaction ID
    pub transaction_id: Option<String>,
    /// Log sequence number
    pub lsn: Option<u64>,
}

/// CDC stream for change data capture
pub struct CdcStream {
    events: RwLock<VecDeque<CdcEvent>>,
    subscribers: RwLock<Vec<Box<dyn Fn(&CdcEvent) + Send + Sync>>>,
    last_lsn: AtomicU64,
    enabled: AtomicBool,
}

impl CdcStream {
    /// Create a new CDC stream
    pub fn new() -> Self {
        Self {
            events: RwLock::new(VecDeque::new()),
            subscribers: RwLock::new(Vec::new()),
            last_lsn: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Publish a CDC event
    pub fn publish(&self, event: CdcEvent) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        // Update LSN
        if let Some(lsn) = event.lsn {
            self.last_lsn.fetch_max(lsn, Ordering::Relaxed);
        }

        // Store event
        self.events.write().push_back(event.clone());

        // Notify subscribers
        for subscriber in self.subscribers.read().iter() {
            subscriber(&event);
        }
    }

    /// Subscribe to CDC events
    pub fn subscribe<F>(&self, callback: F)
    where
        F: Fn(&CdcEvent) + Send + Sync + 'static,
    {
        self.subscribers.write().push(Box::new(callback));
    }

    /// Get events since LSN
    pub fn get_events_since(&self, lsn: u64) -> Vec<CdcEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.lsn.unwrap_or(0) > lsn)
            .cloned()
            .collect()
    }

    /// Get last LSN
    pub fn last_lsn(&self) -> u64 {
        self.last_lsn.load(Ordering::Relaxed)
    }

    /// Enable/disable CDC
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Clear old events (retention)
    pub fn prune(&self, keep_count: usize) {
        let mut events = self.events.write();
        while events.len() > keep_count {
            events.pop_front();
        }
    }

    /// Get pending event count
    pub fn pending_count(&self) -> usize {
        self.events.read().len()
    }
}

impl Default for CdcStream {
    fn default() -> Self {
        Self::new()
    }
}

/// Replay manager for stream recovery
pub struct ReplayManager {
    snapshots: RwLock<HashMap<String, StreamSnapshot>>,
    retention_seconds: u64,
}

/// Stream snapshot for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSnapshot {
    /// Snapshot ID
    pub id: String,
    /// Consumer group
    pub group_id: String,
    /// Topic/stream name
    pub topic: String,
    /// Partition offsets at snapshot time
    pub offsets: HashMap<u32, u64>,
    /// Watermark at snapshot time
    pub watermark: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Processed message IDs (for dedup)
    pub processed_ids: Vec<String>,
}

impl ReplayManager {
    /// Create a new replay manager
    pub fn new(retention_seconds: u64) -> Self {
        Self {
            snapshots: RwLock::new(HashMap::new()),
            retention_seconds,
        }
    }

    /// Take a snapshot
    pub fn take_snapshot(
        &self,
        group_id: &str,
        topic: &str,
        offsets: HashMap<u32, u64>,
        watermark: u64,
        processed_ids: Vec<String>,
    ) -> String {
        let id = format!(
            "{}-{}-{}",
            group_id,
            topic,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        );

        let snapshot = StreamSnapshot {
            id: id.clone(),
            group_id: group_id.to_string(),
            topic: topic.to_string(),
            offsets,
            watermark,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            processed_ids,
        };

        self.snapshots.write().insert(id.clone(), snapshot);
        id
    }

    /// Get a snapshot
    pub fn get_snapshot(&self, id: &str) -> Option<StreamSnapshot> {
        self.snapshots.read().get(id).cloned()
    }

    /// List snapshots for a group/topic
    pub fn list_snapshots(&self, group_id: &str, topic: &str) -> Vec<StreamSnapshot> {
        self.snapshots
            .read()
            .values()
            .filter(|s| s.group_id == group_id && s.topic == topic)
            .cloned()
            .collect()
    }

    /// Clean up old snapshots
    pub fn cleanup(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.snapshots
            .write()
            .retain(|_, s| now - s.timestamp < self.retention_seconds);
    }
}

// ---------------------------------------------------------------------------
// Backpressure Controller
// ---------------------------------------------------------------------------

/// Backpressure state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressureState {
    /// Normal throughput
    Flowing,
    /// Reduced throughput — consumer should slow down
    Throttled,
    /// Consumer should stop polling until pressure eases
    Paused,
}

/// Configuration for the backpressure controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Maximum in-flight (un-committed) messages before throttling
    pub max_in_flight: usize,
    /// Maximum in-flight before pausing entirely
    pub max_in_flight_pause: usize,
    /// Percentage of collection capacity that triggers throttling (0..100)
    pub capacity_throttle_pct: f64,
    /// Percentage of collection capacity that triggers pause (0..100)
    pub capacity_pause_pct: f64,
    /// When throttled, reduce batch size by this factor
    pub throttle_factor: f64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_in_flight: 10_000,
            max_in_flight_pause: 50_000,
            capacity_throttle_pct: 80.0,
            capacity_pause_pct: 95.0,
            throttle_factor: 0.5,
        }
    }
}

/// Controls backpressure between a streaming source and Needle ingestion.
pub struct BackpressureController {
    config: BackpressureConfig,
    in_flight: AtomicU64,
    state: RwLock<BackpressureState>,
    throttle_count: AtomicU64,
    pause_count: AtomicU64,
}

impl BackpressureController {
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            in_flight: AtomicU64::new(0),
            state: RwLock::new(BackpressureState::Flowing),
            throttle_count: AtomicU64::new(0),
            pause_count: AtomicU64::new(0),
        }
    }

    /// Notify the controller that messages were received.
    pub fn on_receive(&self, count: u64) {
        self.in_flight.fetch_add(count, Ordering::Relaxed);
        self.evaluate();
    }

    /// Notify the controller that messages were committed/acked.
    pub fn on_commit(&self, count: u64) {
        let prev = self.in_flight.load(Ordering::Relaxed);
        self.in_flight
            .store(prev.saturating_sub(count), Ordering::Relaxed);
        self.evaluate();
    }

    /// Evaluate the current state based on in-flight count.
    fn evaluate(&self) {
        let in_flight = self.in_flight.load(Ordering::Relaxed) as usize;
        let mut state = self.state.write();

        if in_flight >= self.config.max_in_flight_pause {
            if *state != BackpressureState::Paused {
                self.pause_count.fetch_add(1, Ordering::Relaxed);
                info!(in_flight, "Backpressure: pausing consumer");
            }
            *state = BackpressureState::Paused;
        } else if in_flight >= self.config.max_in_flight {
            if *state == BackpressureState::Flowing {
                self.throttle_count.fetch_add(1, Ordering::Relaxed);
                debug!(in_flight, "Backpressure: throttling consumer");
            }
            *state = BackpressureState::Throttled;
        } else {
            *state = BackpressureState::Flowing;
        }
    }

    /// Evaluate based on collection capacity (vector count vs limit).
    pub fn evaluate_capacity(&self, current_vectors: u64, max_vectors: u64) {
        if max_vectors == 0 {
            return;
        }
        let pct = (current_vectors as f64 / max_vectors as f64) * 100.0;
        let mut state = self.state.write();

        if pct >= self.config.capacity_pause_pct {
            if *state != BackpressureState::Paused {
                self.pause_count.fetch_add(1, Ordering::Relaxed);
            }
            *state = BackpressureState::Paused;
        } else if pct >= self.config.capacity_throttle_pct {
            if *state == BackpressureState::Flowing {
                self.throttle_count.fetch_add(1, Ordering::Relaxed);
            }
            *state = BackpressureState::Throttled;
        }
    }

    /// Current backpressure state.
    pub fn state(&self) -> BackpressureState {
        *self.state.read()
    }

    /// Get the effective batch size given current pressure.
    pub fn effective_batch_size(&self, base_batch_size: usize) -> usize {
        match self.state() {
            BackpressureState::Flowing => base_batch_size,
            BackpressureState::Throttled => {
                (base_batch_size as f64 * self.config.throttle_factor).max(1.0) as usize
            }
            BackpressureState::Paused => 0,
        }
    }

    /// Returns true if the consumer should poll.
    pub fn should_poll(&self) -> bool {
        self.state() != BackpressureState::Paused
    }

    /// Backpressure statistics.
    pub fn stats(&self) -> BackpressureStats {
        BackpressureStats {
            state: self.state(),
            in_flight: self.in_flight.load(Ordering::Relaxed),
            throttle_events: self.throttle_count.load(Ordering::Relaxed),
            pause_events: self.pause_count.load(Ordering::Relaxed),
        }
    }
}

/// Statistics about backpressure events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureStats {
    pub state: BackpressureState,
    pub in_flight: u64,
    pub throttle_events: u64,
    pub pause_events: u64,
}
