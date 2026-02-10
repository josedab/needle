//! Streaming Ingestion Pipeline
//!
//! A Source/Transform/Sink trait-based pipeline framework for real-time vector
//! ingestion with backpressure control and exactly-once semantics.

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// A record flowing through the ingestion pipeline.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub data: RecordData,
    pub timestamp: u64,
    pub source: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// The payload carried by a record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RecordData {
    Text(String),
    Vector(Vec<f32>),
    Json(serde_json::Value),
    Bytes(Vec<u8>),
}

/// Configuration for the ingestion pipeline.
pub struct PipelineConfig {
    pub batch_size: usize,
    pub flush_interval: Duration,
    pub max_buffer_size: usize,
    pub enable_dedup: bool,
    pub max_retries: u32,
    pub backpressure_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            flush_interval: Duration::from_secs(5),
            max_buffer_size: 10_000,
            enable_dedup: true,
            max_retries: 3,
            backpressure_threshold: 0.8,
        }
    }
}

/// Describes how a source is configured.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SourceConfig {
    InMemory,
    Stdin,
    JsonFile(String),
    Custom(String),
}

/// The kind of transform to apply to records.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransformType {
    Identity,
    TextToVector {
        dimensions: usize,
    },
    JsonExtract {
        field: String,
    },
    FilterByField {
        field: String,
        value: serde_json::Value,
    },
    Custom(String),
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// A source of records for the pipeline.
pub trait Source {
    fn name(&self) -> &str;
    fn pull(&mut self) -> Result<Option<Record>>;
    fn acknowledge(&mut self, id: &str) -> Result<()>;
    fn pending_count(&self) -> usize;
}

/// A stateless transformation applied to each record.
pub trait Transform {
    fn name(&self) -> &str;
    fn apply(&self, record: Record) -> Result<Option<Record>>;
}

/// A destination that receives batches of records.
pub trait Sink {
    fn name(&self) -> &str;
    fn push(&mut self, records: &[Record]) -> Result<usize>;
    fn flush(&mut self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Built-in Source implementations
// ---------------------------------------------------------------------------

/// An in-memory source backed by a VecDeque.
pub struct InMemorySource {
    name: String,
    queue: VecDeque<Record>,
    acknowledged: HashSet<String>,
}

impl InMemorySource {
    pub fn new(name: impl Into<String>, records: Vec<Record>) -> Self {
        Self {
            name: name.into(),
            queue: VecDeque::from(records),
            acknowledged: HashSet::new(),
        }
    }
}

impl Source for InMemorySource {
    fn name(&self) -> &str {
        &self.name
    }

    fn pull(&mut self) -> Result<Option<Record>> {
        Ok(self.queue.pop_front())
    }

    fn acknowledge(&mut self, id: &str) -> Result<()> {
        self.acknowledged.insert(id.to_string());
        Ok(())
    }

    fn pending_count(&self) -> usize {
        self.queue.len()
    }
}

// ---------------------------------------------------------------------------
// Built-in Sink implementations
// ---------------------------------------------------------------------------

/// An in-memory sink that collects records.
pub struct InMemorySink {
    name: String,
    records: Vec<Record>,
    total_pushed: usize,
}

impl InMemorySink {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            records: Vec::new(),
            total_pushed: 0,
        }
    }

    pub fn records(&self) -> &[Record] {
        &self.records
    }

    pub fn total_pushed(&self) -> usize {
        self.total_pushed
    }
}

impl Sink for InMemorySink {
    fn name(&self) -> &str {
        &self.name
    }

    fn push(&mut self, records: &[Record]) -> Result<usize> {
        let count = records.len();
        self.records.extend_from_slice(records);
        self.total_pushed += count;
        Ok(count)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Built-in Transform implementations
// ---------------------------------------------------------------------------

/// A no-op transform that passes records through unchanged.
pub struct IdentityTransform;

impl Transform for IdentityTransform {
    fn name(&self) -> &str {
        "identity"
    }

    fn apply(&self, record: Record) -> Result<Option<Record>> {
        Ok(Some(record))
    }
}

/// Converts `RecordData::Text` into `RecordData::Vector` using a deterministic
/// hash-based mock embedding (same approach as `auto_embed.rs`).
pub struct TextToVectorTransform {
    pub dimensions: usize,
}

impl TextToVectorTransform {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    fn hash_text(text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    fn generate_mock_embedding(text: &str, dimensions: usize) -> Vec<f32> {
        let hash = Self::hash_text(text);
        let mut embedding = Vec::with_capacity(dimensions);
        let mut state = hash;

        for _ in 0..dimensions {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        embedding
    }
}

impl Transform for TextToVectorTransform {
    fn name(&self) -> &str {
        "text_to_vector"
    }

    fn apply(&self, mut record: Record) -> Result<Option<Record>> {
        if let RecordData::Text(ref text) = record.data {
            let vec = Self::generate_mock_embedding(text, self.dimensions);
            record.data = RecordData::Vector(vec);
        }
        Ok(Some(record))
    }
}

/// Filters records by matching a metadata field against a value.
pub struct FilterTransform {
    pub field: String,
    pub value: serde_json::Value,
}

impl FilterTransform {
    pub fn new(field: impl Into<String>, value: serde_json::Value) -> Self {
        Self {
            field: field.into(),
            value,
        }
    }
}

impl Transform for FilterTransform {
    fn name(&self) -> &str {
        "filter"
    }

    fn apply(&self, record: Record) -> Result<Option<Record>> {
        match record.metadata.get(&self.field) {
            Some(v) if *v == self.value => Ok(Some(record)),
            _ => Ok(None),
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for the pipeline.
pub struct PipelineStats {
    pub records_read: u64,
    pub records_transformed: u64,
    pub records_written: u64,
    pub records_filtered: u64,
    pub records_failed: u64,
    pub duplicates_skipped: u64,
    pub avg_throughput_per_sec: f64,
    pub last_flush_at: Option<Instant>,
    pub buffer_utilization: f32,
}

impl Default for PipelineStats {
    fn default() -> Self {
        Self {
            records_read: 0,
            records_transformed: 0,
            records_written: 0,
            records_filtered: 0,
            records_failed: 0,
            duplicates_skipped: 0,
            avg_throughput_per_sec: 0.0,
            last_flush_at: None,
            buffer_utilization: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// IngestionPipeline
// ---------------------------------------------------------------------------

/// The main ingestion pipeline orchestrating Source → Transform → Sink.
pub struct IngestionPipeline {
    config: PipelineConfig,
    source: Box<dyn Source>,
    transforms: Vec<Box<dyn Transform>>,
    sink: Box<dyn Sink>,
    stats: PipelineStats,
    seen_ids: HashSet<String>,
    buffer: Vec<Record>,
    start_time: Instant,
}

impl IngestionPipeline {
    /// Create a new pipeline with the given config, source, and sink.
    pub fn new(config: PipelineConfig, source: Box<dyn Source>, sink: Box<dyn Sink>) -> Self {
        Self {
            config,
            source,
            transforms: Vec::new(),
            sink,
            stats: PipelineStats::default(),
            seen_ids: HashSet::new(),
            buffer: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Append a transform stage to the pipeline.
    pub fn add_transform(&mut self, transform: Box<dyn Transform>) -> &mut Self {
        self.transforms.push(transform);
        self
    }

    /// Pull up to `batch_size` records, transform, de-duplicate, buffer, and
    /// flush to the sink when the buffer reaches `batch_size`.
    pub fn process_batch(&mut self) -> Result<usize> {
        let mut processed = 0;

        for _ in 0..self.config.batch_size {
            let record = match self.source.pull()? {
                Some(r) => r,
                None => break,
            };
            self.stats.records_read += 1;

            // De-duplication
            if self.config.enable_dedup && self.seen_ids.contains(&record.id) {
                self.stats.duplicates_skipped += 1;
                continue;
            }

            // Apply transforms
            let mut current = Some(record);
            for t in &self.transforms {
                current = match current {
                    Some(r) => t.apply(r)?,
                    None => break,
                };
            }

            match current {
                Some(r) => {
                    if self.config.enable_dedup {
                        self.seen_ids.insert(r.id.clone());
                    }
                    self.buffer.push(r);
                    self.stats.records_transformed += 1;
                    processed += 1;
                }
                None => {
                    self.stats.records_filtered += 1;
                }
            }
        }

        self.update_buffer_utilization();

        // Flush when buffer is large enough
        if self.buffer.len() >= self.config.batch_size {
            self.flush()?;
        }

        Ok(processed)
    }

    /// Force-flush the buffer to the sink.
    pub fn flush(&mut self) -> Result<usize> {
        if self.buffer.is_empty() {
            return Ok(0);
        }

        let batch: Vec<Record> = self.buffer.drain(..).collect();
        let written = self.sink.push(&batch)?;
        self.sink.flush()?;
        self.stats.records_written += written as u64;
        self.stats.last_flush_at = Some(Instant::now());
        self.update_throughput();
        self.update_buffer_utilization();
        Ok(written)
    }

    /// Single iteration: pull up to batch_size, transform, buffer, maybe flush.
    pub fn run_once(&mut self) -> Result<usize> {
        self.process_batch()
    }

    /// Current pipeline statistics.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Returns `true` when the buffer exceeds the backpressure threshold.
    pub fn is_backpressured(&self) -> bool {
        let ratio = self.buffer.len() as f32 / self.config.max_buffer_size as f32;
        ratio > self.config.backpressure_threshold
    }

    /// Reset all statistics counters.
    pub fn reset_stats(&mut self) {
        self.stats = PipelineStats::default();
        self.start_time = Instant::now();
    }

    fn update_throughput(&mut self) {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.stats.avg_throughput_per_sec = self.stats.records_written as f64 / elapsed;
        }
    }

    fn update_buffer_utilization(&mut self) {
        self.stats.buffer_utilization =
            self.buffer.len() as f32 / self.config.max_buffer_size as f32;
    }
}

// ---------------------------------------------------------------------------
// PipelineBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing an `IngestionPipeline`.
pub struct PipelineBuilder {
    config: Option<PipelineConfig>,
    source: Option<Box<dyn Source>>,
    sink: Option<Box<dyn Sink>>,
    transforms: Vec<Box<dyn Transform>>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: None,
            source: None,
            sink: None,
            transforms: Vec::new(),
        }
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_source(mut self, source: Box<dyn Source>) -> Self {
        self.source = Some(source);
        self
    }

    pub fn with_sink(mut self, sink: Box<dyn Sink>) -> Self {
        self.sink = Some(sink);
        self
    }

    pub fn add_transform(mut self, transform: Box<dyn Transform>) -> Self {
        self.transforms.push(transform);
        self
    }

    pub fn build(self) -> Result<IngestionPipeline> {
        let source = self
            .source
            .ok_or_else(|| NeedleError::InvalidConfig("pipeline requires a source".into()))?;
        let sink = self
            .sink
            .ok_or_else(|| NeedleError::InvalidConfig("pipeline requires a sink".into()))?;
        let config = self.config.unwrap_or_default();

        let mut pipeline = IngestionPipeline::new(config, source, sink);
        for t in self.transforms {
            pipeline.add_transform(t);
        }
        Ok(pipeline)
    }
}

// ---------------------------------------------------------------------------
// Checkpoint for exactly-once semantics
// ---------------------------------------------------------------------------

/// Checkpoint for tracking pipeline progress, enabling exactly-once processing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Checkpoint {
    pub pipeline_id: String,
    pub last_processed_id: Option<String>,
    pub last_processed_offset: u64,
    pub timestamp: u64,
    pub records_processed: u64,
    pub state: HashMap<String, String>,
}

impl Checkpoint {
    pub fn new(pipeline_id: impl Into<String>) -> Self {
        Self {
            pipeline_id: pipeline_id.into(),
            last_processed_id: None,
            last_processed_offset: 0,
            timestamp: 0,
            records_processed: 0,
            state: HashMap::new(),
        }
    }
}

/// Trait for persisting and loading pipeline checkpoints.
pub trait CheckpointStore {
    fn save(&mut self, checkpoint: &Checkpoint) -> Result<()>;
    fn load(&self, pipeline_id: &str) -> Result<Option<Checkpoint>>;
    fn delete(&mut self, pipeline_id: &str) -> Result<()>;
}

/// In-memory implementation of [`CheckpointStore`].
pub struct InMemoryCheckpointStore {
    store: HashMap<String, Checkpoint>,
}

impl InMemoryCheckpointStore {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
}

impl Default for InMemoryCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn save(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        self.store
            .insert(checkpoint.pipeline_id.clone(), checkpoint.clone());
        Ok(())
    }

    fn load(&self, pipeline_id: &str) -> Result<Option<Checkpoint>> {
        Ok(self.store.get(pipeline_id).cloned())
    }

    fn delete(&mut self, pipeline_id: &str) -> Result<()> {
        self.store.remove(pipeline_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Backpressure control
// ---------------------------------------------------------------------------

/// State of the backpressure controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackpressureState {
    Flowing,
    Paused,
}

/// Configuration for a [`BackpressureController`].
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    pub max_buffer_size: usize,
    pub pause_threshold: f32,
    pub resume_threshold: f32,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10_000,
            pause_threshold: 0.8,
            resume_threshold: 0.5,
        }
    }
}

/// Snapshot of backpressure statistics.
#[derive(Debug, Clone)]
pub struct BackpressureStats {
    pub state: BackpressureState,
    pub current_buffer_size: usize,
    pub max_buffer_size: usize,
    pub utilization: f32,
    pub total_pause_duration: Duration,
}

/// Controls ingestion rate based on buffer utilization thresholds.
pub struct BackpressureController {
    config: BackpressureConfig,
    current_buffer_size: usize,
    max_buffer_size: usize,
    pause_threshold: f32,
    resume_threshold: f32,
    state: BackpressureState,
    paused_since: Option<Instant>,
    total_pause_duration: Duration,
}

impl BackpressureController {
    pub fn new(max_buffer: usize, pause_threshold: f32, resume_threshold: f32) -> Self {
        Self {
            config: BackpressureConfig {
                max_buffer_size: max_buffer,
                pause_threshold,
                resume_threshold,
            },
            current_buffer_size: 0,
            max_buffer_size: max_buffer,
            pause_threshold,
            resume_threshold,
            state: BackpressureState::Flowing,
            paused_since: None,
            total_pause_duration: Duration::ZERO,
        }
    }

    /// Update the current buffer size and transition state as needed.
    pub fn update_buffer_size(&mut self, size: usize) {
        self.current_buffer_size = size;
        let utilization = self.utilization();

        match self.state {
            BackpressureState::Flowing => {
                if utilization >= self.pause_threshold {
                    self.state = BackpressureState::Paused;
                    self.paused_since = Some(Instant::now());
                }
            }
            BackpressureState::Paused => {
                if utilization <= self.resume_threshold {
                    if let Some(since) = self.paused_since.take() {
                        self.total_pause_duration += since.elapsed();
                    }
                    self.state = BackpressureState::Flowing;
                }
            }
        }
    }

    pub fn should_pause(&self) -> bool {
        self.state == BackpressureState::Paused
    }

    pub fn should_resume(&self) -> bool {
        self.state == BackpressureState::Flowing
    }

    pub fn state(&self) -> BackpressureState {
        self.state
    }

    pub fn utilization(&self) -> f32 {
        if self.max_buffer_size == 0 {
            return 0.0;
        }
        self.current_buffer_size as f32 / self.max_buffer_size as f32
    }

    pub fn stats(&self) -> BackpressureStats {
        let total_pause = match self.paused_since {
            Some(since) => self.total_pause_duration + since.elapsed(),
            None => self.total_pause_duration,
        };
        BackpressureStats {
            state: self.state,
            current_buffer_size: self.current_buffer_size,
            max_buffer_size: self.max_buffer_size,
            utilization: self.utilization(),
            total_pause_duration: total_pause,
        }
    }

    pub fn config(&self) -> &BackpressureConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Dead letter queue & retry
// ---------------------------------------------------------------------------

/// A record that failed processing.
pub struct FailedRecord {
    pub record: Record,
    pub error: String,
    pub attempt_count: u32,
    pub first_failure: Instant,
    pub last_failure: Instant,
}

/// Queue for records that could not be processed after retries.
pub struct DeadLetterQueue {
    records: VecDeque<FailedRecord>,
    max_size: usize,
    total_enqueued: u64,
}

impl DeadLetterQueue {
    pub fn new(max_size: usize) -> Self {
        Self {
            records: VecDeque::new(),
            max_size,
            total_enqueued: 0,
        }
    }

    /// Enqueue a failed record, evicting the oldest entry if at capacity.
    pub fn enqueue(&mut self, record: Record, error: &str, attempts: u32) {
        if self.records.len() >= self.max_size {
            self.records.pop_front();
        }
        let now = Instant::now();
        self.records.push_back(FailedRecord {
            record,
            error: error.to_string(),
            attempt_count: attempts,
            first_failure: now,
            last_failure: now,
        });
        self.total_enqueued += 1;
    }

    pub fn dequeue(&mut self) -> Option<FailedRecord> {
        self.records.pop_front()
    }

    pub fn peek(&self) -> Option<&FailedRecord> {
        self.records.front()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn drain(&mut self) -> Vec<FailedRecord> {
        self.records.drain(..).collect()
    }

    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued
    }
}

/// Policy controlling retry behaviour with exponential backoff and jitter.
pub struct RetryPolicy {
    pub max_retries: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
}

impl RetryPolicy {
    pub fn new(
        max_retries: u32,
        base_delay: Duration,
        max_delay: Duration,
        backoff_multiplier: f64,
    ) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
            backoff_multiplier,
        }
    }

    /// Compute the delay for the given attempt using exponential backoff with
    /// deterministic jitter (±25 % spread derived from the attempt number).
    pub fn compute_delay(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let delay_ms = base_ms * self.backoff_multiplier.powi(attempt as i32);
        let max_ms = self.max_delay.as_millis() as f64;
        let capped = delay_ms.min(max_ms);

        // Deterministic jitter: vary by ±25 % based on attempt number
        let jitter_factor = 1.0 + 0.25 * ((attempt as f64 * 1.618).sin());
        let final_ms = (capped * jitter_factor).min(max_ms).max(0.0);
        Duration::from_millis(final_ms as u64)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline health
// ---------------------------------------------------------------------------

/// Snapshot of overall pipeline health.
pub struct PipelineHealth {
    pub is_healthy: bool,
    pub backpressure_state: BackpressureState,
    pub dlq_size: usize,
    pub buffer_utilization: f32,
    pub records_per_second: f64,
    pub uptime: Duration,
}

impl PipelineHealth {
    /// Derive health from current pipeline stats and component states.
    pub fn from_stats(
        stats: &PipelineStats,
        backpressure: &BackpressureController,
        dlq: &DeadLetterQueue,
        uptime: Duration,
    ) -> Self {
        let bp_state = backpressure.state();
        let dlq_size = dlq.len();
        let is_healthy = bp_state == BackpressureState::Flowing && dlq_size < 100;

        Self {
            is_healthy,
            backpressure_state: bp_state,
            dlq_size,
            buffer_utilization: stats.buffer_utilization,
            records_per_second: stats.avg_throughput_per_sec,
            uptime,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_record(id: &str, text: &str) -> Record {
        Record {
            id: id.to_string(),
            data: RecordData::Text(text.to_string()),
            timestamp: 1_700_000_000_000,
            source: "test".to_string(),
            metadata: HashMap::new(),
        }
    }

    fn make_record_with_meta(id: &str, text: &str, key: &str, val: serde_json::Value) -> Record {
        let mut meta = HashMap::new();
        meta.insert(key.to_string(), val);
        Record {
            id: id.to_string(),
            data: RecordData::Text(text.to_string()),
            timestamp: 1_700_000_000_000,
            source: "test".to_string(),
            metadata: meta,
        }
    }

    #[test]
    fn test_pipeline_basic_flow() {
        let records = vec![make_record("r1", "hello"), make_record("r2", "world")];
        let source = InMemorySource::new("src", records);
        let sink = InMemorySink::new("dst");

        let config = PipelineConfig {
            batch_size: 10,
            enable_dedup: false,
            ..Default::default()
        };
        let mut pipeline = IngestionPipeline::new(config, Box::new(source), Box::new(sink));

        let processed = pipeline.process_batch().unwrap();
        assert_eq!(processed, 2);

        let flushed = pipeline.flush().unwrap();
        assert_eq!(flushed, 2);
        assert_eq!(pipeline.stats().records_written, 2);
    }

    #[test]
    fn test_pipeline_with_transform() {
        let records = vec![make_record("r1", "hello world")];
        let source = InMemorySource::new("src", records);
        let sink = InMemorySink::new("dst");

        let config = PipelineConfig {
            batch_size: 10,
            enable_dedup: false,
            ..Default::default()
        };
        let mut pipeline = IngestionPipeline::new(config, Box::new(source), Box::new(sink));
        pipeline.add_transform(Box::new(TextToVectorTransform::new(128)));

        pipeline.process_batch().unwrap();
        pipeline.flush().unwrap();

        assert_eq!(pipeline.stats().records_written, 1);
        assert_eq!(pipeline.stats().records_transformed, 1);
    }

    #[test]
    fn test_text_to_vector_transform() {
        let t = TextToVectorTransform::new(64);
        let record = make_record("r1", "some text");

        let result = t.apply(record).unwrap().unwrap();
        match &result.data {
            RecordData::Vector(v) => {
                assert_eq!(v.len(), 64);
                // Deterministic: same input → same output
                let record2 = make_record("r2", "some text");
                let result2 = t.apply(record2).unwrap().unwrap();
                if let RecordData::Vector(v2) = &result2.data {
                    assert_eq!(v, v2);
                } else {
                    panic!("expected Vector");
                }
            }
            _ => panic!("expected Vector"),
        }
    }

    #[test]
    fn test_filter_transform() {
        let filter = FilterTransform::new("category", json!("books"));

        let matching = make_record_with_meta("r1", "text", "category", json!("books"));
        let non_matching = make_record_with_meta("r2", "text", "category", json!("movies"));
        let missing_field = make_record("r3", "text");

        assert!(filter.apply(matching).unwrap().is_some());
        assert!(filter.apply(non_matching).unwrap().is_none());
        assert!(filter.apply(missing_field).unwrap().is_none());
    }

    #[test]
    fn test_deduplication() {
        let records = vec![
            make_record("r1", "hello"),
            make_record("r1", "hello again"),
            make_record("r2", "world"),
        ];
        let source = InMemorySource::new("src", records);
        let sink = InMemorySink::new("dst");

        let config = PipelineConfig {
            batch_size: 10,
            enable_dedup: true,
            ..Default::default()
        };
        let mut pipeline = IngestionPipeline::new(config, Box::new(source), Box::new(sink));

        pipeline.process_batch().unwrap();
        pipeline.flush().unwrap();

        assert_eq!(pipeline.stats().records_written, 2);
        assert_eq!(pipeline.stats().duplicates_skipped, 1);
    }

    #[test]
    fn test_batch_flushing() {
        let records: Vec<Record> = (0..15)
            .map(|i| make_record(&format!("r{}", i), &format!("text {}", i)))
            .collect();
        let source = InMemorySource::new("src", records);
        let sink = InMemorySink::new("dst");

        let config = PipelineConfig {
            batch_size: 10,
            enable_dedup: false,
            ..Default::default()
        };
        let mut pipeline = IngestionPipeline::new(config, Box::new(source), Box::new(sink));

        // First batch: pulls 10, buffer reaches batch_size → auto-flushes
        let processed = pipeline.process_batch().unwrap();
        assert_eq!(processed, 10);
        assert_eq!(pipeline.stats().records_written, 10);

        // Second batch: pulls remaining 5
        pipeline.process_batch().unwrap();
        pipeline.flush().unwrap();
        assert_eq!(pipeline.stats().records_written, 15);
    }

    #[test]
    fn test_backpressure_detection() {
        let source = InMemorySource::new("src", vec![]);
        let sink = InMemorySink::new("dst");

        let config = PipelineConfig {
            batch_size: 1000,
            max_buffer_size: 100,
            backpressure_threshold: 0.8,
            enable_dedup: false,
            ..Default::default()
        };
        let mut pipeline = IngestionPipeline::new(config, Box::new(source), Box::new(sink));

        assert!(!pipeline.is_backpressured());

        // Manually fill the buffer beyond threshold
        for i in 0..85 {
            pipeline.buffer.push(make_record(&format!("r{}", i), "x"));
        }
        assert!(pipeline.is_backpressured());
    }

    #[test]
    fn test_pipeline_stats() {
        let records = vec![make_record("r1", "hello"), make_record("r2", "world")];
        let source = InMemorySource::new("src", records);
        let sink = InMemorySink::new("dst");

        let config = PipelineConfig {
            batch_size: 10,
            enable_dedup: false,
            ..Default::default()
        };
        let mut pipeline = IngestionPipeline::new(config, Box::new(source), Box::new(sink));

        pipeline.process_batch().unwrap();
        assert_eq!(pipeline.stats().records_read, 2);
        assert_eq!(pipeline.stats().records_transformed, 2);

        pipeline.flush().unwrap();
        assert_eq!(pipeline.stats().records_written, 2);
        assert!(pipeline.stats().last_flush_at.is_some());

        pipeline.reset_stats();
        assert_eq!(pipeline.stats().records_read, 0);
        assert_eq!(pipeline.stats().records_written, 0);
    }

    #[test]
    fn test_empty_source() {
        let source = InMemorySource::new("src", vec![]);
        let sink = InMemorySink::new("dst");

        let mut pipeline =
            IngestionPipeline::new(PipelineConfig::default(), Box::new(source), Box::new(sink));

        let processed = pipeline.process_batch().unwrap();
        assert_eq!(processed, 0);

        let flushed = pipeline.flush().unwrap();
        assert_eq!(flushed, 0);
    }

    #[test]
    fn test_pipeline_builder() {
        let source = InMemorySource::new("src", vec![make_record("r1", "hi")]);
        let sink = InMemorySink::new("dst");

        let mut pipeline = PipelineBuilder::new()
            .with_config(PipelineConfig {
                batch_size: 5,
                enable_dedup: false,
                ..Default::default()
            })
            .with_source(Box::new(source))
            .with_sink(Box::new(sink))
            .add_transform(Box::new(IdentityTransform))
            .build()
            .unwrap();

        pipeline.process_batch().unwrap();
        pipeline.flush().unwrap();
        assert_eq!(pipeline.stats().records_written, 1);
    }

    #[test]
    fn test_pipeline_builder_missing_source() {
        let sink = InMemorySink::new("dst");
        let result = PipelineBuilder::new().with_sink(Box::new(sink)).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_transform() {
        let t = IdentityTransform;
        assert_eq!(t.name(), "identity");

        let record = make_record("r1", "hello");
        let result = t.apply(record.clone()).unwrap().unwrap();
        assert_eq!(result.id, "r1");
    }

    #[test]
    fn test_acknowledge_source() {
        let records = vec![make_record("r1", "hello"), make_record("r2", "world")];
        let mut source = InMemorySource::new("src", records);

        assert_eq!(source.pending_count(), 2);

        let r = source.pull().unwrap().unwrap();
        assert_eq!(r.id, "r1");
        assert_eq!(source.pending_count(), 1);

        source.acknowledge("r1").unwrap();
        assert!(source.acknowledged.contains("r1"));
        assert!(!source.acknowledged.contains("r2"));
    }

    // -----------------------------------------------------------------------
    // Checkpoint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_save_load_delete() {
        let mut store = InMemoryCheckpointStore::new();

        let mut cp = Checkpoint::new("pipe-1");
        cp.last_processed_id = Some("r42".to_string());
        cp.last_processed_offset = 42;
        cp.records_processed = 100;
        cp.state.insert("key".to_string(), "value".to_string());

        store.save(&cp).unwrap();

        let loaded = store.load("pipe-1").unwrap().unwrap();
        assert_eq!(loaded.pipeline_id, "pipe-1");
        assert_eq!(loaded.last_processed_id.as_deref(), Some("r42"));
        assert_eq!(loaded.last_processed_offset, 42);
        assert_eq!(loaded.records_processed, 100);
        assert_eq!(loaded.state.get("key").map(|s| s.as_str()), Some("value"));

        store.delete("pipe-1").unwrap();
        assert!(store.load("pipe-1").unwrap().is_none());
    }

    #[test]
    fn test_checkpoint_load_missing() {
        let store = InMemoryCheckpointStore::new();
        assert!(store.load("nonexistent").unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // BackpressureController tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_backpressure_flowing_to_paused_to_flowing() {
        let mut bp = BackpressureController::new(100, 0.8, 0.5);
        assert_eq!(bp.state(), BackpressureState::Flowing);
        assert!(!bp.should_pause());
        assert!(bp.should_resume());

        // Push above pause threshold → Paused
        bp.update_buffer_size(85);
        assert_eq!(bp.state(), BackpressureState::Paused);
        assert!(bp.should_pause());
        assert!(!bp.should_resume());

        // Still above resume threshold → stays Paused
        bp.update_buffer_size(60);
        assert_eq!(bp.state(), BackpressureState::Paused);

        // Drop to resume threshold → Flowing
        bp.update_buffer_size(50);
        assert_eq!(bp.state(), BackpressureState::Flowing);
        assert!(!bp.should_pause());
    }

    #[test]
    fn test_backpressure_utilization() {
        let mut bp = BackpressureController::new(200, 0.8, 0.5);
        assert!((bp.utilization() - 0.0).abs() < f32::EPSILON);

        bp.update_buffer_size(100);
        assert!((bp.utilization() - 0.5).abs() < f32::EPSILON);

        bp.update_buffer_size(200);
        assert!((bp.utilization() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_backpressure_stats() {
        let mut bp = BackpressureController::new(100, 0.8, 0.5);
        bp.update_buffer_size(90);
        let stats = bp.stats();
        assert_eq!(stats.state, BackpressureState::Paused);
        assert_eq!(stats.current_buffer_size, 90);
        assert_eq!(stats.max_buffer_size, 100);
        assert!(stats.utilization >= 0.89);
    }

    #[test]
    fn test_backpressure_zero_max() {
        let bp = BackpressureController::new(0, 0.8, 0.5);
        assert!((bp.utilization() - 0.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // DeadLetterQueue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dlq_enqueue_dequeue() {
        let mut dlq = DeadLetterQueue::new(10);
        assert!(dlq.is_empty());
        assert_eq!(dlq.len(), 0);

        dlq.enqueue(make_record("r1", "data"), "some error", 3);
        assert_eq!(dlq.len(), 1);
        assert!(!dlq.is_empty());
        assert_eq!(dlq.total_enqueued(), 1);

        let failed = dlq.peek().unwrap();
        assert_eq!(failed.record.id, "r1");
        assert_eq!(failed.error, "some error");
        assert_eq!(failed.attempt_count, 3);

        let dequeued = dlq.dequeue().unwrap();
        assert_eq!(dequeued.record.id, "r1");
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_eviction() {
        let mut dlq = DeadLetterQueue::new(2);
        dlq.enqueue(make_record("r1", "a"), "err1", 1);
        dlq.enqueue(make_record("r2", "b"), "err2", 1);
        dlq.enqueue(make_record("r3", "c"), "err3", 1);

        // Oldest (r1) should have been evicted
        assert_eq!(dlq.len(), 2);
        assert_eq!(dlq.total_enqueued(), 3);
        let first = dlq.dequeue().unwrap();
        assert_eq!(first.record.id, "r2");
    }

    #[test]
    fn test_dlq_drain() {
        let mut dlq = DeadLetterQueue::new(10);
        dlq.enqueue(make_record("r1", "a"), "e1", 1);
        dlq.enqueue(make_record("r2", "b"), "e2", 2);

        let drained = dlq.drain();
        assert_eq!(drained.len(), 2);
        assert!(dlq.is_empty());
    }

    // -----------------------------------------------------------------------
    // RetryPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_retry_policy_exponential_backoff() {
        let policy = RetryPolicy::new(
            5,
            Duration::from_millis(100),
            Duration::from_secs(30),
            2.0,
        );

        let d0 = policy.compute_delay(0);
        let d1 = policy.compute_delay(1);
        let d2 = policy.compute_delay(2);

        // Delays should generally increase (with jitter)
        // Base: 100, 200, 400 ms before jitter
        assert!(d0.as_millis() > 0);
        assert!(d1.as_millis() > 0);
        assert!(d2.as_millis() > 0);
        // d2 base (400ms) should be larger than d0 base (100ms) even after jitter
        assert!(d2.as_millis() > d0.as_millis());
    }

    #[test]
    fn test_retry_policy_capped_at_max() {
        let policy = RetryPolicy::new(
            10,
            Duration::from_millis(100),
            Duration::from_millis(500),
            2.0,
        );

        let d10 = policy.compute_delay(10);
        assert!(d10 <= Duration::from_millis(500));
    }

    #[test]
    fn test_retry_policy_default() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.base_delay, Duration::from_millis(100));
        assert_eq!(policy.max_delay, Duration::from_secs(30));
        assert!((policy.backoff_multiplier - 2.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // PipelineHealth tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_health_healthy() {
        let stats = PipelineStats {
            avg_throughput_per_sec: 500.0,
            buffer_utilization: 0.3,
            ..Default::default()
        };
        let bp = BackpressureController::new(100, 0.8, 0.5);
        let dlq = DeadLetterQueue::new(100);
        let uptime = Duration::from_secs(60);

        let health = PipelineHealth::from_stats(&stats, &bp, &dlq, uptime);
        assert!(health.is_healthy);
        assert_eq!(health.backpressure_state, BackpressureState::Flowing);
        assert_eq!(health.dlq_size, 0);
        assert!((health.buffer_utilization - 0.3).abs() < f32::EPSILON);
        assert!((health.records_per_second - 500.0).abs() < f64::EPSILON);
        assert_eq!(health.uptime, Duration::from_secs(60));
    }

    #[test]
    fn test_pipeline_health_unhealthy_backpressure() {
        let stats = PipelineStats::default();
        let mut bp = BackpressureController::new(100, 0.8, 0.5);
        bp.update_buffer_size(90); // triggers Paused
        let dlq = DeadLetterQueue::new(100);

        let health = PipelineHealth::from_stats(&stats, &bp, &dlq, Duration::from_secs(10));
        assert!(!health.is_healthy);
        assert_eq!(health.backpressure_state, BackpressureState::Paused);
    }

    #[test]
    fn test_pipeline_health_unhealthy_dlq() {
        let stats = PipelineStats::default();
        let bp = BackpressureController::new(100, 0.8, 0.5);
        let mut dlq = DeadLetterQueue::new(200);
        for i in 0..100 {
            dlq.enqueue(make_record(&format!("f{}", i), "x"), "err", 3);
        }

        let health = PipelineHealth::from_stats(&stats, &bp, &dlq, Duration::from_secs(10));
        assert!(!health.is_healthy);
        assert_eq!(health.dlq_size, 100);
    }
}
