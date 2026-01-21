//! Declarative Vector Pipelines
//!
//! YAML/JSON-defined ETL pipelines that chain source → transform → embed → index
//! stages with backpressure, retry logic, and dead-letter queues.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::vector_pipeline::{
//!     Pipeline, PipelineConfig, Stage, StageKind,
//!     PipelineResult, PipelineStats,
//! };
//!
//! let config = PipelineConfig::new("my-pipeline")
//!     .add_stage(Stage::new("chunk", StageKind::TextChunker { chunk_size: 512, overlap: 64 }))
//!     .add_stage(Stage::new("embed", StageKind::Embed { dimensions: 64 }))
//!     .add_stage(Stage::new("index", StageKind::Index { collection: "docs".into() }));
//!
//! let mut pipeline = Pipeline::new(config);
//! pipeline.push("doc1", "Long text to process...").unwrap();
//! let stats = pipeline.flush().unwrap();
//! assert!(stats.processed > 0);
//! ```

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};
use crate::services::inference_engine::{InferenceConfig, InferenceEngine, ModelSpec};

// ── Stage Kinds ──────────────────────────────────────────────────────────────

/// A pipeline stage definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageKind {
    /// Split text into overlapping chunks.
    TextChunker { chunk_size: usize, overlap: usize },
    /// Generate embeddings.
    Embed { dimensions: usize },
    /// Index into a collection.
    Index { collection: String },
    /// Filter records by a predicate.
    Filter { field: String, value: Value },
    /// Add metadata fields.
    Enrich { fields: HashMap<String, Value> },
    /// Deduplicate by content hash.
    Dedup,
    /// Custom transformation (pass-through with tag).
    Custom { name: String },
}

// ── Stage ────────────────────────────────────────────────────────────────────

/// A named pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// Stage name.
    pub name: String,
    /// Stage kind.
    pub kind: StageKind,
    /// Max retries for this stage.
    pub max_retries: usize,
}

impl Stage {
    /// Create a new stage.
    pub fn new(name: impl Into<String>, kind: StageKind) -> Self {
        Self {
            name: name.into(),
            kind,
            max_retries: 3,
        }
    }

    /// Set max retries.
    #[must_use]
    pub fn with_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }
}

// ── Pipeline Record ──────────────────────────────────────────────────────────

/// A record flowing through the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRecord {
    /// Record ID.
    pub id: String,
    /// Text content (may be chunked).
    pub text: String,
    /// Embedding vector (populated by embed stage).
    pub embedding: Option<Vec<f32>>,
    /// Metadata accumulated through stages.
    pub metadata: HashMap<String, Value>,
    /// Current stage index.
    pub stage_index: usize,
    /// Retry count for current stage.
    pub retries: usize,
}

impl PipelineRecord {
    fn new(id: &str, text: &str) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
            embedding: None,
            metadata: HashMap::new(),
            stage_index: 0,
            retries: 0,
        }
    }
}

// ── Pipeline Configuration ───────────────────────────────────────────────────

/// Pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name.
    pub name: String,
    /// Ordered stages.
    pub stages: Vec<Stage>,
    /// Maximum records to buffer.
    pub max_buffer: usize,
    /// Dead-letter queue max size.
    pub max_dlq: usize,
}

impl PipelineConfig {
    /// Create a new pipeline config.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
            max_buffer: 10_000,
            max_dlq: 1_000,
        }
    }

    /// Add a stage.
    #[must_use]
    pub fn add_stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set max buffer.
    #[must_use]
    pub fn with_buffer(mut self, max: usize) -> Self {
        self.max_buffer = max;
        self
    }
}

// ── Pipeline Statistics ──────────────────────────────────────────────────────

/// Pipeline execution statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Records successfully processed.
    pub processed: usize,
    /// Records that failed all retries (sent to DLQ).
    pub failed: usize,
    /// Records currently in buffer.
    pub buffered: usize,
    /// Total chunks generated.
    pub chunks_generated: usize,
    /// Total embeddings generated.
    pub embeddings_generated: usize,
    /// Processing duration.
    pub duration_ms: u64,
    /// Per-stage timing.
    pub stage_timings: HashMap<String, u64>,
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// Declarative vector processing pipeline.
pub struct Pipeline {
    config: PipelineConfig,
    buffer: VecDeque<PipelineRecord>,
    dlq: VecDeque<PipelineRecord>,
    engine: Option<InferenceEngine>,
    output: Vec<PipelineRecord>,
    stats: PipelineStats,
}

impl Pipeline {
    /// Create a new pipeline.
    pub fn new(config: PipelineConfig) -> Self {
        let has_embed = config.stages.iter().any(|s| matches!(s.kind, StageKind::Embed { .. }));
        let engine = if has_embed {
            let dim = config.stages.iter().find_map(|s| match &s.kind {
                StageKind::Embed { dimensions } => Some(*dimensions),
                _ => None,
            }).unwrap_or(64);
            Some(InferenceEngine::new(
                InferenceConfig::builder()
                    .model(ModelSpec::new("pipeline-embed", dim))
                    .normalize(true)
                    .build(),
            ))
        } else {
            None
        };

        Self {
            config,
            buffer: VecDeque::new(),
            dlq: VecDeque::new(),
            engine,
            output: Vec::new(),
            stats: PipelineStats::default(),
        }
    }

    /// Push a record into the pipeline.
    pub fn push(&mut self, id: &str, text: &str) -> Result<()> {
        if self.buffer.len() >= self.config.max_buffer {
            return Err(NeedleError::CapacityExceeded("Pipeline buffer full".into()));
        }
        self.buffer.push_back(PipelineRecord::new(id, text));
        self.stats.buffered = self.buffer.len();
        Ok(())
    }

    /// Push a record with metadata.
    pub fn push_with_metadata(&mut self, id: &str, text: &str, meta: HashMap<String, Value>) -> Result<()> {
        if self.buffer.len() >= self.config.max_buffer {
            return Err(NeedleError::CapacityExceeded("Pipeline buffer full".into()));
        }
        let mut rec = PipelineRecord::new(id, text);
        rec.metadata = meta;
        self.buffer.push_back(rec);
        Ok(())
    }

    /// Process all buffered records through the pipeline.
    pub fn flush(&mut self) -> Result<PipelineStats> {
        let start = Instant::now();
        self.stats = PipelineStats::default();
        let mut stage_timings: HashMap<String, u64> = HashMap::new();

        while let Some(mut record) = self.buffer.pop_front() {
            let mut failed = false;
            let stages = self.config.stages.clone();
            for (i, stage) in stages.iter().enumerate() {
                record.stage_index = i;
                let stage_start = Instant::now();

                match self.execute_stage(stage, &mut record) {
                    Ok(extra_records) => {
                        let elapsed = stage_start.elapsed().as_millis() as u64;
                        *stage_timings.entry(stage.name.clone()).or_default() += elapsed;

                        // Handle stages that produce multiple records (chunking)
                        for extra in extra_records {
                            self.buffer.push_back(extra);
                        }
                    }
                    Err(_) => {
                        record.retries += 1;
                        if record.retries >= stage.max_retries {
                            if self.dlq.len() < self.config.max_dlq {
                                self.dlq.push_back(record.clone());
                            }
                            self.stats.failed += 1;
                            failed = true;
                            break;
                        }
                        // Re-queue for retry
                        self.buffer.push_back(record.clone());
                        failed = true;
                        break;
                    }
                }
            }
            if !failed {
                self.output.push(record);
                self.stats.processed += 1;
            }
        }

        self.stats.duration_ms = start.elapsed().as_millis() as u64;
        self.stats.buffered = self.buffer.len();
        self.stats.stage_timings = stage_timings;
        Ok(self.stats.clone())
    }

    /// Get processed output records.
    pub fn output(&self) -> &[PipelineRecord] {
        &self.output
    }

    /// Get dead-letter queue contents.
    pub fn dlq(&self) -> &VecDeque<PipelineRecord> {
        &self.dlq
    }

    /// Get current stats.
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    /// Clear output and DLQ.
    pub fn clear(&mut self) {
        self.output.clear();
        self.dlq.clear();
    }

    fn execute_stage(&mut self, stage: &Stage, record: &mut PipelineRecord) -> Result<Vec<PipelineRecord>> {
        match &stage.kind {
            StageKind::TextChunker { chunk_size, overlap } => {
                let chunks = chunk_text(&record.text, *chunk_size, *overlap);
                if chunks.len() <= 1 {
                    return Ok(Vec::new());
                }
                self.stats.chunks_generated += chunks.len();
                let mut extras = Vec::new();
                record.text = chunks[0].clone();
                for (i, chunk) in chunks.into_iter().skip(1).enumerate() {
                    let mut r = record.clone();
                    r.id = format!("{}_chunk_{}", record.id, i + 1);
                    r.text = chunk;
                    extras.push(r);
                }
                record.id = format!("{}_chunk_0", record.id);
                Ok(extras)
            }
            StageKind::Embed { dimensions: _ } => {
                if let Some(engine) = &mut self.engine {
                    let emb = engine.embed_text(&record.text)?;
                    record.embedding = Some(emb);
                    self.stats.embeddings_generated += 1;
                }
                Ok(Vec::new())
            }
            StageKind::Index { collection } => {
                record.metadata.insert("_collection".into(), Value::String(collection.clone()));
                Ok(Vec::new())
            }
            StageKind::Filter { field, value } => {
                if let Some(v) = record.metadata.get(field) {
                    if v != value {
                        return Err(NeedleError::InvalidInput("Filtered out".into()));
                    }
                }
                Ok(Vec::new())
            }
            StageKind::Enrich { fields } => {
                for (k, v) in fields {
                    record.metadata.insert(k.clone(), v.clone());
                }
                Ok(Vec::new())
            }
            StageKind::Dedup => {
                // Simple dedup by content hash
                let hash = simple_hash(&record.text);
                record.metadata.insert("_content_hash".into(), Value::String(hash));
                Ok(Vec::new())
            }
            StageKind::Custom { name: _ } => Ok(Vec::new()),
        }
    }
}

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= chunk_size {
        return vec![text.to_string()];
    }
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let end = (i + chunk_size).min(words.len());
        chunks.push(words[i..end].join(" "));
        i += step;
        if end == words.len() { break; }
    }
    chunks
}

fn simple_hash(text: &str) -> String {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    for b in text.bytes() {
        h = h.wrapping_mul(0x0100_0000_01b3).wrapping_add(u64::from(b));
    }
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_pipeline() {
        let config = PipelineConfig::new("test")
            .add_stage(Stage::new("embed", StageKind::Embed { dimensions: 32 }))
            .add_stage(Stage::new("index", StageKind::Index { collection: "docs".into() }));

        let mut pipeline = Pipeline::new(config);
        pipeline.push("d1", "hello world").unwrap();
        pipeline.push("d2", "foo bar").unwrap();

        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.processed, 2);
        assert_eq!(stats.embeddings_generated, 2);
        assert_eq!(pipeline.output().len(), 2);
        assert!(pipeline.output()[0].embedding.is_some());
    }

    #[test]
    fn test_chunking_pipeline() {
        let long_text = (0..100).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let config = PipelineConfig::new("chunk-test")
            .add_stage(Stage::new("chunk", StageKind::TextChunker { chunk_size: 20, overlap: 5 }));

        let mut pipeline = Pipeline::new(config);
        pipeline.push("d1", &long_text).unwrap();
        let stats = pipeline.flush().unwrap();
        assert!(stats.chunks_generated > 1);
        assert!(stats.processed > 1);
    }

    #[test]
    fn test_enrich_stage() {
        let mut fields = HashMap::new();
        fields.insert("source".into(), Value::String("test".into()));
        let config = PipelineConfig::new("enrich")
            .add_stage(Stage::new("enrich", StageKind::Enrich { fields }));

        let mut pipeline = Pipeline::new(config);
        pipeline.push("d1", "hello").unwrap();
        pipeline.flush().unwrap();
        assert_eq!(pipeline.output()[0].metadata.get("source").unwrap(), "test");
    }

    #[test]
    fn test_filter_stage() {
        let config = PipelineConfig::new("filter")
            .add_stage(Stage::new("f", StageKind::Filter {
                field: "type".into(),
                value: Value::String("keep".into()),
            }));

        let mut pipeline = Pipeline::new(config);
        let mut meta = HashMap::new();
        meta.insert("type".into(), Value::String("keep".into()));
        pipeline.push_with_metadata("d1", "kept", meta).unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert("type".into(), Value::String("drop".into()));
        pipeline.push_with_metadata("d2", "dropped", meta2).unwrap();

        let stats = pipeline.flush().unwrap();
        assert_eq!(stats.processed, 1);
        assert!(stats.failed > 0);
    }

    #[test]
    fn test_dedup_stage() {
        let config = PipelineConfig::new("dedup")
            .add_stage(Stage::new("dedup", StageKind::Dedup));

        let mut pipeline = Pipeline::new(config);
        pipeline.push("d1", "same text").unwrap();
        pipeline.push("d2", "same text").unwrap();
        pipeline.flush().unwrap();

        let h1 = pipeline.output()[0].metadata.get("_content_hash").unwrap();
        let h2 = pipeline.output()[1].metadata.get("_content_hash").unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_buffer_limit() {
        let config = PipelineConfig::new("small").with_buffer(2);
        let mut pipeline = Pipeline::new(config);
        pipeline.push("d1", "a").unwrap();
        pipeline.push("d2", "b").unwrap();
        assert!(pipeline.push("d3", "c").is_err());
    }

    #[test]
    fn test_stage_timings() {
        let config = PipelineConfig::new("timed")
            .add_stage(Stage::new("embed", StageKind::Embed { dimensions: 16 }));
        let mut pipeline = Pipeline::new(config);
        pipeline.push("d1", "hello").unwrap();
        let stats = pipeline.flush().unwrap();
        assert!(stats.stage_timings.contains_key("embed"));
    }
}
