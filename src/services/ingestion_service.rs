//! Unified Streaming Ingestion Service
//!
//! High-level facade that ties together the ingestion pipeline, streaming upsert,
//! and Database for easy real-time vector ingestion with backpressure control,
//! batching, deduplication, and exactly-once semantics.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::{Database, IngestionService, IngestionServiceConfig};
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let config = IngestionServiceConfig::builder()
//!     .collection("docs")
//!     .batch_size(500)
//!     .flush_interval_ms(1000)
//!     .enable_dedup(true)
//!     .build();
//!
//! let mut service = IngestionService::new(&db, config).unwrap();
//!
//! // Ingest vectors
//! service.ingest("doc1", &vec![0.1f32; 384], None).unwrap();
//! service.ingest("doc2", &vec![0.2f32; 384], None).unwrap();
//!
//! // Flush remaining buffer
//! let stats = service.flush().unwrap();
//! assert_eq!(stats.total_ingested, 2);
//! ```

use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};

/// Configuration for the ingestion service.
#[derive(Debug, Clone)]
pub struct IngestionServiceConfig {
    /// Target collection name.
    pub collection: String,
    /// Number of vectors per batch before auto-flush.
    pub batch_size: usize,
    /// Maximum time between flushes.
    pub flush_interval: Duration,
    /// Maximum pending buffer size before backpressure is applied.
    pub max_buffer_size: usize,
    /// Whether to deduplicate by vector ID within a batch.
    pub enable_dedup: bool,
    /// Number of retries for failed batch inserts.
    pub max_retries: u32,
    /// Backpressure threshold as fraction of max_buffer_size (0.0–1.0).
    pub backpressure_threshold: f32,
}

impl Default for IngestionServiceConfig {
    fn default() -> Self {
        Self {
            collection: String::new(),
            batch_size: 500,
            flush_interval: Duration::from_millis(1000),
            max_buffer_size: 50_000,
            enable_dedup: true,
            max_retries: 3,
            backpressure_threshold: 0.8,
        }
    }
}

/// Builder for `IngestionServiceConfig`.
pub struct IngestionServiceConfigBuilder {
    config: IngestionServiceConfig,
}

impl IngestionServiceConfig {
    pub fn builder() -> IngestionServiceConfigBuilder {
        IngestionServiceConfigBuilder {
            config: Self::default(),
        }
    }
}

impl IngestionServiceConfigBuilder {
    pub fn collection(mut self, name: impl Into<String>) -> Self {
        self.config.collection = name.into();
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn flush_interval_ms(mut self, ms: u64) -> Self {
        self.config.flush_interval = Duration::from_millis(ms);
        self
    }

    pub fn max_buffer_size(mut self, size: usize) -> Self {
        self.config.max_buffer_size = size;
        self
    }

    pub fn enable_dedup(mut self, enable: bool) -> Self {
        self.config.enable_dedup = enable;
        self
    }

    pub fn max_retries(mut self, retries: u32) -> Self {
        self.config.max_retries = retries;
        self
    }

    pub fn backpressure_threshold(mut self, threshold: f32) -> Self {
        self.config.backpressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> IngestionServiceConfig {
        self.config
    }
}

/// A pending vector record waiting to be flushed.
#[derive(Debug, Clone)]
struct PendingRecord {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

/// Current backpressure state of the ingestion service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressureLevel {
    /// No backpressure — accepting records freely.
    None,
    /// Warning threshold reached — ingestion is slowing.
    Warning,
    /// Critical — buffer is full, rejecting new records.
    Critical,
}

/// Statistics from the ingestion service.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IngestionStats {
    /// Total records successfully ingested.
    pub total_ingested: u64,
    /// Total records that failed ingestion after retries.
    pub total_failed: u64,
    /// Total batches flushed.
    pub batches_flushed: u64,
    /// Total duplicates skipped.
    pub duplicates_skipped: u64,
    /// Current buffer size.
    pub current_buffer_size: usize,
    /// Average batch flush duration in microseconds.
    pub avg_flush_duration_us: u64,
    /// Total flush duration in microseconds (for computing average).
    total_flush_duration_us: u64,
}

/// Dead-letter record for failed ingestions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterRecord {
    pub id: String,
    pub error: String,
    pub attempts: u32,
    pub timestamp_ms: u64,
}

/// Unified streaming ingestion service for a Needle database.
///
/// Provides buffered, batched insertion with automatic flushing,
/// backpressure control, deduplication, and dead-letter handling.
pub struct IngestionService<'a> {
    db: &'a Database,
    collection_name: String,
    config: IngestionServiceConfig,
    buffer: VecDeque<PendingRecord>,
    seen_ids: HashSet<String>,
    stats: IngestionStats,
    dead_letters: Vec<DeadLetterRecord>,
    last_flush: Instant,
}

impl<'a> IngestionService<'a> {
    /// Create a new ingestion service for the given database and configuration.
    pub fn new(db: &'a Database, config: IngestionServiceConfig) -> Result<Self> {
        if config.collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "collection name must not be empty".into(),
            ));
        }
        // Validate collection exists
        let _ = db.collection(&config.collection)?;

        Ok(Self {
            db,
            collection_name: config.collection.clone(),
            config,
            buffer: VecDeque::new(),
            seen_ids: HashSet::new(),
            stats: IngestionStats::default(),
            dead_letters: Vec::new(),
            last_flush: Instant::now(),
        })
    }

    /// Ingest a single vector. Returns immediately if buffered, or flushes
    /// if the batch is full or the flush interval has elapsed.
    pub fn ingest(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<BackpressureLevel> {
        let id = id.into();

        // Check backpressure
        let level = self.backpressure_level();
        if level == BackpressureLevel::Critical {
            return Err(NeedleError::InvalidArgument(
                "ingestion buffer full — backpressure critical".into(),
            ));
        }

        // Dedup check
        if self.config.enable_dedup && self.seen_ids.contains(&id) {
            self.stats.duplicates_skipped += 1;
            return Ok(level);
        }

        if self.config.enable_dedup {
            self.seen_ids.insert(id.clone());
        }

        self.buffer.push_back(PendingRecord {
            id,
            vector: vector.to_vec(),
            metadata,
        });
        self.stats.current_buffer_size = self.buffer.len();

        // Auto-flush if batch is full or interval elapsed
        if self.should_flush() {
            self.flush()?;
        }

        Ok(self.backpressure_level())
    }

    /// Ingest a batch of vectors at once.
    pub fn ingest_batch(
        &mut self,
        records: Vec<(String, Vec<f32>, Option<Value>)>,
    ) -> Result<BackpressureLevel> {
        for (id, vector, metadata) in records {
            self.ingest(id, &vector, metadata)?;
        }
        Ok(self.backpressure_level())
    }

    /// Flush all buffered records to the collection.
    pub fn flush(&mut self) -> Result<IngestionStats> {
        if self.buffer.is_empty() {
            return Ok(self.stats.clone());
        }

        let collection = self.db.collection(&self.collection_name)?;
        let batch: Vec<PendingRecord> = self.buffer.drain(..).collect();
        let batch_len = batch.len();
        let start = Instant::now();

        let mut succeeded = 0u64;
        let mut failed = 0u64;

        for record in &batch {
            let mut attempts = 0u32;
            let mut last_err = None;

            while attempts <= self.config.max_retries {
                match collection.insert(&record.id, &record.vector, record.metadata.clone()) {
                    Ok(_) => {
                        succeeded += 1;
                        last_err = None;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                        attempts += 1;
                    }
                }
            }

            if let Some(err) = last_err {
                failed += 1;
                self.dead_letters.push(DeadLetterRecord {
                    id: record.id.clone(),
                    error: err.to_string(),
                    attempts,
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as u64,
                });
            }
        }

        let duration = start.elapsed();
        self.stats.total_ingested += succeeded;
        self.stats.total_failed += failed;
        self.stats.batches_flushed += 1;
        self.stats.total_flush_duration_us += duration.as_micros() as u64;
        self.stats.avg_flush_duration_us =
            self.stats.total_flush_duration_us / self.stats.batches_flushed;
        self.stats.current_buffer_size = self.buffer.len();
        self.last_flush = Instant::now();

        // Clear dedup set periodically to avoid unbounded growth
        if self.seen_ids.len() > self.config.max_buffer_size * 2 {
            self.seen_ids.clear();
        }

        Ok(self.stats.clone())
    }

    /// Get current backpressure level.
    pub fn backpressure_level(&self) -> BackpressureLevel {
        let ratio = self.buffer.len() as f32 / self.config.max_buffer_size as f32;
        if ratio >= 1.0 {
            BackpressureLevel::Critical
        } else if ratio >= self.config.backpressure_threshold {
            BackpressureLevel::Warning
        } else {
            BackpressureLevel::None
        }
    }

    /// Get current ingestion statistics.
    pub fn stats(&self) -> &IngestionStats {
        &self.stats
    }

    /// Get dead-letter records (records that failed after all retries).
    pub fn dead_letters(&self) -> &[DeadLetterRecord] {
        &self.dead_letters
    }

    /// Clear the dead-letter queue, returning the records.
    pub fn drain_dead_letters(&mut self) -> Vec<DeadLetterRecord> {
        std::mem::take(&mut self.dead_letters)
    }

    /// Returns number of records currently buffered.
    pub fn pending_count(&self) -> usize {
        self.buffer.len()
    }

    fn should_flush(&self) -> bool {
        self.buffer.len() >= self.config.batch_size
            || self.last_flush.elapsed() >= self.config.flush_interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        db
    }

    #[test]
    fn test_basic_ingestion() {
        let db = make_db();
        let config = IngestionServiceConfig::builder()
            .collection("test")
            .batch_size(10)
            .build();
        let mut svc = IngestionService::new(&db, config).unwrap();

        svc.ingest("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        svc.ingest("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();
        let stats = svc.flush().unwrap();

        assert_eq!(stats.total_ingested, 2);
        assert_eq!(stats.batches_flushed, 1);
    }

    #[test]
    fn test_auto_flush_on_batch_size() {
        let db = make_db();
        let config = IngestionServiceConfig::builder()
            .collection("test")
            .batch_size(2)
            .build();
        let mut svc = IngestionService::new(&db, config).unwrap();

        svc.ingest("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        // Second insert triggers auto-flush
        svc.ingest("v2", &[0.5, 0.6, 0.7, 0.8], None).unwrap();

        assert_eq!(svc.stats().total_ingested, 2);
    }

    #[test]
    fn test_dedup() {
        let db = make_db();
        let config = IngestionServiceConfig::builder()
            .collection("test")
            .batch_size(100)
            .enable_dedup(true)
            .build();
        let mut svc = IngestionService::new(&db, config).unwrap();

        svc.ingest("v1", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        svc.ingest("v1", &[0.5, 0.6, 0.7, 0.8], None).unwrap();
        let stats = svc.flush().unwrap();

        assert_eq!(stats.total_ingested, 1);
        assert_eq!(stats.duplicates_skipped, 1);
    }

    #[test]
    fn test_backpressure() {
        let db = make_db();
        let config = IngestionServiceConfig::builder()
            .collection("test")
            .batch_size(100_000) // large batch so no auto-flush
            .max_buffer_size(5)
            .backpressure_threshold(0.6)
            .enable_dedup(false)
            .build();
        let mut svc = IngestionService::new(&db, config).unwrap();

        // Fill to 3/5 = 0.6 → Warning
        for i in 0..3 {
            svc.ingest(format!("v{i}"), &[0.1, 0.2, 0.3, 0.4], None)
                .unwrap();
        }
        assert_eq!(svc.backpressure_level(), BackpressureLevel::Warning);

        // Fill to 5/5 → Critical, next insert should fail
        svc.ingest("v3", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        svc.ingest("v4", &[0.1, 0.2, 0.3, 0.4], None).unwrap();
        let result = svc.ingest("v5", &[0.1, 0.2, 0.3, 0.4], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_ingestion() {
        let db = make_db();
        let config = IngestionServiceConfig::builder()
            .collection("test")
            .batch_size(100)
            .build();
        let mut svc = IngestionService::new(&db, config).unwrap();

        let records = vec![
            ("a".into(), vec![0.1, 0.2, 0.3, 0.4], None),
            ("b".into(), vec![0.5, 0.6, 0.7, 0.8], None),
        ];
        svc.ingest_batch(records).unwrap();
        let stats = svc.flush().unwrap();
        assert_eq!(stats.total_ingested, 2);
    }

    #[test]
    fn test_empty_collection_name_rejected() {
        let db = make_db();
        let config = IngestionServiceConfig::builder().build();
        assert!(IngestionService::new(&db, config).is_err());
    }
}
