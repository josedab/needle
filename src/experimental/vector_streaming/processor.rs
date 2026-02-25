#![allow(clippy::unwrap_used)]

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::debug;

use super::consumer::{VectorConsumer, VectorMessage};
use super::current_timestamp;
use super::pipeline::BackpressureController;
use super::producer::VectorProducer;

/// Stream processor for transforming vectors
pub struct StreamProcessor {
    /// Input consumer
    consumer: Arc<VectorConsumer>,
    /// Processing function
    processor: Box<dyn Fn(VectorMessage) -> Option<VectorMessage> + Send + Sync>,
    /// Output producer (optional)
    producer: Option<Arc<VectorProducer>>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new<F>(consumer: Arc<VectorConsumer>, processor: F) -> Self
    where
        F: Fn(VectorMessage) -> Option<VectorMessage> + Send + Sync + 'static,
    {
        Self {
            consumer,
            processor: Box::new(processor),
            producer: None,
        }
    }

    /// Set output producer
    #[must_use]
    pub fn with_producer(mut self, producer: Arc<VectorProducer>) -> Self {
        self.producer = Some(producer);
        self
    }

    /// Process one batch
    pub fn process_batch(&self) -> Result<usize> {
        let batch = match self.consumer.poll()? {
            Some(b) => b,
            None => return Ok(0),
        };

        let mut processed = 0;
        for msg in batch {
            if let Some(transformed) = (self.processor)(msg) {
                if let Some(ref producer) = self.producer {
                    producer.send(&transformed.id, &transformed.vector, transformed.metadata)?;
                }
                processed += 1;
            }
        }

        self.consumer.commit()?;
        Ok(processed)
    }
}

// ============================================================================
// Exactly-Once Semantics Enhancements
// ============================================================================

/// Checkpoint storage for exactly-once recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Consumer group ID
    pub group_id: String,
    /// Topic name
    pub topic: String,
    /// Partition offsets
    pub offsets: HashMap<u32, u64>,
    /// Processed message IDs (idempotency tokens)
    pub processed_ids: HashSet<String>,
    /// Watermark (latest fully processed timestamp)
    pub watermark: u64,
    /// Checkpoint timestamp
    pub timestamp: u64,
    /// Checkpoint sequence number
    pub sequence: u64,
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(group_id: &str, topic: &str) -> Self {
        Self {
            group_id: group_id.to_string(),
            topic: topic.to_string(),
            offsets: HashMap::new(),
            processed_ids: HashSet::new(),
            watermark: 0,
            timestamp: current_timestamp(),
            sequence: 0,
        }
    }

    /// Update offset for a partition
    pub fn update_offset(&mut self, partition: u32, offset: u64) {
        self.offsets.insert(partition, offset);
        self.timestamp = current_timestamp();
        self.sequence += 1;
    }

    /// Mark a message as processed
    pub fn mark_processed(&mut self, id: &str) {
        self.processed_ids.insert(id.to_string());
    }

    /// Check if a message was already processed
    pub fn is_processed(&self, id: &str) -> bool {
        self.processed_ids.contains(id)
    }

    /// Update watermark
    pub fn update_watermark(&mut self, watermark: u64) {
        if watermark > self.watermark {
            self.watermark = watermark;
        }
    }

    /// Trim old processed IDs to prevent unbounded growth
    pub fn trim_processed_ids(&mut self, max_size: usize) {
        if self.processed_ids.len() > max_size {
            // Keep only the most recent entries
            // In practice, would use a bounded LRU set
            let excess = self.processed_ids.len() - max_size;
            let to_remove: Vec<_> = self.processed_ids.iter().take(excess).cloned().collect();
            for id in to_remove {
                self.processed_ids.remove(&id);
            }
        }
    }
}

/// Checkpoint storage trait for persistence
pub trait CheckpointStore: Send + Sync {
    /// Save checkpoint
    fn save(&self, checkpoint: &Checkpoint) -> Result<()>;
    /// Load checkpoint
    fn load(&self, group_id: &str, topic: &str) -> Result<Option<Checkpoint>>;
    /// Delete checkpoint
    fn delete(&self, group_id: &str, topic: &str) -> Result<()>;
}

/// In-memory checkpoint store (for testing)
#[derive(Debug, Default)]
pub struct InMemoryCheckpointStore {
    checkpoints: RwLock<HashMap<String, Checkpoint>>,
}

impl InMemoryCheckpointStore {
    /// Create new in-memory store
    pub fn new() -> Self {
        Self::default()
    }

    fn key(group_id: &str, topic: &str) -> String {
        format!("{}:{}", group_id, topic)
    }
}

impl CheckpointStore for InMemoryCheckpointStore {
    fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        let key = Self::key(&checkpoint.group_id, &checkpoint.topic);
        self.checkpoints.write().insert(key, checkpoint.clone());
        Ok(())
    }

    fn load(&self, group_id: &str, topic: &str) -> Result<Option<Checkpoint>> {
        let key = Self::key(group_id, topic);
        Ok(self.checkpoints.read().get(&key).cloned())
    }

    fn delete(&self, group_id: &str, topic: &str) -> Result<()> {
        let key = Self::key(group_id, topic);
        self.checkpoints.write().remove(&key);
        Ok(())
    }
}

/// Dead letter queue for failed messages
#[derive(Debug)]
pub struct DeadLetterQueue {
    /// Failed messages
    messages: RwLock<VecDeque<FailedMessage>>,
    /// Maximum queue size
    max_size: usize,
    /// Statistics
    stats: RwLock<DlqStats>,
}

/// A message that failed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedMessage {
    /// Original message
    pub message: VectorMessage,
    /// Failure reason
    pub error: String,
    /// Number of retry attempts
    pub retry_count: u32,
    /// First failure timestamp
    pub first_failure: u64,
    /// Last failure timestamp
    pub last_failure: u64,
}

/// DLQ statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DlqStats {
    /// Total messages sent to DLQ
    pub total_failed: u64,
    /// Messages retried successfully
    pub retried_success: u64,
    /// Messages permanently failed
    pub permanently_failed: u64,
    /// Current queue size
    pub queue_size: usize,
}

impl DeadLetterQueue {
    /// Create a new DLQ
    pub fn new(max_size: usize) -> Self {
        Self {
            messages: RwLock::new(VecDeque::with_capacity(max_size)),
            max_size,
            stats: RwLock::new(DlqStats::default()),
        }
    }

    /// Add a failed message
    pub fn push(&self, message: VectorMessage, error: &str, retry_count: u32) {
        let now = current_timestamp();
        let failed = FailedMessage {
            message,
            error: error.to_string(),
            retry_count,
            first_failure: now,
            last_failure: now,
        };

        let mut queue = self.messages.write();
        let mut stats = self.stats.write();

        // Evict oldest if at capacity
        if queue.len() >= self.max_size {
            queue.pop_front();
            stats.permanently_failed += 1;
        }

        queue.push_back(failed);
        stats.total_failed += 1;
        stats.queue_size = queue.len();
    }

    /// Pop a message for retry
    pub fn pop(&self) -> Option<FailedMessage> {
        let mut queue = self.messages.write();
        let msg = queue.pop_front();
        if msg.is_some() {
            self.stats.write().queue_size = queue.len();
        }
        msg
    }

    /// Get all messages (for inspection)
    pub fn peek_all(&self) -> Vec<FailedMessage> {
        self.messages.read().iter().cloned().collect()
    }

    /// Get statistics
    pub fn stats(&self) -> DlqStats {
        self.stats.read().clone()
    }

    /// Mark a retry as successful
    pub fn mark_retry_success(&self) {
        self.stats.write().retried_success += 1;
    }

    /// Clear the queue
    pub fn clear(&self) {
        self.messages.write().clear();
        self.stats.write().queue_size = 0;
    }

    /// Get queue length
    pub fn len(&self) -> usize {
        self.messages.read().len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.messages.read().is_empty()
    }
}

/// Transactional batch processor with exactly-once semantics
pub struct TransactionalProcessor {
    /// Consumer
    consumer: Arc<VectorConsumer>,
    /// Checkpoint store
    checkpoint_store: Arc<dyn CheckpointStore>,
    /// Dead letter queue
    dlq: DeadLetterQueue,
    /// Current checkpoint
    checkpoint: RwLock<Checkpoint>,
    /// Processing function
    processor: Box<dyn Fn(&VectorMessage) -> Result<()> + Send + Sync>,
    /// Maximum retries before DLQ
    max_retries: u32,
    /// Checkpoint interval (number of messages)
    checkpoint_interval: usize,
    /// Messages since last checkpoint
    messages_since_checkpoint: AtomicU64,
    /// Optional backpressure controller
    backpressure: Option<Arc<BackpressureController>>,
}

impl TransactionalProcessor {
    /// Create a new transactional processor
    pub fn new<F>(
        consumer: Arc<VectorConsumer>,
        checkpoint_store: Arc<dyn CheckpointStore>,
        processor: F,
    ) -> Result<Self>
    where
        F: Fn(&VectorMessage) -> Result<()> + Send + Sync + 'static,
    {
        let config = &consumer.config;

        // Load or create checkpoint
        let checkpoint = checkpoint_store
            .load(&config.group_id, &config.topic)?
            .unwrap_or_else(|| Checkpoint::new(&config.group_id, &config.topic));

        Ok(Self {
            consumer,
            checkpoint_store,
            dlq: DeadLetterQueue::new(10000),
            checkpoint: RwLock::new(checkpoint),
            processor: Box::new(processor),
            max_retries: 3,
            checkpoint_interval: 1000,
            messages_since_checkpoint: AtomicU64::new(0),
            backpressure: None,
        })
    }

    /// Set maximum retries
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set checkpoint interval
    #[must_use]
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Attach a backpressure controller to throttle ingestion automatically.
    #[must_use]
    pub fn with_backpressure(mut self, controller: Arc<BackpressureController>) -> Self {
        self.backpressure = Some(controller);
        self
    }

    /// Process a batch with exactly-once semantics
    pub fn process_batch(&self) -> Result<BatchResult> {
        // Check backpressure before polling
        if let Some(bp) = &self.backpressure {
            if !bp.should_poll() {
                return Ok(BatchResult::default());
            }
        }

        let batch = match self.consumer.poll()? {
            Some(b) => b,
            None => return Ok(BatchResult::default()),
        };

        // Apply backpressure-aware batch limiting
        let batch: Vec<_> = if let Some(bp) = &self.backpressure {
            let effective = bp.effective_batch_size(batch.len());
            if effective == 0 {
                return Ok(BatchResult::default());
            }
            bp.on_receive(effective as u64);
            batch.into_iter().take(effective).collect()
        } else {
            batch
        };

        let mut processed = 0;
        let mut skipped = 0;
        let mut failed = 0;

        for msg in batch {
            // Check if already processed (idempotency)
            if self.checkpoint.read().is_processed(&msg.id) {
                skipped += 1;
                continue;
            }

            // Process with retry
            let result = self.process_with_retry(&msg);

            match result {
                Ok(()) => {
                    // Mark as processed
                    self.checkpoint.write().mark_processed(&msg.id);

                    // Update offset
                    if let Some(partition) = msg.partition {
                        self.checkpoint
                            .write()
                            .update_offset(partition as u32, msg.offset);
                    }

                    // Update watermark
                    self.checkpoint.write().update_watermark(msg.timestamp);

                    processed += 1;
                }
                Err(e) => {
                    // Send to DLQ
                    self.dlq.push(msg, &e.to_string(), self.max_retries);
                    failed += 1;
                }
            }
        }

        // Maybe checkpoint
        let count = self
            .messages_since_checkpoint
            .fetch_add(processed as u64, Ordering::Relaxed);
        if count + processed as u64 >= self.checkpoint_interval as u64 {
            self.save_checkpoint()?;
            self.messages_since_checkpoint.store(0, Ordering::Relaxed);
        }

        // Commit offsets
        self.consumer.commit()?;

        // Notify backpressure controller of committed messages
        if let Some(bp) = &self.backpressure {
            bp.on_commit(processed as u64);
        }

        Ok(BatchResult {
            processed,
            skipped,
            failed,
        })
    }

    /// Process a single message with retry
    fn process_with_retry(&self, msg: &VectorMessage) -> Result<()> {
        let mut attempts = 0;
        loop {
            match (self.processor)(msg) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_retries {
                        return Err(e);
                    }
                    // Simple backoff
                    std::thread::sleep(std::time::Duration::from_millis(100 * attempts as u64));
                }
            }
        }
    }

    /// Save checkpoint to store
    pub fn save_checkpoint(&self) -> Result<()> {
        let checkpoint = self.checkpoint.read().clone();
        self.checkpoint_store.save(&checkpoint)?;
        debug!(sequence = checkpoint.sequence, "Checkpoint saved");
        Ok(())
    }

    /// Get current checkpoint
    pub fn checkpoint(&self) -> Checkpoint {
        self.checkpoint.read().clone()
    }

    /// Get DLQ reference
    pub fn dlq(&self) -> &DeadLetterQueue {
        &self.dlq
    }

    /// Retry messages from DLQ
    pub fn retry_dlq(&self) -> Result<usize> {
        let mut retried = 0;

        while let Some(failed) = self.dlq.pop() {
            if failed.retry_count >= self.max_retries {
                // Re-add to DLQ as permanently failed
                self.dlq.push(
                    failed.message,
                    &format!("Max retries exceeded: {}", failed.error),
                    failed.retry_count,
                );
                continue;
            }

            match (self.processor)(&failed.message) {
                Ok(()) => {
                    self.dlq.mark_retry_success();
                    self.checkpoint.write().mark_processed(&failed.message.id);
                    retried += 1;
                }
                Err(e) => {
                    self.dlq
                        .push(failed.message, &e.to_string(), failed.retry_count + 1);
                }
            }
        }

        Ok(retried)
    }
}

/// Result of batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchResult {
    /// Successfully processed messages
    pub processed: usize,
    /// Skipped (already processed) messages
    pub skipped: usize,
    /// Failed messages (sent to DLQ)
    pub failed: usize,
}

impl BatchResult {
    /// Total messages in batch
    pub fn total(&self) -> usize {
        self.processed + self.skipped + self.failed
    }
}

/// Watermark tracker for windowed operations
#[derive(Debug)]
pub struct WatermarkTracker {
    /// Current watermark per partition
    watermarks: RwLock<HashMap<u32, u64>>,
    /// Global watermark (minimum across partitions)
    global_watermark: AtomicU64,
    /// Allowed lateness in milliseconds
    allowed_lateness: u64,
}

impl WatermarkTracker {
    /// Create a new watermark tracker
    pub fn new(allowed_lateness_ms: u64) -> Self {
        Self {
            watermarks: RwLock::new(HashMap::new()),
            global_watermark: AtomicU64::new(0),
            allowed_lateness: allowed_lateness_ms,
        }
    }

    /// Update watermark for a partition
    pub fn update(&self, partition: u32, timestamp: u64) {
        let mut watermarks = self.watermarks.write();
        let current = watermarks.entry(partition).or_insert(0);
        if timestamp > *current {
            *current = timestamp;
        }

        // Update global watermark
        if let Some(min) = watermarks.values().min() {
            self.global_watermark.store(*min, Ordering::SeqCst);
        }
    }

    /// Get current global watermark
    pub fn watermark(&self) -> u64 {
        self.global_watermark.load(Ordering::SeqCst)
    }

    /// Get watermark for a specific partition
    pub fn partition_watermark(&self, partition: u32) -> Option<u64> {
        self.watermarks.read().get(&partition).copied()
    }

    /// Check if a timestamp is late
    pub fn is_late(&self, timestamp: u64) -> bool {
        let watermark = self.watermark();
        watermark > 0 && timestamp + self.allowed_lateness < watermark
    }

    /// Get all partition watermarks
    pub fn all_watermarks(&self) -> HashMap<u32, u64> {
        self.watermarks.read().clone()
    }
}
