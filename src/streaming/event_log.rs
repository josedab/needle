use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;

use super::core::{
    ChangeEvent, OperationType, ResumeToken, StreamError, StreamResult,
    COMPACTION_THRESHOLD,
};

// ============================================================================
// Replay Options
// ============================================================================

/// Options for replaying events from the event log
#[derive(Debug, Clone, Default)]
pub struct ReplayOptions {
    /// Start position (inclusive)
    pub from_position: Option<u64>,
    /// End position (exclusive)
    pub to_position: Option<u64>,
    /// Filter by collection
    pub collection: Option<String>,
    /// Filter by operation types
    pub operations: Option<Vec<OperationType>>,
    /// Maximum number of events to return
    pub limit: Option<usize>,
    /// Skip first N events after filtering
    pub offset: Option<usize>,
}

impl ReplayOptions {
    /// Create new replay options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set start position
    pub fn from(mut self, position: u64) -> Self {
        self.from_position = Some(position);
        self
    }

    /// Set end position
    pub fn to(mut self, position: u64) -> Self {
        self.to_position = Some(position);
        self
    }

    /// Filter by collection
    pub fn collection(mut self, collection: &str) -> Self {
        self.collection = Some(collection.to_string());
        self
    }

    /// Filter by operations
    pub fn operations(mut self, ops: &[OperationType]) -> Self {
        self.operations = Some(ops.to_vec());
        self
    }

    /// Set limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

// ============================================================================
// Event Log (Event Sourcing)
// ============================================================================

/// Append-only event log for event sourcing
///
/// Provides persistent storage of all change events with support for:
/// - Appending new events
/// - Replaying events from any position
/// - Compacting old events
/// - Snapshots and restoration
pub struct EventLog {
    /// Events stored in memory (would be persisted in production)
    events: Arc<RwLock<Vec<ChangeEvent>>>,
    /// Current position counter
    position: AtomicU64,
    /// Whether compaction is in progress
    compacting: Arc<AtomicBool>,
    /// Compaction threshold
    compaction_threshold: usize,
    /// Last compacted position
    last_compacted_position: AtomicU64,
}

impl EventLog {
    /// Create a new event log
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            position: AtomicU64::new(0),
            compacting: Arc::new(AtomicBool::new(false)),
            compaction_threshold: COMPACTION_THRESHOLD,
            last_compacted_position: AtomicU64::new(0),
        }
    }

    /// Create with custom compaction threshold
    pub fn with_compaction_threshold(threshold: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            position: AtomicU64::new(0),
            compacting: Arc::new(AtomicBool::new(false)),
            compaction_threshold: threshold,
            last_compacted_position: AtomicU64::new(0),
        }
    }

    /// Append an event to the log
    pub async fn append(&self, mut event: ChangeEvent) -> StreamResult<u64> {
        if self.compacting.load(Ordering::Relaxed) {
            return Err(StreamError::CompactionInProgress);
        }

        let position = self.position.fetch_add(1, Ordering::SeqCst);
        event.id = position;
        event.resume_token = ResumeToken::new(position, event.timestamp);

        let mut events = self.events.write().await;
        events.push(event);

        Ok(position)
    }

    /// Append multiple events atomically
    pub async fn append_batch(&self, mut events_batch: Vec<ChangeEvent>) -> StreamResult<Vec<u64>> {
        if self.compacting.load(Ordering::Relaxed) {
            return Err(StreamError::CompactionInProgress);
        }

        let mut positions = Vec::with_capacity(events_batch.len());
        let mut events = self.events.write().await;

        for event in events_batch.iter_mut() {
            let position = self.position.fetch_add(1, Ordering::SeqCst);
            event.id = position;
            event.resume_token = ResumeToken::new(position, event.timestamp);
            events.push(event.clone());
            positions.push(position);
        }

        Ok(positions)
    }

    /// Get an event by position
    pub async fn get(&self, position: u64) -> Option<ChangeEvent> {
        let events = self.events.read().await;
        let last_compacted = self.last_compacted_position.load(Ordering::Relaxed);

        if position < last_compacted {
            return None; // Event was compacted
        }

        let index = (position - last_compacted) as usize;
        events.get(index).cloned()
    }

    /// Replay events based on options
    pub async fn replay(&self, options: ReplayOptions) -> StreamResult<Vec<ChangeEvent>> {
        let events = self.events.read().await;
        let last_compacted = self.last_compacted_position.load(Ordering::Relaxed);

        let from = options.from_position.unwrap_or(last_compacted);
        let to = options.to_position.unwrap_or(u64::MAX);

        if from < last_compacted {
            return Err(StreamError::PositionNotFound(from));
        }

        let start_index = (from.saturating_sub(last_compacted)) as usize;

        let mut result: Vec<ChangeEvent> = events
            .iter()
            .skip(start_index)
            .filter(|e| e.id >= from && e.id < to)
            .filter(|e| {
                if let Some(ref collection) = options.collection {
                    e.collection == *collection
                } else {
                    true
                }
            })
            .filter(|e| {
                if let Some(ref ops) = options.operations {
                    ops.contains(&e.operation)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        // Apply offset
        if let Some(offset) = options.offset {
            result = result.into_iter().skip(offset).collect();
        }

        // Apply limit
        if let Some(limit) = options.limit {
            result.truncate(limit);
        }

        Ok(result)
    }

    /// Get the current position
    pub fn current_position(&self) -> u64 {
        self.position.load(Ordering::Relaxed)
    }

    /// Get the number of events in the log
    pub async fn len(&self) -> usize {
        self.events.read().await.len()
    }

    /// Check if the log is empty
    pub async fn is_empty(&self) -> bool {
        self.events.read().await.is_empty()
    }

    /// Check if compaction is needed
    pub async fn needs_compaction(&self) -> bool {
        self.len().await >= self.compaction_threshold
    }

    /// Check if compaction is in progress
    pub fn is_compacting(&self) -> bool {
        self.compacting.load(Ordering::Relaxed)
    }

    /// Get the last compacted position
    pub fn last_compacted_position(&self) -> u64 {
        self.last_compacted_position.load(Ordering::Relaxed)
    }

    /// Compact the log by removing old events
    ///
    /// This keeps only events at or after the specified position
    pub async fn compact(&self, keep_from: u64) -> StreamResult<usize> {
        if self
            .compacting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return Err(StreamError::CompactionInProgress);
        }

        let removed = {
            let mut events = self.events.write().await;
            let last_compacted = self.last_compacted_position.load(Ordering::Relaxed);

            if keep_from <= last_compacted {
                self.compacting.store(false, Ordering::Relaxed);
                return Ok(0);
            }

            let remove_index = (keep_from - last_compacted) as usize;
            let removed_count = remove_index.min(events.len());

            events.drain(0..removed_count);
            self.last_compacted_position
                .store(keep_from, Ordering::Relaxed);

            removed_count
        };

        self.compacting.store(false, Ordering::Relaxed);
        Ok(removed)
    }

    /// Create a snapshot of the current state
    pub async fn snapshot(&self) -> EventLogSnapshot {
        let events = self.events.read().await;
        EventLogSnapshot {
            events: events.clone(),
            position: self.position.load(Ordering::Relaxed),
            last_compacted: self.last_compacted_position.load(Ordering::Relaxed),
        }
    }

    /// Restore from a snapshot
    pub async fn restore(&self, snapshot: EventLogSnapshot) -> StreamResult<()> {
        if self.compacting.load(Ordering::Relaxed) {
            return Err(StreamError::CompactionInProgress);
        }

        let mut events = self.events.write().await;
        *events = snapshot.events;
        self.position.store(snapshot.position, Ordering::Relaxed);
        self.last_compacted_position
            .store(snapshot.last_compacted, Ordering::Relaxed);

        Ok(())
    }

    /// Get events in a range
    pub async fn range(&self, from: u64, to: u64) -> StreamResult<Vec<ChangeEvent>> {
        self.replay(ReplayOptions::new().from(from).to(to)).await
    }

    /// Count events matching criteria
    pub async fn count(&self, options: ReplayOptions) -> StreamResult<usize> {
        Ok(self.replay(options).await?.len())
    }
}

impl Default for EventLog {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Event Log Snapshot
// ============================================================================

/// Snapshot of an event log for backup/restore
#[derive(Debug, Clone)]
pub struct EventLogSnapshot {
    /// Events in the snapshot
    pub events: Vec<ChangeEvent>,
    /// Position at snapshot time
    pub position: u64,
    /// Last compacted position
    pub last_compacted: u64,
}

impl EventLogSnapshot {
    /// Get the number of events in the snapshot
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the snapshot is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}
