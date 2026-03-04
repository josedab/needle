use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::RwLock;

use super::core::{
    ChangeEvent, OperationType, ResumeToken, StreamError, StreamResult, COMPACTION_THRESHOLD,
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::core::OperationType;

    fn make_event(collection: &str, op: OperationType, pos: u64) -> ChangeEvent {
        match op {
            OperationType::Insert => {
                ChangeEvent::insert(collection, &format!("key_{pos}"), vec![1, 2, 3], pos)
            }
            OperationType::Delete => {
                ChangeEvent::delete(collection, &format!("key_{pos}"), pos)
            }
            _ => ChangeEvent::insert(collection, &format!("key_{pos}"), vec![], pos),
        }
    }

    // ====================================================================
    // Append + replay round-trip
    // ====================================================================

    #[tokio::test]
    async fn test_append_and_replay() {
        let log = EventLog::new();

        let pos0 = log.append(make_event("col", OperationType::Insert, 0)).await.unwrap();
        let pos1 = log.append(make_event("col", OperationType::Insert, 1)).await.unwrap();

        assert_eq!(pos0, 0);
        assert_eq!(pos1, 1);

        let events = log.replay(ReplayOptions::new()).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, 0);
        assert_eq!(events[1].id, 1);
    }

    #[tokio::test]
    async fn test_current_position() {
        let log = EventLog::new();
        assert_eq!(log.current_position(), 0);

        log.append(make_event("col", OperationType::Insert, 0)).await.unwrap();
        assert_eq!(log.current_position(), 1);
    }

    #[tokio::test]
    async fn test_len_and_is_empty() {
        let log = EventLog::new();
        assert!(log.is_empty().await);
        assert_eq!(log.len().await, 0);

        log.append(make_event("col", OperationType::Insert, 0)).await.unwrap();
        assert!(!log.is_empty().await);
        assert_eq!(log.len().await, 1);
    }

    // ====================================================================
    // Position-based range queries
    // ====================================================================

    #[tokio::test]
    async fn test_range_query() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        let events = log.range(1, 4).await.unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].id, 1);
        assert_eq!(events[2].id, 3);
    }

    #[tokio::test]
    async fn test_get_by_position() {
        let log = EventLog::new();
        log.append(make_event("col", OperationType::Insert, 0)).await.unwrap();
        log.append(make_event("col", OperationType::Delete, 1)).await.unwrap();

        let event = log.get(0).await;
        assert!(event.is_some());
        assert_eq!(event.unwrap().collection, "col");

        let event = log.get(1).await;
        assert!(event.is_some());

        // Non-existent position
        let event = log.get(99).await;
        assert!(event.is_none());
    }

    // ====================================================================
    // ReplayOptions filtering
    // ====================================================================

    #[tokio::test]
    async fn test_replay_filter_by_collection() {
        let log = EventLog::new();
        log.append(make_event("users", OperationType::Insert, 0)).await.unwrap();
        log.append(make_event("orders", OperationType::Insert, 1)).await.unwrap();
        log.append(make_event("users", OperationType::Delete, 2)).await.unwrap();

        let events = log
            .replay(ReplayOptions::new().collection("users"))
            .await
            .unwrap();
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.collection == "users"));
    }

    #[tokio::test]
    async fn test_replay_filter_by_operation_type() {
        let log = EventLog::new();
        log.append(make_event("col", OperationType::Insert, 0)).await.unwrap();
        log.append(make_event("col", OperationType::Delete, 1)).await.unwrap();
        log.append(make_event("col", OperationType::Insert, 2)).await.unwrap();

        let events = log
            .replay(ReplayOptions::new().operations(&[OperationType::Delete]))
            .await
            .unwrap();
        assert_eq!(events.len(), 1);
    }

    #[tokio::test]
    async fn test_replay_with_limit() {
        let log = EventLog::new();
        for i in 0..10 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        let events = log.replay(ReplayOptions::new().limit(3)).await.unwrap();
        assert_eq!(events.len(), 3);
    }

    #[tokio::test]
    async fn test_replay_with_offset() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        let events = log.replay(ReplayOptions::new().offset(3)).await.unwrap();
        assert_eq!(events.len(), 2);
    }

    #[tokio::test]
    async fn test_replay_from_to() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        let events = log.replay(ReplayOptions::new().from(2).to(4)).await.unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, 2);
        assert_eq!(events[1].id, 3);
    }

    // ====================================================================
    // Batch append
    // ====================================================================

    #[tokio::test]
    async fn test_append_batch() {
        let log = EventLog::new();
        let events = vec![
            make_event("col", OperationType::Insert, 0),
            make_event("col", OperationType::Insert, 1),
            make_event("col", OperationType::Insert, 2),
        ];

        let positions = log.append_batch(events).await.unwrap();
        assert_eq!(positions.len(), 3);
        assert_eq!(positions, vec![0, 1, 2]);
        assert_eq!(log.len().await, 3);
    }

    // ====================================================================
    // Compaction
    // ====================================================================

    #[tokio::test]
    async fn test_compact() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        let removed = log.compact(3).await.unwrap();
        assert_eq!(removed, 3);
        assert_eq!(log.last_compacted_position(), 3);
        assert_eq!(log.len().await, 2);
    }

    #[tokio::test]
    async fn test_compact_below_compacted_noop() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        log.compact(3).await.unwrap();
        let removed = log.compact(2).await.unwrap();
        assert_eq!(removed, 0);
    }

    #[tokio::test]
    async fn test_get_after_compaction_returns_none() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        log.compact(3).await.unwrap();
        assert!(log.get(0).await.is_none());
        assert!(log.get(2).await.is_none());
        assert!(log.get(3).await.is_some());
    }

    #[tokio::test]
    async fn test_replay_after_compaction_position_not_found() {
        let log = EventLog::new();
        for i in 0..5 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        log.compact(3).await.unwrap();
        let result = log.replay(ReplayOptions::new().from(0)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_needs_compaction() {
        let log = EventLog::with_compaction_threshold(5);
        for i in 0..4 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }
        assert!(!log.needs_compaction().await);

        log.append(make_event("col", OperationType::Insert, 4)).await.unwrap();
        assert!(log.needs_compaction().await);
    }

    #[tokio::test]
    async fn test_is_compacting_initially_false() {
        let log = EventLog::new();
        assert!(!log.is_compacting());
    }

    // ====================================================================
    // Snapshot / restore
    // ====================================================================

    #[tokio::test]
    async fn test_snapshot_restore() {
        let log = EventLog::new();
        for i in 0..3 {
            log.append(make_event("col", OperationType::Insert, i)).await.unwrap();
        }

        let snapshot = log.snapshot().await;
        assert_eq!(snapshot.len(), 3);
        assert!(!snapshot.is_empty());

        // Create a new log and restore
        let log2 = EventLog::new();
        log2.restore(snapshot).await.unwrap();
        assert_eq!(log2.len().await, 3);
        assert_eq!(log2.current_position(), 3);
    }

    // ====================================================================
    // Count
    // ====================================================================

    #[tokio::test]
    async fn test_count() {
        let log = EventLog::new();
        log.append(make_event("a", OperationType::Insert, 0)).await.unwrap();
        log.append(make_event("b", OperationType::Insert, 1)).await.unwrap();
        log.append(make_event("a", OperationType::Delete, 2)).await.unwrap();

        let count = log.count(ReplayOptions::new().collection("a")).await.unwrap();
        assert_eq!(count, 2);
    }

    // ====================================================================
    // Compaction barrier (append fails during compaction)
    // ====================================================================

    #[tokio::test]
    async fn test_default_impl() {
        let log = EventLog::default();
        assert!(log.is_empty().await);
    }
}
