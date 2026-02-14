//! Streaming - Change Streams and Real-time Updates
//!
//! Real-time change notification and event streaming for Needle database.
//!
//! This module provides comprehensive support for:
//! - Change streams with async iteration and filtering
//! - Publish/subscribe pattern for real-time updates
//! - Event sourcing with append-only logs
//! - Change replay and compaction
//! - Backpressure handling and buffering
//!
//! # Examples
//!
//! ## Basic Change Stream
//!
//! ```rust,ignore
//! use needle::streaming::{ChangeStream, ChangeEventFilter, OperationType, StreamManager};
//!
//! // Create a stream manager
//! let manager = StreamManager::new();
//!
//! // Create a change stream for a collection
//! let mut stream = manager.create_stream_for_collection("users").await;
//!
//! // Filter for insert and update operations only
//! let filter = ChangeEventFilter::operations(&[
//!     OperationType::Insert,
//!     OperationType::Update,
//! ]);
//! let mut stream = stream.with_filter(filter);
//!
//! // Iterate over changes asynchronously
//! while let Some(event) = stream.next().await {
//!     match event.operation {
//!         OperationType::Insert => println!("New document: {:?}", event.document_key),
//!         OperationType::Update => println!("Updated: {:?}", event.document_key),
//!         _ => {}
//!     }
//! }
//! ```
//!
//! ## Pub/Sub Pattern
//!
//! ```rust,ignore
//! use needle::streaming::PubSub;
//!
//! let pubsub = PubSub::new();
//!
//! // Subscribe to a collection
//! let mut subscriber = pubsub.subscribe("orders").await;
//!
//! // Receive updates
//! while let Some(change) = subscriber.recv().await {
//!     process_order_change(change);
//! }
//! ```
//!
//! ## Event Sourcing with Replay
//!
//! ```rust,ignore
//! use needle::streaming::{EventLog, ReplayOptions};
//!
//! let log = EventLog::new();
//!
//! // Append events
//! log.append(ChangeEvent::insert("users", "user_1", vec![1, 2, 3], 0)).await?;
//!
//! // Replay events from a specific position
//! let events = log.replay(ReplayOptions::new()
//!     .from(1000)
//!     .collection("users")
//! ).await?;
//!
//! // Compact old events
//! log.compact(5000).await?;
//! ```
//!
//! ## Resume from Token
//!
//! ```rust,ignore
//! use needle::streaming::{StreamManager, ResumeToken};
//!
//! let manager = StreamManager::new();
//!
//! // Save resume token before disconnecting
//! let token = stream.resume_token();
//! let token_str = token.as_str().to_string();
//!
//! // Later, resume from token
//! let resume_token = ResumeToken::parse(&token_str)?;
//! let mut stream = manager.create_stream_with_resume(&resume_token).await?;
//! ```

pub mod cdc;
pub mod core;
pub mod event_log;
pub mod pubsub;
pub mod stream_manager;

// Re-export all public types for backwards compatibility
pub use cdc::{
    CdcConfig, CdcConnector, CdcConnectorStats, CdcIngestionPipeline, CdcPipelineStats,
    CdcPosition, DebeziumParser, DebeziumSourceType, KafkaConnector, KafkaConnectorConfig,
    MongoCdcConfig, MongoCdcConnector, PostgresCdcConfig, PostgresCdcConnector, PulsarConnector,
    PulsarConnectorConfig, PulsarSubscriptionPosition,
};
pub use core::{
    ChangeEvent, ChangeEventFilter, OperationType, ResumeToken, StreamError, StreamResult,
};
pub(crate) use core::{COMPACTION_THRESHOLD, DEFAULT_BUFFER_SIZE, DEFAULT_CHANNEL_CAPACITY};
pub use event_log::{EventLog, EventLogSnapshot, ReplayOptions};
pub use pubsub::{PubSub, Subscriber};
pub use stream_manager::{ChangeStream, StreamManager, StreamManagerConfig, StreamStats};

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::mpsc;

    #[test]
    fn test_change_event_insert() {
        let event = ChangeEvent::insert("users", "user_1", vec![1, 2, 3], 0);

        assert_eq!(event.operation, OperationType::Insert);
        assert_eq!(event.collection, "users");
        assert_eq!(event.document_key, Some("user_1".to_string()));
        assert_eq!(event.full_document, Some(vec![1, 2, 3]));
        assert_eq!(event.id, 0);
    }

    #[test]
    fn test_change_event_update() {
        let mut updated_fields = HashMap::new();
        updated_fields.insert("name".to_string(), vec![4, 5, 6]);

        let event = ChangeEvent::update(
            "users",
            "user_1",
            Some(vec![1, 2, 3]),
            updated_fields.clone(),
            vec!["old_field".to_string()],
            1,
        );

        assert_eq!(event.operation, OperationType::Update);
        assert_eq!(event.updated_fields, Some(updated_fields));
        assert_eq!(event.removed_fields, Some(vec!["old_field".to_string()]));
    }

    #[test]
    fn test_change_event_delete() {
        let event = ChangeEvent::delete("users", "user_1", 2);

        assert_eq!(event.operation, OperationType::Delete);
        assert!(event.full_document.is_none());
    }

    #[test]
    fn test_change_event_with_metadata() {
        let event = ChangeEvent::insert("users", "user_1", vec![1], 0)
            .with_metadata("source", "api")
            .with_metadata("version", "1.0");

        assert!(event.metadata.is_some());
        let meta = event.metadata.unwrap();
        assert_eq!(meta.get("source"), Some(&"api".to_string()));
        assert_eq!(meta.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_resume_token() {
        let token = ResumeToken::new(100, 1234567890);

        assert_eq!(token.position, 100);
        assert_eq!(token.timestamp, 1234567890);
        assert_eq!(token.as_str(), "100:1234567890");

        let parsed = ResumeToken::parse("100:1234567890").unwrap();
        assert_eq!(parsed.position, 100);
        assert_eq!(parsed.timestamp, 1234567890);
    }

    #[test]
    fn test_resume_token_invalid() {
        assert!(ResumeToken::parse("invalid").is_err());
        assert!(ResumeToken::parse("100").is_err());
        assert!(ResumeToken::parse("100:abc").is_err());
        assert!(ResumeToken::parse("abc:123").is_err());
    }

    #[test]
    fn test_change_event_filter_collections() {
        let filter = ChangeEventFilter::collections(&["users", "orders"]);

        let users_event = ChangeEvent::insert("users", "1", vec![], 0);
        let orders_event = ChangeEvent::insert("orders", "1", vec![], 0);
        let products_event = ChangeEvent::insert("products", "1", vec![], 0);

        assert!(filter.matches(&users_event));
        assert!(filter.matches(&orders_event));
        assert!(!filter.matches(&products_event));
    }

    #[test]
    fn test_change_event_filter_operations() {
        let filter = ChangeEventFilter::operations(&[OperationType::Insert, OperationType::Update]);

        let insert_event = ChangeEvent::insert("users", "1", vec![], 0);
        let delete_event = ChangeEvent::delete("users", "1", 0);

        assert!(filter.matches(&insert_event));
        assert!(!filter.matches(&delete_event));
    }

    #[test]
    fn test_change_event_filter_combined() {
        let filter = ChangeEventFilter::new()
            .with_collections(&["users"])
            .with_operations(&[OperationType::Insert]);

        let matching_event = ChangeEvent::insert("users", "1", vec![], 0);
        let wrong_collection = ChangeEvent::insert("orders", "1", vec![], 0);
        let wrong_operation = ChangeEvent::delete("users", "1", 0);

        assert!(filter.matches(&matching_event));
        assert!(!filter.matches(&wrong_collection));
        assert!(!filter.matches(&wrong_operation));
    }

    #[test]
    fn test_change_event_filter_document_key_pattern() {
        let filter = ChangeEventFilter::new().with_document_key_pattern("user_");

        let matching = ChangeEvent::insert("users", "user_123", vec![], 0);
        let not_matching = ChangeEvent::insert("users", "admin_1", vec![], 0);

        assert!(filter.matches(&matching));
        assert!(!filter.matches(&not_matching));
    }

    #[test]
    fn test_operation_type_from_str() {
        assert_eq!("insert".parse::<OperationType>(), Ok(OperationType::Insert));
        assert_eq!("UPDATE".parse::<OperationType>(), Ok(OperationType::Update));
        assert_eq!("Delete".parse::<OperationType>(), Ok(OperationType::Delete));
        assert_eq!(
            "createIndex".parse::<OperationType>(),
            Ok(OperationType::CreateIndex)
        );
        assert_eq!(
            "create_index".parse::<OperationType>(),
            Ok(OperationType::CreateIndex)
        );
        assert_eq!("invalid".parse::<OperationType>(), Err(()));
    }

    #[tokio::test]
    async fn test_event_log_append_and_get() {
        let log = EventLog::new();

        let event1 = ChangeEvent::insert("users", "1", vec![1], 0);
        let event2 = ChangeEvent::insert("users", "2", vec![2], 0);

        let pos1 = log.append(event1).await.unwrap();
        let pos2 = log.append(event2).await.unwrap();

        assert_eq!(pos1, 0);
        assert_eq!(pos2, 1);

        let retrieved1 = log.get(0).await.unwrap();
        assert_eq!(retrieved1.document_key, Some("1".to_string()));

        let retrieved2 = log.get(1).await.unwrap();
        assert_eq!(retrieved2.document_key, Some("2".to_string()));
    }

    #[tokio::test]
    async fn test_event_log_append_batch() {
        let log = EventLog::new();

        let events: Vec<ChangeEvent> = (0..5)
            .map(|i| ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0))
            .collect();

        let positions = log.append_batch(events).await.unwrap();

        assert_eq!(positions.len(), 5);
        assert_eq!(positions, vec![0, 1, 2, 3, 4]);
        assert_eq!(log.len().await, 5);
    }

    #[tokio::test]
    async fn test_event_log_replay() {
        let log = EventLog::new();

        for i in 0..10 {
            let collection = if i % 2 == 0 { "users" } else { "orders" };
            let event = ChangeEvent::insert(collection, &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        // Replay all
        let all = log.replay(ReplayOptions::new()).await.unwrap();
        assert_eq!(all.len(), 10);

        // Replay from position
        let from_5 = log.replay(ReplayOptions::new().from(5)).await.unwrap();
        assert_eq!(from_5.len(), 5);

        // Replay with collection filter
        let users_only = log
            .replay(ReplayOptions::new().collection("users"))
            .await
            .unwrap();
        assert_eq!(users_only.len(), 5);

        // Replay with limit
        let limited = log.replay(ReplayOptions::new().limit(3)).await.unwrap();
        assert_eq!(limited.len(), 3);

        // Replay with offset
        let with_offset = log
            .replay(ReplayOptions::new().offset(2).limit(3))
            .await
            .unwrap();
        assert_eq!(with_offset.len(), 3);
        assert_eq!(with_offset[0].id, 2);
    }

    #[tokio::test]
    async fn test_event_log_compaction() {
        let log = EventLog::with_compaction_threshold(5);

        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        assert_eq!(log.len().await, 10);
        assert!(log.needs_compaction().await);

        // Compact, keeping from position 5
        let removed = log.compact(5).await.unwrap();
        assert_eq!(removed, 5);
        assert_eq!(log.len().await, 5);

        // Old events should be gone
        assert!(log.get(0).await.is_none());
        assert!(log.get(4).await.is_none());

        // New events should still be accessible
        let event5 = log.get(5).await.unwrap();
        assert_eq!(event5.document_key, Some("5".to_string()));
    }

    #[tokio::test]
    async fn test_event_log_snapshot_and_restore() {
        let log = EventLog::new();

        for i in 0..5 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        let snapshot = log.snapshot().await;
        assert_eq!(snapshot.events.len(), 5);
        assert_eq!(snapshot.position, 5);

        // Create new log and restore
        let new_log = EventLog::new();
        new_log.restore(snapshot).await.unwrap();

        assert_eq!(new_log.len().await, 5);
        assert_eq!(new_log.current_position(), 5);

        let event = new_log.get(2).await.unwrap();
        assert_eq!(event.document_key, Some("2".to_string()));
    }

    #[tokio::test]
    async fn test_pubsub_subscribe_and_publish() {
        let pubsub = PubSub::new();

        let mut subscriber = pubsub.subscribe("users").await;

        let event = ChangeEvent::insert("users", "1", vec![1, 2, 3], 0);
        pubsub.publish(event.clone()).await.unwrap();

        let received = subscriber.recv().await.unwrap();
        assert_eq!(received.document_key, Some("1".to_string()));
    }

    #[tokio::test]
    async fn test_pubsub_multiple_subscribers() {
        let pubsub = PubSub::new();

        let mut sub1 = pubsub.subscribe("users").await;
        let mut sub2 = pubsub.subscribe("users").await;

        let event = ChangeEvent::insert("users", "1", vec![1], 0);
        pubsub.publish(event).await.unwrap();

        let recv1 = sub1.recv().await.unwrap();
        let recv2 = sub2.recv().await.unwrap();

        assert_eq!(recv1.document_key, recv2.document_key);
    }

    #[tokio::test]
    async fn test_pubsub_filtered_subscription() {
        let pubsub = PubSub::new();

        let filter = ChangeEventFilter::operations(&[OperationType::Insert]);
        let mut subscriber = pubsub.subscribe_with_filter("users", filter).await;

        // Publish insert - should be received
        let insert_event = ChangeEvent::insert("users", "1", vec![1], 0);
        pubsub.publish(insert_event).await.unwrap();

        // Publish delete - should not be received (filtered out)
        let delete_event = ChangeEvent::delete("users", "1", 1);
        pubsub.publish(delete_event).await.unwrap();

        let received = subscriber.try_recv().unwrap();
        assert_eq!(received.operation, OperationType::Insert);

        // Delete should not have been received
        assert!(subscriber.try_recv().is_none());
    }

    #[tokio::test]
    async fn test_pubsub_global_subscription() {
        let pubsub = PubSub::new();

        let mut global_sub = pubsub.subscribe_all().await;

        let users_event = ChangeEvent::insert("users", "1", vec![1], 0);
        let orders_event = ChangeEvent::insert("orders", "1", vec![2], 1);

        pubsub.publish(users_event).await.unwrap();
        pubsub.publish(orders_event).await.unwrap();

        let recv1 = global_sub.recv().await.unwrap();
        let recv2 = global_sub.recv().await.unwrap();

        assert_eq!(recv1.collection, "users");
        assert_eq!(recv2.collection, "orders");
    }

    #[tokio::test]
    async fn test_pubsub_unsubscribe() {
        let pubsub = PubSub::new();

        let subscriber = pubsub.subscribe("users").await;
        assert_eq!(pubsub.subscriber_count("users").await, 1);

        subscriber.unsubscribe();
        pubsub.cleanup_inactive().await;

        assert_eq!(pubsub.subscriber_count("users").await, 0);
    }

    #[tokio::test]
    async fn test_change_stream() {
        let (tx, rx) = mpsc::channel(10);
        let mut stream = ChangeStream::new(rx);

        // Send events
        let event1 = ChangeEvent::insert("users", "1", vec![1], 0);
        let event2 = ChangeEvent::insert("users", "2", vec![2], 1);

        tx.send(event1).await.unwrap();
        tx.send(event2).await.unwrap();

        // Receive events
        let recv1 = stream.next().await.unwrap();
        assert_eq!(recv1.document_key, Some("1".to_string()));
        assert_eq!(stream.position(), 0);

        let recv2 = stream.next().await.unwrap();
        assert_eq!(recv2.document_key, Some("2".to_string()));
        assert_eq!(stream.position(), 1);
    }

    #[tokio::test]
    async fn test_change_stream_with_filter() {
        let (tx, rx) = mpsc::channel(10);
        let filter = ChangeEventFilter::collections(&["users"]);
        let mut stream = ChangeStream::new(rx).with_filter(filter);

        // Send events
        let users_event = ChangeEvent::insert("users", "1", vec![1], 0);
        let orders_event = ChangeEvent::insert("orders", "1", vec![2], 1);
        let users_event2 = ChangeEvent::insert("users", "2", vec![3], 2);

        tx.send(users_event).await.unwrap();
        tx.send(orders_event).await.unwrap();
        tx.send(users_event2).await.unwrap();

        // Should only receive users events
        let recv1 = stream.next().await.unwrap();
        assert_eq!(recv1.collection, "users");
        assert_eq!(recv1.document_key, Some("1".to_string()));

        let recv2 = stream.next().await.unwrap();
        assert_eq!(recv2.collection, "users");
        assert_eq!(recv2.document_key, Some("2".to_string()));
    }

    #[tokio::test]
    async fn test_stream_manager() {
        let manager = StreamManager::new();

        // Create subscriber
        let mut subscriber = manager.subscribe("users").await;

        // Record change
        let event = ChangeEvent::insert("users", "1", vec![1, 2, 3], 0);
        let position = manager.record_change(event).await.unwrap();

        assert_eq!(position, 0);

        // Subscriber should receive the event
        let received = subscriber.recv().await.unwrap();
        assert_eq!(received.document_key, Some("1".to_string()));

        // Event should be in the log
        let from_log = manager.event_log().get(0).await.unwrap();
        assert_eq!(from_log.document_key, Some("1".to_string()));
    }

    #[tokio::test]
    async fn test_stream_manager_record_multiple() {
        let manager = StreamManager::new();

        let events: Vec<ChangeEvent> = (0..5)
            .map(|i| ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0))
            .collect();

        let positions = manager.record_changes(events).await.unwrap();

        assert_eq!(positions.len(), 5);
        assert_eq!(manager.event_log().len().await, 5);
    }

    #[tokio::test]
    async fn test_stream_manager_resume() {
        let manager = StreamManager::new();

        // Record some events
        for i in 0..5 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            manager.record_change(event).await.unwrap();
        }

        // Create stream with resume from position 2
        let resume_token = ResumeToken::new(2, 0);
        let mut stream = manager
            .create_stream_with_resume(&resume_token)
            .await
            .unwrap();

        // Should receive events starting from position 3
        let event = stream.next().await.unwrap();
        assert_eq!(event.id, 3);

        let event = stream.next().await.unwrap();
        assert_eq!(event.id, 4);
    }

    #[tokio::test]
    async fn test_stream_manager_cleanup() {
        let manager = StreamManager::new();

        // Create subscriber and unsubscribe
        let subscriber = manager.subscribe("users").await;
        subscriber.unsubscribe();

        // Cleanup
        manager.cleanup().await;

        // Subscriber count should be 0
        assert_eq!(manager.pubsub().subscriber_count("users").await, 0);
    }

    #[tokio::test]
    async fn test_stream_manager_stats() {
        let manager = StreamManager::new();

        // Record some events
        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            manager.record_change(event).await.unwrap();
        }

        // Subscribe
        let _sub = manager.subscribe("users").await;

        let stats = manager.stats().await;
        assert_eq!(stats.event_log_size, 10);
        assert_eq!(stats.current_position, 10);
        assert_eq!(stats.total_subscribers, 1);
    }

    #[tokio::test]
    async fn test_backpressure_handling() {
        let pubsub = PubSub::with_config(10, 2); // Small channel capacity

        let mut subscriber = pubsub.subscribe("users").await;

        // Fill up the channel and buffer
        for i in 0..15 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], i as u64);
            let result = pubsub.publish(event).await;
            if i >= 12 {
                // Should start failing due to buffer overflow
                assert!(result.is_err());
            }
        }

        // Drain some events
        for _ in 0..2 {
            subscriber.recv().await;
        }

        // Flush buffer should work now
        let flushed = pubsub.flush_buffer().await.unwrap();
        assert!(flushed > 0);
    }

    #[tokio::test]
    async fn test_operation_type_display() {
        assert_eq!(format!("{}", OperationType::Insert), "insert");
        assert_eq!(format!("{}", OperationType::Update), "update");
        assert_eq!(format!("{}", OperationType::Delete), "delete");
        assert_eq!(format!("{}", OperationType::Drop), "drop");
        assert_eq!(format!("{}", OperationType::Rename), "rename");
        assert_eq!(format!("{}", OperationType::CreateIndex), "createIndex");
        assert_eq!(format!("{}", OperationType::DropIndex), "dropIndex");
        assert_eq!(format!("{}", OperationType::Batch), "batch");
    }

    #[test]
    fn test_stream_error_display() {
        assert_eq!(
            format!("{}", StreamError::StreamClosed),
            "Stream has been closed"
        );
        assert_eq!(
            format!("{}", StreamError::BufferOverflow),
            "Buffer overflow - backpressure limit reached"
        );
        assert_eq!(
            format!("{}", StreamError::InvalidResumeToken("abc".to_string())),
            "Invalid resume token: abc"
        );
        assert_eq!(format!("{}", StreamError::Timeout), "Operation timed out");
    }

    #[test]
    fn test_with_before_change() {
        let event = ChangeEvent::delete("users", "1", 0).with_before_change(vec![1, 2, 3]);

        assert_eq!(event.full_document_before_change, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_drop_collection_event() {
        let event = ChangeEvent::drop_collection("users", 0);

        assert_eq!(event.operation, OperationType::Drop);
        assert_eq!(event.collection, "users");
        assert!(event.document_key.is_none());
    }

    #[tokio::test]
    async fn test_concurrent_compaction() {
        let log = Arc::new(EventLog::new());

        // Append events
        for i in 0..100 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        // Try to compact twice simultaneously
        let log1 = Arc::clone(&log);
        let log2 = Arc::clone(&log);

        let handle1 = tokio::spawn(async move { log1.compact(50).await });
        let handle2 = tokio::spawn(async move { log2.compact(50).await });

        let result1 = handle1.await.unwrap();
        let result2 = handle2.await.unwrap();

        // One should succeed, one should fail or return 0
        let success_count = [&result1, &result2]
            .iter()
            .filter(|r| matches!(r, Ok(n) if *n > 0))
            .count();

        assert!(
            success_count <= 1,
            "Only one compaction should remove events"
        );
    }

    #[tokio::test]
    async fn test_replay_after_compaction() {
        let log = EventLog::new();

        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        // Compact first 5 events
        log.compact(5).await.unwrap();

        // Try to replay from compacted position - should fail
        let result = log.replay(ReplayOptions::new().from(0)).await;
        assert!(matches!(result, Err(StreamError::PositionNotFound(0))));

        // Replay from valid position should work
        let events = log.replay(ReplayOptions::new().from(5)).await.unwrap();
        assert_eq!(events.len(), 5);
        assert_eq!(events[0].id, 5);
    }

    #[tokio::test]
    async fn test_event_log_range() {
        let log = EventLog::new();

        for i in 0..10 {
            let event = ChangeEvent::insert("users", &format!("{}", i), vec![i as u8], 0);
            log.append(event).await.unwrap();
        }

        let range = log.range(3, 7).await.unwrap();
        assert_eq!(range.len(), 4);
        assert_eq!(range[0].id, 3);
        assert_eq!(range[3].id, 6);
    }

    #[tokio::test]
    async fn test_replay_with_operations_filter() {
        let log = EventLog::new();

        log.append(ChangeEvent::insert("users", "1", vec![], 0))
            .await
            .unwrap();
        log.append(ChangeEvent::delete("users", "1", 0))
            .await
            .unwrap();
        log.append(ChangeEvent::insert("users", "2", vec![], 0))
            .await
            .unwrap();

        let inserts = log
            .replay(ReplayOptions::new().operations(&[OperationType::Insert]))
            .await
            .unwrap();

        assert_eq!(inserts.len(), 2);
        assert!(inserts.iter().all(|e| e.operation == OperationType::Insert));
    }

    #[test]
    fn test_stream_manager_config_default() {
        let config = StreamManagerConfig::default();

        assert_eq!(config.max_buffer_size, DEFAULT_BUFFER_SIZE);
        assert_eq!(config.channel_capacity, DEFAULT_CHANNEL_CAPACITY);
        assert_eq!(config.compaction_threshold, COMPACTION_THRESHOLD);
        assert_eq!(config.cleanup_interval_secs, 60);
    }

    #[tokio::test]
    async fn test_stream_manager_with_config() {
        let config = StreamManagerConfig {
            max_buffer_size: 512,
            channel_capacity: 128,
            compaction_threshold: 5000,
            cleanup_interval_secs: 30,
        };

        let manager = StreamManager::with_config(config.clone());

        // Verify the manager was created with custom config
        assert_eq!(manager.config.max_buffer_size, 512);
        assert_eq!(manager.config.channel_capacity, 128);
    }

    #[tokio::test]
    async fn test_subscriber_is_active() {
        let pubsub = PubSub::new();
        let subscriber = pubsub.subscribe("test").await;

        assert!(subscriber.is_active());

        subscriber.unsubscribe();

        assert!(!subscriber.is_active());
    }

    #[tokio::test]
    async fn test_change_stream_close() {
        let (tx, rx) = mpsc::channel(10);
        let stream = ChangeStream::new(rx);

        assert!(!stream.is_closed());

        stream.close();

        assert!(stream.is_closed());
        drop(tx);
    }

    #[tokio::test]
    async fn test_event_log_count() {
        let log = EventLog::new();

        for i in 0..10 {
            let collection = if i % 2 == 0 { "users" } else { "orders" };
            let event = ChangeEvent::insert(collection, &format!("{}", i), vec![], 0);
            log.append(event).await.unwrap();
        }

        let total_count = log.count(ReplayOptions::new()).await.unwrap();
        assert_eq!(total_count, 10);

        let users_count = log
            .count(ReplayOptions::new().collection("users"))
            .await
            .unwrap();
        assert_eq!(users_count, 5);
    }

    #[test]
    fn test_event_log_snapshot_is_empty() {
        let snapshot = EventLogSnapshot {
            events: vec![],
            position: 0,
            last_compacted: 0,
        };

        assert!(snapshot.is_empty());
        assert_eq!(snapshot.len(), 0);
    }

    // ========================================================================
    // CDC Tests
    // ========================================================================

    #[test]
    fn test_cdc_position_new() {
        let pos = CdcPosition::new("12345", "my-topic");

        assert_eq!(pos.position, "12345");
        assert_eq!(pos.source, "my-topic");
        assert!(pos.partition.is_none());
    }

    #[test]
    fn test_cdc_position_with_partition() {
        let pos = CdcPosition::new("12345", "my-topic").with_partition(2);

        assert_eq!(pos.partition, Some(2));
    }

    #[test]
    fn test_cdc_position_serialize_parse() {
        let pos = CdcPosition::new("12345", "my-topic").with_partition(2);
        let serialized = pos.serialize();

        let parsed = CdcPosition::parse(&serialized).unwrap();
        assert_eq!(parsed.source, pos.source);
        assert_eq!(parsed.position, pos.position);
        assert_eq!(parsed.partition, pos.partition);
    }

    #[test]
    fn test_cdc_position_serialize_without_partition() {
        let pos = CdcPosition::new("12345", "my-topic");
        let serialized = pos.serialize();

        let parsed = CdcPosition::parse(&serialized).unwrap();
        assert_eq!(parsed.source, "my-topic");
        assert_eq!(parsed.position, "12345");
        assert!(parsed.partition.is_none());
    }

    #[test]
    fn test_cdc_config_default() {
        let config = CdcConfig::default();

        assert_eq!(config.batch_size, 100);
        assert_eq!(config.fetch_timeout_ms, 5000);
        assert_eq!(config.max_retries, 3);
        assert!(!config.exactly_once);
    }

    #[test]
    fn test_debezium_parser_insert() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{
            "op": "c",
            "ts_ms": 1234567890,
            "source": {
                "table": "users",
                "db": "mydb"
            },
            "after": {
                "id": 1,
                "name": "Alice"
            }
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert_eq!(event.operation, OperationType::Insert);
        assert_eq!(event.collection, "users");
        assert!(event.full_document.is_some());
    }

    #[test]
    fn test_debezium_parser_update() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{
            "op": "u",
            "ts_ms": 1234567890,
            "source": {
                "table": "users"
            },
            "before": {
                "id": 1,
                "name": "Alice"
            },
            "after": {
                "id": 1,
                "name": "Bob"
            }
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert_eq!(event.operation, OperationType::Update);
        assert!(event.full_document.is_some());
        assert!(event.full_document_before_change.is_some());
    }

    #[test]
    fn test_debezium_parser_delete() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{
            "op": "d",
            "ts_ms": 1234567890,
            "source": {
                "table": "users"
            },
            "before": {
                "id": 1,
                "name": "Alice"
            }
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert_eq!(event.operation, OperationType::Delete);
    }

    #[test]
    fn test_debezium_parser_with_mapping() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL)
            .with_mapping("users", "user_vectors");

        let json = r#"{
            "op": "c",
            "ts_ms": 1234567890,
            "source": {
                "table": "users"
            },
            "after": {"id": 1}
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert_eq!(event.collection, "user_vectors");
    }

    #[test]
    fn test_debezium_parser_mongodb_format() {
        let parser = DebeziumParser::new(DebeziumSourceType::MongoDB);

        let json = r#"{
            "op": "c",
            "ts_ms": 1234567890,
            "source": {
                "table": "documents"
            },
            "after": {
                "_id": "507f1f77bcf86cd799439011",
                "content": "test"
            }
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert_eq!(event.operation, OperationType::Insert);
        assert_eq!(event.collection, "documents");
    }

    #[test]
    fn test_debezium_parser_with_update_description() {
        let parser = DebeziumParser::new(DebeziumSourceType::MongoDB);

        let json = r#"{
            "op": "u",
            "ts_ms": 1234567890,
            "source": {
                "table": "users"
            },
            "after": {"id": 1, "name": "Bob"},
            "updateDescription": {
                "updatedFields": {"name": "Bob"},
                "removedFields": ["old_field"]
            }
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert!(event.updated_fields.is_some());
        assert!(event.removed_fields.is_some());
        assert_eq!(
            event.removed_fields.as_ref().unwrap(),
            &vec!["old_field".to_string()]
        );
    }

    #[test]
    fn test_debezium_parser_extract_metadata() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{
            "op": "c",
            "ts_ms": 1234567890,
            "source": {
                "table": "users",
                "db": "mydb",
                "schema": "public",
                "connector": "postgresql",
                "lsn": 12345678
            },
            "after": {"id": 1}
        }"#;

        let event = parser.parse_json(json).unwrap();

        assert!(event.metadata.is_some());
        let meta = event.metadata.unwrap();
        assert_eq!(meta.get("database"), Some(&"mydb".to_string()));
        assert_eq!(meta.get("schema"), Some(&"public".to_string()));
        assert_eq!(meta.get("connector"), Some(&"postgresql".to_string()));
        assert_eq!(meta.get("lsn"), Some(&"12345678".to_string()));
    }

    #[test]
    fn test_debezium_parser_invalid_op() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{"op": "x"}"#;
        let result = parser.parse_json(json);

        assert!(result.is_err());
    }

    #[test]
    fn test_debezium_parser_missing_op() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{"ts_ms": 123}"#;
        let result = parser.parse_json(json);

        assert!(result.is_err());
    }

    #[test]
    fn test_kafka_connector_config_default() {
        let config = KafkaConnectorConfig::default();

        assert_eq!(config.brokers, vec!["localhost:9092".to_string()]);
        assert_eq!(config.group_id, "needle-cdc");
        assert_eq!(config.security_protocol, "PLAINTEXT");
        assert_eq!(config.offset_reset, "earliest");
    }

    #[test]
    fn test_postgres_cdc_config_default() {
        let config = PostgresCdcConfig::default();

        assert_eq!(config.slot_name, "needle_slot");
        assert_eq!(config.publication_name, "needle_publication");
        assert!(config.tables.is_empty());
    }

    #[test]
    fn test_mongo_cdc_config_default() {
        let config = MongoCdcConfig::default();

        assert_eq!(config.database, "needle");
        assert_eq!(config.full_document, "updateLookup");
        assert!(config.collections.is_empty());
    }

    #[test]
    fn test_pulsar_connector_config_default() {
        let config = PulsarConnectorConfig::default();

        assert_eq!(config.service_url, "pulsar://localhost:6650");
        assert_eq!(config.topic, "persistent://public/default/needle-cdc");
        assert_eq!(config.subscription, "needle-cdc-subscription");
        assert_eq!(config.consumer_name, "needle-cdc-consumer");
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.receive_timeout_ms, 5000);
        assert!(!config.enable_dead_letter);
    }

    #[test]
    fn test_pulsar_connector_config_builder() {
        let config = PulsarConnectorConfig::new("pulsar://myhost:6650", "my-topic")
            .with_subscription("my-sub")
            .with_consumer_name("my-consumer")
            .with_batch_size(50)
            .with_initial_position(PulsarSubscriptionPosition::Earliest)
            .with_dead_letter("dlq-topic", 5);

        assert_eq!(config.service_url, "pulsar://myhost:6650");
        assert_eq!(config.topic, "my-topic");
        assert_eq!(config.subscription, "my-sub");
        assert_eq!(config.consumer_name, "my-consumer");
        assert_eq!(config.batch_size, 50);
        assert!(config.enable_dead_letter);
        assert_eq!(config.dead_letter_topic, Some("dlq-topic".to_string()));
        assert_eq!(config.max_redelivery_count, 5);
    }

    #[test]
    fn test_pulsar_subscription_position_default() {
        let position = PulsarSubscriptionPosition::default();
        assert!(matches!(position, PulsarSubscriptionPosition::Latest));
    }

    #[tokio::test]
    async fn test_cdc_ingestion_pipeline() {
        let manager = Arc::new(StreamManager::new());
        let pipeline = CdcIngestionPipeline::new(Arc::clone(&manager));

        let event = ChangeEvent::insert("test", "1", vec![1, 2, 3], 0);
        let position = pipeline.ingest(event).await.unwrap();

        assert!(position.is_some());

        let stats = pipeline.stats().await;
        assert_eq!(stats.events_ingested, 1);
    }

    #[tokio::test]
    async fn test_cdc_ingestion_pipeline_with_transformer() {
        let manager = Arc::new(StreamManager::new());
        let pipeline =
            CdcIngestionPipeline::new(Arc::clone(&manager)).with_transformer(|mut event| {
                // Add a prefix to collection name
                event.collection = format!("transformed_{}", event.collection);
                Some(event)
            });

        let event = ChangeEvent::insert("test", "1", vec![1, 2, 3], 0);
        pipeline.ingest(event).await.unwrap();

        let stats = pipeline.stats().await;
        assert_eq!(stats.events_transformed, 1);
    }

    #[tokio::test]
    async fn test_cdc_ingestion_pipeline_filter() {
        let manager = Arc::new(StreamManager::new());
        let pipeline = CdcIngestionPipeline::new(Arc::clone(&manager)).with_transformer(|event| {
            // Filter out delete operations
            if event.operation == OperationType::Delete {
                None
            } else {
                Some(event)
            }
        });

        let insert = ChangeEvent::insert("test", "1", vec![1], 0);
        let delete = ChangeEvent::delete("test", "1", 0);

        pipeline.ingest(insert).await.unwrap();
        pipeline.ingest(delete).await.unwrap();

        let stats = pipeline.stats().await;
        assert_eq!(stats.events_ingested, 1);
        assert_eq!(stats.events_filtered, 1);
    }

    #[tokio::test]
    async fn test_cdc_ingestion_pipeline_checkpoint() {
        let manager = Arc::new(StreamManager::new());
        let pipeline = CdcIngestionPipeline::new(Arc::clone(&manager)).with_checkpoint_interval(5);

        // Ingest some events
        for i in 0..10 {
            let event = ChangeEvent::insert("test", &i.to_string(), vec![i as u8], 0);
            pipeline.ingest(event).await.unwrap();
        }

        // Create a checkpoint
        let position = CdcPosition::new("offset_10", "test-topic");
        pipeline.checkpoint(position.clone()).await.unwrap();

        let last_checkpoint = pipeline.last_checkpoint().await;
        assert!(last_checkpoint.is_some());
        assert_eq!(last_checkpoint.unwrap().position, "offset_10");

        let stats = pipeline.stats().await;
        assert!(stats.last_checkpoint_time.is_some());
    }

    #[test]
    fn test_cdc_connector_stats_default() {
        let stats = CdcConnectorStats::default();

        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.messages_processed, 0);
        assert_eq!(stats.messages_failed, 0);
        assert!(stats.last_error.is_none());
    }

    #[test]
    fn test_debezium_source_types() {
        // Ensure all source types can be created
        let _ = DebeziumParser::new(DebeziumSourceType::PostgreSQL);
        let _ = DebeziumParser::new(DebeziumSourceType::MySQL);
        let _ = DebeziumParser::new(DebeziumSourceType::MongoDB);
        let _ = DebeziumParser::new(DebeziumSourceType::SQLServer);
        let _ = DebeziumParser::new(DebeziumSourceType::Oracle);
        let _ = DebeziumParser::new(DebeziumSourceType::Cassandra);
    }

    #[test]
    fn test_debezium_parser_snapshot_read() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        // "r" operation is a snapshot read, treated as insert
        let json = r#"{
            "op": "r",
            "ts_ms": 1234567890,
            "source": {"table": "users"},
            "after": {"id": 1}
        }"#;

        let event = parser.parse_json(json).unwrap();
        assert_eq!(event.operation, OperationType::Insert);
    }

    #[test]
    fn test_debezium_parser_truncate() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        let json = r#"{
            "op": "t",
            "ts_ms": 1234567890,
            "source": {"table": "users"}
        }"#;

        let event = parser.parse_json(json).unwrap();
        assert_eq!(event.operation, OperationType::Drop);
    }

    #[test]
    fn test_debezium_parser_payload_wrapped() {
        let parser = DebeziumParser::new(DebeziumSourceType::PostgreSQL);

        // Kafka Connect wraps in "payload"
        let json = r#"{
            "payload": {
                "op": "c",
                "ts_ms": 1234567890,
                "source": {"table": "users"},
                "after": {"id": 1}
            }
        }"#;

        let event = parser.parse_json(json).unwrap();
        assert_eq!(event.operation, OperationType::Insert);
        assert_eq!(event.collection, "users");
    }
}
