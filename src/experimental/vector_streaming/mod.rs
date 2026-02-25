#![allow(clippy::unwrap_used)]
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Vector Streaming API
//!
//! **DEPRECATED**: This module overlaps with `crate::streaming` and will be merged in a
//! future release. Prefer using `crate::streaming` for new code.
//!
//! Real-time CDC connectors for ingesting vectors from message queues like Kafka and Pulsar
//! with exactly-once semantics and automatic batching.
//!
//! # Features
//!
//! - **Kafka Consumer**: Consume vectors from Kafka topics with offset management
//! - **Pulsar Consumer**: Consume vectors from Pulsar topics with acknowledgments
//! - **Exactly-Once Semantics**: Transactional ingestion with deduplication
//! - **Automatic Batching**: Configurable batch sizes for optimal throughput
//! - **Backpressure Handling**: Pause/resume based on collection capacity
//! - **Schema Registry**: Support for Avro, JSON, and binary vector formats
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::vector_streaming::{VectorConsumer, ConsumerConfig, VectorMessage};
//!
//! let config = ConsumerConfig::kafka("localhost:9092", "vectors-topic")
//!     .group_id("needle-consumer")
//!     .batch_size(100)
//!     .auto_commit(false);
//!
//! let consumer = VectorConsumer::new(config).await?;
//!
//! // Process vectors
//! while let Some(batch) = consumer.poll().await? {
//!     for msg in batch {
//!         collection.insert(&msg.id, &msg.vector, msg.metadata)?;
//!     }
//!     consumer.commit().await?;
//! }
//! ```

pub mod consumer;
pub mod pipeline;
pub mod processor;
pub mod producer;
pub mod protocol;

// Re-export all public items so `use crate::experimental::vector_streaming::*` still works.

pub use consumer::{
    ConsumerConfig, ConsumerStats, MessageSource, OffsetState, VectorConsumer, VectorFormat,
    VectorMessage,
};

pub use producer::{AckMode, CompressionType, ProducerConfig, ProducerStats, VectorProducer};

pub use processor::{
    BatchResult, Checkpoint, CheckpointStore, DeadLetterQueue, DlqStats, FailedMessage,
    InMemoryCheckpointStore, StreamProcessor, TransactionalProcessor, WatermarkTracker,
};

pub use pipeline::{
    BackpressureConfig, BackpressureController, BackpressureState, BackpressureStats, CdcEvent,
    CdcEventType, CdcStream, ReplayManager, StreamMetrics, StreamMetricsSnapshot, StreamOp,
    StreamSnapshot, VectorStreamPipeline,
};

#[cfg(feature = "cdc-kafka")]
pub use protocol::KafkaVectorConsumer;

#[cfg(feature = "cdc-pulsar")]
pub use protocol::PulsarVectorConsumer;

#[cfg(feature = "cdc-postgres")]
pub use protocol::PostgresVectorConsumer;

#[cfg(feature = "cdc-mongodb")]
pub use protocol::MongoVectorConsumer;

// Helper functions used by sub-modules

use crate::error::{NeedleError, Result};

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn base64_decode(data: &[u8]) -> Result<Vec<u8>> {
    // Simple base64 decode (would use base64 crate in production)
    let s = std::str::from_utf8(data)
        .map_err(|e| NeedleError::InvalidInput(format!("Invalid UTF-8: {}", e)))?;

    // Basic base64 decode
    let mut result = Vec::new();
    let chars: Vec<u8> = s.bytes().filter(|&b| b != b'\n' && b != b'\r').collect();

    for chunk in chars.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let values: Vec<u8> = chunk
            .iter()
            .map(|&c| match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => 0,
                _ => 0,
            })
            .collect();

        result.push((values[0] << 2) | (values[1] >> 4));
        if chunk[2] != b'=' {
            result.push((values[1] << 4) | (values[2] >> 2));
        }
        if chunk[3] != b'=' {
            result.push((values[2] << 6) | values[3]);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::RwLock;
    use std::sync::Arc;

    #[test]
    fn test_consumer_config() {
        let config = ConsumerConfig::kafka("localhost:9092", "test-topic")
            .group_id("test-group")
            .batch_size(50)
            .exactly_once(true);

        assert_eq!(config.source, MessageSource::Kafka);
        assert_eq!(config.brokers, "localhost:9092");
        assert_eq!(config.topic, "test-topic");
        assert_eq!(config.group_id, "test-group");
        assert_eq!(config.batch_size, 50);
        assert!(config.exactly_once);
    }

    #[test]
    fn test_mock_consumer() {
        let config = ConsumerConfig::mock();
        let consumer = VectorConsumer::new(config).unwrap();
        consumer.start().unwrap();

        // Add mock messages
        let messages = vec![
            VectorMessage {
                id: "v1".to_string(),
                vector: vec![1.0, 2.0, 3.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v2".to_string(),
                vector: vec![4.0, 5.0, 6.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        // Poll
        let batch = consumer.poll().unwrap().unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].id, "v1");
        assert_eq!(batch[1].id, "v2");

        consumer.stop().unwrap();
    }

    #[test]
    fn test_deduplication() {
        let config = ConsumerConfig::mock().deduplication(true);
        let consumer = VectorConsumer::new(config).unwrap();
        consumer.start().unwrap();

        // Add duplicate messages
        let messages = vec![
            VectorMessage {
                id: "v1".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v1".to_string(), // Duplicate
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v2".to_string(),
                vector: vec![3.0],
                metadata: None,
                offset: 2,
                partition: None,
                timestamp: 0,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        let batch = consumer.poll().unwrap().unwrap();
        assert_eq!(batch.len(), 2); // Duplicate filtered
        assert_eq!(batch[0].id, "v1");
        assert_eq!(batch[1].id, "v2");

        let stats = consumer.stats();
        assert_eq!(stats.duplicates_filtered, 1);
    }

    #[test]
    fn test_json_parsing() {
        let config = ConsumerConfig::mock();
        let consumer = VectorConsumer::new(config).unwrap();

        let json = r#"{"id": "vec1", "vector": [1.0, 2.0, 3.0], "metadata": {"key": "value"}}"#;
        let msg = consumer.parse_json_message(json, 0).unwrap();

        assert_eq!(msg.id, "vec1");
        assert_eq!(msg.vector, vec![1.0, 2.0, 3.0]);
        assert!(msg.metadata.is_some());
    }

    #[test]
    fn test_binary_parsing() {
        let config = ConsumerConfig::mock().vector_format(VectorFormat::BinaryF32LE);
        let consumer = VectorConsumer::new(config).unwrap();

        // Create binary data for [1.0, 2.0]
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());

        let vector = consumer.parse_binary_vector(&data).unwrap();
        assert_eq!(vector, vec![1.0, 2.0]);
    }

    #[test]
    fn test_producer() {
        let config = ProducerConfig {
            source: MessageSource::Mock,
            ..Default::default()
        };
        let producer = VectorProducer::new(config).unwrap();

        producer.send("v1", &[1.0, 2.0], None).unwrap();
        producer
            .send("v2", &[3.0, 4.0], Some(serde_json::json!({"key": "value"})))
            .unwrap();

        let output = producer.get_mock_output();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].id, "v1");
        assert_eq!(output[1].id, "v2");
    }

    #[test]
    fn test_offset_commit() {
        let config = ConsumerConfig::mock();
        let consumer = VectorConsumer::new(config).unwrap();

        consumer.commit_offset(0, 100).unwrap();
        consumer.commit_offset(1, 200).unwrap();

        let state = consumer.offset_state();
        assert_eq!(state.committed.get(&0), Some(&100));
        assert_eq!(state.committed.get(&1), Some(&200));
    }

    // Exactly-once enhancement tests

    #[test]
    fn test_checkpoint() {
        let mut checkpoint = Checkpoint::new("test-group", "test-topic");

        checkpoint.update_offset(0, 100);
        checkpoint.update_offset(1, 200);
        checkpoint.mark_processed("msg-1");
        checkpoint.mark_processed("msg-2");
        checkpoint.update_watermark(1000);

        assert_eq!(checkpoint.offsets.get(&0), Some(&100));
        assert_eq!(checkpoint.offsets.get(&1), Some(&200));
        assert!(checkpoint.is_processed("msg-1"));
        assert!(checkpoint.is_processed("msg-2"));
        assert!(!checkpoint.is_processed("msg-3"));
        assert_eq!(checkpoint.watermark, 1000);
    }

    #[test]
    fn test_in_memory_checkpoint_store() {
        let store = InMemoryCheckpointStore::new();

        let mut checkpoint = Checkpoint::new("group1", "topic1");
        checkpoint.update_offset(0, 500);

        store.save(&checkpoint).unwrap();

        let loaded = store.load("group1", "topic1").unwrap().unwrap();
        assert_eq!(loaded.group_id, "group1");
        assert_eq!(loaded.offsets.get(&0), Some(&500));

        store.delete("group1", "topic1").unwrap();
        assert!(store.load("group1", "topic1").unwrap().is_none());
    }

    #[test]
    fn test_dead_letter_queue() {
        let dlq = DeadLetterQueue::new(100);

        let msg = VectorMessage {
            id: "failed-msg".to_string(),
            vector: vec![1.0, 2.0],
            metadata: None,
            offset: 0,
            partition: None,
            timestamp: 0,
            key: None,
        };

        dlq.push(msg.clone(), "Processing failed", 1);
        assert_eq!(dlq.len(), 1);

        let stats = dlq.stats();
        assert_eq!(stats.total_failed, 1);
        assert_eq!(stats.queue_size, 1);

        let popped = dlq.pop().unwrap();
        assert_eq!(popped.message.id, "failed-msg");
        assert_eq!(popped.error, "Processing failed");
        assert_eq!(popped.retry_count, 1);

        assert!(dlq.is_empty());
    }

    #[test]
    fn test_watermark_tracker() {
        let tracker = WatermarkTracker::new(1000);

        tracker.update(0, 5000);
        tracker.update(1, 3000);
        tracker.update(2, 7000);

        // Global watermark should be minimum
        assert_eq!(tracker.watermark(), 3000);

        // Partition watermarks
        assert_eq!(tracker.partition_watermark(0), Some(5000));
        assert_eq!(tracker.partition_watermark(1), Some(3000));
        assert_eq!(tracker.partition_watermark(2), Some(7000));

        // Late message check (with 1000ms allowed lateness)
        assert!(tracker.is_late(1000)); // Too late
        assert!(!tracker.is_late(2500)); // Within lateness window
        assert!(!tracker.is_late(4000)); // Ahead of watermark
    }

    #[test]
    fn test_batch_result() {
        let result = BatchResult {
            processed: 10,
            skipped: 2,
            failed: 1,
        };

        assert_eq!(result.total(), 13);
    }

    #[test]
    fn test_transactional_processor() {
        let config = ConsumerConfig::mock();
        let consumer = Arc::new(VectorConsumer::new(config).unwrap());
        let store = Arc::new(InMemoryCheckpointStore::new());

        let processed_ids = Arc::new(RwLock::new(Vec::new()));
        let ids_clone = processed_ids.clone();

        let processor =
            TransactionalProcessor::new(consumer.clone(), store, move |msg: &VectorMessage| {
                ids_clone.write().push(msg.id.clone());
                Ok(())
            })
            .unwrap()
            .with_max_retries(3)
            .with_checkpoint_interval(10);

        consumer.start().unwrap();

        // Add messages
        let messages = vec![
            VectorMessage {
                id: "t1".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: Some(0),
                timestamp: 1000,
                key: None,
            },
            VectorMessage {
                id: "t2".to_string(),
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: Some(0),
                timestamp: 2000,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        // Process
        let result = processor.process_batch().unwrap();
        assert_eq!(result.processed, 2);
        assert_eq!(result.failed, 0);

        // Check idempotency - same messages should be skipped
        let checkpoint = processor.checkpoint();
        assert!(checkpoint.is_processed("t1"));
        assert!(checkpoint.is_processed("t2"));
    }

    #[test]
    fn test_backpressure_flowing() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 100,
            max_in_flight_pause: 500,
            ..Default::default()
        });
        assert_eq!(controller.state(), BackpressureState::Flowing);
        assert!(controller.should_poll());
        assert_eq!(controller.effective_batch_size(50), 50);
    }

    #[test]
    fn test_backpressure_throttle() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 100,
            max_in_flight_pause: 500,
            throttle_factor: 0.25,
            ..Default::default()
        });
        controller.on_receive(150);
        assert_eq!(controller.state(), BackpressureState::Throttled);
        assert!(controller.should_poll());
        assert_eq!(controller.effective_batch_size(100), 25);
    }

    #[test]
    fn test_backpressure_pause_and_recover() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 100,
            max_in_flight_pause: 500,
            ..Default::default()
        });
        controller.on_receive(600);
        assert_eq!(controller.state(), BackpressureState::Paused);
        assert!(!controller.should_poll());
        assert_eq!(controller.effective_batch_size(100), 0);

        // Commit enough to recover
        controller.on_commit(550);
        assert_eq!(controller.state(), BackpressureState::Flowing);
        assert!(controller.should_poll());
    }

    #[test]
    fn test_backpressure_capacity() {
        let controller = BackpressureController::new(BackpressureConfig {
            capacity_throttle_pct: 80.0,
            capacity_pause_pct: 95.0,
            ..Default::default()
        });
        controller.evaluate_capacity(85_000, 100_000);
        assert_eq!(controller.state(), BackpressureState::Throttled);

        controller.evaluate_capacity(96_000, 100_000);
        assert_eq!(controller.state(), BackpressureState::Paused);
    }

    // ========================================================================
    // Extended streaming tests
    // ========================================================================

    #[test]
    fn test_consumer_lifecycle() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        assert!(!consumer.is_running());

        consumer.start().unwrap();
        assert!(consumer.is_running());

        // Double start should error
        let result = consumer.start();
        assert!(result.is_err());

        consumer.stop().unwrap();
        assert!(!consumer.is_running());
    }

    #[test]
    fn test_consumer_poll_not_started() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        // Poll without starting should return None
        let result = consumer.poll().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_consumer_poll_empty_buffer() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        consumer.start().unwrap();
        // Poll with empty buffer should return None
        let result = consumer.poll().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_consumer_batch_size_limit() {
        let config = ConsumerConfig::mock().batch_size(2);
        let consumer = VectorConsumer::new(config).unwrap();
        consumer.start().unwrap();

        // Add 5 messages
        let messages: Vec<VectorMessage> = (0..5)
            .map(|i| VectorMessage {
                id: format!("v{}", i),
                vector: vec![i as f32],
                metadata: None,
                offset: i as u64,
                partition: None,
                timestamp: 0,
                key: None,
            })
            .collect();
        consumer.add_mock_messages(messages);

        // First poll should return 2 (batch_size limit)
        let batch = consumer.poll().unwrap().unwrap();
        assert_eq!(batch.len(), 2);

        // Second poll should return 2 more
        let batch = consumer.poll().unwrap().unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_consumer_stats() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        consumer.start().unwrap();

        consumer.add_mock_messages(vec![VectorMessage {
            id: "v1".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
            offset: 0,
            partition: None,
            timestamp: 0,
            key: None,
        }]);

        consumer.poll().unwrap();
        let stats = consumer.stats();
        assert_eq!(stats.messages_consumed, 1);
        assert!(stats.bytes_consumed > 0);
    }

    #[test]
    fn test_json_parsing_missing_id() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        let json = r#"{"vector": [1.0, 2.0]}"#;
        let result = consumer.parse_json_message(json, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_parsing_missing_vector() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        let json = r#"{"id": "v1"}"#;
        let result = consumer.parse_json_message(json, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_parsing_empty_vector() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        let json = r#"{"id": "v1", "vector": []}"#;
        let result = consumer.parse_json_message(json, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_parsing_invalid_json() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();
        let result = consumer.parse_json_message("not json", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_parsing_wrong_length() {
        let config = ConsumerConfig::mock().vector_format(VectorFormat::BinaryF32LE);
        let consumer = VectorConsumer::new(config).unwrap();
        // 3 bytes is not a multiple of 4
        let result = consumer.parse_binary_vector(&[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_parsing_be() {
        let config = ConsumerConfig::mock().vector_format(VectorFormat::BinaryF32BE);
        let consumer = VectorConsumer::new(config).unwrap();
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_be_bytes());
        let vector = consumer.parse_binary_vector(&data).unwrap();
        assert_eq!(vector, vec![1.0]);
    }

    #[test]
    fn test_producer_batch() {
        let producer = VectorProducer::new(ProducerConfig::default()).unwrap();
        let batch = vec![
            ("id1".to_string(), vec![1.0], None),
            ("id2".to_string(), vec![2.0], None),
        ];
        let sent = producer.send_batch(&batch).unwrap();
        assert_eq!(sent, 2);

        let output = producer.get_mock_output();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_producer_stats() {
        let producer = VectorProducer::new(ProducerConfig::default()).unwrap();
        producer.send("v1", &[1.0], None).unwrap();
        let stats = producer.stats();
        assert_eq!(stats.messages_sent, 1);
    }

    #[test]
    fn test_producer_flush() {
        let producer = VectorProducer::new(ProducerConfig::default()).unwrap();
        assert!(producer.flush().is_ok());
    }

    #[test]
    fn test_stream_processor() {
        let consumer = Arc::new(VectorConsumer::new(ConsumerConfig::mock()).unwrap());
        let producer = Arc::new(VectorProducer::new(ProducerConfig::default()).unwrap());

        consumer.start().unwrap();
        consumer.add_mock_messages(vec![VectorMessage {
            id: "v1".to_string(),
            vector: vec![1.0, 2.0],
            metadata: None,
            offset: 0,
            partition: None,
            timestamp: 0,
            key: None,
        }]);

        let processor = StreamProcessor::new(consumer.clone(), |msg| {
            // Double each value
            let mut m = msg;
            m.vector = m.vector.iter().map(|v| v * 2.0).collect();
            Some(m)
        })
        .with_producer(producer.clone());

        let processed = processor.process_batch().unwrap();
        assert_eq!(processed, 1);

        let output = producer.get_mock_output();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].vector, vec![2.0, 4.0]);
    }

    #[test]
    fn test_stream_processor_filter() {
        let consumer = Arc::new(VectorConsumer::new(ConsumerConfig::mock()).unwrap());
        consumer.start().unwrap();
        consumer.add_mock_messages(vec![
            VectorMessage {
                id: "keep".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "drop".to_string(),
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
        ]);

        let processor = StreamProcessor::new(consumer.clone(), |msg| {
            if msg.id == "keep" {
                Some(msg)
            } else {
                None
            }
        });

        let processed = processor.process_batch().unwrap();
        assert_eq!(processed, 1);
    }

    #[test]
    fn test_checkpoint_trim_processed_ids() {
        let mut cp = Checkpoint::new("g", "t");
        for i in 0..100 {
            cp.mark_processed(&format!("msg-{}", i));
        }
        assert_eq!(cp.processed_ids.len(), 100);

        cp.trim_processed_ids(10);
        assert!(cp.processed_ids.len() <= 10);
    }

    #[test]
    fn test_checkpoint_watermark_only_increases() {
        let mut cp = Checkpoint::new("g", "t");
        cp.update_watermark(100);
        assert_eq!(cp.watermark, 100);

        cp.update_watermark(50); // Lower - should not change
        assert_eq!(cp.watermark, 100);

        cp.update_watermark(200);
        assert_eq!(cp.watermark, 200);
    }

    #[test]
    fn test_dlq_capacity_eviction() {
        let dlq = DeadLetterQueue::new(2);

        for i in 0..3 {
            dlq.push(
                VectorMessage {
                    id: format!("m{}", i),
                    vector: vec![],
                    metadata: None,
                    offset: i as u64,
                    partition: None,
                    timestamp: 0,
                    key: None,
                },
                "error",
                1,
            );
        }

        assert_eq!(dlq.len(), 2);
        let stats = dlq.stats();
        assert_eq!(stats.permanently_failed, 1);
    }

    #[test]
    fn test_dlq_clear() {
        let dlq = DeadLetterQueue::new(10);
        dlq.push(
            VectorMessage {
                id: "m1".to_string(),
                vector: vec![],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            "error",
            0,
        );
        assert!(!dlq.is_empty());

        dlq.clear();
        assert!(dlq.is_empty());
        assert_eq!(dlq.len(), 0);
    }

    #[test]
    fn test_dlq_peek_all() {
        let dlq = DeadLetterQueue::new(10);
        dlq.push(
            VectorMessage {
                id: "m1".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            "err1",
            0,
        );
        dlq.push(
            VectorMessage {
                id: "m2".to_string(),
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
            "err2",
            1,
        );

        let all = dlq.peek_all();
        assert_eq!(all.len(), 2);
        // peek_all should not remove items
        assert_eq!(dlq.len(), 2);
    }

    #[test]
    fn test_backpressure_stats() {
        let controller = BackpressureController::new(BackpressureConfig {
            max_in_flight: 10,
            max_in_flight_pause: 50,
            ..Default::default()
        });

        controller.on_receive(20); // Triggers throttle
        let stats = controller.stats();
        assert_eq!(stats.state, BackpressureState::Throttled);
        assert_eq!(stats.in_flight, 20);
        assert_eq!(stats.throttle_events, 1);
    }

    #[test]
    fn test_backpressure_zero_capacity() {
        let controller = BackpressureController::new(Default::default());
        // Zero max_vectors should not change state
        controller.evaluate_capacity(100, 0);
        assert_eq!(controller.state(), BackpressureState::Flowing);
    }

    #[test]
    fn test_consumer_config_variants() {
        let pulsar = ConsumerConfig::pulsar("pulsar://localhost:6650", "vectors");
        assert_eq!(pulsar.source, MessageSource::Pulsar);

        let pg = ConsumerConfig::postgres("postgres://localhost/db", "embeddings");
        assert_eq!(pg.source, MessageSource::Postgres);

        let mongo = ConsumerConfig::mongodb("mongodb://localhost", "vectors");
        assert_eq!(mongo.source, MessageSource::MongoDB);
    }

    #[test]
    fn test_producer_config_variants() {
        let kafka = ProducerConfig::kafka("localhost:9092", "output");
        assert_eq!(kafka.source, MessageSource::Kafka);

        let pulsar = ProducerConfig::pulsar("pulsar://localhost:6650", "output");
        assert_eq!(pulsar.source, MessageSource::Pulsar);
    }

    #[test]
    fn test_consumer_commit_pending_offsets() {
        let consumer = VectorConsumer::new(ConsumerConfig::mock()).unwrap();

        // Manually add pending offsets (via commit_offset)
        consumer.commit_offset(0, 100).unwrap();
        consumer.commit_offset(1, 200).unwrap();

        // Commit should succeed
        consumer.commit().unwrap();

        let state = consumer.offset_state();
        assert_eq!(state.committed.get(&0), Some(&100));
    }

    #[test]
    fn test_dedup_without_dedup_enabled() {
        let config = ConsumerConfig::mock().deduplication(false);
        let consumer = VectorConsumer::new(config).unwrap();
        consumer.start().unwrap();

        let messages = vec![
            VectorMessage {
                id: "v1".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: None,
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "v1".to_string(),
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: None,
                timestamp: 0,
                key: None,
            },
        ];
        consumer.add_mock_messages(messages);

        let batch = consumer.poll().unwrap().unwrap();
        // Both should come through since dedup is disabled
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_transactional_processor_with_failures() {
        let consumer = Arc::new(VectorConsumer::new(ConsumerConfig::mock()).unwrap());
        let store = Arc::new(InMemoryCheckpointStore::new());

        let processor = TransactionalProcessor::new(
            consumer.clone(),
            store,
            |msg: &VectorMessage| {
                if msg.id == "fail" {
                    Err(NeedleError::InvalidInput("forced failure".into()))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap()
        .with_max_retries(1)
        .with_checkpoint_interval(100);

        consumer.start().unwrap();
        consumer.add_mock_messages(vec![
            VectorMessage {
                id: "ok".to_string(),
                vector: vec![1.0],
                metadata: None,
                offset: 0,
                partition: Some(0),
                timestamp: 0,
                key: None,
            },
            VectorMessage {
                id: "fail".to_string(),
                vector: vec![2.0],
                metadata: None,
                offset: 1,
                partition: Some(0),
                timestamp: 0,
                key: None,
            },
        ]);

        let result = processor.process_batch().unwrap();
        assert_eq!(result.processed, 1);
        assert_eq!(result.failed, 1);

        // Failed message should be in DLQ
        assert_eq!(processor.dlq().len(), 1);
    }

    #[test]
    fn test_transactional_processor_save_checkpoint() {
        let consumer = Arc::new(VectorConsumer::new(ConsumerConfig::mock()).unwrap());
        let store = Arc::new(InMemoryCheckpointStore::new());

        let processor =
            TransactionalProcessor::new(consumer.clone(), store.clone(), |_: &VectorMessage| {
                Ok(())
            })
            .unwrap();

        // Save checkpoint
        processor.save_checkpoint().unwrap();

        // Verify checkpoint was persisted
        let loaded = store.load("needle-consumer", "vectors").unwrap();
        assert!(loaded.is_some());
    }
}
