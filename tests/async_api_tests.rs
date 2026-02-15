//! Integration tests for the async API module.
//!
//! Tests cover all public async operations including database management,
//! collection operations, vector CRUD, search, batch operations, and streaming.
//!
//! Run with: cargo test --test async_api_tests --features server

#![cfg(feature = "server")]

use needle::async_api::{AsyncDatabase, AsyncDatabaseConfig};
use needle::collection::CollectionConfig;
use needle::distance::DistanceFunction;
use needle::metadata::Filter;
use serde_json::json;

/// Helper to create a random vector of given dimension
fn random_vector(dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut hasher = DefaultHasher::new();
    (0..dim)
        .map(|i| {
            (seed + i as u64).hash(&mut hasher);
            (hasher.finish() % 1000) as f32 / 1000.0
        })
        .collect()
}

// ============================================================================
// Database Lifecycle Tests
// ============================================================================

#[tokio::test]
async fn test_in_memory_database() {
    let db = AsyncDatabase::in_memory();
    let collections = db.list_collections().await;
    assert!(collections.is_empty());
}

#[tokio::test]
async fn test_in_memory_with_config() {
    let config = AsyncDatabaseConfig::default()
        .with_max_concurrency(8)
        .with_stream_batch_size(50);

    let db = AsyncDatabase::in_memory_with_config(config);
    let collections = db.list_collections().await;
    assert!(collections.is_empty());
}

#[tokio::test]
async fn test_open_database() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.needle");

    // Create and populate database
    {
        let db = AsyncDatabase::open(db_path.clone()).await.unwrap();
        db.create_collection("test", 128).await.unwrap();
        db.save().await.unwrap();
    }

    // Reopen and verify
    {
        let db = AsyncDatabase::open(db_path).await.unwrap();
        assert!(db.has_collection("test").await);
    }
}

// ============================================================================
// Collection Management Tests
// ============================================================================

#[tokio::test]
async fn test_create_collection() {
    let db = AsyncDatabase::in_memory();

    db.create_collection("vectors", 384).await.unwrap();

    assert!(db.has_collection("vectors").await);
    assert!(!db.has_collection("nonexistent").await);
}

#[tokio::test]
async fn test_create_collection_with_config() {
    let db = AsyncDatabase::in_memory();

    let config = CollectionConfig::new("documents", 768).with_distance(DistanceFunction::Euclidean);

    db.create_collection_with_config(config).await.unwrap();

    assert!(db.has_collection("documents").await);
}

#[tokio::test]
async fn test_create_duplicate_collection_fails() {
    let db = AsyncDatabase::in_memory();

    db.create_collection("test", 128).await.unwrap();

    let result = db.create_collection("test", 128).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_list_collections() {
    let db = AsyncDatabase::in_memory();

    db.create_collection("alpha", 64).await.unwrap();
    db.create_collection("beta", 128).await.unwrap();
    db.create_collection("gamma", 256).await.unwrap();

    let collections = db.list_collections().await;
    assert_eq!(collections.len(), 3);
    assert!(collections.contains(&"alpha".to_string()));
    assert!(collections.contains(&"beta".to_string()));
    assert!(collections.contains(&"gamma".to_string()));
}

#[tokio::test]
async fn test_drop_collection() {
    let db = AsyncDatabase::in_memory();

    db.create_collection("temporary", 64).await.unwrap();
    assert!(db.has_collection("temporary").await);

    let dropped = db.drop_collection("temporary").await.unwrap();
    assert!(dropped);
    assert!(!db.has_collection("temporary").await);

    // Dropping non-existent collection
    let dropped_again = db.drop_collection("temporary").await.unwrap();
    assert!(!dropped_again);
}

#[tokio::test]
async fn test_collection_count() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    assert_eq!(db.count("test").await, 0);

    db.insert("test", "v1", vec![0.1; 64], None).await.unwrap();
    db.insert("test", "v2", vec![0.2; 64], None).await.unwrap();

    assert_eq!(db.count("test").await, 2);
}

// ============================================================================
// Vector CRUD Tests
// ============================================================================

#[tokio::test]
async fn test_insert_vector() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 128).await.unwrap();

    let vector = random_vector(128);
    let metadata = json!({"category": "test", "score": 0.95});

    db.insert("test", "doc1", vector.clone(), Some(metadata.clone()))
        .await
        .unwrap();

    let (retrieved_vec, retrieved_meta) = db.get("test", "doc1").await.unwrap();

    assert_eq!(retrieved_vec.len(), 128);
    assert_eq!(retrieved_meta.unwrap()["category"], "test");
}

#[tokio::test]
async fn test_insert_without_metadata() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    let vector = vec![0.5; 64];
    db.insert("test", "simple", vector.clone(), None)
        .await
        .unwrap();

    let (retrieved_vec, retrieved_meta) = db.get("test", "simple").await.unwrap();
    assert_eq!(retrieved_vec.len(), 64);
    assert!(retrieved_meta.is_none());
}

#[tokio::test]
async fn test_insert_duplicate_fails() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    db.insert("test", "dup", vec![0.1; 64], None).await.unwrap();

    let result = db.insert("test", "dup", vec![0.2; 64], None).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_insert_dimension_mismatch_fails() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 128).await.unwrap();

    // Wrong dimension
    let result = db.insert("test", "wrong", vec![0.1; 64], None).await;
    assert!(result.is_err());
}

// Note: AsyncDatabase doesn't have an update method - use delete + insert pattern

#[tokio::test]
async fn test_delete_vector() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    db.insert("test", "doc1", vec![0.1; 64], None)
        .await
        .unwrap();
    assert_eq!(db.count("test").await, 1);

    let deleted = db.delete("test", "doc1").await.unwrap();
    assert!(deleted);
    assert_eq!(db.count("test").await, 0);

    // Delete non-existent
    let deleted_again = db.delete("test", "doc1").await.unwrap();
    assert!(!deleted_again);
}

#[tokio::test]
async fn test_get_nonexistent_returns_none() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    let result = db.get("test", "ghost").await;
    assert!(result.is_none());
}

// ============================================================================
// Search Tests
// ============================================================================

#[tokio::test]
async fn test_basic_search() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Insert some vectors
    for i in 0..10 {
        let mut vec = vec![0.0; 64];
        vec[0] = i as f32 / 10.0;
        db.insert("test", &format!("doc{}", i), vec, None)
            .await
            .unwrap();
    }

    // Search for vectors similar to [0.5, 0, 0, ...]
    let query = {
        let mut v = vec![0.0; 64];
        v[0] = 0.5;
        v
    };

    let results = db.search("test", query, 3).await.unwrap();

    assert_eq!(results.len(), 3);
    // Results should be ordered by distance
    assert!(results[0].distance <= results[1].distance);
    assert!(results[1].distance <= results[2].distance);
}

#[tokio::test]
async fn test_search_with_filter() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Insert vectors with categories
    for i in 0..20 {
        let category = if i % 2 == 0 { "even" } else { "odd" };
        db.insert(
            "test",
            &format!("doc{}", i),
            random_vector(64),
            Some(json!({"category": category, "index": i})),
        )
        .await
        .unwrap();
    }

    let query = random_vector(64);
    let filter = Filter::parse(&json!({"category": "even"})).unwrap();

    let results = db
        .search_with_filter("test", query, 5, filter)
        .await
        .unwrap();

    assert!(!results.is_empty());
    for result in results {
        // All results should be "even" category
        assert!(result.metadata.is_some());
        assert_eq!(result.metadata.as_ref().unwrap()["category"], "even");
    }
}

#[tokio::test]
async fn test_search_empty_collection() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("empty", 64).await.unwrap();

    let results = db.search("empty", vec![0.1; 64], 10).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_search_k_larger_than_collection() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("small", 64).await.unwrap();

    db.insert("small", "v1", vec![0.1; 64], None).await.unwrap();
    db.insert("small", "v2", vec![0.2; 64], None).await.unwrap();

    let results = db.search("small", vec![0.15; 64], 100).await.unwrap();
    assert_eq!(results.len(), 2); // Only 2 vectors exist
}

// ============================================================================
// Batch Operation Tests
// ============================================================================

#[tokio::test]
async fn test_batch_insert() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    let batch: Vec<(String, Vec<f32>, Option<serde_json::Value>)> = (0..100)
        .map(|i| {
            (
                format!("doc{}", i),
                random_vector(64),
                Some(json!({"index": i})),
            )
        })
        .collect();

    db.batch_insert("test", batch).await.unwrap();

    assert_eq!(db.count("test").await, 100);
}

#[tokio::test]
async fn test_batch_search() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Insert vectors
    for i in 0..50 {
        db.insert("test", &format!("doc{}", i), random_vector(64), None)
            .await
            .unwrap();
    }

    // Multiple queries
    let queries: Vec<Vec<f32>> = (0..5).map(|_| random_vector(64)).collect();

    let all_results = db.batch_search("test", queries.clone(), 3).await.unwrap();

    assert_eq!(all_results.len(), 5);
    for results in &all_results {
        assert_eq!(results.len(), 3);
    }
}

#[tokio::test]
async fn test_batch_delete() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Insert
    for i in 0..20 {
        db.insert("test", &format!("doc{}", i), random_vector(64), None)
            .await
            .unwrap();
    }

    // Delete some
    let ids: Vec<String> = (0..10).map(|i| format!("doc{}", i)).collect();
    let deleted_count = db.batch_delete("test", ids).await.unwrap();

    assert_eq!(deleted_count, 10);
    assert_eq!(db.count("test").await, 10);
}

// ============================================================================
// Concurrent Operation Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_inserts() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    let db = std::sync::Arc::new(db);
    let mut handles = vec![];

    for i in 0..10 {
        let db_clone = db.clone();
        handles.push(tokio::spawn(async move {
            for j in 0..10 {
                let id = format!("doc_{}_{}", i, j);
                db_clone
                    .insert("test", &id, random_vector(64), None)
                    .await
                    .unwrap();
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    assert_eq!(db.count("test").await, 100);
}

#[tokio::test]
async fn test_concurrent_reads_and_writes() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Pre-populate
    for i in 0..50 {
        db.insert("test", &format!("doc{}", i), random_vector(64), None)
            .await
            .unwrap();
    }

    let db = std::sync::Arc::new(db);
    let mut handles = vec![];

    // Concurrent searches
    for _ in 0..5 {
        let db_clone = db.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..10 {
                let results = db_clone.search("test", random_vector(64), 5).await.unwrap();
                assert!(!results.is_empty());
            }
        }));
    }

    // Concurrent inserts
    for i in 0..5 {
        let db_clone = db.clone();
        handles.push(tokio::spawn(async move {
            for j in 0..10 {
                let id = format!("new_{}_{}", i, j);
                db_clone
                    .insert("test", &id, random_vector(64), None)
                    .await
                    .unwrap();
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // 50 original + 50 new
    assert_eq!(db.count("test").await, 100);
}

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
async fn test_bulk_operations_complete() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Insert 100 vectors - should complete without issues
    for i in 0..100 {
        db.insert("test", &format!("doc{}", i), random_vector(64), None)
            .await
            .unwrap();
    }

    assert_eq!(db.count("test").await, 100);
}

// ============================================================================
// Compact Tests
// ============================================================================

#[tokio::test]
async fn test_compact_collection() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    // Insert and delete to create gaps
    for i in 0..100 {
        db.insert("test", &format!("doc{}", i), random_vector(64), None)
            .await
            .unwrap();
    }

    for i in 0..50 {
        db.delete("test", &format!("doc{}", i)).await.unwrap();
    }

    // Compact should reclaim space
    db.compact("test").await.unwrap();

    assert_eq!(db.count("test").await, 50);
}

// ============================================================================
// Export Tests
// ============================================================================

#[tokio::test]
async fn test_export_collection() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    for i in 0..10 {
        db.insert(
            "test",
            &format!("doc{}", i),
            vec![i as f32 / 10.0; 64],
            Some(json!({"index": i})),
        )
        .await
        .unwrap();
    }

    let exported = db.export("test").await.unwrap();

    assert_eq!(exported.len(), 10);
    for (id, vector, metadata) in &exported {
        assert!(id.starts_with("doc"));
        assert_eq!(vector.len(), 64);
        assert!(metadata.is_some());
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_operation_on_nonexistent_collection() {
    let db = AsyncDatabase::in_memory();

    // Insert to non-existent collection
    let result = db.insert("ghost", "doc1", vec![0.1; 64], None).await;
    assert!(result.is_err());

    // Search in non-existent collection
    let result = db.search("ghost", vec![0.1; 64], 10).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_filter_no_matches() {
    let db = AsyncDatabase::in_memory();
    db.create_collection("test", 64).await.unwrap();

    for i in 0..10 {
        db.insert(
            "test",
            &format!("doc{}", i),
            random_vector(64),
            Some(json!({"type": "A"})),
        )
        .await
        .unwrap();
    }

    let filter = Filter::parse(&json!({"type": "B"})).unwrap();
    let results = db
        .search_with_filter("test", random_vector(64), 10, filter)
        .await
        .unwrap();

    assert!(results.is_empty());
}

// ============================================================================
// Database Save/Persistence Tests
// ============================================================================

#[tokio::test]
async fn test_save_and_reload() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("persist.needle");

    // Create and populate
    {
        let db = AsyncDatabase::open(db_path.clone()).await.unwrap();
        db.create_collection("test", 64).await.unwrap();

        for i in 0..10 {
            db.insert(
                "test",
                &format!("doc{}", i),
                vec![i as f32; 64],
                Some(json!({"i": i})),
            )
            .await
            .unwrap();
        }

        db.save().await.unwrap();
    }

    // Reload and verify
    {
        let db = AsyncDatabase::open(db_path).await.unwrap();
        assert!(db.has_collection("test").await);
        assert_eq!(db.count("test").await, 10);

        let (vec, meta) = db.get("test", "doc5").await.unwrap();
        assert!((vec[0] - 5.0).abs() < 0.001);
        assert_eq!(meta.unwrap()["i"], 5);
    }
}
