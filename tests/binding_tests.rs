//! Binding Integration Tests
//!
//! Tests for Python, WASM, and UniFFI bindings.
//! These tests verify the binding modules compile and work correctly.

use needle::{Collection, CollectionConfig, DistanceFunction};
use serde_json::json;

/// Test that Collection can be used in a way compatible with bindings
#[test]
fn test_collection_binding_compatibility() {
    let config = CollectionConfig::new("test", 128).with_distance(DistanceFunction::Cosine);
    let mut collection = Collection::new(config);

    // Insert with metadata (as bindings would)
    let vector: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let metadata = json!({"key": "value", "number": 42});
    collection
        .insert("test_id", &vector, Some(metadata))
        .unwrap();

    // Search (as bindings would)
    let query: Vec<f32> = (0..128).map(|i| (i as f32) / 128.0).collect();
    let results = collection.search(&query, 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "test_id");

    // Get (as bindings would)
    let (vec, meta) = collection.get("test_id").unwrap();
    assert_eq!(vec.len(), 128);
    assert!(meta.is_some());

    // Delete (as bindings would)
    assert!(collection.delete("test_id").unwrap());
    assert!(collection.get("test_id").is_none());
}

/// Test batch operations (used by bindings)
#[test]
fn test_batch_operations_binding_compatibility() {
    let config = CollectionConfig::new("batch_test", 64);
    let mut collection = Collection::new(config);

    // Batch insert
    let ids: Vec<String> = (0..10).map(|i| format!("vec_{}", i)).collect();
    let vectors: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..64).map(|j| ((i * 64 + j) as f32) / 640.0).collect())
        .collect();
    let metadata: Vec<Option<serde_json::Value>> =
        (0..10).map(|i| Some(json!({"index": i}))).collect();

    collection.insert_batch(ids, vectors, metadata).unwrap();
    assert_eq!(collection.len(), 10);

    // Batch search
    let queries: Vec<Vec<f32>> = (0..3)
        .map(|i| (0..64).map(|j| ((i * 64 + j) as f32) / 640.0).collect())
        .collect();
    let results = collection.batch_search(&queries, 5).unwrap();
    assert_eq!(results.len(), 3);
    for r in &results {
        assert!(r.len() <= 5);
    }
}

/// Test serialization (used by bindings for persistence)
#[test]
fn test_serialization_binding_compatibility() {
    let config = CollectionConfig::new("serialize_test", 32);
    let mut collection = Collection::new(config);

    // Add data
    for i in 0..5 {
        let vector: Vec<f32> = (0..32).map(|j| ((i * 32 + j) as f32) / 160.0).collect();
        collection
            .insert(format!("vec_{}", i), &vector, None)
            .unwrap();
    }

    // Serialize
    let bytes = collection.to_bytes().unwrap();
    assert!(!bytes.is_empty());

    // Deserialize
    let restored = Collection::from_bytes(&bytes).unwrap();
    assert_eq!(restored.len(), 5);
    assert_eq!(restored.dimensions(), 32);

    // Verify data is intact
    for i in 0..5 {
        let id = format!("vec_{}", i);
        assert!(restored.get(&id).is_some());
    }
}

/// Test search with metadata filter (used by bindings)
#[test]
fn test_filtered_search_binding_compatibility() {
    use needle::Filter;

    let config = CollectionConfig::new("filter_test", 16);
    let mut collection = Collection::new(config);

    // Insert vectors with different categories
    for i in 0..20 {
        let vector: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32) / 320.0).collect();
        let category = if i % 2 == 0 { "even" } else { "odd" };
        collection
            .insert(
                format!("vec_{}", i),
                &vector,
                Some(json!({"category": category, "value": i})),
            )
            .unwrap();
    }

    // Search with filter
    let query: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
    let filter = Filter::eq("category", "even");
    let results = collection.search_with_filter(&query, 10, &filter).unwrap();

    // All results should be "even"
    for result in &results {
        let meta = result.metadata.as_ref().unwrap();
        assert_eq!(meta["category"], "even");
    }
}

/// Test different distance functions (used by bindings)
#[test]
fn test_distance_functions_binding_compatibility() {
    let distances = [
        DistanceFunction::Cosine,
        DistanceFunction::Euclidean,
        DistanceFunction::DotProduct,
        DistanceFunction::Manhattan,
    ];

    for distance in distances {
        let config = CollectionConfig::new("dist_test", 8).with_distance(distance);
        let mut collection = Collection::new(config);

        let v1: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2: Vec<f32> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        collection.insert("v1", &v1, None).unwrap();
        collection.insert("v2", &v2, None).unwrap();

        let results = collection.search(&v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "v1"); // Should be closest to itself
    }
}

/// Test edge cases that bindings might encounter
#[test]
fn test_edge_cases_binding_compatibility() {
    let config = CollectionConfig::new("edge_test", 4);
    let mut collection = Collection::new(config);

    // Empty search
    let query: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let results = collection.search(&query, 10).unwrap();
    assert!(results.is_empty());

    // Insert and search for more than exists
    collection.insert("only_one", &query, None).unwrap();
    let results = collection.search(&query, 100).unwrap();
    assert_eq!(results.len(), 1);

    // Search with k=0
    let results = collection.search(&query, 0).unwrap();
    assert!(results.is_empty());
}

/// Test that error types work correctly (used by bindings for error handling)
#[test]
fn test_error_handling_binding_compatibility() {
    let config = CollectionConfig::new("error_test", 4);
    let mut collection = Collection::new(config);

    // Dimension mismatch
    let wrong_dim = vec![1.0, 2.0, 3.0]; // 3 dims instead of 4
    let result = collection.insert("bad", &wrong_dim, None);
    assert!(result.is_err());

    // Insert valid vector
    let valid = vec![1.0, 2.0, 3.0, 4.0];
    collection.insert("good", &valid, None).unwrap();

    // Duplicate ID
    let result = collection.insert("good", &valid, None);
    assert!(result.is_err());

    // Delete non-existent
    let result = collection.delete("nonexistent");
    assert!(result.is_ok()); // Should return Ok(false), not error
    assert!(!result.unwrap());
}

/// Test stats and info methods (used by bindings)
#[test]
fn test_stats_binding_compatibility() {
    let config = CollectionConfig::new("stats_test", 64);
    let mut collection = Collection::new(config);

    // Check initial state
    assert_eq!(collection.len(), 0);
    assert!(collection.is_empty());
    assert_eq!(collection.dimensions(), 64);
    assert_eq!(collection.name(), "stats_test");

    // Add some vectors
    for i in 0..10 {
        let vector: Vec<f32> = (0..64).map(|j| ((i * 64 + j) as f32) / 640.0).collect();
        collection
            .insert(format!("vec_{}", i), &vector, None)
            .unwrap();
    }

    // Check updated state
    assert_eq!(collection.len(), 10);
    assert!(!collection.is_empty());

    // Get stats
    let stats = collection.stats();
    assert_eq!(stats.vector_count, 10);
    assert_eq!(stats.dimensions, 64);
}

#[cfg(feature = "uniffi-bindings")]
mod uniffi_tests {
    use needle::uniffi_bindings::*;

    #[test]
    fn test_uniffi_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_uniffi_collection() {
        let collection =
            NeedleCollection::new("test".to_string(), 128, "cosine".to_string()).unwrap();

        assert_eq!(collection.name(), "test");
        assert_eq!(collection.dimensions(), 128);
        assert!(collection.is_empty());
    }
}
