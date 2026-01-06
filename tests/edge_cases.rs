//! Edge case tests for the Needle vector database
//! Tests boundary conditions, error handling, and unusual inputs

use needle::{Collection, CollectionConfig, Database, DistanceFunction};
use serde_json::json;

fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

// ============================================================================
// Empty Collection Tests
// ============================================================================

#[test]
fn test_search_empty_collection() {
    let collection = Collection::with_dimensions("test", 128);
    let query = random_vector(128);

    let results = collection.search(&query, 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_get_from_empty_collection() {
    let collection = Collection::with_dimensions("test", 128);
    assert!(collection.get("nonexistent").is_none());
}

#[test]
fn test_delete_from_empty_collection() {
    let mut collection = Collection::with_dimensions("test", 128);
    let result = collection.delete("nonexistent").unwrap();
    assert!(!result);
}

#[test]
fn test_contains_on_empty_collection() {
    let collection = Collection::with_dimensions("test", 128);
    assert!(!collection.contains("anything"));
}

#[test]
fn test_stats_on_empty_collection() {
    let collection = Collection::with_dimensions("test", 128);
    let stats = collection.stats();
    assert_eq!(stats.vector_count, 0);
    assert_eq!(stats.dimensions, 128);
}

#[test]
fn test_compact_empty_collection() {
    let mut collection = Collection::with_dimensions("test", 128);
    let removed = collection.compact().unwrap();
    assert_eq!(removed, 0);
}

// ============================================================================
// Dimension Edge Cases
// ============================================================================

#[test]
fn test_single_dimension_vectors() {
    let mut collection = Collection::with_dimensions("test", 1);

    collection.insert("v1", &[0.5], None).unwrap();
    collection.insert("v2", &[0.6], None).unwrap();
    collection.insert("v3", &[0.1], None).unwrap();

    let results = collection.search(&[0.55], 2).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_high_dimension_vectors() {
    let dim = 4096;
    let mut collection = Collection::with_dimensions("test", dim);

    for i in 0..10 {
        let vec = random_vector(dim);
        collection.insert(format!("v{}", i), &vec, None).unwrap();
    }

    let query = random_vector(dim);
    let results = collection.search(&query, 5).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_dimension_mismatch_insert() {
    let mut collection = Collection::with_dimensions("test", 128);

    // Wrong dimension should fail
    let wrong_vec = random_vector(64);
    let result = collection.insert("v1", &wrong_vec, None);
    assert!(result.is_err());
}

#[test]
fn test_dimension_mismatch_search() {
    let mut collection = Collection::with_dimensions("test", 128);
    collection.insert("v1", &random_vector(128), None).unwrap();

    // Wrong dimension query should fail
    let wrong_query = random_vector(64);
    let result = collection.search(&wrong_query, 10);
    assert!(result.is_err());
}

// ============================================================================
// Special Float Values
// ============================================================================

#[test]
fn test_zero_vector() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Zero vector - should be insertable
    let zero_vec = vec![0.0, 0.0, 0.0, 0.0];
    collection.insert("zero", &zero_vec, None).unwrap();

    // Should be retrievable
    let (vec, _) = collection.get("zero").unwrap();
    assert_eq!(vec, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_normalized_vector() {
    let mut collection = Collection::with_dimensions("test", 3);

    // Unit vector
    let unit_vec = vec![1.0, 0.0, 0.0];
    collection.insert("unit", &unit_vec, None).unwrap();

    let (vec, _) = collection.get("unit").unwrap();
    assert_eq!(vec, &[1.0, 0.0, 0.0]);
}

#[test]
fn test_negative_values() {
    let mut collection = Collection::with_dimensions("test", 4);

    let vec = vec![-1.0, -0.5, 0.0, 0.5];
    collection.insert("neg", &vec, None).unwrap();

    let (retrieved, _) = collection.get("neg").unwrap();
    assert_eq!(retrieved, &[-1.0, -0.5, 0.0, 0.5]);
}

#[test]
fn test_very_small_values() {
    let mut collection = Collection::with_dimensions("test", 4);

    let vec = vec![1e-38, 1e-38, 1e-38, 1e-38];
    collection.insert("tiny", &vec, None).unwrap();

    let (retrieved, _) = collection.get("tiny").unwrap();
    assert!(retrieved.iter().all(|&v| v > 0.0));
}

#[test]
fn test_large_values() {
    let mut collection = Collection::with_dimensions("test", 4);

    let vec = vec![1e30, 1e30, 1e30, 1e30];
    collection.insert("huge", &vec, None).unwrap();

    let (retrieved, _) = collection.get("huge").unwrap();
    assert!(retrieved.iter().all(|&v| v > 1e29));
}

#[test]
fn test_nan_rejection() {
    let mut collection = Collection::with_dimensions("test", 4);

    let vec = vec![1.0, f32::NAN, 1.0, 1.0];
    let result = collection.insert("nan", &vec, None);
    assert!(result.is_err());
}

#[test]
fn test_infinity_rejection() {
    let mut collection = Collection::with_dimensions("test", 4);

    let vec = vec![1.0, f32::INFINITY, 1.0, 1.0];
    let result = collection.insert("inf", &vec, None);
    assert!(result.is_err());
}

#[test]
fn test_neg_infinity_rejection() {
    let mut collection = Collection::with_dimensions("test", 4);

    let vec = vec![1.0, f32::NEG_INFINITY, 1.0, 1.0];
    let result = collection.insert("neg_inf", &vec, None);
    assert!(result.is_err());
}

// ============================================================================
// ID Edge Cases
// ============================================================================

#[test]
fn test_duplicate_id_rejection() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("dup", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    let result = collection.insert("dup", &[5.0, 6.0, 7.0, 8.0], None);
    assert!(result.is_err());
}

#[test]
fn test_empty_string_id() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Empty string ID - should be allowed
    collection.insert("", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    assert!(collection.contains(""));
}

#[test]
fn test_unicode_id() {
    let mut collection = Collection::with_dimensions("test", 4);

    let unicode_id = "日本語テスト🎉";
    collection.insert(unicode_id, &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    assert!(collection.contains(unicode_id));

    let (vec, _) = collection.get(unicode_id).unwrap();
    assert_eq!(vec, &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_very_long_id() {
    let mut collection = Collection::with_dimensions("test", 4);

    let long_id: String = "x".repeat(10000);
    collection.insert(&long_id, &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    assert!(collection.contains(&long_id));
}

#[test]
fn test_special_characters_in_id() {
    let mut collection = Collection::with_dimensions("test", 4);

    let special_ids = vec![
        "path/to/file",
        "id with spaces",
        "id\twith\ttabs",
        "id\nwith\nnewlines",
        "id\"with\"quotes",
        "id'with'quotes",
        "id<with>brackets",
        "id&with&ampersand",
    ];

    for (i, id) in special_ids.iter().enumerate() {
        let vec = vec![i as f32, 0.0, 0.0, 0.0];
        collection.insert(*id, &vec, None).unwrap();
        assert!(collection.contains(*id));
    }
}

// ============================================================================
// Metadata Edge Cases
// ============================================================================

#[test]
fn test_null_metadata() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    let (_, meta) = collection.get("v1").unwrap();
    assert!(meta.is_none());
}

#[test]
fn test_empty_object_metadata() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({}))).unwrap();
    let (_, meta) = collection.get("v1").unwrap();
    assert_eq!(meta, Some(&json!({})));
}

#[test]
fn test_nested_metadata() {
    let mut collection = Collection::with_dimensions("test", 4);

    let nested = json!({
        "level1": {
            "level2": {
                "level3": {
                    "value": 42
                }
            }
        }
    });

    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(nested.clone())).unwrap();
    let (_, meta) = collection.get("v1").unwrap();
    assert_eq!(meta, Some(&nested));
}

#[test]
fn test_array_metadata() {
    let mut collection = Collection::with_dimensions("test", 4);

    let array = json!(["a", "b", "c", 1, 2, 3]);
    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(array.clone())).unwrap();
    let (_, meta) = collection.get("v1").unwrap();
    assert_eq!(meta, Some(&array));
}

#[test]
fn test_large_metadata() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Create large metadata object
    let large_text: String = "x".repeat(100000);
    let meta = json!({
        "large_field": large_text
    });

    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(meta)).unwrap();
    let (_, retrieved_meta) = collection.get("v1").unwrap();
    assert!(retrieved_meta.is_some());
}

// ============================================================================
// Search Edge Cases
// ============================================================================

#[test]
fn test_search_k_larger_than_collection() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

    // Ask for more results than exist
    let results = collection.search(&[0.5, 0.5, 0.0, 0.0], 100).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_search_k_zero() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let results = collection.search(&[1.0, 0.0, 0.0, 0.0], 0).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_after_delete() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
    collection.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();

    collection.delete("v2").unwrap();

    let results = collection.search(&[0.0, 1.0, 0.0, 0.0], 10).unwrap();
    assert!(!results.iter().any(|r| r.id == "v2"));
}

// ============================================================================
// Distance Function Tests
// ============================================================================

#[test]
fn test_all_distance_functions() {
    for dist in [
        DistanceFunction::Cosine,
        DistanceFunction::Euclidean,
        DistanceFunction::DotProduct,
        DistanceFunction::Manhattan,
    ] {
        let config = CollectionConfig::new("test", 4).with_distance(dist);
        let mut collection = Collection::new(config);

        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let results = collection.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);

        // First result should be most similar to query
        if dist != DistanceFunction::DotProduct {
            // For dot product, "smaller is more similar" is inverted
            assert_eq!(results[0].id, "v1");
        }
    }
}

// ============================================================================
// Batch Operations Edge Cases
// ============================================================================

#[test]
fn test_empty_batch_insert() {
    let mut collection = Collection::with_dimensions("test", 4);

    let result = collection.insert_batch(vec![], vec![], vec![]);
    assert!(result.is_ok());
    assert_eq!(collection.len(), 0);
}

#[test]
fn test_batch_insert_mismatched_lengths() {
    let mut collection = Collection::with_dimensions("test", 4);

    let ids = vec!["v1".to_string(), "v2".to_string()];
    let vectors = vec![vec![1.0, 0.0, 0.0, 0.0]]; // Only one vector
    let metadata = vec![None, None];

    let result = collection.insert_batch(ids, vectors, metadata);
    assert!(result.is_err());
}

#[test]
fn test_delete_batch_with_nonexistent() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    // Delete mix of existing and non-existing
    let ids = ["v1", "nonexistent1", "nonexistent2"];
    let deleted = collection.delete_batch(&ids).unwrap();
    assert_eq!(deleted, 1);
}

// ============================================================================
// Database Edge Cases
// ============================================================================

#[test]
fn test_database_create_duplicate_collection() {
    let db = Database::in_memory();

    db.create_collection("test", 128).unwrap();
    let result = db.create_collection("test", 128);
    assert!(result.is_err());
}

#[test]
fn test_database_get_nonexistent_collection() {
    let db = Database::in_memory();

    let result = db.collection("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_database_drop_nonexistent_collection() {
    let db = Database::in_memory();

    let result = db.drop_collection("nonexistent").unwrap();
    assert!(!result);
}

#[test]
fn test_database_list_empty() {
    let db = Database::in_memory();

    let collections = db.list_collections();
    assert!(collections.is_empty());
}

// ============================================================================
// Validation Edge Cases (new tests for Phase 3)
// ============================================================================

#[test]
#[should_panic(expected = "dimensions must be greater than 0")]
fn test_zero_dimension_collection_panics() {
    // Zero dimensions should panic
    CollectionConfig::new("invalid", 0);
}

#[test]
fn test_large_k_is_clamped() {
    let mut collection = Collection::with_dimensions("test", 8);

    // Insert only 5 vectors
    for i in 0..5 {
        let vector: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32) / 40.0).collect();
        collection.insert(format!("vec_{}", i), &vector, None).unwrap();
    }

    // Search with k=1000000 - should be clamped to 5
    let query: Vec<f32> = vec![0.5; 8];
    let results = collection.search(&query, 1_000_000).unwrap();
    assert_eq!(results.len(), 5, "Results should be clamped to collection size");
}

#[test]
fn test_k_zero_returns_empty() {
    let mut collection = Collection::with_dimensions("test", 8);

    for i in 0..5 {
        let vector: Vec<f32> = (0..8).map(|j| ((i * 8 + j) as f32) / 40.0).collect();
        collection.insert(format!("vec_{}", i), &vector, None).unwrap();
    }

    let query: Vec<f32> = vec![0.5; 8];
    let results = collection.search(&query, 0).unwrap();
    assert!(results.is_empty(), "k=0 should return empty results");
}

#[test]
fn test_nan_vector_rejected() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Vector with NaN should be rejected
    let nan_vector = vec![1.0, f32::NAN, 3.0, 4.0];
    let result = collection.insert("nan_vec", &nan_vector, None);
    assert!(result.is_err(), "NaN vector should be rejected");
}

#[test]
fn test_infinity_vector_rejected() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Vector with Infinity should be rejected
    let inf_vector = vec![1.0, f32::INFINITY, 3.0, 4.0];
    let result = collection.insert("inf_vec", &inf_vector, None);
    assert!(result.is_err(), "Infinity vector should be rejected");
}

#[test]
fn test_negative_infinity_vector_rejected() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Vector with -Infinity should be rejected
    let neg_inf_vector = vec![1.0, f32::NEG_INFINITY, 3.0, 4.0];
    let result = collection.insert("neg_inf_vec", &neg_inf_vector, None);
    assert!(result.is_err(), "Negative infinity vector should be rejected");
}

#[test]
fn test_nan_query_rejected() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Insert valid vector
    let valid_vector = vec![1.0, 2.0, 3.0, 4.0];
    collection.insert("valid", &valid_vector, None).unwrap();

    // Query with NaN should be rejected
    let nan_query = vec![1.0, f32::NAN, 3.0, 4.0];
    let result = collection.search(&nan_query, 5);
    assert!(result.is_err(), "NaN query should be rejected");
}

#[test]
fn test_batch_search_large_k_clamped() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Insert 3 vectors
    for i in 0..3 {
        let vector: Vec<f32> = vec![i as f32; 4];
        collection.insert(format!("vec_{}", i), &vector, None).unwrap();
    }

    // Batch search with k > collection size
    let queries = vec![vec![0.0; 4], vec![1.0; 4]];
    let results = collection.batch_search(&queries, 100).unwrap();

    assert_eq!(results.len(), 2);
    for r in &results {
        assert_eq!(r.len(), 3, "Results should be clamped to collection size");
    }
}

#[test]
fn test_filtered_search_large_k_clamped() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);

    // Insert vectors with category metadata
    for i in 0..10 {
        let vector: Vec<f32> = vec![i as f32 / 10.0; 4];
        let category = if i % 2 == 0 { "even" } else { "odd" };
        collection.insert(
            format!("vec_{}", i),
            &vector,
            Some(json!({"category": category})),
        ).unwrap();
    }

    // Filtered search with very large k
    let query: Vec<f32> = vec![0.5; 4];
    let filter = Filter::eq("category", "even");
    let results = collection.search_with_filter(&query, 1_000_000, &filter).unwrap();

    // Should only return even vectors (5 of them)
    assert!(results.len() <= 5, "Should return at most 5 even vectors");
    for r in &results {
        let meta = r.metadata.as_ref().unwrap();
        assert_eq!(meta["category"], "even");
    }
}

#[test]
#[should_panic(expected = "not finite")]
fn test_sparse_vector_nan_rejected() {
    use needle::SparseVector;

    // Sparse vector with NaN should panic
    SparseVector::new(vec![0, 1, 2], vec![1.0, f32::NAN, 3.0]);
}

#[test]
#[should_panic(expected = "not finite")]
fn test_sparse_vector_inf_rejected() {
    use needle::SparseVector;

    // Sparse vector with Infinity should panic
    SparseVector::new(vec![0, 1, 2], vec![1.0, f32::INFINITY, 3.0]);
}
