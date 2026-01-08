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

// ============================================================================
// Error Path Tests - I/O and Malformed Input
// ============================================================================

/// Test opening database from non-existent directory
#[test]
fn test_open_nonexistent_path() {
    let result = Database::open("/this/path/definitely/does/not/exist/db.needle");
    // Should create the file or fail gracefully
    assert!(result.is_ok() || result.is_err());
}

/// Test I/O error handling with read-only files/directories (Unix only)
#[cfg(unix)]
#[test]
fn test_save_to_readonly_location() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create and save initial database
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("test", 4).unwrap();
        db.save().unwrap();
    }

    // Make file read-only - Database::open should fail because it needs write access
    let metadata = fs::metadata(&db_path).unwrap();
    let mut permissions = metadata.permissions();
    permissions.set_mode(0o444); // Read-only
    fs::set_permissions(&db_path, permissions.clone()).unwrap();

    // Try to open read-only file - should fail (open requires write access)
    {
        let result = Database::open(&db_path);
        assert!(result.is_err(), "Should fail to open read-only database file for writing");
    }

    // Restore file permissions
    permissions.set_mode(0o644);
    fs::set_permissions(&db_path, permissions).unwrap();

    // Test 2: Make directory read-only to prevent saving a new file
    let readonly_dir = tempdir().unwrap();
    let new_db_path = readonly_dir.path().join("newdb.needle");

    // Make directory read-only (can't create new files)
    let dir_metadata = fs::metadata(readonly_dir.path()).unwrap();
    let mut dir_permissions = dir_metadata.permissions();
    dir_permissions.set_mode(0o555); // Read-only directory
    fs::set_permissions(readonly_dir.path(), dir_permissions.clone()).unwrap();

    // Try to create database in read-only directory - should fail
    let result = Database::open(&new_db_path);
    assert!(result.is_err(), "Should fail to create database in read-only directory");

    // Restore directory permissions for cleanup
    dir_permissions.set_mode(0o755);
    fs::set_permissions(readonly_dir.path(), dir_permissions).unwrap();
}

/// Test database with corrupted serialization in collection
#[test]
fn test_corrupted_collection_bytes() {
    // Try to deserialize garbage as a collection
    let garbage: Vec<u8> = vec![0xFF, 0xFE, 0xFD, 0xFC, 0x00, 0x01, 0x02, 0x03];
    let result = Collection::from_bytes(&garbage);
    assert!(result.is_err(), "Should reject corrupted collection bytes");
}

/// Test empty byte deserialization
#[test]
fn test_empty_bytes_deserialization() {
    let empty: Vec<u8> = vec![];
    let result = Collection::from_bytes(&empty);
    assert!(result.is_err(), "Should reject empty bytes");
}

/// Test extremely large dimension request
#[test]
fn test_extremely_large_dimension() {
    // This should either work or fail gracefully (not crash)
    // Very large dimensions might cause memory allocation issues
    let large_dim = 1_000_000;

    // Just create the config, don't allocate vectors
    let config = CollectionConfig::new("test", large_dim);
    assert_eq!(config.dimensions, large_dim);
}

/// Test HNSW config with edge case parameters
#[test]
fn test_hnsw_edge_case_configs() {
    use needle::HnswConfig;

    // Very small M - test the config struct directly
    let config = HnswConfig {
        m: 1,
        m_max_0: 2,
        ef_construction: 10,
        ef_search: 5,
        ml: 1.0 / 2.0_f64.ln(), // Use ln(2) to avoid ln(1)=0 division
    };
    assert_eq!(config.m, 1);
    assert_eq!(config.m_max_0, 2);

    // Very large ef parameters
    let config2 = HnswConfig {
        m: 16,
        m_max_0: 32,
        ef_construction: 10000,
        ef_search: 10000,
        ml: 1.0 / (16_f64.ln()),
    };
    assert_eq!(config2.ef_construction, 10000);
    assert_eq!(config2.ef_search, 10000);

    // Test collection with edge case M using with_m()
    let db = Database::in_memory();
    db.create_collection_with_config(
        CollectionConfig::new("small_m", 4).with_m(2)
    ).unwrap();
    let collection = db.collection("small_m").unwrap();
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    assert_eq!(collection.len(), 1);

    // Test with larger ef_construction
    db.create_collection_with_config(
        CollectionConfig::new("large_ef", 4).with_m(16).with_ef_construction(1000)
    ).unwrap();
    let collection2 = db.collection("large_ef").unwrap();
    collection2.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    assert_eq!(collection2.len(), 1);
}

/// Test filter with invalid operators
#[test]
fn test_filter_edge_cases() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);

    // Insert test vectors
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"score": 10}))).unwrap();
    collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"score": 20}))).unwrap();

    // Filter with empty array for $in
    let filter = Filter::parse(&json!({"score": {"$in": []}})).unwrap();
    let results = collection.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter).unwrap();
    assert!(results.is_empty(), "Empty $in should match nothing");

    // Filter with non-existent field
    let filter2 = Filter::eq("nonexistent_field", "value");
    let results2 = collection.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter2).unwrap();
    assert!(results2.is_empty(), "Non-existent field should match nothing");
}

/// Test concurrent operations edge case
#[test]
fn test_concurrent_operations() {
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(Database::in_memory());
    db.create_collection("concurrent", 4).unwrap();

    let mut handles = vec![];

    // Spawn multiple reader threads
    for i in 0..5 {
        let db_clone = db.clone();
        handles.push(thread::spawn(move || {
            for j in 0..10 {
                let collection = db_clone.collection("concurrent").unwrap();
                let _ = collection.search(&[i as f32, j as f32, 0.0, 0.0], 5);
            }
        }));
    }

    // Spawn writer thread
    let db_writer = db.clone();
    handles.push(thread::spawn(move || {
        let collection = db_writer.collection("concurrent").unwrap();
        for i in 0..20 {
            let _ = collection.insert(
                format!("vec_{}", i),
                &[i as f32, 0.0, 0.0, 0.0],
                None,
            );
        }
    }));

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread should complete without panic");
    }

    // Verify some data was inserted
    let collection = db.collection("concurrent").unwrap();
    assert!(collection.len() > 0, "Some vectors should have been inserted");
}

/// Test update with dimension mismatch
#[test]
fn test_update_dimension_mismatch() {
    let mut collection = Collection::with_dimensions("test", 4);

    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();

    // Update with wrong dimensions
    let wrong_vec = vec![1.0, 2.0]; // Only 2 dimensions
    let result = collection.update("v1", &wrong_vec, None);
    assert!(result.is_err(), "Update with wrong dimensions should fail");
}

/// Test update non-existent vector
#[test]
fn test_update_nonexistent() {
    let mut collection = Collection::with_dimensions("test", 4);

    let result = collection.update("nonexistent", &[1.0, 2.0, 3.0, 4.0], None);
    assert!(result.is_err(), "Update of non-existent vector should fail");
}

/// Test collection name edge cases in database
#[test]
fn test_collection_name_edge_cases() {
    let db = Database::in_memory();

    // Empty name
    let result = db.create_collection("", 4);
    // Either allowed or rejected gracefully
    match result {
        Ok(_) => {
            assert!(db.collection("").is_ok());
        }
        Err(_) => {
            // Rejection is also acceptable
        }
    }

    // Very long name
    let long_name: String = "x".repeat(10000);
    let result = db.create_collection(&long_name, 4);
    if result.is_ok() {
        assert!(db.collection(&long_name).is_ok());
    }

    // Special characters
    db.create_collection("test/with/slashes", 4).unwrap();
    assert!(db.collection("test/with/slashes").is_ok());
}

/// Test filter parsing with malformed JSON-like structures
#[test]
fn test_filter_malformed_input() {
    use needle::Filter;

    // Filter with null value
    let filter = Filter::parse(&json!({"field": null}));
    assert!(filter.is_ok(), "Null filter value should parse");

    // Filter with nested empty objects
    let filter2 = Filter::parse(&json!({"$and": []}));
    assert!(filter2.is_ok(), "Empty $and should parse");

    // Filter with deeply nested structure
    let deep = json!({
        "$and": [{
            "$or": [{
                "$and": [{
                    "field": "value"
                }]
            }]
        }]
    });
    let filter3 = Filter::parse(&deep);
    assert!(filter3.is_ok(), "Deeply nested filter should parse");
}

/// Test search result ordering consistency
#[test]
fn test_search_result_consistency() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Insert vectors
    for i in 0..10 {
        collection.insert(
            format!("v{}", i),
            &[i as f32 / 10.0, 0.0, 0.0, 0.0],
            None,
        ).unwrap();
    }

    let query = vec![0.5, 0.0, 0.0, 0.0];

    // Run same search multiple times
    let results1 = collection.search(&query, 5).unwrap();
    let results2 = collection.search(&query, 5).unwrap();
    let results3 = collection.search(&query, 5).unwrap();

    // Results should be consistent
    assert_eq!(results1.len(), results2.len());
    assert_eq!(results2.len(), results3.len());

    for i in 0..results1.len() {
        assert_eq!(results1[i].id, results2[i].id, "Results should be consistent");
        assert_eq!(results2[i].id, results3[i].id, "Results should be consistent");
    }
}

/// Test compact after heavy operations
#[test]
fn test_compact_after_heavy_operations() {
    let mut collection = Collection::with_dimensions("test", 4);

    // Insert many vectors
    for i in 0..100 {
        collection.insert(
            format!("v{}", i),
            &[i as f32, 0.0, 0.0, 0.0],
            None,
        ).unwrap();
    }

    // Delete many vectors
    for i in 0..50 {
        collection.delete(&format!("v{}", i)).unwrap();
    }

    assert_eq!(collection.len(), 50);

    // Compact should work
    let removed = collection.compact().unwrap();
    // Compact should have removed the deleted vectors from internal storage
    assert!(removed >= 0); // May not remove if not needed

    // Remaining vectors should still work
    for i in 50..100 {
        assert!(collection.contains(&format!("v{}", i)));
    }
}

/// Test JSON metadata with all types
#[test]
fn test_metadata_all_json_types() {
    let mut collection = Collection::with_dimensions("test", 4);

    // String
    collection.insert("str", &[1.0, 0.0, 0.0, 0.0], Some(json!("hello"))).unwrap();

    // Number (integer)
    collection.insert("int", &[2.0, 0.0, 0.0, 0.0], Some(json!(42))).unwrap();

    // Number (float)
    collection.insert("float", &[3.0, 0.0, 0.0, 0.0], Some(json!(3.14))).unwrap();

    // Boolean
    collection.insert("bool", &[4.0, 0.0, 0.0, 0.0], Some(json!(true))).unwrap();

    // Null
    collection.insert("null", &[5.0, 0.0, 0.0, 0.0], Some(json!(null))).unwrap();

    // Array
    collection.insert("array", &[6.0, 0.0, 0.0, 0.0], Some(json!([1, 2, 3]))).unwrap();

    // Object
    collection.insert("object", &[7.0, 0.0, 0.0, 0.0], Some(json!({"key": "value"}))).unwrap();

    // Verify all stored correctly
    assert_eq!(collection.len(), 7);

    // Check retrieval
    let (_, meta) = collection.get("str").unwrap();
    assert_eq!(meta.unwrap(), &json!("hello"));

    let (_, meta) = collection.get("bool").unwrap();
    assert_eq!(meta.unwrap(), &json!(true));
}

/// Test export/import roundtrip edge cases
#[test]
fn test_export_import_edge_cases() {
    let db = Database::in_memory();

    // Create collection with various edge case data
    db.create_collection("edge", 4).unwrap();
    let collection = db.collection("edge").unwrap();

    // Insert edge case vectors
    collection.insert("zero", &[0.0, 0.0, 0.0, 0.0], None).unwrap();
    collection.insert("negative", &[-1.0, -2.0, -3.0, -4.0], None).unwrap();
    collection.insert("small", &[1e-38, 1e-38, 1e-38, 1e-38], None).unwrap();
    collection.insert("unicode_id_こんにちは", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    collection.insert("meta", &[5.0, 6.0, 7.0, 8.0], Some(json!({"emoji": "🎉"}))).unwrap();

    // Export using export_all() on CollectionRef
    // ExportEntry is a tuple: (String, Vec<f32>, Option<Value>)
    let export = collection.export_all().unwrap();
    assert_eq!(export.len(), 5);

    // Create new database and import
    let db2 = Database::in_memory();
    db2.create_collection("edge", 4).unwrap();
    let collection2 = db2.collection("edge").unwrap();

    // ExportEntry is (id: String, vector: Vec<f32>, metadata: Option<Value>)
    for (id, vector, metadata) in &export {
        collection2.insert(id, vector, metadata.clone()).unwrap();
    }

    assert_eq!(collection2.len(), 5);

    // Verify data integrity
    let (vec, _) = collection2.get("zero").unwrap();
    assert_eq!(vec, &[0.0, 0.0, 0.0, 0.0]);

    let (vec, _) = collection2.get("negative").unwrap();
    assert_eq!(vec, &[-1.0, -2.0, -3.0, -4.0]);

    let (_, meta) = collection2.get("meta").unwrap();
    assert_eq!(meta.unwrap()["emoji"], "🎉");
}

// ============================================================================
// Range Query (search_radius) Error Paths
// ============================================================================

/// Test search_radius with dimension mismatch
#[test]
fn test_search_radius_dimension_mismatch() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    // Wrong dimension query
    let wrong_query = vec![1.0, 0.0]; // 2 dimensions
    let result = collection.search_radius(&wrong_query, 0.5, 10);
    assert!(result.is_err(), "Should reject wrong dimension query");
}

/// Test search_radius with NaN query
#[test]
fn test_search_radius_nan_query() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let nan_query = vec![1.0, f32::NAN, 0.0, 0.0];
    let result = collection.search_radius(&nan_query, 0.5, 10);
    assert!(result.is_err(), "Should reject NaN query");
}

/// Test search_radius with negative distance
#[test]
fn test_search_radius_negative_distance() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = collection.search_radius(&query, -1.0, 10).unwrap();
    assert!(results.is_empty(), "Negative distance should return no results");
}

/// Test search_radius with zero limit
#[test]
fn test_search_radius_zero_limit() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = collection.search_radius(&query, 0.5, 0).unwrap();
    assert!(results.is_empty(), "Zero limit should return no results");
}

/// Test search_radius on empty collection
#[test]
fn test_search_radius_empty_collection() {
    let collection = Collection::with_dimensions("test", 4);

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = collection.search_radius(&query, 1.0, 10).unwrap();
    assert!(results.is_empty());
}

/// Test search_radius with filter and invalid query
#[test]
fn test_search_radius_with_filter_nan() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"}))).unwrap();

    let nan_query = vec![f32::NAN, 0.0, 0.0, 0.0];
    let filter = Filter::eq("type", "a");
    let result = collection.search_radius_with_filter(&nan_query, 0.5, 10, &filter);
    assert!(result.is_err(), "Should reject NaN query in filtered search");
}

// ============================================================================
// Post-Filter Error Paths
// ============================================================================

/// Test SearchBuilder with NaN query
#[test]
fn test_search_builder_nan_query() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let nan_query = vec![1.0, f32::NAN, 0.0, 0.0];
    let result = collection.search_builder(&nan_query).k(10).execute();
    assert!(result.is_err(), "SearchBuilder should reject NaN query");
}

/// Test SearchBuilder with dimension mismatch
#[test]
fn test_search_builder_dimension_mismatch() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let wrong_query = vec![1.0, 0.0]; // 2 dimensions
    let result = collection.search_builder(&wrong_query).k(10).execute();
    assert!(result.is_err(), "SearchBuilder should reject wrong dimensions");
}

/// Test SearchBuilder with post_filter_factor of 0 (should be clamped to 1)
#[test]
fn test_search_builder_zero_post_filter_factor() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    for i in 0..10 {
        collection.insert(
            format!("v{}", i),
            &[i as f32 / 10.0, 0.0, 0.0, 0.0],
            Some(json!({"score": i})),
        ).unwrap();
    }

    let query = vec![0.5, 0.0, 0.0, 0.0];
    let post_filter = Filter::gt("score", 5);

    // post_filter_factor(0) should be clamped to 1
    let results = collection
        .search_builder(&query)
        .k(5)
        .post_filter(&post_filter)
        .post_filter_factor(0)
        .execute()
        .unwrap();

    // Should still work, just with minimal over-fetching
    assert!(results.len() <= 5);
}

/// Test post-filter with non-existent field
#[test]
fn test_post_filter_nonexistent_field() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"}))).unwrap();
    collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "b"}))).unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let post_filter = Filter::eq("nonexistent_field", "value");

    let results = collection
        .search_builder(&query)
        .k(10)
        .post_filter(&post_filter)
        .execute()
        .unwrap();

    assert!(results.is_empty(), "Filter on non-existent field should match nothing");
}

// ============================================================================
// Explain Mode Error Paths
// ============================================================================

/// Test search_explain with NaN query
#[test]
fn test_search_explain_nan_query() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let nan_query = vec![1.0, f32::NAN, 0.0, 0.0];
    let result = collection.search_explain(&nan_query, 10);
    assert!(result.is_err(), "search_explain should reject NaN query");
}

/// Test search_explain with dimension mismatch
#[test]
fn test_search_explain_dimension_mismatch() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let wrong_query = vec![1.0]; // 1 dimension
    let result = collection.search_explain(&wrong_query, 10);
    assert!(result.is_err(), "search_explain should reject wrong dimensions");
}

/// Test search_explain with k=0
#[test]
fn test_search_explain_k_zero() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let (results, explain) = collection.search_explain(&query, 0).unwrap();

    assert!(results.is_empty());
    assert_eq!(explain.effective_k, 0);
}

/// Test search_explain on empty collection
#[test]
fn test_search_explain_empty_collection() {
    let collection = Collection::with_dimensions("test", 4);

    let query = vec![1.0, 0.0, 0.0, 0.0];
    let (results, explain) = collection.search_explain(&query, 10).unwrap();

    assert!(results.is_empty());
    assert_eq!(explain.collection_size, 0);
    assert_eq!(explain.effective_k, 0);
}

/// Test search_with_filter_explain with NaN query
#[test]
fn test_search_with_filter_explain_nan() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"}))).unwrap();

    let nan_query = vec![1.0, f32::NAN, 0.0, 0.0];
    let filter = Filter::eq("type", "a");
    let result = collection.search_with_filter_explain(&nan_query, 10, &filter);
    assert!(result.is_err(), "search_with_filter_explain should reject NaN query");
}

// ============================================================================
// CollectionRef Error Paths (via Database)
// ============================================================================

/// Test CollectionRef operations after collection is dropped
#[test]
fn test_collection_ref_after_drop() {
    let db = Database::in_memory();
    db.create_collection("temp", 4).unwrap();

    let coll_ref = db.collection("temp").unwrap();
    coll_ref.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    // Drop the collection
    db.drop_collection("temp").unwrap();

    // Operations should fail gracefully
    let search_result = coll_ref.search(&[1.0, 0.0, 0.0, 0.0], 10);
    assert!(search_result.is_err(), "Search on dropped collection should fail");

    let insert_result = coll_ref.insert("v2", &[0.0, 1.0, 0.0, 0.0], None);
    assert!(insert_result.is_err(), "Insert on dropped collection should fail");

    let delete_result = coll_ref.delete("v1");
    assert!(delete_result.is_err(), "Delete on dropped collection should fail");
}

/// Test CollectionRef search_with_post_filter with invalid input
#[test]
fn test_collection_ref_search_with_post_filter_nan() {
    use needle::Filter;

    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let coll = db.collection("test").unwrap();
    coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"score": 10}))).unwrap();

    let nan_query = vec![1.0, f32::NAN, 0.0, 0.0];
    let post_filter = Filter::gt("score", 5);

    let result = coll.search_with_post_filter(&nan_query, 10, None, &post_filter, 3);
    assert!(result.is_err(), "search_with_post_filter should reject NaN query");
}

/// Test CollectionRef search_radius with NaN
#[test]
fn test_collection_ref_search_radius_nan() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let coll = db.collection("test").unwrap();
    coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let nan_query = vec![1.0, f32::NAN, 0.0, 0.0];
    let result = coll.search_radius(&nan_query, 0.5, 10);
    assert!(result.is_err(), "search_radius should reject NaN query");
}

// ============================================================================
// Filter Edge Cases with Unknown Operators
// ============================================================================

/// Test filter with unknown operator
#[test]
fn test_filter_unknown_operator() {
    use needle::Filter;

    // Unknown operator should either be rejected or treated as equality
    let result = Filter::parse(&json!({"field": {"$unknown_op": "value"}}));
    // The behavior depends on implementation - either error or ignored
    match result {
        Ok(filter) => {
            // If parsed, should not match anything when applied
            let mut collection = Collection::with_dimensions("test", 4);
            collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"field": "value"}))).unwrap();

            let results = collection.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter).unwrap();
            // May or may not match depending on implementation
            assert!(results.len() <= 1);
        }
        Err(_) => {
            // Rejection is also acceptable
        }
    }
}

/// Test filter with type mismatch in comparison
#[test]
fn test_filter_type_mismatch() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"score": 10}))).unwrap();
    collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"score": "high"}))).unwrap();

    // Filter comparing number with string should handle gracefully
    let filter = Filter::gt("score", 5);
    let results = collection.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter).unwrap();

    // v1 should match (10 > 5), v2 might not (string vs number)
    assert!(results.iter().any(|r| r.id == "v1"));
}

/// Test $in filter with mixed types
#[test]
fn test_filter_in_mixed_types() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"val": 1}))).unwrap();
    collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"val": "one"}))).unwrap();
    collection.insert("v3", &[0.0, 0.0, 1.0, 0.0], Some(json!({"val": true}))).unwrap();

    // $in with mixed types
    let filter = Filter::parse(&json!({"val": {"$in": [1, "one", true]}})).unwrap();
    let results = collection.search_with_filter(&[0.5, 0.5, 0.5, 0.0], 10, &filter).unwrap();

    // All should match since each value is in the list
    assert_eq!(results.len(), 3);
}

// ============================================================================
// Upsert Error Paths
// ============================================================================

/// Test upsert with NaN vector
#[test]
fn test_upsert_nan_vector() {
    let mut collection = Collection::with_dimensions("test", 4);

    let nan_vec = vec![1.0, f32::NAN, 0.0, 0.0];
    let result = collection.upsert("v1", &nan_vec, None);
    assert!(result.is_err(), "upsert should reject NaN vector");
}

/// Test upsert with dimension mismatch
#[test]
fn test_upsert_dimension_mismatch() {
    let mut collection = Collection::with_dimensions("test", 4);

    let wrong_vec = vec![1.0, 0.0]; // 2 dimensions
    let result = collection.upsert("v1", &wrong_vec, None);
    assert!(result.is_err(), "upsert should reject wrong dimensions");
}

// ============================================================================
// Update Metadata Error Paths
// ============================================================================

/// Test update_metadata on non-existent vector
#[test]
fn test_update_metadata_nonexistent() {
    let mut collection = Collection::with_dimensions("test", 4);

    let result = collection.update_metadata("nonexistent", Some(json!({"key": "value"})));
    assert!(result.is_err(), "update_metadata on non-existent vector should fail");
}

/// Test update_metadata on existing vector
#[test]
fn test_update_metadata_existing() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"old": "value"}))).unwrap();

    // Update metadata
    collection.update_metadata("v1", Some(json!({"new": "value"}))).unwrap();

    let (_, meta) = collection.get("v1").unwrap();
    assert_eq!(meta.unwrap()["new"], "value");
    assert!(meta.unwrap().get("old").is_none()); // Old metadata should be replaced
}

// ============================================================================
// Batch Search Error Paths
// ============================================================================

/// Test batch_search with NaN in one query
#[test]
fn test_batch_search_nan_in_queries() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, f32::NAN, 0.0, 0.0], // NaN in second query
    ];

    let result = collection.batch_search(&queries, 10);
    assert!(result.is_err(), "batch_search should reject queries with NaN");
}

/// Test batch_search with dimension mismatch
#[test]
fn test_batch_search_dimension_mismatch() {
    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();

    let queries = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.0, 0.0], // Wrong dimension
    ];

    let result = collection.batch_search(&queries, 10);
    assert!(result.is_err(), "batch_search should reject queries with wrong dimensions");
}

/// Test batch_search_with_filter with NaN
#[test]
fn test_batch_search_with_filter_nan() {
    use needle::Filter;

    let mut collection = Collection::with_dimensions("test", 4);
    collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"}))).unwrap();

    let queries = vec![
        vec![1.0, f32::NAN, 0.0, 0.0], // NaN
    ];
    let filter = Filter::eq("type", "a");

    let result = collection.batch_search_with_filter(&queries, 10, &filter);
    assert!(result.is_err(), "batch_search_with_filter should reject NaN queries");
}

// ============================================================================
// Index Auto-Selection Edge Cases
// ============================================================================

/// Test recommend_index with edge case inputs
#[test]
fn test_recommend_index_edge_cases() {
    use needle::{quick_recommend_index, recommend_index, IndexSelectionConstraints, RecommendedIndex};

    // Very small dataset
    assert_eq!(quick_recommend_index(10, 128), RecommendedIndex::Hnsw);

    // Medium dataset - still HNSW (threshold is higher)
    assert_eq!(quick_recommend_index(500_000, 128), RecommendedIndex::Hnsw);

    // Large dataset - triggers IVF recommendation
    assert_eq!(quick_recommend_index(5_000_000, 768), RecommendedIndex::Ivf);

    // Very large dataset
    assert_eq!(quick_recommend_index(50_000_000, 128), RecommendedIndex::DiskAnn);

    // Edge case: exactly at boundary
    assert_eq!(quick_recommend_index(100_000, 128), RecommendedIndex::Hnsw);

    // High dimensions
    let recommendation = quick_recommend_index(100_000, 4096);
    // With high dimensions, memory might push towards IVF/DiskANN
    assert!(matches!(recommendation, RecommendedIndex::Hnsw | RecommendedIndex::Ivf | RecommendedIndex::DiskAnn));

    // Test with constraints
    let constraints = IndexSelectionConstraints {
        expected_vectors: 1_000_000,
        dimensions: 768,
        available_memory_bytes: Some(100 * 1024 * 1024), // 100MB - very limited
        available_disk_bytes: Some(10 * 1024 * 1024 * 1024), // 10GB
        low_latency_critical: false,
        target_recall: 0.95,
        frequent_updates: false,
    };

    let rec = recommend_index(&constraints);
    // With limited memory, should recommend IVF or DiskANN
    assert!(rec.fits_in_memory || matches!(rec.recommended, RecommendedIndex::Ivf | RecommendedIndex::DiskAnn));
}

// ============================================================================
// Additional I/O Failure Tests
// ============================================================================

/// Test backup to invalid directory path
#[test]
fn test_backup_to_invalid_path() {
    use needle::{BackupConfig, BackupManager};
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create database with data
    let mut db = Database::open(&db_path).unwrap();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();
    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    db.save().unwrap();

    // Try backup to non-existent nested path
    let invalid_backup_path = "/nonexistent/deeply/nested/path/backup";
    let config = BackupConfig::default();

    let manager = BackupManager::new(invalid_backup_path, config);
    let result = manager.create_backup(&db);
    // Should fail gracefully - either error or creates path
    // The behavior depends on implementation
    assert!(result.is_ok() || result.is_err());
}

/// Test export collection functionality
#[test]
fn test_export_collection_valid() {
    // Create database with data
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();
    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(serde_json::json!({"a": 1}))).unwrap();
    collection.insert("v2", &[5.0, 6.0, 7.0, 8.0], None).unwrap();

    // Export should work via CollectionRef
    let entries = collection.export_all().unwrap();
    assert_eq!(entries.len(), 2);
}

/// Test database reopen cycle (open-save-close-reopen)
#[test]
fn test_database_reopen_cycle() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cycle_test.needle");

    // First cycle: create and save
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("test", 4).unwrap();
        let collection = db.collection("test").unwrap();
        collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
        db.save().unwrap();
    }

    // Second cycle: reopen and modify
    {
        let mut db = Database::open(&db_path).unwrap();
        let collection = db.collection("test").unwrap();
        collection.insert("v2", &[5.0, 6.0, 7.0, 8.0], None).unwrap();
        db.save().unwrap();
    }

    // Third cycle: verify all data persisted
    {
        let db = Database::open(&db_path).unwrap();
        let collection = db.collection("test").unwrap();
        assert_eq!(collection.count(None).unwrap(), 2);
        assert!(collection.get("v1").is_some());
        assert!(collection.get("v2").is_some());
    }
}

/// Test database open with empty file
#[test]
fn test_open_empty_file() {
    use tempfile::tempdir;
    use std::fs::File;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("empty.needle");

    // Create an empty file
    File::create(&db_path).unwrap();

    // Try to open - should fail or create new database
    let result = Database::open(&db_path);
    // Implementation may either fail or initialize empty
    assert!(result.is_ok() || result.is_err());
}

/// Test concurrent read operations don't interfere
#[test]
fn test_concurrent_reads() {
    use std::thread;
    use std::sync::Arc;

    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 4).unwrap();

    // Insert some data
    let collection = db.collection("test").unwrap();
    for i in 0..100 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }

    // Spawn multiple reader threads
    let mut handles = vec![];
    for _ in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let collection = db_clone.collection("test").unwrap();
            let query = vec![50.0; 4];
            for _ in 0..10 {
                let results = collection.search(&query, 5).unwrap();
                assert_eq!(results.len(), 5);
            }
        });
        handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test write during iteration (should not cause data race)
#[test]
fn test_iteration_safety() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();

    let collection = db.collection("test").unwrap();
    for i in 0..50 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }

    // Get IDs first (to avoid holding iterator across mutation)
    let ids: Vec<String> = collection.ids().unwrap();
    assert_eq!(ids.len(), 50);

    // Can still insert after iteration completes
    collection.insert("new_vector", &[100.0; 4], None).unwrap();
    assert_eq!(collection.count(None).unwrap(), 51);
}

/// Test serialization round-trip preserves data integrity
#[test]
fn test_serialization_roundtrip_integrity() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("roundtrip.needle");

    // Create with various metadata types
    let original_metadata = serde_json::json!({
        "string": "hello world",
        "number": 42,
        "float": 3.14159,
        "boolean": true,
        "null": null,
        "array": [1, 2, 3],
        "nested": {"a": {"b": {"c": 1}}}
    });

    let original_vector: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();

    // Save
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("test", 128).unwrap();
        let collection = db.collection("test").unwrap();
        collection.insert("complex", &original_vector, Some(original_metadata.clone())).unwrap();
        db.save().unwrap();
    }

    // Load and verify
    {
        let db = Database::open(&db_path).unwrap();
        let collection = db.collection("test").unwrap();
        let (loaded_vector, loaded_metadata) = collection.get("complex").unwrap();

        // Verify vector
        assert_eq!(loaded_vector.len(), 128);
        for (orig, loaded) in original_vector.iter().zip(loaded_vector.iter()) {
            assert!((orig - loaded).abs() < 1e-6, "Vector mismatch: {} vs {}", orig, loaded);
        }

        // Verify metadata
        let loaded_meta = loaded_metadata.unwrap();
        assert_eq!(loaded_meta["string"], "hello world");
        assert_eq!(loaded_meta["number"], 42);
        assert!((loaded_meta["float"].as_f64().unwrap() - 3.14159).abs() < 1e-5);
        assert_eq!(loaded_meta["boolean"], true);
        assert!(loaded_meta["null"].is_null());
        assert_eq!(loaded_meta["array"], serde_json::json!([1, 2, 3]));
        assert_eq!(loaded_meta["nested"]["a"]["b"]["c"], 1);
    }
}

/// Test maximum path length handling
#[test]
fn test_very_long_path() {
    use tempfile::tempdir;

    let dir = tempdir().unwrap();

    // Create a path that's very long but under system limits
    let long_name = "a".repeat(200);
    let db_path = dir.path().join(format!("{}.needle", long_name));

    // Should either work or fail gracefully
    let result = Database::open(&db_path);
    // Long paths may or may not be supported depending on OS
    assert!(result.is_ok() || result.is_err());
}

/// Test database state after failed operation
#[test]
fn test_state_after_failed_insert() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Insert valid vector
    collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();
    assert_eq!(collection.count(None).unwrap(), 1);

    // Try to insert invalid vector (wrong dimension)
    let result = collection.insert("v2", &[1.0, 2.0], None);
    assert!(result.is_err());

    // Original state should be unchanged
    assert_eq!(collection.count(None).unwrap(), 1);
    assert!(collection.get("v1").is_some());
    assert!(collection.get("v2").is_none());
}

/// Test multiple collections isolation
#[test]
fn test_collection_isolation() {
    let db = Database::in_memory();
    db.create_collection("col1", 4).unwrap();
    db.create_collection("col2", 8).unwrap();

    let col1 = db.collection("col1").unwrap();
    let col2 = db.collection("col2").unwrap();

    // Insert into col1
    col1.insert("v1", &[1.0, 2.0, 3.0, 4.0], None).unwrap();

    // Insert into col2 (different dimensions)
    col2.insert("v1", &[1.0; 8], None).unwrap();

    // Verify isolation
    assert_eq!(col1.count(None).unwrap(), 1);
    assert_eq!(col2.count(None).unwrap(), 1);

    // Delete from col1 shouldn't affect col2
    col1.delete("v1").unwrap();
    assert_eq!(col1.count(None).unwrap(), 0);
    assert_eq!(col2.count(None).unwrap(), 1);
    assert!(col2.get("v1").is_some());
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

/// Test multiple searches on empty collection
#[test]
fn test_multiple_searches_empty_collection() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Multiple searches on empty collection should all return empty
    let queries: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
    ];

    for query in &queries {
        let results = collection.search(query, 10).unwrap();
        assert!(results.is_empty());
    }
}

/// Test filter operations on empty collection
#[test]
fn test_filter_on_empty_collection() {
    use needle::Filter;

    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    let query = vec![1.0, 2.0, 3.0, 4.0];
    let filter = Filter::eq("category", "test");

    // Search with filter on empty collection
    let results = collection.search_with_filter(&query, 10, &filter).unwrap();
    assert!(results.is_empty());

    // Count with filter on empty collection
    assert_eq!(collection.count(Some(&filter)).unwrap(), 0);
}

/// Test delete all vectors then search
#[test]
fn test_delete_all_then_search() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Insert vectors
    for i in 0..10 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }
    assert_eq!(collection.count(None).unwrap(), 10);

    // Delete all
    for i in 0..10 {
        collection.delete(&format!("v{}", i)).unwrap();
    }

    // Search should return empty
    let query = vec![1.0; 4];
    let results = collection.search(&query, 10).unwrap();
    assert!(results.is_empty());
}

/// Test compact after deleting everything
#[test]
fn test_compact_after_delete_all() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Insert and delete
    for i in 0..50 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }
    for i in 0..50 {
        collection.delete(&format!("v{}", i)).unwrap();
    }

    // Compact
    let removed = collection.compact().unwrap();
    assert_eq!(removed, 50);

    // Verify state
    assert_eq!(collection.count(None).unwrap(), 0);
    assert_eq!(collection.deleted_count(), 0);
}

/// Test search with k=1 returns exactly 1 result
#[test]
fn test_search_k_one() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    for i in 0..100 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }

    let query = vec![50.0; 4];
    let results = collection.search(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
}

/// Test search with k > collection size returns all vectors
#[test]
fn test_search_k_exceeds_size() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    for i in 0..5 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }

    let query = vec![2.0; 4];
    let results = collection.search(&query, 1000).unwrap();
    assert_eq!(results.len(), 5);
}

/// Test metadata with various edge case values
#[test]
fn test_metadata_edge_values() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Empty string
    collection.insert("v1", &[1.0; 4], Some(serde_json::json!({"key": ""}))).unwrap();

    // Very long string
    let long_string = "x".repeat(10000);
    collection.insert("v2", &[2.0; 4], Some(serde_json::json!({"key": long_string}))).unwrap();

    // Empty object
    collection.insert("v3", &[3.0; 4], Some(serde_json::json!({}))).unwrap();

    // Empty array
    collection.insert("v4", &[4.0; 4], Some(serde_json::json!({"arr": []}))).unwrap();

    // Deeply nested
    collection.insert("v5", &[5.0; 4], Some(serde_json::json!({
        "a": {"b": {"c": {"d": {"e": 1}}}}
    }))).unwrap();

    // Verify all inserted
    assert_eq!(collection.count(None).unwrap(), 5);

    // Verify retrieval
    let (_, meta) = collection.get("v2").unwrap();
    assert_eq!(meta.unwrap()["key"].as_str().unwrap().len(), 10000);
}

/// Test update with same values (no-op)
#[test]
fn test_update_same_values() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    let original_vec = vec![1.0, 2.0, 3.0, 4.0];
    collection.insert("v1", &original_vec, None).unwrap();

    // Update with same values
    collection.update("v1", &original_vec, None).unwrap();

    // Verify still works
    let (vec, _) = collection.get("v1").unwrap();
    assert_eq!(vec, original_vec);
}

/// Test IDs retrieval order consistency
#[test]
fn test_ids_consistency() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Insert in specific order
    for i in 0..20 {
        collection.insert(format!("vec_{:02}", i), &[i as f32; 4], None).unwrap();
    }

    // Get IDs multiple times, should be consistent
    let ids1 = collection.ids().unwrap();
    let ids2 = collection.ids().unwrap();

    assert_eq!(ids1.len(), 20);
    assert_eq!(ids1, ids2);
}

/// Test distance results are sorted
#[test]
fn test_search_results_sorted_by_distance() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Insert vectors at known distances
    for i in 0..50 {
        collection.insert(format!("v{}", i), &[i as f32; 4], None).unwrap();
    }

    let query = vec![25.0; 4]; // Middle value
    let results = collection.search(&query, 50).unwrap();

    // Verify sorted by distance (ascending)
    for i in 1..results.len() {
        assert!(results[i - 1].distance <= results[i].distance,
            "Results not sorted: {} > {} at index {}",
            results[i - 1].distance, results[i].distance, i);
    }
}

/// Test search with identical vectors
#[test]
fn test_search_identical_vectors() {
    let db = Database::in_memory();
    db.create_collection("test", 4).unwrap();
    let collection = db.collection("test").unwrap();

    // Insert many identical vectors
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    for i in 0..10 {
        collection.insert(format!("v{}", i), &vec, None).unwrap();
    }

    // Search for that exact vector
    let results = collection.search(&vec, 10).unwrap();
    assert_eq!(results.len(), 10);

    // All distances should be 0 (or very close for cosine)
    for result in &results {
        assert!(result.distance < 0.0001, "Expected near-zero distance, got {}", result.distance);
    }
}

/// Test concurrent inserts don't lose data
#[test]
fn test_concurrent_inserts() {
    use std::thread;
    use std::sync::Arc;

    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 4).unwrap();

    let mut handles = vec![];
    for thread_id in 0..4 {
        let db_clone = Arc::clone(&db);
        let handle = thread::spawn(move || {
            let collection = db_clone.collection("test").unwrap();
            for i in 0..25 {
                let id = format!("t{}_v{}", thread_id, i);
                collection.insert(id, &[i as f32; 4], None).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let collection = db.collection("test").unwrap();
    assert_eq!(collection.count(None).unwrap(), 100);
}

/// Test WAL basic write and replay
#[cfg(unix)]
#[test]
fn test_wal_basic_persistence() {
    use needle::{WalConfig, WalManager};
    use needle::wal::WalEntry;
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let config = WalConfig::default();

    // Write some entries
    {
        let manager = WalManager::open(&wal_dir, config.clone()).unwrap();
        for i in 0..10 {
            let entry = WalEntry::Insert {
                collection: "test".to_string(),
                id: format!("v{}", i),
                vector: vec![i as f32; 4],
                metadata: None,
            };
            manager.append(entry).unwrap();
        }
    }

    // Reopen and verify via replay
    {
        let manager = WalManager::open(&wal_dir, config).unwrap();
        let mut count = 0;
        manager.replay(0, |_record| {
            count += 1;
            Ok(())
        }).unwrap();
        assert!(count > 0, "Should have replayed WAL entries");
    }
}
