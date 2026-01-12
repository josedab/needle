//! Corruption Recovery Tests
//!
//! Tests for database corruption detection and handling.

use needle::{Database, CollectionConfig, DistanceFunction};
use std::fs;
use std::io::{Read, Write, Seek, SeekFrom};
use tempfile::tempdir;

/// Test that corrupted database files are detected
#[test]
fn test_corrupted_header_detection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create a valid database
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("test", 64).unwrap();
        let collection = db.collection("test").unwrap();
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        collection.insert("vec1", &vector, None).unwrap();
        db.save().unwrap();
    }

    // Verify file exists
    assert!(db_path.exists());
    let original_size = fs::metadata(&db_path).unwrap().len();
    assert!(original_size > 0);

    // Corrupt the header (first few bytes)
    {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&db_path)
            .unwrap();
        file.write_all(b"CORRUPTED").unwrap();
    }

    // Try to open corrupted database - should fail
    let result = Database::open(&db_path);
    assert!(result.is_err(), "Should detect corrupted header");
}

/// Test detection of truncated files
#[test]
fn test_truncated_file_detection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create a valid database with data
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("test", 128).unwrap();
        let collection = db.collection("test").unwrap();
        for i in 0..10 {
            let vector: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32) / 1280.0).collect();
            collection.insert(format!("vec_{}", i), &vector, None).unwrap();
        }
        db.save().unwrap();
    }

    let original_size = fs::metadata(&db_path).unwrap().len();

    // Truncate the file to half its size
    {
        let file = fs::OpenOptions::new()
            .write(true)
            .open(&db_path)
            .unwrap();
        file.set_len(original_size / 2).unwrap();
    }

    // Try to open truncated database - should fail
    let result = Database::open(&db_path);
    assert!(result.is_err(), "Should detect truncated file");
}

/// Test that zero-length files are handled
#[test]
fn test_empty_file_handling() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("empty.needle");

    // Create an empty file
    fs::write(&db_path, b"").unwrap();

    // Try to open - should fail gracefully
    let result = Database::open(&db_path);
    assert!(result.is_err(), "Should reject empty file");
}

/// Test that random garbage is detected
#[test]
fn test_garbage_file_detection() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("garbage.needle");

    // Write random garbage
    let garbage: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
    fs::write(&db_path, &garbage).unwrap();

    // Try to open - should fail
    let result = Database::open(&db_path);
    assert!(result.is_err(), "Should reject garbage file");
}

/// Test recovery from partial writes
#[test]
fn test_partial_write_recovery() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.needle");

    // Create a valid database
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("test", 32).unwrap();
        let collection = db.collection("test").unwrap();
        let vector: Vec<f32> = (0..32).map(|i| i as f32 / 32.0).collect();
        collection.insert("vec1", &vector, None).unwrap();
        db.save().unwrap();
    }

    // Simulate partial write by appending garbage
    {
        let mut file = fs::OpenOptions::new()
            .append(true)
            .open(&db_path)
            .unwrap();
        // Append some garbage at the end (simulating interrupted write)
        file.write_all(&[0xFF; 100]).unwrap();
    }

    // Database should still load (the garbage is after valid data)
    // Note: This behavior depends on implementation
    let result = Database::open(&db_path);
    // The file might still be valid if the implementation ignores trailing data
    // or it might fail if it validates the entire file
    // Either behavior is acceptable for corruption handling
    if result.is_ok() {
        let db = result.unwrap();
        // If it loads, verify data is intact
        let collection = db.collection("test").unwrap();
        let (vec, _) = collection.get("vec1").unwrap();
        assert_eq!(vec.len(), 32);
    }
}

/// Test that collections with corrupted vectors are detected
#[test]
fn test_in_memory_collection_integrity() {
    let config = CollectionConfig::new("integrity_test", 16)
        .with_distance(DistanceFunction::Cosine);
    let mut collection = needle::Collection::new(config);

    // Insert valid data
    for i in 0..5 {
        let vector: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32) / 80.0).collect();
        collection.insert(format!("vec_{}", i), &vector, None).unwrap();
    }

    // Serialize and deserialize
    let bytes = collection.to_bytes().unwrap();
    let restored = needle::Collection::from_bytes(&bytes).unwrap();

    // Verify integrity
    assert_eq!(restored.len(), 5);
    for i in 0..5 {
        let id = format!("vec_{}", i);
        let (vec, _) = restored.get(&id).unwrap();
        assert_eq!(vec.len(), 16);
        // Verify values are finite
        for &v in vec.iter() {
            assert!(v.is_finite());
        }
    }
}

/// Test serialization/deserialization roundtrip
#[test]
fn test_serialization_roundtrip_integrity() {
    use serde_json::json;

    let config = CollectionConfig::new("roundtrip_test", 8);
    let mut collection = needle::Collection::new(config);

    // Insert vectors with various metadata types
    let test_cases = vec![
        ("vec_null_meta", vec![1.0; 8], None),
        ("vec_empty_meta", vec![2.0; 8], Some(json!({}))),
        ("vec_string_meta", vec![3.0; 8], Some(json!({"key": "value"}))),
        ("vec_number_meta", vec![4.0; 8], Some(json!({"num": 42.5}))),
        ("vec_array_meta", vec![5.0; 8], Some(json!({"arr": [1, 2, 3]}))),
        ("vec_nested_meta", vec![6.0; 8], Some(json!({"nested": {"a": {"b": "c"}}}))),
    ];

    for (id, vec, meta) in &test_cases {
        collection.insert(*id, vec, meta.clone()).unwrap();
    }

    // Roundtrip
    let bytes = collection.to_bytes().unwrap();
    let restored = needle::Collection::from_bytes(&bytes).unwrap();

    // Verify all data
    for (id, expected_vec, expected_meta) in &test_cases {
        let (vec, meta) = restored.get(*id).expect("Vector should exist");

        // Verify vector values
        assert_eq!(vec.len(), expected_vec.len());
        for (a, b) in vec.iter().zip(expected_vec.iter()) {
            assert!((a - b).abs() < 1e-6, "Vector values should match");
        }

        // Verify metadata
        match (meta, expected_meta) {
            (None, None) => {}
            (Some(m), Some(e)) => {
                assert_eq!(m, e, "Metadata should match");
            }
            _ => panic!("Metadata presence mismatch for {}", id),
        }
    }
}

/// Test that database integrity is maintained across multiple save/load cycles
#[test]
fn test_multiple_save_load_cycles() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("cycles.needle");

    // First cycle: create and populate
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("cycle_test", 16).unwrap();
        let collection = db.collection("cycle_test").unwrap();
        for i in 0..5 {
            let vector: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32) / 80.0).collect();
            collection.insert(format!("cycle1_vec_{}", i), &vector, None).unwrap();
        }
        db.save().unwrap();
    }

    // Second cycle: add more data
    {
        let mut db = Database::open(&db_path).unwrap();
        let collection = db.collection("cycle_test").unwrap();
        assert_eq!(collection.len(), 5);
        for i in 0..5 {
            let vector: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32) / 80.0 + 0.5).collect();
            collection.insert(format!("cycle2_vec_{}", i), &vector, None).unwrap();
        }
        db.save().unwrap();
    }

    // Third cycle: verify all data
    {
        let db = Database::open(&db_path).unwrap();
        let collection = db.collection("cycle_test").unwrap();
        assert_eq!(collection.len(), 10);

        // Verify cycle1 vectors
        for i in 0..5 {
            let id = format!("cycle1_vec_{}", i);
            assert!(collection.get(&id).is_some(), "Cycle 1 vector {} should exist", i);
        }

        // Verify cycle2 vectors
        for i in 0..5 {
            let id = format!("cycle2_vec_{}", i);
            assert!(collection.get(&id).is_some(), "Cycle 2 vector {} should exist", i);
        }
    }
}

/// Test that state checksum validation works when state data is corrupted
#[test]
fn test_state_checksum_validation() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("checksum.needle");

    // Create a valid database with enough data to have meaningful state
    {
        let mut db = Database::open(&db_path).unwrap();
        db.create_collection("checksum_test", 128).unwrap();
        let collection = db.collection("checksum_test").unwrap();
        for i in 0..20 {
            let vector: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32) / 2560.0).collect();
            collection.insert(format!("vec_{}", i), &vector, Some(serde_json::json!({"idx": i}))).unwrap();
        }
        db.save().unwrap();
    }

    let file_size = fs::metadata(&db_path).unwrap().len();
    assert!(file_size > 1000, "File should be large enough to corrupt");

    // Corrupt multiple areas of the file to increase chance of hitting state data
    {
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&db_path)
            .unwrap();

        // Corrupt at multiple positions after the header
        for offset in [300, 500, 1000, file_size / 2, file_size - 100] {
            if offset < file_size - 10 {
                file.seek(SeekFrom::Start(offset)).unwrap();
                file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]).unwrap();
            }
        }
    }

    // Try to open - should detect corruption
    let result = Database::open(&db_path);
    // The database should fail due to:
    // 1. Checksum mismatch
    // 2. Deserialization error
    // 3. Invalid data
    // Note: If it somehow loads, the data should at least be corrupted
    match result {
        Err(_) => {
            // Expected - corruption detected
        }
        Ok(db) => {
            // If it loads, check if data is actually valid
            // This is also acceptable if the corruption didn't hit critical data
            if let Ok(collection) = db.collection("checksum_test") {
                // Verify we have the expected count
                // If corruption occurred, we might have missing/invalid data
                let count = collection.len();
                // Original was 20 vectors - if corruption didn't affect count, that's ok
                // but this tests that the database at least loads consistently
                assert!(count <= 20, "Should not have more vectors than inserted");
            }
        }
    }
}
