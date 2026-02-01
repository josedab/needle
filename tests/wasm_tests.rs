//! WASM Integration Tests
//!
//! These tests verify the WASM bindings work correctly.
//! For actual browser testing, use wasm-pack test --headless --chrome
//!
//! Note: These tests compile with the wasm feature but run in native Rust.
//! They test the logic and API, not actual browser/IndexedDB behavior.

#![cfg(feature = "wasm")]

use needle::wasm::*;

#[test]
fn test_wasm_collection_creation() {
    let collection = WasmCollection::new("test_collection", 128, Some("cosine".to_string()))
        .expect("Should create collection");

    assert_eq!(collection.name(), "test_collection");
    assert_eq!(collection.dimensions(), 128);
    assert_eq!(collection.length(), 0);
    assert!(collection.is_empty());
}

#[test]
fn test_wasm_collection_insert_and_search() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    // Insert vectors
    collection
        .insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None)
        .expect("Should insert vec1");

    collection
        .insert("vec2", vec![0.0, 1.0, 0.0, 0.0], None)
        .expect("Should insert vec2");

    collection
        .insert("vec3", vec![0.0, 0.0, 1.0, 0.0], None)
        .expect("Should insert vec3");

    assert_eq!(collection.length(), 3);
    assert!(!collection.is_empty());

    // Search
    let results = collection
        .search(vec![1.0, 0.1, 0.0, 0.0], 2)
        .expect("Should search");

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id(), "vec1");
}

#[test]
fn test_wasm_collection_with_metadata() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    let metadata = r#"{"category": "test", "score": 42}"#;
    collection
        .insert("vec1", vec![1.0, 0.0, 0.0, 0.0], Some(metadata.to_string()))
        .expect("Should insert with metadata");

    // Note: get() returns JsValue which requires browser environment
    // Just verify we can insert with metadata without errors
    assert_eq!(collection.length(), 1);
    assert!(collection.contains("vec1"));
}

#[test]
fn test_wasm_collection_delete() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    assert_eq!(collection.length(), 1);

    let deleted = collection.delete("vec1").unwrap();
    assert!(deleted);
    assert_eq!(collection.length(), 0);

    let deleted_again = collection.delete("vec1").unwrap();
    assert!(!deleted_again);
}

#[test]
fn test_wasm_collection_contains() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();

    assert!(collection.contains("vec1"));
    assert!(!collection.contains("vec2"));
}

#[test]
fn test_wasm_serialization_roundtrip() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    collection.insert("vec2", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();

    // Serialize to bytes
    let bytes = collection.to_bytes().expect("Should serialize");
    assert!(!bytes.is_empty());

    // Deserialize
    let restored = WasmCollection::from_bytes(&bytes).expect("Should deserialize");
    assert_eq!(restored.length(), 2);
    assert!(restored.contains("vec1"));
    assert!(restored.contains("vec2"));
}

#[test]
fn test_wasm_base64_roundtrip() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    collection.insert("vec1", vec![1.0, 2.0, 3.0, 4.0], None).unwrap();

    // Serialize to base64
    let base64_str = collection.to_base64().expect("Should serialize to base64");
    assert!(!base64_str.is_empty());

    // Deserialize
    let restored = WasmCollection::from_base64(&base64_str).expect("Should deserialize from base64");
    assert_eq!(restored.length(), 1);
    assert!(restored.contains("vec1"));
}

#[test]
fn test_wasm_memory_stats() {
    let collection = WasmCollection::new("test", 128, None).unwrap();

    for i in 0..100 {
        let vector: Vec<f32> = (0..128).map(|j| (i * j) as f32 / 1000.0).collect();
        collection
            .insert(&format!("vec{}", i), vector, None)
            .unwrap();
    }

    let stats = collection.get_memory_stats().expect("Should get memory stats");
    assert_eq!(stats.vectors_count(), 100);
    assert_eq!(stats.dimensions(), 128);
    assert!(stats.total_bytes() > 0);
}

#[test]
fn test_wasm_clear() {
    let collection = WasmCollection::new("test", 4, None).unwrap();

    collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    collection.insert("vec2", vec![0.0, 1.0, 0.0, 0.0], None).unwrap();
    assert_eq!(collection.length(), 2);

    collection.clear().unwrap();
    assert_eq!(collection.length(), 0);
    assert!(collection.is_empty());
}

#[test]
fn test_indexed_db_config() {
    let config = IndexedDbConfig::new("needle_db", "vectors");
    assert_eq!(config.db_name(), "needle_db");
    assert_eq!(config.store_name(), "vectors");
    assert!(config.auto_save());
    assert_eq!(config.save_interval_ms(), 5000);

    let config2 = IndexedDbConfig::new("test_db", "test_store")
        .without_auto_save()
        .with_save_interval(10000);
    assert!(!config2.auto_save());
    assert_eq!(config2.save_interval_ms(), 10000);
}

#[test]
fn test_persistent_collection() {
    let config = IndexedDbConfig::new("test_db", "test_store");
    let mut collection = PersistentCollection::new("test", 4, None, config).unwrap();

    assert!(!collection.is_dirty());

    collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    assert!(collection.is_dirty());

    // Get serialized data (for IndexedDB storage)
    let data = collection.get_serialized_data().unwrap();
    assert!(!data.is_empty());

    // Mark as saved
    collection.mark_saved();
    assert!(!collection.is_dirty());

    // Delete marks dirty again
    collection.delete("vec1").unwrap();
    assert!(collection.is_dirty());
}

#[test]
fn test_persistent_collection_restore() {
    // Create and populate
    let config1 = IndexedDbConfig::new("test_db", "test_store");
    let mut collection1 = PersistentCollection::new("test", 4, None, config1).unwrap();

    collection1.insert("vec1", vec![1.0, 2.0, 3.0, 4.0], None).unwrap();
    collection1.insert("vec2", vec![5.0, 6.0, 7.0, 8.0], None).unwrap();

    let data = collection1.get_serialized_data().unwrap();

    // Restore
    let config2 = IndexedDbConfig::new("test_db", "test_store");
    let collection2 = PersistentCollection::restore_from_data(&data, config2).unwrap();

    // Verify
    let results = collection2.search(vec![1.0, 2.0, 3.0, 4.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id(), "vec1");
}

// Note: SyncStatus uses js_sys::Date which only works in wasm32 environment
// This test is for documentation and should be run with wasm-pack test
#[cfg(target_arch = "wasm32")]
#[test]
fn test_sync_status() {
    let mut status = SyncStatus::new();

    assert!(status.is_online());
    assert_eq!(status.pending_changes(), 0);

    status.add_pending_change();
    status.add_pending_change();
    assert_eq!(status.pending_changes(), 2);

    status.set_online(false);
    assert!(!status.is_online());

    status.clear_pending_changes();
    assert_eq!(status.pending_changes(), 0);
    assert!(status.last_sync_timestamp() > 0.0);
}

#[test]
fn test_indexed_db_helpers_generation() {
    let helpers = get_indexed_db_helpers();
    
    // Verify JavaScript code is generated
    assert!(helpers.contains("NeedleIndexedDb"));
    assert!(helpers.contains("openDatabase"));
    assert!(helpers.contains("save"));
    assert!(helpers.contains("load"));
    assert!(helpers.contains("saveCollection"));
    assert!(helpers.contains("createAutoSaver"));
}

#[test]
fn test_service_worker_helpers_generation() {
    let helpers = get_service_worker_helpers();
    
    // Verify JavaScript code is generated
    assert!(helpers.contains("NeedleServiceWorker"));
    assert!(helpers.contains("register"));
    assert!(helpers.contains("generateScript"));
    assert!(helpers.contains("isOffline"));
    assert!(helpers.contains("onConnectivityChange"));
}

#[test]
fn test_distance_functions() {
    // Test with cosine distance
    let cosine_collection = WasmCollection::new("test", 4, Some("cosine".to_string())).unwrap();
    cosine_collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    
    // Test with euclidean distance
    let euclidean_collection = WasmCollection::new("test", 4, Some("euclidean".to_string())).unwrap();
    euclidean_collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    
    // Test with dot product
    let dot_collection = WasmCollection::new("test", 4, Some("dot".to_string())).unwrap();
    dot_collection.insert("vec1", vec![1.0, 0.0, 0.0, 0.0], None).unwrap();
    
    // All should work
    assert_eq!(cosine_collection.length(), 1);
    assert_eq!(euclidean_collection.length(), 1);
    assert_eq!(dot_collection.length(), 1);
}

// Note: test_invalid_distance_function is skipped because wasm-bindgen's
// JsValue error handling doesn't work on non-wasm32 targets.
// This test should be run with wasm-pack test in a browser environment.

#[test]
fn test_ef_search_parameter() {
    let collection = WasmCollection::new("test", 4, None).unwrap();
    
    // Insert vectors
    for i in 0..100 {
        let vector = vec![(i as f32) / 100.0, 0.0, 0.0, 0.0];
        collection.insert(&format!("vec{}", i), vector, None).unwrap();
    }
    
    // Should be able to set ef_search
    collection.set_ef_search(100).expect("Should set ef_search");
    
    // Search should still work
    let results = collection.search(vec![0.5, 0.0, 0.0, 0.0], 10).unwrap();
    assert_eq!(results.len(), 10);
}

// Note: Actual IndexedDB and browser tests require wasm-pack test environment
// These tests verify the Rust-side logic works correctly
