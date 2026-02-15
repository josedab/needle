//! Concurrent access tests for the Needle vector database
//! Tests thread safety, lock contention, and parallel operations

use needle::Database;
use std::sync::Arc;
use std::thread;

fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

// ============================================================================
// Basic Concurrency Tests
// ============================================================================

#[test]
fn test_concurrent_readers() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 128).unwrap();

    // Insert some vectors first
    {
        let coll = db.collection("test").unwrap();
        for i in 0..100 {
            coll.insert(format!("v{}", i), &random_vector(128), None)
                .unwrap();
        }
    }

    // Spawn many concurrent readers
    let mut handles = vec![];
    for _ in 0..50 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                let coll = db_clone.collection("test").unwrap();
                let query = random_vector(128);
                let results = coll.search(&query, 10).unwrap();
                assert!(!results.is_empty());
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_concurrent_writes() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 128).unwrap();

    let mut handles = vec![];
    for t in 0..10 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..10 {
                let coll = db_clone.collection("test").unwrap();
                let id = format!("v_{}_{}", t, i);
                coll.insert(id, &random_vector(128), None).unwrap();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all vectors were inserted
    let coll = db.collection("test").unwrap();
    assert_eq!(coll.len(), 100);
}

#[test]
fn test_mixed_read_write() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 64).unwrap();

    // Pre-populate
    {
        let coll = db.collection("test").unwrap();
        for i in 0..50 {
            coll.insert(format!("initial_{}", i), &random_vector(64), None)
                .unwrap();
        }
    }

    let mut handles = vec![];

    // Writers
    for t in 0..5 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..20 {
                let coll = db_clone.collection("test").unwrap();
                let id = format!("new_{}_{}", t, i);
                coll.insert(id, &random_vector(64), None).unwrap();
                thread::yield_now();
            }
        }));
    }

    // Readers
    for _ in 0..10 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for _ in 0..20 {
                let coll = db_clone.collection("test").unwrap();
                let query = random_vector(64);
                let _ = coll.search(&query, 5);
                thread::yield_now();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify final state
    let coll = db.collection("test").unwrap();
    assert_eq!(coll.len(), 50 + 5 * 20); // initial + new
}

#[test]
fn test_concurrent_delete_during_search() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 32).unwrap();

    // Insert vectors
    {
        let coll = db.collection("test").unwrap();
        for i in 0..200 {
            coll.insert(format!("v{}", i), &random_vector(32), None)
                .unwrap();
        }
    }

    let mut handles = vec![];

    // Deleters
    for t in 0..4 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                let idx = t * 50 + i;
                let coll = db_clone.collection("test").unwrap();
                let _ = coll.delete(&format!("v{}", idx));
                thread::yield_now();
            }
        }));
    }

    // Searchers - should not panic even during deletions
    for _ in 0..8 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let coll = db_clone.collection("test").unwrap();
                let query = random_vector(32);
                // This should not panic
                let _ = coll.search(&query, 10);
                thread::yield_now();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ============================================================================
// Lock Contention Tests
// ============================================================================

#[test]
fn test_high_contention_single_collection() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("contended", 16).unwrap();

    let mut handles = vec![];

    // Many threads hammering the same collection
    for t in 0..20 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                let coll = db_clone.collection("contended").unwrap();

                if i % 3 == 0 {
                    // Insert
                    let _ = coll.insert(format!("t{}_i{}", t, i), &random_vector(16), None);
                } else if i % 3 == 1 {
                    // Search
                    let _ = coll.search(&random_vector(16), 5);
                } else {
                    // Read
                    let _ = coll.len();
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_multiple_collections_no_contention() {
    let db = Arc::new(Database::in_memory());

    // Create multiple collections
    for i in 0..10 {
        db.create_collection(format!("coll{}", i), 32).unwrap();
    }

    let mut handles = vec![];

    // Each thread works on its own collection
    for t in 0..10 {
        let db_clone = Arc::clone(&db);
        let coll_name = format!("coll{}", t);
        handles.push(thread::spawn(move || {
            let coll = db_clone.collection(&coll_name).unwrap();

            for i in 0..100 {
                coll.insert(format!("v{}", i), &random_vector(32), None)
                    .unwrap();
            }

            for _ in 0..10 {
                let _ = coll.search(&random_vector(32), 5);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Verify each collection has expected count
    for i in 0..10 {
        let coll_name = format!("coll{}", i);
        let coll = db.collection(&coll_name).unwrap();
        assert_eq!(coll.len(), 100);
    }
}

// ============================================================================
// Batch Operation Concurrency
// ============================================================================

#[test]
fn test_concurrent_multiple_searches() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 64).unwrap();

    // Insert vectors
    {
        let coll = db.collection("test").unwrap();
        for i in 0..500 {
            coll.insert(format!("v{}", i), &random_vector(64), None)
                .unwrap();
        }
    }

    let mut handles = vec![];

    // Multiple searches in parallel (simulating batch behavior)
    for _ in 0..10 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            let coll = db_clone.collection("test").unwrap();

            // Perform multiple searches
            for _ in 0..20 {
                let query = random_vector(64);
                let results = coll.search(&query, 10).unwrap();
                assert!(!results.is_empty());
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_stress_insert_search_delete() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("stress", 32).unwrap();

    let iterations = 100;
    let threads = 8;

    let mut handles = vec![];

    for t in 0..threads {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..iterations {
                let coll = db_clone.collection("stress").unwrap();
                let id = format!("t{}_i{}", t, i);

                // Insert
                coll.insert(&id, &random_vector(32), None).unwrap();

                // Search
                let _ = coll.search(&random_vector(32), 5);

                // Occasionally delete
                if i % 10 == 0 {
                    let _ = coll.delete(&id);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

// ============================================================================
// Database-Level Concurrency
// ============================================================================

#[test]
fn test_concurrent_collection_creation() {
    let db = Arc::new(Database::in_memory());

    let mut handles = vec![];

    // Try to create many collections concurrently
    for t in 0..10 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..10 {
                let name = format!("coll_{}_{}", t, i);
                let _ = db_clone.create_collection(&name, 32);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All collections should be created
    let collections = db.list_collections();
    assert_eq!(collections.len(), 100);
}

#[test]
fn test_concurrent_collection_deletion() {
    let db = Arc::new(Database::in_memory());

    // Create collections first
    for i in 0..50 {
        db.create_collection(format!("to_delete_{}", i), 16)
            .unwrap();
    }

    let mut handles = vec![];

    // Delete concurrently
    for t in 0..5 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..10 {
                let name = format!("to_delete_{}", t * 10 + i);
                let _ = db_clone.drop_collection(&name);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All should be deleted
    let collections = db.list_collections();
    assert!(collections.is_empty());
}

// ============================================================================
// Save/Load During Operations
// ============================================================================

#[test]
fn test_operations_during_dirty_state() {
    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 32).unwrap();

    let mut handles = vec![];

    // Writers
    for t in 0..5 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for i in 0..20 {
                let coll = db_clone.collection("test").unwrap();
                coll.insert(format!("t{}_i{}", t, i), &random_vector(32), None)
                    .unwrap();
            }
        }));
    }

    // Periodic dirty check
    for _ in 0..3 {
        let db_clone = Arc::clone(&db);
        handles.push(thread::spawn(move || {
            for _ in 0..10 {
                let _ = db_clone.is_dirty();
                thread::yield_now();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert!(db.is_dirty());
}
