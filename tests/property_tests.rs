//! Property-based tests for Needle vector database

use needle::{Collection, CollectionConfig, Database, DistanceFunction, Filter};
use proptest::prelude::*;
use serde_json::json;

/// Generate a random vector of the given dimension
fn arb_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1.0f32..1.0f32, dim)
}

/// Generate a random vector ID
fn arb_id() -> impl Strategy<Value = String> {
    "[a-z0-9]{1,16}".prop_map(|s| format!("vec_{}", s))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property: Inserting and retrieving a vector should return the same vector
    #[test]
    fn prop_insert_get_roundtrip(
        id in arb_id(),
        vector in arb_vector(64)
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 64).unwrap();
        let coll = db.collection("test").unwrap();

        coll.insert(&id, &vector, None).unwrap();
        let (retrieved, _) = coll.get(&id).unwrap();

        prop_assert_eq!(retrieved.len(), vector.len());
        for (a, b) in retrieved.iter().zip(vector.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }
    }

    /// Property: Search should return at most k results
    #[test]
    fn prop_search_returns_at_most_k(
        vectors in prop::collection::vec(arb_vector(32), 10..50),
        k in 1usize..20
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();
        let coll = db.collection("test").unwrap();

        for (i, vec) in vectors.iter().enumerate() {
            coll.insert(format!("vec_{}", i), vec, None).unwrap();
        }

        let query = &vectors[0];
        let results = coll.search(query, k).unwrap();

        prop_assert!(results.len() <= k);
        prop_assert!(results.len() <= vectors.len());
    }

    /// Property: Deleting a vector should make it unfindable
    #[test]
    fn prop_delete_removes_vector(
        id in arb_id(),
        vector in arb_vector(32)
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();
        let coll = db.collection("test").unwrap();

        coll.insert(&id, &vector, None).unwrap();
        prop_assert!(coll.get(&id).is_some());

        coll.delete(&id).unwrap();
        prop_assert!(coll.get(&id).is_none());
    }

    /// Property: Collection length should match number of inserted vectors
    #[test]
    fn prop_collection_length(
        n in 1usize..50
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 16).unwrap();
        let coll = db.collection("test").unwrap();

        for i in 0..n {
            let vec: Vec<f32> = (0..16).map(|j| (i * 16 + j) as f32 / 100.0).collect();
            coll.insert(format!("vec_{}", i), &vec, None).unwrap();
        }

        prop_assert_eq!(coll.len(), n);
    }

    /// Property: Distances should be non-negative for Euclidean distance
    #[test]
    fn prop_euclidean_distance_non_negative(
        v1 in arb_vector(32),
        v2 in arb_vector(32)
    ) {
        let dist = DistanceFunction::Euclidean.compute(&v1, &v2);
        prop_assert!(dist >= 0.0, "Euclidean distance should be non-negative");
    }

    /// Property: Distance from a vector to itself should be zero
    #[test]
    fn prop_self_distance_is_zero(
        vector in arb_vector(32)
    ) {
        let euclidean_dist = DistanceFunction::Euclidean.compute(&vector, &vector);
        prop_assert!(euclidean_dist.abs() < 1e-6, "Self-distance should be ~0");

        // Cosine distance for self should also be ~0 (if vector is non-zero)
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 1e-6 {
            let cosine_dist = DistanceFunction::Cosine.compute(&vector, &vector);
            prop_assert!(cosine_dist.abs() < 1e-5, "Cosine self-distance should be ~0");
        }
    }

    /// Property: Search results should be sorted by distance (ascending)
    #[test]
    fn prop_search_results_sorted(
        vectors in prop::collection::vec(arb_vector(16), 20..50),
        k in 5usize..15
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 16).unwrap();
        let coll = db.collection("test").unwrap();

        for (i, vec) in vectors.iter().enumerate() {
            coll.insert(format!("vec_{}", i), vec, None).unwrap();
        }

        let query = &vectors[0];
        let results = coll.search(query, k).unwrap();

        // Check results are sorted by distance
        for window in results.windows(2) {
            prop_assert!(
                window[0].distance <= window[1].distance,
                "Results should be sorted by distance"
            );
        }
    }

    /// Property: Metadata filter should only return matching results
    #[test]
    fn prop_filter_matches(
        n in 10usize..30
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 8).unwrap();
        let coll = db.collection("test").unwrap();

        // Insert vectors with category metadata
        for i in 0..n {
            let vec: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32 / 100.0).collect();
            let category = if i % 2 == 0 { "even" } else { "odd" };
            let metadata = json!({"category": category, "index": i});
            coll.insert(format!("vec_{}", i), &vec, Some(metadata)).unwrap();
        }

        let query: Vec<f32> = vec![0.5; 8];
        let filter = Filter::eq("category", "even");
        let results = coll.search_with_filter(&query, n, &filter).unwrap();

        // All results should have category == "even"
        for result in &results {
            if let Some(meta) = &result.metadata {
                prop_assert_eq!(&meta["category"], "even");
            }
        }
    }

    /// Property: Upsert should update existing vectors
    #[test]
    fn prop_upsert_updates(
        id in arb_id(),
        v1 in arb_vector(16),
        v2 in arb_vector(16)
    ) {
        let db = Database::in_memory();
        db.create_collection_with_config(
            CollectionConfig::new("test", 16)
        ).unwrap();
        let mut coll = Collection::new(CollectionConfig::new("test", 16));

        // First insert
        coll.upsert(&id, &v1, None).unwrap();
        let (retrieved1, _) = coll.get(&id).unwrap();
        for (a, b) in retrieved1.iter().zip(v1.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }

        // Upsert with new vector
        coll.upsert(&id, &v2, None).unwrap();
        let (retrieved2, _) = coll.get(&id).unwrap();
        for (a, b) in retrieved2.iter().zip(v2.iter()) {
            prop_assert!((a - b).abs() < 1e-6);
        }

        // Should still be only one vector
        prop_assert_eq!(coll.len(), 1);
    }

    /// Property: Batch delete should remove all specified vectors
    #[test]
    fn prop_batch_delete(
        n in 10usize..30,
        delete_count in 2usize..8
    ) {
        let mut coll = Collection::new(CollectionConfig::new("test", 8));

        // Insert vectors
        for i in 0..n {
            let vec: Vec<f32> = (0..8).map(|j| (i * 8 + j) as f32 / 100.0).collect();
            coll.insert(format!("vec_{}", i), &vec, None).unwrap();
        }

        // Ensure we don't try to delete more than we have
        let actual_delete_count = delete_count.min(n);
        let ids_to_delete: Vec<String> = (0..actual_delete_count)
            .map(|i| format!("vec_{}", i))
            .collect();

        let deleted = coll.delete_batch(&ids_to_delete).unwrap();
        prop_assert_eq!(deleted, actual_delete_count);
        prop_assert_eq!(coll.len(), n - actual_delete_count);

        // Verify deleted vectors are gone
        for id in &ids_to_delete {
            prop_assert!(coll.get(id).is_none());
        }
    }

    /// Property: Search should be idempotent - same query returns same results
    #[test]
    fn prop_search_idempotence(
        vectors in prop::collection::vec(arb_vector(32), 20..50),
        k in 5usize..15
    ) {
        let db = Database::in_memory();
        db.create_collection("test", 32).unwrap();
        let coll = db.collection("test").unwrap();

        for (i, vec) in vectors.iter().enumerate() {
            coll.insert(format!("vec_{}", i), vec, None).unwrap();
        }

        let query = &vectors[0];

        // Run the same search twice
        let results1 = coll.search(query, k).unwrap();
        let results2 = coll.search(query, k).unwrap();

        // Results should be identical
        prop_assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            prop_assert_eq!(&r1.id, &r2.id);
            prop_assert!((r1.distance - r2.distance).abs() < 1e-6);
        }
    }

    /// Property: Triangle inequality for Euclidean distance - d(a,c) <= d(a,b) + d(b,c)
    #[test]
    fn prop_distance_triangle_inequality(
        a in arb_vector(32),
        b in arb_vector(32),
        c in arb_vector(32)
    ) {
        let d_ab = DistanceFunction::Euclidean.compute(&a, &b);
        let d_bc = DistanceFunction::Euclidean.compute(&b, &c);
        let d_ac = DistanceFunction::Euclidean.compute(&a, &c);

        // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        // Allow small epsilon for floating point errors
        prop_assert!(
            d_ac <= d_ab + d_bc + 1e-5,
            "Triangle inequality violated: d(a,c)={} > d(a,b)={} + d(b,c)={}",
            d_ac, d_ab, d_bc
        );
    }

    /// Property: Collection serialization roundtrip - export then import preserves data
    #[test]
    fn prop_serialization_roundtrip(
        vectors in prop::collection::vec(arb_vector(16), 5..20)
    ) {
        // Create database and collection
        let db1 = Database::in_memory();
        db1.create_collection("test", 16).unwrap();
        let coll1 = db1.collection("test").unwrap();

        // Insert vectors with metadata
        for (i, vec) in vectors.iter().enumerate() {
            let metadata = json!({"index": i, "value": format!("item_{}", i)});
            coll1.insert(format!("vec_{}", i), vec, Some(metadata)).unwrap();
        }

        // Export all data
        let exported = coll1.export_all().unwrap();

        // Create second database/collection and import
        let db2 = Database::in_memory();
        db2.create_collection("test2", 16).unwrap();
        let coll2 = db2.collection("test2").unwrap();

        for (id, vector, metadata) in exported {
            coll2.insert(&id, &vector, metadata).unwrap();
        }

        // Verify collections match
        prop_assert_eq!(coll1.len(), coll2.len());

        // Verify each vector
        for i in 0..vectors.len() {
            let id = format!("vec_{}", i);
            let (v1, m1) = coll1.get(&id).unwrap();
            let (v2, m2) = coll2.get(&id).unwrap();

            // Vectors should match
            for (a, b) in v1.iter().zip(v2.iter()) {
                prop_assert!((a - b).abs() < 1e-6);
            }

            // Metadata should match
            prop_assert_eq!(m1, m2);
        }
    }
}

/// Test distance function symmetry
#[test]
fn test_distance_symmetry() {
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::default();

    runner
        .run(&(arb_vector(32), arb_vector(32)), |(v1, v2)| {
            let dist1 = DistanceFunction::Euclidean.compute(&v1, &v2);
            let dist2 = DistanceFunction::Euclidean.compute(&v2, &v1);
            prop_assert!((dist1 - dist2).abs() < 1e-6, "Distance should be symmetric");
            Ok(())
        })
        .unwrap();
}

/// HNSW recall vs brute force - tests algorithmic correctness
/// Property: HNSW should achieve reasonable recall compared to brute force search
/// Note: HNSW is an approximate algorithm, so we use a 50% threshold which should
/// be easily met in normal cases while allowing for edge cases with unusual vector distributions
#[test]
fn test_hnsw_recall_vs_bruteforce() {
    use proptest::test_runner::{Config, TestRunner};

    let config = Config {
        cases: 20, // Fewer cases due to computational cost
        ..Config::default()
    };
    let mut runner = TestRunner::new(config);

    runner
        .run(
            &(
                prop::collection::vec(arb_vector(32), 100..200), // Larger dataset for more stable recall
                arb_vector(32),
                10usize..=20, // Larger k for more stable recall measurement
            ),
            |(vectors, query, k)| {
                let db = Database::in_memory();
                db.create_collection("test", 32).unwrap();
                let coll = db.collection("test").unwrap();

                // Insert all vectors (skip all-zero vectors which can cause edge cases)
                let mut inserted_count = 0;
                for (i, vec) in vectors.iter().enumerate() {
                    let norm: f32 = vec.iter().map(|x| x * x).sum();
                    if norm > 1e-6 {
                        // Skip near-zero vectors
                        coll.insert(format!("vec_{}", i), vec, None).unwrap();
                        inserted_count += 1;
                    }
                }

                // Need at least k+10 vectors for meaningful test
                if inserted_count < k + 10 {
                    return Ok(()); // Skip this test case
                }

                // Get HNSW results
                let hnsw_results: std::collections::HashSet<String> = coll
                    .search(&query, k)
                    .unwrap()
                    .into_iter()
                    .map(|r| r.id)
                    .collect();

                // Compute brute force results (only for inserted vectors)
                let mut brute_force: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| {
                        let norm: f32 = v.iter().map(|x| x * x).sum();
                        norm > 1e-6
                    })
                    .map(|(id, v)| (id, DistanceFunction::Euclidean.compute(&query, v)))
                    .collect();
                brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let brute_force_results: std::collections::HashSet<String> = brute_force
                    .iter()
                    .take(k)
                    .map(|(id, _)| format!("vec_{}", id))
                    .collect();

                // Calculate recall
                let intersection = hnsw_results.intersection(&brute_force_results).count();
                let recall = intersection as f64 / k as f64;

                // HNSW should achieve at least 50% recall on normal datasets
                // This is a conservative threshold that should pass on almost all cases
                // while still catching major algorithmic issues
                prop_assert!(
                    recall >= 0.5,
                    "HNSW recall {} is below threshold 0.5 (found {} of {} correct results)",
                    recall,
                    intersection,
                    k
                );
                Ok(())
            },
        )
        .unwrap();
}

/// Test concurrent read access doesn't cause issues
/// Property: Multiple concurrent searches should return consistent results
#[test]
fn test_concurrent_search_consistency() {
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 32).unwrap();

    // Insert vectors
    {
        let coll = db.collection("test").unwrap();
        for i in 0..100 {
            let vec: Vec<f32> = (0..32).map(|j| ((i * 32 + j) as f32) / 1000.0).collect();
            coll.insert(format!("vec_{}", i), &vec, None).unwrap();
        }
    }

    // Perform concurrent searches
    let query: Vec<f32> = (0..32).map(|i| i as f32 / 100.0).collect();
    let k = 10;

    let handles: Vec<_> = (0..8)
        .map(|_| {
            let db = Arc::clone(&db);
            let q = query.clone();
            thread::spawn(move || {
                let coll = db.collection("test").unwrap();
                let results = coll.search(&q, k).unwrap();
                results.into_iter().map(|r| r.id).collect::<Vec<_>>()
            })
        })
        .collect();

    // Collect all results
    let all_results: Vec<Vec<String>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All concurrent searches should return identical results
    let first = &all_results[0];
    for (i, result) in all_results.iter().enumerate().skip(1) {
        assert_eq!(
            first, result,
            "Concurrent search {} returned different results than search 0",
            i
        );
    }
}

/// Test concurrent insert and search doesn't cause data races
/// Property: Vectors inserted should eventually be findable
#[test]
fn test_concurrent_insert_search() {
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    let db = Arc::new(Database::in_memory());
    db.create_collection("test", 16).unwrap();

    // Pre-populate with some vectors
    {
        let coll = db.collection("test").unwrap();
        for i in 0..50 {
            let vec: Vec<f32> = (0..16).map(|j| ((i * 16 + j) as f32) / 1000.0).collect();
            coll.insert(format!("initial_{}", i), &vec, None).unwrap();
        }
    }

    let db_writer = Arc::clone(&db);
    let db_reader = Arc::clone(&db);

    // Writer thread - inserts new vectors
    let writer = thread::spawn(move || {
        let coll = db_writer.collection("test").unwrap();
        for i in 0..20 {
            let vec: Vec<f32> = (0..16)
                .map(|j| ((i * 16 + j + 1000) as f32) / 1000.0)
                .collect();
            coll.insert(format!("new_{}", i), &vec, None).unwrap();
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Reader thread - performs searches concurrently
    let reader = thread::spawn(move || {
        let coll = db_reader.collection("test").unwrap();
        let mut search_count = 0;
        for _ in 0..50 {
            let query: Vec<f32> = (0..16).map(|i| i as f32 / 100.0).collect();
            let results = coll.search(&query, 10);
            assert!(
                results.is_ok(),
                "Search should not fail during concurrent access"
            );
            search_count += 1;
            thread::sleep(Duration::from_micros(50));
        }
        search_count
    });

    writer.join().expect("Writer thread should not panic");
    let searches = reader.join().expect("Reader thread should not panic");

    assert!(searches > 0, "Should have completed some searches");

    // Verify all new vectors are findable after writes complete
    let coll = db.collection("test").unwrap();
    for i in 0..20 {
        let id = format!("new_{}", i);
        assert!(
            coll.contains(&id),
            "Vector {} should be findable after concurrent operations",
            id
        );
    }
}
