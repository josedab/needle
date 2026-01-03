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
}

/// Test distance function symmetry
#[test]
fn test_distance_symmetry() {
    use proptest::test_runner::TestRunner;

    let mut runner = TestRunner::default();

    runner.run(&(arb_vector(32), arb_vector(32)), |(v1, v2)| {
        let dist1 = DistanceFunction::Euclidean.compute(&v1, &v2);
        let dist2 = DistanceFunction::Euclidean.compute(&v2, &v1);
        prop_assert!((dist1 - dist2).abs() < 1e-6, "Distance should be symmetric");
        Ok(())
    }).unwrap();
}
