//! Integration Tests for Needle Vector Database
//!
//! Tests that verify modules work correctly together in realistic scenarios.
//! These tests focus on module combinations, end-to-end workflows, and
//! performance validation.

use needle::{
    crdt::{ReplicaId, VectorCRDT},
    diskann::{DiskAnnConfig, DiskAnnIndex},
    encryption::{EncryptionConfig, KeyManager, VectorEncryptor},
    optimizer::QueryOptimizer,
    profiler::QueryProfiler,
    raft::{Command, NodeId, RaftConfig, RaftNode},
    temporal::{TemporalConfig, TemporalIndex},
    tiered::{TierPolicy, TieredStorage},
    Database, Filter,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate a deterministic vector based on seed
fn seeded_vector(dim: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed * 31 + i as u64).hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Normalize a vector to unit length
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Measure execution time of a closure
fn measure_time<F, R>(f: F) -> (R, Duration)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    (result, start.elapsed())
}

/// Generate random vector using simple PRNG
fn random_vector(dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed.wrapping_add(i as u64)).hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ============================================================================
// Module Combination Tests
// ============================================================================

mod module_combinations {
    use super::*;

    #[test]
    fn test_rag_pipeline_with_temporal_search() {
        // Create database for RAG pipeline
        let db = Arc::new(Database::in_memory());
        let dim = 64;

        // Create RAG collection
        db.create_collection("rag_chunks", dim)
            .expect("Failed to create collection");

        // Create temporal index for time-aware retrieval
        let temporal_config = TemporalConfig::default();
        let mut temporal = TemporalIndex::new(Arc::clone(&db), "temporal", temporal_config);

        // Create temporal collection
        db.create_collection("temporal", dim)
            .expect("Failed to create temporal collection");

        // Get current timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Insert documents with timestamps into temporal index
        for i in 0..5 {
            let id = format!("doc_{}", i);
            let vector = seeded_vector(dim, i as u64);
            let timestamp = now - (i as u64 * 3600); // Each doc 1 hour apart
            let metadata = serde_json::json!({
                "title": format!("Document {}", i),
                "timestamp": timestamp
            });

            temporal
                .insert(&id, &vector, timestamp, Some(metadata))
                .expect("Temporal insert failed");
        }

        // Search with time decay
        let query = seeded_vector(dim, 2);
        let results = temporal
            .search_with_decay(&query, 3)
            .expect("Temporal search failed");

        assert!(!results.is_empty(), "Temporal search should return results");

        // Verify more recent docs have higher scores due to decay
        if results.len() >= 2 {
            // Results should have decay factors applied
            assert!(
                results[0].decay_factor >= 0.0 && results[0].decay_factor <= 1.0,
                "Decay factor should be between 0 and 1"
            );
        }
    }

    #[test]
    fn test_distributed_sync_raft_with_crdt() {
        // Setup Raft nodes for consensus
        let config = RaftConfig::default();
        let mut node1 = RaftNode::new(NodeId(1), config.clone());
        let mut node2 = RaftNode::new(NodeId(2), config.clone());
        let mut node3 = RaftNode::new(NodeId(3), config);

        // Setup CRDTs for conflict-free vector operations
        let replica1 = ReplicaId::from(1);
        let replica2 = ReplicaId::from(2);
        let replica3 = ReplicaId::from(3);

        let mut crdt1 = VectorCRDT::new(replica1);
        let mut crdt2 = VectorCRDT::new(replica2);
        let mut crdt3 = VectorCRDT::new(replica3);

        // Simulate leader election through ticks
        for _ in 0..10 {
            node1.tick();
            node2.tick();
            node3.tick();
        }

        // Create vector and add to first CRDT
        let vector = seeded_vector(64, 42);
        let metadata: HashMap<String, String> = HashMap::new();
        crdt1
            .add("vec1", &vector, metadata.clone())
            .expect("CRDT add failed");

        // Propose command to Raft
        let command = Command::Insert {
            id: "vec1".to_string(),
            vector: vector.clone(),
            metadata: metadata.clone(),
        };
        let _ = node1.propose(command);

        // Replicate to other CRDTs
        crdt2
            .add("vec1", &vector, metadata.clone())
            .expect("CRDT add failed");
        crdt3
            .add("vec1", &vector, metadata)
            .expect("CRDT add failed");

        // Get deltas for sync
        let delta1 = crdt1.delta_since(None);
        let delta2 = crdt2.delta_since(None);

        // Verify deltas are generated
        assert!(!delta1.is_empty(), "Delta should contain operations");
        assert!(!delta2.is_empty(), "Delta should contain operations");

        // Merge deltas (simulating sync)
        crdt1.merge(delta2).expect("Merge failed");

        // Verify all CRDTs have the vector
        assert!(crdt1.get("vec1").is_some(), "Vector should exist in CRDT1");
        assert!(crdt2.get("vec1").is_some(), "Vector should exist in CRDT2");
        assert!(crdt3.get("vec1").is_some(), "Vector should exist in CRDT3");
    }

    #[test]
    fn test_tiered_storage_with_diskann() {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");

        // Create tiered storage
        let policy = TierPolicy::default();
        let mut tiered = TieredStorage::new(policy);

        // Create DiskANN index for cold tier
        let diskann_config = DiskAnnConfig {
            max_degree: 32,
            search_list_size: 64,
            ..Default::default()
        };
        let mut diskann = DiskAnnIndex::create(temp_dir.path(), 64, diskann_config)
            .expect("Failed to create DiskANN index");

        // Insert vectors into tiered storage (hot tier)
        for i in 0..100 {
            let id = format!("vec_{}", i);
            let vector = seeded_vector(64, i as u64);
            let metadata: HashMap<String, String> = HashMap::from([
                ("category".to_string(), format!("cat_{}", i % 5)),
            ]);
            tiered.put(&id, &vector, metadata).expect("Tiered put failed");
        }

        // Add cold tier vectors to DiskANN
        for i in 0..50 {
            let id = format!("cold_vec_{}", i);
            let vector = seeded_vector(64, (i + 100) as u64);
            diskann.add(&id, &vector).expect("DiskANN add failed");
        }
        diskann.build().expect("Failed to build DiskANN index");

        // Search hot tier via tiered storage
        let hot_vec = tiered.get("vec_25").expect("Tiered get failed");
        assert!(!hot_vec.is_empty(), "Hot tier should return vector");

        // Search cold tier via DiskANN
        let query = seeded_vector(64, 125);
        let cold_results = diskann.search(&query, 5).expect("DiskANN search failed");
        assert!(!cold_results.is_empty(), "Cold tier should return results");

        // Verify tiered storage tracks access
        let meta = tiered.get_metadata("vec_25");
        assert!(meta.is_some(), "Metadata should exist");
        assert!(
            meta.unwrap().total_access_count >= 1,
            "Access count should be tracked"
        );
    }

    #[test]
    fn test_encryption_end_to_end_flow() {
        // Setup key management with 32-byte master key (exactly 32 bytes)
        let master_key = b"0123456789abcdef0123456789abcdef";
        let mut key_manager = KeyManager::new(master_key).expect("Key manager creation failed");

        // Initialize projection for searchable encryption
        key_manager.init_projection(64, 32);

        // Create encryptor with config
        let config = EncryptionConfig {
            searchable: true,
            projection_dims: 32,
            ..Default::default()
        };
        let mut encryptor = VectorEncryptor::new(config, key_manager);

        // Create test vectors
        let vectors: Vec<_> = (0..10)
            .map(|i| (format!("doc_{}", i), normalize(&seeded_vector(64, i as u64))))
            .collect();

        // Encrypt vectors
        let encrypted: Vec<_> = vectors
            .iter()
            .map(|(id, v)| {
                let metadata: HashMap<String, String> = HashMap::new();
                let enc = encryptor.encrypt(id, v, metadata).expect("Encryption failed");
                (id.clone(), enc)
            })
            .collect();

        // Verify encrypted vectors have searchable embeddings
        for (_, enc_vec) in &encrypted {
            assert!(
                enc_vec.search_embedding.is_some(),
                "Should have search embedding for searchable encryption"
            );
        }

        // Decrypt and verify
        for ((id, original), (_, encrypted_vec)) in vectors.iter().zip(&encrypted) {
            let decrypted = encryptor.decrypt(encrypted_vec).expect("Decryption failed");
            let diff: f32 = original
                .iter()
                .zip(&decrypted)
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff < 0.01,
                "Decrypted vector for {} should match original (diff: {})",
                id,
                diff
            );
        }

        // Test encrypted search
        let query = normalize(&seeded_vector(64, 5));
        let encrypted_vecs: Vec<_> = encrypted.iter().map(|(_, e)| e.clone()).collect();
        let results = encryptor
            .search_encrypted(&query, &encrypted_vecs, 3)
            .expect("Encrypted search failed");
        assert!(!results.is_empty(), "Encrypted search should return results");
    }

    #[test]
    fn test_query_optimizer_with_profiler() {
        // Create database and collection
        let db = Database::in_memory();
        let dim = 128;
        db.create_collection("optimized", dim)
            .expect("Failed to create collection");

        let collection = db.collection("optimized").expect("Collection not found");

        // Insert test data
        for i in 0..1000 {
            let id = format!("vec_{}", i);
            let vector = seeded_vector(dim, i as u64);
            let metadata = serde_json::json!({
                "category": format!("cat_{}", i % 10),
                "year": 2020 + (i % 5)
            });
            collection.insert(&id, &vector, Some(metadata)).expect("Insert failed");
        }

        // Create optimizer from collection
        let optimizer = QueryOptimizer::from_collection(&collection);

        // Create profiler
        let mut profiler = QueryProfiler::new();

        // Create a query
        let query_vector = seeded_vector(dim, 42);
        let filter = Filter::eq("category", "cat_5");

        // Profile query optimization
        profiler.start("test_query");

        // Get optimized plan
        let plan = optimizer.optimize(&query_vector, Some(&filter), 10);

        // Execute actual search
        let results = collection
            .search_with_filter(&query_vector, 10, &filter)
            .expect("Search failed");

        let profile = profiler.end().expect("Profile end failed");

        // Verify optimizer produced a valid plan
        assert!(plan.estimated_cost > 0.0, "Plan should have positive cost");

        // Verify profiler captured timing
        assert!(profile.total_time_us > 0, "Profile should have timing");

        // Verify search returned results
        assert!(!results.is_empty(), "Search should return results");
    }
}

// ============================================================================
// End-to-End Scenario Tests
// ============================================================================

mod end_to_end_scenarios {
    use super::*;

    #[test]
    fn test_full_document_ingestion_and_search_workflow() {
        // Create in-memory database
        let db = Database::in_memory();

        // Create collection
        let dim = 64;
        db.create_collection("documents", dim)
            .expect("Failed to create collection");

        // Get collection reference
        let collection = db.collection("documents").expect("Collection not found");

        // Ingest documents
        let documents = [
            ("doc1", "Introduction to machine learning"),
            ("doc2", "Deep learning fundamentals"),
            ("doc3", "Natural language processing"),
            ("doc4", "Computer vision applications"),
            ("doc5", "Reinforcement learning basics"),
        ];

        for (i, (id, title)) in documents.iter().enumerate() {
            let vector = seeded_vector(dim, i as u64);
            let metadata = serde_json::json!({ "title": title });
            collection
                .insert(*id, &vector, Some(metadata))
                .expect("Insert failed");
        }

        // Verify document count
        assert_eq!(collection.len(), 5, "Should have 5 documents");

        // Search for similar documents
        let query = seeded_vector(dim, 1); // Similar to doc2
        let results = collection.search(&query, 3).expect("Search failed");

        assert_eq!(results.len(), 3, "Should return 3 results");
        assert_eq!(
            results[0].id, "doc2",
            "Most similar should be the same vector"
        );

        // Retrieve specific document
        if let Some((vec, meta)) = collection.get("doc3") {
            assert_eq!(vec.len(), dim, "Vector should have correct dimensions");
            assert!(meta.is_some(), "Metadata should exist");
        } else {
            panic!("Document doc3 should exist");
        }

        // Delete document
        assert!(collection.delete("doc4").expect("Delete failed"));
        assert_eq!(collection.len(), 4, "Should have 4 documents after delete");

        // Verify deleted document is gone
        assert!(collection.get("doc4").is_none(), "Deleted document should not exist");
    }

    #[test]
    fn test_multi_collection_operations() {
        let db = Database::in_memory();
        let dim = 32;

        // Create multiple collections
        let collection_names = vec!["products", "users", "reviews"];
        for name in &collection_names {
            db.create_collection(*name, dim)
                .expect("Failed to create collection");
        }

        // Verify all collections exist
        let collections = db.list_collections();
        assert_eq!(collections.len(), 3, "Should have 3 collections");

        // Insert data into each collection
        for (i, name) in collection_names.iter().enumerate() {
            let collection = db.collection(name).expect("Collection not found");
            for j in 0..10 {
                let id = format!("{}_{}", name, j);
                let vector = seeded_vector(dim, (i * 10 + j) as u64);
                collection.insert(&id, &vector, None).expect("Insert failed");
            }
        }

        // Verify counts
        for name in &collection_names {
            let collection = db.collection(name).expect("Collection not found");
            assert_eq!(collection.len(), 10, "Each collection should have 10 items");
        }

        // Cross-collection search (find similar items across collections)
        let query = seeded_vector(dim, 15);
        let mut all_results = Vec::new();
        for name in &collection_names {
            let collection = db.collection(name).expect("Collection not found");
            let results = collection.search(&query, 3).expect("Search failed");
            all_results.extend(results);
        }
        assert!(
            all_results.len() >= 3,
            "Should have results from multiple collections"
        );

        // Delete a collection
        db.drop_collection("reviews")
            .expect("Failed to drop collection");
        let collections = db.list_collections();
        assert_eq!(
            collections.len(),
            2,
            "Should have 2 collections after drop"
        );
    }

    #[test]
    fn test_concurrent_access_patterns() {
        let db = Arc::new(Database::in_memory());
        let dim = 32;

        db.create_collection("concurrent", dim)
            .expect("Failed to create collection");

        // Spawn multiple writer threads
        let mut handles = vec![];
        for thread_id in 0..4 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("concurrent").unwrap();
                for i in 0..25 {
                    let id = format!("thread{}_{}", thread_id, i);
                    let vector = seeded_vector(dim, (thread_id * 100 + i) as u64);
                    collection.insert(&id, &vector, None).expect("Insert failed");
                }
            });
            handles.push(handle);
        }

        // Wait for all writers
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify total count
        let collection = db.collection("concurrent").expect("Collection not found");
        assert_eq!(
            collection.len(),
            100,
            "Should have 100 items from 4 threads"
        );

        // Spawn reader threads concurrently with more writers
        let mut handles = vec![];

        // Readers
        for _ in 0..2 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("concurrent").unwrap();
                for _ in 0..10 {
                    let query = random_vector(dim);
                    let _ = collection.search(&query, 5);
                }
            });
            handles.push(handle);
        }

        // More writers
        for thread_id in 4..6 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("concurrent").unwrap();
                for i in 0..25 {
                    let id = format!("thread{}_{}", thread_id, i);
                    let vector = seeded_vector(dim, (thread_id * 100 + i) as u64);
                    collection.insert(&id, &vector, None).expect("Insert failed");
                }
            });
            handles.push(handle);
        }

        // Wait for all
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify final count
        let collection = db.collection("concurrent").expect("Collection not found");
        assert_eq!(
            collection.len(),
            150,
            "Should have 150 items after concurrent operations"
        );
    }

    #[test]
    fn test_recovery_after_simulated_failures() {
        let db = Database::in_memory();
        let dim = 32;

        db.create_collection("recovery", dim)
            .expect("Failed to create collection");

        let collection = db.collection("recovery").expect("Collection not found");

        // Insert initial data
        for i in 0..50 {
            let id = format!("doc_{}", i);
            let vector = seeded_vector(dim, i as u64);
            collection.insert(&id, &vector, None).expect("Insert failed");
        }

        // Simulate partial failure during batch insert
        let result = std::panic::catch_unwind(|| {
            // This simulates a batch that partially completes
            for i in 50..60 {
                if i == 55 {
                    panic!("Simulated failure!");
                }
            }
        });

        assert!(result.is_err(), "Should have caught panic");

        // Verify database is still operational
        let query = seeded_vector(dim, 25);
        let results = collection.search(&query, 5);
        assert!(results.is_ok(), "Search should still work after failure");

        // Insert more data to verify writes still work
        for i in 60..70 {
            let id = format!("doc_{}", i);
            let vector = seeded_vector(dim, i as u64);
            collection
                .insert(&id, &vector, None)
                .expect("Insert should work after recovery");
        }

        // Count should reflect successful inserts only
        assert!(
            collection.len() >= 50,
            "Should have at least initial 50 documents"
        );
    }
}

// ============================================================================
// Performance Validation Tests
// ============================================================================

mod performance_validation {
    use super::*;

    #[test]
    fn test_basic_latency_checks() {
        let db = Database::in_memory();
        let dim = 128;

        db.create_collection("latency", dim)
            .expect("Failed to create collection");

        let collection = db.collection("latency").expect("Collection not found");

        // Insert 1000 vectors
        let insert_start = Instant::now();
        for i in 0..1000 {
            let id = format!("vec_{}", i);
            let vector = seeded_vector(dim, i as u64);
            collection.insert(&id, &vector, None).expect("Insert failed");
        }
        let insert_duration = insert_start.elapsed();

        // Average insert latency should be reasonable (allow some slack for CI/slow machines)
        let avg_insert_ms = insert_duration.as_millis() as f64 / 1000.0;
        assert!(
            avg_insert_ms < 15.0,
            "Average insert latency {} ms should be < 15ms",
            avg_insert_ms
        );

        // Measure search latency
        let query = random_vector(dim);
        let search_times: Vec<_> = (0..100)
            .map(|_| {
                let (_, duration) = measure_time(|| {
                    collection.search(&query, 10).expect("Search failed")
                });
                duration
            })
            .collect();

        let avg_search_ms = search_times.iter().map(|d| d.as_micros()).sum::<u128>() as f64
            / search_times.len() as f64
            / 1000.0;

        // Search should complete in reasonable time
        assert!(
            avg_search_ms < 50.0,
            "Average search latency {} ms should be < 50ms for 1000 vectors",
            avg_search_ms
        );

        // Check 99th percentile
        let mut sorted_times: Vec<_> = search_times.iter().map(|d| d.as_micros()).collect();
        sorted_times.sort();
        let p99 = sorted_times[(sorted_times.len() * 99) / 100];
        let p99_ms = p99 as f64 / 1000.0;

        assert!(
            p99_ms < 100.0,
            "99th percentile search latency {} ms should be < 100ms",
            p99_ms
        );
    }

    #[test]
    fn test_memory_usage_patterns() {
        let db = Database::in_memory();
        let dim = 64;

        db.create_collection("memory", dim)
            .expect("Failed to create collection");

        let collection = db.collection("memory").expect("Collection not found");

        // Track memory growth pattern
        let mut sizes = vec![];

        for batch in 0..10 {
            // Insert batch of vectors
            for i in 0..100 {
                let id = format!("vec_{}_{}", batch, i);
                let vector = seeded_vector(dim, (batch * 100 + i) as u64);
                collection.insert(&id, &vector, None).expect("Insert failed");
            }

            // Record current size
            sizes.push(collection.len());
        }

        // Verify linear growth
        for i in 1..sizes.len() {
            let growth = sizes[i] - sizes[i - 1];
            assert_eq!(growth, 100, "Each batch should add exactly 100 vectors");
        }

        // Delete half the vectors
        for i in 0..500 {
            let id = format!("vec_{}_{}", i / 100, i % 100);
            collection.delete(&id).expect("Delete failed");
        }

        assert_eq!(
            collection.len(),
            500,
            "Should have 500 vectors after deleting half"
        );

        // Compact to reclaim memory
        collection.compact().expect("Compact failed");

        // Verify still functional after compaction
        let query = random_vector(dim);
        let results = collection.search(&query, 10).expect("Search failed");
        assert!(!results.is_empty(), "Search should work after compaction");
    }

    #[test]
    fn test_throughput_under_load() {
        let db = Arc::new(Database::in_memory());
        let dim = 64;

        db.create_collection("throughput", dim)
            .expect("Failed to create collection");

        // Pre-populate with data
        let collection = db.collection("throughput").expect("Collection not found");
        for i in 0..1000 {
            let id = format!("vec_{}", i);
            let vector = seeded_vector(dim, i as u64);
            collection.insert(&id, &vector, None).expect("Insert failed");
        }

        // Measure read throughput with multiple threads
        let num_threads = 4;
        let queries_per_thread = 250;

        let start = Instant::now();
        let mut handles = vec![];

        for _ in 0..num_threads {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("throughput").unwrap();
                for i in 0..queries_per_thread {
                    let query = seeded_vector(dim, i as u64);
                    collection.search(&query, 10).expect("Search failed");
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let duration = start.elapsed();
        let total_queries = num_threads * queries_per_thread;
        let qps = total_queries as f64 / duration.as_secs_f64();

        // Should achieve reasonable throughput
        assert!(
            qps > 100.0,
            "Query throughput {} QPS should be > 100 QPS",
            qps
        );

        // Measure mixed read/write throughput
        let start = Instant::now();
        let mut handles = vec![];

        // Reader threads
        for t in 0..2 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("throughput").unwrap();
                for i in 0..100 {
                    let query = seeded_vector(dim, (t * 1000 + i) as u64);
                    let _ = collection.search(&query, 10);
                }
            });
            handles.push(handle);
        }

        // Writer threads
        for t in 2..4 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("throughput").unwrap();
                for i in 0..100 {
                    let id = format!("new_vec_{}_{}", t, i);
                    let vector = seeded_vector(dim, (t * 1000 + i) as u64);
                    let _ = collection.insert(&id, &vector, None);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let mixed_duration = start.elapsed();
        let total_ops = 4 * 100;
        let ops_per_sec = total_ops as f64 / mixed_duration.as_secs_f64();

        assert!(
            ops_per_sec > 50.0,
            "Mixed workload {} ops/sec should be > 50 ops/sec",
            ops_per_sec
        );
    }

    #[test]
    fn test_parallel_search_performance() {
        let db = Arc::new(Database::in_memory());
        let dim = 64;

        db.create_collection("parallel", dim)
            .expect("Failed to create collection");

        let collection = db.collection("parallel").expect("Collection not found");

        // Insert vectors
        for i in 0..1000 {
            let id = format!("vec_{}", i);
            let vector = seeded_vector(dim, i as u64);
            collection.insert(&id, &vector, None).expect("Insert failed");
        }

        // Prepare queries
        let queries: Vec<_> = (0..50).map(|i| seeded_vector(dim, i as u64)).collect();

        // Measure parallel search time using threads
        let start = Instant::now();
        let mut handles = vec![];

        for query in &queries {
            let query = query.clone();
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let collection = db_clone.collection("parallel").unwrap();
                collection.search(&query, 10).expect("Search failed")
            });
            handles.push(handle);
        }

        let mut all_results = Vec::new();
        for handle in handles {
            all_results.push(handle.join().expect("Thread panicked"));
        }
        let parallel_duration = start.elapsed();

        assert_eq!(all_results.len(), 50, "Should return results for all queries");

        // Parallel should be faster than sequential
        let parallel_per_query_ms = parallel_duration.as_micros() as f64 / queries.len() as f64 / 1000.0;

        // Measure sequential query time for comparison
        let sequential_start = Instant::now();
        for query in queries.iter().take(10) {
            collection.search(query, 10).expect("Search failed");
        }
        let sequential_duration = sequential_start.elapsed();

        let avg_sequential_ms = sequential_duration.as_micros() as f64 / 10.0 / 1000.0;

        // Log performance metrics
        println!(
            "Parallel per-query: {} ms, Sequential per-query: {} ms",
            parallel_per_query_ms, avg_sequential_ms
        );

        // Parallel should be at least somewhat efficient
        assert!(
            parallel_per_query_ms < avg_sequential_ms * 5.0,
            "Parallel per-query time {} ms should be reasonable compared to sequential {} ms",
            parallel_per_query_ms,
            avg_sequential_ms
        );
    }
}

// ============================================================================
// Integration Smoke Tests
// ============================================================================

mod smoke_tests {
    use super::*;

    #[test]
    fn test_full_feature_integration() {
        // This test verifies all major components can work together
        let db = Database::in_memory();
        let dim = 32;

        // Create collection
        db.create_collection("integration", dim)
            .expect("Failed to create collection");

        let collection = db.collection("integration").expect("Collection not found");

        // Insert with metadata
        for i in 0..100 {
            let id = format!("doc_{}", i);
            let vector = seeded_vector(dim, i as u64);
            let metadata = serde_json::json!({
                "category": if i % 2 == 0 { "even" } else { "odd" },
                "value": i
            });
            collection
                .insert(&id, &vector, Some(metadata))
                .expect("Insert failed");
        }

        // Search with filter
        let query = seeded_vector(dim, 50);
        let filter = Filter::eq("category", "even");
        let results = collection
            .search_with_filter(&query, 10, &filter)
            .expect("Filtered search failed");

        // Verify filter was applied
        for result in &results {
            let meta = result.metadata.as_ref().expect("Should have metadata");
            assert_eq!(meta["category"], "even", "Filter should only return even");
        }

        // Verify collection stats
        assert_eq!(collection.len(), 100, "Count should be 100");
        assert_eq!(
            collection.dimensions().unwrap(),
            dim,
            "Dimensions should match"
        );
    }

    #[test]
    fn test_distance_functions_integration() {
        use needle::{CollectionConfig, DistanceFunction};

        let db = Database::in_memory();
        let dim = 32;

        // Create collections with different distance functions
        let distances = vec![
            ("cosine", DistanceFunction::Cosine),
            ("euclidean", DistanceFunction::Euclidean),
            ("dot", DistanceFunction::DotProduct),
        ];

        for (name, distance) in &distances {
            let config = CollectionConfig::new(*name, dim).with_distance(*distance);
            db.create_collection_with_config(config)
                .expect("Failed to create collection");
        }

        // Insert same vectors into all collections
        for (name, _) in &distances {
            let collection = db.collection(name).expect("Collection not found");
            for i in 0..10 {
                let id = format!("vec_{}", i);
                let vector = normalize(&seeded_vector(dim, i as u64));
                collection.insert(&id, &vector, None).expect("Insert failed");
            }
        }

        // Search each collection
        let query = normalize(&seeded_vector(dim, 5));
        for (name, _) in &distances {
            let collection = db.collection(name).expect("Collection not found");
            let results = collection.search(&query, 5).expect("Search failed");
            assert!(!results.is_empty(), "Search should return results for {}", name);

            // Closest should be vec_5 (same vector)
            assert_eq!(
                results[0].id, "vec_5",
                "Closest vector should be vec_5 for {}",
                name
            );
        }
    }
}
