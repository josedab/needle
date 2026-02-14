//! Advanced and next-gen benchmarks for Needle Vector Database
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use needle::encryption::{EncryptionConfig, KeyManager, VectorEncryptor};
use needle::rag::{ChunkingStrategy, MockEmbedder, RagConfig, RagPipeline};
use needle::temporal::{DecayFunction, TemporalConfig, TemporalIndex};
use needle::backup::{BackupConfig, BackupManager};
use needle::{Collection, Database};
use needle::query_builder::{CollectionProfile, QueryAnalyzer, VisualQueryBuilder};
use needle::drift::{DriftConfig, DriftDetector};
use needle::federated::{Federation, FederationConfig, InstanceConfig, MergeStrategy, RoutingStrategy};
use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Helper Functions
// ============================================================================

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| random_vector(dim)).collect()
}

#[allow(dead_code)]
fn normalized_vector(dim: usize) -> Vec<f32> {
    let mut v = random_vector(dim);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

// ============================================================================
// Advanced Features - RAG Pipeline Throughput
// ============================================================================

fn bench_rag_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("rag_pipeline");
    group.sample_size(20);

    let dim = 128;
    let db = Arc::new(Database::in_memory());
    let config = RagConfig {
        dimensions: dim,
        top_k: 5,
        rerank: false, // Disable for cleaner benchmark
        ..Default::default()
    };

    let mut pipeline = RagPipeline::new(db.clone(), config).unwrap();
    let embedder = MockEmbedder::new(dim);

    // Ingest documents
    let documents = [
        "Machine learning is a subset of artificial intelligence focused on building systems that learn from data.",
        "Deep learning uses neural networks with many layers to model complex patterns in large amounts of data.",
        "Natural language processing enables computers to understand, interpret, and generate human language.",
        "Computer vision is a field that enables machines to interpret and understand visual information.",
        "Reinforcement learning is a type of machine learning where agents learn by interacting with an environment.",
    ];

    for (i, doc) in documents.iter().enumerate() {
        pipeline
            .ingest_document(&format!("doc{}", i), doc, None, &embedder)
            .unwrap();
    }

    group.bench_function("query_latency", |bencher| {
        bencher.iter(|| pipeline.query(black_box("What is machine learning?"), &embedder))
    });

    // Chunking strategies
    group.bench_function("chunking_fixed_size", |bencher| {
        let strategy = ChunkingStrategy::FixedSize {
            chunk_size: 100,
            overlap: 20,
        };
        let text = "A ".repeat(500);
        bencher.iter(|| {
            let db = Arc::new(Database::in_memory());
            let config = RagConfig {
                dimensions: dim,
                ..Default::default()
            };
            let mut p = RagPipeline::new(db, config).unwrap();
            p.ingest_with_strategy("test", black_box(&text), None, &strategy, &embedder)
        })
    });

    group.bench_function("chunking_semantic", |bencher| {
        let strategy = ChunkingStrategy::Semantic { max_chunk_size: 200, min_chunk_size: 50 };
        let text = "This is sentence one. This is sentence two. Another sentence here. And more text follows.".repeat(10);
        bencher.iter(|| {
            let db = Arc::new(Database::in_memory());
            let config = RagConfig { dimensions: dim, ..Default::default() };
            let mut p = RagPipeline::new(db, config).unwrap();
            p.ingest_with_strategy("test", black_box(&text), None, &strategy, &embedder)
        })
    });

    group.finish();
}

// ============================================================================
// Advanced Features - Temporal Search Overhead
// ============================================================================

fn bench_temporal_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_search");
    group.sample_size(30);

    let dim = 128;
    let n = 5_000;
    let db = Arc::new(Database::in_memory());
    db.create_collection("temporal", dim).unwrap();

    let config = TemporalConfig::default();
    let mut index = TemporalIndex::new(db.clone(), "temporal", config);

    // Insert vectors with timestamps
    let base_time = 1700000000u64;
    let vectors = random_vectors(n, dim);
    for (i, vec) in vectors.iter().enumerate() {
        let timestamp = base_time + (i as u64 * 3600); // 1 hour apart
        index
            .insert(&format!("doc{}", i), vec, timestamp, None)
            .unwrap();
    }

    let query = random_vector(dim);

    // Baseline: regular search (via underlying collection)
    let collection = db.collection("temporal").unwrap();
    group.bench_function("baseline_search", |bencher| {
        bencher.iter(|| collection.search(black_box(&query), 10))
    });

    // Temporal search with decay
    group.bench_function("search_with_decay", |bencher| {
        bencher.iter(|| index.search_with_decay(black_box(&query), 10))
    });

    // Temporal search in time range
    let start = base_time + 1000 * 3600;
    let end = base_time + 2000 * 3600;
    group.bench_function("search_in_range", |bencher| {
        bencher.iter(|| index.search_in_range(black_box(&query), 10, start, end))
    });

    // Different decay functions
    for decay in [
        (
            "exponential",
            DecayFunction::Exponential {
                half_life_seconds: 86400,
            },
        ),
        (
            "linear",
            DecayFunction::Linear {
                max_age_seconds: 604800,
            },
        ),
        (
            "gaussian",
            DecayFunction::Gaussian {
                scale_seconds: 172800,
            },
        ),
    ] {
        let decay_config = TemporalConfig {
            decay: decay.1,
            ..Default::default()
        };
        let decay_index = TemporalIndex::new(db.clone(), "temporal", decay_config);

        group.bench_function(format!("decay_{}", decay.0), |bencher| {
            bencher.iter(|| decay_index.search_with_decay(black_box(&query), 10))
        });
    }

    group.finish();
}

// ============================================================================
// Advanced Features - Encryption Overhead
// ============================================================================

fn bench_encryption_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("encryption_overhead");

    let dim = 128;
    let n = 1_000;
    let vectors = random_vectors(n, dim);

    // Baseline: unencrypted operations
    group.bench_function("baseline_insert", |bencher| {
        bencher.iter(|| {
            let mut collection = Collection::with_dimensions("bench", dim);
            for (i, vec) in vectors.iter().enumerate() {
                collection
                    .insert(format!("doc{}", i), black_box(vec), None)
                    .unwrap();
            }
        })
    });

    // Encrypted insert
    let key = vec![0u8; 32];
    let key_manager = KeyManager::new(&key).unwrap();
    let config = EncryptionConfig::default();

    group.bench_function("encrypted_insert", |bencher| {
        bencher.iter(|| {
            let km = KeyManager::new(&key).unwrap();
            let mut encryptor = VectorEncryptor::new(config.clone(), km);
            encryptor.initialize(dim);

            for (i, vec) in vectors.iter().enumerate() {
                encryptor
                    .encrypt(&format!("doc{}", i), black_box(vec), HashMap::new())
                    .unwrap();
            }
        })
    });

    // Encrypt + Decrypt cycle
    let mut encryptor = VectorEncryptor::new(config.clone(), key_manager);
    encryptor.initialize(dim);
    let encrypted: Vec<_> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            encryptor
                .encrypt(&format!("doc{}", i), v, HashMap::new())
                .unwrap()
        })
        .collect();

    group.bench_function("decrypt_batch", |bencher| {
        bencher.iter(|| {
            for enc in &encrypted {
                encryptor.decrypt(black_box(enc)).unwrap();
            }
        })
    });

    // Encrypted search
    let query = random_vector(dim);
    group.bench_function("encrypted_search", |bencher| {
        bencher.iter(|| encryptor.search_encrypted(black_box(&query), &encrypted, 10))
    });

    // Compare searchable vs non-searchable encryption
    let non_searchable_config = EncryptionConfig {
        searchable: false,
        ..Default::default()
    };

    group.bench_function("non_searchable_encrypt", |bencher| {
        bencher.iter(|| {
            let km = KeyManager::new(&key).unwrap();
            let mut enc = VectorEncryptor::new(non_searchable_config.clone(), km);
            for (i, vec) in vectors.iter().take(100).enumerate() {
                enc.encrypt(&format!("doc{}", i), black_box(vec), HashMap::new())
                    .unwrap();
            }
        })
    });

    group.bench_function("searchable_encrypt", |bencher| {
        bencher.iter(|| {
            let km = KeyManager::new(&key).unwrap();
            let mut enc = VectorEncryptor::new(config.clone(), km);
            enc.initialize(dim);
            for (i, vec) in vectors.iter().take(100).enumerate() {
                enc.encrypt(&format!("doc{}", i), black_box(vec), HashMap::new())
                    .unwrap();
            }
        })
    });

    group.finish();
}

// ============================================================================
// Advanced Features - Query Builder Performance
// ============================================================================

fn bench_query_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_builder");

    // Query Analyzer benchmarks
    let analyzer = QueryAnalyzer::new();

    group.bench_function("analyze_semantic_query", |bencher| {
        bencher
            .iter(|| analyzer.analyze(black_box("find similar documents about machine learning")))
    });

    group.bench_function("analyze_filter_query", |bencher| {
        bencher.iter(|| analyzer.analyze(black_box("category = 'books' AND price < 50")))
    });

    group.bench_function("analyze_hybrid_query", |bencher| {
        bencher.iter(|| {
            analyzer.analyze(black_box(
                "search for AI documents where category = 'tech' AND year > 2020",
            ))
        })
    });

    group.bench_function("analyze_complex_query", |bencher| {
        bencher.iter(|| analyzer.analyze(black_box(
            "find all documents similar to 'neural networks' where status = 'active' AND category IN ('ml', 'ai', 'data') AND score >= 0.8"
        )))
    });

    // Visual Query Builder benchmarks
    let small_profile = CollectionProfile::new("small", 128, 1_000);
    let medium_profile = CollectionProfile::new("medium", 384, 100_000);
    let large_profile = CollectionProfile::new("large", 768, 1_000_000);

    let small_builder = VisualQueryBuilder::new(small_profile);
    let medium_builder = VisualQueryBuilder::new(medium_profile);
    let large_builder = VisualQueryBuilder::new(large_profile);

    group.bench_function("build_simple_query_small", |bencher| {
        bencher.iter(|| small_builder.build(black_box("find similar documents")))
    });

    group.bench_function("build_simple_query_large", |bencher| {
        bencher.iter(|| large_builder.build(black_box("find similar documents")))
    });

    group.bench_function("build_filtered_query", |bencher| {
        bencher.iter(|| {
            medium_builder.build(black_box(
                "find products where price < 100 AND category = 'electronics'",
            ))
        })
    });

    group.bench_function("build_complex_query", |bencher| {
        bencher.iter(|| large_builder.build(black_box(
            "find all active documents similar to 'deep learning' where category IN ('ml', 'ai') AND score > 0.7 ORDER BY date DESC LIMIT 50"
        )))
    });

    group.finish();
}

// ============================================================================
// Advanced Features - Federated Search Configuration
// ============================================================================

fn bench_federated_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("federated_search");

    // Federation configuration benchmarks
    group.bench_function("create_federation_default", |bencher| {
        bencher.iter(|| {
            let config = FederationConfig::default();
            Federation::new(config)
        })
    });

    group.bench_function("create_federation_configured", |bencher| {
        bencher.iter(|| {
            let config = FederationConfig::default()
                .with_routing(RoutingStrategy::LatencyAware)
                .with_merge(MergeStrategy::ReciprocalRankFusion)
                .with_timeout(Duration::from_millis(5000));
            Federation::new(config)
        })
    });

    // Instance registration benchmarks
    let config = FederationConfig::default();
    let federation = Federation::new(config);

    group.bench_function("register_instance", |bencher| {
        let mut i = 0;
        bencher.iter(|| {
            i += 1;
            federation.register_instance(InstanceConfig::new(
                format!("instance-{}", i),
                format!("http://localhost:{}", 8080 + i),
            ))
        })
    });

    // Health check benchmarks
    for count in [5, 10, 20] {
        let fed = Federation::new(FederationConfig::default());
        for i in 0..count {
            fed.register_instance(InstanceConfig::new(
                format!("inst-{}", i),
                format!("http://localhost:{}", 9000 + i),
            ));
        }

        group.bench_with_input(
            BenchmarkId::new("health_check", count),
            &count,
            |bencher, _| bencher.iter(|| fed.health()),
        );
    }

    // Stats collection benchmarks
    group.bench_function("collect_stats", |bencher| {
        let fed = Federation::new(FederationConfig::default());
        for i in 0..10 {
            fed.register_instance(InstanceConfig::new(
                format!("stats-inst-{}", i),
                format!("http://localhost:{}", 7000 + i),
            ));
        }
        bencher.iter(|| fed.stats())
    });

    group.finish();
}

// ============================================================================
// Advanced Features - Drift Detection Performance
// ============================================================================

fn bench_drift_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("drift_detection");
    group.sample_size(30);

    let dim = 128;

    // Baseline creation benchmarks
    for &baseline_size in &[50, 100, 500, 1000] {
        let baseline: Vec<Vec<f32>> = (0..baseline_size).map(|_| random_vector(dim)).collect();

        group.bench_with_input(
            BenchmarkId::new("add_baseline", baseline_size),
            &baseline_size,
            |bencher, _| {
                bencher.iter(|| {
                    let config = DriftConfig::default();
                    let mut detector = DriftDetector::new(dim, config);
                    detector.add_baseline(black_box(&baseline)).unwrap()
                })
            },
        );
    }

    // Drift check latency
    let baseline: Vec<Vec<f32>> = (0..100).map(|_| random_vector(dim)).collect();
    let config = DriftConfig::default();
    let mut detector = DriftDetector::new(dim, config);
    detector.add_baseline(&baseline).unwrap();

    group.bench_function("check_single_vector", |bencher| {
        let query = random_vector(dim);
        bencher.iter(|| detector.check(black_box(&query)))
    });

    // Batch drift checking
    for &batch_size in &[10, 50, 100, 500] {
        let vectors: Vec<Vec<f32>> = (0..batch_size).map(|_| random_vector(dim)).collect();

        group.bench_with_input(
            BenchmarkId::new("check_batch", batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    for v in &vectors {
                        let _ = detector.check(black_box(v));
                    }
                })
            },
        );
    }

    // Drift detection at different dimensions
    for &d in &[64, 128, 256, 384] {
        let baseline: Vec<Vec<f32>> = (0..100).map(|_| random_vector(d)).collect();
        let config = DriftConfig::default();
        let mut det = DriftDetector::new(d, config);
        det.add_baseline(&baseline).unwrap();
        let query = random_vector(d);

        group.bench_with_input(
            BenchmarkId::new("check_by_dimension", d),
            &d,
            |bencher, _| bencher.iter(|| det.check(black_box(&query))),
        );
    }

    group.finish();
}

// ============================================================================
// Advanced Features - Backup/Restore Performance
// ============================================================================

fn bench_backup_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("backup_restore");
    group.sample_size(10);

    let dim = 128;

    // Backup creation benchmarks at different sizes
    for &n in &[100, 500, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("create_backup", n),
            &n,
            |bencher, &size| {
                bencher.iter_batched(
                    || {
                        let temp_dir = tempfile::tempdir().unwrap();
                        let backup_dir = temp_dir.path().join("backups");

                        let db = Database::in_memory();
                        db.create_collection("bench", dim).unwrap();
                        let coll = db.collection("bench").unwrap();

                        for i in 0..size {
                            coll.insert(format!("vec_{}", i), &random_vector(dim), None)
                                .unwrap();
                        }

                        let config = BackupConfig::default();
                        let manager = BackupManager::new(&backup_dir, config);
                        (db, manager, temp_dir)
                    },
                    |(db, manager, _temp)| manager.create_backup(black_box(&db)).unwrap(),
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    // Backup with compression vs without
    let n = 1000;
    group.bench_function("backup_uncompressed", |bencher| {
        bencher.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let backup_dir = temp_dir.path().join("backups");

                let db = Database::in_memory();
                db.create_collection("bench", dim).unwrap();
                let coll = db.collection("bench").unwrap();

                for i in 0..n {
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None)
                        .unwrap();
                }

                let config = BackupConfig {
                    compression: false,
                    ..Default::default()
                };
                let manager = BackupManager::new(&backup_dir, config);
                (db, manager, temp_dir)
            },
            |(db, manager, _temp)| manager.create_backup(black_box(&db)).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("backup_compressed", |bencher| {
        bencher.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let backup_dir = temp_dir.path().join("backups");

                let db = Database::in_memory();
                db.create_collection("bench", dim).unwrap();
                let coll = db.collection("bench").unwrap();

                for i in 0..n {
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None)
                        .unwrap();
                }

                let config = BackupConfig {
                    compression: true,
                    ..Default::default()
                };
                let manager = BackupManager::new(&backup_dir, config);
                (db, manager, temp_dir)
            },
            |(db, manager, _temp)| manager.create_backup(black_box(&db)).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });

    // Backup verification
    group.bench_function("verify_backup", |bencher| {
        bencher.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let backup_dir = temp_dir.path().join("backups");

                let db = Database::in_memory();
                db.create_collection("bench", dim).unwrap();
                let coll = db.collection("bench").unwrap();

                for i in 0..500 {
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None)
                        .unwrap();
                }

                let config = BackupConfig {
                    verify: true,
                    ..Default::default()
                };
                let manager = BackupManager::new(&backup_dir, config);
                let metadata = manager.create_backup(&db).unwrap();
                (manager, metadata.id, temp_dir)
            },
            |(manager, backup_id, _temp)| manager.verify_backup(black_box(&backup_id)).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });

    // Restore benchmarks
    group.bench_function("restore_backup_1000", |bencher| {
        bencher.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let backup_dir = temp_dir.path().join("backups");

                let db = Database::in_memory();
                db.create_collection("bench", dim).unwrap();
                let coll = db.collection("bench").unwrap();

                for i in 0..1000 {
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None)
                        .unwrap();
                }

                let config = BackupConfig::default();
                let manager = BackupManager::new(&backup_dir, config);
                let metadata = manager.create_backup(&db).unwrap();
                (manager, metadata.id, temp_dir)
            },
            |(manager, backup_id, _temp)| manager.restore_backup(black_box(&backup_id)).unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });

    // List backups
    group.bench_function("list_backups", |bencher| {
        bencher.iter_batched(
            || {
                let temp_dir = tempfile::tempdir().unwrap();
                let backup_dir = temp_dir.path().join("backups");

                let db = Database::in_memory();
                db.create_collection("bench", dim).unwrap();
                let coll = db.collection("bench").unwrap();

                for i in 0..100 {
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None)
                        .unwrap();
                }

                let config = BackupConfig::default();
                let manager = BackupManager::new(&backup_dir, config);

                // Create multiple backups
                for _ in 0..5 {
                    manager.create_backup(&db).unwrap();
                }

                (manager, temp_dir)
            },
            |(manager, _temp)| manager.list_backups().unwrap(),
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    advanced_benches,
    bench_rag_pipeline,
    bench_temporal_search,
    bench_encryption_overhead,
);

criterion_group!(
    next_gen_benches,
    bench_query_builder,
    bench_federated_search,
    bench_drift_detection,
    bench_backup_restore,
);

criterion_main!(advanced_benches, next_gen_benches);
