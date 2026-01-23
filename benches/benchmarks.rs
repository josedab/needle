//! Benchmarks for Needle Vector Database
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use criterion::Throughput;
use needle::{Collection, Database, DistanceFunction, HnswConfig, HnswIndex, Filter};
use needle::{IvfConfig, IvfIndex, DiskAnnConfig, DiskAnnIndex};
use needle::quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
use needle::rag::{RagPipeline, RagConfig, ChunkingStrategy, MockEmbedder};
use needle::temporal::{TemporalIndex, TemporalConfig, DecayFunction};
use needle::encryption::{VectorEncryptor, EncryptionConfig, KeyManager};
use needle::{QueryAnalyzer, VisualQueryBuilder, CollectionProfile};
use needle::{Federation, FederationConfig, InstanceConfig, RoutingStrategy, MergeStrategy};
use needle::{DriftDetector, DriftConfig};
use needle::{BackupManager, BackupConfig};
use rand::Rng;
use serde_json::json;
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
// Search Benchmarks - Single Query Latency
// ============================================================================

fn bench_search_latency_by_size(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("search_latency_by_collection_size");
    group.sample_size(50);

    for &n in &[1_000, 10_000, 100_000] {
        let vectors = random_vectors(n, dim);
        let mut collection = Collection::with_dimensions("bench", dim);

        for (i, vec) in vectors.iter().enumerate() {
            collection.insert(format!("doc{}", i), vec, None).unwrap();
        }

        let query = random_vector(dim);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("vectors", n),
            &n,
            |bencher, _| {
                bencher.iter(|| collection.search(black_box(&query), 10))
            },
        );
    }

    group.finish();
}

fn bench_search_1m_vectors(c: &mut Criterion) {
    // Separate benchmark for 1M to avoid long setup in parametrized group
    let dim = 128;
    let n = 1_000_000;

    let mut group = c.benchmark_group("search_1m_vectors");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(30));

    let vectors = random_vectors(n, dim);
    let mut collection = Collection::with_dimensions("bench", dim);

    for (i, vec) in vectors.iter().enumerate() {
        collection.insert(format!("doc{}", i), vec, None).unwrap();
    }

    let query = random_vector(dim);

    group.bench_function("k10", |bencher| {
        bencher.iter(|| collection.search(black_box(&query), 10))
    });

    group.bench_function("k100", |bencher| {
        bencher.iter(|| collection.search(black_box(&query), 100))
    });

    group.finish();
}

// ============================================================================
// Search Benchmarks - Batch Search Throughput
// ============================================================================

fn bench_batch_search_throughput(c: &mut Criterion) {
    let dim = 128;
    let n = 50_000;
    let vectors = random_vectors(n, dim);

    let mut collection = Collection::with_dimensions("bench", dim);
    for (i, vec) in vectors.iter().enumerate() {
        collection.insert(format!("doc{}", i), vec, None).unwrap();
    }

    let mut group = c.benchmark_group("batch_search_throughput");

    for &batch_size in &[10, 50, 100, 500, 1000] {
        let queries = random_vectors(batch_size, dim);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| collection.batch_search(black_box(&queries), 10))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Search Benchmarks - Filtered Search Performance
// ============================================================================

fn bench_filtered_search_performance(c: &mut Criterion) {
    let dim = 128;
    let n = 50_000;
    let vectors = random_vectors(n, dim);

    let categories = ["A", "B", "C", "D", "E"];
    let regions = ["US", "EU", "APAC"];
    let mut collection = Collection::with_dimensions("bench", dim);
    for (i, vec) in vectors.iter().enumerate() {
        let metadata = json!({
            "category": categories[i % 5],
            "value": i as f64,
            "type": if i % 2 == 0 { "even" } else { "odd" },
            "region": regions[i % 3],
        });
        collection.insert(format!("doc{}", i), vec, Some(metadata)).unwrap();
    }

    let query = random_vector(dim);

    let mut group = c.benchmark_group("filtered_search");

    // Baseline: no filter
    group.bench_function("no_filter", |bencher| {
        bencher.iter(|| collection.search(black_box(&query), 10))
    });

    // Equality filter (20% selectivity)
    let eq_filter = Filter::eq("category", "A");
    group.bench_function("eq_filter_20pct", |bencher| {
        bencher.iter(|| collection.search_with_filter(black_box(&query), 10, &eq_filter))
    });

    // Range filter (~20% selectivity)
    let range_filter = Filter::and(vec![
        Filter::gte("value", 10000),
        Filter::lt("value", 20000),
    ]);
    group.bench_function("range_filter_20pct", |bencher| {
        bencher.iter(|| collection.search_with_filter(black_box(&query), 10, &range_filter))
    });

    // Compound filter (even + category A = 10% selectivity)
    let compound_filter = Filter::and(vec![
        Filter::eq("type", "even"),
        Filter::eq("category", "A"),
    ]);
    group.bench_function("compound_filter_10pct", |bencher| {
        bencher.iter(|| collection.search_with_filter(black_box(&query), 10, &compound_filter))
    });

    // OR filter (categories A or B = 40% selectivity)
    let or_filter = Filter::or(vec![
        Filter::eq("category", "A"),
        Filter::eq("category", "B"),
    ]);
    group.bench_function("or_filter_40pct", |bencher| {
        bencher.iter(|| collection.search_with_filter(black_box(&query), 10, &or_filter))
    });

    // Highly selective filter (1% selectivity)
    let selective_filter = Filter::and(vec![
        Filter::eq("category", "A"),
        Filter::eq("region", "US"),
        Filter::gte("value", 0),
        Filter::lt("value", 500),
    ]);
    group.bench_function("selective_filter_1pct", |bencher| {
        bencher.iter(|| collection.search_with_filter(black_box(&query), 10, &selective_filter))
    });

    group.finish();
}

// ============================================================================
// Search Benchmarks - Different Distance Metrics
// ============================================================================

fn bench_distance_metrics(c: &mut Criterion) {
    let dim = 384;
    let n = 10_000;

    let mut group = c.benchmark_group("distance_metrics_search");

    for distance in [
        DistanceFunction::Cosine,
        DistanceFunction::Euclidean,
        DistanceFunction::DotProduct,
        DistanceFunction::Manhattan,
    ] {
        let db = Database::in_memory();
        let config = needle::CollectionConfig::new("bench", dim).with_distance(distance);
        db.create_collection_with_config(config).unwrap();

        let collection = db.collection("bench").unwrap();
        let vectors = random_vectors(n, dim);

        for (i, vec) in vectors.iter().enumerate() {
            collection.insert(format!("doc{}", i), vec, None).unwrap();
        }

        let query = random_vector(dim);
        let metric_name = format!("{:?}", distance).to_lowercase();

        group.bench_function(&metric_name, |bencher| {
            bencher.iter(|| collection.search(black_box(&query), 10))
        });
    }

    group.finish();
}

// ============================================================================
// Write Benchmarks - Single Insert Latency
// ============================================================================

fn bench_single_insert_latency(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("single_insert_latency");

    // Insert into empty collection
    group.bench_function("empty_collection", |bencher| {
        bencher.iter_batched(
            || {
                let collection = Collection::with_dimensions("bench", dim);
                let vector = random_vector(dim);
                (collection, vector)
            },
            |(mut collection, vector)| {
                collection.insert("doc0", black_box(&vector), None).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Insert into populated collection (1K vectors)
    let base_vectors = random_vectors(1_000, dim);
    group.bench_function("1k_collection", |bencher| {
        bencher.iter_batched(
            || {
                let mut collection = Collection::with_dimensions("bench", dim);
                for (i, vec) in base_vectors.iter().enumerate() {
                    collection.insert(format!("doc{}", i), vec, None).unwrap();
                }
                let vector = random_vector(dim);
                (collection, vector)
            },
            |(mut collection, vector)| {
                collection.insert("new_doc", black_box(&vector), None).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Insert into larger collection (10K vectors)
    let large_vectors = random_vectors(10_000, dim);
    group.bench_function("10k_collection", |bencher| {
        bencher.iter_batched(
            || {
                let mut collection = Collection::with_dimensions("bench", dim);
                for (i, vec) in large_vectors.iter().enumerate() {
                    collection.insert(format!("doc{}", i), vec, None).unwrap();
                }
                let vector = random_vector(dim);
                (collection, vector)
            },
            |(mut collection, vector)| {
                collection.insert("new_doc", black_box(&vector), None).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Write Benchmarks - Batch Insert Throughput
// ============================================================================

fn bench_batch_insert_throughput(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("batch_insert_throughput");

    for &batch_size in &[100, 500, 1_000, 5_000, 10_000] {
        let vectors = random_vectors(batch_size, dim);

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |bencher, &size| {
                bencher.iter(|| {
                    let mut collection = Collection::with_dimensions("bench", dim);
                    for (i, vec) in vectors.iter().take(size).enumerate() {
                        collection.insert(format!("doc{}", i), black_box(vec), None).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Write Benchmarks - Update Performance
// ============================================================================

fn bench_update_performance(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let vectors = random_vectors(n, dim);

    let mut group = c.benchmark_group("update_performance");

    group.bench_function("update_existing_vector", |bencher| {
        bencher.iter_batched(
            || {
                let mut collection = Collection::with_dimensions("bench", dim);
                for (i, vec) in vectors.iter().enumerate() {
                    collection.insert(format!("doc{}", i), vec, None).unwrap();
                }
                let new_vector = random_vector(dim);
                (collection, new_vector)
            },
            |(mut collection, new_vector)| {
                // Update by delete + insert
                collection.delete("doc5000").unwrap();
                collection.insert("doc5000", black_box(&new_vector), None).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Write Benchmarks - Delete Performance
// ============================================================================

fn bench_delete_performance(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let vectors = random_vectors(n, dim);

    let mut group = c.benchmark_group("delete_performance");

    group.bench_function("delete_single", |bencher| {
        bencher.iter_batched(
            || {
                let mut collection = Collection::with_dimensions("bench", dim);
                for (i, vec) in vectors.iter().enumerate() {
                    collection.insert(format!("doc{}", i), vec, None).unwrap();
                }
                collection
            },
            |mut collection| {
                collection.delete(black_box("doc5000")).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("delete_batch_100", |bencher| {
        bencher.iter_batched(
            || {
                let mut collection = Collection::with_dimensions("bench", dim);
                for (i, vec) in vectors.iter().enumerate() {
                    collection.insert(format!("doc{}", i), vec, None).unwrap();
                }
                collection
            },
            |mut collection| {
                for i in 0..100 {
                    collection.delete(&format!("doc{}", i)).unwrap();
                }
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Index Benchmarks - HNSW Build Time
// ============================================================================

fn bench_hnsw_build_time(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("hnsw_build_time");
    group.sample_size(20);

    for &n in &[1_000, 5_000, 10_000, 25_000] {
        let vectors = random_vectors(n, dim);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("vectors", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
                    for (id, vec) in vectors.iter().enumerate() {
                        index.insert(id, vec, &vectors).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

fn bench_hnsw_build_parameters(c: &mut Criterion) {
    let dim = 128;
    let n = 5_000;
    let vectors = random_vectors(n, dim);

    let mut group = c.benchmark_group("hnsw_build_parameters");
    group.sample_size(10);

    // Different M values
    for &m in &[8, 16, 32, 48] {
        group.bench_with_input(
            BenchmarkId::new("M", m),
            &m,
            |bencher, &m_val| {
                bencher.iter(|| {
                    let config = HnswConfig::with_m(m_val);
                    let mut index = HnswIndex::new(config, DistanceFunction::Euclidean);
                    for (id, vec) in vectors.iter().enumerate() {
                        index.insert(id, vec, &vectors).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Index Benchmarks - Memory Usage (approximated via throughput)
// ============================================================================

fn bench_index_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_memory_efficiency");
    group.sample_size(10);

    // Compare build time at different dimensions (correlates with memory)
    let n = 5_000;
    for &dim in &[64, 128, 256, 384, 512, 768] {
        let vectors = random_vectors(n, dim);

        group.throughput(Throughput::Bytes((n * dim * 4) as u64));
        group.bench_with_input(
            BenchmarkId::new("dimensions", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| {
                    let mut collection = Collection::with_dimensions("bench", dim);
                    for (i, vec) in vectors.iter().enumerate() {
                        collection.insert(format!("doc{}", i), vec, None).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Index Benchmarks - Recall vs Latency Tradeoffs
// ============================================================================

fn bench_recall_latency_tradeoff(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let vectors = random_vectors(n, dim);

    let mut group = c.benchmark_group("recall_latency_tradeoff");

    // Build index once
    let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
    for (id, vec) in vectors.iter().enumerate() {
        index.insert(id, vec, &vectors).unwrap();
    }

    let query = random_vector(dim);

    // Different ef_search values (higher = better recall, slower)
    for &ef_search in &[10, 50, 100, 200, 500] {
        group.bench_with_input(
            BenchmarkId::new("ef_search", ef_search),
            &ef_search,
            |bencher, &ef| {
                bencher.iter(|| index.search_with_ef(black_box(&query), 10, ef, &vectors))
            },
        );
    }

    group.finish();
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
        pipeline.ingest_document(&format!("doc{}", i), doc, None, &embedder).unwrap();
    }

    group.bench_function("query_latency", |bencher| {
        bencher.iter(|| pipeline.query(black_box("What is machine learning?"), &embedder))
    });

    // Chunking strategies
    group.bench_function("chunking_fixed_size", |bencher| {
        let strategy = ChunkingStrategy::FixedSize { chunk_size: 100, overlap: 20 };
        let text = "A ".repeat(500);
        bencher.iter(|| {
            let db = Arc::new(Database::in_memory());
            let config = RagConfig { dimensions: dim, ..Default::default() };
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
        index.insert(&format!("doc{}", i), vec, timestamp, None).unwrap();
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
        ("exponential", DecayFunction::Exponential { half_life_seconds: 86400 }),
        ("linear", DecayFunction::Linear { max_age_seconds: 604800 }),
        ("gaussian", DecayFunction::Gaussian { scale_seconds: 172800 }),
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
                collection.insert(format!("doc{}", i), black_box(vec), None).unwrap();
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
                encryptor.encrypt(&format!("doc{}", i), black_box(vec), HashMap::new()).unwrap();
            }
        })
    });

    // Encrypt + Decrypt cycle
    let mut encryptor = VectorEncryptor::new(config.clone(), key_manager);
    encryptor.initialize(dim);
    let encrypted: Vec<_> = vectors.iter()
        .enumerate()
        .map(|(i, v)| encryptor.encrypt(&format!("doc{}", i), v, HashMap::new()).unwrap())
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
        bencher.iter(|| {
            encryptor.search_encrypted(black_box(&query), &encrypted, 10)
        })
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
                enc.encrypt(&format!("doc{}", i), black_box(vec), HashMap::new()).unwrap();
            }
        })
    });

    group.bench_function("searchable_encrypt", |bencher| {
        bencher.iter(|| {
            let km = KeyManager::new(&key).unwrap();
            let mut enc = VectorEncryptor::new(config.clone(), km);
            enc.initialize(dim);
            for (i, vec) in vectors.iter().take(100).enumerate() {
                enc.encrypt(&format!("doc{}", i), black_box(vec), HashMap::new()).unwrap();
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
        bencher.iter(|| analyzer.analyze(black_box("find similar documents about machine learning")))
    });

    group.bench_function("analyze_filter_query", |bencher| {
        bencher.iter(|| analyzer.analyze(black_box("category = 'books' AND price < 50")))
    });

    group.bench_function("analyze_hybrid_query", |bencher| {
        bencher.iter(|| analyzer.analyze(black_box("search for AI documents where category = 'tech' AND year > 2020")))
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
        bencher.iter(|| medium_builder.build(black_box("find products where price < 100 AND category = 'electronics'")))
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
                format!("http://localhost:{}", 8080 + i)
            ))
        })
    });

    // Health check benchmarks
    for count in [5, 10, 20] {
        let fed = Federation::new(FederationConfig::default());
        for i in 0..count {
            fed.register_instance(InstanceConfig::new(
                format!("inst-{}", i),
                format!("http://localhost:{}", 9000 + i)
            ));
        }

        group.bench_with_input(
            BenchmarkId::new("health_check", count),
            &count,
            |bencher, _| {
                bencher.iter(|| fed.health())
            },
        );
    }

    // Stats collection benchmarks
    group.bench_function("collect_stats", |bencher| {
        let fed = Federation::new(FederationConfig::default());
        for i in 0..10 {
            fed.register_instance(InstanceConfig::new(
                format!("stats-inst-{}", i),
                format!("http://localhost:{}", 7000 + i)
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
            |bencher, _| {
                bencher.iter(|| det.check(black_box(&query)))
            },
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
                            coll.insert(format!("vec_{}", i), &random_vector(dim), None).unwrap();
                        }

                        let config = BackupConfig::default();
                        let manager = BackupManager::new(&backup_dir, config);
                        (db, manager, temp_dir)
                    },
                    |(db, manager, _temp)| {
                        manager.create_backup(black_box(&db)).unwrap()
                    },
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
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None).unwrap();
                }

                let config = BackupConfig { compression: false, ..Default::default() };
                let manager = BackupManager::new(&backup_dir, config);
                (db, manager, temp_dir)
            },
            |(db, manager, _temp)| {
                manager.create_backup(black_box(&db)).unwrap()
            },
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
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None).unwrap();
                }

                let config = BackupConfig { compression: true, ..Default::default() };
                let manager = BackupManager::new(&backup_dir, config);
                (db, manager, temp_dir)
            },
            |(db, manager, _temp)| {
                manager.create_backup(black_box(&db)).unwrap()
            },
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
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None).unwrap();
                }

                let config = BackupConfig { verify: true, ..Default::default() };
                let manager = BackupManager::new(&backup_dir, config);
                let metadata = manager.create_backup(&db).unwrap();
                (manager, metadata.id, temp_dir)
            },
            |(manager, backup_id, _temp)| {
                manager.verify_backup(black_box(&backup_id)).unwrap()
            },
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
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None).unwrap();
                }

                let config = BackupConfig::default();
                let manager = BackupManager::new(&backup_dir, config);
                let metadata = manager.create_backup(&db).unwrap();
                (manager, metadata.id, temp_dir)
            },
            |(manager, backup_id, _temp)| {
                manager.restore_backup(black_box(&backup_id)).unwrap()
            },
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
                    coll.insert(format!("vec_{}", i), &random_vector(dim), None).unwrap();
                }

                let config = BackupConfig::default();
                let manager = BackupManager::new(&backup_dir, config);

                // Create multiple backups
                for _ in 0..5 {
                    manager.create_backup(&db).unwrap();
                }

                (manager, temp_dir)
            },
            |(manager, _temp)| {
                manager.list_backups().unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Quantization Benchmarks (Extended)
// ============================================================================

fn bench_quantization_extended(c: &mut Criterion) {
    let dim = 384;
    let n = 5_000;
    let vectors = random_vectors(n, dim);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let sq = ScalarQuantizer::train(&refs);
    let pq = ProductQuantizer::train(&refs, 8);
    let bq = BinaryQuantizer::train(&refs);

    let mut group = c.benchmark_group("quantization_extended");

    // Batch quantization throughput
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("scalar_batch_quantize", |bencher| {
        bencher.iter(|| {
            for vec in &vectors {
                sq.quantize(black_box(vec));
            }
        })
    });

    group.bench_function("product_batch_encode", |bencher| {
        bencher.iter(|| {
            for vec in &vectors {
                pq.encode(black_box(vec));
            }
        })
    });

    group.bench_function("binary_batch_quantize", |bencher| {
        bencher.iter(|| {
            for vec in &vectors {
                bq.quantize(black_box(vec));
            }
        })
    });

    // Quantized distance computation
    let q1 = bq.quantize(&vectors[0]);
    let q2 = bq.quantize(&vectors[1]);

    group.bench_function("binary_hamming_distance", |bencher| {
        bencher.iter(|| BinaryQuantizer::hamming_distance(black_box(&q1), black_box(&q2)))
    });

    group.finish();
}

// ============================================================================
// Distance Function Benchmarks (by dimension)
// ============================================================================

fn bench_distance_by_dimension(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_by_dimension");

    for &dim in &[64, 128, 256, 384, 512, 768, 1024, 1536] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.bench_with_input(
            BenchmarkId::new("cosine", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| DistanceFunction::Cosine.compute(black_box(&a), black_box(&b)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("euclidean", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| DistanceFunction::Euclidean.compute(black_box(&a), black_box(&b)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dot_product", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| DistanceFunction::DotProduct.compute(black_box(&a), black_box(&b)))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Baseline Comparison Benchmarks
// ============================================================================

fn bench_brute_force_baseline(c: &mut Criterion) {
    let dim = 128;
    let mut group = c.benchmark_group("baseline_comparison");

    for &n in &[1_000, 5_000, 10_000] {
        let vectors = random_vectors(n, dim);
        let query = random_vector(dim);

        // Brute force search
        group.bench_with_input(
            BenchmarkId::new("brute_force", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let mut distances: Vec<(usize, f32)> = vectors.iter()
                        .enumerate()
                        .map(|(i, v)| (i, DistanceFunction::Euclidean.compute(&query, v)))
                        .collect();
                    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    distances.truncate(10);
                    distances
                })
            },
        );

        // HNSW search
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("hnsw", n),
            &n,
            |bencher, _| {
                bencher.iter(|| index.search(black_box(&query), 10, &vectors))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Optimization Validation Benchmarks
// These benchmarks measure the specific operations that were optimized:
// - CRC32 checksum (storage.rs) - measured via file I/O
// - Vec<u8> visited tracking (hnsw.rs) - measured via search operations
// - sort_unstable_by (hnsw.rs) - measured via insert/search operations
// ============================================================================

fn bench_optimization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_validation");

    // Benchmark HNSW search - validates Vec<u8> visited tracking optimization
    let dim = 128;
    let n = 10_000;
    let vectors = random_vectors(n, dim);

    let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
    for (id, vec) in vectors.iter().enumerate() {
        index.insert(id, vec, &vectors).unwrap();
    }

    let query = random_vector(dim);

    group.bench_function("hnsw_search_10k_vectors", |bencher| {
        bencher.iter(|| index.search(black_box(&query), 10, &vectors))
    });

    // Benchmark HNSW insert - validates sort_unstable_by optimization
    group.bench_function("hnsw_insert_into_10k", |bencher| {
        bencher.iter_batched(
            || {
                let mut idx = HnswIndex::with_distance(DistanceFunction::Euclidean);
                for (id, vec) in vectors.iter().enumerate() {
                    idx.insert(id, vec, &vectors).unwrap();
                }
                (idx, random_vector(dim))
            },
            |(mut idx, new_vec)| {
                idx.insert(n, black_box(&new_vec), &vectors).unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark neighbor selection (sort_unstable_by optimization)
    // This is measured indirectly through insert operations with pruning
    let dense_vectors = random_vectors(1000, dim);
    group.bench_function("hnsw_build_1k_dense", |bencher| {
        bencher.iter(|| {
            let mut idx = HnswIndex::new(
                HnswConfig::with_m(32), // Higher M = more neighbor selection
                DistanceFunction::Euclidean
            );
            for (id, vec) in dense_vectors.iter().enumerate() {
                idx.insert(id, vec, &dense_vectors).unwrap();
            }
        })
    });

    // Benchmark storage operations (CRC32 lookup table optimization)
    // Measured via database persistence
    group.bench_function("database_save_1k_vectors", |bencher| {
        bencher.iter_batched(
            || {
                let tmp = tempfile::NamedTempFile::new().unwrap();
                let path = tmp.path().to_string_lossy().to_string();
                let db = Database::open(&path).unwrap();
                db.create_collection("test", dim).unwrap();
                let coll = db.collection("test").unwrap();
                for (i, vec) in dense_vectors.iter().enumerate() {
                    coll.insert(format!("doc{}", i), vec, None).unwrap();
                }
                (db, tmp)
            },
            |(mut db, _tmp)| {
                db.save().unwrap()
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Index Comparison Benchmarks - HNSW vs IVF vs DiskANN
// ============================================================================

/// Benchmark index build time across HNSW, IVF, and DiskANN.
///
/// Note: DiskANN requires filesystem I/O and includes that overhead.
/// For pure in-memory comparison, see HNSW vs IVF results.
fn bench_index_build_comparison(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("index_build_comparison");
    group.sample_size(10);

    for &n in &[1_000, 5_000, 10_000] {
        let vectors = random_vectors(n, dim);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // HNSW build time
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("hnsw", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
                    for (id, vec) in vectors.iter().enumerate() {
                        index.insert(id, vec, &vectors).unwrap();
                    }
                })
            },
        );

        // IVF build time (train + insert)
        let n_clusters = ((n as f64).sqrt() as usize).max(4);
        group.bench_with_input(
            BenchmarkId::new("ivf", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let config = IvfConfig::new(n_clusters);
                    let mut index = IvfIndex::new(dim, config);
                    index.train(&refs).unwrap();
                    for (id, vec) in vectors.iter().enumerate() {
                        index.insert(id, vec).unwrap();
                    }
                })
            },
        );

        // DiskANN build time (includes filesystem I/O)
        group.bench_with_input(
            BenchmarkId::new("diskann", n),
            &n,
            |bencher, _| {
                bencher.iter_batched(
                    || tempfile::tempdir().unwrap(),
                    |tmp_dir| {
                        let config = DiskAnnConfig::default();
                        let mut index = DiskAnnIndex::create(tmp_dir.path(), dim, config).unwrap();
                        for (id, vec) in vectors.iter().enumerate() {
                            index.add(&format!("v{}", id), vec).unwrap();
                        }
                        index.build().unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark search latency across HNSW, IVF, and DiskANN
fn bench_index_search_comparison(c: &mut Criterion) {
    let dim = 128;
    let n = 10_000;
    let vectors = random_vectors(n, dim);
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    let query = random_vector(dim);

    let mut group = c.benchmark_group("index_search_comparison");

    // Build HNSW index
    let mut hnsw_index = HnswIndex::with_distance(DistanceFunction::Euclidean);
    for (id, vec) in vectors.iter().enumerate() {
        hnsw_index.insert(id, vec, &vectors).unwrap();
    }

    // Build IVF index
    let n_clusters = ((n as f64).sqrt() as usize).max(4);
    let config = IvfConfig::new(n_clusters);
    let mut ivf_index = IvfIndex::new(dim, config);
    ivf_index.train(&refs).unwrap();
    for (id, vec) in vectors.iter().enumerate() {
        ivf_index.insert(id, vec).unwrap();
    }

    // Build DiskANN index (on disk)
    let diskann_dir = tempfile::tempdir().unwrap();
    let diskann_config = DiskAnnConfig::default();
    let mut diskann_index = DiskAnnIndex::create(diskann_dir.path(), dim, diskann_config).unwrap();
    for (id, vec) in vectors.iter().enumerate() {
        diskann_index.add(&format!("v{}", id), vec).unwrap();
    }
    diskann_index.build().unwrap();

    // Benchmark search for different k values
    for &k in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new(format!("hnsw_k{}", k), n),
            &k,
            |bencher, &k_val| {
                bencher.iter(|| hnsw_index.search(black_box(&query), k_val, &vectors))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("ivf_k{}", k), n),
            &k,
            |bencher, &k_val| {
                bencher.iter(|| ivf_index.search(black_box(&query), k_val))
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("diskann_k{}", k), n),
            &k,
            |bencher, &k_val| {
                bencher.iter(|| diskann_index.search(black_box(&query), k_val))
            },
        );
    }

    group.finish();
}

/// Benchmark search at different collection sizes (scalability)
fn bench_index_scalability_comparison(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("index_scalability_comparison");
    group.sample_size(20);

    for &n in &[1_000, 5_000, 10_000, 25_000] {
        let vectors = random_vectors(n, dim);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let query = random_vector(dim);

        // Build HNSW
        let mut hnsw_index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        for (id, vec) in vectors.iter().enumerate() {
            hnsw_index.insert(id, vec, &vectors).unwrap();
        }

        // Build IVF
        let n_clusters = ((n as f64).sqrt() as usize).max(4);
        let config = IvfConfig::new(n_clusters);
        let mut ivf_index = IvfIndex::new(dim, config);
        ivf_index.train(&refs).unwrap();
        for (id, vec) in vectors.iter().enumerate() {
            ivf_index.insert(id, vec).unwrap();
        }

        // Build DiskANN
        let diskann_dir = tempfile::tempdir().unwrap();
        let diskann_config = DiskAnnConfig::default();
        let mut diskann_index = DiskAnnIndex::create(diskann_dir.path(), dim, diskann_config).unwrap();
        for (id, vec) in vectors.iter().enumerate() {
            diskann_index.add(&format!("v{}", id), vec).unwrap();
        }
        diskann_index.build().unwrap();

        group.bench_with_input(
            BenchmarkId::new("hnsw", n),
            &n,
            |bencher, _| {
                bencher.iter(|| hnsw_index.search(black_box(&query), 10, &vectors))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ivf", n),
            &n,
            |bencher, _| {
                bencher.iter(|| ivf_index.search(black_box(&query), 10))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("diskann", n),
            &n,
            |bencher, _| {
                bencher.iter(|| diskann_index.search(black_box(&query), 10))
            },
        );
    }

    group.finish();
}

/// Benchmark index comparison at different dimensions
fn bench_index_dimension_comparison(c: &mut Criterion) {
    let n = 5_000;

    let mut group = c.benchmark_group("index_dimension_comparison");
    group.sample_size(15);

    for &dim in &[64, 128, 256, 384] {
        let vectors = random_vectors(n, dim);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let query = random_vector(dim);

        // Build HNSW
        let mut hnsw_index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        for (id, vec) in vectors.iter().enumerate() {
            hnsw_index.insert(id, vec, &vectors).unwrap();
        }

        // Build IVF
        let n_clusters = ((n as f64).sqrt() as usize).max(4);
        let config = IvfConfig::new(n_clusters);
        let mut ivf_index = IvfIndex::new(dim, config);
        ivf_index.train(&refs).unwrap();
        for (id, vec) in vectors.iter().enumerate() {
            ivf_index.insert(id, vec).unwrap();
        }

        // Build DiskANN
        let diskann_dir = tempfile::tempdir().unwrap();
        let diskann_config = DiskAnnConfig::default();
        let mut diskann_index = DiskAnnIndex::create(diskann_dir.path(), dim, diskann_config).unwrap();
        for (id, vec) in vectors.iter().enumerate() {
            diskann_index.add(&format!("v{}", id), vec).unwrap();
        }
        diskann_index.build().unwrap();

        group.bench_with_input(
            BenchmarkId::new("hnsw", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| hnsw_index.search(black_box(&query), 10, &vectors))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ivf", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| ivf_index.search(black_box(&query), 10))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("diskann", dim),
            &dim,
            |bencher, _| {
                bencher.iter(|| diskann_index.search(black_box(&query), 10))
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    search_benches,
    bench_search_latency_by_size,
    bench_batch_search_throughput,
    bench_filtered_search_performance,
    bench_distance_metrics,
);

criterion_group!(
    write_benches,
    bench_single_insert_latency,
    bench_batch_insert_throughput,
    bench_update_performance,
    bench_delete_performance,
);

criterion_group!(
    index_benches,
    bench_hnsw_build_time,
    bench_hnsw_build_parameters,
    bench_index_memory_efficiency,
    bench_recall_latency_tradeoff,
);

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

criterion_group!(
    supplementary_benches,
    bench_quantization_extended,
    bench_distance_by_dimension,
    bench_brute_force_baseline,
);

criterion_group!(
    optimization_benches,
    bench_optimization_validation,
);

criterion_group!(
    index_comparison_benches,
    bench_index_build_comparison,
    bench_index_search_comparison,
    bench_index_scalability_comparison,
    bench_index_dimension_comparison,
);

// Note: 1M vector benchmark is separate due to long runtime
criterion_group! {
    name = large_scale_benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(60));
    targets = bench_search_1m_vectors
}

criterion_main!(
    search_benches,
    write_benches,
    index_benches,
    advanced_benches,
    next_gen_benches,
    supplementary_benches,
    optimization_benches,
    index_comparison_benches,
    // Uncomment to run large-scale benchmarks (slow):
    // large_scale_benches,
);
