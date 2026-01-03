//! Benchmarks for Needle Vector Database
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use criterion::Throughput;
use needle::{Collection, Database, DistanceFunction, HnswConfig, HnswIndex, Filter};
use needle::quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
use needle::rag::{RagPipeline, RagConfig, ChunkingStrategy, MockEmbedder};
use needle::temporal::{TemporalIndex, TemporalConfig, DecayFunction};
use needle::encryption::{VectorEncryptor, EncryptionConfig, KeyManager};
use rand::Rng;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

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
    let documents = vec![
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
    supplementary_benches,
    bench_quantization_extended,
    bench_distance_by_dimension,
    bench_brute_force_baseline,
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
    supplementary_benches,
    // Uncomment to run large-scale benchmarks (slow):
    // large_scale_benches,
);
