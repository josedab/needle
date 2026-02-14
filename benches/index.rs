//! Index benchmarks for Needle Vector Database
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use needle::quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
use needle::{Collection, Database, DistanceFunction, HnswConfig, HnswIndex};
use needle::ivf::{IvfConfig, IvfIndex};
use needle::diskann::{DiskAnnConfig, DiskAnnIndex};
use rand::Rng;

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
// Index Benchmarks - HNSW Build Time
// ============================================================================

fn bench_hnsw_build_time(c: &mut Criterion) {
    let dim = 128;

    let mut group = c.benchmark_group("hnsw_build_time");
    group.sample_size(20);

    for &n in &[1_000, 5_000, 10_000, 25_000] {
        let vectors = random_vectors(n, dim);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
                for (id, vec) in vectors.iter().enumerate() {
                    index.insert(id, vec, &vectors).unwrap();
                }
            })
        });
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
        group.bench_with_input(BenchmarkId::new("M", m), &m, |bencher, &m_val| {
            bencher.iter(|| {
                let config = HnswConfig::with_m(m_val);
                let mut index = HnswIndex::new(config, DistanceFunction::Euclidean);
                for (id, vec) in vectors.iter().enumerate() {
                    index.insert(id, vec, &vectors).unwrap();
                }
            })
        });
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
        group.bench_with_input(BenchmarkId::new("dimensions", dim), &dim, |bencher, _| {
            bencher.iter(|| {
                let mut collection = Collection::with_dimensions("bench", dim);
                for (i, vec) in vectors.iter().enumerate() {
                    collection.insert(format!("doc{}", i), vec, None).unwrap();
                }
            })
        });
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

        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bencher, _| {
            bencher.iter(|| DistanceFunction::Cosine.compute(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("euclidean", dim), &dim, |bencher, _| {
            bencher.iter(|| DistanceFunction::Euclidean.compute(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("dot_product", dim), &dim, |bencher, _| {
            bencher.iter(|| DistanceFunction::DotProduct.compute(black_box(&a), black_box(&b)))
        });
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
        group.bench_with_input(BenchmarkId::new("brute_force", n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut distances: Vec<(usize, f32)> = vectors
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, DistanceFunction::Euclidean.compute(&query, v).unwrap()))
                    .collect();
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                distances.truncate(10);
                distances
            })
        });

        // HNSW search
        let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
        for (id, vec) in vectors.iter().enumerate() {
            index.insert(id, vec, &vectors).unwrap();
        }

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |bencher, _| {
            bencher.iter(|| index.search(black_box(&query), 10, &vectors))
        });
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
            |(mut idx, new_vec)| idx.insert(n, black_box(&new_vec), &vectors).unwrap(),
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
                DistanceFunction::Euclidean,
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
            |(mut db, _tmp)| db.save().unwrap(),
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
        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut index = HnswIndex::with_distance(DistanceFunction::Euclidean);
                for (id, vec) in vectors.iter().enumerate() {
                    index.insert(id, vec, &vectors).unwrap();
                }
            })
        });

        // IVF build time (train + insert)
        let n_clusters = ((n as f64).sqrt() as usize).max(4);
        group.bench_with_input(BenchmarkId::new("ivf", n), &n, |bencher, _| {
            bencher.iter(|| {
                let config = IvfConfig::new(n_clusters);
                let mut index = IvfIndex::new(dim, config);
                index.train(&refs).unwrap();
                for (id, vec) in vectors.iter().enumerate() {
                    index.insert(id, vec).unwrap();
                }
            })
        });

        // DiskANN build time (includes filesystem I/O)
        group.bench_with_input(BenchmarkId::new("diskann", n), &n, |bencher, _| {
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
        });
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
            |bencher, &k_val| bencher.iter(|| ivf_index.search(black_box(&query), k_val)),
        );

        group.bench_with_input(
            BenchmarkId::new(format!("diskann_k{}", k), n),
            &k,
            |bencher, &k_val| bencher.iter(|| diskann_index.search(black_box(&query), k_val)),
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
        let mut diskann_index =
            DiskAnnIndex::create(diskann_dir.path(), dim, diskann_config).unwrap();
        for (id, vec) in vectors.iter().enumerate() {
            diskann_index.add(&format!("v{}", id), vec).unwrap();
        }
        diskann_index.build().unwrap();

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |bencher, _| {
            bencher.iter(|| hnsw_index.search(black_box(&query), 10, &vectors))
        });

        group.bench_with_input(BenchmarkId::new("ivf", n), &n, |bencher, _| {
            bencher.iter(|| ivf_index.search(black_box(&query), 10))
        });

        group.bench_with_input(BenchmarkId::new("diskann", n), &n, |bencher, _| {
            bencher.iter(|| diskann_index.search(black_box(&query), 10))
        });
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
        let mut diskann_index =
            DiskAnnIndex::create(diskann_dir.path(), dim, diskann_config).unwrap();
        for (id, vec) in vectors.iter().enumerate() {
            diskann_index.add(&format!("v{}", id), vec).unwrap();
        }
        diskann_index.build().unwrap();

        group.bench_with_input(BenchmarkId::new("hnsw", dim), &dim, |bencher, _| {
            bencher.iter(|| hnsw_index.search(black_box(&query), 10, &vectors))
        });

        group.bench_with_input(BenchmarkId::new("ivf", dim), &dim, |bencher, _| {
            bencher.iter(|| ivf_index.search(black_box(&query), 10))
        });

        group.bench_with_input(BenchmarkId::new("diskann", dim), &dim, |bencher, _| {
            bencher.iter(|| diskann_index.search(black_box(&query), 10))
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

criterion_group!(
    index_benches,
    bench_hnsw_build_time,
    bench_hnsw_build_parameters,
    bench_index_memory_efficiency,
    bench_recall_latency_tradeoff,
);

criterion_group!(
    supplementary_benches,
    bench_quantization_extended,
    bench_distance_by_dimension,
    bench_brute_force_baseline,
);

criterion_group!(optimization_benches, bench_optimization_validation,);

criterion_group!(
    index_comparison_benches,
    bench_index_build_comparison,
    bench_index_search_comparison,
    bench_index_scalability_comparison,
    bench_index_dimension_comparison,
);

criterion_main!(
    index_benches,
    supplementary_benches,
    optimization_benches,
    index_comparison_benches,
);
