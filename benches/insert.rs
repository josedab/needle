//! Write/mutation benchmarks for Needle Vector Database
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use needle::{Collection, Database};
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
            |(mut collection, vector)| collection.insert("doc0", black_box(&vector), None).unwrap(),
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
                collection
                    .insert("new_doc", black_box(&vector), None)
                    .unwrap()
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
                collection
                    .insert("new_doc", black_box(&vector), None)
                    .unwrap()
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
                        collection
                            .insert(format!("doc{}", i), black_box(vec), None)
                            .unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Write Benchmarks - Concurrent Insert Throughput
// ============================================================================

fn bench_concurrent_insert_throughput(c: &mut Criterion) {
    use rayon::prelude::*;

    let dim = 128;
    let total_vectors = 10_000;

    let mut group = c.benchmark_group("concurrent_insert_throughput");
    group.sample_size(10);

    for &num_threads in &[1, 2, 4, 8] {
        let vectors = random_vectors(total_vectors, dim);

        group.throughput(Throughput::Elements(total_vectors as u64));
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            &num_threads,
            |bencher, &threads| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();

                bencher.iter(|| {
                    let db = Database::in_memory();
                    db.create_collection("bench", dim).unwrap();
                    let coll = db.collection("bench").unwrap();

                    pool.install(|| {
                        vectors
                            .par_iter()
                            .enumerate()
                            .for_each(|(i, vec)| {
                                coll.insert(
                                    format!("doc{}", i),
                                    black_box(vec),
                                    None,
                                )
                                .unwrap();
                            });
                    });
                });
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
                collection
                    .insert("doc5000", black_box(&new_vector), None)
                    .unwrap()
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
            |mut collection| collection.delete(black_box("doc5000")).unwrap(),
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
// Criterion Groups
// ============================================================================

criterion_group!(
    write_benches,
    bench_single_insert_latency,
    bench_batch_insert_throughput,
    bench_concurrent_insert_throughput,
    bench_update_performance,
    bench_delete_performance,
);

criterion_main!(write_benches);
