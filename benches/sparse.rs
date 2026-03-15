//! Sparse vector benchmarks for Needle Vector Database
//!
//! Benchmarks insertion, deletion, and search performance for sparse vectors
//! at various scales and sparsity levels.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use needle::{SparseIndex, SparseVector};
use rand::Rng;

// ============================================================================
// Helpers
// ============================================================================

/// Create a random sparse vector with a given number of non-zero entries
/// spread across `max_dim` dimensions.
fn random_sparse(nnz: usize, max_dim: u32) -> SparseVector {
    let mut rng = rand::thread_rng();
    let mut indices: Vec<u32> = (0..max_dim).collect();
    indices.sort_unstable();
    indices.truncate(nnz.min(max_dim as usize));
    // Shuffle and pick first nnz
    let mut selected: Vec<u32> = Vec::with_capacity(nnz);
    let mut pool: Vec<u32> = (0..max_dim).collect();
    for _ in 0..nnz.min(pool.len()) {
        let idx = rng.gen_range(0..pool.len());
        selected.push(pool.swap_remove(idx));
    }
    selected.sort_unstable();
    let values: Vec<f32> = selected.iter().map(|_| rng.gen::<f32>()).collect();
    SparseVector::new(selected, values)
}

fn build_index(n: usize, nnz: usize, max_dim: u32) -> SparseIndex {
    let mut index = SparseIndex::new();
    for _ in 0..n {
        index.insert(random_sparse(nnz, max_dim));
    }
    index
}

// ============================================================================
// Insert Benchmarks
// ============================================================================

fn bench_sparse_insert(c: &mut Criterion) {
    let max_dim = 30_000;
    let mut group = c.benchmark_group("sparse_insert");
    group.sample_size(20);

    for &(n, nnz) in &[(1_000, 50), (5_000, 50), (10_000, 100)] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), n),
            &(n, nnz),
            |bencher, &(n, nnz)| {
                let vectors: Vec<SparseVector> =
                    (0..n).map(|_| random_sparse(nnz, max_dim)).collect();
                bencher.iter(|| {
                    let mut index = SparseIndex::new();
                    for v in &vectors {
                        index.insert(v.clone());
                    }
                    black_box(&index);
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Search Benchmarks
// ============================================================================

fn bench_sparse_search_dot(c: &mut Criterion) {
    let max_dim = 30_000;
    let mut group = c.benchmark_group("sparse_search_dot");
    group.sample_size(30);

    for &(n, nnz) in &[(1_000, 50), (5_000, 100), (10_000, 100)] {
        let index = build_index(n, nnz, max_dim);
        let query = random_sparse(nnz, max_dim);

        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), n),
            &n,
            |bencher, _| {
                bencher.iter(|| black_box(index.search(&query, 10)));
            },
        );
    }

    group.finish();
}

fn bench_sparse_search_cosine(c: &mut Criterion) {
    let max_dim = 30_000;
    let mut group = c.benchmark_group("sparse_search_cosine");
    group.sample_size(30);

    for &(n, nnz) in &[(1_000, 50), (5_000, 100), (10_000, 100)] {
        let index = build_index(n, nnz, max_dim);
        let query = random_sparse(nnz, max_dim);

        group.bench_with_input(
            BenchmarkId::new(format!("n{n}_nnz{nnz}"), n),
            &n,
            |bencher, _| {
                bencher.iter(|| black_box(index.search_cosine(&query, 10)));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Sparsity Impact Benchmark
// ============================================================================

fn bench_sparse_sparsity_impact(c: &mut Criterion) {
    let max_dim = 30_000;
    let n = 5_000;
    let mut group = c.benchmark_group("sparse_sparsity_impact");
    group.sample_size(20);

    for &nnz in &[10, 50, 200, 500] {
        let index = build_index(n, nnz, max_dim);
        let query = random_sparse(nnz, max_dim);

        group.bench_with_input(BenchmarkId::new("nnz", nnz), &nnz, |bencher, _| {
            bencher.iter(|| black_box(index.search(&query, 10)));
        });
    }

    group.finish();
}

// ============================================================================
// Delete Benchmark
// ============================================================================

fn bench_sparse_delete(c: &mut Criterion) {
    let max_dim = 30_000;
    let nnz = 50;
    let mut group = c.benchmark_group("sparse_delete");
    group.sample_size(20);

    for &n in &[1_000, 5_000] {
        group.bench_with_input(BenchmarkId::new("n", n), &n, |bencher, &n| {
            bencher.iter_batched(
                || build_index(n, nnz, max_dim),
                |mut index| {
                    // Delete half the vectors
                    for id in 0..n / 2 {
                        index.remove(id);
                    }
                    black_box(&index);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_insert,
    bench_sparse_search_dot,
    bench_sparse_search_cosine,
    bench_sparse_sparsity_impact,
    bench_sparse_delete,
);
criterion_main!(benches);
