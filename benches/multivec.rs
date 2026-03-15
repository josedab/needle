//! Multi-vector benchmarks for Needle Vector Database
//!
//! Benchmarks insertion and search for ColBERT-style multi-vector retrieval.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use needle::{MultiVector, MultiVectorConfig, MultiVectorIndex};
use rand::Rng;

// ============================================================================
// Helpers
// ============================================================================

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn random_multivec(dim: usize, ntokens: usize) -> Vec<Vec<f32>> {
    (0..ntokens).map(|_| random_vector(dim)).collect()
}

fn build_index(n: usize, dim: usize, ntokens: usize) -> MultiVectorIndex {
    let mut index = MultiVectorIndex::with_dimensions(dim);
    for i in 0..n {
        let vecs = random_multivec(dim, ntokens);
        index
            .insert(MultiVector::new(format!("d{i}"), vecs))
            .unwrap();
    }
    index
}

// ============================================================================
// Insert Benchmarks
// ============================================================================

fn bench_multivec_insert(c: &mut Criterion) {
    let dim = 128;
    let ntokens = 32;
    let mut group = c.benchmark_group("multivec_insert");
    group.sample_size(10);

    for &n in &[100, 500, 1_000] {
        let docs: Vec<MultiVector> = (0..n)
            .map(|i| MultiVector::new(format!("d{i}"), random_multivec(dim, ntokens)))
            .collect();
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("docs", n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut index = MultiVectorIndex::with_dimensions(dim);
                for doc in &docs {
                    index.insert(doc.clone()).unwrap();
                }
                black_box(&index);
            });
        });
    }

    group.finish();
}

// ============================================================================
// Search Benchmarks
// ============================================================================

fn bench_multivec_search_brute(c: &mut Criterion) {
    let dim = 128;
    let ntokens = 32;
    let mut group = c.benchmark_group("multivec_search_brute");
    group.sample_size(20);

    for &n in &[100, 500, 1_000] {
        let index = build_index(n, dim, ntokens);
        let query = random_multivec(dim, 8);

        group.bench_with_input(BenchmarkId::new("docs", n), &n, |bencher, _| {
            bencher.iter(|| black_box(index.search(&query, 10)));
        });
    }

    group.finish();
}

fn bench_multivec_search_two_stage(c: &mut Criterion) {
    let dim = 128;
    let ntokens = 32;
    let mut group = c.benchmark_group("multivec_search_two_stage");
    group.sample_size(20);

    for &n in &[100, 500, 1_000] {
        let index = build_index(n, dim, ntokens);
        let query = random_multivec(dim, 8);

        group.bench_with_input(BenchmarkId::new("docs", n), &n, |bencher, _| {
            bencher.iter(|| black_box(index.search_two_stage(&query, 10, 4)));
        });
    }

    group.finish();
}

// ============================================================================
// Token Count Impact
// ============================================================================

fn bench_multivec_token_count(c: &mut Criterion) {
    let dim = 128;
    let n = 200;
    let mut group = c.benchmark_group("multivec_token_count");
    group.sample_size(20);

    for &ntokens in &[4, 16, 32, 64] {
        let index = build_index(n, dim, ntokens);
        let query = random_multivec(dim, 8);

        group.bench_with_input(
            BenchmarkId::new("tokens_per_doc", ntokens),
            &ntokens,
            |bencher, _| {
                bencher.iter(|| black_box(index.search(&query, 10)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multivec_insert,
    bench_multivec_search_brute,
    bench_multivec_search_two_stage,
    bench_multivec_token_count,
);
criterion_main!(benches);
