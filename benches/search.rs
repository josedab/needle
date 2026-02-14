//! Search benchmarks for Needle Vector Database
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use needle::{Collection, Database, DistanceFunction, Filter, HnswIndex};
use rand::Rng;
use serde_json::json;

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
        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |bencher, _| {
            bencher.iter(|| collection.search(black_box(&query), 10))
        });
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
            |bencher, _| bencher.iter(|| collection.batch_search(black_box(&queries), 10)),
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
        collection
            .insert(format!("doc{}", i), vec, Some(metadata))
            .unwrap();
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
// Criterion Groups
// ============================================================================

criterion_group!(
    search_benches,
    bench_search_latency_by_size,
    bench_batch_search_throughput,
    bench_filtered_search_performance,
    bench_distance_metrics,
);

// Note: 1M vector benchmark is separate due to long runtime
criterion_group! {
    name = large_scale_benches;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(60));
    targets = bench_search_1m_vectors
}

criterion_main!(
    search_benches,
    // Uncomment to run large-scale benchmarks (slow):
    // large_scale_benches,
);
