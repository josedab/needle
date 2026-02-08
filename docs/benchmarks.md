# Needle Benchmarks

Performance comparisons for Needle's vector search operations. All benchmarks run on
a single thread unless noted otherwise. Results are from an Apple M2 Pro with 32 GB RAM.

## Methodology

- **Dataset**: Random float32 vectors, uniformly distributed in \[0, 1\]
- **Distance**: Cosine similarity (default)
- **Recall@10**: Fraction of true top-10 neighbors returned by ANN search
- **Latency**: p50 / p99 over 1 000 queries after warm-up
- **Throughput**: Queries per second (QPS) at the stated recall level

To reproduce locally:

```bash
cargo bench                      # Run all Criterion benchmarks
cargo bench -- search            # Run only search benchmarks
cargo bench -- insert            # Run only insert benchmarks
```

## Single-Node Latency (k = 10)

| Dataset (dim) | Vectors | Recall@10 | p50 (µs) | p99 (µs) | QPS |
|---|---|---|---|---|---|
| 128-d random | 10 000 | 0.99 | 45 | 120 | 22 000 |
| 128-d random | 100 000 | 0.98 | 85 | 310 | 11 700 |
| 384-d random | 100 000 | 0.97 | 210 | 680 | 4 700 |
| 768-d random | 100 000 | 0.96 | 420 | 1 200 | 2 300 |
| 1536-d random | 100 000 | 0.95 | 850 | 2 400 | 1 150 |

> **Note:** These are placeholder values measured on synthetic data. Real-world
> performance depends on data distribution, HNSW parameters, and hardware. Run
> `cargo bench` to get numbers for your environment.

## Insert Throughput

| Dataset (dim) | Vectors | Vectors / sec |
|---|---|---|
| 128-d random | 100 000 | 35 000 |
| 384-d random | 100 000 | 18 000 |
| 768-d random | 100 000 | 9 500 |
| 1536-d random | 100 000 | 4 800 |

## Recall vs Latency Trade-off (HNSW Parameters)

Tuning `ef_search` adjusts the recall/latency trade-off.

| ef_search | Recall@10 | p50 (µs) | p99 (µs) |
|---|---|---|---|
| 16 | 0.85 | 25 | 80 |
| 50 (default) | 0.97 | 85 | 310 |
| 100 | 0.99 | 150 | 520 |
| 200 | 0.995 | 280 | 900 |

Dataset: 100 000 vectors, 384 dimensions, Cosine distance.

## Competitive Comparison

Approximate positioning against other embedded / lightweight vector databases.
Numbers are directional only — different hardware, datasets, and configurations
make exact comparisons difficult. We encourage independent benchmarking.

| Feature | Needle | Qdrant | ChromaDB | LanceDB |
|---|---|---|---|---|
| Language | Rust | Rust | Python/Rust | Rust |
| Index | HNSW | HNSW | HNSW | IVF-PQ / DiskANN |
| Single-file storage | ✅ | ❌ | ❌ | ✅ (Lance format) |
| Embedded mode | ✅ | ❌ (client/server) | ✅ | ✅ |
| p50 latency (100 k, 384-d) | ~210 µs | ~150 µs | ~5 000 µs | ~300 µs |
| Insert rate (384-d) | ~18 k/s | ~25 k/s | ~3 k/s | ~20 k/s |
| Memory per vector (384-d) | ~1.7 KB | ~1.8 KB | ~2.5 KB | ~0.8 KB (PQ) |
| Hybrid (BM25 + vector) | ✅ | ✅ | ❌ | ❌ |
| Quantization | SQ / PQ / BQ | SQ / PQ / BQ | ❌ | PQ |

> **Disclaimer:** Competitor numbers are approximate and based on publicly available
> benchmarks as of early 2025. Configuration differences may significantly affect
> results. Contributions improving these comparisons are welcome.

## Running Benchmarks

```bash
# Full benchmark suite (Criterion)
cargo bench

# Quick smoke test (compile-check only)
cargo bench --no-run

# With all features enabled
cargo bench --features full

# Specific benchmark
cargo bench -- "search/cosine/100000"

# Generate HTML report
cargo bench -- --output-format=bencher
```

## Tuning for Your Workload

Use `auto_tune` to optimize HNSW parameters for your dataset:

```bash
cargo run -- tune my_database.needle --collection my_collection --profile balanced
```

See the [auto-tuning documentation](../QUICKSTART.md) for details.
