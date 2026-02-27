# Needle Benchmarks

Performance benchmarks for Needle using [Criterion.rs](https://github.com/bheisler/criterion.rs).

## Benchmark Suites

| File | Focus |
|------|-------|
| `benchmarks.rs` | Advanced features: RAG pipeline, encryption, temporal index, drift detection, federation |
| `index.rs` | Index operations: HNSW build/search, quantization (scalar, product, binary) |
| `insert.rs` | Write throughput: single insert, batch insert, insert with metadata |
| `search.rs` | Search performance: basic search, filtered search, batch search, distance functions |

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run a specific benchmark suite
cargo bench --bench benchmarks
cargo bench --bench search
cargo bench --bench insert
cargo bench --bench index

# Run benchmarks matching a pattern
cargo bench -- "hnsw"
cargo bench -- "batch_insert"

# Run with specific features enabled
cargo bench --features full
```

## Interpreting Results

Criterion outputs results to `target/criterion/`. Each benchmark produces:

- **Estimated time**: Mean, median, and confidence interval for each iteration
- **Throughput**: Operations per second (when `Throughput` is configured)
- **Comparison**: Statistical comparison against the previous run (improvement / regression / no change)

After running benchmarks, open `target/criterion/report/index.html` in a browser for interactive HTML reports with plots.

### Key Metrics to Watch

| Metric | Description |
|--------|-------------|
| Mean time | Average execution time per iteration |
| Throughput | Vectors inserted or searched per second |
| Change % | Percentage change vs. previous run (flagged if statistically significant) |

## Adding New Benchmarks

1. Add a new function `fn bench_<name>(c: &mut Criterion)` in the appropriate file
2. Register it in the `criterion_group!` macro at the bottom of the file
3. Use `black_box()` to prevent compiler optimizations on inputs/outputs
4. Use `Database::in_memory()` to avoid file I/O noise

## Tips

- Run benchmarks on a quiet machine for consistent results
- Use `--sample-size` to increase statistical confidence: `cargo bench -- --sample-size 100`
- Compare branches with `critcmp`: `cargo install critcmp && critcmp baseline new`
- Benchmarks use random vectors; results may vary slightly between runs
