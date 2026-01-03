# Needle Fuzzing

This directory contains fuzz targets for testing Needle's parsers and core functions.

## Requirements

Install cargo-fuzz:

```bash
cargo install cargo-fuzz
```

Note: Fuzzing requires nightly Rust:

```bash
rustup install nightly
```

## Available Fuzz Targets

| Target | Description |
|--------|-------------|
| `fuzz_query_parser` | NeedleQL query language parser |
| `fuzz_nl_filter` | Natural language filter parser |
| `fuzz_metadata_filter` | MongoDB-style JSON filter parser |
| `fuzz_distance` | Distance function implementations |

## Running Fuzz Targets

Run a specific target:

```bash
cargo +nightly fuzz run fuzz_query_parser
```

Run with a time limit (e.g., 60 seconds):

```bash
cargo +nightly fuzz run fuzz_query_parser -- -max_total_time=60
```

Run with multiple jobs:

```bash
cargo +nightly fuzz run fuzz_query_parser -- -jobs=4 -workers=4
```

## Corpus

Fuzzing corpus is stored in `fuzz/corpus/<target_name>/`. You can seed it with valid inputs:

```bash
mkdir -p corpus/fuzz_query_parser
echo 'SELECT * FROM vectors WHERE score > 0.5' > corpus/fuzz_query_parser/sample1
echo 'SEARCH "hello world" LIMIT 10' > corpus/fuzz_query_parser/sample2
```

## Coverage

Generate coverage report:

```bash
cargo +nightly fuzz coverage fuzz_query_parser
```

## Reproducing Crashes

If a crash is found, reproduce it:

```bash
cargo +nightly fuzz run fuzz_query_parser fuzz/artifacts/fuzz_query_parser/crash-<hash>
```

## Adding New Fuzz Targets

1. Create a new file in `fuzz_targets/`
2. Add a `[[bin]]` entry in `Cargo.toml`
3. Implement the `fuzz_target!` macro

Example:

```rust
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Your fuzzing logic here
});
```
