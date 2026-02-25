# Needle Development Task Runner
# Install: cargo install just
# Usage: just <recipe>

default:
    @just --list

# Show common feature flag combinations
help-features:
    @echo "Feature flag combinations:"
    @echo "  --features full              All stable features (server, metrics, hybrid, encryption, …)"
    @echo "  --features server            HTTP REST API (Axum)"
    @echo "  --features hybrid            BM25 + vector hybrid search"
    @echo "  --features metrics           Prometheus metrics export"
    @echo "  --features encryption        ChaCha20-Poly1305 at-rest encryption"
    @echo "  --features experimental      Unstable/preview modules"
    @echo "  --features server,metrics    Server with Prometheus (common combo)"

# Format code
fmt:
    cargo fmt

# Check formatting without modifying
fmt-check:
    cargo fmt -- --check

# Run clippy linter with all features and targets (matches CI and Makefile)
lint:
    cargo clippy --all-targets --features full -- -D warnings

# Auto-fix clippy suggestions
lint-fix:
    cargo clippy --features full --fix --allow-dirty --allow-staged

# Run all tests
test:
    cargo test --features full

# Run unit tests only (fast)
test-unit:
    cargo test --lib

# Run integration tests only
test-integration:
    cargo test --tests --features full

# Run documentation tests only
test-doc:
    cargo test --doc --features full

# Continuous check on save (requires: cargo install cargo-watch)
watch:
    #!/usr/bin/env bash
    if ! command -v cargo-watch &> /dev/null; then
        echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"
        exit 1
    fi
    cargo watch -x 'check --features full'

# Continuous test on save — TDD workflow (requires: cargo install cargo-watch)
test-watch:
    #!/usr/bin/env bash
    if ! command -v cargo-watch &> /dev/null; then
        echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"
        exit 1
    fi
    cargo watch -x 'test --lib'

# Continuous clippy lint on save (requires: cargo install cargo-watch)
lint-watch:
    #!/usr/bin/env bash
    if ! command -v cargo-watch &> /dev/null; then
        echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"
        exit 1
    fi
    cargo watch -x 'clippy --all-targets --features full -- -D warnings'

# Full pre-commit check: format, lint, test
check: fmt-check lint test

# Full CI equivalent: fmt, lint, test, doc-check, examples, bench compile
# NOTE: Keep flags in sync with Makefile check-all target
check-all: fmt-check lint test
    RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --features full
    cargo build --examples --features full
    cargo run --example quickstart
    cargo run --example basic_usage
    cargo bench --no-run

# Build with default features
build:
    cargo build

# Build with all features
build-all:
    cargo build --features full

# Build release
build-release:
    cargo build --release --features full

# Build with compilation timing report
profile-build:
    cargo build --features full --timings
    @echo ""
    @echo "Timing report: target/cargo-timings/cargo-timing.html"
    @open target/cargo-timings/cargo-timing.html 2>/dev/null || xdg-open target/cargo-timings/cargo-timing.html 2>/dev/null || echo "Open the above file in a browser to view the report."

# Generate documentation
doc:
    cargo doc --no-deps --features full --open

# Run benchmarks
bench:
    cargo bench

# Check for outdated dependencies (requires: cargo install cargo-outdated)
outdated:
    #!/usr/bin/env bash
    if ! command -v cargo-outdated &> /dev/null; then
        echo "Error: cargo-outdated not found. Install with: cargo install cargo-outdated"
        exit 1
    fi
    cargo outdated -R

# Clean build artifacts
clean:
    cargo clean

# Run the HTTP server locally (NEEDLE_PORT=9090 just serve)
serve:
    @echo "Starting Needle server on 127.0.0.1:{{ env_var_or_default("NEEDLE_PORT", "8080") }}"
    @echo "Tip: change port with NEEDLE_PORT={{ env_var_or_default("NEEDLE_PORT", "9090") }} just serve"
    @echo "Tip: RUST_LOG=debug just serve (for verbose logging)"
    @echo "Tip: 'just serve-env' to auto-load .env before starting"
    RUST_LOG={{ env_var_or_default("RUST_LOG", "info") }} cargo run --features server -- serve -a 127.0.0.1:{{ env_var_or_default("NEEDLE_PORT", "8080") }}

# Run HTTP server with .env auto-loaded
serve-env:
    #!/usr/bin/env bash
    if [ -f .env ]; then
        echo "Loading .env…"
        set -a && source .env && set +a
    elif [ -f .env.example ]; then
        echo "No .env found — loading .env.example as fallback"
        set -a && source .env.example && set +a
    else
        echo "No .env or .env.example found — starting with defaults"
    fi
    RUST_LOG="${RUST_LOG:-info}" cargo run --features server -- serve -a "${NEEDLE_ADDRESS:-127.0.0.1:${NEEDLE_PORT:-8080}}"

# Run the quickstart demo (builds, seeds, searches)
demo:
    ./scripts/quickstart.sh

# Check local environment setup
doctor:
    ./scripts/doctor.sh

# Quick check: format + lint + unit tests (fast feedback)
quick: fmt-check lint test-unit

# Start developing: first-time setup + continuous check on save
dev: setup watch

# Interactive guided walkthrough of the HTTP API
playground:
    ./scripts/playground.sh

# First-time setup: doctor + pre-commit + build
setup:
    #!/usr/bin/env bash
    ./scripts/doctor.sh
    if command -v pre-commit &> /dev/null; then
        pre-commit install
    else
        echo ""
        echo "⚠  pre-commit not found — git hooks not installed."
        echo "   Install with: pip install pre-commit"
        echo "   Then run:     pre-commit install"
        echo ""
    fi
    if ! command -v cargo-audit &> /dev/null; then
        echo "⚠  cargo-audit not found — push hooks won't run audit."
        echo "   Install with: cargo install cargo-audit"
        echo ""
    fi
    cargo build

# Install optional Cargo tools used by other recipes (watch, coverage, outdated, audit)
setup-tools:
    @echo "Installing optional Cargo tools…"
    cargo install cargo-watch cargo-outdated cargo-llvm-cov cargo-audit
    @echo "Done. You can now use: just watch, just coverage, just outdated"

# Run a single test with output: just test-single test_name
test-single NAME:
    cargo test {{NAME}} -- --nocapture

# Verify markdown links in documentation files
verify-docs:
    ./scripts/verify-docs.sh

# Scaffold a new service module: just new-module search my_feature
new-module DOMAIN MODULE:
    ./scripts/new-module.sh {{DOMAIN}} {{MODULE}}

# Scaffold a new example: just new-example my_search_demo
new-example NAME:
    ./scripts/new-example.sh {{NAME}}

# Quick CI gate: fmt-check + lint (all-targets) + unit tests (~3 min), with timing
check-quick:
    #!/usr/bin/env bash
    set -e
    total_start=$(date +%s)
    echo "=== just check-quick ==="

    echo ""; echo "[1/3] fmt-check…"
    step_start=$(date +%s)
    just fmt-check
    step_end=$(date +%s)
    echo "  ↳ fmt-check: $((step_end - step_start))s"

    echo ""; echo "[2/3] lint…"
    step_start=$(date +%s)
    just lint
    step_end=$(date +%s)
    echo "  ↳ lint: $((step_end - step_start))s"

    echo ""; echo "[3/3] test-unit…"
    step_start=$(date +%s)
    just test-unit
    step_end=$(date +%s)
    echo "  ↳ test-unit: $((step_end - step_start))s"

    total_end=$(date +%s)
    echo ""; echo "=== check-quick completed in $((total_end - total_start))s ==="

# Run tests with specific feature flags: just test-feature "server,metrics"
test-feature FEATURES:
    cargo test --features {{FEATURES}}

# Compile all examples and run those that don't require external services
verify-examples:
    #!/usr/bin/env bash
    set -e
    echo "=== Compiling all examples (--features full) ==="
    cargo build --examples --features full
    echo ""
    echo "=== Running examples that don't require external services ==="
    failed=0
    for example in quickstart basic_usage filtered_search persistence quantization \
                   multi_tenant multi_vector sparse_vectors image_search sharding \
                   hybrid_search encryption_usage rag_chatbot metrics_usage diskann_usage; do
        echo ""
        echo "→ $example"
        if cargo run --example "$example" --features full 2>&1; then
            echo "  ✓ $example passed"
        else
            echo "  ✗ $example failed"
            failed=$((failed + 1))
        fi
    done
    echo ""
    if [ "$failed" -gt 0 ]; then
        echo "=== $failed example(s) failed ==="
        exit 1
    else
        echo "=== All examples passed ==="
    fi

# Show tech debt dashboard (unwrap, expect, let _ = counts in src/)
count-debt:
    #!/usr/bin/env bash
    unwrap=$(grep -r 'unwrap()' src/ --include='*.rs' | wc -l | tr -d ' ')
    expect=$(grep -r 'expect(' src/ --include='*.rs' | wc -l | tr -d ' ')
    let_discard=$(grep -r 'let _ =' src/ --include='*.rs' | wc -l | tr -d ' ')
    total_files=$(find src/ -name '*.rs' | wc -l | tr -d ' ')
    total_lines=$(find src/ -name '*.rs' -exec cat {} + | wc -l | tr -d ' ')
    dead_code=$(grep -rl '#!\[allow(dead_code)\]' src/ --include='*.rs' | wc -l | tr -d ' ')
    echo "Tech Debt Dashboard"
    echo "==================="
    echo ""
    echo "Codebase Size"
    echo "  Total .rs files : $total_files"
    echo "  Total lines     : $total_lines"
    echo ""
    echo "Debt Markers"
    echo "  unwrap(): $unwrap | expect(): $expect | let _ =: $let_discard"
    echo "  #![allow(dead_code)] files: $dead_code"
    echo ""
    echo "Top 5 Largest Files"
    find src/ -name '*.rs' -exec wc -l {} + | sort -rn | head -n 6 | tail -n 5 | awk '{printf "  %6d  %s\n", $1, $2}'
    echo ""
    echo "Per-Directory Breakdown (files / lines / unwrap / expect)"
    for dir in $(find src/ -mindepth 1 -maxdepth 1 -type d | sort); do
        d_files=$(find "$dir" -name '*.rs' | wc -l | tr -d ' ')
        d_lines=$(find "$dir" -name '*.rs' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
        d_unwrap=$(grep -r 'unwrap()' "$dir" --include='*.rs' 2>/dev/null | wc -l | tr -d ' ')
        d_expect=$(grep -r 'expect(' "$dir" --include='*.rs' 2>/dev/null | wc -l | tr -d ' ')
        printf "  %-30s %4s files  %6s lines  %4s unwrap()  %4s expect()\n" "$dir" "$d_files" "$d_lines" "$d_unwrap" "$d_expect"
    done
    root_files=$(find src/ -maxdepth 1 -name '*.rs' | wc -l | tr -d ' ')
    root_lines=$(find src/ -maxdepth 1 -name '*.rs' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
    root_unwrap=$(grep -r 'unwrap()' src/ --maxdepth 1 --include='*.rs' 2>/dev/null | wc -l | tr -d ' ')
    root_expect=$(grep -r 'expect(' src/ --maxdepth 1 --include='*.rs' 2>/dev/null | wc -l | tr -d ' ')
    printf "  %-30s %4s files  %6s lines  %4s unwrap()  %4s expect()\n" "src/ (root)" "$root_files" "$root_lines" "$root_unwrap" "$root_expect"

# Start Needle via Docker Compose
docker-up:
    docker compose up -d

# Stop Docker Compose services
docker-down:
    docker compose down

# Build Docker image from source
docker-build:
    docker compose -f docker-compose.source.yml build

# Tail Docker Compose logs
docker-logs:
    docker compose logs -f
