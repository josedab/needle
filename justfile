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

# Run clippy linter with all features (matches CI's -D warnings)
lint:
    cargo clippy --features full -- -D warnings

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

# Continuous check on save (requires: cargo install cargo-watch)
watch:
    #!/usr/bin/env bash
    if ! command -v cargo-watch &> /dev/null; then
        echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"
        exit 1
    fi
    cargo watch -x 'check --features full'

# Full pre-commit check: format, lint, test
check: fmt-check lint test

# Full CI equivalent: fmt, lint, test, doc-check, examples, bench compile
check-all: fmt-check lint test
    RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --features full
    cargo build --examples --features full
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

# Run the HTTP server locally
serve:
    @echo "Tip: RUST_LOG=debug just serve (for verbose logging)"
    cargo run --features server -- serve -a 127.0.0.1:8080

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
