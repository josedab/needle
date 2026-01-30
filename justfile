# Needle Development Task Runner
# Install: cargo install just
# Usage: just <recipe>

default:
    @just --list

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
    cargo run --features server -- serve -a 127.0.0.1:8080

# Run the quickstart demo (builds, seeds, searches)
demo:
    ./scripts/quickstart.sh

# Check local environment setup
doctor:
    ./scripts/doctor.sh

# Quick check: format + lint + unit tests (fast feedback)
quick: fmt-check lint test-unit

# Interactive guided walkthrough of the HTTP API
playground:
    ./scripts/playground.sh

# First-time setup: doctor + pre-commit + build
setup:
    ./scripts/doctor.sh
    (command -v pre-commit && pre-commit install) || true
    cargo build

# Run a single test with output: just test-single test_name
test-single NAME:
    cargo test {{NAME}} -- --nocapture

# Scaffold a new service module: just new-module search my_feature
new-module DOMAIN MODULE:
    ./scripts/new-module.sh {{DOMAIN}} {{MODULE}}
