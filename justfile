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

# Run clippy linter with all features
lint:
    cargo clippy --features full -- -D warnings

# Run all tests
test:
    cargo test --features full

# Run unit tests only (fast)
test-unit:
    cargo test --lib

# Run integration tests only
test-integration:
    cargo test --tests --features full

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
