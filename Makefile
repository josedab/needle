# Needle Development Task Runner (zero-install alternative to justfile)
# Usage: make <recipe>

.PHONY: help quick check build build-all build-release test test-unit test-integration \
        fmt fmt-check lint serve demo doctor doc bench clean playground

help:
	@echo "Available recipes:"
	@echo "  make quick         — Fast feedback: format check + lint + unit tests"
	@echo "  make check         — Full pre-commit: format check + lint + all tests"
	@echo "  make build         — Debug build (default features)"
	@echo "  make build-all     — Debug build (all features)"
	@echo "  make build-release — Release build (all features)"
	@echo "  make test          — Run all tests (all features)"
	@echo "  make test-unit     — Run unit tests only (fast)"
	@echo "  make fmt           — Format code"
	@echo "  make fmt-check     — Check formatting"
	@echo "  make lint          — Run clippy linter"
	@echo "  make serve         — Run HTTP server locally"
	@echo "  make demo          — Run quickstart demo"
	@echo "  make doctor        — Check local environment"
	@echo "  make doc           — Generate and open documentation"
	@echo "  make bench         — Run benchmarks"
	@echo "  make playground    — Interactive guided walkthrough"
	@echo "  make clean         — Clean build artifacts"

# Fast feedback loop
quick: fmt-check lint test-unit

# Full pre-commit check
check: fmt-check lint test

build:
	cargo build

build-all:
	cargo build --features full

build-release:
	cargo build --release --features full

test:
	cargo test --features full

test-unit:
	cargo test --lib

test-integration:
	cargo test --tests --features full

fmt:
	cargo fmt

fmt-check:
	cargo fmt -- --check

lint:
	cargo clippy --features full

serve:
	cargo run --features server -- serve -a 127.0.0.1:8080

demo:
	./scripts/quickstart.sh

doctor:
	./scripts/doctor.sh

doc:
	cargo doc --no-deps --features full --open

bench:
	cargo bench

playground:
	./scripts/playground.sh

clean:
	cargo clean
