# Needle Development Task Runner (zero-install alternative to justfile)
# Usage: make <recipe>

.PHONY: help quick check build build-all build-release test test-unit test-integration \
        fmt fmt-check lint lint-fix watch serve demo doctor doc bench clean playground setup test-single coverage \
        new-module

help:
	@echo "Available recipes:"
	@echo "  make setup         — First-time setup: doctor + pre-commit + build"
	@echo "  make quick         — Fast feedback: format check + lint + unit tests"
	@echo "  make check         — Full pre-commit: format check + lint + all tests"
	@echo "  make build         — Debug build (default features)"
	@echo "  make build-all     — Debug build (all features)"
	@echo "  make build-release — Release build (all features)"
	@echo "  make test          — Run all tests (all features)"
	@echo "  make test-unit     — Run unit tests only (fast)"
	@echo "  make test-single   — Run a single test: make test-single NAME=test_name"
	@echo "  make fmt           — Format code"
	@echo "  make fmt-check     — Check formatting"
	@echo "  make lint          — Run clippy linter"
	@echo "  make lint-fix      — Auto-fix clippy suggestions"
	@echo "  make watch         — Continuous check on file changes (requires cargo-watch)"
	@echo "  make serve         — Run HTTP server locally (NEEDLE_PORT=9090 make serve)"
	@echo "  make demo          — Run quickstart demo"
	@echo "  make doctor        — Check local environment"
	@echo "  make doc           — Generate and open documentation"
	@echo "  make bench         — Run benchmarks"
	@echo "  make coverage      — Generate HTML coverage report (requires cargo-llvm-cov)"
	@echo "  make outdated      — Check for outdated dependencies (requires cargo-outdated)"
	@echo "  make playground    — Interactive guided walkthrough"
	@echo "  make new-module    — Scaffold a new service module (DOMAIN=x NAME=y)"
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
	cargo clippy --all-targets --features full -- -D warnings

lint-fix:
	cargo clippy --features full --fix --allow-dirty --allow-staged

# Continuous check on save (requires: cargo install cargo-watch)
watch:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'check --features full'

NEEDLE_PORT ?= 8080

serve:
	@echo "Starting Needle server on 127.0.0.1:$(NEEDLE_PORT)"
	@echo "Tip: change port with NEEDLE_PORT=9090 make serve"
	cargo run --features server -- serve -a 127.0.0.1:$(NEEDLE_PORT)

demo:
	./scripts/quickstart.sh

doctor:
	./scripts/doctor.sh

doc:
	cargo doc --no-deps --features full --open

bench:
	cargo bench

# Coverage report (requires: cargo install cargo-llvm-cov)
coverage:
	cargo llvm-cov --features full --html --open

# Check for outdated dependencies (requires: cargo install cargo-outdated)
outdated:
	@command -v cargo-outdated > /dev/null 2>&1 || { echo "Error: cargo-outdated not found. Install with: cargo install cargo-outdated"; exit 1; }
	cargo outdated -R

playground:
	./scripts/playground.sh

clean:
	cargo clean

setup:
	./scripts/doctor.sh
	@(command -v pre-commit > /dev/null 2>&1 && pre-commit install) || true
	cargo build

test-single:
	cargo test $(NAME) -- --nocapture

# Scaffold a new service module: make new-module DOMAIN=search NAME=my_feature
new-module:
	@test -n "$(DOMAIN)" || { echo "Usage: make new-module DOMAIN=<domain> NAME=<module_name>"; exit 1; }
	@test -n "$(NAME)" || { echo "Usage: make new-module DOMAIN=<domain> NAME=<module_name>"; exit 1; }
	./scripts/new-module.sh $(DOMAIN) $(NAME)
