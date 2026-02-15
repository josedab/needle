# Needle Development Task Runner (zero-install alternative to justfile)
# Usage: make <recipe>

.PHONY: help quick check check-all build build-all build-release test test-unit test-integration \
        fmt fmt-check lint lint-fix lint-dirty lint-new watch test-watch serve demo doctor doc bench clean playground setup setup-tools dev test-single coverage \
        new-module verify-docs check-quick check-local test-feature count-debt test-changed open-docs \
        docker-up docker-down docker-build docker-logs test-coverage-check

help:
	@echo "Available recipes:"
	@echo "  make setup         — First-time setup: doctor + pre-commit + build"
	@echo "  make setup-tools   — Install optional Cargo tools (cargo-watch, cargo-llvm-cov, …)"
	@echo "  make dev           — Start developing: setup + continuous check on save"
	@echo "  make quick         — Fast feedback: format check + lint + unit tests"
	@echo "  make check         — Full pre-commit: format check + lint + all tests"
	@echo "  make check-quick   — Quick CI gate: fmt-check + lint + unit tests (~3 min)"
	@echo "  make check-local   — Alias for check-quick (recommended pre-push check)"
	@echo "  make check-all     — Full CI equivalent: fmt, lint, test, doc-check, examples"
	@echo "  make build         — Debug build (default features)"
	@echo "  make build-all     — Debug build (all features)"
	@echo "  make build-release — Release build (all features)"
	@echo "  make test          — Run all tests (all features)"
	@echo "  make test-unit     — Run unit tests only (fast)"
	@echo "  make test-single   — Run a single test: make test-single NAME=test_name"
	@echo "  make test-feature  — Test with specific features: make test-feature FEATURES=server,metrics"
	@echo "  make test-changed  — Run tests for modified modules only"
	@echo "  make fmt           — Format code"
	@echo "  make fmt-check     — Check formatting"
	@echo "  make lint          — Run clippy linter"
	@echo "  make lint-fix      — Auto-fix clippy suggestions"
	@echo "  make lint-dirty    — Lint only uncommitted .rs files (fast)"
	@echo "  make lint-new      — Lint filtering out known service/experimental warnings"
	@echo "  make watch         — Continuous check on file changes (requires cargo-watch)"
	@echo "  make test-watch    — Continuous test on save — TDD workflow (requires cargo-watch)"
	@echo "  make serve         — Run HTTP server locally (NEEDLE_PORT=9090 make serve)"
	@echo "  make demo          — Run quickstart demo"
	@echo "  make doctor        — Check local environment"
	@echo "  make doc           — Generate and open documentation"
	@echo "  make open-docs     — Open existing rustdoc (no rebuild)"
	@echo "  make bench         — Run benchmarks"
	@echo "  make coverage      — Generate HTML coverage report (requires cargo-llvm-cov)"
	@echo "  make test-coverage-check — Fail if coverage is below 75% (codecov.yml threshold)"
	@echo "  make outdated      — Check for outdated dependencies (requires cargo-outdated)"
	@echo "  make count-debt    — Show tech debt & module size dashboard"
	@echo "  make verify-docs   — Check that all markdown links resolve"
	@echo "  make playground    — Interactive guided walkthrough"
	@echo "  make new-module    — Scaffold a new service module (DOMAIN=x NAME=y)"
	@echo "  make docker-up     — Start Needle via Docker Compose"
	@echo "  make docker-down   — Stop Docker Compose services"
	@echo "  make docker-build  — Build Docker image from source"
	@echo "  make docker-logs   — Tail Docker Compose logs"
	@echo "  make clean         — Clean build artifacts"
	@echo ""
	@echo "Feature flag combinations:"
	@echo "  --features full              All stable features (server, metrics, hybrid, encryption, …)"
	@echo "  --features server            HTTP REST API (Axum)"
	@echo "  --features hybrid            BM25 + vector hybrid search"
	@echo "  --features metrics           Prometheus metrics export"
	@echo "  --features encryption        ChaCha20-Poly1305 at-rest encryption"
	@echo "  --features experimental      Unstable/preview modules"
	@echo "  --features server,metrics    Server with Prometheus (common combo)"

# Fast feedback loop (with per-step timing)
quick:
	@total_start=$$(date +%s); \
	echo "=== make quick ==="; \
	\
	echo ""; echo "[1/3] fmt-check…"; \
	step_start=$$(date +%s); \
	$(MAKE) --no-print-directory fmt-check; \
	step_end=$$(date +%s); \
	echo "  ↳ fmt-check: $$((step_end - step_start))s"; \
	\
	echo ""; echo "[2/3] lint…"; \
	step_start=$$(date +%s); \
	$(MAKE) --no-print-directory lint; \
	step_end=$$(date +%s); \
	echo "  ↳ lint: $$((step_end - step_start))s"; \
	\
	echo ""; echo "[3/3] test-unit…"; \
	step_start=$$(date +%s); \
	$(MAKE) --no-print-directory test-unit; \
	step_end=$$(date +%s); \
	echo "  ↳ test-unit: $$((step_end - step_start))s"; \
	\
	total_end=$$(date +%s); \
	echo ""; echo "=== quick completed in $$((total_end - total_start))s ==="

# Start developing: first-time setup + continuous check on save
dev:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Run 'make setup-tools' first, or 'cargo install cargo-watch'."; exit 1; }
	$(MAKE) setup
	$(MAKE) watch

# Full pre-commit check (with per-step timing)
check:
	@total_start=$$(date +%s); \
	echo "=== make check ==="; \
	\
	echo ""; echo "[1/3] fmt-check…"; \
	step_start=$$(date +%s); \
	$(MAKE) --no-print-directory fmt-check; \
	step_end=$$(date +%s); \
	echo "  ↳ fmt-check: $$((step_end - step_start))s"; \
	\
	echo ""; echo "[2/3] lint…"; \
	step_start=$$(date +%s); \
	$(MAKE) --no-print-directory lint; \
	step_end=$$(date +%s); \
	echo "  ↳ lint: $$((step_end - step_start))s"; \
	\
	echo ""; echo "[3/3] test…"; \
	step_start=$$(date +%s); \
	$(MAKE) --no-print-directory test; \
	step_end=$$(date +%s); \
	echo "  ↳ test: $$((step_end - step_start))s"; \
	\
	total_end=$$(date +%s); \
	echo ""; echo "=== check completed in $$((total_end - total_start))s ==="

# Full CI equivalent: everything CI runs, locally
# NOTE: Keep flags in sync with justfile check-all recipe
check-all: fmt-check lint test
	RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --features full
	cargo build --examples --features full
	cargo run --example quickstart
	cargo run --example basic_usage
	cargo bench --no-run

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

# Lint only uncommitted .rs files (fast feedback on your changes)
lint-dirty:
	@changed=$$(git diff --name-only --diff-filter=ACMR HEAD -- '*.rs' 2>/dev/null); \
	staged=$$(git diff --name-only --diff-filter=ACMR --cached -- '*.rs' 2>/dev/null); \
	all=$$(printf '%s\n%s' "$$changed" "$$staged" | sort -u | grep '\.rs$$' || true); \
	if [ -z "$$all" ]; then \
		echo "No changed .rs files to lint."; \
	else \
		echo "Linting changed files:"; \
		echo "$$all" | sed 's/^/  /'; \
		pattern=$$(echo "$$all" | paste -sd'|' -); \
		cargo clippy --all-targets --features full -- -D warnings 2>&1 | \
			grep -E "($$pattern|^error)" || echo "✓ No warnings in changed files"; \
	fi

# Lint filtering out known warning sources in services/ and experimental/ (shows only new warnings)
lint-new:
	@echo "Running clippy (filtering known warning sources in services/ and experimental/)…"
	@cargo clippy --all-targets --features full -- -D warnings 2>&1 | \
		grep -v -E '^\s*--> src/(services|experimental)/' | \
		grep -v -E 'src/(services|experimental)/[^ ]+' | \
		grep -v '^$$' || true
	@echo ""
	@echo "Note: Warnings from src/services/ and src/experimental/ are hidden."
	@echo "Run 'make lint' to see all warnings."

# Continuous check on save (requires: cargo install cargo-watch)
watch:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'check --features full'

# Continuous test on save — TDD workflow (requires: cargo install cargo-watch)
test-watch:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'test --lib'

NEEDLE_PORT ?= 8080
RUST_LOG ?= info

serve:
	@echo "Starting Needle server on 127.0.0.1:$(NEEDLE_PORT)"
	@echo "Tip: change port with NEEDLE_PORT=9090 make serve"
	@echo "Tip: RUST_LOG=debug make serve (for verbose logging)"
	RUST_LOG=$(RUST_LOG) cargo run --features server -- serve -a 127.0.0.1:$(NEEDLE_PORT)

demo:
	./scripts/quickstart.sh

doctor:
	./scripts/doctor.sh

doc:
	cargo doc --no-deps --features full --open

# Open existing rustdoc without rebuilding (falls back to make doc if not generated)
open-docs:
	@if [ -f target/doc/needle/index.html ]; then \
		echo "Opening existing docs…"; \
		open target/doc/needle/index.html 2>/dev/null || xdg-open target/doc/needle/index.html 2>/dev/null || \
			echo "Docs at: target/doc/needle/index.html"; \
	else \
		echo "Docs not found — generating with 'make doc'…"; \
		$(MAKE) doc; \
	fi

bench:
	cargo bench

# Coverage report (requires: cargo install cargo-llvm-cov)
coverage:
	cargo llvm-cov --features full --html --open

# Check coverage meets thresholds from codecov.yml (75% project, 80% patch)
# Requires: cargo install cargo-llvm-cov
test-coverage-check:
	@command -v cargo-llvm-cov > /dev/null 2>&1 || { echo "Error: cargo-llvm-cov not found. Install with: cargo install cargo-llvm-cov"; exit 1; }
	@echo "Running coverage check (project threshold: 75%)…"
	@cargo llvm-cov --features full --json 2>/dev/null | \
		python3 -c "import sys,json; d=json.load(sys.stdin); pct=d['data'][0]['totals']['lines']['percent']; \
		print(f'Coverage: {pct:.2f}%'); sys.exit(0 if pct >= 75.0 else 1)" \
		|| { echo '✗ Coverage is below the 75% project threshold (see codecov.yml)'; exit 1; }
	@echo "✓ Coverage meets the 75% project threshold"

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
	@if command -v pre-commit > /dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo ""; \
		echo "⚠  pre-commit not found — git hooks not installed."; \
		echo "   Install with: pip install pre-commit"; \
		echo "   Then run:     pre-commit install"; \
		echo ""; \
	fi
	@if ! command -v cargo-audit > /dev/null 2>&1; then \
		echo "⚠  cargo-audit not found — push hooks won't run audit."; \
		echo "   Install with: cargo install cargo-audit"; \
		echo ""; \
	fi
	cargo build

# Install optional Cargo tools used by other targets (watch, coverage, outdated, audit)
setup-tools:
	@echo "Installing optional Cargo tools…"
	cargo install cargo-watch cargo-outdated cargo-llvm-cov cargo-audit
	@echo "Done. You can now use: make watch, make coverage, make outdated"

test-single:
	cargo test $(NAME) -- --nocapture

verify-docs:
	./scripts/verify-docs.sh

# Scaffold a new service module: make new-module DOMAIN=search NAME=my_feature
new-module:
	@test -n "$(DOMAIN)" || { echo "Usage: make new-module DOMAIN=<domain> NAME=<module_name>"; exit 1; }
	@test -n "$(NAME)" || { echo "Usage: make new-module DOMAIN=<domain> NAME=<module_name>"; exit 1; }
	./scripts/new-module.sh $(DOMAIN) $(NAME)

# Quick CI gate: fmt-check + lint (all-targets) + unit tests (~3 min)
check-quick: fmt-check lint test-unit

# Alias for check-quick — recommended pre-push command (mirrors CI fast gate)
check-local: check-quick

# Run tests with specific feature flags: make test-feature FEATURES=server,metrics
test-feature:
	@test -n "$(FEATURES)" || { echo "Usage: make test-feature FEATURES=server,metrics"; exit 1; }
	cargo test --features $(FEATURES)

# Run tests for modified modules only (based on git diff)
test-changed:
	@changed=$$(git diff --name-only --diff-filter=ACMR HEAD -- 'src/*.rs' 'src/**/*.rs' 2>/dev/null); \
	staged=$$(git diff --name-only --diff-filter=ACMR --cached -- 'src/*.rs' 'src/**/*.rs' 2>/dev/null); \
	all=$$(printf '%s\n%s' "$$changed" "$$staged" | sort -u | grep '\.rs$$' || true); \
	if [ -z "$$all" ]; then \
		echo "No changed .rs files — nothing to test."; \
		exit 0; \
	fi; \
	modules=$$(echo "$$all" | sed -n 's|^src/\(.*\)\.rs$$|\1|p' | sed 's|/mod$$||; s|/|::|g' | sort -u); \
	if [ -z "$$modules" ]; then \
		echo "Could not extract module paths — running all unit tests."; \
		cargo test --lib; \
		exit 0; \
	fi; \
	echo "Testing changed modules:"; \
	echo "$$modules" | sed 's/^/  /'; \
	for mod in $$modules; do \
		cargo test --lib "$$mod" 2>/dev/null || true; \
	done

# Show tech debt dashboard (unwrap, expect, let _ = counts in src/)
count-debt:
	@unwrap=$$(grep -r 'unwrap()' src/ --include='*.rs' | wc -l | tr -d ' '); \
	expect=$$(grep -r 'expect(' src/ --include='*.rs' | wc -l | tr -d ' '); \
	let_discard=$$(grep -r 'let _ =' src/ --include='*.rs' | wc -l | tr -d ' '); \
	total_files=$$(find src/ -name '*.rs' | wc -l | tr -d ' '); \
	total_lines=$$(find src/ -name '*.rs' -exec cat {} + | wc -l | tr -d ' '); \
	dead_code=$$(grep -rl '#!\[allow(dead_code)\]' src/ --include='*.rs' | wc -l | tr -d ' '); \
	echo "Tech Debt Dashboard"; \
	echo "==================="; \
	echo ""; \
	echo "Codebase Size"; \
	echo "  Total .rs files : $$total_files"; \
	echo "  Total lines     : $$total_lines"; \
	echo ""; \
	echo "Debt Markers"; \
	echo "  unwrap(): $$unwrap | expect(): $$expect | let _ =: $$let_discard"; \
	echo "  #![allow(dead_code)] files: $$dead_code"; \
	echo ""; \
	echo "Top 5 Largest Files"; \
	find src/ -name '*.rs' -exec wc -l {} + | sort -rn | head -n 6 | tail -n 5 | awk '{printf "  %6d  %s\n", $$1, $$2}'; \
	echo ""; \
	echo "Per-Directory Breakdown (files / lines)"; \
	for dir in $$(find src/ -mindepth 1 -maxdepth 1 -type d | sort); do \
		d_files=$$(find "$$dir" -name '*.rs' | wc -l | tr -d ' '); \
		d_lines=$$(find "$$dir" -name '*.rs' -exec cat {} + 2>/dev/null | wc -l | tr -d ' '); \
		printf "  %-30s %4s files  %6s lines\n" "$$dir" "$$d_files" "$$d_lines"; \
	done; \
	root_files=$$(find src/ -maxdepth 1 -name '*.rs' | wc -l | tr -d ' '); \
	root_lines=$$(find src/ -maxdepth 1 -name '*.rs' -exec cat {} + 2>/dev/null | wc -l | tr -d ' '); \
	printf "  %-30s %4s files  %6s lines\n" "src/ (root)" "$$root_files" "$$root_lines"

# Docker convenience targets
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose -f docker-compose.source.yml build

docker-logs:
	docker compose logs -f
