# Needle Development Task Runner (zero-install alternative to justfile)
# Usage: make <recipe>

.PHONY: help quick check check-all build build-all build-release test test-unit test-integration \
        fmt fmt-check lint lint-fix lint-dirty lint-new lint-module watch test-watch lint-watch serve serve-env demo doctor doc bench bench-single clean playground setup setup-tools dev test-single coverage \
        new-module verify-docs check-quick check-local test-feature count-debt test-changed open-docs \
        docker-up docker-down docker-build docker-logs test-coverage-check \
        profile-build test-doc new-example verify-examples update-deps \
        check-deps coverage-summary list-large-files scaffold-test

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
	@echo "  make lint-module   — Lint a single module: make lint-module MODULE=collection"
	@echo "  make watch         — Continuous check on file changes (requires cargo-watch)"
	@echo "  make test-watch    — Continuous test on save — TDD workflow (requires cargo-watch)"
	@echo "  make lint-watch    — Continuous clippy lint on save (requires cargo-watch)"
	@echo "  make serve         — Run HTTP server locally (NEEDLE_PORT=9090 make serve)"
	@echo "  make serve-env     — Run HTTP server with .env auto-loaded"
	@echo "  make demo          — Run quickstart demo"
	@echo "  make doctor        — Check local environment"
	@echo "  make doc           — Generate and open documentation"
	@echo "  make open-docs     — Open existing rustdoc (no rebuild)"
	@echo "  make bench         — Run benchmarks"
	@echo "  make bench-single  — Run a single benchmark: make bench-single NAME=search"
	@echo "  make coverage      — Generate HTML coverage report (requires cargo-llvm-cov)"
	@echo "  make test-coverage-check — Fail if coverage is below 75% (codecov.yml threshold)"
	@echo "  make coverage-summary — Print coverage percentage (no browser)"
	@echo "  make outdated      — Check for outdated dependencies (requires cargo-outdated)"
	@echo "  make count-debt    — Show tech debt & module size dashboard"
	@echo "  make list-large-files — List .rs files above threshold (THRESHOLD=1000)"
	@echo "  make update-deps   — Update dependencies and run tests"
	@echo "  make verify-docs   — Check that all markdown links resolve"
	@echo "  make playground    — Interactive guided walkthrough"
	@echo "  make new-module    — Scaffold a new service module (DOMAIN=x NAME=y)"
	@echo "  make docker-up     — Start Needle via Docker Compose"
	@echo "  make docker-down   — Stop Docker Compose services"
	@echo "  make docker-build  — Build Docker image from source"
	@echo "  make docker-logs   — Tail Docker Compose logs"
	@echo "  make profile-build — Build with timing/profiling (cargo build --timings)"
	@echo "  make test-doc      — Run documentation tests (cargo test --doc)"
	@echo "  make new-example   — Scaffold a new example (NAME=x)"
	@echo "  make scaffold-test — Add test boilerplate to a file (FILE=src/path.rs)"
	@echo "  make verify-examples — Compile and check all examples"
	@echo "  make check-deps    — Check all optional tools and print install summary"
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
	$(MAKE) --no-print-directory test-doc
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

# Build with compilation timing report (opens timing report in browser)
profile-build:
	cargo build --features full --timings
	@echo ""
	@echo "Timing report: target/cargo-timings/cargo-timing.html"
	@open target/cargo-timings/cargo-timing.html 2>/dev/null || \
		xdg-open target/cargo-timings/cargo-timing.html 2>/dev/null || \
		echo "Open the above file in a browser to view the report."

test:
	cargo test --features full

test-unit:
	cargo test --lib

test-integration:
	cargo test --tests --features full

test-doc:
	cargo test --doc --features full

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

# Lint a single module: make lint-module MODULE=collection
lint-module:
	@test -n "$(MODULE)" || { echo "Usage: make lint-module MODULE=<module_name>"; echo "Example: make lint-module MODULE=collection"; exit 1; }
	@echo "Linting src/$(MODULE)/…"
	@cargo clippy --all-targets --features full -- -D warnings 2>&1 | \
		grep -E '^\s*--> src/$(MODULE)/|^error|^warning' || echo "✓ No warnings in src/$(MODULE)/"

# Continuous check on save (requires: cargo install cargo-watch)
watch:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'check --features full'

# Continuous test on save — TDD workflow (requires: cargo install cargo-watch)
test-watch:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'test --lib'

# Continuous clippy lint on save (requires: cargo install cargo-watch)
lint-watch:
	@command -v cargo-watch > /dev/null 2>&1 || { echo "Error: cargo-watch not found. Install with: cargo install cargo-watch"; exit 1; }
	cargo watch -x 'clippy --all-targets --features full -- -D warnings'

NEEDLE_PORT ?= 8080
RUST_LOG ?= info

serve:
	@echo "Starting Needle server on 127.0.0.1:$(NEEDLE_PORT)"
	@echo "Tip: change port with NEEDLE_PORT=9090 make serve"
	@echo "Tip: RUST_LOG=debug make serve (for verbose logging)"
	@echo "Tip: RUST_LOG=needle=debug,tower_http=info make serve (per-module filtering)"
	@echo "Tip: 'make serve-env' to auto-load .env before starting"
	RUST_LOG=$(RUST_LOG) cargo run --features server -- serve -a 127.0.0.1:$(NEEDLE_PORT)

# Start server with .env file loaded (sources .env.example as fallback if .env doesn't exist)
serve-env:
	@if [ -f .env ]; then \
		echo "Loading .env…"; \
		set -a && . ./.env && set +a && \
		RUST_LOG=$${RUST_LOG:-info} cargo run --features server -- serve -a $${NEEDLE_ADDRESS:-127.0.0.1:$(NEEDLE_PORT)}; \
	elif [ -f .env.example ]; then \
		echo "No .env found — loading .env.example as fallback"; \
		set -a && . ./.env.example && set +a && \
		RUST_LOG=$${RUST_LOG:-info} cargo run --features server -- serve -a $${NEEDLE_ADDRESS:-127.0.0.1:$(NEEDLE_PORT)}; \
	else \
		echo "No .env or .env.example found — starting with defaults"; \
		$(MAKE) serve; \
	fi

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

# Run a single benchmark: make bench-single NAME=search
bench-single:
	@test -n "$(NAME)" || { echo "Usage: make bench-single NAME=<benchmark_name>"; exit 1; }
	cargo bench -- $(NAME)

# Coverage report (requires: cargo install cargo-llvm-cov)
coverage:
	cargo llvm-cov --features full --html --open

# Check coverage meets thresholds from codecov.yml (75% project, 80% patch)
# Requires: cargo install cargo-llvm-cov
test-coverage-check:
	@command -v cargo-llvm-cov > /dev/null 2>&1 || { echo "Error: cargo-llvm-cov not found. Install with: cargo install cargo-llvm-cov"; exit 1; }
	@command -v python3 > /dev/null 2>&1 || { echo "Error: python3 not found. Install Python 3 from https://www.python.org/ or via your package manager."; exit 1; }
	@echo "Running coverage check (project threshold: 75%)…"
	@cargo llvm-cov --features full --json 2>/dev/null | \
		python3 -c "import sys,json; d=json.load(sys.stdin); pct=d['data'][0]['totals']['lines']['percent']; \
		print(f'Coverage: {pct:.2f}%'); sys.exit(0 if pct >= 75.0 else 1)" \
		|| { echo '✗ Coverage is below the 75% project threshold (see codecov.yml)'; exit 1; }
	@echo "✓ Coverage meets the 75% project threshold"

# Print coverage percentage without opening a browser (requires cargo-llvm-cov + python3)
coverage-summary:
	@command -v cargo-llvm-cov > /dev/null 2>&1 || { echo "Error: cargo-llvm-cov not found. Install with: cargo install cargo-llvm-cov"; exit 1; }
	@command -v python3 > /dev/null 2>&1 || { echo "Error: python3 not found. Install Python 3 from https://www.python.org/ or via your package manager."; exit 1; }
	@echo "Generating coverage summary…"
	@cargo llvm-cov --features full --json 2>/dev/null | \
		python3 -c "import sys,json; d=json.load(sys.stdin); t=d['data'][0]['totals']; \
		print(f\"Lines:     {t['lines']['percent']:.2f}% ({t['lines']['covered']}/{t['lines']['count']})\"); \
		print(f\"Functions: {t['functions']['percent']:.2f}% ({t['functions']['covered']}/{t['functions']['count']})\"); \
		print(f\"Regions:   {t['regions']['percent']:.2f}% ({t['regions']['covered']}/{t['regions']['count']})\")"

# Check for outdated dependencies (requires: cargo install cargo-outdated)
outdated:
	@command -v cargo-outdated > /dev/null 2>&1 || { echo "Error: cargo-outdated not found. Install with: cargo install cargo-outdated"; exit 1; }
	cargo outdated -R

# Update dependencies and verify with tests
update-deps:
	@echo "Updating dependencies…"
	cargo update
	@echo ""
	@echo "Running tests to verify update…"
	cargo test --features full
	@echo ""
	@echo "✓ Dependencies updated and tests passed"

playground:
	./scripts/playground.sh

clean:
	cargo clean

setup:
	@total_start=$$(date +%s); \
	echo "=== make setup ==="; \
	\
	echo ""; echo "[1/3] doctor…"; \
	step_start=$$(date +%s); \
	./scripts/doctor.sh; \
	step_end=$$(date +%s); \
	echo "  ↳ doctor: $$((step_end - step_start))s"; \
	\
	echo ""; echo "[2/3] pre-commit + cargo-audit…"; \
	step_start=$$(date +%s); \
	if command -v pre-commit > /dev/null 2>&1; then \
		pre-commit install; \
	else \
		echo ""; \
		echo "⚠  pre-commit not found — git hooks not installed."; \
		echo "   Install with: pip install pre-commit"; \
		echo "   Then run:     pre-commit install"; \
		echo ""; \
	fi; \
	if ! command -v cargo-audit > /dev/null 2>&1; then \
		echo "⚠  cargo-audit not found — push hooks won't run audit."; \
		echo "   Install with: cargo install cargo-audit"; \
		echo ""; \
	fi; \
	step_end=$$(date +%s); \
	echo "  ↳ pre-commit + cargo-audit: $$((step_end - step_start))s"; \
	\
	echo ""; echo "[3/3] cargo build…"; \
	step_start=$$(date +%s); \
	cargo build; \
	step_end=$$(date +%s); \
	echo "  ↳ cargo build: $$((step_end - step_start))s"; \
	\
	total_end=$$(date +%s); \
	echo ""; echo "=== setup completed in $$((total_end - total_start))s ==="

# Check all optional tools and print a summary with install commands for missing ones
check-deps:
	@echo "Checking optional tools…"; \
	echo ""; \
	missing=0; \
	check_tool() { \
		if command -v "$$1" > /dev/null 2>&1; then \
			printf "  ✓  %-20s found\n" "$$1"; \
		else \
			printf "  ✗  %-20s MISSING  →  %s\n" "$$1" "$$2"; \
			missing=$$((missing + 1)); \
		fi; \
	}; \
	check_tool cargo-watch    "cargo install cargo-watch"; \
	check_tool cargo-llvm-cov "cargo install cargo-llvm-cov"; \
	check_tool cargo-outdated "cargo install cargo-outdated"; \
	check_tool cargo-audit    "cargo install cargo-audit"; \
	check_tool pre-commit     "pip install pre-commit"; \
	check_tool maturin        "pip install maturin"; \
	echo ""; \
	if [ "$$missing" -eq 0 ]; then \
		echo "All optional tools are installed."; \
	else \
		echo "$$missing tool(s) missing. Run 'make setup-tools' to install Cargo tools."; \
	fi

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

# Scaffold a new example: make new-example NAME=my_search_demo
new-example:
	@test -n "$(NAME)" || { echo "Usage: make new-example NAME=<example_name>"; exit 1; }
	./scripts/new-example.sh $(NAME)

# Quick CI gate: fmt-check + lint (all-targets) + unit tests (~3 min)
check-quick:
	@total_start=$$(date +%s); \
	echo "=== make check-quick ==="; \
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
	echo ""; echo "=== check-quick completed in $$((total_end - total_start))s ==="

# Alias for check-quick — recommended pre-push command (mirrors CI fast gate)
check-local: check-quick

# Compile all examples and run those that don't require external services
verify-examples:
	@echo "=== Compiling all examples (--features full) ==="
	cargo build --examples --features full
	@echo ""
	@echo "=== Running examples that don't require external services ==="
	@failed=0; \
	for example in quickstart basic_usage filtered_search persistence quantization \
	               multi_tenant multi_vector sparse_vectors image_search sharding \
	               hybrid_search encryption_usage rag_chatbot metrics_usage diskann_usage; do \
		echo ""; echo "→ $$example"; \
		if cargo run --example "$$example" --features full 2>&1; then \
			echo "  ✓ $$example passed"; \
		else \
			echo "  ✗ $$example failed"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	if [ "$$failed" -gt 0 ]; then \
		echo "=== $$failed example(s) failed ==="; \
		exit 1; \
	else \
		echo "=== All examples passed ==="; \
	fi

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
	echo "Per-Directory Breakdown (files / lines / unwrap / expect)"; \
	for dir in $$(find src/ -mindepth 1 -maxdepth 1 -type d | sort); do \
		d_files=$$(find "$$dir" -name '*.rs' | wc -l | tr -d ' '); \
		d_lines=$$(find "$$dir" -name '*.rs' -exec cat {} + 2>/dev/null | wc -l | tr -d ' '); \
		d_unwrap=$$(grep -r 'unwrap()' "$$dir" --include='*.rs' 2>/dev/null | wc -l | tr -d ' '); \
		d_expect=$$(grep -r 'expect(' "$$dir" --include='*.rs' 2>/dev/null | wc -l | tr -d ' '); \
		printf "  %-30s %4s files  %6s lines  %4s unwrap()  %4s expect()\n" "$$dir" "$$d_files" "$$d_lines" "$$d_unwrap" "$$d_expect"; \
	done; \
	root_files=$$(find src/ -maxdepth 1 -name '*.rs' | wc -l | tr -d ' '); \
	root_lines=$$(find src/ -maxdepth 1 -name '*.rs' -exec cat {} + 2>/dev/null | wc -l | tr -d ' '); \
	root_unwrap=$$(grep -r 'unwrap()' src/ --maxdepth 1 --include='*.rs' 2>/dev/null | wc -l | tr -d ' '); \
	root_expect=$$(grep -r 'expect(' src/ --maxdepth 1 --include='*.rs' 2>/dev/null | wc -l | tr -d ' '); \
	printf "  %-30s %4s files  %6s lines  %4s unwrap()  %4s expect()\n" "src/ (root)" "$$root_files" "$$root_lines" "$$root_unwrap" "$$root_expect"

THRESHOLD ?= 1000

# List all .rs files above THRESHOLD lines (default: 1000), sorted by size
list-large-files:
	@echo "Rust files in src/ exceeding $(THRESHOLD) lines:"; \
	echo ""; \
	find src/ -name '*.rs' -exec wc -l {} + | grep -v ' total$$' | sort -rn | \
		awk -v threshold=$(THRESHOLD) '$$1 > threshold { printf "  %6d  %s\n", $$1, $$2; count++ } END { print ""; print count+0 " file(s) above " threshold " lines" }'

# Scaffold a test module in an existing file: make scaffold-test FILE=src/indexing/hnsw.rs
scaffold-test:
	@test -n "$(FILE)" || { echo "Usage: make scaffold-test FILE=src/path/to/module.rs"; exit 1; }
	@test -f "$(FILE)" || { echo "Error: file '$(FILE)' does not exist."; exit 1; }
	@if grep -q '#\[cfg(test)\]' "$(FILE)"; then \
		echo "File '$(FILE)' already contains a #[cfg(test)] block — skipping."; \
	else \
		echo '' >> "$(FILE)"; \
		echo '#[cfg(test)]' >> "$(FILE)"; \
		echo 'mod tests {' >> "$(FILE)"; \
		echo '    use super::*;' >> "$(FILE)"; \
		echo '' >> "$(FILE)"; \
		echo '    #[test]' >> "$(FILE)"; \
		echo '    fn test_placeholder() {' >> "$(FILE)"; \
		echo '        // TODO: Replace with real tests' >> "$(FILE)"; \
		echo '        assert!(true);' >> "$(FILE)"; \
		echo '    }' >> "$(FILE)"; \
		echo '}' >> "$(FILE)"; \
		echo "✓ Added test scaffold to $(FILE)"; \
	fi

# Docker convenience targets
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose -f docker-compose.source.yml build

docker-logs:
	docker compose logs -f
