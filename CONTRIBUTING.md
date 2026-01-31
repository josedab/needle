# Contributing to Needle

Thank you for your interest in contributing to Needle! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Repository Layout](#repository-layout)
- [Architecture Overview](#architecture-overview)
- [CI Architecture](#ci-architecture)
- [Troubleshooting](#troubleshooting)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

---

## Getting Started

### 15-Minute Dev Quickstart

```bash
git clone https://github.com/anthropics/needle.git
cd needle
./scripts/doctor.sh
cargo build
cargo test --lib
```

If you use `just`:

```bash
cargo install just
just quick
```

### Prerequisites

- Rust 1.85+ (install via [rustup](https://rustup.rs/))
- Git
- (Optional) Docker for integration testing
- (Optional) cargo-fuzz for fuzzing (requires nightly Rust)

### First-Time Setup

```bash
# Clone the repository
git clone https://github.com/anthropics/needle.git
cd needle

# Build the project
cargo build

# Run tests (unit tests + basic integration tests)
cargo test

# Run all tests including experimental/enterprise features
cargo t          # alias for: cargo test --features full

# Run with all features
cargo build --features full
```

---

## Development Setup

### Building

> **Tip**: The Python and WASM bindings live in `crates/needle-python/` (cdylib).
> The root library uses `crate-type = ["rlib"]` so regular `cargo build` and
> `cargo test` are fast. Build Python bindings with `maturin develop`.

> **Faster builds**: For faster incremental builds, configure a faster linker in
> `.cargo/config.toml`. See the commented-out platform-specific sections for
> `mold` (Linux) or `lld` (macOS) configuration.

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Build with specific features
cargo build --features server,metrics
cargo build --features full  # All features
```

### Running

```bash
# Run CLI
cargo run -- info mydb.needle

# Run HTTP server
cargo run --features server -- serve -a 127.0.0.1:8080

# Run with debug logging
RUST_LOG=debug cargo run --features server -- serve
```

### Useful Commands

```bash
# Format code
cargo fmt

# Lint
cargo clippy
cargo clippy --features full

# Run all tests
cargo test
cargo test --features full

# Run specific test
cargo test test_name

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

### Task Runner (Make or Just)

A `Makefile` is included so you can run common tasks without installing anything:

```bash
make help          # Show all available recipes
make quick         # Fast feedback: format check + lint + unit tests
make check         # Full pre-commit: format check + lint + all tests
make serve         # Run the HTTP server locally
make doc           # Generate and open documentation
make doctor        # Check local environment setup
```

Alternatively, install [just](https://github.com/casey/just) (`cargo install just`) for the same recipes:

```bash
just --list        # Show all available recipes
just quick         # Fast feedback
just check         # Full pre-commit
```

### Pre-commit Hooks

We provide pre-commit hooks for automatic formatting and linting:

```bash
pip install pre-commit
pre-commit install
```

This runs `cargo fmt`, `cargo clippy`, and file hygiene checks on every commit.
`cargo test --lib` and `cargo audit` run on push.

---

## Making Changes

### Branching Strategy

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes with clear, focused commits

3. Keep commits atomic and well-described

### Commit Messages

Follow conventional commit format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `perf`: Performance improvements
- `chore`: Maintenance tasks

Examples:
```
feat(hnsw): add batch insert optimization
fix(metadata): handle null values in filter parsing
docs(readme): add hybrid search example
```

---

## Testing

### Running Tests

```bash
# All tests
cargo test

# With all features
cargo test --features full

# Specific test file
cargo test --test property_tests

# With output
cargo test -- --nocapture
```

### Writing Tests

Place tests in the appropriate location:

- **Unit tests**: In `#[cfg(test)] mod tests` within each module
- **Integration tests**: In `tests/` directory
- **Property-based tests**: In `tests/property_tests.rs`

Example unit test:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_insert() {
        let db = Database::in_memory();
        db.create_collection("test", 128).unwrap();

        let collection = db.collection("test").unwrap();
        let vector = vec![0.1; 128];

        collection.insert("doc1", &vector, None).unwrap();

        assert!(collection.contains("doc1"));
    }
}
```

### Fuzzing

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Run a fuzz target
cargo +nightly fuzz run fuzz_metadata_filter

# Available targets:
# - fuzz_query_parser
# - fuzz_nl_filter
# - fuzz_metadata_filter
# - fuzz_distance
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- search
```

---

## Code Style

### Formatting

We use `rustfmt` with default settings:

```bash
cargo fmt
```

### Linting

All code must pass clippy:

```bash
cargo clippy --features full -- -D warnings
```

### Guidelines

1. **Error Handling**: Use `Result<T>` with `NeedleError` for fallible operations
   ```rust
   pub fn example() -> Result<()> {
       self.validate()?;
       Ok(())
   }
   ```

   **`unwrap()` and `expect()` policy:**
   - `unwrap()` is **denied** by workspace lint (`clippy::unwrap_used = "deny"`).
     Use `?` with `Result<T>` or pattern matching instead.
   - **Tech debt note:** Many existing modules have per-module `#[allow(clippy::unwrap_used)]`
     overrides (marked with `// tech debt: unwrap cleanup needed` in `src/lib.rs`).
     These allow the existing `unwrap()` calls while ensuring **new** modules and crates
     are protected by the workspace deny lint. When working in a module that has this
     override, prefer `?` for new code even though `unwrap()` is allowed.
   - `expect()` is audited in CI. Prefer proper error propagation over `expect()`.
   - **Acceptable uses of `expect()`:** Lock poisoning recovery (`lock().expect("...")`),
     compile-time-proven invariants, or one-time init. Add `// allow-expect` on the same
     line to suppress the CI warning.
   - `unwrap()` and `expect()` are fine in `#[cfg(test)]` modules and `src/experimental/`.

2. **Documentation**: Add rustdoc comments for public APIs
   ```rust
   /// Searches for similar vectors.
   ///
   /// # Arguments
   /// * `query` - The query vector
   /// * `k` - Number of results to return
   ///
   /// # Returns
   /// A vector of search results sorted by distance
   pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
       // ...
   }
   ```

3. **Naming**:
   - Use `snake_case` for functions and variables
   - Use `CamelCase` for types and traits
   - Use `SCREAMING_SNAKE_CASE` for constants

4. **Imports**: Group imports in order:
   - Standard library
   - External crates
   - Internal modules

---

## Pull Request Process

### Before Submitting

Run the quick check (format + lint + unit tests):

```bash
make quick         # Fast iteration (~2 min)
```

Before your final push, run the full suite:

```bash
make check         # Full pre-commit: format + lint + all tests
```

Or run the individual commands if you prefer:

1. Ensure all tests pass: `cargo test --features full`
2. Run formatter: `cargo fmt`
3. Run linter: `cargo clippy --features full -- -D warnings`
4. Update documentation if needed
5. Add tests for new functionality

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes.

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Submit PR against `main` branch
2. Automated CI will run tests
3. Maintainer review
4. Address feedback
5. Merge when approved

---

## Repository Layout

### Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Main library and binary source code |
| `crates/` | Workspace sub-crates (`needle-core`, `needle-cli`, `needle-python`) |
| `tests/` | Integration, property-based, and edge-case tests |
| `examples/` | Runnable example programs for each feature |
| `benches/` | Criterion benchmarks |
| `docs/` | Supplementary documentation (OpenAPI spec, guides) |
| `proto/` | Protocol Buffer definitions |
| `sdk/` | Client SDK packages (TypeScript, etc.) |
| `python/` | Python package configuration and stubs |
| `scripts/` | Developer scripts (doctor, quickstart, scaffolding) |
| `deploy/` | Deployment configurations |
| `helm/` | Helm charts for Kubernetes |
| `fuzz/` | Fuzz testing targets (requires nightly Rust) |
| `website/` | Docusaurus documentation website |

### Root Configuration Files

| File | Purpose |
|------|---------|
| `clippy.toml` | Clippy lint configuration (e.g., type complexity threshold) |
| `rustfmt.toml` | Rust formatter settings |
| `rust-toolchain.toml` | Pinned Rust toolchain version for reproducible builds |
| `deny.toml` | `cargo-deny` config — license auditing and banned dependency checks |
| `codecov.yml` | Codecov coverage upload and threshold settings |
| `pyproject.toml` | Python package metadata (used by maturin for PyO3 bindings) |
| `mcp-registry.json` | Model Context Protocol tool manifest for AI agent integration |
| `docker-compose.yml` | Docker Compose for running Needle from pre-built images |
| `docker-compose.source.yml` | Docker Compose for building from source |
| `.pre-commit-config.yaml` | Pre-commit hook definitions (fmt, clippy, audit) |

---

## Architecture Overview

Understanding the codebase structure will help you contribute effectively.

### Key Modules

| Module | Purpose |
|--------|---------|
| `collection/mod.rs` | Vector storage, search, and collection management |
| `database/mod.rs` | Database management, persistence, thread-safe access |
| `database/collection_ref.rs` | Thread-safe `CollectionRef` handle |
| `indexing/hnsw.rs` | HNSW index implementation |
| `metadata.rs` | Metadata storage and filtering |
| `distance.rs` | Distance functions (Cosine, Euclidean, Dot, Manhattan) |
| `server.rs` | HTTP REST API (feature: server) |
| `error.rs` | Error types with structured codes |

### Adding Features

#### MCP Integration

Needle exposes an [MCP](https://modelcontextprotocol.io/) server so AI agents can
manage vector collections directly. The tool manifest lives in
[`mcp-registry.json`](../mcp-registry.json) at the repository root. When you add or
rename an MCP tool, update this file so external registries stay in sync.

#### New Error Type

```rust
// In src/error.rs
#[derive(Error, Debug)]
pub enum NeedleError {
    // ... existing variants
    #[error("My new error: {0}")]
    MyNewError(String),
}
```

#### New Collection Method

1. Add method to `Collection` in `src/collection/mod.rs`
2. Add internal method to `Database`
3. Add public method to `CollectionRef`
4. Export in `src/lib.rs` if public

#### New CLI Command

1. Add variant to `Commands` enum in `src/main.rs`
2. Add match arm in `main()`
3. Implement handler function

#### New REST Endpoint

1. Add handler in `src/server.rs`
2. Register route in `create_router()`
3. Add request/response types

### Thread Safety

Needle uses `parking_lot::RwLock` for concurrency:

```rust
// Read lock (multiple readers allowed)
let guard = self.inner.read();

// Write lock (exclusive access)
let mut guard = self.inner.write();
```

---

## CI Architecture

Our CI pipeline is structured in stages so fast checks gate slower ones:

1. **Quick Check (~2 min)** — Format, clippy, unit tests. Gates all other jobs. Mirrors `make quick` locally.
2. **Parallel jobs (~5-10 min)** — Test matrix (OS × Rust version), full lint, docs, benchmarks, examples, coverage, MSRV.
3. **Feature Matrix (~10 min)** — Tests 8 feature-flag combinations and cross-compile targets (separate workflow: `feature-matrix.yml`).

**Informational / advisory:**
- Security scans (`security.yml`): cargo-audit, cargo-deny, semver checks (advisory until v1.0).
- Coverage uploads to Codecov.

Run `make quick` locally before pushing — it mirrors the fast gate and catches most issues in under 2 minutes.

---

## Troubleshooting

### Build fails with "feature `X` required"

Many modules are behind feature flags. If you see missing type or module errors:

```bash
cargo build --features full    # Enable all features
cargo test --features full     # Test with all features
```

### Rust version mismatch

Needle requires Rust 1.85+. Check your version:

```bash
rustc --version
rustup update stable
```

### cdylib linker errors with `cargo test`

The Python bindings crate (`crates/needle-python`) uses `crate-type = ["cdylib"]` which can cause linker issues. Use `--lib` or target the root package:

```bash
cargo test -p needle --lib     # Test library only
cargo test -p needle           # Test root package
```

### `cargo clippy` warnings on clean checkout

Clippy pedantic is enabled workspace-wide. Some warnings are expected in service modules. Run with the same flags CI uses:

```bash
cargo clippy --features full -- -D warnings
```

### Pre-commit hooks failing

If `pre-commit` hooks fail on install:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files     # Verify setup
```

### Tests fail with "database dropped with unsaved changes"

This warning is expected in tests using `Database::in_memory()` and can be ignored. Tests create in-memory databases that are intentionally not saved.

### Docker build fails

Ensure Docker is running and you have sufficient disk space:

```bash
docker compose build --no-cache
```

### Benchmark results vary wildly

Benchmarks are sensitive to system load. For reliable results:

```bash
# Close other applications, then:
cargo bench -- --warm-up-time 3
```

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@needle-db.io

---

## Recognition

Contributors will be acknowledged in release notes and the project README.

Thank you for contributing to Needle!
