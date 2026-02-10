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
- [Architecture Overview](#architecture-overview)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

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

1. Ensure all tests pass: `cargo test --features full`
2. Run formatter: `cargo fmt`
3. Run linter: `cargo clippy --features full`
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

## Architecture Overview

Understanding the codebase structure will help you contribute effectively.

### Key Modules

| Module | Purpose |
|--------|---------|
| `database.rs` | Database management, collections |
| `collection.rs` | Vector storage and search |
| `hnsw.rs` | HNSW index implementation |
| `metadata.rs` | Metadata storage and filtering |
| `distance.rs` | Distance functions |
| `server.rs` | HTTP REST API |
| `error.rs` | Error types |

### Adding Features

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

1. Add method to `Collection` in `src/collection.rs`
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

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Security**: Email security@anthropic.com

---

## Recognition

Contributors will be acknowledged in release notes and the project README.

Thank you for contributing to Needle!
