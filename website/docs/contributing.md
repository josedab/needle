---
sidebar_position: 11
---

# Contributing to Needle

We welcome contributions to Needle! This guide will help you get started with contributing to the project.

## Ways to Contribute

- **Report bugs**: Found a bug? [Open an issue](https://github.com/anthropics/needle/issues/new?template=bug_report.md)
- **Suggest features**: Have an idea? [Start a discussion](https://github.com/anthropics/needle/discussions/new?category=ideas)
- **Submit fixes**: Fix a bug or typo and submit a pull request
- **Improve docs**: Help make our documentation better
- **Write tests**: Improve test coverage
- **Share examples**: Create example projects or tutorials

## Development Setup

### Prerequisites

- Rust 1.85 or later
- Git
- (Optional) Python 3.8+ for Python binding development
- (Optional) Node.js 18+ for WASM binding development

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/anthropics/needle.git
cd needle

# Build the project
cargo build

# Run tests
cargo test

# Build with all features
cargo build --features full
```

### Project Structure

```
needle/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── main.rs             # CLI application
│   ├── collection/mod.rs   # Collection implementation
│   ├── database/mod.rs     # Database management
│   ├── indexing/hnsw.rs    # HNSW index
│   ├── distance.rs         # Distance functions
│   ├── metadata.rs         # Metadata filtering
│   ├── enterprise/         # Encryption, RBAC, Raft
│   ├── persistence/        # Backup, WAL, versioning
│   ├── search/             # Query planning, reranking
│   └── ...
├── tests/              # Integration tests (12 files)
├── benches/            # Benchmarks
├── crates/             # Workspace crates (core, cli, python)
└── website/            # Documentation site
```

## Code Style

### Rust Guidelines

We follow the standard Rust style guidelines with some additions:

```rust
// Use descriptive variable names
let vector_count = collection.count()?;  // Good
let n = collection.count()?;              // Avoid

// Document public APIs
/// Searches for the k nearest neighbors to the query vector.
///
/// # Arguments
/// * `query` - The query vector
/// * `k` - Number of results to return
///
/// # Returns
/// A vector of search results sorted by distance
pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>

// Handle errors properly
collection.insert(id, &vector, Some(metadata))?;  // Propagate errors
```

### Formatting

```bash
# Format code
cargo fmt

# Check formatting without changes
cargo fmt --check

# Run clippy for lints
cargo clippy --all-targets --all-features
```

### Commit Messages

Write clear, concise commit messages:

```
feat: add batch insert operation

- Implement batch_insert() method on Collection
- Add parallel processing with Rayon
- Include benchmarks for batch operations

Closes #123
```

**Prefixes:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions or fixes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_search_basic

# Run tests with output
cargo test -- --nocapture

# Run tests with specific features
cargo test --features full

# Run property-based tests
cargo test --test property_tests
```

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        // Use in-memory database for tests
        let db = Database::in_memory();
        db.create_collection("test", 4).unwrap();
        let collection = db.collection("test").unwrap();

        // Insert test data
        collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({}))).unwrap();

        // Verify behavior
        let results = collection.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }
}
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench search_benchmark
```

## Pull Request Process

### Before Submitting

1. **Create an issue first** for significant changes
2. **Fork the repository** and create a feature branch
3. **Write tests** for your changes
4. **Update documentation** if needed
5. **Run the test suite** and ensure all tests pass
6. **Format your code** with `cargo fmt`
7. **Run clippy** and fix any warnings

### Submitting a PR

1. Push your branch to your fork
2. Create a pull request against `main`
3. Fill out the PR template
4. Link any related issues
5. Wait for CI to pass
6. Address review feedback

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
How did you test these changes?

## Checklist
- [ ] Tests pass locally
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation updated (if needed)
```

## Feature Flags

When adding new functionality, consider if it should be behind a feature flag:

```toml
# In Cargo.toml
[features]
my_feature = ["optional-dependency"]
```

```rust
// In code
#[cfg(feature = "my_feature")]
pub fn my_new_function() {
    // Implementation
}
```

## Documentation

### Code Documentation

Document all public APIs with rustdoc:

```rust
/// Brief description of the function.
///
/// Longer description if needed, explaining behavior,
/// edge cases, and usage patterns.
///
/// # Arguments
///
/// * `param1` - Description of param1
/// * `param2` - Description of param2
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// List of possible errors
///
/// # Examples
///
/// ```rust
/// let result = my_function(arg1, arg2)?;
/// ```
pub fn my_function(param1: Type1, param2: Type2) -> Result<ReturnType>
```

### Website Documentation

The documentation website is in the `website/` directory:

```bash
cd website
npm install
npm start  # Start development server
```

## Getting Help

- **Questions**: [GitHub Discussions](https://github.com/anthropics/needle/discussions)
- **Bugs**: [GitHub Issues](https://github.com/anthropics/needle/issues)
- **Chat**: [Discord](https://discord.gg/anthropic)

## Recognition

Contributors are recognized in:
- The project's README
- Release notes when their changes are included
- The contributors page on GitHub

Thank you for contributing to Needle!
