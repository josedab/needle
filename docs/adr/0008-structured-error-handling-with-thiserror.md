# ADR-0008: Structured Error Handling with Thiserror

## Status

Accepted

## Context

Error handling in Rust requires choosing between several approaches:

| Approach | Type Safety | Context | Ergonomics | Dependencies |
|----------|-------------|---------|------------|--------------|
| `String` errors | None | Unstructured | Poor | None |
| `Box<dyn Error>` | Minimal | Lost | Medium | None |
| `anyhow::Error` | Minimal | Preserved | Excellent | anyhow |
| Custom enum + thiserror | Full | Structured | Excellent | thiserror |
| Custom enum (manual) | Full | Structured | Poor | None |

For a library like Needle, error handling must:

1. **Be type-safe** — Callers should match on specific error variants
2. **Carry context** — Errors should include relevant details (expected vs actual dimensions, checksums)
3. **Propagate cleanly** — The `?` operator should work without boilerplate
4. **Display well** — Error messages should be human-readable
5. **Be stable** — Error types are part of the public API

## Decision

Use the **`thiserror`** crate to derive error implementations for a structured `NeedleError` enum.

### Error Type Definition

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum NeedleError {
    #[error("Collection '{0}' not found")]
    CollectionNotFound(String),

    #[error("Collection '{0}' already exists")]
    CollectionExists(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Vector '{0}' not found")]
    VectorNotFound(String),

    #[error("Invalid HNSW parameters: {0}")]
    InvalidParameters(String),

    #[error("Storage corruption: expected checksum {expected:#x}, got {actual:#x}")]
    Corruption { expected: u32, actual: u32 },

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("Index not built - call build_index() first")]
    IndexNotBuilt,

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Filter parse error: {0}")]
    FilterParse(String),

    #[error("Operation requires exclusive access")]
    ConcurrentModification,
}

/// Result type alias for Needle operations
pub type Result<T> = std::result::Result<T, NeedleError>;
```

### Key Features

**Structured context fields:**
```rust
// Instead of: "Dimension mismatch"
// We get: "Dimension mismatch: expected 384, got 768"
DimensionMismatch { expected: usize, got: usize }

// Instead of: "Checksum failed"
// We get: "Storage corruption: expected checksum 0xdeadbeef, got 0x12345678"
Corruption { expected: u32, actual: u32 }
```

**Automatic From implementations:**
```rust
#[error("I/O error: {0}")]
Io(#[from] std::io::Error),  // Enables: io_operation()?

#[error("Serialization error: {0}")]
Serialization(#[from] serde_json::Error),  // Enables: serde_json::to_string()?
```

**Pattern matching for callers:**
```rust
match db.search("embeddings", &query, 10) {
    Ok(results) => process(results),
    Err(NeedleError::CollectionNotFound(name)) => {
        eprintln!("Collection '{}' does not exist", name);
    }
    Err(NeedleError::DimensionMismatch { expected, got }) => {
        eprintln!("Query has {} dimensions but collection expects {}", got, expected);
    }
    Err(e) => return Err(e.into()),
}
```

### Code References

- `src/error.rs:1-78` — NeedleError enum definition
- `src/error.rs:77` — Result type alias
- `Cargo.toml:24` — `thiserror = "2.0"` dependency

## Consequences

### Benefits

1. **Zero runtime cost** — thiserror generates code at compile time
2. **Automatic Display** — `#[error("...")]` generates Display implementation
3. **Automatic From** — `#[from]` generates From implementations for `?`
4. **Type-level distinction** — Callers can handle specific errors differently
5. **Rich context** — Structured fields provide debugging information
6. **Stable API** — Error variants are documented and versioned

### Tradeoffs

1. **Compile-time dependency** — thiserror must be compiled (but it's lightweight)
2. **API commitment** — Error variants become public API, hard to change
3. **Verbosity** — Each error type needs explicit definition
4. **No backtraces** — thiserror doesn't capture backtraces (unlike anyhow)

### What This Enabled

- CLI error messages with actionable context
- HTTP API returning structured error responses
- Programmatic error handling in library consumers
- Consistent error formatting across all interfaces

### What This Prevented

- Dynamic error types (all errors must be NeedleError variants)
- Automatic backtrace capture (would need anyhow or manual implementation)
- Error chaining beyond one level (each From converts, losing original type)

### Error Handling Patterns

**Internal functions:**
```rust
fn validate_dimensions(&self, vector: &[f32]) -> Result<()> {
    if vector.len() != self.dimensions {
        return Err(NeedleError::DimensionMismatch {
            expected: self.dimensions,
            got: vector.len(),
        });
    }
    Ok(())
}
```

**Propagation with context:**
```rust
fn load_collection(&self, name: &str) -> Result<Collection> {
    let path = self.collection_path(name);
    let data = std::fs::read(&path)?;  // Io error via #[from]
    let collection: Collection = serde_json::from_slice(&data)?;  // Serialization via #[from]
    Ok(collection)
}
```

**CLI error display:**
```rust
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);  // Uses Display from #[error]
        std::process::exit(1);
    }
}
```

### Comparison with Alternatives

**Why not anyhow?**
- anyhow is designed for applications, not libraries
- Loses type information (callers can't match on variants)
- Better for quick prototyping, not stable APIs

**Why not manual impl Error?**
- Requires ~20 lines of boilerplate per variant
- Easy to make mistakes in Display/Error implementations
- thiserror generates the same code with less maintenance burden

**Why not String errors?**
- No type safety (`Result<T, String>` can't be pattern-matched)
- Context is unstructured and inconsistent
- Poor developer experience for library consumers
