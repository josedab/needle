# ADR-0006: Async API as Optional Feature Layer

## Status

Accepted

## Context

Needle serves multiple deployment scenarios with different concurrency requirements:

| Use Case | Concurrency Model | Runtime Needs |
|----------|-------------------|---------------|
| Embedded library | Sync, single-threaded | None |
| CLI tool | Sync, single-threaded | None |
| Batch processing | Sync, multi-threaded (Rayon) | None |
| HTTP server | Async, event-driven | Tokio |
| gRPC server | Async, event-driven | Tokio |
| WebAssembly | Sync or async | Browser/Node |

The Rust ecosystem is split between synchronous and asynchronous code:

- **Sync code** is simpler, has no runtime overhead, and works everywhere
- **Async code** is more efficient for I/O-bound workloads but requires a runtime (tokio, async-std)

Forcing async on all users would:
1. Add mandatory dependencies (tokio ~1MB compiled)
2. Require async runtimes even for simple embedded use
3. Complicate the API with `.await` everywhere
4. Prevent use in contexts without async support (some embedded systems, WASM without async)

## Decision

Provide **synchronous API as the default** with an **optional async API** via Cargo feature flags.

### Feature Hierarchy

```toml
[features]
default = []

# Base async support
async = ["tokio", "futures"]

# HTTP server (requires async)
server = ["async", "axum", "tower", "tower-http"]

# Full feature set
full = ["server", "metrics", "hybrid"]
```

### API Design

**Synchronous API (default):**
```rust
use needle::Database;

let db = Database::open("vectors.needle")?;
let results = db.search("collection", &query, 10)?;  // Blocking
db.save()?;  // Blocking
```

**Asynchronous API (feature: async):**
```rust
use needle::async_api::AsyncDatabase;

let db = AsyncDatabase::open("vectors.needle").await?;
let results = db.search("collection", &query, 10).await?;  // Non-blocking
db.save().await?;  // Non-blocking
```

### Implementation Strategy

The async API wraps the sync API using `tokio::task::spawn_blocking` for CPU-bound operations:

```rust
// src/async_api.rs
impl AsyncDatabase {
    pub async fn search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        let db = self.inner.clone();
        let collection = collection.to_string();
        let query = query.to_vec();

        tokio::task::spawn_blocking(move || {
            db.search(&collection, &query, k)
        }).await?
    }
}
```

This approach:
- Reuses all sync implementation logic
- Prevents blocking the async runtime with CPU-bound search
- Maintains API parity between sync and async

### Code References

- `Cargo.toml:129-130` — async feature definition
- `src/lib.rs:126-127` — Conditional async_api module export
- `src/async_api.rs` — AsyncDatabase implementation
- Feature dependency chain: `async` → `server` → `web-ui`

## Consequences

### Benefits

1. **Zero overhead for sync users** — No tokio dependency, no runtime initialization
2. **Opt-in complexity** — Async users explicitly choose the async API
3. **Shared implementation** — Async wraps sync, avoiding code duplication
4. **Flexible deployment** — Same crate works embedded, CLI, or server
5. **Compile-time selection** — Features are resolved at build time, not runtime
6. **Smaller binaries** — Sync-only builds exclude async dependencies

### Tradeoffs

1. **API duplication** — Two sets of types (Database, AsyncDatabase) to maintain
2. **spawn_blocking overhead** — Thread pool handoff adds latency (~microseconds)
3. **Feature flag complexity** — Users must understand feature dependencies
4. **Testing burden** — Both APIs need testing, increasing CI time

### What This Enabled

- Embedded use without runtime dependencies
- HTTP server with efficient async I/O (axum)
- Future gRPC support via tonic (async-native)
- WebAssembly builds without tokio (WASM feature is separate)

### What This Prevented

- Single unified API (would force async on everyone or sync on servers)
- True async I/O for storage (file I/O uses spawn_blocking, not async fs)
- Zero-copy async streaming (would need different architecture)

### Dependency Impact

| Feature Set | Additional Dependencies | Binary Size Impact |
|-------------|------------------------|-------------------|
| default | None | Baseline |
| async | tokio, futures | +800KB |
| server | + axum, tower, hyper | +1.5MB |
| full | + prometheus, rust-stemmers | +2MB |

### Usage Patterns

**Embedded application (sync):**
```rust
// Cargo.toml: needle = "0.1"
use needle::Database;

fn process_query(db: &Database, query: &[f32]) -> Vec<SearchResult> {
    db.search("embeddings", query, 10).unwrap()
}
```

**Web server (async):**
```rust
// Cargo.toml: needle = { version = "0.1", features = ["server"] }
use needle::async_api::AsyncDatabase;
use axum::{Router, routing::post};

async fn search_handler(db: AsyncDatabase, query: Query) -> Json<Results> {
    let results = db.search("embeddings", &query.vector, 10).await?;
    Json(results)
}
```

### Future Considerations

If true async I/O becomes necessary (e.g., for cloud storage backends), the async API could be reimplemented with:
- `tokio::fs` for async file operations
- `object_store` crate for S3/GCS async access
- Custom async HNSW traversal

The current spawn_blocking approach provides a migration path without breaking API compatibility.
