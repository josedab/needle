# ADR-0003: Interior Mutability with Parking Lot RwLock

## Status

Accepted

## Context

Needle is designed to be used in multi-threaded environments where:

1. **Multiple readers** — Search operations are read-only and should execute concurrently
2. **Single writer** — Insert/update/delete operations modify state and need exclusive access
3. **Shared ownership** — The database handle should be clonable and usable from multiple threads
4. **Safe API** — Users should not need `unsafe` code or manual synchronization

Rust's ownership system prevents shared mutable state by default. To provide a thread-safe API, we needed to choose between several concurrency primitives:

| Approach | Reader Concurrency | Writer Fairness | Performance |
|----------|-------------------|-----------------|-------------|
| `Mutex<T>` | No (readers block each other) | N/A | Baseline |
| `std::sync::RwLock<T>` | Yes | Writer starvation possible | Good |
| `parking_lot::RwLock<T>` | Yes | Fair (writers prioritized) | Better |
| Lock-free structures | Yes | N/A | Best (but complex) |

## Decision

Use **`Arc<RwLock<DatabaseState>>`** from the `parking_lot` crate for interior mutability, combined with a **`CollectionRef`** wrapper for ergonomic, lifetime-bounded access.

### Core Pattern

```rust
pub struct Database {
    state: Arc<RwLock<DatabaseState>>,
    path: Option<PathBuf>,
    modification_gen: AtomicU64,
    saved_gen: AtomicU64,
}

struct DatabaseState {
    collections: HashMap<String, Collection>,
    // ... other mutable state
}
```

### Lock Acquisition

```rust
// Read operations (concurrent)
pub fn search(&self, collection: &str, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
    let state = self.state.read();  // Multiple readers allowed
    let collection = state.collections.get(collection)?;
    collection.search(query, k)
}

// Write operations (exclusive)
pub fn insert(&self, collection: &str, id: &str, vector: Vec<f32>) -> Result<()> {
    let mut state = self.state.write();  // Exclusive access
    let collection = state.collections.get_mut(collection)?;
    collection.insert(id, vector)
}
```

### CollectionRef for Ergonomic Access

```rust
pub struct CollectionRef<'a> {
    db: &'a Database,
    name: String,
}

impl<'a> CollectionRef<'a> {
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.db.search_internal(&self.name, query, k)
    }

    pub fn insert(&self, id: &str, vector: Vec<f32>) -> Result<()> {
        self.db.insert_internal(&self.name, id, vector)
    }
}
```

This pattern:
- Binds the reference to the database lifetime
- Avoids repeated collection name lookups
- Provides a familiar "handle" API similar to database connections

### Code References

- `src/database.rs:147` — `state: Arc<RwLock<DatabaseState>>`
- `Cargo.toml:39-40` — `parking_lot = "0.12"`
- `src/database.rs:396,578,698-699` — Lock acquisition patterns
- `src/database.rs:1058-1061` — CollectionRef definition

## Consequences

### Benefits

1. **Reader scalability** — Search operations run in parallel without blocking each other
2. **Writer fairness** — parking_lot prevents writer starvation under heavy read load
3. **Performance** — parking_lot is 2-3x faster than std::sync::RwLock in benchmarks
4. **Memory efficiency** — parking_lot::RwLock is 1 word vs std's 40+ bytes
5. **Poison-free** — parking_lot doesn't poison locks on panic, simplifying recovery
6. **Safe API** — Users get thread safety without writing synchronization code
7. **Clone-friendly** — `Arc` enables cheap cloning of database handles

### Tradeoffs

1. **External dependency** — Adds parking_lot crate (though widely used and trusted)
2. **No async-aware locks** — For async code, tokio's RwLock may be preferable (addressed in ADR-0006)
3. **Coarse granularity** — Entire database locked, not per-collection (acceptable for current scale)
4. **Hidden contention** — Lock contention is not visible in the API; may cause unexpected latency

### What This Enabled

- Thread-safe `Database::clone()` for sharing across threads
- Concurrent `batch_search` via Rayon parallelization
- Safe `CollectionRef` API that feels like a database connection handle
- Straightforward testing without thread-safety concerns

### What This Prevented

- Per-collection locking (would increase complexity significantly)
- Lock-free search (would require immutable index snapshots)
- Async-native API in the core library (delegated to optional async feature)

### Design Principles Applied

1. **Interior mutability** — `&self` methods can modify state through the lock
2. **RAII guards** — Locks released automatically when guards drop
3. **Minimal lock scope** — Operations acquire locks only when needed, release promptly
4. **No lock ordering issues** — Single lock means no deadlock possibility

### Usage Example

```rust
use needle::Database;
use std::thread;

let db = Database::open("vectors.needle")?;

// Clone handle for each thread
let handles: Vec<_> = (0..4).map(|_| {
    let db = db.clone();
    thread::spawn(move || {
        // Concurrent searches
        db.search("embeddings", &query, 10)
    })
}).collect();

// All searches run in parallel
for handle in handles {
    let results = handle.join().unwrap()?;
}
```
