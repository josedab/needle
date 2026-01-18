# ADR-0004: Generation-Based Dirty Tracking

## Status

Accepted

## Context

Needle needs to track whether the in-memory database state has unsaved changes. This "dirty tracking" is essential for:

1. **Efficient saves** — Only write to disk when changes exist
2. **User feedback** — Warn users about unsaved changes before exit
3. **Auto-save logic** — Trigger periodic saves only when needed
4. **Replication** — Know when local state diverges from persisted state

A naive approach using a simple boolean flag has a race condition:

```rust
// PROBLEMATIC: Simple dirty flag
struct Database {
    dirty: AtomicBool,
}

impl Database {
    fn insert(&self, ...) {
        // ... insert logic ...
        self.dirty.store(true, Ordering::Release);
    }

    fn save(&self) {
        if self.dirty.load(Ordering::Acquire) {
            // ... write to disk ...
            self.dirty.store(false, Ordering::Release);  // BUG!
        }
    }
}
```

**The bug:** If `insert()` is called between the disk write and `dirty.store(false)`, the new change is silently lost — the flag is cleared even though unsaved changes exist.

## Decision

Implement **generation-based dirty tracking** using two atomic counters with compare-and-swap (CAS) operations:

```rust
pub struct Database {
    state: Arc<RwLock<DatabaseState>>,

    /// Incremented on every modification
    modification_gen: AtomicU64,

    /// Set to modification_gen value after successful save
    saved_gen: AtomicU64,
}
```

### Core Operations

**Marking modifications:**
```rust
fn mark_modified(&self) {
    self.modification_gen.fetch_add(1, Ordering::Release);
}
```

**Checking dirty state:**
```rust
fn is_dirty(&self) -> bool {
    let mod_gen = self.modification_gen.load(Ordering::Acquire);
    let saved_gen = self.saved_gen.load(Ordering::Acquire);
    mod_gen > saved_gen
}
```

**Saving with CAS loop:**
```rust
fn save(&self) -> Result<()> {
    // Capture current generation before save
    let gen_before_save = self.modification_gen.load(Ordering::Acquire);

    // Perform the actual save
    self.write_to_disk()?;

    // CAS loop: only update saved_gen if no concurrent modifications
    loop {
        let current_saved = self.saved_gen.load(Ordering::Acquire);

        // Only update if we're advancing the generation
        if gen_before_save <= current_saved {
            break;  // Another save already recorded a higher generation
        }

        match self.saved_gen.compare_exchange(
            current_saved,
            gen_before_save,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => break,  // Successfully updated
            Err(_) => continue,  // Retry CAS
        }
    }

    Ok(())
}
```

### Code References

- `src/database.rs:150-155` — Generation counter definitions
- `src/database.rs:652-655` — `mark_modified()` implementation
- `src/database.rs:591-606` — CAS loop in save()
- `src/database.rs:644-649` — `is_dirty()` comparison logic

## Consequences

### Benefits

1. **Race-condition free** — Concurrent modifications during save keep the dirty flag set
2. **Lock-free tracking** — No additional locks needed beyond the existing RwLock
3. **ABA-problem immune** — u64 counter won't realistically wrap around (584 years at 1B ops/sec)
4. **Precise dirty detection** — Exact equality check possible, not just boolean
5. **Debuggable** — Generation numbers help trace modification history
6. **Composable** — Works correctly with any number of concurrent writers

### Tradeoffs

1. **Memory overhead** — Two AtomicU64 (16 bytes) vs one AtomicBool (1 byte)
2. **Complexity** — CAS loop is harder to understand than simple store
3. **No modification details** — Tracks *that* changes occurred, not *what* changed

### Why This Works

Consider the race scenario that broke the simple boolean:

```
Thread A (save)              Thread B (insert)
─────────────────            ─────────────────
gen_before = 5
write_to_disk()
                             modification_gen = 6

CAS(saved_gen, 0→5)

is_dirty()?
  mod_gen=6, saved_gen=5
  6 > 5 → TRUE ✓
```

The generation approach correctly identifies that Thread B's modification is unsaved.

### Memory Ordering Rationale

- **Release on modification** — Ensures modification is visible before generation increment
- **Acquire on read** — Ensures we see all modifications before the generation we read
- **AcqRel on CAS** — Combines both for the read-modify-write operation

### What This Enabled

- Safe auto-save functionality without lost updates
- Accurate "unsaved changes" warnings in CLI
- Replication leader can track which changes need propagation
- Multiple concurrent writers with correct dirty state

### What This Prevented

- Per-field change tracking (would need more complex structures)
- Change timestamps (generations are ordinal, not temporal)
- Undo/redo functionality (would need change logs)

### Alternative Considered: Timestamps

Using timestamps instead of generations was considered:

```rust
modification_time: AtomicU64,  // Unix timestamp
saved_time: AtomicU64,
```

**Rejected because:**
- Clock skew between threads could cause incorrect comparisons
- Two modifications in the same millisecond would have the same timestamp
- Monotonic clocks solve some issues but add platform complexity

Generations are simpler and more reliable for this use case.
