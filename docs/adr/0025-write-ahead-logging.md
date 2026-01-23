# ADR-0025: Write-Ahead Logging for Crash Recovery

## Status

Accepted

## Context

Needle's single-file storage format (ADR-0001) provides simplicity but creates a durability challenge:

1. **Atomic writes are expensive** — Rewriting the entire file on every insert is prohibitively slow
2. **Partial writes corrupt data** — A crash during file write can leave the database in an inconsistent state
3. **Batching delays persistence** — Buffering writes improves performance but risks data loss

Traditional databases solve this with Write-Ahead Logging (WAL):
- Changes are first written to a log
- Log entries are fsynced to disk
- Actual data files are updated lazily
- On crash, the log is replayed to recover

### Alternatives Considered

1. **Copy-on-write (like SQLite WAL mode)** — Complex, requires page-level management
2. **Memory-mapped with msync** — OS-dependent behavior, hard to reason about
3. **External transaction log** — Adds operational complexity

## Decision

Needle implements an **optional Write-Ahead Log** with segment-based storage and checkpoint-based recovery:

### WAL Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Write Path                               │
│                                                                  │
│   insert(id, vector)                                            │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ WAL Append  │───▶│   fsync     │───▶│ In-Memory   │        │
│   │ (log entry) │    │  (durable)  │    │   Update    │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Checkpoint Path                             │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Threshold  │───▶│Write Main   │───▶│ Truncate    │        │
│   │  Reached    │    │   File      │    │    WAL      │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       Recovery Path                              │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │ Load Main   │───▶│  Scan WAL   │───▶│   Replay    │        │
│   │    File     │    │  Segments   │    │   Entries   │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### WAL Configuration

```rust
// src/wal.rs
#[derive(Debug, Clone)]
pub struct WalConfig {
    /// Directory for WAL segment files
    pub wal_dir: PathBuf,

    /// Maximum size of a single WAL segment (default: 64MB)
    pub segment_size: usize,

    /// Sync mode for durability guarantees
    pub sync_mode: SyncMode,

    /// Checkpoint trigger threshold (entries or bytes)
    pub checkpoint_threshold: CheckpointThreshold,

    /// Whether to compress WAL entries
    pub compression: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum SyncMode {
    /// fsync after every write (safest, slowest)
    EveryWrite,

    /// fsync after N writes (balanced)
    EveryN(usize),

    /// fsync on checkpoint only (fastest, least durable)
    OnCheckpoint,

    /// Let OS decide (fastest, may lose data on power loss)
    None,
}

#[derive(Debug, Clone)]
pub enum CheckpointThreshold {
    /// Checkpoint after N entries
    Entries(usize),

    /// Checkpoint after N bytes written
    Bytes(usize),

    /// Checkpoint after duration since last checkpoint
    Duration(Duration),

    /// Manual checkpoint only
    Manual,
}
```

### WAL Entry Format

```rust
// Each WAL entry is self-describing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalEntry {
    /// Log Sequence Number (monotonically increasing)
    pub lsn: u64,

    /// Timestamp of the entry
    pub timestamp: u64,

    /// CRC32 checksum for integrity
    pub checksum: u32,

    /// The operation
    pub operation: WalOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalOperation {
    /// Insert a new vector
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },

    /// Update an existing vector
    Update {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },

    /// Delete a vector
    Delete {
        collection: String,
        id: String,
    },

    /// Create a collection
    CreateCollection {
        name: String,
        dimensions: usize,
        config: CollectionConfig,
    },

    /// Drop a collection
    DropCollection {
        name: String,
    },

    /// Transaction boundary markers
    BeginTransaction { txn_id: u64 },
    CommitTransaction { txn_id: u64 },
    AbortTransaction { txn_id: u64 },

    /// Checkpoint marker
    Checkpoint { lsn: u64 },
}
```

### Segment Management

WAL is divided into segments for efficient truncation:

```rust
// src/wal.rs
pub struct WalManager {
    config: WalConfig,
    current_segment: WalSegment,
    segments: Vec<WalSegmentInfo>,
    next_lsn: AtomicU64,
}

pub struct WalSegment {
    id: u64,
    file: File,
    size: usize,
    entry_count: usize,
}

impl WalManager {
    /// Append an entry to the WAL
    pub fn append(&mut self, operation: WalOperation) -> Result<u64> {
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);

        let entry = WalEntry {
            lsn,
            timestamp: current_timestamp(),
            checksum: 0, // Computed below
            operation,
        };

        let bytes = self.serialize_entry(&entry)?;
        let checksum = crc32fast::hash(&bytes);

        // Write length-prefixed entry with checksum
        self.current_segment.write_u32(bytes.len() as u32)?;
        self.current_segment.write_u32(checksum)?;
        self.current_segment.write_all(&bytes)?;

        // Sync based on mode
        match self.config.sync_mode {
            SyncMode::EveryWrite => self.current_segment.sync()?,
            SyncMode::EveryN(n) if self.current_segment.entry_count % n == 0 => {
                self.current_segment.sync()?
            }
            _ => {}
        }

        // Rotate segment if needed
        if self.current_segment.size >= self.config.segment_size {
            self.rotate_segment()?;
        }

        Ok(lsn)
    }

    /// Checkpoint: flush main file and truncate WAL
    pub fn checkpoint(&mut self, database: &Database) -> Result<CheckpointStats> {
        // 1. Write complete database state to main file
        database.save()?;

        // 2. Write checkpoint marker
        let checkpoint_lsn = self.next_lsn.load(Ordering::SeqCst);
        self.append(WalOperation::Checkpoint { lsn: checkpoint_lsn })?;
        self.current_segment.sync()?;

        // 3. Remove old segments
        let removed = self.truncate_before(checkpoint_lsn)?;

        Ok(CheckpointStats {
            lsn: checkpoint_lsn,
            segments_removed: removed,
        })
    }
}
```

### Recovery Process

```rust
impl WalManager {
    /// Recover database state from WAL
    pub fn recover(&self, database: &mut Database) -> Result<RecoveryStats> {
        let mut stats = RecoveryStats::default();

        // Find the last checkpoint
        let checkpoint_lsn = self.find_last_checkpoint()?;
        stats.checkpoint_lsn = checkpoint_lsn;

        // Replay entries after checkpoint
        for segment in self.segments_after(checkpoint_lsn) {
            for entry in segment.read_entries()? {
                // Verify checksum
                if !self.verify_checksum(&entry) {
                    return Err(NeedleError::CorruptedWal(entry.lsn));
                }

                // Apply operation
                match entry.operation {
                    WalOperation::Insert { collection, id, vector, metadata } => {
                        database.collection(&collection)?.insert(&id, &vector, metadata)?;
                        stats.inserts += 1;
                    }
                    WalOperation::Delete { collection, id } => {
                        database.collection(&collection)?.delete(&id)?;
                        stats.deletes += 1;
                    }
                    // ... other operations
                    _ => {}
                }
            }
        }

        Ok(stats)
    }
}
```

### Integration with Database

```rust
impl Database {
    /// Open database with WAL enabled
    pub fn open_with_wal(path: &Path, wal_config: WalConfig) -> Result<Self> {
        // 1. Load main database file
        let mut db = Self::load(path)?;

        // 2. Initialize WAL manager
        let mut wal = WalManager::new(wal_config)?;

        // 3. Replay any uncommitted WAL entries
        let stats = wal.recover(&mut db)?;
        if stats.entries_replayed > 0 {
            log::info!("Recovered {} entries from WAL", stats.entries_replayed);
        }

        db.wal = Some(wal);
        Ok(db)
    }

    /// Insert with WAL logging
    pub fn insert(&mut self, collection: &str, id: &str, vector: &[f32]) -> Result<()> {
        // Log to WAL first (if enabled)
        if let Some(wal) = &mut self.wal {
            wal.append(WalOperation::Insert {
                collection: collection.to_string(),
                id: id.to_string(),
                vector: vector.to_vec(),
                metadata: None,
            })?;
        }

        // Then apply to in-memory state
        self.collection(collection)?.insert(id, vector, None)
    }
}
```

## Consequences

### Benefits

1. **Durability without full rewrites** — Only append to log, not rewrite main file
2. **Fast recovery** — Replay from last checkpoint, not full rebuild
3. **Crash safety** — Incomplete operations are detected and rolled back
4. **Configurable tradeoffs** — Balance durability vs performance via sync modes

### Tradeoffs

1. **Additional disk space** — WAL files consume space until checkpointed
2. **Complexity** — Recovery logic, segment management, checkpoint coordination
3. **Optional feature** — Not all use cases need durability (ephemeral caches)

### What This Enabled

- **Production deployments** — Safe for applications requiring durability
- **Large batch imports** — Can recover from crashes during long imports
- **Audit logging** — WAL entries provide operation history

### What This Prevented

- **Data loss on crash** — Operations are durable once WAL is synced
- **Corruption propagation** — Checksums detect corrupted entries

## References

- WAL implementation: `src/wal.rs` (2049 lines)
- WAL configuration: `src/wal.rs:50-100`
- Recovery logic: `src/wal.rs:400-500`
- Checkpoint logic: `src/wal.rs:300-400`
