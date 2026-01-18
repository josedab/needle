# ADR-0001: Single-File Storage with Memory Mapping

## Status

Accepted

## Context

Needle is designed as "SQLite for vectors" — an embedded vector database that prioritizes simplicity and ease of deployment. The storage layer needed to address several requirements:

1. **Ease of distribution** — Users should be able to backup, copy, version control, and share databases trivially
2. **Crash safety** — Partial writes or crashes should not corrupt data
3. **Performance** — Large databases should not require loading entirely into memory
4. **Simplicity** — No external dependencies like separate index files or configuration directories
5. **Portability** — Databases should be self-contained and platform-independent

Traditional vector databases often use directory-based storage with separate files for indices, vectors, and metadata. While this offers flexibility, it complicates backup procedures and increases the risk of inconsistent states.

## Decision

Needle uses a **single-file storage format** with the following characteristics:

### File Format Structure

```
┌─────────────────────────────────────┐
│ Magic Number: "NEEDLE01" (8 bytes)  │
├─────────────────────────────────────┤
│ Version: u32                        │
├─────────────────────────────────────┤
│ Header (dimensions, counts, offsets)│
├─────────────────────────────────────┤
│ CRC32 Checksum                      │
├─────────────────────────────────────┤
│ Vector Data                         │
├─────────────────────────────────────┤
│ Index Data (HNSW graph)             │
├─────────────────────────────────────┤
│ Metadata (JSON)                     │
└─────────────────────────────────────┘
```

### Key Implementation Details

1. **Magic number validation** (`NEEDLE01`) enables file type detection and prevents loading incompatible files

2. **CRC32 checksums** in the header detect corruption early, before any data is processed

3. **Memory mapping** via `memmap2` crate for files exceeding `MMAP_THRESHOLD` (10MB):
   - Files ≤10MB: Loaded entirely into memory for simplicity
   - Files >10MB: Memory-mapped for efficient random access without full file reads

4. **Atomic saves** using write-to-temp-then-rename pattern:
   ```rust
   // Write to temporary file
   write_to_file(&temp_path)?;
   // Atomic rename (POSIX guarantees)
   std::fs::rename(&temp_path, &final_path)?;
   ```

5. **Version field** enables future format migrations while maintaining backward compatibility

### Code References

- `src/storage.rs:13-22` — Magic number, version, and MMAP_THRESHOLD constants
- `src/storage.rs:24-61` — Header structure with offsets and checksums
- `src/database.rs:587` — Atomic save mechanism

## Consequences

### Benefits

1. **Trivial backup and distribution** — `cp database.needle backup.needle` creates a perfect copy
2. **Version control friendly** — Single file can be committed to Git LFS or stored in artifact repositories
3. **No partial corruption** — Atomic writes ensure either the old or new state exists, never a mix
4. **Efficient large file handling** — Memory mapping avoids loading multi-GB databases entirely
5. **Self-describing format** — Magic number and version enable tooling to identify and validate files
6. **Cloud storage compatible** — Works naturally with S3, GCS, and other object stores

### Tradeoffs

1. **No concurrent writers** — Single-file format prevents multiple processes from writing simultaneously (read-only sharing is safe with mmap)
2. **Full rewrite on save** — Each save rewrites the entire file; not suitable for extremely frequent small updates
3. **Memory mapping limitations** — 32-bit systems have address space constraints for very large files
4. **No incremental backup** — Changes require copying the entire file (mitigated by compression in transport)

### What This Enabled

- Simple CLI operations: `needle info mydb.needle` works on any `.needle` file
- Easy testing: `Database::in_memory()` uses the same code paths without file I/O
- Predictable deployment: No "database directory" configuration or cleanup needed
- Straightforward replication: Raft log can snapshot by copying the single file

### What This Prevented

- Write-ahead logging (WAL) for fine-grained durability
- Concurrent multi-process writes without external coordination
- Incremental/differential backup strategies
- Streaming very large datasets that exceed available disk space during save
