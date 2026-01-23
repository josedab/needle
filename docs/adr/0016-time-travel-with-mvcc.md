# ADR-0016: Time-Travel with MVCC for Vector Search

## Status

Accepted

## Context

Traditional vector databases provide only the current state of the index. However, many applications require historical queries:

1. **Reproducibility** — ML experiments need to reproduce search results from specific points in time
2. **Compliance** — Regulated industries must demonstrate what data was searchable at audit time
3. **Debugging** — "Why did the search return X last Tuesday?" requires historical queries
4. **A/B testing** — Compare search quality across different embedding versions
5. **Rollback** — Recover from bad data ingestion without full restore

Point-in-time queries are standard in relational databases (PostgreSQL, SQL Server) but rare in vector stores, creating a significant gap for enterprise adoption.

## Decision

Needle implements **Multi-Version Concurrency Control (MVCC)** for vectors with the following design:

### Version Model

```rust
/// A vector version in MVCC
pub struct VectorVersion {
    /// The vector data
    pub vector: Vec<f32>,
    /// Metadata at this version
    pub metadata: serde_json::Value,
    /// When this version was created
    pub created_at: DateTime<Utc>,
    /// When this version was superseded (None if current)
    pub deleted_at: Option<DateTime<Utc>>,
    /// Version number (monotonically increasing)
    pub version: u64,
}
```

### Time-Travel Index Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   TimeTravelIndex                           │
├─────────────────────────────────────────────────────────────┤
│  Current Index    │  Standard HNSW for latest vectors       │
├───────────────────┼─────────────────────────────────────────┤
│  Version Store    │  HashMap<VectorId, Vec<VectorVersion>>  │
├───────────────────┼─────────────────────────────────────────┤
│  Temporal Index   │  BTreeMap<DateTime, Vec<VectorId>>      │
│                   │  (For efficient time-range queries)     │
├───────────────────┼─────────────────────────────────────────┤
│  Snapshot Cache   │  LRU<DateTime, HnswSnapshot>            │
│                   │  (Materialized snapshots for hot times) │
└───────────────────┴─────────────────────────────────────────┘
```

### Query Types

```rust
pub struct TimeTravelSearchResult {
    /// The vector ID
    pub id: String,
    /// Distance to query vector
    pub distance: f32,
    /// Metadata at the queried point in time
    pub metadata: serde_json::Value,
    /// The version that was valid at query time
    pub version: u64,
    /// When this version was created
    pub version_created_at: DateTime<Utc>,
}
```

### Time-Travel Query Builder

```rust
let results = index
    .time_travel()
    .as_of(DateTime::parse("2024-01-15T10:30:00Z")?)
    .query(&query_vector)
    .with_filter(json!({"category": "books"}))
    .top_k(10)
    .execute()?;
```

### MVCC Configuration

```rust
pub struct MvccConfig {
    /// How long to retain old versions
    pub retention_period: Duration,
    /// Maximum versions per vector (0 = unlimited)
    pub max_versions_per_vector: usize,
    /// Whether to maintain temporal index
    pub enable_temporal_index: bool,
    /// Snapshot cache size
    pub snapshot_cache_size: usize,
}
```

### Code References

- `src/time_travel.rs:55-90` — `MvccConfig` configuration
- `src/time_travel.rs:91-110` — `VectorVersion` structure
- `src/time_travel.rs:360-400` — `TimeTravelIndex` implementation
- `src/time_travel.rs:402-430` — `TimeTravelSearchResult` structure
- `src/time_travel.rs:1086-1200` — `TimeTravelQueryBuilder` fluent API

## Consequences

### Benefits

1. **Reproducible experiments** — Re-run searches exactly as they were at any point
2. **Compliance ready** — Demonstrate historical state for audits
3. **Safe rollback** — Query "before the bad ingestion" without data restore
4. **Debugging** — Understand how search results evolved over time
5. **A/B comparison** — Query same vectors with different versions simultaneously
6. **Non-blocking reads** — MVCC allows readers and writers to proceed concurrently

### Tradeoffs

1. **Storage overhead** — Each version consumes additional storage
2. **Garbage collection** — Old versions must be periodically cleaned up
3. **Query complexity** — Time-travel queries are slower than current-state queries
4. **Index maintenance** — Temporal index adds write overhead
5. **Memory pressure** — Version store can grow large for frequently updated vectors

### What This Enabled

- ML experiment reproducibility with exact historical search results
- Regulatory compliance for financial and healthcare applications
- "What-if" analysis comparing different embedding models
- Safe data migration with instant rollback capability
- Historical analytics on search pattern evolution

### What This Prevented

- Minimal memory footprint for update-heavy workloads
- Simple storage model (now requires version chains)
- Constant-time deletes (must mark as deleted, not remove)

### Garbage Collection Strategy

```rust
// Background GC process
async fn garbage_collect(&mut self) -> Result<GcStats> {
    let cutoff = Utc::now() - self.config.retention_period;

    let mut removed = 0;
    for (id, versions) in self.version_store.iter_mut() {
        // Keep at least one version (the current one)
        let to_remove: Vec<_> = versions
            .iter()
            .filter(|v| v.deleted_at.map(|d| d < cutoff).unwrap_or(false))
            .filter(|v| v.version != versions.last().map(|l| l.version).unwrap_or(0))
            .collect();

        removed += to_remove.len();
        versions.retain(|v| !to_remove.contains(&v));
    }

    Ok(GcStats { versions_removed: removed })
}
```

### Snapshot Materialization

For frequently queried points in time, the system materializes full HNSW snapshots:

1. **On-demand** — First query at a timestamp builds the snapshot
2. **Cached** — LRU cache retains hot snapshots
3. **Background** — Scheduled jobs can pre-build expected audit timestamps
