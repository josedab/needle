# ADR-0015: Streaming Change Data Capture Integration

## Status

Accepted

## Context

Enterprise applications store source data in operational databases (PostgreSQL, MongoDB, MySQL). Keeping vector embeddings synchronized with source data is critical for search relevance but challenging:

1. **Data freshness** — Embeddings must reflect recent changes for accurate search
2. **Consistency** — Vector store should not contain stale or deleted records
3. **Scalability** — Batch sync jobs don't scale to high-throughput systems
4. **Operational simplicity** — Manual sync pipelines are error-prone and hard to maintain
5. **Exactly-once semantics** — Duplicates or missed updates corrupt search quality

Traditional approaches using periodic batch jobs introduce latency (minutes to hours) and complexity in tracking what has changed.

## Decision

Needle implements **real-time Change Data Capture (CDC)** with the following architecture:

### CDC Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Source DB      │───▶│  CDC Connector  │───▶│  Needle         │
│  (Postgres/     │    │  (Debezium/     │    │  Vector Store   │
│   MongoDB)      │    │   Native)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Kafka/Pulsar   │
                    │  (Optional)     │
                    └─────────────────┘
```

### Connector Types

| Connector | Source | Mechanism |
|-----------|--------|-----------|
| `PostgresCdcConnector` | PostgreSQL | Logical replication (pgoutput) |
| `MongoCdcConnector` | MongoDB | Change streams |
| `KafkaConnector` | Any (via Kafka) | Consumer groups, Avro/JSON |
| `PulsarConnector` | Any (via Pulsar) | Subscriptions, schema registry |

### CDC Position Tracking

```rust
pub struct CdcPosition {
    /// Unique identifier for this position
    pub id: String,
    /// Source-specific position (LSN for Postgres, resume token for Mongo)
    pub position: String,
    /// Timestamp of this position
    pub timestamp: DateTime<Utc>,
    /// Number of records processed at this position
    pub records_processed: u64,
}
```

### Message Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    CDC Pipeline                               │
├──────────────────────────────────────────────────────────────┤
│  1. Capture     │  Database transaction log / change stream  │
│  2. Transform   │  Extract text, apply schema mapping        │
│  3. Embed       │  Generate vectors (batch or streaming)     │
│  4. Upsert      │  Insert/update vectors with dedup          │
│  5. Checkpoint  │  Persist position for exactly-once         │
└──────────────────────────────────────────────────────────────┘
```

### Exactly-Once Semantics

```rust
// Transactional checkpoint with vector upsert
async fn process_batch(&mut self, events: Vec<CdcEvent>) -> Result<()> {
    // 1. Transform and embed
    let vectors = self.embed_events(&events).await?;

    // 2. Begin transaction
    let txn = self.db.begin_transaction()?;

    // 3. Upsert vectors
    for (id, vector, metadata) in vectors {
        txn.upsert(&self.collection, &id, &vector, metadata)?;
    }

    // 4. Update checkpoint (same transaction)
    txn.set_checkpoint(&self.position)?;

    // 5. Commit atomically
    txn.commit()?;

    Ok(())
}
```

### Configuration

```rust
pub struct CdcConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum wait time before processing partial batch
    pub batch_timeout: Duration,
    /// Dead letter queue for failed records
    pub dlq_enabled: bool,
    /// Number of retries before DLQ
    pub max_retries: u32,
    /// Embedding concurrency
    pub embed_concurrency: usize,
}
```

### Code References

- `src/streaming.rs:1681-1700` — `CdcPosition` for checkpoint tracking
- `src/streaming.rs:1761-1780` — `CdcConnectorStats` for monitoring
- `src/streaming.rs:1782-1850` — `CdcConfig` configuration
- `src/streaming.rs:2553-2790` — `PostgresCdcConnector` implementation
- `src/streaming.rs:2810-3070` — `MongoCdcConnector` implementation
- `src/streaming.rs:3083-3200` — `CdcIngestionPipeline` orchestration

## Consequences

### Benefits

1. **Real-time sync** — Sub-second latency from database change to searchable vector
2. **Exactly-once semantics** — Checkpoints ensure no duplicates or missed records
3. **Operational simplicity** — No custom sync jobs or cron schedules
4. **Scalability** — Handles thousands of changes per second
5. **Schema evolution** — Supports adding/removing fields without pipeline changes
6. **Dead letter queue** — Failed records don't block the pipeline

### Tradeoffs

1. **Infrastructure dependency** — Requires Kafka/Pulsar for buffering in high-throughput scenarios
2. **Embedding cost** — Real-time embedding can be expensive at scale
3. **Complexity** — CDC introduces distributed systems challenges (ordering, exactly-once)
4. **Database load** — Logical replication adds some load to source database
5. **Schema coupling** — Changes to source schema may require pipeline updates

### What This Enabled

- Real-time search over live operational data
- Automatic deletion propagation (no orphaned vectors)
- Multi-source aggregation (combine multiple databases)
- Event-driven architectures with vector search
- Compliance-friendly audit trails via checkpoints

### What This Prevented

- Simple single-binary deployments (CDC needs external components)
- Offline-only operation
- Guaranteed ordering across partitions without additional coordination

### Monitoring and Observability

```rust
pub struct CdcConnectorStats {
    /// Total records processed
    pub records_processed: u64,
    /// Records currently in flight
    pub records_in_flight: u64,
    /// Current lag in milliseconds
    pub lag_ms: u64,
    /// Error count by type
    pub errors: HashMap<String, u64>,
    /// Last checkpoint position
    pub last_checkpoint: Option<CdcPosition>,
}
```

Prometheus metrics are exposed for:
- `needle_cdc_records_total` — Total processed records
- `needle_cdc_lag_seconds` — Replication lag
- `needle_cdc_errors_total` — Error count by type
- `needle_cdc_batch_duration_seconds` — Processing time histogram
