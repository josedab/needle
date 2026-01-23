# ADR-0013: Cloud Storage Abstraction Layer

## Status

Accepted

## Context

While Needle's single-file storage (ADR-0001) provides excellent simplicity for local deployments, enterprise and cloud-native applications require seamless integration with object storage services like AWS S3, Google Cloud Storage, and Azure Blob Storage.

Key requirements driving this decision:

1. **Cloud-native deployments** — Modern applications run on Kubernetes, serverless, or managed infrastructure where local storage is ephemeral
2. **Cost optimization** — Hot/cold data tiering reduces storage costs by 40-60% for large datasets
3. **Global distribution** — Multi-region applications need data locality for low latency
4. **Disaster recovery** — Object storage provides built-in replication and durability
5. **Backward compatibility** — Existing local deployments must continue working unchanged

## Decision

Needle implements a **trait-based cloud storage abstraction** with the following architecture:

### Storage Backend Trait

```rust
pub trait StorageBackend: Send + Sync {
    fn read(&self, key: &str) -> impl Future<Output = Result<Vec<u8>>>;
    fn write(&self, key: &str, data: &[u8]) -> impl Future<Output = Result<()>>;
    fn delete(&self, key: &str) -> impl Future<Output = Result<()>>;
    fn list(&self, prefix: &str) -> impl Future<Output = Result<Vec<String>>>;
    fn exists(&self, key: &str) -> impl Future<Output = Result<bool>>;
}
```

### Backend Implementations

| Backend | Use Case | Key Features |
|---------|----------|--------------|
| `S3Storage` | AWS deployments | Multi-part upload, SSE-S3/KMS encryption, IAM roles |
| `GcsStorage` | Google Cloud | Uniform bucket-level access, signed URLs |
| `AzureStorage` | Azure deployments | Block blobs, managed identities |
| `LocalStorage` | Development/testing | File system backed, same interface |

### Three-Tier Caching Strategy

```
┌─────────────────────────────────────────────────┐
│                  L1 Cache                       │
│         (In-memory LRU, ~100MB)                 │
│     Sub-millisecond access, hot vectors         │
├─────────────────────────────────────────────────┤
│                  L2 Cache                       │
│          (Local SSD, ~10GB)                     │
│    Millisecond access, warm data                │
├─────────────────────────────────────────────────┤
│              Cloud Storage                      │
│         (S3/GCS/Azure, unlimited)               │
│   10-100ms access, cold data, durable           │
└─────────────────────────────────────────────────┘
```

### Prefetching and Access Patterns

The system tracks access patterns and implements predictive prefetching:
- **Sequential access** — Prefetch next N vectors when sequential pattern detected
- **Temporal locality** — Keep recently accessed data in L1/L2 caches
- **Query-driven** — HNSW neighbor lists trigger bulk prefetch of connected nodes

### Code References

- `src/cloud_storage.rs:102` — `StorageBackend` trait definition
- `src/cloud_storage.rs:130-350` — S3, GCS, Azure backend implementations
- `src/cloud_storage.rs:400-550` — Caching layer with LRU eviction

## Consequences

### Benefits

1. **Zero code changes for users** — Existing code works; cloud backends are opt-in via configuration
2. **Unified API** — Same operations work across all storage backends
3. **Cost efficiency** — Automatic tiering based on access patterns saves 40-60% on storage
4. **Resilience** — Cloud storage provides 11 9's durability vs. local disk
5. **Global scale** — Multi-region deployments with data locality
6. **Testability** — LocalStorage backend enables testing without cloud credentials

### Tradeoffs

1. **Latency variance** — Cloud operations have higher and more variable latency than local disk
2. **Eventual consistency** — Some cloud backends (S3) have eventual consistency for overwrites
3. **Network dependency** — Requires internet connectivity for cloud backends
4. **Cost for high-throughput** — Per-request pricing can be expensive for very high QPS workloads
5. **Complexity** — Caching introduces cache invalidation challenges

### What This Enabled

- Kubernetes-native deployments with persistent volumes or object storage
- Serverless vector search (Lambda, Cloud Functions) with S3/GCS backing
- Cost-optimized archival of infrequently accessed vector data
- Multi-region active-active deployments
- Integration with cloud-native backup and disaster recovery

### What This Prevented

- True zero-dependency embedded mode (cloud SDKs add binary size)
- Sub-millisecond guarantees for cache misses
- Completely offline operation for cloud-configured instances

## Migration Path

Existing deployments can migrate incrementally:

1. Deploy with local storage (unchanged behavior)
2. Configure cloud backend with local cache
3. Background job migrates data to cloud
4. Switch primary to cloud with local as L2 cache

The trait-based design ensures these steps require only configuration changes.
