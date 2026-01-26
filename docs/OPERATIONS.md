# Needle Production Operations Guide

This guide covers operational best practices for running Needle in production environments, including monitoring, backup/restore procedures, troubleshooting, and performance tuning.

## Table of Contents

- [Deployment Architecture](#deployment-architecture)
- [Monitoring and Observability](#monitoring-and-observability)
- [Backup and Recovery](#backup-and-recovery)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Security Best Practices](#security-best-practices)
- [Capacity Planning](#capacity-planning)

---

## Deployment Architecture

### Single-Node Deployment

For small to medium workloads (up to ~10M vectors), a single Needle instance is sufficient:

```
┌─────────────────────────────────────────────────────────────┐
│                     Application                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Needle (HTTP API)                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Collection  │  │ Collection  │  │ Collection  │  ...    │
│  │   "docs"    │  │  "images"   │  │  "products" │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Storage (Single .needle file)                   │
└─────────────────────────────────────────────────────────────┘
```

### High-Availability Deployment (Raft Cluster)

For production workloads requiring fault tolerance:

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer                            │
└────────┬────────────────┬────────────────┬──────────────────┘
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Node 1  │◄────►│ Node 2  │◄────►│ Node 3  │
    │(Leader) │      │(Follower)│      │(Follower)│
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Storage │      │ Storage │      │ Storage │
    └─────────┘      └─────────┘      └─────────┘
```

**Key considerations:**
- Minimum 3 nodes for fault tolerance (tolerates 1 failure)
- 5 nodes tolerate 2 failures
- Use an odd number of nodes to avoid split-brain scenarios
- Place nodes across availability zones

### Recommended Hardware

| Workload Size | Vectors | RAM | CPU | Storage |
|---------------|---------|-----|-----|---------|
| Small | < 1M | 8 GB | 4 cores | 50 GB SSD |
| Medium | 1-10M | 32 GB | 8 cores | 200 GB NVMe |
| Large | 10-100M | 128 GB | 32 cores | 1 TB NVMe |
| XL | > 100M | 256+ GB | 64 cores | 2+ TB NVMe |

---

## Monitoring and Observability

### Prometheus Metrics

Enable metrics with the `metrics` feature:

```bash
cargo run --features "server,metrics" -- serve -a 0.0.0.0:8080 -d vectors.needle
```

Access metrics at `http://localhost:8080/metrics`.

#### Key Metrics to Monitor

**Search Performance:**
```promql
# Search latency (p99)
histogram_quantile(0.99, rate(needle_search_duration_seconds_bucket[5m]))

# Search throughput
rate(needle_search_total[5m])

# Search errors
rate(needle_search_errors_total[5m])
```

**Index Health:**
```promql
# Vector count per collection
needle_collection_vector_count{collection="docs"}

# Index memory usage
needle_index_memory_bytes

# Distance calculations per search
rate(needle_distance_calculations_total[5m])
```

**System Resources:**
```promql
# Memory-mapped file size
needle_mmap_size_bytes

# WAL size (if enabled)
needle_wal_size_bytes

# Active connections (server mode)
needle_active_connections
```

### Grafana Dashboard

Sample dashboard configuration:

```json
{
  "title": "Needle Vector Database",
  "panels": [
    {
      "title": "Search Latency (p50, p99)",
      "targets": [
        {"expr": "histogram_quantile(0.50, rate(needle_search_duration_seconds_bucket[5m]))"},
        {"expr": "histogram_quantile(0.99, rate(needle_search_duration_seconds_bucket[5m]))"}
      ]
    },
    {
      "title": "Vector Count by Collection",
      "targets": [
        {"expr": "needle_collection_vector_count"}
      ]
    },
    {
      "title": "Error Rate",
      "targets": [
        {"expr": "rate(needle_errors_total[5m])"}
      ]
    }
  ]
}
```

### Health Checks

```bash
# HTTP health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy", "version": "0.1.0"}
```

For Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

### Alerting Rules

Recommended Prometheus alerting rules:

```yaml
groups:
  - name: needle
    rules:
      - alert: NeedleHighSearchLatency
        expr: histogram_quantile(0.99, rate(needle_search_duration_seconds_bucket[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High search latency detected"
          description: "p99 search latency is above 100ms"

      - alert: NeedleHighErrorRate
        expr: rate(needle_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: NeedleWALSizeHigh
        expr: needle_wal_size_bytes > 1073741824
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "WAL size exceeds 1GB, consider checkpointing"
```

---

## Backup and Recovery

### Backup Strategies

#### Cold Backup (Offline)

Safest method - requires stopping the server:

```bash
# Stop the server
systemctl stop needle

# Copy the database file
cp /var/lib/needle/vectors.needle /backup/vectors.needle.$(date +%Y%m%d)

# Restart
systemctl start needle
```

#### Hot Backup (Online)

Using the backup module for consistent online backups:

```rust
use needle::backup::{BackupManager, BackupConfig, CompressionType};

let config = BackupConfig {
    compression: CompressionType::Zstd,
    include_wal: true,
    verify_after_backup: true,
};

let manager = BackupManager::new(config);
manager.create_backup(&db, "/backup/vectors.backup")?;
```

Via CLI:

```bash
# Create compressed backup
needle backup create vectors.needle -o /backup/daily.backup --compress zstd

# Verify backup integrity
needle backup verify /backup/daily.backup

# List backups
needle backup list /backup/
```

#### Incremental Backups

For large databases, use WAL-based incremental backups:

```bash
# Full backup (weekly)
needle backup create vectors.needle -o /backup/full.backup --full

# Incremental backup (daily) - backs up WAL since last checkpoint
needle backup create vectors.needle -o /backup/incr.backup --incremental
```

### Recovery Procedures

#### Full Restore

```bash
# Stop the server
systemctl stop needle

# Restore from backup
needle backup restore /backup/vectors.backup -o /var/lib/needle/vectors.needle

# Verify integrity
needle info /var/lib/needle/vectors.needle

# Start server
systemctl start needle
```

#### Point-in-Time Recovery (with WAL)

```bash
# Restore to a specific LSN
needle backup restore /backup/base.backup \
    --wal-dir /backup/wal/ \
    --target-lsn 12345678 \
    -o /var/lib/needle/vectors.needle
```

#### Disaster Recovery Checklist

1. **Identify the failure** - Corruption? Hardware? Data loss?
2. **Assess data loss window** - Last backup time vs. failure time
3. **Select recovery point** - Full backup + WAL replay if available
4. **Restore to staging** - Never restore directly to production
5. **Verify integrity** - Check vector counts, run test queries
6. **Promote to production** - Switch traffic to restored instance

### Backup Automation

Sample cron configuration:

```bash
# Daily incremental at 2 AM
0 2 * * * /usr/local/bin/needle backup create /var/lib/needle/vectors.needle -o /backup/daily/$(date +\%Y\%m\%d).backup --incremental

# Weekly full backup at 3 AM Sunday
0 3 * * 0 /usr/local/bin/needle backup create /var/lib/needle/vectors.needle -o /backup/weekly/$(date +\%Y\%m\%d).backup --full --compress zstd

# Monthly backup retention cleanup (keep 30 days)
0 4 1 * * find /backup/daily -mtime +30 -delete
```

---

## Performance Tuning

### HNSW Index Parameters

The HNSW index is the primary factor affecting search performance:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` | 16 | Connections per layer. Higher = better recall, more memory |
| `ef_construction` | 200 | Build-time search depth. Higher = better index quality |
| `ef_search` | 50 | Query-time search depth. Higher = better recall, slower |

**Guidelines:**

```rust
use needle::tuning::{auto_tune, TuningConstraints, PerformanceProfile};

// For high-recall workloads (e.g., legal document search)
let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::HighRecall);
let config = auto_tune(&constraints);
// Typical result: M=24, ef_construction=400, ef_search=100

// For low-latency workloads (e.g., real-time recommendations)
let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::LowLatency);
let config = auto_tune(&constraints);
// Typical result: M=12, ef_construction=100, ef_search=30

// For memory-constrained environments
let constraints = TuningConstraints::new(1_000_000, 384)
    .with_memory_budget(500 * 1024 * 1024);  // 500 MB
let config = auto_tune(&constraints);
```

### Memory Optimization

#### Quantization

Reduce memory usage by 4-8x with minimal recall loss:

```rust
use needle::quantization::QuantizationConfig;

// Scalar quantization (4x compression, ~2% recall loss)
let config = QuantizationConfig::scalar();

// Product quantization (8x compression, ~5% recall loss)
let config = QuantizationConfig::product(8);  // 8 subvectors

// Binary quantization (32x compression, ~10% recall loss)
let config = QuantizationConfig::binary();
```

#### Memory-Mapped Files

Files over 10MB are automatically memory-mapped. For explicit control:

```bash
# Disable mmap (useful for NFS)
export NEEDLE_DISABLE_MMAP=1
```

### Query Optimization

#### Batch Queries

Process multiple queries in parallel:

```rust
use rayon::prelude::*;

let queries: Vec<Vec<f32>> = /* multiple query vectors */;
let results: Vec<_> = queries
    .par_iter()
    .map(|q| collection.search(q, 10))
    .collect();
```

#### Filtered Search

Apply metadata filters to reduce search space:

```rust
use needle::metadata::Filter;

// Filter reduces candidates before distance calculation
let filter = Filter::parse(&json!({
    "category": "electronics",
    "price": {"$lt": 1000}
}))?;

let results = collection.search_with_filter(&query, 10, Some(&filter))?;
```

#### Explain Query Performance

```rust
let (results, explain) = collection.search_explain(&query, 10)?;

println!("Candidates considered: {}", explain.candidates_considered);
println!("Distance calculations: {}", explain.distance_calculations);
println!("Layers traversed: {}", explain.layers_traversed);
println!("Total time: {:?}", explain.total_time);
```

### Compaction

After many deletions, compact to reclaim space:

```bash
# Via CLI
needle compact vectors.needle --collection docs

# Via API
POST /collections/docs/compact
```

**When to compact:**
- Deleted vectors exceed 20% of total
- File size is 2x larger than expected
- Search latency has degraded

---

## Troubleshooting

### Common Issues

#### "Collection not found"

```
Error: CollectionNotFound("my_collection")
```

**Causes:**
- Typo in collection name
- Collection was dropped
- Using wrong database file

**Solution:**
```bash
# List available collections
needle collections vectors.needle
```

#### "Dimension mismatch"

```
Error: DimensionMismatch { expected: 384, got: 768 }
```

**Causes:**
- Using different embedding models for insert vs. query
- Model configuration changed

**Solution:**
```bash
# Check collection dimensions
needle stats vectors.needle --collection docs
# Output: dimensions: 384

# Ensure query vectors match
echo "Query vector has $(echo $QUERY | jq length) dimensions"
```

#### High Search Latency

**Diagnostic steps:**

1. Check `ef_search` parameter:
```rust
let results = collection.search_with_params(&query, 10,
    SearchParams { ef_search: 50 })?;  // Default is 50
```

2. Use search explain:
```rust
let (_, explain) = collection.search_explain(&query, 10)?;
println!("Distance calcs: {}", explain.distance_calculations);
```

3. Check filter selectivity:
```bash
# A filter that matches 90% of vectors provides little benefit
needle stats vectors.needle --filter '{"category": "common"}'
```

4. Consider index rebuild if data distribution changed significantly

#### Memory Issues

**Symptoms:**
- OOM kills
- Slow performance
- Swap thrashing

**Solutions:**

1. Enable quantization:
```rust
let config = CollectionConfig::default()
    .with_quantization(QuantizationConfig::scalar());
```

2. Reduce `M` parameter:
```rust
let config = CollectionConfig::default()
    .with_m(12);  // Default is 16
```

3. Use DiskANN for very large collections:
```rust
let config = CollectionConfig::default()
    .with_index_type(IndexType::DiskANN);
```

#### Corruption Recovery

**Symptoms:**
- `CorruptedDatabase` errors
- Checksum mismatches
- Incomplete reads

**Recovery steps:**

1. **Verify corruption:**
```bash
needle verify vectors.needle
# Output: Checksum mismatch at offset 0x1234
```

2. **Attempt automatic repair:**
```bash
needle repair vectors.needle --output vectors.repaired.needle
```

3. **If repair fails, restore from backup:**
```bash
needle backup restore /backup/latest.backup -o vectors.needle
```

4. **For WAL corruption:**
```bash
# Skip corrupted entries
needle wal recover /path/to/wal --skip-corrupted
```

### Debug Logging

Enable detailed logging:

```bash
# Set log level
export RUST_LOG=needle=debug

# Or more specific
export RUST_LOG=needle::hnsw=trace,needle::storage=debug

# Run with logging
needle serve -d vectors.needle 2>&1 | tee needle.log
```

### Performance Profiling

```rust
use needle::profiler::{QueryProfiler, ProfileReport};

let profiler = QueryProfiler::new();
let report = profiler.profile(|| {
    collection.search(&query, 10)
})?;

println!("{}", report);
// Output:
// Phase           Time      %
// ─────────────────────────────
// Quantize query  0.1ms    2%
// Graph traverse  3.2ms   64%
// Distance calc   1.5ms   30%
// Sort results    0.2ms    4%
// ─────────────────────────────
// Total           5.0ms  100%
```

---

## Security Best Practices

### Encryption at Rest

```rust
use needle::encryption::{EncryptionConfig, EncryptionKey};

let key = EncryptionKey::generate();
let config = EncryptionConfig::aes256_gcm(key);

let db = Database::open_encrypted("vectors.needle", config)?;
```

**Key management:**
- Store keys in a secrets manager (Vault, AWS KMS)
- Rotate keys periodically
- Never commit keys to version control

### Network Security

1. **TLS termination** - Use a reverse proxy (nginx, envoy) for HTTPS
2. **Authentication** - Enable API key authentication:
```bash
needle serve -d vectors.needle --api-key-file /etc/needle/keys.json
```

3. **Network isolation** - Bind to internal interfaces only:
```bash
needle serve -d vectors.needle -a 10.0.0.1:8080
```

### Multi-Tenancy Isolation

```rust
use needle::namespace::{NamespaceManager, TenantQuotas};

let manager = NamespaceManager::new();

// Create isolated namespace with quotas
manager.create_namespace("tenant_a", NamespaceConfig {
    quotas: TenantQuotas {
        max_vectors: Some(1_000_000),
        max_storage_bytes: Some(10 * 1024 * 1024 * 1024),
        rate_limit_ops: Some(100),
        ..Default::default()
    },
    ..Default::default()
})?;
```

---

## Capacity Planning

### Memory Estimation

```
Memory (bytes) ≈ vectors × (dimensions × 4 + M × 8 + overhead)

Where:
- dimensions × 4: Vector storage (f32)
- M × 8: HNSW graph edges
- overhead ≈ 100 bytes per vector (IDs, metadata)
```

**Example for 1M vectors, 384 dimensions, M=16:**
```
1,000,000 × (384 × 4 + 16 × 8 + 100) = ~1.7 GB
```

**With scalar quantization:**
```
1,000,000 × (384 × 1 + 16 × 8 + 100) = ~612 MB (4x reduction)
```

### Storage Estimation

```
Disk (bytes) ≈ Memory + WAL buffer + compaction headroom

Typical: Disk ≈ Memory × 1.5
```

### Throughput Planning

| Operation | Single Core | 8 Cores (parallel) |
|-----------|-------------|-------------------|
| Search (1M vectors) | ~1,000 QPS | ~6,000 QPS |
| Insert | ~10,000 ops/s | ~50,000 ops/s |
| Batch insert (1000) | ~5,000 vectors/s | ~30,000 vectors/s |

**Note:** Actual performance varies based on vector dimensions, `ef_search`, hardware, etc.

### Scaling Guidelines

1. **Vertical scaling first** - Easier to manage single node
2. **At 10M+ vectors** - Consider sharding or DiskANN
3. **At 100M+ vectors** - Distributed deployment with Raft
4. **For multi-region** - Federated search across instances

---

## Operational Runbooks

### Runbook: Database Migration

```bash
# 1. Create backup
needle backup create old.needle -o migration.backup --verify

# 2. Create new database with updated config
needle create new.needle
needle create-collection new.needle -n docs -d 384 --m 24

# 3. Export/import data
needle export old.needle --collection docs | needle import new.needle --collection docs

# 4. Verify
needle stats new.needle --collection docs
needle verify new.needle

# 5. Switch traffic (update configs/symlinks)
ln -sf new.needle vectors.needle
systemctl reload needle
```

### Runbook: Emergency Recovery

```bash
# 1. Stop all writes
curl -X POST http://localhost:8080/admin/readonly

# 2. Assess damage
needle verify vectors.needle
needle wal status /var/lib/needle/wal/

# 3. Attempt repair
needle repair vectors.needle -o vectors.repaired.needle

# 4. If repair fails, restore from backup
needle backup restore /backup/latest.backup -o vectors.needle
needle wal recover /backup/wal/ --replay-to vectors.needle

# 5. Verify recovery
needle verify vectors.needle
needle stats vectors.needle

# 6. Resume writes
curl -X POST http://localhost:8080/admin/readwrite
```

### Runbook: Performance Degradation

```bash
# 1. Check metrics
curl http://localhost:8080/metrics | grep needle_search_duration

# 2. Check for compaction need
needle stats vectors.needle
# Look for: deleted_vectors / total_vectors > 0.2

# 3. Run compaction if needed
needle compact vectors.needle

# 4. Check index health
needle info vectors.needle
# Look for: index_memory, graph_connectivity

# 5. Consider retuning
needle tune vectors.needle --profile balanced --apply
```

---

## Appendix: Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEEDLE_LOG_LEVEL` | Log verbosity | `info` |
| `NEEDLE_DISABLE_MMAP` | Disable memory mapping | `false` |
| `NEEDLE_WAL_DIR` | WAL directory | `<db_path>.wal/` |
| `NEEDLE_CHECKPOINT_INTERVAL` | WAL checkpoint frequency | `10000` |
| `NEEDLE_METRICS_PORT` | Metrics endpoint port | `9090` |

### CLI Reference

```bash
# Database operations
needle create <path>                 # Create new database
needle info <path>                   # Show database info
needle verify <path>                 # Verify integrity
needle compact <path>                # Compact deleted vectors
needle repair <path>                 # Attempt corruption repair

# Collection operations
needle create-collection <path> -n <name> -d <dims>
needle collections <path>            # List collections
needle stats <path> -c <collection>  # Show statistics

# Server operations
needle serve -d <path> -a <addr>     # Start HTTP server
needle serve --config <file>         # Start with config file

# Backup operations
needle backup create <path> -o <output>
needle backup restore <backup> -o <path>
needle backup verify <backup>

# Tuning
needle tune <path> --profile <profile>
needle tune <path> --memory-budget <bytes>
```

---

## See Also

- [Production Checklist](production-checklist.md) - Pre-deployment verification checklist
- [Distributed Operations](distributed-operations.md) - Sharding, Raft replication, and clustering
- [Deployment Guide](deployment.md) - Docker, Kubernetes, and cloud deployment
- [Migration Guide](migration-upgrade-guide.md) - Version upgrades and database migrations
- [How-To Guides](how-to-guides.md) - Practical tutorials for common tasks
