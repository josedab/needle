# Production Checklist

A comprehensive checklist for deploying Needle to production. Complete these items before going live.

## Pre-Deployment

### Capacity Planning

- [ ] **Estimate dataset size**: Calculate expected vector count at launch and 12-month projection
- [ ] **Calculate memory requirements**: Use formula `N × (D × 4 + M × 8 × 1.5)` bytes
- [ ] **Select index type**: HNSW (<50M), IVF (50M-500M), or DiskANN (>500M)
- [ ] **Size disk storage**: Allow 2x index size for compaction headroom
- [ ] **Plan for growth**: Design for 2-3x current capacity

```rust
// Capacity estimation helper
use needle::{auto_tune, TuningConstraints};

let constraints = TuningConstraints::new(
    expected_vectors,  // e.g., 10_000_000
    dimensions,        // e.g., 384
)
.with_memory_budget(available_ram);

let estimate = auto_tune(&constraints);
println!("Expected memory: {} GB", estimate.expected_memory_mb / 1024);
```

### Index Configuration

- [ ] **Tune HNSW parameters**: Run benchmarks with your data
- [ ] **Set appropriate ef_search**: Balance recall vs latency
- [ ] **Enable SIMD**: Build with `--features simd` for 2-4x faster distance calculations
- [ ] **Configure batch size**: Optimize for your query patterns

| Profile | M | ef_construction | ef_search | Use Case |
|---------|---|-----------------|-----------|----------|
| High Recall | 32 | 400 | 200 | Compliance, medical |
| Balanced | 16 | 200 | 50 | General purpose |
| Low Latency | 8 | 100 | 20 | Real-time apps |

### Data Quality

- [ ] **Validate vector dimensions**: All vectors must match collection dimension
- [ ] **Normalize vectors**: For cosine distance, consider pre-normalizing
- [ ] **Handle missing data**: Decide on strategy for incomplete records
- [ ] **Test with production data**: Use representative sample for tuning

---

## Security

### Encryption

- [ ] **Enable encryption at rest**: Configure ChaCha20-Poly1305
- [ ] **Secure key management**: Use a secrets manager (Vault, AWS Secrets Manager)
- [ ] **Rotate encryption keys**: Implement key rotation procedure
- [ ] **Encrypt backups**: Ensure backup files are also encrypted

```rust
use needle::{DatabaseConfig, EncryptionConfig};

let encryption = EncryptionConfig::new()
    .with_key_derivation("your-secret-key")
    .with_algorithm("chacha20-poly1305");

let config = DatabaseConfig::new("production.needle")
    .with_encryption(encryption);
```

### Access Control

- [ ] **Define roles**: Create appropriate roles (reader, writer, admin)
- [ ] **Apply least privilege**: Grant minimum necessary permissions
- [ ] **Enable RBAC**: Configure role-based access control
- [ ] **Set up audit logging**: Log all access for compliance

```rust
use needle::security::{AccessController, Role, Permission};

let mut ac = AccessController::new();

ac.create_role("reader", vec![Permission::Read])?;
ac.create_role("writer", vec![Permission::Read, Permission::Write])?;
ac.create_role("admin", vec![Permission::Admin])?;

ac.grant_role("api-service", "writer")?;
ac.grant_role("analytics", "reader")?;
```

### Network Security

- [ ] **Enable TLS**: Configure HTTPS for REST API
- [ ] **Set up authentication**: API keys, JWT, or mTLS
- [ ] **Configure CORS**: Restrict origins for web clients
- [ ] **Use private networking**: Keep database off public internet

---

## Durability & Recovery

### Write-Ahead Log (WAL)

- [ ] **Enable WAL**: Critical for crash recovery
- [ ] **Configure checkpoint interval**: Balance durability vs performance
- [ ] **Set WAL size limits**: Prevent unbounded growth
- [ ] **Test recovery procedure**: Verify WAL replay works correctly

```rust
use needle::{DatabaseConfig, WalConfig};

let wal = WalConfig::new()
    .with_checkpoint_interval(Duration::from_secs(300))
    .with_max_wal_size(1024 * 1024 * 1024)  // 1GB
    .with_sync_mode(SyncMode::Fsync);

let config = DatabaseConfig::new("production.needle")
    .with_wal(wal);
```

### Backup Strategy

- [ ] **Implement automated backups**: Schedule regular full backups
- [ ] **Configure retention policy**: Define backup retention period
- [ ] **Test restore procedure**: Regularly verify backup integrity
- [ ] **Store backups off-site**: Use different region/provider

```rust
use needle::backup::{BackupManager, BackupConfig, BackupType};

let config = BackupConfig::new("/backups/needle")
    .with_compression(true)
    .with_encryption(true)
    .with_retention_days(30);

let manager = BackupManager::new(config);

// Schedule daily full backup
manager.create_backup(&db, BackupType::Full)?;
```

### Disaster Recovery

- [ ] **Document RTO/RPO**: Define recovery time and point objectives
- [ ] **Create runbook**: Step-by-step recovery procedure
- [ ] **Test failover**: Regularly drill recovery scenarios
- [ ] **Set up replication**: For HA requirements (Raft consensus)

---

## Monitoring & Observability

### Metrics

- [ ] **Enable Prometheus metrics**: Build with `--features metrics`
- [ ] **Export key metrics**: Query latency, throughput, memory usage
- [ ] **Set up dashboards**: Create Grafana dashboards for visibility
- [ ] **Configure alerts**: Alert on latency spikes, errors, capacity

```rust
// Required metrics to monitor
// - needle_search_latency_seconds (histogram)
// - needle_insert_latency_seconds (histogram)
// - needle_memory_usage_bytes (gauge)
// - needle_vector_count (gauge)
// - needle_query_throughput (counter)
// - needle_error_count (counter)
```

### Alerting Rules

| Metric | Warning | Critical |
|--------|---------|----------|
| Search latency p99 | > 100ms | > 500ms |
| Error rate | > 0.1% | > 1% |
| Memory usage | > 80% | > 95% |
| Disk usage | > 70% | > 90% |
| WAL lag | > 100MB | > 1GB |

### Logging

- [ ] **Configure structured logging**: JSON format for log aggregation
- [ ] **Set appropriate log levels**: INFO for production, DEBUG for troubleshooting
- [ ] **Enable audit logging**: For compliance and forensics
- [ ] **Set up log rotation**: Prevent disk exhaustion

---

## Performance

### Query Optimization

- [ ] **Use batch search**: For multiple queries, use `batch_search()`
- [ ] **Enable query caching**: Cache frequent queries
- [ ] **Optimize filters**: Use pre-filtering for selective conditions
- [ ] **Profile slow queries**: Use `search_explain()` for analysis

```rust
// Profile query performance
let (results, explain) = collection.search_explain(&query, 10)?;

println!("Total time: {}μs", explain.total_time_us);
println!("Index traversal: {}μs", explain.index_time_us);
println!("Filter evaluation: {}μs", explain.filter_time_us);
println!("Nodes visited: {}", explain.hnsw_stats.visited_nodes);
```

### Resource Limits

- [ ] **Set memory limits**: Prevent OOM conditions
- [ ] **Configure connection limits**: Limit concurrent connections
- [ ] **Implement rate limiting**: Protect against query floods
- [ ] **Set query timeouts**: Prevent runaway queries

```rust
use needle::ServerConfig;

let config = ServerConfig::new()
    .with_max_connections(100)
    .with_query_timeout(Duration::from_secs(30))
    .with_max_batch_size(1000);
```

### Compaction

- [ ] **Schedule regular compaction**: Reclaim space from deletes
- [ ] **Monitor fragmentation**: Track deleted vector ratio
- [ ] **Plan compaction windows**: Run during low-traffic periods

---

## High Availability

### Replication (if using distributed mode)

- [ ] **Configure Raft cluster**: Minimum 3 nodes for fault tolerance
- [ ] **Set up leader election**: Verify automatic failover
- [ ] **Configure replication factor**: At least 3 for durability
- [ ] **Test node failure**: Verify cluster survives node loss

### Load Balancing

- [ ] **Deploy load balancer**: Distribute queries across replicas
- [ ] **Configure health checks**: Remove unhealthy nodes
- [ ] **Set up read replicas**: Scale read throughput
- [ ] **Implement circuit breakers**: Prevent cascade failures

---

## Deployment

### Container Configuration

- [ ] **Set resource requests/limits**: CPU, memory for Kubernetes
- [ ] **Configure liveness probe**: Health check endpoint
- [ ] **Configure readiness probe**: Ready-to-serve check
- [ ] **Use persistent volumes**: For data durability

```yaml
# Kubernetes deployment example
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
```

### Rollout Strategy

- [ ] **Use blue-green deployment**: Zero-downtime updates
- [ ] **Configure gradual rollout**: Canary releases for safety
- [ ] **Plan rollback procedure**: Quick revert if issues arise
- [ ] **Document version compatibility**: Breaking changes awareness

---

## Pre-Launch Verification

### Functional Testing

- [ ] **Run integration tests**: Full API test suite
- [ ] **Verify search accuracy**: Compare against ground truth
- [ ] **Test edge cases**: Empty collections, max dimensions, etc.
- [ ] **Validate filters**: Complex filter combinations

### Load Testing

- [ ] **Run load tests**: Simulate expected traffic
- [ ] **Test peak load**: 2-3x normal traffic
- [ ] **Measure latency distribution**: p50, p95, p99
- [ ] **Identify bottlenecks**: CPU, memory, I/O

### Chaos Testing

- [ ] **Test node failures**: Kill random instances
- [ ] **Test network partitions**: Simulate connectivity issues
- [ ] **Test disk full**: Graceful handling
- [ ] **Test OOM conditions**: Memory exhaustion behavior

---

## Post-Launch

### Monitoring

- [ ] **Watch key metrics**: First 24-48 hours closely
- [ ] **Review error logs**: Address any unexpected errors
- [ ] **Track performance**: Compare against benchmarks
- [ ] **Gather user feedback**: Identify issues early

### Documentation

- [ ] **Update runbooks**: Based on launch learnings
- [ ] **Document configurations**: All production settings
- [ ] **Create troubleshooting guide**: Common issues and fixes
- [ ] **Record architecture decisions**: ADRs for future reference

---

## Checklist Summary

| Category | Items | Critical |
|----------|-------|----------|
| Capacity Planning | 5 | Yes |
| Index Configuration | 4 | Yes |
| Security | 10 | Yes |
| Durability | 10 | Yes |
| Monitoring | 10 | Yes |
| Performance | 10 | No |
| High Availability | 8 | Depends |
| Deployment | 8 | Yes |
| Testing | 12 | Yes |

**Minimum viable checklist for launch**: Capacity Planning + Security + Durability + Monitoring + Deployment + Functional Testing

---

## See Also

- [Operations Guide](OPERATIONS.md) - Day-to-day operations
- [Deployment Guide](deployment.md) - Deployment options
- [Index Selection Guide](index-selection-guide.md) - Choosing the right index
- [Architecture](architecture.md) - System internals
