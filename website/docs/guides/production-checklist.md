---
sidebar_position: 8
---

# Production Checklist

A comprehensive checklist for deploying Needle to production. Complete these items before going live.

---

## Capacity Planning

- **Estimate dataset size**: Calculate expected vector count at launch and 12-month projection
- **Calculate memory requirements**: Use formula `N × (D × 4 + M × 8 × 1.5)` bytes
- **Select index type**: HNSW (&lt;50M), IVF (50M–500M), or DiskANN (&gt;500M). See [Index Selection Guide](/docs/guides/index-selection)
- **Size disk storage**: Allow 2x index size for compaction headroom
- **Plan for growth**: Design for 2–3x current capacity

```rust
use needle::{auto_tune, TuningConstraints};

let constraints = TuningConstraints::new(
    10_000_000,  // expected vectors
    384,         // dimensions
)
.with_memory_budget(8 * 1024 * 1024 * 1024); // 8 GB

let estimate = auto_tune(&constraints);
println!("Expected memory: {} GB", estimate.expected_memory_mb / 1024);
```

---

## Index Configuration

- **Tune HNSW parameters**: Run benchmarks with your actual data distribution
- **Set appropriate ef_search**: Balance recall vs latency
- **Enable SIMD**: Build with `--features simd` for 2–4x faster distance calculations
- **Configure batch size**: Optimize for your query patterns

| Profile | M | ef_construction | ef_search | Use Case |
|---------|---|-----------------|-----------|----------|
| High Recall | 32 | 400 | 200 | Compliance, medical |
| Balanced | 16 | 200 | 50 | General purpose |
| Low Latency | 8 | 100 | 20 | Real-time apps |

---

## Data Quality

- **Validate vector dimensions**: All vectors must match collection dimension
- **Normalize vectors**: For cosine distance, pre-normalize for best results
- **Handle missing data**: Decide on strategy for incomplete records
- **Test with production data**: Use a representative sample for tuning

---

## Security

### Encryption

- Enable encryption at rest (ChaCha20-Poly1305) with the `encryption` feature flag
- Use a secrets manager for key storage (Vault, AWS Secrets Manager)
- Implement key rotation procedures
- Ensure backup files are also encrypted

### Access Control

- Define roles with least privilege (reader, writer, admin)
- Enable RBAC for multi-user environments
- Set up audit logging for compliance

### Network Security

- Enable TLS for the REST API (via reverse proxy or built-in)
- Configure CORS to restrict origins
- Keep the database off the public internet — use private networking

---

## Durability & Recovery

### Backups

- **Schedule automated backups**: Regular full backups of the `.needle` file
- **Test restore procedures**: Regularly verify backup integrity
- **Store backups off-site**: Use a different region or provider

```bash
# Simple backup — single file copy
db.save()?;
cp vectors.needle /backup/vectors.needle.$(date +%Y%m%d)
```

### Disaster Recovery

- Document RTO/RPO targets
- Create step-by-step recovery runbooks
- Test failover scenarios regularly

---

## Monitoring & Observability

### Metrics

Enable Prometheus metrics with `--features metrics`:

```bash
cargo run --features "server,metrics" -- serve -a 0.0.0.0:8080 -d vectors.needle
```

Access metrics at `http://localhost:8080/metrics`.

### Key Metrics

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|---------------------|
| Search latency p99 | &gt; 100ms | &gt; 500ms |
| Error rate | &gt; 0.1% | &gt; 1% |
| Memory usage | &gt; 80% | &gt; 95% |
| Disk usage | &gt; 70% | &gt; 90% |

### Health Checks

```yaml
# Kubernetes
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
readinessProbe:
  httpGet:
    path: /health
    port: 8080
```

---

## Performance

### Query Optimization

- Use `batch_search()` for multiple concurrent queries
- Use `search_explain()` to profile slow queries
- Apply metadata filters to reduce the candidate set
- Consider quantization for memory-bound workloads

```rust
let (results, explain) = collection.search_explain(&query, 10, None)?;
println!("Total time: {:?}", explain.total_time);
println!("Nodes visited: {}", explain.nodes_visited);
```

### Compaction

- Schedule regular compaction to reclaim space from deletions
- Run during low-traffic windows
- Compact when deleted vectors exceed 20% of total

---

## Deployment

### Container Resources

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Rollout Strategy

- Use blue-green deployment for zero-downtime updates (see [Collection Aliasing](/docs/concepts/aliasing))
- Configure canary releases for safety
- Document rollback procedures

---

## Pre-Launch Verification

- [ ] Integration tests pass with production-like data
- [ ] Search accuracy verified against ground truth
- [ ] Load tested at 2–3x expected traffic
- [ ] Monitoring dashboards and alerts configured
- [ ] Backup and restore procedure tested
- [ ] Runbook documented for on-call team

---

## See Also

- [Operations Guide](/docs/advanced/operations) — Day-to-day operations
- [Deployment Guide](/docs/advanced/deployment) — Docker, Kubernetes, Helm
- [Index Selection Guide](/docs/guides/index-selection) — Choosing the right index
