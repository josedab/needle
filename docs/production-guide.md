# Production Deployment Guide

A practical guide for deploying Needle in production environments.

## Prerequisites

- **Rust 1.85+** (for building from source) or **Docker**
- SSD-backed storage recommended for database files
- Build with all production features: `cargo build --release --features full`

## TLS / HTTPS

Needle does not handle TLS directly. Use a reverse proxy for TLS termination.

See [deployment.md](deployment.md#tls--https-configuration) for nginx, Caddy, and Kubernetes Ingress examples.

## Authentication

Enable authentication by setting `require_auth: true` in your server config.
Health (`/health`) and root (`/`) endpoints are public by default.

### API Key Authentication

Configure API keys with per-key roles and optional expiration:

```json
{
  "require_auth": true,
  "api_keys": [
    {
      "key": "your-secret-key",
      "name": "backend-service",
      "roles": ["admin"],
      "active": true,
      "expires_at": 1735689600
    }
  ]
}
```

Supported roles: `admin`, `writer`, `reader`. Pass keys via the `Authorization: Bearer <key>` header.

### JWT Authentication

For service-to-service auth, configure JWT signing:

```json
{
  "jwt_secret": "your-secret-min-32-bytes-long-here",
  "jwt_algorithm": "HS256"
}
```

Supported algorithms: **HS256** (symmetric, default), **RS256**, **ES256** (asymmetric).
Use `jwt_previous_secrets` for zero-downtime key rotation.

### OIDC Integration

For enterprise SSO, configure OIDC validation against your identity provider:

```json
{
  "oidc": {
    "issuer_url": "https://auth.example.com",
    "audience": "needle-api"
  }
}
```

## Rate Limiting

Needle provides three-tier, per-IP rate limiting (powered by the `governor` crate):

| Tier | Operations | Default |
|------|-----------|---------|
| **Read** | GET, HEAD | 100 req/s |
| **Write** | POST, PUT, DELETE, PATCH | 20 req/s |
| **Admin** | save, compact, webhooks | 5 req/s |

Responses include `x-ratelimit-limit` and `x-ratelimit-remaining` headers.

To disable rate limiting entirely, set `requests_per_second: 0`.

## Monitoring & Observability

### Prometheus Metrics

Enable with `--features metrics`. Key metrics to monitor:

| Metric | Type | Description |
|--------|------|-------------|
| `needle_http_requests_total` | Counter | Request count by method/status |
| `needle_http_request_duration_seconds` | Histogram | Latency distribution |
| `needle_http_requests_in_flight` | Gauge | Active requests |
| `needle_collection_vectors` | Gauge | Vector count per collection |
| `needle_collection_deleted_vectors` | Gauge | Soft-deleted vectors pending compaction |
| `needle_memory_bytes` | Gauge | Memory usage by collection/component |
| `needle_search_result_count` | Histogram | Search result distribution |
| `needle_errors_total` | Counter | Errors by type and operation |

A sample scrape config is provided at [`deploy/prometheus.yml`](../deploy/prometheus.yml).

### Grafana Dashboards

Import pre-built dashboards from [`deploy/grafana/`](../deploy/grafana/) (requires Grafana 9.0+):

- **`needle-overview.json`** — QPS, latency, errors, in-flight requests, resource counts
- **`needle-collections.json`** — Per-collection vectors, search latency (p50/p95/p99), memory, HNSW stats

See [OPERATIONS.md](OPERATIONS.md#monitoring-and-observability) for alerting guidance.

## Backup Strategy

### Single-File Backups

The single-file storage format makes backups trivial:

```bash
cp mydb.needle mydb.needle.backup
```

### WAL & Point-in-Time Recovery

Enable WAL (Write-Ahead Log) for crash recovery. With WAL enabled, replay operations to recover to a specific point in time.

See [OPERATIONS.md](OPERATIONS.md#backup-and-recovery) for detailed backup procedures and scheduling.

## Performance Tuning

### HNSW Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `M` | 16 | Higher = better recall, more memory |
| `ef_construction` | 200 | Higher = better index quality, slower builds |
| `ef_search` | 50 | Higher = better recall, slower queries |

Use auto-tuning to optimize for your workload:

```bash
needle tune mydb.needle --collection my_collection
```

See [OPERATIONS.md](OPERATIONS.md#performance-tuning) for detailed tuning guidance.

### Quantization

Reduce memory footprint with quantization:

| Method | Compression | Recall Impact |
|--------|-------------|---------------|
| **Scalar (SQ8)** | 4× | Minimal |
| **Product (PQ)** | 16–256× | Moderate |
| **Binary (BQ)** | 32× | Significant — best for pre-filtering |

See [index-selection-guide.md](index-selection-guide.md) for choosing the right strategy.

### Connection Pooling

For high-throughput server mode, configure connection pooling at the reverse proxy or load balancer level. Needle handles concurrent requests internally via `parking_lot::RwLock`.

## Scaling

### Vertical Scaling

- **Memory**: HNSW indices are memory-resident. Size memory to fit your index plus working set.
- **Storage**: Use SSDs for mmap performance (files >10 MB are automatically memory-mapped).
- **CPU**: Search parallelizes via Rayon; more cores improve `batch_search` throughput.

### Sharding

For datasets exceeding single-node memory, use collection-level sharding — distribute collections across multiple Needle instances.

See [distributed-operations.md](distributed-operations.md) for multi-node setups.

## Security Hardening Checklist

- [ ] Enable TLS via reverse proxy ([deployment.md](deployment.md#tls--https-configuration))
- [ ] Configure authentication (API keys, JWT, or OIDC)
- [ ] Set rate limiting appropriate to your traffic
- [ ] Run as a non-root user
- [ ] Configure CORS for web access
- [ ] Enable encryption at rest (`--features encryption`)
- [ ] Establish a regular backup schedule
- [ ] Monitor with Prometheus + Grafana
- [ ] Review audit logs

See [production-checklist.md](production-checklist.md) for a comprehensive pre-launch checklist.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port already in use | Change with `--address 0.0.0.0:8081` |
| Permission denied | Check file ownership and process user |
| High memory usage | Enable quantization or increase server memory |
| Slow queries | Run `needle tune` and increase `ef_search` |
| Growing deleted vectors | Run `needle compact` to reclaim space |

See [OPERATIONS.md](OPERATIONS.md#troubleshooting) for detailed runbooks.

## See Also

- [Deployment Guide](deployment.md) — Docker, Kubernetes, Helm
- [Operations Manual](OPERATIONS.md) — Day-to-day procedures
- [Production Checklist](production-checklist.md) — Pre-launch verification
- [API Reference](api-reference.md) — Endpoint documentation
