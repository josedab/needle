---
sidebar_position: 7
---

# Deployment Guide

Deploy Needle using Docker, Docker Compose, Kubernetes, or Helm.

---

## Quick Start

```bash
# Fastest: Docker
docker run -d -p 8080:8080 -v needle-data:/data ghcr.io/anthropics/needle:latest

# From source
cargo build --release --features server
./target/release/needle serve -a 0.0.0.0:8080 -d /data/vectors.needle

# One-command demo
./scripts/quickstart.sh
```

---

## Docker

### Build the Image

```bash
docker build -t needle:latest .
```

### Run

```bash
docker run -d \
  --name needle \
  -p 8080:8080 \
  -v needle-data:/data \
  -e RUST_LOG=info \
  needle:latest
```

### Verify

```bash
curl http://localhost:8080/health
```

---

## Docker Compose

### Basic Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  needle:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - needle-data:/data
    environment:
      - RUST_LOG=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 5s
      retries: 3

volumes:
  needle-data:
```

### With Monitoring Stack

```bash
# Start with Prometheus and Grafana
docker compose --profile monitoring up -d
```

This starts:
- **Needle** on port 8080
- **Prometheus** on port 9090
- **Grafana** on port 3000 (admin/admin)

---

## Kubernetes

### Prerequisites

- Kubernetes 1.19+
- kubectl configured
- PersistentVolume provisioner

### Deploy

```bash
kubectl apply -f deploy/kubernetes/
```

### Verify

```bash
kubectl get pods -l app=needle
kubectl port-forward svc/needle 8080:8080
curl http://localhost:8080/health
```

### Resource Configuration

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

### Health Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
```

---

## Helm Chart

### Install

```bash
# From local chart
helm install needle ./helm/needle

# With custom values
helm install needle ./helm/needle \
  --set replicaCount=3 \
  --set persistence.size=100Gi \
  --set resources.limits.memory=8Gi
```

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | PVC size | `10Gi` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `metrics.enabled` | Enable Prometheus metrics | `true` |
| `ingress.enabled` | Enable ingress | `false` |

### Production Values Example

```yaml
# values-prod.yaml
replicaCount: 3

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 2
    memory: 4Gi

persistence:
  enabled: true
  size: 100Gi
  storageClass: ssd

metrics:
  enabled: true

ingress:
  enabled: true
  hosts:
    - host: needle.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: needle-tls
      hosts:
        - needle.example.com
```

---

## Monitoring

### Prometheus Metrics

Enable with `--features metrics`. Needle exposes metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `needle_vectors_total` | Gauge | Total vectors by collection |
| `needle_search_duration_seconds` | Histogram | Search latency |
| `needle_insert_duration_seconds` | Histogram | Insert latency |
| `needle_memory_bytes` | Gauge | Memory usage |
| `needle_requests_total` | Counter | Total HTTP requests |

### Grafana

Import the pre-built dashboard from `deploy/grafana/provisioning/dashboards/`.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEEDLE_DATA_DIR` | Data directory | `/data` |
| `RUST_LOG` | Log level | `info` |
| `NEEDLE_ADDRESS` | Server bind address | `0.0.0.0:8080` |

---

## See Also

- [Operations Guide](/docs/advanced/operations) — Monitoring, backup, tuning
- [Production Checklist](/docs/guides/production-checklist) — Pre-deployment verification
- [Distributed Operations](/docs/advanced/distributed) — Sharding and replication
