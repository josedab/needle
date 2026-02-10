# Needle Deployment Guide

This guide covers deploying Needle in production environments using Docker, Kubernetes, and Helm.

## Table of Contents

- [Quick Start](#quick-start)
- [Docker](#docker)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Helm Chart](#helm-chart)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Production Checklist](#production-checklist)

---

## Quick Start

### One-command Demo

```bash
just demo
# or
./scripts/quickstart.sh
```

### Local Development

```bash
# Fast path: server-only features
cargo run --features server -- serve -a 127.0.0.1:8080

# Release build
cargo build --release --features server
./target/release/needle serve --address 127.0.0.1:8080

# Full feature build (slower)
cargo build --release --features full
```

### Docker (Fastest)

```bash
# Build and run
docker build -t needle:latest .
docker run -p 8080:8080 -v needle-data:/data needle:latest
```

---

## Docker

### Building the Image

```bash
# Standard build
docker build -t needle:latest .

# With build arguments
docker build \
  --build-arg RUST_VERSION=1.75 \
  --build-arg FEATURES=full \
  -t needle:latest .
```

### Running the Container

```bash
# Basic run
docker run -d \
  --name needle \
  -p 8080:8080 \
  -v needle-data:/data \
  needle:latest

# With environment variables
docker run -d \
  --name needle \
  -p 8080:8080 \
  -v needle-data:/data \
  -e RUST_LOG=info \
  -e NEEDLE_DATA_DIR=/data \
  needle:latest

# Custom command
docker run -d \
  --name needle \
  -p 8080:8080 \
  -v /path/to/data:/data \
  needle:latest \
  serve --address 0.0.0.0:8080 --database /data/vectors.needle
```

### Health Check

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

### Commands

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f needle

# Stop services
docker compose down

# Stop and remove volumes
docker compose down -v
```

---

## Kubernetes

### Prerequisites

- Kubernetes cluster (1.19+)
- kubectl configured
- PersistentVolume provisioner (for StatefulSet)

### Deployment

```bash
# Apply all manifests
kubectl apply -f deploy/kubernetes/

# Or individually
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/pvc.yaml
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml
kubectl apply -f deploy/kubernetes/hpa.yaml
```

### Manifest Overview

| File | Description |
|------|-------------|
| `configmap.yaml` | Configuration parameters |
| `deployment.yaml` | Deployment for stateless mode |
| `statefulset.yaml` | StatefulSet for persistent storage |
| `service.yaml` | ClusterIP and LoadBalancer services |
| `pvc.yaml` | PersistentVolumeClaim |
| `hpa.yaml` | Horizontal Pod Autoscaler |
| `crd.yaml` | Custom Resource Definitions |

### Verify Deployment

```bash
# Check pods
kubectl get pods -l app=needle

# Check services
kubectl get svc needle

# View logs
kubectl logs -f deployment/needle

# Port forward for local access
kubectl port-forward svc/needle 8080:8080
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment needle --replicas=3

# HPA is configured for automatic scaling based on CPU/memory
kubectl get hpa needle
```

---

## Helm Chart

### Installation

```bash
# Add the chart (if published)
helm repo add needle https://charts.needle.dev
helm repo update

# Or install from local chart
helm install needle ./helm/needle
```

### Configuration

```bash
# Install with custom values
helm install needle ./helm/needle \
  --set replicaCount=3 \
  --set persistence.size=100Gi \
  --set resources.limits.memory=4Gi

# Using a values file
helm install needle ./helm/needle -f my-values.yaml
```

### Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `needle` |
| `image.tag` | Image tag | `latest` |
| `persistence.enabled` | Enable persistence | `true` |
| `persistence.size` | PVC size | `10Gi` |
| `persistence.storageClass` | Storage class | `""` |
| `resources.limits.cpu` | CPU limit | `2` |
| `resources.limits.memory` | Memory limit | `4Gi` |
| `metrics.enabled` | Enable Prometheus metrics | `true` |
| `ingress.enabled` | Enable ingress | `false` |

### Environment-Specific Values

```bash
# Development
helm install needle ./helm/needle -f helm/needle/values-dev.yaml

# Production
helm install needle ./helm/needle -f helm/needle/values-prod.yaml
```

### Upgrade

```bash
helm upgrade needle ./helm/needle --set image.tag=v1.2.0
```

### Uninstall

```bash
helm uninstall needle
```

---

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEEDLE_DATA_DIR` | Data directory | `/data` |
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | `info` |
| `NEEDLE_ADDRESS` | Server bind address | `0.0.0.0:8080` |
| `NEEDLE_METRICS_PORT` | Prometheus metrics port | `9090` |

### Server Configuration

```bash
# Full server options
needle serve \
  --address 0.0.0.0:8080 \
  --database /data/vectors.needle \
  --metrics-port 9090 \
  --cors-origins "*"
```

---

## Monitoring

### Prometheus Metrics

Needle exposes metrics at `/metrics` when compiled with `--features metrics`:

```bash
# Scrape config for Prometheus
scrape_configs:
  - job_name: 'needle'
    static_configs:
      - targets: ['needle:8080']
```

### Available Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `needle_vectors_total` | Gauge | Total vectors by collection |
| `needle_search_duration_seconds` | Histogram | Search latency |
| `needle_insert_duration_seconds` | Histogram | Insert latency |
| `needle_memory_bytes` | Gauge | Memory usage |
| `needle_requests_total` | Counter | Total HTTP requests |

### Grafana Dashboard

Import the pre-built dashboard from `deploy/grafana/provisioning/dashboards/`.

Access Grafana at http://localhost:3000 (when using docker-compose with monitoring profile).

### Alerting

Example Prometheus alerting rules:

```yaml
groups:
  - name: needle
    rules:
      - alert: NeedleHighLatency
        expr: histogram_quantile(0.95, needle_search_duration_seconds) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High search latency detected"

      - alert: NeedleHighMemory
        expr: needle_memory_bytes > 8e9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
```

---

## Production Checklist

### Security

- [ ] Run as non-root user (default in Docker image)
- [ ] Enable TLS termination (via ingress or load balancer)
- [ ] Configure network policies to restrict access
- [ ] Enable authentication if exposing publicly
- [ ] Review and configure CORS settings

### Performance

- [ ] Set appropriate resource limits
- [ ] Configure HPA for auto-scaling
- [ ] Use SSD-backed storage for persistence
- [ ] Tune HNSW parameters with `needle tune`
- [ ] Enable connection pooling for high-throughput

### Reliability

- [ ] Configure health checks
- [ ] Set up liveness and readiness probes
- [ ] Enable persistence with replicated storage
- [ ] Configure backup schedule
- [ ] Set up monitoring and alerting

### Storage

- [ ] Use PersistentVolumeClaims in Kubernetes
- [ ] Configure appropriate storage class
- [ ] Plan for storage growth
- [ ] Set up regular backups

### Example Production Values

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
  serviceMonitor:
    enabled: true

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt
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

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker logs needle

# Verify permissions on data directory
ls -la /path/to/data
```

**High memory usage:**
```bash
# Check collection stats
curl http://localhost:8080/collections/my_collection

# Consider compaction
curl -X POST http://localhost:8080/collections/my_collection/compact
```

**Slow queries:**
```bash
# Use auto-tuning
needle tune --vectors 1000000 --dimensions 384 --profile high-recall
```

### Getting Help

- Check logs: `docker logs needle` or `kubectl logs -f deployment/needle`
- Health endpoint: `curl http://localhost:8080/health`
- Metrics endpoint: `curl http://localhost:8080/metrics`

---

## See Also

- [Production Checklist](production-checklist.md) - Pre-deployment verification checklist
- [Operations Guide](OPERATIONS.md) - Day-to-day operations and monitoring
- [Distributed Operations](distributed-operations.md) - Sharding, Raft, and clustering
- [WASM Guide](WASM_GUIDE.md) - WebAssembly integration for browser/edge deployment
