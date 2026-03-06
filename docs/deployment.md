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

> **Note**: The hosted Helm repository at `charts.needle.dev` is not yet available. Use the local chart installation method below until the hosted repository is published.

### Installation

```bash
# Install from the local chart bundled in this repository
helm install needle ./helm/needle

# (Future) Once charts.needle.dev is live:
# helm repo add needle https://charts.needle.dev
# helm repo update
# helm install needle needle/needle
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
| `NEEDLE_DATABASE` | Database filename within data directory | `needle.db` |
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | `info` |
| `NEEDLE_ADDRESS` | Server bind address | `0.0.0.0:8080` |
| `NEEDLE_ENCRYPTION_KEY` | Hex-encoded encryption key for encrypted collections (ChaCha20-Poly1305). **Security-critical** — use a secrets manager in production. | — |
| `NEEDLE_METRICS_PORT` | Prometheus metrics port (used in Kubernetes deployments) | `9090` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry OTLP collector endpoint | — |
| `OTEL_SERVICE_NAME` | Service name for distributed traces | `needle` |
| `OPENAI_API_KEY` | API key for OpenAI embeddings | — |
| `COHERE_API_KEY` | API key for Cohere embeddings/reranking | — |
| `HUGGINGFACE_API_KEY` | API key for Hugging Face Inference API | — |
| `HUGGINGFACE_MODEL` | Hugging Face model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `EDGE_CONFIG` | Edge runtime configuration (experimental) | — |
| `BLOB_READ_WRITE_TOKEN` | Blob storage read/write token (experimental) | — |
| `NEEDLE_DB_PATH` | Database file path (used in cloud deploy configs: GCP Cloud Run, AWS App Runner, Azure Container Instances) | `/data/vectors.needle` |
| `NEEDLE_HOST` | Server bind host (used in cloud deploy configs) | `0.0.0.0` |
| `NEEDLE_PORT` | Server bind port (used in cloud deploy configs) | `8080` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON key file (used by GCS cloud storage backend, requires `cloud-storage-gcs` feature) | — |
| `XDG_CACHE_HOME` | Base directory for cached ONNX models and embedded model weights (falls back to `~/.cache`) | — |
| `HOSTNAME` | Hostname included in telemetry/observability metadata (auto-detected if not set) | — |

### Embedding & Observability Configuration

These variables configure embedding providers and tracing. They require the corresponding Cargo features to be enabled.

| Variable | Description | Default | Feature |
|----------|-------------|---------|---------|
| `NEEDLE_EMBEDDING_PROVIDER` | Primary embedding provider (`openai`, `cohere`, `ollama`) | — | `embedding-providers` |
| `NEEDLE_EMBEDDING_FALLBACK` | Fallback provider chain, comma-separated (tried in order if primary fails) | — | `embedding-providers` |
| `NEEDLE_EMBEDDING_STRATEGY` | Routing strategy: `priority_chain`, `lowest_cost`, `lowest_latency`, `round_robin` | `priority_chain` | `embedding-providers` |
| `NEEDLE_MODEL_DIR` | Directory for downloaded ONNX embedding models (falls back to `~/.needle/models/`) | — | `embeddings` |
| `NEEDLE_TRACE_SAMPLE_RATE` | Sampling rate for distributed traces (`0.0` = none, `1.0` = all) | `0.01` | `server` |

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

## TLS / HTTPS Configuration

Needle's HTTP server does **not** handle TLS termination directly. For production deployments, use a reverse proxy or load balancer to terminate TLS in front of Needle.

### Option 1: Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl http2;
    server_name needle.example.com;

    ssl_certificate     /etc/ssl/certs/needle.pem;
    ssl_certificate_key /etc/ssl/private/needle.key;
    ssl_protocols       TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Option 2: Caddy (Automatic HTTPS)

```caddyfile
needle.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

Caddy automatically provisions and renews TLS certificates via Let's Encrypt.

### Option 3: Kubernetes Ingress

See the [Helm Chart section](#helm-chart) — the included ingress template supports TLS:

```yaml
ingress:
  enabled: true
  tls:
    - secretName: needle-tls
      hosts:
        - needle.example.com
```

### Trusted Proxies

When running behind a reverse proxy, configure Needle to trust proxy headers for correct client IP extraction. The server trusts `127.0.0.1` and `::1` by default. Configure additional trusted proxies via `ServerConfig::trusted_proxies`.

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

### Server Won't Start

**Port already in use:**
```bash
# Check what's using the port
lsof -i :8080
# Or change the bind address
NEEDLE_ADDRESS=0.0.0.0:8081 needle serve
```

**Missing data directory:**
```bash
# Ensure the data directory exists and is writable
mkdir -p /data
# Or specify a different path
needle serve --database /tmp/vectors.needle
```

### Database File Permission Errors

```bash
# Check file ownership and permissions
ls -la /path/to/vectors.needle

# Fix permissions (Docker: ensure the container user matches the file owner)
chmod 644 /path/to/vectors.needle
chown $(id -u):$(id -g) /path/to/vectors.needle

# Docker volume mounts — use explicit user mapping
docker run -u $(id -u):$(id -g) -v /host/data:/data ghcr.io/anthropics/needle
```

### OTEL Exporter Connection Failures

```bash
# Verify the OTLP endpoint is reachable
curl -v http://localhost:4317

# Check the environment variable is set correctly
echo $OTEL_EXPORTER_OTLP_ENDPOINT

# Disable tracing temporarily to isolate the issue
unset OTEL_EXPORTER_OTLP_ENDPOINT
needle serve
```

If the collector is running but connections fail, ensure the endpoint uses the correct
protocol (`http` vs `grpc`) and port (`4317` for gRPC, `4318` for HTTP).

### Embedding Provider API Key Errors

```bash
# Verify API key is set
echo $OPENAI_API_KEY | head -c 8

# Test connectivity (OpenAI example)
curl -s https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | head -c 200

# Common causes:
# - Key not exported in the shell (use `export OPENAI_API_KEY=sk-...`)
# - Key not passed into Docker container (add -e OPENAI_API_KEY to docker run)
# - Rate limiting — check provider dashboard for quota status
```

### Docker Volume Mount Issues

```bash
# Container won't start — check logs
docker logs needle

# Verify permissions on data directory
ls -la /path/to/data

# Common fix: ensure the host directory exists before mounting
mkdir -p /host/data
docker run -v /host/data:/data ghcr.io/anthropics/needle

# SELinux systems may require the :z flag
docker run -v /host/data:/data:z ghcr.io/anthropics/needle
```

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

## Edge Deployment

For deploying Needle to serverless edge platforms (Cloudflare Workers, Deno Deploy), see the [Edge Deployment Guide](../deploy/edge/README.md). These templates use the WASM build to run Needle at the edge with minimal latency.
