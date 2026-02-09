# Deployment Resources

This directory contains deployment configurations for running Needle in various environments.

> **Start here:** For most users, **Docker Compose** ([`docker-compose.yml`](../docker-compose.yml)) is the fastest way to get Needle running in production. For Kubernetes environments, use the **Helm chart** ([`helm/needle/`](../helm/needle/README.md)) which provides the most flexibility.

## Directory Structure

```
deploy/
├── kubernetes/       # Raw Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── statefulset.yaml
│   ├── configmap.yaml
│   ├── crd.yaml
│   ├── hpa.yaml
│   └── pvc.yaml
├── edge/             # Edge deployment (Cloudflare Workers, Deno Deploy)
│   ├── README.md
│   ├── cloudflare-worker.js
│   ├── cloudflare-worker/
│   ├── deno-deploy/
│   └── wrangler.toml
├── grafana/          # Grafana dashboard JSON
└── prometheus/       # Prometheus scrape configuration
    └── prometheus.yml
```

## Kubernetes Manifests vs Helm

| Use Case | Recommended |
|----------|-------------|
| Quick testing / CI | Raw manifests (`deploy/kubernetes/`) |
| Production with customization | Helm chart (`helm/needle/`) |
| GitOps (ArgoCD, Flux) | Either — Helm is more flexible |

### Quick Start with Raw Manifests

```bash
kubectl apply -f deploy/kubernetes/
```

### Quick Start with Helm

See [`helm/needle/README.md`](../helm/needle/README.md) for full instructions.

## Grafana Dashboard Setup

1. Import the dashboard JSON from `deploy/grafana/` into your Grafana instance
2. Ensure Prometheus is scraping the Needle `/metrics` endpoint
3. Configure the Prometheus data source in Grafana

```bash
# If using Prometheus Operator, enable the ServiceMonitor in Helm values:
# metrics.serviceMonitor.enabled: true
```

## Edge Deployment

The `edge/` directory contains configurations for deploying Needle's WASM build to edge platforms. See [`edge/README.md`](edge/README.md) for details.
