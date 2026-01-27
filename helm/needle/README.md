# Needle Helm Chart

A Helm chart for deploying [Needle Vector Database](https://github.com/needle-db/needle) on Kubernetes.

- **Chart version**: 0.1.0
- **App version**: 0.1.0

## Prerequisites

- Helm 3.x
- Kubernetes 1.24+
- PV provisioner support in the cluster (if persistence is enabled)

## Quick Install

```bash
# Add the chart (from local directory)
helm install needle ./helm/needle

# Or with custom values
helm install needle ./helm/needle -f my-values.yaml
```

## Values Reference

| Key | Default | Description |
|-----|---------|-------------|
| `replicaCount` | `1` | Number of replicas |
| `image.repository` | `ghcr.io/needle-db/needle` | Container image repository |
| `image.tag` | `""` (appVersion) | Image tag override |
| `image.pullPolicy` | `IfNotPresent` | Image pull policy |
| `service.type` | `ClusterIP` | Kubernetes service type |
| `service.port` | `8080` | Service port |
| `persistence.enabled` | `true` | Enable persistent storage |
| `persistence.size` | `10Gi` | PVC size |
| `persistence.storageClass` | `""` | Storage class (default: cluster default) |
| `needle.logLevel` | `info` | Log level (trace/debug/info/warn/error) |
| `needle.databaseFile` | `needle.db` | Database filename within data directory |
| `needle.address` | `0.0.0.0:8080` | Server listen address |
| `metrics.enabled` | `false` | Enable Prometheus metrics |
| `metrics.serviceMonitor.enabled` | `false` | Create ServiceMonitor for Prometheus Operator |
| `ingress.enabled` | `false` | Enable ingress resource |
| `resources.requests.cpu` | `500m` | CPU request |
| `resources.requests.memory` | `1Gi` | Memory request |
| `resources.limits.cpu` | `2000m` | CPU limit |
| `resources.limits.memory` | `4Gi` | Memory limit |

## Environment-Specific Overrides

Pre-configured value files are provided for common environments:

```bash
# Development (lower resources, debug logging)
helm install needle ./helm/needle -f helm/needle/values-dev.yaml

# Production (higher resources, monitoring enabled)
helm install needle ./helm/needle -f helm/needle/values-prod.yaml
```

## Uninstall

```bash
helm uninstall needle
```

**Note:** PVCs are not automatically deleted. Remove them manually if no longer needed:

```bash
kubectl delete pvc -l app.kubernetes.io/name=needle
```
