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
| `imagePullSecrets` | `[]` | Docker registry pull secrets |
| `nameOverride` | `""` | Override chart name |
| `fullnameOverride` | `""` | Override full resource name |
| `serviceAccount.create` | `true` | Create a service account |
| `serviceAccount.annotations` | `{}` | Service account annotations |
| `serviceAccount.name` | `""` | Service account name (generated if empty) |
| `podAnnotations` | `{}` | Extra pod annotations |
| `podSecurityContext.fsGroup` | `1000` | Pod filesystem group ID |
| `securityContext.capabilities.drop` | `["ALL"]` | Dropped Linux capabilities |
| `securityContext.readOnlyRootFilesystem` | `true` | Mount root filesystem as read-only |
| `securityContext.runAsNonRoot` | `true` | Prevent running as root |
| `securityContext.runAsUser` | `1000` | Container user ID |
| `service.type` | `ClusterIP` | Kubernetes service type |
| `service.port` | `8080` | Service port |
| `ingress.enabled` | `false` | Enable ingress resource |
| `ingress.className` | `""` | Ingress class name |
| `ingress.annotations` | `{}` | Ingress annotations |
| `ingress.hosts` | `[{host: "needle.local", paths: [{path: "/", pathType: "Prefix"}]}]` | Ingress host rules |
| `ingress.tls` | `[]` | Ingress TLS configuration |
| `resources.requests.cpu` | `500m` | CPU request |
| `resources.requests.memory` | `1Gi` | Memory request |
| `resources.limits.cpu` | `2000m` | CPU limit |
| `resources.limits.memory` | `4Gi` | Memory limit |
| `persistence.enabled` | `true` | Enable persistent storage |
| `persistence.storageClass` | `""` | Storage class (default: cluster default) |
| `persistence.accessModes` | `["ReadWriteOnce"]` | PVC access modes |
| `persistence.size` | `10Gi` | PVC size |
| `persistence.annotations` | `{}` | PVC annotations |
| `needle.logLevel` | `info` | Log level (trace/debug/info/warn/error) |
| `needle.databaseFile` | `needle.db` | Database filename within data directory |
| `needle.address` | `0.0.0.0:8080` | Server listen address |
| `needle.extraEnv` | `[]` | Additional environment variables |
| `metrics.enabled` | `false` | Enable Prometheus metrics |
| `metrics.serviceMonitor.enabled` | `false` | Create ServiceMonitor for Prometheus Operator |
| `metrics.serviceMonitor.interval` | `30s` | Scrape interval |
| `metrics.serviceMonitor.scrapeTimeout` | `10s` | Scrape timeout |
| `metrics.serviceMonitor.labels` | `{}` | Extra ServiceMonitor labels |
| `livenessProbe.httpGet.path` | `/health` | Liveness probe endpoint |
| `livenessProbe.initialDelaySeconds` | `10` | Liveness probe initial delay |
| `livenessProbe.periodSeconds` | `30` | Liveness probe period |
| `livenessProbe.timeoutSeconds` | `5` | Liveness probe timeout |
| `livenessProbe.failureThreshold` | `3` | Liveness probe failure threshold |
| `readinessProbe.httpGet.path` | `/health` | Readiness probe endpoint |
| `readinessProbe.initialDelaySeconds` | `5` | Readiness probe initial delay |
| `readinessProbe.periodSeconds` | `10` | Readiness probe period |
| `readinessProbe.timeoutSeconds` | `5` | Readiness probe timeout |
| `readinessProbe.failureThreshold` | `3` | Readiness probe failure threshold |
| `nodeSelector` | `{}` | Node selector constraints |
| `tolerations` | `[]` | Pod tolerations |
| `affinity` | `{}` | Pod affinity rules |
| `podDisruptionBudget.enabled` | `false` | Enable PDB |
| `podDisruptionBudget.minAvailable` | `1` | Minimum available pods during disruption |

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
