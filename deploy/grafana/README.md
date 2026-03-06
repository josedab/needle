# Needle Grafana Dashboards

Pre-built Grafana dashboards for monitoring Needle vector database deployments.

## Dashboards

| File | Description |
|------|-------------|
| `needle-overview.json` | High-level operational health: request rate, latency, errors, vector/collection counts |
| `needle-collections.json` | Per-collection breakdown: vector counts, search latency, memory usage |
| `needle-dashboard.json` | Legacy combined dashboard |

## Prerequisites

- Needle running with `--features server,metrics` (exposes `/metrics` endpoint)
- Prometheus scraping the Needle `/metrics` endpoint
- Grafana with a Prometheus data source configured

## Importing Dashboards

### Via Grafana UI

1. Open Grafana → **Dashboards** → **New** → **Import**
2. Click **Upload dashboard JSON file** and select the `.json` file
3. Select your Prometheus data source when prompted
4. Click **Import**

### Via Provisioning

Copy the JSON files into your Grafana provisioning directory and add a dashboard provider:

```yaml
# /etc/grafana/provisioning/dashboards/needle.yaml
apiVersion: 1
providers:
  - name: Needle
    folder: Needle
    type: file
    options:
      path: /var/lib/grafana/dashboards/needle
```

Then copy the dashboard JSON files to `/var/lib/grafana/dashboards/needle/`.

## Data Source Variable

All dashboards use `${DS_PROMETHEUS}` as the data source variable. On import, Grafana will prompt you to map this to your Prometheus data source.
