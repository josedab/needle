# ADR-0017: Predictive Auto-Scaling System

## Status

Accepted

## Context

Vector databases in production face highly variable workloads:

1. **Traffic spikes** — Marketing campaigns, viral content, or news events cause sudden load increases
2. **Cost optimization** — Over-provisioning wastes resources; under-provisioning degrades user experience
3. **Latency SLAs** — P99 latency must remain consistent despite load variations
4. **Data growth** — Vector counts grow continuously, requiring periodic capacity adjustments
5. **Reactive limitations** — Traditional auto-scaling responds after degradation is already occurring

Cloud-native applications need intelligent scaling that anticipates demand rather than merely reacting to it.

## Decision

Needle implements a **predictive auto-scaling system** with ML-based load forecasting:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoScaler                               │
├─────────────────────────────────────────────────────────────┤
│  Metrics Collector  │  QPS, latency, CPU, memory, GPU       │
├─────────────────────┼───────────────────────────────────────┤
│  Time Series Store  │  Circular buffer of historical data   │
├─────────────────────┼───────────────────────────────────────┤
│  Forecaster         │  Exponential smoothing + seasonality  │
├─────────────────────┼───────────────────────────────────────┤
│  Policy Engine      │  Scaling decisions based on forecast  │
├─────────────────────┼───────────────────────────────────────┤
│  Executor           │  Applies scaling actions              │
└─────────────────────┴───────────────────────────────────────┘
```

### AutoScaler Structure

```rust
pub struct AutoScaler {
    /// Current scaling configuration
    config: AutoScalerConfig,
    /// Historical metrics for forecasting
    metrics_history: CircularBuffer<MetricsSample>,
    /// Forecaster for load prediction
    forecaster: LoadForecaster,
    /// Current resource allocation
    current_resources: ResourceAllocation,
    /// Scaling cooldown tracker
    last_scale_time: Instant,
}
```

### Scaling Policies

| Policy | Trigger | Action |
|--------|---------|--------|
| **Scale Up** | Predicted load > 80% capacity | Add replicas or resources |
| **Scale Down** | Predicted load < 40% for 10min | Remove replicas (with cooldown) |
| **Emergency** | Current latency > SLA | Immediate scale up |
| **Scheduled** | Known traffic patterns | Pre-scale before events |

### Forecasting Model

```rust
/// Triple exponential smoothing (Holt-Winters) for seasonal forecasting
pub struct LoadForecaster {
    /// Level component
    level: f64,
    /// Trend component
    trend: f64,
    /// Seasonal components (hourly, daily, weekly)
    seasonal: Vec<f64>,
    /// Smoothing parameters
    alpha: f64,  // Level
    beta: f64,   // Trend
    gamma: f64,  // Seasonal
}

impl LoadForecaster {
    pub fn predict(&self, periods_ahead: usize) -> f64 {
        let trend_forecast = self.level + (periods_ahead as f64) * self.trend;
        let seasonal_idx = periods_ahead % self.seasonal.len();
        trend_forecast * self.seasonal[seasonal_idx]
    }
}
```

### Resource Allocation Model

```rust
pub struct ResourceAllocation {
    /// Number of query replicas
    pub replicas: usize,
    /// Memory per replica (bytes)
    pub memory_per_replica: usize,
    /// CPU cores per replica
    pub cpu_per_replica: f32,
    /// GPU memory if available
    pub gpu_memory: Option<usize>,
    /// Shard distribution
    pub shards: Vec<ShardAssignment>,
}
```

### Hot/Cold Data Tiering

```
┌─────────────────────────────────────────────────────────────┐
│                      Hot Tier                               │
│            (Memory, <1ms latency)                           │
│      Frequently accessed vectors, recent data               │
├─────────────────────────────────────────────────────────────┤
│                      Warm Tier                              │
│            (SSD, <10ms latency)                             │
│      Moderate access frequency                              │
├─────────────────────────────────────────────────────────────┤
│                      Cold Tier                              │
│            (Cloud storage, <100ms latency)                  │
│      Archival data, infrequent access                       │
└─────────────────────────────────────────────────────────────┘
```

### Code References

- `src/autoscaling.rs:514-650` — `AutoScaler` implementation
- `src/autoscaling.rs:940-1000` — `SharedAutoScaler` thread-safe wrapper
- `src/autoscaling.rs:100-200` — Forecasting algorithms
- `src/autoscaling.rs:300-400` — Policy engine and decision logic

## Consequences

### Benefits

1. **Proactive scaling** — Resources provisioned before demand arrives
2. **Cost optimization** — Right-sized resources reduce cloud spending 20-40%
3. **Consistent latency** — P99 latency maintained during traffic variations
4. **Automatic operation** — No manual capacity planning required
5. **Seasonal awareness** — Learns daily/weekly patterns automatically
6. **Emergency handling** — Immediate response to unexpected spikes

### Tradeoffs

1. **Complexity** — Forecasting adds ML components to the system
2. **Cold start** — New deployments need time to learn patterns
3. **Over-prediction** — Aggressive forecasting may over-provision
4. **External dependencies** — Requires orchestration integration (K8s, ECS)
5. **State management** — Scaling state must be persisted across restarts

### What This Enabled

- Hands-off production operation with SLA guarantees
- Significant cost reduction through right-sizing
- Graceful handling of viral traffic events
- Predictable capacity planning and budgeting
- Integration with cloud-native orchestration

### What This Prevented

- Simple single-node deployments (scaling adds coordination)
- Deterministic resource usage (varies with predictions)
- Zero startup time (forecaster needs historical data)

### Scaling Decision Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Collect       │────▶│  Forecast      │────▶│  Decide        │
│  Metrics       │     │  Load          │     │  Action        │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   QPS, latency,         15min, 1hr,           Scale up/down,
   CPU, memory           24hr forecasts        migrate shards
                                                      │
                                                      ▼
                                              ┌────────────────┐
                                              │  Execute       │
                                              │  Scaling       │
                                              └────────────────┘
```

### Cooldown and Stability

To prevent oscillation:

```rust
const SCALE_UP_COOLDOWN: Duration = Duration::from_secs(60);
const SCALE_DOWN_COOLDOWN: Duration = Duration::from_secs(300);

fn should_scale(&self, action: ScaleAction) -> bool {
    let cooldown = match action {
        ScaleAction::Up => SCALE_UP_COOLDOWN,
        ScaleAction::Down => SCALE_DOWN_COOLDOWN,
    };

    self.last_scale_time.elapsed() > cooldown
}
```
