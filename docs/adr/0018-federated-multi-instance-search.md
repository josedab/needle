# ADR-0018: Federated Multi-Instance Search

## Status

Accepted

## Context

Large-scale deployments often require distributed vector search across multiple instances:

1. **Data locality** — Regulations require data to stay in specific regions (GDPR, data residency)
2. **Horizontal scaling** — Single instances cannot hold billions of vectors
3. **Fault tolerance** — Geographic distribution provides disaster recovery
4. **Latency optimization** — Users should query the nearest instance
5. **Organizational boundaries** — Different teams may own different vector stores

While Raft replication (ADR-0010) handles consensus within a cluster, federation addresses cross-cluster, cross-region, and cross-organization search.

## Decision

Needle implements **federated search** with latency-aware query routing:

### Federation Architecture

```
                          ┌────────────────┐
                          │   Federation   │
                          │   Coordinator  │
                          └───────┬────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
    ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    │   Needle    │        │   Needle    │        │   Needle    │
    │   US-East   │        │   EU-West   │        │   APAC      │
    │   Instance  │        │   Instance  │        │   Instance  │
    └─────────────┘        └─────────────┘        └─────────────┘
```

### Federation Configuration

```rust
pub struct FederationConfig {
    /// Unique ID for this federation
    pub federation_id: String,
    /// Member instances
    pub members: Vec<FederationMember>,
    /// Query routing strategy
    pub routing_strategy: RoutingStrategy,
    /// Result merge strategy
    pub merge_strategy: MergeStrategy,
    /// Timeout for remote queries
    pub query_timeout: Duration,
    /// Maximum parallel queries
    pub max_concurrent_queries: usize,
}
```

### Routing Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `Broadcast` | Query all instances | Global search, highest recall |
| `Nearest` | Query geographically nearest | Latency optimization |
| `HashRouted` | Route based on query hash | Consistent routing for caching |
| `Weighted` | Probability-based routing | Load balancing |
| `Custom` | User-defined routing logic | Complex business rules |

### Federated Search Flow

```
┌──────────┐    ┌──────────────┐    ┌──────────────────┐
│  Query   │───▶│  Route to    │───▶│  Parallel Query  │
│  Arrives │    │  Instances   │    │  Execution       │
└──────────┘    └──────────────┘    └──────────────────┘
                                           │
                                           ▼
┌──────────┐    ┌──────────────┐    ┌──────────────────┐
│  Return  │◀───│  Rank        │◀───│  Merge Results   │
│  Results │    │  Normalize   │    │  from Instances  │
└──────────┘    └──────────────┘    └──────────────────┘
```

### Federation Result Structure

```rust
pub struct FederatedSearchResult {
    /// The vector ID
    pub id: String,
    /// Distance to query
    pub distance: f32,
    /// Which instance returned this result
    pub source_instance: String,
    /// Metadata
    pub metadata: serde_json::Value,
    /// Instance-local ranking
    pub local_rank: usize,
}

pub struct FederatedSearchResponse {
    /// Merged and ranked results
    pub results: Vec<FederatedSearchResult>,
    /// Per-instance response times
    pub instance_latencies: HashMap<String, Duration>,
    /// Instances that timed out or failed
    pub failed_instances: Vec<String>,
    /// Total query time
    pub total_time: Duration,
}
```

### Merge Strategies

```rust
pub enum MergeStrategy {
    /// Simple distance-based merge (default)
    DistanceMerge,
    /// Reciprocal Rank Fusion across instances
    RrfMerge { k: f32 },
    /// Weighted merge based on instance confidence
    WeightedMerge { weights: HashMap<String, f32> },
    /// Custom merge function
    Custom(Box<dyn Fn(Vec<Vec<SearchResult>>) -> Vec<SearchResult>>),
}
```

### Health Monitoring

```rust
pub struct FederationHealth {
    /// Overall federation status
    pub status: HealthStatus,
    /// Per-instance health
    pub instances: HashMap<String, InstanceHealth>,
    /// Recent query success rate
    pub success_rate: f32,
    /// Average cross-instance latency
    pub avg_latency: Duration,
}
```

### Code References

- `src/federated.rs:99-150` — `FederationConfig` configuration
- `src/federated.rs:540-560` — `FederatedSearchResult` structure
- `src/federated.rs:555-580` — `FederatedSearchResponse` with latencies
- `src/federated.rs:838-870` — `FederationHealth` monitoring
- `src/federated.rs:852-1050` — `Federation` implementation
- `src/federated.rs:1077-1130` — `FederationStats` statistics

## Consequences

### Benefits

1. **Global scale** — Search across billions of vectors distributed worldwide
2. **Data residency** — Keep data in required regions while enabling global search
3. **Fault tolerance** — Continue operating when some instances are unavailable
4. **Latency optimization** — Route to nearest instance for speed
5. **Incremental migration** — Add/remove instances without downtime
6. **Organizational flexibility** — Different teams can own different instances

### Tradeoffs

1. **Network latency** — Cross-region queries add 50-200ms latency
2. **Result quality variance** — Distributed search may miss global best matches
3. **Operational complexity** — Multiple instances require coordination
4. **Consistency challenges** — No strong consistency across federation
5. **Cost** — Network transfer between regions adds cloud costs

### What This Enabled

- Global vector search with data locality compliance
- Billion-scale search without single-instance limitations
- Multi-cloud and hybrid cloud deployments
- Gradual migration between infrastructure providers
- Team-owned vector stores with unified search

### What This Prevented

- Strong consistency guarantees across federation
- Atomic cross-instance operations
- Simple single-binary deployment at scale
- Deterministic query latency

### Latency-Aware Routing

```rust
impl Federation {
    fn select_instances(&self, query: &Query) -> Vec<&FederationMember> {
        match &self.config.routing_strategy {
            RoutingStrategy::Broadcast => self.config.members.iter().collect(),

            RoutingStrategy::Nearest => {
                // Sort by measured latency
                let mut members: Vec<_> = self.config.members.iter().collect();
                members.sort_by_key(|m| self.health.instances[&m.id].avg_latency);
                members.into_iter().take(self.config.max_concurrent_queries).collect()
            }

            RoutingStrategy::HashRouted => {
                let hash = hash_query(query);
                let idx = hash % self.config.members.len();
                vec![&self.config.members[idx]]
            }

            // ... other strategies
        }
    }
}
```

### Partial Failure Handling

```rust
async fn federated_search(&self, query: &Query) -> Result<FederatedSearchResponse> {
    let instances = self.select_instances(query);
    let futures: Vec<_> = instances
        .iter()
        .map(|inst| self.query_instance(inst, query))
        .collect();

    // Wait for all with timeout
    let results = timeout(self.config.query_timeout, join_all(futures)).await;

    // Collect successful results, track failures
    let (successes, failures): (Vec<_>, Vec<_>) = results
        .into_iter()
        .partition(|r| r.is_ok());

    // Merge available results
    let merged = self.merge_results(successes)?;

    Ok(FederatedSearchResponse {
        results: merged,
        failed_instances: failures.into_iter().map(|f| f.instance_id).collect(),
        // ...
    })
}
```
