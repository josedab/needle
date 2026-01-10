# Distributed Operations Guide

This guide covers deploying and operating Needle in distributed configurations including Raft replication, horizontal sharding, and combined architectures.

## Overview

Needle supports two distributed patterns:

1. **Raft Replication**: Strong consistency with automatic leader election (read replicas)
2. **Sharding**: Horizontal partitioning across nodes (scale writes and storage)

These can be combined for both high availability and horizontal scaling.

```
                    ┌─────────────────────────────────────────┐
                    │              Query Router               │
                    │         (Load Balancer / SDK)           │
                    └───────────────────┬─────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
       ┌──────▼──────┐          ┌───────▼──────┐          ┌───────▼──────┐
       │   Shard 0   │          │   Shard 1    │          │   Shard 2    │
       │   (Raft)    │          │   (Raft)     │          │   (Raft)     │
       └──────┬──────┘          └───────┬──────┘          └───────┬──────┘
              │                         │                         │
    ┌─────────┼─────────┐     ┌─────────┼─────────┐     ┌─────────┼─────────┐
    │         │         │     │         │         │     │         │         │
  ┌─▼─┐     ┌─▼─┐     ┌─▼─┐ ┌─▼─┐     ┌─▼─┐     ┌─▼─┐ ┌─▼─┐     ┌─▼─┐     ┌─▼─┐
  │ L │     │ F │     │ F │ │ L │     │ F │     │ F │ │ L │     │ F │     │ F │
  └───┘     └───┘     └───┘ └───┘     └───┘     └───┘ └───┘     └───┘     └───┘
  Leader   Follower  Follower

  L = Leader (accepts writes)
  F = Follower (read replicas)
```

---

## Raft Replication

### Concepts

- **Leader**: Single node that accepts all writes
- **Followers**: Replicas that receive replicated writes
- **Quorum**: Majority of nodes must agree for commits
- **Term**: Logical clock for leader elections

### Cluster Topology

| Nodes | Fault Tolerance | Quorum | Recommended For |
|-------|-----------------|--------|-----------------|
| 3 | 1 failure | 2 | Development, small production |
| 5 | 2 failures | 3 | Production |
| 7 | 3 failures | 4 | High availability |

### Configuration

```rust
use needle::distributed::{RaftConfig, RaftCluster};

let config = RaftConfig::new()
    .with_node_id("node-1")
    .with_cluster_nodes(vec![
        "node-1:5000".to_string(),
        "node-2:5000".to_string(),
        "node-3:5000".to_string(),
    ])
    .with_election_timeout(Duration::from_millis(150))
    .with_heartbeat_interval(Duration::from_millis(50))
    .with_snapshot_threshold(10000);

let cluster = RaftCluster::new(config)?;
cluster.start().await?;
```

### Leader Election

Election occurs when:
1. Cluster starts (no leader exists)
2. Leader fails (heartbeat timeout)
3. Leader steps down (graceful)

```rust
// Check current leader
let leader = cluster.leader().await;
println!("Current leader: {:?}", leader);

// Gracefully step down (for maintenance)
if cluster.is_leader() {
    cluster.step_down().await?;
}
```

### Write Path

```
Client → Leader → Log Append → Replicate to Followers → Wait for Quorum → Commit → Apply → Respond
```

1. Client sends write to leader
2. Leader appends to local log
3. Leader replicates to followers
4. Wait for quorum acknowledgment
5. Commit and apply to state machine
6. Respond to client

### Read Path (Options)

| Mode | Consistency | Latency | Use Case |
|------|-------------|---------|----------|
| Leader | Strong | Higher | Financial, compliance |
| Follower | Eventual | Lower | Analytics, search |
| Leased | Strong (with lease) | Medium | General purpose |

```rust
// Strong read (always from leader)
let results = cluster.search_strong(&query, 10).await?;

// Eventual read (any node)
let results = cluster.search_eventual(&query, 10).await?;

// Leased read (strong within lease period)
let results = cluster.search_leased(&query, 10).await?;
```

### Monitoring Raft

```rust
let status = cluster.status().await;

println!("State: {:?}", status.state);           // Leader, Follower, Candidate
println!("Term: {}", status.current_term);
println!("Commit Index: {}", status.commit_index);
println!("Last Applied: {}", status.last_applied);
println!("Peers: {:?}", status.peers);
```

Key metrics to monitor:
- `raft_term`: Current election term
- `raft_commit_index`: Highest committed log index
- `raft_apply_index`: Highest applied log index
- `raft_leader_changes`: Leadership changes (should be infrequent)
- `raft_replication_lag`: Follower lag behind leader

---

## Horizontal Sharding

### Concepts

- **Shard**: A partition of the dataset
- **Shard Key**: Determines which shard owns a vector
- **Consistent Hashing**: Minimizes data movement when adding/removing shards
- **Virtual Nodes**: Improve balance across physical nodes

### Sharding Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| Hash | Hash(vector_id) % shards | Uniform distribution |
| Range | ID ranges per shard | Sequential IDs |
| Custom | User-defined routing | Domain-specific |

### Configuration

```rust
use needle::distributed::{ShardConfig, ShardManager};

let config = ShardConfig::new()
    .with_shard_count(8)
    .with_replication_factor(3)
    .with_virtual_nodes_per_shard(100)
    .with_strategy(ShardStrategy::ConsistentHash);

let manager = ShardManager::new(config)?;

// Add shard nodes
manager.add_node("shard-0", "host1:5000").await?;
manager.add_node("shard-1", "host2:5000").await?;
manager.add_node("shard-2", "host3:5000").await?;
```

### Consistent Hash Ring

```
                      ┌───────────────────────┐
                     ╱                         ╲
                   ╱                             ╲
                 ╱     ┌───┐                      ╲
                │     │S0 │                        │
                │      └───┘                       │
               │                    ┌───┐          │
               │                    │S1 │          │
               │        Hash        └───┘          │
               │        Ring                       │
               │                        ┌───┐      │
                │                       │S2 │     │
                │                       └───┘     │
                 │     ┌───┐                     │
                  ╲    │S3 │                   ╱
                   ╲   └───┘                 ╱
                     ╲                     ╱
                       └───────────────────┘

        vector_id → hash → closest shard clockwise
```

### Insert Operation

```rust
// Router automatically selects shard
let router = ShardRouter::new(&manager);

// Insert routes to correct shard
router.insert("doc-123", &vector, metadata).await?;

// Batch insert with automatic routing
let batch = vec![
    ("doc-1", vec1, meta1),
    ("doc-2", vec2, meta2),
    // ...
];
router.insert_batch(&batch).await?;
```

### Search Operation

Searches are fanned out to all shards and results merged:

```rust
// Search across all shards
let results = router.search(&query, 10).await?;

// Search with filter (may skip shards based on metadata)
let filter = Filter::eq("region", "us-east");
let results = router.search_filtered(&query, 10, &filter).await?;
```

Search flow:
1. Query router receives search request
2. Fan-out to all shards (parallel)
3. Each shard returns top-K results
4. Merge results by distance
5. Return global top-K

---

## Combined Architecture (Sharded + Replicated)

For both horizontal scaling and high availability:

```rust
use needle::distributed::{ClusterConfig, DistributedCluster};

let config = ClusterConfig::new()
    // Sharding config
    .with_shard_count(4)
    .with_strategy(ShardStrategy::ConsistentHash)
    // Replication config (per shard)
    .with_replication_factor(3)
    .with_raft_enabled(true);

let cluster = DistributedCluster::new(config)?;

// Add nodes (automatically assigned to shards)
cluster.add_node("node-1", "host1:5000").await?;
cluster.add_node("node-2", "host2:5000").await?;
// ... add 12 nodes total for 4 shards × 3 replicas
```

### Node Assignment

With 4 shards and replication factor 3, you need 12 nodes:

| Shard | Leader | Follower 1 | Follower 2 |
|-------|--------|------------|------------|
| 0 | node-1 | node-2 | node-3 |
| 1 | node-4 | node-5 | node-6 |
| 2 | node-7 | node-8 | node-9 |
| 3 | node-10 | node-11 | node-12 |

---

## Rebalancing

### When to Rebalance

- Adding new shards
- Removing shards
- Uneven data distribution
- Node capacity changes

### Rebalancing Process

```rust
use needle::distributed::RebalanceCoordinator;

let coordinator = RebalanceCoordinator::new(&cluster);

// Check current balance
let stats = coordinator.get_stats().await?;
println!("Shard sizes: {:?}", stats.shard_sizes);
println!("Imbalance ratio: {:.2}", stats.imbalance_ratio);

// Start rebalancing (throttled to minimize impact)
let config = RebalanceConfig::new()
    .with_max_concurrent_moves(10)
    .with_batch_size(1000)
    .with_throttle_rate(Duration::from_millis(100));

let handle = coordinator.start_rebalance(config).await?;

// Monitor progress
while !handle.is_complete() {
    let progress = handle.progress().await?;
    println!("Progress: {:.1}%", progress.percent_complete);
    tokio::time::sleep(Duration::from_secs(10)).await;
}
```

### Zero-Downtime Rebalancing

1. **Preparation**: Mark target ranges as "migrating"
2. **Copy**: Copy data to new shard (reads still work)
3. **Catch-up**: Replicate changes during copy
4. **Switch**: Atomic metadata update
5. **Cleanup**: Remove old data

---

## Monitoring Distributed Clusters

### Key Metrics

```yaml
# Cluster health
needle_cluster_healthy: 1  # 1 = healthy, 0 = degraded

# Per-shard metrics
needle_shard_vector_count{shard="0"}: 1000000
needle_shard_leader{shard="0"}: "node-1"
needle_shard_followers{shard="0"}: 2

# Replication lag
needle_replication_lag_ms{shard="0", follower="node-2"}: 50

# Cross-shard queries
needle_search_fanout_count: 4  # Number of shards queried
needle_search_merge_time_ms: 5
```

### Health Checks

```rust
// Check cluster health
let health = cluster.health_check().await?;

for (shard_id, status) in &health.shards {
    println!("Shard {}: {:?}", shard_id, status.state);
    println!("  Leader: {:?}", status.leader);
    println!("  Followers: {}/{}", status.healthy_followers, status.total_followers);
}
```

### Alerting Rules

| Condition | Severity | Action |
|-----------|----------|--------|
| Leader election in progress | Warning | Monitor |
| Shard has no leader | Critical | Investigate immediately |
| Replication lag > 1s | Warning | Check network/load |
| Replication lag > 10s | Critical | May lose data on failure |
| Shard imbalance > 20% | Warning | Schedule rebalance |
| Node unreachable | Critical | Check node health |

---

## Troubleshooting

### Split Brain

**Symptoms**: Multiple leaders, inconsistent reads

**Causes**:
- Network partition
- Incorrect cluster configuration
- Clock skew

**Resolution**:
1. Identify which partition has quorum
2. Restart nodes in minority partition
3. Verify network connectivity
4. Check NTP synchronization

```rust
// Force leader step-down if split brain suspected
cluster.force_new_election().await?;
```

### Election Failures

**Symptoms**: No leader elected, cluster unavailable

**Causes**:
- Too few nodes online
- Network issues
- Election timeout too short

**Resolution**:
1. Verify quorum of nodes are healthy
2. Check network connectivity between nodes
3. Increase election timeout if network is slow

```rust
// Adjust timeouts
let config = RaftConfig::new()
    .with_election_timeout(Duration::from_millis(500))  // Increase
    .with_heartbeat_interval(Duration::from_millis(100));
```

### Slow Replication

**Symptoms**: High follower lag, slow writes

**Causes**:
- Network bandwidth saturated
- Slow disk I/O on followers
- Large batch writes

**Resolution**:
1. Check network bandwidth
2. Upgrade disk performance
3. Batch write throttling
4. Add more followers to distribute load

### Data Skew

**Symptoms**: Some shards much larger than others

**Causes**:
- Non-uniform key distribution
- Hot keys
- Incorrect shard key choice

**Resolution**:
1. Review shard key selection
2. Increase virtual nodes
3. Manual rebalancing
4. Consider composite shard keys

---

## Best Practices

### Cluster Sizing

| Dataset Size | Recommended Shards | Nodes (RF=3) |
|--------------|-------------------|--------------|
| < 10M vectors | 1 | 3 |
| 10M - 100M | 2-4 | 6-12 |
| 100M - 1B | 8-16 | 24-48 |
| > 1B | 16+ | 48+ |

### Network Configuration

- Place all cluster nodes in same datacenter/region
- Use dedicated network for Raft communication
- Ensure low-latency (<10ms) between nodes
- Configure appropriate timeouts for your network

### Maintenance Windows

```rust
// Graceful maintenance mode
cluster.enter_maintenance_mode().await?;

// Perform maintenance...

cluster.exit_maintenance_mode().await?;
```

For rolling upgrades:
1. Start with followers
2. Upgrade one node at a time
3. Wait for replication to catch up
4. Step down leader last

---

## See Also

- [Production Checklist](production-checklist.md) - Pre-deployment verification
- [Architecture](architecture.md) - Distributed architecture diagrams
- [Operations Guide](OPERATIONS.md) - Day-to-day operations
- [Index Selection Guide](index-selection-guide.md) - Choosing index types for distributed deployments
