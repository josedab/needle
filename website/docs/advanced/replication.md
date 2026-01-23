---
sidebar_position: 5
---

# Replication

Needle supports replication for high availability and read scaling. This guide covers replication patterns and configuration.

## Replication Patterns

### Primary-Replica

One primary handles writes, replicas serve reads:

```
┌─────────────┐
│   Primary   │ ──── Writes
└──────┬──────┘
       │ Replication
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌──────┐
│Rep 1 │ │Rep 2 │ ──── Reads
└──────┘ └──────┘
```

### Multi-Primary (Raft)

All nodes can handle writes using Raft consensus:

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Node 1  │◄───►│ Node 2  │◄───►│ Node 3  │
│(Leader) │     │(Follower│     │(Follower│
└─────────┘     └─────────┘     └─────────┘
     ▲               ▲               ▲
     └───────────────┼───────────────┘
                     │
              Client Writes
```

## Primary-Replica Setup

### Primary Node

```rust
use needle::{Database, replication::PrimaryNode};

let db = Database::open("primary.needle")?;
let primary = PrimaryNode::new(db, "0.0.0.0:5000")?;

// Start replication server
primary.start().await?;
```

### Replica Node

```rust
use needle::replication::ReplicaNode;

let replica = ReplicaNode::new("replica.needle")?;

// Connect to primary
replica.connect("primary-host:5000").await?;

// Start sync
replica.start_sync().await?;
```

### Configuring Replication

```rust
use needle::replication::ReplicationConfig;

let config = ReplicationConfig {
    sync_interval_ms: 100,      // Sync every 100ms
    batch_size: 1000,           // Sync 1000 changes at a time
    compression: true,          // Compress replication data
    tls: Some(TlsConfig {       // Enable TLS
        cert_path: "cert.pem",
        key_path: "key.pem",
    }),
};

let primary = PrimaryNode::with_config(db, "0.0.0.0:5000", config)?;
```

## Raft Consensus

### Setting Up a Raft Cluster

```rust
use needle::replication::raft::{RaftNode, RaftConfig};

let config = RaftConfig {
    node_id: 1,
    peers: vec![
        "node2.example.com:5000".parse()?,
        "node3.example.com:5000".parse()?,
    ],
    election_timeout_ms: 150..300,
    heartbeat_interval_ms: 50,
};

let node = RaftNode::new("node1.needle", config)?;
node.start().await?;
```

### Cluster Operations

```rust
// Check if this node is the leader
if node.is_leader() {
    println!("I am the leader");
}

// Get current leader
let leader = node.current_leader();

// Force election (for testing)
node.step_down().await?;
```

### Client Usage

```rust
use needle::replication::raft::RaftClient;

let client = RaftClient::new(vec![
    "node1.example.com:5000",
    "node2.example.com:5000",
    "node3.example.com:5000",
])?;

// Writes go to leader
client.insert("documents", "doc1", &vector, metadata).await?;

// Reads can go to any node
let results = client.search("documents", &query, 10, None).await?;
```

## Read Scaling

### Load Balancing Reads

```rust
use needle::replication::LoadBalancer;

let lb = LoadBalancer::new(vec![
    "replica1.example.com:8080",
    "replica2.example.com:8080",
    "replica3.example.com:8080",
]);

// Round-robin reads
let results = lb.search("documents", &query, 10, None).await?;
```

### Read Preferences

```rust
use needle::replication::ReadPreference;

// Read from primary only
let results = client.search_with_preference(
    "documents", &query, 10, None,
    ReadPreference::Primary
).await?;

// Read from nearest replica
let results = client.search_with_preference(
    "documents", &query, 10, None,
    ReadPreference::Nearest
).await?;

// Read from any replica (fastest)
let results = client.search_with_preference(
    "documents", &query, 10, None,
    ReadPreference::Secondary
).await?;
```

## Consistency Levels

### Write Concerns

```rust
use needle::replication::WriteConcern;

// Write acknowledged by primary only (fastest)
client.insert_with_concern(
    "documents", "doc1", &vector, metadata,
    WriteConcern::Primary
).await?;

// Write replicated to majority (default)
client.insert_with_concern(
    "documents", "doc1", &vector, metadata,
    WriteConcern::Majority
).await?;

// Write replicated to all nodes (slowest, safest)
client.insert_with_concern(
    "documents", "doc1", &vector, metadata,
    WriteConcern::All
).await?;
```

### Read Consistency

```rust
use needle::replication::ReadConsistency;

// Linearizable read (consistent but slower)
let results = client.search_with_consistency(
    "documents", &query, 10, None,
    ReadConsistency::Linearizable
).await?;

// Eventually consistent (fast but may be stale)
let results = client.search_with_consistency(
    "documents", &query, 10, None,
    ReadConsistency::Eventual
).await?;
```

## Failover

### Automatic Failover

With Raft, failover is automatic:

```rust
// Configure failover behavior
let config = RaftConfig {
    // ...
    election_timeout_ms: 150..300,  // Detect failures within 300ms
    // ...
};

// Node failures are detected and new leader elected automatically
```

### Manual Failover

```rust
// Promote replica to primary
replica.promote_to_primary().await?;

// Demote old primary to replica
old_primary.demote_to_replica("new-primary:5000").await?;
```

## Monitoring Replication

### Replication Lag

```rust
let lag = replica.replication_lag()?;
println!("Lag: {} operations, {} ms", lag.operations, lag.time_ms);

// Alert if lag is too high
if lag.time_ms > 1000 {
    alert("High replication lag!");
}
```

### Cluster Status

```rust
let status = node.cluster_status().await?;

println!("Leader: {}", status.leader_id);
println!("Term: {}", status.term);
println!("Committed index: {}", status.commit_index);

for peer in status.peers {
    println!("  {} - {} ({} behind)",
        peer.id, peer.state, peer.lag);
}
```

### Prometheus Metrics

```
needle_replication_lag_seconds
needle_replication_operations_total
needle_raft_term
needle_raft_leader{node_id="1"}
needle_raft_commit_index
```

## File-Based Replication

For simpler setups, replicate by copying the database file:

### Using rsync

```bash
#!/bin/bash
while true; do
    # Sync database file
    rsync -avz primary:/data/vectors.needle /data/vectors.needle

    # Reload replica
    curl -X POST http://localhost:8080/reload

    sleep 60
done
```

### Using Litestream

```yaml
# litestream.yml
dbs:
  - path: /data/vectors.needle
    replicas:
      - url: s3://bucket/needle/vectors.needle
```

## Best Practices

### 1. Use Odd Number of Nodes for Raft

3 nodes tolerates 1 failure, 5 nodes tolerate 2 failures.

### 2. Monitor Replication Lag

Alert when lag exceeds acceptable threshold.

### 3. Test Failover

Regularly test failover procedures in non-production.

### 4. Consider Network Partitions

Design for split-brain scenarios—Raft handles this correctly.

### 5. Backup Independently

Replication is not backup—maintain separate backup procedures.

## Next Steps

- [Sharding](/docs/advanced/sharding)
- [Production Deployment](/docs/guides/production)
- [API Reference](/docs/api-reference)
