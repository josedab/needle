# ADR-0010: Raft-Based Replication Layer

## Status

Accepted

## Context

While Needle excels as an embedded single-node database, production deployments often require:

1. **High availability** — Continue serving if one node fails
2. **Read scalability** — Distribute read load across replicas
3. **Durability** — Survive disk failures through replication
4. **Geographic distribution** — Serve users from nearby replicas

Several replication strategies were considered:

| Approach | Consistency | Complexity | Failover |
|----------|-------------|------------|----------|
| Primary-replica (async) | Eventual | Low | Manual |
| Primary-replica (sync) | Strong | Medium | Manual |
| Multi-primary | Conflict resolution needed | High | Automatic |
| **Raft consensus** | Strong (linearizable) | Medium | Automatic |
| Paxos | Strong | High | Automatic |
| CRDTs | Eventual (conflict-free) | Medium | Automatic |

## Decision

Layer **Raft-based replication** on top of the core Database as a composable wrapper, with pluggable networking via a MessageHandler trait.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ReplicatedDatabase                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  RaftNode   │  │  Database   │  │ MessageHandler  │  │
│  │  (consensus)│  │  (storage)  │  │   (network)     │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Replicated Commands

All write operations are serialized as commands and replicated through Raft:

```rust
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ReplicatedCommand {
    CreateCollection {
        name: String,
        dimensions: usize,
        config: Option<CollectionConfig>,
    },
    Insert {
        collection: String,
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    Update {
        collection: String,
        id: String,
        vector: Option<Vec<f32>>,
        metadata: Option<serde_json::Value>,
    },
    Delete {
        collection: String,
        id: String,
    },
    Compact {
        collection: String,
    },
    DropCollection {
        name: String,
    },
}
```

### Message Handler Abstraction

Networking is abstracted via a trait, allowing different transport implementations:

```rust
pub trait MessageHandler: Send + Sync {
    /// Send a Raft message to a peer node
    fn send(&self, to: NodeId, message: RaftMessage) -> Result<()>;

    /// Receive messages (called by the Raft loop)
    fn receive(&self) -> Result<Option<(NodeId, RaftMessage)>>;

    /// Get the list of known peers
    fn peers(&self) -> Vec<NodeId>;
}
```

This enables:
- gRPC transport for production
- TCP transport for simplicity
- In-memory transport for testing
- QUIC transport for edge deployments

### ReplicatedDatabase API

```rust
pub struct ReplicatedDatabase {
    raft: RaftNode,
    db: Database,
    handler: Box<dyn MessageHandler>,
    config: ReplicatedDatabaseConfig,
}

impl ReplicatedDatabase {
    /// Create a new replicated database (starts Raft node)
    pub fn new(
        node_id: NodeId,
        db: Database,
        handler: Box<dyn MessageHandler>,
        config: ReplicatedDatabaseConfig,
    ) -> Result<Self>;

    /// Insert (leader only, replicated)
    pub fn insert(&self, collection: &str, id: &str, vector: Vec<f32>) -> Result<()> {
        self.propose_and_wait(ReplicatedCommand::Insert {
            collection: collection.to_string(),
            id: id.to_string(),
            vector,
            metadata: None,
        })
    }

    /// Search (can run on any node if allow_follower_reads is true)
    pub fn search(&self, collection: &str, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.config.allow_follower_reads || self.raft.is_leader() {
            self.db.search(collection, query, k)
        } else {
            Err(NeedleError::NotLeader)
        }
    }
}
```

### Code References

- `src/database.rs:1677-2180` — ReplicatedDatabase implementation
- `src/database.rs:1689-1721` — ReplicatedCommand enum
- `src/database.rs:1842` — RaftNode composition
- `src/database.rs:1929-1938` — apply_committed() for state machine
- `src/database.rs:1849-1856` — MessageHandler trait

## Consequences

### Benefits

1. **Strong consistency** — Linearizable writes via Raft consensus
2. **Automatic failover** — Leader election on node failure
3. **Composable design** — Replication wraps core Database unchanged
4. **Pluggable networking** — Transport is an implementation detail
5. **Follower reads option** — Trade consistency for read scalability
6. **Familiar algorithm** — Raft is well-documented and battle-tested

### Tradeoffs

1. **Write latency** — Commits require majority acknowledgment
2. **Odd cluster sizes** — Need 3, 5, or 7 nodes for fault tolerance
3. **Network dependency** — Partition can block writes (CP system)
4. **Complexity** — Distributed systems are inherently complex
5. **No multi-region** — Single Raft group has latency constraints

### Consistency Guarantees

| Operation | Guarantee |
|-----------|-----------|
| Write (insert/update/delete) | Linearizable (through leader) |
| Read (leader) | Linearizable |
| Read (follower, allow_follower_reads=true) | Eventual (bounded staleness) |
| Read (follower, allow_follower_reads=false) | Rejected (NotLeader error) |

### What This Enabled

- High-availability deployments for production
- Read scaling via follower replicas
- Automatic recovery from single-node failures
- Consistent snapshots across cluster for backup

### What This Prevented

- Multi-region active-active (would need conflict resolution)
- Writes during network partition (Raft is CP, not AP)
- Sub-millisecond write latency (consensus overhead)
- Heterogeneous cluster (all nodes must run same version)

### Deployment Topology

**Minimum HA cluster (3 nodes):**
```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Node A  │────│ Node B  │────│ Node C  │
│ (Leader)│     │(Follower)│    │(Follower)│
└─────────┘     └─────────┘     └─────────┘
     │               │               │
     ▼               ▼               ▼
  writes         reads*          reads*

* if allow_follower_reads = true
```

**Tolerates 1 node failure.** For 2-node fault tolerance, use 5 nodes.

### Configuration Options

```rust
pub struct ReplicatedDatabaseConfig {
    /// Allow reads from followers (eventual consistency)
    pub allow_follower_reads: bool,

    /// Election timeout range (ms)
    pub election_timeout_min: u64,
    pub election_timeout_max: u64,

    /// Heartbeat interval (ms)
    pub heartbeat_interval: u64,

    /// Snapshot threshold (log entries before snapshot)
    pub snapshot_threshold: u64,
}
```

### Usage Example

```rust
use needle::{Database, ReplicatedDatabase, ReplicatedDatabaseConfig};
use my_transport::GrpcMessageHandler;

// Create local database
let db = Database::open("node1.needle")?;

// Create network handler
let handler = GrpcMessageHandler::new("node1:5000", vec!["node2:5000", "node3:5000"]);

// Wrap in replicated database
let config = ReplicatedDatabaseConfig {
    allow_follower_reads: true,
    ..Default::default()
};

let replicated = ReplicatedDatabase::new(1, db, Box::new(handler), config)?;

// Use normally - writes go through Raft
replicated.insert("embeddings", "doc1", embedding)?;

// Reads can go to any node
let results = replicated.search("embeddings", &query, 10)?;
```
