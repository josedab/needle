# ADR-0033: Consistent Hashing for Horizontal Sharding

## Status

Accepted

## Context

As vector collections grow beyond single-node capacity, horizontal scaling becomes necessary:

1. **Data volume** — Billions of vectors exceed single-machine memory
2. **Query throughput** — Single node can't handle thousands of QPS
3. **Availability** — Single node is a single point of failure
4. **Geographic distribution** — Low latency requires data near users

Sharding approaches considered:

| Approach | Pros | Cons |
|----------|------|------|
| Range-based | Efficient range queries | Hot spots, rebalancing complexity |
| Hash-based | Even distribution | No range queries, full reshard on resize |
| Consistent hashing | Minimal rebalancing | Slightly uneven distribution |
| Directory-based | Full control | Central lookup bottleneck |

## Decision

Implement **consistent hashing** with **virtual nodes** for shard assignment, combined with a **state machine** for safe shard lifecycle management.

### Consistent Hashing

```rust
pub struct ConsistentHashRing {
    ring: BTreeMap<u64, ShardId>,
    virtual_nodes_per_shard: usize,  // Default: 150
}

impl ConsistentHashRing {
    /// Route a key to its shard
    pub fn route(&self, key: &str) -> ShardId {
        let hash = self.hash(key);
        // Find first node >= hash, or wrap to first node
        self.ring.range(hash..).next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, shard)| *shard)
            .expect("ring should not be empty")
    }

    /// Add a shard (creates virtual nodes)
    pub fn add_shard(&mut self, shard_id: ShardId) {
        for i in 0..self.virtual_nodes_per_shard {
            let vnode_key = format!("{}:{}", shard_id.0, i);
            let hash = self.hash(&vnode_key);
            self.ring.insert(hash, shard_id);
        }
    }

    /// Remove a shard (only affects ~1/N of keys)
    pub fn remove_shard(&mut self, shard_id: ShardId) {
        self.ring.retain(|_, s| *s != shard_id);
    }
}
```

### Shard State Machine

```
              ┌──────────────────────────────────────┐
              │                                      │
              ▼                                      │
         ┌────────┐    start_migration    ┌─────────┴───┐
         │ Active │ ───────────────────►  │  ReadOnly   │
         └────────┘                       └──────┬──────┘
              ▲                                  │
              │                                  │ data_transferred
              │         ┌───────────┐            │
              │         │ Migrating │ ◄──────────┘
              │         └─────┬─────┘
              │               │ migration_complete
              │               ▼
              │         ┌───────────┐
              └─────────│  Active   │ (on new node)
                        └───────────┘

        Error path: Any state ───► Offline
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardState {
    Active,     // Accepting reads and writes
    ReadOnly,   // Reads only (preparing for migration)
    Migrating,  // Data being transferred
    Offline,    // Not available
}

pub struct Shard {
    pub id: ShardId,
    pub state: ShardState,
    pub vector_count: AtomicU64,
    pub key_range: Option<(u64, u64)>,  // For range-aware routing
}
```

### ShardManager

```rust
pub struct ShardManager {
    shards: HashMap<ShardId, Shard>,
    ring: ConsistentHashRing,
    config: ShardConfig,
}

impl ShardManager {
    /// Route a vector ID to its shard
    pub fn route_id(&self, id: &str) -> ShardId {
        self.ring.route(id)
    }

    /// Scatter-gather search across all shards
    pub fn search_all(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> ShardResult<Vec<SearchResult>> {
        let shard_results: Vec<Vec<SearchResult>> = self.shards
            .par_iter()
            .filter(|(_, s)| s.state != ShardState::Offline)
            .map(|(_, shard)| shard.search(query, k, filter))
            .collect::<Result<_, _>>()?;

        // Merge and re-rank
        Ok(merge_results(shard_results, k))
    }

    /// Add a new shard (triggers rebalancing)
    pub fn add_shard(&mut self) -> ShardResult<ShardId> {
        let new_id = ShardId::new(self.shards.len() as u32);

        // Add to ring (consistent hashing minimizes movement)
        self.ring.add_shard(new_id);

        // Create shard in ReadOnly state initially
        self.shards.insert(new_id, Shard {
            id: new_id,
            state: ShardState::Active,
            vector_count: AtomicU64::new(0),
            key_range: None,
        });

        // Schedule background rebalancing
        self.schedule_rebalance()?;

        Ok(new_id)
    }
}
```

### Code References

- `src/shard.rs:47-66` — ShardError types
- `src/shard.rs:68-88` — ShardId definition
- `src/shard.rs:90-100` — ShardState enum
- `src/shard.rs` — ShardManager implementation

## Consequences

### Benefits

1. **Minimal rebalancing** — Adding/removing shards moves only ~1/N of data
2. **Even distribution** — Virtual nodes smooth out hash function irregularities
3. **Safe migrations** — State machine prevents reads/writes to migrating shards
4. **Parallel queries** — Scatter-gather leverages all shards simultaneously
5. **No central coordinator** — Any node can compute routing locally

### Tradeoffs

1. **No range queries** — Hash-based routing destroys key ordering
2. **All-shard queries** — Search must touch all shards (can't prune)
3. **Result merging overhead** — Must re-rank results from multiple shards
4. **Virtual node memory** — 150 vnodes × N shards entries in ring

### What This Enabled

- Horizontal scaling beyond single-node limits
- Rolling upgrades (migrate shard, upgrade, migrate back)
- Geographic sharding (route by region hash)
- Gradual capacity expansion (add shards one at a time)

### What This Prevented

- Full data reshuffling when adding nodes
- Split-brain during shard migrations
- Writes to shards mid-migration
- Single-node scaling ceiling

### Shard Count Guidelines

| Vector Count | Recommended Shards | Notes |
|--------------|-------------------|-------|
| < 10M | 1 | Single node sufficient |
| 10M - 100M | 2-4 | Parallel query benefit |
| 100M - 1B | 4-16 | Memory distribution |
| > 1B | 16+ | Scale linearly |

### Rebalancing Process

```
1. Mark source shard as ReadOnly
2. Create target shard (or use existing with capacity)
3. Stream vectors from source to target
4. Update routing ring atomically
5. Mark source as Active (minus migrated keys)
6. Garbage collect migrated vectors from source
```

### Query Flow

```
Client Query
     │
     ▼
┌─────────────┐
│ ShardRouter │ ─── Determines all active shards
└──────┬──────┘
       │
       │ Parallel fan-out
       ▼
┌──────┴──────┬──────────────┬──────────────┐
▼             ▼              ▼              ▼
Shard 0    Shard 1       Shard 2       Shard N
   │          │              │              │
   └──────────┴──────────────┴──────────────┘
                      │
                      ▼ Merge & re-rank
                 ┌─────────┐
                 │ Top-K   │
                 └─────────┘
```
