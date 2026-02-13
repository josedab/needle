---
sidebar_position: 4
---

# Sharding

For datasets too large for a single machine, Needle supports horizontal sharding to distribute vectors across multiple database files or servers.

## Sharding Strategies

### Hash-Based Sharding

Distributes vectors evenly using consistent hashing:

```rust
use needle::sharding::{ShardManager, HashShardingStrategy};

// Create shard manager with 4 shards
let strategy = HashShardingStrategy::new(4);
let manager = ShardManager::new(strategy);

// Add shard databases
manager.add_shard(0, Database::open("shard_0.needle")?);
manager.add_shard(1, Database::open("shard_1.needle")?);
manager.add_shard(2, Database::open("shard_2.needle")?);
manager.add_shard(3, Database::open("shard_3.needle")?);

// Insert - automatically routes to correct shard
manager.insert("documents", "doc1", &vector, metadata)?;

// Search - queries all shards and merges results
let results = manager.search("documents", &query, 10, None)?;
```

### Range-Based Sharding

Shards based on vector ID ranges:

```rust
use needle::sharding::{ShardManager, RangeShardingStrategy};

// Define ranges
let ranges = vec![
    ("a", "m"),  // Shard 0: IDs starting a-l
    ("m", "{"),  // Shard 1: IDs starting m-z
];
let strategy = RangeShardingStrategy::new(ranges);
let manager = ShardManager::new(strategy);
```

### Metadata-Based Sharding

Shard by metadata field (e.g., tenant ID):

```rust
use needle::sharding::{ShardManager, MetadataShardingStrategy};

// Shard by tenant_id field
let strategy = MetadataShardingStrategy::new("tenant_id");
let manager = ShardManager::new(strategy);

// Each tenant's vectors go to their shard
manager.insert("docs", "doc1", &vector, json!({"tenant_id": "acme"}))?;
```

## Shard Management

### Adding Shards

```rust
// Add a new shard
let new_shard = Database::open("shard_4.needle")?;
manager.add_shard(4, new_shard);

// Shard must have the same collections
new_shard.create_collection("documents", 384)?;
```

### Rebalancing

When adding or removing shards, vectors need to be rebalanced:

```rust
use needle::sharding::RebalanceConfig;

let config = RebalanceConfig {
    batch_size: 10000,
    parallel: true,
};

// Rebalance redistributes vectors to new shard layout
manager.rebalance(config).await?;
```

### Removing Shards

```rust
// Remove shard (vectors are migrated first)
manager.remove_shard(3).await?;
```

## Distributed Search

### Parallel Search

Queries all shards in parallel:

```rust
// Search across all shards
let results = manager.search("documents", &query, 10, None)?;

// Results are merged and sorted by distance
for result in results {
    println!("ID: {}, Distance: {}, Shard: {}", result.id, result.distance, result.shard);
}
```

### Filtered Search

Filters are applied on each shard:

```rust
let filter = Filter::parse(&json!({"category": "programming"}))?;
let results = manager.search("documents", &query, 10, Some(&filter))?;
```

### Search with Routing

If you know which shard contains the data:

```rust
// Direct search to specific shard
let results = manager.search_shard(2, "documents", &query, 10, None)?;
```

## Shard Configuration

### Consistent Hashing

Uses virtual nodes for even distribution:

```rust
let strategy = HashShardingStrategy::new(4)
    .with_virtual_nodes(150);  // 150 virtual nodes per shard
```

### Replication Factor

Each vector can be stored on multiple shards:

```rust
let strategy = HashShardingStrategy::new(4)
    .with_replication_factor(2);  // Store on 2 shards
```

## Distributed Architecture

### Single-Node Multi-Shard

Multiple shard files on one machine:

```
┌─────────────────────────────────────────┐
│              Application                │
│                   │                     │
│            ShardManager                 │
│         ┌────┬────┬────┬────┐           │
│         │    │    │    │    │           │
│      shard0 shard1 shard2 shard3        │
│      .needle .needle .needle .needle    │
└─────────────────────────────────────────┘
```

### Multi-Node Cluster

Shards distributed across servers:

```
┌────────────────┐  ┌────────────────┐
│   Node 1       │  │   Node 2       │
│  ┌──────────┐  │  │  ┌──────────┐  │
│  │ Shard 0  │  │  │  │ Shard 2  │  │
│  │ Shard 1  │  │  │  │ Shard 3  │  │
│  └──────────┘  │  │  └──────────┘  │
└────────────────┘  └────────────────┘
        │                   │
        └───────┬───────────┘
                │
        ┌───────────────┐
        │  Coordinator  │
        │  (any node)   │
        └───────────────┘
```

### Coordinator Pattern

```rust
use needle::sharding::{RemoteShard, Coordinator};

// Create coordinator
let coordinator = Coordinator::new();

// Add remote shards
coordinator.add_shard(RemoteShard::new("http://node1:8080", vec![0, 1]));
coordinator.add_shard(RemoteShard::new("http://node2:8080", vec![2, 3]));

// Distributed search
let results = coordinator.search("documents", &query, 10, None).await?;
```

## Performance Considerations

### Latency

Distributed search adds network latency:

| Setup | Typical Latency |
|-------|-----------------|
| Single shard | 5ms |
| 4 local shards | 8ms |
| 4 remote shards | 15-50ms |

### Throughput

Sharding increases throughput:

```
Single shard: ~10,000 QPS
4 shards: ~35,000 QPS (not quite 4x due to coordination)
```

### Memory

Each shard uses less memory:

```
1M vectors, 384 dims
Single shard: 1.5 GB
4 shards: 375 MB each
```

## Migration to Sharding

### From Single Database

```rust
use needle::sharding::migrate_to_shards;

// Create shards
let shards: Vec<Database> = (0..4)
    .map(|i| Database::create(&format!("shard_{}.needle", i)).unwrap())
    .collect();

// Create collections on each shard
for shard in &shards {
    shard.create_collection("documents", 384)?;
}

// Migrate data
let source = Database::open("original.needle")?;
let strategy = HashShardingStrategy::new(4);

migrate_to_shards(&source, &shards, &strategy)?;
```

## Best Practices

### 1. Choose Shard Count Wisely

```
Vectors per shard: 1-10 million ideal
Too few shards: Memory constraints
Too many shards: Coordination overhead
```

### 2. Use Consistent Hashing

Minimizes data movement when adding/removing shards.

### 3. Consider Access Patterns

If queries always filter by a specific field (e.g., tenant), shard by that field for efficiency.

### 4. Monitor Shard Balance

```rust
// Check shard distribution
for (shard_id, db) in manager.shards() {
    let count = db.collection("documents")?.count()?;
    println!("Shard {}: {} vectors", shard_id, count);
}
```

### 5. Plan for Growth

Start with more shards than needed—adding shards requires rebalancing.

## Next Steps

- [Replication](/docs/advanced/replication)
- [Production Deployment](/docs/guides/production)
- [API Reference](/docs/api-reference)
