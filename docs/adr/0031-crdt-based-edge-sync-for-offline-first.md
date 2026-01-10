# ADR-0031: CRDT-Based Edge Sync for Offline-First

## Status

Accepted

## Context

Edge computing and mobile applications require vector databases that work offline and sync when connectivity returns. Traditional approaches face challenges:

1. **Network partitions** — Edge devices frequently lose connectivity
2. **Conflict resolution** — Multiple devices may modify the same data
3. **Causality preservation** — Operations must be applied in correct order
4. **Bandwidth efficiency** — Full sync is expensive; incremental is preferred
5. **Coordination overhead** — Consensus protocols require connectivity

Approaches considered:

| Approach | Offline Support | Conflict Handling | Complexity |
|----------|-----------------|-------------------|------------|
| Last-write-wins | Yes | Data loss | Simple |
| Operational Transform | Yes | Complex merging | High |
| CRDTs | Yes | Automatic, lossless | Medium |
| Raft consensus | No (needs quorum) | None (leader decides) | High |

## Decision

Implement **Conflict-free Replicated Data Types (CRDTs)** with **Hybrid Logical Clocks (HLC)** for offline-first vector synchronization.

### Hybrid Logical Clock

HLCs combine physical timestamps with logical counters to provide:
- Causal ordering without coordination
- Bounded clock drift tolerance
- Unique event identification

```rust
pub struct HLC {
    pub physical: u64,  // Wall clock time (milliseconds)
    pub logical: u32,   // Logical counter for same-millisecond events
    pub replica: u64,   // Replica ID for total ordering
}

impl HLC {
    /// Tick for local event
    pub fn tick(&mut self) -> HLC {
        let now = Self::now();
        if now > self.physical {
            self.physical = now;
            self.logical = 0;
        } else {
            self.logical += 1;
        }
        *self
    }

    /// Merge with remote clock (receive event)
    pub fn merge(&mut self, remote: &HLC) -> HLC {
        let now = Self::now();
        if now > self.physical && now > remote.physical {
            self.physical = now;
            self.logical = 0;
        } else if self.physical > remote.physical {
            self.logical += 1;
        } else if remote.physical > self.physical {
            self.physical = remote.physical;
            self.logical = remote.logical + 1;
        } else {
            self.logical = self.logical.max(remote.logical) + 1;
        }
        *self
    }
}
```

### Vector CRDT Operations

```rust
pub enum VectorOperation {
    Add {
        id: String,
        vector: Vec<f32>,
        metadata: Value,
        timestamp: HLC,
    },
    Update {
        id: String,
        vector: Option<Vec<f32>>,
        metadata: Option<Value>,
        timestamp: HLC,
    },
    Delete {
        id: String,
        timestamp: HLC,  // Tombstone timestamp
    },
}
```

### Delta Synchronization

```rust
pub struct VectorCRDT {
    replica_id: ReplicaId,
    clock: HLC,
    vectors: HashMap<String, VectorEntry>,
    tombstones: HashMap<String, HLC>,  // Deleted IDs with deletion time
    version_vector: HashMap<ReplicaId, HLC>,  // Per-replica progress
}

impl VectorCRDT {
    /// Get operations since last sync with a specific replica
    pub fn delta_since(&self, version: &VersionVector) -> Delta {
        let ops: Vec<VectorOperation> = self.vectors.iter()
            .filter(|(_, entry)| !version.has_seen(&entry.timestamp))
            .map(|(id, entry)| entry.to_operation(id))
            .collect();

        Delta { operations: ops, version: self.version_vector.clone() }
    }

    /// Merge remote delta (idempotent, commutative, associative)
    pub fn merge(&mut self, delta: Delta) -> Result<MergeStats> {
        let mut stats = MergeStats::default();

        for op in delta.operations {
            match op {
                VectorOperation::Add { id, vector, metadata, timestamp } => {
                    if self.should_accept(&id, &timestamp) {
                        self.vectors.insert(id, VectorEntry { vector, metadata, timestamp });
                        stats.added += 1;
                    }
                }
                VectorOperation::Delete { id, timestamp } => {
                    if self.should_accept(&id, &timestamp) {
                        self.vectors.remove(&id);
                        self.tombstones.insert(id, timestamp);
                        stats.deleted += 1;
                    }
                }
                // ... Update handling
            }
        }

        self.version_vector.merge(&delta.version);
        Ok(stats)
    }
}
```

### Conflict Resolution Rules

1. **Add vs. Add** — Higher HLC timestamp wins (deterministic tie-breaker via replica ID)
2. **Update vs. Update** — Higher timestamp wins, field-level merge for metadata
3. **Delete vs. Add/Update** — Higher timestamp wins (delete can be undone by later add)
4. **Tombstone pruning** — Tombstones kept for configurable duration, then pruned

### Code References

- `src/crdt.rs:37-67` — ReplicaId and HLC implementation
- `src/crdt.rs:69-130` — HLC tick and merge operations
- `src/crdt.rs` — VectorCRDT with delta sync and merge

## Consequences

### Benefits

1. **True offline-first** — Full functionality without connectivity
2. **Automatic conflict resolution** — No manual merge required
3. **Convergence guarantee** — All replicas reach same state eventually
4. **Efficient sync** — Delta-based, only transfers changes
5. **Causal consistency** — Operations respect happened-before relationships

### Tradeoffs

1. **Tombstone overhead** — Deleted items leave markers until pruned
2. **No strong consistency** — Replicas may temporarily diverge
3. **Metadata growth** — Version vectors grow with replica count
4. **Complexity** — HLC and CRDT semantics require careful implementation

### What This Enabled

- Mobile apps that work in airplane mode
- Edge deployments with intermittent connectivity
- Multi-device sync without central coordinator
- Disaster recovery via replica divergence detection

### What This Prevented

- Requiring always-on connectivity for writes
- Manual conflict resolution UX
- Data loss from concurrent modifications
- Central point of failure for sync coordination

### Sync Patterns

**Hub-and-spoke:**
```
      ┌──────────┐
      │  Cloud   │
      │  Hub     │
      └────┬─────┘
     ┌─────┼─────┐
     ▼     ▼     ▼
  ┌────┐┌────┐┌────┐
  │Edge││Edge││Edge│
  └────┘└────┘└────┘
```

**Peer-to-peer:**
```
  ┌────┐◄───►┌────┐
  │Edge│     │Edge│
  └──┬─┘     └─┬──┘
     │    ▲    │
     └────┼────┘
          │
       ┌──▼──┐
       │Edge │
       └─────┘
```

Both patterns supported — CRDTs are topology-agnostic.
