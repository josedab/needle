# ADR-0023: Dual Vector ID System

## Status

Accepted

## Context

Vector databases must identify vectors for operations like get, update, and delete. Two identification approaches exist:

### User-Facing Requirements
- **Meaningful identifiers** — Users want IDs like `"doc_12345"` or `"user_abc_profile"`
- **External system integration** — IDs often come from other databases (UUIDs, slugs)
- **Stability** — IDs should not change after insertion

### Internal Requirements
- **Efficient indexing** — HNSW graph uses integer indices for O(1) node lookup
- **Compact storage** — Integer IDs use 8 bytes vs variable-length strings
- **Compaction support** — Deleted vectors leave gaps that should be reclaimed

These requirements conflict: users want strings, but internals need integers.

### Alternatives Considered

1. **String IDs everywhere** — Simple but slow (hash lookups for every graph traversal)
2. **Integer IDs only** — Fast but user-hostile (forces external ID mapping)
3. **Auto-increment with user aliases** — Complex, two sources of truth

## Decision

Needle implements a **dual ID system** with bidirectional mapping:

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                                                                  │
│   insert("doc_123", vector)    search(query) → ["doc_456", ...]│
│   get("doc_123") → vector      delete("doc_123")                │
│                                                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ID Mapping Layer                             │
│                                                                  │
│   external_to_internal: HashMap<String, usize>                  │
│   internal_to_external: Vec<String>                             │
│                                                                  │
│   "doc_123" ←→ 0                                                │
│   "doc_456" ←→ 1                                                │
│   "doc_789" ←→ 2                                                │
│                                                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Internal Systems                             │
│                                                                  │
│   HNSW Graph: Node 0 → [1, 2]    Vector Storage: [v0, v1, v2]  │
│               Node 1 → [0, 2]                                   │
│               Node 2 → [0, 1]                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Type Definitions

```rust
// src/hnsw.rs - Internal ID type
pub type VectorId = usize;

// src/metadata.rs - Mapping structure
pub struct MetadataStore {
    /// Map external string ID to internal index
    external_to_internal: HashMap<String, VectorId>,

    /// Map internal index back to external ID (for search results)
    internal_to_external: Vec<String>,

    /// Metadata associated with each vector
    metadata: Vec<Option<serde_json::Value>>,

    /// Free list for deleted indices (enables compaction)
    free_list: Vec<VectorId>,
}
```

### ID Assignment Flow

```rust
impl MetadataStore {
    /// Assign an internal ID for a new external ID
    pub fn assign_id(&mut self, external_id: &str) -> Result<VectorId> {
        // Check for duplicate
        if self.external_to_internal.contains_key(external_id) {
            return Err(NeedleError::DuplicateId(external_id.to_string()));
        }

        // Reuse from free list if available (compaction-friendly)
        let internal_id = if let Some(recycled) = self.free_list.pop() {
            self.internal_to_external[recycled] = external_id.to_string();
            recycled
        } else {
            let new_id = self.internal_to_external.len();
            self.internal_to_external.push(external_id.to_string());
            self.metadata.push(None);
            new_id
        };

        self.external_to_internal.insert(external_id.to_string(), internal_id);
        Ok(internal_id)
    }

    /// Resolve external ID to internal
    pub fn resolve(&self, external_id: &str) -> Option<VectorId> {
        self.external_to_internal.get(external_id).copied()
    }

    /// Reverse lookup: internal to external
    pub fn external_id(&self, internal_id: VectorId) -> Option<&str> {
        self.internal_to_external.get(internal_id).map(|s| s.as_str())
    }
}
```

### Search Result Translation

```rust
impl Collection {
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // HNSW returns internal IDs
        let internal_results = self.index.search(query, k)?;

        // Translate to external IDs for user
        let results = internal_results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                let external_id = self.metadata.external_id(internal_id)?;
                Some(SearchResult {
                    id: external_id.to_string(),
                    distance,
                    metadata: self.metadata.get(internal_id).cloned(),
                })
            })
            .collect();

        Ok(results)
    }
}
```

### Compaction and ID Recycling

When vectors are deleted, their internal IDs are added to a free list:

```rust
impl MetadataStore {
    pub fn delete(&mut self, external_id: &str) -> Result<VectorId> {
        let internal_id = self.external_to_internal
            .remove(external_id)
            .ok_or_else(|| NeedleError::NotFound(external_id.to_string()))?;

        // Mark slot as available for reuse
        self.free_list.push(internal_id);
        self.internal_to_external[internal_id].clear(); // Keep slot, clear content

        Ok(internal_id)
    }
}
```

During compaction, IDs can be renumbered to eliminate gaps:

```rust
impl Collection {
    pub fn compact(&mut self) -> Result<CompactionStats> {
        // Renumber internal IDs to be contiguous
        // External IDs remain unchanged - users see no difference
        // ...
    }
}
```

## Consequences

### Benefits

1. **User-friendly API** — Users work with meaningful string identifiers
2. **Efficient internals** — Graph traversal uses O(1) integer lookups
3. **Compaction support** — Deleted slots can be reclaimed without user-visible changes
4. **External system integration** — IDs from other databases work directly

### Tradeoffs

1. **Memory overhead** — Two hash maps plus string storage
2. **Translation cost** — Every search result requires ID lookup (mitigated by cache-friendly Vec)
3. **Complexity** — More code paths than single-ID system

### What This Enabled

- **Natural API** — `collection.get("user_profile_123")` instead of `collection.get(42)`
- **Zero-downtime compaction** — Internal renumbering invisible to users
- **Batch imports** — External IDs preserved during bulk load

### What This Prevented

- **ID collision during merges** — External IDs can be namespaced
- **Fragmentation accumulation** — Free list enables slot reuse

## References

- Metadata store: `src/metadata.rs`
- Internal ID type: `src/hnsw.rs:15` (`pub type VectorId = usize`)
- Search result translation: `src/collection.rs:200-250`
