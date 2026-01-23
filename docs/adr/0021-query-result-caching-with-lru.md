# ADR-0021: Query Result Caching with LRU

## Status

Accepted

## Context

Vector similarity search via HNSW is computationally expensive, requiring graph traversal through multiple layers. In production workloads, certain query patterns emerge:

1. **Repeated queries** — The same or very similar queries are issued multiple times (e.g., autocomplete, recommendation refresh)
2. **Hot vectors** — A small subset of vectors appears frequently in results
3. **Latency sensitivity** — Applications often have strict p99 latency requirements

Without caching, every search incurs the full cost of HNSW traversal, even for identical queries issued seconds apart. This creates unnecessary CPU load and latency variance.

### Alternatives Considered

1. **External caching (Redis/Memcached)** — Adds operational complexity and network latency; overkill for embedded use cases
2. **Query result memoization** — Simple but unbounded memory growth
3. **Bloom filters for negative caching** — Only helps with "miss" cases, not repeated hits

## Decision

Needle implements an **optional in-memory LRU (Least Recently Used) cache** for query results with the following design:

### Cache Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Search Request                        │
│              (query vector, k, filter)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                  Cache Key Generation                    │
│     hash(query_vector) + hash(filter) + k + ef_search   │
└─────────────────────┬───────────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
      ┌──────────┐        ┌──────────┐
      │ Cache Hit│        │Cache Miss│
      └────┬─────┘        └────┬─────┘
           │                   │
           │                   ▼
           │           ┌──────────────┐
           │           │ HNSW Search  │
           │           └──────┬───────┘
           │                  │
           │                  ▼
           │           ┌──────────────┐
           │           │ Cache Insert │
           │           └──────┬───────┘
           │                  │
           └────────┬─────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Results    │
            └──────────────┘
```

### Configuration

```rust
// src/collection.rs
pub struct QueryCacheConfig {
    /// Maximum number of cached query results
    pub max_entries: usize,

    /// Time-to-live for cache entries (None = no expiry)
    pub ttl: Option<Duration>,

    /// Whether to cache filtered queries
    pub cache_filtered: bool,
}

impl Default for QueryCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Some(Duration::from_secs(300)), // 5 minutes
            cache_filtered: true,
        }
    }
}
```

### Cache Key Design

The cache key incorporates all factors that affect search results:

```rust
fn cache_key(query: &[f32], k: usize, ef_search: usize, filter: Option<&Filter>) -> u64 {
    let mut hasher = XxHash64::default();

    // Hash the query vector (quantized to reduce near-duplicate misses)
    for &v in query {
        hasher.write_u32((v * 1000.0) as u32);
    }

    hasher.write_usize(k);
    hasher.write_usize(ef_search);

    if let Some(f) = filter {
        hasher.write(f.to_canonical_string().as_bytes());
    }

    hasher.finish()
}
```

### Cache Invalidation

The cache is invalidated on **any write operation** to the collection:

```rust
impl Collection {
    pub fn insert(&mut self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        // ... insert logic ...
        self.query_cache.clear(); // Invalidate entire cache
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<bool> {
        // ... delete logic ...
        self.query_cache.clear();
        Ok(deleted)
    }
}
```

### Statistics

```rust
pub struct QueryCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub hit_rate: f64,
}
```

## Consequences

### Benefits

1. **Reduced latency for repeated queries** — Cache hits return in microseconds vs milliseconds for HNSW traversal
2. **Lower CPU utilization** — Hot queries don't repeatedly traverse the graph
3. **Predictable p99 latency** — Cache hits have consistent timing
4. **Zero external dependencies** — In-process caching via `lru` crate

### Tradeoffs

1. **Memory overhead** — Each cached result consumes memory (configurable limit)
2. **Cache invalidation granularity** — Currently invalidates entire cache on any write; fine-grained invalidation would add complexity
3. **Cold start penalty** — First query after invalidation always misses
4. **Stale results possible** — If TTL is set, results may be slightly outdated

### What This Enabled

- **Interactive search UIs** — Autocomplete and faceted search with sub-millisecond response
- **Recommendation refresh** — Periodic re-ranking without full search cost
- **A/B testing** — Same queries can be issued repeatedly during experiments

### What This Prevented

- **Unbounded memory growth** — LRU eviction ensures memory stays bounded
- **Complex invalidation logic** — Simple "invalidate all on write" is correct if not optimal

## References

- Implementation: `src/collection.rs:45-89` (QueryCacheConfig, QueryCacheStats)
- LRU crate: `Cargo.toml:46` (`lru = "0.12"`)
- Cache key hashing: Uses xxhash for fast, high-quality hashing
