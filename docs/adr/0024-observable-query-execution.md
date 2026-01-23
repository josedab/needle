# ADR-0024: Observable Query Execution with SearchExplain

## Status

Accepted

## Context

Vector search performance is notoriously difficult to debug:

1. **Black-box indices** — HNSW graph traversal is opaque; why did a query take 50ms?
2. **Filter interaction** — Metadata filters can dramatically change performance
3. **Parameter sensitivity** — Small changes to `ef_search` or `k` can have large effects
4. **Capacity planning** — How many queries per second can this deployment handle?

Traditional approaches require external profiling:
- **Flame graphs** — Require instrumented builds and post-processing
- **APM tools** — Add latency and operational complexity
- **Logging** — Verbose, hard to aggregate, impacts performance

For an embedded database aiming at "SQLite for vectors" simplicity, requiring external observability tools contradicts the zero-configuration philosophy.

### Alternatives Considered

1. **External profiling only** — Conflicts with embedded simplicity
2. **Detailed logging** — Performance overhead, hard to analyze
3. **Metrics only (counters/histograms)** — Loses per-query detail

## Decision

Needle implements **first-class query profiling** via the `SearchExplain` structure, returned alongside search results:

### SearchExplain Structure

```rust
// src/collection.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchExplain {
    /// Total wall-clock time for the search
    pub total_time_us: u64,

    /// Time spent in HNSW index traversal
    pub index_time_us: u64,

    /// Time spent evaluating metadata filters
    pub filter_time_us: u64,

    /// Time spent enriching results with metadata
    pub enrich_time_us: u64,

    /// Number of candidates from index before filtering
    pub candidates_before_filter: usize,

    /// Number of candidates after filter application
    pub candidates_after_filter: usize,

    /// HNSW-specific statistics
    pub hnsw_stats: Option<HnswExplain>,

    /// Cache hit information
    pub cache_hit: bool,

    /// Query parameters used
    pub parameters: QueryParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswExplain {
    /// Number of nodes visited during search
    pub visited_nodes: usize,

    /// Number of layers traversed
    pub layers_traversed: usize,

    /// Number of distance computations performed
    pub distance_computations: usize,

    /// Entry point node ID
    pub entry_point: usize,

    /// Time in each layer (microseconds)
    pub layer_times_us: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParameters {
    pub k: usize,
    pub ef_search: usize,
    pub filter: Option<String>,
    pub distance_function: String,
}
```

### API Design

Two search methods are provided:

```rust
impl Collection {
    /// Standard search - returns only results
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let (results, _explain) = self.search_internal(query, k, true)?;
        Ok(results)
    }

    /// Explained search - returns results + profiling data
    pub fn search_explain(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, SearchExplain)> {
        self.search_internal(query, k, false)
    }

    fn search_internal(
        &self,
        query: &[f32],
        k: usize,
        skip_explain: bool,
    ) -> Result<(Vec<SearchResult>, SearchExplain)> {
        let total_start = Instant::now();
        let mut explain = SearchExplain::default();

        // Check cache
        if let Some(cached) = self.cache.get(query, k) {
            explain.cache_hit = true;
            explain.total_time_us = total_start.elapsed().as_micros() as u64;
            return Ok((cached, explain));
        }

        // Index search with timing
        let index_start = Instant::now();
        let (candidates, hnsw_stats) = self.index.search_with_stats(query, k * 10)?;
        explain.index_time_us = index_start.elapsed().as_micros() as u64;
        explain.hnsw_stats = Some(hnsw_stats);
        explain.candidates_before_filter = candidates.len();

        // Filter application with timing
        let filter_start = Instant::now();
        let filtered = self.apply_filter(candidates, &self.current_filter)?;
        explain.filter_time_us = filter_start.elapsed().as_micros() as u64;
        explain.candidates_after_filter = filtered.len();

        // Result enrichment with timing
        let enrich_start = Instant::now();
        let results = self.enrich_results(filtered, k)?;
        explain.enrich_time_us = enrich_start.elapsed().as_micros() as u64;

        explain.total_time_us = total_start.elapsed().as_micros() as u64;

        Ok((results, explain))
    }
}
```

### HNSW Instrumentation

The HNSW index tracks traversal statistics:

```rust
// src/hnsw.rs
impl HnswIndex {
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<(VectorId, f32)>, HnswExplain)> {
        let mut stats = HnswExplain {
            visited_nodes: 0,
            layers_traversed: 0,
            distance_computations: 0,
            entry_point: self.entry_point,
            layer_times_us: Vec::new(),
        };

        let mut current = self.entry_point;

        // Traverse from top layer to bottom
        for layer in (0..=self.max_layer).rev() {
            let layer_start = Instant::now();
            stats.layers_traversed += 1;

            // Greedy search in this layer
            loop {
                let neighbors = self.get_neighbors(current, layer);
                stats.visited_nodes += neighbors.len();

                let mut improved = false;
                for &neighbor in &neighbors {
                    stats.distance_computations += 1;
                    let dist = self.distance(query, neighbor);
                    // ... traversal logic
                }

                if !improved {
                    break;
                }
            }

            stats.layer_times_us.push(layer_start.elapsed().as_micros() as u64);
        }

        Ok((results, stats))
    }
}
```

### HTTP API Integration

The server exposes explain via query parameter:

```rust
// POST /collections/{name}/search?explain=true
#[derive(Deserialize)]
struct SearchRequest {
    query: Vec<f32>,
    k: usize,
    #[serde(default)]
    explain: bool,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    explain: Option<SearchExplain>,
}
```

### Example Output

```json
{
  "results": [
    {"id": "doc_123", "distance": 0.15, "metadata": {"title": "..."}},
    {"id": "doc_456", "distance": 0.23, "metadata": {"title": "..."}}
  ],
  "explain": {
    "total_time_us": 1250,
    "index_time_us": 980,
    "filter_time_us": 150,
    "enrich_time_us": 120,
    "candidates_before_filter": 100,
    "candidates_after_filter": 45,
    "cache_hit": false,
    "hnsw_stats": {
      "visited_nodes": 89,
      "layers_traversed": 4,
      "distance_computations": 356,
      "entry_point": 42,
      "layer_times_us": [50, 120, 380, 430]
    },
    "parameters": {
      "k": 10,
      "ef_search": 50,
      "filter": "category = 'electronics'",
      "distance_function": "cosine"
    }
  }
}
```

## Consequences

### Benefits

1. **Self-service debugging** — Developers can diagnose slow queries without external tools
2. **Zero configuration** — No setup, agents, or infrastructure required
3. **Production-safe** — Explain data is cheap to compute (already tracked internally)
4. **Actionable insights** — Clear breakdown shows where time is spent

### Tradeoffs

1. **API surface area** — Two methods instead of one
2. **Response size** — Explain adds ~500 bytes per response when enabled
3. **Slight overhead** — Timing calls add nanoseconds (negligible)

### What This Enabled

- **Query optimization** — Users can tune `ef_search` based on visited_nodes
- **Filter debugging** — See how many candidates filters eliminate
- **Capacity planning** — Understand QPS limits from timing data
- **Regression detection** — Compare explain output across versions

### What This Prevented

- **Blind tuning** — No more guessing why queries are slow
- **External tool dependency** — No APM required for basic profiling

## References

- SearchExplain struct: `src/collection.rs:127-150`
- HNSW stats: `src/hnsw.rs:300-400`
- HTTP endpoint: `src/server.rs` (search handler with explain parameter)
