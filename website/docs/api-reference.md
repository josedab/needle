---
sidebar_position: 7
---

# API Reference

Complete reference for Needle's Rust API. For language-specific bindings, see [Python](/docs/bindings/python), [JavaScript](/docs/bindings/javascript), or [Swift/Kotlin](/docs/bindings/swift-kotlin).

## Database

### `Database::open`

Opens or creates a database file.

```rust
pub fn open(path: impl AsRef<Path>) -> Result<Database>
```

**Example:**
```rust
let db = Database::open("vectors.needle")?;
```

### `Database::in_memory`

Creates an in-memory database.

```rust
pub fn in_memory() -> Database
```

**Example:**
```rust
let db = Database::in_memory();
```

### `Database::create_collection`

Creates a new collection with default settings (cosine distance).

```rust
pub fn create_collection(
    &self,
    name: impl Into<String>,
    dimensions: usize,
) -> Result<()>
```

**Example:**
```rust
db.create_collection("documents", 384)?;
```

### `Database::create_collection_with_config`

Creates a collection with custom configuration (distance function, HNSW params, etc.).

```rust
pub fn create_collection_with_config(
    &self,
    config: CollectionConfig,
) -> Result<()>
```

**Example:**
```rust
use needle::{CollectionConfig, DistanceFunction};

let config = CollectionConfig::new("documents", 384)
    .with_distance(DistanceFunction::DotProduct);
db.create_collection_with_config(config)?;
```

### `Database::collection`

Gets a reference to a collection.

```rust
pub fn collection(&self, name: &str) -> Result<CollectionRef>
```

### `Database::list_collections`

Lists all collection names.

```rust
pub fn list_collections(&self) -> Result<Vec<String>>
```

### `Database::delete_collection`

Deletes a collection.

```rust
pub fn delete_collection(&self, name: &str) -> Result<()>
```

### `Database::save`

Saves the database to disk.

```rust
pub fn save(&self) -> Result<()>
```

## Aliases

### `Database::create_alias`

Creates an alias pointing to a collection.

```rust
pub fn create_alias(&self, alias: &str, collection: &str) -> Result<()>
```

**Example:**
```rust
db.create_alias("prod", "docs_v2")?;
```

### `Database::update_alias`

Updates an existing alias to point to a different collection.

```rust
pub fn update_alias(&self, alias: &str, collection: &str) -> Result<()>
```

**Example:**
```rust
db.update_alias("prod", "docs_v3")?;
```

### `Database::delete_alias`

Deletes an alias. Returns `true` if it existed.

```rust
pub fn delete_alias(&self, alias: &str) -> Result<bool>
```

### `Database::list_aliases`

Lists all aliases as `(alias_name, collection_name)` tuples.

```rust
pub fn list_aliases(&self) -> Vec<(String, String)>
```

### `Database::get_canonical_name`

Resolves an alias to its target collection name.

```rust
pub fn get_canonical_name(&self, alias: &str) -> Option<String>
```

### `Database::aliases_for_collection`

Gets all aliases pointing to a specific collection.

```rust
pub fn aliases_for_collection(&self, collection: &str) -> Vec<String>
```

## CollectionConfig

Configuration for creating collections.

```rust
pub struct CollectionConfig {
    pub dimensions: usize,
    pub distance: DistanceFunction,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub quantization: Option<QuantizationType>,
    pub default_ttl_seconds: Option<u64>,
    pub lazy_expiration: bool,
}
```

### Builder Methods

```rust
impl CollectionConfig {
    pub fn new(name: impl Into<String>, dimensions: usize) -> Self;
    pub fn with_distance(self, distance: DistanceFunction) -> Self;
    pub fn with_hnsw_m(self, m: usize) -> Self;
    pub fn with_hnsw_ef_construction(self, ef: usize) -> Self;
    pub fn with_quantization(self, quantization: QuantizationType) -> Self;
    pub fn with_default_ttl_seconds(self, ttl: Option<u64>) -> Self;
    pub fn with_lazy_expiration(self, enabled: bool) -> Self;
}
```

**Example:**
```rust
let config = CollectionConfig::new("documents", 384)
    .with_distance(DistanceFunction::Cosine)
    .with_hnsw_m(32)
    .with_hnsw_ef_construction(400)
    .with_quantization(QuantizationType::Scalar)
    .with_default_ttl_seconds(Some(3600)); // 1 hour default TTL
```

## Collection

### `Collection::insert`

Inserts a vector with optional metadata.

```rust
pub fn insert(
    &self,
    id: impl Into<String>,
    vector: &[f32],
    metadata: Option<serde_json::Value>,
) -> Result<()>
```

**Example:**
```rust
collection.insert("doc1", &embedding, Some(json!({"title": "Hello"})))?;
collection.insert("doc2", &embedding, None)?; // No metadata
```

### `Collection::insert_with_ttl`

Inserts a vector with an optional time-to-live.

```rust
pub fn insert_with_ttl(
    &self,
    id: &str,
    vector: &[f32],
    metadata: Option<serde_json::Value>,
    ttl_seconds: Option<u64>,
) -> Result<()>
```

**Example:**
```rust
// Expires in 1 hour
collection.insert_with_ttl("doc1", &embedding, Some(json!({"title": "Hello"})), Some(3600))?;

// Never expires (or uses collection default)
collection.insert_with_ttl("doc2", &embedding, None, None)?;
```

### `Collection::get`

Gets a vector by ID. Returns `None` if not found.

```rust
pub fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)>
```

### `Collection::delete`

Deletes a vector by ID. Returns `Ok(true)` if deleted, `Ok(false)` if not found.

```rust
pub fn delete(&self, id: &str) -> Result<bool>
```

### `Collection::search`

Searches for the k most similar vectors.

```rust
pub fn search(
    &self,
    query: &[f32],
    k: usize,
) -> Result<Vec<SearchResult>>
```

**Example:**
```rust
let results = collection.search(&query_vector, 10)?;
```

### `Collection::search_with_filter`

Searches with metadata filtering applied during the search.

```rust
pub fn search_with_filter(
    &self,
    query: &[f32],
    k: usize,
    filter: &Filter,
) -> Result<Vec<SearchResult>>
```

**Example:**
```rust
let filter = Filter::parse(&json!({"category": "programming"}))?;
let results = collection.search_with_filter(&query_vector, 10, &filter)?;
```

### `Collection::query` (Builder Pattern)

Fluent builder for searches with filters, limits, and distance overrides.

```rust
pub fn query(&self, query: &[f32]) -> SearchParams
```

**Example:**
```rust
let results = collection.query(&query_vector)
    .limit(5)
    .filter(&filter)
    .execute()?;
```

### `Collection::search_with_params`

Searches with custom HNSW parameters.

```rust
pub fn search_with_params(
    &self,
    query: &[f32],
    k: usize,
    filter: Option<&Filter>,
    ef_search: usize,
) -> Result<Vec<SearchResult>>
```

### `Collection::search_explain`

Searches with detailed performance metrics.

```rust
pub fn search_explain(
    &self,
    query: &[f32],
    k: usize,
    filter: Option<&Filter>,
) -> Result<(Vec<SearchResult>, SearchExplain)>
```

### `Collection::batch_search`

Searches multiple queries in parallel.

```rust
pub fn batch_search(
    &self,
    queries: &[&[f32]],
    k: usize,
    filter: Option<&Filter>,
) -> Result<Vec<Vec<SearchResult>>>
```

### `Collection::search_with_options`

Searches with distance override, pre-filter, and post-filter support.

```rust
pub fn search_with_options(
    &self,
    query: &[f32],
    k: usize,
    distance_override: Option<DistanceFunction>,
    filter: Option<&Filter>,
    post_filter: Option<&Filter>,
    post_filter_factor: usize,
) -> Result<Vec<SearchResult>>
```

**Example:**
```rust
use needle::DistanceFunction;

// Override distance function (falls back to brute-force if different from index)
let results = collection.search_with_options(
    &query,
    10,
    Some(DistanceFunction::Euclidean), // Override distance
    Some(&pre_filter),                  // Pre-filter
    Some(&post_filter),                 // Post-filter
    3,                                  // Post-filter factor
)?;
```

### `Collection::search_builder`

Creates a fluent search builder for complex queries.

```rust
pub fn search_builder(&self, query: &[f32]) -> SearchBuilder
```

**Example:**
```rust
let results = collection.search_builder(&query)
    .k(10)
    .filter(&pre_filter)
    .distance(DistanceFunction::Euclidean)  // Override distance function
    .post_filter(&post_filter)
    .post_filter_factor(3)
    .execute()?;
```

## SearchBuilder

Fluent builder for configuring searches.

```rust
impl SearchBuilder {
    pub fn k(self, k: usize) -> Self;
    pub fn filter(self, filter: &Filter) -> Self;
    pub fn distance(self, distance: DistanceFunction) -> Self;
    pub fn post_filter(self, filter: &Filter) -> Self;
    pub fn post_filter_factor(self, factor: usize) -> Self;
    pub fn execute(self) -> Result<Vec<SearchResult>>;
}
```

:::note Distance Override
When `distance()` specifies a different function than the collection's index, Needle automatically falls back to brute-force search. This is slower but ensures correct results.
:::

### `Collection::count`

Returns the number of vectors.

```rust
pub fn count(&self) -> Result<usize>
```

### `Collection::exists`

Checks if a vector exists.

```rust
pub fn exists(&self, id: &str) -> Result<bool>
```

### `Collection::clear`

Removes all vectors.

```rust
pub fn clear(&self) -> Result<()>
```

### `Collection::compact`

Reclaims space from deleted vectors. Also runs expiration sweep first.

```rust
pub fn compact(&self) -> Result<()>
```

### `Collection::info`

Returns collection information.

```rust
pub fn info(&self) -> Result<CollectionInfo>
```

## TTL / Expiration

### `Collection::expire_vectors`

Removes all expired vectors from the collection.

```rust
pub fn expire_vectors(&self) -> Result<usize>
```

**Returns:** Number of vectors removed.

**Example:**
```rust
let removed = collection.expire_vectors()?;
println!("Removed {} expired vectors", removed);
```

### `Collection::ttl_stats`

Returns TTL statistics for the collection.

```rust
pub fn ttl_stats(&self) -> Result<(usize, usize, Option<u64>, Option<u64>)>
```

**Returns:** Tuple of `(total_with_ttl, expired_count, nearest_expiration, furthest_expiration)`.

**Example:**
```rust
let (total, expired, nearest, furthest) = collection.ttl_stats()?;
println!("Total with TTL: {}, Expired: {}", total, expired);
```

### `Collection::needs_expiration_sweep`

Checks if the collection has more than the given ratio of expired vectors.

```rust
pub fn needs_expiration_sweep(&self, threshold: f32) -> Result<bool>
```

**Example:**
```rust
// Check if more than 10% of vectors are expired
if collection.needs_expiration_sweep(0.1)? {
    collection.expire_vectors()?;
}
```

## SearchResult

Result from a search operation.

```rust
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: serde_json::Value,
}
```

## SearchExplain

Performance metrics from search_explain.

```rust
pub struct SearchExplain {
    pub nodes_visited: usize,
    pub distance_computations: usize,
    pub layers_traversed: usize,
    pub index_time: Duration,
    pub filter_time: Duration,
    pub total_time: Duration,
}
```

## Filter

MongoDB-style query filter.

### `Filter::parse`

Parses a JSON filter.

```rust
pub fn parse(json: &serde_json::Value) -> Result<Filter>
```

**Operators:**

| Operator | Example |
|----------|---------|
| `$eq` | `{"field": {"$eq": value}}` |
| `$ne` | `{"field": {"$ne": value}}` |
| `$gt` | `{"field": {"$gt": value}}` |
| `$gte` | `{"field": {"$gte": value}}` |
| `$lt` | `{"field": {"$lt": value}}` |
| `$lte` | `{"field": {"$lte": value}}` |
| `$in` | `{"field": {"$in": [v1, v2]}}` |
| `$nin` | `{"field": {"$nin": [v1, v2]}}` |
| `$and` | `{"$and": [{...}, {...}]}` |
| `$or` | `{"$or": [{...}, {...}]}` |
| `$not` | `{"$not": {...}}` |

**Example:**
```rust
let filter = Filter::parse(&json!({
    "$and": [
        {"category": {"$in": ["books", "articles"]}},
        {"year": {"$gte": 2020}}
    ]
}))?;
```

## DistanceFunction

Available distance metrics.

```rust
pub enum DistanceFunction {
    Cosine,
    Euclidean,
    Dot,
    Manhattan,
}
```

## QuantizationType

Quantization options.

```rust
pub enum QuantizationType {
    Scalar,
    Product {
        num_subvectors: usize,
        num_centroids: usize,
    },
    Binary,
}
```

## Bm25Index

BM25 text index (requires `hybrid` feature).

```rust
impl Bm25Index {
    pub fn default() -> Self;
    pub fn with_config(config: Bm25Config) -> Self;
    pub fn index_document(&mut self, id: &str, text: &str);
    pub fn remove_document(&mut self, id: &str);
    pub fn search(&self, query: &str, k: usize) -> Vec<Bm25Result>;
}
```

## reciprocal_rank_fusion

Fuses vector and BM25 results (requires `hybrid` feature).

```rust
pub fn reciprocal_rank_fusion(
    vector_results: &[SearchResult],
    bm25_results: &[Bm25Result],
    config: &RrfConfig,
    k: usize,
) -> Vec<SearchResult>
```

## auto_tune

Automatically tunes HNSW parameters.

```rust
pub fn auto_tune(constraints: &TuningConstraints) -> TuningResult
```

**Example:**
```rust
let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(4 * 1024 * 1024 * 1024);

let result = auto_tune(&constraints);
```

## NeedleError

Error types returned by Needle operations.

```rust
pub enum NeedleError {
    IoError(std::io::Error),
    CollectionNotFound(String),
    CollectionExists(String),
    CollectionHasAliases(String),  // Cannot drop collection with active aliases
    AliasNotFound(String),
    AliasAlreadyExists(String),
    DimensionMismatch { expected: usize, got: usize },
    InvalidFilter(String),
    VectorNotFound(String),
    SerializationError(String),
    EncryptionError(String),
}
```

## Result Type

Needle uses a custom Result type:

```rust
pub type Result<T> = std::result::Result<T, NeedleError>;
```

## Next Steps

- [Rust Bindings](/docs/bindings/rust) - Complete Rust examples
- [Getting Started](/docs/getting-started) - Quick start guide
- [Core Concepts](/docs/concepts/vectors) - Understanding vectors and search
