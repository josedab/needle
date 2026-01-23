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
pub fn in_memory() -> Result<Database>
```

**Example:**
```rust
let db = Database::in_memory()?;
```

### `Database::create_collection`

Creates a new collection.

```rust
pub fn create_collection(
    &self,
    name: &str,
    dimensions: usize,
    distance: DistanceFunction,
) -> Result<()>
```

**Example:**
```rust
db.create_collection("documents", 384, DistanceFunction::Cosine)?;
```

### `Database::create_collection_with_config`

Creates a collection with custom configuration.

```rust
pub fn create_collection_with_config(
    &self,
    name: &str,
    config: CollectionConfig,
) -> Result<()>
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

## CollectionConfig

Configuration for creating collections.

```rust
pub struct CollectionConfig {
    pub dimensions: usize,
    pub distance: DistanceFunction,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub quantization: Option<QuantizationType>,
}
```

### Builder Methods

```rust
impl CollectionConfig {
    pub fn new(dimensions: usize, distance: DistanceFunction) -> Self;
    pub fn with_hnsw_m(self, m: usize) -> Self;
    pub fn with_hnsw_ef_construction(self, ef: usize) -> Self;
    pub fn with_quantization(self, quantization: QuantizationType) -> Self;
}
```

**Example:**
```rust
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(32)
    .with_hnsw_ef_construction(400)
    .with_quantization(QuantizationType::Scalar);
```

## Collection

### `Collection::insert`

Inserts a vector with metadata.

```rust
pub fn insert(
    &self,
    id: &str,
    vector: &[f32],
    metadata: serde_json::Value,
) -> Result<()>
```

**Example:**
```rust
collection.insert("doc1", &embedding, json!({"title": "Hello"}))?;
```

### `Collection::get`

Gets a vector by ID.

```rust
pub fn get(&self, id: &str) -> Result<VectorEntry>
```

**Returns:** `VectorEntry { id, vector, metadata }`

### `Collection::delete`

Deletes a vector by ID.

```rust
pub fn delete(&self, id: &str) -> Result<()>
```

### `Collection::search`

Searches for similar vectors.

```rust
pub fn search(
    &self,
    query: &[f32],
    k: usize,
    filter: Option<&Filter>,
) -> Result<Vec<SearchResult>>
```

**Example:**
```rust
let results = collection.search(&query_vector, 10, None)?;

let filter = Filter::parse(&json!({"category": "programming"}))?;
let results = collection.search(&query_vector, 10, Some(&filter))?;
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

Reclaims space from deleted vectors.

```rust
pub fn compact(&self) -> Result<()>
```

### `Collection::info`

Returns collection information.

```rust
pub fn info(&self) -> Result<CollectionInfo>
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
