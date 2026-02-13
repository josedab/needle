---
sidebar_position: 1
---

# Rust

Needle is written in Rust, so the Rust API is the most complete and performant. This guide covers the full Rust API.

## Installation

Add Needle to your `Cargo.toml`:

```toml
[dependencies]
needle = "0.1"
serde_json = "1.0"
```

With optional features:

```toml
[dependencies]
needle = { version = "0.1", features = ["simd", "server", "hybrid", "metrics"] }
```

## Quick Start

```rust
use needle::{Database, DistanceFunction, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    // Create or open a database
    let db = Database::open("vectors.needle")?;

    // Create a collection
    db.create_collection("documents", 384)?;

    // Get collection reference
    let collection = db.collection("documents")?;

    // Insert vectors
    collection.insert(
        "doc1",
        &vec![0.1; 384],
        Some(json!({"title": "Hello World"}))
    )?;

    // Search
    let results = collection.search(&vec![0.1; 384], 10)?;

    // Save to disk
    db.save()?;

    Ok(())
}
```

## Database API

### Creating and Opening

```rust
use needle::{Database, DatabaseConfig};

// Open or create database (file-based)
let db = Database::open("vectors.needle")?;

// In-memory database (for testing)
let db = Database::in_memory();

// With custom configuration
let config = DatabaseConfig::new("vectors.needle")
    .with_mmap_threshold(10 * 1024 * 1024);
let db = Database::open_with_config(config)?;
```

### Collection Management

```rust
// Create collection (uses cosine distance by default)
db.create_collection("docs", 384)?;

// With custom config
use needle::CollectionConfig;
let config = CollectionConfig::new("high_quality", 384)
    .with_distance(DistanceFunction::Cosine)
    .with_hnsw_m(32)
    .with_hnsw_ef_construction(400);
db.create_collection_with_config(config)?;

// List collections
let names = db.list_collections()?;

// Check if collection exists
if db.collection("docs").is_ok() {
    // collection exists
}

// Get collection reference
let collection = db.collection("docs")?;

// Delete collection
db.delete_collection("old_collection")?;

// Rename collection
db.rename_collection("old_name", "new_name")?;
```

### Persistence

```rust
// Save changes to disk
db.save()?;

// Save to different path
db.save_to("backup.needle")?;
```

## Collection API

### Vector Operations

```rust
let collection = db.collection("documents")?;

// Insert with metadata
collection.insert("id1", &vector, Some(json!({"key": "value"})))?;

// Insert without metadata
collection.insert("id2", &vector, None)?;

// Get by ID — returns Option, not Result
if let Some((vector, metadata)) = collection.get("id1") {
    println!("Vector: {:?}", vector);
    println!("Metadata: {:?}", metadata);
}

// Delete — returns Ok(true) if found, Ok(false) if not
let was_deleted = collection.delete("id1")?;

// Count vectors
let count = collection.count()?;

// Clear all vectors
collection.clear()?;
```

### Searching

```rust
use needle::Filter;

// Basic search
let results = collection.search(&query_vector, 10)?;

for result in results {
    println!("ID: {}, Distance: {}", result.id, result.distance);
    println!("Metadata: {:?}", result.metadata);
}

// Search with filter
let filter = Filter::parse(&json!({
    "category": "programming",
    "year": {"$gte": 2020}
}))?;
let results = collection.search_with_filter(&query_vector, 10, &filter)?;

// Fluent query builder with filter and limit
let results = collection.query(&query_vector)
    .limit(10)
    .filter(&filter)
    .execute()?;
// all_results is Vec<Vec<SearchResult>>
```

### Search Explain

```rust
// Get detailed search metrics
let (results, explain) = collection.search_explain(&query, 10, None)?;

println!("Nodes visited: {}", explain.nodes_visited);
println!("Distance computations: {}", explain.distance_computations);
println!("Index time: {:?}", explain.index_time);
println!("Filter time: {:?}", explain.filter_time);
```

### Iteration

```rust
// Iterate over all vectors
for (id, vector, metadata) in collection.iter()? {
    println!("{}: {:?}", id, metadata);
}

// Iterate only IDs
for id in collection.iter_ids()? {
    println!("{}", id);
}

// Iterate only metadata
for (id, metadata) in collection.iter_metadata()? {
    println!("{}: {:?}", id, metadata);
}
```

### Maintenance

```rust
// Compact to reclaim space from deletions
collection.compact()?;

// Get collection info
let info = collection.info()?;
println!("Dimensions: {}", info.dimensions);
println!("Count: {}", info.count);
println!("Distance: {:?}", info.distance);
```

## Filtering

### Operators

```rust
use needle::Filter;

// Equality
let filter = Filter::parse(&json!({"status": "active"}))?;

// Comparison
let filter = Filter::parse(&json!({
    "price": {"$gt": 10},
    "stock": {"$gte": 1},
    "rating": {"$lt": 5},
    "views": {"$lte": 1000}
}))?;

// Not equal
let filter = Filter::parse(&json!({
    "status": {"$ne": "deleted"}
}))?;

// In array
let filter = Filter::parse(&json!({
    "category": {"$in": ["books", "movies"]}
}))?;

// Not in array
let filter = Filter::parse(&json!({
    "status": {"$nin": ["deleted", "hidden"]}
}))?;

// Logical AND (explicit)
let filter = Filter::parse(&json!({
    "$and": [
        {"status": "active"},
        {"price": {"$lt": 100}}
    ]
}))?;

// Logical OR
let filter = Filter::parse(&json!({
    "$or": [
        {"category": "electronics"},
        {"category": "computers"}
    ]
}))?;

// Logical NOT
let filter = Filter::parse(&json!({
    "$not": {"status": "deleted"}
}))?;
```

## Distance Functions

```rust
use needle::{CollectionConfig, DistanceFunction};

// Cosine distance (recommended for text embeddings) — this is the default
db.create_collection("text", 384)?;

// Euclidean distance (L2)
let config = CollectionConfig::new("images", 512)
    .with_distance(DistanceFunction::Euclidean);
db.create_collection_with_config(config)?;

// Dot product (for recommendation systems)
let config = CollectionConfig::new("recommendations", 128)
    .with_distance(DistanceFunction::DotProduct);
db.create_collection_with_config(config)?;

// Manhattan distance (L1)
let config = CollectionConfig::new("sparse", 1000)
    .with_distance(DistanceFunction::Manhattan);
db.create_collection_with_config(config)?;
```

## Quantization

```rust
use needle::{CollectionConfig, QuantizationType};

// Scalar quantization (4x compression)
let config = CollectionConfig::new("quantized_sq", 384)
    .with_quantization(QuantizationType::Scalar);

// Product quantization (8-32x compression)
let config = CollectionConfig::new("quantized_pq", 384)
    .with_quantization(QuantizationType::Product {
        num_subvectors: 48,
        num_centroids: 256,
    });

// Binary quantization (32x compression)
let config = CollectionConfig::new("quantized_bq", 384)
    .with_quantization(QuantizationType::Binary);

db.create_collection_with_config(config)?;
```

## Hybrid Search

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

// Create BM25 index
let mut bm25 = Bm25Index::default();

// Index documents
bm25.index_document("doc1", "Introduction to Rust programming");
bm25.index_document("doc2", "Machine learning with Python");

// Search
let bm25_results = bm25.search("Rust programming", 10);

// Fuse with vector results
let config = RrfConfig::default();
let hybrid = reciprocal_rank_fusion(&vector_results, &bm25_results, &config, 10);
```

## Auto-Tuning

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(4 * 1024 * 1024 * 1024)
    .with_latency_target_ms(10);

let result = auto_tune(&constraints);

println!("Recommended M: {}", result.config.hnsw_m);
println!("Recommended ef: {}", result.config.ef_construction);
println!("Estimated recall: {:.1}%", result.estimated_recall * 100.0);
println!("Estimated memory: {} MB", result.estimated_memory_mb);
```

## Error Handling

```rust
use needle::{NeedleError, Result};

fn example() -> Result<()> {
    let db = Database::open("test.needle")?;

    match db.collection("nonexistent") {
        Ok(collection) => { /* use collection */ },
        Err(NeedleError::CollectionNotFound(name)) => {
            println!("Collection '{}' not found", name);
        },
        Err(e) => return Err(e),
    }

    Ok(())
}

// Error types
match result {
    Err(NeedleError::IoError(e)) => { /* I/O error */ },
    Err(NeedleError::CollectionNotFound(name)) => { /* collection doesn't exist */ },
    Err(NeedleError::CollectionExists(name)) => { /* collection already exists */ },
    Err(NeedleError::DimensionMismatch { expected, got }) => { /* wrong dimensions */ },
    Err(NeedleError::InvalidFilter(msg)) => { /* filter parse error */ },
    Err(NeedleError::VectorNotFound(id)) => { /* vector doesn't exist */ },
    _ => { /* other errors */ },
}
```

## Thread Safety

```rust
use std::sync::Arc;
use std::thread;

let db = Arc::new(Database::open("test.needle")?);

// Collections are thread-safe
let collection = db.collection("documents")?;

let handles: Vec<_> = (0..4).map(|i| {
    let coll = collection.clone();
    thread::spawn(move || {
        let results = coll.search(&query, 10).unwrap();
        println!("Thread {}: {} results", i, results.len());
    })
}).collect();

for handle in handles {
    handle.join().unwrap();
}
```

## HTTP Server

```rust
use needle::server;

#[tokio::main]
async fn main() -> needle::Result<()> {
    let db = Database::open("vectors.needle")?;

    // Start server
    server::run("0.0.0.0:8080", db).await?;

    Ok(())
}
```

Or use the CLI:

```bash
needle serve -a 0.0.0.0:8080 -d vectors.needle
```

## Next Steps

- [Python Bindings](/docs/bindings/python)
- [JavaScript Bindings](/docs/bindings/javascript)
- [API Reference](/docs/api-reference)
