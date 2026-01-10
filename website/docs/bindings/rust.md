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
    db.create_collection("documents", 384, DistanceFunction::Cosine)?;

    // Get collection reference
    let collection = db.collection("documents")?;

    // Insert vectors
    collection.insert(
        "doc1",
        &vec![0.1; 384],
        json!({"title": "Hello World"})
    )?;

    // Search
    let results = collection.search(&vec![0.1; 384], 10, None)?;

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
let db = Database::in_memory()?;

// With custom configuration
let config = DatabaseConfig::new()
    .with_mmap_threshold(10 * 1024 * 1024);
let db = Database::open_with_config("vectors.needle", config)?;
```

### Collection Management

```rust
// Create collection
db.create_collection("docs", 384, DistanceFunction::Cosine)?;

// With custom config
use needle::CollectionConfig;
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(32)
    .with_hnsw_ef_construction(400);
db.create_collection_with_config("high_quality", config)?;

// List collections
let names = db.list_collections()?;

// Check if exists
if db.collection_exists("docs")? {
    // ...
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

// Insert
collection.insert("id1", &vector, json!({"key": "value"}))?;

// Get by ID
let entry = collection.get("id1")?;
println!("Vector: {:?}", entry.vector);
println!("Metadata: {:?}", entry.metadata);

// Check if exists
if collection.exists("id1")? {
    // ...
}

// Delete
collection.delete("id1")?;

// Count vectors
let count = collection.count()?;

// Clear all vectors
collection.clear()?;
```

### Searching

```rust
use needle::Filter;

// Basic search
let results = collection.search(&query_vector, 10, None)?;

for result in results {
    println!("ID: {}, Distance: {}", result.id, result.distance);
    println!("Metadata: {:?}", result.metadata);
}

// Search with filter
let filter = Filter::parse(&json!({
    "category": "programming",
    "year": {"$gte": 2020}
}))?;
let results = collection.search(&query_vector, 10, Some(&filter))?;

// Search with custom ef_search
let results = collection.search_with_params(&query_vector, 10, None, 100)?;

// Batch search (parallel)
let queries: Vec<&[f32]> = vec![&query1, &query2, &query3];
let all_results = collection.batch_search(&queries, 10, None)?;
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
use needle::DistanceFunction;

// Cosine distance (recommended for text embeddings)
db.create_collection("text", 384, DistanceFunction::Cosine)?;

// Euclidean distance (L2)
db.create_collection("images", 512, DistanceFunction::Euclidean)?;

// Dot product (for recommendation systems)
db.create_collection("recommendations", 128, DistanceFunction::Dot)?;

// Manhattan distance (L1)
db.create_collection("sparse", 1000, DistanceFunction::Manhattan)?;
```

## Quantization

```rust
use needle::{CollectionConfig, QuantizationType};

// Scalar quantization (4x compression)
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Scalar);

// Product quantization (8-32x compression)
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Product {
        num_subvectors: 48,
        num_centroids: 256,
    });

// Binary quantization (32x compression)
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Binary);

db.create_collection_with_config("quantized", config)?;
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
        let results = coll.search(&query, 10, None).unwrap();
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
