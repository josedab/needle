---
sidebar_position: 2
---

# Collections

Collections are the primary organizational unit in Needle. This page explains how to create, manage, and optimize collections for your use case.

## What is a Collection?

A **collection** is a container for vectors that share the same dimensionality and distance function. Think of it like a table in a relational database, but optimized for vector similarity search.

Each collection includes:
- **Vectors**: The actual embedding data
- **Metadata**: JSON documents associated with each vector
- **HNSW Index**: The search index for fast approximate nearest neighbor queries
- **Optional**: BM25 index for hybrid search, sparse vectors, etc.

## Creating Collections

### Basic Creation

```rust
use needle::{Database, DistanceFunction};

let db = Database::open("mydb.needle")?;

// Create a collection with 384 dimensions and cosine distance
db.create_collection("documents", 384, DistanceFunction::Cosine)?;
```

### With Custom Configuration

```rust
use needle::{CollectionConfig, DistanceFunction, QuantizationType};

let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(32)                    // More connections = better recall
    .with_hnsw_ef_construction(400)     // Higher = better index quality
    .with_quantization(QuantizationType::Scalar);

db.create_collection_with_config("high_quality_docs", config)?;
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `dimensions` | Required | Number of dimensions per vector |
| `distance` | Required | Distance function (Cosine, Euclidean, Dot, Manhattan) |
| `hnsw_m` | 16 | Max connections per node (higher = better recall, more memory) |
| `hnsw_ef_construction` | 200 | Search depth during index build |
| `quantization` | None | Compression type (Scalar, Product, Binary) |

## Working with Collections

### Getting a Collection Reference

```rust
// Get a reference to an existing collection
let collection = db.collection("documents")?;

// The reference is thread-safe and can be cloned
let collection_clone = collection.clone();
```

### Collection Operations

```rust
// Insert a vector
collection.insert("doc1", &embedding, json!({"title": "Hello"}))?;

// Get a vector by ID
let vector = collection.get("doc1")?;

// Delete a vector
collection.delete("doc1")?;

// Count vectors
let count = collection.count()?;

// Search
let results = collection.search(&query, 10, None)?;

// Clear all vectors
collection.clear()?;
```

### Batch Operations

For better performance with multiple operations:

```rust
// Batch search
let queries: Vec<&[f32]> = vec![&query1, &query2, &query3];
let results = collection.batch_search(&queries, 10, None)?;

// Results is Vec<Vec<SearchResult>>
for (i, query_results) in results.iter().enumerate() {
    println!("Query {}: {} results", i, query_results.len());
}
```

## Collection Management

### Listing Collections

```rust
let collections = db.list_collections()?;
for name in collections {
    println!("Collection: {}", name);
}
```

### Getting Collection Info

```rust
let info = collection.info()?;
println!("Name: {}", info.name);
println!("Dimensions: {}", info.dimensions);
println!("Vector count: {}", info.count);
println!("Distance function: {:?}", info.distance);
```

### Deleting Collections

```rust
db.delete_collection("old_collection")?;
```

### Renaming Collections

```rust
db.rename_collection("old_name", "new_name")?;
```

## Data Persistence

### Saving Changes

Changes are kept in memory until you explicitly save:

```rust
// Insert some vectors
collection.insert("doc1", &embedding, json!({}))?;

// Save to disk
db.save()?;
```

### Auto-Save

For automatic persistence, you can implement auto-save:

```rust
use std::time::Duration;
use std::thread;

// Save every 30 seconds
let db_clone = db.clone();
thread::spawn(move || {
    loop {
        thread::sleep(Duration::from_secs(30));
        if let Err(e) = db_clone.save() {
            eprintln!("Auto-save failed: {}", e);
        }
    }
});
```

## Compaction

When you delete vectors, the space isn't immediately reclaimed. Use `compact()` to reclaim space:

```rust
// Delete many vectors
for i in 0..1000 {
    collection.delete(&format!("doc{}", i))?;
}

// Reclaim space
collection.compact()?;
db.save()?;
```

## Multi-Collection Patterns

### Sharding by Category

```rust
// Create separate collections for different categories
db.create_collection("products_electronics", 384, DistanceFunction::Cosine)?;
db.create_collection("products_clothing", 384, DistanceFunction::Cosine)?;
db.create_collection("products_books", 384, DistanceFunction::Cosine)?;

// Route inserts to the appropriate collection
fn insert_product(db: &Database, product: &Product, embedding: &[f32]) -> Result<()> {
    let collection_name = format!("products_{}", product.category);
    let collection = db.collection(&collection_name)?;
    collection.insert(&product.id, embedding, json!({"name": product.name}))?;
    Ok(())
}
```

### Multi-Tenancy

```rust
// Create collections per tenant
fn get_tenant_collection(db: &Database, tenant_id: &str) -> Result<CollectionRef> {
    let name = format!("tenant_{}", tenant_id);

    // Create if doesn't exist
    if !db.collection_exists(&name)? {
        db.create_collection(&name, 384, DistanceFunction::Cosine)?;
    }

    db.collection(&name)
}
```

## Best Practices

### 1. Choose Dimensions Wisely

The dimensionality affects memory usage and search performance:

```rust
// Memory per vector = dimensions * 4 bytes (f32)
// 384 dimensions = 1.5 KB per vector
// 1 million vectors = ~1.5 GB

// For 100M vectors, consider lower dimensions or quantization
```

### 2. Use Appropriate HNSW Parameters

```rust
// For high recall (quality-critical applications)
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(32)
    .with_hnsw_ef_construction(400);

// For lower memory (large-scale applications)
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_hnsw_m(8)
    .with_hnsw_ef_construction(100);
```

### 3. Index Only What You Need

Don't store unnecessary vectors:

```rust
// Filter out low-quality content before indexing
if document.word_count > 50 && document.quality_score > 0.7 {
    collection.insert(&document.id, &embedding, metadata)?;
}
```

### 4. Use Metadata Effectively

Store queryable attributes in metadata:

```rust
collection.insert("doc1", &embedding, json!({
    "title": "Article Title",
    "author": "John Doe",
    "date": "2024-01-15",
    "tags": ["rust", "programming"],
    "word_count": 1500
}))?;

// Then filter efficiently
let filter = Filter::parse(&json!({
    "tags": { "$in": ["rust"] },
    "word_count": { "$gte": 1000 }
}))?;
```

## Next Steps

- [HNSW Index](/docs/concepts/hnsw-index) - Understand the search algorithm
- [Metadata Filtering](/docs/concepts/metadata-filtering) - Filter search results
- [Quantization Guide](/docs/guides/quantization) - Reduce memory usage
