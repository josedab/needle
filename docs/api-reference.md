# Needle API Reference

Complete API reference for Needle vector database.

## Table of Contents

- [Database](#database)
- [Collection](#collection)
- [CollectionRef](#collectionref)
- [Search & Filtering](#search--filtering)
- [Distance Functions](#distance-functions)
- [Quantization](#quantization)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Error Handling](#error-handling)

---

## Database

The main entry point for Needle operations.

### Creating a Database

```rust
use needle::Database;

// In-memory database (no persistence)
let db = Database::in_memory();

// File-backed database
let db = Database::open("vectors.needle")?;

// With custom configuration
let config = DatabaseConfig {
    auto_save: true,
    save_interval: Duration::from_secs(60),
    ..Default::default()
};
let db = Database::with_config("vectors.needle", config)?;
```

### Database Methods

| Method | Description |
|--------|-------------|
| `create_collection(name, dimensions)` | Create a new collection |
| `create_collection_with_config(config)` | Create with custom config |
| `collection(name)` | Get a reference to a collection |
| `drop_collection(name)` | Delete a collection |
| `list_collections()` | List all collection names |
| `save()` | Persist to disk |
| `collection_exists(name)` | Check if collection exists |

### Example

```rust
let db = Database::in_memory();

// Create collection
db.create_collection("documents", 384)?;

// Get reference
let collection = db.collection("documents")?;

// Work with collection
collection.insert("doc1", &embedding, Some(metadata))?;

// Save to disk
db.save()?;
```

---

## Collection

Stores vectors with metadata and provides search capabilities.

### CollectionConfig

```rust
let config = CollectionConfig::new("my_collection", 384)
    .with_distance(DistanceFunction::Cosine)
    .with_hnsw_config(HnswConfig::default().m(32).ef_construction(400));

db.create_collection_with_config(config)?;
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | String | required | Collection name |
| `dimensions` | usize | required | Vector dimensions |
| `distance` | DistanceFunction | Cosine | Distance metric |
| `hnsw` | HnswConfig | default | HNSW parameters |

---

## CollectionRef

Thread-safe reference to a collection obtained from `Database::collection()`.

### Insert Operations

```rust
// Single insert
collection.insert("id", &vector, Some(metadata))?;

// Batch insert
let vectors = vec![
    ("id1", vec![0.1; 384], Some(json!({"category": "a"}))),
    ("id2", vec![0.2; 384], Some(json!({"category": "b"}))),
];
collection.insert_batch(&vectors)?;

// Update (insert with existing ID)
collection.update("id", &new_vector, Some(new_metadata))?;
```

### Retrieve Operations

```rust
// Get by ID
let (vector, metadata) = collection.get("id")?;

// Check existence
if collection.contains("id") {
    // ...
}

// Get all IDs
let ids = collection.all_ids();

// Iterate
for (id, vector, metadata) in collection.iter() {
    // ...
}
```

### Delete Operations

```rust
// Delete single
collection.delete("id")?;

// Clear all
collection.clear()?;

// Compact (reclaim deleted space)
collection.compact()?;
```

### Search Operations

```rust
// Basic search
let results = collection.search(&query_vector, 10)?;

// Search with filter
let filter = Filter::eq("category", "books");
let results = collection.search_with_filter(&query_vector, 10, &filter)?;

// Search with custom ef
let results = collection.search_with_ef(&query_vector, 10, 100)?;

// Using SearchBuilder
let results = collection
    .search_builder(&query_vector)
    .k(10)
    .filter(&filter)
    .ef_search(100)
    .execute()?;

// Batch search (parallel)
let queries = vec![query1, query2, query3];
let all_results = collection.batch_search(&queries, 10)?;
```

### Statistics

```rust
let stats = collection.stats();
println!("Count: {}", stats.count);
println!("Dimensions: {}", stats.dimensions);
println!("Memory: {} bytes", stats.memory_bytes);
```

---

## Search & Filtering

### SearchResult

```rust
pub struct SearchResult {
    pub id: String,           // Vector ID
    pub distance: f32,        // Distance to query
    pub metadata: Option<Value>, // Associated metadata
}
```

### Filter

MongoDB-style query filters for metadata.

```rust
use needle::Filter;
use serde_json::json;

// Equality
let filter = Filter::eq("category", "books");

// Comparison
let filter = Filter::gt("price", 10.0);
let filter = Filter::lte("rating", 5);

// Range
let filter = Filter::and(vec![
    Filter::gte("price", 10.0),
    Filter::lt("price", 50.0),
]);

// Membership
let filter = Filter::in_values("status", vec!["active", "pending"]);

// Logical operators
let filter = Filter::or(vec![
    Filter::eq("category", "books"),
    Filter::eq("category", "electronics"),
]);

let filter = Filter::not(Filter::eq("deleted", true));

// Parse from JSON
let filter = Filter::parse(&json!({
    "$and": [
        {"category": "books"},
        {"price": {"$lt": 50}}
    ]
}))?;
```

### Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal | `{"field": "value"}` |
| `$ne` | Not equal | `{"field": {"$ne": "value"}}` |
| `$gt` | Greater than | `{"field": {"$gt": 10}}` |
| `$gte` | Greater or equal | `{"field": {"$gte": 10}}` |
| `$lt` | Less than | `{"field": {"$lt": 10}}` |
| `$lte` | Less or equal | `{"field": {"$lte": 10}}` |
| `$in` | In array | `{"field": {"$in": [1, 2, 3]}}` |
| `$nin` | Not in array | `{"field": {"$nin": [1, 2]}}` |
| `$and` | Logical AND | `{"$and": [...]}` |
| `$or` | Logical OR | `{"$or": [...]}` |
| `$not` | Logical NOT | `{"$not": {...}}` |

---

## Distance Functions

```rust
use needle::DistanceFunction;

// Available functions
DistanceFunction::Cosine      // 1 - cos(θ), range [0, 2]
DistanceFunction::Euclidean   // L2 distance
DistanceFunction::DotProduct  // Negative dot product
DistanceFunction::Manhattan   // L1 distance

// Manual computation
let distance = DistanceFunction::Cosine.compute(&vec_a, &vec_b);
```

| Function | Best For | Normalized? |
|----------|----------|-------------|
| Cosine | Semantic similarity | No |
| Euclidean | Spatial distance | No |
| DotProduct | Pre-normalized vectors | Yes |
| Manhattan | Sparse vectors | No |

---

## Quantization

Reduce memory usage with quantization.

### Scalar Quantization

```rust
use needle::ScalarQuantizer;

// Train on sample vectors
let sq = ScalarQuantizer::train(&sample_vectors);

// Quantize
let quantized = sq.quantize(&vector);  // Vec<u8>

// Dequantize
let restored = sq.dequantize(&quantized);  // Vec<f32>
```

### Product Quantization

```rust
use needle::ProductQuantizer;

// Train with 8 subvectors
let pq = ProductQuantizer::train(&sample_vectors, 8);

// Encode
let codes = pq.encode(&vector);  // Vec<u8>

// Decode
let restored = pq.decode(&codes);  // Vec<f32>

// Asymmetric distance (faster for search)
let distance = pq.asymmetric_distance(&query, &codes);
```

---

## Configuration

### HnswConfig

```rust
let config = HnswConfig::builder()
    .m(16)              // Connections per node
    .m_max_0(32)        // Connections at layer 0
    .ef_construction(200)  // Construction search width
    .ef_search(50)      // Query search width
    .ml(1.0 / (16.0_f64).ln())  // Level multiplier
    .build();
```

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `m` | 16 | 4-64 | Higher = better recall, more memory |
| `m_max_0` | 32 | m-128 | Bottom layer connections |
| `ef_construction` | 200 | 50-500 | Higher = better index, slower build |
| `ef_search` | 50 | 10-500 | Higher = better recall, slower query |

### Auto-Tuning

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(1_000_000, 384)
    .with_profile(PerformanceProfile::Balanced)
    .with_memory_budget(4 * 1024 * 1024 * 1024);  // 4GB

let result = auto_tune(&constraints);
println!("Recommended M: {}", result.config.m);
println!("Expected recall: {:.2}%", result.expected_recall * 100.0);
```

---

## Advanced Features

### Sparse Vectors

```rust
use needle::{SparseVector, SparseIndex};

// Create sparse vector
let sparse = SparseVector::new(vec![0, 5, 10], vec![1.0, 0.5, 0.3]);

// Create index
let mut index = SparseIndex::new();
index.insert("doc1", &sparse);

// Search
let results = index.search(&query_sparse, 10);
```

### Multi-Vector (ColBERT)

```rust
use needle::{MultiVector, MultiVectorIndex};

// Create multi-vector document
let doc = MultiVector::new(vec![
    vec![0.1; 128],  // Token 1
    vec![0.2; 128],  // Token 2
    vec![0.3; 128],  // Token 3
]);

// MaxSim search
let mut index = MultiVectorIndex::new(128);
index.insert("doc1", &doc);
let results = index.search(&query_multi, 10);
```

### Hybrid Search (BM25 + Vector)

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

// Build BM25 index
let mut bm25 = Bm25Index::default();
bm25.index_document("doc1", "machine learning algorithms");

// Search both
let vector_results = collection.search(&query_vec, 100)?;
let bm25_results = bm25.search("machine learning", 100);

// Fuse with RRF
let config = RrfConfig::default();
let hybrid = reciprocal_rank_fusion(&vector_results, &bm25_results, &config, 10);
```

### IVF Index

```rust
use needle::{IvfIndex, IvfConfig};

// Create IVF index
let config = IvfConfig::new(256)  // 256 clusters
    .with_nprobe(16);             // Search 16 clusters

let mut index = IvfIndex::new(384, config);

// Train on sample data
index.train(&training_vectors)?;

// Insert and search
index.insert(0, &vector)?;
let results = index.search(&query, 10)?;
```

### Reranking

```rust
use needle::{CohereReranker, Reranker};

let reranker = CohereReranker::new("api-key", "rerank-english-v2.0");

let documents = vec!["doc1 text", "doc2 text", "doc3 text"];
let results = reranker.rerank("query", &documents, 2).await?;
```

---

## Error Handling

```rust
use needle::{NeedleError, Result};

fn example() -> Result<()> {
    let db = Database::open("test.needle")?;

    match db.collection("nonexistent") {
        Ok(coll) => { /* use collection */ }
        Err(NeedleError::CollectionNotFound(name)) => {
            println!("Collection {} not found", name);
        }
        Err(e) => return Err(e),
    }

    Ok(())
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `CollectionNotFound` | Collection doesn't exist |
| `CollectionExists` | Collection already exists |
| `DimensionMismatch` | Vector has wrong dimensions |
| `InvalidFilter` | Malformed filter expression |
| `IoError` | File system error |
| `SerializationError` | Data serialization failed |
| `IndexError` | HNSW index error |

---

## HTTP REST API

When compiled with `--features server`:

```bash
cargo run --features server -- serve -a 127.0.0.1:8080 -d vectors.needle
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/collections` | List collections |
| POST | `/collections` | Create collection |
| GET | `/collections/:name` | Get collection info |
| DELETE | `/collections/:name` | Delete collection |
| POST | `/collections/:name/vectors` | Insert vectors |
| GET | `/collections/:name/vectors/:id` | Get vector |
| DELETE | `/collections/:name/vectors/:id` | Delete vector |
| POST | `/collections/:name/search` | Search |
| POST | `/save` | Save database |

### Example Requests

```bash
# Create collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384}'

# Insert vector
curl -X POST http://localhost:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "doc1", "values": [0.1, ...], "metadata": {"title": "Hello"}}]}'

# Search
curl -X POST http://localhost:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, ...], "k": 10, "filter": {"title": "Hello"}}'
```

---

## Metrics

When compiled with `--features metrics`:

```rust
use needle::metrics;

// Get Prometheus metrics
let metrics_text = metrics().gather();
```

Available metrics:
- `needle_vectors_total` - Total vectors by collection
- `needle_search_duration_seconds` - Search latency histogram
- `needle_insert_duration_seconds` - Insert latency histogram
- `needle_memory_bytes` - Memory usage by collection
