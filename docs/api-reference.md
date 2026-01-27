# Needle API Reference

Complete API reference for Needle vector database.

## REST API (OpenAPI)

The HTTP server includes a built-in OpenAPI 3.1 spec available at `/openapi.json` when running:
```bash
cargo run --features server -- serve
# Then: curl http://localhost:8080/openapi.json
```

For a runnable walkthrough, see [HTTP Quickstart](http-quickstart.md).

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
- [MCP-Only Tools](#mcp-only-tools)

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
    path: PathBuf::from("vectors.needle"),
    create_if_missing: true,
    read_only: false,
};
let db = Database::open_with_config(config)?;
```

### Database Methods

| Method | Description |
|--------|-------------|
| `create_collection(name, dimensions)` | Create a new collection |
| `create_collection_with_config(config)` | Create with custom config |
| `collection(name)` | Get a reference to a collection |
| `delete_collection(name)` | Delete a collection |
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

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| **System** | | |
| GET | `/` | Root info (collection/vector counts) |
| GET | `/health` | Health check |
| GET | `/info` | Database info |
| GET | `/metrics` | Prometheus metrics |
| GET | `/openapi.json` | OpenAPI 3.1 specification |
| POST | `/save` | Persist database to disk |
| **Collections** | | |
| GET | `/collections` | List collections |
| POST | `/collections` | Create collection |
| GET | `/collections/:name` | Get collection info |
| DELETE | `/collections/:name` | Delete collection |
| POST | `/collections/:name/compact` | Compact collection (reclaim space) |
| GET | `/collections/:name/export` | Export collection data |
| POST | `/collections/:name/expire` | Expire TTL-based vectors |
| GET | `/collections/:name/ttl-stats` | TTL statistics |
| **Vectors** | | |
| GET | `/collections/:collection/vectors` | List vector IDs (paginated) |
| POST | `/collections/:collection/vectors` | Insert vectors |
| POST | `/collections/:collection/vectors/batch` | Batch insert vectors |
| POST | `/collections/:collection/vectors/upsert` | Upsert a vector |
| GET | `/collections/:collection/vectors/:id` | Get vector by ID |
| DELETE | `/collections/:collection/vectors/:id` | Delete vector |
| POST | `/collections/:collection/vectors/:id/metadata` | Update vector metadata |
| **Text (Auto-Embed)** | | |
| POST | `/collections/:collection/texts` | Insert text (auto-embedded) |
| POST | `/collections/:collection/texts/batch` | Batch insert texts |
| POST | `/collections/:collection/texts/search` | Search by text |
| **Search** | | |
| POST | `/collections/:collection/search` | Vector similarity search |
| POST | `/collections/:collection/search/batch` | Batch search (multiple queries) |
| POST | `/collections/:collection/search/radius` | Radius search (distance threshold) |
| POST | `/collections/:collection/search/graph` | Graph-augmented search |
| POST | `/collections/:collection/search/matryoshka` | Matryoshka coarse-to-fine search |
| POST | `/collections/:collection/search/time-travel` | Search against a snapshot |
| POST | `/collections/:collection/search/estimate` | Search cost estimation |
| **Semantic Cache** | | |
| POST | `/collections/:collection/cache/lookup` | Look up cached response |
| POST | `/collections/:collection/cache/store` | Store prompt/response in cache |
| **Streaming & Diff** | | |
| POST | `/collections/:collection/ingest` | Streaming vector ingest |
| POST | `/collections/:collection/diff` | Diff two collections |
| GET | `/collections/:collection/changes` | Change feed |
| **Benchmarks & Index** | | |
| POST | `/collections/:collection/benchmark` | Run search benchmark |
| GET | `/collections/:collection/index/status` | Index and WAL status |
| **Snapshots** | | |
| GET | `/collections/:name/snapshots` | List snapshots |
| POST | `/collections/:name/snapshots` | Create snapshot |
| POST | `/collections/:name/snapshots/:snapshot/restore` | Restore from snapshot |
| POST | `/collections/:collection/snapshots/diff` | Diff two snapshots |
| **Memory Protocol** | | |
| POST | `/collections/:collection/memory/remember` | Store a memory |
| POST | `/collections/:collection/memory/recall` | Recall memories by similarity |
| DELETE | `/collections/:collection/memory/:memory_id/forget` | Forget a memory |
| **Aliases** | | |
| POST | `/aliases` | Create alias |
| GET | `/aliases` | List all aliases |
| GET | `/aliases/:alias` | Get alias target |
| DELETE | `/aliases/:alias` | Delete alias |
| PUT | `/aliases/:alias` | Update alias target |
| **Webhooks** | | |
| POST | `/webhooks` | Register webhook |
| GET | `/webhooks` | List webhooks |
| DELETE | `/webhooks/:id` | Delete webhook |
| **Admin & Integration** | | |
| GET | `/embeddings/router/status` | Embedding router status |
| GET | `/cluster/status` | Cluster/shard status |
| GET | `/grpc/schema` | gRPC schema info |
| GET | `/tracing/status` | OpenTelemetry tracing status |
| POST | `/mcp` | MCP JSON-RPC endpoint |
| GET | `/mcp/config` | MCP client configuration |
| **UI** | | |
| GET | `/dashboard` | Admin dashboard (HTML) |
| GET | `/playground` | Interactive API playground (HTML) |

---

### System Endpoints

#### `GET /health`

Health check endpoint.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

#### `GET /` or `GET /info`

Database summary information.

```bash
curl http://localhost:8080/info
# {"collections":2,"total_vectors":1500}
```

#### `GET /metrics`

Prometheus-format metrics (requires `metrics` feature).

```bash
curl http://localhost:8080/metrics
# needle_vectors_total{collection="docs"} 1500
# needle_search_duration_seconds_bucket{le="0.01"} 42
```

#### `POST /save`

Persist all in-memory changes to disk.

```bash
curl -X POST http://localhost:8080/save
# {"saved":true}
```

#### `GET /openapi.json`

Returns the OpenAPI 3.1 specification for all endpoints.

```bash
curl http://localhost:8080/openapi.json -o openapi.json
```

---

### Collection Endpoints

#### `POST /collections`

Create a new collection.

**Request Body:**
```json
{
  "name": "docs",
  "dimension": 384,
  "distance": "cosine",
  "hnsw_config": {
    "m": 16,
    "ef_construction": 200
  }
}
```

```bash
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384}'
# {"created":"docs"}
```

#### `GET /collections`

List all collections.

```bash
curl http://localhost:8080/collections
# {"collections":["docs","images"]}
```

#### `GET /collections/:name`

Get collection info and statistics.

```bash
curl http://localhost:8080/collections/docs
# {"name":"docs","dimensions":384,"count":1500,"distance":"cosine"}
```

#### `DELETE /collections/:name`

Delete a collection and all its data.

```bash
curl -X DELETE http://localhost:8080/collections/docs
# {"deleted":"docs"}
```

#### `POST /collections/:name/compact`

Compact collection to reclaim space from deleted vectors.

```bash
curl -X POST http://localhost:8080/collections/docs/compact
# {"compacted":"docs","reclaimed_bytes":4096}
```

#### `GET /collections/:name/export`

Export all vectors and metadata as JSON.

```bash
curl http://localhost:8080/collections/docs/export
# {"vectors":[{"id":"doc1","values":[0.1,...],"metadata":{...}},...],"count":1500}
```

#### `POST /collections/:name/expire`

Remove expired TTL-based vectors.

```bash
curl -X POST http://localhost:8080/collections/docs/expire
# {"collection":"docs","expired_count":5}
```

#### `GET /collections/:name/ttl-stats`

Get TTL expiration statistics.

```bash
curl http://localhost:8080/collections/docs/ttl-stats
# {"collection":"docs","vectors_with_ttl":100,"expired_count":5,"earliest_expiration":"...","latest_expiration":"...","needs_sweep":true}
```

---

### Vector Endpoints

#### `POST /collections/:collection/vectors`

Insert one or more vectors.

**Request Body:**
```json
{
  "vectors": [
    {
      "id": "doc1",
      "values": [0.1, 0.2, 0.3],
      "metadata": {"title": "Hello World"}
    }
  ]
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "doc1", "values": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}}]}'
# {"inserted":1}
```

#### `POST /collections/:collection/vectors/batch`

Batch insert vectors.

**Request Body:**
```json
{
  "vectors": [
    {"id": "doc1", "values": [0.1, 0.2, 0.3], "metadata": {"type": "a"}},
    {"id": "doc2", "values": [0.4, 0.5, 0.6], "metadata": {"type": "b"}}
  ]
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/vectors/batch \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "d1", "values": [0.1,0.2,0.3]}, {"id": "d2", "values": [0.4,0.5,0.6]}]}'
# {"inserted":2,"errors":[]}
```

#### `POST /collections/:collection/vectors/upsert`

Insert or update a single vector. Returns whether the vector was updated (existed) or newly inserted.

**Request Body:**
```json
{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.3],
  "metadata": {"title": "Updated"},
  "ttl_seconds": 3600
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/vectors/upsert \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "vector": [0.1,0.2,0.3], "metadata": {"title": "Updated"}}'
# {"id":"doc1","updated":false}
```

#### `GET /collections/:collection/vectors`

List vector IDs with pagination.

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `offset` | integer | 0 | Starting offset |
| `limit` | integer | 100 | Max IDs to return |

```bash
curl "http://localhost:8080/collections/docs/vectors?offset=0&limit=10"
# {"ids":["doc1","doc2",...],"offset":0,"limit":10,"total":1500}
```

#### `GET /collections/:collection/vectors/:id`

Get a specific vector and its metadata.

```bash
curl http://localhost:8080/collections/docs/vectors/doc1
# {"id":"doc1","values":[0.1,0.2,0.3],"metadata":{"title":"Hello"}}
```

#### `DELETE /collections/:collection/vectors/:id`

Delete a vector by ID.

```bash
curl -X DELETE http://localhost:8080/collections/docs/vectors/doc1
# {"deleted":"doc1"}
```

#### `POST /collections/:collection/vectors/:id/metadata`

Update only the metadata of a vector (without re-indexing the vector).

**Request Body:**
```json
{
  "metadata": {"title": "New Title", "category": "updated"}
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/vectors/doc1/metadata \
  -H "Content-Type: application/json" \
  -d '{"metadata": {"title": "New Title"}}'
# {"updated":"doc1"}
```

---

### Text Endpoints (Auto-Embed)

These endpoints automatically generate embeddings from text using the configured embedding provider.

#### `POST /collections/:collection/texts`

Insert a text document (auto-embedded).

**Request Body:**
```json
{
  "id": "doc1",
  "text": "Machine learning is a subset of artificial intelligence.",
  "metadata": {"source": "textbook"}
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/texts \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "text": "Machine learning is a subset of AI.", "metadata": {"source": "textbook"}}'
# {"id":"doc1","dimensions":384,"text_length":35,"embed_method":"auto"}
```

#### `POST /collections/:collection/texts/batch`

Batch insert text documents.

**Request Body:**
```json
{
  "texts": [
    {"id": "doc1", "text": "First document", "metadata": {}},
    {"id": "doc2", "text": "Second document", "metadata": {}}
  ]
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/texts/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": [{"id": "d1", "text": "Hello"}, {"id": "d2", "text": "World"}]}'
# {"inserted":2,"total":2,"errors":[],"embed_method":"auto"}
```

#### `POST /collections/:collection/texts/search`

Search by text query (auto-embedded).

**Request Body:**
```json
{
  "text": "artificial intelligence",
  "k": 10,
  "filter": {"source": "textbook"}
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/texts/search \
  -H "Content-Type: application/json" \
  -d '{"text": "artificial intelligence", "k": 5}'
# {"results":[{"id":"doc1","distance":0.12,"score":0.88,"text":"...","metadata":{}}],"count":1}
```

---

### Search Endpoints

#### `POST /collections/:collection/search`

Vector similarity search with optional filtering and explanation.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "filter": {"category": "books"},
  "post_filter": {"price": {"$lt": 50}},
  "post_filter_factor": 3,
  "include_vectors": false,
  "explain": false,
  "distance": "cosine"
}
```

**Response:**
```json
{
  "results": [
    {"id": "doc1", "distance": 0.05, "metadata": {"category": "books"}}
  ],
  "explanation": null
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 10, "filter": {"category": "books"}}'
```

#### `POST /collections/:collection/search/batch`

Execute multiple search queries in parallel.

**Request Body:**
```json
{
  "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
  "k": 10,
  "filter": {"category": "books"}
}
```

**Response:**
```json
{
  "results": [
    [{"id": "doc1", "distance": 0.05, "metadata": {}}],
    [{"id": "doc2", "distance": 0.12, "metadata": {}}]
  ]
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search/batch \
  -H "Content-Type: application/json" \
  -d '{"vectors": [[0.1,0.2,0.3],[0.4,0.5,0.6]], "k": 5}'
```

#### `POST /collections/:collection/search/radius`

Find all vectors within a distance threshold.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "max_distance": 0.5,
  "limit": 100,
  "filter": {},
  "include_vectors": false
}
```

**Response:**
```json
{
  "results": [{"id": "doc1", "distance": 0.05, "metadata": {}}],
  "max_distance": 0.5,
  "count": 1
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search/radius \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "max_distance": 0.5}'
```

#### `POST /collections/:collection/search/graph`

Graph-augmented search that traverses connections between vectors.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "max_hops": 3
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "doc1",
      "name": "doc1",
      "vector_score": 0.95,
      "graph_score": 0.8,
      "combined_score": 0.88,
      "hop_count": 1,
      "path": ["entry", "doc1"],
      "properties": {}
    }
  ],
  "count": 1
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search/graph \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "k": 5, "max_hops": 2}'
```

#### `POST /collections/:collection/search/matryoshka`

Matryoshka (coarse-to-fine) search. First searches with truncated dimensions for speed, then re-ranks with full dimensions.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "coarse_dims": 64,
  "oversample": 4,
  "include_vectors": false
}
```

**Response:**
```json
{
  "results": [{"id": "doc1", "distance": 0.05, "metadata": {}}],
  "count": 1,
  "coarse_dims": 64,
  "oversample": 4
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search/matryoshka \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "k": 10, "coarse_dims": 64}'
```

#### `POST /collections/:collection/search/time-travel`

Search against a named snapshot of the collection.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "snapshot": "2024-01-01-backup"
}
```

**Response:**
```json
{
  "results": [{"id": "doc1", "distance": 0.05, "metadata": {}}],
  "count": 1,
  "snapshot": "2024-01-01-backup",
  "note": "..."
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search/time-travel \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "k": 5, "snapshot": "2024-01-01-backup"}'
```

#### `POST /collections/:collection/search/estimate`

Estimate query cost and suggest alternative plans without executing a search.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "filter": {"category": "books"},
  "ef_search": 100
}
```

**Response:**
```json
{
  "collection": "docs",
  "query_dimensions": 3,
  "collection_vectors": 1500,
  "plan": {...},
  "alternatives": [...]
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/search/estimate \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "k": 10}'
```

---

### Semantic Cache Endpoints

#### `POST /collections/:collection/cache/lookup`

Look up a cached response by vector similarity.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "threshold": 0.95
}
```

**Response:**
```json
{
  "hit": true,
  "message": "cached response text",
  "stats": {"lookups": 42, "hits": 30}
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/cache/lookup \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "threshold": 0.95}'
```

#### `POST /collections/:collection/cache/store`

Store a prompt/response pair in the semantic cache.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "response": "The answer is 42.",
  "model": "gpt-4",
  "ttl_seconds": 3600
}
```

**Response:**
```json
{
  "stored": true,
  "collection": "docs",
  "model": "gpt-4",
  "response_length": 17,
  "ttl_seconds": 3600
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/cache/store \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "response": "The answer is 42.", "model": "gpt-4"}'
```

---

### Streaming & Diff Endpoints

#### `POST /collections/:collection/ingest`

Streaming vector ingest with backpressure support.

**Request Body:**
```json
{
  "vectors": [
    {"id": "v1", "values": [0.1, 0.2, 0.3], "metadata": {}},
    {"id": "v2", "values": [0.4, 0.5, 0.6], "metadata": {}}
  ],
  "sequence_id": "seq-001",
  "flush": false
}
```

**Response:**
```json
{
  "accepted": 2,
  "total": 2,
  "errors": [],
  "sequence_id": "seq-001",
  "flushed": false,
  "latency_ms": 5,
  "backpressure": false,
  "collection_size": 1502
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/ingest \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "v1", "values": [0.1,0.2,0.3]}], "flush": true}'
```

#### `POST /collections/:collection/diff`

Compute the diff between two collections.

**Request Body:**
```json
{
  "other_collection": "docs_v2",
  "limit": 100
}
```

**Response:**
```json
{
  "source": "docs",
  "target": "docs_v2",
  "source_count": 1500,
  "target_count": 1600,
  "only_in_source": ["old_doc"],
  "only_in_target": ["new_doc"],
  "modified": ["doc1"],
  "shared_count": 1499,
  "summary": "..."
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/diff \
  -H "Content-Type: application/json" \
  -d '{"other_collection": "docs_v2"}'
```

#### `GET /collections/:collection/changes`

Get a change feed for the collection.

| Query Param | Type | Default | Description |
|-------------|------|---------|-------------|
| `limit` | integer | 100 | Max events |
| `after` | string | — | Return events after this cursor |
| `event_type` | string | — | Filter by event type |

```bash
curl "http://localhost:8080/collections/docs/changes?limit=50"
# {"collection":"docs","vector_count":1500,"feed_config":{...}}
```

---

### Benchmark & Index Endpoints

#### `POST /collections/:collection/benchmark`

Run a search latency benchmark on the collection.

**Request Body:**
```json
{
  "num_queries": 100,
  "k": 10
}
```

**Response:**
```json
{
  "collection": "docs",
  "vectors": 1500,
  "dimensions": 384,
  "k": 10,
  "queries": 100,
  "latency_us": {"p50": 120, "p95": 450, "p99": 800},
  "throughput_qps": 8300
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/benchmark \
  -H "Content-Type: application/json" \
  -d '{"num_queries": 100, "k": 10}'
```

#### `GET /collections/:collection/index/status`

Get HNSW index health and WAL status.

```bash
curl http://localhost:8080/collections/docs/index/status
# {"collection":"docs","index":{"type":"hnsw","layers":4,"entry_point":0},"wal":{},"compaction_recommended":false}
```

---

### Snapshot Endpoints

#### `GET /collections/:name/snapshots`

List available snapshots.

```bash
curl http://localhost:8080/collections/docs/snapshots
# {"snapshots":["2024-01-01-backup","2024-02-01-backup"]}
```

#### `POST /collections/:name/snapshots`

Create a named snapshot.

**Request Body:**
```json
{
  "name": "before-migration"
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/snapshots \
  -H "Content-Type: application/json" \
  -d '{"name": "before-migration"}'
# {"created":true,"name":"before-migration"}
```

#### `POST /collections/:name/snapshots/:snapshot/restore`

Restore a collection from a snapshot.

```bash
curl -X POST http://localhost:8080/collections/docs/snapshots/before-migration/restore
# {"restored":true}
```

#### `POST /collections/:collection/snapshots/diff`

Diff two snapshots.

**Request Body:**
```json
{
  "from": "snapshot-a",
  "to": "snapshot-b"
}
```

**Response:**
```json
{
  "collection": "docs",
  "from": "snapshot-a",
  "to": "snapshot-b",
  "current_vector_count": 1500,
  "available_snapshots": ["snapshot-a", "snapshot-b"],
  "note": "..."
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/snapshots/diff \
  -H "Content-Type: application/json" \
  -d '{"from": "snapshot-a", "to": "snapshot-b"}'
```

---

### Memory Protocol Endpoints

Agentic memory with tiered storage (episodic, semantic, procedural).

#### `POST /collections/:collection/memory/remember`

Store a memory with optional tier and importance.

**Request Body:**
```json
{
  "content": "The user prefers dark mode.",
  "vector": [0.1, 0.2, 0.3],
  "tier": "semantic",
  "importance": 0.8,
  "session_id": "session-123",
  "metadata": {"source": "preferences"}
}
```

**Response:**
```json
{
  "stored": true,
  "memory_id": "mem_abc123",
  "tier": "semantic",
  "importance": 0.8
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/memory/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "vector": [0.1,0.2,0.3], "importance": 0.8}'
```

#### `POST /collections/:collection/memory/recall`

Recall memories by vector similarity with optional filters.

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 5,
  "tier": "semantic",
  "session_id": "session-123",
  "min_importance": 0.5
}
```

**Response:**
```json
{
  "memories": [
    {
      "memory_id": "mem_abc123",
      "distance": 0.05,
      "relevance_score": 0.95,
      "content": "The user prefers dark mode.",
      "tier": "semantic",
      "importance": 0.8,
      "timestamp": "2024-01-15T10:30:00Z",
      "session_id": "session-123"
    }
  ],
  "count": 1
}
```

```bash
curl -X POST http://localhost:8080/collections/docs/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1,0.2,0.3], "k": 5, "min_importance": 0.5}'
```

#### `DELETE /collections/:collection/memory/:memory_id/forget`

Delete a specific memory.

```bash
curl -X DELETE http://localhost:8080/collections/docs/memory/mem_abc123/forget
# {"forgotten":true,"memory_id":"mem_abc123"}
```

---

### Alias Endpoints

Aliases let you refer to collections by alternative names (useful for blue-green deployments).

#### `POST /aliases`

Create an alias pointing to a collection.

**Request Body:**
```json
{
  "alias": "production",
  "collection": "docs_v2"
}
```

```bash
curl -X POST http://localhost:8080/aliases \
  -H "Content-Type: application/json" \
  -d '{"alias": "production", "collection": "docs_v2"}'
# {"created":true,"alias":"production","collection":"docs_v2"}
```

#### `GET /aliases`

List all aliases.

```bash
curl http://localhost:8080/aliases
# {"aliases":[{"alias":"production","collection":"docs_v2"}]}
```

#### `GET /aliases/:alias`

Get the target collection for an alias.

```bash
curl http://localhost:8080/aliases/production
# {"alias":"production","collection":"docs_v2"}
```

#### `PUT /aliases/:alias`

Update an alias to point to a different collection.

**Request Body:**
```json
{
  "collection": "docs_v3"
}
```

```bash
curl -X PUT http://localhost:8080/aliases/production \
  -H "Content-Type: application/json" \
  -d '{"collection": "docs_v3"}'
# {"updated":true,"alias":"production","collection":"docs_v3"}
```

#### `DELETE /aliases/:alias`

Delete an alias.

```bash
curl -X DELETE http://localhost:8080/aliases/production
# {"deleted":true,"alias":"production"}
```

---

### Webhook Endpoints

#### `POST /webhooks`

Register a webhook for collection events.

**Request Body:**
```json
{
  "url": "https://example.com/webhook",
  "secret": "whsec_abc123",
  "collections": ["docs"],
  "event_types": ["insert", "delete", "search"]
}
```

**Response:**
```json
{
  "id": "wh_abc123",
  "url": "https://example.com/webhook",
  "active": true,
  "note": "Webhook registered"
}
```

```bash
curl -X POST http://localhost:8080/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/hook", "collections": ["docs"], "event_types": ["insert"]}'
```

#### `GET /webhooks`

List all registered webhooks.

```bash
curl http://localhost:8080/webhooks
# {"webhooks":[],"note":"..."}
```

#### `DELETE /webhooks/:id`

Delete a webhook.

```bash
curl -X DELETE http://localhost:8080/webhooks/wh_abc123
# {"deleted":true,"id":"wh_abc123"}
```

---

### Admin & Integration Endpoints

#### `GET /embeddings/router/status`

Get the status of the embedding provider router.

```bash
curl http://localhost:8080/embeddings/router/status
# {"router":"active","providers":[...],"collection_pins":{},"configuration":{}}
```

#### `GET /cluster/status`

Get cluster and shard information.

```bash
curl http://localhost:8080/cluster/status
# {"cluster":"standalone","shards":1,"total_collections":2,"replication_factor":1,"note":"..."}
```

#### `GET /grpc/schema`

Get gRPC service schema information.

```bash
curl http://localhost:8080/grpc/schema
# {"schema_version":"1.0","services":[...],"hint":"..."}
```

#### `GET /tracing/status`

Get OpenTelemetry tracing configuration status.

```bash
curl http://localhost:8080/tracing/status
# {"tracing":"enabled","instrumented_operations":[...],"configuration":{}}
```

#### `POST /mcp`

JSON-RPC endpoint for Model Context Protocol integration.

**Request Body:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 1
}
```

```bash
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'
```

#### `GET /mcp/config`

Get MCP client configuration (e.g., for Claude Desktop).

```bash
curl http://localhost:8080/mcp/config
# {"mcpServers":{"needle":{"command":"needle","args":["mcp","--database","vectors.needle"]}}}
```

### Example Requests

```bash
# Create collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimension": 384}'

# Insert vector
curl -X POST http://localhost:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"vectors": [{"id": "doc1", "values": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}}]}'

# Search
curl -X POST http://localhost:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 10, "filter": {"title": "Hello"}}'
```

---

## MCP-Only Tools

The following tools are available exclusively via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)
server and do not have HTTP REST equivalents. Start the MCP server with:

```bash
needle mcp --database vectors.needle
```

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with content, embedding vector, tier (episodic/semantic/procedural), and importance score |
| `recall` | Retrieve relevant memories by vector similarity, with optional tier and importance filters |
| `forget` | Delete a specific memory by its ID |
| `memory_consolidate` | Promote important episodic memories to semantic tier and expire low-importance entries |
| `save_database` | Persist all in-memory changes to disk |

### memory_consolidate

Consolidates a memory collection by scanning episodic-tier memories and:
- **Promoting** memories with importance ≥ `promotion_threshold` (default: 0.7) to the semantic tier
- **Expiring** memories with importance < `expire_below` (default: 0.1)

Parameters:
- `collection` (required) — Name of the memory collection
- `promotion_threshold` (optional, default: 0.7) — Importance threshold for promotion (0.0–1.0)
- `expire_below` (optional, default: 0.1) — Forget memories below this importance (0.0–1.0)

Response includes `scanned`, `promoted`, `forgotten`, and `errors` counts.

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

---

## Cloud Storage

Abstracted storage backends for distributed deployments.

### Storage Backends

```rust
use needle::{S3Backend, S3Config, StorageBackend};

// S3-compatible storage
let config = S3Config::new("my-bucket", "us-east-1")
    .with_endpoint("http://localhost:9000")  // MinIO
    .with_credentials("access_key", "secret_key");

let backend = S3Backend::new(config).await?;

// Basic operations (StorageBackend trait)
backend.write("vectors/v1", &data).await?;
let data = backend.read("vectors/v1").await?;
backend.delete("vectors/v1").await?;
let keys = backend.list("vectors/").await?;
let exists = backend.exists("vectors/v1").await?;
```

### Available Backends

| Backend | Description |
|---------|-------------|
| `S3Backend` | Amazon S3, MinIO, DigitalOcean Spaces |
| `AzureBlobBackend` | Azure Blob Storage |
| `GCSBackend` | Google Cloud Storage |
| `LocalBackend` | Local filesystem |
| `CachedBackend` | Caching wrapper for any backend |

### Caching

```rust
use needle::{CachedBackend, CacheConfig};

let cache_config = CacheConfig::new()
    .with_max_size(1024 * 1024 * 100)  // 100MB cache
    .with_ttl(Duration::from_secs(3600));

let cached = CachedBackend::new(backend, cache_config);
```

---

## Write-Ahead Log (WAL)

Crash recovery and durability guarantees.

```rust
use needle::{WalManager, WalConfig};

// Create WAL manager
let config = WalConfig::default()
    .with_sync_on_write(true)
    .with_max_segment_size(64 * 1024 * 1024);  // 64MB

let wal = WalManager::open("/path/to/wal", config)?;

// Append operations
let lsn = wal.append(WalRecord::Insert {
    collection: "docs".into(),
    id: "doc1".into(),
    vector: vec![0.1; 384],
})?;

// Checkpoint to main storage
wal.checkpoint()?;

// Recovery after crash
let entries = wal.recover(last_known_lsn)?;
```

---

## Sharding

Horizontal scaling with consistent hashing.

```rust
use needle::{ShardManager, ShardConfig, ConsistentHashRing};

// Create shard manager
let config = ShardConfig::new(4)  // 4 shards
    .with_replication(3);         // replication factor of 3

let manager = ShardManager::new(config)?;

// Route operations to shards
let shard_id = manager.get_shard_for_key("doc123");

// Rebalancing
let moves = manager.plan_rebalance()?;
manager.execute_rebalance(moves).await?;
```

### Query Routing

```rust
use needle::{QueryRouter, RouteConfig, LoadBalancing};

let config = RouteConfig::new()
    .with_load_balancing(LoadBalancing::LeastConnections)
    .with_timeout(Duration::from_secs(30));

let router = QueryRouter::new(shards, config);

// Fan-out search to all shards
let results = router.search(&query, k).await?;
```

---

## Multi-Tenancy (Namespaces)

Logical isolation for multi-tenant deployments.

```rust
use needle::{NamespaceManager, Namespace, TenantConfig, AccessLevel};

// Create namespace manager
let mut manager = NamespaceManager::new();

// Create tenant namespace
let config = TenantConfig::new()
    .with_max_vectors(1_000_000)
    .with_max_collections(10);

manager.create_namespace("tenant_123", config)?;

// Get tenant-scoped collection
let ns = manager.namespace("tenant_123")?;
let collection = ns.collection("documents")?;

// Access control
ns.grant_access("user@example.com", AccessLevel::ReadWrite)?;
```

---

## Security & RBAC

Role-based access control with audit logging.

### Access Control

```rust
use needle::{AccessController, Role, Permission, Resource, User};

let mut controller = AccessController::new();

// Define roles
let reader_role = Role::new("reader")
    .with_permission(Permission::Read, Resource::Collection("*".into()));

let writer_role = Role::new("writer")
    .with_permission(Permission::Read, Resource::Collection("*".into()))
    .with_permission(Permission::Write, Resource::Collection("*".into()));

controller.add_role(reader_role)?;
controller.add_role(writer_role)?;

// Assign roles to users
controller.assign_role("user@example.com", "reader")?;

// Check permissions
let ctx = SecurityContext::new("user@example.com");
let decision = controller.check_permission(&ctx, Permission::Write, &resource);
```

### Audit Logging

```rust
use needle::{AuditLogger, FileAuditLog, AuditQuery};

// Create audit logger
let log = FileAuditLog::new("/var/log/needle/audit.json")?;
let logger = AuditLogger::new(Box::new(log));

// Events are logged automatically, query later:
let query = AuditQuery::new()
    .with_action(AuditAction::Search)
    .with_resource("documents")
    .with_time_range(start, end);

let events = logger.query(&query)?;
```

---

## Encryption

Encrypt vectors at rest using ChaCha20-Poly1305.

```rust
use needle::{VectorEncryptor, EncryptionConfig, KeyManager};

// Set up encryption (master key must be at least 32 bytes)
let master_key: [u8; 32] = /* your 32-byte key */;
let key_manager = KeyManager::new(&master_key)?;

let config = EncryptionConfig::new()
    .with_algorithm(Algorithm::ChaCha20Poly1305);

let encryptor = VectorEncryptor::new(key_manager, config);

// Encrypt/decrypt vectors
let encrypted = encryptor.encrypt("doc1", &vector)?;
let decrypted = encryptor.decrypt("doc1", &encrypted)?;
```

---

## Backup & Restore

```rust
use needle::{BackupManager, BackupConfig, BackupType};

let config = BackupConfig::new("/backups")
    .with_compression(true)
    .with_encryption(Some(encryption_key));

let manager = BackupManager::new(config);

// Create backup
let metadata = manager.create_backup(&db, BackupType::Full)?;
println!("Backup created: {}", metadata.id);

// List backups
let backups = manager.list_backups()?;

// Restore
manager.restore(&metadata.id, "/path/to/restore")?;
```

---

## Query Language (NeedleQL)

SQL-like query language for complex operations.

```rust
use needle::{QueryParser, QueryExecutor};

let parser = QueryParser::new();
let executor = QueryExecutor::new(&db);

// Parse and execute queries
let query = parser.parse(r#"
    SELECT * FROM documents
    WHERE vector SIMILAR TO $query AND category = 'tech'
    ORDER BY distance ASC
    LIMIT 10
"#)?;

let results = executor.execute(&query)?;
```

### Query Syntax

```sql
-- Vector similarity search
SELECT * FROM <collection>
WHERE vector SIMILAR TO <query_vector>
[AND <filter_expression>]
[ORDER BY distance ASC|DESC]
[LIMIT <n>]

-- With metadata filters
SELECT * FROM documents
WHERE vector SIMILAR TO $query
  AND category = 'tech'
  AND score > 0.5
LIMIT 10

-- Aggregations
SELECT COUNT(*), AVG(score)
FROM documents
WHERE category = 'tech'
GROUP BY author
```

---

## Embedding Providers

When compiled with `--features embedding-providers`:

```rust
use needle::{OpenAIProvider, OpenAIConfig, EmbeddingProvider};

// OpenAI
let config = OpenAIConfig::new("sk-...")
    .with_model("text-embedding-3-small");
let provider = OpenAIProvider::new(config);

let embeddings = provider.embed(&["Hello world", "How are you?"]).await?;

// Cohere
use needle::{CohereProvider, CohereConfig};
let provider = CohereProvider::new(CohereConfig::new("api-key"));

// Ollama (local)
use needle::{OllamaProvider, OllamaConfig};
let provider = OllamaProvider::new(
    OllamaConfig::new("http://localhost:11434", "nomic-embed-text")
);
```

---

## GPU Acceleration

```rust
use needle::{GpuAccelerator, GpuConfig, GpuBackend};

let config = GpuConfig::new()
    .with_backend(GpuBackend::CUDA)
    .with_device(0);

let gpu = GpuAccelerator::new(config)?;

// Batch distance computation
let distances = gpu.batch_distances(&query, &vectors)?;

// GPU-accelerated search
let results = gpu.knn_search(&query, &vectors, k)?;
```

---

## Clustering & Analytics

```rust
use needle::{KMeans, ClusteringConfig, silhouette_score};

// K-means clustering
let config = ClusteringConfig::default()
    .with_max_iterations(100)
    .with_tolerance(1e-4);

// Fit K-means with k=10 clusters
let vectors_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
let kmeans = KMeans::fit(&vectors_refs, 10, config)?;

// Get cluster assignments
let labels = kmeans.predict(&vectors_refs);

// Evaluate clustering
let score = silhouette_score(&vectors, &labels);
println!("Silhouette score: {:.3}", score);

// Find optimal k with elbow method
let optimal_k = elbow_method(&vectors, 2..20)?;
```

### Anomaly Detection

```rust
use needle::{IsolationForest, LocalOutlierFactor};

// Isolation Forest
let forest = IsolationForest::new(100);  // 100 trees
forest.fit(&vectors)?;
let anomaly_scores = forest.predict(&new_vectors)?;

// Local Outlier Factor
let lof = LocalOutlierFactor::new(20);  // k=20 neighbors
let outlier_scores = lof.fit_predict(&vectors)?;
```

---

## Complete Method Reference

### Database Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `open` | `fn open<P: AsRef<Path>>(path: P) -> Result<Self>` | Open or create a database file |
| `open_with_config` | `fn open_with_config(config: DatabaseConfig) -> Result<Self>` | Open with custom configuration |
| `in_memory` | `fn in_memory() -> Self` | Create an ephemeral in-memory database |
| `path` | `fn path(&self) -> Option<&Path>` | Get database file path (None for in-memory) |
| `create_collection` | `fn create_collection(&self, name: &str, dimensions: usize) -> Result<()>` | Create collection with default settings |
| `create_collection_with_config` | `fn create_collection_with_config(&self, config: CollectionConfig) -> Result<()>` | Create with custom configuration |
| `collection` | `fn collection(&self, name: &str) -> Result<CollectionRef>` | Get thread-safe collection reference |
| `list_collections` | `fn list_collections(&self) -> Vec<String>` | List all collection names |
| `delete_collection` | `fn delete_collection(&self, name: &str) -> Result<bool>` | Delete collection, returns true if existed |
| `has_collection` | `fn has_collection(&self, name: &str) -> bool` | Check if collection exists |
| `create_alias` | `fn create_alias(&self, alias: &str, collection: &str) -> Result<()>` | Create alias for collection |
| `delete_alias` | `fn delete_alias(&self, alias: &str) -> Result<bool>` | Remove alias |
| `update_alias` | `fn update_alias(&self, alias: &str, collection: &str) -> Result<()>` | Change alias target |
| `list_aliases` | `fn list_aliases(&self) -> Vec<(String, String)>` | List all (alias, collection) pairs |
| `get_canonical_name` | `fn get_canonical_name(&self, alias: &str) -> Option<String>` | Resolve alias to collection name |
| `aliases_for_collection` | `fn aliases_for_collection(&self, collection: &str) -> Vec<String>` | Get all aliases pointing to collection |
| `save` | `fn save(&mut self) -> Result<()>` | Persist changes to disk |
| `is_dirty` | `fn is_dirty(&self) -> bool` | Check if there are unsaved changes |

### CollectionRef Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `insert` | `fn insert(&self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()>` | Insert or update vector |
| `insert_batch` | `fn insert_batch(&self, items: &[(&str, &[f32], Option<Value>)]) -> Result<()>` | Insert multiple vectors efficiently |
| `insert_with_ttl` | `fn insert_with_ttl(&self, id: &str, vector: &[f32], metadata: Option<Value>, ttl: Duration) -> Result<()>` | Insert with time-to-live |
| `get` | `fn get(&self, id: &str) -> Option<(Vec<f32>, Option<Value>)>` | Get vector and metadata by ID |
| `get_metadata` | `fn get_metadata(&self, id: &str) -> Option<Value>` | Get only metadata by ID |
| `contains` | `fn contains(&self, id: &str) -> bool` | Check if vector exists |
| `delete` | `fn delete(&self, id: &str) -> Result<bool>` | Delete vector, returns true if existed |
| `update_metadata` | `fn update_metadata(&self, id: &str, metadata: Value) -> Result<()>` | Update only metadata |
| `search` | `fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>` | Basic k-NN search |
| `search_with_filter` | `fn search_with_filter(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<SearchResult>>` | Search with metadata filter |
| `search_with_ef` | `fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>>` | Search with custom ef_search |
| `search_builder` | `fn search_builder(&self, query: &[f32]) -> SearchBuilder` | Get fluent search builder |
| `search_explain` | `fn search_explain(&self, query: &[f32], k: usize) -> Result<(Vec<SearchResult>, SearchExplain)>` | Search with timing breakdown |
| `batch_search` | `fn batch_search(&self, queries: &[&[f32]], k: usize) -> Result<Vec<Vec<SearchResult>>>` | Parallel multi-query search |
| `len` | `fn len(&self) -> usize` | Number of vectors |
| `is_empty` | `fn is_empty(&self) -> bool` | Check if collection has no vectors |
| `dimensions` | `fn dimensions(&self) -> usize` | Vector dimensionality |
| `stats` | `fn stats(&self) -> CollectionStats` | Get detailed statistics |
| `iter` | `fn iter(&self) -> CollectionIter` | Iterate over all vectors |
| `compact` | `fn compact(&self) -> Result<CompactStats>` | Reclaim space from deletions |
| `clear` | `fn clear(&self) -> Result<()>` | Remove all vectors |
| `export` | `fn export(&self) -> Result<Vec<ExportEntry>>` | Export all data for backup |

### SearchBuilder Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `k` | `fn k(self, k: usize) -> Self` | Set number of results to return |
| `filter` | `fn filter(self, filter: &Filter) -> Self` | Set pre-filter (applied during search) |
| `post_filter` | `fn post_filter(self, filter: &Filter) -> Self` | Set post-filter (applied after search) |
| `post_filter_factor` | `fn post_filter_factor(self, factor: usize) -> Self` | Over-fetch factor for post-filtering |
| `ef_search` | `fn ef_search(self, ef: usize) -> Self` | Override ef_search parameter |
| `include_metadata` | `fn include_metadata(self, include: bool) -> Self` | Whether to include metadata in results |
| `distance` | `fn distance(self, d: DistanceFunction) -> Self` | Override distance function (falls back to brute-force) |
| `execute` | `fn execute(self) -> Result<Vec<SearchResult>>` | Execute the search |

### Filter Construction

| Method | Signature | Description |
|--------|-----------|-------------|
| `eq` | `fn eq(field: &str, value: impl Into<Value>) -> Self` | Equality: field == value |
| `ne` | `fn ne(field: &str, value: impl Into<Value>) -> Self` | Not equal: field != value |
| `gt` | `fn gt(field: &str, value: impl Into<Value>) -> Self` | Greater than: field > value |
| `gte` | `fn gte(field: &str, value: impl Into<Value>) -> Self` | Greater or equal: field >= value |
| `lt` | `fn lt(field: &str, value: impl Into<Value>) -> Self` | Less than: field < value |
| `lte` | `fn lte(field: &str, value: impl Into<Value>) -> Self` | Less or equal: field <= value |
| `in_values` | `fn in_values(field: &str, values: Vec<impl Into<Value>>) -> Self` | Membership: field in [...] |
| `nin_values` | `fn nin_values(field: &str, values: Vec<impl Into<Value>>) -> Self` | Not in: field not in [...] |
| `and` | `fn and(filters: Vec<Filter>) -> Self` | Logical AND of filters |
| `or` | `fn or(filters: Vec<Filter>) -> Self` | Logical OR of filters |
| `not` | `fn not(filter: Filter) -> Self` | Logical NOT of filter |
| `parse` | `fn parse(json: &Value) -> Result<Self>` | Parse MongoDB-style JSON filter |
| `matches` | `fn matches(&self, metadata: Option<&Value>) -> bool` | Test if metadata matches filter |

### HnswConfig Builder

| Method | Signature | Description |
|--------|-----------|-------------|
| `m` | `fn m(self, m: usize) -> Self` | Max connections per node (default: 16) |
| `m_max_0` | `fn m_max_0(self, m: usize) -> Self` | Max connections at layer 0 (default: 32) |
| `ef_construction` | `fn ef_construction(self, ef: usize) -> Self` | Build-time search width (default: 200) |
| `ef_search` | `fn ef_search(self, ef: usize) -> Self` | Query-time search width (default: 50) |
| `ml` | `fn ml(self, ml: f64) -> Self` | Level multiplier (default: 1/ln(M)) |

### CollectionConfig Builder

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(name: &str, dimensions: usize) -> Self` | Create with name and dimensions |
| `with_distance` | `fn with_distance(self, d: DistanceFunction) -> Self` | Set distance function |
| `with_hnsw_config` | `fn with_hnsw_config(self, config: HnswConfig) -> Self` | Set HNSW parameters |
| `with_m` | `fn with_m(self, m: usize) -> Self` | Shorthand for HNSW M parameter |
| `with_ef_construction` | `fn with_ef_construction(self, ef: usize) -> Self` | Shorthand for ef_construction |
| `with_query_cache` | `fn with_query_cache(self, config: QueryCacheConfig) -> Self` | Enable query result caching |
| `with_lazy_expiration` | `fn with_lazy_expiration(self, enabled: bool) -> Self` | Enable lazy TTL expiration |

---

## Return Types

### SearchResult

```rust
pub struct SearchResult {
    /// The unique identifier of the vector
    pub id: String,

    /// Distance from the query vector.
    /// - Cosine: 0.0 (identical) to 2.0 (opposite)
    /// - Euclidean: 0.0 (identical) to infinity
    /// - DotProduct: Negative (higher is more similar)
    pub distance: f32,

    /// Optional metadata associated with the vector
    pub metadata: Option<serde_json::Value>,
}
```

### SearchExplain

```rust
pub struct SearchExplain {
    /// Total search time in microseconds
    pub total_time_us: u64,

    /// Time spent traversing HNSW index (microseconds)
    pub index_time_us: u64,

    /// Time spent evaluating metadata filters (microseconds)
    pub filter_time_us: u64,

    /// Time spent enriching results with metadata (microseconds)
    pub enrich_time_us: u64,

    /// Candidates before filtering
    pub candidates_before_filter: usize,

    /// Candidates after filtering
    pub candidates_after_filter: usize,

    /// HNSW traversal statistics
    pub hnsw_stats: SearchStats,

    /// Collection dimensions
    pub dimensions: usize,

    /// Total vectors in collection
    pub collection_size: usize,

    /// Requested k
    pub requested_k: usize,

    /// Actual k used (clamped to collection size)
    pub effective_k: usize,

    /// ef_search parameter used
    pub ef_search: usize,

    /// Whether a filter was applied
    pub filter_applied: bool,

    /// Distance function used
    pub distance_function: String,
}
```

### CollectionStats

```rust
pub struct CollectionStats {
    /// Number of vectors (excluding deleted)
    pub count: usize,

    /// Vector dimensionality
    pub dimensions: usize,

    /// Estimated memory usage in bytes
    pub memory_bytes: usize,

    /// Number of deleted vectors pending compaction
    pub deleted_count: usize,

    /// Distance function used
    pub distance_function: DistanceFunction,

    /// HNSW index statistics
    pub hnsw_stats: HnswStats,
}
```

---

## Error Reference

### Error Categories

| Code Range | Category | Description |
|------------|----------|-------------|
| 1xxx | I/O | File read/write errors |
| 2xxx | Serialization | JSON/binary encoding errors |
| 3xxx | Collection | Collection management errors |
| 4xxx | Vector | Vector operation errors |
| 5xxx | Database | Database-level errors |
| 6xxx | Index | HNSW/IVF index errors |
| 7xxx | Configuration | Invalid configuration |
| 8xxx | Resource | Capacity/memory limits |
| 9xxx | Operational | Timeouts, locks, conflicts |
| 10xxx | Security | Encryption/auth errors |
| 11xxx | Distributed | Consensus/replication errors |
| 12xxx | Backup | Backup/restore errors |
| 13xxx | State | Invalid operations |

### Common Error Variants

| Variant | Code | Description |
|---------|------|-------------|
| `CollectionNotFound(name)` | 3001 | Collection doesn't exist |
| `CollectionAlreadyExists(name)` | 3002 | Collection name taken |
| `DimensionMismatch { expected, got }` | 4003 | Vector has wrong dimensions |
| `InvalidVector(reason)` | 4004 | Vector contains NaN/Inf |
| `AliasNotFound(name)` | 3004 | Alias doesn't exist |
| `AliasAlreadyExists(name)` | 3005 | Alias name taken |
| `CollectionHasAliases(name)` | 3006 | Can't delete collection with aliases |
| `IoError(err)` | 1001 | File system error |
| `Corruption(msg)` | 5002 | Database file corrupted |

### Error Handling Example

```rust
use needle::{Database, NeedleError, ErrorCode};

fn handle_operation() -> needle::Result<()> {
    let db = Database::open("db.needle")?;

    match db.collection("nonexistent") {
        Ok(coll) => { /* use collection */ }
        Err(NeedleError::CollectionNotFound(name)) => {
            // Handle specifically
            println!("Collection '{}' not found, creating...", name);
            db.create_collection(&name, 384)?;
        }
        Err(e) => {
            // Generic handling with error code
            let code = e.error_code();
            println!("Error {}: {} (category: {})",
                code.code(),
                e,
                code.category()
            );
            return Err(e);
        }
    }

    Ok(())
}
```

---

## See Also

- [How-To Guides](how-to-guides.md) - Practical tutorials for common tasks
- [Architecture](architecture.md) - Internal design and data flow diagrams
- [Index Selection Guide](index-selection-guide.md) - HNSW vs IVF vs DiskANN decision guide
- [Production Checklist](production-checklist.md) - Pre-deployment verification
