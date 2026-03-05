# Search Cookbook

Ten recipes for common search patterns in Needle. Each recipe includes Rust, Python, and HTTP examples.

## 1. Basic k-NN Search

Find the `k` nearest vectors to a query.

**Rust:**
```rust
let results = collection.search(&query_vector, 10)?;
for r in &results {
    println!("{}: distance={}", r.id, r.distance);
}
```

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 10}'
```

## 2. Filtered Search (Pre-filter)

Pre-filters are applied during HNSW traversal for maximum efficiency.

**Rust:**
```rust
let filter = Filter::parse(&json!({"category": "science"}))?;
let results = collection.search_with_filter(&query, 10, &filter)?;
```

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "k": 10,
    "filter": {"category": "science", "year": {"$gte": 2020}}
  }'
```

## 3. Range / Radius Search

Find all vectors within a distance threshold.

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/search/radius \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "max_distance": 0.5, "limit": 100}'
```

## 4. Batch Search

Run multiple queries in a single request for throughput.

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/search/batch \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "k": 5,
    "filter": {"category": "science"}
  }'
```

## 5. Search with EXPLAIN

Get profiling data to understand query performance.

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 10, "explain": true}'
```

Response includes `explanation.profiling` with timing breakdown, HNSW traversal stats, and filter selectivity.

## 6. Matryoshka Truncated Search

Use dimension truncation for 3-6× speedup on Matryoshka-trained embeddings.

**Rust:**
```rust
let results = collection.search_matryoshka(&query, 10, 64)?; // truncate to 64 dims
```

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/search/matryoshka \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10, "truncated_dimensions": 64}'
```

## 7. Hybrid Search (BM25 + Vector)

Combine keyword search with vector similarity using Reciprocal Rank Fusion.

**Rust:**
```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

let mut bm25 = Bm25Index::default();
bm25.index_document("doc1", "machine learning and AI");

let bm25_results = bm25.search("machine learning", 10);
let vector_results = collection.search(&query, 10)?;
let hybrid = reciprocal_rank_fusion(&vector_results, &bm25_results, &RrfConfig::default(), 10);
```

## 8. Paginated Search

Iterate through large result sets page by page.

**HTTP:**
```bash
# First page
curl -X POST http://localhost:8080/v1/collections/docs/search \
  -d '{"vector": [0.1, 0.2, 0.3], "k": 10}'
# Response includes: "next_cursor": {"distance": 0.45, "id": "doc_99"}

# Next page
curl -X POST http://localhost:8080/v1/collections/docs/search \
  -d '{
    "vector": [0.1, 0.2, 0.3],
    "k": 10,
    "search_after": {"distance": 0.45, "id": "doc_99"}
  }'
```

## 9. Recommendation (Find Similar to IDs)

Find vectors similar to given examples, excluding specific IDs.

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "positive_ids": ["doc_liked_1", "doc_liked_2"],
    "negative_ids": ["doc_disliked_1"],
    "limit": 10
  }'
```

## 10. Metadata-Only Query

Filter documents without vector similarity.

**HTTP:**
```bash
curl -X POST http://localhost:8080/v1/collections/docs/query \
  -H "Content-Type: application/json" \
  -d '{
    "filter": {"category": "science", "year": {"$gte": 2024}},
    "limit": 50,
    "offset": 0
  }'
```

## Filter Operators Reference

| Operator | Example | Description |
|----------|---------|-------------|
| `$eq` | `{"field": "value"}` | Equal (implicit) |
| `$ne` | `{"field": {"$ne": "value"}}` | Not equal |
| `$gt` / `$gte` | `{"age": {"$gt": 18}}` | Greater than (or equal) |
| `$lt` / `$lte` | `{"age": {"$lt": 65}}` | Less than (or equal) |
| `$in` / `$nin` | `{"tag": {"$in": ["a","b"]}}` | In / not in array |
| `$contains` | `{"name": {"$contains": "needle"}}` | String contains |
| `$startsWith` | `{"name": {"$startsWith": "pre"}}` | String starts with |
| `$endsWith` | `{"name": {"$endsWith": "fix"}}` | String ends with |
| `$and` / `$or` | `{"$and": [{...}, {...}]}` | Logical operators |
| `$not` | `{"$not": {...}}` | Logical negation |
