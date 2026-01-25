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
| `drop_collection` | `fn drop_collection(&self, name: &str) -> Result<bool>` | Delete collection, returns true if existed |
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
