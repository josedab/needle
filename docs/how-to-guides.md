# Needle How-To Guides

Practical tutorials for common tasks with Needle vector database.

## Table of Contents

- [Getting Started](#getting-started)
- [HNSW Parameter Tuning](#hnsw-parameter-tuning)
- [Quantization Strategies](#quantization-strategies)
- [Hybrid Search Setup](#hybrid-search-setup)
- [Backup and Restore](#backup-and-restore)
- [Performance Optimization](#performance-optimization)
- [Migrating Between Index Types](#migrating-between-index-types)
- [Metadata Filtering Patterns](#metadata-filtering-patterns)
- [Multi-Collection Workflows](#multi-collection-workflows)

---

## Getting Started

### Creating Your First Database

```rust
use needle::{Database, CollectionConfig, DistanceFunction};
use serde_json::json;

fn main() -> needle::Result<()> {
    // 1. Create a file-backed database
    let db = Database::open("my_vectors.needle")?;

    // 2. Create a collection with 384 dimensions (common for embedding models)
    db.create_collection("documents", 384)?;

    // 3. Get a thread-safe reference
    let coll = db.collection("documents")?;

    // 4. Insert vectors with metadata
    let embedding = vec![0.1; 384]; // Your actual embedding here
    coll.insert(
        "doc1",
        &embedding,
        Some(json!({
            "title": "Introduction to Vector Search",
            "category": "tutorial",
            "date": "2024-01-15"
        }))
    )?;

    // 5. Search for similar vectors
    let query = vec![0.1; 384];
    let results = coll.search(&query, 10)?;

    for result in results {
        println!("ID: {}, Distance: {:.4}", result.id, result.distance);
    }

    // 6. Save changes to disk
    db.save()?;

    Ok(())
}
```

### Choosing a Distance Function

| Distance Function | Best For | Normalized Vectors? |
|-------------------|----------|---------------------|
| `Cosine` (default) | Text embeddings, semantic similarity | No (handled internally) |
| `CosineNormalized` | Pre-normalized vectors | Yes (faster) |
| `Euclidean` | Spatial data, image features | No |
| `DotProduct` | Recommendation systems with normalized vectors | Yes |
| `Manhattan` | Sparse vectors, categorical data | No |

```rust
use needle::{CollectionConfig, DistanceFunction};

// For semantic similarity (most text embeddings)
let config = CollectionConfig::new("embeddings", 384)
    .with_distance(DistanceFunction::Cosine);

// For spatial/image data
let config = CollectionConfig::new("images", 512)
    .with_distance(DistanceFunction::Euclidean);

// For pre-normalized embeddings (faster)
let config = CollectionConfig::new("normalized", 384)
    .with_distance(DistanceFunction::CosineNormalized);
```

---

## HNSW Parameter Tuning

### Understanding the Parameters

The HNSW index has four key parameters that affect performance:

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `M` | 16 | 4-64 | Connections per node. Higher = better recall, more memory |
| `M_max_0` | 32 | M-128 | Connections at layer 0. Usually 2×M |
| `ef_construction` | 200 | 50-500 | Build-time search width. Higher = better index, slower build |
| `ef_search` | 50 | 10-500 | Query-time search width. Higher = better recall, slower queries |

### Tuning for High Recall (>99%)

Use this when accuracy is critical (e.g., production search, legal documents):

```rust
use needle::{CollectionConfig, HnswConfig};

let hnsw = HnswConfig::default()
    .m(32)               // More connections
    .m_max_0(64)         // Double for layer 0
    .ef_construction(400) // Better index quality
    .ef_search(200);     // More thorough search

let config = CollectionConfig::new("high_recall", 384)
    .with_hnsw_config(hnsw);
```

**Trade-offs:**
- Memory: ~2× default
- Build time: ~2× default
- Query time: ~3-4× default
- Recall: 99%+ vs 95% default

### Tuning for Speed

Use this when latency is critical (e.g., real-time recommendations):

```rust
let hnsw = HnswConfig::default()
    .m(8)                // Fewer connections
    .ef_construction(100) // Faster build
    .ef_search(20);      // Quick queries

let config = CollectionConfig::new("fast", 384)
    .with_hnsw_config(hnsw);
```

**Trade-offs:**
- Memory: 0.5× default
- Build time: 0.5× default
- Query time: 0.3× default
- Recall: ~90%

### Tuning for Memory Efficiency

Use this for large datasets with memory constraints:

```rust
let hnsw = HnswConfig::default()
    .m(8)
    .m_max_0(16)
    .ef_construction(150);

// Combine with quantization for maximum savings
let config = CollectionConfig::new("memory_efficient", 384)
    .with_hnsw_config(hnsw);
```

### Using Auto-Tuning

Let Needle automatically choose optimal parameters:

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

// Define your constraints
let constraints = TuningConstraints::new(1_000_000, 384) // 1M vectors, 384 dims
    .with_profile(PerformanceProfile::HighRecall)
    .with_memory_budget(4 * 1024 * 1024 * 1024); // 4GB limit

let result = auto_tune(&constraints);

println!("Recommended M: {}", result.config.m);
println!("Recommended ef_construction: {}", result.config.ef_construction);
println!("Expected recall: {:.1}%", result.expected_recall * 100.0);
println!("Expected memory: {} MB", result.expected_memory_mb);

// Apply the recommendations
let config = CollectionConfig::new("auto_tuned", 384)
    .with_hnsw_config(result.config);
```

### Performance Profiles

| Profile | M | ef_construction | ef_search | Use Case |
|---------|---|-----------------|-----------|----------|
| `HighRecall` | 32 | 400 | 200 | Production search, legal |
| `Balanced` | 16 | 200 | 50 | General purpose |
| `LowLatency` | 8 | 100 | 20 | Real-time systems |
| `MemoryConstrained` | 8 | 100 | 30 | Edge devices |

---

## Quantization Strategies

Quantization reduces memory usage by compressing vectors. Choose based on your accuracy/memory trade-off.

### When to Use Each Strategy

| Strategy | Compression | Recall Loss | Best For |
|----------|-------------|-------------|----------|
| **None** | 1× | 0% | Small datasets, max accuracy |
| **Scalar (SQ8)** | 4× | <1% | Most use cases |
| **Product (PQ)** | 16-64× | 2-5% | Large datasets |
| **Binary** | 32× | 5-10% | Candidate generation |

### Scalar Quantization (Recommended Default)

Best balance of compression and accuracy:

```rust
use needle::ScalarQuantizer;

// Collect sample vectors for training (1-10% of data)
let samples: Vec<Vec<f32>> = get_sample_vectors();
let sample_refs: Vec<&[f32]> = samples.iter().map(|v| v.as_slice()).collect();

// Train quantizer
let sq = ScalarQuantizer::train(&sample_refs);

// Quantize vectors (4× compression)
let quantized = sq.quantize(&my_vector);  // Vec<u8>

// Dequantize for search
let restored = sq.dequantize(&quantized); // Vec<f32>
```

### Product Quantization (Maximum Compression)

For very large datasets where memory is critical:

```rust
use needle::ProductQuantizer;

// Train with number of subvectors (must divide dimensions evenly)
// 384 dimensions / 8 subvectors = 48 dimensions per subvector
let pq = ProductQuantizer::train(&sample_refs, 8);

// Encode (16× compression with 8 subvectors)
let codes = pq.encode(&my_vector);  // Vec<u8>, length = 8

// Decode
let restored = pq.decode(&codes);

// Fast asymmetric distance (for search)
let distance = pq.asymmetric_distance(&query, &codes);
```

**Choosing Number of Subvectors:**

| Subvectors | Compression | Accuracy | Recommended For |
|------------|-------------|----------|-----------------|
| 4 | 48× | Lower | Maximum compression |
| 8 | 24× | Good | Large datasets |
| 16 | 12× | Better | Medium datasets |
| 32 | 6× | Best | Quality-sensitive |

### Binary Quantization (Fastest)

For candidate generation before re-ranking:

```rust
use needle::BinaryQuantizer;

// Binary quantization (sign of each dimension)
let bq = BinaryQuantizer::new();

let binary = bq.quantize(&my_vector);  // Bitset
let hamming_distance = bq.hamming_distance(&binary1, &binary2);
```

---

## Hybrid Search Setup

Combine vector similarity with keyword search for better results.

### Setting Up BM25 + Vector Search

```rust
use needle::{Database, Bm25Index, reciprocal_rank_fusion, RrfConfig};
use serde_json::json;

// 1. Create database and collection
let db = Database::in_memory();
db.create_collection("documents", 384)?;
let coll = db.collection("documents")?;

// 2. Create BM25 index for keyword search
let mut bm25 = Bm25Index::default();

// 3. Insert documents into both indexes
let documents = vec![
    ("doc1", "Machine learning algorithms for data analysis", vec![0.1; 384]),
    ("doc2", "Deep learning and neural networks", vec![0.2; 384]),
    ("doc3", "Natural language processing techniques", vec![0.3; 384]),
];

for (id, text, embedding) in documents {
    // Vector index
    coll.insert(id, &embedding, Some(json!({"text": text})))?;

    // BM25 index
    bm25.index_document(id, text);
}

// 4. Hybrid search
let query_embedding = vec![0.15; 384];
let query_text = "machine learning";

// Get results from both indexes
let vector_results = coll.search(&query_embedding, 100)?;
let bm25_results = bm25.search(query_text, 100);

// Fuse results with RRF
let config = RrfConfig::default();  // k=60
let hybrid_results = reciprocal_rank_fusion(
    &vector_results,
    &bm25_results,
    &config,
    10  // top k
);

for (id, score) in hybrid_results {
    println!("ID: {}, Hybrid Score: {:.4}", id, score);
}
```

### Tuning RRF Parameters

```rust
use needle::RrfConfig;

// Default: k=60, balanced weighting
let config = RrfConfig::default();

// Favor vector results (higher k)
let vector_biased = RrfConfig::default().with_k(100);

// Favor BM25 results (lower k)
let keyword_biased = RrfConfig::default().with_k(30);

// Custom weighting
let custom = RrfConfig::default()
    .with_k(60)
    .with_vector_weight(0.7)
    .with_bm25_weight(0.3);
```

### Adaptive Fusion

Learn optimal weights from user feedback:

```rust
use needle::{AdaptiveFusion, SearchFeedback, QueryFeatures};

let mut fusion = AdaptiveFusion::default();

// Record user interactions
let feedback = SearchFeedback {
    query_id: "q1".to_string(),
    clicked_results: vec!["doc2".to_string(), "doc5".to_string()],
    query_features: QueryFeatures {
        query_length: 3,
        has_quotes: false,
        domain_specific: true,
    },
};

fusion.record_feedback(feedback);

// Get learned weights
let stats = fusion.get_stats();
println!("Learned vector weight: {:.2}", stats.vector_weight);
println!("Learned BM25 weight: {:.2}", stats.bm25_weight);
```

---

## Backup and Restore

### Creating Backups

```rust
use needle::{BackupManager, BackupConfig, BackupType, Database};
use std::path::Path;

// Configure backup location
let config = BackupConfig::new("/backups/needle")
    .with_compression(true)  // Compress backup files
    .with_retention_days(30); // Auto-delete after 30 days

let manager = BackupManager::new(config);

// Full backup
let db = Database::open("production.needle")?;
let metadata = manager.create_backup(&db, BackupType::Full)?;
println!("Backup created: {}", metadata.id);
println!("Size: {} MB", metadata.size_bytes / 1024 / 1024);

// Incremental backup (only changes since last backup)
let incremental = manager.create_backup(&db, BackupType::Incremental)?;
```

### Listing and Managing Backups

```rust
// List all backups
let backups = manager.list_backups()?;
for backup in backups {
    println!("{}: {} ({} MB, {})",
        backup.id,
        backup.backup_type,
        backup.size_bytes / 1024 / 1024,
        backup.timestamp
    );
}

// Delete old backups
manager.cleanup_old_backups()?;
```

### Restoring from Backup

```rust
// Restore to a new location
let backup_id = "backup_2024-01-15_12-30-00";
manager.restore(backup_id, "/data/restored.needle")?;

// Verify the restored database
let restored_db = Database::open("/data/restored.needle")?;
let collections = restored_db.list_collections();
println!("Restored collections: {:?}", collections);
```

### Scheduled Backups

For production systems, set up periodic backups:

```rust
use std::time::Duration;
use std::thread;

fn backup_scheduler(db: &Database, manager: &BackupManager) {
    loop {
        // Full backup every day
        if let Err(e) = manager.create_backup(db, BackupType::Full) {
            eprintln!("Backup failed: {}", e);
        }

        thread::sleep(Duration::from_secs(24 * 60 * 60));
    }
}
```

---

## Performance Optimization

### Batch Operations

Insert multiple vectors efficiently:

```rust
use needle::Database;
use serde_json::json;

let db = Database::in_memory();
db.create_collection("bulk", 384)?;
let coll = db.collection("bulk")?;

// Batch insert (more efficient than individual inserts)
let vectors: Vec<(&str, Vec<f32>, Option<serde_json::Value>)> = (0..10000)
    .map(|i| {
        (
            format!("doc_{}", i).leak() as &str,
            vec![0.1; 384],
            Some(json!({"index": i}))
        )
    })
    .collect();

// Convert for batch insert
let batch: Vec<(&str, &[f32], Option<serde_json::Value>)> = vectors
    .iter()
    .map(|(id, vec, meta)| (*id, vec.as_slice(), meta.clone()))
    .collect();

coll.insert_batch(&batch)?;
```

### Batch Search

Search multiple queries in parallel:

```rust
// Multiple queries
let queries: Vec<Vec<f32>> = (0..100)
    .map(|_| vec![0.1; 384])
    .collect();

let query_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();

// Parallel batch search (uses Rayon)
let all_results = coll.batch_search(&query_refs, 10)?;

for (i, results) in all_results.iter().enumerate() {
    println!("Query {}: {} results", i, results.len());
}
```

### Memory Mapping for Large Files

Needle automatically uses memory mapping for files over 10MB. For manual control:

```rust
use needle::DatabaseConfig;

let config = DatabaseConfig::new("large_db.needle")
    .with_mmap_threshold(5 * 1024 * 1024);  // 5MB threshold
```

### Compaction

Reclaim space after deletions:

```rust
// After many deletions
for i in 0..1000 {
    coll.delete(&format!("doc_{}", i))?;
}

// Compact to reclaim space
let stats = coll.compact()?;
println!("Reclaimed {} bytes", stats.bytes_reclaimed);
```

### Query Caching

Enable caching for repeated queries:

```rust
use needle::{CollectionConfig, QueryCacheConfig};

let cache_config = QueryCacheConfig::new()
    .with_max_entries(1000)
    .with_ttl_seconds(300);  // 5 minute TTL

let config = CollectionConfig::new("cached", 384)
    .with_query_cache(cache_config);
```

### Profiling Queries

Understand query performance:

```rust
// Get detailed timing breakdown
let (results, explain) = coll.search_explain(&query, 10)?;

println!("Total time: {}μs", explain.total_time_us);
println!("Index time: {}μs", explain.index_time_us);
println!("Filter time: {}μs", explain.filter_time_us);
println!("Nodes visited: {}", explain.hnsw_stats.visited_nodes);
println!("Distance calculations: {}", explain.hnsw_stats.distance_calculations);
```

---

## Migrating Between Index Types

### HNSW to IVF Migration

Use IVF for very large datasets (>10M vectors):

```rust
use needle::{Database, IvfIndex, IvfConfig};

// 1. Export from HNSW collection
let db = Database::open("source.needle")?;
let source_coll = db.collection("documents")?;

// 2. Create IVF index
let ivf_config = IvfConfig::new(256)  // 256 clusters
    .with_nprobe(16);                 // Search 16 clusters

let mut ivf = IvfIndex::new(384, ivf_config);

// 3. Collect training data (10% of vectors)
let training_vectors: Vec<Vec<f32>> = source_coll.iter()
    .take(source_coll.len() / 10)
    .map(|(_, vec, _)| vec)
    .collect();

let training_refs: Vec<&[f32]> = training_vectors
    .iter()
    .map(|v| v.as_slice())
    .collect();

// 4. Train IVF on sample
ivf.train(&training_refs)?;

// 5. Migrate all vectors
for (id, vector, metadata) in source_coll.iter() {
    ivf.insert_with_id(&id, &vector, metadata)?;
}

println!("Migrated {} vectors to IVF", ivf.len());
```

### Choosing Index Type

| Index | Dataset Size | Build Time | Query Time | Memory |
|-------|--------------|------------|------------|--------|
| HNSW | <10M | Medium | Fast | High |
| IVF | 10M-100M | Fast | Medium | Medium |
| DiskANN | >100M | Slow | Medium | Low |

---

## Metadata Filtering Patterns

### Complex Filters

```rust
use needle::Filter;
use serde_json::json;

// Range queries
let price_range = Filter::and(vec![
    Filter::gte("price", 10.0),
    Filter::lt("price", 100.0),
]);

// Multiple conditions with OR
let categories = Filter::or(vec![
    Filter::eq("category", "electronics"),
    Filter::eq("category", "computers"),
    Filter::eq("category", "gadgets"),
]);

// Using $in for cleaner syntax
let categories_in = Filter::in_values("category",
    vec!["electronics", "computers", "gadgets"]);

// Nested conditions
let complex = Filter::and(vec![
    price_range,
    categories,
    Filter::not(Filter::eq("out_of_stock", true)),
]);

// Search with filter
let results = coll.search_with_filter(&query, 10, &complex)?;
```

### Pre-Filter vs Post-Filter

```rust
// Pre-filter: Applied during HNSW traversal (efficient for selective filters)
let selective_filter = Filter::eq("premium_user", true);
let results = coll.search_builder(&query)
    .k(10)
    .filter(&selective_filter)  // Pre-filter
    .execute()?;

// Post-filter: Applied after search (guarantees k candidates before filtering)
let expensive_filter = Filter::parse(&json!({
    "$or": [
        {"score": {"$gt": 0.9}},
        {"verified": true}
    ]
}))?;

let results = coll.search_builder(&query)
    .k(10)
    .post_filter(&expensive_filter)  // Post-filter
    .post_filter_factor(5)           // Fetch 5× candidates
    .execute()?;
```

### Indexing Patterns for Fast Filtering

For frequently filtered fields, consider denormalizing:

```rust
// Instead of filtering on nested fields:
// {"user": {"subscription": "premium"}}

// Denormalize to top-level for faster filtering:
// {"user_subscription": "premium"}

let fast_filter = Filter::eq("user_subscription", "premium");
```

---

## Multi-Collection Workflows

### Organizing Data Across Collections

```rust
use needle::{Database, CollectionConfig, DistanceFunction};

let db = Database::open("multi_modal.needle")?;

// Separate collections for different embedding models
db.create_collection_with_config(
    CollectionConfig::new("openai", 1536)  // text-embedding-3-small
)?;

db.create_collection_with_config(
    CollectionConfig::new("cohere", 1024)  // embed-english-v3.0
)?;

db.create_collection_with_config(
    CollectionConfig::new("clip_images", 512)  // CLIP image embeddings
        .with_distance(DistanceFunction::CosineNormalized)
)?;
```

### Cross-Collection Search

Search across multiple collections and merge results:

```rust
use needle::SearchResult;

fn cross_collection_search(
    db: &Database,
    query_openai: &[f32],
    query_cohere: &[f32],
    k: usize,
) -> needle::Result<Vec<SearchResult>> {
    let openai_coll = db.collection("openai")?;
    let cohere_coll = db.collection("cohere")?;

    // Search both collections
    let openai_results = openai_coll.search(query_openai, k)?;
    let cohere_results = cohere_coll.search(query_cohere, k)?;

    // Merge and re-rank (simple score-based merge)
    let mut combined: Vec<SearchResult> = openai_results
        .into_iter()
        .chain(cohere_results.into_iter())
        .collect();

    combined.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    combined.truncate(k);

    Ok(combined)
}
```

### Blue-Green Deployment with Aliases

```rust
// Create versioned collections
db.create_collection("documents_v1", 384)?;
db.create_collection("documents_v2", 384)?;

// Point production alias to v1
db.create_alias("documents", "documents_v1")?;

// Fill v2 with new embeddings...
let v2 = db.collection("documents_v2")?;
// ... populate v2 ...

// Atomic switch to v2
db.update_alias("documents", "documents_v2")?;

// Applications using "documents" alias now hit v2
let coll = db.collection("documents")?;  // Points to v2

// Cleanup old version when ready
db.drop_collection("documents_v1")?;
```

---

## Summary

| Task | Key Approach |
|------|--------------|
| **High recall** | M=32, ef_search=200, ef_construction=400 |
| **Low latency** | M=8, ef_search=20 |
| **Memory efficiency** | Scalar quantization + lower M |
| **Large datasets** | IVF or DiskANN + Product quantization |
| **Hybrid search** | BM25 + Vector + RRF fusion |
| **Production reliability** | Regular backups + WAL |
| **Query debugging** | Use `search_explain()` |
| **Zero-downtime updates** | Collection aliases for blue-green |

---

## See Also

- [API Reference](api-reference.md) - Complete method documentation
- [Index Selection Guide](index-selection-guide.md) - HNSW vs IVF vs DiskANN decision guide
- [Architecture](architecture.md) - Internal design and data flow diagrams
- [Production Checklist](production-checklist.md) - Pre-deployment verification
- [Operations Guide](OPERATIONS.md) - Day-to-day operations and monitoring
