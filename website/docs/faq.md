---
sidebar_position: 10
---

# FAQ

Frequently asked questions about Needle.

## General

### What is Needle?

Needle is an embedded vector database written in Rust. It's designed to be "SQLite for vectors"—a simple, fast, single-file database for storing and searching vector embeddings.

### Is Needle free?

Yes, Needle is open source under the MIT license. You can use it for any purpose, including commercial applications.

### What languages does Needle support?

Needle is written in Rust and provides bindings for:
- **Rust** (native)
- **Python** (via PyO3)
- **JavaScript/WASM** (via wasm-bindgen)
- **Swift** (via UniFFI)
- **Kotlin** (via UniFFI)

### Does Needle require a server?

No. Needle is embedded—it runs in your application's process. There's no separate server to manage. However, Needle also includes an optional HTTP server mode for multi-language access.

## Technical

### How fast is Needle?

Typical performance on commodity hardware:
- **Search latency**: 3-10ms for 1M vectors
- **Insert throughput**: 10,000-20,000 vectors/second
- **Recall@10**: 95-99% depending on configuration

### What's the maximum dataset size?

Needle can handle millions of vectors on a single machine. Practical limits depend on:
- **Memory**: ~1.7GB per million 384-dimensional vectors (unquantized)
- **Disk**: Database file size grows with vector count
- **Performance**: Search latency increases slightly with size

For larger datasets, use quantization or sharding.

### What embedding dimensions are supported?

Any dimension from 1 to 65,535. Common dimensions:
- 384 (all-MiniLM-L6-v2)
- 768 (BERT-base)
- 1024 (BGE-large)
- 1536 (OpenAI text-embedding-3-small)
- 3072 (OpenAI text-embedding-3-large)

### Which embedding models work with Needle?

Any model that produces fixed-length float vectors:
- OpenAI (text-embedding-3-small, text-embedding-3-large)
- Cohere (embed-v3)
- Sentence Transformers (all-MiniLM-L6-v2, BGE, etc.)
- CLIP (for images)
- Any custom embedding model

Needle stores and searches vectors—it doesn't generate them.

### Is Needle ACID compliant?

Needle provides:
- **Atomicity**: Individual operations are atomic
- **Durability**: Explicit `save()` persists to disk

It does not provide:
- Full transaction isolation
- Multi-operation transactions

For most vector search use cases, this is sufficient. If you need full ACID, consider pgvector.

## Features

### Does Needle support filtering?

Yes. Needle supports MongoDB-style metadata filtering:

```rust
let filter = Filter::parse(&json!({
    "$and": [
        {"category": {"$in": ["books", "articles"]}},
        {"year": {"$gte": 2020}},
        {"status": "published"}
    ]
}))?;

let results = collection.search(&query, 10, Some(&filter))?;
```

### Does Needle support hybrid search?

Yes, with the `hybrid` feature flag. Combine vector search with BM25 text search:

```rust
use needle::{Bm25Index, reciprocal_rank_fusion};

// BM25 for keyword matching
let bm25_results = bm25.search("machine learning", 10);

// Fuse with vector results
let hybrid = reciprocal_rank_fusion(&vector_results, &bm25_results, &config, 10);
```

### Can I update vectors?

Yes. Delete and re-insert, or use upsert logic:

```rust
// Update by delete + insert
collection.delete("doc1")?;
collection.insert("doc1", &new_vector, new_metadata)?;
```

### How do I backup a Needle database?

The entire database is a single file. Just copy it:

```bash
cp vectors.needle vectors.needle.backup
```

For consistent backups of a running database:
```rust
db.save()?; // Ensure all changes are flushed
std::fs::copy("vectors.needle", "backup.needle")?;
```

## Performance

### How do I improve search speed?

1. **Decrease `ef_search`** (trades recall for speed)
2. **Use quantization** (reduces memory and computation)
3. **Enable SIMD** (`features = ["simd"]`)
4. **Use metadata filtering** to reduce candidates

### How do I improve recall?

1. **Increase `ef_search`** (query time parameter)
2. **Increase `M`** (requires rebuild)
3. **Increase `ef_construction`** (requires rebuild)
4. **Use auto-tuning** with `PerformanceProfile::HighRecall`

### Why is my database file large?

Common causes:
1. **Deleted vectors**: Space isn't reclaimed until `compact()`
2. **No quantization**: Full precision uses 4 bytes per dimension
3. **Large metadata**: Metadata is stored with each vector

Solutions:
```rust
collection.compact()?; // Reclaim space from deletions

// Use quantization for new collections
let config = CollectionConfig::new(384, DistanceFunction::Cosine)
    .with_quantization(QuantizationType::Scalar);
```

### How do I reduce memory usage?

1. **Scalar quantization**: 4x reduction
   ```rust
   .with_quantization(QuantizationType::Scalar)
   ```

2. **Product quantization**: 8-32x reduction
   ```rust
   .with_quantization(QuantizationType::Product {
       num_subvectors: 48,
       num_centroids: 256,
   })
   ```

3. **Binary quantization**: 32x reduction
   ```rust
   .with_quantization(QuantizationType::Binary)
   ```

## Troubleshooting

### Error: "Collection not found"

The collection doesn't exist. Create it first:

```rust
if !db.collection_exists("my_collection")? {
    db.create_collection("my_collection", 384, DistanceFunction::Cosine)?;
}
```

### Error: "Dimension mismatch"

The vector you're inserting has different dimensions than the collection:

```rust
// Collection created with 384 dimensions
db.create_collection("docs", 384, DistanceFunction::Cosine)?;

// This will fail - vector has 512 dimensions
collection.insert("id", &vec![0.0; 512], json!({}))?; // Error!
```

### Error: "Vector not found"

The ID doesn't exist in the collection:

```rust
match collection.get("nonexistent") {
    Err(NeedleError::VectorNotFound(_)) => println!("Not found"),
    Ok(entry) => println!("Found: {:?}", entry),
    Err(e) => return Err(e),
}
```

### Search returns no results

1. **Check if collection has vectors**: `collection.count()?`
2. **Check filter syntax**: Try without filter first
3. **Check vector dimensions**: Must match collection
4. **Check distance function**: Cosine requires normalized vectors

### Database file corrupted

If the database file is corrupted:

1. Try opening with recovery mode (if available)
2. Restore from backup
3. Re-index from source data

Prevention:
- Always call `db.save()` before shutdown
- Use proper error handling
- Maintain regular backups

## Best Practices

### Should I use one collection or many?

**One collection** when:
- All vectors have the same dimensions
- You query across all data
- Simpler management

**Multiple collections** when:
- Different vector types (text vs. images)
- Multi-tenant isolation
- Different retention policies

### How often should I save?

```rust
// Option 1: After batch operations
for doc in documents {
    collection.insert(&doc.id, &doc.embedding, doc.metadata)?;
}
db.save()?; // Save once after batch

// Option 2: Periodic saves
// In a background thread
loop {
    thread::sleep(Duration::from_secs(30));
    db.save()?;
}
```

### How do I handle concurrent access?

Needle is thread-safe. Multiple threads can read and write:

```rust
let db = Arc::new(Database::open("vectors.needle")?);

// Clone Arc for each thread
let db_clone = db.clone();
thread::spawn(move || {
    let collection = db_clone.collection("docs")?;
    collection.insert("id", &vec, json!({}))?;
});
```

For high write concurrency, consider batching writes.

## More Questions?

- Check the [GitHub Discussions](https://github.com/anthropics/needle/discussions)
- Open an [issue](https://github.com/anthropics/needle/issues) for bugs
- Read the [API Reference](/docs/api-reference) for detailed documentation
