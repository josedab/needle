---
sidebar_position: 11
---

# Troubleshooting

Common issues and their solutions when working with Needle.

## Installation Issues

### Rust compilation fails

**Symptoms**: `cargo build` fails with linker errors or missing dependencies.

**Solutions**:

1. **Update Rust toolchain**:
   ```bash
   rustup update stable
   ```

2. **Install required system dependencies**:

   **macOS**:
   ```bash
   xcode-select --install
   ```

   **Ubuntu/Debian**:
   ```bash
   sudo apt-get install build-essential pkg-config libssl-dev
   ```

   **Windows**: Install Visual Studio Build Tools with "Desktop development with C++".

3. **Clear cargo cache**:
   ```bash
   cargo clean
   cargo build
   ```

### Python bindings won't install

**Symptoms**: `pip install needle-db` fails.

**Solutions**:

1. **Ensure you have a compatible Python version** (3.8+):
   ```bash
   python --version
   ```

2. **Install from source with maturin**:
   ```bash
   pip install maturin
   git clone https://github.com/anthropics/needle
   cd needle
   maturin build --features python --release
   pip install target/wheels/*.whl
   ```

3. **Use a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install needle-db
   ```

### WASM build fails

**Symptoms**: `wasm-pack build` fails with errors.

**Solutions**:

1. **Install wasm-pack**:
   ```bash
   cargo install wasm-pack
   ```

2. **Add the wasm32 target**:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

3. **Build with correct flags**:
   ```bash
   wasm-pack build --features wasm --target web
   ```

---

## Database Errors

### Error: "Collection not found"

**Cause**: Trying to access a collection that doesn't exist.

**Solution**:
```rust
// Check if collection exists before accessing
if db.collection_exists("my_collection")? {
    let collection = db.collection("my_collection")?;
} else {
    // Create it first
    db.create_collection("my_collection", 384)?;
}
```

**Python**:
```python
if db.collection_exists("my_collection"):
    collection = db.collection("my_collection")
else:
    db.create_collection("my_collection", 384)
```

### Error: "Dimension mismatch"

**Cause**: Inserting a vector with different dimensions than the collection.

**Example**:
```rust
// Collection created with 384 dimensions
db.create_collection("docs", 384)?;

// This will fail - vector has 512 dimensions
collection.insert("id", &vec![0.0; 512], json!({}))?; // Error!
```

**Solution**: Ensure all vectors match the collection's dimension:
```rust
let collection_info = collection.info()?;
let expected_dims = collection_info.dimensions;

if vector.len() != expected_dims {
    return Err(format!(
        "Vector has {} dimensions, expected {}",
        vector.len(),
        expected_dims
    ).into());
}
```

### Error: "Vector not found"

**Cause**: Trying to get or delete a vector that doesn't exist.

**Solution**: Check existence before operating:
```rust
match collection.get("nonexistent") {
    Ok(entry) => println!("Found: {:?}", entry),
    Err(NeedleError::VectorNotFound(_)) => println!("Not found"),
    Err(e) => return Err(e),
}

// Or use exists() first
if collection.exists("my_id")? {
    let entry = collection.get("my_id")?;
}
```

### Error: "Alias not found"

**Cause**: Trying to use an alias that hasn't been created.

**Solution**:
```rust
// Create the alias first
db.create_alias("prod", "my_collection")?;

// Now you can use it
let collection = db.collection("prod")?;
```

### Error: "Alias already exists"

**Cause**: Trying to create an alias that already exists.

**Solution**: Update the existing alias instead:
```rust
// Option 1: Update existing alias
db.update_alias("prod", "new_collection")?;

// Option 2: Delete and recreate
db.delete_alias("prod")?;
db.create_alias("prod", "new_collection")?;
```

### Error: "Cannot drop collection: aliases still reference it"

**Cause**: Trying to delete a collection that has aliases pointing to it.

**Solution**: Remove all aliases first:
```rust
// Find and remove all aliases for this collection
for alias in db.aliases_for_collection("my_collection") {
    db.delete_alias(&alias)?;
}

// Now you can drop the collection
db.delete_collection("my_collection")?;
```

### Error: "Database file corrupted"

**Cause**: File corruption due to improper shutdown, disk errors, or incomplete writes.

**Solutions**:

1. **Restore from backup**:
   ```bash
   cp vectors.needle.backup vectors.needle
   ```

2. **Re-index from source data** if no backup exists

**Prevention**:
- Always call `db.save()` before shutdown
- Use proper error handling
- Maintain regular backups:
  ```rust
  db.save()?;
  std::fs::copy("vectors.needle", "vectors.needle.backup")?;
  ```

---

## Search Issues

### Search returns no results

**Possible causes and solutions**:

1. **Collection is empty**:
   ```rust
   let count = collection.count()?;
   println!("Collection has {} vectors", count);
   ```

2. **Filter is too restrictive**:
   ```rust
   // Try without filter first
   let results = collection.search(&query, 10, None)?;

   // If that works, debug your filter
   let filter = Filter::parse(&json!({"category": "books"}))?;
   let filtered = collection.search(&query, 10, Some(&filter))?;
   ```

3. **Query vector dimensions don't match**:
   ```rust
   let info = collection.info()?;
   assert_eq!(query.len(), info.dimensions);
   ```

4. **Cosine distance with unnormalized vectors**:
   ```rust
   // Normalize your vectors for cosine distance
   fn normalize(v: &mut [f32]) {
       let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
       if norm > 0.0 {
           for x in v.iter_mut() {
               *x /= norm;
           }
       }
   }
   ```

### Search returns unexpected results

**Possible causes**:

1. **Wrong distance function**: Check if you're using the right distance:
   - **Cosine**: Best for normalized embeddings (most common)
   - **Euclidean**: For spatial data
   - **DotProduct**: When vectors are pre-normalized

2. **Low ef_search value**: Increase for better recall:
   ```rust
   let results = collection.search_with_params(&query, 10, None, 200)?;
   ```

3. **Index not built properly**: For large batches, ensure proper indexing:
   ```rust
   // Insert all vectors
   for (id, vec, meta) in vectors {
       collection.insert(&id, &vec, meta)?;
   }
   // Save to ensure index is persisted
   db.save()?;
   ```

### Search is slow

**Solutions**:

1. **Decrease ef_search** (trades accuracy for speed):
   ```rust
   collection.search_with_params(&query, 10, None, 30)?; // Lower ef_search
   ```

2. **Use quantization** to reduce memory and computation:
   ```rust
   let config = CollectionConfig::new(384, DistanceFunction::Cosine)
       .with_quantization(QuantizationType::Scalar);
   ```

3. **Enable SIMD**:
   ```toml
   [dependencies]
   needle = { version = "0.1", features = ["simd"] }
   ```

4. **Use batch search** for multiple queries:
   ```rust
   let queries: Vec<&[f32]> = vec![&q1, &q2, &q3];
   let results = collection.batch_search(&queries, 10, None)?;
   ```

5. **Add metadata filters** to reduce candidates:
   ```rust
   let filter = Filter::parse(&json!({"category": "active"}))?;
   let results = collection.search(&query, 10, Some(&filter))?;
   ```

---

## Performance Issues

### High memory usage

**Solutions**:

1. **Use quantization**:

   | Type | Compression | Quality Impact |
   |------|-------------|----------------|
   | Scalar | 4x | Minimal |
   | Product | 8-32x | Moderate |
   | Binary | 32x | Higher |

   ```rust
   let config = CollectionConfig::new(384, DistanceFunction::Cosine)
       .with_quantization(QuantizationType::Scalar);
   ```

2. **Compact after deletions**:
   ```rust
   collection.compact()?;
   ```

3. **Use memory-mapped files** (automatic for large files):
   - Files >10MB are automatically memory-mapped
   - This reduces RAM usage for large datasets

### Large database file

**Causes and solutions**:

1. **Deleted vectors not reclaimed**: Run compact:
   ```rust
   collection.compact()?;
   db.save()?;
   ```

2. **Large metadata**: Store minimal metadata, keep full content elsewhere:
   ```rust
   // Instead of storing full content
   collection.insert(&id, &vec, json!({
       "id": id,  // Just store a reference
       "preview": &content[..100],  // Short preview
   }))?;
   ```

3. **No quantization**: Enable for smaller file size:
   ```rust
   .with_quantization(QuantizationType::Product {
       num_subvectors: 48,
       num_centroids: 256,
   })
   ```

### Slow inserts

**Solutions**:

1. **Batch your inserts**:
   ```rust
   // Don't save after every insert
   for item in items {
       collection.insert(&item.id, &item.vec, item.meta)?;
   }
   db.save()?; // Save once at the end
   ```

2. **Use lower M and ef_construction** for faster building:
   ```rust
   let config = CollectionConfig::new(384, DistanceFunction::Cosine)
       .with_hnsw_m(12)  // Lower M
       .with_hnsw_ef_construction(100);  // Lower ef_construction
   ```

3. **Insert in parallel** (for independent operations):
   ```rust
   use rayon::prelude::*;

   items.par_iter().for_each(|item| {
       collection.insert(&item.id, &item.vec, item.meta.clone()).unwrap();
   });
   ```

---

## HTTP Server Issues

### Server won't start

**Solutions**:

1. **Check port availability**:
   ```bash
   # Check if port is in use
   lsof -i :8080

   # Use a different port
   needle serve -a 127.0.0.1:3000 -d mydb.needle
   ```

2. **Ensure server feature is enabled**:
   ```bash
   cargo build --features server
   cargo run --features server -- serve -a 127.0.0.1:8080 -d mydb.needle
   ```

### Connection refused

**Solutions**:

1. **Check the bind address**:
   ```bash
   # Bind to all interfaces (not just localhost)
   needle serve -a 0.0.0.0:8080 -d mydb.needle
   ```

2. **Check firewall settings**:
   ```bash
   # Allow port 8080 (Linux)
   sudo ufw allow 8080
   ```

### Request timeout

**Cause**: Large operations taking too long.

**Solutions**:

1. **Increase client timeout**:
   ```bash
   curl --max-time 120 http://localhost:8080/collections/docs/search ...
   ```

2. **Use pagination** for large exports:
   ```bash
   curl "http://localhost:8080/collections/docs/export?limit=1000&offset=0"
   ```

---

## CLI Issues

### Command not found: needle

**Solutions**:

1. **Build and install**:
   ```bash
   cargo build --release
   sudo cp target/release/needle /usr/local/bin/
   ```

2. **Or run via cargo**:
   ```bash
   cargo run --release -- info mydb.needle
   ```

3. **Add to PATH**:
   ```bash
   export PATH="$PATH:/path/to/needle/target/release"
   ```

### JSON input not recognized

**Cause**: Incorrect JSON format for insert command.

**Correct format**:
```bash
# Single vector
echo '{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"Hello"}}' | \
  needle insert mydb.needle -c documents

# Multiple vectors (one JSON per line)
cat << 'EOF' | needle insert mydb.needle -c documents
{"id":"doc1","vector":[0.1,0.2,0.3],"metadata":{"title":"First"}}
{"id":"doc2","vector":[0.4,0.5,0.6],"metadata":{"title":"Second"}}
EOF
```

---

## Getting More Help

If you can't resolve your issue:

1. **Check the FAQ**: [/docs/faq](/docs/faq)
2. **Search GitHub Issues**: [github.com/anthropics/needle/issues](https://github.com/anthropics/needle/issues)
3. **Ask in Discussions**: [github.com/anthropics/needle/discussions](https://github.com/anthropics/needle/discussions)
4. **Join Discord**: [discord.gg/anthropic](https://discord.gg/anthropic)

When reporting issues, include:
- Needle version (`cargo pkgid needle` or `pip show needle-db`)
- Operating system and version
- Rust version (`rustc --version`)
- Minimal reproducible example
- Full error message and stack trace
