---
sidebar_position: 2
---

# Getting Started

This guide will walk you through installing Needle and building your first vector search application.

## Installation

### Rust

Add Needle to your `Cargo.toml`:

```toml
[dependencies]
needle = "0.1"
```

For additional features, enable them in your dependency:

```toml
[dependencies]
needle = { version = "0.1", features = ["server", "hybrid", "metrics"] }
```

### Python

```bash
pip install needle-db
```

### JavaScript/WASM

```bash
npm install @anthropic/needle
```

### From Source

```bash
git clone https://github.com/anthropics/needle
cd needle
cargo build --release
```

## Your First Database

Let's create a simple vector search application step by step.

### 1. Create a Database

```rust
use needle::{Database, DistanceFunction};

fn main() -> needle::Result<()> {
    // Create a new database (or open existing)
    let db = Database::open("quickstart.needle")?;

    // For testing, you can also use an in-memory database
    // let db = Database::in_memory()?;

    Ok(())
}
```

### 2. Create a Collection

Collections are containers for vectors with the same dimensionality. Each collection has its own HNSW index.

```rust
// Create a collection for 384-dimensional vectors
// (common dimension for sentence-transformers/all-MiniLM-L6-v2)
db.create_collection("documents", 384, DistanceFunction::Cosine)?;
```

### 3. Insert Vectors

Vectors are stored with a unique ID and optional metadata:

```rust
use serde_json::json;

let collection = db.collection("documents")?;

// Insert vectors with metadata
collection.insert(
    "doc1",
    &embedding1,
    json!({
        "title": "Introduction to Rust",
        "category": "programming",
        "year": 2024
    })
)?;

collection.insert(
    "doc2",
    &embedding2,
    json!({
        "title": "Machine Learning Basics",
        "category": "ai",
        "year": 2023
    })
)?;
```

### 4. Search for Similar Vectors

```rust
// Search for the 5 most similar vectors
let results = collection.search(&query_vector, 5, None)?;

for result in results {
    println!(
        "ID: {}, Distance: {:.4}, Metadata: {:?}",
        result.id, result.distance, result.metadata
    );
}
```

### 5. Filter with Metadata

Use MongoDB-style filters to narrow down results:

```rust
use needle::Filter;

// Filter by category
let filter = Filter::parse(&json!({
    "category": "programming"
}))?;

let results = collection.search(&query_vector, 5, Some(&filter))?;

// Complex filter with operators
let filter = Filter::parse(&json!({
    "$and": [
        { "category": { "$in": ["programming", "ai"] } },
        { "year": { "$gte": 2023 } }
    ]
}))?;
```

### 6. Save Your Database

```rust
// Save changes to disk
db.save()?;
```

## Complete Example

Here's a complete example putting it all together:

```rust
use needle::{Database, DistanceFunction, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    // Create database and collection
    let db = Database::open("semantic_search.needle")?;
    db.create_collection("articles", 384, DistanceFunction::Cosine)?;
    let collection = db.collection("articles")?;

    // Sample embeddings (in practice, use an embedding model)
    let articles = vec![
        ("rust-intro", vec![0.1; 384], json!({"title": "Intro to Rust", "tags": ["rust", "programming"]})),
        ("ml-basics", vec![0.2; 384], json!({"title": "ML Basics", "tags": ["ai", "ml"]})),
        ("vector-db", vec![0.15; 384], json!({"title": "Vector Databases", "tags": ["database", "ai"]})),
    ];

    // Insert all articles
    for (id, embedding, metadata) in articles {
        collection.insert(id, &embedding, metadata)?;
    }

    // Search for similar articles
    let query = vec![0.12; 384];
    let results = collection.search(&query, 3, None)?;

    println!("Top 3 similar articles:");
    for result in results {
        println!("  {} (distance: {:.4})", result.id, result.distance);
    }

    // Search with filter
    let filter = Filter::parse(&json!({
        "tags": { "$in": ["rust", "database"] }
    }))?;
    let filtered = collection.search(&query, 3, Some(&filter))?;

    println!("\nFiltered results (rust or database tags):");
    for result in filtered {
        println!("  {} (distance: {:.4})", result.id, result.distance);
    }

    // Save and close
    db.save()?;

    Ok(())
}
```

## Using the CLI

Needle includes a powerful command-line interface:

```bash
# Create a new database
needle create mydb.needle

# Create a collection
needle create-collection mydb.needle -n documents -d 384

# Show database info
needle info mydb.needle

# List collections
needle collections mydb.needle

# Insert vectors from JSON
echo '{"id": "doc1", "vector": [0.1, 0.2, ...], "metadata": {"title": "Hello"}}' | \
  needle insert mydb.needle -c documents

# Search
needle search mydb.needle -c documents -q "[0.1, 0.2, ...]" -k 10

# Export collection to JSON
needle export mydb.needle -c documents > backup.json

# Import from JSON
needle import mydb.needle -c documents < backup.json
```

## Using the HTTP Server

For multi-language access, run Needle as an HTTP server:

```bash
# Start the server
needle serve -a 127.0.0.1:8080 -d mydb.needle
```

Then use the REST API:

```bash
# Create a collection
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "docs", "dimensions": 384, "distance": "cosine"}'

# Insert a vector
curl -X POST http://localhost:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": "doc1", "vector": [0.1, 0.2, ...], "metadata": {"title": "Hello"}}'

# Search
curl -X POST http://localhost:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, ...], "k": 10}'
```

## Next Steps

Now that you have Needle running, explore these topics:

- [Core Concepts](/docs/concepts/vectors) - Understand vectors, collections, and HNSW
- [Semantic Search Guide](/docs/guides/semantic-search) - Build a complete semantic search system
- [RAG Applications](/docs/guides/rag) - Power your LLM with retrieval-augmented generation
- [API Reference](/docs/api-reference) - Complete API documentation
