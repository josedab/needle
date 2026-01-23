---
slug: introducing-needle
title: Introducing Needle - SQLite for Vectors
authors: [needle-team]
tags: [announcement, release, vector-search]
---

We're excited to introduce **Needle**, an embedded vector database written in Rust. Think of it as "SQLite for vectors"—a simple, fast, single-file database for storing and searching vector embeddings.

<!-- truncate -->

## Why Needle?

As AI applications become mainstream, the need for efficient vector storage and search has grown exponentially. While there are many excellent vector databases available, we saw an opportunity for something different:

- **Truly embedded**: No servers, no containers, no network calls
- **Single-file storage**: Easy backup, distribution, and version control
- **Zero configuration**: Works out of the box with sensible defaults
- **Blazing fast**: Sub-10ms search on millions of vectors
- **Written in Rust**: Memory safety without garbage collection overhead

## Key Features

### HNSW Indexing

Needle uses the Hierarchical Navigable Small World (HNSW) algorithm for approximate nearest neighbor search. This gives you:

- **3-10ms search latency** on 1M vectors
- **95-99% recall** depending on configuration
- **Automatic parameter tuning** for your workload

### Single-File Storage

Your entire database is one `.needle` file:

```bash
cp vectors.needle backup.needle  # That's your backup
```

### Rich Filtering

MongoDB-style metadata queries work seamlessly with vector search:

```rust
let filter = Filter::parse(&json!({
    "$and": [
        {"category": {"$in": ["books", "articles"]}},
        {"year": {"$gte": 2020}}
    ]
}))?;

let results = collection.search(&query, 10, Some(&filter))?;
```

### Multiple Language Bindings

Use Needle from your language of choice:

- **Rust** (native)
- **Python** via PyO3
- **JavaScript/WASM** for browsers and Node.js
- **Swift** and **Kotlin** for mobile apps

## Getting Started

Add Needle to your project:

```bash
cargo add needle
```

Create and search in five lines:

```rust
let db = Database::open("vectors.needle")?;
db.create_collection("docs", 384, DistanceFunction::Cosine)?;
let collection = db.collection("docs")?;
collection.insert("doc1", &embedding, json!({"title": "Hello"}))?;
let results = collection.search(&query, 10, None)?;
```

## What's Next?

We're just getting started. On our roadmap:

- IVF and DiskANN index types
- Distributed sharding
- RBAC and audit logging
- More language bindings

Check out our [documentation](/docs/) to get started, and give us a star on [GitHub](https://github.com/anthropics/needle)!
