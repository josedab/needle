---
sidebar_position: 1
---

# Semantic Search Guide

This guide walks you through building a complete semantic search system with Needle. Semantic search understands the meaning behind queries, not just keywords.

## Overview

Traditional keyword search matches exact terms. Semantic search uses embeddings to find conceptually similar content:

| Query | Keyword Search | Semantic Search |
|-------|----------------|-----------------|
| "car maintenance" | Documents with "car" and "maintenance" | Also finds "vehicle repair", "auto service" |
| "ML algorithms" | Documents with "ML" and "algorithms" | Also finds "machine learning methods", "AI techniques" |

## Architecture

A semantic search system has three main components:

1. **Embedding Model**: Converts text to vectors
2. **Vector Database**: Stores and indexes vectors (Needle)
3. **Search API**: Handles queries and returns results

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │────▶│  Embedding  │────▶│   Needle    │
│   "How to   │     │   Model     │     │   Vector    │
│   learn     │     │             │     │   Search    │
│   Rust?"    │     │  [0.1, ...] │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Results:   │
                                        │  - Rust     │
                                        │    Tutorial │
                                        │  - Getting  │
                                        │    Started  │
                                        └─────────────┘
```

## Step 1: Set Up the Database

```rust
use needle::{Database, DistanceFunction, CollectionConfig};

fn setup_database() -> needle::Result<Database> {
    let db = Database::open("search.needle")?;

    // Create collection for 384-dim embeddings (all-MiniLM-L6-v2)
    if db.collection("documents").is_err() {
        let config = CollectionConfig::new("documents", 384)
            .with_distance(DistanceFunction::Cosine)
            .with_hnsw_m(16)
            .with_hnsw_ef_construction(200);
        db.create_collection_with_config(config)?;
    }

    Ok(db)
}
```

## Step 2: Choose an Embedding Model

### Option A: OpenAI Embeddings

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

async fn get_embeddings(texts: &[String]) -> Vec<Vec<f32>> {
    let client = Client::new();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap();

    let request = EmbeddingRequest {
        model: "text-embedding-3-small".into(),
        input: texts.to_vec(),
    };

    let response: EmbeddingResponse = client
        .post("https://api.openai.com/v1/embeddings")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    response.data.into_iter().map(|d| d.embedding).collect()
}
```

### Option B: Local ONNX Model (Recommended)

```rust
use needle::EmbeddingModel;

fn setup_embedding_model() -> needle::Result<EmbeddingModel> {
    // Downloads and caches the model automatically
    EmbeddingModel::load("all-MiniLM-L6-v2")
}

fn get_embeddings(model: &EmbeddingModel, texts: &[&str]) -> Vec<Vec<f32>> {
    texts.iter().map(|t| model.encode(t).unwrap()).collect()
}
```

## Step 3: Index Your Documents

```rust
use serde_json::json;

struct Document {
    id: String,
    title: String,
    content: String,
    url: String,
    category: String,
}

fn index_documents(
    db: &Database,
    model: &EmbeddingModel,
    documents: &[Document],
) -> needle::Result<()> {
    let collection = db.collection("documents")?;

    for doc in documents {
        // Combine title and content for embedding
        let text = format!("{}\n\n{}", doc.title, doc.content);
        let embedding = model.encode(&text)?;

        // Store with metadata for filtering and display
        collection.insert(
            &doc.id,
            &embedding,
            Some(json!({
                "title": doc.title,
                "url": doc.url,
                "category": doc.category,
                "content_preview": &doc.content[..200.min(doc.content.len())]
            })),
        )?;
    }

    db.save()?;
    Ok(())
}
```

### Batch Processing for Large Datasets

```rust
fn index_documents_batch(
    db: &Database,
    model: &EmbeddingModel,
    documents: &[Document],
    batch_size: usize,
) -> needle::Result<()> {
    let collection = db.collection("documents")?;

    for chunk in documents.chunks(batch_size) {
        // Batch embed
        let texts: Vec<_> = chunk
            .iter()
            .map(|d| format!("{}\n\n{}", d.title, d.content))
            .collect();

        let embeddings = model.encode_batch(&texts)?;

        // Batch insert
        for (doc, embedding) in chunk.iter().zip(embeddings) {
            collection.insert(
                &doc.id,
                &embedding,
                Some(json!({
                    "title": doc.title,
                    "url": doc.url,
                    "category": doc.category,
                })),
            )?;
        }
    }

    db.save()?;
    Ok(())
}
```

## Step 4: Implement Search

```rust
use needle::Filter;

#[derive(Debug)]
struct SearchResult {
    id: String,
    title: String,
    url: String,
    score: f32,
    preview: String,
}

fn search(
    db: &Database,
    model: &EmbeddingModel,
    query: &str,
    category: Option<&str>,
    limit: usize,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;

    // Embed the query
    let query_embedding = model.encode(query)?;

    // Build filter if category specified
    let filter = category.map(|cat| {
        Filter::parse(&json!({"category": cat})).unwrap()
    });

    // Search
    let results = if let Some(ref f) = filter {
        collection.search_with_filter(&query_embedding, limit, f)?
    } else {
        collection.search(&query_embedding, limit)?
    };

    // Format results
    Ok(results
        .into_iter()
        .map(|r| SearchResult {
            id: r.id,
            title: r.metadata["title"].as_str().unwrap_or("").to_string(),
            url: r.metadata["url"].as_str().unwrap_or("").to_string(),
            score: 1.0 - r.distance, // Convert distance to similarity
            preview: r.metadata["content_preview"]
                .as_str()
                .unwrap_or("")
                .to_string(),
        })
        .collect())
}
```

## Step 5: Build a Search API

```rust
use axum::{extract::Query, routing::get, Json, Router};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    category: Option<String>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    query: String,
    took_ms: u64,
}

async fn search_handler(
    Query(params): Query<SearchParams>,
) -> Json<SearchResponse> {
    let start = std::time::Instant::now();

    let db = get_database();  // Your database instance
    let model = get_model();  // Your embedding model

    let results = search(
        &db,
        &model,
        &params.q,
        params.category.as_deref(),
        params.limit.unwrap_or(10),
    ).unwrap();

    Json(SearchResponse {
        results,
        query: params.q,
        took_ms: start.elapsed().as_millis() as u64,
    })
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/search", get(search_handler));

    axum::Server::bind(&"0.0.0.0:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

## Improving Search Quality

### 1. Query Expansion

Expand queries with synonyms or related terms:

```rust
fn expand_query(query: &str) -> String {
    // Simple approach: add the query in different forms
    format!(
        "{} {} {}",
        query,
        query.to_lowercase(),
        query.replace(" ", "-")
    )
}
```

### 2. Reranking

Use a cross-encoder for more accurate ranking of top results:

```rust
use needle::Reranker;

fn search_with_reranking(
    db: &Database,
    model: &EmbeddingModel,
    reranker: &Reranker,
    query: &str,
    limit: usize,
) -> needle::Result<Vec<SearchResult>> {
    // First stage: vector search (retrieve 3x candidates)
    let candidates = search(db, model, query, None, limit * 3)?;

    // Second stage: rerank with cross-encoder
    let reranked = reranker.rerank(
        query,
        &candidates.iter().map(|c| c.preview.as_str()).collect::<Vec<_>>(),
        limit,
    )?;

    Ok(reranked)
}
```

### 3. Hybrid Search

Combine vector search with keyword matching:

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

fn hybrid_search(
    db: &Database,
    model: &EmbeddingModel,
    bm25: &Bm25Index,
    query: &str,
    limit: usize,
) -> needle::Result<Vec<SearchResult>> {
    let collection = db.collection("documents")?;

    // Vector search
    let query_embedding = model.encode(query)?;
    let vector_results = collection.search(&query_embedding, limit * 2)?;

    // BM25 search
    let bm25_results = bm25.search(query, limit * 2);

    // Fuse results
    let fused = reciprocal_rank_fusion(
        &vector_results,
        &bm25_results,
        &RrfConfig::default(),
        limit,
    );

    // Format and return
    // ...
}
```

See the [Hybrid Search Guide](/docs/guides/hybrid-search) for details.

## Best Practices

### 1. Chunk Long Documents

Embedding models have limited context windows. Chunk long documents:

```rust
fn chunk_document(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut chunks = Vec::new();

    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        chunks.push(words[start..end].join(" "));
        start += chunk_size - overlap;
    }

    chunks
}

// Index chunks separately
for (i, chunk) in chunk_document(&doc.content, 200, 50).iter().enumerate() {
    collection.insert(
        &format!("{}_{}", doc.id, i),
        &model.encode(chunk)?,
        Some(json!({
            "parent_id": doc.id,
            "chunk_index": i,
            "title": doc.title,
        })),
    )?;
}
```

### 2. Store Original Content Separately

Don't bloat metadata with full content:

```rust
// Store vectors with minimal metadata
collection.insert(&doc.id, &embedding, Some(json!({
    "title": doc.title,
    "url": doc.url,
})))?;

// Store full content elsewhere (e.g., another database)
content_store.insert(&doc.id, &doc.content)?;

// Fetch content only for displayed results
for result in &results {
    let content = content_store.get(&result.id)?;
    // Display content
}
```

### 3. Handle Updates

When documents change, update both the embedding and metadata:

```rust
fn update_document(
    db: &Database,
    model: &EmbeddingModel,
    doc: &Document,
) -> needle::Result<()> {
    let collection = db.collection("documents")?;

    // Delete old version
    collection.delete(&doc.id)?;

    // Insert updated version
    let embedding = model.encode(&format!("{}\n\n{}", doc.title, doc.content))?;
    collection.insert(&doc.id, &embedding, Some(json!({
        "title": doc.title,
        "url": doc.url,
        "updated_at": chrono::Utc::now().to_rfc3339(),
    })))?;

    Ok(())
}
```

## Next Steps

- [RAG Applications](/docs/guides/rag) - Use search for LLM context
- [Hybrid Search](/docs/guides/hybrid-search) - Combine with keyword search
- [Production Deployment](/docs/guides/production) - Scale your search system
