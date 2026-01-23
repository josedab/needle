---
slug: building-rag-applications
title: Building RAG Applications with Needle
authors: [needle-team]
tags: [tutorial, rag, vector-search]
---

Retrieval-Augmented Generation (RAG) has become the go-to pattern for building AI applications that need access to custom knowledge. In this guide, we'll build a complete RAG system using Needle.

<!-- truncate -->

## What is RAG?

RAG combines the power of large language models with your own data:

1. **Retrieve** relevant documents using vector similarity search
2. **Augment** the LLM prompt with retrieved context
3. **Generate** a response grounded in your data

This approach gives you:
- Up-to-date information beyond the LLM's training cutoff
- Domain-specific knowledge
- Verifiable, sourced answers
- Reduced hallucinations

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────┐
│  Embedding  │  ← Convert query to vector
│    Model    │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Needle    │  ← Find similar documents
│  Database   │
└─────────────┘
    │
    ▼
┌─────────────┐
│     LLM     │  ← Generate answer with context
│             │
└─────────────┘
    │
    ▼
  Response
```

## Step 1: Index Your Documents

First, chunk your documents and create embeddings:

```rust
use needle::{Database, DistanceFunction};
use serde_json::json;

// Initialize
let db = Database::open("knowledge.needle")?;
db.create_collection("documents", 384, DistanceFunction::Cosine)?;
let collection = db.collection("documents")?;

// Process documents
for doc in documents {
    let chunks = chunk_document(&doc.text, 512, 50);  // size, overlap

    for (i, chunk) in chunks.iter().enumerate() {
        let embedding = embedding_model.encode(chunk)?;

        collection.insert(
            &format!("{}_{}", doc.id, i),
            &embedding,
            json!({
                "source": doc.source,
                "title": doc.title,
                "chunk_index": i,
                "text": chunk
            })
        )?;
    }
}

db.save()?;
```

## Step 2: Implement Retrieval

Create a retrieval function that finds relevant context:

```rust
fn retrieve_context(
    collection: &CollectionRef,
    query: &str,
    embedding_model: &EmbeddingModel,
    top_k: usize,
) -> Result<Vec<String>> {
    // Embed the query
    let query_embedding = embedding_model.encode(query)?;

    // Search for similar documents
    let results = collection.search(&query_embedding, top_k, None)?;

    // Extract text from results
    let context: Vec<String> = results
        .iter()
        .map(|r| r.metadata["text"].as_str().unwrap().to_string())
        .collect();

    Ok(context)
}
```

## Step 3: Build the Prompt

Construct a prompt with retrieved context:

```rust
fn build_prompt(query: &str, context: &[String]) -> String {
    let context_text = context
        .iter()
        .enumerate()
        .map(|(i, c)| format!("[{}] {}", i + 1, c))
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "Use the following context to answer the question. \
         If the context doesn't contain relevant information, say so.\n\n\
         Context:\n{}\n\n\
         Question: {}\n\n\
         Answer:",
        context_text,
        query
    )
}
```

## Step 4: Generate Response

Send the augmented prompt to your LLM:

```rust
async fn generate_response(
    collection: &CollectionRef,
    embedding_model: &EmbeddingModel,
    llm_client: &LlmClient,
    query: &str,
) -> Result<String> {
    // Retrieve relevant context
    let context = retrieve_context(collection, query, embedding_model, 5)?;

    // Build the prompt
    let prompt = build_prompt(query, &context);

    // Generate response
    let response = llm_client.complete(&prompt).await?;

    Ok(response)
}
```

## Improving Retrieval with Hybrid Search

For better results, combine vector search with BM25:

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

// Build BM25 index alongside vector index
let mut bm25 = Bm25Index::default();
for doc in documents {
    bm25.index_document(&doc.id, &doc.text);
}

// Hybrid retrieval
fn hybrid_retrieve(
    collection: &CollectionRef,
    bm25: &Bm25Index,
    query: &str,
    query_embedding: &[f32],
    top_k: usize,
) -> Vec<SearchResult> {
    let vector_results = collection.search(query_embedding, top_k * 2, None)?;
    let bm25_results = bm25.search(query, top_k * 2);

    reciprocal_rank_fusion(
        &vector_results,
        &bm25_results,
        &RrfConfig::default(),
        top_k
    )
}
```

## Production Tips

### 1. Use Metadata Filtering

Scope searches to relevant document types:

```rust
let filter = Filter::parse(&json!({
    "source": "technical_docs",
    "updated_after": "2024-01-01"
}))?;

let results = collection.search(&query_embedding, 10, Some(&filter))?;
```

### 2. Implement Re-ranking

Use a cross-encoder for better precision:

```rust
let candidates = collection.search(&query_embedding, 50, None)?;
let reranked = reranker.rerank(query, &candidates, 10)?;
```

### 3. Cache Embeddings

Avoid recomputing embeddings for repeated queries:

```rust
use std::collections::HashMap;

let mut cache: HashMap<String, Vec<f32>> = HashMap::new();

fn get_embedding(query: &str, cache: &mut HashMap<String, Vec<f32>>) -> Vec<f32> {
    cache.entry(query.to_string())
        .or_insert_with(|| embedding_model.encode(query).unwrap())
        .clone()
}
```

### 4. Monitor and Iterate

Track retrieval quality:

```rust
let (results, explain) = collection.search_explain(&query_embedding, 10, None)?;
metrics.observe("retrieval_latency", explain.total_time);
metrics.observe("nodes_visited", explain.nodes_visited);
```

## Complete Example

See our [RAG Guide](/docs/guides/rag) for a complete, production-ready implementation.

## Conclusion

Needle makes building RAG applications straightforward:

- **Fast retrieval**: Sub-10ms searches keep your application responsive
- **Hybrid search**: Combine semantic and keyword search for better results
- **Filtering**: Scope searches with metadata
- **Simple deployment**: No infrastructure to manage

Happy building!
