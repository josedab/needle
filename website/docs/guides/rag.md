---
sidebar_position: 2
---

# RAG Applications

Retrieval-Augmented Generation (RAG) combines vector search with large language models to create AI systems that can answer questions using your data. This guide shows how to build RAG applications with Needle.

## What is RAG?

RAG is a technique that:
1. **Retrieves** relevant documents using vector search
2. **Augments** an LLM prompt with retrieved context
3. **Generates** responses grounded in your data

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Question  │────▶│   Needle    │────▶│  Retrieved  │
│  "How do I  │     │   Vector    │     │   Context   │
│   deploy?"  │     │   Search    │     │             │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                    ┌─────────────┐     ┌─────────────┐
                    │   Answer    │◀────│    LLM      │
                    │  "To deploy │     │  (GPT-4,    │
                    │   first..." │     │   Claude)   │
                    └─────────────┘     └─────────────┘
```

## Benefits of RAG

- **Accurate**: Responses based on your actual documents
- **Up-to-date**: No need to retrain models for new information
- **Verifiable**: Can cite sources for answers
- **Cost-effective**: Smaller models can perform well with good context

## Basic RAG Implementation

### Step 1: Set Up Needle

```rust
use needle::{Database, DistanceFunction, EmbeddingModel};
use serde_json::json;

struct RagSystem {
    db: Database,
    model: EmbeddingModel,
}

impl RagSystem {
    fn new(db_path: &str) -> needle::Result<Self> {
        let db = Database::open(db_path)?;

        if !db.collection_exists("knowledge")? {
            db.create_collection("knowledge", 384, DistanceFunction::Cosine)?;
        }

        let model = EmbeddingModel::load("all-MiniLM-L6-v2")?;

        Ok(Self { db, model })
    }
}
```

### Step 2: Index Your Knowledge Base

```rust
impl RagSystem {
    fn add_document(&self, id: &str, content: &str, metadata: serde_json::Value) -> needle::Result<()> {
        let collection = self.db.collection("knowledge")?;

        // Chunk the document for better retrieval
        let chunks = self.chunk_text(content, 500, 100);

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_id = format!("{}_{}", id, i);
            let embedding = self.model.encode(chunk)?;

            let mut chunk_metadata = metadata.clone();
            chunk_metadata["chunk_index"] = json!(i);
            chunk_metadata["chunk_text"] = json!(chunk);

            collection.insert(&chunk_id, &embedding, chunk_metadata)?;
        }

        self.db.save()?;
        Ok(())
    }

    fn chunk_text(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
        let sentences: Vec<&str> = text.split(". ").collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut current_size = 0;

        for sentence in sentences {
            let sentence_len = sentence.split_whitespace().count();

            if current_size + sentence_len > chunk_size && !current_chunk.is_empty() {
                chunks.push(current_chunk.clone());

                // Keep overlap from the end
                let words: Vec<&str> = current_chunk.split_whitespace().collect();
                let overlap_start = words.len().saturating_sub(overlap);
                current_chunk = words[overlap_start..].join(" ");
                current_size = overlap;
            }

            current_chunk.push_str(sentence);
            current_chunk.push_str(". ");
            current_size += sentence_len;
        }

        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }
}
```

### Step 3: Retrieve Relevant Context

```rust
impl RagSystem {
    fn retrieve(&self, query: &str, k: usize) -> needle::Result<Vec<String>> {
        let collection = self.db.collection("knowledge")?;
        let query_embedding = self.model.encode(query)?;

        let results = collection.search(&query_embedding, k, None)?;

        Ok(results
            .into_iter()
            .map(|r| r.metadata["chunk_text"].as_str().unwrap_or("").to_string())
            .collect())
    }
}
```

### Step 4: Generate Answers

```rust
use reqwest::Client;

impl RagSystem {
    async fn answer(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Retrieve relevant context
        let contexts = self.retrieve(question, 5)?;

        // Build prompt
        let context_text = contexts.join("\n\n---\n\n");
        let prompt = format!(
            r#"Answer the question based on the provided context. If the context doesn't contain relevant information, say so.

Context:
{}

Question: {}

Answer:"#,
            context_text, question
        );

        // Call LLM (example with OpenAI)
        let response = self.call_openai(&prompt).await?;

        Ok(response)
    }

    async fn call_openai(&self, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
        let client = Client::new();
        let api_key = std::env::var("OPENAI_API_KEY")?;

        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&json!({
                "model": "gpt-4",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }))
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        Ok(response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }
}
```

## Advanced RAG Techniques

### 1. Hybrid Retrieval

Combine semantic search with keyword matching for better recall:

```rust
use needle::{Bm25Index, reciprocal_rank_fusion, RrfConfig};

impl RagSystem {
    fn hybrid_retrieve(&self, query: &str, k: usize) -> needle::Result<Vec<String>> {
        let collection = self.db.collection("knowledge")?;

        // Vector search
        let query_embedding = self.model.encode(query)?;
        let vector_results = collection.search(&query_embedding, k * 2, None)?;

        // BM25 search
        let bm25_results = self.bm25_index.search(query, k * 2);

        // Fuse results
        let fused = reciprocal_rank_fusion(
            &vector_results,
            &bm25_results,
            &RrfConfig::default(),
            k,
        );

        Ok(fused
            .into_iter()
            .map(|r| r.metadata["chunk_text"].as_str().unwrap_or("").to_string())
            .collect())
    }
}
```

### 2. Query Transformation

Improve retrieval by transforming the query:

```rust
impl RagSystem {
    async fn answer_with_query_transform(&self, question: &str) -> Result<String, Box<dyn std::error::Error>> {
        // Generate search queries from the question
        let search_queries = self.generate_search_queries(question).await?;

        // Retrieve for each query
        let mut all_contexts = Vec::new();
        for query in &search_queries {
            let contexts = self.retrieve(query, 3)?;
            all_contexts.extend(contexts);
        }

        // Deduplicate
        all_contexts.sort();
        all_contexts.dedup();

        // Generate answer with expanded context
        self.generate_answer(question, &all_contexts).await
    }

    async fn generate_search_queries(&self, question: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let prompt = format!(
            r#"Generate 3 different search queries to find information relevant to answering this question. Return only the queries, one per line.

Question: {}

Search queries:"#,
            question
        );

        let response = self.call_openai(&prompt).await?;
        Ok(response.lines().map(String::from).collect())
    }
}
```

### 3. Contextual Compression

Compress retrieved documents to include only relevant parts:

```rust
impl RagSystem {
    async fn retrieve_compressed(&self, query: &str, k: usize) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let raw_contexts = self.retrieve(query, k * 2)?;

        let mut compressed = Vec::new();
        for context in raw_contexts {
            let relevant_part = self.compress_context(&context, query).await?;
            if !relevant_part.is_empty() {
                compressed.push(relevant_part);
            }
            if compressed.len() >= k {
                break;
            }
        }

        Ok(compressed)
    }

    async fn compress_context(&self, context: &str, query: &str) -> Result<String, Box<dyn std::error::Error>> {
        let prompt = format!(
            r#"Extract only the sentences from this text that are relevant to answering the query. If nothing is relevant, return an empty string.

Query: {}

Text: {}

Relevant sentences:"#,
            query, context
        );

        self.call_openai(&prompt).await
    }
}
```

### 4. Citation Support

Track sources for generated answers:

```rust
#[derive(Debug)]
struct AnswerWithCitations {
    answer: String,
    citations: Vec<Citation>,
}

#[derive(Debug)]
struct Citation {
    text: String,
    source: String,
    chunk_id: String,
}

impl RagSystem {
    async fn answer_with_citations(&self, question: &str) -> Result<AnswerWithCitations, Box<dyn std::error::Error>> {
        let collection = self.db.collection("knowledge")?;
        let query_embedding = self.model.encode(question)?;
        let results = collection.search(&query_embedding, 5, None)?;

        // Build context with source markers
        let mut context_parts = Vec::new();
        let mut citations = Vec::new();

        for (i, r) in results.iter().enumerate() {
            let text = r.metadata["chunk_text"].as_str().unwrap_or("");
            let source = r.metadata["source"].as_str().unwrap_or("Unknown");

            context_parts.push(format!("[{}] {}", i + 1, text));
            citations.push(Citation {
                text: text.to_string(),
                source: source.to_string(),
                chunk_id: r.id.clone(),
            });
        }

        let prompt = format!(
            r#"Answer the question based on the provided context. Cite sources using [1], [2], etc.

Context:
{}

Question: {}

Answer with citations:"#,
            context_parts.join("\n\n"),
            question
        );

        let answer = self.call_openai(&prompt).await?;

        Ok(AnswerWithCitations { answer, citations })
    }
}
```

## Production Considerations

### Caching

Cache embeddings and LLM responses:

```rust
use std::collections::HashMap;

struct CachedRagSystem {
    rag: RagSystem,
    embedding_cache: HashMap<String, Vec<f32>>,
    response_cache: HashMap<String, String>,
}

impl CachedRagSystem {
    fn get_embedding(&mut self, text: &str) -> needle::Result<Vec<f32>> {
        if let Some(cached) = self.embedding_cache.get(text) {
            return Ok(cached.clone());
        }

        let embedding = self.rag.model.encode(text)?;
        self.embedding_cache.insert(text.to_string(), embedding.clone());
        Ok(embedding)
    }
}
```

### Streaming Responses

Stream LLM responses for better UX:

```rust
use futures::StreamExt;

impl RagSystem {
    async fn answer_streaming(
        &self,
        question: &str,
        mut callback: impl FnMut(&str),
    ) -> Result<String, Box<dyn std::error::Error>> {
        let contexts = self.retrieve(question, 5)?;
        let prompt = self.build_prompt(question, &contexts);

        // Stream from OpenAI
        let mut stream = self.stream_openai(&prompt).await?;
        let mut full_response = String::new();

        while let Some(chunk) = stream.next().await {
            callback(&chunk);
            full_response.push_str(&chunk);
        }

        Ok(full_response)
    }
}
```

### Evaluation

Measure RAG quality:

```rust
struct RagEvaluator {
    rag: RagSystem,
}

impl RagEvaluator {
    async fn evaluate(&self, test_cases: &[(String, String, Vec<String>)]) -> EvaluationResults {
        let mut results = EvaluationResults::default();

        for (question, expected_answer, expected_sources) in test_cases {
            // Check retrieval quality
            let retrieved = self.rag.retrieve(question, 5).unwrap();
            let retrieval_recall = self.compute_recall(&retrieved, expected_sources);

            // Check answer quality
            let answer = self.rag.answer(question).await.unwrap();
            let answer_score = self.score_answer(&answer, expected_answer).await;

            results.add(retrieval_recall, answer_score);
        }

        results
    }
}
```

## Complete Example

```rust
use needle::{Database, DistanceFunction, EmbeddingModel};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    let rag = RagSystem::new("rag_demo.needle")?;

    // Index some documents
    rag.add_document(
        "rust_docs",
        "Rust is a systems programming language focused on safety...",
        json!({"source": "Rust Documentation", "url": "https://doc.rust-lang.org"}),
    )?;

    rag.add_document(
        "needle_docs",
        "Needle is an embedded vector database written in Rust...",
        json!({"source": "Needle Documentation", "url": "https://needle.dev"}),
    )?;

    // Ask questions
    let answer = rag.answer("What is Needle?").await?;
    println!("Answer: {}", answer);

    let answer = rag.answer("How do I create a collection in Needle?").await?;
    println!("Answer: {}", answer);

    Ok(())
}
```

## Next Steps

- [Hybrid Search](/docs/guides/hybrid-search) - Improve retrieval quality
- [Production Deployment](/docs/guides/production) - Scale your RAG system
- [API Reference](/docs/api-reference) - Complete API documentation
