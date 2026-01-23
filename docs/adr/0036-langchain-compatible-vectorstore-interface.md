# ADR-0036: LangChain-Compatible VectorStore Interface

## Status

Accepted

## Context

LangChain has become the dominant framework for building LLM applications, with millions of developers using its abstractions:

1. **Market adoption** — LangChain is the de facto standard for RAG applications
2. **Developer familiarity** — Users expect VectorStore-like interfaces
3. **Ecosystem integration** — LangChain chains, agents, and tools assume specific APIs
4. **Migration ease** — Users switching from Pinecone/Weaviate expect similar patterns

LangChain's VectorStore interface includes:
- `Document` abstraction (page_content + metadata)
- `add_documents()` / `add_texts()`
- `similarity_search()` / `similarity_search_with_score()`
- `max_marginal_relevance_search()` (MMR for diversity)
- Async variants (`asimilarity_search`, etc.)

## Decision

Implement a **LangChain-compatible interface** in `src/langchain.rs` that mirrors LangChain's Python API naming and semantics.

### Document Abstraction

```rust
/// Compatible with LangChain's Document class
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// The main text content (LangChain: page_content)
    pub page_content: String,

    /// Metadata associated with the document
    pub metadata: Value,

    /// Unique identifier
    pub id: String,
}

impl Document {
    pub fn new(content: &str) -> Self {
        Self {
            page_content: content.to_string(),
            metadata: Value::Object(Default::default()),
            id: Uuid::new_v4().to_string(),
        }
    }

    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}
```

### VectorStore Interface

```rust
pub struct NeedleVectorStore {
    collection: Collection,
    config: NeedleVectorStoreConfig,
}

pub struct NeedleVectorStoreConfig {
    pub collection_name: String,
    pub embedding_dimension: usize,
    pub distance_function: DistanceFunction,
    pub text_key: String,  // Metadata key for document text (default: "text")
}

impl NeedleVectorStore {
    // === Core VectorStore methods (LangChain naming) ===

    /// Add documents with pre-computed embeddings
    pub fn add_documents(
        &self,
        documents: &[Document],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<String>> {
        // Returns document IDs
    }

    /// Add texts (convenience wrapper)
    pub fn add_texts(
        &self,
        texts: &[&str],
        embeddings: &[Vec<f32>],
        metadatas: Option<&[Value]>,
    ) -> Result<Vec<String>> {
        let documents: Vec<Document> = texts.iter()
            .zip(metadatas.unwrap_or(&vec![Value::Null; texts.len()]))
            .map(|(text, meta)| Document::new(text).with_metadata(meta.clone()))
            .collect();

        self.add_documents(&documents, embeddings)
    }

    /// Similarity search (returns documents only)
    pub fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        self.similarity_search_with_score(query_embedding, k)
            .map(|results| results.into_iter().map(|(doc, _)| doc).collect())
    }

    /// Similarity search with relevance scores
    pub fn similarity_search_with_score(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(Document, f32)>> {
        let results = self.collection.search(query_embedding, k, None)?;

        results.into_iter()
            .map(|r| {
                let doc = Document {
                    page_content: r.metadata.get(&self.config.text_key)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    metadata: r.metadata.clone(),
                    id: r.id.clone(),
                };
                Ok((doc, 1.0 - r.distance))  // Convert distance to similarity
            })
            .collect()
    }

    /// Maximum Marginal Relevance search (diversity)
    pub fn max_marginal_relevance_search(
        &self,
        query_embedding: &[f32],
        k: usize,
        fetch_k: usize,      // Candidates to consider
        lambda_mult: f32,    // Diversity vs. relevance tradeoff (0-1)
    ) -> Result<Vec<Document>> {
        // Fetch more candidates
        let candidates = self.collection.search(query_embedding, fetch_k, None)?;

        // MMR selection
        let selected = self.mmr_select(&candidates, query_embedding, k, lambda_mult)?;

        Ok(selected)
    }

    // === Async variants (LangChain naming convention) ===

    pub async fn asimilarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        tokio::task::spawn_blocking({
            let this = self.clone();
            let query = query_embedding.to_vec();
            move || this.similarity_search(&query, k)
        }).await?
    }

    pub async fn aadd_documents(
        &self,
        documents: &[Document],
        embeddings: &[Vec<f32>],
    ) -> Result<Vec<String>> {
        // Async wrapper
    }

    // === Filter support ===

    pub fn similarity_search_with_filter(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: &Value,  // LangChain-style filter dict
    ) -> Result<Vec<Document>> {
        let needle_filter = Filter::parse(filter)?;
        let results = self.collection.search(query_embedding, k, Some(&needle_filter))?;
        // ... convert to Documents
    }
}
```

### MMR Implementation

```rust
impl NeedleVectorStore {
    /// Maximum Marginal Relevance selection
    fn mmr_select(
        &self,
        candidates: &[SearchResult],
        query: &[f32],
        k: usize,
        lambda: f32,
    ) -> Result<Vec<Document>> {
        let mut selected: Vec<usize> = Vec::new();
        let mut remaining: HashSet<usize> = (0..candidates.len()).collect();

        while selected.len() < k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_score = f32::NEG_INFINITY;

            for &idx in &remaining {
                let relevance = 1.0 - candidates[idx].distance;

                let diversity = if selected.is_empty() {
                    0.0
                } else {
                    selected.iter()
                        .map(|&s| self.similarity(&candidates[idx], &candidates[s]))
                        .fold(f32::NEG_INFINITY, f32::max)
                };

                let mmr_score = lambda * relevance - (1.0 - lambda) * diversity;

                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_idx = idx;
                }
            }

            selected.push(best_idx);
            remaining.remove(&best_idx);
        }

        Ok(selected.iter().map(|&i| self.to_document(&candidates[i])).collect())
    }
}
```

### Code References

- `src/langchain.rs:86-100` — Document struct matching LangChain
- `src/langchain.rs` — NeedleVectorStore implementation
- `src/langchain.rs` — MMR search implementation

## Consequences

### Benefits

1. **Drop-in replacement** — Existing LangChain code works with minimal changes
2. **Familiar API** — Developers don't need to learn new patterns
3. **Ecosystem compatibility** — Works with LangChain chains, agents, tools
4. **Migration path** — Easy switch from Pinecone, Weaviate, Chroma
5. **Dual API** — Use native Needle API for performance, LangChain API for convenience

### Tradeoffs

1. **API surface duplication** — Similar functionality exposed twice
2. **Naming conventions** — Rust style vs. Python style (`snake_case` vs. `add_documents`)
3. **Text extraction** — Assumes text stored in metadata, not always true
4. **Embedding responsibility** — User must provide embeddings (no built-in embedding)

### What This Enabled

- LangChain tutorials work out-of-the-box with Needle
- Migration from other vector stores in hours, not days
- Integration with LangChain Expression Language (LCEL)
- Compatibility with LangServe and LangSmith

### What This Prevented

- Forcing users to learn Needle-specific APIs
- Breaking existing LangChain application code
- Reimplementing standard patterns (MMR, filtering)

### Usage Example

```rust
use needle::langchain::{Document, NeedleVectorStore, NeedleVectorStoreConfig};
use serde_json::json;

fn main() -> Result<()> {
    // Create vector store
    let config = NeedleVectorStoreConfig::new("documents", 384);
    let store = NeedleVectorStore::new(config)?;

    // Add documents (like LangChain)
    let docs = vec![
        Document::new("Machine learning is a subset of AI.")
            .with_metadata(json!({"source": "intro.txt", "page": 1})),
        Document::new("Neural networks are inspired by the brain.")
            .with_metadata(json!({"source": "nn.txt", "page": 1})),
    ];

    let embeddings = embed_documents(&docs)?;  // Your embedding function
    store.add_documents(&docs, &embeddings)?;

    // Search (like LangChain)
    let query_embedding = embed_query("What is ML?")?;
    let results = store.similarity_search_with_score(&query_embedding, 5)?;

    for (doc, score) in results {
        println!("Score: {:.4}", score);
        println!("Content: {}", doc.page_content);
        println!("Source: {}", doc.metadata["source"]);
    }

    // MMR search for diversity
    let diverse_results = store.max_marginal_relevance_search(
        &query_embedding,
        k: 4,
        fetch_k: 20,
        lambda_mult: 0.5,  // Balance relevance and diversity
    )?;

    Ok(())
}
```

### Python Integration (via PyO3)

```python
from needle import NeedleVectorStore

# Same API as LangChain vector stores
store = NeedleVectorStore(collection_name="docs", embedding_dimension=384)

# Add documents
store.add_documents(documents, embeddings)

# Search
results = store.similarity_search(query_embedding, k=5)

# Use with LangChain
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever())
```
