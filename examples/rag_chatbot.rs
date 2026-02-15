//! RAG Chatbot Example
//!
//! This example demonstrates building a simple Retrieval-Augmented Generation (RAG)
//! chatbot using Needle as the vector database for document retrieval.
//!
//! Run with: cargo run --example rag_chatbot

use needle::{CollectionConfig, Database, DistanceFunction, Filter};
use serde_json::json;

/// Document to be indexed
struct Document {
    id: String,
    title: String,
    content: String,
    source: String,
}

/// Knowledge base for the chatbot
struct KnowledgeBase {
    db: Database,
    collection_name: String,
    #[allow(dead_code)] // Reserved for embedding validation
    embedding_dim: usize,
}

impl KnowledgeBase {
    /// Create a new knowledge base
    fn new(embedding_dim: usize) -> needle::Result<Self> {
        let db = Database::in_memory();

        let collection_name = "knowledge".to_string();
        let config = CollectionConfig::new(&collection_name, embedding_dim)
            .with_distance(DistanceFunction::Cosine);

        db.create_collection_with_config(config)?;

        Ok(Self {
            db,
            collection_name,
            embedding_dim,
        })
    }

    /// Add a document to the knowledge base
    fn add_document(&self, doc: &Document, embedding: &[f32]) -> needle::Result<()> {
        let collection = self.db.collection(&self.collection_name)?;

        let metadata = json!({
            "title": doc.title,
            "source": doc.source,
            "content_preview": &doc.content[..doc.content.len().min(200)],
        });

        collection.insert(&doc.id, embedding, Some(metadata))?;
        Ok(())
    }

    /// Add chunked document (for long documents)
    #[allow(dead_code)] // Example method for demonstration
    fn add_chunked_document(
        &self,
        doc: &Document,
        chunks: &[(String, usize, usize)], // (text, start, end)
        embeddings: &[Vec<f32>],
    ) -> needle::Result<()> {
        let collection = self.db.collection(&self.collection_name)?;

        for (i, ((chunk_text, start, end), embedding)) in
            chunks.iter().zip(embeddings.iter()).enumerate()
        {
            let chunk_id = format!("{}_{}", doc.id, i);

            let metadata = json!({
                "document_id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "chunk_index": i,
                "chunk_text": chunk_text,
                "start_offset": start,
                "end_offset": end,
            });

            collection.insert(&chunk_id, embedding, Some(metadata))?;
        }

        Ok(())
    }

    /// Retrieve relevant documents for a query
    fn retrieve(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        source_filter: Option<&str>,
    ) -> needle::Result<Vec<RetrievalResult>> {
        let collection = self.db.collection(&self.collection_name)?;

        let results = if let Some(source) = source_filter {
            let filter = Filter::eq("source", source);
            collection.search_with_filter(query_embedding, top_k, &filter)?
        } else {
            collection.search(query_embedding, top_k)?
        };

        Ok(results
            .into_iter()
            .map(|r| {
                let metadata = r.metadata.unwrap_or_default();
                RetrievalResult {
                    id: r.id,
                    score: 1.0 - r.distance, // Convert distance to similarity
                    title: metadata["title"].as_str().unwrap_or("").to_string(),
                    source: metadata["source"].as_str().unwrap_or("").to_string(),
                    content: metadata
                        .get("chunk_text")
                        .or(metadata.get("content_preview"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                }
            })
            .collect())
    }
}

/// Result from retrieval
#[allow(dead_code)] // Fields used for demonstration purposes
struct RetrievalResult {
    id: String,
    score: f32,
    title: String,
    source: String,
    content: String,
}

/// Simple RAG chatbot
struct RagChatbot {
    knowledge_base: KnowledgeBase,
    context_window_size: usize,
    system_prompt: String,
}

impl RagChatbot {
    /// Create a new RAG chatbot
    fn new(embedding_dim: usize) -> needle::Result<Self> {
        let knowledge_base = KnowledgeBase::new(embedding_dim)?;

        Ok(Self {
            knowledge_base,
            context_window_size: 5,
            system_prompt: "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain relevant information, say so.".to_string(),
        })
    }

    /// Add knowledge to the chatbot
    fn learn(&self, doc: &Document, embedding: &[f32]) -> needle::Result<()> {
        self.knowledge_base.add_document(doc, embedding)
    }

    /// Generate a response (simulated - in real use, you'd call an LLM)
    fn respond(&self, query: &str, query_embedding: &[f32]) -> needle::Result<ChatResponse> {
        // Retrieve relevant context
        let retrieved =
            self.knowledge_base
                .retrieve(query_embedding, self.context_window_size, None)?;

        // Build context from retrieved documents
        let context: Vec<String> = retrieved
            .iter()
            .map(|r| format!("[{}] {}", r.title, r.content))
            .collect();

        // In a real implementation, you would:
        // 1. Format the context and query into a prompt
        // 2. Send to an LLM (OpenAI, Claude, etc.)
        // 3. Return the LLM's response

        // For this example, we'll return a simulated response
        let prompt = format!(
            "{}\n\nContext:\n{}\n\nQuestion: {}",
            self.system_prompt,
            context.join("\n\n"),
            query
        );

        Ok(ChatResponse {
            answer: format!(
                "Based on the retrieved context, I found {} relevant documents. \
                The most relevant information comes from '{}' (score: {:.2}). \
                [In a real implementation, an LLM would generate a natural response here]",
                retrieved.len(),
                retrieved
                    .first()
                    .map(|r| r.title.as_str())
                    .unwrap_or("unknown"),
                retrieved.first().map(|r| r.score).unwrap_or(0.0)
            ),
            sources: retrieved.iter().map(|r| r.source.clone()).collect(),
            context_used: context,
            prompt_preview: prompt[..prompt.len().min(500)].to_string(),
        })
    }
}

/// Chat response with sources
#[allow(dead_code)] // Fields available for integration with real LLM
struct ChatResponse {
    answer: String,
    sources: Vec<String>,
    context_used: Vec<String>,
    prompt_preview: String,
}

/// Generate a mock embedding (in real use, use an embedding model)
fn mock_embedding(text: &str, dim: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();

    let mut rng_state = seed;
    (0..dim)
        .map(|_| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng_state >> 16) as f32 / 32768.0) - 1.0
        })
        .collect()
}

fn main() -> needle::Result<()> {
    println!("=== RAG Chatbot Example ===\n");

    // Create chatbot with 384-dimensional embeddings
    let embedding_dim = 384;
    let chatbot = RagChatbot::new(embedding_dim)?;

    // Add some knowledge documents
    let documents = vec![
        Document {
            id: "doc1".to_string(),
            title: "Introduction to Vector Databases".to_string(),
            content: "Vector databases are specialized database systems designed to store and query high-dimensional vectors. They are essential for AI applications that use embeddings, such as semantic search, recommendation systems, and retrieval-augmented generation (RAG).".to_string(),
            source: "tutorial/vector-databases".to_string(),
        },
        Document {
            id: "doc2".to_string(),
            title: "HNSW Algorithm".to_string(),
            content: "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It provides excellent query performance with sub-linear complexity by building a multi-layer graph structure where each layer contains a subset of the nodes.".to_string(),
            source: "algorithms/hnsw".to_string(),
        },
        Document {
            id: "doc3".to_string(),
            title: "Building RAG Applications".to_string(),
            content: "RAG (Retrieval-Augmented Generation) combines information retrieval with language models. The process involves: 1) Converting documents to embeddings and storing them, 2) Retrieving relevant documents based on query similarity, 3) Using retrieved context to augment LLM prompts.".to_string(),
            source: "guides/rag".to_string(),
        },
        Document {
            id: "doc4".to_string(),
            title: "Needle Database Features".to_string(),
            content: "Needle is an embedded vector database written in Rust. Key features include: single-file storage, HNSW indexing for fast queries, metadata filtering, multiple distance functions, and quantization for memory efficiency. It's designed to be the SQLite for vectors.".to_string(),
            source: "docs/needle".to_string(),
        },
    ];

    // Index documents
    println!("Indexing {} documents...", documents.len());
    for doc in &documents {
        let embedding = mock_embedding(&doc.content, embedding_dim);
        chatbot.learn(doc, &embedding)?;
    }
    println!("Knowledge base ready!\n");

    // Example queries
    let queries = vec![
        "What is a vector database?",
        "How does HNSW work?",
        "How do I build a RAG application?",
        "What features does Needle have?",
    ];

    for query in queries {
        println!("User: {}", query);

        let query_embedding = mock_embedding(query, embedding_dim);
        let response = chatbot.respond(query, &query_embedding)?;

        println!("Assistant: {}", response.answer);
        println!("Sources: {:?}", response.sources);
        println!();
    }

    // Demonstrate simple text chunking
    println!("\n=== Document Chunking Demo ===\n");

    let long_document = "Vector databases are revolutionizing AI applications. \
        They store embeddings efficiently. \
        HNSW provides fast nearest neighbor search. \
        Needle is designed for simplicity. \
        RAG applications combine retrieval and generation. \
        The future of AI involves sophisticated retrieval systems.";

    // Simple sentence-based chunking
    let chunks = chunk_by_sentences(long_document, 2);
    println!("Original document length: {} chars", long_document.len());
    println!("Number of chunks: {}", chunks.len());

    for (i, (chunk_text, start, end)) in chunks.iter().enumerate() {
        let preview = if chunk_text.len() > 50 {
            format!("{}...", &chunk_text[..50])
        } else {
            chunk_text.clone()
        };
        println!("Chunk {}: \"{}\" (chars {}-{})", i, preview, start, end);
    }

    println!("\nRAG Chatbot example complete!");
    Ok(())
}

/// Simple sentence-based chunking for demonstration
fn chunk_by_sentences(text: &str, sentences_per_chunk: usize) -> Vec<(String, usize, usize)> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut chunk_start = 0;
    let mut sentence_count = 0;
    let mut pos = 0;

    for sentence in text.split_inclusive(['.', '!', '?']) {
        let sentence = sentence.trim();
        if sentence.is_empty() {
            continue;
        }

        if sentence_count >= sentences_per_chunk && !current_chunk.is_empty() {
            let chunk_end = pos;
            chunks.push((current_chunk.trim().to_string(), chunk_start, chunk_end));
            current_chunk = String::new();
            chunk_start = pos;
            sentence_count = 0;
        }

        if !current_chunk.is_empty() {
            current_chunk.push(' ');
        }
        current_chunk.push_str(sentence);
        sentence_count += 1;
        pos += sentence.len() + 1; // +1 for space
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push((current_chunk.trim().to_string(), chunk_start, text.len()));
    }

    chunks
}
