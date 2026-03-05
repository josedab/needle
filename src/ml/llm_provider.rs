#![allow(dead_code)]

//! LLM Provider Integration for RAG Pipeline
//!
//! Provides the `LlmProvider` trait for integrating LLM backends (OpenAI, Anthropic,
//! Ollama) into the RAG pipeline. Supports synchronous generation, streaming
//! via channel-based token delivery, and configurable prompt templates.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::ml::llm_provider::{
//!     LlmProvider, LlmConfig, PromptTemplate, RagWithLlm, MockLlmProvider,
//! };
//! use needle::ml::rag::{RagPipeline, MockEmbedder};
//!
//! // Create LLM provider
//! let llm = MockLlmProvider::new();
//!
//! // Create RAG-with-LLM wrapper
//! let rag_llm = RagWithLlm::new(pipeline, Box::new(llm), LlmConfig::default());
//!
//! // End-to-end: query → retrieve → generate
//! let response = rag_llm.ask("What is machine learning?", &embedder)?;
//! println!("{}", response.answer);
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use super::rag::{Embedder, RagPipeline, RetrievedChunk};

// ============================================================================
// LLM Provider Trait
// ============================================================================

/// Configuration for an LLM generation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// The prompt to send to the LLM.
    pub prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic, 1.0 = creative).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Stop sequences.
    pub stop_sequences: Vec<String>,
    /// System message (if supported by the provider).
    pub system_message: Option<String>,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            max_tokens: 1024,
            temperature: 0.1,
            top_p: 0.9,
            stop_sequences: Vec::new(),
            system_message: None,
        }
    }
}

/// Response from an LLM generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResponse {
    /// Generated text.
    pub text: String,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// Model used for generation.
    pub model: String,
    /// Generation latency in milliseconds.
    pub latency_ms: u64,
    /// Provider-specific metadata.
    pub metadata: HashMap<String, String>,
}

/// A single token in a streaming response.
#[derive(Debug, Clone)]
pub enum StreamToken {
    /// A text token.
    Token(String),
    /// End of generation.
    Done {
        /// Total tokens generated.
        completion_tokens: usize,
    },
    /// An error occurred during generation.
    Error(String),
}

/// Trait for LLM providers.
///
/// Implementors provide text generation capabilities that the RAG pipeline
/// uses to produce answers from retrieved context.
pub trait LlmProvider: Send + Sync {
    /// Provider name (e.g., "openai", "anthropic", "ollama").
    fn name(&self) -> &str;

    /// Model identifier (e.g., "gpt-4", "claude-3-sonnet").
    fn model(&self) -> &str;

    /// Generate a complete response synchronously.
    fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse>;

    /// Generate a streaming response, sending tokens to the provided sender.
    /// Default implementation falls back to synchronous generation.
    fn generate_stream(
        &self,
        request: &GenerationRequest,
        sender: std::sync::mpsc::Sender<StreamToken>,
    ) -> Result<()> {
        let response = self.generate(request)?;
        for word in response.text.split_whitespace() {
            let token = format!("{word} ");
            if sender.send(StreamToken::Token(token)).is_err() {
                break;
            }
        }
        let _ = sender.send(StreamToken::Done {
            completion_tokens: response.completion_tokens,
        });
        Ok(())
    }

    /// Check if the provider is available/healthy.
    fn is_available(&self) -> bool {
        true
    }

    /// Estimate token count for a string (approximate).
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough estimate: ~4 chars per token
        text.len() / 4
    }
}

// ============================================================================
// Prompt Templates
// ============================================================================

/// Prompt template for RAG generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// System message template.
    pub system: String,
    /// User prompt template. Uses `{context}` and `{query}` placeholders.
    pub user: String,
    /// Include chunk citations in prompt.
    pub include_citations: bool,
    /// Maximum context length in characters.
    pub max_context_chars: usize,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self {
            system: "You are a helpful assistant. Answer the user's question based on the \
                      provided context. If the context doesn't contain enough information, \
                      say so. Always cite the source chunks when possible."
                .into(),
            user: "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:".into(),
            include_citations: true,
            max_context_chars: 12000,
        }
    }
}

impl PromptTemplate {
    /// Render the prompt with the given context and query.
    pub fn render(&self, context: &str, query: &str) -> String {
        let truncated_context = if context.len() > self.max_context_chars {
            &context[..self.max_context_chars]
        } else {
            context
        };
        self.user
            .replace("{context}", truncated_context)
            .replace("{query}", query)
    }

    /// Build context string from retrieved chunks.
    pub fn build_context(&self, chunks: &[RetrievedChunk]) -> String {
        let mut context = String::new();
        for (i, chunk) in chunks.iter().enumerate() {
            if self.include_citations {
                context.push_str(&format!("[{}] ", i + 1));
            }
            context.push_str(&chunk.chunk.text);
            context.push('\n');
        }
        context
    }
}

// ============================================================================
// LLM Configuration
// ============================================================================

/// Configuration for the RAG + LLM integration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// Prompt template.
    pub template: PromptTemplate,
    /// Maximum tokens for generation.
    pub max_tokens: usize,
    /// Temperature for generation.
    pub temperature: f32,
    /// Enable response caching.
    pub cache_responses: bool,
    /// Maximum cached responses.
    pub cache_size: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            template: PromptTemplate::default(),
            max_tokens: 1024,
            temperature: 0.1,
            cache_responses: true,
            cache_size: 100,
        }
    }
}

// ============================================================================
// RAG + LLM Response
// ============================================================================

/// Complete RAG + LLM response.
#[derive(Debug, Clone)]
pub struct RagLlmResponse {
    /// The generated answer.
    pub answer: String,
    /// Retrieved chunks used as context.
    pub chunks: Vec<RetrievedChunk>,
    /// Assembled context string.
    pub context: String,
    /// LLM generation metadata.
    pub generation: GenerationResponse,
    /// Total latency (retrieval + generation) in milliseconds.
    pub total_latency_ms: u64,
    /// Retrieval latency in milliseconds.
    pub retrieval_latency_ms: u64,
}

// ============================================================================
// RAG with LLM wrapper
// ============================================================================

/// End-to-end RAG pipeline with LLM generation.
///
/// Combines vector retrieval (via `RagPipeline`) with LLM generation
/// (via `LlmProvider`) into a single `ask()` method call.
pub struct RagWithLlm {
    pipeline: RagPipeline,
    llm: Box<dyn LlmProvider>,
    config: LlmConfig,
    response_cache: parking_lot::Mutex<lru::LruCache<String, RagLlmResponse>>,
}

impl RagWithLlm {
    /// Create a new RAG + LLM wrapper.
    pub fn new(pipeline: RagPipeline, llm: Box<dyn LlmProvider>, config: LlmConfig) -> Self {
        let cache_size = std::num::NonZeroUsize::new(config.cache_size.max(1))
            .expect("cache_size is at least 1");
        Self {
            pipeline,
            llm,
            config,
            response_cache: parking_lot::Mutex::new(lru::LruCache::new(cache_size)),
        }
    }

    /// Ask a question: retrieve context and generate an answer.
    pub fn ask(&self, query: &str, embedder: &dyn Embedder) -> Result<RagLlmResponse> {
        let start = Instant::now();

        // Check cache
        if self.config.cache_responses {
            let mut cache = self.response_cache.lock();
            if let Some(cached) = cache.get(query) {
                return Ok(cached.clone());
            }
        }

        // Step 1: Retrieve
        let retrieval_start = Instant::now();
        let rag_response = self.pipeline.query(query, embedder)?;
        let retrieval_latency_ms = retrieval_start.elapsed().as_millis() as u64;

        // Step 2: Build prompt
        let context = self.config.template.build_context(&rag_response.chunks);
        let prompt = self.config.template.render(&context, query);

        // Step 3: Generate
        let request = GenerationRequest {
            prompt,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            system_message: Some(self.config.template.system.clone()),
            ..Default::default()
        };

        let generation = self.llm.generate(&request)?;

        let response = RagLlmResponse {
            answer: generation.text.clone(),
            chunks: rag_response.chunks,
            context,
            generation,
            total_latency_ms: start.elapsed().as_millis() as u64,
            retrieval_latency_ms,
        };

        // Cache the response
        if self.config.cache_responses {
            let mut cache = self.response_cache.lock();
            cache.put(query.to_string(), response.clone());
        }

        Ok(response)
    }

    /// Ask with streaming response. Returns a receiver for tokens.
    pub fn ask_stream(
        &self,
        query: &str,
        embedder: &dyn Embedder,
    ) -> Result<(Vec<RetrievedChunk>, std::sync::mpsc::Receiver<StreamToken>)> {
        // Retrieve context
        let rag_response = self.pipeline.query(query, embedder)?;
        let context = self.config.template.build_context(&rag_response.chunks);
        let prompt = self.config.template.render(&context, query);

        let request = GenerationRequest {
            prompt,
            max_tokens: self.config.max_tokens,
            temperature: self.config.temperature,
            system_message: Some(self.config.template.system.clone()),
            ..Default::default()
        };

        let (tx, rx) = std::sync::mpsc::channel();
        let chunks = rag_response.chunks;

        // Generate in a streaming fashion
        self.llm.generate_stream(&request, tx)?;

        Ok((chunks, rx))
    }

    /// Get a reference to the underlying RAG pipeline.
    pub fn pipeline(&self) -> &RagPipeline {
        &self.pipeline
    }

    /// Get a mutable reference to the underlying RAG pipeline.
    pub fn pipeline_mut(&mut self) -> &mut RagPipeline {
        &mut self.pipeline
    }

    /// Convenience: ingest a document and set it up for RAG queries.
    /// Combines ingest + query in the smallest possible API.
    pub fn ingest(
        &mut self,
        doc_id: &str,
        text: &str,
        metadata: Option<serde_json::Value>,
        embedder: &dyn Embedder,
    ) -> Result<()> {
        self.pipeline.ingest_document(doc_id, text, metadata, embedder)?;
        Ok(())
    }
}

/// Create a minimal RAG+LLM setup in one call.
///
/// ```rust,ignore
/// let rag = quick_rag_with_llm(db, 384, Box::new(MockLlmProvider::new()))?;
/// rag.ingest("doc1", "Machine learning is...", None, &embedder)?;
/// let answer = rag.ask("What is ML?", &embedder)?;
/// ```
pub fn quick_rag_with_llm(
    db: Arc<crate::database::Database>,
    dimensions: usize,
    llm: Box<dyn LlmProvider>,
) -> Result<RagWithLlm> {
    let config = super::rag::RagConfig {
        dimensions,
        ..Default::default()
    };
    let pipeline = RagPipeline::new(db, config)?;
    Ok(RagWithLlm::new(pipeline, llm, LlmConfig::default()))
}

// ============================================================================
// Mock LLM Provider (for testing)
// ============================================================================

/// Mock LLM provider that generates deterministic responses for testing.
pub struct MockLlmProvider {
    model_name: String,
}

impl MockLlmProvider {
    /// Create a new mock provider.
    pub fn new() -> Self {
        Self {
            model_name: "mock-llm-v1".into(),
        }
    }

    /// Create with a custom model name.
    pub fn with_model(model_name: &str) -> Self {
        Self {
            model_name: model_name.into(),
        }
    }
}

impl Default for MockLlmProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LlmProvider for MockLlmProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn model(&self) -> &str {
        &self.model_name
    }

    fn generate(&self, request: &GenerationRequest) -> Result<GenerationResponse> {
        let prompt_tokens = self.estimate_tokens(&request.prompt);
        let answer = format!(
            "Based on the provided context, here is the answer to your question. \
             [Generated by {} with temperature={}]",
            self.model_name, request.temperature
        );
        let completion_tokens = self.estimate_tokens(&answer);

        Ok(GenerationResponse {
            text: answer,
            prompt_tokens,
            completion_tokens,
            model: self.model_name.clone(),
            latency_ms: 50, // simulated
            metadata: HashMap::new(),
        })
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for `RagWithLlm`.
pub struct RagWithLlmBuilder {
    config: LlmConfig,
    llm: Option<Box<dyn LlmProvider>>,
}

impl RagWithLlmBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            config: LlmConfig::default(),
            llm: None,
        }
    }

    /// Set the LLM provider.
    #[must_use]
    pub fn provider(mut self, llm: Box<dyn LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Set the prompt template.
    #[must_use]
    pub fn template(mut self, template: PromptTemplate) -> Self {
        self.config.template = template;
        self
    }

    /// Set max tokens.
    #[must_use]
    pub fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.config.max_tokens = max_tokens;
        self
    }

    /// Set temperature.
    #[must_use]
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature;
        self
    }

    /// Enable/disable response caching.
    #[must_use]
    pub fn cache_responses(mut self, enabled: bool) -> Self {
        self.config.cache_responses = enabled;
        self
    }

    /// Build the `RagWithLlm` instance.
    pub fn build(self, pipeline: RagPipeline) -> Result<RagWithLlm> {
        let llm = self.llm.ok_or_else(|| {
            NeedleError::InvalidConfig("LLM provider is required".into())
        })?;
        Ok(RagWithLlm::new(pipeline, llm, self.config))
    }
}

impl Default for RagWithLlmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use crate::ml::rag::{MockEmbedder, RagConfig};

    #[test]
    fn test_mock_llm_provider() {
        let llm = MockLlmProvider::new();
        let request = GenerationRequest {
            prompt: "What is machine learning?".into(),
            ..Default::default()
        };
        let response = llm.generate(&request).expect("generation should succeed");
        assert!(!response.text.is_empty());
        assert!(response.completion_tokens > 0);
        assert_eq!(response.model, "mock-llm-v1");
    }

    #[test]
    fn test_prompt_template_render() {
        let template = PromptTemplate::default();
        let rendered = template.render("Some context here", "What is AI?");
        assert!(rendered.contains("Some context here"));
        assert!(rendered.contains("What is AI?"));
    }

    #[test]
    fn test_prompt_template_build_context() {
        let template = PromptTemplate {
            include_citations: true,
            ..Default::default()
        };
        use crate::ml::rag::Chunk;
        let chunks = vec![
            RetrievedChunk {
                chunk: Chunk {
                    id: "c1".into(),
                    document_id: "d1".into(),
                    text: "First chunk.".into(),
                    start_pos: 0,
                    end_pos: 12,
                    chunk_index: 0,
                    total_chunks: 2,
                    parent_id: None,
                    children: vec![],
                    metadata: None,
                },
                score: 0.9,
                rerank_score: None,
                final_score: 0.9,
            },
            RetrievedChunk {
                chunk: Chunk {
                    id: "c2".into(),
                    document_id: "d1".into(),
                    text: "Second chunk.".into(),
                    start_pos: 13,
                    end_pos: 26,
                    chunk_index: 1,
                    total_chunks: 2,
                    parent_id: None,
                    children: vec![],
                    metadata: None,
                },
                score: 0.8,
                rerank_score: None,
                final_score: 0.8,
            },
        ];
        let context = template.build_context(&chunks);
        assert!(context.contains("[1] First chunk."));
        assert!(context.contains("[2] Second chunk."));
    }

    #[test]
    fn test_rag_with_llm_end_to_end() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).expect("pipeline");
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document(
                "doc1",
                "Machine learning is a subset of AI that enables systems to learn.",
                None,
                &embedder,
            )
            .expect("ingest");

        let llm = MockLlmProvider::new();
        let rag_llm = RagWithLlm::new(pipeline, Box::new(llm), LlmConfig::default());

        let response = rag_llm.ask("What is machine learning?", &embedder).expect("ask");
        assert!(!response.answer.is_empty());
        assert!(response.total_latency_ms >= response.retrieval_latency_ms);
    }

    #[test]
    fn test_rag_with_llm_caching() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).expect("pipeline");
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "Test content for caching.", None, &embedder)
            .expect("ingest");

        let llm = MockLlmProvider::new();
        let rag_llm = RagWithLlm::new(
            pipeline,
            Box::new(llm),
            LlmConfig {
                cache_responses: true,
                ..Default::default()
            },
        );

        let r1 = rag_llm.ask("test query", &embedder).expect("ask1");
        let r2 = rag_llm.ask("test query", &embedder).expect("ask2");
        assert_eq!(r1.answer, r2.answer);
    }

    #[test]
    fn test_rag_with_llm_stream() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).expect("pipeline");
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "Streaming test content.", None, &embedder)
            .expect("ingest");

        let llm = MockLlmProvider::new();
        let rag_llm = RagWithLlm::new(pipeline, Box::new(llm), LlmConfig::default());

        let (chunks, rx) = rag_llm
            .ask_stream("test", &embedder)
            .expect("ask_stream");

        let mut tokens = Vec::new();
        while let Ok(token) = rx.recv() {
            match token {
                StreamToken::Token(t) => tokens.push(t),
                StreamToken::Done { .. } => break,
                StreamToken::Error(e) => return Err(format!("Stream error: {e}").into()),
            }
        }
        assert!(!tokens.is_empty());
        assert!(!chunks.is_empty());

        Ok(())
    }

    #[test]
    fn test_builder_pattern() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(db, config).expect("pipeline");

        let rag_llm = RagWithLlmBuilder::new()
            .provider(Box::new(MockLlmProvider::new()))
            .max_tokens(512)
            .temperature(0.5)
            .cache_responses(false)
            .build(pipeline)
            .expect("build");

        assert_eq!(rag_llm.config.max_tokens, 512);
    }

    #[test]
    fn test_builder_requires_provider() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(db, config).expect("pipeline");

        let result = RagWithLlmBuilder::new().build(pipeline);
        assert!(result.is_err());
    }
}
