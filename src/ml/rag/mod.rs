//! RAG (Retrieval-Augmented Generation) Pipeline
//!
//! Complete pipeline for building RAG applications:
//! - Document chunking (semantic, sliding window, hierarchical)
//! - Multi-stage retrieval (dense + sparse + rerank)
//! - Context assembly and prompt building
//! - Citation tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::rag::{RagPipeline, ChunkingStrategy, RagConfig};
//!
//! let pipeline = RagPipeline::new(db, RagConfig::default());
//!
//! // Ingest documents with automatic chunking
//! pipeline.ingest_document("doc1", &text, ChunkingStrategy::Semantic)?;
//!
//! // Query with full RAG pipeline
//! let response = pipeline.query("What is machine learning?", 5)?;
//! println!("Context: {:?}", response.chunks);
//! println!("Citations: {:?}", response.citations);
//! ```

pub mod cache;
pub mod chunking;
pub mod embedder;
pub mod pipeline;

pub use cache::{RagCache, RagCacheStats};
pub use chunking::{DocumentFormat, DocumentLoader, LoadedDocument, RecursiveTextSplitter};
pub use embedder::{Embedder, MockEmbedder};
pub use pipeline::{quick_rag_pipeline, RagPipeline, RagPipelineBuilder};

use serde::{Deserialize, Serialize};

/// Chunking strategy for documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks with overlap
    FixedSize { chunk_size: usize, overlap: usize },
    /// Sliding window with token count
    SlidingWindow {
        window_size: usize,
        step_size: usize,
    },
    /// Semantic chunking based on sentence boundaries
    Semantic {
        max_chunk_size: usize,
        min_chunk_size: usize,
    },
    /// Hierarchical chunking (parent-child relationships)
    Hierarchical {
        levels: Vec<usize>, // chunk sizes for each level
    },
    /// Paragraph-based chunking
    Paragraph { max_paragraphs: usize },
    /// Recursive splitting: try separators in order (\n\n, \n, sentence, word)
    Recursive { chunk_size: usize, chunk_overlap: usize },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Semantic {
            max_chunk_size: 512,
            min_chunk_size: 100,
        }
    }
}

/// Context window optimization strategy
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ContextStrategy {
    /// Include all chunks (no optimization)
    None,
    /// Truncate context to max tokens
    Truncate,
    /// Prioritize chunks by score within budget
    #[default]
    ScorePriority,
    /// Balance coverage vs. relevance
    Balanced {
        /// Weight for diversity (0-1)
        diversity_weight: f32,
    },
    /// Compress context by removing redundancy
    Compress {
        /// Minimum similarity to consider redundant
        redundancy_threshold: f32,
    },
}

/// Configuration for RAG pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Collection name for storing chunks
    pub collection_name: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Default chunking strategy
    pub chunking: ChunkingStrategy,
    /// Number of chunks to retrieve
    pub top_k: usize,
    /// Enable reranking
    pub rerank: bool,
    /// Reranking top-k (rerank this many, return top_k)
    pub rerank_top_k: usize,
    /// Enable hybrid search (dense + sparse)
    pub hybrid_search: bool,
    /// Hybrid search alpha (0 = sparse only, 1 = dense only)
    pub hybrid_alpha: f32,
    /// Include parent chunks in context
    pub include_parents: bool,
    /// Deduplicate similar chunks
    pub deduplicate: bool,
    /// Deduplication threshold
    pub dedup_threshold: f32,
    /// Enable response caching
    pub cache_enabled: bool,
    /// Maximum cache entries
    pub cache_size: usize,
    /// Cache TTL in seconds (0 = no expiry)
    pub cache_ttl_seconds: u64,
    /// Maximum context tokens (approximate, based on chars/4)
    pub max_context_tokens: usize,
    /// Context optimization strategy
    pub context_strategy: ContextStrategy,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            collection_name: "rag_chunks".to_string(),
            dimensions: 384,
            chunking: ChunkingStrategy::default(),
            top_k: 5,
            rerank: true,
            rerank_top_k: 20,
            hybrid_search: true,
            hybrid_alpha: 0.7,
            include_parents: false,
            deduplicate: true,
            dedup_threshold: 0.95,
            cache_enabled: true,
            cache_size: 1000,
            cache_ttl_seconds: 300,   // 5 minutes default
            max_context_tokens: 4096, // ~16K chars default
            context_strategy: ContextStrategy::default(),
        }
    }
}

/// A chunk of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk ID
    pub id: String,
    /// Source document ID
    pub document_id: String,
    /// Chunk text content
    pub text: String,
    /// Start position in original document
    pub start_pos: usize,
    /// End position in original document
    pub end_pos: usize,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Total chunks in document
    pub total_chunks: usize,
    /// Parent chunk ID (for hierarchical)
    pub parent_id: Option<String>,
    /// Child chunk IDs (for hierarchical)
    pub children: Vec<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
}

/// Retrieved chunk with score
#[derive(Debug, Clone)]
pub struct RetrievedChunk {
    /// The chunk
    pub chunk: Chunk,
    /// Retrieval score (distance)
    pub score: f32,
    /// Rerank score (if reranking enabled)
    pub rerank_score: Option<f32>,
    /// Combined score
    pub final_score: f32,
}

/// Citation for a retrieved chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Document ID
    pub document_id: String,
    /// Chunk ID
    pub chunk_id: String,
    /// Text snippet
    pub snippet: String,
    /// Position in document
    pub position: (usize, usize),
    /// Relevance score
    pub score: f32,
}

/// RAG query response
#[derive(Debug, Clone)]
pub struct RagResponse {
    /// Retrieved chunks
    pub chunks: Vec<RetrievedChunk>,
    /// Assembled context
    pub context: String,
    /// Citations
    pub citations: Vec<Citation>,
    /// Query metadata
    pub metadata: RagQueryMetadata,
}

/// Query metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagQueryMetadata {
    /// Total chunks retrieved
    pub chunks_retrieved: usize,
    /// Chunks after deduplication
    pub chunks_after_dedup: usize,
    /// Retrieval latency in ms
    pub retrieval_latency_ms: u64,
    /// Rerank latency in ms
    pub rerank_latency_ms: Option<u64>,
    /// Total latency in ms
    pub total_latency_ms: u64,
}

/// Document with chunking info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Document ID
    pub id: String,
    /// Full text
    pub text: String,
    /// Chunk IDs
    pub chunk_ids: Vec<String>,
    /// Metadata
    pub metadata: Option<serde_json::Value>,
    /// Ingestion timestamp
    pub ingested_at: u64,
}

/// RAG pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub avg_chunks_per_doc: f64,
    pub total_queries: u64,
    pub avg_query_latency_ms: f64,
    pub cache_hit_rate: Option<f64>,
}

/// Options for batch document ingestion
#[derive(Debug, Clone)]
pub struct BatchIngestOptions {
    /// Chunking strategy override (uses pipeline default if None)
    pub chunking: Option<ChunkingStrategy>,
    /// Skip documents that already exist
    pub skip_existing: bool,
}

impl Default for BatchIngestOptions {
    fn default() -> Self {
        Self {
            chunking: None,
            skip_existing: true,
        }
    }
}

/// Result of a batch ingestion operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchIngestResult {
    pub ingested: usize,
    pub skipped: usize,
    pub failed: usize,
    pub errors: Vec<String>,
    pub elapsed_ms: u64,
}

/// Options for multi-query RAG
#[derive(Debug, Clone)]
pub struct MultiQueryOptions {
    /// Number of query expansions to generate
    pub num_expansions: usize,
    /// Merge strategy for results from multiple queries
    pub merge_strategy: MultiQueryMerge,
}

impl Default for MultiQueryOptions {
    fn default() -> Self {
        Self {
            num_expansions: 3,
            merge_strategy: MultiQueryMerge::RoundRobin,
        }
    }
}

/// Strategy for merging results from multiple queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultiQueryMerge {
    /// Interleave results in round-robin order
    RoundRobin,
    /// Reciprocal rank fusion across all result sets
    ReciprocalRankFusion { k: f32 },
    /// Take union and re-sort by best score
    UnionBestScore,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use std::sync::Arc;

    #[test]
    fn test_fixed_size_chunking() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let text = "This is a test document. It has multiple sentences. We want to chunk it.";
        let chunks = pipeline::RagPipeline::chunk_fixed_size(text, 20, 5);

        assert!(!chunks.is_empty());
        for (chunk, start, end) in &chunks {
            assert!(!chunk.is_empty());
            assert!(start < end);
        }
    }

    #[test]
    fn test_semantic_chunking() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let text = "First sentence here. Second sentence follows. Third one is longer and contains more content. Final sentence.";
        let chunks = pipeline.chunk_semantic(text, 50, 10);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_paragraph_chunking() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        let chunks = pipeline::RagPipeline::chunk_paragraphs(text, 2);

        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_ingest_and_query() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest document
        let doc = pipeline
            .ingest_document(
                "doc1",
                "Machine learning is a subset of artificial intelligence.",
                None,
                &embedder,
            )
            .unwrap();

        assert_eq!(doc.id, "doc1");
        assert!(!doc.chunk_ids.is_empty());

        // Query
        let response = pipeline
            .query("What is machine learning?", &embedder)
            .unwrap();

        assert!(!response.chunks.is_empty());
        assert!(!response.context.is_empty());
        assert!(!response.citations.is_empty());
    }

    #[test]
    fn test_text_similarity() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig::default()).unwrap();

        let sim = pipeline::RagPipeline::text_similarity("hello world", "hello world");
        assert!((sim - 1.0).abs() < 0.001);

        let sim = pipeline::RagPipeline::text_similarity("hello world", "goodbye world");
        assert!(sim > 0.0 && sim < 1.0);

        let sim = pipeline::RagPipeline::text_similarity("hello", "goodbye");
        assert!(sim < 0.5);
    }

    #[test]
    fn test_mock_embedder() {
        let embedder = MockEmbedder::new(128);

        let emb1 = embedder.embed("hello world").unwrap();
        let emb2 = embedder.embed("hello world").unwrap();
        let emb3 = embedder.embed("different text").unwrap();

        assert_eq!(emb1.len(), 128);
        assert_eq!(emb1, emb2); // Same text = same embedding
        assert_ne!(emb1, emb3); // Different text = different embedding
    }

    #[test]
    fn test_rag_cache_basic() {
        use cache::{CacheKey, CachedRagResponse};

        let cache = RagCache::new(10, 0); // No TTL

        let key = CacheKey::new("test query", None);
        let response = CachedRagResponse {
            chunks: vec![],
            context: "test context".to_string(),
            citations: vec![],
        };

        // Cache miss initially
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);

        // Put and get
        cache.put(key.clone(), response.clone());
        let cached = cache.get(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().context, "test context");
        assert_eq!(cache.stats().hits, 1);
    }

    #[test]
    fn test_rag_cache_ttl() {
        use cache::{CacheKey, CachedRagResponse};

        let cache = RagCache::new(10, 1); // 1 second TTL

        let key = CacheKey::new("test", None);
        let response = CachedRagResponse {
            chunks: vec![],
            context: "test".to_string(),
            citations: vec![],
        };

        cache.put(key.clone(), response);
        assert!(cache.get(&key).is_some());

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(1100));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_rag_cache_invalidation() {
        use cache::{CacheKey, CachedRagResponse};

        let cache = RagCache::new(10, 0);

        let key1 = CacheKey::new("query1", None);
        let key2 = CacheKey::new("query2", None);

        cache.put(
            key1.clone(),
            CachedRagResponse {
                chunks: vec![],
                context: "1".to_string(),
                citations: vec![],
            },
        );
        cache.put(
            key2.clone(),
            CachedRagResponse {
                chunks: vec![],
                context: "2".to_string(),
                citations: vec![],
            },
        );

        assert_eq!(cache.len(), 2);
        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_rag_cache_hit_rate() {
        use cache::{CacheKey, CachedRagResponse};

        let cache = RagCache::new(10, 0);

        let key = CacheKey::new("test", None);
        let response = CachedRagResponse {
            chunks: vec![],
            context: "test".to_string(),
            citations: vec![],
        };

        // Miss
        cache.get(&key);
        assert_eq!(cache.hit_rate(), 0.0);

        // Put and hit
        cache.put(key.clone(), response);
        cache.get(&key);
        assert_eq!(cache.hit_rate(), 0.5); // 1 hit, 1 miss

        cache.get(&key);
        assert!((cache.hit_rate() - 0.666).abs() < 0.01); // 2 hits, 1 miss
    }

    #[test]
    fn test_pipeline_caching() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            cache_enabled: true,
            cache_size: 100,
            cache_ttl_seconds: 300,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest a document
        pipeline
            .ingest_document(
                "doc1",
                "Machine learning and AI are related fields.",
                None,
                &embedder,
            )
            .unwrap();

        // First query - cache miss
        let response1 = pipeline.query("machine learning", &embedder).unwrap();
        let stats1 = pipeline.cache_stats().unwrap();
        assert_eq!(stats1.misses, 1);
        assert_eq!(stats1.hits, 0);

        // Second query - cache hit
        let response2 = pipeline.query("machine learning", &embedder).unwrap();
        let stats2 = pipeline.cache_stats().unwrap();
        assert_eq!(stats2.hits, 1);

        // Responses should have same content
        assert_eq!(response1.context, response2.context);

        // Different query - cache miss
        pipeline
            .query("artificial intelligence", &embedder)
            .unwrap();
        let stats3 = pipeline.cache_stats().unwrap();
        assert_eq!(stats3.misses, 2);
    }

    #[test]
    fn test_pipeline_cache_disabled() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            cache_enabled: false,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "Test document content.", None, &embedder)
            .unwrap();

        pipeline.query("test", &embedder).unwrap();

        // No cache stats when disabled
        assert!(pipeline.cache_stats().is_none());
        assert!(pipeline.cache_hit_rate().is_none());
    }

    #[test]
    fn test_cache_key_with_filter() {
        use cache::CacheKey;

        let key1 = CacheKey::new("query", None);
        let key2 = CacheKey::new("query", None);
        let key3 = CacheKey::new("different", None);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_context_strategy_none() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::None,
            max_context_tokens: 10, // Very small budget
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest long documents
        pipeline
            .ingest_document("doc1", "This is a long document with many words that should exceed the token budget when using strategy None.", None, &embedder)
            .unwrap();

        let result = pipeline.query("long document", &embedder).unwrap();

        // With ContextStrategy::None, all chunks are included regardless of budget
        assert!(!result.context.is_empty());
    }

    #[test]
    fn test_context_strategy_truncate() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::Truncate,
            max_context_tokens: 20, // ~80 chars
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "This is a sentence. This is another sentence. This is yet another sentence that makes the context very long.", None, &embedder)
            .unwrap();

        let result = pipeline.query("sentence", &embedder).unwrap();

        // Should be truncated
        if result.context.len() > 100 {
            assert!(result.context.contains("[Context truncated]"));
        }
    }

    #[test]
    fn test_context_strategy_score_priority() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::ScorePriority,
            max_context_tokens: 50, // ~200 chars
            top_k: 10,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest multiple documents
        for i in 0..5 {
            pipeline
                .ingest_document(
                    &format!("doc{}", i),
                    &format!("Document {} has some content about topic {}.", i, i),
                    None,
                    &embedder,
                )
                .unwrap();
        }

        let result = pipeline.query("document topic", &embedder).unwrap();

        // Should respect budget and prioritize by score
        assert!(!result.context.is_empty());
        // Context should be within budget (200 chars + some overhead)
        assert!(result.context.len() < 400);
    }

    #[test]
    fn test_context_strategy_balanced() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::Balanced {
                diversity_weight: 0.3,
            },
            max_context_tokens: 100,
            top_k: 10,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest documents with varying similarity
        pipeline
            .ingest_document(
                "doc1",
                "Machine learning is a subset of artificial intelligence.",
                None,
                &embedder,
            )
            .unwrap();
        pipeline
            .ingest_document(
                "doc2",
                "Machine learning uses algorithms to learn from data.",
                None,
                &embedder,
            )
            .unwrap();
        pipeline
            .ingest_document(
                "doc3",
                "Natural language processing handles text analysis.",
                None,
                &embedder,
            )
            .unwrap();

        let result = pipeline.query("machine learning", &embedder).unwrap();

        // Balanced strategy should include diverse content
        assert!(!result.context.is_empty());
    }

    #[test]
    fn test_context_strategy_compress() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::Compress {
                redundancy_threshold: 0.5, // Remove if 50% similar
            },
            max_context_tokens: 200,
            top_k: 10,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Ingest very similar documents
        pipeline
            .ingest_document(
                "doc1",
                "The quick brown fox jumps over the lazy dog.",
                None,
                &embedder,
            )
            .unwrap();
        pipeline
            .ingest_document(
                "doc2",
                "The quick brown fox leaps over the lazy dog.",
                None,
                &embedder,
            )
            .unwrap();
        pipeline
            .ingest_document(
                "doc3",
                "A fast red cat runs under the sleepy cat.",
                None,
                &embedder,
            )
            .unwrap();

        let result = pipeline.query("quick fox", &embedder).unwrap();

        // Compress strategy should remove redundant chunks
        assert!(!result.context.is_empty());
    }

    #[test]
    fn test_context_optimization_empty_chunks() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            context_strategy: ContextStrategy::ScorePriority,
            max_context_tokens: 100,
            ..Default::default()
        };
        let pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        // Query without any documents
        let result = pipeline.query("test query", &embedder).unwrap();

        // Should handle empty chunks gracefully
        assert!(result.context.is_empty());
        assert!(result.chunks.is_empty());
    }

    #[test]
    fn test_pipeline_builder_defaults() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipelineBuilder::new().dimensions(64).build(db).unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.total_documents, 0);
    }

    #[test]
    fn test_pipeline_builder_fluent() {
        let db = Arc::new(Database::in_memory());
        let mut pipeline = RagPipelineBuilder::new()
            .collection("my_rag")
            .dimensions(64)
            .chunker(ChunkingStrategy::FixedSize {
                chunk_size: 100,
                overlap: 20,
            })
            .context_strategy(ContextStrategy::ScorePriority)
            .top_k(5)
            .max_context_tokens(200)
            .with_cache(500, 1800)
            .build(db)
            .unwrap();

        let embedder = MockEmbedder::new(64);
        pipeline
            .ingest_document(
                "doc1",
                "Builder pattern makes it easy to configure.",
                None,
                &embedder,
            )
            .unwrap();

        let result = pipeline.query("builder", &embedder).unwrap();
        assert!(!result.chunks.is_empty());
    }

    #[test]
    fn test_pipeline_builder_into_config() {
        let config = RagPipelineBuilder::new()
            .collection("test")
            .dimensions(128)
            .top_k(20)
            .into_config();
        assert_eq!(config.collection_name, "test");
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.top_k, 20);
    }

    #[test]
    fn test_quick_rag_pipeline() {
        let db = Arc::new(Database::in_memory());
        let pipeline = quick_rag_pipeline(db, 64).unwrap();
        let stats = pipeline.stats();
        assert_eq!(stats.total_documents, 0);
    }

    #[test]
    fn test_batch_ingest() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        let docs = vec![
            ("doc1", "First document about machine learning.", None),
            ("doc2", "Second document about AI.", None),
            ("doc3", "Third document about databases.", None),
        ];

        let result = pipeline.batch_ingest(&docs, &embedder, &BatchIngestOptions::default());
        assert_eq!(result.ingested, 3);
        assert_eq!(result.skipped, 0);
        assert_eq!(result.failed, 0);
        assert_eq!(pipeline.stats().total_documents, 3);

        // Re-ingest with skip_existing should skip all
        let result2 = pipeline.batch_ingest(&docs, &embedder, &BatchIngestOptions::default());
        assert_eq!(result2.skipped, 3);
        assert_eq!(result2.ingested, 0);
    }

    #[test]
    fn test_multi_query() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "Machine learning uses algorithms.", None, &embedder)
            .unwrap();
        pipeline
            .ingest_document("doc2", "AI is transforming industries.", None, &embedder)
            .unwrap();

        let response = pipeline
            .multi_query(
                &["machine learning", "artificial intelligence"],
                &embedder,
                &MultiQueryOptions::default(),
            )
            .unwrap();

        assert!(!response.chunks.is_empty());
    }

    #[test]
    fn test_enhanced_stats() {
        let db = Arc::new(Database::in_memory());
        let config = RagConfig {
            dimensions: 64,
            ..Default::default()
        };
        let mut pipeline = RagPipeline::new(db, config).unwrap();
        let embedder = MockEmbedder::new(64);

        pipeline
            .ingest_document("doc1", "Test document content.", None, &embedder)
            .unwrap();
        pipeline.query("test", &embedder).unwrap();
        pipeline.query("another query", &embedder).unwrap();

        let stats = pipeline.stats();
        assert_eq!(stats.total_queries, 2);
        assert!(stats.avg_query_latency_ms >= 0.0);
    }

    #[test]
    fn test_document_loader_plaintext() {
        let doc = DocumentLoader::load_plaintext("d1", "Hello world.");
        assert_eq!(doc.format, DocumentFormat::PlainText);
        assert!(doc.metadata.is_none());
    }

    #[test]
    fn test_document_loader_markdown() {
        let md = "# Title\n\nBody paragraph.\n\n## Section\n\nMore text.";
        let doc = DocumentLoader::load_markdown("md1", md);
        assert_eq!(doc.format, DocumentFormat::Markdown);
        let meta = doc.metadata.as_ref().unwrap();
        assert_eq!(meta["title"], "Title");
        assert_eq!(meta["headings"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_document_loader_json() {
        let json = r#"{"title":"T","body":"B","n":42}"#;
        let doc = DocumentLoader::load_json("j1", json, &["title", "body"]).unwrap();
        assert!(doc.text.contains("T"));
        assert!(doc.text.contains("B"));
    }

    #[test]
    fn test_recursive_splitter() {
        let splitter = RecursiveTextSplitter::new(40, 5);
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph that is a bit longer.";
        let chunks = splitter.split(text);
        assert!(!chunks.is_empty());
        for (chunk, start, end) in &chunks {
            assert!(!chunk.trim().is_empty());
            assert!(start <= end);
        }
    }

    #[test]
    fn test_recursive_chunking_strategy() {
        let db = Arc::new(Database::in_memory());
        let pipeline = RagPipeline::new(db, RagConfig { dimensions: 64, ..Default::default() }).unwrap();
        let chunks = pipeline.chunk_text("a. b. c. d. e.", &ChunkingStrategy::Recursive { chunk_size: 8, chunk_overlap: 2 });
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_ingest_loaded() {
        let db = Arc::new(Database::in_memory());
        let mut pipeline = RagPipeline::new(db, RagConfig { dimensions: 64, ..Default::default() }).unwrap();
        let embedder = MockEmbedder::new(64);
        let doc = DocumentLoader::load_plaintext("ld1", "Some text for loading.");
        let result = pipeline.ingest_loaded(&doc, &embedder).unwrap();
        assert_eq!(result.id, "ld1");
    }

    #[test]
    fn test_builder_extensions() {
        let db = Arc::new(Database::in_memory());
        let p = RagPipelineBuilder::new()
            .dimensions(64)
            .hybrid_search(0.8)
            .deduplicate(0.9)
            .with_cache(100, 60)
            .build(db)
            .unwrap();
        assert!(p.cache_stats().is_some());
    }
}
