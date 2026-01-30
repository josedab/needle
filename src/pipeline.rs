//! CDC Pipeline Configuration and Execution
//!
//! Provides a declarative pipeline configuration for streaming database changes
//! into Needle vector collections. Supports Postgres, Kafka, and MongoDB sources
//! with automatic embedding generation.
//!
//! # Configuration
//!
//! Pipelines are defined in JSON or YAML format (see `examples/pipeline.yml`).
//! The pipeline executor reads the config and coordinates:
//! 1. Source connection (Postgres logical replication, Kafka consumer, etc.)
//! 2. Text extraction and transformation
//! 3. Embedding generation via configured provider
//! 4. Batch insertion into Needle collection
//!
//! # Example
//!
//! ```rust
//! use needle::pipeline::{PipelineConfig, SourceConfig, DestinationConfig, EmbeddingConfig};
//!
//! let config = PipelineConfig {
//!     name: "my-pipeline".to_string(),
//!     source: SourceConfig::Postgres {
//!         host: "localhost".to_string(),
//!         port: 5432,
//!         database: "myapp".to_string(),
//!         tables: vec!["documents".to_string()],
//!     },
//!     destination: DestinationConfig {
//!         database: "vectors.needle".to_string(),
//!         collection: "documents".to_string(),
//!         dimensions: 384,
//!     },
//!     embedding: EmbeddingConfig {
//!         provider: "openai".to_string(),
//!         model: "text-embedding-3-small".to_string(),
//!     },
//!     batch_size: 100,
//!     flush_interval_secs: 5,
//! };
//!
//! assert_eq!(config.name, "my-pipeline");
//! ```

use serde::{Deserialize, Serialize};
use tracing::warn;

/// Pipeline configuration for CDC-to-Needle sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name
    pub name: String,
    /// Optional description
    #[serde(default)]
    pub description: String,
    /// Source configuration
    pub source: SourceConfig,
    /// Destination Needle collection
    pub destination: DestinationConfig,
    /// Embedding provider configuration
    pub embedding: EmbeddingConfig,
    /// Batch size for inserts
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Flush interval in seconds
    #[serde(default = "default_flush_interval")]
    pub flush_interval_secs: u64,
    /// Max concurrent embedding requests
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    /// Max retries for failed operations
    #[serde(default = "default_retries")]
    pub max_retries: usize,
    /// Optional transform configuration
    #[serde(default)]
    pub transform: TransformConfig,
}

fn default_batch_size() -> usize { 100 }
fn default_flush_interval() -> u64 { 5 }
fn default_concurrency() -> usize { 4 }
fn default_retries() -> usize { 3 }

/// Source configuration for the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SourceConfig {
    /// PostgreSQL logical replication
    #[serde(rename = "postgres")]
    Postgres {
        host: String,
        #[serde(default = "default_pg_port")]
        port: u16,
        database: String,
        #[serde(default)]
        user: String,
        #[serde(default)]
        password: String,
        /// Tables to watch
        tables: Vec<String>,
    },
    /// Kafka consumer
    #[serde(rename = "kafka")]
    Kafka {
        brokers: Vec<String>,
        topic: String,
        #[serde(default = "default_kafka_group")]
        group_id: String,
        #[serde(default = "default_kafka_format")]
        format: String,
    },
    /// MongoDB change streams
    #[serde(rename = "mongodb")]
    MongoDB {
        connection_string: String,
        database: String,
        collection: String,
    },
}

fn default_pg_port() -> u16 { 5432 }
fn default_kafka_group() -> String { "needle-cdc".to_string() }
fn default_kafka_format() -> String { "json".to_string() }

/// Destination Needle collection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DestinationConfig {
    /// Path to .needle database file
    pub database: String,
    /// Collection name
    pub collection: String,
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function
    #[serde(default = "default_distance")]
    pub distance: String,
}

fn default_distance() -> String { "cosine".to_string() }

/// Embedding provider configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider name: openai, cohere, ollama
    pub provider: String,
    /// Model name
    pub model: String,
    /// API key (can use ${ENV_VAR} syntax)
    #[serde(default)]
    pub api_key: String,
    /// Base URL (for Ollama or custom endpoints)
    #[serde(default)]
    pub base_url: String,
}

/// Transform configuration for pre-processing text before embedding.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformConfig {
    /// Skip rows where text content is empty
    #[serde(default)]
    pub skip_empty: bool,
    /// Maximum text length before truncation
    #[serde(default = "default_max_text")]
    pub max_text_length: usize,
    /// Template for combining columns (e.g., "title: {title}\ncontent: {content}")
    #[serde(default)]
    pub prefix_template: String,
}

fn default_max_text() -> usize { 8192 }

/// A single CDC event to be processed by the pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcEvent {
    /// Unique ID for the record
    pub id: String,
    /// Text content to embed
    pub text: String,
    /// Optional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
    /// Operation type
    #[serde(default)]
    pub operation: CdcOperation,
}

/// CDC operation type.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum CdcOperation {
    #[default]
    Insert,
    Update,
    Delete,
}

/// Pipeline execution statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineStats {
    pub events_received: u64,
    pub events_processed: u64,
    pub events_failed: u64,
    pub batches_flushed: u64,
    pub embeddings_generated: u64,
}

/// Validates a pipeline configuration for completeness.
pub fn validate_config(config: &PipelineConfig) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if config.name.is_empty() {
        errors.push("Pipeline name is required".to_string());
    }
    if config.destination.dimensions == 0 {
        errors.push("Destination dimensions must be > 0".to_string());
    }
    if config.embedding.provider.is_empty() {
        errors.push("Embedding provider is required".to_string());
    }
    if config.embedding.model.is_empty() {
        errors.push("Embedding model is required".to_string());
    }
    if config.batch_size == 0 {
        errors.push("Batch size must be > 0".to_string());
    }

    match &config.source {
        SourceConfig::Postgres { tables, .. } if tables.is_empty() => {
            errors.push("At least one Postgres table is required".to_string());
        }
        SourceConfig::Kafka { brokers, .. } if brokers.is_empty() => {
            errors.push("At least one Kafka broker is required".to_string());
        }
        _ => {}
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

/// Parse a pipeline configuration from JSON.
pub fn parse_config(json_str: &str) -> Result<PipelineConfig, String> {
    serde_json::from_str(json_str).map_err(|e| format!("Invalid pipeline config: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_postgres_config() {
        let config_json = json!({
            "name": "test-pipeline",
            "source": {
                "type": "postgres",
                "host": "localhost",
                "port": 5432,
                "database": "myapp",
                "tables": ["documents"]
            },
            "destination": {
                "database": "vectors.needle",
                "collection": "docs",
                "dimensions": 384
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small"
            }
        });

        let config: PipelineConfig = serde_json::from_value(config_json).unwrap();
        assert_eq!(config.name, "test-pipeline");
        assert_eq!(config.destination.dimensions, 384);
        assert_eq!(config.batch_size, 100); // default
        assert!(matches!(config.source, SourceConfig::Postgres { .. }));
    }

    #[test]
    fn test_parse_kafka_config() {
        let config_json = json!({
            "name": "kafka-pipe",
            "source": {
                "type": "kafka",
                "brokers": ["localhost:9092"],
                "topic": "events"
            },
            "destination": {
                "database": "data.needle",
                "collection": "events",
                "dimensions": 768
            },
            "embedding": {
                "provider": "ollama",
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434"
            },
            "batch_size": 500
        });

        let config: PipelineConfig = serde_json::from_value(config_json).unwrap();
        assert_eq!(config.batch_size, 500);
        assert!(matches!(config.source, SourceConfig::Kafka { .. }));
    }

    #[test]
    fn test_validate_config() {
        let valid = PipelineConfig {
            name: "test".to_string(),
            description: String::new(),
            source: SourceConfig::Postgres {
                host: "localhost".to_string(),
                port: 5432,
                database: "db".to_string(),
                user: String::new(),
                password: String::new(),
                tables: vec!["t1".to_string()],
            },
            destination: DestinationConfig {
                database: "v.needle".to_string(),
                collection: "c".to_string(),
                dimensions: 384,
                distance: "cosine".to_string(),
            },
            embedding: EmbeddingConfig {
                provider: "openai".to_string(),
                model: "text-embedding-3-small".to_string(),
                api_key: String::new(),
                base_url: String::new(),
            },
            batch_size: 100,
            flush_interval_secs: 5,
            concurrency: 4,
            max_retries: 3,
            transform: TransformConfig::default(),
        };

        assert!(validate_config(&valid).is_ok());
    }

    #[test]
    fn test_validate_config_errors() {
        let invalid = PipelineConfig {
            name: String::new(),
            description: String::new(),
            source: SourceConfig::Postgres {
                host: "localhost".to_string(),
                port: 5432,
                database: "db".to_string(),
                user: String::new(),
                password: String::new(),
                tables: vec![],
            },
            destination: DestinationConfig {
                database: "v.needle".to_string(),
                collection: "c".to_string(),
                dimensions: 0,
                distance: "cosine".to_string(),
            },
            embedding: EmbeddingConfig {
                provider: String::new(),
                model: String::new(),
                api_key: String::new(),
                base_url: String::new(),
            },
            batch_size: 100,
            flush_interval_secs: 5,
            concurrency: 4,
            max_retries: 3,
            transform: TransformConfig::default(),
        };

        let errors = validate_config(&invalid).unwrap_err();
        assert!(errors.len() >= 4);
    }

    #[test]
    fn test_cdc_event() {
        let event = CdcEvent {
            id: "doc-1".to_string(),
            text: "Hello world".to_string(),
            metadata: json!({"source": "test"}),
            operation: CdcOperation::Insert,
        };
        assert_eq!(event.operation, CdcOperation::Insert);
    }

    #[test]
    fn test_parse_config_string() {
        let json_str = r#"{"name":"p","source":{"type":"kafka","brokers":["b"],"topic":"t"},"destination":{"database":"d","collection":"c","dimensions":128},"embedding":{"provider":"ollama","model":"m"}}"#;
        let config = parse_config(json_str).unwrap();
        assert_eq!(config.name, "p");
    }
}

// ── Declarative Ingestion Pipelines ─────────────────────────────────────────
// YAML/JSON-defined ingestion pipelines: source → chunker → embedder → indexer

/// Declarative ingestion pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionPipeline {
    /// Pipeline name.
    pub name: String,
    /// Pipeline description.
    #[serde(default)]
    pub description: String,
    /// Preset name (overrides individual stages).
    #[serde(default)]
    pub preset: Option<PipelinePreset>,
    /// Source stage configuration.
    pub source: IngestionSource,
    /// Chunker stage configuration.
    #[serde(default)]
    pub chunker: ChunkerConfig,
    /// Optional metadata extraction.
    #[serde(default)]
    pub metadata_extractors: Vec<MetadataExtractor>,
    /// Optional deduplication.
    #[serde(default)]
    pub dedup: Option<DedupConfig>,
    /// Destination collection.
    pub destination: DestinationConfig,
    /// Embedding configuration.
    pub embedding: EmbeddingConfig,
    /// Checkpoint configuration for error recovery.
    #[serde(default)]
    pub checkpoint: CheckpointConfig,
}

/// Opinionated pipeline presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelinePreset {
    /// Optimized for RAG: recursive character splitter, 512 token chunks, 50 token overlap.
    #[serde(rename = "rag-default")]
    RagDefault,
    /// Optimized for semantic search: sentence splitter, larger chunks, less overlap.
    #[serde(rename = "semantic-search")]
    SemanticSearch,
    /// Optimized for recommendation: metadata-heavy, small chunks.
    #[serde(rename = "recommendation")]
    Recommendation,
}

impl PipelinePreset {
    /// Get the chunker config for this preset.
    pub fn chunker_config(&self) -> ChunkerConfig {
        match self {
            PipelinePreset::RagDefault => ChunkerConfig {
                strategy: ChunkStrategy::RecursiveCharacter,
                chunk_size: 512,
                chunk_overlap: 50,
                separators: vec!["\n\n".into(), "\n".into(), ". ".into(), " ".into()],
            },
            PipelinePreset::SemanticSearch => ChunkerConfig {
                strategy: ChunkStrategy::Sentence,
                chunk_size: 1024,
                chunk_overlap: 100,
                separators: vec![". ".into(), "! ".into(), "? ".into()],
            },
            PipelinePreset::Recommendation => ChunkerConfig {
                strategy: ChunkStrategy::RecursiveCharacter,
                chunk_size: 256,
                chunk_overlap: 25,
                separators: vec!["\n".into(), ". ".into()],
            },
        }
    }
}

/// Ingestion source (file, directory, URL, or inline).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IngestionSource {
    /// Single file.
    #[serde(rename = "file")]
    File { path: String },
    /// Directory of files (recursive).
    #[serde(rename = "directory")]
    Directory {
        path: String,
        #[serde(default)]
        glob: Option<String>,
    },
    /// Inline text content.
    #[serde(rename = "inline")]
    Inline { documents: Vec<InlineDocument> },
}

/// An inline document for pipeline input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineDocument {
    /// Document ID.
    pub id: String,
    /// Document text content.
    pub text: String,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Text chunking strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkStrategy {
    /// Split by sentences.
    #[serde(rename = "sentence")]
    Sentence,
    /// Recursive character splitting (LangChain-style).
    #[serde(rename = "recursive-character")]
    RecursiveCharacter,
    /// Fixed-size character windows.
    #[serde(rename = "fixed-size")]
    FixedSize,
}

impl Default for ChunkStrategy {
    fn default() -> Self {
        Self::RecursiveCharacter
    }
}

/// Chunker stage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Chunking strategy.
    #[serde(default)]
    pub strategy: ChunkStrategy,
    /// Target chunk size in characters.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Overlap between chunks in characters.
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    /// Separator strings for splitting.
    #[serde(default = "default_separators")]
    pub separators: Vec<String>,
}

fn default_chunk_size() -> usize { 512 }
fn default_chunk_overlap() -> usize { 50 }
fn default_separators() -> Vec<String> {
    vec!["\n\n".into(), "\n".into(), ". ".into(), " ".into()]
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::default(),
            chunk_size: default_chunk_size(),
            chunk_overlap: default_chunk_overlap(),
            separators: default_separators(),
        }
    }
}

/// Metadata extractor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MetadataExtractor {
    /// Regex-based extraction.
    #[serde(rename = "regex")]
    Regex {
        /// Field name.
        field: String,
        /// Regex pattern with capture groups.
        pattern: String,
    },
    /// Static metadata applied to all chunks.
    #[serde(rename = "static")]
    Static {
        /// Key-value pairs to add.
        fields: std::collections::HashMap<String, serde_json::Value>,
    },
}

/// Deduplication configuration using MinHash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DedupConfig {
    /// Similarity threshold for dedup (0.0-1.0).
    #[serde(default = "default_dedup_threshold")]
    pub threshold: f64,
    /// Number of hash functions.
    #[serde(default = "default_num_hashes")]
    pub num_hashes: usize,
}

fn default_dedup_threshold() -> f64 { 0.9 }
fn default_num_hashes() -> usize { 128 }

/// Checkpoint configuration for error recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Enable checkpointing.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Checkpoint interval (every N documents).
    #[serde(default = "default_checkpoint_interval")]
    pub interval: usize,
    /// Checkpoint file path.
    #[serde(default)]
    pub path: Option<String>,
}

fn default_true() -> bool { true }
fn default_checkpoint_interval() -> usize { 100 }

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 100,
            path: None,
        }
    }
}

// ── Pipeline Execution ──────────────────────────────────────────────────────

/// A text chunk produced by the chunker stage.
#[derive(Debug, Clone)]
pub struct TextChunk {
    /// Source document ID.
    pub doc_id: String,
    /// Chunk index within the document.
    pub chunk_index: usize,
    /// Chunk text content.
    pub text: String,
    /// Extracted metadata.
    pub metadata: serde_json::Value,
}

/// Progress report from pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineProgress {
    /// Total documents to process.
    pub total_documents: usize,
    /// Documents processed so far.
    pub documents_processed: usize,
    /// Chunks generated.
    pub chunks_generated: usize,
    /// Last checkpoint document index.
    pub last_checkpoint: usize,
    /// Errors encountered.
    pub errors: Vec<String>,
}

/// Chunk text using the configured strategy.
pub fn chunk_text(text: &str, config: &ChunkerConfig) -> Vec<String> {
    match config.strategy {
        ChunkStrategy::Sentence => chunk_by_sentence(text, config.chunk_size, config.chunk_overlap),
        ChunkStrategy::RecursiveCharacter => {
            recursive_character_split(text, &config.separators, config.chunk_size, config.chunk_overlap)
        }
        ChunkStrategy::FixedSize => {
            fixed_size_chunk(text, config.chunk_size, config.chunk_overlap)
        }
    }
}

fn chunk_by_sentence(text: &str, max_size: usize, overlap: usize) -> Vec<String> {
    let sentences: Vec<&str> = text
        .split_inclusive(|c: char| c == '.' || c == '!' || c == '?')
        .collect();

    if sentences.is_empty() {
        return if text.is_empty() { vec![] } else { vec![text.to_string()] };
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut overlap_buffer = String::new();

    for sentence in sentences {
        if !current.is_empty() && current.len() + sentence.len() > max_size {
            chunks.push(current.trim().to_string());
            // Keep overlap from end of current chunk
            if overlap > 0 && current.len() > overlap {
                overlap_buffer = current[current.len() - overlap..].to_string();
            }
            current = overlap_buffer.clone();
            overlap_buffer.clear();
        }
        current.push_str(sentence);
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

fn recursive_character_split(
    text: &str,
    separators: &[String],
    max_size: usize,
    overlap: usize,
) -> Vec<String> {
    if text.len() <= max_size {
        return vec![text.to_string()];
    }

    // Try each separator in order
    for sep in separators {
        let parts: Vec<&str> = text.split(sep.as_str()).collect();
        if parts.len() > 1 {
            let mut chunks = Vec::new();
            let mut current = String::new();

            for part in parts {
                if !current.is_empty() && current.len() + sep.len() + part.len() > max_size {
                    chunks.push(current.trim().to_string());
                    // Overlap
                    if overlap > 0 && current.len() > overlap {
                        current = current[current.len() - overlap..].to_string();
                    } else {
                        current.clear();
                    }
                }
                if !current.is_empty() {
                    current.push_str(sep);
                }
                current.push_str(part);
            }

            if !current.trim().is_empty() {
                chunks.push(current.trim().to_string());
            }

            return chunks;
        }
    }

    // Fallback to fixed-size
    fixed_size_chunk(text, max_size, overlap)
}

fn fixed_size_chunk(text: &str, size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || size == 0 {
        return vec![];
    }

    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let step = if size > overlap { size - overlap } else { 1 };
    let mut start = 0;

    while start < chars.len() {
        let end = (start + size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        if !chunk.trim().is_empty() {
            chunks.push(chunk.trim().to_string());
        }
        start += step;
    }

    chunks
}

/// Parse an ingestion pipeline configuration from a JSON string.
pub fn parse_ingestion_pipeline(json_str: &str) -> Result<IngestionPipeline, String> {
    let mut pipeline: IngestionPipeline =
        serde_json::from_str(json_str).map_err(|e| format!("Invalid pipeline config: {e}"))?;

    // Apply preset if specified
    if let Some(preset) = pipeline.preset {
        pipeline.chunker = preset.chunker_config();
    }

    Ok(pipeline)
}

/// Validate an ingestion pipeline configuration.
pub fn validate_ingestion_pipeline(pipeline: &IngestionPipeline) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if pipeline.name.is_empty() {
        errors.push("Pipeline name is required".into());
    }
    if pipeline.destination.dimensions == 0 {
        errors.push("Destination dimensions must be > 0".into());
    }
    if pipeline.embedding.provider.is_empty() {
        errors.push("Embedding provider is required".into());
    }
    if pipeline.chunker.chunk_size == 0 {
        errors.push("Chunk size must be > 0".into());
    }
    if pipeline.chunker.chunk_overlap >= pipeline.chunker.chunk_size {
        errors.push("Chunk overlap must be less than chunk size".into());
    }

    if errors.is_empty() { Ok(()) } else { Err(errors) }
}

// ── MinHash Deduplication ────────────────────────────────────────────────────

/// Simple MinHash fingerprint for near-duplicate detection.
#[derive(Debug, Clone)]
pub struct MinHashFingerprint {
    /// MinHash signature values.
    pub signature: Vec<u64>,
}

impl MinHashFingerprint {
    /// Compute a MinHash fingerprint for a text string using n-gram shingling.
    pub fn compute(text: &str, num_hashes: usize) -> Self {
        let shingles = Self::shingle(text, 3);
        let mut signature = vec![u64::MAX; num_hashes];

        for shingle in &shingles {
            let base_hash = Self::hash_bytes(shingle.as_bytes());
            for (i, sig) in signature.iter_mut().enumerate() {
                let h = base_hash.wrapping_mul(6364136223846793005u64.wrapping_add(i as u64 * 2 + 1));
                *sig = (*sig).min(h);
            }
        }

        Self { signature }
    }

    /// Estimate Jaccard similarity between two fingerprints.
    pub fn similarity(&self, other: &MinHashFingerprint) -> f64 {
        if self.signature.len() != other.signature.len() || self.signature.is_empty() {
            return 0.0;
        }
        let matches = self
            .signature
            .iter()
            .zip(other.signature.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.signature.len() as f64
    }

    fn shingle(text: &str, n: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < n {
            return vec![text.to_string()];
        }
        (0..=chars.len() - n)
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }

    fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// Deduplicate a list of text chunks, returning indices to keep.
pub fn deduplicate_chunks(chunks: &[String], config: &DedupConfig) -> Vec<usize> {
    if chunks.is_empty() {
        return vec![];
    }

    let fingerprints: Vec<MinHashFingerprint> = chunks
        .iter()
        .map(|c| MinHashFingerprint::compute(c, config.num_hashes))
        .collect();

    let mut keep = vec![true; chunks.len()];
    for i in 0..chunks.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..chunks.len() {
            if !keep[j] {
                continue;
            }
            if fingerprints[i].similarity(&fingerprints[j]) >= config.threshold {
                keep[j] = false;
            }
        }
    }

    keep.iter()
        .enumerate()
        .filter_map(|(i, &k)| if k { Some(i) } else { None })
        .collect()
}

// ── Pipeline Checkpoint State ────────────────────────────────────────────────

/// Serializable checkpoint state for pipeline error recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCheckpoint {
    /// Pipeline name.
    pub pipeline_name: String,
    /// Last successfully processed document index.
    pub last_processed_index: usize,
    /// Total documents in the source.
    pub total_documents: usize,
    /// Chunks generated so far.
    pub chunks_generated: usize,
    /// Errors encountered.
    pub errors: Vec<String>,
    /// Timestamp of checkpoint.
    pub timestamp: u64,
}

impl PipelineCheckpoint {
    /// Create a new checkpoint.
    pub fn new(pipeline_name: &str, total_documents: usize) -> Self {
        Self {
            pipeline_name: pipeline_name.into(),
            last_processed_index: 0,
            total_documents,
            chunks_generated: 0,
            errors: Vec::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Update with progress.
    pub fn update(&mut self, processed: usize, chunks: usize) {
        self.last_processed_index = processed;
        self.chunks_generated = chunks;
        self.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    /// Save checkpoint to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> std::result::Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load checkpoint from a JSON file.
    pub fn load(path: &std::path::Path) -> std::result::Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&json).map_err(|e| e.to_string())
    }

    /// Progress as a percentage.
    pub fn progress_pct(&self) -> f64 {
        if self.total_documents == 0 {
            return 0.0;
        }
        (self.last_processed_index as f64 / self.total_documents as f64) * 100.0
    }
}

// ── Pipeline Executor ────────────────────────────────────────────────────────

/// Execute an ingestion pipeline on inline documents, returning text chunks.
pub fn execute_ingestion_pipeline(
    pipeline: &IngestionPipeline,
) -> std::result::Result<Vec<TextChunk>, Vec<String>> {
    validate_ingestion_pipeline(pipeline)?;

    let documents = match &pipeline.source {
        IngestionSource::Inline { documents } => documents.clone(),
        IngestionSource::File { path } => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| vec![format!("Failed to read file '{}': {}", path, e)])?;
            vec![InlineDocument {
                id: path.clone(),
                text,
                metadata: serde_json::Value::Null,
            }]
        }
        IngestionSource::Directory { path, glob } => {
            let dir = std::path::Path::new(path);
            if !dir.is_dir() {
                return Err(vec![format!("Directory not found: {}", path)]);
            }
            let mut docs = Vec::new();
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        let name = entry_path.file_name().unwrap_or_default().to_string_lossy().to_string();
                        if let Some(pattern) = glob {
                            if !name.ends_with(pattern.trim_start_matches('*')) {
                                continue;
                            }
                        }
                        if let Ok(text) = std::fs::read_to_string(&entry_path) {
                            docs.push(InlineDocument {
                                id: name,
                                text,
                                metadata: serde_json::Value::Null,
                            });
                        }
                    }
                }
            }
            docs
        }
    };

    let mut all_chunks: Vec<TextChunk> = Vec::new();
    let mut checkpoint = PipelineCheckpoint::new(&pipeline.name, documents.len());

    for (doc_idx, doc) in documents.iter().enumerate() {
        let raw_chunks = chunk_text(&doc.text, &pipeline.chunker);

        let chunk_texts: Vec<String> = if let Some(dedup) = &pipeline.dedup {
            let keep_indices = deduplicate_chunks(&raw_chunks, dedup);
            keep_indices.into_iter().map(|i| raw_chunks[i].clone()).collect()
        } else {
            raw_chunks
        };

        for (chunk_idx, text) in chunk_texts.iter().enumerate() {
            let mut metadata = doc.metadata.clone();

            for extractor in &pipeline.metadata_extractors {
                match extractor {
                    MetadataExtractor::Static { fields } => {
                        if let Some(obj) = metadata.as_object_mut() {
                            for (k, v) in fields {
                                obj.insert(k.clone(), v.clone());
                            }
                        } else {
                            let mut obj = serde_json::Map::new();
                            for (k, v) in fields {
                                obj.insert(k.clone(), v.clone());
                            }
                            metadata = serde_json::Value::Object(obj);
                        }
                    }
                    MetadataExtractor::Regex { .. } => {
                        // Regex extraction requires a regex dependency; skip in base build
                    }
                }
            }

            all_chunks.push(TextChunk {
                doc_id: doc.id.clone(),
                chunk_index: chunk_idx,
                text: text.clone(),
                metadata,
            });
        }

        checkpoint.update(doc_idx + 1, all_chunks.len());

        if pipeline.checkpoint.enabled
            && (doc_idx + 1) % pipeline.checkpoint.interval == 0
        {
            if let Some(path) = &pipeline.checkpoint.path {
                if let Err(e) = checkpoint.save(std::path::Path::new(path)) {
                    warn!("Failed to save pipeline checkpoint to {}: {}", path, e);
                }
            }
        }
    }

    Ok(all_chunks)
}

#[cfg(test)]
mod ingestion_tests {
    use super::*;

    #[test]
    fn test_sentence_chunking() {
        let text = "Hello world. This is a test. Another sentence. More text here. Final words.";
        let config = ChunkerConfig {
            strategy: ChunkStrategy::Sentence,
            chunk_size: 30,
            chunk_overlap: 5,
            ..Default::default()
        };
        let chunks = chunk_text(text, &config);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
    }

    #[test]
    fn test_recursive_chunking() {
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four.";
        let config = ChunkerConfig {
            strategy: ChunkStrategy::RecursiveCharacter,
            chunk_size: 30,
            chunk_overlap: 5,
            separators: vec!["\n\n".into(), "\n".into(), ". ".into()],
        };
        let chunks = chunk_text(text, &config);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_fixed_size_chunking() {
        let text = "a".repeat(100);
        let config = ChunkerConfig {
            strategy: ChunkStrategy::FixedSize,
            chunk_size: 30,
            chunk_overlap: 5,
            ..Default::default()
        };
        let chunks = chunk_text(text.as_str(), &config);
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_preset_rag_default() {
        let config = PipelinePreset::RagDefault.chunker_config();
        assert_eq!(config.chunk_size, 512);
        assert_eq!(config.chunk_overlap, 50);
    }

    #[test]
    fn test_preset_semantic_search() {
        let config = PipelinePreset::SemanticSearch.chunker_config();
        assert_eq!(config.chunk_size, 1024);
    }

    #[test]
    fn test_parse_ingestion_pipeline() {
        let json = serde_json::json!({
            "name": "my-rag-pipeline",
            "preset": "rag-default",
            "source": {
                "type": "inline",
                "documents": [{"id": "d1", "text": "Hello world"}]
            },
            "destination": {
                "database": "test.needle",
                "collection": "docs",
                "dimensions": 384
            },
            "embedding": {
                "provider": "openai",
                "model": "text-embedding-3-small"
            }
        });

        let pipeline = parse_ingestion_pipeline(&json.to_string()).unwrap();
        assert_eq!(pipeline.name, "my-rag-pipeline");
        assert_eq!(pipeline.chunker.chunk_size, 512); // From rag-default preset
    }

    #[test]
    fn test_validate_ingestion_pipeline() {
        let pipeline = IngestionPipeline {
            name: "test".into(),
            description: String::new(),
            preset: None,
            source: IngestionSource::Inline {
                documents: vec![InlineDocument {
                    id: "d1".into(),
                    text: "Hello".into(),
                    metadata: serde_json::Value::Null,
                }],
            },
            chunker: ChunkerConfig::default(),
            metadata_extractors: vec![],
            dedup: None,
            destination: DestinationConfig {
                database: "test.needle".into(),
                collection: "docs".into(),
                dimensions: 384,
                distance: "cosine".into(),
            },
            embedding: EmbeddingConfig {
                provider: "openai".into(),
                model: "text-embedding-3-small".into(),
                api_key: String::new(),
                base_url: String::new(),
            },
            checkpoint: CheckpointConfig::default(),
        };

        assert!(validate_ingestion_pipeline(&pipeline).is_ok());
    }

    #[test]
    fn test_validate_pipeline_errors() {
        let pipeline = IngestionPipeline {
            name: String::new(),
            description: String::new(),
            preset: None,
            source: IngestionSource::Inline { documents: vec![] },
            chunker: ChunkerConfig {
                chunk_size: 0,
                ..Default::default()
            },
            metadata_extractors: vec![],
            dedup: None,
            destination: DestinationConfig {
                database: "t.needle".into(),
                collection: "c".into(),
                dimensions: 0,
                distance: "cosine".into(),
            },
            embedding: EmbeddingConfig {
                provider: String::new(),
                model: "m".into(),
                api_key: String::new(),
                base_url: String::new(),
            },
            checkpoint: CheckpointConfig::default(),
        };

        let errors = validate_ingestion_pipeline(&pipeline).unwrap_err();
        assert!(errors.len() >= 3);
    }

    #[test]
    fn test_empty_text_chunking() {
        let config = ChunkerConfig::default();
        let chunks = chunk_text("", &config);
        assert!(chunks.is_empty() || chunks.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn test_small_text_no_split() {
        let config = ChunkerConfig {
            chunk_size: 1000,
            ..Default::default()
        };
        let chunks = chunk_text("Short text.", &config);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text.");
    }

    #[test]
    fn test_minhash_identical() {
        let fp1 = MinHashFingerprint::compute("hello world foo bar", 64);
        let fp2 = MinHashFingerprint::compute("hello world foo bar", 64);
        assert!((fp1.similarity(&fp2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_minhash_different() {
        let fp1 = MinHashFingerprint::compute("the quick brown fox jumps", 64);
        let fp2 = MinHashFingerprint::compute("completely unrelated text here now", 64);
        assert!(fp1.similarity(&fp2) < 0.5);
    }

    #[test]
    fn test_deduplicate_chunks() {
        let chunks = vec![
            "hello world this is a test".to_string(),
            "hello world this is a test".to_string(), // exact duplicate
            "completely different text here".to_string(),
        ];
        let config = DedupConfig {
            threshold: 0.8,
            num_hashes: 64,
        };
        let kept = deduplicate_chunks(&chunks, &config);
        assert_eq!(kept.len(), 2); // duplicate removed
        assert!(kept.contains(&0));
        assert!(kept.contains(&2));
    }

    #[test]
    fn test_deduplicate_empty() {
        let config = DedupConfig {
            threshold: 0.9,
            num_hashes: 64,
        };
        assert!(deduplicate_chunks(&[], &config).is_empty());
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("checkpoint.json");

        let mut cp = PipelineCheckpoint::new("test-pipe", 100);
        cp.update(50, 200);
        cp.save(&path).unwrap();

        let loaded = PipelineCheckpoint::load(&path).unwrap();
        assert_eq!(loaded.pipeline_name, "test-pipe");
        assert_eq!(loaded.last_processed_index, 50);
        assert_eq!(loaded.chunks_generated, 200);
        assert!((loaded.progress_pct() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_execute_pipeline_inline() {
        let pipeline = IngestionPipeline {
            name: "test".into(),
            description: String::new(),
            preset: None,
            source: IngestionSource::Inline {
                documents: vec![
                    InlineDocument {
                        id: "doc1".into(),
                        text: "First sentence. Second sentence. Third one here.".into(),
                        metadata: serde_json::json!({"source": "test"}),
                    },
                    InlineDocument {
                        id: "doc2".into(),
                        text: "Another document with some text.".into(),
                        metadata: serde_json::Value::Null,
                    },
                ],
            },
            chunker: ChunkerConfig {
                strategy: ChunkStrategy::Sentence,
                chunk_size: 30,
                chunk_overlap: 5,
                ..Default::default()
            },
            metadata_extractors: vec![],
            dedup: None,
            destination: DestinationConfig {
                database: "t.needle".into(),
                collection: "c".into(),
                dimensions: 128,
                distance: "cosine".into(),
            },
            embedding: EmbeddingConfig {
                provider: "test".into(),
                model: "test".into(),
                api_key: String::new(),
                base_url: String::new(),
            },
            checkpoint: CheckpointConfig { enabled: false, ..Default::default() },
        };

        let chunks = execute_ingestion_pipeline(&pipeline).unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].doc_id, "doc1");
    }

    #[test]
    fn test_execute_pipeline_with_dedup() {
        let pipeline = IngestionPipeline {
            name: "dedup-test".into(),
            description: String::new(),
            preset: None,
            source: IngestionSource::Inline {
                documents: vec![InlineDocument {
                    id: "d1".into(),
                    text: "Same text repeated. Same text repeated. Different ending.".into(),
                    metadata: serde_json::Value::Null,
                }],
            },
            chunker: ChunkerConfig {
                strategy: ChunkStrategy::Sentence,
                chunk_size: 25,
                chunk_overlap: 0,
                ..Default::default()
            },
            metadata_extractors: vec![],
            dedup: Some(DedupConfig {
                threshold: 0.8,
                num_hashes: 64,
            }),
            destination: DestinationConfig {
                database: "t.needle".into(),
                collection: "c".into(),
                dimensions: 128,
                distance: "cosine".into(),
            },
            embedding: EmbeddingConfig {
                provider: "test".into(),
                model: "test".into(),
                api_key: String::new(),
                base_url: String::new(),
            },
            checkpoint: CheckpointConfig::default(),
        };

        let chunks = execute_ingestion_pipeline(&pipeline).unwrap();
        // Dedup should reduce the number of identical chunks
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_execute_pipeline_with_static_metadata() {
        let mut fields = std::collections::HashMap::new();
        fields.insert("pipeline".to_string(), serde_json::json!("test-v1"));

        let pipeline = IngestionPipeline {
            name: "meta-test".into(),
            description: String::new(),
            preset: None,
            source: IngestionSource::Inline {
                documents: vec![InlineDocument {
                    id: "d1".into(),
                    text: "Hello world.".into(),
                    metadata: serde_json::Value::Null,
                }],
            },
            chunker: ChunkerConfig::default(),
            metadata_extractors: vec![MetadataExtractor::Static { fields }],
            dedup: None,
            destination: DestinationConfig {
                database: "t.needle".into(),
                collection: "c".into(),
                dimensions: 128,
                distance: "cosine".into(),
            },
            embedding: EmbeddingConfig {
                provider: "test".into(),
                model: "test".into(),
                api_key: String::new(),
                base_url: String::new(),
            },
            checkpoint: CheckpointConfig::default(),
        };

        let chunks = execute_ingestion_pipeline(&pipeline).unwrap();
        assert!(!chunks.is_empty());
        // Check static metadata was applied
        let meta = &chunks[0].metadata;
        assert_eq!(meta.get("pipeline").and_then(|v| v.as_str()), Some("test-v1"));
    }
}
