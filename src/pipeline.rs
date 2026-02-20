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
