//! Auto-Embedding HTTP Endpoint
//!
//! REST endpoint service that accepts raw text and auto-embeds it using a
//! configured provider before indexing. Eliminates the need for users to
//! bring their own embeddings — text in, vectors indexed.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::auto_embed_endpoint::{
//!     AutoEmbedService, EndpointConfig, TextInsertRequest, TextInsertResponse,
//!     BatchTextInsertRequest, CollectionEmbedConfig,
//! };
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 384).unwrap();
//!
//! let mut svc = AutoEmbedService::new(EndpointConfig::default());
//!
//! // Single text insert
//! let resp = svc.insert_text(
//!     &db,
//!     "docs",
//!     TextInsertRequest {
//!         id: "doc1".into(),
//!         text: "Rust is a systems programming language".into(),
//!         metadata: None,
//!     },
//! ).unwrap();
//! assert!(resp.success);
//!
//! // Batch insert
//! let batch_resp = svc.insert_texts(
//!     &db,
//!     "docs",
//!     BatchTextInsertRequest {
//!         documents: vec![
//!             TextInsertRequest { id: "d2".into(), text: "Python is great".into(), metadata: None },
//!             TextInsertRequest { id: "d3".into(), text: "Go is fast".into(), metadata: None },
//!         ],
//!     },
//! ).unwrap();
//! assert_eq!(batch_resp.inserted, 2);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::ml::local_inference::{InferenceConfig, InferenceEngine};

// ── Configuration ────────────────────────────────────────────────────────────

/// Per-collection embedding configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionEmbedConfig {
    /// Model identifier (from local inference registry).
    pub model_id: String,
    /// Whether to store original text in metadata.
    pub store_text: bool,
}

impl Default for CollectionEmbedConfig {
    fn default() -> Self {
        Self {
            model_id: "mock-384".into(),
            store_text: true,
        }
    }
}

/// Endpoint configuration.
#[derive(Debug, Clone)]
pub struct EndpointConfig {
    /// Default model for collections without explicit config.
    pub default_model: String,
    /// Whether to store original text by default.
    pub default_store_text: bool,
    /// Maximum text length per document (bytes).
    pub max_text_bytes: usize,
    /// Maximum batch size.
    pub max_batch_size: usize,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            default_model: "mock-384".into(),
            default_store_text: true,
            max_text_bytes: 1_000_000,
            max_batch_size: 1000,
        }
    }
}

// ── Request/Response Types ──────────────────────────────────────────────────

/// Request to insert a single text document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInsertRequest {
    pub id: String,
    pub text: String,
    pub metadata: Option<Value>,
}

/// Response from a text insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInsertResponse {
    pub success: bool,
    pub id: String,
    pub dimensions: usize,
    pub text_length: usize,
}

/// Request to insert multiple text documents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTextInsertRequest {
    pub documents: Vec<TextInsertRequest>,
}

/// Response from a batch text insert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTextInsertResponse {
    pub inserted: usize,
    pub failed: usize,
    pub errors: Vec<String>,
}

/// Request to configure auto-embedding for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigureRequest {
    pub model_id: String,
    pub store_text: bool,
}

// ── Service ─────────────────────────────────────────────────────────────────

/// Auto-embedding service for HTTP text-to-vector endpoints.
pub struct AutoEmbedService {
    config: EndpointConfig,
    collection_configs: HashMap<String, CollectionEmbedConfig>,
    engines: HashMap<String, InferenceEngine>,
}

impl AutoEmbedService {
    /// Create a new auto-embed service.
    pub fn new(config: EndpointConfig) -> Self {
        Self {
            config,
            collection_configs: HashMap::new(),
            engines: HashMap::new(),
        }
    }

    /// Configure embedding model for a specific collection.
    pub fn configure_collection(
        &mut self,
        collection: &str,
        embed_config: CollectionEmbedConfig,
    ) -> Result<()> {
        // Pre-initialize the engine to validate the model exists
        let engine = self.get_or_create_engine(&embed_config.model_id)?;
        drop(engine); // validation only — engine is cached for later use
        self.collection_configs
            .insert(collection.to_string(), embed_config);
        Ok(())
    }

    /// Insert a single text document: embed + index.
    pub fn insert_text(
        &mut self,
        db: &Database,
        collection: &str,
        req: TextInsertRequest,
    ) -> Result<TextInsertResponse> {
        self.validate_text(&req.text)?;

        let embed_config = self.resolve_config(collection);
        let engine = self.get_or_create_engine(&embed_config.model_id)?;
        let embedding = engine.embed_text(&req.text)?;
        let dims = embedding.len();

        let mut meta = req.metadata.unwrap_or_else(|| serde_json::json!({}));
        if embed_config.store_text {
            if let Value::Object(ref mut map) = meta {
                map.insert("_text".into(), Value::String(req.text.clone()));
                map.insert("_model".into(), Value::String(embed_config.model_id.clone()));
            }
        }

        let coll = db.collection(collection)?;
        coll.insert(&req.id, &embedding, Some(meta))?;

        Ok(TextInsertResponse {
            success: true,
            id: req.id,
            dimensions: dims,
            text_length: req.text.len(),
        })
    }

    /// Insert multiple text documents in a batch.
    pub fn insert_texts(
        &mut self,
        db: &Database,
        collection: &str,
        req: BatchTextInsertRequest,
    ) -> Result<BatchTextInsertResponse> {
        if req.documents.len() > self.config.max_batch_size {
            return Err(NeedleError::InvalidArgument(format!(
                "Batch size {} exceeds max {}",
                req.documents.len(),
                self.config.max_batch_size
            )));
        }

        let mut inserted = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        for doc in req.documents {
            match self.insert_text(db, collection, doc) {
                Ok(_) => inserted += 1,
                Err(e) => {
                    failed += 1;
                    errors.push(e.to_string());
                }
            }
        }

        Ok(BatchTextInsertResponse {
            inserted,
            failed,
            errors,
        })
    }

    /// Search with raw text query (auto-embeds the query).
    pub fn search_text(
        &mut self,
        db: &Database,
        collection: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<TextSearchResult>> {
        let embed_config = self.resolve_config(collection);
        let engine = self.get_or_create_engine(&embed_config.model_id)?;
        let query_embedding = engine.embed_text(query)?;

        let coll = db.collection(collection)?;
        let results = coll.search(&query_embedding, k)?;

        Ok(results
            .into_iter()
            .map(|r| {
                let text = r
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("_text"))
                    .and_then(|v| v.as_str())
                    .map(String::from);
                TextSearchResult {
                    id: r.id,
                    text,
                    distance: r.distance,
                    score: 1.0 / (1.0 + r.distance),
                    metadata: r.metadata,
                }
            })
            .collect())
    }

    /// Get the configured model for a collection.
    pub fn collection_model(&self, collection: &str) -> String {
        self.collection_configs
            .get(collection)
            .map_or_else(|| self.config.default_model.clone(), |c| c.model_id.clone())
    }

    /// List all configured collections.
    pub fn configured_collections(&self) -> Vec<(&str, &CollectionEmbedConfig)> {
        self.collection_configs
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    fn resolve_config(&self, collection: &str) -> CollectionEmbedConfig {
        self.collection_configs
            .get(collection)
            .cloned()
            .unwrap_or(CollectionEmbedConfig {
                model_id: self.config.default_model.clone(),
                store_text: self.config.default_store_text,
            })
    }

    fn validate_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Err(NeedleError::InvalidArgument("Text cannot be empty".into()));
        }
        if text.len() > self.config.max_text_bytes {
            return Err(NeedleError::InvalidArgument(format!(
                "Text length {} exceeds max {}",
                text.len(),
                self.config.max_text_bytes
            )));
        }
        Ok(())
    }

    fn get_or_create_engine(&mut self, model_id: &str) -> Result<&InferenceEngine> {
        if !self.engines.contains_key(model_id) {
            let config = InferenceConfig {
                model_id: model_id.into(),
                ..InferenceConfig::default()
            };
            let engine = InferenceEngine::new(config)?;
            self.engines.insert(model_id.to_string(), engine);
        }
        Ok(self.engines.get(model_id).expect("just inserted"))
    }
}

/// Search result with text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchResult {
    pub id: String,
    pub text: Option<String>,
    pub distance: f32,
    pub score: f32,
    pub metadata: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> Database {
        let db = Database::in_memory();
        db.create_collection("docs", 384).unwrap();
        db
    }

    #[test]
    fn test_insert_and_search_text() {
        let db = setup();
        let mut svc = AutoEmbedService::new(EndpointConfig::default());

        let resp = svc
            .insert_text(
                &db,
                "docs",
                TextInsertRequest {
                    id: "d1".into(),
                    text: "Rust is a systems programming language".into(),
                    metadata: None,
                },
            )
            .unwrap();
        assert!(resp.success);
        assert_eq!(resp.dimensions, 384);

        let results = svc.search_text(&db, "docs", "systems programming", 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "d1");
        assert!(results[0].text.is_some());
    }

    #[test]
    fn test_batch_insert() {
        let db = setup();
        let mut svc = AutoEmbedService::new(EndpointConfig::default());

        let resp = svc
            .insert_texts(
                &db,
                "docs",
                BatchTextInsertRequest {
                    documents: vec![
                        TextInsertRequest { id: "d1".into(), text: "Hello".into(), metadata: None },
                        TextInsertRequest { id: "d2".into(), text: "World".into(), metadata: None },
                        TextInsertRequest { id: "d3".into(), text: "Foo".into(), metadata: None },
                    ],
                },
            )
            .unwrap();
        assert_eq!(resp.inserted, 3);
        assert_eq!(resp.failed, 0);
    }

    #[test]
    fn test_empty_text_rejected() {
        let db = setup();
        let mut svc = AutoEmbedService::new(EndpointConfig::default());
        let result = svc.insert_text(
            &db,
            "docs",
            TextInsertRequest { id: "d1".into(), text: "".into(), metadata: None },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_text_too_large_rejected() {
        let db = setup();
        let mut svc = AutoEmbedService::new(EndpointConfig {
            max_text_bytes: 10,
            ..Default::default()
        });
        let result = svc.insert_text(
            &db,
            "docs",
            TextInsertRequest {
                id: "d1".into(),
                text: "This text is longer than 10 bytes".into(),
                metadata: None,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_size_limit() {
        let db = setup();
        let mut svc = AutoEmbedService::new(EndpointConfig {
            max_batch_size: 2,
            ..Default::default()
        });
        let result = svc.insert_texts(
            &db,
            "docs",
            BatchTextInsertRequest {
                documents: vec![
                    TextInsertRequest { id: "d1".into(), text: "a".into(), metadata: None },
                    TextInsertRequest { id: "d2".into(), text: "b".into(), metadata: None },
                    TextInsertRequest { id: "d3".into(), text: "c".into(), metadata: None },
                ],
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_configure_collection() {
        let mut svc = AutoEmbedService::new(EndpointConfig::default());
        svc.configure_collection(
            "docs",
            CollectionEmbedConfig {
                model_id: "mock-384".into(),
                store_text: false,
            },
        )
        .unwrap();
        assert_eq!(svc.collection_model("docs"), "mock-384");
        assert_eq!(svc.configured_collections().len(), 1);
    }

    #[test]
    fn test_metadata_preserved() {
        let db = setup();
        let mut svc = AutoEmbedService::new(EndpointConfig::default());
        svc.insert_text(
            &db,
            "docs",
            TextInsertRequest {
                id: "d1".into(),
                text: "test".into(),
                metadata: Some(serde_json::json!({"category": "test"})),
            },
        )
        .unwrap();

        let results = svc.search_text(&db, "docs", "test", 1).unwrap();
        let meta = results[0].metadata.as_ref().unwrap();
        assert_eq!(meta.get("category").unwrap().as_str().unwrap(), "test");
        assert!(meta.get("_text").is_some());
        assert!(meta.get("_model").is_some());
    }

    #[test]
    fn test_default_model_fallback() {
        let svc = AutoEmbedService::new(EndpointConfig {
            default_model: "mock-384".into(),
            ..Default::default()
        });
        assert_eq!(svc.collection_model("unconfigured"), "mock-384");
    }
}
