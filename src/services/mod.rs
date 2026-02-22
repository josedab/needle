//! High-level services: database-level service wrappers with builders and lifecycle management.

#[cfg(feature = "experimental")]
pub mod adaptive_service;
#[cfg(feature = "experimental")]
pub mod ingestion_pipeline;
pub mod ingestion_service;
pub mod multimodal_service;
pub mod pitr_service;
#[cfg(feature = "experimental")]
pub mod plugin_runtime;
pub mod text_collection;
pub mod tiered_service;

// ── Next-Gen Services ────────────────────────────────────────────────────────
pub mod adaptive_optimizer;
pub mod edge_runtime;
pub mod graphrag_service;
pub mod incremental_sync;
pub mod managed_embeddings;
pub mod nl_filter_parser;
pub mod streaming_ingest;
pub mod time_travel_query;
pub mod visual_explorer;
pub mod wasm_sdk;

// ── Next-Gen v2 Services ─────────────────────────────────────────────────────
pub mod inference_engine;
pub mod plugin_api;
pub mod semantic_cache;
pub mod streaming_protocol;
pub mod sync_engine;
pub mod vector_transactions;
pub mod wasm_browser;
