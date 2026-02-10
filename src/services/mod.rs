//! High-level services: database-level service wrappers with builders and lifecycle management.

pub mod ingestion_service;
#[cfg(feature = "experimental")]
pub mod adaptive_service;
pub mod text_collection;
pub mod pitr_service;
pub mod multimodal_service;
pub mod tiered_service;
#[cfg(feature = "experimental")]
pub mod plugin_runtime;
#[cfg(feature = "experimental")]
pub mod ingestion_pipeline;

// ── Next-Gen Services ────────────────────────────────────────────────────────
pub mod streaming_ingest;
pub mod adaptive_optimizer;
pub mod wasm_sdk;
pub mod managed_embeddings;
pub mod time_travel_query;
pub mod graphrag_service;
pub mod edge_runtime;
pub mod nl_filter_parser;
pub mod incremental_sync;
pub mod visual_explorer;
