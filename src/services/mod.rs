//! High-level services: database-level service wrappers with builders and lifecycle management.
//!
//! Services are organized into domain-specific subdirectories:
//!
//! - **`ai/`** — agentic_memory_protocol, agentic_workflow, graph_knowledge_service, ...
//! - **`client/`** — client_sdk, grpc_schema, notebook, ...
//! - **`collection/`** — collection_bundle, collection_federation, collection_rbac, ...
//! - **`compute/`** — adaptive_optimizer, adaptive_service, gpu_kernels, ...
//! - **`embedding/`** — auto_embed_endpoint, embedding_router, inference_engine, ...
//! - **`governance/`** — api_stability, community, compliance, ...
//! - **`infrastructure/`** — cloud_deploy, cloud_service, cluster_bootstrap, ...
//! - **`observability/`** — ann_benchmark, benchmark_runner, benchmark_suite, ...
//! - **`pipeline/`** — cdc_framework, ingestion_pipeline, ingestion_service, ...
//! - **`plugin/`** — plugin_api, plugin_ecosystem, plugin_runtime, ...
//! - **`search/`** — adaptive_index_selector, encrypted_search, needleql_executor, ...
//! - **`storage/`** — backup_command, hnsw_compactor, snapshot_manager, ...
//! - **`sync/`** — change_stream, crdt_sync, distributed_federation, ...

#![allow(missing_docs)]

#[cfg(feature = "experimental")]
pub mod ai;
#[cfg(feature = "experimental")]
pub mod client;
pub mod collection;
#[cfg(feature = "experimental")]
pub mod compute;
#[cfg(feature = "experimental")]
pub mod embedding;
#[cfg(feature = "experimental")]
pub mod governance;
#[cfg(feature = "experimental")]
pub mod infrastructure;
#[cfg(feature = "experimental")]
pub mod observability;
pub mod pipeline;
#[cfg(feature = "experimental")]
pub mod plugin;
#[cfg(feature = "experimental")]
pub mod search;
pub mod storage;
#[cfg(feature = "experimental")]
pub mod sync;

/// Live migration toolkit for importing from external vector databases.
pub mod live_migration_service;

/// Client-side vector caching with LRU eviction, TTL, and invalidation.
pub mod client_cache;

// ── Re-exports for backward compatibility ─────────────────────────────────
// Modules are re-exported at the services:: level so existing code continues
// to work with crate::services::module_name paths.

#[cfg(feature = "experimental")]
pub use ai::agentic_memory_protocol;
#[cfg(feature = "experimental")]
pub use ai::agentic_workflow;
#[cfg(feature = "experimental")]
pub use ai::graph_knowledge_service;
#[cfg(feature = "experimental")]
pub use ai::graph_query;
#[cfg(feature = "experimental")]
pub use ai::graphrag_service;
#[cfg(feature = "experimental")]
pub use ai::llm_cache_middleware;
#[cfg(feature = "experimental")]
pub use ai::llm_tools;
#[cfg(feature = "experimental")]
pub use ai::rag_sdk;
#[cfg(feature = "experimental")]
pub use ai::semantic_cache;
#[cfg(feature = "experimental")]
pub use client::client_sdk;
#[cfg(feature = "experimental")]
pub use client::grpc_schema;
#[cfg(feature = "experimental")]
pub use client::notebook;
#[cfg(feature = "experimental")]
pub use client::python_sdk;
#[cfg(feature = "experimental")]
pub use client::vscode_extension;
#[cfg(feature = "experimental")]
pub use client::webhook_delivery;
#[cfg(feature = "experimental")]
pub use client::ws_protocol;
#[cfg(feature = "experimental")]
pub use collection::collection_bundle;
#[cfg(feature = "experimental")]
pub use collection::collection_federation;
#[cfg(feature = "experimental")]
pub use collection::collection_rbac;
#[cfg(feature = "experimental")]
pub use collection::materialized_views;
#[cfg(feature = "experimental")]
pub use collection::multimodal_collection;
pub use collection::multimodal_service;
pub use collection::pitr_service;
pub use collection::text_collection;
#[cfg(feature = "experimental")]
pub use collection::typed_schema;
pub use collection::vector_namespace;
#[cfg(feature = "experimental")]
pub use compute::adaptive_optimizer;
#[cfg(feature = "experimental")]
pub use compute::adaptive_service;
#[cfg(feature = "experimental")]
pub use compute::gpu_kernels;
#[cfg(feature = "experimental")]
pub use compute::time_travel_query;
#[cfg(feature = "experimental")]
pub use compute::transactional_api;
#[cfg(feature = "experimental")]
pub use compute::vector_transactions;
#[cfg(feature = "experimental")]
pub use embedding::auto_embed_endpoint;
#[cfg(feature = "experimental")]
pub use embedding::embedding_router;
#[cfg(feature = "experimental")]
pub use embedding::inference_engine;
#[cfg(feature = "experimental")]
pub use embedding::managed_embeddings;
#[cfg(feature = "experimental")]
pub use embedding::matryoshka_service;
#[cfg(feature = "experimental")]
pub use embedding::model_downloader;
#[cfg(feature = "experimental")]
pub use embedding::model_runtime;
#[cfg(feature = "experimental")]
pub use embedding::smart_auto_embed;
#[cfg(feature = "experimental")]
pub use embedding::text_to_vector;
#[cfg(feature = "experimental")]
pub use governance::api_stability;
#[cfg(feature = "experimental")]
pub use governance::community;
#[cfg(feature = "experimental")]
pub use governance::compliance;
#[cfg(feature = "experimental")]
pub use governance::format_spec;
#[cfg(feature = "experimental")]
pub use governance::format_validator;
#[cfg(feature = "experimental")]
pub use governance::module_audit;
#[cfg(feature = "experimental")]
pub use governance::unwrap_audit;
#[cfg(feature = "experimental")]
pub use governance::version_control;
#[cfg(feature = "experimental")]
pub use infrastructure::cloud_deploy;
#[cfg(feature = "experimental")]
pub use infrastructure::cloud_service;
#[cfg(feature = "experimental")]
pub use infrastructure::cluster_bootstrap;
#[cfg(feature = "experimental")]
pub use infrastructure::edge_runtime;
#[cfg(feature = "experimental")]
pub use infrastructure::edge_serverless;
#[cfg(feature = "experimental")]
pub use infrastructure::managed_cloud;
#[cfg(feature = "experimental")]
pub use infrastructure::otel_tracing;
#[cfg(feature = "experimental")]
pub use infrastructure::pricing;
#[cfg(feature = "experimental")]
pub use infrastructure::readiness_probe;
#[cfg(feature = "experimental")]
pub use infrastructure::tenant_router;
#[cfg(feature = "experimental")]
pub use observability::ann_benchmark;
#[cfg(feature = "experimental")]
pub use observability::benchmark_runner;
#[cfg(feature = "experimental")]
pub use observability::benchmark_suite;
#[cfg(feature = "experimental")]
pub use observability::drift_monitor;
#[cfg(feature = "experimental")]
pub use observability::evidence_collector;
#[cfg(feature = "experimental")]
pub use observability::triage_report;
#[cfg(feature = "experimental")]
pub use observability::vector_lineage;
#[cfg(feature = "experimental")]
pub use observability::visual_explorer;
#[cfg(feature = "experimental")]
pub use pipeline::cdc_framework;
#[cfg(feature = "experimental")]
pub use pipeline::ingestion_pipeline;
pub use pipeline::ingestion_service;
#[cfg(feature = "experimental")]
pub use pipeline::pipeline_manager;
#[cfg(feature = "experimental")]
pub use pipeline::realtime_streaming;
#[cfg(feature = "experimental")]
pub use pipeline::streaming_ingest;
#[cfg(feature = "experimental")]
pub use pipeline::streaming_protocol;
#[cfg(feature = "experimental")]
pub use pipeline::vector_pipeline;
#[cfg(feature = "experimental")]
pub use plugin::plugin_api;
#[cfg(feature = "experimental")]
pub use plugin::plugin_ecosystem;
#[cfg(feature = "experimental")]
pub use plugin::plugin_runtime;
#[cfg(feature = "experimental")]
pub use plugin::wasm_browser;
#[cfg(feature = "experimental")]
pub use plugin::wasm_persistence;
#[cfg(feature = "experimental")]
pub use plugin::wasm_plugin_runtime;
#[cfg(feature = "experimental")]
pub use plugin::wasm_sdk;
#[cfg(feature = "experimental")]
pub use search::adaptive_index_selector;
#[cfg(feature = "experimental")]
pub use search::encrypted_search;
#[cfg(feature = "experimental")]
pub use search::needleql_executor;
#[cfg(feature = "experimental")]
pub use search::needleql_lsp;
#[cfg(feature = "experimental")]
pub use search::nl_filter_parser;
#[cfg(feature = "experimental")]
pub use search::query_cache_middleware;
#[cfg(feature = "experimental")]
pub use search::query_optimizer;
#[cfg(feature = "experimental")]
pub use search::query_replay;
#[cfg(feature = "experimental")]
pub use storage::backup_command;
#[cfg(feature = "experimental")]
pub use storage::hnsw_compactor;
#[cfg(feature = "experimental")]
pub use storage::snapshot_manager;
#[cfg(feature = "experimental")]
pub use storage::snapshot_time_travel;
#[cfg(feature = "experimental")]
pub use storage::storage_backends;
pub use storage::tiered_service;
#[cfg(feature = "experimental")]
pub use sync::change_stream;
#[cfg(feature = "experimental")]
pub use sync::crdt_sync;
#[cfg(feature = "experimental")]
pub use sync::distributed_federation;
#[cfg(feature = "experimental")]
pub use sync::incremental_sync;
#[cfg(feature = "experimental")]
pub use sync::live_replication;
#[cfg(feature = "experimental")]
pub use sync::multi_writer;
#[cfg(feature = "experimental")]
pub use sync::sync_engine;
