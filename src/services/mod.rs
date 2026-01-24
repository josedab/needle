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
#[cfg(feature = "experimental")]
pub mod adaptive_optimizer;
#[cfg(feature = "experimental")]
pub mod edge_runtime;
#[cfg(feature = "experimental")]
pub mod graphrag_service;
#[cfg(feature = "experimental")]
pub mod incremental_sync;
#[cfg(feature = "experimental")]
pub mod managed_embeddings;
#[cfg(feature = "experimental")]
pub mod nl_filter_parser;
#[cfg(feature = "experimental")]
pub mod streaming_ingest;
#[cfg(feature = "experimental")]
pub mod time_travel_query;
#[cfg(feature = "experimental")]
pub mod visual_explorer;
#[cfg(feature = "experimental")]
pub mod wasm_sdk;

// ── Next-Gen v2 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod inference_engine;
#[cfg(feature = "experimental")]
pub mod plugin_api;
pub mod semantic_cache;
#[cfg(feature = "experimental")]
pub mod streaming_protocol;
#[cfg(feature = "experimental")]
pub mod sync_engine;
#[cfg(feature = "experimental")]
pub mod vector_transactions;
#[cfg(feature = "experimental")]
pub mod wasm_browser;

// ── Next-Gen v3 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod agentic_memory_protocol;
#[cfg(feature = "experimental")]
pub mod graph_knowledge_service;
#[cfg(feature = "experimental")]
pub mod live_replication;
#[cfg(feature = "experimental")]
pub mod llm_cache_middleware;
#[cfg(feature = "experimental")]
pub mod multimodal_collection;
#[cfg(feature = "experimental")]
pub mod plugin_ecosystem;
#[cfg(feature = "experimental")]
pub mod query_optimizer;
#[cfg(feature = "experimental")]
pub mod text_to_vector;
#[cfg(feature = "experimental")]
pub mod transactional_api;
#[cfg(feature = "experimental")]
pub mod wasm_persistence;

// ── Next-Gen v4 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod collection_bundle;
#[cfg(feature = "experimental")]
pub mod collection_federation;
#[cfg(feature = "experimental")]
pub mod collection_rbac;
#[cfg(feature = "experimental")]
pub mod drift_monitor;
#[cfg(feature = "experimental")]
pub mod materialized_views;
#[cfg(feature = "experimental")]
pub mod otel_tracing;
#[cfg(feature = "experimental")]
pub mod query_replay;
#[cfg(feature = "experimental")]
pub mod smart_auto_embed;
#[cfg(feature = "experimental")]
pub mod snapshot_time_travel;
#[cfg(feature = "experimental")]
pub mod vector_pipeline;

// ── Next-Gen v5 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod ann_benchmark;
#[cfg(feature = "experimental")]
pub mod api_stability;
#[cfg(feature = "experimental")]
pub mod cdc_framework;
#[cfg(feature = "experimental")]
pub mod cloud_deploy;
#[cfg(feature = "experimental")]
pub mod community;
#[cfg(feature = "experimental")]
pub mod compliance;
#[cfg(feature = "experimental")]
pub mod format_spec;
#[cfg(feature = "experimental")]
pub mod model_runtime;
#[cfg(feature = "experimental")]
pub mod module_audit;
#[cfg(feature = "experimental")]
pub mod python_sdk;

// ── Next-Gen v6 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod adaptive_index_selector;
#[cfg(feature = "experimental")]
pub mod edge_serverless;
#[cfg(feature = "experimental")]
pub mod encrypted_search;
#[cfg(feature = "experimental")]
pub mod gpu_kernels;
#[cfg(feature = "experimental")]
pub mod graph_query;
#[cfg(feature = "experimental")]
pub mod llm_tools;
#[cfg(feature = "experimental")]
pub mod managed_cloud;
#[cfg(feature = "experimental")]
pub mod multi_writer;
#[cfg(feature = "experimental")]
pub mod realtime_streaming;
#[cfg(feature = "experimental")]
pub mod vector_lineage;

// ── Next-Gen v7 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod benchmark_runner;
#[cfg(feature = "experimental")]
pub mod cluster_bootstrap;
#[cfg(feature = "experimental")]
pub mod evidence_collector;
#[cfg(feature = "experimental")]
pub mod model_downloader;
#[cfg(feature = "experimental")]
pub mod pricing;
#[cfg(feature = "experimental")]
pub mod rag_sdk;
#[cfg(feature = "experimental")]
pub mod triage_report;
#[cfg(feature = "experimental")]
pub mod unwrap_audit;
#[cfg(feature = "experimental")]
pub mod vscode_extension;
#[cfg(feature = "experimental")]
pub mod ws_protocol;

// ── Next-Gen v8 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod hnsw_compactor;
#[cfg(feature = "experimental")]
pub mod matryoshka_service;
#[cfg(feature = "experimental")]
pub mod pipeline_manager;
#[cfg(feature = "experimental")]
pub mod snapshot_manager;
#[cfg(feature = "experimental")]
pub mod wasm_plugin_runtime;

// ── Next-Gen v9 Services ─────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod auto_embed_endpoint;
#[cfg(feature = "experimental")]
pub mod backup_command;
#[cfg(feature = "experimental")]
pub mod change_stream;
#[cfg(feature = "experimental")]
pub mod client_sdk;
#[cfg(feature = "experimental")]
pub mod format_validator;
#[cfg(feature = "experimental")]
pub mod grpc_schema;
#[cfg(feature = "experimental")]
pub mod notebook;
#[cfg(feature = "experimental")]
pub mod query_cache_middleware;
#[cfg(feature = "experimental")]
pub mod readiness_probe;
#[cfg(feature = "experimental")]
pub mod tenant_router;

// ── Next-Gen v10 Services ────────────────────────────────────────────────────
pub mod embedding_router;
pub mod webhook_delivery;

// ── Next-Gen v11 Services ────────────────────────────────────────────────────
#[cfg(feature = "experimental")]
pub mod distributed_federation;
#[cfg(feature = "experimental")]
pub mod needleql_executor;
#[cfg(feature = "experimental")]
pub mod cloud_service;
#[cfg(feature = "experimental")]
pub mod crdt_sync;
#[cfg(feature = "experimental")]
pub mod storage_backends;
#[cfg(feature = "experimental")]
pub mod version_control;
#[cfg(feature = "experimental")]
pub mod agentic_workflow;
#[cfg(feature = "experimental")]
pub mod benchmark_suite;
#[cfg(feature = "experimental")]
pub mod typed_schema;
#[cfg(feature = "experimental")]
pub mod needleql_lsp;
