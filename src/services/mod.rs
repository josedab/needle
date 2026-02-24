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

// ── Next-Gen v3 Services ─────────────────────────────────────────────────────
pub mod agentic_memory_protocol;
pub mod graph_knowledge_service;
pub mod live_replication;
pub mod llm_cache_middleware;
pub mod multimodal_collection;
pub mod plugin_ecosystem;
pub mod query_optimizer;
pub mod text_to_vector;
pub mod transactional_api;
pub mod wasm_persistence;

// ── Next-Gen v4 Services ─────────────────────────────────────────────────────
pub mod collection_bundle;
pub mod collection_federation;
pub mod collection_rbac;
pub mod drift_monitor;
pub mod materialized_views;
pub mod otel_tracing;
pub mod query_replay;
pub mod smart_auto_embed;
pub mod snapshot_time_travel;
pub mod vector_pipeline;

// ── Next-Gen v5 Services ─────────────────────────────────────────────────────
pub mod ann_benchmark;
pub mod api_stability;
pub mod cdc_framework;
pub mod cloud_deploy;
pub mod community;
pub mod compliance;
pub mod format_spec;
pub mod model_runtime;
pub mod module_audit;
pub mod python_sdk;

// ── Next-Gen v6 Services ─────────────────────────────────────────────────────
pub mod adaptive_index_selector;
pub mod edge_serverless;
pub mod encrypted_search;
pub mod gpu_kernels;
pub mod graph_query;
pub mod llm_tools;
pub mod managed_cloud;
pub mod multi_writer;
pub mod realtime_streaming;
pub mod vector_lineage;

// ── Next-Gen v7 Services ─────────────────────────────────────────────────────
pub mod benchmark_runner;
pub mod cluster_bootstrap;
pub mod evidence_collector;
pub mod model_downloader;
pub mod pricing;
pub mod rag_sdk;
pub mod triage_report;
pub mod unwrap_audit;
pub mod vscode_extension;
pub mod ws_protocol;

// ── Next-Gen v8 Services ─────────────────────────────────────────────────────
pub mod hnsw_compactor;
pub mod matryoshka_service;
pub mod pipeline_manager;
pub mod snapshot_manager;
pub mod wasm_plugin_runtime;

// ── Next-Gen v9 Services ─────────────────────────────────────────────────────
pub mod backup_command;
pub mod change_stream;
pub mod client_sdk;
pub mod grpc_schema;
pub mod readiness_probe;
