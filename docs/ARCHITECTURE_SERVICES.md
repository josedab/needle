# Services Architecture — Needle Vector Database

## Overview

The `src/services/` directory contains 90+ high-level service modules that wrap
core database operations with domain-specific APIs. Services are organized into
ten generations (v1–v10), each building on the previous.

> **Note:** Most service modules (v2–v10) are gated behind the `experimental`
> feature flag to reduce default compile times. Build with `--features experimental`
> or `--features full` to include them.

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM / AI Layer                           │
│  llm_tools ← agentic_memory_protocol ← smart_auto_embed        │
│  llm_cache_middleware ← semantic_cache                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                     Search & Query Layer                        │
│  graph_query ← graph_knowledge_service                         │
│  query_optimizer ← cost_estimator (search/)                    │
│  materialized_views ← drift_monitor                            │
│  collection_federation ← collection_rbac                       │
│  encrypted_search (standalone)                                  │
│  adaptive_index_selector (standalone)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                    Data Pipeline Layer                           │
│  vector_pipeline ← streaming_protocol ← cdc_framework          │
│  ingestion_service ← text_to_vector ← inference_engine         │
│  model_runtime ← inference_engine                              │
│  live_replication ← sync_engine                                │
│  multi_writer (standalone Raft)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                      Core Services Layer                        │
│  vector_transactions ← transactional_api                       │
│  snapshot_time_travel (standalone)                              │
│  collection_bundle (standalone)                                │
│  vector_lineage (standalone)                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  managed_cloud ← cloud_deploy                                  │
│  edge_serverless ← wasm_browser ← wasm_persistence            │
│  gpu_kernels (standalone)                                      │
│  realtime_streaming (standalone)                               │
│  otel_tracing (standalone)                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                     Governance Layer                             │
│  api_stability ← module_audit                                  │
│  compliance (standalone)                                       │
│  community (standalone)                                        │
│  ann_benchmark (standalone)                                    │
│  format_spec (standalone)                                      │
│  python_sdk (standalone)                                       │
│  plugin_api ← plugin_ecosystem                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                    Developer Tooling Layer                       │
│  benchmark_runner (standalone)                                  │
│  cluster_bootstrap (standalone)                                 │
│  evidence_collector ← triage_report                            │
│  model_downloader (standalone)                                  │
│  rag_sdk (standalone)                                          │
│  vscode_extension (standalone)                                  │
│  ws_protocol (standalone)                                      │
│  pricing ← unwrap_audit                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ uses
┌──────────────────────────▼──────────────────────────────────────┐
│                  Platform Completeness Layer                     │
│  hnsw_compactor ← snapshot_manager ← pipeline_manager          │
│  matryoshka_service (standalone)                                │
│  wasm_plugin_runtime (standalone)                               │
│  auto_embed_endpoint (standalone)                               │
│  backup_command (standalone)                                    │
│  change_stream ← webhook_delivery                              │
│  client_sdk (standalone)                                       │
│  embedding_router (standalone)                                  │
│  query_cache_middleware (standalone)                             │
│  readiness_probe (standalone)                                   │
│  tenant_router (standalone)                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Generation History

| Generation | Modules | Focus |
|-----------|---------|-------|
| **v1** (core) | ingestion, multimodal, pitr, text_collection, tiered | Basic service wrappers |
| **v2** | inference_engine, semantic_cache, vector_transactions, sync_engine, streaming_protocol, plugin_api, wasm_browser | Core next-gen features |
| **v3** | agentic_memory, text_to_vector, live_replication, query_optimizer, llm_cache, wasm_persistence, graph_knowledge, transactional_api, plugin_ecosystem, multimodal_collection | Advanced integrations |
| **v4** | vector_pipeline, snapshot_time_travel, smart_auto_embed, collection_rbac, materialized_views, collection_federation, drift_monitor, collection_bundle, query_replay, otel_tracing | Production readiness |
| **v5** | api_stability, ann_benchmark, model_runtime, cloud_deploy, community, module_audit, python_sdk, cdc_framework, format_spec, compliance | Project excellence |
| **v6** | managed_cloud, gpu_kernels, realtime_streaming, multi_writer, graph_query, adaptive_index_selector, encrypted_search, vector_lineage, edge_serverless, llm_tools | Market expansion |
| **v7** | benchmark_runner, cluster_bootstrap, evidence_collector, model_downloader, pricing, rag_sdk, triage_report, unwrap_audit, vscode_extension, ws_protocol | Developer tooling |
| **v8** | hnsw_compactor, matryoshka_service, pipeline_manager, snapshot_manager, wasm_plugin_runtime | Index lifecycle |
| **v9** | auto_embed_endpoint, backup_command, change_stream, client_sdk, format_validator, grpc_schema, notebook, query_cache_middleware, readiness_probe, tenant_router | Platform completeness |
| **v10** | embedding_router, webhook_delivery | Event-driven integration |

## Key Dependencies

- All services use `crate::error::{NeedleError, Result}` for error handling
- Core services depend on `crate::collection::Collection` and `crate::database::Database`
- ML services depend on `crate::services::inference_engine::InferenceEngine`
- Search services depend on `crate::search::cost_estimator::CostEstimator`
- Sync services depend on `crate::services::sync_engine::SyncEngine`

## Adding a New Service

See `src/services/README.md` for the step-by-step guide.
