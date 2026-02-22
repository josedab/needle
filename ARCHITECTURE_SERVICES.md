# Services Architecture — Needle Vector Database

## Overview

The `src/services/` directory contains 65+ high-level service modules that wrap
core database operations with domain-specific APIs. Services are organized into
six generations (v1–v6), each building on the previous.

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

## Key Dependencies

- All services use `crate::error::{NeedleError, Result}` for error handling
- Core services depend on `crate::collection::Collection` and `crate::database::Database`
- ML services depend on `crate::services::inference_engine::InferenceEngine`
- Search services depend on `crate::search::cost_estimator::CostEstimator`
- Sync services depend on `crate::services::sync_engine::SyncEngine`

## Adding a New Service

See `src/services/README.md` for the step-by-step guide.
