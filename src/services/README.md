# Needle Services Directory

This directory contains 65+ high-level service modules organized by domain.

## 🔧 Core Database Services

| Module | Description |
|--------|-------------|
| `vector_transactions` | ACID multi-collection transactions with rollback |
| `transactional_api` | Fluent builder for transaction composition |
| `snapshot_time_travel` | Point-in-time database snapshots with CoW chains |
| `collection_bundle` | Portable `.needle-bundle` export/import |
| `collection_federation` | Cross-collection federated search with RRF merging |
| `collection_rbac` | Per-collection RBAC policies with row-level security |
| `format_spec` | `.needle` file format specification and validation |
| `api_stability` | v1.0 API stability manifest with deprecation tracking |

## 🔍 Search & Query

| Module | Description |
|--------|-------------|
| `semantic_cache` | Similarity-based LLM response cache |
| `llm_cache_middleware` | Multi-model cache middleware with analytics |
| `materialized_views` | Pre-computed search results with auto-refresh |
| `query_optimizer` | Adaptive query optimizer with calibration feedback |
| `query_replay` | Record and replay queries for regression testing |
| `drift_monitor` | Embedding distribution drift detection with alerts |
| `graph_query` | Native graph traversal operators for GraphRAG |
| `graph_knowledge_service` | Auto entity/relation extraction with graph search |
| `adaptive_index_selector` | Workload-driven automatic index selection |
| `encrypted_search` | LSH-based search on encrypted vectors |
| `nl_filter_parser` | Natural language → metadata filter translation |

## 🤖 ML & Embeddings

| Module | Description |
|--------|-------------|
| `inference_engine` | Built-in embedding generation engine |
| `model_runtime` | Downloadable model registry with hot-swap |
| `text_to_vector` | Zero-config `insert_text()`/`search_text()` |
| `smart_auto_embed` | Multi-backend embedding chain with cache |
| `managed_embeddings` | Embedding lifecycle management |
| `python_sdk` | Python-style `add()`/`search()`/`get()` API |
| `agentic_memory_protocol` | MCP-optimized episodic/semantic/procedural memory |
| `llm_tools` | OpenAI-compatible function calling tool schemas |

## 📡 Streaming & Replication

| Module | Description |
|--------|-------------|
| `streaming_protocol` | Real-time vector ingestion with backpressure |
| `streaming_ingest` | CDC-style streaming ingestion pipeline |
| `realtime_streaming` | WebSocket change feed with subscriptions |
| `live_replication` | Bi-directional peer replication with delta sync |
| `sync_engine` | Vector clock-based incremental sync protocol |
| `multi_writer` | Raft-based multi-node consensus writer |
| `cdc_framework` | CDC connector framework (Kafka, Postgres, MongoDB) |

## ☁️ Infrastructure & Deployment

| Module | Description |
|--------|-------------|
| `cloud_deploy` | One-click deploy templates (Fly.io, Railway, Render) |
| `managed_cloud` | Multi-tenant cloud control plane with metering |
| `edge_serverless` | WASI runtime for Cloudflare/Deno/Vercel edge |
| `edge_runtime` | Edge-optimized runtime configuration |
| `gpu_kernels` | GPU-accelerated batch distance computation |
| `wasm_browser` | Browser WASM SDK with Web Worker protocol |
| `wasm_persistence` | IndexedDB-style chunked persistence for WASM |
| `wasm_sdk` | Core WASM SDK bindings |

## 🔌 Extensibility

| Module | Description |
|--------|-------------|
| `plugin_api` | Plugin trait API with versioned manifest format |
| `plugin_ecosystem` | Sandboxed plugin execution with permissions |
| `plugin_runtime` | Plugin lifecycle and hook management |
| `vector_pipeline` | Declarative YAML/JSON ETL pipeline engine |
| `ingestion_service` | Vector ingestion and ETL service |
| `ingestion_pipeline` | Multi-stage ingestion pipeline (experimental) |

## 📊 Observability & Compliance

| Module | Description |
|--------|-------------|
| `otel_tracing` | OpenTelemetry span instrumentation for search pipeline |
| `vector_lineage` | Source→model→version provenance tracking |
| `visual_explorer` | Visual data exploration interface |
| `compliance` | SOC2/GDPR/HIPAA audit toolkit with auto-evaluation |
| `module_audit` | Experimental module maturity classification |
| `ann_benchmark` | ann-benchmarks.com compatible benchmark harness |
| `community` | Contributor tracking and good-first-issue generator |

## 📦 Data Management

| Module | Description |
|--------|-------------|
| `text_collection` | Text-first collection wrapper |
| `multimodal_collection` | Multi-modal collection with per-modality config |
| `multimodal_service` | Multi-modal data handling service |
| `pitr_service` | Point-in-time recovery service |
| `tiered_service` | Hot/warm/cold tiered storage |
| `incremental_sync` | Incremental data synchronization |
| `time_travel_query` | Temporal query execution |
| `adaptive_optimizer` | Adaptive performance optimization |
| `adaptive_service` | Adaptive service wrapper (experimental) |
| `graphrag_service` | GraphRAG entity-relation search |

---

## Adding a New Service

1. Create `src/services/your_service.rs` following the module pattern
2. Add `pub mod your_service;` to `src/services/mod.rs`
3. Add `pub use services::your_service;` to `src/lib.rs`
4. Include `#[cfg(test)] mod tests` with unit tests
5. Update this README with the new module
