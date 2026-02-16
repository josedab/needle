# Needle Services Directory

This directory contains 100+ high-level service modules organized by domain into subdirectories.

## Directory Structure

| Directory | Description | Key Modules |
|-----------|-------------|-------------|
| `ai/` | AI/LLM integration, caching, knowledge graphs | `semantic_cache`, `rag_sdk`, `llm_tools`, `graphrag_service` |
| `client/` | Client SDKs and protocol bindings | `client_sdk`, `python_sdk`, `grpc_schema`, `ws_protocol` |
| `collection/` | Collection lifecycle and management | `text_collection`, `pitr_service`, `collection_rbac` |
| `compute/` | Optimization, GPU, and transactions | `adaptive_optimizer`, `gpu_kernels`, `vector_transactions` |
| `embedding/` | Embedding generation and model management | `inference_engine`, `text_to_vector`, `embedding_router` |
| `governance/` | Compliance, auditing, and API stability | `api_stability`, `compliance`, `module_audit` |
| `infrastructure/` | Deployment, scaling, and cloud management | `cloud_service`, `cluster_bootstrap`, `tenant_router` |
| `observability/` | Monitoring, benchmarking, and diagnostics | `drift_monitor`, `benchmark_suite`, `vector_lineage` |
| `pipeline/` | Data ingestion and streaming pipelines | `ingestion_service`, `streaming_ingest`, `cdc_framework` |
| `plugin/` | Plugin system and WASM runtimes | `plugin_api`, `plugin_ecosystem`, `wasm_plugin_runtime` |
| `search/` | Query optimization, caching, and NeedleQL | `query_optimizer`, `needleql_executor`, `encrypted_search` |
| `storage/` | Storage backends, snapshots, and compaction | `tiered_service`, `snapshot_manager`, `hnsw_compactor` |
| `sync/` | Replication, sync, and federation | `sync_engine`, `live_replication`, `distributed_federation` |

## Maturity Matrix

The table below shows the maturity status of each service domain. **All 14 domains currently carry `#[allow(clippy::unwrap_used)]` overrides** and should be treated as scaffolding until those are removed.

| Domain | Files | Tests | Status | Notes |
|--------|------:|------:|--------|-------|
| `ai/` | 10 | 9 | Scaffold | LLM integration, semantic cache, GraphRAG; heavy `unwrap()` usage |
| `client/` | 8 | 7 | Scaffold | SDK stubs for Python, gRPC, WebSocket |
| `collection/` | 11 | 10 | Scaffold | Text collections, PITR, RBAC; some stable modules (see below) |
| `compute/` | 7 | 6 | Scaffold | Adaptive optimizer, GPU kernels, vector transactions |
| `embedding/` | 10 | 9 | Scaffold | Inference engine, embedding router |
| `governance/` | 9 | 8 | Scaffold | Compliance, audit tooling, API stability |
| `infrastructure/` | 11 | 10 | Scaffold | Cloud service, cluster bootstrap, tenant router |
| `observability/` | 9 | 8 | Scaffold | Drift monitor, benchmarks, vector lineage |
| `pipeline/` | 9 | 8 | Scaffold | Ingestion service, streaming ingest, CDC framework |
| `plugin/` | 8 | 7 | Scaffold | Plugin API, WASM runtime |
| `search/` | 7 | 6 | Scaffold | Query optimizer, NeedleQL, encrypted search |
| `storage/` | 7 | 6 | Scaffold | Tiered storage, snapshot manager, HNSW compactor |
| `sync/` | 9 | 8 | Scaffold | Sync engine, live replication, federation |
| **Total** | **125** | **112** | | |

**Status definitions:**
- **Stable** — Production-ready, no `#[allow(clippy::unwrap_used)]`, full test coverage
- **Scaffold** — Structure in place, has `#[allow(clippy::unwrap_used)]` override, needs error-handling cleanup
- **Draft** — Experimental API, subject to breaking changes

## Stable vs Experimental

Most service modules are gated behind `#[cfg(feature = "experimental")]`. The stable services (available with default features) are:

- `pipeline/ingestion_service` — Vector ingestion workflows
- `collection/multimodal_service` — Multi-modal collection management
- `collection/pitr_service` — Point-in-time recovery
- `collection/text_collection` — Text-first collection API
- `storage/tiered_service` — Tiered storage management

## Backward Compatibility

All modules are re-exported at the `services::` level, so existing imports like `crate::services::inference_engine` continue to work unchanged.

## Adding a New Service

1. Create `src/services/<domain>/your_service.rs` in the appropriate domain subdirectory
2. Add `pub mod your_service;` to `src/services/<domain>/mod.rs`
3. Add `pub use <domain>::your_service;` to `src/services/mod.rs` for backward compatibility
4. Add `pub use services::your_service;` to `src/lib.rs` if it's part of the public API
5. Include `#[cfg(test)] mod tests` with unit tests
