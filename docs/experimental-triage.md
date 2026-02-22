# Experimental Module Triage Report

Generated from `triage_report` service module analysis.

## Current State

- **31 modules** in `src/experimental/`
- **40,390 LOC** (22% of total codebase)
- **Target**: <15,000 LOC

## ✅ Promote to Stable (5 modules)

These modules have sufficient test coverage and fill real use cases:

- `clustering` — K-means clustering for IVF and data analysis
- `temporal` — Temporal/versioned vector queries
- `crdt` — Conflict-free replicated data types for sync
- `knowledge_graph` — Knowledge graph construction (1,721 LOC)
- `adaptive_index` — Self-tuning index parameters

**Action**: Move to `src/indexing/` or `src/search/` as appropriate.

## 🔨 Improve — Needs Work (5 modules)

These have potential but need more tests or API cleanup:

- `edge_runtime` (2,204 LOC) — Reduce scope, focus on core edge config
- `agentic_memory` (1,801 LOC) — Overlap with `services/agentic_memory_protocol`
- `distributed_hnsw` — Needs integration tests
- `adaptive_runtime` — Clarify scope vs `adaptive_index`
- `learned_tuning` — Needs benchmark validation

## 📦 Archive as Reference (10 modules)

Move to `examples/experimental/` or a separate `needle-experimental` crate:

- `vector_streaming` (2,930 LOC) — Superseded by `services/realtime_streaming`
- `cloud_control` (2,934 LOC) — Superseded by `services/managed_cloud`
- `edge_optimized` — Overlaps with `services/edge_serverless`
- `edge_partitioning` — Overlaps with `services/cluster_bootstrap`
- `serverless_runtime` — Overlaps with `services/edge_serverless`
- `playground` — Move to standalone tool
- `python_arrow` — Move to `crates/needle-python`
- `platform_adapters` — Low usage
- `graphrag_index` — Superseded by `services/graph_knowledge_service`
- `graph` — Basic graph ops, superseded by `services/graph_query`

## 🗑️ Delete — Pure Scaffolding (11 modules)

These provide minimal value and inflate the codebase:

- `gpu` (3,653 LOC) — CPU fallback only, no real GPU code
- `zero_copy` — Experimental optimization, needs unsafe
- `dedup` — Simple hash dedup, trivial to reimplement
- `rebalance` — Placeholder logic
- `analytics` — Basic counters, superseded by `observe/`
- `streaming_upsert` — Superseded by `services/streaming_protocol`
- `optimizer` — Superseded by `services/query_optimizer`
- `llm_cache` — Superseded by `services/semantic_cache`
- `plugin` — Superseded by `services/plugin_api`
- `plugin_registry` — Superseded by `services/plugin_ecosystem`
- `cloud_control` — Superseded by `services/managed_cloud`

## Impact Summary

| Action | Modules | LOC Removed | LOC Remaining |
|--------|---------|-------------|---------------|
| Promote | 5 | 0 (moved) | — |
| Improve | 5 | 0 | — |
| Archive | 10 | ~15,000 | — |
| Delete | 11 | ~20,000 | — |
| **Total** | **31** | **~35,000** | **~5,000** |

**Projected result**: experimental/ drops from 40,390 LOC to ~5,000 LOC (87% reduction).

## Execution Order

1. **Delete** 11 scaffolding modules (quick, safe — all functionality exists in services/)
2. **Archive** 10 modules to reference location
3. **Promote** 5 modules to stable paths
4. **Improve** 5 modules over next release cycle

## Risk Mitigation

- Run `cargo test --features experimental` before and after each step
- Keep git history for all deleted/archived code
- Update `src/experimental/mod.rs` and `src/lib.rs` re-exports
- Add deprecation notices in CHANGELOG.md
