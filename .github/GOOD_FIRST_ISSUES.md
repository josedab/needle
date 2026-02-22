# Good First Issues — Ready for GitHub

Create these 10 issues on GitHub with the `good first issue` label. Each includes title, description, and estimated effort.

---

## 1. Add doc comments to `src/services/vector_pipeline.rs`
**Labels**: `good first issue`, `documentation`
**Effort**: ~2 hours

Add `///` documentation comments to all public types and methods in the vector pipeline module. Follow patterns in `src/collection/mod.rs` for style reference.

---

## 2. Add edge case tests for `src/services/drift_monitor.rs`
**Labels**: `good first issue`, `testing`
**Effort**: ~3 hours

Add tests for: empty baseline, single-dimension vectors, identical baseline and current distributions, very large dimension counts (1000+).

---

## 3. Replace `unwrap()` with `?` in `src/services/ingestion_service.rs`
**Labels**: `good first issue`, `code quality`
**Effort**: ~2 hours

Find `unwrap()` calls in production code paths (not in `#[cfg(test)]` blocks) and replace with proper `?` error propagation using `NeedleError`.

---

## 4. Create example: RAG chatbot with semantic cache
**Labels**: `good first issue`, `examples`
**Effort**: ~4 hours

Create `examples/rag_with_cache.rs` showing how to combine `RagPipeline` with `SemanticCache` for a cost-efficient RAG pipeline.

---

## 5. Add `Display` impl for `DistanceFunction` enum
**Labels**: `good first issue`, `enhancement`
**Effort**: ~1 hour

Add `std::fmt::Display` implementation for `DistanceFunction` in `src/distance.rs` so it can be printed in logs and error messages.

---

## 6. Add CLI command: `needle validate`
**Labels**: `good first issue`, `cli`
**Effort**: ~4 hours

Add a `validate` subcommand that checks a `.needle` file's header integrity using the `format_spec` module. Report magic number, version, checksums.

---

## 7. Add metadata filter benchmark
**Labels**: `good first issue`, `performance`
**Effort**: ~3 hours

Add a benchmark in `benches/benchmarks.rs` measuring filtered search performance at different selectivity levels (1%, 10%, 50%).

---

## 8. Document the WebSocket change feed protocol
**Labels**: `good first issue`, `documentation`
**Effort**: ~2 hours

Create `docs/websocket-protocol.md` documenting the `ws_protocol` module's message types, handshake flow, and subscription management.

---

## 9. Add `--format json` flag to `needle info` CLI command
**Labels**: `good first issue`, `cli`, `enhancement`
**Effort**: ~2 hours

Add a `--format` flag to the `info` subcommand that outputs database info as JSON instead of human-readable text.

---

## 10. Add Python example for hybrid search
**Labels**: `good first issue`, `examples`, `python`
**Effort**: ~3 hours

Create `python/examples/hybrid_search.py` showing BM25 + vector hybrid search with the Python SDK (requires `hybrid` feature).

---

## How to Create These Issues

```bash
# Using GitHub CLI (gh):
gh issue create --title "Add doc comments to vector_pipeline.rs" \
  --label "good first issue" --label "documentation" \
  --body "Add /// documentation comments to all public types..."
```

Or create them manually at: https://github.com/anthropics/needle/issues/new
