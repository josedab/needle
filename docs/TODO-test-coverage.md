# Test Coverage Tracking

Consolidated from `// TODO: Add tests for this module` comments across the codebase.

## Completed Modules

| Module | File | Tests Added |
|--------|------|-------------|
| Federated Config | `src/search/federated/config.rs` | 11 tests: Default, builders, serialization, enum variants |
| Federated Health | `src/search/federated/health.rs` | 11 tests: Health aggregation, history cap, threshold detection |
| NeedleQL Executor | `src/search/query_lang/executor.rs` | 30 tests: Context, resolve, filter building, execute, serialization |
| NeedleQL Session | `src/search/query_lang/session.rs` | 28 tests: Aggregation parsing/apply, session lifecycle, help text |
| Framework Common | `src/integrations/framework_common.rs` | 12 tests: distance_to_score for all 7 distance functions, FrameworkCollection |
| WASM Bindings | `src/wasm.rs` | 14 tests: base64 encode/decode, React hooks, IndexedDB/SW helpers |
| Beta API | `src/beta_api.rs` | 3 tests: Re-export accessibility verification |
| Experimental API | `src/experimental_api.rs` | 4 tests: Re-export accessibility verification |

## Modules Needing Tests

| Module | File | Priority |
|--------|------|----------|
| Streaming PubSub | `src/streaming/pubsub.rs` | Medium |
| Streaming Event Log | `src/streaming/event_log.rs` | Medium |
| Streaming Manager | `src/streaming/stream_manager.rs` | Medium |
| NeedleQL Parser | `src/search/query_lang/parser.rs` | High |
| NeedleQL Lexer | `src/search/query_lang/lexer.rs` | High |
| NeedleQL AST | `src/search/query_lang/ast.rs` | Medium |
| NeedleQL Optimizer | `src/search/query_lang/optimizer.rs` | Medium |

## Other Code TODOs

| Location | Note |
|----------|------|
| `src/services/pipeline/streaming_ingest.rs:1065` | `self.db.save()` requires mutable ref — needs API change |
