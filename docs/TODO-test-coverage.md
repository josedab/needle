# Test Coverage Tracking

Consolidated from `// TODO: Add tests for this module` comments across the codebase.

## Modules Needing Tests

| Module | File | Priority |
|--------|------|----------|
| WASM bindings | `src/wasm.rs` | Medium |
| Beta API | `src/beta_api.rs` | Low |
| Experimental API | `src/experimental_api.rs` | Low |
| Streaming PubSub | `src/streaming/pubsub.rs` | Medium |
| Streaming Event Log | `src/streaming/event_log.rs` | Medium |
| Streaming Manager | `src/streaming/stream_manager.rs` | Medium |
| Federated Config | `src/search/federated/config.rs` | High |
| Federated Health | `src/search/federated/health.rs` | High |
| NeedleQL Parser | `src/search/query_lang/parser.rs` | High |
| NeedleQL Lexer | `src/search/query_lang/lexer.rs` | High |
| NeedleQL AST | `src/search/query_lang/ast.rs` | Medium |
| NeedleQL Optimizer | `src/search/query_lang/optimizer.rs` | Medium |
| NeedleQL Session | `src/search/query_lang/session.rs` | Medium |
| NeedleQL Executor | `src/search/query_lang/executor.rs` | High |
| Framework Common | `src/integrations/framework_common.rs` | Low |

## Other Code TODOs

| Location | Note |
|----------|------|
| `src/services/pipeline/streaming_ingest.rs:1065` | `self.db.save()` requires mutable ref — needs API change |
