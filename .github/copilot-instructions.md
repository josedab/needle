# Copilot Instructions — Needle Vector Database

## Project Overview
Needle is an embedded vector database written in Rust ("SQLite for vectors"). It provides HNSW-based approximate nearest neighbor search with single-file storage and zero configuration.

## Architecture
- **Core**: `src/collection/`, `src/database/`, `src/indexing/`, `src/distance.rs`, `src/error.rs`, `src/storage.rs`, `src/metadata.rs`
- **API layers**: `src/server.rs` (HTTP REST), `src/main.rs` (CLI), `src/python.rs`, `src/wasm.rs`
- **Enterprise**: `src/enterprise/` (raft, encryption, RBAC, namespaces)
- **Experimental**: `src/experimental/` (APIs may change without notice)

## Code Conventions
- Use `Result<T>` (alias for `std::result::Result<T, NeedleError>`) for fallible operations
- Use `?` for error propagation — no `unwrap()` in production code (tests only)
- `#![deny(unsafe_code)]` is set crate-wide; only `distance.rs`, `storage.rs`, `experimental/gpu.rs`, `experimental/zero_copy.rs` are exempted
- `clippy::pedantic` is enabled workspace-wide via `[workspace.lints.clippy]`
- Builders use `#[must_use]` on methods returning `Self`
- Thread safety: `Database` uses `parking_lot::RwLock`; `CollectionRef` is the concurrent API

## Adding Features
- **New Collection method**: Add to `Collection` in `src/collection/mod.rs`, add internal method to `Database`, expose via `CollectionRef`
- **New CLI command**: Add variant to `Commands` enum in `src/main.rs`, add match arm
- **New REST endpoint**: Add handler in `src/server.rs`, register in `create_router()`
- **New error type**: Add variant to `NeedleError` in `src/error.rs` with `ErrorCode`

## Build & Test
```bash
cargo build                    # Default features
cargo build --features full    # All stable features
cargo test --lib               # Unit tests (fast, 1334 tests)
cargo test --features full     # Full test suite
make quick                     # Format + lint + unit tests
make check                     # Format + lint + all tests
```

## Feature Flags
- `server`: HTTP REST API (Axum)
- `hybrid`: BM25 + vector search
- `encryption`: ChaCha20-Poly1305
- `full`: All stable features
- `experimental`: Unstable modules
