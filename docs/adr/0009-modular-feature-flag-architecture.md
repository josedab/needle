# ADR-0009: Modular Feature-Flag Architecture

## Status

Accepted

## Context

Needle targets multiple deployment scenarios with vastly different requirements:

| Deployment | Needs | Doesn't Need |
|------------|-------|--------------|
| Embedded library | Core search | HTTP server, metrics |
| CLI tool | Core + file I/O | Server, Python bindings |
| HTTP microservice | Core + server + metrics | Python/WASM bindings |
| Python application | Core + PyO3 bindings | Server, WASM |
| Browser/Node.js | Core + WASM | Server, Python, native I/O |
| Mobile (iOS/Kotlin) | Core + UniFFI | Everything else |

Shipping a single binary with all capabilities would:
1. **Bloat binary size** — HTTP server adds ~1.5MB, ONNX embeddings adds ~10MB
2. **Increase compile time** — Full build takes significantly longer
3. **Add unnecessary dependencies** — Embedded users don't need tokio/axum
4. **Create security surface** — Unused features are still attack vectors
5. **Complicate licensing** — Some dependencies have different licenses

Alternative approaches considered:

| Approach | Pros | Cons |
|----------|------|------|
| Monolithic crate | Simple versioning | Bloat, slow builds |
| Workspace of crates | Clean separation | Complex versioning, circular deps |
| Feature flags | Single crate, selective builds | Feature interaction complexity |
| Runtime plugins | Maximum flexibility | Performance overhead, complexity |

## Decision

Use **Cargo feature flags** to conditionally compile orthogonal functionality into a single crate.

### Feature Hierarchy

```toml
[features]
default = []

# Performance optimization
simd = []  # SIMD-optimized distance functions

# Async runtime support
async = ["tokio", "futures"]

# HTTP REST API server
server = ["async", "axum", "tower", "tower-http", "hyper"]

# Prometheus metrics
metrics = ["prometheus"]

# BM25 hybrid search
hybrid = ["rust-stemmers"]

# ONNX embedding inference
embeddings = ["ort", "ndarray", "tokenizers"]

# Convenience: all stable features
full = ["server", "metrics", "hybrid"]

# Language bindings (mutually exclusive deployment targets)
python = ["pyo3", "pythonize"]
wasm = ["wasm-bindgen", "js-sys", "web-sys", "getrandom/js"]
uniffi-bindings = ["uniffi"]
```

### Dependency Gating

```toml
[dependencies]
# Always included (core functionality)
thiserror = "2.0"
serde = { version = "1.0", features = ["derive"] }
parking_lot = "0.12"

# Optional dependencies gated by features
tokio = { version = "1.0", features = ["rt-multi-thread", "sync"], optional = true }
axum = { version = "0.8", optional = true }
prometheus = { version = "0.13", optional = true }
pyo3 = { version = "0.23", features = ["extension-module"], optional = true }
```

### Conditional Compilation

```rust
// src/lib.rs
pub mod collection;
pub mod database;
pub mod hnsw;
pub mod error;

#[cfg(feature = "async")]
pub mod async_api;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "metrics")]
pub mod metrics;

#[cfg(feature = "hybrid")]
pub mod hybrid;

#[cfg(feature = "embeddings")]
pub mod embeddings;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "wasm")]
mod wasm;
```

### Code References

- `Cargo.toml:126-142` — Feature definitions
- `src/lib.rs:114-154` — Conditional module exports
- Feature dependency chain documented in Cargo.toml

## Consequences

### Benefits

1. **Minimal default footprint** — Core library is small and fast to compile
2. **User-controlled dependencies** — Only pull what you need
3. **Single crate versioning** — All features version together
4. **Clean public API** — Unused features don't appear in docs
5. **CI optimization** — Test features independently or together
6. **Deployment flexibility** — Same code, different builds

### Tradeoffs

1. **Feature interaction testing** — Must test feature combinations
2. **Documentation complexity** — Feature-gated APIs need clear docs
3. **Conditional compilation** — `#[cfg]` attributes throughout codebase
4. **No runtime selection** — Features fixed at compile time

### Feature Dependency Graph

```
                    ┌─────────┐
                    │  full   │
                    └────┬────┘
           ┌────────────┼────────────┐
           ▼            ▼            ▼
      ┌────────┐   ┌─────────┐   ┌────────┐
      │ server │   │ metrics │   │ hybrid │
      └────┬───┘   └─────────┘   └────────┘
           ▼
      ┌────────┐
      │ async  │
      └────────┘
           │
           ▼
    ┌──────────────┐
    │ tokio/futures│
    └──────────────┘
```

### Binary Size Impact

| Feature Set | Dependencies Added | Approximate Size |
|-------------|-------------------|------------------|
| default | Core only | ~500KB |
| simd | None (compile flags) | ~500KB |
| async | tokio, futures | +800KB |
| server | + axum, tower, hyper | +1.5MB |
| metrics | + prometheus | +200KB |
| hybrid | + rust-stemmers | +100KB |
| embeddings | + ort, ndarray | +10MB |
| full | server + metrics + hybrid | ~3MB |
| python | + pyo3 | +500KB |
| wasm | + wasm-bindgen | +300KB |

### What This Enabled

- `cargo build` for minimal embedded use
- `cargo build --features server` for HTTP microservice
- `cargo build --features python` for Python wheels
- `cargo build --features full` for feature-complete deployment
- CI matrix testing feature combinations

### What This Prevented

- Runtime feature selection (would need trait objects or enums)
- Dynamic plugin loading (would need libloading)
- Single "batteries included" binary (users choose their batteries)

### Usage Examples

**Embedded library:**
```toml
[dependencies]
needle = "0.1"  # Minimal, no server or bindings
```

**HTTP microservice:**
```toml
[dependencies]
needle = { version = "0.1", features = ["server", "metrics"] }
```

**Python package build:**
```bash
maturin build --features python
```

**Full-featured CLI:**
```bash
cargo install needle --features full
```

### Testing Strategy

```yaml
# CI matrix tests feature combinations
jobs:
  test:
    strategy:
      matrix:
        features:
          - ""  # default
          - "async"
          - "server"
          - "metrics"
          - "hybrid"
          - "full"
          - "simd"
    steps:
      - run: cargo test --features ${{ matrix.features }}
```

### Design Principles

1. **Additive features** — Features add capabilities, never remove
2. **Orthogonal when possible** — metrics doesn't require server
3. **Explicit dependencies** — Feature A requiring B is documented
4. **Stable core** — Core API doesn't change based on features
5. **No default features** — Users explicitly opt-in to extras
