# Workspace Crates

This directory contains the workspace crates that compose the Needle vector database.

| Crate | Description |
|-------|-------------|
| **needle-core** | Core library — vector storage, HNSW indexing, metadata filtering, hybrid search, and the public Rust API. All other crates depend on this. |
| **needle-cli** | Command-line interface — wraps `needle-core` with a `clap`-based CLI for database management, search, import/export, and server mode. Produces the `needle` binary. |
| **needle-python** | Python bindings — `pyo3` cdylib wrapper around `needle-core`, published as the `needle-db` PyPI package. |

## Dependency Graph

```
needle-cli ──► needle-core
needle-python ──► needle-core
```

The root `Cargo.toml` defines the workspace and shared settings (version, edition, lints).
