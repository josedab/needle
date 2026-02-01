# Needle Python SDK

High-level Python client for the [Needle](https://github.com/anthropics/needle) vector database — "SQLite for vectors".

## Installation

```bash
pip install needle-db          # Pure Python wrapper
pip install needle-db[native]  # With Rust-native backend (faster)
```

### Build from Source

```bash
# From the repository root
cd needle
maturin develop --features python
```

> **Prerequisites:** Rust 1.85+, [maturin](https://www.maturin.rs/) (`pip install maturin`).

## Quick Start

```python
import needle_db as needle

# In-memory (ephemeral)
client = needle.Client()

# Or file-backed (persistent)
client = needle.Client("vectors.needle")

# Create a collection
collection = client.get_or_create_collection("docs", dimensions=384)

# Insert vectors
collection.add(
    ids=["doc1", "doc2"],
    vectors=[[0.1] * 384, [0.2] * 384],
    metadatas=[{"title": "Hello"}, {"title": "World"}],
)

# Search
results = collection.query(query_vectors=[[0.1] * 384], n_results=5)
print(results.ids)       # [["doc1", "doc2"]]
print(results.distances) # [[0.0, 0.123...]]
```

### Context Manager (Auto-Save)

```python
with needle.Client("data.needle") as client:
    coll = client.get_or_create_collection("docs", dimensions=128)
    coll.add(ids=["a"], vectors=[[0.5] * 128])
    # Automatically saved on exit
```

## Backend Detection

The SDK uses a Rust-native backend when available, falling back to pure Python:

```python
import needle_db as needle
print(needle.backend())  # "rust-native" or "pure-python"
```

## Package Structure

```
python/
├── needle_db/
│   ├── __init__.py       # Client, Collection, QueryResult, MemoryStore
│   └── pyproject.toml    # Package metadata
├── needle_langchain/     # LangChain integration
├── needle_llamaindex/    # LlamaIndex integration
└── tests/                # Python SDK tests
```

## Development

```bash
# Install in development mode
pip install -e python/needle_db

# Run tests
pytest python/tests/

# Build native extension (requires Rust toolchain)
maturin develop --features python
```

## Related

- [API Reference](../docs/api-reference.md) — Full Rust and REST API documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Contributor guidelines
- [PyPI package](https://pypi.org/project/needle-db/) — Published releases
