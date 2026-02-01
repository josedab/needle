# Python SDK

> **Canonical documentation**: [python/README.md](../python/README.md) — the Python SDK README is the most complete and up-to-date reference.

## Installation

```bash
pip install needle-db          # Pure Python wrapper
pip install needle-db[native]  # With Rust-native backend (faster)
```

### Build from Source

Requirements: Rust 1.85+, Python 3.8+, [maturin](https://www.maturin.rs/)

```bash
pip install maturin
maturin develop --features python
python -c "import needle; print('needle import ok')"
```

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

### Backend Detection

```python
import needle_db as needle
print(needle.backend())  # "rust-native" or "pure-python"
```

## Troubleshooting

- If the build fails, ensure `rustc --version` reports 1.85+.
- On macOS, you may need the Xcode Command Line Tools (`xcode-select --install`).

## Related

- [Python SDK README](../python/README.md) — Full documentation with package structure and development setup
- [API Reference](api-reference.md) — Rust and REST API documentation
- [RAG Quickstart](rag-quickstart.md) — End-to-end RAG pipeline with Python
