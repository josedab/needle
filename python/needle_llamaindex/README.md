# Needle LlamaIndex Integration

> **⚠️ Reference Implementation — Not Production-Ready**
>
> This package is a **reference/in-memory stub** that stores vectors in plain
> Python dicts. It is intended for prototyping, testing, and as a starting point
> for a full Needle-backed LlamaIndex integration. **Do not use it in production
> workloads** — data is not persisted to a Needle database and will be lost when
> the process exits.

This package provides a LlamaIndex vector store wrapper for Needle.

## Install (from source)

```bash
cd python/needle_llamaindex
pip install -e ".[llamaindex]"
```

## Quick Start

```python
from needle_llamaindex import NeedleVectorStoreIndex, TextNode

index = NeedleVectorStoreIndex(collection_name="documents", dimensions=384)

nodes = [TextNode(text="Hello world", embedding=[0.1] * 384, metadata={"source": "demo"})]
index.add(nodes)

results = index.query(query_embedding=[0.1] * 384, similarity_top_k=3)
for r in results:
    print(f"{r.node.text} (score: {r.score:.4f})")
```

## Configuration

`NeedleIndexConfig` controls collection behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | `"llamaindex"` | Name of the Needle collection |
| `dimensions` | `int` | `384` | Vector dimensionality |
| `distance` | `str` | `"cosine"` | Distance function (`"cosine"`, `"euclidean"`, `"dot_product"`) |

```python
index = NeedleVectorStoreIndex(
    collection_name="my_docs",
    dimensions=768,
    distance="euclidean",
)
```

## Advanced Usage

### DocumentChunker

Split long texts into overlapping `TextNode` chunks:

```python
from needle_llamaindex import DocumentChunker

chunker = DocumentChunker(chunk_size=512, overlap=50)
nodes = chunker.chunk("A very long document text...", doc_id="doc1")
# Returns: [TextNode(id_="doc1_0", ...), TextNode(id_="doc1_1", ...)]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` | `512` | Maximum characters per chunk |
| `overlap` | `int` | `50` | Overlapping characters between consecutive chunks |

### Retrieve and Delete

```python
# Get a node by ID
node = index.get_by_id("node-abc")
print(node.text)

# Check index size
print(index.count)  # 1

# Delete a node
index.delete(ref_doc_id="node-abc")
```

### Score Conversion

Distances are converted to similarity scores: `score = 1.0 - distance`. Results are sorted highest-score-first, so a score of `1.0` means an exact match and lower scores indicate less similarity.

## Filtering

Pass a `filters` dict to `query()` for exact-match metadata filtering:

```python
results = index.query(
    query_embedding=[0.1] * 384,
    similarity_top_k=10,
    filters={"source": "demo"},
)
```

Filter values are compared with direct equality (`meta[key] == value`). For MongoDB-style operators (`$eq`, `$ne`, etc.), use the [Needle Python SDK](../README.md) directly.

## Error Handling

```python
from needle_llamaindex import NeedleVectorStoreIndex, TextNode

index = NeedleVectorStoreIndex(dimensions=384)

# add() returns the list of node IDs
ids = index.add([TextNode(text="hello", embedding=[0.1] * 384)])

# get_by_id returns None for missing nodes
node = index.get_by_id("nonexistent")  # None

# delete is a no-op for missing IDs (no error raised)
index.delete(ref_doc_id="nonexistent")
```

## Testing

```bash
cd python/needle_llamaindex
pip install -e ".[llamaindex]"
python -m pytest -xvs
```

If no tests exist yet, you can verify the package loads correctly:

```bash
python -c "from needle_llamaindex import NeedleVectorStoreIndex; print('import ok')"
```

## Contributing

Contributions are welcome! This is a reference implementation and a great place to start contributing to Needle.

1. Fork the repository and create a feature branch
2. Follow the existing code style (type hints, docstrings)
3. Add or update tests for any new functionality
4. Submit a pull request against `main`

See the [project-level CONTRIBUTING.md](../../CONTRIBUTING.md) for general guidelines.

## Related

- [Needle Python SDK](../README.md) — Core Python client
- [API Reference](../../docs/api-reference.md) — Full Rust and REST API documentation
