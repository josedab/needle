# Needle LangChain Integration

> **⚠️ Reference Implementation — Not Production-Ready**
>
> This package is a **reference/in-memory stub** that stores vectors in plain
> Python dicts. It is intended for prototyping, testing, and as a starting point
> for a full Needle-backed LangChain integration. **Do not use it in production
> workloads** — data is not persisted to a Needle database and will be lost when
> the process exits.

This package provides a LangChain vector store wrapper for Needle.

## Install (from source)

```bash
cd python/needle_langchain
pip install -e ".[langchain]"
```

## Quick Start

```python
from needle_langchain import NeedleVectorStore

store = NeedleVectorStore(collection_name="documents", dimensions=384)
store.add_texts(["hello world"], metadatas=[{"source": "demo"}])
results = store.similarity_search(query_vector=[0.1] * 384, k=3)
```

## Configuration

`NeedleVectorStoreConfig` controls collection behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | `"langchain"` | Name of the Needle collection |
| `dimensions` | `int` | `384` | Vector dimensionality |
| `distance` | `str` | `"cosine"` | Distance function (`"cosine"`, `"euclidean"`, `"dot_product"`) |
| `content_key` | `str` | `"page_content"` | Metadata key used to store document text |
| `metadata_key` | `str` | `"metadata"` | Metadata key for document metadata |

```python
store = NeedleVectorStore(
    collection_name="my_docs",
    dimensions=768,
    distance="euclidean",
    content_key="text",
)
```

## Advanced Usage

### Adding Pre-Computed Vectors

Use `add_vectors()` when you already have embeddings:

```python
from needle_langchain import NeedleVectorStore, Document

store = NeedleVectorStore(dimensions=384)

docs = [Document(page_content="Machine learning basics", metadata={"topic": "ml"})]
vectors = [[0.1] * 384]

ids = store.add_vectors(vectors=vectors, documents=docs, ids=["doc1"])
```

### Search with Scores

`similarity_search_with_score()` returns `(Document, float)` tuples sorted by distance (lower is more similar):

```python
results = store.similarity_search_with_score(query_vector=[0.1] * 384, k=5)
for doc, score in results:
    print(f"{doc.page_content} (distance: {score:.4f})")
```

### Retrieve and Delete

```python
# Get a document by ID
doc = store.get_by_id("doc1")
print(doc.page_content)

# Check collection size
print(store.count)  # 1

# Delete documents
store.delete(ids=["doc1"])
```

## Filtering

Pass a `filter` dict to `similarity_search_with_score()` using MongoDB-style operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equal to | `{"status": {"$eq": "active"}}` |
| `$ne` | Not equal to | `{"status": {"$ne": "draft"}}` |
| (implicit) | Direct value match | `{"status": "active"}` |

```python
results = store.similarity_search_with_score(
    query_vector=[0.1] * 384,
    k=10,
    filter={"topic": {"$eq": "ml"}},
)
```

## Error Handling

```python
from needle_langchain import NeedleVectorStore

store = NeedleVectorStore(dimensions=384)

# add_texts / add_vectors return the assigned IDs
ids = store.add_texts(["hello"], metadatas=[{"k": "v"}])

# get_by_id returns None for missing documents
doc = store.get_by_id("nonexistent")  # None

# delete returns True if at least one ID was found
deleted = store.delete(ids=["nonexistent"])  # False
```

## Testing

```bash
cd python/needle_langchain
pip install -e ".[langchain]"
python -m pytest -xvs
```

If no tests exist yet, you can verify the package loads correctly:

```bash
python -c "from needle_langchain import NeedleVectorStore; print('import ok')"
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
