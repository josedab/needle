# Needle LlamaIndex Integration

This package provides a LlamaIndex vector store wrapper for Needle.

## Install (from source)

```bash
cd python/needle_llamaindex
pip install -e ".[llamaindex]"
```

## Usage

```python
from needle_llamaindex import NeedleVectorStore

store = NeedleVectorStore(collection_name="documents", dimensions=384)
store.add(["hello world"], metadatas=[{"source": "demo"}])
results = store.query("hello", top_k=3)
```
