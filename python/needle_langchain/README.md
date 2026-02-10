# Needle LangChain Integration

This package provides a LangChain vector store wrapper for Needle.

## Install (from source)

```bash
cd python/needle_langchain
pip install -e ".[langchain]"
```

## Usage

```python
from needle_langchain import NeedleVectorStore

store = NeedleVectorStore(collection_name="documents", dimensions=384)
store.add_texts(["hello world"], metadatas=[{"source": "demo"}])
results = store.similarity_search("hello", k=3)
```
