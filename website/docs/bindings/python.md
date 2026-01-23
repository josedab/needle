---
sidebar_position: 2
---

# Python

Needle provides Python bindings through PyO3, giving you native performance with Pythonic ergonomics.

## Installation

```bash
pip install needle-db
```

For the latest development version:

```bash
pip install git+https://github.com/anthropics/needle.git#subdirectory=bindings/python
```

## Quick Start

```python
from needle import Database, DistanceFunction

# Create or open a database
db = Database.open("vectors.needle")

# Create a collection
db.create_collection("documents", dimensions=384, distance=DistanceFunction.Cosine)

# Get collection
collection = db.collection("documents")

# Insert vectors
collection.insert("doc1", [0.1] * 384, {"title": "Hello World"})

# Search
results = collection.search([0.1] * 384, k=10)
for result in results:
    print(f"ID: {result.id}, Distance: {result.distance}")

# Save
db.save()
```

## Database API

### Opening Databases

```python
from needle import Database, DatabaseConfig

# File-based database
db = Database.open("vectors.needle")

# In-memory database
db = Database.in_memory()

# With configuration
config = DatabaseConfig(mmap_threshold=10 * 1024 * 1024)
db = Database.open("vectors.needle", config=config)
```

### Collection Management

```python
from needle import CollectionConfig, DistanceFunction

# Create collection
db.create_collection("docs", dimensions=384, distance=DistanceFunction.Cosine)

# With custom config
config = CollectionConfig(
    dimensions=384,
    distance=DistanceFunction.Cosine,
    hnsw_m=32,
    hnsw_ef_construction=400
)
db.create_collection_with_config("high_quality", config)

# List collections
names = db.list_collections()

# Check existence
if db.collection_exists("docs"):
    collection = db.collection("docs")

# Delete collection
db.delete_collection("old_collection")
```

## Collection API

### Vector Operations

```python
collection = db.collection("documents")

# Insert
collection.insert("doc1", embedding, {"title": "Hello"})

# Get by ID
entry = collection.get("doc1")
print(entry.vector)
print(entry.metadata)

# Check existence
if collection.exists("doc1"):
    print("Found!")

# Delete
collection.delete("doc1")

# Count
count = collection.count()

# Clear all
collection.clear()
```

### Searching

```python
# Basic search
results = collection.search(query_vector, k=10)

# With filter
results = collection.search(
    query_vector,
    k=10,
    filter={"category": "programming"}
)

# With custom ef_search
results = collection.search(
    query_vector,
    k=10,
    ef_search=100
)

# Batch search
queries = [query1, query2, query3]
all_results = collection.batch_search(queries, k=10)
```

### Search Results

```python
results = collection.search(query_vector, k=10)

for result in results:
    print(f"ID: {result.id}")
    print(f"Distance: {result.distance}")
    print(f"Metadata: {result.metadata}")
```

## Filtering

```python
# Equality
results = collection.search(query, k=10, filter={"status": "active"})

# Comparison
results = collection.search(query, k=10, filter={
    "price": {"$gt": 10, "$lt": 100}
})

# In array
results = collection.search(query, k=10, filter={
    "category": {"$in": ["books", "movies"]}
})

# Logical operators
results = collection.search(query, k=10, filter={
    "$or": [
        {"category": "electronics"},
        {"price": {"$lt": 50}}
    ]
})

# Complex filter
results = collection.search(query, k=10, filter={
    "$and": [
        {"status": "active"},
        {"$or": [
            {"category": "books"},
            {"rating": {"$gte": 4.0}}
        ]}
    ]
})
```

## NumPy Integration

```python
import numpy as np
from needle import Database, DistanceFunction

db = Database.open("vectors.needle")
db.create_collection("embeddings", 384, DistanceFunction.Cosine)
collection = db.collection("embeddings")

# Insert numpy arrays directly
embedding = np.random.randn(384).astype(np.float32)
collection.insert("doc1", embedding, {"title": "Test"})

# Search with numpy arrays
query = np.random.randn(384).astype(np.float32)
results = collection.search(query, k=10)

# Batch insert
embeddings = np.random.randn(100, 384).astype(np.float32)
for i, emb in enumerate(embeddings):
    collection.insert(f"doc{i}", emb, {"index": i})
```

## Sentence Transformers Integration

```python
from sentence_transformers import SentenceTransformer
from needle import Database, DistanceFunction

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create database
db = Database.open("semantic_search.needle")
db.create_collection("documents", 384, DistanceFunction.Cosine)
collection = db.collection("documents")

# Index documents
documents = [
    {"id": "doc1", "text": "Introduction to machine learning"},
    {"id": "doc2", "text": "Deep learning with Python"},
    {"id": "doc3", "text": "Natural language processing basics"},
]

for doc in documents:
    embedding = model.encode(doc["text"])
    collection.insert(doc["id"], embedding, {"text": doc["text"]})

db.save()

# Search
query = "ML tutorial"
query_embedding = model.encode(query)
results = collection.search(query_embedding, k=3)

for result in results:
    print(f"{result.id}: {result.metadata['text']}")
```

## OpenAI Integration

```python
import openai
from needle import Database, DistanceFunction

# Setup
client = openai.OpenAI()
db = Database.open("openai_search.needle")
db.create_collection("documents", 1536, DistanceFunction.Cosine)
collection = db.collection("documents")

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Index
documents = ["Document 1 content...", "Document 2 content..."]
for i, doc in enumerate(documents):
    embedding = get_embedding(doc)
    collection.insert(f"doc{i}", embedding, {"content": doc})

db.save()

# Search
query = "search query"
query_embedding = get_embedding(query)
results = collection.search(query_embedding, k=5)
```

## Quantization

```python
from needle import CollectionConfig, DistanceFunction, QuantizationType

# Scalar quantization
config = CollectionConfig(
    dimensions=384,
    distance=DistanceFunction.Cosine,
    quantization=QuantizationType.Scalar
)
db.create_collection_with_config("quantized", config)

# Product quantization
config = CollectionConfig(
    dimensions=384,
    distance=DistanceFunction.Cosine,
    quantization=QuantizationType.Product(
        num_subvectors=48,
        num_centroids=256
    )
)
db.create_collection_with_config("pq_quantized", config)
```

## Hybrid Search

```python
from needle import Database, Bm25Index, reciprocal_rank_fusion

db = Database.open("hybrid.needle")
collection = db.collection("documents")

# Create BM25 index
bm25 = Bm25Index()

# Index documents for BM25
for id, metadata in collection.iter_metadata():
    bm25.index_document(id, metadata.get("text", ""))

# Hybrid search
def hybrid_search(query_text, query_embedding, k=10):
    # Vector search
    vector_results = collection.search(query_embedding, k=k*2)

    # BM25 search
    bm25_results = bm25.search(query_text, k=k*2)

    # Fuse
    return reciprocal_rank_fusion(vector_results, bm25_results, k=k)
```

## Async Support

```python
import asyncio
from needle import Database

async def search_async(collection, queries):
    # Run searches concurrently
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(None, collection.search, q, 10, None)
        for q in queries
    ]
    results = await asyncio.gather(*tasks)
    return results

# Usage
async def main():
    db = Database.open("vectors.needle")
    collection = db.collection("documents")

    queries = [query1, query2, query3]
    all_results = await search_async(collection, queries)
```

## Context Manager

```python
from needle import Database

# Auto-save on exit
with Database.open("vectors.needle") as db:
    collection = db.collection("documents")
    collection.insert("doc1", embedding, {})
    # Automatically saved when exiting context
```

## Error Handling

```python
from needle import Database, NeedleError

try:
    db = Database.open("vectors.needle")
    collection = db.collection("nonexistent")
except NeedleError as e:
    print(f"Needle error: {e}")

# Specific error types
from needle import (
    CollectionNotFoundError,
    CollectionExistsError,
    DimensionMismatchError,
    VectorNotFoundError
)

try:
    collection = db.collection("documents")
    collection.get("nonexistent_id")
except VectorNotFoundError:
    print("Vector not found")
except NeedleError as e:
    print(f"Other error: {e}")
```

## Performance Tips

```python
# 1. Use batch operations
embeddings = generate_embeddings(documents)
for i, (doc, emb) in enumerate(zip(documents, embeddings)):
    collection.insert(f"doc{i}", emb, doc)
db.save()  # Save once at the end

# 2. Reuse database connection
db = Database.open("vectors.needle")
# ... use throughout application lifetime

# 3. Use numpy arrays
import numpy as np
embedding = np.array(embedding, dtype=np.float32)

# 4. Batch search for multiple queries
results = collection.batch_search(queries, k=10)
```

## Next Steps

- [JavaScript Bindings](/docs/bindings/javascript)
- [API Reference](/docs/api-reference)
- [Semantic Search Guide](/docs/guides/semantic-search)
