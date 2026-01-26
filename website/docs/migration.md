---
sidebar_position: 13
---

# Migration Guide

Step-by-step guides for migrating to Needle from other vector databases.

## Overview

Migrating to Needle typically involves:

1. **Export** data from your current database
2. **Transform** to Needle's format (if needed)
3. **Import** into Needle
4. **Verify** the migration
5. **Update** your application code

:::tip Migration Size Recommendations
- **< 100K vectors**: Direct migration in a single script
- **100K - 1M vectors**: Batch migration with progress tracking
- **> 1M vectors**: Streaming migration with checkpoints
:::

## From Chroma

### Export from Chroma

```python
import chromadb

# Connect to Chroma
client = chromadb.Client()  # or PersistentClient()
chroma_collection = client.get_collection("my_collection")

# Get all data
data = chroma_collection.get(
    include=["embeddings", "metadatas", "documents"]
)

print(f"Exporting {len(data['ids'])} vectors")
```

### Import to Needle

```python
import needle

# Create Needle database
db = needle.Database.open("vectors.needle")
db.create_collection("my_collection", 384)  # Match your embedding dimension
collection = db.collection("my_collection")

# Import vectors
for i, (id, embedding, metadata) in enumerate(zip(
    data["ids"],
    data["embeddings"],
    data["metadatas"]
)):
    # Chroma stores documents separately; merge if needed
    if data["documents"] and data["documents"][i]:
        metadata = metadata or {}
        metadata["document"] = data["documents"][i]

    collection.insert(id, embedding, metadata)

    if (i + 1) % 10000 == 0:
        print(f"Imported {i + 1} vectors")
        db.save()  # Checkpoint

db.save()
print("Migration complete!")
```

### Key Differences from Chroma

| Feature | Chroma | Needle |
|---------|--------|--------|
| Documents | Stored separately | Store in metadata |
| Where filters | `where={"field": "value"}` | `filter={"field": "value"}` |
| Collection creation | `get_or_create_collection()` | `create_collection()` (explicit) |
| Persistence | Directory-based | Single file |

### Code Changes

**Chroma**:
```python
results = collection.query(
    query_embeddings=[query_vector],
    n_results=10,
    where={"category": "books"}
)
```

**Needle**:
```python
results = collection.search(
    query_vector,
    k=10,
    filter={"category": "books"}
)
```

---

## From Qdrant

### Export from Qdrant

```python
from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient("localhost", port=6333)

# Scroll through all points
all_points = []
offset = None

while True:
    points, offset = client.scroll(
        collection_name="my_collection",
        limit=1000,
        offset=offset,
        with_vectors=True,
        with_payload=True
    )

    all_points.extend(points)

    if offset is None:
        break

print(f"Exported {len(all_points)} vectors")
```

### Import to Needle

```python
import needle

db = needle.Database.open("vectors.needle")
db.create_collection("my_collection", 384)
collection = db.collection("my_collection")

for i, point in enumerate(all_points):
    # Convert Qdrant point to Needle format
    id = str(point.id)  # Qdrant uses UUID or int, Needle uses string
    vector = point.vector
    metadata = point.payload or {}

    collection.insert(id, vector, metadata)

    if (i + 1) % 10000 == 0:
        print(f"Imported {i + 1} vectors")
        db.save()

db.save()
print("Migration complete!")
```

### Key Differences from Qdrant

| Feature | Qdrant | Needle |
|---------|--------|--------|
| IDs | UUID or integer | String |
| Payload | `payload` field | `metadata` field |
| Filter syntax | Qdrant filter DSL | MongoDB-style JSON |
| Named vectors | Supported | Single vector per document |
| Server | Required | Embedded (optional server) |

### Filter Translation

**Qdrant**:
```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="books")),
        FieldCondition(key="price", range=Range(lt=50))
    ]
)
```

**Needle**:
```python
filter = {
    "$and": [
        {"category": "books"},
        {"price": {"$lt": 50}}
    ]
}
```

---

## From Pinecone

### Export from Pinecone

Pinecone doesn't have a direct export API. You need to fetch vectors by ID or use the `fetch` endpoint:

```python
import pinecone

# Initialize
pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("my-index")

# If you have the IDs stored elsewhere
ids = load_your_ids()  # Your list of vector IDs

# Fetch in batches
all_vectors = []
batch_size = 100

for i in range(0, len(ids), batch_size):
    batch_ids = ids[i:i + batch_size]
    response = index.fetch(ids=batch_ids)

    for id, data in response["vectors"].items():
        all_vectors.append({
            "id": id,
            "vector": data["values"],
            "metadata": data.get("metadata", {})
        })

print(f"Exported {len(all_vectors)} vectors")
```

### Alternative: Re-embed from Source

If you don't have IDs stored, re-embed your source data:

```python
from your_embedding_model import embed

# Your source documents
documents = load_your_documents()

all_vectors = []
for doc in documents:
    all_vectors.append({
        "id": doc["id"],
        "vector": embed(doc["text"]),
        "metadata": doc.get("metadata", {})
    })
```

### Import to Needle

```python
import needle

db = needle.Database.open("vectors.needle")
db.create_collection("my_collection", 1536)  # OpenAI embedding dimension
collection = db.collection("my_collection")

for i, item in enumerate(all_vectors):
    collection.insert(item["id"], item["vector"], item["metadata"])

    if (i + 1) % 10000 == 0:
        print(f"Imported {i + 1} vectors")
        db.save()

db.save()
print("Migration complete!")
```

### Key Differences from Pinecone

| Feature | Pinecone | Needle |
|---------|----------|--------|
| Hosting | Cloud only | Self-hosted/Embedded |
| Namespaces | Supported | Use collections |
| Sparse vectors | Supported | Supported |
| Pricing | Per-operation | Free (open source) |
| Data residency | Pinecone servers | Your infrastructure |

---

## From pgvector

### Export from PostgreSQL

```python
import psycopg2
import numpy as np

conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
cur = conn.cursor()

# Fetch all vectors
cur.execute("""
    SELECT id, embedding, metadata
    FROM items
""")

all_vectors = []
for row in cur:
    id, embedding, metadata = row
    # pgvector returns string representation, parse it
    if isinstance(embedding, str):
        embedding = [float(x) for x in embedding.strip('[]').split(',')]
    all_vectors.append({
        "id": str(id),
        "vector": embedding,
        "metadata": metadata or {}
    })

print(f"Exported {len(all_vectors)} vectors")
cur.close()
conn.close()
```

### Import to Needle

```python
import needle

db = needle.Database.open("vectors.needle")
db.create_collection("items", 384)
collection = db.collection("items")

for i, item in enumerate(all_vectors):
    collection.insert(item["id"], item["vector"], item["metadata"])

    if (i + 1) % 10000 == 0:
        print(f"Imported {i + 1} vectors")
        db.save()

db.save()
print("Migration complete!")
```

### Key Differences from pgvector

| Feature | pgvector | Needle |
|---------|----------|--------|
| Query language | SQL | Programmatic API |
| Joins | Full SQL joins | None (dedicated vector DB) |
| Transactions | ACID | Basic |
| Index types | IVFFlat, HNSW | HNSW |
| Integration | PostgreSQL extension | Standalone library |

### Query Translation

**pgvector (SQL)**:
```sql
SELECT id, metadata, embedding <-> '[0.1, 0.2, ...]' AS distance
FROM items
WHERE metadata->>'category' = 'books'
ORDER BY embedding <-> '[0.1, 0.2, ...]'
LIMIT 10;
```

**Needle (Python)**:
```python
results = collection.search(
    query_vector,
    k=10,
    filter={"category": "books"}
)
```

---

## From FAISS

### Export from FAISS

FAISS is an index library, not a database. You need your original vectors:

```python
import faiss
import numpy as np

# Load FAISS index
index = faiss.read_index("my_index.faiss")

# Reconstruct vectors (if stored in index)
num_vectors = index.ntotal
dimension = index.d

vectors = np.zeros((num_vectors, dimension), dtype=np.float32)
for i in range(num_vectors):
    vectors[i] = index.reconstruct(i)

# Load your IDs and metadata separately (FAISS doesn't store these)
ids = load_your_ids()
metadata = load_your_metadata()
```

### Import to Needle

```python
import needle

db = needle.Database.open("vectors.needle")
db.create_collection("my_collection", dimension)
collection = db.collection("my_collection")

for i, (vec, id, meta) in enumerate(zip(vectors, ids, metadata)):
    collection.insert(id, vec.tolist(), meta)

    if (i + 1) % 10000 == 0:
        print(f"Imported {i + 1} vectors")
        db.save()

db.save()
print("Migration complete!")
```

### Key Differences from FAISS

| Feature | FAISS | Needle |
|---------|-------|--------|
| Type | Index library | Full database |
| IDs | Integer only | String |
| Metadata | Not supported | JSON metadata |
| Filtering | Not supported | MongoDB-style |
| Persistence | Manual save/load | Built-in |

---

## From Milvus

### Export from Milvus

```python
from pymilvus import connections, Collection

# Connect
connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")
collection.load()

# Query all vectors
results = collection.query(
    expr="id >= 0",  # Match all
    output_fields=["id", "embedding", "metadata"]
)

all_vectors = []
for item in results:
    all_vectors.append({
        "id": str(item["id"]),
        "vector": item["embedding"],
        "metadata": item.get("metadata", {})
    })

print(f"Exported {len(all_vectors)} vectors")
```

### Import to Needle

```python
import needle

db = needle.Database.open("vectors.needle")
db.create_collection("my_collection", 384)
collection = db.collection("my_collection")

for i, item in enumerate(all_vectors):
    collection.insert(item["id"], item["vector"], item["metadata"])

    if (i + 1) % 10000 == 0:
        print(f"Imported {i + 1} vectors")
        db.save()

db.save()
print("Migration complete!")
```

---

## Rust Migration Example

For Rust applications, here's a generic migration pattern:

```rust
use needle::{Database, CollectionConfig, DistanceFunction};
use serde_json::json;

struct VectorRecord {
    id: String,
    vector: Vec<f32>,
    metadata: serde_json::Value,
}

fn migrate(
    source_vectors: impl Iterator<Item = VectorRecord>,
    output_path: &str,
    dimension: usize,
) -> needle::Result<()> {
    let db = Database::open(output_path)?;

    let config = CollectionConfig::new(dimension, DistanceFunction::Cosine);
    db.create_collection_with_config("migrated", config)?;

    let collection = db.collection("migrated")?;

    let mut count = 0;
    for record in source_vectors {
        collection.insert(&record.id, &record.vector, record.metadata)?;

        count += 1;
        if count % 10000 == 0 {
            println!("Imported {} vectors", count);
            db.save()?;  // Checkpoint
        }
    }

    db.save()?;
    println!("Migration complete! {} vectors imported", count);

    Ok(())
}
```

---

## Verification Checklist

After migration, verify:

```python
import needle

db = needle.Database.open("vectors.needle")
collection = db.collection("my_collection")

# 1. Check vector count
count = collection.count()
print(f"Vector count: {count}")
assert count == expected_count, "Count mismatch!"

# 2. Sample and verify vectors
sample_ids = ["id1", "id2", "id3"]  # Known IDs
for id in sample_ids:
    entry = collection.get(id)
    print(f"ID: {entry.id}, Vector dims: {len(entry.vector)}")
    # Compare with source if possible

# 3. Test search quality
results = collection.search(test_query, k=10)
print("Search results:", [r.id for r in results])
# Compare with expected results from source DB

# 4. Verify metadata
for result in results[:3]:
    print(f"Metadata: {result.metadata}")

print("Verification passed!")
```

---

## Rollback Plan

Keep your source database running until migration is verified:

1. **Dual-write period**: Write to both old and new DB
2. **Shadow read**: Query both, compare results
3. **Gradual cutover**: Route increasing traffic to Needle
4. **Full cutover**: Switch 100% to Needle
5. **Decommission**: Remove old database after stability period

```python
# Example dual-write wrapper
class DualWriteCollection:
    def __init__(self, old_client, needle_collection):
        self.old = old_client
        self.new = needle_collection

    def insert(self, id, vector, metadata):
        # Write to both
        self.old.insert(id, vector, metadata)
        self.new.insert(id, vector, metadata)

    def search(self, query, k, use_needle=False):
        if use_needle:
            return self.new.search(query, k)
        return self.old.search(query, k)
```

---

## Need Help?

- [GitHub Discussions](https://github.com/anthropics/needle/discussions) - Ask migration questions
- [Discord](https://discord.gg/anthropic) - Real-time help
- [Comparison Guide](/docs/comparison) - Feature comparison with other databases
