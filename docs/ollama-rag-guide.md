# Local RAG with Needle + Ollama

Build a fully local, private RAG (Retrieval-Augmented Generation) system using Needle for vector search and Ollama for embeddings and LLM inference. No cloud APIs, no data leaves your machine.

## Prerequisites

```bash
# Install Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Pull an embedding model and a chat model
ollama pull nomic-embed-text
ollama pull llama3.2

# Install Needle (Rust)
cargo install needle
# Or use the server
cargo run --features server -- serve
```

## Architecture

```
Documents → Ollama (embed) → Needle (store) → Query → Ollama (embed) → Needle (search) → Ollama (generate)
```

Everything runs locally. No internet required after model download.

## Step 1: Embed and Index Documents

```bash
# Create a collection
curl -X POST http://localhost:8080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "knowledge", "dimensions": 768, "distance": "cosine"}'

# Embed a document with Ollama
EMBEDDING=$(curl -s http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "Needle is an embedded vector database written in Rust"}' \
  | jq -c '.embedding')

# Insert into Needle
curl -X POST http://localhost:8080/v1/collections/knowledge/vectors \
  -H "Content-Type: application/json" \
  -d "{
    \"id\": \"doc1\",
    \"vector\": $EMBEDDING,
    \"metadata\": {\"text\": \"Needle is an embedded vector database written in Rust\", \"source\": \"readme\"}
  }"
```

## Step 2: Search for Relevant Context

```bash
# Embed the query
QUERY_EMBEDDING=$(curl -s http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "What is Needle?"}' \
  | jq -c '.embedding')

# Search Needle
RESULTS=$(curl -s -X POST http://localhost:8080/v1/collections/knowledge/search \
  -H "Content-Type: application/json" \
  -d "{\"vector\": $QUERY_EMBEDDING, \"k\": 3}")

echo "$RESULTS" | jq '.results[].metadata.text'
```

## Step 3: Generate with Retrieved Context

```bash
# Extract context from search results
CONTEXT=$(echo "$RESULTS" | jq -r '[.results[].metadata.text] | join("\n")')

# Generate answer with Ollama
curl -s http://localhost:11434/api/generate \
  -d "{
    \"model\": \"llama3.2\",
    \"prompt\": \"Based on the following context, answer the question.\\n\\nContext:\\n$CONTEXT\\n\\nQuestion: What is Needle?\\n\\nAnswer:\",
    \"stream\": false
  }" | jq -r '.response'
```

## Python Example

```python
import requests
import json

NEEDLE_URL = "http://localhost:8080/v1"
OLLAMA_URL = "http://localhost:11434"

def embed(text: str) -> list[float]:
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": "nomic-embed-text", "prompt": text
    })
    return resp.json()["embedding"]

def index_document(doc_id: str, text: str, metadata: dict = None):
    vector = embed(text)
    requests.post(f"{NEEDLE_URL}/collections/knowledge/vectors", json={
        "id": doc_id, "vector": vector,
        "metadata": {**(metadata or {}), "text": text}
    })

def search(query: str, k: int = 5) -> list[dict]:
    vector = embed(query)
    resp = requests.post(f"{NEEDLE_URL}/collections/knowledge/search", json={
        "vector": vector, "k": k
    })
    return resp.json()["results"]

def ask(question: str) -> str:
    results = search(question)
    context = "\n".join(r["metadata"]["text"] for r in results if r.get("metadata"))
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": "llama3.2",
        "prompt": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:",
        "stream": False
    })
    return resp.json()["response"]

# Usage
index_document("doc1", "Needle is an embedded vector database written in Rust.")
index_document("doc2", "Needle uses HNSW for fast approximate nearest neighbor search.")
print(ask("How does Needle perform search?"))
```

## Why This Matters

- **100% private**: No data leaves your machine. Perfect for sensitive documents.
- **Zero cost**: No API keys, no cloud bills. Everything runs locally.
- **Fast**: Needle provides sub-10ms search latency. Ollama runs models on your GPU.
- **Simple**: Single-file database. No infrastructure to manage.
