# RAG Quickstart

Build a Retrieval-Augmented Generation (RAG) pipeline with Needle and OpenAI in under 5 minutes.

## Prerequisites

- Needle server running (see below)
- Python 3.9+
- An OpenAI API key

## 1) Start Needle

```bash
# Option A: Docker
docker compose up -d

# Option B: From source
cargo run --features server -- serve -a 127.0.0.1:8080
```

Verify: `curl http://127.0.0.1:8080/health`

## 2) Install Python dependencies

```bash
pip install openai requests
```

## 3) Run the example

```bash
export OPENAI_API_KEY="sk-..."
python examples/rag_openai.py
```

You should see:

```
Created collection 'rag_demo' (1536 dims)
Inserted 5 documents

Query: How does Needle perform fast searches?

Top results:
  [doc2] (distance: 0.3214) HNSW (Hierarchical Navigable Small World) enables sub-10ms ...
  [doc1] (distance: 0.4102) Needle is an embedded vector database written in Rust ...
  [doc5] (distance: 0.4567) Needle supports metadata filtering, multiple distance metrics ...
```

## How it works

1. **Embed** — Each document is converted to a 1536-dimensional vector using OpenAI's `text-embedding-3-small` model.
2. **Store** — Vectors are inserted into a Needle collection via the HTTP API.
3. **Search** — The query is embedded with the same model, then Needle finds the most similar documents using HNSW.
4. **Generate** — (Your step) Pass the retrieved documents as context to an LLM for grounded answers.

## Using curl instead of Python

```bash
# Create collection
curl -X POST http://127.0.0.1:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","dimensions":1536}'

# Insert (replace VECTOR with your embedding)
curl -X POST http://127.0.0.1:8080/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"id":"doc1","vector":[...],"metadata":{"text":"your document"}}'

# Search
curl -X POST http://127.0.0.1:8080/collections/docs/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[...],"k":3}'
```

## Without an OpenAI key

Use the Rust example with mock embeddings instead:

```bash
cargo run --example rag_chatbot
```

## Next steps

- [HTTP Quickstart](http-quickstart.md) — Full HTTP API walkthrough
- [API Reference](api-reference.md) — Complete method documentation
- [How-To Guides](how-to-guides.md) — Hybrid search, filtering, quantization
