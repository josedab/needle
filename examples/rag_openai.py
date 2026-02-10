#!/usr/bin/env python3
"""
RAG example using Needle's HTTP API with OpenAI embeddings.

Prerequisites:
    pip install openai requests

Start the Needle server first:
    cargo run --features server -- serve -a 127.0.0.1:8080

Usage:
    export OPENAI_API_KEY="sk-..."
    python examples/rag_openai.py
"""

import json
import sys

try:
    import openai
    import requests
except ImportError:
    print("Install dependencies: pip install openai requests")
    sys.exit(1)

NEEDLE_URL = "http://127.0.0.1:8080"
COLLECTION = "rag_demo"
MODEL = "text-embedding-3-small"  # 1536 dimensions
DIMENSIONS = 1536

# Sample knowledge base
DOCUMENTS = [
    {"id": "doc1", "text": "Needle is an embedded vector database written in Rust, designed as SQLite for vectors."},
    {"id": "doc2", "text": "HNSW (Hierarchical Navigable Small World) enables sub-10ms approximate nearest neighbor search."},
    {"id": "doc3", "text": "Vector databases store high-dimensional embeddings for semantic similarity search."},
    {"id": "doc4", "text": "RAG combines retrieval of relevant documents with language model generation for grounded answers."},
    {"id": "doc5", "text": "Needle supports metadata filtering, multiple distance metrics, and single-file storage."},
]


def embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenAI."""
    client = openai.OpenAI()
    response = client.embeddings.create(input=texts, model=MODEL)
    return [item.embedding for item in response.data]


def main():
    # 1. Check server is running
    try:
        r = requests.get(f"{NEEDLE_URL}/health", timeout=3)
        r.raise_for_status()
    except requests.ConnectionError:
        print(f"Needle server not reachable at {NEEDLE_URL}")
        print("Start it with: cargo run --features server -- serve -a 127.0.0.1:8080")
        sys.exit(1)

    # 2. Create collection
    requests.delete(f"{NEEDLE_URL}/collections/{COLLECTION}")  # clean slate
    r = requests.post(
        f"{NEEDLE_URL}/collections",
        json={"name": COLLECTION, "dimensions": DIMENSIONS},
    )
    print(f"Created collection '{COLLECTION}' ({DIMENSIONS} dims)")

    # 3. Embed and insert documents
    texts = [doc["text"] for doc in DOCUMENTS]
    embeddings = embed(texts)

    for doc, vector in zip(DOCUMENTS, embeddings):
        requests.post(
            f"{NEEDLE_URL}/collections/{COLLECTION}/vectors",
            json={"id": doc["id"], "vector": vector, "metadata": {"text": doc["text"]}},
        )
    print(f"Inserted {len(DOCUMENTS)} documents")

    # 4. Search
    query = "How does Needle perform fast searches?"
    print(f"\nQuery: {query}")

    query_vector = embed([query])[0]
    r = requests.post(
        f"{NEEDLE_URL}/collections/{COLLECTION}/search",
        json={"vector": query_vector, "k": 3},
    )
    results = r.json()

    print("\nTop results:")
    for result in results:
        rid = result.get("id", "?")
        dist = result.get("distance", 0)
        text = result.get("metadata", {}).get("text", "")
        print(f"  [{rid}] (distance: {dist:.4f}) {text}")

    # 5. Clean up
    requests.delete(f"{NEEDLE_URL}/collections/{COLLECTION}")
    print("\nDone! Collection cleaned up.")


if __name__ == "__main__":
    main()
