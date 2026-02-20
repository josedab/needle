# Needle Examples

Run any example with:

```bash
cargo run --example <name>
```

## Recommended Order

Work through these examples in order to learn Needle's core features:

| # | Example | Description | Command |
|---|---------|-------------|---------|
| 1 | `basic_usage` | Create, insert, search — start here | `cargo run --example basic_usage` |
| 2 | `filtered_search` | MongoDB-style metadata filters | `cargo run --example filtered_search` |
| 3 | `persistence` | Save to disk and reload | `cargo run --example persistence` |
| 4 | `quantization` | Compress vectors to save memory | `cargo run --example quantization` |
| 5 | `sparse_vectors` | TF-IDF / SPLADE sparse vector support | `cargo run --example sparse_vectors` |
| 6 | `multi_vector` | ColBERT-style multi-vector retrieval | `cargo run --example multi_vector` |
| 7 | `image_search` | Image similarity search patterns | `cargo run --example image_search` |
| 8 | `rag_chatbot` | Build a RAG pipeline (mock embeddings) | `cargo run --example rag_chatbot` |
| 9 | `multi_tenant` | Multi-tenancy and namespace isolation | `cargo run --example multi_tenant` |
| 10 | `sharding` | Data sharding and partitioning strategies | `cargo run --example sharding` |

## Feature-flagged Examples

These examples require additional feature flags:

| Example | Features | Command |
|---------|----------|---------|
| `hybrid_search` | `hybrid` | `cargo run --example hybrid_search --features hybrid` |

> Tip: Each example file includes a "Run with:" comment at the top.

## Python Examples

| Example | Description | Prerequisites |
|---------|-------------|---------------|
| `rag_openai.py` | End-to-end RAG with OpenAI embeddings | `pip install openai requests` + running server |

See [RAG Quickstart](../docs/rag-quickstart.md) for the full walkthrough.
