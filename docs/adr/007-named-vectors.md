# ADR: Named Vectors (Multiple Vector Fields Per Document)

## Status: Proposed

## Context

Currently, each document in a Needle collection has exactly one vector. Users building multi-modal or multi-granularity applications need to store multiple embeddings per document (e.g., title_embedding + body_embedding + image_embedding).

Qdrant supports this via "named vectors" where each point can have multiple named vector fields, each with its own index.

## Decision

Add support for named vectors at the collection level:

### Storage Model

```rust
// Current:
struct VectorEntry {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

// Proposed:
struct VectorEntry {
    id: String,
    vectors: HashMap<String, Vec<f32>>,  // named vectors
    metadata: Option<Value>,
}
```

### API Changes

- Collection creation accepts `dimensions: HashMap<String, usize>` or backward-compatible `dimensions: usize` (creates a "default" vector field)
- Insert accepts `vectors: {"title": [...], "body": [...]}` or `vector: [...]` (backward-compat → "default")
- Search accepts `vector_name: "title"` to specify which field to search
- Each named vector gets its own HNSW index

### Migration

- Existing single-vector collections automatically get a "default" vector field name
- Storage format version bump with migration tooling
- Backward-compatible: omitting `vector_name` uses "default"

## Consequences

- Storage format breaking change requiring migration
- ~2x memory per additional vector field (separate HNSW index each)
- Enables multi-modal and multi-granularity search
