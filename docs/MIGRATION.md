# Migration Guide — Needle Vector Database

This guide covers breaking changes and migration paths between Needle versions.

## v0.1.x → v1.0.0 (Planned)

### Deprecated APIs

| Deprecated | Replacement | Since | Removal |
|-----------|-------------|-------|---------|
| `db.drop_collection(name)` | `db.delete_collection(name)` | v0.1.0 | v1.0.0 |

### Migration Steps

#### 1. Replace `drop_collection` → `delete_collection`

**Before:**
```rust
db.drop_collection("my_collection")?;
```

**After:**
```rust
db.delete_collection("my_collection")?;
```

Both methods are functionally identical. `drop_collection` will be removed in v1.0.0.

#### 2. API Stability Tiers

Starting with v1.0.0, the public API is organized into stability tiers:

| Tier | Guarantee | Types |
|------|-----------|-------|
| **Stable** | Semver-protected. Breaking changes only in major versions. | `Database`, `Collection`, `CollectionRef`, `Filter`, `SearchResult`, `HnswConfig`, `DistanceFunction`, `NeedleError` |
| **Beta** | May change in minor versions. | `Bm25Index`, `AsyncDatabase`, `ServerConfig`, `TextEmbedder`, `AutoEmbedder` |
| **Experimental** | May change without notice. Access via `needle::experimental_api::*` | GPU, cloud control, agentic memory, plugins |

#### 3. Prelude Usage

Use the prelude for the most common imports:

```rust
use needle::prelude::*;
// Imports: Database, Collection, CollectionConfig, CollectionRef,
//          SearchResult, SearchParams, DistanceFunction, Filter,
//          HnswConfig, NeedleError, Result
```

#### 4. Feature Flags

Starting v1.0.0, the `full` feature includes all stable features:

```toml
[dependencies]
needle = { version = "1.0", features = ["full"] }
```

Individual features for minimal builds:
- `server` — HTTP REST API
- `hybrid` — BM25 + vector search
- `metrics` — Prometheus metrics
- `encryption` — ChaCha20-Poly1305 at rest

### Checking Your Code

Use the API stability manifest to verify your usage:

```rust
use needle::api_stability::ApiManifest;

let manifest = ApiManifest::default_manifest();
// Check if a type you depend on is stable
assert!(manifest.is_stable("Database"));
assert!(manifest.is_stable("Collection"));
```

### Getting Help

- [API Stability Guide](api-stability.md)
- [GitHub Discussions](https://github.com/anthropics/needle/discussions)
- [SUPPORT.md](SUPPORT.md)
