# Language Bindings Overview

Needle provides official bindings for multiple languages and platforms, enabling vector search across desktop, browser, mobile, and edge environments.

## Supported Bindings

| Language | Mechanism | Platforms | Documentation |
|----------|-----------|-----------|---------------|
| **Rust** | Native | All | [API Reference](api-reference.md) |
| **Python** | PyO3 | Linux, macOS, Windows | [Python SDK](../python/README.md) |
| **JavaScript** | Pure JS + IndexedDB | Browser, Node.js | [JS SDK](../sdk/js/README.md) |
| **WebAssembly** | wasm-bindgen | Browser, Node.js, Deno, Edge | [WASM Guide](WASM_GUIDE.md) |
| **Swift** | UniFFI | iOS, macOS | [Swift & Kotlin](../website/docs/bindings/swift-kotlin.md) |
| **Kotlin** | UniFFI | Android, JVM | [Swift & Kotlin](../website/docs/bindings/swift-kotlin.md) |

## Feature Comparison

| Feature | Rust | Python | JS SDK | WASM | Swift | Kotlin |
|---------|:----:|:------:|:------:|:----:|:-----:|:------:|
| Vector insert/search | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Metadata filtering | ✅ | ✅ | ❌¹ | ✅ | ✅ | ✅ |
| File persistence | ✅ | ✅ | IndexedDB | ❌ | ✅ | ✅ |
| Distance functions | All | All | 3² | All | All | All |
| HNSW configuration | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Thread safety | ✅ | ✅ | Web Workers | ✅ | ✅ | ✅ |
| Hybrid search (BM25) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

¹ JS SDK supports post-search filtering only; use the REST API for MongoDB-style filters.
² Cosine, Euclidean, Dot Product.

## Quick Start by Language

### Python

```bash
pip install needle-db           # Pure Python wrapper
pip install needle-db[native]   # With Rust-native backend (faster)
```

```python
import needle

db = needle.Database.in_memory()
db.create_collection("docs", 384)
coll = db.collection("docs")
coll.insert("doc1", [0.1] * 384, {"title": "Hello"})
results = coll.search([0.1] * 384, k=10)
```

See the full [Python SDK documentation](../python/README.md).

### JavaScript (Browser)

```bash
cd sdk/js && npm install && npm run build
```

```javascript
import { NeedleDB } from '@anthropic/needle';

const db = await NeedleDB.create('demo', { dimensions: 384 });
await db.insert('doc1', new Float32Array(384).fill(0.1), { title: 'Hello' });
const results = await db.search(new Float32Array(384).fill(0.1), 10);
```

See the full [JS SDK documentation](../sdk/js/README.md).

### WebAssembly

```bash
cargo build --target wasm32-unknown-unknown --features wasm
```

See the full [WASM Guide](WASM_GUIDE.md) for browser, Node.js, and edge deployment instructions.

### Swift (iOS/macOS)

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/anthropics/needle.git", from: "0.1.0")
]
```

```swift
import NeedleDB

let db = try Database.open(path: "vectors.needle")
try db.createCollection(name: "documents", dimensions: 384, distance: .cosine)
let collection = try db.collection(name: "documents")
try collection.insert(id: "doc1", vector: Array(repeating: 0.1, count: 384))
let results = try collection.search(query: queryVector, k: 10)
```

### Kotlin (Android/JVM)

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.anthropic:needle:0.1.0")
}
```

```kotlin
val db = Database.open("vectors.needle")
db.createCollection("documents", dimensions = 384, distance = Distance.COSINE)
val collection = db.collection("documents")
collection.insert("doc1", FloatArray(384) { 0.1f })
val results = collection.search(queryVector, k = 10)
```

See the full [Swift & Kotlin documentation](../website/docs/bindings/swift-kotlin.md).

## Framework Integrations

In addition to language bindings, Needle provides integration adapters for popular AI frameworks:

| Framework | Language | Documentation |
|-----------|----------|---------------|
| LangChain | Python | [LangChain Integration](../python/needle_langchain/README.md) |
| LlamaIndex | Python | [LlamaIndex Integration](../python/needle_llamaindex/README.md) |
| Haystack | Rust | Via `needle::integrations::haystack` |
| Semantic Kernel | Rust | Via `needle::integrations::semantic_kernel` |

## Building Bindings from Source

### Python (PyO3)

```bash
pip install maturin
maturin develop --features python
```

### WASM

```bash
cargo build --target wasm32-unknown-unknown --features wasm
```

### Swift/Kotlin (UniFFI)

```bash
cargo build --features uniffi-bindings
cargo run --features uniffi-bindings -- uniffi-bindgen generate \
    --library target/debug/libneedle.dylib --language swift --out-dir ./bindings/swift
```

## Architecture Notes

- **Python**: Uses [PyO3](https://pyo3.rs/) for zero-copy Rust↔Python interop. The `needle` Python module wraps Rust types directly, so performance is near-native.
- **WASM**: Uses [wasm-bindgen](https://rustwasm.github.io/wasm-bindgen/) to compile the Rust core to WebAssembly. Runs entirely client-side with no server required.
- **JS SDK**: A pure JavaScript implementation with IndexedDB persistence, optimized for browser use cases. Does not depend on WASM.
- **Swift/Kotlin**: Uses [UniFFI](https://mozilla.github.io/uniffi-rs/) to generate idiomatic bindings from a shared Rust core, ensuring consistent behavior across iOS and Android.
