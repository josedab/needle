# Needle JavaScript SDK

This SDK provides a browser-focused API with IndexedDB persistence.

## Install (from source)

```bash
cd sdk/js
npm install
```

## Build

```bash
npm run build
```

## Usage

```javascript
import { NeedleDB } from '@anthropic/needle';

const db = await NeedleDB.create('demo', { dimensions: 384 });
await db.insert('doc1', new Float32Array(384).fill(0.1), { title: 'Hello' });
const results = await db.search(new Float32Array(384).fill(0.1), 10);
await db.persist();
```

## API Reference

### `NeedleDB`

| Method / Property | Description |
|-------------------|-------------|
| `NeedleDB.create(name, config)` | Create a new database instance (async) |
| `insert(id, vector, metadata?)` | Insert a single vector with optional metadata |
| `insertBatch(items)` | Insert multiple vectors in one call |
| `search(query, k)` | Search for the `k` nearest vectors |
| `persist()` | Save the database to IndexedDB |
| `clear()` | Remove all vectors from the collection |
| `count` | Number of vectors stored (getter) |
| `isDirty` | Whether unsaved changes exist (getter) |
| `memoryUsage` | Estimated memory usage in bytes (getter) |

### Configuration

```javascript
const db = await NeedleDB.create('my-db', {
  dimensions: 384,           // required: vector dimensionality
  distanceFunction: 'cosine' // optional: 'cosine' | 'euclidean' | 'dot_product'
});
```

### Search with Metadata Filtering

```javascript
// Insert vectors with metadata
await db.insert('doc1', embedding1, { category: 'science', year: 2024 });
await db.insert('doc2', embedding2, { category: 'history', year: 2023 });

// Search returns results with id, distance, and metadata
const results = await db.search(queryVector, 5);
for (const r of results) {
  console.log(`${r.id}: distance=${r.distance}`, r.metadata);
}
```

### Error Handling

```javascript
try {
  const db = await NeedleDB.create('my-db', { dimensions: 384 });
  await db.insert('id', vector, metadata);
} catch (error) {
  // Common errors:
  // - Dimension mismatch (vector length != configured dimensions)
  // - IndexedDB not available (e.g., in some server-side environments)
  // - Persistence failures (storage quota exceeded)
  console.error('Needle error:', error.message);
}
```

### Browser vs Node.js

| Environment | Vector Search | IndexedDB Persistence | Web Workers |
|-------------|:------------:|:--------------------:|:-----------:|
| Browser     | ✅           | ✅                   | ✅          |
| Node.js     | ✅           | ❌ (requires polyfill) | ❌          |

The SDK uses IndexedDB for persistence, which is available natively in browsers. For Node.js usage, consider using the [WASM bindings](../docs/WASM_GUIDE.md) or the [Python SDK](../python/README.md) instead.

## Related

- [WASM Guide](../docs/WASM_GUIDE.md) — WebAssembly integration for browser and edge deployment
- [API Reference](../docs/api-reference.md) — Full Rust and REST API documentation
