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
  distance: 'cosine'         // optional: 'cosine' | 'euclidean' | 'dotproduct'
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

## Filtering

The JS SDK does not currently expose a dedicated `searchWithFilter()` method. To filter results by metadata, post-process after `search()`:

```javascript
const results = await db.search(queryVector, 20); // fetch extra candidates
const filtered = results.filter(r => r.metadata?.category === 'science');
const topK = filtered.slice(0, 5);
```

For server-side MongoDB-style filtering (`$eq`, `$ne`, `$gt`, `$lt`, `$in`, `$or`, etc.), use the [REST API](../docs/api-reference.md) or the [Python SDK](../python/README.md).

## TypeScript

The SDK is written in TypeScript and exports all types. Key interfaces:

```typescript
import type {
  NeedleDBConfig,
  SearchResult,
  InsertOptions,
  PersistenceStats,
} from '@anthropic/needle';

// NeedleDBConfig
interface NeedleDBConfig {
  name?: string;
  dimensions: number;
  distance?: 'cosine' | 'euclidean' | 'dotproduct';
  dbName?: string;               // IndexedDB database name
  progressiveLoading?: boolean;  // lazy index hydration
  maxInMemory?: number;          // max vectors in memory
}

// SearchResult
interface SearchResult {
  id: string;
  distance: number;   // raw distance (lower = more similar)
  score: number;      // 1 / (1 + distance), normalized to [0, 1]
  metadata?: Record<string, unknown>;
}
```

Vectors accept both `Float32Array` and `number[]`:

```typescript
await db.insert('doc1', new Float32Array(384).fill(0.1));
await db.insert('doc2', [0.1, 0.2, /* ... */]);
```

## Performance Tuning

### Batch Inserts

Use `insertBatch()` instead of individual `insert()` calls to reduce overhead:

```javascript
const items = vectors.map((vec, i) => ({
  id: `doc_${i}`,
  vector: vec,
  metadata: { index: i },
}));
const count = await db.insertBatch(items);
```

### Search Caching

The SDK automatically caches recent search results (up to 1,000 queries, 60 s TTL). The cache is cleared on any insert or delete. For latency-sensitive applications, pre-warm the cache with expected queries after loading.

### Memory Budget

Use the `maxInMemory` config option to cap the number of vectors held in memory. Monitor usage via `db.memoryUsage` (estimated bytes). Call `db.persist()` periodically to flush to IndexedDB.

### Web Workers

Offload search to a background thread to keep the main thread responsive:

```typescript
import { createWorkerClient } from '@anthropic/needle';

const client = createWorkerClient(new Worker('./needle-worker.ts'));
await client.init('docs', 384);
await client.insert('doc1', [0.1, 0.2, /* ... */]);
const results = await client.search([0.1, 0.2, /* ... */], 10);
```

See the [Web Worker Message Protocol](#) types (`NeedleWorkerMessage`, `NeedleWorkerResponse`) exported by the package.

## Related

- [WASM Guide](../docs/WASM_GUIDE.md) — WebAssembly integration for browser and edge deployment
- [API Reference](../docs/api-reference.md) — Full Rust and REST API documentation
