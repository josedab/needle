# Needle JavaScript SDK

This SDK provides a browser-focused API with IndexedDB persistence.

## Architecture

> **This is a pure JavaScript SDK, not a WASM binding.**

The JS SDK is a standalone TypeScript implementation that runs entirely in JavaScript without
WebAssembly. It implements its own HNSW index and vector search algorithms in pure JS,
optimized for browser environments.

| Aspect | JS SDK (`sdk/js/`) | WASM Bindings (`src/wasm.rs`) |
|--------|-------------------|------------------------------|
| **Implementation** | Pure TypeScript | Rust compiled to WebAssembly |
| **Persistence** | IndexedDB | None (in-memory only) |
| **Performance** | Good for small-medium datasets | Near-native Rust speed |
| **Bundle size** | Lightweight (~50KB) | Larger (WASM binary) |
| **Dependencies** | None (zero-dep) | Requires WASM runtime |
| **Best for** | Browser apps, prototyping | Performance-critical workloads |

For WASM-based vector search (compiled from the full Rust core), see the [WASM Guide](../docs/WASM_GUIDE.md).

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

// InsertOptions — passed as the third argument to insert()
interface InsertOptions {
  metadata?: Record<string, unknown>;  // arbitrary key-value metadata
  skipDuplicates?: boolean;            // skip if ID already exists
}

// PersistenceStats — returned by persist()
interface PersistenceStats {
  stored: boolean;       // true if data was written to IndexedDB
  sizeBytes: number;     // serialized size in bytes
  vectorCount: number;   // number of vectors persisted
  timestamp: number;     // epoch ms when persistence completed
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

#### `createWorkerClient(worker)`

Creates a typed client that communicates with a Needle Web Worker via `postMessage`.

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| `init(name, dimensions, distance?)` | `string, number, string?` | `Promise<{ success: boolean }>` | Initialize the worker database |
| `insert(vecId, vector, metadata?)` | `string, number[], Record?` | `Promise<{ success: boolean }>` | Insert a vector |
| `search(query, k)` | `number[], number` | `Promise<SearchResult[]>` | Search for nearest neighbors |
| `delete(vecId)` | `string` | `Promise<{ existed: boolean }>` | Delete a vector |
| `persist()` | — | `Promise<{ stats: PersistenceStats }>` | Persist to IndexedDB |
| `clear()` | — | `Promise<void>` | Clear all vectors |
| `stats()` | — | `Promise<{ count, memoryUsage, dirty }>` | Get collection stats |
| `terminate()` | — | `void` | Terminate the worker |

#### Worker Message Protocol

The worker communicates using typed messages. Use these types to implement a custom worker:

```typescript
import type { NeedleWorkerMessage, NeedleWorkerResponse } from '@anthropic/needle';
```

**`NeedleWorkerMessage`** — sent from main thread to worker:

| `type` | Fields | Description |
|--------|--------|-------------|
| `'init'` | `name, dimensions, distance?` | Initialize database |
| `'insert'` | `vecId, vector, metadata?` | Insert a vector |
| `'search'` | `query, k` | Search nearest neighbors |
| `'delete'` | `vecId` | Delete a vector |
| `'persist'` | — | Save to IndexedDB |
| `'clear'` | — | Clear all vectors |
| `'stats'` | — | Request collection stats |

All messages include an `id: string` field for request/response correlation.

**`NeedleWorkerResponse`** — sent from worker to main thread:

| `type` | Fields | Description |
|--------|--------|-------------|
| `'ready'` | `success` | Worker initialized |
| `'inserted'` | `success` | Vector inserted |
| `'results'` | `results: SearchResult[]` | Search results |
| `'deleted'` | `existed` | Whether vector existed |
| `'persisted'` | `stats: PersistenceStats` | Persistence result |
| `'cleared'` | — | Vectors cleared |
| `'stats'` | `count, memoryUsage, dirty` | Collection stats |
| `'error'` | `message` | Error response |

#### Example Worker Implementation

```typescript
// needle-worker.ts
import { NeedleDB, NeedleWorkerMessage, NeedleWorkerResponse } from '@anthropic/needle';

let db: NeedleDB | null = null;

self.onmessage = async (e: MessageEvent<NeedleWorkerMessage>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'init':
        db = await NeedleDB.create(msg.name, { dimensions: msg.dimensions });
        self.postMessage({ id: msg.id, type: 'ready', success: true });
        break;
      case 'search':
        const results = await db!.search(msg.query, msg.k);
        self.postMessage({ id: msg.id, type: 'results', results });
        break;
      case 'insert':
        await db!.insert(msg.vecId, msg.vector, msg.metadata);
        self.postMessage({ id: msg.id, type: 'inserted', success: true });
        break;
      // ... handle delete, persist, clear, stats
    }
  } catch (err) {
    self.postMessage({ id: msg.id, type: 'error', message: String(err) });
  }
};
```

### React Integration

The SDK provides a React hook factory that avoids a hard React dependency.

#### `createUseVectorSearch(react)`

Creates a `useVectorSearch` hook by accepting React's `useState` and `useEffect`:

```typescript
import { useState, useEffect } from 'react';
import { createUseVectorSearch, NeedleDB } from '@anthropic/needle';

const useVectorSearch = createUseVectorSearch({ useState, useEffect });

function SearchComponent({ db, query }: { db: NeedleDB; query: Float32Array }) {
  const { results, loading, error } = useVectorSearch(db, query, 10);

  if (loading) return <div>Searching...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <ul>
      {results.map(r => <li key={r.id}>{r.id}: {r.score.toFixed(3)}</li>)}
    </ul>
  );
}
```

**Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `db` | `NeedleDB \| null` | Database instance (search runs when non-null) |
| `query` | `Float32Array \| number[] \| null` | Query vector (search runs when non-null) |
| `k` | `number` | Number of results to return (default: 10) |

**Returns** `UseVectorSearchResult`:

```typescript
interface UseVectorSearchResult {
  results: SearchResult[];  // search results (empty while loading)
  loading: boolean;         // true while search is in progress
  error: Error | null;      // error if search failed
}
```

See the [Web Worker Message Protocol](#worker-message-protocol) types (`NeedleWorkerMessage`, `NeedleWorkerResponse`) exported by the package.

## Related

- [WASM Guide](../docs/WASM_GUIDE.md) — WebAssembly integration for browser and edge deployment (different from this SDK)
- [Language Bindings Overview](../docs/language-bindings.md) — Comparison of all Needle language bindings
- [API Reference](../docs/api-reference.md) — Full Rust and REST API documentation
