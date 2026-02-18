---
sidebar_position: 9
---

# WASM / JavaScript Guide

Use Needle as a WebAssembly module in browsers, Node.js, Deno, and edge runtimes like Cloudflare Workers.

## Getting Started

### Prerequisites

- Node.js 16+ (for build tools)
- npm or yarn
- Modern browser with WebAssembly support

### Building from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for browser
wasm-pack build --target web --features wasm

# Build for Node.js
wasm-pack build --target nodejs --features wasm

# Build for bundlers (webpack, rollup, etc.)
wasm-pack build --target bundler --features wasm
```

This produces a `pkg/` directory containing:
- `needle_bg.wasm` — The WebAssembly binary
- `needle.js` — JavaScript glue code
- `needle.d.ts` — TypeScript type definitions
- `package.json` — npm package manifest

---

## Installation

### npm (Node.js / Bundlers)

```bash
npm install @anthropic/needle-wasm
```

### CDN (Browser)

```html
<script type="module">
  import init, { WasmCollection } from 'https://unpkg.com/@anthropic/needle-wasm/needle.js';

  await init();
  const collection = new WasmCollection('documents', 384);
</script>
```

---

## Basic Usage

### Creating a Collection

```javascript
import init, { WasmCollection } from '@anthropic/needle-wasm';

await init();

const collection = new WasmCollection('my_documents', 384, 'cosine');
```

**Distance functions:** `'cosine'` (default), `'euclidean'` / `'l2'`, `'dot'` / `'dotproduct'`, `'manhattan'` / `'l1'`.

### Inserting Vectors

```javascript
collection.insert('doc1', new Float32Array([0.1, 0.2, /* ... */]));

// With metadata
collection.insertWithObject('doc2', embedding, {
  title: 'Example Document',
  category: 'tutorial',
});
```

### Searching

```javascript
const results = collection.search(queryVector, 10);

for (const result of results) {
  console.log(`ID: ${result.id}, Distance: ${result.distance}`);
  console.log('Metadata:', result.getMetadata());
}
```

### Filtered Search

```javascript
const filter = JSON.stringify({
  category: 'electronics',
  price: { $lt: 100 }
});

const results = collection.searchWithFilter(queryVector, 10, filter);
```

Supported operators: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`, `$not`.

### Persistence

```javascript
// Serialize to bytes
const bytes = collection.toBytes();

// Store in IndexedDB, localStorage, or send to server
// ...

// Restore later
const restored = WasmCollection.fromBytes(new Uint8Array(savedBytes));
```

---

## API Reference

### WasmCollection

| Method | Signature | Description |
|--------|-----------|-------------|
| `insert` | `(id, vector, metadata?)` | Insert with JSON metadata string |
| `insertWithObject` | `(id, vector, metadata?)` | Insert with JS object metadata |
| `insertBatch` | `(ids[], vectors[], metadata[]?)` | Batch insert |
| `search` | `(query, k)` | Find k nearest neighbors |
| `searchWithFilter` | `(query, k, filter)` | Filtered search |
| `get` | `(id)` | Get by ID |
| `contains` | `(id)` | Check existence |
| `delete` | `(id)` | Delete by ID |
| `setEfSearch` | `(ef)` | Set search quality parameter |
| `toBytes` | `()` | Serialize to bytes |
| `fromBytes` | `(bytes)` | *(static)* Deserialize from bytes |

### SearchResult

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Vector ID |
| `distance` | `number` | Distance from query |
| `getMetadata()` | `object \| null` | Parsed metadata |

---

## Performance Tips

1. **Use `Float32Array`** for vectors — avoids internal conversion.
2. **Batch inserts** with `insertBatch()` for bulk loading.
3. **Tune `ef_search`** — lower (20) for speed, higher (100) for recall.
4. **Use Web Workers** to keep the main thread responsive.
5. **Streaming WASM instantiation** for faster loading:

```javascript
const module = await WebAssembly.compileStreaming(fetch('needle_bg.wasm'));
await init(module);
```

---

## Edge Runtime Deployment

### Cloudflare Workers

```javascript
import init, { WasmCollection } from '@anthropic/needle-wasm';

let collection = null;

export default {
  async fetch(request, env) {
    if (!collection) {
      await init();
      collection = new WasmCollection('vectors', 384);
    }

    const { query, k } = await request.json();
    const results = collection.search(new Float32Array(query), k || 10);

    return Response.json({
      results: results.map(r => ({
        id: r.id,
        distance: r.distance,
        metadata: r.getMetadata()
      }))
    });
  }
};
```

### Vercel Edge Functions

```typescript
export const config = { runtime: 'edge' };

export default async function handler(req) {
  await init();
  const collection = new WasmCollection('vectors', 384);
  const { query, k = 10 } = await req.json();
  const results = collection.search(new Float32Array(query), k);
  return new Response(JSON.stringify({ results }));
}
```

---

## See Also

- [JavaScript Bindings](/docs/bindings/javascript) — JS SDK with IndexedDB persistence
- [API Reference](/docs/api-reference) — Complete method documentation
- [Deployment Guide](/docs/advanced/deployment) — Docker and Kubernetes deployment
