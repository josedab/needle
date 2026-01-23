# Needle WASM/JavaScript Guide

This guide covers using Needle as a WebAssembly module in JavaScript/TypeScript applications. Needle can run entirely in the browser, Node.js, Deno, and edge runtimes like Cloudflare Workers.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [API Reference](#api-reference)
- [Framework Integration](#framework-integration)
- [Performance Tips](#performance-tips)
- [Edge Runtime Deployment](#edge-runtime-deployment)
- [TypeScript Support](#typescript-support)

---

## Getting Started

### Prerequisites

- Node.js 16+ (for build tools)
- npm or yarn
- For browser: Modern browser with WebAssembly support

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
- `needle_bg.wasm` - The WebAssembly binary
- `needle.js` - JavaScript glue code
- `needle.d.ts` - TypeScript type definitions
- `package.json` - npm package manifest

---

## Installation

### npm (Node.js / Bundlers)

```bash
npm install @anthropic/needle-wasm
# or
yarn add @anthropic/needle-wasm
```

### CDN (Browser)

```html
<script type="module">
  import init, { WasmCollection } from 'https://unpkg.com/@anthropic/needle-wasm/needle.js';

  await init();  // Initialize WASM module

  const collection = new WasmCollection('documents', 384);
</script>
```

### Deno

```typescript
import init, { WasmCollection } from "https://esm.sh/@anthropic/needle-wasm";

await init();
```

---

## Basic Usage

### Creating a Collection

```javascript
import init, { WasmCollection } from '@anthropic/needle-wasm';

// Initialize the WASM module (required before first use)
await init();

// Create a collection with 384 dimensions (e.g., for sentence-transformers)
const collection = new WasmCollection('my_documents', 384, 'cosine');

console.log(collection.name);        // 'my_documents'
console.log(collection.dimensions);  // 384
console.log(collection.length);      // 0
```

**Distance functions:**
- `'cosine'` (default) - Cosine similarity
- `'euclidean'` or `'l2'` - Euclidean distance
- `'dot'` or `'dotproduct'` - Dot product (inner product)
- `'manhattan'` or `'l1'` - Manhattan distance

### Inserting Vectors

```javascript
// Insert a single vector
collection.insert('doc1', new Float32Array([0.1, 0.2, 0.3, /* ... 384 dims */]));

// Insert with metadata (as JSON string)
collection.insert('doc2', embedding, JSON.stringify({
  title: 'Example Document',
  category: 'tutorial',
  timestamp: Date.now()
}));

// Insert with metadata as JavaScript object
collection.insertWithObject('doc3', embedding, {
  title: 'Another Document',
  tags: ['javascript', 'wasm']
});
```

### Batch Insertion

For better performance with many vectors:

```javascript
const ids = ['doc1', 'doc2', 'doc3'];
const vectors = [
  new Float32Array([0.1, 0.2, ...]),
  new Float32Array([0.3, 0.4, ...]),
  new Float32Array([0.5, 0.6, ...])
];
const metadata = [
  JSON.stringify({ title: 'Doc 1' }),
  JSON.stringify({ title: 'Doc 2' }),
  JSON.stringify({ title: 'Doc 3' })
];

collection.insertBatch(ids, vectors, metadata);
```

### Searching

```javascript
// Basic search - find 10 nearest neighbors
const queryVector = new Float32Array([0.15, 0.25, ...]);
const results = collection.search(queryVector, 10);

for (const result of results) {
  console.log(`ID: ${result.id}, Distance: ${result.distance}`);

  // Get metadata as parsed object
  const meta = result.getMetadata();
  console.log(`Title: ${meta.title}`);
}
```

### Filtered Search

```javascript
// Search with MongoDB-style filter
const filter = JSON.stringify({
  category: 'electronics',
  price: { $lt: 100 }
});

const results = collection.searchWithFilter(queryVector, 10, filter);
```

**Supported filter operators:**
- `$eq` - Equals
- `$ne` - Not equals
- `$gt`, `$gte` - Greater than (or equal)
- `$lt`, `$lte` - Less than (or equal)
- `$in` - In array
- `$nin` - Not in array
- `$and`, `$or` - Logical operators
- `$not` - Negation

### Getting and Deleting Vectors

```javascript
// Check if vector exists
if (collection.contains('doc1')) {
  // Get vector and metadata
  const data = collection.get('doc1');
  console.log(data.vector);    // Float32Array
  console.log(data.metadata);  // Parsed object
}

// Delete a vector
const deleted = collection.delete('doc1');
console.log(`Deleted: ${deleted}`);
```

### Persistence

```javascript
// Serialize to bytes (for storage)
const bytes = collection.toBytes();

// Save to IndexedDB
const db = await openDB('needle', 1, {
  upgrade(db) {
    db.createObjectStore('collections');
  }
});
await db.put('collections', bytes, 'my_documents');

// Later: Load from bytes
const savedBytes = await db.get('collections', 'my_documents');
const loadedCollection = WasmCollection.fromBytes(new Uint8Array(savedBytes));
```

---

## API Reference

### WasmCollection

#### Constructor

```typescript
new WasmCollection(
  name: string,
  dimensions: number,
  distance?: 'cosine' | 'euclidean' | 'dot' | 'manhattan'
): WasmCollection
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `string` | Collection name |
| `dimensions` | `number` | Vector dimensions |
| `length` | `number` | Number of vectors |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `insert` | `(id: string, vector: Float32Array, metadata?: string) => void` | Insert with JSON metadata string |
| `insertWithObject` | `(id: string, vector: Float32Array, metadata?: object) => void` | Insert with JS object metadata |
| `insertBatch` | `(ids: string[], vectors: Float32Array[], metadata?: string[]) => void` | Batch insert |
| `search` | `(query: Float32Array, k: number) => SearchResult[]` | Find k nearest neighbors |
| `searchWithFilter` | `(query: Float32Array, k: number, filter: string) => SearchResult[]` | Filtered search |
| `get` | `(id: string) => { vector: Float32Array, metadata: object } \| null` | Get by ID |
| `contains` | `(id: string) => boolean` | Check existence |
| `delete` | `(id: string) => boolean` | Delete by ID |
| `isEmpty` | `() => boolean` | Check if empty |
| `setEfSearch` | `(ef: number) => void` | Set search quality parameter |
| `toBytes` | `() => Uint8Array` | Serialize to bytes |
| `fromBytes` | `(bytes: Uint8Array) => WasmCollection` | Deserialize from bytes |

### SearchResult

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `string` | Vector ID |
| `distance` | `number` | Distance from query |
| `metadata` | `string \| null` | Raw JSON metadata |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `getMetadata` | `() => object \| null` | Parsed metadata object |

---

## Framework Integration

### React

```tsx
import { useState, useEffect, useCallback } from 'react';
import init, { WasmCollection } from '@anthropic/needle-wasm';

// Hook for managing Needle collection
function useNeedle(name: string, dimensions: number) {
  const [collection, setCollection] = useState<WasmCollection | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function initialize() {
      await init();
      if (mounted) {
        setCollection(new WasmCollection(name, dimensions));
        setLoading(false);
      }
    }

    initialize();
    return () => { mounted = false; };
  }, [name, dimensions]);

  const search = useCallback((query: Float32Array, k: number) => {
    if (!collection) return [];
    return collection.search(query, k);
  }, [collection]);

  return { collection, loading, search };
}

// Usage in component
function SearchComponent() {
  const { collection, loading, search } = useNeedle('products', 384);
  const [results, setResults] = useState([]);

  const handleSearch = async (queryText: string) => {
    // Get embedding from your embedding service
    const embedding = await getEmbedding(queryText);
    const searchResults = search(new Float32Array(embedding), 10);
    setResults(searchResults);
  };

  if (loading) return <div>Loading vector database...</div>;

  return (
    <div>
      <input onChange={(e) => handleSearch(e.target.value)} />
      {results.map(r => (
        <div key={r.id}>
          {r.id}: {r.distance.toFixed(4)}
        </div>
      ))}
    </div>
  );
}
```

### Vue 3

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue';
import init, { WasmCollection } from '@anthropic/needle-wasm';

const collection = ref<WasmCollection | null>(null);
const results = ref<any[]>([]);
const loading = ref(true);

onMounted(async () => {
  await init();
  collection.value = new WasmCollection('products', 384);
  loading.value = false;
});

async function search(query: string) {
  if (!collection.value) return;

  const embedding = await getEmbedding(query);
  results.value = collection.value.search(new Float32Array(embedding), 10);
}
</script>

<template>
  <div v-if="loading">Loading...</div>
  <div v-else>
    <input @input="e => search(e.target.value)" />
    <div v-for="r in results" :key="r.id">
      {{ r.id }}: {{ r.distance.toFixed(4) }}
    </div>
  </div>
</template>
```

### Next.js

```typescript
// app/api/search/route.ts (App Router)
import { NextRequest, NextResponse } from 'next/server';
import init, { WasmCollection } from '@anthropic/needle-wasm';

let collection: WasmCollection | null = null;

async function getCollection() {
  if (!collection) {
    await init();
    collection = new WasmCollection('documents', 384);
    // Load persisted data if available
  }
  return collection;
}

export async function POST(request: NextRequest) {
  const { query, k = 10 } = await request.json();

  const coll = await getCollection();
  const embedding = await getEmbedding(query);  // Your embedding service
  const results = coll.search(new Float32Array(embedding), k);

  return NextResponse.json({
    results: results.map(r => ({
      id: r.id,
      distance: r.distance,
      metadata: r.getMetadata()
    }))
  });
}
```

### SvelteKit

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import init, { WasmCollection } from '@anthropic/needle-wasm';

  let collection: WasmCollection | null = null;
  let results: any[] = [];
  let loading = true;

  onMount(async () => {
    await init();
    collection = new WasmCollection('documents', 384);
    loading = false;
  });

  async function handleSearch(event: Event) {
    const query = (event.target as HTMLInputElement).value;
    if (!collection || !query) return;

    const embedding = await getEmbedding(query);
    results = collection.search(new Float32Array(embedding), 10);
  }
</script>

{#if loading}
  <p>Loading...</p>
{:else}
  <input on:input={handleSearch} placeholder="Search..." />

  {#each results as result}
    <div>{result.id}: {result.distance.toFixed(4)}</div>
  {/each}
{/if}
```

---

## Performance Tips

### 1. Use Float32Array

Always use `Float32Array` for vectors instead of regular arrays:

```javascript
// Good - direct memory layout
const vector = new Float32Array([0.1, 0.2, 0.3]);

// Bad - requires conversion
const vector = [0.1, 0.2, 0.3];  // Will be converted internally
```

### 2. Batch Operations

Insert vectors in batches rather than one at a time:

```javascript
// Good - single batch operation
collection.insertBatch(ids, vectors, metadata);

// Bad - many individual operations
for (let i = 0; i < vectors.length; i++) {
  collection.insert(ids[i], vectors[i], metadata[i]);
}
```

### 3. Tune ef_search

Balance search quality vs. speed:

```javascript
// Faster but lower recall
collection.setEfSearch(20);

// Slower but higher recall
collection.setEfSearch(100);

// Default is 50 (balanced)
```

### 4. Use Web Workers

Move heavy operations off the main thread:

```javascript
// worker.js
import init, { WasmCollection } from '@anthropic/needle-wasm';

let collection = null;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  if (type === 'init') {
    await init();
    collection = new WasmCollection(payload.name, payload.dimensions);
    self.postMessage({ type: 'ready' });
  }

  if (type === 'search') {
    const results = collection.search(
      new Float32Array(payload.query),
      payload.k
    );
    self.postMessage({ type: 'results', payload: results });
  }
};

// main.js
const worker = new Worker('worker.js', { type: 'module' });
worker.postMessage({ type: 'init', payload: { name: 'docs', dimensions: 384 } });

worker.onmessage = (e) => {
  if (e.data.type === 'results') {
    console.log(e.data.payload);
  }
};
```

### 5. Memory Management

The WASM module manages its own memory. For very large collections:

```javascript
// Serialize and clear periodically to free memory
const bytes = collection.toBytes();
// Store bytes to IndexedDB or similar

// Recreate collection when needed
collection = WasmCollection.fromBytes(bytes);
```

### 6. Streaming WASM Instantiation

For faster loading in browsers:

```javascript
// Use streaming instantiation
const wasmModule = await WebAssembly.compileStreaming(
  fetch('needle_bg.wasm')
);

// Then initialize with the compiled module
await init(wasmModule);
```

---

## Edge Runtime Deployment

### Cloudflare Workers

```javascript
// src/index.js
import init, { WasmCollection } from '@anthropic/needle-wasm';

let collection = null;

export default {
  async fetch(request, env) {
    if (!collection) {
      await init();
      collection = new WasmCollection('vectors', 384);

      // Load from KV if persisted
      const saved = await env.NEEDLE_KV.get('collection', 'arrayBuffer');
      if (saved) {
        collection = WasmCollection.fromBytes(new Uint8Array(saved));
      }
    }

    const url = new URL(request.url);

    if (url.pathname === '/search' && request.method === 'POST') {
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

    if (url.pathname === '/insert' && request.method === 'POST') {
      const { id, vector, metadata } = await request.json();
      collection.insertWithObject(id, new Float32Array(vector), metadata);

      // Persist to KV
      const bytes = collection.toBytes();
      await env.NEEDLE_KV.put('collection', bytes);

      return Response.json({ success: true });
    }

    return new Response('Not found', { status: 404 });
  }
};
```

```toml
# wrangler.toml
name = "needle-worker"
main = "src/index.js"
compatibility_date = "2024-01-01"

[build]
command = "npm run build"

[[kv_namespaces]]
binding = "NEEDLE_KV"
id = "your-kv-namespace-id"
```

### Vercel Edge Functions

```typescript
// api/search.ts
import { NextRequest } from 'next/server';
import init, { WasmCollection } from '@anthropic/needle-wasm';

export const config = {
  runtime: 'edge',
};

let collection: WasmCollection | null = null;

export default async function handler(req: NextRequest) {
  if (!collection) {
    await init();
    collection = new WasmCollection('vectors', 384);
  }

  const { query, k = 10 } = await req.json();
  const results = collection.search(new Float32Array(query), k);

  return new Response(JSON.stringify({
    results: results.map(r => ({
      id: r.id,
      distance: r.distance,
      metadata: r.getMetadata()
    }))
  }), {
    headers: { 'Content-Type': 'application/json' }
  });
}
```

### Deno Deploy

```typescript
// main.ts
import init, { WasmCollection } from "https://esm.sh/@anthropic/needle-wasm";

let collection: WasmCollection | null = null;

Deno.serve(async (req) => {
  if (!collection) {
    await init();
    collection = new WasmCollection("vectors", 384);
  }

  const url = new URL(req.url);

  if (url.pathname === "/search" && req.method === "POST") {
    const { query, k } = await req.json();
    const results = collection.search(new Float32Array(query), k || 10);

    return new Response(JSON.stringify({
      results: results.map(r => ({
        id: r.id,
        distance: r.distance,
        metadata: r.getMetadata()
      }))
    }), {
      headers: { "Content-Type": "application/json" }
    });
  }

  return new Response("Not found", { status: 404 });
});
```

---

## TypeScript Support

### Type Definitions

The package includes TypeScript definitions. For custom types:

```typescript
// types.ts
import type { WasmCollection, SearchResult } from '@anthropic/needle-wasm';

export interface Document {
  id: string;
  title: string;
  content: string;
  embedding: Float32Array;
  metadata: DocumentMetadata;
}

export interface DocumentMetadata {
  category: string;
  timestamp: number;
  author?: string;
  tags?: string[];
}

export interface SearchResultWithMetadata extends SearchResult {
  parsedMetadata: DocumentMetadata;
}

// Helper function with proper types
export function searchDocuments(
  collection: WasmCollection,
  query: Float32Array,
  k: number = 10,
  filter?: Partial<DocumentMetadata>
): SearchResultWithMetadata[] {
  let results: SearchResult[];

  if (filter) {
    results = collection.searchWithFilter(query, k, JSON.stringify(filter));
  } else {
    results = collection.search(query, k);
  }

  return results.map(r => ({
    ...r,
    parsedMetadata: r.getMetadata() as DocumentMetadata
  }));
}
```

### Strict Null Checks

```typescript
// Handle potential null values
function getDocument(collection: WasmCollection, id: string): Document | null {
  const result = collection.get(id);
  if (!result) return null;

  return {
    id,
    vector: result.vector,
    metadata: result.metadata ?? {}
  };
}
```

---

## Common Patterns

### Semantic Search with Embeddings API

```typescript
import init, { WasmCollection } from '@anthropic/needle-wasm';

// Initialize
await init();
const collection = new WasmCollection('documents', 1536);  // OpenAI embedding size

// Embedding function using OpenAI
async function getEmbedding(text: string): Promise<Float32Array> {
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: text
    })
  });

  const data = await response.json();
  return new Float32Array(data.data[0].embedding);
}

// Index documents
async function indexDocument(id: string, text: string, metadata: object) {
  const embedding = await getEmbedding(text);
  collection.insertWithObject(id, embedding, { ...metadata, text });
}

// Semantic search
async function semanticSearch(query: string, k: number = 10) {
  const queryEmbedding = await getEmbedding(query);
  const results = collection.search(queryEmbedding, k);

  return results.map(r => ({
    id: r.id,
    score: 1 - r.distance,  // Convert distance to similarity
    ...r.getMetadata()
  }));
}
```

### RAG (Retrieval Augmented Generation)

```typescript
async function ragQuery(question: string): Promise<string> {
  // 1. Retrieve relevant context
  const queryEmbedding = await getEmbedding(question);
  const results = collection.search(queryEmbedding, 5);

  // 2. Build context from results
  const context = results
    .map(r => r.getMetadata().text)
    .join('\n\n---\n\n');

  // 3. Generate answer with context
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': ANTHROPIC_API_KEY,
      'Content-Type': 'application/json',
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: 'claude-3-sonnet-20240229',
      max_tokens: 1024,
      messages: [{
        role: 'user',
        content: `Based on the following context, answer the question.

Context:
${context}

Question: ${question}`
      }]
    })
  });

  const data = await response.json();
  return data.content[0].text;
}
```

### IndexedDB Persistence

```typescript
import { openDB, IDBPDatabase } from 'idb';

class PersistentCollection {
  private collection: WasmCollection;
  private db: IDBPDatabase;
  private collectionName: string;

  static async create(name: string, dimensions: number): Promise<PersistentCollection> {
    await init();

    const db = await openDB('needle', 1, {
      upgrade(db) {
        db.createObjectStore('collections');
      }
    });

    // Try to load existing collection
    const saved = await db.get('collections', name);
    let collection: WasmCollection;

    if (saved) {
      collection = WasmCollection.fromBytes(new Uint8Array(saved));
    } else {
      collection = new WasmCollection(name, dimensions);
    }

    return new PersistentCollection(collection, db, name);
  }

  private constructor(collection: WasmCollection, db: IDBPDatabase, name: string) {
    this.collection = collection;
    this.db = db;
    this.collectionName = name;
  }

  async insert(id: string, vector: Float32Array, metadata?: object) {
    this.collection.insertWithObject(id, vector, metadata ?? null);
    await this.persist();
  }

  search(query: Float32Array, k: number) {
    return this.collection.search(query, k);
  }

  private async persist() {
    const bytes = this.collection.toBytes();
    await this.db.put('collections', bytes, this.collectionName);
  }
}

// Usage
const collection = await PersistentCollection.create('my_docs', 384);
await collection.insert('doc1', embedding, { title: 'Example' });
const results = collection.search(queryVector, 10);
```

---

## Troubleshooting

### "WASM module not initialized"

Always call `init()` before using any other functions:

```javascript
import init, { WasmCollection } from '@anthropic/needle-wasm';

// Must await init() first
await init();

// Now safe to use
const collection = new WasmCollection('docs', 384);
```

### Memory issues in browser

For large collections, consider:
1. Using smaller dimensions if possible
2. Periodically serializing and clearing
3. Using a Web Worker to isolate memory

### "Lock poisoned" error

This occurs if a previous operation panicked. The collection is no longer usable:

```javascript
try {
  collection.insert(id, vector);
} catch (e) {
  if (e.message.includes('Lock poisoned')) {
    // Recreate the collection
    collection = new WasmCollection('docs', 384);
    // Reload data from persistence
  }
}
```

### Slow initial load

Use streaming instantiation and preload the WASM module:

```html
<!-- Preload in HTML head -->
<link rel="preload" href="needle_bg.wasm" as="fetch" crossorigin>
```

```javascript
// Use streaming instantiation
const wasmModule = await WebAssembly.compileStreaming(
  fetch('needle_bg.wasm')
);
await init(wasmModule);
```
