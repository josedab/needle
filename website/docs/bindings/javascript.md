---
sidebar_position: 3
---

# JavaScript / WebAssembly

Needle provides JavaScript bindings through WebAssembly, enabling vector search in browsers and Node.js.

## Installation

### Node.js

```bash
npm install @anthropic/needle
```

### Browser (CDN)

```html
<script type="module">
import init, { Database, DistanceFunction } from 'https://unpkg.com/@anthropic/needle/web/needle.js';

await init();
// Now use Needle...
</script>
```

### Bundlers (Webpack, Vite, etc.)

```javascript
import init, { Database, DistanceFunction } from '@anthropic/needle';

// Initialize WASM module
await init();
```

## Quick Start

```javascript
import init, { Database, DistanceFunction } from '@anthropic/needle';

async function main() {
  // Initialize WASM
  await init();

  // Create database
  const db = Database.inMemory();

  // Create collection
  db.createCollection('documents', 384, DistanceFunction.Cosine);

  // Get collection
  const collection = db.collection('documents');

  // Insert vectors
  const embedding = new Float32Array(384).fill(0.1);
  collection.insert('doc1', embedding, { title: 'Hello World' });

  // Search
  const query = new Float32Array(384).fill(0.1);
  const results = collection.search(query, 10);

  for (const result of results) {
    console.log(`ID: ${result.id}, Distance: ${result.distance}`);
  }
}

main();
```

## Database API

### Creating Databases

```javascript
import { Database } from '@anthropic/needle';

// In-memory database (browser and Node.js)
const db = Database.inMemory();

// Node.js only: file-based database
const db = await Database.open('vectors.needle');

// Browser: IndexedDB persistence
const db = await Database.openIndexedDB('my-vectors');
```

### Collection Management

```javascript
import { DistanceFunction, CollectionConfig } from '@anthropic/needle';

// Create collection
db.createCollection('docs', 384, DistanceFunction.Cosine);

// With custom config
const config = new CollectionConfig({
  dimensions: 384,
  distance: DistanceFunction.Cosine,
  hnswM: 32,
  hnswEfConstruction: 400
});
db.createCollectionWithConfig('high_quality', config);

// List collections
const names = db.listCollections();

// Check existence
if (db.collectionExists('docs')) {
  const collection = db.collection('docs');
}

// Delete collection
db.deleteCollection('old_collection');
```

### Persistence

```javascript
// Node.js: Save to file
await db.save();

// Browser: Save to IndexedDB
await db.saveToIndexedDB();

// Export as Uint8Array (for download or upload)
const bytes = db.toBytes();

// Import from Uint8Array
const db = Database.fromBytes(bytes);
```

## Collection API

### Vector Operations

```javascript
const collection = db.collection('documents');

// Insert - use Float32Array for best performance
const embedding = new Float32Array([0.1, 0.2, ...]);
collection.insert('doc1', embedding, { title: 'Hello' });

// Also accepts regular arrays (converted internally)
collection.insert('doc2', [0.1, 0.2, ...], { title: 'World' });

// Get by ID
const entry = collection.get('doc1');
console.log(entry.vector);  // Float32Array
console.log(entry.metadata); // { title: 'Hello' }

// Check existence
if (collection.exists('doc1')) {
  console.log('Found!');
}

// Delete
collection.delete('doc1');

// Count
const count = collection.count();

// Clear all
collection.clear();
```

### Searching

```javascript
// Basic search
const query = new Float32Array(384).fill(0.1);
const results = collection.search(query, 10);

// With filter
const results = collection.search(query, 10, {
  category: 'programming'
});

// With custom ef_search
const results = collection.searchWithParams(query, 10, null, 100);

// Process results
for (const result of results) {
  console.log(`ID: ${result.id}`);
  console.log(`Distance: ${result.distance}`);
  console.log(`Metadata: ${JSON.stringify(result.metadata)}`);
}
```

### Filtering

```javascript
// Equality
const results = collection.search(query, 10, { status: 'active' });

// Comparison
const results = collection.search(query, 10, {
  price: { $gt: 10, $lt: 100 }
});

// In array
const results = collection.search(query, 10, {
  category: { $in: ['books', 'movies'] }
});

// Logical operators
const results = collection.search(query, 10, {
  $or: [
    { category: 'electronics' },
    { price: { $lt: 50 } }
  ]
});
```

## Browser Usage

### With Transformers.js

```javascript
import { pipeline } from '@xenova/transformers';
import init, { Database, DistanceFunction } from '@anthropic/needle';

// Initialize
await init();
const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// Create database
const db = Database.inMemory();
db.createCollection('documents', 384, DistanceFunction.Cosine);
const collection = db.collection('documents');

// Index documents
async function indexDocument(id, text, metadata) {
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  const embedding = new Float32Array(output.data);
  collection.insert(id, embedding, { ...metadata, text });
}

await indexDocument('doc1', 'Introduction to machine learning', { category: 'AI' });
await indexDocument('doc2', 'JavaScript programming guide', { category: 'Programming' });

// Search
async function search(query, k = 5) {
  const output = await embedder(query, { pooling: 'mean', normalize: true });
  const embedding = new Float32Array(output.data);
  return collection.search(embedding, k);
}

const results = await search('ML tutorial');
console.log(results);
```

### IndexedDB Persistence

```javascript
import init, { Database } from '@anthropic/needle';

await init();

// Open or create IndexedDB-backed database
const db = await Database.openIndexedDB('my-app-vectors');

// Use normally
db.createCollection('documents', 384, DistanceFunction.Cosine);
const collection = db.collection('documents');
collection.insert('doc1', embedding, metadata);

// Changes are automatically persisted
// Or manually save:
await db.saveToIndexedDB();

// Close when done
db.close();
```

### Web Worker Usage

```javascript
// worker.js
import init, { Database, DistanceFunction } from '@anthropic/needle';

let db;

self.onmessage = async (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'init':
      await init();
      db = Database.inMemory();
      db.createCollection('documents', 384, DistanceFunction.Cosine);
      self.postMessage({ type: 'ready' });
      break;

    case 'insert':
      const collection = db.collection('documents');
      collection.insert(payload.id, new Float32Array(payload.vector), payload.metadata);
      self.postMessage({ type: 'inserted', id: payload.id });
      break;

    case 'search':
      const coll = db.collection('documents');
      const results = coll.search(new Float32Array(payload.query), payload.k);
      self.postMessage({ type: 'results', results });
      break;
  }
};

// main.js
const worker = new Worker('worker.js', { type: 'module' });

worker.postMessage({ type: 'init' });

worker.onmessage = (e) => {
  if (e.data.type === 'ready') {
    // Index documents
    worker.postMessage({
      type: 'insert',
      payload: { id: 'doc1', vector: [...embedding], metadata: { title: 'Test' } }
    });
  }

  if (e.data.type === 'results') {
    console.log('Search results:', e.data.results);
  }
};
```

## Node.js Usage

### File-Based Database

```javascript
import { Database, DistanceFunction } from '@anthropic/needle';

// Open file-based database
const db = await Database.open('vectors.needle');

// Create collection
db.createCollection('documents', 384, DistanceFunction.Cosine);

// Use collection
const collection = db.collection('documents');
collection.insert('doc1', embedding, { title: 'Test' });

// Save changes
await db.save();
```

### With OpenAI

```javascript
import OpenAI from 'openai';
import { Database, DistanceFunction } from '@anthropic/needle';

const openai = new OpenAI();
const db = await Database.open('openai-vectors.needle');
db.createCollection('documents', 1536, DistanceFunction.Cosine);
const collection = db.collection('documents');

async function getEmbedding(text) {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text
  });
  return new Float32Array(response.data[0].embedding);
}

// Index
const docs = ['Document 1...', 'Document 2...'];
for (let i = 0; i < docs.length; i++) {
  const embedding = await getEmbedding(docs[i]);
  collection.insert(`doc${i}`, embedding, { content: docs[i] });
}
await db.save();

// Search
const query = await getEmbedding('search query');
const results = collection.search(query, 5);
```

## TypeScript Support

Full TypeScript definitions are included:

```typescript
import init, {
  Database,
  Collection,
  DistanceFunction,
  CollectionConfig,
  SearchResult,
  Filter
} from '@anthropic/needle';

async function example(): Promise<void> {
  await init();

  const db: Database = Database.inMemory();
  db.createCollection('documents', 384, DistanceFunction.Cosine);

  const collection: Collection = db.collection('documents');

  const embedding: Float32Array = new Float32Array(384).fill(0.1);
  collection.insert('doc1', embedding, { title: 'Hello' });

  const query: Float32Array = new Float32Array(384).fill(0.1);
  const filter: Filter = { category: 'test' };
  const results: SearchResult[] = collection.search(query, 10, filter);

  for (const result of results) {
    console.log(result.id, result.distance, result.metadata);
  }
}
```

## Performance Tips

### Use Float32Array

```javascript
// Good - uses memory directly
const embedding = new Float32Array([0.1, 0.2, 0.3, ...]);
collection.insert('doc1', embedding, {});

// Slower - requires conversion
collection.insert('doc2', [0.1, 0.2, 0.3, ...], {});
```

### Batch Operations

```javascript
// Index multiple documents
for (const doc of documents) {
  collection.insert(doc.id, doc.embedding, doc.metadata);
}
// Save once at the end
await db.save();
```

### Web Worker for Heavy Operations

Move indexing and searching to a Web Worker to keep the main thread responsive.

### Lazy Initialization

```javascript
let db = null;

async function getDatabase() {
  if (!db) {
    await init();
    db = await Database.openIndexedDB('my-vectors');
  }
  return db;
}
```

## Bundle Size

The WASM module is approximately 800KB gzipped. For production:

```javascript
// Dynamic import for code splitting
const needle = await import('@anthropic/needle');
await needle.default(); // init()
```

## Next Steps

- [Swift/Kotlin Bindings](/docs/bindings/swift-kotlin)
- [API Reference](/docs/api-reference)
- [Semantic Search Guide](/docs/guides/semantic-search)
