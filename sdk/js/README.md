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
