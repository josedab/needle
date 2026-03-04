# @anthropic/needle-edge

Edge runtime adapters for the [Needle](https://github.com/anthropics/needle) vector database.

Run vector search on Cloudflare Workers, Deno Deploy, and Bun with <50ms cold starts.

## Installation

```bash
npm install @anthropic/needle-edge
```

## Quick Start

```typescript
import { NeedleEdge } from '@anthropic/needle-edge';

const needle = new NeedleEdge({ dimensions: 384 });

await needle.insert('doc1', new Float32Array(384).fill(0.1), { title: 'Hello' });

const results = await needle.search(new Float32Array(384).fill(0.1), { k: 10 });
console.log(results); // [{ id: 'doc1', distance: 0, metadata: { title: 'Hello' } }]
```

## Platform Adapters

### Cloudflare Workers (R2)

```typescript
import { NeedleEdge } from '@anthropic/needle-edge';
import { CloudflareR2Adapter } from '@anthropic/needle-edge/cloudflare';

export default {
  async fetch(request: Request, env: Env) {
    const needle = new NeedleEdge({
      adapter: new CloudflareR2Adapter(env.MY_BUCKET),
      dimensions: 384,
    });
    await needle.load();
    const results = await needle.search(queryVector, { k: 10 });
    return Response.json(results);
  }
};
```

### Cloudflare Workers (KV)

```typescript
import { CloudflareKVAdapter } from '@anthropic/needle-edge/cloudflare';

const needle = new NeedleEdge({
  adapter: new CloudflareKVAdapter(env.MY_KV),
  dimensions: 384,
});
```

### Deno Deploy

```typescript
import { NeedleEdge } from '@anthropic/needle-edge';
import { DenoKVAdapter } from '@anthropic/needle-edge/deno';

const kv = await Deno.openKv();
const needle = new NeedleEdge({
  adapter: new DenoKVAdapter(kv),
  dimensions: 384,
});
```

### Bun

```typescript
import { NeedleEdge } from '@anthropic/needle-edge';
import { BunFileAdapter } from '@anthropic/needle-edge/bun';

const needle = new NeedleEdge({
  adapter: new BunFileAdapter('./data'),
  dimensions: 384,
});
```

## API

### `NeedleEdge`

| Method | Description |
|--------|-------------|
| `insert(id, vector, metadata?)` | Insert a vector |
| `search(query, options?)` | Search for nearest neighbors |
| `delete(id)` | Delete a vector |
| `save()` | Persist to storage adapter |
| `load()` | Load from storage adapter |
| `count` | Number of vectors |

### `SearchOptions`

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | `number` | `10` | Number of results |
| `filter` | `object` | — | Metadata filter |

## License

MIT
