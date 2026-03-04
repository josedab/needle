/**
 * Cloudflare Workers adapter for Needle Edge.
 *
 * @example
 * ```typescript
 * import { NeedleEdge } from '@anthropic/needle-edge';
 * import { CloudflareR2Adapter } from '@anthropic/needle-edge/cloudflare';
 *
 * export default {
 *   async fetch(request: Request, env: Env) {
 *     const needle = new NeedleEdge({
 *       adapter: new CloudflareR2Adapter(env.MY_BUCKET),
 *       dimensions: 384,
 *     });
 *     await needle.load();
 *     const results = await needle.search(queryVector, { k: 10 });
 *     return Response.json(results);
 *   }
 * };
 * ```
 */

import type { StorageAdapter } from './index';

/** Cloudflare R2 storage adapter. */
export class CloudflareR2Adapter implements StorageAdapter {
  private bucket: R2Bucket;
  private prefix: string;

  constructor(bucket: R2Bucket, prefix: string = 'needle/') {
    this.bucket = bucket;
    this.prefix = prefix;
  }

  async get(key: string): Promise<ArrayBuffer | null> {
    const obj = await this.bucket.get(this.prefix + key);
    return obj ? obj.arrayBuffer() : null;
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    await this.bucket.put(this.prefix + key, value);
  }

  async delete(key: string): Promise<void> {
    await this.bucket.delete(this.prefix + key);
  }

  async list(prefix: string): Promise<string[]> {
    const listed = await this.bucket.list({ prefix: this.prefix + prefix });
    return listed.objects.map(obj => obj.key.replace(this.prefix, ''));
  }
}

/** Cloudflare KV storage adapter. */
export class CloudflareKVAdapter implements StorageAdapter {
  private kv: KVNamespace;
  private prefix: string;

  constructor(kv: KVNamespace, prefix: string = 'needle:') {
    this.kv = kv;
    this.prefix = prefix;
  }

  async get(key: string): Promise<ArrayBuffer | null> {
    return this.kv.get(this.prefix + key, 'arrayBuffer');
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    await this.kv.put(this.prefix + key, value);
  }

  async delete(key: string): Promise<void> {
    await this.kv.delete(this.prefix + key);
  }

  async list(prefix: string): Promise<string[]> {
    const listed = await this.kv.list({ prefix: this.prefix + prefix });
    return listed.keys.map(k => k.name.replace(this.prefix, ''));
  }
}

// Cloudflare Workers type declarations
declare global {
  interface R2Bucket {
    get(key: string): Promise<R2ObjectBody | null>;
    put(key: string, value: ArrayBuffer | string): Promise<void>;
    delete(key: string): Promise<void>;
    list(options?: { prefix?: string }): Promise<{ objects: Array<{ key: string }> }>;
  }
  interface R2ObjectBody {
    arrayBuffer(): Promise<ArrayBuffer>;
  }
  interface KVNamespace {
    get(key: string, type: 'arrayBuffer'): Promise<ArrayBuffer | null>;
    put(key: string, value: ArrayBuffer | string): Promise<void>;
    delete(key: string): Promise<void>;
    list(options?: { prefix?: string }): Promise<{ keys: Array<{ name: string }> }>;
  }
}
