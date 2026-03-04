/**
 * Deno Deploy adapter for Needle Edge.
 *
 * @example
 * ```typescript
 * import { NeedleEdge } from '@anthropic/needle-edge';
 * import { DenoKVAdapter } from '@anthropic/needle-edge/deno';
 *
 * const kv = await Deno.openKv();
 * const needle = new NeedleEdge({
 *   adapter: new DenoKVAdapter(kv),
 *   dimensions: 384,
 * });
 * ```
 */

import type { StorageAdapter } from './index';

/** Deno KV storage adapter. */
export class DenoKVAdapter implements StorageAdapter {
  private kv: Deno.Kv;
  private prefix: string[];

  constructor(kv: Deno.Kv, prefix: string[] = ['needle']) {
    this.kv = kv;
    this.prefix = prefix;
  }

  async get(key: string): Promise<ArrayBuffer | null> {
    const result = await this.kv.get<Uint8Array>([...this.prefix, key]);
    return result.value ? result.value.buffer : null;
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    await this.kv.set([...this.prefix, key], new Uint8Array(value));
  }

  async delete(key: string): Promise<void> {
    await this.kv.delete([...this.prefix, key]);
  }

  async list(prefix: string): Promise<string[]> {
    const keys: string[] = [];
    const iter = this.kv.list<unknown>({ prefix: [...this.prefix, prefix] });
    for await (const entry of iter) {
      const key = entry.key[entry.key.length - 1];
      if (typeof key === 'string') {
        keys.push(key);
      }
    }
    return keys;
  }
}

// Deno KV type declarations
declare global {
  namespace Deno {
    interface Kv {
      get<T>(key: unknown[]): Promise<{ value: T | null }>;
      set(key: unknown[], value: unknown): Promise<void>;
      delete(key: unknown[]): Promise<void>;
      list<T>(options: { prefix: unknown[] }): AsyncIterable<{ key: unknown[]; value: T }>;
    }
    function openKv(): Promise<Kv>;
  }
}
