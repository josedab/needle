/**
 * Bun runtime adapter for Needle Edge.
 *
 * @example
 * ```typescript
 * import { NeedleEdge } from '@anthropic/needle-edge';
 * import { BunFileAdapter } from '@anthropic/needle-edge/bun';
 *
 * const needle = new NeedleEdge({
 *   adapter: new BunFileAdapter('./data'),
 *   dimensions: 384,
 * });
 * ```
 */

import type { StorageAdapter } from './index';

/** Bun file-system storage adapter using Bun's fast I/O. */
export class BunFileAdapter implements StorageAdapter {
  private basePath: string;

  constructor(basePath: string = './needle-data') {
    this.basePath = basePath;
  }

  async get(key: string): Promise<ArrayBuffer | null> {
    const path = `${this.basePath}/${key}`;
    try {
      const file = Bun.file(path);
      if (await file.exists()) {
        return file.arrayBuffer();
      }
      return null;
    } catch {
      return null;
    }
  }

  async put(key: string, value: ArrayBuffer): Promise<void> {
    const path = `${this.basePath}/${key}`;
    const dir = path.substring(0, path.lastIndexOf('/'));
    await Bun.write(path, value);
  }

  async delete(key: string): Promise<void> {
    const path = `${this.basePath}/${key}`;
    try {
      const { unlink } = await import('node:fs/promises');
      await unlink(path);
    } catch {
      // File may not exist
    }
  }

  async list(prefix: string): Promise<string[]> {
    try {
      const { readdir } = await import('node:fs/promises');
      const files = await readdir(this.basePath);
      return files.filter(f => f.startsWith(prefix));
    } catch {
      return [];
    }
  }
}

// Bun type declarations
declare global {
  const Bun: {
    file(path: string): BunFile;
    write(path: string, data: ArrayBuffer | string): Promise<number>;
  };
  interface BunFile {
    exists(): Promise<boolean>;
    arrayBuffer(): Promise<ArrayBuffer>;
    text(): Promise<string>;
  }
}
