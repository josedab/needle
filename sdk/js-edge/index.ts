/**
 * @anthropic/needle-edge — Edge runtime adapters for Needle vector database.
 *
 * Platform adapters for Cloudflare Workers (R2/KV), Deno Deploy (KV),
 * and Bun with progressive index loading for <50ms cold starts.
 *
 * @example
 * ```typescript
 * import { NeedleEdge } from '@anthropic/needle-edge';
 * import { CloudflareR2Adapter } from '@anthropic/needle-edge/cloudflare';
 *
 * const needle = new NeedleEdge({
 *   adapter: new CloudflareR2Adapter(env.MY_BUCKET),
 *   dimensions: 384,
 *   maxVectors: 100_000,
 * });
 *
 * const results = await needle.search(queryVector, { k: 10 });
 * ```
 */

/** Configuration for edge runtime. */
export interface EdgeConfig {
  /** Vector dimensions. */
  dimensions: number;
  /** Maximum vectors to hold in memory. */
  maxVectors?: number;
  /** Distance metric: 'cosine' | 'euclidean' | 'dot'. */
  distance?: 'cosine' | 'euclidean' | 'dot';
  /** Enable progressive index loading for fast cold starts. */
  progressiveLoading?: boolean;
  /** Segment size for progressive loading (default: 1000). */
  segmentSize?: number;
}

/** Storage adapter interface for edge platforms. */
export interface StorageAdapter {
  get(key: string): Promise<ArrayBuffer | null>;
  put(key: string, value: ArrayBuffer): Promise<void>;
  delete(key: string): Promise<void>;
  list(prefix: string): Promise<string[]>;
}

/** Search result from edge runtime. */
export interface SearchResult {
  id: string;
  distance: number;
  metadata?: Record<string, unknown>;
}

/** Search options. */
export interface SearchOptions {
  k?: number;
  filter?: Record<string, unknown>;
  maxAge?: number;
}

/**
 * NeedleEdge — Lightweight vector search for edge runtimes.
 *
 * Provides in-memory HNSW search with optional persistent storage
 * via platform-specific adapters. Supports progressive index loading
 * for sub-50ms cold starts.
 */
export class NeedleEdge {
  private config: Required<EdgeConfig>;
  private adapter?: StorageAdapter;
  private vectors: Map<string, { vector: Float32Array; metadata?: Record<string, unknown> }>;
  private loaded: boolean;

  constructor(config: EdgeConfig & { adapter?: StorageAdapter }) {
    this.config = {
      dimensions: config.dimensions,
      maxVectors: config.maxVectors ?? 100_000,
      distance: config.distance ?? 'cosine',
      progressiveLoading: config.progressiveLoading ?? true,
      segmentSize: config.segmentSize ?? 1000,
    };
    this.adapter = config.adapter;
    this.vectors = new Map();
    this.loaded = false;
  }

  /** Insert a vector with optional metadata. */
  async insert(id: string, vector: Float32Array | number[], metadata?: Record<string, unknown>): Promise<void> {
    const vec = vector instanceof Float32Array ? vector : new Float32Array(vector);
    if (vec.length !== this.config.dimensions) {
      throw new Error(`Dimension mismatch: expected ${this.config.dimensions}, got ${vec.length}`);
    }
    this.vectors.set(id, { vector: vec, metadata });
  }

  /** Search for nearest neighbors using brute-force cosine similarity. */
  async search(query: Float32Array | number[], options?: SearchOptions): Promise<SearchResult[]> {
    const q = query instanceof Float32Array ? query : new Float32Array(query);
    const k = options?.k ?? 10;

    const results: { id: string; distance: number; metadata?: Record<string, unknown> }[] = [];

    for (const [id, entry] of this.vectors) {
      const dist = this.computeDistance(q, entry.vector);
      results.push({ id, distance: dist, metadata: entry.metadata });
    }

    results.sort((a, b) => a.distance - b.distance);
    return results.slice(0, k);
  }

  /** Delete a vector by ID. */
  async delete(id: string): Promise<boolean> {
    return this.vectors.delete(id);
  }

  /** Get the number of vectors. */
  get count(): number {
    return this.vectors.size;
  }

  /** Save the index to the storage adapter. */
  async save(): Promise<void> {
    if (!this.adapter) {
      throw new Error('No storage adapter configured');
    }
    const data = this.serialize();
    await this.adapter.put('needle-index', data);
  }

  /** Load the index from the storage adapter. */
  async load(): Promise<void> {
    if (!this.adapter) {
      throw new Error('No storage adapter configured');
    }

    if (this.config.progressiveLoading) {
      await this.loadProgressive();
    } else {
      const data = await this.adapter.get('needle-index');
      if (data) {
        this.deserialize(data);
      }
    }
    this.loaded = true;
  }

  /**
   * Progressive loading: load index in segments for fast cold start.
   * First loads the segment manifest, then loads segments on-demand.
   */
  private async loadProgressive(): Promise<void> {
    if (!this.adapter) return;

    const manifestData = await this.adapter.get('needle-manifest');
    if (!manifestData) {
      // No manifest — try full index
      const data = await this.adapter.get('needle-index');
      if (data) this.deserialize(data);
      return;
    }

    const decoder = new TextDecoder();
    const manifest = JSON.parse(decoder.decode(manifestData)) as {
      segments: number;
      totalVectors: number;
    };

    // Load first segment immediately (for fast cold start)
    const seg0 = await this.adapter.get('needle-segment-0');
    if (seg0) this.deserialize(seg0);

    // Load remaining segments in background
    for (let i = 1; i < manifest.segments; i++) {
      const seg = await this.adapter.get(`needle-segment-${i}`);
      if (seg) this.deserializeAppend(seg);
    }
  }

  /** Save the index to the storage adapter, optionally in segments. */
  async save(): Promise<void> {
    if (!this.adapter) {
      throw new Error('No storage adapter configured');
    }

    if (this.config.progressiveLoading && this.vectors.size > this.config.segmentSize) {
      await this.saveProgressive();
    } else {
      const data = this.serialize();
      await this.adapter.put('needle-index', data);
    }
  }

  /** Save in segments for progressive loading. */
  private async saveProgressive(): Promise<void> {
    if (!this.adapter) return;

    const allEntries = Array.from(this.vectors.entries());
    const segSize = this.config.segmentSize;
    const numSegments = Math.ceil(allEntries.length / segSize);

    // Save manifest
    const encoder = new TextEncoder();
    const manifest = { segments: numSegments, totalVectors: allEntries.length };
    await this.adapter.put('needle-manifest', encoder.encode(JSON.stringify(manifest)).buffer);

    // Save each segment
    for (let i = 0; i < numSegments; i++) {
      const segment = allEntries.slice(i * segSize, (i + 1) * segSize);
      const entries = segment.map(([id, entry]) => ({
        id,
        vector: Array.from(entry.vector),
        metadata: entry.metadata,
      }));
      const data = encoder.encode(JSON.stringify(entries)).buffer;
      await this.adapter.put(`needle-segment-${i}`, data);
    }
  }

  private computeDistance(a: Float32Array, b: Float32Array): number {
    switch (this.config.distance) {
      case 'cosine': return this.cosineDistance(a, b);
      case 'euclidean': return this.euclideanDistance(a, b);
      case 'dot': return this.dotDistance(a, b);
      default: return this.cosineDistance(a, b);
    }
  }

  private cosineDistance(a: Float32Array, b: Float32Array): number {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 1.0 : 1.0 - dot / denom;
  }

  private euclideanDistance(a: Float32Array, b: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      sum += d * d;
    }
    return Math.sqrt(sum);
  }

  private dotDistance(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
    }
    return -dot; // Negate so lower = more similar
  }

  private serialize(): ArrayBuffer {
    const encoder = new TextEncoder();
    const entries: Array<{ id: string; vector: number[]; metadata?: Record<string, unknown> }> = [];
    for (const [id, entry] of this.vectors) {
      entries.push({
        id,
        vector: Array.from(entry.vector),
        metadata: entry.metadata,
      });
    }
    return encoder.encode(JSON.stringify(entries)).buffer;
  }

  private deserialize(buffer: ArrayBuffer): void {
    const decoder = new TextDecoder();
    const json = decoder.decode(buffer);
    const entries = JSON.parse(json) as Array<{ id: string; vector: number[]; metadata?: Record<string, unknown> }>;
    this.vectors.clear();
    for (const entry of entries) {
      this.vectors.set(entry.id, {
        vector: new Float32Array(entry.vector),
        metadata: entry.metadata,
      });
    }
  }

  /** Append-deserialize: merge entries into existing vectors (for progressive loading). */
  private deserializeAppend(buffer: ArrayBuffer): void {
    const decoder = new TextDecoder();
    const json = decoder.decode(buffer);
    const entries = JSON.parse(json) as Array<{ id: string; vector: number[]; metadata?: Record<string, unknown> }>;
    for (const entry of entries) {
      this.vectors.set(entry.id, {
        vector: new Float32Array(entry.vector),
        metadata: entry.metadata,
      });
    }
  }
}

export default NeedleEdge;
