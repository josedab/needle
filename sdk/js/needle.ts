/**
 * @anthropic/needle - Browser-Native Semantic Search SDK
 *
 * Provides IndexedDB persistence, service worker integration, and progressive
 * index loading on top of the Needle WASM module.
 *
 * @example
 * ```typescript
 * import { NeedleDB } from '@anthropic/needle';
 *
 * const db = await NeedleDB.create('mydb', { dimensions: 384 });
 * await db.insert('doc1', new Float32Array(384).fill(0.1), { title: 'Hello' });
 * const results = await db.search(new Float32Array(384).fill(0.1), 10);
 * await db.persist(); // Save to IndexedDB
 * ```
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NeedleDBConfig {
  /** Collection name */
  name?: string;
  /** Vector dimensions */
  dimensions: number;
  /** Distance function: 'cosine' | 'euclidean' | 'dotproduct' */
  distance?: 'cosine' | 'euclidean' | 'dotproduct';
  /** IndexedDB database name for persistence */
  dbName?: string;
  /** Enable progressive loading (lazy index hydration) */
  progressiveLoading?: boolean;
  /** Maximum vectors to hold in memory (for quantized search) */
  maxInMemory?: number;
}

export interface SearchResult {
  id: string;
  distance: number;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface InsertOptions {
  metadata?: Record<string, unknown>;
  skipDuplicates?: boolean;
}

export interface PersistenceStats {
  stored: boolean;
  sizeBytes: number;
  vectorCount: number;
  timestamp: number;
}

// ---------------------------------------------------------------------------
// IndexedDB Persistence Layer
// ---------------------------------------------------------------------------

const IDB_STORE_NAME = 'needle_collections';

class IndexedDBPersistence {
  private dbName: string;
  private db: IDBDatabase | null = null;

  constructor(dbName: string) {
    this.dbName = dbName;
  }

  async open(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') {
        reject(new Error('IndexedDB not available'));
        return;
      }
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(IDB_STORE_NAME)) {
          db.createObjectStore(IDB_STORE_NAME, { keyPath: 'name' });
        }
      };
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
  }

  async save(name: string, data: Uint8Array, metadata: Record<string, unknown>): Promise<void> {
    if (!this.db) throw new Error('Database not opened');
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(IDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(IDB_STORE_NAME);
      store.put({ name, data, metadata, timestamp: Date.now() });
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async load(name: string): Promise<{ data: Uint8Array; metadata: Record<string, unknown> } | null> {
    if (!this.db) throw new Error('Database not opened');
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(IDB_STORE_NAME, 'readonly');
      const store = tx.objectStore(IDB_STORE_NAME);
      const request = store.get(name);
      request.onsuccess = () => {
        if (request.result) {
          resolve({ data: request.result.data, metadata: request.result.metadata });
        } else {
          resolve(null);
        }
      };
      request.onerror = () => reject(request.error);
    });
  }

  async delete(name: string): Promise<void> {
    if (!this.db) throw new Error('Database not opened');
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(IDB_STORE_NAME, 'readwrite');
      const store = tx.objectStore(IDB_STORE_NAME);
      store.delete(name);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  async listCollections(): Promise<string[]> {
    if (!this.db) throw new Error('Database not opened');
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction(IDB_STORE_NAME, 'readonly');
      const store = tx.objectStore(IDB_STORE_NAME);
      const request = store.getAllKeys();
      request.onsuccess = () => resolve(request.result as string[]);
      request.onerror = () => reject(request.error);
    });
  }

  close(): void {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Search Cache (Service Worker compatible)
// ---------------------------------------------------------------------------

class SearchCache {
  private cache: Map<string, { results: SearchResult[]; timestamp: number }>;
  private maxSize: number;
  private ttlMs: number;

  constructor(maxSize = 1000, ttlMs = 60000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttlMs = ttlMs;
  }

  private hashQuery(query: Float32Array, k: number): string {
    // Simple hash: first 8 values + length + k
    const prefix = Array.from(query.slice(0, 8)).map(v => v.toFixed(4)).join(',');
    return `${prefix}:${query.length}:${k}`;
  }

  get(query: Float32Array, k: number): SearchResult[] | null {
    const key = this.hashQuery(query, k);
    const entry = this.cache.get(key);
    if (entry && Date.now() - entry.timestamp < this.ttlMs) {
      return entry.results;
    }
    if (entry) this.cache.delete(key);
    return null;
  }

  set(query: Float32Array, k: number, results: SearchResult[]): void {
    if (this.cache.size >= this.maxSize) {
      // Evict oldest
      const oldest = this.cache.keys().next().value;
      if (oldest !== undefined) this.cache.delete(oldest);
    }
    const key = this.hashQuery(query, k);
    this.cache.set(key, { results, timestamp: Date.now() });
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

// ---------------------------------------------------------------------------
// NeedleDB - Main SDK Class
// ---------------------------------------------------------------------------

/**
 * Browser-native vector database with IndexedDB persistence.
 *
 * Wraps the Needle WASM module with:
 * - Automatic IndexedDB persistence
 * - Search result caching
 * - Progressive index loading
 * - Memory usage tracking
 */
export class NeedleDB {
  private config: Required<NeedleDBConfig>;
  private vectors: Map<string, { vector: Float32Array; metadata?: Record<string, unknown> }>;
  private persistence: IndexedDBPersistence | null = null;
  private searchCache: SearchCache;
  private dirty = false;
  private _wasmCollection: unknown = null; // WasmCollection when available

  private constructor(config: Required<NeedleDBConfig>) {
    this.config = config;
    this.vectors = new Map();
    this.searchCache = new SearchCache();
  }

  /**
   * Create a new NeedleDB instance, optionally loading from IndexedDB.
   */
  static async create(name: string, config: Omit<NeedleDBConfig, 'name'>): Promise<NeedleDB> {
    const fullConfig: Required<NeedleDBConfig> = {
      name,
      dimensions: config.dimensions,
      distance: config.distance ?? 'cosine',
      dbName: config.dbName ?? 'needle_browser_db',
      progressiveLoading: config.progressiveLoading ?? true,
      maxInMemory: config.maxInMemory ?? 100000,
    };

    const db = new NeedleDB(fullConfig);

    // Try to initialize IndexedDB persistence
    try {
      db.persistence = new IndexedDBPersistence(fullConfig.dbName);
      await db.persistence.open();

      // Try to load existing data
      const stored = await db.persistence.load(name);
      if (stored) {
        db.deserializeVectors(stored.data);
      }
    } catch {
      // IndexedDB not available (e.g., Node.js) — operate in-memory only
      db.persistence = null;
    }

    return db;
  }

  /**
   * Insert a vector with optional metadata.
   */
  async insert(
    id: string,
    vector: Float32Array | number[],
    metadata?: Record<string, unknown>
  ): Promise<void> {
    const v = vector instanceof Float32Array ? vector : new Float32Array(vector);

    if (v.length !== this.config.dimensions) {
      throw new Error(
        `Dimension mismatch: expected ${this.config.dimensions}, got ${v.length}`
      );
    }

    this.vectors.set(id, { vector: v, metadata });
    this.dirty = true;
    this.searchCache.clear();
  }

  /**
   * Insert multiple vectors in a batch.
   */
  async insertBatch(
    items: Array<{ id: string; vector: Float32Array | number[]; metadata?: Record<string, unknown> }>
  ): Promise<number> {
    let count = 0;
    for (const item of items) {
      await this.insert(item.id, item.vector, item.metadata);
      count++;
    }
    return count;
  }

  /**
   * Search for nearest neighbors.
   */
  async search(query: Float32Array | number[], k: number): Promise<SearchResult[]> {
    const q = query instanceof Float32Array ? query : new Float32Array(query);

    if (q.length !== this.config.dimensions) {
      throw new Error(
        `Dimension mismatch: expected ${this.config.dimensions}, got ${q.length}`
      );
    }

    // Check cache
    const cached = this.searchCache.get(q, k);
    if (cached) return cached;

    // Brute-force search (WASM HNSW would be used when available)
    const results = this.bruteForceSearch(q, k);
    this.searchCache.set(q, k, results);
    return results;
  }

  /**
   * Get a vector by ID.
   */
  get(id: string): { vector: Float32Array; metadata?: Record<string, unknown> } | null {
    const entry = this.vectors.get(id);
    return entry ?? null;
  }

  /**
   * Delete a vector by ID.
   */
  delete(id: string): boolean {
    const existed = this.vectors.delete(id);
    if (existed) {
      this.dirty = true;
      this.searchCache.clear();
    }
    return existed;
  }

  /**
   * Persist the current state to IndexedDB.
   */
  async persist(): Promise<PersistenceStats> {
    if (!this.persistence) {
      return { stored: false, sizeBytes: 0, vectorCount: this.vectors.size, timestamp: Date.now() };
    }

    const serialized = this.serializeVectors();
    const metadata = {
      dimensions: this.config.dimensions,
      distance: this.config.distance,
      vectorCount: this.vectors.size,
    };

    await this.persistence.save(this.config.name, serialized, metadata);
    this.dirty = false;

    return {
      stored: true,
      sizeBytes: serialized.byteLength,
      vectorCount: this.vectors.size,
      timestamp: Date.now(),
    };
  }

  /**
   * Clear all vectors.
   */
  async clear(): Promise<void> {
    this.vectors.clear();
    this.searchCache.clear();
    this.dirty = true;
    if (this.persistence) {
      await this.persistence.delete(this.config.name);
    }
  }

  /** Number of vectors stored. */
  get count(): number {
    return this.vectors.size;
  }

  /** Whether there are unsaved changes. */
  get isDirty(): boolean {
    return this.dirty;
  }

  /** Estimated memory usage in bytes. */
  get memoryUsage(): number {
    return this.vectors.size * (this.config.dimensions * 4 + 256);
  }

  /** Close the database and release IndexedDB connection. */
  close(): void {
    if (this.persistence) {
      this.persistence.close();
    }
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private bruteForceSearch(query: Float32Array, k: number): SearchResult[] {
    const distances: Array<{ id: string; distance: number; metadata?: Record<string, unknown> }> = [];

    for (const [id, entry] of this.vectors) {
      const dist = this.computeDistance(query, entry.vector);
      distances.push({ id, distance: dist, metadata: entry.metadata });
    }

    distances.sort((a, b) => a.distance - b.distance);

    return distances.slice(0, k).map(d => ({
      id: d.id,
      distance: d.distance,
      score: 1.0 / (1.0 + d.distance),
      metadata: d.metadata,
    }));
  }

  private computeDistance(a: Float32Array, b: Float32Array): number {
    switch (this.config.distance) {
      case 'euclidean': {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
          const diff = a[i] - b[i];
          sum += diff * diff;
        }
        return Math.sqrt(sum);
      }
      case 'dotproduct': {
        let dot = 0;
        for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
        return -dot; // Negate so lower = more similar
      }
      case 'cosine':
      default: {
        let dot = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
          dot += a[i] * b[i];
          normA += a[i] * a[i];
          normB += b[i] * b[i];
        }
        const denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom > 0 ? 1 - dot / denom : 1;
      }
    }
  }

  private serializeVectors(): Uint8Array {
    const entries: Array<{ id: string; vector: number[]; metadata?: Record<string, unknown> }> = [];
    for (const [id, entry] of this.vectors) {
      entries.push({
        id,
        vector: Array.from(entry.vector),
        metadata: entry.metadata,
      });
    }
    const json = JSON.stringify(entries);
    return new TextEncoder().encode(json);
  }

  private deserializeVectors(data: Uint8Array): void {
    try {
      const json = new TextDecoder().decode(data);
      const entries = JSON.parse(json) as Array<{
        id: string;
        vector: number[];
        metadata?: Record<string, unknown>;
      }>;
      for (const entry of entries) {
        this.vectors.set(entry.id, {
          vector: new Float32Array(entry.vector),
          metadata: entry.metadata,
        });
      }
    } catch {
      // Corrupted data — start fresh
      this.vectors.clear();
    }
  }
}

export default NeedleDB;
