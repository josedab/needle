/**
 * Needle TypeScript Client — typed REST API client for the Needle vector database.
 *
 * @example
 * ```ts
 * const needle = new NeedleClient("http://localhost:8080");
 * await needle.createCollection("docs", 384);
 * await needle.insert("docs", "doc1", [0.1, 0.2, ...], { title: "Hello" });
 * const results = await needle.search("docs", [0.1, 0.2, ...], { k: 10 });
 * ```
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface CollectionInfo {
  name: string;
  dimensions: number;
  count: number;
  deleted_count: number;
}

export interface SearchResult {
  id: string;
  distance: number;
  score: number;
  metadata?: Record<string, unknown>;
  vector?: number[];
}

export interface SearchCursor {
  distance: number;
  id: string;
}

export interface SearchResponse {
  results: SearchResult[];
  next_cursor?: SearchCursor;
  has_more?: boolean;
  explanation?: unknown;
}

export interface SearchOptions {
  k?: number;
  filter?: Record<string, unknown>;
  post_filter?: Record<string, unknown>;
  include_vectors?: boolean;
  explain?: boolean;
  distance?: "cosine" | "euclidean" | "dot" | "manhattan";
  search_after?: SearchCursor;
}

export interface InsertOptions {
  metadata?: Record<string, unknown>;
  ttl_seconds?: number;
}

export interface RateLimitInfo {
  /** Maximum requests per second for this tier */
  limit: number | null;
  /** Remaining requests in current window */
  remaining: number | null;
  /** Seconds until rate limit resets */
  retryAfter: number | null;
}

export interface NeedleClientOptions {
  apiKey?: string;
  timeout?: number;
  /** Maximum number of retries for transient errors (429, 5xx). Default: 3. Set to 0 to disable. */
  maxRetries?: number;
  /** Callback invoked with rate limit info on each response */
  onRateLimit?: (info: RateLimitInfo) => void;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function extractRateLimitInfo(headers: Headers): RateLimitInfo {
  return {
    limit: headers.has("x-ratelimit-limit") ? parseInt(headers.get("x-ratelimit-limit")!, 10) : null,
    remaining: headers.has("x-ratelimit-remaining") ? parseInt(headers.get("x-ratelimit-remaining")!, 10) : null,
    retryAfter: headers.has("retry-after") ? parseInt(headers.get("retry-after")!, 10) : null,
  };
}

// ── Client ───────────────────────────────────────────────────────────────────

export class NeedleClient {
  private baseUrl: string;
  private headers: Record<string, string>;
  private timeout: number;
  private maxRetries: number;
  private onRateLimit?: (info: RateLimitInfo) => void;
  private _lastRateLimitInfo: RateLimitInfo | null = null;

  constructor(baseUrl: string, options?: NeedleClientOptions) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.headers = { "Content-Type": "application/json" };
    this.timeout = options?.timeout ?? 30000;
    this.maxRetries = options?.maxRetries ?? 3;
    this.onRateLimit = options?.onRateLimit;
    if (options?.apiKey) {
      this.headers["X-API-Key"] = options.apiKey;
    }
  }

  /** Returns rate limit info from the most recent response, or null if unavailable. */
  get lastRateLimitInfo(): RateLimitInfo | null {
    return this._lastRateLimitInfo;
  }

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const url = `${this.baseUrl}/v1${path}`;
    const init: RequestInit = {
      method,
      headers: this.headers,
      signal: AbortSignal.timeout(this.timeout),
    };
    if (body !== undefined) {
      init.body = JSON.stringify(body);
    }

    let lastError: NeedleApiError | undefined;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      if (attempt > 0 && lastError) {
        const delay = retryDelay(attempt, lastError.retryAfter);
        await sleep(delay);
      }

      const resp = await fetch(url, init);
      const rateLimitInfo = extractRateLimitInfo(resp.headers);
      if (rateLimitInfo.limit !== null || rateLimitInfo.remaining !== null || rateLimitInfo.retryAfter !== null) {
        this._lastRateLimitInfo = rateLimitInfo;
        this.onRateLimit?.(rateLimitInfo);
      }
      if (!resp.ok) {
        const error = await resp.json().catch(() => ({ error: resp.statusText }));
        lastError = new NeedleApiError(resp.status, error.error, error.code, error.help, rateLimitInfo.retryAfter);
        if (isRetryable(resp.status) && attempt < this.maxRetries) {
          continue;
        }
        throw lastError;
      }
      return resp.json() as Promise<T>;
    }

    throw lastError ?? new NeedleApiError(0, "Request failed after retries");
  }

  // ── Collections ──────────────────────────────────────────────────────────

  async listCollections(): Promise<CollectionInfo[]> {
    return this.request("GET", "/collections");
  }

  async createCollection(name: string, dimensions: number, distance?: string): Promise<void> {
    await this.request("POST", "/collections", { name, dimensions, distance });
  }

  async getCollection(name: string): Promise<CollectionInfo> {
    return this.request("GET", `/collections/${encodeURIComponent(name)}`);
  }

  async deleteCollection(name: string): Promise<void> {
    await this.request("DELETE", `/collections/${encodeURIComponent(name)}`);
  }

  // ── Vectors ──────────────────────────────────────────────────────────────

  async insert(collection: string, id: string, vector: number[], options?: InsertOptions): Promise<void> {
    await this.request("POST", `/collections/${encodeURIComponent(collection)}/vectors`, {
      id,
      vector,
      metadata: options?.metadata,
      ttl_seconds: options?.ttl_seconds,
    });
  }

  async batchInsert(collection: string, vectors: Array<{ id: string; vector: number[]; metadata?: Record<string, unknown>; ttl_seconds?: number }>): Promise<{ inserted: number; errors?: unknown[] }> {
    return this.request("POST", `/collections/${encodeURIComponent(collection)}/vectors/batch`, {
      vectors,
    });
  }

  async get(collection: string, id: string): Promise<{ id: string; vector: number[]; metadata?: Record<string, unknown> }> {
    return this.request("GET", `/collections/${encodeURIComponent(collection)}/vectors/${encodeURIComponent(id)}`);
  }

  async delete(collection: string, id: string): Promise<void> {
    await this.request("DELETE", `/collections/${encodeURIComponent(collection)}/vectors/${encodeURIComponent(id)}`);
  }

  async updateMetadata(collection: string, id: string, metadata: Record<string, unknown>, replace = false): Promise<void> {
    await this.request("POST", `/collections/${encodeURIComponent(collection)}/vectors/${encodeURIComponent(id)}/metadata`, {
      metadata,
      replace,
    });
  }

  // ── Search ───────────────────────────────────────────────────────────────

  async search(collection: string, vector: number[], options?: SearchOptions): Promise<SearchResponse> {
    return this.request("POST", `/collections/${encodeURIComponent(collection)}/search`, {
      vector,
      k: options?.k ?? 10,
      filter: options?.filter,
      post_filter: options?.post_filter,
      include_vectors: options?.include_vectors ?? false,
      explain: options?.explain ?? false,
      distance: options?.distance,
      search_after: options?.search_after,
    });
  }

  async recommend(collection: string, positiveIds: string[], negativeIds?: string[], limit = 10): Promise<{ results: SearchResult[] }> {
    return this.request("POST", `/collections/${encodeURIComponent(collection)}/recommend`, {
      positive_ids: positiveIds,
      negative_ids: negativeIds ?? [],
      limit,
    });
  }

  async query(collection: string, filter: Record<string, unknown>, limit = 100, offset = 0): Promise<{ data: Array<{ id: string; metadata?: unknown }>; pagination: { count: number; total: number; has_more: boolean } }> {
    return this.request("POST", `/collections/${encodeURIComponent(collection)}/query`, {
      filter,
      limit,
      offset,
    });
  }

  async count(collection: string, filter?: Record<string, unknown>): Promise<{ count: number }> {
    return this.request("POST", `/collections/${encodeURIComponent(collection)}/count`, {
      filter,
    });
  }

  // ── Health ───────────────────────────────────────────────────────────────

  async health(): Promise<{ status: string; version: string }> {
    const url = `${this.baseUrl}/health`;
    const resp = await fetch(url, { headers: this.headers, signal: AbortSignal.timeout(this.timeout) });
    return resp.json() as Promise<{ status: string; version: string }>;
  }
}

// ── Retry helpers ────────────────────────────────────────────────────────────

/** Returns true for status codes that are safe to retry. */
function isRetryable(status: number): boolean {
  return status === 429 || status >= 500;
}

/** Computes retry delay in ms with exponential backoff and jitter. */
function retryDelay(attempt: number, retryAfterSeconds: number | null): number {
  if (retryAfterSeconds !== null && retryAfterSeconds > 0) {
    return retryAfterSeconds * 1000;
  }
  const base = Math.min(1000 * 2 ** (attempt - 1), 30000);
  return base + Math.random() * base * 0.5;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Errors ──────────────────────────────────────────────────────────────────

export class NeedleApiError extends Error {
  status: number;
  code?: string;
  help?: string;
  retryAfter: number | null;

  constructor(status: number, message: string, code?: string, help?: string, retryAfter?: number | null) {
    super(message);
    this.name = "NeedleApiError";
    this.status = status;
    this.code = code;
    this.help = help;
    this.retryAfter = retryAfter ?? null;
  }
}
