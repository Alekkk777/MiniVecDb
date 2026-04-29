import { embed } from 'ai';
import type { EmbeddingModel } from 'ai';
import { MicroVecDB } from './MicroVecDB.js';
import type { DbOptions } from './types.js';

const WASM_DIMS = 384;

// ----- Public types ----------------------------------------------------------

export interface SemanticCacheOptions extends DbOptions {
  /**
   * Vercel AI SDK embedding model — must produce 384-dim vectors.
   * e.g. `openai.embedding('text-embedding-3-small', { dimensions: 384 })`
   */
  embeddingModel: EmbeddingModel;
  /**
   * Minimum cosine similarity to count as a cache hit. Default: 0.95.
   * Lower = more aggressive cache (more hits, lower accuracy).
   */
  threshold?: number;
  /**
   * How long cached responses stay alive, in minutes. 0 = immortal.
   */
  ttlMinutes?: number;
  /** GC interval in ms. Default: 60 000. Only relevant when ttlMinutes > 0. */
  gcIntervalMs?: number;
}

interface CacheEntry {
  response: string;
  insertedAt: number;
}

// ----- SemanticCache ---------------------------------------------------------

/**
 * Semantic cache for LLM responses backed by MicroVecDB.
 *
 * Stores prompt→response pairs as 384-dim embeddings. On each call it looks
 * for a semantically similar prompt already in cache; if the cosine similarity
 * exceeds `threshold` the stored response is returned instantly without any
 * LLM call.
 *
 * @example
 * ```ts
 * import { openai } from '@ai-sdk/openai';
 * import { generateText } from 'ai';
 * import { withSemanticCache } from '@microvecdb/core/cache';
 *
 * const cachedGenerate = await withSemanticCache(
 *   {
 *     embeddingModel: openai.embedding('text-embedding-3-small', { dimensions: 384 }),
 *     threshold: 0.95,
 *     ttlMinutes: 60,
 *   },
 *   async (prompt) => {
 *     const { text } = await generateText({ model: openai('gpt-4o'), prompt });
 *     return text;
 *   },
 * );
 *
 * // First call → real LLM, subsequent similar calls → instant cache hit.
 * const answer = await cachedGenerate(userQuestion);
 * ```
 */
export class SemanticCache {
  readonly #db: MicroVecDB;
  readonly #model: EmbeddingModel;
  readonly #threshold: number;
  readonly #ttlMs: number;
  readonly #entries = new Map<number, CacheEntry>();
  #nextId = 0;
  #gcTimer: ReturnType<typeof setInterval> | null = null;

  private constructor(db: MicroVecDB, opts: SemanticCacheOptions) {
    this.#db        = db;
    this.#model     = opts.embeddingModel;
    this.#threshold = opts.threshold   ?? 0.95;
    this.#ttlMs     = (opts.ttlMinutes ?? 0) * 60_000;

    if (this.#ttlMs > 0) {
      const interval = opts.gcIntervalMs ?? 60_000;
      this.#gcTimer = setInterval(() => { this.#gcTick(); }, interval);
      if (
        this.#gcTimer !== null &&
        typeof this.#gcTimer === 'object' &&
        'unref' in this.#gcTimer
      ) {
        (this.#gcTimer as NodeJS.Timeout).unref();
      }
    }
  }

  // ----- Factory ------------------------------------------------------------

  static async create(opts: SemanticCacheOptions): Promise<SemanticCache> {
    const db = await MicroVecDB.init(opts);
    return new SemanticCache(db, opts);
  }

  // ----- Core operations ----------------------------------------------------

  /**
   * Look up a prompt in the cache.
   * Returns the cached response string, or `null` on a miss.
   */
  async lookup(prompt: string): Promise<string | null> {
    if (this.#entries.size === 0) return null;

    const { embedding } = await embed({ model: this.#model, value: prompt });
    this.#assertDims(embedding.length);

    const hits = this.#db.search(new Float32Array(embedding), { limit: 1 });
    if (hits.length === 0) return null;

    const [hit] = hits;
    if (hit.score < this.#threshold) return null;

    return this.#entries.get(hit.id)?.response ?? null;
  }

  /**
   * Store a prompt→response pair in the cache.
   */
  async set(prompt: string, response: string): Promise<void> {
    const { embedding } = await embed({ model: this.#model, value: prompt });
    this.#assertDims(embedding.length);

    const id = this.#nextId++;
    this.#db.insert({ id, vector: new Float32Array(embedding) });
    this.#entries.set(id, { response, insertedAt: Date.now() });
  }

  /**
   * Wrap a fallback function with cache-aside logic.
   *
   * The returned function:
   *   1. Checks the cache for a semantically similar prompt.
   *   2. On hit: returns the cached response.
   *   3. On miss: calls `fn`, stores the result, then returns it.
   */
  wrap(fn: (prompt: string) => Promise<string>): (prompt: string) => Promise<string> {
    return async (prompt: string): Promise<string> => {
      const cached = await this.lookup(prompt);
      if (cached !== null) return cached;
      const response = await fn(prompt);
      await this.set(prompt, response);
      return response;
    };
  }

  // ----- TTL / GC -----------------------------------------------------------

  /** Run one GC cycle manually. Returns the number of entries evicted. */
  runGc(): number {
    if (this.#ttlMs <= 0) return 0;
    return this.#gcTick();
  }

  // ----- Lifecycle ----------------------------------------------------------

  /** Stop the GC timer and release WASM memory. Safe to call multiple times. */
  destroy(): void {
    if (this.#gcTimer !== null) {
      clearInterval(this.#gcTimer);
      this.#gcTimer = null;
    }
    this.#entries.clear();
    this.#db.dispose();
  }

  // ----- Private ------------------------------------------------------------

  #assertDims(got: number): void {
    if (got !== WASM_DIMS) {
      throw new RangeError(
        `[SemanticCache] expected ${WASM_DIMS}-dim embedding, got ${got}. ` +
        `Pass a 384-dim model, e.g. openai.embedding('text-embedding-3-small', { dimensions: 384 }).`,
      );
    }
  }

  #gcTick(): number {
    const tombstoned = this.#db.runGc(this.#ttlMs);
    const cutoff = Date.now() - this.#ttlMs;
    for (const [id, entry] of this.#entries) {
      if (entry.insertedAt < cutoff) this.#entries.delete(id);
    }
    return tombstoned;
  }
}

// ----- Convenience HOF -------------------------------------------------------

/**
 * Wrap an async function that calls an LLM with semantic caching.
 *
 * Identical prompts — or semantically very similar ones — skip the LLM call
 * entirely and return the cached response. The cache self-expires after
 * `ttlMinutes` minutes (0 = never).
 *
 * @example
 * ```ts
 * const cachedGenerate = await withSemanticCache(
 *   {
 *     embeddingModel: openai.embedding('text-embedding-3-small', { dimensions: 384 }),
 *     threshold: 0.95,
 *     ttlMinutes: 60,
 *   },
 *   async (prompt) => {
 *     const { text } = await generateText({ model: openai('gpt-4o'), prompt });
 *     return text;
 *   },
 * );
 *
 * const answer = await cachedGenerate(userQuestion);
 * ```
 */
export async function withSemanticCache(
  opts: SemanticCacheOptions,
  fallback: (prompt: string) => Promise<string>,
): Promise<(prompt: string) => Promise<string>> {
  const cache = await SemanticCache.create(opts);
  return cache.wrap(fallback);
}
