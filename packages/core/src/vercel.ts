import { embed, embedMany, tool, zodSchema, type EmbeddingModel } from 'ai';
import { z } from 'zod';
import { MicroVecDB } from './MicroVecDB.js';
import type { DbOptions } from './types.js';

// The WASM engine is hardcoded to 384-bit quantised vectors.
// Use a matching model, e.g. openai.embedding('text-embedding-3-small').
const WASM_DIMS = 384;

// ----- Public types ----------------------------------------------------------

/**
 * Configuration for {@link VercelMiniVecDb}.
 *
 * Extends `DbOptions` with TTL/GC knobs that turn the store into a
 * self-cleaning Agent Scratchpad — ephemeral observations are tombstoned
 * automatically after `ttlMinutes` minutes.
 */
export interface VercelMiniVecDbConfig extends DbOptions {
  /**
   * How long inserted texts stay alive, in minutes.
   * 0 (default) = immortal — no GC loop is started.
   */
  ttlMinutes?: number;
  /**
   * How often the background GC loop fires, in milliseconds.
   * Default: 60 000 ms (1 minute). Only relevant when `ttlMinutes > 0`.
   */
  gcIntervalMs?: number;
}

/** One result returned by {@link VercelMiniVecDb.retrieve}. */
export interface RetrievalResult {
  /** Original text that was stored via {@link VercelMiniVecDb.add}. */
  text: string;
  /** Normalised similarity score — 1.0 = identical, 0.0 = maximum distance. */
  score: number;
  /** Optional metadata supplied at insert time. */
  metadata?: Record<string, unknown>;
}

// ----- Internal types --------------------------------------------------------

interface DocEntry {
  text: string;
  metadata?: Record<string, unknown>;
  /** Unix timestamp (ms) at insert — mirrors WASM `inserted_at`. */
  insertedAt: number;
}

// ----- Adapter ---------------------------------------------------------------

/**
 * Vercel AI SDK adapter for MiniVecDb.
 *
 * Plugs directly into any `streamText` / `generateText` call via
 * {@link createRetrievalTool}, turning the store into an ephemeral memory
 * layer an LLM agent can query autonomously.
 *
 * **Dimension constraint**: the WASM engine quantises 384-dimensional vectors.
 * Pass a 384-dim model, e.g. `openai.embedding('text-embedding-3-small')`.
 *
 * @example
 * ```ts
 * // Next.js API route
 * import { openai } from '@ai-sdk/openai';
 * import { streamText } from 'ai';
 * import { VercelMiniVecDb } from '@microvecdb/core/vercel';
 *
 * const memory = await VercelMiniVecDb.create(
 *   openai.embedding('text-embedding-3-small'),
 *   { ttlMinutes: 10, gcIntervalMs: 30_000 },
 * );
 *
 * await memory.add(['User said: the ticket number is 42-ABC.']);
 *
 * const result = await streamText({
 *   model: openai('gpt-4o-mini'),
 *   tools: { searchMemory: memory.createRetrievalTool() },
 *   messages,
 * });
 *
 * // When the request finishes:
 * memory.destroy();
 * ```
 */
export class VercelMiniVecDb {
  readonly #db: MicroVecDB;
  readonly #model: EmbeddingModel;
  readonly #docs = new Map<number, DocEntry>();
  readonly #ttlMs: number;
  #gcTimer: ReturnType<typeof setInterval> | null = null;
  #nextId = 0;

  private constructor(
    db: MicroVecDB,
    model: EmbeddingModel,
    config: VercelMiniVecDbConfig,
  ) {
    this.#db    = db;
    this.#model = model;
    this.#ttlMs = (config.ttlMinutes ?? 0) * 60_000;

    if (this.#ttlMs > 0) {
      const interval = config.gcIntervalMs ?? 60_000;
      this.#gcTimer = setInterval(() => { this.#runGcTick(); }, interval);

      // In Node.js / Edge runtimes, prevent the timer from keeping the
      // process alive after the request handler finishes.
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

  /**
   * Async factory — initialises the WASM engine and returns a ready store.
   *
   * @param model   A Vercel AI SDK {@link EmbeddingModel} — must produce
   *                384-dim vectors (e.g. `openai.embedding('text-embedding-3-small')`).
   * @param config  Optional TTL / capacity / persistence config.
   */
  static async create(
    model: EmbeddingModel,
    config: VercelMiniVecDbConfig = {},
  ): Promise<VercelMiniVecDb> {
    const db = await MicroVecDB.init(config);
    return new VercelMiniVecDb(db, model, config);
  }

  // ----- Write --------------------------------------------------------------

  /**
   * Embed `texts` and insert them into the store.
   *
   * Uses Vercel AI SDK's `embedMany` to batch the embedding call.
   *
   * @param texts     Plain strings to store (e.g. agent observations).
   * @param metadatas Optional per-text metadata objects (same order as texts).
   * @returns         Internal integer IDs assigned to each inserted text.
   *
   * @throws {RangeError} If the embedding model returns wrong-dimension vectors.
   */
  async add(
    texts: string[],
    metadatas?: Record<string, unknown>[],
  ): Promise<number[]> {
    if (texts.length === 0) return [];

    const { embeddings } = await embedMany({ model: this.#model, values: texts });

    const now = Date.now();
    const ids: number[] = [];

    for (let i = 0; i < texts.length; i++) {
      const vec = embeddings[i];
      if (vec.length !== WASM_DIMS) {
        throw new RangeError(
          `[VercelMiniVecDb] add[${i}]: expected ${WASM_DIMS}-dim embedding, ` +
          `got ${vec.length}. Use a 384-dim model such as text-embedding-3-small.`,
        );
      }
      const id = this.#nextId++;
      this.#db.insert({ id, vector: new Float32Array(vec) });
      this.#docs.set(id, { text: texts[i], metadata: metadatas?.[i], insertedAt: now });
      ids.push(id);
    }

    return ids;
  }

  // ----- Read ---------------------------------------------------------------

  /**
   * Find the `k` texts closest to `query`.
   *
   * Uses the HNSW index when built, brute-force scan otherwise.
   * Tombstoned (TTL-expired) entries are silently excluded.
   *
   * @param query Natural-language query string; embedded on the fly.
   * @param k     Number of results to return (default: 4).
   * @returns     Array of {@link RetrievalResult}, best match first.
   *
   * @throws {RangeError} If the query embedding has wrong dimensions.
   */
  async retrieve(query: string, k = 4): Promise<RetrievalResult[]> {
    const { embedding } = await embed({ model: this.#model, value: query });

    if (embedding.length !== WASM_DIMS) {
      throw new RangeError(
        `[VercelMiniVecDb] retrieve: expected ${WASM_DIMS}-dim query, ` +
        `got ${embedding.length}.`,
      );
    }

    const hits = this.#db.search(new Float32Array(embedding), { limit: k });
    const results: RetrievalResult[] = [];
    for (const hit of hits) {
      const entry = this.#docs.get(hit.id);
      if (entry !== undefined) {
        results.push({ text: entry.text, score: hit.score, metadata: entry.metadata });
      }
    }
    return results;
  }

  // ----- Index --------------------------------------------------------------

  /**
   * Build (or rebuild) the HNSW approximate nearest-neighbour index.
   *
   * Call this after a large batch `add()` for faster subsequent `retrieve()`.
   */
  buildIndex(): void {
    this.#db.buildIndex();
  }

  // ----- TTL / GC -----------------------------------------------------------

  /**
   * Run one GC cycle immediately, outside the scheduled interval.
   *
   * Useful in tests or when you want eager cleanup after a known-expired entry.
   *
   * @returns Number of entries garbage-collected.
   */
  runGc(): number {
    if (this.#ttlMs <= 0) return 0;
    return this.#runGcTick();
  }

  // ----- Vercel AI SDK tool -------------------------------------------------

  /**
   * Create a Vercel AI SDK tool the LLM can call to search this store.
   *
   * Drop the returned object directly into the `tools` map of any
   * `streamText` / `generateText` call:
   *
   * ```ts
   * const result = await streamText({
   *   model: openai('gpt-4o-mini'),
   *   tools: { searchMemory: memory.createRetrievalTool() },
   *   messages,
   * });
   * ```
   *
   * The LLM receives the top-`k` matches formatted as a numbered list
   * with similarity scores, or a "no context found" message.
   *
   * @param description  Shown to the LLM when deciding whether to use this
   *                     tool. Override to give the model more context about
   *                     what is stored (e.g. "Search session observations").
   * @param k            Number of results the tool returns (default: 4).
   */
  createRetrievalTool(
    description = 'Search the ephemeral memory for relevant context before answering.',
    k = 4,
  ) {
    // `zodSchema()` wraps a Zod schema into the FlexibleSchema that AI SDK v4+
    // expects in the `inputSchema` field (renamed from `parameters` in v3).
    return tool({
      description,
      inputSchema: zodSchema(
        z.object({
          query: z
            .string()
            .describe(
              'The search query to find relevant context in the ephemeral memory',
            ),
        }),
      ),
      execute: async ({ query }: { query: string }): Promise<string> => {
        const results = await this.retrieve(query, k);
        if (results.length === 0) return 'No relevant context found in memory.';
        return results
          .map((r, i) => `[${i + 1}] (score=${r.score.toFixed(3)}) ${r.text}`)
          .join('\n');
      },
    });
  }

  // ----- Lifecycle ----------------------------------------------------------

  /**
   * Stop the GC timer and release WASM memory.
   *
   * Call this when the agent task or request handler is done to prevent
   * the background `setInterval` and WASM allocation from leaking.
   * Safe to call multiple times.
   */
  destroy(): void {
    if (this.#gcTimer !== null) {
      clearInterval(this.#gcTimer);
      this.#gcTimer = null;
    }
    this.#docs.clear();
    this.#db.dispose();
  }

  // ----- Private ------------------------------------------------------------

  /**
   * One GC cycle: tombstone in WASM, then mirror-evict the JS doc map.
   *
   * Both sides use the same TTL cutoff so they stay in lock-step.
   * The WASM `run_gc` reads `Date.now()` internally; the JS side computes
   * `Date.now() - ttlMs` here — both within the same event-loop turn.
   */
  #runGcTick(): number {
    const wasmTombstoned = this.#db.runGc(this.#ttlMs);

    const cutoff = Date.now() - this.#ttlMs;
    for (const [id, entry] of this.#docs) {
      if (entry.insertedAt < cutoff) {
        this.#docs.delete(id);
      }
    }

    return wasmTombstoned;
  }
}
