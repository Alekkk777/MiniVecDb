import { VectorStore } from "@langchain/core/vectorstores";
import { Document } from "@langchain/core/documents";
import type { DocumentInterface } from "@langchain/core/documents";
import type { EmbeddingsInterface } from "@langchain/core/embeddings";
import { MicroVecDB } from "./MicroVecDB.js";
import type { DbOptions } from "./types.js";

// WASM engine is hardcoded to 384-bit quantised vectors.
// Use a 384-dim embedding model (e.g. Xenova/all-MiniLM-L6-v2).
const WASM_DIMS = 384;

// ----- Config ----------------------------------------------------------------

/**
 * Configuration for LangChainMiniVecDb.
 *
 * Extends the standard `DbOptions` with TTL/GC knobs that turn the store into
 * a self-cleaning Agent Scratchpad: old context is automatically tombstoned
 * in WASM and evicted from the JS document map on the same GC tick.
 */
export interface MiniVecDbConfig extends DbOptions {
  /**
   * How long inserted documents stay alive, in minutes.
   * 0 (default) = immortal — no GC loop is started.
   */
  ttlMinutes?: number;
  /**
   * How frequently the background GC loop fires, in milliseconds.
   * Default: 60 000 ms (1 minute).  Only relevant when `ttlMinutes > 0`.
   */
  gcIntervalMs?: number;
}

// ----- Internal types --------------------------------------------------------

interface DocEntry {
  doc: Document;
  /** Unix timestamp (ms) recorded at insert time — mirrors WASM `inserted_at`. */
  insertedAt: number;
}

// ----- Adapter ---------------------------------------------------------------

/**
 * LangChain `VectorStore` adapter for MiniVecDb.
 *
 * Plugs into any LangChain chain or agent that expects a `VectorStore`.
 * When `ttlMinutes` is set, a background GC timer tombstones expired
 * documents in WASM and evicts them from the JS map — turning this into a
 * zero-maintenance Agent Scratchpad.
 *
 * **Dimension constraint**: the WASM engine quantises 384-dimensional float
 * vectors.  Your `Embeddings` instance must output 384-dim vectors (e.g.
 * `HuggingFaceTransformersEmbeddings` with `Xenova/all-MiniLM-L6-v2`).
 * Passing vectors of any other dimension throws a `RangeError`.
 *
 * @example
 * ```ts
 * const store = await LangChainMiniVecDb.create(
 *   new HuggingFaceTransformersEmbeddings({ modelName: "Xenova/all-MiniLM-L6-v2" }),
 *   { capacity: 2_000, ttlMinutes: 10, gcIntervalMs: 30_000 },
 * );
 *
 * await store.addDocuments(agentObservations);
 * const context = await store.similaritySearch(userQuery, 5);
 *
 * // When the agent task finishes — clears the GC timer and WASM memory.
 * store.destroy();
 * ```
 */
export class LangChainMiniVecDb extends VectorStore {
  // LangChainMiniVecDb has no filter support — narrow the type so callers
  // get a compile-time error instead of a silent no-op.
  declare FilterType: never;

  _vectorstoreType(): string {
    return "minivecdb";
  }

  private readonly db: MicroVecDB;
  private readonly docs = new Map<number, DocEntry>();
  private readonly ttlMs: number;
  private gcTimer: ReturnType<typeof setInterval> | null = null;
  private nextId = 0;

  private constructor(
    embeddings: EmbeddingsInterface,
    db: MicroVecDB,
    config: MiniVecDbConfig,
  ) {
    super(embeddings, config);
    this.db = db;
    this.ttlMs = (config.ttlMinutes ?? 0) * 60_000;

    if (this.ttlMs > 0) {
      const interval = config.gcIntervalMs ?? 60_000;
      this.gcTimer = setInterval(() => { this._runGcTick(); }, interval);

      // In Node.js, prevent the GC timer from keeping the process alive after
      // the agent task finishes and `destroy()` hasn't been called explicitly.
      if (
        this.gcTimer !== null &&
        typeof this.gcTimer === "object" &&
        "unref" in this.gcTimer
      ) {
        (this.gcTimer as NodeJS.Timeout).unref();
      }
    }
  }

  // ----- Factory ------------------------------------------------------------

  /**
   * Async factory — loads the WASM engine and returns a ready-to-use store.
   *
   * Provide the same `Embeddings` instance you'll use when calling
   * `addDocuments` / `similaritySearch`.
   */
  static async create(
    embeddings: EmbeddingsInterface,
    config: MiniVecDbConfig = {},
  ): Promise<LangChainMiniVecDb> {
    const db = await MicroVecDB.init(config);
    return new LangChainMiniVecDb(embeddings, db, config);
  }

  // ----- VectorStore overrides ---------------------------------------------

  /**
   * Store pre-computed embedding vectors alongside their source documents.
   *
   * Called internally by `addDocuments` after the `Embeddings` instance has
   * produced float vectors.  You can also call it directly if you have vectors
   * from an external source.
   *
   * @param vectors  Array of 384-dim float arrays — one per document.
   * @param documents  Corresponding `Document` objects (same order as `vectors`).
   * @returns  String IDs assigned to each inserted document.
   */
  async addVectors(
    vectors: number[][],
    documents: Document[],
  ): Promise<string[]> {
    if (vectors.length !== documents.length) {
      throw new RangeError(
        `[LangChainMiniVecDb] addVectors: vectors.length (${vectors.length}) ` +
          `!== documents.length (${documents.length})`,
      );
    }

    const ids: string[] = [];
    const now = Date.now();

    for (let i = 0; i < vectors.length; i++) {
      if (vectors[i].length !== WASM_DIMS) {
        throw new RangeError(
          `[LangChainMiniVecDb] addVectors[${i}]: expected ${WASM_DIMS}-dim vector, ` +
            `got ${vectors[i].length}. ` +
            `Use a 384-dim model such as Xenova/all-MiniLM-L6-v2.`,
        );
      }

      const id = this.nextId++;
      // MicroVecDB.insert validates the id range and vector length.
      this.db.insert({ id, vector: new Float32Array(vectors[i]) });
      this.docs.set(id, { doc: documents[i], insertedAt: now });
      ids.push(String(id));
    }

    return ids;
  }

  /**
   * Embed documents and store them.
   *
   * This is the high-level entry point: pass raw `Document` objects and the
   * configured `Embeddings` instance takes care of producing the vectors.
   * Calls `addVectors` internally after embedding.
   */
  async addDocuments(
    documents: DocumentInterface[],
    options?: Record<string, unknown>,
  ): Promise<string[] | void> {
    const texts = documents.map((d) => d.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return this.addVectors(vectors, documents as Document[]);
  }

  /**
   * Search for the `k` nearest neighbours of a pre-computed query vector.
   *
   * Uses the HNSW index when built, falls back to brute-force scan otherwise.
   * Tombstoned (TTL-expired) documents are silently excluded — WASM handles
   * this natively; the JS map guard is a belt-and-suspenders safety check.
   *
   * @param query  384-dim float query vector (already embedded).
   * @param k  Number of results to return.
   * @returns  `[Document, similarityScore]` pairs, score ∈ [0, 1], descending.
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
  ): Promise<[Document, number][]> {
    if (query.length !== WASM_DIMS) {
      throw new RangeError(
        `[LangChainMiniVecDb] similaritySearchVectorWithScore: ` +
          `expected ${WASM_DIMS}-dim query, got ${query.length}.`,
      );
    }

    const hits = this.db.search(new Float32Array(query), { limit: k });

    const results: [Document, number][] = [];
    for (const hit of hits) {
      const entry = this.docs.get(hit.id);
      // Guard: WASM won't return tombstoned IDs, but defend against
      // the rare race where the GC tick fires between WASM search and
      // JS map lookup on the same event-loop turn.
      if (entry !== undefined) {
        results.push([entry.doc, hit.score]);
      }
    }
    return results;
  }

  // ----- Index management ---------------------------------------------------

  /**
   * Build (or rebuild) the HNSW approximate nearest-neighbour index.
   *
   * Call this after a bulk `addDocuments` for faster subsequent searches.
   * The index is invalidated automatically when a tombstone slot is recycled
   * by a new insert.
   */
  buildIndex(): void {
    this.db.buildIndex();
  }

  // ----- TTL / GC -----------------------------------------------------------

  /**
   * Run one GC cycle immediately, outside the scheduled interval.
   *
   * Useful for testing or when you want eager cleanup after a known
   * long-lived document expires.
   *
   * @returns  Number of documents garbage-collected.
   */
  runGc(): number {
    if (this.ttlMs <= 0) return 0;
    return this._runGcTick();
  }

  // ----- Lifecycle ----------------------------------------------------------

  /**
   * Stop the GC timer and release WASM memory.
   *
   * **Must be called** when the agent task is complete; forgetting this leaks
   * the background `setInterval` and the WASM linear-memory allocation.
   */
  destroy(): void {
    if (this.gcTimer !== null) {
      clearInterval(this.gcTimer);
      this.gcTimer = null;
    }
    this.docs.clear();
    this.db.dispose();
  }

  // ----- Private ------------------------------------------------------------

  /**
   * One GC tick: tombstone WASM and mirror-evict the JS document map.
   *
   * Both sides use the same TTL cutoff so they stay in lock-step.
   * The WASM `run_gc` reads `Date.now()` internally; the JS side computes
   * `Date.now() - ttlMs` here, so the two timestamps are within the same
   * event-loop turn (sub-millisecond difference in practice).
   */
  private _runGcTick(): number {
    const wasmTombstoned = this.db.runGc(this.ttlMs);

    const cutoff = Date.now() - this.ttlMs;
    for (const [id, entry] of this.docs) {
      if (entry.insertedAt < cutoff) {
        this.docs.delete(id);
      }
    }

    return wasmTombstoned;
  }
}
