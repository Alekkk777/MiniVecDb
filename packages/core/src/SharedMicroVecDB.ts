/**
 * SharedMicroVecDB — same API as MicroVecDB but backed by a SharedWorker.
 *
 * One WebAssembly instance runs inside the worker and is shared across every
 * browser tab of the same origin.  This means:
 *   - A single OPFS file is written even when the user has 10 tabs open.
 *   - Tabs never fight over the same database (no double-write corruption).
 *   - Memory usage is constant regardless of tab count.
 *
 * ```ts
 * // Tab A
 * const db = await SharedMicroVecDB.init({ persistenceKey: 'my-db' });
 * db.insert({ id: 1, vector: embedding });
 *
 * // Tab B (same origin, same SharedWorker instance)
 * const db2 = await SharedMicroVecDB.init({ persistenceKey: 'my-db' });
 * const results = await db2.search(queryEmbedding, { limit: 5 });
 * // → Sees the vector inserted by Tab A
 * ```
 *
 * All mutation methods (`insert`, `insertBatch`, `delete`, `compact`) are
 * async because they cross the port boundary.  `search` is also async for
 * the same reason.  `buildIndex` may take seconds on large databases — await
 * it before issuing searches if you need up-to-date results.
 */

import type {
  DbOptions,
  InsertOptions,
  SearchOptions,
  SearchResult,
  DbStats,
} from './types.js';
import type { WorkerRequestBody, WorkerRequest, WorkerResponse, DbStatsPayload } from './worker-protocol.js';

const DEFAULT_CAPACITY           = 1024;
const DEFAULT_M                  = 16;
const DEFAULT_EF_CONSTRUCTION    = 200;
const DEFAULT_LIMIT              = 10;
const VECTOR_DIMS                = 384;

// Increment on every request so we can match responses.
let nextReqId = 1;

export class SharedMicroVecDB {
  #port: MessagePort;
  #opts: Required<DbOptions>;
  // reqId → { resolve, reject }
  #pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  private constructor(port: MessagePort, opts: Required<DbOptions>) {
    this.#port = port;
    this.#opts = opts;

    this.#port.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const res = e.data;
      const handler = this.#pending.get(res.reqId);
      if (!handler) return;
      this.#pending.delete(res.reqId);

      if (res.type === 'ok') {
        handler.resolve(res.payload);
      } else {
        handler.reject(new Error(res.message));
      }
    };
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  /**
   * Connect to the shared worker and initialise (or join) the shared database.
   *
   * @param workerUrl  URL to `worker.js` (the compiled worker script).
   *                   In Vite: `new URL('./worker.ts', import.meta.url)`
   * @param options    Same options as `MicroVecDB.init()`.
   */
  static async init(
    workerUrl: string | URL,
    options: DbOptions = {},
  ): Promise<SharedMicroVecDB> {
    const opts: Required<DbOptions> = {
      capacity:           options.capacity            ?? DEFAULT_CAPACITY,
      persistenceKey:     options.persistenceKey      ?? null,
      hnswM:              options.hnswM               ?? DEFAULT_M,
      hnswEfConstruction: options.hnswEfConstruction  ?? DEFAULT_EF_CONSTRUCTION,
    };

    const worker = new SharedWorker(workerUrl, { type: 'module', name: 'microvecdb' });
    const port = worker.port;
    port.start();

    const client = new SharedMicroVecDB(port, opts);

    await client.#send({
      type:               'init',
      capacity:           opts.capacity,
      persistenceKey:     opts.persistenceKey,
      hnswM:              opts.hnswM,
      hnswEfConstruction: opts.hnswEfConstruction,
    });

    return client;
  }

  // ── Insert ─────────────────────────────────────────────────────────────────

  /**
   * @throws {RangeError} if `id` is not a non-negative integer < 2^31.
   * @throws {RangeError} if `vector.length !== 384`.
   * @throws {TypeError} if any vector element is NaN or Infinity.
   */
  async insert(options: InsertOptions): Promise<void> {
    assertValidId(options.id, 'insert');
    const vec = toF32(options.vector);
    assertValidVector(vec, 'insert');
    await this.#send({ type: 'insert', id: options.id, vector: vec }, [vec.buffer]);
  }

  /**
   * @throws {RangeError} if any `id` is invalid or any vector has wrong dimensions.
   * @throws {TypeError} if any vector element is NaN or Infinity.
   */
  async insertBatch(vectors: InsertOptions[]): Promise<number> {
    if (vectors.length === 0) return 0;
    const ids = new Uint32Array(vectors.length);
    const flat = new Float32Array(vectors.length * VECTOR_DIMS);
    for (let i = 0; i < vectors.length; i++) {
      assertValidId(vectors[i].id, `insertBatch[${i}]`);
      const v = toF32(vectors[i].vector);
      assertValidVector(v, `insertBatch[${i}]`);
      ids[i] = vectors[i].id;
      flat.set(v, i * VECTOR_DIMS);
    }
    return this.#send({ type: 'insert_batch', ids, flat }, [ids.buffer, flat.buffer]) as Promise<number>;
  }

  // ── Search ─────────────────────────────────────────────────────────────────

  /**
   * @throws {RangeError} if `query.length !== 384`.
   * @throws {TypeError} if any query element is NaN or Infinity.
   */
  async search(query: Float32Array | number[], opts: SearchOptions = {}): Promise<SearchResult[]> {
    const qv = toF32(query);
    assertValidVector(qv, 'search');
    const k  = opts.limit    ?? DEFAULT_LIMIT;
    const ef = opts.ef       ?? k * 2;

    // Delegate brute-force / HNSW decision to the worker based on opts.useIndex.
    const useIdx = opts.useIndex !== false;
    let raw: Uint32Array;
    if (useIdx) {
      raw = await this.#send(
        { type: 'search_hnsw', query: qv, k, ef },
        [qv.buffer],
      ) as Uint32Array;
    } else {
      raw = await this.#send(
        { type: 'search_scan', query: qv, k },
        [qv.buffer],
      ) as Uint32Array;
    }
    return decodeResults(raw);
  }

  /**
   * @throws {RangeError} if `query.length !== 384`.
   * @throws {TypeError} if any query element is NaN or Infinity.
   */
  async searchScan(query: Float32Array | number[], k = DEFAULT_LIMIT): Promise<SearchResult[]> {
    const qv = toF32(query);
    assertValidVector(qv, 'searchScan');
    const raw = await this.#send(
      { type: 'search_scan', query: qv, k },
      [qv.buffer],
    ) as Uint32Array;
    return decodeResults(raw);
  }

  // ── Index ──────────────────────────────────────────────────────────────────

  async buildIndex(): Promise<void> {
    await this.#send({ type: 'build_index', m: this.#opts.hnswM, efConstruction: this.#opts.hnswEfConstruction });
  }

  // ── Mutation ───────────────────────────────────────────────────────────────

  async delete(id: number): Promise<boolean> {
    return this.#send({ type: 'delete', id }) as Promise<boolean>;
  }

  async compact(): Promise<number> {
    return this.#send({ type: 'compact' }) as Promise<number>;
  }

  // ── Persistence ────────────────────────────────────────────────────────────

  async save(): Promise<void> {
    await this.#send({ type: 'save' });
  }

  async clearSave(): Promise<void> {
    await this.#send({ type: 'clear_save' });
  }

  async serialize(): Promise<Uint8Array> {
    return this.#send({ type: 'serialize' }) as Promise<Uint8Array>;
  }

  // ── Stats ──────────────────────────────────────────────────────────────────

  async stats(): Promise<DbStats> {
    const s = await this.#send({ type: 'stats' }) as DbStatsPayload;
    return { count: s.count, hasIndex: s.hasIndex };
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  /** Disconnect from the worker port.  The worker keeps running. */
  disconnect(): void {
    this.#port.close();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  #send(
    msg: WorkerRequestBody,
    transfer: Transferable[] = [],
  ): Promise<unknown> {
    const reqId = nextReqId++;
    const full: WorkerRequest = { ...msg, reqId } as WorkerRequest;
    return new Promise((resolve, reject) => {
      this.#pending.set(reqId, { resolve, reject });
      this.#port.postMessage(full, transfer);
    });
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function toF32(v: Float32Array | number[]): Float32Array {
  return v instanceof Float32Array ? v : new Float32Array(v);
}

/**
 * Validate a vector before passing it across the worker port boundary.
 * Short-circuits on the first invalid element.
 */
function assertValidVector(vec: Float32Array, context: string): void {
  if (vec.length !== VECTOR_DIMS) {
    throw new RangeError(
      `[MicroVecDB] ${context}: expected vector of length ${VECTOR_DIMS}, got ${vec.length}`
    );
  }
  if (!vec.every(Number.isFinite)) {
    const badIdx = vec.findIndex(x => !Number.isFinite(x));
    throw new TypeError(
      `[MicroVecDB] ${context}: vector[${badIdx}] = ${vec[badIdx]} — NaN and Infinity are not valid embeddings`
    );
  }
}

/** Validate a document ID before passing it across the worker port boundary. */
function assertValidId(id: number, context: string): void {
  if (!Number.isInteger(id) || id < 0 || id >= 2 ** 31) {
    throw new RangeError(
      `[MicroVecDB] ${context}: id must be a non-negative integer < 2^31, got ${id}`
    );
  }
}

function decodeResults(raw: Uint32Array): SearchResult[] {
  const results: SearchResult[] = [];
  for (let i = 0; i + 1 < raw.length; i += 2) {
    const dist = raw[i + 1];
    results.push({ id: raw[i], distance: dist, score: 1 - dist / VECTOR_DIMS });
  }
  return results;
}
