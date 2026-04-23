import type { WasmVecDbInstance } from './wasm-loader.js';
import { loadWasm } from './wasm-loader.js';
import * as opfs from './opfs.js';
import type {
  DbOptions,
  InsertOptions,
  SearchOptions,
  SearchResult,
  DbStats,
} from './types.js';

const DEFAULT_CAPACITY = 1024;
const DEFAULT_M = 16;
const DEFAULT_EF_CONSTRUCTION = 200;
const DEFAULT_LIMIT = 10;
const VECTOR_DIMS = 384;

/**
 * MicroVecDB — ultra-fast, offline-first vector database for the browser.
 *
 * ```ts
 * const db = await MicroVecDB.init({ capacity: 10_000, persistenceKey: 'my-db' });
 * db.insert({ id: 1, vector: embedding });
 * const results = db.search(queryEmbedding, { limit: 5 });
 * ```
 */
export class MicroVecDB {
  #inner: WasmVecDbInstance;
  #opts: Required<DbOptions>;
  #saveTimer: ReturnType<typeof setTimeout> | null = null;

  private constructor(inner: WasmVecDbInstance, opts: Required<DbOptions>) {
    this.#inner = inner;
    this.#opts = opts;
  }

  // ----- Factory -----------------------------------------------------------

  /**
   * Load the WASM engine and return a ready-to-use database instance.
   *
   * If `persistenceKey` is set, the database is automatically restored from
   * OPFS (or localStorage) if a previous save exists.
   */
  static async init(options: DbOptions = {}): Promise<MicroVecDB> {
    const opts: Required<DbOptions> = {
      capacity:           options.capacity            ?? DEFAULT_CAPACITY,
      persistenceKey:     options.persistenceKey      ?? null,
      hnswM:              options.hnswM               ?? DEFAULT_M,
      hnswEfConstruction: options.hnswEfConstruction  ?? DEFAULT_EF_CONSTRUCTION,
    };

    const glue = await loadWasm();

    let inner: WasmVecDbInstance;
    if (opts.persistenceKey) {
      const saved = await opfs.load(opts.persistenceKey);
      if (saved) {
        try {
          inner = glue.WasmVecDb.deserialize(saved);
        } catch {
          // Corrupted save — start fresh
          inner = glue.WasmVecDb.with_capacity(opts.capacity);
        }
      } else {
        inner = glue.WasmVecDb.with_capacity(opts.capacity);
      }
    } else {
      inner = glue.WasmVecDb.with_capacity(opts.capacity);
    }

    return new MicroVecDB(inner, opts);
  }

  // ----- Insert ------------------------------------------------------------

  /**
   * Insert one vector. Synchronous; takes < 1 ms for typical embeddings.
   *
   * @throws {RangeError} if `id` is not a non-negative integer < 2^31.
   * @throws {RangeError} if `vector.length !== 384`.
   * @throws {TypeError} if any vector element is NaN or Infinity.
   */
  insert(options: InsertOptions): void {
    assertValidId(options.id, 'insert');
    const vec = toFloat32Array(options.vector);
    assertValidVector(vec, 'insert');
    const slot = this.#inner.insert(options.id, vec);
    if (slot === 0xFFFFFFFF) {
      throw new RangeError(
        `[MicroVecDB] insert failed for id=${options.id} — database may be at capacity`
      );
    }
    this.#scheduleSave();
  }

  /**
   * Batch insert — more efficient than repeated `insert()` calls.
   *
   * @returns Number of successfully inserted vectors.
   * @throws {RangeError} if any `id` is invalid or any vector has wrong dimensions.
   * @throws {TypeError} if any vector element is NaN or Infinity.
   */
  insertBatch(vectors: InsertOptions[]): number {
    if (vectors.length === 0) return 0;

    const ids = new Uint32Array(vectors.length);
    const flat = new Float32Array(vectors.length * VECTOR_DIMS);

    for (let i = 0; i < vectors.length; i++) {
      assertValidId(vectors[i].id, `insertBatch[${i}]`);
      const v = toFloat32Array(vectors[i].vector);
      assertValidVector(v, `insertBatch[${i}]`);
      ids[i] = vectors[i].id;
      flat.set(v, i * VECTOR_DIMS);
    }

    const count = this.#inner.insert_batch(ids, flat);
    if (count > 0) this.#scheduleSave();
    return count;
  }

  // ----- Search ------------------------------------------------------------

  /**
   * Find the `limit` closest vectors to `query`.
   *
   * Uses the HNSW index when available (`useIndex: true`, default); falls
   * back to a brute-force scan otherwise.
   *
   * @param query  384-dimensional query vector.
   * @param opts   Search options.
   * @throws {RangeError} if `query.length !== 384`.
   * @throws {TypeError} if any query element is NaN or Infinity.
   */
  search(query: Float32Array | number[], opts: SearchOptions = {}): SearchResult[] {
    const qv = toFloat32Array(query);
    assertValidVector(qv, 'search');
    const k  = opts.limit    ?? DEFAULT_LIMIT;
    const ef = opts.ef       ?? k * 2;
    const useIdx = (opts.useIndex !== false) && this.#inner.has_index();

    const raw = useIdx
      ? this.#inner.search_hnsw(qv, k, ef)
      : this.#inner.search_scan(qv, k);

    return decodeResults(raw);
  }

  // ----- Index -------------------------------------------------------------

  /**
   * Build (or rebuild) the HNSW approximate index.
   *
   * Call this after a bulk insert. Subsequent `search()` calls automatically
   * use the index unless `useIndex: false` is passed.
   */
  buildIndex(): void {
    this.#inner.build_index(this.#opts.hnswM, this.#opts.hnswEfConstruction);
  }

  // ----- Mutation ----------------------------------------------------------

  /**
   * Soft-delete the vector with `id`.
   *
   * Deleted vectors are excluded from searches immediately but their memory
   * is reclaimed only after calling {@link compact}.
   *
   * @returns `true` if the vector was found and marked deleted.
   */
  delete(id: number): boolean {
    const ok = this.#inner.delete(id);
    if (ok) this.#scheduleSave();
    return ok;
  }

  /**
   * Remove all soft-deleted slots and compact memory.
   *
   * **Note**: the HNSW index is cleared after compaction.  Call
   * {@link buildIndex} again if needed.
   *
   * @returns Number of freed slots.
   */
  compact(): number {
    return this.#inner.compact();
  }

  // ----- Persistence -------------------------------------------------------

  /**
   * Explicitly save the database to OPFS (or localStorage).
   * This is also called automatically (debounced) after every mutation.
   */
  async save(): Promise<void> {
    if (!this.#opts.persistenceKey) return;
    const bytes = this.#inner.serialize();
    await opfs.save(this.#opts.persistenceKey, bytes);
  }

  /**
   * Delete the persisted snapshot for this database.
   */
  async clearSave(): Promise<void> {
    if (!this.#opts.persistenceKey) return;
    await opfs.remove(this.#opts.persistenceKey);
  }

  // ----- Zero-copy access --------------------------------------------------

  /**
   * Returns a live `Uint32Array` view into WASM linear memory.
   *
   * This is a zero-copy view — no data is duplicated.  The view is
   * **invalidated** after any `insert` that causes the internal Vec to
   * reallocate (use `with_capacity` upfront to prevent this).
   *
   * @example
   * ```ts
   * const view = db.rawVecsView(); // Uint32Array
   * // Each 12 consecutive u32s = one 384-bit quantised vector
   * ```
   */
  rawVecsView(memory: WebAssembly.Memory): Uint32Array {
    console.warn(
      '[MicroVecDB] rawVecsView() exposes raw WASM linear memory. ' +
      'This view is invalidated after any insert that reallocates the internal Vec. ' +
      'Intended for debugging only — do not use in production.'
    );
    return new Uint32Array(
      memory.buffer,
      this.#inner.raw_vecs_ptr(),
      this.#inner.raw_vecs_len(),
    );
  }

  // ----- Stats -------------------------------------------------------------

  /** Number of slots (including soft-deleted). */
  get count(): number { return this.#inner.len(); }

  /** Whether an HNSW index is ready. */
  get hasIndex(): boolean { return this.#inner.has_index(); }

  /** Returns a snapshot of database statistics. */
  stats(): DbStats {
    return { count: this.count, hasIndex: this.hasIndex };
  }

  // ----- TTL / GC ----------------------------------------------------------

  /**
   * Tombstone every active vector whose age exceeds `ttlMs` milliseconds.
   *
   * The timestamp is read from `Date.now()` inside the WASM engine; no
   * argument is needed from the caller.  Tombstoned vectors are excluded
   * from all subsequent searches without requiring a rebuild.
   *
   * @returns Number of vectors tombstoned.
   */
  runGc(ttlMs: number): number {
    return this.#inner.run_gc(ttlMs);
  }

  // ----- Lifecycle ---------------------------------------------------------

  /**
   * Free the WASM memory backing this instance.
   * The instance must not be used after calling this.
   */
  dispose(): void {
    if (this.#saveTimer !== null) {
      clearTimeout(this.#saveTimer);
      this.#saveTimer = null;
    }
    this.#inner.free();
  }

  // ----- Private -----------------------------------------------------------

  /** Debounced auto-save (500 ms after last mutation). */
  #scheduleSave(): void {
    if (!this.#opts.persistenceKey) return;
    if (this.#saveTimer !== null) clearTimeout(this.#saveTimer);
    this.#saveTimer = setTimeout(() => {
      this.save().catch(() => { /* silently swallow save errors */ });
      this.#saveTimer = null;
    }, 500);
  }
}

// ----- Helpers -------------------------------------------------------------

function toFloat32Array(v: Float32Array | number[]): Float32Array {
  return v instanceof Float32Array ? v : new Float32Array(v);
}

/**
 * Validate a vector before passing it to the WASM boundary.
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

/** Validate a document ID before passing it to the WASM boundary. */
function assertValidId(id: number, context: string): void {
  if (!Number.isInteger(id) || id < 0 || id >= 2 ** 31) {
    throw new RangeError(
      `[MicroVecDB] ${context}: id must be a non-negative integer < 2^31, got ${id}`
    );
  }
}

/**
 * Decode the interleaved `Uint32Array` returned by `search_scan` / `search_hnsw`.
 * Layout: [id₀, dist₀, id₁, dist₁, ...]
 */
function decodeResults(raw: Uint32Array): SearchResult[] {
  const results: SearchResult[] = [];
  for (let i = 0; i + 1 < raw.length; i += 2) {
    const dist = raw[i + 1];
    results.push({
      id:       raw[i],
      distance: dist,
      score:    1 - dist / VECTOR_DIMS,
    });
  }
  return results;
}
