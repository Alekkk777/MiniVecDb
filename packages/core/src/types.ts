/** Options for initialising a MicroVecDB instance. */
export interface DbOptions {
  /** Pre-allocate space for this many vectors. Default: 1024. */
  capacity?: number;
  /**
   * OPFS key for automatic persistence. When set, the database is
   * automatically saved to the Origin Private File System on every insert
   * and restored on init. Null (default) disables persistence.
   */
  persistenceKey?: string | null;
  /** HNSW M parameter (bi-directional links per node). Default: 16. */
  hnswM?: number;
  /** HNSW ef_construction parameter. Default: 200. */
  hnswEfConstruction?: number;
}

/** A single vector to insert into the database. */
export interface InsertOptions {
  /** Caller-supplied document ID. Must be < 2^31. */
  id: number;
  /** 384-dimensional float32 embedding vector. */
  vector: Float32Array | number[];
}

/** Options for a KNN search. */
export interface SearchOptions {
  /** Number of results to return. Default: 10. */
  limit?: number;
  /**
   * HNSW exploration factor. Higher = better recall, slower.
   * Default: `limit * 2`. Only used when an HNSW index has been built.
   */
  ef?: number;
  /**
   * When `true` (default), use the HNSW index if available.
   * Set to `false` to force a brute-force scan.
   */
  useIndex?: boolean;
}

/** One result returned by a search. */
export interface SearchResult {
  /** Document ID supplied at insert time. */
  id: number;
  /**
   * Hamming distance between the query and this vector.
   * Range: [0, 384]. 0 = identical, 384 = completely opposite.
   */
  distance: number;
  /**
   * Normalised similarity score: `1 - distance / 384`.
   * Range: [0.0, 1.0]. 1.0 = identical.
   */
  score: number;
}

/** Snapshot of database statistics. */
export interface DbStats {
  /** Total slot count (including soft-deleted slots). */
  count: number;
  /** Whether an HNSW index has been built. */
  hasIndex: boolean;
}
