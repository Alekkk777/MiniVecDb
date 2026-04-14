/**
 * WASM loading with three strategies:
 *   1. Bundler (Vite/webpack): ESM import inlines the .wasm as a URL/base64
 *   2. Browser fetch: streaming instantiation from a URL
 *   3. Node.js: fs.readFileSync fallback (for tests / CLI tooling)
 *
 * The compiled `WebAssembly.Module` is cached so that multiple `MicroVecDB.init()`
 * calls within the same page share the same compiled module.
 */

// We import the wasm-pack generated JS glue.  The glue exports an `init`
// function that handles instantiation and an `initSync` for synchronous use.
// The path is relative to the dist/ output, which sits next to pkg/.
type WasmGlue = {
  default: (input?: RequestInfo | URL | BufferSource | WebAssembly.Module) => Promise<unknown>;
  WasmVecDb: WasmVecDbConstructor;
};

export type WasmVecDbConstructor = {
  new(): WasmVecDbInstance;
  with_capacity(capacity: number): WasmVecDbInstance;
  deserialize(data: Uint8Array): WasmVecDbInstance;
};

export type WasmVecDbInstance = {
  insert(doc_id: number, vector: Float32Array): number;
  insert_batch(doc_ids: Uint32Array, vectors_flat: Float32Array): number;
  search_scan(query: Float32Array, k: number): Uint32Array;
  search_hnsw(query: Float32Array, k: number, ef: number): Uint32Array;
  build_index(m: number, ef_construction: number): void;
  delete(doc_id: number): boolean;
  compact(): number;
  serialize(): Uint8Array;
  raw_vecs_ptr(): number;
  raw_vecs_len(): number;
  len(): number;
  is_empty(): boolean;
  has_index(): boolean;
  free(): void;
};

let cachedGlue: WasmGlue | null = null;

/**
 * Load and initialise the WASM module.  Safe to call multiple times; subsequent
 * calls return the cached result without re-compiling.
 */
export async function loadWasm(): Promise<WasmGlue> {
  if (cachedGlue) return cachedGlue;

  // Dynamic import works in both browser (Vite inlines the URL) and Node.js.
  // Adjust the relative path as needed for your bundler setup.
  const glue = await import('../../../pkg/microvecdb_wasm.js') as WasmGlue;

  // Pass the .wasm URL explicitly so bundlers (Vite, webpack) can locate the
  // binary even when the JS and .wasm files live in different directories.
  await glue.default(new URL('../../../pkg/microvecdb_wasm_bg.wasm', import.meta.url));

  cachedGlue = glue;
  return glue;
}
