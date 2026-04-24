/**
 * WASM loading with two strategies:
 *   1. Browser / Edge: `new URL('./microvecdb_wasm_bg.wasm', import.meta.url)`
 *      → the bundler (Vite/webpack) or the ESM host resolves the URL, and
 *      wasm-bindgen uses WebAssembly.instantiateStreaming internally.
 *   2. Node.js (file:// URL): `fetch` does not support file:// URLs in Node.
 *      We read the binary directly with `fs.readFileSync` and pass the Buffer
 *      as a BufferSource to the wasm-bindgen init function.
 *
 * The compiled `WebAssembly.Module` is cached so that multiple `MicroVecDB.init()`
 * calls within the same page / process share the same compiled module.
 *
 * PACKAGING NOTE
 * The `.wasm` binary is copied into `dist/` by `postbuild:ts` so that it is
 * included in the npm package alongside the JS bundles.  The path below is
 * therefore always `./microvecdb_wasm_bg.wasm` relative to the chunk file.
 */

// We import the wasm-pack generated JS glue.
type WasmGlue = {
  default: (input?: RequestInfo | URL | BufferSource | WebAssembly.Module | { module_or_path: RequestInfo | URL | BufferSource | WebAssembly.Module }) => Promise<unknown>;
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
  run_gc(ttl_ms: number): number;
  serialize(): Uint8Array;
  raw_vecs_ptr(): number;
  raw_vecs_len(): number;
  len(): number;
  is_empty(): boolean;
  has_index(): boolean;
  free(): void;
};

let cachedGlue: WasmGlue | null = null;

export async function loadWasm(): Promise<WasmGlue> {
  if (cachedGlue) return cachedGlue;

  // The JS glue is bundled by tsup into the same dist/ chunk.
  const glue = await import('../../../pkg/microvecdb_wasm.js') as WasmGlue;

  // The .wasm binary lives next to the bundled chunk in dist/.
  const wasmUrl = new URL('./microvecdb_wasm_bg.wasm', import.meta.url);

  let wasmInput: RequestInfo | URL | BufferSource | WebAssembly.Module = wasmUrl;

  if (wasmUrl.protocol === 'file:') {
    // Node.js: fetch() does not support file:// — read bytes directly.
    const { readFileSync } = await import('node:fs');
    const { fileURLToPath } = await import('node:url');
    wasmInput = readFileSync(fileURLToPath(wasmUrl));
  }

  await glue.default({ module_or_path: wasmInput });

  cachedGlue = glue;
  return glue;
}
