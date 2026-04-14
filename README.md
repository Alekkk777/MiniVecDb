# MicroVecDB

**50 KB · 0 runtime dependencies · 32× less RAM than pgvector · runs entirely in the browser**

A vector database compiled from Rust to WebAssembly. It stores embeddings with 1-bit quantisation, indexes them with HNSW, and searches in microseconds — all inside the browser tab, with no server, no Python, and no data leaving the user's device.

```
npm install @microvecdb/core
```

---

## The problem with every other vector DB

| Tool | Where it runs | RAM / vector | Network round-trip |
|---|---|---|---|
| pgvector | Postgres server | 1,536 B (f32×384) | yes |
| Pinecone | Cloud | — | yes |
| Qdrant | Docker | ~1,600 B | yes |
| Chroma | Python process | ~1,600 B | yes |
| **MicroVecDB** | **Browser tab** | **48 B** | **never** |

Server-based databases are the right choice for multi-user production systems. But for single-user apps (note tools, document readers, local RAG, image galleries) they introduce unnecessary latency, infrastructure cost, and a privacy problem: the user's data travels to your server.

MicroVecDB eliminates all three.

---

## Benchmarks

These numbers are from the included example apps running on a 2023 MacBook Pro (M2), Safari 17.

| Metric | Result | Notes |
|---|---|---|
| **Search latency** (10k vectors) | **0.08 ms** | HNSW, ef=64 |
| **Search latency** (50k vectors) | **0.31 ms** | HNSW, ef=64 |
| **Batch insert** | **0.5 µs / vector** | bulk WASM call |
| **Index build** (10k vectors) | **~180 ms** | M=16, ef_construction=200 |
| **RAM per vector** (384-dim) | **48 B** | vs 1,536 B for f32 |
| **RAM — 1M vectors** | **48 MB** | vs 1.5 GB for f32 |
| **Recall@5** (semantic, real embeddings) | **100%** | all-MiniLM-L6-v2, 20-doc corpus |
| **Recall@5** (visual, pHash fingerprint) | **≥ 95%** | 10 clusters × 5 variants |
| **WASM binary size** | **50 KB** | brotli-compressed: 38 KB |
| **JS wrapper size** | **17 KB** | ESM, tree-shakeable |
| **Runtime dependencies** | **0** | pure WASM + thin JS glue |

> **Recall@5 = 100%** on real sentence embeddings was measured by embedding 20 topic-diverse paragraphs with `all-MiniLM-L6-v2` (384-dim), quantising to 1-bit, building the HNSW index, and querying with the first 50 chars of each paragraph. Every correct document appeared in the top-5 results.

---

## How we achieved these numbers

### 1. 1-bit quantisation — 32× RAM, almost no recall loss

Every `Float32Array(384)` is compressed to 12 × `u32` (384 bits = 48 bytes):

```
f32[384] → sign(x - μ) → bit[384] → u32[12]
```

The sign bit captures which side of the median each dimension falls on. For L2-normalised embeddings (unit hypersphere), this preserves the rank order of nearest neighbours with very high fidelity — semantically close vectors share the vast majority of their sign bits.

**Why it works:** sentence embeddings from models like `all-MiniLM-L6-v2` are not uniformly distributed. Each semantic concept activates a consistent subset of dimensions. After normalisation, vectors for "quantum computing" and "machine learning" land in different regions of the hypersphere, so their sign-bit patterns differ in ~150–200 positions out of 384. Vectors for "quantum computing" and "qubit entanglement" differ in only ~10–30 positions. This 5–15× separation ratio is large enough for HNSW to navigate reliably.

### 2. Hamming distance — ~10× faster than cosine

Once quantised, similarity is computed as:

```rust
fn hamming(a: &[u32; 12], b: &[u32; 12]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}
```

`XOR + count_ones` maps to a single CPU instruction (`POPCNT`) on every modern chip. Comparing two 384-bit vectors takes ~12 POPCNT operations. On the same hardware, comparing two `f32[384]` vectors with dot product takes 384 multiplications + 383 additions. Hamming wins by roughly 10–15×.

### 3. HNSW — O(log n) approximate nearest neighbour

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph:

- **Layer 0**: all vectors, connected to their `M=16` closest neighbours
- **Layer k**: a random ~37% subset of layer k-1 (geometric distribution)

Search starts at the top layer (few nodes, long-range edges) and greedily descends to layer 0 (many nodes, short-range edges). This gives O(log n) average complexity.

Parameters used:
- `M = 16` — edges per node at layer 0
- `ef_construction = 200` — candidate list size during index build (higher = better recall, slower build)
- `ef_search = 64` — candidate list size during search (tunable at query time)

The combination of **HNSW navigation + Hamming distance** means each hop in the graph costs ~12 instructions instead of ~767. On a 10k-vector index, search visits ~80 candidates on average and finishes in 0.08 ms.

### 4. Rust → WASM → browser

The core data structures (flat arena storage, HNSW graph) are written in Rust and compiled to WASM with `wasm-pack`. Key choices:

- **`lol_alloc`**: a minimal allocator for WASM that avoids the 30KB overhead of the default `wee_alloc`
- **`#[inline(always)]`** on hot paths: ensures the Hamming distance function gets inlined by the WASM JIT
- **Flat arena storage**: all vectors stored as a contiguous `Vec<[u32;12]>` — cache-friendly, no pointer chasing
- **`wasm-bindgen`**: zero-copy transfer of `Float32Array` from JS to WASM (no serialisation)

### 5. OPFS persistence — instant reload, no server

The serialised index is written to the browser's [Origin Private File System](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system) using the `FileSystemSyncAccessHandle` API. This gives file-system-level I/O speed (~500 MB/s) without leaving the browser sandbox. On page reload, the 48 MB index for 1M vectors loads in ~100 ms.

---

## Quickstart

```ts
import { MicroVecDB } from '@microvecdb/core';

// 1. Create a database
const db = await MicroVecDB.init({ capacity: 10_000 });

// 2. Insert vectors (Float32Array of length 384)
db.insert({ id: 1, vector: embedding });

// Bulk insert — faster, single WASM call
db.insertBatch([
  { id: 2, vector: vec2 },
  { id: 3, vector: vec3 },
]);

// 3. Build the HNSW index (call once after bulk inserts)
db.buildIndex();

// 4. Search — returns top-k results sorted by similarity
const results = db.search(queryEmbedding, { limit: 5 });
// → [{ id: 1, score: 0.94 }, { id: 3, score: 0.87 }, …]
```

---

## Full API

### `MicroVecDB.init(options?)`

```ts
const db = await MicroVecDB.init({
  capacity: 10_000,          // pre-allocate slots; grows automatically (default: 1024)
  persistenceKey: 'my-app',  // OPFS key for persistence; null = ephemeral (default: null)
  m: 16,                     // HNSW edges per node (default: 16)
  efConstruction: 200,       // HNSW build quality (default: 200)
});
```

### `db.insert({ id, vector })`

```ts
db.insert({ id: 42, vector: new Float32Array(384) });
```

- `id`: non-negative integer (up to 2^31 - 1)
- `vector`: `Float32Array` of exactly 384 elements, all finite

### `db.insertBatch(items)`

```ts
db.insertBatch([
  { id: 0, vector: v0 },
  { id: 1, vector: v1 },
]);
```

Single WASM call — 5–10× faster than calling `insert()` in a loop.

### `db.buildIndex()`

Builds the HNSW graph. Must be called once after bulk inserts. Subsequent `insert()` calls after `buildIndex()` are inserted into the graph incrementally.

### `db.search(queryVector, options?)`

```ts
const results = db.search(queryVec, {
  limit: 5,   // number of results (default: 10)
  ef: 64,     // HNSW search quality; higher = better recall, slower (default: 64)
});
// → Array<{ id: number, score: number }>
// score ∈ [0, 1] — 1 means identical vector
```

### `db.delete(id)`

```ts
const deleted = db.delete(42); // → true if found
```

Marks the slot as deleted. Deleted slots are excluded from search results. Compact after many deletions with `db.compact()`.

### `db.save()` / `db.load()`

```ts
await db.save();   // writes serialised index to OPFS (requires persistenceKey)
await db.load();   // restores from OPFS
```

### `db.stats()`

```ts
const { size, indexBuilt, capacityUsed } = db.stats();
```

### `db.dispose()`

```ts
db.dispose(); // frees WASM memory — call when done
```

---

### SharedMicroVecDB — non-blocking via Web Worker

Runs the database in a SharedWorker so inserts and searches don't block the main thread:

```ts
import { SharedMicroVecDB } from '@microvecdb/core/worker';

const db = await SharedMicroVecDB.init({ capacity: 100_000 });
await db.insertBatch(items);
await db.buildIndex();
const results = await db.search(queryVec, { limit: 5 });
```

Same API as `MicroVecDB`, all methods return `Promise`.

---

## Use cases

### 1. Local RAG — semantic search over documents

Embed a PDF with `@xenova/transformers` (`all-MiniLM-L6-v2`, 384-dim), store chunks in MicroVecDB, search with natural language. No API key. No server. Works offline.

→ See [`examples/pdf-brain`](examples/pdf-brain)

### 2. Visual similarity search

Extract a perceptual fingerprint from images with the Canvas API (zero ML), store in MicroVecDB, click any image to find visually similar ones. 500+ images indexed in < 200ms.

→ See [`examples/visual-search`](examples/visual-search)

### 3. Privacy-first note search

Index the user's notes locally. Search by meaning, not keywords. Notes never leave the device. Combine with `window.ai` (Chrome) or `@xenova/transformers` for fully local embeddings.

### 4. Offline-capable apps

Persist the index to OPFS with `persistenceKey`. The index survives page reloads and works without network. Ideal for PWAs.

### 5. Edge / browser extensions

At 50 KB, MicroVecDB fits in a browser extension without inflating the bundle. Use it for semantic deduplication, smart bookmarks, or local recommendation systems.

### 6. E-commerce visual search

Let users upload a photo and find similar products. No ML model needed — the perceptual fingerprint approach works for colour/shape matching in < 1ms.

---

## Vite setup

Add to `vite.config.ts`:

```ts
export default defineConfig({
  optimizeDeps: {
    exclude: ['@microvecdb/core'],  // prevents esbuild from mangling import.meta.url
  },
  assetsInclude: ['**/*.wasm'],
  server: {
    fs: { allow: ['../..'] },       // needed in monorepos
  },
});
```

For OPFS persistence and SharedWorker mode, add COOP/COEP headers:

```ts
server: {
  headers: {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  },
},
```

---

## Security model

| Layer | Mechanism | Why |
|---|---|---|
| Runtime privacy | JS `#` private fields | Prevents external code from reading internal WASM pointers |
| Input validation | `assertValidVector`, `assertValidId` | Rejects NaN, Infinity, wrong length before they reach WASM |
| Cross-origin isolation | COOP + COEP headers | Required for `SharedArrayBuffer` (multi-threaded WASM mode) |
| Supply chain | SRI hashes in `dist/sri-hashes.json` | Verify build artefacts with `npm run generate-sri` |

---

## Running the examples

From the repo root:

```bash
npm install

# PDF Brain — semantic search over PDFs
npm run dev --workspace=examples/pdf-brain
# → http://localhost:5173

# Visual Matcher — instant image similarity
npm run dev --workspace=examples/visual-search
# → http://localhost:5174
```

---

## Development

```bash
# Build WASM + TypeScript
npm run build --workspace=packages/core

# Build + generate SRI hashes
npm run build:full --workspace=packages/core

# Tests
npm test --workspaces --if-present

# Publish to npm
npm run release --workspace=packages/core  # runs build:full then npm publish
```

Requires: `rustup`, `wasm-pack`, Node.js 18+.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

---

## License

MIT
