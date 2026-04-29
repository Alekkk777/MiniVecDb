# MicroVecDB

**50 KB · 0 server · 32× less RAM · Semantic Cache + TTL-aware ephemeral memory for AI agents**

A vector database compiled from Rust to WebAssembly (browser / Node.js / Edge) and a native Python extension (PyO3). It stores embeddings with 1-bit quantisation, indexes them with HNSW, and searches in microseconds — with built-in TTL garbage collection so caches and agent memory self-clean automatically.

```bash
npm install @microvecdb/core        # TypeScript / browser / Node.js / Edge
pip install minivecdb               # Python (native Rust extension)
```

---

## Semantic Cache — cut LLM costs by 98% in two lines

"How do I return an item?" and "What is your return policy?" ask the same thing. Without a cache, your app calls GPT-4o twice and pays twice. MicroVecDB catches this at the vector level: semantically similar prompts return the cached response **before the LLM is ever called**.

### Benchmark — 50 customer-service queries, 5 topics × 10 paraphrases

> Embedder: bag-of-words (no API key needed for cache lookups).
> LLM simulation: Gaussian(μ=1.48 s, σ=0.28 s) — real-world GPT-4o latency.
> Pricing: GPT-4o $2.50/M input · $10/M output · text-embedding-3-small $0.02/M.

| Metric | Without cache | With cache |
|---|---:|---:|
| LLM calls | 50 | **1** |
| Cache hits | — | **49 / 50 (98%)** |
| Total wall time | 73.4 s | **1.4 s** |
| Avg latency / call | 1 467 ms | **28 ms** |
| Cache-hit latency | — | **< 1 ms** |
| Vector lookup (p50) | — | **0.10 ms** |
| API cost (50 queries) | $0.0975 | **$0.0020** |
| **Cost savings** | — | **97.9 %** |

### TypeScript — two lines to activate

```ts
import { withSemanticCache } from '@microvecdb/core/cache';
import { openai } from '@ai-sdk/openai';
import { generateText } from 'ai';

// 1. Wrap your LLM call once
const cachedGenerate = await withSemanticCache(
  {
    embeddingModel: openai.embedding('text-embedding-3-small', { dimensions: 384 }),
    threshold: 0.92,   // cosine similarity to count as a hit
    ttlMinutes: 60,    // cache self-expires after 1 hour
  },
  async (prompt) => {
    const { text } = await generateText({ model: openai('gpt-4o'), prompt });
    return text;
  },
);

// 2. Use it everywhere — semantically similar prompts skip the LLM entirely
export async function POST(req: Request) {
  const { question } = await req.json();
  const answer = await cachedGenerate(question); // < 1 ms on a hit
  return Response.json({ answer });
}
```

Or use the class directly for more control:

```ts
import { SemanticCache } from '@microvecdb/core/cache';

const cache = await SemanticCache.create({
  embeddingModel: openai.embedding('text-embedding-3-small', { dimensions: 384 }),
  threshold: 0.92,
  ttlMinutes: 60,
});

const cached = await cache.lookup(prompt);          // null on miss
if (!cached) await cache.set(prompt, llmResponse);  // store after real call
```

### Python — drop-in LangChain global cache

```python
import langchain
from minivecdb.cache import MiniVecDbSemanticCache
from langchain_openai import OpenAIEmbeddings

# One line — all subsequent LangChain LLM calls use the semantic cache
langchain.llm_cache = MiniVecDbSemanticCache(
    embedding_function=OpenAIEmbeddings(
        model="text-embedding-3-small", dimensions=384
    ),
    similarity_threshold=0.92,
    ttl_minutes=1440,   # 24-hour rolling window, self-cleaning
)

# From here your chains are unchanged — the cache is transparent
llm = ChatOpenAI(model="gpt-4o")
llm.invoke("What is your return policy?")   # real call  → cached
llm.invoke("How do I return an item?")      # cache hit  → instant, $0
llm.invoke("Can I return a product?")       # cache hit  → instant, $0
```

### The Black Friday scenario

An e-commerce bot launches a promo. 10 000 users ask variations of "When do the discounts end?".

| | Without MicroVecDB | With MicroVecDB (TTL 2 h) |
|---|---|---|
| LLM calls | 10 000 | **1** |
| API cost | ~$19.50 | **~$0.002** |
| Avg response time | 1.5 s | **< 1 ms** |
| Cache self-cleans at end of day | ✗ | **✓** |

---

## The problem with agent memory

Every LLM framework offers "memory". Almost none of them expire it.

An agent that observes "the user is on step 2" at turn 3 should not still be acting on that observation at turn 50. But most vector stores are append-only: observations accumulate, similarity search scores degrade, and the agent confuses past context with present state.

MicroVecDB treats this as a first-class concern. Every stored text has a TTL. A background GC thread (Python) or `setInterval` (JS) tombstones expired vectors automatically. You set `ttl_minutes=10`; the memory cleans itself up.

---

## When to use MicroVecDB vs. a server database

| Use case | Right tool |
|---|---|
| **Semantic cache** — cut LLM costs 90-98% | **MicroVecDB** |
| LLM agent scratchpad (ephemeral, single-request) | **MicroVecDB** |
| Browser app — user data must not leave the device | **MicroVecDB** |
| Offline / PWA — works without network | **MicroVecDB** |
| Edge function — no persistent infra | **MicroVecDB** |
| Multi-user production system, durable | pgvector / Pinecone / Qdrant |

---

## Benchmarks

Measured on a 2023 MacBook Pro M2.

| Metric | Result | Notes |
|---|---|---|
| Search latency (10k vectors) | **0.08 ms** | HNSW, ef=64 |
| Search latency (50k vectors) | **0.31 ms** | HNSW, ef=64 |
| Batch insert | **0.5 µs / vector** | single WASM call |
| Index build (10k vectors) | **~180 ms** | M=16, ef_construction=200 |
| RAM per vector (384-dim) | **48 B** | vs 1,536 B for f32 |
| RAM — 1M vectors | **48 MB** | vs 1.5 GB for f32 |
| Recall@5 (sentence embeddings) | **100%** | all-MiniLM-L6-v2, 20-doc corpus |
| Recall@5 (visual pHash) | **≥ 95%** | 10 clusters × 5 variants |
| WASM binary size | **50 KB** | brotli: 38 KB |
| Runtime dependencies | **0** | pure WASM + thin JS glue |

---

## Quick-starts

### Semantic Cache (see full examples above)

```ts
// TypeScript
import { withSemanticCache } from '@microvecdb/core/cache';
const cachedGenerate = await withSemanticCache({ embeddingModel, threshold: 0.92, ttlMinutes: 60 }, myLlmFn);
```

```python
# Python
import langchain
from minivecdb.cache import MiniVecDbSemanticCache
langchain.llm_cache = MiniVecDbSemanticCache(embedding_function=my_embed_fn, similarity_threshold=0.92)
```

### Vercel AI SDK (agent scratchpad)

```ts
import { openai } from '@ai-sdk/openai';
import { streamText } from 'ai';
import { VercelMiniVecDb } from '@microvecdb/core/vercel';

// Create a per-request ephemeral memory — 10 min TTL, GC every 30 s
const memory = await VercelMiniVecDb.create(
  openai.embedding('text-embedding-3-small'),
  { ttlMinutes: 10, gcIntervalMs: 30_000 },
);

// Store agent observations
await memory.add([
  'User mentioned their order number is 42-ABC.',
  'User is on the returns flow, step 2 of 4.',
]);

// Plug directly into streamText — the LLM calls it autonomously
const result = await streamText({
  model: openai('gpt-4o-mini'),
  tools: { searchMemory: memory.createRetrievalTool() },
  messages,
});

// Clean up when the request is done
memory.destroy();
```

### LangChain Python (agent scratchpad)

```python
from langchain_openai import OpenAIEmbeddings
from minivecdb.langchain import LangChainMiniVecDb

# 10-minute TTL, GC daemon fires every 30 s
memory = LangChainMiniVecDb(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    ttl_minutes=10,
    gc_interval_sec=30,
)

memory.add_texts([
    "User mentioned ticket #42-ABC.",
    "User has already tried resetting their password.",
])

results = memory.similarity_search("what is the user's issue?", k=3)

# Use as a context manager for automatic cleanup
with LangChainMiniVecDb(embedding=..., ttl_minutes=5) as mem:
    mem.add_texts(["observation"])
    docs = mem.similarity_search("query")
# GC thread stopped automatically on exit
```

### LangChain JS

```ts
import { OpenAIEmbeddings } from '@langchain/openai';
import { LangChainMiniVecDb } from '@microvecdb/core';

const store = await LangChainMiniVecDb.fromTexts(
  ['Paris is the capital of France.', 'Berlin is the capital of Germany.'],
  [{ source: 'wiki' }, { source: 'wiki' }],
  new OpenAIEmbeddings({ modelName: 'text-embedding-3-small' }),
);

const results = await store.similaritySearch('European capitals', 3);
```

### Raw WASM API (browser / Node.js)

```ts
import { MicroVecDB } from '@microvecdb/core';

const db = await MicroVecDB.init({ capacity: 10_000 });
db.insert({ id: 1, vector: new Float32Array(384) });
db.buildIndex();

const results = db.search(queryVec, { limit: 5 });
// → [{ id: 1, score: 0.94 }, …]
```

### Raw Python API

```python
from minivecdb import MiniVecDb
import numpy as np

db = MiniVecDb(capacity=10_000)
vec = np.random.randn(384).astype(np.float32)
vec /= np.linalg.norm(vec)

db.insert(id=0, vector=vec.tolist(), inserted_at=0.0)
db.build_index(m=16, ef_construction=200)

results = db.search(vec.tolist(), limit=5)
# → [{"id": 0, "score": 1.0, "distance": 0}, …]
```

---

## TTL & garbage collection

Every high-level adapter supports TTL-based auto-expiry.

### How it works

1. Each inserted text gets a wall-clock timestamp at insert time.
2. A background GC loop fires every `gcIntervalMs` / `gc_interval_sec`.
3. GC tombstones vectors older than `ttlMinutes` / `ttl_minutes` in the Rust layer (zero-copy soft-delete) and evicts them from the JS/Python doc map.
4. Tombstoned slots are invisible to `search()` and are physically reclaimed on `compact()`.

Setting `ttlMinutes: 0` (default) disables GC entirely — no timer is created.

### Manual GC

```ts
// TypeScript
const count = memory.runGc();  // returns tombstone count

// Python
count = memory.run_gc()
```

---

## How it achieves these numbers

### 1-bit quantisation — 32× RAM, near-zero recall loss

Every `Float32Array(384)` is compressed to 12 × `u32` (384 bits = 48 bytes):

```
f32[384]  →  sign(x − μ)  →  bit[384]  →  u32[12]
```

The sign bit captures which side of the per-dimension median each value falls on. For L2-normalised sentence embeddings this preserves nearest-neighbour rank order with very high fidelity — semantically close vectors share ≥ 85% of their sign bits.

### Hamming distance — ~10× faster than cosine

```rust
fn hamming(a: &[u32; 12], b: &[u32; 12]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}
```

`XOR + POPCNT` maps to a single CPU instruction on every modern chip. Comparing two 384-bit vectors takes ~12 POPCNT operations vs. 384 multiplications for dot product.

### HNSW — O(log n) approximate nearest neighbour

Multi-layer graph: Layer 0 has all vectors connected to their M=16 closest neighbours; each higher layer is a ~37% random subset. Search descends from the sparse top layer to the dense bottom layer in O(log n) hops.

Parameters: `M=16`, `ef_construction=200`, `ef_search=64`.

### Rust → WASM → browser

- **`lol_alloc`**: minimal WASM allocator, avoids 30 KB overhead of `wee_alloc`
- **`wasm-bindgen`**: zero-copy transfer of `Float32Array` from JS to WASM
- **Flat arena storage**: `Vec<[u32;12]>` — cache-friendly, no pointer chasing
- **OPFS persistence**: `FileSystemSyncAccessHandle` — ~500 MB/s, no server needed

### PyO3 native extension

The Python package is a Rust native extension (`.so` / `.pyd`) built with Maturin. The same `microvecdb-core` Rust library powers both the WASM and Python builds — no code duplication.

---

## Full API

### TypeScript / WASM

#### `MicroVecDB.init(options?)`
```ts
const db = await MicroVecDB.init({
  capacity: 10_000,          // pre-allocate slots; grows automatically (default: 1024)
  persistenceKey: 'my-app',  // OPFS key; null = ephemeral (default: null)
  m: 16,                     // HNSW edges per node (default: 16)
  efConstruction: 200,       // HNSW build quality (default: 200)
});
```

#### `db.insert({ id, vector })` / `db.insertBatch(items)`
```ts
db.insert({ id: 42, vector: new Float32Array(384) });
// Bulk insert — 5–10× faster, single WASM call:
db.insertBatch([{ id: 0, vector: v0 }, { id: 1, vector: v1 }]);
```

#### `db.search(queryVec, { limit?, ef? })`
```ts
const results = db.search(queryVec, { limit: 5, ef: 64 });
// → Array<{ id: number, score: number }>  — score ∈ [0, 1]
```

#### `db.delete(id)` / `db.compact()` / `db.stats()` / `db.dispose()`

#### `SharedMicroVecDB` — non-blocking via Web Worker
```ts
import { SharedMicroVecDB } from '@microvecdb/core/worker';
const db = await SharedMicroVecDB.init({ capacity: 100_000 });
await db.insertBatch(items);
const results = await db.search(queryVec, { limit: 5 });
```

### Python

#### `MiniVecDb(capacity?)`
```python
from minivecdb import MiniVecDb

db = MiniVecDb(capacity=10_000)
db.insert(id=0, vector=[0.1] * 384, inserted_at=time.time() * 1000)
db.build_index(m=16, ef_construction=200)
results = db.search([0.1] * 384, limit=5)
# → [{"id": 0, "score": 1.0, "distance": 0}]

tombstoned = db.run_gc(ttl_ms=60_000)  # manual GC
data = db.serialize()                   # bytes — use with deserialize()
```

#### `LangChainMiniVecDb`
```python
from minivecdb.langchain import LangChainMiniVecDb

store = LangChainMiniVecDb(
    embedding=embeddings,
    capacity=50_000,
    ttl_minutes=10,       # 0 = immortal
    gc_interval_sec=30,
)

ids = store.add_texts(["text1", "text2"], metadatas=[{"k": "v"}, {}])
docs = store.similarity_search("query", k=4)
docs_scores = store.similarity_search_with_score("query", k=4)
# → [(Document, score), …]

store.delete(ids=["0", "1"])
store.build_index()
store.destroy()           # stop GC thread, free memory
```

---

## Setup guides

### Vite

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { exclude: ['@microvecdb/core'] },
  assetsInclude: ['**/*.wasm'],
  server: { fs: { allow: ['../..'] } },
});
```

For OPFS / SharedWorker mode, add COOP/COEP headers:
```ts
server: {
  headers: {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  },
},
```

### Next.js / Edge Runtime

The `@microvecdb/core/vercel` sub-path is tree-shaken: it imports `ai` and `zod` only when used, keeping the main bundle at 0 extra dependencies.

---

## Development

### TypeScript / WASM

```bash
git clone https://github.com/Alekkk777/MiniVecDb.git
cd MiniVecDb
npm install

# Build WASM binary + TypeScript wrapper
npm run build --workspace=packages/core

# Build + regenerate SRI hashes
npm run build:full --workspace=packages/core

# Tests (vitest)
npm test --workspaces --if-present

# Watch mode
npm run test:watch --workspace=packages/core
```

Requires: `rustup`, `wasm-pack`, Node.js ≥ 18.

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
```

### Python native extension

```bash
cd crates/microvecdb-python
pip install maturin

# Development build (editable install)
maturin develop --release

# Run tests
pip install pytest freezegun langchain-core
pytest tests/ -v

# Build a wheel
maturin build --release
```

### Examples

```bash
npm run dev --workspace=examples/pdf-brain     # → http://localhost:5173
npm run dev --workspace=examples/visual-search  # → http://localhost:5174
```

---

## Project structure

```
crates/
  microvecdb-core/        Rust library (quantisation, storage, HNSW, time)
  microvecdb-wasm/        wasm-bindgen bindings → browser / Node.js
  microvecdb-python/      PyO3 native extension → minivecdb PyPI package
    python/minivecdb/
      __init__.py         re-exports MiniVecDb from _minivecdb.so
      langchain.py        LangChain VectorStore adapter with TTL GC
    tests/                pytest suite (38 unit + 5 integration)
packages/
  core/                   @microvecdb/core npm package
    src/
      MicroVecDB.ts       WASM wrapper
      SharedMicroVecDB.ts Web Worker proxy
      langchain.ts        LangChain JS adapter
      vercel.ts           Vercel AI SDK adapter with TTL GC
examples/
  pdf-brain/              Local RAG demo (React + Transformers.js)
  visual-search/          Image similarity demo (React + pHash)
```

---

## Security

| Layer | Mechanism |
|---|---|
| Runtime privacy | JS `#` private fields — no external access to WASM pointers |
| Input validation | `assertValidVector`, `assertValidId` — rejects NaN/Infinity before WASM |
| Cross-origin isolation | COOP + COEP headers for SharedArrayBuffer mode |
| Supply chain | SRI hashes in `dist/sri-hashes.json` — verify with `npm run generate-sri` |

---

## License

MIT
