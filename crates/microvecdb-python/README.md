# minivecdb

**Ultra-fast 1-bit quantised vector database with TTL garbage collection — Rust native extension for Python.**

```bash
pip install minivecdb
pip install "minivecdb[langchain]"  # + LangChain adapter
```

Same core as [`@microvecdb/core`](https://github.com/Alekkk777/MiniVecDb) (browser / Node.js) — compiled from the same Rust codebase via PyO3 + Maturin.

## Quick start

```python
from minivecdb import MiniVecDb
import numpy as np, time

db = MiniVecDb(capacity=10_000)

vec = np.random.randn(384).astype(np.float32)
vec /= np.linalg.norm(vec)

db.insert(id=0, vector=vec.tolist(), inserted_at=time.time() * 1000)
db.build_index(m=16, ef_construction=200)

results = db.search(vec.tolist(), limit=5)
# → [{"id": 0, "score": 1.0, "distance": 0}]
```

## LangChain adapter

```python
from langchain_openai import OpenAIEmbeddings
from minivecdb.langchain import LangChainMiniVecDb

# Ephemeral agent scratchpad — 10 min TTL
with LangChainMiniVecDb(
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    ttl_minutes=10,
    gc_interval_sec=30,
) as memory:
    memory.add_texts(["User mentioned ticket #42-ABC."])
    docs = memory.similarity_search("what is the ticket number?", k=3)
```

## TTL & GC

Every text has a wall-clock TTL. A daemon GC thread tombstones expired vectors automatically. Set `ttl_minutes=0` (default) to disable.

```python
count = store.run_gc()  # manual GC cycle — returns tombstone count
store.destroy()         # stop GC thread, free native memory
```

## Benchmarks

| Metric | Result |
|---|---|
| Search latency (10k vectors) | 0.08 ms |
| RAM per vector (384-dim) | 48 B (vs 1,536 B f32) |
| Recall@5 (sentence embeddings) | 100% |

## License

MIT
