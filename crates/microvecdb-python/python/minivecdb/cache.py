"""
minivecdb.cache — Semantic cache for LLM responses.

Stores prompt→response pairs as 384-dim embeddings. On each lookup it
finds the most similar cached prompt; if the cosine similarity exceeds
``similarity_threshold`` the stored response is returned instantly.

LangChain integration
---------------------
``MiniVecDbSemanticCache`` extends ``langchain_core.caches.BaseCache``
so it can be set as the global LangChain cache::

    import langchain
    from minivecdb.cache import MiniVecDbSemanticCache
    from langchain_openai import OpenAIEmbeddings

    langchain.llm_cache = MiniVecDbSemanticCache(
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-3-small", dimensions=384
        ),
        similarity_threshold=0.92,
        ttl_minutes=1440,  # 24 h
    )

Standalone usage
----------------
::

    from minivecdb.cache import MiniVecDbSemanticCache

    cache = MiniVecDbSemanticCache(embedding_function=my_embed_fn)
    cache.update("What are your opening hours?", "llm", [Generation(text="9am-6pm")])
    result = cache.lookup("When do you open?", "llm")  # → [Generation(text="9am-6pm")]
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional, Sequence, Union

from minivecdb import MiniVecDb

try:
    from langchain_core.caches import BaseCache
    from langchain_core.outputs import Generation

    RETURN_VAL_TYPE = Sequence[Generation]
    _HAS_LANGCHAIN = True
except ImportError:  # pragma: no cover
    BaseCache = object  # type: ignore[assignment,misc]
    Generation = None  # type: ignore[assignment]
    RETURN_VAL_TYPE = Any  # type: ignore[misc]
    _HAS_LANGCHAIN = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_embed_fn(
    embedding_function: Union[str, Callable, Any],
) -> Callable[[str], list[float]]:
    """
    Accept:
    • A plain callable ``(str) -> list[float]``
    • A LangChain ``Embeddings`` instance (has ``.embed_query``)
    • A string model name (OpenAI shortcut — requires ``langchain-openai``)
    """
    if callable(embedding_function) and not hasattr(embedding_function, "embed_query"):
        return embedding_function

    if hasattr(embedding_function, "embed_query"):
        return embedding_function.embed_query

    if isinstance(embedding_function, str):
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pip install langchain-openai to use a string model name, "
                "or pass an Embeddings instance / callable directly."
            ) from exc
        emb = OpenAIEmbeddings(model=embedding_function, dimensions=384)
        return emb.embed_query

    raise TypeError(
        "embedding_function must be a callable, a LangChain Embeddings instance, "
        f"or a model-name string. Got: {type(embedding_function)!r}"
    )


class _RepeatingTimer:
    """Daemon timer that re-schedules itself after each tick."""

    def __init__(self, interval_s: float, fn: Callable[[], None]) -> None:
        self._interval = interval_s
        self._fn = fn
        self._timer: Optional[threading.Timer] = None
        self._stopped = False

    def start(self) -> None:
        self._schedule()

    def _schedule(self) -> None:
        if self._stopped:
            return
        self._timer = threading.Timer(self._interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        self._fn()
        self._schedule()

    def cancel(self) -> None:
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()


# ── MiniVecDbSemanticCache ────────────────────────────────────────────────────

class MiniVecDbSemanticCache(BaseCache):  # type: ignore[misc]
    """
    LangChain-compatible semantic cache backed by MiniVecDb.

    Parameters
    ----------
    embedding_function:
        How to embed prompts into 384-dim vectors. Accepts:

        * A callable ``(str) -> list[float]``
        * A LangChain ``Embeddings`` instance (e.g. ``OpenAIEmbeddings``)
        * A string model name, e.g. ``"text-embedding-3-small"``
          (requires ``pip install langchain-openai``)

    similarity_threshold:
        Cosine similarity above which a lookup counts as a hit.
        Default 0.95. Lower → more hits, less accuracy.

    ttl_minutes:
        How many minutes a cached response lives before being evicted.
        0 (default) = immortal.

    gc_interval_seconds:
        How often the background GC thread runs. Default 60 s.
        Only relevant when ``ttl_minutes > 0``.

    db_instance:
        Optional pre-created ``MiniVecDb`` instance. A new one is created
        automatically if omitted.
    """

    def __init__(
        self,
        embedding_function: Union[str, Callable[[str], list[float]], Any],
        *,
        similarity_threshold: float = 0.95,
        ttl_minutes: float = 0,
        gc_interval_seconds: float = 60,
        db_instance: Optional[MiniVecDb] = None,
    ) -> None:
        if not _HAS_LANGCHAIN:  # pragma: no cover
            raise ImportError(
                "pip install langchain-core to use MiniVecDbSemanticCache."
            )
        self._db: MiniVecDb = db_instance if db_instance is not None else MiniVecDb()
        self._embed: Callable[[str], list[float]] = _make_embed_fn(embedding_function)
        self._threshold = similarity_threshold
        self._ttl_ms = ttl_minutes * 60_000
        self._entries: dict[int, dict[str, Any]] = {}
        self._next_id = 0
        self._lock = threading.Lock()

        self._timer: Optional[_RepeatingTimer] = None
        if self._ttl_ms > 0:
            self._timer = _RepeatingTimer(gc_interval_seconds, self._gc_tick)
            self._timer.start()

    # ----- BaseCache interface -----------------------------------------------

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Return cached generations for *prompt* or ``None`` on a miss."""
        with self._lock:
            if not self._entries:
                return None
            vec = self._embed(prompt)
            results = self._db.search(vec, k=1)
            if not results:
                return None
            doc_id, score = results[0]
            if score < self._threshold:
                return None
            entry = self._entries.get(doc_id)
            return entry["response"] if entry is not None else None

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE,
    ) -> None:
        """Store *return_val* under the embedding of *prompt*."""
        with self._lock:
            vec = self._embed(prompt)
            doc_id = self._next_id
            self._next_id += 1
            self._db.insert(doc_id, vec)
            self._entries[doc_id] = {
                "response": return_val,
                "inserted_at_ms": time.time() * 1_000,
            }

    def clear(self, **kwargs: Any) -> None:
        """Wipe the entire cache."""
        with self._lock:
            self._db = MiniVecDb()
            self._entries.clear()
            self._next_id = 0

    # ----- Extras ------------------------------------------------------------

    def run_gc(self) -> int:
        """Manually run one GC cycle. Returns number of entries evicted."""
        if self._ttl_ms <= 0:
            return 0
        return self._gc_tick()

    def destroy(self) -> None:
        """Stop the GC timer. Call when the cache is no longer needed."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    # ----- Private -----------------------------------------------------------

    def _gc_tick(self) -> int:
        with self._lock:
            tombstoned = self._db.run_gc(int(self._ttl_ms))
            cutoff_ms = time.time() * 1_000 - self._ttl_ms
            stale = [k for k, v in self._entries.items()
                     if v["inserted_at_ms"] < cutoff_ms]
            for k in stale:
                del self._entries[k]
            return tombstoned
