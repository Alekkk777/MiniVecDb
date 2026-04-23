"""
LangChain ``VectorStore`` adapter for MiniVecDb.

Wraps the native Rust extension (``minivecdb.MiniVecDb``) in the
``langchain_core.vectorstores.VectorStore`` interface, adding:

- Background GC daemon thread for TTL-based document expiry.
- Thread-safe document store (integer ID → LangChain ``Document``).
- ``build_index()`` / ``run_gc()`` / ``destroy()`` extension methods.

Dimension constraint
--------------------
The WASM/native engine quantises **384-dimensional** float vectors.
Use a matching embedding model, e.g. ``all-MiniLM-L6-v2``.

Usage
-----
::

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from minivecdb.langchain import LangChainMiniVecDb

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    store = LangChainMiniVecDb.from_texts(
        texts=["Paris is in France.", "Berlin is in Germany."],
        embedding=embeddings,
        ttl_minutes=10,
        gc_interval_sec=30,
    )

    docs = store.similarity_search("European capitals", k=2)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from minivecdb._minivecdb import MiniVecDb

if TYPE_CHECKING:
    pass

# The native engine is hardcoded to 384-bit quantised vectors.
_DIMS: int = 384


class LangChainMiniVecDb(VectorStore):
    """LangChain ``VectorStore`` adapter for the MiniVecDb native engine.

    When ``ttl_minutes > 0`` a background daemon thread tombstones expired
    vectors in the Rust store and evicts their entries from the Python
    document map on every GC tick, turning this into a zero-maintenance
    Agent Scratchpad: stale context disappears automatically.

    Args:
        embedding:       Embedding model that produces 384-dim vectors.
        ttl_minutes:     How long documents stay alive (0 = immortal).
        gc_interval_sec: How often the GC thread wakes up (default 60 s).
        capacity:        Pre-allocate this many slots (0 = dynamic growth).
        **kwargs:        Forwarded to :class:`VectorStore`.
    """

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        embedding: Embeddings,
        *,
        ttl_minutes: int = 0,
        gc_interval_sec: int = 60,
        capacity: int = 0,
        **kwargs: Any,
    ) -> None:
        self._embedding = embedding
        self._db: MiniVecDb = (
            MiniVecDb.with_capacity(capacity) if capacity > 0 else MiniVecDb()
        )

        # ----- document store ------------------------------------------------
        # Maps internal integer slot-IDs → LangChain Documents.
        # Protected by _lock together with _next_id and _insertion_times.
        self._docstore: dict[int, Document] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

        # ----- TTL / GC ------------------------------------------------------
        self._ttl_sec: float = ttl_minutes * 60.0
        self._ttl_ms: float = self._ttl_sec * 1_000.0
        self._gc_interval_sec: int = gc_interval_sec

        # Only allocated when TTL is active, to avoid overhead for immortal stores.
        self._insertion_times: dict[int, float] = {}
        self._stop_event: threading.Event | None = None
        self._gc_thread: threading.Thread | None = None

        if self._ttl_ms > 0:
            self._insertion_times = {}
            self._stop_event = threading.Event()
            self._gc_thread = threading.Thread(
                target=self._gc_loop,
                daemon=True,          # never blocks interpreter shutdown
                name="minivecdb-gc",
            )
            self._gc_thread.start()

    # ------------------------------------------------------------------ props

    @property
    def embeddings(self) -> Embeddings:
        """The underlying embedding model."""
        return self._embedding

    # ------------------------------------------------------------------ write

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed ``texts`` and insert them into the store.

        Args:
            texts:     Texts to embed and store.
            metadatas: Optional per-text metadata dicts (same order as texts).
            ids:       Ignored — integer IDs are always generated internally.
            **kwargs:  Unused; accepted for interface compatibility.

        Returns:
            String representations of the internal integer IDs.

        Raises:
            ValueError: If ``metadatas`` length doesn't match ``texts``.
            ValueError: If the embedding model returns wrong-dimension vectors.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        # Embed outside the lock — I/O-bound and may be slow.
        vectors = self._embedding.embed_documents(texts_list)

        if len(vectors) != len(texts_list):
            raise ValueError(
                f"Embedding returned {len(vectors)} vectors for "
                f"{len(texts_list)} texts"
            )
        for i, vec in enumerate(vectors):
            if len(vec) != _DIMS:
                raise ValueError(
                    f"texts[{i}]: expected {_DIMS}-dim vector, got {len(vec)}. "
                    "Use a 384-dim embedding model such as all-MiniLM-L6-v2."
                )

        metas: list[dict] = metadatas if metadatas is not None else [{} for _ in texts_list]
        if len(metas) != len(texts_list):
            raise ValueError(
                f"Expected {len(texts_list)} metadatas, got {len(metas)}"
            )

        now = time.time()
        assigned: list[str] = []

        with self._lock:
            for text, vec, meta in zip(texts_list, vectors, metas):
                int_id = self._next_id
                self._next_id += 1
                # insert() raises RuntimeError if capacity is full (no tombstones).
                self._db.insert(int_id, vec)
                self._docstore[int_id] = Document(page_content=text, metadata=meta)
                if self._ttl_ms > 0:
                    self._insertion_times[int_id] = now
                assigned.append(str(int_id))

        return assigned

    # ------------------------------------------------------------------ read

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Return the ``k`` documents most similar to ``query``.

        Uses the HNSW index when built, brute-force scan otherwise.

        Args:
            query: Natural-language query string.
            k:     Number of documents to return.

        Returns:
            List of ``Document`` objects, best match first.
        """
        return [doc for doc, _ in self.similarity_search_with_score(query, k)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return ``(Document, score)`` pairs, score ∈ [0, 1], best first.

        Args:
            query: Natural-language query string.
            k:     Number of results to return.
        """
        query_vec = self._embedding.embed_query(query)
        if len(query_vec) != _DIMS:
            raise ValueError(
                f"Expected {_DIMS}-dim query vector, got {len(query_vec)}"
            )

        # Search outside the lock — pure Rust computation, no GIL needed.
        hits = self._db.search(query_vec, k)

        with self._lock:
            return [
                (self._docstore[int_id], score)
                for int_id, score in hits
                if int_id in self._docstore
            ]

    def get_by_ids(self, ids: list[str]) -> list[Document]:  # type: ignore[override]
        """Retrieve documents by their string IDs (as returned by ``add_texts``).

        Missing IDs are silently skipped.
        """
        with self._lock:
            return [
                self._docstore[int_id]
                for raw in ids
                if (int_id := int(raw)) in self._docstore
            ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Soft-delete documents by string ID.

        Returns ``True`` if at least one document was deleted, ``False`` if
        none of the IDs were found, ``None`` if ``ids`` was ``None``.
        """
        if ids is None:
            return None

        deleted_any = False
        with self._lock:
            for raw in ids:
                int_id = int(raw)
                if self._db.delete(int_id):
                    self._docstore.pop(int_id, None)
                    self._insertion_times.pop(int_id, None)
                    deleted_any = True

        return deleted_any

    # ------------------------------------------------------------------ index

    def build_index(self, m: int = 16, ef_construction: int = 200) -> None:
        """Build (or rebuild) the HNSW approximate nearest-neighbour index.

        Call this after a bulk ``add_texts`` for faster subsequent searches.

        Args:
            m:               Max node connections per layer (default 16).
            ef_construction: Exploration factor during build (default 200).
        """
        self._db.build_index(m, ef_construction)

    # ------------------------------------------------------------------ GC

    def run_gc(self) -> int:
        """Run one GC cycle immediately, outside the scheduled interval.

        Useful after a known long-lived document has expired, or in tests
        where you don't want to wait for the background thread.

        Returns:
            Number of documents garbage-collected.
        """
        if self._ttl_ms <= 0:
            return 0
        return self._gc_tick()

    # ------------------------------------------------------------------ lifecycle

    def destroy(self) -> None:
        """Stop the GC thread and release all resources.

        Safe to call multiple times (idempotent).
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._gc_thread is not None and self._gc_thread.is_alive():
            self._gc_thread.join(timeout=5.0)
        with self._lock:
            self._docstore.clear()
            self._insertion_times.clear()

    def __enter__(self) -> "LangChainMiniVecDb":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    # ------------------------------------------------------------------ factory

    @classmethod
    def from_texts(  # type: ignore[override]
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "LangChainMiniVecDb":
        """Create a store, embed ``texts``, and insert them in one call.

        All ``kwargs`` are forwarded to ``__init__`` (e.g. ``ttl_minutes``,
        ``gc_interval_sec``, ``capacity``).

        Returns:
            A ready-to-query ``LangChainMiniVecDb`` instance.
        """
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    # ------------------------------------------------------------------ private

    def _gc_loop(self) -> None:
        """Background GC loop — runs in the daemon thread."""
        assert self._stop_event is not None
        # wait() returns True when the event fires (destroy called), False on timeout.
        # So we keep looping as long as it returns False (normal timeout path).
        while not self._stop_event.wait(timeout=self._gc_interval_sec):
            self._gc_tick()

    def _gc_tick(self) -> int:
        """One GC cycle: tombstone in Rust, evict from Python map.

        Both sides use the same TTL cutoff, computed in the same call to
        avoid drift between the Rust clock and Python's ``time.time()``.

        Returns:
            Number of documents tombstoned by the Rust engine.
        """
        # Rust side: tombstones slots whose age > ttl_ms (uses SystemTime internally).
        rust_count = self._db.run_gc(self._ttl_ms)

        # Python side: evict from docstore by checking insertion timestamps.
        cutoff = time.time() - self._ttl_sec
        with self._lock:
            expired = [
                id_
                for id_, inserted_at in self._insertion_times.items()
                if inserted_at < cutoff
            ]
            for id_ in expired:
                del self._insertion_times[id_]
                self._docstore.pop(id_, None)

        return rust_count
