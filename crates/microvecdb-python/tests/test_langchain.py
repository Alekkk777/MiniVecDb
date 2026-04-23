"""
Prova del Fuoco — Python LangChain adapter test suite.

Strategy
--------
Unit tests use a pure-Python ``MockMiniVecDb`` in place of the native
extension.  The mock reads ``time.time()`` for every timestamp, which
``freezegun`` controls completely.  Advancing the frozen clock by 70 s
is semantically identical to waiting 70 real seconds — the GC sees the
same age arithmetic as in production.

Background-thread tests use real OS threads with a tiny GC interval and
a threading.Event to coordinate without arbitrary sleeps.

Integration tests (``-m integration``) use the real native extension
with a 20 ms TTL to exercise the full Rust ↔ Python pipeline.

Running
-------
    # unit tests only (fast, no native Rust needed):
    pytest tests/

    # all tests including integration:
    pytest tests/ -m "integration or not integration"
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch, call

import freezegun
import pytest
from freezegun import freeze_time

# Prevent freezegun from scanning broken modules in the conda environment
# (stale transformers submodules that import removed asyncio.coroutine).
freezegun.configure(extend_ignore_list=["transformers", "datasets", "tokenizers"])
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from minivecdb.langchain import LangChainMiniVecDb

# ─────────────────────────────────────────────────────────────────────────────
# Shared constants
# ─────────────────────────────────────────────────────────────────────────────

T0 = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)   # arbitrary frozen origin


# ─────────────────────────────────────────────────────────────────────────────
# Fake embeddings
# ─────────────────────────────────────────────────────────────────────────────

class FakeEmbeddings(Embeddings):
    """All texts → fixed 384-dim [0.5, …] vector.  Zero I/O."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * 384


EMB = FakeEmbeddings()


# ─────────────────────────────────────────────────────────────────────────────
# MockMiniVecDb — pure-Python stand-in for the native extension
#
# Mirrors the five methods LangChainMiniVecDb calls on its inner _db.
# insert() and run_gc() both read time.time(), which freezegun controls —
# so time-based expiry works exactly as in production.
# ─────────────────────────────────────────────────────────────────────────────

class MockMiniVecDb:
    def __init__(self, capacity: int = 0) -> None:
        self._capacity = capacity
        # slot_id → {"vec", "deleted", "inserted_at"}
        self._slots: dict[int, dict[str, Any]] = {}

    @staticmethod
    def with_capacity(capacity: int) -> "MockMiniVecDb":
        return MockMiniVecDb(capacity=capacity)

    # ----- write --------------------------------------------------------------

    def insert(self, id: int, vector: list[float]) -> int:
        # At capacity: recycle first tombstone, else raise
        if self._capacity > 0 and len(self._slots) >= self._capacity:
            for slot_id, slot in self._slots.items():
                if slot["deleted"]:
                    slot.update(vec=vector, deleted=False, inserted_at=time.time())
                    return slot_id
            raise RuntimeError("CapacityExceeded: no tombstone slots available")
        self._slots[id] = {
            "vec": vector,
            "deleted": False,
            "inserted_at": time.time(),   # ← fake clock when tests are running
        }
        return id

    def delete(self, id: int) -> bool:
        slot = self._slots.get(id)
        if slot and not slot["deleted"]:
            slot["deleted"] = True
            return True
        return False

    # ----- read ---------------------------------------------------------------

    def search(self, query: list[float], k: int, ef: int = 50) -> list[tuple[int, float]]:
        return [
            (id_, 1.0)
            for id_, slot in self._slots.items()
            if not slot["deleted"]
        ][:k]

    # ----- GC -----------------------------------------------------------------

    def run_gc(self, ttl_ms: float) -> int:
        now = time.time()   # ← fake clock controls this too
        ttl_sec = ttl_ms / 1_000.0
        tombstoned = 0
        for slot in self._slots.values():
            if not slot["deleted"] and (now - slot["inserted_at"]) > ttl_sec:
                slot["deleted"] = True
                tombstoned += 1
        return tombstoned

    # ----- misc ---------------------------------------------------------------

    def build_index(self, m: int = 16, ef_construction: int = 200) -> None:
        pass  # no-op in mock


# ─────────────────────────────────────────────────────────────────────────────
# Pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> type[MockMiniVecDb]:
    """Replace MiniVecDb in the langchain module with MockMiniVecDb."""
    monkeypatch.setattr("minivecdb.langchain.MiniVecDb", MockMiniVecDb)
    return MockMiniVecDb


@pytest.fixture
def store(mock_db: type[MockMiniVecDb]) -> LangChainMiniVecDb:
    """An immortal (ttl_minutes=0) store backed by the mock."""
    s = LangChainMiniVecDb(EMB)
    yield s
    s.destroy()


def make_ttl_store(
    ttl_minutes: int = 1,
    gc_interval_sec: int = 9_999,  # huge → background thread never fires in tests
    **kwargs: Any,
) -> LangChainMiniVecDb:
    """TTL store wired to the mock (caller must patch first) and manual GC only."""
    return LangChainMiniVecDb(
        EMB, ttl_minutes=ttl_minutes, gc_interval_sec=gc_interval_sec, **kwargs
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. add_texts / similarity_search
# ═════════════════════════════════════════════════════════════════════════════

class TestStorageAndRetrieval:
    def test_single_document_round_trip(self, store: LangChainMiniVecDb) -> None:
        ids = store.add_texts(["The Eiffel Tower is in Paris."])
        assert ids == ["0"]
        results = store.similarity_search("Paris landmark", k=5)
        assert len(results) == 1
        assert results[0].page_content == "The Eiffel Tower is in Paris."

    def test_multiple_documents(self, store: LangChainMiniVecDb) -> None:
        store.add_texts(["Rome", "Berlin", "Tokyo"])
        assert len(store.similarity_search("capitals", k=10)) == 3

    def test_metadata_round_trip(self, store: LangChainMiniVecDb) -> None:
        store.add_texts(["meta doc"], metadatas=[{"source": "wiki", "page": 42}])
        result = store.similarity_search("doc", k=1)[0]
        assert result.metadata == {"source": "wiki", "page": 42}

    def test_similarity_search_with_score(self, store: LangChainMiniVecDb) -> None:
        store.add_texts(["scored doc"])
        pairs = store.similarity_search_with_score("anything", k=5)
        assert len(pairs) == 1
        doc, score = pairs[0]
        assert doc.page_content == "scored doc"
        assert 0.0 < score <= 1.0

    def test_add_texts_returns_sequential_string_ids(
        self, store: LangChainMiniVecDb
    ) -> None:
        ids = store.add_texts(["a", "b", "c"])
        assert ids == ["0", "1", "2"]
        assert all(isinstance(i, str) for i in ids)

    def test_empty_add_texts_returns_empty_list(
        self, store: LangChainMiniVecDb
    ) -> None:
        assert store.add_texts([]) == []

    def test_wrong_vector_dimension_raises_value_error(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        class WrongDimEmb(Embeddings):
            def embed_documents(self, t: list[str]) -> list[list[float]]:
                return [[0.1] * 512 for _ in t]

            def embed_query(self, t: str) -> list[float]:
                return [0.1] * 512

        s = LangChainMiniVecDb(WrongDimEmb())
        with pytest.raises(ValueError, match="384"):
            s.add_texts(["x"])

    def test_metadatas_length_mismatch_raises_value_error(
        self, store: LangChainMiniVecDb
    ) -> None:
        with pytest.raises(ValueError):
            store.add_texts(["a", "b"], metadatas=[{"k": 1}])  # 2 texts, 1 meta

    def test_query_wrong_dimension_raises_value_error(
        self, store: LangChainMiniVecDb
    ) -> None:
        store.add_texts(["hello"])

        class ShortQueryEmb(FakeEmbeddings):
            def embed_query(self, t: str) -> list[float]:
                return [0.5] * 128

        store._embedding = ShortQueryEmb()
        with pytest.raises(ValueError, match="384"):
            store.similarity_search("oops", k=1)


# ═════════════════════════════════════════════════════════════════════════════
# 2. from_texts factory
# ═════════════════════════════════════════════════════════════════════════════

class TestFromTexts:
    def test_creates_ready_store(self, mock_db: type[MockMiniVecDb]) -> None:
        s = LangChainMiniVecDb.from_texts(["doc A", "doc B", "doc C"], embedding=EMB)
        assert len(s.similarity_search("anything", k=10)) == 3
        s.destroy()

    def test_kwargs_forwarded_to_init(self, mock_db: type[MockMiniVecDb]) -> None:
        s = LangChainMiniVecDb.from_texts(
            ["x"], embedding=EMB, ttl_minutes=5, gc_interval_sec=9_999
        )
        assert s._ttl_sec == 300.0
        s.destroy()

    def test_metadatas_forwarded(self, mock_db: type[MockMiniVecDb]) -> None:
        s = LangChainMiniVecDb.from_texts(
            ["tagged"], embedding=EMB, metadatas=[{"tag": "test"}]
        )
        assert s.similarity_search("x", k=1)[0].metadata == {"tag": "test"}
        s.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# 3. get_by_ids / delete
# ═════════════════════════════════════════════════════════════════════════════

class TestGetByIdsAndDelete:
    def test_get_by_ids_returns_correct_docs(
        self, store: LangChainMiniVecDb
    ) -> None:
        ids = store.add_texts(["alpha", "beta", "gamma"])
        fetched = store.get_by_ids(ids[:2])
        assert [d.page_content for d in fetched] == ["alpha", "beta"]

    def test_get_by_ids_skips_missing_ids(
        self, store: LangChainMiniVecDb
    ) -> None:
        store.add_texts(["x"])
        assert store.get_by_ids(["9999"]) == []

    def test_delete_removes_from_search(self, store: LangChainMiniVecDb) -> None:
        ids = store.add_texts(["keep me", "delete me"])
        assert store.delete([ids[1]]) is True
        results = store.similarity_search("anything", k=10)
        assert all(d.page_content != "delete me" for d in results)

    def test_delete_unknown_id_returns_false(
        self, store: LangChainMiniVecDb
    ) -> None:
        assert store.delete(["99999"]) is False

    def test_delete_none_returns_none(self, store: LangChainMiniVecDb) -> None:
        assert store.delete(None) is None


# ═════════════════════════════════════════════════════════════════════════════
# 4.  🔥  Prova del Fuoco — TTL garbage collection
#
# freeze_time() controls time.time() globally, including inside MockMiniVecDb
# and inside LangChainMiniVecDb._gc_tick().  Advancing the frozen clock by
# 70 s is semantically identical to waiting 70 real seconds.
#
# Why two separate freeze_time() blocks instead of a single decorator:
#   • The outer block fixes the insertion timestamp.
#   • The inner block advances the clock for the GC check.
#   • Returning to the outer block (between assertions) restores the original
#     time — no accidental cross-contamination between test steps.
# ═════════════════════════════════════════════════════════════════════════════

class TestTTLGarbageCollection:
    # ── 4a. Classic fire test ─────────────────────────────────────────────

    def test_document_found_before_expiry_gone_after(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """Insert at T=0, confirm found; advance 70 s past 60 s TTL, confirm gone."""
        with freeze_time(T0):
            store = make_ttl_store(ttl_minutes=1)
            store.add_texts(["Ephemeral agent observation: user asked about weather."])

            # ── T = 0: doc just inserted, must be findable ───────────────────
            assert len(store.similarity_search("weather", k=5)) == 1

        # ── T = +70 s: past the TTL ──────────────────────────────────────────
        with freeze_time(T0 + timedelta(seconds=70)):
            n = store._gc_tick()
            assert n == 1                                            # one tombstone

            assert len(store.similarity_search("weather", k=5)) == 0  # gone ✓

        store.destroy()

    # ── 4b. Correct tombstone count ───────────────────────────────────────

    def test_gc_returns_correct_tombstone_count(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """Three docs inserted at T=0; all three must be tombstoned at T=+70 s."""
        with freeze_time(T0):
            store = make_ttl_store(ttl_minutes=1)
            store.add_texts(["doc A", "doc B", "doc C"])

        with freeze_time(T0 + timedelta(seconds=70)):
            assert store._gc_tick() == 3

        store.destroy()

    # ── 4c. Selective expiry ──────────────────────────────────────────────

    def test_selective_expiry_old_gone_fresh_survives(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """doc A (T=0) expires at T=+70 s; doc B (T=+30 s) is still alive."""
        with freeze_time(T0):
            store = make_ttl_store(ttl_minutes=1)
            store.add_texts(["doc A — old"])                       # age at +70 s: 70 s > TTL

        with freeze_time(T0 + timedelta(seconds=30)):
            store.add_texts(["doc B — fresh"])                     # age at +70 s: 40 s < TTL

        with freeze_time(T0 + timedelta(seconds=70)):
            n = store._gc_tick()
            assert n == 1                                          # only doc A expired

            results = store.similarity_search("doc", k=10)
            contents = {d.page_content for d in results}
            assert "doc B — fresh" in contents
            assert "doc A — old"   not in contents

        store.destroy()

    # ── 4d. Document within TTL is never tombstoned ───────────────────────

    def test_document_within_ttl_not_expired(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """At T=+30 s (half the 60 s TTL) the doc must still be alive."""
        with freeze_time(T0):
            store = make_ttl_store(ttl_minutes=1)
            store.add_texts(["still alive at half-TTL"])

        with freeze_time(T0 + timedelta(seconds=30)):
            n = store._gc_tick()
            assert n == 0
            assert len(store.similarity_search("alive", k=5)) == 1

        store.destroy()

    # ── 4e. run_gc() is a no-op for immortal stores ───────────────────────

    def test_run_gc_noop_when_ttl_zero(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        store = LangChainMiniVecDb(EMB)                            # ttl_minutes=0
        store.add_texts(["immortal doc"])
        assert store.run_gc() == 0
        assert len(store.similarity_search("immortal", k=5)) == 1
        store.destroy()

    # ── 4f. Manual run_gc() without waiting for interval ─────────────────

    def test_manual_run_gc_tombstones_after_ttl(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """Huge gc_interval_sec → timer never fires; call run_gc() explicitly."""
        with freeze_time(T0):
            store = make_ttl_store(ttl_minutes=1, gc_interval_sec=999_999)
            store.add_texts(["manual gc test"])

        with freeze_time(T0 + timedelta(seconds=70)):
            n = store.run_gc()
            assert n == 1
            assert len(store.similarity_search("manual", k=5)) == 0

        store.destroy()

    # ── 4g. Second GC pass never double-tombstones ────────────────────────

    def test_gc_does_not_double_tombstone(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        with freeze_time(T0):
            store = make_ttl_store(ttl_minutes=1)
            store.add_texts(["once is enough"])

        with freeze_time(T0 + timedelta(seconds=70)):
            first  = store._gc_tick()
            second = store._gc_tick()          # same expired doc, already tombstoned
            assert first == 1
            assert second == 0                 # not counted again

        store.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# 5. Background GC daemon thread
# ═════════════════════════════════════════════════════════════════════════════

class TestGCDaemonThread:
    # ── Lifecycle checks ─────────────────────────────────────────────────

    def test_no_thread_for_immortal_store(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=0)
        assert store._gc_thread   is None
        assert store._stop_event  is None
        store.destroy()

    def test_thread_started_for_ttl_store(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        assert store._gc_thread is not None
        assert store._gc_thread.is_alive()
        store.destroy()

    def test_thread_is_daemon(self, mock_db: type[MockMiniVecDb]) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        assert store._gc_thread.daemon is True   # must not keep process alive
        store.destroy()

    def test_thread_name(self, mock_db: type[MockMiniVecDb]) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        assert store._gc_thread.name == "minivecdb-gc"
        store.destroy()

    def test_destroy_stops_thread(self, mock_db: type[MockMiniVecDb]) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        thread = store._gc_thread
        assert thread.is_alive()

        store.destroy()
        thread.join(timeout=2.0)

        assert not thread.is_alive()

    def test_destroy_is_idempotent(self, mock_db: type[MockMiniVecDb]) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        store.destroy()
        store.destroy()   # second call must not raise or deadlock

    def test_stop_event_set_by_destroy(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        event = store._stop_event
        assert not event.is_set()
        store.destroy()
        assert event.is_set()

    # ── Thread actually fires _gc_tick ────────────────────────────────────

    def test_gc_loop_calls_gc_tick(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """
        The background loop must call _gc_tick() at least once.
        We coordinate with a threading.Event so there are no sleeps.

        Mechanism:
          1. Patch _gc_tick to set a ``tick_happened`` event on first call.
          2. Use gc_interval_sec=0 so Event.wait(0) never blocks.
          3. Wait up to 1 s for the event — the loop fires it immediately.
        """
        tick_happened = threading.Event()
        original_tick = LangChainMiniVecDb._gc_tick

        def spy_tick(self: LangChainMiniVecDb) -> int:
            result = original_tick(self)
            tick_happened.set()
            return result

        with patch.object(LangChainMiniVecDb, "_gc_tick", spy_tick):
            store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=0)
            fired = tick_happened.wait(timeout=1.0)
            store.destroy()

        assert fired, "_gc_tick was not called within 1 second"

    # ── Interrupt wakes up immediately ────────────────────────────────────

    def test_destroy_interrupts_sleeping_thread(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        """
        destroy() sets the stop_event, which unblocks Event.wait() immediately —
        the thread must finish well before the 9 999-second interval elapses.
        """
        store = LangChainMiniVecDb(EMB, ttl_minutes=1, gc_interval_sec=9_999)
        thread = store._gc_thread

        t0 = time.monotonic()
        store.destroy()
        thread.join(timeout=2.0)
        elapsed = time.monotonic() - t0

        assert not thread.is_alive()
        assert elapsed < 2.0, f"thread took {elapsed:.2f}s to stop — stop_event not used"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Context manager
# ═════════════════════════════════════════════════════════════════════════════

class TestContextManager:
    def test_exit_calls_destroy(self, mock_db: type[MockMiniVecDb]) -> None:
        with patch.object(LangChainMiniVecDb, "destroy") as mock_destroy:
            with LangChainMiniVecDb(EMB) as s:
                s.add_texts(["x"])
            mock_destroy.assert_called_once()

    def test_exit_on_exception_calls_destroy(
        self, mock_db: type[MockMiniVecDb]
    ) -> None:
        with patch.object(LangChainMiniVecDb, "destroy") as mock_destroy:
            with pytest.raises(RuntimeError):
                with LangChainMiniVecDb(EMB) as _:
                    raise RuntimeError("boom")
            mock_destroy.assert_called_once()

    def test_returns_self(self, mock_db: type[MockMiniVecDb]) -> None:
        store = LangChainMiniVecDb(EMB)
        assert store.__enter__() is store
        store.destroy()


# ═════════════════════════════════════════════════════════════════════════════
# 7. build_index
# ═════════════════════════════════════════════════════════════════════════════

class TestBuildIndex:
    def test_build_index_does_not_raise(
        self, store: LangChainMiniVecDb
    ) -> None:
        store.add_texts(["a", "b", "c"])
        store.build_index()                # must not raise

    def test_search_after_build_index(self, store: LangChainMiniVecDb) -> None:
        store.add_texts(["a"] * 5)
        store.build_index()
        assert len(store.similarity_search("a", k=3)) == 3


# ═════════════════════════════════════════════════════════════════════════════
# 8. Integration tests — real native extension, real system clock
#
# These tests touch the actual Rust binary compiled by maturin.
# Run with:   pytest tests/ -m integration
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestIntegrationRealNative:
    """Full-stack tests: real MiniVecDb (Rust) + real time."""

    def test_insert_and_search(self) -> None:
        store = LangChainMiniVecDb(EMB)
        store.add_texts(["Rome", "Berlin", "Tokyo"])
        results = store.similarity_search("European capital", k=3)
        assert len(results) == 3
        store.destroy()

    def test_real_gc_expires_documents(self) -> None:
        """
        Insert at T=0 with a 20 ms TTL, sleep 50 ms, call _gc_tick() manually.
        The Rust engine must tombstone the vector (Rust reads SystemTime::now()).
        """
        store = LangChainMiniVecDb(EMB, ttl_minutes=0, gc_interval_sec=9_999)
        store._ttl_ms  = 20.0    # 20 ms
        store._ttl_sec = 0.02

        store.add_texts(["real expire test"])
        time.sleep(0.06)         # 60 ms >> 20 ms TTL

        n = store._gc_tick()
        assert n >= 1, "Rust GC should have tombstoned at least one vector"
        assert store.similarity_search("expire", k=5) == []
        store.destroy()

    def test_real_selective_expiry(self) -> None:
        store = LangChainMiniVecDb(EMB, ttl_minutes=0, gc_interval_sec=9_999)
        store._ttl_ms  = 30.0
        store._ttl_sec = 0.03

        store.add_texts(["doc A — old"])
        time.sleep(0.05)                       # 50 ms: doc A is now expired
        store.add_texts(["doc B — fresh"])
        time.sleep(0.01)                       # 10 ms: doc B is still alive

        n = store._gc_tick()
        assert n >= 1

        results = store.similarity_search("doc", k=10)
        contents = {d.page_content for d in results}
        assert "doc B — fresh" in contents
        assert "doc A — old"   not in contents
        store.destroy()

    def test_real_build_index_and_search(self) -> None:
        store = LangChainMiniVecDb(EMB)
        store.add_texts([f"document {i}" for i in range(10)])
        store.build_index()
        assert len(store.similarity_search("document", k=5)) == 5
        store.destroy()

    def test_real_serialize_deserialize(self) -> None:
        from minivecdb import MiniVecDb

        db = MiniVecDb()
        db.insert(7, [0.3] * 384)
        blob = db.serialize()
        assert isinstance(blob, bytes)

        db2 = MiniVecDb.deserialize(blob)
        hits = db2.search_scan([0.3] * 384, k=1)
        assert hits[0][0] == 7
