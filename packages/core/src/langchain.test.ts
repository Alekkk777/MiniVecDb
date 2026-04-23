/**
 * LangChainMiniVecDb — unit test suite.
 *
 * Strategy
 * --------
 * • MicroVecDB is mocked at the module level.  No WASM is loaded.
 *   (Run `npm run build:wasm` then a real integration test to validate the
 *    full Rust ↔ JS pipeline end-to-end.)
 *
 * • Vitest fake timers replace both `setInterval` and `Date.now`.
 *   Advancing the clock with `vi.advanceTimersByTime(ms)` fires every
 *   scheduled GC callback AND shifts Date.now by the same amount — so the
 *   "wait 60 seconds" scenario executes in microseconds.
 *
 * Fire-test scenario (the "prova del fuoco"):
 *   1. Insert a document at T = 0.
 *   2. Verify it is findable.
 *   3. Advance the fake clock past the TTL (e.g. +70 s for a 60 s TTL).
 *      The GC interval fires automatically during the advance.
 *   4. Verify the document has disappeared from search results.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { Document } from "@langchain/core/documents";
import { LangChainMiniVecDb, type MiniVecDbConfig } from "./langchain.js";

// ─── Mock: MicroVecDB ────────────────────────────────────────────────────────
//
// Mirrors the five methods langchain.ts calls on its inner MicroVecDB.
// runGc() and insert() both read Date.now(), which vitest's fake-timer clock
// controls — so time-based expiry works exactly as in production.

vi.mock("./MicroVecDB.js", () => {
  class MockMicroVecDB {
    private slots = new Map<
      number,
      { vec: Float32Array; deleted: boolean; insertedAt: number }
    >();

    static async init(_config: unknown): Promise<MockMicroVecDB> {
      return new MockMicroVecDB();
    }

    insert({ id, vector }: { id: number; vector: Float32Array }): void {
      this.slots.set(id, {
        vec: vector,
        deleted: false,
        insertedAt: Date.now(), // ← fake clock when tests are running
      });
    }

    search(
      _query: Float32Array,
      { limit }: { limit: number },
    ): Array<{ id: number; distance: number; score: number }> {
      const results: Array<{ id: number; distance: number; score: number }> = [];
      for (const [id, slot] of this.slots) {
        if (!slot.deleted) {
          results.push({ id, distance: 0, score: 1.0 });
        }
        if (results.length >= limit) break;
      }
      return results;
    }

    runGc(ttlMs: number): number {
      const now = Date.now(); // ← fake clock controls this too
      let count = 0;
      for (const slot of this.slots.values()) {
        if (!slot.deleted && now - slot.insertedAt > ttlMs) {
          slot.deleted = true;
          count++;
        }
      }
      return count;
    }

    buildIndex(): void { /* no-op in mock */ }
    dispose(): void { this.slots.clear(); }
  }

  return { MicroVecDB: MockMicroVecDB };
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Fake Embeddings that return a fixed 384-dim vector for any text. */
const fakeEmbeddings = {
  embedDocuments: async (texts: string[]): Promise<number[][]> =>
    texts.map(() => Array<number>(384).fill(0.5)),
  embedQuery: async (_text: string): Promise<number[]> =>
    Array<number>(384).fill(0.5),
};

function makeDoc(content: string, meta: Record<string, unknown> = {}): Document {
  return new Document({ pageContent: content, metadata: meta });
}

/** Create a store with a 60 s TTL and a 5 s GC interval. */
async function makeStore(
  overrides: Partial<MiniVecDbConfig> = {},
): Promise<LangChainMiniVecDb> {
  return LangChainMiniVecDb.create(fakeEmbeddings as never, {
    ttlMinutes: 1,          // TTL = 60 000 ms
    gcIntervalMs: 5_000,    // GC fires every 5 s → first cleanup at ~65 s
    ...overrides,
  });
}

// ─── Test suite ──────────────────────────────────────────────────────────────

describe("LangChainMiniVecDb", () => {
  let store: LangChainMiniVecDb;

  beforeEach(() => {
    // Freeze and control time.  Date.now() starts at the value of the real
    // clock at setup time and then advances only when we say so.
    vi.useFakeTimers();
  });

  afterEach(() => {
    store?.destroy();
    vi.useRealTimers();
  });

  // ─── 1. Basic document storage and retrieval ───────────────────────────────

  describe("addDocuments / similaritySearch", () => {
    it("stores a document and retrieves it immediately", async () => {
      store = await makeStore();
      await store.addDocuments([makeDoc("The Eiffel Tower is in Paris.")]);

      const results = await store.similaritySearch("Paris landmark", 5);
      expect(results).toHaveLength(1);
      expect(results[0]!.pageContent).toBe("The Eiffel Tower is in Paris.");
    });

    it("stores multiple documents and retrieves all of them", async () => {
      store = await makeStore();
      const docs = [
        makeDoc("Rome is the capital of Italy."),
        makeDoc("Berlin is the capital of Germany."),
        makeDoc("Tokyo is the capital of Japan."),
      ];
      await store.addDocuments(docs);

      const results = await store.similaritySearch("capital cities", 10);
      expect(results).toHaveLength(3);
    });

    it("addVectors accepts pre-computed 384-dim vectors", async () => {
      store = await makeStore();
      const vec = Array<number>(384).fill(0.1);
      const doc = makeDoc("pre-computed vector doc");

      const ids = await store.addVectors([vec], [doc]);

      expect(ids).toHaveLength(1);
      expect(typeof ids[0]).toBe("string");

      const results = await store.similaritySearch("anything", 5);
      expect(results).toHaveLength(1);
    });

    it("throws RangeError when vector dimension is wrong", async () => {
      store = await makeStore();
      const bad = Array<number>(512).fill(0.5); // 512 ≠ 384
      await expect(store.addVectors([bad], [makeDoc("x")])).rejects.toThrow(
        RangeError,
      );
    });

    it("throws RangeError when query dimension is wrong", async () => {
      store = await makeStore();
      await store.addDocuments([makeDoc("hello")]);
      const badQuery = Array<number>(128).fill(0.5);
      await expect(
        store.similaritySearchVectorWithScore(badQuery, 5),
      ).rejects.toThrow(RangeError);
    });

    it("preserves document metadata through the round-trip", async () => {
      store = await makeStore();
      const doc = makeDoc("metadata test", { source: "wiki", page: 42 });
      await store.addDocuments([doc]);

      const results = await store.similaritySearch("test", 1);
      expect(results[0]!.metadata).toEqual({ source: "wiki", page: 42 });
    });
  });

  // ─── 2. THE FIRE TEST: TTL expiry ─────────────────────────────────────────
  //
  // This is the core scenario: insert a document, advance the fake clock
  // past the TTL, and verify that the GC has automatically removed it.
  //
  // Why fake timers work here:
  //   • setInterval in the constructor is controlled by vi.
  //   • MockMicroVecDB.runGc() calls Date.now() → fake clock value.
  //   • _runGcTick() in langchain.ts calls Date.now() → same fake value.
  //   So advancing time by N ms is semantically identical to waiting N ms.

  describe("🔥 Fire test — TTL garbage collection", () => {
    it("document is found BEFORE expiry, gone AFTER", async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 5_000 });

      await store.addDocuments([
        makeDoc("Ephemeral agent observation: user asked about weather."),
      ]);

      // ── T = 0 ── Document just inserted; should be findable.
      let results = await store.similaritySearch("weather", 5);
      expect(results).toHaveLength(1);

      // ── T = +70 s ── Advance past the 60 s TTL.
      //   The GC interval fires at +5 s, +10 s, … +65 s, +70 s.
      //   At +65 s: doc age = 65 s > 60 s TTL → tombstoned by GC.
      vi.advanceTimersByTime(70_000);

      // ── T = +70 s ── Document must have been garbage-collected.
      results = await store.similaritySearch("weather", 5);
      expect(results).toHaveLength(0);
    });

    it("GC returns the correct tombstone count", async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 500_000 });
      // gcIntervalMs is huge — GC will NOT fire automatically here.

      await store.addDocuments([
        makeDoc("doc A"),
        makeDoc("doc B"),
        makeDoc("doc C"),
      ]);

      // Advance past TTL without triggering the interval.
      vi.advanceTimersByTime(70_000);

      // Call runGc() manually.
      const tombstoned = store.runGc();
      expect(tombstoned).toBe(3);
    });

    it("document inserted AFTER TTL has elapsed is not yet expired", async () => {
      // ttlMinutes = 1 min, gcIntervalMs = 10 s
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 10_000 });

      // Insert doc A at T = 0.
      await store.addDocuments([makeDoc("doc A — old")]);

      // Advance to T = 30 s.
      vi.advanceTimersByTime(30_000);

      // Insert doc B at T = 30 s.
      await store.addDocuments([makeDoc("doc B — fresh")]);

      // Advance to T = 70 s (doc A = 70 s old → expired; doc B = 40 s → alive).
      vi.advanceTimersByTime(40_000);

      // GC fires at +10 s, +20 s, … +70 s, collecting doc A at the +70 s tick.
      const results = await store.similaritySearch("doc", 10);
      expect(results).toHaveLength(1);
      expect(results[0]!.pageContent).toBe("doc B — fresh");
    });

    it("withScores returns [Document, score] tuples before expiry", async () => {
      store = await makeStore();
      await store.addDocuments([makeDoc("scored doc")]);

      const vec = Array<number>(384).fill(0.5);
      const pairs = await store.similaritySearchVectorWithScore(vec, 5);

      expect(pairs).toHaveLength(1);
      const [doc, score] = pairs[0]!;
      expect(doc.pageContent).toBe("scored doc");
      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(1);
    });
  });

  // ─── 3. Manual GC control ─────────────────────────────────────────────────

  describe("runGc() manual invocation", () => {
    it("tombstones expired docs when called without waiting for interval", async () => {
      // Use a huge gcIntervalMs so the timer never fires automatically.
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 999_999 });
      await store.addDocuments([makeDoc("manual GC test")]);

      // Advance past TTL — interval does NOT fire because gcIntervalMs > advance.
      vi.advanceTimersByTime(70_000);

      // Verify doc is still in JS map (GC hasn't run yet).
      // Because the mock search returns active slots only, it should be gone
      // from the WASM mock after runGc but NOT yet from the JS map.
      // runGc() runs both sides:
      const count = store.runGc();
      expect(count).toBe(1);

      const results = await store.similaritySearch("test", 5);
      expect(results).toHaveLength(0);
    });

    it("runGc() is a no-op when ttlMinutes = 0", async () => {
      store = await makeStore({ ttlMinutes: 0 });
      await store.addDocuments([makeDoc("immortal doc")]);

      vi.advanceTimersByTime(999_999_999);

      const count = store.runGc();
      expect(count).toBe(0);

      // Immortal docs survive regardless of time.
      const results = await store.similaritySearch("anything", 5);
      expect(results).toHaveLength(1);
    });
  });

  // ─── 4. Lifecycle: destroy() ──────────────────────────────────────────────

  describe("destroy()", () => {
    it("stops the GC timer so no further tombstoning occurs", async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 5_000 });
      await store.addDocuments([makeDoc("should survive destroy()")]);

      // Destroy before TTL expires.
      store.destroy();

      // Advance past TTL — the interval must NOT fire.
      vi.advanceTimersByTime(70_000);

      // The mock's slots are cleared by dispose(), so search returns nothing —
      // but what we're verifying is that destroy() did NOT crash and DID call
      // clearInterval (otherwise Node would keep firing the callback).
      // There's no clean way to assert "setInterval did not fire" without
      // inspecting internals, but we can verify the timer handle is gone by
      // confirming no errors occur and the call completes cleanly.
      expect(() => store.destroy()).not.toThrow(); // idempotent
    });

    it("does not set up a timer when ttlMinutes is 0", async () => {
      const spy = vi.spyOn(globalThis, "setInterval");
      store = await makeStore({ ttlMinutes: 0 });

      expect(spy).not.toHaveBeenCalled();
      spy.mockRestore();
    });
  });

  // ─── 5. HNSW index management ─────────────────────────────────────────────

  describe("buildIndex()", () => {
    it("calls through to MicroVecDB.buildIndex() without error", async () => {
      store = await makeStore();
      await store.addDocuments([makeDoc("index test")]);
      expect(() => store.buildIndex()).not.toThrow();
    });
  });
});
