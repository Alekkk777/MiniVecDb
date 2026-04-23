/**
 * VercelMiniVecDb — unit test suite.
 *
 * Strategy
 * --------
 * • Both the `ai` SDK and `MicroVecDB` are mocked at the module level.
 *   No WASM is loaded; no real embeddings are called.
 *
 * • Vitest fake timers replace `setInterval`, `clearInterval`, and
 *   `Date.now`.  Advancing the clock with `vi.advanceTimersByTime(ms)`
 *   fires every scheduled GC callback AND shifts Date.now by the same
 *   amount — so "wait 70 seconds" executes in microseconds.
 *
 * Fire-test scenario (the "prova del fuoco"):
 *   1. Insert a text at T = 0.
 *   2. Verify it is retrievable.
 *   3. Advance the fake clock past the TTL (+70 s for a 60 s TTL).
 *      The GC interval fires automatically during the advance.
 *   4. Verify the text has disappeared from retrieve results.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { VercelMiniVecDb, type VercelMiniVecDbConfig } from './vercel.js';

// ─── Mock: ai SDK ────────────────────────────────────────────────────────────
//
// embedMany / embed return a fixed 384-dim [0.5, …] vector.
// tool and zodSchema are identity pass-throughs — we test our adapter
// logic, not the Vercel SDK internals.

vi.mock('ai', () => ({
  embedMany: vi.fn(async ({ values }: { values: string[] }) => ({
    embeddings: values.map(() => Array<number>(384).fill(0.5)),
  })),
  embed: vi.fn(async ({ value: _v }: { value: string }) => ({
    embedding: Array<number>(384).fill(0.5),
  })),
  // `tool()` is a type-level identity function in the real SDK.
  tool: vi.fn(<T>(t: T) => t),
  // `zodSchema()` wraps a Zod schema; for testing, pass it through.
  zodSchema: vi.fn(<T>(s: T) => s),
}));

// ─── Mock: MicroVecDB ────────────────────────────────────────────────────────
//
// Mirrors the six methods VercelMiniVecDb calls on its inner MicroVecDB.
// runGc() reads Date.now() — vitest's fake-timer clock controls it — so
// time-based expiry works exactly as in production.

vi.mock('./MicroVecDB.js', () => {
  class MockMicroVecDB {
    private slots = new Map<
      number,
      { vec: Float32Array; deleted: boolean; insertedAt: number }
    >();

    static async init(_cfg: unknown): Promise<MockMicroVecDB> {
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
        if (!slot.deleted) results.push({ id, distance: 0, score: 1.0 });
        if (results.length >= limit) break;
      }
      return results;
    }

    runGc(ttlMs: number): number {
      const now = Date.now(); // ← same fake clock
      let count = 0;
      for (const slot of this.slots.values()) {
        if (!slot.deleted && now - slot.insertedAt > ttlMs) {
          slot.deleted = true;
          count++;
        }
      }
      return count;
    }

    buildIndex(): void { /* no-op */ }
    dispose(): void { this.slots.clear(); }
    has_index(): boolean { return false; }
  }

  return { MicroVecDB: MockMicroVecDB };
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Fake EmbeddingModel accepted by `embed` / `embedMany` mocks above. */
const fakeModel = 'fake-model-384' as unknown as import('ai').EmbeddingModel;

/** Store with a 60 s TTL and a 5 s GC interval. */
async function makeStore(
  overrides: Partial<VercelMiniVecDbConfig> = {},
): Promise<VercelMiniVecDb> {
  return VercelMiniVecDb.create(fakeModel, {
    ttlMinutes:   1,       // TTL = 60 000 ms
    gcIntervalMs: 5_000,   // GC fires every 5 s → first expiry at ~65 s
    ...overrides,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Test suite
// ─────────────────────────────────────────────────────────────────────────────

describe('VercelMiniVecDb', () => {
  let store: VercelMiniVecDb;

  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    store?.destroy();
    vi.useRealTimers();
  });

  // ─── 1. add / retrieve ────────────────────────────────────────────────────

  describe('add / retrieve', () => {
    it('stores a text and retrieves it immediately', async () => {
      store = await makeStore();
      await store.add(['The Eiffel Tower is in Paris.']);

      const results = await store.retrieve('Paris landmark', 5);
      expect(results).toHaveLength(1);
      expect(results[0]!.text).toBe('The Eiffel Tower is in Paris.');
      expect(results[0]!.score).toBeGreaterThan(0);
    });

    it('stores multiple texts and retrieves all of them', async () => {
      store = await makeStore();
      await store.add(['Rome', 'Berlin', 'Tokyo']);

      const results = await store.retrieve('capital cities', 10);
      expect(results).toHaveLength(3);
    });

    it('returns IDs in insertion order', async () => {
      store = await makeStore();
      const ids = await store.add(['a', 'b', 'c']);
      expect(ids).toEqual([0, 1, 2]);
    });

    it('returns empty array for empty input', async () => {
      store = await makeStore();
      expect(await store.add([])).toEqual([]);
    });

    it('preserves metadata through the round-trip', async () => {
      store = await makeStore();
      await store.add(['tagged observation'], [{ source: 'agent', turn: 3 }]);

      const results = await store.retrieve('anything', 1);
      expect(results[0]!.metadata).toEqual({ source: 'agent', turn: 3 });
    });

    it('throws RangeError when the model returns wrong-dimension embeddings', async () => {
      const { embedMany } = await import('ai');
      vi.mocked(embedMany).mockResolvedValueOnce({
        embeddings: [Array<number>(512).fill(0.5)],
        values: ['x'],
        usage: { tokens: 1 },
        warnings: undefined as never,
        responses: undefined as never,
        providerMetadata: undefined as never,
        experimental_providerMetadata: undefined as never,
      });

      store = await makeStore();
      await expect(store.add(['x'])).rejects.toThrow(RangeError);
    });

    it('throws RangeError when the query embedding has wrong dimensions', async () => {
      const { embed } = await import('ai');
      vi.mocked(embed).mockResolvedValueOnce({
        embedding: Array<number>(128).fill(0.5),
        value: 'q',
        usage: { tokens: 1 },
        warnings: undefined as never,
        responses: undefined as never,
        providerMetadata: undefined as never,
        experimental_providerMetadata: undefined as never,
      });

      store = await makeStore();
      await store.add(['hello']);
      await expect(store.retrieve('oops', 1)).rejects.toThrow(RangeError);
    });
  });

  // ─── 2. buildIndex ────────────────────────────────────────────────────────

  describe('buildIndex()', () => {
    it('calls through to MicroVecDB.buildIndex() without error', async () => {
      store = await makeStore();
      await store.add(['index test']);
      expect(() => store.buildIndex()).not.toThrow();
    });
  });

  // ─── 3. createRetrievalTool ───────────────────────────────────────────────

  describe('createRetrievalTool()', () => {
    it('returns an object with description, inputSchema, and execute', async () => {
      store = await makeStore();
      const t = store.createRetrievalTool();

      expect(t).toHaveProperty('description');
      expect(t).toHaveProperty('inputSchema');
      expect(t).toHaveProperty('execute');
      expect(typeof t.execute).toBe('function');
    });

    it('execute() returns formatted results when docs are present', async () => {
      store = await makeStore();
      await store.add(['Paris is in France.', 'Berlin is in Germany.']);

      const t = store.createRetrievalTool();
      const output = await t.execute({ query: 'European capitals' }, {} as never);

      expect(typeof output).toBe('string');
      expect(output).toContain('[1]');
      expect(output).toContain('score=');
    });

    it('execute() returns "no context" message when store is empty', async () => {
      store = await makeStore();
      const t = store.createRetrievalTool();
      const output = await t.execute({ query: 'anything' }, {} as never);
      expect(output).toContain('No relevant context');
    });

    it('respects a custom description', async () => {
      store = await makeStore();
      const t = store.createRetrievalTool('Search session notes');
      expect((t as { description: string }).description).toBe('Search session notes');
    });

    it('execute() reflects TTL-expired docs being gone', async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 5_000 });
      await store.add(['ephemeral observation']);

      // Advance 70 s — GC fires at +5 s, +10 s, … +65 s and tombstones the doc.
      vi.advanceTimersByTime(70_000);

      const t = store.createRetrievalTool();
      const output = await t.execute({ query: 'ephemeral' }, {} as never);
      expect(output).toContain('No relevant context');
    });
  });

  // ─── 4. 🔥 Fire test — TTL garbage collection ────────────────────────────

  describe('🔥 Fire test — TTL garbage collection', () => {
    it('text found BEFORE expiry, gone AFTER', async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 5_000 });
      await store.add(['Ephemeral agent observation: user asked about weather.']);

      // ── T = 0: just inserted ──────────────────────────────────────────────
      let results = await store.retrieve('weather', 5);
      expect(results).toHaveLength(1);

      // ── T = +70 s: past the 60 s TTL ─────────────────────────────────────
      // GC fires at +5, +10, … +65 s; at +65 s the doc age (65 s) > TTL (60 s).
      vi.advanceTimersByTime(70_000);

      results = await store.retrieve('weather', 5);
      expect(results).toHaveLength(0);
    });

    it('GC returns the correct tombstone count', async () => {
      // Use a huge interval so the timer never fires automatically.
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 500_000 });
      await store.add(['doc A', 'doc B', 'doc C']);

      // Advance past TTL without triggering the interval.
      vi.advanceTimersByTime(70_000);

      // Call runGc() manually.
      const tombstoned = store.runGc();
      expect(tombstoned).toBe(3);
    });

    it('text inserted AFTER TTL has elapsed is not yet expired', async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 10_000 });

      // T = 0: insert doc A.
      await store.add(['doc A — old']);

      // T = +30 s: insert doc B.
      vi.advanceTimersByTime(30_000);
      await store.add(['doc B — fresh']);

      // T = +70 s: doc A (age 70 s) > TTL; doc B (age 40 s) < TTL.
      vi.advanceTimersByTime(40_000);

      const results = await store.retrieve('doc', 10);
      expect(results).toHaveLength(1);
      expect(results[0]!.text).toBe('doc B — fresh');
    });

    it('runGc() is a no-op when ttlMinutes is 0', async () => {
      store = await VercelMiniVecDb.create(fakeModel, { ttlMinutes: 0 });
      await store.add(['immortal text']);

      vi.advanceTimersByTime(999_999_999);

      expect(store.runGc()).toBe(0);
      const results = await store.retrieve('immortal', 5);
      expect(results).toHaveLength(1);
    });
  });

  // ─── 5. Manual GC control ─────────────────────────────────────────────────

  describe('runGc() manual invocation', () => {
    it('tombstones expired texts when called without the interval firing', async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 999_999 });
      await store.add(['manual GC test']);

      vi.advanceTimersByTime(70_000);

      const count = store.runGc();
      expect(count).toBe(1);

      const results = await store.retrieve('test', 5);
      expect(results).toHaveLength(0);
    });
  });

  // ─── 6. Lifecycle: destroy() ─────────────────────────────────────────────

  describe('destroy()', () => {
    it('stops the GC timer so no further tombstoning occurs after destroy', async () => {
      store = await makeStore({ ttlMinutes: 1, gcIntervalMs: 5_000 });
      await store.add(['should survive destroy()']);

      store.destroy();

      // Advance past TTL — the interval must NOT fire after destroy.
      vi.advanceTimersByTime(70_000);

      // Idempotency: second destroy must not throw.
      expect(() => store.destroy()).not.toThrow();
    });

    it('does not set up a timer when ttlMinutes is 0', async () => {
      const spy = vi.spyOn(globalThis, 'setInterval');
      store = await VercelMiniVecDb.create(fakeModel, { ttlMinutes: 0 });

      expect(spy).not.toHaveBeenCalled();
      spy.mockRestore();
    });
  });
});
