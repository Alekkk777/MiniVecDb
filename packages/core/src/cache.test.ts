import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { SemanticCache, withSemanticCache, type SemanticCacheOptions } from './cache.js';

// ─── Mock: ai SDK ────────────────────────────────────────────────────────────
// embed() returns a deterministic 384-dim unit vector based on the input string.
vi.mock('ai', () => ({
  embed: vi.fn(async ({ value }: { value: string }) => {
    let s = 0;
    for (let i = 0; i < value.length; i++) s = (s * 31 + value.charCodeAt(i)) >>> 0;
    const v: number[] = [];
    for (let i = 0; i < 384; i++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      v.push((s / 0x100000000) * 2 - 1);
    }
    const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0));
    return { embedding: v.map(x => x / norm) };
  }),
}));

// ─── Mock: MicroVecDB ────────────────────────────────────────────────────────
vi.mock('./MicroVecDB.js', () => {
  class MockMicroVecDB {
    private slots = new Map<number, { vec: Float32Array; deleted: boolean; insertedAt: number }>();

    static async init(_opts: unknown): Promise<MockMicroVecDB> { return new MockMicroVecDB(); }

    insert({ id, vector }: { id: number; vector: Float32Array }): void {
      this.slots.set(id, { vec: vector, deleted: false, insertedAt: Date.now() });
    }

    search(query: Float32Array, { limit = 1 }: { limit?: number } = {}): Array<{ id: number; distance: number; score: number }> {
      const results: Array<{ id: number; distance: number; score: number }> = [];
      for (const [id, slot] of this.slots) {
        if (slot.deleted) continue;
        // Dot product as similarity (both vectors are unit, so cosine = dot)
        let dot = 0;
        for (let i = 0; i < query.length; i++) dot += query[i] * slot.vec[i];
        const score = (dot + 1) / 2; // normalise [-1,1] → [0,1]
        results.push({ id, distance: 1 - score, score });
      }
      results.sort((a, b) => b.score - a.score);
      return results.slice(0, limit);
    }

    runGc(ttlMs: number): number {
      const cutoff = Date.now() - ttlMs;
      let count = 0;
      for (const slot of this.slots.values()) {
        if (!slot.deleted && slot.insertedAt < cutoff) { slot.deleted = true; count++; }
      }
      return count;
    }

    dispose(): void { this.slots.clear(); }
  }
  return { MicroVecDB: MockMicroVecDB };
});

// ─── Helpers ─────────────────────────────────────────────────────────────────

const fakeModel = 'fake-model-384' as unknown as import('ai').EmbeddingModel;

async function makeCache(overrides: Partial<SemanticCacheOptions> = {}): Promise<SemanticCache> {
  return SemanticCache.create({
    embeddingModel: fakeModel,
    threshold: 0.9,
    ...overrides,
  });
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('SemanticCache', () => {
  let cache: SemanticCache;

  beforeEach(() => { vi.useFakeTimers(); });
  afterEach(() => { cache?.destroy(); vi.useRealTimers(); });

  // ── 1. Basic lookup/set ───────────────────────────────────────────────────

  describe('lookup / set', () => {
    it('returns null on empty cache', async () => {
      cache = await makeCache();
      expect(await cache.lookup('anything')).toBeNull();
    });

    it('returns the stored response for the exact same prompt', async () => {
      cache = await makeCache();
      await cache.set('What is the return policy?', 'Returns accepted within 30 days.');
      const result = await cache.lookup('What is the return policy?');
      expect(result).toBe('Returns accepted within 30 days.');
    });

    it('returns null when similarity is below threshold', async () => {
      cache = await makeCache({ threshold: 0.9999 }); // extremely strict
      await cache.set('What is the return policy?', 'Returns accepted within 30 days.');
      // A totally different prompt should be well below 0.9999
      const result = await cache.lookup('How do I reset my password?');
      expect(result).toBeNull();
    });

    it('hits on a semantically similar prompt (same text, same vector)', async () => {
      cache = await makeCache({ threshold: 0.5 }); // lenient
      await cache.set('hello world', 'response A');
      // Same prompt → cosine = 1.0 → normalised score = 1.0 → hit
      expect(await cache.lookup('hello world')).toBe('response A');
    });
  });

  // ── 2. wrap HOF ──────────────────────────────────────────────────────────

  describe('wrap()', () => {
    it('calls fallback on first request and caches result', async () => {
      cache = await makeCache();
      const fallback = vi.fn(async (_p: string) => 'LLM answer');
      const fn = cache.wrap(fallback);

      const r1 = await fn('What is the capital of France?');
      expect(r1).toBe('LLM answer');
      expect(fallback).toHaveBeenCalledTimes(1);
    });

    it('does not call fallback on identical second request', async () => {
      cache = await makeCache();
      const fallback = vi.fn(async (_p: string) => 'LLM answer');
      const fn = cache.wrap(fallback);

      await fn('What is the capital of France?');
      const r2 = await fn('What is the capital of France?');

      expect(r2).toBe('LLM answer');
      expect(fallback).toHaveBeenCalledTimes(1); // still only 1
    });
  });

  // ── 3. withSemanticCache HOF ──────────────────────────────────────────────

  describe('withSemanticCache()', () => {
    it('returns a working cached function', async () => {
      let calls = 0;
      const fn = await withSemanticCache(
        { embeddingModel: fakeModel, threshold: 0.9 },
        async (_p) => { calls++; return `response ${calls}`; },
      );

      const r1 = await fn('test prompt');
      const r2 = await fn('test prompt');

      expect(r1).toBe('response 1');
      expect(r2).toBe('response 1'); // cache hit
      expect(calls).toBe(1);
    });
  });

  // ── 4. TTL / GC ──────────────────────────────────────────────────────────

  describe('TTL garbage collection', () => {
    it('evicts entries after TTL when GC fires', async () => {
      cache = await makeCache({ ttlMinutes: 1, gcIntervalMs: 5_000 });
      await cache.set('ephemeral prompt', 'ephemeral answer');

      expect(await cache.lookup('ephemeral prompt')).toBe('ephemeral answer');

      vi.advanceTimersByTime(70_000); // GC fires, TTL = 60s

      expect(await cache.lookup('ephemeral prompt')).toBeNull();
    });

    it('runGc() manually evicts expired entries', async () => {
      cache = await makeCache({ ttlMinutes: 1, gcIntervalMs: 999_999 });
      await cache.set('will expire', 'value');

      vi.advanceTimersByTime(70_000);

      const count = cache.runGc();
      expect(count).toBe(1);
      expect(await cache.lookup('will expire')).toBeNull();
    });

    it('runGc() is a no-op when ttlMinutes is 0', async () => {
      cache = await makeCache({ ttlMinutes: 0 });
      await cache.set('immortal', 'forever');
      vi.advanceTimersByTime(999_999_999);
      expect(cache.runGc()).toBe(0);
      expect(await cache.lookup('immortal')).toBe('forever');
    });
  });

  // ── 5. Lifecycle ─────────────────────────────────────────────────────────

  describe('destroy()', () => {
    it('is idempotent', async () => {
      cache = await makeCache();
      cache.destroy();
      expect(() => cache.destroy()).not.toThrow();
    });

    it('does not start a timer when ttlMinutes is 0', async () => {
      const spy = vi.spyOn(globalThis, 'setInterval');
      cache = await makeCache({ ttlMinutes: 0 });
      expect(spy).not.toHaveBeenCalled();
      spy.mockRestore();
    });
  });
});
