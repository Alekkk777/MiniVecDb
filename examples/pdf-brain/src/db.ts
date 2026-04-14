import { MicroVecDB } from '@microvecdb/core';
import type { TextChunk } from './pdf-extractor.js';

export interface SearchHit {
  chunk: TextChunk;
  score: number;
}

/**
 * Build a MicroVecDB from an array of text chunks.
 *
 * @param chunks           Extracted text chunks.
 * @param embedFn          Async function that returns a 384-dim Float32Array for a text.
 * @param progressCallback Called after each chunk is embedded with (current, total).
 * @returns                An initialised, indexed MicroVecDB and the original chunks.
 */
export async function buildDatabase(
  chunks: TextChunk[],
  embedFn: (text: string) => Promise<Float32Array>,
  progressCallback?: (current: number, total: number) => void,
): Promise<{ db: MicroVecDB; chunks: TextChunk[] }> {
  const db = await MicroVecDB.init({
    capacity: Math.max(chunks.length * 2, 64),
    // No OPFS persistence — ephemeral session only
    persistenceKey: null,
  });

  const total = chunks.length;

  for (let i = 0; i < total; i++) {
    const chunk = chunks[i]!;
    const vector = await embedFn(chunk.text);
    db.insert({ id: chunk.id, vector });
    progressCallback?.(i + 1, total);
  }

  db.buildIndex();

  return { db, chunks };
}

/**
 * Search the database for the `k` chunks most similar to `queryVec`.
 *
 * @param db       An indexed MicroVecDB instance.
 * @param queryVec 384-dim query embedding.
 * @param chunks   The original TextChunk array (used to resolve IDs → text).
 * @param k        Number of results to return (default 5).
 * @returns        Array of { chunk, score } sorted by descending score.
 */
const MIN_SCORE = 0.70; // below this the query is likely off-topic

export async function queryDatabase(
  db: MicroVecDB,
  queryVec: Float32Array,
  chunks: TextChunk[],
  k = 5,
): Promise<SearchHit[]> {
  const results = db.search(queryVec, { limit: k });

  // Build a fast id→chunk lookup
  const chunkById = new Map<number, TextChunk>(
    chunks.map((c) => [c.id, c]),
  );

  return results
    .map((r) => {
      const chunk = chunkById.get(r.id);
      if (chunk === undefined) return null;
      return { chunk, score: r.score };
    })
    .filter((h): h is SearchHit => h !== null && h.score >= MIN_SCORE);
}
