import { MicroVecDB } from '@microvecdb/core';

export interface ImageEntry {
  id: number;
  filename: string;
  objectUrl: string; // for thumbnail display
}

/**
 * Build a MicroVecDB instance from a list of image files.
 *
 * @param entries    Files paired with their numeric IDs.
 * @param histFn     Function that extracts a 384-dim histogram from a bitmap.
 * @param onProgress Called after each image is processed.
 * @returns          Initialized DB (with HNSW index built) and metadata entries.
 */
export async function buildDatabase(
  entries: { file: File; id: number }[],
  histFn: (img: ImageBitmap) => Float32Array,
  onProgress: (current: number, total: number) => void,
): Promise<{ db: MicroVecDB; entries: ImageEntry[] }> {
  const db = await MicroVecDB.init({ capacity: Math.max(entries.length, 64) });
  const imageEntries: ImageEntry[] = [];

  for (let i = 0; i < entries.length; i++) {
    const { file, id } = entries[i];
    const bitmap = await createImageBitmap(file);
    const vector = histFn(bitmap);
    bitmap.close();

    db.insert({ id, vector });

    imageEntries.push({
      id,
      filename: file.name,
      objectUrl: URL.createObjectURL(file),
    });

    onProgress(i + 1, entries.length);
  }

  db.buildIndex();
  return { db, entries: imageEntries };
}

/**
 * Find the K most similar images to a given histogram vector.
 *
 * @param db        A MicroVecDB instance with a built HNSW index.
 * @param queryVec  384-dimensional query histogram.
 * @param k         Number of results to return (default 5).
 * @returns         Array of { id, score } sorted by descending similarity.
 */
export async function findSimilar(
  db: MicroVecDB,
  queryVec: Float32Array,
  k = 5,
): Promise<Array<{ id: number; score: number }>> {
  const results = db.search(queryVec, { limit: k, ef: k * 4 });
  return results.map((r) => ({ id: r.id, score: r.score }));
}
