import * as pdfjsLib from 'pdfjs-dist';
// Vite ?url suffix: gives us the correct served URL for the worker file,
// resolving through node_modules rather than relative to this module's path.
import pdfjsWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorkerUrl;

export interface TextChunk {
  id: number;
  text: string;
  page: number;
}

interface PageText {
  text: string;
  page: number;
}

/**
 * Load a PDF from an ArrayBuffer and extract per-page text.
 */
async function extractPageTexts(buffer: ArrayBuffer): Promise<PageText[]> {
  const loadingTask = pdfjsLib.getDocument({ data: buffer });
  const pdf = await loadingTask.promise;
  const results: PageText[] = [];

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();
    const pageText = content.items
      .map((item) => {
        // TextItem has `str`; TextMarkedContent does not — guard accordingly
        if ('str' in item) return (item as { str: string }).str;
        return '';
      })
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (pageText.length > 0) {
      results.push({ text: pageText, page: pageNum });
    }
  }

  return results;
}

/**
 * Split a flat array of words (with source-page bookkeeping) into overlapping
 * chunks of approximately `wordsPerChunk` words.
 */
function chunkWords(
  words: { word: string; page: number }[],
  wordsPerChunk: number,
  overlap: number,
): TextChunk[] {
  const chunks: TextChunk[] = [];
  let id = 0;
  let start = 0;

  while (start < words.length) {
    const end = Math.min(start + wordsPerChunk, words.length);
    const slice = words.slice(start, end);
    const text = slice.map((w) => w.word).join(' ');
    // Page attribution: use the page of the first word in the chunk
    const page = slice[0]?.page ?? 1;
    chunks.push({ id, text, page });
    id++;
    // Advance by (wordsPerChunk - overlap); ensure progress
    const step = Math.max(1, wordsPerChunk - overlap);
    start += step;
  }

  return chunks;
}

/**
 * Extract text from a PDF ArrayBuffer and return it split into overlapping
 * chunks of approximately `wordsPerChunk` words.
 *
 * @param buffer       Raw PDF bytes.
 * @param wordsPerChunk  Target chunk size in words (default 200).
 * @param overlap        Word overlap between consecutive chunks (default 20).
 */
export async function extractChunks(
  buffer: ArrayBuffer,
  wordsPerChunk = 200,
  overlap = 20,
): Promise<TextChunk[]> {
  const pageTexts = await extractPageTexts(buffer);

  // Flatten all words while keeping their source page
  const allWords: { word: string; page: number }[] = [];
  for (const { text, page } of pageTexts) {
    const words = text.split(/\s+/).filter((w) => w.length > 0);
    for (const word of words) {
      allWords.push({ word, page });
    }
  }

  if (allWords.length === 0) return [];

  return chunkWords(allWords, wordsPerChunk, overlap);
}
