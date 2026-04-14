import { pipeline, env, type FeatureExtractionPipeline } from '@xenova/transformers';

// Do not check /models/ on the local dev server — Vite returns index.html for
// unknown routes which causes "JSON Parse error: Unrecognized token '<'".
env.allowLocalModels = false;

// Bypass the Cache API ('transformers-cache').
// tryCache() checks localPath and remoteURL regardless of allowLocalModels, so
// a previously cached HTML response (from when COEP was blocking HF CDN fetches)
// would still be returned. Disabling the cache forces a fresh remote fetch.
env.useBrowserCache = false;

// Force WASM single-thread mode so ONNX Runtime does not need SharedArrayBuffer.
env.backends.onnx.wasm.numThreads = 1;

const MODEL_NAME = 'Xenova/all-MiniLM-L6-v2';

/**
 * Load the sentence-embedding pipeline (downloads model on first call,
 * cached in IndexedDB by Transformers.js on subsequent calls).
 */
export async function loadEmbedder(): Promise<FeatureExtractionPipeline> {
  const pipe = await pipeline('feature-extraction', MODEL_NAME, {
    quantized: true,
  });
  return pipe as FeatureExtractionPipeline;
}

/**
 * Embed a single text string into a 384-dimensional Float32Array.
 *
 * @param pipe  A loaded FeatureExtractionPipeline (from `loadEmbedder`).
 * @param text  The text to embed.
 * @returns     Float32Array of length 384, mean-pooled and L2-normalised.
 */
export async function embed(
  pipe: FeatureExtractionPipeline,
  text: string,
): Promise<Float32Array> {
  const output = await pipe(text, { pooling: 'mean', normalize: true });

  // Transformers.js returns a Tensor; `.data` is the underlying TypedArray.
  const data: Float32Array =
    output.data instanceof Float32Array
      ? output.data
      : new Float32Array(output.data as ArrayLike<number>);

  if (data.length !== 384) {
    throw new RangeError(
      `Expected 384-dim embedding, got ${data.length}. Check the model name.`,
    );
  }

  return data;
}
