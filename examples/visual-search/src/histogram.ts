/**
 * Extracts a 384-dimensional color histogram from an image.
 * Layout: 128 bins for R (indices 0–127), 128 for G (128–255), 128 for B (256–383).
 * Each bin is normalized by total pixel count so values are in [0, 1].
 *
 * Works with MicroVecDB's 1-bit quantization: bins > 0 (colors present in the
 * image) become bit 1. Images with similar color distributions share activated
 * bins → low Hamming distance → found as similar.
 */
export function extractHistogram(img: HTMLImageElement | ImageBitmap): Float32Array {
  const size = 64; // resize for uniformity
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size); // RGBA

  const hist = new Float32Array(384); // 128 bin × 3 channels
  for (let i = 0; i < data.length; i += 4) {
    hist[Math.floor(data[i]     / 2)]       += 1; // R: bin [0–127]
    hist[Math.floor(data[i + 1] / 2) + 128] += 1; // G: bin [128–255]
    hist[Math.floor(data[i + 2] / 2) + 256] += 1; // B: bin [256–383]
    // data[i+3] = alpha, ignored
  }
  const totalPixels = size * size;
  for (let i = 0; i < 384; i++) hist[i] /= totalPixels; // normalize [0, 1]
  return hist;
}

/**
 * Load a File as an ImageBitmap using the browser's built-in decoder.
 * Returns a promise that resolves to the decoded bitmap.
 */
export async function imageFromFile(file: File): Promise<ImageBitmap> {
  return createImageBitmap(file);
}
