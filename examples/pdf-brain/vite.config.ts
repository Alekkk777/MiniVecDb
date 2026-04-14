import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  optimizeDeps: {
    exclude: ['@xenova/transformers', '@microvecdb/core'],
  },
  resolve: {
    alias: {
      // Self-contained wasm-only build — avoids the registerBackend crash in Safari
      // under any strict security headers (no external onnxruntime-common dep).
      'onnxruntime-web': path.resolve(
        __dirname,
        '../../node_modules/onnxruntime-web/dist/ort.wasm.min.js',
      ),
    },
  },
  assetsInclude: ['**/*.wasm'],
  server: {
    fs: {
      allow: ['../..'],
    },
    // No COEP: Hugging Face CDN doesn't send Cross-Origin-Resource-Policy,
    // so require-corp blocks model downloads. ONNX runs single-threaded
    // (numThreads=1), MicroVecDB WASM is also single-threaded — no SAB needed.
  },
});
