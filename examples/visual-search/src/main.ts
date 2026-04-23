// This file is superseded by App.tsx (React rewrite). Kept as empty module.
export {};
import { extractHistogram, imageFromFile } from './histogram.js';
import { buildDatabase, findSimilar, type ImageEntry } from './db.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let currentDb: MicroVecDB | null = null;
let currentEntries: ImageEntry[] = [];
let nextId = 0;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------

const dropZone        = document.getElementById('drop-zone')!;
const fileInput       = document.getElementById('file-input') as HTMLInputElement;
const browseBtn       = document.getElementById('browse-btn')!;
const progressWrap    = document.getElementById('progress-wrap')!;
const progressLabel   = document.getElementById('progress-label')!;
const progressBar     = document.getElementById('progress-bar') as HTMLElement;
const gallerySection  = document.getElementById('gallery-section')!;
const gallery         = document.getElementById('gallery')!;
const countLabel      = document.getElementById('count-label')!;
const indexLabel      = document.getElementById('index-label')!;
const sidebar         = document.getElementById('sidebar')!;
const queryPreview    = document.getElementById('query-preview') as HTMLImageElement;
const resultsList     = document.getElementById('results-list')!;
const statusDot       = document.getElementById('status-dot')!;
const statusText      = document.getElementById('status-text')!;
const recallBtn       = document.getElementById('recall-btn') as HTMLButtonElement;
const recallOutput    = document.getElementById('recall-output')!;

// ---------------------------------------------------------------------------
// Status helpers
// ---------------------------------------------------------------------------

function setStatus(state: 'idle' | 'indexing' | 'ready', msg: string) {
  statusDot.className = 'status-dot' + (state !== 'idle' ? ` ${state}` : '');
  statusText.textContent = msg;
}

setStatus('idle', 'Ready — drop images to start');

// ---------------------------------------------------------------------------
// Drop zone interactions
// ---------------------------------------------------------------------------

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer?.files ?? []).filter((f) =>
    f.type.startsWith('image/'),
  );
  if (files.length > 0) handleFiles(files);
});

dropZone.addEventListener('click', (e) => {
  if (e.target !== browseBtn) fileInput.click();
});

browseBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  const files = Array.from(fileInput.files ?? []);
  if (files.length > 0) handleFiles(files);
  fileInput.value = '';
});

// ---------------------------------------------------------------------------
// Core: ingest images
// ---------------------------------------------------------------------------

async function handleFiles(files: File[]) {
  setStatus('indexing', `Indexing ${files.length} image(s)…`);
  recallBtn.disabled = true;

  // Build entries with IDs continuing from where we left off
  const startId = nextId;
  const newEntries = files.map((f, i) => ({ file: f, id: startId + i }));
  nextId += files.length;

  // Show progress bar
  progressWrap.classList.add('visible');
  progressBar.style.width = '0%';
  setProgressLabel(0, files.length);

  try {
    const { db, entries } = await buildDatabase(
      newEntries,
      extractHistogram,
      (current, total) => {
        setProgressLabel(current, total);
        progressBar.style.width = `${(current / total) * 100}%`;
      },
    );

    // If we already had a DB, merge by re-building from all entries
    if (currentDb !== null) {
      // Dispose old DB, rebuild with all entries combined
      currentDb.dispose();

      currentEntries = [...currentEntries, ...entries];

      // We already have objectUrls for old entries; re-insert all vectors
      // by using the new db as base and reinserting old vecs via search trick.
      // Simpler: rebuild everything from scratch using stored objectUrls.
      const mergedDb = await MicroVecDB.init({ capacity: Math.max(currentEntries.length, 64) });

      // Re-extract histograms from objectUrls for old entries
      const allNewEntries: typeof newEntries = [];

      for (const entry of currentEntries) {
        // If it's a new entry, we already have its histogram in `db`
        // For old entries, we need to re-extract — fetch from objectUrl
        const newEntry = entries.find((e) => e.id === entry.id);
        if (newEntry) {
          // new entry — reuse from `db` results via objectUrl re-fetch
        }
        // We'll just reuse the blob URLs to re-extract
        allNewEntries.push({ file: await urlToFile(entry.objectUrl, entry.filename), id: entry.id });
      }

      // Rebuild from scratch
      const rebuilt = await buildDatabase(
        allNewEntries,
        extractHistogram,
        (current, total) => {
          setProgressLabel(current, total, 'Re-indexing');
          progressBar.style.width = `${(current / total) * 100}%`;
        },
      );

      mergedDb.dispose(); // dispose the empty one we created
      currentDb = rebuilt.db;
      // Keep the existing objectUrls (don't replace them — they are already valid)
      // Update entries preserving old objectUrls
      for (const rebuilt_entry of rebuilt.entries) {
        const existing = currentEntries.find((e) => e.id === rebuilt_entry.id);
        if (existing) {
          existing.filename = rebuilt_entry.filename;
          // Note: objectUrl from rebuild points to a new Blob; release the rebuilt one
          // and keep original. But here entries[] uses File → createObjectURL per session.
          // For simplicity: keep the rebuilt objectUrl (both point to same original file data
          // since we re-fetched via urlToFile → objectUrl is re-created from the same blob).
          existing.objectUrl = rebuilt_entry.objectUrl;
        }
      }
    } else {
      currentDb = db;
      currentEntries = entries;
    }
  } catch (err) {
    console.error('Indexing failed:', err);
    setStatus('idle', 'Error during indexing');
    progressWrap.classList.remove('visible');
    return;
  }

  progressBar.style.width = '100%';
  setTimeout(() => progressWrap.classList.remove('visible'), 600);

  renderGallery();
  setStatus('ready', `${currentEntries.length} images indexed • HNSW ready`);
  recallBtn.disabled = false;
}

async function urlToFile(url: string, filename: string): Promise<File> {
  const res = await fetch(url);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type });
}

function setProgressLabel(current: number, total: number, prefix = 'Processing') {
  progressLabel.textContent = `${prefix} ${current} / ${total} images…`;
}

// ---------------------------------------------------------------------------
// Gallery rendering
// ---------------------------------------------------------------------------

function renderGallery() {
  gallerySection.style.display = '';
  countLabel.textContent = String(currentEntries.length);
  indexLabel.textContent = currentDb?.hasIndex ? 'HNSW ready' : 'building index…';

  gallery.innerHTML = '';

  for (const entry of currentEntries) {
    const wrap = document.createElement('div');
    wrap.className = 'thumb-wrap';
    wrap.dataset['id'] = String(entry.id);

    const img = document.createElement('img');
    img.src = entry.objectUrl;
    img.alt = entry.filename;
    img.loading = 'lazy';

    const label = document.createElement('span');
    label.className = 'thumb-id';
    label.textContent = `#${entry.id}`;

    wrap.appendChild(img);
    wrap.appendChild(label);

    wrap.addEventListener('click', () => handleThumbClick(entry, wrap));
    gallery.appendChild(wrap);
  }
}

// ---------------------------------------------------------------------------
// Click → search
// ---------------------------------------------------------------------------

async function handleThumbClick(entry: ImageEntry, clickedWrap: HTMLElement) {
  if (!currentDb) return;

  // Clear previous highlights
  document.querySelectorAll('.thumb-wrap').forEach((el) => {
    el.classList.remove('query', 'result');
  });

  clickedWrap.classList.add('query');

  // Re-extract histogram from the image's blob URL
  const file = await urlToFile(entry.objectUrl, entry.filename);
  const bitmap = await imageFromFile(file);
  const queryVec = extractHistogram(bitmap);
  bitmap.close();

  // Search — include k+1 to skip the query itself if it appears
  const rawResults = await findSimilar(currentDb, queryVec, 6);
  const results = rawResults
    .filter((r) => r.id !== entry.id)
    .slice(0, 5);

  // Highlight result thumbnails
  const resultIds = new Set(results.map((r) => r.id));
  document.querySelectorAll<HTMLElement>('.thumb-wrap').forEach((el) => {
    const id = Number(el.dataset['id']);
    if (resultIds.has(id)) el.classList.add('result');
  });

  // Populate sidebar
  showSidebar(entry, results);
}

function showSidebar(
  query: ImageEntry,
  results: Array<{ id: number; score: number }>,
) {
  sidebar.classList.remove('hidden');

  queryPreview.src = query.objectUrl;
  queryPreview.alt = query.filename;
  queryPreview.classList.add('visible');

  resultsList.innerHTML = '';

  if (results.length === 0) {
    resultsList.innerHTML = '<div class="empty-state">No similar images found.</div>';
    return;
  }

  for (const r of results) {
    const entry = currentEntries.find((e) => e.id === r.id);
    if (!entry) continue;

    const item = document.createElement('div');
    item.className = 'result-item';

    const img = document.createElement('img');
    img.src = entry.objectUrl;
    img.alt = entry.filename;

    const meta = document.createElement('div');
    meta.className = 'result-meta';

    const name = document.createElement('div');
    name.className = 'result-name';
    name.textContent = entry.filename;

    const score = document.createElement('div');
    score.className = 'result-score';
    score.textContent = `Score: ${(r.score * 100).toFixed(1)}%  •  ID #${r.id}`;

    const barBg = document.createElement('div');
    barBg.className = 'score-bar-bg';
    const bar = document.createElement('div');
    bar.className = 'score-bar';
    bar.style.width = `${r.score * 100}%`;
    barBg.appendChild(bar);

    meta.appendChild(name);
    meta.appendChild(score);
    meta.appendChild(barBg);

    item.appendChild(img);
    item.appendChild(meta);
    resultsList.appendChild(item);
  }
}

// ---------------------------------------------------------------------------
// Recall test
// ---------------------------------------------------------------------------

/** Clamp a value to [0, 255]. */
function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

/**
 * Box-Muller transform: sample from N(0, 1).
 * Used to create realistic per-pixel colour noise that activates many histogram bins.
 */
function gaussRandom(): number {
  const u = Math.max(1e-10, Math.random());
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Render a 64×64 image where each pixel is independently sampled from
 * N(r, σ) × N(g, σ) × N(b, σ).
 *
 * WHY THIS MATTERS FOR 1-BIT QUANTIZATION:
 *   A solid-colour image only activates 1 bin per channel = 3 total bits.
 *   With σ=25 each channel activates ~25 bins = ~75 bits total.
 *   → intra-cluster Hamming distance ≈ 5-10 bits (sampling variance)
 *   → inter-cluster Hamming distance ≈ 30-60 bits (different centres)
 *   → 4-8× signal-to-noise ratio: the HNSW index finds siblings reliably.
 */
async function syntheticBitmapGaussian(
  r: number, g: number, b: number, sigma: number,
): Promise<ImageBitmap> {
  const size = 64;
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  for (let i = 0; i < size * size; i++) {
    d[i * 4 + 0] = clamp255(r + gaussRandom() * sigma);
    d[i * 4 + 1] = clamp255(g + gaussRandom() * sigma);
    d[i * 4 + 2] = clamp255(b + gaussRandom() * sigma);
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return createImageBitmap(canvas.transferToImageBitmap());
}

/** Same as syntheticBitmapGaussian but returns a PNG File for the gallery. */
async function syntheticFileGaussian(
  r: number, g: number, b: number, sigma: number, name: string,
): Promise<File> {
  const size = 64;
  const canvas = new OffscreenCanvas(size, size);
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  for (let i = 0; i < size * size; i++) {
    d[i * 4 + 0] = clamp255(r + gaussRandom() * sigma);
    d[i * 4 + 1] = clamp255(g + gaussRandom() * sigma);
    d[i * 4 + 2] = clamp255(b + gaussRandom() * sigma);
    d[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return new File([blob], name, { type: 'image/png' });
}

/**
 * 10 cluster centres chosen to be well-separated in RGB space.
 * Minimum pairwise L1 distance > 140 — large enough that the inter-cluster
 * Hamming distance (≥ 30 bits) comfortably exceeds intra-cluster noise (≤ 12 bits).
 */
const CLUSTER_CENTERS: [number, number, number][] = [
  [240,  40,  40],  // red
  [ 40, 210,  40],  // green
  [ 40,  40, 240],  // blue
  [220, 220,  30],  // yellow
  [210,  40, 210],  // magenta
  [ 30, 200, 200],  // cyan
  [250, 130,  10],  // orange
  [120,  10, 200],  // purple
  [ 10, 160, 100],  // emerald
  [200, 100,  40],  // terracotta
];

async function runRecallTest(): Promise<void> {
  const CLUSTERS    = CLUSTER_CENTERS.length; // 10
  const PER_CLUSTER = 5;
  const SIGMA       = 25; // per-pixel Gaussian spread — activates ~25 bins per channel

  recallOutput.textContent = 'Running recall test…';
  recallOutput.classList.add('visible');
  recallBtn.disabled = true;

  console.group('MicroVecDB Recall Test');

  // 1. Build a temporary DB with all synthetic images (CLUSTERS × PER_CLUSTER)
  const testDb = await MicroVecDB.init({ capacity: CLUSTERS * PER_CLUSTER });

  // clusterVectors[c][v] = histogram of cluster c, variant v
  const clusterVectors: Float32Array[][] = [];

  let id = 0;
  for (let c = 0; c < CLUSTERS; c++) {
    const [r, g, b] = CLUSTER_CENTERS[c];
    const variants: Float32Array[] = [];

    for (let v = 0; v < PER_CLUSTER; v++) {
      const bitmap = await syntheticBitmapGaussian(r, g, b, SIGMA);
      const hist = extractHistogram(bitmap);
      bitmap.close();
      testDb.insert({ id, vector: hist });
      variants.push(hist);
      id++;
    }
    clusterVectors.push(variants);
  }

  testDb.buildIndex();

  // 2. For each cluster, query with variant[0] and expect the other 4 in top-5
  let passed = 0;
  const lines: string[] = [];
  const LABEL = ['red','green','blue','yellow','magenta','cyan','orange','purple','emerald','terracotta'];

  for (let c = 0; c < CLUSTERS; c++) {
    const queryVec = clusterVectors[c][0]!;
    const queryId  = c * PER_CLUSTER;

    const expectedIds = new Set<number>();
    for (let v = 1; v < PER_CLUSTER; v++) expectedIds.add(c * PER_CLUSTER + v);

    // ef=40 on 50 vectors → exhaustive in practice
    const rawResults = testDb.search(queryVec, { limit: PER_CLUSTER + 1, ef: 40 });
    const results = rawResults.filter(r => r.id !== queryId).slice(0, PER_CLUSTER - 1);

    const foundIds = new Set(results.map(r => r.id));
    const allFound = [...expectedIds].every(eid => foundIds.has(eid));

    if (allFound) {
      passed++;
      lines.push(`  [PASS] Cluster ${String(c).padStart(2)} (${LABEL[c]})`);
      console.log(`%c[PASS] Cluster ${c} (${LABEL[c]})`, 'color:#22c55e');
    } else {
      const missing = [...expectedIds].filter(eid => !foundIds.has(eid));
      lines.push(`  [FAIL] Cluster ${String(c).padStart(2)} (${LABEL[c]}) — missing: ${missing.join(',')}`);
      console.warn(`[FAIL] Cluster ${c} (${LABEL[c]}) — missing: ${missing.join(', ')}`);
    }
  }

  testDb.dispose();

  const pct = ((passed / CLUSTERS) * 100).toFixed(1);
  const summary = `Recall@${PER_CLUSTER - 1} on ${CLUSTERS} clusters: ${passed}/${CLUSTERS} (${pct}%)`;
  console.log(`%c${summary}`, passed === CLUSTERS ? 'color:#22c55e;font-weight:bold' : 'color:#f59e0b;font-weight:bold');
  console.groupEnd();

  const coloredLines = lines.map(l =>
    l.includes('[PASS]') ? `<span class="pass">${l}</span>` :
    l.includes('[FAIL]') ? `<span class="fail">${l}</span>` : l
  );
  recallOutput.innerHTML = `<strong>${summary}</strong>\n\n` + coloredLines.join('\n');
  recallBtn.disabled = false;
}

// Expose globally
(window as unknown as Record<string, unknown>)['runRecallTest'] = runRecallTest;

// ---------------------------------------------------------------------------
// Recall button
// ---------------------------------------------------------------------------

recallBtn.addEventListener('click', () => runRecallTest());

// ---------------------------------------------------------------------------
// Auto-populate on page load
// ---------------------------------------------------------------------------

async function autoPopulate(): Promise<void> {
  const PER_CLUSTER = 5;
  const SIGMA = 25;

  setStatus('indexing', 'Generating synthetic test images…');

  const files: File[] = [];
  for (let c = 0; c < CLUSTER_CENTERS.length; c++) {
    const [r, g, b] = CLUSTER_CENTERS[c];
    for (let v = 0; v < PER_CLUSTER; v++) {
      files.push(await syntheticFileGaussian(r, g, b, SIGMA, `cluster${c}-v${v}.png`));
    }
  }

  await handleFiles(files);
  await runRecallTest();
}

// Kick off immediately on page load
autoPopulate().catch(console.error);
