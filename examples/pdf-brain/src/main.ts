// This file is superseded by App.tsx (React rewrite). Kept as empty module.
export {};

// ─── DOM refs ────────────────────────────────────────────────────────────────

const dropZone = document.getElementById('drop-zone') as HTMLDivElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const dropLabel = document.getElementById('drop-label') as HTMLParagraphElement;
const dropIcon = document.getElementById('drop-icon') as HTMLSpanElement;
const fileName = document.getElementById('file-name') as HTMLParagraphElement;
const progressWrap = document.getElementById('progress-wrap') as HTMLDivElement;
const progressFill = document.getElementById('progress-bar-fill') as HTMLDivElement;
const progressText = document.getElementById('progress-text') as HTMLParagraphElement;
const statusMsg = document.getElementById('status-msg') as HTMLParagraphElement;
const queryInput = document.getElementById('query-input') as HTMLInputElement;
const searchBtn = document.getElementById('search-btn') as HTMLButtonElement;
const resultsList = document.getElementById('results-list') as HTMLUListElement;
const resultsEmpty = document.getElementById('results-empty') as HTMLLIElement;
const dbStats = document.getElementById('db-stats') as HTMLDivElement;
const statChunks = document.getElementById('stat-chunks') as HTMLSpanElement;
const autoTestBtn = document.getElementById('auto-test-btn') as HTMLButtonElement;
const autoTestOutput = document.getElementById('auto-test-output') as HTMLDivElement;

// ─── App state ───────────────────────────────────────────────────────────────

let embedder: FeatureExtractionPipeline | null = null;
let activeDb: MicroVecDB | null = null;
let activeChunks: TextChunk[] = [];
let isIndexing = false;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function setProgress(current: number, total: number, label?: string): void {
  const pct = total > 0 ? Math.round((current / total) * 100) : 0;
  progressFill.style.width = `${pct}%`;
  progressText.textContent =
    label ?? (total > 0 ? `Embedding chunk ${current} / ${total}` : 'Preparing…');
}

function setStatus(msg: string, type: 'default' | 'success' | 'error' = 'default'): void {
  statusMsg.textContent = msg;
  statusMsg.className = type === 'default' ? '' : type;
}

function showProgress(visible: boolean): void {
  progressWrap.classList.toggle('visible', visible);
}

function enableSearch(): void {
  queryInput.disabled = false;
  searchBtn.disabled = false;
  dbStats.classList.add('visible');
  statChunks.textContent = `${activeChunks.length} chunks`;
  queryInput.focus();
}

function truncate(text: string, maxChars = 320): string {
  return text.length <= maxChars ? text : text.slice(0, maxChars).trimEnd() + '…';
}

function renderResults(hits: SearchHit[]): void {
  // Clear old results (keep the empty placeholder in DOM)
  resultsList.innerHTML = '';

  if (hits.length === 0) {
    resultsList.appendChild(resultsEmpty);
    resultsEmpty.textContent = 'No results found for that query.';
    return;
  }

  for (let i = 0; i < hits.length; i++) {
    const { chunk, score } = hits[i]!;
    const li = document.createElement('li');
    li.className = 'result-item';
    li.innerHTML = `
      <div class="result-rank">#${i + 1}</div>
      <div class="result-meta">
        <span class="result-score">${(score * 100).toFixed(1)}%</span>
        <span class="result-page">page ${chunk.page}</span>
        <span>chunk&nbsp;#${chunk.id}</span>
      </div>
      <div class="result-text">${truncate(chunk.text)}</div>
    `;
    resultsList.appendChild(li);
  }
}

// ─── Core pipeline ───────────────────────────────────────────────────────────

async function ensureEmbedder(): Promise<FeatureExtractionPipeline> {
  if (embedder) return embedder;
  setProgress(0, 0, 'Loading embedding model (first time may take ~30s)…');
  embedder = await loadEmbedder();
  return embedder;
}

async function processFile(file: File): Promise<void> {
  if (isIndexing) return;
  isIndexing = true;

  // Reset UI
  dropZone.classList.remove('ready');
  showProgress(true);
  queryInput.disabled = true;
  searchBtn.disabled = true;
  dbStats.classList.remove('visible');
  setProgress(0, 0, 'Reading PDF…');
  setStatus('');
  resultsList.innerHTML = '';
  resultsList.appendChild(resultsEmpty);
  resultsEmpty.textContent = 'Results will appear here after a search.';

  try {
    // 1. Read file bytes
    const buffer = await file.arrayBuffer();

    // 2. Extract chunks
    setProgress(0, 0, 'Extracting text from PDF…');
    setStatus('Parsing pages…');
    const chunks = await extractChunks(buffer);
    if (chunks.length === 0) {
      throw new Error('No text found in this PDF. Is it a scanned image?');
    }
    setStatus(`Extracted ${chunks.length} chunks from ${file.name}.`);

    // 3. Load embedder
    const pipe = await ensureEmbedder();

    // 4. Build DB with progress
    setStatus('Generating embeddings…');
    const embedFn = (text: string) => embed(pipe, text);

    const { db, chunks: indexedChunks } = await buildDatabase(
      chunks,
      embedFn,
      (current, total) => {
        setProgress(current, total);
      },
    );

    activeDb = db;
    activeChunks = indexedChunks;

    setProgress(chunks.length, chunks.length, 'Index ready!');
    setStatus(
      `Done — ${chunks.length} chunks indexed with HNSW.`,
      'success',
    );
    dropZone.classList.add('ready');
    dropIcon.textContent = '✅';
    enableSearch();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`Error: ${msg}`, 'error');
    setProgress(0, 0, 'Failed.');
    console.error('[pdf-brain] indexing error', err);
  } finally {
    isIndexing = false;
  }
}

async function handleSearch(): Promise<void> {
  const query = queryInput.value.trim();
  if (!query || !activeDb || activeChunks.length === 0) return;

  searchBtn.disabled = true;
  searchBtn.textContent = '…';

  try {
    const pipe = await ensureEmbedder();
    const queryVec = await embed(pipe, query);
    const hits = await queryDatabase(activeDb, queryVec, activeChunks, 5);
    renderResults(hits);
  } catch (err) {
    console.error('[pdf-brain] search error', err);
    setStatus('Search failed: ' + (err instanceof Error ? err.message : String(err)), 'error');
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
  }
}

// ─── Event wiring ─────────────────────────────────────────────────────────────

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer?.files[0];
  if (file && file.type === 'application/pdf') {
    fileName.textContent = file.name;
    dropLabel.innerHTML = '<strong>Click to choose</strong> or drag a PDF here';
    void processFile(file);
  }
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files?.[0];
  if (file) {
    fileName.textContent = file.name;
    void processFile(file);
  }
});

searchBtn.addEventListener('click', () => void handleSearch());

queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') void handleSearch();
});

// ─── Recall test ─────────────────────────────────────────────────────────────

/**
 * window.runRecallTest()
 *
 * Runs 10 synthetic queries against the currently indexed chunks and prints
 * Recall@5 to the console.
 *
 * Query strategy: take the first sentence (up to 120 chars) of evenly-spaced
 * chunks as the query; expect the same chunk to appear in the top-5 results.
 */
async function runRecallTest(): Promise<void> {
  if (!activeDb || activeChunks.length === 0) {
    console.warn('[recall-test] No indexed PDF. Drop and index a PDF first.');
    return;
  }

  const pipe = await ensureEmbedder();
  const N = 10;
  const step = Math.max(1, Math.floor(activeChunks.length / N));

  // Pick up to N evenly-spaced chunks as test queries
  const testCases: Array<{ chunk: TextChunk; query: string }> = [];
  for (let i = 0; i < N && i * step < activeChunks.length; i++) {
    const chunk = activeChunks[i * step]!;
    // Use the first ~120 chars (roughly one sentence) as the query
    const query = chunk.text.slice(0, 120).trimEnd();
    testCases.push({ chunk, query });
  }

  let hits = 0;
  console.group(`[recall-test] Running Recall@5 over ${testCases.length} queries`);

  for (const { chunk, query } of testCases) {
    const queryVec = await embed(pipe, query);
    const results = await queryDatabase(activeDb, queryVec, activeChunks, 5);
    const found = results.some((r) => r.chunk.id === chunk.id);
    hits += found ? 1 : 0;
    console.log(
      `  chunk#${chunk.id} (page ${chunk.page}) → ${found ? '✅ HIT' : '❌ MISS'}`,
      `| top-5 ids: [${results.map((r) => r.chunk.id).join(', ')}]`,
    );
  }

  console.log(`\nRecall@5: ${hits}/${testCases.length}`);
  console.groupEnd();
}

// Attach to window for console access
(window as unknown as Record<string, unknown>).runRecallTest = runRecallTest;

// ─── Auto Test (built-in corpus) ─────────────────────────────────────────────

async function runAutoTest(): Promise<void> {
  autoTestBtn.disabled = true;
  autoTestBtn.textContent = 'Running…';
  autoTestOutput.style.display = 'block';
  autoTestOutput.textContent = 'Loading embedding model (first run ~30s, then cached)…\n';

  const log = (line: string) => {
    autoTestOutput.textContent += line + '\n';
    autoTestOutput.scrollTop = autoTestOutput.scrollHeight;
  };

  try {
    const pipe = await loadEmbedder();
    log(`Model ready. Embedding ${SAMPLE_CORPUS.length} paragraphs…\n`);

    const db = await MicroVecDB.init({ capacity: SAMPLE_CORPUS.length * 2 });

    for (let i = 0; i < SAMPLE_CORPUS.length; i++) {
      const entry = SAMPLE_CORPUS[i]!;
      const vec = await embed(pipe, entry.text);
      db.insert({ id: entry.id, vector: vec });
      log(`  [${String(i + 1).padStart(2, ' ')}/${SAMPLE_CORPUS.length}] embedded paragraph #${entry.id}`);
    }

    db.buildIndex();
    log(`\nHNSW index built. Running Recall@5 queries…\n`);

    let hits = 0;
    for (const entry of SAMPLE_CORPUS) {
      const queryVec = await embed(pipe, entry.query);
      const results = db.search(queryVec, { limit: 5 });
      const found = results.some(r => r.id === entry.id);
      hits += found ? 1 : 0;
      const topIds = results.map(r => r.id).join(', ');
      log(`  ${found ? '✅' : '❌'} query: "${entry.query.slice(0, 48)}…" → top5: [${topIds}]`);
    }

    db.dispose();

    const pct = ((hits / SAMPLE_CORPUS.length) * 100).toFixed(1);
    log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    log(`Recall@5: ${hits}/${SAMPLE_CORPUS.length} (${pct}%)  — target ≥ 85%`);
    log(hits >= Math.round(SAMPLE_CORPUS.length * 0.85)
      ? `✅ PASS — real semantic embeddings survive 1-bit quantization!`
      : `⚠️  Below target — try increasing ef in search options.`);

  } catch (err) {
    log(`\nError: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    autoTestBtn.disabled = false;
    autoTestBtn.textContent = 'Run Auto Test';
  }
}

autoTestBtn.addEventListener('click', () => void runAutoTest());
