import { useState, useRef, useEffect, useCallback } from 'react';
import { MicroVecDB } from '@microvecdb/core';
import { extractPHash, extractPHashFromFile } from './phash.js';

// ── Types ──────────────────────────────────────────────────────────────────

interface ImageEntry {
  id: number;
  filename: string;
  url: string;
}

interface SearchResult {
  id: number;
  score: number;
}

type Phase = 'idle' | 'indexing' | 'ready' | 'testing';

// ── Gaussian noise helpers (for auto-test) ────────────────────────────────

function gaussRandom(): number {
  const u = Math.max(1e-10, Math.random());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

function clamp255(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

async function syntheticFile(r: number, g: number, b: number, sigma: number, name: string): Promise<File> {
  const SIZE = 64;
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx    = canvas.getContext('2d')!;
  const img    = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    img.data[i * 4]     = clamp255(r + gaussRandom() * sigma);
    img.data[i * 4 + 1] = clamp255(g + gaussRandom() * sigma);
    img.data[i * 4 + 2] = clamp255(b + gaussRandom() * sigma);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const blob = await canvas.convertToBlob({ type: 'image/png' });
  return new File([blob], name, { type: 'image/png' });
}

const CLUSTER_CENTERS: [number, number, number][] = [
  [240, 40,  40], [40,  210, 40], [40,  40,  240], [220, 220, 30],
  [210, 40,  210], [30, 200, 200], [250, 130, 10], [120, 10,  200],
  [10,  160, 100], [200, 100, 40],
];
const PER_CLUSTER = 5;
const SIGMA       = 25;

// ── Badge ─────────────────────────────────────────────────────────────────

function Badge({ children, color = 'zinc' }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    zinc:    'bg-zinc-800 text-zinc-300 border-zinc-700',
    indigo:  'bg-indigo-900/50 text-indigo-300 border-indigo-700',
    emerald: 'bg-emerald-900/50 text-emerald-300 border-emerald-700',
    amber:   'bg-amber-900/50 text-amber-300 border-amber-700',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${colors[color] ?? colors.zinc}`}>
      {children}
    </span>
  );
}

// ── App ───────────────────────────────────────────────────────────────────

export default function App() {
  const [phase, setPhase]         = useState<Phase>('idle');
  const [entries, setEntries]     = useState<ImageEntry[]>([]);
  const [progress, setProgress]   = useState({ pct: 0, label: '' });
  const [selected, setSelected]   = useState<ImageEntry | null>(null);
  const [results, setResults]     = useState<SearchResult[]>([]);
  const [latency, setLatency]     = useState<number | null>(null);
  const [testLog, setTestLog]     = useState<string[]>([]);
  const [dragging, setDragging]   = useState(false);

  const dbRef      = useRef<MicroVecDB | null>(null);
  const nextIdRef  = useRef(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logEndRef  = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [testLog]);

  // ── Indexing ──────────────────────────────────────────────────────────

  const indexFiles = useCallback(async (files: File[]) => {
    if (!files.length) return;
    setPhase('indexing');
    setSelected(null);
    setResults([]);

    const db = await MicroVecDB.init({ capacity: Math.max(files.length * 2, 64) });
    const newEntries: ImageEntry[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i]!;
      const id   = nextIdRef.current++;
      const feat = await extractPHashFromFile(file);
      db.insert({ id, vector: feat });
      newEntries.push({ id, filename: file.name, url: URL.createObjectURL(file) });
      setProgress({ pct: Math.round(((i + 1) / files.length) * 100), label: `Indexing ${i + 1}/${files.length}…` });
    }

    db.buildIndex();
    dbRef.current = db;
    setEntries(prev => {
      const updated = [...prev, ...newEntries];
      return updated;
    });
    setPhase('ready');
  }, []);

  // ── Search ────────────────────────────────────────────────────────────

  const searchSimilar = useCallback(async (entry: ImageEntry) => {
    if (!dbRef.current) return;
    setSelected(entry);

    const bitmap = await createImageBitmap(await fetch(entry.url).then(r => r.blob()));
    const feat   = extractPHash(bitmap);
    bitmap.close();

    const t0      = performance.now();
    const hits    = dbRef.current.search(feat, { limit: 6 });
    const elapsed = performance.now() - t0;

    setLatency(elapsed);
    // Exclude the query image itself
    setResults(hits.filter(h => h.id !== entry.id).slice(0, 5).map(h => ({ id: h.id, score: h.score })));
  }, []);

  // ── Auto test ─────────────────────────────────────────────────────────

  async function runAutoTest() {
    setPhase('testing');
    setTestLog(['Generating synthetic images…']);
    const log = (l: string) => setTestLog(prev => [...prev, l]);

    try {
      const CLUSTERS = CLUSTER_CENTERS.length;
      const TOTAL    = CLUSTERS * PER_CLUSTER;

      const db = await MicroVecDB.init({ capacity: TOTAL * 2 });
      const entryMap = new Map<number, ImageEntry>();
      let globalId = 0;

      for (let c = 0; c < CLUSTERS; c++) {
        const [r, g, b] = CLUSTER_CENTERS[c]!;
        for (let v = 0; v < PER_CLUSTER; v++) {
          const file  = await syntheticFile(r, g, b, SIGMA, `cluster${c}_v${v}.png`);
          const feat  = await extractPHashFromFile(file);
          const id    = globalId++;
          db.insert({ id, vector: feat });
          entryMap.set(id, { id, filename: file.name, url: URL.createObjectURL(file) });
        }
      }

      db.buildIndex();
      log(`\nHNSW index built over ${TOTAL} images. Running Recall@5…\n`);

      let hits = 0;
      for (let c = 0; c < CLUSTERS; c++) {
        const baseId = c * PER_CLUSTER;
        for (let v = 0; v < PER_CLUSTER; v++) {
          const qId   = baseId + v;
          const qEntry = entryMap.get(qId)!;
          const bmp   = await createImageBitmap(await fetch(qEntry.url).then(r => r.blob()));
          const feat  = extractPHash(bmp);
          bmp.close();

          const res      = db.search(feat, { limit: 6, ef: 40 });
          const sameCluster = res.filter(r => r.id !== qId && Math.floor(r.id / PER_CLUSTER) === c);
          const found    = sameCluster.length > 0;
          hits          += found ? 1 : 0;
          const topIds   = res.filter(r => r.id !== qId).slice(0, 5).map(r => r.id).join(', ');
          log(`  ${found ? '✅' : '❌'} img#${qId} (cluster ${c}) → top5 excl. self: [${topIds}]`);
        }
      }

      db.dispose();

      const pct = ((hits / TOTAL) * 100).toFixed(1);
      log(`\n${'━'.repeat(42)}`);
      log(`Recall@5: ${hits}/${TOTAL} (${pct}%)  — target ≥ 95%`);
      log(hits >= Math.round(TOTAL * 0.95)
        ? '✅ PASS — perceptual fingerprint survives 1-bit quantization!'
        : '⚠️  Below target — check cluster separation.');
    } catch (err) {
      log(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPhase('idle');
    }
  }

  // ── Drop handlers ─────────────────────────────────────────────────────

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    if (files.length) void indexFiles(files);
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length) void indexFiles(files);
    e.target.value = '';
  }

  // ── Render ────────────────────────────────────────────────────────────

  const resultIds = new Set(results.map(r => r.id));

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100 font-[Inter,system-ui,sans-serif]">

      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-zinc-800 bg-zinc-900/80 backdrop-blur sticky top-0 z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-purple-500 to-pink-600 flex items-center justify-center text-sm">🔍</div>
          <span className="font-semibold tracking-tight">Visual Matcher</span>
          <span className="text-zinc-600 text-xs hidden sm:block">by MicroVecDB</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge color="zinc">
            {latency !== null ? `Search: ${latency < 1 ? `${(latency * 1000).toFixed(0)}µs` : `${latency.toFixed(2)}ms`}` : 'Search: < 1ms'}
          </Badge>
          {entries.length > 0 && <Badge color="emerald">● {entries.length} indexed</Badge>}
          {phase === 'indexing' && <Badge color="amber">⏳ Indexing…</Badge>}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* Left: controls + gallery */}
        <div className="flex flex-col flex-1 min-w-0 overflow-y-auto p-4 gap-4">

          {/* Controls row */}
          <div className="flex gap-3 flex-wrap">
            {/* Drop zone */}
            <div
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`
                flex items-center gap-3 px-5 py-3 rounded-xl border-2 border-dashed cursor-pointer
                transition-colors flex-1 min-w-48
                ${dragging ? 'border-purple-500 bg-purple-950/30' : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900'}
                ${phase === 'indexing' ? 'pointer-events-none opacity-50' : ''}
              `}
            >
              <span className="text-2xl">🖼️</span>
              <div>
                <p className="text-sm font-semibold">Drop images here</p>
                <p className="text-xs text-zinc-500">or click to browse (500+ supported)</p>
              </div>
              <input ref={fileInputRef} type="file" multiple accept="image/*" className="hidden" onChange={onFileChange} />
            </div>

            {/* Auto test button */}
            <button
              onClick={() => void runAutoTest()}
              disabled={phase !== 'idle' && phase !== 'ready'}
              className="px-5 py-3 rounded-xl bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed border border-zinc-700 text-sm font-semibold transition-colors whitespace-nowrap"
            >
              {phase === 'testing' ? '⏳ Testing…' : '⚡ Auto Test'}
            </button>
          </div>

          {/* Progress */}
          {phase === 'indexing' && (
            <div className="flex flex-col gap-1.5">
              <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-150"
                  style={{ width: `${progress.pct}%` }}
                />
              </div>
              <p className="text-xs text-zinc-400">{progress.label}</p>
            </div>
          )}

          {/* Test log */}
          {testLog.length > 0 && (
            <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4 font-mono text-xs text-zinc-300 max-h-64 overflow-y-auto whitespace-pre-wrap leading-relaxed">
              {testLog.join('\n')}
              <div ref={logEndRef} />
            </div>
          )}

          {/* Gallery */}
          {entries.length > 0 && (
            <div>
              <p className="text-xs text-zinc-500 font-medium uppercase tracking-wider mb-3">
                Gallery — click any image to find similar
              </p>
              {/* CSS columns masonry */}
              <div className="columns-3 sm:columns-4 md:columns-5 lg:columns-6 xl:columns-7 gap-2">
                {entries.map(entry => {
                  const isQuery  = selected?.id === entry.id;
                  const isResult = resultIds.has(entry.id);
                  return (
                    <div
                      key={entry.id}
                      onClick={() => void searchSimilar(entry)}
                      className={`
                        break-inside-avoid mb-2 rounded-lg overflow-hidden cursor-pointer
                        border-2 transition-all duration-150 hover:scale-[1.02]
                        ${isQuery  ? 'border-purple-500 ring-2 ring-purple-500/30' :
                          isResult ? 'border-emerald-500 ring-2 ring-emerald-500/30' :
                                     'border-transparent hover:border-zinc-600'}
                      `}
                    >
                      <img
                        src={entry.url}
                        alt={entry.filename}
                        className="w-full h-auto block"
                        loading="lazy"
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Empty state */}
          {entries.length === 0 && phase === 'idle' && (
            <div className="flex flex-col items-center justify-center flex-1 gap-3 text-zinc-600 py-16">
              <span className="text-5xl">🖼️</span>
              <p className="text-sm">Drop images or run Auto Test to start.</p>
            </div>
          )}
        </div>

        {/* Right: results sidebar */}
        {selected && (
          <aside className="w-64 shrink-0 border-l border-zinc-800 bg-zinc-900 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-zinc-800">
              <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Query</p>
              <img src={selected.url} alt="query" className="w-full rounded-lg object-cover aspect-square" />
              <p className="text-xs text-zinc-500 mt-1.5 truncate">{selected.filename}</p>
            </div>

            <div className="flex-1 overflow-y-auto p-4">
              <p className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
                Similar ({results.length})
              </p>
              <div className="flex flex-col gap-2">
                {results.map((res, i) => {
                  const entry = entries.find(e => e.id === res.id);
                  if (!entry) return null;
                  return (
                    <div
                      key={res.id}
                      onClick={() => void searchSimilar(entry)}
                      className="flex items-center gap-2.5 p-2 rounded-lg bg-zinc-800 border border-zinc-700 cursor-pointer hover:border-zinc-600 transition-colors"
                    >
                      <img src={entry.url} alt="" className="w-10 h-10 rounded-md object-cover shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-semibold text-zinc-200">#{i + 1}</p>
                        <p className="text-xs text-zinc-500 truncate">{entry.filename}</p>
                        <div className="h-1 bg-zinc-700 rounded-full mt-1 overflow-hidden">
                          <div
                            className="h-full bg-emerald-500 rounded-full"
                            style={{ width: `${(res.score * 100).toFixed(0)}%` }}
                          />
                        </div>
                        <p className="text-[10px] text-zinc-500 mt-0.5">{(res.score * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </aside>
        )}
      </div>
    </div>
  );
}
