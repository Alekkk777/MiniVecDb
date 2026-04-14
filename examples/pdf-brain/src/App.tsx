import { useState, useRef, useEffect, useCallback } from 'react';
import { MicroVecDB } from '@microvecdb/core';
import { loadEmbedder, embed } from './embedder.js';
import { extractChunks, type TextChunk } from './pdf-extractor.js';
import { buildDatabase, queryDatabase, type SearchHit } from './db.js';
import { SAMPLE_CORPUS } from './sample-corpus.js';
import type { FeatureExtractionPipeline } from '@xenova/transformers';

// ── Types ──────────────────────────────────────────────────────────────────

type Phase = 'idle' | 'auto-testing' | 'loading-pdf' | 'indexing' | 'ready';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  hits?: SearchHit[];
  error?: boolean;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function Badge({ children, color = 'zinc' }: { children: React.ReactNode; color?: string }) {
  const colors: Record<string, string> = {
    zinc:   'bg-zinc-800 text-zinc-300 border-zinc-700',
    indigo: 'bg-indigo-900/50 text-indigo-300 border-indigo-700',
    emerald:'bg-emerald-900/50 text-emerald-300 border-emerald-700',
    amber:  'bg-amber-900/50 text-amber-300 border-amber-700',
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${colors[color] ?? colors.zinc}`}>
      {children}
    </span>
  );
}

function truncate(text: string, max = 280) {
  return text.length <= max ? text : text.slice(0, max).trimEnd() + '…';
}

// ── Component ──────────────────────────────────────────────────────────────

export default function App() {
  const [phase, setPhase]       = useState<Phase>('idle');
  const [progress, setProgress] = useState({ pct: 0, label: 'Preparing…' });
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery]       = useState('');
  const [autoLog, setAutoLog]   = useState<string[]>([]);
  const [chunkCount, setChunkCount] = useState(0);
  const [pdfName, setPdfName]   = useState('');
  const [dragging, setDragging] = useState(false);

  const embedderRef = useRef<FeatureExtractionPipeline | null>(null);
  const dbRef       = useRef<MicroVecDB | null>(null);
  const chunksRef   = useRef<TextChunk[]>([]);
  const chatEndRef  = useRef<HTMLDivElement>(null);
  const fileInputRef= useRef<HTMLInputElement>(null);
  const logEndRef   = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [autoLog]);

  async function getEmbedder() {
    if (embedderRef.current) return embedderRef.current;
    embedderRef.current = await loadEmbedder();
    return embedderRef.current;
  }

  const pushMsg = useCallback((msg: Omit<Message, 'id'>) => {
    setMessages(prev => [...prev, { ...msg, id: `${Date.now()}-${Math.random()}` }]);
  }, []);

  // ── PDF processing ────────────────────────────────────────────────────

  async function processFile(file: File) {
    setPhase('loading-pdf');
    setProgress({ pct: 0, label: 'Reading PDF…' });
    setPdfName(file.name);
    setMessages([]);
    dbRef.current?.dispose();
    dbRef.current = null;

    try {
      const buffer = await file.arrayBuffer();
      const chunks = await extractChunks(buffer);
      if (!chunks.length) throw new Error('No text found — is this a scanned image PDF?');

      setPhase('indexing');
      setProgress({ pct: 0, label: `Extracted ${chunks.length} chunks — loading model…` });

      const pipe = await getEmbedder();
      const { db, chunks: indexed } = await buildDatabase(
        chunks,
        (t) => embed(pipe, t),
        (cur, tot) => setProgress({
          pct: Math.round((cur / tot) * 100),
          label: `Embedding ${cur} / ${tot} chunks…`,
        }),
      );

      dbRef.current  = db;
      chunksRef.current = indexed;
      setChunkCount(indexed.length);
      setPhase('ready');
      pushMsg({
        role: 'assistant',
        text: `Ready. Indexed **${indexed.length} chunks** from **${file.name}**. Ask me anything.`,
      });
    } catch (err) {
      setPhase('idle');
      pushMsg({ role: 'assistant', text: `Error: ${err instanceof Error ? err.message : String(err)}`, error: true });
    }
  }

  // ── Search ────────────────────────────────────────────────────────────

  async function handleSearch() {
    const q = query.trim();
    if (!q || !dbRef.current || phase !== 'ready') return;

    pushMsg({ role: 'user', text: q });
    setQuery('');

    try {
      const pipe = await getEmbedder();
      const vec  = await embed(pipe, q);
      const hits = await queryDatabase(dbRef.current, vec, chunksRef.current, 5);
      pushMsg({
        role: 'assistant',
        text: hits.length
          ? `Found ${hits.length} relevant passage${hits.length > 1 ? 's' : ''}:`
          : 'No relevant passages found for that query.',
        hits,
      });
    } catch (err) {
      pushMsg({ role: 'assistant', text: `Search error: ${err instanceof Error ? err.message : String(err)}`, error: true });
    }
  }

  // ── Auto test ─────────────────────────────────────────────────────────

  async function runAutoTest() {
    setPhase('auto-testing');
    setAutoLog(['Loading model (first run ~30 s, then cached)…']);

    const log = (line: string) => setAutoLog(prev => [...prev, line]);

    try {
      const pipe = await loadEmbedder();
      log(`Model ready. Embedding ${SAMPLE_CORPUS.length} paragraphs…\n`);

      const db = await MicroVecDB.init({ capacity: SAMPLE_CORPUS.length * 2 });

      for (let i = 0; i < SAMPLE_CORPUS.length; i++) {
        const e = SAMPLE_CORPUS[i]!;
        db.insert({ id: e.id, vector: await embed(pipe, e.text) });
        log(`  [${String(i + 1).padStart(2)}/${SAMPLE_CORPUS.length}] embedded paragraph #${e.id}`);
      }

      db.buildIndex();
      log('\nHNSW index built. Running Recall@5 queries…\n');

      let hits = 0;
      for (const e of SAMPLE_CORPUS) {
        const results = db.search(await embed(pipe, e.query), { limit: 5 });
        const found   = results.some(r => r.id === e.id);
        hits += found ? 1 : 0;
        log(`  ${found ? '✅' : '❌'} "${e.query.slice(0, 50)}…"`);
      }

      db.dispose();

      const pct = ((hits / SAMPLE_CORPUS.length) * 100).toFixed(1);
      log(`\n${'━'.repeat(40)}`);
      log(`Recall@5: ${hits}/${SAMPLE_CORPUS.length} (${pct}%)  — target ≥ 85%`);
      log(hits >= Math.round(SAMPLE_CORPUS.length * 0.85)
        ? '✅ PASS — real semantic embeddings survive 1-bit quantization!'
        : '⚠️  Below target.');
    } catch (err) {
      log(`\nError: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setPhase('idle');
    }
  }

  // ── Drag & drop ───────────────────────────────────────────────────────

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.type === 'application/pdf') void processFile(file);
  }

  function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void processFile(file);
  }

  // ── Render ────────────────────────────────────────────────────────────

  const busy = phase === 'loading-pdf' || phase === 'indexing';

  return (
    <div className="flex flex-col h-screen bg-zinc-950 text-zinc-100 font-[Inter,system-ui,sans-serif]">

      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-zinc-800 bg-zinc-900/80 backdrop-blur sticky top-0 z-10">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-sm">🧠</div>
          <span className="font-semibold tracking-tight">PDF Brain</span>
          <span className="text-zinc-600 text-xs hidden sm:block">by MicroVecDB</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge color="zinc">RAM ~6 MB</Badge>
          {phase === 'ready' && <Badge color="emerald">● {chunkCount} chunks</Badge>}
          {busy && <Badge color="amber">⏳ Indexing…</Badge>}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* Left column — upload + auto test */}
        <aside className="w-72 shrink-0 border-r border-zinc-800 flex flex-col gap-4 p-4 overflow-y-auto">

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`
              flex flex-col items-center gap-3 p-6 rounded-xl border-2 border-dashed cursor-pointer
              transition-colors text-center select-none
              ${dragging ? 'border-indigo-500 bg-indigo-950/30' : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900'}
              ${busy ? 'pointer-events-none opacity-50' : ''}
            `}
          >
            <span className="text-4xl">📄</span>
            <div>
              <p className="text-sm font-semibold">{pdfName || 'Drop a PDF'}</p>
              <p className="text-xs text-zinc-500 mt-1">or click to browse</p>
            </div>
            <input ref={fileInputRef} type="file" accept=".pdf" className="hidden" onChange={onFileChange} />
          </div>

          {/* Progress */}
          {busy && (
            <div className="flex flex-col gap-2">
              <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-150"
                  style={{ width: `${progress.pct}%` }}
                />
              </div>
              <p className="text-xs text-zinc-400">{progress.label}</p>
            </div>
          )}

          {/* Divider */}
          <div className="flex items-center gap-2 text-zinc-600 text-xs">
            <div className="flex-1 h-px bg-zinc-800" />
            or
            <div className="flex-1 h-px bg-zinc-800" />
          </div>

          {/* Auto test */}
          <div className="flex flex-col gap-3 bg-zinc-900 border border-zinc-800 rounded-xl p-4">
            <p className="text-xs text-zinc-400">
              Runs a built-in 20-paragraph corpus through all-MiniLM-L6-v2 and measures Recall@5 — no PDF needed.
            </p>
            <button
              onClick={() => void runAutoTest()}
              disabled={phase !== 'idle'}
              className="w-full py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold transition-colors"
            >
              {phase === 'auto-testing' ? 'Running…' : '⚡ Run Auto Test'}
            </button>

            {autoLog.length > 0 && (
              <div className="bg-zinc-950 rounded-lg p-3 font-mono text-xs text-zinc-300 max-h-60 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                {autoLog.join('\n')}
                <div ref={logEndRef} />
              </div>
            )}
          </div>
        </aside>

        {/* Right column — chat */}
        <div className="flex flex-col flex-1 min-w-0">

          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
            {messages.length === 0 && phase === 'idle' && (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-zinc-600">
                <span className="text-5xl">🧠</span>
                <p className="text-sm">Drop a PDF to start asking questions about it.</p>
              </div>
            )}

            {messages.map(msg => (
              <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`
                  max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed
                  ${msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-br-sm'
                    : msg.error
                      ? 'bg-red-950 border border-red-800 text-red-300 rounded-bl-sm'
                      : 'bg-zinc-800 text-zinc-100 rounded-bl-sm'
                  }
                `}>
                  <p>{msg.text}</p>

                  {msg.hits && msg.hits.length > 0 && (
                    <div className="mt-3 space-y-2">
                      {msg.hits.map((hit, i) => (
                        <div key={i} className="bg-zinc-900 rounded-xl p-3 border border-zinc-700">
                          <div className="flex items-center gap-2 mb-1.5">
                            <span className="text-xs font-bold text-indigo-400">#{i + 1}</span>
                            <span className="text-xs text-zinc-500">p.{hit.chunk.page} · chunk #{hit.chunk.id}</span>
                            <span className="ml-auto text-xs font-semibold text-emerald-400">
                              {(hit.score * 100).toFixed(1)}%
                            </span>
                          </div>
                          <p className="text-xs text-zinc-300 leading-relaxed">{truncate(hit.chunk.text)}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <div className="border-t border-zinc-800 p-3 bg-zinc-900/50">
            <form
              onSubmit={(e) => { e.preventDefault(); void handleSearch(); }}
              className="flex gap-2"
            >
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                disabled={phase !== 'ready'}
                placeholder={phase === 'ready' ? `Ask anything about ${pdfName}…` : 'Load a PDF to search…'}
                className="flex-1 bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 outline-none focus:border-indigo-500 transition-colors disabled:opacity-40"
              />
              <button
                type="submit"
                disabled={phase !== 'ready' || !query.trim()}
                className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl text-sm font-semibold transition-colors"
              >
                Search
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
