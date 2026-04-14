/**
 * Standalone SharedWorker for the index.html demo.
 *
 * This file is intentionally plain JS (not TypeScript) so it can be served
 * directly by `python -m http.server 8080` without a build step.
 *
 * It mirrors the logic in packages/microvecdb/src/worker.ts.
 */

import init, { WasmVecDb } from './pkg/microvecdb_wasm.js';

// ── OPFS helpers (inline, no import from packages/) ──────────────────────────
const OPFS_PREFIX = 'microvecdb_';
const LS_PREFIX   = 'microvecdb:';

async function opfsSave(key, data) {
  if (typeof navigator?.storage?.getDirectory === 'function') {
    try {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle(OPFS_PREFIX + key, { create: true });
      const w = await fh.createWritable();
      const buf = new ArrayBuffer(data.byteLength);
      new Uint8Array(buf).set(data);
      await w.write(buf);
      await w.close();
      return;
    } catch {}
  }
  if (typeof localStorage !== 'undefined') {
    try { localStorage.setItem(LS_PREFIX + key, btoa(String.fromCharCode(...data))); } catch {}
  }
}

async function opfsLoad(key) {
  if (typeof navigator?.storage?.getDirectory === 'function') {
    try {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle(OPFS_PREFIX + key);
      return new Uint8Array(await (await fh.getFile()).arrayBuffer());
    } catch {}
  }
  if (typeof localStorage !== 'undefined') {
    const b64 = localStorage.getItem(LS_PREFIX + key);
    if (b64) {
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
  }
  return null;
}

async function opfsRemove(key) {
  if (typeof navigator?.storage?.getDirectory === 'function') {
    try { const r = await navigator.storage.getDirectory(); await r.removeEntry(OPFS_PREFIX + key); } catch {}
  }
  if (typeof localStorage !== 'undefined') localStorage.removeItem(LS_PREFIX + key);
}

// ── State ────────────────────────────────────────────────────────────────────

let db = null;
let persistenceKey = null;
let saveTimer = null;

// ── WASM bootstrap ───────────────────────────────────────────────────────────

const pendingPorts = [];
let wasmReady = false;

init().then(() => {
  wasmReady = true;
  for (const p of pendingPorts) attachPort(p);
  pendingPorts.length = 0;
});

// ── Connection handler ───────────────────────────────────────────────────────

self.onconnect = (event) => {
  const port = event.ports[0];
  if (wasmReady) attachPort(port);
  else pendingPorts.push(port);
};

function attachPort(port) {
  port.onmessage = async (e) => {
    const req = e.data;
    try {
      const payload = await handleRequest(req);
      port.postMessage({ type: 'ok', reqId: req.reqId, payload });
    } catch (err) {
      port.postMessage({ type: 'error', reqId: req.reqId, message: String(err?.message ?? err) });
    }
  };
  port.start();
}

// ── Dispatcher ───────────────────────────────────────────────────────────────

async function handleRequest(req) {
  switch (req.type) {
    case 'init': {
      if (db !== null) return stats();
      persistenceKey = req.persistenceKey ?? null;
      if (persistenceKey) {
        const saved = await opfsLoad(persistenceKey);
        db = saved ? tryDeserialize(saved, req.capacity) : WasmVecDb.with_capacity(req.capacity);
      } else {
        db = WasmVecDb.with_capacity(req.capacity);
      }
      return stats();
    }
    case 'insert': {
      ensureDb();
      const slot = db.insert(req.id, req.vector);
      if (slot === 0xFFFFFFFF) throw new RangeError(`insert failed for id=${req.id}`);
      scheduleSave();
      return slot;
    }
    case 'insert_batch': {
      ensureDb();
      const n = db.insert_batch(req.ids, req.flat);
      if (n > 0) scheduleSave();
      return n;
    }
    case 'search_scan': {
      ensureDb();
      return db.search_scan(req.query, req.k);
    }
    case 'search_hnsw': {
      ensureDb();
      return db.has_index() ? db.search_hnsw(req.query, req.k, req.ef) : db.search_scan(req.query, req.k);
    }
    case 'build_index': {
      ensureDb();
      db.build_index(req.m, req.efConstruction);
      return null;
    }
    case 'delete': {
      ensureDb();
      const ok = db.delete(req.id);
      if (ok) scheduleSave();
      return ok;
    }
    case 'compact': { ensureDb(); return db.compact(); }
    case 'save':    { ensureDb(); await flushSave(); return null; }
    case 'clear_save': { if (persistenceKey) await opfsRemove(persistenceKey); return null; }
    case 'serialize': { ensureDb(); return db.serialize(); }
    case 'stats':   { ensureDb(); return stats(); }
    case 'dispose': {
      if (saveTimer !== null) { clearTimeout(saveTimer); saveTimer = null; }
      if (db) { db.free(); db = null; }
      return null;
    }
    default: throw new Error(`Unknown request type: ${req.type}`);
  }
}

function ensureDb() {
  if (!db) throw new Error('DB not initialised — call init() first');
}

function stats() {
  return { count: db.len(), hasIndex: db.has_index() };
}

function tryDeserialize(saved, capacity) {
  try { return WasmVecDb.deserialize(saved); }
  catch { return WasmVecDb.with_capacity(capacity); }
}

async function flushSave() {
  if (!persistenceKey || !db) return;
  await opfsSave(persistenceKey, db.serialize());
}

function scheduleSave() {
  if (!persistenceKey) return;
  if (saveTimer !== null) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => { flushSave().catch(() => {}); saveTimer = null; }, 500);
}
