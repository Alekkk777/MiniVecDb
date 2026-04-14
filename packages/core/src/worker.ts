/**
 * MicroVecDB SharedWorker host.
 *
 * One instance of this worker is shared across all tabs of the same origin.
 * Each tab connects via a MessagePort; they all share the same WasmVecDb
 * instance and the same OPFS-persisted database.
 *
 * Usage (from the main thread):
 *   const worker = new SharedWorker(new URL('./worker.js', import.meta.url), { type: 'module' });
 *   // Then use SharedMicroVecDB, which wraps the worker automatically.
 *
 * The worker supports exactly one logical database at a time.  The first
 * `init` call from any port wins; subsequent `init` calls from other ports
 * join the existing database (the options of the first caller are used).
 */

/// <reference lib="webworker" />

// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const self: any; // SharedWorkerGlobalScope (not in TS DOM lib by default)

import init, { WasmVecDb } from '../../../pkg/microvecdb_wasm.js';
import * as opfs from './opfs.js';
import type { WorkerRequest, WorkerResponse, DbStatsPayload } from './worker-protocol.js';

// ── State ─────────────────────────────────────────────────────────────────────

let db: WasmVecDb | null = null;
let persistenceKey: string | null = null;
let saveTimer: ReturnType<typeof setTimeout> | null = null;
let wasmReady = false;

// Queue of ports that connected before WASM was initialised.
const pendingPorts: MessagePort[] = [];

// ── WASM bootstrap (once, regardless of how many ports connect) ───────────────

const wasmReadyPromise = init().then(() => {
  wasmReady = true;
  for (const p of pendingPorts) attachPort(p);
  pendingPorts.length = 0;
});

// ── SharedWorker connection handler ───────────────────────────────────────────

self.onconnect = (event: MessageEvent) => {
  const port = (event as MessageEvent & { ports: MessagePort[] }).ports[0];
  if (wasmReady) {
    attachPort(port);
  } else {
    pendingPorts.push(port);
  }
};

// ── Per-port message handling ─────────────────────────────────────────────────

function attachPort(port: MessagePort) {
  port.onmessage = async (e: MessageEvent<WorkerRequest>) => {
    const req = e.data;
    try {
      const payload = await handleRequest(req);
      const res: WorkerResponse = { type: 'ok', reqId: req.reqId, payload };
      port.postMessage(res);
    } catch (err) {
      const res: WorkerResponse = {
        type: 'error',
        reqId: req.reqId,
        message: err instanceof Error ? err.message : String(err),
      };
      port.postMessage(res);
    }
  };
  port.start();
}

// ── Request dispatcher ────────────────────────────────────────────────────────

async function handleRequest(req: WorkerRequest): Promise<unknown> {
  switch (req.type) {

    case 'init': {
      if (db !== null) {
        // Already initialised — return current stats so the caller can
        // confirm the shared DB is alive.
        return statsPayload();
      }
      persistenceKey = req.persistenceKey;

      if (persistenceKey) {
        const saved = await opfs.load(persistenceKey);
        if (saved) {
          try {
            db = WasmVecDb.deserialize(saved);
          } catch {
            db = WasmVecDb.with_capacity(req.capacity);
          }
        } else {
          db = WasmVecDb.with_capacity(req.capacity);
        }
      } else {
        db = WasmVecDb.with_capacity(req.capacity);
      }

      return statsPayload();
    }

    case 'insert': {
      ensureDb();
      if (!(req.vector instanceof Float32Array) || req.vector.length !== 384 || !req.vector.every(Number.isFinite)) {
        throw new TypeError(`[worker] insert: invalid vector for id=${req.id}`);
      }
      const slot = db!.insert(req.id, req.vector);
      if (slot === 0xFFFFFFFF) throw new RangeError(`insert failed for id=${req.id}`);
      scheduleSave();
      return slot;
    }

    case 'insert_batch': {
      ensureDb();
      if (!(req.flat instanceof Float32Array) || req.flat.length % 384 !== 0) {
        throw new RangeError(`[worker] insert_batch: flat.length (${req.flat.length}) must be a multiple of 384`);
      }
      const count = db!.insert_batch(req.ids, req.flat);
      if (count > 0) scheduleSave();
      return count;
    }

    case 'search_scan': {
      ensureDb();
      return db!.search_scan(req.query, req.k);
    }

    case 'search_hnsw': {
      ensureDb();
      // Graceful fallback: if no index has been built yet, use brute-force scan.
      if (!db!.has_index()) return db!.search_scan(req.query, req.k);
      return db!.search_hnsw(req.query, req.k, req.ef);
    }

    case 'build_index': {
      ensureDb();
      db!.build_index(req.m, req.efConstruction);
      return null;
    }

    case 'delete': {
      ensureDb();
      const ok = db!.delete(req.id);
      if (ok) scheduleSave();
      return ok;
    }

    case 'compact': {
      ensureDb();
      return db!.compact();
    }

    case 'save': {
      ensureDb();
      await flushSave();
      return null;
    }

    case 'clear_save': {
      if (persistenceKey) await opfs.remove(persistenceKey);
      return null;
    }

    case 'serialize': {
      ensureDb();
      return db!.serialize();
    }

    case 'stats': {
      ensureDb();
      return statsPayload();
    }

    case 'dispose': {
      if (saveTimer !== null) { clearTimeout(saveTimer); saveTimer = null; }
      if (db) { db.free(); db = null; }
      return null;
    }

    default: {
      const _exhaustive: never = req;
      throw new Error(`Unknown request type: ${(_exhaustive as WorkerRequest).type}`);
    }
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function ensureDb() {
  if (!db) throw new Error('DB not initialised — call init() first');
}

function statsPayload(): DbStatsPayload {
  return { count: db!.len(), hasIndex: db!.has_index() };
}

async function flushSave() {
  if (!persistenceKey || !db) return;
  const bytes = db.serialize();
  await opfs.save(persistenceKey, bytes);
}

function scheduleSave() {
  if (!persistenceKey) return;
  if (saveTimer !== null) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    flushSave().catch(() => {});
    saveTimer = null;
  }, 500);
}
