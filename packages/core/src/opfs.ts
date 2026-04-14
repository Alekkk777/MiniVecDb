/**
 * Progressive-enhancement persistence via OPFS (Origin Private File System).
 *
 * Fallback chain:
 *   1. OPFS (Chrome 86+, Firefox 111+, Safari 15.2+) — binary blob on disk
 *   2. localStorage — JSON-encoded base64 (≤ 5 MB, for small databases)
 *   3. In-memory only — no persistence
 */

const OPFS_FILENAME_PREFIX = 'microvecdb_';
const LS_KEY_PREFIX = 'microvecdb:';

function lsKey(key: string) { return LS_KEY_PREFIX + key; }

/** Returns true if the OPFS API is available. */
export function isOpfsAvailable(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    typeof navigator.storage !== 'undefined' &&
    typeof navigator.storage.getDirectory === 'function'
  );
}

/** Save `data` bytes under `key`. Uses OPFS → localStorage → silent skip. */
export async function save(key: string, data: Uint8Array): Promise<void> {
  if (isOpfsAvailable()) {
    try {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle(OPFS_FILENAME_PREFIX + key, { create: true });
      const writable = await fh.createWritable();
      // Copy to a plain ArrayBuffer to satisfy the FileSystemWritableFileStream type
      const plain = new ArrayBuffer(data.byteLength);
      new Uint8Array(plain).set(data);
      await writable.write(plain);
      await writable.close();
      return;
    } catch {
      // fall through to localStorage
    }
  }

  // localStorage fallback (base64-encode the binary)
  if (typeof localStorage !== 'undefined') {
    try {
      const b64 = btoa(String.fromCharCode(...data));
      localStorage.setItem(lsKey(key), b64);
    } catch {
      // quota exceeded or SSR environment — silent skip
    }
  }
}

/** Load bytes previously saved under `key`. Returns null if not found. */
export async function load(key: string): Promise<Uint8Array | null> {
  if (isOpfsAvailable()) {
    try {
      const root = await navigator.storage.getDirectory();
      const fh = await root.getFileHandle(OPFS_FILENAME_PREFIX + key);
      const file = await fh.getFile();
      const buf = await file.arrayBuffer();
      return new Uint8Array(buf);
    } catch {
      // fall through
    }
  }

  if (typeof localStorage !== 'undefined') {
    const b64 = localStorage.getItem(lsKey(key));
    if (b64) {
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      return bytes;
    }
  }

  return null;
}

/** Remove persisted data for `key`. */
export async function remove(key: string): Promise<void> {
  if (isOpfsAvailable()) {
    try {
      const root = await navigator.storage.getDirectory();
      await root.removeEntry(OPFS_FILENAME_PREFIX + key);
    } catch {/* not found */}
  }
  if (typeof localStorage !== 'undefined') {
    localStorage.removeItem(lsKey(key));
  }
}
