#!/usr/bin/env node
/**
 * generate-sri.mjs — Compute SRI (Subresource Integrity) SHA-384 hashes
 * for MicroVecDB distribution files.
 *
 * Usage: node scripts/generate-sri.mjs
 * Output: dist/sri-hashes.json
 *
 * Uses only Node.js built-in modules — no external dependencies.
 */

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { resolve, join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PKG_ROOT = resolve(__dirname, '..');
const DIST_DIR = join(PKG_ROOT, 'dist');
const WASM_DIR = resolve(PKG_ROOT, '..', '..', 'pkg');

/**
 * Compute SHA-384 hash of a file and return the SRI hash string.
 * @param {string} filePath - Absolute path to the file
 * @returns {string} SRI hash in "sha384-<base64>" format
 */
function computeSriHash(filePath) {
  const content = readFileSync(filePath);
  const hash = createHash('sha384').update(content).digest('base64');
  return `sha384-${hash}`;
}

/**
 * Collect files from a directory matching given extensions (non-recursive).
 * @param {string} dir
 * @param {string[]} extensions
 */
function collectFiles(dir, extensions) {
  if (!existsSync(dir)) {
    console.error(`[generate-sri] Directory not found: ${dir}`);
    console.error('[generate-sri] Run "npm run build" first.');
    process.exit(1);
  }
  return readdirSync(dir)
    .filter(name => extensions.some(ext => name.endsWith(ext)))
    .map(name => join(dir, name))
    .filter(fp => statSync(fp).isFile());
}

const distFiles = collectFiles(DIST_DIR, ['.js', '.cjs']);
const wasmFiles = collectFiles(WASM_DIR, ['.wasm']);
const allFiles = [...distFiles, ...wasmFiles];

if (allFiles.length === 0) {
  console.error('[generate-sri] No files found. Run "npm run build" first.');
  process.exit(1);
}

const hashes = {};
const rows = [];

for (const filePath of allFiles) {
  const sriHash = computeSriHash(filePath);
  const key = relative(PKG_ROOT, filePath);
  hashes[key] = sriHash;
  rows.push({ file: key, hash: sriHash });
}

const output = { generated: new Date().toISOString(), hashes };
const outputPath = join(DIST_DIR, 'sri-hashes.json');
writeFileSync(outputPath, JSON.stringify(output, null, 2), 'utf8');

console.log('[generate-sri] SRI hashes generated:\n');
for (const { file, hash } of rows) {
  console.log(`  ${file}`);
  console.log(`    integrity="${hash}"\n`);
}
console.log(`[generate-sri] Written to: ${relative(process.cwd(), outputPath)}`);
