# Contributing to MicroVecDB

Thank you for your interest in contributing. This document explains how to set up the project locally, run the tests, and open a pull request.

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Node.js | ≥ 18 | https://nodejs.org |
| Rust | stable | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| wasm-pack | latest | `cargo install wasm-pack` |
| Python | ≥ 3.9 | https://python.org *(for Python package only)* |
| Maturin | ≥ 1.0 | `pip install maturin` *(for Python package only)* |

Add the WASM target:

```bash
rustup target add wasm32-unknown-unknown
```

---

## Setup

```bash
git clone https://github.com/Alekkk777/MiniVecDb.git
cd MiniVecDb
npm install
```

---

## Build

```bash
# Build WASM binary + TypeScript wrapper
npm run build --workspace=packages/core

# Build + regenerate SRI hashes
npm run build:full --workspace=packages/core

# Build Python native extension (editable install)
cd crates/microvecdb-python
maturin develop --release
```

---

## Tests

```bash
# TypeScript — run all workspace tests
npm test --workspaces --if-present

# TypeScript — watch mode (packages/core only)
npm run test:watch --workspace=packages/core

# Python — unit tests (no native extension needed)
cd crates/microvecdb-python
pip install pytest freezegun langchain-core
pytest tests/ -v -m "not integration"

# Python — full suite including native extension (requires maturin develop first)
pytest tests/ -v
```

All tests must pass before opening a PR. The CI will run them automatically on every push.

---

## Running the examples locally

```bash
npm run dev --workspace=examples/pdf-brain     # http://localhost:5173
npm run dev --workspace=examples/visual-search  # http://localhost:5174
```

---

## Project structure

```
crates/
  microvecdb-core/        Rust library (quantisation, storage, HNSW, time)
  microvecdb-wasm/        wasm-bindgen bindings → browser / Node.js
  microvecdb-python/      PyO3 native extension → minivecdb PyPI package
    python/minivecdb/
      __init__.py         re-exports MiniVecDb
      langchain.py        LangChain VectorStore adapter with TTL GC
    tests/                pytest suite (38 unit + 5 integration)
packages/
  core/                   @microvecdb/core npm package
    src/
      MicroVecDB.ts       WASM wrapper
      SharedMicroVecDB.ts Web Worker proxy
      langchain.ts        LangChain JS adapter
      vercel.ts           Vercel AI SDK adapter with TTL GC
examples/
  pdf-brain/              Local RAG demo (React + Transformers.js)
  visual-search/          Image similarity demo (React + pHash)
```

---

## Pull request checklist

Before opening a PR, please verify:

- [ ] `npm test --workspaces --if-present` passes locally
- [ ] `npm run build --workspace=packages/core` completes without errors
- [ ] Your change includes a test or, if not testable, a clear explanation why
- [ ] You have not added runtime dependencies to `packages/core` (it must stay at 0 deps)
- [ ] Commit messages are in the imperative mood ("Add feature", not "Added feature")

---

## What we accept

| Type | Notes |
|---|---|
| Bug fixes | Always welcome — please include a reproduction |
| Performance improvements | Include a benchmark showing the before/after |
| New features | Open a Discussion first to align on the design |
| Documentation | Always welcome |
| New examples | Place them in `examples/` with their own `package.json` |

## What we don't accept

- Increasing the WASM binary beyond 100 KB without a strong justification
- Adding runtime npm dependencies to `packages/core`
- Breaking changes to the public API without a deprecation path

---

## Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add batch delete method
fix: prevent panic on empty search with no index
perf: inline hamming distance in hot path
docs: add SharedMicroVecDB usage example
```

---

## License

By contributing you agree that your contributions will be licensed under the MIT License.
