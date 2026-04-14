# Security Headers for MicroVecDB

## Why these headers are required

MicroVecDB uses `SharedArrayBuffer` to share memory between tabs via a `SharedWorker`.
Modern browsers (Chrome 92+, Firefox 79+, Safari 15.2+) require the page to be
**cross-origin isolated** before granting access to `SharedArrayBuffer`, as a mitigation
against Spectre-class side-channel attacks.

Cross-origin isolation is activated by serving **both** of the following HTTP headers:

| Header | Required value |
|--------|---------------|
| `Cross-Origin-Opener-Policy` | `same-origin` |
| `Cross-Origin-Embedder-Policy` | `require-corp` |

MicroVecDB works without these headers, but `SharedArrayBuffer` will be unavailable
and the `SharedMicroVecDB` (multi-tab) mode will be disabled by the browser.

---

## Configuration examples

### Nginx

```nginx
add_header Cross-Origin-Opener-Policy  "same-origin" always;
add_header Cross-Origin-Embedder-Policy "require-corp" always;
```

### Apache (`.htaccess`)

```apache
Header always set Cross-Origin-Opener-Policy "same-origin"
Header always set Cross-Origin-Embedder-Policy "require-corp"
```

### Vite (`vite.config.ts`)

```typescript
import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
```

### Next.js (`next.config.js`)

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'Cross-Origin-Opener-Policy',  value: 'same-origin' },
          { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
        ],
      },
    ];
  },
};

module.exports = nextConfig;
```

### Express

```javascript
app.use((_req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  next();
});
```

### Python `http.server` (local development)

The default `python3 -m http.server` does **not** send these headers.
Use this drop-in replacement:

```python
from http.server import HTTPServer, SimpleHTTPRequestHandler

class IsolatedHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

if __name__ == '__main__':
    print('Serving on http://localhost:8080 (cross-origin isolated)')
    HTTPServer(('', 8080), IsolatedHandler).serve_forever()
```

Save as `serve.py` and run `python3 serve.py`.

---

## Verifying the configuration

Open DevTools in the browser and run:

```javascript
console.log(self.crossOriginIsolated); // must be true
```

Or check **DevTools → Application → Security** — the page should be marked
as "cross-origin isolated".

The MicroVecDB demo page (`index.html`) also shows a yellow warning banner
when the headers are missing.
