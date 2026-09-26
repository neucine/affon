import fs from 'std:fs'

/** Explicit asset allowlist: never resolve a request path against the filesystem. */
export function load_assets() {
  const root = 'apps/hf-inference/public'
  return new Map<string, HttpServerResponse>([
    [
      '/transcription.js',
      {
        body: fs.readFileSync(`${root}/transcription.js`),
        headers: { 'content-type': 'text/javascript; charset=utf-8' },
      },
    ],
    [
      '/',
      {
        body: fs.readFileSync(`${root}/index.html`),
        headers: { 'content-type': 'text/html; charset=utf-8' },
      },
    ],
    [
      '/app.js',
      {
        body: fs.readFileSync(`${root}/app.js`),
        headers: { 'content-type': 'text/javascript; charset=utf-8' },
      },
    ],
    [
      '/styles.css',
      {
        body: fs.readFileSync(`${root}/styles.css`),
        headers: { 'content-type': 'text/css; charset=utf-8' },
      },
    ],
  ])
}
