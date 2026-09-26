// Run with Affon from the repository root.
import http from 'std:http'
import { download_image } from '../../src/serve/http/image-url.ts'
function assert(value: unknown, message: string) {
  if (!value) throw Error(message)
}
const source = http.serve({
  port: 0,
  maxResponseBytes: 22 * 1024 * 1024,
  handler(request): HttpServerResponse {
    const path = request.url
    if (path === '/redirect')
      return { status: 302, headers: { location: '/image' } }
    if (path === '/relative')
      return { status: 302, headers: { location: 'image' } }
    if (path === '/loop') return { status: 302, headers: { location: '/loop' } }
    if (path === '/credentials')
      return {
        status: 302,
        headers: { location: 'https://user:pass@example.com/image' },
      }
    if (path === '/scheme')
      return { status: 302, headers: { location: 'file:///tmp/image' } }
    if (path === '/large')
      return {
        body: new Uint8Array(21 * 1024 * 1024),
        headers: { 'content-type': 'image/png' },
      }
    if (path === '/html')
      return {
        body: '<html>Not an image</html>',
        headers: { 'content-type': 'text/html' },
      }
    if (path === '/empty')
      return {
        body: new Uint8Array(0),
        headers: { 'content-type': 'image/png' },
      }
    return {
      body: new Uint8Array([137, 80, 78, 71]),
      headers: { 'Content-Type': 'image/png' },
    }
  },
})
try {
  const base = `http://127.0.0.1:${source.port}`
  for (const path of ['/image', '/redirect', '/relative']) {
    const response = await download_image(base + path)
    assert(
      JSON.stringify(Array.from(response.body)) === '[137,80,78,71]',
      'Unexpected image bytes',
    )
    assert(
      response.headers['content-type'] === 'image/png',
      'Unexpected image type',
    )
  }
  for (const url of [
    'file:///tmp/photo.png',
    'https://user:pass@example.com/image',
    `${base}/html`,
    `${base}/loop`,
    `${base}/large`,
    `${base}/empty`,
    `${base}/credentials`,
    `${base}/scheme`,
  ]) {
    let rejected = false
    try {
      await download_image(url)
    } catch {
      rejected = true
    }
    assert(rejected, `Expected rejection: ${url}`)
  }
  console.log(
    'Native image URL checks passed: bytes, redirects, schemes, credentials, content type, size limit, and empty responses.',
  )
} finally {
  source.close()
}
