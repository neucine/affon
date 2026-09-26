import http from 'std:http'
const limit = 20 * 1024 * 1024

function parse_url(input: string) {
  const match = /^(https?):\/\/([^/?#]+)([^#]*)/i.exec(input)
  if (!match || /[\x00-\x20\x7f\\]/.test(input) || match[2].includes('@')) {
    throw Error('Use an HTTP or HTTPS URL without embedded credentials.')
  }
  return {
    origin: `${match[1].toLowerCase()}://${match[2]}`,
    path: match[3] || '/',
  }
}
function resolve_redirect(location: string, base: string) {
  if (/^[A-Za-z][A-Za-z0-9+.-]*:/.test(location)) return location
  const url = parse_url(base)
  if (location.startsWith('//'))
    return base.slice(0, base.indexOf(':') + 1) + location
  if (location.startsWith('/')) return url.origin + location
  const path = url.path.split('?')[0]
  if (location.startsWith('?')) return url.origin + path + location
  if (location.startsWith('#')) return base.split('#')[0] + location
  return url.origin + path.slice(0, path.lastIndexOf('/') + 1) + location
}
function header(headers: Record<string, string | string[]>, name: string) {
  const key = Object.keys(headers).find((key) => key.toLowerCase() === name)
  const value = key ? headers[key] : undefined
  return Array.isArray(value) ? value[0] : value
}

let active = false
/** Native, bounded download for browser decoding. Only one download at a time. */
export async function download_image(input: unknown) {
  if (typeof input !== 'string' || !input || input.length > 8192)
    throw Error('Enter a valid HTTP or HTTPS image URL.')
  if (active)
    throw Error('An image download is already in progress. Try again shortly.')
  parse_url(input)
  active = true
  let timer: ReturnType<typeof setTimeout> | undefined
  let expired = false
  const download = async () => {
    let url = input as string
    for (let redirects = 0; redirects <= 3; redirects++) {
      parse_url(url)
      const response = await http.get(url, {
        maxBytes: limit,
        redirect: 'manual',
        headers: { accept: 'image/png,image/jpeg,image/webp,image/avif' },
      })
      if (expired) throw Error('Image download expired.')
      if ([301, 302, 303, 307, 308].includes(response.status)) {
        const location = header(response.headers, 'location')
        if (!location || redirects === 3)
          throw Error(
            'The image URL has too many redirects or a missing redirect target.',
          )
        url = resolve_redirect(location, url)
        continue
      }
      const type = header(response.headers, 'content-type')
        ?.split(';')[0]
        .trim()
        .toLowerCase()
      if (!response.ok)
        throw Error(`Image download failed (HTTP ${response.status}).`)
      if (
        !['image/png', 'image/jpeg', 'image/webp', 'image/avif'].includes(
          type ?? '',
        )
      )
        throw Error(
          'That URL did not return a PNG, JPEG, WebP, or AVIF image. Use a direct image link.',
        )
      const body = response.bytes()
      if (!body.length) throw Error('The image response was empty.')
      return {
        body,
        headers: {
          'content-type': type!,
          'cache-control': 'no-store',
          'x-content-type-options': 'nosniff',
        },
      }
    }
    throw Error('Could not download image.')
  }
  // Hao cannot yet cancel an in-flight HTTP client request. Keep the slot occupied
  // until native I/O settles, even if the browser has already received a timeout.
  const pending = download().finally(() => {
    active = false
  })
  try {
    return await Promise.race([
      pending,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          expired = true
          reject(
            Error(
              'Image download exceeded 30 seconds. Native I/O may still be finishing.',
            ),
          )
        }, 30000)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}
