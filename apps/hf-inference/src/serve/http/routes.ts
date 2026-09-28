import type { PlaygroundConfig } from '../config.ts'
import { type InferenceModels } from '../../inference/models.ts'
import { generate_text, create_text_generation } from '../../inference/text.ts'
import { transcribe_audio } from '../../inference/speech.ts'
import { classify_audio } from '../../inference/audio.ts'
import { classify_image } from '../../inference/image.ts'
import { download_image } from './image-url.ts'
import { load_assets } from './static.ts'

const json = (value: unknown, status = 200) => ({ status, json: value })

/** HTTP validation and dispatch. Inference modules do not depend on HTTP. */
export function create_handler(
  config: PlaygroundConfig,
  models: InferenceModels,
) {
  const { port, origin, device } = config
  const assets = load_assets()
  let busy = false
  let active: {id: string; generation: ReturnType<typeof create_text_generation>} | undefined
  let expiry: ReturnType<typeof setTimeout> | undefined
  function release() {
    if (expiry !== undefined) clearTimeout(expiry)
    active?.generation.close()
    active = undefined
  }
  function touch() {
    if (expiry !== undefined) clearTimeout(expiry)
    expiry = setTimeout(release, 90000)
  }
  return async (request: HttpServerRequest): Promise<HttpServerResponse> => {
    const [path, query = ''] = request.url.split('?')
    if (request.headers.host !== `127.0.0.1:${port}`)
      return json({ error: 'Invalid host' }, 403)
    if (request.method === 'GET' && assets.has(path)) return assets.get(path)!
    if (request.method === 'GET' && path === '/api/health')
      return json({
        ready: true,
        busy: busy || Boolean(active),
        error: null,
        model: models.texts[models.default_text_model].id,
        default_text_model: models.default_text_model,
        text_models: Object.entries(models.texts).map(([key, text]) => ({key, id: text.id, label: text.label, chat: text.chat})),
        models: [
          ...Object.values(models.texts).map(text => text.id),
          ...Object.values(models.images).map((image) => image.id),
        ],
        speech_model: models.speech
          ? {
              id: models.speech.id,
              backend: 'onnx',
              max_new_tokens: models.speech.model.max_new_tokens,
            }
          : null,
        audio_model: models.audio
          ? { id: models.audio.id, backend: 'onnx' }
          : null,
        image_models: Object.entries(models.images).map(([key, image]) => ({
          key,
          id: image.id,
          label: image.label,
          backend: image.backend,
        })),
        device,
        server: 'affon-native',
      })
    if (
      request.method !== 'POST' ||
      ![
        '/api/generate',
        '/api/generate/start',
        '/api/generate/next',
        '/api/generate/cancel',
        '/api/classify',
        '/api/image-url',
        '/api/classify-audio',
        '/api/transcribe',
      ].includes(path)
    )
      return json({ error: 'Not found' }, 404)
    const source = request.headers.origin
    if (source && source !== origin)
      return json({ error: 'Invalid origin' }, 403)
    if (path === '/api/image-url') {
      if (!request.headers['content-type']?.startsWith('application/json'))
        return json({ error: 'Send application/json' }, 415)
      try {
        return await download_image(request.json<{ url?: unknown }>()?.url)
      } catch (error) {
        return json({ error: String(error) }, 400)
      }
    }
    if (path === '/api/generate/next' || path === '/api/generate/cancel') {
      if (!request.headers['content-type']?.startsWith('application/json'))
        return json({error: 'Send application/json'}, 415)
      let id: unknown
      try { id = request.json<{id?: unknown}>()?.id } catch { return json({error:'Invalid JSON'}, 400) }
      if (!active || id !== active.id) return json({error:'Generation expired or unavailable'}, 404)
      if (busy) return json({error:'A generation step is already running'}, 429)
      if (path.endsWith('/cancel')) { release(); return json({cancelled:true}) }
      busy = true
      try {
        const result = active.generation.next()
        if (result.done) release()
        else touch()
        return json(result)
      } catch (error) {
        release()
        return json({error:String(error)}, 500)
      } finally { busy = false }
    }
    if (busy || active)
      return json(
        {
          error: 'The model is busy. Try again after this completion finishes.',
        },
        429,
      )
    busy = true
    try {
      // Let already-arriving requests observe the busy state before synchronous compute.
      await new Promise<void>((resolve) => setTimeout(resolve, 1))
      if (path === '/api/classify-audio' || path === '/api/transcribe') {
        if (request.headers['content-type'] !== 'audio/wav')
          return json({ error: 'Send WAV bytes as audio/wav' }, 415)
        return json(
          path === '/api/transcribe'
            ? transcribe_audio(models, request.bytes(), device)
            : classify_audio(models, request.bytes(), device),
        )
      }
      if (path === '/api/classify') {
        if (request.headers['content-type'] !== 'application/octet-stream')
          return json(
            { error: 'Send packed RGB8 pixels as application/octet-stream' },
            415,
          )
        const params: Record<string, string> = Object.create(null)
        for (const part of query.split('&')) {
          const [key, value = ''] = part.split('=')
          if (params[key] !== undefined)
            return json({ error: 'Duplicate image dimension' }, 400)
          params[key] = decodeURIComponent(value)
        }
        const width = Number(params.width),
          height = Number(params.height)
        if (
          ![width, height].every(
            (n) => Number.isInteger(n) && n > 0 && n <= 512,
          )
        )
          return json(
            { error: 'Image dimensions must be 1–512 pixels per side.' },
            400,
          )
        const pixels = request.bytes()
        if (pixels.length !== width * height * 3)
          return json(
            { error: 'RGB byte count does not match dimensions.' },
            400,
          )
        return json(
          classify_image(models, pixels, width, height, device, params.model),
        )
      }
      if (!request.headers['content-type']?.startsWith('application/json'))
        return json({ error: 'Send application/json' }, 415)
      const input = request.json<{
        model?: unknown
        prompt?: unknown
        max_new_tokens?: unknown
      } | null>()
      const budget = input?.max_new_tokens ?? 24
      if (
        (input?.model !== undefined && typeof input.model !== 'string') ||
        typeof input?.prompt !== 'string' ||
        !input.prompt.trim() ||
        input.prompt.length > 4096 ||
        !Number.isInteger(budget) ||
        Number(budget) < 1 ||
        Number(budget) > 64
      )
        return json(
          {
            error:
              'Provide a nonempty prompt (up to 4096 characters) and max_new_tokens from 1 to 64.',
          },
          400,
        )
      if (path === '/api/generate/start') {
        const generation = create_text_generation(models, input.prompt, Number(budget), input.model as string | undefined)
        const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`
        active = {id, generation}
        touch()
        return json({id})
      }
      return json(generate_text(models, input.prompt, Number(budget), input.model as string | undefined))
    } catch (error) {
      return json({ error: String(error) }, 400)
    } finally {
      busy = false
    }
  }
}
