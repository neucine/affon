const $ = (id) => document.getElementById(id)
let running = false,
  decoding = false,
  ready = false,
  busy = false,
  selected = null
let audioFile = null,
  audioURL = null,
  audioAvailable = false
const transcription = setupTranscription({
  state: () => ({ running, decoding, ready, busy }),
  setRunning: (value) => {
    running = value
    controls()
  },
  refresh: () => health(),
})
function controls() {
  transcription.controls()
  $('audio-file').disabled = running || decoding
  $('classify-audio').disabled =
    running || decoding || !ready || busy || !audioAvailable || !audioFile
  $('generate').disabled = running || !ready || busy
  $('classify').disabled =
    running ||
    decoding ||
    !ready ||
    busy ||
    !selected ||
    selected.source !== 'upload'
  $('image-model').disabled = running || decoding || !ready || busy
  $('image').disabled = running || decoding
  $('load-url').disabled = running || decoding || !ready || busy
  $('image-url').disabled = running || decoding
}
for (const tab of ['image', 'text', 'audio', 'speech'])
  $(tab + '-tab').addEventListener('click', () => {
    for (const name of ['image', 'text', 'audio', 'speech']) {
      $(name + '-panel').hidden = name !== tab
      $(name + '-tab').setAttribute('aria-selected', String(name === tab))
    }
  })
async function health() {
  try {
    const h = await (await fetch('/api/health')).json()
    ready = h.ready
    busy = h.busy
    transcription.health(h.speech_model)
    audioAvailable = Boolean(h.audio_model)
    $('audio-model-info').textContent = audioAvailable
      ? `${h.audio_model.id} · ONNX graph`
      : 'Audio model is not configured on this server.'
    if (
      h.image_models &&
      JSON.stringify(h.image_models) !== $('image-model').dataset.catalog
    ) {
      const previous = $('image-model').value
      $('image-model').replaceChildren(
        ...h.image_models.map((model) => {
          const option = document.createElement('option')
          option.value = model.key
          option.textContent = model.label
          return option
        }),
      )
      if (h.image_models.some((model) => model.key === previous))
        $('image-model').value = previous
      $('image-model').dataset.catalog = JSON.stringify(h.image_models)
    }
    $('device').textContent = h.device === 'metal' ? 'Metal GPU' : 'CPU'
    $('status').textContent =
      h.error ||
      (ready
        ? busy
          ? 'Running inference…'
          : 'Models ready'
        : 'Loading models…')
  } catch {
    ready = false
    $('status').textContent = 'Server unavailable'
  }
  controls()
}
async function loadImage(file, source = 'upload') {
  selected = null
  decoding = true
  controls()
  $('image-error').textContent = ''
  $('preview').hidden = true
  $('predictions').textContent =
    source === 'upload'
      ? 'Choose Classify uploaded image to run the model.'
      : 'Preparing classification…'
  $('image-stats').textContent = ''
  $('image-info').textContent = ''
  let bitmap
  try {
    if (!file) return
    if (file.size > 20 * 1024 * 1024)
      throw Error('Choose an image smaller than 20 MB.')
    bitmap = await createImageBitmap(file)
    const scale = Math.min(1, 512 / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(bitmap.width * scale))
    canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data
    const rgb = new Uint8Array(canvas.width * canvas.height * 3)
    for (let p = 0, q = 0; p < rgba.length; p += 4) {
      rgb[q++] = rgba[p]
      rgb[q++] = rgba[p + 1]
      rgb[q++] = rgba[p + 2]
    }
    selected = { width: canvas.width, height: canvas.height, rgb, source }
    $('preview').src = canvas.toDataURL('image/png')
    $('preview').hidden = false
    $('image-info').textContent =
      `${file.name} · ${bitmap.width} × ${bitmap.height} → ${canvas.width} × ${canvas.height}`
  } catch (e) {
    $('image-error').textContent = 'Could not open image: ' + e.message
    throw e
  } finally {
    bitmap?.close()
    decoding = false
    controls()
  }
}
$('image').addEventListener('change', () =>
  loadImage($('image').files[0]).catch(() => {}),
)
$('url-form').addEventListener('submit', async (e) => {
  e.preventDefault()
  selected = null
  $('image').value = ''
  decoding = true
  controls()
  $('image-error').textContent = ''
  $('preview').hidden = true
  $('image-info').textContent = 'Downloading image…'
  $('predictions').textContent = 'Waiting for image…'
  $('image-stats').textContent = ''
  try {
    const url = $('image-url').value.trim()
    const response = await fetch('/api/image-url', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ url }),
    })
    if (!response.ok) {
      const data = await response.json()
      throw Error(data.error || 'Download failed')
    }
    const blob = await response.blob()
    await loadImage(
      new File(
        [blob],
        new URL(url).pathname.split('/').pop() || 'Downloaded image',
        { type: blob.type },
      ),
      'url',
    )
    decoding = false
    controls()
    await classifySelected()
  } catch (error) {
    $('image-error').textContent = error.message
    $('image-info').textContent = ''
    $('predictions').textContent = 'No prediction available.'
  } finally {
    decoding = false
    controls()
  }
})
async function classifySelected() {
  if (!selected) return
  running = true
  controls()
  $('image-error').textContent = ''
  $('predictions').textContent = 'Classifying…'
  $('image-stats').textContent = ''
  try {
    const r = await fetch(
      `/api/classify?width=${selected.width}&height=${selected.height}&model=${encodeURIComponent($('image-model').value)}`,
      {
        method: 'POST',
        headers: { 'content-type': 'application/octet-stream' },
        body: selected.rgb,
      },
    )
    const data = await r.json()
    if (!r.ok) throw Error(data.error || 'Classification failed')
    $('predictions').replaceChildren()
    for (const item of data.predictions) {
      const row = document.createElement('div')
      row.className = 'prediction'
      const head = document.createElement('div')
      head.className = 'heading'
      const label = document.createElement('span')
      label.textContent = item.label
      const score = document.createElement('strong')
      score.textContent = (item.score * 100).toFixed(2) + '%'
      head.append(label, score)
      const bar = document.createElement('div')
      bar.className = 'bar'
      const fill = document.createElement('span')
      fill.style.width = item.score * 100 + '%'
      bar.append(fill)
      row.append(head, bar)
      $('predictions').append(row)
    }
    $('image-stats').textContent =
      `${data.label} · ${data.model} · ${data.backend === 'onnx' ? 'ONNX graph' : 'Native adapter'} · ${data.device === 'metal' ? 'Metal GPU' : 'CPU'} · ${data.inference_ms} ms inference · ${data.elapsed_ms} ms total`
  } catch (e) {
    $('image-error').textContent = e.message
    $('predictions').textContent = 'No prediction available.'
  } finally {
    running = false
    health()
  }
}
$('image-model').addEventListener('change', () => {
  $('predictions').textContent = 'Run classification with the selected model.'
  $('image-stats').textContent = ''
  $('image-error').textContent = ''
})
$('classify').addEventListener('click', () => {
  if (selected?.source === 'upload') classifySelected()
})
$('form').addEventListener('submit', async (e) => {
  e.preventDefault()
  running = true
  controls()
  $('error').textContent = ''
  $('status').textContent = 'Generating…'
  try {
    const r = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        prompt: $('prompt').value,
        max_new_tokens: Number($('tokens').value),
      }),
    })
    const data = await r.json()
    if (!r.ok) throw Error(data.error || 'Generation failed')
    $('output').textContent = data.text
    $('stats').textContent =
      `${data.prompt_tokens} prompt tokens · ${data.generated_tokens} generated tokens · ${(data.elapsed_ms / 1000).toFixed(2)} seconds`
  } catch (e) {
    $('error').textContent = e.message
  } finally {
    running = false
    health()
  }
})
health()
setInterval(health, 3000)

$('audio-file').addEventListener('change', () => {
  audioFile = null
  if (audioURL) URL.revokeObjectURL(audioURL)
  audioURL = null
  $('audio-preview').hidden = true
  $('audio-preview').removeAttribute('src')
  $('audio-preview').load()
  $('audio-error').textContent = ''
  $('audio-info').textContent = ''
  $('audio-stats').textContent = ''
  $('audio-predictions').textContent = 'Choose Classify audio to run the model.'
  const file = $('audio-file').files[0]
  if (file) {
    if (file.size > 16 * 1024 * 1024 || !file.size)
      $('audio-error').textContent =
        'Choose a nonempty WAV file no larger than 16 MiB.'
    else {
      audioFile = file
      audioURL = URL.createObjectURL(file)
      $('audio-preview').src = audioURL
      $('audio-preview').hidden = false
      $('audio-info').textContent =
        `${file.name} · ${(file.size / 1024).toFixed(1)} KiB`
    }
  }
  controls()
})
$('classify-audio').addEventListener('click', async () => {
  if (!audioFile || running) return
  running = true
  controls()
  $('audio-error').textContent = ''
  $('audio-stats').textContent = ''
  $('audio-predictions').textContent = 'Classifying…'
  try {
    const response = await fetch('/api/classify-audio', {
      method: 'POST',
      headers: { 'content-type': 'audio/wav' },
      body: audioFile,
    })
    const data = await response.json()
    if (!response.ok) throw Error(data.error || 'Audio classification failed')
    $('audio-predictions').replaceChildren()
    for (const item of data.predictions) {
      const row = document.createElement('div')
      row.className = 'prediction'
      const heading = document.createElement('div')
      heading.className = 'heading'
      const label = document.createElement('span')
      label.textContent = item.label
      const score = document.createElement('strong')
      score.textContent = `${(item.score * 100).toFixed(2)}%`
      heading.append(label, score)
      row.append(heading)
      $('audio-predictions').append(row)
    }
    $('audio-stats').textContent =
      `${data.model} · ONNX graph · ${data.device === 'metal' ? 'Metal GPU' : 'CPU'} · ${data.inference_ms} ms inference · ${data.elapsed_ms} ms total`
    $('audio-info').textContent =
      `${audioFile.name} · ${data.sampling_rate} Hz · ${data.channels} channel(s) · ${data.duration_seconds.toFixed(2)} s${data.truncated ? ' · classified first ' + data.processed_seconds.toFixed(3) + ' s' : ''}`
  } catch (error) {
    $('audio-error').textContent = error.message
    $('audio-predictions').textContent = 'No prediction available.'
  } finally {
    running = false
    health()
  }
})
