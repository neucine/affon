// UI only: WAV decoding, features and inference are handled by Affon on the server.
function setupTranscription(shared) {
  const element = (id) => document.getElementById(id)
  let file = null,
    url = null,
    available = false
  function controls() {
    const state = shared.state()
    element('speech-file').disabled = state.running || state.decoding
    element('transcribe').disabled =
      state.running ||
      state.decoding ||
      !state.ready ||
      state.busy ||
      !available ||
      !file
  }
  element('speech-file').addEventListener('change', () => {
    file = null
    if (url) URL.revokeObjectURL(url)
    url = null
    element('speech-preview').hidden = true
    element('speech-preview').removeAttribute('src')
    element('speech-preview').load()
    element('speech-error').textContent = ''
    element('speech-info').textContent = ''
    element('speech-stats').textContent = ''
    element('transcript').textContent = 'Choose Transcribe to run Whisper.'
    const selected = element('speech-file').files[0]
    if (selected) {
      if (!selected.size || selected.size > 16 * 1024 * 1024)
        element('speech-error').textContent =
          'Choose a nonempty WAV file no larger than 16 MiB.'
      else {
        file = selected
        url = URL.createObjectURL(file)
        element('speech-preview').src = url
        element('speech-preview').hidden = false
        element('speech-info').textContent =
          `${file.name} · ${(file.size / 1024).toFixed(1)} KiB`
      }
    }
    controls()
  })
  element('transcribe').addEventListener('click', async () => {
    if (!file || shared.state().running) return
    shared.setRunning(true)
    element('speech-error').textContent = ''
    element('speech-stats').textContent = ''
    element('transcript').textContent = 'Transcribing…'
    try {
      const response = await fetch('/api/transcribe', {
        method: 'POST',
        headers: { 'content-type': 'audio/wav' },
        body: file,
      })
      const data = await response.json()
      if (!response.ok) throw Error(data.error || 'Transcription failed')
      element('transcript').textContent =
        data.text.trim() || '(No text generated)'
      element('speech-stats').textContent =
        `${data.model} · ONNX · ${data.device === 'metal' ? 'Metal GPU' : 'CPU'} · ${(data.elapsed_ms / 1000).toFixed(2)} s total · ${(data.preprocessing_ms / 1000).toFixed(2)} s preprocessing · ${(data.encoder_ms / 1000).toFixed(2)} s encoder · ${(data.decoder_ms / 1000).toFixed(2)} s decoder${data.truncated ? ' · token limit reached; transcript may be incomplete' : ''}`
      element('speech-info').textContent =
        `${file.name} · ${data.duration_seconds.toFixed(2)} s · ${data.sampling_rate} Hz · ${data.channels} channel(s)`
    } catch (error) {
      element('speech-error').textContent = error.message
      element('transcript').textContent = 'No transcript available.'
    } finally {
      shared.setRunning(false)
      shared.refresh()
    }
  })
  return {
    controls,
    health(model) {
      available = Boolean(model)
      element('speech-model-info').textContent = available
        ? `${model.id} · cached ONNX decoding · up to ${model.max_new_tokens} generated tokens`
        : 'Whisper is not configured on this server.'
    },
  }
}
