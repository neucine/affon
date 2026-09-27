import fs from 'std:fs'
import { tensor, type Device } from 'affon:compute'
import { resample_audio } from './shared/audio.ts'

// Radix-2 FFT in double precision; output rounds to complex64 like HF's NumPy STFT.
function power_spectrum(real: Float64Array) {
  const n = real.length,
    imaginary = new Float64Array(n)
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) [real[i], real[j]] = [real[j], real[i]]
  }
  for (let size = 2; size <= n; size *= 2) {
    for (let start = 0; start < n; start += size) {
      for (let j = 0; j < size / 2; j++) {
        const angle = (-2 * Math.PI * j) / size,
          cos = Math.cos(angle),
          sin = Math.sin(angle)
        const a = start + j,
          b = a + size / 2
        const re = real[b] * cos - imaginary[b] * sin,
          im = real[b] * sin + imaginary[b] * cos
        real[b] = real[a] - re
        imaginary[b] = imaginary[a] - im
        real[a] += re
        imaginary[a] += im
      }
    }
  }
  return Array.from(
    { length: n / 2 + 1 },
    (_, i) => Math.fround(real[i]) ** 2 + Math.fround(imaginary[i]) ** 2,
  )
}

/** AST log-Mel filterbank, matching Transformers' NumPy feature extractor at 16 kHz. */
export function load_ast_processor(directory: string, device: Device = 'cpu') {
  const c = JSON.parse(fs.readFileSync(`${directory}/preprocessor_config.json`))
  if (
    c.feature_extractor_type !== 'ASTFeatureExtractor' ||
    c.sampling_rate !== 16000 ||
    !Number.isInteger(c.num_mel_bins) ||
    c.num_mel_bins < 1 ||
    c.num_mel_bins > 256 ||
    !Number.isInteger(c.max_length) ||
    c.max_length < 1 ||
    c.max_length > 1024 ||
    c.do_normalize !== true ||
    !Number.isFinite(c.mean) ||
    !Number.isFinite(c.std) ||
    c.std <= 0 ||
    (c.padding_value ?? 0) !== 0 ||
    (c.padding_side ?? 'right') !== 'right' ||
    c.return_attention_mask === true
  )
    throw Error('Unsupported AST processor configuration')
  const bins: number = c.num_mel_bins,
    length: number = c.max_length
  const mel = (hz: number) => 1127 * Math.log(1 + hz / 700)
  const low = mel(20),
    high = mel(8000),
    step = (high - low) / (bins + 1)
  const filters = Array.from({ length: bins }, (_, band) => {
    const left = low + band * step,
      middle = left + step,
      right = middle + step
    return Array.from({ length: 257 }, (_, i) => {
      const f = mel((i * 16000) / 512)
      return {
        index: i,
        weight: Math.max(0, Math.min((f - left) / step, (right - f) / step)),
      }
    }).filter((x) => x.weight > 0)
  })
  const window = Array.from(
    { length: 400 },
    (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 399),
  )
  const mean = Math.fround(c.mean),
    divisor = Math.fround(c.std * 2)
  return {
    sampling_rate: 16000,
    num_mel_bins: bins,
    max_samples: (length - 1) * 160 + 400,
    process(samples: Float32Array, sampling_rate: number) {
      const audio = resample_audio(samples, sampling_rate)
      if (audio.length < 400)
        throw Error('AST requires at least 25 ms of audio')
      const frames = Math.min(
        length,
        1 + Math.floor((audio.length - 400) / 160),
      )
      const output = Array.from({ length }, () =>
        Array(bins).fill(Math.fround(-mean / divisor)),
      )
      for (let frame = 0; frame < frames; frame++) {
        const start = frame * 160,
          real = new Float64Array(512)
        let dc = 0
        for (let i = 0; i < 400; i++) dc += audio[start + i]
        dc /= 400
        for (let i = 0; i < 400; i++) {
          const current = audio[start + i] - dc
          const previous = i === 0 ? current : audio[start + i - 1] - dc
          real[i] = (current - 0.97 * previous) * window[i]
        }
        const power = power_spectrum(real)
        output[frame] = filters.map((filter) => {
          const energy = filter.reduce(
            (sum, entry) => sum + power[entry.index] * entry.weight,
            0,
          )
          const log = Math.fround(
            Math.log(Math.max(1.192092955078125e-7, energy)),
          )
          return Math.fround(Math.fround(log - mean) / divisor)
        })
      }
      return tensor([output], { dtype: 'f32', device })
    },
  }
}
