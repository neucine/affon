import fs from 'std:fs'
import {
  tensor,
  stft_power,
  filterbank,
  clamp,
  log,
  div,
  cast,
  max,
  add,
  reshape,
  type Device,
} from 'affon:compute'
import { resample_audio } from './shared/audio.ts'

/** Whisper tiny English: 30-second padded, centered Slaney log-Mel features. */
export function load_whisper_processor(
  directory: string,
  device: Device = 'cpu',
) {
  const c = JSON.parse(fs.readFileSync(`${directory}/preprocessor_config.json`))
  if (
    c.feature_extractor_type !== 'WhisperFeatureExtractor' ||
    c.feature_size !== 80 ||
    c.sampling_rate !== 16000 ||
    c.n_fft !== 400 ||
    c.hop_length !== 160 ||
    c.chunk_length !== 30 ||
    (c.dither ?? 0) !== 0 ||
    (c.padding_value ?? 0) !== 0 ||
    (c.padding_side ?? 'right') !== 'right'
  )
    throw Error('Unsupported Whisper feature configuration')
  const hzToMel = (hz: number) =>
    hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27)
  const melToHz = (mel: number) =>
    mel < 15
      ? mel * (200 / 3)
      : 1000 * Math.exp((mel - 15) * (Math.log(6.4) / 27))
  const points = Array.from({ length: 82 }, (_, i) =>
    melToHz((i * hzToMel(8000)) / 81),
  )
  const filters = Array.from({ length: 80 }, (_, b) =>
    Array.from({ length: 201 }, (_, i) => {
      const hz = i * 40,
        weight =
          (Math.max(
            0,
            Math.min(
              (hz - points[b]) / (points[b + 1] - points[b]),
              (points[b + 2] - hz) / (points[b + 2] - points[b + 1]),
            ),
          ) *
            2) /
          (points[b + 2] - points[b])
      return weight
    }),
  )
  const window = Array.from(
    { length: 400 },
    (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / 400),
  )
  const filterTensor = tensor(filters, { dtype: 'f64', device: 'cpu' })
  const windowTensor = tensor(window, { dtype: 'f64', device: 'cpu' })
  const logBase = tensor(Math.LN10, { dtype: 'f64', device: 'cpu' })
  const four = tensor(4, { dtype: 'f32', device: 'cpu' })
  return {
    process(
      samples: Float32Array,
      sampling_rate: number,
      profile?: (timings: Record<string, number>) => void,
    ) {
      const started = Date.now()
      let fft_ms = 0,
        filter_ms = 0
      const wave = resample_audio(samples, sampling_rate)
      if (wave.length < 400)
        throw Error('Whisper requires at least 25 ms of audio')
      const input_start = Date.now()
      const signal = tensor(Array.from(wave), { dtype: 'f32', device: 'cpu' })
      const input_ms = Date.now() - input_start
      const fft_start = Date.now()
      const spectrum = stft_power(signal, windowTensor, 160, 480000, 3000)
      fft_ms = Date.now() - fft_start
      const filter_start = Date.now()
      const energies = filterbank(spectrum, filterTensor)
      filter_ms = Date.now() - filter_start
      const normalize_start = Date.now()
      const logged = cast(
        div(log(clamp(energies, 1e-10, Number.MAX_VALUE)), logBase),
        'f32',
      )
      const maximum = max(logged).item()
      const normalized = div(
        add(clamp(logged, maximum - 8, maximum), four),
        four,
      )
      const normalize_ms = Date.now() - normalize_start
      const transfer_start = Date.now()
      const result = reshape(normalized, [1, 80, 3000]).to(device)
      profile?.({
        input_ms,
        fft_ms,
        filter_ms,
        normalize_ms,
        transfer_ms: Date.now() - transfer_start,
        total_ms: Date.now() - started,
      })
      return result
    },
  }
}
