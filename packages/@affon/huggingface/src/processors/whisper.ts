import fs from 'std:fs'
import native from 'affon:compute/native'
import { Session, type Device } from 'affon:compute'
import { resample_audio } from './shared/audio.ts'

/** Whisper tiny English: 30-second padded, centered Slaney log-Mel features. */
export function load_whisper_processor(directory: string, device: Device = 'cpu') {
  const c = JSON.parse(fs.readFileSync(`${directory}/preprocessor_config.json`))
  if (c.feature_extractor_type !== 'WhisperFeatureExtractor' || c.feature_size !== 80 || c.sampling_rate !== 16000 || c.n_fft !== 400 || c.hop_length !== 160 || c.chunk_length !== 30 || (c.dither ?? 0) !== 0 || (c.padding_value ?? 0) !== 0 || (c.padding_side ?? 'right') !== 'right') throw Error('Unsupported Whisper feature configuration')
  const hzToMel = (hz: number) => hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27)
  const melToHz = (mel: number) => mel < 15 ? mel * (200 / 3) : 1000 * Math.exp((mel - 15) * (Math.log(6.4) / 27))
  const points = Array.from({ length: 82 }, (_, index) => melToHz(index * hzToMel(8000) / 81))
  const filters = Array.from({ length: 80 }, (_, band) => Array.from({ length: 201 }, (_, index) => {
    const hz = index * 40
    return Math.max(0, Math.min((hz - points[band]) / (points[band + 1] - points[band]), (points[band + 2] - hz) / (points[band + 2] - points[band + 1]))) * 2 / (points[band + 2] - points[band])
  }))
  const window = Array.from({ length: 400 }, (_, index) => 0.5 - 0.5 * Math.cos(2 * Math.PI * index / 400))
  const cpu = new Session({ device: 'cpu' })
  const filterTensor = cpu.tensor(filters, { dtype: 'f64' }) as any
  const windowTensor = cpu.tensor(window, { dtype: 'f64' }) as any
  return {
    process(samples: Float32Array, sampling_rate: number, profile?: (timings: Record<string, number>) => void) {
      const started = Date.now(), wave = resample_audio(samples, sampling_rate)
      if (wave.length < 400) throw Error('Whisper requires at least 25 ms of audio')
      const inputStart = Date.now(), signal = cpu.tensor(Array.from(wave)) as any
      const input_ms = Date.now() - inputStart
      const fftStart = Date.now(), spectrum = native.stftPower(signal, windowTensor, 160, 480000, 3000)
      const fft_ms = Date.now() - fftStart
      const filterStart = Date.now(), energies = native.filterbank(spectrum, filterTensor)
      const filter_ms = Date.now() - filterStart
      const normalizeStart = Date.now()
      const output = energies.to_array() as number[][]
      let maximum = -Infinity
      for (const row of output) for (let index = 0; index < row.length; index++) {
        row[index] = Math.log10(Math.max(1e-10, row[index]))
        maximum = Math.max(maximum, row[index])
      }
      for (const row of output) for (let index = 0; index < row.length; index++) row[index] = Math.fround((Math.min(maximum, Math.max(maximum - 8, row[index])) + 4) / 4)
      const normalize_ms = Date.now() - normalizeStart
      signal.dispose(); spectrum.dispose(); energies.dispose()
      const transferStart = Date.now(), target = new Session({ device })
      const result = target.tensor([output])
      target.dispose()
      profile?.({ input_ms, fft_ms, filter_ms, normalize_ms, transfer_ms: Date.now() - transferStart, total_ms: Date.now() - started })
      return result
    },
    dispose() { filterTensor.dispose(); windowTensor.dispose(); cpu.dispose() },
  }
}
