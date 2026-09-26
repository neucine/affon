import { test, expect, beforeAll, afterAll } from 'std:test'
import fs from 'std:fs'
import { run } from 'std:process'
import { decode_wav, resample_audio, load_processor } from '../src/index.ts'
let directory = ''
beforeAll(async () => {
  directory = (
    await run({ cmd: 'mktemp', args: ['-d', '/tmp/affon-audio-test.XXXXXX'] })
  ).stdout.trim()
  fs.writeFileSync(
    `${directory}/config.json`,
    JSON.stringify({ model_type: 'audio-spectrogram-transformer' }),
  )
  fs.writeFileSync(
    `${directory}/preprocessor_config.json`,
    JSON.stringify({
      feature_extractor_type: 'ASTFeatureExtractor',
      sampling_rate: 16000,
      num_mel_bins: 8,
      max_length: 4,
      mean: -6.845978,
      std: 5.5654526,
      do_normalize: true,
    }),
  )
})
afterAll(async () => {
  if (directory) await run({ cmd: 'rm', args: ['-rf', directory] })
})
test('AST matches independent HF features including DC removal, padding, and truncation', () => {
  const processor = load_processor(directory, { task: 'audio-classification' })
  const cases = JSON.parse(
    fs.readFileSync(
      'packages/@affon/huggingface/test/fixtures/ast-processor.json',
    ),
  )
  for (const c of cases) {
    const samples = Float32Array.from({ length: c.length }, (_, i) =>
      c.silence ? 0 : (((i * 73) % 251) - 125) / 256,
    )
    const actual = (
      processor.process(samples, 16000).to_array() as number[][][]
    ).flat(2)
    const expected = c.features.flat(2)
    expect(actual.every((x, i) => Math.abs(x - expected[i]) < 2e-6)).toBe(true)
  }
  expect(() => processor.process(new Float32Array(10), 16000)).toThrow()
  expect(() => processor.process(new Float32Array([NaN]), 16000)).toThrow()
})
function wav(bits = 16, kind = 1, channels = 2, count = 400) {
  const block = (channels * bits) / 8,
    bytes = new Uint8Array(44 + count * block),
    v = new DataView(bytes.buffer)
  for (const [offset, text] of [
    [0, 'RIFF'],
    [8, 'WAVE'],
    [12, 'fmt '],
    [36, 'data'],
  ] as const)
    for (let i = 0; i < text.length; i++) bytes[offset + i] = text.charCodeAt(i)
  v.setUint32(4, bytes.length - 8, true)
  v.setUint32(16, 16, true)
  v.setUint16(20, kind, true)
  v.setUint16(22, channels, true)
  v.setUint32(24, 16000, true)
  v.setUint32(28, 16000 * block, true)
  v.setUint16(32, block, true)
  v.setUint16(34, bits, true)
  v.setUint32(40, count * block, true)
  for (let i = 44; i < bytes.length; i += bits / 8) {
    if (kind === 3) v.setFloat32(i, 0.5, true)
    else if (bits === 8) v.setUint8(i, 192)
    else if (bits === 16) v.setInt16(i, 16384, true)
    else if (bits === 24) {
      bytes[i] = 0
      bytes[i + 1] = 0
      bytes[i + 2] = 64
    } else v.setInt32(i, 1073741824, true)
  }
  return bytes
}
test('WAV supports PCM widths, float32 and stereo downmix; rejects malformed files', () => {
  for (const [bits, kind] of [
    [8, 1],
    [16, 1],
    [24, 1],
    [32, 1],
    [32, 3],
  ]) {
    const result = decode_wav(wav(bits, kind))
    expect(result.sampling_rate).toBe(16000)
    expect(result.samples.length).toBe(400)
    expect(result.samples.every((x) => x === 0.5)).toBe(true)
  }
  const stereo = wav()
  const v = new DataView(stereo.buffer)
  for (let i = 46; i < stereo.length; i += 4) v.setInt16(i, -16384, true)
  expect(decode_wav(stereo).samples.every((x) => x === 0)).toBe(true)
  expect(() => decode_wav(stereo.subarray(0, 100))).toThrow()
  expect(() => decode_wav(new Uint8Array(44))).toThrow()
  const bad = wav(32, 3)
  new DataView(bad.buffer).setFloat32(44, NaN, true)
  expect(() => decode_wav(bad)).toThrow()
})
test('Resampling preserves passband tones and suppresses downsampling aliases', () => {
  const rms = (x: Float32Array) =>
    Math.sqrt(
      x.slice(100, -100).reduce((s, v) => s + v * v, 0) / (x.length - 200),
    )
  const tone = (freq: number) =>
    Float32Array.from(
      { length: 4800 },
      (_, i) => 0.5 * Math.sin((2 * Math.PI * freq * i) / 48000),
    )
  const pass = resample_audio(tone(1000), 48000),
    stop = resample_audio(tone(12000), 48000)
  expect(pass.length).toBe(1600)
  expect(Math.abs(rms(pass) - Math.sqrt(0.125)) < 0.002).toBe(true)
  expect(rms(stop) < 0.001).toBe(true)
  const up = resample_audio(
    Float32Array.from(
      { length: 800 },
      (_, i) => 0.5 * Math.sin((2 * Math.PI * 1000 * i) / 8000),
    ),
    8000,
  )
  expect(up.length).toBe(1600)
  expect(Math.abs(rms(up) - Math.sqrt(0.125)) < 0.002).toBe(true)
  expect(() => resample_audio(new Float32Array([0]), 0)).toThrow()
})

test('WAV accepts 30 seconds and rejects longer recordings', () => {
  expect(decode_wav(wav(16,1,1,480000)).duration_seconds).toBe(30)
  expect(()=>decode_wav(wav(16,1,1,480001))).toThrow()
})
