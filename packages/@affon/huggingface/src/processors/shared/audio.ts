/** Bounded WAV decoding. Samples are mono floats; stereo channels are averaged. */
export function decode_wav(bytes: Uint8Array) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const tag = (offset: number) =>
    String.fromCharCode(...bytes.subarray(offset, offset + 4))
  if (bytes.length < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE')
    throw Error('Expected a RIFF/WAVE audio file')
  const end = view.getUint32(4, true) + 8
  if (end > bytes.length || end < 12) throw Error('Truncated WAV container')
  let format:
    | {
        kind: number
        channels: number
        rate: number
        bits: number
        block: number
      }
    | undefined
  let data: { offset: number; size: number } | undefined
  for (let offset = 12; offset < end; ) {
    if (offset + 8 > end) throw Error('Truncated WAV chunk header')
    const size = view.getUint32(offset + 4, true),
      start = offset + 8
    if (start + size > end) throw Error('Truncated WAV chunk')
    if (tag(offset) === 'fmt ') {
      if (format || size < 16) throw Error('Invalid WAV format chunk')
      format = {
        kind: view.getUint16(start, true),
        channels: view.getUint16(start + 2, true),
        rate: view.getUint32(start + 4, true),
        block: view.getUint16(start + 12, true),
        bits: view.getUint16(start + 14, true),
      }
    } else if (tag(offset) === 'data') {
      if (data) throw Error('Multiple WAV data chunks are unsupported')
      data = { offset: start, size }
    }
    offset = start + size + (size % 2)
  }
  if (!format || !data) throw Error('WAV requires format and data chunks')
  const { kind, channels, rate, bits, block } = format
  if (
    ![1, 2].includes(channels) ||
    rate < 8000 ||
    rate > 96000 ||
    (kind !== 1 && kind !== 3) ||
    !(kind === 1 ? [8, 16, 24, 32] : [32]).includes(bits) ||
    block !== (channels * bits) / 8 ||
    !data.size ||
    data.size % block
  )
    throw Error('Supported WAV: mono/stereo PCM8/16/24/32 or float32, 8–96 kHz')
  const count = data.size / block
  if (count / rate > 30 || count / rate < 0.025)
    throw Error('Choose audio between 25 ms and 30 seconds')
  const samples = new Float32Array(count)
  for (let i = 0; i < count; i++) {
    let value = 0
    for (let c = 0; c < channels; c++) {
      const p = data.offset + i * block + (c * bits) / 8
      let x: number
      if (kind === 3) x = view.getFloat32(p, true)
      else if (bits === 8) x = (view.getUint8(p) - 128) / 128
      else if (bits === 16) x = view.getInt16(p, true) / 32768
      else if (bits === 24)
        x =
          (((view.getUint8(p) |
            (view.getUint8(p + 1) << 8) |
            (view.getUint8(p + 2) << 16)) <<
            8) >>
            8) /
          8388608
      else x = view.getInt32(p, true) / 2147483648
      if (!Number.isFinite(x) || Math.abs(x) > 1)
        throw Error('WAV samples must be finite values within [-1, 1]')
      value += x / channels
    }
    samples[i] = value
  }
  return {
    samples,
    sampling_rate: rate,
    channels,
    duration_seconds: count / rate,
  }
}

/** Windowed-sinc resampling with an anti-alias low-pass filter; zero extension at boundaries. */
export function resample_audio(
  samples: Float32Array,
  source_rate: number,
  target_rate = 16000,
): Float32Array {
  if (
    ![source_rate, target_rate].every(
      (r) => Number.isInteger(r) && r >= 8000 && r <= 96000,
    )
  )
    throw Error('Unsupported audio sample rate')
  if (
    !samples.length ||
    samples.length > source_rate * 30 ||
    samples.some((x) => !Number.isFinite(x) || Math.abs(x) > 1)
  )
    throw Error('Expected finite audio samples within [-1, 1], up to 30 seconds')
  if (source_rate === target_rate) return samples
  const ratio = target_rate / source_rate,
    cutoff = Math.min(1, ratio) * 0.95
  const radius = Math.ceil(24 / cutoff),
    output = new Float32Array(Math.ceil(samples.length * ratio))
  for (let i = 0; i < output.length; i++) {
    const position = i / ratio,
      center = Math.floor(position)
    let value = 0,
      total = 0
    for (let j = center - radius; j <= center + radius; j++) {
      const distance = position - j
      if (Math.abs(distance) >= radius) continue
      const z = Math.PI * distance * cutoff
      const weight =
        cutoff *
        (z === 0 ? 1 : Math.sin(z) / z) *
        (0.5 + 0.5 * Math.cos((Math.PI * distance) / radius))
      total += weight
      if (j >= 0 && j < samples.length) value += weight * samples[j]
    }
    output[i] = value / total
  }
  return output
}
