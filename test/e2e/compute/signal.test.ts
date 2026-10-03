import { test, expect, values } from 'std:test'
import {
  tensor,
  stft_power,
  filterbank,
  transpose,
  parameter,
  compile,
} from 'affon:compute/legacy'

function reference(
  signal: number[],
  window: number[],
  hop: number,
  length: number,
  frames: number,
) {
  const n = window.length
  return Array.from({ length: Math.floor(n / 2) + 1 }, (_, bin) =>
    Array.from({ length: frames }, (_, frame) => {
      let re = 0,
        im = 0
      for (let i = 0; i < n; i++) {
        let index = frame * hop + i - Math.floor(n / 2)
        if (index < 0) index = -index
        if (index >= length) index = 2 * length - index - 2
        const x = (signal[index] ?? 0) * window[i],
          angle = (-2 * Math.PI * bin * i) / n
        re += x * Math.cos(angle)
        im += x * Math.sin(angle)
      }
      return Math.fround(re) ** 2 + Math.fround(im) ** 2
    }),
  )
}
test('native STFT matches independent DFT for mixed-radix, odd, reflected and padded frames', () => {
  for (const n of [4, 5, 8, 15, 400]) {
    const samples = Array.from({ length: n === 5 ? 4 * n : n + 3 }, (_, i) =>
      Math.fround(Math.sin(i * 0.7) / 4),
    )
    const window = Array.from(
      { length: n },
      (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n),
    )
    const hop = n,
      length = 4 * n,
      frames = 5
    const actual = values(
      stft_power(
        tensor(samples, { dtype: 'f32', device: 'cpu' }),
        tensor(window, { dtype: 'f64', device: 'cpu' }),
        hop,
        length,
        frames,
      ),
    ) as number[][]
    const expected = reference(samples, window, hop, length, frames)
    for (let b = 0; b < actual.length; b++)
      for (let f = 0; f < frames; f++)
        expect(
          Math.abs(actual[b][f] - expected[b][f]) <=
            1e-6 + 1e-6 * Math.abs(expected[b][f]),
        ).toBe(true)
  }
})
test('filterbank matches independent sparse projection', () => {
  const s = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ],
    w = [
      [0, 2, 0],
      [0.25, 0, 0.75],
    ]
  const expected = w.map((row) =>
    s[0].map((_, f) => row.reduce((v, weight, b) => v + weight * s[b][f], 0)),
  )
  expect(
    values(
      filterbank(
        tensor(s, { dtype: 'f64', device: 'cpu' }),
        tensor(w, { dtype: 'f64', device: 'cpu' }),
      ),
    ),
  ).toEqual(expected)
})
test('spectral primitives reject unsupported inputs and compiled calls retain data dependence', () => {
  const x = tensor([1, 2, 3, 4], { dtype: 'f32', device: 'cpu' }),
    w = tensor([1, 1], { dtype: 'f64', device: 'cpu' })
  for (const hop of [0, -1, 1.5, NaN])
    expect(() => stft_power(x, w, hop)).toThrow()
  expect(() => stft_power(x, w, 1, 2, 2)).toThrow()
  expect(() => stft_power(x, w, 1, 4, 6)).toThrow()
  expect(() => stft_power(x.to('metal'), w, 1)).toThrow()
  const matrix = tensor(
    [
      [1, 2],
      [3, 4],
    ],
    { dtype: 'f64', device: 'cpu' },
  )
  expect(() => filterbank(transpose(matrix, 0, 1), matrix)).toThrow()
  expect(() => filterbank(matrix, w)).toThrow()
  const tracked = parameter([4], { dtype: 'f32', device: 'cpu' })
  expect(() => stft_power(tracked, w, 1)).toThrow()
  const compiled = compile((signal) => stft_power(signal, w, 1))
  expect(values(compiled(x))).toEqual(values(stft_power(x, w, 1)))
  const y = tensor([2, 4, 6, 8], { dtype: 'f32', device: 'cpu' })
  expect(values(compiled(y))).toEqual(values(stft_power(y, w, 1)))
})
