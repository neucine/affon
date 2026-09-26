import { test, expect, values } from 'std:test'
import { tensor, layer_norm, reshape, transpose } from 'affon:compute'
function close(actual: number[], expected: number[]) {
  expect(actual.length).toBe(expected.length)
  expect(
    actual.every(
      (v, i) => Number.isFinite(v) && Math.abs(v - expected[i]) < 3e-5,
    ),
  ).toBe(true)
}
function reference(row: number[], eps: number) {
  const mean = row.reduce((a, b) => a + b, 0) / row.length
  const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length
  return row.map((v) => (v - mean) / Math.sqrt(variance + eps))
}
test('Metal row layer norm covers singleton, odd and model widths with independent references', () => {
  for (const width of [1, 7, 384, 1536]) {
    const rows = [
      Array(width).fill(7),
      Array.from({ length: width }, (_, i) => ((i * 7) % 31) - 15),
    ]
    const x = reshape(tensor(rows, { dtype: 'f32', device: 'metal' }), [
      1,
      2,
      width,
    ])
    close(
      (values(layer_norm(x, 2, 0.01)) as number[]).flat(Infinity) as number[],
      rows.flatMap((row) => reference(row, 0.01)),
    )
  }
})
test('Metal normalization retains offset and non-last-axis behavior', () => {
  const x = tensor(
    [
      [1, 2, 4],
      [3, 6, 9],
      [5, 7, 11],
    ],
    { dtype: 'f32', device: 'cpu' },
  )
  const metal = x.to('metal')
  for (const axis of [0, 1]) {
    close(
      (values(layer_norm(metal, axis, 0.1)) as number[]).flat(
        Infinity,
      ) as number[],
      (values(layer_norm(x, axis, 0.1)) as number[]).flat(Infinity) as number[],
    )
    close(
      (values(layer_norm(transpose(metal, 0, 1), axis, 0.1)) as number[]).flat(
        Infinity,
      ) as number[],
      (values(layer_norm(transpose(x, 0, 1), axis, 0.1)) as number[]).flat(
        Infinity,
      ) as number[],
    )
  }
  close(
    (values(layer_norm(metal.slice(['1:', ':']), 1, 0.1)) as number[]).flat(
      Infinity,
    ) as number[],
    (values(layer_norm(x.slice(['1:', ':']), 1, 0.1)) as number[]).flat(
      Infinity,
    ) as number[],
  )
})
