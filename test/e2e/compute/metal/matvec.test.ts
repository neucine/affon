import { test, expect, values } from 'std:test'
import { tensor, matmul, transpose } from 'affon:compute'

test('Metal single-row products match CPU across batches, tails and strided inputs', () => {
  for (const [k, n] of [
    [63, 64],
    [64, 64],
    [65, 63],
    [257, 64],
    [257, 128],
    [257, 129],
    [1025, 63],
    [1500, 64],
    [1024, 127],
    [1023, 64],
    [1025, 129],
  ]) {
    const a = tensor(
      Array.from({ length: 2 }, (_, b) => [
        Array.from({ length: k }, (_, i) => (((i * 3 + b) % 19) - 9) / 32),
      ]),
      { dtype: 'f32' },
    )
    const b = tensor(
      Array.from({ length: 2 }, (_, batch) =>
        Array.from({ length: k }, (_, i) =>
          Array.from(
            { length: n },
            (_, j) => (((i * 7 + j * 3 + batch) % 23) - 11) / 32,
          ),
        ),
      ),
      { dtype: 'f32' },
    )
    const expected = (values(matmul(a, b)) as number[]).flat(
      Infinity,
    ) as number[]
    const actual = (
      values(matmul(a.to('metal'), b.to('metal')).to('cpu')) as number[]
    ).flat(Infinity) as number[]
    expect(actual.length).toBe(expected.length)
    expect(actual.every((v, i) => Math.abs(v - expected[i]) < 1e-4)).toBe(true)
  }
  // Column-major RHS and a sliced vector exercise offsets and non-unit strides.
  const a = tensor([Array.from({ length: 3000 }, (_, i) => (i % 11) / 16)], {
    dtype: 'f32',
  })
  const b = tensor(
    Array.from({ length: 63 }, (_, j) =>
      Array.from({ length: 1500 }, (_, i) => ((i + j) % 7) / 16),
    ),
    { dtype: 'f32' },
  )
  const cpuA = a.slice([':', '1:1501'])
  const gpuA = a.to('metal').slice([':', '1:1501'])
  expect(
    values(matmul(gpuA, transpose(b.to('metal'), 0, 1)).to('cpu')),
  ).toBeAllClose(values(matmul(cpuA, transpose(b, 0, 1))))
})

test('Metal short vector products preserve offsets and non-unit strides', () => {
  const a = tensor([Array.from({ length: 516 }, (_, i) => Math.sin(i) / 8)], {
    dtype: 'f32',
    device: 'cpu',
  })
  const b = tensor(
    Array.from({ length: 129 }, (_, j) =>
      Array.from({ length: 258 }, (_, i) => Math.cos(i * 3 + j) / 8),
    ),
    { dtype: 'f32', device: 'cpu' },
  )
  const run = (device: 'cpu' | 'metal') =>
    matmul(
      a.to(device).slice([':', '1:515:2']),
      transpose(b.to(device).slice(['1:129:2', '1:']), 0, 1),
    )
  expect(values(run('metal').to('cpu'))).toBeAllClose(values(run('cpu')))
})
