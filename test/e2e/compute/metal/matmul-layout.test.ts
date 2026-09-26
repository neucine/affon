import { test, expect, values } from 'std:test'
import { tensor, matmul, reshape, transpose, type Tensor } from 'affon:compute'

function compare(a: Tensor, b: Tensor) {
  const expected = (values(matmul(a, b)) as number[]).flat(Infinity) as number[]
  const actual = (
    values(matmul(a.to('metal'), b.to('metal')).to('cpu')) as number[]
  ).flat(Infinity) as number[]
  expect(actual.length).toBe(expected.length)
  expect(
    actual.every(
      (v, i) =>
        Math.abs(v - expected[i]) <= 1e-4 + 1e-4 * Math.abs(expected[i]),
    ),
  ).toBe(true)
}
test('Metal dense multi-row matmul preserves odd shapes and a size-one batch', () => {
  for (const [m, k, n] of [
    [17, 33, 19],
    [16, 384, 64],
    [3, 7, 1],
  ]) {
    const a = tensor(
      Array.from({ length: m }, (_, i) =>
        Array.from({ length: k }, (_, j) => Math.sin(i * 3 + j) / 8),
      ),
      { dtype: 'f32', device: 'cpu' },
    )
    const b = tensor(
      Array.from({ length: k }, (_, i) =>
        Array.from({ length: n }, (_, j) => Math.cos(i + j * 5) / 8),
      ),
      { dtype: 'f32', device: 'cpu' },
    )
    compare(a, b)
    compare(reshape(a, [1, m, k]), b)
  }
})
test('Metal matmul keeps transpose, offset, column-stride and multi-batch paths correct', () => {
  const a = tensor(
    Array.from({ length: 4 }, (_, i) =>
      Array.from({ length: 8 }, (_, j) => (i * 8 + j - 11) / 16),
    ),
    { dtype: 'f32', device: 'cpu' },
  )
  const b = tensor(
    Array.from({ length: 8 }, (_, i) =>
      Array.from({ length: 5 }, (_, j) => (i - j) / 16),
    ),
    { dtype: 'f32', device: 'cpu' },
  )
  let reference: unknown[] = []
  // Slice after transfer so GPU operands keep their nonzero offsets/strides.
  for (const device of ['cpu', 'metal'] as const) {
    const x = a.to(device),
      y = b.to(device)
    const cases = [
      matmul(x.slice(['1:', ':']), y),
      matmul(transpose(x, 0, 1), x),
      matmul(x.slice([':', '::2']), y.slice(['::2', ':'])),
      matmul(reshape(x, [2, 2, 8]), y),
    ]
    const outputs = cases.map((v) =>
      (values(v.to('cpu')) as number[]).flat(Infinity),
    )
    if (device === 'cpu') reference = outputs
    else expect(outputs).toBeAllClose(reference)
  }
})

test('Metal automatic large-matrix selection covers offset, transpose and broadcast batches', () => {
  const data = Array.from({ length: 129 }, (_, i) =>
    Array.from({ length: 128 }, (_, j) => Math.sin(i * 7 + j) / 32),
  )
  const cpu = tensor(data, { dtype: 'f32', device: 'cpu' })
  const gpu = cpu.to('metal')
  const run = (x: Tensor) => {
    const a = x.slice(['1:', ':'])
    return [
      matmul(a, transpose(a, 0, 1)),
      matmul(transpose(a, 0, 1), a),
      matmul(reshape(a, [2, 64, 128]), transpose(a, 0, 1)),
    ]
  }
  const expected = run(cpu).map(
    (v) => (values(v) as number[]).flat(Infinity) as number[],
  )
  const actual = run(gpu).map(
    (v) => (values(v.to('cpu')) as number[]).flat(Infinity) as number[],
  )
  for (let c = 0; c < expected.length; c++)
    expect(
      actual[c].every(
        (v, i) =>
          Math.abs(v - expected[c][i]) <=
          1e-4 + 1e-4 * Math.abs(expected[c][i]),
      ),
    ).toBe(true)
})

test('Metal single-row selection preserves dense, transposed, offset and batched products', () => {
  for (const [k, n] of [
    [257, 257],
    [384, 384],
    [1536, 64],
    [64, 1500],
  ]) {
    const a = tensor([Array.from({ length: k }, (_, i) => Math.sin(i) / 8)], {
      dtype: 'f32',
      device: 'cpu',
    })
    const b = tensor(
      Array.from({ length: k + 1 }, (_, i) =>
        Array.from({ length: n }, (_, j) => Math.cos(i + j * 3) / 8),
      ),
      { dtype: 'f32', device: 'cpu' },
    )
    const run = (device: 'cpu' | 'metal') => {
      const x = a.to(device),
        y = b.to(device).slice(['1:', ':'])
      return [matmul(x, y), matmul(reshape(x, [1, 1, k]), y)].map(
        (v) => (values(v.to('cpu')) as number[]).flat(Infinity) as number[],
      )
    }
    const expected = run('cpu'),
      actual = run('metal')
    for (let c = 0; c < expected.length; c++)
      expect(
        actual[c].every(
          (v, i) =>
            Math.abs(v - expected[c][i]) <=
            1e-4 + 1e-4 * Math.abs(expected[c][i]),
        ),
      ).toBe(true)
    compare(
      a,
      transpose(
        tensor(
          Array.from({ length: n }, (_, j) =>
            Array.from({ length: k }, (_, i) => Math.cos(i + j * 3) / 8),
          ),
          { dtype: 'f32', device: 'cpu' },
        ),
        0,
        1,
      ),
    )
  }
})
