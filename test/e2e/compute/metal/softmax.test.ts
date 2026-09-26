import { describe, test, expect, values } from 'std:test'
import { mul, softmax, sum, tensor, reshape, transpose } from 'affon:compute'
import { internal_tensor } from '../../../support/compute.ts'

// Metal tests intentionally check grad placement (`grad_device`) and direct
// tracked-input backward behavior, so they keep using the lower-level path.
describe('compute metal softmax', () => {
  test('runs softmax on metal and keeps forward output on device', () => {
    const x = internal_tensor([[1, 2, 3]], { dtype: 'f32', device: 'metal' })
    const y = softmax(x, 1)

    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toBeAllClose([
      [0.09003057, 0.24472847, 0.66524096],
    ])
  })

  test('supports backward through metal softmax with metal grads', () => {
    const x = internal_tensor([[1, 2, 3]], { dtype: 'f32', device: 'metal' })
    const y = softmax(x, 1)
    const loss = sum(mul(y, x))
    loss.backward()

    expect(y.device).toBe('metal')
    expect(x.grad_device).toBe('metal')
    expect(values(x.grad?.to('cpu'))).toBeAllClose([
      [-0.05178652, 0.10395811, 0.9478284],
    ])
  })

  test('keeps softmax finite on metal for extreme finite logits', () => {
    const x = internal_tensor(
      [
        [1e20, 0, -1e20],
        [-1e20, 1e20, 0],
      ],
      { dtype: 'f32', device: 'metal' },
    )
    const y = softmax(x, 1)
    const rows = values(y.to('cpu')) as number[][]

    expect(rows).toBeAllFinite()
    expect(rows).toHaveRowSumsCloseTo(1)

    const loss = sum(mul(y, x))
    loss.backward()
    const grad = values(x.grad?.to('cpu')) as number[][]
    expect(grad).toBeAllFinite()
  })
})

test('dense multidimensional Metal softmax matches stable row references', () => {
  const rows = Array.from({ length: 6 }, (_, r) =>
    Array.from({ length: 1500 }, (_, c) =>
      r === 0 ? 10000 : 10000 + ((c * 7 + r * 13) % 31),
    ),
  )
  const expected = rows.flatMap((row) => {
    const peak = Math.max(...row)
    const exp = row.map((x) => Math.exp(x - peak))
    const total = exp.reduce((a, b) => a + b, 0)
    return exp.map((x) => x / total)
  })
  for (const shape of [
    [2, 3, 1500],
    [1, 2, 3, 1500],
  ]) {
    const x = reshape(tensor(rows, { dtype: 'f32' }).to('metal'), shape)
    const y = softmax(x, shape.length - 1)
    expect(y.device).toBe('metal')
    const actual = (values(y.to('cpu')) as number[]).flat(Infinity) as number[]
    expect(actual.length).toBe(expected.length)
    expect(
      actual.every(
        (v, i) => Number.isFinite(v) && Math.abs(v - expected[i]) < 1e-6,
      ),
    ).toBe(true)
  }
})

test('Metal multidimensional softmax preserves strided and non-last-axis reductions', () => {
  const cpu = tensor(
    [
      [
        [1, 2, 4],
        [3, 0, -1],
      ],
      [
        [8, 2, 5],
        [-4, 0, 2],
      ],
    ],
    { dtype: 'f32' },
  )
  for (const axis of [0, 1, 2]) {
    expect(values(softmax(cpu.to('metal'), axis).to('cpu'))).toBeAllClose(
      values(softmax(cpu, axis)),
    )
    expect(
      values(softmax(transpose(cpu.to('metal'), 0, 2), axis).to('cpu')),
    ).toBeAllClose(values(softmax(transpose(cpu, 0, 2), axis)))
  }
})
