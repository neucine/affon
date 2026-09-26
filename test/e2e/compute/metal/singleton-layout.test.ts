import { test, expect, values } from 'std:test'
import telemetry from 'std:telemetry'
import {
  tensor,
  reshape,
  transpose,
  contiguous,
  add,
  sum,
  grad,
  parameter,
} from 'affon:compute'

function allocations() {
  return (
    telemetry
      .metrics()
      .find(
        (m) => m.scope === 'compute.storage' && m.name === 'allocation_count',
      )?.value ?? 0
  )
}

test('singleton transposes remain dense without allocating a copy', () => {
  for (const device of ['cpu', 'metal'] as const) {
    const x = reshape(
      tensor(
        Array.from({ length: 384 }, (_, i) => i / 16),
        { dtype: 'f32', device },
      ),
      [1, 6, 1, 64],
    )
    const view = transpose(x, 1, 2)
    const before = allocations()
    const dense = contiguous(view)
    expect(allocations() - before).toBe(0)
    const flat = reshape(dense, [384])
    expect(values(flat)).toEqual(Array.from({ length: 384 }, (_, i) => i / 16))
    expect(values(reshape(add(view, view), [384]))).toEqual(
      Array.from({ length: 384 }, (_, i) => i / 8),
    )
  }
})

test('real transposes and offset views still materialize correctly', () => {
  for (const device of ['cpu', 'metal'] as const) {
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: 'f32', device },
    )
    const view = transpose(x, 0, 1)
    expect(values(reshape(contiguous(view), [6]))).toEqual([1, 4, 2, 5, 3, 6])
    const offset = transpose(
      reshape(x, [2, 3, 1]).slice(['1:', ':', ':']),
      1,
      2,
    )
    expect(values(reshape(contiguous(offset), [3]))).toEqual([4, 5, 6])
  }
})

test('singleton contiguous aliases preserve gradients', () => {
  for (const device of ['cpu', 'metal'] as const) {
    const x = parameter([3], { dtype: 'f32', device })
    const y = contiguous(transpose(reshape(x, [1, 3, 1]), 1, 2))
    grad(sum(y), [x])
    expect(values(x.grad!)).toEqual([1, 1, 1])
  }
})
