import { test, expect } from 'std:test'
import { layer_norm, tensor, parameter, copy, grad, sum, mul, compile, transpose } from 'affon:compute/legacy'
import type { Tensor, Shape } from 'affon:compute/legacy'

function close(actual: Tensor<Shape>, expected: number[], tolerance = 2e-4) {
  const values = (actual.to_array() as number[]).flat(Infinity) as number[]
  expect(values.length).toBe(expected.length)
  expect(values.every((x, i) => Number.isFinite(x) && Math.abs(x - expected[i]) <= tolerance)).toBe(true)
}
function normalize(row: number[], eps: number) {
  const mean = row.reduce((a, b) => a + b) / row.length
  const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length
  return row.map(x => (x - mean) / Math.sqrt(variance + eps))
}
test('layer_norm preserves axes and normalizes strided and constant inputs', () => {
  const rows = [[1, 2, 7], [4, 4, 4]]
  const x = tensor(rows, { dtype: 'f32', axes: ['batch', 'feature'] })
  const expected = rows.flatMap(row => normalize(row, 0.01))
  const result = layer_norm(x, 1, 0.01)
  close(result, expected)
  expect(result.axes).toEqual(x.axes)
  close(transpose(layer_norm(transpose(x, 0, 1), 0, 0.01), 0, 1), expected)
})
test('layer_norm gradient matches finite differences including epsilon', () => {
  const data = [1, 2, 7], weights = [2, -3, 0.5], eps = 0.1, h = 0.001
  const x = parameter([1, 3], { dtype: 'f32' }); copy(x, tensor([data], { dtype: 'f32' }))
  grad(sum(mul(layer_norm(x, 1, eps), tensor([weights], { dtype: 'f32' }))), [x])
  const loss = (v: number[]) => normalize(v, eps).reduce((sum, y, i) => sum + y * weights[i], 0)
  const expected = data.map((_, i) => {
    const a = [...data], b = [...data]; a[i] += h; b[i] -= h
    return (loss(a) - loss(b)) / (2 * h)
  })
  close(x.grad!, expected)
})
test('layer_norm compiled execution agrees across shapes and rejects invalid arguments', () => {
  const run = compile((x: Tensor) => layer_norm(x, 1, 0.01))
  for (const rows of [[[1, 2, 7]], [[1, 2], [3, 5]]]) {
    close(run(tensor(rows, { dtype: 'f32' })), rows.flatMap(row => normalize(row, 0.01)))
  }
  const x = tensor([[1,2]], { dtype: 'f32' })
  for (const [axis, eps] of [[-1,1e-5],[0.5,1e-5],[2,1e-5],[1,0],[1,-1],[1,NaN],[1,Infinity]]) {
    expect(() => layer_norm(x,axis,eps)).toThrow()
  }
})
