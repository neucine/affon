import fs from 'std:fs'
import type { Tensor } from 'affon:compute/legacy'
import { test, expect, values } from 'std:test'
import {
  erf,
  tensor,
  transpose,
  compile,
  parameter,
  copy,
  grad,
  sum,
  mul,
} from 'affon:compute/legacy'
const reference = JSON.parse(
  fs.readFileSync('test/e2e/compute/erf-reference.json'),
)

test('erf matches independent math.erf values and preserves axes', () => {
  const input = tensor(reference.inputs, { dtype: 'f32', axes: ['sample'] })
  const output = erf(input)
  expect(output.axes).toEqual(input.axes)
  const result = values(output) as number[]
  expect(result.length).toBe(reference.outputs.length)
  expect(
    result.every(
      (v, i) => Number.isFinite(v) && Math.abs(v - reference.outputs[i]) < 5e-7,
    ),
  ).toBe(true)
})

test('erf supports compiled and strided execution and special values', () => {
  const run = compile((x: Tensor) => erf(x))
  for (const data of [
    [
      [0, 1, -1],
      [2, -2, 3],
    ],
    [[0.5, -0.5]],
  ]) {
    const x = transpose(tensor(data, { dtype: 'f32' }), 0, 1)
    expect(values(run(x))).toBeAllClose(values(erf(x)))
  }
  const result = values(
    erf(tensor([0, Infinity, -Infinity, NaN], { dtype: 'f32' })),
  ) as number[]
  expect(result[0]).toBe(0)
  expect(result[1]).toBe(1)
  expect(result[2]).toBe(-1)
  expect(Number.isNaN(result[3])).toBe(true)
  expect(() => erf(tensor([1, 2], { dtype: 'i64' }))).toThrow()
})

test('erf analytic gradient matches independent values with nonuniform upstream weights', () => {
  const xs = [-2, -0.5, 0, 0.5, 2],
    weights = [1, 2, -3, 4, 5]
  const x = parameter([5], { dtype: 'f32' })
  copy(x, tensor(xs, { dtype: 'f32' }))
  grad(sum(mul(erf(x), tensor(weights, { dtype: 'f32' }))), [x])
  const actual = values(x.grad!) as number[]
  expect(
    actual.every(
      (v, i) =>
        Math.abs(
          v - ((weights[i] * 2) / Math.sqrt(Math.PI)) * Math.exp(-(xs[i] ** 2)),
        ) < 1e-5,
    ),
  ).toBe(true)
})

test('erf retains the documented CPU f64 approximation accuracy', () => {
  const result = values(
    erf(tensor(reference.inputs, { dtype: 'f64', device: 'cpu' })),
  ) as number[]
  expect(
    result.every((v, i) => Math.abs(v - reference.outputs[i]) < 1.5e-7),
  ).toBe(true)
})
