import { test, expect, values } from 'std:test'
import {
  add,
  mul,
  tensor,
  transpose,
  compile,
  no_grad,
  parameter,
  copy,
  grad,
  sum,
} from 'affon:compute'
import type { Tensor } from 'affon:compute'
const fused = compile((x: Tensor, scale: Tensor, bias: Tensor) =>
  add(mul(x, scale), bias),
)

test('compiled Metal affine fuses dense suffix broadcasts with exact two-step rounding', () => {
  for (const data of [
    [
      [1.3, -2.7, 10.2],
      [7.1, 0, -3.3],
    ],
    [[4, 5, 6]],
  ]) {
    const x = tensor(data, { dtype: 'f32', device: 'metal' })
    for (const scaleData of [[2, -1, 0.5], 2]) {
      const scale = tensor(scaleData, { dtype: 'f32', device: 'metal' }),
        bias = tensor([3, 4, -5], { dtype: 'f32', device: 'metal' })
      const fused = compile((a: Tensor, b: Tensor, c: Tensor) =>
        add(mul(a, b), c),
      )
      const expected = values(add(mul(x, scale), bias))
      const actual = no_grad(() => fused(x, scale, bias))
      expect(values(actual)).toEqual(expected)
    }
  }
  const x = tensor([1.0000001192092896], { dtype: 'f32', device: 'metal' })
  const scale = tensor([1.0000001192092896], { dtype: 'f32', device: 'metal' })
  const bias = tensor([-1.000000238418579], { dtype: 'f32', device: 'metal' })
  const rounding = compile((a: Tensor, b: Tensor, c: Tensor) =>
    add(mul(a, b), c),
  )
  expect(values(no_grad(() => rounding(x, scale, bias)))).toEqual(
    values(add(mul(x, scale), bias)),
  )
})

test('affine fusion falls back for non-suffix broadcasts, strided inputs and CPU', () => {
  for (const device of ['cpu', 'metal'] as const) {
    const x = tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: 'f32', device },
    )
    const scale = tensor([[2], [3]], { dtype: 'f32', device }),
      bias = tensor([1, 2, 3], { dtype: 'f32', device })
    expect(values(no_grad(() => fused(x, scale, bias)))).toEqual(
      values(add(mul(x, scale), bias)),
    )
    const xt = transpose(x, 0, 1),
      scalar = tensor(2, { dtype: 'f32', device })
    expect(values(no_grad(() => fused(xt, scalar, scalar)))).toEqual(
      values(add(mul(xt, scalar), scalar)),
    )
  }
})

test('compiled affine retains gradients when grad mode is enabled', () => {
  const x = parameter([2, 3], { dtype: 'f32', device: 'metal' })
  copy(
    x,
    tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { dtype: 'f32', device: 'metal' },
    ),
  )
  const s = tensor([2, 3, 4], { dtype: 'f32', device: 'metal' }),
    b = tensor(1, { dtype: 'f32', device: 'metal' })
  grad(sum(fused(x, s, b)), [x])
  expect(values(x.grad!)).toEqual([
    [2, 3, 4],
    [2, 3, 4],
  ])
})
