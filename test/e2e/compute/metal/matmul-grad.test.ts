import { describe, test, expect, values } from 'std:test'
import { matmul, sum, tensor } from 'affon:compute'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal matmul grad', () => {
  test('keeps forward tensors and covered grads on metal', () => {
    const W = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32', device: 'metal' })
    const x = internal_tensor([[5], [6]], { dtype: 'f32', device: 'metal' })

    const y = matmul(W, x)
    const loss = sum(y)
    loss.backward()

    expect(y.device).toBe('metal')
    expect(loss.device).toBe('metal')
    expect(W.grad_device).toBe('metal')
    expect(x.grad_device).toBe('metal')
    expect(W.grad?.device).toBe('metal')
    expect(x.grad?.device).toBe('metal')
    expect(values(W.grad?.to('cpu'))).toEqual([[5, 6], [5, 6]])
    expect(values(x.grad?.to('cpu'))).toEqual([[4], [6]])
  })

  test('supports batched matmul and batched gradients on cpu', () => {
    const a = internal_tensor([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ], { dtype: 'f32' })
    const b = internal_tensor([
      [[1], [2]],
      [[3], [4]],
    ], { dtype: 'f32' })

    const y = matmul(a, b)
    expect(y.shape).toEqual([2, 2, 1])
    expect(values(y)).toEqual([
      [[5], [11]],
      [[39], [53]],
    ])

    sum(y).backward()

    expect(values(a.grad)).toEqual([
      [[1, 2], [1, 2]],
      [[3, 4], [3, 4]],
    ])
    expect(values(b.grad)).toEqual([
      [[4], [6]],
      [[12], [14]],
    ])
  })

  test('supports batched metal matmul and keeps output on metal', () => {
    const a = tensor([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ], { dtype: 'f32' }).to('metal')
    const b = tensor([
      [[1], [2]],
      [[3], [4]],
    ], { dtype: 'f32' }).to('metal')

    const y = matmul(a, b)
    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([
      [[5], [11]],
      [[39], [53]],
    ])
  })

  test('keeps metal matmul finite for large finite inputs within safe range', () => {
    const a = internal_tensor([
      [1e10, -1e10],
      [5e9, -5e9],
    ], { dtype: 'f32', device: 'metal' })
    const b = internal_tensor([
      [1e-10, 2e-10],
      [-1e-10, 3e-10],
    ], { dtype: 'f32', device: 'metal' })

    const y = matmul(a, b)
    const yValues = values(y.to('cpu')) as number[][]
    expect(yValues.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)

    sum(y).backward()
    const aGrad = values(a.grad?.to('cpu')) as number[][]
    const bGrad = values(b.grad?.to('cpu')) as number[][]
    expect(aGrad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
    expect(bGrad.every((row) => row.every((value) => Number.isFinite(value)))).toBe(true)
  })
})
