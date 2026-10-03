import { describe, test, expect, values } from 'std:test'
import { add, cat, contiguous, permute, relu, reshape, softmax, stack, sum, tensor, transpose } from 'affon:compute/legacy'
import { internal_tensor } from '../../../support/compute.ts'

describe('compute metal noncontiguous', () => {
  test('materializes transposed views for supported metal ops', () => {
    const x = internal_tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32', device: 'metal' })
    const xt = transpose(x, 0, 1)

    const y = add(xt, tensor([[1, 1], [1, 1], [1, 1]], { dtype: 'f32' }).to('metal'))
    const z = relu(xt)
    const s = softmax(xt, 1)

    expect(y.device).toBe('metal')
    expect(values(y.to('cpu'))).toEqual([[2, 5], [3, 6], [4, 7]])
    expect(z.device).toBe('metal')
    expect(values(z.to('cpu'))).toEqual([[1, 4], [2, 5], [3, 6]])
    expect(s.device).toBe('metal')

    sum(y).backward()
    expect(x.grad_device).toBe('metal')
    expect(x.grad?.shape).toEqual([2, 3])
  })

  test('keeps shape-op backward on metal', () => {
    const x = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32', device: 'metal' })
    sum(reshape(x, [4])).backward()
    expect(x.grad_device).toBe('metal')
    expect(values(x.grad?.to('cpu'))).toEqual([[1, 1], [1, 1]])

    const y = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32', device: 'metal' })
    const yc = cat([y, y], 0)
    expect(yc.device).toBe('metal')
    expect(values(yc.to('cpu'))).toEqual([[1, 2], [3, 4], [1, 2], [3, 4]])
    sum(yc).backward()
    expect(y.grad_device).toBe('metal')
    expect(values(y.grad?.to('cpu'))).toEqual([[2, 2], [2, 2]])

    const z = internal_tensor([[1, 2], [3, 4]], { dtype: 'f32', device: 'metal' })
    const zs = stack([z, z], 0)
    expect(zs.device).toBe('metal')
    expect(values(zs.to('cpu'))).toEqual([[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
    sum(zs).backward()
    expect(z.grad_device).toBe('metal')
    expect(values(z.grad?.to('cpu'))).toEqual([[2, 2], [2, 2]])
  })

  test('keeps transposed and reshaped metal views finite for large finite inputs', () => {
    const x = tensor([
      [1e20, 1e10, 5],
      [-1e20, -1e10, -5],
    ], { dtype: 'f32' }).to('metal')

    const xt = transpose(x, 0, 1)
    expect(values(xt.to('cpu'))).toBeAllFinite()
    expect(values(relu(xt).to('cpu'))).toBeAllFinite()
    expect(values(softmax(xt, 1).to('cpu'))).toBeAllFinite()
    expect(values(reshape(x, [3, 2]).to('cpu'))).toBeAllFinite()
  })

  test('reads flat host values from metal tensors and views', () => {
    const x = tensor([[1, 2, 3], [4, 5, 6]], { dtype: 'f32' }).to('metal')
    expect(values(x)).toEqual([[1, 2, 3], [4, 5, 6]])

    const xt = transpose(x, 0, 1)
    expect(values(xt)).toEqual([[1, 4], [2, 5], [3, 6]])
  })
})
