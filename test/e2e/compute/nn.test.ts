import { describe, expect, test, values } from 'std:test'
import { grad, sum } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'
import { internal_tensor } from '../../support/compute.ts'

describe('compute with nn', () => {
  test('nn layers accept compute-native values and attach gradients', () => {
    const linear = nn.Linear(2, 3)
    const linearInput = internal_tensor([
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]],
    ], { dtype: 'f32' })
    const linearOut = linear(linearInput)
    expect(linearOut.shape).toEqual([2, 2, 3])
    grad(sum(linearOut), [linearInput as any, ...linear.parameters])
    expect(values(linearInput.grad)).toBeAllFinite()
    expect(values(linear.weight.grad)).toBeAllFinite()
    expect(values(linear.bias.grad)).toBeAllFinite()

    const layerNorm = nn.LayerNorm(3)
    const normInput = internal_tensor([
      [[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]],
    ], { dtype: 'f32' })
    const normOut = layerNorm(normInput)
    expect(normOut.shape).toEqual([2, 2, 3])
    grad(sum(normOut), [normInput as any, ...layerNorm.parameters])
    expect(values(normInput.grad)).toBeAllFinite()
    expect(values(layerNorm.gamma.grad)).toBeAllFinite()
    expect(values(layerNorm.beta.grad)).toBeAllFinite()

    const batchNorm = nn.BatchNorm(3)
    const batchInput = internal_tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ], { dtype: 'f32' })
    const batchOut = batchNorm(batchInput)
    expect(batchOut.shape).toEqual([4, 3])
    grad(sum(batchOut), [batchInput as any, ...batchNorm.parameters])
    expect(values(batchInput.grad)).toBeAllFinite()
    expect(values(batchNorm.gamma.grad)).toBeAllFinite()
    expect(values(batchNorm.beta.grad)).toBeAllFinite()
  })
})
