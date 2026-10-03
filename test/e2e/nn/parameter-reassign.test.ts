import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import { tensor } from 'affon:compute/legacy'

describe('nn parameter reassignment', () => {
  test('updates Linear weights and bias through property reassignment', () => {
    const layer = nn.Linear(2, 3)
    layer.weight = tensor([[1, 2, 3], [4, 5, 6]])
    layer.bias = tensor([[0.5, -1, 2]])

    const y = layer(tensor([[1, 2]]))
    expect(values(y)).toBeAllClose([[9.5, 11, 17]])
  })

  test('updates LayerNorm affine parameters through property reassignment', () => {
    const layer = nn.LayerNorm(3)
    layer.gamma = tensor([1.5, 0.5, 2])
    layer.beta = tensor([0.25, -0.5, 1])

    const y = layer(tensor([[1, 2, 3]]))
    expect(values(y)).toBeAllClose([[-1.5871035288625852, -0.5, 3.4494713718167804]], { rtol: 1e-5, atol: 1e-6 })
  })
})
