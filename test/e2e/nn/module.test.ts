import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn/legacy'
import {
  add,
  copy,
  matmul,
  module as computeModule,
  parameter,
  tensor,
} from 'affon:compute/legacy'

describe('nn custom module interop', () => {
  test('compute modules use training/eval paths with nn layers', () => {
    const mod = computeModule(
      {
        trainOffset: tensor([1, 1], { dtype: 'f32' }),
        evalOffset: tensor([10, 10], { dtype: 'f32' }),
      },
      (state, x: any) => add(x, state.trainOffset),
      (state, x: any) => add(x, state.evalOffset),
    )

    const x = tensor([1, 2], { dtype: 'f32' })

    expect(values(mod(x))).toEqual([2, 3])
    mod.eval()
    expect(values(mod(x))).toEqual([11, 12])
    mod.train()
    expect(values(mod(x))).toEqual([2, 3])
  })

  test('compute modules discover parameter fields automatically', () => {
    const weight = copy(parameter([2, 2], { dtype: 'f32' }), tensor([[1, 2], [3, 4]], { dtype: 'f32' }))
    const bias = copy(parameter([1, 2], { dtype: 'f32' }), tensor([[0.5, -1]], { dtype: 'f32' }))
    const mod = computeModule(
      { weight, bias },
      (state, x: any) => add(matmul(x, state.weight), state.bias),
    ) as nn.Module & { weight: typeof weight; bias: typeof bias }

    expect(mod.parameters.length).toBe(2)
    expect(values(mod(tensor([[1, 2]], { dtype: 'f32' })))).toBeAllClose([[7.5, 9]])
  })

  test('nn.module_list accepts compute-authored child modules', () => {
    const block = computeModule(
      { linear: nn.Linear(2, 2) },
      (state, x: any) => state.linear(x),
    )
    const layers = nn.module_list([block, block])

    expect(layers.length).toBe(2)
    expect(layers.parameters.named().map(([name]) => name)).toEqual([
      '0.linear.weight',
      '0.linear.bias',
      '1.linear.weight',
      '1.linear.bias',
    ])
  })

  test('module metadata propagates structural paths to submodules', () => {
    const explicit = nn.Linear(2, 2).metadata('shared.proj')
    const block = computeModule(
      {
        linear_layer: nn.Linear(2, 2),
        explicit,
      },
      (state, x: any) => state.explicit(state.linear_layer(x)),
    )
    const model = computeModule(
      {
        decoder_block: block,
        layers: nn.module_list([
          nn.Linear(2, 2),
          computeModule({ inner_layer: nn.Linear(2, 2) }, (state, x: any) => state.inner_layer(x)),
        ]),
      },
      (state, x: any) => state.layers[1](state.layers[0](state.decoder_block(x))),
    ).metadata('model')

    expect(model.module_path).toBe('model')
    expect(block.module_path).toBe('model.decoder_block')
    expect(block.linear_layer.module_path).toBe('model.decoder_block.linear_layer')
    expect(explicit.module_path).toBe('shared.proj')
    expect(model.layers[0].module_path).toBe('model.layers.0')
    expect(model.layers[1].module_path).toBe('model.layers.1')
    expect(model.layers[1].inner_layer.module_path).toBe('model.layers.1.inner_layer')
  })
})
