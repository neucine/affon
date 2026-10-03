import { describe, test, expect, values } from 'std:test'
import checkpoint from 'affon:checkpoint'
import { add, copy, matmul, module as computeModule, parameter, relu, tensor } from 'affon:compute/legacy'
import nn from 'affon:nn/legacy'
import { metalAvailable } from '../../support/metal.ts'

function makeTwoLayerModule() {
  return computeModule({
    linear1: nn.Linear(2, 3),
    linear2: nn.Linear(3, 1),
  }, (state, x: any) => state.linear2(relu(state.linear1(x))))
}

describe('nn save and load', () => {
  test('custom module save/load keeps stable key naming', () => {
    const model = makeTwoLayerModule()
    checkpoint.save(model.state(), '/tmp/affon-test-custom-module.safetensors')
    const state = checkpoint.load('/tmp/affon-test-custom-module.safetensors')
    const keys = Object.keys(state).sort()

    expect(keys).toEqual([
      'linear1.bias',
      'linear1.weight',
      'linear2.bias',
      'linear2.weight',
    ])
    expect(state['linear1.weight'].shape).toEqual([2, 3])
  })

  test('Sequential save/load keeps index-based key naming and round-trips outputs', () => {
    const m1 = nn.Sequential(
      nn.Linear(3, 4),
      relu,
      nn.Linear(4, 2),
    )
    m1.save('/tmp/affon-test-seq.safetensors')

    const state = checkpoint.load('/tmp/affon-test-seq.safetensors')
    expect(Object.keys(state).sort()).toEqual([
      'layers.0.bias',
      'layers.0.weight',
      'layers.2.bias',
      'layers.2.weight',
    ])

    const m2 = nn.Sequential(
      nn.Linear(3, 4),
      relu,
      nn.Linear(4, 2),
    )
    m2.load('/tmp/affon-test-seq.safetensors')

    const x = tensor([[1, 2, 3]], { dtype: 'f32' })
    expect(values(m1(x))).toEqual(values(m2(x)))
  })

  test.skip(() => !metalAvailable())('checkpoint.save materializes metal tensors to safetensors', () => {
    const state = {
      weight: tensor([[1, 2], [3, 4]], { dtype: 'f32' }).to('metal'),
      bias: tensor([5, 6], { dtype: 'f32' }).to('metal'),
    }

    checkpoint.save(state, '/tmp/affon-test-metal-state.safetensors')
    const loaded = checkpoint.load('/tmp/affon-test-metal-state.safetensors')

    expect(values(loaded.weight)).toEqual([[1, 2], [3, 4]])
    expect(values(loaded.bias)).toEqual([5, 6])
  })

  test('module_list exposes prefixed parameter traversal for built-in layers and compute children', () => {
    const listed = nn.module_list([
      nn.Linear(2, 3),
      computeModule({ proj: nn.Linear(3, 1) }, (state, x: any) => state.proj(x)),
    ])

    expect(listed.length).toBe(2)
    expect(listed.parameters.named().map(([name]) => name)).toEqual([
      '0.weight',
      '0.bias',
      '1.proj.weight',
      '1.proj.bias',
    ])
  })

  test('dtype options are preserved on common nn modules', () => {
    const linear = nn.Linear(2, 3, { dtype: 'f32' })
    const layerNorm = nn.LayerNorm(4, { dtype: 'f32' })
    const embedding = nn.Embedding(8, 4, { dtype: 'f32' })

    expect(linear.weight.dtype).toBe('f32')
    expect(layerNorm.gamma.dtype).toBe('f32')
    expect(embedding.weight.dtype).toBe('f32')
  })
})
