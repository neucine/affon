import { describe, expect, test } from 'std:test'
import { Session, Tensor, gradient, program, type FormalTensor } from 'affon:compute'
import { conv1d, conv_transpose1d, embedding, layer_norm, linear } from 'affon:nn'
import { cross_entropy, mean } from 'affon:ops'

describe('Program neural-network authoring', () => {
  test('authors reusable linear, embedding, normalization, and ordinary function composition', () => {
    const input = linear({ out_features: 4 })
    const norm = layer_norm()
    const output = linear({ out_features: 2, bias: false })
    const encoder = ({ x }: { x: FormalTensor }, name = 'encoder') => {
      let value = input({ x }, `${name}.input`)
      value = norm({ x: value }, `${name}.norm`)
      return output({ x: value }, `${name}.output`)
    }
    const model = program('canonical_nn', p => encoder({ x: p.argument('x', Tensor.f32([2, 3])) }, 'encoder'))
    expect(model.inspect().parameters.map(value => value.name)).toEqual([
      'encoder.input.weight', 'encoder.input.bias', 'encoder.norm.weight', 'encoder.norm.bias', 'encoder.output.weight',
    ])
    const modelInspection = model.inspect()
    expect(modelInspection.nodes[modelInspection.outputs[0]].spec.shape).toEqual([2, 2])

    const tokens = embedding({ num_embeddings: 8, embedding_dim: 3 })
    const lookup = program('lookup', p => tokens({ indices: p.argument('indices', Tensor.i64([2])) }, 'tokens'))
    expect(lookup.inspect().parameters[0].name).toBe('tokens.weight')
    const lookupInspection = lookup.inspect()
    expect(lookupInspection.nodes[lookupInspection.outputs[0]].spec.shape).toEqual([2, 3])
  })

  test('authors cross entropy as an ordinary operation and validates label specs', () => {
    const loss = program('classification_loss', p => {
      const logits = p.argument('logits', Tensor.f32([2, 3]))
      const labels = p.argument('labels', Tensor.i64([2]))
      return cross_entropy(logits, labels)
    })
    const inspection = loss.inspect()
    expect(inspection.nodes[inspection.outputs[0]].spec.shape).toEqual([1])

    expect(() => program('bad_loss_dtype', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])),
      p.argument('labels', Tensor.f32([2])),
    ))).toThrow('i64')
    expect(() => program('bad_loss_shape', p => cross_entropy(
      p.argument('logits', Tensor.f32([2, 3])),
      p.argument('labels', Tensor.i64([3])),
    ))).toThrow('without its class dimension')
  })

  test('executes canonical layers with Session-owned values', () => {
    const head = linear({ out_features: 2 })
    const model = program('linear_runtime', p => head({ x: p.argument('x', Tensor.f32([2, 2])) }, 'head'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { seed: 7 })
    const x = session.tensor([[1, 0], [0, 1]])
    const result = session.compile(model).run({ x }, state) as any
    expect(result.shape).toEqual([2, 2])
    expect((result.to_array() as number[][]).flat().every(Number.isFinite)).toBe(true)
    result.dispose()
    x.dispose()
    state.dispose()
    session.dispose()
  })

  test('executes grouped Conv1D with stride, dilation, and padding', () => {
    const layer = conv1d({ out_channels: 2, kernel_size: 2, stride: 2, padding: 1, dilation: 2, groups: 2, bias: false })
    const model = program('conv1d_runtime', p => layer({ x: p.argument('x', Tensor.f32([1, 2, 5])) }, 'conv'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { parameters: { 'conv.weight': [[[1, 2]], [[3, 4]]] } })
    const input = session.tensor([[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]])
    const output = session.compile(model).run({ x: input }, state) as any
    expect(output.shape).toEqual([1, 2, 3])
    expect(output.to_array()).toEqual([[[4, 10, 4], [80, 220, 120]]])
    output.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('executes transposed Conv1D with stride and output padding', () => {
    const layer = conv_transpose1d({ out_channels: 1, kernel_size: 2, stride: 2, output_padding: 1, bias: false })
    const model = program('conv_transpose_runtime', p => layer({ x: p.argument('x', Tensor.f32([1, 1, 3])) }, 'deconv'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { parameters: { 'deconv.weight': [[[2, 3]]] } })
    const input = session.tensor([[[1, 2, 3]]])
    const output = session.compile(model).run({ x: input }, state) as any
    expect(output.shape).toEqual([1, 1, 7])
    expect(output.to_array()).toEqual([[[2, 3, 4, 6, 6, 9, 0]]])
    output.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('Conv1D preserves the channel and kernel weight order', () => {
    const layer = conv1d({ out_channels: 1, kernel_size: 2, bias: false })
    const model = program('conv1d_channels', p => layer({ x: p.argument('x', Tensor.f32([1, 2, 3])) }, 'conv'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { parameters: { 'conv.weight': [[[1, 2], [3, 4]]] } })
    const input = session.tensor([[[1, 2, 3], [10, 20, 30]]])
    const output = session.compile(model).run({ x: input }, state) as any
    expect(output.to_array()).toEqual([[[115, 188]]])
    output.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('Conv1D differentiates through its slices and matrix multiplication', () => {
    const layer = conv1d({ out_channels: 1, kernel_size: 2, bias: false })
    const objective = program('conv1d_gradient', p => mean(layer({ x: p.argument('x', Tensor.f32([1, 1, 3])) }, 'conv')))
    const derivative = gradient(objective, 'conv.weight')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative, { parameters: { 'conv.weight': [[[1, 1]]] } })
    const input = session.tensor([[[1, 2, 3]]])
    const result = session.compile(derivative).run({ x: input }, state) as any
    expect(result.to_array()).toEqual([[[1.5, 2.5]]])
    result.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('transposed Conv1D differentiates its flipped kernel', () => {
    const layer = conv_transpose1d({ out_channels: 1, kernel_size: 2, bias: false })
    const objective = program('conv_transpose_gradient', p => mean(layer({ x: p.argument('x', Tensor.f32([1, 1, 3])) }, 'deconv')))
    const derivative = gradient(objective, 'deconv.weight')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative, { parameters: { 'deconv.weight': [[[2, 3]]] } })
    const input = session.tensor([[[1, 2, 3]]])
    const result = session.compile(derivative).run({ x: input }, state) as any
    expect(result.to_array()).toEqual([[[1.5, 1.5]]])
    result.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('strided Conv1D differentiates through stepped slices', () => {
    const layer = conv1d({ out_channels: 1, kernel_size: 2, stride: 2, bias: false })
    const objective = program('strided_conv_gradient', p => mean(layer({ x: p.argument('x', Tensor.f32([1, 1, 5])) }, 'conv')))
    const derivative = gradient(objective, 'conv.weight')
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(derivative, { parameters: { 'conv.weight': [[[1, 2]]] } })
    const input = session.tensor([[[1, 2, 3, 4, 5]]])
    const result = session.compile(derivative).run({ x: input }, state) as any
    expect(result.to_array()).toEqual([[[2, 3]]])
    result.dispose(); input.dispose(); state.dispose(); session.dispose()
  })

  test('grouped transposed Conv1D combines padding and dilation', () => {
    const layer = conv_transpose1d({ out_channels: 2, kernel_size: 2, stride: 2, padding: 1, dilation: 2, output_padding: 1, groups: 2, bias: false })
    const model = program('grouped_deconv', p => layer({ x: p.argument('x', Tensor.f32([1, 2, 2])) }, 'deconv'))
    const session = new Session({ device: 'cpu' })
    const state = session.initialize(model, { parameters: { 'deconv.weight': [[[1, 2]], [[3, 4]]] } })
    const input = session.tensor([[[1, 2], [10, 20]]])
    const result = session.compile(model).run({ x: input }, state) as any
    expect(result.shape).toEqual([1, 2, 4])
    expect(result.to_array()).toEqual([[[0, 4, 0, 4], [0, 100, 0, 80]]])
    result.dispose(); input.dispose(); state.dispose(); session.dispose()
  })
})
