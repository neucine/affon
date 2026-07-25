import { describe, test, expect, values } from 'std:test'
import nn from 'affon:nn'
import { axes, compile, grad, softmax, sum, tensor } from 'affon:compute'
import { internal_tensor } from './support/compute.ts'
import { metalAvailable } from './support/metal.ts'

import {
  DecoderBlock,
  DecoderInputEmbedding,
  FeedForward,
  apply_causal_mask,
  causal_mask,
  position_ids,
  SelfAttention,
  sinusoidal_encoding,
} from '../src/index.ts'

function trackedF32(data: number[] | number[][] | number[][][], device?: 'cpu' | 'metal') {
  const rank = Array.isArray((data as any)[0])
    ? (Array.isArray((data as any)[0][0]) ? 3 : 2)
    : 1
  const namedAxes = rank === 3
    ? [axes.batch, axes.token, axes.feature]
    : rank === 2
      ? [axes.batch, axes.feature]
      : [axes.token]
  return internal_tensor(data as any, { dtype: 'f32', device, axes: namedAxes })
}

// This file still keeps a few direct tracked-input backward checks on purpose.
// They validate transformer stability for non-parameter leaves and Metal runtime
// behavior beyond the public `grad(loss, params)` training path.
describe('@affon/transformers', () => {
  test('exposes masking and position helpers', () => {
    expect(values(causal_mask(4))).toEqual([
      [0, 1, 1, 1],
      [0, 0, 1, 1],
      [0, 0, 0, 1],
      [0, 0, 0, 0],
    ])

    const masked = apply_causal_mask(tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ], { dtype: 'f32' }))
    expect(values(masked)).toEqual([
      [1, -1000000000, -1000000000],
      [4, 5, -1000000000],
      [7, 8, 9],
    ])

    const enc = sinusoidal_encoding(3, 4)
    expect(enc.shape).toEqual([3, 4])
    const rows = values(enc) as number[][]
    expect(rows[0][0]).toBe(0)
    expect(rows[0][1]).toBe(1)

    expect(values(position_ids(4))).toEqual([0, 1, 2, 3])
    expect(causal_mask(4).axes).toEqual(['token', 'token'])
    expect(enc.axes).toEqual(['token', 'feature'])
    expect(position_ids(4).axes).toEqual(['token'])
  })

  test('builds decoder input embeddings and feed-forward outputs', () => {
    const embed = DecoderInputEmbedding(32, 6, { positional: 'learned', maxSeqLen: 16 })
    expect(embed.token_embedding.weight.axes).toEqual(['vocab', 'feature'])
    expect(embed.position_embedding?.weight.axes).toEqual(['token', 'feature'])
    const embedded = embed(tensor([
      [1, 2, 3],
      [4, 5, 6],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] }))
    expect(embedded.shape).toEqual([2, 3, 6])
    expect(embed.parameters.length).toBe(2)

    const ff = FeedForward(6, { hiddenDim: 12 })
    expect(ff.up.weight.axes).toEqual(['feature', 'hidden'])
    expect(ff.down.weight.axes).toEqual(['feature', 'hidden'])
    const fed = ff(embedded)
    expect(fed.shape).toEqual([2, 3, 6])
    expect(ff.parameters.length).toBe(4)

    const attn = SelfAttention(6, { numHeads: 2, causal: true })
    expect(attn.q_proj.weight.axes).toEqual(['feature', 'hidden'])
    const attended = attn(embedded)
    expect(attended.shape).toEqual([2, 3, 6])
    expect(attn.parameters.length).toBe(8)

    const block = DecoderBlock(6, { numHeads: 2, hiddenDim: 12, causal: true })
    const blocked = block(embedded)
    expect(blocked.shape).toEqual([2, 3, 6])
    expect(block.parameters.length).toBe(16)
  })

  test('keeps compiled self-attention correct across differing batch shapes', () => {
    const attn = SelfAttention(8, { numHeads: 2, causal: true })
    const compiled = compile((x) => attn(x as any))

    const x2 = tensor([
      [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
      ],
      [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
      ],
    ], { dtype: 'f32', axes: [axes.batch, axes.token, axes.feature] })
    const x3 = tensor([
      [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
      ],
      [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
      ],
      [
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 0, 0],
      ],
    ], { dtype: 'f32', axes: [axes.batch, axes.token, axes.feature] })

    const eager2 = attn(x2).to_array()
    const eager3 = attn(x3).to_array()

    expect(compiled(x2).to_array()).toEqual(eager2)
    expect(compiled.run(x3).to_array()).toEqual(eager3)

    const graphProgram = (compiled as any).graph?.(x3)
    expect(graphProgram ?? null).toBe(null)
  })

  test('reports matching finite diagnostic failures through nn diagnostics', () => {
    nn.diagnostics.configure({ mode: 'error', include: ['finite:SelfAttention.project3d.*'] })
    try {
      const attn = SelfAttention(4, { numHeads: 2, causal: true })
      attn.q_proj.weight = tensor([
        [Number.NaN, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
      ], { dtype: 'f32' }) as any

      expect(() => attn(tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
      ]], { dtype: 'f32', axes: [axes.batch, axes.token, axes.feature] }))).toThrow('SelfAttention.project3d.linear produced a non-finite tensor')
    } finally {
      nn.diagnostics.configure({ mode: 'off', include: [], exclude: [] })
    }
  })

  test.skip(() => !metalAvailable())('keeps self-attention finite on metal for large finite activations', () => {
    setDevice('cpu')
    setDevice('metal')
    const attn = SelfAttention(8, { numHeads: 2, causal: true })

    const x = trackedF32([[
      [1e10, -1e10, 5e9, -5e9, 1e9, -1e9, 1e8, -1e8],
      [9e9, -9e9, 4e9, -4e9, 8e8, -8e8, 7e7, -7e7],
      [8e9, -8e9, 3e9, -3e9, 7e8, -7e8, 6e7, -6e7],
    ]], 'metal')

    const y = values(attn(x).to('cpu')) as number[][][]
    expect(y.every((batch) => batch.every((row) => row.every((value) => Number.isFinite(value))))).toBe(true)
    setDevice('cpu')
  })

  test.skip(() => !metalAvailable())('keeps feed-forward backward finite on metal for large finite activations', () => {
    setDevice('cpu')
    setDevice('metal')

    const ff = FeedForward(8, { hiddenDim: 16 })
    const x = trackedF32([[
      [1e10, -1e10, 5e9, -5e9, 1e9, -1e9, 1e8, -1e8],
      [9e9, -9e9, 4e9, -4e9, 8e8, -8e8, 7e7, -7e7],
      [8e9, -8e9, 3e9, -3e9, 7e8, -7e8, 6e7, -6e7],
    ]], 'metal')

    sum(ff(x)).backward()

    expect(values(x.grad?.to('cpu'))).toBeAllFinite()
    for (const param of ff.parameters) {
      if (param.grad !== null) expect(values(param.grad.to('cpu'))).toBeAllFinite()
    }
    setDevice('cpu')
  })

  test.skip(() => !metalAvailable())('keeps decoder block backward finite on metal for large finite activations', () => {
    setDevice('cpu')
    setDevice('metal')

    const block = DecoderBlock(8, { numHeads: 2, hiddenDim: 16, causal: true })
    const x = trackedF32([[
      [1e10, -1e10, 5e9, -5e9, 1e9, -1e9, 1e8, -1e8],
      [9e9, -9e9, 4e9, -4e9, 8e8, -8e8, 7e7, -7e7],
      [8e9, -8e9, 3e9, -3e9, 7e8, -7e8, 6e7, -6e7],
    ]], 'metal')

    sum(block(x)).backward()

    expect(values(x.grad?.to('cpu'))).toBeAllFinite()
    for (const param of block.parameters) {
      if (param.grad !== null) expect(values(param.grad.to('cpu'))).toBeAllFinite()
    }
    setDevice('cpu')
  })

  test.skip(() => !metalAvailable())('keeps causal masking and masked softmax finite on metal', () => {
    setDevice('cpu')
    setDevice('metal')

    const scores = tensor([[
      [1e10, -1e10, 5e9],
      [9e9, 4e9, -9e9],
      [8e9, 7e9, 6e9],
    ]], { dtype: 'f32' }).to('metal')

    const masked = apply_causal_mask(scores)
    expect(values(masked.to('cpu'))).toBeAllFinite()

    const weights = softmax(masked, 2)
    expect(values(weights.to('cpu'))).toBeAllFinite()
    setDevice('cpu')
  })
})
