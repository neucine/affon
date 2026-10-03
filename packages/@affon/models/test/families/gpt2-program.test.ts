import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_gpt2 } from '../../src/gpt2/model.ts'

test('runs GPT-2 forward, decode sessions, and generation through Programs', () => {
  const source = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dimensions: number[]): any => dimensions.length
      ? Array.from({ length: dimensions[0] }, () => data(dimensions.slice(1)))
      : fill
    return source.tensor(data(shape)) as Tensor
  }
  const affine = (input: number, output: number) => ({ weight: value([input, output]), bias: value([output], 0) })
  const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
  const config = { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 }
  const model = create_gpt2(config, {
    tokenEmbedding: value([7, 4]),
    positionEmbedding: value([4, 4]),
    finalNorm: norm(),
    blocks: [{ attentionNorm: norm(), feedForwardNorm: norm(), qkv: affine(4, 12), attentionOutput: affine(4, 4), expand: affine(4, 8), contract: affine(8, 4) }],
  } as any)

  const forward = model.forward([1, 2])
  expect(forward.logits.shape).toEqual([1, 2, 7])
  const decode = model.create_session()
  const first = decode.forward([1, 2])
  const second = decode.forward([3])
  expect(decode.length).toBe(3)
  expect(second.logits.shape).toEqual([1, 1, 7])
  expect(model.generate([1], 2)).toEqual([1, 0, 0])

  for (const result of [forward, first, second]) {
    result.logits.dispose()
    for (const hidden of result.hidden_states) hidden.dispose()
  }
  model.dispose()
  source.dispose()
})
