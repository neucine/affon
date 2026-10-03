import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_llama } from '../../src/llama/model.ts'

test('runs Llama forward, decode sessions, and generation through Programs', () => {
  const source = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dimensions: number[]): any => dimensions.length
      ? Array.from({ length: dimensions[0] }, () => data(dimensions.slice(1)))
      : fill
    return source.tensor(data(shape)) as Tensor
  }
  const config = {
    width: 4,
    innerWidth: 8,
    heads: 2,
    kvHeads: 1,
    layers: 1,
    contextLength: 4,
    vocabSize: 7,
    epsilon: 1e-5,
    ropeTheta: 10_000,
  }
  const model = create_llama(config, {
    tokenEmbedding: value([7, 4]),
    finalNorm: value([4], 1),
    blocks: [{
      attentionNorm: value([4], 1),
      feedForwardNorm: value([4], 1),
      query: value([4, 4]),
      key: value([4, 2]),
      value: value([4, 2]),
      attentionOutput: value([4, 4]),
      gate: value([4, 8]),
      up: value([4, 8]),
      down: value([8, 4]),
    }],
  } as any)

  const forward = model.forward([1, 2])
  expect(forward.logits.shape).toEqual([1, 2, 7])
  expect(forward.hidden_states.map(hidden => hidden.shape)).toEqual([[1, 2, 4], [1, 2, 4]])
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
