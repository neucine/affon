import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_llama } from '../../src/llama/model.ts'

test('authors Llama forward windows as caller-executed Programs', () => {
  const source = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dimensions: number[]): any => dimensions.length
      ? Array.from({ length: dimensions[0] }, () => data(dimensions.slice(1)))
      : fill
    return source.tensor(data(shape)) as Tensor
  }
  const diagonal = (rows: number, columns: number): Tensor => source.tensor(
    Array.from({ length: rows }, (_, row) =>
      Array.from({ length: columns }, (_, column) => row === column ? 1 : 0)),
  )
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
    tokenEmbedding: source.tensor(Array.from({ length: 7 }, (_, row) =>
      Array.from({ length: 4 }, (_, column) => row % 4 === column ? 1 : 0))),
    finalNorm: value([4], 1),
    blocks: [{
      attentionNorm: value([4], 1),
      feedForwardNorm: value([4], 1),
      query: diagonal(4, 4),
      key: diagonal(4, 2),
      value: diagonal(4, 2),
      attentionOutput: diagonal(4, 4),
      gate: value([4, 8]),
      up: value([4, 8]),
      down: value([8, 4]),
    }],
  } as any)

  const runtime = new Session({ device: 'cpu' })
  const program = model.forward(3, 2)
  const state = runtime.initialize(program, { parameters: model.parameters })
  const ids = runtime.tensor([1, 2, 3], { dtype: 'i64' })
  const mask = runtime.tensor([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]], { dtype: 'i64' })
  const [logits, ...hiddenStates] = runtime.compile(program).run({ ids, mask }, state) as Tensor[]
  expect(logits.shape).toEqual([1, 1, 7])
  expect(hiddenStates.map(hidden => hidden.shape)).toEqual([[1, 1, 4], [1, 1, 4]])
  const changedIds = runtime.tensor([1, 1, 3], { dtype: 'i64' })
  const changed = runtime.compile(program).run({ ids: changedIds, mask }, state) as Tensor[]
  expect(changed[0].to_array()).not.toEqual(logits.to_array())
  const changedAgain = runtime.compile(program).run({ ids: changedIds, mask }, state) as Tensor[]
  expect(changedAgain[0].to_array()).toEqual(changed[0].to_array())
  const restrictedMask = runtime.tensor([[[[0, 1, 1], [0, 0, 1], [0, 1, 1]]]], { dtype: 'i64' })
  const restricted = runtime.compile(program).run({ ids, mask: restrictedMask }, state) as Tensor[]
  expect(restricted[0].to_array()).not.toEqual(logits.to_array())
  expect(() => model.forward(3, 3)).toThrow()

  for (const tensor of [logits, ...hiddenStates, ...changed, ...changedAgain, ...restricted, ids, changedIds, mask, restrictedMask]) tensor.dispose()
  state.dispose()
  runtime.dispose()
  source.dispose()
})
