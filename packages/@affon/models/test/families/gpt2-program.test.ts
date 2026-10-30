import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_gpt2 } from '../../src/gpt2/model.ts'

test('authors GPT-2 forward windows as caller-executed Programs', () => {
  const source = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dimensions: number[]): any => dimensions.length
      ? Array.from({ length: dimensions[0] }, () => data(dimensions.slice(1)))
      : fill
    return source.tensor(data(shape)) as Tensor
  }
  const affine = (input: number, output: number) => ({ weight: value([input, output]), bias: value([output], 0) })
  const projection = () => ({ weight: source.tensor(Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) => row === column ? 1 : 0))), bias: value([4], 0) })
  const qkv = () => ({ weight: source.tensor(Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 12 }, (_, column) => row === column % 4 ? 1 : 0))), bias: value([12], 0) })
  const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
  const config = { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 }
  const model = create_gpt2(config, {
    tokenEmbedding: source.tensor(Array.from({ length: 7 }, (_, row) =>
      Array.from({ length: 4 }, (_, column) => row % 4 === column ? 1 : 0))),
    positionEmbedding: value([4, 4]),
    finalNorm: norm(),
    blocks: [{ attentionNorm: norm(), feedForwardNorm: norm(), qkv: qkv(), attentionOutput: projection(), expand: affine(4, 8), contract: affine(8, 4) }],
  } as any)

  const runtime = new Session({ device: 'cpu' })
  const program = model.forward(3, 2)
  const state = runtime.initialize(program, { parameters: model.parameters })
  const ids = runtime.tensor([1, 2, 3], { dtype: 'i64' })
  const positions = runtime.tensor([0, 1, 2], { dtype: 'i64' })
  const mask = runtime.tensor([[[[0, 1, 1], [0, 0, 1], [0, 0, 0]]]], { dtype: 'i64' })
  const [logits, ...hiddenStates] = runtime.compile(program).run({ ids, positions, mask }, state) as Tensor[]
  expect(logits.shape).toEqual([1, 1, 7])
  expect(hiddenStates.map(hidden => hidden.shape)).toEqual([[1, 1, 4], [1, 1, 4]])
  const changedIds = runtime.tensor([1, 1, 3], { dtype: 'i64' })
  const changed = runtime.compile(program).run({ ids: changedIds, positions, mask }, state) as Tensor[]
  expect(changed[0].to_array()).not.toEqual(logits.to_array())
  const changedAgain = runtime.compile(program).run({ ids: changedIds, positions, mask }, state) as Tensor[]
  expect(changedAgain[0].to_array()).toEqual(changed[0].to_array())
  const restrictedMask = runtime.tensor([[[[0, 1, 1], [0, 0, 1], [0, 1, 1]]]], { dtype: 'i64' })
  const restricted = runtime.compile(program).run({ ids, positions, mask: restrictedMask }, state) as Tensor[]
  expect(restricted[0].to_array()).not.toEqual(logits.to_array())
  expect(() => model.forward(5)).toThrow()

  for (const tensor of [logits, ...hiddenStates, ...changed, ...changedAgain, ...restricted, ids, changedIds, positions, mask, restrictedMask]) tensor.dispose()
  state.dispose()
  runtime.dispose()
  source.dispose()
})
