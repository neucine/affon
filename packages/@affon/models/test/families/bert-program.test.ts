import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import { create_bert } from '../../src/bert/model.ts'
import type { Tensor } from 'affon:compute'

for (const dtype of ['f32', 'bf16'] as const) test(`runs BERT ${dtype} parameters through Program and Session`, () => {
  const source = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dimensions: number[]): any => dimensions.length
      ? Array.from({ length: dimensions[0] }, () => data(dimensions.slice(1)))
      : fill
    return source.tensor(data(shape), { dtype }) as Tensor
  }
  const affine = (input: number, output: number) => ({ weight: value([input, output]), bias: value([output], 0) })
  const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
  const projection = () => ({ weight: source.tensor(Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) => row === column ? 1 : 0)), { dtype }), bias: value([4], 0) })
  const block = () => ({ query: projection(), key: projection(), value: projection(), attentionOutput: projection(), attentionNorm: norm(), feedForwardNorm: norm(), expand: affine(4, 8), contract: affine(8, 4) })
  const config = { width: 4, innerWidth: 8, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, typeVocabSize: 2, epsilon: 1e-5 }
  const tokenEmbedding = source.tensor(Array.from({ length: 7 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) => column === row % 4 ? 1 : 0)), { dtype })
  const model = create_bert(config, { tokenEmbedding, positionEmbedding: value([4, 4]), typeEmbedding: value([2, 4]), embeddingNorm: norm(), pooler: affine(4, 4), blocks: [block()] } as any)

  const runtime = new Session({ device: 'cpu' })
  const program = model.forward(1, 2)
  const state = runtime.initialize(program, { parameters: model.parameters })
  const inputs = {
    ids: runtime.tensor([[1, 2]], { dtype: 'i64' }),
    types: runtime.tensor([[0, 0]], { dtype: 'i64' }),
    positions: runtime.tensor([[0, 1]], { dtype: 'i64' }),
    valid: runtime.tensor([[[1], [1]]]),
    attention_mask: runtime.tensor([[[[0, 0]]]], { dtype: 'i64' }),
  }
  const [pooled, pooler, ...hiddenStates] = runtime.compile(program).run(inputs, state) as Tensor[]
  expect(hiddenStates.at(-1)!.shape).toEqual([1, 2, 4])
  expect(pooled.shape).toEqual([1, 4])
  expect(pooler.shape).toEqual([1, 4])
  expect((hiddenStates.at(-1)!.to_array() as number[][][]).flat(Infinity).every(Number.isFinite)).toBe(true)

  const changedIds = runtime.tensor([[2, 1]], { dtype: 'i64' })
  const changed = runtime.compile(program).run({ ...inputs, ids: changedIds }, state) as Tensor[]
  expect(changed.at(-1)!.to_array()).not.toEqual(hiddenStates.at(-1)!.to_array())
  const changedAgain = runtime.compile(program).run({ ...inputs, ids: changedIds }, state) as Tensor[]
  expect(changedAgain.at(-1)!.to_array()).toEqual(changed.at(-1)!.to_array())
  const masked = runtime.tensor([[[[0, 1]]]], { dtype: 'i64' })
  const maskedResult = runtime.compile(program).run({ ...inputs, attention_mask: masked }, state) as Tensor[]
  expect(maskedResult.at(-1)!.to_array()).not.toEqual(hiddenStates.at(-1)!.to_array())
  masked.dispose()
  for (const tensor of maskedResult) tensor.dispose()
  changedIds.dispose()
  for (const tensor of [...changed, ...changedAgain]) tensor.dispose()

  for (const tensor of [pooled, pooler, ...hiddenStates, ...Object.values(inputs)]) tensor.dispose()
  state.dispose()
  runtime.dispose()
  source.dispose()
})
