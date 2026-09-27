import { expect, test } from 'std:test'
import { tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_gpt2, create_bert, create_vit } from '../../src/index.ts'

function value(shape: number[], fill = 0.1): Tensor {
  const data = (dims: number[]): any => dims.length ? Array.from({ length: dims[0] }, () => data(dims.slice(1))) : fill
  return tensor(data(shape), { dtype: 'f32', device: 'cpu' })
}
const affine = (input: number, output: number) => ({ weight: value([input, output]), bias: value([output], 0) })
const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
const block = () => ({ query: affine(4, 4), key: affine(4, 4), value: affine(4, 4), attentionOutput: affine(4, 4), attentionNorm: norm(), feedForwardNorm: norm(), expand: affine(4, 8), contract: affine(8, 4) })

test('constructs GPT-2 directly without HF artifacts and owns independent decode sessions', () => {
  const config = { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 }
  const weights = { tokenEmbedding: value([7, 4]), positionEmbedding: value([4, 4]), finalNorm: norm(), blocks: [{ attentionNorm: norm(), feedForwardNorm: norm(), qkv: affine(4, 12), attentionOutput: affine(4, 4), expand: affine(4, 8), contract: affine(8, 4) }] }
  const model = create_gpt2(config, weights)
  const a = model.create_session(), b = model.create_session()
  expect(a.forward([1, 2]).logits.shape).toEqual([1, 2, 7])
  expect(a.length).toBe(2); expect(b.length).toBe(0)
  expect(model.generate([1], 2)).toEqual([1, 0, 0])
  expect(() => create_gpt2(config, { ...weights, tokenEmbedding: value([6, 4]) })).toThrow()
})

test('constructs BERT from model parameters and validates masks and tensor shapes', () => {
  const config = { width: 4, innerWidth: 8, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, typeVocabSize: 2, epsilon: 1e-5 }
  const weights = { tokenEmbedding: value([7, 4]), positionEmbedding: value([4, 4]), typeEmbedding: value([2, 4]), embeddingNorm: norm(), pooler: affine(4, 4), blocks: [block()] }
  const model = create_bert(config, weights)
  expect(model.forward([[1, 2]], [[1, 1]], [[0, 0]]).output.shape).toEqual([1, 2, 4])
  expect(() => model.forward([[1]], [[0]], [[0]])).toThrow()
  expect(() => create_bert(config, { ...weights, pooler: affine(4, 3) })).toThrow()
})

test('constructs ViT independently of image processing and rejects incompatible patches', () => {
  const config = { width: 4, innerWidth: 8, heads: 2, layers: 1, imageSize: 4, patchSize: 2, epsilon: 1e-5 }
  const weights = { classToken: value([1, 1, 4]), positionEmbedding: value([1, 5, 4]), patchProjection: { weight: value([4, 3, 2, 2]), bias: value([4]) }, finalNorm: norm(), classifier: affine(4, 3), blocks: [block()] }
  expect(create_vit(config, weights).forward(value([1, 3, 4, 4])).output.shape).toEqual([1, 3])
  expect(() => create_vit({ ...config, patchSize: 3 }, weights)).toThrow()
  expect(() => create_vit(config, { ...weights, positionEmbedding: value([1, 4, 4]) })).toThrow()
})
