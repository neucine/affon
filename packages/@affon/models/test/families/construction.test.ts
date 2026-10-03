import { expect, test } from 'std:test'
import { Session } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { create_gpt2, create_bert, create_vit } from '../../src/index.ts'

function fixture() {
  const session = new Session({ device: 'cpu' })
  const value = (shape: number[], fill = 0.1): Tensor => {
    const data = (dims: number[]): any => dims.length ? Array.from({ length: dims[0] }, () => data(dims.slice(1))) : fill
    return session.tensor(data(shape)) as Tensor
  }
  const affine = (input: number, output: number) => ({ weight: value([input, output]), bias: value([output], 0) })
  const norm = () => ({ weight: value([4], 1), bias: value([4], 0) })
  const block = () => ({ query: affine(4, 4), key: affine(4, 4), value: affine(4, 4), attentionOutput: affine(4, 4), attentionNorm: norm(), feedForwardNorm: norm(), expand: affine(4, 8), contract: affine(8, 4) })
  return { session, value, affine, norm, block }
}

test('constructs GPT-2 from the family barrel and validates parameter shapes', () => {
  const { session, value, affine, norm } = fixture()
  const config = { width: 4, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, epsilon: 1e-5 }
  const weights = { tokenEmbedding: value([7, 4]), positionEmbedding: value([4, 4]), finalNorm: norm(), blocks: [{ attentionNorm: norm(), feedForwardNorm: norm(), qkv: affine(4, 12), attentionOutput: affine(4, 4), expand: affine(4, 8), contract: affine(8, 4) }] }
  const model = create_gpt2(config, weights)
  const result = model.forward([1, 2])
  expect(result.logits.shape).toEqual([1, 2, 7])
  expect(() => create_gpt2(config, { ...weights, tokenEmbedding: value([6, 4]) })).toThrow()
  result.logits.dispose(); for (const hidden of result.hidden_states) hidden.dispose()
  model.dispose(); session.dispose()
})

test('constructs BERT from the family barrel and validates masks', () => {
  const { session, value, affine, norm, block } = fixture()
  const config = { width: 4, innerWidth: 8, heads: 2, layers: 1, contextLength: 4, vocabSize: 7, typeVocabSize: 2, epsilon: 1e-5 }
  const model = create_bert(config, { tokenEmbedding: value([7, 4]), positionEmbedding: value([4, 4]), typeEmbedding: value([2, 4]), embeddingNorm: norm(), pooler: affine(4, 4), blocks: [block()] })
  const result = model.forward([[1, 2]], [[1, 1]], [[0, 0]])
  expect(result.output.shape).toEqual([1, 2, 4])
  expect(() => model.forward([[1]], [[0]], [[0]])).toThrow()
  for (const tensor of [result.pooled, result.pooler, ...result.hidden_states]) tensor.dispose()
  model.dispose(); session.dispose()
})

test('constructs ViT from the family barrel and rejects incompatible patches', () => {
  const { session, value, affine, norm, block } = fixture()
  const config = { width: 4, innerWidth: 8, heads: 2, layers: 1, imageSize: 4, patchSize: 2, epsilon: 1e-5 }
  const weights = { classToken: value([1, 1, 4]), positionEmbedding: value([1, 5, 4]), patchProjection: { weight: value([4, 3, 2, 2]), bias: value([4]) }, finalNorm: norm(), classifier: affine(4, 3), blocks: [block()] }
  const model = create_vit(config, weights)
  const pixels = value([1, 3, 4, 4])
  const result = model.forward(pixels)
  expect(result.output.shape).toEqual([1, 3])
  expect(() => create_vit({ ...config, patchSize: 3 }, weights)).toThrow()
  result.output.dispose(); for (const hidden of result.hidden_states) hidden.dispose()
  pixels.dispose(); model.dispose(); session.dispose()
})
