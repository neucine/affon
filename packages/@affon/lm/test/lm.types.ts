import { tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'
import { createLookupTokenizer } from '../../tokenizers/src/index.ts'

import {
  CausalLMLoss,
  DecoderModel,
  createPackedTextCorpusFromConfig,
  generate,
  loadTokenRows,
  pack_token_windows,
  saveTokenRows,
} from '../src/index.ts'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
type HasDType<T, D extends 'f32' | 'f64'> = T extends Tensor<any, D> ? true : false
function assertType<T extends true>() {}

const tokenizer = createLookupTokenizer({
  '<bos>': 0,
  '<eos>': 1,
  '<unk>': 2,
  hello: 3,
}, {
  specialTokens: { bos: '<bos>', eos: '<eos>', unk: '<unk>' },
})

const packedFromConfig = createPackedTextCorpusFromConfig(tokenizer, {
  paths: ['train-a.txt', 'train-b.txt'],
  validationPaths: ['val.txt'],
  addBos: true,
  addEos: true,
  seqLen: 3,
  stride: 1,
  tokenCacheKey: 'train-v1',
  preprocess: {
    replacements: [{ from: '<unk>', to: '' }],
    normalizeWhitespace: true,
  },
})
assertType<IsExact<typeof packedFromConfig.validationRows, number[][]>>()
saveTokenRows('cache.json', packedFromConfig.trainRows, { key: 'train-v1' })
const loadedTokenRows = loadTokenRows('cache.json')
assertType<IsExact<typeof loadedTokenRows, number[][]>>()

const tokenWindows = pack_token_windows([
  [1, 2, 3, 4],
  [4, 5, 6, 7],
], {
  seqLen: 3,
  stride: 1,
})
assertType<IsExact<typeof tokenWindows, number[][]>>()

const model = DecoderModel(32000, 16, {
  numLayers: 2,
  numHeads: 4,
  hiddenDim: 64,
  causal: true,
  positional: 'learned',
  maxSeqLen: 64,
})
const tokenIds = tensor([
  [1, 2, 3],
  [4, 5, 6],
], { dtype: 'f32' }) as Tensor<[number, number], 'f32'>
const logits = model(tokenIds)
assertType<HasDType<typeof logits, 'f32'>>()

const criterion = CausalLMLoss()
const lossTokens = tensor([
  [1, 2, 3, 4],
  [4, 5, 6, 7],
], { dtype: 'f32' })
const lossInputs = lossTokens.slice([':', `0:${lossTokens.shape[1] as number - 1}`])
const loss = criterion(model(lossInputs as any), lossTokens)
assertType<IsExact<typeof loss, Tensor<[1], 'f32'>>>()

const generated = generate(model, tokenIds, { max_new_tokens: 2 })
assertType<HasDType<typeof generated, 'f32'>>()

const generatedWithForbidden = generate(model, tokenIds, { max_new_tokens: 2, forbidden_token_ids: [0, 1] })
assertType<HasDType<typeof generatedWithForbidden, 'f32'>>()

const generatedWithSampling = generate(model, tokenIds, { max_new_tokens: 2, temperature: 0.8, top_k: 5 })
assertType<HasDType<typeof generatedWithSampling, 'f32'>>()
