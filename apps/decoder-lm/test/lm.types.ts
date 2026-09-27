import { tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'

import { DecoderModel } from '../src/model.ts'
import { CausalLMLoss, generate } from '../src/causal-lm.ts'

type IsExact<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false
type HasDType<T, D extends 'f32' | 'f64'> = T extends Tensor<any, D> ? true : false
function assertType<T extends true>() {}

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

const cudaOrdinalTokenIds = tensor([[1, 2, 3]], {
  dtype: 'f32',
  device: 'cuda:1',
})
const generatedOnCudaOrdinal = generate(model, cudaOrdinalTokenIds, {
  max_new_tokens: 1,
  forbidden_token_ids: [0],
})
assertType<HasDType<typeof generatedOnCudaOrdinal, 'f32'>>()
