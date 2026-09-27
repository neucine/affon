import { tensor } from 'affon:compute'
import type { Tensor } from 'affon:compute'

import {
  DecoderBlock,
  DecoderInputEmbedding,
  FeedForward,
  SelfAttention,
  apply_causal_mask,
  causal_mask,
  position_ids,
  sinusoidal_encoding,
} from '../../src/index.ts'

type IsAssignable<A, B> = A extends B ? true : false
type HasDType<T, D extends 'f32' | 'f64'> = T extends Tensor<any, D> ? true : false
function assertType<T extends true>() {}

const mask = causal_mask(4, { dtype: 'f32' })
assertType<IsAssignable<typeof mask, Tensor<[number, number], 'f32' | 'f64'>>>()
assertType<IsAssignable<typeof mask.shape, [number, number]>>()

const masked = apply_causal_mask(tensor([[1, 2], [3, 4]], { dtype: 'f32' }))
assertType<IsAssignable<typeof masked, Tensor<number[], 'f32' | 'f64'>>>()

const pos = position_ids(8)
assertType<HasDType<typeof pos, 'f32'>>()

const sin = sinusoidal_encoding(8, 16, { dtype: 'f32' })
assertType<IsAssignable<typeof sin, Tensor<[number, number], 'f32' | 'f64'>>>()

const embed = DecoderInputEmbedding(32000, 16, { positional: 'learned', maxSeqLen: 64 })
const tokenIds = tensor([
  [1, 2, 3],
  [4, 5, 6],
], { dtype: 'f32' })
const tokenIds2d = tokenIds as Tensor<[number, number], 'f32'>
const embedded = embed(tokenIds2d)
assertType<HasDType<typeof embedded, 'f32'>>()
assertType<IsAssignable<typeof embed.module_path, string | null>>()
assertType<IsAssignable<typeof embed.parameters['length'], number>>()

const ff = FeedForward(16, { hiddenDim: 64 })
const ffOut = ff(embedded)
assertType<HasDType<typeof ffOut, 'f32'>>()

const attn = SelfAttention(16, { numHeads: 4, causal: true })
const attnOut = attn(embedded)
assertType<HasDType<typeof attnOut, 'f32'>>()

const block = DecoderBlock(16, { numHeads: 4, hiddenDim: 64, causal: true })
const blockOut = block(embedded)
assertType<HasDType<typeof blockOut, 'f32'>>()
