import nn from 'affon:nn'
import { add, axes, module as computeModule, reshape } from 'affon:compute'
import type { Tensor } from 'affon:compute'

import { FeedForward, type FeedForwardModule, type FeedForwardOptions } from './feedforward.ts'
import { SelfAttention, type SelfAttentionModule, type SelfAttentionOptions } from './attention.ts'

export interface DecoderBlockOptions extends FeedForwardOptions, SelfAttentionOptions {
  dropout?: number
}

export type DecoderBlockModule = nn.Module<[Tensor<number[], 'f32'>], Tensor<number[], 'f32'>> & {
  attn_norm: nn.LayerNormLayer<number[], 'f32'>
  self_attention: SelfAttentionModule
  attn_dropout?: nn.DropoutLayer<number[], 'f32'>
  ff_norm: nn.LayerNormLayer<number[], 'f32'>
  feed_forward: FeedForwardModule
  ff_dropout?: nn.DropoutLayer<number[], 'f32'>
}

function withSequenceAxes(value: Tensor<number[], 'f32'>): Tensor<number[], 'f32'> {
  return reshape(value, value.shape as number[], { axes: [axes.batch, axes.token, axes.feature] }) as Tensor<number[], 'f32'>
}

export function DecoderBlock(
  dModel: number,
  opts: DecoderBlockOptions,
): DecoderBlockModule {
  if (!Number.isInteger(dModel) || dModel <= 0) {
    throw new AffonError('invalid_arg', 'DecoderBlock dModel must be a positive integer')
  }

  const attn_norm = nn.LayerNorm<number[], 'f32'>(dModel, { dtype: 'f32' })
  const self_attention = SelfAttention(dModel, {
    numHeads: opts.numHeads,
    causal: opts.causal,
  })
  const dropout = opts.dropout ?? 0
  if (!Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
    throw new AffonError('invalid_arg', 'DecoderBlock dropout must be in the range [0, 1)')
  }
  const attn_dropout = dropout > 0 ? nn.Dropout<number[], 'f32'>(dropout) : undefined
  const ff_norm = nn.LayerNorm<number[], 'f32'>(dModel, { dtype: 'f32' })
  const feed_forward = FeedForward(dModel, {
    hiddenDim: opts.hiddenDim,
  })
  const ff_dropout = dropout > 0 ? nn.Dropout<number[], 'f32'>(dropout) : undefined

	  return computeModule({
	    attn_norm,
	    self_attention,
	    attn_dropout,
	    ff_norm,
	    feed_forward,
	    ff_dropout,
	  } as any, function (_state, x: Tensor<number[], 'f32'>): Tensor<number[], 'f32'> {
      if (x.ndim !== 3) {
        throw new AffonError('invalid_shape', 'DecoderBlock expects input shaped [batch, seq, d_model]')
      }
      if (x.shape[2] !== dModel) {
        throw new AffonError('shape_mismatch', 'DecoderBlock input last dimension must equal dModel')
      }

	      let attn_normOut: Tensor<number[], 'f32'> | null = withSequenceAxes(attn_norm(x) as Tensor<number[], 'f32'>)
	      nn.diagnostics.assert('finite', { path: 'DecoderBlock.attn_norm', value: attn_normOut })
	      let attnOut: Tensor<number[], 'f32'> | null = self_attention(attn_normOut as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>
	      attn_normOut = null
	      nn.diagnostics.assert('finite', { path: 'DecoderBlock.self_attention', value: attnOut })
	      if (attn_dropout) {
	        attnOut = attn_dropout(attnOut as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>
	        nn.diagnostics.assert('finite', { path: 'DecoderBlock.attentionDropout', value: attnOut })
	      }
	      let afterAttn: Tensor<number[], 'f32'> | null = withSequenceAxes(add(x, attnOut as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>)
	      attnOut = null
	      nn.diagnostics.assert('finite', { path: 'DecoderBlock.afterAttentionResidual', value: afterAttn })
	      let ff_normOut: Tensor<number[], 'f32'> | null = withSequenceAxes(ff_norm(afterAttn as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>)
	      nn.diagnostics.assert('finite', { path: 'DecoderBlock.ff_norm', value: ff_normOut })
	      let ffOut: Tensor<number[], 'f32'> | null = feed_forward(ff_normOut as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>
	      ff_normOut = null
	      nn.diagnostics.assert('finite', { path: 'DecoderBlock.feed_forward', value: ffOut })
	      if (ff_dropout) {
	        ffOut = ff_dropout(ffOut as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>
	        nn.diagnostics.assert('finite', { path: 'DecoderBlock.feed_forwardDropout', value: ffOut })
	      }
	      const out = withSequenceAxes(add(afterAttn as Tensor<number[], 'f32'>, ffOut as Tensor<number[], 'f32'>) as Tensor<number[], 'f32'>)
      afterAttn = null
      ffOut = null
      nn.diagnostics.assert('finite', { path: 'DecoderBlock.output', value: out })
      return out
  }) as DecoderBlockModule
}
