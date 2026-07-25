import nn from 'affon:nn'
import { add, axes, cast, module as computeModule, unsqueeze } from 'affon:compute'
import type { DType, Tensor } from 'affon:compute'

import { position_ids, sinusoidal_encoding } from '../sequence.ts'

export interface DecoderInputEmbeddingOptions {
  positional?: 'none' | 'sinusoidal' | 'learned'
  maxSeqLen?: number
  dropout?: number
}

export type DecoderInputEmbeddingModule = nn.Module<[Tensor<[number, number], 'f32'>], Tensor<number[], 'f32'>> & {
  token_embedding: nn.EmbeddingLayer<number, number, 'f32'>
  position_embedding?: nn.EmbeddingLayer<number, number, 'f32'>
  dropout?: nn.DropoutLayer<number[], 'f32'>
}

function graphSafeDevice<S extends number[], T extends DType>(value: Tensor<S, T>): 'cpu' | 'metal' | undefined {
  try {
    return value.device as 'cpu' | 'metal'
  } catch {
    return undefined
  }
}

function alignIdsDType<S extends number[]>(
  ids: Tensor<S, DType>,
  like: Tensor<number[], DType>,
): Tensor<S, 'f32'> {
  const idsDevice = graphSafeDevice(ids)
  const likeDevice = graphSafeDevice(like)
  if (ids.dtype === like.dtype) {
    if (idsDevice == null || likeDevice == null || idsDevice === likeDevice) {
      return ids as Tensor<S, 'f32'>
    }
    return ids.to(likeDevice) as Tensor<S, 'f32'>
  }
  const casted = cast(ids, like.dtype)
  if (likeDevice == null) return casted as Tensor<S, 'f32'>
  return casted.to(likeDevice) as Tensor<S, 'f32'>
}

export function DecoderInputEmbedding(
  vocabSize: number,
  dModel: number,
  opts?: DecoderInputEmbeddingOptions,
): DecoderInputEmbeddingModule {
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new AffonError('invalid_arg', 'DecoderInputEmbedding vocabSize must be a positive integer')
  }
  if (!Number.isInteger(dModel) || dModel <= 0) {
    throw new AffonError('invalid_arg', 'DecoderInputEmbedding dModel must be a positive integer')
  }
  const positional = opts?.positional ?? 'learned'
  const maxSeqLen = opts?.maxSeqLen ?? 2048
  const dropout = opts?.dropout ?? 0
  if (positional === 'learned' && (!Number.isInteger(maxSeqLen) || maxSeqLen <= 0)) {
    throw new AffonError('invalid_arg', 'DecoderInputEmbedding maxSeqLen must be a positive integer')
  }
  if (!Number.isFinite(dropout) || dropout < 0 || dropout >= 1) {
    throw new AffonError('invalid_arg', 'DecoderInputEmbedding dropout must be in the range [0, 1)')
  }

  const token_embedding = nn.Embedding<number, number, 'f32'>(vocabSize, dModel, {
    dtype: 'f32',
    axes: [axes.vocab, axes.feature],
  })
  const position_embedding = positional === 'learned'
    ? nn.Embedding<number, number, 'f32'>(maxSeqLen, dModel, { dtype: 'f32', axes: [axes.token, axes.feature] })
    : undefined
  const dropoutLayer = dropout > 0 ? nn.Dropout<number[], 'f32'>(dropout) : undefined

	  return computeModule({
	    token_embedding,
	    position_embedding,
	    dropout: dropoutLayer,
	  } as any, function (_state, tokenIds: Tensor<[number, number], 'f32'>): Tensor<number[], 'f32'> {
      if (tokenIds.ndim !== 2) {
        throw new AffonError('invalid_shape', 'DecoderInputEmbedding expects token ids shaped [batch, seq]')
      }
      const batch = tokenIds.shape[0]
      const seqLen = tokenIds.shape[1]
      let alignedTokenIds: Tensor<[number, number], 'f32'> | null = alignIdsDType(tokenIds, token_embedding.weight)
      let tokenVectors: Tensor<number[], 'f32'> | null = token_embedding(alignedTokenIds)
      alignedTokenIds = null
      nn.diagnostics.assert('finite', { path: 'DecoderInputEmbedding.token_embedding', value: tokenVectors })
      if (positional === 'none') return tokenVectors as Tensor<number[], 'f32'>

      let pos: Tensor<[number], 'f32'> | null = position_ids(seqLen as number)
      let posVectors: Tensor<number[], 'f32'> | null = positional === 'learned'
        ? position_embedding!(alignIdsDType(pos, position_embedding!.weight))
        : sinusoidal_encoding(seqLen as number, dModel) as Tensor<number[], 'f32'>
      pos = null
      let expanded: Tensor<number[], 'f32'> | null = unsqueeze(posVectors as Tensor<number[], 'f32'>, 0)
      posVectors = null
      nn.diagnostics.assert('finite', { path: 'DecoderInputEmbedding.position_embedding', value: expanded })
	      const out = add(tokenVectors, expanded) as Tensor<number[], 'f32'>
      tokenVectors = null
      expanded = null
      nn.diagnostics.assert('finite', { path: 'DecoderInputEmbedding.output', value: out })
	      const withDropout = dropoutLayer ? dropoutLayer(out) as Tensor<number[], 'f32'> : out
      nn.diagnostics.assert('finite', { path: 'DecoderInputEmbedding.dropout', value: withDropout })
      if (withDropout.shape[0] !== batch || withDropout.shape[1] !== seqLen || withDropout.shape[2] !== dModel) {
        throw new AffonError('internal', 'DecoderInputEmbedding produced an unexpected shape')
      }
      return withDropout as Tensor<number[], 'f32'>
  }) as DecoderInputEmbeddingModule
}
