import nn from 'affon:nn'
import { add, axes, contiguous, matmul, module as computeModule, reshape, transpose } from 'affon:compute'
import type { Tensor } from 'affon:compute'

import {
  DecoderInputEmbedding,
  type DecoderInputEmbeddingModule,
  type DecoderInputEmbeddingOptions,
} from '../../transformers/src/decoder/embedding.ts'
import {
  DecoderBlock,
  type DecoderBlockOptions,
} from '../../transformers/src/decoder/block.ts'

export interface DecoderModelOptions extends DecoderInputEmbeddingOptions, DecoderBlockOptions {
  numLayers: number
  tieEmbeddings?: boolean
}

export type DecoderModelModule = nn.Module<[Tensor<number[], 'f32'>], Tensor<number[], 'f32'>> & {
  embedding: DecoderInputEmbeddingModule
  blocks: nn.ModuleList
  final_norm: nn.LayerNormLayer<number[], 'f32'>
  lm_head: nn.LinearLayer<number, number, 'f32'> | null
}

function projectHiddenStatesWithLinearHead(
  lm_head: nn.LinearLayer<number, number, 'f32'>,
  hiddenStates: Tensor<[number, number, number], 'f32'>,
): Tensor<number[], 'f32'> {
  const logits = add(
    matmul(hiddenStates, lm_head.weight as Tensor<[number, number], 'f32'>, { hint: 'projection', source: 'higher_level_module' }),
    lm_head.bias as Tensor<[1, number], 'f32'>,
  ) as Tensor<number[], 'f32'>
  nn.diagnostics.assert('finite', { path: 'DecoderModel.projectHiddenStatesWithLinearHead.logits', value: logits })
  return logits
}

function flattenSequenceHiddenStates(
  hiddenStates: Tensor<[number, number, number], 'f32'>,
): { batchSize: number; seqLen: number; flatHiddenStates: Tensor<number[], 'f32'> } {
  const batchSize = hiddenStates.shape[0] as number
  const seqLen = hiddenStates.shape[1] as number
  const contiguousHiddenStates = contiguous(hiddenStates)
  nn.diagnostics.assert('finite', { path: 'DecoderModel.flattenSequenceHiddenStates.contiguous', value: contiguousHiddenStates })

  const flatHiddenStates = reshape(
    contiguousHiddenStates,
    [batchSize * seqLen, hiddenStates.shape[2] as number],
    { axes: [axes.token, axes.feature] },
  )
  nn.diagnostics.assert('finite', { path: 'DecoderModel.flattenSequenceHiddenStates.flat', value: flatHiddenStates })

  return { batchSize, seqLen, flatHiddenStates }
}

function reshapeLogitsToSequence(
  flatLogits: Tensor<number[], 'f32'>,
  batchSize: number,
  seqLen: number,
  outDim: number,
): Tensor<number[], 'f32'> {
  const contiguousLogits = contiguous(flatLogits)
  nn.diagnostics.assert('finite', { path: 'DecoderModel.reshapeLogitsToSequence.contiguous', value: contiguousLogits })

  const sequenceLogits = reshape(contiguousLogits, [batchSize, seqLen, outDim], { axes: [axes.batch, axes.token, axes.vocab] })
  nn.diagnostics.assert('finite', { path: 'DecoderModel.reshapeLogitsToSequence.output', value: sequenceLogits })
  return sequenceLogits
}

function projectHiddenStatesWithTiedEmbedding(
  weight: Tensor<[number, number], 'f32'>,
  hiddenStates: Tensor<[number, number, number], 'f32'>,
): Tensor<number[], 'f32'> {
  const { batchSize, seqLen, flatHiddenStates } = flattenSequenceHiddenStates(hiddenStates)
  const vocabSize = weight.shape[0] as number
  const flatHiddenStatesT = transpose(flatHiddenStates, 0, 1)
  nn.diagnostics.assert('finite', { path: 'DecoderModel.projectHiddenStatesWithTiedEmbedding.hiddenT', value: flatHiddenStatesT })

  // Keep the large embedding table in its original contiguous layout and
  // transpose the smaller activation side to avoid packing a [d_model, vocab]
  // weight view every training step when embeddings are tied.
  const vocabByTokenLogits = matmul(weight, flatHiddenStatesT, { hint: 'projection', source: 'higher_level_module' })
  nn.diagnostics.assert('finite', { path: 'DecoderModel.projectHiddenStatesWithTiedEmbedding.vocabByToken', value: vocabByTokenLogits })

  const flatLogits = transpose(vocabByTokenLogits, 0, 1)
  nn.diagnostics.assert('finite', { path: 'DecoderModel.projectHiddenStatesWithTiedEmbedding.logits', value: flatLogits })
  return reshapeLogitsToSequence(flatLogits as Tensor<number[], 'f32'>, batchSize, seqLen, vocabSize)
}

function runDecoderBlocks(
  blocks: nn.ModuleList,
  hiddenStates: Tensor<number[], 'f32'>,
): Tensor<number[], 'f32'> {
  let currentHiddenStates = hiddenStates

  for (let i = 0; i < blocks.length; i++) {
    currentHiddenStates = blocks[i](currentHiddenStates)
    nn.diagnostics.assert('finite', { path: `DecoderModel.block.${i}`, value: currentHiddenStates })
  }

  return currentHiddenStates
}

function logitsFromHiddenStates(
  hiddenStates: Tensor<number[], 'f32'>,
  final_norm: nn.LayerNormLayer<number[], 'f32'>,
  embedding: DecoderInputEmbeddingModule,
  lm_head: nn.LinearLayer<number, number, 'f32'> | null,
  tieEmbeddings: boolean,
  vocabSize: number,
): Tensor<number[], 'f32'> {
  const normalizedHiddenStates = final_norm(hiddenStates)
  nn.diagnostics.assert('finite', { path: 'DecoderModel.final_norm', value: normalizedHiddenStates })

  if (tieEmbeddings) {
    return projectHiddenStatesWithTiedEmbedding(
      embedding.token_embedding.weight as Tensor<[number, number], 'f32'>,
      normalizedHiddenStates as Tensor<[number, number, number], 'f32'>,
    )
  }

  return projectHiddenStatesWithLinearHead(
    lm_head!,
    normalizedHiddenStates as Tensor<[number, number, number], 'f32'>,
  )
}

export function DecoderModel(
  vocabSize: number,
  dModel: number,
  opts: DecoderModelOptions,
): DecoderModelModule {
  if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
    throw new AffonError('invalid_arg', 'DecoderModel vocabSize must be a positive integer')
  }
  if (!Number.isInteger(dModel) || dModel <= 0) {
    throw new AffonError('invalid_arg', 'DecoderModel dModel must be a positive integer')
  }
  if (!Number.isInteger(opts.numLayers) || opts.numLayers <= 0) {
    throw new AffonError('invalid_arg', 'DecoderModel numLayers must be a positive integer')
  }

  const embedding = DecoderInputEmbedding(vocabSize, dModel, {
    positional: opts.positional,
    maxSeqLen: opts.maxSeqLen,
    dropout: opts.dropout,
  })
  const blocks = nn.module_list(opts.numLayers, (index) => {
    const block = DecoderBlock(dModel, {
      numHeads: opts.numHeads,
      hiddenDim: opts.hiddenDim,
      causal: opts.causal,
      dropout: opts.dropout,
    })
    return block
  })
  const final_norm = nn.LayerNorm<number[], 'f32'>(dModel, { dtype: 'f32' })
  const tieEmbeddings = opts.tieEmbeddings ?? false
  const lm_head = tieEmbeddings
    ? null
    : nn.Linear<number, number, 'f32'>(dModel, vocabSize, { dtype: 'f32' })

  return computeModule({
    embedding,
    blocks,
    final_norm,
    lm_head,
  } as any, function (_state, tokenIds: Tensor<number[], 'f32'>): Tensor<number[], 'f32'> {
      if (tokenIds.ndim !== 2) {
        throw new AffonError('invalid_shape', 'DecoderModel expects token ids shaped [batch, seq]')
      }

      const embeddedTokens = embedding(tokenIds as Tensor<[number, number], 'f32'>)
      nn.diagnostics.assert('finite', { path: 'DecoderModel.embedding', value: embeddedTokens })

      const hiddenStates = runDecoderBlocks(blocks, embeddedTokens as Tensor<number[], 'f32'>)
      const logits = logitsFromHiddenStates(
        hiddenStates,
        final_norm,
        embedding,
        lm_head,
        tieEmbeddings,
        vocabSize,
      )
      nn.diagnostics.assert('finite', { path: 'DecoderModel.logits', value: logits })
      return logits
  }).metadata('decoder') as DecoderModelModule
}
