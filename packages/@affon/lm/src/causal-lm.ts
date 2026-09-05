import nn from 'affon:nn'
import {
  add,
  argmax,
  axes,
  cast,
  cat,
  clamp,
  contiguous,
  div,
  exp,
  gather,
  log,
  max,
  mean,
  move,
  neg,
  no_grad,
  one_hot,
  rand,
  reshape,
  masked_fill,
  squeeze,
  sub,
  sum,
  tensor,
  topk,
  unsqueeze,
} from 'affon:compute'
import type { DType, Device, Tensor } from 'affon:compute'

import type { DecoderModelModule } from './model.ts'

interface CausalLMShape {
  batchSize: number
  inputSeqLen: number
  targetSeqLen: number
  vocabSize: number
}

export type DecoderLMForward = (tokenIds: Tensor<[number, number], 'f32'>) => Tensor<number[], 'f32'>

export interface GenerateOptions {
  max_new_tokens?: number
  forbidden_token_ids?: number[]
  temperature?: number
  top_k?: number
}

const FORBIDDEN_LOGIT = -1e30

interface ResolvedGenerateOptions {
  max_new_tokens: number
  forbidden_token_ids?: number[]
  temperature: number
  top_k?: number
}

function alignTokenIdsToLogits<S extends number[]>(
  ids: Tensor<S, DType>,
  logits: Tensor<number[], DType>,
): Tensor<S, 'f32'> {
  if (ids.dtype === logits.dtype && ids.device === logits.device) {
    return ids as Tensor<S, 'f32'>
  }
  if (ids.dtype === logits.dtype) {
    return move(ids, logits.device) as Tensor<S, 'f32'>
  }
  return move(cast(ids, logits.dtype), logits.device) as Tensor<S, 'f32'>
}

function resolveGenerationOptions(opts: GenerateOptions): ResolvedGenerateOptions {
  if (!Number.isInteger(opts.max_new_tokens) || opts.max_new_tokens! <= 0) {
    throw new AffonError('invalid_arg', 'generate max_new_tokens must be a positive integer')
  }

  const temperature = opts.temperature ?? 0
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new AffonError('invalid_arg', 'generate temperature must be a non-negative number')
  }

  if (opts.top_k !== undefined && (!Number.isInteger(opts.top_k) || opts.top_k <= 0)) {
    throw new AffonError('invalid_arg', 'generate top_k must be a positive integer')
  }

  return {
    max_new_tokens: opts.max_new_tokens!,
    forbidden_token_ids: opts.forbidden_token_ids,
    temperature,
    top_k: opts.top_k,
  }
}

function assertCausalLMInputs(
  logits: Tensor<number[], 'f32'>,
  tokenIds: Tensor<number[], 'f32'>,
): CausalLMShape {
  if (logits.ndim !== 3) {
    throw new AffonError('invalid_shape', 'CausalLMLoss expects logits shaped [batch, seq, vocab]')
  }
  if (tokenIds.ndim !== 2) {
    throw new AffonError('invalid_shape', 'CausalLMLoss expects token ids shaped [batch, seq]')
  }

  const batchSize = logits.shape[0] as number
  const inputSeqLen = logits.shape[1] as number
  const vocabSize = logits.shape[2] as number
  const targetBatchSize = tokenIds.shape[0] as number
  const targetSeqLen = tokenIds.shape[1] as number

  if (targetBatchSize !== batchSize) {
    throw new AffonError('shape_mismatch', 'CausalLMLoss token ids batch must match logits batch')
  }
  if (targetSeqLen !== inputSeqLen + 1) {
    throw new AffonError(
      'shape_mismatch',
      'CausalLMLoss token ids sequence length must equal logits sequence length + 1',
    )
  }

  return { batchSize, inputSeqLen, targetSeqLen, vocabSize }
}

function nextTokenTargets(
  tokenIds: Tensor<number[], 'f32'>,
  shape: CausalLMShape,
): Tensor<number[], 'f32'> {
  return tokenIds.slice([':', `1:${shape.targetSeqLen}`]) as Tensor<number[], 'f32'>
}

function flattenSequenceLogits(
  logits: Tensor<number[], 'f32'>,
  shape: CausalLMShape,
): Tensor<[number, number], 'f32'> {
  return reshape(logits, [shape.batchSize * shape.inputSeqLen, shape.vocabSize], { axes: [axes.token, axes.vocab] }) as Tensor<[number, number], 'f32'>
}

function flattenTargetIndices(
  targets: Tensor<number[], 'f32'>,
  shape: CausalLMShape,
): Tensor<[number, 1], 'f32'> {
  return reshape(contiguous(targets), [shape.batchSize * shape.inputSeqLen, 1], { axes: [axes.token, axes.vocab] }) as Tensor<[number, 1], 'f32'>
}

function candidateLogitsFromLastStep(
  logits: Tensor<number[], 'f32'>,
): Tensor<number[], 'f32'> {
  const lastStepIndex = (logits.shape[1] as number) - 1
  return squeeze(logits.slice([':', `${lastStepIndex}:${lastStepIndex + 1}`, ':']), 1) as Tensor<number[], 'f32'>
}

function validateForbiddenTokenIds(
  forbidden_token_ids: readonly number[] | undefined,
  vocabSize: number,
): Set<number> {
  if (!forbidden_token_ids || forbidden_token_ids.length === 0) {
    return new Set()
  }

  for (const tokenId of forbidden_token_ids) {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
      throw new AffonError(
        'invalid_arg',
        'generate forbidden_token_ids must be valid token ids for the model vocab',
      )
    }
  }

  return new Set(forbidden_token_ids)
}

function forbiddenTokenMask(
  forbidden_token_ids: ReadonlySet<number>,
  vocabSize: number,
  device: Device,
): Tensor<[1, number], 'i64'> | null {
  if (forbidden_token_ids.size === 0) return null
  if (forbidden_token_ids.size >= vocabSize) {
    throw new AffonError('invalid_arg', 'generate forbidden_token_ids excludes the full vocabulary')
  }

  const forbidden_ids = tensor(Array.from(forbidden_token_ids), { dtype: 'f32', device })
  const hot = one_hot(forbidden_ids, vocabSize)
  const collapsed = sum(hot, 0, false)
  const mask = cast(collapsed, 'i64') as Tensor<[number], 'i64'>
  return unsqueeze(mask, 0) as Tensor<[1, number], 'i64'>
}

function sampleNextTokenTensor(
  logits: Tensor<number[], 'f32'>,
  temperature: number,
  top_k?: number,
): Tensor<number[], 'f32'> {
  const scaledLogits = div(
    logits,
    tensor(temperature, { dtype: logits.dtype, device: logits.device }),
  ) as Tensor<number[], 'f32'>

  if (top_k !== undefined) {
    const vocabSize = logits.shape[1] as number
    const effective_k = Math.min(top_k, vocabSize)
    const result = topk(scaledLogits, effective_k, 1)
    const clamped = clamp(
      rand(result.values.shape as number[], { dtype: result.values.dtype, device: result.values.device }),
      1e-6,
      1 - 1e-6,
    )
    const gumbel = neg(log(neg(log(clamped)))) as Tensor<number[], 'f32'>
    const sampledPositions = argmax(add(result.values, gumbel), 1, false) as Tensor<number[], 'i64'>
    // Token outputs are f32; converting here also supports CUDA's f32 gather.
    const selected = gather(
      cast(result.indices, 'f32'),
      1,
      reshape(sampledPositions, [logits.shape[0] as number, 1], { axes: [axes.batch, axes.vocab] }),
    ) as Tensor<[number, 1], 'f32'>
    return squeeze(selected, 1) as Tensor<number[], 'f32'>
  }

  const clamped = clamp(
    rand(logits.shape as number[], { dtype: logits.dtype, device: logits.device }),
    1e-6,
    1 - 1e-6,
  )
  const gumbel = neg(log(neg(log(clamped)))) as Tensor<number[], 'f32'>
  return cast(argmax(add(scaledLogits, gumbel), 1, false), 'f32') as Tensor<number[], 'f32'>
}

function appendNextTokenTensor(
  current: Tensor<number[], 'f32'>,
  nextTokenIds: Tensor<number[], 'f32'>,
): Tensor<number[], 'f32'> {
  const batchSize = current.shape[0] as number
  const nextColumn = reshape(nextTokenIds, [batchSize, 1], { axes: [axes.batch, axes.token] }) as Tensor<[number, number], 'f32'>

  if (nextColumn.dtype === current.dtype && nextColumn.device === current.device) {
    return cat([current, nextColumn], 1) as Tensor<number[], 'f32'>
  }
  return cat([current, move(nextColumn, current.device)], 1) as Tensor<number[], 'f32'>
}

function teacherForcedCrossEntropy(
  logits: Tensor<number[], 'f32'>,
  tokenIds: Tensor<number[], 'f32'>,
): Tensor<[1], 'f32'> {
  const shape = assertCausalLMInputs(logits, tokenIds)
  const targets = nextTokenTargets(tokenIds, shape)
  const alignedTargets = alignTokenIdsToLogits(targets, logits)
  const flatLogits = flattenSequenceLogits(logits, shape)
  const targetIndex = flattenTargetIndices(alignedTargets, shape)

  const selectedLogits = gather(flatLogits, 1, targetIndex) as Tensor<number[], 'f32'>
  const rowMax = max(flatLogits, 1, true) as Tensor<number[], 'f32'>
  const shiftedLogits = sub(flatLogits, rowMax)
  const rowExpSum = sum(exp(shiftedLogits), 1, true) as Tensor<number[], 'f32'>
  const rowLogSumExp = add(log(rowExpSum), rowMax)

  return mean(sub(rowLogSumExp, selectedLogits)) as Tensor<[1], 'f32'>
}

function indexedCrossEntropyIfAvailable(
  logits: Tensor<number[], 'f32'>,
  tokenIds: Tensor<number[], 'f32'>,
): Tensor<[1], 'f32'> | null {
  if (logits.ndim !== 3 || tokenIds.ndim !== 2) return null

  const batchSize = logits.shape[0] as number
  const inputSeqLen = logits.shape[1] as number
  const vocabSize = logits.shape[2] as number
  const targetBatchSize = tokenIds.shape[0] as number
  const targetSeqLen = tokenIds.shape[1] as number

  if (targetBatchSize !== batchSize || targetSeqLen !== inputSeqLen + 1) return null

  const targets = tokenIds.slice([':', `1:${targetSeqLen}`]) as Tensor<number[], 'f32'>
  const flatLogits = reshape(logits, [batchSize * inputSeqLen, vocabSize], { axes: [axes.token, axes.vocab] }) as Tensor<[number, number], 'f32'>
  const flatTargets = reshape(contiguous(targets), [batchSize * inputSeqLen], { axes: [axes.token] }) as Tensor<[number], 'f32'>
  return nn.CrossEntropyLoss({ target: 'index' })(flatLogits, flatTargets) as Tensor<[1], 'f32'>
}

export function causal_lm_eval_loss_forward(
  logits: Tensor<number[], 'f32'>,
  tokenIds: Tensor<number[], 'f32'>,
): Tensor<[1], 'f32'> {
  return indexedCrossEntropyIfAvailable(logits, tokenIds)
    ?? teacherForcedCrossEntropy(logits, tokenIds)
}

export function CausalLMLoss(): (
  logits: Tensor<number[], 'f32'>,
  tokenIds: Tensor<number[], 'f32'>,
) => Tensor<[1], 'f32'> {
  return (logits, tokenIds) =>
    indexedCrossEntropyIfAvailable(logits, tokenIds)
    ?? teacherForcedCrossEntropy(logits, tokenIds)
}

export function generate(
  model: DecoderModelModule,
  tokenIds: Tensor<number[], 'f32'>,
  opts: GenerateOptions,
  forward?: DecoderLMForward,
): Tensor<number[], 'f32'> {
  if (tokenIds.ndim !== 2) {
    throw new AffonError('invalid_shape', 'generate expects token ids shaped [batch, seq]')
  }

  const resolvedOpts = resolveGenerationOptions(opts)
  const decode: DecoderLMForward = forward ?? ((input) => model(input) as Tensor<number[], 'f32'>)

  return no_grad(function (): Tensor<number[], 'f32'> {
    let current = tokenIds
    let forbidden_token_ids: Set<number> | null = null
    let greedyForbiddenMask: Tensor<[1, number], 'i64'> | null = null

    for (let step = 0; step < resolvedOpts.max_new_tokens; step++) {
      const logits = decode(current as Tensor<[number, number], 'f32'>)
      let candidateLogits = candidateLogitsFromLastStep(logits)
      if (resolvedOpts.temperature === 0) {
        if (forbidden_token_ids === null) {
          forbidden_token_ids = validateForbiddenTokenIds(
            resolvedOpts.forbidden_token_ids,
            candidateLogits.shape[1] as number,
          )
        }
        if (greedyForbiddenMask === null) {
          greedyForbiddenMask = forbiddenTokenMask(
            forbidden_token_ids,
            candidateLogits.shape[1] as number,
            candidateLogits.device,
          )
        }
        if (greedyForbiddenMask) {
          candidateLogits = masked_fill(candidateLogits, greedyForbiddenMask, FORBIDDEN_LOGIT) as Tensor<number[], 'f32'>
        }
        const nextTokenIds = cast(argmax(candidateLogits, 1, false), 'f32') as Tensor<number[], 'f32'>
        current = appendNextTokenTensor(current, nextTokenIds)
      } else {
        const vocabSize = candidateLogits.shape[1] as number
        if (forbidden_token_ids === null) {
          forbidden_token_ids = validateForbiddenTokenIds(resolvedOpts.forbidden_token_ids, vocabSize)
        }
        if (greedyForbiddenMask === null) {
          greedyForbiddenMask = forbiddenTokenMask(
            forbidden_token_ids,
            vocabSize,
            candidateLogits.device,
          )
        }
        if (greedyForbiddenMask) {
          candidateLogits = masked_fill(candidateLogits, greedyForbiddenMask, FORBIDDEN_LOGIT) as Tensor<number[], 'f32'>
        }
        current = appendNextTokenTensor(
          current,
          sampleNextTokenTensor(candidateLogits, resolvedOpts.temperature, resolvedOpts.top_k),
        )
      }
    }

    return current
  })
}
