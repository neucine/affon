import { add, contiguous, cross_entropy, div, embedding, gelu, masked_fill, matmul, mean, mul, reshape, softmax, sqrt, sub, transpose } from 'affon:ops'
import { Tensor, optimize, program, type FormalTensor, type Program } from 'affon:compute'
import type { Optimizer } from 'affon:optim'

export interface DecoderModelOptions {
  numLayers: number
  numHeads: number
  hiddenDim: number
  causal?: boolean
  positional?: 'learned'
  maxSeqLen: number
  dropout?: number
  tieEmbeddings?: boolean
}

export interface DecoderModel {
  readonly vocabSize: number
  readonly dModel: number
  readonly options: Readonly<DecoderModelOptions>
  forward(batchSize: number, sequenceLength: number): DecoderProgram
  loss(batchSize: number, sequenceLength: number): DecoderProgram
  train(batchSize: number, sequenceLength: number, optimizer: Optimizer): DecoderProgram
}

type DecoderProgram = Program<Record<string, FormalTensor>, FormalTensor>

function positiveInteger(value: number, name: string) {
  if (!Number.isInteger(value) || value <= 0) throw new AffonError('invalid_arg', `${name} must be a positive integer`)
}

/** Author a decoder as composable Programs without owning execution resources. */
export function DecoderModel(vocabSize: number, dModel: number, options: DecoderModelOptions): DecoderModel {
  positiveInteger(vocabSize, 'DecoderModel vocabSize')
  positiveInteger(dModel, 'DecoderModel dModel')
  positiveInteger(options.numLayers, 'DecoderModel numLayers')
  positiveInteger(options.numHeads, 'DecoderModel numHeads')
  positiveInteger(options.hiddenDim, 'DecoderModel hiddenDim')
  positiveInteger(options.maxSeqLen, 'DecoderModel maxSeqLen')
  if (dModel % options.numHeads) throw new AffonError('invalid_arg', 'DecoderModel dModel must be divisible by numHeads')
  if (options.causal === false) throw new AffonError('invalid_arg', 'DecoderModel requires causal attention')
  if (options.positional !== undefined && options.positional !== 'learned') throw new AffonError('invalid_arg', 'DecoderModel supports learned positions')
  if (options.dropout !== undefined && options.dropout !== 0) throw new AffonError('invalid_arg', 'Program decoder does not support dropout')

  const opts = Object.freeze({ ...options })
  const forwardPrograms = new Map<string, DecoderProgram>()
  const lossPrograms = new Map<string, DecoderProgram>()

  function dimensions(batch: number, length: number) {
    positiveInteger(batch, 'DecoderModel batch size')
    positiveInteger(length, 'DecoderModel sequence length')
    if (length > opts.maxSeqLen) throw new AffonError('invalid_shape', 'DecoderModel sequence exceeds maxSeqLen')
  }

  function core(batch: number, length: number): DecoderProgram {
    dimensions(batch, length)
    const key = `core:${batch}x${length}`
    const cached = forwardPrograms.get(key)
    if (cached) return cached
    const headWidth = dModel / opts.numHeads
    const projection = (name: string, input: number, output: number) => program(name, p => {
      const value = p.argument('input', Tensor.f32([batch, length, input]))
      const weight = p.parameter('weight', Tensor.f32([input, output]), { initializer: { kind: 'xavier_uniform' } })
      const bias = p.parameter('bias', Tensor.f32([output]), { initializer: { kind: 'zeros' } })
      return add(matmul(value, weight), bias)
    })
    const normalization = program('decoder_norm', p => {
      const value = p.argument('input', Tensor.f32([batch, length, dModel]))
      const centered = sub(value, mean(value, -1, true))
      const epsilon = p.constant('epsilon', 1e-5, Tensor.f32([1]))
      const normalized = div(centered, sqrt(add(mean(mul(centered, centered), -1, true), epsilon)))
      const weight = p.parameter('weight', Tensor.f32([dModel]), { initializer: { kind: 'ones' } })
      const bias = p.parameter('bias', Tensor.f32([dModel]), { initializer: { kind: 'zeros' } })
      return add(mul(normalized, weight), bias)
    })
    const modelProjection = projection('decoder_projection', dModel, dModel)
    const expandProjection = projection('decoder_expand_projection', dModel, opts.hiddenDim)
    const contractProjection = projection('decoder_contract_projection', opts.hiddenDim, dModel)
    const lmHead = opts.tieEmbeddings ? undefined : projection('decoder_lm_head', dModel, vocabSize)
    const embeddings = program('decoder_embeddings', p => add(
      embedding(
        p.argument('token_embedding', Tensor.f32([vocabSize, dModel])),
        p.argument('token_ids', Tensor.i64([batch, length], { axes: ['batch', 'token'] })),
      ),
      embedding(
        p.argument('position_embedding', Tensor.f32([opts.maxSeqLen, dModel])),
        p.argument('position_ids', Tensor.i64([length])),
      ),
    ))
    const tiedHead = opts.tieEmbeddings ? program('decoder_tied_head', p => matmul(
      p.argument('input', Tensor.f32([batch, length, dModel])),
      transpose(p.argument('token_embedding', Tensor.f32([vocabSize, dModel])), [1, 0]),
    )) : undefined
    const attention = program('decoder_attention', p => {
      const query = p.argument('query', Tensor.f32([batch, opts.numHeads, length, headWidth]))
      const key = p.argument('key', Tensor.f32([batch, opts.numHeads, length, headWidth]))
      const value = p.argument('value', Tensor.f32([batch, opts.numHeads, length, headWidth]))
      const mask = p.argument('mask', Tensor.i64([1, 1, length, length]))
      const scale = p.argument('scale', Tensor.f32([1]))
      const weights = softmax(masked_fill(div(matmul(query, transpose(key, [0, 1, 3, 2])), scale), mask, -3.4028234663852886e38), 3)
      return reshape(contiguous(transpose(matmul(weights, value), [0, 2, 1, 3])), [batch, length, dModel])
    })
    const block = program('decoder_block', p => {
      let x = p.argument('input', Tensor.f32([batch, length, dModel]))
      const mask = p.argument('mask', Tensor.i64([1, 1, length, length]))
      const scale = p.argument('scale', Tensor.f32([1]))
      const split = (value: FormalTensor) => transpose(reshape(value, [batch, length, opts.numHeads, headWidth]), [0, 2, 1, 3])
      const normalized = normalization({ input: x }, 'attention_norm')
      const query = split(modelProjection({ input: normalized }, 'query'))
      const key = split(modelProjection({ input: normalized }, 'key'))
      const value = split(modelProjection({ input: normalized }, 'value'))
      const attended = attention({ query, key, value, mask, scale }, 'attention')
      const projected = modelProjection({ input: attended }, 'attention_output')
      x = add(x, contiguous(projected))
      const hidden = normalization({ input: x }, 'feed_forward_norm')
      const expanded = expandProjection({ input: hidden }, 'expand')
      const contracted = contractProjection({ input: gelu(expanded) }, 'contract')
      return add(x, contiguous(contracted))
    })
    const source = program('decoder_model', p => {
      const ids = p.argument('token_ids', Tensor.i64([batch, length], { axes: ['batch', 'token'] }))
      const parameter = (name: string, shape: readonly number[], initializer: any = { kind: 'xavier_uniform' }) => p.parameter(name, Tensor.f32(shape), { initializer })
      const tokenEmbedding = parameter('token_embedding', [vocabSize, dModel], { kind: 'normal', standard_deviation: 0.02 })
      const positionEmbedding = parameter('position_embedding', [opts.maxSeqLen, dModel], { kind: 'normal', standard_deviation: 0.02 })
      const positions = p.constant('position_ids', Array.from({ length }, (_, index) => index), Tensor.i64([length]))
      let x = embeddings({ token_ids: ids, token_embedding: tokenEmbedding, position_ids: positions, position_embedding: positionEmbedding }, 'embeddings')
      const allow = Array.from({ length }, (_, row) => Array.from({ length }, (_, column) => column > row ? 1 : 0))
      const mask = p.constant('causal_mask', [[allow]], Tensor.i64([1, 1, length, length]))
      const scale = p.constant('attention_scale', Math.sqrt(headWidth), Tensor.f32([1]))
      for (let layer = 0; layer < opts.numLayers; layer++) {
        x = block({ input: x, mask, scale }, `layer_${layer}`)
      }
      const normalized = normalization({ input: x }, 'final_norm')
      if (tiedHead) return tiedHead({ input: normalized, token_embedding: tokenEmbedding }, 'lm_head')
      return lmHead!({ input: normalized }, 'lm_head')
    })
    forwardPrograms.set(key, source)
    return source
  }

  function loss(batch: number, length: number): DecoderProgram {
    dimensions(batch, length)
    const key = `${batch}x${length}`
    const cached = lossPrograms.get(key)
    if (cached) return cached
    const source = program('decoder_loss', p => {
      const logits = p.argument('logits', Tensor.f32([batch, length, vocabSize]))
      const labels = p.argument('labels', Tensor.i64([batch, length], { axes: ['batch', 'token'] }))
      return cross_entropy(logits, labels)
    })
    lossPrograms.set(key, source)
    return source
  }

  return Object.freeze({
    vocabSize,
    dModel,
    options: opts,
    forward: (batchSize: number, sequenceLength: number) => core(batchSize, sequenceLength),
    loss,
    train: (batchSize: number, sequenceLength: number, optimizer: Optimizer) => optimize(core(batchSize, sequenceLength), loss(batchSize, sequenceLength), optimizer),
  })
}
