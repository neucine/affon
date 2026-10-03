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
  forward(batchSize: number, sequenceLength: number): Program<readonly FormalTensor[], FormalTensor>
  loss(batchSize: number, sequenceLength: number): Program<readonly FormalTensor[], FormalTensor>
  train(batchSize: number, sequenceLength: number, optimizer: Optimizer): Program<readonly FormalTensor[], FormalTensor>
}

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
  const forwardPrograms = new Map<string, Program<readonly FormalTensor[], FormalTensor>>()
  const lossPrograms = new Map<string, Program<readonly FormalTensor[], FormalTensor>>()

  function dimensions(batch: number, length: number) {
    positiveInteger(batch, 'DecoderModel batch size')
    positiveInteger(length, 'DecoderModel sequence length')
    if (length > opts.maxSeqLen) throw new AffonError('invalid_shape', 'DecoderModel sequence exceeds maxSeqLen')
  }

  function core(batch: number, length: number): Program<readonly FormalTensor[], FormalTensor> {
    dimensions(batch, length)
    const key = `core:${batch}x${length}`
    const cached = forwardPrograms.get(key)
    if (cached) return cached
    const headWidth = dModel / opts.numHeads
    const source = program('decoder_model', p => {
      const ids = p.argument('token_ids', Tensor.i64([batch, length], { axes: ['batch', 'token'] }))
      const parameter = (name: string, shape: readonly number[], initializer: any = { kind: 'xavier_uniform' }) => p.parameter(name, Tensor.f32(shape), { initializer })
      const scalar = (name: string, value: number) => p.constant(name, value, Tensor.f32([1]))
      const dense = (x: FormalTensor, prefix: string, input: number, output: number) => add(matmul(x, parameter(`${prefix}_weight`, [input, output])), parameter(`${prefix}_bias`, [output], { kind: 'zeros' }))
      const norm = (x: FormalTensor, prefix: string) => {
        const centered = sub(x, mean(x, -1, true))
        const normalized = div(centered, sqrt(add(mean(mul(centered, centered), -1, true), scalar(`${prefix}_epsilon`, 1e-5))))
        return add(mul(normalized, parameter(`${prefix}_weight`, [dModel], { kind: 'ones' })), parameter(`${prefix}_bias`, [dModel], { kind: 'zeros' }))
      }
      const tokenEmbedding = parameter('token_embedding', [vocabSize, dModel], { kind: 'normal', standard_deviation: 0.02 })
      const positionEmbedding = parameter('position_embedding', [opts.maxSeqLen, dModel], { kind: 'normal', standard_deviation: 0.02 })
      const positions = p.constant('position_ids', Array.from({ length }, (_, index) => index), Tensor.i64([length]))
      let x = add(embedding(tokenEmbedding, ids), embedding(positionEmbedding, positions))
      const allow = Array.from({ length }, (_, row) => Array.from({ length }, (_, column) => column > row ? 1 : 0))
      const mask = p.constant('causal_mask', [[allow]], Tensor.i64([1, 1, length, length]))
      const scale = scalar('attention_scale', Math.sqrt(headWidth))
      const split = (value: FormalTensor) => transpose(reshape(value, [batch, length, opts.numHeads, headWidth]), [0, 2, 1, 3])
      for (let layer = 0; layer < opts.numLayers; layer++) {
        const prefix = `layer_${layer}`
        const input = norm(x, `${prefix}_attention_norm`)
        const q = split(dense(input, `${prefix}_query`, dModel, dModel))
        const k = split(dense(input, `${prefix}_key`, dModel, dModel))
        const v = split(dense(input, `${prefix}_value`, dModel, dModel))
        const attended = reshape(contiguous(transpose(matmul(softmax(masked_fill(div(matmul(q, transpose(k, [0, 1, 3, 2])), scale), mask, -3.4028234663852886e38), 3), v), [0, 2, 1, 3])), [batch, length, dModel])
        x = add(x, dense(attended, `${prefix}_attention_output`, dModel, dModel))
        const hidden = norm(x, `${prefix}_feed_forward_norm`)
        x = add(x, dense(gelu(dense(hidden, `${prefix}_expand`, dModel, opts.hiddenDim)), `${prefix}_contract`, opts.hiddenDim, dModel))
      }
      const normalized = norm(x, 'final_norm')
      return opts.tieEmbeddings ? matmul(normalized, transpose(tokenEmbedding, [1, 0])) : dense(normalized, 'lm_head', dModel, vocabSize)
    })
    forwardPrograms.set(key, source)
    return source
  }

  function loss(batch: number, length: number): Program<readonly FormalTensor[], FormalTensor> {
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
