import { add, contiguous, div, embedding as embedding_lookup, gelu, masked_fill, matmul, reshape, softmax, transpose } from 'affon:ops'
import { Tensor, program, type Callable, type FormalTensor, type Program } from 'affon:compute'
import { embedding, layer_norm, linear } from 'affon:nn'

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

export interface DecoderModel extends Callable<{ token_ids: FormalTensor }, FormalTensor> {
  readonly vocabSize: number
  readonly dModel: number
  readonly options: Readonly<DecoderModelOptions>
}

type DecoderProgram = Program<Record<string, FormalTensor>, FormalTensor>
const programCaches = new WeakMap<DecoderModel, Map<string, DecoderProgram>>()

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
  const components = new Map<string, DecoderProgram>()
  const project = linear({ out_features: dModel })
  const expand = linear({ out_features: opts.hiddenDim })
  const contract = linear({ out_features: dModel })
  const normalize = layer_norm({ normalized_shape: dModel })
  const positionEmbedding = embedding({ num_embeddings: opts.maxSeqLen, embedding_dim: dModel })
  const languageModelHead = opts.tieEmbeddings ? undefined : linear({ out_features: vocabSize })

  function dimensions(batch: number, length: number) {
    positiveInteger(batch, 'DecoderModel batch size')
    positiveInteger(length, 'DecoderModel sequence length')
    if (length > opts.maxSeqLen) throw new AffonError('invalid_shape', 'DecoderModel sequence exceeds maxSeqLen')
  }

  function core(batch: number, length: number, axes?: readonly string[]): DecoderProgram {
    dimensions(batch, length)
    const key = `core:${batch}x${length}:${axes?.join(',') ?? ''}`
    const cached = components.get(key)
    if (cached) return cached
    const headWidth = dModel / opts.numHeads
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
      const normalized = normalize({ x }, 'attention.norm')
      const query = split(project({ x: normalized }, 'attention.query'))
      const key = split(project({ x: normalized }, 'attention.key'))
      const value = split(project({ x: normalized }, 'attention.value'))
      const attended = attention({ query, key, value, mask, scale }, 'attention')
      const projected = project({ x: attended }, 'attention.output')
      x = add(x, contiguous(projected))
      const hidden = normalize({ x }, 'feed_forward.norm')
      const expanded = expand({ x: hidden }, 'feed_forward.expand')
      const contracted = contract({ x: gelu(expanded) }, 'feed_forward.contract')
      return add(x, contiguous(contracted))
    })
    const source = program('decoder_model', p => {
      const ids = p.argument('token_ids', Tensor.i64([batch, length], axes ? { axes } : undefined))
      // The token table is owned here because the tied output head deliberately shares it.
      const tokenEmbedding = p.parameter('token_embedding.weight', Tensor.f32([vocabSize, dModel]), {
        initializer: { kind: 'normal', standard_deviation: 0.02 },
      })
      const positions = p.constant('position_ids', Array.from({ length }, (_, index) => index), Tensor.i64([length]))
      let x = add(
        embedding_lookup(tokenEmbedding, ids),
        positionEmbedding({ indices: positions }, 'position_embedding'),
      )
      const allow = Array.from({ length }, (_, row) => Array.from({ length }, (_, column) => column > row ? 1 : 0))
      const mask = p.constant('causal_mask', [[allow]], Tensor.i64([1, 1, length, length]))
      const scale = p.constant('attention_scale', Math.sqrt(headWidth), Tensor.f32([1]))
      for (let layer = 0; layer < opts.numLayers; layer++) {
        x = block({ input: x, mask, scale }, `blocks.${layer}`)
      }
      const normalized = normalize({ x }, 'final_norm')
      if (tiedHead) return tiedHead({ input: normalized, token_embedding: tokenEmbedding }, 'lm_head')
      return languageModelHead!({ x: normalized }, 'lm_head')
    })
    components.set(key, source)
    return source
  }

  const callable = ((bindings: { token_ids: FormalTensor }, instance = 'decoder') => {
    if (!bindings || typeof bindings !== 'object' || Array.isArray(bindings) || Object.keys(bindings).length !== 1 || !('token_ids' in bindings)) {
      throw new TypeError('DecoderModel expects only the token_ids binding')
    }
    const ids = bindings.token_ids
    if (!ids?.spec || ids.spec.dtype !== 'i64' || ids.spec.shape.length !== 2) {
      throw new TypeError('DecoderModel token_ids must be an i64 tensor shaped [batch, sequence]')
    }
    const [batch, length] = ids.spec.shape
    return core(batch, length, ids.spec.axes)({ token_ids: ids }, instance)
  }) as DecoderModel
  Object.defineProperties(callable, {
    vocabSize: { value: vocabSize, enumerable: true },
    dModel: { value: dModel, enumerable: true },
    options: { value: opts, enumerable: true },
  })
  return Object.freeze(callable)
}

/** Internal execution boundary used by the app; public examples author this Program explicitly. */
export function decoderProgram(model: DecoderModel, batch: number, length: number): DecoderProgram {
  positiveInteger(batch, 'decoderProgram batch size')
  positiveInteger(length, 'decoderProgram sequence length')
  let programs = programCaches.get(model)
  if (!programs) {
    programs = new Map()
    programCaches.set(model, programs)
  }
  const key = `${batch}x${length}`
  const cached = programs.get(key)
  if (cached) return cached
  const source = program('decoder_lm', p => model({
    token_ids: p.argument('token_ids', Tensor.i64([batch, length], { axes: ['batch', 'token'] })),
  }, 'decoder'))
  programs.set(key, source)
  return source
}
