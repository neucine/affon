import { add, contiguous, div, embedding, gelu, masked_fill, matmul, mean, mul, reshape, softmax, sqrt, sub, transpose } from 'affon:ops'
import {
  Session,
  Tensor,
  program,
  optimize,
  type Device,
  type ExecutionState,
  type FormalTensor,
  type Program,
} from 'affon:compute'
import { adam, adamw, type Optimizer } from 'affon:optim'

export interface DecoderModelOptions {
  numLayers: number
  numHeads: number
  hiddenDim: number
  causal?: boolean
  positional?: 'learned'
  maxSeqLen: number
  dropout?: number
  tieEmbeddings?: boolean
  device?: Device
  seed?: number
}

export interface DecoderModelModule {
  (tokenIds: Tensor): Tensor
  readonly vocabSize: number
  readonly dModel: number
  readonly options: Readonly<DecoderModelOptions>
  readonly device: Device
  readonly session: Session
  readonly parameters: readonly Tensor[]
  readonly executionState: ExecutionState | null
  forward(tokenIds: Tensor): Tensor
  source(batchSize: number, sequenceLength: number): Program
  trainBatch(tokenIds: readonly (readonly number[])[], optimizer?: Optimizer): number
  state(): Record<string, Tensor>
  restore(values: Readonly<Record<string, Tensor>>): void
  train(): DecoderModelModule
  eval(): DecoderModelModule
  dispose(): void
}

type Compiled = { source: Program; executable: ReturnType<Session['compile']> }

function positiveInteger(value: number, name: string) {
  if (!Number.isInteger(value) || value <= 0) throw new AffonError('invalid_arg', `${name} must be a positive integer`)
}

export function DecoderModel(vocabSize: number, dModel: number, opts: DecoderModelOptions): DecoderModelModule {
  positiveInteger(vocabSize, 'DecoderModel vocabSize')
  positiveInteger(dModel, 'DecoderModel dModel')
  positiveInteger(opts.numLayers, 'DecoderModel numLayers')
  positiveInteger(opts.numHeads, 'DecoderModel numHeads')
  positiveInteger(opts.hiddenDim, 'DecoderModel hiddenDim')
  positiveInteger(opts.maxSeqLen, 'DecoderModel maxSeqLen')
  if (dModel % opts.numHeads) throw new AffonError('invalid_arg', 'DecoderModel dModel must be divisible by numHeads')
  if (opts.causal === false) throw new AffonError('invalid_arg', 'DecoderModel requires causal attention')
  if (opts.positional !== undefined && opts.positional !== 'learned') throw new AffonError('invalid_arg', 'DecoderModel supports learned positions')
  if (opts.dropout !== undefined && opts.dropout !== 0) throw new AffonError('invalid_arg', 'Program decoder does not support dropout')

  const device = opts.device ?? 'cpu'
  const session = new Session({ device })
  const compiled = new Map<string, Compiled>()
  let executionState: ExecutionState | null = null
  let disposed = false

  function build(batch: number, length: number): Program {
    positiveInteger(batch, 'DecoderModel batch size')
    positiveInteger(length, 'DecoderModel sequence length')
    if (length > opts.maxSeqLen) throw new AffonError('invalid_shape', 'DecoderModel sequence exceeds maxSeqLen')
    const key = `${batch}x${length}`
    const existing = compiled.get(key)
    if (existing) return existing.source
    const headWidth = dModel / opts.numHeads
    const source = program('decoder_model', p => {
      const ids = p.argument('token_ids', Tensor.i64([batch, length], { axes: ['batch', 'token'] }))
      const parameter = (name: string, shape: readonly number[], initializer: any = { kind: 'xavier_uniform' }) =>
        p.parameter(name, Tensor.f32(shape), { initializer })
      const scalar = (name: string, value: number) => p.constant(name, value, Tensor.f32([1]))
      const dense = (x: FormalTensor, prefix: string, input: number, output: number) =>
        add(matmul(x, parameter(`${prefix}_weight`, [input, output])), parameter(`${prefix}_bias`, [output], { kind: 'zeros' }))
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
      return dense(normalized, 'lm_head', dModel, vocabSize)
    })
    compiled.set(key, { source, executable: session.compile(source) })
    if (!executionState) executionState = session.initialize(source, { seed: opts.seed ?? 0 })
    return source
  }

  function forward(tokenIds: Tensor): Tensor {
    if (disposed) throw new Error('DecoderModel has been disposed')
    if (tokenIds.ndim !== 2) throw new AffonError('invalid_shape', 'DecoderModel expects token ids shaped [batch, seq]')
    const batch = tokenIds.shape[0], length = tokenIds.shape[1]
    const source = build(batch, length)
    const input = session.tensor(tokenIds.to_array() as any, { dtype: 'i64', axes: ['batch', 'token'] })
    try { return session.compile(source).run({ token_ids: input }, executionState!) as Tensor }
    finally { input.dispose() }
  }

  function trainBatch(rows: readonly (readonly number[])[], optimizer: Optimizer = adam()): number {
    if (rows.length === 0 || rows.some(row => row.length < 2 || row.length !== rows[0].length)) throw new AffonError('invalid_shape', 'trainBatch expects equal token windows')
    const batch = rows.length, length = rows[0].length - 1
    build(batch, length)
    const model = program('decoder_model', p => {
      const ids = p.argument('token_ids', Tensor.i64([batch, length]))
      const labels = p.argument('labels', Tensor.i64([batch, length]))
      const table = p.parameter('token_embedding', Tensor.f32([vocabSize, dModel]), { initializer: { kind: 'normal', standard_deviation: 0.02 } })
      const head = p.parameter('lm_head_weight', Tensor.f32([dModel, vocabSize]), { initializer: { kind: 'xavier_uniform' } })
      const bias = p.parameter('lm_head_bias', Tensor.f32([vocabSize]), { initializer: { kind: 'zeros' } })
      return p.nn.cross_entropy(add(matmul(embedding(table, ids), head), bias), labels)
    })
    const step = optimize(model, optimizer)
    const inputs = session.tensor(rows.map(row => row.slice(0, -1)), { dtype: 'i64' })
    const labels = session.tensor(rows.map(row => row.slice(1)), { dtype: 'i64' })
    try {
      const result = session.compile(step).run({ token_ids: inputs, labels }, executionState!) as Tensor
      try { return result.item() } finally { result.dispose() }
    } finally { inputs.dispose(); labels.dispose() }
  }

  function state(): Record<string, Tensor> {
    if (!executionState) build(1, 1)
    return executionState!.parameters as Record<string, Tensor>
  }

  function restore(values: Readonly<Record<string, Tensor>>) {
    const current = state()
    for (const key of Object.keys(current)) {
      const value = values[key]
      if (!value || value.dtype !== current[key].dtype || JSON.stringify(value.shape) !== JSON.stringify(current[key].shape)) throw new AffonError('shape_mismatch', `Invalid decoder checkpoint parameter: ${key}`)
    }
    for (const [key, old] of Object.entries(current)) {
      const replacement = session.tensor(values[key].to_array() as any, { dtype: values[key].dtype })
      old.dispose(); executionState!.parameters[key] = replacement
    }
  }

  function dispose() {
    if (disposed) return
    executionState?.dispose(); session.dispose(); compiled.clear(); disposed = true
  }

  const callable = ((tokenIds: Tensor) => forward(tokenIds)) as DecoderModelModule
  Object.defineProperties(callable, {
    vocabSize: { value: vocabSize }, dModel: { value: dModel }, options: { value: Object.freeze({ ...opts }) },
    device: { value: device }, session: { value: session },
    executionState: { get: () => executionState }, parameters: { get: () => Object.values(state()) },
  })
  callable.forward = forward
  callable.source = build
  callable.trainBatch = trainBatch
  callable.state = state
  callable.restore = restore
  callable.train = () => callable
  callable.eval = () => callable
  callable.dispose = dispose
  return callable
}

export { adam, adamw }
