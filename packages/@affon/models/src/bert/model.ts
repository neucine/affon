import { add, contiguous, div, embedding, erf, masked_fill, matmul, mean, mul, reshape, slice, softmax, sqrt, sub, sum, tanh, transpose } from 'affon:ops'
import { Session, Tensor, program, type Device, type ExecutionState, type FormalTensor, type Program } from 'affon:compute'
import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import type { ModelAffineWeights, ModelTensor } from '../shared/parameters.ts'

export interface BertConfig { width: number; innerWidth: number; heads: number; layers: number; vocabSize: number; contextLength: number; typeVocabSize: number; epsilon: number }
export interface BertWeights {
  blocks: { query: ModelAffineWeights; key: ModelAffineWeights; value: ModelAffineWeights; attentionOutput: ModelAffineWeights; attentionNorm: ModelAffineWeights; feedForwardNorm: ModelAffineWeights; expand: ModelAffineWeights; contract: ModelAffineWeights }[]
  tokenEmbedding: ModelTensor; positionEmbedding: ModelTensor; typeEmbedding: ModelTensor; embeddingNorm: ModelAffineWeights; pooler: ModelAffineWeights
}

type CompiledBert = { source: Program; state: ExecutionState }

/** Construct f32 BERT as shape-specialized Programs backed by one Session.
 * Supplied tensors are copied into session-owned ExecutionState parameters.
 */
export function create_bert(config: BertConfig, weights: BertWeights, device: Device = 'cpu') {
  if (!Number.isInteger(config.width) || config.width <= 0 || !Number.isInteger(config.heads) || config.heads <= 0 || config.width % config.heads || !Number.isInteger(config.layers) || config.layers <= 0 || weights.blocks.length !== config.layers || !Number.isFinite(config.epsilon) || config.epsilon <= 0) throw new Error('Invalid model dimensions or normalization')
  const check = parameter_checks(device)
  const width = config.width
  positive_dimensions(config.innerWidth)
  for (const block of weights.blocks) {
    for (const params of [block.query, block.key, block.value, block.attentionOutput]) check.affine(params, [width, width])
    check.affine(block.attentionNorm, [width]); check.affine(block.feedForwardNorm, [width])
    check.affine(block.expand, [width, config.innerWidth]); check.affine(block.contract, [config.innerWidth, width])
  }
  positive_dimensions(config.vocabSize, config.contextLength, config.typeVocabSize)
  check.tensor(weights.tokenEmbedding, [config.vocabSize, width])
  check.tensor(weights.positionEmbedding, [config.contextLength, width])
  check.tensor(weights.typeEmbedding, [config.typeVocabSize, width])
  check.affine(weights.embeddingNorm, [width]); check.affine(weights.pooler, [width, width])

  const session = new Session({ device })
  const compiled = new Map<string, CompiledBert>()
  const parameterValues: Record<string, ModelTensor> = {}
  const remember = (name: string, value: ModelTensor) => { parameterValues[name] = value; return name }

  remember('token_embedding', weights.tokenEmbedding)
  remember('position_embedding', weights.positionEmbedding)
  remember('type_embedding', weights.typeEmbedding)
  remember('embedding_norm_weight', weights.embeddingNorm.weight)
  remember('embedding_norm_bias', weights.embeddingNorm.bias)
  remember('pooler_weight', weights.pooler.weight)
  remember('pooler_bias', weights.pooler.bias)
  weights.blocks.forEach((block, index) => {
    for (const [part, affine] of Object.entries(block) as [string, ModelAffineWeights][]) {
      remember(`block_${index}_${part}_weight`, affine.weight)
      remember(`block_${index}_${part}_bias`, affine.bias)
    }
  })

  const parameter = (p: any, name: string, shape: readonly number[]) => p.parameter(name, Tensor.f32(shape)) as FormalTensor
  const affine = (p: any, name: string, shape: readonly number[]) => ({
    weight: parameter(p, `${name}_weight`, shape),
    bias: parameter(p, `${name}_bias`, shape.length === 1 ? shape : [shape[shape.length - 1]]),
  })

  function build(batch: number, length: number): CompiledBert {
    const key = `${batch}x${length}`
    const cached = compiled.get(key)
    if (cached) return cached
    const source = program(`bert_${batch}_${length}`, p => {
      const ids = p.argument('ids', Tensor.i64([batch, length]))
      const types = p.argument('types', Tensor.i64([batch, length]))
      const valid = p.argument('valid', Tensor.f32([batch, length, 1]))
      const attentionMask = p.argument('attention_mask', Tensor.i64([batch, 1, 1, length]))
      const positions = p.argument('positions', Tensor.i64([batch, length]))
      const tokenEmbedding = parameter(p, 'token_embedding', [config.vocabSize, width])
      const positionEmbedding = parameter(p, 'position_embedding', [config.contextLength, width])
      const typeEmbedding = parameter(p, 'type_embedding', [config.typeVocabSize, width])
      const embeddingNorm = affine(p, 'embedding_norm', [width])
      const poolerWeights = affine(p, 'pooler', [width, width])
      const epsilon = p.constant('epsilon', config.epsilon, Tensor.f32([1]))
      const sqrtTwo = p.constant('sqrt_two', Math.sqrt(2), Tensor.f32([1]))
      const erfScale = p.constant('erf_scale', 0.5, Tensor.f32([1]))
      const one = p.constant('one', 1, Tensor.f32([1]))

      const dense = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => add(matmul(x, params.weight), params.bias)
      const norm = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => {
        const centered = sub(x, mean(x, -1, true))
        const variance = mean(mul(centered, centered), -1, true)
        return add(mul(div(centered, sqrt(add(variance, epsilon))), params.weight), params.bias)
      }
      const erfGelu = (x: FormalTensor) => mul(mul(x, erfScale), add(erf(div(x, sqrtTwo)), one))
      const split = (x: FormalTensor) => transpose(reshape(x, [batch, length, config.heads, width / config.heads]), [0, 2, 1, 3])
      const attention = (q: FormalTensor, k: FormalTensor, v: FormalTensor) => {
        const scale = p.constant(`attention_scale_${q.id}`, Math.sqrt(width / config.heads), Tensor.f32([1]))
        const scores = masked_fill(div(matmul(split(q), transpose(split(k), [0, 1, 3, 2])), scale), attentionMask, -3.4028234663852886e38)
        return reshape(contiguous(transpose(matmul(softmax(scores, 3), split(v)), [0, 2, 1, 3])), [batch, length, width])
      }

      let x = norm(add(add(embedding(tokenEmbedding, ids), embedding(positionEmbedding, positions)), embedding(typeEmbedding, types)), embeddingNorm)
      const hiddenStates: FormalTensor[] = [x]
      for (let index = 0; index < config.layers; index++) {
        const prefix = `block_${index}`
        const query = affine(p, `${prefix}_query`, [width, width])
        const keyWeights = affine(p, `${prefix}_key`, [width, width])
        const value = affine(p, `${prefix}_value`, [width, width])
        const attentionOutput = affine(p, `${prefix}_attentionOutput`, [width, width])
        const attentionNorm = affine(p, `${prefix}_attentionNorm`, [width])
        const feedForwardNorm = affine(p, `${prefix}_feedForwardNorm`, [width])
        const expand = affine(p, `${prefix}_expand`, [width, config.innerWidth])
        const contract = affine(p, `${prefix}_contract`, [config.innerWidth, width])
        const context = attention(dense(x, query), dense(x, keyWeights), dense(x, value))
        x = norm(add(x, dense(context, attentionOutput)), attentionNorm)
        x = norm(add(x, dense(erfGelu(dense(x, expand)), contract)), feedForwardNorm)
        hiddenStates.push(x)
      }
      const pooledMean = div(sum(mul(x, valid), 1), sum(valid, 1))
      const pooled = div(pooledMean, sqrt(sum(mul(pooledMean, pooledMean), 1, true)))
      const firstToken = reshape(contiguous(slice(x, [{ start: 0, stop: batch }, { start: 0, stop: 1 }, { start: 0, stop: width }])), [batch, width])
      const pooler = tanh(dense(firstToken, poolerWeights))
      return [pooled, pooler, ...hiddenStates]
    })
    const state = session.initialize(source, { parameters: parameterValues })
    const result = { source, state }
    compiled.set(key, result)
    return result
  }

  function forward(ids: number[][], masks: number[][], types: number[][]) {
    const batch = ids.length, length = ids[0]?.length ?? 0
    if (!batch || !length || length > config.contextLength || masks.length !== batch || types.length !== batch) throw new Error('Invalid BERT batch')
    for (let b = 0; b < batch; b++) {
      for (const [row, limit] of [[ids[b], config.vocabSize], [types[b], config.typeVocabSize], [masks[b], 2]] as [number[], number][]) {
        if (row.length !== length || row.some(value => !Number.isInteger(value) || value < 0 || value >= limit)) throw new Error('Invalid BERT input row')
      }
      if (!masks[b].some(value => value === 1)) throw new Error('All-masked BERT inputs are not supported')
    }
    const executable = build(batch, length)
    const values = {
      ids: session.tensor(ids, { dtype: 'i64' }),
      types: session.tensor(types, { dtype: 'i64' }),
      positions: session.tensor(Array.from({ length: batch }, () => Array.from({ length }, (_, index) => index)), { dtype: 'i64' }),
      valid: session.tensor(masks.map(row => row.map(value => [value]))),
      attention_mask: session.tensor(masks.map(row => [[[...row.map(value => 1 - value)]]]), { dtype: 'i64' }),
    } as Record<string, any>
    try {
      const outputs = session.compile(executable.source).run(values, executable.state) as Tensor[]
      const hidden_states = outputs.slice(2)
      return { output: hidden_states[hidden_states.length - 1], pooled: outputs[0], pooler: outputs[1], hidden_states }
    } finally {
      for (const value of Object.values(values)) value.dispose()
    }
  }

  function dispose() {
    for (const value of compiled.values()) value.state.dispose()
    compiled.clear()
    session.dispose()
  }

  return { config, forward, dispose }
}
