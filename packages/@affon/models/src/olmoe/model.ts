import { add, cast, cat, contiguous, div, embedding, eq, index_select, masked_fill, matmul, mean, mul, neg, reshape, silu, slice, softmax, sqrt, sum, topk, transpose, where } from 'affon:ops'
import { Tensor, program, type FormalTensor } from 'affon:core'
import { f32_parameter, parameter_checks, positive_dimensions, type ModelTensor } from '../shared/parameters.ts'

export interface OlmoeConfig {
  width: number; innerWidth: number; heads: number; kvHeads: number; layers: number
  experts: number; expertsPerToken: number; contextLength: number; vocabSize: number
  epsilon: number; ropeTheta: number; normalizeTopK: boolean
}
export interface OlmoeExpertWeights { gate: ModelTensor; up: ModelTensor; down: ModelTensor }
export interface OlmoeWeights {
  tokenEmbedding: ModelTensor; finalNorm: ModelTensor; lmHead: ModelTensor
  blocks: { attentionNorm: ModelTensor; feedForwardNorm: ModelTensor; queryNorm: ModelTensor; keyNorm: ModelTensor
    query: ModelTensor; key: ModelTensor; value: ModelTensor; attentionOutput: ModelTensor; router: ModelTensor
    experts: OlmoeExpertWeights[] }[]
}

/** OLMoE forward Program. Expert selection is exact, but all experts run for each token. */
export function create_olmoe(config: OlmoeConfig, weights: OlmoeWeights) {
  const { width, innerWidth, heads, kvHeads, layers, experts, expertsPerToken, contextLength, vocabSize } = config
  positive_dimensions(width, innerWidth, heads, kvHeads, layers, experts, expertsPerToken, contextLength, vocabSize)
  if (width % heads || heads % kvHeads || (width / heads) % 2 || expertsPerToken > experts || weights.blocks.length !== layers
    || !Number.isFinite(config.epsilon) || config.epsilon <= 0 || !Number.isFinite(config.ropeTheta) || config.ropeTheta <= 0)
    throw Error('Invalid OLMoE dimensions or normalization')
  const headWidth = width / heads, kvWidth = kvHeads * headWidth
  const check = parameter_checks().tensor
  check(weights.tokenEmbedding, [vocabSize, width]); check(weights.finalNorm, [width]); check(weights.lmHead, [width, vocabSize])
  for (const block of weights.blocks) {
    check(block.attentionNorm, [width]); check(block.feedForwardNorm, [width]); check(block.queryNorm, [width]); check(block.keyNorm, [kvWidth])
    check(block.query, [width, width]); check(block.key, [width, kvWidth]); check(block.value, [width, kvWidth]); check(block.attentionOutput, [width, width])
    check(block.router, [width, experts])
    if (block.experts.length !== experts) throw Error('Invalid OLMoE expert count')
    for (const expert of block.experts) {
      check(expert.gate, [width, innerWidth]); check(expert.up, [width, innerWidth]); check(expert.down, [innerWidth, width])
    }
  }
  const parameters: Record<string, ModelTensor> = { token_embedding: weights.tokenEmbedding, final_norm: weights.finalNorm, lm_head: weights.lmHead }
  weights.blocks.forEach((block, i) => {
    for (const [key, value] of Object.entries(block)) if (key !== 'experts') parameters[`block_${i}_${key}`] = value as ModelTensor
    block.experts.forEach((expert, j) => { for (const [key, value] of Object.entries(expert)) parameters[`block_${i}_expert_${j}_${key}`] = value })
  })
  const parameter = (p: any, name: string, shape: number[]) => f32_parameter(p, name, shape, parameters[name])
  function forward(length: number, outputStart = 0) {
    positive_dimensions(length)
    if (length > contextLength || !Number.isInteger(outputStart) || outputStart < 0 || outputStart >= length) throw Error('Invalid OLMoE sequence or output window')
    return program('olmoe', p => {
      const ids = p.argument('ids', Tensor.i64([length]))
      const mask = p.argument('mask', Tensor.i64([1, 1, length, length]))
      const scalar = (name: string, value: number) => p.constant(name, value, Tensor.f32([1]))
      const one = scalar('one', 1), epsilon = scalar('epsilon', config.epsilon), zero = scalar('zero', 0)
      const scale = scalar('attention_scale', 1 / Math.sqrt(headWidth))
      const repeatIds = p.constant('repeat_ids', Array.from({ length: heads }, (_, i) => Math.floor(i / (heads / kvHeads))), Tensor.i64([heads]))
      const angles = Array.from({ length }, (_, token) => Array.from({ length: headWidth }, (_, i) => token / Math.pow(config.ropeTheta, 2 * (i % (headWidth / 2)) / headWidth)))
      const cosine = p.constant('rope_cosine', [angles.map(row => [row.map(Math.cos)])], Tensor.f32([1, length, 1, headWidth]))
      const sine = p.constant('rope_sine', [angles.map(row => [row.map(Math.sin)])], Tensor.f32([1, length, 1, headWidth]))
      const norm = (x: FormalTensor, weight: FormalTensor, axis: number) => mul(mul(x, div(one, sqrt(add(mean(mul(x, x), axis, true), epsilon)))), weight)
      const rope = (value: FormalTensor, count: number) => {
        const shaped = reshape(value, [1, length, count, headWidth])
        const first = contiguous(slice(shaped, [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start: 0, stop: count }, { start: 0, stop: headWidth / 2 }]))
        const second = contiguous(slice(shaped, [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start: 0, stop: count }, { start: headWidth / 2, stop: headWidth }]))
        return add(mul(shaped, cosine), mul(cat([neg(second), first], 3), sine))
      }
      const expand = (value: FormalTensor) => index_select(transpose(reshape(value, [1, length, kvHeads, headWidth]), [0, 2, 1, 3]), 1, repeatIds)
      let x = reshape(embedding(parameter(p, 'token_embedding', [vocabSize, width]), ids), [1, length, width])
      const hiddenStates: FormalTensor[] = []
      for (let i = 0; i < layers; i++) {
        hiddenStates.push(x)
        const prefix = `block_${i}`
        const get = (name: string, shape: number[]) => parameter(p, `${prefix}_${name}`, shape)
        const normalized = norm(x, get('attentionNorm', [width]), 2)
        const q = transpose(rope(norm(matmul(normalized, get('query', [width, width])), get('queryNorm', [width]), 2), heads), [0, 2, 1, 3])
        const k = reshape(rope(norm(matmul(normalized, get('key', [width, kvWidth])), get('keyNorm', [kvWidth]), 2), kvHeads), [1, length, kvWidth])
        const v = matmul(normalized, get('value', [width, kvWidth]))
        const scores = masked_fill(mul(matmul(q, transpose(expand(k), [0, 1, 3, 2])), scale), mask, -3.4028234663852886e38)
        const attention = reshape(contiguous(transpose(matmul(softmax(scores, 3), expand(v)), [0, 2, 1, 3])), [1, length, width])
        x = add(x, matmul(attention, get('attentionOutput', [width, width])))
        const feed = norm(x, get('feedForwardNorm', [width]), 2)
        const probabilities = softmax(matmul(feed, get('router', [width, experts])), 2)
        const [selectedWeights, selectedIds] = topk(probabilities, expertsPerToken, 2)
        const weightsForSelected = config.normalizeTopK ? div(selectedWeights, sum(selectedWeights, 2, true)) : selectedWeights
        let mixed: FormalTensor | undefined
        for (let expert = 0; expert < experts; expert++) {
          const expertId = p.constant(`${prefix}_expert_id_${expert}`, expert, Tensor.i64([1]))
          const route = reshape(sum(where(eq(selectedIds, expertId), weightsForSelected, zero), 2, true), [1, length, 1])
          const gate = get(`expert_${expert}_gate`, [width, innerWidth])
          const up = get(`expert_${expert}_up`, [width, innerWidth])
          const down = get(`expert_${expert}_down`, [innerWidth, width])
          const contribution = mul(matmul(mul(silu(matmul(feed, gate)), matmul(feed, up)), down), route)
          mixed = mixed ? add(mixed, contribution) : contribution
        }
        x = add(x, mixed!)
      }
      x = norm(x, parameter(p, 'final_norm', [width]), 2)
      hiddenStates.push(x)
      const suffix = (value: FormalTensor) => contiguous(slice(value, [{ start: 0, stop: 1 }, { start: outputStart, stop: length }, { start: 0, stop: width }]))
      return [matmul(suffix(x), parameter(p, 'lm_head', [width, vocabSize])), ...hiddenStates.map(suffix)]
    })
  }
  return { config, forward, parameters }
}
