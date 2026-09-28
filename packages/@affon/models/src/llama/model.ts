import { add, cat, contiguous, div, index_select, masked_fill, matmul, mean, mul, neg, no_grad, permute, reshape, silu, softmax, sqrt, square, tensor, transpose } from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { parameter_checks, positive_dimensions } from '../shared/parameters.ts'

export interface LlamaConfig {
  width: number; innerWidth: number; heads: number; kvHeads: number; layers: number
  contextLength: number; vocabSize: number; epsilon: number; ropeTheta: number; eosTokenId?: number
}
export interface LlamaWeights {
  tokenEmbedding: Tensor; finalNorm: Tensor
  blocks: { attentionNorm: Tensor; feedForwardNorm: Tensor; query: Tensor; key: Tensor; value: Tensor; attentionOutput: Tensor; gate: Tensor; up: Tensor; down: Tensor }[]
}

/** Tied-head, bias-free Llama decoder with full, non-interleaved RoPE and f32 execution.
 * Linear weights use [input, output]. Sessions own sequence-major, unrepeated K/V.
 * This bounded variant is used by SmolLM2; scaled RoPE and other Llama variants are not implied.
 */
export function create_llama(config: LlamaConfig, weights: LlamaWeights, device: Device = 'cpu') {
  const { width: d, innerWidth: inner, heads, kvHeads, layers, contextLength, vocabSize } = config
  positive_dimensions(d, inner, heads, kvHeads, layers, contextLength, vocabSize)
  if (d % heads || heads % kvHeads || (d / heads) % 2 || weights.blocks.length !== layers
    || !Number.isFinite(config.epsilon) || config.epsilon <= 0 || !Number.isFinite(config.ropeTheta) || config.ropeTheta <= 0)
    throw Error('Invalid Llama dimensions, RoPE or normalization')
  if (config.eosTokenId !== undefined && (!Number.isInteger(config.eosTokenId) || config.eosTokenId < 0 || config.eosTokenId >= vocabSize)) throw Error('Invalid EOS token')
  const head = d / heads, kvWidth = kvHeads * head
  const check = parameter_checks(device).tensor
  check(weights.tokenEmbedding, [vocabSize, d]); check(weights.finalNorm, [d])
  for (const b of weights.blocks) {
    check(b.attentionNorm, [d]); check(b.feedForwardNorm, [d])
    check(b.query, [d, d]); check(b.key, [d, kvWidth]); check(b.value, [d, kvWidth]); check(b.attentionOutput, [d, d])
    check(b.gate, [d, inner]); check(b.up, [d, inner]); check(b.down, [inner, d])
  }
  const scalar = (n: number) => tensor(n, { dtype: 'f32', device })
  const epsilon = scalar(config.epsilon), one = scalar(1), scale = scalar(1 / Math.sqrt(head))
  const norm = (x: Tensor, w: Tensor) => mul(mul(x, div(one, sqrt(add(mean(square(x), 2, true), epsilon)))), w)
  const repeatIds = tensor(Array.from({ length: heads }, (_, i) => Math.floor(i / (heads / kvHeads))), { dtype: 'f32', device })
  const outputWeight = transpose(weights.tokenEmbedding, 0, 1)
  function validate(ids: readonly number[]) {
    if (!ids.length || ids.length > contextLength || ids.some(id => !Number.isInteger(id) || id < 0 || id >= vocabSize))
      throw Error('Expected nonempty valid token IDs within the Llama context limit')
  }
  type Cache = { length: number; layers: { key: Tensor; value: Tensor }[] }
  function run(ids: readonly number[], past?: Cache, retainCache = false) {
    validate(ids)
    const offset = past?.length ?? 0, length = ids.length, total = offset + length
    if (total > contextLength) throw Error('Llama cache exceeds context limit')
    return no_grad(() => {
      let x = reshape(index_select(weights.tokenEmbedding, 0, tensor(Array.from(ids), { dtype: 'f32', device })), [1, length, d])
      const angles = Array.from({ length }, (_, t) => Array.from({ length: head }, (_, i) => (offset + t) / Math.pow(config.ropeTheta, 2 * (i % (head / 2)) / head)))
      const cos = tensor([angles.map(row => [row.map(Math.cos)])], { dtype: 'f32', device })
      const sin = tensor([angles.map(row => [row.map(Math.sin)])], { dtype: 'f32', device })
      const rope = (value: Tensor, nHeads: number) => {
        const v = reshape(value, [1, length, nHeads, head])
        const rotated = cat([neg(contiguous(v.slice([':', ':', ':', `${head / 2}:`]))), contiguous(v.slice([':', ':', ':', `0:${head / 2}`]))], 3)
        return add(mul(v, cos), mul(rotated, sin))
      }
      const mask = length === 1 ? null : tensor(Array.from({ length }, (_, row) => Array.from({ length: total }, (_, col) => col > offset + row ? 1 : 0)), { dtype: 'f32', device })
      const cache: Cache = { length: total, layers: [] }, hidden_states: Tensor[] = []
      for (let i = 0; i < layers; i++) {
        hidden_states.push(x)
        const b = weights.blocks[i], normalized = norm(x, b.attentionNorm)
        const q = permute(rope(matmul(normalized, b.query), heads), [0, 2, 1, 3])
        const nextKey = reshape(rope(matmul(normalized, b.key), kvHeads), [1, length, kvWidth])
        const nextValue = matmul(normalized, b.value)
        const key = past ? cat([past.layers[i].key, nextKey], 1) : nextKey
        const value = past ? cat([past.layers[i].value, nextValue], 1) : nextValue
        if (retainCache) cache.layers.push({ key, value })
        const expand = (v: Tensor) => index_select(permute(reshape(v, [1, total, kvHeads, head]), [0, 2, 1, 3]), 1, repeatIds)
        const scores = mul(matmul(q, permute(expand(key), [0, 1, 3, 2])), scale)
        const attention = matmul(softmax(mask ? masked_fill(scores, mask, -3.4028234663852886e38) : scores, 3), expand(value))
        x = add(x, matmul(reshape(contiguous(permute(attention, [0, 2, 1, 3])), [1, length, d]), b.attentionOutput))
        const normalizedFF = norm(x, b.feedForwardNorm)
        x = add(x, matmul(mul(silu(matmul(normalizedFF, b.gate)), matmul(normalizedFF, b.up)), b.down))
      }
      x = norm(x, weights.finalNorm); hidden_states.push(x)
      return { logits: matmul(x, outputWeight), hidden_states, cache }
    })
  }
  function forward(ids: readonly number[]) {
    const { logits, hidden_states } = run(ids)
    return { logits, hidden_states }
  }
  function create_session() {
    let cache: Cache | undefined
    return {
      get length() { return cache?.length ?? 0 },
      reset() { cache = undefined },
      forward(ids: readonly number[]) {
        const result = run(ids, cache, true)
        cache = result.cache
        return { logits: result.logits, hidden_states: result.hidden_states }
      },
    }
  }
  function generate(ids: readonly number[], max_new_tokens: number, options: { use_cache?: boolean } = {}) {
    validate(ids)
    if (!Number.isInteger(max_new_tokens) || max_new_tokens < 0 || ids.length + max_new_tokens > contextLength) throw Error('Generation budget must be nonnegative and fit within the context limit')
    const output = Array.from(ids), session = options.use_cache === false ? null : create_session()
    try {
      for (let i = 0; i < max_new_tokens; i++) {
        const input = session && i ? [output[output.length - 1]] : output
        const logits = session ? session.forward(input).logits : forward(input).logits
        const row = (logits.slice([0, input.length - 1, ':']).to_array() as number[]).flat(Infinity) as number[]
        let best = 0
        for (let j = 1; j < row.length; j++) if (row[j] > row[best]) best = j
        output.push(best)
        if (best === config.eosTokenId) break
      }
      return output
    } finally { session?.reset() }
  }
  return { config, forward, create_session, generate }
}
