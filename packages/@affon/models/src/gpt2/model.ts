import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import {
  add, cat, contiguous, layer_norm, div, gelu, index_select, masked_fill, matmul,
  mul, no_grad, permute, reshape, softmax, tensor, transpose,
} from 'affon:compute'
import type { Device, Tensor } from 'affon:compute'
import { causal_mask } from '../shared/sequence.ts'
import type { AffineWeights } from '../shared/encoder.ts'


export interface GPT2Config { width: number; heads: number; layers: number; contextLength: number; vocabSize: number; epsilon: number; eosTokenId?: number; scaleAttention?: boolean }
export interface GPT2Weights {
  tokenEmbedding: Tensor; positionEmbedding: Tensor; finalNorm: AffineWeights
  blocks: { attentionNorm: AffineWeights; feedForwardNorm: AffineWeights; qkv: AffineWeights; attentionOutput: AffineWeights; expand: AffineWeights; contract: AffineWeights }[]
}

/** Construct f32 GPT2 from normalized configuration and prepared tensors.
 * Affine weights use [input, output] layout; normalization uses vectors.
 * Tensors must be on the requested device and remain alive with the model.
 */
export function create_gpt2(config: GPT2Config, weights: GPT2Weights, device: Device = 'cpu') {
  if (!Number.isInteger(config.width) || config.width <= 0 || !Number.isInteger(config.heads) || config.heads <= 0 || config.width % config.heads || !Number.isInteger(config.layers) || config.layers <= 0 || weights.blocks.length !== config.layers || !Number.isFinite(config.epsilon) || config.epsilon <= 0) throw new Error('Invalid model dimensions or normalization')
  const check = parameter_checks(device)
  const width = config.width
  positive_dimensions(config.contextLength, config.vocabSize)
  if (config.eosTokenId !== undefined && (!Number.isInteger(config.eosTokenId) || config.eosTokenId < 0 || config.eosTokenId >= config.vocabSize)) throw new Error('Invalid EOS token')
  check.tensor(weights.tokenEmbedding, [config.vocabSize, width])
  check.tensor(weights.positionEmbedding, [config.contextLength, width])
  check.affine(weights.finalNorm, [width])
  for (const block of weights.blocks) {
    const inner = block.expand?.bias?.shape[0]
    positive_dimensions(inner)
    check.affine(block.attentionNorm, [width]); check.affine(block.feedForwardNorm, [width])
    check.affine(block.qkv, [width, 3 * width]); check.affine(block.attentionOutput, [width, width])
    check.affine(block.expand, [width, inner]); check.affine(block.contract, [inner, width])
  }
  const d = config.width
  const scalar = (value: number) => tensor(value, { dtype: 'f32', device })
  const norm = (x: Tensor, params: AffineWeights) => add(mul(
    layer_norm(x, 2, config.epsilon), params.weight), params.bias)
  const linear = (x: Tensor, params: AffineWeights) => add(matmul(x, params.weight), params.bias)

  function validateIds(ids: readonly number[]) {
    if (ids.length === 0 || ids.length > config.contextLength || ids.some(id => !Number.isInteger(id) || id < 0 || id >= config.vocabSize)) {
      throw new Error('Expected nonempty valid token IDs within the GPT-2 context limit')
    }
  }

  type Cache = { length: number; layers: { key: Tensor; value: Tensor }[] }
  function run(ids: readonly number[], past?: Cache, cache_enabled = false) {
    validateIds(ids)
    const offset = past?.length ?? 0
    if (offset + ids.length > config.contextLength) throw Error('GPT-2 cache exceeds context limit')
    return no_grad(() => {
      const length = ids.length
      const tokenIds = tensor(Array.from(ids), { dtype: 'f32', device })
      const positions = tensor(Array.from({ length }, (_, i) => offset + i), { dtype: 'f32', device })
      let x = reshape(add(index_select(weights.tokenEmbedding, 0, tokenIds),
        index_select(weights.positionEmbedding, 0, positions)), [1, length, d])
      const hidden_states: Tensor[] = []
      const head = d / config.heads
      const mask = offset === 0 ? causal_mask(length, { device }) : length === 1 ? null
        : tensor(Array.from({ length }, (_, row) => Array.from({ length: offset + length }, (_, col) => col > offset + row ? 1 : 0)), { dtype: 'f32', device })
      const cache: Cache = { length: offset + length, layers: [] }
      for (let i = 0; i < config.layers; i++) {
        hidden_states.push(x)
        const block = weights.blocks[i]
        const qkv = linear(norm(x, block.attentionNorm), block.qkv)
        const split = (offset: number) => contiguous(qkv.slice([':', ':', `${offset}:${offset + d}`]))
        const split_heads = (value: Tensor, tokens: number) => permute(reshape(value,
          [1, tokens, config.heads, head]), [0, 2, 1, 3])
        // Store sequence-major K/V so append is one contiguous copy per input,
        // rather than a separate Metal buffer copy for every attention head.
        const key = past ? cat([past.layers[i].key, split(d)], 1) : split(d)
        const value = past ? cat([past.layers[i].value, split(2 * d)], 1) : split(2 * d)
        if (cache_enabled) cache.layers.push({ key, value })
        const q = split_heads(split(0), length)
        const k = split_heads(key, offset + length), v = split_heads(value, offset + length)
        let scores = matmul(q, permute(k, [0, 1, 3, 2]))
        if (config.scaleAttention !== false) scores = div(scores, scalar(Math.sqrt(head)))
        const attention = matmul(softmax(mask ? masked_fill(scores, mask, -3.4028234663852886e38) : scores, 3), v)
        const merged = reshape(contiguous(permute(attention, [0, 2, 1, 3])), [1, length, d])
        x = add(x, linear(merged, block.attentionOutput))
        x = add(x, linear(gelu(linear(norm(x, block.feedForwardNorm), block.expand)), block.contract))
      }
      x = norm(x, weights.finalNorm)
      hidden_states.push(x)
      return { logits: matmul(x, transpose(weights.tokenEmbedding, 0, 1)), hidden_states, cache }
    })
  }

  function forward(ids: readonly number[]): { logits: Tensor; hidden_states: Tensor[] } {
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
        // Commit only after a successful forward; invalid inputs preserve the session.
        cache = result.cache
        return { logits: result.logits, hidden_states: result.hidden_states }
      },
    }
  }

  function generate(ids: readonly number[], max_new_tokens: number, options: { use_cache?: boolean } = {}): number[] {
    validateIds(ids)
    if (!Number.isInteger(max_new_tokens) || max_new_tokens < 0 || ids.length + max_new_tokens > config.contextLength) {
      throw new Error('Generation budget must be nonnegative and fit within the context limit')
    }
    const output = Array.from(ids)
    const session = options.use_cache === false ? null : create_session()
    try {
    for (let i = 0; i < max_new_tokens; i++) {
      const input = session && i > 0 ? [output[output.length - 1]] : output
      const result = session ? session.forward(input).logits : forward(input).logits
      const row = (result.slice([0, input.length - 1, ':']).to_array() as number[]).flat(Infinity) as number[]
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
