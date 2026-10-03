import { add, contiguous, div, embedding, gelu, layer_norm, masked_fill, matmul, mul, reshape, slice, softmax, transpose } from 'affon:ops'
import { Session, Tensor, program, type Device, type ExecutionState, type FormalTensor, type Program } from 'affon:compute'
import { positive_dimensions, parameter_checks } from '../shared/parameters.ts'
import type { ModelAffineWeights, ModelTensor } from '../shared/parameters.ts'

export interface GPT2Config { width: number; heads: number; layers: number; contextLength: number; vocabSize: number; epsilon: number; eosTokenId?: number; scaleAttention?: boolean }
export interface GPT2Weights {
  tokenEmbedding: ModelTensor; positionEmbedding: ModelTensor; finalNorm: ModelAffineWeights
  blocks: { attentionNorm: ModelAffineWeights; feedForwardNorm: ModelAffineWeights; qkv: ModelAffineWeights; attentionOutput: ModelAffineWeights; expand: ModelAffineWeights; contract: ModelAffineWeights }[]
}

type CompiledGPT2 = { source: Program; state: ExecutionState }

/** Construct f32 GPT-2 as shape-specialized Programs backed by one Session. */
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

  const session = new Session({ device })
  const compiled = new Map<string, CompiledGPT2>()
  const parameterValues: Record<string, ModelTensor> = {
    token_embedding: weights.tokenEmbedding,
    position_embedding: weights.positionEmbedding,
    final_norm_weight: weights.finalNorm.weight,
    final_norm_bias: weights.finalNorm.bias,
  }
  weights.blocks.forEach((block, index) => {
    for (const [part, affine] of Object.entries(block) as [string, ModelAffineWeights][]) {
      parameterValues[`block_${index}_${part}_weight`] = affine.weight
      parameterValues[`block_${index}_${part}_bias`] = affine.bias
    }
  })

  const parameter = (p: any, name: string, shape: readonly number[]) => p.parameter(name, Tensor.f32(shape)) as FormalTensor
  const affine = (p: any, name: string, shape: readonly number[]) => ({
    weight: parameter(p, `${name}_weight`, shape),
    bias: parameter(p, `${name}_bias`, shape.length === 1 ? shape : [shape[shape.length - 1]]),
  })

  function build(length: number, outputStart: number): CompiledGPT2 {
    const cacheKey = `${length}:${outputStart}`
    const cached = compiled.get(cacheKey)
    if (cached) return cached
    const source = program(`gpt2_${length}_${outputStart}`, p => {
      const ids = p.argument('ids', Tensor.i64([length]))
      const positions = p.argument('positions', Tensor.i64([length]))
      const mask = p.argument('mask', Tensor.i64([1, 1, length, length]))
      const tokenEmbedding = parameter(p, 'token_embedding', [config.vocabSize, width])
      const positionEmbedding = parameter(p, 'position_embedding', [config.contextLength, width])
      const finalNorm = affine(p, 'final_norm', [width])
      const scalar = (name: string, value: number) => p.constant(name, value, Tensor.f32([1]))
      const dense = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => add(matmul(x, params.weight), params.bias)
      const norm = (x: FormalTensor, params: { weight: FormalTensor; bias: FormalTensor }) => add(mul(layer_norm(x, 2, config.epsilon), params.weight), params.bias)
      let x = reshape(add(embedding(tokenEmbedding, ids), embedding(positionEmbedding, positions)), [1, length, width])
      const hiddenStates: FormalTensor[] = []
      const headWidth = width / config.heads
      const full = (start: number, stop: number) => [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start, stop }]
      const splitHeads = (value: FormalTensor) => transpose(reshape(value, [1, length, config.heads, headWidth]), [0, 2, 1, 3])
      for (let index = 0; index < config.layers; index++) {
        hiddenStates.push(x)
        const prefix = `block_${index}`
        const attentionNorm = affine(p, `${prefix}_attentionNorm`, [width])
        const feedForwardNorm = affine(p, `${prefix}_feedForwardNorm`, [width])
        const qkvWeights = affine(p, `${prefix}_qkv`, [width, 3 * width])
        const attentionOutput = affine(p, `${prefix}_attentionOutput`, [width, width])
        const expand = affine(p, `${prefix}_expand`, [width, weights.blocks[index].expand.bias.shape[0]])
        const contract = affine(p, `${prefix}_contract`, [weights.blocks[index].expand.bias.shape[0], width])
        const qkv = dense(norm(x, attentionNorm), qkvWeights)
        const q = splitHeads(contiguous(slice(qkv, full(0, width))))
        const k = splitHeads(contiguous(slice(qkv, full(width, 2 * width))))
        const v = splitHeads(contiguous(slice(qkv, full(2 * width, 3 * width))))
        let scores = matmul(q, transpose(k, [0, 1, 3, 2]))
        if (config.scaleAttention !== false) scores = div(scores, scalar(`attention_scale_${index}`, Math.sqrt(headWidth)))
        const attention = reshape(contiguous(transpose(matmul(softmax(masked_fill(scores, mask, -3.4028234663852886e38), 3), v), [0, 2, 1, 3])), [1, length, width])
        x = add(x, dense(attention, attentionOutput))
        x = add(x, dense(gelu(dense(norm(x, feedForwardNorm), expand)), contract))
      }
      x = norm(x, finalNorm)
      hiddenStates.push(x)
      const suffix = (value: FormalTensor) => contiguous(slice(value, [{ start: 0, stop: 1 }, { start: outputStart, stop: length }, { start: 0, stop: width }]))
      const logits = matmul(suffix(x), transpose(tokenEmbedding, [1, 0]))
      return [logits, ...hiddenStates.map(suffix)]
    })
    const result = { source, state: session.initialize(source, { parameters: parameterValues }) }
    compiled.set(cacheKey, result)
    return result
  }

  function validateIds(ids: readonly number[]) {
    if (ids.length === 0 || ids.length > config.contextLength || ids.some(id => !Number.isInteger(id) || id < 0 || id >= config.vocabSize)) throw new Error('Expected nonempty valid token IDs within the GPT-2 context limit')
  }

  function execute(ids: readonly number[], outputStart = 0): { logits: Tensor; hidden_states: Tensor[] } {
    validateIds(ids)
    if (outputStart < 0 || outputStart >= ids.length) throw new Error('Invalid GPT-2 output window')
    const compiledProgram = build(ids.length, outputStart)
    const input = session.tensor(Array.from(ids), { dtype: 'i64' }) as Tensor
    const positions = session.tensor(Array.from({ length: ids.length }, (_, index) => index), { dtype: 'i64' }) as Tensor
    const mask = session.tensor([[Array.from({ length: ids.length }, (_, row) => Array.from({ length: ids.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' }) as Tensor
    try {
      const outputs = session.compile(compiledProgram.source).run({ ids: input, positions, mask }, compiledProgram.state) as Tensor[]
      return { logits: outputs[0], hidden_states: outputs.slice(1) }
    } finally {
      input.dispose(); positions.dispose(); mask.dispose()
    }
  }

  function forward(ids: readonly number[]) { return execute(ids) }

  function create_session() {
    let history: number[] = []
    return {
      get length() { return history.length },
      reset() { history = [] },
      forward(ids: readonly number[]) {
        validateIds(ids)
        if (history.length + ids.length > config.contextLength) throw Error('GPT-2 cache exceeds context limit')
        const next = [...history, ...ids]
        const result = execute(next, history.length)
        history = next
        return result
      },
    }
  }

  function generate(ids: readonly number[], max_new_tokens: number, options: { use_cache?: boolean } = {}): number[] {
    validateIds(ids)
    if (!Number.isInteger(max_new_tokens) || max_new_tokens < 0 || ids.length + max_new_tokens > config.contextLength) throw new Error('Generation budget must be nonnegative and fit within the context limit')
    const output = Array.from(ids)
    const decode = options.use_cache === false ? null : create_session()
    for (let index = 0; index < max_new_tokens; index++) {
      const input = decode && index > 0 ? [output[output.length - 1]] : output
      const result = decode ? decode.forward(input) : forward(input)
      try {
        const rows = result.logits.to_array() as number[][][]
        const row = rows[0][rows[0].length - 1]
        let best = 0
        for (let candidate = 1; candidate < row.length; candidate++) if (row[candidate] > row[best]) best = candidate
        output.push(best)
        if (best === config.eosTokenId) break
      } finally {
        result.logits.dispose()
        for (const hidden of result.hidden_states) hidden.dispose()
      }
    }
    return output
  }

  function dispose() {
    for (const value of compiled.values()) value.state.dispose()
    compiled.clear()
    session.dispose()
  }

  return { config, forward, create_session, generate, dispose }
}
