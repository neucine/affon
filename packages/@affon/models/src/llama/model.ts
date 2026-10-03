import { add, cat, contiguous, div, embedding, index_select, masked_fill, matmul, mean, mul, neg, reshape, silu, slice, softmax, sqrt, transpose } from 'affon:ops'
import { Session, Tensor, program, type Device, type ExecutionState, type FormalTensor, type Program } from 'affon:compute'
import { parameter_checks, positive_dimensions } from '../shared/parameters.ts'
import type { ModelTensor } from '../shared/parameters.ts'

export interface LlamaConfig {
  width: number; innerWidth: number; heads: number; kvHeads: number; layers: number
  contextLength: number; vocabSize: number; epsilon: number; ropeTheta: number; eosTokenId?: number
}
export interface LlamaWeights {
  tokenEmbedding: ModelTensor; finalNorm: ModelTensor
  blocks: { attentionNorm: ModelTensor; feedForwardNorm: ModelTensor; query: ModelTensor; key: ModelTensor; value: ModelTensor; attentionOutput: ModelTensor; gate: ModelTensor; up: ModelTensor; down: ModelTensor }[]
}

type CompiledLlama = { source: Program; state: ExecutionState }

/** Tied-head, bias-free Llama decoder executed as shape-specialized Programs. */
export function create_llama(config: LlamaConfig, weights: LlamaWeights, device: Device = 'cpu') {
  const { width, innerWidth, heads, kvHeads, layers, contextLength, vocabSize } = config
  positive_dimensions(width, innerWidth, heads, kvHeads, layers, contextLength, vocabSize)
  if (width % heads || heads % kvHeads || (width / heads) % 2 || weights.blocks.length !== layers || !Number.isFinite(config.epsilon) || config.epsilon <= 0 || !Number.isFinite(config.ropeTheta) || config.ropeTheta <= 0) throw Error('Invalid Llama dimensions, RoPE or normalization')
  if (config.eosTokenId !== undefined && (!Number.isInteger(config.eosTokenId) || config.eosTokenId < 0 || config.eosTokenId >= vocabSize)) throw Error('Invalid EOS token')
  const headWidth = width / heads, kvWidth = kvHeads * headWidth
  const check = parameter_checks(device).tensor
  check(weights.tokenEmbedding, [vocabSize, width]); check(weights.finalNorm, [width])
  for (const block of weights.blocks) {
    check(block.attentionNorm, [width]); check(block.feedForwardNorm, [width])
    check(block.query, [width, width]); check(block.key, [width, kvWidth]); check(block.value, [width, kvWidth]); check(block.attentionOutput, [width, width])
    check(block.gate, [width, innerWidth]); check(block.up, [width, innerWidth]); check(block.down, [innerWidth, width])
  }

  const session = new Session({ device })
  const compiled = new Map<string, CompiledLlama>()
  const parameterValues: Record<string, ModelTensor> = { token_embedding: weights.tokenEmbedding, final_norm: weights.finalNorm }
  weights.blocks.forEach((block, index) => {
    for (const [part, value] of Object.entries(block) as [string, ModelTensor][]) parameterValues[`block_${index}_${part}`] = value
  })
  const parameter = (p: any, name: string, shape: readonly number[]) => p.parameter(name, Tensor.f32(shape)) as FormalTensor

  function build(length: number, outputStart: number): CompiledLlama {
    const cacheKey = `${length}:${outputStart}`
    const cached = compiled.get(cacheKey)
    if (cached) return cached
    const source = program(`llama_${length}_${outputStart}`, p => {
      const ids = p.argument('ids', Tensor.i64([length]))
      const mask = p.argument('mask', Tensor.i64([1, 1, length, length]))
      const tokenEmbedding = parameter(p, 'token_embedding', [vocabSize, width])
      const finalNorm = parameter(p, 'final_norm', [width])
      const scalar = (name: string, value: number) => p.constant(name, value, Tensor.f32([1]))
      const epsilon = scalar('epsilon', config.epsilon)
      const one = scalar('one', 1)
      const scale = scalar('attention_scale', 1 / Math.sqrt(headWidth))
      const repeatIds = p.constant('repeat_ids', Array.from({ length: heads }, (_, index) => Math.floor(index / (heads / kvHeads))), Tensor.i64([heads]))
      const angles = Array.from({ length }, (_, token) => Array.from({ length: headWidth }, (_, index) => token / Math.pow(config.ropeTheta, 2 * (index % (headWidth / 2)) / headWidth)))
      const cosine = p.constant('rope_cosine', [angles.map(row => [row.map(Math.cos)])], Tensor.f32([1, length, 1, headWidth]))
      const sine = p.constant('rope_sine', [angles.map(row => [row.map(Math.sin)])], Tensor.f32([1, length, 1, headWidth]))
      const norm = (x: FormalTensor, weight: FormalTensor) => mul(mul(x, div(one, sqrt(add(mean(mul(x, x), 2, true), epsilon)))), weight)
      const rope = (value: FormalTensor, headCount: number) => {
        const reshaped = reshape(value, [1, length, headCount, headWidth])
        const first = contiguous(slice(reshaped, [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start: 0, stop: headCount }, { start: 0, stop: headWidth / 2 }]))
        const second = contiguous(slice(reshaped, [{ start: 0, stop: 1 }, { start: 0, stop: length }, { start: 0, stop: headCount }, { start: headWidth / 2, stop: headWidth }]))
        return add(mul(reshaped, cosine), mul(cat([neg(second), first], 3), sine))
      }
      const expand = (value: FormalTensor) => index_select(transpose(reshape(value, [1, length, kvHeads, headWidth]), [0, 2, 1, 3]), 1, repeatIds)
      let x = reshape(embedding(tokenEmbedding, ids), [1, length, width])
      const hiddenStates: FormalTensor[] = []
      for (let index = 0; index < layers; index++) {
        hiddenStates.push(x)
        const prefix = `block_${index}`
        const attentionNorm = parameter(p, `${prefix}_attentionNorm`, [width])
        const feedForwardNorm = parameter(p, `${prefix}_feedForwardNorm`, [width])
        const query = parameter(p, `${prefix}_query`, [width, width])
        const key = parameter(p, `${prefix}_key`, [width, kvWidth])
        const value = parameter(p, `${prefix}_value`, [width, kvWidth])
        const attentionOutput = parameter(p, `${prefix}_attentionOutput`, [width, width])
        const gate = parameter(p, `${prefix}_gate`, [width, innerWidth])
        const up = parameter(p, `${prefix}_up`, [width, innerWidth])
        const down = parameter(p, `${prefix}_down`, [innerWidth, width])
        const normalized = norm(x, attentionNorm)
        const q = transpose(rope(matmul(normalized, query), heads), [0, 2, 1, 3])
        const k = reshape(rope(matmul(normalized, key), kvHeads), [1, length, kvWidth])
        const v = matmul(normalized, value)
        const scores = masked_fill(mul(matmul(q, transpose(expand(k), [0, 1, 3, 2])), scale), mask, -3.4028234663852886e38)
        const attention = reshape(contiguous(transpose(matmul(softmax(scores, 3), expand(v)), [0, 2, 1, 3])), [1, length, width])
        x = add(x, matmul(attention, attentionOutput))
        const normalizedFeedForward = norm(x, feedForwardNorm)
        x = add(x, matmul(mul(silu(matmul(normalizedFeedForward, gate)), matmul(normalizedFeedForward, up)), down))
      }
      x = norm(x, finalNorm)
      hiddenStates.push(x)
      const suffix = (value: FormalTensor) => contiguous(slice(value, [{ start: 0, stop: 1 }, { start: outputStart, stop: length }, { start: 0, stop: width }]))
      return [matmul(suffix(x), transpose(tokenEmbedding, [1, 0])), ...hiddenStates.map(suffix)]
    })
    const result = { source, state: session.initialize(source, { parameters: parameterValues }) }
    compiled.set(cacheKey, result)
    return result
  }

  function validate(ids: readonly number[]) {
    if (!ids.length || ids.length > contextLength || ids.some(id => !Number.isInteger(id) || id < 0 || id >= vocabSize)) throw Error('Expected nonempty valid token IDs within the Llama context limit')
  }

  function execute(ids: readonly number[], outputStart = 0): { logits: Tensor; hidden_states: Tensor[] } {
    validate(ids)
    const compiledProgram = build(ids.length, outputStart)
    const input = session.tensor(Array.from(ids), { dtype: 'i64' }) as Tensor
    const mask = session.tensor([[Array.from({ length: ids.length }, (_, row) => Array.from({ length: ids.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' }) as Tensor
    try {
      const outputs = session.compile(compiledProgram.source).run({ ids: input, mask }, compiledProgram.state) as Tensor[]
      return { logits: outputs[0], hidden_states: outputs.slice(1) }
    } finally {
      input.dispose(); mask.dispose()
    }
  }

  const forward = (ids: readonly number[]) => execute(ids)
  function create_session() {
    let history: number[] = []
    return {
      get length() { return history.length },
      reset() { history = [] },
      forward(ids: readonly number[]) {
        validate(ids)
        if (history.length + ids.length > contextLength) throw Error('Llama cache exceeds context limit')
        const next = [...history, ...ids]
        const result = execute(next, history.length)
        history = next
        return result
      },
    }
  }
  function generate(ids: readonly number[], max_new_tokens: number, options: { use_cache?: boolean } = {}) {
    validate(ids)
    if (!Number.isInteger(max_new_tokens) || max_new_tokens < 0 || ids.length + max_new_tokens > contextLength) throw Error('Generation budget must be nonnegative and fit within the context limit')
    const output = Array.from(ids), decode = options.use_cache === false ? null : create_session()
    for (let index = 0; index < max_new_tokens; index++) {
      const input = decode && index ? [output[output.length - 1]] : output
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
    compiled.clear(); session.dispose()
  }
  return { config, forward, create_session, generate, dispose }
}
