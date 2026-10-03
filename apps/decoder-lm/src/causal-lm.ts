import { Session, type Tensor } from 'affon:compute'
import type { DecoderModelModule } from './model.ts'

export type DecoderLMForward = (tokenIds: Tensor) => Tensor

export interface GenerateOptions {
  max_new_tokens?: number
  forbidden_token_ids?: number[]
  temperature?: number
  top_k?: number
}

function flat(value: Tensor): number[] { return (value.to_array() as any[]).flat(Infinity).map(Number) }

function causalLoss(logits: Tensor, tokenIds: Tensor): number {
  if (logits.ndim !== 3) throw new AffonError('invalid_shape', 'CausalLMLoss expects logits shaped [batch, seq, vocab]')
  if (tokenIds.ndim !== 2) throw new AffonError('invalid_shape', 'CausalLMLoss expects token ids shaped [batch, seq + 1]')
  const [batch, sequence, vocabulary] = logits.shape
  if (tokenIds.shape[0] !== batch || tokenIds.shape[1] !== sequence + 1) throw new AffonError('shape_mismatch', 'CausalLMLoss token dimensions do not match logits')
  const scores = flat(logits), ids = flat(tokenIds)
  let loss = 0
  for (let row = 0; row < batch * sequence; row++) {
    const target = ids[Math.floor(row / sequence) * (sequence + 1) + row % sequence + 1]
    if (!Number.isInteger(target) || target < 0 || target >= vocabulary) throw new AffonError('invalid_arg', 'CausalLMLoss target is outside the vocabulary')
    const start = row * vocabulary
    let maximum = -Infinity
    for (let index = 0; index < vocabulary; index++) maximum = Math.max(maximum, scores[start + index])
    let denominator = 0
    for (let index = 0; index < vocabulary; index++) denominator += Math.exp(scores[start + index] - maximum)
    loss += Math.log(denominator) + maximum - scores[start + target]
  }
  return loss / (batch * sequence)
}

export function causal_lm_eval_loss_forward(logits: Tensor, tokenIds: Tensor): Tensor {
  const session = new Session({ device: logits.device })
  const result = session.tensor([causalLoss(logits, tokenIds)])
  const dispose = result.dispose.bind(result)
  let disposed = false
  result.dispose = () => {
    if (disposed) return
    disposed = true
    dispose()
    session.dispose()
  }
  return result
}

export function CausalLMLoss(): (logits: Tensor, tokenIds: Tensor) => Tensor {
  return causal_lm_eval_loss_forward
}

function options(value: GenerateOptions) {
  const max = value.max_new_tokens
  if (!Number.isInteger(max) || max! <= 0) throw new AffonError('invalid_arg', 'generate max_new_tokens must be a positive integer')
  const temperature = value.temperature ?? 0
  if (!Number.isFinite(temperature) || temperature < 0) throw new AffonError('invalid_arg', 'generate temperature must be non-negative')
  if (value.top_k !== undefined && (!Number.isInteger(value.top_k) || value.top_k <= 0)) throw new AffonError('invalid_arg', 'generate top_k must be positive')
  return { max: max!, temperature, topK: value.top_k, forbidden: new Set(value.forbidden_token_ids ?? []) }
}

function choose(logits: number[], resolved: ReturnType<typeof options>): number {
  const allowed = logits.map((score, id) => ({ score, id })).filter(entry => !resolved.forbidden.has(entry.id)).sort((a, b) => b.score - a.score)
  if (!allowed.length) throw new AffonError('invalid_arg', 'generate forbidden_token_ids excludes the full vocabulary')
  if (resolved.temperature === 0) return allowed[0].id
  const candidates = allowed.slice(0, resolved.topK ?? allowed.length)
  const maximum = candidates[0].score
  const weights = candidates.map(entry => Math.exp((entry.score - maximum) / resolved.temperature))
  const total = weights.reduce((sum, weight) => sum + weight, 0)
  let sample = Math.random() * total
  for (let index = 0; index < candidates.length; index++) { sample -= weights[index]; if (sample <= 0) return candidates[index].id }
  return candidates.at(-1)!.id
}

export function generate(model: DecoderModelModule, tokenIds: Tensor, opts: GenerateOptions, forward?: DecoderLMForward): Tensor {
  if (tokenIds.ndim !== 2) throw new AffonError('invalid_shape', 'generate expects token ids shaped [batch, seq]')
  const resolved = options(opts)
  const rows = tokenIds.to_array() as number[][]
  if (resolved.forbidden.size >= model.vocabSize || [...resolved.forbidden].some(id => !Number.isInteger(id) || id < 0 || id >= model.vocabSize)) throw new AffonError('invalid_arg', 'generate forbidden_token_ids must be valid and leave at least one token')
  const decode = forward ?? model.forward
  for (let step = 0; step < resolved.max; step++) {
    const input = model.session.tensor(rows, { dtype: 'i64' })
    const logits = decode(input)
    try {
      const values = logits.to_array() as number[][][]
      for (let batch = 0; batch < rows.length; batch++) rows[batch].push(choose(values[batch].at(-1)!, resolved))
    } finally { logits.dispose(); input.dispose() }
  }
  return model.session.tensor(rows, { dtype: 'i64', axes: ['batch', 'token'] })
}
