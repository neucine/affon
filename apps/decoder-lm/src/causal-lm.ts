import type { ExecutionState, Session, Tensor } from 'affon:compute'
import { decoderProgram, type DecoderModel } from './model.ts'

export interface GenerateOptions {
  max_new_tokens?: number
  forbidden_token_ids?: number[]
  temperature?: number
  top_k?: number
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

/** Host-side token selection around explicit Program execution. */
export function generate(model: DecoderModel, session: Session, state: ExecutionState, tokenIds: Tensor, opts: GenerateOptions): Tensor {
  if (!session.owns(tokenIds)) throw new AffonError('invalid_arg', 'generate tokenIds must belong to the supplied Session')
  if (tokenIds.ndim !== 2) throw new AffonError('invalid_shape', 'generate expects token ids shaped [batch, seq]')
  const resolved = options(opts)
  const rows = tokenIds.to_array() as number[][]
  if (resolved.forbidden.size >= model.vocabSize || [...resolved.forbidden].some(id => !Number.isInteger(id) || id < 0 || id >= model.vocabSize)) throw new AffonError('invalid_arg', 'generate forbidden_token_ids must be valid and leave at least one token')
  for (let step = 0; step < resolved.max; step++) {
    const input = session.tensor(rows, { dtype: 'i64', axes: ['batch', 'token'] })
    const logits = session.compile(decoderProgram(model, rows.length, rows[0].length)).run({ token_ids: input }, state) as Tensor
    try {
      const values = logits.to_array() as number[][][]
      for (let batch = 0; batch < rows.length; batch++) rows[batch].push(choose(values[batch].at(-1)!, resolved))
    } finally { logits.dispose(); input.dispose() }
  }
  return session.tensor(rows, { dtype: 'i64', axes: ['batch', 'token'] })
}
