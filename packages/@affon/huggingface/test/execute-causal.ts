import { Session, type Tensor } from 'affon:compute'
import type { ModelTensor } from '../../models/src/shared/parameters.ts'

export interface CausalDefinition {
  config: { vocab_size?: number; n_positions?: number; max_position_embeddings?: number; eos_token_id?: number }
  parameters: Readonly<Record<string, ModelTensor>>
  forward(length: number, outputStart?: number): import('affon:compute').Program
}

export function execute_causal(model: CausalDefinition, ids: readonly number[], outputStart = 0, device = 'cpu') {
  const session = new Session({ device: device as any })
  const source = model.forward(ids.length, outputStart)
  const state = session.initialize(source, { parameters: model.parameters })
  const inputs: Record<string, Tensor> = {
    ids: session.tensor(Array.from(ids), { dtype: 'i64' }),
    mask: session.tensor([[Array.from({ length: ids.length }, (_, row) => Array.from({ length: ids.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' }),
  }
  if (source.inspect().arguments.some(argument => argument.name === 'positions')) {
    inputs.positions = session.tensor(Array.from({ length: ids.length }, (_, index) => index), { dtype: 'i64' })
  }
  try {
    const [logits, ...hidden_states] = session.compile(source).run(inputs, state) as Tensor[]
    return { logits: logits.to_array(), hidden_states: hidden_states.map(value => value.to_array()) }
  } finally {
    for (const input of Object.values(inputs)) input.dispose()
    state.dispose()
    session.dispose()
  }
}

export function generate_causal(model: CausalDefinition, ids: readonly number[], budget: number, device = 'cpu') {
  const context = model.config.n_positions ?? model.config.max_position_embeddings ?? 0
  if (!ids.length || !Number.isInteger(budget) || budget < 0 || ids.length + budget > context) throw Error('Invalid generation request')
  const output = Array.from(ids)
  for (let index = 0; index < budget; index++) {
    const logits = execute_causal(model, output, output.length - 1, device).logits as number[][][]
    const row = logits[0][0]
    let best = 0
    for (let candidate = 1; candidate < row.length; candidate++) if (row[candidate] > row[best]) best = candidate
    output.push(best)
    if (best === model.config.eos_token_id) break
  }
  return output
}
