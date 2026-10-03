import { Session, type Device, type Executable, type ExecutionState, type Program, type Tensor, type TensorData } from 'affon:compute'
import type { ModelTensor } from '../../../../packages/@affon/models/src/shared/parameters.ts'

export interface ProgramDefinition {
  parameters: Readonly<Record<string, ModelTensor>>
}

export interface GraphProgramDefinition extends ProgramDefinition {
  forward: Program
  output_names: readonly string[]
}

type ProgramInput = ModelTensor | { data: TensorData; dtype?: 'f32' | 'i64' }

export class GraphProgramRuntime {
  readonly session: Session
  readonly state: ExecutionState
  readonly executable: Executable

  constructor(readonly graph: GraphProgramDefinition, device: Device) {
    this.session = new Session({ device })
    try {
      this.state = this.session.initialize(graph.forward, { parameters: graph.parameters })
      this.executable = this.session.compile(graph.forward)
    } catch (error) { this.session.dispose(); throw error }
  }

  forward(values: Readonly<Record<string, ProgramInput>>) {
    const inputs = Object.fromEntries(Object.entries(values).map(([name, value]) => [name,
      'data' in value
        ? this.session.tensor(value.data, { dtype: value.dtype ?? 'f32' })
        : this.session.tensor(value.to_array(), { dtype: value.dtype }),
    ])) as Record<string, Tensor>
    try {
      const result = this.executable.run(inputs, this.state) as Tensor | Tensor[]
      const outputs = Array.isArray(result) ? result : [result]
      return Object.fromEntries(this.graph.output_names.map((name, index) => [name, outputs[index]])) as Record<string, Tensor>
    } finally { for (const input of Object.values(inputs)) input.dispose() }
  }

  dispose() { this.state.dispose(); this.session.dispose() }
}

export interface WhisperProgramDefinition {
  generation: {
    width: number
    heads: number
    layers: number
    vocabSize: number
    contextLength: number
    cacheCapacity: number
    prefix: number[]
    eosTokenId: number
    suppressTokens: number[]
    beginSuppressTokens: number[]
  }
  encoder: GraphProgramDefinition
  cross: GraphProgramDefinition
  decoder: GraphProgramDefinition
  embeddings: ModelTensor
  positions: ModelTensor
  decode(ids: number[]): string
}

/** App-owned greedy policy over three device-neutral Whisper Programs. */
export class WhisperProgramRuntime {
  readonly encoder: GraphProgramRuntime
  readonly cross: GraphProgramRuntime
  readonly decoder: GraphProgramRuntime
  readonly max_new_tokens: number
  private readonly embeddingValues: number[][]
  private readonly positionValues: number[][]

  constructor(readonly model: WhisperProgramDefinition, device: Device) {
    const config = model.generation
    if (!config.width || !config.heads || !config.layers || !config.vocabSize || !config.contextLength ||
      config.width % config.heads || config.cacheCapacity > config.contextLength ||
      config.prefix.length < 1 || config.prefix.length >= config.cacheCapacity) throw Error('Invalid Whisper dimensions or prefix')
    const valid = (id: number) => Number.isInteger(id) && id >= 0 && id < config.vocabSize
    if (![...config.prefix, config.eosTokenId, ...config.suppressTokens, ...config.beginSuppressTokens].every(valid)) throw Error('Invalid Whisper generation tokens')
    this.embeddingValues = model.embeddings.to_array() as number[][]
    this.positionValues = model.positions.to_array() as number[][]
    if (this.embeddingValues.length !== config.vocabSize || this.embeddingValues.some(row => row.length !== config.width) ||
      this.positionValues.length !== config.contextLength || this.positionValues.some(row => row.length !== config.width)) throw Error('Invalid Whisper parameter shapes')
    this.encoder = new GraphProgramRuntime(model.encoder, device)
    try { this.cross = new GraphProgramRuntime(model.cross, device) }
    catch (error) { this.encoder.dispose(); throw error }
    try { this.decoder = new GraphProgramRuntime(model.decoder, device) }
    catch (error) { this.cross.dispose(); this.encoder.dispose(); throw error }
    this.max_new_tokens = config.cacheCapacity - config.prefix.length
  }

  transcribe(features: ModelTensor, on_step?: (step: number, logits: number[]) => void, on_phase?: (phase: string, elapsed_ms: number) => void) {
    const config = this.model.generation
    const phase = <T>(name: string, run: () => T): T => {
      if (!on_phase) return run()
      const start = Date.now(), result = run()
      on_phase(name, Date.now() - start)
      return result
    }
    const started = Date.now()
    const encodedOutput = phase('encoder', () => this.encoder.forward({ features }))
    const encoded = encodedOutput.output
    if (!encoded) throw Error('Whisper encoder omitted output')
    const encoder_ms = Date.now() - started
    let crossValues: Record<string, Tensor>
    try { crossValues = phase('cross', () => this.cross.forward({ encoded })) }
    finally { for (const value of new Set(Object.values(encodedOutput))) value.dispose() }
    const ids = [...config.prefix]
    const begin = new Set(config.beginSuppressTokens), suppress = new Set(config.suppressTokens)
    const headWidth = config.width / config.heads
    const pastValues = Array.from({ length: config.layers * 2 }, () => [
      Array.from({ length: config.heads }, () => Array.from({ length: config.cacheCapacity }, () => Array(headWidth).fill(0))),
    ]) as number[][][][][]
    let position = 0
    try {
      while (position < config.cacheCapacity - 1) {
        const inputs: Record<string, ProgramInput> = {
          ...crossValues,
          embeddings: { data: [[this.embeddingValues[ids[position]]]] },
          position: { data: [[this.positionValues[position]]] },
          mask: { data: [[[Array.from({ length: config.cacheCapacity + 1 }, (_, index) => index < position || index === config.cacheCapacity ? 0 : -10000)]]] },
        }
        pastValues.forEach((value, index) => { inputs[`past_${index}`] = { data: value } })
        const output = phase('decoder_graph', () => this.decoder.forward(inputs))
        try {
          phase('cache_update', () => {
            for (let index = 0; index < pastValues.length; index++) {
              const present = output[`present_${index}`]
              if (!present) throw Error(`Whisper decoder omitted present_${index}`)
              const values = present.to_array() as number[][][][]
              const sourcePosition = values[0]?.[0]?.length === 1 ? 0 : position
              for (let head = 0; head < config.heads; head++) pastValues[index][0][head][position] = Array.from(values[0][head][sourcePosition])
            }
          })
          const logits = phase('readback', () => {
            const value = output.logits?.to_array() as any
            const row = Array.isArray(value?.[0]?.[0]) ? value[0][value[0].length - 1] : value
            if (!Array.isArray(row) || row.length !== config.vocabSize) throw Error('Whisper decoder returned invalid logits')
            return row as number[]
          })
          position++
          if (position < config.prefix.length) continue
          const step = position - config.prefix.length
          on_step?.(step, logits)
          const best = phase('selection', () => {
            let selected = -1, score = -Infinity
            for (let index = 0; index < logits.length; index++) if (logits[index] > score && !suppress.has(index) && !(step === 0 && begin.has(index))) { score = logits[index]; selected = index }
            if (selected < 0) throw Error('Whisper produced no finite candidate')
            return selected
          })
          ids.push(best)
          if (best === config.eosTokenId) break
        } finally { for (const value of new Set(Object.values(output))) value.dispose() }
      }
    } finally { for (const value of new Set(Object.values(crossValues))) value.dispose() }
    const text = phase('text_decode', () => this.model.decode(ids))
    const inference_ms = Date.now() - started
    return { text, tokens: ids, encoder_ms, inference_ms, decoder_ms: inference_ms - encoder_ms, truncated: ids.at(-1) !== config.eosTokenId }
  }

  dispose() { this.decoder.dispose(); this.cross.dispose(); this.encoder.dispose() }
}

export interface CausalProgramDefinition extends ProgramDefinition {
  config: { vocab_size: number; n_positions?: number; max_position_embeddings?: number; eos_token_id?: number }
  forward(length: number, outputStart?: number): Program
}

export class CausalProgramRuntime {
  readonly session: Session
  readonly state: ExecutionState

  constructor(readonly model: CausalProgramDefinition, device: Device) {
    this.session = new Session({ device })
    try { this.state = this.session.initialize(model.forward(1), { parameters: model.parameters }) }
    catch (error) { this.session.dispose(); throw error }
  }

  forward(ids: readonly number[], outputStart = 0) {
    if (!ids.length || ids.some(id => !Number.isInteger(id) || id < 0 || id >= this.model.config.vocab_size)) throw Error('Expected nonempty valid token IDs')
    const source = this.model.forward(ids.length, outputStart)
    const inputs: Record<string, Tensor> = {
      ids: this.session.tensor(Array.from(ids), { dtype: 'i64' }),
      mask: this.session.tensor([[Array.from({ length: ids.length }, (_, row) => Array.from({ length: ids.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' }),
    }
    if (source.inspect().arguments.some(argument => argument.name === 'positions')) {
      inputs.positions = this.session.tensor(Array.from({ length: ids.length }, (_, index) => index), { dtype: 'i64' })
    }
    try {
      const result = this.session.compile(source).run(inputs, this.state) as Tensor | Tensor[]
      const [logits, ...hiddenStates] = Array.isArray(result) ? result : [result]
      return { logits, hidden_states: hiddenStates }
    } finally {
      for (const input of Object.values(inputs)) input.dispose()
    }
  }

  generate(ids: readonly number[], budget: number) {
    const context = this.model.config.n_positions ?? this.model.config.max_position_embeddings ?? 0
    if (!Number.isInteger(budget) || budget < 0 || ids.length + budget > context) throw Error('Generation must fit the model context')
    const output = Array.from(ids)
    for (let index = 0; index < budget; index++) {
      const result = this.forward(output, output.length - 1)
      try {
        const row = (result.logits.to_array() as number[][][])[0][0]
        let best = 0
        for (let candidate = 1; candidate < row.length; candidate++) if (row[candidate] > row[best]) best = candidate
        output.push(best)
        if (best === this.model.config.eos_token_id) break
      } finally {
        result.logits.dispose()
        for (const hidden of result.hidden_states) hidden.dispose()
      }
    }
    return output
  }

  dispose() { this.state.dispose(); this.session.dispose() }
}

export function execute_vit(
  model: ProgramDefinition & { forward: Program },
  pixels: ModelTensor,
  device: Device,
) {
  const session = new Session({ device })
  let state: ExecutionState
  try { state = session.initialize(model.forward, { parameters: model.parameters }) }
  catch (error) { session.dispose(); throw error }
  const input = session.tensor(pixels.to_array(), { dtype: 'f32' })
  try {
    const [output, ...hidden_states] = session.compile(model.forward).run({ pixels: input }, state) as Tensor[]
    return { output, hidden_states, dispose: () => { output.dispose(); for (const value of hidden_states) value.dispose(); state.dispose(); session.dispose() } }
  } catch (error) {
    state.dispose(); session.dispose(); throw error
  } finally { input.dispose() }
}

export function execute_bert(
  model: ProgramDefinition & { forward(batch: number, length: number): Program },
  ids: number[][],
  masks: number[][],
  types: number[][],
  device: Device,
) {
  const batch = ids.length, length = ids[0]?.length ?? 0
  if (!batch || !length || masks.length !== batch || types.length !== batch || masks.some(row => !row.some(Boolean))) throw Error('Invalid BERT batch')
  const source = model.forward(batch, length)
  const session = new Session({ device })
  let state: ExecutionState
  try { state = session.initialize(source, { parameters: model.parameters }) }
  catch (error) { session.dispose(); throw error }
  const inputs = {
    ids: session.tensor(ids, { dtype: 'i64' }),
    types: session.tensor(types, { dtype: 'i64' }),
    positions: session.tensor(Array.from({ length: batch }, () => Array.from({ length }, (_, index) => index)), { dtype: 'i64' }),
    valid: session.tensor(masks.map(row => row.map(value => [value]))),
    attention_mask: session.tensor(masks.map(row => [[[...row.map(value => 1 - value)]]]), { dtype: 'i64' }),
  }
  try {
    const [pooled, pooler, ...hidden_states] = session.compile(source).run(inputs, state) as Tensor[]
    return { output: hidden_states.at(-1)!, pooled, pooler, hidden_states, dispose: () => { pooled.dispose(); pooler.dispose(); for (const value of hidden_states) value.dispose(); state.dispose(); session.dispose() } }
  } catch (error) {
    state.dispose(); session.dispose(); throw error
  } finally { for (const input of Object.values(inputs)) input.dispose() }
}
