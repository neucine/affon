import { program, Session, Tensor, type Device, type Executable, type ExecutionState, type Program, type TensorData } from 'affon:compute'
import { where } from 'affon:ops'
import type { ModelTensor } from '../../../../packages/@affon/models/src/shared/parameters.ts'

export interface ProgramDefinition {
  parameters: Readonly<Record<string, ModelTensor>>
}

export interface GraphProgramDefinition extends ProgramDefinition {
  forward: Program
  output_names: readonly string[]
}

type ProgramInput = ModelTensor | Tensor | { data: TensorData; dtype?: 'f32' | 'i64' }

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
    const created: Tensor[] = []
    const inputs = Object.fromEntries(Object.entries(values).map(([name, value]) => {
      if (!('data' in value) && this.session.owns(value)) return [name, value]
      const tensor = 'data' in value
        ? this.session.tensor(value.data, { dtype: value.dtype ?? 'f32' })
        : this.session.tensor(value.to_array(), { dtype: value.dtype })
      created.push(tensor)
      return [name, tensor]
    })) as Record<string, Tensor>
    try {
      const result = this.executable.run(inputs, this.state) as Tensor | Tensor[]
      const outputs = Array.isArray(result) ? result : [result]
      return Object.fromEntries(this.graph.output_names.map((name, index) => [name, outputs[index]])) as Record<string, Tensor>
    } finally { for (const input of created) input.dispose() }
  }

  /** Copy a foreign-session tensor once so repeated graph calls keep it device-resident. */
  import(value: ModelTensor | Tensor) {
    if (this.session.owns(value)) return value as Tensor
    return this.session.tensor(value.to_array(), { dtype: value.dtype })
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
  readonly cacheUpdate: Executable
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
    const cacheSpec = Tensor.f32([1, config.heads, config.cacheCapacity, config.width / config.heads])
    const presentSpec = Tensor.f32([1, config.heads, 1, config.width / config.heads])
    const cacheUpdate = program('whisper_cache_update', p => {
      const mask = p.argument('mask', Tensor.i64([1, 1, config.cacheCapacity, 1]))
      return Array.from({ length: config.layers * 2 }, (_, index) => where(
        mask,
        p.argument(`present_${index}`, presentSpec),
        p.argument(`past_${index}`, cacheSpec),
      ))
    })
    try { this.cacheUpdate = this.decoder.session.compile(cacheUpdate) }
    catch (error) { this.decoder.dispose(); this.cross.dispose(); this.encoder.dispose(); throw error }
    this.max_new_tokens = config.cacheCapacity - config.prefix.length
  }

  transcribe(
    features: ModelTensor,
    on_step?: (step: number, logits: number[]) => void,
    on_phase?: (phase: string, elapsed_ms: number) => void,
    max_new_tokens = this.max_new_tokens,
  ) {
    const config = this.model.generation
    if (!Number.isInteger(max_new_tokens) || max_new_tokens < 1 || max_new_tokens > this.max_new_tokens)
      throw Error(`Whisper generation budget must be 1–${this.max_new_tokens} tokens`)
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
    let foreignCrossValues: Record<string, Tensor>
    try { foreignCrossValues = phase('cross', () => this.cross.forward({ encoded })) }
    finally { for (const value of new Set(Object.values(encodedOutput))) value.dispose() }
    const crossValues = phase('cross_copy', () => Object.fromEntries(
      Object.entries(foreignCrossValues).map(([name, value]) => [name, this.decoder.import(value)]),
    ))
    for (const value of new Set(Object.values(foreignCrossValues))) value.dispose()
    const ids = [...config.prefix]
    const begin = new Set(config.beginSuppressTokens), suppress = new Set(config.suppressTokens)
    const headWidth = config.width / config.heads
    let pastValues = Array.from({ length: config.layers * 2 }, () => this.decoder.session.tensor([
      Array.from({ length: config.heads }, () => Array.from({ length: config.cacheCapacity }, () => Array(headWidth).fill(0))),
    ]))
    let position = 0
    try {
      while (position < config.cacheCapacity - 1) {
        const inputs: Record<string, ProgramInput> = {
          ...crossValues,
          embeddings: { data: [[this.embeddingValues[ids[position]]]] },
          position: { data: [[this.positionValues[position]]] },
          mask: { data: [[[Array.from({ length: config.cacheCapacity + 1 }, (_, index) => index < position || index === config.cacheCapacity ? 0 : -10000)]]] },
        }
        pastValues.forEach((value, index) => { inputs[`past_${index}`] = value })
        const output = phase('decoder_graph', () => this.decoder.forward(inputs))
        try {
          phase('cache_update', () => {
            const updateInputs: Record<string, Tensor> = {}
            const updateMask = this.decoder.session.tensor([[
              Array.from({ length: config.cacheCapacity }, (_, index) => [index === position ? 1 : 0]),
            ]], { dtype: 'i64' })
            updateInputs.mask = updateMask
            for (let index = 0; index < pastValues.length; index++) {
              const present = output[`present_${index}`]
              if (!present) throw Error(`Whisper decoder omitted present_${index}`)
              updateInputs[`past_${index}`] = pastValues[index]
              updateInputs[`present_${index}`] = present
            }
            let next: Tensor[]
            try { next = this.cacheUpdate.run(updateInputs) as Tensor[] }
            finally { updateMask.dispose() }
            for (const value of pastValues) value.dispose()
            pastValues = next
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
          if (best === config.eosTokenId || ids.length - config.prefix.length >= max_new_tokens) break
        } finally { for (const value of new Set(Object.values(output))) value.dispose() }
      }
    } finally {
      for (const value of pastValues) value.dispose()
      for (const value of new Set(Object.values(crossValues))) value.dispose()
    }
    const text = phase('text_decode', () => this.model.decode(ids))
    const inference_ms = Date.now() - started
    return { text, tokens: ids, encoder_ms, inference_ms, decoder_ms: inference_ms - encoder_ms, truncated: ids.at(-1) !== config.eosTokenId }
  }

  dispose() { this.decoder.dispose(); this.cross.dispose(); this.encoder.dispose() }
}

export interface CausalProgramDefinition extends ProgramDefinition {
  config: { vocab_size: number; n_positions?: number; max_position_embeddings?: number; eos_token_id?: number }
  forward(length: number, outputStart?: number): Program
  prefill?(length: number): Program
  decode?(position: number): Program
}

export class ModelOutputError extends Error {}

export class NonFiniteLogitsError extends ModelOutputError {
  constructor(position: number, index: number) {
    super(`Non-finite model logit at token position ${position}, vocabulary index ${index}`)
    this.name = 'NonFiniteLogitsError'
  }
}

export function finite_argmax(row: readonly number[], expected: number, position: number) {
  if (!Array.isArray(row) || row.length !== expected) throw new ModelOutputError(`Invalid model logits at token position ${position}`)
  let best = 0
  for (let index = 0; index < row.length; index++) {
    if (!Number.isFinite(row[index])) throw new NonFiniteLogitsError(position, index)
    if (row[index] > row[best]) best = index
  }
  return best
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

  /** Stateful greedy generation using device-resident K/V tensors when supplied by the model. */
  createGeneration(ids: readonly number[], budget: number) {
    const context = this.model.config.n_positions ?? this.model.config.max_position_embeddings ?? 0
    if (!ids.length || ids.some(id => !Number.isInteger(id) || id < 0 || id >= this.model.config.vocab_size)) throw Error('Expected nonempty valid token IDs')
    if (!Number.isInteger(budget) || budget < 0 || ids.length + budget > context) throw Error('Generation must fit the model context')
    const output = Array.from(ids)
    let caches: Tensor[] = [], done = budget === 0
    const cached = Boolean(this.model.prefill && this.model.decode)
    const release = () => { for (const value of caches) value.dispose(); caches = [] }
    return {
      get done() { return done },
      close() { done = true; release() },
      next: () => {
        if (done) throw Error('Generation has finished')
        let logits: Tensor, nextCaches: Tensor[] = [], hiddenStates: Tensor[] = []
        if (cached) {
          const first = caches.length === 0
          const source = first ? this.model.prefill!(output.length) : this.model.decode!(output.length - 1)
          const inputs: Record<string, Tensor> = {
            ids: this.session.tensor(first ? output : [output.at(-1)!], { dtype: 'i64' }),
          }
          if (first) inputs.mask = this.session.tensor([[Array.from({ length: output.length }, (_, row) => Array.from({ length: output.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' })
          else for (let index = 0; index < caches.length / 2; index++) {
            inputs[`past_${index}_k`] = caches[index * 2]
            inputs[`past_${index}_v`] = caches[index * 2 + 1]
          }
          try {
            const values = this.session.compile(source).run(inputs, this.state) as Tensor[]
            ;[logits, ...nextCaches] = values
          } finally {
            inputs.ids.dispose()
            inputs.mask?.dispose()
          }
        } else {
          const result = this.forward(output, output.length - 1)
          logits = result.logits
          hiddenStates = result.hidden_states
        }
        try {
          const row = (logits.to_array() as number[][][])[0][0]
          const best = finite_argmax(row, this.model.config.vocab_size, output.length)
          output.push(best)
          if (cached) { release(); caches = nextCaches; nextCaches = [] }
          done = best === this.model.config.eos_token_id || output.length - ids.length === budget
          return { ids: Array.from(output), done }
        } finally {
          logits.dispose()
          for (const hidden of hiddenStates) hidden.dispose()
          for (const value of nextCaches) value.dispose()
          if (done) release()
        }
      },
    }
  }

  generate(ids: readonly number[], budget: number) {
    if (budget === 0) return Array.from(ids)
    const generation = this.createGeneration(ids, budget)
    let result = Array.from(ids)
    try { while (!generation.done) result = generation.next().ids }
    finally { generation.close() }
    return result
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
