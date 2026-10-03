import { Session } from 'affon:compute'
import type { Device, Tensor, TensorData } from 'affon:compute'
import { parameter_checks, positive_dimensions } from '../shared/parameters.ts'
import type { ModelTensor } from '../shared/parameters.ts'

export interface WhisperConfig {
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

/** Prepared stages exchange session tensors and keep model graph ownership outside
 * the greedy orchestration layer. */
export interface WhisperStages {
  encode(features: Tensor): Tensor
  cross(encoded: Tensor): Record<string, Tensor>
  step(inputs: Record<string, Tensor>): Record<string, Tensor>
}
export interface WhisperParameters {
  embeddings: ModelTensor
  positions: ModelTensor
}

/** Greedy Whisper execution with request-local caches and caller-provided text decoding. */
export function create_whisper(
  config: WhisperConfig,
  stages: WhisperStages,
  parameters: WhisperParameters,
  decode: (ids: number[]) => string,
  device: Device = 'cpu',
) {
  positive_dimensions(config.width, config.heads, config.layers, config.vocabSize, config.contextLength, config.cacheCapacity)
  if (config.width % config.heads || config.cacheCapacity > config.contextLength || config.prefix.length < 1 || config.prefix.length >= config.cacheCapacity) throw Error('Invalid Whisper dimensions or prefix')
  const validId = (id: number) => Number.isInteger(id) && id >= 0 && id < config.vocabSize
  if (![...config.prefix, config.eosTokenId, ...config.suppressTokens, ...config.beginSuppressTokens].every(validId)) throw Error('Invalid Whisper generation tokens')
  const { embeddings, positions } = parameters
  const check = parameter_checks(device)
  check.tensor(embeddings as any, [config.vocabSize, config.width])
  check.tensor(positions as any, [config.contextLength, config.width])
  const embeddingValues = embeddings.to_array() as number[][]
  const positionValues = positions.to_array() as number[][]
  const session = new Session({ device })
  const headWidth = config.width / config.heads

  function transcribe(
    features: Tensor,
    on_step?: (step: number, logits: number[]) => void,
    on_phase?: (phase: string, elapsed_ms: number) => void,
  ) {
    function phase<T>(name: string, run: () => T): T {
      if (!on_phase) return run()
      const start = Date.now()
      const result = run()
      on_phase(name, Date.now() - start)
      return result
    }
    const started = Date.now()
    const encoded = phase('encoder', () => stages.encode(features))
    const encoder_ms = Date.now() - started
    const ids: number[] = [...config.prefix]
    const begin = new Set<number>(config.beginSuppressTokens)
    const suppress = new Set<number>(config.suppressTokens)
    const crossValues = phase('cross', () => stages.cross(encoded))
    const pastValues = Array.from({ length: config.layers * 2 }, () => [
      Array.from({ length: config.heads }, () => Array.from({ length: config.cacheCapacity }, () => Array(headWidth).fill(0))),
    ]) as number[][][][][]
    let position = 0
    while (position < config.cacheCapacity - 1) {
      const token = ids[position]
      const owned: Tensor[] = []
      const make = (value: TensorData, dtype: 'f32' | 'i64' = 'f32') => {
        const tensor = session.tensor(value, { dtype }) as Tensor
        owned.push(tensor)
        return tensor
      }
      const inputs: Record<string, Tensor> = phase('inputs', () => ({
        ...crossValues,
        embeddings: make([[embeddingValues[token]]]),
        position: make([[positionValues[position]]]),
        mask: make([[[Array.from({ length: config.cacheCapacity + 1 }, (_, index) => index < position || index === config.cacheCapacity ? 0 : -10000)]]]),
      }))
      pastValues.forEach((value, index) => { inputs[`past_${index}`] = make(value) })
      let output: Record<string, Tensor>
      try {
        output = phase('decoder_graph', () => stages.step(inputs))
      } finally {
        for (const value of owned) value.dispose()
      }
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
      for (const value of new Set(Object.values(output))) value.dispose()
      position++
      if (position < config.prefix.length) continue
      const step = position - config.prefix.length
      on_step?.(step, logits)
      const best = phase('selection', () => {
        let best = -1, score = -Infinity
        for (let index = 0; index < logits.length; index++) if (logits[index] > score && !suppress.has(index) && !(step === 0 && begin.has(index))) { score = logits[index]; best = index }
        if (best < 0) throw Error('Whisper produced no finite candidate')
        return best
      })
      ids.push(best)
      if (best === config.eosTokenId) break
    }
    const text = phase('text_decode', () => decode(ids))
    const inference_ms = Date.now() - started
    return { text, tokens: ids, encoder_ms, inference_ms, decoder_ms: inference_ms - encoder_ms, truncated: ids[ids.length - 1] !== config.eosTokenId }
  }
  function dispose() { session.dispose() }
  return { config, max_new_tokens: config.cacheCapacity - config.prefix.length, transcribe, dispose }
}
