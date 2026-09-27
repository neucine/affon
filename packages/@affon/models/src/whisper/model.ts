import telemetry from 'std:telemetry'
import {
  tensor,
  where,
  reshape,
  contiguous,
  type Tensor,
  type Device,
} from 'affon:compute'
import { parameter_checks, positive_dimensions } from '../shared/parameters.ts'

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

/** Prepared stages use batch-one features and fixed-capacity self-attention caches.
 * step consumes embeddings, position, mask, past_N, and cross-stage outputs;
 * it returns logits and present_N tensors for each layer's key/value pair.
 */
export interface WhisperStages {
  encode(features: Tensor): Tensor
  cross(encoded: Tensor): Record<string, Tensor>
  step(inputs: Record<string, Tensor>): Record<string, Tensor>
}
export interface WhisperParameters {
  embeddings: Tensor
  positions: Tensor
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
  check.tensor(embeddings, [config.vocabSize, config.width])
  check.tensor(positions, [config.contextLength, config.width])
  function transcribe(
    features: Tensor,
    on_step?: (step: number, logits: number[]) => void,
    on_phase?: (phase: string, elapsed_ms: number) => void,
  ) {
    // Optional diagnostic observer; ordinary inference avoids phase snapshots.
    function phase<T>(name: string, run: () => T): T {
      if (!on_phase) return run()
      const start = Date.now()
      const result = run()
      on_phase(name, Date.now() - start)
      return result
    }
    const started = Date.now(),
      encoded = phase('encoder', () =>
        telemetry.trace(
          'hf.whisper.encoder',
          () => stages.encode(features),
        ),
      )
    const encoder_ms = Date.now() - started,
      ids: number[] = [...config.prefix]
    const begin = new Set<number>(config.beginSuppressTokens),
      suppress = new Set<number>(config.suppressTokens)
    const cross_values = phase('cross', () =>
      telemetry.trace('hf.whisper.cross', () => stages.cross(encoded)),
    )
    let past: Tensor[] = phase('cache_init', () =>
      Array.from({ length: config.layers * 2 }, () =>
        tensor(
          [
            Array.from({ length: config.heads }, () =>
              Array.from({ length: config.cacheCapacity }, () => Array(config.width / config.heads).fill(0)),
            ),
          ],
          { dtype: 'f32', device },
        ),
      ),
    )
    let position = 0
    while (position < config.cacheCapacity - 1) {
      const token = ids[position]
      const inputs: Record<string, Tensor> = phase('inputs', () => ({
        ...cross_values,
        embeddings: reshape(
          contiguous(embeddings.slice([token, ':'])),
          [1, 1, config.width],
        ),
        position: reshape(
          contiguous(positions.slice([position, ':'])),
          [1, 1, config.width],
        ),
        mask: tensor(
          [
            [
              [
                Array.from({ length: config.cacheCapacity + 1 }, (_, i) =>
                  i < position || i === config.cacheCapacity ? 0 : -10000,
                ),
              ],
            ],
          ],
          { dtype: 'f32', device },
        ),
      }))
      past.forEach((value, i) => {
        inputs[`past_${i}`] = value
      })
      const output = phase('decoder_graph', () =>
        telemetry.trace('hf.whisper.decoder', () => stages.step(inputs)),
      )
      phase('cache_update', () => {
        const write = tensor(
          [
            [
              Array.from({ length: config.cacheCapacity }, (_, i) => [
                i === position ? 1 : 0,
              ]),
            ],
          ],
          { dtype: 'i64', device },
        )
        // Write only the current cache slot; masked future slots remain zero.
        const writeMask = reshape(write, [1, 1, config.cacheCapacity, 1])
        past = past.map((value, i) =>
          where(writeMask, output[`present_${i}`], value),
        )
      })
      const logits = phase(
        'readback',
        () =>
          reshape(output.logits, [config.vocabSize]).to_array() as number[],
      )
      position++
      if (position < config.prefix.length) continue
      const step = position - config.prefix.length
      on_step?.(step, logits)
      const best = phase('selection', () => {
        let best = -1,
          score = -Infinity
        // Only candidates that improve the score need suppression lookups.
        for (let i = 0; i < logits.length; i++)
          if (
            logits[i] > score &&
            !suppress.has(i) &&
            !(step === 0 && begin.has(i))
          ) {
            score = logits[i]
            best = i
          }
        if (best < 0) throw Error('Whisper produced no finite candidate')
        return best
      })
      ids.push(best)
      if (best === config.eosTokenId) break
    }
    const text = phase('text_decode', () =>
      decode(ids),
    )
    const inference_ms = Date.now() - started
    return {
      text,
      tokens: ids,
      encoder_ms,
      inference_ms,
      // Includes cross-attention setup, cache updates, token selection and decoding.
      decoder_ms: inference_ms - encoder_ms,
      truncated: ids[ids.length - 1] !== config.eosTokenId,
    }
  }
  return {
    config,
    max_new_tokens: config.cacheCapacity - config.prefix.length,
    transcribe,
  }
}
