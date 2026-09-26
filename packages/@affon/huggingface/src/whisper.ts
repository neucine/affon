import fs from 'std:fs'
import telemetry from 'std:telemetry'
import checkpoint from 'affon:checkpoint'
import {
  tensor,
  where,
  reshape,
  contiguous,
  type Tensor,
  type Device,
} from 'affon:compute'
import { load_graph } from '../../onnx/src/index.ts'
import { createHFTokenizerFromFile } from '../../tokenizers/src/index.ts'
export type WhisperOptions = {
  task: 'automatic-speech-recognition'
  backend: 'onnx'
  graph_dir: string
  device?: Device
}
/** Bounded English Whisper greedy decoding; the encoder output is reused across steps. */
export function load_whisper(directory: string, options: WhisperOptions) {
  if (options.backend !== 'onnx' || !options.graph_dir)
    throw Error('Whisper requires prepared ONNX graphs')
  const device = options.device ?? 'cpu',
    root = options.graph_dir
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  const generation = JSON.parse(fs.readFileSync(`${root}/whisper.json`))
  if (
    config.model_type !== 'whisper' ||
    config.d_model !== 384 ||
    config.decoder_layers !== 4 ||
    config.decoder_attention_heads !== 6 ||
    config.num_mel_bins !== 80 ||
    !Number.isInteger(generation.width) ||
    generation.width < 3 ||
    generation.width > config.max_target_positions ||
    generation.model !== 'openai/whisper-tiny.en'
  )
    throw Error('Unsupported prepared Whisper configuration')
  const validId = (id: unknown) =>
    typeof id === 'number' &&
    Number.isInteger(id) &&
    id >= 0 &&
    id < config.vocab_size
  if (
    !Array.isArray(generation.prefix) ||
    generation.prefix.length !== 2 ||
    !generation.prefix.every(validId) ||
    !validId(generation.eos) ||
    !validId(generation.pad) ||
    ![generation.suppress_tokens, generation.begin_suppress_tokens].every(
      (ids) => Array.isArray(ids) && ids.every(validId),
    )
  )
    throw Error('Invalid Whisper generation configuration')
  const encoder = load_graph(`${root}/encoder`, device),
    cross = load_graph(`${root}/cross`, device),
    decoder = load_graph(`${root}/step`, device)
  if (
    decoder.graph.inputs.past_0?.[2] !== generation.width ||
    decoder.graph.inputs.mask?.[3] !== generation.width + 1
  )
    throw Error(
      'Whisper cache capacity does not match prepared graph; re-export and convert the step graph',
    )
  const loaded = checkpoint.load(`${root}/embeddings.safetensors`) as Record<
    string,
    Tensor
  >
  const embeddings = loaded.embeddings.to(device)
  const positions = (
    checkpoint.load(`${root}/positions.safetensors`) as Record<string, Tensor>
  ).positions.to(device)
  if (
    JSON.stringify(embeddings.shape) !==
      JSON.stringify([config.vocab_size, 384]) ||
    JSON.stringify(positions.shape) !==
      JSON.stringify([config.max_target_positions, 384])
  )
    throw Error('Invalid Whisper embedding shapes')
  const tokenizer = createHFTokenizerFromFile(`${directory}/tokenizer.json`)
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
          () => encoder.forward({ features }).output,
        ),
      )
    const encoder_ms = Date.now() - started,
      ids: number[] = [...generation.prefix]
    const begin = new Set<number>(generation.begin_suppress_tokens),
      suppress = new Set<number>(generation.suppress_tokens)
    const cross_values = phase('cross', () =>
      telemetry.trace('hf.whisper.cross', () => cross.forward({ encoded })),
    )
    let past: Tensor[] = phase('cache_init', () =>
      Array.from({ length: 8 }, () =>
        tensor(
          [
            Array.from({ length: 6 }, () =>
              Array.from({ length: generation.width }, () => Array(64).fill(0)),
            ),
          ],
          { dtype: 'f32', device },
        ),
      ),
    )
    let position = 0
    while (position < generation.width - 1) {
      const token = ids[position]
      const inputs: Record<string, Tensor> = phase('inputs', () => ({
        ...cross_values,
        embeddings: reshape(
          contiguous(embeddings.slice([token, ':'])),
          [1, 1, 384],
        ),
        position: reshape(
          contiguous(positions.slice([position, ':'])),
          [1, 1, 384],
        ),
        mask: tensor(
          [
            [
              [
                Array.from({ length: generation.width + 1 }, (_, i) =>
                  i < position || i === generation.width ? 0 : -10000,
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
        telemetry.trace('hf.whisper.decoder', () => decoder.forward(inputs)),
      )
      phase('cache_update', () => {
        const write = tensor(
          [
            [
              Array.from({ length: generation.width }, (_, i) => [
                i === position ? 1 : 0,
              ]),
            ],
          ],
          { dtype: 'i64', device },
        )
        // Write only the current cache slot; masked future slots remain zero.
        const writeMask = reshape(write, [1, 1, generation.width, 1])
        past = past.map((value, i) =>
          where(writeMask, output[`present_${i}`], value),
        )
      })
      const logits = phase(
        'readback',
        () =>
          reshape(output.logits, [config.vocab_size]).to_array() as number[],
      )
      position++
      if (position < generation.prefix.length) continue
      const step = position - generation.prefix.length
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
      if (best === generation.eos) break
    }
    const text = phase('text_decode', () =>
      tokenizer.decode(ids, { skipSpecialTokens: true }),
    )
    const inference_ms = Date.now() - started
    return {
      text,
      tokens: ids,
      encoder_ms,
      inference_ms,
      // Includes cross-attention setup, cache updates, token selection and decoding.
      decoder_ms: inference_ms - encoder_ms,
      truncated: ids[ids.length - 1] !== generation.eos,
    }
  }
  return {
    config,
    backend: 'onnx' as const,
    max_new_tokens: generation.width - generation.prefix.length,
    transcribe,
  }
}
