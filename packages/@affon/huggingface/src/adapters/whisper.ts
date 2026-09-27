import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import type { Tensor, Device } from 'affon:compute'
import { load_graph } from '../../../onnx/src/index.ts'
import { createHFTokenizerFromFile } from '../../../tokenizers/src/index.ts'
import { create_whisper } from '../../../models/src/whisper/index.ts'
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
  const model = create_whisper({
    width: config.d_model, heads: config.decoder_attention_heads,
    layers: config.decoder_layers, vocabSize: config.vocab_size,
    contextLength: config.max_target_positions, cacheCapacity: generation.width,
    prefix: generation.prefix, eosTokenId: generation.eos,
    suppressTokens: generation.suppress_tokens, beginSuppressTokens: generation.begin_suppress_tokens,
  }, {
    encode: (features) => encoder.forward({ features }).output,
    cross: (encoded) => cross.forward({ encoded }),
    step: (inputs) => decoder.forward(inputs),
  }, { embeddings, positions }, (ids) => tokenizer.decode(ids, { skipSpecialTokens: true }), device)
  return { ...model, config, backend: 'onnx' as const }
}
