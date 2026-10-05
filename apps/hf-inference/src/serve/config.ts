import { getEnv } from 'std:process'
import fs from 'std:fs'
import { SMOLLM2_MODELS, type SmolLM2Size, type InferenceOptions } from '../inference/models.ts'

export interface PlaygroundConfig extends InferenceOptions {
  port: number
  origin: string
}

/** Runtime settings shared by the HTTP server and model loaders. */
export function load_config(): PlaygroundConfig {
  const device = getEnv('AFFON_DEVICE') ?? 'cpu'
  if (!['cpu', 'metal', 'cuda'].includes(device))
    throw Error('Playground supports cpu, metal or cuda')
  const port = Number(getEnv('AFFON_SERVE_PORT') ?? 8765)
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw Error('Invalid AFFON_SERVE_PORT')
  const working_dir = getEnv('PWD')
  if (!working_dir?.startsWith('/'))
    throw Error('Run the playground from the repository root or set AFFON_HF_CACHE')
  const artifacts = 'apps/hf-inference/artifacts'
  const cache_dir = getEnv('AFFON_HF_CACHE') ?? `${working_dir}/${artifacts}/hf-cache`
  const prepared = (name: string, directory: string, marker = 'graph.json') =>
    getEnv(name) ?? (fs.existsSync(`${directory}/${marker}`) ? directory : undefined)
  const whisper_directory = `${artifacts}/whisper`
  const whisper_dir = getEnv('AFFON_WHISPER_DIR') ??
    (['source/config.json', 'source/tokenizer.json', 'whisper.json',
      'encoder/graph.json', 'cross/graph.json', 'step/graph.json',
      'embeddings.safetensors', 'positions.safetensors']
      .every((file) => fs.existsSync(`${whisper_directory}/${file}`))
      ? whisper_directory : undefined)
  const smol = getEnv('AFFON_SMOLLM2')
  const size = smol === '1' ? '135M' : smol
  if (size && size !== 'all' && !Object.hasOwn(SMOLLM2_MODELS, size)) throw Error('AFFON_SMOLLM2 must be 1, 135M, 360M, 1.7B or all')
  return {
    text_only: getEnv('AFFON_TEXT_ONLY') === '1',
    smollm2: size as SmolLM2Size | 'all' | undefined,
    vit_onnx_dir: prepared('AFFON_VIT_ONNX_DIR', `${artifacts}/vit-onnx`),
    mobilenet_dir: prepared('AFFON_MOBILENET_DIR', `${artifacts}/mobilenet/source`, 'config.json'),
    mobilenet_onnx_dir: prepared('AFFON_MOBILENET_ONNX_DIR', `${artifacts}/mobilenet`),
    ast_dir: prepared('AFFON_AST_DIR', `${artifacts}/ast/source`, 'config.json'),
    ast_onnx_dir: prepared('AFFON_AST_ONNX_DIR', `${artifacts}/ast`),
    whisper_dir,
    device: device as InferenceOptions['device'],
    port,
    origin: `http://127.0.0.1:${port}`,
    cache_dir,
    vision_cache_dir: getEnv('AFFON_HF_VISION_CACHE') ?? cache_dir,
    local_files_only: getEnv('AFFON_HF_OFFLINE') === '1',
  }
}
