import { getEnv } from 'std:process'
import type { InferenceOptions } from '../inference/models.ts'

export interface PlaygroundConfig extends InferenceOptions {
  port: number
  origin: string
}

/** Runtime settings shared by the HTTP server and model loaders. */
export function load_config(): PlaygroundConfig {
  const device = getEnv('AFFON_DEVICE') ?? 'cpu'
  if (device !== 'cpu' && device !== 'metal')
    throw Error('Playground supports cpu or metal')
  const port = Number(getEnv('AFFON_SERVE_PORT') ?? 8765)
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw Error('Invalid AFFON_SERVE_PORT')
  const cache_dir = getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache'
  return {
    vit_onnx_dir: getEnv('AFFON_VIT_ONNX_DIR') ?? undefined,
    mobilenet_dir: getEnv('AFFON_MOBILENET_DIR') ?? undefined,
    mobilenet_onnx_dir: getEnv('AFFON_MOBILENET_ONNX_DIR') ?? undefined,
    ast_dir: getEnv('AFFON_AST_DIR') ?? undefined,
    ast_onnx_dir: getEnv('AFFON_AST_ONNX_DIR') ?? undefined,
    whisper_dir: getEnv('AFFON_WHISPER_DIR') ?? undefined,
    device,
    port,
    origin: `http://127.0.0.1:${port}`,
    cache_dir,
    vision_cache_dir: getEnv('AFFON_HF_VISION_CACHE') ?? cache_dir,
    local_files_only: getEnv('AFFON_HF_OFFLINE') === '1',
  }
}
