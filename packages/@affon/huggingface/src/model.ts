import { load_llama } from './adapters/llama.ts'
import { load_whisper, type WhisperOptions } from './adapters/whisper.ts'
import fs from 'std:fs'
import type { Device } from 'affon:compute'
import { load_gpt2 } from './adapters/gpt2.ts'
import { load_bert } from './adapters/bert.ts'
import { load_vit } from './adapters/vit.ts'
import { load_onnx_classifier } from './onnx.ts'
import type { OnnxModelOptions, OnnxClassifier } from './onnx.ts'

/** Native model contracts by supported task; inputs remain domain-specific. */
export type ModelsByTask = {
  'text-generation': ReturnType<typeof load_gpt2> | ReturnType<typeof load_llama>
  'feature-extraction': ReturnType<typeof load_bert>
  'image-classification': ReturnType<typeof load_vit>
}
export type NativeModelTask = keyof ModelsByTask
export type ModelTask =
  | NativeModelTask
  | 'audio-classification'
  | 'automatic-speech-recognition'
export type ModelOptions<T extends NativeModelTask> = {
  task: T
  device?: Device
  backend?: 'native'
}

/**
 * Load a supported HF architecture and task from a prepared local directory.
 * @param directory Directory containing config.json and one f32 or BF16 model.safetensors (BF16 widens to f32).
 * @param options Explicit task and execution device (CPU by default).
 * @returns Native model with a task-specific forward signature.
 * @throws If the task/model-type combination or model configuration is unsupported.
 * @example
 * const model = load_model('/models/bert', { task: 'feature-extraction' })
 * @remarks Experimental. Does not download, convert weights, or execute Python.
 */
export function load_model(
  directory: string,
  options: WhisperOptions,
): ReturnType<typeof load_whisper>
export function load_model(
  directory: string,
  options: OnnxModelOptions,
): OnnxClassifier
export function load_model<T extends NativeModelTask>(
  directory: string,
  options: ModelOptions<T>,
): ModelsByTask[T]
export function load_model(
  directory: string,
  options: ModelOptions<NativeModelTask> | OnnxModelOptions | WhisperOptions,
):
  | ModelsByTask[NativeModelTask]
  | OnnxClassifier
  | ReturnType<typeof load_whisper> {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  const device = options.device ?? 'cpu'
  if (!/^(cpu|metal|cuda(:\d+)?)$/.test(device))
    throw new Error(`Invalid HF device: ${device}`)
  if (options.task === 'automatic-speech-recognition')
    return load_whisper(directory, options)
  if (options.backend === 'onnx') return load_onnx_classifier(config, options)
  if (options.backend !== undefined && options.backend !== 'native')
    throw Error('Unsupported HF execution backend')
  if (options.task === 'text-generation' && config.model_type === 'gpt2')
    return load_gpt2(directory, device)
  if (options.task === 'text-generation' && config.model_type === 'llama')
    return load_llama(directory, device)
  if (options.task === 'feature-extraction' && config.model_type === 'bert')
    return load_bert(directory, device)
  if (options.task === 'image-classification' && config.model_type === 'vit')
    return load_vit(directory, device)
  throw new Error(
    `Unsupported HF model/task: ${config.model_type}/${options.task}`,
  )
}
