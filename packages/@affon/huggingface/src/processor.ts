import { load_ast_processor } from './processors/ast.ts'
import { load_whisper_processor } from './processors/whisper.ts'
import fs from 'std:fs'
import type { Device } from 'affon:compute'
import { createHFTokenizerFromFile } from '../../tokenizers/src/index.ts'
import { load_bert_processor } from './processors/bert.ts'
import { process_mobilenet_image } from './processors/mobilenet.ts'
import { process_rgb_image } from './processors/vit.ts'
import type { ModelTask } from './model.ts'

export type ProcessorsByTask = {
  'automatic-speech-recognition': ReturnType<typeof load_whisper_processor>
  'audio-classification': ReturnType<typeof load_ast_processor>
  'text-generation': ReturnType<typeof createHFTokenizerFromFile>
  'feature-extraction': ReturnType<typeof load_bert_processor>
  'image-classification': {
    process: (rgb: number[][][]) => ReturnType<typeof process_rgb_image>
  }
}
/** Select the supported processor from local HF artifacts and the requested task.
 * @param directory Local model snapshot directory.
 * @param options Task and output tensor device (CPU by default).
 * @returns A domain-specific tokenizer, BERT batch encoder, or RGB image processor.
 */
export function load_processor<T extends ModelTask>(
  directory: string,
  options: { task: T; device?: Device },
): ProcessorsByTask[T]
export function load_processor(
  directory: string,
  options: { task: ModelTask; device?: Device },
): ProcessorsByTask[ModelTask] {
  const config = JSON.parse(fs.readFileSync(`${directory}/config.json`))
  if (config.model_type === 'gpt2' && options.task === 'text-generation')
    return createHFTokenizerFromFile(`${directory}/tokenizer.json`)
  if (config.model_type === 'bert' && options.task === 'feature-extraction')
    return load_bert_processor(directory)
  if (config.model_type === 'vit' && options.task === 'image-classification') {
    const device: Device = options.device ?? 'cpu'
    return {
      process: (rgb: number[][][]) => process_rgb_image(directory, rgb, device),
    }
  }
  if (
    config.model_type === 'mobilenet_v2' &&
    options.task === 'image-classification'
  )
    return {
      process: (rgb: number[][][]) =>
        process_mobilenet_image(directory, rgb, options.device ?? 'cpu'),
    }
  if (
    config.model_type === 'audio-spectrogram-transformer' &&
    options.task === 'audio-classification'
  )
    return load_ast_processor(directory, options.device ?? 'cpu')
  if (
    config.model_type === 'whisper' &&
    options.task === 'automatic-speech-recognition'
  )
    return load_whisper_processor(directory, options.device ?? 'cpu')
  throw new Error(
    `Unsupported HF processor/task: ${config.model_type}/${options.task}`,
  )
}
