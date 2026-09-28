import {
  from_pretrained,
  type ModelsByTask, type ProcessorsByTask,
  load_model,
  load_processor,
} from '../../../../packages/@affon/huggingface/src/index.ts'
/** Model loading options, independent of HTTP and environment variables. */
export interface InferenceOptions {
  smollm2?: boolean | SmolLM2Size
  text_only?: boolean
  device: 'cpu' | 'metal' | 'cuda'
  cache_dir: string
  vision_cache_dir: string
  vit_onnx_dir?: string
  mobilenet_dir?: string
  mobilenet_onnx_dir?: string
  whisper_dir?: string
  ast_dir?: string
  ast_onnx_dir?: string
  local_files_only: boolean
}

export const TEXT_MODEL = {
  id: 'distilbert/distilgpt2',
  revision: '2290a62682d06624634c1f46a6ad5be0f47f38aa',
} as const
export const SMOLLM2_MODELS = {
  '135M': { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', revision: '12fd25f77366fa6b3b4b768ec3050bf629380bac' },
  '360M': { id: 'HuggingFaceTB/SmolLM2-360M-Instruct', revision: 'a10cc1512eabd3dde888204e902eca88bddb4951' },
  '1.7B': { id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', revision: '31b70e2e869a7173562077fd711b654946d38674' },
} as const
export type SmolLM2Size = keyof typeof SMOLLM2_MODELS
export const SMOLLM2_MODEL = SMOLLM2_MODELS['135M']
type ImageEntry = {
  id: string; label: string; backend: 'native' | 'onnx'
  model: Pick<ModelsByTask['image-classification'], 'config' | 'forward'> | {
    config: {id2label?: Record<string,string>}
    forward: (pixels: ReturnType<ProcessorsByTask['image-classification']['process']>) => {output: ReturnType<ModelsByTask['image-classification']['forward']>['output']}
  }
  processor: ProcessorsByTask['image-classification']
}
export const IMAGE_MODEL = {
  id: 'google/vit-base-patch16-224',
  revision: '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3',
} as const

/** Load pinned native checkpoints and configured local graphs once for reuse. */
export async function load_models(config: InferenceOptions) {
  const size = config.smollm2 === true ? '135M' : config.smollm2 || undefined
  if (size && !Object.hasOwn(SMOLLM2_MODELS, size)) throw Error('Unknown SmolLM2 size')
  if (config.text_only && !size) throw Error('Text-only mode requires a SmolLM2 size')
  const smolSpec = size ? SMOLLM2_MODELS[size] : undefined
  const primarySpec = config.text_only ? smolSpec! : TEXT_MODEL
  const default_text_model = config.text_only ? 'smollm2' : 'distilgpt2'
  const { model, processor } = await from_pretrained(primarySpec.id, {
    revision: primarySpec.revision, cache_dir: config.cache_dir,
    task: 'text-generation', device: config.device, local_files_only: config.local_files_only,
  })
  const texts: Record<string, { id: string; label: string; chat: boolean; model: typeof model; processor: typeof processor }> = {
    [default_text_model]: { id: primarySpec.id, label: config.text_only ? `SmolLM2 ${size} Instruct` : 'DistilGPT-2', chat: Boolean(config.text_only), model, processor },
  }
  if (smolSpec && !config.text_only) {
    const smollm2 = await from_pretrained(smolSpec.id, {
      revision: smolSpec.revision, cache_dir: config.cache_dir,
      task: 'text-generation', device: config.device, local_files_only: config.local_files_only,
    })
    texts['smollm2'] = { id: smolSpec.id, label: `SmolLM2 ${size} Instruct`, chat: true, model: smollm2.model, processor: smollm2.processor }
  }
  if (config.text_only) return {text: {model, processor}, texts, default_text_model, images: {} as Record<string, ImageEntry>, vision: undefined, audio: undefined, speech: undefined}
  const vision = await from_pretrained(IMAGE_MODEL.id, {
    revision: IMAGE_MODEL.revision,
    cache_dir: config.vision_cache_dir,
    task: 'image-classification',
    device: config.device,
    local_files_only: config.local_files_only,
  })
  const images: Record<string, ImageEntry> = {
    'vit-native': {
      id: IMAGE_MODEL.id,
      label: 'ViT native',
      backend: 'native',
      model: vision.model,
      processor: vision.processor,
    },
  }
  if (config.vit_onnx_dir)
    images['vit-onnx'] = {
      id: IMAGE_MODEL.id,
      label: 'ViT ONNX',
      backend: 'onnx',
      processor: vision.processor,
      model: load_model(vision.directory, {
        task: 'image-classification',
        backend: 'onnx',
        graph_dir: config.vit_onnx_dir,
        device: config.device,
      }),
    }
  if (Boolean(config.mobilenet_dir) !== Boolean(config.mobilenet_onnx_dir))
    throw Error(
      'Configure both AFFON_MOBILENET_DIR and AFFON_MOBILENET_ONNX_DIR',
    )
  if (config.mobilenet_dir && config.mobilenet_onnx_dir)
    images['mobilenet-onnx'] = {
      id: 'google/mobilenet_v2_1.0_224',
      label: 'MobileNet ONNX',
      backend: 'onnx',
      processor: load_processor(config.mobilenet_dir, {
        task: 'image-classification',
        device: config.device,
      }),
      model: load_model(config.mobilenet_dir, {
        task: 'image-classification',
        backend: 'onnx',
        graph_dir: config.mobilenet_onnx_dir,
        device: config.device,
      }),
    }
  if (Boolean(config.ast_dir) !== Boolean(config.ast_onnx_dir))
    throw Error('Configure both AFFON_AST_DIR and AFFON_AST_ONNX_DIR')
  const audio =
    config.ast_dir && config.ast_onnx_dir
      ? {
          id: 'MIT/ast-finetuned-speech-commands-v2',
          processor: load_processor(config.ast_dir, {
            task: 'audio-classification',
            device: config.device,
          }),
          model: load_model(config.ast_dir, {
            task: 'audio-classification',
            backend: 'onnx',
            graph_dir: config.ast_onnx_dir,
            device: config.device,
          }),
        }
      : undefined
  if (
    audio &&
    (audio.model.input_shape[1] * 160 + 240 !== audio.processor.max_samples ||
      audio.model.input_shape[2] !== audio.processor.num_mel_bins)
  )
    throw Error('AST processor dimensions do not match model input')
  const speech = config.whisper_dir
    ? {
        id: 'openai/whisper-tiny.en',
        model: load_model(`${config.whisper_dir}/source`, {
          task: 'automatic-speech-recognition',
          backend: 'onnx',
          graph_dir: config.whisper_dir,
          device: config.device,
        }),
        processor: load_processor(`${config.whisper_dir}/source`, {
          task: 'automatic-speech-recognition',
          device: config.device,
        }),
      }
    : undefined
  return { text: { model, processor }, texts, default_text_model, vision, images, audio, speech }
}
export type InferenceModels = Awaited<ReturnType<typeof load_models>>
