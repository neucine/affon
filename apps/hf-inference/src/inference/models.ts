import {
  from_pretrained,
  load_model,
  load_processor,
} from '../../../../packages/@affon/huggingface/src/index.ts'
/** Model loading options, independent of HTTP and environment variables. */
export interface InferenceOptions {
  device: 'cpu' | 'metal'
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
export const IMAGE_MODEL = {
  id: 'google/vit-base-patch16-224',
  revision: '3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3',
} as const

/** Load pinned native checkpoints and configured local graphs once for reuse. */
export async function load_models(config: InferenceOptions) {
  const { model, processor } = await from_pretrained(TEXT_MODEL.id, {
    revision: TEXT_MODEL.revision,
    cache_dir: config.cache_dir,
    task: 'text-generation',
    device: config.device,
    local_files_only: config.local_files_only,
  })
  const vision = await from_pretrained(IMAGE_MODEL.id, {
    revision: IMAGE_MODEL.revision,
    cache_dir: config.vision_cache_dir,
    task: 'image-classification',
    device: config.device,
    local_files_only: config.local_files_only,
  })
  const images: Record<
    string,
    {
      id: string
      label: string
      backend: 'native' | 'onnx'
      model: {
        config: { id2label?: Record<string, string> }
        forward: (pixels: ReturnType<typeof vision.processor.process>) => {
          output: ReturnType<typeof vision.model.forward>['output']
        }
      }
      processor: typeof vision.processor
    }
  > = {
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
  return { text: { model, processor }, vision, images, audio, speech }
}
export type InferenceModels = Awaited<ReturnType<typeof load_models>>
