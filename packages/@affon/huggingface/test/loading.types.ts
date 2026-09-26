import { load_model, from_pretrained, load_processor } from '../src/index.ts'
import type { Tensor } from 'affon:compute'

// Compile-only: task selection must preserve each domain's input signature.
function contracts(pixels: Tensor) {
  const text = load_model('/models/gpt2', { task: 'text-generation' })
  text.generate([1, 2], 3)
  text.generate([1, 2], 3, { use_cache: false })
  const session = text.create_session()
  session.forward([1, 2])
  session.forward([3])
  session.reset()
  const bert = load_model('/models/bert', { task: 'feature-extraction' })
  bert.forward([[1]], [[1]], [[0]])
  const vit = load_model('/models/vit', { task: 'image-classification' })
  vit.forward(pixels)
  const image_processor = load_processor('/models/vit', { task: 'image-classification' })
  image_processor.process([[[0, 0, 0]]])
  // @ts-expect-error Image processors do not tokenize text.
  image_processor.encode('hello')
  // @ts-expect-error Image classifiers cannot generate tokens.
  vit.generate([1], 1)
  // @ts-expect-error BERT requires batched IDs, masks, and type IDs.
  bert.forward([1, 2])
  // @ts-expect-error Audio currently requires the ONNX backend.
  load_model('/models/audio', { task: 'audio-classification' })
}

async function hub_contract() {
  const bundle = await from_pretrained('owner/model', { task: 'text-generation', revision: '0123456789abcdef0123456789abcdef01234567', cache_dir: '/tmp/models' })
  bundle.model.generate(bundle.processor.encode('hello'), 4)
  // @ts-expect-error Text tokenizer does not process RGB pixels.
  bundle.processor.process([[[0, 0, 0]]])
}

function graph_contract(pixels: Tensor) {
  const options = {task:'image-classification',backend:'onnx',graph_dir:'/models/converted',device:'cpu'} as const
  const model = load_model('/models/source', options)
  model.forward(pixels).output
  load_processor('/models/source', options).process([[[0,0,0]]])
  // @ts-expect-error Graph image classifier does not expose GPT-2 generation.
  model.generate([1], 2)
  // @ts-expect-error Hidden-state outputs are not promised by the graph classifier.
  model.forward(pixels).hidden_states
  // @ts-expect-error Prepared graph directory is required.
  load_model('/models/source', {task:'image-classification',backend:'onnx'})
  // @ts-expect-error Text generation through ONNX is not implemented.
  load_model('/models/source', {task:'text-generation',backend:'onnx',graph_dir:'/models/graph'})
}

function audio_contract(features: Tensor) {
  const audio = load_model('/models/ast', {task:'audio-classification',backend:'onnx',graph_dir:'/graphs/ast'})
  audio.forward(features)
  const processor = load_processor('/models/ast', {task:'audio-classification'})
  processor.process(new Float32Array(16000),16000)
  // @ts-expect-error Audio processor takes a waveform and sample rate, not RGB pixels.
  processor.process([[[0,0,0]]])
}

function speech_contract(features: Tensor) {
 const speech=load_model('/models/whisper/source',{task:'automatic-speech-recognition',backend:'onnx',graph_dir:'/models/whisper'})
 speech.transcribe(features)
 load_processor('/models/whisper/source',{task:'automatic-speech-recognition'}).process(new Float32Array(16000),16000)
 // @ts-expect-error Speech has a transcription contract, not a classifier forward method.
 speech.forward(features)
 // @ts-expect-error ASR currently needs prepared ONNX graphs.
 load_model('/models/whisper',{task:'automatic-speech-recognition'})
}
