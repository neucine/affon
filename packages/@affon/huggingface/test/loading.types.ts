import { load_model, from_pretrained, load_processor } from '../src/index.ts'
import type { Tensor } from 'affon:compute'

// Compile-only: task selection must preserve each domain's input signature.
function contracts(pixels: Tensor) {
  const text = load_model('/models/gpt2', { task: 'text-generation' })
  text.forward(2)
  text.parameters
  const bert = load_model('/models/bert', { task: 'feature-extraction' })
  bert.forward(1, 1)
  bert.parameters
  const vit = load_model('/models/vit', { task: 'image-classification' })
  vit.forward.inspect()
  vit.parameters
  const image_processor = load_processor('/models/vit', { task: 'image-classification' })
  image_processor.process([[[0, 0, 0]]])
  // @ts-expect-error Image processors do not tokenize text.
  image_processor.encode('hello')
  // @ts-expect-error Image classifiers have a fixed Program, not a shape factory.
  vit.forward(pixels)
  // @ts-expect-error BERT Program dimensions are numeric.
  bert.forward([1], [1])
  // @ts-expect-error Audio currently requires the ONNX backend.
  load_model('/models/audio', { task: 'audio-classification' })
}

async function hub_contract() {
  const bundle = await from_pretrained('owner/model', { task: 'text-generation', revision: '0123456789abcdef0123456789abcdef01234567', cache_dir: '/tmp/models' })
  bundle.model.forward(bundle.processor.encode('hello').length)
  // @ts-expect-error Text tokenizer does not process RGB pixels.
  bundle.processor.process([[[0, 0, 0]]])
}

function graph_contract(pixels: Tensor) {
  const options = {task:'image-classification',backend:'onnx',graph_dir:'/models/converted'} as const
  const model = load_model('/models/source', options)
  model.forward.inspect()
  model.parameters
  model.input_name
  load_processor('/models/source', {task:'image-classification',device:'cpu'}).process([[[0,0,0]]])
  // @ts-expect-error Graph image classifier does not expose GPT-2 generation.
  model.generate([1], 2)
  // @ts-expect-error Graph definitions do not execute themselves.
  model.forward(pixels)
  // @ts-expect-error Prepared graph directory is required.
  load_model('/models/source', {task:'image-classification',backend:'onnx'})
  // @ts-expect-error Text generation through ONNX is not implemented.
  load_model('/models/source', {task:'text-generation',backend:'onnx',graph_dir:'/models/graph'})
}

function audio_contract(features: Tensor) {
  const audio = load_model('/models/ast', {task:'audio-classification',backend:'onnx',graph_dir:'/graphs/ast'})
  audio.forward.inspect()
  const processor = load_processor('/models/ast', {task:'audio-classification'})
  processor.process(new Float32Array(16000),16000)
  // @ts-expect-error Audio processor takes a waveform and sample rate, not RGB pixels.
  processor.process([[[0,0,0]]])
}

function speech_contract(features: Tensor) {
 const speech=load_model('/models/whisper/source',{task:'automatic-speech-recognition',backend:'onnx',graph_dir:'/models/whisper'})
 speech.encoder.forward.inspect()
 speech.cross.forward.inspect()
 speech.decoder.forward.inspect()
 speech.embeddings
 speech.positions
 speech.generation.cacheCapacity
 load_processor('/models/whisper/source',{task:'automatic-speech-recognition'}).process(new Float32Array(16000),16000)
 // @ts-expect-error Program definitions do not own transcription policy or execution.
 speech.transcribe(features)
 // @ts-expect-error ASR currently needs prepared ONNX graphs.
 load_model('/models/whisper',{task:'automatic-speech-recognition'})
}
