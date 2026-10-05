// Run from repository root after export-audio-model.py and conversion.
import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import type { Device } from 'affon:compute'
import { load_model, load_processor } from '../../../packages/@affon/huggingface/src/index.ts'
import { GraphProgramRuntime } from '../src/inference/program-runtime.ts'
const directory = getEnv('AFFON_AST_ONNX_DIR') ?? 'apps/hf-inference/artifacts/ast'
const source = getEnv('AFFON_AST_DIR') ?? `${directory}/source`
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const reference = checkpoint.load(`${directory}/reference.safetensors`)
const processor = load_processor(source, { task: 'audio-classification', device })
const model = load_model(source, { task: 'audio-classification', backend: 'onnx', graph_dir: directory })
const runtime = new GraphProgramRuntime(model, device)
const indices = Object.keys(reference).filter(key => key.endsWith('.waveform')).map(key => Number(key.split('.')[0].slice(5))).sort((a,b)=>a-b)
for (const index of indices) {
  const waveform = new Float32Array(reference[`case_${index}.waveform`].to_array() as number[])
  const features = processor.process(waveform, 16000)
  const actual = (features.to_array() as number[][][]).flat(2)
  const expected = (reference[`case_${index}.features`].to_array() as number[][][]).flat(2)
  const feature_error = Math.max(...actual.map((v,i)=>Math.abs(v-expected[i])))
  const start = Date.now()
  const output = runtime.forward({ [model.input_name]: features })[model.output_name]
  const logits = (output.to_array() as number[][])[0]
  const inference_ms = Date.now() - start
  const truth = (reference[`case_${index}.pytorch`].to_array() as number[][])[0]
  const logit_error = Math.max(...logits.map((v,i)=>Math.abs(v-truth[i])))
  const passed = feature_error < 2e-6 && logits.every((v,i)=>Math.abs(v-truth[i]) <= 1e-4 + 1e-4*Math.abs(truth[i]))
  console.log(JSON.stringify({index,device,feature_error,logit_error,inference_ms,top1:logits.indexOf(Math.max(...logits)),passed}))
  if (!passed) throw Error('Audio reference mismatch')
  output.dispose(); features.dispose()
}
runtime.dispose()
