// Batch-one MobileNetV2 service. Run from the repository root.
import fs from 'std:fs'
import http from 'std:http'
import { getEnv } from 'std:process'
import type { Device } from 'affon:compute'
import { load_graph } from '../../../packages/@affon/onnx/src/index.ts'
import { process_mobilenet_image } from '../../../packages/@affon/huggingface/src/processors/mobilenet.ts'

const started = Date.now()
const directory = getEnv('MODEL_DIR')
if (!directory) throw Error('MODEL_DIR must contain graph.json, weights.safetensors and source/preprocessor_config.json')
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const name = getEnv('MODEL_NAME') ?? 'vision'
const port = Number(getEnv('PORT') ?? '8080')
if (!/^[a-zA-Z0-9_-]+$/.test(name)) throw Error('Invalid MODEL_NAME')
if (!Number.isInteger(port) || port < 1 || port > 65535) throw Error('Invalid PORT')
const config = JSON.parse(fs.readFileSync(`${directory}/source/preprocessor_config.json`))
if (config.image_processor_type !== 'MobileNetV2ImageProcessor') throw Error('This service requires MobileNetV2 preprocessing')
const model = load_graph(directory, device)
if (Object.keys(model.graph.inputs).join(',') !== 'pixels'
    || JSON.stringify(model.graph.inputs.pixels) !== '[1,3,224,224]'
    || model.graph.outputs.join(',') !== 'logits') throw Error('Expected pixels [1,3,224,224] → logits graph')
// Check the complete processor/model contract before accepting traffic.
const probe = model.forward({pixels: process_mobilenet_image(`${directory}/source`, [[[0, 0, 0]]], device)}).logits
if (probe.shape.length !== 2 || probe.shape[0] !== 1) throw Error('Expected batch-one classification logits')
const outputShape = probe.shape
probe.dispose()
const base = `/v2/models/${name}`
const json = (value: unknown, status = 200) => ({status, json: value})
function validate(input: any) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw Error('Expected a JSON object')
  if (input.id !== undefined && typeof input.id !== 'string') throw Error('id must be a string')
  if (input.parameters !== undefined && Object.keys(input.parameters ?? {}).length) throw Error('Request parameters are unsupported')
  if (input.outputs !== undefined && (!Array.isArray(input.outputs) || input.outputs.length !== 1
      || input.outputs[0]?.name !== 'logits' || input.outputs[0]?.parameters !== undefined)) throw Error('Only logits output is supported')
  if (!Array.isArray(input.inputs) || input.inputs.length !== 1) throw Error('Expected one rgb input')
  const rgb = input.inputs[0]
  if (!rgb || rgb.name !== 'rgb' || rgb.datatype !== 'UINT8' || rgb.parameters !== undefined) throw Error('Expected rgb UINT8 input without parameters')
  const shape = rgb.shape
  if (!Array.isArray(shape) || shape.length !== 4 || shape[0] !== 1 || shape[3] !== 3
      || !shape.every(Number.isInteger) || shape[1] < 1 || shape[1] > 512 || shape[2] < 1 || shape[2] > 512
      || Math.max(shape[1], shape[2]) > 4 * Math.min(shape[1], shape[2])) {
    throw Error('Expected shape [1,height,width,3], sides 1–512 and aspect ratio at most 4')
  }
  if (!Array.isArray(rgb.data) || rgb.data.length !== shape[1] * shape[2] * 3
      || rgb.data.some((v: number) => !Number.isInteger(v) || v < 0 || v > 255)) throw Error('Expected flat RGB8 data matching shape')
  return {height: shape[1], width: shape[2], data: rgb.data}
}

http.serve({
  hostname: getEnv('HOST') ?? '127.0.0.1', port,
  maxBodyBytes: 4 * 1024 * 1024, maxResponseBytes: 1024 * 1024, requestTimeoutMs: 120000,
  handler(request: HttpServerRequest): HttpServerResponse {
    const path = request.url.split('?')[0]
    if (request.method === 'GET') {
      if (['/v2/health/live', '/v2/health/ready', `${base}/ready`].includes(path)) return {status: 200, body: ''}
      if (path === '/v2') return json({name: 'affon-vision', version: 'experimental', extensions: []})
      if (path === base) return json({name, platform: 'affon-onnx-static',
        inputs: [{name: 'rgb', datatype: 'UINT8', shape: [1, -1, -1, 3]}],
        outputs: [{name: 'logits', datatype: 'FP32', shape: outputShape}]})
    }
    if (path !== `${base}/infer`) return json({error: 'Not found'}, 404)
    if (request.method !== 'POST') return json({error: 'Use POST'}, 405)
    if (request.headers['content-type']?.split(';')[0].trim().toLowerCase() !== 'application/json') return json({error: 'Use application/json'}, 415)
    let input: any, rgb: ReturnType<typeof validate>
    try { input = request.json(); rgb = validate(input) }
    catch (error) { return json({error: String(error)}, 400) }
    try {
      const start = Date.now()
      const pixels = device === 'cpu'
        ? process_mobilenet_image(`${directory}/source`, rgb.data, device, [rgb.height, rgb.width])
        : process_mobilenet_image(`${directory}/source`, Array.from({length: rgb.height}, (_, y) => Array.from({length: rgb.width}, (_, x) => {
          const offset = (y * rgb.width + x) * 3
          return rgb.data.slice(offset, offset + 3)
        })), device)
      const prepared = Date.now()
      const logits = model.forward({pixels}).logits
      try {
        const data = (logits.to_array() as number[]).flat(Infinity)
        if (data.some(v => !Number.isFinite(v))) throw Error('Nonfinite model output')
        const finished = Date.now()
        return json({model_name: name, ...(input.id !== undefined ? {id: input.id} : {}),
          parameters: {preprocess_ms: prepared - start, inference_ms: finished - prepared},
          outputs: [{name: 'logits', datatype: 'FP32', shape: [...logits.shape], data}]})
      } finally { logits.dispose(); pixels.dispose() }
    } catch (error) { return json({error: String(error)}, 500) }
  },
})
console.log(JSON.stringify({ready: true, model: name, device, load_and_probe_ms: Date.now() - started, port}))
