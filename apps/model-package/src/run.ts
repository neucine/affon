// Invoked by package.py after verification. Adapters own their request contracts.
import fs from 'std:fs'
import { getEnv } from 'std:process'
import { Session, type Device, type Tensor } from 'affon:compute'
import { load_graph } from '../../../packages/@affon/onnx/src/index.ts'
import { load_llama } from '../../../packages/@affon/huggingface/src/index.ts'

const stage = getEnv('MODEL_PACKAGE_STAGE')
if (!stage) throw Error('Use package.py run to verify the package before execution')
const request = JSON.parse(fs.readFileSync(`${stage}/request.json`))
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const directory = getEnv('MODEL_PACKAGE_MODEL_DIR') ?? `${stage}/model`
const adapters: Record<string, () => unknown> = {
  'affon/prepared-onnx/v1': () => {
    if (!request.inputs || typeof request.inputs !== 'object' || Array.isArray(request.inputs)) {
      throw Error('Expected inputs: a map of tensor names to nested numeric arrays')
    }
    const model = load_graph(directory, device)
    const session = new Session({ device })
    const inputs: Record<string, Tensor> = {}
    for (const [name, data] of Object.entries(request.inputs)) {
      inputs[name] = session.tensor(data as number[], {dtype: 'f32'})
    }
    const outputs: Record<string, unknown> = {}
    for (const [name, value] of Object.entries(model.forward(inputs))) {
      outputs[name] = {shape: value.shape, data: (value.to_array() as number[]).flat(Infinity)}
      value.dispose()
    }
    for (const value of Object.values(inputs)) value.dispose()
    model.dispose()
    session.dispose()
    return {outputs}
  },
  'affon/llama-tokens/v1': () => {
    if (!Array.isArray(request.input_ids) || !Number.isInteger(request.max_new_tokens)
        || request.max_new_tokens < 0) throw Error('Expected input_ids and nonnegative integer max_new_tokens')
    const model = load_llama(directory, device)
    const token_ids = model.generate(request.input_ids, request.max_new_tokens)
    model.dispose()
    return {token_ids}
  },
}
const adapter = adapters[getEnv('MODEL_PACKAGE_RUNTIME') ?? '']
if (!adapter) throw Error('Unsupported runtime adapter')
fs.writeFileSync(`${stage}/response.json`, JSON.stringify(adapter()))
