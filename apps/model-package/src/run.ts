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
    const model = load_graph(directory)
    const session = new Session({ device })
    const state = session.initialize(model.forward, { parameters: model.parameters })
    const inputs: Record<string, Tensor> = {}
    for (const [name, data] of Object.entries(request.inputs)) {
      inputs[name] = session.tensor(data as number[], {dtype: 'f32'})
    }
    const evaluated = session.compile(model.forward).run(inputs, state) as Tensor | Tensor[]
    const values = Array.isArray(evaluated) ? evaluated : [evaluated]
    const outputs: Record<string, unknown> = {}
    for (const [index, name] of model.output_names.entries()) {
      const value = values[index]
      outputs[name] = {shape: value.shape, data: (value.to_array() as number[]).flat(Infinity)}
      value.dispose()
    }
    for (const value of Object.values(inputs)) value.dispose()
    state.dispose()
    session.dispose()
    return {outputs}
  },
  'affon/llama-tokens/v1': () => {
    if (!Array.isArray(request.input_ids) || !Number.isInteger(request.max_new_tokens)
        || request.max_new_tokens < 0) throw Error('Expected input_ids and nonnegative integer max_new_tokens')
    const model = load_llama(directory)
    const session = new Session({ device })
    const state = session.initialize(model.forward(1), { parameters: model.parameters })
    const token_ids = [...request.input_ids]
    try {
      for (let step = 0; step < request.max_new_tokens; step++) {
        const source = model.forward(token_ids.length, token_ids.length - 1)
        const ids = session.tensor(token_ids, { dtype: 'i64' })
        const mask = session.tensor([[Array.from({ length: token_ids.length }, (_, row) => Array.from({ length: token_ids.length }, (_, column) => column > row ? 1 : 0))]], { dtype: 'i64' })
        let next = 0
        try {
          const [logits, ...hidden] = session.compile(source).run({ ids, mask }, state) as Tensor[]
          try {
            const row = (logits.to_array() as number[][][])[0][0]
            for (let candidate = 1; candidate < row.length; candidate++) if (row[candidate] > row[next]) next = candidate
          } finally { logits.dispose(); for (const value of hidden) value.dispose() }
        } finally { ids.dispose(); mask.dispose() }
        token_ids.push(next)
        if (next === model.config.eos_token_id) break
      }
    } finally { state.dispose(); session.dispose() }
    return {token_ids}
  },
}
const adapter = adapters[getEnv('MODEL_PACKAGE_RUNTIME') ?? '']
if (!adapter) throw Error('Unsupported runtime adapter')
fs.writeFileSync(`${stage}/response.json`, JSON.stringify(adapter()))
