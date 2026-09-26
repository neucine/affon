import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import type { Device, Tensor } from 'affon:compute'
import { load_vit, process_rgb_image } from '../../../packages/@affon/huggingface/src/index.ts'

const directory = getEnv('AFFON_HF_MODEL_DIR')!
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const reference = checkpoint.load(`${directory}/reference.safetensors`) as Record<string, Tensor>
const model = load_vit(directory, device)
const state: Record<string, Tensor> = {}
const cases = Object.keys(reference).filter(key => /^case_\d+\.pixels$/.test(key)).sort()
if (!cases.length) throw Error('No ViT pixel cases in reference checkpoint')
for (const key of cases) {
  const prefix = key.slice(0, -'.pixels'.length)
  const rgb = reference[`${prefix}.rgb`]
  const pixels = rgb ? process_rgb_image(directory, rgb.to_array() as number[][][], device) : reference[key]
  state[`${prefix}.pixels`] = pixels
  const result = model.forward(pixels)
  state[`${prefix}.output`] = result.output
  result.hidden_states.forEach((hidden, index) => { state[`${prefix}.hidden_${index}`] = hidden })
  // Preserve the original case-zero capture names for the sensitivity probe.
  if (prefix === 'case_0') {
    state.output = result.output
    result.hidden_states.forEach((hidden, index) => { state[`hidden_${index}`] = hidden })
  }
}
checkpoint.save(state, `${directory}/native-hidden-${device}.safetensors`)
console.log(`Captured ${cases.length} ViT cases on ${device}`)
