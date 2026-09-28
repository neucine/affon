import { from_pretrained } from '../../../packages/@affon/huggingface/src/index.ts'
import { getEnv } from 'std:process'
import { SMOLLM2_MODELS, type SmolLM2Size } from '../src/inference/models.ts'
import type { Device } from 'affon:compute'
const size = (getEnv('AFFON_SMOLLM2_SIZE') ?? '135M') as SmolLM2Size
if (!Object.hasOwn(SMOLLM2_MODELS, size)) throw Error('Invalid SmolLM2 size')
const SMOLLM2_MODEL = SMOLLM2_MODELS[size]
const {model, processor, directory} = await from_pretrained(SMOLLM2_MODEL.id, {
  revision: SMOLLM2_MODEL.revision,
  cache_dir: getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache',
  task: 'text-generation', device: (getEnv('AFFON_DEVICE') ?? 'cpu') as Device,
  local_files_only: getEnv('AFFON_HF_OFFLINE') === '1',
})
if (!('encode_chat' in processor)) throw Error('Missing chat processor')
const ids = processor.encode_chat([{role:'user',content:'What is the capital of France?'}])
const result = model.generate(ids, 16)
console.log(JSON.stringify({directory, model: SMOLLM2_MODEL.id, completion: processor.decode(result.slice(ids.length), {skipSpecialTokens:true}), generated_tokens:result.length-ids.length}))
