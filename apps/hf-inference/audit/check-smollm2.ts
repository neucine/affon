import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { getEnv, run } from 'std:process'
import { load_llama, load_smollm2_processor, format_smollm2_chat } from '../../../packages/@affon/huggingface/src/index.ts'
import type { Device, Tensor } from 'affon:compute'
const directory = getEnv('AFFON_SMOLLM2_DIR') ?? '/tmp/affon-smollm-source'
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const ref = JSON.parse(fs.readFileSync(`${directory}/smollm2-reference.json`))
for (const name of ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors']) {
  const hash = (await run({cmd: 'shasum', args: ['-a', '256', `${directory}/${name}`]})).stdout.split(/\s/)[0]
  if (hash !== ref.sha256[name]) throw Error(`Reference source checksum mismatch: ${name}`)
}
const tensors = checkpoint.load(`${directory}/smollm2-reference.safetensors`) as Record<string, Tensor>
const processor = load_smollm2_processor(directory), model = load_llama(directory, device)
const results: unknown[] = []
const numerical_failures: {name: string; failures: number; elements: number; max_error: number}[] = []
function equal(a: unknown, b: unknown, name: string) {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw Error(`Mismatch: ${name}`)
}
function close(actual: Tensor, expected: { shape: readonly number[]; to_array(): unknown }, name: string) {
  equal(actual.shape, expected.shape, `${name} shape`)
  const a = (actual.to_array() as number[]).flat(Infinity) as number[], b = (expected.to_array() as number[]).flat(Infinity) as number[]
  let maxError = 0, failures = 0
  for (let i = 0; i < a.length; i++) {
    const error = Math.abs(a[i] - b[i]); maxError = Math.max(maxError, error)
    if (!Number.isFinite(error) || error > 5e-4 + 1e-4 * Math.abs(b[i])) failures++
  }
  if (failures) {
    numerical_failures.push({name, failures, elements:a.length, max_error:maxError})
    console.log(`${name}: ${failures}/${a.length} outside 5e-4 + 1e-4*abs(ref); max=${maxError}`)
  }
  return maxError
}
for (let i = 0; i < ref.cases.length; i++) {
  const c = ref.cases[i], start = Date.now(), messages = [{role: 'user' as const, content: c.prompt}]
  equal(format_smollm2_chat(messages), c.formatted, 'chat template')
  equal(processor.encode_chat(messages), c.ids, 'tokenization')
  const output = model.forward(c.ids)
  for (let j = 0; j < c.hidden_count; j++) close(output.hidden_states[j], tensors[`case.${i}.hidden.${j}`], `case ${i} hidden ${j}`)
  const logitError = close(output.logits, tensors[`case.${i}.logits`], `case ${i} logits`)
  const ids = model.generate(c.ids, 8)
  equal(ids, c.generated, 'cached greedy IDs')
  equal(model.generate(c.ids, 8, {use_cache: false}), c.generated, 'uncached greedy IDs')
  equal(processor.decode(ids.slice(c.ids.length), {skipSpecialTokens: true}), c.completion, 'completion')
  // Multi-token append tests both RoPE offsets and the offset causal mask.
  const session = model.create_session(), split = Math.floor(c.ids.length / 2)
  session.forward(c.ids.slice(0, split))
  const full = tensors[`case.${i}.logits`].to_array() as number[][][]
  close(session.forward(c.ids.slice(split)).logits, { shape: [full.length, full[0].length - split, full[0][0].length], to_array: () => full.map(row => row.slice(split)) }, `case ${i} cached chunk`)
  session.reset()
  const result = {prompt: c.prompt, completion: c.completion, logit_max_error: logitError, elapsed_ms: Date.now() - start}
  results.push(result); console.log(JSON.stringify(result))
}
fs.writeFileSync(`${directory}/smollm2-${device}.json`, JSON.stringify({device, model: ref.model, revision: ref.revision, tolerance: {atol:5e-4, rtol:1e-4}, results, numerical_failures, passed:numerical_failures.length === 0}, null, 2))
if (numerical_failures.length) throw Error(`SmolLM2 ${device}: ${numerical_failures.length} numerical checks failed; see report`)
console.log(`SmolLM2 ${device}: all reference checks passed`)
