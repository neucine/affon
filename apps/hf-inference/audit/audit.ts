import fs from 'std:fs'
import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import telemetry from 'std:telemetry'
import type { Device, Tensor } from 'affon:compute'
import { createHFTokenizerFromFile } from '../../../packages/@affon/tokenizers/src/index.ts'
import { load_model } from '../../../packages/@affon/huggingface/src/index.ts'
import { compare_values } from './compare.ts'

const directory = getEnv('AFFON_HF_MODEL_DIR')
if (!directory) throw new Error('Set AFFON_HF_MODEL_DIR to the prepared reference directory')
const device = getEnv('AFFON_DEVICE') ?? 'cpu'
if (!/^(cpu|metal|cuda(:\d+)?)$/.test(device)) throw new Error('Invalid AFFON_DEVICE')
const reference_directory = getEnv('AFFON_HF_REFERENCE_DIR') ?? directory
const manifest = JSON.parse(fs.readFileSync(`${reference_directory}/reference.json`))
if (manifest.format !== 'affon-hf-inference-reference/v1' || !Array.isArray(manifest.cases) || manifest.cases.length === 0) {
  throw new Error('Invalid or empty reference manifest')
}
const started = Date.now()
const memory = () => Object.fromEntries(telemetry.metrics()
  .filter(metric => ['runtime.memory', 'compute.storage', 'compute.memory'].includes(metric.scope)
    && (metric.name.includes('bytes') || metric.name.includes('footprint')))
  .map(metric => [`${metric.scope}.${metric.name}`, metric.value]))
const memory_samples = [{ phase: 'before_load', values: memory() }]
const model = load_model(directory, { task: 'text-generation', device: device as Device })
const load_ms = Date.now() - started
memory_samples.push({ phase: 'after_load', values: memory() })
const tokenizer = createHFTokenizerFromFile(`${directory}/tokenizer.json`)
const reference = checkpoint.load(`${reference_directory}/reference.safetensors`) as Record<string, Tensor>
const results: { name: string; passed: boolean; [key: string]: unknown }[] = []
function check(name: string, fn: () => { passed: boolean; [key: string]: unknown }) {
  try { results.push({ name, ...fn() }) }
  catch (error) { results.push({ name, passed: false, error: String(error) }) }
}
function compareTensor(name: string, actual: Tensor, start?: number, end?: number) {
  const source = reference[name]
  if (!source) throw new Error(`Missing reference for ${name}`)
  const sourceValues = source.to_array() as number[][][]
  const expectedValues = start === undefined ? sourceValues : sourceValues.map(row => row.slice(start, end))
  const expectedShape = start === undefined ? source.shape : [source.shape[0], end! - start!, source.shape[2]]
  if (JSON.stringify(actual.shape) !== JSON.stringify(expectedShape)) throw new Error(`Shape mismatch for ${name}`)
  const flatten = (value: Tensor) => (value.to_array() as number[]).flat(Infinity) as number[]
  return compare_values(flatten(actual), expectedValues.flat(Infinity) as number[])
}
for (let i = 0; i < manifest.cases.length; i++) {
  const sample = manifest.cases[i]
  check(`case_${i}.tokenizer`, () => {
    const actual = tokenizer.encode(sample.prompt)
    return { passed: JSON.stringify(actual) === JSON.stringify(sample.input_ids), actual, expected: sample.input_ids }
  })
  // Reference IDs deliberately isolate model errors from tokenizer errors.
  check(`case_${i}.forward`, () => {
    const start = Date.now()
    const output = model.forward(sample.input_ids)
    const logits = compareTensor(`case_${i}.logits`, output.logits)
    const hidden_states = output.hidden_states.map((hidden, layer) => compareTensor(`case_${i}.hidden_${layer}`, hidden))
    return { passed: logits.passed && hidden_states.every(result => result.passed), logits, hidden_states,
      forward_and_comparison_ms: Date.now() - start }
  })
  check(`case_${i}.cached_forward`, () => {
    const session = model.create_session()
    const checks = []
    for (let start = 0; start < sample.input_ids.length;) {
      const end = Math.min(sample.input_ids.length, start + (start === 0 ? 1 : 2))
      const output = session.forward(sample.input_ids.slice(start, end))
      checks.push(compareTensor(`case_${i}.logits`, output.logits, start, end))
      output.hidden_states.forEach((hidden, layer) => checks.push(compareTensor(`case_${i}.hidden_${layer}`, hidden, start, end)))
      start = end
    }
    session.reset()
    return { passed: checks.every(result => result.passed), checks }
  })
  check(`case_${i}.generation`, () => {
    const start = Date.now()
    const actual = model.generate(sample.input_ids, sample.max_new_tokens)
    return { passed: JSON.stringify(actual) === JSON.stringify(sample.generated_ids), actual,
      expected: sample.generated_ids, cached_generation_ms: Date.now() - start }
  })
  check(`case_${i}.decode`, () => {
    const actual = tokenizer.decode(sample.generated_ids, { skipSpecialTokens: false })
    return { passed: actual === sample.decoded, actual, expected: sample.decoded }
  })
  memory_samples.push({ phase: `after_case_${i}`, values: memory() })
}
for (const [name, fn] of [
  ['empty_input', () => model.forward([])],
  ['invalid_token', () => model.forward([model.config.vocab_size])],
  ['context_overflow', () => model.generate([0], model.config.n_positions)],
] as const) {
  check(name, () => {
    try { fn(); return { passed: false } }
    catch { return { passed: true } }
  })
}
const report = {
  format: 'affon-hf-inference-audit/v1', model_id: manifest.model_id, revision: manifest.revision,
  device, dtype: 'f32', reference_versions: manifest.versions,
  build_profile: getEnv('AFFON_AUDIT_BUILD') ?? 'unspecified',
  weight_origin: manifest.weight_origin ?? 'pretrained',
  scope: 'batch=1, unpadded, eager GPT-2; full forward and cached greedy generation; reference artifacts converted to f32 SafeTensors',
  load_ms, memory_samples, passed: results.every(result => result.passed), results,
}
const reportPath = getEnv('AFFON_HF_REPORT') ?? `${directory}/audit-${device.replace(':', '-')}.json`
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2))
console.log(JSON.stringify(report, null, 2))
if (!report.passed) throw new Error(`Inference audit found gaps; see ${reportPath}`)
