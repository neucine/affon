import fs from 'std:fs'
import { getEnv } from 'std:process'
import telemetry from 'std:telemetry'
import checkpoint from 'affon:checkpoint'
import type { Device, Tensor } from 'affon:compute'
import { load_bert_processor } from '../../../packages/@affon/huggingface/src/index.ts'
import { load_model, process_rgb_image } from '../../../packages/@affon/huggingface/src/index.ts'
import { compare_values } from './compare.ts'

const directory = getEnv('AFFON_HF_MODEL_DIR')
if (!directory) throw new Error('Set AFFON_HF_MODEL_DIR')
const deviceName = getEnv('AFFON_DEVICE') ?? 'cpu'
if (!/^(cpu|metal|cuda(:\d+)?)$/.test(deviceName)) throw new Error('Invalid AFFON_DEVICE')
const device = deviceName as Device
const reference_directory = getEnv('AFFON_HF_REFERENCE_DIR') ?? directory
const manifest = JSON.parse(fs.readFileSync(`${reference_directory}/reference.json`))
if (manifest.format !== 'affon-hf-domain-reference/v1' || !['bert', 'vit'].includes(manifest.family) || !manifest.cases?.length) throw new Error('Invalid domain manifest')
const reference = checkpoint.load(`${reference_directory}/reference.safetensors`) as Record<string, Tensor>
const results: { name: string; passed: boolean; [key: string]: unknown }[] = []
const memory_samples: { phase: string; values: Record<string, number> }[] = []
function memory(phase: string) {
  memory_samples.push({ phase, values: Object.fromEntries(telemetry.metrics()
    .filter(metric => ['runtime.memory', 'compute.storage', 'compute.memory'].includes(metric.scope)
      && (metric.name.includes('bytes') || metric.name.includes('footprint')))
    .map(metric => [`${metric.scope}.${metric.name}`, metric.value])) })
}
function check(name: string, fn: () => { passed: boolean; [key: string]: unknown }) {
  try { results.push({ name, ...fn() }) }
  catch (error) { results.push({ name, passed: false, error: String(error) }) }
}
function compare(name: string, value: Tensor) {
  const target = reference[name]
  if (!target || JSON.stringify(target.shape) !== JSON.stringify(value.shape)) throw new Error(`Shape mismatch for ${name}`)
  const flatten = (x: Tensor) => (x.to_array() as number[]).flat(Infinity) as number[]
  return { tensor: name, ...compare_values(flatten(value), flatten(target)) }
}
function compareForward(index: number, value: { output: Tensor; hidden_states: Tensor[]; pooled?: Tensor; pooler?: Tensor }) {
  const output = compare(`case_${index}.output`, value.output)
  const hidden = value.hidden_states.map((state, layer) => compare(`case_${index}.hidden_${layer}`, state))
  const checks = [output, ...hidden]
  if (value.pooled) checks.push(compare(`case_${index}.pooled`, value.pooled))
  if (value.pooler) checks.push(compare(`case_${index}.pooler`, value.pooler))
  return { passed: checks.every(result => result.passed), output_passed: output.passed,
    hidden_states_passed: hidden.every(result => result.passed),
    first_failing_hidden_state: hidden.findIndex(result => !result.passed) < 0 ? null : hidden.findIndex(result => !result.passed), checks }
}
memory('before_model_load')
const start = Date.now()
try {
  if (manifest.family === 'bert') {
    const model = load_model(directory, { task: 'feature-extraction', device })
    const processor = load_bert_processor(directory)
    memory('after_model_load')
    for (let i = 0; i < manifest.cases.length; i++) {
      const sample = manifest.cases[i]
      let prepared: ReturnType<typeof processor.encode_batch> | undefined
      check(`case_${i}.processor`, () => {
        const actual = processor.encode_batch(sample.texts, sample.pairs)
        const expected = { input_ids: sample.input_ids, token_type_ids: sample.token_type_ids, attention_mask: sample.attention_mask }
        const passed = JSON.stringify(actual) === JSON.stringify(expected)
        if (passed) prepared = actual
        return { passed, actual, expected }
      })
      check(`case_${i}.forward`, () => {
        const input = prepared ?? sample
        return { ...compareForward(i, model.forward(input.input_ids, input.attention_mask, input.token_type_ids)), input_source: prepared ? 'native_processor' : 'reference' }
      })
      memory(`after_case_${i}`)
    }
    check('reject_all_masked', () => {
      try { model.forward([[0]], [[0]], [[0]]); return { passed: false } }
      catch { return { passed: true } }
    })
  } else {
    const model = load_model(directory, { task: 'image-classification', device })
    memory('after_model_load')
    for (let i = 0; i < manifest.cases.length; i++) {
      const sample = manifest.cases[i]
      let prepared: Tensor | undefined
      check(`case_${i}.processor`, () => {
        const rgb = sample.input_kind === 'reference_rgb'
          ? reference[`case_${i}.rgb`].to_array() as number[][][]
          : Array.from({ length: sample.height }, (_, y) => Array.from({ length: sample.width }, (_, x) =>
          Array.from({ length: 3 }, (_, c) => (x * 3 + y * 5 + c * 47) % 256)))
        const pixels = process_rgb_image(directory, rgb, device)
        const result = compare(`case_${i}.pixels`, pixels)
        if (result.passed) prepared = pixels
        return result
      })
      check(`case_${i}.forward`, () => {
        const value = model.forward(prepared ?? reference[`case_${i}.pixels`])
        const comparison = compareForward(i, value)
        const row = (value.output.to_array() as number[][])[0]
        let top1 = 0
        for (let j = 1; j < row.length; j++) if (row[j] > row[top1]) top1 = j
        return { ...comparison, input_source: prepared ? 'native_processor' : 'reference', top1, expected_top1: sample.top1[0], passed: comparison.passed && top1 === sample.top1[0] }
      })
      memory(`after_case_${i}`)
    }
  }
} catch (error) { results.push({ name: 'setup', passed: false, error: String(error) }) }
const report = {
  format: 'affon-hf-domain-audit/v1', family: manifest.family, model_id: manifest.model_id,
  revision: manifest.revision, reference_versions: manifest.versions, device, dtype: 'f32',
  build_profile: getEnv('AFFON_AUDIT_BUILD') ?? 'unspecified',
  implementation: '@affon/huggingface eager adapters; A&S erf-GELU approximation; ViT patch convolution expressed as matmul',
  scope: 'Forward uses native processor outputs when their parity check passes; reference inputs are used only to isolate processor failures',
  elapsed_ms: Date.now() - start, memory_samples, passed: results.every(result => result.passed), results,
}
const reportPath = getEnv('AFFON_HF_REPORT') ?? `${directory}/audit-${deviceName.replace(':', '-')}.json`
fs.writeFileSync(reportPath, JSON.stringify(report, null, 2))
console.log(JSON.stringify(report, null, 2))
if (!report.passed) throw new Error(`Domain audit found gaps; see ${reportPath}`)
