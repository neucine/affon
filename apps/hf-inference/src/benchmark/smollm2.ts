/** One model per process. CPU/Metal/CUDA share this workload; CUDA requires its own run. */
import fs from 'std:fs'
import { create_phase_profiler } from './telemetry.ts'
import telemetry from 'std:telemetry'
import { getEnv } from 'std:process'
import { from_pretrained } from '../../../../packages/@affon/huggingface/src/index.ts'
import { SMOLLM2_MODELS, type SmolLM2Size } from '../inference/models.ts'
import type { Device, Tensor } from 'affon:compute'
const size = (getEnv('AFFON_SMOLLM2_SIZE') ?? '360M') as SmolLM2Size
if (!Object.hasOwn(SMOLLM2_MODELS, size)) throw Error('Invalid AFFON_SMOLLM2_SIZE')
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const report = getEnv('AFFON_BENCH_REPORT')
if (!report) throw Error('Set AFFON_BENCH_REPORT')
const spec = SMOLLM2_MODELS[size], samples: unknown[] = []
function memory(phase: string) {
  samples.push({phase, values: Object.fromEntries(telemetry.metrics().filter(m => ['runtime.memory','compute.storage','compute.memory'].includes(m.scope) && (m.name.includes('bytes') || m.name.includes('footprint'))).map(m => [`${m.scope}.${m.name}`,m.value]))})
}
function next(logits: Tensor, length: number) {
  const row = (logits.to_array() as number[][][])[0][length - 1]
  let id = 0
  for (let i=1;i<row.length;i++) if (row[i]>row[id]) id=i
  return id
}
const profile = create_phase_profiler()
memory('before_load')
const {model, processor} = await profile.async('hf.smollm2.load', () => from_pretrained(spec.id, {revision:spec.revision,cache_dir:getEnv('AFFON_HF_CACHE') ?? '/tmp/affon-hub-cache',local_files_only:getEnv('AFFON_HF_OFFLINE') !== '0',device,task:'text-generation'}))
const load_ms = profile.phases[0].elapsed_ms
memory('after_load')
if (!('encode_chat' in processor)) throw Error('Missing chat processor')
const prompt = 'Explain why the sky is blue in two sentences.'
const ids = processor.encode_chat([{role:'user',content:prompt}]), results: unknown[] = []
// A warmup plus two measured requests, fixed 16-token budget and synchronized readback.
for (let iteration=0;iteration<3;iteration++) {
  const session=model.create_session(), generated:number[]=[]
  const phase = iteration === 0 ? 'warmup' : `measured_${iteration}`
  profile.sync(`hf.smollm2.${phase}.prefill`, () => {
    generated.push(next(session.forward(ids).logits, ids.length))
  })
  const prefill_ms=profile.phases[profile.phases.length-1].elapsed_ms
  profile.sync(`hf.smollm2.${phase}.decode`, () => {
  while (generated.length<16 && generated[generated.length-1]!==model.config.eos_token_id)
    generated.push(next(session.forward([generated[generated.length-1]]).logits,1))
  })
  const decode_ms=profile.phases[profile.phases.length-1].elapsed_ms
  results.push({warmup:iteration===0,prompt_tokens:ids.length,generated_tokens:generated.length,prefill_ms,prefill_tokens_per_second:ids.length*1000/Math.max(prefill_ms,1),decode_ms,decode_tokens_per_second:(generated.length-1)*1000/Math.max(decode_ms,1),completion:processor.decode(generated,{skipSpecialTokens:true})})
  memory(`iteration_${iteration}`);session.reset()
  console.log(JSON.stringify(results[results.length-1]))
}
fs.writeFileSync(report,JSON.stringify({model:spec.id,revision:spec.revision,device,precision:'f32 (BF16 storage)',load_ms,prompt,results,memory_samples:samples,profiles:profile.phases,metal_command_timing_requested:getEnv('AFFON_METAL_COMMAND_TIMING') === '1',notes:['Native spans and counter deltas come from std:telemetry; elapsed_ms remains host wall timing.', 'Metal GPU totals are command-buffer durations, not per-kernel timings; use only when metal_gpu_timing_complete is true.', 'Gauge peaks are process-lifetime snapshots, not phase-local peaks.', 'Profiling collection adds overhead; compare throughput in separate runs.', 'Load includes offline snapshot hashing and tokenizer creation.','Telemetry samples are not a continuous peak; collect OS peak RSS separately.','Decode excludes prefill and counts only subsequent token steps.']},null,2))
console.log(`Wrote ${report}`)
