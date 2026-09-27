import fs from 'std:fs'
import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import telemetry from 'std:telemetry'
import * as compute from 'affon:compute'
import native from 'affon:compute/native'

const root = getEnv('COMPUTE_BENCH_FIXTURES')!
const config = JSON.parse(fs.readFileSync(`${root}/manifest.json`))
const diagnostic = getEnv('COMPUTE_BENCH_DISPATCH_ONLY') === '1'
const dispatchMetrics = () => Object.fromEntries(telemetry.metrics()
  .filter(m => m.scope === 'compute.execution' && m.name.startsWith('metal_dispatch_'))
  .map(m => [m.name, m.value]))
const choiceMetrics = () => Object.fromEntries(telemetry.metrics()
  .filter(m => m.scope === 'compute.execution' && (m.name.startsWith('metal_dispatch_') || m.name.startsWith('fusion_hit_matmul_add')))
  .map(m => [m.name, m.value]))
const device = config.device === 'mps' ? 'metal' : 'cpu'
const results = compute.no_grad(() => config.cases.map((c: any) => {
  const saved = checkpoint.load(`${root}/${c.id}.safetensors`) as Record<string, compute.Tensor>
  const inputs = Object.fromEntries(Object.entries(saved).map(([k, v]) => [k, v.to(device)]))
  let a = inputs.a, b = inputs.b
  if (c.narrow_inputs) {
    const narrow = (t: compute.Tensor) => compute.slice(t,compute.range(1,t.shape[0]-1),compute.range(1,t.shape[1]-1))
    a=narrow(a); if (b) b=narrow(b)
  }
  if (c.transpose_a) a = compute.transpose(a, 0, 1)
  if (c.transpose_b) b = compute.transpose(b, b.shape.length - 2, b.shape.length - 1)
  const fused = c.fused ? compute.compile((x: compute.Tensor, weight: compute.Tensor, bias: compute.Tensor) => {
    const value = compute.add(compute.matmul(x, weight), bias)
    return c.fused === 'gelu' ? compute.gelu(value) : value
  }) : null
  function body(): compute.Tensor {
    if (c.kind === 'sequence') {
      let x = a
      for (let i = 0; i < c.length; i++) x = fused ? fused(x, b, inputs.bias) : c.op === 'matmul'
        ? compute.relu(compute.matmul(x, b)) : compute.add(compute.relu(x), b)
      return x
    }
    if (/^(min|max|variance|std|argmin|argmax)_(axis|all)$/.test(c.op))
      return (compute as any)[c.op.split('_')[0]](a, c.axis ?? undefined)
    switch (c.op) {
      case 'gather': return compute.gather(a, c.axis, inputs.index)
      case 'index_select': return compute.index_select(a, c.axis, inputs.index)
      case 'reshape': return compute.reshape(a, c.target)
      case 'contiguous': return compute.contiguous(a)
      case 'permute': return compute.permute(a, c.axes)
      case 'slice': return compute.slice(a, compute.range(1,a.shape[0]-1), compute.range(2,a.shape[1]-1,2))
      case 'squeeze': return compute.squeeze(a)
      case 'unsqueeze': return compute.unsqueeze(a,c.axis)
      case 'cat': return compute.cat([a,b ?? a],c.axis)
      case 'stack': return compute.stack([a,b ?? a],c.axis)
      case 'matmul': return compute.matmul(a, b)
      case 'layer_norm': return compute.layer_norm(a, c.axis, 1e-5)
      case 'softmax': return compute.softmax(a, c.axis)
      case 'sum_axis': return compute.sum(a, c.axis)
      case 'mean_axis': return compute.mean(a, c.axis)
      case 'sum_all': return compute.sum(a)
      case 'mean_all': return compute.mean(a)
      default: return c.b ? (compute as any)[c.op](a, b) : (compute as any)[c.op](a)
    }
  }
  let scope_stats: any = null
  function run() {
    return c.mode === 'scoped' && device === 'metal'
      ? scoped() : body()
  }
  function scoped() {
    const { value, ...stats } = native.$with_graph_execution(body)
    if (diagnostic) scope_stats = stats
    return value
  }
  const before = diagnostic ? choiceMetrics() : {}
  let output = run()
  const dispatch = diagnostic ? Object.fromEntries(Object.entries(choiceMetrics())
    .map(([name, value]) => [name, value - (before[name] ?? 0)] as const)
    .filter(([, count]) => count > 0)) : {}
  if (output.dtype !== (c.output_dtype === 'torch.int64' ? 'i64' : 'f32'))
    throw Error(`${c.id}: output dtype mismatch: ${output.dtype}`)
  const actual = (output.to_array() as number[]).flat(Infinity) as number[]
  const expected = c.expected as number[]
  if (actual.length !== expected.length || JSON.stringify(output.shape) !== JSON.stringify(c.output_shape))
    throw Error(`${c.id}: output shape mismatch`)
  let max_error = 0
  for (let i = 0; i < actual.length; i++) {
    const error = Math.abs(actual[i] - expected[i])
    if (!Number.isFinite(error) || error > (c.output_dtype === 'torch.int64' ? 0 : config.atol + config.rtol * Math.abs(expected[i])))
      throw Error(`${c.id}: mismatch at ${i}: ${actual[i]} vs ${expected[i]}`)
    max_error = Math.max(max_error, error)
  }
  if (diagnostic) {
    if (c.fused && device === 'metal' && dispatch[c.fused === 'gelu' ? 'fusion_hit_matmul_add_gelu_count' : 'fusion_hit_matmul_add_count'] !== c.length)
      throw Error(`${c.id}: expected every compiled call to use the fused epilogue`)
    if (device === 'metal' && c.op === 'matmul' && !Object.keys(dispatch).length)
      throw Error(`${c.id}: missing native dispatch events; rebuild Affon with dispatch tracing`)
    return { id: c.id, correct: true, max_absolute_error: max_error, dispatch, scope_stats,
      implementation: Object.keys(dispatch).length ? 'observed native dispatch' : 'not instrumented' }
  }
  for (let i = 0; i < config.warmup; i++) output = run()
  const repeats = c.completion === 'host-view' ? config.view_repeats : config.repeats
  const samples = []
  const sample_totals = []
  for (let sample = 0; sample < config.samples; sample++) {
    const start = Date.now()
    for (let repeat = 0; repeat < repeats; repeat++) output = run()
    const elapsed = Date.now() - start
    samples.push(elapsed / repeats)
    sample_totals.push(elapsed)
  }
  // Readback/checking and fixture preparation stay outside timing. Eager kernels
  // and scoped graph return already guarantee completion on this backend.
  return { id: c.id, samples_ms: samples, sample_totals_ms: sample_totals,
    correct: true, max_absolute_error: max_error, implementation: 'automatic (not traced)' }
}))
if (!diagnostic && Object.keys(dispatchMetrics()).length)
  throw Error('Dispatch tracing leaked into the timing worker')
fs.writeFileSync(getEnv('COMPUTE_BENCH_OUTPUT')!, JSON.stringify({ results }, null, 2))
