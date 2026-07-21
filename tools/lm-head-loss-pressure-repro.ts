import {
  clip_grad_norm,
  contiguous,
  cross_entropy_indexed,
  grad,
  matmul,
  parameter,
  reshape,
  tensor,
  transpose,
} from 'affon:compute'
import telemetry from 'std:telemetry'

const batch = 4
const seq = 255
const dModel = 256
const vocab = 50257
const tokens = batch * seq

const weight = parameter([vocab, dModel], { dtype: 'f32' }).randn().to('metal')
const hidden = parameter([tokens, dModel], { dtype: 'f32' }).randn().to('metal')
const targets = tensor(
  Array.from({ length: tokens }, (_, index) => index % vocab),
  { dtype: 'i64' },
).to('metal')
const params = [weight, hidden]

function metricValue(group: string, name: string): number {
  return telemetry.metrics().find((metric) => metric.scope === group && metric.name === name)?.value ?? 0
}

function mb(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1)
}

for (let i = 1; i <= 30; i++) {
  const hiddenT = transpose(hidden, 0, 1)
  const vocabByToken = matmul(weight, hiddenT, { hint: 'projection', source: 'higher_level_module' })
  const flatLogits = transpose(vocabByToken, 0, 1)
  const logits = reshape(contiguous(flatLogits), [tokens, vocab])
  const loss = cross_entropy_indexed(logits, targets, 1)
  grad(loss, params)
  clip_grad_norm(params, 1.0)
  if (i % 5 === 0) {
    console.log(
      `lm head loss pressure iteration ${i}/30`
      + ` metal_live_mb=${mb(metricValue('compute.storage', 'live_metal_bytes'))}`
      + ` metal_device_mb=${mb(metricValue('compute.memory', 'metal_device_current_allocated_bytes'))}`
      + ` metal_pool_mb=${mb(metricValue('compute.memory', 'metal_pool_live_bytes'))}`,
    )
  }
}

console.log('lm head loss pressure done')
