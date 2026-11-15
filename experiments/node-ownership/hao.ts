import { Tensor } from 'affon:core'
import telemetry from 'std:telemetry'

function liveBytes(): number {
  return telemetry.metrics().find(metric =>
    metric.scope === 'compute.storage' && metric.name === 'live_bytes'
  )?.value ?? 0
}

function allocateAndDrop(count: number, elements: number, explicit: boolean): void {
  const tensors: Tensor[] = []
  for (let i = 0; i < count; i++) tensors.push(Tensor.zeros([elements], { dtype: 'f32' }))
  console.log(JSON.stringify({ label: 'before-drop', liveBytes: liveBytes() }))
  if (explicit) {
    for (const tensor of tensors) tensor.dispose()
    console.log(JSON.stringify({ label: 'after-explicit', liveBytes: liveBytes() }))
  }
}

console.log(JSON.stringify({ label: 'start', liveBytes: liveBytes() }))
allocateAndDrop(500, 262144, false)
console.log(JSON.stringify({ label: 'after-function-return', liveBytes: liveBytes() }))
allocateAndDrop(500, 262144, true)
console.log(JSON.stringify({ label: 'after-explicit-function-return', liveBytes: liveBytes() }))
