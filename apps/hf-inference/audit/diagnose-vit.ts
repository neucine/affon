import fs from 'std:fs'
import { getEnv } from 'std:process'
import checkpoint from 'affon:checkpoint'
import type { Device, Tensor } from 'affon:compute'
import { encoder_ops, erf_gelu } from '../../../packages/@affon/huggingface/src/encoder-ops.ts'
import { compare_values } from './compare.ts'

const directory = getEnv('AFFON_HF_MODEL_DIR')!
const device = (getEnv('AFFON_DEVICE') ?? 'cpu') as Device
const manifest = JSON.parse(fs.readFileSync(`${directory}/diagnostics.json`))
const data = checkpoint.load(`${directory}/diagnostics.safetensors`) as Record<string, Tensor>
const ops = encoder_ops(directory, device)
const results = []
for (const entry of manifest.cases) {
  const input = data[`${entry.name}.input`].to(device)
  const expected = data[`${entry.name}.output`]
  let output: Tensor
  if (entry.kind === 'gelu') output = erf_gelu(input)
  else if (entry.kind === 'norm') {
    const dim = input.shape[input.ndim - 1]
    ops.require_weight(`${entry.name}.weight`, [dim]); ops.require_weight(`${entry.name}.bias`, [dim])
    output = ops.norm(input, entry.name, entry.eps)
  } else if (entry.kind === 'attention') {
    const dim = input.shape[input.ndim - 1]
    for (const name of ['query', 'key', 'value']) {
      ops.require_weight(`${entry.name}.${name}.weight`, [dim, dim], true)
      ops.require_weight(`${entry.name}.${name}.bias`, [dim])
    }
    output = ops.attention(ops.dense(input, `${entry.name}.query`), ops.dense(input, `${entry.name}.key`), ops.dense(input, `${entry.name}.value`), entry.heads)
  } else {
    const din = input.shape[input.ndim - 1], dout = expected.shape[expected.ndim - 1]
    ops.require_weight(`${entry.name}.weight`, [dout, din], true); ops.require_weight(`${entry.name}.bias`, [dout])
    output = ops.dense(input, entry.name)
  }
  const flatten = (x: Tensor) => (x.to_array() as number[]).flat(Infinity) as number[]
  results.push({ name: entry.name, kind: entry.kind, ...compare_values(flatten(output), flatten(expected)) })
}
const report = { device, passed: results.every(result => result.passed), scope: 'Each operation receives the exact PyTorch input, isolating local from accumulated error', results }
fs.writeFileSync(`${directory}/diagnostic-${device}.json`, JSON.stringify(report, null, 2))
console.log(JSON.stringify(report, null, 2))
if (!report.passed) throw new Error('Isolated ViT operation parity failed')
