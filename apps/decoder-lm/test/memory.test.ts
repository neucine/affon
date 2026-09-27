import { describe, expect, test } from 'std:test'
import { axes, no_grad, seed, tensor } from 'affon:compute'
import telemetry from 'std:telemetry'

import { DecoderModel } from '../src/model.ts'
import { metalAvailable } from './support/metal.ts'

function metricValue(
  metrics: ReturnType<typeof telemetry.metrics>,
  group: string,
  name: string,
): number {
  const entry = metrics.find((metric) => metric.scope === group && metric.name === name)
  return entry?.value ?? 0
}

describe.skip(() => !metalAvailable() || typeof (globalThis as any).Affon?.trimMemory !== 'function')('decoder-lm memory', () => {
  test('decoder forward churn reuses Metal buffers and trims back near baseline', () => {
    const trimMemory = (globalThis as any).Affon.trimMemory as () => number
    setDevice('cpu')
    trimMemory()
    seed(7)

    setDevice('metal')
    const model = DecoderModel(64, 32, {
      numLayers: 1,
      numHeads: 4,
      hiddenDim: 64,
      causal: true,
      positional: 'learned',
      maxSeqLen: 16,
      tieEmbeddings: true,
    })
    const tokens = tensor([
      [1, 2, 3, 4, 5, 6, 7, 8],
      [8, 7, 6, 5, 4, 3, 2, 1],
    ], { dtype: 'f32', axes: [axes.batch, axes.token] }).to('metal')

    const baseline = telemetry.metrics()
    no_grad(() => {
      for (let i = 0; i < 16; i += 1) {
        let logits = model(tokens)
        expect(logits.device).toBe('metal')
        expect(logits.shape).toEqual([2, 8, 64])
        logits = null as any
      }
    })

    const during = telemetry.metrics()
    const trimmed = trimMemory()
    const after = telemetry.metrics()
    setDevice('cpu')

    expect(metricValue(during, 'compute.storage', 'allocation_count') > metricValue(baseline, 'compute.storage', 'allocation_count')).toBe(true)
    expect(metricValue(during, 'compute.memory', 'device_pool_hit_count') >= metricValue(baseline, 'compute.memory', 'device_pool_hit_count')).toBe(true)
    expect(metricValue(during, 'compute.storage', 'reuse_count') >= metricValue(baseline, 'compute.storage', 'reuse_count')).toBe(true)
    expect(trimmed >= 0).toBe(true)
    expect(metricValue(after, 'compute.memory', 'device_pool_live_bytes') <= metricValue(during, 'compute.memory', 'device_pool_live_bytes')).toBe(true)
    expect(metricValue(after, 'compute.storage', 'live_metal_bytes') <= metricValue(baseline, 'compute.storage', 'live_metal_bytes') + 1024 * 1024).toBe(true)
  })
})
