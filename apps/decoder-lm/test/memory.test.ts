import { describe, expect, test } from 'std:test'
import telemetry from 'std:telemetry'
import { Session } from 'affon:compute'
import { DecoderModel } from '../src/model.ts'
import { metalAvailable } from './support/metal.ts'

function metricValue(metrics: ReturnType<typeof telemetry.metrics>, group: string, name: string): number {
  return metrics.find(metric => metric.scope === group && metric.name === name)?.value ?? 0
}

describe.skip(() => !metalAvailable() || typeof (globalThis as any).Affon?.trimMemory !== 'function')('decoder-lm memory', () => {
  test('decoder Program forward releases Metal outputs and trims pooled memory', () => {
    const trimMemory = (globalThis as any).Affon.trimMemory as () => number
    trimMemory()
    const model = DecoderModel(64, 32, { numLayers: 1, numHeads: 4, hiddenDim: 64, causal: true, positional: 'learned', maxSeqLen: 16, tieEmbeddings: true })
    const session = new Session({ device: 'metal' }), source = model.forward(2, 8), state = session.initialize(source, { seed: 7 })
    const tokens = session.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], { dtype: 'i64' })
    const baseline = telemetry.metrics()
    for (let index = 0; index < 16; index++) {
      const logits = session.compile(source).run({ token_ids: tokens }, state)
      expect(logits.device).toBe('metal')
      expect(logits.shape).toEqual([2, 8, 64])
      logits.dispose()
    }
    const during = telemetry.metrics()
    const trimmed = trimMemory()
    tokens.dispose()
    state.dispose(); session.dispose()
    const after = telemetry.metrics()
    expect(metricValue(during, 'compute.storage', 'allocation_count') > metricValue(baseline, 'compute.storage', 'allocation_count')).toBe(true)
    expect(trimmed >= 0).toBe(true)
    expect(metricValue(after, 'compute.memory', 'device_pool_live_bytes') <= metricValue(during, 'compute.memory', 'device_pool_live_bytes')).toBe(true)
  })
})
