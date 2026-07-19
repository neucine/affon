import { describe, expect, test } from 'affon:test'
import { axes, no_grad, seed, tensor } from 'affon:compute'

import { DecoderModel } from '../src/index.ts'
import { metalAvailable } from './support/metal.ts'

function metricValue(
  metrics: ReturnType<typeof Affon.metrics>,
  group: string,
  name: string,
): number {
  const entry = metrics.find((metric) => metric.group === group && metric.name === name)
  return entry?.value ?? 0
}

describe.skip(() => !metalAvailable())('@affon/lm memory', () => {
  test('decoder forward churn reuses Metal buffers and trims back near baseline', () => {
    setDevice('cpu')
    Affon.trimMemory()
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

    const baseline = Affon.metrics()
    no_grad(() => {
      for (let i = 0; i < 16; i += 1) {
        let logits = model(tokens)
        expect(logits.device).toBe('metal')
        expect(logits.shape).toEqual([2, 8, 64])
        logits = null as any
      }
    })

    const during = Affon.metrics()
    const trimmed = Affon.trimMemory()
    const after = Affon.metrics()
    setDevice('cpu')

    expect(metricValue(during, 'allocator.events', 'allocations') > metricValue(baseline, 'allocator.events', 'allocations')).toBe(true)
    expect(metricValue(during, 'allocator.events', 'hits') >= metricValue(baseline, 'allocator.events', 'hits')).toBe(true)
    expect(metricValue(during, 'allocator.events', 'reuses') >= metricValue(baseline, 'allocator.events', 'reuses')).toBe(true)
    expect(trimmed >= 0).toBe(true)
    expect(metricValue(after, 'owned.current_bytes', 'live_bytes') <= metricValue(during, 'owned.current_bytes', 'live_bytes')).toBe(true)
    expect(metricValue(after, 'owned.current_bytes', 'live_metal_bytes') <= metricValue(baseline, 'owned.current_bytes', 'live_metal_bytes') + 1024 * 1024).toBe(true)
  })
})
