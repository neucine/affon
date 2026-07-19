import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { describe, expect, test, values } from 'std:test'
import { tensor } from 'affon:compute'

describe('checkpoint bundle', () => {
  test('saves and loads a checkpoint bundle with tensor groups and manifest metadata', () => {
    const prefix = `/tmp/affon-checkpoint-bundle-${Date.now()}-${Math.floor(Math.random() * 1e6)}`
    const state = {
      weight: tensor([[1, 2], [3, 4]], { dtype: 'f32' }),
    }
    const optimizer = {
      momentum: tensor([0.1, 0.2], { dtype: 'f32' }),
    }

    checkpoint.saveBundle(prefix, {
      state,
      tensorGroups: { optimizer },
      manifest: {
        format: 'affon-test-checkpoint-bundle/v1',
        step: 3,
      },
    })

    const manifest = JSON.parse(fs.readFileSync(`${prefix}.json`)) as {
      format: string
      step: number
      statePath: string
      tensorGroupPaths: Record<string, string> | null
    }
    expect(manifest.format).toBe('affon-test-checkpoint-bundle/v1')
    expect(manifest.step).toBe(3)
    expect(manifest.statePath).toBe(`${prefix.split('/').pop()}.safetensors`)
    expect(manifest.tensorGroupPaths?.optimizer).toBe(`${prefix.split('/').pop()}.optimizer.safetensors`)

    const loaded = checkpoint.loadBundle(prefix)
    expect(values(loaded.state.weight)).toEqual([[1, 2], [3, 4]])
    expect(values(loaded.tensorGroups.optimizer.momentum)).toBeAllClose([0.1, 0.2])
    expect(loaded.manifest.format).toBe('affon-test-checkpoint-bundle/v1')
  })
})
