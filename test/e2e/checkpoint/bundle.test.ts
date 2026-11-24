import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import { Session } from 'affon:core'
import { describe, expect, test, values } from 'std:test'
import { captureError } from '../../support/errors.ts'
import { createTempPath, removeTempPath } from '../../support/temp.ts'

describe('checkpoint bundle', () => {
  test('saves and loads a checkpoint bundle with tensor groups and manifest metadata', async () => {
    const session = new Session({ device: 'cpu' })
    const prefix = await createTempPath('affon-checkpoint-bundle')
    const state = {
      weight: session.tensor([[1, 2], [3, 4]], { dtype: 'f32' }),
    }
    const optimizer = {
      momentum: session.tensor([0.1, 0.2], { dtype: 'f32' }),
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
    loaded.state.weight.dispose(); loaded.tensorGroups.optimizer.momentum.dispose()

    fs.writeFileSync(`${prefix}.json`, JSON.stringify({
      ...manifest,
      basePath: prefix.slice(0, prefix.lastIndexOf('/')),
      tensorGroupPaths: { optimizer: 'missing-optimizer.safetensors' },
    }))
    expect(captureError(() => checkpoint.loadBundle(prefix))).toBeInstanceOf(Error)
    fs.writeFileSync(`${prefix}.json`, JSON.stringify({
      ...manifest,
      basePath: prefix.slice(0, prefix.lastIndexOf('/')),
    }))
    const recovered = checkpoint.loadBundle(prefix)
    expect(values(recovered.tensorGroups.optimizer.momentum)).toBeAllClose([0.1, 0.2])
    recovered.state.weight.dispose(); recovered.tensorGroups.optimizer.momentum.dispose()
    session.dispose()
    await removeTempPath(prefix)
  })

  test('missing and malformed bundle files fail, then a valid bundle still loads', async () => {
    const session = new Session({ device: 'cpu' })
    const prefix = await createTempPath('affon-bundle-recovery')
    try {
      const missing = captureError(() => checkpoint.loadBundle(prefix))
      expect(missing).toBeInstanceOf(Error)

      const weight = session.tensor([2, 4], { dtype: 'f32' })
      checkpoint.saveBundle(prefix, { state: { weight }, manifest: { step: 7 } })
      weight.dispose()
      const original = fs.readFileSync(`${prefix}.json`)
      fs.writeFileSync(`${prefix}.json`, '{broken')
      expect(captureError(() => checkpoint.loadBundle(prefix))).toBeInstanceOf(Error)

      fs.writeFileSync(`${prefix}.json`, JSON.stringify({ statePath: 'missing.safetensors' }))
      expect(captureError(() => checkpoint.loadBundle(prefix))).toBeInstanceOf(Error)

      fs.writeFileSync(`${prefix}.json`, original)
      const loaded = checkpoint.loadBundle(prefix)
      expect(loaded.manifest.step).toBe(7)
      expect(values(loaded.state.weight)).toEqual([2, 4])
      loaded.state.weight.dispose()
    } finally {
      session.dispose()
      await removeTempPath(prefix)
    }
  })

  test('a failed tensor group write does not publish a manifest and can be retried', async () => {
    const session = new Session({ device: 'cpu' })
    const prefix = await createTempPath('affon-bundle-partial')
    try {
      const weight = session.tensor([1, 3])
      const disposed = session.tensor([0])
      disposed.dispose()
      expect(captureError(() => checkpoint.saveBundle(prefix, {
        state: { weight }, tensorGroups: { optimizer: { bad: disposed } },
      }))).toBeInstanceOf(Error)
      expect(captureError(() => checkpoint.loadBundle(prefix))).toBeInstanceOf(Error)

      const momentum = session.tensor([0.5])
      checkpoint.saveBundle(prefix, { state: { weight }, tensorGroups: { optimizer: { momentum } } })
      const loaded = checkpoint.loadBundle(prefix)
      expect(values(loaded.state.weight)).toEqual([1, 3])
      expect(values(loaded.tensorGroups.optimizer.momentum)).toEqual([0.5])
      loaded.state.weight.dispose(); loaded.tensorGroups.optimizer.momentum.dispose()
      weight.dispose(); momentum.dispose()
    } finally {
      session.dispose()
      await removeTempPath(prefix)
    }
  })
})
