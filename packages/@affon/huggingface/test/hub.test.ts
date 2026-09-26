import fs from 'std:fs'
import { run } from 'std:process'
import { describe, expect, test } from 'std:test'
import { snapshot_download, load_processor } from '../src/index.ts'

const revision = '0123456789abcdef0123456789abcdef01234567'
async function rejects(fn: () => Promise<unknown>, message: string) {
  let failure = ''
  try { await fn() } catch (error) { failure = String(error) }
  expect(failure.includes(message)).toBe(true)
}
describe('HF snapshots', () => {
  test('rejects mutable revisions and invalid repository paths before network access', async () => {
    await rejects(() => snapshot_download('owner/model', { revision: 'main', cache_dir: '/tmp' }), 'immutable')
    await rejects(() => snapshot_download('../model', { revision, cache_dir: '/tmp' }), 'repository')
    await rejects(() => snapshot_download('owner/model', { revision, cache_dir: 'relative' }), 'absolute')
  })
  test('offline cache validates content and rejects incomplete or corrupted snapshots', async () => {
    const root = (await run({ cmd: 'mktemp', args: ['-d', '/tmp/affon-hf-cache-test.XXXXXX'] })).stdout.trim()
    try {
      const options = { revision, cache_dir: root, local_files_only: true }
      await rejects(() => snapshot_download('owner/model', options), 'not cached')
      const directory = `${root}/models--owner--model/${revision}`
      await run({ cmd: 'mkdir', args: ['-p', directory] })
      fs.writeFileSync(`${directory}/config.json`, '{"model_type":"gpt2"}')
      // Snapshot integrity is separate from SafeTensors parsing in load_model.
      fs.writeFileSync(`${directory}/model.safetensors`, 'test weight bytes')
      const files = []
      for (const name of ['config.json', 'model.safetensors']) {
        const sha256 = (await run({ cmd: 'shasum', args: ['-a', '256', `${directory}/${name}`] })).stdout.split(/\s/)[0]
        files.push({ name, size: fs.statSync(`${directory}/${name}`).size, sha256 })
      }
      fs.writeFileSync(`${directory}/affon-snapshot.json`, JSON.stringify({ format: 'affon-hf-snapshot/v1', model_id: 'owner/model', revision, files }))
      expect(await snapshot_download('owner/model', options)).toBe(directory)
      fs.writeFileSync(`${directory}/model.safetensors`, 'TEST weight bytes')
      await rejects(() => snapshot_download('owner/model', options), 'integrity failure')
      files[0].name = '../config.json'
      fs.writeFileSync(`${directory}/affon-snapshot.json`, JSON.stringify({ format: 'affon-hf-snapshot/v1', model_id: 'owner/model', revision, files }))
      await rejects(() => snapshot_download('owner/model', options), 'Invalid HF cache manifest')
    } finally {
      await run({ cmd: 'rm', args: ['-rf', root] })
    }
  })
  test('processor dispatch rejects unsupported task/model combinations', () => {
    expect(() => load_processor('packages/@affon/huggingface/test/fixtures/bert', { task: 'image-classification' })).toThrow('Unsupported HF processor/task')
  })
  test('legacy ViT processor config applies scalar size and rescaling defaults', () => {
    const processor = load_processor('packages/@affon/huggingface/test/fixtures/vit', { task: 'image-classification' })
    const pixels = processor.process([[[255, 0, 255]]])
    expect(pixels.shape).toEqual([1, 3, 2, 2])
    expect(pixels.to_array()).toEqual([[[[1, 1], [1, 1]], [[-1, -1], [-1, -1]], [[1, 1], [1, 1]]]])
  })
})
