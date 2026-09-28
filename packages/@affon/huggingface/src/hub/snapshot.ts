import fs from 'std:fs'
import http from 'std:http'
import { run } from 'std:process'
import { parse_checkpoint_index } from './checkpoint-index.ts'

const metadataFiles = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']
const indexName = 'model.safetensors.index.json'
const maxWeightBytes = 8 * 1024 ** 3, maxSnapshotBytes = 16 * 1024 ** 3
const isWeight = (name: string) => /^[A-Za-z0-9][A-Za-z0-9_-]*\.safetensors$/.test(name)
export type HubOptions = {
  /** Immutable 40-character Hub commit SHA; branch names are intentionally rejected. */
  revision: string
  /** Absolute writable directory. Cache entries are namespaced by repository and revision. */
  cache_dir: string
  /** Require a complete, checksum-verified cache entry; never contact the Hub. */
  local_files_only?: boolean
}
type Manifest = { format: string; model_id: string; revision: string; files: { name: string; size: number; sha256: string }[] }

async function command(cmd: string, args: string[]) {
  const result = await run({ cmd, args, check: false, maxOutputBytes: 4 * 1024 * 1024 })
  if (result.exitCode !== 0) throw new Error(`HF ${cmd} failed: ${result.stderr.slice(0, 500)}`)
  return result.stdout.trim()
}
async function digest(path: string) {
  const hash = (await command('shasum', ['-a', '256', path])).split(/\s/)[0]
  if (!/^[a-f0-9]{64}$/.test(hash)) throw new Error('Invalid SHA-256 response')
  return hash
}
function shards(directory: string) {
  const files = [...new Set(Object.values(parse_checkpoint_index(JSON.parse(fs.readFileSync(`${directory}/${indexName}`)))))]
  if (files.length > 256) throw Error('HF checkpoint exceeds 256 shards')
  return files
}
function validateLayout(directory: string, names: Set<string>) {
  if (!names.has('config.json')) throw Error('HF snapshot has no config.json')
  const indexed = names.has(indexName)
  const weights = indexed ? shards(directory) : ['model.safetensors']
  if (weights.some(name => !names.has(name)) || [...names].some(name => isWeight(name) && !weights.includes(name)))
    throw Error('HF snapshot is missing weights or contains unindexed shards')
}

/** Download a pinned public single-file or indexed-shard SafeTensors snapshot.
 * Streaming transport; each weight file is bounded to 8 GiB, snapshot to 16 GiB.
 * Completion is published only after every referenced file has been acquired.
 */
export async function snapshot_download(model_id: string, options: HubOptions): Promise<string> {
  if (!/^[A-Za-z0-9_-][A-Za-z0-9._-]*(\/[A-Za-z0-9_-][A-Za-z0-9._-]*)?$/.test(model_id)
    || model_id.includes('--') || model_id.includes('..')) throw Error('Invalid HF repository ID')
  if (!/^[a-f0-9]{40}$/.test(options.revision)) throw Error('HF revision must be an immutable 40-character commit SHA')
  if (!options.cache_dir.startsWith('/') || options.cache_dir.includes('\0')) throw Error('HF cache_dir must be an absolute path')
  const directory = `${options.cache_dir}/models--${model_id.replace('/', '--')}/${options.revision}`
  const manifest_path = `${directory}/affon-snapshot.json`
  if (fs.existsSync(manifest_path)) {
    const manifest = JSON.parse(fs.readFileSync(manifest_path)) as Manifest
    if (manifest.format !== 'affon-hf-snapshot/v1' || manifest.model_id !== model_id || manifest.revision !== options.revision
      || !Array.isArray(manifest.files) || !manifest.files.some(file => file.name === 'config.json')
      || manifest.files.some(file => typeof file.name !== 'string' || !(metadataFiles.includes(file.name) || file.name === indexName || isWeight(file.name)))
      || new Set(manifest.files.map(file => file.name)).size !== manifest.files.length)
      throw Error('Invalid HF cache manifest')
    let total = 0
    for (const file of manifest.files) {
      if (typeof file.name !== 'string' || !(metadataFiles.includes(file.name) || file.name === indexName || isWeight(file.name))
        || !Number.isSafeInteger(file.size) || file.size <= 0 || file.size > (isWeight(file.name) ? maxWeightBytes : 20 * 1024 ** 2)
        || !/^[a-f0-9]{64}$/.test(file.sha256) || (total += file.size) > maxSnapshotBytes
        || !fs.existsSync(`${directory}/${file.name}`) || fs.statSync(`${directory}/${file.name}`).size !== file.size
        || await digest(`${directory}/${file.name}`) !== file.sha256) throw Error(`HF cache integrity failure: ${file.name}`)
    }
    validateLayout(directory, new Set(manifest.files.map(file => file.name)))
    return directory
  }
  if (options.local_files_only) throw Error(`HF snapshot is not cached: ${model_id}@${options.revision}`)
  const base = `https://huggingface.co/${model_id}`
  const response = await http.get(`https://huggingface.co/api/models/${model_id}/revision/${options.revision}`)
  if (!response.ok) throw Error(`HF repository lookup failed: HTTP ${response.status}`)
  const info = response.json<{ sha: string; siblings: { rfilename: string }[] }>()
  if (info.sha !== options.revision || !Array.isArray(info.siblings)) throw Error('HF returned an unexpected repository revision')
  const available = new Set(info.siblings.map(entry => entry.rfilename))
  if (!available.has('model.safetensors') && !available.has(indexName)) throw Error('HF snapshot requires SafeTensors; pickle conversion is not supported')
  if (!available.has('config.json')) throw Error('HF snapshot has no config.json')
  await command('mkdir', ['-p', directory])
  const manifest: Manifest = { format: 'affon-hf-snapshot/v1', model_id, revision: options.revision, files: [] }
  let total = 0
  async function download(name: string) {
    const target = `${directory}/${name}`, temporary = `${target}.partial-${Date.now()}-${Math.random().toString(16).slice(2)}`
    try {
      const downloaded = await http.download(`${base}/resolve/${options.revision}/${name}`, temporary, {
        maxBytes: Math.min(isWeight(name) ? maxWeightBytes : 20 * 1024 ** 2, maxSnapshotBytes - total),
      })
      const size = downloaded.bytesWritten
      if (!size) throw Error(`Empty HF artifact: ${name}`)
      total += size
      if (total > maxSnapshotBytes) throw Error('HF snapshot exceeds 16 GiB')
      if (name.endsWith('.json')) JSON.parse(fs.readFileSync(temporary))
      await command('mv', ['-f', temporary, target])
      manifest.files.push({name, size, sha256: downloaded.sha256})
    } finally { await command('rm', ['-f', temporary]) }
  }
  for (const name of metadataFiles.filter(name => available.has(name))) await download(name)
  if (available.has('model.safetensors')) await download('model.safetensors')
  else {
    await download(indexName)
    const weights = shards(directory)
    if (weights.some(name => !available.has(name))) throw Error('HF index references a missing shard')
    for (const name of weights) await download(name)
  }
  validateLayout(directory, new Set(manifest.files.map(file => file.name)))
  const temporary = `${manifest_path}.partial-${Date.now()}-${Math.random().toString(16).slice(2)}`
  fs.writeFileSync(temporary, JSON.stringify(manifest, null, 2))
  await command('mv', ['-f', temporary, manifest_path])
  return directory
}
