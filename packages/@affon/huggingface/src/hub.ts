import fs from 'std:fs'
import http from 'std:http'
import { run } from 'std:process'

const files = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']
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

/**
 * Download a pinned public HF model into a verified local snapshot.
 * @param model_id Hub repository name, optionally prefixed by its owner.
 * @param options Immutable revision, absolute cache directory, and offline policy.
 * @returns Local directory consumable by load_model and load_processor.
 * @remarks Uses native HTTP streaming; shasum, mkdir, mv, and rm remain required on PATH. No Python is used.
 * Only a single model.safetensors is supported; pickle and shards are rejected.
 */
export async function snapshot_download(model_id: string, options: HubOptions): Promise<string> {
  if (!/^[A-Za-z0-9_-][A-Za-z0-9._-]*(\/[A-Za-z0-9_-][A-Za-z0-9._-]*)?$/.test(model_id)
    || model_id.includes('--') || model_id.includes('..')) throw new Error('Invalid HF repository ID')
  if (!/^[a-f0-9]{40}$/.test(options.revision)) throw new Error('HF revision must be an immutable 40-character commit SHA')
  if (!options.cache_dir.startsWith('/') || options.cache_dir.includes('\0')) throw new Error('HF cache_dir must be an absolute path')
  const directory = `${options.cache_dir}/models--${model_id.replace('/', '--')}/${options.revision}`
  const manifest_path = `${directory}/affon-snapshot.json`
  if (fs.existsSync(manifest_path)) {
    const manifest = JSON.parse(fs.readFileSync(manifest_path)) as Manifest
    if (manifest.format !== 'affon-hf-snapshot/v1' || manifest.model_id !== model_id || manifest.revision !== options.revision
      || !Array.isArray(manifest.files) || !['config.json', 'model.safetensors'].every(name => manifest.files.some(file => file.name === name))
      || new Set(manifest.files.map(file => file.name)).size !== manifest.files.length) throw new Error('Invalid HF cache manifest')
    for (const file of manifest.files) {
      if (!files.includes(file.name) || !Number.isSafeInteger(file.size) || file.size <= 0 || !/^[a-f0-9]{64}$/.test(file.sha256)
        || !fs.existsSync(`${directory}/${file.name}`) || fs.statSync(`${directory}/${file.name}`).size !== file.size
        || await digest(`${directory}/${file.name}`) !== file.sha256) throw new Error(`HF cache integrity failure: ${file.name}`)
    }
    return directory
  }
  if (options.local_files_only) throw new Error(`HF snapshot is not cached: ${model_id}@${options.revision}`)
  const base = `https://huggingface.co/${model_id}`
  const response = await http.get(`https://huggingface.co/api/models/${model_id}/revision/${options.revision}`)
  if (!response.ok) throw new Error(`HF repository lookup failed: HTTP ${response.status}`)
  const info = response.json<{ sha: string; siblings: { rfilename: string }[] }>()
  if (info.sha !== options.revision || !Array.isArray(info.siblings)) throw new Error('HF returned an unexpected repository revision')
  const available = new Set(info.siblings.map((entry: { rfilename: string }) => entry.rfilename))
  if (!available.has('model.safetensors')) throw new Error(available.has('model.safetensors.index.json') ? 'Sharded HF checkpoints are not supported yet' : 'HF snapshot requires model.safetensors; pickle conversion is not supported')
  if (!available.has('config.json')) throw new Error('HF snapshot has no config.json')
  await command('mkdir', ['-p', directory])
  const manifest: Manifest = { format: 'affon-hf-snapshot/v1', model_id, revision: options.revision, files: [] }
  for (const name of files.filter(name => available.has(name))) {
    const target = `${directory}/${name}`
    const temporary = `${target}.partial-${Date.now()}-${Math.random().toString(16).slice(2)}`
    try {
      const downloaded = await http.download(`${base}/resolve/${options.revision}/${name}`, temporary, {
        maxBytes: name === 'model.safetensors' ? 500 * 1024 * 1024 : 20 * 1024 * 1024,
      })
      const size = downloaded.bytesWritten
      if (!size) throw new Error(`Empty HF artifact: ${name}`)
      const sha256 = downloaded.sha256
      if (name.endsWith('.json')) JSON.parse(fs.readFileSync(temporary))
      await command('mv', ['-f', temporary, target])
      manifest.files.push({ name, size, sha256 })
    } finally {
      await command('rm', ['-f', temporary])
    }
  }
  const temporary = `${manifest_path}.partial-${Date.now()}-${Math.random().toString(16).slice(2)}`
  fs.writeFileSync(temporary, JSON.stringify(manifest, null, 2))
  await command('mv', ['-f', temporary, manifest_path])
  return directory
}
