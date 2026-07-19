import fs from 'std:fs'
import { loadStateTree, restorePersistedState, saveStateTree } from 'affon:compute/persistence.ts'

function restore(target: any, source: string | Record<string, any>): any {
  const state = typeof source === 'string' ? loadStateTree(source) : source
  return restorePersistedState(target, state)
}

function dirname(path: string): string {
  const normalized = path.replaceAll('\\', '/')
  const index = normalized.lastIndexOf('/')
  if (index < 0) return '.'
  if (index === 0) return '/'
  return normalized.slice(0, index)
}

function basename(path: string): string {
  const normalized = path.replaceAll('\\', '/')
  const index = normalized.lastIndexOf('/')
  return index < 0 ? normalized : normalized.slice(index + 1)
}

function isAbsolutePath(path: string): boolean {
  return path.startsWith('/') || /^[A-Za-z]:[\\/]/.test(path)
}

function joinPath(base: string, leaf: string): string {
  if (base === '.' || base.length === 0) return leaf
  if (base.endsWith('/')) return `${base}${leaf}`
  return `${base}/${leaf}`
}

type LoadedState = Record<string, any>
type TensorGroups = Record<string, LoadedState>

type CheckpointBundleManifest = Record<string, unknown> & {
  statePath: string
  tensorGroupPaths?: Record<string, string> | null
  basePath?: string | null
}

type CheckpointBundle = {
  state: LoadedState
  tensorGroups: TensorGroups
  manifest: CheckpointBundleManifest
}

function bundlePaths(prefix: string, groupNames: readonly string[]): { manifestPath: string; statePath: string; tensorGroupPaths: Record<string, string> } {
  const tensorGroupPaths: Record<string, string> = {}
  for (let i = 0; i < groupNames.length; i++) {
    tensorGroupPaths[groupNames[i]] = `${prefix}.${groupNames[i]}.safetensors`
  }
  return {
    manifestPath: `${prefix}.json`,
    statePath: `${prefix}.safetensors`,
    tensorGroupPaths,
  }
}

function saveBundle(
  prefix: string,
  bundle: {
    state: LoadedState
    tensorGroups?: TensorGroups
    manifest?: Record<string, unknown>
  },
): void {
  if (typeof prefix !== 'string' || prefix.length === 0) {
    throw new AffonError('invalid_arg', 'checkpoint.saveBundle(prefix, bundle) expects a non-empty prefix')
  }
  const groups = bundle.tensorGroups ?? {}
  const groupNames = Object.keys(groups)
  const paths = bundlePaths(prefix, groupNames)

  saveStateTree(bundle.state, paths.statePath)
  for (let i = 0; i < groupNames.length; i++) {
    const name = groupNames[i]
    saveStateTree(groups[name], paths.tensorGroupPaths[name])
  }

  const manifestDir = dirname(paths.manifestPath)
  const tensorGroupPaths: Record<string, string> = {}
  for (let i = 0; i < groupNames.length; i++) {
    const name = groupNames[i]
    tensorGroupPaths[name] = basename(paths.tensorGroupPaths[name])
  }
  const manifest: CheckpointBundleManifest = {
    ...(bundle.manifest ?? {}),
    statePath: basename(paths.statePath),
    tensorGroupPaths: groupNames.length > 0 ? tensorGroupPaths : null,
    basePath: manifestDir,
  }
  fs.writeFileSync(paths.manifestPath, JSON.stringify(manifest, null, 2))
}

function loadBundle(prefix: string): CheckpointBundle {
  if (typeof prefix !== 'string' || prefix.length === 0) {
    throw new AffonError('invalid_arg', 'checkpoint.loadBundle(prefix) expects a non-empty prefix')
  }
  const manifestPath = `${prefix}.json`
  const manifest = JSON.parse(fs.readFileSync(manifestPath)) as CheckpointBundleManifest
  const manifestDir = dirname(manifestPath)
  const basePath = manifest.basePath ?? manifestDir
  const statePath = isAbsolutePath(manifest.statePath)
    ? manifest.statePath
    : joinPath(basePath, manifest.statePath)
  const tensorGroups: TensorGroups = {}
  const tensorGroupPaths = manifest.tensorGroupPaths ?? {}
  for (const [name, groupPath] of Object.entries(tensorGroupPaths)) {
    const resolved = isAbsolutePath(groupPath)
      ? groupPath
      : joinPath(basePath, groupPath)
    tensorGroups[name] = loadStateTree(resolved)
  }
  return {
    state: loadStateTree(statePath),
    tensorGroups,
    manifest,
  }
}

const checkpoint = {
  save: saveStateTree,
  load: loadStateTree,
  restore,
  saveBundle,
  loadBundle,
}

export const save = checkpoint.save
export const load = checkpoint.load
export { restore, saveBundle, loadBundle }
export default checkpoint
