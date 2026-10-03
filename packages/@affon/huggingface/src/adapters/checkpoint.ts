import fs from 'std:fs'
import checkpoint from 'affon:checkpoint'
import type { ModelTensor } from '../../../models/src/shared/parameters.ts'
import { parse_checkpoint_index } from '../hub/checkpoint-index.ts'

/** Validate the complete checkpoint catalog without loading its tensor payloads.
 * Reads individual tensors on demand so source and device copies are bounded to
 * one weight. Index ownership must agree exactly with each shard's metadata.
 */
export function open_checkpoint(directory: string) {
  const indexPath = `${directory}/model.safetensors.index.json`
  const indexed = fs.existsSync(indexPath)
  const ownership = indexed ? parse_checkpoint_index(JSON.parse(fs.readFileSync(indexPath))) : undefined
  const files = ownership ? [...new Set(Object.values(ownership))] : ['model.safetensors']
  const catalog: Record<string, {file: string; dtype: string; shape: number[]}> = Object.create(null)
  for (const file of files) {
    const metadata = checkpoint.inspect(`${directory}/${file}`)
    for (const [name, info] of Object.entries(metadata)) {
      if (Object.hasOwn(catalog, name)) throw Error(`Duplicate checkpoint tensor: ${name}`)
      if (ownership && ownership[name] !== file) throw Error(`Checkpoint index ownership mismatch: ${name}`)
      catalog[name] = {file, ...info}
    }
  }
  if (ownership) for (const name of Object.keys(ownership))
    if (!Object.hasOwn(catalog, name)) throw Error(`Missing indexed checkpoint tensor: ${name}`)
  return {
    catalog,
    read(name: string): ModelTensor {
      if (!Object.hasOwn(catalog, name)) throw Error(`Missing checkpoint tensor: ${name}`)
      return (checkpoint.load(`${directory}/${catalog[name].file}`, {names:[name]}) as Record<string, ModelTensor>)[name]
    },
  }
}
