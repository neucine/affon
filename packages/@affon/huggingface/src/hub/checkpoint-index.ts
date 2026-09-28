/** HF shard indexes may reference only flat SafeTensors filenames, never paths. */
export function parse_checkpoint_index(value: unknown): Record<string, string> {
  const map = (value as {weight_map?: unknown} | null)?.weight_map
  if (!map || typeof map !== 'object' || Array.isArray(map) || !Object.keys(map).length)
    throw Error('Invalid HF checkpoint index weight_map')
  const result: Record<string, string> = Object.create(null)
  for (const [name, file] of Object.entries(map)) {
    if (!name || name === '__metadata__' || typeof file !== 'string'
      || !/^[A-Za-z0-9][A-Za-z0-9_-]*\.safetensors$/.test(file))
      throw Error('Invalid HF checkpoint shard reference')
    result[name] = file
  }
  return result
}
