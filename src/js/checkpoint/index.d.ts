interface CheckpointModule {
  save(state: Record<string, unknown>, path: string): void
  inspect(path: string): Record<string, {dtype: string; shape: number[]}>
  load(path: string, options?: {names?: readonly string[]}): Record<string, import('affon:core').Tensor>
  restore<T extends Record<string, unknown>>(
    target: T,
    source: string | Record<string, import('affon:core').Tensor>,
  ): T
  saveBundle(
    prefix: string,
    bundle: {
      state: Record<string, import('affon:core').Tensor>
      tensorGroups?: Record<string, Record<string, import('affon:core').Tensor>>
      manifest?: Record<string, unknown>
    },
  ): void
  loadBundle(prefix: string): {
    state: Record<string, import('affon:core').Tensor>
    tensorGroups: Record<string, Record<string, import('affon:core').Tensor>>
    manifest: Record<string, unknown> & {
      statePath: string
      tensorGroupPaths?: Record<string, string> | null
      basePath?: string | null
    }
  }
}
