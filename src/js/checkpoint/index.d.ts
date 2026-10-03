interface CheckpointModule {
  save(state: Record<string, unknown>, path: string): void
  inspect(path: string): Record<string, {dtype: string; shape: number[]}>
  load(path: string, options?: {names?: readonly string[]}): Record<string, import('affon:compute').Tensor>
  restore<T extends Record<string, unknown>>(
    target: T,
    source: string | Record<string, import('affon:compute').Tensor>,
  ): T
  saveBundle(
    prefix: string,
    bundle: {
      state: Record<string, import('affon:compute').Tensor>
      tensorGroups?: Record<string, Record<string, import('affon:compute').Tensor>>
      manifest?: Record<string, unknown>
    },
  ): void
  loadBundle(prefix: string): {
    state: Record<string, import('affon:compute').Tensor>
    tensorGroups: Record<string, Record<string, import('affon:compute').Tensor>>
    manifest: Record<string, unknown> & {
      statePath: string
      tensorGroupPaths?: Record<string, string> | null
      basePath?: string | null
    }
  }
}
