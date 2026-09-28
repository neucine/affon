declare module "affon:checkpoint" {
  import type { ComputeState, Module, Tensor } from "affon:compute"

  type CheckpointState = ComputeState | Module<any, any>
  type LoadedCheckpoint = ComputeState
  type CheckpointTensorGroups = Record<string, LoadedCheckpoint>
  type CheckpointBundleManifest = Record<string, unknown> & {
    statePath: string
    tensorGroupPaths?: Record<string, string> | null
    basePath?: string | null
  }
  type LoadedCheckpointBundle = {
    state: LoadedCheckpoint
    tensorGroups: CheckpointTensorGroups
    manifest: CheckpointBundleManifest
  }

  /**
   * @summary Save a module or structured state tree to disk.
   * @category Persistence
   * @semantics
   * Flattens named tensor leaves into a SafeTensors-backed checkpoint file.
   */
  export function save(state: CheckpointState, path: string): void

  /**
   * @summary Load a checkpoint state dictionary from disk.
   * @category Persistence
   * @semantics
   * Reads a serialized checkpoint file and returns the named tensor mapping.
   * Supports F32, F64, I64, and BF16 SafeTensors entries (BF16 widens to f32). Optional `__metadata__`
   * string mappings are ignored; malformed entries and other dtypes are rejected.
   */
  export function load(path: string, options?: { names?: readonly string[] }): LoadedCheckpoint

  /** Read validated SafeTensors metadata without allocating tensor payloads. Dtypes are storage dtypes. */
  export function inspect(path: string): Record<string, { dtype: "F32" | "F64" | "I64" | "BF16"; shape: number[] }>

  /**
   * @summary Restore checkpoint values into an existing module or state tree.
   * @category Persistence
   * @semantics
   * Applies checkpoint tensor values in place, preserving the identity of the
   * target module or target state tree.
   */
  export function restore<T extends CheckpointState>(target: T, source: string | LoadedCheckpoint): T

  /**
   * @summary Save a checkpoint bundle with a JSON manifest plus tensor payload files.
   * @category Persistence
   * @semantics
   * Writes `<prefix>.json`, `<prefix>.safetensors`, and optional named tensor
   * group files such as `<prefix>.optimizer.safetensors`.
   */
  export function saveBundle(
    prefix: string,
    bundle: {
      state: LoadedCheckpoint
      tensorGroups?: CheckpointTensorGroups
      manifest?: Record<string, unknown>
    },
  ): void

  /**
   * @summary Load a checkpoint bundle from a prefix.
   * @category Persistence
   * @semantics
   * Reads a JSON manifest and the tensor payload files it references.
   */
  export function loadBundle(prefix: string): LoadedCheckpointBundle

  const checkpoint: {
    save: typeof save
    load: typeof load
    inspect: typeof inspect
    restore: typeof restore
    saveBundle: typeof saveBundle
    loadBundle: typeof loadBundle
  }

  export default checkpoint
}
